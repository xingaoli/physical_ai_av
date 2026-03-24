#!/usr/bin/env python3
"""
Meta-Action Auto-Labeling for Keyframe Selection.

This script implements rule-based meta-action detectors as described in the CoC paper.
It labels each frame with longitudinal and lateral meta-actions, then identifies
keyframes at meta-action transition points.

Meta Actions (from Table 5):
    Longitudinal: Gentle accelerate, Gentle decelerate, Maintain speed, Reverse,
                  Strong accelerate, Strong decelerate, Stop
    Lateral: Steer left, Steer right, Sharp steer left, Sharp steer right,
             Reverse left, Reverse right, Go straight

Keyframes are identified at moments when meta-action transitions occur.

Usage:
    python3 auto_labeling/meta_action/3_meta_action_annotation.py --chunks chunk_0000

    add visualize first video
    python3 auto_labeling/meta_action/3_meta_action_annotation.py --chunks chunk_0000 --viz
"""

import os
import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
import numpy as np
from scipy.ndimage import median_filter
from scipy.spatial.transform import Rotation as spt_Rotation
from tqdm import tqdm

# Setup path for imports from same directory
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# Import configuration from shared config module
from meta_action_config import (
    MetaActionConfig,
    LONG_SEMANTIC_GROUP,
    LAT_SEMANTIC_GROUP,
)

# Import visualization function from 4_visualize_meta_actions
try:
    from a4_visualize_meta_actions import plot_meta_actions
except ImportError:
    # Fallback if the file is not available
    plot_meta_actions = None


def load_env(env_path: str = ".env") -> dict:
    """Load environment variables from .env file."""
    env_vars = {}
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
    return env_vars


class MetaActionDetector:
    """
    Rule-based meta-action detector for ego vehicle motion.

    Implements detectors from CoC paper Table 5 for labeling atomic
    meta-actions at frame level (10Hz).
    """

    def __init__(self, config: Optional[MetaActionConfig] = None):
        self.config = config or MetaActionConfig()

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess egomotion data for robust detection.

        Key changes:
        1. Transform velocity and acceleration to LOCAL vehicle coordinate system
        2. Keep curvature unchanged (it's coordinate-system independent)

        Local coordinate system:
        - Origin: current vehicle position
        - X-axis: vehicle forward direction
        - Y-axis: vehicle left direction
        - Z-axis: vehicle up direction
        """
        df = df.copy()

        # Convert quaternions to rotation matrices
        quats = np.stack([
            df['qx'].values,
            df['qy'].values,
            df['qz'].values,
            df['qw'].values
        ], axis=1)
        rots = spt_Rotation.from_quat(quats)
        rots_inv = rots.inv()

        # Transform velocity to local frame: local_v = R_inv @ world_v
        world_v = np.stack([
            df['vx'].values,
            df['vy'].values,
            df['vz'].values
        ], axis=1)
        local_v = rots_inv.apply(world_v)
        df['local_vx'] = local_v[:, 0]  # Longitudinal (forward = positive, reverse = negative)
        df['local_vy'] = local_v[:, 1]  # Lateral
        df['local_vz'] = local_v[:, 2]  # Vertical

        # Transform acceleration to local frame: local_a = R_inv @ world_a
        world_a = np.stack([
            df['ax'].values,
            df['ay'].values,
            df['az'].values
        ], axis=1)
        local_a = rots_inv.apply(world_a)
        df['local_ax'] = local_a[:, 0]  # Longitudinal acceleration
        df['local_ay'] = local_a[:, 1]  # Lateral acceleration
        df['local_az'] = local_a[:, 2]  # Vertical acceleration

        # Speed magnitude (always positive, use local velocities)
        df['speed'] = np.sqrt(df['local_vx']**2 + df['local_vy']**2)

        # Longitudinal acceleration (use local_ax)
        df['long_accel'] = df['local_ax']

        # Smooth acceleration to reduce noise
        df['long_accel_smooth'] = median_filter(
            df['long_accel'],
            size=self.config.acceleration_window
        )

        # Calculate yaw (heading) from quaternion
        df['yaw'] = np.arctan2(
            2 * (df['qw'] * df['qz'] + df['qx'] * df['qy']),
            1 - 2 * (df['qy']**2 + df['qz']**2)
        )

        # Calculate yaw rate (rate of heading change) in rad/s
        dt = 0.1
        df['yaw_rate'] = np.gradient(df['yaw'], dt)

        # Smooth yaw rate (kept for reference, but curvature is preferred)
        df['yaw_rate_smooth'] = median_filter(
            df['yaw_rate'],
            size=self.config.curvature_window
        )

        # Curvature is coordinate-system independent (geometric property)
        # Just smooth it, no transformation needed
        df['curvature_smooth'] = median_filter(
            df['curvature'].fillna(0),
            size=self.config.curvature_window
        )

        return df

    def detect_longitudinal(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect longitudinal meta-actions using LOCAL coordinate system values.

        Logic:
        1. |local_vx| < 0.5 → Stop (too slow, might be noise)
        2. local_vx < -0.5 → Reverse (definitely backing up)
        3. local_vx > 0 → Forward motion, check acceleration

        Uses:
        - local_vx: longitudinal velocity (negative = reversing)
        - local_ax: longitudinal acceleration

        Returns:
            pd.Series: Longitudinal meta-action label for each frame
        """
        labels = []

        for _, row in df.iterrows():
            local_vx = row['local_vx']
            accel = row['long_accel_smooth']

            # 1. Check for stop (|local_vx| <= 0.5 filters out noise)
            if abs(local_vx) <= 0.5:
                labels.append('Stop')

            # 2. Check for reverse (local_vx < -0.5 means definitely backing up)
            elif local_vx < -0.5:
                labels.append('Reverse')

            # 3. Forward motion (local_vx > 0.5), check acceleration
            elif accel > self.config.strong_accel_threshold:
                labels.append('Strong accelerate')

            elif accel > self.config.gentle_accel_threshold:
                labels.append('Gentle accelerate')

            elif accel < self.config.strong_decel_threshold:
                labels.append('Strong decelerate')

            elif accel < self.config.gentle_decel_threshold:
                labels.append('Gentle decelerate')

            # Default: maintain speed
            else:
                labels.append('Maintain speed')

        return pd.Series(labels, index=df.index)

    def detect_lateral(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect lateral meta-actions using LOCAL coordinate system.

        Uses:
        - local_vx: to check if reversing (negative = backing up)
        - curvature: to detect turning (coordinate-system independent) [DEFAULT]
        - yaw_rate: alternative (noisy)

        Curvature (1/m):
        - Positive: turning left
        - Negative: turning right
        - Near zero: going straight

        Thresholds (curvature):
        - |κ| < 0.01: Go straight
        - 0.01 ≤ |κ| < 0.05: Gentle steer
        - |κ| ≥ 0.05: Sharp steer

        Returns:
            pd.Series: Lateral meta-action label for each frame
        """
        labels = []

        # Get thresholds based on which metric we're using
        if self.config.use_yaw_rate:
            sharp_thresh = self.config.sharp_steer_threshold_yaw_rate
            gentle_thresh = self.config.gentle_steer_threshold_yaw_rate
            metric_col = 'yaw_rate_smooth'
        else:
            sharp_thresh = self.config.sharp_steer_threshold_curvature
            gentle_thresh = self.config.gentle_steer_threshold_curvature
            metric_col = 'curvature_smooth'

        for _, row in df.iterrows():
            metric_value = row[metric_col]

            # Check if reversing (using local_vx from vehicle frame)
            if row['local_vx'] < -0.5:
                # During reverse, check steering direction
                # Same logic as forward: negative = right, positive = left
                if metric_value < -gentle_thresh:
                    labels.append('Reverse right')   # negative curvature = right turn
                elif metric_value > gentle_thresh:
                    labels.append('Reverse left')    # positive curvature = left turn
                else:
                    labels.append('Reverse straight')

            # Normal forward driving
            elif metric_value < -sharp_thresh:
                labels.append('Sharp steer right')  # negative curvature = right turn

            elif metric_value > sharp_thresh:
                labels.append('Sharp steer left')   # positive curvature = left turn

            elif metric_value < -gentle_thresh:
                labels.append('Steer right')

            elif metric_value > gentle_thresh:
                labels.append('Steer left')

            # Default: go straight
            else:
                labels.append('Go straight')

        return pd.Series(labels, index=df.index)

    def detect_meta_actions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect both longitudinal and lateral meta-actions for all frames.

        Note: Debouncing and short-state filtering are now handled in
        KeyframeSelector to work on semantic labels directly.

        Args:
            df: Egomotion dataframe with columns [timestamp, vx, vy, vz, curvature]

        Returns:
            DataFrame with added columns:
            - long_action: fine-grained longitudinal label
            - lat_action: fine-grained lateral label
        """
        # Preprocess
        df_processed = self.preprocess_data(df)

        # Detect meta-actions (raw labels)
        df_processed['long_action'] = self.detect_longitudinal(df_processed)
        df_processed['lat_action'] = self.detect_lateral(df_processed)

        return df_processed


class KeyframeSelector:
    """
    Select keyframes based on meta-action transitions.

    Following the CoC paper: "we treat the frame at which a meta action
    transition occurs as a decision-making moment"

    Processing pipeline:
    1. Debounce semantic labels to prevent noise-induced state transitions
    2. Filter short states to remove brief but real state changes
    3. Find transition points in semantic labels
    4. Suppress terminal returns (self-terminating state exits)
    5. Merge nearby keyframes
    """

    def __init__(self, config: MetaActionConfig, min_transition_gap: int = 5):
        """
        Args:
            config: MetaActionConfig with debounce/filter parameters
            min_transition_gap: Minimum frames between consecutive keyframes
                              to avoid duplicate detections
        """
        self.config = config
        self.min_gap = min_transition_gap

    def filter_short_states(self, labels: pd.Series, min_duration: int) -> pd.Series:
        """
        过滤掉持续时间过短的状态（包括高频噪声）。

        既能处理1-2帧的高频噪声，又能过滤掉持续时间过短的真实状态变化。
        使用向前传播（forward propagation）：短状态被"吸收"进前一个状态。

        Args:
            labels: 原始标签序列
            min_duration: 状态的最小持续时间，短于此会被合并到前一个状态

        Returns:
            过滤后的标签序列
        """
        arr = labels.values.copy()
        result = arr.copy()

        i = 0
        while i < len(arr):
            j = i
            while j < len(arr) and arr[j] == arr[i]:
                j += 1

            duration = j - i

            if duration < min_duration:
                if i == 0:
                    # 开头的短状态：用后继状态向前填充
                    result[i:j] = arr[j] if j < len(arr) else arr[i]
                else:
                    # 中间的短状态：用前一个状态填充（向前传播）
                    result[i:j] = result[i - 1]

            i = j

        return pd.Series(result, index=labels.index)

    def suppress_terminal_return(
        self,
        keyframes: List[int],
        semantic_labels: pd.Series,
        self_terminating: tuple,
        terminal_return: str,
    ) -> List[int]:
        """
        统一的自终止状态抑制：
        如果某关键帧的转换是 self_terminating_state -> terminal_return，
        则认为是正常结束而非新决策，抑制该关键帧。

        适用于：
          纵向: decelerating/accelerating -> cruise
          横向: turning_left/turning_right -> straight

        核心思想：进入自终止状态是决策点（需要避障、加速超车等），
                 退出该状态只是恢复正常驾驶，不是新决策。

        Args:
            keyframes: 原始关键帧列表（已排序）
            semantic_labels: 语义标签序列
            self_terminating: 自终止状态元组
            terminal_return: 终止返回状态名称

        Returns:
            抑制后的关键帧列表
        """
        if len(keyframes) < 2:
            return keyframes

        result = [keyframes[0]]
        for kf in keyframes[1:]:
            state_before = semantic_labels.iloc[kf - 1] if kf > 0 else None
            state_after = semantic_labels.iloc[kf]

            # 检查是否是自终止状态的正常退出
            if state_before in self_terminating and state_after == terminal_return:
                continue

            result.append(kf)

        return result

    def find_transitions(self, labels: pd.Series) -> List[int]:
        """
        Find indices where label transitions occur.

        Selects the frame 0.5s (5 frames) before each transition as a keyframe.
        Constraints:
        - Keyframe >= 20: ensures 2s (20 frames) of historical context before keyframe
        - Keyframe <= 140: ensures 6s (60 frames) of future trajectory prediction
          (total video is 201 frames, indices 0-200)

        Therefore, transition index i must satisfy: 25 <= i <= 145

        Args:
            labels: Series of meta-action labels

        Returns:
            List of frame indices where transitions occur
        """
        transitions = []

        # i-5 >= 20 => i >= 25 (2s history before keyframe)
        # i-5 <= 140 => i <= 145 (keyframe + 60 <= 200 for 6s prediction)
        for i in range(25, min(146, len(labels))):
            if labels.iloc[i] != labels.iloc[i-1]:
                transitions.append(i)

        return transitions

    def suppress_intra_action_keyframes(
        self,
        keyframes: List[int],
        long_semantic: pd.Series,
        lat_semantic: pd.Series,
    ) -> List[int]:
        """
        根据白名单抑制伴随变化关键帧。

        新的白名单机制：
        - 检查某个维度是否保持不变
        - 如果保持不变，检查另一个维度的变化是否在抑制列表中

        示例：
        - 转弯中加速：横向不变(turn_left) + 纵向变加速 → 抑制 ✓
        - 转弯起始：横向变(straight→turn_left) → 保留 ✓ (不在抑制列表)
        - 转弯中急刹：横向不变(turn_left) + 纵向变减速 → 保留 ✓ (减速不在抑制列表)

        Args:
            keyframes: 原始关键帧列表（已排序）
            long_semantic: 纵向语义标签序列
            lat_semantic: 横向语义标签序列

        Returns:
            抑制后的关键帧列表
        """
        if not keyframes:
            return []

        result = [keyframes[0]]

        for kf in keyframes[1:]:
            prev_kf = result[-1]

            curr_lat = lat_semantic.iloc[kf]
            curr_long = long_semantic.iloc[kf]
            prev_lat = lat_semantic.iloc[prev_kf]
            prev_long = long_semantic.iloc[prev_kf]

            # 横向没变，检查纵向变化是否是伴随动作
            if curr_lat == prev_lat:
                # 额外检查：两个关键帧之间横向是否真的一直没变
                lat_between = lat_semantic.iloc[prev_kf:kf]
                if lat_between.nunique() == 1:  # 中间只有一种状态
                    suppress_set = self.config.suppress_if_lat_unchanged.get(curr_lat, set())
                    if curr_long in suppress_set:
                        # 特殊处理：straight + accelerating 组合
                        # 只有从 stopped 起步的 accelerate 才保留（可能是红灯变绿灯等重要时刻）
                        # 从其他状态（如 cruise）的 accelerate 则抑制（伴随动作）
                        if curr_lat == 'straight' and curr_long == 'accelerating':
                            if prev_long != 'stopped':
                                continue  # 从非 stopped 状态加速，抑制这个关键帧
                        else:
                            continue  # 其他情况按原逻辑抑制这个关键帧

            # 纵向没变，检查横向变化是否是伴随动作
            if curr_long == prev_long:
                long_between = long_semantic.iloc[prev_kf:kf]
                if long_between.nunique() == 1:
                    suppress_set = self.config.suppress_if_long_unchanged.get(curr_long, set())
                    if curr_lat in suppress_set:
                        continue  # 抑制这个关键帧

            result.append(kf)

        return result

    def merge_nearby_keyframes(self, keyframes: List[int]) -> List[int]:
        """
        Merge keyframes that are within min_gap frames of each other.

        When a longitudinal and lateral transition occur within min_gap frames,
        they likely represent the same decision-making moment. This method
        keeps only the first keyframe in each cluster.

        Args:
            keyframes: Sorted list of frame indices

        Returns:
            Filtered list with nearby keyframes merged
        """
        if not keyframes:
            return []

        merged = [keyframes[0]]
        for kf in keyframes[1:]:
            if kf - merged[-1] > self.min_gap:
                merged.append(kf)
            # else: skip this keyframe, it's too close to the previous one

        return merged

    def select_keyframes(
        self,
        df: pd.DataFrame,
        include_first_frame: bool = True,
        include_last_frame: bool = False,
    ) -> Dict[str, List[int]]:
        """
        Select keyframes based on meta-action transitions.

        Processing pipeline (unified for longitudinal and lateral):
        1. Map fine-grained actions to semantic groups
        2. Filter short states to remove noise and insignificant changes
           (8 frames for longitudinal, 5 frames for lateral)
        3. Find transition points in semantic labels
        4. Suppress terminal returns (self-terminating state exits)
        5. Merge longitudinal and lateral keyframes
        6. Suppress intra-action keyframes (whitelist-based accompanying changes)
        7. Merge nearby keyframes

        Args:
            df: DataFrame with 'long_action', 'lat_action' columns
            include_first_frame: Include frame 0 as keyframe
            include_last_frame: Include last frame as keyframe

        Returns:
            Dict with:
                - 'longitudinal': keyframes from longitudinal transitions
                - 'lateral': keyframes from lateral transitions
                - 'combined': union of both with nearby keyframes merged
        """
        # 1. 语义组映射
        long_semantic = df['long_action'].map(LONG_SEMANTIC_GROUP)
        lat_semantic = df['lat_action'].map(LAT_SEMANTIC_GROUP)

        # 2. 短状态过滤：移除持续时间过短的状态（包括高频噪声）
        # 纵向8帧(0.8s)：过滤噪声，保留真实制动
        # 横向5帧(0.5s)：过滤噪声，保留真实变道
        long_semantic = self.filter_short_states(long_semantic, self.config.min_state_duration_long)
        lat_semantic = self.filter_short_states(lat_semantic, self.config.min_state_duration_lat)

        # 3. 找转换点
        long_kfs = self.find_transitions(long_semantic)
        lat_kfs = self.find_transitions(lat_semantic)

        # 可选：添加首尾帧，添加首帧以进行自终止状态抑制和伴随动作抑制
        if include_first_frame:
            long_kfs.insert(0, 0)
            lat_kfs.insert(0, 0)

        if include_last_frame:
            long_kfs.append(len(df) - 1)
            lat_kfs.append(len(df) - 1)

        # 4. 自终止状态抑制（纵向 + 横向统一处理）
        # 进入自终止状态是决策点，退出该状态（回到正常状态）不是新决策
        long_kfs = self.suppress_terminal_return(
            long_kfs, long_semantic,
            self_terminating=self.config.self_terminating_long,
            terminal_return=self.config.terminal_return_long,
        )
        lat_kfs = self.suppress_terminal_return(
            lat_kfs, lat_semantic,
            self_terminating=self.config.self_terminating_lat,
            terminal_return=self.config.terminal_return_lat,
        )

        # 5. 合并纵向和横向关键帧
        combined = sorted(set(long_kfs + lat_kfs))

        # 6. 伴随动作抑制（白名单机制）
        # 对于白名单内的组合（如转弯+加速），检查是否是同一动作的延续
        combined = self.suppress_intra_action_keyframes(combined, long_semantic, lat_semantic)

        # 7. 合并邻近关键帧
        combined = self.merge_nearby_keyframes(combined)

        # 如果添加了首帧来进行自终止状态抑制和伴随动作抑制，则最后要移除首帧
        if long_kfs[0] == 0:
            long_kfs.remove(0)
        if lat_kfs[0] == 0:
            lat_kfs.remove(0)
        if combined[0] == 0:
            combined.remove(0)

        return {
            'longitudinal': sorted(long_kfs),
            'lateral': sorted(lat_kfs),
            'combined': combined,
        }


def process_single_video(
    video_uuid: str,
    egomotion_path: Path,
    detector: MetaActionDetector,
    selector: KeyframeSelector
) -> Optional[Dict]:
    """
    Process a single video's egomotion data.

    Returns:
        Dict with annotation results or None if error
    """
    try:
        # Load egomotion data
        df = pd.read_parquet(egomotion_path)

        # Detect meta-actions
        df_labeled = detector.detect_meta_actions(df)

        # Select keyframes
        keyframes = selector.select_keyframes(df_labeled)

        # Extract keyframe data
        keyframe_data = []
        for idx in keyframes['combined']:
            row = df_labeled.iloc[idx]
            keyframe_data.append({
                'frame_index': int(idx),
                'timestamp_us': int(row['timestamp']),
                'timestamp_sec': round(row['timestamp'] / 1e6, 2),
                'long_action': row['long_action'],
                'lat_action': row['lat_action'],
                'speed': round(row['speed'], 3),
                'acceleration': round(row['long_accel_smooth'], 3),
                'yaw_rate': round(row['yaw_rate_smooth'], 4),
                'curvature': round(row['curvature_smooth'], 6),  # keep for reference
            })

        # Compute action statistics
        action_stats = {
            'longitudinal': df_labeled['long_action'].value_counts().to_dict(),
            'lateral': df_labeled['lat_action'].value_counts().to_dict(),
        }

        # Extract smooth data series for visualization (for consistency)
        smooth_data = {
            'timestamp_sec': [round(t / 1e6, 2) for t in df_labeled['timestamp'].tolist()],
            'speed': [round(s, 3) for s in df_labeled['speed'].tolist()],
            'acceleration': [round(a, 3) for a in df_labeled['long_accel_smooth'].tolist()],
            'yaw_rate': [round(y, 4) for y in df_labeled['yaw_rate_smooth'].tolist()],
            'curvature': [round(c, 6) for c in df_labeled['curvature_smooth'].tolist()],  # add curvature for lateral viz
            'long_action': df_labeled['long_action'].tolist(),
            'lat_action': df_labeled['lat_action'].tolist(),
        }

        return {
            'video_uuid': video_uuid.replace('.egomotion', ''),
            'num_frames': len(df),
            'num_keyframes': len(keyframes['combined']),
            'keyframes': keyframe_data,
            'action_statistics': action_stats,
            'keyframe_indices': keyframes,
            'smooth_data': smooth_data,  # Full smooth data series for visualization
        }

    except Exception as e:
        print(f"  Error processing {video_uuid}: {e}")
        return None


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Meta-action auto-labeling for keyframe selection',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--chunks',
        nargs='+',
        default=None,
        help='Specific chunk IDs to process (e.g., chunk_0000)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for annotations (default: data_dir/labels/meta_actions)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to JSON config file with custom thresholds'
    )
    parser.add_argument(
        '--viz',
        action='store_true',
        help='Generate visualization plots for first video in each chunk'
    )
    args = parser.parse_args()

    # Load environment
    script_dir = Path(__file__).parent.parent.parent
    env_path = script_dir / ".env"
    env_vars = load_env(env_path)

    data_dir = Path(env_vars.get('PHYSICAL_AI_AV_DATA_DIR', '/home/xingao/data/PhysicalAI-Autonomous-Vehicles-base-wo-lidar-radar'))
    egomotion_dir = data_dir / "labels" / "egomotion_corrected"
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / "labels" / "meta_actions"

    print(f"Data directory: {data_dir}")
    print(f"Egomotion directory: {egomotion_dir}")
    print(f"Output directory: {output_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config if provided
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = MetaActionConfig(**config_dict)
        print(f"Loaded custom config from {args.config}")
    else:
        config = MetaActionConfig()

    # Initialize detector and selector
    detector = MetaActionDetector(config)
    selector = KeyframeSelector(config, min_transition_gap=5)

    # Get chunks to process
    if args.chunks:
        chunk_files = [egomotion_dir / f"egomotion.{c}.zip" for c in args.chunks]
    else:
        chunk_files = sorted(egomotion_dir.glob("egomotion.chunk_*.zip"))

    if not chunk_files:
        print(f"No egomotion chunks found in {egomotion_dir}")
        return

    print(f"\nWill process {len(chunk_files)} chunks")
    print("="*80)

    # Process each chunk
    import zipfile

    all_results = []

    for chunk_zip in chunk_files:
        chunk_name = chunk_zip.stem.replace('egomotion.', '')
        print(f"\nProcessing {chunk_name}...")

        # Extract to temp
        temp_dir = output_dir / f".temp_{chunk_name}"
        temp_dir.mkdir(exist_ok=True)

        with zipfile.ZipFile(chunk_zip, 'r') as zf:
            zf.extractall(temp_dir)

        # Process each parquet file
        parquet_files = list(temp_dir.glob("*.egomotion.parquet"))
        # parquet_files = [temp_dir / "01d3588e-bca7-4a18-8e74-c6cfe9e996db.egomotion.parquet"]
        chunk_results = []

        for parquet_path in tqdm(parquet_files, desc=f"  {chunk_name}", leave=False):
            video_uuid = parquet_path.stem # 86de1c0c
            result = process_single_video(video_uuid, parquet_path, detector, selector)

            if result:
                chunk_results.append(result)

                # Optional: visualize first video
                if args.viz and len(chunk_results) == 1 and plot_meta_actions is not None:
                    try:
                        output_path = output_dir / f"{video_uuid}_meta_actions.png"
                        plot_meta_actions(result, output_path)
                    except Exception as e:
                        print(f"    Warning: Visualization failed: {e}")

        # Save chunk results - each video as separate JSON file in a directory
        if chunk_results:
            # Create directory for this chunk
            chunk_dir = output_dir / f"meta_actions.{chunk_name}"
            chunk_dir.mkdir(parents=True, exist_ok=True)

            # Save each video as individual JSON file
            for result in chunk_results:
                video_uuid = result['video_uuid']
                # Remove .egomotion suffix if present
                video_uuid = video_uuid.replace('.egomotion', '')
                json_path = chunk_dir / f"{video_uuid}.meta_actions.json"
                with open(json_path, 'w') as f:
                    json.dump(result, f, indent=2)

            print(f"  ✓ Saved {len(chunk_results)} annotations to {chunk_dir.name}/")

            all_results.extend(chunk_results)

        # Cleanup temp
        for p in parquet_files:
            p.unlink()
        temp_dir.rmdir()

    # Generate summary statistics
    if all_results:
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)

        total_frames = sum(r['num_frames'] for r in all_results)
        total_keyframes = sum(r['num_keyframes'] for r in all_results)
        avg_keyframes_per_video = total_keyframes / len(all_results)

        print(f"Total videos processed: {len(all_results)}")
        print(f"Total frames: {total_frames}")
        print(f"Total keyframes: {total_keyframes}")
        print(f"Average keyframes per video: {avg_keyframes_per_video:.2f}")
        print(f"Compression ratio: {total_frames / total_keyframes:.2f}x")

        # Aggregate action statistics
        all_long_actions = {}
        all_lat_actions = {}

        for result in all_results:
            for action, count in result['action_statistics']['longitudinal'].items():
                all_long_actions[action] = all_long_actions.get(action, 0) + count
            for action, count in result['action_statistics']['lateral'].items():
                all_lat_actions[action] = all_lat_actions.get(action, 0) + count

        print("\nLongitudinal action distribution:")
        for action, count in sorted(all_long_actions.items(), key=lambda x: -x[1]):
            pct = 100 * count / sum(all_long_actions.values())
            print(f"  {action}: {count} ({pct:.1f}%)")

        print("\nLateral action distribution:")
        for action, count in sorted(all_lat_actions.items(), key=lambda x: -x[1]):
            pct = 100 * count / sum(all_lat_actions.values())
            print(f"  {action}: {count} ({pct:.1f}%)")

    print("\n" + "="*80)
    print("✓ Meta-action annotation complete!")
    print(f"Output saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
