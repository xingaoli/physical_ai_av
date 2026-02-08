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
    python3 tools/3_meta_action_annotation.py --chunks chunk_0000
"""

import os
import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict

import pandas as pd
import numpy as np
from scipy.ndimage import median_filter
from tqdm import tqdm

# Import visualization function from 4_visualize_meta_actions
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
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


@dataclass
class MetaActionConfig:
    """Configuration thresholds for meta-action detection."""

    # Longitudinal thresholds (m/s²)
    strong_accel_threshold: float = 2.0
    gentle_accel_threshold: float = 0.5
    gentle_decel_threshold: float = -0.5
    strong_decel_threshold: float = -2.0
    stop_speed_threshold: float = 0.5  # m/s

    # Lateral thresholds (curvature-based, 1/m)
    sharp_steer_threshold: float = 0.01
    gentle_steer_threshold: float = 0.002

    # Smoothing parameters
    acceleration_window: int = 5  # frames for median filter
    curvature_window: int = 5


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

        Applies median filtering to reduce noise while preserving edges.
        """
        df = df.copy()

        # Calculate forward velocity magnitude (longitudinal speed)
        df['speed'] = np.sqrt(df['vx']**2 + df['vy']**2)

        # Calculate longitudinal acceleration (forward direction)
        df['long_accel'] = np.gradient(df['speed'], df['timestamp'] / 1e6)

        # Smooth acceleration to reduce noise
        df['long_accel_smooth'] = median_filter(
            df['long_accel'],
            size=self.config.acceleration_window
        )

        # Smooth curvature to reduce noise
        df['curvature_smooth'] = median_filter(
            df['curvature'].fillna(0),
            size=self.config.curvature_window
        )

        return df

    def detect_longitudinal(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect longitudinal meta-actions.

        Returns:
            pd.Series: Longitudinal meta-action label for each frame
        """
        labels = []

        for _, row in df.iterrows():
            speed = row['speed']
            accel = row['long_accel_smooth']

            # Check for reverse (negative velocity in x-direction indicates backing up)
            if row['vx'] < -0.5:
                labels.append('Reverse')

            # Check for stop
            elif speed < self.config.stop_speed_threshold:
                labels.append('Stop')

            # Check acceleration states
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
        Detect lateral meta-actions.

        Returns:
            pd.Series: Lateral meta-action label for each frame
        """
        labels = []

        for _, row in df.iterrows():
            curvature = row['curvature_smooth']

            # Check if reversing
            if row['vx'] < -0.5:
                # During reverse, check steering direction
                if curvature < -self.config.gentle_steer_threshold:
                    labels.append('Reverse left')
                elif curvature > self.config.gentle_steer_threshold:
                    labels.append('Reverse right')
                else:
                    labels.append('Reverse')

            # Normal forward driving
            elif curvature < -self.config.sharp_steer_threshold:
                labels.append('Sharp steer left')

            elif curvature > self.config.sharp_steer_threshold:
                labels.append('Sharp steer right')

            elif curvature < -self.config.gentle_steer_threshold:
                labels.append('Steer left')

            elif curvature > self.config.gentle_steer_threshold:
                labels.append('Steer right')

            # Default: go straight
            else:
                labels.append('Go straight')

        return pd.Series(labels, index=df.index)

    def detect_meta_actions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect both longitudinal and lateral meta-actions for all frames.

        Args:
            df: Egomotion dataframe with columns [timestamp, vx, vy, vz, curvature]

        Returns:
            DataFrame with added columns: [long_action, lat_action, speed, long_accel, curvature_smooth]
        """
        # Preprocess
        df_processed = self.preprocess_data(df)

        # Detect meta-actions
        df_processed['long_action'] = self.detect_longitudinal(df_processed)
        df_processed['lat_action'] = self.detect_lateral(df_processed)

        return df_processed


class KeyframeSelector:
    """
    Select keyframes based on meta-action transitions.

    Following the CoC paper: "we treat the frame at which a meta action
    transition occurs as a decision-making moment"
    """

    def __init__(self, min_transition_gap: int = 5):
        """
        Args:
            min_transition_gap: Minimum frames between consecutive keyframes
                              to avoid duplicate detections
        """
        self.min_gap = min_transition_gap

    def find_transitions(self, labels: pd.Series) -> List[int]:
        """
        Find indices where label transitions occur.

        Args:
            labels: Series of meta-action labels

        Returns:
            List of frame indices where transitions occur
        """
        transitions = []

        for i in range(1, len(labels)):
            if labels.iloc[i] != labels.iloc[i-1]:
                transitions.append(i)

        return transitions

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
        include_last_frame: bool = True
    ) -> Dict[str, List[int]]:
        """
        Select keyframes based on meta-action transitions.

        Args:
            df: DataFrame with 'long_action' and 'lat_action' columns
            include_first_frame: Include frame 0 as keyframe
            include_last_frame: Include last frame as keyframe

        Returns:
            Dict with:
                - 'longitudinal': keyframes from longitudinal transitions
                - 'lateral': keyframes from lateral transitions
                - 'combined': union of both with nearby keyframes merged
        """
        # Find transitions for each dimension
        long_transitions = self.find_transitions(df['long_action'])
        lat_transitions = self.find_transitions(df['lat_action'])

        # Optionally add first/last frames
        longitudinal = long_transitions.copy()
        lateral = lat_transitions.copy()

        if include_first_frame:
            longitudinal.insert(0, 0)
            lateral.insert(0, 0)

        if include_last_frame:
            longitudinal.append(len(df) - 1)
            lateral.append(len(df) - 1)

        # Combine and deduplicate
        combined = sorted(set(longitudinal + lateral))

        # # Merge nearby keyframes (longitudinal + lateral transitions within min_gap)
        # combined = self.merge_nearby_keyframes(combined)

        return {
            'longitudinal': sorted(longitudinal),
            'lateral': sorted(lateral),
            'combined': combined
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
                'curvature': round(row['curvature_smooth'], 6),
            })

        # Compute action statistics
        action_stats = {
            'longitudinal': df_labeled['long_action'].value_counts().to_dict(),
            'lateral': df_labeled['lat_action'].value_counts().to_dict(),
        }

        return {
            'video_uuid': video_uuid,
            'num_frames': len(df),
            'num_keyframes': len(keyframes['combined']),
            'keyframes': keyframe_data,
            'action_statistics': action_stats,
            'keyframe_indices': keyframes
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
    script_dir = Path(__file__).parent.parent
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
    selector = KeyframeSelector(min_transition_gap=5)

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
            video_uuid = parquet_path.stem
            result = process_single_video(video_uuid, parquet_path, detector, selector)

            if result:
                chunk_results.append(result)

                # Optional: visualize first video
                if args.viz and len(chunk_results) == 1 and plot_meta_actions is not None:
                    try:
                        # Load raw egomotion data for full-resolution visualization
                        df_raw = pd.read_parquet(parquet_path)
                        output_path = output_dir / f"{video_uuid}_meta_actions.png"
                        plot_meta_actions(result, df_raw, output_path)
                    except Exception as e:
                        print(f"    Warning: Visualization failed: {e}")

        # Save chunk results
        if chunk_results:
            chunk_output = output_dir / f"meta_actions.{chunk_name}.json"
            with open(chunk_output, 'w') as f:
                json.dump(chunk_results, f, indent=2)
            print(f"  ✓ Saved {len(chunk_results)} annotations to {chunk_output.name}")

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
