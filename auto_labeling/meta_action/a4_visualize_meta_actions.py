#!/usr/bin/env python3
"""
Visualize meta-action annotations for videos.

Usage:
    # Visualize a single video
    python3 auto_labeling/meta_action/a4_visualize_meta_actions.py <chunk_id> <video_uuid>

    # Visualize all videos in a chunk
    python3 auto_labeling/meta_action/a4_visualize_meta_actions.py <chunk_id>
"""

import sys
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

# Setup path for imports from same directory
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# Import configuration from shared config module
from meta_action_config import (
    MetaActionConfig,
    LONG_SEMANTIC_COLORS,
    LAT_SEMANTIC_COLORS,
    LONG_SEMANTIC_GROUP,
    LAT_SEMANTIC_GROUP,
)


def load_annotation(chunk_id: str, video_uuid: str, labels_dir: Path) -> dict:
    """Load meta-action annotation JSON."""
    meta_actions_base = labels_dir / "meta_actions"
    chunk_dir = meta_actions_base / f"meta_actions.{chunk_id}"

    if not chunk_dir.exists():
        raise FileNotFoundError(f"Annotation directory not found: {chunk_dir}")

    # Remove .egomotion suffix if present
    video_uuid = video_uuid.replace('.egomotion', '')
    target_file = chunk_dir / f"{video_uuid}.meta_actions.json"

    if not target_file.exists():
        raise ValueError(f"Video {video_uuid} not found in chunk {chunk_id}. Looking for: {target_file}")

    if not target_file.exists():
        raise ValueError(f"Video {video_uuid} not found in chunk {chunk_id}")

    with open(target_file, 'r') as f:
        video_data = json.load(f)
        return video_data


def plot_meta_actions(video_data: dict, output_path: Path = None):
    """
    Create visualization plots for meta-action timeline using smooth_data.

    Visualizes at semantic group level (coarse-grained) rather than fine-grained
    meta-actions, for clearer understanding of driving behavior.

    Args:
        video_data: Meta-action annotation data with smooth_data
        output_path: Path to save the figure
    """
    if 'smooth_data' not in video_data:
        raise ValueError("smooth_data not found in video_data. Please regenerate with latest annotation script.")

    smooth_data = video_data['smooth_data']
    timestamps = smooth_data['timestamp_sec']
    speeds = smooth_data['speed']
    accels = smooth_data['acceleration']
    curvatures = smooth_data['curvature']
    long_actions = smooth_data['long_action']
    lat_actions = smooth_data['lat_action']

    # Convert fine-grained actions to semantic groups for visualization
    long_semantic = [LONG_SEMANTIC_GROUP.get(a, 'cruise') for a in long_actions]
    lat_semantic = [LAT_SEMANTIC_GROUP.get(a, 'straight') for a in lat_actions]

    keyframes = video_data['keyframes']
    kf_timestamps = [kf['timestamp_sec'] for kf in keyframes]

    fig, axes = plt.subplots(4, 1, figsize=(14, 10))
    fig.suptitle(f"Meta-Action Timeline: {video_data['video_uuid']}", fontsize=14, fontweight='bold')

    # Use semantic colors (coarse-grained)
    long_colors = LONG_SEMANTIC_COLORS
    lat_colors = LAT_SEMANTIC_COLORS

    # Config for threshold lines (use same values as annotation script)
    config = MetaActionConfig()

    # Plot 1: Speed
    axes[0].plot(timestamps, speeds, 'b-', linewidth=1.5)
    axes[0].fill_between(timestamps, 0, speeds, alpha=0.3)
    axes[0].set_ylabel('Speed (m/s)', fontsize=10, fontweight='bold')
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].set_title('Speed Profile', fontsize=11)

    # Plot 2: Acceleration with semantic group background
    for i in range(len(timestamps) - 1):
        frame_idx = int(timestamps[i] * 10)  # 10Hz
        if frame_idx < len(long_semantic):
            semantic = long_semantic[frame_idx]
            color = long_colors.get(semantic, '#95A5A6')
            axes[1].axvspan(timestamps[i], timestamps[i+1], alpha=0.4, color=color)

    axes[1].plot(timestamps, accels, 'r-', linewidth=2, zorder=10)
    axes[1].axhline(0, color='black', linestyle='--', alpha=0.7, linewidth=1.5)
    # Use threshold values from config
    axes[1].axhline(config.gentle_accel_threshold, color='green', linestyle=':', alpha=0.5, label='Gentle accel')
    axes[1].axhline(config.gentle_decel_threshold, color='red', linestyle=':', alpha=0.5, label='Gentle decel')
    axes[1].axhline(config.strong_accel_threshold, color='darkgreen', linestyle=':', alpha=0.5, label='Strong accel')
    axes[1].axhline(config.strong_decel_threshold, color='darkred', linestyle=':', alpha=0.5, label='Strong decel')
    axes[1].set_ylabel('Accel (m/s²)', fontsize=10, fontweight='bold')
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].set_title('Longitudinal Acceleration & Semantic States', fontsize=11)

    # Add legend for longitudinal semantic groups
    long_patches = [mpatches.Patch(color=c, label=a, alpha=0.6) for a, c in sorted(long_colors.items())]
    axes[1].legend(handles=long_patches, loc='upper right', fontsize=8, ncol=2, framealpha=0.9)

    # Plot 3: Curvature with semantic group background
    for i in range(len(timestamps) - 1):
        frame_idx = int(timestamps[i] * 10)  # 10Hz
        if frame_idx < len(lat_semantic):
            semantic = lat_semantic[frame_idx]
            color = lat_colors.get(semantic, '#95A5A6')
            axes[2].axvspan(timestamps[i], timestamps[i+1], alpha=0.4, color=color)

    axes[2].plot(timestamps, curvatures, 'g-', linewidth=2, zorder=10)
    axes[2].axhline(0, color='black', linestyle='--', alpha=0.7, linewidth=1.5)
    # Use threshold values from config (curvature thresholds in 1/m)
    axes[2].axhline(config.gentle_steer_threshold_curvature, color='orange', linestyle=':', alpha=0.5, label='Gentle steer')
    axes[2].axhline(-config.gentle_steer_threshold_curvature, color='orange', linestyle=':', alpha=0.5)
    axes[2].axhline(config.sharp_steer_threshold_curvature, color='darkorange', linestyle=':', alpha=0.5, label='Sharp steer')
    axes[2].axhline(-config.sharp_steer_threshold_curvature, color='darkorange', linestyle=':', alpha=0.5)
    axes[2].set_ylabel('Curvature (1/m)', fontsize=10, fontweight='bold')
    axes[2].grid(True, alpha=0.3, linestyle='--')
    axes[2].set_title('Lateral Curvature & Semantic States', fontsize=11)

    # Add legend for lateral semantic groups
    lat_patches = [mpatches.Patch(color=c, label=a, alpha=0.6) for a, c in sorted(lat_colors.items())]
    axes[2].legend(handles=lat_patches, loc='upper right', fontsize=8, ncol=2, framealpha=0.9)

    # Plot 4: Keyframe markers
    y_positions = [1] * len(kf_timestamps)
    axes[3].scatter([i for i in kf_timestamps], y_positions, c='red', s=100, marker='|', linewidths=3, zorder=10)

    # Add keyframe count text
    axes[3].text(0.02, 0.5, f'Total: {len(kf_timestamps)} keyframes',
                 transform=axes[3].transAxes, fontsize=12, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    axes[3].set_ylim(0.5, 1.5)
    axes[3].set_yticks([])
    axes[3].set_xlabel('Time (seconds)', fontsize=10, fontweight='bold')
    axes[3].set_title('Keyframe Selection (semantic state transitions)', fontsize=11)
    axes[3].grid(True, axis='x', alpha=0.3, linestyle='--')

    # Align x-axis across all subplots
    xlim = (timestamps[0], timestamps[-1])
    for ax in axes:
        ax.set_xlim(xlim)

    # Add statistics as text
    stats_text = f"Video: {video_data['num_frames']} frames ({video_data['num_frames']/10:.1f}s)\n"
    stats_text += f"Keyframes: {video_data['num_keyframes']}\n"
    if video_data['num_keyframes'] > 0:
        stats_text += f"Compression: {video_data['num_frames']/video_data['num_keyframes']:.1f}x"
    else:
        stats_text += "Compression: N/A (no keyframes)"

    fig.text(0.98, 0.5, stats_text, fontsize=10, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
             fontfamily='monospace')

    plt.tight_layout(rect=[0, 0, 0.85, 0.96])

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization to {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 auto_labeling/meta_action/a4_visualize_meta_actions.py <chunk_id> [video_uuid]")
        print("\nExample:")
        print("  # Visualize all videos in a chunk")
        print("  python3 auto_labeling/meta_action/a4_visualize_meta_actions.py chunk_0000")
        print("\n  # Visualize a single video")
        print("  python3 auto_labeling/meta_action/a4_visualize_meta_actions.py chunk_0000 86de1c0c-e9cd-44ef-aad2-211c6b8a00da")
        sys.exit(1)

    chunk_id = sys.argv[1]
    video_uuid = sys.argv[2] if len(sys.argv) > 2 else None

    # Load environment
    script_dir = Path(__file__).parent.parent.parent
    env_path = script_dir / ".env"

    env_vars = {}
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()

    data_dir = Path(env_vars.get('PHYSICAL_AI_AV_DATA_DIR',
                                   '/home/xingao/data/PhysicalAI-Autonomous-Vehicles-base-wo-lidar-radar'))
    labels_dir = data_dir / "labels"
    meta_actions_base = labels_dir / "meta_actions"
    chunk_dir = meta_actions_base / f"meta_actions.{chunk_id}"

    # Create visualization output directory
    vis_output_dir = meta_actions_base / f"meta_actions.{chunk_id}.vis"
    vis_output_dir.mkdir(parents=True, exist_ok=True)

    if not chunk_dir.exists():
        print(f"Error: Annotation directory not found: {chunk_dir}")
        sys.exit(1)

    # Get list of videos to process
    if video_uuid:
        # Single video
        video_uuids = [video_uuid]
    else:
        # All videos in chunk
        video_uuids = []
        for f in chunk_dir.glob("*.json"):
            if f.name.endswith(".meta_actions.json"):
                uuid = f.name.replace(".meta_actions.json", "")
                video_uuids.append(uuid)
        video_uuids = sorted(video_uuids)
        print(f"Found {len(video_uuids)} videos in {chunk_id}")

    print(f"\n{'='*80}")
    print(f"Meta-Action Visualization: {chunk_id}")
    print(f"{'='*80}")
    print(f"Output directory: {vis_output_dir}")
    print(f"{'='*80}\n")

    # Process each video
    for uuid in tqdm(video_uuids, desc="Visualizing"):
        try:
            # Load annotation
            video_data = load_annotation(chunk_id, uuid, labels_dir)

            # Generate visualization
            output_path = vis_output_dir / f"{uuid}_meta_actions.png"
            plot_meta_actions(video_data, output_path)

        except Exception as e:
            print(f"\nError processing {uuid}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*80}")
    print(f"✓ Visualization complete!")
    print(f"Output saved to: {vis_output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
