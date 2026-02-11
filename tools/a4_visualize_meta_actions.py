#!/usr/bin/env python3
"""
Visualize meta-action annotations for a single video.

Usage:
    python3 tools/visualize_meta_actions.py <chunk_id> <video_uuid>
"""

import sys
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def load_annotation(chunk_id: str, video_uuid: str, data_dir: Path) -> dict:
    """Load meta-action annotation JSON."""
    chunk_dir = data_dir / f"meta_actions.{chunk_id}"

    if not chunk_dir.exists():
        raise FileNotFoundError(f"Annotation directory not found: {chunk_dir}")

    # Remove .egomotion suffix if present
    video_uuid = video_uuid.replace('.egomotion', '')
    target_file = chunk_dir / f"{video_uuid}.meta_actions.json"

    if not target_file.exists():
        raise ValueError(f"Video {video_uuid} not found in chunk {chunk_id}")

    with open(target_file, 'r') as f:
        video_data = json.load(f)
        return video_data


def plot_meta_actions(video_data: dict, output_path: Path = None):
    """
    Create visualization plots for meta-action timeline using smooth_data.

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
    yaw_rates = smooth_data['yaw_rate']
    long_actions = smooth_data['long_action']
    lat_actions = smooth_data['lat_action']

    keyframes = video_data['keyframes']
    kf_timestamps = [kf['timestamp_sec'] for kf in keyframes]

    fig, axes = plt.subplots(4, 1, figsize=(14, 10))
    fig.suptitle(f"Meta-Action Timeline: {video_data['video_uuid']}", fontsize=14, fontweight='bold')

    # Color maps
    long_colors = {
        'Stop': '#FF6B6B',
        'Maintain speed': '#95A5A6',
        'Gentle accelerate': '#90EE90',
        'Strong accelerate': '#2ECC71',
        'Gentle decelerate': '#FFB6C1',
        'Strong decelerate': '#E74C3C',
        'Reverse': '#9B59B6',
    }

    lat_colors = {
        'Go straight': '#95A5A6',
        'Steer left': '#ADD8E6',
        'Steer right': '#FFFACD',
        'Sharp steer left': '#3498DB',
        'Sharp steer right': '#F1C40F',
        'Reverse left': '#9B59B6',
        'Reverse right': '#E67E22',
        'Reverse': '#8E44AD',
    }

    # Plot 1: Speed
    axes[0].plot(timestamps, speeds, 'b-', linewidth=1.5)
    axes[0].fill_between(timestamps, 0, speeds, alpha=0.3)
    axes[0].set_ylabel('Speed (m/s)', fontsize=10, fontweight='bold')
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].set_title('Speed Profile', fontsize=11)

    # Plot 2: Acceleration with keyframe-based longitudinal action background
    kf_long_actions = [kf['long_action'] for kf in keyframes]
    for i in range(len(kf_timestamps) - 1):
        action = kf_long_actions[i]
        color = long_colors.get(action, '#95A5A6')
        axes[1].axvspan(kf_timestamps[i], kf_timestamps[i+1], alpha=0.4, color=color)

    axes[1].plot(timestamps, accels, 'r-', linewidth=2, zorder=10)
    axes[1].axhline(0, color='black', linestyle='--', alpha=0.7, linewidth=1.5)
    axes[1].axhline(0.5, color='green', linestyle=':', alpha=0.5, label='Gentle accel')
    axes[1].axhline(-0.5, color='red', linestyle=':', alpha=0.5, label='Gentle decel')
    axes[1].axhline(2.0, color='darkgreen', linestyle=':', alpha=0.5, label='Strong accel')
    axes[1].axhline(-2.0, color='darkred', linestyle=':', alpha=0.5, label='Strong decel')
    axes[1].set_ylabel('Accel (m/s²)', fontsize=10, fontweight='bold')
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].set_title('Longitudinal Acceleration & Meta-Actions', fontsize=11)

    # Add legend for longitudinal
    long_patches = [mpatches.Patch(color=c, label=a, alpha=0.6) for a, c in sorted(long_colors.items())]
    axes[1].legend(handles=long_patches, loc='upper right', fontsize=7, ncol=4, framealpha=0.9)

    # Plot 3: Yaw Rate with keyframe-based lateral action background
    kf_lat_actions = [kf['lat_action'] for kf in keyframes]
    for i in range(len(kf_timestamps) - 1):
        action = kf_lat_actions[i]
        color = lat_colors.get(action, '#95A5A6')
        axes[2].axvspan(kf_timestamps[i], kf_timestamps[i+1], alpha=0.4, color=color)

    axes[2].plot(timestamps, yaw_rates, 'g-', linewidth=2, zorder=10)
    axes[2].axhline(0, color='black', linestyle='--', alpha=0.7, linewidth=1.5)
    axes[2].axhline(0.02, color='orange', linestyle=':', alpha=0.5)
    axes[2].axhline(-0.02, color='orange', linestyle=':', alpha=0.5)
    axes[2].axhline(0.08, color='darkorange', linestyle=':', alpha=0.5)
    axes[2].axhline(-0.08, color='darkorange', linestyle=':', alpha=0.5)
    axes[2].set_ylabel('Yaw Rate (rad/s)', fontsize=10, fontweight='bold')
    axes[2].grid(True, alpha=0.3, linestyle='--')
    axes[2].set_title('Lateral Yaw Rate & Meta-Actions', fontsize=11)

    # Add legend for lateral
    lat_patches = [mpatches.Patch(color=c, label=a, alpha=0.6) for a, c in sorted(lat_colors.items())]
    axes[2].legend(handles=lat_patches, loc='upper right', fontsize=7, ncol=4, framealpha=0.9)

    # Plot 4: Keyframe markers
    y_positions = [1] * len(kf_timestamps)
    axes[3].scatter(kf_timestamps, y_positions, c='red', s=100, marker='|', linewidths=3, zorder=10)

    # Add keyframe count text
    axes[3].text(0.02, 0.5, f'Total: {len(kf_timestamps)} keyframes',
                 transform=axes[3].transAxes, fontsize=12, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    axes[3].set_ylim(0.5, 1.5)
    axes[3].set_yticks([])
    axes[3].set_xlabel('Time (seconds)', fontsize=10, fontweight='bold')
    axes[3].set_title('Keyframe Selection (meta-action transitions)', fontsize=11)
    axes[3].grid(True, axis='x', alpha=0.3, linestyle='--')

    # Align x-axis across all subplots
    xlim = (timestamps[0], timestamps[-1])
    for ax in axes:
        ax.set_xlim(xlim)

    # Add statistics as text
    stats_text = f"Video: {video_data['num_frames']} frames ({video_data['num_frames']/10:.1f}s)\n"
    stats_text += f"Keyframes: {video_data['num_keyframes']}\n"
    stats_text += f"Compression: {video_data['num_frames']/video_data['num_keyframes']:.1f}x"

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
    if len(sys.argv) < 3:
        print("Usage: python3 tools/visualize_meta_actions.py <chunk_id> <video_uuid> [output_path]")
        print("\nExample:")
        print("  python3 tools/visualize_meta_actions.py chunk_0000 86de1c0c-e9cd-44ef-aad2-211c6b8a00da")
        print("\nTo find video_uuids, check the meta_actions.{chunk_id}/ directory")
        sys.exit(1)

    chunk_id = sys.argv[1]
    video_uuid = sys.argv[2]
    output_path = Path(sys.argv[3]) if len(sys.argv) > 3 else None

    # Load environment
    import os

    script_dir = Path(__file__).parent.parent
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
    output_dir = data_dir / "labels" / "meta_actions"

    try:
        # Load annotation
        video_data = load_annotation(chunk_id, video_uuid, output_dir)

        print(f"\n{'='*80}")
        print(f"Meta-Action Visualization")
        print(f"{'='*80}")
        print(f"Video UUID: {video_data['video_uuid']}")
        print(f"Total frames: {video_data['num_frames']}")
        print(f"Keyframes: {video_data['num_keyframes']}")
        print(f"Compression ratio: {video_data['num_frames']/video_data['num_keyframes']:.2f}x")
        print(f"\nLongitudinal actions: {video_data['action_statistics']['longitudinal']}")
        print(f"Lateral actions: {video_data['action_statistics']['lateral']}")
        print(f"{'='*80}\n")

        # Generate visualization
        if not output_path:
            output_path = output_dir / f"{video_uuid}_meta_actions.png"

        plot_meta_actions(video_data, output_path)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
