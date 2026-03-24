#!/usr/bin/env python3
"""
Visualize egomotion data to understand coordinate systems.

This script helps visualize:
1. World coordinate trajectory (x, y path)
2. Velocities (vx, vy) vs local velocities (local_vx, local_vy)
3. Accelerations (ax, ay) vs local accelerations
4. Curvature changes

Usage:
    python3 auto_labeling/key_frame/visualize_egomotion.py
"""

import argparse
import zipfile
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.spatial.transform import Rotation as spt_Rotation


def load_env(env_path: Path) -> dict:
    """Load environment variables from .env file."""
    env_vars = {}
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
    return env_vars


def load_egomotion(video_uuid: str, chunk_id: str, egomotion_dir: Path) -> Optional[pd.DataFrame]:
    """Load egomotion data for a video."""
    zip_path = egomotion_dir / f"egomotion.{chunk_id}.zip"
    if not zip_path.exists():
        return None

    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            egomotion_file = f"{video_uuid}.egomotion.parquet"
            if egomotion_file not in zf.namelist():
                return None

            with zf.open(egomotion_file) as f:
                df = pd.read_parquet(f)
                return df
    except Exception as e:
        print(f"Warning: Failed to load egomotion for {video_uuid}: {e}")
        return None


def compute_local_frame_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform world frame data to local vehicle frame.

    For each frame i:
    - Origin: position at frame i
    - X-axis: vehicle forward direction at frame i
    - Y-axis: vehicle left direction at frame i
    - Z-axis: vehicle up direction at frame i

    This tells us: at each moment, what is the velocity/acceleration in the
    vehicle's own coordinate system?
    """
    result = df.copy()

    # Arrays to store local frame data
    local_vx_list = []
    local_vy_list = []
    local_ax_list = []
    local_ay_list = []

    n_frames = len(df)

    for i in range(n_frames):
        # Get current frame pose
        quat = np.array([df.iloc[i]['qx'], df.iloc[i]['qy'],
                        df.iloc[i]['qz'], df.iloc[i]['qw']])
        rot = spt_Rotation.from_quat(quat)
        rot_inv = rot.inv()

        # Transform velocity: local_v = R_inv @ world_v
        world_v = np.array([df.iloc[i]['vx'], df.iloc[i]['vy'], df.iloc[i]['vz']])
        local_v = rot_inv.apply(world_v)
        local_vx_list.append(local_v[0])
        local_vy_list.append(local_v[1])

        # Transform acceleration: local_a = R_inv @ world_a
        world_a = np.array([df.iloc[i]['ax'], df.iloc[i]['ay'], df.iloc[i]['az']])
        local_a = rot_inv.apply(world_a)
        local_ax_list.append(local_a[0])
        local_ay_list.append(local_a[1])

    result['local_vx'] = local_vx_list
    result['local_vy'] = local_vy_list
    result['local_ax'] = local_ax_list
    result['local_ay'] = local_ay_list

    # Speed magnitude
    result['speed_world'] = np.sqrt(df['vx']**2 + df['vy']**2)
    result['speed_local'] = np.sqrt(result['local_vx']**2 + result['local_vy']**2)

    # Heading (yaw) from quaternion
    quats = np.stack([df['qx'].values, df['qy'].values,
                     df['qz'].values, df['qw'].values], axis=1)
    rots = spt_Rotation.from_quat(quats)
    mats = rots.as_matrix()
    # yaw = atan2(R[1,0], R[0,0])
    result['yaw'] = np.arctan2(mats[:, 1, 0], mats[:, 0, 0])

    return result


def analyze_trajectory_types(df: pd.DataFrame) -> Dict:
    """Analyze trajectory characteristics to determine type."""
    data = compute_local_frame_data(df)

    # Check for reverse (negative local_vx)
    reverse_frames = (data['local_vx'] < -0.5).sum()
    reverse_ratio = reverse_frames / len(data)

    # Check for turning (yaw change)
    yaw_change = np.abs(data['yaw'].iloc[-1] - data['yaw'].iloc[0])
    if yaw_change > np.pi:
        yaw_change = 2*np.pi - yaw_change

    # Check lateral movement
    lateral_disp = np.abs(data['y'].iloc[-1] - data['y'].iloc[0])

    return {
        'reverse_ratio': reverse_ratio,
        'yaw_change_deg': np.degrees(yaw_change),
        'lateral_disp': lateral_disp,
        'type': 'mixed'
    }


def visualize_egomotion(video_uuid: str, chunk_id: str, df: pd.DataFrame, output_dir: Path):
    """Create comprehensive visualization of egomotion data."""

    data = compute_local_frame_data(df)

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f'Egomotion Analysis: {video_uuid[:8]}... ({chunk_id})',
                 fontsize=14, fontweight='bold')

    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Time axis
    t = (data['timestamp'] - data['timestamp'].iloc[0]) / 1e6  # Convert to seconds

    # 1. Trajectory (Top-Down View)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(data['x'], data['y'], 'b-', linewidth=2, label='Path')
    ax1.plot(data['x'].iloc[0], data['y'].iloc[0], 'go', markersize=10, label='Start')
    ax1.plot(data['x'].iloc[-1], data['y'].iloc[-1], 'ro', markersize=10, label='End')

    # Draw vehicle direction at start, middle, end
    for i, label in [(0, 'Start'), [len(data)//2, 'Mid'], [-1, 'End']]:
        quat = np.array([data.iloc[i]['qx'], data.iloc[i]['qy'],
                        data.iloc[i]['qz'], data.iloc[i]['qw']])
        rot = spt_Rotation.from_quat(quat)
        mat = rot.as_matrix()
        forward = mat[:2, 0] * 5  # Scale for visibility

        ax1.arrow(data.iloc[i]['x'], data.iloc[i]['y'],
                 forward[0], forward[1],
                 head_width=2, head_length=1, fc='r', ec='r')
        ax1.text(data.iloc[i]['x'], data.iloc[i]['y'] + 3, label,
                ha='center', fontsize=8)

    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Trajectory (Top-Down)')
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. Heading (Yaw) over time
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, np.degrees(data['yaw']), 'g-', linewidth=2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Yaw (degrees)')
    ax2.set_title('Vehicle Heading')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color='k', linestyle='--', alpha=0.3)

    # 3. Speed comparison
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(t, data['speed_world'], 'b-', label='World frame speed', alpha=0.7)
    ax3.plot(t, data['speed_local'], 'r--', label='Local frame speed', alpha=0.7)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Speed (m/s)')
    ax3.set_title('Speed Magnitude')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. World frame velocity
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(t, data['vx'], 'r-', label='vx (world)', linewidth=1.5)
    ax4.plot(t, data['vy'], 'b-', label='vy (world)', linewidth=1.5)
    ax4.axhline(-0.5, color='r', linestyle='--', alpha=0.5, label='Reverse threshold')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Velocity (m/s)')
    ax4.set_title('World Frame Velocity')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Local frame velocity
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(t, data['local_vx'], 'r-', label='local_vx (forward)', linewidth=2)
    ax5.plot(t, data['local_vy'], 'b-', label='local_vy (lateral)', linewidth=1.5, alpha=0.7)
    ax5.axhline(-0.5, color='r', linestyle='--', alpha=0.5, label='Reverse threshold')
    ax5.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Velocity (m/s)')
    ax5.set_title('Local Frame Velocity (Vehicle Perspective)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Highlight reverse regions
    reverse_mask = data['local_vx'] < -0.5
    if reverse_mask.any():
        ax5.fill_between(t, -10, 10, where=reverse_mask,
                        color='red', alpha=0.2, label='Reversing')

    # 6. World frame acceleration
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(t, data['ax'], 'r-', label='ax (world)', linewidth=1.5)
    ax6.plot(t, data['ay'], 'b-', label='ay (world)', linewidth=1.5, alpha=0.7)
    ax6.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Acceleration (m/s²)')
    ax6.set_title('World Frame Acceleration')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # 7. Local frame acceleration
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.plot(t, data['local_ax'], 'r-', label='local_ax (longitudinal)', linewidth=2)
    ax7.plot(t, data['local_ay'], 'b-', label='local_ay (lateral)', linewidth=1.5, alpha=0.7)
    ax7.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax7.axhline(1.0, color='g', linestyle='--', alpha=0.5, label='Accel threshold (+)')
    ax7.axhline(-1.0, color='orange', linestyle='--', alpha=0.5, label='Decel threshold (-)')
    ax7.set_xlabel('Time (s)')
    ax7.set_ylabel('Acceleration (m/s²)')
    ax7.set_title('Local Frame Acceleration (Vehicle Perspective)')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # 8. Curvature
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.plot(t, data['curvature'], 'purple', linewidth=2)
    ax8.set_xlabel('Time (s)')
    ax8.set_ylabel('Curvature (1/m)')
    ax8.set_title('Path Curvature')
    ax8.grid(True, alpha=0.3)
    ax8.axhline(0, color='k', linestyle='-', alpha=0.3)

    # 9. Analysis summary
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')

    # Compute statistics
    stats = analyze_trajectory_types(data)

    summary_text = f"""
    TRAJECTORY ANALYSIS

    Duration: {t.iloc[-1]:.1f} s
    Distance: {np.sqrt((data['x'].iloc[-1]-data['x'].iloc[0])**2 +
                      (data['y'].iloc[-1]-data['y'].iloc[0])**2):.1f} m

    REVERSE ANALYSIS
    World vx < -0.5: {(data['vx'] < -0.5).sum()} frames
    Local vx < -0.5: {(data['local_vx'] < -0.5).sum()} frames
    Reverse ratio: {stats['reverse_ratio']*100:.1f}%

    TURNING ANALYSIS
    Yaw change: {stats['yaw_change_deg']:.1f}°
    Lateral disp: {stats['lateral_disp']:.1f} m

    SPEED STATISTICS
    Max speed: {data['speed_local'].max():.1f} m/s
    Mean speed: {data['speed_local'].mean():.1f} m/s
    Min speed: {data['speed_local'].min():.1f} m/s
    """

    ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # Save figure
    output_file = output_dir / f"{video_uuid}_egomotion_analysis.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file.name}")

    return stats


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Visualize egomotion data to understand coordinate systems',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--videos',
        nargs='+',
        default=None,
        help='Specific video UUIDs to analyze (default: sample random videos)'
    )
    parser.add_argument(
        '--chunk',
        type=str,
        default='chunk_0000',
        help='Chunk ID to analyze (default: chunk_0000)'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=5,
        help='Number of random videos to sample (default: 5)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for visualizations (default: data_dir/labels/trajectories/viz)'
    )
    args = parser.parse_args()

    # Load environment
    script_dir = Path(__file__).parent.parent.parent
    env_path = script_dir / ".env"
    env_vars = load_env(env_path)

    data_dir = Path(env_vars.get('PHYSICAL_AI_AV_DATA_DIR',
                                   '/home/xingao/data/PhysicalAI-Autonomous-Vehicles-base'))

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = data_dir / "labels" / "trajectories" / "egomotion_viz"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("Egomotion Data Visualization")
    print("="*80)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Chunk: {args.chunk}")
    print("="*80)

    egomotion_dir = data_dir / "labels" / "egomotion_corrected"
    meta_actions_dir = data_dir / "labels" / "meta_actions" / f"meta_actions.{args.chunk}"

    # Get video list
    if args.videos:
        video_uuids = args.videos
    else:
        # Sample from meta-actions
        import json
        meta_files = list(meta_actions_dir.glob("*.meta_actions.json"))
        video_uuids = [f.name.replace('.meta_actions.json', '') for f in meta_files]

        import random
        random.seed(42)
        video_uuids = random.sample(video_uuids, min(args.num_samples, len(video_uuids)))

    print(f"\nAnalyzing {len(video_uuids)} videos...\n")

    all_stats = []

    for video_uuid in video_uuids:
        print(f"[{video_uuid[:8]}]")

        # Load egomotion
        df = load_egomotion(video_uuid, args.chunk, egomotion_dir)
        if df is None:
            print(f"  ✗ Failed to load egomotion")
            continue

        print(f"  Loaded {len(df)} frames")

        # Visualize
        try:
            stats = visualize_egomotion(video_uuid, args.chunk, df, output_dir)
            stats['video_uuid'] = video_uuid
            all_stats.append(stats)
        except Exception as e:
            print(f"  ✗ Visualization failed: {e}")

    # Summary statistics
    if all_stats:
        print(f"\n{'='*80}")
        print("Summary Statistics")
        print(f"{'='*80}")

        df_stats = pd.DataFrame(all_stats)

        print(f"\nReverse analysis (using local frame):")
        high_reverse = df_stats[df_stats['reverse_ratio'] > 0.1]
        print(f"  High reverse content (>10%): {len(high_reverse)}/{len(df_stats)}")

        print(f"\nTurning analysis:")
        high_turn = df_stats[df_stats['yaw_change_deg'] > 30]
        print(f"  Significant turns (>30°): {len(high_turn)}/{len(df_stats)}")

        print(f"\n{'='*80}")
        print(f"✓ Complete! Visualizations saved to: {output_dir}")
        print(f"{'='*80}")


if __name__ == "__main__":
    main()
