#!/usr/bin/env python3
"""
Visualize extracted trajectories and physical quantity distributions.

This script visualizes:
1. All normalized trajectories (sampled)
2. Physical quantity distributions (speed, acceleration, curvature)
3. Trajectory statistics

Usage:
    # Visualize trajectories from all chunks
    python3 auto_labeling/traj_cluster/1_visualize_trajectories.py

    # Visualize specific chunks
    python3 auto_labeling/traj_cluster/1_visualize_trajectories.py --chunks chunk_0000 chunk_0001

    # Show more samples
    python3 auto_labeling/traj_cluster/1_visualize_trajectories.py --n-samples 500
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm


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


def load_trajectories(trajectories_dir: Path, chunks: List[str] = None) -> pd.DataFrame:
    """
    Load trajectory data from parquet files.

    Args:
        trajectories_dir: Directory containing trajectory parquet files
        chunks: Optional list of specific chunks to load

    Returns:
        DataFrame with all trajectories
    """
    all_trajectories = []

    # Find trajectory files
    if chunks:
        trajectory_files = [trajectories_dir / f"trajectories.{c}" / "trajectories.parquet" for c in chunks]
        trajectory_files = [f for f in trajectory_files if f.exists()]
    else:
        trajectory_files = sorted(trajectories_dir.glob("*/trajectories.parquet"))

    print(f"Found {len(trajectory_files)} trajectory files")

    for f in tqdm(trajectory_files, desc="Loading trajectories"):
        df = pd.read_parquet(f)
        all_trajectories.append(df)

    if not all_trajectories:
        return pd.DataFrame()

    trajectories_df = pd.concat(all_trajectories, ignore_index=True)
    print(f"Loaded {len(trajectories_df)} trajectories")

    return trajectories_df


def plot_all_trajectories(
    trajectories_df: pd.DataFrame,
    output_path: Path,
    n_samples: int = None,
    title_suffix: str = "",
    use_time_gradient: bool = True
):
    """
    Plot a sample of all trajectories with time gradient color.

    Note: x is longitudinal (forward direction), y is lateral (left/right).
    We plot y on x-axis and x on y-axis to match vehicle's perspective.

    Args:
        trajectories_df: DataFrame with trajectory data
        output_path: Path to save the plot
        n_samples: Number of trajectories to sample (None = use all)
        title_suffix: Suffix to add to title
        use_time_gradient: Whether to use time gradient coloring
    """
    # Sample trajectories
    if n_samples and len(trajectories_df) > n_samples:
        sample_df = trajectories_df.sample(n=n_samples, random_state=42)
        print(f"  Plotting {len(sample_df)} samples (from {len(trajectories_df)} total)...")
    else:
        sample_df = trajectories_df
        print(f"  Plotting all {len(sample_df)} trajectories...")

    # Use a cleaner, more aesthetic style
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor('#1a1a2e')

    # Beautiful colormap: from cyan (start) to magenta (end)
    cmap = plt.get_cmap('plasma')

    # Plot 1: Top-down view
    ax1 = axes[0]
    ax1.set_facecolor('#16213e')

    if use_time_gradient:
        # Plot with time gradient - cleaner look
        for _, row in sample_df.iterrows():
            dx = np.array(row['dx'])
            dy = np.array(row['dy'])
            n_points = len(dx)

            # Plot each segment with gradient color
            for i in range(n_points - 1):
                color = cmap(i / (n_points - 1))
                ax1.plot(dy[i:i+2], dx[i:i+2], color=color, alpha=0.3, linewidth=0.8)
    else:
        # Simple gradient lines
        for _, row in sample_df.iterrows():
            dx = np.array(row['dx'])
            dy = np.array(row['dy'])
            ax1.plot(dy, dx, alpha=0.25, linewidth=0.6, color='cyan')

    # Start point - subtle and clean
    ax1.scatter([0], [0], c='#00d4ff', s=60, marker='o', zorder=5, alpha=0.9,
               edgecolors='white', linewidth=0.5, label='Start')

    ax1.set_xlabel('Lateral Y (m)', fontsize=11, color='#e0e0e0', fontweight='bold')
    ax1.set_ylabel('Longitudinal X (m)', fontsize=11, color='#e0e0e0', fontweight='bold')
    ax1.set_title(f'Trajectory Overview{title_suffix}', fontsize=13, color='#e0e0e0', fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.15, linestyle='-', color='#4a5568', linewidth=0.5)
    ax1.axis('equal')
    ax1.axhline(0, color='#00d4ff', linestyle='--', alpha=0.2, linewidth=1)
    ax1.axvline(0, color='#4a5568', linestyle='--', alpha=0.2, linewidth=1)
    ax1.legend(loc='upper right', facecolor='#16213e', edgecolor='#4a5568', labelcolor='#e0e0e0')
    ax1.tick_params(colors='#a0aec0')

    # Plot 2: Zoomed in
    ax2 = axes[1]
    ax2.set_facecolor('#16213e')

    if use_time_gradient:
        for _, row in sample_df.iterrows():
            dx = np.array(row['dx'])
            dy = np.array(row['dy'])
            n_points = len(dx)

            for i in range(n_points - 1):
                color = cmap(i / (n_points - 1))
                ax2.plot(dy[i:i+2], dx[i:i+2], color=color, alpha=0.3, linewidth=0.8)
    else:
        for _, row in sample_df.iterrows():
            dx = np.array(row['dx'])
            dy = np.array(row['dy'])
            ax2.plot(dy, dx, alpha=0.25, linewidth=0.6, color='cyan')

    ax2.scatter([0], [0], c='#00d4ff', s=60, marker='o', zorder=5, alpha=0.9,
               edgecolors='white', linewidth=0.5, label='Start')

    ax2.set_xlabel('Lateral Y (m)', fontsize=11, color='#e0e0e0', fontweight='bold')
    ax2.set_ylabel('Longitudinal X (m)', fontsize=11, color='#e0e0e0', fontweight='bold')
    ax2.set_title(f'Trajectory Detail (Zoomed){title_suffix}', fontsize=13, color='#e0e0e0', fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.15, linestyle='-', color='#4a5568', linewidth=0.5)
    ax2.axis('equal')
    ax2.set_ylim(-5, 60)
    ax2.set_xlim(-30, 30)
    ax2.axhline(0, color='#00d4ff', linestyle='--', alpha=0.2, linewidth=1)
    ax2.axvline(0, color='#4a5568', linestyle='--', alpha=0.2, linewidth=1)
    ax2.legend(loc='upper right', facecolor='#16213e', edgecolor='#4a5568', labelcolor='#e0e0e0')
    ax2.tick_params(colors='#a0aec0')

    # Add colorbar for time gradient
    if use_time_gradient:
        from matplotlib.cm import ScalarMappable
        sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=8))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label('Time from start (seconds)', fontsize=10, color='#e0e0e0', fontweight='bold')
        cbar.ax.yaxis.set_tick_params(color='#a0aec0')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#a0aec0')

    # Add info text - cleaner style
    info_text = f"{len(trajectories_df):,} total | {len(sample_df)} shown"
    fig.text(0.5, 0.02, info_text, ha='center', fontsize=11, color='#a0aec0',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#16213e', edgecolor='#4a5568', alpha=0.8))

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()

    print(f"  Saved to {output_path}")


def plot_physical_quantity_distribution(
    trajectories_df: pd.DataFrame,
    output_path: Path,
    title_suffix: str = ""
):
    """
    Plot physical quantity distributions (speed, acceleration, curvature).

    This helps understand the distribution of driving scenarios without meta-action labels.

    Args:
        trajectories_df: DataFrame with trajectory data
        output_path: Path to save the plot
        title_suffix: Suffix to add to title
    """
    print("Plotting physical quantity distributions...")

    # Collect statistics from all trajectories
    all_speeds = []
    all_accels = []
    all_curvatures = []
    all_final_speeds = []
    all_mean_accels = []

    for _, row in tqdm(trajectories_df.iterrows(), total=len(trajectories_df), desc="Extracting quantities"):
        speed = np.array(row['speed'])
        accel = np.array(row['acceleration'])  # local_ax (longitudinal)
        curvature = np.array(row['curvature'])

        all_speeds.extend(speed.tolist())
        all_accels.extend(accel.tolist())
        all_curvatures.extend(curvature.tolist())
        all_final_speeds.append(speed[-1])
        all_mean_accels.append(np.mean(accel))

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Physical Quantity Distribution{title_suffix}', fontsize=14, fontweight='bold')

    # 1. Speed distribution
    ax1 = axes[0, 0]
    ax1.hist(all_speeds, bins=100, alpha=0.7, color='green', edgecolor='black')
    ax1.set_xlabel('Speed (m/s)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax1.set_title('Speed Distribution (All Frames)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axvline(np.median(all_speeds), color='red', linestyle='--', linewidth=2,
                label=f"Median: {np.median(all_speeds):.1f} m/s")
    ax1.legend()

    # 2. Acceleration distribution
    ax2 = axes[0, 1]
    ax2.hist(all_accels, bins=100, alpha=0.7, color='orange', edgecolor='black')
    ax2.set_xlabel('Longitudinal Acceleration (m/s²)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax2.set_title('Acceleration Distribution (All Frames)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axvline(0, color='black', linestyle='-', alpha=0.5, linewidth=1)
    ax2.axvline(np.median(all_accels), color='red', linestyle='--', linewidth=2,
                label=f"Median: {np.median(all_accels):.2f} m/s²")
    ax2.axvline(0.5, color='green', linestyle=':', alpha=0.5, label='Gentle accel (+0.5)')
    ax2.axvline(-0.5, color='red', linestyle=':', alpha=0.5, label='Gentle decel (-0.5)')
    ax2.legend(fontsize=8)

    # 3. Curvature distribution
    ax3 = axes[0, 2]
    ax3.hist(all_curvatures, bins=100, alpha=0.7, color='cyan', edgecolor='black')
    ax3.set_xlabel('Curvature (1/m)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax3.set_title('Curvature Distribution (All Frames)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.axvline(0, color='black', linestyle='-', alpha=0.5, linewidth=1)
    ax3.axvline(np.median(all_curvatures), color='red', linestyle='--', linewidth=2,
                label=f"Median: {np.median(all_curvatures):.3f} 1/m")
    ax3.axvline(0.01, color='orange', linestyle=':', alpha=0.5, label='Gentle steer (±0.01)')
    ax3.axvline(-0.01, color='orange', linestyle=':', alpha=0.5)
    ax3.axvline(0.05, color='darkorange', linestyle=':', alpha=0.5, label='Sharp steer (±0.05)')
    ax3.axvline(-0.05, color='darkorange', linestyle=':', alpha=0.5)
    ax3.legend(fontsize=8)
    ax3.set_xlim(-0.15, 0.15)

    # 4. Final speed distribution (per trajectory)
    ax4 = axes[1, 0]
    ax4.hist(all_final_speeds, bins=50, alpha=0.7, color='purple', edgecolor='black')
    ax4.set_xlabel('Final Speed (m/s)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax4.set_title('Final Speed Distribution (Per Trajectory)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.axvline(np.median(all_final_speeds), color='red', linestyle='--', linewidth=2,
                label=f"Median: {np.median(all_final_speeds):.1f} m/s")
    ax4.legend()

    # 5. Mean acceleration distribution (per trajectory)
    ax5 = axes[1, 1]
    ax5.hist(all_mean_accels, bins=50, alpha=0.7, color='brown', edgecolor='black')
    ax5.set_xlabel('Mean Longitudinal Acceleration (m/s²)', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax5.set_title('Mean Acceleration Distribution (Per Trajectory)', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.axvline(0, color='black', linestyle='-', alpha=0.5, linewidth=1)
    ax5.axvline(np.median(all_mean_accels), color='red', linestyle='--', linewidth=2,
                label=f"Median: {np.median(all_mean_accels):.2f} m/s²")
    ax5.legend()

    # 6. 2D histogram: Speed vs Acceleration
    ax6 = axes[1, 2]
    h = ax6.hist2d(all_speeds, all_accels, bins=50, cmap='Blues')
    ax6.set_xlabel('Speed (m/s)', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Acceleration (m/s²)', fontsize=11, fontweight='bold')
    ax6.set_title('Speed vs Acceleration (Density)', fontsize=12, fontweight='bold')
    ax6.axhline(0, color='white', linestyle='-', alpha=0.3, linewidth=1)
    plt.colorbar(h[3], ax=ax6, label='Count')

    # Add total count
    fig.text(0.5, 0.02, f"Total trajectories: {len(trajectories_df):,} | Total frames: {len(all_speeds):,}",
             ha='center', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved to {output_path}")


def plot_trajectory_statistics(
    trajectories_df: pd.DataFrame,
    output_path: Path,
    title_suffix: str = ""
):
    """
    Plot trajectory statistics (displacement, heading change, etc.).

    Args:
        trajectories_df: DataFrame with trajectory data
        output_path: Path to save the plot
        title_suffix: Suffix to add to title
    """
    print("Plotting trajectory statistics...")

    # Extract statistics from each trajectory
    stats = []

    for _, row in tqdm(trajectories_df.iterrows(), total=len(trajectories_df), desc="Extracting stats"):
        dx = np.array(row['dx'])
        dy = np.array(row['dy'])
        dyaw = np.array(row['dyaw'])
        speed = np.array(row['speed'])
        local_vx = np.array(row['local_vx'])
        acceleration = np.array(row['acceleration'])
        curvature = np.array(row['curvature'])

        # Final displacement
        final_disp = np.sqrt(dx[-1]**2 + dy[-1]**2)

        # Lateral displacement
        initial_yaw = dyaw[0]
        lateral_disp = dy[-1] * np.cos(initial_yaw) - dx[-1] * np.sin(initial_yaw)

        # Heading change
        heading_change = np.abs(dyaw[-1] - dyaw[0])

        # Mean speed
        mean_speed = np.mean(speed)

        # Final longitudinal velocity
        final_vx = local_vx[-1]

        # Mean acceleration
        mean_accel = np.mean(acceleration)

        # Max curvature (absolute)
        max_curvature = np.max(np.abs(curvature))

        stats.append({
            'final_displacement': final_disp,
            'lateral_displacement': lateral_disp,
            'heading_change': heading_change,
            'mean_speed': mean_speed,
            'final_vx': final_vx,
            'mean_accel': mean_accel,
            'max_curvature': max_curvature,
        })

    stats_df = pd.DataFrame(stats)

    # Create figure
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    # 1. Final displacement histogram
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(stats_df['final_displacement'], bins=50, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Final Displacement (m)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax1.set_title('Distribution of Final Displacement', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axvline(stats_df['final_displacement'].median(), color='red', linestyle='--',
                label=f"Median: {stats_df['final_displacement'].median():.1f}m")
    ax1.legend()

    # 2. Lateral displacement histogram
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(stats_df['lateral_displacement'], bins=50, alpha=0.7, color='orange', edgecolor='black')
    ax2.set_xlabel('Lateral Displacement (m)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax2.set_title('Distribution of Lateral Displacement', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axvline(stats_df['lateral_displacement'].median(), color='red', linestyle='--',
                label=f"Median: {stats_df['lateral_displacement'].median():.1f}m")
    ax2.legend()

    # 3. Heading change histogram
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(np.degrees(stats_df['heading_change']), bins=50, alpha=0.7, color='green', edgecolor='black')
    ax3.set_xlabel('Heading Change (degrees)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax3.set_title('Distribution of Heading Change', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.axvline(np.degrees(stats_df['heading_change']).median(), color='red', linestyle='--',
                label=f"Median: {np.degrees(stats_df['heading_change']).median():.1f}°")
    ax3.legend()

    # 4. Mean speed histogram
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(stats_df['mean_speed'], bins=50, alpha=0.7, color='purple', edgecolor='black')
    ax4.set_xlabel('Mean Speed (m/s)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax4.set_title('Distribution of Mean Speed', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.axvline(stats_df['mean_speed'].median(), color='red', linestyle='--',
                label=f"Median: {stats_df['mean_speed'].median():.1f} m/s")
    ax4.legend()

    # 5. Scatter: lateral vs heading change
    ax5 = fig.add_subplot(gs[2, 0])
    scatter = ax5.scatter(np.degrees(stats_df['heading_change']), stats_df['lateral_displacement'],
                         c=stats_df['mean_speed'], cmap='viridis', alpha=0.3, s=1)
    ax5.set_xlabel('Heading Change (degrees)', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Lateral Displacement (m)', fontsize=11, fontweight='bold')
    ax5.set_title('Lateral Displacement vs Heading Change', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax5, label='Mean Speed (m/s)')

    # 6. Scatter: displacement vs speed
    ax6 = fig.add_subplot(gs[2, 1])
    scatter = ax6.scatter(stats_df['final_displacement'], stats_df['mean_speed'],
                         c=np.degrees(stats_df['heading_change']), cmap='coolwarm', alpha=0.3, s=1)
    ax6.set_xlabel('Final Displacement (m)', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Mean Speed (m/s)', fontsize=11, fontweight='bold')
    ax6.set_title('Displacement vs Speed', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax6, label='Heading Change (deg)')

    plt.suptitle(f'Trajectory Statistics{title_suffix}', fontsize=14, fontweight='bold')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved to {output_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Visualize extracted trajectories',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--chunks',
        nargs='+',
        default=None,
        help='Specific chunk IDs to visualize (default: all)'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=5000,
        help='Number of trajectory samples to plot (default: 5000)'
    )
    args = parser.parse_args()

    # Load environment
    script_dir = Path(__file__).parent.parent.parent
    env_path = script_dir / ".env"
    env_vars = load_env(env_path)

    data_dir = Path(env_vars.get('PHYSICAL_AI_AV_DATA_DIR',
                                   '/home/xingao/data/PhysicalAI-Autonomous-Vehicles-base-wo-lidar-radar'))
    trajectories_dir = data_dir / "labels" / "trajectories"
    output_dir = trajectories_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("Trajectory Visualization")
    print("="*80)
    print(f"Data directory: {data_dir}")
    print(f"Trajectories directory: {trajectories_dir}")
    print(f"Output directory: {output_dir}")
    print("="*80)

    # Load trajectories
    print("\nLoading trajectories...")
    trajectories_df = load_trajectories(trajectories_dir, args.chunks)

    if trajectories_df.empty:
        print("No trajectories found. Please run trajectory extraction first.")
        return

    # Generate visualizations
    print("\n" + "="*80)
    print("Generating visualizations...")
    print("="*80 + "\n")

    # 1. All trajectories overview
    print("\n[TRAJECTORIES OVERVIEW]")
    plot_all_trajectories(
        trajectories_df,
        output_dir / 'trajectories.png',
        n_samples=args.n_samples,
        title_suffix=f" ({len(trajectories_df):,} total)",
        use_time_gradient=True
    )

    # 2. Physical quantity distribution
    print("\n[PHYSICAL QUANTITY DISTRIBUTION]")
    plot_physical_quantity_distribution(
        trajectories_df,
        output_dir / 'physical_quantity_distribution.png',
        title_suffix=f" ({len(trajectories_df):,} trajectories)"
    )

    # 3. Trajectory statistics
    print("\n[TRAJECTORY STATISTICS]")
    plot_trajectory_statistics(
        trajectories_df,
        output_dir / 'trajectory_statistics.png',
        title_suffix=f" ({len(trajectories_df):,} trajectories)"
    )

    # Summary
    print(f"\n{'='*80}")
    print("✓ Visualization complete!")
    print(f"{'='*80}")
    print(f"Total trajectories: {len(trajectories_df):,}")
    print(f"Output saved to: {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
