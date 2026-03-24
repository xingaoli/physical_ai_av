#!/usr/bin/env python3
"""
Analyze the distribution of longitudinal acceleration and curvature
across the entire dataset to help determine optimal thresholds.

This script:
1. Loads all egomotion data from all chunks
2. Computes local_ax (longitudinal acceleration) and curvature
3. Plots histograms showing the distribution
4. Suggests threshold values based on percentiles
"""

import os
import zipfile
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from scipy.spatial.transform import Rotation as spt_Rotation
from tqdm import tqdm

# Use a nice style for plots
plt.style.use('seaborn-v0_8-whitegrid')


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


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess egomotion data to get local acceleration and curvature.

    Same logic as in 3_meta_action_annotation.py
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

    # Transform acceleration to local frame: local_a = R_inv @ world_a
    world_a = np.stack([
        df['ax'].values,
        df['ay'].values,
        df['az'].values
    ], axis=1)
    local_a = rots_inv.apply(world_a)
    df['local_ax'] = local_a[:, 0]  # Longitudinal acceleration

    # Smooth acceleration
    df['long_accel_smooth'] = median_filter(df['local_ax'], size=5)

    # Smooth curvature
    df['curvature_smooth'] = median_filter(df['curvature'].fillna(0), size=5)

    return df


def collect_all_data(
    egomotion_dir: Path,
    max_chunks: int = None
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Collect all long_accel_smooth and curvature_smooth values from all chunks.

    Returns:
        Tuple of (all_accel, all_curvature, total_videos)
    """
    chunk_files = sorted(egomotion_dir.glob("egomotion.chunk_*.zip"))

    if max_chunks:
        chunk_files = chunk_files[:max_chunks]

    print(f"Found {len(chunk_files)} chunk files")

    all_accel = []
    all_curvature = []
    total_videos = 0

    for chunk_zip in tqdm(chunk_files, desc="Processing chunks"):
        chunk_name = chunk_zip.stem.replace('egomotion.', '')

        # Create temp directory
        temp_dir = egomotion_dir / f".temp_analysis_{chunk_name}"
        temp_dir.mkdir(exist_ok=True)

        try:
            # Extract zip
            with zipfile.ZipFile(chunk_zip, 'r') as zf:
                zf.extractall(temp_dir)

            # Process each parquet file
            parquet_files = list(temp_dir.glob("*.egomotion.parquet"))

            for parquet_path in parquet_files:
                try:
                    df = pd.read_parquet(parquet_path)
                    df_processed = preprocess_data(df)

                    all_accel.extend(df_processed['long_accel_smooth'].tolist())
                    all_curvature.extend(df_processed['curvature_smooth'].tolist())
                    total_videos += 1

                except Exception as e:
                    print(f"  Error processing {parquet_path.name}: {e}")

        finally:
            # Cleanup
            for p in temp_dir.glob("*.egomotion.parquet"):
                p.unlink()
            try:
                temp_dir.rmdir()
            except:
                pass

    return np.array(all_accel), np.array(all_curvature), total_videos


def plot_distributions(
    accel: np.ndarray,
    curvature: np.ndarray,
    output_path: Path
):
    """
    Plot histograms of acceleration and curvature distributions
    with current thresholds marked.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Current thresholds from the script
    strong_accel_thresh = 2.0
    gentle_accel_thresh = 0.5
    gentle_decel_thresh = -0.5
    strong_decel_thresh = -2.0

    sharp_steer_thresh = 0.05
    gentle_steer_thresh = 0.01

    # 1. Longitudinal Acceleration - Full Range
    ax = axes[0, 0]
    counts, bins, patches = ax.hist(accel, bins=200, density=True, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.3)
    ax.axvline(strong_accel_thresh, color='red', linestyle='--', linewidth=2, label=f'Strong accel (+{strong_accel_thresh})')
    ax.axvline(gentle_accel_thresh, color='orange', linestyle='--', linewidth=2, label=f'Gentle accel (+{gentle_accel_thresh})')
    ax.axvline(gentle_decel_thresh, color='orange', linestyle='--', linewidth=2, label=f'Gentle decel ({gentle_decel_thresh})')
    ax.axvline(strong_decel_thresh, color='darkred', linestyle='--', linewidth=2, label=f'Strong decel ({strong_decel_thresh})')
    ax.axvline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    ax.set_xlabel('Longitudinal Acceleration (m/s²)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Longitudinal Acceleration Distribution (Full Range)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 2. Longitudinal Acceleration - Zoomed (focus on central region)
    ax = axes[0, 1]
    zoom_mask = (accel > -3) & (accel < 3)
    ax.hist(accel[zoom_mask], bins=150, density=True, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.3)
    ax.axvline(strong_accel_thresh, color='red', linestyle='--', linewidth=2, label=f'Strong accel (+{strong_accel_thresh})')
    ax.axvline(gentle_accel_thresh, color='orange', linestyle='--', linewidth=2, label=f'Gentle accel (+{gentle_accel_thresh})')
    ax.axvline(gentle_decel_thresh, color='orange', linestyle='--', linewidth=2, label=f'Gentle decel ({gentle_decel_thresh})')
    ax.axvline(strong_decel_thresh, color='darkred', linestyle='--', linewidth=2, label=f'Strong decel ({strong_decel_thresh})')
    ax.axvline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    ax.set_xlabel('Longitudinal Acceleration (m/s²)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Longitudinal Acceleration Distribution (Zoomed: ±3 m/s²)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 3. Curvature - Full Range
    ax = axes[1, 0]
    ax.hist(curvature, bins=200, density=True, alpha=0.7, color='forestgreen', edgecolor='black', linewidth=0.3)
    ax.axvline(sharp_steer_thresh, color='red', linestyle='--', linewidth=2, label=f'Sharp (+{sharp_steer_thresh})')
    ax.axvline(-sharp_steer_thresh, color='red', linestyle='--', linewidth=2, label=f'Sharp (-{sharp_steer_thresh})')
    ax.axvline(gentle_steer_thresh, color='orange', linestyle='--', linewidth=2, label=f'Gentle (+{gentle_steer_thresh})')
    ax.axvline(-gentle_steer_thresh, color='orange', linestyle='--', linewidth=2, label=f'Gentle (-{gentle_steer_thresh})')
    ax.axvline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    ax.set_xlabel('Curvature (1/m)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Curvature Distribution (Full Range)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    # 4. Curvature - Zoomed (focus on central region)
    ax = axes[1, 1]
    zoom_mask = (curvature > -0.1) & (curvature < 0.1)
    ax.hist(curvature[zoom_mask], bins=150, density=True, alpha=0.7, color='forestgreen', edgecolor='black', linewidth=0.3)
    ax.axvline(sharp_steer_thresh, color='red', linestyle='--', linewidth=2, label=f'Sharp (+{sharp_steer_thresh})')
    ax.axvline(-sharp_steer_thresh, color='red', linestyle='--', linewidth=2, label=f'Sharp (-{sharp_steer_thresh})')
    ax.axvline(gentle_steer_thresh, color='orange', linestyle='--', linewidth=2, label=f'Gentle (+{gentle_steer_thresh})')
    ax.axvline(-gentle_steer_thresh, color='orange', linestyle='--', linewidth=2, label=f'Gentle (-{gentle_steer_thresh})')
    ax.axvline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    ax.set_xlabel('Curvature (1/m)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Curvature Distribution (Zoomed: ±0.1 1/m)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {output_path}")


def find_valley_threshold(data: np.ndarray, min_val: float, max_val: float, n_points: int = 100) -> float:
    """
    Find the threshold in a range that minimizes the number of data points nearby.
    This finds a "valley" in the distribution where few data points exist.

    Args:
        data: 1D array of positive values (e.g., abs(accel))
        min_val: minimum threshold to consider
        max_val: maximum threshold to consider
        n_points: number of points to evaluate

    Returns:
        The threshold value with the fewest nearby points
    """
    test_values = np.linspace(min_val, max_val, n_points)
    min_count = float('inf')
    best_threshold = min_val

    for thresh in test_values:
        # Count points within ±5% of threshold (or ±0.05 for accel)
        margin = max(0.05, thresh * 0.05)
        count = np.sum((data >= thresh - margin) & (data <= thresh + margin))
        if count < min_count:
            min_count = count
            best_threshold = thresh

    return best_threshold


def print_statistics(accel: np.ndarray, curvature: np.ndarray):
    """Print percentile-based statistics to help choose thresholds."""
    print("\n" + "="*80)
    print("LONGITUDINAL ACCELERATION STATISTICS")
    print("="*80)

    accel_percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print(f"{'Percentile':<15} {'Value (m/s²)':<15} {'Description'}")
    print("-"*80)
    for p in accel_percentiles:
        val = np.percentile(np.abs(accel), p)
        desc = ""
        if p == 90:
            desc = "~ Gentle accel threshold candidate"
        elif p == 95:
            desc = "~ Strong accel threshold candidate"
        print(f"{p}%<15 {val:>14.3f}      {desc}")

    print(f"\nCurrent thresholds:")
    print(f"  Gentle accel:  +{0.5} m/s²")
    print(f"  Strong accel:  +{2.0} m/s²")
    print(f"  Gentle decel:   {0.5} m/s²")
    print(f"  Strong decel:   {2.0} m/s²")

    # Find valley threshold for gentle accel
    best_gentle = find_valley_threshold(np.abs(accel), 0.4, 1.0, 60)
    print(f"\n🔍 Recommended gentle accel threshold (valley detection): {best_gentle:.3f} m/s²")

    # Count frames near thresholds
    print(f"\nFrames near threshold boundaries (±0.05 m/s²):")
    for thresh, name in [(0.5, "gentle (current)"), (best_gentle, f"gentle (rec {best_gentle:.2f})"), (2.0, "strong")]:
        near = np.sum((accel >= thresh - 0.05) & (accel <= thresh + 0.05))
        near_neg = np.sum((accel >= -thresh - 0.05) & (accel <= -thresh + 0.05))
        total_near = near + near_neg
        pct = 100 * total_near / len(accel)
        print(f"  ±0.05 around {name} ({thresh:+.2f}): {total_near:,} frames ({pct:.2f}%)")

    print("\n" + "="*80)
    print("CURVATURE STATISTICS")
    print("="*80)

    print(f"{'Percentile':<15} {'Value (1/m)':<15} {'Description'}")
    print("-"*80)
    for p in accel_percentiles:
        val = np.percentile(np.abs(curvature), p)
        desc = ""
        if p == 90:
            desc = "~ Gentle steer threshold candidate"
        elif p == 95:
            desc = "~ Sharp steer threshold candidate"
        print(f"{p}%<15 {val:>14.4f}      {desc}")

    print(f"\nCurrent thresholds:")
    print(f"  Gentle steer:  ±{0.01} 1/m")
    print(f"  Sharp steer:   ±{0.05} 1/m")

    # Find valley threshold for gentle steer
    best_gentle_steer = find_valley_threshold(np.abs(curvature), 0.005, 0.03, 50)
    print(f"\n🔍 Recommended gentle steer threshold (valley detection): {best_gentle_steer:.4f} 1/m")

    # Count frames near thresholds
    print(f"\nFrames near threshold boundaries (±0.001 1/m):")
    for thresh, name in [(0.01, "gentle (current)"), (best_gentle_steer, f"gentle (rec {best_gentle_steer:.4f})"), (0.05, "sharp")]:
        near_pos = np.sum((curvature >= thresh - 0.001) & (curvature <= thresh + 0.001))
        near_neg = np.sum((curvature >= -thresh - 0.001) & (curvature <= -thresh + 0.001))
        near = near_pos + near_neg
        pct = 100 * near / len(curvature)
        print(f"  ±0.001 around {name} (±{thresh:.4f}): {near:,} frames ({pct:.2f}%)")


def main():
    """Main function."""
    import argparse
    parser = argparse.ArgumentParser(description='Analyze meta-action threshold distributions')
    parser.add_argument('--max-chunks', type=int, default=None, help='Limit number of chunks to process')
    parser.add_argument('--output', type=str, default=None, help='Output path for plot')
    args = parser.parse_args()

    # Load environment
    script_dir = Path(__file__).parent.parent.parent
    env_path = script_dir / ".env"
    env_vars = load_env(env_path)

    data_dir = Path(env_vars.get('PHYSICAL_AI_AV_DATA_DIR'))
    egomotion_dir = data_dir / "labels" / "egomotion_corrected"

    print(f"Data directory: {data_dir}")
    print(f"Egomotion directory: {egomotion_dir}")

    # Collect all data
    print("\nCollecting data from all chunks...")
    accel, curvature, total_videos = collect_all_data(egomotion_dir, args.max_chunks)

    print(f"\nCollected {len(accel):,} frames from {total_videos} videos")

    # Print statistics
    print_statistics(accel, curvature)

    # Plot distributions
    output_path = Path(args.output) if args.output else script_dir / "threshold_analysis.png"
    plot_distributions(accel, curvature, output_path)

    print(f"\n✓ Analysis complete!")


if __name__ == "__main__":
    main()
