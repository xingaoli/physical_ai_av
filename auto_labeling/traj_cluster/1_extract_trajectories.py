#!/usr/bin/env python3
"""
Extract trajectories from egomotion data for trajectory clustering.

This script extracts trajectories at fixed intervals (every 0.5s) from raw egomotion data,
without relying on meta-action filtering. This provides an unbiased view of the distribution.

Key points:
- Start from index 0, extract every 5th frame (0.5s interval)
- Extract 17 points per trajectory (8s total)
- Position coordinates: relative to first frame (t0) coordinate system
- Velocity/Acceleration: in EACH frame's local vehicle coordinate system

Usage:
    python3 auto_labeling/traj_cluster/1_extract_trajectories.py --chunks chunk_0000
"""

import argparse
import zipfile
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
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


def extract_trajectory_from_start_idx(
    egomotion_df: pd.DataFrame,
    start_idx: int,
    trajectory_points: int = 17,  # 17 points @ 0.5s interval = 8s
    time_interval: int = 5,  # 5 frames = 0.5s at 10Hz
) -> Optional[Dict]:
    """
    Extract trajectory starting from a given index at fixed intervals.

    Args:
        egomotion_df: DataFrame with egomotion data (201 rows for 20.1s at 10Hz)
        start_idx: Starting frame index
        trajectory_points: Number of points to extract (default 17 = 8s at 0.5s intervals)
        time_interval: Frame interval between points (default 5 = 0.5s)

    Returns:
        Dictionary with trajectory data, or None if extraction fails

    Coordinate transformations:
        - Position (dx, dy, dz): relative to FIRST frame (t0) coordinate system
        - Velocity (local_vx, local_vy, local_vz): in EACH frame's local vehicle frame
        - Acceleration (local_ax, local_ay, local_az): in EACH frame's local vehicle frame
    """
    # Calculate the actual end index
    end_idx = start_idx + (trajectory_points - 1) * time_interval

    # Check if we have enough frames
    if end_idx >= len(egomotion_df):
        return None

    # Get frame indices at fixed intervals
    frame_indices = [start_idx + i * time_interval for i in range(trajectory_points)]
    segment = egomotion_df.iloc[frame_indices].copy().reset_index(drop=True)

    # === Transform positions to FIRST frame (t0) coordinate system ===
    # Get t0 (first frame) pose
    t0_quat = np.array([
        segment.iloc[0]['qx'],
        segment.iloc[0]['qy'],
        segment.iloc[0]['qz'],
        segment.iloc[0]['qw']
    ])
    t0_xyz = np.array([
        segment.iloc[0]['x'],
        segment.iloc[0]['y'],
        segment.iloc[0]['z']
    ])

    # Create rotation object for t0 and get inverse (world -> t0_ego transformation)
    t0_rot = spt_Rotation.from_quat(t0_quat)
    t0_rot_inv = t0_rot.inv()

    # Extract all world positions
    world_xyz = np.stack([
        segment['x'].values,
        segment['y'].values,
        segment['z'].values
    ], axis=1)  # Shape: (N, 3)

    # Transform to t0 ego frame: local = R_inv @ (world - t0_xyz)
    local_xyz_t0 = t0_rot_inv.apply(world_xyz - t0_xyz)

    # Store position relative to t0
    dx = local_xyz_t0[:, 0]  # Longitudinal (forward)
    dy = local_xyz_t0[:, 1]  # Lateral (left)
    dz = local_xyz_t0[:, 2]  # Vertical

    # === Transform rotations to t0 frame ===
    world_quats = np.stack([
        segment['qx'].values,
        segment['qy'].values,
        segment['qz'].values,
        segment['qw'].values
    ], axis=1)

    # local_quat = t0_rot_inv * world_quat
    local_rots = t0_rot_inv * spt_Rotation.from_quat(world_quats)

    # Extract yaw from local rotation matrices
    # yaw = atan2(R[1,0], R[0,0])
    local_mats = local_rots.as_matrix()
    dyaw = np.arctan2(local_mats[:, 1, 0], local_mats[:, 0, 0])

    # === Transform velocities to EACH frame's local vehicle coordinate system ===
    # This is CRITICAL: velocity should be in the CURRENT frame's local frame, not t0 frame
    # Reference: meta_action annotation preprocessing

    # Get rotation for each frame and its inverse
    rots = spt_Rotation.from_quat(world_quats)
    rots_inv = rots.inv()

    # Transform velocity: local_v = R_inv @ world_v
    world_v = np.stack([
        segment['vx'].values,
        segment['vy'].values,
        segment['vz'].values
    ], axis=1)
    local_v = rots_inv.apply(world_v)

    local_vx = local_v[:, 0]  # Longitudinal velocity (forward = positive, reverse = negative)
    local_vy = local_v[:, 1]  # Lateral velocity
    local_vz = local_v[:, 2]  # Vertical velocity

    # === Transform accelerations to EACH frame's local vehicle coordinate system ===
    world_a = np.stack([
        segment['ax'].values,
        segment['ay'].values,
        segment['az'].values
    ], axis=1)
    local_a = rots_inv.apply(world_a)

    local_ax = local_a[:, 0]  # Longitudinal acceleration
    local_ay = local_a[:, 1]  # Lateral acceleration
    local_az = local_a[:, 2]  # Vertical acceleration

    # === Compute derived quantities ===
    # Speed magnitude (scalar, always positive)
    speed = np.sqrt(local_vx**2 + local_vy**2)

    # Curvature is already in the data (coordinate-system independent)
    curvature = segment['curvature'].values

    return {
        'frame_indices': frame_indices,
        'dx': dx,
        'dy': dy,
        'dz': dz,
        'dyaw': dyaw,
        'local_vx': local_vx,
        'local_vy': local_vy,
        'local_vz': local_vz,
        'local_ax': local_ax,
        'local_ay': local_ay,
        'local_az': local_az,
        'speed': speed,
        'acceleration': local_ax,  # Longitudinal acceleration for convenience
        'curvature': curvature,
        'timestamp': segment['timestamp'].values,
    }


def extract_trajectories_from_video(
    video_uuid: str,
    egomotion_df: pd.DataFrame,
    trajectory_points: int = 17,
    time_interval: int = 5,
) -> list[dict]:
    """
    Extract all trajectories from a video at fixed intervals.

    Starting from index 0, extract trajectories every 0.5s (5 frames).
    This provides unbiased sampling of the distribution.

    Args:
        video_uuid: Video identifier
        egomotion_df: Egomotion DataFrame (201 frames)
        trajectory_points: Number of points per trajectory (default 17)
        time_interval: Frame interval between points (default 5)

    Returns:
        List of all trajectories (not separated by normal/reverse)
    """
    total_frames = len(egomotion_df)
    max_start_idx = total_frames - (trajectory_points - 1) * time_interval

    # Start from index 0, every 5 frames
    start_indices = list(range(0, max_start_idx, time_interval))

    all_trajectories = []

    for start_idx in start_indices:
        traj = extract_trajectory_from_start_idx(
            egomotion_df, start_idx, trajectory_points, time_interval
        )

        if traj is None:
            continue

        # Add metadata
        traj['video_uuid'] = video_uuid
        traj['start_idx'] = start_idx

        all_trajectories.append(traj)

    return all_trajectories


def load_egomotion_files_from_chunk(
    chunk_id: str,
    egomotion_dir: Path,
    temp_dir: Path
) -> List[tuple[str, pd.DataFrame]]:
    """
    Load all egomotion files from a chunk zip.

    Args:
        chunk_id: Chunk identifier
        egomotion_dir: Directory containing egomotion zip files
        temp_dir: Temporary directory for extraction

    Returns:
        List of (video_uuid, egomotion_df) tuples
    """
    zip_path = egomotion_dir / f"egomotion.{chunk_id}.zip"
    if not zip_path.exists():
        print(f"Warning: Egomotion zip not found: {zip_path}")
        return []

    results = []

    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(temp_dir)

    for parquet_file in temp_dir.glob("*.egomotion.parquet"):
        video_uuid = parquet_file.stem
        try:
            df = pd.read_parquet(parquet_file)
            results.append((video_uuid, df))
        except Exception as e:
            print(f"Warning: Failed to load {parquet_file}: {e}")

    # Cleanup
    for f in temp_dir.glob("*"):
        f.unlink()

    return results


def process_chunk(
    chunk_id: str,
    data_dir: Path,
    output_dir: Path,
    trajectory_points: int = 17,
    time_interval: int = 5,
) -> Dict[str, int]:
    """
    Process all videos in a chunk.

    Args:
        chunk_id: Chunk identifier
        data_dir: Base data directory
        output_dir: Output directory for trajectories
        trajectory_points: Number of points per trajectory
        time_interval: Frame interval between points

    Returns:
        Statistics dictionary
    """
    labels_dir = data_dir / "labels"
    egomotion_dir = labels_dir / "egomotion_corrected"
    temp_dir = output_dir / f".temp_{chunk_id}"
    temp_dir.mkdir(exist_ok=True)

    print(f"  Loading egomotion data for {chunk_id}...")
    egomotion_data = load_egomotion_files_from_chunk(chunk_id, egomotion_dir, temp_dir)
    print(f"  Found {len(egomotion_data)} videos")

    # Create output paths
    chunk_output_dir = output_dir / f"trajectories.{chunk_id}"
    chunk_output_dir.mkdir(parents=True, exist_ok=True)

    all_trajectories = []

    # Process each video
    for video_uuid, ego_df in tqdm(egomotion_data, desc=f"Extracting {chunk_id}", leave=False):
        # Extract trajectories at fixed intervals
        trajs = extract_trajectories_from_video(
            video_uuid, ego_df, trajectory_points, time_interval
        )

        all_trajectories.extend(trajs)

    # Save trajectories
    print(f"  Saving trajectories...")

    if all_trajectories:
        trajectories_df = pd.DataFrame(all_trajectories)
        output_path = chunk_output_dir / "trajectories.parquet"
        trajectories_df.to_parquet(output_path, index=False)
        print(f"    Total: {len(all_trajectories)} trajectories -> {output_path.name}")

    # Cleanup temp
    temp_dir.rmdir()

    return {
        'total': len(all_trajectories)
    }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Extract trajectories from egomotion data for clustering',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--chunks',
        nargs='+',
        default=None,
        help='Specific chunk IDs to process (default: all)'
    )
    parser.add_argument(
        '--trajectory-points',
        type=int,
        default=17,
        help='Number of points per trajectory (default: 17 = 8s at 0.5s intervals)'
    )
    parser.add_argument(
        '--time-interval',
        type=int,
        default=5,
        help='Frame interval between points (default: 5 = 0.5s at 10Hz)'
    )
    args = parser.parse_args()

    # Load environment
    script_dir = Path(__file__).parent.parent.parent
    env_path = script_dir / ".env"
    env_vars = load_env(env_path)

    data_dir = Path(env_vars.get('PHYSICAL_AI_AV_DATA_DIR',
                                   '/home/xingao/data/PhysicalAI-Autonomous-Vehicles-base-wo-lidar-radar'))
    output_dir = data_dir / "labels" / "trajectories"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("Fixed-Interval Trajectory Extraction (for Clustering)")
    print("="*80)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Trajectory points: {args.trajectory_points} points")
    print(f"Time interval: {args.time_interval} frames ({args.time_interval * 0.1:.1f}s at 10Hz)")
    print(f"Total duration: {(args.trajectory_points - 1) * args.time_interval * 0.1:.1f}s")
    print("="*80)

    # Get available chunks
    labels_dir = data_dir / "labels"
    egomotion_dir = labels_dir / "egomotion_corrected"

    if not egomotion_dir.exists():
        print(f"Error: Egomotion directory not found: {egomotion_dir}")
        return

    all_chunk_zips = sorted(egomotion_dir.glob("egomotion.chunk_*.zip"))
    all_chunk_ids = [z.stem.replace("egomotion.", "") for z in all_chunk_zips]

    if args.chunks:
        chunk_ids_to_process = [c for c in args.chunks if c in all_chunk_ids]
    else:
        chunk_ids_to_process = all_chunk_ids

    if not chunk_ids_to_process:
        print("Error: No valid chunks to process")
        return

    print(f"\nWill process {len(chunk_ids_to_process)} chunks:")
    for cid in chunk_ids_to_process:
        print(f"  - {cid}")

    # Process chunks
    print(f"\n{'='*80}")
    print("Processing...")
    print(f"{'='*80}\n")

    total_stats = {'total': 0}

    for chunk_id in chunk_ids_to_process:
        print(f"\n[{chunk_id}]")
        stats = process_chunk(
            chunk_id, data_dir, output_dir,
            args.trajectory_points, args.time_interval
        )
        total_stats['total'] += stats['total']

    # Summary
    print(f"\n{'='*80}")
    print("✓ Trajectory extraction complete!")
    print(f"{'='*80}")
    print(f"Total trajectories: {total_stats['total']}")
    print(f"Output saved to: {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
