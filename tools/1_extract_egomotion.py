#!/usr/bin/env python3
"""
Extract egomotion data using global timestamps.

This script creates egomotion data aligned with the global 10Hz timeline,
ensuring all 7 cameras use exactly the same timestamps for perfect alignment.

Output format matches the original egomotion structure:
- Organized by chunk (egomotion.chunk_0000.zip)
- Each video has {uuid}.egomotion.parquet with global timestamps

Usage:
    python3 tools/extract_egomotion_global.py --chunks chunk_0000
"""

import os
import zipfile
import argparse
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np
import physical_ai_av
from tqdm import tqdm


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


def process_chunk(
    chunk_id: str,
    data_dir: Path,
    output_dir: Path,
    avdi: physical_ai_av.OfflinePhysicalAIAVDatasetInterface,
    global_timestamps: np.ndarray
):
    """
    Process all videos in a chunk.

    Args:
        chunk_id: Chunk ID (e.g., 'chunk_0000')
        data_dir: Base data directory
        output_dir: Output directory for corrected egomotion
        avdi: Dataset interface
        global_timestamps: Global timeline timestamps
    """
    # Get all video UUIDs in this chunk
    clip_index = avdi.clip_index
    videos_in_chunk = clip_index[clip_index['chunk'] == int(chunk_id.replace('chunk_', ''))].index.tolist()

    if not videos_in_chunk:
        print(f"Warning: No videos found in {chunk_id}")
        return

    print(f"\nProcessing {chunk_id} ({len(videos_in_chunk)} videos)...")

    # Create temporary directory to collect parquet files
    temp_dir = output_dir / f".temp_{chunk_id}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Process each video
    for video_uuid in tqdm(videos_in_chunk, desc=f"Processing {chunk_id}", leave=False):
        try:
            # Load egomotion data
            egomotion = avdi.get_clip_feature(
                video_uuid,
                avdi.features.LABELS.EGOMOTION,
                maybe_stream=False
            )

            # Interpolate at global timestamps
            ego_data = egomotion(global_timestamps)

            # Extract data
            translations = ego_data.pose.translation
            rotations = ego_data.pose.rotation.as_quat()
            velocities = ego_data.velocity
            accelerations = ego_data.acceleration
            curvatures = ego_data.curvature

            # Build dataframe with original column order
            data = {
                'timestamp': global_timestamps,
                'qx': rotations[:, 0],
                'qy': rotations[:, 1],
                'qz': rotations[:, 2],
                'qw': rotations[:, 3],
                'x': translations[:, 0],
                'y': translations[:, 1],
                'z': translations[:, 2],
                'vx': velocities[:, 0],
                'vy': velocities[:, 1],
                'vz': velocities[:, 2],
                'ax': accelerations[:, 0],
                'ay': accelerations[:, 1],
                'az': accelerations[:, 2],
                'curvature': curvatures.flatten(),  # Shape is (N, 1), need to flatten to (N,)
            }

            df = pd.DataFrame(data)

            # Save to temp directory
            parquet_path = temp_dir / f"{video_uuid}.egomotion.parquet"
            df.to_parquet(parquet_path, index=False)

        except Exception as e:
            print(f"    Error processing {video_uuid}: {e}")
            continue

    # Create zip file
    zip_path = output_dir / f"egomotion.{chunk_id}.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for parquet_file in temp_dir.glob("*.parquet"):
            zf.write(parquet_file, arcname=parquet_file.name)

    # Clean up temp directory
    for parquet_file in temp_dir.glob("*.parquet"):
        parquet_file.unlink()
    temp_dir.rmdir()

    print(f"  ✓ Created {zip_path.name}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Extract egomotion with global timestamps',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--chunks',
        nargs='+',
        default=None,
        help='Specific chunk IDs to process (e.g., chunk_0000)'
    )
    args = parser.parse_args()

    # Load environment
    script_dir = Path(__file__).parent.parent
    env_path = script_dir / ".env"
    env_vars = load_env(env_path)

    data_dir = Path(env_vars.get('PHYSICAL_AI_AV_DATA_DIR', '/home/xingao/data/PhysicalAI-Autonomous-Vehicles-base-wo-lidar-radar'))
    output_dir = data_dir / "labels" / "egomotion_corrected"

    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define global 10Hz timeline (same as extract_camera_frames_simple.py)
    video_duration_sec = 20.0
    sample_rate_hz = 10.0
    global_timestamps = np.arange(
        0,
        video_duration_sec * 1_000_000,
        int(1_000_000 / sample_rate_hz)
    ).astype(np.int64)

    print(f"\nGlobal timeline: {len(global_timestamps)} frames at {sample_rate_hz}Hz ({video_duration_sec}s)")
    print(f"Timestamp range: {global_timestamps[0]} to {global_timestamps[-1]} μs")

    # Initialize dataset interface
    avdi = physical_ai_av.OfflinePhysicalAIAVDatasetInterface(data_dir=str(data_dir))

    # Get available chunks
    all_chunk_ids = [f"chunk_{i:04d}" for i in sorted(avdi.clip_index['chunk'].unique())]

    if args.chunks:
        chunk_ids_to_process = [c for c in args.chunks if c in all_chunk_ids]
    else:
        chunk_ids_to_process = all_chunk_ids

    print(f"\nWill process {len(chunk_ids_to_process)} chunks:")
    for chunk_id in chunk_ids_to_process:
        print(f"  - {chunk_id}")

    # Process each chunk
    for chunk_id in chunk_ids_to_process:
        process_chunk(
            chunk_id=chunk_id,
            data_dir=data_dir,
            output_dir=output_dir,
            avdi=avdi,
            global_timestamps=global_timestamps
        )

    print("\n" + "="*80)
    print("✓ Egomotion extraction complete!")
    print(f"Output saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
