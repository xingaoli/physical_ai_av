#!/usr/bin/env python3
"""
Extract video frames from PhysicalAI AV Dataset using a global 10Hz timeline.

All cameras use the same global timestamps to extract frames, ensuring alignment.
The global timeline is 0-20 seconds at 10Hz (200 frames total).

Usage:
    # Extract all cameras for chunk_0000
    python3 tools/extract_camera_frames.py --chunks chunk_0000

    # Extract specific cameras for chunk_0000
    python3 tools/extract_camera_frames.py --chunks chunk_0000 --cameras camera_front_wide_120fov camera_front_tele_30fov

    # Extract multiple chunks
    python3 tools/extract_camera_frames.py --chunks chunk_0000 chunk_0004 chunk_0008

    # Extract all chunks (if --chunks is omitted, processes all chunks)
    python3 tools/extract_camera_frames.py

    # Extract specific cameras from all chunks
    python3 tools/extract_camera_frames.py --cameras camera_front_wide_120fov camera_rear_left_70fov

    # Extract single camera from single chunk
    python3 tools/extract_camera_frames.py --chunks chunk_0000 --cameras camera_front_wide_120fov
"""

import os
import argparse
from pathlib import Path
from typing import List

import cv2
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


def check_video_already_processed(
    video_uuid: str,
    camera_name: str,
    chunk_id: str,
    output_base_dir: Path,
    expected_min_frames: int
) -> bool:
    """
    Check if a video has already been processed.

    A video is considered processed if:
    1. The images directory exists
    2. The number of extracted frames >= expected_min_frames
    3. The timestamps.parquet file exists

    Args:
        video_uuid: UUID identifying the video
        camera_name: Name of the camera
        chunk_id: Which chunk this video belongs to
        output_base_dir: Base output directory
        expected_min_frames: Minimum expected number of frames

    Returns:
        True if video is already processed, False otherwise
    """
    video_output_dir = output_base_dir / camera_name / f"{camera_name}.{chunk_id}" / video_uuid
    images_dir = video_output_dir / "images"
    timestamps_path = video_output_dir / "anno" / "timestamps.parquet"

    # Check if directories and files exist
    if not images_dir.exists():
        return False
    if not timestamps_path.exists():
        return False

    # Count extracted frames
    extracted_files = list(images_dir.glob('frame_*.jpg'))
    if len(extracted_files) < expected_min_frames:
        return False

    return True


def process_video_all_cameras(
    video_uuid: str,
    chunk_id: str,
    data_dir: Path,
    output_base_dir: Path,
    cameras_to_process: List[str],
    global_timestamps: np.ndarray,
    avdi: physical_ai_av.PhysicalAIAVDatasetInterface,
    skip_existing: bool = True,
    expected_min_frames: int = 200  # Expect at least 15 seconds at 10Hz
):
    """
    Process a single video across all cameras using global timestamps.

    Args:
        video_uuid: UUID identifying the video
        chunk_id: Which chunk this video belongs to
        data_dir: Base data directory
        output_base_dir: Base output directory
        cameras_to_process: List of camera names to process
        global_timestamps: Global 10Hz timeline in microseconds
        avdi: PhysicalAIAVDatasetInterface instance
        skip_existing: If True, skip videos that have already been processed
        expected_min_frames: Minimum expected number of frames for skip check
    """
    for camera_name in cameras_to_process:
        # Check if already processed
        if skip_existing:
            if check_video_already_processed(video_uuid, camera_name, chunk_id, output_base_dir, expected_min_frames):
                continue  # Already processed, skip

        # Create output directories
        video_output_dir = output_base_dir / camera_name / f"{camera_name}.{chunk_id}" / video_uuid
        images_dir = video_output_dir / "images"
        anno_dir = video_output_dir / "anno"

        images_dir.mkdir(parents=True, exist_ok=True)
        anno_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Get camera object
            attr_name = camera_name.upper()
            camera = avdi.get_clip_feature(video_uuid, getattr(avdi.features.CAMERA, attr_name))

            # Decode frames at global timestamps
            # NOTE: No filtering! If timestamps are out of range, API will raise ValueError
            # This ensures all cameras stay aligned with the same global timeline
            frames, actual_timestamps = camera.decode_images_from_timestamps(global_timestamps)

            # Save frames (convert RGB to BGR for cv2.imwrite)
            for i, frame in enumerate(frames):
                frame_filename = images_dir / f"frame_{i:06d}.jpg"
                # Official API returns RGB, but cv2.imwrite expects BGR
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(frame_filename), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])

            # Save actual timestamps
            timestamps_df = pd.DataFrame({'timestamp': actual_timestamps})
            timestamps_df.index.name = 'frame_index'
            timestamps_df.to_parquet(anno_dir / "timestamps.parquet")

            # Save blurred_boxes (we can save empty for now since we don't have the original data)
            # The API doesn't provide blurred_boxes, so we skip it

        except Exception as e:
            print(f"    Error processing {video_uuid}/{camera_name}: {e}")
            continue


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Extract video frames using global 10Hz timeline',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--chunks',
        nargs='+',
        default=None,
        help='Specific chunk IDs to process'
    )
    parser.add_argument(
        '--cameras',
        nargs='+',
        default=None,
        help='Specific camera names to process'
    )
    args = parser.parse_args()

    # Load environment variables
    script_dir = Path(__file__).parent.parent
    env_path = script_dir / ".env"
    env_vars = load_env(env_path)

    data_dir = Path(env_vars.get('PHYSICAL_AI_AV_DATA_DIR', '/home/xingao/data/PhysicalAI-Autonomous-Vehicles-base-wo-lidar-radar'))
    output_dir = data_dir / "samples"

    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")

    # Define global 10Hz timeline: 0-20 seconds (200 frames at 10Hz)
    # Values in microseconds
    video_duration_sec = 20.0
    sample_rate_hz = 10.0
    global_timestamps = np.arange(0, video_duration_sec * 1_000_000, int(1_000_000 / sample_rate_hz)).astype(np.int64)

    print(f"Global timeline: {len(global_timestamps)} frames at {sample_rate_hz}Hz ({video_duration_sec}s)")
    print(f"Timestamp range: {global_timestamps[0]} to {global_timestamps[-1]} μs")

    # Initialize dataset interface
    avdi = physical_ai_av.OfflinePhysicalAIAVDatasetInterface(data_dir=str(data_dir))

    # Get list of camera directories (filter out hidden directories)
    camera_dir = data_dir / "camera"
    camera_names = sorted([d.name for d in camera_dir.iterdir() if d.is_dir() and not d.name.startswith('.')])

    # Filter cameras if specified
    if args.cameras:
        camera_names = [c for c in camera_names if c in args.cameras]
        if not camera_names:
            print(f"Error: No matching cameras found for: {args.cameras}")
            return

    print(f"\nFound {len(camera_names)} cameras to process:")
    for cam in camera_names:
        print(f"  - {cam}")

    # Get chunks to process (using first camera as reference)
    ref_camera = camera_names[0]
    ref_camera_path = camera_dir / ref_camera
    chunk_files = sorted([f.name for f in ref_camera_path.glob("*.zip")])
    all_chunk_ids = [f.replace(f"{ref_camera}.", "").replace(".zip", "") for f in chunk_files]

    if args.chunks:
        chunk_ids_to_process = [cid for cid in all_chunk_ids if cid in args.chunks]
    else:
        chunk_ids_to_process = all_chunk_ids

    print(f"\nWill process {len(chunk_ids_to_process)} chunks:")
    for chunk_id in chunk_ids_to_process:
        print(f"  - {chunk_id}")

    # Process each chunk
    for chunk_id in chunk_ids_to_process:
        print(f"\n{'='*80}")
        print(f"Processing {chunk_id}")
        print(f"{'='*80}")

        # Get all video UUIDs in this chunk
        import zipfile
        ref_chunk_path = data_dir / "camera" / ref_camera / f"{ref_camera}.{chunk_id}.zip"
        ref_uuids = set()

        with zipfile.ZipFile(ref_chunk_path, 'r') as zf:
            video_files = [f for f in zf.namelist() if f.endswith('.mp4')]
            for vf in video_files:
                parts = Path(vf).stem.split('.')
                if len(parts) >= 1:
                    ref_uuids.add(parts[0])

        ref_uuids = sorted(ref_uuids)
        print(f"\nFound {len(ref_uuids)} videos")

        # Process each video across all cameras
        for uuid in tqdm(ref_uuids, desc=f"Processing {chunk_id}", leave=True):
            process_video_all_cameras(
                video_uuid=uuid,
                chunk_id=chunk_id,
                data_dir=data_dir,
                output_base_dir=output_dir,
                cameras_to_process=camera_names,
                global_timestamps=global_timestamps,
                avdi=avdi
            )

    print("\n" + "="*80)
    print("✓ Video frame extraction complete!")
    print(f"Output saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
