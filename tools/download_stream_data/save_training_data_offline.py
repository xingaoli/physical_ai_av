# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cache training data from local offline chunks 0-49 into pickle files.

Each clip gets 8 timestamps (5s, 6s, ..., 12s), one .pkl per chunk.

Usage:
    python tools/download_stream_data/save_training_data_offline.py              # Full run (chunks 0-49)
    python tools/download_stream_data/save_training_data_offline.py --debug      # Debug: chunk 0, first 3 clips only

Env:
    ALPAMAYO_DATA_DIR   : data directory (default: ~/code/Alpamayo/data/PhysicalAI-Autonomous-Vehicles)

Output:
    output_offline/chunk_XXXX/{clip_id}.pkl
        Structure: dict[t0_us, {"data": {...}}]
        data keys: image_frames, camera_indices, ego_history_xyz, ego_history_rot,
                   ego_future_xyz, ego_future_rot, relative_timestamps, absolute_timestamps,
                   t0_us, clip_id, extr, intr, vehicle_dimensions
"""

import argparse
import io
import os
import pickle
import zipfile
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
import scipy.spatial.transform as spt
import torch
from einops import rearrange
from tqdm import tqdm

import physical_ai_av.video as video
import physical_ai_av.egomotion as egomotion_module
from physical_ai_av.calibration import CameraIntrinsics, SensorExtrinsics, VehicleDimensions


def load_physical_aiavdataset_local(
    clip_id: str,
    data_dir: str | None = None,
    t0_us: int = 5_100_000,
    num_history_steps: int = 16,
    num_future_steps: int = 64,
    time_step: float = 0.1,
    camera_features: list | None = None,
    num_frames: int = 4,
    extrinsics_df: pd.DataFrame | None = None,
    intrinsics_df: pd.DataFrame | None = None,
    dimensions_series: pd.Series | None = None,
) -> dict[str, Any]:
    if data_dir is None:
        data_dir = os.environ.get(
            "ALPAMAYO_DATA_DIR",
            "/home/xingao/code/Alpamayo1.5/data/PhysicalAI-Autonomous-Vehicles",
        )

    clip_index = pd.read_parquet(os.path.join(data_dir, "clip_index.parquet"))
    chunk_id = clip_index.loc[clip_id, "chunk"]

    if camera_features is None:
        camera_features = [
            "camera_cross_left_120fov",
            "camera_front_wide_120fov",
            "camera_cross_right_120fov",
            "camera_front_tele_30fov",
        ]

    camera_name_to_index = {
        "camera_cross_left_120fov": 0,
        "camera_front_wide_120fov": 1,
        "camera_cross_right_120fov": 2,
        "camera_rear_left_70fov": 3,
        "camera_rear_tele_30fov": 4,
        "camera_rear_right_70fov": 5,
        "camera_front_tele_30fov": 6,
    }

    # Load egomotion
    egomotion_zip_path = os.path.join(
        data_dir, "labels", "egomotion", f"egomotion.chunk_{chunk_id:04d}.zip"
    )
    with zipfile.ZipFile(egomotion_zip_path, "r") as zf:
        egomotion_df = pd.read_parquet(io.BytesIO(zf.read(f"{clip_id}.egomotion.parquet")))

    assert t0_us > num_history_steps * time_step * 1_000_000, (
        "t0_us must be greater than the history time range"
    )

    history_offsets_us = np.arange(
        -(num_history_steps - 1) * time_step * 1_000_000,
        time_step * 1_000_000 / 2,
        time_step * 1_000_000,
    ).astype(np.int64)
    history_timestamps = t0_us + history_offsets_us

    future_offsets_us = np.arange(
        time_step * 1_000_000,
        (num_future_steps + 0.5) * time_step * 1_000_000,
        time_step * 1_000_000,
    ).astype(np.int64)
    future_timestamps = t0_us + future_offsets_us

    egomotion_state = egomotion_module.EgomotionState.from_egomotion_df(egomotion_df)
    egomotion = egomotion_state.create_interpolator(egomotion_df["timestamp"].values)

    ego_history = egomotion(history_timestamps)
    ego_history_xyz = ego_history.pose.translation
    ego_history_quat = ego_history.pose.rotation.as_quat()

    ego_future = egomotion(future_timestamps)
    ego_future_xyz = ego_future.pose.translation
    ego_future_quat = ego_future.pose.rotation.as_quat()

    t0_xyz = ego_history_xyz[-1].copy()
    t0_quat = ego_history_quat[-1].copy()
    t0_rot = spt.Rotation.from_quat(t0_quat)
    t0_rot_inv = t0_rot.inv()

    ego_history_xyz_local = t0_rot_inv.apply(ego_history_xyz - t0_xyz)
    ego_future_xyz_local = t0_rot_inv.apply(ego_future_xyz - t0_xyz)

    ego_history_rot_local = (t0_rot_inv * spt.Rotation.from_quat(ego_history_quat)).as_matrix()
    ego_future_rot_local = (t0_rot_inv * spt.Rotation.from_quat(ego_future_quat)).as_matrix()

    ego_history_xyz_tensor = torch.from_numpy(ego_history_xyz_local).float().unsqueeze(0).unsqueeze(0)
    ego_history_rot_tensor = torch.from_numpy(ego_history_rot_local).float().unsqueeze(0).unsqueeze(0)
    ego_future_xyz_tensor = torch.from_numpy(ego_future_xyz_local).float().unsqueeze(0).unsqueeze(0)
    ego_future_rot_tensor = torch.from_numpy(ego_future_rot_local).float().unsqueeze(0).unsqueeze(0)

    # Load camera images
    image_frames_list = []
    camera_indices_list = []
    timestamps_list = []

    image_timestamps = np.array(
        [t0_us - (num_frames - 1 - i) * int(time_step * 1_000_000) for i in range(num_frames)],
        dtype=np.int64,
    )

    for cam_feature in camera_features:
        camera_zip_path = os.path.join(
            data_dir, "camera", cam_feature, f"{cam_feature}.chunk_{chunk_id:04d}.zip"
        )

        with zipfile.ZipFile(camera_zip_path, "r") as zf:
            video_data = io.BytesIO(zf.read(f"{clip_id}.{cam_feature}.mp4"))
            frame_timestamps_df = pd.read_parquet(
                io.BytesIO(zf.read(f"{clip_id}.{cam_feature}.timestamps.parquet"))
            )
            frame_timestamps = frame_timestamps_df["timestamp"].values

            reader = video.SeekVideoReader(
                video_data=video_data,
                timestamps=frame_timestamps,
            )
            frames, _ = reader.decode_images_from_timestamps(image_timestamps)

        frames_tensor = torch.from_numpy(frames)
        frames_tensor = rearrange(frames_tensor, "t h w c -> t c h w")

        cam_idx = camera_name_to_index.get(cam_feature, 0)
        image_frames_list.append(frames_tensor)
        camera_indices_list.append(cam_idx)
        timestamps_list.append(torch.from_numpy(image_timestamps.astype(np.int64)))

    image_frames = torch.stack(image_frames_list, dim=0)
    camera_indices = torch.tensor(camera_indices_list, dtype=torch.int64)
    all_timestamps = torch.stack(timestamps_list, dim=0)

    sort_order = torch.argsort(camera_indices)
    image_frames = image_frames[sort_order]
    camera_indices = camera_indices[sort_order]
    all_timestamps = all_timestamps[sort_order]

    camera_tmin = all_timestamps.min()
    relative_timestamps = (all_timestamps - camera_tmin).float() * 1e-6

    result: dict[str, Any] = {
        "image_frames": image_frames,
        "camera_indices": camera_indices,
        "ego_history_xyz": ego_history_xyz_tensor,
        "ego_history_rot": ego_history_rot_tensor,
        "ego_future_xyz": ego_future_xyz_tensor,
        "ego_future_rot": ego_future_rot_tensor,
        "relative_timestamps": relative_timestamps,
        "absolute_timestamps": all_timestamps,
        "t0_us": t0_us,
        "clip_id": clip_id,
    }

    if extrinsics_df is not None:
        result["extr"] = SensorExtrinsics.from_extrinsics_df(extrinsics_df.loc[clip_id])

    if intrinsics_df is not None:
        result["intr"] = CameraIntrinsics.from_intrinsics_df(intrinsics_df.loc[clip_id])

    if dimensions_series is not None:
        result["vehicle_dimensions"] = VehicleDimensions(
            length=dimensions_series["length"],
            width=dimensions_series["width"],
            height=dimensions_series["height"],
            rear_axle_to_bbox_center=dimensions_series["rear_axle_to_bbox_center"],
            wheelbase=dimensions_series["wheelbase"],
            track_width=dimensions_series["track_width"],
        )

    return result


def _load_chunk_calibration(data_dir: str, chunk_id: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def _path(name):
        return os.path.join(data_dir, "calibration", name, f"{name}.chunk_{chunk_id:04d}.parquet")

    extr_df = pd.read_parquet(_path("sensor_extrinsics"))
    intr_df = pd.read_parquet(_path("camera_intrinsics"))
    dim_df = pd.read_parquet(_path("vehicle_dimensions"))
    return extr_df, intr_df, dim_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Only process chunk 0, first 3 clips")
    args = parser.parse_args()

    data_dir = os.environ.get(
        "ALPAMAYO_DATA_DIR",
        "/home/xingao/code/Alpamayo/data/PhysicalAI-Autonomous-Vehicles",
    )
    # output_dir = os.path.join(os.path.dirname(__file__), "output_offline")
    output_dir = "/mnt/hdd_data/public_data/PhysicalAI-Autonomous-Vehicles-cache"
    os.makedirs(output_dir, exist_ok=True)

    clip_index = pd.read_parquet(os.path.join(data_dir, "clip_index.parquet"))

    chunk_ids = list(range(50))  # 0-49

    t0_list_us = [5_000_000, 6_000_000, 7_000_000, 8_000_000,
                  9_000_000, 10_000_000, 11_000_000, 12_000_000]

    if args.debug:
        chunk_ids = [0]
        print("[debug] Only processing chunk 0")

    for chunk_id in tqdm(chunk_ids, desc="Processing chunks"):
        chunk_clips = clip_index[clip_index["chunk"] == chunk_id]
        clip_ids = chunk_clips.index.tolist()

        if args.debug:
            clip_ids = clip_ids[:3]
            print(f"[debug] Limiting to {len(clip_ids)} clips")

        chunk_dir = os.path.join(output_dir, f"chunk_{chunk_id:04d}")
        os.makedirs(chunk_dir, exist_ok=True)

        # Load existing per-clip data to support resume
        clip_existing: dict[str, dict[int, dict[str, Any]]] = {}
        for clip_id in clip_ids:
            clip_path = os.path.join(chunk_dir, f"{clip_id}.pkl")
            if os.path.exists(clip_path):
                with open(clip_path, "rb") as f:
                    clip_existing[clip_id] = pickle.load(f)

        pending: list[tuple[str, int]] = []
        for clip_id in clip_ids:
            existing = clip_existing.get(clip_id, {})
            for t0_us in t0_list_us:
                if t0_us in existing:
                    continue
                pending.append((clip_id, t0_us))

        if not pending:
            tqdm.write(f"  Chunk {chunk_id}: all done, skipping")
            continue

        extrinsics_df, intrinsics_df, dimensions_df = _load_chunk_calibration(data_dir, chunk_id)

        round_num = 0
        while pending:
            round_num += 1
            failed: list[tuple[str, int]] = []
            new_results: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)

            for clip_id, t0_us in tqdm(
                pending, desc=f"  Round {round_num} ({len(pending)} tasks)", leave=False
            ):
                try:
                    dim_series = dimensions_df.loc[clip_id] if clip_id in dimensions_df.index else None
                    data = load_physical_aiavdataset_local(
                        clip_id=clip_id,
                        data_dir=data_dir,
                        t0_us=t0_us,
                        extrinsics_df=extrinsics_df,
                        intrinsics_df=intrinsics_df,
                        dimensions_series=dim_series,
                    )
                    new_results[clip_id][t0_us] = {"data": data}
                except Exception as e:
                    tqdm.write(f"Error clip={clip_id[:8]}, t0={t0_us}: {e}")
                    failed.append((clip_id, t0_us))

            for clip_id, t0_dict in new_results.items():
                existing = clip_existing.setdefault(clip_id, {})
                existing.update(t0_dict)
                clip_path = os.path.join(chunk_dir, f"{clip_id}.pkl")
                with open(clip_path, "wb") as f:
                    pickle.dump(existing, f)

            if not failed:
                break

            tqdm.write(
                f"  Round {round_num}: {len(pending) - len(failed)} ok, "
                f"{len(failed)} failed. Retrying..."
            )
            pending = failed

        saved = sum(1 for cid in clip_ids if os.path.exists(os.path.join(chunk_dir, f"{cid}.pkl")))
        print(f"  Chunk {chunk_id}: saved {saved} clips to {chunk_dir}/")


if __name__ == "__main__":
    main()
