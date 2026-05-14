#!/usr/bin/env python3
"""
Generate Chain of Causation (CoC) labels for driving scenarios using VLM.

Usage:
    # Dry run (no VLM call, print only)
    python auto_labeling/label_coc/generate_coc.py --dry-run

    # Print CoC to stdout (no visualization, no file)
    python auto_labeling/label_coc/generate_coc.py --no-vis

    # Save results to JSON
    python auto_labeling/label_coc/generate_coc.py --no-vis --output-json results.json

    # With custom prompt file
    python auto_labeling/label_coc/generate_coc.py --input-json auto_labeling/label_coc/prompt/construction_zone.json --output-json auto_labeling/label_coc/output/construction_zone/constrcution_zone_coc.json --output-dir auto_labeling/label_coc/output/construction_zone/
"""

import os
import json
import zipfile
import io
import base64
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import scipy.spatial.transform as spt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from openai import OpenAI

import physical_ai_av.egomotion as egomotion_module
import physical_ai_av.video as video


# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------

def _get_chunk_number(data_dir: str, clip_id: str) -> int:
    clip_index = pd.read_parquet(os.path.join(data_dir, "clip_index.parquet"))
    return clip_index.loc[clip_id, "chunk"]


def extract_future_trajectory(
    data_dir: str,
    clip_id: str,
    t0_us: int,
    duration_s: float = 6.0,
    time_step: float = 0.5,
) -> np.ndarray:
    """Extract ego future trajectory from t0 to t0+duration_s, normalized to t0 BEV frame."""
    chunk_id = _get_chunk_number(data_dir, clip_id)

    egomotion_zip_path = os.path.join(
        data_dir, "labels", "egomotion", f"egomotion.chunk_{chunk_id:04d}.zip"
    )
    with zipfile.ZipFile(egomotion_zip_path, "r") as zf:
        egomotion_df = pd.read_parquet(
            io.BytesIO(zf.read(f"{clip_id}.egomotion.parquet"))
        )

    egomotion_state = egomotion_module.EgomotionState.from_egomotion_df(egomotion_df)
    egomotion = egomotion_state.create_interpolator(egomotion_df["timestamp"].values)

    num_points = int(duration_s / time_step) + 1
    offsets_us = (np.arange(num_points) * time_step * 1_000_000).astype(np.int64)
    timestamps = t0_us + offsets_us

    ego = egomotion(timestamps)
    ego_xyz = ego.pose.translation
    t0_xyz = ego_xyz[0].copy()
    t0_quat = ego.pose.rotation.as_quat()[0]
    t0_rot_inv = spt.Rotation.from_quat(t0_quat).inv()

    traj_local = t0_rot_inv.apply(ego_xyz - t0_xyz)
    return traj_local


def get_frame_at_timestamp(
    data_dir: str,
    clip_id: str,
    timestamp_us: int,
    camera_feature: str = "camera_front_wide_120fov",
) -> np.ndarray:
    """Decode the frame at the given timestamp, return RGB numpy array."""
    chunk_number = _get_chunk_number(data_dir, clip_id)

    camera_zip_path = os.path.join(
        data_dir, "camera", camera_feature,
        f"{camera_feature}.chunk_{chunk_number:04d}.zip"
    )
    if not os.path.exists(camera_zip_path):
        raise FileNotFoundError(f"Camera zip not found: {camera_zip_path}")

    with zipfile.ZipFile(camera_zip_path, "r") as zf:
        video_data = io.BytesIO(zf.read(f"{clip_id}.{camera_feature}.mp4"))
        timestamps_df = pd.read_parquet(
            io.BytesIO(zf.read(f"{clip_id}.{camera_feature}.timestamps.parquet"))
        )
        frame_timestamps = timestamps_df["timestamp"].values

        reader = video.SeekVideoReader(
            video_data=video_data,
            timestamps=frame_timestamps,
        )
        frames, _ = reader.decode_images_from_timestamps(
            np.array([timestamp_us], dtype=np.int64)
        )
        if frames.shape[0] == 0:
            raise ValueError(f"No frame decoded for timestamp {timestamp_us} us")

        return frames[0]


def encode_frame_to_base64(frame_rgb: np.ndarray) -> str:
    """Encode an RGB frame to base64 JPEG string."""
    image_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode("utf-8")


def get_future_meta_actions(
    data_dir: str,
    clip_id: str,
    t0_us: int,
    duration_s: float = 6.0,
    time_step: float = 0.5,
) -> list[tuple[str, str]]:
    """Get longitudinal and lateral meta-action labels for the next `duration_s` from t0."""
    chunk_number = _get_chunk_number(data_dir, clip_id)

    meta_path = os.path.join(
        data_dir, "labels", "meta_actions",
        f"meta_actions.chunk_{chunk_number:04d}",
        f"{clip_id}.meta_actions.json",
    )
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    smooth = meta["smooth_data"]
    long_actions = smooth["long_action"]
    lat_actions = smooth["lat_action"]

    start_idx = int(t0_us // 100_000)
    num_points = int(duration_s / time_step) + 1

    result = []
    for i in range(num_points):
        idx = start_idx + int(i * time_step * 10)
        if idx >= len(long_actions):
            break
        result.append((long_actions[idx], lat_actions[idx]))
    return result


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_trajs(traj: np.ndarray) -> str:
    pairs = [f"({p[0]:.2f}, {p[1]:.2f})" for p in traj]
    return f"[{', '.join(pairs)}]"


def format_meta_actions(actions: list[tuple[str, str]]) -> str:
    pairs = [f"({long}, {lat})" for long, lat in actions]
    return f"[{', '.join(pairs)}]"


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def save_visualization(
    output_path: str,
    frame_rgb: np.ndarray,
    traj: np.ndarray,
    actions: list[tuple[str, str]],
    coc: str,
    clip_id: str,
    timestamp_us: int,
    time_step: float = 0.5,
):
    """Save a composite visualization: camera image, BEV trajectory, meta-actions, CoC."""
    fig = plt.figure(figsize=(22, 12))
    gs = fig.add_gridspec(
        2, 2, height_ratios=[5, 2.5], width_ratios=[3, 1],
        hspace=0.30, wspace=0.20,
    )

    # --- Camera image ---
    ax_cam = fig.add_subplot(gs[0, 0])
    ax_cam.imshow(frame_rgb)
    ax_cam.set_title(
        f"{clip_id[:8]}...  t = {timestamp_us / 1e6:.1f}s", fontsize=13, loc="left"
    )
    ax_cam.axis("off")

    # --- BEV trajectory (rotate_90cc: plot_x=-y, plot_y=x) ---
    ax_bev = fig.add_subplot(gs[0, 1])
    xs, ys = traj[:, 0], traj[:, 1]
    px, py = -ys, xs
    n = len(xs)
    ax_bev.plot(0, 0, "rs", markersize=14, zorder=5, label="Ego (t=0)")
    for i in range(n - 1):
        c = plt.cm.winter(i / max(n - 1, 1))
        ax_bev.plot(px[i : i + 2], py[i : i + 2], "-o", color=c, markersize=4, linewidth=1.5)
    ax_bev.plot(px[-1], py[-1], "g^", markersize=10, zorder=5, label="t=6.0s")
    ax_bev.set_xlabel("x  [m]")
    ax_bev.set_ylabel("y  forward [m]")
    ax_bev.set_title("BEV Trajectory", fontsize=13, loc="left")
    x_margin = 3.0
    ax_bev.set_xlim(-x_margin, x_margin)
    ax_bev.set_ylim(-0.5, py.max() * 1.1 + 1)
    ax_bev.grid(True, alpha=0.3)
    ax_bev.legend(fontsize=10, loc="upper left")

    # --- Meta-actions (2-column) ---
    ax_meta = fig.add_subplot(gs[1, 0])
    ax_meta.axis("off")
    ax_meta.text(0.01, 0.97, "Meta-Actions (2 Hz):", fontsize=11, fontweight="bold",
                 transform=ax_meta.transAxes, va="top")

    col_size = (len(actions) + 1) // 2
    left_lines, right_lines = [], []
    for i, (long, lat) in enumerate(actions):
        t = i * time_step
        line = f"t={t:4.1f}s   {long:<25s}| {lat}"
        (left_lines if i < col_size else right_lines).append(line)

    ax_meta.text(0.01, 0.82, "\n".join(left_lines), fontsize=9, fontfamily="monospace",
                 transform=ax_meta.transAxes, va="top")
    ax_meta.text(0.50, 0.82, "\n".join(right_lines), fontsize=9, fontfamily="monospace",
                 transform=ax_meta.transAxes, va="top")

    # --- CoC ---
    ax_coc = fig.add_subplot(gs[1, 1])
    ax_coc.axis("off")
    coc_text = coc if coc else "[No CoC generated]"
    ax_coc.text(0.05, 0.95, "CoC:", fontsize=12, fontweight="bold",
                transform=ax_coc.transAxes, va="top")
    lines, line = [], ""
    for word in coc_text.split(" "):
        if len(line) + len(word) + 1 > 40:
            lines.append(line)
            line = word
        else:
            line = (line + " " + word).strip()
    if line:
        lines.append(line)
    wrapped = "\n".join(lines)
    ax_coc.text(0.05, 0.75, wrapped, fontsize=11,
                transform=ax_coc.transAxes, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Visualization saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    script_dir = Path(__file__).parent
    default_data_dir = str(
        Path(__file__).parent.parent.parent / "data" / "PhysicalAI-Autonomous-Vehicles"
    )

    parser = argparse.ArgumentParser(
        description="Generate CoC labels using VLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Dry run only
  python generate_coc.py --dry-run

  # Print to stdout, no visualization
  python generate_coc.py --no-vis

  # Save to JSON + visualization
  python generate_coc.py --output-json results.json

  # Custom input + prompt auto-detected from stem
  python generate_coc.py --input-json heavy_rain.json --output-json output/heavy_rain_coc.json
""",
    )
    parser.add_argument("--data-dir", default=default_data_dir)
    parser.add_argument(
        "--input-json", default=str(script_dir / "up_down_hill.json"),
        help="Path to JSON file with clip entries",
    )
    parser.add_argument("--output-json", default=None,
                        help="Path to save results as JSON (same format as input, with coc field added)")
    parser.add_argument(
        "--prompt-file", default=None,
        help="Path to prompt text file. Default: <input_json stem>.txt in the same directory",
    )
    parser.add_argument("--output-dir", default=str(script_dir / "output"),
                        help="Directory to save visualization images")
    parser.add_argument("--no-vis", action="store_true",
                        help="Skip visualization")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--base-url", default="http://0.0.0.0:8080/v1")
    parser.add_argument("--model", default="ckpts/Qwen3.6-27B-int4-AutoRound")
    parser.add_argument("--dry-run", action="store_true", help="Skip VLM calls")
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--presence-penalty", type=float, default=1.5)
    parser.add_argument("--no-thinking", action="store_true", default=False,
                        help="Disable chain-of-thought reasoning")
    args = parser.parse_args()

    # Load prompt: explicit --prompt-file, or auto-detect from input stem
    if args.prompt_file:
        prompt_path = Path(args.prompt_file)
    else:
        input_stem = Path(args.input_json).stem
        prompt_path = Path(args.input_json).parent / f"{input_stem}.txt"
    if not prompt_path.exists():
        parser.error(f"Prompt file not found: {prompt_path} (--prompt-file or <input_stem>.txt)")
    prompt = prompt_path.read_text(encoding="utf-8")
    print(f"Loaded prompt from {prompt_path}")

    # Output directory for vis
    if not args.no_vis:
        os.makedirs(args.output_dir, exist_ok=True)

    with open(args.input_json, "r", encoding="utf-8") as f:
        entries = json.load(f)

    print(f"Loaded {len(entries)} entries from {args.input_json}")

    if not args.dry_run:
        client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    results = []
    for entry in entries:
        clip_id = entry["clip_id"]
        timestamp_us = entry["timestamp_us"]
        text_label = entry.get("text", "")

        print(f"\n{'='*60}")
        print(f"Clip: {clip_id[:8]}..., ts: {timestamp_us} us, label: {text_label}")

        frame_rgb = get_frame_at_timestamp(args.data_dir, clip_id, timestamp_us)
        image_b64 = encode_frame_to_base64(frame_rgb)
        traj = extract_future_trajectory(args.data_dir, clip_id, timestamp_us)
        actions = get_future_meta_actions(args.data_dir, clip_id, timestamp_us)

        trajs_str = format_trajs(traj)
        meta_str = format_meta_actions(actions)

        # print(f"  Trajectory: trajs:{trajs_str}")
        # print(f"  Meta-actions: meta_actions:{meta_str}")

        coc = ""
        if not args.dry_run:
            prompt_text = (
                prompt
                + f"\n\n# Current Input\n"
                + f"**Image**: [See attached image]\n"
                + f"**Trajectories**: `trajs:{trajs_str}`\n"
                + f"**Meta-Actions**: `meta_actions:{meta_str}`\n\n"
                + f"# Response\n"
            )

            try:
                response = client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_b64}"
                                    },
                                },
                                {"type": "text", "text": prompt_text},
                            ],
                        }
                    ],
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    presence_penalty=args.presence_penalty,
                    extra_body={
                        "top_k": 20,
                        "chat_template_kwargs": {"enable_thinking": not args.no_thinking},
                    },
                )
                coc = response.choices[0].message.content.strip()
                print(f"  CoC: {coc}")
                usage = response.usage
                print(f"  [tokens] prompt={usage.prompt_tokens}, completion={usage.completion_tokens}")
            except Exception as e:
                print(f"  VLM call failed: {e}")

        # Build result entry
        result = dict(entry)
        result["coc"] = coc
        results.append(result)

        if not args.no_vis:
            vis_path = os.path.join(
                args.output_dir, f"{clip_id[:8]}_{timestamp_us // 1000000}s.png"
            )
            save_visualization(vis_path, frame_rgb, traj, actions, coc, clip_id, timestamp_us)

    # Save JSON
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
