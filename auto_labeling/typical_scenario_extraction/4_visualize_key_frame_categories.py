#!/usr/bin/env python3
"""
Visualize grouped keyframe categories.

Outputs under labels/typical_scenario_key_frame/key_frame by default:
  - category_distribution_pie.png
  - one 4x3 contact sheet per category group

Usage:
    python auto_labeling/typical_scenario_extraction/4_visualize_key_frame_categories.py

    python auto_labeling/typical_scenario_extraction/4_visualize_key_frame_categories.py \
        --seed 7 --tile-width 768
"""

import argparse
import io
import json
import os
import random
import re
import zipfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps
from tqdm import tqdm

import physical_ai_av.video as video


ROWS = 4
COLS = 3
SAMPLES_PER_CATEGORY = ROWS * COLS
DEFAULT_CAMERA_FEATURE = "camera_front_wide_120fov"


def default_data_dir() -> Path:
    env_data_dir = os.getenv("ALPAMAYO_DATA_DIR")
    if env_data_dir:
        return Path(env_data_dir)

    repo_data_dir = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "PhysicalAI-Autonomous-Vehicles"
    )
    if repo_data_dir.exists():
        return repo_data_dir

    return Path("/home/xingao/code/Alpamayo1.5/data/PhysicalAI-Autonomous-Vehicles")


def category_sort_value(category: str) -> tuple[int, int | str]:
    try:
        return (0, int(category))
    except ValueError:
        return (1, category)


def group_sort_key(group_key: str) -> tuple[str, tuple[int, int | str]]:
    prompt_name, _, category = group_key.rpartition("_category_")
    return prompt_name, category_sort_value(category)


def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def parse_chunk_number(item: dict[str, Any]) -> int:
    chunk_number = item.get("chunk_number")
    if isinstance(chunk_number, int):
        return chunk_number

    chunk_id = item.get("chunk_id", "")
    if isinstance(chunk_id, str) and chunk_id.startswith("chunk_"):
        return int(chunk_id.replace("chunk_", "", 1))

    raise ValueError(f"Cannot infer chunk number from item: {item}")


def load_groups(path: Path) -> dict[str, list[dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        groups = json.load(f)
    if not isinstance(groups, dict):
        raise ValueError(f"Expected grouped category JSON object, got {type(groups).__name__}")

    ordered = {}
    for key in sorted(groups.keys(), key=group_sort_key):
        items = groups[key]
        if not isinstance(items, list):
            raise ValueError(f"Expected list for group {key}, got {type(items).__name__}")
        ordered[key] = items
    return ordered


def decode_frame_at_timestamp(
    data_dir: Path,
    clip_id: str,
    chunk_number: int,
    timestamp_us: int,
    camera_feature: str,
) -> np.ndarray:
    camera_zip_path = (
        data_dir
        / "camera"
        / camera_feature
        / f"{camera_feature}.chunk_{chunk_number:04d}.zip"
    )
    if not camera_zip_path.exists():
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
        target_timestamps = np.array([timestamp_us], dtype=np.int64)
        frames, _ = reader.decode_images_from_timestamps(target_timestamps)
        if frames.shape[0] == 0:
            raise ValueError(f"No frame decoded for timestamp {timestamp_us} us")
        return frames[0]


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    x1, y1, x2, y2 = box
    w, h = text_size(draw, text, font)
    draw.text((x1 + (x2 - x1 - w) / 2, y1 + (y2 - y1 - h) / 2),
              text, font=font, fill=fill)


def make_frame_area(
    item: dict[str, Any],
    data_dir: Path,
    camera_feature: str,
    tile_width: int,
    tile_height: int,
) -> Image.Image:
    chunk_number = parse_chunk_number(item)
    frame = decode_frame_at_timestamp(
        data_dir=data_dir,
        clip_id=item["clip_id"],
        chunk_number=chunk_number,
        timestamp_us=int(item["timestamp_us"]),
        camera_feature=camera_feature,
    )
    frame_img = Image.fromarray(frame.astype(np.uint8)).convert("RGB")
    resized = ImageOps.contain(frame_img, (tile_width, tile_height), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (tile_width, tile_height), (8, 8, 8))
    x = (tile_width - resized.width) // 2
    y = (tile_height - resized.height) // 2
    canvas.paste(resized, (x, y))
    return canvas


def make_placeholder(
    message: str,
    tile_width: int,
    tile_height: int,
    font: ImageFont.ImageFont,
) -> Image.Image:
    canvas = Image.new("RGB", (tile_width, tile_height), (55, 55, 55))
    draw = ImageDraw.Draw(canvas)
    draw_centered_text(
        draw,
        (20, 20, tile_width - 20, tile_height - 20),
        message[:160],
        font,
        (255, 230, 230),
    )
    return canvas


def make_tile(
    item: dict[str, Any],
    frame_cache: dict[tuple[str, int, int], Image.Image],
    data_dir: Path,
    camera_feature: str,
    tile_width: int,
    tile_height: int,
    label_height: int,
    small_font: ImageFont.ImageFont,
) -> Image.Image:
    tile = Image.new("RGB", (tile_width, tile_height + label_height), (255, 255, 255))
    draw = ImageDraw.Draw(tile)
    draw.rectangle([0, 0, tile_width, label_height], fill=(24, 24, 24))

    timestamp_s = int(item.get("timestamp_us", 0)) / 1_000_000
    label = (
        f"{str(item.get('clip_id', ''))[:8]}  "
        f"kf={item.get('keyframe_index')}  "
        f"t={timestamp_s:.1f}s"
    )
    draw.text((12, 10), label, font=small_font, fill=(245, 245, 245))

    try:
        cache_key = (
            item["clip_id"],
            parse_chunk_number(item),
            int(item["timestamp_us"]),
        )
        if cache_key not in frame_cache:
            frame_cache[cache_key] = make_frame_area(
                item, data_dir, camera_feature, tile_width, tile_height)
        frame_area = frame_cache[cache_key]
    except Exception as exc:
        frame_area = make_placeholder(
            f"decode failed: {exc}", tile_width, tile_height, small_font)

    tile.paste(frame_area, (0, label_height))
    draw.rectangle([0, 0, tile_width - 1, tile_height + label_height - 1],
                   outline=(220, 220, 220), width=2)
    return tile


def select_samples(
    items: list[dict[str, Any]],
    n: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if not items:
        return []
    if len(items) >= n:
        return rng.sample(items, n)

    selected = list(items)
    while len(selected) < n:
        selected.append(rng.choice(items))
    rng.shuffle(selected)
    return selected


def save_contact_sheet(
    group_key: str,
    items: list[dict[str, Any]],
    output_path: Path,
    data_dir: Path,
    camera_feature: str,
    tile_width: int,
    tile_height: int,
    rng: random.Random,
) -> None:
    title_height = 76
    label_height = 42
    sheet_width = COLS * tile_width
    sheet_height = title_height + ROWS * (tile_height + label_height)

    title_font = load_font(32, bold=True)
    small_font = load_font(20)

    sheet = Image.new("RGB", (sheet_width, sheet_height), (250, 250, 250))
    draw = ImageDraw.Draw(sheet)
    title = f"{group_key}  |  {len(items)} samples"
    draw_centered_text(draw, (0, 0, sheet_width, title_height),
                       title, title_font, (20, 20, 20))

    samples = select_samples(items, SAMPLES_PER_CATEGORY, rng)
    frame_cache: dict[tuple[str, int, int], Image.Image] = {}
    for idx, item in enumerate(samples):
        row = idx // COLS
        col = idx % COLS
        tile = make_tile(
            item=item,
            frame_cache=frame_cache,
            data_dir=data_dir,
            camera_feature=camera_feature,
            tile_width=tile_width,
            tile_height=tile_height,
            label_height=label_height,
            small_font=small_font,
        )
        x = col * tile_width
        y = title_height + row * (tile_height + label_height)
        sheet.paste(tile, (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=95)


def save_pie_chart(groups: dict[str, list[dict[str, Any]]], output_path: Path) -> None:
    labels = list(groups.keys())
    counts = [len(groups[key]) for key in labels]
    total = sum(counts)

    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(16, 10), dpi=180)

    def autopct(pct: float) -> str:
        return f"{pct:.1f}%" if pct >= 1.5 else ""

    wedges, _, autotexts = ax.pie(
        counts,
        labels=None,
        startangle=90,
        counterclock=False,
        autopct=autopct,
        pctdistance=0.75,
        colors=colors,
        wedgeprops={"linewidth": 0.8, "edgecolor": "white"},
        textprops={"fontsize": 9, "color": "black"},
    )
    for text in autotexts:
        text.set_fontweight("bold")

    legend_labels = [f"{label}: {count}" for label, count in zip(labels, counts)]
    ax.legend(
        wedges,
        legend_labels,
        title="Category counts",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=9,
        title_fontsize=10,
        frameon=False,
    )
    ax.set_title(f"Typical Scenario Keyframe Categories ({total} labels)",
                 fontsize=16, fontweight="bold")
    ax.axis("equal")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    data_dir = default_data_dir()
    default_input = (
        data_dir
        / "labels"
        / "typical_scenario_key_frame"
        / "typical_scenario_key_frame_category.json"
    )
    default_output_dir = (
        data_dir
        / "labels"
        / "typical_scenario_key_frame"
        / "key_frame"
    )

    parser = argparse.ArgumentParser(
        description="Visualize keyframe category distribution and examples."
    )
    parser.add_argument("--input", type=Path, default=default_input,
                        help="Input grouped category JSON path")
    parser.add_argument("--output-dir", type=Path, default=default_output_dir,
                        help="Directory for pie chart and contact sheets")
    parser.add_argument("--data-dir", type=Path, default=data_dir,
                        help="PhysicalAI-AV data directory")
    parser.add_argument("--camera-feature", type=str, default=DEFAULT_CAMERA_FEATURE)
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling")
    parser.add_argument("--tile-width", type=int, default=640,
                        help="Width of each sampled frame in the contact sheet")
    parser.add_argument("--tile-height", type=int, default=360,
                        help="Height of each sampled frame in the contact sheet")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    groups = load_groups(args.input)
    rng = random.Random(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    pie_path = args.output_dir / "category_distribution_pie.png"
    save_pie_chart(groups, pie_path)
    print(f"Saved pie chart: {pie_path}")

    for group_key, items in tqdm(groups.items(), desc="Rendering category grids"):
        output_path = args.output_dir / f"{safe_filename(group_key)}_grid.png"
        save_contact_sheet(
            group_key=group_key,
            items=items,
            output_path=output_path,
            data_dir=args.data_dir,
            camera_feature=args.camera_feature,
            tile_width=args.tile_width,
            tile_height=args.tile_height,
            rng=rng,
        )

    print(f"Saved {len(groups)} category grids to: {args.output_dir}")


if __name__ == "__main__":
    main()
