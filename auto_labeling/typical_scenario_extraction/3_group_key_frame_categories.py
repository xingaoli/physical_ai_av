#!/usr/bin/env python3
"""
Group non-empty keyframe classification labels by prompt/category.

Input:
  labels/typical_scenario_key_frame/typical_scenario_key_frame.json

Output:
  labels/typical_scenario_key_frame/typical_scenario_key_frame_category.json

The output is cluster-like:
  {
    "road_facilities_category_5": [
      {
        "clip_id": "...",
        "chunk_id": "chunk_0000",
        "chunk_number": 0,
        "keyframe_index": 73,
        "timestamp_us": 7300000,
        "prompt_name": "road_facilities",
        "category": "5",
        "description": "..."
      }
    ]
  }
"""

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


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


def parse_chunk_number(chunk_id: Any) -> int | None:
    if not isinstance(chunk_id, str) or not chunk_id.startswith("chunk_"):
        return None
    try:
        return int(chunk_id.replace("chunk_", "", 1))
    except ValueError:
        return None


def normalize_category(category: Any) -> str:
    if category is None:
        return ""
    return str(category).strip()


def category_sort_value(category: str) -> tuple[int, int | str]:
    try:
        return (0, int(category))
    except ValueError:
        return (1, category)


def group_sort_key(group_key: str) -> tuple[str, tuple[int, int | str]]:
    prompt_name, _, category = group_key.rpartition("_category_")
    return prompt_name, category_sort_value(category)


def build_category_groups(results: list[dict[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], Counter]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    stats: Counter = Counter()

    for clip_entry in results:
        clip_id = clip_entry.get("clip_id")
        chunk_id = clip_entry.get("chunk_id", "")
        chunk_number = parse_chunk_number(chunk_id)

        for keyframe_result in clip_entry.get("keyframe_results", []):
            keyframe_index = keyframe_result.get("keyframe_index")
            timestamp_us = keyframe_result.get("timestamp_us")

            for prompt_label in keyframe_result.get("prompt_labels", []):
                stats["total_prompt_labels"] += 1

                prompt_name = prompt_label.get("prompt_name")
                label = prompt_label.get("label")
                if not isinstance(label, dict):
                    stats["invalid_labels"] += 1
                    continue
                if "error" in label:
                    stats["error_labels"] += 1
                    continue

                category = normalize_category(label.get("category"))
                if not category:
                    stats["empty_labels"] += 1
                    continue

                group_key = f"{prompt_name}_category_{category}"
                groups[group_key].append({
                    "clip_id": clip_id,
                    "chunk_id": chunk_id,
                    "chunk_number": chunk_number,
                    "keyframe_index": keyframe_index,
                    "timestamp_us": timestamp_us,
                    "prompt_name": prompt_name,
                    "category": category,
                    "description": label.get("description", ""),
                })
                stats["kept_labels"] += 1

    ordered_groups = {
        key: groups[key]
        for key in sorted(groups.keys(), key=group_sort_key)
    }
    stats["groups"] = len(ordered_groups)
    return ordered_groups, stats


def parse_args() -> argparse.Namespace:
    data_dir = default_data_dir()
    default_input = (
        data_dir
        / "labels"
        / "typical_scenario_key_frame"
        / "typical_scenario_key_frame.json"
    )
    default_output = (
        data_dir
        / "labels"
        / "typical_scenario_key_frame"
        / "typical_scenario_key_frame_category.json"
    )

    parser = argparse.ArgumentParser(
        description="Group non-empty keyframe classification labels by category."
    )
    parser.add_argument("--input", type=Path, default=default_input,
                        help="Input typical_scenario_key_frame.json path")
    parser.add_argument("--output", type=Path, default=default_output,
                        help="Output grouped category JSON path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input JSON not found: {args.input}")

    with open(args.input, "r", encoding="utf-8") as f:
        results = json.load(f)
    if not isinstance(results, list):
        raise ValueError(f"Expected top-level JSON list, got {type(results).__name__}")

    groups, stats = build_category_groups(results)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(groups, f, ensure_ascii=False, indent=2)

    print(f"Loaded: {args.input} ({len(results)} clips)")
    print(f"Saved:  {args.output}")
    print(f"Groups: {stats['groups']}, kept labels: {stats['kept_labels']}")
    print(
        "Skipped: "
        f"{stats['empty_labels']} empty, "
        f"{stats['error_labels']} error, "
        f"{stats['invalid_labels']} invalid"
    )
    for key, items in groups.items():
        print(f"  {key}: {len(items)}")


if __name__ == "__main__":
    main()
