#!/usr/bin/env python3
"""
Extract structured scene descriptions for keyframes using VLM.

This script:
1. Reads meta_actions chunk directories to find keyframes for each clip
2. Extracts corresponding frames from camera_front_wide_120fov video
3. Calls VLM API to extract structured scene information
4. Writes results to labels/key_frame_description directory

Usage Examples:
    # 1. Dry run: process chunk 0 with max 2 clips
    python auto_labeling/key_frame_cluster/1_extract_keyframe_description.py --dry-run --chunks 0 --max-clips-per-chunk 2

    # 2. Process specific chunks
    python auto_labeling/key_frame_cluster/1_extract_keyframe_description.py --chunks 0-2 --base-url http://0.0.0.0:8000/v1 --model ckpts/Qwen3.6-27B-FP8

    # 3. Process all chunks
    python auto_labeling/key_frame_cluster/1_extract_keyframe_description.py
"""

import json
import os
import zipfile
import io
import re
import time
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm
import argparse
import cv2
import numpy as np
import pandas as pd
import base64
import physical_ai_av.video as video
from dotenv import load_dotenv


SCENE_EXTRACTION_PROMPT = """# Role
You are an expert autonomous driving scene perception AI. Analyze the provided front-wide-view driving image and extract key environmental & traffic factors that directly impact immediate driving decisions (speed, lane keeping, braking, yielding, etc.).

# Output Format
Return ONLY a valid JSON object. Do NOT include markdown formatting, explanations, or any extra text. The JSON must strictly follow the schema below.

# Field Specifications
1. time_of_day: String. Must be exactly one of: "day" | "night" | "unknown".
2. weather: String. Open-ended. Describe current conditions concisely (e.g., "clear", "heavy rain", "dense fog", "sandstorm", "snowy", "wet & reflective").
3. traffic_control: Array of objects. For each visible control element:
   - type: String. Open-ended. (e.g., "traffic_light", "stop_sign", "speed_limit_sign").
   - description: String. Open-ended. Brief state or content (e.g., "red light", "30 km/h limit", "octagonal stop sign").
   - bbox: Array of 4 numbers [x1, y1, x2, y2] representing top-left and bottom-right pixel coordinates.
4. key_objects: Array of objects. Identify dynamic/static objects critical for decision-making:
   - type: String. MUST be exactly one of: "vehicle" | "pedestrian" | "cyclist" | "animal" | "obstacle" | "other".
   - sub_type: String. Open-ended. Fine-grained classification when distinguishable. Leave as "" if not applicable.
     * vehicle: "ambulance", "fire_truck", "police_car", "bus", "truck", "sedan", "SUV", "van", "construction_vehicle", etc.
     * pedestrian: "child", "construction_worker", "person_with_disability", etc.
     * cyclist: "bicyclist", "motorcyclist", "scooter_rider", "e_bike_rider", etc.
     * animal: "dog", "deer", "cat", etc.
     * obstacle: "debris", "traffic_cone", "barrier", "construction_sign", "fallen_tree_branch", etc.
     * other: "train", "tram", "horse", "stroller", etc.
   - description: String. Open-ended. Focus on state, action, trajectory, or immediate risk context (e.g., "crossing left-to-right", "stopped in lane", "moving slowly near curb", "partially occluded", "fallen tree branch").
   - bbox: [x1, y1, x2, y2]
5. road_state: Array of objects. Focus on road surface anomalies or conditions:
   - description: String. Open-ended. e.g., "pothole in right lane", "ice patch", "construction zone markings", "oil spill".
   - bbox: [x1, y1, x2, y2]

# JSON Template
{
  "time_of_day": "day | night | unknown",
  "weather": "string",
  "traffic_control": [
    {"type": "string", "description": "string", "bbox": [x1, y1, x2, y2]}
  ],
  "key_objects": [
    {"type": "vehicle | pedestrian | cyclist | animal | obstacle | other", "sub_type": "string", "description": "string", "bbox": [x1, y1, x2, y2]}
  ],
  "road_state": [
    {"description": "string", "bbox": [x1, y1, x2, y2]}
  ]
}

# Critical Rules
- If no instances exist for an array field, return [] (empty list).
- bbox format MUST be [x_min, y_min, x_max, y_max] in pixel coordinates. Ensure coordinates are within image boundaries.
- Prioritize objects/conditions that require immediate driver or system reaction. Ignore background noise.
- Output MUST be parseable by standard JSON parsers. No trailing commas, no comments."""


def decode_frame_at_timestamp(data_dir: str, clip_id: str, chunk_number: int,
                               timestamp_us: int, camera_feature: str = "camera_front_wide_120fov") -> np.ndarray:
    camera_zip_path = os.path.join(
        data_dir, "camera", camera_feature,
        f"{camera_feature}.chunk_{chunk_number:04d}.zip"
    )

    if not os.path.exists(camera_zip_path):
        raise FileNotFoundError(f"Camera zip not found: {camera_zip_path}")

    with zipfile.ZipFile(camera_zip_path, 'r') as zf:
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


def encode_image_to_base64(image: np.ndarray) -> str:
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    return image_base64


def parse_vlm_json_response(text: str) -> dict:
    """Extract and parse JSON from VLM response, stripping markdown if present."""
    text = text.strip()
    # Remove ```json ... ``` wrapping
    m = re.search(r'```(?:json)?\s*(.*?)```', text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    return json.loads(text)


def call_vlm_for_scene(image_base64: str, client: OpenAI, model: str = "default") -> dict:
    """Call VLM API to extract structured scene description."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": SCENE_EXTRACTION_PROMPT
                        }
                    ]
                }
            ],
            temperature=0.1,
            max_tokens=4096,
            extra_body={
                "top_k": 20,
                "top_p": 0.9,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )

        result_text = response.choices[0].message.content.strip()
        return parse_vlm_json_response(result_text)

    except json.JSONDecodeError as e:
        print(f"    Warning: Failed to parse VLM JSON response: {e}")
        print(f"    Raw response: {result_text[:200]}...")
        return {"error": "json_parse_error", "raw": result_text[:500]}
    except Exception as e:
        print(f"    Error: VLM API call failed: {e}")
        return {"error": str(e)}


def process_single_clip(clip_id: str, keyframes: list, data_dir: str,
                         chunk_number: int, client: OpenAI, dry_run: bool = False,
                         model: str = "default") -> dict:
    """Process a single clip: extract scene description for each keyframe."""
    result = {
        "clip_id": clip_id,
        "description_lists": []
    }

    for kf in keyframes:
        frame_index = kf["frame_index"]
        # frame_index * 1e5 = timestamp in microseconds
        timestamp_us = int(frame_index * 1e5)

        entry = {
            "keyframe_index": frame_index,
            "timestamp_us": timestamp_us,
        }

        if dry_run:
            print(f"    [Dry Run] clip: {clip_id[:8]}..., keyframe: {frame_index}, timestamp: {timestamp_us} us")
            entry["description"] = {
                "time_of_day": "day",
                "weather": "clear",
                "traffic_control": [],
                "key_objects": [],
                "road_state": []
            }
            result["description_lists"].append(entry)
            continue

        try:
            image = decode_frame_at_timestamp(data_dir, clip_id, chunk_number, timestamp_us)
            image_base64 = encode_image_to_base64(image)
            description = call_vlm_for_scene(image_base64, client, model)
            entry["description"] = description
            print(f"    [OK] clip: {clip_id[:8]}..., keyframe: {frame_index}, "
                  f"objects: {len(description.get('key_objects', []))}, "
                  f"controls: {len(description.get('traffic_control', []))}")
        except Exception as e:
            print(f"    [FAIL] clip: {clip_id[:8]}..., keyframe: {frame_index}, error: {e}")
            entry["description"] = {"error": str(e)}

        result["description_lists"].append(entry)

    return result


def main():
    env_path = Path(__file__).parent.parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded .env from: {env_path}")

    default_data_dir = "/home/xingao/code/Alpamayo1.5/data/PhysicalAI-Autonomous-Vehicles"
    data_dir = os.getenv("ALPAMAYO_DATA_DIR", default_data_dir)

    default_meta_actions_dir = os.path.join(data_dir, "labels", "meta_actions")
    default_output_dir = os.path.join(data_dir, "labels", "key_frame_description")

    parser = argparse.ArgumentParser(description="Extract keyframe scene descriptions using VLM")
    parser.add_argument("--meta-actions-dir", type=str, default=default_meta_actions_dir,
                        help="Input directory with meta_actions chunk dirs")
    parser.add_argument("--output-dir", type=str, default=default_output_dir,
                        help="Output directory for scene descriptions")
    parser.add_argument("--data-dir", type=str, default=data_dir,
                        help="Base data directory")
    parser.add_argument("--api-key", type=str, default="EMPTY",
                        help="OpenAI API Key (default EMPTY)")
    parser.add_argument("--base-url", type=str, default="http://0.0.0.0:8000/v1",
                        help="OpenAI API Base URL")
    parser.add_argument("--model", type=str, default="ckpts/Qwen3.6-27B-FP8",
                        help="Model name to use")
    parser.add_argument("--dry-run", action="store_true",
                        help="Test mode, do not call VLM")
    parser.add_argument("--chunks", type=str, default=None,
                        help="Specify chunk numbers to process. Supports: "
                             "comma-separated (e.g., '0,5,10'), "
                             "range (e.g., '0-5'), "
                             "or mixed (e.g., '0,3-5,8').")
    parser.add_argument("--max-clips-per-chunk", type=int, default=None,
                        help="Max number of clips to process per chunk (for debugging)")
    parser.add_argument("--save-every", type=int, default=500,
                        help="Save intermediate results every N clips (default 1000)")

    args = parser.parse_args()

    print(f"Data directory: {args.data_dir}")
    print(f"Meta actions directory: {args.meta_actions_dir}")
    print(f"Output directory: {args.output_dir}")
    if args.chunks:
        print(f"Specified chunks: {args.chunks}")
    if args.max_clips_per_chunk:
        print(f"Max clips per chunk: {args.max_clips_per_chunk}")

    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url
    )

    # Find all meta_actions chunk directories
    meta_actions_path = Path(args.meta_actions_dir)
    all_chunk_dirs = sorted(meta_actions_path.glob("meta_actions.chunk_*"))
    # Filter out .vis directories
    all_chunk_dirs = [d for d in all_chunk_dirs if d.is_dir() and not d.name.endswith('.vis')]

    print(f"\nFound {len(all_chunk_dirs)} total meta_actions chunk directories")

    if not all_chunk_dirs:
        print("Error: No meta_actions chunk directories found")
        return

    def parse_chunk_spec(chunk_spec: str) -> set:
        chunk_numbers = set()
        parts = chunk_spec.split(',')
        for part in parts:
            part = part.strip()
            if '-' in part:
                start, end = part.split('-', 1)
                chunk_numbers.update(range(int(start.strip()), int(end.strip()) + 1))
            else:
                chunk_numbers.add(int(part))
        return chunk_numbers

    if args.chunks:
        target_chunk_numbers = parse_chunk_spec(args.chunks)
        chunk_dirs = []
        for d in all_chunk_dirs:
            chunk_num_str = d.name.replace("meta_actions.chunk_", "")
            chunk_number = int(chunk_num_str)
            if chunk_number in target_chunk_numbers:
                chunk_dirs.append(d)
        chunk_dirs = sorted(chunk_dirs,
                            key=lambda d: int(d.name.replace("meta_actions.chunk_", "")))
        print(f"Filtered to {len(chunk_dirs)} specified chunk(s)")
        if not chunk_dirs:
            print(f"Error: None of the specified chunks exist. Available: "
                  f"{[d.name for d in all_chunk_dirs[:5]]}...")
            return
    else:
        chunk_dirs = all_chunk_dirs

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Global clip counter for intermediate saves
    global_clip_count = 0
    save_interval = args.save_every
    # Accumulate results across chunks for batched saving
    pending_results = []
    saved_files = []

    total_clips = 0
    for chunk_dir in tqdm(chunk_dirs, desc="Processing chunks"):
        chunk_number = int(chunk_dir.name.replace("meta_actions.chunk_", ""))
        chunk_name = f"chunk_{chunk_number:04d}"

        print(f"\n{'='*60}")
        print(f"Processing {chunk_name}")
        print(f"{'='*60}")

        # Find all clip meta_action files
        clip_files = sorted(chunk_dir.glob("*.meta_actions.json"))
        print(f"Found {len(clip_files)} clips in this chunk")

        if args.max_clips_per_chunk:
            clip_files = clip_files[:args.max_clips_per_chunk]
            print(f"Limited to {len(clip_files)} clips")

        for clip_file in tqdm(clip_files, desc=f"  [{chunk_name}] clips"):
            clip_id = clip_file.name.replace(".meta_actions.json", "")

            try:
                with open(clip_file, 'r', encoding='utf-8') as f:
                    meta_data = json.load(f)

                keyframes = meta_data.get("keyframes", [])
                if not keyframes:
                    continue

                result = process_single_clip(
                    clip_id, keyframes, args.data_dir, chunk_number,
                    client, args.dry_run, args.model
                )
                result["chunk_id"] = chunk_name

                pending_results.append(result)
                global_clip_count += 1

            except Exception as e:
                print(f"  Error: Failed to process clip {clip_id}: {e}")
                pending_results.append({
                    "chunk_id": chunk_name,
                    "clip_id": clip_id,
                    "description_lists": [],
                    "error": str(e)
                })
                global_clip_count += 1

            # Save intermediate results periodically
            if global_clip_count % save_interval == 0 and pending_results:
                save_idx = (global_clip_count // save_interval) - 1
                save_file = output_path / f"{save_idx * save_interval}-{global_clip_count - 1}.json"
                with open(save_file, 'w', encoding='utf-8') as f:
                    json.dump(pending_results, f, ensure_ascii=False, indent=2)
                print(f"\n  Intermediate save: {save_file} ({len(pending_results)} clips)")
                saved_files.append(save_file)
                pending_results = []

            total_clips += 1

    # Save remaining results
    if pending_results:
        start_idx = (global_clip_count // save_interval) * save_interval
        save_file = output_path / f"{start_idx}-{global_clip_count - 1}.json"
        with open(save_file, 'w', encoding='utf-8') as f:
            json.dump(pending_results, f, ensure_ascii=False, indent=2)
        print(f"\n  Final save: {save_file} ({len(pending_results)} clips)")
        saved_files.append(save_file)

    # Merge all intermediate JSON files into one
    all_results = []
    for json_file in saved_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            all_results.extend(json.load(f))

    merged_file = output_path / "kf_desc.json"
    with open(merged_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n  Merged {len(all_results)} clips into {merged_file}")

    print(f"\n{'='*60}")
    print(f"All done! Results saved to: {output_path}")
    print(f"Total clips processed: {total_clips}")


if __name__ == "__main__":
    main()
