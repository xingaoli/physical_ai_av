#!/usr/bin/env python3
"""
Multi-task parallel classification of driving scenes using VLM.

This script:
1. Reads all prompts from auto_labeling/typical_scenario_extraction/prompt/
2. Reads clip_index.parquet to discover clips and their chunk assignments
3. Extracts frames from 2.5s to 13.5s at 0.5s intervals
4. Sends all (prompt × frame) requests concurrently (max 16 in-flight)
5. Retries failed requests with logging
6. Writes results to labels/typical_scenario directory

Usage Examples:
    # 1. Dry run
    python auto_labeling/typical_scenario_extraction/1_classify_all_parallel.py --dry-run --chunks 0 --max-clips-per-chunk 2

    # 2. Process specific chunks without thinking
    python auto_labeling/typical_scenario_extraction/1_classify_all_parallel.py --chunks 0-2 --no-thinking

    # 3. Retry only failed tasks from a previous run
    python auto_labeling/typical_scenario_extraction/1_classify_all_parallel.py --retry-from labels/typical_scenario/typical_scenario.json

    # 4. Process all chunks
    python auto_labeling/typical_scenario_extraction/1_classify_all_parallel.py
"""

import asyncio
import json
import os
import zipfile
import io
import re
from pathlib import Path
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
import argparse
import cv2
import numpy as np
import pandas as pd
import base64
import physical_ai_av.video as video
from dotenv import load_dotenv


# Timestamps from 2.5s to 13.5s at 0.5s intervals (in microseconds)
TIMESTAMPS_US = [int(t * 1e6) for t in np.arange(2.5, 13.51, 0.5)]

MAX_CONCURRENCY = 4
MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def load_prompts(prompt_dir: str) -> dict[str, str]:
    """Load all .txt prompts from directory. Returns {stem: content}."""
    prompts = {}
    for p in sorted(Path(prompt_dir).glob("*.txt")):
        prompts[p.stem] = p.read_text(encoding="utf-8").strip()
    return prompts


# ---------------------------------------------------------------------------
# Frame decode / encode
# ---------------------------------------------------------------------------

def decode_frames_at_timestamps(data_dir: str, clip_id: str, chunk_number: int,
                                 timestamps_us: list[int],
                                 camera_feature: str = "camera_front_wide_120fov") -> dict[int, np.ndarray]:
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
        reader = video.SeekVideoReader(video_data=video_data, timestamps=frame_timestamps)
        target = np.array(timestamps_us, dtype=np.int64)
        frames, decoded_ts = reader.decode_images_from_timestamps(target)
        return {int(timestamps_us[i]): frames[i] for i in range(frames.shape[0])}


def encode_image_to_base64(image: np.ndarray) -> str:
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode('utf-8')


# ---------------------------------------------------------------------------
# VLM call
# ---------------------------------------------------------------------------

def parse_vlm_json_response(text: str) -> dict:
    text = text.strip()
    m = re.search(r'```(?:json)?\s*(.*?)```', text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    return json.loads(text)


async def call_vlm(image_base64: str, prompt: str, client: AsyncOpenAI,
                   model: str, enable_thinking: bool) -> dict:
    """Single VLM API call with retry."""
    extra_body = {
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    if enable_thinking:
        extra_body["thinking_token_budget"] = 4000
    response = await client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                {"type": "text", "text": prompt},
            ]
        }],
        max_tokens=512,
        temperature=0.6,
        top_p=0.95,
        presence_penalty=1.5,
        extra_body=extra_body,
    )
    result_text = response.choices[0].message.content.strip()
    try:
        return parse_vlm_json_response(result_text)
    except json.JSONDecodeError as e:
        print(f"\n  [JSON parse failed] raw response: {result_text[:500]}", flush=True)
        raise


# ---------------------------------------------------------------------------
# Task: one (clip, timestamp, prompt) unit
# ---------------------------------------------------------------------------

class Task:
    """Represents a single (clip_id, timestamp_us, prompt_name) classification request."""
    __slots__ = ("clip_id", "chunk_number", "timestamp_us", "prompt_name",
                 "image_base64", "result", "error", "retries")

    def __init__(self, clip_id: str, chunk_number: int, timestamp_us: int,
                 prompt_name: str, image_base64: str):
        self.clip_id = clip_id
        self.chunk_number = chunk_number
        self.timestamp_us = timestamp_us
        self.prompt_name = prompt_name
        self.image_base64 = image_base64
        self.result = None
        self.error = None
        self.retries = 0

    @property
    def key(self) -> str:
        return f"{self.clip_id}|{self.timestamp_us}|{self.prompt_name}"


async def execute_task(task: Task, client: AsyncOpenAI, model: str,
                       enable_thinking: bool, prompt_map: dict[str, str]) -> Task:
    """Execute one task. On success sets task.result; on failure sets task.error."""
    try:
        task.result = await call_vlm(
            task.image_base64, prompt_map[task.prompt_name],
            client, model, enable_thinking,
        )
        task.error = None
    except json.JSONDecodeError as e:
        task.error = f"json_parse_error: {e}"
        task.result = None
    except Exception as e:
        task.error = str(e)
        task.result = None
    return task


# ---------------------------------------------------------------------------
# Parallel runner with bounded concurrency and retry
# ---------------------------------------------------------------------------

async def run_tasks_with_retry(tasks: list[Task], client: AsyncOpenAI,
                                model: str, enable_thinking: bool,
                                prompt_map: dict[str, str],
                                semaphore: asyncio.Semaphore) -> list[Task]:
    """Run tasks with bounded concurrency, then retry failures up to MAX_RETRIES rounds."""

    async def _guarded(t: Task) -> Task:
        async with semaphore:
            return await execute_task(t, client, model, enable_thinking, prompt_map)

    async def _run_batch(batch: list[Task], desc: str) -> list[Task]:
        if not batch:
            return []
        coros = [_guarded(t) for t in batch]
        return await tqdm_asyncio.gather(*coros, desc=desc)

    # Initial run
    await _run_batch(tasks, f"  API ({len(tasks)} tasks)")

    # Retry rounds
    for retry_round in range(1, MAX_RETRIES + 1):
        failed = [t for t in tasks if t.error is not None]
        if not failed:
            break
        print(f"\n  Retry round {retry_round}: {len(failed)} failed tasks")
        for f in failed[:5]:
            print(f"    - {f.key}: {f.error}")
        if len(failed) > 5:
            print(f"    ... and {len(failed) - 5} more")

        for t in failed:
            t.retries += 1
        await _run_batch(failed, f"  Retry-{retry_round} ({len(failed)} tasks)")

    # Final failure summary
    final_failed = [t for t in tasks if t.error is not None]
    if final_failed:
        print(f"\n  WARNING: {len(final_failed)} tasks still failed after {MAX_RETRIES} retries:")
        for t in final_failed[:10]:
            print(f"    - {t.key}: {t.error}")
        if len(final_failed) > 10:
            print(f"    ... and {len(final_failed) - 10} more")

    return tasks


# ---------------------------------------------------------------------------
# Clip-level processing
# ---------------------------------------------------------------------------

def build_tasks_for_clip(clip_id: str, chunk_number: int, data_dir: str,
                         prompt_names: list[str]) -> list[Task]:
    """Decode frames and build Task objects for one clip."""
    frames = decode_frames_at_timestamps(data_dir, clip_id, chunk_number, TIMESTAMPS_US)
    tasks = []
    for ts_us in TIMESTAMPS_US:
        if ts_us not in frames:
            # Create tasks with empty image — they will fail and get logged
            for pname in prompt_names:
                t = Task(clip_id, chunk_number, ts_us, pname, "")
                t.error = "frame_not_found"
                tasks.append(t)
            continue
        img_b64 = encode_image_to_base64(frames[ts_us])
        for pname in prompt_names:
            tasks.append(Task(clip_id, chunk_number, ts_us, pname, img_b64))
    return tasks


# ---------------------------------------------------------------------------
# Retry from previous results file
# ---------------------------------------------------------------------------

def collect_failed_tasks_from_results(results: list[dict], data_dir: str,
                                       prompt_map: dict[str, str]) -> list[Task]:
    """Scan a previous results file and build Task objects for failed entries."""
    tasks = []
    for clip_entry in results:
        clip_id = clip_entry["clip_id"]
        chunk_number = int(clip_entry.get("chunk_id", "chunk_0000").replace("chunk_", ""))
        for fr in clip_entry.get("frame_results", []):
            ts_us = fr["timestamp_us"]
            for plabel in fr.get("prompt_labels", []):
                if "error" in plabel.get("label", {}):
                    pname = plabel["prompt_name"]
                    try:
                        frames = decode_frames_at_timestamps(data_dir, clip_id, chunk_number, [ts_us])
                        if ts_us in frames:
                            img_b64 = encode_image_to_base64(frames[ts_us])
                            t = Task(clip_id, chunk_number, ts_us, pname, img_b64)
                            t.retries = plabel.get("retries", 0)
                            tasks.append(t)
                        else:
                            print(f"  SKIP {clip_id[:8]}... t={ts_us / 1e6:.1f}s frame not found")
                    except Exception as e:
                        print(f"  SKIP {clip_id[:8]}... t={ts_us / 1e6:.1f}s decode error: {e}")
    return tasks


def merge_retry_results(original_results: list[dict], retry_tasks: list[Task]) -> list[dict]:
    """Merge retried task results back into the original results structure."""
    # Build lookup: (clip_id, timestamp_us, prompt_name) -> Task
    retry_lookup = {}
    for t in retry_tasks:
        if t.result is not None:
            retry_lookup[(t.clip_id, t.timestamp_us, t.prompt_name)] = t

    fixed = 0
    for clip_entry in original_results:
        clip_id = clip_entry["clip_id"]
        for fr in clip_entry.get("frame_results", []):
            ts_us = fr["timestamp_us"]
            for plabel in fr.get("prompt_labels", []):
                pname = plabel["prompt_name"]
                key = (clip_id, ts_us, pname)
                if key in retry_lookup:
                    plabel["label"] = retry_lookup[key].result
                    if "error" in plabel:
                        del plabel["error"]
                    fixed += 1
    print(f"  Merged {fixed} fixed results back into original data")
    return original_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_chunk_spec(chunk_spec: str) -> set:
    chunk_numbers = set()
    for part in chunk_spec.split(','):
        part = part.strip()
        if '-' in part:
            start, end = part.split('-', 1)
            chunk_numbers.update(range(int(start.strip()), int(end.strip()) + 1))
        else:
            chunk_numbers.add(int(part))
    return chunk_numbers


def main():
    env_path = Path(__file__).parent.parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded .env from: {env_path}")

    default_data_dir = "/home/xingao/code/Alpamayo1.5/data/PhysicalAI-Autonomous-Vehicles"
    data_dir = os.getenv("ALPAMAYO_DATA_DIR", default_data_dir)

    default_output_dir = os.path.join(data_dir, "labels", "typical_scenario")
    default_prompt_dir = str(Path(__file__).parent / "prompt")

    parser = argparse.ArgumentParser(
        description="Parallel multi-task scene classification using VLM")
    parser.add_argument("--output-dir", type=str, default=default_output_dir)
    parser.add_argument("--prompt-dir", type=str, default=default_prompt_dir,
                        help="Directory containing prompt .txt files")
    parser.add_argument("--data-dir", type=str, default=data_dir)
    parser.add_argument("--api-key", type=str, default="EMPTY")
    parser.add_argument("--base-url", type=str, default="http://0.0.0.0:8080/v1")
    parser.add_argument("--model", type=str, default="ckpts/Qwen3.6-27B-int4-AutoRound")
    parser.add_argument("--no-thinking", action="store_true",
                        help="Disable thinking mode for VLM")
    parser.add_argument("--max-concurrency", type=int, default=MAX_CONCURRENCY,
                        help=f"Max concurrent API requests (default {MAX_CONCURRENCY})")
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES,
                        help=f"Max retry rounds for failed tasks (default {MAX_RETRIES})")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--chunks", type=str, default=None,
                        help="Chunk numbers: '0,5,10' or '0-5' or '0,3-5,8'")
    parser.add_argument("--max-clips-per-chunk", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--retry-from", type=str, default=None,
                        help="Path to a previous results JSON file; only retry failed tasks")

    args = parser.parse_args()
    enable_thinking = not args.no_thinking

    # Load prompts
    prompt_map = load_prompts(args.prompt_dir)
    prompt_names = sorted(prompt_map.keys())
    print(f"Loaded {len(prompt_map)} prompts: {prompt_names}")

    total_tasks_per_clip = len(TIMESTAMPS_US) * len(prompt_names)
    print(f"Frames per clip: {len(TIMESTAMPS_US)} "
          f"({TIMESTAMPS_US[0] / 1e6:.1f}s - {TIMESTAMPS_US[-1] / 1e6:.1f}s)")
    print(f"Tasks per clip: {total_tasks_per_clip} "
          f"({len(TIMESTAMPS_US)} frames × {len(prompt_names)} prompts)")
    print(f"Max concurrency: {args.max_concurrency}")
    print(f"Thinking: {'enabled' if enable_thinking else 'disabled'}")
    print(f"Output: {args.output_dir}")

    # ---- Retry mode ----
    if args.retry_from:
        asyncio.run(_retry_mode(args, prompt_map, data_dir))
        return

    # ---- Normal mode ----
    asyncio.run(_normal_mode(args, prompt_map, prompt_names, data_dir))


async def _retry_mode(args, prompt_map, data_dir):
    """Retry failed tasks from a previous results file."""
    retry_path = Path(args.retry_from)
    if not retry_path.exists():
        print(f"Error: retry file not found: {retry_path}")
        return

    print(f"\nRetry mode: loading {retry_path}")
    with open(retry_path, 'r', encoding='utf-8') as f:
        original_results = json.load(f)
    print(f"Loaded {len(original_results)} clips")

    # Collect failed tasks
    tasks = collect_failed_tasks_from_results(original_results, data_dir, prompt_map)
    if not tasks:
        print("No failed tasks found. Nothing to retry.")
        return
    print(f"Found {len(tasks)} failed tasks to retry")

    client = AsyncOpenAI(api_key=args.api_key, base_url=args.base_url)
    semaphore = asyncio.Semaphore(args.max_concurrency)

    tasks = await run_tasks_with_retry(
        tasks, client, args.model, not args.no_thinking, prompt_map, semaphore)

    # Merge back
    original_results = merge_retry_results(original_results, tasks)

    # Save
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    merged_file = output_path / "typical_scenario.json"
    with open(merged_file, 'w', encoding='utf-8') as f:
        json.dump(original_results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved merged results to {merged_file}")

    # Stats
    still_failed = sum(1 for t in tasks if t.error is not None)
    fixed = sum(1 for t in tasks if t.result is not None)
    print(f"Retry stats: {fixed} fixed, {still_failed} still failed")


async def _normal_mode(args, prompt_map, prompt_names, data_dir):
    """Normal processing mode: discover clips via clip_index.parquet and camera zips."""
    camera_feature = "camera_front_wide_120fov"
    camera_dir = os.path.join(data_dir, "camera", camera_feature)

    # Load clip_index.parquet
    clip_index_path = os.path.join(data_dir, "clip_index.parquet")
    if not os.path.exists(clip_index_path):
        print(f"Error: clip_index.parquet not found at {clip_index_path}")
        return
    clip_df = pd.read_parquet(clip_index_path)
    print(f"\nLoaded clip_index.parquet: {len(clip_df)} total clips")

    # Filter valid clips
    clip_df = clip_df[clip_df["clip_is_valid"] == True]  # noqa: E712

    # Determine which chunks to process based on existing camera zips
    existing_zips = {}
    for p in Path(camera_dir).glob(f"{camera_feature}.chunk_*.zip"):
        chunk_num = int(p.name.replace(f"{camera_feature}.chunk_", "").replace(".zip", ""))
        existing_zips[chunk_num] = p
    print(f"Found {len(existing_zips)} camera chunk zips "
          f"(chunks {min(existing_zips)}-{max(existing_zips)})")

    if args.chunks:
        target = parse_chunk_spec(args.chunks)
        target = target & set(existing_zips.keys())
        if not target:
            print("Error: None of the specified chunks have camera zips")
            return
    else:
        target = set(existing_zips.keys())

    # Filter clips to those in target chunks that have camera data
    clip_df = clip_df[clip_df["chunk"].isin(target)]
    print(f"Processing {len(clip_df)} clips across {len(target)} chunks")

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    client = AsyncOpenAI(api_key=args.api_key, base_url=args.base_url)
    semaphore = asyncio.Semaphore(args.max_concurrency)

    global_clip_count = 0
    pending_results = []
    saved_files = []

    # Group by chunk for ordered processing
    for chunk_number in sorted(target):
        chunk_name = f"chunk_{chunk_number:04d}"
        chunk_clips = clip_df[clip_df["chunk"] == chunk_number]
        clip_ids = chunk_clips.index.tolist()

        print(f"\n{'='*60}")
        print(f"Processing {chunk_name} ({len(clip_ids)} clips)")
        print(f"{'='*60}")

        if args.max_clips_per_chunk:
            clip_ids = clip_ids[:args.max_clips_per_chunk]
            print(f"Limited to {len(clip_ids)} clips")

        for clip_id in clip_ids:
            if args.dry_run:
                print(f"  [Dry Run] {clip_id[:8]}... "
                      f"({len(TIMESTAMPS_US)} frames × {len(prompt_names)} prompts = "
                      f"{len(TIMESTAMPS_US) * len(prompt_names)} tasks)")
                result = {
                    "clip_id": clip_id,
                    "chunk_id": chunk_name,
                    "frame_results": [
                        {"timestamp_us": ts, "prompt_labels": [
                            {"prompt_name": pn, "label": {}} for pn in prompt_names
                        ]}
                        for ts in TIMESTAMPS_US
                    ],
                }
                pending_results.append(result)
                global_clip_count += 1
                continue

            try:
                tasks = build_tasks_for_clip(clip_id, chunk_number, args.data_dir, prompt_names)

                runnable = [t for t in tasks if t.error is None]
                frame_missing = [t for t in tasks if t.error is not None]

                if frame_missing:
                    print(f"  {clip_id[:8]}... {len(frame_missing)} tasks skipped (frame missing)")

                tasks = await run_tasks_with_retry(
                    runnable, client, args.model, not args.no_thinking,
                    prompt_map, semaphore)

                all_tasks = tasks + frame_missing
                result = _organize_clip_result(clip_id, chunk_name, all_tasks)
                pending_results.append(result)
                global_clip_count += 1

                ok = sum(1 for t in all_tasks if t.result is not None)
                fail = sum(1 for t in all_tasks if t.error is not None)
                print(f"  {clip_id[:8]}... done: {ok} ok, {fail} failed")

            except Exception as e:
                print(f"  [FAIL] {clip_id[:8]}... clip error: {e}")
                pending_results.append({
                    "clip_id": clip_id,
                    "chunk_id": chunk_name,
                    "frame_results": [],
                    "error": str(e),
                })
                global_clip_count += 1

            # Periodic save
            if global_clip_count % args.save_every == 0 and pending_results:
                save_idx = (global_clip_count // args.save_every) - 1
                save_file = output_path / f"{save_idx * args.save_every}-{global_clip_count - 1}.json"
                with open(save_file, 'w', encoding='utf-8') as f:
                    json.dump(pending_results, f, ensure_ascii=False, indent=2)
                print(f"\n  Intermediate save: {save_file} ({len(pending_results)} clips)")
                saved_files.append(save_file)
                pending_results = []

    # Save remaining
    if pending_results:
        start_idx = (global_clip_count // args.save_every) * args.save_every
        save_file = output_path / f"{start_idx}-{global_clip_count - 1}.json"
        with open(save_file, 'w', encoding='utf-8') as f:
            json.dump(pending_results, f, ensure_ascii=False, indent=2)
        print(f"\n  Final save: {save_file} ({len(pending_results)} clips)")
        saved_files.append(save_file)

    # Merge
    all_results = []
    for jf in saved_files:
        with open(jf, 'r', encoding='utf-8') as f:
            all_results.extend(json.load(f))

    merged_file = output_path / "typical_scenario.json"
    with open(merged_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n  Merged {len(all_results)} clips into {merged_file}")

    # Final stats
    total_ok = 0
    total_fail = 0
    for r in all_results:
        for fr in r.get("frame_results", []):
            for pl in fr.get("prompt_labels", []):
                label = pl.get("label", {})
                if "error" in label:
                    total_fail += 1
                else:
                    total_ok += 1
    print(f"\n{'='*60}")
    print(f"Done! Total: {global_clip_count} clips, "
          f"{total_ok} ok, {total_fail} failed")


def _organize_clip_result(clip_id: str, chunk_name: str, tasks: list[Task]) -> dict:
    """Organize flat task list into structured clip result."""
    # Group by timestamp
    ts_groups: dict[int, list[Task]] = {}
    for t in tasks:
        ts_groups.setdefault(t.timestamp_us, []).append(t)

    frame_results = []
    for ts_us in TIMESTAMPS_US:
        group = ts_groups.get(ts_us, [])
        prompt_labels = []
        for t in group:
            entry = {"prompt_name": t.prompt_name}
            if t.result is not None:
                entry["label"] = t.result
            elif t.error is not None:
                entry["label"] = {"error": t.error}
                entry["retries"] = t.retries
            prompt_labels.append(entry)
        frame_results.append({
            "timestamp_us": ts_us,
            "prompt_labels": prompt_labels,
        })

    return {
        "clip_id": clip_id,
        "chunk_id": chunk_name,
        "frame_results": frame_results,
    }


if __name__ == "__main__":
    main()
