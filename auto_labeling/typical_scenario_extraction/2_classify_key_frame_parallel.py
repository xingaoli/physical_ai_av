#!/usr/bin/env python3
"""
Multi-task parallel classification of keyframes using VLM.

This script:
1. Reads all prompts from auto_labeling/typical_scenario_extraction/prompt/
2. Reads meta_actions chunk directories to discover keyframes for each clip
3. Decodes keyframe images from camera_front_wide_120fov video
4. Sends all (prompt × keyframe) requests concurrently with bounded concurrency
5. Retries failed requests with logging
6. Writes results to labels/typical_scenario_key_frame directory

Usage Examples:
    # 1. Dry run
    python auto_labeling/typical_scenario_extraction/2_classify_key_frame_parallel.py --dry-run --chunks 0 --max-clips-per-chunk 2

    # 2. Process specific chunks without thinking
    python auto_labeling/typical_scenario_extraction/2_classify_key_frame_parallel.py --chunks 0-2 --no-thinking

    # 3. Retry only failed tasks from a previous run
    python auto_labeling/typical_scenario_extraction/2_classify_key_frame_parallel.py --retry-from labels/typical_scenario_key_frame/typical_scenario_key_frame.json

    # 4. Process all chunks
    python auto_labeling/typical_scenario_extraction/2_classify_key_frame_parallel.py
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


MAX_CONCURRENCY = 8
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

def decode_frame_at_timestamp(data_dir: str, clip_id: str, chunk_number: int,
                              timestamp_us: int,
                              camera_feature: str = "camera_front_wide_120fov") -> np.ndarray:
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
        target = np.array([timestamp_us], dtype=np.int64)
        frames, _ = reader.decode_images_from_timestamps(target)
        if frames.shape[0] == 0:
            raise ValueError(f"No frame decoded for timestamp {timestamp_us} us")
        return frames[0]


def encode_image_to_base64(image: np.ndarray) -> str:
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode('utf-8')


# ---------------------------------------------------------------------------
# Keyframe loading from meta_actions
# ---------------------------------------------------------------------------

def load_keyframes_for_clip(meta_actions_dir: str, clip_id: str,
                            chunk_number: int) -> list[dict]:
    """Load keyframes from meta_actions JSON for a single clip."""
    meta_file = Path(meta_actions_dir) / f"meta_actions.chunk_{chunk_number:04d}" / f"{clip_id}.meta_actions.json"
    if not meta_file.exists():
        return []
    with open(meta_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get("keyframes", [])


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
        max_tokens=8192,
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
# Task: one (clip, keyframe, prompt) unit
# ---------------------------------------------------------------------------

class Task:
    """Represents a single (clip_id, keyframe, prompt_name) classification request."""
    __slots__ = ("clip_id", "chunk_number", "keyframe_index", "timestamp_us",
                 "prompt_name", "image_base64", "result", "error", "retries")

    def __init__(self, clip_id: str, chunk_number: int, keyframe_index: int,
                 timestamp_us: int, prompt_name: str, image_base64: str):
        self.clip_id = clip_id
        self.chunk_number = chunk_number
        self.keyframe_index = keyframe_index
        self.timestamp_us = timestamp_us
        self.prompt_name = prompt_name
        self.image_base64 = image_base64
        self.result = None
        self.error = None
        self.retries = 0

    @property
    def key(self) -> str:
        return f"{self.clip_id}|kf{self.keyframe_index}|{self.prompt_name}"


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
                         meta_actions_dir: str, prompt_names: list[str]) -> list[Task]:
    """Decode keyframe images and build Task objects for one clip."""
    keyframes = load_keyframes_for_clip(meta_actions_dir, clip_id, chunk_number)
    if not keyframes:
        return []

    tasks = []
    for kf in keyframes:
        kf_index = kf["frame_index"]
        timestamp_us = kf.get("timestamp_us", int(kf_index * 1e5))
        try:
            image = decode_frame_at_timestamp(data_dir, clip_id, chunk_number, timestamp_us)
            img_b64 = encode_image_to_base64(image)
        except Exception as e:
            for pname in prompt_names:
                t = Task(clip_id, chunk_number, kf_index, timestamp_us, pname, "")
                t.error = f"frame_decode_error: {e}"
                tasks.append(t)
            continue
        for pname in prompt_names:
            tasks.append(Task(clip_id, chunk_number, kf_index, timestamp_us, pname, img_b64))
    return tasks


# ---------------------------------------------------------------------------
# Retry from previous results file
# ---------------------------------------------------------------------------

def collect_failed_tasks_from_results(results: list[dict], data_dir: str,
                                       meta_actions_dir: str,
                                       prompt_map: dict[str, str]) -> list[Task]:
    """Scan a previous results file and build Task objects for failed entries."""
    tasks = []
    for clip_entry in results:
        clip_id = clip_entry["clip_id"]
        chunk_number = int(clip_entry.get("chunk_id", "chunk_0000").replace("chunk_", ""))
        for kfr in clip_entry.get("keyframe_results", []):
            kf_index = kfr["keyframe_index"]
            timestamp_us = kfr["timestamp_us"]
            for plabel in kfr.get("prompt_labels", []):
                if "error" in plabel.get("label", {}):
                    pname = plabel["prompt_name"]
                    try:
                        image = decode_frame_at_timestamp(data_dir, clip_id, chunk_number, timestamp_us)
                        img_b64 = encode_image_to_base64(image)
                        t = Task(clip_id, chunk_number, kf_index, timestamp_us, pname, img_b64)
                        t.retries = plabel.get("retries", 0)
                        tasks.append(t)
                    except Exception as e:
                        print(f"  SKIP {clip_id[:8]}... kf={kf_index} decode error: {e}")
    return tasks


def merge_retry_results(original_results: list[dict], retry_tasks: list[Task]) -> list[dict]:
    """Merge retried task results back into the original results structure."""
    retry_lookup = {}
    for t in retry_tasks:
        if t.result is not None:
            retry_lookup[(t.clip_id, t.keyframe_index, t.prompt_name)] = t

    fixed = 0
    for clip_entry in original_results:
        clip_id = clip_entry["clip_id"]
        for kfr in clip_entry.get("keyframe_results", []):
            kf_index = kfr["keyframe_index"]
            for plabel in kfr.get("prompt_labels", []):
                pname = plabel["prompt_name"]
                key = (clip_id, kf_index, pname)
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

    default_meta_actions_dir = os.path.join(data_dir, "labels", "meta_actions")
    default_output_dir = os.path.join(data_dir, "labels", "typical_scenario_key_frame")
    default_prompt_dir = str(Path(__file__).parent / "prompt")

    parser = argparse.ArgumentParser(
        description="Parallel multi-task keyframe classification using VLM")
    parser.add_argument("--output-dir", type=str, default=default_output_dir)
    parser.add_argument("--prompt-dir", type=str, default=default_prompt_dir,
                        help="Directory containing prompt .txt files")
    parser.add_argument("--data-dir", type=str, default=data_dir)
    parser.add_argument("--meta-actions-dir", type=str, default=default_meta_actions_dir,
                        help="Directory containing meta_actions chunk directories")
    parser.add_argument("--api-key", type=str, default="EMPTY")
    parser.add_argument("--base-url", type=str, default="http://0.0.0.0:8080/v1")
    parser.add_argument("--model", type=str, default="ckpts/Qwen3.6-27B-int4-AutoRound")
    parser.add_argument("--no-thinking", action="store_true",
                        help="Disable thinking mode for VLM")
    parser.add_argument("--max-concurrency", type=int, default=MAX_CONCURRENCY,
                        help=f"Max concurrent API requests (default {MAX_CONCURRENCY})")
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES,
                        help=f"Max retry rounds for failed tasks (default {MAX_RETRIES})")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Number of clips to decode and submit per batch (default 5)")
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
    tasks = collect_failed_tasks_from_results(original_results, data_dir,
                                              args.meta_actions_dir, prompt_map)
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
    merged_file = output_path / "typical_scenario_key_frame.json"
    with open(merged_file, 'w', encoding='utf-8') as f:
        json.dump(original_results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved merged results to {merged_file}")

    # Stats
    still_failed = sum(1 for t in tasks if t.error is not None)
    fixed = sum(1 for t in tasks if t.result is not None)
    print(f"Retry stats: {fixed} fixed, {still_failed} still failed")


async def _normal_mode(args, prompt_map, prompt_names, data_dir):
    """Normal processing mode: discover clips via meta_actions directories."""
    meta_actions_path = Path(args.meta_actions_dir)
    all_chunk_dirs = sorted(meta_actions_path.glob("meta_actions.chunk_*"))
    all_chunk_dirs = [d for d in all_chunk_dirs if d.is_dir() and not d.name.endswith('.vis')]

    if not all_chunk_dirs:
        print(f"Error: No meta_actions chunk directories found in {args.meta_actions_dir}")
        return
    print(f"\nFound {len(all_chunk_dirs)} meta_actions chunk directories")

    if args.chunks:
        target = parse_chunk_spec(args.chunks)
        chunk_dirs = []
        for d in all_chunk_dirs:
            chunk_num = int(d.name.replace("meta_actions.chunk_", ""))
            if chunk_num in target:
                chunk_dirs.append(d)
        chunk_dirs = sorted(chunk_dirs,
                            key=lambda d: int(d.name.replace("meta_actions.chunk_", "")))
        if not chunk_dirs:
            print(f"Error: None of the specified chunks exist. Available: "
                  f"{[d.name for d in all_chunk_dirs[:5]]}...")
            return
        print(f"Filtered to {len(chunk_dirs)} specified chunk(s)")
    else:
        chunk_dirs = all_chunk_dirs

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    client = AsyncOpenAI(api_key=args.api_key, base_url=args.base_url)
    semaphore = asyncio.Semaphore(args.max_concurrency)

    BATCH_SIZE = args.batch_size

    global_clip_count = 0
    pending_results = []
    saved_files = []

    # Collect all clip entries first
    all_clip_entries = []
    for chunk_dir in chunk_dirs:
        chunk_number = int(chunk_dir.name.replace("meta_actions.chunk_", ""))
        chunk_name = f"chunk_{chunk_number:04d}"

        clip_files = sorted(chunk_dir.glob("*.meta_actions.json"))
        if args.max_clips_per_chunk:
            clip_files = clip_files[:args.max_clips_per_chunk]

        for clip_file in clip_files:
            clip_id = clip_file.name.replace(".meta_actions.json", "")
            all_clip_entries.append((clip_id, chunk_number, chunk_name))

    print(f"\nTotal clips to process: {len(all_clip_entries)}, batch size: {BATCH_SIZE}")

    # Process clips in batches
    for batch_start in range(0, len(all_clip_entries), BATCH_SIZE):
        batch = all_clip_entries[batch_start:batch_start + BATCH_SIZE]
        print(f"\n{'='*60}")
        print(f"Batch {batch_start // BATCH_SIZE + 1}: "
              f"clips {batch_start+1}-{batch_start+len(batch)}")
        print(f"{'='*60}")

        # Phase 1: Decode images for this batch
        batch_tasks = []
        batch_error_tasks = []
        clip_ranges = {}  # clip_id -> (start, end, estart, eend)

        for clip_id, chunk_number, chunk_name in batch:
            if args.dry_run:
                continue

            try:
                tasks = build_tasks_for_clip(clip_id, chunk_number, args.data_dir,
                                             args.meta_actions_dir, prompt_names)
            except Exception as e:
                print(f"  [FAIL] {clip_id[:8]}... clip decode error: {e}")
                pending_results.append({
                    "clip_id": clip_id,
                    "chunk_id": chunk_name,
                    "keyframe_results": [],
                    "error": str(e),
                })
                global_clip_count += 1
                continue

            runnable = [t for t in tasks if t.error is None]
            error = [t for t in tasks if t.error is not None]

            start = len(batch_tasks)
            batch_tasks.extend(runnable)
            end = len(batch_tasks)

            estart = len(batch_error_tasks)
            batch_error_tasks.extend(error)
            eend = len(batch_error_tasks)

            clip_ranges[clip_id] = (start, end, estart, eend)

        # Dry run handling
        if args.dry_run:
            for clip_id, chunk_number, chunk_name in batch:
                keyframes = load_keyframes_for_clip(args.meta_actions_dir, clip_id, chunk_number)
                result = {
                    "clip_id": clip_id,
                    "chunk_id": chunk_name,
                    "keyframe_results": [
                        {"keyframe_index": kf["frame_index"],
                         "timestamp_us": kf.get("timestamp_us", int(kf["frame_index"] * 1e5)),
                         "prompt_labels": [
                             {"prompt_name": pn, "label": {}} for pn in prompt_names
                         ]}
                        for kf in keyframes
                    ],
                }
                pending_results.append(result)
                global_clip_count += 1
                n_tasks = len(keyframes) * len(prompt_names)
                print(f"  [Dry Run] {clip_id[:8]}... ({n_tasks} tasks)")
            continue

        # Phase 2: Submit batch tasks concurrently
        if batch_tasks:
            batch_tasks = await run_tasks_with_retry(
                batch_tasks, client, args.model, not args.no_thinking,
                prompt_map, semaphore)

        # Phase 3: Organize results for this batch
        for clip_id, chunk_number, chunk_name in batch:
            if clip_id not in clip_ranges:
                continue  # already handled as decode failure
            start, end, estart, eend = clip_ranges[clip_id]

            clip_tasks = batch_tasks[start:end] + batch_error_tasks[estart:eend]
            result = _organize_clip_result(clip_id, chunk_name, clip_tasks)
            pending_results.append(result)
            global_clip_count += 1

            ok = sum(1 for t in clip_tasks if t.result is not None)
            fail = sum(1 for t in clip_tasks if t.error is not None)
            print(f"  {clip_id[:8]}... done: {ok} ok, {fail} failed")

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

    merged_file = output_path / "typical_scenario_key_frame.json"
    with open(merged_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n  Merged {len(all_results)} clips into {merged_file}")

    # Final stats
    total_ok = 0
    total_fail = 0
    for r in all_results:
        for kfr in r.get("keyframe_results", []):
            for pl in kfr.get("prompt_labels", []):
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
    # Group by keyframe
    kf_groups: dict[int, list[Task]] = {}
    for t in tasks:
        kf_groups.setdefault(t.keyframe_index, []).append(t)

    # Load keyframe metadata for timestamp info
    keyframe_results = []
    for kf_index in sorted(kf_groups.keys()):
        group = kf_groups[kf_index]
        timestamp_us = group[0].timestamp_us
        prompt_labels = []
        for t in group:
            entry = {"prompt_name": t.prompt_name}
            if t.result is not None:
                entry["label"] = t.result
            elif t.error is not None:
                entry["label"] = {"error": t.error}
                entry["retries"] = t.retries
            prompt_labels.append(entry)
        keyframe_results.append({
            "keyframe_index": kf_index,
            "timestamp_us": timestamp_us,
            "prompt_labels": prompt_labels,
        })

    return {
        "clip_id": clip_id,
        "chunk_id": chunk_name,
        "keyframe_results": keyframe_results,
    }


if __name__ == "__main__":
    main()
