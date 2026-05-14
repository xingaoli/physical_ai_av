#!/usr/bin/env python3
"""
Merge specified clusters from clustering results.

Usage:
    python auto_labeling/key_frame_cluster/3_merge_clusters.py \
        --input labels/key_frame_description/cluster_results_road_state_pca_kmeans.json \
        --merge "[[0,3,4],[5,8,9]]"

    # Or via a JSON file:
    python auto_labeling/key_frame_cluster/3_merge_clusters.py \
        --input cluster_results_road_state_pca_kmeans.json \
        --merge-file merge_plan.json

merge_plan.json example:
    [[0, 3, 4], [5, 8, 9]]
"""

import json
import argparse
import os
from collections import OrderedDict


def load_clusters(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f, object_pairs_hook=OrderedDict)


def parse_merge_groups(merge_str: str) -> list[list[int]]:
    return json.loads(merge_str)


def merge_clusters(clusters: dict, merge_groups: list[list[int]]) -> dict:
    # Build index: cluster_N -> members
    cluster_ids = set()
    for key in clusters:
        if key.startswith("cluster_"):
            cluster_ids.add(int(key.split("_", 1)[1]))

    # Track which clusters are consumed by merges
    consumed = set()
    merged_results = OrderedDict()

    for group in merge_groups:
        # Validate
        for cid in group:
            key = f"cluster_{cid}"
            if key not in clusters:
                print(f"Warning: {key} not found, skipping")
                continue
            consumed.add(cid)

        # Merge members from all clusters in the group
        merged_members = []
        for cid in group:
            key = f"cluster_{cid}"
            if key in clusters:
                merged_members.extend(clusters[key])

        # Use the first cluster id in the group as the new id
        new_key = f"cluster_{group[0]}"
        merged_results[new_key] = merged_members
        print(f"  Merged [{', '.join(f'cluster_{c}' for c in group)}] -> {new_key} ({len(merged_members)} samples)")

    # Add remaining (unmerged) clusters
    for key, members in clusters.items():
        if key == "noise":
            continue
        if key.startswith("cluster_"):
            cid = int(key.split("_", 1)[1])
            if cid not in consumed:
                merged_results[key] = members

    # Add noise at the end if present
    if "noise" in clusters:
        merged_results["noise"] = clusters["noise"]

    # Sort by member count descending
    sorted_results = OrderedDict(
        sorted(merged_results.items(), key=lambda kv: len(kv[1]), reverse=True)
    )

    return sorted_results


def main():
    parser = argparse.ArgumentParser(description="Merge clusters from clustering results")
    parser.add_argument("--input", type=str, required=True, help="Input cluster results JSON")
    parser.add_argument("--merge", type=str, default=None,
                        help='Merge groups as JSON string, e.g. "[[0,3,4],[5,8,9]]"')
    parser.add_argument("--merge-file", type=str, default=None,
                        help="Path to a JSON file containing merge groups")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: <input>_merged.json)")
    args = parser.parse_args()

    if not args.merge and not args.merge_file:
        parser.error("Must specify --merge or --merge-file")

    if args.merge and args.merge_file:
        parser.error("Specify only one of --merge or --merge-file")

    # Load
    clusters = load_clusters(args.input)
    print(f"Loaded {len(clusters)} clusters from {args.input}")

    # Parse merge groups
    if args.merge:
        merge_groups = parse_merge_groups(args.merge)
    else:
        with open(args.merge_file, "r") as f:
            merge_groups = json.load(f)

    print(f"Merge plan: {merge_groups}")

    # Merge
    merged = merge_clusters(clusters, merge_groups)

    # Output path
    if args.output:
        output_path = args.output
    else:
        base, ext = os.path.splitext(args.input)
        output_path = f"{base}_merged{ext}"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {output_path} ({len(merged)} clusters)")


if __name__ == "__main__":
    main()
