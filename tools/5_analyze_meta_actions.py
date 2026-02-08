#!/usr/bin/env python3
"""
Analyze meta-action statistics across processed chunks.

Usage:
    python3 tools/analyze_meta_actions.py --chunks chunk_0000 chunk_0001
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict, Counter

import pandas as pd
import matplotlib.pyplot as plt


def load_annotations(output_dir: Path, chunks: list) -> list:
    """Load all meta-action annotations from specified chunks."""
    all_data = []

    for chunk_id in chunks:
        annotation_file = output_dir / f"meta_actions.{chunk_id}.json"

        if not annotation_file.exists():
            print(f"Warning: {annotation_file} not found, skipping...")
            continue

        with open(annotation_file, 'r') as f:
            chunk_data = json.load(f)
            all_data.extend(chunk_data)

        print(f"✓ Loaded {len(chunk_data)} videos from {chunk_id}")

    return all_data


def compute_statistics(all_data: list) -> dict:
    """Compute comprehensive statistics."""
    stats = {
        'total_videos': len(all_data),
        'total_frames': sum(v['num_frames'] for v in all_data),
        'total_keyframes': sum(v['num_keyframes'] for v in all_data),
        'avg_keyframes_per_video': sum(v['num_keyframes'] for v in all_data) / len(all_data),
        'compression_ratio': sum(v['num_frames'] for v in all_data) / sum(v['num_keyframes'] for v in all_data),
    }

    # Aggregate action counts
    long_actions = Counter()
    lat_actions = Counter()

    for video in all_data:
        long_actions.update(video['action_statistics']['longitudinal'])
        lat_actions.update(video['action_statistics']['lateral'])

    stats['longitudinal_distribution'] = dict(long_actions)
    stats['lateral_distribution'] = dict(lat_actions)

    return stats


def print_statistics(stats: dict, all_data: list):
    """Print formatted statistics."""
    print("\n" + "="*80)
    print("META-ACTION STATISTICS")
    print("="*80)

    print(f"\n📊 Overall Statistics:")
    print(f"  Total videos: {stats['total_videos']}")
    print(f"  Total frames: {stats['total_frames']:,}")
    print(f"  Total keyframes: {stats['total_keyframes']:,}")
    print(f"  Average keyframes per video: {stats['avg_keyframes_per_video']:.2f}")
    print(f"  Compression ratio: {stats['compression_ratio']:.2f}x")

    print(f"\n🚗 Longitudinal Meta-Actions:")
    long_total = sum(stats['longitudinal_distribution'].values())
    for action, count in sorted(stats['longitudinal_distribution'].items(), key=lambda x: -x[1]):
        pct = 100 * count / long_total
        bar = '█' * int(pct / 2)
        print(f"  {action:20s}: {count:6d} ({pct:5.1f}%) {bar}")

    print(f"\n🔄 Lateral Meta-Actions:")
    lat_total = sum(stats['lateral_distribution'].values())
    for action, count in sorted(stats['lateral_distribution'].items(), key=lambda x: -x[1]):
        pct = 100 * count / lat_total
        bar = '█' * int(pct / 2)
        print(f"  {action:20s}: {count:6d} ({pct:5.1f}%) {bar}")

    # Find interesting patterns
    print(f"\n🔍 Interesting Patterns:")

    # Videos with most keyframes
    sorted_by_kf = sorted(all_data, key=lambda v: v['num_keyframes'], reverse=True)
    print(f"  Top 5 videos by keyframe count:")
    for i, video in enumerate(sorted_by_kf[:5], 1):
        uuid_short = video['video_uuid'][:8] + '...'
        print(f"    {i}. {uuid_short}: {video['num_keyframes']} keyframes ({video['num_keyframes']/video['num_frames']*100:.1f}% of frames)")

    # Videos with most action diversity
    diversity_scores = []
    for video in all_data:
        long_diversity = len(video['action_statistics']['longitudinal'])
        lat_diversity = len(video['action_statistics']['lateral'])
        total_diversity = long_diversity + lat_diversity
        diversity_scores.append((video, total_diversity, long_diversity, lat_diversity))

    diversity_scores.sort(key=lambda x: x[1], reverse=True)
    print(f"\n  Top 5 videos by action diversity:")
    for i, (video, total, long_div, lat_div) in enumerate(diversity_scores[:5], 1):
        uuid_short = video['video_uuid'][:8] + '...'
        print(f"    {i}. {uuid_short}: {total} unique actions ({long_div} longitudinal, {lat_div} lateral)")

    print("="*80)


def plot_statistics(stats: dict, output_dir: Path):
    """Generate visualization plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Meta-Action Statistics Analysis', fontsize=16, fontweight='bold')

    # 1. Longitudinal distribution (pie)
    ax1 = axes[0, 0]
    long_data = stats['longitudinal_distribution']
    colors = plt.cm.Set3(range(len(long_data)))
    ax1.pie(long_data.values(), labels=long_data.keys(), autopct='%1.1f%%',
            colors=colors, startangle=90)
    ax1.set_title('Longitudinal Meta-Actions Distribution', fontsize=12, fontweight='bold')

    # 2. Lateral distribution (pie)
    ax2 = axes[0, 1]
    lat_data = stats['lateral_distribution']
    colors = plt.cm.Pastel1(range(len(lat_data)))
    ax2.pie(lat_data.values(), labels=lat_data.keys(), autopct='%1.1f%%',
            colors=colors, startangle=90)
    ax2.set_title('Lateral Meta-Actions Distribution', fontsize=12, fontweight='bold')

    # 3. Longitudinal distribution (bar)
    ax3 = axes[1, 0]
    long_items = sorted(long_data.items(), key=lambda x: -x[1])
    actions, counts = zip(*long_items)
    bars = ax3.barh(range(len(actions)), counts, color=plt.cm.Set3(range(len(actions))))
    ax3.set_yticks(range(len(actions)))
    ax3.set_yticklabels(actions)
    ax3.set_xlabel('Frame Count')
    ax3.set_title('Longitudinal Actions (Absolute Counts)', fontsize=12, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax3.text(count, i, f' {count:,}', va='center', fontweight='bold')

    # 4. Lateral distribution (bar)
    ax4 = axes[1, 1]
    lat_items = sorted(lat_data.items(), key=lambda x: -x[1])
    actions, counts = zip(*lat_items)
    bars = ax4.barh(range(len(actions)), counts, color=plt.cm.Pastel1(range(len(actions))))
    ax4.set_yticks(range(len(actions)))
    ax4.set_yticklabels(actions)
    ax4.set_xlabel('Frame Count')
    ax4.set_title('Lateral Actions (Absolute Counts)', fontsize=12, fontweight='bold')
    ax4.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax4.text(count, i, f' {count:,}', va='center', fontweight='bold')

    # Add summary text
    summary_text = f"Total Videos: {stats['total_videos']}\n"
    summary_text += f"Total Frames: {stats['total_frames']:,}\n"
    summary_text += f"Total Keyframes: {stats['total_keyframes']:,}\n"
    summary_text += f"Avg Keyframes/Video: {stats['avg_keyframes_per_video']:.1f}\n"
    summary_text += f"Compression: {stats['compression_ratio']:.1f}x"

    fig.text(0.98, 0.5, summary_text, fontsize=11, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, edgecolor='orange', linewidth=2),
             fontfamily='monospace', fontweight='bold')

    plt.tight_layout(rect=[0, 0, 0.85, 0.96])

    # Save figure
    output_path = output_dir / "meta_action_statistics.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved statistics plot to {output_path}")

    plt.close()


def analyze_action_sequences(all_data: list):
    """Analyze common action sequences."""
    print("\n🔄 Action Sequence Analysis:")

    # Extract action sequences for each video
    long_sequences = []
    lat_sequences = []

    for video in all_data:
        if video['keyframes']:
            long_seq = [kf['long_action'] for kf in video['keyframes']]
            lat_seq = [kf['lat_action'] for kf in video['keyframes']]

            # Extract transitions (remove consecutive duplicates)
            long_transitions = [long_seq[i] for i in range(1, len(long_seq)) if long_seq[i] != long_seq[i-1]]
            lat_transitions = [lat_seq[i] for i in range(1, len(lat_seq)) if lat_seq[i] != lat_seq[i-1]]

            long_sequences.extend(long_transitions)
            lat_sequences.extend(lat_transitions)

    # Count common transitions
    long_trans_counts = Counter(long_sequences)
    lat_trans_counts = Counter(lat_sequences)

    print(f"\n  Most common longitudinal actions (at transition points):")
    for action, count in long_trans_counts.most_common(5):
        print(f"    {action}: {count} occurrences")

    print(f"\n  Most common lateral actions (at transition points):")
    for action, count in lat_trans_counts.most_common(5):
        print(f"    {action}: {count} occurrences")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze meta-action statistics',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--chunks',
        nargs='+',
        default=None,
        help='Chunk IDs to analyze (e.g., chunk_0000)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: data_dir/labels/meta_actions)'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate visualization plots'
    )
    args = parser.parse_args()

    # Load environment
    import os

    script_dir = Path(__file__).parent.parent
    env_path = script_dir / ".env"

    def load_env(env_path: str) -> dict:
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

    env_vars = load_env(env_path)
    env_vars = load_env(str(env_path))

    data_dir = Path(env_vars.get('PHYSICAL_AI_AV_DATA_DIR',
                                   '/home/xingao/data/PhysicalAI-Autonomous-Vehicles-base-wo-lidar-radar'))
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / "labels" / "meta_actions"

    # Determine chunks to analyze
    if args.chunks:
        chunks_to_analyze = args.chunks
    else:
        # Find all available chunks
        chunk_files = sorted(output_dir.glob("meta_actions.chunk_*.json"))
        chunks_to_analyze = [f.stem.replace('meta_actions.', '') for f in chunk_files]

    if not chunks_to_analyze:
        print(f"No meta-action annotations found in {output_dir}")
        print("Run 'python3 tools/3_meta_action_annotation.py' first to generate annotations.")
        return

    print(f"\nLoading annotations from {len(chunks_to_analyze)} chunk(s)...")

    # Load data
    all_data = load_annotations(output_dir, chunks_to_analyze)

    if not all_data:
        print("No data loaded!")
        return

    # Compute statistics
    stats = compute_statistics(all_data)

    # Print statistics
    print_statistics(stats, all_data)

    # Analyze sequences
    analyze_action_sequences(all_data)

    # Generate plots
    if args.plot:
        plot_statistics(stats, output_dir)


if __name__ == "__main__":
    main()
