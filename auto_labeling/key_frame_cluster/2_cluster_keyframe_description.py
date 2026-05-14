#!/usr/bin/env python3
"""
Cluster keyframe descriptions: weather, traffic_control.description, key_objects.sub_type
each clustered **separately**.

Output per field:
  - cluster_results_{field}.json  — clusters with clip_id/keyframe_index mapping
  - cluster_vis_{field}.png       — 2D visualization

Methods:
  - coarse_umap_hdbscan: 粗粒度聚类（少量大簇）
  - fine_umap_hdbscan:   细粒度聚类（多量小簇）
  - pca_kmeans:          传统 K-Means 聚类

Usage:
    python auto_labeling/key_frame_cluster/2_cluster_keyframe_description.py

    python auto_labeling/key_frame_cluster/2_cluster_keyframe_description.py \
        --method fine_umap_hdbscan --min-cluster-size 10

    python auto_labeling/key_frame_cluster/2_cluster_keyframe_description.py --compare-all
"""

import json
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import matplotlib
import argparse
import warnings

warnings.filterwarnings("ignore")

matplotlib.use("Agg")
plt.rcParams["font.sans-serif"].insert(0, "AR PL UMing CN")
plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------------------
# Data loading — extract each field separately
# ---------------------------------------------------------------------------

def load_data(data_path: str):
    """Load keyframe descriptions and return per-field sample lists with shared metadata."""
    raw = json.load(open(data_path, "r", encoding="utf-8"))

    # Each field gets its own list of texts and metadata indices
    field_data = {
        "weather": {"samples": [], "meta": []},
        "traffic_control": {"samples": [], "meta": []},
        "key_objects": {"samples": [], "meta": []},
        "road_state": {"samples": [], "meta": []},
    }

    for entry in raw:
        clip_id = entry.get("clip_id", "unknown")
        chunk_id = entry.get("chunk_id", "")
        chunk_number = int(chunk_id.replace("chunk_", "")) if chunk_id.startswith("chunk_") else -1

        for kf in entry.get("description_lists", []):
            desc = kf.get("description", {})
            if desc.get("error"):
                continue

            base_meta = {
                "clip_id": clip_id,
                "keyframe_index": kf.get("keyframe_index"),
                "timestamp_us": kf.get("timestamp_us"),
                "chunk_id": chunk_id,
                "chunk_number": chunk_number,
            }

            # --- weather ---
            w = desc.get("weather", "").strip()
            if w:
                field_data["weather"]["samples"].append(w)
                field_data["weather"]["meta"].append({**base_meta, "text": w})

            # --- traffic_control: each description as independent sample ---
            for tc in desc.get("traffic_control", []):
                d = tc.get("description", "").strip()
                if d:
                    field_data["traffic_control"]["samples"].append(d)
                    field_data["traffic_control"]["meta"].append({**base_meta, "text": d})

            # --- key_objects: each sub_type as independent sample ---
            for ko in desc.get("key_objects", []):
                st = ko.get("description", "").strip()
                if st:
                    field_data["key_objects"]["samples"].append(st)
                    field_data["key_objects"]["meta"].append({**base_meta, "text": st})

            # --- road_state: each description as independent sample ---
            for rs in desc.get("road_state", []):
                d = rs.get("description", "").strip()
                if d:
                    field_data["road_state"]["samples"].append(d)
                    field_data["road_state"]["meta"].append({**base_meta, "text": d})

    return field_data


# ---------------------------------------------------------------------------
# Clustering methods
# ---------------------------------------------------------------------------

def run_fine_umap_hdbscan(embeddings, min_cluster_size=10, umap_components=5):
    from sklearn.metrics.pairwise import cosine_distances

    n = len(embeddings)
    n_neighbors = min(15, n - 1)          # 保留局部细节 → 空间碎片化
    n_components = min(umap_components, n - 1)
    mcs = max(min_cluster_size, n // 200)
    ms = max(2, mcs // 3)                 # 低门槛 → 更容易形成簇

    # 聚类用指定维度
    reducer = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=0.05, n_components=n_components,
        metric="cosine", random_state=42,
    )
    reduced = reducer.fit_transform(embeddings)

    dist_matrix = cosine_distances(reduced).astype(np.float64)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=mcs, min_samples=ms,
        metric="precomputed", cluster_selection_method="leaf",
    )
    labels = clusterer.fit_predict(dist_matrix)

    # 可视化用 2D（避免二次降维）
    if n_components != 2:
        vis_reducer = umap.UMAP(
            n_neighbors=n_neighbors, min_dist=0.05, n_components=2,
            metric="cosine", random_state=42,
        )
        vis_coords = vis_reducer.fit_transform(embeddings)
    else:
        vis_coords = reduced

    return labels, reduced, vis_coords


def run_coarse_umap_hdbscan(embeddings, min_cluster_size=10, umap_components=5):
    from sklearn.metrics.pairwise import cosine_distances

    n = len(embeddings)
    n_neighbors = min(50, n - 1)          # 全局平滑 → 空间均匀
    n_components = min(umap_components, n - 1)
    mcs = max(min_cluster_size, n // 200)
    ms = max(3, mcs // 2)                 # 高门槛 → 更难形成簇

    # 聚类用指定维度
    reducer = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=0.1, n_components=n_components,
        metric="cosine", random_state=42,
    )
    reduced = reducer.fit_transform(embeddings)

    dist_matrix = cosine_distances(reduced).astype(np.float64)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=mcs, min_samples=ms,
        metric="precomputed", cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(dist_matrix)

    # 可视化用 2D（避免二次降维）
    if n_components != 2:
        vis_reducer = umap.UMAP(
            n_neighbors=n_neighbors, min_dist=0.1, n_components=2,
            metric="cosine", random_state=42,
        )
        vis_coords = vis_reducer.fit_transform(embeddings)
    else:
        vis_coords = reduced

    return labels, reduced, vis_coords


def run_pca_kmeans(embeddings, min_cluster_size=10, max_clusters=50):
    n = len(embeddings)
    nc = min(50, n - 1)
    pca = PCA(n_components=nc, random_state=42)
    reduced = pca.fit_transform(embeddings)

    k_min = max(3, min_cluster_size)
    k_max = min(max_clusters, n // max(min_cluster_size, 1))
    if k_max <= k_min:
        k_max = k_min + 5
    K_range = range(k_min, k_max + 1, 5)
    inertias = []
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(reduced)
        inertias.append(km.inertia_)

    if len(inertias) > 2:
        sd = np.diff(inertias, 2)
        optimal_k = list(K_range)[np.argmax(sd) + 1]
    else:
        optimal_k = k_min

    km = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    labels = km.fit_predict(reduced)

    return labels, reduced, None  # None 表示用 t-SNE 可视化


METHODS = {
    "coarse_umap_hdbscan": ("粗粒度UMAP+HDBSCAN", run_coarse_umap_hdbscan),
    "fine_umap_hdbscan": ("细粒度UMAP+HDBSCAN", run_fine_umap_hdbscan),
    "pca_kmeans": ("PCA+K-Means", run_pca_kmeans),
}


# ---------------------------------------------------------------------------
# Evaluation & visualization
# ---------------------------------------------------------------------------

def evaluate(embeddings, labels):
    unique = set(labels.tolist())
    n_clusters = len(unique) - (1 if -1 in unique else 0)
    n_noise = int(np.sum(labels == -1))
    stats = {
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_ratio": round(n_noise / len(labels) * 100, 1),
    }
    if n_clusters >= 2 and n_noise < len(labels):
        mask = labels != -1
        if len(set(labels[mask])) >= 2:
            stats["silhouette"] = round(
                float(silhouette_score(embeddings[mask], labels[mask], metric="cosine")), 4
            )
            stats["calinski_harabasz"] = round(
                float(calinski_harabasz_score(embeddings[mask], labels[mask])), 1
            )
    return stats


def visualize_2d(embeddings, labels, save_path, title="Clustering", perplexity=None, vis_coords=None):
    n = len(labels)
    if n <= 1:
        return
    unique = set(labels.tolist())
    n_clusters = len(unique) - (1 if -1 in unique else 0)

    # 如果提供了预计算的 2D 坐标，直接用；否则降维
    if vis_coords is not None:
        coords = vis_coords
    elif n > 3:
        perp = perplexity if perplexity is not None else min(30, max(5, n // 3))
        try:
            coords = TSNE(n_components=2, random_state=42, perplexity=perp,
                          max_iter=1000, learning_rate="auto").fit_transform(embeddings)
        except Exception:
            coords = PCA(n_components=2, random_state=42).fit_transform(embeddings)
    else:
        coords = embeddings[:, :2]

    fig, ax = plt.subplots(figsize=(14, 10))
    colors = plt.cm.Spectral(np.linspace(0, 1, max(len(unique), 2)))

    for i, lbl in enumerate(sorted(unique)):
        mask = labels == lbl
        if lbl == -1:
            ax.scatter(coords[mask, 0], coords[mask, 1], c="gray", marker="x",
                       alpha=0.3, s=20, label="Noise")
        else:
            ax.scatter(coords[mask, 0], coords[mask, 1], c=[colors[i % len(colors)]],
                       alpha=0.6, s=30, label=f"C{lbl} ({int(mask.sum())})")

    ax.set_title(f"{title}\n{n_clusters} clusters, {n} samples", fontsize=13)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, ncol=2)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Save cluster results
# ---------------------------------------------------------------------------

def save_results(labels, meta_list, output_path):
    clusters = defaultdict(list)
    for idx, lbl in enumerate(labels.tolist()):
        key = "noise" if lbl == -1 else f"cluster_{lbl}"
        clusters[key].append(meta_list[idx])

    sorted_clusters = dict(sorted(clusters.items(), key=lambda kv: len(kv[1]), reverse=True))
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sorted_clusters, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {output_path} ({len(sorted_clusters)} clusters)")
    return sorted_clusters


def print_summary(sorted_clusters):
    for key, members in sorted_clusters.items():
        if key == "noise":
            print(f"    [noise] {len(members)} samples")
            continue
        # count top texts
        text_counts = defaultdict(int)
        for m in members:
            text_counts[m.get("text", "")] += 1
        top = sorted(text_counts.items(), key=lambda x: -x[1])[:3]
        print(f"    [{key}] {len(members)} samples  top: {top}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        from dotenv import load_dotenv
        load_dotenv(env_path)

    default_data_dir = os.getenv(
        "ALPAMAYO_DATA_DIR",
        "/home/xingao/code/Alpamayo1.5/data/PhysicalAI-Autonomous-Vehicles",
    )
    default_input = os.path.join(default_data_dir, "labels", "key_frame_description", "kf_desc.json")
    default_output_dir = os.path.join(default_data_dir, "labels", "key_frame_description")

    parser = argparse.ArgumentParser(description="Cluster keyframe descriptions (separate per field)")
    parser.add_argument("--input", type=str, default=default_input)
    parser.add_argument("--output-dir", type=str, default=default_output_dir)
    parser.add_argument("--model", type=str, default="ckpts/bge-large-en-v1.5")
    parser.add_argument("--method", type=str, default="fine_umap_hdbscan",
                        choices=list(METHODS.keys()))
    parser.add_argument("--min-cluster-size", type=int, default=10)
    parser.add_argument("--max-clusters", type=int, default=50,
                        help="Maximum clusters for KMeans (default: 50)")
    parser.add_argument("--umap-components", type=int, default=5,
                        help="UMAP n_components for dimensionality reduction (default: 5)")
    parser.add_argument("--compare-all", action="store_true")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--tsne-perplexity", type=int, default=None,
                        help="TSNE perplexity for visualization (default: auto-calculated)")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load
    print(f"Loading: {args.input}")
    field_data = load_data(args.input)
    for field, fd in field_data.items():
        print(f"  {field}: {len(fd['samples'])} samples")

    # 2. Encode (shared model across fields)
    print(f"\nEncoding with {args.model}...")
    encoder = SentenceTransformer(args.model)
    all_embeddings = {}
    for field, fd in field_data.items():
        if not fd["samples"]:
            print(f"  {field}: no samples, skipping")
            continue
        print(f"  Encoding {field} ({len(fd['samples'])} texts)...")
        all_embeddings[field] = encoder.encode(
            fd["samples"], show_progress_bar=True, convert_to_tensor=False,
            normalize_embeddings=True
        )

    # 3. Cluster each field independently
    for field, embeddings in all_embeddings.items():
        fd = field_data[field]
        print(f"\n{'='*60}")
        print(f"Clustering field: {field} ({len(fd['samples'])} samples)")
        print(f"{'='*60}")

        if args.compare_all:
            best_method = None
            best_sil = -2
            for mk, (mname, mfunc) in METHODS.items():
                print(f"  Method: {mname}")
                labels, reduced, vis_coords = mfunc(embeddings, args.min_cluster_size,
                                                    args.max_clusters if mk == "pca_kmeans" else args.umap_components)
                stats = evaluate(embeddings, labels)
                print(f"    clusters={stats['n_clusters']}  noise={stats['n_noise']} ({stats['noise_ratio']}%)")
                if "silhouette" in stats:
                    print(f"    silhouette={stats['silhouette']}  CH={stats['calinski_harabasz']}")
                    if stats["silhouette"] > best_sil:
                        best_sil = stats["silhouette"]
                        best_method = mk

                # Save all results in compare-all mode
                result_path = os.path.join(args.output_dir, f"cluster_results_{field}_{mk}.json")
                save_results(labels, fd["meta"], result_path)
                vis_path = os.path.join(args.output_dir, f"cluster_vis_{field}_{mk}.png")
                visualize_2d(reduced, labels, vis_path, title=f"{field} — {mname}",
                            perplexity=args.tsne_perplexity, vis_coords=vis_coords)

            print(f"  Best by silhouette: {best_method} (silhouette={best_sil:.4f})")
            print(f"  All results saved. Please manually review and choose.")
            continue  # Skip the "final run" section
        else:
            chosen_method = args.method

        # Final run with chosen method
        mname, mfunc = METHODS[chosen_method]
        print(f"  Final: {mname}")
        labels, reduced, vis_coords = mfunc(embeddings, args.min_cluster_size,
                                            args.max_clusters if chosen_method == "pca_kmeans" else args.umap_components)
        stats = evaluate(embeddings, labels)
        print(f"    clusters={stats['n_clusters']}  noise={stats['n_noise']} ({stats['noise_ratio']}%)")
        if "silhouette" in stats:
            print(f"    silhouette={stats['silhouette']}  CH={stats['calinski_harabasz']}")

        # Save
        result_path = os.path.join(args.output_dir, f"cluster_results_{field}_{chosen_method}.json")
        sorted_clusters = save_results(labels, fd["meta"], result_path)

        vis_path = os.path.join(args.output_dir, f"cluster_vis_{field}_{chosen_method}.png")
        visualize_2d(reduced, labels, vis_path, title=f"{field} — {mname}",
                    perplexity=args.tsne_perplexity, vis_coords=vis_coords)

        # Summary
        print(f"  Summary:")
        print_summary(sorted_clusters)

    print(f"\nDone. Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
