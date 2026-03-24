#!/usr/bin/env python3
"""
Cluster trajectories using K-means clustering.

This script:
1. Loads extracted trajectories from parquet files (updated format from 1_extract_trajectories.py)
2. Extracts semantic features (38 dimensions) for driving behavior classification
3. Clusters trajectories using K-means
4. Visualizes results and analyzes cluster characteristics

Target behaviors: Straight, Left Turn, Right Turn, Left Lane Change, Right Lane Change, U-Turn

Usage:
    # Cluster trajectories (default 12 clusters)
    python3 auto_labeling/traj_cluster/2_cluster_trajectories.py

    # Specify number of clusters
    python3 auto_labeling/traj_cluster/2_cluster_trajectories.py --n-clusters 10

    # Find optimal K using silhouette analysis
    python3 auto_labeling/traj_cluster/2_cluster_trajectories.py --find-optimal-k
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm


def load_env(env_path: Path) -> dict:
    """Load environment variables from .env file."""
    env_vars = {}
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
    return env_vars


def load_trajectories(trajectories_dir: Path) -> pd.DataFrame:
    """
    Load all trajectory data from parquet files.

    The new format stores all trajectories in trajectories.{chunk_id}/trajectories.parquet.

    Args:
        trajectories_dir: Directory containing trajectory parquet files
            (e.g., data_dir/labels/trajectories/)

    Returns:
        Combined DataFrame with all trajectories
    """
    all_trajectories = []

    # Find all trajectory files (new format: trajectories.{chunk_id}/trajectories.parquet)
    traj_files = sorted(trajectories_dir.glob("trajectories.*/trajectories.parquet"))

    print(f"Found {len(traj_files)} trajectory files")

    if not traj_files:
        print("No trajectory files found! Please run 1_extract_trajectories.py first.")
        return pd.DataFrame()

    for f in traj_files:
        df = pd.read_parquet(f)
        all_trajectories.append(df)

    combined_df = pd.concat(all_trajectories, ignore_index=True)
    print(f"Loaded {len(combined_df)} total trajectories")

    return combined_df


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi] range."""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


def trajectory_to_features(trajectory: Dict) -> np.ndarray:
    """Extract trajectory features: all 17 points (dx, dy)."""
    dx = np.array(trajectory['dx'])
    dy = np.array(trajectory['dy'])
    n = len(dx)

    if n == 0:
        return np.zeros(34)

    features = []
    for i in range(n):
        features.extend([dx[i], dy[i]])

    return np.array(features)


def prepare_feature_matrix(
    trajectories_df: pd.DataFrame,
    scaler: Optional[StandardScaler] = None
) -> Tuple[np.ndarray, StandardScaler]:
    """
    Convert trajectories to feature matrix for clustering.

    Args:
        trajectories_df: DataFrame with trajectory data
        scaler: Optional fitted scaler for consistent preprocessing

    Returns:
        Tuple of (feature_matrix, scaler)
    """
    print("Extracting features from trajectories...")

    features_list = []
    for _, row in tqdm(trajectories_df.iterrows(), total=len(trajectories_df), desc="Extracting features"):
        traj = {
            'dx': row['dx'],
            'dy': row['dy'],
        }
        features = trajectory_to_features(traj)
        features_list.append(features)

    feature_matrix = np.array(features_list)
    print(f"Feature matrix shape: {feature_matrix.shape}")

    # Handle NaN/Inf values
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    # Scale features
    if scaler is None:
        scaler = StandardScaler()
        feature_matrix_scaled = scaler.fit_transform(feature_matrix)
    else:
        feature_matrix_scaled = scaler.transform(feature_matrix)

    return feature_matrix_scaled, scaler


def find_optimal_clusters(
    feature_matrix: np.ndarray,
    max_k: int = 20,
    output_dir: Optional[Path] = None
) -> int:
    """
    Find optimal number of clusters using silhouette analysis.

    Args:
        feature_matrix: Scaled feature matrix
        max_k: Maximum number of clusters to try
        output_dir: Directory to save silhouette plot

    Returns:
        Optimal number of clusters
    """
    print(f"\nFinding optimal K (max {max_k})...")

    silhouette_scores = []
    k_range = range(2, max_k + 1)

    for k in tqdm(k_range, desc="Computing silhouette scores"):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(feature_matrix)
        score = silhouette_score(feature_matrix, labels)
        silhouette_scores.append(score)
        print(f"  K={k}: Silhouette={score:.4f}")

    # Find optimal K (maximum silhouette score)
    optimal_k = k_range[np.argmax(silhouette_scores)]

    # Plot silhouette scores
    if output_dir:
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, silhouette_scores, 'bo-')
        plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal K={optimal_k}')
        plt.xlabel('Number of clusters (K)')
        plt.ylabel('Silhouette score')
        plt.title('Silhouette analysis for optimal K')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / 'silhouette_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nSilhouette plot saved to {output_dir / 'silhouette_analysis.png'}")

    print(f"\nOptimal K = {optimal_k}")

    return optimal_k


def cluster_trajectories(
    feature_matrix: np.ndarray,
    n_clusters: int
) -> np.ndarray:
    """Cluster trajectories using K-means."""
    print(f"\nClustering with K-means (K={n_clusters})...")

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(feature_matrix)

    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"\nFound {len(unique_labels)} clusters:")
    for label, count in zip(unique_labels, counts):
        print(f"  Cluster {label}: {count} trajectories ({count/len(labels)*100:.1f}%)")

    return labels


def visualize_clusters(
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    output_dir: Path
):
    """
    Visualize clusters using PCA.

    Args:
        feature_matrix: Feature matrix
        labels: Cluster labels
        output_dir: Directory to save plots
    """
    print("\nGenerating visualizations...")

    # PCA to 2D for visualization
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(feature_matrix)

    # Plot clusters
    plt.figure(figsize=(12, 10))

    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, max(20, n_clusters)))

    for label, color in zip(unique_labels, colors):
        mask = labels == label
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1],
                   c=[color], label=f'Cluster {label}', alpha=0.6, s=20)

    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
    plt.title('Trajectory Clusters (PCA visualization)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'clusters_pca.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"PCA visualization saved to {output_dir / 'clusters_pca.png'}")


def visualize_sample_trajectories(
    trajectories_df: pd.DataFrame,
    labels: np.ndarray,
    output_dir: Path,
    n_samples_per_cluster: int = 1000
):
    """
    Visualize sample trajectories from each cluster.

    Clusters are sorted by size (largest first).

    Args:
        trajectories_df: DataFrame with trajectory data
        labels: Cluster labels
        output_dir: Directory to save plots
        n_samples_per_cluster: Number of sample trajectories per cluster
    """
    print("\nGenerating sample trajectory plots...")

    trajectories_df = trajectories_df.copy()
    trajectories_df['cluster'] = labels

    # Sort clusters by size (largest first)
    cluster_sizes = pd.Series(labels).value_counts().sort_values(ascending=False)
    unique_clusters = cluster_sizes.index.tolist()
    n_clusters = len(unique_clusters)

    n_cols = 4
    n_rows = (n_clusters + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
    fig.patch.set_facecolor('#f0f0f0')
    axes_flat = axes.flatten() if n_clusters > 1 else [axes]

    for idx, cluster_id in enumerate(unique_clusters):
        ax = axes_flat[idx]
        ax.set_facecolor('#ffffff')

        cluster_df = trajectories_df[trajectories_df['cluster'] == cluster_id]

        # Sample trajectories
        if len(cluster_df) > n_samples_per_cluster:
            sample_df = cluster_df.sample(n_samples_per_cluster, random_state=42)
        else:
            sample_df = cluster_df

        # Plot each trajectory
        for _, row in sample_df.iterrows():
            dx = np.array(row['dx'])
            dy = np.array(row['dy'])
            ax.plot(dy, dx, alpha=0.4, linewidth=1, color='#1f77b4')

        # Mark start point
        ax.scatter([0], [0], c='#2ca02c', s=70, marker='o', zorder=5,
                  edgecolors='black', linewidths=1.5, label='Start')

        # Title with cluster info
        ax.set_title(f'Cluster {cluster_id} (n={len(cluster_df)})',
                    fontsize=10, fontweight='bold', color='black')

        ax.set_xlabel('Lateral Y (m)', color='black')
        ax.set_ylabel('Longitudinal X (m)', color='black')
        ax.tick_params(axis='x', colors='black')
        ax.tick_params(axis='y', colors='black')
        ax.grid(True, alpha=0.3, color='#cccccc', linestyle='--')
        ax.axis('equal')
        ax.legend(loc='upper right', fontsize=7)

    # Hide unused subplots
    for idx in range(len(unique_clusters), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / 'sample_trajectories.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Sample trajectories saved to {output_dir / 'sample_trajectories.png'}")


def analyze_cluster_characteristics(
    trajectories_df: pd.DataFrame,
    labels: np.ndarray,
    output_dir: Path
):
    """
    Analyze trajectory-based characteristics of each cluster.
    Results are sorted by cluster size (largest first).
    """
    print("\nAnalyzing cluster characteristics...")

    trajectories_df = trajectories_df.copy()
    trajectories_df['cluster'] = labels

    # Get cluster sizes and sort
    cluster_sizes = pd.Series(labels).value_counts()
    sorted_cluster_ids = cluster_sizes.sort_values(ascending=False).index.tolist()

    cluster_stats = []

    for cluster_id in sorted_cluster_ids:
        cluster_df = trajectories_df[trajectories_df['cluster'] == cluster_id]

        stats = {
            'cluster': cluster_id,
            'count': len(cluster_df),
            'percentage': len(cluster_df) / len(trajectories_df) * 100,
        }

        # Sample for analysis
        sample_df = cluster_df if len(cluster_df) <= 500 else cluster_df.sample(500, random_state=42)

        # Position statistics
        final_dx = []
        final_dy = []
        max_lateral = []
        path_lengths = []

        for _, row in sample_df.iterrows():
            dx = np.array(row['dx'])
            dy = np.array(row['dy'])

            final_dx.append(dx[-1])
            final_dy.append(dy[-1])
            max_lateral.append(np.max(np.abs(dy)))

            path_len = np.sum(np.sqrt(np.diff(dx)**2 + np.diff(dy)**2)) if len(dx) > 1 else 0
            path_lengths.append(path_len)

        stats['final_dx_mean'] = np.mean(final_dx)
        stats['final_dy_mean'] = np.mean(final_dy)
        stats['max_lateral_mean'] = np.mean(max_lateral)
        stats['path_length_mean'] = np.mean(path_lengths)

        cluster_stats.append(stats)

    stats_df = pd.DataFrame(cluster_stats)

    # Format for display
    display_df = stats_df.copy()
    for col in display_df.columns:
        if col.endswith('_mean'):
            display_df[col] = display_df[col].round(2)
        elif col == 'percentage':
            display_df[col] = display_df[col].round(1)

    stats_df.to_csv(output_dir / 'cluster_characteristics.csv', index=False)

    # Also save as markdown
    with open(output_dir / 'cluster_characteristics.md', 'w') as f:
        f.write("# Cluster Characteristics\n\n")
        markdown_table = display_df.to_markdown(index=False)
        if markdown_table:
            f.write(markdown_table)
        else:
            f.write(display_df.to_string(index=False))

        # Add column explanations
        f.write("\n## Column Explanations\n\n")
        f.write("| Column | Description |\n")
        f.write("|--------|-------------|\n")
        f.write("| cluster | Cluster ID |\n")
        f.write("| count | Number of trajectories |\n")
        f.write("| percentage | Percentage of total |\n")
        f.write("| final_dx_mean | Average final longitudinal position (m) |\n")
        f.write("| final_dy_mean | Average final lateral position (m) |\n")
        f.write("| max_lateral_mean | Average max absolute lateral position (m) |\n")
        f.write("| path_length_mean | Average path length (m) |\n")

    # Save a simplified summary
    with open(output_dir / 'cluster_summary.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CLUSTER SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        for _, row in stats_df.iterrows():
            f.write(f"Cluster {row['cluster']} ({row['count']} trajectories, {row['percentage']:.1f}%)\n")
            f.write(f"  Final Position: dx={row['final_dx_mean']:.1f}m, dy={row['final_dy_mean']:.1f}m\n")
            f.write(f"  Max Lateral: {row['max_lateral_mean']:.1f}m\n")
            f.write(f"  Path Length: {row['path_length_mean']:.1f}m\n")
            f.write("\n")

    print("\nCluster characteristics:")
    print(display_df.to_string())
    print(f"\nSaved to:")
    print(f"  - CSV: {output_dir / 'cluster_characteristics.csv'}")
    print(f"  - Markdown: {output_dir / 'cluster_characteristics.md'}")
    print(f"  - Summary: {output_dir / 'cluster_summary.txt'}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Cluster trajectories using K-means',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--n-clusters',
        type=int,
        default=12,
        help='Number of clusters for K-means (default: 12)'
    )
    parser.add_argument(
        '--pca-components',
        type=int,
        default=None,
        help='Apply PCA reduction before clustering (default: no PCA)'
    )
    parser.add_argument(
        '--find-optimal-k',
        action='store_true',
        help='Find optimal K using silhouette analysis'
    )
    parser.add_argument(
        '--max-k',
        type=int,
        default=20,
        help='Maximum K to try for silhouette analysis (default: 20)'
    )
    args = parser.parse_args()

    # Load environment
    script_dir = Path(__file__).parent.parent.parent
    env_path = script_dir / ".env"
    env_vars = load_env(env_path)

    data_dir = Path(env_vars.get('PHYSICAL_AI_AV_DATA_DIR',
                                   '/home/xingao/data/PhysicalAI-Autonomous-Vehicles-base-wo-lidar-radar'))
    trajectories_dir = data_dir / "labels" / "trajectories"
    output_dir = trajectories_dir / "clustering_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("Trajectory Clustering Analysis")
    print("="*80)
    print(f"Data directory: {data_dir}")
    print(f"Trajectories directory: {trajectories_dir}")
    print(f"Output directory: {output_dir}")
    print("="*80)

    # Load trajectories
    print("\nLoading trajectories...")
    trajectories_df = load_trajectories(trajectories_dir)

    if trajectories_df.empty:
        print("No trajectories found. Please run trajectory extraction first:")
        print("  python3 auto_labeling/traj_cluster/1_extract_trajectories.py")
        return

    print(f"\nTotal trajectories: {len(trajectories_df)}")

    # Extract features
    feature_matrix, _ = prepare_feature_matrix(trajectories_df)

    # Setup PCA if requested
    fitted_pca = None
    if args.pca_components:
        print(f"\nApplying PCA reduction: {feature_matrix.shape[1]} → {args.pca_components} dimensions...")
        pca_for_clustering = PCA(n_components=args.pca_components, random_state=42)
        feature_matrix = pca_for_clustering.fit_transform(feature_matrix)
        fitted_pca = pca_for_clustering
        total_var = fitted_pca.explained_variance_ratio_.sum()
        print(f"PCA explains {total_var:.2%} of variance")

    # Find optimal K if requested
    if args.find_optimal_k:
        n_clusters = find_optimal_clusters(feature_matrix, args.max_k, output_dir)
        print(f"\nUsing optimal K = {n_clusters}")
    else:
        n_clusters = args.n_clusters

    # Cluster trajectories
    print("\n" + "="*80)
    print("Clustering trajectories...")
    print("="*80)
    labels = cluster_trajectories(feature_matrix, n_clusters)

    # Visualize
    visualize_sample_trajectories(trajectories_df, labels, output_dir)
    visualize_clusters(feature_matrix, labels, output_dir)

    # Analyze
    analyze_cluster_characteristics(trajectories_df, labels, output_dir)

    # Save results
    trajectories_clustered = trajectories_df.copy()
    trajectories_clustered['cluster'] = labels
    trajectories_clustered.to_parquet(output_dir / 'trajectories_clustered.parquet', index=False)

    # Summary
    print(f"\n{'='*80}")
    print("✓ Clustering complete!")
    print(f"{'='*80}")
    print(f"Total trajectories: {len(trajectories_df)} clustered into {n_clusters} groups")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
