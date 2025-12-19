"""K-means clustering algorithm for task 1."""
import json
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import Dict, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch

from data_loader import ClusterDataset, Sample
from viz import visualize_clustering

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


def extract_features(dataset: ClusterDataset) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features from dataset using raw pixel values.
    
    Args:
        dataset: Dataset to extract features from
        
    Returns:
        Tuple of (features, true_labels)
    """
    features_list = []
    labels_list = []
    
    # Use raw pixel values
    for image, label in dataset:
        features_list.append(image.numpy().flatten())
        labels_list.append(label)
    
    features = np.array(features_list)
    labels = np.array(labels_list)
    
    return features, labels


def evaluate_clustering(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    n_clusters: int
) -> Dict[str, float]:
    """
    Evaluate clustering results.
    
    Args:
        true_labels: True cluster labels
        pred_labels: Predicted cluster labels
        n_clusters: Number of clusters
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Adjusted Rand Index (ARI)
    ari = adjusted_rand_score(true_labels, pred_labels)
    
    # Normalized Mutual Information (NMI)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    
    # Accuracy (requires mapping clusters to true labels)
    # Find best mapping between clusters and true labels using greedy assignment
    # Build confusion matrix
    confusion_matrix = np.zeros((n_clusters, n_clusters), dtype=np.int64)
    for true_label, pred_label in zip(true_labels, pred_labels):
        confusion_matrix[true_label, pred_label] += 1
    
    # Greedy assignment: assign each predicted cluster to the true cluster with most overlap
    used_true = set()
    used_pred = set()
    total_correct = 0
    
    # Sort by overlap size (descending)
    overlaps = []
    for i in range(n_clusters):
        for j in range(n_clusters):
            overlaps.append((confusion_matrix[i, j], i, j))
    overlaps.sort(reverse=True)
    
    # Greedy assignment
    for overlap, true_idx, pred_idx in overlaps:
        if true_idx not in used_true and pred_idx not in used_pred:
            total_correct += overlap
            used_true.add(true_idx)
            used_pred.add(pred_idx)
    
    accuracy = total_correct / len(true_labels)
    
    return {
        'ARI': ari,
        'NMI': nmi,
        'Accuracy': accuracy
    }



def run_kmeans(
    dataset_dir: str = "datasets/dataset",
    labels_path: str = "datasets/cluster_labels.json",
    n_clusters: int = 6,
    max_iters: int = 300,
    random_state: int = 42,
    n_init: int = 10,
    visualize: bool = True,
    output_dir: str = "visualizations",
    reduction_method: str = "pca"
) -> Dict:
    """
    Run K-means clustering on the dataset using scikit-learn.
    
    Args:
        dataset_dir: Directory containing images
        labels_path: Path to cluster labels JSON file
        n_clusters: Number of clusters
        max_iters: Maximum iterations for k-means
        random_state: Random seed
        n_init: Number of times k-means will be run with different centroid seeds
        
    Returns:
        Dictionary containing clustering results and metrics
    """
    print("=" * 60)
    print("K-means Clustering (scikit-learn)")
    print("=" * 60)
    
    # Load dataset (use all data for clustering, no train/test split)
    print("\nLoading dataset...")
    dataset_dir = Path(dataset_dir)
    labels_path = Path(labels_path)
    
    # Load labels
    with labels_path.open("r", encoding="utf-8") as f:
        labels_map = json.load(f)
    
    # Build samples
    class_to_idx = {}
    samples = []
    for file_name, class_name in labels_map.items():
        image_path = dataset_dir / file_name
        if not image_path.exists():
            raise FileNotFoundError(f"Image {image_path} missing for label entry.")
        label_idx = class_to_idx.setdefault(class_name, len(class_to_idx))
        samples.append(Sample(image_path, label_idx))
    
    dataset = ClusterDataset(samples)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of classes: {len(class_to_idx)}")
    print(f"Classes: {list(class_to_idx.keys())}")
    
    # Extract features (raw pixel values)
    print("\nExtracting features (raw pixel values)...")
    features, true_labels = extract_features(dataset)
    print(f"Feature shape: {features.shape}")
    
    # Normalize features
    print("\nNormalizing features...")
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    # Run K-means using scikit-learn
    print(f"\nRunning K-means with {n_clusters} clusters...")
    kmeans = KMeans(
        n_clusters=n_clusters,
        max_iter=max_iters,
        random_state=random_state,
        n_init=n_init,
        init='k-means++'
    )
    kmeans.fit(features)
    pred_labels = kmeans.labels_
    
    print(f"Number of iterations: {kmeans.n_iter_}")
    print(f"Inertia (within-cluster sum of squares): {kmeans.inertia_:.2f}")
    
    # Evaluate results
    print("\nEvaluating clustering results...")
    metrics = evaluate_clustering(true_labels, pred_labels, n_clusters)
    
    print("\n" + "=" * 60)
    print("Clustering Results")
    print("=" * 60)
    print(f"Adjusted Rand Index (ARI): {metrics['ARI']:.4f}")
    print(f"Normalized Mutual Information (NMI): {metrics['NMI']:.4f}")
    print(f"Accuracy: {metrics['Accuracy']:.4f}")
    print("=" * 60)
    
    # Visualize results if requested
    if visualize:
        visualize_clustering(
            features=features,
            pred_labels=pred_labels,
            true_labels=true_labels,
            dataset=dataset,
            class_to_idx=class_to_idx,
            output_dir=output_dir,
            method=reduction_method
        )
    
    return {
        'kmeans': kmeans,
        'pred_labels': pred_labels,
        'true_labels': true_labels,
        'metrics': metrics,
        'class_to_idx': class_to_idx,
        'scaler': scaler,
        'features': features,
        'dataset': dataset
    }


if __name__ == "__main__":
    # Run K-means clustering
    results = run_kmeans(
        dataset_dir="datasets/dataset",
        labels_path="datasets/cluster_labels.json",
        n_clusters=6,
        max_iters=300,
        random_state=42,
        n_init=10,
        visualize=True,
        output_dir="visualizations",
        reduction_method="pca"  # or "tsne" for t-SNE (slower but often better)
    )
