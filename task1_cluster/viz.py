"""Visualization functions for clustering results."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from typing import Dict
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch

from data_loader import ClusterDataset

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 柔和的颜色列表
COLORS = [
    '#9bbf8a', '#82afda', '#f79059', 
    '#ffbe7a', '#fa8878', '#9573A6', 
    '#b0cba8', '#3480b8', '#8dcec8'
]


def visualize_clustering(
    features: np.ndarray,
    pred_labels: np.ndarray,
    true_labels: np.ndarray,
    dataset: ClusterDataset,
    class_to_idx: Dict[str, int],
    output_dir: str = "visualizations",
    method: str = "pca",
    n_samples_per_cluster: int = 5
):
    """
    Visualize clustering results with beautiful styling.
    
    Args:
        features: Feature matrix
        pred_labels: Predicted cluster labels
        true_labels: True labels
        dataset: Dataset object
        class_to_idx: Mapping from class names to indices
        output_dir: Directory to save visualization images
        method: Dimensionality reduction method ('pca' or 'tsne')
        n_samples_per_cluster: Number of sample images to show per cluster
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create reverse mapping
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    # Verify label mapping by checking a few samples
    print("\nVerifying label mapping...")
    print(f"  Class to index mapping: {class_to_idx}")
    print(f"  Index to class mapping: {idx_to_class}")
    
    # Check first few samples to verify labels
    print("  Checking first 5 samples:")
    for i in range(min(5, len(dataset))):
        image, label = dataset[i]
        class_name = idx_to_class.get(label, f"Unknown({label})")
        print(f"    Sample {i}: label={label}, class={class_name}")
    
    # Check label distribution
    unique_labels, counts = np.unique(true_labels, return_counts=True)
    print(f"  Label distribution:")
    for label, count in zip(unique_labels, counts):
        class_name = idx_to_class.get(label, f"Unknown({label})")
        print(f"    {class_name} (label {label}): {count} samples")
    
    print("\nGenerating visualizations...")
    
    # 1. Dimensionality reduction and scatter plot
    print("  - Creating 2D scatter plot...")
    if method == "pca":
        reducer = PCA(n_components=2, random_state=42)
        reduced_features = reducer.fit_transform(features)
        method_name = "PCA"
    else:  # tsne
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        reduced_features = reducer.fit_transform(features)
        method_name = "t-SNE"
    
    n_clusters = len(np.unique(pred_labels))
    n_classes = len(class_to_idx)
    
    # Find best mapping between predicted clusters and true classes
    # Build confusion matrix
    confusion_matrix = np.zeros((n_classes, n_clusters), dtype=np.int64)
    for true_label, pred_label in zip(true_labels, pred_labels):
        confusion_matrix[true_label, pred_label] += 1
    
    # Greedy assignment: find which true class each predicted cluster corresponds to
    cluster_to_class = {}  # Maps cluster_id -> class_id
    used_classes = set()
    used_clusters = set()
    
    # Sort by overlap size (descending)
    overlaps = []
    for i in range(n_classes):
        for j in range(n_clusters):
            overlaps.append((confusion_matrix[i, j], i, j))
    overlaps.sort(reverse=True)
    
    # Greedy assignment
    for overlap, class_idx, cluster_idx in overlaps:
        if class_idx not in used_classes and cluster_idx not in used_clusters:
            cluster_to_class[cluster_idx] = class_idx
            used_classes.add(class_idx)
            used_clusters.add(cluster_idx)
    
    # Fill remaining clusters (if any)
    for cluster_id in range(n_clusters):
        if cluster_id not in cluster_to_class:
            # Assign to the class with most overlap
            best_class = np.argmax(confusion_matrix[:, cluster_id])
            cluster_to_class[cluster_id] = best_class
    
    # Create figure with two subplots side by side
    print("  - Creating 2D scatter plot...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor('white')
    
    # Plot 1: Predicted clusters (colored by true class for comparison)
    ax = axes[0]
    # Color each point by its true class, so colors match with right plot
    # But show cluster boundaries in legend
    for cluster_id in range(n_clusters):
        cluster_mask = pred_labels == cluster_id
        # For each class in this cluster, plot separately to show true class colors
        for class_id in range(n_classes):
            # Points that are in this cluster AND have this true class
            mask = cluster_mask & (true_labels == class_id)
            if np.any(mask):
                color = COLORS[class_id % len(COLORS)]
                # Show cluster label only once per cluster (for the dominant class)
                mapped_class = cluster_to_class.get(cluster_id, -1)
                if class_id == mapped_class:
                    # Show cluster with corresponding class name
                    class_name = idx_to_class.get(mapped_class, f'Class {mapped_class}')
                    label = f'Cluster {cluster_id} ({class_name})'
                else:
                    label = None
                ax.scatter(
                    reduced_features[mask, 0], 
                    reduced_features[mask, 1], 
                    c=color,
                    alpha=1.0,  # 完全不透明
                    s=30,
                    edgecolors='white',
                    linewidths=0.5,
                    label=label
                )
    
    ax.set_title(f'Predicted Clusters ({method_name})', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel(f'{method_name} Component 1', fontsize=13)
    ax.set_ylabel(f'{method_name} Component 2', fontsize=13)
    ax.grid(False)  # 去掉网格线
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    ax.tick_params(colors='#666666')
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=10)
    
    # Plot 2: True labels
    ax = axes[1]
    for class_id in range(n_classes):
        mask = true_labels == class_id
        color = COLORS[class_id % len(COLORS)]
        ax.scatter(
            reduced_features[mask, 0], 
            reduced_features[mask, 1], 
            c=color,
            alpha=1.0,  # 完全不透明
            s=30,
            edgecolors='white',
            linewidths=0.5,
            label=idx_to_class[class_id]
        )
    
    ax.set_title(f'True Labels ({method_name})', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel(f'{method_name} Component 1', fontsize=13)
    ax.set_ylabel(f'{method_name} Component 2', fontsize=13)
    ax.grid(False)  # 去掉网格线
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')
    ax.tick_params(colors='#666666')
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=10)
    
    plt.tight_layout()
    scatter_path = output_path / f'clustering_scatter_{method}.png'
    plt.savefig(scatter_path, format='png', bbox_inches='tight', facecolor='white', dpi=300)
    plt.close()
    print(f"    Saved: {scatter_path}")
    
    # 2. Confusion matrix with better styling
    print("  - Creating confusion matrix...")
    confusion_matrix = np.zeros((len(class_to_idx), n_clusters), dtype=np.int64)
    for true_label, pred_label in zip(true_labels, pred_labels):
        confusion_matrix[true_label, pred_label] += 1
    
    # Reorder columns to match class order (so diagonal shows best alignment)
    # Create a mapping: class_id -> cluster_id
    class_to_cluster = {v: k for k, v in cluster_to_class.items()}
    
    # Create column order: for each class, find its corresponding cluster
    column_order = []
    column_labels = []
    for class_id in range(len(class_to_idx)):
        if class_id in class_to_cluster:
            cluster_id = class_to_cluster[class_id]
            column_order.append(cluster_id)
            column_labels.append(f'Cluster {cluster_id}\n({idx_to_class[class_id]})')
        else:
            # If no cluster mapped to this class, find the cluster with most overlap
            best_cluster = np.argmax(confusion_matrix[class_id, :])
            column_order.append(best_cluster)
            column_labels.append(f'Cluster {best_cluster}\n({idx_to_class[class_id]})')
    
    # Handle remaining clusters (if n_clusters > n_classes)
    used_clusters = set(column_order)
    for cluster_id in range(n_clusters):
        if cluster_id not in used_clusters:
            column_order.append(cluster_id)
            mapped_class = cluster_to_class.get(cluster_id, -1)
            if mapped_class >= 0:
                column_labels.append(f'Cluster {cluster_id}\n({idx_to_class[mapped_class]})')
            else:
                column_labels.append(f'Cluster {cluster_id}')
    
    # Reorder confusion matrix columns
    reordered_matrix = confusion_matrix[:, column_order]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('white')
    
    # Use a softer colormap (light blue)
    im = ax.imshow(reordered_matrix, cmap='Blues', aspect='auto', vmin=0, vmax=reordered_matrix.max())
    
    # Add text annotations
    for i in range(len(class_to_idx)):
        for j in range(len(column_order)):
            value = reordered_matrix[i, j]
            text_color = 'white' if value > reordered_matrix.max() / 2 else 'black'
            ax.text(j, i, value,
                   ha="center", va="center", 
                   color=text_color, 
                   fontweight='bold',
                   fontsize=11)
    
    ax.set_xticks(range(len(column_order)))
    ax.set_xticklabels(column_labels, fontsize=11, rotation=0, ha='center')
    ax.set_yticks(range(len(class_to_idx)))
    ax.set_yticklabels([idx_to_class[i] for i in range(len(class_to_idx))], fontsize=12)
    ax.set_xlabel('Predicted Cluster (with corresponding class)', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel('True Class', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_title('Confusion Matrix: True Classes vs Predicted Clusters', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.colorbar(im, ax=ax, label='Number of Samples', shrink=0.8)
    plt.tight_layout()
    confusion_path = output_path / 'confusion_matrix.png'
    plt.savefig(confusion_path, format='png', bbox_inches='tight', facecolor='white', dpi=300)
    plt.close()
    print(f"    Saved: {confusion_path}")
    
    # 3. Sample images from each cluster with better styling
    print("  - Creating sample images visualization...")
    fig, axes = plt.subplots(n_clusters, n_samples_per_cluster, 
                            figsize=(n_samples_per_cluster * 2.2, n_clusters * 2.2))
    fig.patch.set_facecolor('white')
    
    if n_clusters == 1:
        axes = axes.reshape(1, -1)
    
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(pred_labels == cluster_id)[0]
        # Randomly sample images from this cluster
        if len(cluster_indices) > n_samples_per_cluster:
            np.random.seed(42)  # For reproducibility
            sample_indices = np.random.choice(cluster_indices, n_samples_per_cluster, replace=False)
        else:
            sample_indices = cluster_indices
            # Pad with repeats if needed
            while len(sample_indices) < n_samples_per_cluster:
                sample_indices = np.concatenate([sample_indices, cluster_indices])
            sample_indices = sample_indices[:n_samples_per_cluster]
        
        for col, idx in enumerate(sample_indices):
            image, label = dataset[idx]
            # Convert tensor to numpy and adjust for display
            if isinstance(image, torch.Tensor):
                img = image.numpy()
                if img.shape[0] == 3:  # CHW format
                    img = img.transpose(1, 2, 0)
                img = np.clip(img, 0, 1)
            else:
                img = image
            
            axes[cluster_id, col].imshow(img)
            axes[cluster_id, col].axis('off')
            true_class = idx_to_class[label]
            axes[cluster_id, col].set_title(f'{true_class}', fontsize=9, pad=5)
        
        # Add cluster label on the left with better styling
        # Use the color of the corresponding true class (mapped from cluster)
        mapped_class = cluster_to_class.get(cluster_id, cluster_id % n_classes)
        cluster_color = COLORS[mapped_class % len(COLORS)]
        class_name = idx_to_class.get(mapped_class, f'Class {mapped_class}')
        axes[cluster_id, 0].text(-0.08, 0.5, f'Cluster {cluster_id}\n({class_name})', 
                                transform=axes[cluster_id, 0].transAxes,
                                rotation=90, va='center', ha='right',
                                fontsize=12, fontweight='bold',
                                color=cluster_color)
    
    plt.suptitle('Sample Images from Each Cluster', fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    samples_path = output_path / 'cluster_samples.png'
    plt.savefig(samples_path, format='png', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    Saved: {samples_path}")
    
    print(f"\nAll visualizations saved to: {output_path}")
