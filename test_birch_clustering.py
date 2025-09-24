#!/usr/bin/env python3
"""
Test script to verify BIRCH clustering implementation for fiducial detection.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import Birch

def test_birch_clustering():
    """Test the BIRCH clustering approach used in fiducial detection."""

    # Create test data similar to fiducial localizations
    np.random.seed(42)

    # Main cluster (fiducial)
    n_fiducial = 500
    center_x, center_y = 100, 150
    fiducial_x = np.random.normal(center_x, 2.0, n_fiducial)  # 2 nm precision
    fiducial_y = np.random.normal(center_y, 1.5, n_fiducial)  # 1.5 nm precision

    # Add some noise/outliers
    n_noise = 50
    noise_x = np.random.uniform(90, 110, n_noise)
    noise_y = np.random.uniform(140, 160, n_noise)

    # Combine data
    X = np.vstack([
        np.concatenate([fiducial_x, noise_x]),
        np.concatenate([fiducial_y, noise_y])
    ]).T

    print(f"Created test data: {len(X)} points ({n_fiducial} fiducial + {n_noise} noise)")

    # Test BIRCH with different sample sizes
    sample_sizes = [100, 300, len(X)]  # Test subsampling vs full data

    fig, axes = plt.subplots(2, len(sample_sizes), figsize=(15, 8))

    for i, sample_size in enumerate(sample_sizes):
        print(f"\n--- Testing with sample_size={sample_size} ---")

        # Sample data for training if needed
        if len(X) > sample_size:
            sample_indices = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X[sample_indices]
            print(f"Training on {sample_size} sampled points from {len(X)} total")
        else:
            X_sample = X
            print(f"Training on all {len(X)} points")

        # Configure BIRCH (similar to implementation)
        loc_precision = 2.5  # Similar to typical localization precision
        threshold_distance = loc_precision * 1.5

        birch = Birch(
            threshold=threshold_distance,
            branching_factor=50,
            n_clusters=None,
            compute_labels=True
        )

        # Train on sample
        birch.fit(X_sample)
        sample_labels = birch.labels_

        # Predict on full dataset
        all_labels = birch.predict(X)

        # Analyze results
        n_clusters = len(set(all_labels)) - (1 if -1 in all_labels else 0)
        n_noise = np.sum(all_labels == -1)

        print(f"Found {n_clusters} clusters, {n_noise} noise points")
        print(f"Noise fraction: {n_noise/len(X):.2%}")

        # Find largest cluster (main fiducial)
        if n_clusters > 0:
            cluster_sizes = [(label, np.sum(all_labels == label))
                           for label in set(all_labels) if label != -1]
            largest_cluster_label = max(cluster_sizes, key=lambda x: x[1])[0]
            main_cluster_mask = all_labels == largest_cluster_label
            main_cluster_size = np.sum(main_cluster_mask)

            print(f"Largest cluster: {main_cluster_size} points ({main_cluster_size/n_fiducial:.1%} of fiducial)")

            # Plot sample data
            ax1 = axes[0, i]
            if len(X_sample) < len(X):
                ax1.scatter(X[:, 0], X[:, 1], c='lightgray', s=5, alpha=0.3, label='Full data')

            unique_labels = set(sample_labels)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

            for label, color in zip(unique_labels, colors):
                if label == -1:
                    color = 'black'
                mask = sample_labels == label
                ax1.scatter(X_sample[mask, 0], X_sample[mask, 1],
                           c=[color], s=20, alpha=0.8,
                           label=f'Sample Cluster {label}' if label != -1 else 'Sample Noise')

            ax1.set_title(f'Sample Training (n={len(X_sample)})')
            ax1.set_xlabel('X (nm)')
            ax1.set_ylabel('Y (nm)')
            ax1.legend(fontsize=8)
            ax1.grid(True, alpha=0.3)

            # Plot full prediction results
            ax2 = axes[1, i]
            unique_labels = set(all_labels)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

            for label, color in zip(unique_labels, colors):
                if label == -1:
                    color = 'black'
                mask = all_labels == label
                ax2.scatter(X[mask, 0], X[mask, 1],
                           c=[color], s=10, alpha=0.7,
                           label=f'Cluster {label}' if label != -1 else 'Noise')

            # Highlight main cluster
            if n_clusters > 0:
                ax2.scatter(X[main_cluster_mask, 0], X[main_cluster_mask, 1],
                           s=15, facecolors='none', edgecolors='red', linewidth=1,
                           label='Main Cluster')

            ax2.set_title(f'Full Prediction (n={len(X)})')
            ax2.set_xlabel('X (nm)')
            ax2.set_ylabel('Y (nm)')
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('birch_clustering_test.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\n✅ BIRCH clustering test completed!")
    print(f"Results saved to: birch_clustering_test.png")

    return True

if __name__ == "__main__":
    test_birch_clustering()