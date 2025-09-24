#!/usr/bin/env python3
"""
Test script to verify 2σ Gaussian pre-filtering logic for DBSCAN optimization.
"""
import numpy as np
import matplotlib.pyplot as plt

def test_gaussian_prefilter():
    """Test the 2σ Gaussian pre-filtering logic."""

    # Create test data with a cluster and outliers
    np.random.seed(42)

    # Main cluster (Gaussian)
    n_main = 1000
    x_main = np.random.normal(10, 2, n_main)
    y_main = np.random.normal(15, 1.5, n_main)

    # Add some outliers
    n_outliers = 100
    x_outliers = np.random.uniform(0, 25, n_outliers)
    y_outliers = np.random.uniform(0, 25, n_outliers)

    # Combine data
    X = np.vstack([
        np.concatenate([x_main, x_outliers]),
        np.concatenate([y_main, y_outliers])
    ]).T

    print(f"Created test data: {len(X)} total points ({n_main} main + {n_outliers} outliers)")

    # Apply 2σ Gaussian pre-filtering (same logic as in DriftCorrectionFunctions.py)
    mean_x, mean_y = np.mean(X[:, 0]), np.mean(X[:, 1])
    std_x, std_y = np.std(X[:, 0]), np.std(X[:, 1])

    print(f"Statistics: mean=({mean_x:.2f}, {mean_y:.2f}), std=({std_x:.2f}, {std_y:.2f})")

    # Apply 2σ elliptical filter: (dx/σx)² + (dy/σy)² <= 4.0
    if std_x > 0 and std_y > 0:
        dx = (X[:, 0] - mean_x) / std_x
        dy = (X[:, 1] - mean_y) / std_y
        elliptical_distance = dx**2 + dy**2
        within_2sigma = elliptical_distance <= 4.0  # 2σ threshold

        # Filter the data
        X_filtered = X[within_2sigma]
        outlier_mask = ~within_2sigma

        print(f"Pre-filtering: {len(X)} → {len(X_filtered)} points (removed {np.sum(outlier_mask)} outliers)")

        # Plot results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Original data
        ax1.scatter(X[:, 0], X[:, 1], c='blue', alpha=0.6, s=2)
        ax1.scatter(X[outlier_mask, 0], X[outlier_mask, 1], c='red', alpha=0.8, s=10, label=f'Outliers ({np.sum(outlier_mask)})')
        ax1.set_title('Original Data')
        ax1.legend()
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.grid(True, alpha=0.3)

        # Filtered data
        ax2.scatter(X_filtered[:, 0], X_filtered[:, 1], c='green', alpha=0.6, s=2, label=f'Filtered ({len(X_filtered)})')
        ax2.set_title('After 2σ Gaussian Pre-filtering')
        ax2.legend()
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('gaussian_prefilter_test.png', dpi=150)
        plt.show()

        # Calculate efficiency
        outlier_removal_rate = np.sum(outlier_mask) / n_outliers
        main_cluster_retention = np.sum(within_2sigma[:n_main]) / n_main

        print(f"✅ Outlier removal rate: {outlier_removal_rate:.1%}")
        print(f"✅ Main cluster retention: {main_cluster_retention:.1%}")

        return True
    else:
        print("❌ No variation in data")
        return False

if __name__ == "__main__":
    test_gaussian_prefilter()