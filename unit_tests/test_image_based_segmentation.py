#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for memory-efficient image-based segmentation

This script demonstrates how to use segment_locs_by_rendered_image() as a
memory-efficient alternative to DBSCAN clustering.

Example usage:
    python unit_tests/test_image_based_segmentation.py

Author: Claude Code / jbeckwith
Date: 2025-11-03
"""

import sys
import os
import numpy as np
import pandas as pd

# Add src to path

from pyS3M.postprocess import segment_locs_by_rendered_image
import pyS3M.IOFunctions as IOFunctions

IO = IOFunctions.IO_Functions()


def test_basic_segmentation():
    """Test basic segmentation with synthetic data."""
    print("=" * 60)
    print("Test 1: Basic Segmentation with Synthetic Data")
    print("=" * 60)

    # Create synthetic data with 3 clusters
    np.random.seed(42)
    n_locs_per_cluster = 500

    # Cluster 1: centered at (50, 50)
    cluster1_x = np.random.normal(50, 2, n_locs_per_cluster)
    cluster1_y = np.random.normal(50, 2, n_locs_per_cluster)

    # Cluster 2: centered at (150, 150)
    cluster2_x = np.random.normal(150, 3, n_locs_per_cluster)
    cluster2_y = np.random.normal(150, 3, n_locs_per_cluster)

    # Cluster 3: centered at (100, 200)
    cluster3_x = np.random.normal(100, 1.5, n_locs_per_cluster)
    cluster3_y = np.random.normal(200, 1.5, n_locs_per_cluster)

    # Combine all clusters
    all_x = np.concatenate([cluster1_x, cluster2_x, cluster3_x])
    all_y = np.concatenate([cluster1_y, cluster2_y, cluster3_y])

    # Create localization DataFrame
    locs = pd.DataFrame(
        {
            "xc": all_x,
            "yc": all_y,
            "xc_err": np.random.uniform(0.5, 1.5, len(all_x)),
            "yc_err": np.random.uniform(0.5, 1.5, len(all_y)),
            "photons": np.random.uniform(1000, 5000, len(all_x)),
            "frame": np.random.randint(0, 1000, len(all_x)),
        }
    )

    print(f"Created {len(locs)} synthetic localizations in 3 clusters")

    # Run segmentation
    # For synthetic test data, treat each coordinate unit as 1 nm
    aggregate_locs, stats = segment_locs_by_rendered_image(
        locs,
        width=250,
        height=250,
        oversampling=8,
        pixel_size_nm=1.0,  # Treat coordinates as nanometers for test
        min_area_um2=0.001,  # 0.001 µm² = 1000 nm² (very small for test)
        min_localizations=50,  # Lower threshold for test data
        threshold_method="percentile",  # Use percentile instead of otsu for sparse data
        callback="console",
    )

    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    print(f"Total aggregates found: {len(stats)}")
    print(f"Total localizations in aggregates: {len(aggregate_locs)}")
    print(f"\nPer-aggregate statistics:")
    print(stats)

    return aggregate_locs, stats


def test_with_real_data(h5_file_path):
    """Test with real experimental data."""
    print("\n" + "=" * 60)
    print("Test 2: Segmentation with Real Data")
    print("=" * 60)

    # Load data
    print(f"Loading data from: {h5_file_path}")
    locs = pd.read_hdf(h5_file_path)

    print(f"Loaded {len(locs)} localizations")
    print(f"Columns: {list(locs.columns)}")

    # Estimate image dimensions
    width = np.ceil(locs["xc"].max()).astype(int) + 10
    height = np.ceil(locs["yc"].max()).astype(int) + 10

    print(f"Image dimensions: {width} x {height} pixels")

    # Run segmentation
    aggregate_locs, stats = segment_locs_by_rendered_image(
        locs,
        width=width,
        height=height,
        oversampling=8,
        min_area_um2=3.1,  # 3.1 µm² minimum area
        min_localizations=100,
        threshold_method="li",  # Li thresholding often works better for real data
        callback="console",
    )

    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    print(f"Total aggregates found: {len(stats)}")
    print(f"Total localizations in aggregates: {len(aggregate_locs)}")
    print(
        f"Percentage of localizations in aggregates: "
        f"{100 * len(aggregate_locs) / len(locs):.1f}%"
    )

    print(f"\nAggregate size distribution:")
    print(f"  Min area: {stats['area_nm2'].min():.1f} nm²")
    print(f"  Max area: {stats['area_nm2'].max():.1f} nm²")
    print(f"  Mean area: {stats['area_nm2'].mean():.1f} nm²")

    print(f"\nLocalizations per aggregate:")
    print(f"  Min: {stats['n_localizations'].min()}")
    print(f"  Max: {stats['n_localizations'].max()}")
    print(f"  Mean: {stats['n_localizations'].mean():.1f}")

    # Save results
    output_dir = os.path.dirname(h5_file_path)
    base_name = os.path.splitext(os.path.basename(h5_file_path))[0]

    aggregate_locs_path = os.path.join(
        output_dir, f"{base_name}_aggregatelocs.h5"
    )
    stats_path = os.path.join(output_dir, f"{base_name}_aggregatestats.h5")

    aggregate_locs.to_hdf(aggregate_locs_path, key="locs", mode="w")
    stats.to_hdf(stats_path, key="stats", mode="w")

    print(f"\nSaved results:")
    print(f"  Aggregate localizations: {aggregate_locs_path}")
    print(f"  Aggregate statistics: {stats_path}")

    return aggregate_locs, stats


def compare_with_dbscan(locs, width, height):
    """
    Compare memory usage and performance with DBSCAN.

    WARNING: This function actually runs DBSCAN - only use with small datasets!
    """
    print("\n" + "=" * 60)
    print("Test 3: Performance Comparison with DBSCAN")
    print("=" * 60)

    import time
    from sklearn.cluster import DBSCAN

    # Check if we have enough data
    if len(locs) == 0:
        print("No localizations to compare - skipping DBSCAN comparison")
        return

    # Test with image-based segmentation
    print("\n1. Image-based segmentation:")
    start_time = time.time()
    agg_locs, stats = segment_locs_by_rendered_image(
        locs,
        width=width,
        height=height,
        pixel_size_nm=1.0,  # Treat coordinates as nanometers for test
        min_area_um2=0.001,  # 0.001 µm² = 1000 nm²
        min_localizations=50,
        threshold_method="percentile",
        callback=None,  # No progress bar for timing
    )
    image_time = time.time() - start_time
    print(f"   Time: {image_time:.2f} seconds")
    print(f"   Aggregates found: {len(stats)}")

    # Test with DBSCAN
    print("\n2. DBSCAN clustering:")
    X = locs[["xc", "yc"]].values
    loc_precision = 0.5 * (locs["xc_err"].mean() + locs["yc_err"].mean())

    # Handle NaN case
    if np.isnan(loc_precision) or loc_precision <= 0:
        print("   Error: Invalid localization precision - cannot run DBSCAN")
        return

    start_time = time.time()
    db = DBSCAN(min_samples=50, eps=loc_precision)
    labels = db.fit_predict(X)
    dbscan_time = time.time() - start_time

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"   Time: {dbscan_time:.2f} seconds")
    print(f"   Clusters found: {n_clusters}")

    print(f"\n3. Comparison:")
    if image_time > 0:
        print(f"   Speedup: {dbscan_time / image_time:.1f}x faster with image-based method")
    print(
        f"   Memory: Image-based uses ~{width * height * 4 / 1024:.1f} KB "
        f"(vs DBSCAN's O(n²) distance matrix)"
    )


def main():
    """Main test function."""

    # Test 1: Synthetic data
    # Create fresh synthetic data for testing (not from the test function result)
    np.random.seed(42)
    n_locs_per_cluster = 500

    # Cluster 1: centered at (50, 50)
    cluster1_x = np.random.normal(50, 2, n_locs_per_cluster)
    cluster1_y = np.random.normal(50, 2, n_locs_per_cluster)

    # Cluster 2: centered at (150, 150)
    cluster2_x = np.random.normal(150, 3, n_locs_per_cluster)
    cluster2_y = np.random.normal(150, 3, n_locs_per_cluster)

    # Cluster 3: centered at (100, 200)
    cluster3_x = np.random.normal(100, 1.5, n_locs_per_cluster)
    cluster3_y = np.random.normal(200, 1.5, n_locs_per_cluster)

    # Combine all clusters
    all_x = np.concatenate([cluster1_x, cluster2_x, cluster3_x])
    all_y = np.concatenate([cluster1_y, cluster2_y, cluster3_y])

    # Create localization DataFrame
    test_locs = pd.DataFrame(
        {
            "xc": all_x,
            "yc": all_y,
            "xc_err": np.random.uniform(0.5, 1.5, len(all_x)),
            "yc_err": np.random.uniform(0.5, 1.5, len(all_y)),
            "photons": np.random.uniform(1000, 5000, len(all_x)),
            "frame": np.random.randint(0, 1000, len(all_x)),
        }
    )

    aggregate_locs, stats = test_basic_segmentation()

    # Test 3: Performance comparison (using fresh synthetic data)
    if len(test_locs) > 0 and len(test_locs) < 5000:  # Only compare if dataset is small
        compare_with_dbscan(test_locs, width=250, height=250)
    else:
        print("\nSkipping DBSCAN comparison - dataset too large or empty")

    # Test 2: Real data (if available)
    # Example: Uncomment and provide path to your HDF5 file
    # test_with_real_data("/path/to/your/colour_clustered.h5")


if __name__ == "__main__":
    main()
