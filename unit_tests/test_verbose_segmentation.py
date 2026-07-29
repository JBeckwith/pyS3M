#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for verbose mode of segment_locs_by_rendered_image

This script demonstrates the verbose diagnostic plotting feature.

Example usage:
    MPLBACKEND=Agg PYTHONPATH="src:$PYTHONPATH" python unit_tests/test_verbose_segmentation.py

Author: Claude Code / jbeckwith
Date: 2025-11-03
"""

import sys
import os
import numpy as np
import pandas as pd

# Add src to path

from pyS3M.postprocess import segment_locs_by_rendered_image


def test_verbose_mode():
    """Test verbose plotting with synthetic data."""
    print("=" * 70)
    print("Testing Verbose Mode with Synthetic Data")
    print("=" * 70)

    # Create synthetic data with 3 well-separated clusters
    np.random.seed(42)
    n_locs_per_cluster = 300

    # Create larger, more spread out clusters for better visibility
    clusters = []

    # Cluster 1: Large cluster at (500, 500) with sigma=20
    cluster1_x = np.random.normal(500, 20, n_locs_per_cluster)
    cluster1_y = np.random.normal(500, 20, n_locs_per_cluster)
    clusters.append((cluster1_x, cluster1_y))

    # Cluster 2: Medium cluster at (1500, 1500) with sigma=15
    cluster2_x = np.random.normal(1500, 15, n_locs_per_cluster)
    cluster2_y = np.random.normal(1500, 15, n_locs_per_cluster)
    clusters.append((cluster2_x, cluster2_y))

    # Cluster 3: Small cluster at (1000, 2000) with sigma=10
    cluster3_x = np.random.normal(1000, 10, n_locs_per_cluster)
    cluster3_y = np.random.normal(2000, 10, n_locs_per_cluster)
    clusters.append((cluster3_x, cluster3_y))

    # Combine all clusters
    all_x = np.concatenate([c[0] for c in clusters])
    all_y = np.concatenate([c[1] for c in clusters])

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

    print(f"\nCreated {len(locs)} synthetic localizations in 3 clusters")
    print(f"Image size: 2500 x 2500 nm")

    # Run segmentation with verbose=True
    print("\nRunning segmentation with verbose=True...")
    print("This will display 3 diagnostic plots:")
    print("  1. Rendered super-resolved image")
    print("  2. Binary thresholded mask")
    print("  3. Labeled regions (green=valid, red=rejected)")
    print()

    aggregate_locs, stats = segment_locs_by_rendered_image(
        locs,
        width=2500,
        height=2500,
        oversampling=8,
        pixel_size_nm=1.0,  # Treat coordinates as nanometers
        min_area_um2=0.001,  # 0.001 µm² = 1000 nm²
        min_localizations=50,
        threshold_method="li",  # Li works better for sparse data
        callback=None,  # No console output
        verbose=True,  # Enable diagnostic plots
    )

    print("\n" + "=" * 70)
    print("Results:")
    print("=" * 70)
    print(f"Total aggregates found: {len(stats)}")
    print(f"Total localizations in aggregates: {len(aggregate_locs)}")

    if len(stats) > 0:
        print(f"\nAggregate statistics:")
        print(f"  Min area: {stats['area_nm2'].min():.1f} nm²")
        print(f"  Max area: {stats['area_nm2'].max():.1f} nm²")
        print(f"  Mean area: {stats['area_nm2'].mean():.1f} nm²")
        print(f"\n  Min locs: {stats['n_localizations'].min()}")
        print(f"  Max locs: {stats['n_localizations'].max()}")
        print(f"  Mean locs: {stats['n_localizations'].mean():.1f}")

    return aggregate_locs, stats


if __name__ == "__main__":
    # Set matplotlib backend for display
    import matplotlib
    # Use TkAgg or another GUI backend if available, otherwise Agg
    try:
        matplotlib.use('TkAgg')
    except:
        try:
            matplotlib.use('Qt5Agg')
        except:
            print("Warning: No GUI backend available, using Agg (no display)")
            matplotlib.use('Agg')

    aggregate_locs, stats = test_verbose_mode()

    print("\n" + "=" * 70)
    print("Test complete!")
    print("=" * 70)
