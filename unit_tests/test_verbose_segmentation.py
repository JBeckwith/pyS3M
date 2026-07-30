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

    # Create synthetic data with 3 well-separated clusters.
    # Kept at the same small scale as test_image_based_segmentation.py's
    # test_basic_segmentation (width/height=250, oversampling=8 -> 2000x2000
    # rendered image) -- this test only needs to exercise the verbose diagnostic
    # plotting code path, not a production-scale render (a naive 2500x2500 @
    # oversampling=8 version of this test previously rendered a 20000x20000-pixel
    # image, taking minutes and ~15GB RSS for no added test coverage).
    np.random.seed(42)
    n_locs_per_cluster = 300

    clusters = []

    # Cluster 1: at (50, 50) with sigma=2
    cluster1_x = np.random.normal(50, 2, n_locs_per_cluster)
    cluster1_y = np.random.normal(50, 2, n_locs_per_cluster)
    clusters.append((cluster1_x, cluster1_y))

    # Cluster 2: at (150, 150) with sigma=1.5
    cluster2_x = np.random.normal(150, 1.5, n_locs_per_cluster)
    cluster2_y = np.random.normal(150, 1.5, n_locs_per_cluster)
    clusters.append((cluster2_x, cluster2_y))

    # Cluster 3: at (100, 200) with sigma=1
    cluster3_x = np.random.normal(100, 1, n_locs_per_cluster)
    cluster3_y = np.random.normal(200, 1, n_locs_per_cluster)
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
    print(f"Image size: 250 x 250 nm")

    # Run segmentation with verbose=True
    print("\nRunning segmentation with verbose=True...")
    print("This will display 3 diagnostic plots:")
    print("  1. Rendered super-resolved image")
    print("  2. Binary thresholded mask")
    print("  3. Labeled regions (green=valid, red=rejected)")
    print()

    aggregate_locs, stats = segment_locs_by_rendered_image(
        locs,
        width=250,
        height=250,
        oversampling=8,
        pixel_size_nm=1.0,  # Treat coordinates as nanometers
        min_area_nm2=1000.0,  # 0.001 µm² = 1000 nm²
        min_localisations=50,
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
