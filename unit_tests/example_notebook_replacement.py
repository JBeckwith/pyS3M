#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example replacement for DBSCAN-based aggregate detection in notebooks.

This script shows how to replace the memory-intensive DBSCAN approach
with the new image-based segmentation approach.

BEFORE (Memory-intensive DBSCAN approach):
=========================================
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
import gc

X = coords_data[['xc', 'yc']].values
loc_precision = 3 * (coords_data["xc_err"].mean() + coords_data["yc_err"].mean())
db_clusters = DBSCAN(min_samples=100, eps=loc_precision, algorithm='ball_tree')
labels = db_clusters.fit_predict(X)
# ... then filter by ConvexHull area ...


AFTER (Memory-efficient image-based approach):
==============================================
This script shows the replacement code.

Author: Claude Code / jbeckwith
Date: 2025-11-03
"""

import sys
import os
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import postprocess as _postprocess
import IOFunctions

IO = IOFunctions.IO_Functions()


def process_bacterial_aggregates(
    image_folder,
    min_area_um2=3.1,
    min_localizations=100,
    oversampling=8,
    threshold_method="li",
):
    """
    Memory-efficient aggregate detection for bacterial imaging data.

    This replaces the DBSCAN + ConvexHull approach with image-based segmentation.

    Parameters
    ----------
    image_folder : str
        Path to folder containing color_clustered.h5 file(s)
    min_area_um2 : float
        Minimum aggregate area in square microns (default: 3.1)
    min_localizations : int
        Minimum localizations per aggregate (default: 100)
    oversampling : int
        Super-resolution rendering oversampling (default: 8)
    threshold_method : str
        Thresholding method: 'otsu', 'li', or 'percentile' (default: 'li')

    Returns
    -------
    None (saves files to disk)
    """
    import HelperFunctions

    H_F = HelperFunctions.Helper_Functions()

    # Find all color_clustered files
    analysis_files = H_F.file_search(image_folder, "colour_clustered.h5", "")

    if len(analysis_files) == 0:
        print(f"No colour_clustered.h5 files found in {image_folder}")
        return

    print(f"Found {len(analysis_files)} file(s) to process\n")

    for file_idx, file in enumerate(analysis_files, 1):
        print("=" * 70)
        print(f"Processing file {file_idx}/{len(analysis_files)}")
        print(f"File: {os.path.basename(file)}")
        print("=" * 70)

        # Load data
        corrected_locs = pd.read_hdf(file)
        print(f"Loaded {len(corrected_locs)} localizations")

        # Estimate image dimensions from data
        width = int(np.ceil(corrected_locs["xc"].max())) + 10
        height = int(np.ceil(corrected_locs["yc"].max())) + 10
        print(f"Image dimensions: {width} x {height} pixels")

        # Run memory-efficient segmentation
        print("\nRunning image-based segmentation...")
        aggregate_locs, per_aggregate_stats = _postprocess.segment_locs_by_rendered_image(
            corrected_locs,
            width=width,
            height=height,
            oversampling=oversampling,
            min_area_um2=min_area_um2,
            min_localizations=min_localizations,
            threshold_method=threshold_method,
            callback="console",
        )

        # Rename columns to match expected output format
        aggregate_locs_output = aggregate_locs.rename(
            columns={"aggregate_id": "cluster_id", "aggregate_area_nm2": "cluster_area_nm2"}
        )
        per_aggregate_stats_output = per_aggregate_stats.rename(
            columns={"aggregate_id": "cluster_id"}
        )

        # Save results
        base_path = file.split(".h5")[0]
        aggregatelocs_path = base_path + "_aggregatelocs.h5"
        avgstats_path = base_path + "_averagedoveraggregates.h5"

        IO._write_h5_database(
            df=aggregate_locs_output,
            filepath=aggregatelocs_path,
            append=False,
            normalise_photons=False,
        )

        IO._write_h5_database(
            df=per_aggregate_stats_output,
            filepath=avgstats_path,
            append=False,
            normalise_photons=False,
        )

        print(f"\n✓ Saved results:")
        print(f"  {os.path.basename(aggregatelocs_path)}")
        print(f"  {os.path.basename(avgstats_path)}")

        # Print summary statistics
        print(f"\nSummary:")
        print(f"  Total aggregates: {len(per_aggregate_stats)}")
        print(
            f"  Localizations in aggregates: {len(aggregate_locs)} "
            f"({100 * len(aggregate_locs) / len(corrected_locs):.1f}%)"
        )
        print(
            f"  Mean area: {per_aggregate_stats['area_nm2'].mean() / 1e6:.2f} µm² "
            f"(range: {per_aggregate_stats['area_nm2'].min() / 1e6:.2f} - "
            f"{per_aggregate_stats['area_nm2'].max() / 1e6:.2f} µm²)"
        )
        print(
            f"  Mean locs/aggregate: {per_aggregate_stats['n_localizations'].mean():.0f} "
            f"(range: {per_aggregate_stats['n_localizations'].min()} - "
            f"{per_aggregate_stats['n_localizations'].max()})"
        )
        print()


def example_notebook_cell():
    """
    Example code to replace the DBSCAN cell in your notebook.

    Copy and paste this into your Jupyter notebook cell.
    """

    example_code = """
# Memory-efficient aggregate detection (replaces DBSCAN approach)
from src import postprocess as _postprocess

# Define parameters
MIN_AREA_UM2 = 3.1  # Minimum area threshold in µm²
MIN_LOCS = 100      # Minimum localizations per aggregate

# Load your data (adjust path as needed)
analysis_files = H_F.file_search(image_folder, "colour_clustered.h5", "")
file = analysis_files[0]  # Or loop over all files
corrected_locs = pd.read_hdf(file)

# Get image dimensions
width = int(np.ceil(corrected_locs["xc"].max())) + 10
height = int(np.ceil(corrected_locs["yc"].max())) + 10

# Run segmentation (MUCH more memory efficient than DBSCAN!)
aggregate_locs, per_aggregate_stats = _postprocess.segment_locs_by_rendered_image(
    corrected_locs,
    width=width,
    height=height,
    oversampling=8,
    min_area_um2=MIN_AREA_UM2,
    min_localizations=MIN_LOCS,
    threshold_method='li',  # or 'otsu', 'percentile'
    callback='console'
)

# Rename columns to match old format
aggregate_locs = aggregate_locs.rename(
    columns={'aggregate_id': 'cluster_id', 'aggregate_area_nm2': 'cluster_area_nm2'}
)
per_aggregate_stats = per_aggregate_stats.rename(
    columns={'aggregate_id': 'cluster_id'}
)

# Save results
IO._write_h5_database(
    df=aggregate_locs,
    filepath=file.split('.h5')[0] + '_aggregatelocs.h5',
    append=False,
    normalise_photons=False
)

IO._write_h5_database(
    df=per_aggregate_stats,
    filepath=file.split('.h5')[0] + '_averagedoveraggregates.h5',
    append=False,
    normalise_photons=False
)

print(f"Found {len(per_aggregate_stats)} aggregates with {len(aggregate_locs)} localizations")
"""

    print("EXAMPLE NOTEBOOK CELL CODE:")
    print("=" * 70)
    print(example_code)
    print("=" * 70)


if __name__ == "__main__":
    # Show example code
    example_notebook_cell()

    # Example usage (uncomment and adjust path):
    # process_bacterial_aggregates(
    #     image_folder="/path/to/your/data/folder",
    #     min_area_um2=3.1,
    #     min_localizations=100
    # )
