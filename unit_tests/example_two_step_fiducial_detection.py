#!/usr/bin/env python3
"""
Example: Three-step fiducial detection with DBSCAN clustering validation.

This demonstrates how to use the new comprehensive approach for fiducial detection:
1. detect_high_density_regions_from_image() - finds potential fiducial regions from rendered image
2. select_puncta_from_regions() - selects localizations within those regions
3. identify_real_fiducials_with_clustering() - validates real fiducials using DBSCAN clustering

This separation allows for debugging at each step, tuning parameters independently,
and filtering out noise to identify only genuine fiducial markers.
"""

import numpy as np
import sys
import os
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from DriftCorrectionFunctions import Drift_Correction_Functions


def example_workflow():
    """Complete example workflow showing the two-step fiducial detection process."""

    print("Two-Step Fiducial Detection Workflow")
    print("=" * 50)

    # Initialize the drift correction functions
    DCF = Drift_Correction_Functions()

    # Step 0: Load your data (example with synthetic data)
    print("Step 0: Preparing data...")
    locs, info, smoothed_image = create_example_data()

    print(f"  - {len(locs)} localizations loaded")
    print(f"  - Image size: {smoothed_image.shape}")
    print(f"  - Pixel size: {info[0]['Pixelsize']} nm")

    # Step 1: Detect high-density regions from smoothed/rendered image
    print("\nStep 1: Detecting high-density regions...")

    region_centers, binary_mask, threshold, density_metadata = DCF.detect_high_density_regions_from_image(
        smoothed_image=smoothed_image,
        histogram_bins=128,                    # Number of histogram bins for threshold calculation
        threshold_percentile=98.0,             # Percentile threshold (adjust based on data)
        pixelsize=info[0]['Pixelsize'],        # Pixel size for scale bars
        output_figure_path="step1_regions.png",
        title="Step 1: High-Density Region Detection",
        create_plot=True                       # Set to False to skip visualization
    )

    print(f"  ✓ Detected {len(region_centers)} potential regions")
    print(f"  ✓ Threshold used: {threshold:.3f}")
    print(f"  ✓ Region area fraction: {density_metadata['region_area_fraction']:.3f}")

    # Inspect what was detected
    if len(region_centers) == 0:
        print("  ⚠ No regions detected - try lowering threshold_percentile")
        return

    print(f"  ✓ Region centers (y, x): {region_centers[:5]}...")  # Show first 5

    # Step 2: Select puncta (localizations) from detected regions
    # Uses optimized postprocess.picked_locs for efficient rectangular selection
    print("\nStep 2: Selecting puncta from detected regions...")

    selected_puncta, selection_metadata = DCF.select_puncta_from_regions(
        locs=locs,                             # Your localization data
        region_centers=region_centers,         # Output from Step 1
        binary_mask=binary_mask,               # Output from Step 1
        pixelsize=info[0]['Pixelsize'],
        selection_box_size_nm=1000.0,         # Box size around each region center (adjust as needed)
        min_localizations_per_region=20,      # Minimum locs for valid fiducial (adjust as needed)
        output_figure_path="step2_puncta.png",
        title="Step 2: Puncta Selection",
        create_plot=True
    )

    print(f"  ✓ Selected {len(selected_puncta)} valid fiducial candidates")
    print(f"  ✓ Selection criteria applied:")
    print(f"    - Box size: {selection_metadata['selection_criteria']['selection_box_size_nm']} nm")
    print(f"    - Min locs: {selection_metadata['selection_criteria']['min_localizations']}")

    # Show rejection reasons
    reasons = selection_metadata['rejection_reasons']
    print(f"  ✓ Rejection summary:")
    print(f"    - Accepted: {reasons['accepted']}")
    print(f"    - Too few localizations: {reasons['too_few_localizations']}")

    # Step 3: Analyze results and prepare for drift correction
    print("\nStep 3: Analyzing selected fiducials...")

    if len(selected_puncta) == 0:
        print("  ⚠ No valid fiducials found - try adjusting selection criteria")
        return

    for i, (puncta, stats) in enumerate(zip(selected_puncta, selection_metadata['region_statistics'])):
        print(f"  Fiducial {i+1}:")
        print(f"    - {stats['n_localizations']} localizations")
        print(f"    - Center: ({stats['center_x']:.1f}, {stats['center_y']:.1f}) pixels")
        print(f"    - Centroid: ({stats['mean_x']:.1f}, {stats['mean_y']:.1f}) pixels")
        print(f"    - Frame range: {stats['frame_range'][0]} - {stats['frame_range'][1]}")
        print(f"    - Localization spread: {stats['std_x']:.2f} x {stats['std_y']:.2f} pixels")

        if 'mean_photons' in stats:
            print(f"    - Mean photons: {stats['mean_photons']:.0f}")

    # Step 3: Apply DBSCAN clustering to identify real fiducials (optional but recommended)
    print(f"\nStep 3: Applying DBSCAN clustering to validate fiducials...")

    try:
        validated_fiducials, clustering_metadata = drift_corrector.identify_real_fiducials_with_clustering(
            selected_puncta=selected_puncta,
            precision_factor=3.0,      # Adjust based on your localization precision
            min_samples_factor=0.6,    # Fraction of total frames (adjust as needed)
            frame_count=int(info[0]['Frames']),
            output_figure_path="step3_clustering.png",
            title="Step 3: DBSCAN Clustering Validation",
            create_plot=True
        )

        print(f"  ✓ Validated {len(validated_fiducials)} real fiducials from {len(selected_puncta)} candidates")
        print(f"  ✓ Validation rate: {clustering_metadata['validation_rate']*100:.1f}%")
        print(f"  ✓ Total validated localizations: {clustering_metadata['total_validated_locs']}")

        if clustering_metadata['region_details']:
            print(f"  ✓ Clustering details:")
            for i, meta in enumerate(clustering_metadata['region_details']):
                print(f"    Fiducial {i+1}: {meta['validated_n_locs']} locs, "
                      f"noise={meta['noise_fraction']*100:.1f}%")

        # Use validated fiducials for subsequent processing
        final_fiducials = validated_fiducials
        final_metadata = clustering_metadata

    except Exception as e:
        print(f"  ⚠ Clustering validation failed: {e}")
        print(f"  → Using unvalidated puncta (not recommended for production)")
        final_fiducials = selected_puncta
        final_metadata = selection_metadata

    # Step 4: Use validated fiducials for drift correction
    print(f"\nStep 4: Ready for drift correction with {len(final_fiducials)} validated fiducials")
    print("  → These validated fiducials can now be used with existing drift correction methods")
    print("  → Each fiducial array contains localizations for one validated marker")

    return final_fiducials, final_metadata


def create_example_data():
    """Create example data for demonstration."""

    # Create synthetic localizations with fiducial-like clusters
    np.random.seed(123)

    # Define fiducial positions
    fiducial_positions = [
        (20, 20), (20, 80), (80, 20), (80, 80), (50, 50)
    ]

    all_locs = []
    n_frames = 1000

    # Create localizations around each fiducial
    for fid_x, fid_y in fiducial_positions:
        n_locs = np.random.randint(50, 150)  # Variable number of localizations

        for _ in range(n_locs):
            # Scatter around fiducial position (localization precision)
            x = fid_x + np.random.normal(0, 0.3)
            y = fid_y + np.random.normal(0, 0.3)
            frame = np.random.randint(1, n_frames + 1)
            photons = np.random.normal(2000, 400)

            all_locs.append([x, y, frame, max(photons, 500), 0.15, 0.15])

    # Add background localizations
    for _ in range(200):
        x = np.random.uniform(5, 95)
        y = np.random.uniform(5, 95)
        frame = np.random.randint(1, n_frames + 1)
        photons = np.random.normal(1200, 300)

        all_locs.append([x, y, frame, max(photons, 300), 0.2, 0.2])

    # Convert to record array
    all_data = np.array(all_locs)
    locs = np.rec.array(
        (all_data[:, 0], all_data[:, 1], all_data[:, 2].astype(int),
         all_data[:, 3], all_data[:, 4], all_data[:, 5]),
        dtype=[("xc", "f4"), ("yc", "f4"), ("frame", "i4"),
               ("photons", "f4"), ("xc_err", "f4"), ("yc_err", "f4")]
    )

    # Create metadata
    info = [{"Width": 100.0, "Height": 100.0, "Frames": float(n_frames), "Pixelsize": 100.0}]

    # Create smoothed image (simulate rendered image)
    image = np.zeros((100, 100))
    x_bins = np.arange(0, 101)
    y_bins = np.arange(0, 101)
    hist, _, _ = np.histogram2d(locs.yc, locs.xc, bins=[y_bins, x_bins])

    # Apply smoothing
    from scipy.ndimage import gaussian_filter
    smoothed_image = gaussian_filter(hist, sigma=2.5)

    return locs, info, smoothed_image


def parameter_tuning_tips():
    """Print tips for parameter tuning."""

    print("\n" + "=" * 50)
    print("PARAMETER TUNING TIPS")
    print("=" * 50)

    print("\nStep 1 - detect_high_density_regions_from_image():")
    print("  threshold_percentile:")
    print("    - Start with 95-99% for sparse data")
    print("    - Lower to 90-95% if no regions detected")
    print("    - Higher (99.5%+) for very dense data")
    print("  histogram_bins:")
    print("    - 64-256 bins usually work well")
    print("    - More bins = finer threshold control")

    print("\nStep 2 - select_puncta_from_regions():")
    print("  selection_box_size_nm:")
    print("    - Start with 600-1000 nm")
    print("    - Increase if fiducials are spread out")
    print("    - Decrease if too much background included")
    print("  min_localizations_per_region:")
    print("    - 10-50 depending on fiducial density")
    print("    - Too low = noisy false positives")
    print("    - Too high = miss real fiducials")

    print("\nStep 3 - identify_real_fiducials_with_clustering():")
    print("  precision_factor:")
    print("    - 2-5x localization precision for eps parameter")
    print("    - Lower = tighter clustering (fewer fiducials)")
    print("    - Higher = looser clustering (more fiducials)")
    print("  min_samples_factor:")
    print("    - 0.01-0.6 fraction of total frames")
    print("    - Higher = stricter validation (fewer fiducials)")
    print("    - Lower = more permissive (more fiducials)")
    print("  Note: Clustering automatically filters out noise regions")

    print("\nDebugging:")
    print("  - Set create_plot=True to visualize each step")
    print("  - Check detection metadata for statistics")
    print("  - Examine rejection_reasons to tune parameters")
    print("  - Check clustering validation_rate and noise_fraction")
    print("  - Start with lenient criteria, then tighten")


def main():
    """Run the complete example workflow."""

    try:
        # Run the main workflow
        selected_puncta, metadata = example_workflow()

        # Show parameter tuning tips
        parameter_tuning_tips()

        print(f"\n{'='*50}")
        print("WORKFLOW COMPLETE")
        print(f"{'='*50}")
        print(f"✓ Generated visualization files:")
        print(f"  - step1_regions_*.png (density detection)")
        print(f"  - step2_puncta_*.png (puncta selection)")
        print(f"✓ Selected {len(selected_puncta) if selected_puncta else 0} fiducial candidates ready for drift correction")

    except Exception as e:
        print(f"✗ Workflow failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()