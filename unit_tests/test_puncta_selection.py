#!/usr/bin/env python3
"""
Test script for the new select_puncta_from_regions function.

This demonstrates the two-step process:
1. Detect high-density regions from a smoothed image
2. Select puncta (localizations) from those detected regions
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


def create_synthetic_fiducial_data():
    """Create synthetic localization data with fiducial-like puncta."""

    np.random.seed(42)

    # Create several fiducial regions with localizations
    fiducial_centers = [
        (25, 25),   # Top-left fiducial
        (25, 75),   # Top-right fiducial
        (75, 25),   # Bottom-left fiducial
        (75, 75),   # Bottom-right fiducial
        (50, 50),   # Center fiducial
    ]

    # Parameters
    n_frames = 1000
    pixelsize = 100.0  # nm per pixel

    # Generate localizations around each fiducial center
    all_localizations = []

    for fid_id, (center_x, center_y) in enumerate(fiducial_centers):
        # Each fiducial gets different numbers of localizations
        if fid_id == 0:  # First fiducial: many localizations
            n_locs_this_fid = 150
        elif fid_id == 1:  # Second fiducial: moderate
            n_locs_this_fid = 80
        elif fid_id == 2:  # Third fiducial: few (might be rejected)
            n_locs_this_fid = 25
        elif fid_id == 3:  # Fourth fiducial: very few (likely rejected)
            n_locs_this_fid = 8
        else:  # Fifth fiducial: good amount
            n_locs_this_fid = 120

        for _ in range(n_locs_this_fid):
            # Random frame
            frame = np.random.randint(1, n_frames + 1)

            # Position scatter around fiducial center (localization precision)
            x_scatter = np.random.normal(0, 0.2)  # 20nm precision
            y_scatter = np.random.normal(0, 0.2)

            x_pos = center_x + x_scatter
            y_pos = center_y + y_scatter

            # Photon count
            photons = np.random.normal(2000, 500)
            photons = max(photons, 500)

            # Localization precision estimates
            xc_err = 0.15
            yc_err = 0.15

            all_localizations.append([x_pos, y_pos, frame, photons, xc_err, yc_err])

    # Add some background/noise localizations
    n_background = 300
    for _ in range(n_background):
        x_pos = np.random.uniform(5, 95)
        y_pos = np.random.uniform(5, 95)
        frame = np.random.randint(1, n_frames + 1)
        photons = np.random.normal(1500, 400)
        photons = max(photons, 300)

        all_localizations.append([x_pos, y_pos, frame, photons, 0.2, 0.2])

    # Convert to numpy record array
    all_data = np.array(all_localizations)

    locs = np.rec.array(
        (
            all_data[:, 0],  # xc
            all_data[:, 1],  # yc
            all_data[:, 2].astype(int),  # frame
            all_data[:, 3],  # photons
            all_data[:, 4],  # xc_err
            all_data[:, 5],  # yc_err
        ),
        dtype=[
            ("xc", "f4"),
            ("yc", "f4"),
            ("frame", "i4"),
            ("photons", "f4"),
            ("xc_err", "f4"),
            ("yc_err", "f4"),
        ]
    )

    # Create metadata
    info = [
        {
            "Width": 100.0,
            "Height": 100.0,
            "Frames": float(n_frames),
            "Pixelsize": pixelsize,
        }
    ]

    return locs, info, fiducial_centers


def create_smoothed_image_from_locs(locs, image_size=100, blur_sigma=2.0):
    """Create a smoothed image from localizations for density detection."""

    # Create histogram of localization positions
    image = np.zeros((image_size, image_size))

    # Bin localizations into pixels
    x_bins = np.arange(0, image_size + 1)
    y_bins = np.arange(0, image_size + 1)

    hist, _, _ = np.histogram2d(locs.yc, locs.xc, bins=[y_bins, x_bins])

    # Apply Gaussian smoothing
    from scipy.ndimage import gaussian_filter
    smoothed_image = gaussian_filter(hist, sigma=blur_sigma)

    return smoothed_image


def test_puncta_selection_pipeline():
    """Test the complete pipeline: density detection -> puncta selection."""
    print("=== Testing Complete Puncta Selection Pipeline ===")

    # Create synthetic data
    locs, info, true_fiducial_centers = create_synthetic_fiducial_data()
    print(f"Created {len(locs)} synthetic localizations")
    print(f"True fiducial centers: {true_fiducial_centers}")

    # Create smoothed image for density detection
    smoothed_image = create_smoothed_image_from_locs(locs, image_size=100, blur_sigma=3.0)
    print(f"Created smoothed image: {smoothed_image.shape}")

    # Initialize drift correction functions
    DCF = Drift_Correction_Functions()

    # Step 1: Detect high-density regions
    print("\nStep 1: Detecting high-density regions...")
    try:
        region_centers, binary_mask, threshold, density_metadata = DCF.detect_high_density_regions_from_image(
            smoothed_image=smoothed_image,
            histogram_bins=64,
            threshold_percentile=90.0,  # Lower threshold to catch more regions
            pixelsize=100.0,
            output_figure_path="pipeline_density_detection.png",
            title="Pipeline Test: Density Detection"
        )

        print(f"✓ Detected {len(region_centers)} high-density regions")
        print(f"  Threshold used: {threshold:.2f}")
        print(f"  Detected centers: {region_centers}")

    except Exception as e:
        print(f"✗ Density detection failed: {e}")
        return False

    # Step 2: Select puncta from detected regions
    # Uses postprocess.picked_locs for optimized rectangular selection
    print("\nStep 2: Selecting puncta from detected regions...")
    try:
        selected_puncta, selection_metadata = DCF.select_puncta_from_regions(
            locs=locs,
            region_centers=region_centers,
            binary_mask=binary_mask,
            pixelsize=100.0,
            selection_box_size_nm=800.0,  # 800nm box size
            min_localizations_per_region=15,  # Minimum 15 localizations
            output_figure_path="pipeline_puncta_selection.png",
            title="Pipeline Test: Puncta Selection"
        )

        print(f"✓ Selected {len(selected_puncta)} valid puncta regions")
        print(f"  Selection criteria: {selection_metadata['selection_criteria']}")
        print(f"  Rejection reasons: {selection_metadata['rejection_reasons']}")

        # Print details for each selected region
        for i, (puncta, stats) in enumerate(zip(selected_puncta, selection_metadata['region_statistics'])):
            print(f"  Region {i+1}: {len(puncta)} locs, center=({stats['center_x']:.1f}, {stats['center_y']:.1f}), "
                  f"frames {stats['frame_range'][0]}-{stats['frame_range'][1]}")

        return True

    except Exception as e:
        import traceback
        print(f"✗ Puncta selection failed: {e}")
        print(f"  Details: {traceback.format_exc()}")
        return False


def test_puncta_selection_parameters():
    """Test puncta selection with different parameter settings."""
    print("\n=== Testing Different Selection Parameters ===")

    # Create synthetic data
    locs, info, _ = create_synthetic_fiducial_data()
    smoothed_image = create_smoothed_image_from_locs(locs)

    DCF = Drift_Correction_Functions()

    # Get regions from density detection
    region_centers, binary_mask, _, _ = DCF.detect_high_density_regions_from_image(
        smoothed_image=smoothed_image,
        threshold_percentile=90.0,
        create_plot=False
    )

    print(f"Using {len(region_centers)} detected regions for parameter testing")

    # Test different selection box sizes
    box_sizes = [400.0, 600.0, 800.0, 1000.0]  # nm

    for box_size in box_sizes:
        try:
            selected_puncta, metadata = DCF.select_puncta_from_regions(
                locs=locs,
                region_centers=region_centers,
                binary_mask=binary_mask,
                selection_box_size_nm=box_size,
                min_localizations_per_region=10,
                create_plot=False
            )

            total_locs = sum(len(puncta) for puncta in selected_puncta)
            print(f"  Box {box_size:6.0f}nm: {len(selected_puncta):2d} regions, {total_locs:4d} total locs")

        except Exception as e:
            print(f"  Box {box_size:6.0f}nm: FAILED - {e}")

    # Test different localization count thresholds
    min_locs_list = [5, 10, 15, 20, 30]

    print(f"\nTesting minimum localization thresholds:")
    for min_locs in min_locs_list:
        try:
            selected_puncta, metadata = DCF.select_puncta_from_regions(
                locs=locs,
                region_centers=region_centers,
                binary_mask=binary_mask,
                selection_box_size_nm=700.0,
                min_localizations_per_region=min_locs,
                create_plot=False
            )

            reasons = metadata['rejection_reasons']
            print(f"  Min {min_locs:2d} locs: {len(selected_puncta):2d} accepted, "
                  f"{reasons['too_few_localizations']:2d} too few")

        except Exception as e:
            print(f"  Min {min_locs:2d} locs: FAILED - {e}")


def test_edge_cases():
    """Test edge cases for puncta selection."""
    print("\n=== Testing Edge Cases ===")

    DCF = Drift_Correction_Functions()

    # Test 1: Empty region centers
    print("Test 1: Empty region centers...")
    locs, _, _ = create_synthetic_fiducial_data()
    try:
        selected_puncta, metadata = DCF.select_puncta_from_regions(
            locs=locs,
            region_centers=[],  # Empty list
            binary_mask=np.zeros((100, 100), dtype=bool),
            create_plot=False
        )
        print(f"  ✓ Empty regions: {len(selected_puncta)} puncta selected")
    except Exception as e:
        print(f"  ✗ Empty regions failed: {e}")

    # Test 2: Very strict criteria (should reject all)
    print("Test 2: Very strict criteria...")
    smoothed_image = create_smoothed_image_from_locs(locs)
    region_centers, binary_mask, _, _ = DCF.detect_high_density_regions_from_image(
        smoothed_image=smoothed_image,
        threshold_percentile=90.0,
        create_plot=False
    )

    try:
        selected_puncta, metadata = DCF.select_puncta_from_regions(
            locs=locs,
            region_centers=region_centers,
            binary_mask=binary_mask,
            selection_box_size_nm=100.0,  # Very small box
            min_localizations_per_region=1000,  # Very high minimum
            create_plot=False
        )
        reasons = metadata['rejection_reasons']
        print(f"  ✓ Strict criteria: {len(selected_puncta)} accepted, "
              f"{reasons['too_few_localizations']} rejected (too few)")
    except Exception as e:
        print(f"  ✗ Strict criteria failed: {e}")

    # Test 3: Single region center
    print("Test 3: Single region center...")
    try:
        selected_puncta, metadata = DCF.select_puncta_from_regions(
            locs=locs,
            region_centers=[(50, 50)],  # Single center
            binary_mask=np.ones((100, 100), dtype=bool),
            selection_box_size_nm=1000.0,
            min_localizations_per_region=10,
            create_plot=False
        )
        print(f"  ✓ Single region: {len(selected_puncta)} puncta selected")
        if selected_puncta:
            print(f"    Selected {len(selected_puncta[0])} localizations")
    except Exception as e:
        print(f"  ✗ Single region failed: {e}")


def main():
    """Run all puncta selection tests."""
    print("Testing Puncta Selection from Detected Regions")
    print("=" * 60)

    # Run tests
    results = []
    results.append(test_puncta_selection_pipeline())
    test_puncta_selection_parameters()  # No return value
    test_edge_cases()  # No return value

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    passed = sum(1 for r in results if r is True)
    total = len(results)
    print(f"  Core tests passed: {passed}/{total}")

    if passed == total:
        print("✓ All core tests passed! The puncta selection function is working correctly.")
        print("\nGenerated test images:")
        print("  - pipeline_density_detection_*.png (density detection)")
        print("  - pipeline_puncta_selection_*.png (puncta selection)")
    else:
        print("✗ Some core tests failed. Check the output above for details.")

    print("\nThe two-step process works as follows:")
    print("  1. detect_high_density_regions_from_image() -> region_centers, binary_mask")
    print("  2. select_puncta_from_regions() -> list of localization arrays per region")
    print("\nThis provides clear separation and debugging capability for fiducial detection.")


if __name__ == "__main__":
    main()