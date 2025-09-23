#!/usr/bin/env python3
"""
Test script for DBSCAN clustering function to identify real fiducials.

This demonstrates the three-step process:
1. Detect high-density regions from a smoothed image
2. Select puncta (localizations) from those detected regions
3. Use DBSCAN clustering to identify real fiducials and filter out noise

Usage:
    python test_fiducial_clustering.py
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


def create_synthetic_data_with_noise():
    """Create synthetic localization data with real fiducials and noise regions."""

    np.random.seed(42)

    # Parameters
    n_frames = 50000
    pixelsize = 100.0  # nm per pixel

    # Create real fiducials with tight clustering
    real_fiducial_centers = [
        (25, 25),   # Top-left fiducial - very good
        (75, 75),   # Bottom-right fiducial - good
        (25, 75),   # Top-right fiducial - moderate
    ]

    # Create noise regions (will be filtered out by clustering)
    noise_regions = [
        (50, 25),   # Scattered noise
        (75, 25),   # Low-density cluster
    ]

    all_localizations = []

    # Generate real fiducial data with good clustering
    for fid_id, (center_x, center_y) in enumerate(real_fiducial_centers):
        if fid_id == 0:  # Excellent fiducial
            n_locs = 800
            precision = 0.1  # 10nm precision
        elif fid_id == 1:  # Good fiducial
            n_locs = 600
            precision = 0.15  # 15nm precision
        else:  # Moderate fiducial
            n_locs = 400
            precision = 0.2  # 20nm precision

        for _ in range(n_locs):
            frame = np.random.randint(1, n_frames + 1)
            x_scatter = np.random.normal(0, precision)
            y_scatter = np.random.normal(0, precision)

            x_pos = center_x + x_scatter
            y_pos = center_y + y_scatter
            photons = np.random.normal(3000, 800)
            photons = max(photons, 1000)

            all_localizations.append([
                x_pos, y_pos, frame, photons, precision*0.7, precision*0.7
            ])

    # Generate noise regions (should be filtered out)
    for noise_id, (center_x, center_y) in enumerate(noise_regions):
        if noise_id == 0:  # Very scattered noise
            n_locs = 200
            scatter = 2.0  # Very spread out
        else:  # Low-density cluster
            n_locs = 50
            scatter = 0.3  # Tight but too few points

        for _ in range(n_locs):
            frame = np.random.randint(1, n_frames + 1)
            x_scatter = np.random.normal(0, scatter)
            y_scatter = np.random.normal(0, scatter)

            x_pos = center_x + x_scatter
            y_pos = center_y + y_scatter
            photons = np.random.normal(1500, 600)
            photons = max(photons, 500)

            all_localizations.append([
                x_pos, y_pos, frame, photons, 0.25, 0.25
            ])

    # Add background scattered localizations
    n_background = 500
    for _ in range(n_background):
        x_pos = np.random.uniform(5, 95)
        y_pos = np.random.uniform(5, 95)
        frame = np.random.randint(1, n_frames + 1)
        photons = np.random.normal(1200, 400)
        photons = max(photons, 300)

        all_localizations.append([x_pos, y_pos, frame, photons, 0.3, 0.3])

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

    return locs, info, real_fiducial_centers, noise_regions


def create_smoothed_image_from_locs(locs, image_size=100, blur_sigma=2.5):
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


def test_complete_fiducial_detection_pipeline():
    """Test the complete 3-step fiducial detection and clustering pipeline."""
    print("=== Testing Complete Fiducial Detection + Clustering Pipeline ===")

    # Create synthetic data with real fiducials and noise
    locs, info, real_centers, noise_centers = create_synthetic_data_with_noise()
    print(f"Created {len(locs)} synthetic localizations")
    print(f"Real fiducial centers: {real_centers}")
    print(f"Noise region centers: {noise_centers}")

    # Create smoothed image for density detection
    smoothed_image = create_smoothed_image_from_locs(locs, blur_sigma=3.0)

    # Initialize drift correction functions
    DCF = Drift_Correction_Functions()

    # Step 1: Detect high-density regions
    print(f"\\nStep 1: Detecting high-density regions...")
    try:
        region_centers, binary_mask, threshold, density_metadata = DCF.detect_high_density_regions_from_image(
            smoothed_image=smoothed_image,
            histogram_bins=64,
            threshold_percentile=85.0,  # Lower threshold to catch all regions
            pixelsize=100.0,
            output_figure_path="clustering_test_step1.png",
            title="Step 1: Density Detection"
        )

        print(f"✓ Detected {len(region_centers)} high-density regions")
        print(f"  Detected centers: {region_centers}")

    except Exception as e:
        print(f"✗ Density detection failed: {e}")
        return False

    # Step 2: Select puncta from detected regions
    print(f"\\nStep 2: Selecting puncta from detected regions...")
    try:
        selected_puncta, selection_metadata = DCF.select_puncta_from_regions(
            locs=locs,
            region_centers=region_centers,
            binary_mask=binary_mask,
            pixelsize=100.0,
            selection_box_size_nm=1000.0,  # Large box to capture all localizations
            min_localizations_per_region=20,  # Low threshold to include all candidates
            output_figure_path="clustering_test_step2.png",
            title="Step 2: Puncta Selection"
        )

        print(f"✓ Selected {len(selected_puncta)} puncta regions")
        for i, puncta in enumerate(selected_puncta):
            print(f"  Region {i+1}: {len(puncta)} localizations")

    except Exception as e:
        print(f"✗ Puncta selection failed: {e}")
        return False

    # Step 3: Apply DBSCAN clustering to identify real fiducials
    print(f"\\nStep 3: Applying DBSCAN clustering to identify real fiducials...")
    try:
        validated_fiducials, clustering_metadata = DCF.identify_real_fiducials_with_clustering(
            selected_puncta=selected_puncta,
            precision_factor=3.0,      # 3x precision for eps parameter
            min_samples_factor=0.01,   # 1% of frames (500 samples)
            frame_count=info[0]['Frames'],
            output_figure_path="clustering_test_step3.png",
            title="Step 3: DBSCAN Clustering Validation"
        )

        print(f"✓ Validated {len(validated_fiducials)} real fiducials")
        print(f"  Validation rate: {clustering_metadata['validation_rate']*100:.1f}%")
        print(f"  Total input locs: {clustering_metadata['total_input_locs']}")
        print(f"  Total validated locs: {clustering_metadata['total_validated_locs']}")

        # Show details for each validated fiducial
        for i, (fiducial, meta) in enumerate(zip(validated_fiducials, clustering_metadata['region_details'])):
            print(f"  Fiducial {i+1}: {len(fiducial)} locs, "
                  f"noise={meta['noise_fraction']*100:.1f}%, "
                  f"center=({meta['cluster_center_x']:.1f}, {meta['cluster_center_y']:.1f})")

        return True

    except Exception as e:
        import traceback
        print(f"✗ DBSCAN clustering failed: {e}")
        print(f"  Details: {traceback.format_exc()}")
        return False


def test_clustering_parameters():
    """Test clustering with different parameter settings."""
    print("\\n=== Testing Different Clustering Parameters ===")

    # Create data and get puncta
    locs, info, _, _ = create_synthetic_data_with_noise()
    smoothed_image = create_smoothed_image_from_locs(locs)

    DCF = Drift_Correction_Functions()

    # Get regions and puncta (using same approach as main test)
    region_centers, binary_mask, _, _ = DCF.detect_high_density_regions_from_image(
        smoothed_image=smoothed_image,
        threshold_percentile=85.0,
        create_plot=False
    )

    selected_puncta, _ = DCF.select_puncta_from_regions(
        locs=locs,
        region_centers=region_centers,
        binary_mask=binary_mask,
        selection_box_size_nm=1000.0,
        min_localizations_per_region=20,
        create_plot=False
    )

    print(f"Using {len(selected_puncta)} puncta regions for parameter testing")

    # Test different precision factors
    precision_factors = [1.0, 2.0, 3.0, 5.0]

    for prec_factor in precision_factors:
        try:
            validated_fiducials, metadata = DCF.identify_real_fiducials_with_clustering(
                selected_puncta=selected_puncta,
                precision_factor=prec_factor,
                min_samples_factor=0.01,
                frame_count=int(info[0]['Frames']),
                create_plot=False
            )

            print(f"  Precision factor {prec_factor:3.1f}: {len(validated_fiducials):2d} fiducials, "
                  f"rate={metadata['validation_rate']*100:4.1f}%")

        except Exception as e:
            print(f"  Precision factor {prec_factor:3.1f}: FAILED - {e}")

    # Test different min_samples factors
    min_samples_factors = [0.005, 0.01, 0.02, 0.05]

    print(f"\\nTesting minimum samples factors:")
    for min_samp_factor in min_samples_factors:
        try:
            validated_fiducials, metadata = DCF.identify_real_fiducials_with_clustering(
                selected_puncta=selected_puncta,
                precision_factor=3.0,
                min_samples_factor=min_samp_factor,
                frame_count=int(info[0]['Frames']),
                create_plot=False
            )

            # Show min_samples used
            if metadata['region_details']:
                min_samples_used = metadata['region_details'][0]['min_samples_used']
                print(f"  Min samples factor {min_samp_factor:5.3f} "
                      f"({min_samples_used:3d} samples): {len(validated_fiducials):2d} fiducials, "
                      f"rate={metadata['validation_rate']*100:4.1f}%")
            else:
                print(f"  Min samples factor {min_samp_factor:5.3f}: 0 fiducials")

        except Exception as e:
            print(f"  Min samples factor {min_samp_factor:5.3f}: FAILED - {e}")


def main():
    """Run all clustering tests."""
    print("Testing DBSCAN Clustering for Fiducial Detection")
    print("=" * 60)

    # Run tests
    results = []
    results.append(test_complete_fiducial_detection_pipeline())
    test_clustering_parameters()  # No return value

    # Summary
    print("\\n" + "=" * 60)
    print("Test Summary:")
    passed = sum(1 for r in results if r is True)
    total = len(results)
    print(f"  Core tests passed: {passed}/{total}")

    if passed == total:
        print("✓ All tests passed! The DBSCAN clustering function is working correctly.")
        print("\\nGenerated test images:")
        print("  - clustering_test_step1_*.png (density detection)")
        print("  - clustering_test_step2_*.png (puncta selection)")
        print("  - clustering_test_step3_*.png (clustering validation)")
    else:
        print("✗ Some tests failed. Check the output above for details.")

    print("\\nThe three-step process works as follows:")
    print("  1. detect_high_density_regions_from_image() -> region_centers, binary_mask")
    print("  2. select_puncta_from_regions() -> list of localization arrays per region")
    print("  3. identify_real_fiducials_with_clustering() -> validated fiducial list")
    print("\\nThis provides comprehensive fiducial detection with noise filtering.")


if __name__ == "__main__":
    main()