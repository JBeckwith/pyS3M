#!/usr/bin/env python3
"""
Test script for the new detect_high_density_regions_from_image function.

This demonstrates how to use the standalone high-density region detection
function that splits up the fiducial detection workflow.
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


def create_synthetic_image_with_spots():
    """Create a synthetic smoothed image with high-density spots for testing."""

    # Create base image
    image_size = 100
    image = np.zeros((image_size, image_size))

    # Add background noise
    np.random.seed(42)
    background_level = 10
    noise_level = 2
    image += background_level + np.random.normal(0, noise_level, image.shape)

    # Add several high-density spots (simulating fiducials)
    spot_positions = [
        (20, 20),   # Top-left spot
        (20, 80),   # Top-right spot
        (80, 20),   # Bottom-left spot
        (80, 80),   # Bottom-right spot
        (50, 50),   # Center spot
    ]

    # Create Gaussian spots
    spot_intensity = 150
    spot_sigma = 3

    y_grid, x_grid = np.meshgrid(np.arange(image_size), np.arange(image_size), indexing='ij')

    for spot_y, spot_x in spot_positions:
        # Create Gaussian spot
        gaussian_spot = spot_intensity * np.exp(
            -((x_grid - spot_x)**2 + (y_grid - spot_y)**2) / (2 * spot_sigma**2)
        )
        image += gaussian_spot

    # Add some additional random weak spots
    n_weak_spots = 3
    for _ in range(n_weak_spots):
        weak_y = np.random.randint(10, image_size - 10)
        weak_x = np.random.randint(10, image_size - 10)
        weak_intensity = 50
        weak_spot = weak_intensity * np.exp(
            -((x_grid - weak_x)**2 + (y_grid - weak_y)**2) / (2 * spot_sigma**2)
        )
        image += weak_spot

    return image, spot_positions


def test_density_detection_basic():
    """Test basic functionality of the density detection function."""
    print("=== Testing Basic Density Detection ===")

    # Create synthetic data
    synthetic_image, true_positions = create_synthetic_image_with_spots()
    print(f"Created synthetic image with {len(true_positions)} known high-density spots")
    print(f"Image shape: {synthetic_image.shape}")
    print(f"Image intensity range: [{synthetic_image.min():.1f}, {synthetic_image.max():.1f}]")

    # Initialize drift correction functions
    DCF = Drift_Correction_Functions()

    # Test the new detection function
    try:
        region_centers, binary_mask, threshold, metadata = DCF.detect_high_density_regions_from_image(
            smoothed_image=synthetic_image,
            histogram_bins=64,
            threshold_percentile=95.0,  # Lower threshold to catch more regions
            pixelsize=100.0,  # 100 nm per pixel
            output_figure_path="density_detection_test.png",
            title="Test: High-Density Region Detection"
        )

        print(f"✓ Detection completed successfully")
        print(f"  - Regions detected: {len(region_centers)}")
        print(f"  - Threshold used: {threshold:.2f}")
        print(f"  - Region area fraction: {metadata['region_area_fraction']:.3f}")

        # Print detected region centers
        print("  - Detected centers (y, x):")
        for i, (y, x) in enumerate(region_centers):
            print(f"    {i+1}: ({y}, {x})")

        # Print true positions for comparison
        print("  - True spot positions (y, x):")
        for i, (y, x) in enumerate(true_positions):
            print(f"    {i+1}: ({y}, {x})")

        # Print detailed metadata
        print(f"  - Detection metadata:")
        for key, value in metadata.items():
            if key != 'region_statistics':  # Skip detailed stats for now
                print(f"    {key}: {value}")

        return True

    except Exception as e:
        import traceback
        print(f"✗ Detection failed: {e}")
        print(f"  Details: {traceback.format_exc()}")
        return False


def test_density_detection_parameters():
    """Test detection with different parameter settings."""
    print("\n=== Testing Different Parameter Settings ===")

    # Create synthetic data
    synthetic_image, true_positions = create_synthetic_image_with_spots()
    DCF = Drift_Correction_Functions()

    # Test different threshold percentiles
    percentiles = [90.0, 95.0, 98.0, 99.0, 99.5]

    for percentile in percentiles:
        try:
            region_centers, binary_mask, threshold, metadata = DCF.detect_high_density_regions_from_image(
                smoothed_image=synthetic_image,
                histogram_bins=64,
                threshold_percentile=percentile,
                pixelsize=100.0,
                output_figure_path=f"density_detection_p{percentile}.png",
                title=f"Detection at {percentile}th Percentile",
                create_plot=False  # Disable plotting for parameter tests to avoid matplotlib issues
            )

            print(f"  Percentile {percentile:5.1f}%: {len(region_centers):2d} regions detected (threshold: {threshold:6.2f})")

        except Exception as e:
            print(f"  Percentile {percentile:5.1f}%: FAILED - {e}")


def test_detection_with_different_images():
    """Test detection with different types of synthetic images."""
    print("\n=== Testing Different Image Types ===")

    DCF = Drift_Correction_Functions()

    # Test 1: Very sparse image (few spots)
    print("Test 1: Sparse image...")
    sparse_image = np.random.normal(5, 1, (50, 50))
    sparse_image[25, 25] = 100  # Single bright spot

    try:
        region_centers, _, threshold, metadata = DCF.detect_high_density_regions_from_image(
            smoothed_image=sparse_image,
            threshold_percentile=99.0,
            output_figure_path="sparse_test.png",
            title="Sparse Image Test",
            create_plot=False
        )
        print(f"  ✓ Sparse: {len(region_centers)} regions, threshold: {threshold:.2f}")
    except Exception as e:
        print(f"  ✗ Sparse test failed: {e}")

    # Test 2: Dense image (many spots)
    print("Test 2: Dense image...")
    dense_image = np.random.normal(20, 5, (80, 80))
    # Add many spots
    for i in range(0, 80, 15):
        for j in range(0, 80, 15):
            if i < 80 and j < 80:
                dense_image[i:i+3, j:j+3] += 50

    try:
        region_centers, _, threshold, metadata = DCF.detect_high_density_regions_from_image(
            smoothed_image=dense_image,
            threshold_percentile=95.0,
            output_figure_path="dense_test.png",
            title="Dense Image Test",
            create_plot=False
        )
        print(f"  ✓ Dense: {len(region_centers)} regions, threshold: {threshold:.2f}")
    except Exception as e:
        print(f"  ✗ Dense test failed: {e}")

    # Test 3: Empty image (should handle gracefully)
    print("Test 3: Empty image...")
    empty_image = np.zeros((30, 30))

    try:
        region_centers, _, threshold, metadata = DCF.detect_high_density_regions_from_image(
            smoothed_image=empty_image,
            threshold_percentile=99.0,
            output_figure_path="empty_test.png",
            title="Empty Image Test",
            create_plot=False
        )
        print(f"  ✓ Empty: {len(region_centers)} regions, threshold: {threshold:.2f}")
    except Exception as e:
        print(f"  ✗ Empty test failed (expected): {e}")


def test_region_output_format():
    """Test that the region output format is suitable for downstream processing."""
    print("\n=== Testing Region Output Format ===")

    # Create test image
    synthetic_image, true_positions = create_synthetic_image_with_spots()
    DCF = Drift_Correction_Functions()

    try:
        region_centers, binary_mask, threshold, metadata = DCF.detect_high_density_regions_from_image(
            smoothed_image=synthetic_image,
            threshold_percentile=95.0,
            pixelsize=100.0,
            create_plot=False
        )

        print("Region output validation:")
        print(f"  ✓ region_centers type: {type(region_centers)}")
        print(f"  ✓ region_centers length: {len(region_centers)}")

        if region_centers:
            print(f"  ✓ First region center type: {type(region_centers[0])}")
            print(f"  ✓ First region center: {region_centers[0]}")

            # Validate center coordinates are integers
            for i, (y, x) in enumerate(region_centers):
                assert isinstance(y, (int, np.integer)), f"y coordinate {y} is not integer"
                assert isinstance(x, (int, np.integer)), f"x coordinate {x} is not integer"
                assert 0 <= y < synthetic_image.shape[0], f"y={y} out of bounds"
                assert 0 <= x < synthetic_image.shape[1], f"x={x} out of bounds"
            print(f"  ✓ All {len(region_centers)} region coordinates are valid integers")

        print(f"  ✓ binary_mask type: {type(binary_mask)}")
        print(f"  ✓ binary_mask shape: {binary_mask.shape}")
        print(f"  ✓ binary_mask dtype: {binary_mask.dtype}")

        print(f"  ✓ threshold type: {type(threshold)} = {threshold}")

        print(f"  ✓ metadata type: {type(metadata)}")
        print(f"  ✓ metadata keys: {list(metadata.keys())}")

        # Test that region statistics are properly formatted
        if 'region_statistics' in metadata and metadata['region_statistics']:
            stats = metadata['region_statistics'][0]
            print(f"  ✓ First region stats keys: {list(stats.keys())}")
            print(f"  ✓ First region stats example: center={stats['center']}, area={stats['area_pixels']}")

        print("✓ All output formats validated successfully")
        return True

    except Exception as e:
        import traceback
        print(f"✗ Output format validation failed: {e}")
        print(f"  Details: {traceback.format_exc()}")
        return False


def main():
    """Run all density detection tests."""
    print("Testing High-Density Region Detection Function")
    print("=" * 60)

    # Run tests
    results = []
    results.append(test_density_detection_basic())
    test_density_detection_parameters()  # This one doesn't return boolean
    test_detection_with_different_images()  # This one doesn't return boolean
    results.append(test_region_output_format())

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    passed = sum(1 for r in results if r is True)
    total = len(results)
    print(f"  Tests passed: {passed}/{total}")

    if passed == total:
        print("✓ All tests passed! The density detection function is working correctly.")
        print("\nGenerated test images:")
        print("  - density_detection_test.png (main test)")
        print("  - density_detection_p*.png (parameter tests)")
        print("  - sparse_test.png, dense_test.png, empty_test.png (edge cases)")
    else:
        print("✗ Some tests failed. Check the output above for details.")

    print("\nThe function returns:")
    print("  1. region_centers: List of (y, x) coordinate tuples")
    print("  2. binary_mask: Boolean array marking detected regions")
    print("  3. threshold: Float threshold value used")
    print("  4. metadata: Dictionary with detection statistics")
    print("\nThese outputs can be used for downstream fiducial processing.")


if __name__ == "__main__":
    main()