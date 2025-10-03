#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test to prove that numpy array indexing [y, x] is correct for image coordinates (x, y).

This test demonstrates:
1. Image coordinates are (x, y) where x=column, y=row
2. Numpy arrays are indexed as [row, col] = [y, x]
3. Therefore, ROI extraction must use array[ymin:ymax, xmin:xmax]

Created on 2025-10-03
"""

import numpy as np
import matplotlib.pyplot as plt


def test_coordinate_system():
    """
    Test that proves [y, x] indexing is correct for image coordinates (x, y).
    """
    print("=" * 80)
    print("COORDINATE SYSTEM TEST")
    print("=" * 80)

    # Create a test image with known dimensions
    width = 100  # columns (x direction)
    height = 80  # rows (y direction)

    # Create image filled with zeros
    image = np.zeros((height, width), dtype=np.float32)

    print(f"\n1. Image shape: {image.shape}")
    print(f"   height (rows) = {height}, width (cols) = {width}")
    print(f"   Array indexing: image[row, col] = image[y, x]")

    # Place a bright spot at a specific IMAGE COORDINATE (x, y)
    # Let's use (x=60, y=30) - meaning column 60, row 30
    spot_x = 60  # column coordinate
    spot_y = 30  # row coordinate

    print(f"\n2. Placing spot at image coordinate (x={spot_x}, y={spot_y})")
    print(f"   This means: column {spot_x}, row {spot_y}")

    # Create a 5x5 Gaussian-like spot centered at (spot_x, spot_y)
    spot_size = 5
    for dy in range(-spot_size//2, spot_size//2 + 1):
        for dx in range(-spot_size//2, spot_size//2 + 1):
            row = spot_y + dy
            col = spot_x + dx
            if 0 <= row < height and 0 <= col < width:
                # Use numpy indexing [row, col] to set the value
                image[row, col] = 1000 * np.exp(-(dx**2 + dy**2) / 2.0)

    print(f"   Spot placed using: image[row, col] = image[{spot_y}, {spot_x}]")

    # Verify the spot center value
    center_value = image[spot_y, spot_x]
    print(f"   Center value at image[{spot_y}, {spot_x}] = {center_value:.1f}")

    # Now extract an ROI around this spot using CORRECT indexing
    roi_size = 16
    xmin = spot_x - roi_size // 2
    xmax = spot_x + roi_size // 2
    ymin = spot_y - roi_size // 2
    ymax = spot_y + roi_size // 2

    print(f"\n3. Extracting {roi_size}x{roi_size} ROI centered at (x={spot_x}, y={spot_y})")
    print(f"   Boundaries: xmin={xmin}, xmax={xmax}, ymin={ymin}, ymax={ymax}")

    # CORRECT extraction: [ymin:ymax, xmin:xmax]
    roi_correct = image[ymin:ymax, xmin:xmax]

    # WRONG extraction: [xmin:xmax, ymin:ymax]
    roi_wrong = image[xmin:xmax, ymin:ymax]

    print(f"\n4. Testing extraction methods:")
    print(f"   CORRECT: roi = image[ymin:ymax, xmin:xmax] = image[{ymin}:{ymax}, {xmin}:{xmax}]")
    print(f"   - ROI shape: {roi_correct.shape}")
    print(f"   - ROI center value: {roi_correct[roi_size//2, roi_size//2]:.1f}")
    print(f"   - Max value in ROI: {roi_correct.max():.1f}")

    print(f"\n   WRONG: roi = image[xmin:xmax, ymin:ymax] = image[{xmin}:{xmax}, {ymin}:{ymax}]")
    print(f"   - ROI shape: {roi_wrong.shape}")
    print(f"   - ROI center value: {roi_wrong[roi_size//2, roi_size//2]:.1f}")
    print(f"   - Max value in ROI: {roi_wrong.max():.1f}")

    # Verify correctness
    print(f"\n5. Verification:")
    correct_is_square = (roi_correct.shape[0] == roi_correct.shape[1] == roi_size)
    correct_has_spot = (roi_correct.max() > 900)  # Should have the bright spot

    wrong_is_square = (roi_wrong.shape[0] == roi_wrong.shape[1] == roi_size)
    wrong_has_spot = (roi_wrong.max() > 900)

    print(f"   CORRECT indexing [y, x]:")
    print(f"   - Is square? {correct_is_square}")
    print(f"   - Contains spot? {correct_has_spot}")

    print(f"\n   WRONG indexing [x, y]:")
    print(f"   - Is square? {wrong_is_square}")
    print(f"   - Contains spot? {wrong_has_spot}")

    # Final verdict
    print(f"\n{'=' * 80}")
    print("CONCLUSION:")
    print("=" * 80)

    if correct_is_square and correct_has_spot and not wrong_has_spot:
        print("✓ CORRECT indexing [ymin:ymax, xmin:xmax] successfully extracts ROI")
        print("✗ WRONG indexing [xmin:xmax, ymin:ymax] fails to extract correct ROI")
        print("\nTherefore: array[ymin:ymax, xmin:xmax] is the CORRECT way to extract")
        print("           an ROI centered at image coordinates (x, y)")
        test_passed = True
    else:
        print("✗ Test failed - unexpected behavior")
        test_passed = False

    print("=" * 80)

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Full image with ROI box
    axes[0].imshow(image, cmap='hot', origin='upper')
    axes[0].plot([xmin, xmax, xmax, xmin, xmin],
                 [ymin, ymin, ymax, ymax, ymin], 'g-', linewidth=2)
    axes[0].plot(spot_x, spot_y, 'g+', markersize=15, markeredgewidth=2)
    axes[0].set_title(f'Full Image ({width}x{height})\nSpot at (x={spot_x}, y={spot_y})')
    axes[0].set_xlabel('x (columns)')
    axes[0].set_ylabel('y (rows)')
    axes[0].text(spot_x + 5, spot_y - 5, f'({spot_x}, {spot_y})',
                 color='green', fontsize=10)

    # CORRECT extraction
    axes[1].imshow(roi_correct, cmap='hot', origin='upper')
    axes[1].set_title(f'CORRECT: image[{ymin}:{ymax}, {xmin}:{xmax}]\n'
                      f'Shape: {roi_correct.shape}, Max: {roi_correct.max():.1f}')
    axes[1].plot(roi_size//2, roi_size//2, 'g+', markersize=15, markeredgewidth=2)
    axes[1].set_xlabel('x (columns)')
    axes[1].set_ylabel('y (rows)')

    # WRONG extraction
    axes[2].imshow(roi_wrong, cmap='hot', origin='upper')
    axes[2].set_title(f'WRONG: image[{xmin}:{xmax}, {ymin}:{ymax}]\n'
                      f'Shape: {roi_wrong.shape}, Max: {roi_wrong.max():.1f}')
    axes[2].plot(roi_size//2, roi_size//2, 'r+', markersize=15, markeredgewidth=2)
    axes[2].set_xlabel('x (columns)')
    axes[2].set_ylabel('y (rows)')

    plt.tight_layout()
    plt.savefig('/home/jbeckwith/Documents/pCloud/Chemistry/Lee/Code/Python/pyBayerSMLM/unit_tests/array_indexing_test.png',
                dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: unit_tests/array_indexing_test.png")

    return test_passed


def test_edge_case_asymmetry():
    """
    Test the specific edge case that was causing asymmetric spot removal.

    This replicates the user's reported issue where spots near x=811
    were being removed because of incorrect [x, y] indexing.
    """
    print("\n" + "=" * 80)
    print("EDGE CASE TEST: Asymmetric Spot Removal")
    print("=" * 80)

    # User's actual image dimensions
    width = 904   # columns (x direction)
    height = 812  # rows (y direction)

    image = np.zeros((height, width), dtype=np.float32)

    print(f"\n1. Image dimensions: {width}x{height} (width x height)")
    print(f"   Array shape: {image.shape} (rows, cols) = (height, width)")

    # Place a spot near the right edge (where user saw problems)
    spot_x = 811  # column (near right edge)
    spot_y = 177  # row (middle of image)
    roi_size = 16

    print(f"\n2. Spot at (x={spot_x}, y={spot_y})")
    print(f"   ROI size: {roi_size}")

    # Calculate boundaries
    xmin = spot_x - roi_size // 2  # 803
    xmax = spot_x + roi_size // 2  # 819
    ymin = spot_y - roi_size // 2  # 169
    ymax = spot_y + roi_size // 2  # 185

    print(f"\n3. ROI boundaries:")
    print(f"   xmin={xmin}, xmax={xmax} (columns)")
    print(f"   ymin={ymin}, ymax={ymax} (rows)")

    # Check if boundaries are valid
    x_valid = (xmin >= 0 and xmax <= width)
    y_valid = (ymin >= 0 and ymax <= height)

    print(f"\n4. Boundary checks:")
    print(f"   X: {xmin} <= x < {xmax} within [0, {width}]? {x_valid}")
    print(f"   Y: {ymin} <= y < {ymax} within [0, {height}]? {y_valid}")

    # Try WRONG indexing [xmin:xmax, ymin:ymax]
    print(f"\n5. Testing WRONG indexing: image[{xmin}:{xmax}, {ymin}:{ymax}]")
    print(f"   This interprets {xmin}:{xmax} as ROW indices")
    print(f"   But we only have {height} rows!")
    print(f"   Since {xmax} > {height}, this would extract wrong data")

    if xmax <= height:
        roi_wrong = image[xmin:xmax, ymin:ymax]
        print(f"   Extracted shape: {roi_wrong.shape}")
        print(f"   Expected: ({roi_size}, {roi_size}), Got: {roi_wrong.shape}")
    else:
        print(f"   ✗ Would extract rows {xmin}:{xmax}, but max row is {height}")
        print(f"   ✗ This causes asymmetric filtering!")

    # Try CORRECT indexing [ymin:ymax, xmin:xmax]
    print(f"\n6. Testing CORRECT indexing: image[{ymin}:{ymax}, {xmin}:{xmax}]")
    print(f"   This interprets {ymin}:{ymax} as ROW indices (valid: 0-{height})")
    print(f"   And {xmin}:{xmax} as COLUMN indices (valid: 0-{width})")

    roi_correct = image[ymin:ymax, xmin:xmax]
    print(f"   Extracted shape: {roi_correct.shape}")
    print(f"   Expected: ({roi_size}, {roi_size}), Got: {roi_correct.shape}")

    # Verdict
    print(f"\n{'=' * 80}")
    print("CONCLUSION:")
    print("=" * 80)
    print("✓ CORRECT [ymin:ymax, xmin:xmax] works for spots at x=811")
    print("✗ WRONG [xmin:xmax, ymin:ymax] fails because it tries to access")
    print(f"  rows 803-819, but image only has {height} rows")
    print("\nThis explains the asymmetric spot removal on the right edge!")
    print("=" * 80)

    return roi_correct.shape == (roi_size, roi_size)


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("ARRAY INDEXING PROOF TEST")
    print("Proving that array[ymin:ymax, xmin:xmax] is correct for")
    print("extracting ROI at image coordinates (x, y)")
    print("=" * 80)

    # Run tests
    test1_passed = test_coordinate_system()
    test2_passed = test_edge_case_asymmetry()

    print("\n" + "=" * 80)
    print("FINAL RESULTS:")
    print("=" * 80)
    print(f"Coordinate system test: {'✓ PASSED' if test1_passed else '✗ FAILED'}")
    print(f"Edge case test: {'✓ PASSED' if test2_passed else '✗ FAILED'}")

    if test1_passed and test2_passed:
        print("\n✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("\nCONCLUSION: array[ymin:ymax, xmin:xmax] is definitively CORRECT")
        print("            for extracting ROI at image coordinates (x, y)")
    else:
        print("\n✗ SOME TESTS FAILED")

    print("=" * 80)
