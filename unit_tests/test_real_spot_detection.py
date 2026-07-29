#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test using REAL spot detection functions to verify coordinate format.

This test will:
1. Create an image with spots at known locations
2. Use the actual SpotDetectionFunctions to detect them
3. Verify what format detected_puncta actually uses
4. Test ROI extraction with different indexing methods

Created on 2025-10-03
"""

import numpy as np
import sys
import os

# Add src to path

from pyS3M.SpotDetectionFunctions import SpotDetection_Functions


def create_test_image():
    """Create a test image with 3 bright spots at known locations."""
    # Image dimensions (not square to make x/y swaps obvious)
    height = 120  # rows
    width = 150   # cols

    image = np.zeros((height, width), dtype=np.float32)

    # Add background
    image += 10.0

    # True spot locations (x, y) = (column, row)
    true_spots = [
        (40, 30, "Spot 1"),
        (110, 50, "Spot 2"),
        (70, 95, "Spot 3"),
    ]

    # Add Gaussian spots
    sigma = 1.5
    amplitude = 2000

    for x, y, label in true_spots:
        # Create Gaussian centered at (x, y)
        for dy in range(-10, 11):
            for dx in range(-10, 11):
                row = y + dy
                col = x + dx
                if 0 <= row < height and 0 <= col < width:
                    gauss = amplitude * np.exp(-(dx**2 + dy**2) / (2 * sigma**2))
                    image[row, col] += gauss

    # Add noise
    np.random.seed(42)
    image += np.random.normal(0, 5, image.shape)

    # Create variance map (uniform for simplicity)
    variance = np.ones_like(image) * 25.0

    return image, variance, true_spots


def test_real_spot_detection():
    """Test with real spot detection functions."""
    print("=" * 80)
    print("TEST: Real Spot Detection Coordinate Format")
    print("=" * 80)

    # Create test image
    image, variance, true_spots = create_test_image()
    height, width = image.shape

    print(f"\n1. Image: {width}x{height} (width x height)")
    print(f"   Array shape: {image.shape} (rows, cols)")
    print(f"\n2. True spot locations (x, y):")
    for x, y, label in true_spots:
        print(f"   {label}: x={x}, y={y}")
        print(f"            In array: image[{y}, {x}] = {image[y, x]:.1f}")

    # Use real spot detection
    print(f"\n3. Running REAL spot detection...")
    spot_detector = SpotDetection_Functions()

    detected_puncta = spot_detector.detect_puncta_in_image(
        image=image,
        variance=variance,
        pfa=1e-6,
        wavelength=0.65,
        pixel_size=0.069,
        NA=1.49,
    )

    print(f"   Detected {len(detected_puncta)} spots")
    print(f"   detected_puncta shape: {detected_puncta.shape}")
    print(f"   detected_puncta dtype: {detected_puncta.dtype}")

    # Analyze what we got
    print(f"\n4. Analyzing detected coordinates:")
    for i in range(min(5, len(detected_puncta))):
        coord0 = detected_puncta[i, 0]
        coord1 = detected_puncta[i, 1]
        print(f"   Spot {i+1}: [{coord0}, {coord1}]")

        # Check which interpretation makes sense
        # Interpretation A: [0]=x, [1]=y
        if coord0 < width and coord1 < height:
            value_A = image[int(coord1), int(coord0)]
            print(f"      If [row, col] = [y, x]: image[{int(coord1)}, {int(coord0)}] = {value_A:.1f}")

        # Interpretation B: [0]=y, [1]=x
        if coord0 < height and coord1 < width:
            value_B = image[int(coord0), int(coord1)]
            print(f"      If [row, col] = [x, y]: image[{int(coord0)}, {int(coord1)}] = {value_B:.1f}")

    # Compare with true locations
    print(f"\n5. Matching detected spots to true locations:")
    for i, (true_x, true_y, label) in enumerate(true_spots):
        # Find closest detected spot
        if len(detected_puncta) > 0:
            # Try both interpretations

            # Interpretation: detected_puncta is [row, col] = [y, x]
            detected_y_A = detected_puncta[:, 0]
            detected_x_A = detected_puncta[:, 1]
            dist_A = np.sqrt((detected_x_A - true_x)**2 + (detected_y_A - true_y)**2)
            closest_A_idx = np.argmin(dist_A)
            closest_A_dist = dist_A[closest_A_idx]

            # Interpretation: detected_puncta is [x, y]
            detected_x_B = detected_puncta[:, 0]
            detected_y_B = detected_puncta[:, 1]
            dist_B = np.sqrt((detected_x_B - true_x)**2 + (detected_y_B - true_y)**2)
            closest_B_idx = np.argmin(dist_B)
            closest_B_dist = dist_B[closest_B_idx]

            print(f"\n   {label} at true (x={true_x}, y={true_y}):")
            print(f"      If detected_puncta is [y, x]:")
            print(f"         Closest: spot {closest_A_idx+1} at ({detected_x_A[closest_A_idx]:.0f}, {detected_y_A[closest_A_idx]:.0f}), dist={closest_A_dist:.1f}")
            print(f"      If detected_puncta is [x, y]:")
            print(f"         Closest: spot {closest_B_idx+1} at ({detected_x_B[closest_B_idx]:.0f}, {detected_y_B[closest_B_idx]:.0f}), dist={closest_B_dist:.1f}")

            if closest_A_dist < closest_B_dist:
                print(f"      → Format is [y, x] (row, col)")
            else:
                print(f"      → Format is [x, y]")

    print(f"\n{'=' * 80}")
    print("CONCLUSION:")
    print("=" * 80)

    # Determine format by checking which gives better matches
    total_dist_A = 0
    total_dist_B = 0
    for true_x, true_y, label in true_spots:
        detected_y_A = detected_puncta[:, 0]
        detected_x_A = detected_puncta[:, 1]
        dist_A = np.sqrt((detected_x_A - true_x)**2 + (detected_y_A - true_y)**2)
        total_dist_A += np.min(dist_A)

        detected_x_B = detected_puncta[:, 0]
        detected_y_B = detected_puncta[:, 1]
        dist_B = np.sqrt((detected_x_B - true_x)**2 + (detected_y_B - true_y)**2)
        total_dist_B += np.min(dist_B)

    if total_dist_A < total_dist_B:
        print("✓ detected_puncta uses [row, col] = [y, x] format")
        print(f"  Total distance: {total_dist_A:.2f} vs {total_dist_B:.2f}")
        print("\nThis means:")
        print("  detected_puncta[i, 0] = row = y")
        print("  detected_puncta[i, 1] = col = x")
        print("\nFor correct extraction:")
        print("  ycentre = detected_puncta[i, 0]  # Get row")
        print("  xcentre = detected_puncta[i, 1]  # Get col")
        print("  roi = image[ymin:ymax, xmin:xmax]  # Correct numpy indexing")
    else:
        print("✓ detected_puncta uses [x, y] format")
        print(f"  Total distance: {total_dist_B:.2f} vs {total_dist_A:.2f}")
        print("\nThis means:")
        print("  detected_puncta[i, 0] = x")
        print("  detected_puncta[i, 1] = y")
        print("\nFor correct extraction:")
        print("  xcentre = detected_puncta[i, 0]  # Get x")
        print("  ycentre = detected_puncta[i, 1]  # Get y")
        print("  roi = image[ymin:ymax, xmin:xmax]  # Correct numpy indexing")

    print("=" * 80)

    return detected_puncta, true_spots, image


def visualize_detection(image, detected_puncta, true_spots):
    """Create visualization showing detected vs true locations."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    height, width = image.shape

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Test both interpretations
    interpretations = [
        {"name": "[y, x] format", "x_idx": 1, "y_idx": 0},
        {"name": "[x, y] format", "x_idx": 0, "y_idx": 1},
    ]

    for idx, interp in enumerate(interpretations):
        ax = axes[idx]
        ax.imshow(image, cmap='hot', origin='upper', aspect='auto')
        ax.set_title(f'Interpretation: detected_puncta is {interp["name"]}', fontsize=10)
        ax.set_xlabel('x (columns)')
        ax.set_ylabel('y (rows)')

        # Plot true locations (blue squares)
        for x, y, label in true_spots:
            ax.plot(x, y, 'bs', markersize=12, markerfacecolor='none',
                   markeredgewidth=2)
            ax.text(x + 3, y - 3, label, color='blue', fontsize=8)

        # Plot detected locations (green crosses)
        detected_x = detected_puncta[:, interp["x_idx"]]
        detected_y = detected_puncta[:, interp["y_idx"]]

        for i in range(len(detected_puncta)):
            ax.plot(detected_x[i], detected_y[i], 'g+', markersize=15, markeredgewidth=3)
            ax.text(detected_x[i] + 3, detected_y[i] + 3, f'D{i+1}', color='green', fontsize=6)

        # Draw ROI boxes (16x16)
        roi_size = 16
        for i in range(len(detected_puncta)):
            xc = detected_x[i]
            yc = detected_y[i]
            xmin = xc - roi_size/2
            ymin = yc - roi_size/2

            rect = patches.Rectangle((xmin, ymin), roi_size, roi_size,
                                    linewidth=2, edgecolor='cyan', facecolor='none')
            ax.add_patch(rect)

    # Summary panel
    ax = axes[2]
    ax.axis('off')

    summary = "COORDINATE FORMAT TEST\n"
    summary += "=" * 30 + "\n\n"
    summary += f"Image: {width}×{height}\n"
    summary += f"(width × height)\n\n"

    summary += "True spots (blue □):\n"
    for x, y, label in true_spots:
        summary += f"  {label}: ({x}, {y})\n"

    summary += f"\nDetected ({len(detected_puncta)} spots):\n"
    summary += "Green + markers\n\n"

    summary += "Cyan boxes show 16×16 ROI\n"
    summary += "around each detected spot\n\n"

    summary += "LEFT: If format is [y, x]\n"
    summary += "RIGHT: If format is [x, y]\n\n"

    summary += "Correct format: Green +\n"
    summary += "should overlap blue □\n"

    ax.text(0.1, 0.5, summary, fontsize=10, verticalalignment='center',
           family='monospace')

    plt.tight_layout()
    output_path = '/home/jbeckwith/Documents/pCloud/Chemistry/Lee/Code/Python/pyBayerSMLM/unit_tests/real_spot_detection_test.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n6. Visualization saved to: {output_path}")

    return output_path


if __name__ == "__main__":
    detected_puncta, true_spots, image = test_real_spot_detection()
    visualize_detection(image, detected_puncta, true_spots)
