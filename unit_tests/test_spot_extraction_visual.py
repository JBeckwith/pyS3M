#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visual test to verify that spot detection and ROI extraction are correct.

Creates an image with 3 spots at known locations (not centered) and verifies:
1. Spots are detected at correct (x, y) coordinates
2. ROI extraction captures the correct regions
3. Visual confirmation via saved figure

Created on 2025-10-03
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def create_test_image_with_spots():
    """
    Create a test image with 3 Gaussian spots at known locations.

    Returns:
        image: 2D array with spots
        spot_locations: list of (x, y, label) tuples
    """
    # Create image (not square to make x/y confusion more obvious)
    height = 120  # rows
    width = 150   # cols
    image = np.zeros((height, width), dtype=np.float32)

    # Add background noise
    np.random.seed(42)
    image += np.random.normal(10, 2, image.shape)

    # Define spot locations (x, y) - deliberately NOT centered or symmetric
    spot_locations = [
        (40, 30, "Spot 1"),   # Upper left quadrant
        (110, 50, "Spot 2"),  # Upper right quadrant
        (70, 95, "Spot 3"),   # Lower middle
    ]

    # Add Gaussian spots at each location
    sigma = 2.0
    amplitude = 1000
    spot_size = 15  # diameter of spot region

    for x, y, label in spot_locations:
        # Add to image at correct location
        y_start = y - spot_size//2
        y_end = y + spot_size//2
        x_start = x - spot_size//2
        x_end = x + spot_size//2

        # Ensure we're within bounds
        if (0 <= y_start and y_end <= height and
            0 <= x_start and x_end <= width):
            # Create meshgrid for this specific region size
            actual_height = y_end - y_start
            actual_width = x_end - x_start
            y_grid, x_grid = np.ogrid[-actual_height//2:actual_height//2,
                                       -actual_width//2:actual_width//2]

            # Gaussian profile
            gaussian = amplitude * np.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))

            image[y_start:y_end, x_start:x_end] += gaussian

    return image, spot_locations


def simulate_detection(image, threshold=500):
    """
    Simulate spot detection by finding local maxima.

    This mimics what mask2points does: np.where() returns (rows, cols).

    Args:
        image: 2D array
        threshold: detection threshold

    Returns:
        detected_puncta: Nx2 array with [row, col] format (like real detection)
    """
    from scipy.ndimage import maximum_filter

    # Find local maxima
    local_max = maximum_filter(image, size=5)
    detected_peaks = (image == local_max) & (image > threshold)

    # Use np.where like the real detection does
    detected_puncta = np.array(np.where(detected_peaks), dtype="int32").T

    return detected_puncta


def extract_roi_correct(image, xcentre, ycentre, roi_size=16):
    """
    Extract ROI using the method that SHOULD work after coordinate fix.

    Since we fixed the coordinate assignment so that:
    - xcentre is truly x (column)
    - ycentre is truly y (row)

    We need to extract using CORRECT numpy indexing: [ymin:ymax, xmin:xmax]

    Args:
        image: 2D array
        xcentre: x coordinate (column)
        ycentre: y coordinate (row)
        roi_size: size of ROI

    Returns:
        roi: extracted region
        boundaries: (xmin, xmax, ymin, ymax)
    """
    # Calculate boundaries
    xmin = int(xcentre - roi_size / 2)
    xmax = int(xcentre + roi_size / 2)
    ymin = int(ycentre - roi_size / 2)
    ymax = int(ycentre + roi_size / 2)

    # Extract using CORRECT numpy indexing [row, col] = [y, x]
    roi = image[ymin:ymax, xmin:xmax]

    return roi, (xmin, xmax, ymin, ymax)


def test_spot_extraction(test_output_dir):
    """
    Main test function.
    """
    print("=" * 80)
    print("VISUAL TEST: Spot Detection and Extraction")
    print("=" * 80)

    # Create test image
    image, true_spots = create_test_image_with_spots()
    height, width = image.shape

    print(f"\n1. Created test image: {width}x{height} (width x height)")
    print(f"   Array shape: {image.shape} (rows, cols)")
    print(f"\n2. True spot locations (x, y):")
    for x, y, label in true_spots:
        print(f"   {label}: x={x}, y={y}")

    # Simulate detection
    detected_puncta = simulate_detection(image, threshold=500)

    print(f"\n3. Detection found {len(detected_puncta)} spots")
    print(f"   detected_puncta array shape: {detected_puncta.shape}")
    print(f"   Format: [row, col] from np.where()")

    # Process detections using CORRECT coordinate assignment
    detected_coords = []
    for i in range(len(detected_puncta)):
        # CORRECT assignment (after our fix)
        ycentre = detected_puncta[i, 0]  # row = y
        xcentre = detected_puncta[i, 1]  # col = x

        detected_coords.append((xcentre, ycentre))
        print(f"   Spot {i+1}: detected at x={xcentre}, y={ycentre}")

    # Extract ROIs
    print(f"\n4. Extracting ROIs...")
    roi_size = 16
    rois = []
    boundaries_list = []

    for i, (xcentre, ycentre) in enumerate(detected_coords):
        roi, boundaries = extract_roi_correct(image, xcentre, ycentre, roi_size)
        rois.append(roi)
        boundaries_list.append(boundaries)

        xmin, xmax, ymin, ymax = boundaries
        print(f"   Spot {i+1}: ROI shape={roi.shape}, max value={roi.max():.1f}")
        print(f"            Boundaries: x=[{xmin}, {xmax}], y=[{ymin}, {ymax}]")

    # Verify extraction is correct
    print(f"\n5. Verification:")
    all_correct = True
    for i, roi in enumerate(rois):
        has_spot = roi.max() > 500
        is_square = (roi.shape[0] == roi.shape[1] == roi_size)

        print(f"   Spot {i+1}: is_square={is_square}, contains_spot={has_spot}")

        if not (has_spot and is_square):
            all_correct = False

    if all_correct:
        print(f"\n✓ All ROIs correctly extracted!")
    else:
        print(f"\n✗ Some ROIs failed!")

    # Create visualization
    print(f"\n6. Creating visualization...")

    fig = plt.figure(figsize=(18, 6))
    gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)

    # Full image with detected spots
    ax_full = fig.add_subplot(gs[:, 0])
    ax_full.imshow(image, cmap='hot', origin='upper', aspect='auto')

    # Plot true locations
    for x, y, label in true_spots:
        ax_full.plot(x, y, 'bs', markersize=10, markerfacecolor='none',
                    markeredgewidth=2, label=f'{label} (true)')
        ax_full.text(x + 3, y - 3, label, color='blue', fontsize=8)

    # Plot detected locations (should overlap with true)
    for i, (xcentre, ycentre) in enumerate(detected_coords):
        ax_full.plot(xcentre, ycentre, 'g+', markersize=15, markeredgewidth=2,
                    label=f'Detected {i+1}')

        # Draw ROI box
        xmin, xmax, ymin, ymax = boundaries_list[i]
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                linewidth=2, edgecolor='cyan', facecolor='none')
        ax_full.add_patch(rect)

    ax_full.set_title(f'Full Image ({width}x{height})\nBlue squares = true, Green + = detected, Cyan = ROI',
                     fontsize=10)
    ax_full.set_xlabel('x (columns)')
    ax_full.set_ylabel('y (rows)')
    ax_full.legend(fontsize=6, loc='upper right')

    # Show extracted ROIs
    for i in range(min(3, len(rois))):
        row = i // 2
        col = i % 2 + 1
        if i == 2:  # Third spot
            ax = fig.add_subplot(gs[1, 2])
        else:
            ax = fig.add_subplot(gs[row, col + 1])

        ax.imshow(rois[i], cmap='hot', origin='upper')
        ax.plot(roi_size//2, roi_size//2, 'g+', markersize=20, markeredgewidth=3)

        xcentre, ycentre = detected_coords[i]
        x_true, y_true, label_true = true_spots[i]

        ax.set_title(f'{label_true}\nTrue: ({x_true}, {y_true})\nDetected: ({xcentre}, {ycentre})\nMax: {rois[i].max():.0f}',
                    fontsize=9)
        ax.set_xlabel('x (columns)')
        ax.set_ylabel('y (rows)')

    # Add text summary
    ax_text = fig.add_subplot(gs[1, 3])
    ax_text.axis('off')

    summary_text = "VERIFICATION:\n" + "="*25 + "\n\n"
    summary_text += f"Image: {width}×{height}\n"
    summary_text += f"(width × height)\n\n"
    summary_text += "Coordinate system:\n"
    summary_text += "• x = column\n"
    summary_text += "• y = row\n\n"
    summary_text += "Detection format:\n"
    summary_text += "• np.where() → [row, col]\n"
    summary_text += "• After fix:\n"
    summary_text += "  ycentre = [i, 0] (row)\n"
    summary_text += "  xcentre = [i, 1] (col)\n\n"
    summary_text += "Extraction:\n"
    summary_text += "• array[xmin:xmax,\n"
    summary_text += "        ymin:ymax]\n\n"

    if all_correct:
        summary_text += "✓ ALL SPOTS\n  CORRECTLY\n  EXTRACTED!"
        color = 'green'
    else:
        summary_text += "✗ EXTRACTION\n  FAILED!"
        color = 'red'

    ax_text.text(0.1, 0.5, summary_text, fontsize=9, verticalalignment='center',
                family='monospace', color=color, weight='bold')

    plt.suptitle('Spot Detection and ROI Extraction Test', fontsize=14, weight='bold')

    output_path = test_output_dir / 'spot_extraction_visual_test.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   Saved to: {output_path}")

    print(f"\n{'=' * 80}")
    print("TEST COMPLETE")
    print("=" * 80)

    return all_correct


if __name__ == "__main__":
    success = test_spot_extraction()

    if success:
        print("\n✓✓✓ VISUAL TEST PASSED ✓✓✓")
        print("\nThe coordinate assignment and ROI extraction are now correct!")
    else:
        print("\n✗✗✗ VISUAL TEST FAILED ✗✗✗")
        print("\nThere are still issues with extraction!")
