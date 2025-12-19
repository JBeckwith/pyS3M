#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for Bayer spot detection components.

Tests individual functions before integration testing.

Created: December 19, 2025
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
module_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, module_dir)

from BayerSpotDetection import extract_bayer_channels, map_coordinates_to_full_resolution


def test_extract_bayer_channels_rggb():
    """Test Bayer channel extraction with RGGB pattern."""
    print("Test 1: Extract Bayer channels (RGGB pattern)")
    print("-" * 60)

    # Create synthetic Bayer image with known pattern
    H, W = 8, 8
    bayer = np.zeros((H, W), dtype=np.uint16)

    # Assign known values to each channel
    # RGGB pattern:
    #   R at (even, even) -> value 100
    #   G at (even, odd) and (odd, even) -> value 200
    #   B at (odd, odd) -> value 300

    bayer[0::2, 0::2] = 100  # Red
    bayer[0::2, 1::2] = 200  # Green
    bayer[1::2, 0::2] = 200  # Green
    bayer[1::2, 1::2] = 300  # Blue

    print("Input Bayer pattern (8×8):")
    print(bayer)

    # Extract channels
    red, green, blue, coord_info = extract_bayer_channels(bayer, 'RGGB')

    print(f"\nExtracted Red channel (4×4):")
    print(red)
    print(f"Expected all 100: {np.all(red == 100)}")

    print(f"\nExtracted Green channel (8×4):")
    print(green)
    print(f"Expected all 200: {np.all(green == 200)}")

    print(f"\nExtracted Blue channel (4×4):")
    print(blue)
    print(f"Expected all 300: {np.all(blue == 300)}")

    # Verify shapes
    assert red.shape == (H//2, W//2), f"Red shape incorrect: {red.shape}"
    assert green.shape == (H, W//2), f"Green shape incorrect: {green.shape}"
    assert blue.shape == (H//2, W//2), f"Blue shape incorrect: {blue.shape}"

    # Verify values
    assert np.all(red == 100), "Red values incorrect"
    assert np.all(green == 200), "Green values incorrect"
    assert np.all(blue == 300), "Blue values incorrect"

    # Verify coordinate info
    assert coord_info['pattern'] == 'RGGB'
    assert coord_info['red_offset'] == (0, 0)
    assert coord_info['blue_offset'] == (1, 1)

    print("\n✓ PASSED: Bayer channel extraction (RGGB)")
    return True


def test_extract_bayer_other_patterns():
    """Test Bayer channel extraction with other patterns."""
    print("\n" + "="*60)
    print("Test 2: Extract Bayer channels (all patterns)")
    print("-" * 60)

    H, W = 6, 6

    patterns = ['RGGB', 'GRBG', 'GBRG', 'BGGR']

    for pattern in patterns:
        print(f"\nTesting pattern: {pattern}")

        # Create pattern-specific image
        bayer = np.zeros((H, W), dtype=np.uint16)

        if pattern == 'RGGB':
            bayer[0::2, 0::2] = 100; bayer[0::2, 1::2] = 200
            bayer[1::2, 0::2] = 200; bayer[1::2, 1::2] = 300
        elif pattern == 'GRBG':
            bayer[0::2, 0::2] = 200; bayer[0::2, 1::2] = 100
            bayer[1::2, 0::2] = 300; bayer[1::2, 1::2] = 200
        elif pattern == 'GBRG':
            bayer[0::2, 0::2] = 200; bayer[0::2, 1::2] = 300
            bayer[1::2, 0::2] = 100; bayer[1::2, 1::2] = 200
        elif pattern == 'BGGR':
            bayer[0::2, 0::2] = 300; bayer[0::2, 1::2] = 200
            bayer[1::2, 0::2] = 200; bayer[1::2, 1::2] = 100

        red, green, blue, coord_info = extract_bayer_channels(bayer, pattern)

        # Verify extraction
        assert np.all(red == 100), f"Red extraction failed for {pattern}"
        assert np.all(green == 200), f"Green extraction failed for {pattern}"
        assert np.all(blue == 300), f"Blue extraction failed for {pattern}"

        print(f"  ✓ {pattern}: Extraction correct")

    print("\n✓ PASSED: All Bayer patterns")
    return True


def test_coordinate_mapping_rggb():
    """Test coordinate mapping from subsampled to full resolution."""
    print("\n" + "="*60)
    print("Test 3: Coordinate mapping (RGGB)")
    print("-" * 60)

    # Create synthetic detections in subsampled space
    # Format: [y, x, frame] (from detect_puncta_in_stack_parallel)
    red_dets = np.array([
        [0, 0, 0],  # y=0, x=0 sub -> y=0, x=0 full
        [1, 1, 0],  # y=1, x=1 sub -> y=2, x=2 full
    ])

    green_dets = np.array([
        [0, 0, 0],  # y=0, x=0 sub -> y=0, x=1 full (even row)
        [1, 0, 0],  # y=1, x=0 sub -> y=1, x=0 full (odd row)
    ])

    blue_dets = np.array([
        [0, 0, 0],  # y=0, x=0 sub -> y=1, x=1 full
        [1, 1, 0],  # y=1, x=1 sub -> y=3, x=3 full
    ])

    coord_info = {
        'pattern': 'RGGB',
        'red_offset': (0, 0),
        'green_offsets': [(0, 1), (1, 0)],
        'blue_offset': (1, 1)
    }

    # Test red mapping
    print("\nRed detections (checkerboard 2×):")
    red_full = map_coordinates_to_full_resolution(red_dets, 'red', coord_info)
    print(f"  Input:  y={red_dets[:, 0]}, x={red_dets[:, 1]}")
    print(f"  Output: y={red_full[:, 0]}, x={red_full[:, 1]}")
    print(f"  Expected: y=[0, 2], x=[0, 2]")
    assert np.array_equal(red_full[:, 0], [0, 2]), "Red y mapping incorrect"
    assert np.array_equal(red_full[:, 1], [0, 2]), "Red x mapping incorrect"
    print("  ✓ Red mapping correct")

    # Test green mapping (quincunx)
    print("\nGreen detections (quincunx):")
    green_full = map_coordinates_to_full_resolution(green_dets, 'green', coord_info)
    print(f"  Input:  y={green_dets[:, 0]}, x={green_dets[:, 1]}")
    print(f"  Output: y={green_full[:, 0]}, x={green_full[:, 1]}")
    print(f"  Expected: y=[0, 1], x=[1, 0]")
    assert np.array_equal(green_full[:, 0], [0, 1]), "Green y mapping incorrect"
    assert np.array_equal(green_full[:, 1], [1, 0]), "Green x mapping incorrect"
    print("  ✓ Green mapping correct")

    # Test blue mapping
    print("\nBlue detections (checkerboard 2×):")
    blue_full = map_coordinates_to_full_resolution(blue_dets, 'blue', coord_info)
    print(f"  Input:  y={blue_dets[:, 0]}, x={blue_dets[:, 1]}")
    print(f"  Output: y={blue_full[:, 0]}, x={blue_full[:, 1]}")
    print(f"  Expected: y=[1, 3], x=[1, 3]")
    assert np.array_equal(blue_full[:, 0], [1, 3]), "Blue y mapping incorrect"
    assert np.array_equal(blue_full[:, 1], [1, 3]), "Blue x mapping incorrect"
    print("  ✓ Blue mapping correct")

    print("\n✓ PASSED: Coordinate mapping")
    return True


def test_round_trip():
    """Test round-trip: extract channels -> detect at known positions -> map back."""
    print("\n" + "="*60)
    print("Test 4: Round-trip extraction and mapping")
    print("-" * 60)

    # Create Bayer image with a bright spot at known full-resolution position
    H, W = 20, 20
    bayer = np.zeros((H, W), dtype=np.uint16) + 50  # Background

    # Add bright spots at specific full-resolution coordinates
    # Red spot at (4, 6) - should appear at (2, 3) in red channel
    # Green spot at (5, 8) - should appear at (5, 4) in green channel
    # Blue spot at (7, 9) - should appear at (3, 4) in blue channel

    # For RGGB pattern:
    # Red at even rows, even cols
    # Green at (even, odd) and (odd, even)
    # Blue at odd rows, odd cols

    bayer[4, 6] = 1000  # Red position (even, even)
    bayer[5, 8] = 1000  # Green position (odd, even)
    bayer[7, 9] = 1000  # Blue position (odd, odd)

    print("Created Bayer image with bright spots:")
    print(f"  Red at full (4, 6)")
    print(f"  Green at full (5, 8)")
    print(f"  Blue at full (7, 9)")

    # Extract channels
    red, green, blue, coord_info = extract_bayer_channels(bayer, 'RGGB')

    # Find bright pixels in each channel
    red_bright = np.where(red == 1000)
    green_bright = np.where(green == 1000)
    blue_bright = np.where(blue == 1000)

    print("\nFound in subsampled channels:")
    print(f"  Red: y={red_bright[0]}, x={red_bright[1]}")
    print(f"  Green: y={green_bright[0]}, x={green_bright[1]}")
    print(f"  Blue: y={blue_bright[0]}, x={blue_bright[1]}")

    # Map back to full resolution
    # NOTE: detect_puncta_in_stack_parallel returns [y, x, frame] format
    if len(red_bright[0]) > 0:
        red_det = np.array([[red_bright[0][0], red_bright[1][0], 0]])  # [y, x, frame]
        red_full = map_coordinates_to_full_resolution(red_det, 'red', coord_info)
        print(f"\n  Red mapped back to: ({red_full[0, 0]:.0f}, {red_full[0, 1]:.0f})")  # [y, x, frame]
        print(f"    Expected: (4, 6)")
        assert red_full[0, 0] == 4 and red_full[0, 1] == 6, "Red round-trip failed"

    if len(green_bright[0]) > 0:
        green_det = np.array([[green_bright[0][0], green_bright[1][0], 0]])  # [y, x, frame]
        green_full = map_coordinates_to_full_resolution(green_det, 'green', coord_info)
        print(f"  Green mapped back to: ({green_full[0, 0]:.0f}, {green_full[0, 1]:.0f})")
        print(f"    Expected: (5, 8)")
        assert green_full[0, 0] == 5 and green_full[0, 1] == 8, "Green round-trip failed"

    if len(blue_bright[0]) > 0:
        blue_det = np.array([[blue_bright[0][0], blue_bright[1][0], 0]])  # [y, x, frame]
        blue_full = map_coordinates_to_full_resolution(blue_det, 'blue', coord_info)
        print(f"  Blue mapped back to: ({blue_full[0, 0]:.0f}, {blue_full[0, 1]:.0f})")
        print(f"    Expected: (7, 9)")
        assert blue_full[0, 0] == 7 and blue_full[0, 1] == 9, "Blue round-trip failed"

    print("\n✓ PASSED: Round-trip test")
    return True


def main():
    """Run all component tests."""
    print("="*60)
    print("BAYER SPOT DETECTION COMPONENT TESTS")
    print("="*60)

    tests = [
        test_extract_bayer_channels_rggb,
        test_extract_bayer_other_patterns,
        test_coordinate_mapping_rggb,
        test_round_trip
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append((test.__name__, result))
        except Exception as e:
            print(f"\n✗ FAILED: {test.__name__}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            results.append((test.__name__, False))

    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {name}")

    all_passed = all(r[1] for r in results)
    if all_passed:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️  Some tests failed")

    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
