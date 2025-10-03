#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEFINITIVE TEST: Proving the existence of two compensating bugs.

The codebase has TWO bugs that cancel each other out:
1. detected_puncta stores [row, col] but is treated as [x, y]
2. ROI extraction was using [x, y] indexing instead of [y, x]

My recent "fix" to use [y, x] indexing actually BROKE the code by
fixing only ONE of the two compensating bugs!

Created on 2025-10-03
"""

import numpy as np


def test_original_buggy_code():
    """
    Test the ORIGINAL code behavior with compensating bugs.
    """
    print("=" * 80)
    print("TEST 1: ORIGINAL CODE (Two compensating bugs)")
    print("=" * 80)

    # Create image with spot at known location
    image = np.zeros((100, 80), dtype=np.float32)  # 100 rows, 80 columns

    # Place spot at row=60, col=30
    # In image coordinates: y=60, x=30
    true_y = 60
    true_x = 30
    image[true_y, true_x] = 1000

    print(f"\n1. Image: {image.shape} (100 rows, 80 cols)")
    print(f"   Spot at image[{true_y}, {true_x}]")
    print(f"   Image coordinates: y={true_y}, x={true_x}")

    # Simulate detection (using np.where like mask2points)
    detected_puncta = np.array(np.where(image > 500), dtype="int32").T

    print(f"\n2. Detection result:")
    print(f"   detected_puncta = {detected_puncta}")
    print(f"   detected_puncta[0, 0] = {detected_puncta[0, 0]} (this is ROW = y)")
    print(f"   detected_puncta[0, 1] = {detected_puncta[0, 1]} (this is COL = x)")

    # BUG #1: Code treats detected_puncta[i, 0] as x (but it's actually y!)
    i = 0
    xcentre = detected_puncta[i, 0]  # Gets row (y=60) but calls it x
    ycentre = detected_puncta[i, 1]  # Gets col (x=30) but calls it y

    print(f"\n3. BUG #1: Code assigns coordinates backwards:")
    print(f"   xcentre = detected_puncta[{i}, 0] = {xcentre}")
    print(f"   ycentre = detected_puncta[{i}, 1] = {ycentre}")
    print(f"   But detected_puncta[i,0] is ROW (y), not x!")
    print(f"   And detected_puncta[i,1] is COL (x), not y!")
    print(f"   So xcentre={xcentre} is actually y, ycentre={ycentre} is actually x")

    # Calculate ROI boundaries
    roi_size = 16
    xmin = xcentre - roi_size // 2  # Actually y - 8
    xmax = xcentre + roi_size // 2  # Actually y + 8
    ymin = ycentre - roi_size // 2  # Actually x - 8
    ymax = ycentre + roi_size // 2  # Actually x + 8

    print(f"\n4. ROI boundaries (using swapped coordinates):")
    print(f"   xmin = {xmin}, xmax = {xmax} (but these are actually y values!)")
    print(f"   ymin = {ymin}, ymax = {ymax} (but these are actually x values!)")

    # BUG #2 (ORIGINAL): Extract using [xmin:xmax, ymin:ymax]
    roi_original_buggy = image[xmin:xmax, ymin:ymax]

    print(f"\n5. BUG #2 (ORIGINAL): Extract with [xmin:xmax, ymin:ymax]:")
    print(f"   roi = image[{xmin}:{xmax}, {ymin}:{ymax}]")
    print(f"   But xmin/xmax are actually y values, ymin/ymax are x values!")
    print(f"   So this is really: image[y-8:y+8, x-8:x+8]")
    print(f"   Which is image[{true_y-8}:{true_y+8}, {true_x-8}:{true_x+8}]")
    print(f"   ROI shape: {roi_original_buggy.shape}")
    print(f"   ROI max: {roi_original_buggy.max()}")
    print(f"   Contains spot? {roi_original_buggy.max() > 500}")

    print(f"\n{'=' * 80}")
    print("RESULT: Two bugs cancel out - spot is correctly extracted!")
    print("=" * 80)

    return roi_original_buggy


def test_half_fixed_code():
    """
    Test the code with only bug #2 fixed (my recent change).
    """
    print("\n" + "=" * 80)
    print("TEST 2: HALF-FIXED CODE (Only bug #2 fixed)")
    print("=" * 80)

    # Same setup
    image = np.zeros((100, 80), dtype=np.float32)
    true_y = 60
    true_x = 30
    image[true_y, true_x] = 1000

    print(f"\n1. Same image setup:")
    print(f"   Spot at y={true_y}, x={true_x}")

    # Same detection (still has bug #1)
    detected_puncta = np.array(np.where(image > 500), dtype="int32").T

    i = 0
    xcentre = detected_puncta[i, 0]  # Still getting y=60
    ycentre = detected_puncta[i, 1]  # Still getting x=30

    print(f"\n2. BUG #1 still present:")
    print(f"   xcentre = {xcentre} (actually y)")
    print(f"   ycentre = {ycentre} (actually x)")

    roi_size = 16
    xmin = xcentre - roi_size // 2
    xmax = xcentre + roi_size // 2
    ymin = ycentre - roi_size // 2
    ymax = ycentre + roi_size // 2

    # "FIX" #2: Extract using [ymin:ymax, xmin:xmax] (my recent change)
    roi_half_fixed = image[ymin:ymax, xmin:xmax]

    print(f"\n3. \"FIX\" #2: Extract with [ymin:ymax, xmin:xmax]:")
    print(f"   roi = image[{ymin}:{ymax}, {xmin}:{xmax}]")
    print(f"   But ymin/ymax are actually x values, xmin/xmax are y values!")
    print(f"   So this is: image[x-8:x+8, y-8:y+8]")
    print(f"   Which is image[{true_x-8}:{true_x+8}, {true_y-8}:{true_y+8}]")
    print(f"   ROI shape: {roi_half_fixed.shape}")
    print(f"   ROI max: {roi_half_fixed.max()}")
    print(f"   Contains spot? {roi_half_fixed.max() > 500}")

    print(f"\n{'=' * 80}")
    print("RESULT: Fixing only one bug breaks the code!")
    print("=" * 80)

    return roi_half_fixed


def test_fully_fixed_code():
    """
    Test with BOTH bugs fixed.
    """
    print("\n" + "=" * 80)
    print("TEST 3: FULLY FIXED CODE (Both bugs fixed)")
    print("=" * 80)

    # Same setup
    image = np.zeros((100, 80), dtype=np.float32)
    true_y = 60
    true_x = 30
    image[true_y, true_x] = 1000

    print(f"\n1. Same image setup:")
    print(f"   Spot at y={true_y}, x={true_x}")

    # Detection still returns [row, col]
    detected_puncta = np.array(np.where(image > 500), dtype="int32").T

    # FIX #1: Correctly interpret detected_puncta
    i = 0
    ycentre = detected_puncta[i, 0]  # Correctly get row (y)
    xcentre = detected_puncta[i, 1]  # Correctly get col (x)

    print(f"\n2. FIX #1: Correctly assign coordinates:")
    print(f"   ycentre = detected_puncta[{i}, 0] = {ycentre} (row = y)")
    print(f"   xcentre = detected_puncta[{i}, 1] = {xcentre} (col = x)")

    roi_size = 16
    xmin = xcentre - roi_size // 2
    xmax = xcentre + roi_size // 2
    ymin = ycentre - roi_size // 2
    ymax = ycentre + roi_size // 2

    print(f"\n3. ROI boundaries (correct):")
    print(f"   xmin = {xmin}, xmax = {xmax} (x values)")
    print(f"   ymin = {ymin}, ymax = {ymax} (y values)")

    # FIX #2: Extract using correct indexing
    roi_fully_fixed = image[ymin:ymax, xmin:xmax]

    print(f"\n4. FIX #2: Extract with [ymin:ymax, xmin:xmax]:")
    print(f"   roi = image[{ymin}:{ymax}, {xmin}:{xmax}]")
    print(f"   This is: image[y-8:y+8, x-8:x+8]")
    print(f"   Which is image[{true_y-8}:{true_y+8}, {true_x-8}:{true_x+8}]")
    print(f"   ROI shape: {roi_fully_fixed.shape}")
    print(f"   ROI max: {roi_fully_fixed.max()}")
    print(f"   Contains spot? {roi_fully_fixed.max() > 500}")

    print(f"\n{'=' * 80}")
    print("RESULT: Both bugs fixed - spot is correctly extracted!")
    print("=" * 80)

    return roi_fully_fixed


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("DEFINITIVE TEST: Two Compensating Bugs")
    print("=" * 80)

    roi1 = test_original_buggy_code()
    roi2 = test_half_fixed_code()
    roi3 = test_fully_fixed_code()

    print("\n" + "=" * 80)
    print("SUMMARY:")
    print("=" * 80)
    print(f"Original code (2 bugs):     spot found = {roi1.max() > 500}")
    print(f"Half-fixed (1 bug):         spot found = {roi2.max() > 500}")
    print(f"Fully fixed (0 bugs):       spot found = {roi3.max() > 500}")

    print("\n" + "=" * 80)
    print("CONCLUSION:")
    print("=" * 80)
    print("The original code had TWO bugs that canceled out:")
    print("  BUG #1: xcentre = detected_puncta[i, 0]  # Should be [i, 1]")
    print("  BUG #2: roi = image[xmin:xmax, ymin:ymax]  # Should be [ymin:ymax, xmin:xmax]")
    print("\nMy recent \"fix\" only fixed BUG #2, which BROKE the code!")
    print("\nTo properly fix: EITHER revert bug #2 fix, OR also fix bug #1")
    print("=" * 80)
