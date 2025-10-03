#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test to prove that spot detection returns (x, y) coordinates.

This test demonstrates that:
1. np.where() returns (row_indices, col_indices)
2. mask2points transposes this to get [row, col] pairs
3. But the documentation says it returns "xy coordinates"
4. Therefore the first column is actually y (row) and second is x (col)

Wait... let me verify this carefully!

Created on 2025-10-03
"""

import numpy as np


def test_np_where_behavior():
    """
    Test what np.where() actually returns.
    """
    print("=" * 80)
    print("TESTING np.where() BEHAVIOR")
    print("=" * 80)

    # Create a simple test mask
    mask = np.zeros((5, 7), dtype=bool)  # 5 rows, 7 columns

    # Set one pixel to True at a known location
    # Let's use row=2, col=4
    mask[2, 4] = True

    print(f"\n1. Created mask with shape {mask.shape} (5 rows, 7 cols)")
    print(f"   Set mask[2, 4] = True")
    print(f"   This is row 2, column 4")

    # Use np.where to find the True pixel
    result = np.where(mask)

    print(f"\n2. np.where(mask) returns:")
    print(f"   result = {result}")
    print(f"   result[0] = {result[0]} (first element)")
    print(f"   result[1] = {result[1]} (second element)")

    print(f"\n3. What do these mean?")
    print(f"   result[0][0] = {result[0][0]} <- This is the ROW index")
    print(f"   result[1][0] = {result[1][0]} <- This is the COLUMN index")

    # Now transpose it like mask2points does
    coords = np.array(result).T

    print(f"\n4. After transpose (like mask2points does):")
    print(f"   coords = {coords}")
    print(f"   coords[0, 0] = {coords[0, 0]} <- First coordinate (row)")
    print(f"   coords[0, 1] = {coords[0, 1]} <- Second coordinate (col)")

    print(f"\n5. Interpretation:")
    print(f"   If we call this 'xy coordinates' as the docstring says,")
    print(f"   then coords[i, 0] would be 'x' and coords[i, 1] would be 'y'")
    print(f"   But we know coords[0, 0] = {coords[0, 0]} is the ROW")
    print(f"   And coords[0, 1] = {coords[0, 1]} is the COLUMN")
    print(f"\n   Since rows are Y and columns are X:")
    print(f"   coords[i, 0] = {coords[0, 0]} = row = Y")
    print(f"   coords[i, 1] = {coords[0, 1]} = col = X")

    print(f"\n{'=' * 80}")
    print("CRITICAL FINDING:")
    print("=" * 80)
    print("The docstring says 'xy coordinates' but np.where returns (rows, cols)!")
    print("After transpose, we get [[row, col]] = [[y, x]] NOT [[x, y]]!")
    print(f"\nSo detected_puncta[i, 0] = {coords[0, 0]} is Y (row)")
    print(f"And detected_puncta[i, 1] = {coords[0, 1]} is X (col)")
    print("=" * 80)

    return coords


def test_with_spot_detection_logic():
    """
    Test using the actual mask2points function logic.
    """
    print("\n" + "=" * 80)
    print("TESTING ACTUAL SPOT DETECTION BEHAVIOR")
    print("=" * 80)

    # Create an image with a known bright spot
    image = np.zeros((80, 100), dtype=np.float32)  # 80 rows, 100 cols

    # Place spot at row=30, col=60
    image[30, 60] = 1000

    print(f"\n1. Image shape: {image.shape} (80 rows, 100 cols)")
    print(f"   Placed bright pixel at image[30, 60]")
    print(f"   This is row=30, col=60 in array indexing")
    print(f"   Which corresponds to y=30, x=60 in image coordinates")

    # Simulate detection: threshold image
    mask = image > 500

    # Use mask2points logic
    coords = np.array(np.where(mask), dtype="int32").T

    print(f"\n2. Detection found {len(coords)} spot(s)")
    print(f"   coords = {coords}")
    print(f"   coords[0, 0] = {coords[0, 0]}")
    print(f"   coords[0, 1] = {coords[0, 1]}")

    print(f"\n3. Question: Are these stored as [x, y] or [y, x]?")
    print(f"   We placed the spot at row=30, col=60")
    print(f"   coords[0, 0] = {coords[0, 0]} <- This matches row=30")
    print(f"   coords[0, 1] = {coords[0, 1]} <- This matches col=60")
    print(f"\n   Therefore: coords[i, 0] = row = y")
    print(f"              coords[i, 1] = col = x")

    print(f"\n4. But the code uses: xcentre = detected_puncta[i, 0]")
    print(f"                       ycentre = detected_puncta[i, 1]")
    print(f"\n   If coords are stored as [row, col] = [y, x]:")
    print(f"   xcentre = detected_puncta[i, 0] = {coords[0, 0]} = row = y  <- WRONG!")
    print(f"   ycentre = detected_puncta[i, 1] = {coords[0, 1]} = col = x  <- WRONG!")

    print(f"\n{'=' * 80}")
    print("POTENTIAL BUG FOUND!")
    print("=" * 80)
    print("The code assigns:")
    print("  xcentre = detected_puncta[i, 0]  # Gets row (y)")
    print("  ycentre = detected_puncta[i, 1]  # Gets col (x)")
    print("\nBut it should be:")
    print("  xcentre = detected_puncta[i, 1]  # Get col (x)")
    print("  ycentre = detected_puncta[i, 0]  # Get row (y)")
    print("=" * 80)

    return coords


def verify_with_actual_code():
    """
    Check how the code actually uses detected_puncta.
    """
    print("\n" + "=" * 80)
    print("CHECKING ACTUAL CODE USAGE")
    print("=" * 80)

    # Simulate SR_Functions.py line 174-175:
    # xcentre = detected_puncta[i, 0]
    # ycentre = detected_puncta[i, 1]

    # Create simulated detected_puncta using np.where logic
    mask = np.zeros((100, 80), dtype=bool)
    mask[60, 30] = True  # row=60, col=30

    detected_puncta = np.array(np.where(mask), dtype="int32").T

    print(f"\n1. Simulated detection on 100x80 image (100 rows, 80 cols)")
    print(f"   Spot placed at mask[60, 30] (row=60, col=30)")

    i = 0
    xcentre = detected_puncta[i, 0]
    ycentre = detected_puncta[i, 1]

    print(f"\n2. Code does:")
    print(f"   xcentre = detected_puncta[{i}, 0] = {xcentre}")
    print(f"   ycentre = detected_puncta[{i}, 1] = {ycentre}")

    print(f"\n3. Analysis:")
    print(f"   np.where returns (row_indices, col_indices)")
    print(f"   After transpose: [[row, col]] = [[{xcentre}, {ycentre}]]")
    print(f"   But code assigns: xcentre={xcentre}, ycentre={ycentre}")
    print(f"\n   Since we placed spot at row=60, col=30:")
    print(f"   xcentre should be 30 (col), but got {xcentre} (row)")
    print(f"   ycentre should be 60 (row), but got {ycentre} (col)")

    print(f"\n   X and Y are SWAPPED in the current code!")

    print(f"\n{'=' * 80}")
    print("CONCLUSION:")
    print("=" * 80)
    print("detected_puncta stores coordinates as [row, col] = [y, x]")
    print("But the code treats them as [x, y]")
    print("\nThis means:")
    print("  xcentre = detected_puncta[i, 0]  gets Y (should get X)")
    print("  ycentre = detected_puncta[i, 1]  gets X (should get Y)")
    print("\nSo when we later extract ROI with:")
    print("  data[ymin:ymax, xmin:xmax]")
    print("We're actually using swapped coordinates!")
    print("\nThe SWAP CANCELS OUT - that's why it seems to work!")
    print("=" * 80)


if __name__ == "__main__":
    coords1 = test_np_where_behavior()
    coords2 = test_with_spot_detection_logic()
    verify_with_actual_code()

    print("\n" + "=" * 80)
    print("FINAL ANALYSIS:")
    print("=" * 80)
    print("1. np.where() returns (rows, cols)")
    print("2. After transpose: detected_puncta has [row, col] = [y, x]")
    print("3. Code assigns: xcentre = [i,0] = row = y  (SWAPPED!)")
    print("                 ycentre = [i,1] = col = x  (SWAPPED!)")
    print("4. ROI extraction uses: array[ymin:ymax, xmin:xmax]")
    print("   But xmin/xmax are actually y values, ymin/ymax are actually x values!")
    print("5. So it becomes: array[x_values, y_values] (SWAPPED BACK!)")
    print("\nTwo wrongs made a right! The code has compensating bugs.")
    print("=" * 80)
