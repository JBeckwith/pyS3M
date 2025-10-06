# Coordinate System Audit

## Current State (After Fixes)

### 1. Spot Detection Output (`SpotDetectionFunctions.py`)

**Format**: `[row, col]` = `[y, x]`

```python
detected_puncta = np.array(np.where(mask), dtype="int32").T
# detected_puncta[i, 0] = row = y
# detected_puncta[i, 1] = col = x
```

**Source**: Line 748 in `SpotDetectionFunctions.py`:
```python
def mask2points(self, mask: np.ndarray) -> np.ndarray:
    return np.array(np.where(mask), dtype="int32").T
```

### 2. Coordinate Assignment in SR_Functions (`SR_Functions.py`)

**FIXED** - Now correctly handles `[y, x]` format from detection:

```python
# Line 176-177
ycentre = detected_puncta[i, 0]  # First index is row (y)
xcentre = detected_puncta[i, 1]  # Second index is col (x)
```

### 3. Array Indexing (`SR_Functions.py`, `SM_extractionfunctions.py`)

**FIXED** - All ROI extractions now use correct numpy indexing `[y, x]`:

```python
# SR_Functions.py line 202-207
raw_roi = data_for_fitting[ymin:ymax, xmin:xmax]
gain_roi = gain_map[ymin:ymax, xmin:xmax]
# etc.

# SM_extractionfunctions.py line 159
trace_matrix[i, :] = np.sum(
    np.sum(image_stack[:, ymin:ymax, xmin:xmax], axis=-1), axis=-1
)
```

### 4. Localization Data (DataFrame format)

**Format**: Columns `xc` and `yc` where `xc` is x-coordinate, `yc` is y-coordinate

Used in:
- `CoordinateProcessing.py`
- `AIMAlgorithm.py`
- `SM_extractionfunctions.py`
- `DriftCorrectionFunctions.py`

When converted to arrays for clustering:
```python
X = np.vstack([loc_data["xc"], loc_data["yc"]]).T
# X[:, 0] = xc (x-coordinate)
# X[:, 1] = yc (y-coordinate)
```

**This is [x, y] format** - INCONSISTENT with detected_puncta!

### 5. SM_extractionfunctions.py `locations` array

**INCONSISTENCY FOUND!**

```python
# Line 151-152
locations[0, i] = np.nanmean(data["xc"][dbscan_labels == label].to_numpy())
locations[1, i] = np.nanmean(data["yc"][dbscan_labels == label].to_numpy())
# locations[0, i] = x
# locations[1, i] = y
```

**This is [x, y] format** but then used as:
```python
# Line 153-156
xmin = int(locations[0, i]) - int(image_size / 2)  # Correct: x from index 0
ymin = int(locations[1, i]) - int(image_size / 2)  # Correct: y from index 1
# Extract ROI
trace_matrix[i, :] = np.sum(
    np.sum(image_stack[:, ymin:ymax, xmin:xmax], axis=-1), axis=-1
)  # Correct: [y, x] indexing
```

**Status**: This is actually CORRECT! The `locations` array stores [x, y] and uses it correctly.

## Inconsistencies

### Main Inconsistency: Two Different Coordinate Array Formats

1. **Spot Detection**: `detected_puncta[i, 0]` = y, `detected_puncta[i, 1]` = x (from `np.where`)
2. **Localization Data**: `X[i, 0]` = x, `X[i, 1]` = y (from DataFrame columns)

This is potentially confusing but **NOT a bug** as long as:
- Code that handles `detected_puncta` knows it's `[y, x]` ✓ (now fixed)
- Code that handles localization arrays knows they're `[x, y]` ✓ (already correct)

### Why This Inconsistency Exists

- `np.where()` returns `(row_indices, col_indices)` = `(y, x)` - this is numpy convention
- DataFrames have explicit column names `xc`, `yc` - naturally creates `[x, y]` arrays
- These come from different sources and are used in different contexts

## Recommendations

### Option 1: Keep Current System (RECOMMENDED)

**Pros**:
- Follows numpy conventions where applicable
- DataFrame approach is explicit and self-documenting
- Each format is internally consistent
- Now that coordinate bugs are fixed, it works correctly

**Cons**:
- Two different conventions in same codebase
- Requires careful attention when converting between formats

**Required Actions**:
1. ✓ Document this convention clearly (this file)
2. ✓ Add comments wherever coordinate arrays are created/used
3. ✓ Ensure all coordinate handling code is aware of which format it's using

### Option 2: Standardize Everything to [x, y]

Would require:
- Modifying `mask2points` to return `[[col, row]]` instead of `[[row, col]]`
- Or swapping immediately after detection
- More natural for users thinking in (x, y) coordinates

**Cons**:
- Goes against numpy convention (arrays are [row, col])
- Requires changing core detection code
- Risk of introducing new bugs

### Option 3: Standardize Everything to [y, x]

Would require:
- Changing all DataFrame-based coordinate arrays to `[yc, xc]` order
- Updating all clustering and analysis code

**Cons**:
- Less intuitive (x usually comes before y in math/plotting)
- Large code changes required
- Higher risk

## Current Status: ACCEPTABLE

The codebase now has:
1. ✓ Correct handling of `detected_puncta` as `[y, x]`
2. ✓ Correct handling of localization arrays as `[x, y]`
3. ✓ Correct numpy array indexing `[y, x]` everywhere
4. ✓ No mixing of conventions within the same operation

**No further changes required** - the inconsistency is managed and documented.

## Code Comments Added

To prevent future confusion, comments have been added:

```python
# SR_Functions.py line 174-177
# detected_puncta stores [row, col, frame] from np.where()
# row = y, col = x (confirmed by test_real_spot_detection.py)
ycentre = detected_puncta[i, 0]  # First index is row (y)
xcentre = detected_puncta[i, 1]  # Second index is col (x)
```

Similar comments added throughout extraction code.
