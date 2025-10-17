# EVER Multi-File Bug Report

**Date:** 2025-10-17
**Reporter:** Claude Code
**File:** `src/SR_Functions.py`
**Method:** `fit_imaging_data()` lines 1589-1617

## Summary

EVER background subtraction only processes ~101 frames per 1000-frame chunk, causing most frames to be skipped entirely.

## Bug Description

The current implementation loads a single EVER window (e.g., 100 frames) centered on the chunk's middle frame and tries to extract all chunk frames from this window. This fails because:

1. Chunk size: 1000 frames (e.g., frames 0-999)
2. Chunk middle: frame 500
3. EVER window: frames 450-550 (101 frames)
4. Extraction attempt: tries to extract frames [0:1000] from 101-frame window
5. Result: Only 101 frames processed

## Root Cause

**Line 1590:** `chunk_middle_frame = chunk_start + len(chunk_frames) // 2`

This assumes one EVER window can cover the entire chunk, which is only true if `chunk_size <= ever_window`.

## Expected Behavior

EVER should be applied to **every frame** in the chunk by:
- Loading a window around each target frame
- Computing EVER background for that frame
- Processing all frames 0-999, not just 450-550

## Current Behavior (Incorrect)

```python
# Load ONE window for entire chunk (WRONG)
chunk_middle_frame = chunk_start + len(chunk_frames) // 2  # e.g., 500
ever_frames = load_window(chunk_middle_frame)  # frames 450-550 (101 frames)
ever_result = compute_ever(ever_frames)  # 101 frames
extracted = ever_result[0:1000]  # ONLY GETS 101 FRAMES!
```

## Correct Approach (Options)

### Option 1: Process frame-by-frame
```python
for frame_idx in range(chunk_start, chunk_end):
    ever_frames = load_window_for_frame(frame_idx)  # 100 frames around this frame
    ever_result_for_frame = compute_ever(ever_frames)[center_idx]  # Single frame
    processed_frames.append(ever_result_for_frame)
```

### Option 2: Sliding window over chunk
```python
# Process in batches smaller than EVER window
batch_size = ever_window // 2  # e.g., 50 frames
for batch_start in range(chunk_start, chunk_end, batch_size):
    batch_end = min(batch_start + batch_size, chunk_end)
    for frame_idx in range(batch_start, batch_end):
        ever_frames = load_window_for_frame(frame_idx)
        ...
```

### Option 3: Pre-compute EVER for all frames
```python
# Load all frames in chunk + buffer for EVER
buffer_start = max(0, chunk_start - ever_window//2)
buffer_end = min(total_frames, chunk_end + ever_window//2)
all_frames = load_frames(buffer_start, buffer_end)
ever_result = compute_ever(all_frames)
# Extract just the chunk frames
chunk_ever = ever_result[chunk_start-buffer_start:chunk_end-buffer_start]
```

## Test Evidence

Test: `unit_tests/test_ever_multifile.py`
- Files: 2 × 500 frames = 1000 total
- EVER window: 100 frames
- Chunk size: 1000 frames

**Results:**
- Standard: 4,705 localizations (frames 0-999)
- EVER: 1,518 localizations (frames 0-600, only 202 unique frames)
- **Missing: 798 frames not processed**

**Detection output:**
```
Detecting puncta:   0%|          | 0/101 [00:00<?, ?it/s]
```
Only 101 frames detected (should be 500).

## Impact

- **Severity:** HIGH
- 80% of frames skipped when `chunk_size > ever_window`
- File boundaries handled correctly, but chunk processing broken
- No frame duplication (good), but massive frame loss (bad)

## Recommendation

Implement Option 3 (pre-compute EVER for chunk with buffer) as it:
- Processes all frames correctly
- Minimizes redundant EVER computations
- Handles file boundaries properly (already implemented)
- Maintains memory efficiency (one chunk at a time)

## Related Code

- `_load_frames_for_ever_window()`: Works correctly (loads cross-file windows)
- `_compute_ever_background()`: Works correctly (processes all frames in input)
- **Bug location:** Lines 1589-1617 in `fit_imaging_data()`
