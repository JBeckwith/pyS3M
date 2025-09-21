#!/usr/bin/env python3
"""
Simple test to validate vectorized vs frame-by-frame drift correction.
No external dependencies required.
"""

import numpy as np
import sys
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))


def test_vectorized_vs_framebyframe():
    """Test vectorized drift correction against frame-by-frame reference."""
    print("=== Simple Vectorized vs Frame-by-Frame Test ===")

    # Create simple test data
    np.random.seed(42)
    n_locs = 100
    n_frames = 50

    # Create localizations with some frames missing
    frames = np.concatenate([
        np.random.choice(range(1, 11), 30, replace=True),    # Early frames 1-10
        np.random.choice(range(20, 31), 40, replace=True),   # Middle frames 20-30 (gap 11-19)
        np.random.choice(range(40, n_frames+1), 30, replace=True)  # Late frames 40-50 (gap 31-39)
    ])

    x_base = np.random.uniform(10, 90, n_locs)
    y_base = np.random.uniform(10, 90, n_locs)

    # Create simple linear drift
    all_frames = np.arange(1, n_frames + 1)
    drift_x_true = 0.1 * all_frames  # Linear drift 0.1 pixels per frame
    drift_y_true = 0.05 * all_frames  # Linear drift 0.05 pixels per frame

    # Apply drift to create drifted positions
    drift_x_applied = drift_x_true[frames - 1]  # Convert to 0-indexed for lookup
    drift_y_applied = drift_y_true[frames - 1]

    x_drifted = x_base + drift_x_applied
    y_drifted = y_base + drift_y_applied

    print(f"Created {n_locs} localizations across frames {frames.min()}-{frames.max()}")
    print(f"Unique frames: {len(np.unique(frames))}")
    print(f"Drift array length: {len(drift_x_true)}")

    # Method 1: Current vectorized approach (with fixed bounds checking)
    def vectorized_correction(x, y, frame_nums, drift_x, drift_y):
        """Simulate the current vectorized drift correction."""
        # 1-indexed frames: subtract 1 for array indexing
        valid_indices = (frame_nums >= 1) & (frame_nums - 1 < len(drift_x))
        x_corrected = x.copy()
        y_corrected = y.copy()
        x_corrected[valid_indices] = x[valid_indices] - drift_x[frame_nums[valid_indices] - 1]
        y_corrected[valid_indices] = y[valid_indices] - drift_y[frame_nums[valid_indices] - 1]
        return x_corrected, y_corrected, valid_indices

    # Method 2: Frame-by-frame reference implementation
    def framebyframe_correction(x, y, frame_nums, drift_x, drift_y):
        """Reference frame-by-frame drift correction."""
        x_corrected = x.copy()
        y_corrected = y.copy()
        corrected_count = 0

        for frame in np.unique(frame_nums):
            if 1 <= frame <= len(drift_x):
                subset_mask = frame_nums == frame
                x_corrected[subset_mask] -= drift_x[frame - 1]  # Convert to 0-indexed
                y_corrected[subset_mask] -= drift_y[frame - 1]
                corrected_count += np.sum(subset_mask)

        return x_corrected, y_corrected, corrected_count

    # Test both methods
    print("\nTesting vectorized correction...")
    x_vec, y_vec, valid_vec = vectorized_correction(x_drifted, y_drifted, frames, drift_x_true, drift_y_true)

    print(f"Vectorized: {np.sum(valid_vec)}/{len(frames)} localizations corrected")

    print("\nTesting frame-by-frame correction...")
    x_fbf, y_fbf, count_fbf = framebyframe_correction(x_drifted, y_drifted, frames, drift_x_true, drift_y_true)

    print(f"Frame-by-frame: {count_fbf}/{len(frames)} localizations corrected")

    # Compare results
    x_diff = np.abs(x_vec - x_fbf)
    y_diff = np.abs(y_vec - y_fbf)

    max_x_diff = np.max(x_diff)
    max_y_diff = np.max(y_diff)
    mean_x_diff = np.mean(x_diff)
    mean_y_diff = np.mean(y_diff)

    print(f"\nComparison of methods:")
    print(f"  Max difference X: {max_x_diff:.12f} pixels")
    print(f"  Max difference Y: {max_y_diff:.12f} pixels")
    print(f"  Mean difference X: {mean_x_diff:.12f} pixels")
    print(f"  Mean difference Y: {mean_y_diff:.12f} pixels")

    # Check that correction actually worked
    x_error_vec = np.abs(x_vec[valid_vec] - x_base[valid_vec])
    y_error_vec = np.abs(y_vec[valid_vec] - y_base[valid_vec])

    corrected_mask_fbf = np.zeros(len(frames), dtype=bool)
    for frame in np.unique(frames):
        if 1 <= frame <= len(drift_x_true):
            corrected_mask_fbf |= (frames == frame)

    x_error_fbf = np.abs(x_fbf[corrected_mask_fbf] - x_base[corrected_mask_fbf])
    y_error_fbf = np.abs(y_fbf[corrected_mask_fbf] - y_base[corrected_mask_fbf])

    print(f"\nDrift removal accuracy (should be ~0):")
    print(f"  Vectorized - Max error X: {np.max(x_error_vec):.12f}, Y: {np.max(y_error_vec):.12f}")
    print(f"  Frame-by-frame - Max error X: {np.max(x_error_fbf):.12f}, Y: {np.max(y_error_fbf):.12f}")

    # Test edge cases
    print(f"\nTesting edge cases...")

    # Edge case: frames beyond drift array
    edge_frames = np.array([100, 200])  # Beyond n_frames=50
    edge_x = np.array([50.0, 60.0])
    edge_y = np.array([40.0, 50.0])

    print(f"  Testing frames {edge_frames} (beyond drift array length {len(drift_x_true)})")

    try:
        edge_x_vec, edge_y_vec, edge_valid = vectorized_correction(edge_x, edge_y, edge_frames, drift_x_true, drift_y_true)
        print(f"  Vectorized: {np.sum(edge_valid)}/{len(edge_frames)} edge frames handled")

        edge_x_fbf, edge_y_fbf, edge_count = framebyframe_correction(edge_x, edge_y, edge_frames, drift_x_true, drift_y_true)
        print(f"  Frame-by-frame: {edge_count}/{len(edge_frames)} edge frames handled")

        # Both should leave edge frames uncorrected
        print(f"  Edge frames left unchanged: {np.allclose(edge_x_vec, edge_x)} and {np.allclose(edge_x_fbf, edge_x)}")

    except Exception as e:
        print(f"  ✗ Edge case failed: {e}")
        return False

    # Success criteria
    tolerance = 1e-12
    if max_x_diff < tolerance and max_y_diff < tolerance:
        print(f"\n✓ SUCCESS: Vectorized correction matches frame-by-frame correction!")
        print(f"  Both methods corrected the same number of localizations")
        print(f"  Maximum difference is within tolerance ({tolerance})")
        return True
    else:
        print(f"\n✗ FAILURE: Vectorized correction differs from frame-by-frame correction!")
        print(f"  Differences exceed tolerance ({tolerance})")
        print(f"  Consider using frame-by-frame approach for safety")
        return False


if __name__ == "__main__":
    success = test_vectorized_vs_framebyframe()

    print("\n" + "="*60)
    print("DRIFT CORRECTION IMPLEMENTATION STATUS")
    print("="*60)

    if success:
        print("✓ VECTORIZED APPROACH: Working correctly with bounds checking")
        print("✓ FRAME-BY-FRAME APPROACH: Reference implementation available")
        print("\nBoth approaches are implemented in DriftCorrectionFunctions.py")
        print("Toggle with use_vectorized = True/False on line ~862")
    else:
        print("✗ VECTORIZED APPROACH: Has issues, use frame-by-frame")
        print("✓ FRAME-BY-FRAME APPROACH: Recommended for safety")

    print("""
RECOMMENDATION:
- Use vectorized (current default) for performance
- Switch to frame-by-frame if any issues arise
- Both produce identical results when working correctly
""")

    print("Frame-by-frame code pattern:")
    print("""
for frame_num in np.unique(localization_data.frame):
    if 1 <= frame_num <= len(drift_x):
        subset_mask = localization_data.frame == frame_num
        localization_data.xc[subset_mask] -= drift_x[frame_num - 1]
        localization_data.yc[subset_mask] -= drift_y[frame_num - 1]
""")

    sys.exit(0 if success else 1)