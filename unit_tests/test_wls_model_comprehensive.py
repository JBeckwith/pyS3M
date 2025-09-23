#!/usr/bin/env python3
"""
Comprehensive test of WLS_model_nobounds optimization against original implementation.
Tests across wide range of parameters to ensure numerical identity.
"""

import numpy as np
import sys
import os
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

import gaussoptfuncs
from numba import jit

@jit(nopython=True, nogil=True)
def original_WLS_model_nobounds(
    params,
    data,
    masks,
    background_bayer_matrix,
    bayer_matrix,
    x,
    gauss_2d,
):
    """
    Original implementation for comparison testing.
    """
    for i in np.arange(masks.shape[-1]):
        pixels = masks[:, :, i].ravel()
        bayer_matrix[pixels] = params[-3 + i] ** 2
        background_bayer_matrix[pixels] = params[-6 + i] ** 2
    bayer_matrix = bayer_matrix.reshape(len(x), len(x))
    background_bayer_matrix = background_bayer_matrix.reshape(len(x), len(x))
    gauss_2d[:, :] = (
        np.multiply(
            bayer_matrix,
            gaussoptfuncs.gaussian_unscaled_model(
                gauss_2d[:, :],
                x,
                len(x),
                params[0],
                params[1],
                params[2],
                params[3],
            ),
        )
        + background_bayer_matrix
    )
    return gauss_2d

def test_parameter_ranges():
    """Test across wide range of realistic parameters."""
    print("=== Comprehensive Parameter Range Testing ===")

    image_size = 16
    x = np.arange(image_size)

    # Create Bayer masks
    bayer_masks = np.zeros((image_size, image_size, 3), dtype=bool)
    bayer_masks[0::2, 0::2, 0] = True  # Red (channel 0)
    bayer_masks[0::2, 1::2, 1] = True  # Green (channel 1)
    bayer_masks[1::2, 0::2, 1] = True  # Green (channel 1)
    bayer_masks[1::2, 1::2, 2] = True  # Blue (channel 2)

    # Test parameter ranges
    test_cases = []

    # Position parameters
    x_centers = [1.0, 8.0, 15.0, 3.7, 12.3]
    y_centers = [1.0, 8.0, 15.0, 4.2, 11.8]
    sigmas_x = [0.5, 1.0, 1.5, 2.0, 3.0, 0.8, 2.7]
    sigmas_y = [0.5, 1.0, 1.5, 2.0, 3.0, 0.9, 2.4]

    # Background parameters (B, G, R)
    backgrounds = [
        [50, 60, 70],
        [100, 120, 80],
        [10, 15, 12],
        [200, 250, 180],
        [5, 8, 6],
        [300, 320, 280]
    ]

    # Amplitude parameters (B, G, R)
    amplitudes = [
        [500, 600, 400],
        [1000, 1200, 800],
        [100, 150, 90],
        [2000, 2500, 1800],
        [50, 80, 40],
        [5000, 6000, 4500]
    ]

    # Generate comprehensive test cases
    np.random.seed(42)
    n_random_tests = 50

    for i in range(n_random_tests):
        x_center = np.random.uniform(2, 14)
        y_center = np.random.uniform(2, 14)
        sx = np.random.uniform(0.5, 3.0)
        sy = np.random.uniform(0.5, 3.0)

        bg_B = np.random.uniform(10, 300)
        bg_G = np.random.uniform(10, 300)
        bg_R = np.random.uniform(10, 300)

        amp_B = np.random.uniform(100, 5000)
        amp_G = np.random.uniform(100, 5000)
        amp_R = np.random.uniform(100, 5000)

        params = np.array([x_center, y_center, sx, sy, bg_B, bg_G, bg_R, amp_B, amp_G, amp_R])
        test_cases.append(params)

    # Add systematic boundary cases
    boundary_cases = [
        # Extreme positions
        [0.1, 0.1, 1.5, 1.5, 100, 120, 80, 1000, 1200, 800],
        [15.9, 15.9, 1.5, 1.5, 100, 120, 80, 1000, 1200, 800],

        # Extreme sigmas
        [8.0, 8.0, 0.1, 0.1, 100, 120, 80, 1000, 1200, 800],
        [8.0, 8.0, 5.0, 5.0, 100, 120, 80, 1000, 1200, 800],

        # Extreme backgrounds
        [8.0, 8.0, 1.5, 1.5, 1, 1, 1, 1000, 1200, 800],
        [8.0, 8.0, 1.5, 1.5, 1000, 1000, 1000, 1000, 1200, 800],

        # Extreme amplitudes
        [8.0, 8.0, 1.5, 1.5, 100, 120, 80, 1, 1, 1],
        [8.0, 8.0, 1.5, 1.5, 100, 120, 80, 10000, 12000, 8000],

        # Asymmetric sigmas
        [8.0, 8.0, 0.5, 3.0, 100, 120, 80, 1000, 1200, 800],
        [8.0, 8.0, 3.0, 0.5, 100, 120, 80, 1000, 1200, 800],
    ]

    for boundary_case in boundary_cases:
        test_cases.append(np.array(boundary_case))

    print(f"Testing {len(test_cases)} parameter combinations...")

    max_diffs = []
    rmses = []
    failed_tests = []

    for i, params in enumerate(test_cases):
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(test_cases)}")

        # Prepare arrays
        data = np.zeros((image_size, image_size))
        background_bayer_matrix = np.zeros(image_size * image_size)
        bayer_matrix = np.zeros(image_size * image_size)
        gauss_2d_orig = np.zeros((image_size, image_size))
        gauss_2d_new = np.zeros((image_size, image_size))

        # Test original implementation
        result_orig = original_WLS_model_nobounds(
            params, data, bayer_masks, background_bayer_matrix,
            bayer_matrix, x, gauss_2d_orig
        ).copy()

        # Test new implementation
        result_new = gaussoptfuncs.WLS_model_nobounds(
            params, bayer_masks, x, gauss_2d_new
        ).copy()

        # Compare results
        max_diff = np.max(np.abs(result_new - result_orig))
        rmse = np.sqrt(np.mean((result_new - result_orig)**2))

        max_diffs.append(max_diff)
        rmses.append(rmse)

        # Check for significant differences
        if max_diff > 1e-10:
            failed_tests.append({
                'test_id': i,
                'params': params,
                'max_diff': max_diff,
                'rmse': rmse
            })

    # Results summary
    print(f"\n=== Test Results Summary ===")
    print(f"Total tests: {len(test_cases)}")
    print(f"Failed tests (max_diff > 1e-10): {len(failed_tests)}")
    print(f"Overall max difference: {np.max(max_diffs):.2e}")
    print(f"Overall max RMSE: {np.max(rmses):.2e}")
    print(f"Mean max difference: {np.mean(max_diffs):.2e}")
    print(f"Mean RMSE: {np.mean(rmses):.2e}")

    if len(failed_tests) == 0:
        print("✅ ALL TESTS PASSED - Implementations are numerically identical")
        return True
    else:
        print(f"❌ {len(failed_tests)} TESTS FAILED")
        print("\nFirst 5 failed tests:")
        for i, fail in enumerate(failed_tests[:5]):
            print(f"  Test {fail['test_id']}: max_diff={fail['max_diff']:.2e}, params={fail['params']}")
        return False

def test_edge_cases():
    """Test specific edge cases that might cause numerical issues."""
    print("\n=== Edge Cases Testing ===")

    image_size = 16
    x = np.arange(image_size)

    # Create Bayer masks
    bayer_masks = np.zeros((image_size, image_size, 3), dtype=bool)
    bayer_masks[0::2, 0::2, 0] = True  # Red
    bayer_masks[0::2, 1::2, 1] = True  # Green
    bayer_masks[1::2, 0::2, 1] = True  # Green
    bayer_masks[1::2, 1::2, 2] = True  # Blue

    edge_cases = [
        # Very small values
        [8.0, 8.0, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],

        # Very large values
        [8.0, 8.0, 100.0, 100.0, 10000, 10000, 10000, 100000, 100000, 100000],

        # Zero background
        [8.0, 8.0, 1.5, 1.5, 0.0, 0.0, 0.0, 1000, 1200, 800],

        # Zero amplitude
        [8.0, 8.0, 1.5, 1.5, 100, 120, 80, 0.0, 0.0, 0.0],

        # Equal values
        [8.0, 8.0, 1.0, 1.0, 100, 100, 100, 1000, 1000, 1000],

        # Extreme aspect ratio
        [8.0, 8.0, 0.1, 10.0, 100, 120, 80, 1000, 1200, 800],
        [8.0, 8.0, 10.0, 0.1, 100, 120, 80, 1000, 1200, 800],
    ]

    all_passed = True

    for i, edge_case in enumerate(edge_cases):
        params = np.array(edge_case)

        # Prepare arrays
        data = np.zeros((image_size, image_size))
        background_bayer_matrix = np.zeros(image_size * image_size)
        bayer_matrix = np.zeros(image_size * image_size)
        gauss_2d_orig = np.zeros((image_size, image_size))
        gauss_2d_new = np.zeros((image_size, image_size))

        try:
            # Test original implementation
            result_orig = original_WLS_model_nobounds(
                params, data, bayer_masks, background_bayer_matrix,
                bayer_matrix, x, gauss_2d_orig
            ).copy()

            # Test new implementation
            result_new = gaussoptfuncs.WLS_model_nobounds(
                params, bayer_masks, x, gauss_2d_new
            ).copy()

            # Compare results
            max_diff = np.max(np.abs(result_new - result_orig))
            rmse = np.sqrt(np.mean((result_new - result_orig)**2))

            if max_diff > 1e-10:
                print(f"❌ Edge case {i+1} FAILED: max_diff={max_diff:.2e}")
                print(f"   Params: {params}")
                all_passed = False
            else:
                print(f"✅ Edge case {i+1} PASSED: max_diff={max_diff:.2e}")

        except Exception as e:
            print(f"❌ Edge case {i+1} ERROR: {e}")
            print(f"   Params: {params}")
            all_passed = False

    return all_passed

def main():
    """Run comprehensive testing."""
    print("=" * 60)
    print("COMPREHENSIVE WLS_MODEL_NOBOUNDS VALIDATION")
    print("=" * 60)

    # Test parameter ranges
    range_passed = test_parameter_ranges()

    # Test edge cases
    edge_passed = test_edge_cases()

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 60)

    if range_passed and edge_passed:
        print("🎉 ALL TESTS PASSED - New implementation is numerically identical to original")
        print("✅ Safe to use optimized WLS_model_nobounds across all parameter ranges")
    else:
        print("❌ SOME TESTS FAILED - Implementation differences detected")
        if not range_passed:
            print("   - Parameter range tests failed")
        if not edge_passed:
            print("   - Edge case tests failed")

    return range_passed and edge_passed

if __name__ == "__main__":
    main()