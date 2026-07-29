#!/usr/bin/env python3
"""
Simple color fitting optimization analysis.
Focus on key bottlenecks and practical improvements.
"""

import numpy as np
import time
import sys
import os
from pathlib import Path
from numba import jit

# Add src directory to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"

import pyS3M.gaussoptfuncs as gaussoptfuncs

@jit(nopython=True, nogil=True)
def optimized_gaussian_model(gauss_2d, x, len_x, x0, y0, sx, sy):
    """Optimized Gaussian computation with pre-computed inverses."""
    sx_inv_sq = 1.0 / (sx * sx)
    sy_inv_sq = 1.0 / (sy * sy)

    for i in range(len_x):
        dx = x[i] - x0
        dx_sq_norm = dx * dx * sx_inv_sq
        for j in range(len_x):
            dy = x[j] - y0
            dy_sq_norm = dy * dy * sy_inv_sq
            gauss_2d[j, i] = np.exp(-0.5 * (dx_sq_norm + dy_sq_norm))

    return gauss_2d

@jit(nopython=True, nogil=True)
def optimized_WLS_model_fused(params, data, masks, background_bayer_matrix, bayer_matrix, x, gauss_2d):
    """
    Optimized version: Fuse Gaussian computation with Bayer pattern application.
    This eliminates intermediate arrays and reduces memory bandwidth.

    Parameter mapping matches original WLS_model_nobounds:
    - params[-6 + i] for backgrounds (i=0,1,2 -> params[4,5,6])
    - params[-3 + i] for amplitudes (i=0,1,2 -> params[7,8,9])
    - Mask channels: [0]=first color, [1]=second color, [2]=third color
    """
    len_x = len(x)

    # Pre-compute squared parameters matching original indexing
    # params[-6+i] for backgrounds: i=0->params[4], i=1->params[5], i=2->params[6]
    # params[-3+i] for amplitudes: i=0->params[7], i=1->params[8], i=2->params[9]
    bg_chan0 = params[4] * params[4]    # Background for mask channel 0
    bg_chan1 = params[5] * params[5]    # Background for mask channel 1
    bg_chan2 = params[6] * params[6]    # Background for mask channel 2
    amp_chan0 = params[7] * params[7]   # Amplitude for mask channel 0
    amp_chan1 = params[8] * params[8]   # Amplitude for mask channel 1
    amp_chan2 = params[9] * params[9]   # Amplitude for mask channel 2

    # Pre-compute Gaussian parameters
    x0, y0, sx, sy = params[0], params[1], params[2], params[3]

    # Use same normalization as gaussian_unscaled_model
    norm_x = 0.3989422804014327 / sx
    norm_y = 0.3989422804014327 / sy

    # Fused computation: Gaussian + Bayer pattern in single pass
    for i in range(len_x):
        # Compute gaussian components once per row
        gauss_x = norm_x * np.exp(-0.5 * ((x[i] - x0) / sx) ** 2)

        for j in range(len_x):
            gauss_y = norm_y * np.exp(-0.5 * ((x[j] - y0) / sy) ** 2)
            gauss_val = gauss_x * gauss_y

            # Apply Bayer pattern directly matching original channel order
            if masks[j, i, 0]:  # Channel 0
                gauss_2d[j, i] = amp_chan0 * gauss_val + bg_chan0
            elif masks[j, i, 1]:  # Channel 1
                gauss_2d[j, i] = amp_chan1 * gauss_val + bg_chan1
            else:  # Channel 2 (masks[j, i, 2])
                gauss_2d[j, i] = amp_chan2 * gauss_val + bg_chan2

    return gauss_2d

@jit(nopython=True, nogil=True)
def optimized_WLS_model_separated(params, data, masks, background_bayer_matrix, bayer_matrix, x, gauss_2d):
    """
    Alternative optimization: Separate Gaussian computation with optimized masking.
    Uses the same parameter indexing and Gaussian normalization as original.
    """
    len_x = len(x)

    # Pre-compute squared parameters matching original indexing
    bg_vals = np.array([params[4] * params[4], params[5] * params[5], params[6] * params[6]])
    amp_vals = np.array([params[7] * params[7], params[8] * params[8], params[9] * params[9]])

    # Compute Gaussian using the same method as gaussian_unscaled_model
    x0, y0, sx, sy = params[0], params[1], params[2], params[3]
    norm_x = 0.3989422804014327 / sx
    norm_y = 0.3989422804014327 / sy

    for i in range(len_x):
        gauss_x = norm_x * np.exp(-0.5 * ((x[i] - x0) / sx) ** 2)
        for j in range(len_x):
            gauss_y = norm_y * np.exp(-0.5 * ((x[j] - y0) / sy) ** 2)
            gauss_2d[j, i] = gauss_x * gauss_y

    # Apply Bayer pattern with lookup
    for i in range(len_x):
        for j in range(len_x):
            if masks[j, i, 0]:      # Channel 0
                channel = 0
            elif masks[j, i, 1]:    # Channel 1
                channel = 1
            else:                   # Channel 2
                channel = 2

            gauss_2d[j, i] = amp_vals[channel] * gauss_2d[j, i] + bg_vals[channel]

    return gauss_2d

def analyze_current_performance():
    """Analyze current implementation performance."""
    print("=== Analyzing Current Implementation ===")

    image_size = 16
    x = np.arange(image_size)

    # Create Bayer masks
    bayer_masks = np.zeros((image_size, image_size, 3), dtype=bool)
    bayer_masks[0::2, 0::2, 0] = True  # Red
    bayer_masks[0::2, 1::2, 1] = True  # Green
    bayer_masks[1::2, 0::2, 1] = True  # Green
    bayer_masks[1::2, 1::2, 2] = True  # Blue

    # Test parameters
    params = np.array([8.0, 8.0, 1.5, 1.5, 100, 120, 80, 1000, 1200, 800])
    data = np.zeros((image_size, image_size))
    gauss_2d = np.zeros((image_size, image_size))

    n_runs = 1000

    # Profile current implementation
    times = []
    for i in range(n_runs):
        # Reset arrays
        gauss_2d.fill(0.0)

        start = time.perf_counter()
        result = gaussoptfuncs.WLS_model_nobounds(
            params, bayer_masks, x, gauss_2d
        )
        times.append(time.perf_counter() - start)

    current_time = np.mean(times)
    print(f"Current implementation:")
    print(f"  Mean time: {current_time*1000:.4f} ms")
    print(f"  Rate: {1/current_time:.0f} calls/second")
    print(f"  Time for 100k fits: {100000*current_time:.1f} seconds")

    return current_time, bayer_masks, x, params, data, gauss_2d

def benchmark_optimizations():
    """Benchmark optimization approaches."""
    print("\n=== Benchmarking Optimizations ===")

    current_time, bayer_masks, x, params, data, gauss_2d = analyze_current_performance()

    optimizations = [
        ("Fused Computation", optimized_WLS_model_fused),
        ("Separated + Lookup", optimized_WLS_model_separated),
    ]

    n_runs = 1000
    results = {}

    for name, func in optimizations:
        print(f"\nTesting {name}:")

        times = []
        for i in range(n_runs):
            # Reset arrays
            gauss_2d.fill(0.0)

            # Create temporary arrays for optimization functions that still need them
            bg_matrix = np.zeros(len(x) * len(x))
            bayer_matrix = np.zeros(len(x) * len(x))

            start = time.perf_counter()
            result = func(params, data, bayer_masks, bg_matrix, bayer_matrix, x, gauss_2d)
            times.append(time.perf_counter() - start)

        mean_time = np.mean(times)
        speedup = current_time / mean_time

        print(f"  Mean time: {mean_time*1000:.4f} ms")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Rate: {1/mean_time:.0f} calls/second")
        print(f"  Time for 100k fits: {100000*mean_time:.1f} seconds")

        results[name] = {
            'time': mean_time,
            'speedup': speedup,
            'rate': 1/mean_time
        }

    return results

def validate_optimizations():
    """Validate that optimizations produce correct results."""
    print("\n=== Validating Optimization Correctness ===")

    image_size = 16
    x = np.arange(image_size)

    # Create test data
    bayer_masks = np.zeros((image_size, image_size, 3), dtype=bool)
    bayer_masks[0::2, 0::2, 0] = True  # Red
    bayer_masks[0::2, 1::2, 1] = True  # Green
    bayer_masks[1::2, 0::2, 1] = True  # Green
    bayer_masks[1::2, 1::2, 2] = True  # Blue

    params = np.array([8.0, 8.0, 1.5, 1.5, 100, 120, 80, 1000, 1200, 800])
    data = np.zeros((image_size, image_size))
    gauss_2d = np.zeros((image_size, image_size))

    # Get reference result
    reference = gaussoptfuncs.WLS_model_nobounds(
        params, bayer_masks, x, gauss_2d
    ).copy()

    # Test optimized versions
    optimizations = [
        ("Fused", optimized_WLS_model_fused),
        ("Separated", optimized_WLS_model_separated),
    ]

    for name, func in optimizations:
        # Reset arrays
        gauss_2d.fill(0.0)

        # Create temporary arrays for the optimization functions that still need them
        background_bayer_matrix = np.zeros(image_size * image_size)
        bayer_matrix = np.zeros(image_size * image_size)

        result = func(params, data, bayer_masks, background_bayer_matrix, bayer_matrix, x, gauss_2d)

        max_diff = np.max(np.abs(result - reference))
        rmse = np.sqrt(np.mean((result - reference)**2))

        print(f"{name}: Max diff = {max_diff:.2e}, RMSE = {rmse:.2e}")

        if max_diff < 1e-10:
            print(f"  ✓ {name} is numerically identical")
        elif max_diff < 1e-6:
            print(f"  ~ {name} is numerically close")
        else:
            print(f"  ✗ {name} has significant differences!")

def analyze_bottlenecks():
    """Identify specific bottlenecks in the current implementation."""
    print("\n=== Bottleneck Analysis ===")

    print("Current WLS_model_nobounds bottlenecks:")
    print("1. **Mask Processing**: Loop over color channels with ravel/reshape operations")
    print("2. **Array Operations**: Multiple temporary arrays (bayer_matrix, background_bayer_matrix)")
    print("3. **Memory Bandwidth**: Separate Gaussian computation + masking requires multiple passes")
    print("4. **Parameter Squaring**: Repeated squaring operations inside loops")

    print("\nOptimization strategies:")
    print("1. **Fuse Operations**: Combine Gaussian computation with Bayer masking")
    print("2. **Pre-compute**: Calculate squared parameters once outside loops")
    print("3. **Eliminate Temporaries**: Apply masking directly to output")
    print("4. **Loop Optimization**: Better memory access patterns")

def main():
    """Main optimization analysis."""
    print("="*60)
    print("COLOR FITTING OPTIMIZATION ANALYSIS")
    print("="*60)

    # Analyze bottlenecks
    analyze_bottlenecks()

    # Validate correctness
    validate_optimizations()

    # Benchmark optimizations
    results = benchmark_optimizations()

    # Summary
    print("\n" + "="*60)
    print("OPTIMIZATION SUMMARY")
    print("="*60)

    best_speedup = 0
    best_method = ""

    for method, result in results.items():
        speedup = result['speedup']
        if speedup > best_speedup:
            best_speedup = speedup
            best_method = method

        # Calculate time savings for 100k puncta
        original_time_100k = 16.3  # seconds from parallel benchmark
        optimized_time_100k = original_time_100k / speedup
        time_saved = original_time_100k - optimized_time_100k

        print(f"{method}:")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  100k puncta: {optimized_time_100k:.1f}s (saves {time_saved:.1f}s)")

    print(f"\nBest optimization: {best_method}")
    print(f"  Speedup: {best_speedup:.2f}x")

    original_100k = 16.3
    optimized_100k = original_100k / best_speedup
    total_savings = original_100k - optimized_100k

    print(f"\nPractical Impact:")
    print(f"  Current color fitting (100k puncta): {original_100k:.1f} seconds")
    print(f"  Optimized color fitting (100k puncta): {optimized_100k:.1f} seconds")
    print(f"  Time savings: {total_savings:.1f} seconds ({100*total_savings/original_100k:.1f}% faster)")
    print(f"  New ratio vs position-only: {optimized_100k/8.0:.1f}x (vs current {original_100k/8.0:.1f}x)")

if __name__ == "__main__":
    main()