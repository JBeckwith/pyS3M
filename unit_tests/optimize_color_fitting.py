#!/usr/bin/env python3
"""
Optimization analysis and improved implementations for WLS_model_nobounds.

This script identifies bottlenecks in color fitting and proposes optimized versions
while preserving the same color information.
"""

import numpy as np
import time
import sys
import os
from pathlib import Path
from numba import jit
from typing import Tuple

# Add src directory to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

import gaussoptfuncs

class ColorFittingOptimizer:
    """Analyze and optimize color fitting performance."""

    def __init__(self, image_size: int = 16):
        self.image_size = image_size
        self.x = np.arange(image_size)

        # Pre-create Bayer masks
        self.bayer_masks = np.zeros((image_size, image_size, 3), dtype=bool)
        self.bayer_masks[0::2, 0::2, 0] = True  # Red
        self.bayer_masks[0::2, 1::2, 1] = True  # Green
        self.bayer_masks[1::2, 0::2, 1] = True  # Green
        self.bayer_masks[1::2, 1::2, 2] = True  # Blue

        # Pre-allocate reusable arrays
        self.background_bayer_matrix = np.zeros(image_size * image_size)
        self.bayer_matrix = np.zeros(image_size * image_size)
        self.gauss_2d = np.zeros((image_size, image_size))

        print(f"Color Fitting Optimizer initialized for {image_size}×{image_size} images")

    def analyze_bottlenecks(self):
        """Analyze performance bottlenecks in current implementation."""
        print("\n=== Analyzing Current Implementation Bottlenecks ===")

        # Test parameters
        params = np.array([8.0, 8.0, 1.5, 1.5, 100, 120, 80, 1000, 1200, 800])  # x,y,sx,sy,bg_B,bg_G,bg_R,amp_B,amp_G,amp_R
        data = np.zeros((self.image_size, self.image_size))
        n_runs = 1000

        # Profile current implementation
        times = []
        for i in range(n_runs):
            start = time.perf_counter()
            result = gaussoptfuncs.WLS_model_nobounds(
                params, self.bayer_masks, self.x, self.gauss_2d
            )
            times.append(time.perf_counter() - start)

        print(f"Current implementation:")
        print(f"  Mean time: {np.mean(times)*1000:.4f} ms")
        print(f"  Std dev: {np.std(times)*1000:.4f} ms")
        print(f"  Rate: {1/np.mean(times):.0f} calls/second")

        return np.mean(times)

    @staticmethod
    @jit(nopython=True, nogil=True)
    def optimized_gaussian_unscaled_model(gauss_2d, x, len_x, x0, y0, sx, sy):
        """Optimized version of gaussian_unscaled_model with loop unrolling."""
        sx_inv_sq = 1.0 / (sx * sx)
        sy_inv_sq = 1.0 / (sy * sy)

        for i in range(len_x):
            dx = x[i] - x0
            dx_sq = dx * dx * sx_inv_sq
            for j in range(len_x):
                dy = x[j] - y0  # x array used for both dimensions
                dy_sq = dy * dy * sy_inv_sq
                gauss_2d[j, i] = np.exp(-0.5 * (dx_sq + dy_sq))

        return gauss_2d

    @staticmethod
    @jit(nopython=True, nogil=True)
    def optimized_WLS_model_nobounds_v1(
        params, data, masks, background_bayer_matrix, bayer_matrix, x, gauss_2d
    ):
        """
        Optimized version 1: Pre-compute masks and minimize array operations.
        """
        len_x = len(x)

        # Reset matrices
        background_bayer_matrix.fill(0.0)
        bayer_matrix.fill(0.0)

        # Unroll the mask loop for better performance (assuming RGB = 3 channels)
        # Blue channel (i=0)
        amp_B_sq = params[7] * params[7]  # params[-3 + 0]
        bg_B_sq = params[4] * params[4]   # params[-6 + 0]

        # Green channel (i=1)
        amp_G_sq = params[8] * params[8]  # params[-3 + 1]
        bg_G_sq = params[5] * params[5]   # params[-6 + 1]

        # Red channel (i=2)
        amp_R_sq = params[9] * params[9]  # params[-3 + 2]
        bg_R_sq = params[6] * params[6]   # params[-6 + 2]

        # Apply masks directly without ravel/reshape operations
        for i in range(len_x):
            for j in range(len_x):
                idx = j * len_x + i

                if masks[j, i, 0]:  # Blue pixel
                    bayer_matrix[idx] = amp_B_sq
                    background_bayer_matrix[idx] = bg_B_sq
                elif masks[j, i, 1]:  # Green pixel
                    bayer_matrix[idx] = amp_G_sq
                    background_bayer_matrix[idx] = bg_G_sq
                else:  # Red pixel (masks[j, i, 2])
                    bayer_matrix[idx] = amp_R_sq
                    background_bayer_matrix[idx] = bg_R_sq

        # Reshape once
        bayer_matrix_2d = bayer_matrix.reshape(len_x, len_x)
        background_bayer_matrix_2d = background_bayer_matrix.reshape(len_x, len_x)

        # Compute Gaussian
        gauss_2d = ColorFittingOptimizer.optimized_gaussian_unscaled_model(
            gauss_2d, x, len_x, params[0], params[1], params[2], params[3]
        )

        # Final computation
        for i in range(len_x):
            for j in range(len_x):
                gauss_2d[j, i] = bayer_matrix_2d[j, i] * gauss_2d[j, i] + background_bayer_matrix_2d[j, i]

        return gauss_2d

    @staticmethod
    @jit(nopython=True, nogil=True)
    def optimized_WLS_model_nobounds_v2(
        params, data, masks, background_bayer_matrix, bayer_matrix, x, gauss_2d
    ):
        """
        Optimized version 2: Fused loops to minimize memory accesses.
        """
        len_x = len(x)

        # Pre-compute squared parameters
        amp_B_sq = params[7] * params[7]
        bg_B_sq = params[4] * params[4]
        amp_G_sq = params[8] * params[8]
        bg_G_sq = params[5] * params[5]
        amp_R_sq = params[9] * params[9]
        bg_R_sq = params[6] * params[6]

        # Pre-compute Gaussian parameters
        x0, y0, sx, sy = params[0], params[1], params[2], params[3]
        sx_inv_sq = 1.0 / (sx * sx)
        sy_inv_sq = 1.0 / (sy * sy)

        # Fused loop: compute Gaussian and apply Bayer pattern in one pass
        for i in range(len_x):
            dx = x[i] - x0
            dx_sq = dx * dx * sx_inv_sq

            for j in range(len_x):
                dy = x[j] - y0
                dy_sq = dy * dy * sy_inv_sq

                # Compute Gaussian value
                gauss_val = np.exp(-0.5 * (dx_sq + dy_sq))

                # Apply Bayer pattern directly
                if masks[j, i, 0]:  # Blue pixel
                    gauss_2d[j, i] = amp_B_sq * gauss_val + bg_B_sq
                elif masks[j, i, 1]:  # Green pixel
                    gauss_2d[j, i] = amp_G_sq * gauss_val + bg_G_sq
                else:  # Red pixel
                    gauss_2d[j, i] = amp_R_sq * gauss_val + bg_R_sq

        return gauss_2d

    @staticmethod
    @jit(nopython=True, nogil=True)
    def optimized_WLS_model_nobounds_v3(
        params, data, masks, background_bayer_matrix, bayer_matrix, x, gauss_2d
    ):
        """
        Optimized version 3: Vectorized Bayer pattern with lookup tables.
        """
        len_x = len(x)

        # Create lookup tables for amplitudes and backgrounds
        amp_lookup = np.array([params[7] * params[7],  # Blue
                              params[8] * params[8],   # Green
                              params[9] * params[9]])  # Red

        bg_lookup = np.array([params[4] * params[4],   # Blue
                             params[5] * params[5],    # Green
                             params[6] * params[6]])   # Red

        # Pre-compute Gaussian parameters
        x0, y0, sx, sy = params[0], params[1], params[2], params[3]
        sx_inv_sq = 1.0 / (sx * sx)
        sy_inv_sq = 1.0 / (sy * sy)

        # Compute Gaussian and apply Bayer pattern
        for i in range(len_x):
            dx = x[i] - x0
            dx_sq = dx * dx * sx_inv_sq

            for j in range(len_x):
                dy = x[j] - y0
                dy_sq = dy * dy * sy_inv_sq
                gauss_val = np.exp(-0.5 * (dx_sq + dy_sq))

                # Find channel index
                if masks[j, i, 0]:      # Blue
                    channel = 0
                elif masks[j, i, 1]:    # Green
                    channel = 1
                else:                   # Red
                    channel = 2

                gauss_2d[j, i] = amp_lookup[channel] * gauss_val + bg_lookup[channel]

        return gauss_2d

    def benchmark_optimizations(self):
        """Benchmark all optimization variants."""
        print("\n=== Benchmarking Optimizations ===")

        params = np.array([8.0, 8.0, 1.5, 1.5, 100, 120, 80, 1000, 1200, 800])
        data = np.zeros((self.image_size, self.image_size))
        n_runs = 1000

        # Test original implementation
        original_time = self.analyze_bottlenecks()

        # Test optimized versions
        optimizations = [
            ("Optimized V1 (Pre-compute + Unroll)", self.optimized_WLS_model_nobounds_v1),
            ("Optimized V2 (Fused Loops)", self.optimized_WLS_model_nobounds_v2),
            ("Optimized V3 (Lookup Tables)", self.optimized_WLS_model_nobounds_v3),
        ]

        results = {}

        for name, func in optimizations:
            print(f"\nTesting {name}:")

            times = []
            for i in range(n_runs):
                # Reset arrays
                self.background_bayer_matrix.fill(0.0)
                self.bayer_matrix.fill(0.0)
                self.gauss_2d.fill(0.0)

                start = time.perf_counter()
                result = func(
                    params, data, self.bayer_masks, self.background_bayer_matrix,
                    self.bayer_matrix, self.x, self.gauss_2d
                )
                times.append(time.perf_counter() - start)

            mean_time = np.mean(times)
            speedup = original_time / mean_time

            print(f"  Mean time: {mean_time*1000:.4f} ms")
            print(f"  Speedup: {speedup:.2f}x")
            print(f"  Rate: {1/mean_time:.0f} calls/second")

            results[name] = {
                'time': mean_time,
                'speedup': speedup,
                'rate': 1/mean_time
            }

        return results

    def validate_correctness(self):
        """Validate that optimized versions produce identical results."""
        print("\n=== Validating Correctness ===")

        params = np.array([8.0, 8.0, 1.5, 1.5, 100, 120, 80, 1000, 1200, 800])
        data = np.zeros((self.image_size, self.image_size))

        # Get reference result
        self.background_bayer_matrix.fill(0.0)
        self.bayer_matrix.fill(0.0)
        self.gauss_2d.fill(0.0)

        reference = gaussoptfuncs.WLS_model_nobounds(
            params, self.bayer_masks, self.x, self.gauss_2d
        ).copy()

        # Test each optimization
        optimizations = [
            ("V1", self.optimized_WLS_model_nobounds_v1),
            ("V2", self.optimized_WLS_model_nobounds_v2),
            ("V3", self.optimized_WLS_model_nobounds_v3),
        ]

        for name, func in optimizations:
            self.background_bayer_matrix.fill(0.0)
            self.bayer_matrix.fill(0.0)
            self.gauss_2d.fill(0.0)

            result = func(
                params, data, self.bayer_masks, self.background_bayer_matrix,
                self.bayer_matrix, self.x, self.gauss_2d
            )

            max_diff = np.max(np.abs(result - reference))
            rmse = np.sqrt(np.mean((result - reference)**2))

            print(f"{name}: Max diff = {max_diff:.2e}, RMSE = {rmse:.2e}")

            if max_diff < 1e-10:
                print(f"  ✓ {name} is numerically identical")
            elif max_diff < 1e-6:
                print(f"  ~ {name} is numerically close")
            else:
                print(f"  ✗ {name} has significant differences!")

def main():
    """Main function to run optimization analysis."""
    print("="*60)
    print("COLOR FITTING OPTIMIZATION ANALYSIS")
    print("="*60)

    optimizer = ColorFittingOptimizer(image_size=16)

    # Validate correctness first
    optimizer.validate_correctness()

    # Benchmark optimizations
    results = optimizer.benchmark_optimizations()

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

        print(f"{method}:")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  For 100k fits: {100000/result['rate']:.1f} seconds (vs {100000*0.45:.1f}s original)")

    print(f"\nBest optimization: {best_method} ({best_speedup:.2f}x speedup)")
    print(f"Potential time savings for 100k puncta: {100000*(0.45/1000 - 0.45/1000/best_speedup):.1f} seconds")

    return results

if __name__ == "__main__":
    main()