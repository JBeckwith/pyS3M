#!/usr/bin/env python3
"""
Realistic fitting benchmark using the actual gaussoptfuncs functions.
This should provide more accurate timing comparisons.
"""

import numpy as np
import time
import sys
import os
from pathlib import Path
from typing import Tuple, List, Dict
import gc

# Add src directory to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Import the actual fitting functions
import gaussoptfuncs
from scipy.optimize import leastsq

class RealisticFittingBenchmark:
    """Benchmark using actual gaussoptfuncs fitting functions."""

    def __init__(self, image_size: int = 16, target_puncta: int = 10000):
        self.image_size = image_size
        self.target_puncta = target_puncta

        # Set up coordinate grids
        self.x = np.arange(image_size)

        # Create Bayer masks for color fitting (3 colors: R, G, B)
        self.bayer_masks = np.zeros((image_size, image_size, 3), dtype=bool)
        # RGGB Bayer pattern
        self.bayer_masks[0::2, 0::2, 0] = True  # Red
        self.bayer_masks[0::2, 1::2, 1] = True  # Green
        self.bayer_masks[1::2, 0::2, 1] = True  # Green
        self.bayer_masks[1::2, 1::2, 2] = True  # Blue

        print(f"Realistic Benchmark Configuration:")
        print(f"  - Image size: {image_size}x{image_size} pixels")
        print(f"  - Target puncta: {target_puncta:,}")
        print(f"  - Using actual gaussoptfuncs")

    def generate_position_test_data(self, n_images: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Generate test data for position-only fitting."""
        images = []
        weights = []

        np.random.seed(42)

        for i in range(n_images):
            # Generate realistic parameters
            margin = 3
            x_center = np.random.uniform(margin, self.image_size - margin)
            y_center = np.random.uniform(margin, self.image_size - margin)
            sx = sy = np.random.uniform(1.0, 2.0)  # PSF width
            background = np.random.uniform(50, 200)
            amplitude = np.random.uniform(500, 3000)

            # Generate clean Gaussian image
            image = np.zeros((self.image_size, self.image_size))
            gauss_2d = np.zeros((self.image_size, self.image_size))

            # Use the actual gaussian generation function
            params = np.array([x_center, y_center, sx, sy, background, amplitude])
            image = gaussoptfuncs.WLS_nocolour_model_nobounds(
                params, image, self.x, gauss_2d
            )

            # Add Poisson noise
            image = np.random.poisson(np.maximum(image, 0.1)).astype(float)

            # Generate weights
            weights_img = 1.0 / np.maximum(image, 1.0)

            images.append(image)
            weights.append(weights_img)

        return images, weights

    def generate_color_test_data(self, n_images: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Generate test data for color fitting."""
        images = []
        weights = []

        np.random.seed(42)  # Same seed for fair comparison

        for i in range(n_images):
            # Same spatial parameters
            margin = 3
            x_center = np.random.uniform(margin, self.image_size - margin)
            y_center = np.random.uniform(margin, self.image_size - margin)
            sx = sy = np.random.uniform(1.0, 2.0)

            # Different backgrounds and amplitudes for R, G, B
            bg_R, bg_G, bg_B = 100, 120, 80
            amp_R, amp_G, amp_B = 1000, 1200, 800

            # Parameters for color fitting: [x, y, sx, sy, bg_B, bg_G, bg_R, amp_B, amp_G, amp_R]
            params = np.array([x_center, y_center, sx, sy, bg_B, bg_G, bg_R, amp_B, amp_G, amp_R])

            # Generate clean Bayer image using actual color function
            image = np.zeros((self.image_size, self.image_size))
            gauss_2d = np.zeros((self.image_size, self.image_size))

            image = gaussoptfuncs.WLS_model_nobounds(
                params, self.bayer_masks, self.x, gauss_2d
            )

            # Add Poisson noise
            image = np.random.poisson(np.maximum(image, 0.1)).astype(float)

            # Generate weights
            weights_img = 1.0 / np.maximum(image, 1.0)

            images.append(image)
            weights.append(weights_img)

        return images, weights

    def benchmark_position_fitting(self, images: List[np.ndarray], weights: List[np.ndarray]) -> Dict:
        """Benchmark position-only fitting using actual gaussoptfuncs."""
        print("\n=== Position-Only Fitting (WLS_nocolour_model_nobounds) ===")

        fit_times = []
        successful_fits = 0
        n_images = len(images)

        for i, (image, weight) in enumerate(zip(images, weights)):
            if i % 1000 == 0 and i > 0:
                print(f"  Progress: {i:,}/{n_images:,}")

            # Initial guess for position fitting [x, y, sx, sy, bg, amp]
            center = self.image_size // 2
            max_val = np.max(image)
            min_val = np.min(image)

            initial_guess = np.array([
                center, center,           # x, y center
                1.5, 1.5,                # sigma x, y
                min_val,                 # background
                max_val - min_val        # amplitude
            ])

            start_time = time.perf_counter()

            try:
                # Use the actual fitting function from ImageAnalysisFunctions
                size = int(image.shape[0])
                ravelsize = int(np.prod(image.shape))

                pfit, pcov, infodict, errmsg, success = leastsq(
                    gaussoptfuncs.WLS_chi_nocolour_nobounds,
                    x0=initial_guess,
                    args=(image, weight, size, ravelsize),
                    full_output=True,
                    ftol=1e-6,
                    xtol=1e-6,
                )

                fit_time = time.perf_counter() - start_time
                fit_times.append(fit_time)

                if success in [1, 2, 3, 4]:
                    successful_fits += 1

            except Exception as e:
                fit_time = time.perf_counter() - start_time
                fit_times.append(fit_time)

        results = {
            'total_fits': n_images,
            'successful_fits': successful_fits,
            'success_rate': successful_fits / n_images,
            'total_time': sum(fit_times),
            'mean_time_per_fit': np.mean(fit_times),
            'fits_per_second': n_images / sum(fit_times)
        }

        print(f"Position-Only Results:")
        print(f"  Total fits: {results['total_fits']:,}")
        print(f"  Successful: {results['successful_fits']:,} ({results['success_rate']:.1%})")
        print(f"  Total time: {results['total_time']:.2f} seconds")
        print(f"  Mean time per fit: {results['mean_time_per_fit']*1000:.3f} ms")
        print(f"  Rate: {results['fits_per_second']:.0f} fits/second")

        return results

    def benchmark_color_fitting(self, images: List[np.ndarray], weights: List[np.ndarray]) -> Dict:
        """Benchmark color fitting using actual gaussoptfuncs."""
        print("\n=== Color Fitting (WLS_model_nobounds) ===")

        fit_times = []
        successful_fits = 0
        n_images = len(images)

        # Pre-allocate arrays for color fitting
        background_bayer_matrix = np.zeros(self.image_size * self.image_size)
        bayer_matrix = np.zeros(self.image_size * self.image_size)
        gauss_2d = np.zeros((self.image_size, self.image_size))

        for i, (image, weight) in enumerate(zip(images, weights)):
            if i % 1000 == 0 and i > 0:
                print(f"  Progress: {i:,}/{n_images:,}")

            # Initial guess for color fitting [x, y, sx, sy, bg_B, bg_G, bg_R, amp_B, amp_G, amp_R]
            center = self.image_size // 2
            max_val = np.max(image)
            min_val = np.min(image)

            initial_guess = np.array([
                center, center,           # x, y center
                1.5, 1.5,                # sigma x, y
                min_val, min_val, min_val,  # background B, G, R
                max_val/3, max_val/3, max_val/3  # amplitude B, G, R
            ])

            start_time = time.perf_counter()

            try:
                # Use the actual color fitting function
                pfit, pcov, infodict, errmsg, success = leastsq(
                    gaussoptfuncs.WLS_chi_nobounds,
                    x0=initial_guess,
                    args=(image, weight, self.bayer_masks, background_bayer_matrix,
                          bayer_matrix, self.x, gauss_2d),
                    full_output=True,
                    ftol=1e-6,
                    xtol=1e-6,
                )

                fit_time = time.perf_counter() - start_time
                fit_times.append(fit_time)

                if success in [1, 2, 3, 4]:
                    successful_fits += 1

            except Exception as e:
                fit_time = time.perf_counter() - start_time
                fit_times.append(fit_time)

        results = {
            'total_fits': n_images,
            'successful_fits': successful_fits,
            'success_rate': successful_fits / n_images,
            'total_time': sum(fit_times),
            'mean_time_per_fit': np.mean(fit_times),
            'fits_per_second': n_images / sum(fit_times)
        }

        print(f"Color Fitting Results:")
        print(f"  Total fits: {results['total_fits']:,}")
        print(f"  Successful: {results['successful_fits']:,} ({results['success_rate']:.1%})")
        print(f"  Total time: {results['total_time']:.2f} seconds")
        print(f"  Mean time per fit: {results['mean_time_per_fit']*1000:.3f} ms")
        print(f"  Rate: {results['fits_per_second']:.0f} fits/second")

        return results

    def run_benchmark(self):
        """Run the complete realistic benchmark."""
        print("="*60)
        print("REALISTIC FITTING SPEED BENCHMARK")
        print("Using actual gaussoptfuncs functions")
        print("="*60)

        # Use smaller batches for testing
        batch_size = min(2000, self.target_puncta)
        n_batches = (self.target_puncta + batch_size - 1) // batch_size

        print(f"Running {n_batches} batches of {batch_size:,} images each")

        position_totals = {'total_time': 0, 'total_fits': 0, 'successful_fits': 0}
        color_totals = {'total_time': 0, 'total_fits': 0, 'successful_fits': 0}

        for batch in range(n_batches):
            current_batch_size = min(batch_size, self.target_puncta - batch * batch_size)
            print(f"\n--- Batch {batch + 1}/{n_batches} ({current_batch_size:,} images) ---")

            # Position fitting
            print("Generating position test data...")
            pos_images, pos_weights = self.generate_position_test_data(current_batch_size)
            pos_results = self.benchmark_position_fitting(pos_images, pos_weights)

            del pos_images, pos_weights
            gc.collect()

            # Color fitting
            print("\nGenerating color test data...")
            color_images, color_weights = self.generate_color_test_data(current_batch_size)
            color_results = self.benchmark_color_fitting(color_images, color_weights)

            del color_images, color_weights
            gc.collect()

            # Accumulate results
            position_totals['total_time'] += pos_results['total_time']
            position_totals['total_fits'] += pos_results['total_fits']
            position_totals['successful_fits'] += pos_results['successful_fits']

            color_totals['total_time'] += color_results['total_time']
            color_totals['total_fits'] += color_results['total_fits']
            color_totals['successful_fits'] += color_results['successful_fits']

        # Final results
        print("\n" + "="*60)
        print("FINAL REALISTIC BENCHMARK RESULTS")
        print("="*60)

        pos_rate = position_totals['total_fits'] / position_totals['total_time']
        color_rate = color_totals['total_fits'] / color_totals['total_time']
        speedup_ratio = pos_rate / color_rate

        print(f"\nPosition-Only Fitting (WLS_nocolour_model_nobounds):")
        print(f"  Total puncta: {position_totals['total_fits']:,}")
        print(f"  Success rate: {100*position_totals['successful_fits']/position_totals['total_fits']:.1f}%")
        print(f"  Total time: {position_totals['total_time']:.2f} seconds")
        print(f"  Rate: {pos_rate:.0f} fits/second")
        print(f"  Time for 100k puncta: {100000 / pos_rate:.1f} seconds ({100000 / pos_rate / 60:.1f} minutes)")

        print(f"\nColor Fitting (WLS_model_nobounds):")
        print(f"  Total puncta: {color_totals['total_fits']:,}")
        print(f"  Success rate: {100*color_totals['successful_fits']/color_totals['total_fits']:.1f}%")
        print(f"  Total time: {color_totals['total_time']:.2f} seconds")
        print(f"  Rate: {color_rate:.0f} fits/second")
        print(f"  Time for 100k puncta: {100000 / color_rate:.1f} seconds ({100000 / color_rate / 60:.1f} minutes)")

        print(f"\nPerformance Comparison:")
        if speedup_ratio > 1:
            print(f"  Position-only is {speedup_ratio:.1f}x faster than color fitting")
        else:
            print(f"  Color fitting is {1/speedup_ratio:.1f}x faster than position-only")

        return {
            'position_rate': pos_rate,
            'color_rate': color_rate,
            'speedup_ratio': speedup_ratio
        }


def main():
    """Run the realistic benchmark."""
    benchmark = RealisticFittingBenchmark(
        image_size=16,
        target_puncta=5000  # Smaller for testing
    )

    results = benchmark.run_benchmark()

    print(f"\nMachine: {os.uname().nodename}")
    print("Benchmark complete!")

    return results


if __name__ == "__main__":
    results = main()