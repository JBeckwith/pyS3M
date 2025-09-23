#!/usr/bin/env python3
"""
Parallel fitting benchmark to test real-world performance with multiprocessing.
This compares single-threaded vs parallel fitting speeds using the actual
pyBayerSMLM parallel processing infrastructure.
"""

import numpy as np
import time
import sys
import os
import multiprocessing
from pathlib import Path
from typing import List, Tuple, Dict
import gc

# Add src directory to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Import required modules
import ImageAnalysisFunctions
from ImageAnalysisFunctions import FittingStrategy, FittingConstants
import gaussoptfuncs

class ParallelFittingBenchmark:
    """Benchmark parallel vs single-threaded fitting performance."""

    def __init__(self, image_size: int = 16):
        self.image_size = image_size
        self.analyzer = ImageAnalysisFunctions.Image_Analysis_Functions()

        # Check system capabilities
        self.n_cores = multiprocessing.cpu_count()
        self.n_workers = min(
            FittingConstants.MAX_WORKERS,
            max(1, int(FittingConstants.WORKER_RATIO * self.n_cores))
        )

        print(f"Parallel Benchmark Configuration:")
        print(f"  - CPU cores: {self.n_cores}")
        print(f"  - Workers: {self.n_workers} ({FittingConstants.WORKER_RATIO:.1%} of cores)")
        print(f"  - Tasks per worker: {FittingConstants.TASKS_PER_WORKER}")
        print(f"  - Image size: {image_size}×{image_size} pixels")

    def generate_test_puncta(self, n_puncta: int, strategy: FittingStrategy) -> Tuple[List, List, List, List, List, List]:
        """
        Generate test puncta data for parallel fitting.

        Args:
            n_puncta: Number of puncta to generate
            strategy: Fitting strategy (NOCOLOUR or STANDARD)

        Returns:
            Tuple of (puncta, smoothed_puncta, weights, coords, planes, masks)
        """
        print(f"Generating {n_puncta:,} test puncta for {strategy.value} fitting...")

        puncta = []
        smoothed_puncta = []
        weights = []
        relative_coords = []
        planes = []
        masks = []

        np.random.seed(42)  # Reproducible results

        for i in range(n_puncta):
            if i % 100000 == 0 and i > 0:
                print(f"  Generated {i:,}/{n_puncta:,}")

            # Generate random punctum parameters
            margin = 3
            x_center = np.random.uniform(margin, self.image_size - margin)
            y_center = np.random.uniform(margin, self.image_size - margin)
            sx = sy = np.random.uniform(1.0, 2.5)
            amplitude = np.random.uniform(500, 3000)
            background = np.random.uniform(50, 200)

            if strategy == FittingStrategy.NOCOLOUR:
                # Generate position-only image
                image = np.zeros((self.image_size, self.image_size))
                gauss_2d = np.zeros((self.image_size, self.image_size))
                x = np.arange(self.image_size)

                params = np.array([x_center, y_center, sx, sy, background, amplitude])
                image = gaussoptfuncs.WLS_nocolour_model_nobounds(params, image, x, gauss_2d)

                # Add Poisson noise
                image = np.random.poisson(np.maximum(image, 0.1)).astype(float)

                # Simple weights
                weight = 1.0 / np.maximum(image, 1.0)

                puncta.append(image)
                smoothed_puncta.append(image.copy())  # Same as raw for simplicity
                weights.append(weight)
                # No masks needed for NOCOLOUR

            else:  # STANDARD color fitting
                # Generate Bayer-filtered image
                bayer_masks = np.zeros((self.image_size, self.image_size, 3), dtype=bool)
                bayer_masks[0::2, 0::2, 0] = True  # Red
                bayer_masks[0::2, 1::2, 1] = True  # Green
                bayer_masks[1::2, 0::2, 1] = True  # Green
                bayer_masks[1::2, 1::2, 2] = True  # Blue

                # Different backgrounds and amplitudes for R, G, B
                bg_R, bg_G, bg_B = 100, 120, 80
                amp_R = amplitude * 1.0
                amp_G = amplitude * 1.2
                amp_B = amplitude * 0.8

                params = np.array([x_center, y_center, sx, sy, bg_B, bg_G, bg_R, amp_B, amp_G, amp_R])

                image = np.zeros((self.image_size, self.image_size))
                gauss_2d = np.zeros((self.image_size, self.image_size))
                x = np.arange(self.image_size)

                image = gaussoptfuncs.WLS_model_nobounds(
                    params, bayer_masks, x, gauss_2d
                )

                # Add Poisson noise
                image = np.random.poisson(np.maximum(image, 0.1)).astype(float)

                # Color-aware weights
                weight = 1.0 / np.maximum(image, 1.0)

                puncta.append(image)
                smoothed_puncta.append(image.copy())
                weights.append(weight)
                masks.append(bayer_masks)

            # Common data for all strategies
            relative_coords.append([0.0, 0.0])  # No offset
            planes.append(0)  # Single plane

        return puncta, smoothed_puncta, weights, relative_coords, planes, (masks if strategy == FittingStrategy.STANDARD else None)


    def benchmark_parallel(self, puncta: List, smoothed_puncta: List, weights: List,
                         relative_coords: List, planes: List, strategy: FittingStrategy,
                         masks: List = None) -> Dict:
        """Benchmark parallel fitting."""
        print(f"\n=== {strategy.value.upper()} Fitting ({self.n_workers} workers) ===")

        n_puncta = len(puncta)
        start_time = time.perf_counter()

        # Use the parallel method
        fit_params, _ = self.analyzer.fit_puncta_parallel_method(
            puncta, smoothed_puncta, weights, relative_coords, planes, strategy, masks
        )

        total_time = time.perf_counter() - start_time

        # Count successful fits
        successful_fits = np.sum(~np.isnan(fit_params[:, 0]))

        results = {
            'total_fits': n_puncta,
            'successful_fits': successful_fits,
            'success_rate': successful_fits / n_puncta,
            'total_time': total_time,
            'fits_per_second': n_puncta / total_time,
            'time_per_fit_ms': (total_time / n_puncta) * 1000
        }

        print(f"Results:")
        print(f"  Total fits: {results['total_fits']:,}")
        print(f"  Successful: {results['successful_fits']:,} ({results['success_rate']:.1%})")
        print(f"  Total time: {results['total_time']:.2f} seconds")
        print(f"  Rate: {results['fits_per_second']:.0f} fits/second")
        print(f"  Time per fit: {results['time_per_fit_ms']:.3f} ms")
        print(f"  Efficiency: {(results['fits_per_second'] / self.n_workers):.0f} fits/second/worker")

        return results

    def run_benchmark(self, n_puncta: int = 10000):
        """Run parallel fitting benchmark."""
        print("="*70)
        print("PARALLEL FITTING BENCHMARK")
        print("="*70)
        print(f"Testing with {n_puncta:,} puncta")

        # Test position-only fitting (NOCOLOUR)
        print(f"\n{'='*20} POSITION-ONLY FITTING {'='*20}")

        # Generate test data
        puncta_nc, smoothed_nc, weights_nc, coords_nc, planes_nc, _ = self.generate_test_puncta(
            n_puncta, FittingStrategy.NOCOLOUR
        )

        # Parallel benchmark
        results_nc = self.benchmark_parallel(
            puncta_nc, smoothed_nc, weights_nc, coords_nc, planes_nc, FittingStrategy.NOCOLOUR
        )

        # Clear memory
        del puncta_nc, smoothed_nc, weights_nc, coords_nc, planes_nc
        gc.collect()

        # Test color fitting (STANDARD)
        print(f"\n{'='*25} COLOR FITTING {'='*25}")

        # Generate color test data
        puncta_c, smoothed_c, weights_c, coords_c, planes_c, masks_c = self.generate_test_puncta(
            n_puncta, FittingStrategy.STANDARD
        )

        # Parallel benchmark
        results_c = self.benchmark_parallel(
            puncta_c, smoothed_c, weights_c, coords_c, planes_c, FittingStrategy.STANDARD, masks_c
        )

        # Clear memory
        del puncta_c, smoothed_c, weights_c, coords_c, planes_c, masks_c
        gc.collect()

        # Final comparison
        print("\n" + "="*70)
        print("PARALLEL BENCHMARK SUMMARY")
        print("="*70)

        speedup_ratio = results_nc['fits_per_second'] / results_c['fits_per_second']

        print(f"\nPosition-Only Fitting (WLS_nocolour_model_nobounds):")
        print(f"  Rate: {results_nc['fits_per_second']:.0f} fits/second")
        print(f"  Time for 100k puncta: {100000/results_nc['fits_per_second']:.1f} seconds")
        print(f"  Time for 1M puncta: {1000000/results_nc['fits_per_second']:.1f} seconds ({1000000/results_nc['fits_per_second']/60:.1f} minutes)")

        print(f"\nColor Fitting (WLS_model_nobounds):")
        print(f"  Rate: {results_c['fits_per_second']:.0f} fits/second")
        print(f"  Time for 100k puncta: {100000/results_c['fits_per_second']:.1f} seconds")
        print(f"  Time for 1M puncta: {1000000/results_c['fits_per_second']:.1f} seconds ({1000000/results_c['fits_per_second']/60:.1f} minutes)")

        print(f"\nPerformance Comparison:")
        print(f"  Position-only is {speedup_ratio:.1f}x faster than color fitting")
        print(f"  Time difference for 100k puncta: {100000/results_c['fits_per_second'] - 100000/results_nc['fits_per_second']:.1f} seconds")

        print(f"\nParallel Efficiency ({self.n_workers} workers):")
        print(f"  Position-only: {results_nc['fits_per_second']/self.n_workers:.0f} fits/second/worker")
        print(f"  Color fitting: {results_c['fits_per_second']/self.n_workers:.0f} fits/second/worker")

        return {
            'nocolour_results': results_nc,
            'color_results': results_c,
            'speedup_ratio': speedup_ratio,
            'n_workers': self.n_workers
        }

def main():
    """Main function."""
    # Test with a reasonable number of puncta to demonstrate parallel performance
    benchmark = ParallelFittingBenchmark(image_size=16)

    # Start with a smaller test for development
    results = benchmark.run_benchmark(n_puncta=100000)

    print(f"\nMachine: {os.uname().nodename}")
    print("Parallel benchmark complete!")

    return results

if __name__ == "__main__":
    results = main()