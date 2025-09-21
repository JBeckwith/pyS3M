#!/usr/bin/env python3
"""
Simple fitting benchmark to compare position-only vs color fitting approaches.

This script tests the performance difference between:
1. Position-only fitting: Simple 2D Gaussian fitting (6 parameters)
2. Color fitting: Bayer-pattern aware fitting (9+ parameters)

Measures FITTING time (not simulation) for a target number of puncta.
"""

import numpy as np
import time
import sys
import os
from pathlib import Path
from typing import Tuple, List, Dict
import gc
from scipy.optimize import minimize

class SimpleFittingBenchmark:
    """Simple benchmark for position vs color fitting without external dependencies."""

    def __init__(self,
                 image_size: int = 16,
                 pixel_size: float = 100.0,  # nm
                 psf_width: float = 160.0,   # nm
                 target_puncta: int = 10000):  # Reduced for testing
        """
        Initialize benchmark parameters.

        Args:
            image_size: Size of square images (pixels)
            pixel_size: Camera pixel size (nm)
            psf_width: PSF FWHM (nm)
            target_puncta: Target number of puncta to fit
        """
        self.image_size = image_size
        self.pixel_size = pixel_size
        self.psf_width = psf_width
        self.target_puncta = target_puncta

        # Calculate PSF sigma in pixels
        self.psf_sigma = (psf_width / pixel_size) / (2 * np.sqrt(2 * np.log(2)))

        # Create coordinate grids
        x_coords = np.arange(image_size)
        y_coords = np.arange(image_size)
        self.X, self.Y = np.meshgrid(x_coords, y_coords)

        # Create simple Bayer pattern (RGGB)
        self.bayer_pattern = np.zeros((image_size, image_size, 3), dtype=bool)
        # Red pixels
        self.bayer_pattern[0::2, 0::2, 0] = True
        # Green pixels
        self.bayer_pattern[0::2, 1::2, 1] = True
        self.bayer_pattern[1::2, 0::2, 1] = True
        # Blue pixels
        self.bayer_pattern[1::2, 1::2, 2] = True

        print(f"Simple Benchmark Configuration:")
        print(f"  - Image size: {image_size}x{image_size} pixels")
        print(f"  - Pixel size: {pixel_size:.1f} nm")
        print(f"  - PSF width: {psf_width:.1f} nm (σ = {self.psf_sigma:.2f} pixels)")
        print(f"  - Target puncta: {target_puncta:,}")

    def gaussian_2d(self, params: np.ndarray) -> np.ndarray:
        """
        Generate 2D Gaussian model.

        Args:
            params: [x_center, y_center, sigma_x, sigma_y, background, amplitude]

        Returns:
            2D Gaussian image
        """
        x0, y0, sx, sy, bg, amp = params

        gauss = bg + amp * np.exp(
            -0.5 * ((self.X - x0)**2 / sx**2 + (self.Y - y0)**2 / sy**2)
        )
        return gauss

    def position_objective(self, params: np.ndarray, data: np.ndarray, weights: np.ndarray) -> float:
        """
        Objective function for position-only fitting.

        Args:
            params: [x_center, y_center, sigma_x, sigma_y, background, amplitude]
            data: Observed image data
            weights: Fitting weights

        Returns:
            Weighted chi-squared value
        """
        model = self.gaussian_2d(params)
        residuals = (data - model) * weights
        return np.sum(residuals**2)

    def color_objective(self, params: np.ndarray, data: np.ndarray, weights: np.ndarray) -> float:
        """
        Objective function for color fitting with Bayer pattern.

        Args:
            params: [x_center, y_center, sigma_x, sigma_y, bg_R, bg_G, bg_B, amp_R, amp_G, amp_B]
            data: Observed Bayer-filtered image data
            weights: Fitting weights

        Returns:
            Weighted chi-squared value
        """
        x0, y0, sx, sy, bg_R, bg_G, bg_B, amp_R, amp_G, amp_B = params

        # Generate model for each color channel
        model = np.zeros_like(data)

        # Red pixels
        gauss_R = bg_R + amp_R * np.exp(
            -0.5 * ((self.X - x0)**2 / sx**2 + (self.Y - y0)**2 / sy**2)
        )
        model[self.bayer_pattern[:, :, 0]] = gauss_R[self.bayer_pattern[:, :, 0]]

        # Green pixels
        gauss_G = bg_G + amp_G * np.exp(
            -0.5 * ((self.X - x0)**2 / sx**2 + (self.Y - y0)**2 / sy**2)
        )
        model[self.bayer_pattern[:, :, 1]] = gauss_G[self.bayer_pattern[:, :, 1]]

        # Blue pixels
        gauss_B = bg_B + amp_B * np.exp(
            -0.5 * ((self.X - x0)**2 / sx**2 + (self.Y - y0)**2 / sy**2)
        )
        model[self.bayer_pattern[:, :, 2]] = gauss_B[self.bayer_pattern[:, :, 2]]

        residuals = (data - model) * weights
        return np.sum(residuals**2)

    def generate_position_test_images(self, n_images: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Generate test images for position-only fitting."""
        images = []
        weights = []

        np.random.seed(42)  # Reproducible results

        for i in range(n_images):
            # Random punctum parameters
            margin = 3
            x_center = np.random.uniform(margin, self.image_size - margin)
            y_center = np.random.uniform(margin, self.image_size - margin)
            amplitude = np.random.uniform(500, 3000)
            background = np.random.uniform(50, 200)
            sx = self.psf_sigma * np.random.uniform(0.8, 1.2)
            sy = self.psf_sigma * np.random.uniform(0.8, 1.2)

            # Generate clean image
            params = [x_center, y_center, sx, sy, background, amplitude]
            image = self.gaussian_2d(params)

            # Add Poisson noise
            image = np.random.poisson(np.maximum(image, 0.1)).astype(float)

            # Calculate weights
            weights_img = 1.0 / np.maximum(image, 1.0)

            images.append(image)
            weights.append(weights_img)

        return images, weights

    def generate_color_test_images(self, n_images: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Generate Bayer-filtered test images for color fitting."""
        images = []
        weights = []

        np.random.seed(42)  # Same seed for fair comparison

        for i in range(n_images):
            # Same random parameters
            margin = 3
            x_center = np.random.uniform(margin, self.image_size - margin)
            y_center = np.random.uniform(margin, self.image_size - margin)
            base_amplitude = np.random.uniform(500, 3000)
            sx = self.psf_sigma * np.random.uniform(0.8, 1.2)
            sy = self.psf_sigma * np.random.uniform(0.8, 1.2)

            # Different backgrounds and amplitudes for R, G, B
            bg_R, bg_G, bg_B = 100, 120, 80
            amp_R = base_amplitude * 1.0
            amp_G = base_amplitude * 1.2
            amp_B = base_amplitude * 0.8

            # Generate Bayer-filtered image
            image = np.zeros((self.image_size, self.image_size))

            # Red pixels
            gauss_R = bg_R + amp_R * np.exp(
                -0.5 * ((self.X - x_center)**2 / sx**2 + (self.Y - y_center)**2 / sy**2)
            )
            image[self.bayer_pattern[:, :, 0]] = gauss_R[self.bayer_pattern[:, :, 0]]

            # Green pixels
            gauss_G = bg_G + amp_G * np.exp(
                -0.5 * ((self.X - x_center)**2 / sx**2 + (self.Y - y_center)**2 / sy**2)
            )
            image[self.bayer_pattern[:, :, 1]] = gauss_G[self.bayer_pattern[:, :, 1]]

            # Blue pixels
            gauss_B = bg_B + amp_B * np.exp(
                -0.5 * ((self.X - x_center)**2 / sx**2 + (self.Y - y_center)**2 / sy**2)
            )
            image[self.bayer_pattern[:, :, 2]] = gauss_B[self.bayer_pattern[:, :, 2]]

            # Add Poisson noise
            image = np.random.poisson(np.maximum(image, 0.1)).astype(float)

            # Calculate weights
            weights_img = 1.0 / np.maximum(image, 1.0)

            images.append(image)
            weights.append(weights_img)

        return images, weights

    def benchmark_position_fitting(self, images: List[np.ndarray], weights: List[np.ndarray]) -> Dict:
        """Benchmark position-only fitting."""
        print("\n=== Position-Only Fitting Benchmark ===")

        fit_times = []
        successful_fits = 0
        n_images = len(images)

        for i, (image, weight) in enumerate(zip(images, weights)):
            if i % 1000 == 0 and i > 0:
                print(f"  Progress: {i:,}/{n_images:,} ({100*i/n_images:.1f}%)")

            # Initial guess
            center = self.image_size // 2
            max_val = np.max(image)
            min_val = np.min(image)

            initial_guess = np.array([
                center, center,                          # x, y center
                self.psf_sigma, self.psf_sigma,         # sigma x, y
                min_val, max_val - min_val              # background, amplitude
            ])

            # Bounds to keep fit reasonable
            bounds = [
                (0, self.image_size),                    # x center
                (0, self.image_size),                    # y center
                (0.5, 5.0),                             # sigma x
                (0.5, 5.0),                             # sigma y
                (0, max_val),                           # background
                (0, max_val * 2)                        # amplitude
            ]

            start_time = time.perf_counter()

            try:
                result = minimize(
                    self.position_objective,
                    initial_guess,
                    args=(image, weight),
                    bounds=bounds,
                    method='L-BFGS-B'
                )

                fit_time = time.perf_counter() - start_time
                fit_times.append(fit_time)

                if result.success:
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
        print(f"  Fitting rate: {results['fits_per_second']:.0f} fits/second")

        return results

    def benchmark_color_fitting(self, images: List[np.ndarray], weights: List[np.ndarray]) -> Dict:
        """Benchmark color fitting."""
        print("\n=== Color Fitting Benchmark ===")

        fit_times = []
        successful_fits = 0
        n_images = len(images)

        for i, (image, weight) in enumerate(zip(images, weights)):
            if i % 1000 == 0 and i > 0:
                print(f"  Progress: {i:,}/{n_images:,} ({100*i/n_images:.1f}%)")

            # Initial guess for color fitting
            center = self.image_size // 2
            max_val = np.max(image)
            min_val = np.min(image)

            initial_guess = np.array([
                center, center,                          # x, y center
                self.psf_sigma, self.psf_sigma,         # sigma x, y
                min_val, min_val, min_val,              # background R, G, B
                max_val/3, max_val/3, max_val/3         # amplitude R, G, B
            ])

            # Bounds for color fitting
            bounds = [
                (0, self.image_size),                    # x center
                (0, self.image_size),                    # y center
                (0.5, 5.0),                             # sigma x
                (0.5, 5.0),                             # sigma y
                (0, max_val), (0, max_val), (0, max_val),  # background R, G, B
                (0, max_val), (0, max_val), (0, max_val)   # amplitude R, G, B
            ]

            start_time = time.perf_counter()

            try:
                result = minimize(
                    self.color_objective,
                    initial_guess,
                    args=(image, weight),
                    bounds=bounds,
                    method='L-BFGS-B'
                )

                fit_time = time.perf_counter() - start_time
                fit_times.append(fit_time)

                if result.success:
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
        print(f"  Fitting rate: {results['fits_per_second']:.0f} fits/second")

        return results

    def run_benchmark(self):
        """Run the complete benchmark."""
        print("="*60)
        print("SIMPLE FITTING SPEED BENCHMARK")
        print("="*60)

        # Calculate batch size to manage memory
        batch_size = min(5000, self.target_puncta)
        n_batches = (self.target_puncta + batch_size - 1) // batch_size

        print(f"Running {n_batches} batches of {batch_size:,} images each")

        position_totals = {'total_time': 0, 'total_fits': 0, 'successful_fits': 0}
        color_totals = {'total_time': 0, 'total_fits': 0, 'successful_fits': 0}

        for batch in range(n_batches):
            current_batch_size = min(batch_size, self.target_puncta - batch * batch_size)
            print(f"\n--- Batch {batch + 1}/{n_batches} ({current_batch_size:,} images) ---")

            # Position fitting
            print("Generating position test images...")
            pos_images, pos_weights = self.generate_position_test_images(current_batch_size)
            pos_results = self.benchmark_position_fitting(pos_images, pos_weights)

            del pos_images, pos_weights
            gc.collect()

            # Color fitting
            print("\nGenerating color test images...")
            color_images, color_weights = self.generate_color_test_images(current_batch_size)
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
        print("FINAL BENCHMARK RESULTS")
        print("="*60)

        pos_rate = position_totals['total_fits'] / position_totals['total_time']
        color_rate = color_totals['total_fits'] / color_totals['total_time']
        speedup_ratio = pos_rate / color_rate

        print(f"\nPosition-Only Fitting:")
        print(f"  Total puncta: {position_totals['total_fits']:,}")
        print(f"  Success rate: {100*position_totals['successful_fits']/position_totals['total_fits']:.1f}%")
        print(f"  Total time: {position_totals['total_time']:.2f} seconds")
        print(f"  Rate: {pos_rate:.0f} fits/second")
        print(f"  Time for 100k puncta: {100000 / pos_rate:.1f} seconds")

        print(f"\nColor Fitting:")
        print(f"  Total puncta: {color_totals['total_fits']:,}")
        print(f"  Success rate: {100*color_totals['successful_fits']/color_totals['total_fits']:.1f}%")
        print(f"  Total time: {color_totals['total_time']:.2f} seconds")
        print(f"  Rate: {color_rate:.0f} fits/second")
        print(f"  Time for 100k puncta: {100000 / color_rate:.1f} seconds")

        print(f"\nPerformance Comparison:")
        print(f"  Position-only is {speedup_ratio:.1f}x faster than color fitting")

        return {
            'position_rate': pos_rate,
            'color_rate': color_rate,
            'speedup_ratio': speedup_ratio
        }


def main():
    """Run the benchmark."""
    # Test with smaller number first
    benchmark = SimpleFittingBenchmark(
        image_size=16,
        pixel_size=100.0,
        psf_width=160.0,
        target_puncta=10000  # Start with 10k for testing
    )

    results = benchmark.run_benchmark()

    print(f"\nTo scale to 100,000 puncta:")
    print(f"  Position-only would take: {100000 / results['position_rate']:.1f} seconds")
    print(f"  Color fitting would take: {100000 / results['color_rate']:.1f} seconds")

    return results


if __name__ == "__main__":
    results = main()