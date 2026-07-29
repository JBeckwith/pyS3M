#!/usr/bin/env python3
"""
Benchmark script to compare fitting speeds between position-only and color fitting.

Compares:
- WLS_nocolour_model_nobounds: Position-only fitting (same spectral response)
- WLS_model_nobounds: Color fitting (different spectral responses per pixel)

Tests the FITTING time (not simulation time) for 100,000 puncta.
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

# Import required modules
import pyS3M.gaussoptfuncs as gaussoptfuncs
import pyS3M.PSFFunctions as PSFFunctions
import pyS3M.SpectralFunctions as SpectralFunctions
import pyS3M.MaskFunctions as MaskFunctions
from scipy.optimize import leastsq


class FittingBenchmark:
    """Benchmark fitting performance for position vs color fitting."""

    def __init__(self,
                 image_size: int = 16,
                 pixel_size: float = 100.0,  # nm
                 psf_width: float = 160.0,   # nm
                 target_puncta: int = 100000):
        """
        Initialize benchmark parameters.

        Args:
            image_size: Size of square images (pixels)
            pixel_size: Camera pixel size (nm)
            psf_width: PSF FWHM (nm)
            target_puncta: Target number of puncta to fit for timing
        """
        self.image_size = image_size
        self.pixel_size = pixel_size
        self.psf_width = psf_width
        self.target_puncta = target_puncta

        # Calculate PSF sigma in pixels
        self.psf_sigma = (psf_width / pixel_size) / (2 * np.sqrt(2 * np.log(2)))

        # Set up coordinate grids
        self.x = np.arange(image_size)

        # Initialize spectral and mask functions for color fitting
        self.spectral_funcs = SpectralFunctions.SpectralFunctions()
        self.mask_funcs = MaskFunctions.MaskFunctions()

        # Set up Bayer pattern (RGB)
        self.bayer_masks = self.mask_funcs.return_bayer_masks(
            colours=['R', 'G', 'B'],
            image_size=image_size
        )

        print(f"Benchmark Configuration:")
        print(f"  - Image size: {image_size}x{image_size} pixels")
        print(f"  - Pixel size: {pixel_size:.1f} nm")
        print(f"  - PSF width: {psf_width:.1f} nm (σ = {self.psf_sigma:.2f} pixels)")
        print(f"  - Target puncta: {target_puncta:,}")
        print(f"  - Bayer pattern: {self.bayer_masks.shape}")

    def generate_test_images(self, n_images: int) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Generate test images for benchmarking.

        Args:
            n_images: Number of test images to generate

        Returns:
            Tuple of (images, weights, true_params)
        """
        images = []
        weights = []
        true_params_list = []

        np.random.seed(42)  # Reproducible results

        for i in range(n_images):
            # Random punctum position (away from edges)
            margin = 3
            x_center = np.random.uniform(margin, self.image_size - margin)
            y_center = np.random.uniform(margin, self.image_size - margin)

            # Random photon count (realistic range)
            amplitude = np.random.uniform(500, 3000)

            # Random background
            background = np.random.uniform(50, 200)

            # Slight PSF variation
            sx = self.psf_sigma * np.random.uniform(0.8, 1.2)
            sy = self.psf_sigma * np.random.uniform(0.8, 1.2)

            # Generate clean image
            image = np.zeros((self.image_size, self.image_size))
            for py in range(self.image_size):
                for px in range(self.image_size):
                    # 2D Gaussian
                    gauss_val = amplitude * np.exp(
                        -0.5 * ((px - x_center)**2 / sx**2 + (py - y_center)**2 / sy**2)
                    )
                    image[py, px] = background + gauss_val

            # Add Poisson noise
            image = np.random.poisson(image).astype(float)

            # Generate weights (inverse variance)
            weights_img = 1.0 / np.maximum(image, 1.0)

            images.append(image)
            weights.append(weights_img)

            # Store true parameters for validation
            true_params = np.array([x_center, y_center, sx, sy, background, amplitude])
            true_params_list.append(true_params)

        return images, weights, true_params_list

    def generate_color_test_images(self, n_images: int) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Generate test images with different spectral responses per pixel (Bayer pattern).

        Args:
            n_images: Number of test images to generate

        Returns:
            Tuple of (bayer_images, weights, true_params)
        """
        images = []
        weights = []
        true_params_list = []

        np.random.seed(42)  # Same seed for fair comparison

        # Background levels for R, G, B pixels
        bg_R = 100
        bg_G = 120
        bg_B = 80

        # Amplitude scaling for R, G, B pixels (simulating spectral response)
        amp_R = 1.0
        amp_G = 1.2
        amp_B = 0.8

        for i in range(n_images):
            # Same random parameters as non-color version
            margin = 3
            x_center = np.random.uniform(margin, self.image_size - margin)
            y_center = np.random.uniform(margin, self.image_size - margin)
            base_amplitude = np.random.uniform(500, 3000)
            sx = self.psf_sigma * np.random.uniform(0.8, 1.2)
            sy = self.psf_sigma * np.random.uniform(0.8, 1.2)

            # Generate Bayer-filtered image
            image = np.zeros((self.image_size, self.image_size))

            for py in range(self.image_size):
                for px in range(self.image_size):
                    # Determine pixel color from Bayer pattern
                    if self.bayer_masks[py, px, 0]:  # Red pixel
                        background = bg_R
                        amplitude = base_amplitude * amp_R
                    elif self.bayer_masks[py, px, 1]:  # Green pixel
                        background = bg_G
                        amplitude = base_amplitude * amp_G
                    else:  # Blue pixel
                        background = bg_B
                        amplitude = base_amplitude * amp_B

                    # 2D Gaussian
                    gauss_val = amplitude * np.exp(
                        -0.5 * ((px - x_center)**2 / sx**2 + (py - y_center)**2 / sy**2)
                    )
                    image[py, px] = background + gauss_val

            # Add Poisson noise
            image = np.random.poisson(image).astype(float)

            # Generate weights
            weights_img = 1.0 / np.maximum(image, 1.0)

            images.append(image)
            weights.append(weights_img)

            # Store true parameters (9 parameters for color: x, y, sx, sy, bg_R, bg_G, bg_B, amp_R, amp_G, amp_B)
            # But we'll use 6 for comparison: x, y, sx, sy, avg_bg, base_amplitude
            avg_bg = (bg_R + bg_G + bg_B) / 3
            true_params = np.array([x_center, y_center, sx, sy, avg_bg, base_amplitude])
            true_params_list.append(true_params)

        return images, weights, true_params_list

    def benchmark_position_fitting(self, images: List[np.ndarray],
                                 weights: List[np.ndarray]) -> Dict[str, float]:
        """
        Benchmark position-only fitting using WLS_nocolour_model_nobounds.

        Args:
            images: List of test images
            weights: List of weight matrices

        Returns:
            Dictionary with timing results
        """
        print("\n=== Position-Only Fitting Benchmark ===")
        print("Using WLS_nocolour_model_nobounds (same spectral response)")

        n_images = len(images)
        fit_times = []
        successful_fits = 0

        for i, (image, weight) in enumerate(zip(images, weights)):
            if i % 1000 == 0:
                print(f"  Progress: {i:,}/{n_images:,} ({100*i/n_images:.1f}%)")

            # Initial guess [x, y, sx, sy, background, amplitude]
            center = self.image_size // 2
            max_val = np.max(image)
            min_val = np.min(image)

            initial_guess = np.array([
                center, center,               # x, y center
                self.psf_sigma, self.psf_sigma,  # sigma x, y
                min_val,                      # background
                max_val                       # amplitude
            ])

            # Time the fitting process
            start_time = time.perf_counter()

            try:
                # Perform fitting using the same approach as ImageAnalysisFunctions
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
                print(f"  Warning: Fit {i} failed: {e}")

        results = {
            'total_fits': n_images,
            'successful_fits': successful_fits,
            'success_rate': successful_fits / n_images,
            'total_time': sum(fit_times),
            'mean_time_per_fit': np.mean(fit_times),
            'median_time_per_fit': np.median(fit_times),
            'std_time_per_fit': np.std(fit_times),
            'fits_per_second': n_images / sum(fit_times)
        }

        print(f"\nPosition-Only Fitting Results:")
        print(f"  Total fits: {results['total_fits']:,}")
        print(f"  Successful fits: {results['successful_fits']:,} ({results['success_rate']:.1%})")
        print(f"  Total time: {results['total_time']:.2f} seconds")
        print(f"  Mean time per fit: {results['mean_time_per_fit']*1000:.3f} ms")
        print(f"  Median time per fit: {results['median_time_per_fit']*1000:.3f} ms")
        print(f"  Fitting rate: {results['fits_per_second']:.0f} fits/second")

        return results

    def benchmark_color_fitting(self, images: List[np.ndarray],
                               weights: List[np.ndarray]) -> Dict[str, float]:
        """
        Benchmark color fitting using WLS_model_nobounds.

        Args:
            images: List of Bayer-filtered test images
            weights: List of weight matrices

        Returns:
            Dictionary with timing results
        """
        print("\n=== Color Fitting Benchmark ===")
        print("Using WLS_model_nobounds (different spectral responses)")

        n_images = len(images)
        fit_times = []
        successful_fits = 0

        # Pre-allocate arrays for color fitting
        background_bayer_matrix = np.zeros(self.image_size * self.image_size)
        bayer_matrix = np.zeros(self.image_size * self.image_size)
        gauss_2d = np.zeros((self.image_size, self.image_size))

        for i, (image, weight) in enumerate(zip(images, weights)):
            if i % 1000 == 0:
                print(f"  Progress: {i:,}/{n_images:,} ({100*i/n_images:.1f}%)")

            # Initial guess for color fitting [x, y, sx, sy, bg_B, bg_G, bg_R, amp_B, amp_G, amp_R]
            center = self.image_size // 2
            max_val = np.max(image)
            min_val = np.min(image)

            initial_guess = np.array([
                center, center,               # x, y center
                self.psf_sigma, self.psf_sigma,  # sigma x, y
                min_val, min_val, min_val,    # background B, G, R
                max_val, max_val, max_val     # amplitude B, G, R
            ])

            # Time the fitting process
            start_time = time.perf_counter()

            try:
                # Perform color fitting using the same approach as the color fitting functions
                size = int(image.shape[0])
                ravelsize = int(np.prod(image.shape))

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
                print(f"  Warning: Fit {i} failed: {e}")

        results = {
            'total_fits': n_images,
            'successful_fits': successful_fits,
            'success_rate': successful_fits / n_images,
            'total_time': sum(fit_times),
            'mean_time_per_fit': np.mean(fit_times),
            'median_time_per_fit': np.median(fit_times),
            'std_time_per_fit': np.std(fit_times),
            'fits_per_second': n_images / sum(fit_times)
        }

        print(f"\nColor Fitting Results:")
        print(f"  Total fits: {results['total_fits']:,}")
        print(f"  Successful fits: {results['successful_fits']:,} ({results['success_rate']:.1%})")
        print(f"  Total time: {results['total_time']:.2f} seconds")
        print(f"  Mean time per fit: {results['mean_time_per_fit']*1000:.3f} ms")
        print(f"  Median time per fit: {results['median_time_per_fit']*1000:.3f} ms")
        print(f"  Fitting rate: {results['fits_per_second']:.0f} fits/second")

        return results

    def run_benchmark(self):
        """Run the complete benchmark comparison."""
        print("="*60)
        print("FITTING SPEED BENCHMARK")
        print("="*60)
        print("Comparing position-only vs color fitting performance")
        print(f"Target: {self.target_puncta:,} puncta fits")

        # Calculate how many images we need to reach target puncta
        # We'll test in batches to manage memory
        batch_size = min(10000, self.target_puncta)
        n_batches = (self.target_puncta + batch_size - 1) // batch_size

        print(f"\nRunning {n_batches} batches of {batch_size:,} images each")

        position_results = {'total_time': 0, 'total_fits': 0, 'successful_fits': 0, 'fit_times': []}
        color_results = {'total_time': 0, 'total_fits': 0, 'successful_fits': 0, 'fit_times': []}

        for batch in range(n_batches):
            current_batch_size = min(batch_size, self.target_puncta - batch * batch_size)
            print(f"\n--- Batch {batch + 1}/{n_batches} ({current_batch_size:,} images) ---")

            # Generate test images for position fitting
            print("Generating position-only test images...")
            pos_images, pos_weights, pos_true_params = self.generate_test_images(current_batch_size)

            # Benchmark position fitting
            pos_batch_results = self.benchmark_position_fitting(pos_images, pos_weights)

            # Clear memory
            del pos_images, pos_weights, pos_true_params
            gc.collect()

            # Generate test images for color fitting (same random seed for fairness)
            print("\nGenerating color test images...")
            color_images, color_weights, color_true_params = self.generate_color_test_images(current_batch_size)

            # Benchmark color fitting
            color_batch_results = self.benchmark_color_fitting(color_images, color_weights)

            # Clear memory
            del color_images, color_weights, color_true_params
            gc.collect()

            # Accumulate results
            position_results['total_time'] += pos_batch_results['total_time']
            position_results['total_fits'] += pos_batch_results['total_fits']
            position_results['successful_fits'] += pos_batch_results['successful_fits']

            color_results['total_time'] += color_batch_results['total_time']
            color_results['total_fits'] += color_batch_results['total_fits']
            color_results['successful_fits'] += color_batch_results['successful_fits']

        # Calculate final statistics
        print("\n" + "="*60)
        print("FINAL BENCHMARK RESULTS")
        print("="*60)

        pos_rate = position_results['total_fits'] / position_results['total_time']
        color_rate = color_results['total_fits'] / color_results['total_time']
        speedup_ratio = pos_rate / color_rate

        print(f"\nPosition-Only Fitting (WLS_nocolour_model_nobounds):")
        print(f"  Total puncta fitted: {position_results['total_fits']:,}")
        print(f"  Successful fits: {position_results['successful_fits']:,} ({100*position_results['successful_fits']/position_results['total_fits']:.1f}%)")
        print(f"  Total time: {position_results['total_time']:.2f} seconds")
        print(f"  Average rate: {pos_rate:.0f} fits/second")
        print(f"  Time per 100k puncta: {100000 / pos_rate:.1f} seconds")

        print(f"\nColor Fitting (WLS_model_nobounds):")
        print(f"  Total puncta fitted: {color_results['total_fits']:,}")
        print(f"  Successful fits: {color_results['successful_fits']:,} ({100*color_results['successful_fits']/color_results['total_fits']:.1f}%)")
        print(f"  Total time: {color_results['total_time']:.2f} seconds")
        print(f"  Average rate: {color_rate:.0f} fits/second")
        print(f"  Time per 100k puncta: {100000 / color_rate:.1f} seconds")

        print(f"\nPerformance Comparison:")
        print(f"  Position-only is {speedup_ratio:.1f}x faster than color fitting")
        print(f"  Color fitting takes {speedup_ratio:.1f}x longer than position-only")

        if speedup_ratio > 1:
            time_saved = (100000 / color_rate) - (100000 / pos_rate)
            print(f"  Time saved per 100k puncta: {time_saved:.1f} seconds")

        return {
            'position_results': position_results,
            'color_results': color_results,
            'speedup_ratio': speedup_ratio,
            'position_rate': pos_rate,
            'color_rate': color_rate
        }


def main():
    """Main function to run the benchmark."""
    # You can adjust these parameters
    benchmark = FittingBenchmark(
        image_size=16,      # 16x16 pixel images (typical for SMLM)
        pixel_size=100.0,   # 100nm pixels
        psf_width=160.0,    # 160nm FWHM PSF
        target_puncta=100000  # Target 100,000 puncta
    )

    results = benchmark.run_benchmark()

    print("\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)
    print(f"Machine tested: {os.uname().nodename}")
    print(f"Results saved for analysis.")

    return results


if __name__ == "__main__":
    results = main()