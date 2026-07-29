#!/usr/bin/env python3
"""
Final corrected fitting benchmark with proper error handling and debugging.
"""

import numpy as np
import time
import sys
import os
from pathlib import Path
import traceback

# Add src directory to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"

import pyS3M.gaussoptfuncs as gaussoptfuncs
from scipy.optimize import leastsq

def test_simple_fits():
    """Test basic fitting functionality before full benchmark."""
    print("Testing basic fitting functionality...")

    # Create simple test image
    image_size = 16
    x = np.arange(image_size)

    # Create simple 2D Gaussian
    center = 8
    sigma = 1.5
    amplitude = 1000
    background = 100

    # Generate test image using position-only function
    image = np.zeros((image_size, image_size))
    gauss_2d = np.zeros((image_size, image_size))

    # Parameters: [x, y, sx, sy, bg, amp]
    params_pos = np.array([center, center, sigma, sigma, background, amplitude])

    try:
        image = gaussoptfuncs.WLS_nocolour_model_nobounds(params_pos, image, x, gauss_2d)
        print(f"✓ Position-only model generation works")
        print(f"  Image range: {image.min():.1f} to {image.max():.1f}")
    except Exception as e:
        print(f"✗ Position-only model failed: {e}")
        return False

    # Test position-only fitting
    weights = np.ones_like(image)
    size = image_size
    ravelsize = image_size * image_size

    # Initial guess slightly off
    initial_guess = np.array([center+0.5, center+0.5, sigma*0.8, sigma*0.8, background*0.8, amplitude*0.8])

    try:
        start_time = time.perf_counter()
        pfit, pcov, infodict, errmsg, success = leastsq(
            gaussoptfuncs.WLS_chi_nocolour_nobounds,
            x0=initial_guess,
            args=(image, weights, size, ravelsize),
            full_output=True,
            ftol=1e-6,
            xtol=1e-6,
        )
        fit_time = time.perf_counter() - start_time

        print(f"✓ Position-only fitting works")
        print(f"  Success code: {success}")
        print(f"  Fit time: {fit_time*1000:.2f} ms")
        print(f"  True params: {params_pos}")
        print(f"  Fitted params: {pfit}")

    except Exception as e:
        print(f"✗ Position-only fitting failed: {e}")
        print(f"  Error details: {traceback.format_exc()}")
        return False

    # Test color fitting setup
    bayer_masks = np.zeros((image_size, image_size, 3), dtype=bool)
    bayer_masks[0::2, 0::2, 0] = True  # Red
    bayer_masks[0::2, 1::2, 1] = True  # Green
    bayer_masks[1::2, 0::2, 1] = True  # Green
    bayer_masks[1::2, 1::2, 2] = True  # Blue

    # Parameters for color: [x, y, sx, sy, bg_B, bg_G, bg_R, amp_B, amp_G, amp_R]
    params_color = np.array([center, center, sigma, sigma,
                           background*0.8, background, background*1.2,  # bg B, G, R
                           amplitude*0.8, amplitude, amplitude*1.2])    # amp B, G, R

    try:
        color_image = np.zeros((image_size, image_size))
        color_gauss_2d = np.zeros((image_size, image_size))

        color_image = gaussoptfuncs.WLS_model_nobounds(
            params_color, bayer_masks, x, color_gauss_2d
        )

        print(f"✓ Color model generation works")
        print(f"  Color image range: {color_image.min():.1f} to {color_image.max():.1f}")

        # Test color fitting
        color_weights = np.ones_like(color_image)
        initial_guess_color = params_color * 0.9  # Slight perturbation

        start_time = time.perf_counter()
        pfit_color, pcov_color, infodict_color, errmsg_color, success_color = leastsq(
            gaussoptfuncs.WLS_chi_nobounds,
            x0=initial_guess_color,
            args=(color_image, bayer_masks, color_weights, image_size, image_size*image_size),
            full_output=True,
            ftol=1e-6,
            xtol=1e-6,
        )
        color_fit_time = time.perf_counter() - start_time

        print(f"✓ Color fitting works")
        print(f"  Success code: {success_color}")
        print(f"  Fit time: {color_fit_time*1000:.2f} ms")
        print(f"  Speed ratio: {color_fit_time/fit_time:.1f}x slower than position-only")

    except Exception as e:
        print(f"✗ Color fitting failed: {e}")
        print(f"  Error details: {traceback.format_exc()}")
        return False

    return True

def benchmark_fitting_speeds(n_fits=1000):
    """Benchmark the two fitting approaches."""
    print(f"\nBenchmarking {n_fits} fits...")

    image_size = 16
    x = np.arange(image_size)

    # Bayer masks
    bayer_masks = np.zeros((image_size, image_size, 3), dtype=bool)
    bayer_masks[0::2, 0::2, 0] = True  # Red
    bayer_masks[0::2, 1::2, 1] = True  # Green
    bayer_masks[1::2, 0::2, 1] = True  # Green
    bayer_masks[1::2, 1::2, 2] = True  # Blue

    # Position-only benchmark
    print("\nPosition-only fitting benchmark...")
    position_times = []
    position_successes = 0

    np.random.seed(42)

    for i in range(n_fits):
        if i % 200 == 0:
            print(f"  {i}/{n_fits}")

        # Generate random test image
        center = np.random.uniform(4, 12)
        sigma = np.random.uniform(1.0, 2.5)
        amplitude = np.random.uniform(500, 2000)
        background = np.random.uniform(50, 150)

        params = np.array([center, center, sigma, sigma, background, amplitude])
        image = np.zeros((image_size, image_size))
        gauss_2d = np.zeros((image_size, image_size))

        image = gaussoptfuncs.WLS_nocolour_model_nobounds(params, image, x, gauss_2d)
        image = np.random.poisson(np.maximum(image, 0.1)).astype(float)
        weights = 1.0 / np.maximum(image, 1.0)

        # Perturbed initial guess
        initial_guess = params + np.random.normal(0, 0.1 * params)

        start_time = time.perf_counter()
        try:
            pfit, pcov, infodict, errmsg, success = leastsq(
                gaussoptfuncs.WLS_chi_nocolour_nobounds,
                x0=initial_guess,
                args=(image, weights, image_size, image_size*image_size),
                full_output=True,
                ftol=1e-6,
                xtol=1e-6,
                maxfev=1000
            )
            fit_time = time.perf_counter() - start_time
            position_times.append(fit_time)

            if success in [1, 2, 3, 4]:
                position_successes += 1

        except Exception:
            fit_time = time.perf_counter() - start_time
            position_times.append(fit_time)

    # Color fitting benchmark
    print("\nColor fitting benchmark...")
    color_times = []
    color_successes = 0

    np.random.seed(42)  # Same random sequence

    for i in range(n_fits):
        if i % 200 == 0:
            print(f"  {i}/{n_fits}")

        # Same random parameters
        center = np.random.uniform(4, 12)
        sigma = np.random.uniform(1.0, 2.5)
        base_amplitude = np.random.uniform(500, 2000)

        # Different backgrounds and amplitudes for R, G, B
        bg_R, bg_G, bg_B = 100, 120, 80
        amp_R = base_amplitude * 1.0
        amp_G = base_amplitude * 1.2
        amp_B = base_amplitude * 0.8

        params = np.array([center, center, sigma, sigma, bg_B, bg_G, bg_R, amp_B, amp_G, amp_R])
        image = np.zeros((image_size, image_size))
        gauss_2d = np.zeros((image_size, image_size))

        image = gaussoptfuncs.WLS_model_nobounds(
            params, bayer_masks, x, gauss_2d
        )
        image = np.random.poisson(np.maximum(image, 0.1)).astype(float)
        weights = 1.0 / np.maximum(image, 1.0)

        # Perturbed initial guess
        initial_guess = params + np.random.normal(0, 0.1 * params)

        start_time = time.perf_counter()
        try:
            pfit, pcov, infodict, errmsg, success = leastsq(
                gaussoptfuncs.WLS_chi_nobounds,
                x0=initial_guess,
                args=(image, bayer_masks, weights, image_size, image_size*image_size),
                full_output=True,
                ftol=1e-6,
                xtol=1e-6,
                maxfev=1000
            )
            fit_time = time.perf_counter() - start_time
            color_times.append(fit_time)

            if success in [1, 2, 3, 4]:
                color_successes += 1

        except Exception:
            fit_time = time.perf_counter() - start_time
            color_times.append(fit_time)

    # Results
    print("\n" + "="*60)
    print("FITTING SPEED BENCHMARK RESULTS")
    print("="*60)

    pos_mean_time = np.mean(position_times)
    color_mean_time = np.mean(color_times)

    pos_rate = 1.0 / pos_mean_time
    color_rate = 1.0 / color_mean_time

    speedup = color_mean_time / pos_mean_time

    print(f"\nPosition-Only Fitting (WLS_nocolour_model_nobounds):")
    print(f"  Success rate: {position_successes}/{n_fits} ({100*position_successes/n_fits:.1f}%)")
    print(f"  Mean time per fit: {pos_mean_time*1000:.3f} ms")
    print(f"  Fitting rate: {pos_rate:.0f} fits/second")
    print(f"  Time for 100k fits: {100000/pos_rate:.1f} seconds ({100000/pos_rate/60:.1f} minutes)")

    print(f"\nColor Fitting (WLS_model_nobounds):")
    print(f"  Success rate: {color_successes}/{n_fits} ({100*color_successes/n_fits:.1f}%)")
    print(f"  Mean time per fit: {color_mean_time*1000:.3f} ms")
    print(f"  Fitting rate: {color_rate:.0f} fits/second")
    print(f"  Time for 100k fits: {100000/color_rate:.1f} seconds ({100000/color_rate/60:.1f} minutes)")

    print(f"\nPerformance Comparison:")
    if speedup > 1:
        print(f"  Color fitting is {speedup:.1f}x slower than position-only")
        print(f"  Position-only is {1/speedup:.1f}x faster than color fitting")
    else:
        print(f"  Position-only is {1/speedup:.1f}x slower than color fitting")
        print(f"  Color fitting is {speedup:.1f}x faster than position-only")

    return {
        'position_rate': pos_rate,
        'color_rate': color_rate,
        'speedup': speedup,
        'position_success_rate': position_successes/n_fits,
        'color_success_rate': color_successes/n_fits
    }

def main():
    """Main function."""
    print("="*60)
    print("FITTING SPEED BENCHMARK")
    print("Comparing WLS_nocolour_model_nobounds vs WLS_model_nobounds")
    print("="*60)

    # First test basic functionality
    if not test_simple_fits():
        print("Basic tests failed. Exiting.")
        return

    # Run benchmark
    results = benchmark_fitting_speeds(1000)

    print(f"\nSUMMARY for 100,000 puncta on {os.uname().nodename}:")
    print(f"  Position-only: {100000/results['position_rate']:.0f} seconds")
    print(f"  Color fitting: {100000/results['color_rate']:.0f} seconds")

    return results

if __name__ == "__main__":
    main()