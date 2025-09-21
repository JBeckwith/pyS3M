#!/usr/bin/env python3
"""
Quick test to verify fitting benchmark works and estimate times.
"""

import numpy as np
import time
from scipy.optimize import minimize

def gaussian_2d(X, Y, params):
    """Generate 2D Gaussian."""
    x0, y0, sx, sy, bg, amp = params
    return bg + amp * np.exp(-0.5 * ((X - x0)**2 / sx**2 + (Y - y0)**2 / sy**2))

def position_objective(params, data, weights, X, Y):
    """Position-only fitting objective."""
    model = gaussian_2d(X, Y, params)
    residuals = (data - model) * weights
    return np.sum(residuals**2)

def color_objective(params, data, weights, X, Y, bayer_pattern):
    """Color fitting objective with Bayer pattern."""
    x0, y0, sx, sy, bg_R, bg_G, bg_B, amp_R, amp_G, amp_B = params

    model = np.zeros_like(data)

    # Red pixels
    gauss_R = bg_R + amp_R * np.exp(-0.5 * ((X - x0)**2 / sx**2 + (Y - y0)**2 / sy**2))
    model[bayer_pattern[:, :, 0]] = gauss_R[bayer_pattern[:, :, 0]]

    # Green pixels
    gauss_G = bg_G + amp_G * np.exp(-0.5 * ((X - x0)**2 / sx**2 + (Y - y0)**2 / sy**2))
    model[bayer_pattern[:, :, 1]] = gauss_G[bayer_pattern[:, :, 1]]

    # Blue pixels
    gauss_B = bg_B + amp_B * np.exp(-0.5 * ((X - x0)**2 / sx**2 + (Y - y0)**2 / sy**2))
    model[bayer_pattern[:, :, 2]] = gauss_B[bayer_pattern[:, :, 2]]

    residuals = (data - model) * weights
    return np.sum(residuals**2)

def main():
    print("Quick Fitting Speed Test")
    print("="*40)

    # Small test parameters
    image_size = 16
    n_test = 100  # Just 100 images for quick test

    # Set up coordinates
    x_coords = np.arange(image_size)
    y_coords = np.arange(image_size)
    X, Y = np.meshgrid(x_coords, y_coords)

    # Create Bayer pattern
    bayer_pattern = np.zeros((image_size, image_size, 3), dtype=bool)
    bayer_pattern[0::2, 0::2, 0] = True  # Red
    bayer_pattern[0::2, 1::2, 1] = True  # Green
    bayer_pattern[1::2, 0::2, 1] = True  # Green
    bayer_pattern[1::2, 1::2, 2] = True  # Blue

    print(f"Testing with {n_test} images of size {image_size}x{image_size}")

    # Test position-only fitting
    print("\nPosition-only fitting test...")
    position_times = []

    np.random.seed(42)
    for i in range(n_test):
        if i % 25 == 0:
            print(f"  {i}/{n_test}")

        # Generate test image
        true_params = [8, 8, 1.5, 1.5, 100, 1000]  # x, y, sx, sy, bg, amp
        image = gaussian_2d(X, Y, true_params)
        image = np.random.poisson(np.maximum(image, 0.1))
        weights = 1.0 / np.maximum(image, 1.0)

        # Initial guess
        initial_guess = [8.5, 8.5, 1.0, 1.0, 50, 500]
        bounds = [(0, 16), (0, 16), (0.5, 5), (0.5, 5), (0, 2000), (0, 5000)]

        start_time = time.perf_counter()
        try:
            result = minimize(position_objective, initial_guess,
                            args=(image, weights, X, Y), bounds=bounds, method='L-BFGS-B')
            fit_time = time.perf_counter() - start_time
            position_times.append(fit_time)
        except:
            position_times.append(0.1)  # Fallback time

    # Test color fitting
    print("\nColor fitting test...")
    color_times = []

    np.random.seed(42)
    for i in range(n_test):
        if i % 25 == 0:
            print(f"  {i}/{n_test}")

        # Generate Bayer test image
        image = np.zeros((image_size, image_size))

        # Different parameters for R, G, B
        x0, y0, sx, sy = 8, 8, 1.5, 1.5
        bg_R, bg_G, bg_B = 100, 120, 80
        amp_R, amp_G, amp_B = 1000, 1200, 800

        # Red pixels
        gauss_R = bg_R + amp_R * np.exp(-0.5 * ((X - x0)**2 / sx**2 + (Y - y0)**2 / sy**2))
        image[bayer_pattern[:, :, 0]] = gauss_R[bayer_pattern[:, :, 0]]

        # Green pixels
        gauss_G = bg_G + amp_G * np.exp(-0.5 * ((X - x0)**2 / sx**2 + (Y - y0)**2 / sy**2))
        image[bayer_pattern[:, :, 1]] = gauss_G[bayer_pattern[:, :, 1]]

        # Blue pixels
        gauss_B = bg_B + amp_B * np.exp(-0.5 * ((X - x0)**2 / sx**2 + (Y - y0)**2 / sy**2))
        image[bayer_pattern[:, :, 2]] = gauss_B[bayer_pattern[:, :, 2]]

        image = np.random.poisson(np.maximum(image, 0.1))
        weights = 1.0 / np.maximum(image, 1.0)

        # Initial guess for color fitting
        initial_guess = [8.5, 8.5, 1.0, 1.0, 50, 60, 40, 500, 600, 400]
        bounds = [(0, 16), (0, 16), (0.5, 5), (0.5, 5),
                  (0, 2000), (0, 2000), (0, 2000), (0, 5000), (0, 5000), (0, 5000)]

        start_time = time.perf_counter()
        try:
            result = minimize(color_objective, initial_guess,
                            args=(image, weights, X, Y, bayer_pattern), bounds=bounds, method='L-BFGS-B')
            fit_time = time.perf_counter() - start_time
            color_times.append(fit_time)
        except:
            color_times.append(0.2)  # Fallback time

    # Results
    print("\n" + "="*40)
    print("RESULTS")
    print("="*40)

    pos_mean = np.mean(position_times)
    color_mean = np.mean(color_times)
    speedup = color_mean / pos_mean

    pos_rate = 1.0 / pos_mean
    color_rate = 1.0 / color_mean

    print(f"\nPosition-only fitting:")
    print(f"  Mean time per fit: {pos_mean*1000:.2f} ms")
    print(f"  Rate: {pos_rate:.1f} fits/second")
    print(f"  Time for 100k fits: {100000/pos_rate:.1f} seconds ({100000/pos_rate/60:.1f} minutes)")

    print(f"\nColor fitting:")
    print(f"  Mean time per fit: {color_mean*1000:.2f} ms")
    print(f"  Rate: {color_rate:.1f} fits/second")
    print(f"  Time for 100k fits: {100000/color_rate:.1f} seconds ({100000/color_rate/60:.1f} minutes)")

    print(f"\nComparison:")
    print(f"  Color fitting is {speedup:.1f}x slower than position-only")
    print(f"  Position-only is {1/speedup:.1f}x faster than color fitting")

    return {
        'position_rate': pos_rate,
        'color_rate': color_rate,
        'speedup': speedup
    }

if __name__ == "__main__":
    main()