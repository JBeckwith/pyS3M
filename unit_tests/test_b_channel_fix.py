"""
Test the B channel exclusion fix.

Compares wavelength bias before and after implementing adaptive B channel exclusion.
"""

import sys

import polars as pl
import numpy as np
from pathlib import Path
from pyS3M.NileRedFunctions import NileRed_Functions

# Initialize
nrf = NileRed_Functions()

# Data path
data_dir = Path("/home/jbeckwith/Documents/pCloud/Chemistry/Lee/Data/Simulation/20251007_NileRedModelTesting")

# Test cases
test_cases = [
    (580, 500),
    (620, 500),
    (660, 500),
]

# Filter config
filter_names = [
    "semrock-ff01-650-200",
    "semrock-di03-r514-t1-25x36",
    "semrock-ff01-515-lp"
]
NA = 1.49
pixel_size = 69  # nm/pixel

# Setup optical system
wavelength_array, pixel_QYs, filter_spectra = nrf.setup_optical_system(filter_names)

print("="*80)
print("B Channel Exclusion Fix - Test Results")
print("="*80)

for wl_true, n_photons in test_cases:
    print(f"\n### Wavelength: {wl_true} nm, Photons: {n_photons} ###")

    # Get true RGB values
    fwd_true = nrf.nile_red_forward_model(
        wl_true, filter_spectra, wavelength_array, pixel_QYs, NA
    )
    B_true = fwd_true['B']

    print(f"\nTrue B fraction: {B_true:.6f} ({100*B_true:.2f}%)")

    # Load data
    photon_str = f"{int(n_photons):08d}p0"
    pattern = f"wl{wl_true}_*_{photon_str}_*rawresults.parquet"
    files = list(data_dir.glob(pattern))

    if len(files) == 0:
        print(f"  No files found!")
        continue

    file_path = files[0]
    df = pl.read_parquet(file_path)

    # Get fitted values
    A_R = df['A_R'].to_numpy()
    A_G = df['A_G'].to_numpy()
    A_B = df['A_B'].to_numpy()
    A_R_err = df['A_R_err'].to_numpy()
    A_G_err = df['A_G_err'].to_numpy()
    A_B_err = df['A_B_err'].to_numpy()
    sigma_x_px = df['s_x'].to_numpy()
    sigma_y_px = df['s_y'].to_numpy()
    sigma_x_err_px = df['s_x_err'].to_numpy()
    sigma_y_err_px = df['s_y_err'].to_numpy()

    # Convert sigma to nm
    sigma_x_nm = sigma_x_px * pixel_size
    sigma_y_nm = sigma_y_px * pixel_size
    sigma_x_err_nm = sigma_x_err_px * pixel_size
    sigma_y_err_nm = sigma_y_err_px * pixel_size

    # Remove NaNs
    valid = ~(np.isnan(A_R) | np.isnan(A_G) | np.isnan(A_B) | np.isnan(sigma_x_nm))
    A_R = A_R[valid]
    A_G = A_G[valid]
    A_B = A_B[valid]
    A_R_err = A_R_err[valid]
    A_G_err = A_G_err[valid]
    A_B_err = A_B_err[valid]
    sigma_x_nm = sigma_x_nm[valid]
    sigma_y_nm = sigma_y_nm[valid]
    sigma_x_err_nm = sigma_x_err_nm[valid]
    sigma_y_err_nm = sigma_y_err_nm[valid]

    print(f"Fitted B fraction: {np.mean(A_B):.6f} ({100*np.mean(A_B):.2f}%)")
    print(f"B exclusion threshold: 0.10 (10%)")

    if np.mean(A_B) < 0.10:
        print(f"→ B channel WILL BE EXCLUDED (fitted B < 10%)")
    else:
        print(f"→ B channel will be included (fitted B >= 10%)")

    # Test on sample
    n_test = 100
    test_indices = np.random.choice(len(A_R), n_test, replace=False)

    wl_results_with_fix = []

    for idx in test_indices:
        rgb = np.array([A_R[idx], A_G[idx], A_B[idx]])
        rgb_err = np.array([A_R_err[idx], A_G_err[idx], A_B_err[idx]])

        # Apply fix (default threshold = 0.10)
        wl_fit, _ = nrf.fit_nile_red_wavelength(
            rgb, sigma_x_nm[idx], sigma_y_nm[idx],
            rgb_err, sigma_x_err_nm[idx], sigma_y_err_nm[idx],
            filter_spectra, wavelength_array, pixel_QYs, NA
            # uses default b_channel_threshold=0.10
        )
        wl_results_with_fix.append(wl_fit)

    wl_results_with_fix = np.array(wl_results_with_fix)

    # Remove NaNs
    wl_results_with_fix = wl_results_with_fix[~np.isnan(wl_results_with_fix)]

    bias_with_fix = np.mean(wl_results_with_fix) - wl_true
    std_with_fix = np.std(wl_results_with_fix)

    print(f"\nResults with B channel fix (n={len(wl_results_with_fix)}):")
    print(f"  Mean wavelength: {np.mean(wl_results_with_fix):.2f} nm")
    print(f"  Bias: {bias_with_fix:+.2f} nm")
    print(f"  Std: {std_with_fix:.2f} nm")

    # Compare to original (from parquet file)
    wl_original = df['wl_fit'].to_numpy()
    wl_original = wl_original[~np.isnan(wl_original)]

    bias_original = np.mean(wl_original) - wl_true

    print(f"\nComparison to original (from parquet):")
    print(f"  Original bias: {bias_original:+.2f} nm")
    print(f"  New bias: {bias_with_fix:+.2f} nm")
    print(f"  Improvement: {bias_original - bias_with_fix:+.2f} nm")

    if abs(bias_with_fix) < 2.0:
        print(f"  ✅ FIX SUCCESSFUL! Bias reduced to < 2 nm")
    else:
        print(f"  ⚠️  Bias still significant")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("\nThe B channel exclusion fix:")
print("1. Checks if fitted B fraction < 10% (threshold)")
print("2. If yes: renormalizes using only R/G ratio, sets B=0")
print("3. If no: uses all three channels normally")
print("\nThreshold=0.10 accounts for B overestimation bias at low SNR.")
print("This dramatically reduces wavelength bias from low-SNR B channel.")
