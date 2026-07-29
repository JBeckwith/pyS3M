"""
Analyze the fitted parameters from simulations to understand bias source.

Check:
1. Are RGB amplitude values biased?
2. Are sigma values biased?
3. Do the biases in RGB/sigma cause the wavelength bias?
4. Is chi-squared indicating poor fit quality or wrong error estimates?
5. Does error propagation method affect the bias?
"""

import sys

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pyS3M.NileRedFunctions import NileRed_Functions

# Initialize
nrf = NileRed_Functions()

# Data path
data_dir = Path("/home/jbeckwith/Documents/pCloud/Chemistry/Lee/Data/Simulation/20251007_NileRedModelTesting")

# Test a few wavelengths at low photon count
test_cases = [
    (580, 500),
    (620, 500),
    (660, 500),
    (580, 2000),
    (620, 2000),
    (660, 2000),
]

# Filter config
filter_names = [
    "semrock-ff01-650-200",
    "semrock-di03-r514-t1-25x36",
    "semrock-ff01-515-lp"
]
NA = 1.49

# Setup optical system for forward model
import pyS3M.SpectralFunctions as SpectralFunctions
import pyS3M.PSFFunctions as PSFFunctions
spectral_funcs = SpectralFunctions.Spectral_Funcs()
psf_funcs = PSFFunctions.PSF_Functions()

wavelength_array, pixel_QYs, filter_spectra = nrf.setup_optical_system(filter_names)

print("="*80)
print("Analysis of Fitted Parameter Bias")
print("="*80)

for wl_true, n_photons in test_cases:
    print(f"\n### Wavelength: {wl_true} nm, Photons: {n_photons} ###\n")

    # Get true values from forward model
    fwd_true = nrf.nile_red_forward_model(
        wl_true, filter_spectra, wavelength_array, pixel_QYs, NA
    )

    # Normalize RGB (this is what we actually fit for wavelength)
    R_true = fwd_true['R']
    G_true = fwd_true['G']
    B_true = fwd_true['B']
    rgb_total_true = R_true + G_true + B_true
    R_norm_true = R_true / rgb_total_true
    G_norm_true = G_true / rgb_total_true
    B_norm_true = B_true / rgb_total_true
    sigma_true_nm = fwd_true['sigma_x']  # in nm
    sigma_true_px = sigma_true_nm / 69.0  # Convert to pixels for comparison with fit

    print(f"True values:")
    print(f"  R={R_true:.6f}, G={G_true:.6f}, B={B_true:.6f} (raw)")
    print(f"  R_norm={R_norm_true:.6f}, G_norm={G_norm_true:.6f}, B_norm={B_norm_true:.6f} (normalized)")
    print(f"  σ={sigma_true_nm:.4f} nm ({sigma_true_px:.4f} px)")

    # Find corresponding file
    photon_str = f"{int(n_photons):08d}"
    if n_photons < 1000:
        photon_str = f"{int(n_photons):08d}p0"
    else:
        frac = int((n_photons - int(n_photons)) * 100)
        photon_str = f"{int(n_photons):08d}p{frac:02d}"

    pattern = f"wl{wl_true}_*_{photon_str}_*rawresults.parquet"
    files = list(data_dir.glob(pattern))

    if len(files) == 0:
        print(f"  ⚠️  No files found matching: {pattern}")
        continue

    file_path = files[0]
    print(f"\n  Loading: {file_path.name}")

    # Load data
    df = pl.read_parquet(file_path)

    # Get fitted values (using correct column names)
    A_R_fit = df['A_R'].to_numpy()
    A_G_fit = df['A_G'].to_numpy()
    A_B_fit = df['A_B'].to_numpy()
    A_R_err = df['A_R_err'].to_numpy()
    A_G_err = df['A_G_err'].to_numpy()
    A_B_err = df['A_B_err'].to_numpy()
    sigma_x_fit_px = df['s_x'].to_numpy()  # in pixels from fit
    sigma_y_fit_px = df['s_y'].to_numpy()  # in pixels from fit
    sigma_x_fit_nm = sigma_x_fit_px * 69.0  # Convert to nm
    sigma_y_fit_nm = sigma_y_fit_px * 69.0  # Convert to nm
    sigma_x_err_px = df['s_x_err'].to_numpy()
    sigma_y_err_px = df['s_y_err'].to_numpy()
    sigma_x_err_nm = sigma_x_err_px * 69.0
    sigma_y_err_nm = sigma_y_err_px * 69.0
    wl_fit = df['wl_fit'].to_numpy()
    chi_sqr = df['chi_sqr'].to_numpy()

    # Remove NaNs
    valid = ~(np.isnan(A_R_fit) | np.isnan(A_G_fit) | np.isnan(A_B_fit) |
              np.isnan(sigma_x_fit_nm) | np.isnan(wl_fit) | np.isnan(chi_sqr))
    A_R_fit = A_R_fit[valid]
    A_G_fit = A_G_fit[valid]
    A_B_fit = A_B_fit[valid]
    A_R_err = A_R_err[valid]
    A_G_err = A_G_err[valid]
    A_B_err = A_B_err[valid]
    sigma_x_fit_px = sigma_x_fit_px[valid]
    sigma_y_fit_px = sigma_y_fit_px[valid]
    sigma_x_fit_nm = sigma_x_fit_nm[valid]
    sigma_y_fit_nm = sigma_y_fit_nm[valid]
    sigma_x_err_px = sigma_x_err_px[valid]
    sigma_y_err_px = sigma_y_err_px[valid]
    sigma_x_err_nm = sigma_x_err_nm[valid]
    sigma_y_err_nm = sigma_y_err_nm[valid]
    wl_fit = wl_fit[valid]
    chi_sqr = chi_sqr[valid]

    # Normalize fitted RGB values (this is what goes into wavelength fitting)
    rgb_total_fit = A_R_fit + A_G_fit + A_B_fit
    R_norm_fit = A_R_fit / rgb_total_fit
    G_norm_fit = A_G_fit / rgb_total_fit
    B_norm_fit = A_B_fit / rgb_total_fit

    print(f"\n  Chi-squared analysis (n={len(chi_sqr)}):")
    print(f"    Mean: {np.mean(chi_sqr):.3f}")
    print(f"    Median: {np.median(chi_sqr):.3f}")
    print(f"    Std: {np.std(chi_sqr):.3f}")
    if np.mean(chi_sqr) > 1.5:
        print(f"    ⚠️  High chi-squared suggests errors underestimated or model mismatch")
    elif np.mean(chi_sqr) < 0.5:
        print(f"    ⚠️  Low chi-squared suggests errors overestimated")
    else:
        print(f"    ✓ Chi-squared near 1.0 indicates good fit quality")

    print(f"\n  Raw fitted amplitudes (mean ± std, n={len(A_R_fit)}):")
    print(f"    A_R: {np.mean(A_R_fit):.6f} ± {np.std(A_R_fit):.6f}  (true: {R_true:.6f}, bias: {np.mean(A_R_fit) - R_true:+.6f})")
    print(f"    A_G: {np.mean(A_G_fit):.6f} ± {np.std(A_G_fit):.6f}  (true: {G_true:.6f}, bias: {np.mean(A_G_fit) - G_true:+.6f})")
    print(f"    A_B: {np.mean(A_B_fit):.6f} ± {np.std(A_B_fit):.6f}  (true: {B_true:.6f}, bias: {np.mean(A_B_fit) - B_true:+.6f})")

    print(f"\n  Normalized RGB (what goes into wavelength fit):")
    print(f"    R_norm: {np.mean(R_norm_fit):.6f} ± {np.std(R_norm_fit):.6f}  (true: {R_norm_true:.6f}, bias: {np.mean(R_norm_fit) - R_norm_true:+.6f})")
    print(f"    G_norm: {np.mean(G_norm_fit):.6f} ± {np.std(G_norm_fit):.6f}  (true: {G_norm_true:.6f}, bias: {np.mean(G_norm_fit) - G_norm_true:+.6f})")
    print(f"    B_norm: {np.mean(B_norm_fit):.6f} ± {np.std(B_norm_fit):.6f}  (true: {B_norm_true:.6f}, bias: {np.mean(B_norm_fit) - B_norm_true:+.6f})")

    print(f"\n  Sigma values:")
    print(f"    σ_x: {np.mean(sigma_x_fit_nm):.4f} ± {np.std(sigma_x_fit_nm):.4f} nm  (true: {sigma_true_nm:.4f}, bias: {np.mean(sigma_x_fit_nm) - sigma_true_nm:+.4f} nm)")
    print(f"    σ_y: {np.mean(sigma_y_fit_nm):.4f} ± {np.std(sigma_y_fit_nm):.4f} nm  (true: {sigma_true_nm:.4f}, bias: {np.mean(sigma_y_fit_nm) - sigma_true_nm:+.4f} nm)")

    # Wavelength results from simulation
    wl_mean = np.mean(wl_fit)
    wl_std = np.std(wl_fit)
    wl_bias = wl_mean - wl_true

    print(f"\n  Wavelength (from simulation):")
    print(f"    Fitted: {wl_mean:.2f} ± {wl_std:.2f} nm")
    print(f"    Bias: {wl_bias:+.2f} nm ({100*wl_bias/wl_true:+.2f}%)")

    # TEST 1: Can we predict the bias from the mean fitted RGB/sigma?
    print(f"\n  === TEST 1: Predicting wavelength from mean fitted parameters ===")

    rgb_from_fits = np.array([np.mean(R_norm_fit), np.mean(G_norm_fit), np.mean(B_norm_fit)])
    sigma_x_from_fits_nm = np.mean(sigma_x_fit_nm)  # Use nm values
    sigma_y_from_fits_nm = np.mean(sigma_y_fit_nm)

    # Very small errors for inverse fit (we want to see what wavelength these values predict)
    rgb_errors_tiny = np.ones(3) * 1e-6
    sigma_errors_tiny = 1e-6

    wl_from_mean_params, _ = nrf.fit_nile_red_wavelength(
        rgb_from_fits, sigma_x_from_fits_nm, sigma_y_from_fits_nm,
        rgb_errors_tiny, sigma_errors_tiny, sigma_errors_tiny,
        filter_spectra, wavelength_array, pixel_QYs, NA
    )

    predicted_bias = wl_from_mean_params - wl_true

    print(f"    Using mean fitted RGB_norm/σ → λ = {wl_from_mean_params:.2f} nm")
    print(f"    Predicted bias: {predicted_bias:+.2f} nm")
    print(f"    Actual bias: {wl_bias:+.2f} nm")
    print(f"    Difference: {wl_bias - predicted_bias:+.2f} nm")

    if abs(wl_bias - predicted_bias) < 1.0:
        print(f"    ✓ Wavelength bias is explained by RGB/σ fitting bias")
    else:
        print(f"    ⚠️  Additional bias source in wavelength fitting step")

    # TEST 2: Compare error propagation methods
    print(f"\n  === TEST 2: Testing different error propagation methods ===")

    # Method A: Current error propagation (as implemented in simulation)
    # This propagates errors through the normalization
    total_err = np.sqrt(A_R_err**2 + A_G_err**2 + A_B_err**2)
    R_norm_err_propagated = R_norm_fit * np.sqrt(
        (A_R_err/A_R_fit)**2 + (total_err/rgb_total_fit)**2
    )
    G_norm_err_propagated = G_norm_fit * np.sqrt(
        (A_G_err/A_G_fit)**2 + (total_err/rgb_total_fit)**2
    )
    B_norm_err_propagated = B_norm_fit * np.sqrt(
        (A_B_err/A_B_fit)**2 + (total_err/rgb_total_fit)**2
    )

    # Method B: Use raw amplitude errors directly (no normalization error propagation)
    R_norm_err_raw = A_R_err / rgb_total_fit
    G_norm_err_raw = A_G_err / rgb_total_fit
    B_norm_err_raw = A_B_err / rgb_total_fit

    # Method C: Constant small errors (to isolate bias from uncertainty)
    err_constant = 1e-3

    # Fit a sample of localizations with each method
    n_sample = min(100, len(R_norm_fit))
    indices = np.random.choice(len(R_norm_fit), n_sample, replace=False)

    wl_results_A = []  # Propagated errors
    wl_results_B = []  # Raw errors
    wl_results_C = []  # Constant errors

    for idx in indices:
        rgb_sample = np.array([R_norm_fit[idx], G_norm_fit[idx], B_norm_fit[idx]])

        # Method A
        rgb_err_A = np.array([R_norm_err_propagated[idx], G_norm_err_propagated[idx], B_norm_err_propagated[idx]])
        wl_A, _ = nrf.fit_nile_red_wavelength(
            rgb_sample, sigma_x_fit_nm[idx], sigma_y_fit_nm[idx],
            rgb_err_A, sigma_x_err_nm[idx], sigma_y_err_nm[idx],
            filter_spectra, wavelength_array, pixel_QYs, NA
        )
        wl_results_A.append(wl_A)

        # Method B
        rgb_err_B = np.array([R_norm_err_raw[idx], G_norm_err_raw[idx], B_norm_err_raw[idx]])
        wl_B, _ = nrf.fit_nile_red_wavelength(
            rgb_sample, sigma_x_fit_nm[idx], sigma_y_fit_nm[idx],
            rgb_err_B, sigma_x_err_nm[idx], sigma_y_err_nm[idx],
            filter_spectra, wavelength_array, pixel_QYs, NA
        )
        wl_results_B.append(wl_B)

        # Method C
        rgb_err_C = np.ones(3) * err_constant
        wl_C, _ = nrf.fit_nile_red_wavelength(
            rgb_sample, sigma_x_fit_nm[idx], sigma_y_fit_nm[idx],
            rgb_err_C, err_constant, err_constant,
            filter_spectra, wavelength_array, pixel_QYs, NA
        )
        wl_results_C.append(wl_C)

    wl_results_A = np.array(wl_results_A)
    wl_results_B = np.array(wl_results_B)
    wl_results_C = np.array(wl_results_C)

    bias_A = np.mean(wl_results_A) - wl_true
    bias_B = np.mean(wl_results_B) - wl_true
    bias_C = np.mean(wl_results_C) - wl_true

    print(f"    Method A (propagated errors): bias = {bias_A:+.2f} nm, std = {np.std(wl_results_A):.2f} nm")
    print(f"    Method B (raw amp errors):     bias = {bias_B:+.2f} nm, std = {np.std(wl_results_B):.2f} nm")
    print(f"    Method C (constant errors):    bias = {bias_C:+.2f} nm, std = {np.std(wl_results_C):.2f} nm")

    if abs(bias_A - bias_B) < 1.0 and abs(bias_B - bias_C) < 1.0:
        print(f"    ✓ Error method doesn't significantly affect bias")
    else:
        print(f"    ⚠️  Error propagation method affects the bias!")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("\nKey findings:")
print("1. Check if normalized RGB values have systematic bias")
print("2. Check if sigma values have systematic bias")
print("3. Check if chi-squared indicates fit quality issues")
print("4. Test if error propagation method affects wavelength bias")
print("\nIf bias exists in normalized RGB or sigma, it will propagate to wavelength")
print("regardless of error estimation method (errors affect uncertainty, not bias).")
