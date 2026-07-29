"""
Diagnostic script to investigate wavelength bias in Nile Red model.

Checks:
1. Forward model accuracy (wavelength -> RGB -> recovered wavelength)
2. LUT interpolation accuracy
3. Inverse fitting algorithm behavior
4. Systematic trends in bias
"""

import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pyS3M.NileRedFunctions import NileRed_Functions
import pyS3M.SpectralFunctions as SpectralFunctions
import pyS3M.PSFFunctions as PSFFunctions

# Initialize
nrf = NileRed_Functions()
spectral_funcs = SpectralFunctions.Spectral_Funcs()
psf_funcs = PSFFunctions.PSF_Functions()

# Test wavelengths
wavelengths_test = np.arange(580, 685, 5)
n_tests = len(wavelengths_test)

# Filter configuration (Nile Red)
filter_names = [
    "semrock-ff01-650-200",
    "semrock-di03-r514-t1-25x36",
    "semrock-ff01-515-lp"
]
NA = 1.49

# Setup optical system
wavelength_array, pixel_QYs, filter_spectra = nrf.setup_optical_system(filter_names)

print("="*80)
print("Nile Red Wavelength Bias Diagnostic")
print("="*80)

# Test 1: Forward model roundtrip (no noise)
print("\n### Test 1: Perfect Forward Model Roundtrip (No Noise) ###\n")
print("Testing if the inverse model can perfectly recover wavelengths from")
print("the forward model when there's no noise...\n")

roundtrip_results = []

for wl_true in wavelengths_test:
    # Forward model: wavelength -> RGB + sigma
    fwd_result = nrf.nile_red_forward_model(
        wl_true, filter_spectra, wavelength_array, pixel_QYs, NA
    )

    # Create observed data (perfect, no noise)
    observed_rgb = np.array([fwd_result['R'], fwd_result['G'], fwd_result['B']])
    observed_sigma_x = fwd_result['sigma_x']
    observed_sigma_y = fwd_result['sigma_y']

    # Inverse model: RGB + sigma -> wavelength
    # Use very small errors to simulate "perfect" data
    rgb_errors = np.ones(3) * 1e-10
    sigma_errors = 1e-10

    wl_recovered, pred = nrf.fit_nile_red_wavelength(
        observed_rgb, observed_sigma_x, observed_sigma_y,
        rgb_errors, sigma_errors, sigma_errors,
        filter_spectra, wavelength_array, pixel_QYs, NA
    )

    bias = wl_recovered - wl_true

    roundtrip_results.append({
        'wavelength_true': wl_true,
        'wavelength_recovered': wl_recovered,
        'bias': bias,
        'R_true': fwd_result['R'],
        'G_true': fwd_result['G'],
        'B_true': fwd_result['B'],
        'sigma_true': fwd_result['sigma_x']
    })

    print(f"λ_true={wl_true:.1f} nm → λ_recovered={wl_recovered:.2f} nm, bias={bias:+.2f} nm")

df_roundtrip = pd.DataFrame(roundtrip_results)

print(f"\nRoundtrip bias statistics:")
print(f"  Mean bias: {df_roundtrip['bias'].mean():+.2f} nm")
print(f"  Std bias: {df_roundtrip['bias'].std():.2f} nm")
print(f"  Max bias: {df_roundtrip['bias'].abs().max():.2f} nm")
print(f"  RMS bias: {np.sqrt((df_roundtrip['bias']**2).mean()):.2f} nm")

# Test 2: LUT vs Full Forward Model
print("\n### Test 2: LUT Accuracy vs Full Forward Model ###\n")
print("Checking if the LUT interpolation introduces systematic errors...\n")

lut_comparison = []

for wl_true in wavelengths_test:
    # Full forward model
    fwd_full = nrf.nile_red_forward_model(
        wl_true, filter_spectra, wavelength_array, pixel_QYs, NA
    )

    # LUT-based forward model
    fwd_lut = nrf.nile_red_forward_model_lut(
        wl_true, filter_names, NA
    )

    # Compare
    r_diff = fwd_lut['R'] - fwd_full['R']
    g_diff = fwd_lut['G'] - fwd_full['G']
    b_diff = fwd_lut['B'] - fwd_full['B']
    sigma_diff = fwd_lut['sigma_x'] - fwd_full['sigma_x']

    lut_comparison.append({
        'wavelength': wl_true,
        'R_diff': r_diff,
        'G_diff': g_diff,
        'B_diff': b_diff,
        'sigma_diff': sigma_diff,
        'rgb_rmse': np.sqrt(r_diff**2 + g_diff**2 + b_diff**2)
    })

    if abs(r_diff) > 0.001 or abs(g_diff) > 0.001 or abs(b_diff) > 0.001:
        print(f"λ={wl_true:.1f} nm: ΔR={r_diff:+.4f}, ΔG={g_diff:+.4f}, ΔB={b_diff:+.4f}, Δσ={sigma_diff:+.4f} nm")

df_lut = pd.DataFrame(lut_comparison)
print(f"\nLUT accuracy:")
print(f"  Max RGB RMSE: {df_lut['rgb_rmse'].max():.6f}")
print(f"  Max sigma error: {df_lut['sigma_diff'].abs().max():.6f} nm")

# Test 3: Check RGB ratios
print("\n### Test 3: RGB Ratios Across Wavelength Range ###\n")
print("Checking if RGB ratios change monotonically with wavelength...\n")

print("Wavelength   R/G ratio   B/G ratio   R/(R+G+B)")
print("-" * 55)
for wl in wavelengths_test:
    fwd = nrf.nile_red_forward_model(
        wl, filter_spectra, wavelength_array, pixel_QYs, NA
    )
    r, g, b = fwd['R'], fwd['G'], fwd['B']
    total = r + g + b
    print(f"{wl:5.0f} nm    {r/g:8.4f}   {b/g:8.4f}   {r/total:8.4f}")

# Check for non-monotonicity
r_g_ratios = []
for wl in wavelengths_test:
    fwd = nrf.nile_red_forward_model(
        wl, filter_spectra, wavelength_array, pixel_QYs, NA
    )
    r_g_ratios.append(fwd['R'] / fwd['G'])

r_g_ratios = np.array(r_g_ratios)
non_monotonic = np.sum(np.diff(r_g_ratios) < 0)

print(f"\nR/G ratio monotonicity: {non_monotonic} reversals (should be 0 for monotonic)")

# Test 4: Check if bias correlates with specific features
print("\n### Test 4: Correlation Analysis ###\n")

# Load the actual simulation results
csv_path = "/home/jbeckwith/Documents/pCloud/Chemistry/Lee/Data/Simulation/20251007_NileRedModelTesting/wavelength_precision_summary.csv"
df_sim = pd.read_csv(csv_path)

# Filter for a specific photon count
photon_count = 500
df_500 = df_sim[df_sim['n_photons'] == photon_count].copy()

# Merge with forward model results
df_merged = df_500.merge(df_roundtrip, on='wavelength_true', suffixes=('_sim', '_perfect'))

print(f"Simulation with {photon_count} photons:")
print(f"  Mean bias (with noise): {df_merged['wavelength_bias_sim'].mean():+.2f} nm")
print(f"  Mean bias (perfect): {df_merged['bias'].mean():+.2f} nm")
print(f"  Difference: {(df_merged['wavelength_bias_sim'] - df_merged['bias']).mean():+.2f} nm")

print("\nBias vs wavelength_true correlation:", np.corrcoef(df_merged['wavelength_true'], df_merged['wavelength_bias_sim'])[0,1])
print("Bias vs R/G ratio correlation:", np.corrcoef(df_merged['R_true'], df_merged['wavelength_bias_sim'])[0,1])

# Test 5: Check optimization algorithm behavior
print("\n### Test 5: Optimization Algorithm Stability ###\n")
print("Testing if different initial guesses converge to same solution...\n")

test_wl = 620.0
fwd = nrf.nile_red_forward_model(
    test_wl, filter_spectra, wavelength_array, pixel_QYs, NA
)

observed_rgb = np.array([fwd['R'], fwd['G'], fwd['B']])
rgb_errors = np.ones(3) * 1e-6

# Try different initial guesses
initial_guesses = [580, 600, 620, 640, 660, 680]
recovered_wavelengths = []

for init_guess in initial_guesses:
    # Temporarily change the default
    nrf_test = NileRed_Functions(wavelength_center_init=init_guess)

    wl_recovered, _ = nrf_test.fit_nile_red_wavelength(
        observed_rgb, fwd['sigma_x'], fwd['sigma_y'],
        rgb_errors, 1e-6, 1e-6,
        filter_spectra, wavelength_array, pixel_QYs, NA
    )

    recovered_wavelengths.append(wl_recovered)
    print(f"Initial guess: {init_guess} nm → Recovered: {wl_recovered:.2f} nm")

print(f"\nStandard deviation of recovered wavelengths: {np.std(recovered_wavelengths):.4f} nm")
print(f"Range: {np.ptp(recovered_wavelengths):.4f} nm")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("\n1. PERFECT ROUNDTRIP TEST:")
print(f"   - RMS bias: {np.sqrt((df_roundtrip['bias']**2).mean()):.2f} nm")
if np.sqrt((df_roundtrip['bias']**2).mean()) > 0.1:
    print("   ⚠️  SIGNIFICANT BIAS DETECTED in perfect data!")
    print("   → The inverse model cannot perfectly recover wavelengths")
    print("   → This suggests a model mismatch or optimization issue")
else:
    print("   ✓ Roundtrip is accurate for perfect data")

print("\n2. LUT ACCURACY:")
print(f"   - Max RGB error: {df_lut['rgb_rmse'].max():.6f}")
if df_lut['rgb_rmse'].max() > 0.001:
    print("   ⚠️  LUT interpolation may contribute to errors")
else:
    print("   ✓ LUT is accurate")

print("\n3. RGB MONOTONICITY:")
print(f"   - R/G ratio reversals: {non_monotonic}")
if non_monotonic > 0:
    print("   ⚠️  Non-monotonic RGB ratios may cause ambiguity")
else:
    print("   ✓ RGB ratios are monotonic")

print("\n4. OPTIMIZATION STABILITY:")
stdev_converged = np.std(recovered_wavelengths)
print(f"   - Convergence std dev: {stdev_converged:.4f} nm")
if stdev_converged > 0.1:
    print("   ⚠️  Optimization is sensitive to initial guess")
    print("   → May indicate multiple local minima")
else:
    print("   ✓ Optimization is stable")

print("\n5. LIKELY CAUSES OF BIAS:")
rms_perfect = np.sqrt((df_roundtrip['bias']**2).mean())
if rms_perfect > 1.0:
    print("   - PRIMARY: Forward/inverse model mismatch")
    print("   - The inverse model systematically misestimates wavelengths")
    print("   - Possible causes:")
    print("     * PSF width calculation method (1st moment vs weighted)")
    print("     * Normalization of RGB values")
    print("     * Optimization algorithm getting stuck in local minima")
elif df_lut['rgb_rmse'].max() > 0.001:
    print("   - PRIMARY: LUT interpolation errors")
else:
    print("   - Bias appears to be mainly from photon noise")
    print("   - Systematic bias is minimal for perfect data")

print("\n" + "="*80)
