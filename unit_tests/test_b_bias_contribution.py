"""
Test if B channel bias contributes to wavelength bias and check error propagation.

1. Fit wavelength with and without B channel
2. Check if error propagation through normalization is correct
3. Verify if we need Jacobian for the transformation from (A_R, A_G, A_B) -> (R_norm, G_norm, B_norm)
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

# Test 580 nm at 500 photons
wl_true = 580
n_photons = 500
pixel_size = 69  # nm/pixel

# Filter config
filter_names = [
    "semrock-ff01-650-200",
    "semrock-di03-r514-t1-25x36",
    "semrock-ff01-515-lp"
]
NA = 1.49

# Setup optical system
wavelength_array, pixel_QYs, filter_spectra = nrf.setup_optical_system(filter_names)

# Get true values
fwd_true = nrf.nile_red_forward_model(
    wl_true, filter_spectra, wavelength_array, pixel_QYs, NA
)

print("="*80)
print(f"B Channel Bias Contribution to Wavelength Bias: {wl_true} nm, {n_photons} photons")
print("="*80)

# Load data
photon_str = f"{int(n_photons):08d}p0"
pattern = f"wl{wl_true}_*_{photon_str}_*rawresults.parquet"
files = list(data_dir.glob(pattern))
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

print(f"\nNote: A_R, A_G, A_B in parquet are already NORMALIZED fractions")
print(f"Mean values: R={np.mean(A_R):.6f}, G={np.mean(A_G):.6f}, B={np.mean(A_B):.6f}")
print(f"Sum: {np.mean(A_R + A_G + A_B):.6f} (should be 1.0)")

# ERROR PROPAGATION ANALYSIS
print("\n" + "="*80)
print("ERROR PROPAGATION THROUGH NORMALIZATION")
print("="*80)

print("\nCurrent method (from Multicolour_Simulation_Functions.py:848-851):")
print("  total_err = sqrt(R_err² + G_err² + B_err²)")
print("  R_norm_err = R_norm * sqrt((R_err/R)² + (total_err/total)²)")
print("  G_norm_err = G_norm * sqrt((G_err/G)² + (total_err/total)²)")
print("  B_norm_err = B_norm * sqrt((B_err/B)² + (total_err/total)²)")

print("\nBUT WAIT - The parquet file already has NORMALIZED values!")
print("So A_R, A_G, A_B are fractions, not absolute photons")
print("And A_R_err, A_G_err, A_B_err are errors on the FRACTIONS")

# Let's check: are the errors already propagated or raw?
# If they're on normalized values, they should be correlated (R+G+B=1 constraint)

# Check if errors are correlated as expected for normalized values
sample_size = 1000
sample_indices = np.random.choice(len(A_R), sample_size, replace=False)

print(f"\nChecking error structure (sample of {sample_size}):")
print(f"  Mean A_R_err: {np.mean(A_R_err[sample_indices]):.6f}")
print(f"  Mean A_G_err: {np.mean(A_G_err[sample_indices]):.6f}")
print(f"  Mean A_B_err: {np.mean(A_B_err[sample_indices]):.6f}")

# Check if sum of errors follows propagation rules
# For independent errors: σ_sum² = σ_R² + σ_G² + σ_B²
# But for normalized values with constraint R+G+B=1: sum is correlated!

print("\n" + "="*80)
print("CORRECT ERROR PROPAGATION FORMULA")
print("="*80)

print("\nFor transformation: (A_R, A_G, A_B) → R_norm = A_R/(A_R+A_G+A_B)")
print("\nJacobian derivatives:")
print("  ∂R_norm/∂A_R = (A_G + A_B) / total²")
print("  ∂R_norm/∂A_G = -A_R / total²")
print("  ∂R_norm/∂A_B = -A_R / total²")
print("\nGeneralized error propagation:")
print("  σ²(R_norm) = (∂R/∂A_R)²σ²(A_R) + (∂R/∂A_G)²σ²(A_G) + (∂R/∂A_B)²σ²(A_B)")
print("             + 2(∂R/∂A_R)(∂R/∂A_G)Cov(A_R,A_G) + ...")

print("\nIF A_R, A_G, A_B are INDEPENDENT (no covariance):")
print("  σ²(R_norm) = [(A_G+A_B)²σ²(A_R) + A_R²σ²(A_G) + A_R²σ²(A_B)] / total⁴")

# TEST: Fit wavelength WITH and WITHOUT B channel
print("\n" + "="*80)
print("TEST: Wavelength Fitting With vs Without B Channel")
print("="*80)

# Sample for testing
n_test = 100
test_indices = np.random.choice(len(A_R), n_test, replace=False)

wl_with_B = []
wl_without_B = []

for idx in test_indices:
    rgb_with_B = np.array([A_R[idx], A_G[idx], A_B[idx]])

    # Method 1: Use R, G, B (current method)
    rgb_err = np.array([A_R_err[idx], A_G_err[idx], A_B_err[idx]])
    wl1, _ = nrf.fit_nile_red_wavelength(
        rgb_with_B, sigma_x_nm[idx], sigma_y_nm[idx],
        rgb_err, sigma_x_err_nm[idx], sigma_y_err_nm[idx],
        filter_spectra, wavelength_array, pixel_QYs, NA
    )
    wl_with_B.append(wl1)

    # Method 2: Use only R and G ratios, ignore B
    # Renormalize: R/(R+G), G/(R+G), B=0
    R_renorm = A_R[idx] / (A_R[idx] + A_G[idx])
    G_renorm = A_G[idx] / (A_R[idx] + A_G[idx])
    rgb_no_B = np.array([R_renorm, G_renorm, 0.0])

    # Propagate errors for renormalization
    total_RG = A_R[idx] + A_G[idx]
    # ∂(R/(R+G))/∂R = G/(R+G)², ∂(R/(R+G))/∂G = -R/(R+G)²
    R_renorm_err = np.sqrt(
        (A_G[idx]/total_RG**2 * A_R_err[idx])**2 +
        (A_R[idx]/total_RG**2 * A_G_err[idx])**2
    )
    G_renorm_err = np.sqrt(
        (A_R[idx]/total_RG**2 * A_G_err[idx])**2 +
        (A_G[idx]/total_RG**2 * A_R_err[idx])**2
    )
    rgb_err_no_B = np.array([R_renorm_err, G_renorm_err, 1e-6])

    wl2, _ = nrf.fit_nile_red_wavelength(
        rgb_no_B, sigma_x_nm[idx], sigma_y_nm[idx],
        rgb_err_no_B, sigma_x_err_nm[idx], sigma_y_err_nm[idx],
        filter_spectra, wavelength_array, pixel_QYs, NA
    )
    wl_without_B.append(wl2)

wl_with_B = np.array(wl_with_B)
wl_without_B = np.array(wl_without_B)

# Remove any NaN values
valid_wl = ~(np.isnan(wl_with_B) | np.isnan(wl_without_B))
wl_with_B = wl_with_B[valid_wl]
wl_without_B = wl_without_B[valid_wl]

bias_with_B = np.mean(wl_with_B) - wl_true
bias_without_B = np.mean(wl_without_B) - wl_true

print(f"\nResults (n={len(wl_with_B)} valid fits):")
print(f"\nWith B channel (R, G, B):")
print(f"  Mean wavelength: {np.mean(wl_with_B):.2f} nm")
print(f"  Bias: {bias_with_B:+.2f} nm")
print(f"  Std: {np.std(wl_with_B):.2f} nm")

print(f"\nWithout B channel (R/(R+G), G/(R+G), 0):")
print(f"  Mean wavelength: {np.mean(wl_without_B):.2f} nm")
print(f"  Bias: {bias_without_B:+.2f} nm")
print(f"  Std: {np.std(wl_without_B):.2f} nm")

print(f"\nDifference in bias: {bias_with_B - bias_without_B:+.2f} nm")

if abs(bias_with_B - bias_without_B) > 5:
    print("  ⚠️  B channel bias significantly affects wavelength!")
else:
    print("  ✓ B channel bias has minimal effect on wavelength")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("\n1. B channel has 104% bias but very small true fraction (2.7%)")
print("2. Need to check if B bias propagates to wavelength significantly")
print("3. Current error propagation may not properly account for normalization constraint")
print("4. Should verify if A_R_err, A_G_err, A_B_err already include normalization")
