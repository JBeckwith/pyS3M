"""
Investigate improved error estimation for low photon counts.

Key insight: We now KNOW that B is systematically biased at low SNR.
Can we use this knowledge to improve error estimates?

Approaches:
1. Inflate B error based on SNR (empirical correction)
2. Use bootstrap/Monte Carlo to estimate true uncertainty
3. Cramér-Rao lower bound (CRLB) for theoretical minimum
4. Asymmetric errors (bias + variance)
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

# Test 580 nm at 500 photons (worst case for B)
wl_true = 580
n_photons = 500
pixel_size = 69

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
print("Improved Error Estimation for Low Photon Counts")
print("="*80)

print(f"\nTest case: {wl_true} nm, {n_photons} photons")
print(f"True B fraction: {fwd_true['B']:.6f} (very low!)")

# Load data
photon_str = f"{int(n_photons):08d}p0"
pattern = f"wl{wl_true}_*_{photon_str}_*rawresults.parquet"
files = list(data_dir.glob(pattern))
df = pl.read_parquet(files[0])

# Get data
A_R = df['A_R'].to_numpy()
A_G = df['A_G'].to_numpy()
A_B = df['A_B'].to_numpy()
A_R_err = df['A_R_err'].to_numpy()
A_G_err = df['A_G_err'].to_numpy()
A_B_err = df['A_B_err'].to_numpy()

# Remove NaNs
valid = ~(np.isnan(A_R) | np.isnan(A_G) | np.isnan(A_B))
A_R = A_R[valid]
A_G = A_G[valid]
A_B = A_B[valid]
A_R_err = A_R_err[valid]
A_G_err = A_G_err[valid]
A_B_err = A_B_err[valid]

print(f"\nCurrent error estimates (mean ± std):")
print(f"  σ_R: {np.mean(A_R_err):.6f} ± {np.std(A_R_err):.6f}")
print(f"  σ_G: {np.mean(A_G_err):.6f} ± {np.std(A_G_err):.6f}")
print(f"  σ_B: {np.mean(A_B_err):.6f} ± {np.std(A_B_err):.6f}")

# APPROACH 1: Check if current errors capture the actual variance
print(f"\n" + "="*80)
print("APPROACH 1: Compare Fit Errors to Empirical Variance")
print("="*80)

empirical_std_R = np.std(A_R - fwd_true['R'])
empirical_std_G = np.std(A_G - fwd_true['G'])
empirical_std_B = np.std(A_B - fwd_true['B'])

mean_fit_err_R = np.mean(A_R_err)
mean_fit_err_G = np.mean(A_G_err)
mean_fit_err_B = np.mean(A_B_err)

print(f"\nR channel:")
print(f"  Fit error (mean): {mean_fit_err_R:.6f}")
print(f"  Empirical std: {empirical_std_R:.6f}")
print(f"  Ratio (empirical/fit): {empirical_std_R / mean_fit_err_R:.2f}x")

print(f"\nG channel:")
print(f"  Fit error (mean): {mean_fit_err_G:.6f}")
print(f"  Empirical std: {empirical_std_G:.6f}")
print(f"  Ratio (empirical/fit): {empirical_std_G / mean_fit_err_G:.2f}x")

print(f"\nB channel:")
print(f"  Fit error (mean): {mean_fit_err_B:.6f}")
print(f"  Empirical std: {empirical_std_B:.6f}")
print(f"  Ratio (empirical/fit): {empirical_std_B / mean_fit_err_B:.2f}x")

if empirical_std_B / mean_fit_err_B > 1.5:
    print(f"  ⚠️  B errors are UNDERESTIMATED by {empirical_std_B / mean_fit_err_B:.1f}x")

# APPROACH 2: SNR-based error inflation
print(f"\n" + "="*80)
print("APPROACH 2: SNR-Based Error Inflation")
print("="*80)

# Estimate background from simulation parameters
background_per_channel = 40.0 / 3.0  # ~13.3 photons/channel

# Calculate SNR for each channel
R_signal = n_photons * fwd_true['R']
G_signal = n_photons * fwd_true['G']
B_signal = n_photons * fwd_true['B']

SNR_R = R_signal / np.sqrt(R_signal + background_per_channel)
SNR_G = G_signal / np.sqrt(G_signal + background_per_channel)
SNR_B = B_signal / np.sqrt(B_signal + background_per_channel)

print(f"\nSignal-to-Noise Ratios:")
print(f"  SNR_R: {SNR_R:.2f}")
print(f"  SNR_G: {SNR_G:.2f}")
print(f"  SNR_B: {SNR_B:.2f}")

# Empirical relationship: error inflation factor vs SNR
# From our data: low SNR channels have underestimated errors
def error_inflation_factor(snr):
    """
    Empirical error inflation factor based on SNR.
    At SNR ~ 1, errors are underestimated by ~3x
    At SNR > 10, errors are accurate
    """
    if snr < 2:
        return 3.0
    elif snr < 5:
        return 2.0
    elif snr < 10:
        return 1.5
    else:
        return 1.0

inflation_R = error_inflation_factor(SNR_R)
inflation_G = error_inflation_factor(SNR_G)
inflation_B = error_inflation_factor(SNR_B)

print(f"\nProposed error inflation factors:")
print(f"  R: {inflation_R:.1f}x (SNR={SNR_R:.1f})")
print(f"  G: {inflation_G:.1f}x (SNR={SNR_G:.1f})")
print(f"  B: {inflation_B:.1f}x (SNR={SNR_B:.1f})")

corrected_err_R = mean_fit_err_R * inflation_R
corrected_err_G = mean_fit_err_G * inflation_G
corrected_err_B = mean_fit_err_B * inflation_B

print(f"\nCorrected errors:")
print(f"  σ_R: {mean_fit_err_R:.6f} → {corrected_err_R:.6f}")
print(f"  σ_G: {mean_fit_err_G:.6f} → {corrected_err_G:.6f}")
print(f"  σ_B: {mean_fit_err_B:.6f} → {corrected_err_B:.6f}")

print(f"\nComparison to empirical std:")
print(f"  R: corrected/empirical = {corrected_err_R / empirical_std_R:.2f}")
print(f"  G: corrected/empirical = {corrected_err_G / empirical_std_G:.2f}")
print(f"  B: corrected/empirical = {corrected_err_B / empirical_std_B:.2f}")

# APPROACH 3: Account for systematic bias separately
print(f"\n" + "="*80)
print("APPROACH 3: Separate Systematic Bias from Random Error")
print("="*80)

bias_R = np.mean(A_R) - fwd_true['R']
bias_G = np.mean(A_G) - fwd_true['G']
bias_B = np.mean(A_B) - fwd_true['B']

print(f"\nSystematic biases:")
print(f"  Bias_R: {bias_R:+.6f}")
print(f"  Bias_G: {bias_G:+.6f}")
print(f"  Bias_B: {bias_B:+.6f}")

# Total uncertainty = sqrt(bias² + variance²)
total_uncertainty_R = np.sqrt(bias_R**2 + empirical_std_R**2)
total_uncertainty_G = np.sqrt(bias_G**2 + empirical_std_G**2)
total_uncertainty_B = np.sqrt(bias_B**2 + empirical_std_B**2)

print(f"\nTotal uncertainty (bias + variance):")
print(f"  R: {total_uncertainty_R:.6f}")
print(f"  G: {total_uncertainty_G:.6f}")
print(f"  B: {total_uncertainty_B:.6f}")

# APPROACH 4: Asymmetric errors for B
print(f"\n" + "="*80)
print("APPROACH 4: Asymmetric Errors for Biased Channels")
print("="*80)

# B has asymmetric distribution (floor at 0, long tail above)
percentile_25 = np.percentile(A_B, 25)
percentile_50 = np.percentile(A_B, 50)
percentile_75 = np.percentile(A_B, 75)

error_minus_B = percentile_50 - percentile_25
error_plus_B = percentile_75 - percentile_50

print(f"\nB channel distribution:")
print(f"  25th percentile: {percentile_25:.6f}")
print(f"  50th percentile (median): {percentile_50:.6f}")
print(f"  75th percentile: {percentile_75:.6f}")
print(f"  True value: {fwd_true['B']:.6f}")

print(f"\nAsymmetric errors for B:")
print(f"  B = {percentile_50:.6f} +{error_plus_B:.6f} -{error_minus_B:.6f}")
print(f"  (vs symmetric σ_B = {mean_fit_err_B:.6f})")

print(f"\n" + "="*80)
print("SUMMARY & RECOMMENDATIONS")
print("="*80)

print("""
1. Current fit errors UNDERESTIMATE true uncertainty for low-SNR channels
   - B errors are underestimated by ~3x when SNR ≈ 1

2. SNR-based error inflation:
   - Multiply fit errors by inflation factor based on SNR
   - SNR < 2: inflate by 3x
   - SNR 2-5: inflate by 2x
   - SNR 5-10: inflate by 1.5x
   - SNR > 10: use as-is

3. Separate systematic bias from random errors:
   - Report bias (known from calibration/simulation)
   - Report variance (from fit errors, possibly inflated)
   - Total uncertainty = sqrt(bias² + variance²)

4. For B channel at low SNR:
   - Consider excluding from fits (already implemented ✓)
   - If included, use asymmetric errors or inflate by 3x
   - Account for known positive bias

IMPLEMENTATION:
- Add SNR calculation to fitting code
- Apply error inflation based on SNR
- Optionally report separate bias and variance
""")
