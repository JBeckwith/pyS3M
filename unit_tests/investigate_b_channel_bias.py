"""
Investigate why B channel is massively overestimated (104% bias at 580 nm).

Check:
1. Actual photon counts in each channel
2. Background levels
3. Signal-to-noise ratio
4. Background subtraction accuracy
"""

import sys
sys.path.insert(0, '../src')

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from NileRedFunctions import NileRed_Functions

# Initialize
nrf = NileRed_Functions()

# Data path
data_dir = Path("/home/jbeckwith/Documents/pCloud/Chemistry/Lee/Data/Simulation/20251007_NileRedModelTesting")

# Test 580 nm at 500 photons (worst case for B)
wl_true = 580
n_photons = 500

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

# Get true values
fwd_true = nrf.nile_red_forward_model(
    wl_true, filter_spectra, wavelength_array, pixel_QYs, NA
)

R_true = fwd_true['R']
G_true = fwd_true['G']
B_true = fwd_true['B']

# Expected photons per channel
R_photons_expected = n_photons * R_true
G_photons_expected = n_photons * G_true
B_photons_expected = n_photons * B_true

print("="*80)
print(f"B Channel Bias Investigation: {wl_true} nm, {n_photons} photons")
print("="*80)

print(f"\nExpected photon distribution:")
print(f"  Total photons: {n_photons}")
print(f"  R fraction: {R_true:.6f} → {R_photons_expected:.1f} photons")
print(f"  G fraction: {G_true:.6f} → {G_photons_expected:.1f} photons")
print(f"  B fraction: {B_true:.6f} → {B_photons_expected:.1f} photons")
print(f"  B is {B_photons_expected:.1f} photons - very weak signal!")

# Find file
photon_str = f"{int(n_photons):08d}p0"
pattern = f"wl{wl_true}_*_{photon_str}_*rawresults.parquet"
files = list(data_dir.glob(pattern))

if len(files) == 0:
    print(f"No files found!")
    sys.exit(1)

file_path = files[0]
print(f"\nLoading: {file_path.name}")

# Load data
df = pl.read_parquet(file_path)

# Get fitted values
A_R = df['A_R'].to_numpy()
A_G = df['A_G'].to_numpy()
A_B = df['A_B'].to_numpy()
bg_R = df['bg_R'].to_numpy()
bg_G = df['bg_G'].to_numpy()
bg_B = df['bg_B'].to_numpy()
photons_total = df['photons'].to_numpy()
chi_sqr = df['chi_sqr'].to_numpy()

# Remove NaNs
valid = ~(np.isnan(A_R) | np.isnan(A_G) | np.isnan(A_B) | np.isnan(photons_total))
A_R = A_R[valid]
A_G = A_G[valid]
A_B = A_B[valid]
bg_R = bg_R[valid]
bg_G = bg_G[valid]
bg_B = bg_B[valid]
photons_total = photons_total[valid]
chi_sqr = chi_sqr[valid]

print(f"\nValid localizations: {len(A_R)}")

# Note: A_R, A_G, A_B in the parquet file are already NORMALIZED fractions
# photons_total contains the absolute photon count
# So absolute photons in each channel:
R_photons_fit = A_R * photons_total
G_photons_fit = A_G * photons_total
B_photons_fit = A_B * photons_total

print(f"\nFitted photon counts (mean ± std):")
print(f"  Total: {np.mean(photons_total):.1f} ± {np.std(photons_total):.1f}")
print(f"  R: {np.mean(R_photons_fit):.1f} ± {np.std(R_photons_fit):.1f}  (expected: {R_photons_expected:.1f}, bias: {np.mean(R_photons_fit) - R_photons_expected:+.1f})")
print(f"  G: {np.mean(G_photons_fit):.1f} ± {np.std(G_photons_fit):.1f}  (expected: {G_photons_expected:.1f}, bias: {np.mean(G_photons_fit) - G_photons_expected:+.1f})")
print(f"  B: {np.mean(B_photons_fit):.1f} ± {np.std(B_photons_fit):.1f}  (expected: {B_photons_expected:.1f}, bias: {np.mean(B_photons_fit) - B_photons_expected:+.1f})")

print(f"\nBackground fractions (normalized, mean ± std):")
print(f"  bg_R: {np.mean(bg_R):.6f} ± {np.std(bg_R):.6f}")
print(f"  bg_G: {np.mean(bg_G):.6f} ± {np.std(bg_G):.6f}")
print(f"  bg_B: {np.mean(bg_B):.6f} ± {np.std(bg_B):.6f}")
print(f"  Expected (uniform background): 0.333333 for each channel")

# Signal-to-noise analysis
# Background is 40 photons/pixel split across ~9 pixels (3x3 PSF region)
# But it's color-dependent
background_per_channel = 40.0 / 3.0  # ~13.3 photons per channel
print(f"\nSignal-to-Background Ratio (SBR):")
print(f"  R: {R_photons_expected / background_per_channel:.2f}")
print(f"  G: {G_photons_expected / background_per_channel:.2f}")
print(f"  B: {B_photons_expected / background_per_channel:.2f}")
print(f"  → B has very poor SBR! Background is comparable to signal!")

# Check correlation between B overestimation and total photons
print(f"\nCorrelation analysis:")
print(f"  Corr(B_fraction, total_photons): {np.corrcoef(A_B, photons_total)[0,1]:.3f}")
print(f"  Corr(R_fraction, total_photons): {np.corrcoef(A_R, photons_total)[0,1]:.3f}")
print(f"  Corr(B_fraction, chi_sqr): {np.corrcoef(A_B, chi_sqr)[0,1]:.3f}")

# Look at distribution
print(f"\nB channel distribution:")
print(f"  Median: {np.median(A_B):.6f}")
print(f"  25th percentile: {np.percentile(A_B, 25):.6f}")
print(f"  75th percentile: {np.percentile(A_B, 75):.6f}")
print(f"  Min: {np.min(A_B):.6f}")
print(f"  Max: {np.max(A_B):.6f}")
print(f"  True value: {B_true:.6f}")

# Check if there's a floor effect
n_below_zero = np.sum(A_B < 0)
n_below_true = np.sum(A_B < B_true)
print(f"\n  Localizations with B < 0: {n_below_zero} ({100*n_below_zero/len(A_B):.1f}%)")
print(f"  Localizations with B < true: {n_below_true} ({100*n_below_true/len(A_B):.1f}%)")

# Hypothesis: Background subtraction error
# If background in B is slightly overestimated, signal photons get counted as background
# If background in B is slightly underestimated, background gets counted as signal
print(f"\n" + "="*80)
print("HYPOTHESIS")
print("="*80)
print("\nB channel has very low signal (~14 photons) vs background (~13 photons).")
print("SBR ≈ 1, making B extremely sensitive to background subtraction errors.")
print("\nPossible causes:")
print("1. Background estimation includes some signal photons")
print("2. Fitting algorithm compensates for poor B SNR by inflating B fraction")
print("3. Correlation between R/G/B fractions (must sum to 1) redistributes error")
print("4. Lower bound on B fraction prevents negative values, biasing upward")

# Create histogram
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(A_B, bins=50, alpha=0.7, edgecolor='black')
axes[0].axvline(B_true, color='r', linestyle='--', linewidth=2, label=f'True: {B_true:.6f}')
axes[0].axvline(np.mean(A_B), color='g', linestyle='--', linewidth=2, label=f'Mean: {np.mean(A_B):.6f}')
axes[0].set_xlabel('B Fraction')
axes[0].set_ylabel('Count')
axes[0].set_title(f'B Channel Distribution\n{wl_true} nm, {n_photons} photons')
axes[0].legend()
axes[0].grid(alpha=0.3)

# R vs B scatter
axes[1].scatter(A_R, A_B, alpha=0.1, s=1)
axes[1].axhline(B_true, color='r', linestyle='--', alpha=0.5)
axes[1].axvline(R_true, color='r', linestyle='--', alpha=0.5)
axes[1].set_xlabel('R Fraction')
axes[1].set_ylabel('B Fraction')
axes[1].set_title('R vs B Correlation')
axes[1].grid(alpha=0.3)

# G vs B scatter
axes[2].scatter(A_G, A_B, alpha=0.1, s=1)
axes[2].axhline(B_true, color='r', linestyle='--', alpha=0.5)
axes[2].axvline(G_true, color='r', linestyle='--', alpha=0.5)
axes[2].set_xlabel('G Fraction')
axes[2].set_ylabel('B Fraction')
axes[2].set_title('G vs B Correlation')
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('b_channel_bias_investigation.png', dpi=150)
print(f"\nSaved plot: b_channel_bias_investigation.png")
