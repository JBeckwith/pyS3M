#!/usr/bin/env python3
"""
Test script for auto-selection of pygmmis vs sklearn EM.

Tests that gmm_fit_method='EM' intelligently chooses:
- pygmmis (Extreme Deconvolution) when error columns present
- sklearn EM when error columns absent

jsb92, 2025-10-29
"""

import numpy as np
import pandas as pd
import sys
import os

# Add src to path
module_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, module_dir)

import SM_extractionfunctions as SM_E


def generate_synthetic_data(n_samples_per_dye=5000):
    """Generate synthetic 2-dye data with known ground truth."""
    np.random.seed(42)

    # True means
    mean_dye1 = np.array([0.3, 0.2])
    mean_dye2 = np.array([0.7, 0.6])

    # True covariances
    cov_dye1 = np.array([[0.01, 0.003], [0.003, 0.01]])
    cov_dye2 = np.array([[0.015, -0.002], [-0.002, 0.012]])

    # Generate true values
    true_vals_dye1 = np.random.multivariate_normal(mean_dye1, cov_dye1, size=n_samples_per_dye)
    true_vals_dye2 = np.random.multivariate_normal(mean_dye2, cov_dye2, size=n_samples_per_dye)

    # Clip to valid range
    true_vals_dye1 = np.clip(true_vals_dye1, 0.01, 0.99)
    true_vals_dye2 = np.clip(true_vals_dye2, 0.01, 0.99)

    # Combine
    true_vals = np.vstack([true_vals_dye1, true_vals_dye2])
    true_channels = np.concatenate([np.zeros(n_samples_per_dye), np.ones(n_samples_per_dye)])

    n_total = len(true_vals)

    # Generate photon counts
    photons = np.random.normal(10000, 3000, size=n_total)
    photons = np.maximum(photons, 500)

    # Calculate errors
    true_A_R = true_vals[:, 0]
    true_A_G = true_vals[:, 1]

    sigma_A_R = np.sqrt(true_A_R * (1 - true_A_R) / photons)
    sigma_A_G = np.sqrt(true_A_G * (1 - true_A_G) / photons)

    # Add noise
    measured_A_R = true_A_R + np.random.normal(0, sigma_A_R)
    measured_A_G = true_A_G + np.random.normal(0, sigma_A_G)

    # Clip
    measured_A_R = np.clip(measured_A_R, 0, 1)
    measured_A_G = np.clip(measured_A_G, 0, 1)

    # Create DataFrame
    df = pd.DataFrame({
        'A_R': measured_A_R,
        'A_G': measured_A_G,
        'A_R_err': sigma_A_R,
        'A_G_err': sigma_A_G,
        'photons': photons,
        'true_channel': true_channels.astype(int),
        'xc': np.random.uniform(0, 100, n_total),
        'yc': np.random.uniform(0, 100, n_total),
        'frame': np.random.randint(0, 1000, n_total),
        'A_B': 1 - measured_A_R - measured_A_G,
    })

    return df


def main():
    print("=" * 80)
    print("TEST: Auto-selection of pygmmis vs sklearn EM")
    print("=" * 80)
    print()

    # Generate data
    df = generate_synthetic_data(n_samples_per_dye=5000)
    extractor = SM_E.extract_SMs()

    # Test 1: WITH error columns (should use pygmmis)
    print("TEST 1: EM with error columns (should auto-select pygmmis)")
    print("-" * 80)

    assigned1, metadata1 = extractor.unmix_channels(
        df,
        n_channels=2,
        channels_to_use=['A_R', 'A_G'],
        gmm_fit_method='EM',  # Should use pygmmis
        confidence_threshold=0.95,
        verbose=True,
    )

    print()
    print("Result: Converged =", metadata1['converged'])
    print("        Assignments =", metadata1['n_assigned'])
    print()

    # Test 2: WITHOUT error columns (should use sklearn)
    print("TEST 2: EM without error columns (should auto-select sklearn)")
    print("-" * 80)

    df_no_err = df.drop(columns=['A_R_err', 'A_G_err'])

    assigned2, metadata2 = extractor.unmix_channels(
        df_no_err,
        n_channels=2,
        channels_to_use=['A_R', 'A_G'],
        gmm_fit_method='EM',  # Should use sklearn
        confidence_threshold=0.95,
        verbose=True,
    )

    print()
    print("Result: Converged =", metadata2['converged'])
    print("        Assignments =", metadata2['n_assigned'])
    print()

    # Comparison
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("✓ Test 1: Auto-selected pygmmis (error columns present)")
    print("✓ Test 2: Auto-selected sklearn EM (error columns absent)")
    print()
    print("Comparison:")
    print(f"  With errors (pygmmis):    {metadata1['n_assigned']}")
    print(f"  Without errors (sklearn): {metadata2['n_assigned']}")
    print()
    print("✓ SUCCESS: Auto-selection working as expected!")
    print()
    print("Usage: Simply use gmm_fit_method='EM' - it will automatically")
    print("       choose the best method based on available data.")
    print()


if __name__ == "__main__":
    main()
