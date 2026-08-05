#!/usr/bin/env python3
"""
Test script comparing pygmmis Extreme Deconvolution with current replication method.

This script generates synthetic 2-dye SMLM data with known ground truth and per-point
measurement uncertainties, then compares:
1. sklearn EM with point replication (current method)
2. pygmmis Extreme Deconvolution (new method)

Expected outcome: Both should recover similar means/covariances, but pygmmis should be
faster and more theoretically sound.

jsb92, 2025-10-29
"""

import numpy as np
import pandas as pd
import sys
import os
import time

# Add src to path
module_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pyS3M.SM_extractionfunctions as SM_E


def generate_synthetic_2dye_data(
    n_samples_per_dye=5000,
    mean_dye1=(0.3, 0.2),
    mean_dye2=(0.7, 0.6),
    cov_dye1=None,
    cov_dye2=None,
    photon_mean=10000,
    photon_std=3000,
    random_state=42,
):
    """
    Generate synthetic 2-dye SMLM data with known ground truth.

    Creates data with:
    - Two dye populations with specified means and covariances
    - Per-point photon counts (Gaussian distributed)
    - Per-point measurement uncertainties based on photon statistics
    - Added Gaussian noise based on uncertainties

    Args:
        n_samples_per_dye: Number of localizations per dye
        mean_dye1, mean_dye2: True (A_R, A_G) means for each dye
        cov_dye1, cov_dye2: True covariance matrices (if None, uses defaults)
        photon_mean: Mean photon count per localization
        photon_std: Std dev of photon counts
        random_state: Random seed for reproducibility

    Returns:
        df: DataFrame with columns:
            - A_R, A_G: Normalized RGB ratios (with noise)
            - A_R_err, A_G_err: Per-point measurement uncertainties
            - photons: Total photon count per localization
            - true_channel: Ground truth channel assignment (0 or 1)
            - true_A_R, true_A_G: Noise-free values
    """
    np.random.seed(random_state)

    # Default covariances (if not provided)
    if cov_dye1 is None:
        cov_dye1 = np.array([[0.01, 0.003], [0.003, 0.01]])  # Slight correlation
    if cov_dye2 is None:
        cov_dye2 = np.array([[0.015, -0.002], [-0.002, 0.012]])  # Slight negative correlation

    # Generate true (noise-free) values for dye 1
    true_vals_dye1 = np.random.multivariate_normal(mean_dye1, cov_dye1, size=n_samples_per_dye)
    # Clip to valid range [0, 1]
    true_vals_dye1 = np.clip(true_vals_dye1, 0.01, 0.99)

    # Generate true (noise-free) values for dye 2
    true_vals_dye2 = np.random.multivariate_normal(mean_dye2, cov_dye2, size=n_samples_per_dye)
    # Clip to valid range [0, 1]
    true_vals_dye2 = np.clip(true_vals_dye2, 0.01, 0.99)

    # Combine both dyes
    true_vals = np.vstack([true_vals_dye1, true_vals_dye2])
    true_channels = np.concatenate([np.zeros(n_samples_per_dye), np.ones(n_samples_per_dye)])

    n_total = len(true_vals)

    # Generate photon counts (Gaussian distributed, clipped to positive)
    photons = np.random.normal(photon_mean, photon_std, size=n_total)
    photons = np.maximum(photons, 500)  # Minimum 500 photons

    # Calculate per-point measurement uncertainties based on photon statistics
    # For normalized ratios: sigma_ratio ≈ sqrt(ratio * (1 - ratio) / N_photons)
    true_A_R = true_vals[:, 0]
    true_A_G = true_vals[:, 1]

    # Binomial statistics for ratio errors
    sigma_A_R = np.sqrt(true_A_R * (1 - true_A_R) / photons)
    sigma_A_G = np.sqrt(true_A_G * (1 - true_A_G) / photons)

    # Add measurement noise (Gaussian) based on uncertainties
    measured_A_R = true_A_R + np.random.normal(0, sigma_A_R)
    measured_A_G = true_A_G + np.random.normal(0, sigma_A_G)

    # Clip to valid range [0, 1]
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
        'true_A_R': true_A_R,
        'true_A_G': true_A_G,
        # Add dummy columns for compatibility
        'xc': np.random.uniform(0, 100, n_total),
        'yc': np.random.uniform(0, 100, n_total),
        'frame': np.random.randint(0, 1000, n_total),
        'A_B': 1 - measured_A_R - measured_A_G,  # Normalize to sum=1
    })

    return df


def compare_fitting_methods(df, verbose=True):
    """
    Compare sklearn EM (with replication) vs pygmmis Extreme Deconvolution.

    Args:
        df: DataFrame from generate_synthetic_2dye_data()
        verbose: Print detailed comparison

    Returns:
        results: Dict with comparison metrics
    """
    extractor = SM_E.extract_SMs()

    if verbose:
        print("=" * 80)
        print("COMPARISON: sklearn EM (replication) vs pygmmis Extreme Deconvolution")
        print("=" * 80)
        print(f"Dataset: {len(df)} localizations ({len(df[df.true_channel==0])} dye1, "
              f"{len(df[df.true_channel==1])} dye2)")
        print(f"Mean photons: {df.photons.mean():.0f} ± {df.photons.std():.0f}")
        print(f"Mean errors: A_R={df.A_R_err.mean():.4f}, A_G={df.A_G_err.mean():.4f}")
        print()

    results = {}

    # Ground truth (for comparison)
    true_means = np.array([
        [df[df.true_channel == 0]['true_A_R'].mean(), df[df.true_channel == 0]['true_A_G'].mean()],
        [df[df.true_channel == 1]['true_A_R'].mean(), df[df.true_channel == 1]['true_A_G'].mean()],
    ])

    if verbose:
        print("Ground Truth Means:")
        print(f"  Dye 1: A_R={true_means[0, 0]:.4f}, A_G={true_means[0, 1]:.4f}")
        print(f"  Dye 2: A_R={true_means[1, 0]:.4f}, A_G={true_means[1, 1]:.4f}")
        print()

    # Test 1: EM with error columns (should auto-select pygmmis)
    print("-" * 80)
    print("Method 1: EM with error columns (auto → pygmmis)")
    print("-" * 80)

    start = time.time()
    assigned_em, metadata_em = extractor.unmix_channels(
        df,
        n_channels=2,
        channels_to_use=['A_R', 'A_G'],
        gmm_fit_method='EM',  # Should auto-select pygmmis since error columns present
        confidence_threshold=0.95,
        verbose=False,
    )
    time_em = time.time() - start

    means_em = metadata_em['means']
    cov_em = metadata_em['covariances']
    converged_em = metadata_em['converged']

    # Calculate accuracy against ground truth
    # Need to account for permutation - channels may be swapped
    assigned_mask_em = assigned_em['channel'] >= 0
    n_correct_em_direct = np.sum((assigned_em.loc[assigned_mask_em, 'channel'] ==
                                   assigned_em.loc[assigned_mask_em, 'true_channel']))
    n_correct_em_swapped = np.sum((assigned_em.loc[assigned_mask_em, 'channel'] ==
                                    (1 - assigned_em.loc[assigned_mask_em, 'true_channel'])))
    # Use the better permutation
    n_correct_em = max(n_correct_em_direct, n_correct_em_swapped)
    accuracy_em = n_correct_em / len(assigned_em[assigned_mask_em]) if assigned_mask_em.sum() > 0 else 0

    # Mean error (distance from ground truth)
    # Need to match recovered channels to true channels (may be permuted)
    perm = [0, 1] if np.linalg.norm(means_em[0] - true_means[0]) < np.linalg.norm(means_em[0] - true_means[1]) else [1, 0]
    mean_error_em = np.linalg.norm(means_em[perm] - true_means, axis=1).mean()

    results['EM'] = {
        'means': means_em[perm],
        'covariances': cov_em[perm],
        'converged': converged_em,
        'time': time_em,
        'accuracy': accuracy_em,
        'mean_error': mean_error_em,
        'n_assigned': metadata_em['n_assigned'],
    }

    if verbose:
        print(f"  Time: {time_em:.3f} s")
        print(f"  Converged: {converged_em}")
        print(f"  Fitted means:")
        print(f"    Dye 1: A_R={means_em[perm[0], 0]:.4f}, A_G={means_em[perm[0], 1]:.4f}")
        print(f"    Dye 2: A_R={means_em[perm[1], 0]:.4f}, A_G={means_em[perm[1], 1]:.4f}")
        print(f"  Mean error from ground truth: {mean_error_em:.5f}")
        print(f"  Accuracy: {accuracy_em*100:.2f}% ({n_correct_em}/{len(assigned_em[assigned_em['channel']>=0])})")
        print(f"  Assignments: {metadata_em['n_assigned']}")
        print()

    # Test 2: EM without error columns (should use sklearn EM)
    print("-" * 80)
    print("Method 2: EM without error columns (auto → sklearn)")
    print("-" * 80)

    # Remove error columns
    df_no_err = df.drop(columns=['A_R_err', 'A_G_err'])

    start = time.time()
    assigned_sklearn, metadata_sklearn = extractor.unmix_channels(
        df_no_err,
        n_channels=2,
        channels_to_use=['A_R', 'A_G'],
        gmm_fit_method='EM',  # Should auto-select sklearn since NO error columns
        confidence_threshold=0.95,
        verbose=False,
    )
    time_sklearn = time.time() - start

    means_pygmmis = metadata_pygmmis['means']
    cov_pygmmis = metadata_pygmmis['covariances']
    converged_pygmmis = metadata_pygmmis['converged']

    # Calculate accuracy
    # Need to account for permutation - channels may be swapped
    assigned_mask_pg = assigned_pygmmis['channel'] >= 0
    n_correct_pg_direct = np.sum((assigned_pygmmis.loc[assigned_mask_pg, 'channel'] ==
                                   assigned_pygmmis.loc[assigned_mask_pg, 'true_channel']))
    n_correct_pg_swapped = np.sum((assigned_pygmmis.loc[assigned_mask_pg, 'channel'] ==
                                    (1 - assigned_pygmmis.loc[assigned_mask_pg, 'true_channel'])))
    # Use the better permutation
    n_correct_pygmmis = max(n_correct_pg_direct, n_correct_pg_swapped)
    accuracy_pygmmis = n_correct_pygmmis / len(assigned_pygmmis[assigned_mask_pg]) if assigned_mask_pg.sum() > 0 else 0

    # Mean error (need to match permutation)
    perm_pg = [0, 1] if np.linalg.norm(means_pygmmis[0] - true_means[0]) < np.linalg.norm(means_pygmmis[0] - true_means[1]) else [1, 0]
    mean_error_pygmmis = np.linalg.norm(means_pygmmis[perm_pg] - true_means, axis=1).mean()

    results['pygmmis'] = {
        'means': means_pygmmis[perm_pg],
        'covariances': cov_pygmmis[perm_pg],
        'converged': converged_pygmmis,
        'time': time_pygmmis,
        'accuracy': accuracy_pygmmis,
        'mean_error': mean_error_pygmmis,
        'n_assigned': metadata_pygmmis['n_assigned'],
    }

    if verbose:
        print(f"  Time: {time_pygmmis:.3f} s")
        print(f"  Converged: {converged_pygmmis}")
        print(f"  Fitted means:")
        print(f"    Dye 1: A_R={means_pygmmis[perm_pg[0], 0]:.4f}, A_G={means_pygmmis[perm_pg[0], 1]:.4f}")
        print(f"    Dye 2: A_R={means_pygmmis[perm_pg[1], 0]:.4f}, A_G={means_pygmmis[perm_pg[1], 1]:.4f}")
        print(f"  Mean error from ground truth: {mean_error_pygmmis:.5f}")
        print(f"  Accuracy: {accuracy_pygmmis*100:.2f}% ({n_correct_pygmmis}/{len(assigned_pygmmis[assigned_pygmmis['channel']>=0])})")
        print(f"  Assignments: {metadata_pygmmis['n_assigned']}")
        print()

    # Comparison summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Speedup: {time_em / time_pygmmis:.2f}x (pygmmis is {'faster' if time_pygmmis < time_em else 'slower'})")
    print(f"Mean error: EM={mean_error_em:.5f}, pygmmis={mean_error_pygmmis:.5f} "
          f"(pygmmis is {'better' if mean_error_pygmmis < mean_error_em else 'worse'})")
    print(f"Accuracy: EM={accuracy_em*100:.2f}%, pygmmis={accuracy_pygmmis*100:.2f}% "
          f"(difference: {(accuracy_pygmmis - accuracy_em)*100:+.2f}%)")
    print()

    # Statistical test: Are the fitted means significantly different?
    mean_diff = np.linalg.norm(results['EM']['means'] - results['pygmmis']['means'])
    print(f"Difference between EM and pygmmis fitted means: {mean_diff:.5f}")

    if mean_diff < 0.01:
        print("✓ PASS: Both methods agree (difference < 0.01)")
    else:
        print("⚠ WARNING: Methods disagree significantly")

    print()

    return results


def test_scaling(n_samples_list=[20, 100], verbose=False):
    """
    Confirm EM and pygmmis GMM fitting both work at more than one dataset size,
    each producing a converged, correctly-shaped, finite fit.

    n_samples_list was originally [1000, 5000, 10000, 50000] as a genuine
    performance-scaling benchmark (the largest size alone took ~16s); shrunk
    since a unit test only needs to confirm both code paths still work across
    sizes, not measure how fast they are.

    Args:
        n_samples_list: List of sample sizes to test
        verbose: Print results

    Returns:
        scaling_results: Dict with timing results
    """
    if verbose:
        print("\n" + "=" * 80)
        print("SCALING TEST: How do methods scale with dataset size?")
        print("=" * 80)

    extractor = SM_E.extract_SMs()
    scaling_results = {'n_samples': [], 'time_em': [], 'time_pygmmis': [], 'speedup': []}

    for n_samples_per_dye in n_samples_list:
        if verbose:
            print(f"\nTesting with {n_samples_per_dye*2} total samples...")

        # Generate data
        df = generate_synthetic_2dye_data(n_samples_per_dye=n_samples_per_dye, random_state=42)

        # Time EM
        start = time.time()
        assigned_em, metadata_em = extractor.unmix_channels(
            df, n_channels=2, channels_to_use=['A_R', 'A_G'],
            gmm_fit_method='EM', confidence_threshold=0.95, verbose=False,
        )
        time_em = time.time() - start

        # Time pygmmis
        start = time.time()
        assigned_pygmmis, metadata_pygmmis = extractor.unmix_channels(
            df, n_channels=2, channels_to_use=['A_R', 'A_G'],
            gmm_fit_method='extreme_deconvolution', confidence_threshold=0.95, verbose=False,
        )
        time_pygmmis = time.time() - start

        # Real assertions: both methods must actually fit this dataset size,
        # not just run without raising.
        assert len(assigned_em) == len(df)
        assert len(assigned_pygmmis) == len(df)
        assert metadata_em['converged'], f"EM did not converge at n={n_samples_per_dye*2}"
        assert metadata_pygmmis['converged'], f"pygmmis did not converge at n={n_samples_per_dye*2}"
        assert metadata_em['means'].shape == (2, 2)
        assert metadata_pygmmis['means'].shape == (2, 2)
        assert np.all(np.isfinite(metadata_em['means']))
        assert np.all(np.isfinite(metadata_pygmmis['means']))
        assert np.all(np.isfinite(metadata_em['covariances']))
        assert np.all(np.isfinite(metadata_pygmmis['covariances']))
        assert time_em > 0
        assert time_pygmmis > 0

        speedup = time_em / time_pygmmis

        scaling_results['n_samples'].append(n_samples_per_dye * 2)
        scaling_results['time_em'].append(time_em)
        scaling_results['time_pygmmis'].append(time_pygmmis)
        scaling_results['speedup'].append(speedup)

        if verbose:
            print(f"  EM: {time_em:.3f} s")
            print(f"  pygmmis: {time_pygmmis:.3f} s")
            print(f"  Speedup: {speedup:.2f}x")

    if verbose:
        print("\n" + "=" * 80)
        print("SCALING SUMMARY")
        print("=" * 80)
        print(f"{'N_samples':<12} {'EM (s)':<10} {'pygmmis (s)':<12} {'Speedup':<10}")
        print("-" * 44)
        for i, n in enumerate(scaling_results['n_samples']):
            print(f"{n:<12} {scaling_results['time_em'][i]:<10.3f} "
                  f"{scaling_results['time_pygmmis'][i]:<12.3f} "
                  f"{scaling_results['speedup'][i]:<10.2f}x")
        print()

    return scaling_results


def main():
    """Run all tests."""
    print("Testing pygmmis integration for error-aware GMM fitting")
    print("=" * 80)
    print()

    # Test 1: Basic accuracy test
    print("TEST 1: Accuracy comparison with synthetic data")
    print("-" * 80)
    df = generate_synthetic_2dye_data(n_samples_per_dye=5000, random_state=42)
    results = compare_fitting_methods(df, verbose=True)

    # Test 2: Scaling test
    print("\nTEST 2: Performance scaling")
    print("-" * 80)
    scaling = test_scaling(n_samples_list=[1000, 5000, 10000], verbose=True)

    # Final verdict
    print("=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)

    # Check if pygmmis is better or comparable
    if (results['pygmmis']['mean_error'] <= results['EM']['mean_error'] * 1.1 and
        results['pygmmis']['accuracy'] >= results['EM']['accuracy'] - 0.02):
        print("✓ SUCCESS: pygmmis produces comparable or better results than EM replication")
    else:
        print("⚠ WARNING: pygmmis results differ significantly from EM replication")

    if scaling['speedup'][-1] > 1.0:
        print(f"✓ SUCCESS: pygmmis is {scaling['speedup'][-1]:.1f}x faster than EM replication at scale")
    else:
        print(f"⚠ pygmmis is slower than EM replication by {1/scaling['speedup'][-1]:.1f}x")

    print()
    print("Implementation complete! pygmmis can now be used with:")
    print("  gmm_fit_method='extreme_deconvolution'")
    print("or")
    print("  gmm_fit_method='pygmmis'")
    print()


if __name__ == "__main__":
    main()
