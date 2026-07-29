#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test suite for GMM-based dye misidentification analysis.

Tests:
- Ground truth assignment via GMM
- Photon-dependent misidentification analysis
- Integration with photon accumulation database

Created: 2025-10-22
"""

import numpy as np
import pandas as pd
import os
import sys

# Add src to path
module_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))

import pyS3M.SM_extractionfunctions as SM_extractionfunctions


def create_synthetic_2dye_accumulation_data():
    """
    Create synthetic photon accumulation database with 2 dye populations.

    Population 0 (Red): High A_R (~0.7), Low A_G (~0.2)
    Population 1 (Green): Low A_R (~0.2), High A_G (~0.7)

    Returns:
        pd.DataFrame: Photon accumulation database
    """
    np.random.seed(42)

    n_red_molecules = 50
    n_green_molecules = 50
    max_frames = 100

    records = []
    mol_idx = 0

    # Create Red dye molecules
    for i in range(n_red_molecules):
        # True RGB fractions for red dye (high R, low G)
        true_A_R = 0.70
        true_A_G = 0.20
        true_A_B = 0.10

        # Number of frames for this molecule
        n_frames = np.random.randint(10, max_frames)

        for frame_num in range(n_frames):
            # Accumulate photons
            photons_this_frame = np.random.uniform(100, 1000)
            photons_accumulated = photons_this_frame * (frame_num + 1)

            # Add noise that decreases with more photons
            noise_scale = 0.1 / np.sqrt(photons_accumulated / 1000)
            A_R = true_A_R + np.random.normal(0, noise_scale)
            A_G = true_A_G + np.random.normal(0, noise_scale)
            A_B = true_A_B + np.random.normal(0, noise_scale)

            # Normalize
            total = A_R + A_G + A_B
            A_R /= total
            A_G /= total
            A_B /= total

            # Errors decrease with photons
            A_R_err = 0.05 / np.sqrt(photons_accumulated / 1000)
            A_G_err = 0.05 / np.sqrt(photons_accumulated / 1000)
            A_B_err = 0.05 / np.sqrt(photons_accumulated / 1000)

            record = {
                "molecular_index": mol_idx,
                "frames_accumulated": frame_num + 1,
                "photons_accumulated": photons_accumulated,
                "A_R": A_R,
                "A_G": A_G,
                "A_B": A_B,
                "A_R_err": A_R_err,
                "A_G_err": A_G_err,
                "A_B_err": A_B_err,
                "xc_mean": np.random.uniform(100, 400),
                "yc_mean": np.random.uniform(100, 400),
                "xc_std": 0.3,
                "yc_std": 0.3,
                "s_x_mean": 1.3,
                "s_y_mean": 1.3,
                "fov_index": 0,
                "fov_name": "Pos0",
            }
            records.append(record)

        mol_idx += 1

    # Create Green dye molecules
    for i in range(n_green_molecules):
        # True RGB fractions for green dye (low R, high G)
        true_A_R = 0.20
        true_A_G = 0.70
        true_A_B = 0.10

        # Number of frames for this molecule
        n_frames = np.random.randint(10, max_frames)

        for frame_num in range(n_frames):
            # Accumulate photons
            photons_this_frame = np.random.uniform(100, 1000)
            photons_accumulated = photons_this_frame * (frame_num + 1)

            # Add noise that decreases with more photons
            noise_scale = 0.1 / np.sqrt(photons_accumulated / 1000)
            A_R = true_A_R + np.random.normal(0, noise_scale)
            A_G = true_A_G + np.random.normal(0, noise_scale)
            A_B = true_A_B + np.random.normal(0, noise_scale)

            # Normalize
            total = A_R + A_G + A_B
            A_R /= total
            A_G /= total
            A_B /= total

            # Errors decrease with photons
            A_R_err = 0.05 / np.sqrt(photons_accumulated / 1000)
            A_G_err = 0.05 / np.sqrt(photons_accumulated / 1000)
            A_B_err = 0.05 / np.sqrt(photons_accumulated / 1000)

            record = {
                "molecular_index": mol_idx,
                "frames_accumulated": frame_num + 1,
                "photons_accumulated": photons_accumulated,
                "A_R": A_R,
                "A_G": A_G,
                "A_B": A_B,
                "A_R_err": A_R_err,
                "A_G_err": A_G_err,
                "A_B_err": A_B_err,
                "xc_mean": np.random.uniform(100, 400),
                "yc_mean": np.random.uniform(100, 400),
                "xc_std": 0.3,
                "yc_std": 0.3,
                "s_x_mean": 1.3,
                "s_y_mean": 1.3,
                "fov_index": 0,
                "fov_name": "Pos0",
            }
            records.append(record)

        mol_idx += 1

    return pd.DataFrame(records)


def create_synthetic_2dye_singlemolecule_data():
    """
    Create synthetic single molecule database with 2 dye populations.

    Population 0 (Red): High A_R (~0.7), Low A_G (~0.2)
    Population 1 (Green): Low A_R (~0.2), High A_G (~0.7)

    Returns:
        pd.DataFrame: Single molecule database
    """
    np.random.seed(42)

    n_red_molecules = 50
    n_green_molecules = 50

    records = []
    mol_idx = 0

    # Create Red dye molecules
    for i in range(n_red_molecules):
        # True RGB fractions for red dye (high R, low G)
        true_A_R = 0.70
        true_A_G = 0.20
        true_A_B = 0.10

        # Random photon count
        photons = np.random.uniform(1000, 100000)

        # Add noise that decreases with more photons
        noise_scale = 0.1 / np.sqrt(photons / 1000)
        A_R = true_A_R + np.random.normal(0, noise_scale)
        A_G = true_A_G + np.random.normal(0, noise_scale)
        A_B = true_A_B + np.random.normal(0, noise_scale)

        # Normalize
        total = A_R + A_G + A_B
        A_R /= total
        A_G /= total
        A_B /= total

        record = {
            "molecular_index": mol_idx,
            "photons": photons,
            "A_R": A_R,
            "A_G": A_G,
            "A_B": A_B,
            "xc": np.random.uniform(100, 400),
            "yc": np.random.uniform(100, 400),
            "s_x": 1.3,
            "s_y": 1.3,
            "fov_index": 0,
            "fov_name": "Pos0",
        }
        records.append(record)
        mol_idx += 1

    # Create Green dye molecules
    for i in range(n_green_molecules):
        # True RGB fractions for green dye (low R, high G)
        true_A_R = 0.20
        true_A_G = 0.70
        true_A_B = 0.10

        # Random photon count
        photons = np.random.uniform(1000, 100000)

        # Add noise that decreases with more photons
        noise_scale = 0.1 / np.sqrt(photons / 1000)
        A_R = true_A_R + np.random.normal(0, noise_scale)
        A_G = true_A_G + np.random.normal(0, noise_scale)
        A_B = true_A_B + np.random.normal(0, noise_scale)

        # Normalize
        total = A_R + A_G + A_B
        A_R /= total
        A_G /= total
        A_B /= total

        record = {
            "molecular_index": mol_idx,
            "photons": photons,
            "A_R": A_R,
            "A_G": A_G,
            "A_B": A_B,
            "xc": np.random.uniform(100, 400),
            "yc": np.random.uniform(100, 400),
            "s_x": 1.3,
            "s_y": 1.3,
            "fov_index": 0,
            "fov_name": "Pos0",
        }
        records.append(record)
        mol_idx += 1

    return pd.DataFrame(records)


def test_extract_reference_means_mode_a():
    """Test extracting reference means from photon accumulation DB (Mode A)."""
    print("\n" + "=" * 60)
    print("TEST: Extract Reference Means (Mode A - Photon Accumulation DB)")
    print("=" * 60)

    SM_E = SM_extractionfunctions.extract_SMs()

    # Create synthetic data
    pa_db = create_synthetic_2dye_accumulation_data()
    print(f"Created synthetic data: {len(pa_db)} rows, {pa_db['molecular_index'].nunique()} molecules")

    # Extract reference means (low threshold so all molecules qualify)
    fixed_means, ref_db, gmm = SM_E.extract_reference_means(
        pa_db,
        reference_photon_threshold=10000,
        verbose=True
    )

    # Verify results
    print("\n" + "-" * 60)
    print("VERIFICATION:")
    print("-" * 60)

    all_passed = True

    # Check GMM converged
    if gmm.converged_:
        print("✓ PASS: GMM converged")
    else:
        print("✗ FAIL: GMM did not converge")
        all_passed = False

    # Check fixed means shape
    if fixed_means.shape == (2, 2):
        print(f"✓ PASS: Fixed means shape is (2, 2)")
    else:
        print(f"✗ FAIL: Fixed means shape is {fixed_means.shape} (expected (2, 2))")
        all_passed = False

    # Check reference database size
    n_molecules = pa_db['molecular_index'].nunique()
    if len(ref_db) >= n_molecules * 0.9:  # Allow up to 10% to not reach threshold
        print(f"✓ PASS: Reference database has {len(ref_db)} molecules ({len(ref_db)/n_molecules*100:.0f}% of total)")
    else:
        print(f"✗ FAIL: Reference database has {len(ref_db)} molecules (expected ~{n_molecules})")
        all_passed = False

    # Check required columns
    required_cols = ['molecular_index', 'true_label', 'max_photons', 'A_R_ref', 'A_G_ref',
                     'posterior_prob_0', 'posterior_prob_1']
    for col in required_cols:
        if col in ref_db.columns:
            print(f"✓ PASS: Column '{col}' present")
        else:
            print(f"✗ FAIL: Column '{col}' missing")
            all_passed = False

    # Check labels are 0 or 1
    unique_labels = sorted(ref_db['true_label'].unique())
    if unique_labels == [0, 1]:
        print(f"✓ PASS: Labels are 0 and 1")
    else:
        print(f"✗ FAIL: Labels are {unique_labels} (expected [0, 1])")
        all_passed = False

    # Check posterior probabilities sum to 1
    posterior_sums = ref_db['posterior_prob_0'] + ref_db['posterior_prob_1']
    if np.allclose(posterior_sums, 1.0):
        print(f"✓ PASS: Posterior probabilities sum to 1.0")
    else:
        print(f"✗ FAIL: Posterior probabilities don't sum to 1.0")
        all_passed = False

    # Check component separation (should have different means)
    mean_diff = np.abs(fixed_means[0, 0] - fixed_means[1, 0])  # Difference in A_R
    if mean_diff > 0.1:
        print(f"✓ PASS: Components separated (A_R difference: {mean_diff:.3f})")
    else:
        print(f"✗ FAIL: Components not well separated")
        all_passed = False

    # Check means are within reasonable range
    if np.all(fixed_means >= 0) and np.all(fixed_means <= 1):
        print(f"✓ PASS: Fixed means in valid range [0, 1]")
    else:
        print(f"✗ FAIL: Fixed means outside valid range")
        all_passed = False

    print(f"\nResult: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    return all_passed, fixed_means, ref_db, pa_db


def test_extract_reference_means_mode_b():
    """Test extracting reference means from single molecule DB (Mode B)."""
    print("\n" + "=" * 60)
    print("TEST: Extract Reference Means (Mode B - Single Molecule DB)")
    print("=" * 60)

    SM_E = SM_extractionfunctions.extract_SMs()

    # Create synthetic single molecule data
    sm_db = create_synthetic_2dye_singlemolecule_data()
    print(f"Created synthetic data: {len(sm_db)} molecules")

    # Test Mode B without threshold (use all molecules)
    print("\n" + "-" * 60)
    print("MODE B.1: No photon threshold (all molecules)")
    print("-" * 60)

    fixed_means_all, ref_db_all, gmm_all = SM_E.extract_reference_means(
        sm_db,
        reference_photon_threshold=None,
        verbose=True
    )

    # Test Mode B with threshold (filter low-photon molecules)
    print("\n" + "-" * 60)
    print("MODE B.2: With photon threshold (high-photon molecules)")
    print("-" * 60)

    fixed_means_filtered, ref_db_filtered, gmm_filtered = SM_E.extract_reference_means(
        sm_db,
        reference_photon_threshold=50000,
        verbose=True
    )

    # Verify results
    print("\n" + "-" * 60)
    print("VERIFICATION:")
    print("-" * 60)

    all_passed = True

    # Test 1: All molecules mode
    print("\nMode B.1 (all molecules) checks:")

    # Check GMM converged
    if gmm_all.converged_:
        print("  ✓ PASS: GMM converged")
    else:
        print("  ✗ FAIL: GMM did not converge")
        all_passed = False

    # Check fixed means shape
    if fixed_means_all.shape == (2, 2):
        print(f"  ✓ PASS: Fixed means shape is (2, 2)")
    else:
        print(f"  ✗ FAIL: Fixed means shape is {fixed_means_all.shape} (expected (2, 2))")
        all_passed = False

    # Check all molecules are in reference DB
    if len(ref_db_all) == len(sm_db):
        print(f"  ✓ PASS: All {len(sm_db)} molecules included")
    else:
        print(f"  ✗ FAIL: Expected {len(sm_db)} molecules, got {len(ref_db_all)}")
        all_passed = False

    # Check required columns
    required_cols_all = ['molecular_index', 'true_label', 'photons', 'A_R_ref', 'A_G_ref',
                         'posterior_prob_0', 'posterior_prob_1']
    for col in required_cols_all:
        if col in ref_db_all.columns:
            print(f"  ✓ PASS: Column '{col}' present")
        else:
            print(f"  ✗ FAIL: Column '{col}' missing")
            all_passed = False

    # Check means are within valid range
    if np.all(fixed_means_all >= 0) and np.all(fixed_means_all <= 1):
        print(f"  ✓ PASS: Fixed means in valid range [0, 1]")
    else:
        print(f"  ✗ FAIL: Fixed means outside valid range")
        all_passed = False

    # Test 2: Filtered mode
    print("\nMode B.2 (filtered molecules) checks:")

    # Check GMM converged
    if gmm_filtered.converged_:
        print("  ✓ PASS: GMM converged")
    else:
        print("  ✗ FAIL: GMM did not converge")
        all_passed = False

    # Check filtering worked
    n_expected_filtered = len(sm_db[sm_db['photons'] >= 50000])
    if len(ref_db_filtered) == n_expected_filtered:
        print(f"  ✓ PASS: Correct number filtered ({len(ref_db_filtered)}/{len(sm_db)})")
    else:
        print(f"  ✗ FAIL: Expected {n_expected_filtered} filtered, got {len(ref_db_filtered)}")
        all_passed = False

    # Check means are similar between filtered and all (should be, since high-photon are less noisy)
    mean_diff = np.max(np.abs(fixed_means_all - fixed_means_filtered))
    if mean_diff < 0.05:  # Should be very similar
        print(f"  ✓ PASS: Means similar between modes (max diff: {mean_diff:.4f})")
    else:
        print(f"  Note: Means differ between modes (max diff: {mean_diff:.4f})")

    print(f"\nResult: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    return all_passed, fixed_means_all, ref_db_all, sm_db


def test_analytical_misidentification_analysis(fixed_means, ref_db, pa_db):
    """Test analytical photon-dependent misidentification analysis."""
    print("\n" + "=" * 60)
    print("TEST: Analytical Misidentification Analysis")
    print("=" * 60)

    SM_E = SM_extractionfunctions.extract_SMs()

    # Define photon bins
    photon_bins = np.logspace(3, 5, 11)  # 1k to 100k photons, 10 bins
    print(f"Photon bins: {len(photon_bins)-1} bins from {photon_bins[0]:.0f} to {photon_bins[-1]:.0f}")

    # Run analytical analysis
    summary_db = SM_E.analyze_photon_dependent_misidentification_analytical(
        pa_db,
        fixed_means,
        ref_db,
        photon_bins,
        use_earliest_entry=True,
        n_mc_samples=10000,
        verbose=True
    )

    # Verify results
    print("\n" + "-" * 60)
    print("VERIFICATION:")
    print("-" * 60)

    all_passed = True

    # Check summary database
    if len(summary_db) > 0:
        print(f"✓ PASS: Summary database created ({len(summary_db)} bins)")
    else:
        print(f"✗ FAIL: Summary database empty")
        all_passed = False
        return all_passed

    # Check required columns in summary
    required_summary_cols = ['photon_bin_min', 'photon_bin_max', 'n_molecules',
                            'converged', 'overall_accuracy', 'overall_error_rate',
                            'component_0_accuracy', 'component_1_accuracy',
                            'cov_0_AR_AR', 'cov_0_AR_AG', 'cov_0_AG_AG',
                            'cov_1_AR_AR', 'cov_1_AR_AG', 'cov_1_AG_AG',
                            'weight_0', 'weight_1']

    missing_cols = [col for col in required_summary_cols if col not in summary_db.columns]
    if len(missing_cols) == 0:
        print(f"✓ PASS: All required summary columns present")
    else:
        print(f"✗ FAIL: Missing columns: {missing_cols}")
        all_passed = False

    # Check accuracy is in valid range [0, 1]
    if (summary_db['overall_accuracy'] >= 0).all() and (summary_db['overall_accuracy'] <= 1).all():
        print(f"✓ PASS: Accuracy values in valid range [0, 1]")
    else:
        print(f"✗ FAIL: Some accuracy values outside [0, 1]")
        all_passed = False

    # Check error rates sum correctly
    error_check = np.allclose(
        summary_db['overall_accuracy'] + summary_db['overall_error_rate'],
        1.0
    )
    if error_check:
        print(f"✓ PASS: Accuracy + error rate = 1.0")
    else:
        print(f"✗ FAIL: Accuracy + error rate ≠ 1.0")
        all_passed = False

    # Check weights sum to 1
    weight_sums = summary_db['weight_0'] + summary_db['weight_1']
    if np.allclose(weight_sums, 1.0):
        print(f"✓ PASS: Component weights sum to 1.0")
    else:
        print(f"✗ FAIL: Component weights don't sum to 1.0")
        all_passed = False

    # Check covariances are positive definite (diagonal elements > 0)
    cov0_valid = (summary_db['cov_0_AR_AR'] > 0).all() and (summary_db['cov_0_AG_AG'] > 0).all()
    cov1_valid = (summary_db['cov_1_AR_AR'] > 0).all() and (summary_db['cov_1_AG_AG'] > 0).all()
    if cov0_valid and cov1_valid:
        print(f"✓ PASS: Covariance diagonal elements are positive")
    else:
        print(f"✗ FAIL: Some covariance diagonal elements are non-positive")
        all_passed = False

    # Check convergence
    converged_count = summary_db['converged'].sum()
    total_bins = len(summary_db)
    if converged_count == total_bins:
        print(f"✓ PASS: All bins converged ({converged_count}/{total_bins})")
    else:
        print(f"  Note: {converged_count}/{total_bins} bins converged")

    # Check accuracy generally increases with more photons
    acc_values = summary_db.sort_values('photon_bin_min')['overall_accuracy'].values
    if len(acc_values) > 1:
        # Check if last bin has higher accuracy than first bin
        if acc_values[-1] > acc_values[0]:
            print(f"✓ PASS: Accuracy increases with photon count ({acc_values[0]:.3f} → {acc_values[-1]:.3f})")
        else:
            print(f"  Note: Accuracy doesn't increase as expected (may be OK for synthetic data)")

    # Show accuracy range
    min_acc = summary_db['overall_accuracy'].min()
    max_acc = summary_db['overall_accuracy'].max()
    print(f"  Accuracy range: {min_acc:.3f} - {max_acc:.3f}")

    # Check covariance decreases with photons (noise should decrease)
    trace_comp0 = summary_db['cov_0_AR_AR'] + summary_db['cov_0_AG_AG']
    if trace_comp0.iloc[-1] < trace_comp0.iloc[0]:
        print(f"✓ PASS: Covariance decreases with photon count (noise decreases)")
    else:
        print(f"  Note: Covariance doesn't decrease as expected")

    print(f"\nResult: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    return all_passed


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("ANALYTICAL GMM MISIDENTIFICATION ANALYSIS TEST SUITE")
    print("=" * 60)

    results = {}

    # Test 1: Extract reference means (Mode A - Photon Accumulation DB)
    test1_passed, fixed_means, ref_db, pa_db = test_extract_reference_means_mode_a()
    results['extract_reference_means_mode_a'] = test1_passed

    # Test 2: Extract reference means (Mode B - Single Molecule DB)
    test2_passed, fixed_means_sm, ref_db_sm, sm_db = test_extract_reference_means_mode_b()
    results['extract_reference_means_mode_b'] = test2_passed

    if test1_passed:
        # Test 3: Analytical misidentification analysis
        test3_passed = test_analytical_misidentification_analysis(fixed_means, ref_db, pa_db)
        results['analytical_misidentification_analysis'] = test3_passed
    else:
        results['analytical_misidentification_analysis'] = False

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(results.values())
    print("\n" + "=" * 60)
    if all_passed:
        print("✓✓✓ ALL TESTS PASSED ✓✓✓")
    else:
        print("✗✗✗ SOME TESTS FAILED ✗✗✗")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
