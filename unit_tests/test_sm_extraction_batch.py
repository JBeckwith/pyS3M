#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test suite for multi-FOV single molecule extraction functionality.

Tests the new batch processing methods:
- _extract_fov_name()
- extract_single_molecules_batch()
- build_photon_accumulation_database()
- analyze_multi_fov_dataset()

Created: 2025-10-22
"""

import numpy as np
import pandas as pd
import os
import sys
import tempfile
import shutil

# Add src to path
module_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))

import pyS3M.SM_extractionfunctions as SM_extractionfunctions
import pyS3M.IOFunctions as IOFunctions


def create_synthetic_localization_data(n_molecules=50, frames_per_molecule=10, fov_name="Pos0"):
    """
    Create synthetic localization data for testing.

    Args:
        n_molecules (int): Number of molecules to simulate
        frames_per_molecule (int): Frames per molecule (with some variation)
        fov_name (str): FOV identifier for output filename

    Returns:
        pd.DataFrame: Synthetic localization data with realistic columns
    """
    records = []

    for mol_idx in range(n_molecules):
        # Random position for this molecule (same across frames with small jitter)
        x_center = np.random.uniform(10, 500)
        y_center = np.random.uniform(10, 500)

        # Random number of frames (around frames_per_molecule)
        n_frames = max(3, frames_per_molecule + np.random.randint(-3, 4))

        for frame_idx in range(n_frames):
            # Position jitter (localization precision ~20 nm = 0.3 pixels at 69 nm/pixel)
            xc = x_center + np.random.normal(0, 0.3)
            yc = y_center + np.random.normal(0, 0.3)

            # Random photon counts (realistic range)
            photons = np.random.uniform(500, 5000)

            # Random RGB fractions (normalized)
            rgb = np.random.dirichlet([2, 3, 1])  # Bias toward G
            A_R = rgb[0] * photons
            A_G = rgb[1] * photons
            A_B = rgb[2] * photons

            # Realistic errors (scales as 1/sqrt(photons))
            A_R_err = A_R / np.sqrt(photons) * 0.1
            A_G_err = A_G / np.sqrt(photons) * 0.1
            A_B_err = A_B / np.sqrt(photons) * 0.1

            # PSF widths
            s_x = np.random.uniform(1.2, 1.5)
            s_y = np.random.uniform(1.2, 1.5)

            # Chi-squared (good fits)
            chi_sqr = np.random.gamma(5, 0.2)  # Mean ~1.0

            record = {
                "frame": frame_idx * 5,  # Non-consecutive frames
                "xc": xc,
                "yc": yc,
                "xc_err": 0.3,
                "yc_err": 0.3,
                "A_R": A_R,
                "A_G": A_G,
                "A_B": A_B,
                "A_R_err": A_R_err,
                "A_G_err": A_G_err,
                "A_B_err": A_B_err,
                "photons": photons,
                "s_x": s_x,
                "s_y": s_y,
                "chi_sqr": chi_sqr,
            }
            records.append(record)

    return pd.DataFrame(records)


def test_extract_fov_name():
    """Test FOV name extraction from filenames."""
    print("\n" + "=" * 60)
    print("TEST 1: FOV Name Extraction")
    print("=" * 60)

    SM_E = SM_extractionfunctions.extract_SMs()

    test_cases = [
        ("/path/to/Pos0_undrifted_locs.h5", "Pos0"),
        ("/path/to/Pos15_data.h5", "Pos15"),
        ("/path/to/Pos123_test.h5", "Pos123"),
        ("/path/to/nopattern.h5", None),
        ("/path/to/position_5.h5", None),
        ("Pos7_file.h5", "Pos7"),
    ]

    all_passed = True
    for filepath, expected in test_cases:
        result = SM_E._extract_fov_name(filepath)
        passed = result == expected
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {filepath} -> {result} (expected {expected})")
        all_passed = all_passed and passed

    print(f"\nResult: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    return all_passed


def test_batch_processing():
    """Test multi-FOV batch processing."""
    print("\n" + "=" * 60)
    print("TEST 2: Multi-FOV Batch Processing")
    print("=" * 60)

    # Create temporary directory for test files
    temp_dir = tempfile.mkdtemp()
    print(f"Using temp directory: {temp_dir}")

    try:
        SM_E = SM_extractionfunctions.extract_SMs()
        IO = IOFunctions.IO_Functions()

        # Create 3 synthetic FOV files
        n_fovs = 3
        fov_files = []
        expected_molecules_per_fov = []

        for i in range(n_fovs):
            fov_name = f"Pos{i}"
            n_molecules = 20 + i * 5  # 20, 25, 30 molecules
            expected_molecules_per_fov.append(n_molecules)

            # Create synthetic data
            loc_data = create_synthetic_localization_data(
                n_molecules=n_molecules,
                frames_per_molecule=10,
                fov_name=fov_name
            )

            # Save to HDF5
            filepath = os.path.join(temp_dir, f"{fov_name}_undrifted_locs.h5")
            IO._write_h5_database(loc_data, filepath, normalise_photons=False, append=False)
            fov_files.append(filepath)
            print(f"Created {fov_name}: {len(loc_data)} localizations")

        # Test batch processing with HDBSCAN
        print("\nRunning batch processing...")
        sm_db, sf_db = SM_E.extract_single_molecules_batch(
            fov_files,
            clustering_method="HDBSCAN",
            min_cluster_size=3,
            verbose=True
        )

        # Verify results
        print("\n" + "-" * 60)
        print("VERIFICATION:")
        print("-" * 60)

        all_passed = True

        # Check FOV columns exist
        required_cols = ["fov_index", "fov_name", "molecular_index"]
        for col in required_cols:
            if col in sm_db.columns and col in sf_db.columns:
                print(f"✓ PASS: Column '{col}' present in both databases")
            else:
                print(f"✗ FAIL: Column '{col}' missing")
                all_passed = False

        # Check FOV names
        unique_fov_names = sorted(sm_db["fov_name"].unique())
        expected_fov_names = ["Pos0", "Pos1", "Pos2"]
        if unique_fov_names == expected_fov_names:
            print(f"✓ PASS: FOV names correct: {unique_fov_names}")
        else:
            print(f"✗ FAIL: FOV names incorrect: {unique_fov_names} (expected {expected_fov_names})")
            all_passed = False

        # Check molecular_index uniqueness
        unique_mol_ids = sm_db["molecular_index"].unique()
        if len(unique_mol_ids) == len(sm_db):
            print(f"✓ PASS: All molecular_index values are unique ({len(unique_mol_ids)} molecules)")
        else:
            print(f"✗ FAIL: Duplicate molecular_index values")
            all_passed = False

        # Check molecular_index range per FOV
        for fov_idx in range(n_fovs):
            fov_molecules = sm_db[sm_db["fov_index"] == fov_idx]
            mol_ids = sorted(fov_molecules["molecular_index"].values)
            if len(mol_ids) > 0:
                min_id, max_id = mol_ids[0], mol_ids[-1]
                print(f"  FOV {fov_idx} (Pos{fov_idx}): molecular_index range {min_id}-{max_id} ({len(mol_ids)} molecules)")

        # Check frame database size
        print(f"\nSingle molecule database: {len(sm_db)} rows")
        print(f"Single frame database: {len(sf_db)} rows")

        if len(sf_db) > len(sm_db):
            print(f"✓ PASS: Frame database larger than molecule database")
        else:
            print(f"✗ FAIL: Frame database should be larger")
            all_passed = False

        print(f"\nResult: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
        return all_passed, temp_dir, sm_db, sf_db

    except Exception as e:
        print(f"\n✗ EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        shutil.rmtree(temp_dir)
        return False, None, None, None


def test_photon_accumulation(sf_db):
    """Test photon accumulation database building."""
    print("\n" + "=" * 60)
    print("TEST 3: Photon Accumulation Database")
    print("=" * 60)

    SM_E = SM_extractionfunctions.extract_SMs()

    # Build photon accumulation database
    pa_db = SM_E.build_photon_accumulation_database(sf_db, verbose=True)

    # Verify results
    print("\n" + "-" * 60)
    print("VERIFICATION:")
    print("-" * 60)

    all_passed = True

    # Check required columns
    required_cols = [
        "molecular_index", "frames_accumulated", "photons_accumulated",
        "A_R", "A_G", "A_B", "A_R_err", "A_G_err", "A_B_err",
        "xc_mean", "yc_mean", "xc_std", "yc_std"
    ]

    for col in required_cols:
        if col in pa_db.columns:
            print(f"✓ PASS: Column '{col}' present")
        else:
            print(f"✗ FAIL: Column '{col}' missing")
            all_passed = False

    # Check RGB normalization (should sum to 1.0)
    rgb_sum = pa_db["A_R"] + pa_db["A_G"] + pa_db["A_B"]
    close_to_one = np.abs(rgb_sum - 1.0) < 1e-6
    if np.all(close_to_one):
        print(f"✓ PASS: RGB normalized (sum = 1.0) for all rows")
    else:
        n_bad = np.sum(~close_to_one)
        print(f"✗ FAIL: RGB normalization failed for {n_bad} rows")
        all_passed = False

    # Check photons_accumulated is monotonically increasing per molecule
    unique_mols = pa_db["molecular_index"].unique()
    monotonic_ok = True

    for mol_id in unique_mols[:10]:  # Check first 10 molecules
        mol_data = pa_db[pa_db["molecular_index"] == mol_id]
        photons = mol_data["photons_accumulated"].values
        if not np.all(np.diff(photons) > 0):
            print(f"✗ FAIL: Photons not monotonically increasing for molecule {mol_id}")
            monotonic_ok = False
            all_passed = False
            break

    if monotonic_ok:
        print(f"✓ PASS: Photons monotonically increasing per molecule")

    # Check frames_accumulated
    for mol_id in unique_mols[:10]:
        mol_data = pa_db[pa_db["molecular_index"] == mol_id]
        frames = mol_data["frames_accumulated"].values
        expected_frames = np.arange(1, len(mol_data) + 1)
        if np.array_equal(frames, expected_frames):
            pass  # OK
        else:
            print(f"✗ FAIL: frames_accumulated not sequential for molecule {mol_id}")
            all_passed = False
            break
    else:
        print(f"✓ PASS: frames_accumulated sequential (1, 2, 3, ...)")

    # Test filtering by photon range
    print("\nTesting photon range filtering...")
    filtered = pa_db[
        (pa_db["photons_accumulated"] >= 1000) &
        (pa_db["photons_accumulated"] < 1100)
    ]
    print(f"  Molecules with 1000-1100 photons: {len(filtered)} rows")

    if len(filtered) > 0:
        print(f"✓ PASS: Photon range filtering works")
    else:
        print(f"  (No molecules in this range - expected for small test dataset)")

    # Show example molecule
    if len(unique_mols) > 0:
        example_mol = unique_mols[0]
        example_data = pa_db[pa_db["molecular_index"] == example_mol]
        print(f"\nExample molecule {example_mol}:")
        print(f"  Frames: {example_data['frames_accumulated'].min()} - {example_data['frames_accumulated'].max()}")
        print(f"  Photons: {example_data['photons_accumulated'].min():.1f} - {example_data['photons_accumulated'].max():.1f}")
        print(f"  A_R range: {example_data['A_R'].min():.3f} - {example_data['A_R'].max():.3f}")
        print(f"  A_G range: {example_data['A_G'].min():.3f} - {example_data['A_G'].max():.3f}")
        print(f"  A_B range: {example_data['A_B'].min():.3f} - {example_data['A_B'].max():.3f}")

    print(f"\nResult: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    return all_passed


def test_full_workflow(temp_dir):
    """Test complete analyze_multi_fov_dataset workflow with saving."""
    print("\n" + "=" * 60)
    print("TEST 4: Full Workflow with File Saving")
    print("=" * 60)

    SM_E = SM_extractionfunctions.extract_SMs()

    # Get list of test files
    fov_files = sorted([
        os.path.join(temp_dir, f)
        for f in os.listdir(temp_dir)
        if f.endswith(".h5")
    ])

    print(f"Found {len(fov_files)} FOV files")

    # Create output directory
    output_dir = os.path.join(temp_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    # Run full workflow
    sm_db, sf_db, pa_db = SM_E.analyze_multi_fov_dataset(
        fov_files,
        clustering_method="HDBSCAN",
        build_accumulation=True,
        min_cluster_size=3,
        output_folder=output_dir,
        output_prefix="test_analysis",
        verbose=True
    )

    # Verify output files
    print("\n" + "-" * 60)
    print("VERIFICATION:")
    print("-" * 60)

    all_passed = True

    expected_files = [
        "test_analysis_single_molecules.h5",
        "test_analysis_single_frames.h5",
        "test_analysis_photon_accumulation.h5",
    ]

    for filename in expected_files:
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            size_kb = os.path.getsize(filepath) / 1024
            print(f"✓ PASS: {filename} exists ({size_kb:.1f} KB)")
        else:
            print(f"✗ FAIL: {filename} missing")
            all_passed = False

    # Test loading saved files
    print("\nTesting file loading...")
    try:
        sm_loaded = pd.read_hdf(os.path.join(output_dir, "test_analysis_single_molecules.h5"))
        sf_loaded = pd.read_hdf(os.path.join(output_dir, "test_analysis_single_frames.h5"))
        pa_loaded = pd.read_hdf(os.path.join(output_dir, "test_analysis_photon_accumulation.h5"))

        if len(sm_loaded) == len(sm_db):
            print(f"✓ PASS: Single molecule database loaded correctly ({len(sm_loaded)} rows)")
        else:
            print(f"✗ FAIL: Single molecule database size mismatch")
            all_passed = False

        if len(sf_loaded) == len(sf_db):
            print(f"✓ PASS: Single frame database loaded correctly ({len(sf_loaded)} rows)")
        else:
            print(f"✗ FAIL: Single frame database size mismatch")
            all_passed = False

        if len(pa_loaded) == len(pa_db):
            print(f"✓ PASS: Photon accumulation database loaded correctly ({len(pa_loaded)} rows)")
        else:
            print(f"✗ FAIL: Photon accumulation database size mismatch")
            all_passed = False

    except Exception as e:
        print(f"✗ FAIL: Error loading files: {e}")
        all_passed = False

    print(f"\nResult: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    return all_passed


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("MULTI-FOV SINGLE MOLECULE EXTRACTION TEST SUITE")
    print("=" * 60)

    results = {}

    # Test 1: FOV name extraction
    results["fov_name_extraction"] = test_extract_fov_name()

    # Test 2: Batch processing
    batch_passed, temp_dir, sm_db, sf_db = test_batch_processing()
    results["batch_processing"] = batch_passed

    if batch_passed and temp_dir is not None:
        # Test 3: Photon accumulation
        results["photon_accumulation"] = test_photon_accumulation(sf_db)

        # Test 4: Full workflow
        results["full_workflow"] = test_full_workflow(temp_dir)

        # Cleanup
        print("\n" + "=" * 60)
        print("CLEANUP")
        print("=" * 60)
        print(f"Removing temp directory: {temp_dir}")
        shutil.rmtree(temp_dir)
        print("✓ Cleanup complete")
    else:
        results["photon_accumulation"] = False
        results["full_workflow"] = False

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
