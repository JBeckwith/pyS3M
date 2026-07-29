#!/usr/bin/env python3
"""
Unit tests for optimal dye selector implementation.
"""
import sys
import os
import numpy as np
import types

# Add src to path

import pyS3M.Multicolour_Simulation_Functions as Multicolour_Simulation_Functions
import pyS3M.SpectralFunctions as SpectralFunctions
import pyS3M.MaskFunctions as MaskFunctions
import pyS3M.sCMOSFunctions as sCMOSFunctions


def test_filter_dyes_by_photons():
    """Test the _filter_dyes_by_photons method."""
    print("=" * 60)
    print("Testing _filter_dyes_by_photons")
    print("=" * 60)

    # Initialize classes
    MSF = Multicolour_Simulation_Functions.MultiC_Sim_Funcs()
    S_F = SpectralFunctions.Spectral_Funcs()

    # Define test data
    single_molecule_dyes = np.array([
        ["ATTO 488", 2073],
        ["Cy3B", 23195],
        ["ATTO 565", 11600],
        ["ATTO 643", 23327],
        ["ATTO 647N", 18448],
        ["Alexa Fluor 647", 10348],
    ], dtype="object")

    potential_dyes = [
        "Cy3B", "ATTO 565", "ATTO 643", "ATTO 647N", "Alexa Fluor 647"
    ]

    filters = [
        "semrock-nf03-405-488-561-635e",
        "semrock-di03-r405-488-561-635-t1-25x36",
        "semrock-bsp01-785r"
    ]

    # Get camera parameters
    R, G, B, wavelength = S_F.getpixelefficiency()

    # Create minimal camera parameters dict
    camera_parameters = {
        "pixel_QYs": np.vstack([B, G, R]),
    }

    # Test with default threshold (500 photons)
    print("\n1. Testing with min_photons_per_100ms=500")
    print("-" * 60)

    result = MSF._filter_dyes_by_photons(
        potential_dyes=potential_dyes,
        single_molecule_dyes=single_molecule_dyes,
        filters=filters,
        camera_parameters=camera_parameters,
        wavelength=wavelength,
        min_photons_per_100ms=500,
    )

    print(f"\nPotential dyes: {len(potential_dyes)}")
    print(f"Viable dyes: {len(result['viable_dyes'])}")
    print(f"\nViable: {result['viable_dyes']}")
    print(f"Rejected: {set(potential_dyes) - set(result['viable_dyes'])}")

    print("\nExpected photons per dye:")
    for dye in potential_dyes:
        if dye in result['expected_photons']:
            photons = result['expected_photons'][dye]
            status = "✓ VIABLE" if dye in result['viable_dyes'] else "✗ REJECTED"
            print(f"  {dye:20s}: {photons:7.0f} photons  {status}")

    # Test with higher threshold
    print("\n2. Testing with min_photons_per_100ms=10000")
    print("-" * 60)

    result2 = MSF._filter_dyes_by_photons(
        potential_dyes=potential_dyes,
        single_molecule_dyes=single_molecule_dyes,
        filters=filters,
        camera_parameters=camera_parameters,
        wavelength=wavelength,
        min_photons_per_100ms=10000,
    )

    print(f"\nViable dyes: {len(result2['viable_dyes'])}")
    print(f"Viable: {result2['viable_dyes']}")

    # Validate results
    assert len(result['viable_dyes']) > 0, "Should have at least some viable dyes"
    assert len(result2['viable_dyes']) < len(result['viable_dyes']), "Higher threshold should reduce viable dyes"

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)

    return result


def test_optimal_dye_selector():
    """Test the complete optimal_dye_selector_simulated method (small test)."""
    print("\n" + "=" * 60)
    print("Testing optimal_dye_selector_simulated (2 dyes, quick test)")
    print("=" * 60)

    # Initialize classes
    MSF = Multicolour_Simulation_Functions.MultiC_Sim_Funcs()
    S_F = SpectralFunctions.Spectral_Funcs()
    M_F = MaskFunctions.Mask_Functions()
    sCMOS = sCMOSFunctions.sCMOS_Functions()

    # Create smoothing function
    smoothing_function = types.SimpleNamespace()
    smoothing_function.args = {"sigma": 0.85}
    smoothing_function.data_arg = "image"
    smoothing_function.smoothing_function = sCMOS.gaussian_filter_stack

    # Define test data
    single_molecule_dyes = np.array([
        ["ATTO 488", 2073],
        ["Cy3B", 23195],
        ["ATTO 565", 11600],
        ["ATTO 643", 23327],
        ["ATTO 647N", 18448],
        ["Alexa Fluor 647", 10348],
    ], dtype="object")

    # Test with just 3 dyes to keep it fast
    potential_dyes = ["Cy3B", "ATTO 643", "ATTO 647N"]

    filters = [
        "semrock-nf03-405-488-561-635e",
        "semrock-di03-r405-488-561-635-t1-25x36",
        "semrock-bsp01-785r"
    ]

    # Get camera parameters
    R, G, B, wavelength = S_F.getpixelefficiency()

    # Create full camera parameters dict
    camera_parameters = {
        "gain": np.full((12, 12), 1.0),
        "offset": np.full((12, 12), 100),
        "variance": np.full((12, 12), 1e-12),
        "readnoise": 1e-12,
        "rqe": np.full((12, 12), 1.0),
        "pixel_QYs": np.vstack([B, G, R]),
        "masks": M_F.get_masks(size_x=12, size_y=12),
        "pixel_order": ['B', 'G', 'R'],
        "pixel_order_indices": {'B': 0, 'G': 1, 'R': 2}
    }

    # Select optimal 2 dyes (greedy, fast)
    result = MSF.optimal_dye_selector_simulated(
        potential_dyes=potential_dyes,
        single_molecule_dyes=single_molecule_dyes,
        filters=filters,
        camera_parameters=camera_parameters,
        wavelength=wavelength,
        n_dyes_desired=2,
        min_photons_per_100ms=500,
        n_simulations=100,  # Small for speed
        smoothing_function=smoothing_function,
        exhaustive_search=False,  # Use greedy
        verbose=True
    )

    print("\n" + "=" * 60)
    print("✓ Test completed successfully!")
    print("=" * 60)

    return result


if __name__ == "__main__":
    test_filter_dyes_by_photons()
    test_optimal_dye_selector()
