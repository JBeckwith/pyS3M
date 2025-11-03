#!/usr/bin/env python3
"""
Example workflow for using the optimal dye selector with visualization.

This demonstrates the typical usage pattern:
1. Run optimal_dye_selector_simulated()
2. Call plot_dye_selection_results() on the output
"""
import sys
import os
import numpy as np
import types

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import Multicolour_Simulation_Functions
import SpectralFunctions
import MaskFunctions
import sCMOSFunctions


def main():
    """Example workflow for dye selection."""

    # ========================================
    # STEP 1: Initialize and setup
    # ========================================
    print("Initializing...")
    MSF = Multicolour_Simulation_Functions.MultiC_Sim_Funcs()
    S_F = SpectralFunctions.Spectral_Funcs()
    M_F = MaskFunctions.Mask_Functions()
    sCMOS = sCMOSFunctions.sCMOS_Functions()

    # Create smoothing function
    smoothing_function = types.SimpleNamespace()
    smoothing_function.args = {"sigma": 1.2}
    smoothing_function.data_arg = "image"
    smoothing_function.smoothing_function = sCMOS.gaussian_filter_stack

    # ========================================
    # STEP 2: Define dyes and parameters
    # ========================================

    # Dye photon counts (photons per 100ms)
    single_molecule_dyes = np.array([
        ["CF488A", 15000],
        ["ATTO 565", 11600],
        ["Cy3B", 23195],
        ["ATTO 643", 23327],
        ["ATTO 647N", 18448],
        ["Alexa Fluor 647", 10348],
    ], dtype="object")

    # Candidate dyes to choose from
    potential_dyes = [
        "CF488A", "ATTO 565", "Cy3B", "ATTO 643",
        "ATTO 647N", "Alexa Fluor 647"
    ]

    # Filter configuration
    filters = [
        "semrock-nf03-405-488-561-635e",
        "semrock-di03-r405-488-561-635-t1-25x36",
        "semrock-bsp01-785r"
    ]

    # Camera parameters
    R, G, B, wavelength = S_F.getpixelefficiency()
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

    # ========================================
    # STEP 3: Run dye selection
    # ========================================
    print("\nRunning dye selection...")
    print("=" * 60)

    result = MSF.optimal_dye_selector_simulated(
        potential_dyes=potential_dyes,
        single_molecule_dyes=single_molecule_dyes,
        filters=filters,
        camera_parameters=camera_parameters,
        wavelength=wavelength,
        n_dyes_desired=4,  # Select best 4 dyes
        min_photons_per_100ms=500,
        n_simulations=500,
        smoothing_function=smoothing_function,
        exhaustive_search=False,  # Use greedy (fast)
        verbose=True
    )

    # ========================================
    # STEP 4: Visualize results (EASY!)
    # ========================================
    print("\n" + "=" * 60)
    print("Creating visualization...")
    print("=" * 60)

    output_path = os.path.join(
        os.path.dirname(__file__),
        "../outputs/example_dye_selection.png"
    )

    # Create outputs directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # ONE LINE TO PLOT EVERYTHING!
    fig, axes = MSF.plot_dye_selection_results(
        result,
        save_path=output_path,
        show=False
    )

    print(f"\n✓ Visualization saved to: {output_path}")

    # ========================================
    # STEP 5: Print summary
    # ========================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nSelected {len(result['selected_dyes'])} dyes:")
    for i, dye in enumerate(result['selected_dyes'], 1):
        acc = result['separability_stats']['accuracy_per_dye'][i-1]
        photons_src = result['expected_photons'][dye]
        print(f"  {i}. {dye:20s} - {acc:.1%} accuracy ({photons_src:.0f} photons/100ms)")

    print(f"\nOverall classification accuracy: {result['overall_accuracy']:.1%}")

    # Check for problematic pairs
    print("\nPotential confusion pairs (accuracy < 90%):")
    conf_matrix = result['confusion_matrix']
    for i in range(len(result['selected_dyes'])):
        for j in range(len(result['selected_dyes'])):
            if i != j and conf_matrix[i, j] > 0.1:
                print(f"  {result['selected_dyes'][i]} → {result['selected_dyes'][j]}: "
                      f"{conf_matrix[i, j]:.1%} misclassification")

    print("\n" + "=" * 60)
    print("✓ Workflow complete!")
    print("=" * 60)

    return result


if __name__ == "__main__":
    result = main()
