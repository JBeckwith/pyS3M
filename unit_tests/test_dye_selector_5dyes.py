#!/usr/bin/env python3
"""
Test optimal dye selector with 5 dyes and visualization.
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


def test_5_dye_selection_with_plots():
    """Test 5-dye selection with confusion matrix and visualization."""
    print("=" * 60)
    print("Testing 5-dye selection with plots")
    print("=" * 60)

    # Initialize classes
    MSF = Multicolour_Simulation_Functions.MultiC_Sim_Funcs()
    S_F = SpectralFunctions.Spectral_Funcs()
    M_F = MaskFunctions.Mask_Functions()
    sCMOS = sCMOSFunctions.sCMOS_Functions()

    # Create smoothing function
    smoothing_function = types.SimpleNamespace()
    smoothing_function.args = {"sigma": 1.2}
    smoothing_function.data_arg = "image"
    smoothing_function.smoothing_function = sCMOS.gaussian_filter_stack

    # Define test data - 5 dyes spanning the spectrum
    single_molecule_dyes = np.array([
        ["CF488A", 15000],
        ["ATTO 565", 11600],
        ["Cy3B", 23195],
        ["ATTO 643", 23327],
        ["ATTO 647N", 18448],
    ], dtype="object")

    potential_dyes = [
        "CF488A", "ATTO 565", "Cy3B", "ATTO 643", "ATTO 647N"
    ]

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

    # Select all 5 dyes (use greedy for speed)
    result = MSF.optimal_dye_selector_simulated(
        potential_dyes=potential_dyes,
        single_molecule_dyes=single_molecule_dyes,
        filters=filters,
        camera_parameters=camera_parameters,
        wavelength=wavelength,
        n_dyes_desired=5,
        min_photons_per_100ms=500,
        n_simulations=500,  # 500 per dye for good statistics
        smoothing_function=smoothing_function,
        exhaustive_search=False,  # Use greedy
        verbose=True
    )

    print("\n" + "=" * 60)
    print("DETAILED RESULTS")
    print("=" * 60)

    # Check confusion matrix dimensions
    conf_matrix = result['confusion_matrix']
    print(f"\nConfusion matrix shape: {conf_matrix.shape}")
    assert conf_matrix.shape == (5, 5), f"Expected 5x5 matrix, got {conf_matrix.shape}"

    # Print detailed confusion matrix
    print("\nDetailed Confusion Matrix:")
    print("(rows = true dye, columns = classified as)")
    print()

    # Header
    dye_names = result['selected_dyes']
    header = "True \\ Pred |" + "|".join([f" {dye:12s}" for dye in dye_names])
    print(header)
    print("-" * len(header))

    # Matrix rows
    for i, true_dye in enumerate(dye_names):
        row_str = f"{true_dye:12s} |"
        for j in range(len(dye_names)):
            row_str += f" {conf_matrix[i, j]:12.3f}|"
        print(row_str)

    print()
    print(f"Diagonal (correct classifications): {np.diag(conf_matrix)}")
    print(f"Per-dye accuracy: {result['separability_stats']['accuracy_per_dye']}")
    print(f"Overall accuracy: {result['overall_accuracy']:.3f}")

    # Plot the color distributions with Gaussian fits
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATION")
    print("=" * 60)

    output_path = os.path.join(
        os.path.dirname(__file__),
        "../outputs/dye_color_distributions_5dyes.png"
    )

    # Create outputs directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = MSF.plot_dye_color_distributions(
        dye_simulations=result['dye_simulations'],
        dye_gaussians=result['dye_gaussians'],
        dye_names=result['selected_dyes'],
        save_path=output_path,
        show=False,
        n_std=2.0
    )

    print(f"\nVisualization saved to: {output_path}")
    print("\nThis plot shows:")
    print("  - Scatter points: Individual simulated molecules")
    print("  - Solid circles: Mean (A_R, A_G) for each dye")
    print("  - Dashed ellipses: 2σ confidence regions (95%)")
    print()
    print("If the Gaussian assumption is valid:")
    print("  1. Points should be roughly elliptical around the mean")
    print("  2. ~95% of points should fall within the 2σ ellipse")
    print("  3. No strong outliers or multimodal distributions")

    # Validate Gaussian assumption
    print("\n" + "=" * 60)
    print("VALIDATING GAUSSIAN ASSUMPTION")
    print("=" * 60)

    from scipy.stats import chi2

    for dye_name in result['selected_dyes']:
        A_R = result['dye_simulations'][dye_name]['A_R']
        A_G = result['dye_simulations'][dye_name]['A_G']

        # Remove NaNs
        valid = ~(np.isnan(A_R) | np.isnan(A_G))
        A_R_valid = A_R[valid]
        A_G_valid = A_G[valid]

        # Get Gaussian parameters
        mean = result['dye_gaussians'][dye_name]['mean']
        cov = result['dye_gaussians'][dye_name]['covariance']

        # Calculate Mahalanobis distance for each point
        X = np.vstack([A_R_valid, A_G_valid]).T
        diff = X - mean
        cov_inv = np.linalg.inv(cov)
        mahal_dist_sq = np.sum(diff @ cov_inv * diff, axis=1)

        # Chi-squared test: for 2D Gaussian, Mahalanobis distance^2 ~ chi2(2)
        # 95% of points should have distance^2 < chi2(2, 0.95) = 5.99
        chi2_95 = chi2.ppf(0.95, df=2)
        fraction_within_95 = np.mean(mahal_dist_sq < chi2_95)

        print(f"\n{dye_name}:")
        print(f"  Expected within 2σ (95%): 95%")
        print(f"  Observed within 2σ: {fraction_within_95 * 100:.1f}%")

        if abs(fraction_within_95 - 0.95) < 0.05:
            print(f"  ✓ Gaussian assumption is valid")
        else:
            print(f"  ⚠️  Deviation from Gaussian: {(fraction_within_95 - 0.95) * 100:+.1f}%")

    print("\n" + "=" * 60)
    print("✓ Test completed successfully!")
    print("=" * 60)

    return result


if __name__ == "__main__":
    result = test_5_dye_selection_with_plots()
