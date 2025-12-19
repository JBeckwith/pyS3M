#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple validation test for Bayer spot detection using synthetic Gaussian spots.

This is a lightweight version that doesn't require camera calibration files.
Creates simple synthetic spots to test the coordinate mapping and detection logic.

Created: December 19, 2025
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Add src to path
module_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, module_dir)

import SpotDetectionFunctions
import BayerSpotDetection
import MaskFunctions
import Multicolour_Simulation_Functions as MSF
import PSFFunctions
import sCMOSFunctions
import SpectralFunctions
from SpectralFunctions import SpectralDataType
import IOFunctions


def create_synthetic_bayer_image(
    spot_positions: np.ndarray,
    image_size: tuple = (512, 512),
    spot_intensity: float = 2000,
    background: float = 10,
    dye_name: str = 'ATTO 565',
    pixel_size: float = 69,  # nm
    NA: float = 1.49,
    pattern: str = 'RGGB'
) -> np.ndarray:
    """Create synthetic Bayer image with Gaussian spots using proper simulation.

    Uses Multicolour_Simulation_Functions and MaskFunctions for realistic imaging.

    Args:
        spot_positions: Array of [y, x] positions in pixels
        image_size: (H, W) tuple
        spot_intensity: Number of photons per spot (default: 2000)
        background: Background photons per pixel (default: 10)
        dye_name: Dye to simulate (default: 'ATTO 565')
        pixel_size: Pixel size in nm
        NA: Numerical aperture
        pattern: Bayer pattern

    Returns:
        bayer_image: Synthetic Bayer image
    """
    H, W = image_size

    # Initialize modules
    psf = PSFFunctions.PSF_Functions()
    scmos = sCMOSFunctions.sCMOS_Functions()
    mask_gen = MaskFunctions.Mask_Functions()
    spectral = SpectralFunctions.Spectral_Funcs()

    # Create simple camera calibration (uniform gain/offset/variance)
    gain = np.ones((H, W), dtype=np.float32)
    offset = np.zeros((H, W), dtype=np.float32)
    variance = np.ones((H, W), dtype=np.float32)
    rqe = np.ones((H, W), dtype=np.float32)

    # Get Bayer masks using MaskFunctions
    mosaic_unit = BayerSpotDetection.get_mosaic_unit_from_pattern(pattern)
    masks = mask_gen.get_masks(H, W, mosaic_unit)

    # Load actual dye spectral data
    wavelengths_nm = np.arange(400, 801, 1)

    # Get emission spectrum for dye (returns array of shape (1, n_wavelengths))
    emission_data = spectral.get_spectral_data(
        [dye_name], wavelengths_nm, data_type=SpectralDataType.DYE
    )
    emission = emission_data[0]  # Get first (and only) spectrum

    # Get filter transmission data for RGB channels
    filter_names = ['Omega550LP', 'Omega600_50m', 'Omega650LP']  # Example filters
    try:
        # Returns array of shape (3, n_wavelengths)
        filters_data = spectral.get_spectral_data(
            filter_names, wavelengths_nm, data_type=SpectralDataType.FILTER
        )
        # pixel_QYs should be shape (n_colors, n_wavelengths) = (3, n_wavelengths)
        pixel_QYs = np.vstack([
            filters_data[0],  # B: long pass 550
            filters_data[1],  # G: bandpass 600
            filters_data[2]   # R: long pass 650
        ])
    except Exception as e:
        print(f"  Warning: Could not load filters ({e}), using fallback")
        # Fallback: simple RGB filters with small non-zero values everywhere
        # Shape: (3, n_wavelengths) for B, G, R
        pixel_QYs = np.ones((3, len(wavelengths_nm))) * 0.01  # Small baseline
        pixel_QYs[0, wavelengths_nm >= 550] = 1.0  # B: long pass 550
        pixel_QYs[1, (wavelengths_nm >= 575) & (wavelengths_nm <= 625)] = 1.0  # G: bandpass 600
        pixel_QYs[2, wavelengths_nm >= 650] = 1.0  # R: long pass 650

    # Calculate dye pixel efficiency (emission shape: (n_wavelengths,), pixel_QYs shape: (3, n_wavelengths))
    dye_pixel_efficiency = np.dot(pixel_QYs, emission)  # Result shape: (3,)

    camera_params = {
        'gain': gain,
        'offset': offset,
        'variance': variance,
        'readnoise': 1.0,
        'rqe': rqe,
        'masks': masks,
        'pixel_QYs': pixel_QYs,
        'pixel_order': ['B', 'G', 'R'],
        'pixel_order_indices': {'B': 0, 'G': 1, 'R': 2}
    }

    # Convert pixel positions to nm coordinates
    # spot_positions is (n_spots, 2) with [y, x]
    # Format expected: (n_molecules, 2, n_frames) based on notebooks
    n_spots = len(spot_positions)
    x0y0 = np.zeros((n_spots, 2, 1))  # (n_spots, [x,y], n_frames=1)
    for i in range(n_spots):
        x0y0[i, 0, 0] = spot_positions[i, 1] * pixel_size  # x coordinate
        x0y0[i, 1, 0] = spot_positions[i, 0] * pixel_size  # y coordinate

    # Prepare inputs for simulation
    x0y0_dict = {'dye1': x0y0}
    photons_dict = {'dye1': np.full(n_spots, spot_intensity, dtype=int)}

    # Calculate average emission wavelength from spectrum
    avg_wavelength = np.sum(wavelengths_nm * emission) / np.sum(emission) / 1000  # Convert to microns

    # Create smoothing function object (as expected by gen_camera_image_stack)
    import types
    sigma_nm = psf.sigma_PSF(avg_wavelength, NA)
    sigma_px = sigma_nm / pixel_size

    smoothing_fn = types.SimpleNamespace()
    smoothing_fn.args = {"sigma": sigma_px}
    smoothing_fn.extent = sigma_px
    smoothing_fn.smoothing_function = scmos.gaussian_filter_stack
    smoothing_fn.data_arg = "image"

    # Generate Bayer image using Multicolour_Simulation_Functions
    simulator = MSF.MultiC_Sim_Funcs()

    bayer_stack, _, _ = simulator.gen_camera_image_stack(
        camera_calibration=camera_params,
        wavelength=wavelengths_nm,
        average_emission_wavelengths=avg_wavelength,
        dye_pixel_efficiency=dye_pixel_efficiency,
        n_photons=photons_dict,
        x0y0=x0y0_dict,
        smoothing_function=smoothing_fn,
        background_photons=background,
        background_colour=[1, 1, 1],
        NA=NA,
        pixel_size=pixel_size,
        return_normal_image=False,
        use_vectorized_photoelectrons=True
    )

    # Return image (simulation returns 2D when n_frames=1)
    print(f"  Bayer image shape: {bayer_stack.shape}")
    return bayer_stack


def simple_demosaic(bayer_image: np.ndarray, pattern: str = 'RGGB') -> np.ndarray:
    """Simple grayscale demosaic by extracting and averaging color channels.

    Uses MaskFunctions to properly extract channels, then averages them.

    Args:
        bayer_image: Bayer patterned image
        pattern: Bayer pattern

    Returns:
        gray_image: Demosaiced grayscale image
    """
    from scipy.ndimage import convolve

    H, W = bayer_image.shape

    # Use MaskFunctions to get proper Bayer masks
    mask_gen = MaskFunctions.Mask_Functions()
    mosaic_unit = BayerSpotDetection.get_mosaic_unit_from_pattern(pattern)
    masks = mask_gen.get_masks(H, W, mosaic_unit)

    # Extract and interpolate each channel
    r_raw = np.zeros_like(bayer_image)
    g_raw = np.zeros_like(bayer_image)
    b_raw = np.zeros_like(bayer_image)

    r_raw[masks['R']] = bayer_image[masks['R']]
    g_raw[masks['G']] = bayer_image[masks['G']]
    b_raw[masks['B']] = bayer_image[masks['B']]

    # Simple interpolation kernel
    kernel = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]], dtype=np.float32) / 16

    r_interp = convolve(r_raw, kernel, mode='reflect')
    g_interp = convolve(g_raw, kernel, mode='reflect')
    b_interp = convolve(b_raw, kernel, mode='reflect')

    # Average RGB channels
    gray = (r_interp + g_interp + b_interp) / 3

    return gray


def test_coordinate_mapping():
    """Test that coordinate mapping works correctly."""
    print("="*60)
    print("Test 1: Coordinate Mapping")
    print("="*60)

    # Create a single spot at a known position
    true_position = np.array([[100, 150]])  # [y, x]

    # Create Bayer image using proper simulation
    # Single PSF of ~2000 photons ATTO 565 with ~10 background photons per pixel
    bayer_img = create_synthetic_bayer_image(
        true_position,
        image_size=(256, 256),
        spot_intensity=2000,
        background=10,
        dye_name='ATTO 565',
        pixel_size=69,
        NA=1.49
    )

    # Initialize detector
    spot_detector = SpotDetectionFunctions.SpotDetection_Functions()

    # Detect using Bayer-aware method (need to add frame dimension)
    detections, metadata = BayerSpotDetection.detect_spots_bayer_multichannel(
        bayer_img[np.newaxis, :, :],  # Add frame dimension
        spot_detector=spot_detector,
        pattern='RGGB',
        pfa=1e-3,
        sigma=1.5,
        wavelength=0.58,
        pixel_size=0.069,
        NA=1.49
    )

    print(f"\nTrue position: y={true_position[0, 0]}, x={true_position[0, 1]}")

    if len(detections) > 0:
        # Find closest detection
        distances = np.sqrt(
            (detections[:, 0] - true_position[0, 0])**2 +
            (detections[:, 1] - true_position[0, 1])**2
        )
        closest_idx = np.argmin(distances)
        detected_y, detected_x = detections[closest_idx, :2]

        print(f"Detected position: y={detected_y:.1f}, x={detected_x:.1f}")
        print(f"Error: {distances[closest_idx]:.2f} pixels")

        if distances[closest_idx] < 3.0:
            print("✓ PASS: Position recovered within 3 pixels")
            return True
        else:
            print("✗ FAIL: Position error > 3 pixels")
            return False
    else:
        print("✗ FAIL: No spots detected")
        return False


def test_detection_comparison():
    """Compare Bayer-aware vs demosaic detection."""
    print("\n" + "="*60)
    print("Test 2: Detection Method Comparison")
    print("="*60)

    # Create multiple spots
    np.random.seed(42)
    n_spots = 20
    ground_truth = np.column_stack([
        np.random.uniform(50, 200, n_spots),  # y
        np.random.uniform(50, 200, n_spots)   # x
    ])

    # Create Bayer image using proper simulation
    # Multiple PSFs of ~2000 photons ATTO 565 with ~10 background photons per pixel
    bayer_img = create_synthetic_bayer_image(
        ground_truth,
        image_size=(256, 256),
        spot_intensity=2000,
        background=10,
        dye_name='ATTO 565',
        pixel_size=69,
        NA=1.49
    )

    # Demosaic
    gray_img = simple_demosaic(bayer_img)

    # Initialize detector
    spot_detector = SpotDetectionFunctions.SpotDetection_Functions()

    # Bayer-aware detection
    print("\nBayer-aware detection...")
    bayer_dets, metadata = BayerSpotDetection.detect_spots_bayer_multichannel(
        bayer_img[np.newaxis, :, :],  # Add frame dimension
        spot_detector=spot_detector,
        pattern='RGGB',
        pfa=1e-3,
        sigma=1.5,
        wavelength=0.58,
        pixel_size=0.069,
        NA=1.49
    )

    # Demosaic detection
    print("\nDemosaic-then-detect...")
    demosaic_dets = spot_detector.detect_puncta_in_stack_parallel(
        gray_img[np.newaxis, :, :],  # Add frame dimension
        pfa=1e-3,
        sigma=1.5,
        wavelength=0.58,
        pixel_size=0.069,
        NA=1.49
    )

    print(f"\nGround truth: {len(ground_truth)} spots")
    print(f"Bayer-aware detected: {len(bayer_dets)} spots")
    print(f"Demosaic detected: {len(demosaic_dets)} spots")

    # Simple matching: count how many ground truth spots are within 3 pixels
    def count_matches(detections, ground_truth, threshold=3.0):
        if len(detections) == 0:
            return 0

        matches = 0
        for gt in ground_truth:
            distances = np.sqrt(
                (detections[:, 0] - gt[0])**2 +
                (detections[:, 1] - gt[1])**2
            )
            if np.min(distances) < threshold:
                matches += 1
        return matches

    bayer_matches = count_matches(bayer_dets, ground_truth)
    demosaic_matches = count_matches(demosaic_dets, ground_truth)

    print(f"\nBayer-aware recall: {bayer_matches}/{len(ground_truth)} = {bayer_matches/len(ground_truth):.1%}")
    print(f"Demosaic recall: {demosaic_matches}/{len(ground_truth)} = {demosaic_matches/len(ground_truth):.1%}")

    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(bayer_img, cmap='gray')
    axes[0].scatter(ground_truth[:, 1], ground_truth[:, 0],
                   s=100, facecolors='none', edgecolors='lime', linewidths=2)
    axes[0].set_title(f'Ground Truth ({len(ground_truth)} spots)')
    axes[0].axis('off')

    axes[1].imshow(bayer_img, cmap='gray')
    axes[1].scatter(ground_truth[:, 1], ground_truth[:, 0],
                   s=100, facecolors='none', edgecolors='lime', linewidths=1, alpha=0.5)
    axes[1].scatter(bayer_dets[:, 1], bayer_dets[:, 0],
                   s=60, c='red', marker='x', linewidths=2)
    axes[1].set_title(f'Bayer-Aware ({len(bayer_dets)} detected)')
    axes[1].axis('off')

    axes[2].imshow(gray_img, cmap='gray')
    axes[2].scatter(ground_truth[:, 1], ground_truth[:, 0],
                   s=100, facecolors='none', edgecolors='lime', linewidths=1, alpha=0.5)
    axes[2].scatter(demosaic_dets[:, 1], demosaic_dets[:, 0],
                   s=60, c='blue', marker='+', linewidths=2)
    axes[2].set_title(f'Demosaic ({len(demosaic_dets)} detected)')
    axes[2].axis('off')

    plt.tight_layout()

    save_path = os.path.join(os.path.dirname(__file__), 'bayer_detection_simple_test.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to: {save_path}")

    plt.show()

    # Success if both methods detect at least 70% of spots
    if bayer_matches >= 0.7 * len(ground_truth) and demosaic_matches >= 0.7 * len(ground_truth):
        print("\n✓ PASS: Both methods detect ≥70% of spots")
        return True
    else:
        print("\n✗ FAIL: Detection rate too low")
        return False


def test_parameter_sweep():
    """Test detection performance across different photon counts and backgrounds."""
    print("\n" + "="*60)
    print("Test 3: Parameter Sweep (Photons vs Background)")
    print("="*60)

    # Parameter ranges to test (reduced for faster initial validation)
    photon_counts = [1000, 2000, 4000]
    background_levels = [10, 20]

    # Create a grid of spots for systematic testing
    image_size = (256, 256)
    border = 40

    # 5x5 grid of spots
    x_positions = np.linspace(border, image_size[1] - border, 5)
    y_positions = np.linspace(border, image_size[0] - border, 5)
    xx, yy = np.meshgrid(x_positions, y_positions)
    ground_truth = np.column_stack([yy.ravel(), xx.ravel()])  # [y, x] format

    n_spots = len(ground_truth)
    print(f"\nUsing {n_spots} spots in 5×5 grid")

    spot_detector = SpotDetectionFunctions.SpotDetection_Functions()

    results = []

    for photons in photon_counts:
        for background in background_levels:
            print(f"\nTesting: {photons} photons, {background} background photons/px")

            # Create image
            bayer_img = create_synthetic_bayer_image(
                ground_truth,
                image_size=image_size,
                spot_intensity=photons,
                background=background,
                dye_name='ATTO 565'
            )

            # Bayer-aware detection
            bayer_dets, _ = BayerSpotDetection.detect_spots_bayer_multichannel(
                bayer_img[np.newaxis, :, :],
                spot_detector=spot_detector,
                pattern='RGGB',
                pfa=1e-3,
                sigma=1.5,
                wavelength=0.58,
                pixel_size=0.069,
                NA=1.49
            )

            # Demosaic detection
            gray_img = simple_demosaic(bayer_img)
            demosaic_dets = spot_detector.detect_puncta_in_stack_parallel(
                gray_img[np.newaxis, :, :],
                pfa=1e-3,
                sigma=1.5,
                wavelength=0.58,
                pixel_size=0.069,
                NA=1.49
            )

            # Calculate matches
            def count_matches(detections, threshold=3.0):
                if len(detections) == 0:
                    return 0
                matches = 0
                for gt in ground_truth:
                    distances = np.sqrt(
                        (detections[:, 0] - gt[0])**2 +
                        (detections[:, 1] - gt[1])**2
                    )
                    if np.min(distances) < threshold:
                        matches += 1
                return matches

            bayer_recall = count_matches(bayer_dets) / len(ground_truth)
            demosaic_recall = count_matches(demosaic_dets) / len(ground_truth)

            bayer_precision = count_matches(bayer_dets) / len(bayer_dets) if len(bayer_dets) > 0 else 0
            demosaic_precision = count_matches(demosaic_dets) / len(demosaic_dets) if len(demosaic_dets) > 0 else 0

            results.append({
                'photons': photons,
                'background': background,
                'snr': photons / background,
                'bayer_recall': bayer_recall,
                'demosaic_recall': demosaic_recall,
                'bayer_precision': bayer_precision,
                'demosaic_precision': demosaic_precision,
                'bayer_n_detected': len(bayer_dets),
                'demosaic_n_detected': len(demosaic_dets)
            })

            print(f"  Bayer:    Recall={bayer_recall:.1%}, Precision={bayer_precision:.1%}, N={len(bayer_dets)}")
            print(f"  Demosaic: Recall={demosaic_recall:.1%}, Precision={demosaic_precision:.1%}, N={len(demosaic_dets)}")

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Group by background level
    for bg_level in background_levels:
        bg_results = [r for r in results if r['background'] == bg_level]
        photons_list = [r['photons'] for r in bg_results]

        # Plot recall
        axes[0, 0].plot(photons_list,
                       [r['bayer_recall'] * 100 for r in bg_results],
                       'o-', label=f'Bayer (BG={bg_level})')
        axes[0, 0].plot(photons_list,
                       [r['demosaic_recall'] * 100 for r in bg_results],
                       's--', label=f'Demosaic (BG={bg_level})', alpha=0.7)

        # Plot precision
        axes[0, 1].plot(photons_list,
                       [r['bayer_precision'] * 100 for r in bg_results],
                       'o-', label=f'Bayer (BG={bg_level})')
        axes[0, 1].plot(photons_list,
                       [r['demosaic_precision'] * 100 for r in bg_results],
                       's--', label=f'Demosaic (BG={bg_level})', alpha=0.7)

    # Plot recall vs SNR
    axes[1, 0].plot([r['snr'] for r in results],
                    [r['bayer_recall'] * 100 for r in results],
                    'ro', label='Bayer-Aware', markersize=8)
    axes[1, 0].plot([r['snr'] for r in results],
                    [r['demosaic_recall'] * 100 for r in results],
                    'bs', label='Demosaic', markersize=8, alpha=0.7)

    # Plot recall improvement
    improvements = [(r['bayer_recall'] - r['demosaic_recall']) * 100 for r in results]
    axes[1, 1].plot([r['snr'] for r in results], improvements,
                    'go', markersize=8)
    axes[1, 1].axhline(0, color='k', linestyle='--', alpha=0.3)

    # Configure axes
    axes[0, 0].set_xlabel('Photon Count')
    axes[0, 0].set_ylabel('Recall (%)')
    axes[0, 0].set_title('Recall vs Photon Count')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_xlabel('Photon Count')
    axes[0, 1].set_ylabel('Precision (%)')
    axes[0, 1].set_title('Precision vs Photon Count')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_xlabel('SNR (Photons/Background)')
    axes[1, 0].set_ylabel('Recall (%)')
    axes[1, 0].set_title('Recall vs SNR')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xscale('log')

    axes[1, 1].set_xlabel('SNR (Photons/Background)')
    axes[1, 1].set_ylabel('Recall Improvement (%)')
    axes[1, 1].set_title('Bayer-Aware Advantage')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xscale('log')

    plt.tight_layout()

    save_path = os.path.join(os.path.dirname(__file__), 'bayer_detection_parameter_sweep.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved parameter sweep plot to: {save_path}")

    # Summary statistics
    avg_bayer_recall = np.mean([r['bayer_recall'] for r in results])
    avg_demosaic_recall = np.mean([r['demosaic_recall'] for r in results])
    avg_improvement = (avg_bayer_recall - avg_demosaic_recall) * 100

    print(f"\n{'='*60}")
    print("Parameter Sweep Summary:")
    print(f"{'='*60}")
    print(f"Average Bayer-Aware Recall:   {avg_bayer_recall:.1%}")
    print(f"Average Demosaic Recall:      {avg_demosaic_recall:.1%}")
    print(f"Average Improvement:          {avg_improvement:+.1f}%")
    print(f"{'='*60}")

    return avg_bayer_recall >= avg_demosaic_recall


if __name__ == '__main__':
    print("="*60)
    print("Bayer Spot Detection Simple Validation")
    print("="*60)

    # Run tests
    test1_pass = test_coordinate_mapping()
    test2_pass = test_detection_comparison()
    test3_pass = test_parameter_sweep()

    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print(f"Coordinate Mapping:     {'PASS' if test1_pass else 'FAIL'}")
    print(f"Detection Comparison:   {'PASS' if test2_pass else 'FAIL'}")
    print(f"Parameter Sweep:        {'PASS' if test3_pass else 'FAIL'}")
    print("="*60)

    if test1_pass and test2_pass and test3_pass:
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed")
        sys.exit(1)
