#!/usr/bin/env python3
"""
Test script for SpotDetectionValidation.py

Quick test to validate the spot detection validation framework
before running the full bootstrap analysis.
"""

import os
import sys
import numpy as np
import warnings

# Add src module directory to path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(src_dir)
# Also add current directory for SpotDetectionValidation import
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)

# Activate virtual environment
try:
    import subprocess

    result = subprocess.run(
        ["workon", "pyBayerSMLM"], capture_output=True, text=True, shell=True
    )
    print("Virtual environment activation result:", result.returncode)
except Exception as e:
    print(f"Warning: Could not activate virtual environment: {e}")

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

try:
    from SpotDetectionValidation import SpotDetectionValidator, ValidationConfig

    print("✓ Successfully imported SpotDetectionValidation")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)


def test_single_simulation():
    """Test a single simulation run to validate the pipeline."""
    print("\n" + "=" * 60)
    print("TESTING SINGLE SPOT DETECTION SIMULATION")
    print("=" * 60)

    # Create minimal configuration for testing
    config = ValidationConfig(
        grid_size=3,  # Small 3x3 grid for testing
        grid_spacing_microns=1.0,
        image_size_pixels=100,  # Small image
        n_photons_range=(2000, 3000),
        n_bootstrap=1,  # Single test
        pfa=1e-3,  # Less stringent for testing
        detection_tolerance_nm=200.0,
    )

    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    camera_path = os.path.join(base_dir, "Camera_Calibrations", "Ximea_Camera")

    print(f"Camera calibration path: {camera_path}")
    print(f"Camera path exists: {os.path.exists(camera_path)}")

    if not os.path.exists(camera_path):
        print("✗ Camera calibration path does not exist!")
        return False

    # List calibration files
    calib_files = os.listdir(camera_path)
    print(f"Calibration files found: {calib_files}")

    expected_files = [
        "gain.tif",
        "offset.tif",
        "variance.tif",
        "readnoise.tif",
        "rqe.tif",
    ]
    missing_files = [f for f in expected_files if f not in calib_files]
    if missing_files:
        print(f"✗ Missing calibration files: {missing_files}")
        return False

    print("✓ All calibration files found")

    # Create validator
    try:
        validator = SpotDetectionValidator(config)
        print("✓ SpotDetectionValidator created successfully")
    except Exception as e:
        print(f"✗ Error creating validator: {e}")
        return False

    # Test camera calibration loading
    try:
        camera_params = validator.load_camera_calibration(camera_path)
        print("✓ Camera calibration loaded successfully")
        print(f"  Camera dimensions: {camera_params['full_dimensions']}")
        print(f"  Gain map shape: {camera_params['gain'].shape}")
    except Exception as e:
        print(f"✗ Error loading camera calibration: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test grid position generation
    try:
        ground_truth_positions = validator.generate_grid_positions()
        print("✓ Ground truth positions generated")
        print(f"  Grid positions shape: {ground_truth_positions.shape}")
        print(
            f"  Position range: [{ground_truth_positions.min():.1f}, {ground_truth_positions.max():.1f}] nm"
        )
    except Exception as e:
        print(f"✗ Error generating grid positions: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test camera region extraction
    try:
        camera_region = validator.extract_camera_region(100, 100)
        print("✓ Camera region extracted")
        print(f"  Region gain shape: {camera_region['gain'].shape}")
    except Exception as e:
        print(f"✗ Error extracting camera region: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test spectral data loading
    try:
        spectral_data = validator._load_spectral_data()
        print("✓ Spectral data loaded")
        wavelength, dye_spectrum, absolute_QYs, avg_wl, dye_efficiency = spectral_data
        print(f"  Wavelength range: {wavelength.min():.0f}-{wavelength.max():.0f} nm")
        print(f"  Average emission wavelength: {avg_wl:.1f} nm")
    except Exception as e:
        print(f"✗ Error loading spectral data: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test camera image simulation
    try:
        bayer_image, photoelectron_image = validator.simulate_camera_image(
            ground_truth_positions, camera_region, 2500
        )
        print("✓ Camera image simulated")
        print(f"  Bayer image shape: {bayer_image.shape}")
        print(f"  Photoelectron image shape: {photoelectron_image.shape}")
        print(
            f"  Photoelectron image range: [{photoelectron_image.min():.1f}, {photoelectron_image.max():.1f}]"
        )
    except Exception as e:
        print(f"✗ Error simulating camera image: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test spot detection with both Bayer processing conditions
    try:
        # Test with Bayer averaging
        detected_positions_bayer = validator.detect_spots(
            photoelectron_image, camera_region, bayer_processing=True
        )
        print("✓ Spot detection with Bayer averaging completed")
        print(f"  Ground truth spots: {len(ground_truth_positions)}")
        print(f"  Detected spots (Bayer=True): {len(detected_positions_bayer)}")

        # Test without Bayer averaging
        detected_positions_no_bayer = validator.detect_spots(
            photoelectron_image, camera_region, bayer_processing=False
        )
        print("✓ Spot detection without Bayer averaging completed")
        print(f"  Detected spots (Bayer=False): {len(detected_positions_no_bayer)}")

        if len(detected_positions_bayer) > 0:
            print(
                f"  Bayer=True position range: [{detected_positions_bayer.min():.1f}, {detected_positions_bayer.max():.1f}] pixels"
            )
        if len(detected_positions_no_bayer) > 0:
            print(
                f"  Bayer=False position range: [{detected_positions_no_bayer.min():.1f}, {detected_positions_no_bayer.max():.1f}] pixels"
            )

    except Exception as e:
        print(f"✗ Error in spot detection: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test performance evaluation for both conditions
    try:
        metrics_bayer = validator.evaluate_detection_performance(
            detected_positions_bayer, ground_truth_positions, bayer_processing=True
        )
        metrics_no_bayer = validator.evaluate_detection_performance(
            detected_positions_no_bayer, ground_truth_positions, bayer_processing=False
        )
        print("✓ Performance evaluation completed")
        print(f"  BAYER AVERAGING (True):")
        print(f"    True positives: {metrics_bayer.true_positives}")
        print(f"    False positives: {metrics_bayer.false_positives}")
        print(f"    False negatives: {metrics_bayer.false_negatives}")
        print(f"    Precision: {metrics_bayer.precision:.3f}")
        print(f"    Recall: {metrics_bayer.recall:.3f}")
        print(f"    F1-score: {metrics_bayer.f1_score:.3f}")
        print(f"  NO BAYER AVERAGING (False):")
        print(f"    True positives: {metrics_no_bayer.true_positives}")
        print(f"    False positives: {metrics_no_bayer.false_positives}")
        print(f"    False negatives: {metrics_no_bayer.false_negatives}")
        print(f"    Precision: {metrics_no_bayer.precision:.3f}")
        print(f"    Recall: {metrics_no_bayer.recall:.3f}")
        print(f"    F1-score: {metrics_no_bayer.f1_score:.3f}")
    except Exception as e:
        print(f"✗ Error in performance evaluation: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n✓ Single simulation test completed successfully!")
    return True


def test_mini_bootstrap():
    """Test a mini bootstrap run with 5 samples."""
    print("\n" + "=" * 60)
    print("TESTING MINI BOOTSTRAP RUN")
    print("=" * 60)

    # Create configuration for mini bootstrap
    config = ValidationConfig(
        grid_size=5,  # 5x5 grid
        grid_spacing_microns=1.0,
        image_size_pixels=120,
        n_photons_range=(1500, 4000),
        n_bootstrap=5,  # Small bootstrap
        pfa=1e-4,
        detection_tolerance_nm=150.0,
        test_bayer_processing=True,  # Test both Bayer conditions
    )

    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    camera_path = os.path.join(base_dir, "Camera_Calibrations", "Ximea_Camera")
    save_folder = os.path.join(base_dir, "validation_results", "spot_detection_test")

    # Create validator and run mini bootstrap
    try:
        validator = SpotDetectionValidator(config)
        results = validator.run_validation_bootstrap(camera_path, save_folder)
        print("✓ Mini bootstrap completed successfully!")
        print(f"  Results saved to: {save_folder}")
        print(f"  Number of samples: {len(results)}")

        # Show summary statistics
        if len(results) > 0:
            print(f"  Mean precision: {results['precision'].mean():.3f}")
            print(f"  Mean recall: {results['recall'].mean():.3f}")
            print(f"  Mean F1-score: {results['f1_score'].mean():.3f}")
            print(
                f"  Mean false positive rate: {results['false_positive_rate'].mean():.4f}"
            )

        return True

    except Exception as e:
        print(f"✗ Error in mini bootstrap: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("SPOT DETECTION VALIDATION TEST SUITE")
    print("Starting validation tests...")

    # Test 1: Single simulation
    test1_passed = test_single_simulation()

    if not test1_passed:
        print("\n✗ Single simulation test failed. Stopping here.")
        return False

    # Test 2: Mini bootstrap
    test2_passed = test_mini_bootstrap()

    if test1_passed and test2_passed:
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("The SpotDetectionValidation framework is ready for full runs.")
        print("=" * 60)
        return True
    else:
        print("\n" + "=" * 60)
        print("❌ SOME TESTS FAILED")
        print("Please fix issues before running full validation.")
        print("=" * 60)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
