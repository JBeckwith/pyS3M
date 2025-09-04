#!/usr/bin/env python3
"""
Quick smoke test for spot detection validation - just test first couple samples
"""

import sys
import os

# Add src module directory to path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(src_dir)

try:
    from SpotDetectionValidation import SpotDetectionValidator, ValidationConfig

    print("Running quick smoke test...")

    # Minimal config for testing
    config = ValidationConfig(
        grid_size=3,  # Small 3x3 grid
        n_bootstrap=2,  # Just 2 samples
        n_photons_range=(2000, 3000),  # Reasonable photon count
        test_bayer_processing=False,  # Single condition to speed up
    )

    # Paths (using same structure as main)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    camera_path = os.path.join(base_dir, "Camera_Calibrations", "Ximea_Camera")
    save_folder = os.path.join(base_dir, "validation_results", "spot_detection_test")

    print(f"Camera path: {camera_path}")
    print(f"Camera path exists: {os.path.exists(camera_path)}")

    if not os.path.exists(camera_path):
        print("✗ Camera path does not exist - cannot run validation")
        sys.exit(1)

    # Create validator and run quick test
    validator = SpotDetectionValidator(config)

    print("Starting validation test...")
    results = validator.run_validation_bootstrap(camera_path, save_folder)

    print(f"✓ Test completed! Results shape: {results.shape}")
    print(f"  Columns: {list(results.columns)}")

    if len(results) > 0:
        print(f"  Sample results:")
        print(results.head())

except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
