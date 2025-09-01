#!/usr/bin/env python3
"""
Test camera image simulation to verify pixel_QYs fix works
"""

import sys
import os
import numpy as np

# Add src module directory to path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(src_dir)

try:
    from SpotDetectionValidation import SpotDetectionValidator, ValidationConfig
    
    print("Testing camera image simulation fix...")
    
    # Minimal config for testing
    config = ValidationConfig(
        grid_size=2,  # Just 2x2 grid for quick test
        image_size_pixels=50,  # Small image
        n_photons_range=(1000, 2000)
    )
    
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    camera_path = os.path.join(base_dir, "Camera_Calibrations", "Ximea_Camera")
    
    if not os.path.exists(camera_path):
        print("✗ Camera path does not exist - cannot run simulation test")
        sys.exit(1)
    
    # Create validator
    validator = SpotDetectionValidator(config)
    
    # Load camera calibration
    validator.load_camera_calibration(camera_path)
    
    # Generate small grid positions (2x2 = 4 puncta)
    positions = validator.generate_grid_positions()
    print(f"Generated {len(positions)} puncta positions")
    
    # Extract a small camera region for testing
    camera_region = validator.extract_camera_region(100, 100)  # Start at row 100, col 100
    
    # Check if pixel_QYs is properly set
    if "pixel_QYs" not in camera_region:
        print("✗ pixel_QYs missing from camera region")
        sys.exit(1)
        
    print(f"✓ pixel_QYs shape: {camera_region['pixel_QYs'].shape}")
    print(f"✓ pixel_order: {camera_region['pixel_order']}")
    
    # Test image simulation
    try:
        bayer_image, photoelectron_image = validator.simulate_camera_image(
            positions, camera_region, 2000  # 2000 photons per punctum
        )
        
        print(f"✓ Camera image simulation successful!")
        print(f"  Bayer image shape: {bayer_image.shape}")
        print(f"  Photoelectron image shape: {photoelectron_image.shape}")
        print(f"  Bayer image range: {bayer_image.min():.1f} - {bayer_image.max():.1f}")
        print(f"  Photoelectron image range: {photoelectron_image.min():.1f} - {photoelectron_image.max():.1f}")
        
        print("\n✓ All tests passed! Camera simulation now works correctly.")
        
    except Exception as e:
        print(f"✗ Camera simulation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)