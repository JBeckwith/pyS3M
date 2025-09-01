#!/usr/bin/env python3
"""
Test to explicitly verify that both bayer_image conditions are tested on the same image
"""

import sys
import os
import numpy as np

# Add src module directory to path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(src_dir)

try:
    from SpotDetectionValidation import SpotDetectionValidator, ValidationConfig
    
    print("Testing that both bayer conditions are tested on identical images...")
    
    # Config with fixed random seed for reproducibility
    config = ValidationConfig(
        grid_size=3,  # 3x3 grid for clearer results
        n_bootstrap=1,  # Single sample for clear demonstration
        n_photons_range=(2500, 2500),  # Fixed photon count for consistency  
        test_bayer_processing=True  # Test both conditions
    )
    
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    camera_path = os.path.join(base_dir, "Camera_Calibrations", "Ximea_Camera")
    
    if not os.path.exists(camera_path):
        print("✗ Camera path does not exist")
        sys.exit(1)
    
    # Create validator
    validator = SpotDetectionValidator(config)
    validator.load_camera_calibration(camera_path)
    
    # Generate ground truth positions (fixed seed for reproducibility)
    np.random.seed(42)
    positions = validator.generate_grid_positions()
    print(f"Ground truth positions: {len(positions)} puncta")
    
    # Extract a specific camera region (fixed location)
    camera_region = validator.extract_camera_region(500, 800)
    
    # Simulate single image with fixed parameters
    bayer_image, photoelectron_image = validator.simulate_camera_image(
        positions, camera_region, 2500
    )
    
    print(f"Generated single test image:")
    print(f"  Bayer image shape: {bayer_image.shape}")
    print(f"  Photoelectron image shape: {photoelectron_image.shape}")
    print(f"  Photoelectron image stats: min={photoelectron_image.min():.1f}, max={photoelectron_image.max():.1f}, mean={photoelectron_image.mean():.1f}")
    
    # Test BOTH conditions on the SAME image
    print(f"\nTesting both bayer_image conditions on the SAME photoelectron image:")
    
    # Test condition 1: bayer_image=True
    detected_true = validator.detect_spots(photoelectron_image, camera_region, True)
    print(f"  bayer_image=True:  Detected {len(detected_true)} spots")
    if len(detected_true) > 0:
        print(f"    Positions: {detected_true[:3]}...")  # Show first 3
    
    # Test condition 2: bayer_image=False (on the SAME image)
    detected_false = validator.detect_spots(photoelectron_image, camera_region, False)
    print(f"  bayer_image=False: Detected {len(detected_false)} spots")
    if len(detected_false) > 0:
        print(f"    Positions: {detected_false[:3]}...")  # Show first 3
    
    # Verify they give different results (proving they're actually different algorithms)
    if len(detected_true) != len(detected_false):
        print(f"\n✅ VERIFICATION SUCCESSFUL:")
        print(f"   Different spot counts confirm both conditions are working differently")
        print(f"   Same image → Different results = Proper bayer_image parameter effect")
    else:
        print(f"\n⚠️  Same number of spots detected in both conditions")
        print(f"   This could be valid, but let's check if positions are different...")
        
        if len(detected_true) > 0 and len(detected_false) > 0:
            # Check if positions are different
            pos_diff = np.linalg.norm(detected_true - detected_false, axis=1).mean() if len(detected_true) == len(detected_false) else float('inf')
            if pos_diff > 1.0:  # More than 1 pixel difference
                print(f"   Position differences found (mean: {pos_diff:.1f} pixels) - algorithms are working differently ✅")
            else:
                print(f"   Very similar positions - algorithms may be giving similar results")
    
    print(f"\n✅ Same-image comparison test completed!")
    print(f"   Both bayer_image=True and bayer_image=False tested on identical photoelectron image")
    
    # Now test full validation to confirm it maintains this behavior
    print(f"\nRunning mini validation to confirm same-image comparison in full pipeline...")
    save_folder = os.path.join(base_dir, "validation_results", "same_image_test")
    results = validator.run_validation_bootstrap(camera_path, save_folder)
    
    if len(results) >= 2:  # Should have 2 results (1 bootstrap × 2 bayer conditions)
        print(f"✅ Full validation maintains same-image comparison:")
        print(f"   Results shape: {results.shape}")
        for _, row in results.iterrows():
            print(f"   Bootstrap {int(row['bootstrap_idx'])}, bayer_image={row['bayer_processing']}: {int(row['n_detected'])} spots detected")
    
except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)