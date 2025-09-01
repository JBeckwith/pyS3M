#!/usr/bin/env python3
"""
Test to verify that bayer_image parameter is properly working in both conditions
"""

import sys
import os

# Add src module directory to path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(src_dir)

try:
    from SpotDetectionValidation import SpotDetectionValidator, ValidationConfig
    
    print("Testing bayer_image parameter functionality...")
    
    # Config that tests both bayer_image conditions
    config = ValidationConfig(
        grid_size=2,  # Small 2x2 grid  
        n_bootstrap=1,  # Single sample for quick test
        n_photons_range=(2000, 3000),  # Reasonable photon count
        test_bayer_processing=True  # TEST BOTH CONDITIONS
    )
    
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    camera_path = os.path.join(base_dir, "Camera_Calibrations", "Ximea_Camera")
    save_folder = os.path.join(base_dir, "validation_results", "bayer_comparison_test")
    
    print(f"Camera path: {camera_path}")
    print(f"Camera path exists: {os.path.exists(camera_path)}")
    
    if not os.path.exists(camera_path):
        print("✗ Camera path does not exist - cannot run validation")
        sys.exit(1)
    
    # Create validator and run test with both bayer conditions
    validator = SpotDetectionValidator(config)
    
    print("Starting validation with both bayer_image=True and bayer_image=False...")
    results = validator.run_validation_bootstrap(camera_path, save_folder)
    
    print(f"✓ Test completed! Results shape: {results.shape}")
    print(f"  Columns: {list(results.columns)}")
    
    if len(results) > 0:
        print(f"\n  Results by bayer_processing condition:")
        for bayer_val in results['bayer_processing'].unique():
            subset = results[results['bayer_processing'] == bayer_val]
            print(f"    bayer_image={bayer_val}: {len(subset)} samples")
            if len(subset) > 0:
                print(f"      Mean precision: {subset['precision'].mean():.3f}")
                print(f"      Mean recall: {subset['recall'].mean():.3f}")
                print(f"      Mean detected spots: {subset['n_detected'].mean():.1f}")
    
    print(f"\n✓ Bayer processing comparison test completed successfully!")
    print(f"  Both bayer_image=True and bayer_image=False conditions were tested")
    
except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)