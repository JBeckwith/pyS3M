#!/usr/bin/env python3
"""
Test script for the unified DriftCorrectionFunctions module.

Demonstrates usage of the strategy pattern for drift correction
combining RCC and AIM approaches.
"""

import numpy as np
import sys
import os

# Add src directory to path
from pathlib import Path
project_root = Path(__file__).parent.parent
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

from DriftCorrectionFunctions import (
    Drift_Correction_Functions,
    DriftMethod, 
    DriftParameters,
    DriftCorrectionFactory
)


def create_test_data():
    """Create synthetic localization data for testing."""
    
    # Generate synthetic localizations with artificial drift
    np.random.seed(42)
    n_locs = 5000
    n_frames = 1000
    
    # Base coordinates
    x_base = np.random.uniform(0, 100, n_locs)
    y_base = np.random.uniform(0, 100, n_locs) 
    frames = np.random.randint(1, n_frames + 1, n_locs)
    
    # Add artificial drift (sinusoidal)
    drift_x = 2 * np.sin(2 * np.pi * frames / 200)  # 2 pixel amplitude
    drift_y = 1.5 * np.cos(2 * np.pi * frames / 300)  # 1.5 pixel amplitude
    
    x_drifted = x_base + drift_x
    y_drifted = y_base + drift_y
    
    # Create record array
    locs = np.rec.array(
        (x_drifted, y_drifted, frames, np.ones(n_locs) * 1000),
        dtype=[("xc", "f4"), ("yc", "f4"), ("frame", "i4"), ("photons", "f4")]
    )
    
    # Create metadata
    info = [{
        "Width": 100.0,
        "Height": 100.0, 
        "Frames": float(n_frames),
        "Pixelsize": 100.0  # nm per pixel
    }]
    
    return locs, info, (drift_x, drift_y)


def test_drift_correction_factory():
    """Test the DriftCorrectionFactory."""
    print("=== Testing DriftCorrectionFactory ===")
    
    # Test available methods
    methods = DriftCorrectionFactory.available_methods()
    print(f"Available methods: {[m.value for m in methods]}")
    
    # Test creating correctors
    for method in methods:
        corrector = DriftCorrectionFactory.create_corrector(method)
        print(f"{method.value}: {corrector.__class__.__name__} (3D: {corrector.supports_3d()})")
    
    print()


def test_main_interface():
    """Test the main Drift_Correction_Functions interface."""
    print("=== Testing Main Interface ===")
    
    # Create test data
    locs, info, true_drift = create_test_data()
    print(f"Created {len(locs)} synthetic localizations with artificial drift")
    
    # Initialize drift corrector
    DCF = Drift_Correction_Functions()
    print(f"Available methods: {DCF.available_methods()}")
    
    # Test method info
    for method in DCF.available_methods():
        info_dict = DCF.method_info(method)
        print(f"Method '{method}': {info_dict['description']}")
    
    print()


def test_parameter_validation():
    """Test parameter validation."""
    print("=== Testing Parameter Validation ===")
    
    # Test valid parameters
    try:
        params = DriftParameters(segmentation=50, intersect_d=0.5, roi_r=1.0)
        params.validate()
        print("✓ Valid parameters accepted")
    except Exception as e:
        print(f"✗ Valid parameters rejected: {e}")
    
    # Test invalid parameters
    try:
        params = DriftParameters(segmentation=-10)
        params.validate()
        print("✗ Invalid segmentation accepted")
    except Exception as e:
        print(f"✓ Invalid segmentation rejected: {e}")
    
    try:
        params = DriftParameters(intersect_d=-1.0)
        params.validate()
        print("✗ Invalid intersect_d accepted") 
    except Exception as e:
        print(f"✓ Invalid intersect_d rejected: {e}")
    
    print()


def test_rcc_method():
    """Test RCC method (if available)."""
    print("=== Testing RCC Method ===")
    
    try:
        locs, info, true_drift = create_test_data()
        DCF = Drift_Correction_Functions()
        
        # Try RCC correction
        corrected_locs, drift_result = DCF.undrift(
            locs.copy(), info, method="rcc", segmentation=100
        )
        
        print(f"✓ RCC correction completed")
        print(f"  Method used: {drift_result.method_used.value}")
        print(f"  Drift range X: [{drift_result.drift_x.min():.2f}, {drift_result.drift_x.max():.2f}]")
        print(f"  Drift range Y: [{drift_result.drift_y.min():.2f}, {drift_result.drift_y.max():.2f}]")
        print(f"  Metadata: {drift_result.metadata}")
        
    except Exception as e:
        import traceback
        print(f"✗ RCC method failed: {e}")
        print(f"  Details: {traceback.format_exc()}")
        
    print()


def test_aim_method():
    """Test AIM method."""
    print("=== Testing AIM Method ===")
    
    try:
        locs, info, true_drift = create_test_data()
        DCF = Drift_Correction_Functions()
        
        # Try AIM correction
        corrected_locs, drift_result = DCF.undrift(
            locs.copy(), info, method="aim", 
            segmentation=100, intersect_d=0.3, roi_r=1.0
        )
        
        print(f"✓ AIM correction completed")
        print(f"  Method used: {drift_result.method_used.value}")
        print(f"  Drift range X: [{drift_result.drift_x.min():.2f}, {drift_result.drift_x.max():.2f}]")
        print(f"  Drift range Y: [{drift_result.drift_y.min():.2f}, {drift_result.drift_y.max():.2f}]")
        print(f"  Metadata: {drift_result.metadata}")
        
    except Exception as e:
        print(f"✗ AIM method failed: {e}")
        
    print()


def test_auto_method():
    """Test automatic method selection."""
    print("=== Testing Auto Method Selection ===")
    
    try:
        locs, info, true_drift = create_test_data()
        DCF = Drift_Correction_Functions()
        
        # Try auto correction
        corrected_locs, drift_result = DCF.undrift(
            locs.copy(), info, method="auto", segmentation=100
        )
        
        print(f"✓ Auto correction completed")
        print(f"  Method selected: {drift_result.method_used.value}")
        print(f"  Selection reason: {drift_result.metadata.get('auto_selection_reason', 'N/A')}")
        print(f"  Drift range X: [{drift_result.drift_x.min():.2f}, {drift_result.drift_x.max():.2f}]")
        print(f"  Drift range Y: [{drift_result.drift_y.min():.2f}, {drift_result.drift_y.max():.2f}]")
        
    except Exception as e:
        print(f"✗ Auto method failed: {e}")
        
    print()


def test_backward_compatibility():
    """Test backward compatibility functions."""
    print("=== Testing Backward Compatibility ===")
    
    try:
        from DriftCorrectionFunctions import undrift_rcc, undrift_aim, undrift_auto
        
        locs, info, true_drift = create_test_data()
        
        # Test convenience functions
        print("Testing undrift_auto convenience function...")
        corrected_locs, drift_result = undrift_auto(locs.copy(), info, segmentation=100)
        print(f"✓ undrift_auto: {drift_result.method_used.value} method selected")
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
    except Exception as e:
        print(f"✗ Backward compatibility test failed: {e}")
        
    print()


def main():
    """Run all tests."""
    print("Testing Unified Drift Correction Module")
    print("=" * 50)
    
    test_drift_correction_factory()
    test_main_interface() 
    test_parameter_validation()
    test_rcc_method()
    test_aim_method()
    test_auto_method()
    test_backward_compatibility()
    
    print("Testing completed!")


if __name__ == "__main__":
    main()