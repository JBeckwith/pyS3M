#!/usr/bin/env python3
"""
Quick test to verify the spectral data loading fix works correctly.
"""

import sys
import os

# Add src module directory to path
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(src_dir)

# Test spectral data loading
try:
    from SpotDetectionValidation import SpotDetectionValidator, ValidationConfig

    print("Testing spectral data loading fix...")

    # Create validator with minimal config
    config = ValidationConfig(
        grid_size=2,  # Small grid for quick test
        n_bootstrap=1,  # Single bootstrap sample
        test_bayer_processing=False,  # Single condition only
    )

    validator = SpotDetectionValidator(config)

    # Test spectral data loading
    spectral_data = validator._load_spectral_data()
    wavelength, dye_spectrum, absolute_QYs, avg_wavelength, dye_efficiency = (
        spectral_data
    )

    print(f"✓ Spectral data loaded successfully")
    print(
        f"  Wavelength range: {wavelength.min():.0f}-{wavelength.max():.0f} nm ({len(wavelength)} points)"
    )
    print(f"  Average emission wavelength: {avg_wavelength:.1f} nm")
    print(f"  Dye spectrum shape: {dye_spectrum.shape}")
    print(f"  Camera QE shape: {absolute_QYs.shape}")
    print(f"  Dye efficiency shape: {dye_efficiency.shape}")

    print("\n✓ All tests passed! SpotDetectionValidation should now work correctly.")

except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
