#!/usr/bin/env python
"""
Example: Fit Nile Red wavelengths from HDF5 localization file

This script demonstrates how to use the fit_wavelengths_from_h5() convenience
function to extract Nile Red emission wavelengths from existing localization data.

Usage:
    python fit_nile_red_from_h5.py <input.h5> <output.h5>

Author: Claude Code (Anthropic)
Date: October 21, 2025
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import NileRedFunctions
import SpectralFunctions


def main():
    """Main function to fit Nile Red wavelengths from HDF5 file."""

    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python fit_nile_red_from_h5.py <input.h5> [output.h5]")
        print("\nExample:")
        print("  python fit_nile_red_from_h5.py results.h5 results_with_wavelengths.h5")
        sys.exit(1)

    input_h5 = sys.argv[1]
    output_h5 = sys.argv[2] if len(sys.argv) > 2 else None

    # Check input file exists
    if not os.path.exists(input_h5):
        print(f"Error: Input file not found: {input_h5}")
        sys.exit(1)

    # Initialize Nile Red functions
    nrf = NileRedFunctions.NileRed_Functions()

    # Define optical configuration
    # These are the default Nile Red filters - adjust as needed for your setup
    filter_names = [
        "semrock-ff01-650-200",  # Emission filter
        "semrock-di03-r514-t1-25x36",  # Dichroic
        "semrock-ff01-515-lp",  # Long-pass filter
    ]

    # Setup camera parameters with pixel quantum efficiencies
    print("Loading pixel quantum efficiencies...")
    sf = SpectralFunctions.Spectral_Funcs()
    R, G, B, wavelength = sf.getpixelefficiency()
    pixel_QYs = np.vstack([B, G, R])

    camera_parameters = {
        "pixel_QYs": pixel_QYs,
        "wavelength": wavelength,
    }

    # Fit wavelengths from HDF5 file
    print(f"\nProcessing: {input_h5}")
    print(f"Output: {output_h5 if output_h5 else '(not saving)'}")

    df_with_wavelengths = nrf.fit_wavelengths_from_h5(
        h5_path=input_h5,
        filter_names=filter_names,
        camera_parameters=camera_parameters,
        wavelength_bounds=(500.0, 750.0),  # Nile Red emission range
        NA=1.49,  # Numerical aperture - adjust for your objective
        pixel_size=69.0,  # Camera pixel size in nm - adjust for your camera
        output_path=output_h5,
        cpu_fraction=0.9,  # Use 90% of available CPUs
        verbose=True,
    )

    print("\nDone! Wavelength columns added:")
    print("  - wl_fit: Fitted wavelength (nm)")
    print("  - wl_fit_err: Wavelength fit error (nm)")

    # Show sample of results
    if len(df_with_wavelengths) > 0:
        print("\nSample of fitted wavelengths:")
        print(
            df_with_wavelengths[
                ["A_R", "A_G", "A_B", "s_x", "s_y", "photons", "wl_fit"]
            ].head(10)
        )


if __name__ == "__main__":
    main()
