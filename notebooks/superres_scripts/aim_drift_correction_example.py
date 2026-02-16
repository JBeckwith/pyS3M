#!/usr/bin/env python3
"""
AIM Drift Correction Example Script

This example demonstrates how to perform drift correction using the AIM (Adaptive 
Intersection Maximization) method with the new DriftCorrectionFunctions.py module.

AIM drift correction is particularly effective for:
- Dense localizations where correlation methods may struggle
- Data with heterogeneous labeling patterns
- Super-resolution datasets with complex structures

Author: Claude Code Assistant
Created: September 3, 2025
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

try:
    import DriftCorrectionFunctions as DCF
    import IOFunctions
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the pyBayerSMLM directory with src/ available")
    sys.exit(1)


def load_example_data() -> Tuple[np.recarray, list]:
    """
    Load example localization data for drift correction.

    In practice, you would load your own localization data using:
    - IOFunctions.IO_Functions().read_localisations() for CSV files
    - pandas.read_hdf() for HDF5 files
    - Or your preferred data loading method

    Returns:
        Tuple of (localizations, metadata_info)
    """
    # Create synthetic example data with simulated drift
    np.random.seed(42)  # For reproducible results

    n_frames = 1000
    n_locs_per_frame = 50
    total_locs = n_frames * n_locs_per_frame

    # Simulate drift pattern (sinusoidal + linear trend)
    frames = np.arange(n_frames)
    true_drift_x = (
        2.0 * np.sin(frames / 100) + 0.01 * frames
    )  # 2 pixel sine wave + linear
    true_drift_y = (
        1.5 * np.cos(frames / 80) + 0.005 * frames
    )  # 1.5 pixel cosine + linear

    # Create localization data
    xc = []
    yc = []
    frame_list = []

    for frame in range(n_frames):
        # Base coordinates (simulate clustered structures)
        base_x = np.random.choice(
            [20, 40, 60, 80], n_locs_per_frame
        ) + np.random.normal(0, 2, n_locs_per_frame)
        base_y = np.random.choice(
            [15, 35, 55, 75], n_locs_per_frame
        ) + np.random.normal(0, 2, n_locs_per_frame)

        # Add drift
        drifted_x = base_x + true_drift_x[frame]
        drifted_y = base_y + true_drift_y[frame]

        # Add localization precision noise
        final_x = drifted_x + np.random.normal(
            0, 0.02, n_locs_per_frame
        )  # 20nm precision
        final_y = drifted_y + np.random.normal(
            0, 0.02, n_locs_per_frame
        )  # 20nm precision

        xc.extend(final_x)
        yc.extend(final_y)
        frame_list.extend([frame] * n_locs_per_frame)

    # Create record array in the format expected by DriftCorrectionFunctions
    locs = np.rec.array(
        (
            np.array(xc),
            np.array(yc),
            np.array(frame_list),
            np.random.exponential(1000, total_locs),  # photons
        ),
        dtype=[("xc", "f"), ("yc", "f"), ("frame", "i"), ("photons", "f")],
    )

    # Create metadata info (required for drift correction)
    info = [
        {
            "Width": 100,  # Image width in pixels
            "Height": 100,  # Image height in pixels
            "Frames": n_frames,  # Total number of frames
            "Pixelsize": 0.1,  # Pixel size in micrometers (100nm pixels)
        }
    ]

    print(
        f"✅ Created example data: {len(locs)} localizations across {n_frames} frames"
    )
    print(
        f"   True drift range: X = {true_drift_x.min():.2f} to {true_drift_x.max():.2f} pixels"
    )
    print(
        f"                     Y = {true_drift_y.min():.2f} to {true_drift_y.max():.2f} pixels"
    )

    return locs, info


def perform_aim_drift_correction(
    locs: np.recarray,
    info: list,
    segmentation: int = 100,
    intersect_d: float = 20 / 69,
    roi_r: float = 60 / 69,
    display: bool = True,
) -> Tuple[np.recarray, DCF.DriftResult]:
    """
    Perform AIM drift correction with customizable parameters.

    Args:
        locs: Localization data
        info: Metadata info list
        segmentation: Number of frames per drift segment (default: 100)
        intersect_d: Intersection distance threshold in camera pixels (default: 20/69 ≈ 0.29)
        roi_r: Search region radius in camera pixels (default: 60/69 ≈ 0.87)
        display: Whether to show drift plots (default: True)

    Returns:
        Tuple of (corrected_localizations, drift_result)
    """
    print("\n" + "=" * 60)
    print("PERFORMING AIM DRIFT CORRECTION")
    print("=" * 60)

    # Initialize drift correction functions
    drift_corrector = DCF.Drift_Correction_Functions()

    # Show available methods
    print(f"Available drift correction methods: {drift_corrector.available_methods()}")

    # Perform AIM drift correction with custom parameters
    print(f"\nAIM Parameters:")
    print(f"  - Segmentation: {segmentation} frames per segment")
    print(f"  - Intersection distance: {intersect_d:.3f} camera pixels")
    print(f"  - ROI radius: {roi_r:.3f} camera pixels")
    print(f"  - Display plots: {display}")

    corrected_locs, drift_result = drift_corrector.undrift(
        locs=locs,
        info=info,
        method="aim",  # Use AIM method
        segmentation=segmentation,  # Frames per drift segment
        intersect_d=intersect_d,  # Intersection distance threshold
        roi_r=roi_r,  # Search region radius
        display=display,  # Show drift plots
    )

    print(f"\n✅ Drift correction completed!")
    print(f"   Method used: {drift_result.method}")
    print(f"   Frames processed: {len(drift_result.drift_x)}")
    print(
        f"   X drift range: {drift_result.drift_x.min():.3f} to {drift_result.drift_x.max():.3f} pixels"
    )
    print(
        f"   Y drift range: {drift_result.drift_y.min():.3f} to {drift_result.drift_y.max():.3f} pixels"
    )

    return corrected_locs, drift_result


def analyze_correction_quality(
    original_locs: np.recarray,
    corrected_locs: np.recarray,
    drift_result: DCF.DriftResult,
) -> Dict[str, float]:
    """
    Analyze the quality of drift correction by comparing before/after statistics.

    Args:
        original_locs: Original localization data
        corrected_locs: Drift-corrected localization data
        drift_result: Drift correction results

    Returns:
        Dictionary with quality metrics
    """
    print("\n" + "=" * 60)
    print("ANALYZING CORRECTION QUALITY")
    print("=" * 60)

    # Calculate coordinate standard deviations as measure of tightness
    orig_std_x = np.std(original_locs.xc)
    orig_std_y = np.std(original_locs.yc)
    corr_std_x = np.std(corrected_locs.xc)
    corr_std_y = np.std(corrected_locs.yc)

    # Calculate improvement ratios
    improvement_x = orig_std_x / corr_std_x
    improvement_y = orig_std_y / corr_std_y

    metrics = {
        "original_std_x": orig_std_x,
        "original_std_y": orig_std_y,
        "corrected_std_x": corr_std_x,
        "corrected_std_y": corr_std_y,
        "improvement_x": improvement_x,
        "improvement_y": improvement_y,
        "drift_magnitude": np.sqrt(
            drift_result.drift_x**2 + drift_result.drift_y**2
        ).max(),
    }

    print(f"Coordinate spread before correction:")
    print(f"  X std: {orig_std_x:.3f} pixels")
    print(f"  Y std: {orig_std_y:.3f} pixels")

    print(f"\nCoordinate spread after correction:")
    print(f"  X std: {corr_std_x:.3f} pixels")
    print(f"  Y std: {corr_std_y:.3f} pixels")

    print(f"\nImprovement factors:")
    print(f"  X improvement: {improvement_x:.2f}x")
    print(f"  Y improvement: {improvement_y:.2f}x")
    print(f"  Maximum drift magnitude: {metrics['drift_magnitude']:.3f} pixels")

    return metrics


def save_results(
    corrected_locs: np.recarray, drift_result: DCF.DriftResult, output_path: str = None
):
    """
    Save drift correction results to files.

    Args:
        corrected_locs: Corrected localization data
        drift_result: Drift correction results
        output_path: Base path for output files (optional)
    """
    if output_path is None:
        output_path = "/tmp/aim_drift_correction_results"

    print(f"\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    # Save corrected localizations as CSV
    corrected_df = pd.DataFrame(corrected_locs)
    csv_path = f"{output_path}_corrected_locs.csv"
    corrected_df.to_csv(csv_path, index=False)
    print(f"✅ Corrected localizations saved to: {csv_path}")

    # Save drift data
    drift_df = pd.DataFrame(
        {
            "frame": np.arange(len(drift_result.drift_x)),
            "drift_x_pixels": drift_result.drift_x,
            "drift_y_pixels": drift_result.drift_y,
        }
    )
    drift_path = f"{output_path}_drift_trace.csv"
    drift_df.to_csv(drift_path, index=False)
    print(f"✅ Drift trace saved to: {drift_path}")

    return csv_path, drift_path


def plot_before_after_comparison(
    original_locs: np.recarray,
    corrected_locs: np.recarray,
    drift_result: DCF.DriftResult,
):
    """
    Create before/after comparison plots.

    Args:
        original_locs: Original localization data
        corrected_locs: Corrected localization data
        drift_result: Drift correction results
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Original data scatter plot
    ax1.scatter(original_locs.xc, original_locs.yc, s=1, alpha=0.1, c="red")
    ax1.set_title("Before Correction")
    ax1.set_xlabel("X (pixels)")
    ax1.set_ylabel("Y (pixels)")
    ax1.set_aspect("equal")

    # Corrected data scatter plot
    ax2.scatter(corrected_locs.xc, corrected_locs.yc, s=1, alpha=0.1, c="blue")
    ax2.set_title("After AIM Correction")
    ax2.set_xlabel("X (pixels)")
    ax2.set_ylabel("Y (pixels)")
    ax2.set_aspect("equal")

    # Drift trace X
    frames = np.arange(len(drift_result.drift_x))
    ax3.plot(frames, drift_result.drift_x, "b-", linewidth=1)
    ax3.set_title("X Drift Trace")
    ax3.set_xlabel("Frame")
    ax3.set_ylabel("Drift X (pixels)")
    ax3.grid(True, alpha=0.3)

    # Drift trace Y
    ax4.plot(frames, drift_result.drift_y, "r-", linewidth=1)
    ax4.set_title("Y Drift Trace")
    ax4.set_xlabel("Frame")
    ax4.set_ylabel("Drift Y (pixels)")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "/tmp/aim_drift_correction_comparison.png", dpi=150, bbox_inches="tight"
    )
    print(f"✅ Comparison plot saved to: /tmp/aim_drift_correction_comparison.png")
    plt.show()


def main():
    """
    Main function demonstrating complete AIM drift correction workflow.
    """
    print("AIM Drift Correction Example")
    print("=" * 60)

    # 1. Load example data
    locs, info = load_example_data()

    # 2. Perform AIM drift correction
    corrected_locs, drift_result = perform_aim_drift_correction(
        locs,
        info,
        segmentation=100,  # Process 100 frames at a time
        intersect_d=20 / 69,  # Default intersection distance
        roi_r=60 / 69,  # Default search radius
        display=True,  # Show drift plots during processing
    )

    # 3. Analyze correction quality
    metrics = analyze_correction_quality(locs, corrected_locs, drift_result)

    # 4. Save results
    csv_path, drift_path = save_results(corrected_locs, drift_result)

    # 5. Create comparison plots
    plot_before_after_comparison(locs, corrected_locs, drift_result)

    # 6. Summary
    print(f"\n" + "=" * 60)
    print("WORKFLOW COMPLETE")
    print("=" * 60)
    print(f"✅ Original data: {len(locs)} localizations")
    print(f"✅ Corrected data: {len(corrected_locs)} localizations")
    print(f"✅ Method: {drift_result.method}")
    print(f"✅ X improvement: {metrics['improvement_x']:.2f}x")
    print(f"✅ Y improvement: {metrics['improvement_y']:.2f}x")
    print(f"✅ Files saved: {csv_path}, {drift_path}")


if __name__ == "__main__":
    # Usage examples:

    # Basic usage with default parameters:
    # python aim_drift_correction_example.py

    # For your own data, replace the load_example_data() function with:
    #
    # def load_your_data():
    #     io = IOFunctions.IO_Functions()
    #     locs = io.read_localisations("your_localizations.csv")
    #     info = [{"Width": 256, "Height": 256, "Frames": 10000, "Pixelsize": 0.1}]
    #     return locs, info

    main()
