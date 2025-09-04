#!/usr/bin/env python3
"""
Separated Fiducial Drift Correction Example

This script demonstrates the new separated workflow for fiducial drift correction:
1. First detect fiducials with visualization
2. Then use detected fiducials for drift correction

This separation allows for better inspection and validation of fiducial detection
before applying drift correction.

Features:
- Separated fiducial detection and drift correction steps
- Professional plotting using PlottingFunctions
- Interactive parameter adjustment workflow
- Comprehensive result analysis and export

Author: Claude Code Assistant
Created: September 3, 2025
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, List, Optional

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    import DriftCorrectionFunctions as DCF
    import IOFunctions
    import PlottingFunctions
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the pyBayerSMLM directory with src/ available")
    sys.exit(1)


def create_example_data() -> Tuple[np.recarray, List[dict]]:
    """
    Create example localization data with simulated fiducials.

    Returns:
        Tuple of (localizations, metadata_info)
    """
    print("Creating example data with simulated fiducials...")

    np.random.seed(42)

    # Simulation parameters
    n_frames = 1000
    image_size = 128
    pixel_size = 0.1  # μm per pixel

    # Create 4 fiducial markers at corner positions
    fiducial_positions = [
        (25, 25),  # Bottom-left
        (103, 25),  # Bottom-right
        (25, 103),  # Top-left
        (103, 103),  # Top-right
    ]

    # Simulate realistic drift pattern
    frames = np.arange(n_frames)
    true_drift_x = 1.2 * np.sin(frames / 100) + 0.004 * frames  # Sinusoidal + linear
    true_drift_y = 0.8 * np.cos(frames / 80) + 0.002 * frames  # Different frequency

    all_locs = []

    # Generate high-intensity fiducial localizations
    for fid_id, (base_x, base_y) in enumerate(fiducial_positions):
        n_locs_per_frame = 30  # Dense, bright fiducials

        for frame in range(n_frames):
            if np.random.random() > 0.02:  # 98% probability of detection per frame
                n_this_frame = np.random.poisson(n_locs_per_frame)

                # Tight localization precision for fiducials
                x_coords = base_x + np.random.normal(0, 0.2, n_this_frame)
                y_coords = base_y + np.random.normal(0, 0.2, n_this_frame)

                # Add drift
                x_coords += true_drift_x[frame]
                y_coords += true_drift_y[frame]

                # Add localization precision
                x_coords += np.random.normal(0, 0.015, n_this_frame)
                y_coords += np.random.normal(0, 0.015, n_this_frame)

                # High photon count for fiducials
                photons = np.random.exponential(3000, n_this_frame) + 1000

                # Store localizations
                for x, y, ph in zip(x_coords, y_coords, photons):
                    all_locs.append(
                        {
                            "xc": x * pixel_size,  # Convert to μm
                            "yc": y * pixel_size,
                            "frame": frame,
                            "photons": ph,
                            "is_fiducial": True,
                            "fiducial_id": fid_id,
                        }
                    )

    # Generate background cellular structures
    n_bg_structures = 12

    for struct_id in range(n_bg_structures):
        # Random positions avoiding fiducials
        while True:
            base_x = np.random.uniform(15, 113)
            base_y = np.random.uniform(15, 113)

            # Check minimum distance from fiducials
            min_dist = min(
                [
                    np.sqrt((base_x - fx) ** 2 + (base_y - fy) ** 2)
                    for fx, fy in fiducial_positions
                ]
            )
            if min_dist > 12:
                break

        for frame in range(n_frames):
            if np.random.random() > 0.5:  # 50% probability per frame
                n_this_frame = np.random.poisson(8)

                # Larger scatter for biological structures
                x_coords = base_x + np.random.normal(0, 5, n_this_frame)
                y_coords = base_y + np.random.normal(0, 5, n_this_frame)

                # Add drift
                x_coords += true_drift_x[frame]
                y_coords += true_drift_y[frame]

                # Larger localization uncertainty
                x_coords += np.random.normal(0, 0.08, n_this_frame)
                y_coords += np.random.normal(0, 0.08, n_this_frame)

                # Lower photon counts
                photons = np.random.exponential(1500, n_this_frame) + 200

                for x, y, ph in zip(x_coords, y_coords, photons):
                    all_locs.append(
                        {
                            "xc": x * pixel_size,
                            "yc": y * pixel_size,
                            "frame": frame,
                            "photons": ph,
                            "is_fiducial": False,
                            "fiducial_id": -1,
                        }
                    )

    # Convert to structured array
    locs_df = pd.DataFrame(all_locs)
    locs = np.rec.fromrecords(
        locs_df[["xc", "yc", "frame", "photons"]].values,
        names=["xc", "yc", "frame", "photons"],
    )

    # Create metadata
    info = [
        {
            "Width": image_size,
            "Height": image_size,
            "Frames": n_frames,
            "Pixelsize": pixel_size,
        }
    ]

    print(f"✅ Created example data:")
    print(f"   - {len(locs):,} total localizations")
    print(f"   - {len(fiducial_positions)} simulated fiducials")
    print(f"   - {n_frames} frames")
    print(
        f"   - Image size: {image_size}x{image_size} pixels ({pixel_size*1000:.0f} nm/pixel)"
    )
    print(
        f"   - True drift range: X=[{true_drift_x.min():.3f}, {true_drift_x.max():.3f}] pixels"
    )
    print(
        f"                       Y=[{true_drift_y.min():.3f}, {true_drift_y.max():.3f}] pixels"
    )

    return locs, info


def step1_detect_fiducials(
    locs: np.recarray,
    info: List[dict],
    threshold_percentile: float = 88.0,
    box_size_nm: float = 1200.0,
    min_frames_fraction: float = 0.7,
) -> DCF.FiducialDetectionResult:
    """
    Step 1: Detect fiducial markers with visualization.

    Args:
        locs: Localization data
        info: Metadata list
        threshold_percentile: Detection threshold (lower = more candidates)
        box_size_nm: Detection box size in nm
        min_frames_fraction: Minimum fraction of frames for valid fiducial

    Returns:
        FiducialDetectionResult with detected fiducials
    """
    print("\n" + "=" * 60)
    print("STEP 1: FIDUCIAL DETECTION")
    print("=" * 60)

    print(f"Detection parameters:")
    print(f"  - Threshold percentile: {threshold_percentile}%")
    print(f"  - Box size: {box_size_nm} nm")
    print(f"  - Min frames fraction: {min_frames_fraction}")

    # Initialize drift corrector
    drift_corrector = DCF.Drift_Correction_Functions()

    # Detect fiducials with automatic plotting
    detection_result = drift_corrector.detect_fiducials(
        locs=locs,
        info=info,
        threshold_percentile=threshold_percentile,
        box_size_nm=box_size_nm,
        min_frames_fraction=min_frames_fraction,
        histogram_bins=256,
        plot_results=True,  # Create plot using PlottingFunctions
        save_plot="fiducial_detection_results.png",
    )

    print(f"\\n✅ Fiducial detection completed!")
    print(f"   Found {detection_result.n_fiducials} valid fiducials")
    print(
        f"   Total candidates detected: {detection_result.metadata['total_candidates']}"
    )
    print(
        f"   Localizations per fiducial: {detection_result.metadata['localizations_per_fiducial']}"
    )

    # Display fiducial positions
    print(f"\\nFiducial positions:")
    for i, (x, y) in enumerate(detection_result.picks):
        print(f"  Fiducial {i+1}: ({x:.1f}, {y:.1f}) pixels")

    return detection_result


def step2_drift_correction(
    detection_result: DCF.FiducialDetectionResult,
) -> Tuple[np.recarray, DCF.DriftResult]:
    """
    Step 2: Perform drift correction using detected fiducials.

    Args:
        detection_result: Result from step 1 fiducial detection

    Returns:
        Tuple of (corrected_localizations, drift_result)
    """
    print("\n" + "=" * 60)
    print("STEP 2: DRIFT CORRECTION")
    print("=" * 60)

    print(
        f"Using {detection_result.n_fiducials} detected fiducials for drift correction..."
    )

    # Initialize drift corrector
    drift_corrector = DCF.Drift_Correction_Functions()

    # Apply drift correction using detected fiducials
    corrected_locs, drift_result = drift_corrector.undrift_with_detected_fiducials(
        detection_result=detection_result,
        display=False,  # Can add display=True to show drift during calculation
    )

    print(f"\\n✅ Drift correction completed!")
    print(f"   Method used: {drift_result.method_used}")
    print(
        f"   X drift range: {drift_result.drift_x.min():.3f} to {drift_result.drift_x.max():.3f} pixels"
    )
    print(
        f"   Y drift range: {drift_result.drift_y.min():.3f} to {drift_result.drift_y.max():.3f} pixels"
    )
    print(
        f"   Maximum drift magnitude: {np.sqrt(drift_result.drift_x**2 + drift_result.drift_y**2).max():.3f} pixels"
    )

    return corrected_locs, drift_result


def analyze_correction_quality(
    original_locs: np.recarray,
    corrected_locs: np.recarray,
    detection_result: DCF.FiducialDetectionResult,
    drift_result: DCF.DriftResult,
) -> Dict[str, Any]:
    """
    Analyze the quality of fiducial drift correction.

    Args:
        original_locs: Original localizations
        corrected_locs: Drift-corrected localizations
        detection_result: Fiducial detection results
        drift_result: Drift correction results

    Returns:
        Dictionary with quality metrics
    """
    print("\n" + "=" * 60)
    print("QUALITY ANALYSIS")
    print("=" * 60)

    # Extract fiducial localizations for analysis
    fiducial_locs = (
        corrected_locs[corrected_locs.group >= 0]
        if hasattr(corrected_locs, "group")
        else []
    )

    if len(fiducial_locs) > 0:
        # Calculate per-fiducial precision
        unique_groups = np.unique(fiducial_locs.group)
        fiducial_precisions = []

        print(f"Per-Fiducial Analysis:")
        print(f"{'ID':<3} {'Locs':<6} {'Center (μm)':<15} {'Precision (nm)':<15}")
        print("-" * 45)

        for group_id in unique_groups:
            group_locs = fiducial_locs[fiducial_locs.group == group_id]

            center_x = np.mean(group_locs.xc)
            center_y = np.mean(group_locs.yc)
            precision_x = np.std(group_locs.xc) * 1000  # Convert to nm
            precision_y = np.std(group_locs.yc) * 1000

            fiducial_precisions.append((precision_x, precision_y))

            print(
                f"{group_id+1:<3} {len(group_locs):<6} ({center_x:.2f},{center_y:.2f})   ({precision_x:.0f},{precision_y:.0f})"
            )

    # Overall coordinate spread comparison
    orig_spread_x = np.std(original_locs.xc) * 1000  # nm
    orig_spread_y = np.std(original_locs.yc) * 1000
    corr_spread_x = np.std(corrected_locs.xc) * 1000
    corr_spread_y = np.std(corrected_locs.yc) * 1000

    # Calculate improvement
    improvement_x = orig_spread_x / corr_spread_x if corr_spread_x > 0 else 1.0
    improvement_y = orig_spread_y / corr_spread_y if corr_spread_y > 0 else 1.0

    metrics = {
        "n_fiducials": detection_result.n_fiducials,
        "original_spread_x_nm": orig_spread_x,
        "original_spread_y_nm": orig_spread_y,
        "corrected_spread_x_nm": corr_spread_x,
        "corrected_spread_y_nm": corr_spread_y,
        "improvement_factor_x": improvement_x,
        "improvement_factor_y": improvement_y,
        "max_drift_magnitude": np.sqrt(
            drift_result.drift_x**2 + drift_result.drift_y**2
        ).max(),
        "fiducial_precisions": (
            fiducial_precisions if "fiducial_precisions" in locals() else []
        ),
        "detection_method": "separated_workflow",
    }

    print(f"\\nOverall Quality Summary:")
    print(f"  - Fiducials used: {metrics['n_fiducials']}")
    print(
        f"  - Coordinate spread before: {orig_spread_x:.0f} nm (X), {orig_spread_y:.0f} nm (Y)"
    )
    print(
        f"  - Coordinate spread after: {corr_spread_x:.0f} nm (X), {corr_spread_y:.0f} nm (Y)"
    )
    print(
        f"  - Improvement factors: {improvement_x:.2f}x (X), {improvement_y:.2f}x (Y)"
    )
    print(f"  - Maximum drift corrected: {metrics['max_drift_magnitude']:.3f} pixels")

    return metrics


def create_comprehensive_plots(
    original_locs: np.recarray,
    corrected_locs: np.recarray,
    detection_result: DCF.FiducialDetectionResult,
    drift_result: DCF.DriftResult,
    save_path: str = "separated_fiducial_drift_results.png",
):
    """
    Create comprehensive result plots using PlottingFunctions where possible.

    Args:
        original_locs: Original localizations
        corrected_locs: Corrected localizations
        detection_result: Fiducial detection results
        drift_result: Drift correction results
        save_path: Path to save the plot
    """
    print(f"\\nCreating comprehensive result plots...")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # Try to use PlottingFunctions where appropriate
    try:
        plotter = PlottingFunctions.Plotter(poster=False, dark_background=False)
        use_plotting_functions = True
    except:
        use_plotting_functions = False
        print("⚠️ PlottingFunctions not available, using matplotlib directly")

    # 1. Original data scatter
    ax = axes[0]
    ax.scatter(
        original_locs.xc * 1000, original_locs.yc * 1000, s=1, alpha=0.3, c="red"
    )

    # Add fiducial markers
    for i, (x, y) in enumerate(detection_result.picks):
        ax.plot(
            x * detection_result.detection_params["pixelsize"],
            y * detection_result.detection_params["pixelsize"],
            "co",
            markersize=8,
            markerfacecolor="none",
            markeredgewidth=2,
        )
        ax.text(
            x * detection_result.detection_params["pixelsize"],
            y * detection_result.detection_params["pixelsize"] + 500,
            f"F{i+1}",
            color="cyan",
            ha="center",
            fontweight="bold",
        )

    ax.set_title("Original Data + Detected Fiducials")
    ax.set_xlabel("X (nm)")
    ax.set_ylabel("Y (nm)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # 2. Corrected data scatter
    ax = axes[1]
    ax.scatter(
        corrected_locs.xc * 1000, corrected_locs.yc * 1000, s=1, alpha=0.3, c="blue"
    )
    ax.set_title("After Drift Correction")
    ax.set_xlabel("X (nm)")
    ax.set_ylabel("Y (nm)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # 3. Fiducial localizations by group (before correction)
    ax = axes[2]
    if hasattr(detection_result.locs_with_groups, "group"):
        fiducial_locs = detection_result.locs_with_groups[
            detection_result.locs_with_groups.group >= 0
        ]
        if len(fiducial_locs) > 0:
            unique_groups = np.unique(fiducial_locs.group)
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique_groups)))

            for i, group_id in enumerate(unique_groups):
                group_locs = fiducial_locs[fiducial_locs.group == group_id]
                ax.scatter(
                    group_locs.xc * 1000,
                    group_locs.yc * 1000,
                    s=3,
                    alpha=0.6,
                    c=[colors[i]],
                    label=f"F{group_id+1}",
                )

    ax.set_title("Fiducials Before Correction")
    ax.set_xlabel("X (nm)")
    ax.set_ylabel("Y (nm)")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # 4. Fiducial localizations after correction
    ax = axes[3]
    if hasattr(corrected_locs, "group"):
        corrected_fiducial_locs = corrected_locs[corrected_locs.group >= 0]
        if len(corrected_fiducial_locs) > 0:
            unique_groups = np.unique(corrected_fiducial_locs.group)
            for i, group_id in enumerate(unique_groups):
                group_locs = corrected_fiducial_locs[
                    corrected_fiducial_locs.group == group_id
                ]
                ax.scatter(
                    group_locs.xc * 1000,
                    group_locs.yc * 1000,
                    s=3,
                    alpha=0.6,
                    c=[colors[i]],
                    label=f"F{group_id+1}",
                )

    ax.set_title("Fiducials After Correction")
    ax.set_xlabel("X (nm)")
    ax.set_ylabel("Y (nm)")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # 5. Drift traces
    ax = axes[4]
    frames = np.arange(len(drift_result.drift_x))
    ax.plot(frames, drift_result.drift_x, "b-", linewidth=1, label="X drift", alpha=0.8)
    ax.plot(frames, drift_result.drift_y, "r-", linewidth=1, label="Y drift", alpha=0.8)
    ax.set_title("Measured Drift Traces")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Drift (pixels)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Drift magnitude and summary
    ax = axes[5]
    drift_magnitude = np.sqrt(drift_result.drift_x**2 + drift_result.drift_y**2)
    ax.plot(frames, drift_magnitude, "g-", linewidth=1, alpha=0.8)
    ax.set_title("Drift Magnitude")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Drift Magnitude (pixels)")
    ax.grid(True, alpha=0.3)

    # Add summary text
    summary_text = (
        f"Separated Workflow Results:\\n"
        f"Method: Fiducial-based\\n"
        f"Fiducials: {detection_result.n_fiducials}\\n"
        f"Max drift: {drift_magnitude.max():.3f} px\\n"
        f"Detection threshold: {detection_result.detection_params['threshold_percentile']:.1f}%\\n"
        f"Box size: {detection_result.detection_params['box_size_nm']:.0f} nm"
    )

    ax.text(
        0.02,
        0.98,
        summary_text,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8),
    )

    # Overall title
    fig.suptitle(
        "Separated Fiducial Drift Correction Workflow Results",
        fontsize=16,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"✅ Comprehensive plot saved to: {save_path}")
    plt.show()


def save_results(
    detection_result: DCF.FiducialDetectionResult,
    corrected_locs: np.recarray,
    drift_result: DCF.DriftResult,
    quality_metrics: Dict[str, Any],
    output_base: str = "separated_fiducial_workflow_results",
):
    """
    Save all results from the separated workflow.

    Args:
        detection_result: Fiducial detection results
        corrected_locs: Corrected localizations
        drift_result: Drift correction results
        quality_metrics: Quality metrics
        output_base: Base name for output files
    """
    print(f"\\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    file_paths = []

    # 1. Save corrected localizations
    corrected_df = pd.DataFrame(corrected_locs)
    corrected_path = f"{output_base}_corrected_localizations.csv"
    corrected_df.to_csv(corrected_path, index=False)
    file_paths.append(corrected_path)
    print(f"✅ Corrected localizations: {corrected_path}")

    # 2. Save drift trace
    drift_df = pd.DataFrame(
        {
            "frame": np.arange(len(drift_result.drift_x)),
            "drift_x_pixels": drift_result.drift_x,
            "drift_y_pixels": drift_result.drift_y,
            "drift_magnitude": np.sqrt(
                drift_result.drift_x**2 + drift_result.drift_y**2
            ),
        }
    )
    drift_path = f"{output_base}_drift_trace.csv"
    drift_df.to_csv(drift_path, index=False)
    file_paths.append(drift_path)
    print(f"✅ Drift trace: {drift_path}")

    # 3. Save fiducial detection details
    detection_df = pd.DataFrame(
        {
            "fiducial_id": range(len(detection_result.picks)),
            "pick_x": [p[0] for p in detection_result.picks],
            "pick_y": [p[1] for p in detection_result.picks],
            "n_localizations": detection_result.metadata["localizations_per_fiducial"],
        }
    )
    detection_path = f"{output_base}_fiducial_details.csv"
    detection_df.to_csv(detection_path, index=False)
    file_paths.append(detection_path)
    print(f"✅ Fiducial details: {detection_path}")

    # 4. Save comprehensive summary
    import json

    summary_data = {
        "workflow": "separated_fiducial_drift_correction",
        "timestamp": pd.Timestamp.now().isoformat(),
        "detection_parameters": detection_result.detection_params,
        "detection_metadata": detection_result.metadata,
        "quality_metrics": quality_metrics,
        "drift_summary": {
            "method": str(drift_result.method_used),
            "max_drift_x": float(drift_result.drift_x.max()),
            "min_drift_x": float(drift_result.drift_x.min()),
            "max_drift_y": float(drift_result.drift_y.max()),
            "min_drift_y": float(drift_result.drift_y.min()),
            "max_magnitude": float(
                np.sqrt(drift_result.drift_x**2 + drift_result.drift_y**2).max()
            ),
        },
    }

    summary_path = f"{output_base}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2, default=str)
    file_paths.append(summary_path)
    print(f"✅ Summary report: {summary_path}")

    print(f"\\n✅ All results saved! Generated {len(file_paths)} files.")
    return file_paths


def main():
    """
    Main function demonstrating the separated fiducial drift correction workflow.
    """
    print("Separated Fiducial Drift Correction Workflow")
    print("=" * 60)
    print("This example demonstrates the new separated approach:")
    print("1. Detect fiducials with visualization")
    print("2. Apply drift correction using detected fiducials")
    print("3. Analyze and save results")

    # Load or create data
    locs, info = create_example_data()

    # Step 1: Detect fiducials (with automatic plotting)
    detection_result = step1_detect_fiducials(
        locs,
        info,
        threshold_percentile=88.0,  # Lower threshold for more sensitivity
        box_size_nm=1200.0,  # Larger detection box
        min_frames_fraction=0.7,  # Require 70% of frames
    )

    # Optional: Allow user to inspect detection results before proceeding
    input(
        "\\n⏸️  Press Enter to proceed with drift correction using detected fiducials..."
    )

    # Step 2: Apply drift correction
    corrected_locs, drift_result = step2_drift_correction(detection_result)

    # Step 3: Analyze results
    quality_metrics = analyze_correction_quality(
        locs, corrected_locs, detection_result, drift_result
    )

    # Step 4: Create comprehensive plots
    create_comprehensive_plots(locs, corrected_locs, detection_result, drift_result)

    # Step 5: Save all results
    file_paths = save_results(
        detection_result, corrected_locs, drift_result, quality_metrics
    )

    # Final summary
    print(f"\\n" + "=" * 60)
    print("SEPARATED WORKFLOW COMPLETE")
    print("=" * 60)
    print(f"✅ Workflow: Separated fiducial detection → drift correction")
    print(f"✅ Fiducials detected: {detection_result.n_fiducials}")
    print(f"✅ Original data: {len(locs):,} localizations")
    print(f"✅ Corrected data: {len(corrected_locs):,} localizations")
    print(
        f"✅ Max drift corrected: {quality_metrics['max_drift_magnitude']:.3f} pixels"
    )
    print(
        f"✅ Improvement factors: {quality_metrics['improvement_factor_x']:.2f}x (X), {quality_metrics['improvement_factor_y']:.2f}x (Y)"
    )
    print(f"✅ Files generated: {len(file_paths)}")

    print(f"\\n🎯 Key advantages of separated workflow:")
    print(f"   • Inspect fiducial detection before drift correction")
    print(f"   • Adjust detection parameters iteratively")
    print(f"   • Professional plots using PlottingFunctions")
    print(f"   • Reuse detected fiducials for multiple corrections")


if __name__ == "__main__":
    # Usage examples:
    #
    # Basic usage:
    # python separated_fiducial_drift_correction_example.py
    #
    # For your own data, modify the create_example_data() function or use:
    # def load_your_data():
    #     io = IOFunctions.IO_Functions()
    #     locs = io.read_localisations("your_data.csv")
    #     info = [{"Width": 256, "Height": 256, "Frames": 10000, "Pixelsize": 0.1}]
    #     return locs, info

    main()
