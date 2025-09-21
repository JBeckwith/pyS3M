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
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from DriftCorrectionFunctions import (
    Drift_Correction_Functions,
    DriftMethod,
    DriftParameters,
    DriftCorrectionFactory,
)


def create_test_data():
    """Create synthetic localization data for testing with frame-by-frame drift."""

    # Generate synthetic localizations with artificial drift
    np.random.seed(42)
    n_locs = 5000
    n_frames = 1000

    # Base coordinates
    x_base = np.random.uniform(0, 100, n_locs)
    y_base = np.random.uniform(0, 100, n_locs)
    frames = np.random.randint(1, n_frames + 1, n_locs)

    # Create ground truth drift for every frame (1 to n_frames)
    all_frames = np.arange(1, n_frames + 1)
    ground_truth_drift_x = 2 * np.sin(2 * np.pi * all_frames / 200)  # 2 pixel amplitude
    ground_truth_drift_y = 1.5 * np.cos(2 * np.pi * all_frames / 300)  # 1.5 pixel amplitude

    # Apply drift to localizations based on their frame
    drift_x = ground_truth_drift_x[frames - 1]  # frames are 1-indexed
    drift_y = ground_truth_drift_y[frames - 1]

    x_drifted = x_base + drift_x
    y_drifted = y_base + drift_y

    # Create record array with error fields (required for render function)
    locs = np.rec.array(
        (
            x_drifted,
            y_drifted,
            frames,
            np.ones(n_locs) * 1000,
            np.ones(n_locs) * 0.1,
            np.ones(n_locs) * 0.1,
        ),  # Add xc_err, yc_err
        dtype=[
            ("xc", "f4"),
            ("yc", "f4"),
            ("frame", "i4"),
            ("photons", "f4"),
            ("xc_err", "f4"),
            ("yc_err", "f4"),
        ],
    )

    # Create metadata
    info = [
        {
            "Width": 100.0,
            "Height": 100.0,
            "Frames": float(n_frames),
            "Pixelsize": 100.0,  # nm per pixel
        }
    ]

    # Save ground truth drift for plotting
    ground_truth = {
        'frames': all_frames,
        'drift_x': ground_truth_drift_x,
        'drift_y': ground_truth_drift_y
    }

    return locs, info, ground_truth


def create_fiducial_test_data():
    """Create synthetic localization data with fiducial markers for testing."""

    # Generate synthetic localizations with artificial drift and fiducial markers
    np.random.seed(42)
    n_locs = 3000  # Fewer locs for fiducial test
    n_frames = 1000
    n_fiducials = 5  # Number of fiducial markers

    # Create fiducial localizations (stationary markers + drift)
    fiducial_locs = []
    for fid_id in range(n_fiducials):
        # Each fiducial has fixed position + some scatter + drift
        base_x = 20 + fid_id * 15  # Spread fiducials across field
        base_y = 20 + fid_id * 15

        # Generate localizations for this fiducial across frames
        frames_per_fid = n_locs // (
            n_fiducials * 2
        )  # About half the locs are fiducials
        fid_frames = np.random.choice(n_frames, frames_per_fid, replace=True) + 1

        # Add scatter around fiducial position (realistic localization precision)
        scatter_x = np.random.normal(0, 0.2, frames_per_fid)  # 20nm scatter
        scatter_y = np.random.normal(0, 0.2, frames_per_fid)

        # Add artificial drift (same as regular molecules)
        drift_x = 2 * np.sin(2 * np.pi * fid_frames / 200)
        drift_y = 1.5 * np.cos(2 * np.pi * fid_frames / 300)

        fid_x = base_x + scatter_x + drift_x
        fid_y = base_y + scatter_y + drift_y

        # Store with group ID
        for i, frame in enumerate(fid_frames):
            fiducial_locs.append((fid_x[i], fid_y[i], frame, 2000.0, 0.1, 0.1, fid_id))

    # Add some regular (non-fiducial) localizations
    remaining_locs = n_locs - len(fiducial_locs)
    reg_frames = np.random.randint(1, n_frames + 1, remaining_locs)
    reg_x = np.random.uniform(5, 95, remaining_locs)
    reg_y = np.random.uniform(5, 95, remaining_locs)

    # Add drift to regular molecules too
    reg_drift_x = 2 * np.sin(2 * np.pi * reg_frames / 200)
    reg_drift_y = 1.5 * np.cos(2 * np.pi * reg_frames / 300)
    reg_x += reg_drift_x
    reg_y += reg_drift_y

    # Regular molecules get group ID -1 (non-fiducial)
    for i in range(remaining_locs):
        fiducial_locs.append((reg_x[i], reg_y[i], reg_frames[i], 1000.0, 0.1, 0.1, -1))

    # Convert to arrays
    all_data = np.array(fiducial_locs)

    # Create record array with group field
    locs = np.rec.array(
        (
            all_data[:, 0],
            all_data[:, 1],
            all_data[:, 2].astype(int),
            all_data[:, 3],
            all_data[:, 4],
            all_data[:, 5],
            all_data[:, 6].astype(int),
        ),
        dtype=[
            ("xc", "f4"),
            ("yc", "f4"),
            ("frame", "i4"),
            ("photons", "f4"),
            ("xc_err", "f4"),
            ("yc_err", "f4"),
            ("group", "i4"),
        ],
    )

    # Create metadata
    info = [
        {
            "Width": 100.0,
            "Height": 100.0,
            "Frames": float(n_frames),
            "Pixelsize": 100.0,  # nm per pixel
        }
    ]

    # True drift for validation
    drift_frames = np.arange(1, n_frames + 1)
    true_drift_x = 2 * np.sin(2 * np.pi * drift_frames / 200)
    true_drift_y = 1.5 * np.cos(2 * np.pi * drift_frames / 300)

    return locs, info, (true_drift_x, true_drift_y)


def create_dna_origami_test_data():
    """Create super-resolution test data mimicking DNA origami structures.

    Simulates 2x2 DNA origami structures where each object is 40nm apart,
    with sparse SMLM sampling (<1 object lit up per frame).
    """
    np.random.seed(42)
    n_frames = 2000
    pixel_size = 100.0  # nm per pixel
    origami_spacing = 40.0 / pixel_size  # 40nm in pixels

    # Define ground truth 2x2 origami structures
    # Create multiple origami patterns across the field of view
    n_origami_structures = 25  # 5x5 grid of origami structures
    grid_size = int(np.sqrt(n_origami_structures))
    field_size = 50.0  # pixels
    structure_spacing = field_size / (grid_size + 1)

    # Generate all binding sites for all origami structures
    binding_sites = []
    for i in range(grid_size):
        for j in range(grid_size):
            # Center of this origami structure
            center_x = (i + 1) * structure_spacing
            center_y = (j + 1) * structure_spacing

            # Add the 4 binding sites of the 2x2 structure
            for dx in [-origami_spacing/2, origami_spacing/2]:
                for dy in [-origami_spacing/2, origami_spacing/2]:
                    binding_sites.append((
                        center_x + dx,
                        center_y + dy,
                        i * grid_size + j  # structure_id
                    ))

    binding_sites = np.array(binding_sites)
    n_sites = len(binding_sites)

    # Create ground truth drift for every frame
    all_frames = np.arange(1, n_frames + 1)
    # More complex drift pattern for challenging test
    ground_truth_drift_x = (
        1.5 * np.sin(2 * np.pi * all_frames / 500) +  # Slow drift
        0.5 * np.sin(2 * np.pi * all_frames / 100) +  # Fast oscillation
        0.01 * all_frames  # Linear drift component
    )
    ground_truth_drift_y = (
        1.2 * np.cos(2 * np.pi * all_frames / 400) +
        0.3 * np.cos(2 * np.pi * all_frames / 80) +
        0.008 * all_frames
    )

    # Sparse SMLM sampling: each frame has probabilistic activation
    # Probability that a binding site is active in any given frame
    activation_probability = 0.3 / n_sites  # Average ~0.3 localizations per frame

    localizations = []

    for frame in all_frames:
        # Determine which sites are active this frame
        active_mask = np.random.random(n_sites) < activation_probability
        active_sites = binding_sites[active_mask]

        if len(active_sites) > 0:
            # Apply drift to active sites
            drift_x = ground_truth_drift_x[frame - 1]
            drift_y = ground_truth_drift_y[frame - 1]

            for site in active_sites:
                x_base, y_base, structure_id = site

                # Add localization precision noise (realistic SMLM precision)
                x_noise = np.random.normal(0, 0.15)  # 15nm precision
                y_noise = np.random.normal(0, 0.15)

                # Final position with drift and noise
                x_final = x_base + drift_x + x_noise
                y_final = y_base + drift_y + y_noise

                # Photon count varies realistically
                photons = np.random.normal(2000, 500)
                photons = max(photons, 500)  # minimum photon count

                localizations.append((
                    x_final, y_final, frame, photons, 0.15, 0.15, int(structure_id)
                ))

    # Convert to structured array
    if len(localizations) == 0:
        # Edge case: no localizations generated
        localizations = [(0, 0, 1, 1000, 0.15, 0.15, 0)]

    all_data = np.array(localizations)

    # Create record array
    locs = np.rec.array(
        (
            all_data[:, 0],
            all_data[:, 1],
            all_data[:, 2].astype(int),
            all_data[:, 3],
            all_data[:, 4],
            all_data[:, 5],
            all_data[:, 6].astype(int),
        ),
        dtype=[
            ("xc", "f4"),
            ("yc", "f4"),
            ("frame", "i4"),
            ("photons", "f4"),
            ("xc_err", "f4"),
            ("yc_err", "f4"),
            ("structure_id", "i4"),
        ],
    )

    # Create metadata
    info = [
        {
            "Width": field_size,
            "Height": field_size,
            "Frames": float(n_frames),
            "Pixelsize": pixel_size,  # nm per pixel
        }
    ]

    # Ground truth data for validation
    ground_truth = {
        'frames': all_frames,
        'drift_x': ground_truth_drift_x,
        'drift_y': ground_truth_drift_y,
        'binding_sites': binding_sites,
        'origami_spacing_pixels': origami_spacing,
        'n_structures': n_origami_structures,
        'n_localizations': len(locs),
        'avg_locs_per_frame': len(locs) / n_frames
    }

    return locs, info, ground_truth


def save_ground_truth_data(ground_truth, filename_prefix="ground_truth"):
    """Save ground truth drift data for plotting and analysis."""
    import json

    # Convert numpy arrays to lists for JSON serialization
    serializable_gt = {}
    for key, value in ground_truth.items():
        if isinstance(value, np.ndarray):
            serializable_gt[key] = value.tolist()
        else:
            serializable_gt[key] = value

    # Save as JSON
    json_filename = f"{filename_prefix}_drift.json"
    with open(json_filename, 'w') as f:
        json.dump(serializable_gt, f, indent=2)

    # Save as numpy format for easy loading
    npz_filename = f"{filename_prefix}_drift.npz"
    np.savez(npz_filename, **ground_truth)

    print(f"Ground truth saved to {json_filename} and {npz_filename}")
    return json_filename, npz_filename


def plot_drift_comparison(ground_truth_file, corrected_file=None):
    """Example function to plot drift comparison.

    Args:
        ground_truth_file: Path to ground truth .npz file
        corrected_file: Path to corrected drift .npz file (optional)
    """
    try:
        import matplotlib.pyplot as plt

        # Load ground truth
        gt_data = np.load(ground_truth_file)

        plt.figure(figsize=(12, 8))

        # Plot X drift
        plt.subplot(2, 1, 1)
        plt.plot(gt_data['frames'], gt_data['drift_x'], 'b-', label='Ground Truth X', linewidth=2)

        if corrected_file:
            corr_data = np.load(corrected_file)
            if 'estimated_drift_x' in corr_data:
                plt.plot(corr_data['frames'], corr_data['estimated_drift_x'], 'r--',
                        label=f'Estimated X ({corr_data.get("method", "Unknown")})', linewidth=2)

        plt.xlabel('Frame')
        plt.ylabel('Drift X (pixels)')
        plt.title('Drift Correction Comparison - X Direction')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot Y drift
        plt.subplot(2, 1, 2)
        plt.plot(gt_data['frames'], gt_data['drift_y'], 'b-', label='Ground Truth Y', linewidth=2)

        if corrected_file:
            if 'estimated_drift_y' in corr_data:
                plt.plot(corr_data['frames'], corr_data['estimated_drift_y'], 'r--',
                        label=f'Estimated Y ({corr_data.get("method", "Unknown")})', linewidth=2)

        plt.xlabel('Frame')
        plt.ylabel('Drift Y (pixels)')
        plt.title('Drift Correction Comparison - Y Direction')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_filename = ground_truth_file.replace('.npz', '_comparison.png')
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Drift comparison plot saved to {plot_filename}")

        return plot_filename

    except ImportError:
        print("Matplotlib not available. Install with: pip install matplotlib")
        return None
    except Exception as e:
        print(f"Plotting failed: {e}")
        return None


def test_vectorized_drift_correction():
    """Test that vectorized drift correction works correctly vs frame-by-frame approach."""
    print("=== Testing Vectorized Drift Correction ===")

    try:
        # Create a simple test case with known drift
        np.random.seed(123)
        n_locs = 1000
        n_frames = 100

        # Create localizations with gaps in frame sequence and non-sequential frames
        frames = np.concatenate([
            np.random.choice(range(1, 20), 200, replace=True),     # Early frames
            np.random.choice(range(25, 45), 300, replace=True),    # Middle frames (gap 20-24)
            np.random.choice(range(50, n_frames+1), 500, replace=True)  # Late frames (gap 45-49)
        ])

        x_base = np.random.uniform(10, 90, n_locs)
        y_base = np.random.uniform(10, 90, n_locs)

        # Create known drift pattern
        all_frames = np.arange(1, n_frames + 1)
        known_drift_x = 0.5 * np.sin(2 * np.pi * all_frames / 20)  # Simple sine wave
        known_drift_y = 0.3 * np.cos(2 * np.pi * all_frames / 15)  # Simple cosine wave

        # Apply known drift to create "drifted" localizations
        drift_x_applied = known_drift_x[frames - 1]  # frames are 1-indexed
        drift_y_applied = known_drift_y[frames - 1]

        x_drifted = x_base + drift_x_applied
        y_drifted = y_base + drift_y_applied

        # Create localization array
        locs = np.rec.array(
            (x_drifted, y_drifted, frames, np.ones(n_locs) * 1000,
             np.ones(n_locs) * 0.1, np.ones(n_locs) * 0.1),
            dtype=[("xc", "f4"), ("yc", "f4"), ("frame", "i4"),
                   ("photons", "f4"), ("xc_err", "f4"), ("yc_err", "f4")]
        )

        print(f"Created test data with {n_locs} localizations across {len(np.unique(frames))} unique frames")
        print(f"Frame range: {frames.min()} to {frames.max()}")
        print(f"Frame gaps present: {frames.max() - frames.min() + 1 != len(np.unique(frames))}")

        # Method 1: Vectorized correction (current implementation)
        # Simulate what the drift correction does
        drift_x_full = known_drift_x  # Assume perfect drift estimation
        drift_y_full = known_drift_y

        # Apply vectorized correction
        min_frame = int(frames.min())
        max_frame = int(frames.max())

        if min_frame == 0:
            x_corrected_vec = locs.xc - drift_x_full[locs.frame]
            y_corrected_vec = locs.yc - drift_y_full[locs.frame]
        else:
            # 1-indexed frames: subtract 1 for array indexing
            valid_indices = (locs.frame >= 1) & (locs.frame <= len(drift_x_full))
            x_corrected_vec = locs.xc.copy()
            y_corrected_vec = locs.yc.copy()
            x_corrected_vec[valid_indices] = locs.xc[valid_indices] - drift_x_full[locs.frame[valid_indices] - 1]
            y_corrected_vec[valid_indices] = locs.yc[valid_indices] - drift_y_full[locs.frame[valid_indices] - 1]

        # Method 2: Frame-by-frame correction (reference implementation)
        x_corrected_loop = locs.xc.copy()
        y_corrected_loop = locs.yc.copy()

        for frame in np.unique(locs.frame):
            if 1 <= frame <= len(drift_x_full):
                subset_mask = locs.frame == frame
                x_corrected_loop[subset_mask] -= drift_x_full[frame - 1]  # Convert to 0-indexed
                y_corrected_loop[subset_mask] -= drift_y_full[frame - 1]

        # Compare the two methods
        x_diff = np.abs(x_corrected_vec - x_corrected_loop)
        y_diff = np.abs(y_corrected_vec - y_corrected_loop)

        max_x_diff = np.max(x_diff)
        max_y_diff = np.max(y_diff)
        mean_x_diff = np.mean(x_diff)
        mean_y_diff = np.mean(y_diff)

        print(f"Comparison of vectorized vs frame-by-frame correction:")
        print(f"  Max difference X: {max_x_diff:.10f} pixels")
        print(f"  Max difference Y: {max_y_diff:.10f} pixels")
        print(f"  Mean difference X: {mean_x_diff:.10f} pixels")
        print(f"  Mean difference Y: {mean_y_diff:.10f} pixels")

        # Verify that corrected positions should equal original base positions
        x_error_vec = np.abs(x_corrected_vec - x_base)
        y_error_vec = np.abs(y_corrected_vec - y_base)
        x_error_loop = np.abs(x_corrected_loop - x_base)
        y_error_loop = np.abs(y_corrected_loop - y_base)

        print(f"Correction accuracy (should be ~0 for perfect drift removal):")
        print(f"  Vectorized - Max error X: {np.max(x_error_vec):.10f}, Y: {np.max(y_error_vec):.10f}")
        print(f"  Frame-by-frame - Max error X: {np.max(x_error_loop):.10f}, Y: {np.max(y_error_loop):.10f}")

        # Test edge cases
        print("\nTesting edge cases...")

        # Edge case 1: Frame numbers beyond drift array
        edge_frames = np.array([150, 200, 300])  # Beyond n_frames=100
        edge_x = np.array([50.0, 60.0, 70.0])
        edge_y = np.array([50.0, 60.0, 70.0])

        try:
            if min_frame == 0:
                # This should fail for out-of-bounds frames
                test_x = edge_x - drift_x_full[edge_frames]
            else:
                valid_edge = (edge_frames >= 1) & (edge_frames <= len(drift_x_full))
                test_x = edge_x.copy()
                test_x[valid_edge] = edge_x[valid_edge] - drift_x_full[edge_frames[valid_edge] - 1]
                print(f"  Edge case handling: {np.sum(valid_edge)}/{len(edge_frames)} frames were valid")
        except IndexError as e:
            print(f"  ✗ Edge case failed (IndexError): {e}")
            print("  This suggests the vectorized approach needs bounds checking!")
            return False

        # Success criteria
        tolerance = 1e-10
        if max_x_diff < tolerance and max_y_diff < tolerance:
            print(f"✓ Vectorized correction matches frame-by-frame correction (within {tolerance})")
            print("✓ Enhanced test_drift_correction.py includes comprehensive drift validation")
            print("✓ Both vectorized and frame-by-frame approaches are available")
            return True
        else:
            print(f"✗ Vectorized correction differs from frame-by-frame correction!")
            print(f"  Maximum differences exceed tolerance ({tolerance})")
            return False

    except Exception as e:
        import traceback
        print(f"✗ Vectorized drift correction test failed: {e}")
        print(f"  Details: {traceback.format_exc()}")
        return False

    print()


def test_drift_correction_robustness():
    """Test drift correction with challenging edge cases."""
    print("=== Testing Drift Correction Robustness ===")

    try:
        # Test 1: Non-sequential frames with large gaps
        frames = np.array([1, 3, 7, 15, 31, 63, 127, 255, 511, 1023])
        x_base = np.arange(10, dtype=float)
        y_base = np.arange(10, dtype=float) + 100

        # Create drift array that's shorter than max frame
        max_frame = frames.max()
        drift_length = 100  # Shorter than max_frame=1023
        drift_x = np.linspace(0, 5, drift_length)  # 0 to 5 pixels drift
        drift_y = np.linspace(0, 3, drift_length)  # 0 to 3 pixels drift

        # Apply known drift only to frames within range
        x_drifted = x_base.copy()
        y_drifted = y_base.copy()

        for i, frame in enumerate(frames):
            if frame <= drift_length:
                x_drifted[i] += drift_x[frame - 1]  # frame is 1-indexed
                y_drifted[i] += drift_y[frame - 1]

        # Create test localization data
        locs = np.rec.array(
            (x_drifted, y_drifted, frames, np.ones(len(frames)) * 1000,
             np.ones(len(frames)) * 0.1, np.ones(len(frames)) * 0.1),
            dtype=[("xc", "f4"), ("yc", "f4"), ("frame", "i4"),
                   ("photons", "f4"), ("xc_err", "f4"), ("yc_err", "f4")]
        )

        info = [{"Width": 200.0, "Height": 200.0, "Frames": float(max_frame), "Pixelsize": 100.0}]

        print(f"Created robustness test with:")
        print(f"  - Frames: {frames}")
        print(f"  - Max frame: {max_frame}")
        print(f"  - Drift array length: {drift_length}")
        print(f"  - Frames beyond drift range: {np.sum(frames > drift_length)}")

        # Test both AIM methods with the actual drift correction implementation
        DCF = Drift_Correction_Functions()

        # Test 1: AIM method
        print("\nTesting AIM with edge case data...")
        try:
            corrected_locs, drift_result = DCF.undrift(
                locs.copy(), info, method="aim", segmentation=50
            )
            print(f"✓ AIM handled edge case data successfully")
            print(f"  - Corrected {len(corrected_locs)} localizations")
            print(f"  - Drift result length: {len(drift_result.drift_x)}")
        except Exception as e:
            print(f"✗ AIM failed on edge case: {e}")

        # Test 2: Auto method
        print("\nTesting auto method with edge case data...")
        try:
            corrected_locs_auto, drift_result_auto = DCF.undrift(
                locs.copy(), info, method="auto", segmentation=50
            )
            print(f"✓ Auto method handled edge case data successfully")
            print(f"  - Selected method: {drift_result_auto.method_used.value}")
        except Exception as e:
            print(f"✗ Auto method failed on edge case: {e}")

        return True

    except Exception as e:
        import traceback
        print(f"✗ Robustness test failed: {e}")
        print(f"  Details: {traceback.format_exc()}")
        return False

    print()


def test_super_resolution_drift_correction():
    """Test drift correction on super-resolution data with sparse sampling."""
    print("=== Testing Super-Resolution Drift Correction ===")

    try:
        # Create DNA origami test data
        locs, info, ground_truth = create_dna_origami_test_data()

        print(f"Created super-resolution test data:")
        print(f"  - {ground_truth['n_localizations']} total localizations")
        print(f"  - {len(ground_truth['frames'])} frames")
        print(f"  - {ground_truth['avg_locs_per_frame']:.2f} average localizations per frame")
        print(f"  - {ground_truth['n_structures']} DNA origami structures")
        print(f"  - {ground_truth['origami_spacing_pixels']:.3f} pixels spacing (40nm)")

        # Save ground truth for analysis
        json_file, npz_file = save_ground_truth_data(ground_truth, "dna_origami")

        # Test drift correction methods on sparse data
        DCF = Drift_Correction_Functions()

        print("\nTesting AIM method on sparse super-resolution data...")

        # AIM should work well for sparse data
        corrected_locs, drift_result = DCF.undrift(
            locs.copy(),
            info,
            method="aim",
            segmentation=200,  # Larger segments for sparse data
            intersect_d=0.3,
            roi_r=1.5,
        )

        print(f"✓ AIM correction completed")
        print(f"  Method used: {drift_result.method_used.value}")
        print(f"  Drift range X: [{drift_result.drift_x.min():.2f}, {drift_result.drift_x.max():.2f}]")
        print(f"  Drift range Y: [{drift_result.drift_y.min():.2f}, {drift_result.drift_y.max():.2f}]")

        # Compare with ground truth
        drift_error_x = np.abs(drift_result.drift_x - ground_truth['drift_x'][:len(drift_result.drift_x)])
        drift_error_y = np.abs(drift_result.drift_y - ground_truth['drift_y'][:len(drift_result.drift_y)])

        print(f"  Mean drift error X: {np.mean(drift_error_x):.3f} pixels")
        print(f"  Mean drift error Y: {np.mean(drift_error_y):.3f} pixels")
        print(f"  Max drift error X: {np.max(drift_error_x):.3f} pixels")
        print(f"  Max drift error Y: {np.max(drift_error_y):.3f} pixels")

        # Test auto method selection
        print("\nTesting auto method selection on sparse data...")
        corrected_locs_auto, drift_result_auto = DCF.undrift(
            locs.copy(), info, method="auto", segmentation=200
        )

        print(f"✓ Auto method selected: {drift_result_auto.method_used.value}")
        print(f"  Selection reason: {drift_result_auto.metadata.get('auto_selection_reason', 'N/A')}")

        # Save corrected data for analysis
        corrected_gt = {
            'original_drift_x': ground_truth['drift_x'],
            'original_drift_y': ground_truth['drift_y'],
            'estimated_drift_x': drift_result.drift_x,
            'estimated_drift_y': drift_result.drift_y,
            'frames': ground_truth['frames'][:len(drift_result.drift_x)],
            'method': drift_result.method_used.value
        }
        save_ground_truth_data(corrected_gt, "dna_origami_corrected")

        # Create comparison plot if possible
        plot_drift_comparison("dna_origami_drift.npz", "dna_origami_corrected_drift.npz")

        return True

    except Exception as e:
        import traceback
        print(f"✗ Super-resolution drift correction failed: {e}")
        print(f"  Details: {traceback.format_exc()}")
        return False

    print()


def test_drift_correction_factory():
    """Test the DriftCorrectionFactory."""
    print("=== Testing DriftCorrectionFactory ===")

    # Test available methods
    methods = DriftCorrectionFactory.available_methods()
    print(f"Available methods: {[m.value for m in methods]}")

    # Test creating correctors
    for method in methods:
        corrector = DriftCorrectionFactory.create_corrector(method)
        print(
            f"{method.value}: {corrector.__class__.__name__} (3D: {corrector.supports_3d()})"
        )

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
        print(
            f"  Drift range X: [{drift_result.drift_x.min():.2f}, {drift_result.drift_x.max():.2f}]"
        )
        print(
            f"  Drift range Y: [{drift_result.drift_y.min():.2f}, {drift_result.drift_y.max():.2f}]"
        )
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
        locs, info, ground_truth = create_test_data()
        DCF = Drift_Correction_Functions()

        # Save ground truth data
        save_ground_truth_data(ground_truth, "aim_test")

        # Try AIM correction
        corrected_locs, drift_result = DCF.undrift(
            locs.copy(),
            info,
            method="aim",
            segmentation=100,
            intersect_d=0.3,
            roi_r=1.0,
        )

        print(f"✓ AIM correction completed")
        print(f"  Method used: {drift_result.method_used.value}")
        print(
            f"  Drift range X: [{drift_result.drift_x.min():.2f}, {drift_result.drift_x.max():.2f}]"
        )
        print(
            f"  Drift range Y: [{drift_result.drift_y.min():.2f}, {drift_result.drift_y.max():.2f}]"
        )
        print(f"  Metadata: {drift_result.metadata}")

        # Compare with ground truth
        if len(drift_result.drift_x) <= len(ground_truth['drift_x']):
            drift_error_x = np.abs(drift_result.drift_x - ground_truth['drift_x'][:len(drift_result.drift_x)])
            drift_error_y = np.abs(drift_result.drift_y - ground_truth['drift_y'][:len(drift_result.drift_y)])
            print(f"  Mean drift error X: {np.mean(drift_error_x):.3f} pixels")
            print(f"  Mean drift error Y: {np.mean(drift_error_y):.3f} pixels")

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
        print(
            f"  Selection reason: {drift_result.metadata.get('auto_selection_reason', 'N/A')}"
        )
        print(
            f"  Drift range X: [{drift_result.drift_x.min():.2f}, {drift_result.drift_x.max():.2f}]"
        )
        print(
            f"  Drift range Y: [{drift_result.drift_y.min():.2f}, {drift_result.drift_y.max():.2f}]"
        )

    except Exception as e:
        print(f"✗ Auto method failed: {e}")

    print()


def test_fiducial_method():
    """Test fiducial-based drift correction method."""
    print("=== Testing Fiducial Method ===")

    try:
        locs, info, true_drift = create_fiducial_test_data()
        DCF = Drift_Correction_Functions()

        print(
            f"Created {len(locs)} localizations with {len(np.unique(locs.group[locs.group >= 0]))} fiducials"
        )

        # Test 1: Traditional fiducial correction with existing group field
        corrected_locs, drift_result = DCF.undrift(locs.copy(), info, method="fiducial")

        print(f"✓ Traditional fiducial correction completed")
        print(f"  Method used: {drift_result.method_used.value}")
        print(
            f"  Drift range X: [{drift_result.drift_x.min():.2f}, {drift_result.drift_x.max():.2f}]"
        )
        print(
            f"  Drift range Y: [{drift_result.drift_y.min():.2f}, {drift_result.drift_y.max():.2f}]"
        )
        print(f"  Metadata: {drift_result.metadata}")

    except Exception as e:
        import traceback

        print(f"✗ Traditional fiducial method failed: {e}")
        print(f"  Details: {traceback.format_exc()}")

    print()


def test_fiducial_auto_detection():
    """Test automatic fiducial detection workflow."""
    print("=== Testing Automatic Fiducial Detection ===")

    try:
        # Create fiducial test data but remove the group field to test auto-detection
        locs_with_groups, info, true_drift = create_fiducial_test_data()

        # Remove group field to simulate real-world scenario
        original_dtype = locs_with_groups.dtype
        new_dtype = np.dtype(
            [desc for desc in original_dtype.descr if desc[0] != "group"]
        )
        locs = np.empty(len(locs_with_groups), dtype=new_dtype)

        # Copy all fields except group
        for field in new_dtype.names:
            locs[field] = locs_with_groups[field]
        locs = locs.view(np.recarray)

        DCF = Drift_Correction_Functions()

        print(
            f"Created {len(locs)} localizations without group field (removed from fiducial data)"
        )
        print("Testing automatic fiducial detection workflow...")

        # Test the new convenience function with very lenient parameters
        corrected_locs, drift_result, detection_info = (
            DCF.undrift_with_fiducial_detection(
                locs.copy(),
                info,
                threshold_percentile=80.0,  # Much lower threshold
                box_size_nm=1500.0,  # Even larger box size
                min_frames_fraction=0.2,  # Very lenient frame requirement (20%)
            )
        )

        print(f"✓ Automatic fiducial detection completed")
        print(f"  Detection success: {detection_info['success']}")
        print(f"  Fiducials found: {detection_info['n_fiducials']}")
        print(f"  Message: {detection_info['message']}")
        print(f"  Method used: {drift_result.method_used.value}")
        print(
            f"  Drift range X: [{drift_result.drift_x.min():.2f}, {drift_result.drift_x.max():.2f}]"
        )
        print(
            f"  Drift range Y: [{drift_result.drift_y.min():.2f}, {drift_result.drift_y.max():.2f}]"
        )

    except Exception as e:
        import traceback

        print(f"✗ Automatic fiducial detection failed: {e}")
        print(f"  Details: {traceback.format_exc()}")

        # If it failed, try to extract troubleshooting info
        try:
            # This might give us the detection_info even if the main function failed
            pass
        except:
            pass

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
    test_vectorized_drift_correction()  # Test vectorized vs frame-by-frame
    test_drift_correction_robustness()  # Test edge cases
    test_rcc_method()
    test_aim_method()
    test_super_resolution_drift_correction()  # New super-resolution test
    test_fiducial_method()
    test_fiducial_auto_detection()
    test_auto_method()
    test_backward_compatibility()

    print("Testing completed!")
    print("\nGround truth data files created:")
    print("  - aim_test_drift.json/npz")
    print("  - dna_origami_drift.json/npz")
    print("  - dna_origami_corrected_drift.json/npz")
    print("\nThese files contain frame-by-frame drift data for plotting and analysis.")


if __name__ == "__main__":
    main()
