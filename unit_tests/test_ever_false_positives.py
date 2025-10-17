#!/usr/bin/env python3
"""
Test EVER analysis for false positive detection.

Creates simulated data with:
- Bright puncta (500 photons each)
- Low flat background (50 ADU) - easy to detect without EVER
- Random puncta positions per frame (no repeats)
- Ground truth tracking to verify EVER doesn't create spurious localizations

The key question: Does EVER find MORE puncta than actually exist?

Expected behavior:
- Standard: Should find ~10,000 localizations (10 per frame × 1000 frames)
- EVER: Should find ~10,000 localizations (same count, not more!)
- If EVER finds significantly MORE, it's creating false positives

Author: Claude Code (Anthropic)
Date: 2025-10-17
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import os
import tempfile
from pathlib import Path
from SR_Functions import SuperRes_Functions, TemporalMedianMode
from MaskFunctions import Mask_Functions
import tifffile
import json

print('='*80)
print('EVER False Positive Test: Ground Truth Validation')
print('='*80)

# Test parameters
n_frames_per_file = 500
n_files = 2
n_total_frames = n_frames_per_file * n_files
n_puncta_per_frame = 10
image_size = 64  # pixels (smaller for speed)
background_adu = 50  # LOW background - easy detection without EVER
puncta_photons = 500  # photons per puncta (BRIGHT spots)
puncta_sigma = 1.5  # pixels
ever_window = 100  # frames

# Camera parameters
gain = 0.5  # e-/ADU
offset = 100  # ADU
variance = 5.0  # ADU^2 (some noise)
readnoise = 2.0  # e-

print(f'\nTest Configuration:')
print(f'  Files: {n_files} × {n_frames_per_file} frames = {n_total_frames} total')
print(f'  Image size: {image_size}×{image_size} pixels')
print(f'  Background: {background_adu} ADU (LOW - easy detection)')
print(f'  Puncta per frame: {n_puncta_per_frame} (BRIGHT - {puncta_photons} photons each)')
print(f'  Expected total puncta: {n_total_frames * n_puncta_per_frame}')
print(f'  EVER window: {ever_window} frames')
print(f'  Signal/Background ratio: {puncta_photons/gain/background_adu:.1f}:1 (peak)')

# Create temporary directory
temp_dir = tempfile.mkdtemp(prefix='ever_falsepos_test_')
print(f'\nTemporary directory: {temp_dir}')

# Generate simulated data WITH GROUND TRUTH
print('\nGenerating simulated data with ground truth...')

np.random.seed(42)  # Reproducible

# Store ground truth positions
ground_truth = {
    'frame': [],
    'x': [],
    'y': [],
    'photons': []
}

for file_idx in range(n_files):
    print(f'  Creating file {file_idx+1}/{n_files}...')

    # Create stack
    stack = np.ones((n_frames_per_file, image_size, image_size), dtype=np.uint16)
    stack = (stack * background_adu).astype(np.uint16)

    for frame_idx in range(n_frames_per_file):
        frame = stack[frame_idx].astype(np.float64)

        global_frame_idx = file_idx * n_frames_per_file + frame_idx

        # Add puncta (Gaussian spots) at RANDOM positions each frame
        for puncta_idx in range(n_puncta_per_frame):
            # Random position (avoid edges)
            x = np.random.uniform(10, image_size-10)
            y = np.random.uniform(10, image_size-10)

            # Store ground truth
            ground_truth['frame'].append(global_frame_idx)
            ground_truth['x'].append(x)
            ground_truth['y'].append(y)
            ground_truth['photons'].append(puncta_photons)

            # Create Gaussian
            y_grid, x_grid = np.ogrid[0:image_size, 0:image_size]
            gaussian = puncta_photons * np.exp(
                -((x_grid - x)**2 + (y_grid - y)**2) / (2 * puncta_sigma**2)
            ) / (2 * np.pi * puncta_sigma**2)

            # Convert photons to ADU (gain = 0.5 e-/ADU)
            gaussian_adu = gaussian / gain

            frame += gaussian_adu

        # Add Gaussian noise (approximate Poisson + readnoise)
        frame += np.random.normal(0, np.sqrt(variance), frame.shape)

        stack[frame_idx] = np.clip(frame, 0, 65535).astype(np.uint16)

    # Save TIFF
    filename = os.path.join(temp_dir, f'simulated_{file_idx+1:03d}.tif')
    tifffile.imwrite(filename, stack, photometric='minisblack')
    print(f'    Saved: {filename} ({stack.shape[0]} frames)')

# Save ground truth to file
ground_truth_file = os.path.join(temp_dir, 'ground_truth.npz')
np.savez(
    ground_truth_file,
    frame=np.array(ground_truth['frame']),
    x=np.array(ground_truth['x']),
    y=np.array(ground_truth['y']),
    photons=np.array(ground_truth['photons'])
)
print(f'\nSaved ground truth: {ground_truth_file}')
print(f'  Total ground truth puncta: {len(ground_truth["frame"])}')

# Create metadata file (ImageJ JSON format)
metadata_file = os.path.join(temp_dir, 'metadata_simulated_001.txt')
metadata = {
    "FrameKey-0-0-0": {
        "ROI": f"0-0-{image_size}-{image_size}"  # y-x-width-height format
    },
    "Summary": {
        "IntendedDimensions": {
            "time": n_total_frames
        }
    }
}
with open(metadata_file, 'w') as f:
    json.dump(metadata, f, indent=2)

print('\n✓ Data generation complete')

# Setup camera calibration
print('\nSetting up camera calibration...')

M_F = Mask_Functions()
masks = M_F.get_masks(size_x=image_size, size_y=image_size)

camera_calibration = {
    'gain': np.ones((image_size, image_size)) * gain,
    'offset': np.ones((image_size, image_size)) * offset,
    'variance': np.ones((image_size, image_size)) * variance,
    'readnoise': np.ones((image_size, image_size)) * readnoise,
    'rqe': np.ones((image_size, image_size)),
    'masks': masks,
}

# Initialize SuperRes_Functions
srf = SuperRes_Functions()

# Create file list
tiff_files = sorted([os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith('.tif')])
print(f'Found {len(tiff_files)} TIFF files: {[os.path.basename(f) for f in tiff_files]}')

# Create smoothing function
import sCMOSFunctions
import types
scmos = sCMOSFunctions.sCMOS_Functions()
smoothing_function = types.SimpleNamespace()
smoothing_function.args = {"sigma": 1.5}
smoothing_function.extent = 1.5
smoothing_function.smoothing_function = scmos.gaussian_filter_stack
smoothing_function.data_arg = "image"

# Test 1: Standard analysis (no EVER)
print('\n' + '='*80)
print('TEST 1: Standard Analysis (No EVER)')
print('='*80)

print('\nRunning standard localization...')

try:
    # Run fit_imaging_data - saves to HDF5
    srf.fit_imaging_data(
        temp_dir,
        smoothing_function,
        gain_map=camera_calibration['gain'],
        offset_map=camera_calibration['offset'],
        rqe=camera_calibration['rqe'],
        read_noise=camera_calibration['readnoise'],
        variance=camera_calibration['variance'],
        pfa=1e-3,
        ROI_size=20,
        peak_wavelength=0.638,
        NA=1.49,
        pixel_size=0.1,  # microns
        sigma=1.5,
        fraction_true=0.2,
        image_type=".tif",
        use_variance_aware_demosaic=True,
        temporal_median_mode=TemporalMedianMode.NONE,
        ever_window=100,
    )

    # Load results from HDF5
    import pandas as pd
    results_standard = pd.read_hdf(os.path.join(temp_dir, 'Localisations.h5'), 'data')

    n_locs_standard = len(results_standard)
    print(f'\n✓ Standard analysis complete')
    print(f'  Localizations found: {n_locs_standard}')
    print(f'  Ground truth: {len(ground_truth["frame"])}')
    print(f'  Recovery rate: {n_locs_standard / len(ground_truth["frame"]) * 100:.1f}%')

    # Check frame coverage
    frames_found = results_standard['frame'].values
    unique_frames_std = np.unique(frames_found)
    print(f'  Unique frames: {len(unique_frames_std)} / {n_total_frames}')

    # Check locs per frame
    locs_per_frame = np.bincount(frames_found.astype(int), minlength=n_total_frames)
    avg_locs_per_frame = np.mean(locs_per_frame[locs_per_frame > 0])
    print(f'  Average locs/frame: {avg_locs_per_frame:.2f} (expected: {n_puncta_per_frame})')

except Exception as e:
    print(f'\n✗ Standard analysis FAILED: {e}')
    import traceback
    traceback.print_exc()
    results_standard = None
    n_locs_standard = 0

# Test 2: EVER analysis
print('\n' + '='*80)
print('TEST 2: EVER Analysis (100 Frame Window)')
print('='*80)

print('\nRunning EVER localization...')

try:
    # Run fit_imaging_data with EVER enabled
    srf.fit_imaging_data(
        temp_dir,
        smoothing_function,
        gain_map=camera_calibration['gain'],
        offset_map=camera_calibration['offset'],
        rqe=camera_calibration['rqe'],
        read_noise=camera_calibration['readnoise'],
        variance=camera_calibration['variance'],
        pfa=1e-3,
        ROI_size=20,
        peak_wavelength=0.638,
        NA=1.49,
        pixel_size=0.1,  # microns
        sigma=1.5,
        fraction_true=0.2,
        image_type=".tif",
        use_variance_aware_demosaic=True,
        temporal_median_mode=TemporalMedianMode.DETECTION_AND_FITTING,
        ever_window=ever_window,
    )

    # Load results from HDF5
    import pandas as pd
    results_ever = pd.read_hdf(os.path.join(temp_dir, 'Localisations_EVER.h5'), 'data')

    n_locs_ever = len(results_ever)
    print(f'\n✓ EVER analysis complete')
    print(f'  Localizations found: {n_locs_ever}')
    print(f'  Ground truth: {len(ground_truth["frame"])}')
    print(f'  Recovery rate: {n_locs_ever / len(ground_truth["frame"]) * 100:.1f}%')

    # Check frame coverage
    frames_found_ever = results_ever['frame'].values
    unique_frames_ever = np.unique(frames_found_ever)
    print(f'  Unique frames: {len(unique_frames_ever)} / {n_total_frames}')

    # Check locs per frame
    locs_per_frame_ever = np.bincount(frames_found_ever.astype(int), minlength=n_total_frames)
    avg_locs_per_frame_ever = np.mean(locs_per_frame_ever[locs_per_frame_ever > 0])
    print(f'  Average locs/frame: {avg_locs_per_frame_ever:.2f} (expected: {n_puncta_per_frame})')

    # Check for frames with TOO MANY localizations (potential false positives)
    frames_with_excess = np.where(locs_per_frame_ever > n_puncta_per_frame * 1.5)[0]
    if len(frames_with_excess) > 0:
        print(f'\n  ⚠️ WARNING: {len(frames_with_excess)} frames have >50% more locs than expected!')
        print(f'    Sample frames: {frames_with_excess[:10]}')
        print(f'    Locs in those frames: {locs_per_frame_ever[frames_with_excess[:10]]}')

except Exception as e:
    print(f'\n✗ EVER analysis FAILED: {e}')
    import traceback
    traceback.print_exc()
    results_ever = None
    n_locs_ever = 0

# Test 3: Compare to ground truth
print('\n' + '='*80)
print('GROUND TRUTH COMPARISON')
print('='*80)

if results_standard is not None and results_ever is not None:
    n_ground_truth = len(ground_truth['frame'])

    print(f'\nLocalization counts vs ground truth:')
    print(f'  Ground truth: {n_ground_truth}')
    print(f'  Standard:     {n_locs_standard} ({n_locs_standard/n_ground_truth*100:.1f}%)')
    print(f'  EVER:         {n_locs_ever} ({n_locs_ever/n_ground_truth*100:.1f}%)')

    ratio = n_locs_ever / n_locs_standard if n_locs_standard > 0 else 0
    print(f'\nEVER vs Standard:')
    print(f'  Ratio: {ratio:.3f}')

    # Check for false positives
    if n_locs_ever > n_ground_truth * 1.1:
        excess = n_locs_ever - n_ground_truth
        print(f'\n⚠️ CRITICAL: EVER found {excess} MORE localizations than ground truth!')
        print(f'  This indicates FALSE POSITIVES!')
        print(f'  False positive rate: {excess/n_locs_ever*100:.1f}%')
    elif n_locs_standard > n_ground_truth * 1.1:
        excess = n_locs_standard - n_ground_truth
        print(f'\n⚠️ CRITICAL: Standard found {excess} MORE localizations than ground truth!')
        print(f'  Both methods have false positives (not EVER-specific)')
        print(f'  False positive rate: {excess/n_locs_standard*100:.1f}%')
    else:
        print(f'\n✓ GOOD: Both methods close to ground truth (within 10%)')

    # Check if EVER creates MORE false positives than standard
    if n_locs_ever > n_locs_standard * 1.1:
        print(f'\n⚠️ WARNING: EVER found {(ratio-1)*100:.1f}% MORE locs than standard!')
        print(f'  EVER may be creating false positives')
    elif n_locs_ever < n_locs_standard * 0.9:
        print(f'\n⚠️ WARNING: EVER found {(1-ratio)*100:.1f}% FEWER locs than standard!')
        print(f'  EVER may be too aggressive (removing real spots)')
    else:
        print(f'\n✓ GOOD: EVER and Standard have similar counts (within 10%)')
        print(f'  No evidence of EVER-specific false positives')

    # Per-frame analysis
    print(f'\nPer-frame statistics:')
    locs_per_frame_std = np.bincount(results_standard['frame'].values.astype(int), minlength=n_total_frames)
    locs_per_frame_ever = np.bincount(results_ever['frame'].values.astype(int), minlength=n_total_frames)

    # Frames where EVER finds significantly more
    excess_frames = np.where(locs_per_frame_ever > locs_per_frame_std + 3)[0]
    if len(excess_frames) > 0:
        print(f'  Frames where EVER has ≥3 more locs than standard: {len(excess_frames)}')
        print(f'    Sample frames: {excess_frames[:10]}')
        print(f'    Standard counts: {locs_per_frame_std[excess_frames[:10]]}')
        print(f'    EVER counts:     {locs_per_frame_ever[excess_frames[:10]]}')
    else:
        print(f'  ✓ No frames with excessive EVER localizations')

else:
    print('\n✗ Cannot compare - one or both analyses failed')

# Cleanup
print('\n' + '='*80)
print('CLEANUP')
print('='*80)

print(f'\nRemoving temporary files from: {temp_dir}')
import shutil
shutil.rmtree(temp_dir)
print('✓ Cleanup complete')

print('\n' + '='*80)
print('TEST COMPLETE')
print('='*80)
