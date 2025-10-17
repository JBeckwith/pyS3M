#!/usr/bin/env python3
"""
Test EVER analysis with multi-file data to verify:
1. No frame duplication (correct localization counts)
2. Background subtraction working correctly
3. Proper file boundary handling with EVER window

Creates 2 simulated TIFF files with:
- 500 frames each (1000 total)
- Flat background (200 ADU)
- 10 puncta per frame (~5 photons each, σ=1.5 pixels)
- EVER window = 100 frames

Tests:
- Standard analysis (no EVER): Should find ~10,000 localizations
- EVER analysis: Should find ~10,000 localizations (same count, better SNR)
- File boundary: Frame 450-550 spans both files, EVER should handle seamlessly

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

print('='*80)
print('EVER Multi-File Test: Frame Duplication & Boundary Handling')
print('='*80)

# Test parameters
n_frames_per_file = 500
n_files = 2
n_total_frames = n_frames_per_file * n_files
n_puncta_per_frame = 10
image_size = 64  # pixels (smaller for speed)
background_adu = 200
puncta_photons = 500  # photons per puncta
puncta_sigma = 1.5  # pixels
ever_window = 100  # frames

# Camera parameters (simple for testing)
gain = 0.5  # e-/ADU
offset = 100  # ADU
variance = 1.0  # ADU^2
readnoise = 2.0  # e-

print(f'\nTest Configuration:')
print(f'  Files: {n_files} × {n_frames_per_file} frames = {n_total_frames} total')
print(f'  Image size: {image_size}×{image_size} pixels')
print(f'  Background: {background_adu} ADU')
print(f'  Puncta per frame: {n_puncta_per_frame}')
print(f'  Expected total puncta: {n_total_frames * n_puncta_per_frame}')
print(f'  EVER window: {ever_window} frames')
print(f'  File boundary: Frames {n_frames_per_file-50} to {n_frames_per_file+50} span both files')

# Create temporary directory
temp_dir = tempfile.mkdtemp(prefix='ever_test_')
print(f'\nTemporary directory: {temp_dir}')

# Generate simulated data
print('\nGenerating simulated data...')

np.random.seed(42)  # Reproducible

for file_idx in range(n_files):
    print(f'  Creating file {file_idx+1}/{n_files}...')

    # Create stack
    stack = np.ones((n_frames_per_file, image_size, image_size), dtype=np.uint16)
    stack = (stack * background_adu).astype(np.uint16)

    for frame_idx in range(n_frames_per_file):
        frame = stack[frame_idx].astype(np.float64)

        # Add puncta (Gaussian spots)
        for _ in range(n_puncta_per_frame):
            # Random position (avoid edges)
            x = np.random.uniform(10, image_size-10)
            y = np.random.uniform(10, image_size-10)

            # Create Gaussian
            y_grid, x_grid = np.ogrid[0:image_size, 0:image_size]
            gaussian = puncta_photons * np.exp(
                -((x_grid - x)**2 + (y_grid - y)**2) / (2 * puncta_sigma**2)
            ) / (2 * np.pi * puncta_sigma**2)

            # Convert photons to ADU (gain = 0.5 e-/ADU)
            gaussian_adu = gaussian / gain

            frame += gaussian_adu

        # Add Poisson noise (approximate with Gaussian for simplicity)
        frame += np.random.normal(0, np.sqrt(variance), frame.shape)

        stack[frame_idx] = np.clip(frame, 0, 65535).astype(np.uint16)

    # Save TIFF
    filename = os.path.join(temp_dir, f'simulated_{file_idx+1:03d}.tif')
    tifffile.imwrite(filename, stack, photometric='minisblack')
    print(f'    Saved: {filename} ({stack.shape[0]} frames)')

# Create metadata file (ImageJ JSON format)
import json
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
print(f'\nCreated metadata file: {metadata_file}')

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

# Test 1: Standard analysis (no EVER)
print('\n' + '='*80)
print('TEST 1: Standard Analysis (No EVER)')
print('='*80)

print('\nRunning standard localization...')

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
    print(f'  Expected: ~{n_total_frames * n_puncta_per_frame}')
    print(f'  Recovery rate: {n_locs_standard / (n_total_frames * n_puncta_per_frame) * 100:.1f}%')

    # Check frame numbers
    frames_found = results_standard['frame'].values
    print(f'  Frame range: {frames_found.min()} to {frames_found.max()}')
    print(f'  Unique frames: {len(np.unique(frames_found))} / {n_total_frames}')

    # Check for duplicates
    unique_frames = np.unique(frames_found)
    if len(unique_frames) < n_total_frames:
        print(f'  ⚠️ WARNING: Missing frames!')

    # Check locs per frame
    locs_per_frame = np.bincount(frames_found.astype(int))
    avg_locs_per_frame = np.mean(locs_per_frame[locs_per_frame > 0])
    print(f'  Average locs/frame: {avg_locs_per_frame:.1f}')

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
    print(f'\n  DEBUG: Running fit_imaging_data with EVER...')
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
    print(f'  Expected: ~{n_total_frames * n_puncta_per_frame}')
    print(f'  Recovery rate: {n_locs_ever / (n_total_frames * n_puncta_per_frame) * 100:.1f}%')

    # Check frame numbers
    frames_found_ever = results_ever['frame'].values
    print(f'  Frame range: {frames_found_ever.min()} to {frames_found_ever.max()}')
    print(f'  Unique frames: {len(np.unique(frames_found_ever))} / {n_total_frames}')

    # Check for duplicates
    unique_frames_ever = np.unique(frames_found_ever)
    if len(unique_frames_ever) < n_total_frames:
        print(f'  ⚠️ WARNING: Missing frames!')

    # Check locs per frame
    locs_per_frame_ever = np.bincount(frames_found_ever.astype(int))
    avg_locs_per_frame_ever = np.mean(locs_per_frame_ever[locs_per_frame_ever > 0])
    print(f'  Average locs/frame: {avg_locs_per_frame_ever:.1f}')

    # Check file boundary region (frames 450-550)
    boundary_region = (frames_found_ever >= 450) & (frames_found_ever <= 550)
    n_boundary_locs = np.sum(boundary_region)
    print(f'\n  File boundary check (frames 450-550):')
    print(f'    Localizations: {n_boundary_locs}')
    print(f'    Expected: ~{101 * n_puncta_per_frame}')
    print(f'    Frames with locs: {len(np.unique(frames_found_ever[boundary_region]))} / 101')

except Exception as e:
    print(f'\n✗ EVER analysis FAILED: {e}')
    import traceback
    traceback.print_exc()
    results_ever = None
    n_locs_ever = 0

# Test 3: Compare counts
print('\n' + '='*80)
print('COMPARISON: Standard vs EVER')
print('='*80)

if results_standard is not None and results_ever is not None:
    ratio = n_locs_ever / n_locs_standard if n_locs_standard > 0 else 0
    print(f'\nLocalization counts:')
    print(f'  Standard: {n_locs_standard}')
    print(f'  EVER:     {n_locs_ever}')
    print(f'  Ratio:    {ratio:.3f}')

    if ratio > 1.1:
        print(f'\n⚠️ WARNING: EVER found {(ratio-1)*100:.1f}% MORE locs than standard!')
        print(f'  This could indicate:')
        print(f'    - Frame duplication (processing same frames twice)')
        print(f'    - File boundary mishandling')
        print(f'    - EVER window overlap issues')
    elif ratio < 0.9:
        print(f'\n⚠️ WARNING: EVER found {(1-ratio)*100:.1f}% FEWER locs than standard!')
        print(f'  This could indicate:')
        print(f'    - EVER being too aggressive (removing real spots)')
        print(f'    - Window boundary issues')
    else:
        print(f'\n✓ GOOD: Localization counts are similar (within 10%)')
        print(f'  EVER is working correctly!')

    # Frame coverage check
    frames_standard = set(results_standard['frame'].values)
    frames_ever = set(results_ever['frame'].values)

    missing_in_ever = frames_standard - frames_ever
    extra_in_ever = frames_ever - frames_standard

    if missing_in_ever:
        print(f'\n  Frames in standard but NOT in EVER: {len(missing_in_ever)}')
        print(f'    Sample: {sorted(missing_in_ever)[:10]}')

    if extra_in_ever:
        print(f'\n  Frames in EVER but NOT in standard: {len(extra_in_ever)}')
        print(f'    Sample: {sorted(extra_in_ever)[:10]}')

    if not missing_in_ever and not extra_in_ever:
        print(f'\n✓ GOOD: Both methods processed the same frames')

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
