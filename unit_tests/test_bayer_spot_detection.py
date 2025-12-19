#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script comparing Bayer-specific spot detection vs demosaiced detection.

Compares two approaches:
1. Standard: Demosaic → Detect on RGB channels
2. Bayer: Extract raw channels → Detect on subsampled data → Map coordinates

Created: December 19, 2025
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import time

# Add src to path
module_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, module_dir)

from SpotDetectionFunctions import SpotDetection_Functions
from BayerSpotDetection import detect_spots_bayer_multichannel, extract_bayer_channels
import IOFunctions

def load_test_data(data_path):
    """Load experimental data from OME-TIFF file.

    Args:
        data_path: Path to .ome.tif file

    Returns:
        data: Image stack (frames × H × W)
        metadata: Dictionary with file metadata
    """
    print(f"Loading data from: {data_path}")

    with tiff.TiffFile(data_path) as tif:
        data = tif.asarray()

        # Try to get metadata
        try:
            metadata = tif.ome_metadata
        except:
            metadata = None

    print(f"  Data shape: {data.shape}")
    print(f"  Data dtype: {data.dtype}")
    print(f"  Data range: [{data.min()}, {data.max()}]")

    return data, metadata


def demosaic_and_split_channels(bayer_stack, pattern='RGGB'):
    """Demosaic Bayer stack and split into RGB channels.

    Args:
        bayer_stack: Raw Bayer image stack (frames × H × W)
        pattern: Bayer pattern

    Returns:
        red_stack: Red channel stack
        green_stack: Green channel stack
        blue_stack: Blue channel stack
    """
    print("Demosaicing Bayer stack...")

    n_frames = bayer_stack.shape[0]
    H, W = bayer_stack.shape[1:]

    # Initialize RGB stacks
    red_stack = np.zeros((n_frames, H, W), dtype=np.float32)
    green_stack = np.zeros((n_frames, H, W), dtype=np.float32)
    blue_stack = np.zeros((n_frames, H, W), dtype=np.float32)

    # Simple bilinear demosaicing for each frame
    for i in range(n_frames):
        if i % 100 == 0:
            print(f"  Processing frame {i}/{n_frames}")

        frame = bayer_stack[i].astype(np.float32)

        if pattern == 'RGGB':
            # Extract raw channels
            red_raw = np.zeros_like(frame)
            green_raw = np.zeros_like(frame)
            blue_raw = np.zeros_like(frame)

            red_raw[0::2, 0::2] = frame[0::2, 0::2]
            green_raw[0::2, 1::2] = frame[0::2, 1::2]
            green_raw[1::2, 0::2] = frame[1::2, 0::2]
            blue_raw[1::2, 1::2] = frame[1::2, 1::2]

            # Simple interpolation (average neighbors)
            from scipy.ndimage import convolve

            # Interpolation kernels
            kernel_rb = np.array([[0.25, 0.5, 0.25],
                                 [0.5,  0,   0.5],
                                 [0.25, 0.5, 0.25]])
            kernel_g = np.array([[0, 0.25, 0],
                                [0.25, 0, 0.25],
                                [0, 0.25, 0]])

            # Interpolate
            red_interp = convolve(red_raw, kernel_rb, mode='nearest')
            red_stack[i] = np.where(red_raw > 0, red_raw, red_interp)

            green_interp = convolve(green_raw, kernel_g, mode='nearest')
            green_stack[i] = np.where(green_raw > 0, green_raw, green_interp)

            blue_interp = convolve(blue_raw, kernel_rb, mode='nearest')
            blue_stack[i] = np.where(blue_raw > 0, blue_raw, blue_interp)

    print(f"  Demosaiced shapes: R={red_stack.shape}, G={green_stack.shape}, B={blue_stack.shape}")

    return red_stack, green_stack, blue_stack


def run_standard_detection(rgb_stacks, spot_detector, pfa=1e-4, sigma=1.5):
    """Run standard spot detection on demosaiced RGB channels.

    Args:
        rgb_stacks: Tuple of (red_stack, green_stack, blue_stack)
        spot_detector: SpotDetection_Functions instance
        pfa: Probability of false alarm
        sigma: PSF width

    Returns:
        detections_by_channel: Dictionary mapping channel to detection array
        timing: Detection timing info
    """
    print("\n" + "="*60)
    print("STANDARD DETECTION (Demosaiced)")
    print("="*60)

    red_stack, green_stack, blue_stack = rgb_stacks
    detections_by_channel = {}
    timing = {}

    for channel_name, data in [('red', red_stack), ('green', green_stack), ('blue', blue_stack)]:
        print(f"\nDetecting in {channel_name} channel...")
        print(f"  Shape: {data.shape}")

        t_start = time.time()

        detections = spot_detector.detect_puncta_in_stack_parallel(
            data,
            pfa=pfa,
            sigma=sigma,
            fraction_true=0.2,
            return_quality=False
        )

        t_end = time.time()
        timing[channel_name] = t_end - t_start

        if detections is not None and len(detections) > 0:
            detections_by_channel[channel_name] = detections
            print(f"  Found {len(detections)} spots ({timing[channel_name]:.2f}s)")
        else:
            detections_by_channel[channel_name] = np.array([])
            print(f"  No spots found ({timing[channel_name]:.2f}s)")

    total_time = sum(timing.values())
    total_spots = sum(len(d) for d in detections_by_channel.values())
    print(f"\nTotal: {total_spots} spots in {total_time:.2f}s")

    return detections_by_channel, timing


def run_bayer_detection(bayer_stack, spot_detector, pattern='RGGB', pfa=1e-4, sigma=1.5):
    """Run Bayer-specific spot detection on raw data.

    Args:
        bayer_stack: Raw Bayer image stack
        spot_detector: SpotDetection_Functions instance
        pattern: Bayer pattern
        pfa: Probability of false alarm
        sigma: PSF width (full resolution)

    Returns:
        detections_by_channel: Dictionary mapping channel to detection array
        metadata: Detection metadata
        timing: Detection timing info
    """
    print("\n" + "="*60)
    print("BAYER DETECTION (Raw channels)")
    print("="*60)

    t_start = time.time()

    detections_by_channel, metadata = detect_spots_bayer_multichannel(
        bayer_stack,
        spot_detector,
        pattern=pattern,
        pfa=pfa,
        sigma=sigma,
        fraction_true=0.2,
        return_quality=False
    )

    t_end = time.time()
    total_time = t_end - t_start

    total_spots = sum(len(d) for d in detections_by_channel.values())
    print(f"\nTotal: {total_spots} spots in {total_time:.2f}s")

    timing = {'total': total_time}

    return detections_by_channel, metadata, timing


def compare_detections(standard_dets, bayer_dets, channels=['red', 'green', 'blue']):
    """Compare detection results between standard and Bayer approaches.

    Args:
        standard_dets: Detections from standard approach
        bayer_dets: Detections from Bayer approach
        channels: List of channels to compare

    Returns:
        comparison: Dictionary with comparison statistics
    """
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)

    comparison = {}

    for channel in channels:
        std = standard_dets.get(channel, np.array([]))
        bayer = bayer_dets.get(channel, np.array([]))

        n_std = len(std)
        n_bayer = len(bayer)

        print(f"\n{channel.upper()} channel:")
        print(f"  Standard: {n_std} spots")
        print(f"  Bayer:    {n_bayer} spots")
        print(f"  Difference: {n_bayer - n_std} ({(n_bayer-n_std)/max(n_std, 1)*100:+.1f}%)")

        comparison[channel] = {
            'n_standard': n_std,
            'n_bayer': n_bayer,
            'difference': n_bayer - n_std,
            'percent_diff': (n_bayer - n_std) / max(n_std, 1) * 100
        }

    total_std = sum(comp['n_standard'] for comp in comparison.values())
    total_bayer = sum(comp['n_bayer'] for comp in comparison.values())

    print(f"\nTOTAL:")
    print(f"  Standard: {total_std} spots")
    print(f"  Bayer:    {total_bayer} spots")
    print(f"  Difference: {total_bayer - total_std} ({(total_bayer-total_std)/max(total_std, 1)*100:+.1f}%)")

    comparison['total'] = {
        'n_standard': total_std,
        'n_bayer': total_bayer,
        'difference': total_bayer - total_std,
        'percent_diff': (total_bayer - total_std) / max(total_std, 1) * 100
    }

    return comparison


def visualize_detections(bayer_frame, standard_dets, bayer_dets, frame_idx=0, save_path=None):
    """Visualize detection results for a single frame.

    Args:
        bayer_frame: Raw Bayer frame
        standard_dets: Standard detections dictionary
        bayer_dets: Bayer detections dictionary
        frame_idx: Frame index to visualize
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Define colors for channels
    channel_colors = {'red': 'red', 'green': 'lime', 'blue': 'blue'}

    # Plot 1: Raw Bayer image with standard detections
    ax = axes[0]
    ax.imshow(bayer_frame, cmap='gray', vmin=np.percentile(bayer_frame, 1),
             vmax=np.percentile(bayer_frame, 99))

    for channel, dets in standard_dets.items():
        if len(dets) > 0:
            frame_mask = dets[:, 0] == frame_idx
            frame_dets = dets[frame_mask]
            if len(frame_dets) > 0:
                ax.scatter(frame_dets[:, 2], frame_dets[:, 1],
                          c=channel_colors[channel], s=100, alpha=0.5,
                          marker='o', linewidths=1, edgecolors='white',
                          label=f'{channel} ({len(frame_dets)})')

    ax.set_title(f'Standard Detection (Demosaiced)\nFrame {frame_idx}', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.axis('off')

    # Plot 2: Raw Bayer image with Bayer detections
    ax = axes[1]
    ax.imshow(bayer_frame, cmap='gray', vmin=np.percentile(bayer_frame, 1),
             vmax=np.percentile(bayer_frame, 99))

    for channel, dets in bayer_dets.items():
        if len(dets) > 0:
            frame_mask = dets[:, 0] == frame_idx
            frame_dets = dets[frame_mask]
            if len(frame_dets) > 0:
                ax.scatter(frame_dets[:, 2], frame_dets[:, 1],
                          c=channel_colors[channel], s=100, alpha=0.5,
                          marker='s', linewidths=1, edgecolors='white',
                          label=f'{channel} ({len(frame_dets)})')

    ax.set_title(f'Bayer Detection (Raw Channels)\nFrame {frame_idx}', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.axis('off')

    # Plot 3: Overlay comparison
    ax = axes[2]
    ax.imshow(bayer_frame, cmap='gray', vmin=np.percentile(bayer_frame, 1),
             vmax=np.percentile(bayer_frame, 99))

    for channel in ['red', 'green', 'blue']:
        std_dets = standard_dets.get(channel, np.array([]))
        bay_dets = bayer_dets.get(channel, np.array([]))

        if len(std_dets) > 0:
            frame_mask = std_dets[:, 0] == frame_idx
            frame_dets = std_dets[frame_mask]
            if len(frame_dets) > 0:
                ax.scatter(frame_dets[:, 2], frame_dets[:, 1],
                          c=channel_colors[channel], s=120, alpha=0.3,
                          marker='o', linewidths=2, edgecolors='white',
                          label=f'{channel} std')

        if len(bay_dets) > 0:
            frame_mask = bay_dets[:, 0] == frame_idx
            frame_dets = bay_dets[frame_mask]
            if len(frame_dets) > 0:
                ax.scatter(frame_dets[:, 2], frame_dets[:, 1],
                          c=channel_colors[channel], s=60, alpha=0.7,
                          marker='x', linewidths=2,
                          label=f'{channel} bayer')

    ax.set_title(f'Overlay Comparison\nCircles=Standard, X=Bayer', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9, fontsize=8)
    ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved figure to: {save_path}")

    return fig


def main():
    """Main test function."""
    # Test parameters
    data_path = '/media/jbeckwith/Ezra Seagat/test_script/20mW638_10p561_NF_SP_1_MMStack_2-Pos000_000.ome.tif'
    pattern = 'RGGB'  # Assumed pattern - verify with camera specs
    pfa = 1e-4
    sigma = 1.5

    # Use subset of frames for faster testing
    n_frames_test = 10

    print("="*60)
    print("BAYER SPOT DETECTION COMPARISON TEST")
    print("="*60)
    print(f"Data: {os.path.basename(data_path)}")
    print(f"Pattern: {pattern}")
    print(f"PFA: {pfa}")
    print(f"Sigma: {sigma}")
    print(f"Test frames: {n_frames_test}")

    # Load data
    data, metadata = load_test_data(data_path)

    # Use subset for testing
    if data.ndim == 3:
        data = data[:n_frames_test]
    elif data.ndim == 2:
        data = data[np.newaxis, ...]  # Add frame dimension

    # Initialize spot detector
    spot_detector = SpotDetection_Functions()

    # Method 1: Standard demosaic + detect
    print("\n" + "="*60)
    print("METHOD 1: DEMOSAIC + DETECT")
    print("="*60)
    red_stack, green_stack, blue_stack = demosaic_and_split_channels(data, pattern)
    standard_dets, standard_timing = run_standard_detection(
        (red_stack, green_stack, blue_stack), spot_detector, pfa, sigma
    )

    # Method 2: Bayer-specific detection
    print("\n" + "="*60)
    print("METHOD 2: BAYER-SPECIFIC DETECTION")
    print("="*60)
    bayer_dets, bayer_metadata, bayer_timing = run_bayer_detection(
        data, spot_detector, pattern, pfa, sigma
    )

    # Compare results
    comparison = compare_detections(standard_dets, bayer_dets)

    # Visualize results for first frame
    save_dir = os.path.dirname(__file__)
    save_path = os.path.join(save_dir, 'bayer_detection_comparison.png')
    fig = visualize_detections(data[0], standard_dets, bayer_dets, 0, save_path)
    plt.show()

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
