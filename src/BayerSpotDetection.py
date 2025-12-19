#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bayer-Patterned Spot Detection Wrapper

Applies existing spot detection methods to raw Bayer-patterned data by:
1. Extracting subsampled color channels
2. Running standard spot detection on each channel
3. Mapping coordinates back to full resolution

This preserves noise independence while reusing proven detection statistics.

Based on approach described in claude/spot_detection_analysis_bayeradaptation.md

Created: December 19, 2025
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
import warnings
import MaskFunctions


def get_mosaic_unit_from_pattern(pattern: str) -> np.ndarray:
    """Convert Bayer pattern string to mosaic unit array.

    Args:
        pattern: Bayer pattern string ('RGGB', 'GRBG', 'GBRG', 'BGGR')

    Returns:
        mosaic_unit: 2×2 array defining the Bayer pattern
    """
    pattern = pattern.upper()

    # Define mosaic units for standard Bayer patterns
    # Format: [[top-left, top-right], [bottom-left, bottom-right]]
    pattern_map = {
        'RGGB': np.array([['R', 'G'], ['G', 'B']]),
        'GRBG': np.array([['G', 'R'], ['B', 'G']]),
        'GBRG': np.array([['G', 'B'], ['R', 'G']]),
        'BGGR': np.array([['B', 'G'], ['G', 'R']])
    }

    if pattern not in pattern_map:
        raise ValueError(f"Invalid Bayer pattern: {pattern}. "
                       "Must be one of: {list(pattern_map.keys())}")

    return pattern_map[pattern]


def extract_bayer_channels(
    bayer_image: np.ndarray,
    pattern: str = 'RGGB'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Extract raw color channels from Bayer image without interpolation.

    Uses MaskFunctions to generate channel masks based on Bayer pattern.

    Args:
        bayer_image: Raw Bayer image (H×W)
        pattern: Bayer pattern ('RGGB', 'GRBG', 'GBRG', 'BGGR')

    Returns:
        red: Red channel samples (H/2 × W/2 for checkerboard)
        green: Green channel samples at quincunx positions
        blue: Blue channel samples (H/2 × W/2 for checkerboard)
        coord_info: Dictionary with coordinate mapping information and masks
    """
    H, W = bayer_image.shape
    pattern = pattern.upper()

    # Get mosaic unit for this pattern
    mosaic_unit = get_mosaic_unit_from_pattern(pattern)

    # Use MaskFunctions to generate masks
    mask_gen = MaskFunctions.Mask_Functions()
    masks = mask_gen.get_masks(H, W, mosaic_unit)

    # Extract channels using masks
    # For R and B (checkerboard): extract at 2:1 spacing
    # For G (quincunx): preserve alternating row structure
    red_mask = masks['R']
    green_mask = masks['G']
    blue_mask = masks['B']

    # Find first occurrence of each color to determine offsets
    red_y, red_x = np.where(red_mask)
    red_offset = (red_y[0], red_x[0])

    blue_y, blue_x = np.where(blue_mask)
    blue_offset = (blue_y[0], blue_x[0])

    # Extract red and blue on checkerboard grid
    red = bayer_image[red_mask].reshape(H//2, W//2)
    blue = bayer_image[blue_mask].reshape(H//2, W//2)

    # Extract green on quincunx grid (alternating rows)
    green = np.zeros((H, W//2), dtype=bayer_image.dtype)
    for row in range(H):
        green_row_mask = green_mask[row, :]
        green[row, :] = bayer_image[row, green_row_mask]

    coord_info = {
        'pattern': pattern,
        'mosaic_unit': mosaic_unit,
        'red_offset': red_offset,
        'blue_offset': blue_offset,
        'masks': masks
    }

    return red, green, blue, coord_info


def map_coordinates_to_full_resolution(
    detections: np.ndarray,
    channel: str,
    coord_info: Dict
) -> np.ndarray:
    """Map subsampled coordinates back to full Bayer image resolution.

    Args:
        detections: Array of detections with columns [y, x, frame, ...]
                   (format from detect_puncta_in_stack_parallel)
        channel: Color channel ('red', 'green', 'blue')
        coord_info: Coordinate mapping info from extract_bayer_channels

    Returns:
        detections_full: Detections with full-resolution coordinates [y, x, frame, ...]
    """
    if detections is None or len(detections) == 0:
        return detections

    detections_full = detections.copy()
    channel = channel.lower()

    # CRITICAL: detect_puncta_in_stack_parallel returns [y, x, frame, ...]
    # detections[:, 0] = y (row)
    # detections[:, 1] = x (col)
    # detections[:, 2] = frame

    # For checkerboard patterns (red, blue): 2× spacing
    if channel == 'red':
        y_offset, x_offset = coord_info['red_offset']
        detections_full[:, 0] = detections[:, 0] * 2 + y_offset  # y coords (row)
        detections_full[:, 1] = detections[:, 1] * 2 + x_offset  # x coords (col)

    elif channel == 'blue':
        y_offset, x_offset = coord_info['blue_offset']
        detections_full[:, 0] = detections[:, 0] * 2 + y_offset
        detections_full[:, 1] = detections[:, 1] * 2 + x_offset

    elif channel == 'green':
        # Quincunx pattern: alternating rows
        # Row i in green corresponds to row i in full image
        # But x-coordinate depends on row parity
        y_coords = detections[:, 0].astype(int)  # y is in column 0
        x_coords = detections[:, 1].astype(int)  # x is in column 1

        # For RGGB: green pixels alternate
        #   Even rows (0, 2, 4...): green at odd full columns (1, 3, 5...)
        #   Odd rows (1, 3, 5...): green at even full columns (0, 2, 4...)
        detections_full[:, 0] = y_coords  # y stays the same
        detections_full[:, 1] = x_coords * 2 + (1 - y_coords % 2)  # x depends on row

    else:
        raise ValueError(f"Unknown channel: {channel}")

    return detections_full


def merge_nearby_detections(
    detections_by_channel: Dict[str, np.ndarray],
    distance_threshold: float = 2.0
) -> np.ndarray:
    """Merge detections from different channels that are at the same physical location.

    When detecting on raw Bayer channels, the same fluorophore can appear in multiple
    color channels (especially for broadband emitters or due to spectral crosstalk).
    This function merges detections that are within distance_threshold pixels.

    Args:
        detections_by_channel: Dict mapping channel name to detections array [y, x, frame, ...]
        distance_threshold: Maximum distance (pixels) to consider detections as duplicates

    Returns:
        merged_detections: Combined array with duplicates removed [y, x, frame, ...]
    """
    # Combine all detections
    all_detections = []
    for channel, dets in detections_by_channel.items():
        if len(dets) > 0:
            all_detections.append(dets)

    if len(all_detections) == 0:
        return np.array([])

    combined = np.vstack(all_detections)

    if len(combined) == 0:
        return combined

    # Group by frame
    frames = combined[:, 2].astype(int)
    unique_frames = np.unique(frames)

    merged = []
    for frame in unique_frames:
        frame_mask = frames == frame
        frame_dets = combined[frame_mask]

        if len(frame_dets) == 0:
            continue

        # Use simple distance-based clustering
        # Keep track of which detections have been merged
        used = np.zeros(len(frame_dets), dtype=bool)

        for i in range(len(frame_dets)):
            if used[i]:
                continue

            # Find all detections within distance_threshold
            y, x = frame_dets[i, 0], frame_dets[i, 1]
            distances = np.sqrt((frame_dets[:, 0] - y)**2 + (frame_dets[:, 1] - x)**2)
            nearby = distances < distance_threshold

            # Merge nearby detections by averaging coordinates
            nearby_dets = frame_dets[nearby]
            merged_det = np.mean(nearby_dets, axis=0)
            merged_det[2] = frame  # Keep frame as integer

            merged.append(merged_det)
            used[nearby] = True

    return np.array(merged) if len(merged) > 0 else np.array([])


def detect_spots_bayer_multichannel(
    bayer_image: np.ndarray,
    spot_detector,  # SpotDetection_Functions instance
    pattern: str = 'RGGB',
    pfa: float = 1e-4,
    sigma: float = 1.5,
    variance: Optional[np.ndarray] = None,
    channels: Optional[List[str]] = None,
    merge_distance: float = 2.0,
    **kwargs
) -> Tuple[np.ndarray, Dict]:
    """Detect spots in raw Bayer image using per-channel detection.

    This function extracts color channels from raw Bayer data and applies
    standard spot detection to each channel independently, preserving noise
    independence.

    IMPORTANT: The same physical fluorophore can be detected in multiple color
    channels (due to broadband emission or spectral crosstalk). This function
    ALWAYS merges nearby detections across channels to avoid counting the same
    spot multiple times.

    Args:
        bayer_image: Raw Bayer-patterned image (can be 2D or 3D stack)
        spot_detector: Instance of SpotDetection_Functions class
        pattern: Bayer pattern string ('RGGB', 'GRBG', 'GBRG', 'BGGR')
        pfa: Probability of false alarm
        sigma: Threshold sigma multiplier for intensity filtering (dimensionless, typically 1.5)
        variance: Optional variance map for sCMOS noise
        channels: List of channels to detect (['red', 'green', 'blue'])
        merge_distance: Distance threshold (pixels) for merging duplicates (default: 2.0)
        **kwargs: Additional arguments passed to detect_puncta_in_stack_parallel
                  (e.g., wavelength, pixel_size, NA for PSF calculation)

    Returns:
        merged_detections: Array of merged detections [y, x, frame, ...]
        metadata: Dictionary with extraction info and statistics including per-channel counts
    """
    if channels is None:
        channels = ['red', 'green', 'blue']

    # Handle 3D stacks
    if bayer_image.ndim == 3:
        n_frames = bayer_image.shape[0]
        # Process first frame to get channel info
        red, green, blue, coord_info = extract_bayer_channels(
            bayer_image[0], pattern
        )
        # Stack all frames
        red_stack = np.zeros((n_frames,) + red.shape, dtype=bayer_image.dtype)
        green_stack = np.zeros((n_frames,) + green.shape, dtype=bayer_image.dtype)
        blue_stack = np.zeros((n_frames,) + blue.shape, dtype=bayer_image.dtype)

        for i in range(n_frames):
            r, g, b, _ = extract_bayer_channels(bayer_image[i], pattern)
            red_stack[i] = r
            green_stack[i] = g
            blue_stack[i] = b

        channel_data = {
            'red': red_stack,
            'green': green_stack,
            'blue': blue_stack
        }
    else:
        # Single frame
        red, green, blue, coord_info = extract_bayer_channels(bayer_image, pattern)
        # Add frame dimension for compatibility with detect_puncta_in_stack_parallel
        channel_data = {
            'red': red[np.newaxis, ...],
            'green': green[np.newaxis, ...],
            'blue': blue[np.newaxis, ...]
        }

    # Detect spots in each channel
    detections_by_channel = {}
    metadata = {
        'pattern': pattern,
        'coord_info': coord_info,
        'channel_shapes': {},
        'n_detections': {}
    }

    for channel in channels:
        channel_lower = channel.lower()
        if channel_lower not in channel_data:
            warnings.warn(f"Channel {channel} not found, skipping")
            continue

        data = channel_data[channel_lower]
        metadata['channel_shapes'][channel_lower] = data.shape

        print(f"Detecting spots in {channel} channel (shape: {data.shape})...")

        # NOTE: sigma is the threshold multiplier (dimensionless), NOT PSF width
        # It should be passed unchanged - PSF width is calculated internally from
        # wavelength, NA, and pixel_size in detect_puncta_in_stack_parallel

        # Extract variance for this channel if provided
        if variance is not None:
            if variance.ndim == 2:
                # Single variance map - subsample it
                _, _, _, coord_info_temp = extract_bayer_channels(variance, pattern)
                if channel_lower == 'red':
                    var_channel = variance[0::2, 0::2]
                elif channel_lower == 'blue':
                    var_channel = variance[1::2, 1::2]
                else:  # green
                    var_channel = np.zeros((variance.shape[0], variance.shape[1]//2))
                    var_channel[0::2, :] = variance[0::2, 1::2]
                    var_channel[1::2, :] = variance[1::2, 0::2]
            else:
                var_channel = None
        else:
            var_channel = None

        # Run detection on subsampled channel
        # Pass sigma unchanged (it's a dimensionless threshold multiplier)
        detections = spot_detector.detect_puncta_in_stack_parallel(
            data,
            variance=var_channel,
            pfa=pfa,
            sigma=sigma,
            **kwargs
        )

        # Map coordinates to full resolution
        if detections is not None and len(detections) > 0:
            detections_full = map_coordinates_to_full_resolution(
                detections, channel_lower, coord_info
            )
            detections_by_channel[channel_lower] = detections_full
            metadata['n_detections'][channel_lower] = len(detections_full)
            print(f"  Found {len(detections_full)} spots in {channel} channel")
        else:
            detections_by_channel[channel_lower] = np.array([])
            metadata['n_detections'][channel_lower] = 0
            print(f"  No spots found in {channel} channel")

    # Always merge nearby detections across channels
    print(f"\nMerging nearby detections across channels (distance < {merge_distance} px)...")
    merged = merge_nearby_detections(detections_by_channel, merge_distance)
    n_before = sum(metadata['n_detections'].values())
    n_after = len(merged)
    print(f"  Before merge: {n_before} total detections")
    print(f"  After merge: {n_after} total detections")
    print(f"  Removed {n_before - n_after} duplicates ({(n_before-n_after)/max(n_before,1)*100:.1f}%)")

    metadata['n_detections_merged'] = n_after
    metadata['n_duplicates_removed'] = n_before - n_after
    metadata['merge_distance'] = merge_distance

    return merged, metadata
