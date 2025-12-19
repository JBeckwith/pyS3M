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
        detections: Array of detections with columns [frame, y, x, ...]
        channel: Color channel ('red', 'green', 'blue')
        coord_info: Coordinate mapping info from extract_bayer_channels

    Returns:
        detections_full: Detections with full-resolution coordinates
    """
    if detections is None or len(detections) == 0:
        return detections

    detections_full = detections.copy()
    channel = channel.lower()

    # For checkerboard patterns (red, blue): 2× spacing
    if channel == 'red':
        y_offset, x_offset = coord_info['red_offset']
        detections_full[:, 1] = detections[:, 1] * 2 + y_offset  # y coords
        detections_full[:, 2] = detections[:, 2] * 2 + x_offset  # x coords

    elif channel == 'blue':
        y_offset, x_offset = coord_info['blue_offset']
        detections_full[:, 1] = detections[:, 1] * 2 + y_offset
        detections_full[:, 2] = detections[:, 2] * 2 + x_offset

    elif channel == 'green':
        # Quincunx pattern: alternating rows
        # Row i in green corresponds to row i in full image
        # But x-coordinate depends on row parity
        y_coords = detections[:, 1].astype(int)
        x_coords = detections[:, 2].astype(int)

        # For RGGB: green pixels alternate
        #   Even rows (0, 2, 4...): green at odd full columns (1, 3, 5...)
        #   Odd rows (1, 3, 5...): green at even full columns (0, 2, 4...)
        detections_full[:, 1] = y_coords  # y stays the same
        detections_full[:, 2] = x_coords * 2 + (1 - y_coords % 2)  # x depends on row

    else:
        raise ValueError(f"Unknown channel: {channel}")

    return detections_full


def detect_spots_bayer_multichannel(
    bayer_image: np.ndarray,
    spot_detector,  # SpotDetection_Functions instance
    pattern: str = 'RGGB',
    pfa: float = 1e-4,
    sigma: float = 1.5,
    variance: Optional[np.ndarray] = None,
    channels: Optional[List[str]] = None,
    **kwargs
) -> Tuple[Dict[str, np.ndarray], Dict]:
    """Detect spots in raw Bayer image using per-channel detection.

    This function extracts color channels from raw Bayer data and applies
    standard spot detection to each channel independently, preserving noise
    independence.

    Args:
        bayer_image: Raw Bayer-patterned image (can be 2D or 3D stack)
        spot_detector: Instance of SpotDetection_Functions class
        pattern: Bayer pattern string ('RGGB', 'GRBG', 'GBRG', 'BGGR')
        pfa: Probability of false alarm
        sigma: PSF width in pixels (full resolution)
        variance: Optional variance map for sCMOS noise
        channels: List of channels to detect (['red', 'green', 'blue'])
        **kwargs: Additional arguments passed to detect_puncta_in_stack_parallel

    Returns:
        detections_by_channel: Dictionary mapping channel name to detection array
        metadata: Dictionary with extraction info and statistics
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

        # For subsampled data, PSF sigma needs adjustment
        # Checkerboard (R,B): 2× sampling → effective sigma is sigma/2
        # Quincunx (G): sqrt(2)× sampling → effective sigma is sigma/sqrt(2)
        if channel_lower in ['red', 'blue']:
            sigma_effective = sigma / 2.0
        else:  # green
            sigma_effective = sigma / np.sqrt(2.0)

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
        detections = spot_detector.detect_puncta_in_stack_parallel(
            data,
            variance=var_channel,
            pfa=pfa,
            sigma=sigma_effective,
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

    return detections_by_channel, metadata
