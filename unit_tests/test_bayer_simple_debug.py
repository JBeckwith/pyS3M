#!/usr/bin/env python3
"""Quick debug script to test single spot simulation and detection."""

import sys
import os
import numpy as np

module_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, module_dir)

import SpotDetectionFunctions
import BayerSpotDetection

# Create simple synthetic spot directly
print("Creating simple synthetic Gaussian spot...")

image_size = (128, 128)
spot_y, spot_x = 64, 64  # Center of image
print(f"Ground truth position: y={spot_y}, x={spot_x}")

# Create Gaussian spot
y_coords = np.arange(image_size[0])
x_coords = np.arange(image_size[1])
xx, yy = np.meshgrid(x_coords, y_coords)

# Simple Gaussian PSF
sigma = 1.5
intensity = 2000
background = 10

gaussian = intensity * np.exp(
    -((xx - spot_x)**2 + (yy - spot_y)**2) / (2 * sigma**2)
)

# Add Poisson noise
image = np.random.poisson(gaussian + background).astype(np.float32)

print(f"Image shape: {image.shape}")
print(f"Image stats: min={image.min()}, max={image.max()}, mean={image.mean():.1f}")

# Apply simple Bayer mask (RGGB)
bayer_image = np.zeros_like(image)
bayer_image[0::2, 0::2] = image[0::2, 0::2]  # R
bayer_image[0::2, 1::2] = image[0::2, 1::2]  # G
bayer_image[1::2, 0::2] = image[1::2, 0::2]  # G
bayer_image[1::2, 1::2] = image[1::2, 1::2]  # B

print(f"\nBayer image created")
print(f"Bayer stats: min={bayer_image.min()}, max={bayer_image.max()}, mean={bayer_image.mean():.1f}")

# Test Bayer-aware detection
print("\n" + "="*60)
print("Testing Bayer-Aware Detection")
print("="*60)

spot_detector = SpotDetectionFunctions.SpotDetection_Functions()

detections, metadata = BayerSpotDetection.detect_spots_bayer_multichannel(
    bayer_image[np.newaxis, :, :],  # Add frame dimension
    spot_detector=spot_detector,
    pattern='RGGB',
    pfa=1e-3,
    sigma=1.5,
    wavelength=0.58,
    pixel_size=0.069,
    NA=1.49
)

print(f"\nDetected {len(detections)} spots")

if len(detections) > 0:
    print("\nDetected positions:")
    for i, det in enumerate(detections):
        y_det, x_det, frame = det[:3]
        error = np.sqrt((y_det - spot_y)**2 + (x_det - spot_x)**2)
        print(f"  Spot {i}: y={y_det:.1f}, x={x_det:.1f}, frame={frame}, error={error:.2f} px")

    # Find closest detection
    distances = np.sqrt(
        (detections[:, 0] - spot_y)**2 +
        (detections[:, 1] - spot_x)**2
    )
    closest_idx = np.argmin(distances)
    closest_error = distances[closest_idx]

    print(f"\nClosest detection error: {closest_error:.2f} pixels")

    if closest_error < 3.0:
        print("✓ PASS: Position recovered within 3 pixels")
    else:
        print("✗ FAIL: Position error > 3 pixels")
else:
    print("✗ FAIL: No spots detected")

print("\nMetadata:")
print(f"  Per-channel detections: {metadata['n_detections']}")
print(f"  Merged detections: {metadata['n_detections_merged']}")
print(f"  Duplicates removed: {metadata['n_duplicates_removed']}")
