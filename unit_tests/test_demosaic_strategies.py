#!/usr/bin/env python3
"""
Test script to verify that all demosaicing strategies work correctly.

This tests the refactored sCMOSFunctions with multiple demosaicing strategies.
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sCMOSFunctions import sCMOS_Functions

def create_test_bayer_image(size=64):
    """Create a simple test Bayer pattern image."""
    # Create a simple pattern with some spots
    image = np.random.rand(size, size) * 10 + 100  # Background around 100

    # Add a few bright spots
    y_spots = [16, 32, 48]
    x_spots = [16, 32, 48]
    for y, x in zip(y_spots, x_spots):
        # Gaussian-ish spot
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                if 0 <= y+dy < size and 0 <= x+dx < size:
                    dist = np.sqrt(dy**2 + dx**2)
                    image[y+dy, x+dx] += 1000 * np.exp(-dist**2 / 2)

    return image.astype(np.float32)


def test_all_strategies():
    """Test that all demosaicing strategies work."""
    print("Testing all demosaicing strategies\n" + "="*60)

    # Create test data
    bayer_image = create_test_bayer_image(64)
    variance_map = np.ones_like(bayer_image) * 1.0  # Uniform variance
    offset_map = np.ones_like(bayer_image) * 100.0  # Uniform offset
    gain = 0.5  # ADU per photoelectron

    scmos = sCMOS_Functions()

    # Test each strategy
    strategies = ['malvar', 'bilinear', 'ddfapd', 'menon2007']

    for strategy in strategies:
        print(f"\nTesting strategy: {strategy}")
        print("-" * 40)

        try:
            # Test variance_aware_demosaic
            result = scmos.variance_aware_demosaic(
                CFA=bayer_image,
                variance_map=variance_map,
                offset_map=offset_map,
                gain=gain,
                grayscale=True,
                strategy=strategy
            )

            print(f"  ✓ variance_aware_demosaic: {result.shape}, range: [{result.min():.1f}, {result.max():.1f}]")

            # Test bayer_demosaic_stack
            rgb_result, gray_result = scmos.bayer_demosaic_stack(
                bayer_image,
                grayscale=True,
                strategy=strategy
            )

            print(f"  ✓ bayer_demosaic_stack RGB: {rgb_result.shape}")
            print(f"  ✓ bayer_demosaic_stack gray: {gray_result.shape}")

            # Test with 3D stack
            bayer_stack = np.stack([bayer_image, bayer_image], axis=0)
            rgb_stack, gray_stack = scmos.bayer_demosaic_stack(
                bayer_stack,
                grayscale=True,
                strategy=strategy
            )

            print(f"  ✓ bayer_demosaic_stack (3D) RGB: {rgb_stack.shape}")
            print(f"  ✓ bayer_demosaic_stack (3D) gray: {gray_stack.shape}")

        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            return False

    return True


def test_backward_compatibility():
    """Test that the old function name still works."""
    print("\n\nTesting backward compatibility\n" + "="*60)

    bayer_image = create_test_bayer_image(64)
    variance_map = np.ones_like(bayer_image) * 1.0
    offset_map = np.ones_like(bayer_image) * 100.0
    gain = 0.5

    scmos = sCMOS_Functions()

    try:
        # Old function name should still work
        result = scmos.variance_aware_malvar_demosaic(
            CFA=bayer_image,
            variance_map=variance_map,
            offset_map=offset_map,
            gain=gain,
            grayscale=True
        )

        print(f"✓ variance_aware_malvar_demosaic still works!")
        print(f"  Result shape: {result.shape}, range: [{result.min():.1f}, {result.max():.1f}]")
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_invalid_strategy():
    """Test that invalid strategy raises appropriate error."""
    print("\n\nTesting error handling\n" + "="*60)

    bayer_image = create_test_bayer_image(64)
    variance_map = np.ones_like(bayer_image) * 1.0

    scmos = sCMOS_Functions()

    try:
        result = scmos.variance_aware_demosaic(
            CFA=bayer_image,
            variance_map=variance_map,
            gain=0.5,
            grayscale=True,
            strategy='invalid_strategy'
        )
        print("✗ FAILED: Should have raised ValueError")
        return False
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")
        return True
    except Exception as e:
        print(f"✗ FAILED: Wrong exception type: {e}")
        return False


def test_strategy_comparison():
    """Quick visual comparison of different strategies."""
    print("\n\nStrategy Comparison\n" + "="*60)

    bayer_image = create_test_bayer_image(128)
    variance_map = np.ones_like(bayer_image) * 1.0
    offset_map = np.ones_like(bayer_image) * 100.0
    gain = 0.5

    scmos = sCMOS_Functions()
    strategies = ['malvar', 'bilinear', 'ddfapd', 'menon2007']

    results = {}
    for strategy in strategies:
        result = scmos.variance_aware_demosaic(
            CFA=bayer_image,
            variance_map=variance_map,
            offset_map=offset_map,
            gain=gain,
            grayscale=True,
            strategy=strategy
        )
        results[strategy] = result

        # Calculate simple metrics
        mean_val = np.mean(result)
        std_val = np.std(result)
        max_val = np.max(result)

        print(f"\n{strategy:12s}: mean={mean_val:6.1f}, std={std_val:5.1f}, max={max_val:7.1f}")

    # Compare differences between strategies
    print("\n" + "="*60)
    print("Comparing strategies (RMSE differences):")
    print("="*60)

    for i, s1 in enumerate(strategies):
        for s2 in strategies[i+1:]:
            diff = results[s1] - results[s2]
            rmse = np.sqrt(np.mean(diff**2))
            rel_rmse = rmse / np.mean(results[s1]) * 100
            print(f"{s1:12s} vs {s2:12s}: RMSE = {rmse:6.2f} ({rel_rmse:5.2f}%)")

    return True


if __name__ == '__main__':
    print("Testing demosaicing strategy refactoring\n")
    print("="*60)

    all_passed = True

    all_passed &= test_all_strategies()
    all_passed &= test_backward_compatibility()
    all_passed &= test_invalid_strategy()
    all_passed &= test_strategy_comparison()

    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("="*60)
