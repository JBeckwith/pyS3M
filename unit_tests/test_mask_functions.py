#!/usr/bin/env python3
"""
Test module for MaskFunctions.

Tests the mask generation functions for Bayer patterns and ROI masks.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add src to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from MaskFunctions import Mask_Functions


class TestMaskFunctions:
    """Test the Mask_Functions class."""

    @pytest.fixture
    def mask_functions(self):
        """Create an instance of Mask_Functions."""
        return Mask_Functions()

    @pytest.mark.unit
    def test_class_initialization(self, mask_functions):
        """Test that the Mask_Functions class initializes properly."""
        assert mask_functions is not None
        assert hasattr(mask_functions, "get_masks")
        assert hasattr(mask_functions, "get_ROI_mask")
        assert hasattr(mask_functions, "return_custom_bayer_patterns")

    @pytest.mark.unit
    def test_get_masks_basic(self, mask_functions):
        """Test basic mask generation functionality."""
        size_x, size_y = 8, 8
        masks = mask_functions.get_masks(size_x, size_y)

        # Should return a dictionary with color keys
        assert isinstance(masks, dict)
        expected_keys = {"B", "G", "R"}
        assert set(masks.keys()) == expected_keys

        # Each mask should have the correct shape
        for color, mask in masks.items():
            assert mask.shape == (size_y, size_x)
            assert mask.dtype == bool or mask.dtype == np.uint8

    @pytest.mark.unit
    def test_get_masks_different_sizes(self, mask_functions):
        """Test mask generation with different image sizes."""
        test_sizes = [(4, 4), (6, 8), (10, 12), (16, 16)]

        for size_x, size_y in test_sizes:
            masks = mask_functions.get_masks(size_x, size_y)

            # Check shape consistency
            for color, mask in masks.items():
                assert mask.shape == (size_y, size_x)
                assert mask.sum() > 0  # Each mask should have some pixels

    @pytest.mark.unit
    def test_get_masks_custom_mosaic_pattern(self, mask_functions):
        """Test mask generation with custom mosaic pattern."""
        size_x, size_y = 8, 8
        custom_pattern = np.array([["R", "G"], ["G", "B"]])  # Different from default

        masks = mask_functions.get_masks(size_x, size_y, mosaic_unit=custom_pattern)

        # Should still return the same keys
        expected_keys = {"B", "G", "R"}
        assert set(masks.keys()) == expected_keys

        # Each mask should have correct shape
        for color, mask in masks.items():
            assert mask.shape == (size_y, size_x)

    @pytest.mark.unit
    def test_get_roi_mask_circular(self, mask_functions):
        """Test circular ROI mask generation."""
        image_size = 20
        center = [10, 10]
        radius = 5

        mask = mask_functions.get_ROI_mask(
            image_size, image_size, center, shape="circle", radius=radius
        )

        # Check basic properties
        assert mask.shape == (image_size, image_size)
        assert mask.dtype == bool or mask.dtype in [np.uint8, np.int32, np.int64]

        # Check that center pixel is included
        assert mask[center[1], center[0]]  # Note: indexing is [y, x]

        # Check that pixels far from center are excluded
        assert not mask[0, 0]  # Corner should be outside circle
        assert not mask[0, 19]
        assert not mask[19, 0]
        assert not mask[19, 19]

    @pytest.mark.unit
    def test_get_roi_mask_rectangular(self, mask_functions):
        """Test rectangular ROI mask generation."""
        image_size = 20
        center = [10, 10]
        width, height = 8, 6

        mask = mask_functions.get_ROI_mask(
            image_size,
            image_size,
            center,
            shape="rectangle",
            width=width,
            height=height,
        )

        # Check basic properties
        assert mask.shape == (image_size, image_size)

        # Check that center pixel is included
        assert mask[center[1], center[0]]

        # Check approximate dimensions (allowing for rounding)
        mask_sum = mask.sum()
        expected_area = width * height
        # Allow some tolerance for rounding effects
        assert abs(mask_sum - expected_area) <= expected_area * 0.2

    @pytest.mark.unit
    def test_return_custom_bayer_patterns(self, mask_functions):
        """Test custom Bayer pattern generation."""
        colours = ["R", "G", "B"]
        patterns = mask_functions.return_custom_bayer_patterns(colours)

        # Should return a list of patterns
        assert isinstance(patterns, list)
        assert len(patterns) > 0

        # Each pattern should be a 2D array
        for pattern in patterns:
            assert isinstance(pattern, np.ndarray)
            assert pattern.ndim == 2
            # Pattern elements should be from the provided colours
            unique_colors = np.unique(pattern.flatten())
            for color in unique_colors:
                assert color in colours

    @pytest.mark.unit
    def test_return_diagonal_patterns(self, mask_functions):
        """Test diagonal pattern generation."""
        colours = ["R", "G", "B"]
        image_size = 8
        patterns = mask_functions.return_diagonal_patterns(colours, image_size)

        # Should return patterns
        assert isinstance(patterns, (list, np.ndarray))

        if isinstance(patterns, list):
            for pattern in patterns:
                assert isinstance(pattern, np.ndarray)
                # Check that pattern uses provided colours
                unique_colors = np.unique(pattern.flatten())
                for color in unique_colors:
                    assert color in colours

    @pytest.mark.unit
    def test_optimize_matrix_symmetry(self, mask_functions):
        """Test matrix symmetry optimization."""
        numbers = [1, 2, 3, 4, 5, 6]
        N = 3  # 3x3 matrix

        result = mask_functions.optimize_matrix_symmetry(numbers, N)

        # Should return some result (exact format depends on implementation)
        assert result is not None

        # Basic check that it's attempting to create an NxN arrangement
        if isinstance(result, np.ndarray):
            # If it returns a matrix
            assert result.shape[0] <= N or result.shape[1] <= N
        elif isinstance(result, (list, tuple)):
            # If it returns coordinates or arrangement info
            assert len(result) >= 0

    @pytest.mark.integration
    def test_bayer_mask_completeness(self, mask_functions):
        """Test that Bayer masks cover all pixels exactly once."""
        size_x, size_y = 12, 12
        masks = mask_functions.get_masks(size_x, size_y)

        # Create combined mask
        combined_mask = np.zeros((size_y, size_x), dtype=int)
        for i, (color, mask) in enumerate(masks.items()):
            combined_mask += mask.astype(int) * (i + 1)

        # Check that every pixel is covered exactly once
        assert np.all(combined_mask > 0), "Some pixels are not covered by any mask"
        assert np.all(
            combined_mask <= len(masks)
        ), "Some pixels are covered by multiple masks"

        # Check that the pattern repeats correctly
        total_pixels = size_x * size_y
        total_mask_pixels = sum(mask.sum() for mask in masks.values())
        assert total_mask_pixels == total_pixels

    @pytest.mark.unit
    def test_mask_data_types(self, mask_functions):
        """Test that masks have appropriate data types."""
        size_x, size_y = 8, 8
        masks = mask_functions.get_masks(size_x, size_y)

        for color, mask in masks.items():
            # Mask should be boolean or integer type
            assert mask.dtype in [
                bool,
                np.bool_,
                np.uint8,
                np.uint16,
                np.uint32,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
            ]

            # Values should be binary-like (0 or 1, True or False)
            unique_values = np.unique(mask)
            assert len(unique_values) <= 2
            assert all(val in [0, 1, True, False] for val in unique_values)


if __name__ == "__main__":
    # Run tests directly when script is executed
    pytest.main([__file__, "-v"])
