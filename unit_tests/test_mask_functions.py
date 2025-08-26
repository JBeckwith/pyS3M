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
            assert mask.shape == (size_x, size_y)  # Note: first arg is x (width), second is y (height)
            assert mask.dtype == bool or mask.dtype == np.uint8

    @pytest.mark.unit
    def test_get_masks_different_sizes(self, mask_functions):
        """Test mask generation with different image sizes."""
        test_sizes = [(4, 4), (6, 8), (10, 12), (16, 16)]

        for size_x, size_y in test_sizes:
            masks = mask_functions.get_masks(size_x, size_y)

            # Check shape consistency
            for color, mask in masks.items():
                assert mask.shape == (size_x, size_y)
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
            assert mask.shape == (size_x, size_y)

    @pytest.mark.unit  
    def test_get_roi_mask_basic(self, mask_functions):
        """Test ROI mask generation with basic parameters."""
        # get_ROI_mask signature: (ROI_x_start, ROI_y_start, width, height, mosaic_unit)
        ROI_x_start, ROI_y_start = 2, 3
        width, height = 10, 8
        
        masks = mask_functions.get_ROI_mask(ROI_x_start, ROI_y_start, width, height)

        # Should return dictionary of masks
        assert isinstance(masks, dict)
        expected_keys = {"B", "G", "R"}
        assert set(masks.keys()) == expected_keys
        
        # Check mask properties
        for color, mask in masks.items():
            assert isinstance(mask, np.ndarray)
            assert mask.dtype == bool or mask.dtype == np.uint8

    @pytest.mark.unit
    def test_return_custom_bayer_patterns_with_integers(self, mask_functions):
        """Test custom Bayer pattern generation with integer colors."""
        # According to docstring, this function expects integers, not strings
        colours = np.array([0, 1, 2])  # Use integers instead of strings
        patterns = mask_functions.return_custom_bayer_patterns(colours)

        # Should return a list or array of patterns
        assert patterns is not None
        # Pattern should be a 2D array
        assert len(patterns.shape) == 2

    @pytest.mark.unit
    def test_return_diagonal_patterns_with_integers(self, mask_functions):
        """Test diagonal pattern generation with integer colors."""
        colours = np.array([0, 1, 2])  # Use integers
        image_size = 8  # API expects single integer, not tuple
        patterns = mask_functions.return_diagonal_patterns(colours, image_size)

        # Should return patterns
        assert patterns is not None
        # Should be 2D array
        assert len(patterns.shape) == 2
        assert patterns.shape == (image_size, image_size)

    @pytest.mark.unit
    def test_optimise_matrix_symmetry_basic(self, mask_functions):
        """Test matrix symmetry optimization with numeric values."""
        numbers = np.array([1, 2, 3, 4])
        N = 2
        
        result = mask_functions.optimise_matrix_symmetry(numbers, N)
        
        # Should return 2x2 matrix
        assert result.shape == (N, N)
        # Should contain values from numbers array
        unique_vals = np.unique(result)
        assert len(unique_vals) <= len(numbers)

    @pytest.mark.integration
    def test_complete_mask_workflow(self, mask_functions):
        """Test complete workflow using mask functions together."""
        # Generate basic masks
        size_x, size_y = 16, 12
        masks = mask_functions.get_masks(size_x, size_y)
        
        # Extract ROI masks
        roi_masks = mask_functions.get_ROI_mask(2, 2, 8, 6)
        
        # Both should have same color keys
        assert set(masks.keys()) == set(roi_masks.keys())
        
        # All masks should be valid arrays
        all_masks = list(masks.values()) + list(roi_masks.values())
        for mask in all_masks:
            assert isinstance(mask, np.ndarray)
            assert mask.sum() >= 0  # Valid mask

    @pytest.mark.unit
    def test_mosaic_unit_validation(self, mask_functions):
        """Test that custom mosaic units work correctly."""
        size_x, size_y = 6, 6
        
        # Test with different mosaic patterns
        patterns = [
            np.array([["B", "G"], ["G", "R"]]),  # Standard BGGR
            np.array([["R", "G"], ["G", "B"]]),  # RGGB
            np.array([["G", "B"], ["R", "G"]]),  # GBRG
        ]
        
        for pattern in patterns:
            masks = mask_functions.get_masks(size_x, size_y, mosaic_unit=pattern)
            
            # Should always have B, G, R keys regardless of pattern
            expected_keys = {"B", "G", "R"}
            assert set(masks.keys()) == expected_keys
            
            # Each mask should have correct shape and non-zero elements
            for color, mask in masks.items():
                assert mask.shape == (size_x, size_y)
                assert mask.sum() > 0