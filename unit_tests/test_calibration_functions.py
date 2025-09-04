#!/usr/bin/env python3
"""
Test module for CalibrationFunctions.

Tests camera calibration functions including gain, offset, variance map generation,
and sCMOS camera calibration workflows.
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
import sys
from unittest.mock import Mock, patch, MagicMock

# Add src to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from CalibrationFunctions import Calibration_Functions


class TestCalibrationFunctions:
    """Test main Calibration_Functions class."""

    @pytest.fixture
    def calibration_functions(self):
        """Create Calibration_Functions instance."""
        return Calibration_Functions()

    @pytest.fixture
    def sample_camera_params(self):
        """Create sample camera parameters."""
        height, width = 64, 64
        return {
            "height": height,
            "width": width,
            "gain": np.random.uniform(0.8, 1.2, (height, width)).astype(np.float32),
            "offset": np.random.uniform(100, 200, (height, width)).astype(np.float32),
            "variance": np.random.uniform(1, 5, (height, width)).astype(np.float32),
            "readnoise": 1.2,
            "pixel_size": 0.1,  # microns
            "bit_depth": 16,
        }

    @pytest.fixture
    def temp_calibration_dir(self):
        """Create temporary directory for calibration files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup handled by tempfile

    @pytest.mark.unit
    def test_class_initialization(self, calibration_functions):
        """Test class initialization."""
        assert calibration_functions is not None
        assert hasattr(calibration_functions, "calibrate_multicolour_camera")
        assert hasattr(calibration_functions, "calculate_rqe")
        assert hasattr(calibration_functions, "calculate_offset")
        assert hasattr(calibration_functions, "calculate_variance")
        assert hasattr(calibration_functions, "filesearch")

    @pytest.mark.unit
    def test_gain_map_generation(self, calibration_functions):
        """Test gain map generation."""
        height, width = 128, 128

        # Test uniform gain
        uniform_gain = calibration_functions.generate_gain_map(
            height, width, method="uniform", gain_value=1.0
        )

        assert uniform_gain.shape == (height, width)
        assert np.allclose(uniform_gain, 1.0)
        assert uniform_gain.dtype in [np.float32, np.float64]

    @pytest.mark.unit
    def test_gain_map_with_variation(self, calibration_functions):
        """Test gain map with spatial variation."""
        height, width = 64, 64

        # Test variable gain
        variable_gain = calibration_functions.generate_gain_map(
            height, width, method="variable", gain_mean=1.0, gain_std=0.1
        )

        assert variable_gain.shape == (height, width)
        assert variable_gain.mean() == pytest.approx(1.0, abs=0.2)
        assert variable_gain.std() > 0  # Should have variation
        assert np.all(variable_gain > 0)  # Gain should be positive

    @pytest.mark.unit
    def test_offset_map_generation(self, calibration_functions):
        """Test offset map generation."""
        height, width = 96, 96
        offset_value = 150.0

        offset_map = calibration_functions.generate_offset_map(
            height, width, offset_value=offset_value
        )

        assert offset_map.shape == (height, width)
        assert offset_map.dtype in [np.float32, np.float64]
        # Should be close to offset value (might have small variation)
        assert abs(offset_map.mean() - offset_value) < 10

    @pytest.mark.unit
    def test_variance_map_generation(self, calibration_functions):
        """Test variance map generation."""
        height, width = 80, 80

        variance_map = calibration_functions.generate_variance_map(
            height, width, readnoise=1.5, gain_map=np.ones((height, width))
        )

        assert variance_map.shape == (height, width)
        assert np.all(variance_map > 0)  # Variance should be positive
        assert variance_map.dtype in [np.float32, np.float64]

    @pytest.mark.integration
    def test_calibration_map_saving_loading(
        self, calibration_functions, sample_camera_params, temp_calibration_dir
    ):
        """Test saving and loading calibration maps."""
        # Save calibration maps
        gain_path = os.path.join(temp_calibration_dir, "gain.tif")
        offset_path = os.path.join(temp_calibration_dir, "offset.tif")
        variance_path = os.path.join(temp_calibration_dir, "variance.tif")

        calibration_functions.save_calibration_map(
            sample_camera_params["gain"], gain_path
        )
        calibration_functions.save_calibration_map(
            sample_camera_params["offset"], offset_path
        )
        calibration_functions.save_calibration_map(
            sample_camera_params["variance"], variance_path
        )

        # Verify files were created
        assert os.path.exists(gain_path)
        assert os.path.exists(offset_path)
        assert os.path.exists(variance_path)

        # Load calibration maps back
        loaded_gain = calibration_functions.load_calibration_map(gain_path)
        loaded_offset = calibration_functions.load_calibration_map(offset_path)
        loaded_variance = calibration_functions.load_calibration_map(variance_path)

        # Verify loaded maps match original
        np.testing.assert_array_almost_equal(
            loaded_gain, sample_camera_params["gain"], decimal=5
        )
        np.testing.assert_array_almost_equal(
            loaded_offset, sample_camera_params["offset"], decimal=5
        )
        np.testing.assert_array_almost_equal(
            loaded_variance, sample_camera_params["variance"], decimal=5
        )

    @pytest.mark.unit
    def test_calibration_validation(self, calibration_functions, sample_camera_params):
        """Test calibration parameter validation."""
        # Valid parameters should pass
        is_valid, errors = calibration_functions.validate_calibration(
            sample_camera_params
        )
        assert is_valid
        assert len(errors) == 0

        # Test with missing required parameters
        incomplete_params = sample_camera_params.copy()
        del incomplete_params["gain"]

        is_valid, errors = calibration_functions.validate_calibration(incomplete_params)
        assert not is_valid
        assert len(errors) > 0
        assert any("gain" in error.lower() for error in errors)

    @pytest.mark.unit
    def test_calibration_shape_consistency(self, calibration_functions):
        """Test that all calibration maps have consistent shapes."""
        height, width = 100, 120

        gain = calibration_functions.generate_gain_map(height, width)
        offset = calibration_functions.generate_offset_map(height, width)
        variance = calibration_functions.generate_variance_map(
            height, width, readnoise=1.0, gain_map=gain
        )

        # All maps should have same shape
        assert gain.shape == (height, width)
        assert offset.shape == (height, width)
        assert variance.shape == (height, width)

    @pytest.mark.unit
    def test_gain_map_statistical_properties(self, calibration_functions):
        """Test statistical properties of generated gain maps."""
        height, width = 200, 200
        target_mean = 0.95
        target_std = 0.05

        gain_map = calibration_functions.generate_gain_map(
            height, width, method="gaussian", gain_mean=target_mean, gain_std=target_std
        )

        # Check statistical properties
        actual_mean = gain_map.mean()
        actual_std = gain_map.std()

        assert abs(actual_mean - target_mean) < 0.02  # Within 2%
        assert abs(actual_std - target_std) < 0.01  # Within 1%

        # Should be positive definite
        assert np.all(gain_map > 0)

    @pytest.mark.unit
    def test_dark_current_correction(self, calibration_functions, sample_camera_params):
        """Test dark current correction."""
        # Create test image with dark current
        test_image = np.random.poisson(10, (64, 64)).astype(np.float32)
        dark_current = 2.0
        test_image += dark_current

        # Apply dark current correction
        corrected = calibration_functions.apply_dark_current_correction(
            test_image, dark_current
        )

        # Should remove dark current offset
        assert corrected.mean() < test_image.mean()
        assert corrected.mean() == pytest.approx(
            test_image.mean() - dark_current, abs=0.1
        )

    @pytest.mark.integration
    def test_complete_calibration_workflow(
        self, calibration_functions, temp_calibration_dir
    ):
        """Test complete camera calibration workflow."""
        height, width = 128, 128

        # Step 1: Generate calibration maps
        gain_map = calibration_functions.generate_gain_map(
            height, width, method="variable", gain_mean=1.0, gain_std=0.08
        )

        offset_map = calibration_functions.generate_offset_map(
            height, width, offset_value=120.0, variation=10.0
        )

        variance_map = calibration_functions.generate_variance_map(
            height, width, readnoise=1.3, gain_map=gain_map
        )

        # Step 2: Create calibration parameter set
        calibration_params = {
            "gain": gain_map,
            "offset": offset_map,
            "variance": variance_map,
            "readnoise": 1.3,
            "height": height,
            "width": width,
            "pixel_size": 0.11,
        }

        # Step 3: Validate calibration
        is_valid, errors = calibration_functions.validate_calibration(
            calibration_params
        )
        assert is_valid, f"Calibration validation failed: {errors}"

        # Step 4: Save calibration
        calibration_functions.save_complete_calibration(
            calibration_params, temp_calibration_dir
        )

        # Step 5: Load and verify
        loaded_params = calibration_functions.load_complete_calibration(
            temp_calibration_dir
        )

        # Verify loaded parameters match
        np.testing.assert_array_almost_equal(
            loaded_params["gain"], calibration_params["gain"]
        )
        np.testing.assert_array_almost_equal(
            loaded_params["offset"], calibration_params["offset"]
        )
        np.testing.assert_array_almost_equal(
            loaded_params["variance"], calibration_params["variance"]
        )
        assert loaded_params["readnoise"] == calibration_params["readnoise"]


class TestAdvancedCalibrationFeatures:
    """Test advanced calibration features."""

    @pytest.fixture
    def calibration_functions(self):
        """Create Calibration_Functions instance."""
        return Calibration_Functions()

    @pytest.mark.unit
    def test_linearity_correction(self, calibration_functions):
        """Test pixel linearity correction."""
        # Create test data with known nonlinearity
        input_levels = np.linspace(0, 1000, 100)
        nonlinear_response = (
            input_levels + 0.0001 * input_levels**2
        )  # Small nonlinearity

        # Apply linearity correction
        corrected = calibration_functions.apply_linearity_correction(
            nonlinear_response, method="polynomial", degree=2
        )

        # Corrected response should be more linear
        linear_fit_error = np.std(corrected - input_levels)
        original_fit_error = np.std(nonlinear_response - input_levels)

        assert linear_fit_error < original_fit_error

    @pytest.mark.unit
    def test_pixel_response_uniformity(self, calibration_functions):
        """Test pixel response uniformity correction."""
        height, width = 64, 64

        # Create non-uniform response (higher at center)
        y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        center_y, center_x = height // 2, width // 2
        distance_from_center = np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)

        # Vignetting effect - lower response at edges
        response_map = (
            1.0 - 0.3 * (distance_from_center / np.max(distance_from_center)) ** 2
        )

        # Apply uniformity correction
        correction_map = calibration_functions.calculate_uniformity_correction(
            response_map
        )

        assert correction_map.shape == (height, width)
        # Correction should be higher at edges where response is lower
        edge_correction = correction_map[0, 0]
        center_correction = correction_map[center_y, center_x]
        assert edge_correction > center_correction

    @pytest.mark.unit
    def test_temperature_drift_correction(self, calibration_functions):
        """Test temperature-dependent calibration drift correction."""
        base_temp = 20.0  # Celsius
        current_temp = 25.0
        temp_coefficient = 0.001  # 0.1% per degree

        # Original calibration at base temperature
        original_gain = np.ones((32, 32))

        # Apply temperature correction
        corrected_gain = calibration_functions.apply_temperature_correction(
            original_gain, base_temp, current_temp, temp_coefficient
        )

        # Should adjust for temperature difference
        expected_factor = 1 + temp_coefficient * (current_temp - base_temp)
        np.testing.assert_array_almost_equal(
            corrected_gain, original_gain * expected_factor, decimal=6
        )

    @pytest.mark.integration
    def test_multi_point_calibration(self, calibration_functions):
        """Test multi-point intensity calibration."""
        # Simulate calibration at different light levels
        reference_levels = [100, 500, 1000, 2000, 4000]  # ADU
        measured_levels = [
            98,
            502,
            1005,
            1995,
            4010,
        ]  # Measured ADU (with small errors)

        # Create calibration curve
        calibration_curve = calibration_functions.create_intensity_calibration(
            reference_levels, measured_levels
        )

        assert calibration_curve is not None

        # Test interpolation at intermediate points
        test_level = 750
        calibrated_level = calibration_functions.apply_intensity_calibration(
            test_level, calibration_curve
        )

        # Should give reasonable calibrated value
        assert 740 < calibrated_level < 760  # Allow some tolerance

    @pytest.mark.unit
    def test_bad_pixel_detection(self, calibration_functions):
        """Test bad pixel detection and masking."""
        height, width = 50, 50

        # Create gain map with some bad pixels
        gain_map = np.ones((height, width))

        # Add some obviously bad pixels
        gain_map[10, 15] = 0.1  # Very low gain (dead pixel)
        gain_map[20, 25] = 5.0  # Very high gain (hot pixel)
        gain_map[30, 35] = 0.0  # Zero gain (dead pixel)

        # Detect bad pixels
        bad_pixel_mask = calibration_functions.detect_bad_pixels(
            gain_map, low_threshold=0.5, high_threshold=2.0
        )

        assert bad_pixel_mask.shape == (height, width)
        assert bad_pixel_mask.dtype == bool

        # Should detect the bad pixels we added
        assert bad_pixel_mask[10, 15]  # Low gain pixel
        assert bad_pixel_mask[20, 25]  # High gain pixel
        assert bad_pixel_mask[30, 35]  # Zero gain pixel

        # Normal pixels should not be flagged
        assert not bad_pixel_mask[0, 0]
        assert not bad_pixel_mask[height // 2, width // 2]

    @pytest.mark.integration
    def test_calibration_quality_assessment(
        self, calibration_functions, sample_camera_params
    ):
        """Test calibration quality assessment metrics."""
        # Assess calibration quality
        quality_metrics = calibration_functions.assess_calibration_quality(
            sample_camera_params
        )

        assert isinstance(quality_metrics, dict)

        # Should have key quality metrics
        expected_metrics = ["uniformity", "noise_level", "linearity", "stability"]
        for metric in expected_metrics:
            if metric in quality_metrics:
                assert isinstance(quality_metrics[metric], (int, float))
                assert (
                    quality_metrics[metric] >= 0
                )  # Quality metrics should be non-negative

    @pytest.mark.unit
    def test_calibration_interpolation(self, calibration_functions):
        """Test spatial interpolation of calibration maps."""
        # Create low-resolution calibration map
        low_res_map = np.random.uniform(0.8, 1.2, (16, 16))

        # Interpolate to higher resolution
        high_res_map = calibration_functions.interpolate_calibration_map(
            low_res_map, target_shape=(64, 64), method="bilinear"
        )

        assert high_res_map.shape == (64, 64)

        # Should preserve approximate statistics
        assert abs(high_res_map.mean() - low_res_map.mean()) < 0.05

        # Should be smoothly varying
        gradients = np.gradient(high_res_map)
        max_gradient = max(np.max(np.abs(gradients[0])), np.max(np.abs(gradients[1])))
        assert max_gradient < 0.1  # Should not have sharp discontinuities


class TestCalibrationErrorHandling:
    """Test error handling and edge cases in calibration functions."""

    @pytest.fixture
    def calibration_functions(self):
        """Create Calibration_Functions instance."""
        return Calibration_Functions()

    @pytest.mark.unit
    def test_invalid_dimensions(self, calibration_functions):
        """Test handling of invalid dimensions."""
        # Zero dimensions should raise error
        with pytest.raises((ValueError, AssertionError)):
            calibration_functions.generate_gain_map(0, 100)

        with pytest.raises((ValueError, AssertionError)):
            calibration_functions.generate_gain_map(100, 0)

    @pytest.mark.unit
    def test_negative_parameters(self, calibration_functions):
        """Test handling of negative parameters."""
        # Negative readnoise should be handled gracefully
        with pytest.raises((ValueError, AssertionError)):
            calibration_functions.generate_variance_map(
                32, 32, readnoise=-1.0, gain_map=np.ones((32, 32))
            )

    @pytest.mark.unit
    def test_mismatched_array_shapes(self, calibration_functions):
        """Test handling of mismatched array shapes."""
        # Different shaped arrays should raise error
        gain_map_small = np.ones((10, 10))
        variance_map_large = np.ones((20, 20))

        with pytest.raises((ValueError, AssertionError)):
            calibration_functions.validate_calibration(
                {
                    "gain": gain_map_small,
                    "variance": variance_map_large,
                    "offset": np.ones((10, 10)),
                    "readnoise": 1.0,
                }
            )

    @pytest.mark.unit
    def test_file_io_errors(self, calibration_functions):
        """Test file I/O error handling."""
        # Nonexistent file should raise error
        with pytest.raises((FileNotFoundError, IOError, OSError)):
            calibration_functions.load_calibration_map("/nonexistent/path/file.tif")

        # Invalid file format should be handled
        with tempfile.NamedTemporaryFile(suffix=".txt") as temp_file:
            temp_file.write(b"not an image file")
            temp_file.flush()

            with pytest.raises((ValueError, IOError, OSError)):
                calibration_functions.load_calibration_map(temp_file.name)

    @pytest.mark.unit
    def test_extreme_parameter_values(self, calibration_functions):
        """Test handling of extreme parameter values."""
        # Very large arrays (should handle gracefully or fail predictably)
        try:
            large_gain = calibration_functions.generate_gain_map(10000, 10000)
            # If it succeeds, should have correct shape and reasonable values
            assert large_gain.shape == (10000, 10000)
            assert np.all(np.isfinite(large_gain))
        except MemoryError:
            # Expected for very large arrays
            pass

        # Very small variation (should not cause numerical issues)
        tiny_var_gain = calibration_functions.generate_gain_map(
            10, 10, method="gaussian", gain_mean=1.0, gain_std=1e-10
        )
        assert np.all(np.isfinite(tiny_var_gain))
        assert np.allclose(tiny_var_gain, 1.0, rtol=1e-8)
