#!/usr/bin/env python3
"""
Test module for ImageAnalysisFunctions.

Tests the refactored ImageAnalysisFunctions module with strategy pattern,
including fitting methods, parameter validation, and numerical accuracy.
"""

import pytest
import numpy as np
from pathlib import Path
import sys
from typing import List, Tuple, Optional, Dict, Any
import warnings

# Add src to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from ImageAnalysisFunctions import (
    Image_Analysis_Functions,
    FittingStrategy,
    FittingParameters,
    FittingConstants,
    FittingValidationError,
    FittingResultProcessor,
)


class TestFittingStrategy:
    """Test the FittingStrategy enum."""

    @pytest.mark.unit
    def test_strategy_enum_values(self):
        """Test that all strategy enum values are correct."""
        expected_strategies = {
            "STANDARD": "standard",
            "NOCOLOUR": "nocolour",
            "JUSTCOLOUR": "justcolour",
            "RAWCOLOUR": "rawcolour",
            "POSTHENCOLOUR": "posthencolour",
        }

        for attr_name, expected_value in expected_strategies.items():
            strategy = getattr(FittingStrategy, attr_name)
            assert strategy.value == expected_value

    @pytest.mark.unit
    def test_strategy_enum_completeness(self):
        """Test that enum contains expected number of strategies."""
        strategies = list(FittingStrategy)
        assert len(strategies) == 5

        strategy_values = {s.value for s in strategies}
        expected_values = {
            "standard",
            "nocolour",
            "justcolour",
            "rawcolour",
            "posthencolour",
        }
        assert strategy_values == expected_values


class TestFittingConstants:
    """Test the FittingConstants class."""

    @pytest.mark.unit
    def test_constant_values(self):
        """Test that constants have expected values."""
        assert FittingConstants.DEFAULT_FTOL == 1e-2
        assert FittingConstants.DEFAULT_XTOL == 1e-2
        assert FittingConstants.MAX_WORKERS == 60
        assert FittingConstants.WORKER_RATIO == 0.9
        assert FittingConstants.TASKS_PER_WORKER == 100

    @pytest.mark.unit
    def test_parameter_dimensions(self):
        """Test parameter dimensions for all strategies."""
        expected_dims = {
            FittingStrategy.STANDARD: {"fit": 12, "error": 10},
            FittingStrategy.NOCOLOUR: {"fit": 8, "error": 6},
            FittingStrategy.JUSTCOLOUR: {"fit": 10, "error": 8},
            FittingStrategy.RAWCOLOUR: {"fit": 10, "error": 8},
            FittingStrategy.POSTHENCOLOUR: {"fit": 10, "error": 8},
        }

        for strategy, expected in expected_dims.items():
            actual = FittingConstants.PARAM_DIMENSIONS[strategy]
            assert actual == expected


class TestFittingParameters:
    """Test the FittingParameters dataclass."""

    @pytest.fixture
    def sample_puncta_data(self):
        """Create sample puncta data for testing."""
        n_puncta = 3
        size = 11

        puncta = [np.random.randn(size, size) + 100 for _ in range(n_puncta)]
        smoothed_puncta = [np.random.randn(size, size) + 100 for _ in range(n_puncta)]
        weights = [np.ones((size, size)) for _ in range(n_puncta)]
        relative_coords = [[0.0, 0.0] for _ in range(n_puncta)]
        planes = [0] * n_puncta
        masks = [np.random.randn(3, size, size) for _ in range(n_puncta)]

        return {
            "puncta": puncta,
            "smoothed_puncta": smoothed_puncta,
            "weights": weights,
            "relative_coords": relative_coords,
            "planes": planes,
            "masks": masks,
        }

    @pytest.mark.unit
    def test_valid_parameters(self, sample_puncta_data):
        """Test that valid parameters are accepted."""
        # Test with masks (for colour strategy)
        params = FittingParameters(
            puncta=sample_puncta_data["puncta"],
            smoothed_puncta=sample_puncta_data["smoothed_puncta"],
            weights=sample_puncta_data["weights"],
            relative_coords=sample_puncta_data["relative_coords"],
            planes=sample_puncta_data["planes"],
            strategy=FittingStrategy.STANDARD,
            masks=sample_puncta_data["masks"],
        )
        # Should not raise exception
        assert params.strategy == FittingStrategy.STANDARD

        # Test without masks (for no-colour strategy)
        params_no_colour = FittingParameters(
            puncta=sample_puncta_data["puncta"],
            smoothed_puncta=sample_puncta_data["smoothed_puncta"],
            weights=sample_puncta_data["weights"],
            relative_coords=sample_puncta_data["relative_coords"],
            planes=sample_puncta_data["planes"],
            strategy=FittingStrategy.NOCOLOUR,
        )
        assert params_no_colour.strategy == FittingStrategy.NOCOLOUR

    @pytest.mark.unit
    def test_inconsistent_array_lengths(self, sample_puncta_data):
        """Test that inconsistent array lengths raise validation error."""
        # Make one array shorter
        sample_puncta_data["smoothed_puncta"] = sample_puncta_data["smoothed_puncta"][
            :-1
        ]

        with pytest.raises(FittingValidationError, match="same length"):
            FittingParameters(
                puncta=sample_puncta_data["puncta"],
                smoothed_puncta=sample_puncta_data["smoothed_puncta"],
                weights=sample_puncta_data["weights"],
                relative_coords=sample_puncta_data["relative_coords"],
                planes=sample_puncta_data["planes"],
                strategy=FittingStrategy.STANDARD,
                masks=sample_puncta_data["masks"],
            )

    @pytest.mark.unit
    def test_missing_masks_for_colour_strategy(self, sample_puncta_data):
        """Test that missing masks for colour strategy raises error."""
        with pytest.raises(FittingValidationError, match="requires masks"):
            FittingParameters(
                puncta=sample_puncta_data["puncta"],
                smoothed_puncta=sample_puncta_data["smoothed_puncta"],
                weights=sample_puncta_data["weights"],
                relative_coords=sample_puncta_data["relative_coords"],
                planes=sample_puncta_data["planes"],
                strategy=FittingStrategy.STANDARD,
                # Missing masks parameter
            )

    @pytest.mark.unit
    def test_wrong_masks_length(self, sample_puncta_data):
        """Test that wrong masks length raises error."""
        sample_puncta_data["masks"] = sample_puncta_data["masks"][:-1]  # Make shorter

        with pytest.raises(FittingValidationError, match="same length"):
            FittingParameters(
                puncta=sample_puncta_data["puncta"],
                smoothed_puncta=sample_puncta_data["smoothed_puncta"],
                weights=sample_puncta_data["weights"],
                relative_coords=sample_puncta_data["relative_coords"],
                planes=sample_puncta_data["planes"],
                strategy=FittingStrategy.STANDARD,
                masks=sample_puncta_data["masks"],
            )


class TestFittingResultProcessor:
    """Test the FittingResultProcessor class."""

    @pytest.mark.unit
    def test_calculate_errors_valid_covariance(self):
        """Test error calculation with valid covariance matrix."""
        # Create a simple 3x3 covariance matrix
        pcov = np.array([[1.0, 0.1, 0.0], [0.1, 2.0, 0.0], [0.0, 0.0, 0.5]])

        errors = FittingResultProcessor.calculate_errors(pcov, FittingStrategy.NOCOLOUR)

        expected_errors = [1.0, np.sqrt(2.0), np.sqrt(0.5)]
        np.testing.assert_allclose(errors[:3], expected_errors, rtol=1e-10)

        # Should pad to expected size for NOCOLOUR (6 errors)
        assert len(errors) == 6
        assert all(np.isnan(errors[3:]))

    @pytest.mark.unit
    def test_calculate_errors_none_covariance(self):
        """Test error calculation with None covariance."""
        errors = FittingResultProcessor.calculate_errors(None, FittingStrategy.STANDARD)

        expected_size = FittingConstants.PARAM_DIMENSIONS[FittingStrategy.STANDARD][
            "error"
        ]
        assert len(errors) == expected_size
        assert all(np.isnan(errors))

    @pytest.mark.unit
    def test_calculate_errors_infinite_covariance(self):
        """Test error calculation with infinite covariance."""
        pcov = np.full((3, 3), np.inf)

        errors = FittingResultProcessor.calculate_errors(pcov, FittingStrategy.NOCOLOUR)

        expected_size = FittingConstants.PARAM_DIMENSIONS[FittingStrategy.NOCOLOUR][
            "error"
        ]
        assert len(errors) == expected_size
        assert all(np.isnan(errors))

    @pytest.mark.unit
    def test_calculate_errors_negative_diagonal(self):
        """Test error calculation with negative diagonal elements."""
        pcov = np.array([[-1.0, 0.0], [0.0, 2.0]])

        errors = FittingResultProcessor.calculate_errors(pcov, FittingStrategy.NOCOLOUR)

        # Negative diagonal should give 0.0 error
        assert errors[0] == 0.0
        assert errors[1] == np.sqrt(2.0)

    @pytest.mark.unit
    def test_process_fit_results_standard_strategy(self):
        """Test processing of fit results for STANDARD strategy."""
        # Mock fitting parameters: [x, y, sy, sx, bg_B, bg_G, bg_R, A_B, A_G, A_R]
        pfit = np.array([5.5, 6.2, 1.2, 1.1, 2.0, 3.0, 2.5, 10.0, 15.0, 12.0])
        pcov = np.eye(len(pfit)) * 0.1  # Simple diagonal covariance
        relative_coords = [10.0, 20.0]
        size = 11
        chisqr = 1.23

        pfit_final, errors = FittingResultProcessor.process_fit_results(
            pfit, pcov, size, relative_coords, FittingStrategy.STANDARD, chisqr
        )

        # Check that coordinates were adjusted
        assert pfit_final[0] == 5.5 + 10.0  # x coordinate
        assert pfit_final[1] == 6.2 + 20.0  # y coordinate

        # Check that sx/sy were swapped in output: [x, y, sx, sy, ...]
        assert pfit_final[2] == 1.1  # sx (was pfit[3])
        assert pfit_final[3] == 1.2  # sy (was pfit[2])

        # Check that backgrounds and amplitudes were squared
        np.testing.assert_allclose(
            pfit_final[4:7], [4.0, 9.0, 6.25]
        )  # squared backgrounds
        np.testing.assert_allclose(
            pfit_final[7:10], [100.0, 225.0, 144.0]
        )  # squared amplitudes

        # Check chi-squared was appended
        assert pfit_final[-1] == chisqr

        # Check errors were calculated
        assert (
            len(errors)
            == FittingConstants.PARAM_DIMENSIONS[FittingStrategy.STANDARD]["error"]
        )

    @pytest.mark.unit
    def test_process_fit_results_invalid_position(self):
        """Test processing when position is outside image bounds."""
        pfit = np.array([-1.0, 5.0, 1.0, 1.0, 2.0, 3.0])  # x position is negative
        pcov = np.eye(len(pfit)) * 0.1
        relative_coords = [0.0, 0.0]
        size = 10

        pfit_final, errors = FittingResultProcessor.process_fit_results(
            pfit, pcov, size, relative_coords, FittingStrategy.NOCOLOUR
        )

        # Should return NaN arrays for invalid positions
        assert np.all(np.isnan(pfit_final))
        assert np.all(np.isnan(errors))

    @pytest.mark.unit
    def test_process_fit_results_empty_input(self):
        """Test processing with empty/None input."""
        pfit_final, errors = FittingResultProcessor.process_fit_results(
            None, None, 10, [0, 0], FittingStrategy.STANDARD
        )

        dims = FittingConstants.PARAM_DIMENSIONS[FittingStrategy.STANDARD]
        assert len(pfit_final) == dims["fit"]
        assert len(errors) == dims["error"]
        assert np.all(np.isnan(pfit_final))
        assert np.all(np.isnan(errors))


class TestSyntheticDataGeneration:
    """Test synthetic data generation for fitting tests."""

    def generate_synthetic_gaussian_punctum(
        self,
        size: int = 11,
        amplitude: float = 1000.0,
        background: float = 100.0,
        x_center: float = 5.0,
        y_center: float = 5.0,
        sigma: float = 1.2,
        noise_level: float = 10.0,
        random_seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Generate a synthetic Gaussian punctum for testing.

        Args:
            size: Size of the image (size x size).
            amplitude: Peak amplitude of the Gaussian.
            background: Background intensity level.
            x_center: X position of the Gaussian center.
            y_center: Y position of the Gaussian center.
            sigma: Standard deviation of the Gaussian.
            noise_level: Standard deviation of noise to add.
            random_seed: Random seed for reproducible noise.

        Returns:
            Tuple of (synthetic_image, ground_truth_parameters).
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:size, 0:size]

        # Generate Gaussian profile
        gaussian = amplitude * np.exp(
            -((x_coords - x_center) ** 2 + (y_coords - y_center) ** 2) / (2 * sigma**2)
        )

        # Add background and noise
        image = gaussian + background + np.random.normal(0, noise_level, (size, size))

        # Ensure non-negative values
        image = np.maximum(image, 0)

        ground_truth = {
            "x_center": x_center,
            "y_center": y_center,
            "amplitude": amplitude,
            "background": background,
            "sigma": sigma,
        }

        return image.astype(np.float32), ground_truth

    def generate_synthetic_colour_punctum(
        self,
        size: int = 11,
        amplitudes: Tuple[float, float, float] = (800.0, 1200.0, 1000.0),
        backgrounds: Tuple[float, float, float] = (80.0, 120.0, 100.0),
        x_center: float = 5.0,
        y_center: float = 5.0,
        sigma: float = 1.2,
        noise_level: float = 10.0,
        random_seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Generate a synthetic colour punctum with masks.

        Args:
            size: Size of the image (size x size).
            amplitudes: Peak amplitudes for (Blue, Green, Red) channels.
            backgrounds: Background levels for (Blue, Green, Red) channels.
            x_center: X position of the Gaussian center.
            y_center: Y position of the Gaussian center.
            sigma: Standard deviation of the Gaussian.
            noise_level: Standard deviation of noise to add.
            random_seed: Random seed for reproducible noise.

        Returns:
            Tuple of (synthetic_image, colour_masks, ground_truth_parameters).
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:size, 0:size]

        # Generate Gaussian profile
        gaussian_profile = np.exp(
            -((x_coords - x_center) ** 2 + (y_coords - y_center) ** 2) / (2 * sigma**2)
        )

        # Create colour channels
        image = np.zeros((size, size))
        masks = np.zeros((3, size, size))

        for i, (amp, bg) in enumerate(zip(amplitudes, backgrounds)):
            channel = amp * gaussian_profile + bg
            channel += np.random.normal(0, noise_level, (size, size))
            channel = np.maximum(channel, 0)

            image += channel
            masks[i] = np.ones((size, size))  # Simple uniform mask

        ground_truth = {
            "x_center": x_center,
            "y_center": y_center,
            "amplitudes": amplitudes,
            "backgrounds": backgrounds,
            "sigma": sigma,
        }

        return image.astype(np.float32), masks.astype(np.float32), ground_truth

    @pytest.mark.unit
    def test_synthetic_gaussian_generation(self):
        """Test synthetic Gaussian punctum generation."""
        image, ground_truth = self.generate_synthetic_gaussian_punctum(
            size=11,
            amplitude=1000.0,
            background=100.0,
            x_center=5.0,
            y_center=5.0,
            sigma=1.2,
            random_seed=42,
        )

        # Check image properties
        assert image.shape == (11, 11)
        assert image.dtype == np.float32
        assert np.all(image >= 0)

        # Check that peak is approximately at expected location
        peak_y, peak_x = np.unravel_index(np.argmax(image), image.shape)
        assert abs(peak_x - 5.0) <= 1.0
        assert abs(peak_y - 5.0) <= 1.0

        # Check ground truth
        expected_keys = {"x_center", "y_center", "amplitude", "background", "sigma"}
        assert set(ground_truth.keys()) == expected_keys

    @pytest.mark.unit
    def test_synthetic_colour_generation(self):
        """Test synthetic colour punctum generation."""
        image, masks, ground_truth = self.generate_synthetic_colour_punctum(
            size=11, random_seed=42
        )

        # Check image properties
        assert image.shape == (11, 11)
        assert image.dtype == np.float32
        assert np.all(image >= 0)

        # Check masks
        assert masks.shape == (3, 11, 11)
        assert masks.dtype == np.float32
        assert np.all(masks >= 0)

        # Check ground truth
        expected_keys = {"x_center", "y_center", "amplitudes", "backgrounds", "sigma"}
        assert set(ground_truth.keys()) == expected_keys
        assert len(ground_truth["amplitudes"]) == 3
        assert len(ground_truth["backgrounds"]) == 3


class TestImageAnalysisFunctionsBasic:
    """Basic tests for the ImageAnalysisFunctions class."""

    @pytest.fixture
    def analysis_functions(self):
        """Create an instance of Image_Analysis_Functions."""
        return Image_Analysis_Functions()

    @pytest.mark.unit
    def test_class_initialization(self, analysis_functions):
        """Test that the class initializes properly."""
        assert analysis_functions is not None
        # Check that required methods exist
        assert hasattr(analysis_functions, "fit_puncta_method")
        assert hasattr(analysis_functions, "fit_puncta_parallel_method")

    @pytest.mark.unit
    def test_available_strategies(self):
        """Test that all fitting strategies are available."""
        strategies = list(FittingStrategy)
        assert len(strategies) == 5

        strategy_names = [s.value for s in strategies]
        expected_names = [
            "standard",
            "nocolour",
            "justcolour",
            "rawcolour",
            "posthencolour",
        ]
        assert set(strategy_names) == set(expected_names)


if __name__ == "__main__":
    # Run tests directly when script is executed
    pytest.main([__file__, "-v"])
