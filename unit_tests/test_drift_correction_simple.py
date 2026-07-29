"""
Simple integration tests for drift correction based on actual user code.

This tests the exact API that users interact with, ensuring the refactoring
didn't break the primary use cases.
"""

import sys
import os

import pytest
import numpy as np
import pandas as pd
import pyS3M.DriftCorrectionFunctions as DCF


def generate_test_data(n_locs=1000, n_frames=100, width=256, height=256):
    """Generate synthetic localization data matching user's format."""
    frames = np.random.randint(0, n_frames, n_locs)
    x = np.random.uniform(0, width, n_locs)
    y = np.random.uniform(0, height, n_locs)
    photons = np.random.uniform(1000, 5000, n_locs)

    df = pd.DataFrame({
        'x': x,
        'y': y,
        'xc': x + width/2,  # Centered coordinates
        'yc': y + height/2,
        'photons': photons,
        'frame': frames
    })

    return df


class TestDriftCorrectionUserAPI:
    """Test the actual user-facing API for drift correction."""

    def test_drift_corrector_initialization(self):
        """Test that DriftCorrectionFunctions can be initialized (the bug we fixed)."""
        # This was failing with: TypeError: CoordinateProcessor() takes no arguments
        drift_corrector = DCF.Drift_Correction_Functions()

        assert drift_corrector is not None
        assert hasattr(drift_corrector, 'undrift')
        # Verify submodules are initialized
        assert drift_corrector.coordinate_processor is not None
        assert drift_corrector.aim_corrector is not None
        assert drift_corrector.fiducial_detector is not None

    def test_undrift_with_aim_method(self):
        """Test undrift with AIM method using user's exact code pattern."""
        # Generate test data
        loc_data = generate_test_data(n_locs=2000, n_frames=50, width=256, height=256)
        width = 256
        height = 256

        # User's exact code pattern
        drift_corrector = DCF.Drift_Correction_Functions()

        info = [{
            "Width": width,
            "Height": height,
            "Frames": np.max(loc_data['frame']),
            "Pixelsize": 69,
        }]

        # This should work without errors
        corrected_locs, drift_result = drift_corrector.undrift(
            locs=loc_data.to_records(index=False),
            info=info,
            method="aim",
            segmentation=20,
            intersect_d=20/69,
            roi_r=60/69,
        )

        # Convert back to DataFrame
        corrected_locs_df = pd.DataFrame(corrected_locs)

        # Basic checks
        assert len(corrected_locs_df) > 0, "Should have corrected localizations"
        assert 'x' in corrected_locs_df.columns, "Should have x coordinates"
        assert 'y' in corrected_locs_df.columns, "Should have y coordinates"
        assert hasattr(drift_result, 'drift_x'), "Should have drift_x in result"
        assert hasattr(drift_result, 'drift_y'), "Should have drift_y in result"
        assert len(drift_result.drift_x) > 0, "drift_x should not be empty"
        assert len(drift_result.drift_y) > 0, "drift_y should not be empty"

    def test_undrift_auto_method(self):
        """Test automatic method selection."""
        loc_data = generate_test_data(n_locs=1000, n_frames=30)

        drift_corrector = DCF.Drift_Correction_Functions()

        info = [{
            "Width": 256,
            "Height": 256,
            "Frames": np.max(loc_data['frame']),
            "Pixelsize": 69,
        }]

        # Should automatically choose a suitable method
        corrected_locs, drift_result = drift_corrector.undrift(
            locs=loc_data.to_records(index=False),
            info=info,
            method="auto"
        )

        corrected_locs_df = pd.DataFrame(corrected_locs)
        assert len(corrected_locs_df) > 0

    def test_drift_result_structure(self):
        """Test that drift result has expected structure."""
        loc_data = generate_test_data(n_locs=1000, n_frames=30)

        drift_corrector = DCF.Drift_Correction_Functions()

        info = [{
            "Width": 256,
            "Height": 256,
            "Frames": np.max(loc_data['frame']),
            "Pixelsize": 69,
        }]

        corrected_locs, drift_result = drift_corrector.undrift(
            locs=loc_data.to_records(index=False),
            info=info,
            method="aim"
        )

        # Check drift result is an object with drift_x and drift_y attributes
        assert hasattr(drift_result, 'drift_x'), "drift_result should have drift_x"
        assert hasattr(drift_result, 'drift_y'), "drift_result should have drift_y"

        # Check drift arrays have reasonable length (one value per frame).
        # Use >= because drift length may be n_frames or n_frames+1 depending on 0/1-indexing.
        n_frames = info[0]['Frames']
        assert len(drift_result.drift_x) >= n_frames
        assert len(drift_result.drift_y) >= n_frames


class TestModuleAccessibility:
    """Test that all refactored modules are accessible."""

    def test_can_import_modules(self):
        """Test that all drift correction modules can be imported."""
        # These imports should all succeed after refactoring
        from pyS3M.CoordinateProcessing import CoordinateProcessor
        from pyS3M.drift_correction.aim import AIMDriftCorrector
        from pyS3M.FiducialDetection import FiducialDetector

        assert CoordinateProcessor is not None
        assert AIMDriftCorrector is not None
        assert FiducialDetector is not None

    def test_submodules_initialized(self):
        """Test that DriftCorrectionFunctions properly initializes all submodules."""
        drift_corr = DCF.Drift_Correction_Functions()

        # Check all submodules exist
        assert hasattr(drift_corr, 'coordinate_processor')
        assert hasattr(drift_corr, 'aim_corrector')
        assert hasattr(drift_corr, 'fiducial_detector')

        # Check they're not None
        assert drift_corr.coordinate_processor is not None
        assert drift_corr.aim_corrector is not None
        assert drift_corr.fiducial_detector is not None


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v', '--tb=short'])
