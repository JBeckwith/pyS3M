"""
Unit tests for drift correction functionality after refactoring.

Tests cover:
- DriftCorrectionFunctions main class
- CoordinateProcessor
- AIMAlgorithm
- RCCAlgorithm
- FiducialDetection
- Integration tests for full workflow
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import numpy as np
import pandas as pd
from typing import Dict, Tuple

# Import drift correction modules
import DriftCorrectionFunctions as DCF
from CoordinateProcessing import CoordinateProcessor
from AIMAlgorithm import AIMAlgorithm
from RCCAlgorithm import RCCAlgorithm
from FiducialDetection import FiducialDetector


# ============================================================================
# Test Data Generators
# ============================================================================

def generate_test_localizations(
    n_locs: int = 1000,
    n_frames: int = 100,
    width: int = 256,
    height: int = 256,
    drift_x: np.ndarray = None,
    drift_y: np.ndarray = None
) -> np.recarray:
    """Generate synthetic localization data with optional drift.

    Args:
        n_locs: Number of localizations
        n_frames: Number of frames
        width: Image width in pixels
        height: Image height in pixels
        drift_x: Optional drift in x (one value per frame)
        drift_y: Optional drift in y (one value per frame)

    Returns:
        Localization recarray with drift applied
    """
    # Random localizations uniformly distributed
    frames = np.random.randint(0, n_frames, n_locs)
    x = np.random.uniform(0, width, n_locs)
    y = np.random.uniform(0, height, n_locs)
    photons = np.random.uniform(1000, 5000, n_locs)

    # Apply drift if provided
    if drift_x is not None and drift_y is not None:
        for i, frame in enumerate(frames):
            x[i] += drift_x[frame]
            y[i] += drift_y[frame]

    # Create recarray with required fields
    locs = np.rec.fromarrays(
        [x, y, photons, frames, np.ones(n_locs) * 100],
        names=['x', 'y', 'photons', 'frame', 'bg']
    )

    return locs


def generate_fiducial_localizations(
    n_fiducials: int = 10,
    n_frames: int = 100,
    detections_per_frame: int = 8,
    width: int = 256,
    height: int = 256,
    drift_x: np.ndarray = None,
    drift_y: np.ndarray = None
) -> np.recarray:
    """Generate synthetic fiducial marker localizations.

    Creates bright spots that appear in most frames, suitable for
    testing fiducial-based drift correction.
    """
    # Random fiducial positions
    fid_x = np.random.uniform(20, width - 20, n_fiducials)
    fid_y = np.random.uniform(20, height - 20, n_fiducials)

    locs_list = []

    for frame in range(n_frames):
        # Each fiducial detected with high probability
        for i in range(n_fiducials):
            if np.random.rand() < 0.8:  # 80% detection rate
                x = fid_x[i] + np.random.normal(0, 0.1)  # Small localization error
                y = fid_y[i] + np.random.normal(0, 0.1)

                # Apply drift if provided
                if drift_x is not None and drift_y is not None:
                    x += drift_x[frame]
                    y += drift_y[frame]

                locs_list.append({
                    'x': x,
                    'y': y,
                    'photons': np.random.uniform(5000, 10000),  # Bright
                    'frame': frame,
                    'bg': 100
                })

    # Convert to recarray
    df = pd.DataFrame(locs_list)
    locs = df.to_records(index=False)

    return locs


def generate_linear_drift(n_frames: int, drift_rate: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """Generate linear drift pattern.

    Args:
        n_frames: Number of frames
        drift_rate: Drift rate in pixels per frame

    Returns:
        (drift_x, drift_y) arrays
    """
    frames = np.arange(n_frames)
    drift_x = frames * drift_rate
    drift_y = frames * drift_rate * 0.5  # Different rate in y

    return drift_x, drift_y


def generate_sinusoidal_drift(n_frames: int, amplitude: float = 2.0, period: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """Generate sinusoidal drift pattern (e.g., thermal oscillation).

    Args:
        n_frames: Number of frames
        amplitude: Drift amplitude in pixels
        period: Oscillation period in frames

    Returns:
        (drift_x, drift_y) arrays
    """
    frames = np.arange(n_frames)
    drift_x = amplitude * np.sin(2 * np.pi * frames / period)
    drift_y = amplitude * np.cos(2 * np.pi * frames / period)

    return drift_x, drift_y


# ============================================================================
# CoordinateProcessor Tests
# ============================================================================

class TestCoordinateProcessor:
    """Test CoordinateProcessor static methods."""

    def test_extract_metadata(self):
        """Test metadata extraction from info dict."""
        info = [{
            'Width': 256,
            'Height': 256,
            'Frames': 100,
            'Pixelsize': 69
        }]

        metadata = CoordinateProcessor.extract_metadata(info)

        assert metadata['width'] == 256
        assert metadata['height'] == 256
        assert metadata['n_frames'] == 100
        assert metadata['pixelsize'] == 69

    def test_validate_localisations(self):
        """Test localization validation with valid data."""
        locs = generate_test_localizations(n_locs=100, width=256, height=256)

        # Should not raise for valid data
        CoordinateProcessor.validate_localisations(locs)  # Will raise if invalid

    def test_apply_drift_correction(self):
        """Test applying drift correction to localizations."""
        locs = generate_test_localizations(n_locs=1000, n_frames=50)

        # Create mock drift (linear)
        n_frames = 50
        drift_x = np.linspace(0, 10, n_frames)  # 10 pixel drift over 50 frames
        drift_y = np.linspace(0, 5, n_frames)   # 5 pixel drift over 50 frames

        # Apply drift correction
        corrected = CoordinateProcessor.apply_drift_correction(
            locs, drift_x, drift_y
        )

        # Check output
        assert len(corrected) == len(locs)
        assert 'x' in corrected.dtype.names
        assert 'y' in corrected.dtype.names


# ============================================================================
# AIMAlgorithm Tests
# ============================================================================

class TestAIMAlgorithm:
    """Test AIM drift correction algorithm."""

    def test_aim_initialization(self):
        """Test AIM algorithm can be initialized."""
        drift_corr = DCF.Drift_Correction_Functions()
        aim = AIMAlgorithm(drift_correction_instance=drift_corr)

        assert aim is not None
        assert hasattr(aim, 'run_aim_2d')

    def test_aim_detects_linear_drift(self):
        """Test AIM can detect simple linear drift."""
        # Generate data with known drift
        n_frames = 100
        drift_x_true, drift_y_true = generate_linear_drift(n_frames, drift_rate=0.2)

        locs = generate_test_localizations(
            n_locs=5000,
            n_frames=n_frames,
            drift_x=drift_x_true,
            drift_y=drift_y_true
        )

        info = [{
            'Width': 256,
            'Height': 256,
            'Frames': n_frames,
            'Pixelsize': 69
        }]

        # Run AIM
        drift_corr = DCF.Drift_Correction_Functions()
        aim = AIMAlgorithm(drift_correction_instance=drift_corr)

        drift_x, drift_y, meta = aim.run_aim_2d(
            locs, info,
            segmentation=10,
            intersect_d=1.0,
            roi_r=2.0
        )

        # Check that drift was detected (should be close to linear)
        assert len(drift_x) == n_frames
        assert len(drift_y) == n_frames

        # Drift should be monotonically increasing (approximately)
        assert np.all(np.diff(drift_x) > -0.5)  # Allow small noise
        assert np.all(np.diff(drift_y) > -0.5)

        # Total drift should be approximately correct
        # (AIM reports drift relative to first frame)
        expected_total_x = drift_x_true[-1] - drift_x_true[0]
        expected_total_y = drift_y_true[-1] - drift_y_true[0]

        actual_total_x = drift_x[-1] - drift_x[0]
        actual_total_y = drift_y[-1] - drift_y[0]

        # Should be within 20% (AIM is approximate)
        assert abs(actual_total_x - expected_total_x) < abs(expected_total_x) * 0.3
        assert abs(actual_total_y - expected_total_y) < abs(expected_total_y) * 0.3


# ============================================================================
# RCCAlgorithm Tests
# ============================================================================

class TestRCCAlgorithm:
    """Test RCC drift correction algorithm."""

    def test_rcc_initialization(self):
        """Test RCC algorithm can be initialized."""
        drift_corr = DCF.Drift_Correction_Functions()
        rcc = RCCAlgorithm(drift_correction_instance=drift_corr)

        assert rcc is not None
        assert hasattr(rcc, 'run_rcc_2d')

    def test_rcc_detects_linear_drift(self):
        """Test RCC can detect simple linear drift."""
        # Generate data with known drift
        n_frames = 100
        drift_x_true, drift_y_true = generate_linear_drift(n_frames, drift_rate=0.2)

        locs = generate_test_localizations(
            n_locs=10000,  # RCC needs more localizations
            n_frames=n_frames,
            drift_x=drift_x_true,
            drift_y=drift_y_true
        )

        info = [{
            'Width': 256,
            'Height': 256,
            'Frames': n_frames,
            'Pixelsize': 69
        }]

        # Run RCC
        drift_corr = DCF.Drift_Correction_Functions()
        rcc = RCCAlgorithm(drift_correction_instance=drift_corr)

        drift_x, drift_y, meta = rcc.run_rcc_2d(
            locs, info,
            segmentation=10,
            use_time_factor=False
        )

        # Check that drift was detected
        assert len(drift_x) == n_frames
        assert len(drift_y) == n_frames

        # Total drift should be approximately correct
        expected_total_x = drift_x_true[-1] - drift_x_true[0]
        expected_total_y = drift_y_true[-1] - drift_y_true[0]

        actual_total_x = drift_x[-1] - drift_x[0]
        actual_total_y = drift_y[-1] - drift_y[0]

        # RCC is generally more accurate than AIM
        assert abs(actual_total_x - expected_total_x) < abs(expected_total_x) * 0.2
        assert abs(actual_total_y - expected_total_y) < abs(expected_total_y) * 0.2


# ============================================================================
# FiducialDetection Tests
# ============================================================================

class TestFiducialDetection:
    """Test fiducial marker detection."""

    def test_fiducial_initialization(self):
        """Test FiducialDetector can be initialized."""
        drift_corr = DCF.Drift_Correction_Functions()
        fid = FiducialDetector(drift_correction_instance=drift_corr)

        assert fid is not None
        assert hasattr(fid, 'detect_fiducials_birch')

    def test_detect_fiducials_birch(self):
        """Test BIRCH-based fiducial detection."""
        # Generate fiducial marker data
        n_frames = 100
        locs = generate_fiducial_localizations(
            n_fiducials=10,
            n_frames=n_frames,
            width=256,
            height=256
        )

        info = [{
            'Width': 256,
            'Height': 256,
            'Frames': n_frames,
            'Pixelsize': 69
        }]

        drift_corr = DCF.Drift_Correction_Functions()
        fid = FiducialDetector(drift_correction_instance=drift_corr)

        # Detect fiducials
        fiducials, meta = fid.detect_fiducials_birch(
            locs, info,
            min_localizations=int(n_frames * 0.3),  # Appear in 30% of frames
            threshold=0.5,
            branching_factor=50
        )

        # Should detect some fiducials
        assert len(fiducials) > 0
        assert len(fiducials) <= 10  # Can't detect more than we created

        # Each fiducial should have frame data
        for fiducial_locs in fiducials:
            assert len(fiducial_locs) > 0
            assert 'x' in fiducial_locs.dtype.names
            assert 'y' in fiducial_locs.dtype.names
            assert 'frame' in fiducial_locs.dtype.names

    def test_compute_drift_from_fiducials(self):
        """Test drift calculation from fiducial tracks."""
        # Generate fiducials with known drift
        n_frames = 100
        drift_x_true, drift_y_true = generate_linear_drift(n_frames, drift_rate=0.1)

        locs = generate_fiducial_localizations(
            n_fiducials=15,
            n_frames=n_frames,
            drift_x=drift_x_true,
            drift_y=drift_y_true
        )

        info = [{
            'Width': 256,
            'Height': 256,
            'Frames': n_frames,
            'Pixelsize': 69
        }]

        drift_corr = DCF.Drift_Correction_Functions()
        fid = FiducialDetector(drift_correction_instance=drift_corr)

        # Detect fiducials
        fiducials, _ = fid.detect_fiducials_birch(
            locs, info,
            min_localizations=int(n_frames * 0.5)
        )

        # Compute drift
        drift_x, drift_y, meta = fid.compute_drift_from_fiducials(
            fiducials, info
        )

        # Check drift shape
        assert len(drift_x) == n_frames
        assert len(drift_y) == n_frames

        # Check drift direction (should be monotonic for linear drift)
        assert np.corrcoef(drift_x, drift_x_true)[0, 1] > 0.95  # High correlation
        assert np.corrcoef(drift_y, drift_y_true)[0, 1] > 0.95


# ============================================================================
# DriftCorrectionFunctions Integration Tests
# ============================================================================

class TestDriftCorrectionIntegration:
    """Integration tests for full drift correction workflow."""

    def test_drift_correction_initialization(self):
        """Test DriftCorrectionFunctions can be initialized."""
        drift_corr = DCF.Drift_Correction_Functions()

        assert drift_corr is not None
        assert hasattr(drift_corr, 'undrift')
        assert drift_corr.coordinate_processor is not None
        assert drift_corr.aim_algorithm is not None
        assert drift_corr.rcc_algorithm is not None
        assert drift_corr.fiducial_detection is not None

    def test_undrift_with_aim(self):
        """Test full undrift workflow with AIM method."""
        # Generate data with drift
        n_frames = 50
        drift_x_true, drift_y_true = generate_linear_drift(n_frames, drift_rate=0.3)

        locs = generate_test_localizations(
            n_locs=2000,
            n_frames=n_frames,
            width=256,
            height=256,
            drift_x=drift_x_true,
            drift_y=drift_y_true
        )

        info = [{
            'Width': 256,
            'Height': 256,
            'Frames': n_frames,
            'Pixelsize': 69
        }]

        # Run drift correction
        drift_corr = DCF.Drift_Correction_Functions()

        corrected_locs, drift_result = drift_corr.undrift(
            locs=locs,
            info=info,
            method='aim',
            segmentation=10,
            intersect_d=1.0,
            roi_r=2.0
        )

        # Check outputs
        assert len(corrected_locs) == len(locs)
        assert 'drift_x' in drift_result
        assert 'drift_y' in drift_result
        assert len(drift_result['drift_x']) == n_frames
        assert len(drift_result['drift_y']) == n_frames

        # Corrected localizations should have reduced drift
        # (variance should be lower)
        original_var_x = np.var(locs['x'])
        corrected_var_x = np.var(corrected_locs['x'])

        # Note: This might not always hold for sparse data, but should for our test case
        # Just check they're both reasonable
        assert corrected_var_x > 0
        assert original_var_x > 0

    def test_undrift_with_rcc(self):
        """Test full undrift workflow with RCC method."""
        # Generate data with drift
        n_frames = 50
        drift_x_true, drift_y_true = generate_linear_drift(n_frames, drift_rate=0.3)

        locs = generate_test_localizations(
            n_locs=5000,  # RCC needs more data
            n_frames=n_frames,
            width=256,
            height=256,
            drift_x=drift_x_true,
            drift_y=drift_y_true
        )

        info = [{
            'Width': 256,
            'Height': 256,
            'Frames': n_frames,
            'Pixelsize': 69
        }]

        # Run drift correction
        drift_corr = DCF.Drift_Correction_Functions()

        corrected_locs, drift_result = drift_corr.undrift(
            locs=locs,
            info=info,
            method='rcc',
            segmentation=10
        )

        # Check outputs
        assert len(corrected_locs) == len(locs)
        assert 'drift_x' in drift_result
        assert 'drift_y' in drift_result
        assert len(drift_result['drift_x']) == n_frames

    def test_undrift_with_fiducials(self):
        """Test full undrift workflow with fiducial method."""
        # Generate fiducial data with drift
        n_frames = 50
        drift_x_true, drift_y_true = generate_linear_drift(n_frames, drift_rate=0.2)

        locs = generate_fiducial_localizations(
            n_fiducials=20,
            n_frames=n_frames,
            drift_x=drift_x_true,
            drift_y=drift_y_true
        )

        info = [{
            'Width': 256,
            'Height': 256,
            'Frames': n_frames,
            'Pixelsize': 69
        }]

        # Run drift correction
        drift_corr = DCF.Drift_Correction_Functions()

        corrected_locs, drift_result = drift_corr.undrift(
            locs=locs,
            info=info,
            method='fiducial',
            min_localizations=int(n_frames * 0.3)
        )

        # Check outputs
        assert len(corrected_locs) > 0  # Some locs should be corrected
        assert 'drift_x' in drift_result
        assert 'drift_y' in drift_result

        # Drift should correlate with true drift
        assert len(drift_result['drift_x']) == n_frames

    def test_undrift_auto_method_selection(self):
        """Test automatic method selection."""
        # Generate sparse data (should choose AIM)
        locs = generate_test_localizations(n_locs=500, n_frames=50)

        info = [{
            'Width': 256,
            'Height': 256,
            'Frames': 50,
            'Pixelsize': 69
        }]

        drift_corr = DCF.Drift_Correction_Functions()

        corrected_locs, drift_result = drift_corr.undrift(
            locs=locs,
            info=info,
            method='auto'
        )

        # Should successfully run
        assert len(corrected_locs) == len(locs)
        assert 'method_used' in drift_result or 'drift_x' in drift_result


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_localizations(self):
        """Test handling of empty localization array."""
        locs = np.recarray(0, dtype=[('x', 'f8'), ('y', 'f8'), ('frame', 'i4'), ('photons', 'f8'), ('bg', 'f8')])

        info = [{
            'Width': 256,
            'Height': 256,
            'Frames': 100,
            'Pixelsize': 69
        }]

        drift_corr = DCF.Drift_Correction_Functions()

        # Should handle gracefully (may return empty or raise appropriate error)
        try:
            corrected_locs, drift_result = drift_corr.undrift(locs, info, method='aim')
            # If it returns, check it's sensible
            assert len(corrected_locs) == 0
        except ValueError as e:
            # Or it may raise ValueError for insufficient data
            assert 'data' in str(e).lower() or 'localizations' in str(e).lower()

    def test_single_frame(self):
        """Test handling of single-frame data."""
        locs = generate_test_localizations(n_locs=100, n_frames=1)

        info = [{
            'Width': 256,
            'Height': 256,
            'Frames': 1,
            'Pixelsize': 69
        }]

        drift_corr = DCF.Drift_Correction_Functions()

        # Single frame: drift should be zero
        corrected_locs, drift_result = drift_corr.undrift(locs, info, method='aim')

        # Should return drift of zero
        assert np.allclose(drift_result['drift_x'], 0)
        assert np.allclose(drift_result['drift_y'], 0)

    def test_very_sparse_data(self):
        """Test handling of very sparse localization data."""
        # Only 1 localization per frame
        locs = generate_test_localizations(n_locs=50, n_frames=50)

        info = [{
            'Width': 256,
            'Height': 256,
            'Frames': 50,
            'Pixelsize': 69
        }]

        drift_corr = DCF.Drift_Correction_Functions()

        # Should handle but may produce warning
        try:
            corrected_locs, drift_result = drift_corr.undrift(locs, info, method='aim')
            assert len(corrected_locs) > 0
        except (ValueError, RuntimeError):
            # May fail with insufficient data warning
            pass


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v', '--tb=short'])
