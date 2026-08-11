#!/usr/bin/env python3
"""
Full coverage tests for pyS3M.drift_correction._base — shared dataclasses,
enum, exception, and abstract DriftCorrector interface.

Part of the drift_correction/ package coverage push (claude/TODO.md PRIORITY 1);
see test_drift_correction_{aim,fiducial,auto,facade}.py for the rest of the package.
"""
from __future__ import annotations

import numpy as np
import pytest

from pyS3M.drift_correction._base import (
    DriftCorrectionError,
    DriftCorrector,
    DriftMethod,
    DriftParameters,
    DriftResult,
    FiducialDetectionResult,
)


class TestDriftMethod:
    def test_values(self):
        assert DriftMethod.AIM.value == "aim"
        assert DriftMethod.FIDUCIAL.value == "fiducial"
        assert DriftMethod.AUTO.value == "auto"


class TestDriftParameters:
    def test_defaults_compute_intersect_d_and_roi_r_from_pixel_size(self):
        p = DriftParameters()
        assert p.intersect_d is not None
        assert p.roi_r is not None
        assert p.intersect_d > 0
        assert p.roi_r > 0

    def test_explicit_intersect_d_and_roi_r_not_overridden(self):
        p = DriftParameters(intersect_d=1.5, roi_r=3.0)
        assert p.intersect_d == 1.5
        assert p.roi_r == 3.0

    def test_validate_passes_for_defaults(self):
        DriftParameters().validate()  # should not raise

    def test_validate_segmentation_not_positive(self):
        p = DriftParameters(segmentation=0)
        with pytest.raises(DriftCorrectionError, match="Segmentation must be positive"):
            p.validate()

    def test_validate_intersect_d_not_positive(self):
        p = DriftParameters(intersect_d=0)
        with pytest.raises(DriftCorrectionError, match="Intersection distance must be positive"):
            p.validate()

    def test_validate_fiducial_threshold_percentile_out_of_range(self):
        p = DriftParameters(fiducial_threshold_percentile=0)
        with pytest.raises(DriftCorrectionError, match="threshold percentile"):
            p.validate()

    def test_validate_fiducial_threshold_percentile_above_100(self):
        p = DriftParameters(fiducial_threshold_percentile=101)
        with pytest.raises(DriftCorrectionError, match="threshold percentile"):
            p.validate()

    def test_validate_fiducial_box_size_not_positive(self):
        p = DriftParameters(fiducial_box_size_nm=0)
        with pytest.raises(DriftCorrectionError, match="box size must be positive"):
            p.validate()

    def test_validate_fiducial_min_frames_fraction_out_of_range(self):
        p = DriftParameters(fiducial_min_frames_fraction=0)
        with pytest.raises(DriftCorrectionError, match="minimum frames fraction"):
            p.validate()

    def test_validate_fiducial_histogram_bins_not_positive(self):
        p = DriftParameters(fiducial_histogram_bins=0)
        with pytest.raises(DriftCorrectionError, match="histogram bins must be positive"):
            p.validate()

    def test_validate_roi_r_not_positive(self):
        p = DriftParameters(roi_r=-1)
        with pytest.raises(DriftCorrectionError, match="ROI radius must be positive"):
            p.validate()


class TestDriftResult:
    def test_metadata_defaults_to_empty_dict(self):
        r = DriftResult(drift_x=np.array([0.0]), drift_y=np.array([0.0]))
        assert r.metadata == {}

    def test_metadata_explicit_not_overridden(self):
        r = DriftResult(drift_x=np.array([0.0]), drift_y=np.array([0.0]), metadata={"a": 1})
        assert r.metadata == {"a": 1}

    def test_to_rec_array_2d(self):
        r = DriftResult(drift_x=np.array([1.0, 2.0]), drift_y=np.array([3.0, 4.0]))
        rec = r.to_rec_array()
        assert list(rec.dtype.names) == ["x", "y"]
        np.testing.assert_allclose(rec.x, [1.0, 2.0])
        np.testing.assert_allclose(rec.y, [3.0, 4.0])

    def test_to_rec_array_3d(self):
        r = DriftResult(
            drift_x=np.array([1.0]), drift_y=np.array([2.0]), drift_z=np.array([3.0])
        )
        rec = r.to_rec_array()
        assert list(rec.dtype.names) == ["xc", "yc", "z"]
        np.testing.assert_allclose(rec.z, [3.0])


class TestFiducialDetectionResult:
    def test_construction(self):
        result = FiducialDetectionResult(
            picks=[(1.0, 2.0)],
            picked_localisations=[np.recarray((0,), dtype=[("x", float)])],
            detection_image=np.zeros((4, 4)),
            locs_with_groups=np.recarray((0,), dtype=[("x", float)]),
            n_fiducials=1,
            detection_params={},
            metadata={},
        )
        assert result.n_fiducials == 1


class TestDriftCorrectorAbstractInterface:
    """The abstract base's own `pass`-bodied methods aren't reachable through
    any concrete subclass (they all override rather than calling super()) --
    exercise them directly via a minimal subclass that forwards to super()."""

    def test_abstract_methods_reachable_via_super(self):
        class _PassThrough(DriftCorrector):
            def calculate_drift(self, locs, info, params):
                return super().calculate_drift(locs, info, params)

            def supports_3d(self):
                return super().supports_3d()

        instance = _PassThrough()
        assert instance.calculate_drift(None, None, None) is None
        assert instance.supports_3d() is None

    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            DriftCorrector()

    def test_correct_drift_workflow(self, monkeypatch):
        """correct_drift() validates params, validates locs, calls
        calculate_drift(), then applies the correction -- exercise the full
        template-method workflow with a minimal concrete corrector."""
        import pyS3M.CoordinateProcessing as CoordinateProcessing

        locs = np.rec.fromarrays(
            [np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([0, 1])],
            names=["xc", "yc", "frame"],
        )

        class _FakeCorrector(DriftCorrector):
            def calculate_drift(self, locs, info, params):
                return DriftResult(
                    drift_x=np.array([0.0, 0.0]), drift_y=np.array([0.0, 0.0])
                )

            def supports_3d(self):
                return False

        corrected, result = _FakeCorrector().correct_drift(locs, [{}], DriftParameters())
        assert len(corrected) == len(locs)
        assert isinstance(result, DriftResult)
