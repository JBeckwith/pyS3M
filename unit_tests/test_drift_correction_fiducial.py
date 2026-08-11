#!/usr/bin/env python3
"""
Full coverage tests for pyS3M.drift_correction.fiducial — FiducialDriftCorrector.

Part of the drift_correction/ package coverage push (claude/TODO.md PRIORITY 1).
Uses the shared `real_fitted_drift_fixture` (unit_tests/conftest.py) for the
auto-detection path, which needs real bright-spot density to find anything
real (test_tiffs/drift_correction/'s gold-nanoparticle fixture, ~19s fit,
shared with test_drift_correction_facade.py so it only runs once per session).
Everything else uses small hand-built synthetic recarrays.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest

from pyS3M.drift_correction._base import DriftCorrectionError, DriftParameters, DriftMethod
from pyS3M.drift_correction.fiducial import FiducialDriftCorrector


def _fiducial_locs(n_frames=10, n_fiducials=3, seed=0):
    """locs with a 'group' field: n_fiducials static-ish groups, present every frame."""
    rng = np.random.default_rng(seed)
    xc, yc, frame, group = [], [], [], []
    centres = rng.uniform(10, 90, (n_fiducials, 2))
    for f in range(n_frames):
        for g, (cx, cy) in enumerate(centres):
            xc.append(cx + rng.normal(0, 0.3))
            yc.append(cy + rng.normal(0, 0.3))
            frame.append(f)
            group.append(g)
    return np.rec.fromarrays(
        [np.array(xc), np.array(yc), np.array(frame), np.array(group, dtype=np.int32)],
        names=["xc", "yc", "frame", "group"],
    )


class TestSupports3DAndInit:
    def test_supports_3d_is_false(self):
        assert FiducialDriftCorrector().supports_3d() is False

    def test_init_does_not_raise(self):
        FiducialDriftCorrector()  # __init__ is a bare pass


class TestCalculateDriftWithGroupField:
    def test_normal_path_multi_fiducial(self):
        locs = _fiducial_locs(n_frames=10, n_fiducials=3)
        info = [{"Width": 100, "Height": 100, "Frames": 10, "Pixelsize": 69}]
        result = FiducialDriftCorrector().calculate_drift(locs, info, DriftParameters())
        assert result.method_used == DriftMethod.FIDUCIAL
        assert result.metadata["n_fiducials"] == 3
        assert len(result.drift_x) == 10
        assert np.all(np.isfinite(result.drift_x))

    def test_single_fiducial_skips_weighted_averaging_branch(self):
        locs = _fiducial_locs(n_frames=10, n_fiducials=1)
        info = [{"Width": 100, "Height": 100, "Frames": 10, "Pixelsize": 69}]
        result = FiducialDriftCorrector().calculate_drift(locs, info, DriftParameters())
        assert result.metadata["n_fiducials"] == 1

    def test_no_fiducial_localisations_raises(self):
        """A 'group' field present but every value negative (unlabelled)."""
        locs = np.rec.fromarrays(
            [np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.array([0, 1]), np.array([-1, -1])],
            names=["xc", "yc", "frame", "group"],
        )
        info = [{"Width": 100, "Height": 100, "Frames": 2, "Pixelsize": 69}]
        with pytest.raises(DriftCorrectionError, match="No fiducial localisations found"):
            FiducialDriftCorrector().calculate_drift(locs, info, DriftParameters())


class TestCalculateDriftWithoutGroupField:
    def test_no_group_and_auto_detect_disabled_raises(self):
        locs = np.rec.fromarrays(
            [np.array([1.0]), np.array([1.0]), np.array([0])], names=["xc", "yc", "frame"]
        )
        info = [{"Width": 100, "Height": 100, "Frames": 1, "Pixelsize": 69}]
        params = DriftParameters(auto_detect_fiducials=False)
        with pytest.raises(DriftCorrectionError, match="requires 'group' field"):
            FiducialDriftCorrector().calculate_drift(locs, info, params)

    def test_no_group_and_auto_detect_enabled_uses_real_fixture(self, real_fitted_drift_fixture):
        """auto_detect_fiducials=True (the default) triggers
        _detect_and_add_fiducials -- needs real bright-spot density to find
        anything, hence the real gold-nanoparticle fixture."""
        locs = real_fitted_drift_fixture["locs_rec"]
        assert not hasattr(locs, "group")
        info = real_fitted_drift_fixture["info"]
        result = FiducialDriftCorrector().calculate_drift(locs, info, DriftParameters())
        assert result.metadata["n_fiducials"] > 0
        assert np.all(np.isfinite(result.drift_x))
        assert np.all(np.isfinite(result.drift_y))


class TestGroupFiducials:
    def test_negative_groups_skipped_empty_groups_skipped(self):
        locs = np.rec.fromarrays(
            [
                np.array([1.0, 2.0, 3.0, 4.0]),
                np.array([1.0, 2.0, 3.0, 4.0]),
                np.array([0, 0, 1, 1]),
                np.array([-1, 0, 0, 2], dtype=np.int32),
            ],
            names=["xc", "yc", "frame", "group"],
        )
        picked = FiducialDriftCorrector()._group_fiducials(locs)
        # group -1 excluded; group 0 has 1 member; group 2 has 1 member
        assert len(picked) == 2
        assert all(len(g) > 0 for g in picked)


class TestCalculateCoordinateDrift:
    def test_zero_fiducials_returns_zeros(self):
        drift = FiducialDriftCorrector()._calculate_coordinate_drift([], 5, "xc")
        np.testing.assert_array_equal(drift, np.zeros(5, dtype=np.float32))

    def test_multi_fiducial_weighted_average_branch(self):
        rng = np.random.default_rng(0)
        n_frames = 10
        f0 = np.rec.fromarrays(
            [np.full(n_frames, 5.0), np.full(n_frames, 5.0), np.arange(n_frames)],
            names=["xc", "yc", "frame"],
        )
        f1 = np.rec.fromarrays(
            [5 + rng.normal(0, 0.5, n_frames), 5 + rng.normal(0, 0.5, n_frames), np.arange(n_frames)],
            names=["xc", "yc", "frame"],
        )
        drift = FiducialDriftCorrector()._calculate_coordinate_drift([f0, f1], n_frames, "xc")
        assert len(drift) == n_frames
        assert np.all(np.isfinite(drift))

    def test_coordinate_missing_on_one_fiducial_is_skipped(self):
        """hasattr(fiducial_locs, coordinate) False branch: one fiducial
        lacks a 'yc' field entirely."""
        n_frames = 5
        f0 = np.rec.fromarrays(
            [np.arange(n_frames, dtype=float), np.arange(n_frames)], names=["xc", "frame"]
        )
        drift = FiducialDriftCorrector()._calculate_coordinate_drift([f0], n_frames, "yc")
        np.testing.assert_array_equal(drift, np.zeros(n_frames, dtype=np.float32))


class TestAddGroupField:
    def test_assigns_group_ids_by_frame_and_position_match(self):
        locs = np.rec.fromarrays(
            [
                np.array([1.0, 2.0, 3.0]),
                np.array([1.0, 2.0, 3.0]),
                np.array([0, 1, 2]),
            ],
            names=["xc", "yc", "frame"],
        )
        fiducial0 = locs[[0]].copy()  # matches the first row exactly
        new_locs = FiducialDriftCorrector()._add_group_field(locs, [fiducial0], picks=[])
        assert "group" in new_locs.dtype.names
        assert new_locs.group[0] == 0
        assert new_locs.group[1] == -1
        assert new_locs.group[2] == -1


class TestDetectAndAddFiducialsErrorBranches:
    def test_render_none_raises(self, monkeypatch, real_fitted_drift_fixture):
        import pyS3M.drift_correction.fiducial as fiducial_mod

        monkeypatch.setattr(fiducial_mod, "render", None)
        locs = real_fitted_drift_fixture["locs_rec"]
        info = real_fitted_drift_fixture["info"]
        with pytest.raises(DriftCorrectionError, match="requires render module"):
            FiducialDriftCorrector()._detect_and_add_fiducials(locs, info, DriftParameters())

    def test_no_picks_detected_raises(self, monkeypatch, real_fitted_drift_fixture):
        import pyS3M.drift_correction.fiducial as fiducial_mod

        monkeypatch.setattr(
            fiducial_mod, "render",
            type("R", (), {"render": staticmethod(lambda **kw: (None, fiducial_mod.np.zeros((50, 50))))}),
        )
        locs = real_fitted_drift_fixture["locs_rec"]
        info = real_fitted_drift_fixture["info"]
        # threshold_percentile=100 on an all-zero image -> localise finds nothing
        params = DriftParameters(fiducial_threshold_percentile=99.9999999)
        with pytest.raises(DriftCorrectionError, match="No fiducial candidates detected"):
            FiducialDriftCorrector()._detect_and_add_fiducials(locs, info, params)

    def test_localise_import_error_raises(self, monkeypatch, real_fitted_drift_fixture):
        monkeypatch.setitem(sys.modules, "localise", None)
        locs = real_fitted_drift_fixture["locs_rec"]
        info = real_fitted_drift_fixture["info"]
        with pytest.raises(DriftCorrectionError, match="localise module required"):
            FiducialDriftCorrector()._detect_and_add_fiducials(locs, info, DriftParameters())

    def test_postprocess_import_error_raises(self, monkeypatch, real_fitted_drift_fixture):
        monkeypatch.setitem(sys.modules, "postprocess", None)
        locs = real_fitted_drift_fixture["locs_rec"]
        info = real_fitted_drift_fixture["info"]
        with pytest.raises(DriftCorrectionError, match="postprocess module required"):
            FiducialDriftCorrector()._detect_and_add_fiducials(locs, info, DriftParameters())

    def test_no_valid_picks_after_min_frames_filter_raises(self, real_fitted_drift_fixture):
        locs = real_fitted_drift_fixture["locs_rec"]
        info = real_fitted_drift_fixture["info"]
        # Impossible minimum -- every candidate pick gets filtered out.
        params = DriftParameters(fiducial_min_frames_fraction=1.0)
        with pytest.raises(DriftCorrectionError, match="No fiducials found with minimum"):
            FiducialDriftCorrector()._detect_and_add_fiducials(locs, info, params)


class TestModuleImportFallbacks:
    """The module-level try/except ImportError blocks (FiducialDetector/
    DriftPlotter, and render/postprocess) are environment-dependent -- they
    only take their except branch when those modules genuinely can't be
    imported. Force that by blocking the import via sys.modules and
    reloading, then restore so the rest of the suite sees a normal module.
    """

    def test_fiducial_detector_import_failure_fallback(self, monkeypatch):
        import importlib
        import pyS3M.drift_correction.fiducial as fiducial_mod

        monkeypatch.setitem(sys.modules, "pyS3M.FiducialDetection", None)
        try:
            with pytest.warns(UserWarning, match="Could not import FiducialDetector"):
                importlib.reload(fiducial_mod)
            assert fiducial_mod._drift_plotter is None
            assert fiducial_mod._fiducial_detector is None
        finally:
            monkeypatch.undo()
            importlib.reload(fiducial_mod)

    def test_render_postprocess_import_failure_fallback(self, monkeypatch):
        import importlib
        import pyS3M.drift_correction.fiducial as fiducial_mod

        monkeypatch.setitem(sys.modules, "pyS3M.render", None)
        monkeypatch.setitem(sys.modules, "pyS3M.postprocess", None)
        try:
            with pytest.warns(UserWarning, match="Could not import render/postprocess"):
                importlib.reload(fiducial_mod)
            assert fiducial_mod.render is None
            assert fiducial_mod.postprocess is None
        finally:
            monkeypatch.undo()
            importlib.reload(fiducial_mod)
