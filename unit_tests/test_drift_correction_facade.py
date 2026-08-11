#!/usr/bin/env python3
"""
Full coverage tests for pyS3M.drift_correction._facade — Drift_Correction_Functions,
the main public-facing drift correction class.

Part of the drift_correction/ package coverage push (claude/TODO.md PRIORITY 1).
By far the biggest file in the package (642 statements, was at 10% coverage).
Strategy:

- Trivial delegation methods (convert_pixels_to_nm, etc.) are tested against a
  mocked coordinate_processor/fiducial_detector -- they're one-line passthroughs,
  the real CoordinateProcessor/FiducialDetector logic has its own coverage.
- Pure-logic private helpers (_add_group_field, _filter_fiducials_fast,
  _find_indices_in_original_locs, apply_validated_fiducial_drift_correction,
  detect_high_density_regions_from_image, identify_real_fiducials_with_clustering)
  are tested with small hand-built synthetic recarrays/images -- no fixture needed.
- The multi-step orchestration methods that need real bright-spot density to find
  anything (undrift_with_fiducial_detection, detect_fiducials incl. both the
  temporal-chunking and non-chunking branches) use the shared
  `real_fitted_drift_fixture` (unit_tests/conftest.py, test_tiffs/drift_correction/,
  ~19s, shared with test_drift_correction_fiducial.py so it only runs once per
  session).
- Plotting methods (_plot_single_gaussian_validation) are smoke-tested only
  (matplotlib Agg backend, doesn't crash, produces a figure) -- visual
  correctness isn't pytest's job here.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from pyS3M.drift_correction._base import DriftCorrectionError, DriftParameters, DriftMethod
from pyS3M.drift_correction._facade import Drift_Correction_Functions


# ======================================================================
# __init__ and its two try/except ImportError fallbacks
# ======================================================================

class TestInit:
    def test_normal_construction(self):
        dcf = Drift_Correction_Functions(camera="ximea")
        assert dcf.pixel_size == pytest.approx(0.069)
        assert dcf.factory is not None
        assert dcf.aim_corrector is not None
        assert dcf.plotter is not None
        assert dcf.fiducial_detector is not None
        assert dcf.coordinate_processor is not None

    def test_explicit_pixel_size(self):
        dcf = Drift_Correction_Functions(camera="zwo", pixel_size=0.05)
        assert dcf.pixel_size == 0.05

    def test_fiducial_detector_import_failure_fallback(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pyS3M.FiducialDetection", None)
        dcf = Drift_Correction_Functions(camera="ximea")
        assert dcf.plotter is None
        assert dcf.fiducial_detector is None

    def test_coordinate_processor_import_failure_fallback(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pyS3M.CoordinateProcessing", None)
        dcf = Drift_Correction_Functions(camera="ximea")
        assert dcf.coordinate_processor is None


# ======================================================================
# available_methods / method_info
# ======================================================================

class TestAvailableMethodsAndInfo:
    def test_available_methods(self):
        dcf = Drift_Correction_Functions()
        methods = dcf.available_methods()
        assert set(methods) == {"aim", "fiducial", "auto"}

    def test_method_info_by_string(self):
        dcf = Drift_Correction_Functions()
        info = dcf.method_info("aim")
        assert info["name"] == "aim"
        assert info["supports_3d"] is True
        assert info["class"] == "AIMDriftCorrector"
        assert isinstance(info["description"], str)

    def test_method_info_by_enum(self):
        dcf = Drift_Correction_Functions()
        info = dcf.method_info(DriftMethod.FIDUCIAL)
        assert info["name"] == "fiducial"
        assert info["supports_3d"] is False


# ======================================================================
# Delegation methods
# ======================================================================

class TestFiducialDetectorDelegation:
    def test_delegates_when_available(self):
        dcf = Drift_Correction_Functions()
        dcf.fiducial_detector = MagicMock()
        dcf.fiducial_detector.identify_real_fiducials_with_clustering.return_value = "result"
        out = dcf.identify_real_fiducials_with_clustering_delegated(1, foo=2)
        assert out == "result"
        dcf.fiducial_detector.identify_real_fiducials_with_clustering.assert_called_once_with(1, foo=2)

    def test_raises_when_unavailable(self):
        dcf = Drift_Correction_Functions()
        dcf.fiducial_detector = None
        with pytest.raises(RuntimeError, match="FiducialDetector module not available"):
            dcf.identify_real_fiducials_with_clustering_delegated()


class TestCoordinateProcessorDelegation:
    @pytest.mark.parametrize("method_name", [
        "convert_pixels_to_nm", "convert_nm_to_pixels",
        "apply_drift_correction", "create_spatial_grid",
        "bin_localisations_spatially",
    ])
    def test_delegates_when_available(self, method_name):
        dcf = Drift_Correction_Functions()
        dcf.coordinate_processor = MagicMock()
        getattr(dcf.coordinate_processor, method_name).return_value = "ok"
        out = getattr(dcf, method_name)(1, 2, kw=3)
        assert out == "ok"
        getattr(dcf.coordinate_processor, method_name).assert_called_once_with(1, 2, kw=3)

    @pytest.mark.parametrize("method_name", [
        "convert_pixels_to_nm", "convert_nm_to_pixels",
        "apply_drift_correction", "create_spatial_grid",
        "bin_localisations_spatially",
    ])
    def test_raises_when_unavailable(self, method_name):
        dcf = Drift_Correction_Functions()
        dcf.coordinate_processor = None
        with pytest.raises(RuntimeError, match="CoordinateProcessor module not available"):
            getattr(dcf, method_name)()


class TestRunAim2DAnd3D:
    def _small_locs_and_info(self, n_locs=300, n_frames=20, width=100.0, height=100.0):
        rng = np.random.default_rng(0)
        frames = rng.integers(0, n_frames, n_locs)
        x = rng.uniform(0, width, n_locs)
        y = rng.uniform(0, height, n_locs)
        locs = np.rec.fromarrays([x, y, frames], names=["xc", "yc", "frame"])
        info = [{"Width": width, "Height": height, "Frames": n_frames, "Pixelsize": 69}]
        return locs, info

    def test_run_aim_2d_default_params(self):
        dcf = Drift_Correction_Functions()
        locs, info = self._small_locs_and_info()
        drift_x, drift_y, meta = dcf.run_aim_2d(locs, info)
        assert np.all(np.isfinite(drift_x))
        assert np.all(np.isfinite(drift_y))
        assert "segmentation" in meta

    def test_run_aim_2d_explicit_params(self):
        dcf = Drift_Correction_Functions()
        locs, info = self._small_locs_and_info()
        drift_x, drift_y, meta = dcf.run_aim_2d(
            locs, info, segmentation=5, intersect_d=1.0, roi_r=2.0
        )
        assert meta["segmentation"] == 5

    def test_run_aim_3d(self):
        dcf = Drift_Correction_Functions()
        rng = np.random.default_rng(1)
        n_locs, n_frames, width, height = 300, 20, 100.0, 100.0
        frames = rng.integers(0, n_frames, n_locs)
        x = rng.uniform(0, width, n_locs)
        y = rng.uniform(0, height, n_locs)
        z = rng.uniform(-200, 200, n_locs)
        locs = np.rec.fromarrays([x, y, z, frames], names=["xc", "yc", "z", "frame"])
        info = [{"Width": width, "Height": height, "Frames": n_frames, "Pixelsize": 69}]
        drift_x, drift_y, drift_z, meta = dcf.run_aim_3d(
            locs, info, segmentation=5, intersect_d=1.0, roi_r=2.0
        )
        assert np.all(np.isfinite(drift_z))


# ======================================================================
# Module-level import fallback (render/postprocess)
# ======================================================================

class TestModuleImportFallback:
    def test_render_postprocess_import_failure_fallback(self, monkeypatch):
        import importlib
        import pyS3M.drift_correction._facade as facade_mod

        monkeypatch.setitem(sys.modules, "pyS3M.render", None)
        monkeypatch.setitem(sys.modules, "pyS3M.postprocess", None)
        try:
            with pytest.warns(UserWarning, match="Could not import render/postprocess"):
                importlib.reload(facade_mod)
            assert facade_mod.render is None
            assert facade_mod.postprocess is None
        finally:
            monkeypatch.undo()
            importlib.reload(facade_mod)


# ======================================================================
# _add_group_field / _find_indices_in_original_locs / _manual_add_group_field
# ======================================================================

class TestAddGroupField:
    def test_assigns_group_by_frame_and_position(self):
        dcf = Drift_Correction_Functions()
        locs = np.rec.fromarrays(
            [np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]), np.array([0, 1, 2])],
            names=["xc", "yc", "frame"],
        )
        fiducial0 = locs[[0]].copy()
        new_locs = dcf._add_group_field(locs, [fiducial0], picks=[])
        assert new_locs.group[0] == 0
        assert new_locs.group[1] == -1


class TestFindIndicesInOriginalLocs:
    def test_unique_positions(self):
        dcf = Drift_Correction_Functions()
        locs = np.rec.fromarrays(
            [np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]), np.array([0, 1, 2])],
            names=["xc", "yc", "frame"],
        )
        fid = locs[[0, 2]].copy()
        idx = dcf._find_indices_in_original_locs(locs, fid)
        np.testing.assert_array_equal(sorted(idx), [0, 2])

    def test_duplicate_positions_hashed_as_list(self):
        """Three locs sharing the same (frame, x, y) key exercise both the
        isinstance(..., list) branches: the second occurrence converts the
        dict value to a list (build-time), the third appends to that
        existing list (build-time) -- lookup then extends with the list."""
        dcf = Drift_Correction_Functions()
        locs = np.rec.fromarrays(
            [np.array([1.0, 1.0, 1.0, 3.0]), np.array([1.0, 1.0, 1.0, 3.0]), np.array([0, 0, 0, 2])],
            names=["xc", "yc", "frame"],
        )
        fid = locs[[0, 1, 2]].copy()
        idx = dcf._find_indices_in_original_locs(locs, fid)
        assert sorted(idx) == [0, 0, 0, 1, 1, 1, 2, 2, 2]


class TestManualAddGroupField:
    def test_adds_group_column(self):
        dcf = Drift_Correction_Functions()
        locs = np.rec.fromarrays(
            [np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.array([0, 1])],
            names=["xc", "yc", "frame"],
        )
        group = np.array([0, -1], dtype=np.int32)
        new_locs = dcf._manual_add_group_field(locs, group)
        assert "group" in new_locs.dtype.names
        np.testing.assert_array_equal(new_locs.group, group)


class TestAddGroupFieldToLocs:
    def _locs(self, n=20):
        rng = np.random.default_rng(0)
        return np.rec.fromarrays(
            [rng.uniform(0, 50, n), rng.uniform(0, 50, n), np.arange(n)],
            names=["xc", "yc", "frame"],
        )

    def test_empty_picked_locs_list(self):
        dcf = Drift_Correction_Functions()
        locs = self._locs()
        out = dcf._add_group_field_to_locs(locs, [])
        assert np.all(out.group == -1)

    def test_few_fiducials_no_progress_bar(self):
        dcf = Drift_Correction_Functions()
        locs = self._locs()
        fiducials = [locs[[i]].copy() for i in range(3)]  # <=5, no progress bar
        out = dcf._add_group_field_to_locs(locs, fiducials)
        assert (out.group >= 0).sum() == 3

    def test_many_fiducials_shows_progress_bar(self):
        """len(picked_locs_list) > 5 triggers the progress-bar branch."""
        dcf = Drift_Correction_Functions()
        locs = self._locs(n=20)
        fiducials = [locs[[i]].copy() for i in range(8)]
        out = dcf._add_group_field_to_locs(locs, fiducials)
        assert (out.group >= 0).sum() == 8

    def test_lib_import_error_falls_back_to_manual(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pyS3M.lib", None)
        dcf = Drift_Correction_Functions()
        locs = self._locs()
        fiducials = [locs[[0]].copy()]
        out = dcf._add_group_field_to_locs(locs, fiducials)
        assert out.group[0] == 0


# ======================================================================
# detect_high_density_regions_from_image / select_puncta_from_regions
# ======================================================================

class TestDetectHighDensityRegionsFromImage:
    def test_finds_regions_in_synthetic_image(self):
        dcf = Drift_Correction_Functions()
        rng = np.random.default_rng(0)
        img = rng.uniform(0, 1, (60, 60))
        img[10:15, 10:15] = 100
        img[40:45, 40:45] = 100
        centres, mask, threshold, meta = dcf.detect_high_density_regions_from_image(
            img, threshold_percentile=97, create_plot=False
        )
        assert meta["n_regions_detected"] >= 1
        assert mask.shape == img.shape

    def test_empty_image_raises(self):
        dcf = Drift_Correction_Functions()
        with pytest.raises(DriftCorrectionError, match="no non-zero values"):
            dcf.detect_high_density_regions_from_image(np.zeros((10, 10)), create_plot=False)

    def test_create_plot_true_uses_plotter(self):
        dcf = Drift_Correction_Functions()
        rng = np.random.default_rng(0)
        img = rng.uniform(0, 1, (60, 60))
        img[10:15, 10:15] = 100
        dcf.detect_high_density_regions_from_image(
            img, threshold_percentile=99.9, create_plot=True
        )

    def test_create_plot_true_no_plotter_warns(self):
        dcf = Drift_Correction_Functions()
        dcf.plotter = None
        rng = np.random.default_rng(0)
        img = rng.uniform(0, 1, (60, 60))
        img[10:15, 10:15] = 100
        dcf.detect_high_density_regions_from_image(
            img, threshold_percentile=99.9, create_plot=True
        )


class TestSelectPunctaFromRegions:
    def test_delegates_when_available(self):
        dcf = Drift_Correction_Functions()
        dcf.fiducial_detector = MagicMock()
        dcf.fiducial_detector.select_puncta_from_regions.return_value = ("a", "b")
        out = dcf.select_puncta_from_regions(
            locs=None, region_centres=[], binary_mask=None
        )
        assert out == ("a", "b")

    def test_raises_when_unavailable(self):
        dcf = Drift_Correction_Functions()
        dcf.fiducial_detector = None
        with pytest.raises(RuntimeError, match="FiducialDetector not available"):
            dcf.select_puncta_from_regions(locs=None, region_centres=[], binary_mask=None)


# ======================================================================
# identify_real_fiducials_with_clustering / _plot_single_gaussian_validation
# ======================================================================

class TestIdentifyRealFiducialsWithClustering:
    def _puncta(self, n, cx, cy, spread=0.5, seed=0):
        rng = np.random.default_rng(seed)
        return np.rec.fromarrays(
            [cx + rng.normal(0, spread, n), cy + rng.normal(0, spread, n)],
            names=["xc", "yc"],
        )

    def test_invalid_retention_percentage_raises(self):
        dcf = Drift_Correction_Functions()
        with pytest.raises(ValueError, match="retention_percentage must be between"):
            dcf.identify_real_fiducials_with_clustering([self._puncta(20, 0, 0)], retention_percentage=0)

    def test_empty_input(self):
        dcf = Drift_Correction_Functions()
        validated, meta = dcf.identify_real_fiducials_with_clustering([], create_plot=False)
        assert validated == []
        assert meta["validation_rate"] == 0

    def test_region_below_10_locs_skipped(self):
        dcf = Drift_Correction_Functions()
        validated, meta = dcf.identify_real_fiducials_with_clustering(
            [self._puncta(5, 0, 0)], frame_count=100, create_plot=False
        )
        assert validated == []

    def test_region_kept_below_min_samples_after_radial_filter_discarded(self):
        """Distinct from test_region_below_min_samples_skipped: this region
        has enough raw points to pass the initial n_locs/min_samples gate,
        but a tiny retention_percentage discards so many via the radial
        threshold that n_kept ends up below min_samples anyway."""
        dcf = Drift_Correction_Functions()
        validated, meta = dcf.identify_real_fiducials_with_clustering(
            [self._puncta(20, 0, 0, spread=20, seed=1)],
            frame_count=100, retention_percentage=0.001, create_plot=False,
        )
        assert validated == []

    def test_region_below_min_samples_skipped(self):
        dcf = Drift_Correction_Functions()
        # frame_count default (100000) -> min_samples=70, this region has only 20
        validated, meta = dcf.identify_real_fiducials_with_clustering(
            [self._puncta(20, 0, 0)], create_plot=False
        )
        assert validated == []

    def test_valid_region_validated_with_plotting(self):
        dcf = Drift_Correction_Functions()
        validated, meta = dcf.identify_real_fiducials_with_clustering(
            [self._puncta(20, 10, 10)], frame_count=100, create_plot=True
        )
        assert len(validated) == 1
        assert meta["n_validated_fiducials"] == 1

    def test_valid_region_no_plotter_warns(self):
        dcf = Drift_Correction_Functions()
        dcf.plotter = None
        validated, meta = dcf.identify_real_fiducials_with_clustering(
            [self._puncta(20, 10, 10)], frame_count=100, create_plot=True
        )
        assert len(validated) == 1

    def test_gaussian_fitting_failure_is_caught_and_skipped(self, monkeypatch):
        """A puncta array that makes GaussianMixture.fit raise is caught by
        the broad except and logged/skipped rather than propagating."""
        import sklearn.mixture

        def _boom(self, X):
            raise ValueError("synthetic failure")

        monkeypatch.setattr(sklearn.mixture.GaussianMixture, "fit", _boom)
        dcf = Drift_Correction_Functions()
        validated, meta = dcf.identify_real_fiducials_with_clustering(
            [self._puncta(20, 10, 10)], frame_count=100, create_plot=False
        )
        assert validated == []

    def test_large_region_triggers_extra_gc(self):
        """n_locs > 10000 hits the extra `if n_locs > 10000: gc.collect()` line."""
        dcf = Drift_Correction_Functions()
        validated, meta = dcf.identify_real_fiducials_with_clustering(
            [self._puncta(10001, 10, 10, spread=0.2)], frame_count=100, create_plot=False
        )
        assert len(validated) == 1


class TestPlotSingleGaussianValidation:
    """Smoke test only -- matplotlib Agg backend, doesn't crash, no pixel
    assertions (visual correctness is the notebooks' job)."""

    def test_runs_without_error_with_plotter(self, tmp_path):
        dcf = Drift_Correction_Functions()
        rng = np.random.default_rng(0)
        n = 20
        puncta = np.rec.fromarrays(
            [10 + rng.normal(0, 1, n), 10 + rng.normal(0, 1, n)], names=["xc", "yc"]
        )
        kept_mask = np.zeros(n, dtype=bool)
        kept_mask[:15] = True
        metadata = {
            "original_n_locs": n, "validated_n_locs": 15, "retention_rate": 0.75,
            "gaussian_sigma_nm": 12.3, "radial_threshold_nm": 30.0,
        }
        dcf._plot_single_gaussian_validation(
            puncta, puncta[kept_mask], kept_mask,
            np.abs(rng.normal(0, 1, n)), 0, metadata,
            str(tmp_path / "out.png"), "Test", r_threshold=2.0, display=False,
        )
        assert list(tmp_path.glob("*gaussian_region*"))

    def test_no_plotter_uses_manual_scatter(self):
        dcf = Drift_Correction_Functions()
        dcf.plotter = None
        rng = np.random.default_rng(0)
        n = 10
        puncta = np.rec.fromarrays(
            [10 + rng.normal(0, 1, n), 10 + rng.normal(0, 1, n)], names=["xc", "yc"]
        )
        kept_mask = np.zeros(n, dtype=bool)
        kept_mask[:5] = True
        metadata = {
            "original_n_locs": n, "validated_n_locs": 5, "retention_rate": 0.5,
            "gaussian_sigma_nm": 12.3, "radial_threshold_nm": 30.0,
        }
        dcf._plot_single_gaussian_validation(
            puncta, puncta[kept_mask], kept_mask,
            np.abs(rng.normal(0, 1, n)), 0, metadata,
            None, "Test", r_threshold=2.0, display=False,
        )

    def test_all_discarded_and_high_retention_colour_branches(self):
        """Covers the retention_rate quality-colour branches: all discarded
        (kept_mask all False -> discarded_mask branch only) and a >0.95
        retention rate (lightyellow branch)."""
        dcf = Drift_Correction_Functions()
        rng = np.random.default_rng(0)
        n = 10
        puncta = np.rec.fromarrays(
            [10 + rng.normal(0, 1, n), 10 + rng.normal(0, 1, n)], names=["xc", "yc"]
        )
        kept_mask = np.zeros(n, dtype=bool)  # nothing kept
        metadata = {
            "original_n_locs": n, "validated_n_locs": 0, "retention_rate": 0.97,
            "gaussian_sigma_nm": 12.3, "radial_threshold_nm": 30.0,
        }
        dcf._plot_single_gaussian_validation(
            puncta, puncta[kept_mask], kept_mask,
            np.abs(rng.normal(0, 1, n)), 0, metadata,
            None, "Test", r_threshold=2.0, display=False,
        )

    def test_import_error_fallback(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pyS3M.PlottingBase", None)
        dcf = Drift_Correction_Functions()
        rng = np.random.default_rng(0)
        n = 5
        puncta = np.rec.fromarrays(
            [rng.normal(0, 1, n), rng.normal(0, 1, n)], names=["xc", "yc"]
        )
        kept_mask = np.ones(n, dtype=bool)
        metadata = {
            "original_n_locs": n, "validated_n_locs": n, "retention_rate": 1.0,
            "gaussian_sigma_nm": 1.0, "radial_threshold_nm": 1.0,
        }
        # Should return quietly (logged warning), not raise.
        dcf._plot_single_gaussian_validation(
            puncta, puncta[kept_mask], kept_mask,
            np.abs(rng.normal(0, 1, n)), 0, metadata,
            None, "Test", r_threshold=2.0, display=False,
        )


# ======================================================================
# _filter_fiducials_fast / apply_validated_fiducial_drift_correction
# ======================================================================

class TestFilterFiducialsFast:
    def test_normal_case(self):
        dcf = Drift_Correction_Functions()
        rng = np.random.default_rng(0)
        all_x = rng.normal(0, 1, (20, 5))
        all_y = rng.normal(0, 1, (20, 5))
        valid, meta = dcf._filter_fiducials_fast(all_x, all_y)
        assert valid.shape == (5,)
        assert "median_variance" in meta

    def test_all_nan_variance_returns_all_invalid(self):
        dcf = Drift_Correction_Functions()
        all_x = np.full((10, 3), np.nan)
        all_y = np.full((10, 3), np.nan)
        valid, meta = dcf._filter_fiducials_fast(all_x, all_y)
        assert not np.any(valid)
        assert meta == {}

    def test_finite_variance_but_all_nan_rms(self):
        """A degenerate case where every fiducial's x and y are each valid
        at a single, *disjoint* frame -- per-column variance is finite
        (single-point variance = 0) but x**2+y**2 is NaN at every frame
        (never both finite at once), so RMS ends up all-NaN despite
        variance being computable. Exercises the len(finite_rms) == 0
        early-return, distinct from the len(finite_variances) == 0 one."""
        dcf = Drift_Correction_Functions()
        n_frames, n_fid = 5, 3
        all_x = np.full((n_frames, n_fid), np.nan)
        all_y = np.full((n_frames, n_fid), np.nan)
        all_x[0, :] = [1.0, 2.0, 3.0]
        all_y[1, :] = [1.0, 2.0, 3.0]
        valid, meta = dcf._filter_fiducials_fast(all_x, all_y)
        assert np.all(valid)
        assert meta["n_rms_filtered"] == 0
        assert np.isnan(meta["median_rms"])

    def test_outlier_fiducial_filtered_by_variance(self):
        dcf = Drift_Correction_Functions()
        rng = np.random.default_rng(0)
        all_x = rng.normal(0, 0.1, (20, 5))
        all_y = rng.normal(0, 0.1, (20, 5))
        all_x[:, 0] = rng.normal(0, 50, 20)  # wild outlier fiducial
        all_y[:, 0] = rng.normal(0, 50, 20)
        valid, meta = dcf._filter_fiducials_fast(all_x, all_y)
        assert not valid[0]
        assert meta["n_variance_filtered"] >= 1


class TestApplyValidatedFiducialDriftCorrection:
    def _make_fiducial(self, n_frames, cx, cy, drift, with_err=True, seed=0):
        rng = np.random.default_rng(seed)
        frames = np.arange(n_frames)
        xc = cx + drift[:, 0] + rng.normal(0, 0.05, n_frames)
        yc = cy + drift[:, 1] + rng.normal(0, 0.05, n_frames)
        if with_err:
            xc_err = np.full(n_frames, 0.05)
            yc_err = np.full(n_frames, 0.05)
            return np.rec.fromarrays(
                [xc, yc, frames, xc_err, yc_err],
                names=["xc", "yc", "frame", "xc_err", "yc_err"],
            )
        return np.rec.fromarrays([xc, yc, frames], names=["xc", "yc", "frame"])

    def _combined_locs(self, fiducials):
        xc = np.concatenate([f.xc for f in fiducials])
        yc = np.concatenate([f.yc for f in fiducials])
        frame = np.concatenate([f.frame for f in fiducials])
        return np.rec.fromarrays([xc, yc, frame], names=["xc", "yc", "frame"])

    def test_no_validated_fiducials_raises(self):
        dcf = Drift_Correction_Functions()
        locs = np.rec.fromarrays(
            [np.array([1.0]), np.array([1.0]), np.array([0])], names=["xc", "yc", "frame"]
        )
        with pytest.raises(ValueError, match="No validated fiducials provided"):
            dcf.apply_validated_fiducial_drift_correction(locs, [])

    def test_with_error_fields_weighted(self):
        dcf = Drift_Correction_Functions()
        n_frames = 8
        drift = np.stack([np.linspace(0, 1, n_frames), np.linspace(0, 0.5, n_frames)], axis=1)
        fiducials = [
            self._make_fiducial(n_frames, 10, 10, drift, seed=0),
            self._make_fiducial(n_frames, 20, 20, drift, seed=1),
            self._make_fiducial(n_frames, 30, 30, drift, seed=2),
        ]
        locs = self._combined_locs(fiducials)
        corrected, info = dcf.apply_validated_fiducial_drift_correction(locs, fiducials)
        assert "is_fiducial" in corrected.dtype.names
        assert len(corrected) == len(locs)
        assert np.all(corrected.is_fiducial)  # every loc here IS a fiducial loc

    def test_without_error_fields_uniform_weights(self):
        dcf = Drift_Correction_Functions()
        n_frames = 8
        drift = np.stack([np.linspace(0, 1, n_frames), np.linspace(0, 0.5, n_frames)], axis=1)
        fiducials = [
            self._make_fiducial(n_frames, 10, 10, drift, with_err=False, seed=0),
            self._make_fiducial(n_frames, 20, 20, drift, with_err=False, seed=1),
        ]
        locs = self._combined_locs(fiducials)
        corrected, info = dcf.apply_validated_fiducial_drift_correction(locs, fiducials)
        assert len(corrected) == len(locs)

    def test_fiducial_with_duplicate_frame_is_skipped(self):
        """A fiducial cluster with two localisations in the same frame hits
        the `len(frames) == len(np.unique(frames))`-False skip branch."""
        dcf = Drift_Correction_Functions()
        n_frames = 8
        drift = np.stack([np.linspace(0, 1, n_frames), np.linspace(0, 0.5, n_frames)], axis=1)
        good = self._make_fiducial(n_frames, 10, 10, drift, seed=0)
        # duplicate-frame fiducial: two locs claiming frame 0
        dup = np.rec.fromarrays(
            [np.array([20.0, 20.1]), np.array([20.0, 20.1]), np.array([0, 0]),
             np.array([0.05, 0.05]), np.array([0.05, 0.05])],
            names=["xc", "yc", "frame", "xc_err", "yc_err"],
        )
        locs = self._combined_locs([good, dup])
        corrected, info = dcf.apply_validated_fiducial_drift_correction(locs, [good, dup])
        assert len(corrected) > 0

    def test_all_nan_after_median_subtraction_raises(self):
        """Every fiducial is empty -> all_corrected_x stays all-NaN."""
        dcf = Drift_Correction_Functions()
        empty = np.rec.fromarrays(
            [np.array([]), np.array([]), np.array([], dtype=int)],
            names=["xc", "yc", "frame"],
        )
        locs = np.rec.fromarrays(
            [np.array([1.0]), np.array([1.0]), np.array([0])], names=["xc", "yc", "frame"]
        )
        with pytest.raises(ValueError, match="No valid fiducials found after median subtraction"):
            dcf.apply_validated_fiducial_drift_correction(locs, [empty])


# ======================================================================
# detect_fiducials / _detect_fiducials_with_chunking / _link_candidates_across_chunks
# undrift_with_fiducial_detection
#
# These need real bright-spot density to find anything real -- use the
# shared real_fitted_drift_fixture (test_tiffs/drift_correction/, gold
# nanoparticle fiducials), also used by test_drift_correction_fiducial.py so
# the ~19s fit only happens once per session.
# ======================================================================

class TestLinkCandidatesAcrossChunks:
    def test_unmatched_candidate_starts_new_track(self):
        """A candidate too far from any existing track tip starts its own
        new track (the `for remaining_candidate in chunk_candidates:
        tracks.append([remaining_candidate])` branch)."""
        dcf = Drift_Correction_Functions()
        candidates = [
            (10.0, 10.0, 0, 0.5),
            (500.0, 500.0, 1, 1.5),  # far from (10,10) -> new track
        ]
        tracks = dcf._link_candidates_across_chunks(
            candidates, n_chunks=2, max_distance_nm=10.0, pixelsize=1.0
        )
        assert len(tracks) == 2
        assert all(len(t) == 1 for t in tracks)


class TestDetectFiducials:
    def test_render_none_raises(self, monkeypatch, real_fitted_drift_fixture):
        import pyS3M.drift_correction._facade as facade_mod

        monkeypatch.setattr(facade_mod, "render", None)
        dcf = Drift_Correction_Functions()
        with pytest.raises(DriftCorrectionError, match="requires render module"):
            dcf.detect_fiducials(
                real_fitted_drift_fixture["locs_rec"], real_fitted_drift_fixture["info"]
            )

    def test_chunked_default_finds_all_fiducials(self, real_fitted_drift_fixture):
        dcf = Drift_Correction_Functions()
        result = dcf.detect_fiducials(
            real_fitted_drift_fixture["locs_rec"], real_fitted_drift_fixture["info"],
            plot_results=False, use_temporal_chunking=True, n_chunks=5,
        )
        assert result.n_fiducials > 0
        assert "group" in result.locs_with_groups.dtype.names

    def test_chunked_with_plotting(self, real_fitted_drift_fixture):
        dcf = Drift_Correction_Functions()
        result = dcf.detect_fiducials(
            real_fitted_drift_fixture["locs_rec"], real_fitted_drift_fixture["info"],
            plot_results=True, use_temporal_chunking=True, n_chunks=5,
        )
        assert result.n_fiducials > 0

    def test_chunked_no_plotter_warns(self, real_fitted_drift_fixture):
        dcf = Drift_Correction_Functions()
        dcf.plotter = None
        result = dcf.detect_fiducials(
            real_fitted_drift_fixture["locs_rec"], real_fitted_drift_fixture["info"],
            plot_results=True, use_temporal_chunking=True, n_chunks=5,
        )
        assert result.n_fiducials > 0

    def test_non_chunked_path(self, real_fitted_drift_fixture):
        dcf = Drift_Correction_Functions()
        result = dcf.detect_fiducials(
            real_fitted_drift_fixture["locs_rec"], real_fitted_drift_fixture["info"],
            plot_results=False, use_temporal_chunking=False,
        )
        assert result.n_fiducials >= 1

    def test_non_chunked_localise_import_error_raises(self, monkeypatch, real_fitted_drift_fixture):
        monkeypatch.setitem(sys.modules, "pyS3M.localise", None)
        dcf = Drift_Correction_Functions()
        with pytest.raises(DriftCorrectionError, match="localise module required"):
            dcf.detect_fiducials(
                real_fitted_drift_fixture["locs_rec"], real_fitted_drift_fixture["info"],
                plot_results=False, use_temporal_chunking=False,
            )

    def test_postprocess_import_error_raises(self, monkeypatch, real_fitted_drift_fixture):
        monkeypatch.setitem(sys.modules, "pyS3M.postprocess", None)
        dcf = Drift_Correction_Functions()
        with pytest.raises(DriftCorrectionError, match="postprocess module required"):
            dcf.detect_fiducials(
                real_fitted_drift_fixture["locs_rec"], real_fitted_drift_fixture["info"],
                plot_results=False, use_temporal_chunking=False,
            )

    def test_min_frames_fraction_too_strict_raises(self, real_fitted_drift_fixture):
        dcf = Drift_Correction_Functions()
        with pytest.raises(DriftCorrectionError, match="No fiducials found with minimum"):
            dcf.detect_fiducials(
                real_fitted_drift_fixture["locs_rec"], real_fitted_drift_fixture["info"],
                plot_results=False, use_temporal_chunking=False,
                min_frames_fraction=1.0,
            )

    def test_non_chunked_zero_picks_raises(self, real_fitted_drift_fixture):
        dcf = Drift_Correction_Functions()
        with pytest.raises(DriftCorrectionError, match="No fiducial candidates detected"):
            dcf.detect_fiducials(
                real_fitted_drift_fixture["locs_rec"], real_fitted_drift_fixture["info"],
                plot_results=False, use_temporal_chunking=False,
                threshold_percentile=100.0,
            )

    def test_chunking_import_error_raises(self, monkeypatch, real_fitted_drift_fixture):
        monkeypatch.setitem(sys.modules, "pyS3M.localise", None)
        dcf = Drift_Correction_Functions()
        with pytest.raises(
            DriftCorrectionError, match="localise and render modules required for chunked"
        ):
            dcf.detect_fiducials(
                real_fitted_drift_fixture["locs_rec"], real_fitted_drift_fixture["info"],
                plot_results=False, use_temporal_chunking=True,
            )

    def test_chunking_skips_empty_chunks(self, real_fitted_drift_fixture):
        """A huge n_chunks (more chunks than there is meaningful frame
        spread) leaves many chunks with zero localisations -- exercises the
        per-chunk `if len(chunk_locs) == 0: continue` branch."""
        dcf = Drift_Correction_Functions()
        with pytest.raises(DriftCorrectionError):
            # Still expected to eventually fail (too few candidates survive
            # such fine chunking) -- the empty-chunk branch is what's under
            # test here, not a successful detection.
            dcf.detect_fiducials(
                real_fitted_drift_fixture["locs_rec"], real_fitted_drift_fixture["info"],
                plot_results=False, use_temporal_chunking=True, n_chunks=350,
            )

    def test_chunking_per_chunk_detection_failure_is_caught_and_continues(
        self, monkeypatch, real_fitted_drift_fixture
    ):
        """One chunk's identify_in_image raising is caught, logged, and
        skipped -- detection still succeeds using the remaining chunks."""
        import pyS3M.localise as localise_mod

        orig = localise_mod.identify_in_image
        calls = {"n": 0}

        def flaky(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("synthetic per-chunk failure")
            return orig(*args, **kwargs)

        monkeypatch.setattr(localise_mod, "identify_in_image", flaky)
        dcf = Drift_Correction_Functions()
        result = dcf.detect_fiducials(
            real_fitted_drift_fixture["locs_rec"], real_fitted_drift_fixture["info"],
            plot_results=False, use_temporal_chunking=True, n_chunks=5,
        )
        assert result.n_fiducials > 0
        assert calls["n"] >= 2

    def test_chunking_no_candidates_in_any_chunk_raises(self, monkeypatch, real_fitted_drift_fixture):
        import pyS3M.localise as localise_mod

        def always_fails(*args, **kwargs):
            raise RuntimeError("synthetic failure")

        monkeypatch.setattr(localise_mod, "identify_in_image", always_fails)
        dcf = Drift_Correction_Functions()
        with pytest.raises(DriftCorrectionError, match="No candidates found in any temporal chunk"):
            dcf.detect_fiducials(
                real_fitted_drift_fixture["locs_rec"], real_fitted_drift_fixture["info"],
                plot_results=False, use_temporal_chunking=True, n_chunks=5,
            )

    def test_unexpected_exception_is_wrapped(self, monkeypatch, real_fitted_drift_fixture):
        """Any non-DriftCorrectionError raised inside the try block is
        re-wrapped as a DriftCorrectionError (the final except clause) --
        postprocess.picked_locs is inside that try, unlike extract_metadata
        (called before the try even starts)."""
        import pyS3M.postprocess as postprocess_mod

        def _boom(*args, **kwargs):
            raise KeyError("synthetic failure")

        monkeypatch.setattr(postprocess_mod, "picked_locs", _boom)
        dcf = Drift_Correction_Functions()
        with pytest.raises(DriftCorrectionError, match="Fiducial detection failed"):
            dcf.detect_fiducials(
                real_fitted_drift_fixture["locs_rec"], real_fitted_drift_fixture["info"],
                use_temporal_chunking=False,
            )


class TestUndriftWithFiducialDetection:
    def test_render_none_raises(self, monkeypatch, real_fitted_drift_fixture):
        import pyS3M.drift_correction._facade as facade_mod

        monkeypatch.setattr(facade_mod, "render", None)
        dcf = Drift_Correction_Functions()
        with pytest.raises(DriftCorrectionError, match="render module required"):
            dcf.undrift_with_fiducial_detection(
                real_fitted_drift_fixture["locs_rec"], real_fitted_drift_fixture["info"]
            )

    def test_full_pipeline_succeeds(self, real_fitted_drift_fixture):
        dcf = Drift_Correction_Functions()
        result = dcf.undrift_with_fiducial_detection(
            real_fitted_drift_fixture["locs_rec"], real_fitted_drift_fixture["info"],
            create_plots=False,
        )
        assert result.metadata["detection_method"] == "automatic"
        assert result.metadata["n_fiducials_validated"] > 0
        assert np.all(np.isfinite(result.drift_x))

    def test_no_regions_detected_raises(self, real_fitted_drift_fixture):
        dcf = Drift_Correction_Functions()
        with pytest.raises(DriftCorrectionError, match="No fiducial regions detected"):
            dcf.undrift_with_fiducial_detection(
                real_fitted_drift_fixture["locs_rec"], real_fitted_drift_fixture["info"],
                threshold_percentile=100.0, create_plots=False,
            )

    def test_no_regions_selected_raises(self, real_fitted_drift_fixture):
        dcf = Drift_Correction_Functions()
        with pytest.raises(DriftCorrectionError, match="No valid fiducials with"):
            dcf.undrift_with_fiducial_detection(
                real_fitted_drift_fixture["locs_rec"], real_fitted_drift_fixture["info"],
                min_localisations_per_region=10**9, create_plots=False,
            )

    def test_no_fiducials_validated_raises(self, real_fitted_drift_fixture):
        dcf = Drift_Correction_Functions()
        with pytest.raises(DriftCorrectionError, match="No fiducials passed validation"):
            dcf.undrift_with_fiducial_detection(
                real_fitted_drift_fixture["locs_rec"], real_fitted_drift_fixture["info"],
                retention_percentage=1e-9, create_plots=False,
            )

    def test_with_plots(self, real_fitted_drift_fixture, tmp_path):
        dcf = Drift_Correction_Functions()
        result = dcf.undrift_with_fiducial_detection(
            real_fitted_drift_fixture["locs_rec"], real_fitted_drift_fixture["info"],
            create_plots=True, output_dir=str(tmp_path),
        )
        assert result.metadata["n_fiducials_validated"] > 0
