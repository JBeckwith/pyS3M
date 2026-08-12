#!/usr/bin/env python3
"""
Full coverage tests for pyS3M.FiducialDetection -- FiducialDetector (detection/
selection/validation logic) and DriftPlotter (its plotting utilities).

Small synthetic data throughout -- no fixture needed, nothing here requires
real acquisition data.
"""
from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from pyS3M.FiducialDetection import FiducialDetector, DriftPlotter
from pyS3M.Constants import AnalysisConfig
from pyS3M.drift_correction._base import FiducialDetectionResult
import pyS3M.FiducialDetection as fd_mod


def _locs(n=50, seed=0, with_photons=True, with_frame=True):
    rng = np.random.default_rng(seed)
    arrays = [rng.uniform(0, 200, n), rng.uniform(0, 200, n)]
    names = ["xc", "yc"]
    if with_frame:
        arrays.append(rng.integers(0, 50, n))
        names.append("frame")
    if with_photons:
        arrays.append(rng.uniform(1000, 5000, n))
        names.append("photons")
    return np.rec.fromarrays(arrays, names=names)


def _puncta(n, cx, cy, spread=0.5, seed=0):
    rng = np.random.default_rng(seed)
    return np.rec.fromarrays(
        [cx + rng.normal(0, spread, n), cy + rng.normal(0, spread, n)],
        names=["xc", "yc"],
    )


# ======================================================================
# Module-level scipy import fallback
# ======================================================================

class TestModuleImportFallback:
    def test_scipy_import_failure_fallback(self, monkeypatch):
        import importlib

        monkeypatch.setitem(sys.modules, "scipy", None)
        monkeypatch.setitem(sys.modules, "scipy.optimize", None)
        try:
            with pytest.warns(UserWarning, match="scipy not available"):
                importlib.reload(fd_mod)
            assert fd_mod.ndimage is None
            assert fd_mod.curve_fit is None
        finally:
            monkeypatch.undo()
            importlib.reload(fd_mod)


# ======================================================================
# FiducialDetector.detect_high_density_regions_from_image
# ======================================================================

class TestDetectHighDensityRegionsFromImage:
    def test_empty_image_raises(self):
        fd = FiducialDetector()
        with pytest.raises(ValueError, match="no non-zero values"):
            fd.detect_high_density_regions_from_image(np.zeros((10, 10)), create_plot=False)

    def test_ndimage_none_raises(self, monkeypatch):
        monkeypatch.setattr(fd_mod, "ndimage", None)
        fd = FiducialDetector()
        img = np.random.default_rng(0).uniform(0, 1, (10, 10))
        with pytest.raises(RuntimeError, match="scipy.ndimage required"):
            fd.detect_high_density_regions_from_image(img, create_plot=False)

    def test_success_with_plot(self):
        fd = FiducialDetector()
        rng = np.random.default_rng(0)
        img = rng.uniform(0, 1, (30, 30))
        img[5:10, 5:10] = 100
        centres, mask, threshold, meta = fd.detect_high_density_regions_from_image(
            img, threshold_percentile=97, create_plot=True
        )
        assert meta["n_regions_detected"] >= 1


# ======================================================================
# FiducialDetector.select_puncta_from_regions
# ======================================================================

class TestSelectPunctaFromRegions:
    def test_postprocess_unavailable_raises(self, monkeypatch):
        monkeypatch.setattr(fd_mod, "postprocess", None)
        monkeypatch.setattr(fd_mod, "_ensure_postprocess", lambda: None)
        fd = FiducialDetector()
        with pytest.raises(RuntimeError, match="postprocess module not available"):
            fd.select_puncta_from_regions(locs=None, region_centres=[(1, 1)], binary_mask=None, create_plot=False)

    def test_empty_region_centres(self):
        fd = FiducialDetector()
        selected, meta = fd.select_puncta_from_regions(locs=None, region_centres=[], binary_mask=None, create_plot=False)
        assert selected == []
        assert meta["n_regions_selected"] == 0

    def test_success_with_rejection_and_plot(self):
        fd = FiducialDetector()
        locs = _locs(n=100, seed=1)
        # region 0 centred on a real point (should be selected); region 1 far
        # away in empty space (should be rejected -- too few localisations).
        region_centres = [(float(locs.yc[0]), float(locs.xc[0])), (5000.0, 5000.0)]
        binary_mask = np.zeros((250, 250), dtype=bool)
        selected, meta = fd.select_puncta_from_regions(
            locs, region_centres, binary_mask,
            selection_box_size_nm=5000, pixelsize=1.0,
            min_localisations_per_region=1, create_plot=True,
        )
        assert meta["n_regions_selected"] >= 1
        assert meta["n_regions_rejected"] >= 1

    def test_picked_locs_none_fallback(self, monkeypatch):
        fd = FiducialDetector()
        monkeypatch.setattr(fd_mod, "postprocess", type("PP", (), {"picked_locs": staticmethod(lambda **kw: None)}))
        monkeypatch.setattr(fd_mod, "_ensure_postprocess", lambda: fd_mod.postprocess)
        locs = _locs(n=20, seed=2)
        selected, meta = fd.select_puncta_from_regions(
            locs, [(1.0, 1.0)], np.zeros((10, 10), dtype=bool), create_plot=False
        )
        assert selected == []

    def test_progress_callback_at_region_100(self):
        """region_id % 100 == 0 and region_id > 0 -- needs >100 regions."""
        fd_calls = []
        cfg = AnalysisConfig(
            logging_callback=lambda m: fd_calls.append(("log", m)),
            progress_callback=lambda f, m: fd_calls.append(("prog", f, m)),
        )
        fd = FiducialDetector(config=cfg)
        rng = np.random.default_rng(3)
        n_locs = 300
        locs = np.rec.fromarrays(
            [rng.uniform(0, 200, n_locs), rng.uniform(0, 200, n_locs),
             rng.integers(0, 50, n_locs), rng.uniform(1000, 5000, n_locs)],
            names=["xc", "yc", "frame", "photons"],
        )
        region_centres = [(float(locs.yc[i]), float(locs.xc[i])) for i in range(101)]
        selected, meta = fd.select_puncta_from_regions(
            locs, region_centres, np.zeros((250, 250), dtype=bool),
            selection_box_size_nm=50000, pixelsize=1.0,
            min_localisations_per_region=1, create_plot=False,
        )
        assert any(c[0] == "log" and "Processed" in c[1] for c in fd_calls)
        assert any(c[0] == "prog" for c in fd_calls)


# ======================================================================
# FiducialDetector.remove_puncta_locs
# ======================================================================

class TestRemovePunctaLocs:
    def test_empty_selected_puncta_returns_locs_unchanged(self):
        fd = FiducialDetector()
        locs = _locs(n=5, with_photons=False)
        out = fd.remove_puncta_locs(locs, [])
        assert out is locs

    def test_removes_matching_rows(self):
        fd = FiducialDetector()
        locs = np.rec.fromarrays(
            [np.array([1, 2, 3]), np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])],
            names=["frame", "xc", "yc"],
        )
        puncta = locs[[0]].copy()
        cfg = AnalysisConfig(logging_callback=lambda m: None)
        fd2 = FiducialDetector(config=cfg)
        out = fd2.remove_puncta_locs(locs, [puncta])
        assert len(out) == 2
        assert 1 not in out.frame


# ======================================================================
# FiducialDetector.identify_real_fiducials_with_clustering
# ======================================================================

class TestIdentifyRealFiducialsWithClustering:
    def test_progress_callback_and_success(self):
        calls = []
        cfg = AnalysisConfig(progress_callback=lambda f, m: calls.append((f, m)))
        fd = FiducialDetector(config=cfg)
        puncta = _puncta(20, 10, 10)
        validated, meta = fd.identify_real_fiducials_with_clustering([puncta], create_plot=True)
        assert len(validated) == 1
        assert len(calls) == 1

    def test_empty_puncta_region_skipped(self):
        fd = FiducialDetector()
        empty = np.recarray((0,), dtype=[("xc", "f8"), ("yc", "f8")])
        validated, meta = fd.identify_real_fiducials_with_clustering([empty], create_plot=False)
        assert validated == []

    def test_n_keep_below_min_samples_skipped(self):
        fd = FiducialDetector()
        puncta = _puncta(20, 10, 10)
        validated, meta = fd.identify_real_fiducials_with_clustering(
            [puncta], retention_percentage=0.5, min_samples_factor=0.7, create_plot=False
        )
        assert validated == []

    def test_exception_during_fit_is_caught(self, monkeypatch):
        fd = FiducialDetector()
        puncta = _puncta(20, 10, 10)
        monkeypatch.setattr(np, "std", lambda *a, **k: (_ for _ in ()).throw(ValueError("boom")))
        with pytest.warns(UserWarning, match="Failed to validate region"):
            validated, meta = fd.identify_real_fiducials_with_clustering([puncta], create_plot=False)
        assert validated == []


# ======================================================================
# FiducialDetector's private _plot_* wrappers
# ======================================================================

class TestPrivatePlotWrappers:
    @pytest.mark.parametrize("method_name,args", [
        ("_plot_density_detection_results",
         (np.zeros((5, 5)), np.zeros((5, 5), dtype=bool), [], np.array([1]), np.array([0, 1]), 0.5, 100.0, None, "t")),
        ("_plot_puncta_selection_results",
         (_locs(5, with_photons=False, with_frame=False), [], [], np.zeros((5, 5), dtype=bool), [], 5.0, 100.0, None, "t", True, 1000)),
        ("_plot_clustering_results",
         ([], [], [], None, "t")),
    ])
    def test_is_available_false_returns_early(self, monkeypatch, method_name, args):
        fd = FiducialDetector()
        monkeypatch.setattr(fd_mod, "is_available", lambda name: False)
        getattr(fd, method_name)(*args)  # should just return, no error

    @pytest.mark.parametrize("method_name,args,match", [
        ("_plot_density_detection_results",
         (np.zeros((5, 5)), np.zeros((5, 5), dtype=bool), [], np.array([1]), np.array([0, 1]), 0.5, 100.0, None, "t"),
         "Failed to create density detection plots"),
        ("_plot_puncta_selection_results",
         (_locs(5, with_photons=False, with_frame=False), [], [], np.zeros((5, 5), dtype=bool), [], 5.0, 100.0, None, "t", True, 1000),
         "Failed to create puncta selection plots"),
        ("_plot_clustering_results",
         ([], [], [], None, "t"),
         "Failed to create clustering plots"),
    ])
    def test_exception_in_plotter_is_caught(self, monkeypatch, method_name, args, match):
        fd = FiducialDetector()
        monkeypatch.setattr(fd_mod, "DriftPlotter", lambda config=None: (_ for _ in ()).throw(RuntimeError("boom")))
        with pytest.warns(UserWarning, match=match):
            getattr(fd, method_name)(*args)

    def test_density_detection_success_path(self):
        fd = FiducialDetector()
        fd._plot_density_detection_results(
            np.random.default_rng(0).uniform(0, 1, (10, 10)),
            np.zeros((10, 10), dtype=bool), [(1, 1)],
            np.array([1, 2]), np.array([0, 1, 2]), 0.5, 100.0, None, "t",
        )

    def test_puncta_selection_success_path(self):
        fd = FiducialDetector()
        locs = _locs(10, with_photons=False, with_frame=False)
        fd._plot_puncta_selection_results(
            locs, [locs[[0]]], [(1, 1)], np.zeros((10, 10), dtype=bool),
            [{"centre_y": 1, "centre_x": 1, "n_localisations": 1}], 5.0, 100.0, None, "t", True, 1000,
        )

    def test_clustering_results_success_path(self):
        fd = FiducialDetector()
        puncta = [_puncta(5, 1, 1)]
        fd._plot_clustering_results(puncta, puncta, [{"region_id": 0}], None, "t")


# ======================================================================
# DriftPlotter.plot_fiducial_detection_steps
# ======================================================================

class TestPlotFiducialDetectionSteps:
    def _result(self, n=30, seed=0, empty=False):
        if empty:
            return FiducialDetectionResult(
                picks=[], picked_localisations=[],
                detection_image=np.random.default_rng(0).uniform(0, 1, (20, 20)),
                locs_with_groups=np.recarray((0,), dtype=[("xc", "f8"), ("yc", "f8"), ("group", "i4")]),
                n_fiducials=0, detection_params={}, metadata={},
            )
        rng = np.random.default_rng(seed)
        group = rng.integers(0, 3, n)
        locs = np.rec.fromarrays([rng.uniform(0, 10, n), rng.uniform(0, 10, n), group], names=["xc", "yc", "group"])
        return FiducialDetectionResult(
            picks=[(1.0, 2.0), (3.0, 4.0)], picked_localisations=[locs],
            detection_image=rng.uniform(0, 1, (20, 20)),
            locs_with_groups=locs, n_fiducials=3, detection_params={},
            metadata={"total_localisations": n, "threshold_used": 0.7},
        )

    def test_empty_everything(self):
        dp = DriftPlotter()
        dp.plot_fiducial_detection_steps(
            np.random.default_rng(0).uniform(0, 1, (20, 20)), None, 0.5,
            [], [], self._result(empty=True), [{"Pixelsize": 0.069}], save_path=None,
        )

    def test_full_with_hist_and_groups(self):
        dp = DriftPlotter()
        hist = (np.array([1, 2, 3]), np.array([0, 1, 2, 3]))
        dp.plot_fiducial_detection_steps(
            np.random.default_rng(0).uniform(0, 1, (20, 20)), hist, 0.5,
            [(1, 2), (3, 4)], [(1, 2)], self._result(), [{"Pixelsize": 0.069}], save_path=None,
        )

    def test_group_plotting_exception_is_caught_inline(self):
        """The ax4 group-plotting block has its own inner try/except (distinct
        from the method's outer one) -- force it via a locs_with_groups whose
        `.group` access raises, and confirm the method still completes
        (falls through to ax4.text(...) with the error message) rather than
        propagating or hitting the outer except."""
        dp = DriftPlotter()
        result = self._result()

        class _BadGroupLocs:
            def __len__(self):
                return 10

            @property
            def group(self):
                raise RuntimeError("bad group field")

        result.locs_with_groups = _BadGroupLocs()
        dp.plot_fiducial_detection_steps(
            np.random.default_rng(0).uniform(0, 1, (20, 20)), None, 0.5,
            [], [], result, [{"Pixelsize": 0.069}], save_path=None,
        )

    def test_outer_exception_is_caught(self, monkeypatch):
        dp = DriftPlotter()
        monkeypatch.setattr(dp, "two_column_plot", lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")))
        # Should not raise -- caught by the method's own except Exception.
        dp.plot_fiducial_detection_steps(
            np.random.default_rng(0).uniform(0, 1, (20, 20)), None, 0.5,
            [], [], self._result(empty=True), [{"Pixelsize": 0.069}], save_path=None,
        )


# ======================================================================
# DriftPlotter.plot_fiducial_detection_results
# ======================================================================

class TestPlotFiducialDetectionResults:
    def _result_and_info(self, n_group_locs=0):
        rng = np.random.default_rng(0)
        n = 30
        group = rng.integers(0, 3, n) if n_group_locs else np.full(n, -1)
        locs = np.rec.fromarrays([rng.uniform(0, 10, n), rng.uniform(0, 10, n), group], names=["xc", "yc", "group"])
        result = FiducialDetectionResult(
            picks=[(1.0, 2.0), (3.0, 4.0)], picked_localisations=[locs],
            detection_image=rng.uniform(0, 1, (20, 20)),
            locs_with_groups=locs, n_fiducials=2,
            detection_params={"threshold_percentile": 99.0, "box_size_nm": 900.0, "min_frames_fraction": 0.8},
            metadata={"total_candidates": 5, "threshold_used": 0.5},
        )
        info = [{"Width": 20, "Height": 20, "Frames": 10, "Pixelsize": 69.0}]
        return result, info

    def test_success_with_fiducial_locs(self):
        dp = DriftPlotter()
        result, info = self._result_and_info(n_group_locs=1)
        dp.plot_fiducial_detection_results(result, info, save_path=None)

    def test_success_no_fiducial_locs(self):
        dp = DriftPlotter()
        result, info = self._result_and_info(n_group_locs=0)
        dp.plot_fiducial_detection_results(result, info, save_path=None)

    def test_exception_is_caught(self):
        """Missing required metadata (Width/Height/Frames) makes
        extract_metadata raise -- exercises the except branch."""
        dp = DriftPlotter()
        result, _ = self._result_and_info()
        dp.plot_fiducial_detection_results(result, [{"Pixelsize": 69.0}], save_path=None)


# ======================================================================
# DriftPlotter.plot_region_data_with_datashader
# ======================================================================

class TestPlotRegionDataWithDatashader:
    def test_single_dataset(self):
        dp = DriftPlotter()
        fig, ax = dp.one_column_plot(width=3.5, height=3.5)
        rng = np.random.default_rng(0)
        data_list = [{"xc": rng.uniform(0, 10, 20), "yc": rng.uniform(0, 10, 20)}]
        dp.plot_region_data_with_datashader(ax, data_list, ["red"], "t")

    def test_multi_dataset(self):
        dp = DriftPlotter()
        fig, ax = dp.one_column_plot(width=3.5, height=3.5)
        rng = np.random.default_rng(0)
        data_list = [
            {"xc": rng.uniform(0, 10, 20), "yc": rng.uniform(0, 10, 20)},
            {"xc": rng.uniform(0, 10, 20), "yc": rng.uniform(0, 10, 20)},
        ]
        dp.plot_region_data_with_datashader(ax, data_list, ["red", "blue"], "t")

    def test_no_colors(self):
        dp = DriftPlotter()
        fig, ax = dp.one_column_plot(width=3.5, height=3.5)
        rng = np.random.default_rng(0)
        data_list = [{"xc": rng.uniform(0, 10, 20), "yc": rng.uniform(0, 10, 20)}]
        dp.plot_region_data_with_datashader(ax, data_list, [], "t")


# ======================================================================
# DriftPlotter.plot_clustering_overlay
# ======================================================================

class TestPlotClusteringOverlay:
    def test_small_dataset_both_types(self):
        dp = DriftPlotter()
        fig, ax = dp.one_column_plot(width=3.5, height=3.5)
        rng = np.random.default_rng(0)
        n = 50
        all_x, all_y = rng.uniform(0, 10, n), rng.uniform(0, 10, n)
        types = ["original"] * 25 + ["validated"] * 25
        dp.plot_clustering_overlay(ax, all_x, all_y, types, "t")

    def test_large_dataset_datashader_branch(self):
        dp = DriftPlotter()
        fig, ax = dp.one_column_plot(width=3.5, height=3.5)
        rng = np.random.default_rng(0)
        n = 10001
        all_x, all_y = rng.uniform(0, 10, n), rng.uniform(0, 10, n)
        types = ["original"] * 5001 + ["validated"] * 5000
        dp.plot_clustering_overlay(ax, all_x, all_y, types, "t")

    def test_only_original_type(self):
        dp = DriftPlotter()
        fig, ax = dp.one_column_plot(width=3.5, height=3.5)
        rng = np.random.default_rng(0)
        n = 10
        all_x, all_y = rng.uniform(0, 10, n), rng.uniform(0, 10, n)
        types = ["original"] * n
        dp.plot_clustering_overlay(ax, all_x, all_y, types, "t")


# ======================================================================
# DriftPlotter.plot_puncta_selection_results
# ======================================================================

class TestPlotPunctaSelectionResults:
    def _locs_recarray(self, n, seed=0):
        rng = np.random.default_rng(seed)
        return np.rec.fromarrays([rng.uniform(0, 10, n), rng.uniform(0, 10, n)], names=["xc", "yc"])

    def test_small_dataset_plain_scatter(self):
        dp = DriftPlotter()
        all_locs = self._locs_recarray(50)
        selected = [self._locs_recarray(5, seed=1)]
        stats = [{"centre_y": 1, "centre_x": 1, "n_localisations": 5}]
        dp.plot_puncta_selection_results(
            all_locs, selected, [(1, 1)], np.zeros((20, 20), dtype=bool), stats,
            5.0, 100.0, None, "t", use_datashader_threshold=10000,
        )

    def test_large_dataset_preview_branch(self):
        dp = DriftPlotter()
        all_locs = self._locs_recarray(2000)
        dp.plot_puncta_selection_results(
            all_locs, [], [], np.zeros((20, 20), dtype=bool), [],
            5.0, 100.0, None, "t", use_datashader_threshold=1000,
        )

    def test_no_region_centres_or_stats(self):
        dp = DriftPlotter()
        all_locs = self._locs_recarray(10)
        dp.plot_puncta_selection_results(
            all_locs, [], [], np.zeros((20, 20), dtype=bool), [],
            5.0, 100.0, "/tmp/out.png", "t",
        )

    def test_exception_is_caught(self, monkeypatch):
        dp = DriftPlotter()
        monkeypatch.setattr(dp, "two_column_plot", lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")))
        dp.plot_puncta_selection_results(
            self._locs_recarray(5), [], [], np.zeros((5, 5), dtype=bool), [],
            5.0, 100.0, None, "t",
        )


# ======================================================================
# DriftPlotter.plot_individual_clustering_details
# ======================================================================

class TestPlotIndividualClusteringDetails:
    def _puncta_list(self, n, count, seed=0):
        rng = np.random.default_rng(seed)
        return [np.rec.fromarrays([rng.uniform(0, 10, n), rng.uniform(0, 10, n)], names=["xc", "yc"]) for _ in range(count)]

    def test_no_validated_fiducials_warns_and_returns(self):
        dp = DriftPlotter()
        dp.plot_individual_clustering_details([], [], [], "/tmp/base", "t")

    def test_single_validated_fiducial(self, tmp_path):
        dp = DriftPlotter()
        selected = self._puncta_list(10, 1)
        validated = self._puncta_list(5, 1, seed=1)
        dp.plot_individual_clustering_details(
            selected, validated, [{"region_id": 0}], str(tmp_path / "base"), "t"
        )

    def test_multiple_validated_fiducials_multi_row(self, tmp_path):
        dp = DriftPlotter()
        selected = self._puncta_list(10, 4)
        validated = self._puncta_list(5, 4, seed=1)
        meta = [{"region_id": i} for i in range(4)]
        dp.plot_individual_clustering_details(selected, validated, meta, str(tmp_path / "base"), "t")

    def test_exception_is_caught(self, monkeypatch):
        dp = DriftPlotter()
        monkeypatch.setattr(dp, "two_column_plot", lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")))
        dp.plot_individual_clustering_details(
            self._puncta_list(5, 1), self._puncta_list(5, 1), [{"region_id": 0}], "/tmp/base", "t",
        )


# ======================================================================
# DriftPlotter.plot_clustering_results
# ======================================================================

class TestPlotClusteringResults:
    def _puncta_list(self, n, count, seed=0):
        rng = np.random.default_rng(seed)
        return [np.rec.fromarrays([rng.uniform(0, 10, n), rng.uniform(0, 10, n)], names=["xc", "yc"]) for _ in range(count)]

    def test_no_regions_warns_and_returns(self):
        dp = DriftPlotter()
        dp.plot_clustering_results([], [], [], None, "t")

    def test_few_regions_with_legend_and_output_path(self, tmp_path):
        dp = DriftPlotter()
        selected = self._puncta_list(5, 3)
        validated = self._puncta_list(2, 3, seed=1)
        dp.plot_clustering_results(selected, validated, [], str(tmp_path / "out.png"), "t")

    def test_many_regions_no_legend(self):
        dp = DriftPlotter()
        selected = self._puncta_list(5, 12)
        validated = self._puncta_list(2, 12, seed=1)
        dp.plot_clustering_results(selected, validated, [], None, "t")

    def test_some_empty_selected_regions(self):
        """Covers the `else: retention_rates.append(0)` branch for a region
        with zero selected puncta."""
        dp = DriftPlotter()
        empty = np.recarray((0,), dtype=[("xc", "f8"), ("yc", "f8")])
        selected = [empty, self._puncta_list(5, 1)[0]]
        validated = [empty, self._puncta_list(2, 1, seed=1)[0]]
        dp.plot_clustering_results(selected, validated, [], None, "t")

    def test_exception_is_caught(self, monkeypatch):
        dp = DriftPlotter()
        monkeypatch.setattr(dp, "two_column_plot", lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")))
        dp.plot_clustering_results(self._puncta_list(5, 1), self._puncta_list(2, 1), [], None, "t")


# ======================================================================
# DriftPlotter.plot_clustering_summary_only
# ======================================================================

class TestPlotClusteringSummaryOnly:
    def _puncta_list(self, n, count, seed=0):
        rng = np.random.default_rng(seed)
        return [np.rec.fromarrays([rng.uniform(0, 10, n), rng.uniform(0, 10, n)], names=["xc", "yc"]) for _ in range(count)]

    def test_no_regions_warns_and_returns(self):
        dp = DriftPlotter()
        dp.plot_clustering_summary_only([], [], [], None, "t")

    def test_polyfit_branch_with_output_path(self, tmp_path):
        """len(n_locs) > 3 triggers the polyfit trend-line branch."""
        dp = DriftPlotter()
        selected = self._puncta_list(5, 5)
        validated = self._puncta_list(2, 5, seed=1)
        dp.plot_clustering_summary_only(selected, validated, [], str(tmp_path / "out.png"), "t")

    def test_few_regions_no_polyfit_no_output_path(self):
        dp = DriftPlotter()
        selected = self._puncta_list(5, 2)
        validated = self._puncta_list(2, 2, seed=1)
        dp.plot_clustering_summary_only(selected, validated, [], None, "t")

    def test_exception_is_caught(self, monkeypatch):
        dp = DriftPlotter()
        monkeypatch.setattr(dp, "two_column_plot", lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")))
        dp.plot_clustering_summary_only(self._puncta_list(5, 1), self._puncta_list(2, 1), [], None, "t")


# ======================================================================
# DriftPlotter.create_separate_plots
# ======================================================================

class TestCreateSeparatePlots:
    def test_success_with_hist_and_output_path(self, tmp_path):
        dp = DriftPlotter()
        rng = np.random.default_rng(0)
        img = rng.uniform(0, 1, (20, 20))
        mask = np.zeros((20, 20), dtype=bool)
        hist, bin_edges = np.histogram(img.flatten(), bins=10)
        dp.create_separate_plots(
            img, mask, [(5, 5)], hist, bin_edges, 0.5, 100.0, str(tmp_path / "out.png"), "t"
        )

    def test_empty_hist_no_output_path(self):
        dp = DriftPlotter()
        rng = np.random.default_rng(0)
        img = rng.uniform(0, 1, (20, 20))
        mask = np.zeros((20, 20), dtype=bool)
        dp.create_separate_plots(img, mask, [], np.array([]), np.array([]), 0.5, 100.0, None, "t")

    def test_exception_is_caught(self, monkeypatch):
        dp = DriftPlotter()
        monkeypatch.setattr(dp, "two_column_plot", lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")))
        dp.create_separate_plots(
            np.zeros((5, 5)), np.zeros((5, 5), dtype=bool), [], np.array([]), np.array([]),
            0.5, 100.0, None, "t",
        )
