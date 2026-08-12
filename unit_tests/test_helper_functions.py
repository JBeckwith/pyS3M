"""Full coverage tests for pyS3M.HelperFunctions -- ROI geometry, parallel-chunk
distribution, metadata-fallback file search, and elapsed-time formatting helpers.
"""
import types

import numpy as np
import pytest

import pyS3M.HelperFunctions as HelperFunctions


@pytest.fixture
def helper():
    return HelperFunctions.Helper_Functions()


class TestFileSearch:
    def test_finds_and_sorts_alphanumerically(self, helper, tmp_path):
        for name in ["scan_10.txt", "scan_2.txt", "scan_1.txt"]:
            (tmp_path / name).write_text("x")
        (tmp_path / "unrelated.dat").write_text("x")
        results = helper.file_search(tmp_path, "scan", ".txt")
        names = [p.split("/")[-1] for p in results]
        assert names == ["scan_1.txt", "scan_2.txt", "scan_10.txt"]

    def test_second_string_filters_results(self, helper, tmp_path):
        (tmp_path / "metadata_a.json").write_text("x")
        (tmp_path / "metadata_b.txt").write_text("x")
        results = helper.file_search(tmp_path, "metadata", ".json")
        assert len(results) == 1
        assert results[0].endswith("metadata_a.json")

    def test_no_matches_returns_empty(self, helper, tmp_path):
        assert helper.file_search(tmp_path, "nope", "") == []


class TestCropCalibrationMaps:
    def test_crops_all_maps_consistently(self, helper):
        maps = {
            "gain": np.arange(100, dtype=np.float32).reshape(10, 10),
            "offset": np.arange(100, dtype=np.float32).reshape(10, 10) * 2,
        }
        cropped = helper.crop_calibration_maps(maps, start_x=2, start_y=3, width=4, height=5)
        assert cropped["gain"].shape == (5, 4)
        np.testing.assert_array_equal(cropped["gain"], maps["gain"][3:8, 2:6])
        np.testing.assert_array_equal(cropped["offset"], maps["offset"][3:8, 2:6])


class TestCalculateRoiBounds:
    def test_valid_centered_roi(self, helper):
        bounds = helper.calculate_roi_bounds(50.0, 50.0, roi_size=10, width=100, height=100)
        assert bounds == (45, 55, 45, 55)

    def test_non_square_roi_returns_none(self, helper):
        # Asymmetric near-edge position clamps to a non-square region.
        bounds = helper.calculate_roi_bounds(2.0, 50.0, roi_size=20, width=100, height=100)
        assert bounds is None

    def test_below_min_size_returns_none(self, helper):
        bounds = helper.calculate_roi_bounds(0.0, 0.0, roi_size=2, width=100, height=100, min_roi_size=4)
        assert bounds is None


class TestCalculateParallelChunks:
    def test_basic_distribution(self, helper):
        n_workers, n_tasks, items_per_task, start_indices = helper.calculate_parallel_chunks(1000)
        assert sum(items_per_task) == 1000
        assert len(items_per_task) == n_tasks
        assert start_indices[0] == 0

    def test_small_total_items_caps_tasks(self, helper):
        n_workers, n_tasks, items_per_task, start_indices = helper.calculate_parallel_chunks(3)
        assert n_tasks <= 3
        assert sum(items_per_task) == 3


class TestLoadMetadataRoi:
    def test_finds_metadata_and_delegates_to_io(self, helper, tmp_path):
        (tmp_path / "img_metadata.json").write_text("{}")

        io_stub = types.SimpleNamespace(
            metadata_reader_imageJ=lambda path: (5, 10, 40, 30),
        )
        result = helper.load_metadata_roi(tmp_path, io_stub)
        assert result == (5, 10, 40, 30)

    def test_no_metadata_with_fallback_returns_defaults(self, helper, tmp_path):
        result = helper.load_metadata_roi(tmp_path, io_functions=None, use_fallback=True)
        assert result == (0, 0, None, None)

    def test_no_metadata_without_fallback_raises(self, helper, tmp_path):
        with pytest.raises(FileNotFoundError, match="No metadata files found"):
            helper.load_metadata_roi(tmp_path, io_functions=None, use_fallback=False)


class TestFormatElapsedTime:
    def test_seconds(self, helper):
        value, unit = helper.format_elapsed_time(45.3)
        assert unit == "s"
        assert value == pytest.approx(45.3)

    def test_minutes(self, helper):
        value, unit = helper.format_elapsed_time(180.0)
        assert unit == "min"
        assert value == pytest.approx(3.0)

    def test_hours(self, helper):
        value, unit = helper.format_elapsed_time(7320.0)
        assert unit == "hours"
        assert value == pytest.approx(2.0333, abs=1e-3)
