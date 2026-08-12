"""Full coverage tests for pyS3M.CalibrationFunctions -- gain/offset/variance/
read-noise/relative-QE camera calibration from flat-field + dark TIFF stacks.

Complements the existing `test_calibration_improvements.py` (vectorised gain,
combined offset+variance, chunked reads -- already well covered there) with
the rest of the file: `__init__` dependency injection, `filesearch`/
`_discover_intensity_strings`, the full `calibrate_multicolour_camera`
orchestration (rgb and nir modes, both happy-path and validation-error
branches), `calculate_rqe`, and `_process_calibration_files`'s
`high_memory=True` path.

Small synthetic TIFF stacks throughout (tiny image sizes, few frames),
written directly via `tifffile` into `tmp_path` directory trees matching the
real folder-naming convention the code expects (colour-named subfolders for
rgb mode, a `dark` folder, `Intensity_<n>_...` filenames).
"""
from __future__ import annotations

import numpy as np
import pytest
import tifffile

import pyS3M.CalibrationFunctions as CalibrationFunctions
import pyS3M.IOFunctions as IOFunctions

RNG = np.random.default_rng(7)


def _write_tiff(path, frames):
    # photometric="minisblack" avoids tifffile guessing RGB/RGBA when the
    # frame axis happens to be length 3 or 4 (matches _write_tiffs in
    # test_calibration_improvements.py).
    tifffile.imwrite(str(path), frames, photometric="minisblack")


def _make_cf(**kwargs):
    io = IOFunctions.IO_Functions()
    return CalibrationFunctions.Calibration_Functions(io_functions=io, **kwargs)


def _make_rgb_calibration_tree(tmp_path, size=16, n_frames=6, n_intensities=2,
                                 offset_mean=100.0, offset_std=5.0,
                                 signal_scale=(50.0, 120.0)):
    """Build a dark/ + B/ + G/ + R/ tree with matching Intensity_NN files."""
    dark_dir = tmp_path / "dark"
    dark_dir.mkdir()
    dark_frames = RNG.normal(offset_mean, offset_std, (n_frames, size, size)).astype(np.float32)
    _write_tiff(dark_dir / "dark_001.tif", dark_frames)

    for colour in ["B", "G", "R"]:
        colour_dir = tmp_path / colour
        colour_dir.mkdir()
        for k in range(n_intensities):
            level_mean = offset_mean + np.linspace(*signal_scale, n_intensities)[k]
            frames = RNG.normal(level_mean, offset_std, (n_frames, size, size)).astype(np.float32)
            _write_tiff(colour_dir / f"Intensity_{k:02d}_001.tif", frames)
    return tmp_path


def _make_nir_calibration_tree(tmp_path, size=16, n_frames=6, n_intensities=2,
                                 offset_mean=100.0, offset_std=5.0,
                                 signal_scale=(50.0, 120.0)):
    dark_dir = tmp_path / "dark"
    dark_dir.mkdir()
    dark_frames = RNG.normal(offset_mean, offset_std, (n_frames, size, size)).astype(np.float32)
    _write_tiff(dark_dir / "dark_001.tif", dark_frames)

    nir_dir = tmp_path / "nir_flat"
    nir_dir.mkdir()
    for k in range(n_intensities):
        level_mean = offset_mean + np.linspace(*signal_scale, n_intensities)[k]
        frames = RNG.normal(level_mean, offset_std, (n_frames, size, size)).astype(np.float32)
        _write_tiff(nir_dir / f"Intensity_{k:02d}_001.tif", frames)
    return tmp_path


# ======================================================================
# __init__
# ======================================================================

class TestInit:
    def test_default_ximea(self):
        cf = _make_cf()
        assert cf.mosaic_unit.shape == (2, 2)
        assert set(np.unique(cf.mosaic_unit)) == {"B", "G", "R"}

    def test_zwo_camera(self):
        cf = _make_cf(camera="zwo")
        assert cf.mosaic_unit.shape == (2, 2)

    def test_explicit_mosaic_unit_overrides_camera(self):
        custom = np.array([["R", "G"], ["G", "B"]])
        cf = _make_cf(mosaic_unit=custom)
        np.testing.assert_array_equal(cf.mosaic_unit, custom)

    def test_dependency_injection(self):
        import types
        fake_io = types.SimpleNamespace()
        fake_mask = types.SimpleNamespace()
        fake_helper = types.SimpleNamespace()
        cf = CalibrationFunctions.Calibration_Functions(
            io_functions=fake_io, mask_functions=fake_mask, helper_functions=fake_helper,
        )
        assert cf.io is fake_io
        assert cf.Mask is fake_mask
        assert cf.helper is fake_helper

    def test_defaults_construct_real_dependencies(self):
        cf = CalibrationFunctions.Calibration_Functions()
        assert cf.io is not None
        assert cf.Mask is not None
        assert cf.helper is not None


# ======================================================================
# filesearch / _discover_intensity_strings
# ======================================================================

class TestFilesearch:
    def test_finds_matching_files(self, tmp_path):
        (tmp_path / "dark_001.tif").write_text("x")
        (tmp_path / "other_002.tif").write_text("x")
        (tmp_path / "dark_notes.txt").write_text("x")
        cf = _make_cf()
        found = cf.filesearch(tmp_path, ".tif", "dark")
        assert list(found) == ["dark_001.tif"]


class TestDiscoverIntensityStrings:
    def test_finds_and_sorts_unique_intensities(self, tmp_path):
        for name in ["Intensity_02_001.tif", "Intensity_00_001.tif", "Intensity_01_001.tif"]:
            (tmp_path / name).write_text("x")
        cf = _make_cf()
        out = cf._discover_intensity_strings(tmp_path, ".tif")
        assert list(out) == ["Intensity_00", "Intensity_01", "Intensity_02"]


# ======================================================================
# calculate_rqe
# ======================================================================

class TestCalculateRqe:
    def test_uniform_image_gives_rqe_near_one(self):
        cf = _make_cf()
        size = 20
        offset = np.full((size, size), 100.0)
        gain = np.full((size, size), 2.0)
        intensity = np.full((size, size), 500.0)
        rqe = cf.calculate_rqe(intensity, offset, gain)
        np.testing.assert_allclose(rqe, 1.0, atol=1e-6)


# ======================================================================
# _process_calibration_files high_memory=True path
# ======================================================================

class TestProcessCalibrationFilesHighMemory:
    def test_matches_low_memory_path(self, tmp_path):
        size, n_frames = 12, 5
        dark_dir = tmp_path / "dark"
        dark_dir.mkdir()
        frames = RNG.normal(100.0, 5.0, (n_frames, size, size)).astype(np.float32)
        _write_tiff(dark_dir / "dark_001.tif", frames)

        cf_low = _make_cf(high_memory=False, chunk_size=3)
        cf_high = _make_cf(high_memory=True)

        offset_low = cf_low.calculate_offset(str(dark_dir), "dark")
        offset_high = cf_high.calculate_offset(str(dark_dir), "dark")

        np.testing.assert_allclose(offset_low, offset_high, rtol=1e-4)
        np.testing.assert_allclose(offset_high, frames.mean(axis=0), rtol=1e-4)

    def test_single_page_file_treated_as_single_frame(self, tmp_path):
        # A genuinely single-page TIFF reads back 2D -- exercises the
        # len(image.shape) == 2 / n_frames == 1 branch directly (no transpose
        # needed, matches process_single_frame_fn's contract as-is).
        dark_dir = tmp_path / "dark"
        dark_dir.mkdir()
        frame = RNG.normal(100.0, 5.0, (10, 10)).astype(np.float32)
        _write_tiff(dark_dir / "dark_001.tif", frame)

        cf_high = _make_cf(high_memory=True)
        offset = cf_high.calculate_offset(str(dark_dir), "dark")
        np.testing.assert_allclose(offset, frame, rtol=1e-5)


# ======================================================================
# calculate_offset_and_variance -- 2D-chunk reshape branch
# ======================================================================

class TestCalculateOffsetAndVarianceSingleFrameChunk:
    def test_chunk_size_one_forces_2d_chunk_reshape(self, tmp_path):
        # chunk_size=1 with a real multi-frame file makes each per-chunk
        # read return a single frame -- read_tiff with a length-1 frame list
        # can come back 2D, exercising the chunk.ndim==2 reshape branch.
        dark_dir = tmp_path / "dark"
        dark_dir.mkdir()
        n_frames, size = 5, 10
        frames = RNG.normal(100.0, 5.0, (n_frames, size, size)).astype(np.float32)
        _write_tiff(dark_dir / "dark_001.tif", frames)

        cf = _make_cf(chunk_size=1)
        offset, variance = cf.calculate_offset_and_variance(str(dark_dir), "dark")
        np.testing.assert_allclose(offset, frames.mean(axis=0), rtol=1e-4)


# ======================================================================
# Directory-name-matches-intensity-string debug-log branch
# ======================================================================

class TestDirectoryNameMatchesIntensityString:
    def test_offset_when_dir_name_equals_intensity_string(self, tmp_path):
        # Path(directory).name == intensity_string exercises the "if" debug
        # log branch in both calculate_offset (_process_calibration_files)
        # and calculate_offset_and_variance -- the existing test suite always
        # uses a tmp_path dir name that differs from the intensity string.
        dark_dir = tmp_path / "dark"
        dark_dir.mkdir()
        frames = RNG.normal(100.0, 5.0, (4, 8, 8)).astype(np.float32)
        _write_tiff(dark_dir / "dark_001.tif", frames)

        cf = _make_cf(chunk_size=2)
        offset = cf.calculate_offset(str(dark_dir), "dark")
        offset2, variance2 = cf.calculate_offset_and_variance(str(dark_dir), "dark")
        np.testing.assert_allclose(offset, frames.mean(axis=0), rtol=1e-4)
        np.testing.assert_allclose(offset2, frames.mean(axis=0), rtol=1e-4)


# ======================================================================
# calibrate_multicolour_camera
# ======================================================================

class TestCalibrateMulticolourCameraValidation:
    def test_wrong_number_of_dark_folders_returns_none(self, tmp_path):
        (tmp_path / "dark1").mkdir()
        (tmp_path / "dark2").mkdir()
        cf = _make_cf()
        result = cf.calibrate_multicolour_camera(str(tmp_path))
        assert result is None

    def test_no_dark_folder_returns_none(self, tmp_path):
        (tmp_path / "B").mkdir()
        cf = _make_cf()
        result = cf.calibrate_multicolour_camera(str(tmp_path))
        assert result is None

    def test_nir_wrong_number_of_flat_folders_returns_none(self, tmp_path):
        # Dark offset/variance is computed before the NIR-folder-count check,
        # so the dark folder needs a real, readable TIFF.
        dark_dir = tmp_path / "dark"
        dark_dir.mkdir()
        frames = RNG.normal(100.0, 5.0, (5, 10, 10)).astype(np.float32)
        _write_tiff(dark_dir / "dark_001.tif", frames)
        (tmp_path / "nir1").mkdir()
        (tmp_path / "nir2").mkdir()
        cf = _make_cf()
        result = cf.calibrate_multicolour_camera(str(tmp_path), mode="nir")
        assert result is None

    def test_rgb_mismatched_intensity_counts_returns_none(self, tmp_path):
        dark_dir = tmp_path / "dark"
        dark_dir.mkdir()
        frames = RNG.normal(100.0, 5.0, (4, 10, 10)).astype(np.float32)
        _write_tiff(dark_dir / "dark_001.tif", frames)

        for colour, n_int in [("B", 2), ("G", 1), ("R", 2)]:
            colour_dir = tmp_path / colour
            colour_dir.mkdir()
            for k in range(n_int):
                sig = RNG.normal(150.0, 5.0, (4, 10, 10)).astype(np.float32)
                _write_tiff(colour_dir / f"Intensity_{k:02d}_001.tif", sig)

        cf = _make_cf(chunk_size=2)
        result = cf.calibrate_multicolour_camera(str(tmp_path))
        assert result is None


class TestCalibrateMulticolourCameraRgb:
    def test_full_rgb_calibration_produces_valid_maps(self, tmp_path):
        tree = _make_rgb_calibration_tree(tmp_path, size=16, n_frames=6, n_intensities=2)
        cf = _make_cf(chunk_size=3)
        result = cf.calibrate_multicolour_camera(str(tree))
        assert result is not None
        offset, variance, gain, readnoise, rqe = result
        assert offset.shape == (16, 16)
        assert variance.shape == (16, 16)
        assert gain.shape == (16, 16)
        assert readnoise.shape == (16, 16)
        assert rqe.shape == (16, 16)
        # Output TIFFs should be written to disk.
        assert (tmp_path / "offset.tif").exists()
        assert (tmp_path / "gain.tif").exists()
        assert (tmp_path / "variance.tif").exists()
        assert (tmp_path / "readnoise.tif").exists()
        assert (tmp_path / "rqe.tif").exists()
        assert (tmp_path / "A.tif").exists()
        assert (tmp_path / "B.tif").exists()


class TestCalibrateMulticolourCameraNir:
    def test_full_nir_calibration_produces_valid_maps(self, tmp_path):
        tree = _make_nir_calibration_tree(tmp_path, size=16, n_frames=6, n_intensities=2)
        cf = _make_cf(chunk_size=3)
        result = cf.calibrate_multicolour_camera(str(tree), mode="nir")
        assert result is not None
        offset, variance, gain, readnoise, rqe = result
        assert offset.shape == (16, 16)
        assert gain.shape == (16, 16)
