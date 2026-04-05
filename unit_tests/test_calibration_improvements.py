"""
Tests for the three sCMOS calibration performance improvements:
  1. Vectorised gain computation (replaces pixel-wise loop)
  2. Combined single-pass offset + variance (replaces two separate passes)
  3. Chunked frame reading (replaces exception-based frame-by-frame loop)

Each improvement is tested by comparing its output to a reference
implementation (either the old code reimplemented here, or analytical values)
on small synthetic data so the suite runs quickly.
"""

import os
import sys
import tempfile

import numpy as np
import pytest
import tifffile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import CalibrationFunctions
import IOFunctions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def _write_tiffs(directory, frames, prefix="dark"):
    """Write a (N, H, W) array as one multi-page TIFF per call.

    The file name contains both the prefix and '.tif' so that
    CalibrationFunctions.filesearch(dir, '.tif', prefix) finds it.
    """
    path = os.path.join(directory, f"{prefix}_001.tif")
    tifffile.imwrite(path, frames)
    return path


def _make_calibration_functions(**kwargs):
    """Return a CalibrationFunctions instance wired to a real IO_Functions."""
    io = IOFunctions.IO_Functions()
    return CalibrationFunctions.Calibration_Functions(io_functions=io, **kwargs)


# ---------------------------------------------------------------------------
# Reference implementations (old algorithms, small and readable)
# ---------------------------------------------------------------------------

def _gain_loop_reference(A, B):
    """Original pixel-wise gain loop, reimplemented for reference."""
    gain = np.full(A.shape[:2], np.nan, dtype=np.float64)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            Bi = B[i, j, :]
            Ai = A[i, j, :]
            denom = np.dot(Bi, Bi)
            if denom > 0:
                gain[i, j] = np.dot(Bi, Ai) / denom
    return gain


def _compute_offset_reference(frames):
    """Per-pixel mean across frames (N, H, W)."""
    return frames.mean(axis=0)


def _compute_variance_reference(frames, offset):
    """Per-pixel variance using the two-pass formula that the old code used:
       variance = mean(X²) - offset²
    """
    return (frames.astype(np.float64) ** 2).mean(axis=0) - offset.astype(np.float64) ** 2


# ---------------------------------------------------------------------------
# Improvement 1: Vectorised gain
# ---------------------------------------------------------------------------

class TestGainVectorisation:
    """The vectorised einsum gain must match the reference loop exactly."""

    def _run_vectorised(self, A, B):
        denom = np.einsum('ijk,ijk->ij', B, B)
        numer = np.einsum('ijk,ijk->ij', A, B)
        return np.where(denom > 0, numer / denom, np.nan)

    def test_small_random_array(self):
        """Vectorised result matches loop on a 10×8 sensor with 5 intensities."""
        H, W, P = 10, 8, 5
        A = RNG.random((H, W, P)).astype(np.float64)
        B = RNG.random((H, W, P)).astype(np.float64)

        ref = _gain_loop_reference(A, B)
        vec = self._run_vectorised(A, B)

        np.testing.assert_allclose(vec, ref, rtol=1e-12,
                                   err_msg="Vectorised gain differs from loop reference")

    def test_zero_denominator_yields_nan(self):
        """Pixels where B is all-zero must produce NaN, not inf or a number."""
        H, W, P = 4, 4, 3
        A = RNG.random((H, W, P)).astype(np.float64)
        B = RNG.random((H, W, P)).astype(np.float64)
        B[2, 3, :] = 0.0  # force zero denominator at one pixel

        vec = self._run_vectorised(A, B)

        assert np.isnan(vec[2, 3]), "Expected NaN for zero-denominator pixel"
        # All other pixels should be finite
        mask = np.ones((H, W), dtype=bool)
        mask[2, 3] = False
        assert np.all(np.isfinite(vec[mask])), "Non-zero pixels should be finite"

    def test_single_intensity(self):
        """With one intensity level (P=1) gain = A/B element-wise."""
        H, W = 6, 5
        A = RNG.random((H, W, 1)).astype(np.float64) + 0.1
        B = RNG.random((H, W, 1)).astype(np.float64) + 0.1

        vec = self._run_vectorised(A, B)
        expected = A[:, :, 0] / B[:, :, 0]

        np.testing.assert_allclose(vec, expected, rtol=1e-12)

    def test_matches_reference_with_varied_magnitudes(self):
        """Test over a range of signal magnitudes (100–10000 ADU typical range)."""
        H, W, P = 15, 12, 8
        A = RNG.uniform(10, 10000, (H, W, P))
        B = RNG.uniform(10, 10000, (H, W, P))

        ref = _gain_loop_reference(A, B)
        vec = self._run_vectorised(A, B)

        np.testing.assert_allclose(vec, ref, rtol=1e-10)


# ---------------------------------------------------------------------------
# Improvement 2: Combined single-pass offset + variance
# ---------------------------------------------------------------------------

class TestCombinedOffsetVariance:
    """calculate_offset_and_variance must agree with the two-pass reference."""

    @pytest.fixture
    def tiff_dir(self, tmp_path):
        """Write 30 frames (N=30, H=12, W=10) as a single multi-page TIFF."""
        N, H, W = 30, 12, 10
        # True mean=200, true std=15 (so true variance=225)
        frames = RNG.normal(200.0, 15.0, size=(N, H, W)).astype(np.float32)
        _write_tiffs(str(tmp_path), frames, prefix="dark")
        return str(tmp_path), frames

    def test_offset_close_to_reference(self, tiff_dir):
        directory, frames = tiff_dir
        cf = _make_calibration_functions(chunk_size=10)
        offset, _ = cf.calculate_offset_and_variance(directory, "dark")

        ref_offset = _compute_offset_reference(frames)

        np.testing.assert_allclose(
            offset, ref_offset, rtol=1e-4,
            err_msg="Offset from combined pass differs from reference mean"
        )

    def test_variance_close_to_reference(self, tiff_dir):
        directory, frames = tiff_dir
        cf = _make_calibration_functions(chunk_size=10)
        offset, variance = cf.calculate_offset_and_variance(directory, "dark")

        ref_offset = _compute_offset_reference(frames)
        ref_variance = _compute_variance_reference(frames, ref_offset)

        np.testing.assert_allclose(
            variance, ref_variance, rtol=1e-3,
            err_msg="Variance from combined pass differs from reference"
        )

    def test_variance_is_non_negative(self, tiff_dir):
        directory, _ = tiff_dir
        cf = _make_calibration_functions(chunk_size=5)
        _, variance = cf.calculate_offset_and_variance(directory, "dark")
        assert np.all(variance >= -1e-5), "Variance must not be significantly negative"

    def test_output_dtype_is_float32(self, tiff_dir):
        directory, _ = tiff_dir
        cf = _make_calibration_functions(chunk_size=10)
        offset, variance = cf.calculate_offset_and_variance(directory, "dark")
        assert offset.dtype == np.float32
        assert variance.dtype == np.float32

    def test_matches_two_pass_on_same_data(self, tmp_path):
        """Combined single-pass result must agree with the old two-pass result
        to within float32 precision (the only source of difference)."""
        N, H, W = 40, 8, 6
        frames = RNG.normal(100.0, 5.0, size=(N, H, W)).astype(np.float32)
        _write_tiffs(str(tmp_path), frames, prefix="dark")

        cf = _make_calibration_functions(chunk_size=15)

        # New combined method
        offset_new, variance_new = cf.calculate_offset_and_variance(str(tmp_path), "dark")

        # Reference: two-pass from raw frames
        ref_offset = _compute_offset_reference(frames)
        ref_variance = _compute_variance_reference(frames, ref_offset)

        np.testing.assert_allclose(offset_new, ref_offset.astype(np.float32), rtol=1e-4)
        np.testing.assert_allclose(variance_new, ref_variance.astype(np.float32), rtol=1e-3)

    def test_offset_matches_old_calculate_offset(self, tmp_path):
        """Offset from calculate_offset_and_variance must match calculate_offset
        called on the same files — i.e. the new method agrees with the old code path."""
        N, H, W = 40, 8, 6
        frames = RNG.normal(100.0, 5.0, size=(N, H, W)).astype(np.float32)
        _write_tiffs(str(tmp_path), frames, prefix="dark")

        cf = _make_calibration_functions(chunk_size=15)

        offset_old = cf.calculate_offset(str(tmp_path), "dark")
        offset_new, _ = cf.calculate_offset_and_variance(str(tmp_path), "dark")

        np.testing.assert_allclose(
            offset_new, offset_old, rtol=1e-5,
            err_msg="calculate_offset_and_variance offset differs from calculate_offset"
        )

    def test_variance_matches_old_calculate_variance(self, tmp_path):
        """Variance from calculate_offset_and_variance must match calculate_variance
        called on the same files with the same offset — i.e. the new method agrees
        with the old code path for single-channel data."""
        N, H, W = 40, 8, 6
        frames = RNG.normal(100.0, 5.0, size=(N, H, W)).astype(np.float32)
        _write_tiffs(str(tmp_path), frames, prefix="dark")

        cf = _make_calibration_functions(chunk_size=15)

        # Old path: compute offset first, then variance with that offset
        offset_old = cf.calculate_offset(str(tmp_path), "dark")
        variance_old = cf.calculate_variance(offset_old, str(tmp_path), "dark")

        # New path: single pass
        _, variance_new = cf.calculate_offset_and_variance(str(tmp_path), "dark")

        np.testing.assert_allclose(
            variance_new, variance_old, rtol=1e-4,
            err_msg="calculate_offset_and_variance variance differs from calculate_variance"
        )


# ---------------------------------------------------------------------------
# Improvement 3: Chunked reads in _process_calibration_files
# ---------------------------------------------------------------------------

class TestChunkedReads:
    """calculate_offset and calculate_variance must give identical results
    regardless of chunk_size (1, 5, or equal to total frame count)."""

    @pytest.fixture
    def tiff_dir(self, tmp_path):
        N, H, W = 25, 10, 8
        frames = RNG.normal(150.0, 10.0, size=(N, H, W)).astype(np.float32)
        _write_tiffs(str(tmp_path), frames, prefix="dark")
        return str(tmp_path), frames, N

    def test_offset_invariant_to_chunk_size(self, tiff_dir):
        directory, frames, _ = tiff_dir

        offsets = {}
        for cs in [1, 5, 25, 100]:
            cf = _make_calibration_functions(chunk_size=cs)
            offsets[cs] = cf.calculate_offset(directory, "dark")

        ref = offsets[1]
        for cs, off in offsets.items():
            np.testing.assert_allclose(
                off, ref, rtol=1e-5,
                err_msg=f"Offset differs between chunk_size=1 and chunk_size={cs}"
            )

    def test_variance_invariant_to_chunk_size(self, tiff_dir):
        directory, frames, _ = tiff_dir

        # Compute a shared offset first (chunk_size=10)
        cf_ref = _make_calibration_functions(chunk_size=10)
        offset = cf_ref.calculate_offset(directory, "dark")

        variances = {}
        for cs in [1, 5, 25, 100]:
            cf = _make_calibration_functions(chunk_size=cs)
            variances[cs] = cf.calculate_variance(offset, directory, "dark")

        ref = variances[1]
        for cs, var in variances.items():
            np.testing.assert_allclose(
                var, ref, rtol=1e-5,
                err_msg=f"Variance differs between chunk_size=1 and chunk_size={cs}"
            )

    def test_chunked_offset_matches_analytical_mean(self, tiff_dir):
        """Offset computed via chunked reads must equal the true per-pixel mean."""
        directory, frames, _ = tiff_dir
        cf = _make_calibration_functions(chunk_size=7)
        offset = cf.calculate_offset(directory, "dark")

        ref = _compute_offset_reference(frames)
        np.testing.assert_allclose(offset, ref, rtol=1e-4)

    def test_get_n_frames_correct(self, tiff_dir):
        """IOFunctions.get_n_frames must return the true frame count."""
        directory, frames, N = tiff_dir
        io = IOFunctions.IO_Functions()
        path = os.path.join(directory, "dark_001.tif")
        assert io.get_n_frames(path) == N

    def test_chunked_offset_matches_old_exception_loop(self, tmp_path):
        """calculate_offset using chunked reads must match the result of the old
        exception-based frame-by-frame loop on the same files.

        The old loop is reconstructed here using read_tiff(path, frame_idx) until
        IndexError, exactly as _process_calibration_files used to work.
        """
        N, H, W = 25, 10, 8
        frames = RNG.normal(150.0, 10.0, size=(N, H, W)).astype(np.float32)
        _write_tiffs(str(tmp_path), frames, prefix="dark")

        io = IOFunctions.IO_Functions()
        path = os.path.join(str(tmp_path), "dark_001.tif")

        # Reconstruct the old exception-based accumulator loop
        frame0 = io.read_tiff(path, 0)
        acc = np.zeros(frame0.shape, dtype=np.float64)
        n = 0
        while True:
            try:
                frame = io.read_tiff(path, n)
                acc += frame
                n += 1
            except (IOError, OSError, IndexError, ValueError):
                break
        offset_old_loop = (acc / n).astype(np.float32)

        # New chunked path
        cf = _make_calibration_functions(chunk_size=7)
        offset_new = cf.calculate_offset(str(tmp_path), "dark")

        np.testing.assert_allclose(
            offset_new, offset_old_loop, rtol=1e-5,
            err_msg="Chunked calculate_offset differs from old exception-based loop"
        )

    def test_chunked_variance_matches_old_exception_loop(self, tmp_path):
        """calculate_variance using chunked reads must match the old frame-by-frame
        loop on the same files."""
        N, H, W = 25, 10, 8
        frames = RNG.normal(150.0, 10.0, size=(N, H, W)).astype(np.float32)
        _write_tiffs(str(tmp_path), frames, prefix="dark")

        io = IOFunctions.IO_Functions()
        path = os.path.join(str(tmp_path), "dark_001.tif")

        # Compute shared offset via old loop
        frame0 = io.read_tiff(path, 0)
        acc = np.zeros(frame0.shape, dtype=np.float64)
        n = 0
        while True:
            try:
                frame = io.read_tiff(path, n)
                acc += frame
                n += 1
            except (IOError, OSError, IndexError, ValueError):
                break
        offset = (acc / n).astype(np.float32)

        # Old loop variance: accumulate sum(frame^2 - offset^2)
        offset_sq = offset.astype(np.float64) ** 2
        var_acc = np.zeros(frame0.shape, dtype=np.float64)
        for idx in range(n):
            frame = io.read_tiff(path, idx).astype(np.float64)
            var_acc += frame ** 2 - offset_sq
        variance_old_loop = (var_acc / n).astype(np.float32)

        # New chunked path (same offset fed in)
        cf = _make_calibration_functions(chunk_size=7)
        variance_new = cf.calculate_variance(offset, str(tmp_path), "dark")

        np.testing.assert_allclose(
            variance_new, variance_old_loop, rtol=1e-4,
            err_msg="Chunked calculate_variance differs from old exception-based loop"
        )


# ---------------------------------------------------------------------------
# Correctness bug: offset used before fully populated
# ---------------------------------------------------------------------------

class TestOffsetBugFixed:
    """
    Demonstrate that the old code used a partially-filled offset when computing
    per-channel variance in calibrate_multicolour_camera, and that
    calculate_offset_and_variance gives the correct per-channel variance.

    We test calculate_offset_and_variance independently for each channel
    (as the fixed calibrate_multicolour_camera now does) and verify the
    variance is correct rather than inflated by a wrong-channel offset.
    """

    @pytest.fixture
    def two_channel_dirs(self, tmp_path):
        """Create synthetic dark-like frames for two channels with known stats."""
        N, H, W = 30, 10, 8
        # Channel A: mean=100, std=5  → variance≈25
        frames_A = RNG.normal(100.0, 5.0, size=(N, H, W)).astype(np.float32)
        # Channel B: mean=200, std=10 → variance≈100
        frames_B = RNG.normal(200.0, 10.0, size=(N, H, W)).astype(np.float32)

        dir_A = tmp_path / "A"
        dir_B = tmp_path / "B"
        dir_A.mkdir()
        dir_B.mkdir()

        _write_tiffs(str(dir_A), frames_A, prefix="Intensity_01")
        _write_tiffs(str(dir_B), frames_B, prefix="Intensity_01")

        return str(dir_A), str(dir_B), frames_A, frames_B

    def test_each_channel_variance_is_correct(self, two_channel_dirs):
        """Per-channel variance from calculate_offset_and_variance should match
        the true variance of that channel's frames, not be contaminated by the
        other channel's mean."""
        dir_A, dir_B, frames_A, frames_B = two_channel_dirs
        cf = _make_calibration_functions(chunk_size=10)

        off_A, var_A = cf.calculate_offset_and_variance(dir_A, "Intensity_01")
        off_B, var_B = cf.calculate_offset_and_variance(dir_B, "Intensity_01")

        # Reference: true per-pixel variance from raw frames
        ref_var_A = _compute_variance_reference(frames_A, _compute_offset_reference(frames_A))
        ref_var_B = _compute_variance_reference(frames_B, _compute_offset_reference(frames_B))

        np.testing.assert_allclose(var_A, ref_var_A.astype(np.float32), rtol=1e-3,
                                   err_msg="Channel A variance incorrect")
        np.testing.assert_allclose(var_B, ref_var_B.astype(np.float32), rtol=1e-3,
                                   err_msg="Channel B variance incorrect")

    def test_old_behaviour_would_give_wrong_variance(self, two_channel_dirs):
        """Show that computing variance with the wrong (partial) offset gives a
        different answer — confirming the bug existed and is now absent."""
        dir_A, dir_B, frames_A, frames_B = two_channel_dirs
        cf = _make_calibration_functions(chunk_size=10)

        # Correct: each channel uses its own offset
        off_A, var_A_correct = cf.calculate_offset_and_variance(dir_A, "Intensity_01")
        off_B, var_B_correct = cf.calculate_offset_and_variance(dir_B, "Intensity_01")

        # Old bug: channel B variance computed with channel A's offset (wrong mean)
        var_B_bugged = cf.calculate_variance(off_A, dir_B, "Intensity_01")

        # The bugged variance will NOT equal the correct variance because off_A ≠ off_B
        assert not np.allclose(var_B_bugged, var_B_correct, rtol=1e-2), (
            "Expected bugged and correct variances to differ, but they are the same. "
            "This suggests off_A == off_B (unlikely with these synthetic data)."
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
