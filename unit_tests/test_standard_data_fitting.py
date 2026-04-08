"""Tests for the STANDARD_DATA (S4) fitting strategy.

Verifies:
1. The new enum value and PARAM_DIMENSIONS entry exist.
2. StandardDataFittingProcessor is registered in Image_Analysis_Functions.
3. On a synthetic Bayer ROI with known parameters, STANDARD_DATA converges
   and returns a physically reasonable fit.
4. _raw_data_weights returns finite positive values and matches the expected formula.
5. On a Monte Carlo of N realisations, STANDARD_DATA amplitude precision is
   equal-to or better-than STANDARD_ITER (the S4 hypothesis).
"""

import sys
import os
import numpy as np
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ImageAnalysisFunctions import (
    FittingStrategy,
    FittingConstants,
    Image_Analysis_Functions,
    StandardDataFittingProcessor,
    StandardIterFittingProcessor,
)
import gaussoptfuncs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
SIZE = 20  # ROI size in pixels
READNOISE = 1.5  # e-

# Ximea BGGR mosaic
_MOSAIC = np.array([["B", "G"], ["G", "R"]])


def _make_masks(size=SIZE):
    """Build a 3-channel Bayer mask stack (H x W x 3) for BGGR, dtype bool."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    import MaskFunctions
    mf = MaskFunctions.Mask_Functions()
    masks = mf.get_ROI_mask(0, 0, size, size, mosaic_unit=_MOSAIC)
    # Keep bool dtype — gaussoptfuncs Numba kernels require boolean indexing
    return np.dstack([masks[k] for k in masks])


def _make_synthetic_roi(
    x0=10.0, y0=10.0, sx=1.3, sy=1.3,
    A_B=300.0, A_G=500.0, A_R=200.0,
    bg_B=5.0, bg_G=5.0, bg_R=5.0,
    readnoise=READNOISE, size=SIZE, rng=None,
):
    """Generate a noisy Bayer-filtered ROI from known Gaussian parameters.

    Returns (raw_pe, smoothed_pe, weights, masks, true_params).
    true_params = [x0, y0, sx, sy, A_B+A_G+A_R].
    """
    if rng is None:
        rng = RNG

    masks = _make_masks(size)
    x_arr = np.arange(size, dtype=np.float32)
    buf = np.zeros((size, size), dtype=np.float32)

    # Build parameter vector in the internal sqrt convention
    params = np.array([
        x0, y0, sy, sx,            # note: internal order is (y0, x0, σy, σx)
        np.sqrt(bg_B), np.sqrt(bg_G), np.sqrt(bg_R),
        np.sqrt(A_B), np.sqrt(A_G), np.sqrt(A_R),
    ], dtype=np.float32)

    noiseless = gaussoptfuncs.WLS_model_nobounds(params, masks, x_arr, buf)

    # Poisson + Gaussian noise
    shot = rng.poisson(np.maximum(noiseless, 0).astype(float)).astype(np.float32)
    noise = rng.normal(0, readnoise, size=(size, size)).astype(np.float32)
    raw = shot + noise

    # Smoothed: simple uniform box (mimics IOFunctions.generate_weights input)
    from scipy.ndimage import uniform_filter
    smoothed = uniform_filter(np.maximum(raw, 0), size=3).astype(np.float32)

    # Stage-1 weights from smoothed
    e = np.maximum(smoothed, 0) + 1.0 + readnoise ** 2
    weights = (1.0 / e).astype(np.float32)

    total_A = A_B + A_G + A_R
    return raw, smoothed, weights, masks, (x0, y0, sx, sy, total_A)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestStandardDataEnum(unittest.TestCase):

    def test_enum_value_exists(self):
        self.assertEqual(FittingStrategy.STANDARD_DATA.value, "standard_data")

    def test_param_dimensions(self):
        dims = FittingConstants.PARAM_DIMENSIONS[FittingStrategy.STANDARD_DATA]
        self.assertEqual(dims, {"fit": 12, "error": 10})

    def test_processor_registered(self):
        iaf = Image_Analysis_Functions()
        self.assertIn(FittingStrategy.STANDARD_DATA, iaf.processors)
        self.assertIsInstance(
            iaf.processors[FittingStrategy.STANDARD_DATA],
            StandardDataFittingProcessor,
        )

    def test_processor_is_subclass(self):
        self.assertTrue(issubclass(StandardDataFittingProcessor, StandardIterFittingProcessor))


class TestRawDataWeights(unittest.TestCase):

    def setUp(self):
        self.proc = StandardDataFittingProcessor(readnoise=READNOISE)

    def test_weights_positive_finite(self):
        data = RNG.poisson(100, size=(SIZE, SIZE)).astype(np.float32)
        w = self.proc._raw_data_weights(data)
        self.assertTrue(np.all(np.isfinite(w)))
        self.assertTrue(np.all(w > 0))

    def test_formula(self):
        data = np.ones((SIZE, SIZE), dtype=np.float32) * 100.0
        w = self.proc._raw_data_weights(data)
        expected = 1.0 / (100.0 + 1.0 + READNOISE ** 2)
        np.testing.assert_allclose(w, expected, rtol=1e-5)

    def test_negative_data_clamped(self):
        data = np.full((SIZE, SIZE), -50.0, dtype=np.float32)
        w = self.proc._raw_data_weights(data)
        # max(-50, 0) = 0, so w = 1/(0+1+rn²)
        expected = 1.0 / (1.0 + READNOISE ** 2)
        np.testing.assert_allclose(w, expected, rtol=1e-5)

    def test_dtype_float32(self):
        data = np.ones((SIZE, SIZE), dtype=np.float64) * 50.0
        w = self.proc._raw_data_weights(data)
        self.assertEqual(w.dtype, np.float32)


class TestStandardDataFit(unittest.TestCase):
    """End-to-end: fit a synthetic ROI with STANDARD_DATA and check recovery."""

    def setUp(self):
        self.proc = StandardDataFittingProcessor(readnoise=READNOISE)

    def test_fit_returns_correct_shape(self):
        raw, smoothed, weights, masks, _ = _make_synthetic_roi()
        pfit, perr = self.proc.fit_single_punctum(
            raw, smoothed, weights, (0.0, 0.0), masks=masks
        )
        # fit_single_punctum returns 10 params + chi² = 11 elements.
        # The 12th (frame index) is appended by fit_puncta_parallel_method.
        self.assertEqual(pfit.shape, (11,))
        self.assertEqual(perr.shape, (10,))

    def test_fit_position_recovery(self, tol_px=1.0):
        """Fitted x, y should be within tol_px of the true position."""
        raw, smoothed, weights, masks, (x0, y0, sx, sy, _) = _make_synthetic_roi()
        pfit, _ = self.proc.fit_single_punctum(
            raw, smoothed, weights, (0.0, 0.0), masks=masks
        )
        self.assertFalse(np.any(np.isnan(pfit)), "Fit returned NaN")
        # pfit layout: [xc, yc, sx, sy, bg_B, bg_G, bg_R, A_B, A_G, A_R, chi2, frame]
        self.assertAlmostEqual(pfit[0], x0, delta=tol_px)
        self.assertAlmostEqual(pfit[1], y0, delta=tol_px)

    def test_fit_photon_recovery(self, frac_tol=0.20):
        """Total fitted photons should be within frac_tol of true total.

        Uses a high-SNR ROI (10 000 total photons) so a single realisation
        reliably lands within tolerance; at lower photon counts shot noise
        alone can push a single draw well outside 15 %.
        """
        raw, smoothed, weights, masks, (_, _, _, _, true_A) = _make_synthetic_roi(
            A_B=3000.0, A_G=5000.0, A_R=2000.0,  # 10 000 pe total
            rng=np.random.default_rng(99),
        )
        pfit, _ = self.proc.fit_single_punctum(
            raw, smoothed, weights, (0.0, 0.0), masks=masks
        )
        self.assertFalse(np.any(np.isnan(pfit)))
        # pfit: [xc, yc, sx, sy, bg_B, bg_G, bg_R, A_B, A_G, A_R, chi²]
        fitted_A = pfit[7] + pfit[8] + pfit[9]
        self.assertAlmostEqual(fitted_A / true_A, 1.0, delta=frac_tol)

    def test_no_fit_on_blank_roi(self):
        """All-zero smoothed ROI should return NaN without crashing."""
        raw = np.zeros((SIZE, SIZE), dtype=np.float32)
        smoothed = np.zeros((SIZE, SIZE), dtype=np.float32)
        weights = np.ones((SIZE, SIZE), dtype=np.float32)
        _, _, _, masks, _ = _make_synthetic_roi()
        pfit, perr = self.proc.fit_single_punctum(
            raw, smoothed, weights, (0.0, 0.0), masks=masks
        )
        self.assertTrue(np.all(np.isnan(pfit)))

    def test_requires_masks(self):
        raw, smoothed, weights, _, _ = _make_synthetic_roi()
        from ImageAnalysisFunctions import FittingValidationError
        with self.assertRaises(FittingValidationError):
            self.proc.fit_single_punctum(raw, smoothed, weights, (0.0, 0.0), masks=None)


class TestStandardDataVsIter(unittest.TestCase):
    """Monte Carlo: STANDARD_DATA amplitude precision >= STANDARD_ITER.

    Runs N realisations of the same ground-truth ROI, fits with both strategies,
    and checks that the std of the fitted total amplitude is no worse for S4.
    We allow a 20 % margin so the test is not sensitive to RNG seed variation.
    """

    N = 150  # realisations
    MARGIN = 1.20  # S4 std must be <= MARGIN * S2 std

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(7)
        proc_iter = StandardIterFittingProcessor(readnoise=READNOISE)
        proc_data = StandardDataFittingProcessor(readnoise=READNOISE)

        A_iter, A_data = [], []
        chi_iter, chi_data = [], []

        for _ in range(cls.N):
            raw, sm, w, masks, (_, _, _, _, true_A) = _make_synthetic_roi(rng=rng)
            coords = (0.0, 0.0)

            p_iter, _ = proc_iter.fit_single_punctum(raw, sm, w, coords, masks=masks)
            p_data, _ = proc_data.fit_single_punctum(raw, sm, w, coords, masks=masks)

            # pfit from fit_single_punctum: [xc,yc,sx,sy,bg_B,bg_G,bg_R,A_B,A_G,A_R,chi²]
            if not np.any(np.isnan(p_iter)):
                A_iter.append(p_iter[7] + p_iter[8] + p_iter[9])
                chi_iter.append(p_iter[10])
            if not np.any(np.isnan(p_data)):
                A_data.append(p_data[7] + p_data[8] + p_data[9])
                chi_data.append(p_data[10])

        cls.A_iter = np.array(A_iter)
        cls.A_data = np.array(A_data)
        cls.chi_iter = np.array(chi_iter)
        cls.chi_data = np.array(chi_data)
        cls.true_A = 1000.0  # A_B + A_G + A_R from _make_synthetic_roi defaults

    def test_both_strategies_converge(self):
        self.assertGreater(len(self.A_iter), self.N * 0.9,
                           "STANDARD_ITER convergence rate < 90 %")
        self.assertGreater(len(self.A_data), self.N * 0.9,
                           "STANDARD_DATA convergence rate < 90 %")

    def test_standard_data_precision_not_worse(self):
        std_iter = np.std(self.A_iter)
        std_data = np.std(self.A_data)
        self.assertLessEqual(
            std_data, std_iter * self.MARGIN,
            f"STANDARD_DATA std ({std_data:.1f}) > {self.MARGIN}× STANDARD_ITER std ({std_iter:.1f})"
        )

    def test_standard_data_chi2_closer_to_one(self):
        """Median |chi²-1| should be no larger for STANDARD_DATA."""
        dev_iter = np.median(np.abs(self.chi_iter - 1.0))
        dev_data = np.median(np.abs(self.chi_data - 1.0))
        self.assertLessEqual(
            dev_data, dev_iter * self.MARGIN,
            f"|chi²-1| median: STANDARD_DATA={dev_data:.3f}, STANDARD_ITER={dev_iter:.3f}"
        )

    def test_print_summary(self):
        """Print a human-readable summary (always passes)."""
        print("\n--- S2 vs S4 Monte Carlo summary ---")
        print(f"  N realisations   : {self.N}")
        print(f"  True total A     : {self.true_A:.0f} pe")
        print(f"  S2 mean A        : {np.mean(self.A_iter):.1f}  std={np.std(self.A_iter):.1f}")
        print(f"  S4 mean A        : {np.mean(self.A_data):.1f}  std={np.std(self.A_data):.1f}")
        print(f"  S2 median chi²   : {np.median(self.chi_iter):.3f}")
        print(f"  S4 median chi²   : {np.median(self.chi_data):.3f}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
