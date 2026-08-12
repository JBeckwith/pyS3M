#!/usr/bin/env python3
"""
Full coverage tests for pyS3M.simulation.pattern_source -- turning a pattern
image into per-dye candidate positions, blinking schedules, and (via
simulate_acquisition) full synthetic Bayer acquisitions.

Deliberately tiny throughout: small hand-built RGBA arrays (tens of pixels)
for the pure image/geometry functions, and n_frames=1-3 for the
simulate_acquisition integration tests -- the rendered camera FOV is fixed
at ~145x145 px regardless of pattern resolution (DEFAULT_PATTERN_FOV_UM is a
fixed 10 um span), so pattern size doesn't buy speed there; frame count does.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import pyS3M.simulation.pattern_source as ps
from pyS3M.AnalysisPipeline import AnalysisPipeline
from pyS3M.Constants import AnalysisConfig


# ======================================================================
# Shared fixtures
# ======================================================================

@pytest.fixture(scope="module")
def pipe():
    cfg = AnalysisConfig(display=False)
    p = AnalysisPipeline(camera="ximea", config=cfg)
    p.load_calibration("Camera_Calibrations/Ximea_Camera")
    return p


@pytest.fixture(scope="module")
def cam_maps(pipe):
    camera_pixel_size_nm = pipe.pixel_size * 1000.0
    # A 20x20 pattern always implies the same ~145x145 FOV (fixed 10um span).
    width, height = ps.image_fov_camera_pixels(
        np.full((20, 20, 3), 255, dtype=np.uint8), camera_pixel_size_nm,
    )
    return {
        "width": width, "height": height,
        "gain": pipe.gain_map[:height, :width],
        "offset": pipe.offset_map[:height, :width],
        "variance": pipe.variance[:height, :width],
        "rqe": pipe.rqe[:height, :width],
    }


@pytest.fixture(scope="module")
def spectral_funcs():
    import pyS3M.SpectralFunctions as SpectralFunctions
    return SpectralFunctions.Spectral_Funcs()


@pytest.fixture(scope="module")
def nile_red_funcs():
    import pyS3M.NileRedFunctions as NileRedFunctions
    return NileRedFunctions.NileRed_Functions()


def _two_colour_pattern(size=20):
    """White background, a black square (top-left) and a grey square
    (bottom-right) -- two well-separated, well-populated foreground colours."""
    img = np.full((size, size, 3), 255, dtype=np.uint8)
    img[2:8, 2:8] = [0, 0, 0]
    img[size - 8:size - 2, size - 8:size - 2] = [128, 128, 128]
    return img


def _one_colour_pattern(size=20):
    img = np.full((size, size, 3), 255, dtype=np.uint8)
    img[5:15, 5:15] = [0, 0, 0]
    return img


def _flat_pattern(size=10, colour=(200, 50, 50)):
    """No foreground at all -- every pixel is the same colour, so
    detect_palette's auto-background picks it and mask_from_image returns {}."""
    return np.full((size, size, 3), colour, dtype=np.uint8)


# ======================================================================
# default_pattern_pixel_size_nm / image_fov_camera_pixels
# ======================================================================

class TestPixelSizeAndFov:
    def test_default_pattern_pixel_size_nm(self):
        assert ps.default_pattern_pixel_size_nm(1000, fov_um=10.0) == pytest.approx(10.0)

    def test_image_fov_camera_pixels_from_array(self):
        img = _one_colour_pattern()
        w, h = ps.image_fov_camera_pixels(img, camera_pixel_size_nm=69.0)
        assert w > 0 and h > 0

    def test_image_fov_camera_pixels_from_path(self, tmp_path):
        from PIL import Image
        path = tmp_path / "p.png"
        Image.fromarray(_one_colour_pattern()).save(path)
        w, h = ps.image_fov_camera_pixels(path, camera_pixel_size_nm=69.0)
        assert w > 0 and h > 0


# ======================================================================
# load_pattern_image / _as_rgba
# ======================================================================

class TestLoadAndAsRgba:
    def test_load_pattern_image(self, tmp_path):
        from PIL import Image
        path = tmp_path / "p.png"
        Image.fromarray(_one_colour_pattern()).save(path)
        arr = ps.load_pattern_image(path)
        assert arr.shape[-1] == 4
        assert arr.dtype == np.uint8

    def test_as_rgba_from_path(self, tmp_path):
        from PIL import Image
        path = tmp_path / "p.png"
        Image.fromarray(_one_colour_pattern()).save(path)
        arr = ps._as_rgba(path)
        assert arr.shape[-1] == 4

    def test_as_rgba_grayscale(self):
        gray = np.full((10, 10), 100, dtype=np.uint8)
        arr = ps._as_rgba(gray)
        assert arr.shape == (10, 10, 4)
        assert np.all(arr[..., 3] == 255)
        np.testing.assert_array_equal(arr[..., 0], gray)

    def test_as_rgba_rgb(self):
        rgb = _one_colour_pattern()
        arr = ps._as_rgba(rgb)
        assert arr.shape == (20, 20, 4)
        assert np.all(arr[..., 3] == 255)

    def test_as_rgba_already_rgba(self):
        rgba = np.dstack([_one_colour_pattern(), np.full((20, 20), 200, dtype=np.uint8)])
        arr = ps._as_rgba(rgba)
        assert arr.shape == (20, 20, 4)
        np.testing.assert_array_equal(arr[..., 3], 200)


# ======================================================================
# detect_palette
# ======================================================================

class TestDetectPalette:
    def test_auto_background_and_foreground(self):
        bg, fg = ps.detect_palette(_two_colour_pattern())
        assert bg == (255, 255, 255)
        assert (0, 0, 0) in fg
        assert (128, 128, 128) in fg

    def test_explicit_background(self):
        bg, fg = ps.detect_palette(_two_colour_pattern(), background=(255, 255, 255))
        assert bg == (255, 255, 255)
        assert len(fg) == 2

    def test_min_fraction_filters_rare_colours(self):
        img = _two_colour_pattern(size=40)
        # A single rare pixel below min_fraction should not be reported.
        img[0, 0] = [1, 2, 3]
        bg, fg = ps.detect_palette(img, min_fraction=0.01)
        assert (1, 2, 3) not in fg

    def test_merge_distance_dedups_near_colours(self):
        img = np.full((20, 20, 3), 255, dtype=np.uint8)
        img[2:8, 2:8] = [10, 10, 10]
        img[10:16, 10:16] = [12, 12, 12]  # near-duplicate of the above
        bg, fg = ps.detect_palette(img, merge_distance=24.0)
        assert len(fg) == 1

    def test_no_foreground_flat_image(self):
        bg, fg = ps.detect_palette(_flat_pattern())
        assert fg == []


# ======================================================================
# mask_from_image
# ======================================================================

class TestMaskFromImage:
    def test_two_colours(self):
        masks = ps.mask_from_image(_two_colour_pattern())
        assert set(masks.keys()) == {(0, 0, 0), (128, 128, 128)}
        assert masks[(0, 0, 0)].sum() > 0
        assert masks[(128, 128, 128)].sum() > 0

    def test_no_foreground_returns_empty_dict(self):
        assert ps.mask_from_image(_flat_pattern()) == {}

    def test_all_zero_score_colour_gets_empty_mask(self):
        # detect_palette only looks at RGB (ignores alpha), so a colour
        # whose pixels are all fully transparent (alpha=0) is still
        # reported as a foreground colour there, but contributes nothing
        # to mask_from_image's own alpha-gated score -- an all-False mask,
        # not a skipped/missing entry.
        size = 20
        rgb = np.full((size, size, 3), 255, dtype=np.uint8)
        rgb[2:10, 2:10] = [0, 0, 0]
        alpha = np.full((size, size), 255, dtype=np.uint8)
        rgb[12:18, 12:18] = [50, 50, 50]
        alpha[12:18, 12:18] = 0
        rgba = np.dstack([rgb, alpha])

        masks = ps.mask_from_image(rgba)
        assert masks[(0, 0, 0)].sum() > 0
        assert masks[(50, 50, 50)].sum() == 0

    def test_otsu_value_error_fallback(self, monkeypatch):
        # threshold_otsu doesn't actually raise on any realistic score array
        # with the installed skimage version (it special-cases constant
        # arrays), so force the except-ValueError fallback directly to
        # verify it still produces a sane mask rather than crashing.
        monkeypatch.setattr(
            ps, "threshold_otsu",
            lambda score: (_ for _ in ()).throw(ValueError("forced")),
        )
        masks = ps.mask_from_image(_one_colour_pattern())
        assert masks[(0, 0, 0)].sum() > 0  # threshold=0.0 -> any score>0 kept


# ======================================================================
# sample_n_positions_in_mask
# ======================================================================

class TestSampleNPositionsInMask:
    def test_empty_mask_returns_empty(self):
        mask = np.zeros((10, 10), dtype=bool)
        pos = ps.sample_n_positions_in_mask(mask, n=5)
        assert pos.shape == (2, 0)

    def test_n_le_zero_returns_empty(self):
        mask = np.ones((10, 10), dtype=bool)
        pos = ps.sample_n_positions_in_mask(mask, n=0)
        assert pos.shape == (2, 0)

    def test_with_replacement_default(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[3:6, 3:6] = True
        rng = np.random.default_rng(0)
        pos = ps.sample_n_positions_in_mask(mask, n=20, rng=rng)
        assert pos.shape == (2, 20)

    def test_min_dist_rejection_sampling_success(self):
        mask = np.ones((30, 30), dtype=bool)
        rng = np.random.default_rng(1)
        pos = ps.sample_n_positions_in_mask(mask, n=10, rng=rng, min_dist_px=3.0)
        assert pos.shape[1] == 10
        from scipy.spatial.distance import pdist
        assert pdist(pos.T).min() >= 3.0 - 1e-9

    def test_min_dist_rejection_sampling_shortfall_warns(self):
        # Tiny mask, large separation, many requested points -> can't fit
        # them all; should warn and return fewer than n.
        mask = np.zeros((5, 5), dtype=bool)
        mask[2, 2] = True
        rng = np.random.default_rng(2)
        with pytest.warns(UserWarning, match="Could only place"):
            pos = ps.sample_n_positions_in_mask(
                mask, n=10, rng=rng, min_dist_px=50.0, max_tries_per_point=5,
            )
        assert pos.shape[1] < 10

    def test_min_dist_all_rejected_returns_empty(self):
        # The very first draw is always accepted unconditionally (nothing to
        # compare it against yet), so "returns empty" is only reachable when
        # the rejection loop runs zero times at all: max_tries_per_point=0.
        mask = np.ones((5, 5), dtype=bool)
        rng = np.random.default_rng(3)
        with pytest.warns(UserWarning, match="Could only place 0/1"):
            pos = ps.sample_n_positions_in_mask(
                mask, n=1, rng=rng, min_dist_px=1.0, max_tries_per_point=0,
            )
        assert pos.shape == (2, 0)


# ======================================================================
# sample_positions_in_mask / sample_positions_per_colour
# ======================================================================

class TestSamplePositionsInMask:
    def test_normal_case(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[5:15, 5:15] = True
        rng = np.random.default_rng(0)
        pos = ps.sample_positions_in_mask(mask, density_per_um2=50.0, rng=rng)
        assert pos.shape[0] == 2

    def test_zero_density_gives_zero_candidates(self):
        mask = np.ones((10, 10), dtype=bool)
        pos = ps.sample_positions_in_mask(mask, density_per_um2=0.0, rng=np.random.default_rng(0))
        assert pos.shape == (2, 0)

    def test_default_pixel_size_and_rng(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[5:15, 5:15] = True
        pos = ps.sample_positions_in_mask(mask, density_per_um2=50.0)
        assert pos.shape[0] == 2

    def test_min_dist_nm_forwarded(self):
        # Small density/pixel size so expected_n stays tiny (a handful of
        # points) -- min_dist rejection sampling scales with n *and* the
        # requested separation, so an unrealistically large n here would
        # make this single test take minutes.
        mask = np.ones((20, 20), dtype=bool)
        rng = np.random.default_rng(0)
        pos = ps.sample_positions_in_mask(
            mask, density_per_um2=5.0, pixel_size_nm=50.0, rng=rng, min_dist_nm=20.0,
        )
        assert pos.shape[0] == 2


class TestSamplePositionsPerColour:
    def test_multiple_colours_default_rng(self):
        masks = {
            (0, 0, 0): np.ones((10, 10), dtype=bool),
            (1, 1, 1): np.ones((10, 10), dtype=bool),
        }
        out = ps.sample_positions_per_colour(masks, density_per_um2=50.0)
        assert set(out.keys()) == set(masks.keys())

    def test_shared_rng(self):
        masks = {(0, 0, 0): np.ones((10, 10), dtype=bool)}
        rng = np.random.default_rng(0)
        out = ps.sample_positions_per_colour(masks, density_per_um2=50.0, rng=rng)
        assert (0, 0, 0) in out


# ======================================================================
# plot_sampled_positions / plot_pattern_masks
# ======================================================================

class TestPlotSampledPositions:
    def test_with_points_and_names(self):
        img = _two_colour_pattern()
        positions = {(0, 0, 0): np.array([[3.0, 4.0], [3.0, 4.0]]), (128, 128, 128): np.zeros((2, 0))}
        fig = ps.plot_sampled_positions(img, positions, colour_names={(0, 0, 0): "dye A"})
        assert fig is not None

    def test_without_names(self):
        img = _one_colour_pattern()
        positions = {(0, 0, 0): np.array([[3.0], [3.0]])}
        fig = ps.plot_sampled_positions(img, positions)
        assert fig is not None


class TestPlotPatternMasks:
    def test_multi_mask(self):
        img = _two_colour_pattern()
        masks = ps.mask_from_image(img)
        fig = ps.plot_pattern_masks(img, masks, colour_names={(0, 0, 0): "dye A"})
        assert fig is not None

    def test_zero_masks_single_panel(self):
        img = _flat_pattern()
        fig = ps.plot_pattern_masks(img, {})
        assert fig is not None


# ======================================================================
# duty_cycle / pool_size_for_density
# ======================================================================

class TestDutyCycle:
    def test_normal(self):
        assert ps.duty_cycle(0.01, 0.5) == pytest.approx(0.01 / 0.51)

    def test_both_zero(self):
        assert ps.duty_cycle(0.0, 0.0) == 0.0


class TestPoolSizeForDensity:
    def test_normal(self):
        assert ps.pool_size_for_density(1.0, 10.0, 100) == 1000

    def test_minimum_one(self):
        assert ps.pool_size_for_density(0.0, 10.0, 100) == 1


# ======================================================================
# blinking_state_schedule
# ======================================================================

class TestBlinkingStateSchedule:
    def test_invalid_modality_raises(self):
        with pytest.raises(ValueError, match="modality must be"):
            ps.blinking_state_schedule(5, 10, modality="bogus")

    def test_zero_candidates(self):
        out = ps.blinking_state_schedule(0, 10)
        assert out.shape == (10, 0)

    def test_storm_bleaches(self):
        rng = np.random.default_rng(0)
        out = ps.blinking_state_schedule(
            50, 200, on_rate=0.3, off_rate=0.3, modality="STORM",
            bleach_after_cycles=2, rng=rng,
        )
        assert out.shape == (200, 50)
        # By the end, bleached candidates should have stopped turning on --
        # total ON count late in the movie should be lower than early on.
        assert out[-1].sum() <= out[:5].any(axis=0).sum()

    def test_paint_no_bleach(self):
        rng = np.random.default_rng(0)
        out = ps.blinking_state_schedule(
            20, 100, on_rate=0.1, off_rate=0.2, modality="PAINT", rng=rng,
        )
        assert out.shape == (100, 20)
        assert out.any()  # some activity throughout, no permanent bleaching


# ======================================================================
# build_x0y0_track
# ======================================================================

class TestBuildX0y0Track:
    def test_normal(self):
        positions = np.array([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)
        on_state = np.array([[True, False], [False, True]])  # (2 frames, 2 candidates)
        track = ps.build_x0y0_track(positions, on_state)
        assert track.shape == (2, 2, 2)
        np.testing.assert_allclose(track[0, :, 0], [1.0, 3.0])
        np.testing.assert_allclose(track[0, :, 1], [1.0e7, 1.0e7])

    def test_shape_mismatch_raises(self):
        positions = np.zeros((2, 3))
        on_state = np.zeros((5, 2), dtype=bool)
        with pytest.raises(ValueError, match="does not match"):
            ps.build_x0y0_track(positions, on_state)


# ======================================================================
# per_frame_photon_budget
# ======================================================================

class TestPerFramePhotonBudget:
    def test_default_rng(self):
        out = ps.per_frame_photon_budget(10, photon_range=(100.0, 200.0))
        assert out.shape == (10,)
        assert np.all((out >= 100.0) & (out <= 200.0))

    def test_with_rng(self):
        out = ps.per_frame_photon_budget(5, rng=np.random.default_rng(0))
        assert out.shape == (5,)


# ======================================================================
# Scatterer / scatterer_spectrum
# ======================================================================

class TestScatterer:
    def test_init_and_repr(self):
        s = ps.Scatterer(638.0, label="gold NP")
        assert s.wavelength_nm == 638.0
        assert "gold NP" in repr(s)

    def test_eq_and_hash(self):
        a = ps.Scatterer(638.0, "gold")
        b = ps.Scatterer(638.0, "gold")
        c = ps.Scatterer(561.0, "gold")
        assert a == b
        assert a != c
        assert a != "not a scatterer"
        assert hash(a) == hash(b)

    def test_scatterer_spectrum(self, spectral_funcs):
        s = ps.Scatterer(638.0)
        wl = np.linspace(600.0, 700.0, 50)
        spec = ps.scatterer_spectrum(s, wl, spectral_funcs)
        assert spec.shape == wl.shape
        assert np.argmax(spec) == np.argmin(np.abs(wl - 638.0))


# ======================================================================
# NileRedEnvironment / nile_red_environment_spectrum
# ======================================================================

class TestNileRedEnvironment:
    def test_init_default_label(self):
        e = ps.NileRedEnvironment(620.0)
        assert "620" in e.label
        assert "620" in repr(e)

    def test_init_explicit_label(self):
        e = ps.NileRedEnvironment(620.0, label="lipid droplet")
        assert e.label == "lipid droplet"

    def test_eq_and_hash(self):
        a = ps.NileRedEnvironment(620.0, "x")
        b = ps.NileRedEnvironment(620.0, "x")
        c = ps.NileRedEnvironment(640.0, "x")
        assert a == b
        assert a != c
        assert a != "nope"
        assert hash(a) == hash(b)

    def test_spectrum_without_filters(self, nile_red_funcs):
        e = ps.NileRedEnvironment(620.0)
        wl = np.linspace(560.0, 700.0, 100)
        spec = ps.nile_red_environment_spectrum(e, wl, nile_red_funcs)
        assert spec.shape == wl.shape

    def test_spectrum_with_filters(self, nile_red_funcs, spectral_funcs):
        e = ps.NileRedEnvironment(620.0)
        wl = np.linspace(560.0, 700.0, 100)
        filter_spectra = spectral_funcs.get_dye_or_filter_data(
            names=[spectral_funcs.filter_names[0]], wavelength=wl, dye_or_filter=False,
        )
        spec = ps.nile_red_environment_spectrum(e, wl, nile_red_funcs, filter_spectra=filter_spectra)
        assert spec.shape == wl.shape


# ======================================================================
# simulate_acquisition
# ======================================================================

class TestSimulateAcquisition:
    def test_basic_single_dye(self, cam_maps):
        bayer_stack, gt, w, h, avg_wl = ps.simulate_acquisition(
            image=_one_colour_pattern(), colour_to_dye={(0, 0, 0): "ATTO 647N"},
            camera="ximea", pixel_size_um=0.069,
            gain_map=cam_maps["gain"], offset_map=cam_maps["offset"],
            variance_map=cam_maps["variance"], rqe_map=cam_maps["rqe"],
            n_frames=2, density_per_um2=0.05, background_photons=10.0,
            rng=np.random.default_rng(0),
        )
        assert bayer_stack.shape[0] == 2
        assert len(gt) > 0
        assert avg_wl > 0

    def test_default_rng(self, cam_maps):
        # n_frames=2, not 1: gen_camera_image_stack np.squeeze()s a
        # single-frame output, dropping the leading axis entirely.
        bayer_stack, gt, w, h, avg_wl = ps.simulate_acquisition(
            image=_one_colour_pattern(), colour_to_dye={(0, 0, 0): "ATTO 647N"},
            camera="ximea", pixel_size_um=0.069,
            gain_map=cam_maps["gain"], offset_map=cam_maps["offset"],
            variance_map=cam_maps["variance"], rqe_map=cam_maps["rqe"],
            n_frames=2, density_per_um2=0.05, background_photons=10.0,
        )
        assert bayer_stack.shape[0] == 2

    def test_all_masks_empty_raises(self, cam_maps):
        with pytest.raises(ValueError, match="No candidate localisations"):
            ps.simulate_acquisition(
                image=_flat_pattern(), colour_to_dye={(200, 50, 50): "ATTO 647N"},
                camera="ximea", pixel_size_um=0.069,
                gain_map=cam_maps["gain"], offset_map=cam_maps["offset"],
                variance_map=cam_maps["variance"], rqe_map=cam_maps["rqe"],
                n_frames=1, density_per_um2=0.05,
                rng=np.random.default_rng(0),
            )

    def test_colour_not_in_masks_skipped(self, cam_maps):
        # (9, 9, 9) never appears in the pattern -> positions_by_colour.get
        # falls back to the empty-array default, hitting the `continue`.
        bayer_stack, gt, w, h, avg_wl = ps.simulate_acquisition(
            image=_one_colour_pattern(),
            colour_to_dye={(0, 0, 0): "ATTO 647N", (9, 9, 9): "ATTO 647N"},
            camera="ximea", pixel_size_um=0.069,
            gain_map=cam_maps["gain"], offset_map=cam_maps["offset"],
            variance_map=cam_maps["variance"], rqe_map=cam_maps["rqe"],
            n_frames=2, density_per_um2=0.05, background_photons=10.0,
            rng=np.random.default_rng(0),
        )
        assert bayer_stack.shape[0] == 2

    def test_two_colours_same_dye_merged(self, cam_maps):
        bayer_stack, gt, w, h, avg_wl = ps.simulate_acquisition(
            image=_two_colour_pattern(),
            colour_to_dye={(0, 0, 0): "ATTO 647N", (128, 128, 128): "ATTO 647N"},
            camera="ximea", pixel_size_um=0.069,
            gain_map=cam_maps["gain"], offset_map=cam_maps["offset"],
            variance_map=cam_maps["variance"], rqe_map=cam_maps["rqe"],
            n_frames=1, density_per_um2=0.05, background_photons=10.0,
            rng=np.random.default_rng(0),
        )
        assert set(gt["dye"]) == {"ATTO 647N"}

    def test_scatterer_dye(self, cam_maps):
        bayer_stack, gt, w, h, avg_wl = ps.simulate_acquisition(
            image=_one_colour_pattern(),
            colour_to_dye={(0, 0, 0): ps.Scatterer(638.0, label="gold NP")},
            camera="ximea", pixel_size_um=0.069,
            gain_map=cam_maps["gain"], offset_map=cam_maps["offset"],
            variance_map=cam_maps["variance"], rqe_map=cam_maps["rqe"],
            n_frames=1, density_per_um2=0.05, background_photons=10.0,
            rng=np.random.default_rng(0),
        )
        assert set(gt["dye"]) == {"gold NP"}

    def test_nile_red_environment_dye_no_filters(self, cam_maps):
        bayer_stack, gt, w, h, avg_wl = ps.simulate_acquisition(
            image=_one_colour_pattern(),
            colour_to_dye={(0, 0, 0): ps.NileRedEnvironment(620.0)},
            camera="ximea", pixel_size_um=0.069,
            gain_map=cam_maps["gain"], offset_map=cam_maps["offset"],
            variance_map=cam_maps["variance"], rqe_map=cam_maps["rqe"],
            n_frames=2, density_per_um2=0.05, background_photons=10.0,
            rng=np.random.default_rng(0),
        )
        assert bayer_stack.shape[0] == 2

    def test_nile_red_environment_dye_with_filters(self, cam_maps):
        # Must pick a filter that actually overlaps the 620nm environment's
        # emission band -- an out-of-band filter (e.g. filter_names[0], a
        # ~900nm bandpass) zeroes the spectrum entirely, NaN-ing the
        # trapz-normalisation and everything downstream (a real but
        # unrelated degenerate-input behaviour, not what this test targets).
        bayer_stack, gt, w, h, avg_wl = ps.simulate_acquisition(
            image=_one_colour_pattern(),
            colour_to_dye={(0, 0, 0): ps.NileRedEnvironment(620.0)},
            camera="ximea", pixel_size_um=0.069,
            gain_map=cam_maps["gain"], offset_map=cam_maps["offset"],
            variance_map=cam_maps["variance"], rqe_map=cam_maps["rqe"],
            n_frames=2, density_per_um2=0.05, background_photons=10.0,
            nile_red_filter_names=["chroma-d620-40m"],
            rng=np.random.default_rng(0),
        )
        assert bayer_stack.shape[0] == 2

    def test_drift_nm(self, cam_maps):
        drift = np.zeros((2, 2))
        drift[1] = [50.0, -30.0]
        bayer_stack, gt, w, h, avg_wl = ps.simulate_acquisition(
            image=_one_colour_pattern(), colour_to_dye={(0, 0, 0): "ATTO 647N"},
            camera="ximea", pixel_size_um=0.069,
            gain_map=cam_maps["gain"], offset_map=cam_maps["offset"],
            variance_map=cam_maps["variance"], rqe_map=cam_maps["rqe"],
            n_frames=2, density_per_um2=0.05, background_photons=10.0,
            drift_nm=drift, rng=np.random.default_rng(0),
        )
        assert bayer_stack.shape[0] == 2

    def test_min_separation_nm(self, cam_maps):
        bayer_stack, gt, w, h, avg_wl = ps.simulate_acquisition(
            image=_one_colour_pattern(), colour_to_dye={(0, 0, 0): "ATTO 647N"},
            camera="ximea", pixel_size_um=0.069,
            gain_map=cam_maps["gain"], offset_map=cam_maps["offset"],
            variance_map=cam_maps["variance"], rqe_map=cam_maps["rqe"],
            n_frames=2, density_per_um2=0.05, background_photons=10.0,
            min_separation_nm=200.0, rng=np.random.default_rng(0),
        )
        assert bayer_stack.shape[0] == 2

    def test_frame_chunk_size_evenly_divides(self, cam_maps):
        bayer_stack, gt, w, h, avg_wl = ps.simulate_acquisition(
            image=_one_colour_pattern(), colour_to_dye={(0, 0, 0): "ATTO 647N"},
            camera="ximea", pixel_size_um=0.069,
            gain_map=cam_maps["gain"], offset_map=cam_maps["offset"],
            variance_map=cam_maps["variance"], rqe_map=cam_maps["rqe"],
            n_frames=4, density_per_um2=0.05, background_photons=10.0,
            frame_chunk_size=2, rng=np.random.default_rng(0),
        )
        assert bayer_stack.shape[0] == 4

    def test_frame_chunk_size_with_singleton_remainder(self, cam_maps):
        # 3 frames, chunk_size=2 -> chunks of [2, 1]; the last chunk's
        # single-frame output loses its leading axis and must be restored.
        bayer_stack, gt, w, h, avg_wl = ps.simulate_acquisition(
            image=_one_colour_pattern(), colour_to_dye={(0, 0, 0): "ATTO 647N"},
            camera="ximea", pixel_size_um=0.069,
            gain_map=cam_maps["gain"], offset_map=cam_maps["offset"],
            variance_map=cam_maps["variance"], rqe_map=cam_maps["rqe"],
            n_frames=3, density_per_um2=0.05, background_photons=10.0,
            frame_chunk_size=2, rng=np.random.default_rng(0),
        )
        assert bayer_stack.shape[0] == 3

    def test_frame_chunk_size_dtype_upgrade(self, cam_maps, monkeypatch):
        # Force the second chunk to come back in a wider dtype than the
        # first, exercising the accumulator dtype-upgrade branch.
        import pyS3M.simulation.multicolour as mc

        real_gen = mc.MultiC_Sim_Funcs_Refactored.gen_camera_image_stack
        state = {"n": 0}

        def fake_gen(self, *args, **kwargs):
            bayer, a, b = real_gen(self, *args, **kwargs)
            state["n"] += 1
            if state["n"] == 1:
                bayer = bayer.astype(np.uint8)
            else:
                bayer = bayer.astype(np.float32)
            return bayer, a, b

        monkeypatch.setattr(mc.MultiC_Sim_Funcs_Refactored, "gen_camera_image_stack", fake_gen)
        bayer_stack, gt, w, h, avg_wl = ps.simulate_acquisition(
            image=_one_colour_pattern(), colour_to_dye={(0, 0, 0): "ATTO 647N"},
            camera="ximea", pixel_size_um=0.069,
            gain_map=cam_maps["gain"], offset_map=cam_maps["offset"],
            variance_map=cam_maps["variance"], rqe_map=cam_maps["rqe"],
            n_frames=2, density_per_um2=0.05, background_photons=10.0,
            frame_chunk_size=1, rng=np.random.default_rng(0),
        )
        assert bayer_stack.dtype == np.float32
