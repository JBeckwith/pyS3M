"""Smoke tests for PlottingBase.py.

Per this project's Tier 3 testing policy: these check that each plotting
function/method runs without raising, returns the right type/shape, and
exercises its branches -- not pixel-level image correctness (that's what the
example notebooks' by-eye verification is for). Data throughout is
deliberately tiny (4-50 points, 4-16px images) to keep the suite fast.
"""

import sys
import importlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from pyS3M.PlottingBase import (
    PublicationConstants,
    PlottingConfig,
    BasePlotter,
    ImagePlotMixin,
    TernaryPlotMixin,
    DatashaderMixin,
    PublicationPlotter,
    AnalysisPlotter,
    plot_bayer_pattern,
    _safe_tight_layout,
)


@pytest.fixture
def plotter():
    return PublicationPlotter()


@pytest.fixture
def analysis_plotter():
    return AnalysisPlotter()


def _mpltern_ax():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="ternary")
    return fig, ax


# ======================================================================
# Module-level helpers
# ======================================================================

class TestMplternUnavailable:
    def test_import_warning_on_missing_mpltern(self, monkeypatch):
        import pyS3M.PlottingBase as pb_mod

        monkeypatch.setitem(sys.modules, "mpltern", None)
        try:
            with pytest.warns(ImportWarning, match="mpltern not available"):
                importlib.reload(pb_mod)
            assert pb_mod.MPLTERN_AVAILABLE is False
        finally:
            # Restore real mpltern *before* monkeypatch's own teardown so the
            # module is back in its normal, working state for every other
            # test in this file (mirrors the ruptures-guard technique used
            # for StepDetector.py's equivalent optional-import guard).
            monkeypatch.undo()
            importlib.reload(pb_mod)


class TestPlotBayerPattern:
    def test_default(self):
        fig, ax = plt.subplots()
        out = plot_bayer_pattern(ax)
        assert out is ax
        plt.close(fig)

    @pytest.mark.parametrize("pattern", ["BGGR", "GBRG", "RGGB", "GRBG"])
    def test_all_patterns(self, pattern):
        fig, ax = plt.subplots()
        plot_bayer_pattern(ax, pattern=pattern, size=4)
        plt.close(fig)

    def test_lines_and_marker_dark_background(self):
        fig, ax = plt.subplots()
        plot_bayer_pattern(
            ax, size=4, vline=2.0, hline=2.0, marker_pos=(1, 1),
            dark_background=True,
        )
        plt.close(fig)

    def test_light_background_custom_line_color(self):
        fig, ax = plt.subplots()
        plot_bayer_pattern(
            ax, size=4, vline=2.0, hline=2.0, dark_background=False,
            line_color="red",
        )
        plt.close(fig)


class TestSafeTightLayout:
    def test_normal(self):
        fig, ax = plt.subplots()
        _safe_tight_layout(fig)
        plt.close(fig)

    def test_swallows_exception(self, monkeypatch):
        fig, ax = plt.subplots()

        def _raise():
            raise RuntimeError("forced tight_layout failure")

        monkeypatch.setattr(fig, "tight_layout", _raise)
        _safe_tight_layout(fig)  # must not raise
        plt.close(fig)


# ======================================================================
# PlottingConfig
# ======================================================================

class TestPlottingConfig:
    def test_standard_defaults(self):
        cfg = PlottingConfig()
        assert cfg.font_size == PublicationConstants.STANDARD_FONT_SIZE

    def test_poster_mode(self):
        cfg = PlottingConfig(poster_mode=True)
        assert cfg.font_size == PublicationConstants.POSTER_FONT_SIZE
        assert cfg.tick_labelsize == PublicationConstants.POSTER_TICK_LABELSIZE

    def test_dark_background(self):
        cfg = PlottingConfig(dark_background=True)
        assert cfg.DEFAULT_GRID_COLOR == "white"
        assert cfg.DEFAULT_SCALEBAR_COLOR == "white"
        plt.style.use("default")  # restore for subsequent tests


# ======================================================================
# BasePlotter core methods
# ======================================================================

class TestBasePlotterCore:
    def test_create_figure_default_and_custom(self, plotter):
        fig, ax = plotter.create_figure()
        plt.close(fig)
        fig, ax = plotter.create_figure(figsize=(2, 2), dpi=50, facecolor="white", edgecolor="black")
        plt.close(fig)

    def test_one_column_plot_single_panel(self, plotter):
        fig, ax = plotter.one_column_plot()
        plt.close(fig)

    def test_one_column_plot_multi_panel(self, plotter):
        fig, axs = plotter.one_column_plot(npanels=2, ratios=[1, 2], height=5, width=2.5)
        assert len(axs) == 2
        plt.close(fig)

    def test_one_column_plot_ratio_mismatch_raises(self, plotter):
        with pytest.raises(ValueError, match="Number of ratios"):
            plotter.one_column_plot(npanels=2, ratios=[1])

    def test_one_column_plot_width_warns(self, plotter):
        with pytest.warns(UserWarning, match="exceeds one-column"):
            fig, ax = plotter.one_column_plot(width=10)
        plt.close(fig)

    def test_one_column_plot_height_warns(self, plotter):
        with pytest.warns(UserWarning, match="exceeds maximum"):
            fig, ax = plotter.one_column_plot(height=20)
        plt.close(fig)

    def test_two_column_plot_shapes(self, plotter):
        fig, ax = plotter.two_column_plot()
        plt.close(fig)
        fig, axs = plotter.two_column_plot(nrows=1, ncols=2)
        assert len(axs) == 2
        plt.close(fig)
        fig, axs = plotter.two_column_plot(nrows=2, ncols=1)
        assert len(axs) == 2
        plt.close(fig)
        fig, axs = plotter.two_column_plot(nrows=2, ncols=2, height_ratios=[1, 1], width_ratios=[1, 1])
        assert axs.shape == (2, 2)
        plt.close(fig)

    def test_two_column_plot_big(self, plotter):
        fig, ax = plotter.two_column_plot(big=True)
        plt.close(fig)
        fig, axs = plotter.two_column_plot(nrows=2, big=True)
        plt.close(fig)

    def test_two_column_plot_width_warns(self, plotter):
        with pytest.warns(UserWarning, match="exceeds two-column"):
            fig, ax = plotter.two_column_plot(width=20)
        plt.close(fig)

    def test_two_column_plot_height_warns(self, plotter):
        with pytest.warns(UserWarning, match="exceeds maximum"):
            fig, ax = plotter.two_column_plot(height=20)
        plt.close(fig)

    def test_two_column_plot_subplot_kw(self, plotter):
        fig, ax = plotter.two_column_plot(subplot_kw={"projection": "polar"})
        plt.close(fig)

    def test_setup_axis_variants(self, plotter):
        fig, ax = plt.subplots()
        plotter.setup_axis(ax, xlabel="x", ylabel="y", title="t", grid=True,
                            grid_alpha=0.5, equal_aspect=True)
        plt.close(fig)
        fig, ax = plt.subplots()
        plotter.setup_axis(ax, grid=False, spine_style="none")
        plt.close(fig)
        fig, ax = plt.subplots()
        plotter.setup_axis(ax, spine_style="left-bottom")
        plt.close(fig)

    def test_create_image_plot_auto_and_explicit(self, plotter):
        data = np.arange(16, dtype=float).reshape(4, 4)
        fig, ax = plt.subplots()
        plotter.create_image_plot(ax, data)
        plt.close(fig)
        fig, ax = plt.subplots()
        plotter.create_image_plot(ax, data, vmin=0, vmax=15, cmap="gray", origin="upper", interpolation="nearest")
        plt.close(fig)

    def test_add_colorbar(self, plotter):
        fig, ax = plt.subplots()
        im = ax.imshow(np.arange(16).reshape(4, 4))
        cbar = plotter.add_colorbar(im, ax, label="counts")
        assert cbar is not None
        plt.close(fig)

    def test_add_scalebar_nm_and_um_labels(self, plotter):
        fig, ax = plt.subplots()
        plotter.add_scalebar(ax, pixelsize=69.0, length_nm=500.0)
        plotter.add_scalebar(ax, pixelsize=69.0, length_nm=2000.0)
        plotter.add_scalebar(ax, pixelsize=69.0, length_nm=500.0, label="custom", color="red", fontsize=8)
        plt.close(fig)

    def test_line_plot(self, plotter):
        x = np.arange(5.0)
        y = x ** 2
        fig, ax = plt.subplots()
        plotter.line_plot(ax, x, y, label="squared", xlim=(0, 5), ylim=(0, 25))
        plt.close(fig)
        fig, ax = plt.subplots()
        plotter.line_plot(ax, x, y, grid=False)
        plt.close(fig)

    def test_line_plot_with_error(self, plotter):
        x = np.arange(5.0)
        y = x ** 2
        yerr = np.ones(5)
        fig, ax = plt.subplots()
        plotter.line_plot_with_error(ax, x, y, yerr, label="err", xlim=(0, 5), ylim=(0, 30))
        plt.close(fig)
        fig, ax = plt.subplots()
        plotter.line_plot_with_error(ax, x, y, yerr, grid=False)
        plt.close(fig)

    def test_scatter_plot_default_colors(self, plotter):
        x = np.arange(5.0)
        y = x ** 2
        fig, ax = plt.subplots()
        plotter.scatter_plot(ax, x, y, label="pts", xlim=(0, 5), ylim=(0, 25))
        plt.close(fig)

    def test_scatter_plot_custom_facecolor(self, plotter):
        x = np.arange(5.0)
        y = x ** 2
        fig, ax = plt.subplots()
        plotter.scatter_plot(ax, x, y, facecolor="blue", edgecolor="red", rasterized=True, grid=False)
        plt.close(fig)

    def test_histogram_plot(self, plotter):
        data = np.linspace(0, 1, 50)
        fig, ax = plt.subplots()
        plotter.histogram_plot(ax, data, label="h", xlim=(0, 1), ylim=(0, 50), density=True)
        plt.close(fig)
        fig, ax = plt.subplots()
        plotter.histogram_plot(ax, data, grid=False)
        plt.close(fig)

    def test_image_plot_axes_off_with_title_colorbar_scalebar(self, plotter):
        data = np.arange(16, dtype=float).reshape(4, 4)
        fig, ax = plt.subplots()
        ax_out, im = plotter.image_plot(
            ax, data, title="micrograph", colorbar=True, colorbar_label="ph",
            scalebar=True, pixelsize=69.0, scalebarsize=100.0,
        )
        assert im is not None
        plt.close(fig)

    def test_image_plot_axes_on_no_title(self, plotter):
        data = np.arange(16, dtype=float).reshape(4, 4)
        fig, ax = plt.subplots()
        plotter.image_plot(ax, data, show_axes=True, xlabel="x", ylabel="y", vmin=0, vmax=15)
        plt.close(fig)

    def test_colour_image_plot_on_and_off(self, plotter):
        data = np.random.rand(4, 4, 3)
        fig, ax = plt.subplots()
        plotter.colour_image_plot(ax, data, label="lbl", cbar="on", sbar="on")
        plt.close(fig)
        fig, ax = plt.subplots()
        plotter.colour_image_plot(ax, data, cbar="off", sbar="off", vmin=0.0, vmax=1.0)
        plt.close(fig)

    def test_save_or_show_saves_and_closes(self, plotter, tmp_path):
        fig, ax = plt.subplots()
        out = tmp_path / "out.png"
        plotter.save_or_show(fig, save_path=str(out), show=False, dpi=50)
        assert out.exists()

    def test_save_or_show_no_save_show_false(self, plotter):
        fig, ax = plt.subplots()
        plotter.save_or_show(fig, show=False)  # neither saves nor shows, just closes


# ======================================================================
# ImagePlotMixin
# ======================================================================

class TestCreateDarkToColorCmap:
    def test_named_bright_color(self):
        cmap = ImagePlotMixin._create_dark_to_color_cmap("cyan")
        assert cmap is not None

    def test_arbitrary_matplotlib_color(self):
        cmap = ImagePlotMixin._create_dark_to_color_cmap("darkred")
        assert cmap is not None

    def test_colormap_name_fallback(self):
        cmap = ImagePlotMixin._create_dark_to_color_cmap("hot")
        assert cmap is not None


class TestMultichannelOverlayGaps:
    """Extra branches not already hit by test_multichannel_overlay.py."""

    def test_alphas_deprecated_warns(self, plotter):
        img1 = np.random.rand(4, 4)
        img2 = np.random.rand(4, 4)
        fig, ax = plt.subplots()
        with pytest.warns(DeprecationWarning):
            plotter.multichannel_overlay_plot(ax, [img1, img2], alphas=[1.0, 1.0])
        plt.close(fig)

    def test_mismatched_cmaps_raises(self, plotter):
        img1 = np.random.rand(4, 4)
        img2 = np.random.rand(4, 4)
        fig, ax = plt.subplots()
        with pytest.raises(ValueError, match="colormaps"):
            plotter.multichannel_overlay_plot(ax, [img1, img2], cmaps=["cyan"])
        plt.close(fig)

    def test_mismatched_brightness_boost_length_raises(self, plotter):
        img1 = np.random.rand(4, 4)
        img2 = np.random.rand(4, 4)
        fig, ax = plt.subplots()
        with pytest.raises(ValueError, match="brightness_boost"):
            plotter.multichannel_overlay_plot(ax, [img1, img2], brightness_boost=[1.0])
        plt.close(fig)

    def test_nonpositive_brightness_boost_raises(self, plotter):
        img1 = np.random.rand(4, 4)
        img2 = np.random.rand(4, 4)
        fig, ax = plt.subplots()
        with pytest.raises(ValueError, match="Brightness boost"):
            plotter.multichannel_overlay_plot(ax, [img1, img2], brightness_boost=[1.0, 0.0])
        plt.close(fig)

    def test_mismatched_vmins_length_raises(self, plotter):
        img1 = np.random.rand(4, 4)
        img2 = np.random.rand(4, 4)
        fig, ax = plt.subplots()
        with pytest.raises(ValueError, match="vmin"):
            plotter.multichannel_overlay_plot(ax, [img1, img2], vmins=[0.0])
        plt.close(fig)

    def test_mismatched_vmaxs_length_raises(self, plotter):
        img1 = np.random.rand(4, 4)
        img2 = np.random.rand(4, 4)
        fig, ax = plt.subplots()
        with pytest.raises(ValueError, match="vmax"):
            plotter.multichannel_overlay_plot(ax, [img1, img2], vmaxs=[1.0])
        plt.close(fig)


class TestMakeAnimatedGif:
    def test_grayscale_with_colorbar_and_label(self, plotter, tmp_path):
        stack = np.random.rand(2, 4, 4) * 100
        out = tmp_path / "anim.gif"
        plotter.make_animated_gif(
            stack, str(out), vmin=0, vmax=100, label="frame", cbar=True,
            width=1.5, height=1.5, fps=5, dpi=50,
        )
        assert out.exists()

    def test_rgb_stack(self, plotter, tmp_path):
        stack = (np.random.rand(2, 4, 4, 3) * 255).astype(np.uint8)
        out = tmp_path / "anim_rgb.gif"
        plotter.make_animated_gif(stack, str(out), width=1.5, height=1.5, fps=5, dpi=50)
        assert out.exists()

    def test_rgb_stack_float_normalizes(self, plotter, tmp_path):
        stack = np.random.rand(2, 4, 4, 3).astype(np.float64)
        out = tmp_path / "anim_rgb_float.gif"
        plotter.make_animated_gif(
            stack, str(out), vmin=0.0, vmax=1.0, width=1.5, height=1.5, fps=5, dpi=50,
        )
        assert out.exists()


class TestMakeAnimatedGifMultipanel:
    def test_pattern_and_image_panels(self, plotter, tmp_path):
        fig, axs = plt.subplots(1, 2, squeeze=False)
        plot_types = np.array([["pattern", "image"]])
        xpositions = np.array([2.0, 3.0])
        ypositions = np.array([2.0, 3.0])
        # Column index maps to dye index -- column 1 ("image") needs
        # images_for_figures[1] to exist, so at least 2 "dyes" here even
        # though column 0 ("pattern") never reads this array.
        images = np.random.rand(2, 2, 4, 4)  # (n_dyes, n_frames, H, W)
        out = tmp_path / "multipanel.gif"
        plotter.make_animated_gif_multipanel(
            fig, axs, plot_types, xpositions, ypositions, images,
            n_pixels=4, filename=str(out), fps=5, dpi=50,
            marker_color=["cornflowerblue", "orange"],
        )
        assert out.exists()

    def test_off_panel_and_single_marker_color(self, plotter, tmp_path):
        fig, axs = plt.subplots(1, 2, squeeze=False)
        plot_types = np.array([["pattern", "off"]])
        xpositions = np.array([2.0])
        ypositions = np.array([2.0])
        images = np.random.rand(1, 1, 4, 4)
        out = tmp_path / "multipanel_off.gif"
        plotter.make_animated_gif_multipanel(
            fig, axs, plot_types, xpositions, ypositions, images,
            n_pixels=4, filename=str(out), fps=5, dpi=50, marker_color="white",
        )
        assert out.exists()


# ======================================================================
# TernaryPlotMixin
# ======================================================================

class TestCreateTernaryPlot:
    def test_basic(self, plotter):
        R = np.array([0.6, 0.2, 0.3])
        G = np.array([0.2, 0.6, 0.3])
        B = np.array([0.2, 0.2, 0.4])
        fig, ax = plotter.create_ternary_plot(R, G, B, title="t")
        plt.close(fig)

    def test_with_colors_no_grid_no_black_bg(self, plotter):
        R = np.array([0.6, 0.2, 0.3])
        G = np.array([0.2, 0.6, 0.3])
        B = np.array([0.2, 0.2, 0.4])
        colors = np.array(["red", "green", "blue"])
        fig, ax = plotter.create_ternary_plot(
            R, G, B, colors=colors, show_grid=False, black_background=False,
        )
        plt.close(fig)

    def test_mismatched_lengths_raises(self, plotter):
        with pytest.raises(ValueError, match="same length"):
            plotter.create_ternary_plot(np.array([1.0]), np.array([1.0, 2.0]), np.array([1.0]))

    def test_mpltern_import_error(self, plotter, monkeypatch):
        monkeypatch.setitem(sys.modules, "mpltern", None)
        with pytest.raises(ImportError, match="mpltern"):
            plotter.create_ternary_plot(np.array([0.3]), np.array([0.3]), np.array([0.4]))


class TestCreateTernaryDensity:
    def test_basic_needs_normalization(self, plotter):
        rng = np.random.default_rng(0)
        R = rng.uniform(1, 5, 30)
        G = rng.uniform(1, 5, 30)
        B = rng.uniform(1, 5, 30)
        fig, ax = plotter.create_ternary_density(R, G, B, gridsize=5, title="density")
        plt.close(fig)

    def test_log_scale_custom_labels_no_colorbar(self, plotter):
        rng = np.random.default_rng(1)
        R = rng.uniform(0.1, 0.4, 30)
        G = rng.uniform(0.1, 0.4, 30)
        B = 1.0 - R - G
        fig, ax = plotter.create_ternary_density(
            R, G, B, gridsize=5, log_scale=True, show_colorbar=False,
            labels={"R": "Red ch"}, show_grid=False,
        )
        plt.close(fig)

    def test_mismatched_lengths_raises(self, plotter):
        with pytest.raises(ValueError, match="same length"):
            plotter.create_ternary_density(np.array([1.0]), np.array([1.0, 2.0]), np.array([1.0]))

    def test_mpltern_import_error(self, plotter, monkeypatch):
        monkeypatch.setitem(sys.modules, "mpltern", None)
        with pytest.raises(ImportError, match="mpltern"):
            plotter.create_ternary_density(np.array([0.3]), np.array([0.3]), np.array([0.4]))


class TestPlotTernaryKdeContours:
    @staticmethod
    def _rgb(n=30, seed=0):
        rng = np.random.default_rng(seed)
        R = rng.uniform(0.2, 0.4, n)
        G = rng.uniform(0.2, 0.4, n)
        B = 1.0 - R - G
        return R, G, B

    def test_default_confidence_levels(self, plotter):
        fig, ax = _mpltern_ax()
        R, G, B = self._rgb()
        plotter.plot_ternary_kde_contours(ax, R, G, B, color="red", label="dye")
        plt.close(fig)

    def test_auto_levels(self, plotter):
        fig, ax = _mpltern_ax()
        R, G, B = self._rgb(n=50, seed=1)
        plotter.plot_ternary_kde_contours(ax, R, G, B, levels="auto")
        plt.close(fig)

    def test_int_levels(self, plotter):
        fig, ax = _mpltern_ax()
        R, G, B = self._rgb(n=50, seed=2)
        plotter.plot_ternary_kde_contours(ax, R, G, B, levels=4)
        plt.close(fig)

    def test_general_confidence_level_branch(self, plotter):
        fig, ax = _mpltern_ax()
        R, G, B = self._rgb(n=50, seed=3)
        plotter.plot_ternary_kde_contours(
            ax, R, G, B, levels=[0.5, 0.75, 0.9],
            linewidths=[1.0], linestyles=["dashed"],
        )
        plt.close(fig)

    def test_silverman_and_float_bandwidth(self, plotter):
        fig, ax = _mpltern_ax()
        R, G, B = self._rgb(n=30, seed=4)
        plotter.plot_ternary_kde_contours(ax, R, G, B, bandwidth="silverman")
        plotter.plot_ternary_kde_contours(ax, R, G, B, bandwidth=0.3)
        plt.close(fig)

    def test_invalid_bandwidth_returns_none(self, plotter):
        fig, ax = _mpltern_ax()
        R, G, B = self._rgb()
        result = plotter.plot_ternary_kde_contours(ax, R, G, B, bandwidth="nonsense")
        assert result is None
        plt.close(fig)

    def test_too_few_points_returns_none(self, plotter):
        fig, ax = _mpltern_ax()
        R, G, B = self._rgb(n=5, seed=5)
        result = plotter.plot_ternary_kde_contours(ax, R, G, B)
        assert result is None
        plt.close(fig)

    def test_nan_values_removed(self, plotter):
        fig, ax = _mpltern_ax()
        R, G, B = self._rgb(n=30, seed=6)
        R = R.copy()
        R[0] = np.nan
        plotter.plot_ternary_kde_contours(ax, R, G, B)
        plt.close(fig)

    def test_mismatched_lengths_raises(self, plotter):
        fig, ax = _mpltern_ax()
        with pytest.raises(ValueError, match="same length"):
            plotter.plot_ternary_kde_contours(ax, np.array([1.0]), np.array([1.0, 2.0]), np.array([1.0]))
        plt.close(fig)

    def test_kde_creation_exception_returns_none(self, plotter):
        fig, ax = _mpltern_ax()
        # All-identical points -> singular covariance -> gaussian_kde raises internally.
        R = np.full(15, 0.3)
        G = np.full(15, 0.3)
        B = np.full(15, 0.4)
        result = plotter.plot_ternary_kde_contours(ax, R, G, B)
        assert result is None
        plt.close(fig)

    def test_unnormalized_data_gets_normalized(self, plotter):
        fig, ax = _mpltern_ax()
        rng = np.random.default_rng(7)
        R = rng.uniform(1, 5, 30)
        G = rng.uniform(1, 5, 30)
        B = rng.uniform(1, 5, 30)
        plotter.plot_ternary_kde_contours(ax, R, G, B)
        plt.close(fig)

    def test_kde_evaluation_exception_returns_none(self, plotter, monkeypatch):
        from scipy.stats import gaussian_kde
        fig, ax = _mpltern_ax()
        R, G, B = self._rgb(n=30, seed=8)

        def _raise(self, *a, **kw):
            raise RuntimeError("forced KDE evaluation failure")

        monkeypatch.setattr(gaussian_kde, "__call__", _raise)
        result = plotter.plot_ternary_kde_contours(ax, R, G, B)
        assert result is None
        plt.close(fig)

    def test_auto_levels_too_few_nonzero_returns_none(self, plotter):
        fig, ax = _mpltern_ax()
        R, G, B = self._rgb(n=30, seed=9)
        result = plotter.plot_ternary_kde_contours(ax, R, G, B, levels="auto", grid_resolution=2)
        assert result is None
        plt.close(fig)

    def test_list_levels_too_few_nonzero_uses_all_valid(self, plotter):
        fig, ax = _mpltern_ax()
        R, G, B = self._rgb(n=30, seed=10)
        plotter.plot_ternary_kde_contours(ax, R, G, B, grid_resolution=2)
        plt.close(fig)

    def test_general_confidence_level_clamps_index(self, plotter):
        fig, ax = _mpltern_ax()
        R, G, B = self._rgb(n=50, seed=11)
        # conf > 1.0 pushes searchsorted's index past the end of sorted_kde
        # (cumsum_normalized's last entry is exactly 1.0), exercising the
        # idx-clamp branch.
        plotter.plot_ternary_kde_contours(ax, R, G, B, levels=[1.5])
        plt.close(fig)

    def test_contour_plotting_exception_returns_none(self, plotter, monkeypatch):
        fig, ax = _mpltern_ax()
        R, G, B = self._rgb(n=30, seed=12)

        def _raise(*a, **kw):
            raise RuntimeError("forced tricontour failure")

        monkeypatch.setattr(ax, "tricontour", _raise)
        result = plotter.plot_ternary_kde_contours(ax, R, G, B)
        assert result is None
        plt.close(fig)


class TestPlotTernaryKde:
    @staticmethod
    def _rgb(n=30, seed=0):
        rng = np.random.default_rng(seed)
        R = rng.uniform(0.2, 0.4, n)
        G = rng.uniform(0.2, 0.4, n)
        B = 1.0 - R - G
        return R, G, B

    def test_default(self, plotter):
        fig, ax = _mpltern_ax()
        R, G, B = self._rgb()
        cf = plotter.plot_ternary_kde(ax, R, G, B, grid_resolution=20)
        assert cf is not None
        plt.close(fig)

    def test_silverman_and_float_bandwidth_no_colorbar(self, plotter):
        fig, ax = _mpltern_ax()
        R, G, B = self._rgb(seed=1)
        plotter.plot_ternary_kde(ax, R, G, B, bandwidth="silverman", grid_resolution=20, show_colorbar=False)
        fig2, ax2 = _mpltern_ax()
        plotter.plot_ternary_kde(ax2, R, G, B, bandwidth=0.3, grid_resolution=20)
        plt.close(fig)
        plt.close(fig2)

    def test_invalid_bandwidth_returns_none(self, plotter):
        fig, ax = _mpltern_ax()
        R, G, B = self._rgb()
        assert plotter.plot_ternary_kde(ax, R, G, B, bandwidth="nonsense") is None
        plt.close(fig)

    def test_too_few_points_returns_none(self, plotter):
        fig, ax = _mpltern_ax()
        R, G, B = self._rgb(n=5, seed=2)
        assert plotter.plot_ternary_kde(ax, R, G, B) is None
        plt.close(fig)

    def test_nan_values_removed(self, plotter):
        fig, ax = _mpltern_ax()
        R, G, B = self._rgb(seed=3)
        R = R.copy()
        R[0] = np.inf
        plotter.plot_ternary_kde(ax, R, G, B, grid_resolution=20)
        plt.close(fig)

    def test_mismatched_lengths_raises(self, plotter):
        fig, ax = _mpltern_ax()
        with pytest.raises(ValueError, match="same length"):
            plotter.plot_ternary_kde(ax, np.array([1.0]), np.array([1.0, 2.0]), np.array([1.0]))
        plt.close(fig)

    def test_kde_creation_exception_returns_none(self, plotter):
        fig, ax = _mpltern_ax()
        R = np.full(15, 0.3)
        G = np.full(15, 0.3)
        B = np.full(15, 0.4)
        assert plotter.plot_ternary_kde(ax, R, G, B) is None
        plt.close(fig)

    def test_unnormalized_data_gets_normalized(self, plotter):
        fig, ax = _mpltern_ax()
        rng = np.random.default_rng(20)
        R = rng.uniform(1, 5, 30)
        G = rng.uniform(1, 5, 30)
        B = rng.uniform(1, 5, 30)
        plotter.plot_ternary_kde(ax, R, G, B, grid_resolution=20)
        plt.close(fig)

    def test_kde_evaluation_exception_returns_none(self, plotter, monkeypatch):
        from scipy.stats import gaussian_kde
        fig, ax = _mpltern_ax()
        R, G, B = self._rgb(seed=21)

        def _raise(self, *a, **kw):
            raise RuntimeError("forced KDE evaluation failure")

        monkeypatch.setattr(gaussian_kde, "__call__", _raise)
        assert plotter.plot_ternary_kde(ax, R, G, B) is None
        plt.close(fig)

    def test_filled_contour_plotting_exception_returns_none(self, plotter, monkeypatch):
        fig, ax = _mpltern_ax()
        R, G, B = self._rgb(seed=22)

        def _raise(*a, **kw):
            raise RuntimeError("forced tricontourf failure")

        monkeypatch.setattr(ax, "tricontourf", _raise)
        assert plotter.plot_ternary_kde(ax, R, G, B, grid_resolution=20) is None
        plt.close(fig)


class TestPlotTernaryScatterGaps:
    """Extra branches not already hit by test_ternary_scatter.py."""

    def test_nan_points_removed_and_color_array_filtered(self, plotter):
        fig, ax = _mpltern_ax()
        R = np.array([0.3, np.nan, 0.4])
        G = np.array([0.3, 0.3, 0.3])
        B = np.array([0.4, 0.4, 0.3])
        colors = np.array(["red", "green", "blue"])
        result = plotter.plot_ternary_scatter(ax, R, G, B, color=colors)
        assert result is not None
        plt.close(fig)

    def test_all_points_invalid_returns_none(self, plotter):
        fig, ax = _mpltern_ax()
        R = np.array([np.nan, np.nan])
        G = np.array([np.nan, np.nan])
        B = np.array([np.nan, np.nan])
        result = plotter.plot_ternary_scatter(ax, R, G, B)
        assert result is None
        plt.close(fig)

    def test_mismatched_lengths_raises(self, plotter):
        fig, ax = _mpltern_ax()
        with pytest.raises(ValueError, match="same length"):
            plotter.plot_ternary_scatter(ax, np.array([1.0]), np.array([1.0, 2.0]), np.array([1.0]))
        plt.close(fig)


# ======================================================================
# DatashaderMixin (via AnalysisPlotter)
# ======================================================================

class TestDatashaderMixin:
    def test_init_datashader_available(self, analysis_plotter):
        assert analysis_plotter.datashader_available is True

    def test_import_error_warns(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "datashader", None)
        with pytest.warns(UserWarning, match="Datashader not available"):
            ap = AnalysisPlotter()
        assert ap.datashader_available is False

    def test_plot_large_scatter_matplotlib_path(self, analysis_plotter):
        fig, ax = plt.subplots()
        x = np.arange(10.0)
        y = np.arange(10.0)
        analysis_plotter.plot_large_scatter(ax, x, y, threshold=1000)
        plt.close(fig)

    def test_plot_large_scatter_downsample(self, analysis_plotter, monkeypatch):
        fig, ax = plt.subplots()
        x = np.arange(100.0)
        y = np.arange(100.0)
        # Force the matplotlib downsample branch rather than the (available)
        # real datashader path, which would otherwise win at this threshold.
        monkeypatch.setattr(analysis_plotter, "datashader_available", False)
        analysis_plotter.plot_large_scatter(
            ax, x, y, threshold=10, downsample=True, downsample_factor=5,
        )
        plt.close(fig)

    def test_plot_large_scatter_default_threshold(self, analysis_plotter):
        fig, ax = plt.subplots()
        x = np.arange(5.0)
        y = np.arange(5.0)
        analysis_plotter.plot_large_scatter(ax, x, y)  # threshold=None -> self.datashader_threshold
        plt.close(fig)

    def test_plot_large_scatter_datashader_path(self, analysis_plotter):
        fig, ax = plt.subplots()
        rng = np.random.default_rng(0)
        x = rng.uniform(0, 10, 50)
        y = rng.uniform(0, 10, 50)
        c = rng.uniform(0, 1, 50)
        fig.canvas.draw()
        analysis_plotter.plot_large_scatter(ax, x, y, c=c, threshold=5)
        plt.close(fig)

    def test_plot_large_scatter_datashader_no_color_explicit_canvas(self, analysis_plotter):
        fig, ax = plt.subplots()
        rng = np.random.default_rng(1)
        x = rng.uniform(0, 10, 50)
        y = rng.uniform(0, 10, 50)
        analysis_plotter.plot_large_scatter(ax, x, y, threshold=5, canvas_size=(64, 64))
        plt.close(fig)

    def test_plot_multi_dataset_scatter_matplotlib_path(self, analysis_plotter):
        fig, ax = plt.subplots()
        datasets = [
            {"x": np.arange(5.0), "y": np.arange(5.0)},
            {"x": np.arange(5.0), "y": np.arange(5.0) * 2},
        ]
        analysis_plotter.plot_multi_dataset_scatter(
            ax, datasets, labels=["a", "b"], threshold=1000, sizes=5.0,
        )
        plt.close(fig)

    def test_plot_multi_dataset_scatter_per_dataset_downsample(self, analysis_plotter, monkeypatch):
        fig, ax = plt.subplots()
        datasets = [
            {"x": np.arange(50.0), "y": np.arange(50.0)},
            {"x": np.arange(5.0), "y": np.arange(5.0)},
        ]
        monkeypatch.setattr(analysis_plotter, "datashader_available", False)
        analysis_plotter.plot_multi_dataset_scatter(ax, datasets, threshold=10)
        plt.close(fig)

    def test_plot_multi_dataset_scatter_default_threshold(self, analysis_plotter):
        fig, ax = plt.subplots()
        datasets = [{"x": np.arange(5.0), "y": np.arange(5.0)}]
        analysis_plotter.plot_multi_dataset_scatter(ax, datasets)  # threshold=None
        plt.close(fig)

    def test_plot_multi_dataset_scatter_datashader_path(self, analysis_plotter):
        fig, ax = plt.subplots()
        rng = np.random.default_rng(2)
        datasets = [
            {"x": rng.uniform(0, 10, 30), "y": rng.uniform(0, 10, 30)},
            {"x": rng.uniform(0, 10, 30), "y": rng.uniform(0, 10, 30)},
        ]
        fig.canvas.draw()
        analysis_plotter.plot_multi_dataset_scatter(ax, datasets, labels=["a", "b"], threshold=5)
        plt.close(fig)

    def test_create_preview_plot_no_downsample(self, analysis_plotter):
        fig, ax = plt.subplots()
        x = np.arange(5.0)
        y = np.arange(5.0)
        analysis_plotter.create_preview_plot(ax, x, y, preview_points=10)
        plt.close(fig)

    @pytest.mark.parametrize("method", ["random", "uniform", "density"])
    def test_create_preview_plot_downsample_methods(self, analysis_plotter, method):
        fig, ax = plt.subplots()
        rng = np.random.default_rng(3)
        x = rng.uniform(0, 10, 100)
        y = rng.uniform(0, 10, 100)
        analysis_plotter.create_preview_plot(ax, x, y, preview_points=20, method=method)
        plt.close(fig)

    def test_create_preview_plot_invalid_method_raises(self, analysis_plotter):
        fig, ax = plt.subplots()
        x = np.arange(20.0)
        y = np.arange(20.0)
        with pytest.raises(ValueError, match="Unknown downsampling method"):
            analysis_plotter.create_preview_plot(ax, x, y, preview_points=5, method="bogus")
        plt.close(fig)


# ======================================================================
# PublicationPlotter-specific methods
# ======================================================================

class TestPublicationPlotterHelpers:
    def test_get_plot_font_size_standard(self, plotter):
        assert plotter._get_plot_font_size("standard") == 8
        assert plotter._get_plot_font_size("scatter") == 7

    def test_get_plot_font_size_poster(self):
        poster_plotter = PublicationPlotter(poster=True)
        assert poster_plotter._get_plot_font_size() == 15

    def test_image_scatter_plot_colorbar_on(self, plotter):
        fig, ax = plt.subplots()
        data = np.arange(16, dtype=float).reshape(4, 4)
        x = np.array([1.0, 2.0])
        y = np.array([1.0, 2.0])
        plotter.image_scatter_plot(ax, data, x, y, label="lbl")
        plt.close(fig)

    def test_image_scatter_plot_colorbar_off(self, plotter):
        fig, ax = plt.subplots()
        data = np.arange(16, dtype=float).reshape(4, 4)
        x = np.array([1.0])
        y = np.array([1.0])
        plotter.image_scatter_plot(ax, data, x, y, cbar="off", vmin=0, vmax=15)
        plt.close(fig)

    def test_line_error_plot(self, plotter):
        fig, ax = plt.subplots()
        x = np.arange(5.0)
        y = x ** 2
        yerror = np.ones(5)
        plotter.line_error_plot(ax, x, y, yerror, xlim=(0, 5), ylim=(0, 30), label="e")
        plt.close(fig)
        fig, ax = plt.subplots()
        plotter.line_error_plot(ax, x, y, yerror)
        plt.close(fig)

    def test_make_animated_gif_image_wrapper(self, plotter, tmp_path):
        stack = np.random.rand(2, 4, 4) * 100
        out = tmp_path / "wrapper.gif"
        plotter.make_animated_gif_image(stack, n_frames=2, filename=str(out), width=1.5, height=1.5)
        assert out.exists()

    def test_ternary_scatter_plot(self, plotter):
        fig, axs = plt.subplots(2, 1)
        R = np.array([0.4, 0.3])
        G = np.array([0.3, 0.4])
        B = np.array([0.3, 0.3])
        colours = np.array(["red", "blue"])
        out_fig, out_axs = plotter.ternary_scatter_plot(fig, axs, R, G, B, colours, black_background=False)
        assert out_fig is fig
        plt.close(fig)

    def test_ternary_scatter_plot_black_background_default(self, plotter):
        fig, axs = plt.subplots(2, 1)
        R = np.array([0.4, 0.3])
        G = np.array([0.3, 0.4])
        B = np.array([0.3, 0.3])
        colours = np.array(["red", "blue"])
        plotter.ternary_scatter_plot(fig, axs, R, G, B, colours)
        plt.close(fig)

    def test_ternary_scatter_plot_mpltern_import_error(self, plotter, monkeypatch):
        fig, axs = plt.subplots(2, 1)
        monkeypatch.setitem(sys.modules, "mpltern", None)
        with pytest.raises(ImportError, match="mpltern"):
            plotter.ternary_scatter_plot(
                fig, axs, np.array([0.4]), np.array([0.3]), np.array([0.3]), np.array(["red"]),
            )
        plt.close(fig)

    def test_ternary_contour_plot(self, plotter):
        fig, axs = plt.subplots(2, 1)
        t = np.array([0.4, 0.3, 0.5])
        l = np.array([0.3, 0.4, 0.2])
        r = np.array([0.3, 0.3, 0.3])
        R = np.array([0.4, 0.3])
        G = np.array([0.3, 0.4])
        B = np.array([0.3, 0.3])
        out_fig, out_axs = plotter.ternary_contour_plot(fig, axs, t, l, r, R, G, B, black_background=False)
        assert out_fig is fig
        plt.close(fig)

    def test_ternary_contour_plot_mpltern_import_error(self, plotter, monkeypatch):
        fig, axs = plt.subplots(2, 1)
        monkeypatch.setitem(sys.modules, "mpltern", None)
        t = np.array([0.4])
        l = np.array([0.3])
        r = np.array([0.3])
        with pytest.raises(ImportError, match="mpltern"):
            plotter.ternary_contour_plot(fig, axs, t, l, r, t, l, r)
        plt.close(fig)


# ======================================================================
# AnalysisPlotter construction
# ======================================================================

class TestAnalysisPlotterInit:
    def test_default_construction(self):
        ap = AnalysisPlotter()
        assert ap.config.DEFAULT_FIGSIZE == (10, 6)

    def test_custom_threshold(self):
        ap = AnalysisPlotter(datashader_threshold=50)
        assert ap.datashader_threshold == 50
