#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Class related to making figure-quality plots.

**DEPRECATED:** This module is maintained for backwards compatibility only.
New code should use `PlottingBase.PublicationPlotter` directly instead of
`PlottingFunctions.Plotter`.

Probably best to set your default sans-serif font to Helvetica before you make
figures: https://fowlerlab.org/2019/01/03/changing-the-sans-serif-font-to-helvetica/

The maximum published width for a one-column
figure is 3.33 inches (240 pt). The maximum width for a two-column
figure is 6.69 inches (17 cm). The maximum depth of figures should
be 8 ¼ in. (21.1 cm).

panel labels are 8 point font, ticks are 7 point font,
annotations and legends are 6 point font.
"""
from typing import Optional, List, Tuple, Any, Union
import numpy as np
import warnings

# Use centralised import management and base plotting classes
from ImportManager import get_module, safe_import, is_available
from PlottingBase import PublicationPlotter, PlottingConfig

# Get modules through import manager
matplotlib = get_module("matplotlib")
plt = get_module("matplotlib.pyplot")

# For backwards compatibility, keep direct imports but manage them centrally
try:
    import matplotlib.ticker as plticker
    from matplotlib.ticker import MultipleLocator
    from matplotlib.animation import FuncAnimation, PillowWriter
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
except ImportError as e:
    print(f"⚠️ Some matplotlib components not available: {e}")
    plticker = None
    MultipleLocator = None
    FuncAnimation = None
    PillowWriter = None
    make_axes_locatable = None
    AnchoredSizeBar = None


class PlotConstants:
    """Constants for plotting configuration."""

    ONE_COLUMN_WIDTH = 3.33  # inches
    TWO_COLUMN_WIDTH = 6.69  # inches
    MAX_HEIGHT = 8.25  # inches

    # Font sizes
    POSTER_FONT_SIZE = 12
    POSTER_LINE_WIDTH = 1.0
    STANDARD_FONT_SIZE = 7
    STANDARD_LINE_WIDTH = 0.5

    # Plot font sizes
    POSTER_PLOT_FONT = 15
    STANDARD_PLOT_FONT = 8
    SCATTER_PLOT_FONT = 7.0

    # Default parameters
    DEFAULT_PANEL_HEIGHT_RATIO = 3.5
    DEFAULT_PERCENTILES = (1, 99)
    DEFAULT_GRID_ALPHA = 0.25
    DEFAULT_GRID_LINE_WIDTH = 0.25


class Plotter(PublicationPlotter):
    """A class for creating figure-quality plots with consistent styling.

    **DEPRECATED:** This class is a compatibility wrapper around PublicationPlotter.
    Use `PlottingBase.PublicationPlotter` directly in new code instead.

    Now inherits from PublicationPlotter for enhanced functionality while
    maintaining backwards compatibility with existing code.
    """

    def __init__(self, poster: bool = False, dark_background: bool = False):
        """Initialize the Plotter class.

        Args:
            poster: Whether to use poster-style formatting (larger fonts/lines).
            dark_background: Whether to use dark background style.
        """
        warnings.warn(
            "PlottingFunctions.Plotter is deprecated and will be removed in a future version. "
            "Use PlottingBase.PublicationPlotter directly instead.",
            DeprecationWarning,
            stacklevel=2
        )
        # Initialize base plotter with appropriate configuration
        super().__init__(poster=poster, dark_background=dark_background)

        # Maintain backwards compatibility
        self.poster = poster
        self.db = dark_background

    def _get_font_and_line_params(self) -> Tuple[int, float]:
        """Get font size and line width based on poster mode.

        Returns:
            Tuple of (font_size, line_width).
        """
        if self.poster:
            return PlotConstants.POSTER_FONT_SIZE, PlotConstants.POSTER_LINE_WIDTH
        return PlotConstants.STANDARD_FONT_SIZE, PlotConstants.STANDARD_LINE_WIDTH

    def _get_plot_font_size(self, plot_type: str = "standard") -> int:
        """Get font size for plot elements.

        Args:
            plot_type: Type of plot ("standard" or "scatter").

        Returns:
            Font size for the specified plot type.
        """
        if self.poster:
            return PlotConstants.POSTER_PLOT_FONT

        font_sizes = {
            "standard": PlotConstants.STANDARD_PLOT_FONT,
            "scatter": PlotConstants.SCATTER_PLOT_FONT,
        }
        return font_sizes.get(plot_type, PlotConstants.STANDARD_PLOT_FONT)

    def _setup_grid(self, axs, grid_color: Optional[str] = None) -> None:
        """Set up grid styling for axes.

        Args:
            axs: Matplotlib axes object.
            grid_color: Colour for grid lines. If None, uses white for dark mode, gray otherwise.
        """
        if grid_color is None:
            grid_color = "white" if self.db else "gray"

        axs.grid(
            True,
            which="both",
            ls="--",
            c=grid_color,
            lw=PlotConstants.DEFAULT_GRID_LINE_WIDTH,
            alpha=PlotConstants.DEFAULT_GRID_ALPHA,
        )

    def _configure_rcparams(
        self, figsize: List[float], font_size: int, line_width: float
    ) -> None:
        """Configure matplotlib rcParams.

        Args:
            figsize: Figure size as [width, height] in inches.
            font_size: Font size for plots.
            line_width: Line width for axes.
        """
        plt.rcParams["figure.figsize"] = figsize
        plt.rcParams["font.size"] = font_size
        plt.rcParams["svg.fonttype"] = "none"
        matplotlib.rcParams["pdf.fonttype"] = 42
        matplotlib.rcParams["ps.fonttype"] = 42
        plt.rcParams["axes.linewidth"] = line_width
        plt.rcParams["figure.constrained_layout.use"] = True

        if self.db:
            plt.style.use("dark_background")

    def _setup_axis_ticks(self, axes, line_width: float) -> None:
        """Configure axis tick parameters.

        Args:
            axes: Matplotlib axes object(s).
            line_width: Width for tick marks.
        """

        def configure_single_axis(ax):
            ax.xaxis.set_tick_params(width=line_width, length=line_width * 4)
            ax.yaxis.set_tick_params(width=line_width, length=line_width * 4)
            ax.tick_params(axis="both", pad=1.2)

        if hasattr(axes, "__iter__"):
            for ax in np.atleast_1d(axes):
                configure_single_axis(ax)
        else:
            configure_single_axis(axes)

    def _validate_ratios(self, ratios: List[float], expected_length: int) -> None:
        """Validate that ratios list has correct length.

        Args:
            ratios: List of ratio values.
            expected_length: Expected length of ratios list.

        Raises:
            ValueError: If ratios length doesn't match expected length.
        """
        if len(ratios) != expected_length:
            raise ValueError(
                f"Number of ratios ({len(ratios)}) must match expected length ({expected_length})"
            )

    def _calculate_figure_dimensions(
        self,
        width: Optional[float],
        height: Optional[float],
        npanels: int,
        ratios: List[float],
        default_width: float = PlotConstants.ONE_COLUMN_WIDTH,
    ) -> Tuple[float, float]:
        """Calculate figure dimensions with proper aspect ratio support.

        Args:
            width: Desired width in inches.
            height: Desired height in inches.
            npanels: Number of panels.
            ratios: Height ratios for panels.
            default_width: Default width to use if not specified.

        Returns:
            Tuple of (width, height) in inches.
        """
        if width is not None and height is not None:
            # Both specified - use them directly (respecting maximum constraints)
            xsize = min(width, PlotConstants.TWO_COLUMN_WIDTH)
            ysize = min(height, PlotConstants.MAX_HEIGHT)
        elif width is not None and height is None:
            # Width specified, calculate proportional height
            xsize = min(width, default_width)
            ysize = min(width * sum(ratios), PlotConstants.MAX_HEIGHT)
        elif width is None and height is not None:
            # Height specified, use default width
            xsize = default_width
            ysize = min(height, PlotConstants.MAX_HEIGHT)
        else:
            # Neither specified - use defaults
            xsize = default_width
            ysize = min(
                PlotConstants.DEFAULT_PANEL_HEIGHT_RATIO * npanels,
                PlotConstants.MAX_HEIGHT,
            )

        return xsize, ysize

    def one_column_plot(
        self,
        npanels: int = 1,
        ratios: List[float] = [1],
        height: Optional[float] = None,
        width: Optional[float] = None,
    ) -> Tuple[Any, Any]:
        """Create a one-column width figure.

        Args:
            npanels: Number of panels.
            ratios: List of height ratios for panels.
            height: Override height of figure in inches.
            width: Override width of figure in inches.

        Returns:
            Tuple of (figure, axes) objects.

        Raises:
            ValueError: If ratios length doesn't match npanels.
        """
        self._validate_ratios(ratios, npanels)

        font_size, line_width = self._get_font_and_line_params()
        xsize, ysize = self._calculate_figure_dimensions(width, height, npanels, ratios)

        self._configure_rcparams([xsize, ysize], font_size, line_width)

        fig, axs = plt.subplots(npanels, 1, height_ratios=ratios, frameon=False)
        self._setup_axis_ticks(axs, line_width)

        return fig, axs

    def two_column_plot(
        self,
        nrows: int = 1,
        ncolumns: int = 1,
        heightratio: List[float] = [1],
        widthratio: List[float] = [1],
        width: float = 0,
        height: float = 0,
        big: bool = False,
    ) -> Tuple[Any, Any]:
        """Create a two-column width figure.

        Args:
            nrows: Number of rows.
            ncolumns: Number of columns.
            heightratio: List of height ratios for rows.
            widthratio: List of width ratios for columns.
            height: Override height of figure in inches.
            width: Override width of figure in inches.
            big: Use larger figure size if True.

        Returns:
            Tuple of (figure, axes) objects.

        Raises:
            ValueError: If ratio lengths don't match row/column counts.
        """
        self._validate_ratios(heightratio, nrows)
        self._validate_ratios(widthratio, ncolumns)

        font_size, line_width = self._get_font_and_line_params()

        # Calculate dimensions
        if width == 0:
            xsize = 5 * ncolumns if big else PlotConstants.TWO_COLUMN_WIDTH
        else:
            xsize = width

        if height == 0:
            ysize = 5 * nrows if big else 3 * nrows
        else:
            ysize = height

        self._configure_rcparams([xsize, ysize], font_size, line_width)

        fig, axs = plt.subplots(
            nrows,
            ncolumns,
            height_ratios=heightratio,
            width_ratios=widthratio,
            frameon=False,
        )

        # Simplified axis configuration
        for ax in np.atleast_1d(axs).flat:
            ax.xaxis.set_tick_params(width=line_width, length=line_width * 4)
            ax.yaxis.set_tick_params(width=line_width, length=line_width * 4)
            ax.tick_params(axis="both", pad=1.2)

        return fig, axs

    def _setup_ternary_axis(
        self,
        ax,
        maj_loc: float,
        min_loc: float,
        maxt: float,
        maxl: float,
        maxr: float,
        trianglesize: float,
    ) -> None:
        """Configure ternary plot axis.

        Args:
            ax: Ternary plot axis.
            maj_loc: Major tick interval.
            min_loc: Minor tick interval.
            maxt: Maximum t value.
            maxl: Maximum l value.
            maxr: Maximum r value.
            trianglesize: Size of triangle.
        """
        for axis in [ax.taxis, ax.laxis, ax.raxis]:
            axis.set_major_locator(MultipleLocator(maj_loc))
            axis.set_minor_locator(MultipleLocator(min_loc))

        ax.set_ternary_lim(
            maxt - trianglesize,
            maxt,
            maxl - trianglesize,
            maxl,
            maxr - trianglesize,
            maxr,
        )

        ax.set_tlabel(r"pixel 1 QE")
        ax.set_llabel(r"pixel 3 QE")
        ax.set_rlabel(r"pixel 2 QE")

        ax.grid(lw=0.5, alpha=0.25, ls="--", which="both", axis="both", color="white")

    def ternary_scatter_plot(
        self,
        fig,
        axs,
        R: np.ndarray,
        G: np.ndarray,
        B: np.ndarray,
        colours: np.ndarray,
        xlevel: int = 1,
        ylevel: int = 2,
        location_pos: int = 2,
        maj_loc: float = 0.2,
        min_loc: float = 0.1,
        maxt: float = 1,
        maxl: float = 1,
        maxr: float = 1,
        trianglesize: float = 1,
        s: float = 25,
        lws: float = 0.5,
    ) -> Tuple[Any, Any]:
        """Create a ternary scatter plot.

        Args:
            fig: Figure object.
            axs: Axes array.
            R: R scatter points.
            G: G scatter points.
            B: B scatter points.
            colours: RGBA colors for each scatter point.
            xlevel: X level for subplot.
            ylevel: Y level for subplot.
            location_pos: Position for subplot.
            maj_loc: Major tick interval.
            min_loc: Minor tick interval.
            maxt: Maximum t value.
            maxl: Maximum l value.
            maxr: Maximum r value.
            trianglesize: Size of triangle.
            s: Scatter point size.
            lws: Line width for scatter points.

        Returns:
            Tuple of (figure, axes) objects.
        """
        axs[1].remove()

        try:
            import mpltern

            ax = fig.add_subplot(xlevel, ylevel, location_pos, projection="ternary")
        except ImportError:
            raise ImportError(
                "mpltern is required for ternary plots. Install with: pip install mpltern"
            )

        self._setup_ternary_axis(ax, maj_loc, min_loc, maxt, maxl, maxr, trianglesize)

        ax.scatter(
            R,
            G,
            B,
            s=s,
            facecolors="None",
            edgecolors=colours,
            lw=lws,
            marker="o",
        )
        return fig, axs

    def ternary_contour_plot(
        self,
        fig,
        axs,
        t: np.ndarray,
        l: np.ndarray,
        r: np.ndarray,
        R: np.ndarray,
        G: np.ndarray,
        B: np.ndarray,
        maj_loc: float = 0.2,
        min_loc: float = 0.1,
        gridsize: int = 100,
        bins: Optional[int] = None,
        cmap: str = "gist_gray",
        maxt: float = 1,
        maxl: float = 1,
        maxr: float = 1,
        trianglesize: float = 1,
        ecolour: str = "red",
        s: float = 25,
        lws: float = 0.5,
    ) -> Tuple[Any, Any]:
        """Create a ternary contour plot.

        Args:
            fig: Figure object.
            axs: Axes array.
            t: T data.
            l: L data.
            r: R data.
            R: R scatter points.
            G: G scatter points.
            B: B scatter points.
            maj_loc: Major tick interval.
            min_loc: Minor tick interval.
            gridsize: Grid size for hexbin.
            bins: Number of bins.
            cmap: Colormap.
            maxt: Maximum t value.
            maxl: Maximum l value.
            maxr: Maximum r value.
            trianglesize: Size of triangle.
            ecolour: Edge color for scatter points.
            s: Scatter point size.
            lws: Line width for scatter points.

        Returns:
            Tuple of (figure, axes) objects.
        """
        axs[1].remove()

        try:
            import mpltern

            ax = fig.add_subplot(2, 1, 2, projection="ternary")
        except ImportError:
            raise ImportError(
                "mpltern is required for ternary plots. Install with: pip install mpltern"
            )

        self._setup_ternary_axis(ax, maj_loc, min_loc, maxt, maxl, maxr, trianglesize)

        ax.hexbin(
            t,
            l,
            r,
            gridsize=gridsize,
            edgecolors="none",
            bins=bins,
            cmap=cmap,
            rasterized=True,
        )

        ax.scatter(
            R,
            G,
            B,
            s=s,
            facecolors="None",
            edgecolors=ecolour,
            lw=lws,
            marker="o",
        )
        return fig, axs

    def _calculate_limits(
        self, data: np.ndarray, padding_factor: float = 0.1
    ) -> np.ndarray:
        """Calculate plot limits with padding.

        Args:
            data: Data array.
            padding_factor: Fraction of data range to use as padding.

        Returns:
            Array of [min, max] limits with padding.
        """
        data_min, data_max = np.min(data), np.max(data)
        padding = (data_max - data_min) * padding_factor
        return np.array([data_min - padding, data_max + padding])

    def line_plot(
        self,
        axs,
        x: np.ndarray,
        y: np.ndarray,
        xlim: Optional[np.ndarray] = None,
        ylim: Optional[np.ndarray] = None,
        color: str = "k",
        lw: float = 0.75,
        label: str = "",
        xaxislabel: str = "x axis",
        yaxislabel: str = "y axis",
        ls: str = "-",
        alpha: float = 1,
    ):
        """Create a line plot.

        **DEPRECATED:** Use the base class method instead.

        Args:
            axs: Axes object.
            x: X data.
            y: Y data.
            xlim: X axis limits.
            ylim: Y axis limits.
            color: Line color.
            lw: Line width.
            label: Line label.
            xaxislabel: X axis label.
            yaxislabel: Y axis label.
            ls: Line style.
            alpha: Line transparency.

        Returns:
            Modified axes object.
        """
        # Delegate to base class with parameter name translation
        xlim_tuple = tuple(xlim) if xlim is not None else None
        ylim_tuple = tuple(ylim) if ylim is not None else None

        return super().line_plot(
            axs, x, y,
            xlabel=xaxislabel,
            ylabel=yaxislabel,
            xlim=xlim_tuple,
            ylim=ylim_tuple,
            color=color,
            linewidth=lw,
            linestyle=ls,
            label=label,
            alpha=alpha,
        )

    def line_error_plot(
        self,
        axs,
        x: np.ndarray,
        y: np.ndarray,
        yerror: np.ndarray,
        xlim: Optional[np.ndarray] = None,
        ylim: Optional[np.ndarray] = None,
        color: str = "k",
        lw: float = 0.75,
        label: str = "",
        xaxislabel: str = "x axis",
        yaxislabel: str = "y axis",
        ls: str = "-",
        alpha: float = 1.0,
    ):
        """Create a line plot with error bands.

        **DEPRECATED:** Use the base class method instead.

        Args:
            axs: Axes object.
            x: X data.
            y: Y data.
            yerror: Y error data.
            xlim: X axis limits.
            ylim: Y axis limits.
            color: Line color.
            lw: Line width.
            label: Line label.
            xaxislabel: X axis label.
            yaxislabel: Y axis label.
            ls: Line style.
            alpha: Error band transparency.

        Returns:
            Modified axes object.
        """
        # Delegate to base class with parameter name translation
        xlim_tuple = tuple(xlim) if xlim is not None else None
        ylim_tuple = tuple(ylim) if ylim is not None else None

        return super().line_plot_with_error(
            axs, x, y, yerror,
            xlabel=xaxislabel,
            ylabel=yaxislabel,
            xlim=xlim_tuple,
            ylim=ylim_tuple,
            color=color,
            linewidth=lw,
            label=label,
            alpha=alpha,
        )

    def histogram_plot(
        self,
        axs,
        data: np.ndarray,
        bins: Union[int, np.ndarray],
        xlim: Optional[np.ndarray] = None,
        ylim: Optional[np.ndarray] = None,
        histcolor: str = "gray",
        xaxislabel: str = "x axis",
        alpha: float = 1,
        histtype: str = "bar",
        density: bool = True,
        label: str = "",
    ):
        """Create a histogram plot.

        **DEPRECATED:** Use the base class method instead.

        Args:
            axs: Axes object.
            data: Data array.
            bins: Bin specification.
            xlim: X axis limits.
            ylim: Y axis limits.
            histcolor: Histogram color.
            xaxislabel: X axis label.
            alpha: Histogram transparency.
            histtype: Histogram type.
            density: Whether to plot as probability density.
            label: Histogram label.

        Returns:
            Modified axes object.
        """
        # Delegate to base class with parameter name translation
        xlim_tuple = tuple(xlim) if xlim is not None else None
        ylim_tuple = tuple(ylim) if ylim is not None else None
        ylabel = "probability density" if density else "Counts"

        return super().histogram_plot(
            axs, data,
            bins=bins,
            xlabel=xaxislabel,
            ylabel=ylabel,
            xlim=xlim_tuple,
            ylim=ylim_tuple,
            color=histcolor,
            alpha=alpha,
            density=density,
            label=label,
        )

    def scatter_plot(
        self,
        axs,
        x: np.ndarray,
        y: np.ndarray,
        xlim: Optional[np.ndarray] = None,
        ylim: Optional[np.ndarray] = None,
        label: str = "",
        edgecolor: str = "k",
        facecolor: str = "white",
        s: float = 5,
        lw: float = 0.75,
        xaxislabel: str = "x axis",
        yaxislabel: str = "y axis",
        alpha: float = 1,
        marker: str = "o",
        rasterized: bool = False,
    ):
        """Create a scatter plot.

        **DEPRECATED:** Use the base class method instead.

        Args:
            axs: Axes object.
            x: X data.
            y: Y data.
            xlim: X axis limits.
            ylim: Y axis limits.
            label: Scatter label.
            edgecolor: Edge color.
            facecolor: Face color.
            s: Marker size.
            lw: Line width.
            xaxislabel: X axis label.
            yaxislabel: Y axis label.
            alpha: Marker transparency.
            marker: Marker style.
            rasterized: Whether to rasterize markers.

        Returns:
            Modified axes object.
        """
        # Delegate to base class with parameter name translation
        xlim_tuple = tuple(xlim) if xlim is not None else None
        ylim_tuple = tuple(ylim) if ylim is not None else None

        return super().scatter_plot(
            axs, x, y,
            xlabel=xaxislabel,
            ylabel=yaxislabel,
            xlim=xlim_tuple,
            ylim=ylim_tuple,
            edgecolor=edgecolor,
            facecolor=facecolor,
            size=s,
            linewidth=lw,
            marker=marker,
            label=label,
            alpha=alpha,
            rasterized=rasterized,
        )

    def _setup_colorbar(self, im, axs, cbarlabel: str, location: str = "right") -> None:
        """Set up colorbar for plots using consolidated base functionality.

        Args:
            im: Image object.
            axs: Axes object.
            cbarlabel: Colorbar label.
            location: Colorbar location.
        """
        # Use the base class colorbar functionality
        cbar = self.add_colorbar(im, axs, label=cbarlabel, location=location)

        # Apply custom font sizing for publication quality
        font_size = self._get_plot_font_size()
        cbar.ax.tick_params(labelsize=font_size - 1, pad=0.1, width=0.5, length=2)

    def contourf_plot(
        self,
        axs,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        levels: int = 10,
        cmap: str = "gist_gray",
        cbar: str = "on",
        cbarlabel: str = "photons",
        label: str = "",
        labelcolor: str = "white",
        xaxislabel: str = "xaxislabel",
        yaxislabel: str = "yaxislabel",
    ):
        """Create a contour fill plot.

        Args:
            axs: Axes object.
            X: X coordinates.
            Y: Y coordinates.
            Z: Z values.
            levels: Number of contour levels.
            cmap: Colormap.
            cbar: Whether to show colorbar.
            cbarlabel: Colorbar label.
            label: Plot label.
            labelcolor: Label color.
            xaxislabel: X axis label.
            yaxislabel: Y axis label.

        Returns:
            Modified axes object.
        """
        font_size = self._get_plot_font_size()

        im = axs.contourf(X, Y, Z, levels=levels, cmap=cmap)

        if cbar == "on":
            self._setup_colorbar(im, axs, cbarlabel, "right")

        axs.set_xlabel(xaxislabel, fontsize=font_size)
        axs.set_ylabel(yaxislabel, fontsize=font_size)

        axs.annotate(
            label,
            xy=(5, 5),
            xytext=(20, 60),
            xycoords="data",
            color=labelcolor,
            fontsize=font_size - 1,
        )

        return axs

    def _setup_scalebar(
        self,
        axs,
        pixelsize: float,
        scalebarsize: float,
        scalebarlabel: str,
        labelcolor: str,
        location: str = "lower right",
    ) -> None:
        """Set up scale bar for image plots using consolidated base functionality.

        Args:
            axs: Axes object.
            pixelsize: Pixel size in nm.
            scalebarsize: Scale bar size in nm.
            scalebarlabel: Scale bar label.
            labelcolor: Label color.
            location: Scale bar location.
        """
        # Use the base class scalebar functionality
        self.add_scalebar(
            axs,
            pixelsize=pixelsize,
            length_nm=scalebarsize,
            location=location,
            color=labelcolor,
            label=scalebarlabel,
        )

    def image_plot(
        self,
        axs,
        data: np.ndarray,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        cmap: str = "gist_gray",
        cbar: str = "on",
        cbarlabel: str = "photoelectrons",
        label: str = "",
        labelcolor: str = "white",
        pixelsize: float = 69,
        sbar: str = "on",
        scalebarsize: float = 10000,
        scalebarlabel: str = "10 μm",
        alpha: float = 1,
        plotmask: bool = False,
        mask: Optional[np.ndarray] = None,
        maskcolor: str = "white",
    ):
        """Create an image plot.

        Args:
            axs: Axes object.
            data: Image data.
            vmin: Minimum display value.
            vmax: Maximum display value.
            cmap: Colormap.
            cbar: Whether to show colorbar.
            cbarlabel: Colorbar label.
            label: Image label.
            labelcolor: Label color.
            pixelsize: Pixel size in nm.
            sbar: Whether to show scale bar.
            scalebarsize: Scale bar size in nm.
            scalebarlabel: Scale bar label.
            alpha: Image transparency.
            plotmask: Whether to plot mask overlay.
            mask: Mask array.
            maskcolor: Mask color.

        Returns:
            Modified axes object.
        """
        font_size = self._get_plot_font_size()

        if vmin is None:
            vmin = np.percentile(data.ravel(), PlotConstants.DEFAULT_PERCENTILES[0])
        if vmax is None:
            vmax = np.percentile(data.ravel(), PlotConstants.DEFAULT_PERCENTILES[1])

        im = axs.imshow(
            data, vmin=vmin, vmax=vmax, cmap=cmap, alpha=alpha, origin="lower"
        )

        if cbar == "on":
            self._setup_colorbar(im, axs, cbarlabel, "left")

        axs.set_xticks([])
        axs.set_yticks([])

        if sbar == "on":
            self._setup_scalebar(
                axs, pixelsize, scalebarsize, scalebarlabel, labelcolor
            )

        axs.annotate(
            label,
            xy=(5, 5),
            xytext=(20, 60),
            xycoords="data",
            color=labelcolor,
            fontsize=font_size - 1,
        )

        if plotmask and mask is not None:
            axs.contour(mask, [0.5], linewidths=0.75, colors=maskcolor)

        return axs

    def image_scatter_plot(
        self,
        axs,
        data: np.ndarray,
        xdata: np.ndarray,
        ydata: np.ndarray,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        cmap: str = "gist_gray",
        cbar: str = "on",
        cbarlabel: str = "photons",
        label: str = "",
        labelcolor: str = "white",
        pixelsize: float = 69,
        scalebarsize: float = 10000,
        scalebarlabel: str = "10 μm",
        alpha: float = 1,
        scatteralpha: float = 1,
        scattercolor: str = "red",
        facecolor: str = "None",
        marker: str = "o",
        s: float = 20,
        lws: float = 0.75,
    ):
        """Create an image plot with scatter overlay.

        Args:
            axs: Axes object.
            data: Image data.
            xdata: Scatter X coordinates.
            ydata: Scatter Y coordinates.
            vmin: Minimum display value.
            vmax: Maximum display value.
            cmap: Colormap.
            cbar: Whether to show colorbar.
            cbarlabel: Colorbar label.
            label: Image label.
            labelcolor: Label color.
            pixelsize: Pixel size in nm.
            scalebarsize: Scale bar size in nm.
            scalebarlabel: Scale bar label.
            alpha: Image transparency.
            scatteralpha: Scatter transparency.
            scattercolor: Scatter color.
            s: Scatter marker size.
            lws: Scatter line width.

        Returns:
            Modified axes object.
        """
        font_size = self._get_plot_font_size()

        if vmin is None:
            vmin = np.percentile(data.ravel(), PlotConstants.DEFAULT_PERCENTILES[0])
        if vmax is None:
            vmax = np.percentile(data.ravel(), PlotConstants.DEFAULT_PERCENTILES[1])

        im = axs.imshow(
            data, vmin=vmin, vmax=vmax, cmap=cmap, alpha=alpha, origin="lower"
        )

        if cbar == "on":
            self._setup_colorbar(im, axs, cbarlabel, "left")

        axs.set_xticks([])
        axs.set_yticks([])

        self._setup_scalebar(axs, pixelsize, scalebarsize, scalebarlabel, labelcolor)

        axs.annotate(
            label,
            xy=(5, 5),
            xytext=(20, 60),
            xycoords="data",
            color=labelcolor,
            fontsize=font_size - 1,
        )

        axs.scatter(
            xdata,
            ydata,
            lw=lws,
            edgecolor=scattercolor,
            s=s,
            marker=marker,
            facecolors=facecolor,
            alpha=scatteralpha,
        )
        return axs

    def _create_camera_pattern_overlay(
        self, ax, xsize: float, n_pixels: int = 13
    ) -> None:
        """Create camera pattern overlay with optimised drawing.

        Args:
            ax: Axes object.
            xsize: Figure width for scaling.
            n_pixels: Number of pixels in pattern.
        """
        # Grid setup
        myInterval = 1.0
        loc = plticker.MultipleLocator(base=myInterval)
        ax.xaxis.set_major_locator(loc)
        ax.yaxis.set_major_locator(loc)

        # Grid lines
        ax.grid(
            which="major",
            axis="both",
            linestyle="-",
            lw=(3 / 13) * xsize,
            color="white",
        )

        # Tick configuration
        ax.tick_params("both", length=0, width=(2 / 13) * xsize, which="major")
        ax.tick_params("both", length=0, width=(1 / 13) * xsize, which="minor")

        # Spine configuration
        for axis in ["top", "bottom", "left", "right"]:
            ax.spines[axis].set_linewidth((2 / 13) * xsize)
            ax.spines[axis].set_color("white")

        # Pattern lines - optimised version
        ystart = np.arange(0, n_pixels)
        xstart = np.arange(0, n_pixels)

        for yval in ystart:
            for xval in xstart:
                if (yval % 2 == 0 and xval % 2 == 0) or (
                    yval % 2 != 0 and xval % 2 != 0
                ):
                    # Vertical lines for specific pattern
                    linepos = xval + np.arange(0.1, 1, 0.1)
                    for l in linepos:
                        ax.vlines(
                            x=l,
                            ymin=yval,
                            ymax=yval + 1,
                            lw=(5 / 13) * xsize,
                            color="white",
                        )

        # Diagonal pattern - simplified
        xlin = np.linspace(0, 1, 100)  # Reduced points for performance
        yscalingfactor = np.arange(-0.85, 1.05, 0.6)  # Reduced iterations

        for yval in ystart:
            for xval in xstart:
                if (yval % 2 != 0 and xval % 2 == 0) or (
                    yval % 2 == 0 and xval % 2 != 0
                ):
                    for ys in yscalingfactor:
                        xlin_temp = xlin + xval
                        if yval % 2 != 0 and xval % 2 == 0:
                            ylin = xlin + yval + ys
                        else:
                            ylin = (1 - xlin) + yval + ys

                        # Filter points
                        mask = (ylin > yval + 0.01) & (ylin < yval + 0.99)
                        if np.any(mask):
                            ax.plot(
                                xlin_temp[mask],
                                ylin[mask],
                                lw=(5 / 13) * xsize,
                                color="white",
                            )

    def make_camera_pattern(
        self,
        ax,
        xsize: float,
        array: np.ndarray,
        xlim: int = 10,
        ylim: int = 10,
        scatter: bool = False,
        xscatter: Union[float, np.ndarray] = 0,
        yscatter: Union[float, np.ndarray] = 0,
        s: float = 5,
        scolor: str = "green",
    ):
        """Create camera pattern visualization.

        Args:
            ax: Axes object.
            xsize: Figure width for scaling.
            array: Base image array.
            xlim: X axis limit.
            ylim: Y axis limit.
            scatter: Whether to add scatter points.
            xscatter: X coordinates for scatter.
            yscatter: Y coordinates for scatter.
            s: Scatter point size.
            scolor: Scatter point color.

        Returns:
            Modified axes object.
        """
        # Display base image
        ax.imshow(array, cmap="gist_gray")
        ax.set_ylim([0, xlim])
        ax.set_xlim([0, ylim])
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])

        # Create pattern overlay
        self._create_camera_pattern_overlay(ax, xsize)

        # Add scatter points if requested
        if scatter:
            ax.scatter(
                xscatter,
                yscatter,
                facecolor=scolor,
                edgecolors=None,
                lw=0,
                s=s,
                marker="o",
                zorder=np.inf,
            )
            ax.set_ylim([0, xlim])
            ax.set_xlim([0, ylim])

        return ax

    def make_animated_gif_image(
        self,
        image: np.ndarray,
        n_frames: int,
        filename: str,
        vmin: float = 0,
        vmax: float = 150,
        pixelsize: float = 69,
        scalebarsize: float = 300,
        scalebarlabel: str = "300 nm",
        label: str = "",
        fontsz: int = 6,
        cbarlabel: str = "# of photoelectrons",
        cbar: bool = False,
        width: float = 3,
        height: float = 3,
    ) -> None:
        """Create animated GIF from image sequence.

        .. deprecated::
            Use :meth:`make_animated_gif` instead, which also supports RGB images.

        Args:
            image: Image stack.
            n_frames: Number of frames (ignored, extracted from image shape).
            filename: Output filename.
            vmin: Minimum display value.
            vmax: Maximum display value.
            pixelsize: Pixel size in nm.
            scalebarsize: Scale bar size in nm.
            scalebarlabel: Scale bar label.
            label: Image label.
            fontsz: Font size.
            cbarlabel: Colorbar label.
            cbar: Whether to show colorbar.
            width: Figure width.
            height: Figure height.
        """
        warnings.warn(
            "make_animated_gif_image is deprecated. Use make_animated_gif instead, "
            "which also supports RGB images.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.make_animated_gif(
            image=image,
            filename=filename,
            vmin=vmin,
            vmax=vmax,
            pixelsize=pixelsize,
            scalebarsize=scalebarsize,
            scalebarlabel=scalebarlabel,
            label=label,
            fontsz=fontsz,
            cbarlabel=cbarlabel,
            cbar=cbar,
            width=width,
            height=height,
        )
