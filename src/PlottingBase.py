"""
PlottingBase.py

Base plotting utilities and common patterns for pyBayerSMLM.
Consolidates common functionality from PlottingFunctions.py and DriftPlotting.py.

:authors: jsb92
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any, Union
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

# Import mpltern to register ternary projection with matplotlib
# This allows using projection="ternary" in add_subplot()
try:
    import mpltern
    MPLTERN_AVAILABLE = True
except ImportError:
    MPLTERN_AVAILABLE = False
    warnings.warn(
        "mpltern not available. Ternary plots will not work. "
        "Install with: pip install mpltern",
        ImportWarning
    )

class PublicationConstants:
    """Constants for publication-quality plots following journal standards.

    Based on common journal requirements for scientific publications:
    - One-column max width: 3.33 inches (240 pt)
    - Two-column max width: 6.69 inches (17 cm)
    - Maximum depth: 8.25 inches (21.1 cm)

    Font hierarchy (from PlottingFunctions):
    - Tick labels: 7 pt
    - Axis labels (panel titles): 8 pt
    - Legends and annotations: 6 pt

    This ensures figures meet publication standards by default while allowing
    explicit overrides for presentations, posters, or other special cases.
    """

    # Figure dimensions (inches)
    ONE_COLUMN_WIDTH = 3.33
    TWO_COLUMN_WIDTH = 6.69
    MAX_HEIGHT = 8.25

    # Standard mode (for publications)
    STANDARD_FONT_SIZE = 7  # Base font size
    STANDARD_TICK_LABELSIZE = 7  # Tick labels (CORRECTED from 8)
    STANDARD_AXIS_LABELSIZE = 8  # Axis titles/panel labels
    STANDARD_LEGEND_FONTSIZE = 6  # Legends and annotations
    STANDARD_LINE_WIDTH = 0.5  # CORRECTED from 1.0
    STANDARD_TICK_LENGTH = 2.0  # 4 * line_width

    # Poster mode (for presentations)
    POSTER_FONT_SIZE = 12
    POSTER_TICK_LABELSIZE = 12
    POSTER_AXIS_LABELSIZE = 15
    POSTER_LEGEND_FONTSIZE = 10
    POSTER_LINE_WIDTH = 1.0
    POSTER_TICK_LENGTH = 4.0

    # Default panel sizing
    DEFAULT_PANEL_HEIGHT_RATIO = 3.5  # Height per panel in one-column plots
    DEFAULT_TWO_COLUMN_ROW_HEIGHT = 3.0  # Height per row in two-column plots


@dataclass
class PlottingConfig:
    """Configuration class for consistent plotting styles across pyBayerSMLM.

    This class now properly implements publication standards from PlottingFunctions.
    Default values follow journal requirements for single-molecule microscopy papers.

    Font hierarchy:
    - Base font (general text): 7pt
    - Axis labels (x/y axis titles): 8pt
    - Tick labels (numbers on axes): 7pt
    - Legends and annotations: 6pt

    For poster mode, all fonts are scaled appropriately (12pt base, 15pt axis labels, etc.)
    """

    # Display properties
    DEFAULT_DPI: int = 600  # High DPI for publication quality
    DEFAULT_SAVE_DPI: int = 600  # Match display DPI for consistency

    # Color schemes
    DEFAULT_COLORMAP: str = "gist_gray"
    DEFAULT_SCATTER_COLOR: str = "blue"
    DEFAULT_GRID_COLOR: str = "gray"
    DEFAULT_GRID_ALPHA: float = 0.3

    # Image display percentiles
    DEFAULT_VMIN_PERCENTILE: float = 1.0
    DEFAULT_VMAX_PERCENTILE: float = 99.0

    # Publication standards (set in __post_init__ based on poster mode)
    font_size: int = PublicationConstants.STANDARD_FONT_SIZE
    tick_labelsize: int = PublicationConstants.STANDARD_TICK_LABELSIZE
    axis_labelsize: int = PublicationConstants.STANDARD_AXIS_LABELSIZE
    legend_fontsize: int = PublicationConstants.STANDARD_LEGEND_FONTSIZE
    line_width: float = PublicationConstants.STANDARD_LINE_WIDTH
    tick_length: float = PublicationConstants.STANDARD_TICK_LENGTH

    # Marker properties
    DEFAULT_MARKER_SIZE: float = 1.0

    # Colorbar properties
    DEFAULT_COLORBAR_WIDTH: str = "5%"
    DEFAULT_COLORBAR_PAD: float = 0.05

    # Scale bar properties
    DEFAULT_SCALEBAR_COLOR: str = "white"
    DEFAULT_SCALEBAR_FONTSIZE: int = 10

    # Mode flags
    poster_mode: bool = False
    dark_background: bool = False

    def __post_init__(self):
        """Set up matplotlib parameters based on configuration."""
        # Use poster values if in poster mode
        if self.poster_mode:
            self.font_size = PublicationConstants.POSTER_FONT_SIZE
            self.tick_labelsize = PublicationConstants.POSTER_TICK_LABELSIZE
            self.axis_labelsize = PublicationConstants.POSTER_AXIS_LABELSIZE
            self.legend_fontsize = PublicationConstants.POSTER_LEGEND_FONTSIZE
            self.line_width = PublicationConstants.POSTER_LINE_WIDTH
            self.tick_length = PublicationConstants.POSTER_TICK_LENGTH

        # Configure matplotlib globally with proper font hierarchy
        matplotlib.rcParams.update(
            {
                # Font sizes (proper hierarchy)
                "font.size": self.font_size,  # Base font (7pt standard, 12pt poster)
                "axes.labelsize": self.axis_labelsize,  # Axis labels (8pt standard, 15pt poster)
                "axes.titlesize": self.axis_labelsize,  # Panel titles (8pt standard, 15pt poster)
                "xtick.labelsize": self.tick_labelsize,  # Tick labels (7pt standard, 12pt poster)
                "ytick.labelsize": self.tick_labelsize,  # Tick labels (7pt standard, 12pt poster)
                "legend.fontsize": self.legend_fontsize,  # Legends (6pt standard, 10pt poster)
                # Line widths
                "axes.linewidth": self.line_width,  # Axis spines (0.5pt standard, 1.0pt poster)
                "xtick.major.width": self.line_width,  # Tick marks
                "ytick.major.width": self.line_width,
                "xtick.major.size": self.tick_length,  # Tick length
                "ytick.major.size": self.tick_length,
                # Padding
                "xtick.major.pad": 1.2,
                "ytick.major.pad": 1.2,
                # Layout
                "figure.constrained_layout.use": True,  # Auto-adjust spacing
                # Font embedding for publications
                "svg.fonttype": "none",  # Save text as text (not paths) in SVG
                "pdf.fonttype": 42,  # Embed fonts as TrueType (editable) in PDF
                "ps.fonttype": 42,  # Same for PostScript
            }
        )

        # Dark background adjustments
        if self.dark_background:
            self.DEFAULT_GRID_COLOR = "white"
            self.DEFAULT_SCALEBAR_COLOR = "white"
            plt.style.use("dark_background")


class BasePlotter(ABC):
    """Base class for all plotting functionality in pyBayerSMLM.

    This class provides common plotting utilities and patterns that are shared
    across different plotting modules in the codebase. It follows the DRY principle
    by consolidating repeated plotting patterns.
    """

    def __init__(self, config: Optional[PlottingConfig] = None):
        """Initialize the base plotter.

        Args:
            config: Plotting configuration. If None, uses default configuration.
        """
        self.config = config or PlottingConfig()
        self._setup_matplotlib()

    def _setup_matplotlib(self):
        """Setup matplotlib with consistent configuration."""
        # Configure matplotlib backend if needed
        if matplotlib.get_backend() == "Agg":
            warnings.warn("Using Agg backend - plots will not display interactively")

    def create_figure(
        self,
        figsize: Optional[Tuple[float, float]] = None,
        dpi: Optional[int] = None,
        facecolor: str = "white",
        edgecolor: str = "black",
    ) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        """Create a standardised figure with single axis.

        Args:
            figsize: Figure size in inches (width, height)
            dpi: Dots per inch for figure resolution
            facecolor: Figure face color
            edgecolor: Figure edge color

        Returns:
            Tuple of (figure, axis)
        """
        figsize = figsize or self.config.DEFAULT_FIGSIZE
        dpi = dpi or self.config.DEFAULT_DPI

        fig, ax = plt.subplots(
            figsize=figsize, dpi=dpi, facecolor=facecolor, edgecolor=edgecolor
        )
        return fig, ax

    def one_column_plot(
        self,
        npanels: int = 1,
        ratios: Optional[List[float]] = None,
        height: Optional[float] = None,
        width: Optional[float] = None,
    ) -> Tuple[matplotlib.figure.Figure, Union[matplotlib.axes.Axes, np.ndarray]]:
        """Create a one-column width publication-quality figure.

        Defaults to 3.33" width, 3.5" per panel height at 600 DPI.

        Args:
            npanels: Number of vertical panels
            ratios: Height ratios for panels. If None, all panels equal height.
            height: Total figure height in inches. If None, uses standard (3.5" per panel).
            width: Figure width in inches. If None, uses one-column standard (3.33").

        Returns:
            Tuple of (figure, axes). axes is single Axes if npanels=1, else 1D array.
        """
        if ratios is None:
            ratios = [1] * npanels

        if len(ratios) != npanels:
            raise ValueError(f"Number of ratios ({len(ratios)}) must match npanels ({npanels})")

        # Calculate dimensions with publication standards
        if width is not None:
            xsize = width
            if width > PublicationConstants.ONE_COLUMN_WIDTH:
                warnings.warn(
                    f"Width {width:.2f}\" exceeds one-column standard "
                    f"({PublicationConstants.ONE_COLUMN_WIDTH:.2f}\")"
                )
        else:
            xsize = PublicationConstants.ONE_COLUMN_WIDTH  # Default: 3.33"

        if height is not None:
            ysize = height
        else:
            ysize = min(
                PublicationConstants.DEFAULT_PANEL_HEIGHT_RATIO * npanels,
                PublicationConstants.MAX_HEIGHT
            )

        fig, axs = plt.subplots(
            nrows=npanels, ncols=1,
            figsize=(xsize, ysize),
            height_ratios=ratios,
            frameon=False,
            squeeze=False,
            dpi=self.config.DEFAULT_DPI,  # 600 DPI
        )

        # Configure tick parameters
        for ax in axs.flat:
            ax.xaxis.set_tick_params(width=self.config.line_width, length=self.config.tick_length)
            ax.yaxis.set_tick_params(width=self.config.line_width, length=self.config.tick_length)

        # Return appropriately squeezed axes
        if npanels == 1:
            return fig, axs[0, 0]
        else:
            return fig, axs[:, 0]

    def two_column_plot(
        self,
        nrows: int = 1,
        ncols: int = 1,
        height_ratios: Optional[List[float]] = None,
        width_ratios: Optional[List[float]] = None,
        width: Optional[float] = None,
        height: Optional[float] = None,
        big: bool = False,
    ) -> Tuple[matplotlib.figure.Figure, Union[matplotlib.axes.Axes, np.ndarray]]:
        """Create a two-column width publication-quality figure.

        Defaults to 6.69" width, 3.0" per row height at 600 DPI.

        Args:
            nrows: Number of rows
            ncols: Number of columns
            height_ratios: Relative heights of rows. If None, all equal.
            width_ratios: Relative widths of columns. If None, all equal.
            width: Total figure width in inches. If None, uses two-column standard (6.69").
            height: Total figure height in inches. If None, uses standard (3.0" per row).
            big: If True, allows larger sizes for presentations (5" per dimension).

        Returns:
            Tuple of (figure, axes). axes shape depends on nrows/ncols:
                - nrows=1, ncols=1: single Axes
                - nrows=1: 1D array (columns)
                - ncols=1: 1D array (rows)
                - else: 2D array
        """
        if height_ratios is None:
            height_ratios = [1] * nrows
        if width_ratios is None:
            width_ratios = [1] * ncols

        # Calculate dimensions
        if width is not None:
            xsize = width
            if width > PublicationConstants.TWO_COLUMN_WIDTH and not big:
                warnings.warn(
                    f"Width {width:.2f}\" exceeds two-column standard "
                    f"({PublicationConstants.TWO_COLUMN_WIDTH:.2f}\")"
                )
        else:
            if big:
                xsize = 5.0 * ncols
            else:
                xsize = PublicationConstants.TWO_COLUMN_WIDTH  # Default: 6.69"

        if height is not None:
            ysize = height
        else:
            if big:
                ysize = min(5.0 * nrows, PublicationConstants.MAX_HEIGHT)
            else:
                ysize = min(
                    PublicationConstants.DEFAULT_TWO_COLUMN_ROW_HEIGHT * nrows,
                    PublicationConstants.MAX_HEIGHT
                )

        fig, axs = plt.subplots(
            nrows=nrows, ncols=ncols,
            figsize=(xsize, ysize),
            height_ratios=height_ratios,
            width_ratios=width_ratios,
            frameon=False,
            squeeze=False,
            dpi=self.config.DEFAULT_DPI,
        )

        # Configure tick parameters
        for ax in axs.flat:
            ax.xaxis.set_tick_params(width=self.config.line_width, length=self.config.tick_length)
            ax.yaxis.set_tick_params(width=self.config.line_width, length=self.config.tick_length)

        # Return appropriately squeezed axes
        if nrows == 1 and ncols == 1:
            return fig, axs[0, 0]
        elif nrows == 1:
            return fig, axs[0, :]
        elif ncols == 1:
            return fig, axs[:, 0]
        else:
            return fig, axs

    def setup_axis(
        self,
        ax: matplotlib.axes.Axes,
        xlabel: str = "",
        ylabel: str = "",
        title: str = "",
        grid: bool = True,
        grid_alpha: Optional[float] = None,
        equal_aspect: bool = False,
        spine_style: Optional[str] = None,
    ) -> None:
        """Configure axis with standard settings.

        Args:
            ax: Matplotlib axis to configure
            xlabel: X-axis label
            ylabel: Y-axis label
            title: Axis title
            grid: Whether to show grid
            grid_alpha: Grid transparency (uses default if None)
            equal_aspect: Whether to set equal aspect ratio
            spine_style: Style for axis spines ('box', 'left-bottom', 'none')
        """
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)

        if grid:
            alpha = grid_alpha or self.config.DEFAULT_GRID_ALPHA
            ax.grid(
                True,
                alpha=alpha,
                color=self.config.DEFAULT_GRID_COLOR,
                linestyle="--",
                linewidth=0.5,
            )

        if equal_aspect:
            ax.set_aspect("equal")

        # Configure spines
        if spine_style == "none":
            for spine in ax.spines.values():
                spine.set_visible(False)
        elif spine_style == "left-bottom":
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    def create_image_plot(
        self,
        ax: matplotlib.axes.Axes,
        data: np.ndarray,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        cmap: str = None,
        origin: str = "lower",
        interpolation = None,
    ) -> matplotlib.image.AxesImage:
        """Create standardised image plot.

        Args:
            ax: Axis to plot on
            data: 2D array to display
            vmin: Minimum value for colormap (auto-calculated if None)
            vmax: Maximum value for colormap (auto-calculated if None)
            cmap: Colormap name (uses default if None)
            origin: Image origin ('lower' or 'upper')
            interpolation: Interpolation method (None uses matplotlib's default behavior,
                which avoids rendering artifacts. Can be set to 'none', 'nearest', etc. if needed.)

        Returns:
            AxesImage object for further customisation
        """
        cmap = cmap or self.config.DEFAULT_COLORMAP

        # Auto-calculate vmin/vmax using percentiles if not provided
        if vmin is None:
            vmin = np.percentile(data.ravel(), self.config.DEFAULT_VMIN_PERCENTILE)
        if vmax is None:
            vmax = np.percentile(data.ravel(), self.config.DEFAULT_VMAX_PERCENTILE)

        im = ax.imshow(
            data,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            origin=origin,
            interpolation=interpolation,
        )

        return im

    def add_colorbar(
        self,
        im: matplotlib.image.AxesImage,
        ax: matplotlib.axes.Axes,
        label: str = "",
        location: str = "right",
        size: Optional[str] = None,
        pad: Optional[float] = None,
    ) -> matplotlib.colorbar.Colorbar:
        """Add colorbar to image plot.

        Args:
            im: Image object to create colorbar for
            ax: Axis containing the image
            label: Colorbar label
            location: Colorbar location ('right', 'left', 'top', 'bottom')
            size: Colorbar size as percentage (e.g., '5%')
            pad: Padding between axis and colorbar

        Returns:
            Colorbar object
        """
        size = size or self.config.DEFAULT_COLORBAR_WIDTH
        pad = pad or self.config.DEFAULT_COLORBAR_PAD

        divider = make_axes_locatable(ax)
        cax = divider.append_axes(location, size=size, pad=pad)

        cbar = plt.colorbar(im, cax=cax)
        if label:
            cbar.set_label(label)

        return cbar

    def add_scalebar(
        self,
        ax: matplotlib.axes.Axes,
        pixelsize: float,
        length_nm: float,
        location: str = "lower right",
        color: str = None,
        fontsize: Optional[int] = None,
        label: Optional[str] = None,
    ) -> AnchoredSizeBar:
        """Add scale bar to plot.

        Args:
            ax: Axis to add scale bar to
            pixelsize: Size of one pixel in nanometres
            length_nm: Length of scale bar in nanometres
            location: Scale bar location
            color: Scale bar color
            fontsize: Font size for scale bar text
            label: Custom label (auto-generated if None)

        Returns:
            AnchoredSizeBar object
        """
        color = color or self.config.DEFAULT_SCALEBAR_COLOR
        fontsize = fontsize or self.config.DEFAULT_SCALEBAR_FONTSIZE

        # Convert nanometres to pixels
        length_pixels = length_nm / pixelsize

        # Generate label if not provided
        if label is None:
            if length_nm >= 1000:
                label = f"{length_nm/1000:.1f} μm"
            else:
                label = f"{length_nm:.0f} nm"

        scalebar = AnchoredSizeBar(
            ax.transData,
            length_pixels,
            label,
            loc=location,
            pad=0.5,
            color=color,
            frameon=False,
            size_vertical=length_pixels / 20,
            fontproperties={"size": fontsize},
        )

        ax.add_artist(scalebar)
        return scalebar

    def create_grouped_scatter(
        self,
        ax: matplotlib.axes.Axes,
        data_groups: Dict[str, Dict[str, np.ndarray]],
        colors: Optional[List[str]] = None,
        markers: Optional[List[str]] = None,
        sizes: Optional[Union[float, List[float]]] = None,
        alpha: float = 0.7,
    ) -> List[matplotlib.collections.PathCollection]:
        """Create scatter plot with grouped data and automatic coloring.

        Args:
            ax: Axis to plot on
            data_groups: Dictionary mapping group names to data dicts with 'x' and 'y' keys
            colors: List of colors for each group (auto-generated if None)
            markers: List of markers for each group
            sizes: Marker sizes (single value or list)
            alpha: Transparency

        Returns:
            List of scatter plot collections
        """
        if colors is None:
            # Use matplotlib color cycle
            prop_cycle = plt.rcParams["axes.prop_cycle"]
            colors = prop_cycle.by_key()["color"]

        if markers is None:
            markers = ["o"] * len(data_groups)

        if isinstance(sizes, (int, float)):
            sizes = [sizes] * len(data_groups)
        elif sizes is None:
            sizes = [self.config.DEFAULT_MARKER_SIZE] * len(data_groups)

        scatters = []
        for i, (group_name, data) in enumerate(data_groups.items()):
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            size = sizes[i % len(sizes)]

            scatter = ax.scatter(
                data["x"],
                data["y"],
                c=color,
                marker=marker,
                s=size,
                alpha=alpha,
                label=group_name,
            )
            scatters.append(scatter)

        if len(data_groups) > 1:
            ax.legend()

        return scatters

    # Convenience plotting methods for common plot types
    def line_plot(
        self,
        ax: matplotlib.axes.Axes,
        x: np.ndarray,
        y: np.ndarray,
        xlabel: str = "x axis",
        ylabel: str = "y axis",
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        color: str = "k",
        linewidth: float = 1.0,
        linestyle: str = "-",
        label: str = "",
        alpha: float = 1.0,
        grid: bool = True,
    ) -> matplotlib.axes.Axes:
        """Create a line plot with consistent styling.

        Args:
            ax: Axes object to plot on
            x: X data
            y: Y data
            xlabel: X axis label
            ylabel: Y axis label
            xlim: X axis limits (min, max)
            ylim: Y axis limits (min, max)
            color: Line color
            linewidth: Line width
            linestyle: Line style ('-', '--', '-.', ':')
            label: Line label for legend
            alpha: Line transparency
            grid: Whether to show grid

        Returns:
            Modified axes object
        """
        ax.plot(x, y, color=color, linewidth=linewidth, linestyle=linestyle,
                label=label, alpha=alpha)

        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        self.setup_axis(ax, xlabel=xlabel, ylabel=ylabel, grid=grid)

        if label:
            ax.legend()

        return ax

    def line_plot_with_error(
        self,
        ax: matplotlib.axes.Axes,
        x: np.ndarray,
        y: np.ndarray,
        yerr: np.ndarray,
        xlabel: str = "x axis",
        ylabel: str = "y axis",
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        color: str = "k",
        linewidth: float = 1.0,
        label: str = "",
        alpha: float = 0.3,
        grid: bool = True,
    ) -> matplotlib.axes.Axes:
        """Create a line plot with error bars/shading.

        Args:
            ax: Axes object to plot on
            x: X data
            y: Y data
            yerr: Y error values
            xlabel: X axis label
            ylabel: Y axis label
            xlim: X axis limits (min, max)
            ylim: Y axis limits (min, max)
            color: Line and error color
            linewidth: Line width
            label: Line label for legend
            alpha: Error shading transparency
            grid: Whether to show grid

        Returns:
            Modified axes object
        """
        ax.plot(x, y, color=color, linewidth=linewidth, label=label)
        ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=alpha)

        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        self.setup_axis(ax, xlabel=xlabel, ylabel=ylabel, grid=grid)

        if label:
            ax.legend()

        return ax

    def scatter_plot(
        self,
        ax: matplotlib.axes.Axes,
        x: np.ndarray,
        y: np.ndarray,
        xlabel: str = "x axis",
        ylabel: str = "y axis",
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        color: str = "k",
        edgecolor: str = "k",
        facecolor: str = "white",
        size: float = 20,
        marker: str = "o",
        label: str = "",
        alpha: float = 1.0,
        linewidth: float = 0.75,
        rasterized: bool = False,
        grid: bool = True,
    ) -> matplotlib.axes.Axes:
        """Create a scatter plot with consistent styling.

        Args:
            ax: Axes object to plot on
            x: X data
            y: Y data
            xlabel: X axis label
            ylabel: Y axis label
            xlim: X axis limits (min, max)
            ylim: Y axis limits (min, max)
            color: Overall marker color (overridden by facecolor/edgecolor)
            edgecolor: Marker edge color
            facecolor: Marker face color
            size: Marker size
            marker: Marker style
            label: Scatter label for legend
            alpha: Marker transparency
            linewidth: Edge line width
            rasterized: Whether to rasterize markers (recommended for >1000 points)
            grid: Whether to show grid

        Returns:
            Modified axes object
        """
        ax.scatter(
            x, y,
            s=size,
            c=color if facecolor == "white" and edgecolor == "k" else None,
            edgecolors=edgecolor,
            facecolors=facecolor if facecolor != "white" or edgecolor != "k" else None,
            marker=marker,
            label=label,
            alpha=alpha,
            linewidths=linewidth,
            rasterized=rasterized,
        )

        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        self.setup_axis(ax, xlabel=xlabel, ylabel=ylabel, grid=grid)

        if label:
            ax.legend()

        return ax

    def histogram_plot(
        self,
        ax: matplotlib.axes.Axes,
        data: np.ndarray,
        bins: Union[int, np.ndarray] = 50,
        xlabel: str = "Value",
        ylabel: str = "Counts",
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        color: str = "blue",
        edgecolor = None,
        alpha: float = 0.7,
        density: bool = False,
        label: str = "",
        grid: bool = True,
    ) -> matplotlib.axes.Axes:
        """Create a histogram with consistent styling.

        Args:
            ax: Axes object to plot on
            data: Data to histogram
            bins: Number of bins or bin edges
            xlabel: X axis label
            ylabel: Y axis label
            xlim: X axis limits (min, max)
            ylim: Y axis limits (min, max)
            color: Bar color
            edgecolor: Bar edge color
            alpha: Bar transparency
            density: Whether to normalize to probability density
            label: Histogram label for legend
            grid: Whether to show grid

        Returns:
            Modified axes object
        """
        ax.hist(
            data,
            bins=bins,
            color=color,
            edgecolor=edgecolor,
            alpha=alpha,
            density=density,
            label=label,
        )

        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        self.setup_axis(ax, xlabel=xlabel, ylabel=ylabel, grid=grid)

        if label:
            ax.legend()

        return ax

    def image_plot(
        self,
        ax: matplotlib.axes.Axes,
        image: np.ndarray,
        xlabel: str = "",
        ylabel: str = "",
        title: str = "",
        cmap: str = "gray",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        colorbar: bool = True,
        colorbar_label: str = "",
        aspect: str = "equal",
        interpolation: str = "nearest",
        origin: str = "lower",
    ) -> Tuple[matplotlib.axes.Axes, matplotlib.image.AxesImage]:
        """Create an image plot with consistent styling.

        Args:
            ax: Axes object to plot on
            image: 2D image data
            xlabel: X axis label
            ylabel: Y axis label
            title: Plot title
            cmap: Colormap name
            vmin: Minimum value for colormap (auto if None)
            vmax: Maximum value for colormap (auto if None)
            colorbar: Whether to add colorbar
            colorbar_label: Label for colorbar
            aspect: Aspect ratio ('equal', 'auto', or float)
            interpolation: Interpolation method
            origin: Image origin ('lower' or 'upper')

        Returns:
            Tuple of (modified axes, image object)
        """
        # Auto-calculate vmin/vmax using percentiles if not provided
        if vmin is None:
            vmin = np.percentile(image, 1)
        if vmax is None:
            vmax = np.percentile(image, 99)

        im = ax.imshow(
            image,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect=aspect,
            interpolation=interpolation,
            origin=origin,
        )

        self.setup_axis(ax, xlabel=xlabel, ylabel=ylabel, title=title, grid=False)

        if colorbar:
            self.add_colorbar(ax, im, label=colorbar_label)

        return ax, im

    def contour_plot(
        self,
        ax: matplotlib.axes.Axes,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        xlabel: str = "x axis",
        ylabel: str = "y axis",
        title: str = "",
        levels: int = 10,
        cmap: str = "viridis",
        colorbar: bool = True,
        colorbar_label: str = "",
        filled: bool = True,
    ) -> Tuple[matplotlib.axes.Axes, Any]:
        """Create a contour plot with consistent styling.

        Args:
            ax: Axes object to plot on
            x: X coordinates (1D or 2D array)
            y: Y coordinates (1D or 2D array)
            z: Z values (2D array)
            xlabel: X axis label
            ylabel: Y axis label
            title: Plot title
            levels: Number of contour levels
            cmap: Colormap name
            colorbar: Whether to add colorbar
            colorbar_label: Label for colorbar
            filled: Whether to use filled contours (contourf) vs lines (contour)

        Returns:
            Tuple of (modified axes, contour object)
        """
        if filled:
            contours = ax.contourf(x, y, z, levels=levels, cmap=cmap)
        else:
            contours = ax.contour(x, y, z, levels=levels, cmap=cmap)

        self.setup_axis(ax, xlabel=xlabel, ylabel=ylabel, title=title, grid=False)

        if colorbar:
            self.add_colorbar(ax, contours, label=colorbar_label)

        return ax, contours

    def overlay_localisations_with_contours(
        self,
        ax: matplotlib.axes.Axes,
        image_data: np.ndarray,
        positions_x: np.ndarray,
        positions_y: np.ndarray,
        colors: Optional[Union[np.ndarray, list]] = None,
        pixelsize: float = 69.0,
        marker_size: float = 50,
        marker_style: str = 'x',
        marker_linewidth: float = 1.5,
        contour_sigma: float = 50.0,
        contour_levels: int = 3,
        contour_alpha: float = 0.6,
        show_image: bool = True,
        image_cmap: str = 'gray',
        image_vmin: Optional[float] = None,
        image_vmax: Optional[float] = None,
    ) -> matplotlib.axes.Axes:
        """
        Overlay super-resolved localisations as crosses with Gaussian contours on an image.

        This function is useful for comparing localised positions to raw camera images,
        showing both the precise localisation (cross) and the uncertainty/PSF (contour).
        Positions are automatically shifted by 0.5 pixels to align with matplotlib's
        imshow coordinate system. Axis labels and ticks are removed for clean display.

        Args:
            ax: Axes object to plot on
            image_data: 2D array of camera image data (e.g., Bayer-filtered image)
            positions_x: X coordinates of localisations in nm
            positions_y: Y coordinates of localisations in nm
            colors: Color for each localisation (RGB tuple, hex string, or matplotlib color).
                    If None, uses default cycle. If single color, applies to all.
            pixelsize: Physical pixel size in nm (default: 69.0 for camera pixels)
            marker_size: Size of cross markers (default: 50)
            marker_style: Matplotlib marker style (default: 'x' for crosses)
            marker_linewidth: Line width for markers (default: 1.5)
            contour_sigma: Standard deviation of Gaussian contour in nm (default: 50.0)
            contour_levels: Number of contour levels to draw (default: 3)
            contour_alpha: Transparency of contours (default: 0.6)
            show_image: Whether to show the background image (default: True)
            image_cmap: Colormap for background image (default: 'gray')
            image_vmin: Minimum value for image colormap (auto if None)
            image_vmax: Maximum value for image colormap (auto if None)

        Returns:
            Modified axes object with no axis labels or ticks

        Example:
            >>> fig, ax = plotter.one_column_plot()
            >>> ax = plotter.overlay_localisations_with_contours(
            ...     ax, bayer_image, x_coords, y_coords,
            ...     colors=['red', 'blue', 'green'],
            ...     contour_sigma=30.0
            ... )
        """
        # Convert positions from nm to pixels and shift by half pixel
        # to align with matplotlib's imshow coordinate system
        pos_x_pixels = positions_x / pixelsize + 0.5
        pos_y_pixels = positions_y / pixelsize + 0.5
        contour_sigma_pixels = contour_sigma / pixelsize

        # Show background image if requested
        if show_image:
            ax.imshow(
                image_data,
                cmap=image_cmap,
                vmin=image_vmin,
                vmax=image_vmax,
                origin='lower',
                extent=[0, image_data.shape[1], 0, image_data.shape[0]],
            )

        # Handle colors
        if colors is None:
            # Use default color cycle
            colors = [f'C{i%10}' for i in range(len(positions_x))]
        elif isinstance(colors, (str, tuple)):
            # Single color for all
            colors = [colors] * len(positions_x)
        elif len(colors) != len(positions_x):
            raise ValueError(f"Number of colors ({len(colors)}) must match number of positions ({len(positions_x)})")

        # Get image dimensions
        image_height, image_width = image_data.shape

        # Plot each localization
        for i, (x_px, y_px, color) in enumerate(zip(pos_x_pixels, pos_y_pixels, colors)):
            # Plot cross marker
            ax.scatter(
                x_px, y_px,
                marker=marker_style,
                s=marker_size,
                c=[color],
                linewidths=marker_linewidth,
                zorder=10,
            )

            # Create local high-resolution grid around this localization
            # Use 4*sigma extent for smooth, circular contours
            extent = 4 * contour_sigma_pixels
            grid_resolution = 0.1  # Fine resolution for smooth circles

            x_local = np.arange(x_px - extent, x_px + extent, grid_resolution)
            y_local = np.arange(y_px - extent, y_px + extent, grid_resolution)
            X_local, Y_local = np.meshgrid(x_local, y_local)

            # Generate Gaussian contour
            gaussian = np.exp(-((X_local - x_px)**2 + (Y_local - y_px)**2) / (2 * contour_sigma_pixels**2))

            # Normalize to [0, 1]
            gaussian = gaussian / gaussian.max()

            # Draw contours at specific levels
            levels = np.linspace(0.1, 0.9, contour_levels)
            ax.contour(
                X_local, Y_local, gaussian,
                levels=levels,
                colors=[color],
                linewidths=1.0,
                alpha=contour_alpha,
                zorder=9,
            )

        # Set axis limits to match image
        ax.set_xlim(0, image_width)
        ax.set_ylim(0, image_height)
        ax.set_aspect('equal')

        # Remove axis labels and ticks for cleaner image display
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')

        return ax

    def save_or_show(
        self,
        fig: matplotlib.figure.Figure,
        save_path: Optional[str] = None,
        show: bool = True,
        dpi: Optional[int] = None,
        bbox_inches: str = "tight",
        facecolor: str = "white",
        edgecolor: str = "none",
    ) -> None:
        """Save figure to file and/or display it.

        Args:
            fig: Figure to save/show
            save_path: Path to save figure (no saving if None)
            show: Whether to display the figure
            dpi: Resolution for saving (uses default if None)
            bbox_inches: Bounding box for saved figure
            facecolor: Face color for saved figure
            edgecolor: Edge color for saved figure
        """
        if save_path:
            dpi = dpi or self.config.DEFAULT_SAVE_DPI
            fig.savefig(
                save_path,
                dpi=dpi,
                bbox_inches=bbox_inches,
                facecolor=facecolor,
                edgecolor=edgecolor,
            )
            print(f"✅ Plot saved to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)


class ImagePlotMixin:
    """Mixin providing enhanced image plotting capabilities."""

    @staticmethod
    def _create_dark_to_color_cmap(color_name: str) -> LinearSegmentedColormap:
        """Create colormap from black to specified color for dark background overlays.

        Args:
            color_name: Color name or any matplotlib-compatible color string.
                Recommended bright colors for dark backgrounds:
                - 'cyan' - Excellent visibility
                - 'yellow' - Excellent visibility
                - 'orange' - Good red alternative
                - 'pink' - Bright red/magenta alternative
                - 'coral', 'salmon', 'tomato' - Various red/orange shades
                - 'lime' - Brighter green
                - 'hotpink' - Very bright pink
                Less visible on dark: 'red', 'blue', 'magenta' (use brighter alternatives)

        Returns:
            LinearSegmentedColormap: Colormap ranging from black (0,0,0) to target color.

        Example:
            >>> cmap = ImagePlotMixin._create_dark_to_color_cmap('cyan')
            >>> plt.imshow(data, cmap=cmap)
        """
        # Define common color mappings for microscopy
        # Brighter colors work better on dark backgrounds
        color_dict = {
            'cyan': (0, 1, 1),           # Bright - works great on dark
            'yellow': (1, 1, 0),         # Bright - works great on dark
            'magenta': (1, 0, 1),        # Medium brightness
            'green': (0, 1, 0),          # Bright
            'red': (1, 0, 0),            # Darker - can get lost
            'blue': (0, 0, 1),           # Darker - can get lost
            'orange': (1, 0.65, 0),      # Medium-bright - good alternative to red
            'lime': (0.75, 1, 0),        # Brighter than green
            'pink': (1, 0.4, 0.7),       # Brighter than magenta - good alternative to red
            'hotpink': (1, 0.41, 0.71),  # Similar to pink but standard name
            'deeppink': (1, 0.08, 0.58), # Darker pink
            'coral': (1, 0.5, 0.31),     # Lighter red/orange - good alternative
            'salmon': (1, 0.55, 0.41),   # Light orange/pink
            'tomato': (1, 0.39, 0.28),   # Bright red-orange
        }

        if color_name in color_dict:
            rgb = color_dict[color_name]
        else:
            # Try to parse as matplotlib color
            rgb = matplotlib.colors.to_rgb(color_name)

        # Create colormap: black (0,0,0) -> target color
        colors = [(0, 0, 0), rgb]
        n_bins = 256
        cmap = LinearSegmentedColormap.from_list(
            f'black_to_{color_name}', colors, N=n_bins
        )

        return cmap

    def create_image_with_overlay(
        self,
        ax: matplotlib.axes.Axes,
        image: np.ndarray,
        overlay_points: Optional[Dict[str, np.ndarray]] = None,
        overlay_regions: Optional[List[patches.Rectangle]] = None,
        **image_kwargs,
    ) -> matplotlib.image.AxesImage:
        """Create image plot with optional overlays.

        Args:
            ax: Axis to plot on
            image: Background image data
            overlay_points: Dictionary with 'x', 'y' arrays for point overlays
            overlay_regions: List of rectangular patches to overlay
            **image_kwargs: Arguments passed to create_image_plot

        Returns:
            AxesImage object
        """
        im = self.create_image_plot(ax, image, **image_kwargs)

        # Add point overlays
        if overlay_points:
            ax.scatter(
                overlay_points["x"],
                overlay_points["y"],
                c="red",
                s=10,
                marker="x",
                alpha=0.8,
            )

        # Add region overlays
        if overlay_regions:
            for region in overlay_regions:
                ax.add_patch(region)

        return im

    def multichannel_overlay_plot(
        self,
        axs,
        images: List[np.ndarray],
        cmaps: Optional[List[str]] = None,
        alphas: Optional[List[float]] = None,
        vmins: Optional[List[float]] = None,
        vmaxs: Optional[List[float]] = None,
        brightness_boost: Optional[List[float]] = None,
        pixelsize: float = 5.0,
        sbar: str = "on",
        scalebarsize: float = 1000,
        scalebarlabel: str = "1 μm",
        cbar: str = "off",
        cbarlabels: Optional[List[str]] = None,
        background_color: str = "black",
    ):
        """Create multichannel overlay plot of rendered super-resolution images.

        Overlays multiple grayscale images with different colormaps and transparency
        to visualize multi-color SMLM data. Uses additive blending on a dark background for
        publication-quality multi-channel visualization.

        Args:
            axs: Axes object to plot on.
            images: List of 2D numpy arrays (rendered images, one per channel).
            cmaps: List of colormap names for each channel. Defaults to ['cyan', 'yellow']
                for 2 channels. Recommended bright colors for dark backgrounds:
                'cyan', 'yellow', 'orange', 'pink', 'coral', 'salmon', 'lime', 'hotpink'.
                Avoid: 'red', 'blue', 'magenta' (too dark, use brighter alternatives).
            alphas: List of transparency values (0-1) for each channel. Defaults to 0.7
                for all channels.
            vmins: List of minimum intensity values for each channel. If None, uses
                1st percentile for each image.
            vmaxs: List of maximum intensity values for each channel. If None, uses
                99th percentile for each image.
            brightness_boost: List of multiplicative brightness factors for each channel.
                Values > 1.0 increase brightness (useful for dim filamentous structures),
                values < 1.0 decrease brightness, 1.0 = no change. Default is None (all 1.0).
                Example: [1.0, 2.5] boosts channel 2 by 2.5x to match brighter channel 1.
            pixelsize: Pixel size in nanometers for scale bar calculation.
            sbar: Whether to show scale bar ('on' or 'off').
            scalebarsize: Scale bar size in nanometers.
            scalebarlabel: Scale bar label text.
            cbar: Whether to show colorbars ('on' or 'off'). Default is 'off'.
            cbarlabels: List of colorbar labels for each channel (only used if cbar='on').
            background_color: Background color ('black' or 'white'). Default is 'black'.

        Returns:
            Modified axes object.

        Example:
            >>> # Render two channels
            >>> _, img1 = render(locs_ch1, info, oversampling=20, blur_method='gaussian')
            >>> _, img2 = render(locs_ch2, info, oversampling=20, blur_method='gaussian')
            >>>
            >>> # Create overlay plot
            >>> plotter = PublicationPlotter()
            >>> fig, ax = plotter.create_figure(figsize=(10, 10))
            >>> plotter.multichannel_overlay_plot(
            ...     ax, [img1, img2],
            ...     cmaps=['cyan', 'yellow'],
            ...     pixelsize=5.0,
            ...     scalebarsize=1000,
            ...     scalebarlabel='1 μm'
            ... )
            >>>
            >>> # Boost brightness of dim filamentous channel
            >>> plotter.multichannel_overlay_plot(
            ...     ax, [img_globular, img_filaments],
            ...     cmaps=['cyan', 'red'],
            ...     brightness_boost=[1.0, 2.5],  # Boost filaments 2.5x
            ...     pixelsize=5.0
            ... )

        Performance Notes:
            - Images should be pre-rendered using render.py functions
            - All images must have identical dimensions
            - Uses percentile-based intensity scaling by default
        """
        n_channels = len(images)

        # Validate inputs
        if n_channels < 2:
            raise ValueError("Need at least 2 images for multichannel overlay")

        # Check all images have same shape
        ref_shape = images[0].shape
        for i, img in enumerate(images[1:], 1):
            if img.shape != ref_shape:
                raise ValueError(
                    f"Image {i} shape {img.shape} doesn't match image 0 shape {ref_shape}"
                )

        # Set default colormaps (bright colors that work well on dark backgrounds)
        if cmaps is None:
            default_cmaps = ['cyan', 'yellow', 'pink', 'lime', 'orange', 'hotpink']
            cmaps = default_cmaps[:n_channels]
        elif len(cmaps) != n_channels:
            raise ValueError(f"Expected {n_channels} colormaps, got {len(cmaps)}")

        # Set default alphas
        if alphas is None:
            alphas = [0.7] * n_channels
        elif len(alphas) != n_channels:
            raise ValueError(f"Expected {n_channels} alpha values, got {len(alphas)}")

        # Validate alpha values
        for alpha in alphas:
            if not 0 <= alpha <= 1:
                raise ValueError(f"Alpha values must be in [0, 1], got {alpha}")

        # Set default brightness boost
        if brightness_boost is None:
            brightness_boost = [1.0] * n_channels
        elif len(brightness_boost) != n_channels:
            raise ValueError(f"Expected {n_channels} brightness_boost values, got {len(brightness_boost)}")

        # Validate brightness boost values
        for boost in brightness_boost:
            if boost <= 0:
                raise ValueError(f"Brightness boost must be > 0, got {boost}")

        # Set default vmin/vmax using percentiles
        if vmins is None:
            vmins = [np.percentile(img.ravel(), 1.0) for img in images]
        elif len(vmins) != n_channels:
            raise ValueError(f"Expected {n_channels} vmin values, got {len(vmins)}")

        if vmaxs is None:
            vmaxs = [np.percentile(img.ravel(), 99.0) for img in images]
        elif len(vmaxs) != n_channels:
            raise ValueError(f"Expected {n_channels} vmax values, got {len(vmaxs)}")

        # Set up axes with dark background
        axs.set_facecolor(background_color)
        axs.set_xticks([])
        axs.set_yticks([])
        axs.set_aspect('equal')

        # Create colormaps and overlay images
        image_artists = []
        for i, (image, cmap_name, alpha, vmin, vmax, boost) in enumerate(
            zip(images, cmaps, alphas, vmins, vmaxs, brightness_boost)
        ):
            # Create dark-to-color colormap
            cmap = self._create_dark_to_color_cmap(cmap_name)

            # Normalize image to [0, 1]
            img_norm = (image - vmin) / (vmax - vmin)
            img_norm = np.clip(img_norm, 0, 1)

            # Apply brightness boost
            if boost != 1.0:
                img_norm = img_norm * boost
                img_norm = np.clip(img_norm, 0, 1)

            # Plot channel
            im = axs.imshow(
                img_norm,
                cmap=cmap,
                alpha=alpha,
                origin='lower',
                interpolation=None,
            )
            image_artists.append(im)

        # Add colorbars if requested
        if cbar == "on":
            divider = make_axes_locatable(axs)

            # Position colorbars side by side on the right
            for i, im in enumerate(image_artists):
                # Calculate padding (first colorbar at 0.05, subsequent ones offset)
                pad = 0.05 + i * 0.10

                cax = divider.append_axes("right", size="2%", pad=pad)
                colorbar = plt.colorbar(im, cax=cax)

                # Set label if provided
                if cbarlabels and i < len(cbarlabels):
                    label_color = 'white' if background_color == 'black' else 'black'
                    colorbar.set_label(cbarlabels[i], color=label_color)

                # Style ticks for dark background
                if background_color == 'black':
                    colorbar.ax.yaxis.set_tick_params(color='white')
                    plt.setp(plt.getp(colorbar.ax.axes, 'yticklabels'), color='white')

        # Add scale bar if requested
        if sbar == "on":
            scalebar_color = 'white' if background_color == 'black' else 'black'

            # Use add_scalebar method from BasePlotter
            self.add_scalebar(
                axs,
                pixelsize=pixelsize,
                length_nm=scalebarsize,
                location='lower right',
                color=scalebar_color,
                label=scalebarlabel,
            )

        return axs


class TernaryPlotMixin:
    """Mixin for creating ternary (3-component) plots.

    Provides methods for plotting RGB color data on ternary diagrams
    using the mpltern library. Handles both scatter and density plots.
    """

    def create_ternary_plot(
        self,
        R: np.ndarray,
        G: np.ndarray,
        B: np.ndarray,
        colors: Optional[np.ndarray] = None,
        marker_size: float = 10,
        marker_alpha: float = 0.5,
        edge_width: float = 0.5,
        title: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
        show_grid: bool = True,
        grid_spacing: float = 0.1,
        figsize: Tuple[float, float] = (6, 5),
        rasterized: bool = False,
        **kwargs
    ) -> Tuple[Any, Any]:
        """Create a standalone ternary scatter plot for RGB data.

        This method creates a single-panel ternary plot showing the distribution
        of R, G, B values. The RGB values should be normalized (sum to 1).

        Args:
            R: Red channel values (normalized, 0-1)
            G: Green channel values (normalized, 0-1)
            B: Blue channel values (normalized, 0-1)
            colors: Optional RGBA colors for each point. If None, uses point density coloring.
            marker_size: Size of scatter markers (default: 10)
            marker_alpha: Transparency of markers (default: 0.5)
            edge_width: Width of marker edges (default: 0.5)
            title: Plot title (optional)
            labels: Dictionary with keys 'R', 'G', 'B' for axis labels (optional)
            show_grid: Whether to show grid lines (default: True)
            grid_spacing: Spacing between grid lines (default: 0.1)
            figsize: Figure size as (width, height) (default: (6, 5))
            rasterized: Whether to rasterize scatter points (default: False)
            **kwargs: Additional arguments passed to ax.scatter()

        Returns:
            Tuple of (fig, ax) where ax is a ternary axis

        Example:
            >>> from PlottingBase import PublicationPlotter
            >>> plotter = PublicationPlotter()
            >>>
            >>> # Normalize your RGB data
            >>> total = R + G + B
            >>> R_norm = R / total
            >>> G_norm = G / total
            >>> B_norm = B / total
            >>>
            >>> # Create plot
            >>> fig, ax = plotter.create_ternary_plot(
            ...     R_norm, G_norm, B_norm,
            ...     title='Color Distribution',
            ...     marker_size=5
            ... )
            >>> fig.savefig('ternary_plot.png', dpi=300, bbox_inches='tight')

        Notes:
            - Requires mpltern: `pip install mpltern`
            - RGB values should be normalized (sum to 1 for each point)
            - If not normalized, the function will normalize them automatically
        """
        try:
            import mpltern
        except ImportError:
            raise ImportError(
                "mpltern is required for ternary plots. Install with: pip install mpltern"
            )

        # Validate inputs
        if len(R) != len(G) or len(R) != len(B):
            raise ValueError("R, G, B arrays must have the same length")

        # Convert to numpy arrays if needed
        R = np.asarray(R)
        G = np.asarray(G)
        B = np.asarray(B)

        # Check for and handle normalization
        totals = R + G + B
        if not np.allclose(totals, 1.0, atol=1e-6):
            # Normalize
            R = R / totals
            G = G / totals
            B = B / totals
            print(f"Warning: RGB values were not normalized. Automatically normalized to sum=1")

        # Create figure with ternary projection
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='ternary')

        # Set up axis labels with colors
        default_labels = {'R': 'Red', 'G': 'Green', 'B': 'Blue'}
        if labels is not None:
            default_labels.update(labels)

        # Note: scatter(R, G, B) means t=R (top), l=G (left), r=B (right)
        ax.set_tlabel(default_labels['R'], color='darkred', fontsize=12)
        ax.set_llabel(default_labels['G'], color='darkgreen', fontsize=12)
        ax.set_rlabel(default_labels['B'], color='darkblue', fontsize=12)

        # Color the tick marks and tick labels
        ax.taxis.set_tick_params(colors='darkred', which='both', length=5, width=1.5)
        ax.laxis.set_tick_params(colors='darkgreen', which='both', length=5, width=1.5)
        ax.raxis.set_tick_params(colors='darkblue', which='both', length=5, width=1.5)

        # Color the axis lines (spines)
        # Note: In ternary plots, each axis runs along the OPPOSITE side:
        # - taxis (top vertex, R) runs along bottom edge = 'tside'
        # - laxis (left vertex, G) runs along right edge = 'rside'
        # - raxis (right vertex, B) runs along left edge = 'lside'
        ax.spines['lside'].set_color('darkred')      # Left edge = R axis
        ax.spines['rside'].set_color('darkgreen')    # Right edge = G axis
        ax.spines['tside'].set_color('darkblue')     # Bottom edge = B axis
        ax.spines['lside'].set_linewidth(1.5)
        ax.spines['rside'].set_linewidth(1.5)
        ax.spines['tside'].set_linewidth(1.5)

        # Set up grid with colored gridlines
        if show_grid:
            from matplotlib.ticker import MultipleLocator
            for axis in [ax.taxis, ax.laxis, ax.raxis]:
                axis.set_major_locator(MultipleLocator(grid_spacing))

            # Color the gridlines to match the axes
            ax.grid(True, which='major', alpha=0.3, linestyle='--', linewidth=0.5)
            ax.taxis.grid(color='darkred', alpha=0.3, linestyle='--', linewidth=0.5)
            ax.laxis.grid(color='darkgreen', alpha=0.3, linestyle='--', linewidth=0.5)
            ax.raxis.grid(color='darkblue', alpha=0.3, linestyle='--', linewidth=0.5)

        # Create scatter plot
        # Note: mpltern uses (t, l, r) ordering where t=top, l=left/bottom-left, r=right/bottom-right
        # For RGB with standard orientation: t=R (top), l=G (bottom-left), r=B (bottom-right)
        # We swap to: scatter(R, G, B) so Blue ends up at bottom-right as expected
        if colors is not None:
            # User-provided colors
            scatter = ax.scatter(
                R, G, B,
                c=colors,
                s=marker_size,
                alpha=marker_alpha,
                linewidths=edge_width,
                edgecolors='none' if edge_width == 0 else 'black',
                rasterized=rasterized,
                **kwargs
            )
        else:
            # No colors specified - use default blue
            scatter = ax.scatter(
                R, G, B,
                s=marker_size,
                alpha=marker_alpha,
                linewidths=edge_width,
                edgecolors='none' if edge_width == 0 else 'black',
                rasterized=rasterized,
                **kwargs
            )

        # Set title
        if title:
            ax.set_title(title, pad=20)

        # Adjust layout
        plt.tight_layout()

        return fig, ax

    def create_ternary_density(
        self,
        R: np.ndarray,
        G: np.ndarray,
        B: np.ndarray,
        gridsize: int = 50,
        cmap: str = 'viridis',
        show_colorbar: bool = True,
        title: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
        show_grid: bool = True,
        grid_spacing: float = 0.1,
        figsize: Tuple[float, float] = (7, 5),
        log_scale: bool = False,
        **kwargs
    ) -> Tuple[Any, Any]:
        """Create a ternary density plot (hexbin) for RGB data.

        This method creates a single-panel ternary plot showing the density
        distribution of R, G, B values using hexagonal binning.

        Args:
            R: Red channel values (normalized, 0-1)
            G: Green channel values (normalized, 0-1)
            B: Blue channel values (normalized, 0-1)
            gridsize: Number of hexagons in x direction (default: 50)
            cmap: Colormap name (default: 'viridis')
            show_colorbar: Whether to show colorbar (default: True)
            title: Plot title (optional)
            labels: Dictionary with keys 'R', 'G', 'B' for axis labels (optional)
            show_grid: Whether to show grid lines (default: True)
            grid_spacing: Spacing between grid lines (default: 0.1)
            figsize: Figure size as (width, height) (default: (7, 5))
            log_scale: Use logarithmic color scale (default: False)
            **kwargs: Additional arguments passed to ax.hexbin()

        Returns:
            Tuple of (fig, ax) where ax is a ternary axis

        Example:
            >>> plotter = PublicationPlotter()
            >>> fig, ax = plotter.create_ternary_density(
            ...     R_norm, G_norm, B_norm,
            ...     gridsize=100,
            ...     cmap='hot',
            ...     title='Color Density Distribution'
            ... )

        Notes:
            - Requires mpltern: `pip install mpltern`
            - RGB values should be normalized (sum to 1 for each point)
            - Use larger gridsize for smoother density visualization
        """
        try:
            import mpltern
        except ImportError:
            raise ImportError(
                "mpltern is required for ternary plots. Install with: pip install mpltern"
            )

        # Validate inputs
        if len(R) != len(G) or len(R) != len(B):
            raise ValueError("R, G, B arrays must have the same length")

        # Convert to numpy arrays if needed
        R = np.asarray(R)
        G = np.asarray(G)
        B = np.asarray(B)

        # Check for and handle normalization
        totals = R + G + B
        if not np.allclose(totals, 1.0, atol=1e-6):
            # Normalize
            R = R / totals
            G = G / totals
            B = B / totals
            print(f"Warning: RGB values were not normalized. Automatically normalized to sum=1")

        # Create figure with ternary projection
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='ternary')

        # Set up axis labels with colors
        default_labels = {'R': 'Red', 'G': 'Green', 'B': 'Blue'}
        if labels is not None:
            default_labels.update(labels)

        # Note: scatter(R, G, B) means t=R (top), l=G (left), r=B (right)
        ax.set_tlabel(default_labels['R'], color='darkred', fontsize=12)
        ax.set_llabel(default_labels['G'], color='darkgreen', fontsize=12)
        ax.set_rlabel(default_labels['B'], color='darkblue', fontsize=12)

        # Color the tick marks and tick labels
        ax.taxis.set_tick_params(colors='darkred', which='both', length=5, width=1.5)
        ax.laxis.set_tick_params(colors='darkgreen', which='both', length=5, width=1.5)
        ax.raxis.set_tick_params(colors='darkblue', which='both', length=5, width=1.5)

        # Color the axis lines (spines)
        # Note: In ternary plots, each axis runs along the OPPOSITE side:
        # - taxis (top vertex, R) runs along bottom edge = 'tside'
        # - laxis (left vertex, G) runs along right edge = 'rside'
        # - raxis (right vertex, B) runs along left edge = 'lside'
        ax.spines['lside'].set_color('darkred')      # Left edge = R axis
        ax.spines['rside'].set_color('darkgreen')    # Right edge = G axis
        ax.spines['tside'].set_color('darkblue')     # Bottom edge = B axis
        ax.spines['lside'].set_linewidth(1.5)
        ax.spines['rside'].set_linewidth(1.5)
        ax.spines['tside'].set_linewidth(1.5)

        # Set up grid with colored gridlines
        if show_grid:
            from matplotlib.ticker import MultipleLocator
            for axis in [ax.taxis, ax.laxis, ax.raxis]:
                axis.set_major_locator(MultipleLocator(grid_spacing))

            # Color the gridlines to match the axes
            ax.grid(True, which='major', alpha=0.3, linestyle='--', linewidth=0.5)
            ax.taxis.grid(color='darkred', alpha=0.3, linestyle='--', linewidth=0.5)
            ax.laxis.grid(color='darkgreen', alpha=0.3, linestyle='--', linewidth=0.5)
            ax.raxis.grid(color='darkblue', alpha=0.3, linestyle='--', linewidth=0.5)

        # Create hexbin density plot
        # Note: mpltern uses (t, l, r) ordering where t=top, l=left, r=right
        # For RGB: scatter(R, G, B) means t=R (top), l=G (left), r=B (right)
        if log_scale:
            bins = 'log'
        else:
            bins = None

        hexbin = ax.hexbin(
            R, G, B,
            gridsize=gridsize,
            cmap=cmap,
            bins=bins,
            edgecolors='none',
            rasterized=True,
            **kwargs
        )

        # Add colorbar
        if show_colorbar:
            cbar = plt.colorbar(hexbin, ax=ax, pad=0.05)
            cbar.set_label('Count' if not log_scale else 'Count (log scale)', rotation=270, labelpad=20)

        # Set title
        if title:
            ax.set_title(title, pad=20)

        # Adjust layout
        plt.tight_layout()

        return fig, ax

    def plot_ternary_kde_contours(
        self,
        ax,
        R: np.ndarray,
        G: np.ndarray,
        B: np.ndarray,
        color: str = 'blue',
        label: Optional[str] = None,
        levels: Union[List[float], str] = [0.5, 0.9, 0.99],
        bandwidth: Union[float, str] = 'scott',
        linewidths: Union[float, List[float]] = 2.0,
        linestyles: Union[str, List[str]] = 'solid',
        alpha: float = 0.8,
        grid_resolution: int = 100,
        **kwargs
    ) -> None:
        """Plot KDE contour lines on an existing ternary axis.

        This function calculates a 2D kernel density estimate in (R, G) space
        and plots confidence contours on a ternary diagram. This provides a
        cleaner visualization of dye separability compared to scatter plots with
        alpha transparency, avoiding the visual "overlap deception" issue.

        Args:
            ax: mpltern TernaryAxes object to plot on
            R: Red channel values (normalized, 0-1)
            G: Green channel values (normalized, 0-1)
            B: Blue channel values (normalized, 0-1) - used for validation only
            color: Color for contour lines (default: 'blue')
            label: Label for legend (optional)
            levels: Contour levels as:
                   - List of floats: [0.5, 0.9, 0.99] for confidence levels
                   - 'auto': Automatically choose N levels
                   - int: Number of levels to auto-generate
            bandwidth: KDE bandwidth selection:
                      - 'scott': Scott's rule (default, recommended)
                      - 'silverman': Silverman's rule
                      - float: Manual bandwidth value
            linewidths: Line width(s) for contours. Can be:
                       - Single float: Same width for all levels
                       - List of floats: Width per level (inner to outer)
            linestyles: Line style(s) for contours ('solid', 'dashed', 'dotted')
            alpha: Transparency for contour lines (0-1, default: 0.8)
            grid_resolution: Resolution of KDE evaluation grid (default: 100)
            **kwargs: Additional arguments passed to ax.tricontour()

        Returns:
            None (modifies ax in place)

        Example:
            >>> import matplotlib.pyplot as plt
            >>> import mpltern
            >>> from PlottingBase import PublicationPlotter
            >>>
            >>> # Create ternary figure
            >>> fig = plt.figure(figsize=(8, 6))
            >>> ax = fig.add_subplot(projection='ternary')
            >>>
            >>> # Plot KDE contours for multiple dyes
            >>> plotter = PublicationPlotter()
            >>> plotter.plot_ternary_kde_contours(
            ...     ax, R_dye1, G_dye1, B_dye1,
            ...     color='red', label='ATTO 655',
            ...     levels=[0.5, 0.9, 0.99],
            ...     linewidths=[1, 2, 3]
            ... )
            >>> plotter.plot_ternary_kde_contours(
            ...     ax, R_dye2, G_dye2, B_dye2,
            ...     color='blue', label='JF646',
            ...     levels=[0.5, 0.9, 0.99],
            ...     linewidths=[1, 2, 3]
            ... )
            >>> ax.legend()
            >>> plt.show()

        Notes:
            - Requires scipy for KDE calculation
            - Requires mpltern for ternary plotting
            - KDE is computed in 2D (R, G) space since B = 1 - R - G
            - Confidence levels [0.5, 0.9, 0.99] correspond to approximately
              [1.18σ, 4.60σ, 9.21σ] for 2D Gaussian distributions
            - Use varying linewidths to emphasize core vs tail of distribution
            - This method avoids the "alpha transparency deception" where
              overlapping scatter points visually exaggerate overlap

        See Also:
            create_ternary_plot: Basic ternary scatter plot
            create_ternary_density: Hexbin density plot
        """
        from scipy.stats import gaussian_kde

        # Validate inputs
        if len(R) != len(G) or len(R) != len(B):
            raise ValueError("R, G, B arrays must have the same length")

        # Convert to numpy arrays
        R = np.asarray(R, dtype=np.float64)
        G = np.asarray(G, dtype=np.float64)
        B = np.asarray(B, dtype=np.float64)

        # Remove any NaN or inf values
        valid_mask = np.isfinite(R) & np.isfinite(G) & np.isfinite(B)
        if not np.all(valid_mask):
            n_invalid = (~valid_mask).sum()
            print(f"Warning: Removed {n_invalid} invalid values from KDE calculation")
            R = R[valid_mask]
            G = G[valid_mask]
            B = B[valid_mask]

        if len(R) < 10:
            print(f"Warning: Only {len(R)} valid points for KDE. Skipping contour plot.")
            return

        # Check normalization (should sum to 1)
        totals = R + G + B
        if not np.allclose(totals, 1.0, atol=1e-3):
            print(f"Warning: RGB values not normalized (sum={np.mean(totals):.3f}). Normalizing...")
            R = R / totals
            G = G / totals
            B = B / totals

        # Compute KDE in 2D (R, G) space
        # Note: B = 1 - R - G is redundant, so we only need 2D
        data = np.vstack([R, G])

        try:
            if bandwidth == 'scott':
                kde = gaussian_kde(data, bw_method='scott')
            elif bandwidth == 'silverman':
                kde = gaussian_kde(data, bw_method='silverman')
            elif isinstance(bandwidth, (int, float)):
                kde = gaussian_kde(data, bw_method=float(bandwidth))
            else:
                raise ValueError(f"Invalid bandwidth: {bandwidth}")
        except Exception as e:
            print(f"Error creating KDE: {e}")
            print(f"Data shape: {data.shape}, R range: [{R.min():.3f}, {R.max():.3f}], G range: [{G.min():.3f}, {G.max():.3f}]")
            return

        # Create evaluation grid in (R, G) space
        # Grid covers valid ternary space: R + G <= 1, R >= 0, G >= 0
        r_grid = np.linspace(0, 1, grid_resolution)
        g_grid = np.linspace(0, 1, grid_resolution)
        R_grid, G_grid = np.meshgrid(r_grid, g_grid)

        # Mask invalid points (where R + G > 1)
        valid_ternary = (R_grid + G_grid) <= 1.0
        B_grid = 1.0 - R_grid - G_grid

        # Evaluate KDE only on valid ternary points
        valid_indices = valid_ternary.ravel()
        grid_points_all = np.vstack([R_grid.ravel(), G_grid.ravel()])
        grid_points_valid = grid_points_all[:, valid_indices]

        try:
            kde_values_valid = kde(grid_points_valid)
        except Exception as e:
            print(f"Error evaluating KDE: {e}")
            return

        # Create full KDE array with zeros for invalid points
        kde_values_full = np.zeros(R_grid.size)
        kde_values_full[valid_indices] = kde_values_valid
        kde_values = kde_values_full.reshape(R_grid.shape)

        # Determine contour levels
        if isinstance(levels, str) and levels == 'auto':
            # Auto-select levels based on KDE values
            valid_kde = kde_values[valid_ternary]
            levels_to_plot = np.percentile(valid_kde, [10, 30, 50, 70, 90])
        elif isinstance(levels, int):
            # Generate N evenly-spaced levels
            valid_kde = kde_values[valid_ternary]
            levels_to_plot = np.linspace(valid_kde.min(), valid_kde.max(), levels)
        else:
            # Convert confidence levels to density levels
            # For 2D Gaussian: P(inside ellipse) = 1 - exp(-r²/2)
            # Solving: r² = -2*ln(1-P)
            # Density at radius r: exp(-r²/2) / (2π σ²)
            # But we use empirical approach: percentiles of KDE values
            valid_kde = kde_values[valid_ternary]
            levels_array = np.asarray(levels)

            # Map confidence levels to KDE density thresholds
            # Use only non-zero KDE values (many grid points are outside data region)
            nonzero_kde = valid_kde[valid_kde > 0]

            if len(nonzero_kde) < 10:
                print(f"Warning: Only {len(nonzero_kde)} non-zero KDE values. Using all valid values.")
                nonzero_kde = valid_kde

            # Sort KDE values (highest to lowest)
            sorted_kde = np.sort(nonzero_kde)[::-1]
            n_points = len(sorted_kde)

            levels_to_plot = []
            for conf in levels_array:
                # For confidence level conf, we want the threshold where
                # conf% of the probability mass is above it
                # For a 2D Gaussian: 50% → ~0.39 of peak, 90% → ~0.105 of peak, 99% → ~0.018 of peak
                # Use this as approximation
                if conf == 0.5:
                    # 50% contour: roughly 0.4 of maximum
                    level_val = sorted_kde[0] * 0.4
                elif conf == 0.9:
                    # 90% contour: roughly 0.1 of maximum
                    level_val = sorted_kde[0] * 0.1
                elif conf == 0.99:
                    # 99% contour: roughly 0.02 of maximum
                    level_val = sorted_kde[0] * 0.02
                else:
                    # General case: use empirical quantile
                    idx = int(conf * n_points)
                    if idx >= n_points:
                        idx = n_points - 1
                    level_val = sorted_kde[idx]

                levels_to_plot.append(level_val)
            levels_to_plot = np.array(levels_to_plot)

        # Ensure levels are sorted (lowest to highest) for proper contour plotting
        levels_to_plot = np.sort(levels_to_plot)

        # Handle linewidths and linestyles
        if isinstance(linewidths, (int, float)):
            linewidths_list = [linewidths] * len(levels_to_plot)
        else:
            linewidths_list = list(linewidths)
            if len(linewidths_list) < len(levels_to_plot):
                # Repeat last value
                linewidths_list += [linewidths_list[-1]] * (len(levels_to_plot) - len(linewidths_list))

        if isinstance(linestyles, str):
            linestyles_list = [linestyles] * len(levels_to_plot)
        else:
            linestyles_list = list(linestyles)
            if len(linestyles_list) < len(levels_to_plot):
                linestyles_list += [linestyles_list[-1]] * (len(levels_to_plot) - len(linestyles_list))

        # Plot contours on ternary axis
        # mpltern tricontour expects (t, l, r) = (R, G, B) and values
        try:
            contour = ax.tricontour(
                R_grid.ravel(),
                G_grid.ravel(),
                B_grid.ravel(),
                kde_values.ravel(),
                levels=levels_to_plot,
                colors=[color] * len(levels_to_plot),
                linewidths=linewidths_list,
                linestyles=linestyles_list,
                alpha=alpha,
                **kwargs
            )

            # Add label for legend (only to first contour level)
            if label is not None:
                # Create a dummy line for legend
                from matplotlib.lines import Line2D
                legend_line = Line2D([0], [0], color=color, linewidth=linewidths_list[0],
                                    linestyle=linestyles_list[0], alpha=alpha, label=label)
                # Store it for potential legend creation
                if not hasattr(ax, '_kde_legend_handles'):
                    ax._kde_legend_handles = []
                ax._kde_legend_handles.append(legend_line)

        except Exception as e:
            print(f"Error plotting contours: {e}")
            print(f"Levels: {levels_to_plot}")
            print(f"KDE values range: [{np.nanmin(kde_values):.6f}, {np.nanmax(kde_values):.6f}]")
            return


class DatashaderMixin:
    """Mixin for handling large datasets with datashader when available.

    This mixin automatically switches between matplotlib and datashader rendering
    based on dataset size, with user-tunable thresholds for optimal performance.
    """

    def __init__(self, *args, datashader_threshold: int = 1000, **kwargs):
        """Initialize DatashaderMixin.

        Args:
            datashader_threshold: Number of points above which to use datashader.
                Default is 1000. Set to None to disable auto-switching.
            *args, **kwargs: Passed to parent class
        """
        super().__init__(*args, **kwargs)

        self.datashader_threshold = datashader_threshold
        self._datashader_warned = False  # Track if we've shown warning

        # Try to import datashader components
        try:
            import datashader as ds
            import datashader.transfer_functions as tf
            import pandas as pd

            self.ds = ds
            self.tf = tf
            self.pd = pd
            self.datashader_available = True
        except ImportError:
            self.datashader_available = False
            if datashader_threshold is not None and not self._datashader_warned:
                warnings.warn(
                    "Datashader not available. Install with 'pip install datashader' "
                    "for better performance with large datasets (>1k points). "
                    "Falling back to matplotlib which may be slow for >10k points."
                )
                self._datashader_warned = True

    def plot_large_scatter(
        self,
        ax: matplotlib.axes.Axes,
        x: np.ndarray,
        y: np.ndarray,
        c: Optional[np.ndarray] = None,
        threshold: Optional[int] = None,
        canvas_size: Optional[Tuple[int, int]] = None,
        cmap: str = "viridis",
        downsample: bool = False,
        downsample_factor: int = 10,
        **scatter_kwargs,
    ) -> Union[matplotlib.image.AxesImage, matplotlib.collections.PathCollection]:
        """Plot scatter data, using datashader for large datasets.

        This method automatically selects the best rendering method based on
        dataset size. For datasets larger than the threshold, it uses datashader
        for fast rendering. For smaller datasets, it uses matplotlib for better
        interactivity.

        Args:
            ax: Axis to plot on
            x: X coordinates
            y: Y coordinates
            c: Color values (optional, for matplotlib or datashader aggregation)
            threshold: Use datashader if more than this many points.
                If None, uses self.datashader_threshold. Set to None to force matplotlib.
            canvas_size: Canvas size for datashader rendering (width, height).
                If None, uses figure DPI and size.
            cmap: Colormap for datashader or matplotlib
            downsample: If True and using matplotlib, downsample large datasets
            downsample_factor: Keep every Nth point when downsampling
            **scatter_kwargs: Arguments for matplotlib scatter (s, alpha, marker, etc.)

        Returns:
            Either AxesImage (datashader) or PathCollection (matplotlib)

        Examples:
            >>> # Small dataset - uses matplotlib
            >>> plotter.plot_large_scatter(ax, x[:500], y[:500])

            >>> # Large dataset - auto-switches to datashader
            >>> plotter.plot_large_scatter(ax, x, y)  # 50k points

            >>> # Force matplotlib with downsampling
            >>> plotter.plot_large_scatter(ax, x, y, threshold=None,
            ...                           downsample=True, downsample_factor=10)
        """
        # Determine threshold
        if threshold is None:
            threshold = (
                self.datashader_threshold if self.datashader_threshold else float("inf")
            )

        n_points = len(x)
        use_datashader = (
            n_points > threshold and self.datashader_available and threshold is not None
        )

        if use_datashader:
            # Use datashader for large datasets
            if canvas_size is None:
                # Auto-determine canvas size from figure
                bbox = ax.get_window_extent().transformed(
                    ax.figure.dpi_scale_trans.inverted()
                )
                canvas_size = (
                    int(bbox.width * ax.figure.dpi),
                    int(bbox.height * ax.figure.dpi),
                )

            return self._plot_with_datashader(ax, x, y, c, canvas_size, cmap)
        else:
            # Use matplotlib
            if downsample and n_points > threshold:
                # Downsample for preview
                indices = np.arange(0, n_points, downsample_factor)
                x_down = x[indices]
                y_down = y[indices]
                c_down = c[indices] if c is not None else None

                scatter = ax.scatter(
                    x_down, y_down, c=c_down, cmap=cmap, **scatter_kwargs
                )
                ax.set_title(
                    ax.get_title() + f" (showing {len(indices)}/{n_points} points)"
                )
                return scatter
            else:
                return ax.scatter(x, y, c=c, cmap=cmap, **scatter_kwargs)

    def _plot_with_datashader(
        self,
        ax: matplotlib.axes.Axes,
        x: np.ndarray,
        y: np.ndarray,
        c: Optional[np.ndarray],
        canvas_size: Tuple[int, int],
        cmap: str,
    ) -> matplotlib.image.AxesImage:
        """Create datashader plot.

        Args:
            ax: Axis to plot on
            x: X coordinates
            y: Y coordinates
            c: Color values (optional, used for aggregation)
            canvas_size: Canvas size (width, height)
            cmap: Colormap name

        Returns:
            AxesImage object
        """
        # Create DataFrame
        if c is not None:
            df = self.pd.DataFrame({"x": x, "y": y, "c": c})
            agg_column = "c"
            agg_func = "mean"  # Average color values
        else:
            df = self.pd.DataFrame({"x": x, "y": y})
            agg_column = None
            agg_func = "count"  # Count points per pixel

        # Create canvas with bounds
        x_range = (float(x.min()), float(x.max()))
        y_range = (float(y.min()), float(y.max()))

        canvas = self.ds.Canvas(
            plot_width=canvas_size[0],
            plot_height=canvas_size[1],
            x_range=x_range,
            y_range=y_range,
        )

        # Aggregate points
        if agg_column:
            agg = canvas.points(df, "x", "y", agg=self.ds.mean(agg_column))
        else:
            agg = canvas.points(df, "x", "y")

        # Create image with proper colormap
        try:
            # Try using colormap from colorcet if available
            img = self.tf.shade(agg, cmap=cmap, how="linear")
        except (ValueError, KeyError):
            # Fall back to default if colormap not found
            img = self.tf.shade(agg, how="linear")

        # Convert to numpy array and display
        img_array = img.to_numpy()
        extent = [x_range[0], x_range[1], y_range[0], y_range[1]]

        im = ax.imshow(
            img_array,
            extent=extent,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
        )

        # Add note about rendering method
        n_points = len(x)
        ax.text(
            0.02,
            0.98,
            f"Datashader: {n_points:,} points",
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            alpha=0.7,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
        )

        return im

    def plot_multi_dataset_scatter(
        self,
        ax: matplotlib.axes.Axes,
        datasets: List[Dict[str, np.ndarray]],
        labels: Optional[List[str]] = None,
        colors: Optional[List[str]] = None,
        threshold: Optional[int] = None,
        canvas_size: Optional[Tuple[int, int]] = None,
        alpha: float = 0.6,
        sizes: Optional[Union[float, List[float]]] = None,
        **scatter_kwargs,
    ) -> List[Union[matplotlib.image.AxesImage, matplotlib.collections.PathCollection]]:
        """Plot multiple datasets efficiently, auto-selecting best rendering method.

        For large multi-dataset plots, this method intelligently chooses between:
        - Individual datashader layers (best for very large datasets)
        - Matplotlib scatter with downsampling (good for moderate datasets)
        - Standard matplotlib scatter (best for small datasets)

        Args:
            ax: Axis to plot on
            datasets: List of dicts with 'x' and 'y' keys
            labels: Labels for each dataset (for legend)
            colors: Colors for each dataset
            threshold: Total points threshold for datashader
            canvas_size: Canvas size for datashader
            alpha: Transparency for matplotlib rendering
            sizes: Marker sizes (single or per-dataset)
            **scatter_kwargs: Additional matplotlib scatter kwargs

        Returns:
            List of plot objects (one per dataset)

        Example:
            >>> datasets = [
            ...     {'x': locs1['xc'], 'y': locs1['yc']},
            ...     {'x': locs2['xc'], 'y': locs2['yc']}
            ... ]
            >>> plotter.plot_multi_dataset_scatter(ax, datasets,
            ...     labels=['Fiducial 1', 'Fiducial 2'])
        """
        if threshold is None:
            threshold = (
                self.datashader_threshold if self.datashader_threshold else float("inf")
            )

        total_points = sum(len(d["x"]) for d in datasets)
        use_datashader = (
            total_points > threshold
            and self.datashader_available
            and threshold is not None
        )

        # Prepare colors and sizes
        if colors is None:
            prop_cycle = plt.rcParams["axes.prop_cycle"]
            colors = prop_cycle.by_key()["color"]

        if isinstance(sizes, (int, float)):
            sizes = [sizes] * len(datasets)
        elif sizes is None:
            sizes = [self.config.DEFAULT_MARKER_SIZE] * len(datasets)

        plots = []

        if use_datashader and len(datasets) > 1:
            # Combine datasets with category labels for datashader
            all_x = np.concatenate([d["x"] for d in datasets])
            all_y = np.concatenate([d["y"] for d in datasets])
            categories = np.concatenate(
                [np.full(len(d["x"]), i, dtype=int) for i, d in enumerate(datasets)]
            )

            # Convert to categorical for datashader
            df = self.pd.DataFrame({"x": all_x, "y": all_y, "category": categories})
            df["category"] = df["category"].astype("category")

            if canvas_size is None:
                bbox = ax.get_window_extent().transformed(
                    ax.figure.dpi_scale_trans.inverted()
                )
                canvas_size = (
                    int(bbox.width * ax.figure.dpi),
                    int(bbox.height * ax.figure.dpi),
                )

            x_range = (float(all_x.min()), float(all_x.max()))
            y_range = (float(all_y.min()), float(all_y.max()))

            canvas = self.ds.Canvas(
                plot_width=canvas_size[0],
                plot_height=canvas_size[1],
                x_range=x_range,
                y_range=y_range,
            )

            # Aggregate by category
            agg = canvas.points(
                df, "x", "y", agg=self.ds.by("category", self.ds.count())
            )

            # Create color mapping
            color_key = {i: colors[i % len(colors)] for i in range(len(datasets))}
            img = self.tf.shade(
                agg, color_key=color_key, how="linear", alpha=int(alpha * 255)
            )

            img_array = img.to_numpy()
            extent = [x_range[0], x_range[1], y_range[0], y_range[1]]

            im = ax.imshow(
                img_array,
                extent=extent,
                origin="lower",
                aspect="auto",
                interpolation="nearest",
            )
            plots.append(im)

            # Add legend manually for datashader
            if labels:
                from matplotlib.patches import Patch

                legend_elements = [
                    Patch(facecolor=colors[i % len(colors)], label=labels[i])
                    for i in range(len(datasets))
                ]
                ax.legend(handles=legend_elements)

            # Add rendering note
            ax.text(
                0.02,
                0.98,
                f"Datashader: {total_points:,} points",
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment="top",
                alpha=0.7,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
            )

        else:
            # Use matplotlib for each dataset
            for i, dataset in enumerate(datasets):
                x, y = dataset["x"], dataset["y"]
                color = colors[i % len(colors)]
                size = sizes[i % len(sizes)]
                label = labels[i] if labels else None

                # Check if individual dataset is too large
                if len(x) > threshold and threshold is not None:
                    # Downsample this dataset
                    downsample_factor = max(1, len(x) // threshold)
                    indices = np.arange(0, len(x), downsample_factor)
                    x_down = x[indices]
                    y_down = y[indices]

                    scatter = ax.scatter(
                        x_down,
                        y_down,
                        c=color,
                        s=size,
                        alpha=alpha,
                        label=label,
                        **scatter_kwargs,
                    )

                    if i == 0:  # Only show note once
                        ax.text(
                            0.02,
                            0.98,
                            f"Downsampled: {len(indices):,}/{total_points:,} points",
                            transform=ax.transAxes,
                            fontsize=8,
                            verticalalignment="top",
                            alpha=0.7,
                            bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.5),
                        )
                else:
                    scatter = ax.scatter(
                        x,
                        y,
                        c=color,
                        s=size,
                        alpha=alpha,
                        label=label,
                        **scatter_kwargs,
                    )

                plots.append(scatter)

            if labels:
                ax.legend()

        return plots

    def create_preview_plot(
        self,
        ax: matplotlib.axes.Axes,
        x: np.ndarray,
        y: np.ndarray,
        preview_points: int = 5000,
        method: str = "random",
        **scatter_kwargs,
    ) -> matplotlib.collections.PathCollection:
        """Create fast preview plot by intelligently downsampling large datasets.

        Args:
            ax: Axis to plot on
            x: X coordinates
            y: Y coordinates
            preview_points: Target number of points for preview
            method: Downsampling method ('random', 'uniform', 'density')
                - 'random': Random sampling
                - 'uniform': Evenly spaced sampling
                - 'density': Density-aware sampling (keeps more points in sparse regions)
            **scatter_kwargs: Additional scatter plot arguments

        Returns:
            PathCollection from scatter plot

        Example:
            >>> # Quick preview of 100k points showing 5k representative points
            >>> plotter.create_preview_plot(ax, x, y, preview_points=5000, method='density')
        """
        n_points = len(x)

        if n_points <= preview_points:
            # No downsampling needed
            return ax.scatter(x, y, **scatter_kwargs)

        # Select indices based on method
        if method == "random":
            indices = np.random.choice(n_points, preview_points, replace=False)
        elif method == "uniform":
            step = n_points // preview_points
            indices = np.arange(0, n_points, step)[:preview_points]
        elif method == "density":
            # Density-aware sampling: use 2D histogram to identify sparse/dense regions
            # Keep more points from sparse regions for better coverage
            hist, x_edges, y_edges = np.histogram2d(x, y, bins=50)

            # Assign each point to a bin
            x_bin_idx = np.digitize(x, x_edges) - 1
            y_bin_idx = np.digitize(y, y_edges) - 1

            # Clip to valid range
            x_bin_idx = np.clip(x_bin_idx, 0, hist.shape[0] - 1)
            y_bin_idx = np.clip(y_bin_idx, 0, hist.shape[1] - 1)

            # Calculate sampling probability (inversely proportional to density)
            densities = hist[x_bin_idx, y_bin_idx]
            # Avoid division by zero
            probs = 1.0 / (densities + 1)
            probs /= probs.sum()

            indices = np.random.choice(n_points, preview_points, replace=False, p=probs)
        else:
            raise ValueError(f"Unknown downsampling method: {method}")

        # Plot downsampled data
        scatter = ax.scatter(x[indices], y[indices], **scatter_kwargs)

        # Add note about preview
        ax.text(
            0.02,
            0.98,
            f"Preview: {len(indices):,}/{n_points:,} points ({method})",
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            alpha=0.7,
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
        )

        return scatter


class PublicationPlotter(TernaryPlotMixin, BasePlotter, ImagePlotMixin):
    """Publication-quality plotter with enhanced image and ternary plot capabilities.

    This class provides high-quality plotting functionality suitable for
    scientific publications, with consistent styling and professional appearance.
    Includes support for ternary (3-component) plots via the TernaryPlotMixin.
    """

    def __init__(self, poster: bool = False, dark_background: bool = False):
        """Initialize publication plotter.

        Args:
            poster: Whether to use poster-style formatting (12pt fonts vs 7pt, 1.0pt lines vs 0.5pt)
            dark_background: Whether to use dark background theme
        """
        # Create config with proper poster and dark background flags
        # Must set flags BEFORE __post_init__ runs (which happens at creation time for dataclass)
        config = PlottingConfig(poster_mode=poster, dark_background=dark_background)

        super().__init__(config)

        # Store mode for helper methods
        self.poster = poster
        self.dark_background = dark_background

    def one_column_plot(
        self,
        npanels: int = 1,
        ratios: Optional[List[float]] = None,
        height: Optional[float] = None,
        width: Optional[float] = None,
    ) -> Tuple[matplotlib.figure.Figure, Union[matplotlib.axes.Axes, np.ndarray]]:
        """Create a one-column width publication-quality figure.

        Follows journal standards:
        - Default width: 3.33 inches (240 pt) - one-column max
        - Default height: 3.5 inches per panel (capped at 8.25 inches)
        - Proper font hierarchy: 7pt ticks, 8pt axis labels, 6pt legends
        - Line width: 0.5pt (standard) or 1.0pt (poster)

        Args:
            npanels: Number of vertical panels (rows)
            ratios: Height ratios for panels. If None, uses equal heights.
            height: Override total height in inches (capped at 8.25")
            width: Override width in inches (capped at 3.33" for publications)

        Returns:
            Tuple of (figure, axes). If npanels=1, returns single axis; otherwise array.

        Example:
            >>> plotter = PublicationPlotter()
            >>> fig, ax = plotter.one_column_plot()  # Single panel, 3.33" × 3.5"
            >>> fig, axs = plotter.one_column_plot(npanels=2, ratios=[2, 1])  # Two panels
        """
        # Default to equal ratios
        if ratios is None:
            ratios = [1] * npanels

        # Validate ratios
        if len(ratios) != npanels:
            raise ValueError(
                f"Number of ratios ({len(ratios)}) must match npanels ({npanels})"
            )

        # Calculate dimensions with publication standards
        if width is not None:
            xsize = width
            if width > PublicationConstants.ONE_COLUMN_WIDTH:
                warnings.warn(
                    f"Width {width:.2f}\" exceeds one-column standard "
                    f"({PublicationConstants.ONE_COLUMN_WIDTH}\")",
                    UserWarning
                )
        else:
            xsize = PublicationConstants.ONE_COLUMN_WIDTH

        if height is not None:
            ysize = height
            if height > PublicationConstants.MAX_HEIGHT:
                warnings.warn(
                    f"Height {height:.2f}\" exceeds maximum "
                    f"({PublicationConstants.MAX_HEIGHT}\")",
                    UserWarning
                )
        else:
            # Default: 3.5 inches per panel, capped at max height
            ysize = min(
                PublicationConstants.DEFAULT_PANEL_HEIGHT_RATIO * npanels,
                PublicationConstants.MAX_HEIGHT
            )

        # Create figure with proper dimensions and DPI
        fig, axs = plt.subplots(
            nrows=npanels,
            ncols=1,
            figsize=(xsize, ysize),
            height_ratios=ratios,
            frameon=False,
            squeeze=False,
            dpi=self.config.DEFAULT_DPI,
        )

        # Configure tick parameters for all axes
        for ax in axs.flat:
            ax.xaxis.set_tick_params(
                width=self.config.line_width,
                length=self.config.tick_length
            )
            ax.yaxis.set_tick_params(
                width=self.config.line_width,
                length=self.config.tick_length
            )

        # Return axes with appropriate squeeze behavior
        if npanels == 1:
            return fig, axs[0, 0]
        else:
            return fig, axs[:, 0]

    def two_column_plot(
        self,
        nrows: int = 1,
        ncols: int = 1,
        height_ratios: Optional[List[float]] = None,
        width_ratios: Optional[List[float]] = None,
        width: Optional[float] = None,
        height: Optional[float] = None,
        big: bool = False,
    ) -> Tuple[matplotlib.figure.Figure, Union[matplotlib.axes.Axes, np.ndarray]]:
        """Create a two-column width publication-quality figure.

        Follows journal standards:
        - Default width: 6.69 inches (17 cm) - two-column max
        - Default height: 3.0 inches per row (5.0 if big=True)
        - Proper font hierarchy: 7pt ticks, 8pt axis labels, 6pt legends
        - Line width: 0.5pt (standard) or 1.0pt (poster)

        Args:
            nrows: Number of rows
            ncols: Number of columns
            height_ratios: Height ratios for rows. If None, uses equal heights.
            width_ratios: Width ratios for columns. If None, uses equal widths.
            width: Override total width in inches (capped at 6.69" for publications)
            height: Override total height in inches (capped at 8.25")
            big: Use larger default sizing (5" per row/col) for posters

        Returns:
            Tuple of (figure, axes). Returns appropriately squeezed array based on dimensions.

        Example:
            >>> plotter = PublicationPlotter()
            >>> fig, axs = plotter.two_column_plot(nrows=2, ncols=2)  # 2×2 grid
            >>> fig, (ax1, ax2) = plotter.two_column_plot(nrows=1, ncols=2)  # 1×2 row
        """
        # Default to equal ratios
        if height_ratios is None:
            height_ratios = [1] * nrows
        if width_ratios is None:
            width_ratios = [1] * ncols

        # Validate ratios
        if len(height_ratios) != nrows:
            raise ValueError(
                f"Number of height_ratios ({len(height_ratios)}) must match nrows ({nrows})"
            )
        if len(width_ratios) != ncols:
            raise ValueError(
                f"Number of width_ratios ({len(width_ratios)}) must match ncols ({ncols})"
            )

        # Calculate dimensions
        if width is not None:
            xsize = width
            if width > PublicationConstants.TWO_COLUMN_WIDTH and not big:
                warnings.warn(
                    f"Width {width:.2f}\" exceeds two-column standard "
                    f"({PublicationConstants.TWO_COLUMN_WIDTH}\")",
                    UserWarning
                )
        else:
            if big:
                xsize = 5.0 * ncols
            else:
                xsize = PublicationConstants.TWO_COLUMN_WIDTH

        if height is not None:
            ysize = height
            if height > PublicationConstants.MAX_HEIGHT:
                warnings.warn(
                    f"Height {height:.2f}\" exceeds maximum "
                    f"({PublicationConstants.MAX_HEIGHT}\")",
                    UserWarning
                )
        else:
            if big:
                ysize = min(5.0 * nrows, PublicationConstants.MAX_HEIGHT)
            else:
                ysize = min(
                    PublicationConstants.DEFAULT_TWO_COLUMN_ROW_HEIGHT * nrows,
                    PublicationConstants.MAX_HEIGHT
                )

        # Create figure with proper dimensions and DPI
        fig, axs = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(xsize, ysize),
            height_ratios=height_ratios,
            width_ratios=width_ratios,
            frameon=False,
            squeeze=False,
            dpi=self.config.DEFAULT_DPI,
        )

        # Configure tick parameters for all axes
        for ax in axs.flat:
            ax.xaxis.set_tick_params(
                width=self.config.line_width,
                length=self.config.tick_length
            )
            ax.yaxis.set_tick_params(
                width=self.config.line_width,
                length=self.config.tick_length
            )

        # Return axes with appropriate squeeze behavior
        if nrows == 1 and ncols == 1:
            return fig, axs[0, 0]
        elif nrows == 1:
            return fig, axs[0, :]
        elif ncols == 1:
            return fig, axs[:, 0]
        else:
            return fig, axs


class AnalysisPlotter(TernaryPlotMixin, DatashaderMixin, ImagePlotMixin, BasePlotter):
    """Analysis-focused plotter with large dataset handling and ternary plots.

    This class is optimised for interactive data analysis and exploration,
    with support for large datasets and quick visualisation. Automatically
    switches to datashader for datasets >1k points (configurable).
    Includes support for ternary (3-component) plots via the TernaryPlotMixin.

    Note: MRO is TernaryPlotMixin -> DatashaderMixin -> ImagePlotMixin -> BasePlotter
    to ensure proper initialization order.
    """

    def __init__(self, datashader_threshold: int = 1000):
        """Initialize analysis plotter with defaults optimised for exploration.

        Args:
            datashader_threshold: Number of points above which to use datashader
                for faster rendering. Default is 1000. Set to None to disable
                auto-switching and always use matplotlib.
        """
        config = PlottingConfig()
        config.DEFAULT_FIGSIZE = (10, 6)
        config.DEFAULT_DPI = 100  # Lower DPI for faster rendering

        # Initialize with proper MRO
        super().__init__(config=config, datashader_threshold=datashader_threshold)
