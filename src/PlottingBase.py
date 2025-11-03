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


@dataclass
class PlottingConfig:
    """Configuration class for consistent plotting styles across pyBayerSMLM."""

    # Default figure properties
    DEFAULT_DPI: int = 150
    DEFAULT_FIGSIZE: Tuple[float, float] = (8, 6)
    DEFAULT_SAVE_DPI: int = 300

    # Color schemes
    DEFAULT_COLORMAP: str = "gist_gray"
    DEFAULT_SCATTER_COLOR: str = "blue"
    DEFAULT_GRID_COLOR: str = "gray"
    DEFAULT_GRID_ALPHA: float = 0.3

    # Image display percentiles
    DEFAULT_VMIN_PERCENTILE: float = 1.0
    DEFAULT_VMAX_PERCENTILE: float = 99.0

    # Font and line properties
    DEFAULT_FONT_SIZE: int = 12
    DEFAULT_LINE_WIDTH: float = 1.0
    DEFAULT_MARKER_SIZE: float = 1.0

    # Colorbar properties
    DEFAULT_COLORBAR_WIDTH: str = "5%"
    DEFAULT_COLORBAR_PAD: float = 0.05

    # Scale bar properties
    DEFAULT_SCALEBAR_COLOR: str = "white"
    DEFAULT_SCALEBAR_FONTSIZE: int = 10

    def __post_init__(self):
        """Set up matplotlib parameters based on configuration."""
        matplotlib.rcParams.update(
            {
                "font.size": self.DEFAULT_FONT_SIZE,
                "axes.linewidth": self.DEFAULT_LINE_WIDTH,
                "xtick.major.width": self.DEFAULT_LINE_WIDTH,
                "ytick.major.width": self.DEFAULT_LINE_WIDTH,
            }
        )


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

    def create_subplots(
        self,
        nrows: int = 1,
        ncols: int = 1,
        figsize: Optional[Tuple[float, float]] = None,
        height_ratios: Optional[List[float]] = None,
        width_ratios: Optional[List[float]] = None,
        **kwargs,
    ) -> Tuple[matplotlib.figure.Figure, Union[matplotlib.axes.Axes, np.ndarray]]:
        """Create standardised subplots.

        Args:
            nrows: Number of rows
            ncols: Number of columns
            figsize: Figure size in inches (width, height)
            height_ratios: Heights of the rows
            width_ratios: Widths of the columns
            **kwargs: Additional arguments passed to plt.subplots

        Returns:
            Tuple of (figure, axes)
        """
        figsize = figsize or self.config.DEFAULT_FIGSIZE

        # Calculate appropriate figsize for subplots
        if figsize == self.config.DEFAULT_FIGSIZE:
            figsize = (figsize[0] * ncols, figsize[1] * nrows)

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=figsize,
            gridspec_kw={
                "height_ratios": height_ratios,
                "width_ratios": width_ratios,
            },
            **kwargs,
        )

        return fig, axes

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


class PublicationPlotter(BasePlotter, ImagePlotMixin):
    """Publication-quality plotter with enhanced image capabilities.

    This class provides high-quality plotting functionality suitable for
    scientific publications, with consistent styling and professional appearance.
    """

    def __init__(self, poster: bool = False, dark_background: bool = False):
        """Initialize publication plotter.

        Args:
            poster: Whether to use poster-style formatting (larger fonts, etc.)
            dark_background: Whether to use dark background theme
        """
        config = PlottingConfig()

        if poster:
            config.DEFAULT_FONT_SIZE = 16
            config.DEFAULT_FIGSIZE = (12, 8)

        if dark_background:
            config.DEFAULT_GRID_COLOR = "white"
            config.DEFAULT_SCALEBAR_COLOR = "white"

        super().__init__(config)

        if dark_background:
            plt.style.use("dark_background")


class AnalysisPlotter(DatashaderMixin, ImagePlotMixin, BasePlotter):
    """Analysis-focused plotter with large dataset handling.

    This class is optimised for interactive data analysis and exploration,
    with support for large datasets and quick visualisation. Automatically
    switches to datashader for datasets >1k points (configurable).

    Note: MRO is DatashaderMixin -> ImagePlotMixin -> BasePlotter to ensure
    proper initialization order.
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
