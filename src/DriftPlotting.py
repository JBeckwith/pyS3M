"""
DriftPlotting.py

Plotting utilities for drift correction visualization.
Extracted from DriftCorrectionFunctions.py for better code organisation.
Now uses consolidated plotting base classes for consistency.

:authors: Claude Code
:copyright: Copyright (c) 2025 pyBayerSMLM
"""

from typing import Optional, List, Dict, Any, Tuple
import numpy as np

# Use centralised import management
from ImportManager import (
    get_module,
    is_available,
    safe_import,
    get_postprocess,
    get_render,
    get_imageprocess,
)
from PlottingBase import AnalysisPlotter, PlottingConfig, PublicationPlotter

# Get modules through import manager
plt = get_module("matplotlib.pyplot")
patches = safe_import(
    "matplotlib.patches",
    error_message="matplotlib.patches not available - some plotting features may not work",
)

# Get local modules through import manager
render = get_render()
imageprocess = get_imageprocess()
postprocess = get_postprocess()

# Lazy load PublicationPlotter for backwards compatibility
_PublicationPlotter = None


def _ensure_plotter():
    """Lazy load PublicationPlotter class."""
    global _PublicationPlotter
    if _PublicationPlotter is None:
        try:
            from PlottingBase import PublicationPlotter as PP
            _PublicationPlotter = PP
        except ImportError:
            _PublicationPlotter = None
    return _PublicationPlotter

# We'll need to handle FiducialDetectionResult via parameter typing
# to avoid circular imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from DriftCorrectionFunctions import FiducialDetectionResult


class DriftPlotter(AnalysisPlotter):
    """Specialised drift correction plotting utilities.

    Inherits from AnalysisPlotter for consistent styling and enhanced functionality
    including large dataset support via datashader when available.
    """

    def __init__(self):
        """Initialize the drift plotter with analysis-optimised settings."""
        super().__init__()

    def plot_fiducial_detection_steps(
        self,
        image: np.ndarray,
        hist: tuple,
        threshold: float,
        all_picks: list,
        valid_picks: list,
        result: "FiducialDetectionResult",
        info: List[dict],
        save_path: Optional[str] = None,
    ) -> None:
        """Create step-by-step visualization of fiducial detection process."""
        plotter_class = _ensure_plotter()
        if plotter_class is None:
            print("⚠️ PublicationPlotter not available, skipping step-by-step plots")
            return

        try:
            # Create plotter instance
            plotter = plotter_class(poster=False, dark_background=False)

            # Create the comprehensive figure
            fig, axes = plotter.two_column_plot(
                ncolumns=2,
                nrows=3,
                widthratio=[1.0, 1.0],
                heightratio=[1.0, 1.0, 0.8],
                width=14,
                height=12,
            )

            pixelsize_nm = info[0]["Pixelsize"] * 1000 if info else 100.0

            # Plot 1: Original rendered image with all candidates
            ax1 = axes[0, 0]
            plotter.image_plot(
                ax1,
                image,
                pixelsize=pixelsize_nm / 1000,
                vmax=np.percentile(image, 99),
                vmin=0,
                title=f"Rendered Image ({len(all_picks)} candidates)",
                colorbar=True,
            )

            # Overlay all candidate positions
            if all_picks:
                picks_array = np.array(all_picks)
                ax1.scatter(
                    picks_array[:, 1],  # x coordinates
                    picks_array[:, 0],  # y coordinates
                    c="red",
                    s=20,
                    alpha=0.7,
                    marker="x",
                    label=f"All candidates ({len(all_picks)})",
                )
                ax1.legend()

            # Plot 2: Histogram analysis
            ax2 = axes[0, 1]
            if hist:
                hist_values, bin_edges = hist
                bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
                ax2.bar(
                    bin_centres,
                    hist_values,
                    width=bin_centres[1] - bin_centres[0],
                    alpha=0.7,
                    color="skyblue",
                    edgecolor="black",
                )
                ax2.axvline(
                    threshold,
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    label=f"Threshold: {threshold:.1f}",
                )
                self.setup_axis(
                    ax2,
                    xlabel="Intensity",
                    ylabel="Frequency",
                    title="Intensity Histogram Analysis",
                    grid=True,
                )
                ax2.legend()

            # Plot 3: Valid fiducials after filtering
            ax3 = axes[1, 0]
            plotter.image_plot(
                ax3,
                image,
                pixelsize=pixelsize_nm / 1000,
                vmax=np.percentile(image, 99),
                vmin=0,
                title=f"Valid Fiducials ({len(valid_picks)})",
                colorbar=True,
            )

            # Overlay valid fiducial positions
            if valid_picks:
                valid_array = np.array(valid_picks)
                ax3.scatter(
                    valid_array[:, 1],  # x coordinates
                    valid_array[:, 0],  # y coordinates
                    c="lime",
                    s=30,
                    alpha=0.8,
                    marker="o",
                    edgecolors="black",
                    linewidth=0.5,
                    label=f"Valid fiducials ({len(valid_picks)})",
                )
                ax3.legend()

            # Plot 4: Final fiducial localisations
            ax4 = axes[1, 1]
            if result.locs_with_groups is not None and len(result.locs_with_groups) > 0:
                try:
                    # Plot fiducials by group
                    unique_groups = np.unique(result.locs_with_groups.group)
                    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_groups)))

                    for i, group_id in enumerate(unique_groups):
                        group_locs = result.locs_with_groups[
                            result.locs_with_groups.group == group_id
                        ]
                        ax4.scatter(
                            group_locs.xc,
                            group_locs.yc,
                            c=[colors[i]],
                            s=8,
                            alpha=0.7,
                            label=f"Fid {group_id} ({len(group_locs)})",
                        )

                    self.setup_axis(
                        ax4,
                        xlabel="X (pixels)",
                        ylabel="Y (pixels)",
                        title=f"Grouped Localisations ({result.n_fiducials} fiducials)",
                        grid=True,
                        equal_aspect=True,
                    )
                    ax4.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

                except Exception as e:
                    ax4.text(
                        0.5,
                        0.5,
                        f"Error plotting groups: {e}",
                        transform=ax4.transAxes,
                        ha="centre",
                        va="centre",
                    )

            # Plot 5: Summary statistics
            ax5 = axes[2, :]  # Span both columns
            ax5.axis("off")

            # Create summary text
            summary_text = "Detection Summary:\n"
            summary_text += f"• Total candidates found: {len(all_picks)}\n"
            summary_text += f"• Valid fiducials: {len(valid_picks)}\n"
            summary_text += f"• Final fiducial groups: {result.n_fiducials}\n"
            if result.metadata:
                summary_text += f"• Total localisations: {result.metadata.get('total_localisations', 'N/A')}\n"
                summary_text += f"• Threshold used: {result.metadata.get('threshold_used', 'N/A'):.1f}\n"

            ax5.text(
                0.05,
                0.9,
                summary_text,
                transform=ax5.transAxes,
                fontsize=12,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.5),
            )

            plt.tight_layout()

            # Use consolidated save/show functionality
            self.save_or_show(fig, save_path=save_path, show=True, dpi=300)

        except Exception as e:
            print(f"⚠️ Failed to create fiducial detection steps plot: {e}")
            import traceback

            traceback.print_exc()

    def plot_fiducial_detection_results(
        self,
        result: "FiducialDetectionResult",
        info: List[dict],
        save_path: Optional[str] = None,
    ) -> None:
        """Create a plot of fiducial detection results using PlottingBase."""
        plotter_class = _ensure_plotter()
        if plotter_class is None:
            print("⚠️ PublicationPlotter not available, skipping plot creation")
            return

        try:
            # Import here to avoid circular imports
            from DriftCorrectionFunctions import CoordinateProcessor

            # Create plotter instance
            plotter = plotter_class(poster=False, dark_background=False)

            # Extract metadata for plotting
            meta = CoordinateProcessor.extract_metadata(info)
            pixelsize = meta.get("pixelsize", 130.0)  # nm

            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

            # Left plot: Detection image with fiducial markers
            # Get fiducial coordinates for scatter overlay (fixed coordinate order)
            fiducial_x = [
                pick[0] for pick in result.picks
            ]  # X coordinates (first element)
            fiducial_y = [
                pick[1] for pick in result.picks
            ]  # Y coordinates (second element)

            plotter.image_scatter_plot(
                ax1,
                data=result.detection_image,
                xdata=np.array(fiducial_x),
                ydata=np.array(fiducial_y),
                cmap="hot",
                cbar="on",
                cbarlabel="Intensity",
                label=f"Detected Fiducials ({result.n_fiducials})",
                labelcolor="cyan",
                pixelsize=pixelsize,
                scalebarsize=1000,  # 1μm scale bar
                scalebarlabel="1 μm",
                scattercolor="cyan",
                s=100,  # Larger marker size
                scatteralpha=0.8,
            )
            ax1.set_title("Fiducial Detection Results")

            # Right plot: Fiducial localisations colored by group
            fiducial_locs = result.locs_with_groups[result.locs_with_groups.group >= 0]

            if len(fiducial_locs) > 0:
                unique_groups = np.unique(fiducial_locs.group)
                try:
                    # Use basic colors - colormaps are causing issues with Pylance
                    color_list = [
                        "red",
                        "blue",
                        "green",
                        "orange",
                        "purple",
                        "brown",
                        "pink",
                        "gray",
                        "olive",
                        "cyan",
                    ]
                    colors = (color_list * (len(unique_groups) // len(color_list) + 1))[
                        : len(unique_groups)
                    ]
                except:
                    # Ultimate fallback
                    colors = ["red"] * len(unique_groups)

                for i, group_id in enumerate(unique_groups):
                    group_locs = fiducial_locs[fiducial_locs.group == group_id]
                    ax2.scatter(
                        group_locs.xc * 1000,  # Convert to nm for plotting
                        group_locs.yc * 1000,
                        s=2,
                        alpha=0.6,
                        c=[colors[i]],
                        label=f"Fiducial {group_id+1} ({len(group_locs)} locs)",
                        rasterized=True,
                    )

            self.setup_axis(
                ax2,
                xlabel="X (nm)",
                ylabel="Y (nm)",
                title="Fiducial Localisations by Group",
                grid=True,
                equal_aspect=True,
            )
            ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

            # Add summary text
            summary_text = (
                f"Detection Summary:\n"
                f"• Threshold: {result.detection_params['threshold_percentile']:.1f}%\n"
                f"• Box size: {result.detection_params['box_size_nm']:.0f} nm\n"
                f"• Min frames: {result.detection_params['min_frames_fraction']:.1%}\n"
                f"• Candidates found: {result.metadata['total_candidates']}\n"
                f"• Valid fiducials: {result.n_fiducials}"
            )

            fig.text(
                0.02,
                0.98,
                summary_text,
                transform=fig.transFigure,
                verticalalignment="top",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
            )

            plt.tight_layout()
            plt.subplots_adjust(left=0.15)  # Make room for summary text

            # Use consolidated save/show functionality
            self.save_or_show(fig, save_path=save_path, show=True, dpi=150)

        except Exception as e:
            print(f"⚠️ Failed to create fiducial detection plot: {e}")

    def plot_region_data_with_datashader(self, ax, data_list, color_list, title):
        """Plot region data using optimised rendering based on dataset size.

        Uses inherited DatashaderMixin functionality for large datasets,
        with automatic fallback to matplotlib for smaller datasets.
        """
        total_points = sum(len(data) for data in data_list)

        # Use consolidated plotting approach with new multi-dataset method
        if len(data_list) == 1:
            # Single dataset - use the optimised large scatter method
            data = data_list[0]
            self.plot_large_scatter(
                ax,
                data["xc"],
                data["yc"],
                threshold=10000,  # Higher threshold for single dataset
                cmap=color_list[0] if color_list else "blue",
                s=2,
                alpha=0.7,
                rasterized=True,
            )
        else:
            # Multiple datasets - use new multi-dataset plotting
            datasets = [{"x": data["xc"], "y": data["yc"]} for data in data_list]
            labels = [f"Group {i+1}" for i in range(len(data_list))]

            self.plot_multi_dataset_scatter(
                ax,
                datasets,
                labels=labels,
                colors=color_list if color_list else None,
                threshold=10000,  # Threshold for total points
                alpha=0.6,
                sizes=2.0,
                rasterized=True,
            )

        # Use standardised axis setup
        self.setup_axis(
            ax,
            xlabel="X (pixels)",
            ylabel="Y (pixels)",
            title=title,
            grid=True,
            equal_aspect=True,
        )

    def plot_clustering_overlay(self, ax, all_x, all_y, types, title):
        """Plot clustering overlay showing original vs validated points.

        Uses consolidated plotting base with automatic optimisation for large datasets.
        """
        # Separate data by type
        original_mask = np.array(types) == "original"
        validated_mask = np.array(types) == "validated"

        # Prepare datasets for multi-dataset plotting
        datasets = []
        labels = []
        colors = []
        sizes = []

        if np.any(original_mask):
            datasets.append({"x": all_x[original_mask], "y": all_y[original_mask]})
            labels.append("Original")
            colors.append("lightgray")
            sizes.append(1.0)

        if np.any(validated_mask):
            datasets.append({"x": all_x[validated_mask], "y": all_y[validated_mask]})
            labels.append("Validated")
            colors.append("red")
            sizes.append(3.0)

        # Use new multi-dataset plotting with appropriate threshold
        total_points = len(all_x)
        if total_points > 10000:
            # Use optimized plotting for large datasets
            scatters = self.plot_multi_dataset_scatter(
                ax,
                datasets,
                labels=labels,
                colors=colors,
                sizes=sizes,
                threshold=10000,
                alpha=0.6,
                rasterized=True,
            )
        else:
            # Standard matplotlib for smaller datasets
            for i, (dataset, label, color, size) in enumerate(
                zip(datasets, labels, colors, sizes)
            ):
                alpha = 0.3 if label == "Original" else 0.9
                ax.scatter(
                    dataset["x"],
                    dataset["y"],
                    c=color,
                    s=size,
                    alpha=alpha,
                    label=label,
                    rasterized=True,
                )
            ax.legend()

        # Use standardised axis setup
        self.setup_axis(
            ax,
            xlabel="X (pixels)",
            ylabel="Y (pixels)",
            title=title,
            grid=True,
            equal_aspect=True,
        )

    def plot_puncta_selection_results(
        self,
        all_locs: np.recarray,
        selected_puncta: List[np.recarray],
        region_centres: List[Tuple[int, int]],
        binary_mask: np.ndarray,
        region_stats: List[Dict[str, Any]],
        box_size_pixels: float,
        pixelsize: float,
        output_figure_path: Optional[str],
        title: str,
        plot_individual_regions: bool = True,
        use_datashader_threshold: int = 10000,
    ) -> None:
        """Create visualization of puncta selection results with optimized rendering."""
        try:
            import matplotlib.pyplot as plt

            # Create comprehensive figure
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            axes = axes.flatten()

            # Plot 1: All localisations with intelligent downsampling
            ax = axes[0]
            n_locs = len(all_locs)

            if n_locs > use_datashader_threshold:
                # Use preview plot for very large datasets
                self.create_preview_plot(
                    ax,
                    all_locs.xc,
                    all_locs.yc,
                    preview_points=use_datashader_threshold,
                    method="density",  # Density-aware sampling
                    s=1,
                    alpha=0.5,
                    c="blue",
                    rasterized=True,
                )
            else:
                ax.scatter(
                    all_locs.xc,
                    all_locs.yc,
                    s=1,
                    alpha=0.5,
                    c="blue",
                    label="All localisations",
                    rasterized=True,
                )

            # Plot region centres
            if region_centres:
                centres = np.array(region_centres)
                ax.scatter(
                    centres[:, 0],
                    centres[:, 1],
                    s=100,
                    c="red",
                    marker="x",
                    linewidth=2,
                    label=f"Region centres ({len(region_centres)})",
                )

            self.setup_axis(
                ax,
                xlabel="X (pixels)",
                ylabel="Y (pixels)",
                title=f"All Localisations ({n_locs:,} points)",
                grid=True,
                equal_aspect=True,
            )
            ax.legend()

            # Plot 2: Selected puncta by region
            ax = axes[1]
            if selected_puncta:
                datasets = []
                labels = []
                for i, puncta in enumerate(selected_puncta):
                    if len(puncta) > 0:
                        datasets.append({"x": puncta.xc, "y": puncta.yc})
                        labels.append(f"Region {i} ({len(puncta)})")

                if datasets:
                    self.plot_multi_dataset_scatter(
                        ax,
                        datasets,
                        labels=labels if len(datasets) <= 10 else None,
                        threshold=use_datashader_threshold,
                        alpha=0.6,
                        sizes=2.0,
                        rasterized=True,
                    )

            self.setup_axis(
                ax,
                xlabel="X (pixels)",
                ylabel="Y (pixels)",
                title=f"Selected Puncta ({len(selected_puncta)} regions)",
                grid=True,
                equal_aspect=True,
            )

            # Plot 3: Binary mask
            ax = axes[2]
            ax.imshow(binary_mask, cmap="gray", origin="lower")
            if region_centres:
                centres = np.array(region_centres)
                ax.scatter(
                    centres[:, 0], centres[:, 1], s=50, c="red", marker="o", alpha=0.7
                )
            self.setup_axis(
                ax,
                xlabel="X (pixels)",
                ylabel="Y (pixels)",
                title="Density Mask",
                grid=False,
            )

            # Plot 4: Statistics summary
            ax = axes[3]
            if region_stats:
                stats_text = [
                    f"PUNCTA SELECTION SUMMARY",
                    f"",
                    f"Total localisations: {n_locs:,}",
                    f"Detected regions: {len(region_centres)}",
                    f"Selected regions: {len(selected_puncta)}",
                    f"Box size: {box_size_pixels:.1f} px ({box_size_pixels * pixelsize:.1f} nm)",
                    f"Pixel size: {pixelsize:.2f} nm",
                ]

                # Add per-region stats if available
                if region_stats and len(region_stats) > 0:
                    stats_text.extend(
                        [
                            f"",
                            f"Average locs per region: {np.mean([len(p) for p in selected_puncta]):.1f}",
                            f"Min/Max locs: {min(len(p) for p in selected_puncta)}/{max(len(p) for p in selected_puncta)}",
                        ]
                    )

                ax.text(
                    0.05,
                    0.95,
                    "\n".join(stats_text),
                    transform=ax.transAxes,
                    fontsize=11,
                    verticalalignment="top",
                    fontfamily="monospace",
                    bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
                )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")

            plt.suptitle(title, fontsize=14, fontweight="bold")
            plt.tight_layout()

            # Use consolidated save/show functionality
            self.save_or_show(fig, save_path=output_figure_path, show=True, dpi=150)

        except Exception as e:
            print(f"⚠️ Failed to create puncta selection plot: {e}")
            import traceback

            traceback.print_exc()

    def plot_individual_clustering_details(
        self,
        selected_puncta: List[np.recarray],
        validated_fiducials: List[np.recarray],
        clustering_metadata: List[Dict[str, Any]],
        base_path: str,
        title: str,
    ) -> None:
        """Create detailed plots for individual clustering results using PlottingBase."""
        try:
            import matplotlib.pyplot as plt

            n_validated = len(validated_fiducials)
            if n_validated == 0:
                print("⚠️ No validated fiducials to plot")
                return

            # Create subplots layout
            cols = min(3, n_validated)
            rows = (n_validated + cols - 1) // cols

            fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
            if n_validated == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes
            else:
                axes = axes.flatten()

            for i, (selected, validated, metadata) in enumerate(
                zip(
                    selected_puncta[:n_validated],
                    validated_fiducials[:n_validated],
                    clustering_metadata[:n_validated],
                )
            ):
                ax = axes[i]

                # Plot selected puncta
                if len(selected) > 0:
                    ax.scatter(
                        selected.xc,
                        selected.yc,
                        c="lightblue",
                        s=20,
                        alpha=0.6,
                        label="Selected",
                    )

                # Plot validated fiducials
                if len(validated) > 0:
                    ax.scatter(
                        validated.xc,
                        validated.yc,
                        c="red",
                        s=30,
                        alpha=0.8,
                        label="Validated",
                    )

                ax.set_title(f"Region {i}: {len(validated)} validated")
                ax.set_xlabel("X (nm)")
                ax.set_ylabel("Y (nm)")
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_aspect("equal", adjustable="box")

            # Hide unused subplots
            for i in range(n_validated, len(axes)):
                axes[i].set_visible(False)

            plt.suptitle(f"{title} - Individual Clustering Details", fontsize=14)
            plt.tight_layout()

            # Save plot
            filename = f"{base_path}_details.png"
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"✅ Individual clustering details saved to: {filename}")
            plt.show()

        except Exception as e:
            print(f"⚠️ Error creating individual clustering details: {e}")

    def plot_clustering_results(
        self,
        selected_puncta: List[np.recarray],
        validated_fiducials: List[np.recarray],
        clustering_metadata: List[Dict[str, Any]],
        output_figure_path: Optional[str],
        title: str,
    ) -> None:
        """Create visualization of DBSCAN clustering results using PlottingBase."""
        try:
            import matplotlib.pyplot as plt

            n_regions = len(selected_puncta)
            n_validated = len(validated_fiducials)

            if n_regions == 0:
                print("⚠️ No puncta regions to plot")
                return

            # Create summary figure
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()

            # Plot 1: Original puncta overview
            ax = axes[0]
            for i, puncta in enumerate(selected_puncta):
                if len(puncta) > 0:
                    color = plt.cm.tab10(i % 10)
                    ax.scatter(
                        puncta.xc,
                        puncta.yc,
                        c=[color],
                        s=20,
                        alpha=0.6,
                        label=f"Region {i}",
                    )
            ax.set_title(f"Original Puncta ({n_regions} regions)")
            ax.set_xlabel("X (nm)")
            ax.set_ylabel("Y (nm)")
            ax.grid(True, alpha=0.3)
            if n_regions <= 10:
                ax.legend()

            # Plot 2: Validated fiducials
            ax = axes[1]
            for i, validated in enumerate(validated_fiducials):
                if len(validated) > 0:
                    color = plt.cm.tab10(i % 10)
                    ax.scatter(
                        validated.xc,
                        validated.yc,
                        c=[color],
                        s=30,
                        alpha=0.8,
                        label=f"Validated {i}",
                    )
            ax.set_title(f"Validated Fiducials ({n_validated} regions)")
            ax.set_xlabel("X (nm)")
            ax.set_ylabel("Y (nm)")
            ax.grid(True, alpha=0.3)
            if n_validated <= 10:
                ax.legend()

            # Plot 3: Retention statistics
            ax = axes[2]
            retention_rates = []
            region_sizes = []
            for i, (selected, validated) in enumerate(
                zip(selected_puncta, validated_fiducials)
            ):
                if len(selected) > 0:
                    retention = len(validated) / len(selected) * 100
                    retention_rates.append(retention)
                    region_sizes.append(len(selected))
                else:
                    retention_rates.append(0)
                    region_sizes.append(0)

            if retention_rates:
                bars = ax.bar(range(len(retention_rates)), retention_rates, alpha=0.7)
                ax.set_xlabel("Region ID")
                ax.set_ylabel("Retention Rate (%)")
                ax.set_title("Clustering Retention Rates")
                ax.grid(True, alpha=0.3)

                # Add value labels on bars
                for i, (bar, rate) in enumerate(zip(bars, retention_rates)):
                    if rate > 0:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 1,
                            f"{rate:.1f}%",
                            ha="centre",
                            va="bottom",
                            fontsize=8,
                        )

            # Plot 4: Summary statistics
            ax = axes[3]
            total_selected = sum(len(p) for p in selected_puncta)
            total_validated = sum(len(v) for v in validated_fiducials)
            overall_retention = (
                (total_validated / total_selected * 100) if total_selected > 0 else 0
            )

            stats_text = [
                f"Regions analyzed: {n_regions}",
                f"Total puncta: {total_selected}",
                f"Validated fiducials: {total_validated}",
                f"Overall retention: {overall_retention:.1f}%",
                (
                    f"Avg per region: {total_validated/n_regions:.1f}"
                    if n_regions > 0
                    else "Avg per region: 0"
                ),
            ]

            ax.text(
                0.1,
                0.9,
                "\n".join(stats_text),
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
            ax.set_title("Summary Statistics")

            plt.suptitle(f"{title} - Clustering Results", fontsize=14)
            plt.tight_layout()

            if output_figure_path:
                base_path = (
                    output_figure_path.rsplit(".", 1)[0]
                    if "." in output_figure_path
                    else output_figure_path
                )
                filename = f"{base_path}_clustering_results.png"
                plt.savefig(filename, dpi=300, bbox_inches="tight")
                print(f"✅ Clustering results saved to: {filename}")
            else:
                plt.show()

            plt.close()

        except Exception as e:
            print(f"⚠️ Error creating clustering results plot: {e}")

    def plot_clustering_summary_only(
        self,
        selected_puncta: List[np.recarray],
        validated_fiducials: List[np.recarray],
        clustering_metadata: List[Dict[str, Any]],
        output_figure_path: Optional[str],
        title: str,
    ) -> None:
        """Create summary visualization only (individual clusters already plotted per iteration)."""
        try:
            import matplotlib.pyplot as plt

            n_regions = len(selected_puncta)
            n_validated = len(validated_fiducials)

            if n_regions == 0:
                print("⚠️ No data to plot in summary")
                return

            # Create summary figure with 2x2 layout
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()

            # Plot 1: Validation statistics distribution
            ax = axes[0]
            n_locs = [len(selected) for selected in selected_puncta]
            n_validated_per_region = [
                len(validated) for validated in validated_fiducials
            ]
            retention_rates = [
                (len(v) / len(s) * 100) if len(s) > 0 else 0
                for s, v in zip(selected_puncta, validated_fiducials)
            ]

            x_pos = range(len(retention_rates))
            bars = ax.bar(x_pos, retention_rates, alpha=0.7, color="skyblue")
            ax.set_xlabel("Region ID")
            ax.set_ylabel("Retention Rate (%)")
            ax.set_title("Validation Retention by Region")
            ax.grid(True, alpha=0.3)

            # Add mean line
            if retention_rates:
                mean_retention = np.mean(retention_rates)
                ax.axhline(
                    mean_retention,
                    color="red",
                    linestyle="--",
                    label=f"Mean: {mean_retention:.1f}%",
                )
                ax.legend()

            # Plot 2: Quality distribution histogram
            ax = axes[1]
            if retention_rates:
                ax.hist(
                    retention_rates,
                    bins=10,
                    alpha=0.7,
                    color="lightgreen",
                    edgecolor="black",
                )
                ax.set_xlabel("Retention Rate (%)")
                ax.set_ylabel("Number of Regions")
                ax.set_title("Quality Distribution")
                ax.grid(True, alpha=0.3)

            # Plot 3: Size vs retention scatter
            ax = axes[2]
            if n_locs and retention_rates:
                colors = [
                    "red" if r < 50 else "orange" if r < 75 else "green"
                    for r in retention_rates
                ]
                scatter = ax.scatter(n_locs, retention_rates, c=colors, alpha=0.7, s=50)
                ax.set_xlabel("Initial Puncta Count")
                ax.set_ylabel("Retention Rate (%)")
                ax.set_title("Size vs Quality")
                ax.grid(True, alpha=0.3)

                # Add trend line if enough data
                if len(n_locs) > 3:
                    z = np.polyfit(n_locs, retention_rates, 1)
                    p = np.poly1d(z)
                    ax.plot(
                        sorted(n_locs), p(sorted(n_locs)), "r--", alpha=0.8, linewidth=1
                    )

            # Plot 4: Summary statistics text
            ax = axes[3]
            total_input = sum(n_locs)
            total_output = sum(n_validated_per_region)
            overall_retention = (
                (total_output / total_input * 100) if total_input > 0 else 0
            )

            stats_text = [
                f"CLUSTERING SUMMARY",
                f"",
                f"Regions processed: {n_regions}",
                f"Total input puncta: {total_input}",
                f"Total validated fiducials: {total_output}",
                f"Overall retention rate: {overall_retention:.1f}%",
                f"",
                (
                    f"Mean retention: {np.mean(retention_rates):.1f}% ± {np.std(retention_rates):.1f}%"
                    if retention_rates
                    else "Mean retention: 0%"
                ),
                (
                    f"Best region: {max(retention_rates):.1f}%"
                    if retention_rates
                    else "Best region: 0%"
                ),
                (
                    f"Worst region: {min(retention_rates):.1f}%"
                    if retention_rates
                    else "Worst region: 0%"
                ),
                f"Regions >50% retention: {sum(1 for r in retention_rates if r > 50)}/{n_regions}",
            ]

            ax.text(
                0.05,
                0.95,
                "\n".join(stats_text),
                transform=ax.transAxes,
                fontsize=11,
                verticalalignment="top",
                fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")

            plt.suptitle(
                f"{title} - Clustering Summary", fontsize=14, fontweight="bold"
            )
            plt.tight_layout()

            if output_figure_path:
                base_path = (
                    output_figure_path.rsplit(".", 1)[0]
                    if "." in output_figure_path
                    else output_figure_path
                )
                filename = f"{base_path}_clustering_summary.png"
                plt.savefig(filename, dpi=300, bbox_inches="tight")
                print(f"✅ Clustering summary saved to: {filename}")
            else:
                plt.show()

            plt.close()

        except Exception as e:
            print(f"⚠️ Error creating clustering summary plot: {e}")

    def create_separate_plots(
        self,
        smoothed_image: np.ndarray,
        binary_mask: np.ndarray,
        region_centres: List[Tuple[int, int]],
        hist: np.ndarray,
        bin_edges: np.ndarray,
        threshold: float,
        pixelsize: float,
        output_figure_path: Optional[str],
        title: str,
    ) -> None:
        """Create separate detailed plots for density detection analysis."""
        plotter_class = _ensure_plotter()
        if plotter_class is None:
            print("⚠️ PublicationPlotter not available, skipping separate plots")
            return

        try:
            plotter = plotter_class(poster=False)

            # Create a basic visualization using PlottingBase
            fig, axes = plotter.two_by_two_plot()

            # Plot 1: Smoothed image
            im1 = axes[0].imshow(smoothed_image, cmap="hot", origin="lower")
            axes[0].set_title("Smoothed Image")
            axes[0].set_xlabel("X (pixels)")
            axes[0].set_ylabel("Y (pixels)")

            # Plot 2: Binary mask
            axes[1].imshow(binary_mask, cmap="gray", origin="lower")
            axes[1].set_title("Binary Mask")
            axes[1].set_xlabel("X (pixels)")
            axes[1].set_ylabel("Y (pixels)")

            # Plot 3: Histogram
            if len(hist) > 0 and len(bin_edges) > 0:
                bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
                axes[2].plot(bin_centres, hist)
                axes[2].axvline(
                    threshold,
                    color="red",
                    linestyle="--",
                    label=f"Threshold: {threshold:.2f}",
                )
                axes[2].set_xlabel("Intensity")
                axes[2].set_ylabel("Count")
                axes[2].set_title("Intensity Histogram")
                axes[2].legend()

            # Plot 4: Detected regions overlay
            axes[3].imshow(smoothed_image, cmap="hot", origin="lower", alpha=0.7)
            for i, (cy, cx) in enumerate(region_centres):
                axes[3].plot(
                    cx, cy, "wo", markersize=8, markeredgecolor="red", markeredgewidth=2
                )
                axes[3].text(
                    cx, cy, str(i), color="white", ha="centre", va="centre", fontsize=8
                )
            axes[3].set_title(f"{title} - {len(region_centres)} Regions")
            axes[3].set_xlabel("X (pixels)")
            axes[3].set_ylabel("Y (pixels)")

            plt.tight_layout()

            if output_figure_path:
                base_path = (
                    output_figure_path.rsplit(".", 1)[0]
                    if "." in output_figure_path
                    else output_figure_path
                )
                filename = f"{base_path}_density_detection.png"
                plt.savefig(filename, dpi=300, bbox_inches="tight")
                print(f"✅ Density detection plots saved to: {filename}")
            else:
                plt.show()

        except Exception as e:
            print(f"⚠️ Error creating separate plots: {e}")
            # Fallback to basic matplotlib if PlottingBase fails
            try:
                import matplotlib.pyplot as plt

                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                axes = axes.flatten()

                # Simple fallback visualization
                axes[0].imshow(smoothed_image, cmap="hot")
                axes[0].set_title("Smoothed Image")

                axes[1].imshow(binary_mask, cmap="gray")
                axes[1].set_title("Binary Mask")

                if len(hist) > 0:
                    axes[2].plot(hist)
                    axes[2].set_title("Histogram")

                axes[3].imshow(smoothed_image, cmap="hot", alpha=0.7)
                for cx, cy in region_centres:
                    axes[3].plot(cy, cx, "wo", markersize=6)
                axes[3].set_title(f"{len(region_centres)} Regions Detected")

                plt.tight_layout()
                if output_figure_path:
                    plt.savefig(output_figure_path, dpi=300, bbox_inches="tight")
                plt.show()
            except Exception as fallback_error:
                print(f"⚠️ Fallback plotting also failed: {fallback_error}")
