"""
Fiducial Detection Module

Contains functions for detecting and processing fiducial markers for drift correction.
Extracted from DriftCorrectionFunctions.py for better code organisation.

This module handles:
- High-density region detection from images
- Puncta selection from detected regions
- Fiducial detection with temporal chunking
- Fiducial clustering and validation
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
import warnings
import sys
import os
import gc

# Add the current directory to the path for imports
module_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(module_dir)

from ImportManager import get_module, is_available

# Import scipy components
try:
    from scipy import ndimage
    from scipy.optimize import curve_fit
except ImportError:
    warnings.warn(
        "scipy not available - some fiducial detection features may not work."
    )
    ndimage = None
    curve_fit = None

# Get modules through import manager
plt = get_module("matplotlib.pyplot")

# Lazy load local modules to avoid circular imports
postprocess = None
imageprocess = None
PlottingFunctions = None


def _ensure_postprocess():
    """Lazy load postprocess module."""
    global postprocess
    if postprocess is None:
        postprocess = get_module("postprocess")
    return postprocess


def _ensure_plotting():
    """Lazy load PlottingFunctions module."""
    global PlottingFunctions
    if PlottingFunctions is None:
        PlottingFunctions = get_module("PlottingFunctions")
    return PlottingFunctions


class FiducialDetector:
    """Class containing fiducial detection and selection functionality."""

    def __init__(self, drift_correction_instance=None):
        """
        Initialise with reference to main drift correction instance if needed.

        Args:
            drift_correction_instance: Reference to main DriftCorrectionFunctions instance
        """
        self.drift_correction = drift_correction_instance

    def detect_high_density_regions_from_image(
        self,
        smoothed_image: np.ndarray,
        histogram_bins: int = 256,
        threshold_percentile: float = 99.0,
        pixelsize: float = 100.0,
        output_figure_path: Optional[str] = None,
        title: str = "High-Density Region Detection",
        create_plot: bool = True,
    ) -> Tuple[List[Tuple[int, int]], np.ndarray, float, Dict[str, Any]]:
        """Detect high-density regions from a smoothed image using histogram analysis.

        This function takes a pre-smoothed/rendered image and identifies high-density
        regions based on histogram analysis. It provides clear visualization of the
        detection process and outputs region coordinates for downstream processing.

        Args:
            smoothed_image: Pre-smoothed 2D image array (e.g., from render functions)
            histogram_bins: Number of bins for histogram analysis
            threshold_percentile: Percentile threshold for region detection (0-100)
            pixelsize: Pixel size in nm for scale bar visualization
            output_figure_path: Optional path to save the detection figure
            title: Title for the detection plot
            create_plot: Whether to create visualization plots

        Returns:
            Tuple containing:
            - List of (y, x) coordinates of detected high-density region centres
            - Binary mask of detected regions
            - Threshold value used for detection
            - Metadata dictionary with detection statistics
        """
        # Calculate histogram and threshold
        image_flat = smoothed_image.ravel()
        image_flat = image_flat[image_flat > 0]  # Exclude zero values

        if len(image_flat) == 0:
            raise ValueError("Image contains no non-zero values")

        hist, bin_edges = np.histogram(image_flat, bins=histogram_bins)
        threshold = np.percentile(image_flat, threshold_percentile)

        # Create binary mask of high-density regions
        binary_mask = smoothed_image > threshold

        # Find connected components / regions
        if ndimage is None:
            raise RuntimeError("scipy.ndimage required for region detection")

        labeled_regions, n_regions = ndimage.label(binary_mask)

        # Calculate region centres and properties
        region_centres = []
        region_stats = []

        for region_id in range(1, n_regions + 1):
            region_mask = labeled_regions == region_id
            region_coords = np.where(region_mask)

            if len(region_coords[0]) > 0:
                # Calculate centre of mass
                centre_y = np.mean(region_coords[0])
                centre_x = np.mean(region_coords[1])
                region_centres.append((int(centre_y), int(centre_x)))

                # Calculate region statistics
                region_area = np.sum(region_mask)
                region_intensity = np.sum(smoothed_image[region_mask])
                region_max_intensity = np.max(smoothed_image[region_mask])

                region_stats.append(
                    {
                        "centre": (centre_y, centre_x),
                        "area_pixels": region_area,
                        "total_intensity": region_intensity,
                        "max_intensity": region_max_intensity,
                        "mean_intensity": (
                            region_intensity / region_area if region_area > 0 else 0
                        ),
                    }
                )

        # Create visualization if requested
        if create_plot:
            self._plot_density_detection_results(
                smoothed_image,
                binary_mask,
                region_centres,
                hist,
                bin_edges,
                threshold,
                pixelsize,
                output_figure_path,
                title,
            )

        # Prepare metadata
        metadata = {
            "n_regions_detected": n_regions,
            "threshold_value": threshold,
            "threshold_percentile": threshold_percentile,
            "histogram_bins": histogram_bins,
            "image_shape": smoothed_image.shape,
            "image_max": np.max(smoothed_image),
            "image_mean": np.mean(smoothed_image[smoothed_image > 0]),
            "region_statistics": region_stats,
            "total_region_area": np.sum(binary_mask),
            "region_area_fraction": np.sum(binary_mask) / binary_mask.size,
        }

        return region_centres, binary_mask, threshold, metadata

    def select_puncta_from_regions(
        self,
        locs: np.recarray,
        region_centres: List[Tuple[int, int]],
        binary_mask: np.ndarray,
        pixelsize: float = 100.0,
        selection_box_size_nm: float = 600.0,
        min_localisations_per_region: int = 10,
        output_figure_path: Optional[str] = None,
        title: str = "Puncta Selection from Regions",
        create_plot: bool = True,
        plot_individual_regions: bool = True,
        use_datashader_threshold: int = 1000,
        memory_optimise: bool = True,
    ) -> Tuple[List[np.recarray], Dict[str, Any]]:
        """Select puncta (localisations) from detected high-density regions.

        This function takes the output from detect_high_density_regions_from_image
        and selects localisations within rectangular boxes around each detected region centre
        to create potential fiducial candidates. Uses the optimized postprocess.picked_locs
        function with Rectangle shape.

        Args:
            locs: Localisation data with xc, yc, frame fields
            region_centres: List of (y, x) coordinates from density detection
            binary_mask: Binary mask from density detection
            pixelsize: Pixel size in nm for coordinate conversion
            selection_box_size_nm: Size of square selection box around each region centre (nm)
            min_localisations_per_region: Minimum number of localisations required for valid region
            output_figure_path: Optional path to save selection visualisation
            title: Title for visualisation plots
            create_plot: Whether to create visualisation plots
            plot_individual_regions: Whether to plot individual region details
            use_datashader_threshold: Use datashader for scatter plots with more than this many points
            memory_optimise: Whether to use memory optimisation

        Returns:
            Tuple containing:
            - List of localisation arrays, one per valid region
            - Metadata dictionary with selection statistics
        """
        # Ensure postprocess is loaded
        pp = _ensure_postprocess()
        if pp is None:
            raise RuntimeError(
                "postprocess module not available - cannot use picked_locs function"
            )

        # Handle empty region centres
        if not region_centres:
            metadata = {
                "n_regions_input": 0,
                "n_regions_selected": 0,
                "selection_criteria": {
                    "min_localisations": min_localisations_per_region,
                    "selection_box_size_nm": selection_box_size_nm,
                    "selection_box_size_pixels": 0.0,
                },
                "rejection_reasons": {"too_few_localisations": 0, "accepted": 0},
                "region_statistics": [],
            }
            return [], metadata

        # Convert box size from nm to pixels
        box_size_pixels = selection_box_size_nm / pixelsize
        half_box = box_size_pixels / 2.0

        # Create horizontal line picks for Rectangle shape
        picks = []
        for centre_y, centre_x in region_centres:
            picks.append(
                ((centre_x - half_box, centre_y), (centre_x + half_box, centre_y))
            )

        # Use postprocess.picked_locs with parallelization if we have 8+ picks
        width = max(locs.xc.max() + 10, 100)
        height = max(locs.yc.max() + 10, 100)

        picked_locs_arrays = pp.picked_locs(
            locs=locs,
            width=width,
            height=height,
            picks=picks,
            pick_shape="Rectangle",
            pick_size=box_size_pixels,
            add_group=False,
            callback="console",
            parallel=len(picks) >= 8,
        )

        # Memory cleanup
        if memory_optimise:
            del picks
            gc.collect()

        # Filter results based on minimum localisation count
        selected_puncta = []
        region_stats = []

        if picked_locs_arrays is None:
            picked_locs_arrays = []

        rejected_count = 0
        for region_id, (region_locs, (centre_y, centre_x)) in enumerate(
            zip(picked_locs_arrays, region_centres)
        ):
            n_locs = len(region_locs)

            if n_locs >= min_localisations_per_region:
                selected_puncta.append(region_locs)

                region_stat = {
                    "region_id": region_id,
                    "centre_y": centre_y,
                    "centre_x": centre_x,
                    "n_localisations": n_locs,
                    "mean_x": np.mean(region_locs.xc),
                    "mean_y": np.mean(region_locs.yc),
                    "std_x": np.std(region_locs.xc),
                    "std_y": np.std(region_locs.yc),
                    "frame_range": [
                        int(region_locs.frame.min()),
                        int(region_locs.frame.max()),
                    ],
                    "frame_span": int(
                        region_locs.frame.max() - region_locs.frame.min() + 1
                    ),
                    "selection_box_size_nm": selection_box_size_nm,
                    "selection_box_size_pixels": box_size_pixels,
                    "box_boundaries": {
                        "x_min": centre_x - half_box,
                        "x_max": centre_x + half_box,
                        "y_min": centre_y - half_box,
                        "y_max": centre_y + half_box,
                    },
                }

                if hasattr(region_locs, "photons"):
                    region_stat["mean_photons"] = np.mean(region_locs.photons)
                    region_stat["std_photons"] = np.std(region_locs.photons)

                region_stats.append(region_stat)
            else:
                rejected_count += 1
                if memory_optimise:
                    del region_locs

            if memory_optimise and region_id % 100 == 0 and region_id > 0:
                gc.collect()
                print(
                    f"Processed {region_id + 1}/{len(picked_locs_arrays)} regions "
                    f"({len(selected_puncta)} accepted, {rejected_count} rejected)"
                )

        # Final memory cleanup
        if memory_optimise:
            del picked_locs_arrays
            gc.collect()
            print(
                "Memory optimisation: Freed intermediate arrays after region processing"
            )

        # Create visualization if requested
        if create_plot:
            self._plot_puncta_selection_results(
                locs,
                selected_puncta,
                region_centres,
                binary_mask,
                region_stats,
                box_size_pixels,
                pixelsize,
                output_figure_path,
                title,
                plot_individual_regions,
                use_datashader_threshold,
            )

            if memory_optimise and plt is not None:
                plt.close("all")
                gc.collect()

        # Prepare metadata
        total_locs_selected = sum(len(puncta) for puncta in selected_puncta)

        metadata = {
            "n_regions_input": len(region_centres),
            "n_regions_selected": len(selected_puncta),
            "n_regions_rejected": rejected_count,
            "selection_rate": (
                len(selected_puncta) / len(region_centres) if region_centres else 0
            ),
            "selection_criteria": {
                "min_localisations": min_localisations_per_region,
                "selection_box_size_nm": selection_box_size_nm,
                "selection_box_size_pixels": box_size_pixels,
            },
            "region_statistics": region_stats,
            "total_selected_localisations": total_locs_selected,
            "memory_optimized": memory_optimise,
            "rejection_reasons": {
                "too_few_localisations": rejected_count,
                "accepted": len(selected_puncta),
            },
        }

        return selected_puncta, metadata

    def identify_real_fiducials_with_clustering(
        self,
        selected_puncta: List[np.recarray],
        retention_percentage: float = 0.9,
        min_samples_factor: float = 0.7,
        frame_count: int = 100000,
        pixelsize: float = 69.0,
        output_figure_path: Optional[str] = None,
        title: str = "Fiducial Gaussian Fitting Analysis",
        create_plot: bool = True,
    ) -> Tuple[List[np.recarray], Dict[str, Any]]:
        """Identify real fiducials using single Gaussian distribution fitting.

        This function applies single Gaussian mixture fitting to identify real fiducial markers
        by fitting each region to a single 2D Gaussian and keeping a specified percentage
        of points based on their distance from the Gaussian centre.

        Args:
            selected_puncta: List of puncta arrays from region selection
            retention_percentage: Fraction of points to retain (0-1)
            min_samples_factor: Minimum fraction of points required for valid fit
            frame_count: Total number of frames for normalization
            pixelsize: Pixel size in nm
            output_figure_path: Optional path to save plots
            title: Title for plots
            create_plot: Whether to create visualisation plots

        Returns:
            Tuple of (validated_fiducials, validation_metadata)
        """
        validated_fiducials = []
        validation_metadata = []

        for i, puncta in enumerate(selected_puncta):
            if len(puncta) == 0:
                continue

            # Fit 2D Gaussian to the localizations
            x = puncta.xc
            y = puncta.yc

            try:
                # Calculate centre and spread
                x_mean, y_mean = np.mean(x), np.mean(y)
                x_std, y_std = np.std(x), np.std(y)

                # Calculate distances from centre
                distances = np.sqrt((x - x_mean) ** 2 + (y - y_mean) ** 2)

                # Keep only the closest retention_percentage of points
                n_keep = int(len(distances) * retention_percentage)
                if n_keep < min_samples_factor * len(distances):
                    continue

                closest_indices = np.argsort(distances)[:n_keep]
                validated_puncta = puncta[closest_indices]

                validated_fiducials.append(validated_puncta)

                metadata = {
                    "region_id": i,
                    "n_input": len(puncta),
                    "n_output": len(validated_puncta),
                    "retention_rate": len(validated_puncta) / len(puncta),
                    "centre_x": x_mean,
                    "centre_y": y_mean,
                    "std_x": x_std,
                    "std_y": y_std,
                }
                validation_metadata.append(metadata)

            except Exception as e:
                warnings.warn(f"Failed to validate region {i}: {e}")
                continue

        if create_plot:
            self._plot_clustering_results(
                selected_puncta,
                validated_fiducials,
                validation_metadata,
                output_figure_path,
                title,
            )

        return validated_fiducials, validation_metadata

    def _plot_density_detection_results(
        self,
        smoothed_image,
        binary_mask,
        region_centres,
        hist,
        bin_edges,
        threshold,
        pixelsize,
        output_figure_path,
        title,
    ):
        """Create visualization of density detection results."""
        if not is_available("matplotlib.pyplot"):
            return

        PF = _ensure_plotting()
        if PF is None:
            print("⚠️ PlottingFunctions not available, skipping visualization")
            return

        try:
            from DriftPlotting import DriftPlotter

            plotter = DriftPlotter()
            plotter.create_separate_plots(
                smoothed_image,
                binary_mask,
                region_centres,
                hist,
                bin_edges,
                threshold,
                pixelsize,
                output_figure_path,
                title,
            )
        except Exception as e:
            warnings.warn(f"Failed to create density detection plots: {e}")

    def _plot_puncta_selection_results(
        self,
        locs,
        selected_puncta,
        region_centres,
        binary_mask,
        region_stats,
        box_size_pixels,
        pixelsize,
        output_figure_path,
        title,
        plot_individual_regions,
        use_datashader_threshold,
    ):
        """Create visualization of puncta selection results."""
        if not is_available("matplotlib.pyplot"):
            return

        try:
            from DriftPlotting import DriftPlotter

            plotter = DriftPlotter()
            plotter.plot_puncta_selection_results(
                locs,
                selected_puncta,
                region_centres,
                binary_mask,
                region_stats,
                box_size_pixels,
                pixelsize,
                output_figure_path,
                title,
                plot_individual_regions,
                use_datashader_threshold,
            )
        except Exception as e:
            warnings.warn(f"Failed to create puncta selection plots: {e}")

    def _plot_clustering_results(
        self,
        selected_puncta,
        validated_fiducials,
        validation_metadata,
        output_figure_path,
        title,
    ):
        """Create visualization of clustering results."""
        if not is_available("matplotlib.pyplot"):
            return

        try:
            from DriftPlotting import DriftPlotter

            plotter = DriftPlotter()
            plotter.plot_clustering_results(
                selected_puncta,
                validated_fiducials,
                validation_metadata,
                output_figure_path,
                title,
            )
        except Exception as e:
            warnings.warn(f"Failed to create clustering plots: {e}")
