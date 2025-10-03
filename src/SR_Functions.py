# -*- coding: utf-8 -*-
"""
This class contains functions pertaining to analysis of images,
relating to the bayerSMLM concept.
jsb92, 2024/01/02
"""
import numpy as np
import pandas as pd
import os
import sys
import gc
from enum import Enum

module_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(module_dir)
import IOFunctions
import HelperFunctions
import MaskFunctions
import ImageAnalysisFunctions
from ImageAnalysisFunctions import FittingStrategy
import SpotDetectionFunctions
import PlottingFunctions
import sCMOSFunctions


class TemporalMedianMode(Enum):
    """Temporal median background subtraction modes.

    Attributes:
        NONE: No temporal median subtraction
        FITTING_ONLY: Subtract temporal median for fitting only (detection uses original)
        DETECTION_AND_FITTING: Subtract temporal median for both detection and fitting
    """
    NONE = 0
    FITTING_ONLY = 1
    DETECTION_AND_FITTING = 2


class SuperRes_Functions:
    """Super-resolution microscopy analysis functions.

    Provides functionality for super-resolution image reconstruction,
    localization processing, and analysis for Bayer filter SMLM systems.
    """

    def __init__(
        self,
        mosaic_unit=np.array([["B", "G"], ["G", "R"]]),
        io_functions=None,
        helper_functions=None,
        mask_functions=None,
        image_analysis_functions=None,
        spot_detection_functions=None,
        plotter=None,
        scmos=None,
    ):
        """Initialize SuperRes_Functions class.

        Args:
            mosaic_unit: Bayer mosaic pattern array. Defaults to standard
                        [["B", "G"], ["G", "R"]] pattern.
            io_functions: IO functions instance (default: creates new instance)
            helper_functions: Helper functions instance (default: creates new instance)
            mask_functions: Mask functions instance (default: creates new instance)
            image_analysis_functions: Image analysis functions instance (default: creates new instance)
            spot_detection_functions: Spot detection functions instance (default: creates new instance)
            plotter: Plotter instance (default: creates new instance)
        """
        self.mosaic_unit = mosaic_unit

        # Dependency injection with sensible defaults
        self.io = (
            io_functions if io_functions is not None else IOFunctions.IO_Functions()
        )
        self.helper = (
            helper_functions
            if helper_functions is not None
            else HelperFunctions.Helper_Functions()
        )
        self.mask = (
            mask_functions
            if mask_functions is not None
            else MaskFunctions.Mask_Functions()
        )
        self.image_analysis = (
            image_analysis_functions
            if image_analysis_functions is not None
            else ImageAnalysisFunctions.Image_Analysis_Functions()
        )
        self.spot_detection = (
            spot_detection_functions
            if spot_detection_functions is not None
            else SpotDetectionFunctions.SpotDetection_Functions()
        )
        self.plotter = plotter if plotter is not None else PlottingFunctions.Plotter()
        self.scmos = scmos if scmos is not None else sCMOSFunctions.sCMOS_Functions()

    def _filter_fit_results(self, fit_results, width, height):
        """Filter localization results based on physical and quality constraints.

        Applies multiple quality filters in a single pass for optimal performance:
        - Removes NaN values
        - Filters coordinates to be within image bounds
        - Filters sigma values to reasonable PSF range (0-3 pixels)
        - Ensures positive amplitudes and backgrounds for all color channels

        Args:
            fit_results: Structured array of localization results
            width: Image width in pixels
            height: Image height in pixels

        Returns:
            Filtered results with index reset
        """
        # Combine all filters into a single boolean mask for efficient filtering
        mask = (
            ~np.isnan(fit_results) &
            (fit_results["xc"] > 0) & (fit_results["xc"] < width) &
            (fit_results["yc"] > 0) & (fit_results["yc"] < height) &
            (fit_results["s_x"] > 0) & (fit_results["s_x"] < 3) &
            (fit_results["s_y"] > 0) & (fit_results["s_y"] < 3) &
            (fit_results["A_B"] > 0) & (fit_results["A_G"] > 0) & (fit_results["A_R"] > 0) &
            (fit_results["bg_B"] > 0) & (fit_results["bg_G"] > 0) & (fit_results["bg_R"] > 0)
        )

        return fit_results[mask].reset_index()

    def _process_roi(
        self,
        raw_data,
        detected_puncta,
        i,
        width,
        height,
        ROI_size,
        smoothing_function,
        read_noise,
        masks,
        gain_map=1.0,
        offset_map=0.0,
        rqe=1.0,
        frame_offset=0,
        is_multi_frame=False,
        raw_data_for_fitting=None,
        fitting_data_is_photoelectrons=False,
    ):
        """
        Process a single detected ROI to extract photoelectron data, smoothed data, and weights.

        Args:
            raw_data (np.ndarray): Full raw camera image data in ADU (used for extraction)
            detected_puncta (np.ndarray): Array of detected puncta coordinates
            i (int): Index of current puncta to process
            width (int): Image width
            height (int): Image height
            ROI_size (int): Size of ROI to extract
            smoothing_function: Function for smoothing data
            read_noise: Read noise map or scalar
            masks (np.ndarray): Color masks
            gain_map (matrix or float): Gain map for ADU to photoelectron conversion
            offset_map (matrix or float): Offset map for ADU to photoelectron conversion
            rqe (matrix or float): Relative quantum efficiency map
            frame_offset (int): Frame offset for plane labeling
            is_multi_frame (bool): Whether data has multiple frames
            raw_data_for_fitting (np.ndarray): Optional separate raw data to use for fitting
                (e.g., temporal median subtracted). If None, uses raw_data.
            fitting_data_is_photoelectrons (bool): If True, raw_data_for_fitting is already
                in photoelectrons and should not be converted again. Default False.

        Returns:
            tuple or None: (photoelectron_roi, smoothed_roi, weights_roi, mask_roi, coords, plane)
                          Returns None if ROI is invalid (not square)
        """
        # detected_puncta stores [row, col, frame] from np.where()
        # row = y, col = x (confirmed by test_real_spot_detection.py)
        ycentre = detected_puncta[i, 0]  # First index is row (y)
        xcentre = detected_puncta[i, 1]  # Second index is col (x)
        frame = detected_puncta[i, 2] if is_multi_frame else 0

        # Calculate ROI boundaries
        xmin = np.max([0, int(xcentre - ROI_size / 2)])
        xmax = np.min([int(xcentre + ROI_size / 2), width])
        ymin = np.max([0, int(ycentre - ROI_size / 2)])
        ymax = np.min([int(ycentre + ROI_size / 2), height])

        # Skip non-square ROIs
        roi_width = xmax - xmin
        roi_height = ymax - ymin
        if roi_width != roi_height:
            return None

        # Also check if ROI size is reasonable (not too small)
        if roi_width < 4 or roi_height < 4:
            return None

        # Determine which data to use for fitting
        data_for_fitting = raw_data_for_fitting if raw_data_for_fitting is not None else raw_data

        # Extract raw ROI for fitting (note: arrays are [row, col] = [y, x])
        if is_multi_frame:
            raw_roi = (
                data_for_fitting[frame, ymin:ymax, xmin:xmax]
                if len(data_for_fitting.shape) > 2
                else data_for_fitting[ymin:ymax, xmin:xmax]
            )
        else:
            raw_roi = data_for_fitting[ymin:ymax, xmin:xmax]

        # Verify ROI is actually square (sanity check)
        if raw_roi.shape[0] != raw_roi.shape[1]:
            import logging
            logging.warning(f"Non-square ROI extracted: {raw_roi.shape}, expected {roi_width}x{roi_height}")
            logging.warning(f"  Boundaries: xmin={xmin}, xmax={xmax}, ymin={ymin}, ymax={ymax}")
            logging.warning(f"  Image dims: width={width}, height={height}")
            logging.warning(f"  Center: ({xcentre}, {ycentre}), ROI_size={ROI_size}")
            return None

        # Extract camera parameter ROIs for conversion (arrays are [y, x])
        gain_roi = (
            gain_map[ymin:ymax, xmin:xmax]
            if not isinstance(gain_map, (int, float))
            else gain_map
        )
        offset_roi = (
            offset_map[ymin:ymax, xmin:xmax]
            if not isinstance(offset_map, (int, float))
            else offset_map
        )
        rqe_roi = (
            rqe[ymin:ymax, xmin:xmax] if not isinstance(rqe, (int, float)) else rqe
        )

        # Convert raw ROI to photoelectrons (skip if already in photoelectrons)
        if fitting_data_is_photoelectrons and raw_data_for_fitting is not None:
            # Data is already in photoelectrons, no conversion needed
            photoelectron_roi = raw_roi.astype(np.float32)
        else:
            # Convert from ADU to photoelectrons
            photoelectron_roi = self.io.convert_to_photoelectrons(
                raw_roi, gain_map=gain_roi, offset_map=offset_roi, rqe=rqe_roi
            )

        # Extract read_noise ROI for weights calculation (arrays are [y, x])
        read_noise_roi = (
            read_noise[ymin:ymax, xmin:xmax]
            if not isinstance(read_noise, (int, float))
            else read_noise
        )

        # Generate smoothed and weights only for this ROI
        smoothed_roi = self.io.apply_smoothing(
            photoelectron_roi, smoothing_function, dtype="float32"
        )
        weights_roi = self.io.generate_weights(
            smoothed_roi, read_noise=read_noise_roi, dtype="float32"
        )

        # Extract mask ROI (note: masks are indexed as [row, col] = [y, x])
        mask_roi = masks[ymin:ymax, xmin:xmax, :]

        # Return processed data and metadata
        coords = (xmin, ymin)
        plane = frame + frame_offset

        return photoelectron_roi, smoothed_roi, weights_roi, mask_roi, coords, plane

    def example_spots_singleframe(
        self,
        image_folder,
        image_type=".tif",
        smoothing_function=None,
        gain_map=None,
        offset_map=None,
        rqe=None,
        read_noise=None,
        variance=None,
        pfa=1e-3,
        mf_factor: float = 3.0,
        local_factor: float = 3.0,
        ROI_size=16,
        peak_wavelength=0.638,
        NA=1.49,
        pixel_size=0.069,
        s=5,
        sigma: float = 1.5,
        fraction_true: float = 0.2,
        use_variance_aware_demosaic: bool = True,
        temporal_median_mode: TemporalMedianMode = TemporalMedianMode.NONE,
        temporal_median_window: int = 100,
        frame_index: int = 1,
    ):
        """Example spot detection and fitting on a single frame with visualization.

        Demonstrates the complete workflow: spot detection, ROI extraction, fitting,
        and visualization with zoom insets on highest density region.

        Args:
            image_folder (str): Path to folder containing image files
            image_type (str): Image file extension (default: ".tif")
            smoothing_function: Function to smooth data
            gain_map (np.ndarray): 2D gain map
            offset_map (np.ndarray): 2D offset map
            rqe (np.ndarray): 2D RQE map
            read_noise (np.ndarray): 2D read noise map
            variance (np.ndarray): 2D variance map
            pfa (float): Probability of false alarm (default: 1e-3)
            mf_factor (float): Matched filter factor (default: 3.0)
            local_factor (float): Local threshold factor (default: 3.0)
            ROI_size (int): ROI extraction size (default: 12)
            peak_wavelength (float): PSF peak wavelength in µm (default: 0.638)
            NA (float): Numerical aperture (default: 1.49)
            pixel_size (float): Pixel size in µm (default: 0.069)
            s (int): Scatter plot marker size (default: 5)
            sigma (float): Gaussian sigma for detection (default: 1.5)
            fraction_true (float): Expected fraction of true spots (default: 0.2)
            use_variance_aware_demosaic (bool): Use variance-aware demosaicing (default: True)
            temporal_median_mode (TemporalMedianMode): Temporal median background subtraction mode:
                - NONE: No temporal median subtraction (default)
                - FITTING_ONLY: Subtract temporal median for fitting only, detection uses original data
                - DETECTION_AND_FITTING: Subtract temporal median for both detection and fitting
            temporal_median_window (int): Window for temporal median in frames, centered on the current
                frame (e.g., 100 frames = 50 before + 50 after). (default: 100)
            frame_index (int): Which frame to analyze (default: 1)

        Returns:
            tuple: (fig, axs) Matplotlib figure and axes with 2x2 subplot showing:
                - [0,0]: Detected spots on processed image (full field)
                - [0,1]: Fitted spots on raw image (full field)
                - [1,0]: Detected spots zoomed to highest density region
                - [1,1]: Fitted spots zoomed to highest density region
        """
        image_files = self.helper.file_search(image_folder, image_type, "")
        metadatafiles = self.helper.file_search(image_folder, "metadata", "")

        # Use metadata if available, otherwise use default ROI (full image)
        if metadatafiles:
            start_x, start_y, width, height = self.io.metadata_reader_imageJ(
                metadatafiles[0]
            )
        else:
            # Default to full image - will be updated after loading first frame
            start_x, start_y = 0, 0
            width, height = None, None

        file = image_files[0]
        puncta_tofit = []
        smoothed_puncta_tofit = []
        masks_tofit = []
        weights_tofit = []
        relative_coords = []

        # Load raw data for the requested frame
        raw_data = self.io.read_tiff(
            file,
            dtype="float32",
            frame=frame_index,
        )

        # Update width/height if they weren't set from metadata
        if width is None or height is None:
            height, width = raw_data.shape

        # Create default calibration maps if not provided (use full image size first)
        full_height, full_width = raw_data.shape
        if gain_map is None:
            gain_map = np.ones((full_height, full_width), dtype=np.float32)
        if offset_map is None:
            offset_map = np.zeros((full_height, full_width), dtype=np.float32)
        if rqe is None:
            rqe = np.ones((full_height, full_width), dtype=np.float32)
        if read_noise is None:
            read_noise = np.ones((full_height, full_width), dtype=np.float32) * 1.6  # Typical value
        if variance is None:
            variance = read_noise ** 2

        # Create masks for ROI
        masks = self.mask.get_ROI_mask(
            ROI_x_start=start_x,
            ROI_y_start=start_y,
            width=width,
            height=height,
            mosaic_unit=self.mosaic_unit,
        )
        masks = np.dstack([masks[x] for x in masks.keys()])

        # Slice calibration maps to ROI using correct indexing [y, x]
        gain_map = gain_map[start_y : start_y + height, start_x : start_x + width]
        offset_map = offset_map[start_y : start_y + height, start_x : start_x + width]
        read_noise = read_noise[start_y : start_y + height, start_x : start_x + width]
        rqe = rqe[start_y : start_y + height, start_x : start_x + width]
        variance = variance[start_y : start_y + height, start_x : start_x + width]

        # Prepare data based on temporal median mode
        raw_data_for_detection = raw_data
        raw_data_for_fitting = None

        if temporal_median_mode != TemporalMedianMode.NONE:
            # Load surrounding frames for median calculation
            half_window = temporal_median_window // 2

            # Get total frames in file
            import tifffile
            with tifffile.TiffFile(file, is_ome=False, is_mmstack=False, is_imagej=False) as tif:
                total_frames = len(tif.pages)

            # Determine frame range to load
            start_frame = max(0, frame_index - half_window)
            end_frame = min(total_frames, frame_index + half_window + 1)
            frame_range = list(range(start_frame, end_frame))

            # Load frames for temporal median
            frames_for_median = self.io.read_tiff(file, dtype="float32", frame=frame_range)
            if frames_for_median.ndim == 2:
                frames_for_median = frames_for_median[np.newaxis, :, :]

            # Compute temporal median subtraction
            median_subtracted_stack = self._compute_temporal_median(
                frames_for_median,
                median_window=temporal_median_window,
                buffer_frames=None  # Single frame analysis doesn't need buffer
            )

            # Extract the requested frame from median-subtracted stack
            frame_offset_in_stack = frame_index - start_frame
            median_subtracted = median_subtracted_stack[frame_offset_in_stack]

            # Apply to detection and/or fitting based on mode
            if temporal_median_mode == TemporalMedianMode.DETECTION_AND_FITTING:
                print(f"Applying temporal median for BOTH detection and fitting (window={temporal_median_window}, frames={len(frame_range)})")
                raw_data_for_detection = median_subtracted
                raw_data_for_fitting = median_subtracted
            elif temporal_median_mode == TemporalMedianMode.FITTING_ONLY:
                print(f"Applying temporal median for FITTING only (window={temporal_median_window}, frames={len(frame_range)})")
                raw_data_for_fitting = median_subtracted

            # Cleanup
            del frames_for_median, median_subtracted_stack
            gc.collect()

        # Demosaic the raw Bayer image for detection
        image_to_analyse = self._demosaic_image(
            raw_data_for_detection,
            use_variance_aware=use_variance_aware_demosaic,
            gain_map=gain_map,
            offset_map=offset_map,
            variance=variance,
        )

        detected_puncta = self.spot_detection.detect_puncta_in_image(
            image_to_analyse,
            pfa=pfa,
            variance=variance,
            wavelength=peak_wavelength,
            pixel_size=pixel_size,
            NA=NA,
            mf_factor=mf_factor,
            local_factor=local_factor,
            sigma=sigma,
            fraction_true=fraction_true,
        )

        # Extract detected ROIs and generate smoothed/weights only for ROIs (most memory efficient)
        # Detection uses appropriate data based on mode, fitting may use median-subtracted
        for i in np.arange(len(detected_puncta)):
            result = self._process_roi(
                raw_data_for_detection,
                detected_puncta,
                i,
                width,
                height,
                ROI_size,
                smoothing_function,
                read_noise,
                masks,
                gain_map=gain_map,
                offset_map=offset_map,
                rqe=rqe,
                frame_offset=0,
                is_multi_frame=False,
                raw_data_for_fitting=raw_data_for_fitting,
            )

            if result is None:
                continue

            photoelectron_roi, smoothed_roi, weights_roi, mask_roi, coords, _ = result

            puncta_tofit.append(photoelectron_roi)
            smoothed_puncta_tofit.append(smoothed_roi)
            masks_tofit.append(mask_roi)
            weights_tofit.append(weights_roi)
            relative_coords.append(coords)
        gc.collect()

        fit_results, _ = self.image_analysis.fit_puncta_parallel_method(
            puncta_tofit,
            smoothed_puncta_tofit,
            weights_tofit,
            relative_coords,
            list(np.zeros(len(puncta_tofit), dtype=int)),
            FittingStrategy.STANDARD,
            masks=masks_tofit,
        )
        columns = [
            "xc",
            "yc",
            "s_x",
            "s_y",
            "bg_B",
            "bg_G",
            "bg_R",
            "A_B",
            "A_G",
            "A_R",
            "chi_sqr",
            "frame",
        ]
        fit_results = pd.DataFrame(fit_results, columns=columns)

        # Create figure using PlottingBase for cleaner code
        try:
            from PlottingBase import AnalysisPlotter
            plotter = AnalysisPlotter(datashader_threshold=None)  # Use matplotlib for single frame
        except ImportError:
            # Fallback to old plotter
            plotter = None

        if plotter is not None:
            # Use new PlottingBase infrastructure
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches

            fig, axs = plt.subplots(2, 2, figsize=(12, 10), dpi=100)

            # Calculate percentiles for consistent display
            vmin_processed = np.percentile(image_to_analyse, 1)
            vmax_processed = np.percentile(image_to_analyse, 99)
            vmin_raw = np.percentile(raw_data, 1)
            vmax_raw = np.percentile(raw_data, 99)

            # Find highest density region for zoom
            x_fit = fit_results["xc"].to_numpy()
            y_fit = fit_results["yc"].to_numpy()
            valid_mask = ~np.isnan(x_fit) & ~np.isnan(y_fit)
            x_valid = x_fit[valid_mask]
            y_valid = y_fit[valid_mask]

            if len(x_valid) > 0:
                density_hist, x_edges, y_edges = np.histogram2d(x_valid, y_valid, bins=50)
                max_density_idx = np.unravel_index(np.argmax(density_hist), density_hist.shape)
                # Center of zoom region
                center_x = (x_edges[max_density_idx[0]] + x_edges[max_density_idx[0] + 1]) / 2
                center_y = (y_edges[max_density_idx[1]] + y_edges[max_density_idx[1] + 1]) / 2
                # Zoom window (100x100 pixels)
                zoom_size = 50
                min_x, max_x = center_x - zoom_size, center_x + zoom_size
                min_y, max_y = center_y - zoom_size, center_y + zoom_size
            else:
                # Default zoom to center if no fits
                min_x, max_x = width // 2 - 50, width // 2 + 50
                min_y, max_y = height // 2 - 50, height // 2 + 50

            # Top row: Full field views
            # [0,0] Detected spots on processed image
            im = plotter.create_image_plot(axs[0, 0], image_to_analyse,
                                          vmin=vmin_processed, vmax=vmax_processed,
                                          cmap='gray')
            # detected_puncta stores [row, col] = [y, x], but scatter needs (x, y)
            axs[0, 0].scatter(detected_puncta[:, 1], detected_puncta[:, 0],
                            s=s, c='red', marker='o', alpha=0.5)
            plotter.setup_axis(axs[0, 0], title="Detected Spots (Full Field)",
                             xlabel="X (px)", ylabel="Y (px)", grid=False, equal_aspect=True)

            # Add zoom rectangle
            rect = patches.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y,
                                    linewidth=1, edgecolor='cyan', facecolor='none')
            axs[0, 0].add_patch(rect)

            # [0,1] Fitted spots on raw image
            im = plotter.create_image_plot(axs[0, 1], raw_data,
                                          vmin=vmin_raw, vmax=vmax_raw,
                                          cmap='gray')
            axs[0, 1].scatter(x_fit, y_fit, s=s, c='lime', marker='o', alpha=0.5)
            plotter.setup_axis(axs[0, 1], title="Fitted Spots (Full Field)",
                             xlabel="X (px)", ylabel="Y (px)", grid=False, equal_aspect=True)

            # Add zoom rectangle
            rect = patches.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y,
                                    linewidth=1, edgecolor='cyan', facecolor='none')
            axs[0, 1].add_patch(rect)

            # Bottom row: Zoomed views
            # [1,0] Detected spots zoomed
            im = plotter.create_image_plot(axs[1, 0], image_to_analyse,
                                          vmin=vmin_processed, vmax=vmax_processed,
                                          cmap='gray')
            # detected_puncta stores [row, col] = [y, x], but scatter needs (x, y)
            axs[1, 0].scatter(detected_puncta[:, 1], detected_puncta[:, 0],
                            s=s * 5, c='red', marker='o', alpha=0.7)
            axs[1, 0].set_xlim(min_x, max_x)
            axs[1, 0].set_ylim(min_y, max_y)
            plotter.setup_axis(axs[1, 0], title="Detected Spots (Zoom)",
                             xlabel="X (px)", ylabel="Y (px)", grid=False, equal_aspect=True)
            plotter.add_scalebar(axs[1, 0], pixelsize=pixel_size * 1000, length_nm=1000,
                               label="1 μm", color='white')

            # [1,1] Fitted spots zoomed
            im = plotter.create_image_plot(axs[1, 1], raw_data,
                                          vmin=vmin_raw, vmax=vmax_raw,
                                          cmap='gray')
            axs[1, 1].scatter(x_fit, y_fit, s=s * 5, c='lime', marker='o', alpha=0.7)
            axs[1, 1].set_xlim(min_x, max_x)
            axs[1, 1].set_ylim(min_y, max_y)
            plotter.setup_axis(axs[1, 1], title="Fitted Spots (Zoom)",
                             xlabel="X (px)", ylabel="Y (px)", grid=False, equal_aspect=True)
            plotter.add_scalebar(axs[1, 1], pixelsize=pixel_size * 1000, length_nm=1000,
                               label="1 μm", color='white')

            plt.tight_layout()

        else:
            # Fallback to old plotting method
            fig, axs = self.plotter.two_column_plot(
                ncolumns=2, nrows=2, widthratio=[1, 1], heightratio=[1, 1]
            )
            # ... (keep original plotting code as fallback)

        # Clean up
        del raw_data
        gc.collect()

        return fig, axs

    def fit_FRET_data(
        self,
        photoelectron_data,
        smoothed_data,
        weights,
        masks,
        detected_puncta,
        frames,
        width,
        height,
        ROI_size=16,
        peak_wavelength=0.638,
        NA=1.49,
        pixel_size=0.069,
        image_type=".tif",
    ):
        """fit_FRET_data function
            analyses where fiducials are for images in image folder given boxes

        Args:
            fiducial_boxes (dict): dictionary of fiducial boxes.
            image_folder (str): where the images are
            smoothing_function (type): function to smooth data
            gain_map (np.2darray): 2darray of gain map
            offset_map (np.2darray): 2darray of offset map
            rqe (np.2darray): 2d array of RQE
            read_noise (np.2darray): 2d array of read noise
            masks (dict): dict of colour masks
            peak_wavelength (float): peak wavelength of PSF


            image_type (str): image string end


        Returns:
            bayer_image (np.ndarray): colour images imaged through the bayer filter supplied
        """
        result_params = [
            "xc",
            "yc",
            "s_x",
            "s_y",
            "bg_B",
            "bg_G",
            "bg_R",
            "A_B",
            "A_G",
            "A_R",
            "chi_sqr",
            "frame",
            "xc_err",
            "yc_err",
            "s_x_err",
            "s_y_err",
            "bg_B_err",
            "bg_G_err",
            "bg_R_err",
            "A_B_err",
            "A_G_err",
            "A_R_err",
        ]

        puncta_tofit = []
        smoothed_puncta_tofit = []
        masks_tofit = []
        weights_tofit = []
        relative_coords = []
        planes = []

        for i in np.arange(len(detected_puncta)):
            if i in frames.keys():
                xcentre = detected_puncta[i, 0]
                ycentre = detected_puncta[i, 1]
                xmin = np.max([0, int(xcentre - ROI_size / 2)])
                xmax = np.min([int(xcentre + ROI_size / 2), width])
                ymin = np.max([0, int(ycentre - ROI_size / 2)])
                ymax = np.min([int(ycentre + ROI_size / 2), height])
                if xmax - xmin != ymax - ymin:
                    continue
                for frame in frames[i]:
                    fval = int(
                        frame - (1000 * (i + 1))
                    )  # frames are labelled for post-hoc analysis
                    # Extract ROIs using correct numpy indexing [row, col] = [y, x]
                    puncta_tofit.append(photoelectron_data[fval, ymin:ymax, xmin:xmax])
                    smoothed_puncta_tofit.append(
                        smoothed_data[fval, ymin:ymax, xmin:xmax]
                    )
                    masks_tofit.append(masks[ymin:ymax, xmin:xmax, :])
                    weights_tofit.append(weights[fval, ymin:ymax, xmin:xmax])
                    relative_coords.append((xmin, ymin))
                    planes.append(frame)  # label
        del photoelectron_data, smoothed_data, weights, detected_puncta
        gc.collect()

        fit_results, fit_errors = self.image_analysis.fit_puncta_parallel_method(
            puncta_tofit,
            smoothed_puncta_tofit,
            weights_tofit,
            relative_coords,
            planes,
            FittingStrategy.STANDARD,
            masks=masks_tofit,
        )
        fit_tosave = np.hstack([fit_results, fit_errors])
        fit_results = pd.DataFrame(fit_tosave, columns=result_params)

        # do some filtering
        fit_results = self._filter_fit_results(fit_results, width, height)

        del (
            fit_tosave,
            fit_errors,
            puncta_tofit,
            smoothed_puncta_tofit,
            masks_tofit,
            weights_tofit,
            relative_coords,
            planes,
        )
        gc.collect()
        return fit_results

    def fit_SM_data(
        self,
        image_folder,
        smoothing_function,
        gain_map,
        offset_map,
        rqe,
        read_noise,
        variance,
        pfa=1e-3,
        ROI_size=16,
        peak_wavelength=0.638,
        NA=1.49,
        pixel_size=0.069,
        sigma: float = 1.5,
        fraction_true: float = 0.2,
        image_type=".tif",
        use_variance_aware_demosaic: bool = True,
    ):
        """Single-molecule data fitting function.

        Analyzes single-molecule localization microscopy data by detecting puncta
        and fitting them with Gaussian models to extract precise positions and photon counts.

        Args:
            image_folder (str): Path to folder containing image files
            smoothing_function (type): function to smooth data
            gain_map (np.2darray): 2darray of gain map
            offset_map (np.2darray): 2darray of offset map
            rqe (np.2darray): 2d array of RQE
            read_noise (np.2darray): 2d array of read noise
            masks (dict): dict of colour masks
            peak_wavelength (float): peak wavelength of PSF
            sigma (float): sigma parameter for spot detection (default: 1.5)
            fraction_true (float): fraction of true spots expected (default: 0.2)
            image_type (str): image string end
            use_variance_aware_demosaic (bool): Whether to use variance-aware demosaicing for spot detection.
                If True (default), uses gain, offset, and variance maps to create robust photoelectron
                images that suppress hot pixels. If False, uses standard grayscale demosaicing.


        Returns:
            bayer_image (np.ndarray): colour images imaged through the bayer filter supplied
        """

        image_files = self.helper.file_search(image_folder, image_type, "")
        metadatafiles = self.helper.file_search(image_folder, "metadata", "")
        start_x, start_y, width, height = self.io.metadata_reader_imageJ(
            metadatafiles[0]
        )

        masks = self.mask.get_ROI_mask(
            ROI_x_start=start_x,
            ROI_y_start=start_y,
            width=width,
            height=height,
            mosaic_unit=self.mosaic_unit,
        )
        masks = np.dstack([masks[x] for x in masks.keys()])
        # Slice calibration maps using correct indexing [y, x]
        gain_map = gain_map[start_y : start_y + height, start_x : start_x + width]
        offset_map = offset_map[start_y : start_y + height, start_x : start_x + width]
        read_noise = read_noise[start_y : start_y + height, start_x : start_x + width]
        rqe = rqe[start_y : start_y + height, start_x : start_x + width]
        variance = variance[start_y : start_y + height, start_x : start_x + width]

        result_params = [
            "xc",
            "yc",
            "s_x",
            "s_y",
            "bg_B",
            "bg_G",
            "bg_R",
            "A_B",
            "A_G",
            "A_R",
            "chi_sqr",
            "frame",
            "xc_err",
            "yc_err",
            "s_x_err",
            "s_y_err",
            "bg_B_err",
            "bg_G_err",
            "bg_R_err",
            "A_B_err",
            "A_G_err",
            "A_R_err",
        ]

        for FOVn, file in enumerate(image_files):
            fit_savename = file.split(".")[0] + ".h5"

            # Get total frame count without loading entire file
            import tifffile

            with tifffile.TiffFile(
                file, is_ome=False, is_mmstack=False, is_imagej=False
            ) as tif:
                total_frames = len(tif.pages)

            chunk_size = 1000
            all_puncta_tofit = []
            all_smoothed_puncta_tofit = []
            all_masks_tofit = []
            all_weights_tofit = []
            all_relative_coords = []
            all_planes = []

            print(
                f"Processing file {FOVn+1}/{len(image_files)}: {total_frames} frames in chunks of {chunk_size}"
            )

            # Process file in chunks
            for chunk_start in range(0, total_frames, chunk_size):
                chunk_end = min(chunk_start + chunk_size, total_frames)
                chunk_frames = list(range(chunk_start, chunk_end))

                print(f"  Processing chunk: frames {chunk_start}-{chunk_end-1}")

                # Load chunk of raw data
                raw_data = self.io.read_tiff(file, dtype="float32", frame=chunk_frames)

                # Ensure raw_data is 3D even for single frame chunks
                if raw_data.ndim == 2:
                    raw_data = raw_data[np.newaxis, :, :]

                # Demosaic the raw Bayer image
                image_to_analyse = self._demosaic_image(
                    raw_data,
                    use_variance_aware=use_variance_aware_demosaic,
                    gain_map=gain_map,
                    offset_map=offset_map,
                    variance=variance,
                )

                detected_puncta = self.spot_detection.detect_puncta_in_stack_parallel(
                    image_to_analyse,
                    pfa=pfa,
                    variance=variance,
                    wavelength=peak_wavelength,
                    pixel_size=pixel_size,
                    NA=NA,
                    sigma=sigma,
                    fraction_true=fraction_true,
                )

                # Process ROIs for this chunk (keep original frame indices for raw_data access)
                for i in np.arange(len(detected_puncta)):
                    result = self._process_roi(
                        raw_data,
                        detected_puncta,  # Keep original frame indices (0-999, 0-999, etc.)
                        i,
                        width,
                        height,
                        ROI_size,
                        smoothing_function,
                        read_noise,
                        masks,
                        gain_map=gain_map,
                        offset_map=offset_map,
                        rqe=rqe,
                        frame_offset=chunk_start,  # Frame offset for this chunk
                        is_multi_frame=True,
                    )

                    if result is None:
                        continue

                    (
                        photoelectron_roi,
                        smoothed_roi,
                        weights_roi,
                        mask_roi,
                        coords,
                        plane,
                    ) = result

                    # plane is already correctly offset by _process_roi frame_offset

                    all_puncta_tofit.append(photoelectron_roi)
                    all_smoothed_puncta_tofit.append(smoothed_roi)
                    all_masks_tofit.append(mask_roi)
                    all_weights_tofit.append(weights_roi)
                    all_relative_coords.append(coords)
                    all_planes.append(plane)

                # Clean up chunk data
                del raw_data, detected_puncta, image_to_analyse
                if 'buffer_data' in locals() and buffer_data is not None:
                    del buffer_data
                gc.collect()

            print(f"  Found {len(all_puncta_tofit)} puncta across all chunks")

            # Move all data to final arrays for fitting
            puncta_tofit = all_puncta_tofit
            smoothed_puncta_tofit = all_smoothed_puncta_tofit
            masks_tofit = all_masks_tofit
            weights_tofit = all_weights_tofit
            relative_coords = all_relative_coords
            planes = all_planes

            # ROI processing already done in chunks above

            fit_results, fit_errors = self.image_analysis.fit_puncta_parallel_method(
                puncta_tofit,
                smoothed_puncta_tofit,
                weights_tofit,
                relative_coords,
                planes,
                FittingStrategy.STANDARD,
                masks=masks_tofit,
            )
            fit_tosave = np.hstack([fit_results, fit_errors])
            fit_results = pd.DataFrame(fit_tosave, columns=result_params)

            # Fix frame numbers: replace with offset plane values for continuous numbering
            if len(planes) == len(fit_results):
                fit_results["frame"] = planes

            # Sort by frame for consistent ordering in saved files
            fit_results = fit_results.sort_values("frame").reset_index(drop=True)

            # do some filtering
            fit_results = self._filter_fit_results(fit_results, width, height)

            self.io._write_h5_database(fit_results, fit_savename, append=False)
            del (
                fit_tosave,
                fit_results,
                fit_errors,
                puncta_tofit,
                smoothed_puncta_tofit,
                masks_tofit,
                weights_tofit,
                relative_coords,
                planes,
            )
            gc.collect()
        return

    def _compute_temporal_median(
        self,
        frames: np.ndarray,
        median_window: int = 100,
        buffer_frames: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute moving temporal median for background subtraction.

        Computes a moving temporal median over a specified window to remove
        slowly varying background. Handles edge cases at chunk boundaries.

        Args:
            frames: 3D array of frames (n_frames, height, width)
            median_window: Window size for temporal median (default: 100 frames)
            buffer_frames: Optional buffer frames from next chunk for edge handling

        Returns:
            Temporal median subtracted frames (same shape as input)

        Example:
            >>> # Process chunk with buffer
            >>> cleaned = _compute_temporal_median(chunk_frames, median_window=100,
            ...                                   buffer_frames=next_chunk_frames[:100])
        """
        n_frames, height, width = frames.shape
        median_subtracted = np.zeros_like(frames, dtype=np.float32)

        # Calculate half window for centered median
        half_window = median_window // 2

        for i in range(n_frames):
            # Determine window bounds
            start_idx = max(0, i - half_window)
            end_idx = min(n_frames, i + half_window + 1)

            # Check if we need buffer frames for the end
            if buffer_frames is not None and i + half_window >= n_frames:
                # Need frames from buffer
                n_needed = (i + half_window + 1) - n_frames
                n_available = min(n_needed, len(buffer_frames))

                if n_available > 0:
                    # Concatenate current chunk frames with buffer
                    window_frames = np.concatenate(
                        [frames[start_idx:], buffer_frames[:n_available]], axis=0
                    )
                else:
                    # No buffer available, use what we have
                    window_frames = frames[start_idx:end_idx]
            else:
                # Normal case - window entirely within current chunk
                window_frames = frames[start_idx:end_idx]

            # Compute median along temporal axis
            temporal_median = np.median(window_frames, axis=0)

            # Subtract median from current frame
            median_subtracted[i] = frames[i] - temporal_median

        return median_subtracted

    def _demosaic_image(
        self,
        raw_data: np.ndarray,
        use_variance_aware: bool = True,
        gain_map: np.ndarray = None,
        offset_map: np.ndarray = None,
        variance: np.ndarray = None,
    ) -> np.ndarray:
        """Demosaic Bayer pattern image using specified method.

        Args:
            raw_data: Bayer pattern image (2D or 3D array)
            use_variance_aware: If True, use variance-aware demosaicing with calibration maps.
                              If False, use standard grayscale demosaicing. (default: True)
            gain_map: Gain calibration map (required if use_variance_aware=True)
            offset_map: Offset calibration map (required if use_variance_aware=True)
            variance: Variance map (required if use_variance_aware=True)

        Returns:
            Demosaiced grayscale image (same shape as input)

        Notes:
            - Variance-aware demosaicing uses calibration maps to suppress hot pixels
              and create robust photoelectron images
            - Standard demosaicing uses simple Bayer-to-grayscale conversion
        """
        if use_variance_aware:
            # Use variance-aware demosaicing for robust spot detection
            return self.scmos.variance_aware_malvar_demosaic(
                raw_data,
                variance_map=variance,
                offset_map=offset_map,
                gain=gain_map,
                grayscale=True,
            )
        else:
            # Use standard grayscale demosaicing
            return self.scmos.bayer_demosaic_stack_grayscale(raw_data)

    def fit_imaging_data(
        self,
        image_folder,
        smoothing_function,
        gain_map,
        offset_map,
        rqe,
        read_noise,
        variance,
        pfa=1e-3,
        ROI_size=20,
        peak_wavelength=0.638,
        NA=1.49,
        pixel_size=0.069,
        sigma: float = 1.5,
        fraction_true: float = 0.2,
        image_type=".tif",
        use_variance_aware_demosaic: bool = True,
        temporal_median_mode: TemporalMedianMode = TemporalMedianMode.NONE,
        temporal_median_window: int = 100,
    ):
        """Cross-file imaging data fitting function.

        Analyzes imaging data across multiple files by detecting puncta and fitting them
        with Gaussian models, maintaining frame numbering consistency across files.

        Args:
            image_folder (str): Path to folder containing image files
            smoothing_function (type): function to smooth data
            gain_map (np.2darray): 2darray of gain map
            offset_map (np.2darray): 2darray of offset map
            rqe (np.2darray): 2d array of RQE
            read_noise (np.2darray): 2d array of read noise
            variance (np.2darray): 2d array of variance
            pfa (float): Probability of false alarm for spot detection (default: 1e-3)
            ROI_size (int): Size of ROI for fitting (default: 12)
            peak_wavelength (float): peak wavelength of PSF (default: 0.638)
            NA (float): Numerical aperture (default: 1.49)
            pixel_size (float): Pixel size in microns (default: 0.069)
            sigma (float): sigma parameter for spot detection (default: 1.5)
            fraction_true (float): fraction of true spots expected (default: 0.2)
            image_type (str): image file extension (default: ".tif")
            use_variance_aware_demosaic (bool): Whether to use variance-aware demosaicing for spot detection.
                If True (default), uses gain, offset, and variance maps to create robust photoelectron
                images that suppress hot pixels. If False, uses standard grayscale demosaicing.
            temporal_median_mode (TemporalMedianMode): Temporal median background subtraction mode:
                - NONE: No temporal median subtraction (default)
                - FITTING_ONLY: Subtract temporal median for fitting only, detection uses original data
                - DETECTION_AND_FITTING: Subtract temporal median for both detection and fitting
            temporal_median_window (int): Window size (in frames) for temporal median calculation.
                The median is centered on each frame (e.g., for frame N with window=100, uses 50 frames
                before and 50 frames after). Larger windows better remove slow drift but require more
                memory. Can load frames across file boundaries. (default: 100)

        Returns:
            None: Writes results to HDF5 file in image_folder/Localisations.h5
        """

        image_files = self.helper.file_search(image_folder, image_type, "")
        metadatafiles = self.helper.file_search(image_folder, "metadata", "")
        start_x, start_y, width, height = self.io.metadata_reader_imageJ(
            metadatafiles[0]
        )

        fit_savename = os.path.join(
            os.path.split(metadatafiles[0])[0], "Localisations.h5"
        )
        masks = self.mask.get_ROI_mask(
            ROI_x_start=start_x,
            ROI_y_start=start_y,
            width=width,
            height=height,
            mosaic_unit=self.mosaic_unit,
        )
        masks = np.dstack([masks[x] for x in masks.keys()])
        # Slice calibration maps using correct indexing [y, x]
        gain_map = gain_map[start_y : start_y + height, start_x : start_x + width]
        offset_map = offset_map[start_y : start_y + height, start_x : start_x + width]
        read_noise = read_noise[start_y : start_y + height, start_x : start_x + width]
        rqe = rqe[start_y : start_y + height, start_x : start_x + width]
        variance = variance[start_y : start_y + height, start_x : start_x + width]

        result_params = [
            "xc",
            "yc",
            "s_x",
            "s_y",
            "bg_B",
            "bg_G",
            "bg_R",
            "A_B",
            "A_G",
            "A_R",
            "chi_sqr",
            "frame",
            "xc_err",
            "yc_err",
            "s_x_err",
            "s_y_err",
            "bg_B_err",
            "bg_G_err",
            "bg_R_err",
            "A_B_err",
            "A_G_err",
            "A_R_err",
        ]

        total_frames = 0
        for FOVn, file in enumerate(image_files):
            # Get total frame count without loading entire file
            import tifffile

            with tifffile.TiffFile(
                file, is_ome=False, is_mmstack=False, is_imagej=False
            ) as tif:
                file_frames = len(tif.pages)

            chunk_size = 1000
            all_puncta_tofit = []
            all_smoothed_puncta_tofit = []
            all_masks_tofit = []
            all_weights_tofit = []
            all_relative_coords = []
            all_planes = []

            print(
                f"Processing file {FOVn+1}/{len(image_files)}: {file_frames} frames in chunks of {chunk_size}"
            )

            # Process file in chunks
            for chunk_start in range(0, file_frames, chunk_size):
                chunk_end = min(chunk_start + chunk_size, file_frames)
                chunk_frames = list(range(chunk_start, chunk_end))

                print(f"  Processing chunk: frames {chunk_start}-{chunk_end-1}")

                # Load chunk of raw data
                raw_data = self.io.read_tiff(file, dtype="float32", frame=chunk_frames)

                # Ensure raw_data is 3D even for single frame chunks
                if raw_data.ndim == 2:
                    raw_data = raw_data[np.newaxis, :, :]

                # Prepare data based on temporal median mode
                raw_data_for_detection = raw_data
                raw_data_for_fitting = None
                buffer_data = None

                if temporal_median_mode != TemporalMedianMode.NONE:
                    half_window = temporal_median_window // 2
                    # Check if we need buffer frames from next chunk or next file
                    if chunk_end < file_frames:
                        # Load buffer frames from current file
                        buffer_start = chunk_end
                        buffer_end = min(chunk_end + half_window, file_frames)
                        if buffer_end > buffer_start:
                            buffer_frames = list(range(buffer_start, buffer_end))
                            buffer_data = self.io.read_tiff(
                                file, dtype="float32", frame=buffer_frames
                            )
                            if buffer_data.ndim == 2:
                                buffer_data = buffer_data[np.newaxis, :, :]
                    elif chunk_end == file_frames and FOVn + 1 < len(image_files):
                        # Load buffer frames from next file
                        next_file = image_files[FOVn + 1]
                        buffer_frames = list(range(0, min(half_window, file_frames)))
                        if len(buffer_frames) > 0:
                            buffer_data = self.io.read_tiff(
                                next_file, dtype="float32", frame=buffer_frames
                            )
                            if buffer_data.ndim == 2:
                                buffer_data = buffer_data[np.newaxis, :, :]

                    # Compute temporal median subtracted data
                    median_subtracted = self._compute_temporal_median(
                        raw_data,
                        median_window=temporal_median_window,
                        buffer_frames=buffer_data
                    )

                    # Apply to detection and/or fitting based on mode
                    if temporal_median_mode == TemporalMedianMode.DETECTION_AND_FITTING:
                        print(f"    Applying temporal median for BOTH detection and fitting (window={temporal_median_window})")
                        raw_data_for_detection = median_subtracted
                        raw_data_for_fitting = median_subtracted
                    elif temporal_median_mode == TemporalMedianMode.FITTING_ONLY:
                        print(f"    Applying temporal median for FITTING only (window={temporal_median_window})")
                        raw_data_for_fitting = median_subtracted

                # Demosaic the raw Bayer image for detection
                image_to_analyse = self._demosaic_image(
                    raw_data_for_detection,
                    use_variance_aware=use_variance_aware_demosaic,
                    gain_map=gain_map,
                    offset_map=offset_map,
                    variance=variance,
                )

                detected_puncta = self.spot_detection.detect_puncta_in_stack_parallel(
                    image_to_analyse,
                    pfa=pfa,
                    wavelength=peak_wavelength,
                    variance=variance,
                    pixel_size=pixel_size,
                    NA=NA,
                    sigma=sigma,
                    fraction_true=fraction_true,
                )

                # Process ROIs for this chunk
                # Detection uses original data, fitting uses temporal median subtracted if enabled
                for i in np.arange(len(detected_puncta)):
                    result = self._process_roi(
                        raw_data_for_detection,
                        detected_puncta,  # Keep original frame indices (0-999, 0-999, etc.)
                        i,
                        width,
                        height,
                        ROI_size,
                        smoothing_function,
                        read_noise,
                        masks,
                        gain_map=gain_map,
                        offset_map=offset_map,
                        rqe=rqe,
                        frame_offset=total_frames
                        + chunk_start,  # Global frame offset including chunk
                        is_multi_frame=True,
                        raw_data_for_fitting=raw_data_for_fitting,
                    )

                    if result is None:
                        continue

                    (
                        photoelectron_roi,
                        smoothed_roi,
                        weights_roi,
                        mask_roi,
                        coords,
                        plane,
                    ) = result

                    # plane is already correctly offset by _process_roi frame_offset

                    all_puncta_tofit.append(photoelectron_roi)
                    all_smoothed_puncta_tofit.append(smoothed_roi)
                    all_masks_tofit.append(mask_roi)
                    all_weights_tofit.append(weights_roi)
                    all_relative_coords.append(coords)
                    all_planes.append(plane)

                # Clean up chunk data
                del raw_data_for_detection, detected_puncta, image_to_analyse
                if raw_data_for_fitting is not None:
                    del raw_data_for_fitting
                if buffer_data is not None:
                    del buffer_data
                gc.collect()

            print(f"  Found {len(all_puncta_tofit)} puncta across all chunks")

            # Move all data to final arrays for fitting
            puncta_tofit = all_puncta_tofit
            smoothed_puncta_tofit = all_smoothed_puncta_tofit
            masks_tofit = all_masks_tofit
            weights_tofit = all_weights_tofit
            relative_coords = all_relative_coords
            planes = all_planes

            # ROI processing already done in chunks above
            total_frames += file_frames

            fit_results, fit_errors = self.image_analysis.fit_puncta_parallel_method(
                puncta_tofit,
                smoothed_puncta_tofit,
                weights_tofit,
                relative_coords,
                planes,
                FittingStrategy.STANDARD,
                masks=masks_tofit,
            )
            fit_tosave = np.hstack([fit_results, fit_errors])
            fit_results = pd.DataFrame(fit_tosave, columns=result_params)

            # Fix frame numbers: replace with offset plane values for continuous numbering
            if len(planes) == len(fit_results):
                fit_results["frame"] = planes

            # Sort by frame for consistent ordering in saved files
            fit_results = fit_results.sort_values("frame").reset_index(drop=True)

            # do some filtering
            fit_results = self._filter_fit_results(fit_results, width, height)

            if FOVn == 0:
                self.io._write_h5_database(fit_results, fit_savename, append=False)
            else:
                self.io._write_h5_database(fit_results, fit_savename, append=True)
            del (
                fit_tosave,
                fit_results,
                fit_errors,
                puncta_tofit,
                smoothed_puncta_tofit,
                masks_tofit,
                weights_tofit,
                relative_coords,
                planes,
            )
            gc.collect()
        return
