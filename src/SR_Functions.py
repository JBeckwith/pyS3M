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


class SuperRes_Functions:
    """Super-resolution microscopy analysis functions.

    Provides functionality for super-resolution image reconstruction,
    localization processing, and analysis for Bayer filter SMLM systems.
    """

    def __init__(self, 
                 mosaic_unit=np.array([["B", "G"], ["G", "R"]]),
                 io_functions=None,
                 helper_functions=None,
                 mask_functions=None,
                 image_analysis_functions=None,
                 spot_detection_functions=None,
                 plotter=None,
                 scmos=None):
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
        self.io = io_functions if io_functions is not None else IOFunctions.IO_Functions()
        self.helper = helper_functions if helper_functions is not None else HelperFunctions.Helper_Functions()
        self.mask = mask_functions if mask_functions is not None else MaskFunctions.Mask_Functions()
        self.image_analysis = image_analysis_functions if image_analysis_functions is not None else ImageAnalysisFunctions.Image_Analysis_Functions()
        self.spot_detection = spot_detection_functions if spot_detection_functions is not None else SpotDetectionFunctions.SpotDetection_Functions()
        self.plotter = plotter if plotter is not None else PlottingFunctions.Plotter()
        self.scmos = scmos if scmos is not None else sCMOSFunctions.sCMOS_Functions()

    def _filter_fit_results(self, fit_results, width, height):
        fit_results = fit_results[~np.isnan(fit_results)]
        fit_results = fit_results[fit_results["xc"] > 0]
        fit_results = fit_results[fit_results["xc"] < width]
        fit_results = fit_results[fit_results["yc"] > 0]
        fit_results = fit_results[fit_results["yc"] < height]

        fit_results = fit_results[fit_results["s_x"] > 0]
        fit_results = fit_results[fit_results["s_x"] < 3]
        fit_results = fit_results[fit_results["s_y"] > 0]
        fit_results = fit_results[fit_results["s_y"] < 3]

        fit_results = fit_results[fit_results["A_B"] > 0]
        fit_results = fit_results[fit_results["A_G"] > 0]
        fit_results = fit_results[fit_results["A_R"] > 0]

        fit_results = fit_results[fit_results["bg_B"] > 0]
        fit_results = fit_results[fit_results["bg_G"] > 0]
        fit_results = fit_results[fit_results["bg_R"] > 0]

        fit_results = fit_results.reset_index()
        return fit_results

    def _process_roi(
        self,
        photoelectron_data,
        detected_puncta,
        i,
        width,
        height,
        ROI_size,
        smoothing_function,
        read_noise,
        masks,
        frame_offset=0,
        is_multi_frame=False,
    ):
        """
        Process a single detected ROI to extract photoelectron data, smoothed data, and weights.

        Args:
            photoelectron_data (np.ndarray): Full photoelectron image data
            detected_puncta (np.ndarray): Array of detected puncta coordinates
            i (int): Index of current puncta to process
            width (int): Image width
            height (int): Image height
            ROI_size (int): Size of ROI to extract
            smoothing_function: Function for smoothing data
            read_noise: Read noise map or scalar
            masks (np.ndarray): Color masks
            frame_offset (int): Frame offset for plane labeling
            is_multi_frame (bool): Whether data has multiple frames

        Returns:
            tuple or None: (photoelectron_roi, smoothed_roi, weights_roi, mask_roi, coords, plane)
                          Returns None if ROI is invalid (not square)
        """
        xcentre = detected_puncta[i, 0]
        ycentre = detected_puncta[i, 1]
        frame = detected_puncta[i, 2] if is_multi_frame else 0

        # Calculate ROI boundaries
        xmin = np.max([0, int(xcentre - ROI_size / 2)])
        xmax = np.min([int(xcentre + ROI_size / 2), width])
        ymin = np.max([0, int(ycentre - ROI_size / 2)])
        ymax = np.min([int(ycentre + ROI_size / 2), height])

        # Skip non-square ROIs
        if xmax - xmin != ymax - ymin:
            return None

        # Extract photoelectron ROI
        if is_multi_frame:
            photoelectron_roi = (
                photoelectron_data[frame, xmin:xmax, ymin:ymax]
                if len(photoelectron_data.shape) > 2
                else photoelectron_data[xmin:xmax, ymin:ymax]
            )
        else:
            photoelectron_roi = photoelectron_data[xmin:xmax, ymin:ymax]

        # Extract read_noise ROI for weights calculation
        read_noise_roi = (
            read_noise[xmin:xmax, ymin:ymax]
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

        # Extract mask ROI
        mask_roi = masks[xmin:xmax, ymin:ymax, :]

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
        ROI_size=12,
        peak_wavelength=0.638,
        NA=1.49,
        pixel_size=0.069,
        s=5,
        perc_threshold=95,
        bayer_image=False,
    ):
        """example_spots_singleframe function
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
        image_files = self.helper.file_search(image_folder, image_type, "")
        metadatafiles = self.helper.file_search(image_folder, "metadata", "")
        start_x, start_y, width, height = self.io.metadata_reader_imageJ(metadatafiles[0])

        masks = self.mask.get_ROI_mask(
            ROI_x_start=start_x,
            ROI_y_start=start_y,
            width=width,
            height=height,
            mosaic_unit=self.mosaic_unit,
        )
        masks = np.dstack([masks[x] for x in masks.keys()])
        gain_map = gain_map[start_x : start_x + width, start_y : start_y + height]
        offset_map = offset_map[start_x : start_x + width, start_y : start_y + height]
        read_noise = read_noise[start_x : start_x + width, start_y : start_y + height]
        rqe = rqe[start_x : start_x + width, start_y : start_y + height]
        variance = variance[start_x : start_x + width, start_y : start_y + height]

        file = image_files[0]
        puncta_tofit = []
        smoothed_puncta_tofit = []
        masks_tofit = []
        weights_tofit = []
        relative_coords = []

        # Load photoelectron data using updated workflow
        photoelectron_data = self.io.read_tiff_tophotoelectrons(
            file,
            dtype="float32",
            gain_map=gain_map,
            offset_map=offset_map,
            rqe=rqe,
            frame=1,
        )

        detected_puncta = self.spot_detection.detect_puncta_in_image(
            photoelectron_data,
            pfa=pfa,
            variance=variance,
            wavelength=peak_wavelength,
            pixel_size=pixel_size,
            NA=NA,
            bayer_image=bayer_image,
            mf_factor=mf_factor,
            local_factor=local_factor,
            perc_threshold=perc_threshold,
        )

        # Extract detected ROIs and generate smoothed/weights only for ROIs (most memory efficient)
        for i in np.arange(len(detected_puncta)):
            result = self._process_roi(
                photoelectron_data,
                detected_puncta,
                i,
                width,
                height,
                ROI_size,
                smoothing_function,
                read_noise,
                masks,
                frame_offset=0,
                is_multi_frame=False,
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

        # Photoelectron data already available for plotting
        # No need to reload - photoelectron_data already exists from detection step

        fig, axs = self.plotter.two_column_plot(
            ncolumns=2, nrows=2, widthratio=[1, 1], heightratio=[1, 1]
        )
        image_to_show = self.scmos.var_weighted_uniform_filter(photoelectron_data, variance_map=variance, kernel_size=2)
        axs[0, 0] = self.plotter.image_scatter_plot(
            axs=axs[0, 0],
            data=image_to_show,
            xdata=detected_puncta[:, 0],
            ydata=detected_puncta[:, 1],
            vmin=np.percentile(image_to_show, 1),
            vmax=np.percentile(image_to_show, 99),
            s=s,
        )

        axs[0, 1] = self.plotter.image_scatter_plot(
            axs=axs[0, 1],
            data=image_to_show,
            xdata=fit_results["xc"].to_numpy(),
            ydata=fit_results["yc"].to_numpy(),
            vmin=np.percentile(image_to_show, 1),
            vmax=np.percentile(image_to_show, 99),
            s=s,
            scattercolor="#32cd32",
        )
        x = fit_results["xc"].to_numpy()
        y = fit_results["yc"].to_numpy()
        filter = ~np.isnan(x) & ~np.isnan(y)
        x = x[filter]
        y = y[filter]
        density_values, xedges, yedges = np.histogram2d(x=x, y=y, bins=50)
        max_density = np.unravel_index(
            np.argmax(density_values), shape=density_values.shape
        )
        max_y = int(xedges[max_density[0]]) + 100
        min_y = max_y - 200
        max_x = int(yedges[max_density[1]]) + 100
        min_x = max_x - 200

        import matplotlib.patches as patches

        rect = patches.Rectangle(
            (min_x, min_y), np.abs(max_x-min_x), np.abs(max_y-min_y), linewidth=0.5, edgecolor="white", facecolor="none"
        )

        # Add the patch to the Axes
        axs[0,0].add_patch(rect)
        axs[0,1].add_patch(rect)


        axs[1, 0] = self.plotter.image_scatter_plot(
            axs=axs[1, 0],
            data=image_to_show,
            vmin=np.percentile(image_to_show, 1),
            vmax=np.percentile(image_to_show, 99),
            xdata=detected_puncta[:, 0],
            ydata=detected_puncta[:, 1],
            s=s * 5,
        )
        axs[1, 0].set_ylim([min_y, max_y])
        axs[1, 0].set_xlim([min_x, max_x])
        axs[1, 1] = self.plotter.image_scatter_plot(
            axs=axs[1, 1],
            data=image_to_show,
            vmin=np.percentile(image_to_show, 1),
            vmax=np.percentile(image_to_show, 99),
            xdata=fit_results["xc"].to_numpy(),
            ydata=fit_results["yc"].to_numpy(),
            s=s * 5,
            scattercolor="#32cd32",
        )
        axs[1, 1].set_ylim([min_y, max_y])
        axs[1, 1].set_xlim([min_x, max_x])

        # Clean up plotting data
        del photoelectron_data
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
        ROI_size=12,
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
                    puncta_tofit.append(photoelectron_data[fval, xmin:xmax, ymin:ymax])
                    smoothed_puncta_tofit.append(
                        smoothed_data[fval, xmin:xmax, ymin:ymax]
                    )
                    masks_tofit.append(masks[xmin:xmax, ymin:ymax, :])
                    weights_tofit.append(weights[fval, xmin:xmax, ymin:ymax])
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
        ROI_size=12,
        peak_wavelength=0.638,
        NA=1.49,
        pixel_size=0.069,
        perc_threshold=98,
        image_type=".tif",
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


            image_type (str): image string end


        Returns:
            bayer_image (np.ndarray): colour images imaged through the bayer filter supplied
        """

        image_files = self.helper.file_search(image_folder, image_type, "")
        metadatafiles = self.helper.file_search(image_folder, "metadata", "")
        start_x, start_y, width, height = self.io.metadata_reader_imageJ(metadatafiles[0])

        masks = self.mask.get_ROI_mask(
            ROI_x_start=start_x,
            ROI_y_start=start_y,
            width=width,
            height=height,
            mosaic_unit=self.mosaic_unit,
        )
        masks = np.dstack([masks[x] for x in masks.keys()])
        gain_map = gain_map[start_x : start_x + width, start_y : start_y + height]
        offset_map = offset_map[start_x : start_x + width, start_y : start_y + height]
        read_noise = read_noise[start_x : start_x + width, start_y : start_y + height]
        rqe = rqe[start_x : start_x + width, start_y : start_y + height]
        variance = variance[start_x : start_x + width, start_y : start_y + height]

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
            puncta_tofit = []
            smoothed_puncta_tofit = []
            masks_tofit = []
            weights_tofit = []
            relative_coords = []
            planes = []

            fit_savename = file.split(".")[0] + ".h5"

            # Load photoelectron data using updated workflow
            photoelectron_data = self.io.read_tiff_tophotoelectrons(
                file, dtype="float32", gain_map=gain_map, offset_map=offset_map, rqe=rqe
            )

            detected_puncta = self.spot_detection.detect_puncta_in_stack_parallel(
                photoelectron_data,
                pfa=pfa,
                variance=variance,
                wavelength=peak_wavelength,
                pixel_size=pixel_size,
                NA=NA,
                perc_threshold=perc_threshold,
            )

            # Extract detected ROIs and generate smoothed/weights only for ROIs (most memory efficient)
            for i in np.arange(len(detected_puncta)):
                result = self._process_roi(
                    photoelectron_data,
                    detected_puncta,
                    i,
                    width,
                    height,
                    ROI_size,
                    smoothing_function,
                    read_noise,
                    masks,
                    frame_offset=0,
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

                puncta_tofit.append(photoelectron_roi)
                smoothed_puncta_tofit.append(smoothed_roi)
                masks_tofit.append(mask_roi)
                weights_tofit.append(weights_roi)
                relative_coords.append(coords)
                planes.append(plane)

            del photoelectron_data, detected_puncta
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
        ROI_size=12,
        peak_wavelength=0.638,
        NA=1.49,
        pixel_size=0.069,
        perc_threshold=98,
        image_type=".tif",
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
            masks (dict): dict of colour masks
            peak_wavelength (float): peak wavelength of PSF


            image_type (str): image string end


        Returns:
            bayer_image (np.ndarray): colour images imaged through the bayer filter supplied
        """

        image_files = self.helper.file_search(image_folder, image_type, "")
        metadatafiles = self.helper.file_search(image_folder, "metadata", "")
        start_x, start_y, width, height = self.io.metadata_reader_imageJ(metadatafiles[0])

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
        gain_map = gain_map[start_x : start_x + width, start_y : start_y + height]
        offset_map = offset_map[start_x : start_x + width, start_y : start_y + height]
        read_noise = read_noise[start_x : start_x + width, start_y : start_y + height]
        rqe = rqe[start_x : start_x + width, start_y : start_y + height]
        variance = variance[start_x : start_x + width, start_y : start_y + height]

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
            puncta_tofit = []
            smoothed_puncta_tofit = []
            masks_tofit = []
            weights_tofit = []
            relative_coords = []
            planes = []

            # Load photoelectron data using updated workflow
            photoelectron_data = self.io.read_tiff_tophotoelectrons(
                file, dtype="float32", gain_map=gain_map, offset_map=offset_map, rqe=rqe
            )

            detected_puncta = self.spot_detection.detect_puncta_in_stack_parallel(
                photoelectron_data,
                pfa=pfa,
                wavelength=peak_wavelength,
                variance=variance,
                pixel_size=pixel_size,
                NA=NA,
                perc_threshold=perc_threshold,
            )

            # Track the highest frame number for this file
            file_frames = (
                photoelectron_data.shape[0] if len(photoelectron_data.shape) > 2 else 1
            )

            # Extract detected ROIs and generate smoothed/weights only for ROIs (most memory efficient)
            for i in np.arange(len(detected_puncta)):
                result = self._process_roi(
                    photoelectron_data,
                    detected_puncta,
                    i,
                    width,
                    height,
                    ROI_size,
                    smoothing_function,
                    read_noise,
                    masks,
                    frame_offset=total_frames,
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

                puncta_tofit.append(photoelectron_roi)
                smoothed_puncta_tofit.append(smoothed_roi)
                masks_tofit.append(mask_roi)
                weights_tofit.append(weights_roi)
                relative_coords.append(coords)
                planes.append(plane)

            total_frames += file_frames
            del photoelectron_data, detected_puncta
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
