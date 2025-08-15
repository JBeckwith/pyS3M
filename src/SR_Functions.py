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

IO = IOFunctions.IO_Functions()

import sCMOSFunctions

sCMOS = sCMOSFunctions.sCMOS_Functions()

import PSFFunctions

PSF_F = PSFFunctions.PSF_Functions()

import HelperFunctions

H_F = HelperFunctions.Helper_Functions()

from src import MaskFunctions

M_F = MaskFunctions.Mask_Functions()

from src import ImageAnalysisFunctions

I_AF = ImageAnalysisFunctions.Image_Analysis_Functions()

from src import SpotDetectionFunctions

SD_F = SpotDetectionFunctions.SpotDetection_Functions()

from src import PlottingFunctions

plotter = PlottingFunctions.Plotter()


class SuperRes_Functions:
    """Super-resolution microscopy analysis functions.
    
    Provides functionality for super-resolution image reconstruction,
    localization processing, and analysis for Bayer filter SMLM systems.
    """
    
    def __init__(self, mosaic_unit=np.array([["B", "G"], ["G", "R"]])):
        """Initialize SuperRes_Functions class.
        
        Args:
            mosaic_unit: Bayer mosaic pattern array. Defaults to standard 
                        [["B", "G"], ["G", "R"]] pattern.
        """
        self.mosaic_unit = mosaic_unit

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

    def example_spots_singleframe(
        self,
        image,
        pfa=1e-3,
        ROI_size=12,
        peak_wavelength=0.638,
        NA=1.49,
        pixel_size=0.069,
        s=5,
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
        detected_puncta = SD_F.detect_puncta_in_image(
            image,
            pfa=pfa,
            wavelength=peak_wavelength,
            pixel_size=pixel_size,
            NA=NA,
        )
        fig, axs = plotter.two_column_plot()
        axs = plotter.image_scatter_plot(
            axs=axs,
            data=image,
            xdata=detected_puncta[:, 0],
            ydata=detected_puncta[:, 1],
            s=s,
        )
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

        fit_results, fit_errors = I_AF.fit_puncta_parallel(
            puncta_tofit,
            smoothed_puncta_tofit,
            masks_tofit,
            weights_tofit,
            relative_coords,
            planes,
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
        pfa=1e-3,
        ROI_size=12,
        peak_wavelength=0.638,
        NA=1.49,
        pixel_size=0.069,
        image_type=".tif",
    ):
        """fiducial_correction function
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

        image_files = H_F.file_search(image_folder, image_type, "")
        metadatafiles = H_F.file_search(image_folder, "metadata", "")
        start_x, start_y, width, height = IO.metadata_reader_imageJ(metadatafiles[0])

        masks = M_F.get_ROI_mask(
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
            n_frames = np.arange(IO.get_num_pages_in_TIF(file), dtype=int)
            photoelectron_data, smoothed_data, weights = IO.read_tiff_tophotoelectrons(
                file,
                smoothing_function,
                dtype=np.float32,
                frame=n_frames,
                gain_map=gain_map,
                offset_map=offset_map,
                read_noise=read_noise,
                rqe=rqe,
            )
            detected_puncta = SD_F.detect_puncta_in_stack_parallel(
                photoelectron_data,
                pfa=pfa,
                wavelength=peak_wavelength,
                pixel_size=pixel_size,
                NA=NA,
            )
            for i in np.arange(len(detected_puncta)):
                xcentre = detected_puncta[i, 0]
                ycentre = detected_puncta[i, 1]
                frame = detected_puncta[i, 2]
                xmin = np.max([0, int(xcentre - ROI_size / 2)])
                xmax = np.min([int(xcentre + ROI_size / 2), width])
                ymin = np.max([0, int(ycentre - ROI_size / 2)])
                ymax = np.min([int(ycentre + ROI_size / 2), height])

                if xmax - xmin != ymax - ymin:
                    continue
                puncta_tofit.append(photoelectron_data[frame, xmin:xmax, ymin:ymax])
                smoothed_puncta_tofit.append(smoothed_data[frame, xmin:xmax, ymin:ymax])
                masks_tofit.append(masks[xmin:xmax, ymin:ymax, :])
                weights_tofit.append(weights[frame, xmin:xmax, ymin:ymax])
                relative_coords.append((xmin, ymin))
                planes.append(frame)
            del photoelectron_data, smoothed_data, weights, detected_puncta
            gc.collect()

            fit_results, fit_errors = I_AF.fit_puncta_parallel(
                puncta_tofit,
                smoothed_puncta_tofit,
                masks_tofit,
                weights_tofit,
                relative_coords,
                planes,
            )
            fit_tosave = np.hstack([fit_results, fit_errors])
            fit_results = pd.DataFrame(fit_tosave, columns=result_params)

            # do some filtering
            fit_results = self._filter_fit_results(fit_results, width, height)

            IO._write_h5_database(fit_results, fit_savename, append=False)
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
        pfa=1e-3,
        ROI_size=12,
        peak_wavelength=0.638,
        NA=1.49,
        pixel_size=0.069,
        image_type=".tif",
    ):
        """fiducial_correction function
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

        image_files = H_F.file_search(image_folder, image_type, "")
        metadatafiles = H_F.file_search(image_folder, "metadata", "")
        start_x, start_y, width, height = IO.metadata_reader_imageJ(metadatafiles[0])

        fit_savename = os.path.join(
            os.path.split(metadatafiles[0])[0], "Localisations.h5"
        )
        masks = M_F.get_ROI_mask(
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

            n_frames = np.arange(IO.get_num_pages_in_TIF(file), dtype=int)
            photoelectron_data, smoothed_data, weights = IO.read_tiff_tophotoelectrons(
                file,
                smoothing_function,
                dtype=np.float32,
                frame=n_frames,
                gain_map=gain_map,
                offset_map=offset_map,
                read_noise=read_noise,
                rqe=rqe,
            )
            detected_puncta = SD_F.detect_puncta_in_stack_parallel(
                photoelectron_data,
                pfa=pfa,
                wavelength=peak_wavelength,
                pixel_size=pixel_size,
                NA=NA,
            )
            for i in np.arange(len(detected_puncta)):
                xcentre = detected_puncta[i, 0]
                ycentre = detected_puncta[i, 1]
                frame = detected_puncta[i, 2]
                xmin = np.max([0, int(xcentre - ROI_size / 2)])
                xmax = np.min([int(xcentre + ROI_size / 2), width])
                ymin = np.max([0, int(ycentre - ROI_size / 2)])
                ymax = np.min([int(ycentre + ROI_size / 2), height])

                if xmax - xmin != ymax - ymin:
                    continue
                puncta_tofit.append(photoelectron_data[frame, xmin:xmax, ymin:ymax])
                smoothed_puncta_tofit.append(smoothed_data[frame, xmin:xmax, ymin:ymax])
                masks_tofit.append(masks[xmin:xmax, ymin:ymax, :])
                weights_tofit.append(weights[frame, xmin:xmax, ymin:ymax])
                relative_coords.append((xmin, ymin))
                planes.append(frame + total_frames)
            total_frames += frame
            del photoelectron_data, smoothed_data, weights, detected_puncta
            gc.collect()

            fit_results, fit_errors = I_AF.fit_puncta_parallel(
                puncta_tofit,
                smoothed_puncta_tofit,
                masks_tofit,
                weights_tofit,
                relative_coords,
                planes,
            )
            fit_tosave = np.hstack([fit_results, fit_errors])
            fit_results = pd.DataFrame(fit_tosave, columns=result_params)

            # do some filtering
            fit_results = self._filter_fit_results(fit_results, width, height)

            if FOVn == 0:
                IO._write_h5_database(fit_results, fit_savename, append=False)
            else:
                IO._write_h5_database(fit_results, fit_savename, append=True)
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
