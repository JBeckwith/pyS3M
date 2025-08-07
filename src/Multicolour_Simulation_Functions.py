import os
import sys
import numpy as np
from copy import deepcopy
import time
from scipy.spatial.distance import cdist
import gc
import pandas as pd

module_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(module_dir)

import IOFunctions

IO = IOFunctions.IO_Functions()
import PSFFunctions

PSF = PSFFunctions.PSF_Functions()

import sCMOSFunctions

sCMOS = sCMOSFunctions.sCMOS_Functions()

import ImageAnalysisFunctions

I_AF = ImageAnalysisFunctions.Image_Analysis_Functions()


class MultiC_Sim_Funcs:
    def __init__(self, mosaic_unit=None):
        self = self
        return

    def _fit_averager(self, fit_results, n_bootstrap):
        """
        fit_averager; takes fits from demosaicking and averages

        Args:
            fit_results (pd.DataFrame): pandas object of fit results.
            n_bootstrap (int): how many fits to average to

        Returns:
            new_fit_result (pd.DataFrame): pandas object of new fit results.

        """
        xc_toextract = fit_results["xc"].to_numpy()
        yc_toextract = fit_results["yc"].to_numpy()
        sx_toextract = fit_results["s_x"].to_numpy()
        sy_toextract = fit_results["s_y"].to_numpy()
        b_toextract = fit_results["b"].to_numpy()
        A_toextract = fit_results["A"].to_numpy()
        chi_toextract = fit_results["chi_sqr"].to_numpy()

        xc = np.zeros(n_bootstrap)
        yc = np.zeros(n_bootstrap)
        sx = np.zeros(n_bootstrap)
        sy = np.zeros(n_bootstrap)
        A_B = np.zeros(n_bootstrap)
        A_G = np.zeros(n_bootstrap)
        A_R = np.zeros(n_bootstrap)
        bg_B = np.zeros(n_bootstrap)
        bg_G = np.zeros(n_bootstrap)
        bg_R = np.zeros(n_bootstrap)
        chi_sqr = np.zeros(n_bootstrap)
        data_for_pd = {}
        indices = np.arange(0, n_bootstrap * 3, 3)
        for i, index in enumerate(indices[:-1]):
            xc[i] = np.nanmean(xc_toextract[index : indices[i + 1]])
            yc[i] = np.nanmean(yc_toextract[index : indices[i + 1]])
            sx[i] = np.nanmean(sx_toextract[index : indices[i + 1]])
            sy[i] = np.nanmean(sy_toextract[index : indices[i + 1]])
            chi_sqr[i] = np.nanmean(chi_toextract[index : indices[i + 1]])
            A = np.nansum(A_toextract[index : indices[i + 1]])
            b = np.nansum(b_toextract[index : indices[i + 1]])
            bg_B[i] = b_toextract[index] / b
            bg_G[i] = b_toextract[index + 1] / b
            bg_R[i] = b_toextract[index + 2] / b
            A_B[i] = A_toextract[index] / A
            A_G[i] = A_toextract[index + 1] / A
            A_R[i] = A_toextract[index + 2] / A
        data_for_pd["xc"] = xc
        data_for_pd["yc"] = yc
        data_for_pd["s_x"] = sx
        data_for_pd["s_y"] = sy
        data_for_pd["bg_B"] = bg_B
        data_for_pd["bg_G"] = bg_G
        data_for_pd["bg_R"] = bg_R
        data_for_pd["A_B"] = A_B
        data_for_pd["A_G"] = A_G
        data_for_pd["A_R"] = A_R
        data_for_pd["chi_sqr"] = chi_sqr
        data_for_pd["frame"] = np.arange(n_bootstrap)

        new_fit_result = pd.DataFrame(data_for_pd)
        return new_fit_result

    def gen_camera_image_stack(
        self,
        camera_calibration,
        wavelength,
        average_emission_wavelengths,
        dye_pixel_efficiency,
        n_photons,
        x0y0,
        smoothing_function,
        background_photons=0,
        background_colour=[1, 1, 1],
        NA=1.49,
        pixel_size=69,
        return_normal_image=False,
    ):
        """gen_camera_image_stack function
            generates image stack of dyes on bayer filter supplied

        Args:
            camera_calibration (dict): dictionary of gain, offset, variance and rqe. Image size will be derived from this.
            wavelength (np.1darray): wavelengths in same units as object_locs
            average_emission_wavelengths (np.1darray): average emission wavelength of dyes
            dye_pixel_efficiency (np.2darray): 2d array of N dyes * pixel colours
            n_photons (dict): how many photons each dye outputs per object
            x0y0 (dict): object locations per dye, in same units as wavelength
            smootihng_function (type): smoothing function
            background_photons (float): average number of background photons per pixel
            background_colour (np.1darray): array of background QY * pixel colours (default 1,1,1)
            NA (float): numerical aperture of scope in question
            pixel_size (float): pixel size of microscope, default given in nm
            return_normal_image (boolean): if true, returns normal image

        Returns:
            bayer_image (np.ndarray): colour images imaged through the bayer filter supplied
            smoothed_image (np.ndarray): smoothed colour image
            normal_image (np.ndarray, optional): image if no bayer filter applied
        """
        try:
            if len(wavelength) != camera_calibration["pixel_QYs"].shape[1]:
                raise Exception("pixel_QYs not defined at all wavelengths.")
            if len(dye_pixel_efficiency.shape) > 1:
                if len(x0y0.keys()) != dye_pixel_efficiency.shape[0]:
                    raise Exception(
                        "x0y0 dictionary does not contain correct number of localisation arrays."
                    )
            else:
                if len(x0y0.keys()) != 1:
                    raise Exception(
                        "x0y0 dictionary does not contain correct number of localisation arrays."
                    )
        except Exception as error:
            print("Caught this error: " + repr(error))
            return

        dye_names = x0y0.keys()

        gain = camera_calibration["gain"]
        offset = camera_calibration["offset"]
        variance = camera_calibration["variance"]
        relative_QE = camera_calibration["rqe"]

        sigma_x = PSF.sigma_PSF(average_emission_wavelengths, NA)
        sigma_y = sigma_x
        pixel_colours = camera_calibration["pixel_order"]

        if return_normal_image == True:
            overall_QY = np.sum(
                dye_pixel_efficiency, axis=len(dye_pixel_efficiency.shape) - 1
            )

        w = gain.shape[0]
        h = gain.shape[1]
        try:
            s = n_photons[list(dye_names)[0]].shape[0]
        except:
            s = 1

        x = np.linspace(0, (pixel_size * w) - pixel_size, w)

        masks = camera_calibration["masks"]

        abs_QE = np.zeros([w, h, len(dye_names)])
        for j, dye in enumerate(dye_names):
            for i, colour in enumerate(pixel_colours):
                try:
                    dpe = dye_pixel_efficiency[j, i]
                except:
                    try:
                        dpe = dye_pixel_efficiency[i]
                    except:
                        dpe = dye_pixel_efficiency
                abs_QE[:, :, j] += masks[colour] * dpe

        background_photons_perdye = np.divide(background_photons, len(dye_names))
        background_photons_matrix = np.zeros([w, h, len(dye_names)])

        for j, dye in enumerate(dye_names):
            for i, colour in enumerate(pixel_colours):
                try:
                    dpe = dye_pixel_efficiency[j, i]
                except:
                    try:
                        dpe = dye_pixel_efficiency[i]
                    except:
                        dpe = dye_pixel_efficiency
                if dpe != 0:
                    background_photons_matrix[:, :, j] = background_photons_matrix[
                        :, :, j
                    ] + (
                        masks[colour]
                        * (background_colour[i] / dpe)
                        * background_photons_perdye
                    )
                else:
                    background_photons_matrix[:, :, j] = background_photons_matrix[
                        :, :, j
                    ] + (masks[colour] * 0 * background_photons_perdye)

        bayer_image = np.zeros([s, w, h])

        if return_normal_image == True:
            normal_image = np.zeros([s, w, h])

        for frame in np.arange(s):

            n_photons_hitting_detector = np.zeros(
                [gain.shape[0], gain.shape[1], len(dye_names)], dtype=int
            )
            n_photoelectrons = np.zeros_like(n_photons_hitting_detector)

            for j, dye in enumerate(dye_names):
                try:
                    n_photons_this_frame = n_photons[dye][frame]
                except:
                    n_photons_this_frame = n_photons[dye]
                if n_photons_this_frame > 0:
                    try:
                        x0 = x0y0[dye][frame, 0, :]
                        y0 = x0y0[dye][frame, 1, :]
                    except:
                        x0 = x0y0[dye][0, :]
                        y0 = x0y0[dye][1, :]

                    try:
                        sigma_x = sigma_x[j]
                        sigma_y = sigma_y[j]
                    except:
                        sigma_x = sigma_x
                        sigma_y = sigma_y

                    try:
                        abs_QE_fordye = abs_QE[:, :, j]
                    except:
                        abs_QE_fordye = abs_QE

                    photon_spatial_pdf = PSF.gen_spatial_PSF(
                        x,
                        sigma_x,
                        sigma_y,
                        x0,
                        y0,
                        np.array([int(n_photons_this_frame)]),
                        relative_QE,
                    )

                    n_photons_hitting_detector[:, :, j] = (
                        PSF.gen_photons_hitting_detector(
                            photon_spatial_pdf, background_photons_matrix[:, :, j]
                        )
                    )
                    n_photoelectrons[:, :, j] = PSF.gen_photoelectrons(
                        n_photons_hitting_detector[:, :, j], abs_QE_fordye
                    )

            bayer_image[frame, :, :] = PSF.photoelectrons_to_image(
                np.sum(n_photoelectrons, axis=-1), gain, offset, variance
            )

        if return_normal_image == True:
            for frame in np.arange(s):
                for j, dye in enumerate(dye_names):
                    try:
                        n_photons_this_frame = n_photons[dye][frame]
                    except:
                        n_photons_this_frame = n_photons[dye]
                    try:
                        x0 = x0y0[dye][frame, 0, :]
                        y0 = x0y0[dye][frame, 1, :]
                    except:
                        x0 = x0y0[dye][0, :]
                        y0 = x0y0[dye][1, :]

                    try:
                        sigma_x = sigma_x[j]
                        sigma_y = sigma_y[j]
                    except:
                        sigma_x = sigma_x
                        sigma_y = sigma_y

                    try:
                        abs_QE = overall_QY[j]
                    except:
                        abs_QE = overall_QY

                    photon_spatial_pdf = PSF.gen_spatial_PSF(
                        x,
                        sigma_x,
                        sigma_y,
                        x0,
                        y0,
                        np.array([int(n_photons_this_frame)]),
                        relative_QE,
                    )

                    n_photons_hitting_detector[:, :, j] = (
                        PSF.gen_photons_hitting_detector(
                            photon_spatial_pdf, background_photons_perdye
                        )
                    )

                    n_photoelectrons[:, :, j] = PSF.gen_photoelectrons(
                        n_photons_hitting_detector[:, :, j], abs_QE
                    )

                    image_matrix = PSF.photoelectrons_to_image(
                        np.sum(n_photoelectrons, axis=-1), gain, offset, variance
                    )
                    normal_image[frame, :, :] = deepcopy(image_matrix)

        smoothing_args = smoothing_function.args
        smoothing_args[smoothing_function.data_arg] = bayer_image
        smoothed_image = smoothing_function.smoothing_function(**smoothing_args)

        if return_normal_image == True:
            return (
                np.squeeze(bayer_image),
                np.squeeze(smoothed_image),
                np.squeeze(normal_image),
            )
        else:
            return np.squeeze(bayer_image), np.squeeze(smoothed_image), None

    def test_demosaic_fit_method(
        self,
        dye,
        filters,
        wavelength,
        camera_parameters,
        save_folder,
        n_photon_space,
        smoothing_function,
        starting_flag="simulation_",
        n_bootstrap=10000,
        background_photons=5.0,
        NA=1.49,
        pixel_size=69,
        cpu_fraction=0.9,
        single_dye_spectrum=None,
        save_raw_results=False,
        subtractx0y0=False,
    ):
        """test_demosaic_fit_method function
            generates images of a single dye molecule
            and tests fitting on them given method specified

        Args:
            dye (str or np.1darray): dye to test fitting with.
            filters (list): list of filters in the microscope
            wavelength (np.1darray): wavelengths camera detects in same units as object_locs
            camera_parameters (dict): dictionary of gain, offset, variance,
                                    readnoise, rqe, camera masks, pixel_QYs, pixel_order,
                                    and pixel_order_indices.
                                    These sizes must all be the same and will
                                    define simulated image size.
            fitter (python object): function, along with flags, to define fitting.
                                    Should be structured with:
                                        .fit_function
                                        .default_params
                                        .error_type
            save_folder (str): folder to save bootstrapping results
            n_photon_space (np.1darray): range of number of photons
            n_bootstrap (int): number of simulation repetitions
            background_photons (float): average number of background photons per image pixel
            n_photons (dict): how many photons each dye outputs per object
            NA (float): numerical aperture of scope in question
            pixel_size (float): pixel size of microscope, default given in nm
            pixel_error_extent (int): amount of spreading in variance smoothing
        """
        import polars as pl
        import SpectralFunctions

        S_F = SpectralFunctions.Spectral_Funcs()

        try:
            default_cparams = [
                "gain",
                "offset",
                "variance",
                "readnoise",
                "rqe",
                "masks",
                "pixel_QYs",
                "pixel_order",
                "pixel_order_indices",
            ]
            if not all(param in camera_parameters for param in default_cparams):
                raise Exception("camera_parameters missing needed keys.")
            if len(wavelength) != camera_parameters["pixel_QYs"].shape[1]:
                raise Exception("pixel_QYs not defined at all wavelengths.")
        except Exception as error:
            print("Caught this error: " + repr(error))
            return

        default_params = [
            "xc",
            "yc",
            "s_x",
            "s_y",
            "b",
            "A",
            "chi_sqr",
            "frame",
        ]

        analysis_save_params = [
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

        image_size = pixel_size * np.array(camera_parameters["gain"].shape)
        x0 = np.full(n_bootstrap, image_size[0] / 2) + np.random.uniform(
            low=-pixel_size, high=pixel_size, size=n_bootstrap
        )
        y0 = np.full(n_bootstrap, image_size[1] / 2) + np.random.uniform(
            low=-pixel_size, high=pixel_size, size=n_bootstrap
        )
        x0y0 = {}

        x0y0["dye"] = np.zeros([n_bootstrap, 2, 1])
        x0y0["dye"][:, :, :] = np.array([[x0, y0]]).T

        if "simulated_" in dye:
            average_emission_wavelength, dye_pixel_efficiency = (
                S_F.get_pixel_fractions_rawspectra(
                    single_dye_spectrum,
                    wavelength,
                    camera_parameters["pixel_QYs"],
                )
            )
        else:
            average_emission_wavelength, dye_pixel_efficiency = (
                S_F.get_pixel_fractions_dye_and_filters(
                    dye, filters, wavelength, camera_parameters["pixel_QYs"]
                )
            )
        sigma_PSF = PSF.sigma_PSF(average_emission_wavelength, NA)

        fit_RMSE_mean = np.zeros([len(analysis_save_params) - 1, len(n_photon_space)])
        fit_std = np.zeros([len(analysis_save_params) - 1, len(n_photon_space)])

        dye_fit_expectation = dye_pixel_efficiency / np.sum(dye_pixel_efficiency)

        expected_parameters = np.array(
            [
                (image_size[0] / 2),
                (image_size[1] / 2),
                sigma_PSF / pixel_size,
                sigma_PSF / pixel_size,
            ]
        )
        expected_parameters = np.hstack(
            [
                expected_parameters,
                np.array(
                    [
                        background_photons / 3,
                        background_photons / 3,
                        background_photons / 3,
                    ]
                ).ravel(),
            ]
        )
        expected_parameters = np.hstack(
            [expected_parameters, dye_fit_expectation.ravel()]
        )

        parameters_to_save = analysis_save_params[:-2]

        real_params = pl.DataFrame(
            data=np.expand_dims(expected_parameters, 0),
            schema=parameters_to_save,
        )
        dyestr = dye.replace("/", "-")
        real_params.write_csv(
            os.path.join(
                save_folder,
                starting_flag
                + "LM_method"
                + "_"
                + dyestr
                + "_fittesting_input_parameters.csv",
            )
        )
        start = time.time()
        for i, n_photon in enumerate(n_photon_space):
            n_photons = {}
            n_photons["dye"] = np.full(n_bootstrap, n_photon)

            bayer_image, smoothed_image, _ = self.gen_camera_image_stack(
                camera_parameters,
                wavelength,
                average_emission_wavelength,
                dye_pixel_efficiency,
                n_photons,
                x0y0,
                smoothing_function=smoothing_function,
                background_photons=background_photons,
                NA=NA,
                pixel_size=pixel_size,
                return_normal_image=False,
            )

            photoelectron_data = np.divide(
                np.divide(
                    np.subtract(bayer_image, camera_parameters["offset"]),
                    camera_parameters["gain"],
                ),
                camera_parameters["rqe"],
            )

            photoelectron_data = sCMOS.bayer_demosaic_stack(photoelectron_data)

            smoothed_data = np.divide(
                np.divide(
                    np.subtract(smoothed_image, camera_parameters["offset"]),
                    camera_parameters["gain"],
                ),
                camera_parameters["rqe"],
            )

            smoothed_data = sCMOS.bayer_demosaic_stack(smoothed_data)

            def bayer_destacker(RGB_image):
                destacked_image = np.zeros(
                    [RGB_image.shape[0] * 3, RGB_image.shape[1], RGB_image.shape[2]]
                )
                index = 0
                for i in np.arange(RGB_image.shape[0]):
                    for j in np.arange(3):
                        destacked_image[index] = RGB_image[i, :, :, j]
                        index += 1
                return destacked_image

            photoelectron_data = bayer_destacker(photoelectron_data)
            smoothed_data = bayer_destacker(smoothed_data)

            error_data = deepcopy(smoothed_data)
            error_data[error_data < 0] = 0
            error_data = error_data + 1
            error_map = np.add(error_data, np.square(camera_parameters["readnoise"]))
            weights_map = np.power(error_map, -1)

            puncta_tofit = []
            smoothed_puncta_tofit = []
            weights_tofit = []
            relative_coords = []
            planes = []

            for frame in np.arange(n_bootstrap * 3):
                puncta_tofit.append(photoelectron_data[frame, :, :])
                smoothed_puncta_tofit.append(smoothed_data[frame, :, :])
                weights_tofit.append(weights_map[frame, :, :])
                relative_coords.append((0, 0))
                planes.append(frame)

            del photoelectron_data, smoothed_data, weights_map
            gc.collect()

            fit_results, _ = I_AF.fit_nocolour_puncta_parallel(
                puncta_tofit,
                smoothed_puncta_tofit,
                weights_tofit,
                relative_coords,
                planes,
            )
            fit_results = pd.DataFrame(fit_results, columns=default_params)

            fit_results = fit_results.sort_values(by=["frame"])

            fit_results = self._fit_averager(fit_results, n_bootstrap)

            if save_raw_results == True:
                if subtractx0y0 == True:
                    fit_results["xc"] = fit_results["xc"] - (x0 / pixel_size)
                    fit_results["yc"] = fit_results["yc"] - (y0 / pixel_size)

                fit_results.to_csv(
                    os.path.join(
                        save_folder,
                        starting_flag
                        + "LM_method"
                        + "_"
                        + dyestr
                        + "_"
                        + str(np.around(n_photon, 2)).replace(".", "p").zfill(10)
                        + "_fittesting_rawresults.csv",
                    )
                )
            for loc, param in enumerate(analysis_save_params[:-1]):
                if param == "xc":
                    fit_RMSE_mean[loc, i] = pixel_size * np.nanmean(
                        np.sqrt(
                            np.square(
                                (fit_results[param].to_numpy() - (x0 / pixel_size))
                            )
                        )
                    )
                    fit_std[loc, i] = pixel_size * np.nanstd(
                        np.sqrt(
                            np.square(
                                (fit_results[param].to_numpy() - (x0 / pixel_size))
                            )
                        )
                    )
                elif param == "chi_sqr":
                    colour_loc = np.expand_dims(dye_fit_expectation, 0)
                    colour = np.vstack(
                        [
                            fit_results["A_B"].to_numpy(),
                            fit_results["A_G"].to_numpy(),
                            fit_results["A_R"].to_numpy(),
                        ]
                    ).T
                    distances = cdist(colour, colour_loc)
                    fit_RMSE_mean[loc, i] = np.nanmean(distances)
                    fit_std[loc, i] = np.nanstd(distances)
                elif param == "yc":
                    fit_RMSE_mean[loc, i] = pixel_size * np.nanmean(
                        np.sqrt(
                            np.square(
                                (fit_results[param].to_numpy() - (y0 / pixel_size))
                            )
                        )
                    )
                    fit_std[loc, i] = pixel_size * np.nanmean(
                        np.sqrt(
                            np.square(
                                (fit_results[param].to_numpy() - (y0 / pixel_size))
                            )
                        )
                    )
                elif (param == "sx") or (param == "sy"):
                    fit_RMSE_mean[loc, i] = pixel_size * np.nanmean(
                        np.sqrt(
                            np.square(
                                (
                                    fit_results[param].to_numpy()
                                    - expected_parameters[loc]
                                )
                            )
                        )
                    )
                    fit_std[loc, i] = pixel_size * np.nanstd(
                        (fit_results[param].to_numpy())
                    )
                else:
                    fit_RMSE_mean[loc, i] = np.nanmean(
                        np.sqrt(
                            np.square(
                                (
                                    fit_results[param].to_numpy()
                                    - expected_parameters[loc]
                                )
                            )
                        )
                    )
                    fit_std[loc, i] = np.nanstd((fit_results[param].to_numpy()))
            print(
                "Analysed photon flux {}/{}    Time elapsed: {:.3f} min".format(
                    i + 1, len(n_photon_space), (time.time() - start) / 60.0
                ),
                end="\r",
                flush=True,
            )

        save_params = analysis_save_params[:-2]
        save_params.append("colour_distance")
        IO.save_simulation_results(
            save_folder,
            starting_flag,
            save_params,
            n_photon_space,
            fit_RMSE_mean,
            fit_std,
            pixel_size,
            NA,
            background_photons,
            "LM_fitting",
            "Gaussian_Smoother",
            smoothing_function.extent,
            dye,
        )
        return

    def test_fit_method(
        self,
        dye,
        filters,
        wavelength,
        camera_parameters,
        save_folder,
        n_photon_space,
        smoothing_function,
        starting_flag="simulation_",
        n_bootstrap=10000,
        background_photons=5.0,
        background_colour=[1, 1, 1],
        NA=1.49,
        pixel_size=69,
        cpu_fraction=0.9,
        single_dye_spectrum=None,
        save_raw_results=False,
        subtractx0y0=False,
        saverawimages=False
    ):
        """test_single_dye_fit_method function
            generates images of a single dye molecule
            and tests fitting on them given method specified

        Args:
            dye (str or np.1darray): dye to test fitting with.
            filters (list): list of filters in the microscope
            wavelength (np.1darray): wavelengths camera detects in same units as object_locs
            camera_parameters (dict): dictionary of gain, offset, variance,
                                    readnoise, rqe, camera masks, pixel_QYs, pixel_order,
                                    and pixel_order_indices.
                                    These sizes must all be the same and will
                                    define simulated image size.
            fitter (python object): function, along with flags, to define fitting.
                                    Should be structured with:
                                        .fit_function
                                        .default_params
                                        .error_type
            save_folder (str): folder to save bootstrapping results
            n_photon_space (np.1darray): range of number of photons
            n_bootstrap (int): number of simulation repetitions
            background_photons (float): average number of background photons per image pixel
            n_photons (dict): how many photons each dye outputs per object
            NA (float): numerical aperture of scope in question
            pixel_size (float): pixel size of microscope, default given in nm
            pixel_error_extent (int): amount of spreading in variance smoothing
        """
        import polars as pl
        import SpectralFunctions

        S_F = SpectralFunctions.Spectral_Funcs()

        try:
            default_cparams = [
                "gain",
                "offset",
                "variance",
                "readnoise",
                "rqe",
                "masks",
                "pixel_QYs",
                "pixel_order",
                "pixel_order_indices",
            ]
            if not all(param in camera_parameters for param in default_cparams):
                raise Exception("camera_parameters missing needed keys.")
            if len(wavelength) != camera_parameters["pixel_QYs"].shape[1]:
                raise Exception("pixel_QYs not defined at all wavelengths.")
        except Exception as error:
            print("Caught this error: " + repr(error))
            return

        default_params = [
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

        image_size = pixel_size * np.array(camera_parameters["gain"].shape)
        x0 = np.full(n_bootstrap, image_size[0] / 2) + np.random.uniform(
            low=-pixel_size, high=pixel_size, size=n_bootstrap
        )
        y0 = np.full(n_bootstrap, image_size[1] / 2) + np.random.uniform(
            low=-pixel_size, high=pixel_size, size=n_bootstrap
        )
        x0y0 = {}

        x0y0["dye"] = np.zeros([n_bootstrap, 2, 1])
        x0y0["dye"][:, :, :] = np.array([[x0, y0]]).T

        if "simulated_" in dye:
            average_emission_wavelength, dye_pixel_efficiency = (
                S_F.get_pixel_fractions_rawspectra(
                    single_dye_spectrum,
                    wavelength,
                    camera_parameters["pixel_QYs"],
                )
            )
        else:
            average_emission_wavelength, dye_pixel_efficiency = (
                S_F.get_pixel_fractions_dye_and_filters(
                    dye, filters, wavelength, camera_parameters["pixel_QYs"]
                )
            )
        sigma_PSF = PSF.sigma_PSF(average_emission_wavelength, NA)

        fit_RMSE_mean = np.zeros([len(default_params) - 1, len(n_photon_space)])
        fit_std = np.zeros([len(default_params) - 1, len(n_photon_space)])

        dye_fit_expectation = dye_pixel_efficiency / np.sum(dye_pixel_efficiency)

        expected_parameters = np.array(
            [
                (image_size[0] / 2),
                (image_size[1] / 2),
                sigma_PSF / pixel_size,
                sigma_PSF / pixel_size,
            ]
        )
        expected_parameters = np.hstack(
            [
                expected_parameters,
                np.array(
                    [
                        background_photons / 3,
                        background_photons / 3,
                        background_photons / 3,
                    ]
                ).ravel(),
            ]
        )
        expected_parameters = np.hstack(
            [expected_parameters, dye_fit_expectation.ravel()]
        )

        parameters_to_save = default_params[:-2]

        real_params = pl.DataFrame(
            data=np.expand_dims(expected_parameters, 0),
            schema=parameters_to_save,
        )
        dyestr = dye.replace("/", "-")
        real_params.write_csv(
            os.path.join(
                save_folder,
                starting_flag
                + "LM_method"
                + "_"
                + dyestr
                + "_fittesting_input_parameters.csv",
            )
        )
        X0Y0 = {}
        X0Y0['x0'] = x0
        X0Y0['y0'] = y0
        pl.DataFrame(X0Y0).write_csv(os.path.join(
            save_folder,
            starting_flag
            + "LM_method"
            + "_"
            + dyestr
            + "_fittesting_input_groundtruthpositions.csv",
        ))
        
        start = time.time()
        masks_3d = np.dstack(
            [camera_parameters["masks"][x] for x in camera_parameters["masks"].keys()]
        )
        for i, n_photon in enumerate(n_photon_space):
            n_photons = {}
            n_photons["dye"] = np.full(n_bootstrap, n_photon)

            bayer_image, smoothed_image, _ = self.gen_camera_image_stack(
                camera_parameters,
                wavelength,
                average_emission_wavelength,
                dye_pixel_efficiency,
                n_photons,
                x0y0,
                smoothing_function=smoothing_function,
                background_photons=background_photons,
                background_colour=background_colour,
                NA=NA,
                pixel_size=pixel_size,
                return_normal_image=False,
            )
            if saverawimages == True:
                IO.write_tiff(bayer_image, os.path.join(
                    save_folder,
                    starting_flag
                    + "LM_method"
                    + "_"
                    + dyestr
                    + "_"
                    + str(np.around(n_photon, 2)).replace(".", "p").zfill(10)
                    + "_rawbayerimage.tiff",
                ))

            photoelectron_data = np.divide(
                np.divide(
                    np.subtract(bayer_image, camera_parameters["offset"]),
                    camera_parameters["gain"],
                ),
                camera_parameters["rqe"],
            )

            smoothed_data = np.divide(
                np.divide(
                    np.subtract(smoothed_image, camera_parameters["offset"]),
                    camera_parameters["gain"],
                ),
                camera_parameters["rqe"],
            )

            error_data = deepcopy(smoothed_data)
            error_data[error_data < 0] = 0
            error_data = error_data + 1
            error_map = np.add(error_data, np.square(camera_parameters["readnoise"]))
            weights_map = np.power(error_map, -1)

            puncta_tofit = []
            smoothed_puncta_tofit = []
            masks_tofit = []
            weights_tofit = []
            relative_coords = []
            planes = []

            for frame in np.arange(n_bootstrap):
                puncta_tofit.append(photoelectron_data[frame, :, :])
                smoothed_puncta_tofit.append(smoothed_data[frame, :, :])
                masks_tofit.append(masks_3d)
                weights_tofit.append(weights_map[frame, :, :])
                relative_coords.append((0, 0))
                planes.append(frame)

            del photoelectron_data, smoothed_data, weights_map
            gc.collect()

            fit_results, _ = I_AF.fit_puncta_parallel(
                puncta_tofit,
                smoothed_puncta_tofit,
                masks_tofit,
                weights_tofit,
                relative_coords,
                planes,
            )
            fit_results = pd.DataFrame(fit_results, columns=default_params)

            fit_results = fit_results.sort_values(by=["frame"])
            fit_results["photons"] = (
                fit_results["A_R"] + fit_results["A_G"] + fit_results["A_B"]
            )
            for cparam in np.array(["A_R", "A_G", "A_B"]):
                fit_results[cparam] = fit_results[cparam] / fit_results["photons"]
            if save_raw_results == True:
                if subtractx0y0 == True:
                    fit_results["xc"] = fit_results["xc"] - (x0 / pixel_size)
                    fit_results["yc"] = fit_results["yc"] - (y0 / pixel_size)

                fit_results.to_csv(
                    os.path.join(
                        save_folder,
                        starting_flag
                        + "LM_method"
                        + "_"
                        + dyestr
                        + "_"
                        + str(np.around(n_photon, 2)).replace(".", "p").zfill(10)
                        + "_fittesting_rawresults.csv",
                    )
                )
            for loc, param in enumerate(default_params[:-1]):
                if param == "xc":
                    fit_RMSE_mean[loc, i] = pixel_size * np.nanmean(
                        np.sqrt(
                            np.square(
                                (fit_results[param].to_numpy() - (x0 / pixel_size))
                            )
                        )
                    )
                    fit_std[loc, i] = pixel_size * np.nanstd(
                        np.sqrt(
                            np.square(
                                (fit_results[param].to_numpy() - (x0 / pixel_size))
                            )
                        )
                    )
                elif param == "chi_sqr":
                    colour_loc = np.expand_dims(dye_fit_expectation, 0)
                    colour = np.vstack(
                        [
                            fit_results["A_B"].to_numpy(),
                            fit_results["A_G"].to_numpy(),
                            fit_results["A_R"].to_numpy(),
                        ]
                    ).T
                    distances = cdist(colour, colour_loc)
                    fit_RMSE_mean[loc, i] = np.nanmean(distances)
                    fit_std[loc, i] = np.nanstd(distances)
                elif param == "yc":
                    fit_RMSE_mean[loc, i] = pixel_size * np.nanmean(
                        np.sqrt(
                            np.square(
                                (fit_results[param].to_numpy() - (y0 / pixel_size))
                            )
                        )
                    )
                    fit_std[loc, i] = pixel_size * np.nanmean(
                        np.sqrt(
                            np.square(
                                (fit_results[param].to_numpy() - (y0 / pixel_size))
                            )
                        )
                    )
                elif (param == "sx") or (param == "sy"):
                    fit_RMSE_mean[loc, i] = pixel_size * np.nanmean(
                        np.sqrt(
                            np.square(
                                (
                                    fit_results[param].to_numpy()
                                    - expected_parameters[loc]
                                )
                            )
                        )
                    )
                    fit_std[loc, i] = pixel_size * np.nanstd(
                        (fit_results[param].to_numpy())
                    )
                else:
                    fit_RMSE_mean[loc, i] = np.nanmean(
                        np.sqrt(
                            np.square(
                                (
                                    fit_results[param].to_numpy()
                                    - expected_parameters[loc]
                                )
                            )
                        )
                    )
                    fit_std[loc, i] = np.nanstd((fit_results[param].to_numpy()))
            print(
                "Analysed photon flux {}/{}    Time elapsed: {:.3f} min".format(
                    i + 1, len(n_photon_space), (time.time() - start) / 60.0
                ),
                end="\r",
                flush=True,
            )

        save_params = default_params[:-2]
        save_params.append("colour_distance")
        IO.save_simulation_results(
            save_folder,
            starting_flag,
            save_params,
            n_photon_space,
            fit_RMSE_mean,
            fit_std,
            pixel_size,
            NA,
            background_photons,
            "LM_fitting",
            "Gaussian_Smoother",
            smoothing_function.extent,
            dye,
        )
        return
