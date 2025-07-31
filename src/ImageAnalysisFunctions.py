# -*- coding: utf-8 -*-
"""
This class contains functions pertaining to analysis of images,
relating to the bayerSMLM concept.
jsb92, 2024/01/02
"""
import numpy as np
from scipy.optimize import least_squares, leastsq
import os
import sys
from scipy.spatial.distance import cdist
from numba import jit
import multiprocessing
from concurrent import futures
from tqdm import tqdm

module_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(module_dir)
import IOFunctions

IO = IOFunctions.IO_Functions()

import sCMOSFunctions

sCMOS = sCMOSFunctions.sCMOS_Functions()

import PSFFunctions

PSF_F = PSFFunctions.PSF_Functions()

import gaussoptfuncs


class Image_Analysis_Functions:
    def __init__(self):
        self = self
        return

    def WLS_fit_colour_nobounds(
        self,
        data,
        initial_guess,
        masks,
        weights,
        relative_coords,
        display=False,
    ):
        """
        Returns a 2D gaussian colour fit to data.


        Args:
            data (numpy.ndarray): Data to fit.
            initial_guess (numpy.ndarray): Initial guess for the fitter.
            masks (np.3darray): 3d array of colour masks
            weights (np.ndarray): weights for the fitter

        Returns:
            pfit_leastsq (np.ndarray): fit parameters
            perr_leastsq (np.ndarray): error on fit parameters
            (optional) fit (np.2darray): fit
        """
        size = int(data.shape[0])
        ravelsize = int(np.prod(data.shape))

        pfit, pcov, infodict, errmsg, success = leastsq(
            gaussoptfuncs.WLS_chi_nobounds,
            x0=initial_guess,
            args=(data, masks, weights, size, ravelsize),
            full_output=True,
            ftol=1e-2,
            xtol=1e-2,
        )
        if success not in np.array([1, 2, 3, 4]):
            pfit_leastsq = np.full(12, np.nan)
            perr_leastsq = np.full(10, np.nan)
            return pfit_leastsq, perr_leastsq
        chisqr = np.sum(
            np.square(
                gaussoptfuncs.WLS_chi_nobounds(
                    pfit, data, masks, weights, size, ravelsize
                )
            )
        ) / (len(data.ravel()) - len(initial_guess))
        if (len(data.ravel()) > len(initial_guess)) and pcov is not None:
            s_sq = chisqr
            pcov = pcov * s_sq
        else:
            pcov = np.inf

        error = []
        for i in range(len(pfit)):
            try:
                error.append(np.absolute(pcov[i][i]) ** 0.5)
            except:
                error.append(0.00)

        pfit_leastsq = pfit
        pfit_leastsq[:2] = pfit_leastsq[:2] + relative_coords
        pfit_leastsq[-6:] = np.square(pfit_leastsq[-6:])
        pfit_leastsq = np.append(pfit_leastsq, chisqr)
        perr_leastsq = np.array(error)
        if display == False:
            return pfit_leastsq, perr_leastsq
        else:
            size = data.shape
            x = np.arange(size[0])
            background_bayer_matrix = np.empty(ravelsize, dtype=np.float32)
            bayer_matrix = np.empty(ravelsize, dtype=np.float32)
            gauss_2d = np.empty((size, size), dtype=np.float32)
            fit = gaussoptfuncs.WLS_model_nobounds(
                pfit, data, masks, background_bayer_matrix, bayer_matrix, x, gauss_2d
            )
            return pfit_leastsq, perr_leastsq, fit

    def fit_punctum(self, punctum, smoothed_punctum, masks, weights, relative_coords):
        """
        function to fit puncta

        Args:
            puncta (list): list of np.2darray puncta
            puncta (list): list of np.2darray smoothed puncta
            masks (list): list of np.2darray masks
            weights (list): list of np.2darray weights
            pixel_order (np.1darray): pixel order
            relative_coords (list): list of starting coords
            plane (int): plane location

        Returns:
            pfit_leastsq (np.ndarray): fit parameters
            perr_leastsq (np.ndarray): error on fit parameters
        """
        initial_guess = np.empty(10, dtype=np.float32)
        initial_guess[:] = gaussoptfuncs.initial_guess(smoothed_punctum, punctum, masks)
        pfit_leastsq, perr_leastsq = self.WLS_fit_colour_nobounds(
            punctum,
            initial_guess,
            masks,
            weights,
            relative_coords,
        )
        return pfit_leastsq, perr_leastsq

    def fit_puncta(
        self, puncta, smoothed_puncta, masks, weights, relative_coords, planes
    ):
        """
        function to fit puncta

        Args:
            puncta (list): list of np.2darray puncta
            puncta (list): list of np.2darray smoothed puncta
            masks (list): list of np.2darray masks
            weights (list): list of np.2darray weights
            pixel_order (np.1darray): pixel order
            relative_coords (list): list of starting coords
            planes (list): list of planes

        Returns:
            pfit_leastsq (np.ndarray): fit parameters
            perr_leastsq (np.ndarray): error on fit parameters
        """
        pfit_leastsq = np.empty((len(puncta), 12), dtype=np.float32)
        perr_leastsq = np.empty((len(puncta), 10), dtype=np.float32)

        perr_leastsq.fill(np.nan)
        perr_leastsq.fill(np.nan)
        for i, punctum in enumerate(puncta):
            pfit_leastsq[i, :11], perr_leastsq[i] = self.fit_punctum(
                punctum,
                smoothed_puncta[i],
                masks[i],
                weights[i],
                relative_coords[i],
            )
            pfit_leastsq[i, -1] = planes[i]
        return pfit_leastsq, perr_leastsq

    def fit_puncta_parallel(
        self,
        puncta,
        smoothed_puncta,
        masks,
        weights,
        relative_coords,
        planes,
        asynch=False,
    ):
        """
        function to fit puncta in parallel

        Args:
            puncta (list): list of np.2darray puncta
            puncta (list): list of np.2darray smoothed puncta
            masks (list): list of np.2darray masks
            weights (list): list of np.2darray weights
            relative_coords (list): list of starting coords
            planes (list): list of planes

        Returns:
            pfit_leastsq (np.ndarray): fit parameters
            perr_leastsq (np.ndarray): error on fit parameters
        """
        n_workers = min(
            60, max(1, int(0.9 * multiprocessing.cpu_count()))
        )  # Python crashes when using >64 cores
        n_puncta = len(puncta)
        n_tasks = 100 * n_workers
        puncta_per_task = [
            (
                int(n_puncta / n_tasks + 1)
                if _ < n_puncta % n_tasks
                else int(n_puncta / n_tasks)
            )
            for _ in range(n_tasks)
        ]
        start_indices = np.cumsum([0] + puncta_per_task[:-1])
        fs = []
        executor = futures.ProcessPoolExecutor(n_workers)
        for i, n_puncta_task in zip(start_indices, puncta_per_task):
            fs.append(
                executor.submit(
                    self.fit_puncta,
                    puncta[i : i + n_puncta_task],
                    smoothed_puncta[i : i + n_puncta_task],
                    masks[i : i + n_puncta_task],
                    weights[i : i + n_puncta_task],
                    relative_coords[i : i + n_puncta_task],
                    planes[i : i + n_puncta_task],
                )
            )
        if asynch:
            return fs
        with tqdm(desc="LM fitting", total=n_tasks, unit="task") as progress_bar:
            for f in futures.as_completed(fs):
                progress_bar.update()
        return self.fits_from_futures(fs)

    def fits_from_futures(self, fs):
        """
        function to return fits and errors from a futures object

        Args:
            fs (futures object): object

        Returns:
            pfit_leastsq (np.ndarray): fit parameters
            perr_leastsq (np.ndarray): error on fit parameters
        """
        pfit_leastsq = [_.result()[0] for _ in fs]
        perr_leastsq = [_.result()[1] for _ in fs]
        return np.vstack(pfit_leastsq), np.vstack(perr_leastsq)

    @staticmethod
    @jit(nopython=True, nogil=True)
    def gaussian_model(x, xc, yc, A, sigma):
        """
        function to simulate a gaussian psf.

        Args:
            x (np.1darray): x matrix
            xc (float): centroid position in x.
            yc (float): centroid position in y.
            A (float): amplitude.
            sigma (float): width

        Returns:
            data (numpy.ndarray): simulated 2D gaussian.
        """
        array_tofill = np.empty((len(x), len(x)), np.float32)
        return A * gaussoptfuncs.gaussian_unscaled_model(array_tofill, x, xc, yc, sigma)

    # def nilered_model(
    #     self, peak, colours, filterdata, width, skew, wavelength, pixel_QYs
    # ):
    #     """
    #     Returns a colour value wrt a simulation (skew gaussian)

    #     Args:
    #         peak (float): peak position.
    #         colours (np.1darray): fit colours
    #         filterdata (np.1darray): filters in the scope
    #         width (float): width of spectrum
    #         skew (float): skew of spectrum
    #         wavelength (np.1darray): wavlengths to evaluate at
    #         pixel_QYs (np.ndarray): pixel QYs at wavelengths

    #     Returns:
    #         diff (np.ndarray): difference of simulated spectrum in colour space
    #     """
    #     params = [peak, width, skew]
    #     spectrum = np.multiply(S_F.skew_gaussian_model(params, wavelength), filterdata)
    #     spectrum = np.divide(spectrum, np.nansum(spectrum))
    #     average_emission_wavelength, pixel_colours = S_F.get_pixel_fractions_rawspectra(
    #         spectrum, wavelength, pixel_QYs
    #     )
    #     return np.nansum(
    #         np.diag(cdist(np.expand_dims(colours, 0), np.expand_dims(pixel_colours, 0)))
    #     )

    def nilered_minimiser(
        self, colours, nilered_params, intial_guess, minwavelength, maxwavelength
    ):
        """
        Returns a guess of where the colours space corresponds to nile red peak

        Args:
            colours (numpy.ndarray): colours.
            nilered_model (function): nile red peak to colours function
            nilered_params (dict): nile red function args


        Returns:
            res.x (OptimizeResult): peak parameter
        """
        filterdata = nilered_params["filter_data"]
        width = nilered_params["width"]
        skew = nilered_params["skew"]
        wavelength = nilered_params["wavelength"]
        pixel_QYs = nilered_params["pixel_QYs"]
        bounds = (minwavelength, maxwavelength)
        result = least_squares(
            self.nilered_model,
            x0=intial_guess,
            method="trf",
            args=(colours, filterdata, width, skew, wavelength, pixel_QYs),
            bounds=bounds,
        )
        return result.x

    def WLS_nocolour_model(self, params, data):
        """
        Calculate a coloured gaussian based on input params.


        Args:
            params (numpy.ndarray): Input parameters.
            data (numpy.ndarray): Data to fit.

        Returns:
            gauss_2d (numpy.ndarray): 2d bayer filtered gaussian.
        """
        x = np.arange(data.shape[0])
        return (
            self.gaussian_model(x, params[0], params[1], params[2], params[3])
            + params[4]
        )

    def WLS_model(self, params, data, masks):
        """
        Calculate a coloured gaussian based on input params.


        Args:
            params (numpy.ndarray): Input parameters.
            data (numpy.ndarray): Data to fit.
            masks (np.3darray): 3d array of colour masks

        Returns:
            gauss_2d (numpy.ndarray): 2d bayer filtered gaussian.
        """
        truecolour = params[-2:]
        truecolour = np.hstack([truecolour, 1 - np.sum(truecolour)])

        background_photoelectrons = params[4]
        truecolour_bg = params[-4:-2]
        truecolour_bg = np.hstack([truecolour_bg, 1 - np.sum(truecolour_bg)])
        background_bayer_matrix = np.zeros(data.shape)
        bayer_matrix = np.zeros(data.shape)
        for i in np.arange(masks.shape[-1]):
            bayer_matrix[masks[:, :, i]] = truecolour[i]
            background_bayer_matrix[masks[:, :, i]] = (
                background_photoelectrons * truecolour_bg[i]
            )
        x = np.arange(data.shape[0])
        return (
            np.multiply(
                bayer_matrix,
                self.gaussian_model(x, params[0], params[1], params[2], params[3]),
            )
            + background_bayer_matrix
        )

    # def WLS_skewgaussian_chi(
    #     self,
    #     params,
    #     data,
    #     masks,
    #     weights,
    #     wavelength,
    #     filterdata,
    #     pixel_QYs,
    #     summing=False,
    # ):
    #     """
    #     Calculate the chi vector for the weighted least squares model, generating a skew-gaussian spectrum for the spectral guess.

    #     Args:
    #         params (numpy.ndarray): Input parameters.
    #         data (numpy.ndarray): Data to fit.
    #         masks (np.3darray): 3d array of colour masks
    #         weights (np.ndarray): weights for the chi

    #     Returns:
    #         chi (numpy.ndarray): Vector of chi.
    #     """
    #     sg_params = [params[-3], params[-2], params[-1]]
    #     spectrum = np.multiply(
    #         S_F.skew_gaussian_model(sg_params, wavelength), filterdata
    #     )
    #     spectrum = np.divide(spectrum, np.nansum(spectrum))
    #     average_emission_wavelength, pixel_colours = S_F.get_pixel_fractions_rawspectra(
    #         spectrum, wavelength, pixel_QYs
    #     )

    #     gauss_2d = self.WLS_model(params, data, masks)
    #     chi = np.multiply(weights, np.square(np.subtract(data, gauss_2d)))
    #     if summing == False:
    #         return np.sqrt(chi.ravel())
    #     else:
    #         return np.nansum(chi)

    def reduced_chisqared(self, data, fit, weights, initial_guess):
        chi = np.multiply(weights, np.square(np.subtract(data, fit)))
        red_chi = np.nansum(chi) / (len(chi.ravel()) - len(initial_guess))
        return red_chi

    def WLS_chi(self, params, data, masks, weights, summing=False):
        """
        Calculate the chi vector for the weighted least squares model.

        Args:
            params (numpy.ndarray): Input parameters.
            data (numpy.ndarray): Data to fit.
            masks (np.3darray): 3d array of colour masks
            weights (np.ndarray): weights for the chi

        Returns:
            chi (numpy.ndarray): Vector of chi.
        """

        gauss_2d = self.WLS_model(params, data, masks)
        chi = np.multiply(weights, np.square(np.subtract(data, gauss_2d)))
        if summing == False:
            return np.sqrt(chi.ravel())
        else:
            return np.nansum(chi)

    def WLS_chi_nocolour(self, params, data, weights, summing=False):
        """
        Calculate the chi vector for the weighted least squares model.

        Args:
            params (numpy.ndarray): Input parameters.
            data (numpy.ndarray): Data to fit.
            masks (np.3darray): 3d array of colour masks
            weights (np.ndarray): weights for the chi

        Returns:
            chi (numpy.ndarray): Vector of chi.
        """

        gauss_2d = self.WLS_nocolour_model(params, data)
        chi = np.multiply(weights, np.square(np.subtract(data, gauss_2d)))
        if summing == False:
            return np.sqrt(chi.ravel())
        else:
            return np.nansum(chi)

    def WLS_fit(
        self,
        data,
        initial_guess,
        weights,
        display=False,
        minsigma=0.9,
        maxsigma=2.5,
        maxb=np.inf,
    ):
        """
        Returns a 2D gaussian colour fit to data.


        Args:
            data (numpy.ndarray): Data to fit.
            initial_guess (numpy.ndarray): Initial guess for the fitter.
            masks (np.3darray): 3d array of colour masks
            weights (np.ndarray): weights for the fitter

        Returns:
            result (OptimizeResult): fit parameters
        """
        bounds = [
            (0, 0, 0, minsigma, 0),
            (data.shape[0], data.shape[1], np.inf, maxsigma, maxb),
        ]

        result = least_squares(
            self.WLS_chi_nocolour,
            x0=initial_guess,
            method="trf",
            bounds=bounds,
            args=(data, weights, False),
        )

        if display == False:
            return result
        else:
            fit = self.WLS_nocolour_model(result.x, data)
            return result, fit

    def WLS_fit_colour(
        self,
        data,
        initial_guess,
        masks,
        weights,
        pixel_order,
        display=False,
        max_background=np.inf,
        background_Bmin=0,
        background_Gmin=0,
        background_Bmax=0,
        background_Gmax=0,
        Bmin=0,
        Gmin=0,
        Bmax=1,
        Gmax=1,
        minsigma=0.9,
        maxsigma=2.5,
    ):
        """
        Returns a 2D gaussian colour fit to data.


        Args:
            data (numpy.ndarray): Data to fit.
            initial_guess (numpy.ndarray): Initial guess for the fitter.
            masks (np.3darray): 3d array of colour masks
            weights (np.ndarray): weights for the fitter
            pixel_order (np.1darray): how to order the colour masks for the fitter.
                                    set up so it should go in wavelength order
                                    (i.e. B, G, R)
            max_background (float): can encode prior knowledge about the background
            background_Bmin, background_Gmin (float): background B/G ratios
            background_Bmax, background_Gmax (float): background B/G ratios
            Bmin, Gmin (float): can encode prior knowledge about min colour contents expected
            Bmax, Gmax (float): can encode prior knowledge about max colour contents expected


        Returns:
            result (OptimizeResult): fit parameters
        """
        masks = masks[:, :, pixel_order]
        bounds = [
            (0, 0, 0, minsigma, 0, background_Bmin, background_Gmin, Bmin, Gmin),
            (
                data.shape[0],
                data.shape[1],
                np.inf,
                maxsigma,
                max_background,
                background_Bmax,
                background_Gmax,
                Bmax,
                Gmax,
            ),
        ]

        result = least_squares(
            self.WLS_chi,
            x0=initial_guess,
            method="trf",
            bounds=bounds,
            args=(data, masks, weights, False),
        )
        J = result.jac
        try:
            std = np.sqrt(
                np.diagonal(
                    np.linalg.pinv(J.T @ J)
                    * (result.fun.T @ result.fun / (result.fun.size - result.x.size))
                )
            )
        except:
            std = np.full_like(result.x, np.inf)

        if display == False:
            return result, std
        else:
            fit = self.WLS_model(result.x, data, masks)
            return result, std, fit

    def WLS_fit_nilered(
        self,
        data,
        initial_guess,
        masks,
        weights,
        pixel_order,
        nilered_params,
        display=False,
        Bmin=0,
        Gmin=0,
        Rmin=0,
        Bmax=1,
        Gmax=1,
        Rmax=1,
        minsigma=0.9,
        maxsigma=2.5,
        maxb=np.inf,
    ):
        """
        Returns a 2D gaussian colour fit to data.


        Args:
            data (numpy.ndarray): Data to fit.
            initial_guess (numpy.ndarray): Initial guess for the fitter.
            masks (np.3darray): 3d array of colour masks
            weights (np.ndarray): weights for the fitter
            pixel_order (np.1darray): how to order the colour masks for the fitter.
                                    set up so it should go in wavelength order
                                    (i.e. B, G, R)
            Bmin, Gmin, Rmin (float): can encode prior knowledge about min colour contents expected
            Bmax, Gmax, Rmax (float): can encode prior knowledge about max colour contents expected


        Returns:
            result (OptimizeResult): fit parameters
        """
        masks = masks[:, :, pixel_order]
        bounds = [
            (0, 0, 0, minsigma, 0, Bmin, Gmin, Rmin),
            (data.shape[0], data.shape[1], np.inf, maxsigma, maxb, Bmax, Gmax, Rmax),
        ]

        result = least_squares(
            self.WLS_chi,
            x0=initial_guess,
            method="trf",
            bounds=bounds,
            args=(data, masks, weights, False),
        )
        result.x[5:] = result.x[5:] / np.sum(result.x[5:])
        peak_wavelength = self.nilered_minimiser(
            result.x[5:], nilered_params, 600.0, minwavelength=550, maxwavelength=750.0
        )
        result.x = result.x[:5]
        result.x = np.append(result.x, peak_wavelength)
        return result
