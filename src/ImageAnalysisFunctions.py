# -*- coding: utf-8 -*-
"""
This class contains functions pertaining to analysis of images,
relating to the bayerSMLM concept.
jsb92, 2024/01/02
"""
import numpy as np
from scipy.optimize import leastsq
import os
import sys
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

    def fit_nocolour_puncta_parallel(
        self,
        puncta,
        smoothed_puncta,
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
                    self.fit_nocolour_puncta,
                    puncta[i : i + n_puncta_task],
                    smoothed_puncta[i : i + n_puncta_task],
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

    def fit_nocolour_puncta(
        self, puncta, smoothed_puncta, weights, relative_coords, planes
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
        pfit_leastsq = np.empty((len(puncta), 8), dtype=np.float32)
        perr_leastsq = np.empty((len(puncta), 6), dtype=np.float32)

        perr_leastsq.fill(np.nan)
        perr_leastsq.fill(np.nan)
        for i, punctum in enumerate(puncta):
            pfit_leastsq[i, :7], perr_leastsq[i] = self.fit_nocolour_punctum(
                punctum,
                smoothed_puncta[i],
                weights[i],
                relative_coords[i],
            )
            pfit_leastsq[i, -1] = planes[i]
        return pfit_leastsq, perr_leastsq

    def fit_nocolour_punctum(self, punctum, smoothed_punctum, weights, relative_coords):
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
        initial_guess = np.empty(6, dtype=np.float32)
        initial_guess[:] = gaussoptfuncs.initial_nocolour_guess(
            smoothed_punctum, punctum
        )
        pfit_leastsq, perr_leastsq = self.WLS_fit_nocolour_nobounds(
            punctum,
            initial_guess,
            weights,
            relative_coords,
        )
        return pfit_leastsq, perr_leastsq

    def WLS_fit_nocolour_nobounds(
        self,
        data,
        initial_guess,
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
            gaussoptfuncs.WLS_chi_nocolour_nobounds,
            x0=initial_guess,
            args=(data, weights, size, ravelsize),
            full_output=True,
            ftol=1e-2,
            xtol=1e-2,
        )
        if success not in np.array([1, 2, 3, 4]):
            pfit_leastsq = np.full(7, np.nan)
            perr_leastsq = np.full(5, np.nan)
            return pfit_leastsq, perr_leastsq
        chisqr = np.sum(
            np.square(
                gaussoptfuncs.WLS_chi_nocolour_nobounds(
                    pfit, data, weights, size, ravelsize
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
        pfit_leastsq[-2:] = np.square(pfit_leastsq[-2:])
        pfit_leastsq = np.append(pfit_leastsq, chisqr)
        perr_leastsq = np.array(error)
        if display == False:
            return pfit_leastsq, perr_leastsq
        else:
            size = data.shape
            x = np.arange(size[0])
            gauss_2d = np.empty((size, size), dtype=np.float32)
            fit = gaussoptfuncs.WLS_nocolour_model_nobounds(pfit, data, x, gauss_2d)
            return pfit_leastsq, perr_leastsq, fit
