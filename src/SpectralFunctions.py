#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 08:59:38 2024

@author: jbeckwith
"""

# -*- coding: utf-8 -*-
"""
This class contains functions pertaining to analysis of images,
relating to the bayerSMLM concept.
jsb92, 2024/01/02
"""
import numpy as np
import os
import sys
import duckdb
import polars as pl
from scipy.optimize import least_squares
from scipy.constants import electron_volt, Planck, c
from scipy.special import erf

module_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(module_dir)
import IOFunctions

# Connect to database
spectra_folder = os.path.join(os.path.split(module_dir)[0], 'Spectra')

IO = IOFunctions.IO_Functions()


class Spectral_Funcs:
    def __init__(self):
        self = self
        conn = duckdb.connect(os.path.join(spectra_folder, 'spectral_data.duckdb'), read_only=True)
        self.dye_names = list(conn.sql("SELECT dye_name FROM dye_summary").df()['dye_name'])
        self.filter_names = list(conn.sql("""SELECT filter_name FROM filter_summary""").df()['filter_name'])
        conn.close()
        return

    @staticmethod
    def getpixelefficiency(
        filename=os.path.abspath("../Spectra/Camera_QE/CS505CU_QE.csv"),
    ):
        """
        Gets pixel efficiency for a camera from a .csv file

        Args:
            filename (str): The name of the JSON file to load.

        Returns:
            R: red pixel QE
            G: green pixel QE
            B: blue pixel QE
        """
        data = pl.read_csv(filename)  # read data
        wavelength_coarse = data["wavelength"].to_numpy()  # read wavelength
        R_coarse = data["R"].to_numpy()  # read red
        G_coarse = data["G"].to_numpy()  # read green
        B_coarse = data["B"].to_numpy()  # read blue
        wavelength = np.arange(np.min(wavelength_coarse), np.max(wavelength_coarse))
        R = np.interp(x=wavelength, xp=wavelength_coarse, fp=R_coarse)
        G = np.interp(x=wavelength, xp=wavelength_coarse, fp=G_coarse)
        B = np.interp(x=wavelength, xp=wavelength_coarse, fp=B_coarse)
        return R, G, B, wavelength

    @staticmethod
    def getobjectiveefficiency(
        wavelength,
        filename=os.path.abspath(
            "../Spectra/Objective_Absorption/Nikon_ApoTIRF_100x.csv"
        ),
    ):
        """
        Gets pixel efficiency for our objective from a .csv file

        Args:
            wavelength (np.1darray): wavlelengths to interpolate at
            filename (str): The name of the JSON file to load.

        Returns:
            T (transmission at wavelength)
        """
        data = pl.read_csv(filename)  # read data
        wavelength_coarse = data["wavelength"].to_numpy()  # read wavelength
        transmission_coarse = np.array(data["transmission"].to_numpy(), dtype=float)
        T = np.interp(wavelength, wavelength_coarse, transmission_coarse)
        return T

    def FHWM_sigma_conversion(self, x, sigma_given=True):
        """
        Converts between FWHM and sigma. If given a True for sigma_given,
        converts to FWHM and vice versa

        Args:
            x (float): parameter to convert
            sigma_given (boolean): if True, converts to FWHM and vice versa

        Returns:
            y (float): converted parameter
        """
        if sigma_given == True:
            return np.multiply(np.multiply(2, np.sqrt(np.multiply(2, np.log(2)))), x)
        else:
            return np.divide(x, 2 * np.sqrt(2 * np.log(2)))

    def moment_calculations(self, x, fx, order=3):
        """
        returns moments. See Bultmann, T. & Ernsting, N. P
                         J. Phys. Chem. 100, 19417–19424 (1996).

        Args:
            x (np.1darray): x data
            fx (np.1darray): fx data
            order (int): order of moments to return

        Returns:
            moments (list): list of moments
        """

        m0 = np.trapz(x=x, y=fx)
        m1 = np.divide(np.trapz(x=x, y=np.multiply(fx, x)), m0)
        m2 = np.sqrt(np.trapz(y=np.multiply(np.power(x - m1, 2), fx), x=x) / m0)
        m3 = np.power(
            np.trapz(y=np.multiply(np.power(x - m1, 3), fx), x=x) / m0, 1.0 / 3
        )
        moments = np.array([m0, m1, m2, m3])
        return moments[:order]

    def spectral_initial_guess(self, spectrum, wavelength, model_length=3):
        """
        returns an initial guess for fitting functions.

        Args:
            spectrum (np.1darray): spectral data
            wavelength (np.1darray): wavelength data
            model_length (int): if 3, returns first 3 moments.
                                for models with skew, can return skew with 4

        Returns:
            initial_guess (list): list of initial guess parameters
        """
        energy = self.wavelength_to_energy(wavelength)
        weighting_factor = np.multiply(np.power(energy, -3), np.square(wavelength))
        spectrum_forguess = np.multiply(spectrum, weighting_factor)
        sort_E = np.argsort(energy)
        energy = energy[sort_E]
        spectrum_forguess = spectrum_forguess[sort_E]
        initial_guess = self.moment_calculations(
            energy, spectrum_forguess, model_length
        )
        return np.nan_to_num(initial_guess)

    def wavelength_to_energy(self, wavelength):
        """
        wavelength_to_energy for fitting spectra.
        Returns energy in units of electron volt.

        Args:
            wavelength (np.1darray): wavelength

        Returns:
            energy (np.1darray): energy in eV
        """
        return np.divide(
            np.divide(np.multiply(Planck, c), np.multiply(wavelength, 1e-9)),
            electron_volt,
        )

    def skew_gaussian_model(self, params, x):
        """
        skew_gaussian_model for fitting spectra.
        See equations 16--18 of Beckwith, J. S., Rumble, C. A. & Vauthey, E.
                                Int. Rev. Phys. Chem. 39, 135–216 (2020).

        Args:
            params (np.1darray): parameters for fit

        Returns:
            skew_gaussian (np.1darray): 1d array of intensities at x
        """
        mu = params[0]
        sigma = params[1]
        alpha = params[2]

        gaussian = self.gaussian_model(params, x)
        skew = 1 + erf(np.multiply(alpha, np.divide(np.subtract(x, mu), sigma)))
        skew_gaussian = np.multiply(gaussian, skew)
        return skew_gaussian

    def gaussian_model(self, params, x):
        """
        gaussian_model for fitting spectra.

        Args:
            params (np.1darray): parameters for fit

        Returns:
            gaussian (np.1darray): 1d array of intensities at x
        """
        mu = params[0]
        sigma = params[1]
        gaussian = np.multiply(
            np.divide(
                1, np.sqrt(np.multiply(2.0, np.multiply(np.pi, np.square(sigma))))
            ),
            np.exp(
                -np.divide(
                    np.square(np.subtract(x, mu)), np.multiply(2.0, np.square(sigma))
                )
            ),
        )
        return gaussian

    def chi2_spectrum(
        self,
        params,
        wavelength,
        spectrum,
        model="gaussian",
        weights=None,
        return_fit=False,
    ):
        """
        generate chi2 for fitting spectra.
        Takes the dipole moment representation (Angulo, G., Grampp, G. &
                                                Rosspeintner, A. Spectrochim.
                                                Acta. A. Mol. Biomol. Spectrosc.
                                                65, 727–731 (2006).)
        into account when fitting the spectra; thus weights fit by nu^-3 then
        by wavelength^-2

        Args:
            params (np.1darray): parameters for fit

        Returns:
            gaussian (np.1darray): 1d array of intensities at x
        """

        energy = self.wavelength_to_energy(wavelength)
        weighting_factor = np.divide(np.power(energy, -3), np.square(wavelength))
        if model == "gaussian":
            spectrum_1d = np.multiply(
                self.gaussian_model(params, energy), weighting_factor
            )
        elif model == "skew-gaussian":
            spectrum_1d = np.multiply(
                self.skew_gaussian_model(params, energy), weighting_factor
            )
        spectrum_1d = np.multiply(
            params[0], np.divide(spectrum_1d, np.trapz(x=wavelength, y=spectrum_1d))
        )
        if weights is None:
            chi = np.subtract(spectrum, spectrum_1d)
        else:
            chi = np.sqrt(
                np.multiply(weights, np.square(np.subtract(spectrum, spectrum_1d)))
            )
        if return_fit == False:
            return chi.ravel()
        else:
            return spectrum_1d

    def spectral_fit_dye(
        self,
        spectrum,
        wavelength,
        initial_guess,
        model="gaussian",
        weights=None,
        display=False,
    ):
        """
        Returns a 2D gaussian colour fit to data.


        Args:
            spectrum (numpy.1darray): Data to fit.
            wavelength (np.1darray): wavelength to fit
            initial_guess (numpy.ndarray): Initial guess for the fitter.
            model (str): model string; takes 'gaussian', 'skew-gaussian', and 'DHO'
            weights (np.ndarray): weights for the fitter
            display (bool): display fit result

        Returns:
            result (OptimizeResult): fit parameters
        """
        if model == "gaussian":
            bounds = [
                (0, 0, 0),
                (np.inf, np.inf, np.inf),
            ]
        elif model == "skew-gaussian":
            bounds = [
                (0, 0, 0, -np.inf),
                (np.inf, np.inf, np.inf, np.inf),
            ]

        result = least_squares(
            self.chi2_spectrum,
            x0=initial_guess,
            method="trf",
            bounds=bounds,
            args=(wavelength, spectrum, model, weights),
        )
        if display == False:
            return result
        else:
            fit = self.chi2_spectrum(
                result.x,
                wavelength,
                spectrum,
                model=model,
                weights=weights,
                return_fit=True,
            )
            return result, fit

    def get_pixel_fractions_rawspectra(self, spectra, wavelength, pixel_QYs):
        """
        Gets dye or filter data for raw spectrum specified.

        Args:
            spectra (np.ndarray): The spectra provided.
            wavelength (np.1darray): wavelength array to read out the dye data at
            pixel_QYs (np.2darray): pixel QYs for camera in question (same length as wavelength)

        Returns:
            average_emission_wavelengths (np.1darray): average emission wavelengths of dyes
            dye_pixel_efficiency (np.ndarray): pixel efficiencies per dye

        """
        average_emission_wavelengths = np.trapz(
            y=wavelength * (spectra.T / np.trapz(x=wavelength, y=spectra)).T,
            x=wavelength,
        )
        dye_pixel_efficiency = np.dot(spectra, pixel_QYs.T)

        return (
            np.squeeze(average_emission_wavelengths),
            np.squeeze(dye_pixel_efficiency),
        )

    def get_pixel_fractions_dye_and_filters(self, dyes, filters, wavelength, pixel_QYs):
        """
        Gets dye or filter data for dye/filter specified.

        Args:
            dyes (list): The name of the dyes.
            filters (list): The name of the filters in the microscope.
            wavelength (np.1darray): wavelength array to read out the dye data at
            pixel_QYs (np.2darray): pixel QYs for camera in question (same length as wavelength)

        Returns:
            average_emission_wavelengths (np.1darray): average emission wavelengths of dyes
            dye_pixel_efficiency (np.ndarray): pixel efficiencies per dye

        """
        if filters == None:
            filter_spectra = np.ones_like(wavelength)
        else:
            filter_spectra = np.prod(
                self.get_dye_or_filter_data(
                    filters, wavelength=wavelength, dye_or_filter=False
                ),
                axis=0,
            )

        dye_spectra = self.get_dye_or_filter_data(dyes, wavelength=wavelength)
        dye_at_detector_spectra = np.array(
            np.multiply(dye_spectra, filter_spectra).T
            / np.sum(np.multiply(dye_spectra, filter_spectra), axis=1)
        ).T
        average_emission_wavelengths = np.trapz(
            y=wavelength
            * (
                dye_at_detector_spectra.T
                / np.trapz(x=wavelength, y=dye_at_detector_spectra)
            ).T,
            x=wavelength,
        )
        dye_pixel_efficiency = np.dot(dye_at_detector_spectra, pixel_QYs.T)

        return (
            np.squeeze(average_emission_wavelengths),
            np.squeeze(dye_pixel_efficiency),
        )

    def get_dye_or_filter_data(self, names, wavelength, dye_or_filter=True):
        """
        Gets dye  or filter data for dye/filter specified.

        Args:
            names (list): The name of the dye in question.
            wavelength (np.1darray): wavelength array to read out the dye data at
            dye_or_filter (boolean): if true, looks up dyes. If false, looks up filters

        Returns:
            spectra (np.ndarray): area-normalised fluorescence spectra or filter spectrum
        """
        conn = duckdb.connect(os.path.join(spectra_folder, 'spectral_data.duckdb'), read_only=True)
        if not isinstance(names, list):
            names = [names]
        try:
            for name in names:
                if dye_or_filter == True:
                    if name not in self.dye_names:
                        raise Exception(str(name)+" dye not in the database.")
                else:
                    if name not in self.filter_names:
                        raise Exception(str(name)+" filter not in the database.")
        except Exception as error:
            conn.close()
            print("Caught this error: " + repr(error))
            return

        spectra = np.zeros([len(names), len(wavelength)])
        if dye_or_filter == True:
            for i, dye_name in enumerate(names):
                try:
                    spectrum = conn.sql("""SELECT * FROM dyes 
                                            WHERE dye_name = '"""+dye_name+"""'
                                            ORDER BY wavelength_nm
                                            """).df()
                    spectrum_wl = spectrum['wavelength_nm'].to_numpy()
                    spectrum_fl = spectrum['emission_intensity'].to_numpy()
                    spectrum_fl[spectrum_fl < 0.0] = 0.0
                    dye_rescaled = np.interp(
                        x=wavelength,
                        xp=spectrum_wl,
                        fp=spectrum_fl,
                        left=0,
                        right=0,
                    )
                    spectra[i, :] = dye_rescaled / np.nansum(dye_rescaled)
                except:
                    continue
        else:
            for i, filter_name in enumerate(names):
                try:
                    spectrum = conn.sql("""SELECT * FROM filters 
                                            WHERE filter_name = '"""+filter_name+"""'
                                            ORDER BY wavelength_nm
                                            """).df()
                    spectrum_wl = spectrum['wavelength_nm'].to_numpy()
                    spectrum_tm = spectrum['transmission_pct'].to_numpy()
                    spectrum_tm[spectrum_tm < 0.0] = 0.0
                    filter_rescaled = np.interp(
                        x=wavelength,
                        xp=spectrum_wl,
                        fp=spectrum_tm,
                        left=0,
                        right=0,
                    )
                    spectra[i, :] = filter_rescaled
                except:
                    continue
        conn.close()
        return spectra
