#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 10:33:06 2024

@author: jbeckwith
"""

import os
import sys
import numpy as np

module_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(module_dir)

import IOFunctions

IO = IOFunctions.IO_Functions()
import PSFFunctions

PSF = PSFFunctions.PSF_Functions()

import sCMOSFunctions

sCMOS = sCMOSFunctions.sCMOS_Functions()

import Multicolour_Simulation_Functions

MSF = Multicolour_Simulation_Functions.MultiC_Sim_Funcs()

import MaskFunctions

Mask = MaskFunctions.Mask_Functions()

import SpectralFunctions

S_F = SpectralFunctions.Spectral_Funcs()


class ToyModel_Functions:
    """Toy model functions for testing and demonstration purposes.

    Provides simplified implementations and test functions for validating
    multicolour SMLM analysis pipelines and algorithms.
    """

    def __init__(self):
        """Initialize ToyModel_Functions class."""
        pass

    def simulate_npixel_ndye_toymodel(
        self,
        save_folder,
        wavelength,
        n_pixelcolours,
        mixing_parameter,
        n_dyes,
        fitter,
        n_photon_space=4000.0,
        dye_FWHM=50.0,
        pixel_size=69,
        pixel_QY=1.0,
        gain=1.0,
        offset=100.0,
        variance=0.0,
        readnoise=0.0,
        rqe=1.0,
        pixel_arrangement="bayer",
        n_bootstrap=1000,
    ):
        """simulate_npixel_ndye_toymodel function
            generates, from wavelengths, pixel colours, pixel_spectra, dyes,
            dye_spectra, pixel_arrangements, a ~numerical judgement of how well
            this hypothetical toy camera arrangement would do. Saves fitting
            etc. in csv files.

        Args:
            wavelength (np.1darray): wavelengths of pixel sensitivities and dyes
            n_pixelcolours (int): how many pixel colours there are
            mixing_parameter (float): how much overlap each of the pixels should have.
                                    Max value is 1, in which case all pixels totally
                                    overlap with their neighbours.
            n_dyes (int): how many dyes to consider
            fitter (python object): function, along with flags, to define fitting.
                                    Should be structured with:
                                        .fit_function
                                        .default_params
                                        .error_type
            dye_FWHM (float): how wide the dye spectra should be, in wavelength units
            pixel_size (float): how big pixels are in nm
            pixel_QY (float): pixel QY when sensitive to a wavelength. Cannot be higher than 1.
            gain (float): flat gain value to be given to simulator
            offset (float): flat offset value to be given to simulator
            variance (float): flat variance value to be given to simulator
            readnoise (float): flat readnoise value to be given to simulator
            rqe (float): flat relative quantum yield value to be given to simulator
            pixel_arrangement (str): how to arrange the N pixels.
                                    Options are: 'bayer'; emulates bayer mask (central pixel colour weighted double)
                                                'R-bayer'; bayer mask but redder and central colours swap places
                                                'B-bayer'; bayer mask but bluer and central colours swap places
                                                'diagonal'; diagonal mask
                                                'random'; mask is totally randomised
                                                NB if any bayer option is given and the number of
                                                pixels has a real square root, then
                                                they will simply be arranged in a square sqrt(N) by sqrt(N)
            n_bootstrap (int): how many times to test the model
        """
        try:
            default_paparams = [
                "bayer",
                "R-bayer",
                "B-bayer",
                "diagonal",
            ]
            if pixel_arrangement not in default_paparams:
                raise Exception(
                    "Incorrect pixel arrangement selected; option not available."
                )
        except Exception as error:
            print("Caught this error: " + repr(error))
            return

        pixel_QY = min(pixel_QY, 1.0)
        image_size = n_pixelcolours * 10

        def sorted_from_middle(lst, reverse=False):
            if len(lst) <= 1:
                return lst
            tail = sorted([lst[-1], lst[0]], reverse=reverse)
            return sorted_from_middle(lst[1:-1], reverse) + tail

        if pixel_arrangement == "diagonal":
            masks = Mask.get_masks(
                Mask.return_diagonal_patterns(np.arange(n_pixelcolours), image_size),
                image_size,
                image_size,
            )
        else:  # Handle all bayer patterns
            color_arrangements = {
                "bayer": lambda x: sorted_from_middle(list(np.arange(x))),
                "R-bayer": lambda x: np.arange(x),
                "B-bayer": lambda x: np.arange(x)[::-1],
            }
            colour_arrangement = color_arrangements[pixel_arrangement](n_pixelcolours)
            masks = Mask.get_masks(
                Mask.return_custom_bayer_patterns(colour_arrangement),
                image_size,
                image_size,
            )

        # generate pixel spectra
        pixel_spectra = np.zeros([n_pixelcolours, len(wavelength)])
        pixel_cutoff_wavelengths = np.linspace(
            np.min(wavelength), np.max(wavelength), n_pixelcolours + 1
        )
        mix_shift = 0.5 * (np.diff(pixel_cutoff_wavelengths)[0] * mixing_parameter)
        for c in np.arange(n_pixelcolours):
            minloc = np.argmin(
                np.abs(wavelength - (pixel_cutoff_wavelengths[c] - mix_shift))
            )
            maxloc = np.argmin(
                np.abs(wavelength - (pixel_cutoff_wavelengths[c + 1] + mix_shift))
            )
            pixel_spectra[c, minloc:maxloc] = pixel_QY

        dye_centers = np.linspace(np.min(wavelength), np.max(wavelength), n_dyes + 2)[
            1:-1
        ]

        dye_spectra = np.zeros([n_dyes, len(wavelength)])
        sigma = S_F.FHWM_sigma_conversion(dye_FWHM, False)
        for d in np.arange(n_dyes):
            params = (
                1,
                dye_centers[d],
                sigma,
            )
            dye_spectra[d, :] = S_F.gaussian_model(params, wavelength)
            dye_spectra[d, :] /= np.sum(dye_spectra[d, :])

        # Set camera parameters
        camera_parameters = {
            param: np.full((image_size, image_size), value)
            for param, value in {
                "gain": gain,
                "offset": offset,
                "variance": variance,
                "readnoise": readnoise,
                "rqe": rqe,
            }.items()
        }
        camera_parameters.update(
            {
                "masks": masks,
                "pixel_QYs": pixel_spectra,
                "pixel_order": np.arange(n_pixelcolours),
                "pixel_order_indices": np.arange(n_pixelcolours),
            }
        )

        n_photon_space = (
            np.array([n_photon_space])
            if isinstance(n_photon_space, float)
            else n_photon_space
        )
        starting_flag = str(
            "mixing_parameter"
            + str(np.around(mixing_parameter, 2)).replace(".", "p")
            + "_n_dyes_"
            + str(n_dyes)
            + "_n_pixelcolours_"
            + str(n_pixelcolours)
            + "_"
            + "fitter_",
        )
        for i, centre in enumerate(dye_centers):
            dye = (
                "simulated_gaussian_mu_"
                + str(np.around(centre, 1)).replace(".", "p")
                + "_FWHM_"
                + str(np.around(dye_FWHM, 1))
            )
            fitter.single_dye_spectrum = dye_spectra[i, :]
            fitter.dye_library = dye_spectra
            MSF.test_single_dye_fit_method(
                dye,
                [],
                wavelength,
                camera_parameters,
                fitter,
                save_folder,
                n_photon_space,
                starting_flag,
                n_bootstrap=n_bootstrap,
                background_photons=0.0,
                NA=1.49,
                pixel_size=pixel_size,
                cpu_fraction=1,
            )
        return
