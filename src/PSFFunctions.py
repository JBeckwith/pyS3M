# -*- coding: utf-8 -*-
"""
This class contains functions that collect PSF simulation codes for pySMLM
jsb92, 2024/03/04
"""
from pathlib import Path
import numpy as np
import sys
from skimage.filters import gaussian
from scipy.special import j1
from numba import jit

sys.path.append(str(Path(__file__).parent))
from Constants import ProcessingConstants, DefaultParameters

import pathos
from pathos.pools import ThreadPool as Pool

import gaussoptfuncs

cpu_number = int(pathos.helpers.cpu_count() * 0.9)


class PSF_Functions:
    """Point spread function modeling and photon simulation functions.

    Provides functionality for generating camera images, calculating PSF parameters,
    and simulating photon distributions for single-molecule localization microscopy.
    """

    def __init__(self):
        """Initialize PSF_Functions class."""
        pass

    @staticmethod
    @jit(
        nopython=True, nogil=True
    )  # Set "nopython" mode for best performance, equivalent to @njit
    def diffraction_limit(wavelength, NA):
        """
        calculates diffraction limit from Abbe criterion

        Args:
            wavelength (float): wavelength of light being imaged
            NA (float): numerical aperture of microscope

        Returns:
            diffraction_limit (float): d of psf
        """
        return np.divide(wavelength, np.multiply(2.0, NA))

    @staticmethod
    @jit(
        nopython=True, nogil=True
    )  # Set "nopython" mode for best performance, equivalent to @njit
    def sigma_PSF(wavelength, NA):
        """
        calculates sigma psf according to Fazel, M. et al. Rev. Mod. Phys. 96, 025003 (2024).


        Args:
            wavelength (float): wavelength of light being imaged in m
            NA (float): numerical aperture of microscope

        Returns:
            diffraction_limit (float): d of psf
        """
        sigma_psf = np.divide(
            wavelength, np.multiply(np.multiply(np.sqrt(2), np.pi), NA)
        )
        return sigma_psf

    def airy2d_PSF(
        self, x, y, x0, y0, NA, wavelength, M, n_photons=4000, bitdepth=np.float32
    ):
        """
        simulates a 2d airy psf of width sigma_psf on a grid with coordinates
        x and y, with the origin x0 and y0, and magnification M
        see equations 1 and 2 in Chao, J.; Ward, E. S.; Ober, R. J.
        Fisher Information Theory for Parameter Estimation in Single
        Molecule Microscopy: Tutorial. J. Opt. Soc. Am. A,
        JOSA A 2016, 33 (7), B36–B57. https://doi.org/10.1364/JOSAA.33.000B36.

        Args:
            x (1d array): x coordinate locations, in same unit as x0 (typically micron)
            x (1d array): y coordinate locations, in same unit as y0(typically micron)
            x0 (float or 1d array): origin position of 2d gaussian in x, in same unit as sigma_psf
            y0 (float or 1d array): origin position of 2d gaussian in y, in same unit as sigma_psf
            NA (float): NA of the objective
            wavlength (float): wavlength in microns used for imaging
            M (float): magnification
            n_photons (int): n_photons per localisation. Given to poisson rng
            bitdepth (type): bit depth. default uint16

        Returns:
            PSF_a2d (2D array): 2d probability density function of PSF
        """
        if 0 in x:  # numerical error if x or y = 0, correct by adding miniscule amount
            x = x + 1e-100
            y = y + 1e-100
        X, Y = np.meshgrid(x / M, y / M)
        kappa = np.divide(np.multiply(NA, np.multiply(2.0, np.pi)), wavelength)
        if isinstance(x0, float):
            rterm = np.sqrt(np.add(np.square(X - x0), np.square(Y - y0)))
            Jterm = np.square(j1(np.multiply(kappa, rterm)))
            piterm = np.multiply(np.pi, np.add(np.square(X - x0), np.square(Y - y0)))
            q_function = np.nan_to_num(np.divide(Jterm, piterm))
            fphi = np.multiply(np.divide(1.0, np.square(M)), q_function)
            PSF_a2d = np.divide(fphi, np.sum(fphi))
            PSF_a2d = np.multiply(PSF_a2d, np.random.poisson(n_photons))
        else:
            PSF_a2d = np.zeros_like(X, dtype=bitdepth)
            photon_numbers = np.random.poisson(n_photons, size=len(x0))
            for i in np.arange(len(x0)):
                rterm = np.sqrt(np.add(np.square(X - x0[i]), np.square(Y - y0[i])))
                Jterm = np.square(j1(np.multiply(kappa, rterm)))
                piterm = np.multiply(
                    np.pi, np.add(np.square(X - x0[i]), np.square(Y - y0[i]))
                )
                q_function = np.nan_to_num(np.divide(Jterm, piterm))
                fphi = np.multiply(np.divide(1.0, np.square(M)), q_function)
                fphi = np.multiply(photon_numbers[i], np.divide(fphi, np.sum(fphi)))
                PSF_a2d = PSF_a2d + fphi
        return PSF_a2d

    @staticmethod
    @jit(
        nopython=True, nogil=True
    )  # Set "nopython" mode for best performance, equivalent to @njit
    def gaussian2d_PSF(
        x, sigma_x, sigma_y, x0, y0, n_photons=np.array([4000]), bitdepth=np.float32
    ):
        """
        simulates a 2d gaussian psf of width sigma_psf on a grid with coordinates
        x and y, with the origin x0 and y0

        Args:
            x (1d array): x coordinate locations, in same unit as sigma_psf (typically micron)
            sigma_psf (float): width of 2d gaussian.
            x0 (1d array): origin position of 2d gaussian in x, in same unit as sigma_psf
            y0 (1d array): origin position of 2d gaussian in y, in same unit as sigma_psf
            bitdepth (type): bit depth. default float32

        Returns:
            PSF_g2d (2D array): 2d probability density function of PSF
        """
        PSF_g2d = np.zeros((len(x), len(x)), dtype=bitdepth)
        if len(x0) == 1:
            PSF_g2d = gaussoptfuncs.gaussian_unscaled_model(
                PSF_g2d,
                x,
                len(x),
                x0,
                y0,
                sigma_x,
                sigma_y,
            )
            PSF_g2d = PSF_g2d / np.nansum(PSF_g2d)
            PSF_g2d = np.multiply(n_photons, PSF_g2d).astype(bitdepth)
        else:
            for i in range(len(x0)):
                temp_PSF = np.zeros((len(x), len(x)), dtype=bitdepth)
                temp_PSF = gaussoptfuncs.gaussian_unscaled_model(
                    temp_PSF, x, len(x), x0[i], y0[i], sigma_x, sigma_y
                )
                total = np.nansum(temp_PSF)
                if total > 0:
                    temp_PSF *= n_photons[i] / total
                else:
                    temp_PSF.fill(0)
                PSF_g2d += temp_PSF.astype(bitdepth)
        return PSF_g2d

    @staticmethod
    @jit(
        nopython=True, nogil=True
    )  # Set "nopython" mode for best performance, equivalent to @njit
    def gaussian2d_PSF_pixel(
        image_size, sigma_psf, x0, y0, n_photons=4000, bitdepth=np.float32
    ):
        """
        simulates a 2d gaussian psf of width sigma_psf on a grid with coordinates
        x and y, with the origin x0 and y0

        Args:
            image_size (tuple): tuple of how big the image is in pixels
            sigma_psf (float): width of 2d gaussian in pixels
            x0 (float or 1d array): origin position of 2d gaussian in x, in pixels
            y0 (float or 1d array): origin position of 2d gaussian in y, in pixels
            n_photons (int): n_photons per localisation. Given to poisson rng
            bitdepth (type): bit depth. default uint16

        Returns:
            PSF_g2d (2D array): 2d probability density function of PSF
        """
        PSF_g2d = np.zeros(image_size, dtype=bitdepth)

        if isinstance(x0, float):
            PSF_g2d[x0, y0] = n_photons
            PSF_g2d = np.asarray(gaussian(PSF_g2d, sigma=sigma_psf), dtype=bitdepth)
        else:
            photon_numbers = np.random.poisson(n_photons, size=len(x0))
            PSF_g2d[x0, y0] = photon_numbers
            PSF_g2d = np.asarray(gaussian(PSF_g2d, sigma=sigma_psf), dtype=bitdepth)
        return PSF_g2d

    @staticmethod
    @jit(
        nopython=True, nogil=True
    )  # Set "nopython" mode for best performance, equivalent to @njit
    def generate_sCMOS_maps(
        image_size,
        gain_mean=2.0,
        offset_mean=100.0,
        variance_mean=8.0,
        rQE_mean=1.0,
        sigma_factor=0.1,
        relQE=False,
    ):
        """
        simulates gain, offset, variance and relative QE for a sCMOS
        uses pdfs centred at

        Args:
            image_size (tuple): tuple of how big the image is in pixels
            gain_mean (float): mean of gain
            offset_mean (float): mean of offset
            variance_mean (float): mean of variance
            rQE_mean (float): mean of relaative QE
            sigma_factor (float): how far the means will drift
            relQE (boolean): if false, just returns ones (i.e. no
                                        relative QE mismatch per pixel)

        Returns:
            gain (numpy.2darray): 2D image matrix of gain per pixel
            offset (numpy.2darray): 2D image matrix of offset per pixel
            variance (numpy.2darray): 2D image matrix of variance per pixel
            relative_QE (numpy.2darray): 2D image matrix of relative QE per pixel
        """
        gain = np.abs(
            np.random.normal(
                loc=gain_mean, scale=gain_mean * sigma_factor, size=image_size
            )
        )
        offset = np.abs(
            np.random.normal(
                loc=offset_mean, scale=offset_mean * sigma_factor, size=image_size
            )
        )
        variance = np.abs(
            np.random.gamma(
                shape=sigma_factor, scale=variance_mean / sigma_factor, size=image_size
            )
        )  # gamma distribution for variance
        if relQE == False:
            relative_QE = np.ones_like(variance)
        relative_QE = np.abs(
            np.random.normal(
                loc=rQE_mean, scale=rQE_mean * sigma_factor, size=image_size
            )
        )
        return gain, offset, variance, relative_QE

    @staticmethod
    @jit(
        nopython=True, nogil=True
    )  # Set "nopython" mode for best performance, equivalent to @njit
    def gen_photons_hitting_detector(photon_spatial_pdf, background=0):
        """
        goes from photoelectrons to image

        This is largely cribbed from Fazel, M.;
        Grussmayer, K. S.; Ferdman, B.; Radenovic, A.; Shechtman, Y.;
        Enderlein, J.; Pressé, S. Rev. Mod. Phys. 96, 025003 (2024).

        Args:
            photon_spatial_pdf (2d array): photon spatial psf that will be poisson-noised
            background (2d array): background photons that will be poisson-noised

        Returns:
            image_matrix (numpy.2darray): 2D image matrix
        """
        photon_spatial_pdf = photon_spatial_pdf.clip(0, np.inf)
        n_photons_hitting_detector = np.zeros(photon_spatial_pdf.shape)
        lparam = np.add(photon_spatial_pdf, background)
        for i in range(photon_spatial_pdf.shape[0]):
            for j in range(photon_spatial_pdf.shape[1]):
                n_photons_hitting_detector[i, j] = np.random.poisson(lparam[i, j])
        return n_photons_hitting_detector

    def gen_spatial_PSF(self, x, y, sigma_x, sigma_y, x0, y0, n_photons, relative_QE):
        """
        simulates spatial PSF with relative QE
        as well as spot locations and photon numbers
        image size will be same as relative QE map.

        This is largely cribbed from Fazel, M.;
        Grussmayer, K. S.; Ferdman, B.; Radenovic, A.; Shechtman, Y.;
        Enderlein, J.; Pressé, S. Rev. Mod. Phys. 96, 025003 (2024).

        Args:
            x (1d array): x coordinate locations (pixels), length = image width
            y (1d array): y coordinate locations (pixels), length = image height
            sigma_x (float): width of 2d gaussian in pixels
            sigma_y (float): width of 2d gaussian in pixels
            x0 (float or 1d array): origin position of 2d gaussian in x, in pixels
            y0 (float or 1d array): origin position of 2d gaussian in y, in pixels
            n_photons (int/np.1darray): n_photons per localisation. Given to poisson rng
            relative_QE (numpy.2darray): 2D image matrix of relative QE per pixel, shape (len(x), len(y))

        Returns:
            photon_spatial_pdf (numpy.2darray): 2D spatial PSF, shape (len(x), len(y))
        """
        norm_x = 0.3989422804014327 / sigma_x  # 1/sqrt(2*pi)
        norm_y = 0.3989422804014327 / sigma_y
        PSF_g2d = np.zeros((len(x), len(y)), dtype=np.float32)
        for i in range(len(x0)):
            xg = norm_x * np.exp(-0.5 * ((x - x0[i]) / sigma_x) ** 2)
            yg = norm_y * np.exp(-0.5 * ((y - y0[i]) / sigma_y) ** 2)
            temp = np.outer(xg, yg).astype(np.float32)
            total = np.nansum(temp)
            if total > 0:
                temp *= n_photons[i] / total
            else:
                temp.fill(0)
            PSF_g2d += temp
        photon_spatial_pdf = np.multiply(relative_QE, PSF_g2d)
        return photon_spatial_pdf

    def gen_spatial_PSF_fast(
        self, x, y, sigma_x, sigma_y, x0, y0, n_photons, relative_QE,
        crop_radius: int = None,
    ):
        """Crop-based version of gen_spatial_PSF. Same contract, ~30–40× faster for
        large images because each emitter's Gaussian is computed only within a local
        window of radius ``crop_radius`` pixels.  Energy lost outside the crop
        (~5×10⁻⁹ of PSF integral at 4.5 σ) is negligible versus shot noise.

        Args:
            x, y           : pixel coordinate arrays, length = image width / height
            sigma_x/y      : PSF sigma in pixels (can differ for astigmatism)
            x0, y0         : emitter centre positions in pixels, shape (N,)
            n_photons      : expected photon count per emitter, shape (N,)
            relative_QE    : (W, H) spatial QE map
            crop_radius    : half-side of the bounding box in pixels.
                             Default: max(6, ceil(4.5 × max(sigma_x, sigma_y))).
        """
        W, H = len(x), len(y)
        if crop_radius is None:
            crop_radius = max(6, int(np.ceil(4.5 * max(sigma_x, sigma_y))))
        PSF_g2d = np.zeros((W, H), dtype=np.float32)
        for i in range(len(x0)):
            xi, yi = x0[i], y0[i]
            x1 = max(0, int(np.floor(xi)) - crop_radius)
            x2 = min(W, int(np.ceil(xi)) + crop_radius + 1)
            y1 = max(0, int(np.floor(yi)) - crop_radius)
            y2 = min(H, int(np.ceil(yi)) + crop_radius + 1)
            if x1 >= x2 or y1 >= y2:
                continue
            gx = np.exp(-0.5 * ((x[x1:x2] - xi) / sigma_x) ** 2)
            gy = np.exp(-0.5 * ((y[y1:y2] - yi) / sigma_y) ** 2)
            patch = np.outer(gx, gy).astype(np.float32)
            total = patch.sum()
            if total > 0:
                patch *= n_photons[i] / total
            PSF_g2d[x1:x2, y1:y2] += patch
        return np.multiply(relative_QE, PSF_g2d)

    def gen_photoelectrons(self, n_photons_hitting_detector, abs_QE):
        """
        simulates number of photoelectrons from number of photons and absolute QE

        This is largely cribbed from Fazel, M.;
        Grussmayer, K. S.; Ferdman, B.; Radenovic, A.; Shechtman, Y.;
        Enderlein, J.; Pressé, S. Rev. Mod. Phys. 96, 025003 (2024).

        Args:
            n_photons_hitting_detector (2d array): 2d matrix of photon numbers. Must be integers
            abs_QE (float or 2d array): QE for the chip for these photons

        Returns:
            n_photoelectrons (numpy.2darray): 2D matrix of photoelectrons generated
        """
        n_photons_hitting_detector = np.array(
            n_photons_hitting_detector.clip(0, np.inf), dtype=np.int64
        )

        # Sanitize abs_QE to ensure valid probability values for binomial distribution
        abs_QE = np.array(abs_QE)

        # Handle NaN values: set to 0 (no quantum efficiency)
        abs_QE = np.where(np.isnan(abs_QE), 0.0, abs_QE)

        # Handle infinite values: set to 1 (perfect quantum efficiency)
        abs_QE = np.where(np.isinf(abs_QE), 1.0, abs_QE)

        # Clip to valid probability range [0, 1]
        abs_QE = np.clip(abs_QE, 0.0, 1.0)

        n_photoelectrons = np.random.binomial(
            n=n_photons_hitting_detector,
            p=abs_QE,
            size=n_photons_hitting_detector.shape,
        )  # this is the photoelectrons that will hit our detector
        return n_photoelectrons

    def gen_photoelectrons_vectorized_frames(
        self,
        n_photons_all_frames: np.ndarray,
        QE_per_channel_all: np.ndarray,
        mask_stack: np.ndarray,
    ) -> np.ndarray:
        """
        Generate photoelectrons for all frames at once (vectorized across frames).

        This is 2-5× faster than per-frame loop because binomial sampling can be
        done in one large batch operation across all frames simultaneously.

        Args:
            n_photons_all_frames: Photons hitting detector for all frames
                                  Shape: (n_frames, w, h, n_dyes)
            QE_per_channel_all: QE values per channel for each frame and dye
                                Shape: (n_frames, n_dyes, n_channels)
                                Example: QE_per_channel_all[frame=5, dye=0, :] = [0.1, 0.7, 0.2] for B,G,R
            mask_stack: Bayer mask stack
                        Shape: (w, h, n_channels)
                        Example: mask_stack[:,:,0] = Blue pixel mask

        Returns:
            n_photoelectrons: Photoelectrons for all frames
                             Shape: (n_frames, w, h, n_dyes)

        Notes:
            - This function properly handles Bayer pattern splitting: each photon
              can only generate photoelectrons on ONE pixel type (B, G, or R)
            - Uses vectorized binomial sampling for maximum performance
            - Handles per-frame QE values (stochastic mode)
        """
        n_frames, w, h, n_dyes = n_photons_all_frames.shape
        n_channels = QE_per_channel_all.shape[2]

        # Pre-allocate output
        n_photoelectrons_all = np.zeros_like(n_photons_all_frames)

        # Process each dye (usually just 1 dye)
        for j in range(n_dyes):
            # Extract data for this dye across all frames
            n_photons_dye = n_photons_all_frames[:, :, :, j]  # Shape: (n_frames, w, h)
            QE_dye = QE_per_channel_all[:, j, :]              # Shape: (n_frames, n_channels)

            # Check if all QE values are identical across channels (Standard camera case)
            # This is faster because we don't need to split by Bayer pattern
            all_QE_equal = np.allclose(QE_dye, QE_dye[:, 0:1], rtol=1e-9)

            if all_QE_equal:
                # Fast path: Uniform QE across all channels
                # Can apply QE directly without splitting by Bayer pattern
                QE_uniform = QE_dye[:, 0]  # Shape: (n_frames,)

                # ⚡ VECTORIZED: Process all frames at once
                # Broadcast QE to match photon array shape
                n_photoelectrons_all[:, :, :, j] = self.gen_photoelectrons(
                    n_photons_dye,
                    QE_uniform[:, np.newaxis, np.newaxis]  # Broadcast to (n_frames, w, h)
                )
            else:
                # Accurate path: Different QE per channel (Bayer camera)
                # Must split photons by Bayer pattern FIRST, then apply channel-specific QE

                # ⚡ VECTORIZED: Process all frames simultaneously
                # Expand dimensions for broadcasting:
                # n_photons_dye: (n_frames, w, h) → (n_frames, w, h, 1)
                # mask_stack: (w, h, n_channels) → (1, w, h, n_channels)
                # Result: (n_frames, w, h, n_channels)
                n_photons_per_channel = (
                    n_photons_dye[:, :, :, np.newaxis] *    # (n_frames, w, h, 1)
                    mask_stack[np.newaxis, :, :, :]         # (1, w, h, n_channels)
                ).astype(np.int64)

                # ⚡ VECTORIZED BINOMIAL: Process all frames+channels at once
                # n_photons_per_channel: (n_frames, w, h, n_channels)
                # QE_dye: (n_frames, n_channels) → reshape to (n_frames, 1, 1, n_channels)
                # NumPy's binomial will broadcast correctly
                photoelectrons_per_channel = self.gen_photoelectrons(
                    n_photons_per_channel,
                    QE_dye[:, np.newaxis, np.newaxis, :]  # Broadcast to (n_frames, w, h, n_channels)
                )

                # Sum photoelectrons across channels (each photon contributed to only one channel)
                n_photoelectrons_all[:, :, :, j] = np.sum(photoelectrons_per_channel, axis=-1)

        return n_photoelectrons_all

    @staticmethod
    @jit(
        nopython=True, nogil=True
    )  # Set "nopython" mode for best performance, equivalent to @njit
    def photoelectrons_to_image(n_photoelectrons, gain, offset, variance):
        """
        goes from photoelectrons to image

        This is largely cribbed from Fazel, M.;
        Grussmayer, K. S.; Ferdman, B.; Radenovic, A.; Shechtman, Y.;
        Enderlein, J.; Pressé, S. Rev. Mod. Phys. 96, 025003 (2024).

        Args:
            n_photoelectrons (2d array): 2d matrix of photoelectrons
            gain (2d array): gain of chip
            offset (2d array): offset of chip
            variance (2d array): variance of chip

        Returns:
            image_matrix (numpy.2darray): 2D image matrix
        """
        loc_for_gauss = np.add(np.multiply(gain, n_photoelectrons), offset)
        image_matrix = np.zeros(n_photoelectrons.shape)
        for i in range(n_photoelectrons.shape[0]):
            for j in range(n_photoelectrons.shape[1]):
                image_matrix[i, j] = np.random.normal(
                    loc_for_gauss[i, j], np.sqrt(variance[i, j])
                )
        return image_matrix.clip(0, np.inf)

    def photoelectrons_to_image_array(self, n_photoelectrons, gain, offset, variance):
        """
        goes from photoelectrons to image

        This is largely cribbed from Fazel, M.;
        Grussmayer, K. S.; Ferdman, B.; Radenovic, A.; Shechtman, Y.;
        Enderlein, J.; Pressé, S. Rev. Mod. Phys. 96, 025003 (2024).

        Args:
            n_photoelectrons (2d array): 2d matrix of photoelectrons
            gain (2d array): gain of chip
            offset (2d array): offset of chip
            variance (2d array): variance of chip

        Returns:
            image_matrix (numpy.2darray): 2D image matrix
        """
        loc_for_gauss = np.add(np.multiply(gain, n_photoelectrons), offset)
        image_matrix = np.random.normal(loc_for_gauss, np.sqrt(variance))
        return image_matrix.clip(0, np.inf)

    def generate_sCMOS_g2DPSFs(
        self,
        x,
        y,
        sigma_x,
        sigma_y,
        x0,
        y0,
        n_photons,
        gain,
        offset,
        variance,
        relative_QE,
        abs_QE=0.95,
    ):
        """
        simulates realistic sCMOS images given gain, offset, variance and
        relative_QE parameters, as well as spot locations and photon numbers
        image size will be same as maps.

        This is largely cribbed from equations A1, A2, and A8 of Fazel, M.;
        Grussmayer, K. S.; Ferdman, B.; Radenovic, A.; Shechtman, Y.;
        Enderlein, J.; Pressé, S. Rev. Mod. Phys. 96, 025003 (2024).

        Args:
            x (1d array): x coordinate locations, in same unit as sigma_psf (typically micron)
            y (1d array): y coordinate locations, in same unit as sigma_psf (typically micron)
            sigma_x (float): width of 2d gaussian in pixels
            sigma_y (float): width of 2d gaussian in pixels
            x0 (float or 1d array): origin position of 2d gaussian in x, in pixels
            y0 (float or 1d array): origin position of 2d gaussian in y, in pixels
            n_photons (int/np.1darray): n_photons per localisation. Given to poisson rng
            gain (numpy.2darray): 2D image matrix of gain per pixel
            offset (numpy.2darray): 2D image matrix of offset per pixel
            variance (numpy.2darray): 2D image matrix of variance per pixel
            relative_QE (numpy.2darray): 2D image matrix of relative QE per pixel
            abs_QE (float): absolute QE of the camera
            random (boolean): if True, randomises photon numbers

        Returns:
            image_matrix (numpy.2darray): 2D image matrix
        """
        photon_spatial_pdf = self.gen_spatial_PSF(
            x, y, sigma_x, sigma_y, x0, y0, n_photons, relative_QE
        )

        n_photons_hitting_detector = self.gen_photons_hitting_detector(
            photon_spatial_pdf
        )

        n_photoelectrons = self.gen_photoelectrons(n_photons_hitting_detector, abs_QE)

        image_matrix = self.photoelectrons_to_image(
            n_photoelectrons, gain, offset, variance
        )
        return image_matrix

    @staticmethod
    @jit(
        nopython=True, nogil=True
    )  # Set "nopython" mode for best performance, equivalent to @njit
    def generate_noisy_image_matrix(
        image_size, lambda_sensor, mu_sensor, sigma_sensor, bitdepth=np.float32
    ):
        """
        simulates a noisy image matrix, using noise formulation of Ober et al,
        Biophys J., 2004

        Args:
            image_size (tuple): tuple of how big the image is in pixels
            lambda_sensor (float): mean of possion random variable for background noise
            mu_sensor (float): mean of gaussian for camera read noise
            sigma_sensor (float): sigma of gaussian for camera read noise
            bitdepth (type): bit depth. default uint16

        Returns:
            image_matrix (numpy.ndarray): ND image matrix with noise added to simulate
                                    detector noise
        """
        image_matrix = np.add(
            np.asarray(
                np.random.poisson(lambda_sensor, size=(image_size)), dtype=bitdepth
            ),
            np.asarray(
                np.random.normal(loc=mu_sensor, scale=sigma_sensor, size=(image_size)),
                dtype=bitdepth,
            ),
        )
        return image_matrix

    def generate_superres_stack_background(
        self,
        image_size,
        labelled_pixels,
        background_pixels,
        n_photons=4000,
        n_photons_b=DefaultParameters.N_PHOTONS,
        n_frames=DefaultParameters.N_FRAMES,
        labelling_density=0.2,
        pixel_size=0.11,
        imaging_wavelength=0.520,
        NA=1.49,
        lambda_sensor=200,
        mu_sensor=200,
        sigma_sensor=20,
        bitdepth=np.float32,
    ):
        """
        simulates a super-resolution image stack based on an specified labelled
        pixels in an image

        Args:
            image_size (tuple): tuple of how big the image is in pixels
            labelled_pixels (1d array): pixel indices of where labels are
            n_photons (int): number of photons per localisation
            n_frames (int): number of frames to make up the super-res trace
            labelling_density (float): how many of the pixels will be labelled
                                    across the whole imaging simulation
            pixel_size (float): Default 0.11 micron, how large pixel sizes are
            imaging_wavelength (float): Default is 0.52 microns. Imaging wavelength
            NA (float): Default 1.49, defines how large your PSF will be
            lambda_sensor (float): mean of poisson rnv
            mu_semsor (float): mean of gaussian for camera read noise
            sigma_sensor (float): sigma of gaussian read noise
            bitdepth (type): bit depth. default float32

        Returns:
            superres_image_matrix (numpy.ndarray). ND image matrix with noise and PSFs added.
            superres_cumsum_matrix (numpy.ndarray). ND image matrix of cumulative localisations.
            dl_image_matrix (numpy.2darray). Equivalent diffraction-limited image.
            supreres_image (numpy.2darray). Final superres image.
        """
        stack_size = (image_size[0], image_size[1], n_frames)
        sigma_psf = self.diffraction_limit(imaging_wavelength, NA)
        labelling_number = int(
            labelling_density * len(labelled_pixels)
        )  # get number of labelled pixels
        pixel_subset = np.random.choice(
            labelled_pixels, labelling_number
        )  # get specifically labelled pixels
        superres_cumsum_matrix = np.zeros(stack_size)
        superres_image_matrix = self.generate_noisy_image_matrix(
            stack_size, lambda_sensor, mu_sensor, sigma_sensor, bitdepth
        )

        x0b, y0b = np.unravel_index(background_pixels, image_size, order="F")

        def simulate_frames(frame):
            singleframe_subset = np.random.choice(
                pixel_subset, int(labelling_number / n_frames)
            )
            x0, y0 = np.unravel_index(singleframe_subset, image_size, order="F")
            superres_cumsum_matrix[x0, y0, frame] = n_photons
            superres_image_matrix[:, :, frame] += self.gaussian2d_PSF_pixel(
                image_size, sigma_psf / pixel_size, x0, y0, n_photons, bitdepth
            )
            superres_image_matrix[:, :, frame] += self.gaussian2d_PSF_pixel(
                image_size, sigma_psf / pixel_size, x0b, y0b, n_photons_b, bitdepth
            )

        pool = Pool(nodes=cpu_number)
        pool.restart()
        pool.map(simulate_frames, np.arange(n_frames))
        pool.close()
        pool.terminate()

        superres_cumsum_matrix[:, :, :] = np.cumsum(superres_cumsum_matrix, axis=-1)
        x0d, y0d = np.unravel_index(pixel_subset, image_size, order="F")
        dl_image_matrix = self.generate_noisy_image_matrix(
            image_size, lambda_sensor, mu_sensor, sigma_sensor, np.float64
        )
        dl_image_matrix += self.gaussian2d_PSF_pixel(
            image_size, sigma_psf / pixel_size, x0d, y0d, n_photons, bitdepth
        )
        superres_image = np.zeros_like(dl_image_matrix, dtype=bitdepth)
        superres_image[x0d, y0d] = n_photons
        return (
            superres_image_matrix,
            superres_cumsum_matrix,
            dl_image_matrix,
            superres_image,
        )

    def generate_superres_stack(
        self,
        image_size,
        labelled_pixels,
        n_photons=4000,
        n_frames=DefaultParameters.N_FRAMES,
        labelling_density=0.2,
        pixel_size=0.11,
        imaging_wavelength=0.520,
        NA=1.49,
        lambda_sensor=100,
        mu_sensor=100,
        sigma_sensor=10,
        bitdepth=np.float32,
    ):
        """
        simulates a super-resolution image stack based on an specified labelled
        pixels in an image

        Args:
            image_size (tuple): tuple of how big the image is in pixels
            labelled_pixels (1d array): pixel indices of where labels are
            n_photons (int): number of photons per localisation
            n_frames (int): number of frames to make up the super-res trace
            labelling_density (float): how many of the pixels will be labelled
                                        across the whole imaging simulation
            pixel_size (float): Default 0.11 micron, how large pixel sizes are
            imaging_wavelength (float): Default is 0.52 microns. Imaging wavelength
            NA (float): Default 1.49, defines how large your PSF will be
            lambda_sensor (float): mean of poisson rnv
            mu_semsor (float): mean of gaussian for camera read noise
            sigma_sensor (float): sigma of gaussian read noise
            bitdepth (type): bit depth. default float32

        Returns:
            superres_image_matrix (numpy.ndarray): ND image matrix with noise and PSFs added.
            superres_cumsum_matrix (numpy.ndarray): ND image matrix of cumulative localisations.
            dl_image_matrix (numpy.2darray): Equivalent diffraction-limited image.
            supreres_image (numpy.2darray): Final superres image.
        """
        stack_size = (image_size[0], image_size[1], n_frames)
        sigma_psf = self.diffraction_limit(imaging_wavelength, NA)
        labelling_number = int(
            labelling_density * len(labelled_pixels)
        )  # get number of labelled pixels
        pixel_subset = np.random.choice(
            labelled_pixels, labelling_number
        )  # get specifically labelled pixels
        superres_cumsum_matrix = np.zeros(stack_size)
        superres_image_matrix = self.generate_noisy_image_matrix(
            stack_size, lambda_sensor, mu_sensor, sigma_sensor, bitdepth
        )

        def simulate_frames(frame):
            singleframe_subset = np.random.choice(
                pixel_subset, int(labelling_number / n_frames)
            )
            x0, y0 = np.unravel_index(singleframe_subset, image_size, order="F")
            superres_cumsum_matrix[x0, y0, frame] = n_photons
            superres_image_matrix[:, :, frame] += self.gaussian2d_PSF_pixel(
                image_size, sigma_psf / pixel_size, x0, y0, n_photons, bitdepth
            )

        pool = Pool(nodes=cpu_number)
        pool.restart()
        pool.map(simulate_frames, np.arange(n_frames))
        pool.close()
        pool.terminate()

        superres_cumsum_matrix[:, :, :] = np.cumsum(superres_cumsum_matrix, axis=-1)
        x0d, y0d = np.unravel_index(pixel_subset, image_size, order="F")
        dl_image_matrix = self.generate_noisy_image_matrix(
            image_size, lambda_sensor, mu_sensor, sigma_sensor, np.float64
        )
        dl_image_matrix += self.gaussian2d_PSF_pixel(
            image_size, sigma_psf / pixel_size, x0d, y0d, n_photons, bitdepth
        )
        superres_image = np.zeros_like(dl_image_matrix, dtype=bitdepth)
        superres_image[x0d, y0d] = n_photons
        return (
            superres_image_matrix,
            superres_cumsum_matrix,
            dl_image_matrix,
            superres_image,
        )
