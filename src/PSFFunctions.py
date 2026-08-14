# -*- coding: utf-8 -*-
"""
This class contains functions that collect PSF simulation codes for pySMLM
jsb92, 2024/03/04
"""
from pathlib import Path
import numpy as np
import sys
from numba import jit, prange

sys.path.append(str(Path(__file__).parent))


def _sanitize_QE(qe: np.ndarray) -> np.ndarray:
    """Match gen_photoelectrons' QE handling: NaN -> 0, Inf -> 1, clip to [0, 1].

    Small (n_frames,)/(n_frames, n_channels) arrays -- plain NumPy, not numba,
    since there's nothing to parallelize at this size.
    """
    qe = np.asarray(qe, dtype=np.float64)
    qe = np.where(np.isnan(qe), 0.0, qe)
    qe = np.where(np.isinf(qe), 1.0, qe)
    return np.ascontiguousarray(np.clip(qe, 0.0, 1.0))


@jit(nopython=True, parallel=True, nogil=True, cache=True)
def _binomial_batch_uniform_p(n_photons, p_per_frame):
    """Per-pixel binomial sampling, one QE value per frame, parallel over frames.

    Args:
        n_photons: (n_frames, w, h) int32, non-negative photon counts.
        p_per_frame: (n_frames,) float64, QE in [0, 1] for each frame.

    Returns:
        (n_frames, w, h) int32 photoelectron counts.
    """
    n_frames, w, h = n_photons.shape
    out = np.empty((n_frames, w, h), dtype=np.int32)
    for f in prange(n_frames):
        p = p_per_frame[f]
        for i in range(w):
            for j in range(h):
                out[f, i, j] = np.random.binomial(n_photons[f, i, j], p)
    return out


@jit(nopython=True, parallel=True, nogil=True, cache=True)
def _binomial_batch_per_channel(n_photons, p_per_channel):
    """Per-pixel-per-channel binomial sampling, parallel over frames.

    Each photon can only generate a photoelectron on the one Bayer channel it
    actually lands on, so channel contributions are summed directly into the
    output here.

    Args:
        n_photons: (n_frames, w, h, n_channels) int32, non-negative photon
            counts already split by Bayer channel.
        p_per_channel: (n_frames, n_channels) float64, QE in [0, 1] per
            channel/frame.

    Returns:
        (n_frames, w, h) int32 photoelectron counts, summed over channels.
    """
    n_frames, w, h, n_channels = n_photons.shape
    out = np.zeros((n_frames, w, h), dtype=np.int32)
    for f in prange(n_frames):
        for c in range(n_channels):
            p = p_per_channel[f, c]
            for i in range(w):
                for j in range(h):
                    out[f, i, j] += np.random.binomial(n_photons[f, i, j, c], p)
    return out


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
        # np.clip(..., 0, None) (not np.inf) avoids an unnecessary float64
        # upcast + two redundant full-size copies -- clipping an int64 array
        # against a float bound (np.inf) forces NumPy to promote to float64
        # for the clip, which the previous np.array(..., dtype=np.int64) then
        # had to copy back down; at (n_frames, w, h, n_channels) scale in the
        # vectorized caller this tripled peak memory for no numeric benefit
        # (there is no upper bound here, only a floor at 0).
        # int32, not int64: np.random.binomial's legacy RandomState uses the
        # platform C `long` for `n` internally (32-bit on Windows), so an
        # explicit int64 array fails a 'safe'-casting check on Windows only.
        # Photon counts never approach int32's ~2.1e9 ceiling, so this is
        # lossless on every platform.
        n_photons_hitting_detector = np.clip(
            n_photons_hitting_detector, 0, None
        ).astype(np.int32, copy=False)

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

        Binomial sampling runs through a Numba ``prange``-parallelised kernel
        (see ``_binomial_batch_uniform_p``/``_binomial_batch_per_channel``).

        Args:
            n_photons_all_frames: Photons hitting detector for all frames
                                  Shape: (n_frames, w, h, n_dyes)
            QE_per_channel_all: QE values per channel for each frame and dye
                                Shape: (n_frames, n_dyes, n_channels)
                                Example: QE_per_channel_all[frame=5, dye=0, :] = [0.1, 0.7, 0.2] for B,G,R
            mask_stack: Bayer mask stack. Shape ``(w, h, n_channels)`` for a static
                pattern broadcast across all frames, or ``(n_frames, w, h, n_channels)``
                for a per-frame pattern (e.g. a genuinely random pixel-colour
                arrangement drawn independently per bootstrap sample rather than one
                fixed pattern). Example: ``mask_stack[:,:,0]`` = Blue pixel mask
                (static case).

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

            # int32: see gen_photoelectrons' matching comment above -- numba's
            # np.random.binomial needs an integer `n`, and this stays
            # 'safe'-castable to the platform C `long` (32-bit on Windows).
            n_photons_dye_clean = np.ascontiguousarray(
                np.clip(n_photons_dye, 0, None).astype(np.int32, copy=False)
            )

            if all_QE_equal:
                # Fast path: Uniform QE across all channels
                # Can apply QE directly without splitting by Bayer pattern
                QE_uniform = _sanitize_QE(QE_dye[:, 0])  # Shape: (n_frames,)

                # ⚡ PARALLEL: per-pixel binomial sampling across all frames at once
                n_photoelectrons_all[:, :, :, j] = _binomial_batch_uniform_p(
                    n_photons_dye_clean, QE_uniform
                )
            else:
                # Accurate path: Different QE per channel (Bayer camera)
                # Must split photons by Bayer pattern FIRST, then apply channel-specific QE

                # Expand dimensions for broadcasting:
                # n_photons_dye: (n_frames, w, h) → (n_frames, w, h, 1)
                # mask_stack (static): (w, h, n_channels) → (1, w, h, n_channels), broadcast to all frames
                # mask_stack (per-frame): (n_frames, w, h, n_channels), already frame-aligned
                # Result: (n_frames, w, h, n_channels)
                if mask_stack.ndim == 4:
                    mask_stack_broadcast = mask_stack               # already (n_frames, w, h, n_channels)
                else:
                    mask_stack_broadcast = mask_stack[np.newaxis, :, :, :]  # (1, w, h, n_channels)

                n_photons_per_channel = np.ascontiguousarray(
                    np.clip(
                        n_photons_dye[:, :, :, np.newaxis] *    # (n_frames, w, h, 1)
                        mask_stack_broadcast,
                        0, None,
                    ).astype(np.int32, copy=False)
                )
                QE_dye_clean = _sanitize_QE(QE_dye)  # Shape: (n_frames, n_channels)

                # ⚡ PARALLEL: per-pixel-per-channel binomial sampling, channels
                # summed inline (each photon contributes to exactly one channel)
                n_photoelectrons_all[:, :, :, j] = _binomial_batch_per_channel(
                    n_photons_per_channel, QE_dye_clean
                )

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

