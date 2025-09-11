#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 16:27:38 2024

@author: jbeckwith
"""

import numpy as np
import os
import sys
from concurrent import futures
from scipy.ndimage import uniform_filter
from skimage.filters import gaussian, median
from skimage.measure import block_reduce
from skimage.transform import resize
from colour_demosaicing import demosaicing_CFA_Bayer_Malvar2004

module_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(module_dir)

try:
    import ProgressUtils
except ImportError:
    ProgressUtils = None


class sCMOS_Functions:
    def __init__(
        self,
    ):
        """
        Initialises class.
        """
        return

    def variance_aware_malvar_demosaic(
        self,
        CFA: np.ndarray,
        variance_map: np.ndarray,
        offset_map: np.ndarray | None = None,
        gain: float | np.ndarray = 1.0,
        grayscale: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Alternative approach to variance-aware Malvar demosaicing.
        Applies variance weighting to the input before standard Malvar demosaicing.

        Args:
            CFA: Input CFA data, shape (H, W) or (frames, H, W)
            variance_map: Variance map, shape (H, W) - same spatial dimensions as CFA
            offset_map: Offset map, shape (H, W) - same spatial dimensions as CFA
            gain: Conversion gain from ADU to photoelectrons, scalar or array of shape (H, W)
            grayscale: Whether to return grayscale image

        Returns:
            result: Demosaiced image
            grayscale: Grayscale image if requested, None otherwise
        """
        CFA = np.asarray(CFA, dtype=np.float32)
        variance_map = np.asarray(variance_map, dtype=np.float32)

        # Ensure variance_map is 2D
        if variance_map.ndim > 2:
            variance_map = np.squeeze(variance_map)

        # Handle gain matrix (can be scalar or 2D array)
        if isinstance(gain, np.ndarray):
            gain = np.asarray(gain, dtype=np.float32)
            # Ensure gain is 2D if it's an array
            if gain.ndim > 2:
                gain = np.squeeze(gain)

        # Apply offset correction if provided
        if offset_map is not None:
            offset_map = np.asarray(offset_map, dtype=np.float32)
            # Ensure offset_map is 2D
            if offset_map.ndim > 2:
                offset_map = np.squeeze(offset_map)

            # Handle dimensional broadcasting for offset subtraction
            if CFA.ndim == 3:  # (frames, H, W)
                CFA = CFA - offset_map[np.newaxis, :, :]
            else:  # (H, W)
                CFA = CFA - offset_map

        # Convert from ADU to photoelectrons with proper broadcasting
        if isinstance(gain, np.ndarray) and CFA.ndim == 3:
            # Broadcast gain matrix for 3D CFA
            CFA_pe = CFA / gain[np.newaxis, :, :]
            variance_pe = variance_map / (gain**2)
        elif isinstance(gain, np.ndarray):
            # 2D CFA with 2D gain
            CFA_pe = CFA / gain
            variance_pe = variance_map / (gain**2)
        else:
            # Scalar gain
            CFA_pe = CFA / gain
            variance_pe = variance_map / (gain**2)

        # Calculate weights from variance (inverse variance weighting)
        weights = 1.0 / (variance_pe + 1e-12)

        # Handle dimensional broadcasting for variance weighting
        if CFA_pe.ndim == 3:  # (frames, H, W)
            # Apply variance weighting with proper broadcasting
            weighted_CFA = CFA_pe * weights[np.newaxis, :, :]
        else:  # (H, W)
            weighted_CFA = CFA_pe * weights

        # Normalize by the average weight to maintain overall intensity
        avg_weight = np.mean(weights)
        weighted_CFA = weighted_CFA / avg_weight

        # Now apply standard Malvar demosaicing to the weighted CFA
        result, grayscale_result = self.bayer_demosaic_stack(
            weighted_CFA, grayscale=grayscale
        )
        if grayscale:
            return grayscale_result
        else:
            return result

    def bayer_demosaic_stack_grayscale(self, image):
        """
        Apply colour demosaicking across an entire image stack with parallel processing.

        Args:
            image (np.ndarray): Input image as a NumPy array of shape (H, W) or (C, H, W)
                                where H is height, W is width, and C is the number of channels.

        Returns:
            grayscale_image (np.ndarray): Grayscale demosaiced image
        """
        image = image.astype(np.float32)

        if len(image.shape) <= 2:
            # Single frame - process directly
            rgb_image = demosaicing_CFA_Bayer_Malvar2004(image)
            return np.sum(rgb_image, axis=-1)

        # Multi-frame processing with parallel execution
        n_frames = image.shape[0]
        n_workers = min(os.cpu_count() or 1, 24)  # Limit to reasonable number

        # Create task distribution similar to spot detection
        n_tasks = min(100 * n_workers, n_frames)
        frames_per_task = [
            (
                int(n_frames / n_tasks + 1)
                if _ < n_frames % n_tasks
                else int(n_frames / n_tasks)
            )
            for _ in range(n_tasks)
        ]
        start_indices = np.cumsum([0] + frames_per_task[:-1])

        # Submit parallel tasks
        fs = []
        with futures.ProcessPoolExecutor(n_workers) as executor:
            for i, n_frame_task in zip(start_indices, frames_per_task):
                if n_frame_task > 0:  # Only submit if there are frames to process
                    fs.append(
                        executor.submit(
                            _demosaic_frames_standalone,
                            image[i : i + n_frame_task, :, :],
                        )
                    )

        # Collect results in correct order
        results = []
        if ProgressUtils is not None:
            with ProgressUtils.analysis_progress_bar(
                total=len(fs), desc="Demosaicing frames"
            ) as progress_bar:
                for f in fs:
                    results.append(f.result())
                    progress_bar.update(1)
        else:
            # Fallback without progress bar - preserve order
            for f in fs:
                results.append(f.result())

        # Concatenate results in correct order
        grayscale_image = np.concatenate(results, axis=0)

        return grayscale_image

    def bayer_demosaic_stack(self, image, grayscale=False):
        """
        Apply colour demosaicking across an entire image stack.

        Args:
            image (np.ndarray): Input image as a NumPy array of shape (H, W) or (C, H, W)
                                where H is height, W is width, and C is the number of channels.

        Returns:
            RGB_image (np.ndarray): binned image
        """

        image = image.astype(np.float32)
        if len(image.shape) > 2:
            RGB_image = np.zeros([image.shape[0], image.shape[1], image.shape[2], 3])
            for i in np.arange(image.shape[0]):
                RGB_image[i, :, :, :] = demosaicing_CFA_Bayer_Malvar2004(image[i, :, :])
        else:
            BGR_image = demosaicing_CFA_Bayer_Malvar2004(image)
            RGB_image = np.zeros_like(BGR_image)
            RGB_image = BGR_image
        if grayscale:
            # Convert to grayscale by summing the RGB channels
            grayscale_image = np.sum(RGB_image, axis=-1)
            return RGB_image, grayscale_image
        else:
            return RGB_image, None

    def bayer_bin_stack(self, image, bin_width=2):
        """
        Apply binning of the noise across the four pixels of the bayer mask.

        Args:
            image (np.ndarray): Input image as a NumPy array of shape (H, W) or (C, H, W)
                                where H is height, W is width, and C is the number of channels.
            bin_width (int): width of bins.

        Returns:
            binned_image (np.ndarray): binned image
        """
        image = image.astype(np.float32)
        binned_image = np.zeros_like(image)
        if len(image.shape) > 2:
            for i in np.arange(image.shape[0]):
                binned_image[i, :, :] = resize(
                    block_reduce(image[i, :, :], block_size=bin_width),
                    image[i, :, :].shape,
                    anti_aliasing=False,
                    order=0,
                )
        else:
            binned_image = resize(
                block_reduce(image, block_size=bin_width),
                image.shape,
                anti_aliasing=False,
                order=0,
            )
        return binned_image

    def gaussian_filter_stack(self, image, sigma):
        """
        Apply gaussian filter to an image using a gaussian of width sigma.

        Args:
            image (np.ndarray): Input image as a NumPy array of shape (H, W) or (C, H, W)
                                where H is height, W is width, and C is the number of channels.
            sigma (float): width of smoothing kernel.

        Returns:
            filtered_image (np.ndarray): smoothed image
        """
        image = image.astype(np.float32)
        filtered_image = np.zeros_like(image)
        if len(image.shape) > 2:
            for i in np.arange(image.shape[0]):
                filtered_image[i, :, :] = gaussian(image[i, :, :], sigma=sigma)
        else:
            filtered_image = gaussian(image, sigma=sigma)
        return filtered_image

    def median_filter_stack(self, image, footprint):
        """
        Apply median filter to an image using a specified footprint kernel.

        Args:
            image (np.ndarray): Input image as a NumPy array of shape (H, W) or (C, H, W)
                                where H is height, W is width, and C is the number of channels.
            footprint (np.2darray): smoothing kernel.

        Returns:
            filtered_image (np.ndarray): smoothed image
        """
        image = image.astype(np.float32)
        filtered_image = np.zeros_like(image)
        for i in np.arange(image.shape[0]):
            filtered_image[i, :, :] = median(image[i, :, :], footprint=footprint)
        return filtered_image

    def var_weighted_uniform_filter(self, image, variance_map, kernel_size):
        """
        Apply variance normalisation to an image using a local kernel.
        This function is from Huang, F. et al. Nat. Meth. 10, 653–658 (2013).

        Args:
            image (np.ndarray): Input image as a NumPy array of shape (H, W) or (C, H, W)
                                where H is height, W is width, and C is the number of channels.

            variance_map (np.ndarray): Variance map as a NumPy array of shape (H, W) or
                                        (C, H, W). If it has shape (H, W), it will
                                        be replicated across all channels of the image.

            kernel_size (int): Size of the square kernel used
                                for convolution (e.g., 3 for a 3x3 kernel).

        Returns:

            uniform_image (np.ndarray): Variance-normalised image of
                                        the same shape as the input image.
        """
        image = image.astype(np.float32)
        variance_map = variance_map.astype(np.float32)

        # If variance_map is 2D, replicate it across the channels to match image depth
        if (variance_map.ndim == 2) and (image.ndim > 2):
            variance_map = np.repeat(
                variance_map[np.newaxis, :, :], image.shape[0], axis=0
            )

        # Element-wise division of the image by the variance map
        normalised_image = np.divide(image, variance_map)

        # Create a kernel of ones with size (kernel_size, kernel_size)

        # Perform 2D convolution on each channel of the normalised image
        if image.ndim > 2:
            convolved_image = np.stack(
                [
                    uniform_filter(
                        normalised_image[i, :, :], kernel_size, mode="constant"
                    )
                    for i in range(image.shape[0])
                ],
                axis=0,
            )
        else:
            convolved_image = uniform_filter(
                normalised_image, kernel_size, mode="constant"
            )
        # Compute the weight map by convolving 1.0 / variance_map[:,:,0] with the kernel
        if image.ndim > 2:
            weight_map = uniform_filter(
                np.reciprocal(variance_map[0, :, :]), kernel_size, mode="constant"
            )
        else:
            weight_map = uniform_filter(
                np.reciprocal(variance_map), kernel_size, mode="constant"
            )

        # Replicate the weight map across all channels
        if image.ndim > 2:
            weight_map = np.repeat(
                weight_map[np.newaxis, :, :], convolved_image.shape[0], axis=0
            )

        # Normalise the convolved image by dividing by the weight map
        uniform_image = np.divide(convolved_image, weight_map)

        return uniform_image


# Module-level standalone function for multiprocessing (pickleable)
def _demosaic_frames_standalone(image_chunk: np.ndarray) -> np.ndarray:
    """
    Standalone function for demosaicing a chunk of frames.

    Args:
        image_chunk: Image chunk of shape (n_frames, H, W)

    Returns:
        Grayscale demosaiced images of shape (n_frames, H, W)
    """
    n_frames, height, width = image_chunk.shape
    results = np.zeros((n_frames, height, width), dtype=np.float32)

    for i in range(n_frames):
        # Demosaic to RGB
        rgb_image = demosaicing_CFA_Bayer_Malvar2004(image_chunk[i, :, :])
        # Convert to grayscale by summing RGB channels
        results[i, :, :] = np.sum(rgb_image, axis=-1)

    return results
