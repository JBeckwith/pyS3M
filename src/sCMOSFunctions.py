#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 16:27:38 2024

@author: jbeckwith
"""

import numpy as np
import os
import sys
from scipy.ndimage import uniform_filter
from skimage.filters import gaussian, median
from skimage.measure import block_reduce
from skimage.transform import resize

module_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(module_dir)
import IOFunctions

IO = IOFunctions.IO_Functions()


class sCMOS_Functions:
    def __init__(
        self,
    ):
        """
        Initialises class.
        """
        return

    def bayer_bin_stack(self, image, bin_width=2):
        """
        Apply binning of the nosie across the four pixels of the bayer mask.

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
        Apply variance normalization to an image using a local kernel.
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

            uniform_image (np.ndarray): Variance-normalized image of
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
        normalized_image = np.divide(image, variance_map)

        # Create a kernel of ones with size (kernel_size, kernel_size)

        # Perform 2D convolution on each channel of the normalized image
        if image.ndim > 2:
            convolved_image = np.stack(
                [
                    uniform_filter(
                        normalized_image[i, :, :], kernel_size, mode="constant"
                    )
                    for i in range(image.shape[0])
                ],
                axis=0,
            )
        else:
            convolved_image = uniform_filter(
                normalized_image, kernel_size, mode="constant"
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

        # Normalize the convolved image by dividing by the weight map
        uniform_image = np.divide(convolved_image, weight_map)

        return uniform_image
