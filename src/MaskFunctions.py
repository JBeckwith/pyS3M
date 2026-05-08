#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 11:50:17 2024

@author: jbeckwith
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import sys

import numpy as np
from numpy.typing import NDArray

sys.path.append(str(Path(__file__).parent))


class Mask_Functions:
    """Spatial masking operations for Bayer filter pattern analysis.

    Provides functionality for creating and optimising Bayer-type patterns,
    spatial filtering, and mask operations for multicolour SMLM.
    """

    def __init__(self, camera: str = "ximea", mosaic_unit: NDArray | None = None) -> None:
        """Initialize Mask_Functions class.

        Args:
            camera: Camera model name used to set default ``mosaic_unit``.
                Currently ``"ximea"`` (BGGR) or ``"zwo"`` (RGGB).
                Overridden by an explicit *mosaic_unit* kwarg.
            mosaic_unit: Bayer mosaic pattern array.  If ``None``, taken
                from *camera* defaults.
        """
        import CameraDefaults
        config = CameraDefaults.get_camera_config(camera)
        self.mosaic_unit = mosaic_unit if mosaic_unit is not None else config.mosaic_unit

    def optimise_matrix_symmetry(self, numbers: NDArray, N: int) -> NDArray[np.float64]:
        """
        optimise_matrix_symmetry to get most symmetric bayer-type pattern.
        Things that come at the start of the matrix will be placed preferentially
        in the diagonal. Thus, to replicate the bayer pattern, items would be:
            ['B','R','G']

        Args:
            numbers (np.1darray): numbers to put in matrix
            N (int): matrix is N by N

        Returns:
            matrix (np.2darray): Optimally symmetric array

        """
        # Initialize an N x N matrix filled with NANs
        matrix = np.full((N, N), np.NAN)

        # Place largest numbers along the diagonal first
        for i in range(min(N, len(numbers))):
            matrix[i][i] = numbers[i]

        # Remaining numbers to be placed symmetrically
        remaining_nums = numbers[min(N, len(numbers)) :]

        # Strategy for placing remaining numbers
        def place_symmetrically(num):
            # Find first available symmetrical position
            for i in range(N):
                for j in range(i + 1, N):
                    if np.isnan(matrix[i][j]) and np.isnan(matrix[j][i]):
                        matrix[i][j] = num
                        matrix[j][i] = num
                        return True
            return False

        # Place remaining numbers
        for num in remaining_nums:
            if not place_symmetrically(num):
                if len(numbers) == N * N:
                    return np.array(numbers).reshape(N, N)
                else:
                    for i in range(N):
                        for j in range(N):
                            if np.isnan(matrix[i][j]):
                                matrix[i][j] = num
                                break
                        if matrix[i][j] == num:
                            break

        # first check no NaNs
        if np.sum(np.isnan(matrix)) > 0:
            remaining_number_inds = np.where(~np.isin(numbers, matrix))[0]
            remaining_numbers = []
            for ind in remaining_number_inds:
                remaining_numbers.append(numbers[ind])
            if len(remaining_numbers) == 0:
                remaining_numbers = numbers
            else:
                remaining_numbers.append(numbers)
            nan_locations = np.where(np.isnan(matrix))
            for j in np.arange(len(nan_locations)):
                matrix[nan_locations[j][0], nan_locations[j][1]] = remaining_numbers[j]

        if len(np.unique(matrix)) != len(numbers):
            remaining_number_inds = np.where(~np.isin(numbers, matrix))[0]
            remaining_numbers = []
            for ind in remaining_number_inds:
                remaining_numbers.append(numbers[ind])
            unq, unq_idx, unq_cnt = np.unique(
                matrix, return_inverse=True, return_counts=True
            )
            cnt_mask = unq_cnt > 1
            (cnt_idx,) = np.nonzero(cnt_mask)
            idx_mask = np.in1d(unq_idx, cnt_idx)
            (idx_idx,) = np.nonzero(idx_mask)
            srt_idx = np.argsort(unq_idx[idx_mask])
            dup_idx = np.split(idx_idx[srt_idx], np.cumsum(unq_cnt[cnt_mask])[:-1])
            ltp = np.unravel_index(np.array([x[0] for x in dup_idx]), (N, N))
            ltp_order = np.argsort(ltp[1])[::-1]
            ltp = np.array(ltp)
            for i, num in enumerate(remaining_numbers):
                x, y = ltp[:, ltp_order[i]]
                matrix[x, y] = num
            matrix = matrix.reshape(N, N)
        return np.array(matrix)

    def return_custom_bayer_patterns(self, colours: NDArray[np.int_]) -> NDArray[np.float64]:
        """
        Return a custom bayer pattern based on colours (represented by integers).
        See Figure 3a of Parmar, M. & Reeves, S. J. IEEE Transactions
                        on Image Processing 19, 3190–3203 (2010).

        Args:
            colours (np.1darray of ints): numbers represent colour

        Returns:
            bayer_pattern (np.2darray): An array containing where particular colours
                                    are in the bayer pattern.
        """
        bayer_pattern = self.optimise_matrix_symmetry(
            colours, int(np.ceil(np.sqrt(len(colours))))
        )
        return bayer_pattern

    def return_diagonal_patterns(self, colours: NDArray[np.int_], image_size: int) -> NDArray[np.float64]:
        """
        Return a diagonal pattern based on colours for an image of image_size.
        See Figure 3e of Parmar, M. & Reeves, S. J. IEEE Transactions
                        on Image Processing 19, 3190–3203 (2010).

        Args:
            colours (np.1darray of ints): numbers represent colour
            image_size (int): square image size

        Returns:
            diagonal_pattern (np.2darray): An array containing where particular colours
                                    are in a diagonal pattern.
        """
        n_colours = len(colours)
        diagonal_pattern = np.zeros([image_size, image_size])
        n_diagonals = np.arange(-image_size + 1, image_size)
        colour_selection = np.tile(colours, int(np.ceil(len(n_diagonals) / n_colours)))
        for i, j in enumerate(n_diagonals):
            diagonal_pattern[np.eye(len(diagonal_pattern), k=j, dtype="bool")] = (
                colour_selection[i]
            )
        return diagonal_pattern

    def get_ROI_mask(
        self,
        ROI_x_start: int,
        ROI_y_start: int,
        width: int,
        height: int,
        mosaic_unit: NDArray | None = None,
    ) -> dict[Any, NDArray[np.bool_]]:
        """
        Generates a mask and then reshapes based on ROI.

        Args:
            ROI_x_start (int): An integer saying where the ROI started (x)
            ROI_y_start (int): An integer saying where the ROI started (y)
            width (int): An integer saying how big the image is in the x direction.
            height (int): An integer saying how big the image is in the y direction.

        Returns:
            masks (dict): A dictionary containing the assigned masks.
        """
        if mosaic_unit is None:
            mosaic_unit = self.mosaic_unit
        # Note: get_masks uses (size_x, size_y) but internally treats size_x as height (rows)
        # and size_y as width (columns) due to legacy naming convention
        size_x = ROI_y_start + height  # size_x is actually rows
        size_y = ROI_x_start + width  # size_y is actually columns
        masks = self.get_masks(size_x, size_y, mosaic_unit)
        for colour in masks:
            # Numpy indexing is [row, col] = [y, x]
            masks[colour] = masks[colour][ROI_y_start:, ROI_x_start:]
        return masks

    def get_stacked_masks(
        self, ROI_x_start: int, ROI_y_start: int, width: int, height: int, mosaic_unit: NDArray | None = None
    ) -> NDArray[np.bool_]:
        """Get ROI masks and stack into 3D array for fitting.

        Convenience method that combines get_ROI_mask() with np.dstack() to create
        a 3D mask array suitable for multi-channel fitting operations.

        Args:
            ROI_x_start (int): Starting x coordinate (column) of ROI
            ROI_y_start (int): Starting y coordinate (row) of ROI
            width (int): Width of ROI (pixels)
            height (int): Height of ROI (pixels)
            mosaic_unit (np.ndarray, optional): Mosaic pattern. If None, uses default RGGB pattern.

        Returns:
            np.ndarray: 3D array of shape (height, width, n_channels) containing stacked masks
        """
        if mosaic_unit is None:
            mosaic_unit = self.mosaic_unit

        masks = self.get_ROI_mask(
            ROI_x_start=ROI_x_start,
            ROI_y_start=ROI_y_start,
            width=width,
            height=height,
            mosaic_unit=mosaic_unit,
        )
        return np.dstack([masks[x] for x in masks.keys()])

    def get_masks(self, size_x: int, size_y: int, mosaic_unit: NDArray | None = None) -> dict[Any, NDArray[np.bool_]]:
        """
        Assigns the appropriate masks based on the mosaic unit values.

        Args:
            size_x (int): An integer saying how big the image is in the x direction.
            size_y (int): An integer saying how big the image is in the y direction.
            mosaic_unit (np.2darray): unit of mosaic on the camera

        Returns:
            masks (dict): A dictionary containing the assigned masks.
        """
        if mosaic_unit is None:
            mosaic_unit = self.mosaic_unit
        masks = {}
        default_unit = np.zeros_like(mosaic_unit)
        colours = np.unique(mosaic_unit)

        if not size_x % 2:
            repeat_size_x = int(size_x / 2)
        else:
            repeat_size_x = int(size_x + 1 / 2)

        if not size_y % 2:
            repeat_size_y = int(size_y / 2)
        else:
            repeat_size_y = int(size_y + 1 / 2)

        if mosaic_unit.shape != (size_x, size_y):
            for colour in colours:
                x, y = np.where(mosaic_unit == colour)
                mask = np.zeros_like(default_unit, dtype=bool)
                mask[x, y] = True
                masks[colour] = np.tile(mask, (repeat_size_x, repeat_size_y))
                masks[colour] = masks[colour][:size_x, :size_y]
        else:
            for colour in colours:
                x, y = np.where(mosaic_unit == colour)
                mask = np.zeros_like(default_unit, dtype=bool)
                mask[x, y] = True
                masks[colour] = mask
                masks[colour] = masks[colour][:size_x, :size_y]
        return masks
