#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 14:55:31 2025

@author: jbeckwith
"""
import numpy as np
from numba import jit


@jit(nopython=True, nogil=True)
def gaussian_unscaled_model(
    array_tofill: np.ndarray,
    x: np.ndarray,
    size: int,
    x0: float,
    y0: float,
    sigma_x: float,
    sigma_y: float,
) -> np.ndarray:
    """gauss2d: Returns 2D gaussian

    Args:
        array_tofill (np.ndarray): array to be filled
        x (np.1darray): x array
        size (int): how big the array is
        x0 (float): x-coordinate of the center of the gaussian
        y0 (float): y-coordinate of the center of the gaussian
        sigma (float): standard deviation of the gaussian

    Returns:
        signal (float): signal at particular (x, y) location"""
    norm_x = 0.3989422804014327 / sigma_x  # empirical
    norm_y = 0.3989422804014327 / sigma_y  # empirical
    xg = norm_x * np.exp(-0.5 * ((x - x0) / sigma_x) ** 2)
    yg = norm_y * np.exp(-0.5 * ((x - y0) / sigma_y) ** 2)
    for i in range(size):
        for j in range(size):
            array_tofill[i, j] = xg[i] * yg[j]
    return array_tofill


@jit(nopython=True, nogil=True)
def WLS_justcolour_model_nobounds(
    params,
    x,
    gauss_2d,
    locparams,
):
    """
    Calculate a coloured gaussian based on input params.


    Args:
        params (numpy.ndarray): Input parameters.
        data (numpy.ndarray): Data to fit.
        masks (np.3darray): 3d array of colour masks

    Returns:
        gauss_2d (numpy.ndarray): 2d bayer filtered gaussian.
    """

    gauss_2d[:, :] = (
        np.multiply(
            params[0] ** 2,
            gaussian_unscaled_model(
                gauss_2d[:, :],
                x,
                len(x),
                locparams[0],
                locparams[1],
                locparams[2],
                locparams[3],
            ),
        )
        + params[1] ** 2
    )
    return gauss_2d


@jit(nopython=True, nogil=True)
def WLS_nocolour_model_nobounds(
    params,
    data,
    x,
    gauss_2d,
):
    """
    Calculate a coloured gaussian based on input params.


    Args:
        params (numpy.ndarray): Input parameters.
        data (numpy.ndarray): Data to fit.
        masks (np.3darray): 3d array of colour masks

    Returns:
        gauss_2d (numpy.ndarray): 2d bayer filtered gaussian.
    """

    gauss_2d[:, :] = (
        np.multiply(
            params[5] ** 2,
            gaussian_unscaled_model(
                gauss_2d[:, :],
                x,
                len(x),
                params[0],
                params[1],
                params[2],
                params[3],
            ),
        )
        + params[4] ** 2
    )
    return gauss_2d


@jit(nopython=True, nogil=True)
def WLS_rawcolour_model_nobounds(
    params,
    data,
    masks,
    background_bayer_matrix,
    bayer_matrix,
    x,
    gauss_2d,
    locparams,
):
    """
    Calculate a coloured gaussian based on input params.


    Args:
        params (numpy.ndarray): Input parameters.
        data (numpy.ndarray): Data to fit.
        masks (np.3darray): 3d array of colour masks

    Returns:
        gauss_2d (numpy.ndarray): 2d bayer filtered gaussian.
    """

    for i in np.arange(masks.shape[-1]):
        pixels = masks[:, :, i].ravel()
        bayer_matrix[pixels] = params[-3 + i] ** 2
        background_bayer_matrix[pixels] = params[-6 + i] ** 2
    bayer_matrix = bayer_matrix.reshape(len(x), len(x))
    background_bayer_matrix = background_bayer_matrix.reshape(len(x), len(x))
    gauss_2d[:, :] = (
        np.multiply(
            bayer_matrix,
            gaussian_unscaled_model(
                gauss_2d[:, :],
                x,
                len(x),
                locparams[0],
                locparams[1],
                locparams[2],
                locparams[3],
            ),
        )
        + background_bayer_matrix
    )
    return gauss_2d


@jit(nopython=True, nogil=True)
def WLS_model_nobounds(
    params,
    masks,
    x,
    gauss_2d,
):
    """
    Calculate a coloured gaussian based on input params.


    Args:
        params (numpy.ndarray): Input parameters.
        masks (np.3darray): 3d array of colour masks
        x (numpy.ndarray): coordinate array
        gauss_2d (numpy.ndarray): output array to fill

    Returns:
        gauss_2d (numpy.ndarray): 2d bayer filtered gaussian.
    """

    len_x = len(x)

    # Create lookup tables for amplitudes and backgrounds
    amp_lookup = np.array(
        [
            params[7] * params[7],  # Blue
            params[8] * params[8],  # Green
            params[9] * params[9],
        ]
    )  # Red

    bg_lookup = np.array(
        [
            params[4] * params[4],  # Blue
            params[5] * params[5],  # Green
            params[6] * params[6],
        ]
    )  # Red

    # First compute the Gaussian using the exact same method as gaussian_unscaled_model
    gauss_2d = gaussian_unscaled_model(
        gauss_2d, x, len_x, params[0], params[1], params[2], params[3]
    )

    # Apply Bayer pattern efficiently per channel (avoids conditionals in inner loop)
    for channel in range(3):
        for i in range(len_x):
            for j in range(len_x):
                if masks[j, i, channel]:
                    gauss_2d[j, i] = (
                        amp_lookup[channel] * gauss_2d[j, i] + bg_lookup[channel]
                    )

    return gauss_2d


@jit(nopython=True, nogil=True)
def WLS_chi_nobounds(params, data, masks, weights, size, ravelsize):
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
    x = np.arange(size, dtype=np.float32)
    gauss_2d = np.zeros((size, size), dtype=np.float32)
    chi = np.zeros((size, size), dtype=np.float32)
    gauss_2d[:, :] = WLS_model_nobounds(params, masks, x, gauss_2d)
    chi[:, :] = np.multiply(weights, np.square(np.subtract(data, gauss_2d)))
    return np.sqrt(chi.ravel())


@jit(nopython=True, nogil=True)
def WLS_chi_nocolour_nobounds(params, data, weights, size, ravelsize):
    """
    Calculate the chi vector for the weighted least squares model.

    Args:
        params (numpy.ndarray): Input parameters.
        data (numpy.ndarray): Data to fit.
        weights (np.ndarray): weights for the chi

    Returns:
        chi (numpy.ndarray): Vector of chi.
    """
    x = np.arange(size, dtype=np.float32)
    gauss_2d = np.zeros((size, size), dtype=np.float32)
    chi = np.zeros((size, size), dtype=np.float32)
    gauss_2d[:, :] = WLS_nocolour_model_nobounds(params, data, x, gauss_2d)
    chi[:, :] = np.multiply(weights, np.square(np.subtract(data, gauss_2d)))
    return np.sqrt(chi.ravel())


@jit(nopython=True, nogil=True)
def WLS_chi_justcolour_nobounds(params, data, weights, size, locparams):
    """
    Calculate the chi vector for the weighted least squares model.

    Args:
        params (numpy.ndarray): Input parameters.
        data (numpy.ndarray): Data to fit.
        weights (np.ndarray): weights for the chi

    Returns:
        chi (numpy.ndarray): Vector of chi.
    """
    x = np.arange(size, dtype=np.float32)
    gauss_2d = np.zeros((size, size), dtype=np.float32)
    chi = np.zeros((size, size), dtype=np.float32)
    gauss_2d[:, :] = WLS_justcolour_model_nobounds(params, x, gauss_2d, locparams)
    chi[:, :] = np.multiply(weights, np.square(np.subtract(data, gauss_2d)))
    return np.sqrt(chi.ravel())


@jit(nopython=True, nogil=True)
def WLS_rawcolour_chi_nobounds(
    params, data, masks, weights, size, ravelsize, locparams
):
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
    x = np.arange(size, dtype=np.float32)
    background_bayer_matrix = np.zeros(ravelsize, dtype=np.float32)
    bayer_matrix = np.zeros(ravelsize, dtype=np.float32)
    gauss_2d = np.zeros((size, size), dtype=np.float32)
    chi = np.zeros((size, size), dtype=np.float32)
    gauss_2d[:, :] = WLS_rawcolour_model_nobounds(
        params,
        data,
        masks,
        background_bayer_matrix,
        bayer_matrix,
        x,
        gauss_2d,
        locparams,
    )
    chi[:, :] = np.multiply(weights, np.square(np.subtract(data, gauss_2d)))
    return np.sqrt(chi.ravel())


@jit(nopython=True, nogil=True)
def _sum_and_centre_of_mass(smoothed_data, size):
    x_ig = 0.0
    y_ig = 0.0
    A = 0.0
    for i in range(size):
        for j in range(size):
            x_ig += smoothed_data[i, j] * i
            y_ig += smoothed_data[i, j] * j
            A += smoothed_data[i, j]
    x_ig /= A
    y_ig /= A
    return np.abs(A), np.abs(x_ig), np.abs(y_ig)


@jit(nopython=True, nogil=True)
def _initial_sigma(smoothed_data, x_ig, y_ig, A, size):
    sum_deviation_y = 0.0
    sum_deviation_x = 0.0
    for i in range(size):
        for j in range(size):
            sum_deviation_y += smoothed_data[i, j] * (i - y_ig) ** 2
            sum_deviation_x += smoothed_data[i, j] * (j - x_ig) ** 2
    sy = np.sqrt(sum_deviation_y / A)
    sx = np.sqrt(sum_deviation_x / A)
    return np.abs(sy), np.abs(sx)


@jit(nopython=True)
def initial_nocolour_guess(smoothed_data, raw_data):
    """
    initial_guess of gaussian input parameters.

    Args:
        smoothed_data (np.2darray): smoothed photoelectron data matrix.
        raw_data (np.2darray): raw photoelectron data matrix

    Returns:
        x_ig (float): centroid guess in x.
        y_ig (float): centroid guess in y.
        sigma (float): sigma guess
        bB (float): background guess Blue
        bG (float): background guess Green
        bR (float): background guess Red
        A_ig (float): amplitude guess for all colours
    """
    flattened_rawdata = raw_data.ravel()
    b = np.min(np.abs(flattened_rawdata))
    ig_data = np.abs(smoothed_data)
    bs_data = ig_data - np.abs(np.min(ig_data))
    size = bs_data.shape[0]
    A, x_ig, y_ig = _sum_and_centre_of_mass(bs_data, size)
    sigma_y, sigma_x = _initial_sigma(bs_data, x_ig, y_ig, A, size)
    return x_ig, y_ig, sigma_y, sigma_x, b, A


@jit(nopython=True)
def initial_justcolour_guess(smoothed_data, raw_data):
    """
    initial_guess of gaussian input parameters.

    Args:
        smoothed_data (np.2darray): smoothed photoelectron data matrix.
        raw_data (np.2darray): raw photoelectron data matrix

    Returns:
        b (float): background guess
        A_ig (float): amplitude guess
    """
    flattened_rawdata = raw_data.ravel()
    b = np.min(np.abs(flattened_rawdata))
    ig_data = np.abs(smoothed_data)
    bs_data = ig_data - np.abs(np.min(ig_data))
    A = np.sum(bs_data)
    return A, b


@jit(nopython=True)
def initial_rawcolour_guess(smoothed_data, raw_data, masks):
    """
    initial_guess of gaussian input parameters.

    Args:
        smoothed_data (np.2darray): smoothed photoelectron data matrix.
        raw_data (np.2darray): raw photoelectron data matrix

    Returns:
        b (float): background guess
        A_ig (float): amplitude guess
    """
    BG_matrix = np.zeros(masks.shape[-1])
    flattened_rawdata = raw_data.ravel()
    for i in np.arange(masks.shape[-1]):
        pixels = masks[:, :, i].ravel()
        BG_matrix[i] = np.min(np.abs(flattened_rawdata[pixels]))
    bB, bG, bR = BG_matrix
    ig_data = np.abs(smoothed_data)
    bs_data = ig_data - np.abs(np.min(ig_data))
    size = bs_data.shape[0]
    A = np.sum(bs_data)
    A_ig = A / 3.0
    return bB, bG, bR, A_ig, A_ig, A_ig


@jit(nopython=True)
def initial_guess(smoothed_data, raw_data, masks):
    """
    initial_guess of gaussian input parameters.

    Args:
        smoothed_data (np.2darray): smoothed photoelectron data matrix.
        raw_data (np.2darray): raw photoelectron data matrix
        masks (np.3darray): mask matrix

    Returns:
        x_ig (float): centroid guess in x.
        y_ig (float): centroid guess in y.
        sigma (float): sigma guess
        bB (float): background guess Blue
        bG (float): background guess Green
        bR (float): background guess Red
        A_ig (float): amplitude guess for all colours
    """
    BG_matrix = np.zeros(masks.shape[-1])
    flattened_rawdata = raw_data.ravel()
    for i in np.arange(masks.shape[-1]):
        pixels = masks[:, :, i].ravel()
        BG_matrix[i] = np.min(np.abs(flattened_rawdata[pixels]))
    bB, bG, bR = BG_matrix
    ig_data = np.abs(smoothed_data)
    bs_data = ig_data - np.abs(np.min(ig_data))
    size = bs_data.shape[0]
    A, x_ig, y_ig = _sum_and_centre_of_mass(bs_data, size)
    sigma_y, sigma_x = _initial_sigma(bs_data, x_ig, y_ig, A, size)
    A_ig = A / 3.0
    return x_ig, y_ig, sigma_y, sigma_x, bB, bG, bR, A_ig, A_ig, A_ig
