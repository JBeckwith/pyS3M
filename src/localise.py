"""
    localise
    ~~~~~~~~~~~~~~~~

    Identify and localise single puncta in a frame sequence

    :original authors: Joerg Schnitzbauer, Maximilian Thomas Strauss, 2016-2018
    Updated by jsb92, 2025/08/18
"""

import os
import sys
import numpy as np
import dask.array as da
import numba
import multiprocessing as mp
import ctypes
from concurrent.futures import ThreadPoolExecutor
import threading
from itertools import chain

module_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(module_dir)
from ImportManager import get_module
import postprocess
import os
from datetime import datetime
import pandas as pd

plt = get_module("matplotlib.pyplot")

MAX_LOCS = int(1e6)

_C_FLOAT_POINTER = ctypes.POINTER(ctypes.c_float)
LOCS_DTYPE = [
    ("frame", "u4"),
    ("xc", "f4"),  # Changed from "x" to "xc" for consistency
    ("yc", "f4"),  # Changed from "y" to "yc" for consistency
    ("photons", "f4"),
    ("sx", "f4"),
    ("sy", "f4"),
    ("bg", "f4"),
    (
        "xc_err",
        "f4",
    ),  # Changed from "lpx" to "xc_err" for consistency with SR_Functions.py
    (
        "yc_err",
        "f4",
    ),  # Changed from "lpy" to "yc_err" for consistency with SR_Functions.py
    ("net_gradient", "f4"),
    ("likelihood", "f4"),
    ("iterations", "i4"),
]

MEAN_COLS = [
    "frame",
    "xc",  # Changed from "x" to "xc" for consistency
    "yc",  # Changed from "y" to "yc" for consistency
    "photons",
    "sx",
    "sy",
    "bg",
    "xc_err",  # Changed from "lpx" to "xc_err" for consistency with SR_Functions.py
    "yc_err",  # Changed from "lpy" to "yc_err" for consistency with SR_Functions.py
    "ellipticity",
    "net_gradient",
    "z",
    "d_zcalib",
]
SET_COLS = ["Frames", "Height", "Width", "Box Size", "Min. Net Gradient", "Pixelsize"]


@numba.jit(nopython=True, nogil=True, cache=False)
def local_maxima(frame, box):
    """Finds pixels with maximum value within a region of interest"""
    Y, X = frame.shape
    maxima_map = np.zeros(frame.shape, np.uint8)
    box_half = int(box / 2)
    box_half_1 = box_half + 1
    for i in range(box_half, Y - box_half_1):
        for j in range(box_half, X - box_half_1):
            local_frame = frame[
                i - box_half : i + box_half + 1,
                j - box_half : j + box_half + 1,
            ]
            flat_max = np.argmax(local_frame)
            i_local_max = int(flat_max / box)
            j_local_max = int(flat_max % box)
            if (i_local_max == box_half) and (j_local_max == box_half):
                maxima_map[i, j] = 1
    y, x = np.where(maxima_map)
    return y, x


@numba.jit(nopython=True, nogil=True, cache=False)
def gradient_at(frame, y, x, i):
    gy = frame[y + 1, x] - frame[y - 1, x]
    gx = frame[y, x + 1] - frame[y, x - 1]
    return gy, gx


@numba.jit(nopython=True, nogil=True, cache=False)
def net_gradient(frame, y, x, box, uy, ux):
    box_half = int(box / 2)
    ng = np.zeros(len(x), dtype=np.float32)
    for i, (yi, xi) in enumerate(zip(y, x)):
        for k_index, k in enumerate(range(yi - box_half, yi + box_half + 1)):
            for l_index, m in enumerate(range(xi - box_half, xi + box_half + 1)):
                if not (k == yi and m == xi):
                    gy, gx = gradient_at(frame, k, m, i)
                    ng[i] += gy * uy[k_index, l_index] + gx * ux[k_index, l_index]
    return ng


@numba.jit(nopython=True, nogil=True, cache=False)
def identify_in_image(image, minimum_ng, box):
    y, x = local_maxima(image, box)
    box_half = int(box / 2)
    # Now comes basically a meshgrid
    ux = np.zeros((box, box), dtype=np.float32)
    uy = np.zeros((box, box), dtype=np.float32)
    for i in range(box):
        val = box_half - i
        ux[:, i] = uy[i, :] = val
    unorm = np.sqrt(ux**2 + uy**2)
    ux /= unorm
    uy /= unorm
    ng = net_gradient(image, y, x, box, uy, ux)
    positives = ng > minimum_ng
    y = y[positives]
    x = x[positives]
    ng = ng[positives]
    return y, x, ng


def identify_in_frame(frame, minimum_ng, box, roi=None):
    if roi is not None:
        frame = frame[roi[0][0] : roi[1][0], roi[0][1] : roi[1][1]]
    image = np.float32(frame)  # otherwise numba goes crazy
    y, x, net_gradient = identify_in_image(image, minimum_ng, box)
    if roi is not None:
        y += roi[0][0]
        x += roi[0][1]
    return y, x, net_gradient


def identify_frame(frame, minimum_ng, box, frame_number, roi=None, resultqueue=None):
    y, x, net_gradient = identify_in_frame(frame, minimum_ng, box, roi)
    frame = frame_number * np.ones(len(x))
    result = np.rec.array(
        (frame, x, y, net_gradient),
        dtype=[
            ("frame", "i"),
            ("xc", "i"),
            ("yc", "i"),
            ("net_gradient", "f4"),
        ],  # Changed to "xc", "yc"
    )
    if resultqueue is not None:
        resultqueue.put(result)
    return result


def identify_by_frame_number(movie, minimum_ng, box, frame_number, roi=None, lock=None):
    if lock is not None:
        with lock:
            frame = movie[frame_number]
    else:
        frame = movie[frame_number]
    y, x, net_gradient = identify_in_frame(frame, minimum_ng, box, roi)
    frame = frame_number * np.ones(len(x))
    return np.rec.array(
        (frame, x, y, net_gradient),
        dtype=[
            ("frame", "i"),
            ("xc", "i"),
            ("yc", "i"),
            ("net_gradient", "f4"),
        ],  # Changed to "xc", "yc"
    )


def _identify_worker(movie, current, minimum_ng, box, roi, lock):
    n_frames = len(movie)
    identifications = []
    while True:
        with lock:
            index = current[0]
            if index == n_frames:
                return identifications
            current[0] += 1
        identifications.append(
            identify_by_frame_number(movie, minimum_ng, box, index, roi, lock)
        )


def identifications_from_futures(futures):
    ids_list_of_lists = [_.result() for _ in futures]
    ids_list = list(chain(*ids_list_of_lists))
    ids = np.hstack(ids_list).view(np.recarray)
    ids.sort(kind="mergesort", order="frame")
    return ids


@numba.jit(nopython=True, cache=False)
def _cut_spots_numba(movie, ids_frame, ids_x, ids_y, box):
    n_spots = len(ids_x)
    r = int(box / 2)
    spots = np.zeros((n_spots, box, box), dtype=movie.dtype)
    for id, (frame, xc, yc) in enumerate(zip(ids_frame, ids_x, ids_y)):
        spots[id] = movie[frame, yc - r : yc + r + 1, xc - r : xc + r + 1]
    return spots


@numba.jit(nopython=True, cache=False)
def _cut_spots_frame(frame, frame_number, ids_frame, ids_x, ids_y, r, start, N, spots):
    for j in range(start, N):
        if ids_frame[j] > frame_number:
            break
        if ids_frame[j] < frame_number:
            break
        yc = ids_y[j]
        xc = ids_x[j]
        spots[j] = frame[yc - r : yc + r + 1, xc - r : xc + r + 1]
    return j


@numba.jit(nopython=True, cache=False)
def _cut_spots_daskmov(movie, l_mov, ids_frame, ids_x, ids_y, box, spots):
    """Cuts the spots out of a movie frame by frame.

    Args:
        movie : 3D array (t, x, y)
            the image data (can be dask or numpy array)
        l_mov : 1D array, len 1
            lenght of the movie (=t); in array to satisfy the combination of
            range() and guvectorization
        ids_frame, ids_x, ids_y : 1D array (k)
            spot positions in the image data. Length: number of spots
            identified
        box : uneven int
            the cut spot box size
        spots : 3D array (k, box, box)
            the cut spots
    Returns:
        spots : as above
            the image-data filled spots
    """
    r = int(box / 2)
    N = len(ids_frame)
    start = 0
    for frame_number in range(l_mov[0]):
        frame = movie[frame_number, :, :]
        start = _cut_spots_frame(
            frame,
            frame_number,
            ids_frame,
            ids_x,
            ids_y,
            r,
            start,
            N,
            spots,
        )
    return spots


def _cut_spots_framebyframe(movie, ids_frame, ids_x, ids_y, box, spots):
    """Cuts the spots out of a movie frame by frame.

    Args:
        movie : 3D array (t, x, y)
            the image data (can be dask or numpy array)
        ids_frame, ids_x, ids_y : 1D array (k)
            spot positions in the image data. Length: number of spots
            identified
        box : uneven int
            the cut spot box size
        spots : 3D array (k, box, box)
            the cut spots
    Returns:
        spots : as above
            the image-data filled spots
    """
    r = int(box / 2)
    N = len(ids_frame)
    start = 0
    for frame_number, frame in enumerate(movie):
        start = _cut_spots_frame(
            frame,
            frame_number,
            ids_frame,
            ids_x,
            ids_y,
            r,
            start,
            N,
            spots,
        )
    return spots


def _cut_spots(movie, ids, box):
    N = len(ids.frame)
    if isinstance(movie, np.ndarray):
        return _cut_spots_numba(
            movie, ids.frame, ids.xc, ids.yc, box
        )  # Changed from .x, .y to .xc, .yc
    else:
        """Assumes that identifications are in order of frames!"""

        N = len(ids.frame)
        spots = np.zeros((N, box, box), dtype=movie.dtype)
        spots = _cut_spots_framebyframe(
            movie, ids.frame, ids.xc, ids.yc, box, spots
        )  # Changed from .x, .y to .xc, .yc
        return spots


def _to_photons(spots, camera_info):
    spots = np.float32(spots)
    baseline = camera_info["Baseline"]
    sensitivity = camera_info["Sensitivity"]
    gain = camera_info["Gain"]
    # since v0.6.0: remove quantum efficiency to better reflect precision
    # qe = camera_info["Qe"]
    return (spots - baseline) * sensitivity / (gain)


def get_spots(movie, identifications, box, camera_info):
    spots = _cut_spots(movie, identifications, box)
    return _to_photons(spots, camera_info)


def locs_from_fits(identifications, theta, CRLBs, likelihoods, iterations, box):
    box_offset = int(box / 2)
    y = theta[:, 0] + identifications.yc - box_offset  # Changed from .y to .yc
    x = theta[:, 1] + identifications.xc - box_offset  # Changed from .x to .xc
    yc_err = np.sqrt(CRLBs[:, 0])  # Changed from lpy to yc_err
    xc_err = np.sqrt(CRLBs[:, 1])  # Changed from lpx to xc_err
    locs = np.rec.array(
        (
            identifications.frame,
            x,
            y,
            theta[:, 2],
            theta[:, 5],
            theta[:, 4],
            theta[:, 3],
            xc_err,  # Changed from lpx to xc_err
            yc_err,  # Changed from lpy to yc_err
            identifications.net_gradient,
            likelihoods,
            iterations,
        ),
        dtype=LOCS_DTYPE,
    )
    locs.sort(kind="mergesort", order="frame")
    return locs


def check_nena(locs, info, callback=None):
    # Nena
    print("Calculating NeNA.. ", end="")
    locs = locs[0:MAX_LOCS]
    try:
        result, best_result = postprocess.nena(locs, info, callback=callback)
        nena_px = best_result
    except Exception as e:
        print(e)
        nena_px = float("nan")

    print(f"{nena_px:.2f} px.")

    return nena_px


def check_kinetics(locs, info):
    print("Linking.. ", end="")
    locs = locs[0:MAX_LOCS]
    locs = postprocess.link(locs, info=info)
    len_mean = locs.len.mean()
    print(f"Mean lenght {len_mean:.2f} frames.")

    return len_mean


def check_drift(locs, info, callback=None):
    steps = int(len(locs) // (MAX_LOCS))
    steps = max(1, steps)

    locs = locs[::steps]

    n_frames = info[0]["Frames"]

    segmentation = max(1, int(n_frames // 10))

    print(f"Estimating drift with segmentation {segmentation}")
    drift, locs = postprocess.undrift(
        locs,
        info,
        segmentation,
        display=False,
        rcc_callback=callback,
    )

    drift_x = float(drift["x"].mean())
    drift_y = float(drift["y"].mean())

    print(f"Drift is X: {drift_x:.2f}, Y: {drift_y:.2f}.")

    return (drift_x, drift_y)
