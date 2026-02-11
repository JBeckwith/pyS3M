"""
    gui/postprocess
    ~~~~~~~~~~~~~~~~~~~~

    Data analysis of localisation lists

    :authors: Joerg Schnitzbauer, Maximilian Thomas Strauss, 2015-2018
    :copyright: Copyright (c) 2015-2018 Jungmann Lab, MPI Biochemistry
"""

import os
import sys
import numpy as np
import numba

from scipy import interpolate
from scipy.special import iv
from scipy.spatial import distance

from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import itertools
import lmfit
from collections import OrderedDict

module_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(module_dir)
from ImportManager import get_module, is_available
import lib
import render
import imageprocess
from threading import Thread
import ProgressUtils
from numpy.lib.recfunctions import stack_arrays
from sklearn.neighbors import NearestNeighbors as NN

plt = get_module("matplotlib.pyplot")


def _plot_drift_analysis(drift, shift_x, shift_y, bounds, save_path=None):
    """Create standardized drift analysis plot using consolidated plotting."""
    if not is_available("matplotlib.pyplot"):
        print("⚠️ Matplotlib not available - skipping drift plot display")
        return None, None

    try:
        from PlottingBase import AnalysisPlotter

        plotter = AnalysisPlotter()

        fig, axes = plotter.two_column_plot(nrows=1, ncols=2, width=17, height=6, big=True)
        fig.suptitle("Estimated drift")

        # Calculate time points for original measurements
        t = (bounds[1:] + bounds[:-1]) / 2

        # Left panel: Time series
        ax1 = axes[0]
        ax1.plot(drift.x, label="x interpolated")
        ax1.plot(drift.y, label="y interpolated")
        ax1.plot(t, shift_x, "o", label="x")
        ax1.plot(t, shift_y, "o", label="y")
        plotter.setup_axis(
            ax1, xlabel="Frame", ylabel="Drift (pixel)", title="", legend=True
        )

        # Right panel: Trajectory
        ax2 = axes[1]
        ax2.plot(drift.x, drift.y)
        ax2.plot(shift_x, shift_y, "o")
        plotter.setup_axis(ax2, xlabel="x", ylabel="y", equal_aspect=True)

        plotter.save_or_show(fig, save_path=save_path)
        return fig, axes

    except ImportError:
        # Fallback to basic matplotlib if PlottingBase not available
        if plt is None:
            print("⚠️ Plotting not available - skipping drift analysis display")
            return None, None

        fig = plt.figure(figsize=(17, 6))
        plt.suptitle("Estimated drift")

        t = (bounds[1:] + bounds[:-1]) / 2

        plt.subplot(1, 2, 1)
        plt.plot(drift.x, label="x interpolated")
        plt.plot(drift.y, label="y interpolated")
        plt.plot(t, shift_x, "o", label="x")
        plt.plot(t, shift_y, "o", label="y")
        plt.legend(loc="best")
        plt.xlabel("Frame")
        plt.ylabel("Drift (pixel)")

        plt.subplot(1, 2, 2)
        plt.plot(drift.x, drift.y)
        plt.plot(shift_x, shift_y, "o")
        plt.axis("equal")
        plt.xlabel("x")
        plt.ylabel("y")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)
        return fig, None


def get_index_blocks(locs, width, height, size, callback=None):
    # Sort locs by indices
    x_index = np.uint32(locs.xc / size)
    y_index = np.uint32(locs.yc / size)
    sort_indices = np.lexsort([x_index, y_index])
    locs = locs[sort_indices]
    x_index = x_index[sort_indices]
    y_index = y_index[sort_indices]
    # Allocate block info arrays
    n_blocks_y, n_blocks_x = index_blocks_shape(width, height, size)
    block_starts = np.zeros((n_blocks_y, n_blocks_x), dtype=np.uint32)
    block_ends = np.zeros((n_blocks_y, n_blocks_x), dtype=np.uint32)
    K, L = block_starts.shape
    # Fill in block starts and ends
    thread = Thread(
        target=_fill_index_blocks,
        args=(block_starts, block_ends, x_index, y_index),
    )
    thread.start()
    thread.join()
    return locs, size, x_index, y_index, block_starts, block_ends, K, L


def index_blocks_shape(width, height, size):
    """Returns the shape of the index grid, given the movie and grid sizes"""
    n_blocks_x = int(np.ceil(width / size))
    n_blocks_y = int(np.ceil(height / size))
    return n_blocks_y, n_blocks_x


@numba.jit(nopython=True, nogil=True)
def n_block_locs_at(x, y, size, K, L, block_starts, block_ends):
    x_index = np.uint32(x / size)
    y_index = np.uint32(y / size)
    step = 0
    for k in range(y_index - 1, y_index + 2):
        if 0 < k < K:
            for l in range(x_index - 1, x_index + 2):
                if 0 < l < L:
                    if step == 0:
                        n_block_locs = np.uint32(block_ends[k][l] - block_starts[k][l])
                        step = 1
                    else:
                        n_block_locs += np.uint32(block_ends[k][l] - block_starts[k][l])
    return n_block_locs


def get_block_locs_at(x, y, index_blocks):
    locs, size, _, _, block_starts, block_ends, K, L = index_blocks
    x_index = np.uint32(x / size)
    y_index = np.uint32(y / size)
    indices = []
    for k in range(y_index - 1, y_index + 2):
        if 0 <= k < K:
            for l in range(x_index - 1, x_index + 2):
                if 0 <= l < L:
                    indices.append(list(range(block_starts[k, l], block_ends[k, l])))
    indices = list(itertools.chain(*indices))
    return locs[indices]


@numba.jit(nopython=True, nogil=True)
def _fill_index_blocks(block_starts, block_ends, x_index, y_index):
    Y, X = block_starts.shape
    N = len(x_index)
    k = 0
    for i in range(Y):
        for j in range(X):
            k = _fill_index_block(
                block_starts, block_ends, N, x_index, y_index, i, j, k
            )


@numba.jit(nopython=True, nogil=True)
def _fill_index_block(block_starts, block_ends, N, x_index, y_index, i, j, k):
    block_starts[i, j] = k
    while k < N and y_index[k] == i and x_index[k] == j:
        k += 1
    block_ends[i, j] = k
    return k


def picked_locs(
    locs,
    width,
    height,
    picks,
    pick_shape,
    pick_size=None,
    add_group=True,
    callback=None,
    parallel=False,
):
    """Finds picked localisations.

    Parameters
    ----------
    locs : np.recarray
        Localization list.
    info : list of dicts
        Metadata of the localisations list.
    picks : list
        List of picks.
    pick_shape : {'Circle', 'Rectangle', 'Polygon'}
        Shape of the pick.
    pick_size : float (default=None)
        Size of the pick. Radius for the circles, width for the
        rectangles, None for the polygons.
    add_group : boolean (default=True)
        True if group id should be added to locs. Each pick will be
        assigned a different id.
    callback : function (default=None)
        Function to display progress. If "console", tqdm is used to
        display the progress. If None, no progress is displayed.
    parallel : bool (default=False)
        Whether to use parallel processing for Rectangle and Circle picks.
        Uses chunk-based processing - picks are distributed across
        worker processes in chunks, not one job per pick.
        Only beneficial for 8+ picks. Supports both Circle and Rectangle shapes.

    Returns
    -------
    picked_locs : list of np.recarrays
        List of np.recarrays, each containing locs from one pick.
    """

    # Use parallel processing for picks if requested
    if parallel and len(picks) >= 8:
        if pick_shape == "Rectangle":
            return _parallel_picked_locs_rectangle(
                locs, width, height, picks, pick_size, add_group, callback
            )
        elif pick_shape == "Circle":
            return _parallel_picked_locs_circle(
                locs, width, height, picks, pick_size, add_group, callback
            )

    if len(picks):
        picked_locs = []
        progress_bar_context = None
        progress = None
        if callback == "console":
            progress_bar_context = ProgressUtils.analysis_progress_bar(
                total=len(picks), desc="Picking locs"
            )
            progress = progress_bar_context.__enter__()

        if pick_shape == "Circle":
            index_blocks = get_index_blocks(locs, width, height, pick_size)
            for i, pick in enumerate(picks):
                x, y = pick
                block_locs = get_block_locs_at(x, y, index_blocks)
                group_locs = lib.locs_at(x, y, block_locs, pick_size)
                if add_group:
                    group = i * np.ones(len(group_locs), dtype=np.int32)
                    group_locs = lib.append_to_rec(group_locs, group, "group")
                group_locs.sort(kind="mergesort", order="frame")
                picked_locs.append(group_locs)

                if callback == "console":
                    progress.update(1)
                elif callback is not None:
                    callback(i + 1)

        elif pick_shape == "Rectangle":
            for i, pick in enumerate(picks):
                (xs, ys), (xe, ye) = pick
                X, Y = lib.get_pick_rectangle_corners(xs, ys, xe, ye, pick_size)
                x_min = min(X)
                x_max = max(X)
                y_min = min(Y)
                y_max = max(Y)
                group_locs = locs[locs.xc > x_min]
                group_locs = group_locs[group_locs.xc < x_max]
                group_locs = group_locs[group_locs.yc > y_min]
                group_locs = group_locs[group_locs.yc < y_max]
                group_locs = lib.locs_in_rectangle(group_locs, X, Y)
                # store rotated coordinates in x_rot and y_rot
                angle = 0.5 * np.pi - np.arctan2((ye - ys), (xe - xs))
                x_shifted = group_locs.xc - xs
                y_shifted = group_locs.yc - ys
                x_pick_rot = x_shifted * np.cos(angle) - y_shifted * np.sin(angle)
                y_pick_rot = x_shifted * np.sin(angle) + y_shifted * np.cos(angle)
                group_locs = lib.append_to_rec(group_locs, x_pick_rot, "x_pick_rot")
                group_locs = lib.append_to_rec(group_locs, y_pick_rot, "y_pick_rot")
                if add_group:
                    group = i * np.ones(len(group_locs), dtype=np.int32)
                    group_locs = lib.append_to_rec(group_locs, group, "group")
                group_locs.sort(kind="mergesort", order="frame")
                picked_locs.append(group_locs)

                if callback == "console":
                    progress.update(1)
                elif callback is not None:
                    callback(i + 1)

        elif pick_shape == "Polygon":
            for i, pick in enumerate(picks):
                X, Y = lib.get_pick_polygon_corners(pick)
                if X is None:
                    if callback == "console":
                        progress.update(1)
                    elif callback is not None:
                        callback(i + 1)
                    continue
                group_locs = locs[locs.xc > min(X)]
                group_locs = group_locs[group_locs.xc < max(X)]
                group_locs = group_locs[group_locs.yc > min(Y)]
                group_locs = group_locs[group_locs.yc < max(Y)]
                group_locs = lib.locs_in_polygon(group_locs, X, Y)
                if add_group:
                    group = i * np.ones(len(group_locs), dtype=np.int32)
                    group_locs = lib.append_to_rec(group_locs, group, "group")
                group_locs.sort(kind="mergesort", order="frame")
                picked_locs.append(group_locs)

                if callback == "console":
                    progress.update(1)
                elif callback is not None:
                    callback(i + 1)

        else:
            raise ValueError(
                "Invalid pick shape. Please choose from 'Circle', 'Rectangle', "
                "'Polygon'."
            )

        # Clean up progress bar
        if progress_bar_context is not None:
            progress_bar_context.__exit__(None, None, None)

        return picked_locs


@numba.jit(nopython=True, nogil=True, cache=True)
def pick_similar(
    x,
    y_shift,
    y_base,
    min_n_locs,
    max_n_locs,
    min_rmsd,
    max_rmsd,
    x_r,
    y_r1,
    y_r2,
    locs_xy,
    block_starts,
    block_ends,
    K,
    L,
    x_similar,
    y_similar,
    r,
    d2,
):
    for i, x_grid in enumerate(x):
        x_range = x_r[i]
        # y_grid is shifted for odd columns
        if i % 2:
            y = y_shift
            y_r = y_r1
        else:
            y = y_base
            y_r = y_r2
        for j, y_grid in enumerate(y):
            y_range = y_r[j]
            n_block_locs = _n_block_locs_at(
                x_range, y_range, K, L, block_starts, block_ends
            )
            if n_block_locs >= min_n_locs:
                block_locs_xy = _get_block_locs_at(
                    x_range,
                    y_range,
                    locs_xy,
                    block_starts,
                    block_ends,
                    K,
                    L,
                )
                picked_locs_xy = _locs_at(x_grid, y_grid, block_locs_xy, r)
                if picked_locs_xy.shape[1] > 1:
                    # Move to COM peak
                    x_test_old = x_grid
                    y_test_old = y_grid
                    x_test = np.mean(picked_locs_xy[0])
                    y_test = np.mean(picked_locs_xy[1])
                    count = 0
                    while (
                        np.abs(x_test - x_test_old) > 1e-3
                        or np.abs(y_test - y_test_old) > 1e-3
                    ):
                        count += 1
                        # skip the locs if the loop is too long
                        if count > 500:
                            break
                        x_test_old = x_test
                        y_test_old = y_test
                        picked_locs_xy = _locs_at(x_test, y_test, block_locs_xy, r)
                        if picked_locs_xy.shape[1] > 1:
                            x_test = np.mean(picked_locs_xy[0])
                            y_test = np.mean(picked_locs_xy[1])
                        else:
                            break
                    if np.all(
                        (x_similar - x_test) ** 2 + (y_similar - y_test) ** 2 > d2
                    ):
                        if min_n_locs <= picked_locs_xy.shape[1] <= max_n_locs:
                            if min_rmsd <= _rmsd_at_com(picked_locs_xy) <= max_rmsd:
                                x_similar = np.append(x_similar, x_test)
                                y_similar = np.append(y_similar, y_test)
    return x_similar, y_similar


@numba.jit(nopython=True, nogil=True)
def _n_block_locs_at(x_range, y_range, K, L, block_starts, block_ends, cache=True):
    step = 0
    for k in range(y_range - 1, y_range + 2):
        if 0 < k < K:
            for l in range(x_range - 1, x_range + 2):
                if 0 < l < L:
                    if step == 0:
                        n_block_locs = np.uint32(block_ends[k][l] - block_starts[k][l])
                        step = 1
                    else:
                        n_block_locs += np.uint32(block_ends[k][l] - block_starts[k][l])
    return n_block_locs


@numba.jit(nopython=True, nogil=True, cache=True)
def _get_block_locs_at(
    x_range,
    y_range,
    locs_xy,
    block_starts,
    block_ends,
    K,
    L,
):
    step = 0
    for k in range(y_range - 1, y_range + 2):
        if 0 < k < K:
            for l in range(x_range - 1, x_range + 2):
                if 0 < l < L:
                    if block_ends[k, l] - block_starts[k, l] > 0:
                        # numba does not work if you attach arange to an empty list so the first step is different
                        # this is because of dtype issues
                        if step == 0:
                            indices = np.arange(
                                float(block_starts[k, l]),
                                float(block_ends[k, l]),
                                dtype=np.uint32,
                            )
                            step = 1
                        else:
                            indices = np.concatenate(
                                (
                                    indices,
                                    np.arange(
                                        float(block_starts[k, l]),
                                        float(block_ends[k, l]),
                                        dtype=np.uint32,
                                    ),
                                )
                            )
    return locs_xy[:, indices]


@numba.jit(nopython=True, nogil=True, cache=True)
def _locs_at(x, y, locs_xy, r):
    dx = locs_xy[0] - x
    dy = locs_xy[1] - y
    r2 = r**2
    is_picked = dx**2 + dy**2 < r2
    return locs_xy[:, is_picked]


@numba.jit(nopython=True, nogil=True)
def _rmsd_at_com(locs_xy):
    com_x = np.mean(locs_xy[0])
    com_y = np.mean(locs_xy[1])
    return np.sqrt(np.mean((locs_xy[0] - com_x) ** 2 + (locs_xy[1] - com_y) ** 2))


@numba.jit(nopython=True, nogil=True)
def _distance_histogram(
    locs,
    bin_size,
    r_max,
    x_index,
    y_index,
    block_starts,
    block_ends,
    start,
    chunk,
):
    x = locs.xc
    y = locs.yc
    dh_len = np.uint32(r_max / bin_size)
    dh = np.zeros(dh_len, dtype=np.uint32)
    r_max_2 = r_max**2
    K, L = block_starts.shape
    end = min(start + chunk, len(locs))
    for i in range(start, end):
        xi = x[i]
        yi = y[i]
        ki = y_index[i]
        li = x_index[i]
        for k in range(ki, ki + 2):
            if k < K:
                for l in range(li, li + 2):
                    if l < L:
                        for j in range(block_starts[k, l], block_ends[k, l]):
                            if j > i:
                                dx2 = (xi - x[j]) ** 2
                                if dx2 < r_max_2:
                                    dy2 = (yi - y[j]) ** 2
                                    if dy2 < r_max_2:
                                        d = np.sqrt(dx2 + dy2)
                                        if d < r_max:
                                            bin = np.uint32(d / bin_size)
                                            if bin < dh_len:
                                                dh[bin] += 1
    return dh


def distance_histogram(locs, info, bin_size, r_max):
    locs, size, x_index, y_index, b_starts, b_ends, K, L = get_index_blocks(
        locs, info, r_max
    )
    N = len(locs)
    n_threads = min(
        60, max(1, int(0.75 * mp.cpu_count()))
    )  # Python crashes when using >64 cores
    chunk = int(N / n_threads)
    starts = range(0, N, chunk)
    args = [
        (
            locs,
            bin_size,
            r_max,
            x_index,
            y_index,
            b_starts,
            b_ends,
            start,
            chunk,
        )
        for start in starts
    ]
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(_distance_histogram, *_) for _ in args]
    results = [future.result() for future in futures]
    return np.sum(results, axis=0)


def nena(locs, info, callback=None):
    bin_centers, dnfl_ = next_frame_neighbor_distance_histogram(locs, callback)

    def func(d, delta_a, s, ac, dc, sc):
        a = ac + delta_a  # make sure a >= ac
        p_single = a * (d / (2 * s**2)) * np.exp(-(d**2) / (4 * s**2))
        p_short = ac / (sc * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((d - dc) / sc) ** 2)
        return p_single + p_short

    pdf_model = lmfit.Model(func)
    params = lmfit.Parameters()
    area = np.trapz(dnfl_, bin_centers)
    median_lp = np.mean(
        [np.median(locs.xc_err), np.median(locs.yc_err)]
    )  # Changed to _err convention
    params.add("delta_a", value=0.8 * area, min=0)
    params.add("s", value=median_lp, min=0)
    params.add("ac", value=0.1 * area, min=0)
    params.add("dc", value=2 * median_lp, min=0)
    params.add("sc", value=median_lp, min=0)
    result = pdf_model.fit(dnfl_, params, d=bin_centers)
    return result, result.best_values["s"]


def next_frame_neighbor_distance_histogram(locs, callback=None):
    locs.sort(kind="mergesort", order="frame")
    frame = locs.frame
    x = locs.xc
    y = locs.yc
    if hasattr(locs, "group"):
        group = locs.group
    else:
        group = np.zeros(len(locs), dtype=np.int32)
    bin_size = 0.001
    d_max = 1.0
    return _nfndh(frame, x, y, group, d_max, bin_size, callback)


def _nfndh(frame, x, y, group, d_max, bin_size, callback=None):
    N = len(frame)
    bins = np.arange(0, d_max, bin_size)
    dnfl = np.zeros(len(bins))
    one_percent = int(N / 100)
    starts = one_percent * np.arange(100)
    for k, start in enumerate(starts):
        for i in range(start, start + one_percent):
            _fill_dnfl(N, frame, x, y, group, i, d_max, dnfl, bin_size)
        if callback is not None:
            callback(k + 1)
    bin_centers = bins + bin_size / 2
    return bin_centers, dnfl


@numba.jit(nopython=True)
def _fill_dnfl(N, frame, x, y, group, i, d_max, dnfl, bin_size):
    frame_i = frame[i]
    x_i = x[i]
    y_i = y[i]
    group_i = group[i]
    min_frame = frame_i + 1
    for min_index in range(i + 1, N):
        if frame[min_index] >= min_frame:
            break
    max_frame = frame_i + 1
    for max_index in range(min_index, N):
        if frame[max_index] > max_frame:
            break
    d_max_2 = d_max**2
    for j in range(min_index, max_index):
        if group[j] == group_i:
            dx2 = (x_i - x[j]) ** 2
            if dx2 <= d_max_2:
                dy2 = (y_i - y[j]) ** 2
                if dy2 <= d_max_2:
                    d = np.sqrt(dx2 + dy2)
                    if d <= d_max:
                        bin = int(d / bin_size)
                        dnfl[bin] += 1


def pair_correlation(locs, info, bin_size, r_max):
    dh = distance_histogram(locs, info, bin_size, r_max)
    # Start with r-> otherwise area will be 0
    bins_lower = np.arange(bin_size, r_max + bin_size, bin_size)

    if bins_lower.shape[0] > dh.shape[0]:
        bins_lower = bins_lower[:-1]
    area = np.pi * bin_size * (2 * bins_lower + bin_size)
    return bins_lower, dh / area


@numba.jit(nopython=True, nogil=True)
def _local_density(
    locs, radius, x_index, y_index, block_starts, block_ends, start, chunk
):
    x = locs.xc
    y = locs.yc
    N = len(x)
    r2 = radius**2
    end = min(start + chunk, N)
    density = np.zeros(N, dtype=np.uint32)
    for i in range(start, end):
        yi = y[i]
        xi = x[i]
        ki = y_index[i]
        li = x_index[i]
        di = 0
        for k in range(ki - 1, ki + 2):
            for l in range(li - 1, li + 2):
                j_min = block_starts[k, l]
                j_max = block_ends[k, l]
                for j in range(j_min, j_max):
                    dx2 = (xi - x[j]) ** 2
                    if dx2 < r2:
                        dy2 = (yi - y[j]) ** 2
                        if dy2 < r2:
                            d2 = dx2 + dy2
                            if d2 < r2:
                                di += 1
        density[i] = di
    return density


def compute_local_density(locs, info, radius):
    locs, x_index, y_index, block_starts, block_ends, K, L = get_index_blocks(
        locs, info, radius
    )
    N = len(locs)
    n_threads = min(
        60, max(1, int(0.75 * mp.cpu_count()))
    )  # Python crashes when using >64 cores
    chunk = int(N / n_threads)
    starts = range(0, N, chunk)
    args = [
        (
            locs,
            radius,
            x_index,
            y_index,
            block_starts,
            block_ends,
            start,
            chunk,
        )
        for start in starts
    ]
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(_local_density, *_) for _ in args]
    density = np.sum([future.result() for future in futures], axis=0)
    locs = lib.remove_from_rec(locs, "density")
    return lib.append_to_rec(locs, density, "density")


def compute_dark_times(locs, group=None):

    if "len" not in locs.dtype.names:
        raise AttributeError("Length not found. Please link localisations first.")
    dark = dark_times(locs, group)
    locs = lib.append_to_rec(locs, np.int32(dark), "dark")
    locs = locs[locs.dark != -1]
    return locs


def dark_times(locs, group=None):
    last_frame = locs.frame + locs.len - 1
    if group is None:
        if hasattr(locs, "group"):
            group = locs.group
        else:
            group = np.zeros(len(locs))
    dark = _dark_times(locs, group, last_frame)
    return dark


@numba.jit(nopython=True)
def _dark_times(locs, group, last_frame):
    N = len(locs)
    max_frame = locs.frame.max()
    dark = max_frame * np.ones(len(locs), dtype=np.int32)
    for i in range(N):
        for j in range(N):
            if (group[i] == group[j]) and (i != j):
                dark_ij = locs.frame[i] - last_frame[j]
                if (dark_ij > 0) and (dark_ij < dark[i]):
                    dark[i] = dark_ij
    for i in range(N):
        if dark[i] == max_frame:
            dark[i] = -1
    return dark


def link(
    locs,
    info,
    r_max=0.05,
    max_dark_time=1,
    combine_mode="average",
    remove_ambiguous_lengths=True,
):
    if len(locs) == 0:
        linked_locs = locs.copy()
        if hasattr(locs, "frame"):
            linked_locs = lib.append_to_rec(
                linked_locs, np.array([], dtype=np.int32), "len"
            )
            linked_locs = lib.append_to_rec(
                linked_locs, np.array([], dtype=np.int32), "n"
            )
        if hasattr(locs, "photons"):
            linked_locs = lib.append_to_rec(
                linked_locs, np.array([], dtype=np.float32), "photon_rate"
            )
    else:
        locs.sort(kind="mergesort", order="frame")
        if hasattr(locs, "group"):
            group = locs.group
        else:
            group = np.zeros(len(locs), dtype=np.int32)
        link_group = get_link_groups(locs, r_max, max_dark_time, group)
        if combine_mode == "average":
            linked_locs = link_loc_groups(
                locs,
                info,
                link_group,
                remove_ambiguous_lengths=remove_ambiguous_lengths,
            )
        elif combine_mode == "refit":
            pass  # TODO
    return linked_locs


def weighted_variance(locs):
    n = len(locs)
    w = locs.photons
    x = locs.xc
    y = locs.yc
    xWbarx = np.average(locs.xc, weights=w)
    xWbary = np.average(locs.yc, weights=w)
    wbarx = np.mean(locs.xc_err)  # Changed to _err convention
    wbary = np.mean(locs.yc_err)  # Changed to _err convention
    variance_x = (
        n
        / ((n - 1) * sum(w) ** 2)
        * (
            sum((w * x - wbarx * xWbarx) ** 2)
            - 2 * xWbarx * sum((w - wbarx) * (w * x - wbarx * xWbarx))
            + xWbarx**2 * sum((w - wbarx) ** 2)
        )
    )
    variance_y = (
        n
        / ((n - 1) * sum(w) ** 2)
        * (
            sum((w * y - wbary * xWbary) ** 2)
            - 2 * xWbary * sum((w - wbary) * (w * y - wbary * xWbary))
            + xWbary**2 * sum((w - wbary) ** 2)
        )
    )
    return variance_x, variance_y


# Combine localisations: calculate the properties of the group
def cluster_combine(locs):
    print("Combining localisations...")
    combined_locs = []
    unique_groups = np.unique(locs["group"])
    with ProgressUtils.analysis_progress_bar(
        total=len(unique_groups), desc="Combining localisations"
    ) as pbar:
        for group in unique_groups:
            temp = locs[locs["group"] == group]
        cluster = np.unique(temp["cluster"])
        n_cluster = len(cluster)
        mean_frame = np.zeros(n_cluster)
        std_frame = np.zeros(n_cluster)
        com_x = np.zeros(n_cluster)
        com_y = np.zeros(n_cluster)
        std_x = np.zeros(n_cluster)
        std_y = np.zeros(n_cluster)
        group_id = np.zeros(n_cluster)
        n = np.zeros(n_cluster, dtype=np.int32)
        for i, clusterval in enumerate(cluster):
            cluster_locs = temp[temp["cluster"] == clusterval]
            mean_frame[i] = np.mean(cluster_locs.frame)
            com_x[i] = np.average(cluster_locs.xc, weights=cluster_locs.photons)
            com_y[i] = np.average(cluster_locs.yc, weights=cluster_locs.photons)
            std_frame[i] = np.std(cluster_locs.frame)
            std_x[i] = np.std(cluster_locs.xc) / np.sqrt(len(cluster_locs))
            std_y[i] = np.std(cluster_locs.yc) / np.sqrt(len(cluster_locs))
            n[i] = len(cluster_locs)
            group_id[i] = group
        clusters = np.rec.array(
            (
                group_id,
                cluster,
                mean_frame,
                com_x,
                com_y,
                std_frame,
                std_x,
                std_y,
                n,
            ),
            dtype=[
                ("group", group.dtype),
                ("cluster", cluster.dtype),
                ("mean_frame", "f4"),
                ("xc", "f4"),  # Changed from "x" to "xc"
                ("yc", "f4"),  # Changed from "y" to "yc"
                ("std_frame", "f4"),
                ("xc_err", "f4"),  # Changed from "lpx" to "xc_err"
                ("yc_err", "f4"),  # Changed from "lpy" to "yc_err"
                ("n", "i4"),
            ],
        )
        combined_locs.append(clusters)

    combined_locs = stack_arrays(combined_locs, asrecarray=True, usemask=False)

    return combined_locs


def cluster_combine_dist(locs):
    print("Calculating distances...")
    print("XY")
    combined_locs = []
    with ProgressUtils.analysis_progress_bar(
        total=len(np.unique(locs["group"])), desc="Calculating distances"
    ) as pbar:
        for group in np.unique(locs["group"]):
            temp = locs[locs["group"] == group]
            cluster = np.unique(temp["cluster"])
            n_cluster = len(cluster)
            mean_frame = temp["mean_frame"]
            std_frame = temp["std_frame"]
            com_x = temp["xc"]  # Changed from "x" to "xc"
            com_y = temp["yc"]  # Changed from "y" to "yc"
            std_x = temp["xc_err"]  # Changed from "lpx" to "xc_err"
            std_y = temp["yc_err"]  # Changed from "lpy" to "yc_err"
            group_id = temp["group"]
            n = temp["n"]
            min_dist = np.zeros(n_cluster)

            for i, clusterval in enumerate(cluster):
                # find nearest neighbor in xyz
                group_locs = temp[temp["cluster"] != clusterval]
                cluster_locs = temp[temp["cluster"] == clusterval]
                ref_point_xy = np.array([cluster_locs.xc, cluster_locs.yc])
                all_points_xy = np.array([group_locs.xc, group_locs.yc])
                distances_xy = distance.cdist(
                    ref_point_xy.transpose(), all_points_xy.transpose()
                )
                min_dist[i] = np.amin(distances_xy)

            clusters = np.rec.array(
                (
                    group_id,
                    cluster,
                    mean_frame,
                    com_x,
                    com_y,
                    std_frame,
                    std_x,
                    std_y,
                    n,
                    min_dist,
                ),
                dtype=[
                    ("group", group.dtype),
                    ("cluster", cluster.dtype),
                    ("mean_frame", "f4"),
                    ("xc", "f4"),  # Changed from "x" to "xc"
                    ("yc", "f4"),  # Changed from "y" to "yc"
                    ("std_frame", "f4"),
                    ("xc_err", "f4"),  # Changed from "lpx" to "xc_err"
                    ("yc_err", "f4"),  # Changed from "lpy" to "yc_err"
                    ("n", "i4"),
                    ("min_dist", "f4"),
                ],
            )
            combined_locs.append(clusters)
            pbar.update(1)

    combined_locs = stack_arrays(combined_locs, asrecarray=True, usemask=False)
    return combined_locs


@numba.jit(nopython=True)
def get_link_groups(locs, d_max, max_dark_time, group):
    """Assumes that locs are sorted by frame"""
    frame = locs.frame
    x = locs.xc
    y = locs.yc
    N = len(x)
    link_group = -np.ones(N, dtype=np.int32)
    current_link_group = -1
    for i in range(N):
        if link_group[i] == -1:  # loc has no group yet
            current_link_group += 1
            link_group[i] = current_link_group
            current_index = i
            next_loc_index_in_group = _get_next_loc_index_in_link_group(
                current_index,
                link_group,
                N,
                frame,
                x,
                y,
                d_max,
                max_dark_time,
                group,
            )
            while next_loc_index_in_group != -1:
                link_group[next_loc_index_in_group] = current_link_group
                current_index = next_loc_index_in_group
                next_loc_index_in_group = _get_next_loc_index_in_link_group(
                    current_index,
                    link_group,
                    N,
                    frame,
                    x,
                    y,
                    d_max,
                    max_dark_time,
                    group,
                )
    return link_group


@numba.jit(nopython=True)
def _get_next_loc_index_in_link_group(
    current_index, link_group, N, frame, x, y, d_max, max_dark_time, group
):
    current_frame = frame[current_index]
    current_x = x[current_index]
    current_y = y[current_index]
    current_group = group[current_index]
    min_frame = current_frame + 1
    for min_index in range(current_index + 1, N):
        if frame[min_index] >= min_frame:
            break
    max_frame = current_frame + max_dark_time + 1
    for max_index in range(min_index, N):
        if frame[max_index] > max_frame:
            break
    else:
        max_index = N
    d_max_2 = d_max**2
    for j in range(min_index, max_index):
        if group[j] == current_group:
            if link_group[j] == -1:
                dx2 = (current_x - x[j]) ** 2
                if dx2 <= d_max_2:
                    dy2 = (current_y - y[j]) ** 2
                    if dy2 <= d_max_2:
                        if dx2 + dy2 <= d_max_2:
                            return j
    return -1


@numba.jit(nopython=True)
def _link_group_count(link_group, n_locs, n_groups):
    result = np.zeros(n_groups, dtype=np.uint32)
    for i in range(n_locs):
        i_ = link_group[i]
        result[i_] += 1
    return result


@numba.jit(nopython=True)
def _link_group_sum(column, link_group, n_locs, n_groups):
    result = np.zeros(n_groups, dtype=column.dtype)
    for i in range(n_locs):
        i_ = link_group[i]
        result[i_] += column[i]
    return result


@numba.jit(nopython=True)
def _link_group_mean(column, link_group, n_locs, n_groups, n_locs_per_group):
    group_sum = _link_group_sum(column, link_group, n_locs, n_groups)
    result = np.empty(
        n_groups, dtype=np.float32
    )  # this ensures float32 after the division
    result[:] = group_sum / n_locs_per_group
    return result


@numba.jit(nopython=True)
def _link_group_weighted_mean(
    column, weights, link_group, n_locs, n_groups, n_locs_per_group
):
    sum_weights = _link_group_sum(weights, link_group, n_locs, n_groups)
    return (
        _link_group_mean(column * weights, link_group, n_locs, n_groups, sum_weights),
        sum_weights,
    )


@numba.jit(nopython=True)
def _link_group_min_max(column, link_group, n_locs, n_groups):
    min_ = np.empty(n_groups, dtype=column.dtype)
    max_ = np.empty(n_groups, dtype=column.dtype)
    min_[:] = column.max()
    max_[:] = column.min()
    for i in range(n_locs):
        i_ = link_group[i]
        value = column[i]
        if value < min_[i_]:
            min_[i_] = value
        if value > max_[i_]:
            max_[i_] = value
    return min_, max_


@numba.jit(nopython=True)
def _link_group_last(column, link_group, n_locs, n_groups):
    result = np.zeros(n_groups, dtype=column.dtype)
    for i in range(n_locs):
        i_ = link_group[i]
        result[i_] = column[i]
    return result


def link_loc_groups(locs, info, link_group, remove_ambiguous_lengths=True):
    n_locs = len(link_group)
    n_groups = link_group.max() + 1
    n_ = _link_group_count(link_group, n_locs, n_groups)
    columns = OrderedDict()
    if hasattr(locs, "frame"):
        first_frame_, last_frame_ = _link_group_min_max(
            locs.frame, link_group, n_locs, n_groups
        )
        columns["frame"] = first_frame_
    if hasattr(locs, "xc"):
        weights_x = 1 / locs.xc_err**2  # Changed to _err convention
        columns["xc"], sum_weights_x_ = _link_group_weighted_mean(
            locs.xc, weights_x, link_group, n_locs, n_groups, n_
        )
    if hasattr(locs, "yc"):
        weights_y = 1 / locs.yc_err**2  # Changed to _err convention
        columns["yc"], sum_weights_y_ = _link_group_weighted_mean(
            locs.yc, weights_y, link_group, n_locs, n_groups, n_
        )
    if hasattr(locs, "photons"):
        columns["photons"] = _link_group_sum(locs.photons, link_group, n_locs, n_groups)
    if hasattr(locs, "s_x"):
        columns["s_x"] = _link_group_mean(locs.s_x, link_group, n_locs, n_groups, n_)
    if hasattr(locs, "s_y"):
        columns["s_y"] = _link_group_mean(locs.s_y, link_group, n_locs, n_groups, n_)
    if hasattr(locs, "bg"):
        columns["bg"] = _link_group_sum(locs.bg, link_group, n_locs, n_groups)
    if hasattr(locs, "xc"):  # Changed from "x" to "xc"
        columns["xc_err"] = np.sqrt(
            1 / sum_weights_x_
        )  # Changed from "lpx" to "xc_err"
    if hasattr(locs, "yc"):  # Changed from "y" to "yc"
        columns["yc_err"] = np.sqrt(
            1 / sum_weights_y_
        )  # Changed from "lpy" to "yc_err"
    if hasattr(locs, "ellipticity"):
        columns["ellipticity"] = _link_group_mean(
            locs.ellipticity, link_group, n_locs, n_groups, n_
        )
    if hasattr(locs, "net_gradient"):
        columns["net_gradient"] = _link_group_mean(
            locs.net_gradient, link_group, n_locs, n_groups, n_
        )
    if hasattr(locs, "likelihood"):
        columns["likelihood"] = _link_group_mean(
            locs.likelihood, link_group, n_locs, n_groups, n_
        )
    if hasattr(locs, "iterations"):
        columns["iterations"] = _link_group_mean(
            locs.iterations, link_group, n_locs, n_groups, n_
        )
    if hasattr(locs, "z"):
        columns["z"] = _link_group_mean(locs.z, link_group, n_locs, n_groups, n_)
    if hasattr(locs, "d_zcalib"):
        columns["d_zcalib"] = _link_group_mean(
            locs.d_zcalib, link_group, n_locs, n_groups, n_
        )
    if hasattr(locs, "group"):
        columns["group"] = _link_group_last(locs.group, link_group, n_locs, n_groups)
    if hasattr(locs, "frame"):
        columns["len"] = last_frame_ - first_frame_ + 1
    columns["n"] = n_
    if hasattr(locs, "photons"):
        columns["photon_rate"] = np.float32(columns["photons"] / n_)
    linked_locs = np.rec.array(list(columns.values()), names=list(columns.keys()))
    if remove_ambiguous_lengths:
        valid = np.logical_and(first_frame_ > 0, last_frame_ < info[0]["Frames"])
        linked_locs = linked_locs[valid]
    return linked_locs


def localisation_precision(photons, s, bg, em):
    """
    Calculates the theoretical localisation precision according to
    Mortensen et al., Nat Meth, 2010 for a 2D unweighted Gaussian fit.
    """
    s2 = s**2
    sa2 = s2 + 1 / 12
    v = sa2 * (16 / 9 + (8 * np.pi * sa2 * bg) / photons) / photons
    if em:
        v *= 2
    with np.errstate(invalid="ignore"):
        return np.sqrt(v)


def n_segments(info, segmentation):
    n_frames = info[0]["Frames"]
    return int(np.round(n_frames / segmentation))


def segment(locs, info, segmentation, kwargs={}, callback=None):
    Y = info[0]["Height"]
    X = info[0]["Width"]
    n_frames = info[0]["Frames"]
    n_seg = n_segments(info, segmentation)
    bounds = np.linspace(0, n_frames - 1, n_seg + 1, dtype=np.uint32)
    segments = np.zeros((n_seg, Y, X))
    if callback is None:
        with ProgressUtils.analysis_progress_bar(
            total=n_seg, desc="Generating segments"
        ) as pbar:
            for i in range(n_seg):
                segment_locs = locs[
                    (locs.frame >= bounds[i]) & (locs.frame < bounds[i + 1])
                ]
                _, segments[i] = render.render(segment_locs, info, **kwargs)
                pbar.update(1)
    else:
        callback(0)
        for i in range(n_seg):
            segment_locs = locs[
                (locs.frame >= bounds[i]) & (locs.frame < bounds[i + 1])
            ]
            _, segments[i] = render.render(segment_locs, info, **kwargs)
            callback(i + 1)
    return bounds, segments


def undrift(
    locs,
    info,
    segmentation,
    display=True,
    save_path=None,
    segmentation_callback=None,
    rcc_callback=None,
):
    """Undrift by RCC.

    Parameters
    ----------
    locs : np.recarray
        Localization data
    info : list
        Metadata
    segmentation : int
        Segmentation parameter
    display : bool, optional
        Whether to display drift analysis plot (default: True)
    save_path : str, optional
        Path to save drift analysis plot (default: None)
    segmentation_callback : callable, optional
        Callback for segmentation progress
    rcc_callback : callable, optional
        Callback for RCC progress

    Returns
    -------
    drift : np.recarray
        Calculated drift values
    locs : np.recarray
        Undrifted localisation data
    """

    bounds, segments = segment(
        locs,
        info,
        segmentation,
        {"blur_method": "gaussian", "min_blur_width": 1},
        segmentation_callback,
    )
    shift_y, shift_x = imageprocess.rcc(segments, 32, rcc_callback)
    t = (bounds[1:] + bounds[:-1]) / 2
    drift_x_pol = interpolate.InterpolatedUnivariateSpline(t, shift_x, k=3)
    drift_y_pol = interpolate.InterpolatedUnivariateSpline(t, shift_y, k=3)
    t_inter = np.arange(info[0]["Frames"])
    drift = (drift_x_pol(t_inter), drift_y_pol(t_inter))
    drift = np.rec.array(drift, dtype=[("x", "f"), ("y", "f")])

    if display:
        _plot_drift_analysis(drift, shift_x, shift_y, bounds, save_path=save_path)

    locs.xc -= drift.x[locs.frame]
    locs.yc -= drift.y[locs.frame]
    return drift, locs


def undrift_from_picked(picked_locs, n_frames):
    """Finds drift from picked localisations. Note that unlike other
    undrifting functions, this function does not return undrifted
    localisations but only drift."""

    drift_x = _undrift_from_picked_coordinate(picked_locs, n_frames, "xc")
    drift_y = _undrift_from_picked_coordinate(picked_locs, n_frames, "yc")

    # A rec array to store the applied drift
    drift = (drift_x, drift_y)
    drift = np.rec.array(drift, dtype=[("xc", "f"), ("yc", "f")])
    return drift


def _undrift_from_picked_coordinate(picked_locs, n_frames, coordinate):
    """Calculates drift in a given coordinate.

    Parameters
    ----------
    picked_locs : list
        List of np.recarrays with locs for each pick.
    n_frames : number of frames
    coordinate : {"x", "y", "z"}
        Spatial coordinate where drift is to be found.

    Returns
    -------
    drift_mean : np.array
        Average drift across picks for all frames
    """

    n_picks = len(picked_locs)

    # Drift per pick per frame
    drift = np.empty((n_picks, n_frames))
    drift.fill(np.nan)

    # Remove center of mass offset
    for i, locs in enumerate(picked_locs):
        coordinates = getattr(locs, coordinate)
        drift[i, locs.frame.astype(int)] = coordinates - np.mean(coordinates)

    # Mean drift over picks
    drift_mean = np.nanmean(drift, 0)
    # Square deviation of each pick's drift to mean drift along frames
    sd = (drift - drift_mean) ** 2
    # Mean of square deviation for each pick
    msd = np.nanmean(sd, 1)
    # New mean drift over picks
    # where each pick is weighted according to its msd
    nan_mask = np.isnan(drift)
    drift = np.ma.MaskedArray(drift, mask=nan_mask)
    drift_mean = np.ma.average(drift, axis=0, weights=1 / msd)
    drift_mean = drift_mean.filled(np.nan)

    # Linear interpolation for frames without localisations
    def nan_helper(y):
        return np.isnan(y), lambda z: z.nonzero()[0]

    nans, nonzero = nan_helper(drift_mean)
    drift_mean[nans] = np.interp(nonzero(nans), nonzero(~nans), drift_mean[~nans])
    return drift_mean


def align(locs, infos, display=False):
    images = []
    for i, (locs_, info_) in enumerate(zip(locs, infos)):
        _, image = render.render(locs_, info_, blur_method="smooth")
        images.append(image)
    shift_y, shift_x = imageprocess.rcc(images)
    print("Image x shifts: {}".format(shift_x))
    print("Image y shifts: {}".format(shift_y))
    for i, (locs_, dx, dy) in enumerate(zip(locs, shift_x, shift_y)):
        locs_.yc -= dy
        locs_.xc -= dx
    return locs


def groupprops(locs, callback=None):
    try:
        locs = locs[locs.dark != -1]
    except AttributeError:
        pass
    group_ids = np.unique(locs.group)
    n = len(group_ids)
    n_cols = len(locs.dtype)
    names = ["group", "n_events"] + list(
        itertools.chain(*[(_ + "_mean", _ + "_std") for _ in locs.dtype.names])
    )
    formats = ["i4", "i4"] + 2 * n_cols * ["f4"]
    groups = np.recarray(n, formats=formats, names=names)
    if callback is not None:
        callback(0)
        for i, group_id in enumerate(group_ids):
            group_locs = locs[locs.group == group_id]
            groups["group"][i] = group_id
            groups["n_events"][i] = len(group_locs)
            for name in locs.dtype.names:
                groups[name + "_mean"][i] = np.mean(group_locs[name])
                groups[name + "_std"][i] = np.std(group_locs[name])
            callback(i + 1)
    else:
        with ProgressUtils.analysis_progress_bar(
            total=len(group_ids), desc="Calculating group statistics"
        ) as pbar:
            for i, group_id in enumerate(group_ids):
                group_locs = locs[locs.group == group_id]
                groups["group"][i] = group_id
                groups["n_events"][i] = len(group_locs)
                for name in locs.dtype.names:
                    groups[name + "_mean"][i] = np.mean(group_locs[name])
                    groups[name + "_std"][i] = np.std(group_locs[name])
                pbar.update(1)
    return groups


def nn_analysis(
    x1,
    x2,
    y1,
    y2,
    z1,
    z2,
    nn_count,
    same_channel,
):
    # coordinates are in nm
    if z1 is not None:  # 3D
        input1 = np.stack((x1, y1, z1)).T
        input2 = np.stack((x2, y2, z2)).T
    else:  # 2D
        input1 = np.stack((x1, y1)).T
        input2 = np.stack((x2, y2)).T
    if same_channel:
        model = NN(n_neighbors=nn_count + 1)
    else:
        model = NN(n_neighbors=nn_count)
    model.fit(input1)
    nn, _ = model.kneighbors(input2)
    if same_channel:
        nn = nn[:, 1:]  # ignore the zero distance
    return nn


def _parallel_picked_locs_rectangle(
    locs, width, height, picks, pick_size, add_group=True, callback=None
):
    """
    Parallelised version of rectangle picking for picked_locs.

    Uses ThreadPoolExecutor with shared memory for efficient parallel processing.
    Much faster than ProcessPoolExecutor due to reduced serialization overhead.

    Args:
        locs: Localisation data
        width: Image width
        height: Image height
        picks: List of rectangle picks in format [((xs,ys), (xe,ye)), ...]
        pick_size: Size of the pick region
        add_group: Whether to add group IDs to localisations
        callback: Progress callback function

    Returns:
        List of localisation arrays, one per pick
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import multiprocessing as mp
    import math

    if len(picks) == 0:
        return []

    # Only use parallel processing if we have enough picks to make it worthwhile
    if len(picks) < 8:
        return _serial_picked_locs_rectangle(
            locs, width, height, picks, pick_size, add_group, callback
        )

    try:
        # Calculate optimal thread count - use more threads since they're lightweight
        n_threads = min(mp.cpu_count() * 2, len(picks), 16)  # Up to 16 threads

        # Set up progress tracking
        progress_bar_context = None
        progress = None
        total_completed = 0

        if callback == "console":
            import ProgressUtils

            progress_bar_context = ProgressUtils.analysis_progress_bar(
                total=len(picks), desc="Picking locs (parallel threads)"
            )
            progress = progress_bar_context.__enter__()

        try:
            # Process individual picks with ThreadPool (no chunking needed)
            with ThreadPoolExecutor(max_workers=n_threads) as executor:
                # Submit individual pick processing jobs
                future_to_index = {}
                for i, pick in enumerate(picks):
                    future = executor.submit(
                        _process_single_rectangle_pick,
                        locs,
                        i,
                        pick,
                        pick_size,
                        add_group,
                    )
                    future_to_index[future] = i

                # Collect results as they complete (more efficient than waiting sequentially)
                picked_locs = [None] * len(picks)

                for future in as_completed(future_to_index):
                    try:
                        original_idx, result = future.result()
                        picked_locs[original_idx] = result

                        total_completed += 1

                        # Update progress
                        if callback == "console" and progress:
                            progress.update(1)
                        elif callback is not None:
                            callback(total_completed)

                    except Exception as e:
                        # Handle individual pick failures gracefully
                        original_idx = future_to_index[future]
                        print(
                            f"Warning: Processing pick {original_idx} failed with error: {e}"
                        )
                        picked_locs[original_idx] = np.array([], dtype=locs.dtype).view(
                            np.recarray
                        )

                        if callback == "console" and progress:
                            progress.update(1)
                        elif callback is not None:
                            callback(total_completed)

            # Fill any remaining None positions (shouldn't happen, but safety check)
            for i in range(len(picked_locs)):
                if picked_locs[i] is None:
                    picked_locs[i] = np.array([], dtype=locs.dtype).view(np.recarray)

            return picked_locs

        finally:
            # Cleanup progress bar
            if progress_bar_context:
                progress_bar_context.__exit__(None, None, None)

    except Exception as e:
        print(f"Parallel processing failed ({e}), falling back to serial processing")
        # Fall back to serial processing
        return _serial_picked_locs_rectangle(
            locs, width, height, picks, pick_size, add_group, callback
        )


def _process_single_rectangle_pick(locs, original_idx, pick, pick_size, add_group):
    """
    Process a single rectangle pick (efficient thread-based processing).

    Args:
        locs: Localisation data (shared in memory across threads)
        original_idx: Original index of this pick
        pick: Rectangle pick in format ((xs,ys), (xe,ye))
        pick_size: Size of the pick region
        add_group: Whether to add group IDs

    Returns:
        Tuple of (original_index, filtered_localisations)
    """
    try:
        # Import required modules
        import lib
        import numpy as np

        # Process the single pick
        (xs, ys), (xe, ye) = pick

        # Get rectangle corners
        X, Y = lib.get_pick_rectangle_corners(xs, ys, xe, ye, pick_size)
        x_min, x_max = min(X), max(X)
        y_min, y_max = min(Y), max(Y)

        # Filter localisations by bounding box (fast pre-filter)
        group_locs = locs[
            (locs.xc > x_min)
            & (locs.xc < x_max)
            & (locs.yc > y_min)
            & (locs.yc < y_max)
        ]

        # Apply precise rectangle filtering
        group_locs = lib.locs_in_rectangle(group_locs, X, Y)

        # Add rotated coordinates
        angle = 0.5 * np.pi - np.arctan2((ye - ys), (xe - xs))
        x_shifted = group_locs.xc - xs
        y_shifted = group_locs.yc - ys
        x_pick_rot = x_shifted * np.cos(angle) - y_shifted * np.sin(angle)
        y_pick_rot = x_shifted * np.sin(angle) + y_shifted * np.cos(angle)

        group_locs = lib.append_to_rec(group_locs, x_pick_rot, "x_pick_rot")
        group_locs = lib.append_to_rec(group_locs, y_pick_rot, "y_pick_rot")

        # Add group ID if requested (use original index for group ID)
        if add_group:
            group = original_idx * np.ones(len(group_locs), dtype=np.int32)
            group_locs = lib.append_to_rec(group_locs, group, "group")

        # Sort by frame
        group_locs.sort(kind="mergesort", order="frame")

        return (original_idx, group_locs)

    except Exception as e:
        # Return empty result on error
        return (original_idx, np.array([], dtype=locs.dtype).view(np.recarray))


def _process_rectangle_pick_chunk(
    locs, chunk_indices, chunk_picks, pick_size, add_group
):
    """
    DEPRECATED: Process a chunk of rectangle picks (efficient chunk-based multiprocessing).

    This function is kept for backward compatibility but is no longer used.
    The new ThreadPoolExecutor approach processes individual picks instead of chunks.
    """
    try:
        # Import required modules (needed in each worker process)
        import lib
        import numpy as np

        results = []

        # Process each pick in the chunk
        for i, (original_idx, pick) in enumerate(zip(chunk_indices, chunk_picks)):
            try:
                (xs, ys), (xe, ye) = pick

                # Get rectangle corners
                X, Y = lib.get_pick_rectangle_corners(xs, ys, xe, ye, pick_size)
                x_min, x_max = min(X), max(X)
                y_min, y_max = min(Y), max(Y)

                # Filter localisations by bounding box
                group_locs = locs[
                    (locs.xc > x_min)
                    & (locs.xc < x_max)
                    & (locs.yc > y_min)
                    & (locs.yc < y_max)
                ]

                # Apply precise rectangle filtering
                group_locs = lib.locs_in_rectangle(group_locs, X, Y)

                # Add rotated coordinates
                angle = 0.5 * np.pi - np.arctan2((ye - ys), (xe - xs))
                x_shifted = group_locs.xc - xs
                y_shifted = group_locs.yc - ys
                x_pick_rot = x_shifted * np.cos(angle) - y_shifted * np.sin(angle)
                y_pick_rot = x_shifted * np.sin(angle) + y_shifted * np.cos(angle)

                group_locs = lib.append_to_rec(group_locs, x_pick_rot, "x_pick_rot")
                group_locs = lib.append_to_rec(group_locs, y_pick_rot, "y_pick_rot")

                # Add group ID if requested (use original index for group ID)
                if add_group:
                    group = original_idx * np.ones(len(group_locs), dtype=np.int32)
                    group_locs = lib.append_to_rec(group_locs, group, "group")

                # Sort by frame
                group_locs.sort(kind="mergesort", order="frame")

                results.append((original_idx, group_locs))

            except Exception as e:
                # Add empty result for failed pick
                import numpy as np

                empty_array = np.array([], dtype=locs.dtype).view(np.recarray)
                results.append((original_idx, empty_array))

        return results

    except Exception as e:
        # Return empty results for entire chunk on error
        import numpy as np

        empty_results = []
        for original_idx in chunk_indices:
            empty_array = np.array([], dtype=locs.dtype).view(np.recarray)
            empty_results.append((original_idx, empty_array))
        return empty_results


def _parallel_picked_locs_circle(
    locs, width, height, picks, pick_size, add_group=True, callback=None
):
    """
    Parallelised version of circle picking for picked_locs.

    Uses ThreadPoolExecutor with shared memory for efficient parallel processing.
    Much faster than ProcessPoolExecutor due to reduced serialization overhead.

    Args:
        locs: Localisation data
        width: Image width
        height: Image height
        picks: List of circle picks in format [(x, y), ...]
        pick_size: Radius of the pick circles
        add_group: Whether to add group IDs to localisations
        callback: Progress callback function

    Returns:
        List of localisation arrays, one per pick
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import multiprocessing as mp
    import math

    if len(picks) == 0:
        return []

    # Only use parallel processing if we have enough picks to make it worthwhile
    if len(picks) < 8:
        return _serial_picked_locs_circle(
            locs, width, height, picks, pick_size, add_group, callback
        )

    try:
        # Calculate optimal thread count - use more threads since they're lightweight
        n_threads = min(mp.cpu_count() * 2, len(picks), 16)  # Up to 16 threads

        # Set up progress tracking
        progress_bar_context = None
        progress = None
        total_completed = 0

        if callback == "console":
            import ProgressUtils

            progress_bar_context = ProgressUtils.analysis_progress_bar(
                total=len(picks), desc="Picking locs (parallel threads)"
            )
            progress = progress_bar_context.__enter__()

        try:
            # Process individual picks with ThreadPool (no chunking needed)
            with ThreadPoolExecutor(max_workers=n_threads) as executor:
                # Submit individual pick processing jobs
                future_to_index = {}
                for i, pick in enumerate(picks):
                    future = executor.submit(
                        _process_single_circle_pick,
                        locs,
                        width,
                        height,
                        i,
                        pick,
                        pick_size,
                        add_group,
                    )
                    future_to_index[future] = i

                # Collect results as they complete (more efficient than waiting sequentially)
                picked_locs = [None] * len(picks)

                for future in as_completed(future_to_index):
                    try:
                        original_idx, result = future.result()
                        picked_locs[original_idx] = result

                        total_completed += 1

                        # Update progress
                        if callback == "console" and progress:
                            progress.update(1)
                        elif callback is not None:
                            callback(total_completed)

                    except Exception as e:
                        # Handle individual pick failures gracefully
                        original_idx = future_to_index[future]
                        print(
                            f"Warning: Processing circle pick {original_idx} failed with error: {e}"
                        )
                        picked_locs[original_idx] = np.array([], dtype=locs.dtype).view(
                            np.recarray
                        )

                        if callback == "console" and progress:
                            progress.update(1)
                        elif callback is not None:
                            callback(total_completed)

            # Fill any remaining None positions (shouldn't happen, but safety check)
            for i in range(len(picked_locs)):
                if picked_locs[i] is None:
                    picked_locs[i] = np.array([], dtype=locs.dtype).view(np.recarray)

            return picked_locs

        finally:
            # Cleanup progress bar
            if progress_bar_context:
                progress_bar_context.__exit__(None, None, None)

    except Exception as e:
        print(f"Parallel processing failed ({e}), falling back to serial processing")
        # Fall back to serial processing
        return _serial_picked_locs_circle(
            locs, width, height, picks, pick_size, add_group, callback
        )


def _process_single_circle_pick(
    locs, width, height, original_idx, pick, pick_size, add_group
):
    """
    Process a single circle pick (efficient thread-based processing).

    Args:
        locs: Localisation data (shared in memory across threads)
        width: Image width
        height: Image height
        original_idx: Original index of this pick
        pick: Circle pick coordinate (x, y)
        pick_size: Radius of the pick circle
        add_group: Whether to add group IDs

    Returns:
        Tuple of (original_index, filtered_localisations)
    """
    try:
        # Import required modules
        import lib
        import numpy as np

        # Process the single pick
        x, y = pick

        # Use spatial index for efficient filtering
        index_blocks = get_index_blocks(locs, width, height, pick_size)
        block_locs = get_block_locs_at(x, y, index_blocks)
        group_locs = lib.locs_at(x, y, block_locs, pick_size)

        # Add group ID if requested (use original index for group ID)
        if add_group:
            group = original_idx * np.ones(len(group_locs), dtype=np.int32)
            group_locs = lib.append_to_rec(group_locs, group, "group")

        # Sort by frame
        group_locs.sort(kind="mergesort", order="frame")

        return (original_idx, group_locs)

    except Exception as e:
        # Return empty result on error
        return (original_idx, np.array([], dtype=locs.dtype).view(np.recarray))


def _process_circle_pick_chunk(
    locs, width, height, chunk_indices, chunk_picks, pick_size, add_group
):
    """
    DEPRECATED: Process a chunk of circle picks (efficient chunk-based multiprocessing).

    This function is kept for backward compatibility but is no longer used.
    The new ThreadPoolExecutor approach processes individual picks instead of chunks.

    Args:
        locs: Localisation data
        width: Image width
        height: Image height
        chunk_indices: List of original indices for picks in this chunk
        chunk_picks: List of circle picks in format [(x, y), ...]
        pick_size: Radius of the pick circles
        add_group: Whether to add group IDs

    Returns:
        List of (original_index, filtered_localisations) tuples
    """
    try:
        # Import required modules (needed in each worker process)
        import lib
        import numpy as np

        results = []

        # Create index blocks for efficient spatial queries (once per chunk)
        index_blocks = get_index_blocks(locs, width, height, pick_size)

        # Process each pick in the chunk
        for original_idx, pick in zip(chunk_indices, chunk_picks):
            try:
                x, y = pick

                # Get localisations in spatial vicinity
                block_locs = get_block_locs_at(x, y, index_blocks)

                # Apply circular filtering
                group_locs = lib.locs_at(x, y, block_locs, pick_size)

                # Add group ID if requested (use original index for group ID)
                if add_group:
                    group = original_idx * np.ones(len(group_locs), dtype=np.int32)
                    group_locs = lib.append_to_rec(group_locs, group, "group")

                # Sort by frame
                group_locs.sort(kind="mergesort", order="frame")

                results.append((original_idx, group_locs))

            except Exception as e:
                # Add empty result for failed pick
                import numpy as np

                empty_array = np.array([], dtype=locs.dtype).view(np.recarray)
                results.append((original_idx, empty_array))

        return results

    except Exception as e:
        # Return empty results for entire chunk on error
        import numpy as np

        empty_results = []
        for original_idx in chunk_indices:
            empty_array = np.array([], dtype=locs.dtype).view(np.recarray)
            empty_results.append((original_idx, empty_array))
        return empty_results


def _serial_picked_locs_circle(
    locs, width, height, picks, pick_size, add_group=True, callback=None
):
    """
    Serial version of circle picking (fallback for parallel processing).

    Args:
        locs: Localisation data
        width: Image width
        height: Image height
        picks: List of circle picks in format [(x, y), ...]
        pick_size: Radius of pick circles
        add_group: Whether to add group IDs
        callback: Progress callback function

    Returns:
        List of localisation arrays, one per pick
    """
    try:
        import lib
    except ImportError:
        raise ImportError("lib module required for circle picking")

    picked_locs = []

    # Set up progress tracking
    progress_bar_context = None
    progress = None
    if callback == "console":
        import ProgressUtils

        progress_bar_context = ProgressUtils.analysis_progress_bar(
            total=len(picks), desc="Picking locs (serial circles)"
        )
        progress = progress_bar_context.__enter__()

    try:
        # Create index blocks for efficient spatial queries
        index_blocks = get_index_blocks(locs, width, height, pick_size)

        for i, pick in enumerate(picks):
            try:
                x, y = pick

                # Get localisations in spatial vicinity
                block_locs = get_block_locs_at(x, y, index_blocks)

                # Apply circular filtering
                group_locs = lib.locs_at(x, y, block_locs, pick_size)

                if add_group:
                    group = i * np.ones(len(group_locs), dtype=np.int32)
                    group_locs = lib.append_to_rec(group_locs, group, "group")

                group_locs.sort(kind="mergesort", order="frame")
                picked_locs.append(group_locs)

                # Update progress
                if callback == "console" and progress:
                    progress.update(1)
                elif callback is not None:
                    callback(i + 1)

            except Exception as e:
                print(f"Warning: Processing pick {i} failed: {e}")
                picked_locs.append(np.array([], dtype=locs.dtype).view(np.recarray))

                # Still update progress on error
                if callback == "console" and progress:
                    progress.update(1)
                elif callback is not None:
                    callback(i + 1)

    finally:
        # Cleanup progress bar
        if progress_bar_context:
            progress_bar_context.__exit__(None, None, None)

    return picked_locs


def _serial_picked_locs_rectangle(
    locs, width, height, picks, pick_size, add_group=True, callback=None
):
    """
    Serial version of rectangle picking (fallback for parallel processing).

    Args:
        locs: Localisation data
        width: Image width
        height: Image height
        picks: List of rectangle picks
        pick_size: Size of pick region
        add_group: Whether to add group IDs
        callback: Progress callback function

    Returns:
        List of localisation arrays, one per pick
    """
    try:
        import lib
    except ImportError:
        raise ImportError("lib module required for rectangle picking")

    picked_locs = []

    # Set up progress tracking
    progress_bar_context = None
    progress = None
    if callback == "console":
        import ProgressUtils

        progress_bar_context = ProgressUtils.analysis_progress_bar(
            total=len(picks), desc="Picking locs (serial fallback)"
        )
        progress = progress_bar_context.__enter__()

    try:
        for i, pick in enumerate(picks):
            try:
                (xs, ys), (xe, ye) = pick
                X, Y = lib.get_pick_rectangle_corners(xs, ys, xe, ye, pick_size)
                x_min, x_max = min(X), max(X)
                y_min, y_max = min(Y), max(Y)

                # Filter localisations
                group_locs = locs[
                    (locs.xc > x_min)
                    & (locs.xc < x_max)
                    & (locs.yc > y_min)
                    & (locs.yc < y_max)
                ]
                group_locs = lib.locs_in_rectangle(group_locs, X, Y)

                # Add rotated coordinates
                angle = 0.5 * np.pi - np.arctan2((ye - ys), (xe - xs))
                x_shifted = group_locs.xc - xs
                y_shifted = group_locs.yc - ys
                x_pick_rot = x_shifted * np.cos(angle) - y_shifted * np.sin(angle)
                y_pick_rot = x_shifted * np.sin(angle) + y_shifted * np.cos(angle)

                group_locs = lib.append_to_rec(group_locs, x_pick_rot, "x_pick_rot")
                group_locs = lib.append_to_rec(group_locs, y_pick_rot, "y_pick_rot")

                if add_group:
                    group = i * np.ones(len(group_locs), dtype=np.int32)
                    group_locs = lib.append_to_rec(group_locs, group, "group")

                group_locs.sort(kind="mergesort", order="frame")
                picked_locs.append(group_locs)

                # Update progress
                if callback == "console" and progress:
                    progress.update(1)
                elif callback is not None:
                    callback(i + 1)

            except Exception as e:
                print(f"Warning: Processing pick {i} failed: {e}")
                picked_locs.append(np.array([], dtype=locs.dtype).view(np.recarray))

                # Still update progress on error
                if callback == "console" and progress:
                    progress.update(1)
                elif callback is not None:
                    callback(i + 1)

    finally:
        # Cleanup progress bar
        if progress_bar_context:
            progress_bar_context.__exit__(None, None, None)

    return picked_locs


def segment_locs_by_rendered_image(
    locs,
    width,
    height,
    oversampling=8,
    pixel_size_nm=69.0,
    min_area_nm2=100.0,
    min_localisations=100,
    threshold_method="otsu",
    blur_method="smooth",
    callback=None,
    verbose=False,
):
    """
    Memory-efficient aggregate detection using image-based segmentation.

    This function replaces memory-intensive DBSCAN clustering with a more efficient
    approach:
    1. Render localisations to a super-resolved image
    2. Apply thresholding (Otsu or Li) to detect objects
    3. Filter objects by area threshold
    4. Extract localisations within valid regions

    This approach is ~10-100x more memory efficient than DBSCAN for large datasets.

    Parameters
    ----------
    locs : pd.DataFrame or np.recarray
        Localization data with at least 'xc', 'yc' columns
    width : float
        Image width in pixels (camera pixels, not super-res)
    height : float
        Image height in pixels (camera pixels, not super-res)
    oversampling : int, optional
        Super-resolution oversampling factor (default: 8)
    pixel_size_nm : float, optional
        Size of one camera pixel in nanometers (default: 69.0 nm)
        Used to convert areas from pixels to physical units.
    min_area_nm2 : float, optional
        Minimum area in square nanometers for valid aggregates (default: 100.0 nm²)
    min_localisations : int, optional
        Minimum number of localisations per aggregate (default: 100)
    threshold_method : str, optional
        Thresholding method: 'otsu', 'li', or 'percentile' (default: 'otsu')
    blur_method : str, optional
        Blur method for rendering: 'smooth', 'gaussian', or None (default: 'smooth')
    callback : callable or str, optional
        Progress callback. If 'console', uses tqdm progress bar.
    verbose : bool, optional
        If True, display diagnostic plots showing the rendered image,
        binary mask, and labeled regions (default: False)

    Returns
    -------
    aggregate_locs : pd.DataFrame or np.recarray
        Localizations within valid aggregates with added columns:
        - 'aggregate_id': Unique ID for each aggregate
        - 'aggregate_area_nm2': Area of the aggregate in nm²
    per_aggregate_stats : pd.DataFrame or np.recarray
        Per-aggregate statistics with one row per aggregate

    Examples
    --------
    >>> # Basic usage
    >>> agg_locs, stats = segment_locs_by_rendered_image(
    ...     locs, width=2048, height=2048, min_area_nm2=100.0
    ... )
    >>>
    >>> # With custom threshold and progress bar
    >>> agg_locs, stats = segment_locs_by_rendered_image(
    ...     locs, width=2048, height=2048,
    ...     threshold_method='li',
    ...     min_localisations=50,
    ...     min_area_nm2=50.0,
    ...     callback='console'
    ... )
    """
    try:
        # Import required modules
        from skimage import filters, measure
        import pandas as pd
    except ImportError as e:
        raise ImportError(
            f"Required module not found: {e}. "
            "Please install scikit-image: pip install scikit-image"
        )

    # Convert to pandas DataFrame if numpy recarray
    if isinstance(locs, np.recarray):
        locs_df = pd.DataFrame(locs)
    else:
        locs_df = locs.copy()

    # Validate required columns
    if "xc" not in locs_df.columns or "yc" not in locs_df.columns:
        raise ValueError("Localizations must have 'xc' and 'yc' columns")

    # Step 1: Render super-resolved image
    if callback == "console" or callback is not None:
        print("Step 1/5: Rendering super-resolved image...")

    info = [{
    "Width": width,         # Image width in pixels
    "Height": height,        # Image height in pixels
    "Frames": np.max(locs_df['frame']),      # Total number of frames in the movie
    "Pixelsize": pixel_size_nm,        # pixel size always in nm
    }]

    n_locs, rendered_image = render.render(
        locs=locs_df.to_records(index=False) if isinstance(locs_df, pd.DataFrame) else locs_df,
        oversampling=oversampling,
        info=info,
        blur_method=blur_method,
    )

    # Step 2: Apply thresholding
    if callback == "console" or callback is not None:
        print(f"Step 2/5: Applying {threshold_method} thresholding...")

    if threshold_method == "otsu":
        threshold = filters.threshold_otsu(rendered_image)
    elif threshold_method == "li":
        threshold = filters.threshold_li(rendered_image)
    elif threshold_method == "percentile":
        threshold = np.percentile(rendered_image[rendered_image > 0], 95)
    else:
        raise ValueError(
            f"Unknown threshold_method: {threshold_method}. "
            "Choose from 'otsu', 'li', or 'percentile'"
        )

    binary_image = rendered_image > threshold

    # Step 3: Label connected components
    if callback == "console" or callback is not None:
        print("Step 3/5: Detecting connected regions...")

    label_image = measure.label(binary_image)
    regions = measure.regionprops(label_image)

    # Step 4: Filter by area and count localisations
    if callback == "console" or callback is not None:
        print("Step 4/5: Filtering aggregates by size and localisation count...")

    valid_regions = []
    # Calculate super-resolved pixel size
    # If oversampling=8 and camera pixel=69nm, then super-res pixel = 69/8 = 8.625nm
    superres_pixel_size_nm = pixel_size_nm / oversampling

    # Convert min_area from nm² to super-resolved pixels²
    min_area_pixels = min_area_nm2 / (superres_pixel_size_nm ** 2)

    for region in regions:
        # Check area threshold
        if region.area >= min_area_pixels:
            # Create mask for this region
            minr, minc, maxr, maxc = region.bbox

            # Convert to camera pixel coordinates
            x_min = minc / oversampling
            x_max = maxc / oversampling
            y_min = minr / oversampling
            y_max = maxr / oversampling

            # Find localisations in bounding box
            in_bbox = (
                (locs_df["xc"] >= x_min)
                & (locs_df["xc"] <= x_max)
                & (locs_df["yc"] >= y_min)
                & (locs_df["yc"] <= y_max)
            )
            bbox_locs = locs_df[in_bbox]

            # Further filter by actual region mask
            x_sr = ((bbox_locs["xc"] - x_min) * oversampling).astype(int)
            y_sr = ((bbox_locs["yc"] - y_min) * oversampling).astype(int)

            # Clamp coordinates to mask bounds
            x_sr = np.clip(x_sr, 0, maxc - minc - 1)
            y_sr = np.clip(y_sr, 0, maxr - minr - 1)

            # Check which localisations are inside the region
            region_mask = region.image
            in_region = region_mask[y_sr, x_sr]

            n_locs_in_region = np.sum(in_region)

            # Check localisation count threshold
            if n_locs_in_region >= min_localisations:
                valid_regions.append({
                    "region": region,
                    "label": region.label,
                    "area_nm2": region.area * (superres_pixel_size_nm ** 2),
                    "n_localisations": n_locs_in_region,
                })

    # Display diagnostic plots if verbose
    if verbose:
        try:
            import matplotlib.pyplot as plt
            from PlottingBase import AnalysisPlotter

            plotter = AnalysisPlotter()

            # Create a 3-panel figure
            fig, axs = plotter.two_column_plot(
                nrows=3, ncols=1, height_ratios=[1, 1, 1]
            )

            # Panel 1: Rendered super-resolved image
            im1 = plotter.create_image_plot(
                axs[0],
                rendered_image.T,
                cmap="inferno",
                origin="lower",
            )
            plotter.add_colorbar(im1, axs[0], label="Localisations")
            plotter.add_scalebar(axs[0], pixelsize=superres_pixel_size_nm, length_nm=1000, location="lower right")
            axs[0].set_title(f"Step 1: Rendered Image ({len(locs_df)} locs)", fontsize=10)

            # Panel 2: Binary thresholded image
            im2 = plotter.create_image_plot(
                axs[1],
                binary_image.T.astype(float),  # Convert boolean to float for plotting
                cmap="gray",
                vmin=0,
                vmax=1,
                origin="lower",
            )
            plotter.add_scalebar(axs[1], pixelsize=superres_pixel_size_nm, length_nm=1000, location="lower right")
            axs[1].set_title(
                f"Step 2: Binary Mask (threshold={threshold:.2f}, {threshold_method})",
                fontsize=10
            )

            # Panel 3: Labeled regions with valid aggregates highlighted
            # Create a color image showing valid vs invalid regions
            display_image = np.zeros_like(label_image, dtype=float)
            valid_labels = [r["label"] for r in valid_regions]

            for region in regions:
                if region.label in valid_labels:
                    display_image[label_image == region.label] = 2  # Valid aggregates
                else:
                    display_image[label_image == region.label] = 1  # Rejected regions

            im3 = plotter.create_image_plot(
                axs[2],
                display_image.T,
                cmap="Set1",
                vmin=0,
                vmax=3,
                origin="lower",
            )
            plotter.add_colorbar(im3, axs[2], label="Region type (1=rejected, 2=valid)")
            plotter.add_scalebar(axs[2], pixelsize=superres_pixel_size_nm, length_nm=1000, location="lower right")
            axs[2].set_title(
                f"Step 3: Labeled Regions ({len(valid_regions)}/{len(regions)} valid)",
                fontsize=10
            )

            plotter.save_or_show(fig, show=True)

        except Exception as e:
            print(f"Warning: Could not display verbose plots: {e}")
            import traceback
            traceback.print_exc()

    if len(valid_regions) == 0:
        # Provide diagnostic information
        if len(regions) > 0:
            region_areas_nm2 = [r.area * (superres_pixel_size_nm ** 2) for r in regions]
            print(
                f"Warning: No valid aggregates found (tried {len(regions)} regions).\n"
                f"  Threshold value: {threshold:.4f} ({threshold_method} method)\n"
                f"  Min area threshold: {min_area_nm2:.1f} nm² ({min_area_pixels:.1f} pixels)\n"
                f"  Min localisations: {min_localisations}\n"
                f"  Region areas found: min={min(region_areas_nm2):.1f} nm², "
                f"max={max(region_areas_nm2):.1f} nm², mean={np.mean(region_areas_nm2):.1f} nm²\n"
                f"Suggestions:\n"
                f"  - Try threshold_method='li' or 'percentile' (current: '{threshold_method}')\n"
                f"  - Try reducing min_area_nm2 (current: {min_area_nm2:.1f} nm²)\n"
                f"  - Try reducing min_localisations (current: {min_localisations})\n"
                f"  - Use verbose=True to visualize the thresholding"
            )
        else:
            print(
                f"Warning: No regions detected after thresholding.\n"
                f"  Threshold value: {threshold:.4f} ({threshold_method} method)\n"
                f"  Image stats: min={rendered_image.min():.4f}, max={rendered_image.max():.4f}, "
                f"mean={rendered_image.mean():.4f}\n"
                f"Suggestions:\n"
                f"  - Try threshold_method='li' or 'percentile' (current: '{threshold_method}')\n"
                f"  - Use verbose=True to visualize the image and threshold"
            )
        # Return empty results
        empty_locs = locs_df.iloc[:0].copy()
        empty_locs["aggregate_id"] = pd.Series(dtype=int)
        empty_locs["aggregate_area_nm2"] = pd.Series(dtype=float)
        empty_stats = pd.DataFrame(columns=["aggregate_id", "area_nm2", "n_localisations"])
        return empty_locs, empty_stats

    if callback == "console" or callback is not None:
        print(
            f"Found {len(valid_regions)} valid aggregates "
            f"(from {len(regions)} total regions)"
        )

    # Step 5: Extract localisations and compute statistics
    if callback == "console" or callback is not None:
        print("Step 5/5: Extracting localisations and computing statistics...")

    aggregate_locs_list = []
    per_aggregate_stats_list = []

    # Set up progress bar for aggregates
    progress_bar_context = None
    progress = None
    if callback == "console":
        progress_bar_context = ProgressUtils.analysis_progress_bar(
            total=len(valid_regions), desc="Processing aggregates"
        )
        progress = progress_bar_context.__enter__()

    try:
        for i, region_info in enumerate(valid_regions):
            region = region_info["region"]
            minr, minc, maxr, maxc = region.bbox

            # Convert to camera pixel coordinates
            x_min = minc / oversampling
            x_max = maxc / oversampling
            y_min = minr / oversampling
            y_max = maxr / oversampling

            # Find localisations in bounding box
            in_bbox = (
                (locs_df["xc"] >= x_min)
                & (locs_df["xc"] <= x_max)
                & (locs_df["yc"] >= y_min)
                & (locs_df["yc"] <= y_max)
            )
            bbox_locs = locs_df[in_bbox].copy()

            # Further filter by actual region mask
            x_sr = ((bbox_locs["xc"] - x_min) * oversampling).astype(int)
            y_sr = ((bbox_locs["yc"] - y_min) * oversampling).astype(int)

            # Clamp coordinates to mask bounds
            x_sr = np.clip(x_sr, 0, maxc - minc - 1)
            y_sr = np.clip(y_sr, 0, maxr - minr - 1)

            # Check which localisations are inside the region
            region_mask = region.image
            in_region = region_mask[y_sr, x_sr]

            # Get localisations in this aggregate
            aggregate_locs = bbox_locs[in_region].copy()

            # Add aggregate ID and area
            aggregate_locs["aggregate_id"] = i
            aggregate_locs["aggregate_area_nm2"] = region_info["area_nm2"]

            aggregate_locs_list.append(aggregate_locs)

            # Compute per-aggregate statistics
            stats = {
                "aggregate_id": i,
                "area_nm2": region_info["area_nm2"],
                "n_localisations": len(aggregate_locs),
            }

            # Add mean values for numerical columns
            # Use the original column names (without _mean suffix) for compatibility
            # with downstream analysis functions (e.g., Nile Red functions)
            for col in aggregate_locs.columns:
                # Skip columns we don't want to average
                if col in ["aggregate_id", "aggregate_area_nm2", "frame"]:
                    continue

                # Skip error columns (they'll be handled separately)
                if col.endswith("_err"):
                    continue

                if np.issubdtype(aggregate_locs[col].dtype, np.number):
                    # Sum photons instead of averaging
                    if col == "photons":
                        stats[col] = aggregate_locs[col].sum()
                    # Use weighted mean if error column exists
                    elif f"{col}_err" in aggregate_locs.columns:
                        # Weight by 1/error (not 1/error²) as per NileRedFunctions convention
                        errors = aggregate_locs[f"{col}_err"]
                        weights = 1.0 / errors
                        stats[col] = np.average(aggregate_locs[col], weights=weights)

                        # Propagate error: σ_mean = 1 / sqrt(Σ(1/σ_i²))
                        stats[f"{col}_err"] = 1.0 / np.sqrt(np.sum(1.0 / errors**2))
                    else:
                        # Simple mean for columns without errors
                        stats[col] = aggregate_locs[col].mean()

            per_aggregate_stats_list.append(stats)

            # Update progress
            if callback == "console" and progress:
                progress.update(1)
            elif callable(callback):
                callback(i + 1)

    finally:
        if progress_bar_context:
            progress_bar_context.__exit__(None, None, None)

    # Combine results
    aggregate_locs_combined = pd.concat(aggregate_locs_list, ignore_index=True)
    per_aggregate_stats = pd.DataFrame(per_aggregate_stats_list)

    # Add frame column to per_aggregate_stats for compatibility with IO functions
    # Since this is aggregate-level data, we use frame=0 as a placeholder
    if "frame" not in per_aggregate_stats.columns:
        per_aggregate_stats["frame"] = 0

    if callback == "console" or callback is not None:
        print(
            f"✓ Complete! Extracted {len(aggregate_locs_combined)} localisations "
            f"in {len(valid_regions)} aggregates"
        )

    return aggregate_locs_combined, per_aggregate_stats
