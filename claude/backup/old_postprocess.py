"""
Backup of dead code removed from src/postprocess.py on 2026-04-10.

These functions were confirmed unused across all src/ files and notebooks.
Reasons for removal:
  - Block-index / pick_similar cluster: only used by pick_similar, which has no callers
  - distance_histogram / pair_correlation / local_density: superseded by modern pipeline
  - compute_dark_times / dark_times: no callers found
  - weighted_variance: no callers found
  - cluster_combine / cluster_combine_dist: superseded by SM_extractionfunctions
  - localisation_precision: Mortensen formula, no callers (use FRCFunctions / a utility if needed)
  - frc_resolution: superseded by FRCFunctions.py
  - align: no callers found
  - groupprops: no callers found
  - nn_analysis: no callers found

The _link_group_count/sum/mean/weighted_mean/min_max JIT functions were separately
moved to LinkingFunctions.py (canonical home) and imported from there.
"""

import numpy as np
import numba
import multiprocessing as mp
import itertools
from concurrent.futures import ThreadPoolExecutor


# ---------------------------------------------------------------------------
# Block-index infrastructure (only used by pick_similar, which is unused)
# ---------------------------------------------------------------------------

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
                    x_range, y_range, locs_xy, block_starts, block_ends, K, L,
                )
                picked_locs_xy = _locs_at(x_grid, y_grid, block_locs_xy, r)
                if picked_locs_xy.shape[1] > 1:
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
def _get_block_locs_at(x_range, y_range, locs_xy, block_starts, block_ends, K, L):
    step = 0
    for k in range(y_range - 1, y_range + 2):
        if 0 < k < K:
            for l in range(x_range - 1, x_range + 2):
                if 0 < l < L:
                    if block_ends[k, l] - block_starts[k, l] > 0:
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


# ---------------------------------------------------------------------------
# Distance / pair-correlation statistics (no callers)
# ---------------------------------------------------------------------------

@numba.jit(nopython=True, nogil=True)
def _distance_histogram(
    locs, bin_size, r_max, x_index, y_index, block_starts, block_ends, start, chunk,
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
    n_threads = min(60, max(1, int(0.75 * mp.cpu_count())))
    chunk = int(N / n_threads)
    starts = range(0, N, chunk)
    args = [
        (locs, bin_size, r_max, x_index, y_index, b_starts, b_ends, start, chunk)
        for start in starts
    ]
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(_distance_histogram, *_) for _ in args]
    results = [future.result() for future in futures]
    return np.sum(results, axis=0)


def pair_correlation(locs, info, bin_size, r_max):
    dh = distance_histogram(locs, info, bin_size, r_max)
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
    n_threads = min(60, max(1, int(0.75 * mp.cpu_count())))
    chunk = int(N / n_threads)
    starts = range(0, N, chunk)
    args = [
        (locs, radius, x_index, y_index, block_starts, block_ends, start, chunk)
        for start in starts
    ]
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(_local_density, *_) for _ in args]
    density = np.sum([future.result() for future in futures], axis=0)
    import lib
    locs = lib.remove_from_rec(locs, "density")
    return lib.append_to_rec(locs, density, "density")


# ---------------------------------------------------------------------------
# Dark-time analysis (no callers)
# ---------------------------------------------------------------------------

def compute_dark_times(locs, group=None):
    if "len" not in locs.dtype.names:
        raise AttributeError("Length not found. Please link localisations first.")
    dark = dark_times(locs, group)
    import lib
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


# ---------------------------------------------------------------------------
# Weighted variance (no callers)
# ---------------------------------------------------------------------------

def weighted_variance(locs):
    n = len(locs)
    w = locs.photons
    x = locs.xc
    y = locs.yc
    xWbarx = np.average(locs.xc, weights=w)
    xWbary = np.average(locs.yc, weights=w)
    wbarx = np.mean(locs.xc_err)
    wbary = np.mean(locs.yc_err)
    variance_x = (
        n / ((n - 1) * sum(w) ** 2)
        * (
            sum((w * x - wbarx * xWbarx) ** 2)
            - 2 * xWbarx * sum((w - wbarx) * (w * x - wbarx * xWbarx))
            + xWbarx**2 * sum((w - wbarx) ** 2)
        )
    )
    variance_y = (
        n / ((n - 1) * sum(w) ** 2)
        * (
            sum((w * y - wbary * xWbary) ** 2)
            - 2 * xWbary * sum((w - wbary) * (w * y - wbary * xWbary))
            + xWbary**2 * sum((w - wbary) ** 2)
        )
    )
    return variance_x, variance_y


# ---------------------------------------------------------------------------
# Cluster combine (superseded by SM_extractionfunctions; no callers)
# ---------------------------------------------------------------------------

def cluster_combine(locs):
    print("Combining localisations...")
    combined_locs = []
    unique_groups = np.unique(locs["group"])
    import ProgressUtils
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
            (group_id, cluster, mean_frame, com_x, com_y, std_frame, std_x, std_y, n),
            dtype=[
                ("group", group.dtype), ("cluster", cluster.dtype),
                ("mean_frame", "f4"), ("xc", "f4"), ("yc", "f4"),
                ("std_frame", "f4"), ("xc_err", "f4"), ("yc_err", "f4"), ("n", "i4"),
            ],
        )
        combined_locs.append(clusters)
    from numpy.lib.recfunctions import stack_arrays
    combined_locs = stack_arrays(combined_locs, asrecarray=True, usemask=False)
    return combined_locs


def cluster_combine_dist(locs):
    print("Calculating distances...")
    combined_locs = []
    from scipy.spatial import distance
    import ProgressUtils
    with ProgressUtils.analysis_progress_bar(
        total=len(np.unique(locs["group"])), desc="Calculating distances"
    ) as pbar:
        for group in np.unique(locs["group"]):
            temp = locs[locs["group"] == group]
            cluster = np.unique(temp["cluster"])
            n_cluster = len(cluster)
            mean_frame = temp["mean_frame"]
            std_frame = temp["std_frame"]
            com_x = temp["xc"]
            com_y = temp["yc"]
            std_x = temp["xc_err"]
            std_y = temp["yc_err"]
            group_id = temp["group"]
            n = temp["n"]
            min_dist = np.zeros(n_cluster)
            for i, clusterval in enumerate(cluster):
                group_locs = temp[temp["cluster"] != clusterval]
                cluster_locs = temp[temp["cluster"] == clusterval]
                ref_point_xy = np.array([cluster_locs.xc, cluster_locs.yc])
                all_points_xy = np.array([group_locs.xc, group_locs.yc])
                distances_xy = distance.cdist(
                    ref_point_xy.transpose(), all_points_xy.transpose()
                )
                min_dist[i] = np.amin(distances_xy)
            clusters = np.rec.array(
                (group_id, cluster, mean_frame, com_x, com_y,
                 std_frame, std_x, std_y, n, min_dist),
                dtype=[
                    ("group", group.dtype), ("cluster", cluster.dtype),
                    ("mean_frame", "f4"), ("xc", "f4"), ("yc", "f4"),
                    ("std_frame", "f4"), ("xc_err", "f4"), ("yc_err", "f4"),
                    ("n", "i4"), ("min_dist", "f4"),
                ],
            )
            combined_locs.append(clusters)
            pbar.update(1)
    from numpy.lib.recfunctions import stack_arrays
    combined_locs = stack_arrays(combined_locs, asrecarray=True, usemask=False)
    return combined_locs


# ---------------------------------------------------------------------------
# Localisation precision (Mortensen 2010) — no callers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# frc_resolution — superseded by FRCFunctions.py; no callers
# ---------------------------------------------------------------------------

def frc_resolution(
    xc,
    yc,
    pixel_size: float,
    oversampling: int = 10,
    threshold: str = "1/7",
):
    """Estimate image resolution via Fourier Ring Correlation (1FRC).

    Use FRCFunctions.py instead.
    """
    try:
        import frc as frc_pkg
    except ImportError:
        raise ImportError(
            "frc package is required for FRC analysis. Install with: pip install frc"
        )
    xc = np.asarray(xc, dtype=float)
    yc = np.asarray(yc, dtype=float)
    rendered_pixel_nm = pixel_size / oversampling
    xc_nm = xc * pixel_size
    yc_nm = yc * pixel_size
    x_min, x_max = xc_nm.min(), xc_nm.max()
    y_min, y_max = yc_nm.min(), yc_nm.max()
    n_px_x = max(int(np.ceil((x_max - x_min) / rendered_pixel_nm)), 2)
    n_px_y = max(int(np.ceil((y_max - y_min) / rendered_pixel_nm)), 2)
    img, _, _ = np.histogram2d(
        xc_nm, yc_nm,
        bins=[n_px_x, n_px_y],
        range=[[x_min, x_max], [y_min, y_max]],
    )
    img = frc_pkg.util.square_image(img, add_padding=True)
    img_size = img.shape[0]
    img = frc_pkg.util.apply_tukey(img)
    frc_curve = frc_pkg.one_frc(img)
    n_rings = len(frc_curve)
    xs_nm = np.arange(n_rings) / (img_size * rendered_pixel_nm)
    resolution_nm, _, threshold_fn = frc_pkg.frc_res(
        xs_nm, frc_curve, img_size, threshold=threshold
    )
    return resolution_nm, frc_curve, xs_nm, threshold_fn


# ---------------------------------------------------------------------------
# align, groupprops, nn_analysis — no callers
# ---------------------------------------------------------------------------

def align(locs, infos, display=False):
    import render
    import imageprocess
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
    import ProgressUtils
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


def nn_analysis(x1, x2, y1, y2, z1, z2, nn_count, same_channel):
    from sklearn.neighbors import NearestNeighbors as NN
    if z1 is not None:
        input1 = np.stack((x1, y1, z1)).T
        input2 = np.stack((x2, y2, z2)).T
    else:
        input1 = np.stack((x1, y1)).T
        input2 = np.stack((x2, y2)).T
    if same_channel:
        model = NN(n_neighbors=nn_count + 1)
    else:
        model = NN(n_neighbors=nn_count)
    model.fit(input1)
    nn, _ = model.kneighbors(input2)
    if same_channel:
        nn = nn[:, 1:]
    return nn
