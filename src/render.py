"""
    jsb92 updated render functions
    
    original code
    picasso.render
    ~~~~~~~~~~~~~~~~~~~~

    Render single molecule localizations to a super-resolution image

    :original author: Joerg Schnitzbauer, 2015
    :copyright: Copyright (c) MIT License
"""

import time
import os
import sys

import numpy as np
import numba
import scipy.signal as signal
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv

_DRAW_MAX_SIGMA = 3


def render(
    locs,
    info=None,
    oversampling=1,
    viewport=None,
    blur_method=None,
    min_blur_width=0,
    cparam="A_R",
    c_min=0.3,
    c_max=0.75,
    mindensperc=1,
    maxdensperc=99.9,
    densitymin=0.1,
    cmap_string="jet",
):
    """
    Renders locs.

    Parameters
    ----------
    locs : np.recarray
        Localizations to be rendered
    info : dict (default=None)
        Contains metadata for locs. Needed only if no viewport
        specified
    oversampling : float (default=1)
        Number of super-resolution pixels per camera pixel
    viewport : list or tuple (default=None)
        Field of view to be rendered. If None, all locs are rendered
    blur_method : str (default=None)
        Defines localizations' blur. The string has to be one of
        'gaussian', 'gaussian_iso', 'gaussian_colour', 'smooth', 'convolve'. If None,
        no blurring is applied.
    min_blur_width : float (default=0)
        Minimum size of blur (pixels)

    Raises
    ------
    Exception
        If blur_method not one of 'gaussian', 'gaussian_iso', 'gaussian_colour', 'smooth',
        'convolve' or None

    Returns
    -------
    int
        Number of localizations rendered
    np.array
        Rendered image
    """

    if viewport is None:
        try:
            # all locs
            viewport = [(0, 0), (info[0]["Height"], info[0]["Width"])]
        except TypeError:
            raise ValueError("Need info if no viewport is provided.")
    (y_min, x_min), (y_max, x_max) = viewport
    if blur_method is None:
        # no blur
        return render_hist(
            locs,
            oversampling,
            y_min,
            x_min,
            y_max,
            x_max,
        )
    elif blur_method == "gaussian":
        # individual localization precision
        return render_gaussian(
            locs,
            oversampling,
            y_min,
            x_min,
            y_max,
            x_max,
            min_blur_width,
        )
    elif blur_method == "gaussian_colour":
        return render_gaussian_colour(
            locs,
            oversampling,
            y_min,
            x_min,
            y_max,
            x_max,
            min_blur_width,
            cparam,
            c_min,
            c_max,
            mindensperc,
            maxdensperc,
            densitymin,
            cmap_string,
        )
    elif blur_method == "gaussian_iso":
        # individual localization precision (same for x and y)
        return render_gaussian_iso(
            locs,
            oversampling,
            y_min,
            x_min,
            y_max,
            x_max,
            min_blur_width,
        )
    elif blur_method == "smooth":
        # one pixel blur
        return render_smooth(
            locs,
            oversampling,
            y_min,
            x_min,
            y_max,
            x_max,
        )
    elif blur_method == "convolve":
        # global localization precision
        return render_convolve(
            locs,
            oversampling,
            y_min,
            x_min,
            y_max,
            x_max,
            min_blur_width,
        )
    else:
        raise Exception("blur_method not understood.")


@numba.njit
def _render_colour_setup(
    locs,
    oversampling,
    y_min,
    x_min,
    y_max,
    x_max,
):
    """
    Finds coordinates to be rendered and sets up an empty image array.

    Parameters
    ----------
    locs : np.recarray
        Localizations
    oversampling : float
        Number of super-resolution pixels per camera pixel
    y_min : float
        Minimum y coordinate to be rendered (pixels)
    x_min : float
        Minimum x coordinate to be rendered (pixels)
    y_max : float
        Maximum y coordinate to be rendered (pixels)
    x_max : float
        Maximum x coordinate to be rendered (pixels)

    Returns
    -------
    np.array
        Empty image array
    int
        Number of pixels in y
    int
        Number of pixels in x
    np.array
        x coordinates to be rendered
    np.array
        y coordinates to be rendered
    np.array
        Indeces of locs to be rendered
    """

    n_pixel_y = int(np.ceil(oversampling * (y_max - y_min)))
    n_pixel_x = int(np.ceil(oversampling * (x_max - x_min)))
    x = locs.xc
    y = locs.yc
    in_view = (x > x_min) & (y > y_min) & (x < x_max) & (y < y_max)
    x = x[in_view]
    y = y[in_view]
    x = oversampling * (x - x_min)
    y = oversampling * (y - y_min)
    image_total = np.zeros((n_pixel_y, n_pixel_x), dtype=np.float32)
    image_spectral = np.zeros((n_pixel_y, n_pixel_x), dtype=np.float32)
    return image_total, image_spectral, n_pixel_y, n_pixel_x, x, y, in_view


@numba.njit
def _render_setup(
    locs,
    oversampling,
    y_min,
    x_min,
    y_max,
    x_max,
):
    """
    Finds coordinates to be rendered and sets up an empty image array.

    Parameters
    ----------
    locs : np.recarray
        Localizations
    oversampling : float
        Number of super-resolution pixels per camera pixel
    y_min : float
        Minimum y coordinate to be rendered (pixels)
    x_min : float
        Minimum x coordinate to be rendered (pixels)
    y_max : float
        Maximum y coordinate to be rendered (pixels)
    x_max : float
        Maximum x coordinate to be rendered (pixels)

    Returns
    -------
    np.array
        Empty image array
    int
        Number of pixels in y
    int
        Number of pixels in x
    np.array
        x coordinates to be rendered
    np.array
        y coordinates to be rendered
    np.array
        Indeces of locs to be rendered
    """

    n_pixel_y = int(np.ceil(oversampling * (y_max - y_min)))
    n_pixel_x = int(np.ceil(oversampling * (x_max - x_min)))
    x = locs.xc
    y = locs.yc
    in_view = (x > x_min) & (y > y_min) & (x < x_max) & (y < y_max)
    x = x[in_view]
    y = y[in_view]
    x = oversampling * (x - x_min)
    y = oversampling * (y - y_min)
    image = np.zeros((n_pixel_y, n_pixel_x), dtype=np.float32)
    return image, n_pixel_y, n_pixel_x, x, y, in_view


@numba.njit
def _fill(image, x, y):
    """
    Fills image with x and y coordinates.
    Image is not blurred.

    Parameters
    ----------
    image : np.array
        Empty image array
    x : np.array
        x coordinates to be rendered
    y : np.array
        y coordinates to be rendered
    """

    x = x.astype(np.int32)
    y = y.astype(np.int32)
    for i, j in zip(x, y):
        image[j, i] += 1


@numba.njit
def _fill_colour_gaussian(
    image_total, image_spectral, x, y, sx, sy, colour, n_pixel_x, n_pixel_y
):
    """
    Fills image with blurred x and y coordinates.
    Localization precisions (sx and sy) are treated as standard
    deviations of the guassians to be rendered.

    Parameters
    ----------
    image : np.array
        Empty image array
    x : np.array
        x coordinates to be rendered
    y : np.array
        y coordinates to be rendered
    sx : np.array
        Localization precision in x for each loc
    sy : np.array
        Localization precision in y for each loc
    colour: np.array
        Colour information for each loc
    n_pixel_x : int
        Number of pixels in x
    n_pixel_y : int
        Number of pixels in y
    """

    # render each localization separately
    for x_, y_, sx_, sy_, colour_ in zip(x, y, sx, sy, colour):

        # get min and max indeces to draw the given localization
        max_y = _DRAW_MAX_SIGMA * sy_
        i_min = np.int32(y_ - max_y)
        if i_min < 0:
            i_min = 0
        i_max = np.int32(y_ + max_y + 1)
        if i_max > n_pixel_y:
            i_max = n_pixel_y
        max_x = _DRAW_MAX_SIGMA * sx_
        j_min = np.int32(x_ - max_x)
        if j_min < 0:
            j_min = 0
        j_max = np.int32(x_ + max_x) + 1
        if j_max > n_pixel_x:
            j_max = n_pixel_x

        # draw a localization as a 2D guassian PDF
        for i in range(i_min, i_max):
            for j in range(j_min, j_max):
                val = np.exp(
                    -(
                        (j - x_ + 0.5) ** 2 / (2 * sx_**2)
                        + (i - y_ + 0.5) ** 2 / (2 * sy_**2)
                    )
                ) / (2 * np.pi * sx_ * sy_)
                image_total[i, j] += val
                image_spectral[i, j] += val * colour_


@numba.njit
def _fill_gaussian(image, x, y, sx, sy, n_pixel_x, n_pixel_y):
    """
    Fills image with blurred x and y coordinates.
    Localization precisions (sx and sy) are treated as standard
    deviations of the guassians to be rendered.

    Parameters
    ----------
    image : np.array
        Empty image array
    x : np.array
        x coordinates to be rendered
    y : np.array
        y coordinates to be rendered
    sx : np.array
        Localization precision in x for each loc
    sy : np.array
        Localization precision in y for each loc
    n_pixel_x : int
        Number of pixels in x
    n_pixel_y : int
        Number of pixels in y
    """

    # render each localization separately
    for x_, y_, sx_, sy_ in zip(x, y, sx, sy):

        # get min and max indeces to draw the given localization
        max_y = _DRAW_MAX_SIGMA * sy_
        i_min = np.int32(y_ - max_y)
        if i_min < 0:
            i_min = 0
        i_max = np.int32(y_ + max_y + 1)
        if i_max > n_pixel_y:
            i_max = n_pixel_y
        max_x = _DRAW_MAX_SIGMA * sx_
        j_min = np.int32(x_ - max_x)
        if j_min < 0:
            j_min = 0
        j_max = np.int32(x_ + max_x) + 1
        if j_max > n_pixel_x:
            j_max = n_pixel_x

        # draw a localization as a 2D guassian PDF
        for i in range(i_min, i_max):
            for j in range(j_min, j_max):
                image[i, j] += np.exp(
                    -(
                        (j - x_ + 0.5) ** 2 / (2 * sx_**2)
                        + (i - y_ + 0.5) ** 2 / (2 * sy_**2)
                    )
                ) / (2 * np.pi * sx_ * sy_)


def render_hist(
    locs,
    oversampling,
    y_min,
    x_min,
    y_max,
    x_max,
):
    """
    Renders locs with no blur.

    Parameters
    ----------
    locs : np.recarray
        Localizations to be rendered
    oversampling : float (default=1)
        Number of super-resolution pixels per camera pixel
    y_min : float
        Minimum y coordinate to be rendered (pixels)
    x_min : float
        Minimum x coordinate to be rendered (pixels)
    y_max : float
        Maximum y coordinate to be rendered (pixels)
    x_max : float
        Maximum x coordinate to be rendered (pixels)

    Returns
    -------
    int
        Number of localizations rendered
    np.array
        Rendered image
    """

    image, n_pixel_y, n_pixel_x, x, y, in_view = _render_setup(
        locs,
        oversampling,
        y_min,
        x_min,
        y_max,
        x_max,
    )
    _fill(image, x, y)
    return len(x), image


def render_gaussian_colour(
    locs,
    oversampling,
    y_min,
    x_min,
    y_max,
    x_max,
    min_blur_width,
    cparam,
    c_min,
    c_max,
    mindensperc,
    maxdensperc,
    densitymin,
    cmap_string,
):
    """
    Renders locs with with individual localization precision which
    differs in x and y. Renders a spectral image based on a colour parameter.

    Parameters
    ----------
    locs : np.recarray
        Localizations to be rendered
    oversampling : float (default=1)
        Number of super-resolution pixels per camera pixel
    y_min : float
        Minimum y coordinate to be rendered (pixels)
    x_min : float
        Minimum x coordinate to be rendered (pixels)
    y_max : float
        Maximum y coordinate to be rendered (pixels)
    x_max : float
        Maximum x coordinate to be rendered (pixels)
    min_blur_width : float
        Minimum localization precision (pixels)
    ang : tuple (default=None)
        Rotation angles of locs around x, y and z axes. If None,
        locs are not rotated.

    Returns
    -------
    int
        Number of localizations rendered
    np.array
        Rendered image
    """

    image_total, image_spectral, n_pixel_y, n_pixel_x, x, y, in_view = (
        _render_colour_setup(
            locs,
            oversampling,
            y_min,
            x_min,
            y_max,
            x_max,
        )
    )

    blur_width = oversampling * np.maximum(locs.xc_err, min_blur_width)
    blur_height = oversampling * np.maximum(locs.yc_err, min_blur_width)
    sy = blur_height[in_view]
    sx = blur_width[in_view]
    color = locs[cparam][in_view]

    _fill_colour_gaussian(
        image_total, image_spectral, x, y, sx, sy, color, n_pixel_x, n_pixel_y
    )

    non_zero = image_total > 0
    image_spectral[non_zero] = image_spectral[non_zero] / image_total[non_zero]
    image_spectral[~non_zero] = 0

    min_density = np.percentile(image_total, mindensperc)
    max_density = np.percentile(image_total, maxdensperc)
    cmap = plt.get_cmap(cmap_string)
    normalised_density = np.clip(
        (image_total - min_density) / (max_density - min_density), 0, 1
    )

    normalised_wl = np.clip((image_spectral - c_min) / (c_max - c_min), 0, 1)
    rgb = cmap(normalised_wl)[..., :3]
    rgb[normalised_density < densitymin] = 0
    hsv = rgb_to_hsv(rgb)
    hsv[..., 1] = normalised_density
    image_colour_gaussian = hsv_to_rgb(hsv)

    return len(x), image_total, image_colour_gaussian


def render_gaussian(
    locs,
    oversampling,
    y_min,
    x_min,
    y_max,
    x_max,
    min_blur_width,
):
    """
    Renders locs with with individual localization precision which
    differs in x and y.

    Parameters
    ----------
    locs : np.recarray
        Localizations to be rendered
    oversampling : float (default=1)
        Number of super-resolution pixels per camera pixel
    y_min : float
        Minimum y coordinate to be rendered (pixels)
    x_min : float
        Minimum x coordinate to be rendered (pixels)
    y_max : float
        Maximum y coordinate to be rendered (pixels)
    x_max : float
        Maximum x coordinate to be rendered (pixels)
    min_blur_width : float
        Minimum localization precision (pixels)
    ang : tuple (default=None)
        Rotation angles of locs around x, y and z axes. If None,
        locs are not rotated.

    Returns
    -------
    int
        Number of localizations rendered
    np.array
        Rendered image
    """

    image, n_pixel_y, n_pixel_x, x, y, in_view = _render_setup(
        locs,
        oversampling,
        y_min,
        x_min,
        y_max,
        x_max,
    )

    blur_width = oversampling * np.maximum(locs.xc_err, min_blur_width)
    blur_height = oversampling * np.maximum(locs.yc_err, min_blur_width)
    sy = blur_height[in_view]
    sx = blur_width[in_view]

    _fill_gaussian(image, x, y, sx, sy, n_pixel_x, n_pixel_y)

    return len(x), image


def render_gaussian_iso(
    locs,
    oversampling,
    y_min,
    x_min,
    y_max,
    x_max,
    min_blur_width,
):
    """
    Renders locs with with individual localization precision which
    is the same in x and y.

    Parameters
    ----------
    locs : np.recarray
        Localizations to be rendered
    oversampling : float (default=1)
        Number of super-resolution pixels per camera pixel
    y_min : float
        Minimum y coordinate to be rendered (pixels)
    x_min : float
        Minimum x coordinate to be rendered (pixels)
    y_max : float
        Maximum y coordinate to be rendered (pixels)
    x_max : float
        Maximum x coordinate to be rendered (pixels)
    min_blur_width : float
        Minimum localization precision (pixels)
    ang : tuple (default=None)
        Rotation angles of locs around x, y and z axes. If None,
        locs are not rotated.

    Returns
    -------
    int
        Number of localizations rendered
    np.array
        Rendered image
    """

    image, n_pixel_y, n_pixel_x, x, y, in_view = _render_setup(
        locs,
        oversampling,
        y_min,
        x_min,
        y_max,
        x_max,
    )

    blur_width = oversampling * np.maximum(locs.xc_err, min_blur_width)
    blur_height = oversampling * np.maximum(locs.yc_err, min_blur_width)
    sy = (blur_height[in_view] + blur_width[in_view]) / 2
    sx = sy

    _fill_gaussian(image, x, y, sx, sy, n_pixel_x, n_pixel_y)

    return len(x), image


def render_convolve(
    locs,
    oversampling,
    y_min,
    x_min,
    y_max,
    x_max,
    min_blur_width,
):
    """
    Renders locs with with global localization precision, i.e. each
    localization is blurred by the median localization precision in x
    and y.

    Parameters
    ----------
    locs : np.recarray
        Localizations to be rendered
    oversampling : float (default=1)
        Number of super-resolution pixels per camera pixel
    y_min : float
        Minimum y coordinate to be rendered (pixels)
    x_min : float
        Minimum x coordinate to be rendered (pixels)
    y_max : float
        Maximum y coordinate to be rendered (pixels)
    x_max : float
        Maximum x coordinate to be rendered (pixels)
    min_blur_width : float
        Minimum localization precision (pixels)
    ang : tuple (default=None)
        Rotation angles of locs around x, y and z axes. If None,
        locs are not rotated.

    Returns
    -------
    int
        Number of localizations rendered
    np.array
        Rendered image
    """

    image, n_pixel_y, n_pixel_x, x, y, in_view = _render_setup(
        locs,
        oversampling,
        y_min,
        x_min,
        y_max,
        x_max,
    )
    n = len(x)
    if n == 0:
        return 0, image
    else:
        _fill(image, x, y)
        blur_width = oversampling * max(
            np.median(locs.xc_err[in_view]), min_blur_width
        )
        blur_height = oversampling * max(
            np.median(locs.yc_err[in_view]), min_blur_width
        )
        return n, _fftconvolve(image, blur_width, blur_height)


def render_smooth(
    locs,
    oversampling,
    y_min,
    x_min,
    y_max,
    x_max,
):
    """
    Renders locs with with blur of one display pixel (set by
    oversampling)

    Parameters
    ----------
    locs : np.recarray
        Localizations to be rendered
    oversampling : float (default=1)
        Number of super-resolution pixels per camera pixel
    y_min : float
        Minimum y coordinate to be rendered (pixels)
    x_min : float
        Minimum x coordinate to be rendered (pixels)
    y_max : float
        Maximum y coordinate to be rendered (pixels)
    x_max : float
        Maximum x coordinate to be rendered (pixels)
    ang : tuple (default=None)
        Rotation angles of locs around x, y and z axes. If None,
        locs are not rotated.

    Returns
    -------
    int
        Number of localizations rendered
    np.array
        Rendered image
    """

    image, n_pixel_y, n_pixel_x, x, y, in_view = _render_setup(
        locs,
        oversampling,
        y_min,
        x_min,
        y_max,
        x_max,
    )

    n = len(x)
    if n == 0:
        return 0, image
    else:
        _fill(image, x, y)
        return n, _fftconvolve(image, 1, 1)


def _fftconvolve(image, blur_width, blur_height):
    """
    Blurs (convolves) 2D image using fast fourier transform.

    Parameters
    ----------
    image : np.array
        Image with renderd but not blurred locs
    blur_width : float
        Blur width
    blur_height
        Blur height

    Returns
    -------
    np.array
        Blurred image
    """

    kernel_width = 10 * int(np.round(blur_width)) + 1
    kernel_height = 10 * int(np.round(blur_height)) + 1
    kernel_y = signal.windows.gaussian(kernel_height, blur_height)
    kernel_x = signal.windows.gaussian(kernel_width, blur_width)
    kernel = np.outer(kernel_y, kernel_x)
    kernel /= kernel.sum()
    return signal.fftconvolve(image, kernel, mode="same")
