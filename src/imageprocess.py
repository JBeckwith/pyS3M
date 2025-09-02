"""
    jsb92 updated imageprocess functions
    
    original code
    picasso.imageprocess
    ~~~~~~~~~~~~~~~~~~~~

    Image processing functions

    :author: Joerg Schnitzbauer, 2016
    :copyright: Copyright (c) MIT License
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from numpy import fft
import lmfit

module_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(module_dir)
import ProgressUtils
import lib
import render
import localise  # Changed from localize to localise
import postprocess


def xcorr(imageA, imageB):
    FimageA = fft.fft2(imageA)
    CFimageB = np.conj(fft.fft2(imageB))
    return fft.fftshift(np.real(fft.ifft2((FimageA * CFimageB)))) / np.sqrt(
        imageA.size
    )


def get_image_shift(imageA, imageB, box, roi=None, display=False):
    """Computes the shift from imageA to imageB"""
    if (np.sum(imageA) == 0) or (np.sum(imageB) == 0):
        return 0, 0
    # Compute image correlation
    XCorr = xcorr(imageA, imageB)
    # Cut out center roi
    Y, X = imageA.shape
    if roi is not None:
        Y_ = int((Y - roi) / 2)
        X_ = int((X - roi) / 2)
        if Y_ > 0:
            XCorr = XCorr[Y_:-Y_, :]
        else:
            Y_ = 0
        if X_ > 0:
            XCorr = XCorr[:, X_:-X_]
        else:
            X_ = 0
    else:
        Y_ = X_ = 0
    # A quarter of the fit ROI
    fit_X = int(box / 2)
    # A coordinate grid for the fitting ROI
    y, x = np.mgrid[-fit_X : fit_X + 1, -fit_X : fit_X + 1]
    # Find the brightest pixel and cut out the fit ROI
    y_max_, x_max_ = np.unravel_index(XCorr.argmax(), XCorr.shape)
    FitROI = XCorr[
        y_max_ - fit_X : y_max_ + fit_X + 1,
        x_max_ - fit_X : x_max_ + fit_X + 1,
    ]

    dimensions = FitROI.shape

    if 0 in dimensions or dimensions[0] != dimensions[1]:
        xc, yc = 0, 0
    else:
        # The fit model
        def flat_2d_gaussian(a, xc, yc, s, b):
            A = a * np.exp(-0.5 * ((x - xc) ** 2 + (y - yc) ** 2) / s**2) + b
            return A.flatten()

        gaussian2d = lmfit.Model(
            flat_2d_gaussian, name="2D Gaussian", independent_vars=[]
        )

        # Set up initial parameters and fit
        params = lmfit.Parameters()
        params.add("a", value=FitROI.max(), vary=True, min=0)
        params.add("xc", value=0, vary=True)
        params.add("yc", value=0, vary=True)
        params.add("s", value=1, vary=True, min=0)
        params.add("b", value=FitROI.min(), vary=True, min=0)
        results = gaussian2d.fit(FitROI.flatten(), params)

        # Get maximum coordinates and add offsets
        xc = results.best_values["xc"]
        yc = results.best_values["yc"]
        xc += X_ + x_max_
        yc += Y_ + y_max_

        if display:
            plt.figure(figsize=(17, 10))
            plt.subplot(1, 3, 1)
            plt.imshow(imageA, interpolation="none")
            plt.subplot(1, 3, 2)
            plt.imshow(imageB, interpolation="none")
            plt.subplot(1, 3, 3)
            plt.imshow(XCorr, interpolation="none")
            plt.plot(xc, yc, "x")
            plt.show()
            plt.close()

        xc -= np.floor(X / 2)
        yc -= np.floor(Y / 2)

    return -yc, -xc


def rcc(segments, max_shift=None, callback=None):
    n_segments = len(segments)
    shifts_x = np.zeros((n_segments, n_segments))
    shifts_y = np.zeros((n_segments, n_segments))
    n_pairs = int(n_segments * (n_segments - 1) / 2)
    flag = 0
    if callback is None:
        with ProgressUtils.analysis_progress_bar(
            total=n_pairs, desc="Correlating image pairs"
        ) as progress_bar:
            for i in range(n_segments - 1):
                for j in range(i + 1, n_segments):
                    shifts_y[i, j], shifts_x[i, j] = get_image_shift(
                        segments[i], segments[j], 5, max_shift
                    )
                    progress_bar.update(1)
                    flag += 1
    else:
        callback(0)
        for i in range(n_segments - 1):
            for j in range(i + 1, n_segments):
                shifts_y[i, j], shifts_x[i, j] = get_image_shift(
                    segments[i], segments[j], 5, max_shift
                )
                flag += 1
                callback(flag)

    return lib.minimize_shifts(shifts_x, shifts_y)


def find_fiducials(locs, info):
    """Finds the xy coordinates of regions with high density of
    localizations, likely originating from fiducial markers.

    Uses picasso.localise.identify_in_image with threshold set to 99th
    percentile of the image histogram. The image is rendered using
    one-pixel-blur, see picasso.render.render.


    Parameters
    ----------
    locs : np.recarray
        Localizations.
    info : list of dicts
        Localizations' metadata (from the corresponding .yaml file).

    Returns
    -------
    picks : list of (2,) tuples
        Coordinates of fiducial markers. Each list element corresponds
        to (x, y) coordinates of one fiducial marker.
    box : int
        Size of the box used for the fiducial marker identification.
        Can be set as the pick diameter in pixels for undrifting.
    """

    image = render.render(
        locs=locs,
        info=info,
        oversampling=1,
        viewport=None,
        blur_method="smooth",
    )[1]
    hist = np.histogram(image.flatten(), bins=256)
    threshold = np.percentile(hist[0], 99)
    # box size should be an odd number, corresponding to approximately
    # 900 nm
    pixelsize = 130
    for inf in info:
        if val := inf.get("Pixelsize"):
            pixelsize = val
            break
    box = int(np.round(900 / pixelsize))
    box = box + 1 if box % 2 == 0 else box

    # find the local maxima and translate to pick coordinates
    y, x, _ = localise.identify_in_image(
        image, threshold, box=box
    )  # Changed from _localize to _localise
    picks = [(xi, yi) for xi, yi in zip(x, y)]

    # select the picks with appropriate number of localizations
    n_frames = 0
    for inf in info:
        if val := inf.get("Frames"):
            n_frames = val
            break
    min_n = 0.8 * n_frames
    picked_locs = postprocess.picked_locs(
        locs,
        info,
        picks,
        "Circle",
        pick_size=box / 2,
        add_group=False,
    )
    picks = [pick for i, pick in enumerate(picks) if len(picked_locs[i]) > min_n]
    return picks, box
