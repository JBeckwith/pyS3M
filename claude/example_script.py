#!/usr/bin/env python3
"""
Code Ocean example script for pyS3M single-molecule localisation.

Layout assumed on Code Ocean:
  /code/src/                              Python source files
  /code/Camera_Calibrations/Ximea_Camera/ calibration TIFFs
  /data/                                  input TIFF + metadata (read-only)
  /results/                               output directory
"""

import os, sys, shutil, types, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

# ── paths ─────────────────────────────────────────────────────────────────────
src_dir     = '/code/src'
calib_dir   = '/code/Camera_Calibrations/Ximea_Camera'
data_dir    = '/data'
results_dir = '/results'

sys.path.insert(0, src_dir)

from IOFunctions    import IO_Functions
from SR_Functions   import SuperRes_Functions
from sCMOSFunctions import sCMOS_Functions
from PlottingBase   import PublicationPlotter

# ── copy TIFF + metadata to /results so fit_SM_data can write .h5 there ───────
copied_files = []
for pattern in ('*.tif', '*.ome.tif', '*metadata*'):
    for f in glob.glob(os.path.join(data_dir, pattern)):
        dst = os.path.join(results_dir, os.path.basename(f))
        shutil.copy(f, dst)
        copied_files.append(dst)

image_folder = results_dir

# ── calibration maps ──────────────────────────────────────────────────────────
io = IO_Functions()
gain_map   = io.read_tiff(os.path.join(calib_dir, 'gain.tif'))
offset_map = io.read_tiff(os.path.join(calib_dir, 'offset.tif'))
read_noise = io.read_tiff(os.path.join(calib_dir, 'readnoise.tif'))
rqe        = io.read_tiff(os.path.join(calib_dir, 'rqe.tif'))
variance   = io.read_tiff(os.path.join(calib_dir, 'variance.tif'))

# ── smoothing function ────────────────────────────────────────────────────────
sCMOS = sCMOS_Functions()
smoothing_function                    = types.SimpleNamespace()
smoothing_function.smoothing_function = sCMOS.gaussian_filter_stack
smoothing_function.args               = {'sigma': 1.5}
smoothing_function.extent             = 1.5
smoothing_function.data_arg           = 'image'

# ── detect & localise ─────────────────────────────────────────────────────────
SR = SuperRes_Functions(camera='ximea')
SR.fit_SM_data(
    image_folder       = image_folder,
    smoothing_function = smoothing_function,
    gain_map           = gain_map,
    offset_map         = offset_map,
    rqe                = rqe,
    read_noise         = read_noise,
    variance           = variance,
    pfa                = 1e-4,
    peak_wavelength    = 0.65,
)

# ── remove copied TIFF + metadata, keep only the .h5 ─────────────────────────
for f in copied_files:
    os.remove(f)

# ── read results ──────────────────────────────────────────────────────────────
# write_h5_database normalises A_B/A_G/A_R to fractions (sum = 1) by default
# and stores their pre-normalisation sum as 'photons'.
dfs = [io.read_h5_database(f) for f in glob.glob(os.path.join(results_dir, '*.h5'))]
df  = pd.concat(dfs, ignore_index=True)

A_B     = df['A_B'].dropna().values
A_G     = df['A_G'].dropna().values
A_R     = df['A_R'].dropna().values
photons = df['photons'].dropna().values

# ── figure 1: colour fractions (A_B, A_G, A_R) ───────────────────────────────
plotter   = PublicationPlotter()
fig1, ax1 = plotter.two_column_plot(nrows=1, ncols=1)

for data, colour, label in [
    (A_B, '#2166ac', 'A_B'),
    (A_G, '#4dac26', 'A_G'),
    (A_R, '#d01c8b', 'A_R'),
]:
    bins = np.histogram_bin_edges(data, 'fd')
    plotter.histogram_plot(
        ax1, data,
        bins   = bins,
        color  = colour,
        alpha  = 0.5,
        label  = label,
        xlabel = 'Colour fraction',
        ylabel = 'Counts',
        grid   = True,
    )

ax1.legend()
plotter.save_or_show(
    fig1,
    save_path = os.path.join(results_dir, 'colour_fractions.png'),
    show      = False,
)

# ── figure 2: photons per punctum ────────────────────────────────────────────
fig2, ax2 = plotter.two_column_plot(nrows=1, ncols=1)

bins_ph = np.histogram_bin_edges(photons, 'fd')
plotter.histogram_plot(
    ax2, photons,
    bins   = bins_ph,
    color  = '#636363',
    xlabel = 'Photons per punctum',
    ylabel = 'Counts',
    grid   = True,
)

plotter.save_or_show(
    fig2,
    save_path = os.path.join(results_dir, 'photons_per_punctum.png'),
    show      = False,
)
