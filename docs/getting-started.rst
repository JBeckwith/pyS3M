Introduction
============

``pyS3M`` is a Python package written in support of spatial-spectral single-molecule
spectroscopy analysis by Joseph S. Beckwith.  The preprint describing the method can
be found on `bioRxiv <https://www.biorxiv.org/content/10.64898/2026.04.08.715690v1>`_.

The package provides a set of Python classes that can be used from scripts,
interactive Jupyter notebooks, or a bundled PyQt6 GUI to:

- Fit Bayer-mosaic SMLM data across multiple colour channels simultaneously
- Correct lateral drift using fiducial beads or image cross-correlation
- Cluster and co-localise multi-channel localisations
- Simulate multicolour SMLM experiments for method validation
- Detect and characterise single-molecule steps and FRET transitions

Requires Python >=3.11, <3.13 (tested on 3.12.3) — the ceiling comes from a real
dependency constraint (``colour-demosaicing`` caps at <3.13), not an arbitrary choice.

Installation
============

Clone the repository, then from its root:

.. code-block:: bash

   pip install .

This installs ``pyS3M`` as a real package (``import pyS3M.SR_Functions``, etc. works
from anywhere — no ``sys.path`` hacks needed) along with its core analysis
dependencies. Optional extras layer on top as needed:

.. code-block:: bash

   pip install .[notebooks]  # jupyterlab, napari, seaborn, xarray, plotly, ...
   pip install .[docs]       # Sphinx + the Read the Docs theme, for building docs locally
   pip install .[dev]        # pytest, coverage, black, build

Extras can be combined, e.g. ``pip install .[notebooks,dev]``. For an editable install
while developing ``pyS3M`` itself, add ``-e``: ``pip install -e .[dev]``.

Running the GUI
================

.. code-block:: bash

   pys3m-gui

(installed as a console script by ``pip install .``), or equivalently
``python run_gui.py`` from the repository root without installing.

Getting Started
===============

Example notebooks covering the main workflows are provided in ``notebooks/``.
A minimal analysis looks like:

.. code-block:: python

   from pathlib import Path
   from pyS3M.AnalysisPipeline import AnalysisPipeline, FittingConfig
   from pyS3M.Constants import AnalysisConfig, FilteringCriteria
   from pyS3M.clustering import ClusteringConfig

   cfg = AnalysisConfig(display=False, save_figures=True,
                        output_dir=Path("results/"), dpi=150)
   pipe = AnalysisPipeline(camera="ximea", config=cfg)
   pipe.load_calibration(Path("Camera_Calibrations/Ximea_Camera"))

   fc = FittingConfig(peak_wavelength=0.638, pfa=1e-3)
   pipe.fit(data_dir, mode="smlm", fitting_config=fc)

   locs = pipe.load_localisations(data_dir)
   sm_db, sf_db = pipe.filter_and_cluster(locs)

See ``notebooks/`` for fuller worked examples (per-camera calibration, drift
correction, FRC, channel unmixing, simulation).

Contributing
============

Patches and contributions are welcome.  Please see ``CONTRIBUTING.md`` and
``CODE_OF_CONDUCT.md`` in the repository root for details.
