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

The code has been tested on Python 3.10 and 3.12.

Installation
============

Clone the repository and install the requirements into a virtual environment:

.. code-block:: bash

   git clone <repository-url>
   cd pyS3M
   pip install -r requirements.txt

All source modules live in ``src/``.  Add this directory to your Python path
before importing:

.. code-block:: python

   import sys
   sys.path.insert(0, 'src')

Getting Started
===============

Example notebooks covering the main workflows are provided in ``notebooks/``.
A minimal analysis looks like:

.. code-block:: python

   import sys
   sys.path.insert(0, 'src')

   from SR_Functions import SuperRes_Functions

   # Initialise with your camera model
   srf = SuperRes_Functions(camera='ximea')

   # Fit a single TIFF stack
   results = srf.fit_SM_data(
       directory='path/to/data/',
       filenames=['my_movie.tif'],
   )

Contributing
============

Patches and contributions are welcome.  Please see ``CONTRIBUTING.md`` and
``CODE_OF_CONDUCT.md`` in the repository root for details.
