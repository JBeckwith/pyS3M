## pyS3M

`pyS3M` (written in support of https://www.biorxiv.org/content/10.64898/2026.04.08.715690v1)
is a Python package of classes for analysing spatial-spectral single-molecule localisation
microscopy data — fitting, quality filtering, clustering, drift correction, FRC, and
simulation — usable from scripts, notebooks, or its desktop GUI. Example notebooks are
provided under `notebooks/analyses/` (fitting through resolution estimation) and
`notebooks/simulations/` (generating your own synthetic acquisitions), each running
end-to-end against data already bundled with the repo.

Documentation: https://pys3m.readthedocs.io/en/latest/index.html

## Installation

Requires Python >=3.11, <3.13 (tested on 3.12.3) — the ceiling comes from a real
dependency constraint (`colour-demosaicing` caps at <3.13).

Clone the repository, then from its root:

```bash
pip install .
```

This installs `pyS3M` as a real package (`import pyS3M.SR_Functions`, etc. works from
anywhere — no `sys.path` hacks needed) along with its core analysis dependencies. Optional
extras layer on top as needed:

```bash
pip install .[notebooks]  # jupyterlab, seaborn, xarray, plotly, ...
pip install .[docs]       # Sphinx + the Read the Docs theme, for building docs locally
pip install .[dev]        # pytest, coverage, black, build
```

Extras can be combined, e.g. `pip install .[notebooks,dev]`. For an editable install while
developing `pyS3M` itself, add `-e`: `pip install -e .[dev]`.

## Running the GUI

```bash
pys3m-gui
```

(installed as a console script by `pip install .`), or equivalently `python run_gui.py`
from the repository root without installing.

## Quickstart

See the Getting Started guide for a minimal worked example and installation/GUI details:
https://pys3m.readthedocs.io/en/latest/getting-started.html

See `notebooks/analyses/` for fuller worked examples (multi-FOV fitting, drift correction,
clustering, channel unmixing, FRC, Nile Red) and `notebooks/simulations/` for how to
generate your own synthetic acquisitions.

## License

Copyright © 2026, Cambridge Enterprise Limited, all rights reserved. This software is
provided **for academic use only** — see `LICENSE` for the full text. For commercial use,
contact `ls.ipportfolio@enterprise.cam.ac.uk` quoting LEE-11475-25.

## Contributing

Patches and contributions are very welcome! Please see `CONTRIBUTING.md` and
`CODE_OF_CONDUCT.md` for more details.
