## pyS3M

`pyS3M` (written in support of https://www.biorxiv.org/content/10.64898/2026.04.08.715690v1)
is a Python package of classes for analysing spatial-spectral single-molecule localisation
microscopy data — fitting, quality filtering, clustering, drift correction, FRC, and
simulation — usable from scripts, notebooks, or its desktop GUI. Example notebooks are
provided under `notebooks/`, showing worked analyses end-to-end.

Documentation: https://pys3m.readthedocs.io/en/latest/index.html

## Installation

Requires Python >=3.11, <3.13 (tested on 3.12.3) — the ceiling comes from a real
dependency constraint (`colour-demosaicing` caps at <3.13), not an arbitrary choice.

Clone the repository, then from its root:

```bash
pip install .
```

This installs `pyS3M` as a real package (`import pyS3M.SR_Functions`, etc. works from
anywhere — no `sys.path` hacks needed) along with its core analysis dependencies. Optional
extras layer on top as needed:

```bash
pip install .[notebooks]  # jupyterlab, napari, seaborn, xarray, plotly, ...
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

```python
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
```

See `notebooks/` for fuller worked examples (per-camera calibration, drift correction, FRC,
channel unmixing, simulation).

## License

Copyright © 2026, Cambridge Enterprise Limited, all rights reserved. This software is
provided **for academic use only** — see `LICENSE` for the full text. For commercial use,
contact `ls.ipportfolio@enterprise.cam.ac.uk` quoting LEE-11475-25.

## Contributing

Patches and contributions are very welcome! Please see `CONTRIBUTING.md` and
`CODE_OF_CONDUCT.md` for more details.
