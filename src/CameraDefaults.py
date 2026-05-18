#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Camera configuration defaults for pyS3M.

Provides pixel size and Bayer mosaic unit for each supported camera.
Pass ``camera="ximea"`` or ``camera="zwo"`` to any class ``__init__``
to load the appropriate defaults automatically.

To add a new camera, insert an entry in :data:`CAMERAS`::

    CAMERAS["mycam"] = CameraConfig(
        pixel_size=0.065,
        mosaic_unit=np.array([["R", "G"], ["G", "B"]]),
    )
"""

import numpy as np
from dataclasses import dataclass
from pathlib import Path

_QE_DIR = Path(__file__).parent.parent / "Spectra" / "Camera_QE"


@dataclass
class CameraConfig:
    """Hardware-specific defaults for a single camera model.

    Attributes:
        pixel_size: Physical pixel size in µm.
        mosaic_unit: 2×2 Bayer unit-cell array,
            e.g. ``np.array([["B","G"],["G","R"]])``.
        qe_file: Absolute path to the per-channel QE CSV file
            (columns: wavelength, R, G, B).
    """

    pixel_size: float
    mosaic_unit: np.ndarray
    qe_file: str


#: Registry of known camera configurations.
CAMERAS = {
    "ximea": CameraConfig(
        pixel_size=0.069,
        mosaic_unit=np.array([["B", "G"], ["G", "R"]]),
        qe_file=_QE_DIR / "CS505CU_QE.csv",
    ),
    "zwo": CameraConfig(
        pixel_size=0.0715,
        mosaic_unit=np.array([["R", "G"], ["G", "B"]]),
        qe_file=_QE_DIR / "ASI585MC_QE.csv",
    ),
}


def get_camera_config(name: str) -> CameraConfig:
    """Return the :class:`CameraConfig` for *name*.

    Args:
        name: Camera identifier (case-insensitive).
            Currently ``"ximea"`` or ``"zwo"``.

    Returns:
        :class:`CameraConfig` with ``pixel_size`` and ``mosaic_unit``.

    Raises:
        ValueError: If *name* is not in :data:`CAMERAS`.
    """
    key = name.lower()
    if key not in CAMERAS:
        raise ValueError(
            f"Unknown camera '{name}'. "
            f"Available cameras: {sorted(CAMERAS.keys())}"
        )
    return CAMERAS[key]
