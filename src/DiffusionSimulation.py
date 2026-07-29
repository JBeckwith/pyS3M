#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backward-compatibility shim — DiffusionSimulation moved to simulation/diffusion.py."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from pyS3M.simulation.diffusion import (  # noqa: F401
    Molecule,
    BindingKinetics,
    LangevinDiffusion2D,
    DiffusionSimulator2D,
    compute_msd_from_trajectory,
    estimate_D_from_msd,
    autocorrFFT,
    msd_fft,
    PMin_XM,
    estimate_D_OLSF,
    CameraAdapter,
)
