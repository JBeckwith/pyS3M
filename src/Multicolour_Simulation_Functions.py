#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backward-compatibility shim — Multicolour_Simulation_Functions moved to simulation/multicolour.py."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from simulation.multicolour import (  # noqa: F401
    FittingStrategy,
    CameraParameters,
    SimulationConfig,
    SimulationValidationError,
    FittingResultProcessor,
    MultiC_Sim_Funcs_Refactored,
    MultiC_Sim_Funcs_Compatibility,
    MultiC_Sim_Funcs,
)
