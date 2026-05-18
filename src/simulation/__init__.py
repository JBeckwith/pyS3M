"""
simulation — simulation subpackage for pyS3M.

Provides:
    diffusion    — 2D Langevin diffusion, MSD, binding kinetics (DiffusionSimulation)
    multicolour  — Bayer-camera simulation, bootstrap fitting, file I/O (Multicolour_Simulation_Functions)

Backward-compat shims in src/ re-export these modules under their original names.
"""

from .diffusion import (
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

from .multicolour import (
    FittingStrategy,
    CameraParameters,
    SimulationConfig,
    SimulationValidationError,
    FittingResultProcessor,
    MultiC_Sim_Funcs_Refactored,
    MultiC_Sim_Funcs_Compatibility,
    MultiC_Sim_Funcs,
)

__all__ = [
    # diffusion
    "Molecule",
    "BindingKinetics",
    "LangevinDiffusion2D",
    "DiffusionSimulator2D",
    "compute_msd_from_trajectory",
    "estimate_D_from_msd",
    "autocorrFFT",
    "msd_fft",
    "PMin_XM",
    "estimate_D_OLSF",
    "CameraAdapter",
    # multicolour
    "FittingStrategy",
    "CameraParameters",
    "SimulationConfig",
    "SimulationValidationError",
    "FittingResultProcessor",
    "MultiC_Sim_Funcs_Refactored",
    "MultiC_Sim_Funcs_Compatibility",
    "MultiC_Sim_Funcs",
]
