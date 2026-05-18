"""
drift_correction — drift correction subpackage for pyS3M.

Public API re-exported here for convenience; the canonical import path
remains ``import DriftCorrectionFunctions`` for backward compatibility.
"""

from ._base import (
    DriftMethod,
    DriftCorrectionError,
    DriftParameters,
    DriftResult,
    FiducialDetectionResult,
    DriftCorrector,
)
from .aim import AIMDriftCorrector
from .fiducial import FiducialDriftCorrector
from .auto import (
    AutoDriftCorrector,
    DriftCorrectionFactory,
    undrift_aim,
    undrift_auto,
)
from ._facade import Drift_Correction_Functions

__all__ = [
    "DriftMethod",
    "DriftCorrectionError",
    "DriftParameters",
    "DriftResult",
    "FiducialDetectionResult",
    "DriftCorrector",
    "AIMDriftCorrector",
    "FiducialDriftCorrector",
    "AutoDriftCorrector",
    "DriftCorrectionFactory",
    "undrift_aim",
    "undrift_auto",
    "Drift_Correction_Functions",
]
