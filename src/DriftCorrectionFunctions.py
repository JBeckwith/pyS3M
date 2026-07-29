"""
DriftCorrectionFunctions.py

Backward-compatibility shim. The drift correction implementation has been
reorganised into the drift_correction/ subpackage. This module re-exports
everything so existing callers need no changes.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pyS3M.drift_correction._base import (
    DriftMethod,
    DriftCorrectionError,
    DriftParameters,
    DriftResult,
    FiducialDetectionResult,
    DriftCorrector,
)
from pyS3M.drift_correction.aim import AIMDriftCorrector
from pyS3M.drift_correction.fiducial import FiducialDriftCorrector
from pyS3M.drift_correction.auto import (
    AutoDriftCorrector,
    DriftCorrectionFactory,
    undrift_aim,
    undrift_auto,
)
from pyS3M.drift_correction._facade import Drift_Correction_Functions

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
