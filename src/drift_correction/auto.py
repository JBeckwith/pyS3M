"""
drift_correction/auto.py

AutoDriftCorrector, DriftCorrectionFactory, and backward-compatible
module-level convenience functions.

:authors: Claude Code (based on Joerg Schnitzbauer, Maximilian Thomas Strauss, Hongqiang Ma, Maomao Chen)
:copyright: Copyright (c) 2025 pyS3M
"""

import warnings
from typing import Optional, Callable, Tuple

import numpy as np

from ._base import DriftMethod, DriftCorrectionError, DriftParameters, DriftResult, DriftCorrector
from .aim import AIMDriftCorrector
from .fiducial import FiducialDriftCorrector

try:
    from CoordinateProcessing import CoordinateProcessor, SegmentationHandler
except ImportError:
    warnings.warn("Could not import CoordinateProcessing. AutoDriftCorrector may be limited.")
    CoordinateProcessor = None
    SegmentationHandler = None


class AutoDriftCorrector(DriftCorrector):
    """Automatic drift corrector — always uses AIM."""

    def __init__(self):
        self.aim_corrector = AIMDriftCorrector()

    def supports_3d(self) -> bool:
        return True

    def calculate_drift(
        self, locs: np.recarray, info: list, params: DriftParameters
    ) -> DriftResult:
        """Apply AIM drift correction."""
        meta = CoordinateProcessor.extract_metadata(info)
        n_segments = max(1, SegmentationHandler.n_segments(
            int(meta["n_frames"]), params.segmentation
        ))
        avg_locs_per_segment = len(locs) / n_segments

        result = self.aim_corrector.calculate_drift(locs, info, params)
        result.method_used = DriftMethod.AIM
        result.metadata["auto_selection_reason"] = (
            f"Selected aim based on {avg_locs_per_segment:.1f} locs/segment"
        )

        return result


class DriftCorrectionFactory:
    """Factory for creating drift correctors."""

    _correctors = {
        DriftMethod.AIM: AIMDriftCorrector,
        DriftMethod.FIDUCIAL: FiducialDriftCorrector,
        DriftMethod.AUTO: AutoDriftCorrector,
    }

    @classmethod
    def create_corrector(cls, method: DriftMethod) -> DriftCorrector:
        """Create drift corrector instance.

        Args:
            method: Drift correction method

        Returns:
            Drift corrector instance

        Raises:
            DriftCorrectionError: If method not supported
        """
        if method not in cls._correctors:
            raise DriftCorrectionError(f"Unsupported drift method: {method}")

        return cls._correctors[method]()

    @classmethod
    def available_methods(cls) -> list:
        """Get list of available drift correction methods."""
        return list(cls._correctors.keys())


# Convenience functions for backward compatibility
def undrift_aim(
    locs: np.recarray,
    info: list,
    segmentation: int = 100,
    intersect_d: float = 20 / 69,
    roi_r: float = 60 / 69,
    progress: Optional[Callable] = None,
) -> Tuple[np.recarray, DriftResult]:
    """Apply AIM drift correction (backward compatible interface).

    Args:
        locs: Localisation data
        info: Metadata list
        segmentation: Frames per segment
        intersect_d: Intersection distance (camera pixels)
        roi_r: Search region radius (camera pixels)
        progress: Progress callback

    Returns:
        Tuple of (corrected_locs, drift_result)
    """
    params = DriftParameters(
        segmentation=segmentation,
        intersect_d=intersect_d,
        roi_r=roi_r,
        progress_callback=progress,
    )

    corrector = DriftCorrectionFactory.create_corrector(DriftMethod.AIM)
    return corrector.correct_drift(locs, info, params)


def undrift_auto(
    locs: np.recarray, info: list, **kwargs
) -> Tuple[np.recarray, DriftResult]:
    """Apply automatic drift correction method selection.

    Args:
        locs: Localisation data
        info: Metadata list
        **kwargs: Parameters passed to DriftParameters

    Returns:
        Tuple of (corrected_locs, drift_result)
    """
    params = DriftParameters(**kwargs)

    corrector = DriftCorrectionFactory.create_corrector(DriftMethod.AUTO)
    return corrector.correct_drift(locs, info, params)
