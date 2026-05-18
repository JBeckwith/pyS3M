#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AnalysisPipeline — high-level orchestrator for pyS3M experiments.

Wires together calibration, fitting, quality filtering, clustering, and
drift correction into a single interface suitable for GUI and script use.
"""
from __future__ import annotations

import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
import logging
import sys

import numpy as np
import pandas as pd
from numpy.typing import NDArray

sys.path.append(str(Path(__file__).parent))

from Constants import AnalysisConfig, FilteringCriteria
from clustering import ClusteringConfig

logger = logging.getLogger(__name__)


@dataclass
class FittingConfig:
    """Detection and fitting parameters shared across all pipeline modes.

    Fields map directly to the keyword arguments accepted by every
    ``SuperRes_Functions.fit_*`` method.  Mode-specific extras
    (e.g. ``n_frames_sum``, ``use_elliptical``) can be passed as
    ``**kwargs`` to :meth:`AnalysisPipeline.fit`.

    Example::

        fc = FittingConfig(peak_wavelength=0.647, pfa=1e-4)
        pipe.fit(data_dir, mode="imaging", fitting_config=fc)
    """

    pfa: float = 1e-3
    ROI_size: int = 16
    peak_wavelength: float = 0.638
    NA: float = 1.49
    sigma: float = 1.5
    fraction_true: float = 0.2
    image_type: str = ".tif"
    use_variance_aware_demosaic: bool = True


class AnalysisPipeline:
    """High-level orchestrator for pyS3M analysis workflows.

    Holds shared state (calibration maps, camera configuration, sub-function
    instances) and dispatches to the appropriate fitting, filtering, and
    drift-correction routines.

    Sub-function instances (``sr``, ``sm``, ``dcf``) are created lazily on
    first access so construction is cheap.

    Example (headless script)::

        from pathlib import Path
        from AnalysisPipeline import AnalysisPipeline, FittingConfig
        from Constants import AnalysisConfig, FilteringCriteria
        from clustering import ClusteringConfig

        cfg = AnalysisConfig(display=False, save_figures=True,
                             output_dir=Path("results/"), dpi=150)
        pipe = AnalysisPipeline(camera="ximea", config=cfg)
        pipe.load_calibration(Path("Camera_Calibrations/Ximea_Camera"))

        fc = FittingConfig(peak_wavelength=0.638, pfa=1e-3)
        pipe.fit(data_dir, mode="smlm", fitting_config=fc)

        locs = pipe.load_localisations(data_dir)
        sm_db, sf_db = pipe.filter_and_cluster(locs)
    """

    def __init__(
        self,
        camera: str = "ximea",
        pixel_size: float | None = None,
        config: AnalysisConfig | None = None,
    ) -> None:
        """Initialise pipeline with camera defaults.

        Args:
            camera: Camera model name (``"ximea"`` or ``"zwo"``).
            pixel_size: Pixel size in µm.  ``None`` → from camera defaults.
            config: :class:`~Constants.AnalysisConfig` controlling display and
                I/O behaviour.  Defaults to interactive mode with no auto-save.
        """
        import CameraDefaults
        cam_cfg = CameraDefaults.get_camera_config(camera)
        self.camera = camera
        self.pixel_size = pixel_size if pixel_size is not None else cam_cfg.pixel_size
        self.config = config if config is not None else AnalysisConfig()

        # Lazy-initialised sub-function instances
        self._sr: Any | None = None
        self._sm: Any | None = None
        self._dcf: Any | None = None

        # Calibration state — populated by load_calibration() or calibrate()
        self.gain_map: NDArray[np.float32] | None = None
        self.offset_map: NDArray[np.float32] | None = None
        self.rqe: NDArray[np.float32] | None = None
        self.read_noise: NDArray[np.float32] | None = None
        self.variance: NDArray[np.float32] | None = None

        logger.info(
            "AnalysisPipeline initialised (camera=%s, pixel_size=%.4f µm)",
            camera,
            self.pixel_size,
        )

    # ------------------------------------------------------------------
    # Lazy-property accessors for sub-function instances
    # ------------------------------------------------------------------

    @property
    def sr(self) -> Any:
        """Lazily-created :class:`~SR_Functions.SuperRes_Functions` instance."""
        if self._sr is None:
            import SR_Functions
            self._sr = SR_Functions.SuperRes_Functions(
                camera=self.camera,
                pixel_size=self.pixel_size,
                config=self.config,
            )
        return self._sr

    @property
    def sm(self) -> Any:
        """Lazily-created :class:`~SM_extractionfunctions.extract_SMs` instance."""
        if self._sm is None:
            import SM_extractionfunctions
            self._sm = SM_extractionfunctions.extract_SMs(
                camera=self.camera,
                pixel_size=self.pixel_size,
            )
        return self._sm

    @property
    def dcf(self) -> Any:
        """Lazily-created :class:`~DriftCorrectionFunctions.Drift_Correction_Functions` instance."""
        if self._dcf is None:
            import DriftCorrectionFunctions
            self._dcf = DriftCorrectionFunctions.Drift_Correction_Functions(
                camera=self.camera,
                pixel_size=self.pixel_size,
            )
        return self._dcf

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def load_calibration(self, cal_dir: Path | str) -> None:
        """Load pre-computed calibration maps from *cal_dir*.

        Expects files named ``gain.tif``, ``offset.tif``, ``variance.tif``,
        ``readnoise.tif``, and ``rqe.tif`` in *cal_dir*.

        Args:
            cal_dir: Directory containing the calibration ``.tif`` files.

        Raises:
            FileNotFoundError: If any required file is absent.
        """
        import IOFunctions
        io = IOFunctions.IO_Functions()
        cal_dir = Path(cal_dir)

        required = {
            "gain_map":   "gain.tif",
            "offset_map": "offset.tif",
            "variance":   "variance.tif",
            "read_noise": "readnoise.tif",
            "rqe":        "rqe.tif",
        }
        for attr, filename in required.items():
            path = cal_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"Calibration file not found: {path}")
            setattr(self, attr, io.read_tiff(str(path), dtype="float32"))

        logger.info("Calibration loaded from %s", cal_dir)
        if self.config.logging_callback:
            self.config.logging_callback(f"Calibration loaded from {cal_dir}")

    def calibrate(self, cal_dir: Path | str, imtype: str = ".tif") -> None:
        """Compute calibration from flat-field / dark frames in *cal_dir*.

        Runs :meth:`~CalibrationFunctions.Calibration_Functions.calibrate_multicolour_camera`
        and caches the resulting maps so subsequent :meth:`fit` calls can use them.

        Args:
            cal_dir: Directory containing flat-field and dark-frame images.
            imtype: Image file extension (default ``".tif"``).

        Raises:
            RuntimeError: If :meth:`~CalibrationFunctions.Calibration_Functions.calibrate_multicolour_camera`
                returns ``None``.
        """
        import CalibrationFunctions
        cf = CalibrationFunctions.Calibration_Functions(camera=self.camera)
        result = cf.calibrate_multicolour_camera(cal_dir, imtype=imtype)
        if result is None:
            raise RuntimeError(
                f"calibrate_multicolour_camera returned None for {cal_dir}"
            )
        offset, variance, gain, rqe, read_noise = result
        self.offset_map = offset
        self.variance = variance
        self.gain_map = gain
        self.rqe = rqe
        self.read_noise = read_noise

        logger.info("Calibration computed from %s", cal_dir)
        if self.config.logging_callback:
            self.config.logging_callback(f"Calibration computed from {cal_dir}")

    # ------------------------------------------------------------------
    # Smoothing function factory
    # ------------------------------------------------------------------

    def make_smoothing_function(self, sigma: float = 1.5) -> Any:
        """Build the smoothing namespace consumed by the fitting methods.

        Args:
            sigma: Gaussian sigma for smoothing (default ``1.5``).

        Returns:
            ``types.SimpleNamespace`` with ``smoothing_function``, ``args``,
            ``data_arg``, and ``extent`` attributes as expected by
            :meth:`~IOFunctions.IO_Functions.apply_smoothing`.
        """
        import sCMOSFunctions
        scmos = sCMOSFunctions.sCMOS_Functions()
        sf = types.SimpleNamespace()
        sf.smoothing_function = scmos.gaussian_filter_stack
        sf.args = {"sigma": sigma}
        sf.data_arg = "image"
        sf.extent = sigma
        return sf

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def _require_calibration(self) -> None:
        if self.gain_map is None:
            raise RuntimeError(
                "Calibration not loaded. "
                "Call load_calibration() or calibrate() before fit()."
            )

    def fit(
        self,
        image_folder: Path | str,
        mode: Literal["smlm", "fret", "qd", "tracking", "imaging"] = "smlm",
        fitting_config: FittingConfig | None = None,
        smoothing_sigma: float = 1.5,
        **kwargs: Any,
    ) -> None:
        """Run a fitting pipeline on all images in *image_folder*.

        Calibration must have been loaded first via :meth:`load_calibration`
        or :meth:`calibrate`.  Results are written to HDF5 files alongside
        each input TIFF (same convention as the underlying ``fit_*`` methods).

        Args:
            image_folder: Folder containing the TIFF image files.
            mode: Pipeline variant:

                * ``"smlm"``     → :meth:`~SR_Functions.SuperRes_Functions.fit_SM_data` (delegates to ``_fit_files``)
                * ``"fret"``     → :meth:`~SR_Functions.SuperRes_Functions.fit_FRET_data`
                * ``"qd"``       → :meth:`~SR_Functions.SuperRes_Functions.fit_QD_data`
                * ``"tracking"`` → :meth:`~SR_Functions.SuperRes_Functions.fit_tracking_data`
                * ``"imaging"``  → :meth:`~SR_Functions.SuperRes_Functions.fit_imaging_data` (delegates to ``_fit_files``)

            fitting_config: Shared detection / fitting parameters.  ``None``
                uses :class:`FittingConfig` defaults.
            smoothing_sigma: Gaussian sigma for the smoothing pre-filter.
            **kwargs: Extra keyword arguments forwarded verbatim to the
                underlying ``fit_*`` method (e.g. ``n_frames_sum=50``,
                ``use_elliptical=True``).

        Raises:
            RuntimeError: If calibration has not been loaded.
            ValueError: If *mode* is not recognised.
        """
        self._require_calibration()
        fc = fitting_config if fitting_config is not None else FittingConfig()
        sf = self.make_smoothing_function(smoothing_sigma)

        common: dict[str, Any] = dict(
            smoothing_function=sf,
            gain_map=self.gain_map,
            offset_map=self.offset_map,
            rqe=self.rqe,
            read_noise=self.read_noise,
            variance=self.variance,
            pfa=fc.pfa,
            ROI_size=fc.ROI_size,
            peak_wavelength=fc.peak_wavelength,
            NA=fc.NA,
            pixel_size=self.pixel_size,
            sigma=fc.sigma,
            fraction_true=fc.fraction_true,
            image_type=fc.image_type,
            use_variance_aware_demosaic=fc.use_variance_aware_demosaic,
        )
        common.update(kwargs)

        method_map: dict[str, Any] = {
            "smlm":     self.sr.fit_SM_data,
            "fret":     self.sr.fit_FRET_data,
            "qd":       self.sr.fit_QD_data,
            "tracking": self.sr.fit_tracking_data,
            "imaging":  self.sr.fit_imaging_data,
        }
        if mode not in method_map:
            raise ValueError(
                f"Unknown mode {mode!r}. Choose from: {list(method_map)}"
            )

        logger.info("Starting %s fit on %s", mode, image_folder)
        if self.config.progress_callback:
            self.config.progress_callback(0.0, f"Starting {mode} fit")

        method_map[mode](image_folder, **common)

        logger.info("%s fit complete", mode)
        if self.config.progress_callback:
            self.config.progress_callback(1.0, f"{mode} fit complete")

    # ------------------------------------------------------------------
    # Loading results
    # ------------------------------------------------------------------

    def load_localisations(
        self,
        folder: Path | str,
        pattern: str = "*.h5",
        start_frame: int = 0,
    ) -> pd.DataFrame:
        """Load and concatenate HDF5 localisation files from *folder*.

        Args:
            folder: Folder containing the ``.h5`` result files.
            pattern: Glob pattern for file selection (default ``"*.h5"``).
            start_frame: Discard localisations with ``frame < start_frame``.

        Returns:
            Concatenated :class:`~pandas.DataFrame` of all localisations,
            or an empty DataFrame if no matching files are found.
        """
        import IOFunctions
        io = IOFunctions.IO_Functions()
        h5_files = sorted(Path(folder).glob(pattern))
        if not h5_files:
            logger.warning("No %s files found in %s", pattern, folder)
            return pd.DataFrame()

        dfs = [io.read_h5_database(str(f)) for f in h5_files]
        df = pd.concat(dfs, ignore_index=True)
        if start_frame > 0:
            df = df[df["frame"] >= start_frame].reset_index(drop=True)

        logger.info(
            "Loaded %d localisations from %d file(s) in %s",
            len(df), len(h5_files), folder,
        )
        return df

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def filter_and_cluster(
        self,
        locs: pd.DataFrame,
        criteria: FilteringCriteria | None = None,
        clustering_config: ClusteringConfig | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Quality-filter and cluster a localisation DataFrame.

        Delegates to one of the ``extract_single_molecules_*`` methods on
        :attr:`sm`.  Quality filtering is performed inside that method.

        Args:
            locs: Raw localisation data (e.g. from :meth:`load_localisations`).
            criteria: Quality-filtering thresholds.  ``None`` → defaults from
                :class:`~Constants.FilteringCriteria`.
            clustering_config: Algorithm parameters.  ``None`` → defaults from
                :class:`~clustering.ClusteringConfig` (HDBSCAN,
                ``min_cluster_size=10``).

        Returns:
            ``(sm_db, sf_db)`` as returned by the chosen
            ``extract_single_molecules_*`` method:

            * *sm_db* — averaged single-molecule database
            * *sf_db* — single-frame database with ``molecular_index`` column

        Raises:
            ValueError: If ``clustering_config.clustering_method`` is not
                ``"HDBSCAN"``, ``"DBSCAN"``, or ``"LINKED"``.
        """
        cc = clustering_config if clustering_config is not None else ClusteringConfig()
        filt = criteria if criteria is not None else FilteringCriteria()

        method = cc.clustering_method.upper()
        dispatch: dict[str, Any] = {
            "HDBSCAN": self.sm.extract_single_molecules_HDBSCAN,
            "DBSCAN":  self.sm.extract_single_molecules_DBSCAN,
            "LINKED":  self.sm.extract_single_molecules_linked,
        }
        if method not in dispatch:
            raise ValueError(
                f"Unknown clustering method {method!r}. "
                f"Choose from: {list(dispatch)}"
            )

        logger.info("Clustering with %s (min_cluster_size=%d)", method, cc.min_cluster_size)
        return dispatch[method](locs, config=cc, criteria=filt)

    # ------------------------------------------------------------------
    # Drift correction
    # ------------------------------------------------------------------

    def undrift(
        self,
        locs: np.recarray,
        info: list,
        method: str = "auto",
        **params: Any,
    ) -> tuple[np.recarray, Any]:
        """Apply drift correction to localisation data.

        Args:
            locs: Localisation recarray.
            info: Picasso-style metadata list.
            method: Drift correction method (``"aim"``, ``"fiducial"``,
                or ``"auto"``).
            **params: Forwarded verbatim to the underlying corrector.

        Returns:
            ``(corrected_locs, drift_result)`` tuple from
            :meth:`~DriftCorrectionFunctions.Drift_Correction_Functions.undrift`.
        """
        return self.dcf.undrift(locs, info, method=method, **params)
