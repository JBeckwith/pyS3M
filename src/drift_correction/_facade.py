"""
drift_correction/_facade.py

Drift_Correction_Functions — the main public-facing class providing
the full drift correction API. Kept as a single unit for backward
compatibility; delegates strategy execution to the corrector classes
in aim.py, fiducial.py, and auto.py.

:authors: Claude Code (based on Joerg Schnitzbauer, Maximilian Thomas Strauss, Hongqiang Ma, Maomao Chen)
:copyright: Copyright (c) 2025 pyBayerSMLM
"""

import gc
import warnings
from typing import Optional, Callable, Tuple, Union, Dict, Any, List

import numpy as np
import matplotlib.pyplot as plt

import ProgressUtils
from ._base import (
    DriftMethod, DriftCorrectionError, DriftParameters,
    DriftResult, FiducialDetectionResult,
)
from .auto import DriftCorrectionFactory
from .fiducial import FiducialDriftCorrector
import logging
logger = logging.getLogger(__name__)


try:
    import render
    import postprocess
except ImportError:
    warnings.warn("Could not import render/postprocess modules.")
    render = None
    postprocess = None

try:
    from CoordinateProcessing import CoordinateProcessor
except ImportError:
    warnings.warn("Could not import CoordinateProcessing.")
    CoordinateProcessor = None


class Drift_Correction_Functions:
    """Main class providing drift correction functionality.

    This class follows the established pattern in the codebase of
    organizing functions within a class structure.
    """

    def __init__(self, camera: str = "ximea", pixel_size: float = None):
        """Initialize drift correction functions.

        Args:
            camera: Camera model name (``"ximea"`` or ``"zwo"``). Sets pixel_size
                used as default when building DriftParameters.
            pixel_size: Physical pixel size in µm. If None, taken from camera defaults.
        """
        import CameraDefaults
        config = CameraDefaults.get_camera_config(camera)
        self.pixel_size = pixel_size if pixel_size is not None else config.pixel_size

        self.factory = DriftCorrectionFactory()

        # Initialize plotting and specialised algorithm modules
        try:
            from FiducialDetection import DriftPlotter, FiducialDetector

            self.plotter = DriftPlotter()
            self.fiducial_detector = FiducialDetector(drift_correction_instance=self)
        except ImportError:
            self.plotter = None
            self.fiducial_detector = None

        try:
            from AIMAlgorithm import AIMAlgorithm

            self.aim_algorithm = AIMAlgorithm(drift_correction_instance=self)
        except ImportError:
            self.aim_algorithm = None

        try:
            from CoordinateProcessing import CoordinateProcessor

            self.coordinate_processor = CoordinateProcessor()
        except ImportError:
            self.coordinate_processor = None

    def undrift(
        self,
        locs: np.recarray,
        info: list,
        method: Union[str, DriftMethod] = "auto",
        **params,
    ) -> Tuple[np.recarray, DriftResult]:
        """Universal drift correction interface.

        Args:
            locs: Localisation data
            info: Metadata list
            method: Drift correction method ("aim", "fiducial", "auto")
            **params: Method-specific parameters

        Returns:
            Tuple of (corrected_locs, drift_result)

        Example:
            >>> DCF = Drift_Correction_Functions()
            >>> corrected_locs, drift = DCF.undrift(locs, info, method="aim", segmentation=50)
            >>> corrected_locs, drift = DCF.undrift(locs, info, method="fiducial")
        """
        # Convert string to enum if needed
        if isinstance(method, str):
            method = DriftMethod(method.lower())

        # Create parameters
        drift_params = DriftParameters(**params)

        # Get corrector and apply
        corrector = self.factory.create_corrector(method)
        return corrector.correct_drift(locs, info, drift_params)

    def available_methods(self) -> list:
        """Get available drift correction methods."""
        return [method.value for method in self.factory.available_methods()]

    def method_info(self, method: Union[str, DriftMethod]) -> Dict[str, Any]:
        """Get information about a drift correction method.

        Args:
            method: Drift method to query

        Returns:
            Dictionary with method information
        """
        if isinstance(method, str):
            method = DriftMethod(method.lower())

        corrector = self.factory.create_corrector(method)

        return {
            "name": method.value,
            "supports_3d": corrector.supports_3d(),
            "class": corrector.__class__.__name__,
            "description": (
                corrector.__class__.__doc__.split("\n")[0]
                if corrector.__class__.__doc__
                else ""
            ),
        }

    # Delegation methods for specialised algorithm modules

    # Fiducial Detection delegation methods
    def identify_real_fiducials_with_clustering_delegated(self, *args, **kwargs):
        """Delegate to FiducialDetector.identify_real_fiducials_with_clustering"""
        if self.fiducial_detector is None:
            raise RuntimeError("FiducialDetector module not available")
        return self.fiducial_detector.identify_real_fiducials_with_clustering(
            *args, **kwargs
        )

    # AIM Algorithm delegation methods
    def run_aim_2d(self, *args, **kwargs):
        """Delegate to AIMAlgorithm.run_aim_2d"""
        if self.aim_algorithm is None:
            raise RuntimeError("AIMAlgorithm module not available")
        return self.aim_algorithm.run_aim_2d(*args, **kwargs)

    def run_aim_3d(self, *args, **kwargs):
        """Delegate to AIMAlgorithm.run_aim_3d"""
        if self.aim_algorithm is None:
            raise RuntimeError("AIMAlgorithm module not available")
        return self.aim_algorithm.run_aim_3d(*args, **kwargs)

    # Coordinate Processing delegation methods
    def convert_pixels_to_nm(self, *args, **kwargs):
        """Delegate to CoordinateProcessor.convert_pixels_to_nm"""
        if self.coordinate_processor is None:
            raise RuntimeError("CoordinateProcessor module not available")
        return self.coordinate_processor.convert_pixels_to_nm(*args, **kwargs)

    def undrift_with_fiducial_detection(
        self,
        locs: np.recarray,
        info: list,
        histogram_bins: int = 256,
        threshold_percentile: float = 99.0,
        box_size_nm: float = 600.0,
        min_localisations_per_region: int = 100,
        retention_percentage: float = 0.9,
        create_plots: bool = False,
        output_dir: str = "./fiducial_detection",
    ) -> DriftResult:
        """Automatically detect fiducials and perform drift correction.

        This is a high-level convenience method that:
        1. Renders localisations to an image
        2. Detects high-density regions (potential fiducials)
        3. Selects localisations within those regions
        4. Validates fiducials using clustering
        5. Performs fiducial-based drift correction

        Args:
            locs: Localisation data (xc, yc, frame fields required)
            info: Metadata list containing image dimensions and frame info
            histogram_bins: Number of bins for histogram analysis
            threshold_percentile: Percentile threshold for fiducial detection (0-100)
            box_size_nm: Size of selection box around each fiducial (nm)
            min_localisations_per_region: Minimum localisations required per fiducial
            retention_percentage: Fraction of points to retain during validation (0-1)
            create_plots: Whether to create diagnostic plots
            output_dir: Directory to save plots (if create_plots=True)

        Returns:
            DriftResult object with drift_x, drift_y arrays and metadata

        Raises:
            DriftCorrectionError: If fiducial detection or drift correction fails
        """
        # Extract metadata
        meta = CoordinateProcessor.extract_metadata(info)
        pixelsize = meta.get("pixelsize", 100.0)
        n_frames = int(meta["n_frames"])

        # Step 1: Render localisations to image
        logger.info("Step 1/5: Rendering localisations to image...")
        if render is None:
            raise DriftCorrectionError("render module required for fiducial detection")

        _, image = render.render(
            locs=locs,
            info=info,
            oversampling=1,
            blur_method="smooth",
        )

        # Step 2: Detect high-density regions
        logger.info("Step 2/5: Detecting high-density regions...")
        region_centres, binary_mask, threshold, detection_meta = (
            self.fiducial_detector.detect_high_density_regions_from_image(
                smoothed_image=image,
                histogram_bins=histogram_bins,
                threshold_percentile=threshold_percentile,
                pixelsize=pixelsize,
                output_figure_path=(
                    f"{output_dir}/01_density_detection.png" if create_plots else None
                ),
                create_plot=create_plots,
            )
        )

        logger.info(f"  Found {detection_meta['n_regions_detected']} potential fiducial regions")

        if detection_meta["n_regions_detected"] == 0:
            raise DriftCorrectionError(
                "No fiducial regions detected. Try lowering threshold_percentile."
            )

        # Step 3: Select puncta from regions
        logger.info("Step 3/5: Selecting localisations from regions...")
        selected_puncta, selection_meta = (
            self.fiducial_detector.select_puncta_from_regions(
                locs=locs,
                region_centres=region_centres,
                binary_mask=binary_mask,
                pixelsize=pixelsize,
                selection_box_size_nm=box_size_nm,
                min_localisations_per_region=min_localisations_per_region,
                output_figure_path=(
                    f"{output_dir}/02_puncta_selection.png" if create_plots else None
                ),
                create_plot=create_plots,
            )
        )

        logger.info(f"  Selected {selection_meta['n_regions_selected']} fiducial candidates")

        if selection_meta["n_regions_selected"] == 0:
            raise DriftCorrectionError(
                f"No valid fiducials with >={min_localisations_per_region} localisations. "
                "Try lowering min_localisations_per_region or threshold_percentile."
            )

        # Step 4: Validate fiducials using clustering
        logger.info("Step 4/5: Validating fiducials with clustering...")
        validated_fiducials, validation_meta = (
            self.fiducial_detector.identify_real_fiducials_with_clustering(
                selected_puncta=selected_puncta,
                retention_percentage=retention_percentage,
                pixelsize=pixelsize,
                output_figure_path=(
                    f"{output_dir}/03_fiducial_validation.png" if create_plots else None
                ),
                create_plot=create_plots,
            )
        )

        logger.info(f"  Validated {len(validated_fiducials)} final fiducials")

        if len(validated_fiducials) == 0:
            raise DriftCorrectionError(
                "No fiducials passed validation. Check your detection parameters."
            )

        # Step 5: Add group field and perform drift correction
        logger.info("Step 5/5: Performing fiducial-based drift correction...")
        locs_with_groups = self._add_group_field(
            locs, validated_fiducials, region_centres
        )

        # Use the fiducial corrector directly
        fiducial_corrector = FiducialDriftCorrector()
        params = DriftParameters()  # Use default parameters
        result = fiducial_corrector.calculate_drift(locs_with_groups, info, params)

        # Add detection metadata to result
        result.metadata.update(
            {
                "detection_method": "automatic",
                "n_regions_detected": detection_meta["n_regions_detected"],
                "n_fiducials_selected": selection_meta["n_regions_selected"],
                "n_fiducials_validated": len(validated_fiducials),
                "detection_params": {
                    "histogram_bins": histogram_bins,
                    "threshold_percentile": threshold_percentile,
                    "box_size_nm": box_size_nm,
                    "min_localisations_per_region": min_localisations_per_region,
                    "retention_percentage": retention_percentage,
                },
            }
        )

        logger.info(f"✓ Drift correction complete using {len(validated_fiducials)} fiducials")

        return result

    def _add_group_field(
        self, locs: np.recarray, picked_locs: list, picks: list
    ) -> np.recarray:
        """Add group field to localisations based on fiducial assignments.

        Args:
            locs: Original localisations
            picked_locs: List of localisations for each fiducial
            picks: List of pick coordinates (not used, for compatibility)

        Returns:
            New recarray with group field added
        """
        # Create group field array, initialize with -1 (non-fiducial)
        group = np.full(len(locs), -1, dtype=np.int32)

        # Assign group IDs to fiducial localisations
        for group_id, fiducial_locs in enumerate(picked_locs):
            # Find indices of these localisations in original array
            for fid_loc in fiducial_locs:
                # Match by frame and coordinate (within small tolerance)
                matches = (
                    (locs.frame == fid_loc.frame)
                    & (np.abs(locs.xc - fid_loc.xc) < 0.1)
                    & (np.abs(locs.yc - fid_loc.yc) < 0.1)
                )
                group[matches] = group_id

        # Create new dtype with group field
        original_dtype = locs.dtype
        group_dtype = np.dtype(original_dtype.descr + [("group", "i4")])

        # Create new recarray with group field
        new_locs = np.empty(len(locs), dtype=group_dtype)

        # Copy original data
        for field in original_dtype.names:
            new_locs[field] = locs[field]

        # Add group data
        new_locs["group"] = group

        # Convert to recarray
        return new_locs.view(np.recarray)

    def convert_nm_to_pixels(self, *args, **kwargs):
        """Delegate to CoordinateProcessor.convert_nm_to_pixels"""
        if self.coordinate_processor is None:
            raise RuntimeError("CoordinateProcessor module not available")
        return self.coordinate_processor.convert_nm_to_pixels(*args, **kwargs)

    def apply_drift_correction(self, *args, **kwargs):
        """Delegate to CoordinateProcessor.apply_drift_correction"""
        if self.coordinate_processor is None:
            raise RuntimeError("CoordinateProcessor module not available")
        return self.coordinate_processor.apply_drift_correction(*args, **kwargs)

    def create_spatial_grid(self, *args, **kwargs):
        """Delegate to CoordinateProcessor.create_spatial_grid"""
        if self.coordinate_processor is None:
            raise RuntimeError("CoordinateProcessor module not available")
        return self.coordinate_processor.create_spatial_grid(*args, **kwargs)

    def bin_localisations_spatially(self, *args, **kwargs):
        """Delegate to CoordinateProcessor.bin_localisations_spatially"""
        if self.coordinate_processor is None:
            raise RuntimeError("CoordinateProcessor module not available")
        return self.coordinate_processor.bin_localisations_spatially(*args, **kwargs)

    def _detect_fiducials_with_chunking(
        self,
        locs: np.recarray,
        info: list,
        threshold_percentile: float,
        box_size_nm: float,
        histogram_bins: int,
        n_chunks: int,
        max_linking_distance_nm: float,
        pixelsize: float,
    ) -> tuple:
        """Detect fiducials using temporal chunking for drift-robust detection."""
        try:
            import localise
            import render
        except ImportError:
            raise DriftCorrectionError(
                "localise and render modules required for chunked fiducial detection"
            )

        # Get frame range
        min_frame = int(locs.frame.min())
        max_frame = int(locs.frame.max())
        total_frames = max_frame - min_frame + 1

        # Create temporal chunks
        chunk_size = total_frames // n_chunks
        chunk_boundaries = []
        for i in range(n_chunks):
            start_frame = min_frame + i * chunk_size
            if i == n_chunks - 1:
                end_frame = max_frame  # Include remaining frames in last chunk
            else:
                end_frame = min_frame + (i + 1) * chunk_size - 1
            chunk_boundaries.append((start_frame, end_frame))

        logger.info(f"Detecting fiducials using {n_chunks} temporal chunks")

        # Find candidates in each chunk
        chunk_candidates = []
        chunk_images = []
        all_chunk_histograms = []

        for chunk_idx, (start_frame, end_frame) in enumerate(chunk_boundaries):
            # Extract localisations for this chunk
            chunk_mask = (locs.frame >= start_frame) & (locs.frame <= end_frame)
            chunk_locs = locs[chunk_mask]

            if len(chunk_locs) == 0:
                logger.warning(f"Warning: Chunk {chunk_idx + 1} has no localisations")
                continue

            logger.info(f"Chunk {chunk_idx + 1}/{n_chunks}: frames {start_frame}-{end_frame} ({len(chunk_locs)} locs)")

            # Render this chunk
            chunk_image = render.render(
                locs=chunk_locs,
                info=info,
                oversampling=1,
                viewport=None,
                blur_method="smooth",
            )[1]
            chunk_images.append(chunk_image)

            # Create histogram for this chunk
            chunk_hist = np.histogram(chunk_image.flatten(), bins=histogram_bins)
            all_chunk_histograms.append(chunk_hist[0])

            # Use threshold percentile for this chunk
            chunk_threshold = np.percentile(chunk_hist[0], threshold_percentile)

            # Calculate box size
            box = int(np.round(box_size_nm / pixelsize))
            box = box + 1 if box % 2 == 0 else box  # Ensure odd

            # Find candidates in this chunk
            try:
                y, x, _ = localise.identify_in_image(
                    chunk_image, chunk_threshold, box=box
                )
                half_box = box // 2
                chunk_picks = [
                    (xi, yi, chunk_idx, (start_frame + end_frame) / 2)
                    for xi, yi in zip(x, y)
                ]
                chunk_candidates.extend(chunk_picks)
                logger.info(f"  Found {len(chunk_picks)} candidates")
            except Exception as e:
                logger.warning(f"  Warning: Failed to detect in chunk {chunk_idx + 1}: {e}")
                continue

        logger.info(f"Total candidates across all chunks: {len(chunk_candidates)}")

        # Link candidates across chunks to form tracks
        if len(chunk_candidates) == 0:
            raise DriftCorrectionError("No candidates found in any temporal chunk")

        linked_tracks = self._link_candidates_across_chunks(
            chunk_candidates, n_chunks, max_linking_distance_nm, pixelsize
        )

        logger.info(f"Linked candidates into {len(linked_tracks)} potential fiducial tracks")

        # Convert tracks back to picks (use average position)
        half_box = int(np.round(box_size_nm / pixelsize)) // 2
        picks = []
        for track in linked_tracks:
            if (
                len(track) >= n_chunks * 0.6
            ):  # Require track to appear in >60% of chunks
                avg_x = np.mean([pos[0] for pos in track])
                avg_y = np.mean([pos[1] for pos in track])
                picks.append(((avg_x - half_box, avg_y), (avg_x + half_box, avg_y)))

        # Create combined image and histogram for visualization
        if len(chunk_images) > 0:
            combined_image = np.mean(chunk_images, axis=0)
            combined_hist_counts = np.sum(all_chunk_histograms, axis=0)
            if len(all_chunk_histograms) > 0:
                bin_edges = np.histogram(combined_image.flatten(), bins=histogram_bins)[1]
                combined_hist = (combined_hist_counts, bin_edges)
            else:
                combined_hist = np.histogram(combined_image.flatten(), bins=histogram_bins)
            threshold = np.percentile(combined_hist_counts, threshold_percentile)
        else:
            combined_image = np.zeros((100, 100))
            combined_hist = np.histogram(combined_image.flatten(), bins=histogram_bins)
            threshold = 0

        logger.info(f"Final result: {len(picks)} robust fiducial candidates")
        return picks, combined_image, combined_hist, threshold

    def _link_candidates_across_chunks(
        self, candidates: list, n_chunks: int, max_distance_nm: float, pixelsize: float
    ) -> list:
        """Link candidates across temporal chunks to form tracks."""
        max_distance_pixels = max_distance_nm / pixelsize

        # Group candidates by chunk
        chunks_candidates = [[] for _ in range(n_chunks)]
        for candidate in candidates:
            x, y, chunk_idx, avg_frame = candidate
            chunks_candidates[chunk_idx].append((x, y, chunk_idx, avg_frame))

        # Start tracks from first chunk
        tracks = []
        for candidate in chunks_candidates[0]:
            tracks.append([candidate])

        # Extend tracks through subsequent chunks
        for chunk_idx in range(1, n_chunks):
            chunk_candidates = chunks_candidates[chunk_idx]

            for track in tracks:
                if len(track) == 0:
                    continue

                last_pos = track[-1]
                last_x, last_y = last_pos[0], last_pos[1]

                best_candidate = None
                best_distance = float("inf")

                for candidate in chunk_candidates:
                    x, y = candidate[0], candidate[1]
                    distance = np.sqrt((x - last_x) ** 2 + (y - last_y) ** 2)

                    if distance < max_distance_pixels and distance < best_distance:
                        best_distance = distance
                        best_candidate = candidate

                if best_candidate is not None:
                    track.append(best_candidate)
                    chunk_candidates.remove(best_candidate)

            for remaining_candidate in chunk_candidates:
                tracks.append([remaining_candidate])

        min_length = int(n_chunks * 0.6)
        robust_tracks = [track for track in tracks if len(track) >= min_length]

        return robust_tracks

    def detect_fiducials(
        self,
        locs: np.recarray,
        info: list,
        threshold_percentile: float = 99.0,
        box_size_nm: float = 900.0,
        min_frames_fraction: float = 0.8,
        histogram_bins: int = 256,
        plot_results: bool = True,
        save_plot: Optional[str] = None,
        use_temporal_chunking: bool = True,
        n_chunks: int = 10,
        max_linking_distance_nm: float = 500.0,
    ) -> FiducialDetectionResult:
        """Detect fiducial markers in localisation data."""
        if render is None:
            raise DriftCorrectionError(
                "Fiducial detection requires render module"
            )

        # Extract metadata
        meta = CoordinateProcessor.extract_metadata(info)
        pixelsize = meta.get("pixelsize", 100.0)  # nm
        n_frames = int(meta["n_frames"])
        width = int(meta["width"])
        height = int(meta["height"])

        # Store detection parameters
        detection_params = {
            "threshold_percentile": threshold_percentile,
            "box_size_nm": box_size_nm,
            "min_frames_fraction": min_frames_fraction,
            "histogram_bins": histogram_bins,
            "pixelsize": pixelsize,
        }

        # Calculate box size from nanometer specification
        box = int(np.round(box_size_nm / pixelsize))
        box = box + 1 if box % 2 == 0 else box  # Ensure odd

        try:
            if use_temporal_chunking:
                picks, image, hist, threshold = self._detect_fiducials_with_chunking(
                    locs,
                    info,
                    threshold_percentile,
                    box_size_nm,
                    histogram_bins,
                    n_chunks,
                    max_linking_distance_nm,
                    pixelsize,
                )
            else:
                image = render.render(
                    locs=locs,
                    info=info,
                    oversampling=1,
                    viewport=None,
                    blur_method="smooth",
                )[1]

                hist = np.histogram(image.flatten(), bins=histogram_bins)
                threshold = np.percentile(hist[0], threshold_percentile)

                try:
                    import localise
                except ImportError:
                    raise DriftCorrectionError(
                        "localise module required for fiducial detection"
                    )

                y, x, _ = localise.identify_in_image(image, threshold, box=box)
                half_box = box // 2
                picks = [
                    ((xi - half_box, yi), (xi + half_box, yi)) for xi, yi in zip(x, y)
                ]

            if len(picks) == 0:
                raise DriftCorrectionError(
                    f"No fiducial candidates detected with threshold percentile {threshold_percentile}%. "
                    "Try lowering threshold_percentile."
                )

            # Filter picks by minimum localisations per fiducial
            min_n = min_frames_fraction * n_frames

            try:
                import postprocess
            except ImportError:
                raise DriftCorrectionError(
                    "postprocess module required for fiducial detection"
                )

            temp_picked_locs = postprocess.picked_locs(
                locs,
                width,
                height,
                picks,
                "Rectangle",
                pick_size=box,
                add_group=False,
                parallel=True,
            )

            valid_picks = []
            valid_picked_locs = []
            for i, pick in enumerate(picks):
                if len(temp_picked_locs[i]) > min_n:
                    valid_picks.append(pick)
                    valid_picked_locs.append(temp_picked_locs[i])

            if len(valid_picks) == 0:
                raise DriftCorrectionError(
                    f"No fiducials found with minimum {min_n:.0f} localisations. "
                    f"Try lowering min_frames_fraction (currently {min_frames_fraction}) "
                    f"or threshold_percentile (currently {threshold_percentile}%)."
                )

            locs_with_groups = self._add_group_field_to_locs(locs, valid_picked_locs)

            result = FiducialDetectionResult(
                picks=valid_picks,
                picked_localisations=valid_picked_locs,
                detection_image=image,
                locs_with_groups=locs_with_groups,
                n_fiducials=len(valid_picks),
                detection_params=detection_params,
                metadata={
                    "total_candidates": len(picks),
                    "threshold_used": threshold,
                    "box_size_pixels": box,
                    "min_localisations_required": min_n,
                    "localisations_per_fiducial": [
                        len(locs) for locs in valid_picked_locs
                    ],
                },
            )

            if plot_results:
                if self.plotter is not None:
                    self.plotter.plot_fiducial_detection_steps(
                        image,
                        hist,
                        threshold,
                        picks,
                        valid_picks,
                        result,
                        info,
                        save_plot,
                    )
                else:
                    logger.warning("⚠️ DriftPlotter not available, skipping step-by-step plots")

            return result

        except Exception as e:
            if isinstance(e, DriftCorrectionError):
                raise
            else:
                raise DriftCorrectionError(f"Fiducial detection failed: {str(e)}")

    def _add_group_field_to_locs(
        self, locs: np.recarray, picked_locs_list: List[np.recarray]
    ) -> np.recarray:
        """Add group field to localisations based on fiducial assignments."""
        group = np.full(len(locs), -1, dtype=np.int32)

        if len(picked_locs_list) > 0:
            show_progress = len(locs) > 500_000 or len(picked_locs_list) > 5
            progress_bar_context = None
            progress_bar = None

            if show_progress:
                progress_bar_context = ProgressUtils.clean_progress_bar(
                    total=len(picked_locs_list),
                    desc=f"Adding group field to {len(locs):,} localisations (index-based)",
                )
                progress_bar = progress_bar_context.__enter__()

            try:
                for group_id, fiducial_locs in enumerate(picked_locs_list):
                    if len(fiducial_locs) > 0:
                        indices = self._find_indices_in_original_locs(
                            locs, fiducial_locs
                        )
                        group[indices] = group_id

                        if progress_bar:
                            progress_bar.update(1)

            finally:
                if progress_bar_context:
                    progress_bar_context.__exit__(None, None, None)

        try:
            import lib

            return lib.append_to_rec(locs, group, "group")
        except ImportError:
            return self._manual_add_group_field(locs, group)

    def _find_indices_in_original_locs(
        self, locs: np.recarray, fiducial_locs: np.recarray
    ) -> np.ndarray:
        """Find indices of fiducial localisations in the original localisation array."""
        round_factor = 1e6

        locs_frames = locs.frame.astype(np.int32)
        locs_xc_rounded = np.round(locs.xc * round_factor).astype(np.int64)
        locs_yc_rounded = np.round(locs.yc * round_factor).astype(np.int64)

        hash_to_index = {}
        for i, (frame, x_rounded, y_rounded) in enumerate(
            zip(locs_frames, locs_xc_rounded, locs_yc_rounded)
        ):
            key = (frame, x_rounded, y_rounded)

            if key in hash_to_index:
                if isinstance(hash_to_index[key], list):
                    hash_to_index[key].append(i)
                else:
                    hash_to_index[key] = [hash_to_index[key], i]
            else:
                hash_to_index[key] = i

        indices = []

        fid_frames = fiducial_locs.frame.astype(np.int32)
        fid_xc_rounded = np.round(fiducial_locs.xc * round_factor).astype(np.int64)
        fid_yc_rounded = np.round(fiducial_locs.yc * round_factor).astype(np.int64)

        for frame, x_rounded, y_rounded in zip(
            fid_frames, fid_xc_rounded, fid_yc_rounded
        ):
            key = (frame, x_rounded, y_rounded)

            if key in hash_to_index:
                idx_or_list = hash_to_index[key]
                if isinstance(idx_or_list, list):
                    indices.extend(idx_or_list)
                else:
                    indices.append(idx_or_list)

        return np.array(indices, dtype=np.int64)

    def _manual_add_group_field(
        self, locs: np.recarray, group: np.ndarray
    ) -> np.recarray:
        """Fallback method for adding group field manually."""
        original_dtype = locs.dtype
        group_dtype = np.dtype(original_dtype.descr + [("group", "i4")])

        new_locs = np.empty(len(locs), dtype=group_dtype)

        for field in original_dtype.names:
            new_locs[field] = locs[field]

        new_locs["group"] = group

        return new_locs.view(np.recarray)

    def detect_high_density_regions_from_image(
        self,
        smoothed_image: np.ndarray,
        histogram_bins: int = 256,
        threshold_percentile: float = 99.0,
        pixelsize: float = 100.0,
        output_figure_path: Optional[str] = None,
        title: str = "High-Density Region Detection",
        create_plot: bool = True,
    ) -> Tuple[List[Tuple[int, int]], np.ndarray, float, Dict[str, Any]]:
        """Detect high-density regions from a smoothed image using histogram analysis."""
        image_flat = smoothed_image.ravel()
        image_flat = image_flat[image_flat > 0]

        if len(image_flat) == 0:
            raise DriftCorrectionError("Image contains no non-zero values")

        hist, bin_edges = np.histogram(image_flat, bins=histogram_bins)
        threshold = np.percentile(image_flat, threshold_percentile)

        binary_mask = smoothed_image > threshold

        from scipy import ndimage

        labeled_regions, n_regions = ndimage.label(binary_mask)

        region_centres = []
        region_stats = []

        for region_id in range(1, n_regions + 1):
            region_mask = labeled_regions == region_id
            region_coords = np.where(region_mask)

            if len(region_coords[0]) > 0:
                centre_y = np.mean(region_coords[0])
                centre_x = np.mean(region_coords[1])
                region_centres.append((int(centre_y), int(centre_x)))

                region_area = np.sum(region_mask)
                region_intensity = np.sum(smoothed_image[region_mask])
                region_max_intensity = np.max(smoothed_image[region_mask])

                region_stats.append(
                    {
                        "centre": (centre_y, centre_x),
                        "area_pixels": region_area,
                        "total_intensity": region_intensity,
                        "max_intensity": region_max_intensity,
                        "mean_intensity": (
                            region_intensity / region_area if region_area > 0 else 0
                        ),
                    }
                )

        if create_plot:
            if self.plotter is not None:
                self.plotter.create_separate_plots(
                    smoothed_image,
                    binary_mask,
                    region_centres,
                    hist,
                    bin_edges,
                    threshold,
                    pixelsize,
                    output_figure_path,
                    title,
                )
            else:
                logger.warning("⚠️ DriftPlotter not available, skipping density detection plots")

        metadata = {
            "n_regions_detected": n_regions,
            "threshold_value": threshold,
            "threshold_percentile": threshold_percentile,
            "histogram_bins": histogram_bins,
            "image_shape": smoothed_image.shape,
            "image_max": np.max(smoothed_image),
            "image_mean": np.mean(smoothed_image[smoothed_image > 0]),
            "region_statistics": region_stats,
            "total_region_area": np.sum(binary_mask),
            "region_area_fraction": np.sum(binary_mask) / binary_mask.size,
        }

        return region_centres, binary_mask, threshold, metadata

    def select_puncta_from_regions(
        self,
        locs: np.recarray,
        region_centres: List[Tuple[int, int]],
        binary_mask: np.ndarray,
        pixelsize: float = 100.0,
        selection_box_size_nm: float = 600.0,
        min_localisations_per_region: int = 10,
        output_figure_path: Optional[str] = None,
        title: str = "Puncta Selection from Regions",
        create_plot: bool = True,
        plot_individual_regions: bool = True,
        use_datashader_threshold: int = 1000,
        memory_optimize: bool = True,
    ) -> Tuple[List[np.recarray], Dict[str, Any]]:
        """Select puncta (localisations) from detected high-density regions."""
        if postprocess is None:
            raise RuntimeError(
                "postprocess module not available - cannot use picked_locs function"
            )

        if not region_centres:
            metadata = {
                "n_regions_input": 0,
                "n_regions_selected": 0,
                "selection_criteria": {
                    "min_localisations": min_localisations_per_region,
                    "selection_box_size_nm": selection_box_size_nm,
                    "selection_box_size_pixels": 0.0,
                },
                "rejection_reasons": {"too_few_localisations": 0, "accepted": 0},
                "region_statistics": [],
            }
            return [], metadata

        box_size_pixels = selection_box_size_nm / pixelsize
        half_box = box_size_pixels / 2.0

        picks = []
        for centre_y, centre_x in region_centres:
            picks.append(
                ((centre_x - half_box, centre_y), (centre_x + half_box, centre_y))
            )

        width = max(locs.xc.max() + 10, 100)
        height = max(locs.yc.max() + 10, 100)

        picked_locs_arrays = postprocess.picked_locs(
            locs=locs,
            width=width,
            height=height,
            picks=picks,
            pick_shape="Rectangle",
            pick_size=box_size_pixels,
            add_group=False,
            callback="console",
            parallel=len(picks) >= 8,
        )

        if memory_optimize:
            del picks
            gc.collect()

        selected_puncta = []
        region_stats = []

        if picked_locs_arrays is None:
            picked_locs_arrays = []

        rejected_count = 0
        for region_id, (region_locs, (centre_y, centre_x)) in enumerate(
            zip(picked_locs_arrays, region_centres)
        ):
            n_locs = len(region_locs)

            if n_locs >= min_localisations_per_region:
                selected_puncta.append(region_locs)

                region_stat = {
                    "region_id": region_id,
                    "centre_y": centre_y,
                    "centre_x": centre_x,
                    "n_localisations": n_locs,
                    "mean_x": np.mean(region_locs.xc),
                    "mean_y": np.mean(region_locs.yc),
                    "std_x": np.std(region_locs.xc),
                    "std_y": np.std(region_locs.yc),
                    "frame_range": [
                        int(region_locs.frame.min()),
                        int(region_locs.frame.max()),
                    ],
                    "frame_span": int(
                        region_locs.frame.max() - region_locs.frame.min() + 1
                    ),
                    "selection_box_size_nm": selection_box_size_nm,
                    "selection_box_size_pixels": box_size_pixels,
                    "box_boundaries": {
                        "x_min": centre_x - half_box,
                        "x_max": centre_x + half_box,
                        "y_min": centre_y - half_box,
                        "y_max": centre_y + half_box,
                    },
                }

                if hasattr(region_locs, "photons"):
                    region_stat["mean_photons"] = np.mean(region_locs.photons)
                    region_stat["std_photons"] = np.std(region_locs.photons)

                region_stats.append(region_stat)
            else:
                rejected_count += 1
                if memory_optimize:
                    del region_locs

            if memory_optimize and region_id % 100 == 0 and region_id > 0:
                gc.collect()
                logger.info(f"Processed {region_id + 1}/{len(picked_locs_arrays)} regions " f"({len(selected_puncta)} accepted, {rejected_count} rejected)")

        if memory_optimize:
            del picked_locs_arrays
            gc.collect()
            logger.info(f"Memory optimisation: Freed intermediate arrays after region processing")

        if create_plot:
            if self.plotter is not None:
                self.plotter.plot_puncta_selection_results(
                    locs,
                    selected_puncta,
                    region_centres,
                    binary_mask,
                    region_stats,
                    box_size_pixels,
                    pixelsize,
                    output_figure_path,
                    title,
                    plot_individual_regions,
                    use_datashader_threshold,
                )
            else:
                logger.warning("⚠️ DriftPlotter not available, skipping puncta selection plots")

            if memory_optimize:
                plt.close("all")
                gc.collect()

        total_locs_selected = sum(len(puncta) for puncta in selected_puncta)

        metadata = {
            "n_regions_input": len(region_centres),
            "n_regions_selected": len(selected_puncta),
            "n_regions_rejected": rejected_count,
            "selection_rate": (
                len(selected_puncta) / len(region_centres) if region_centres else 0
            ),
            "selection_criteria": {
                "min_localisations": min_localisations_per_region,
                "selection_box_size_nm": selection_box_size_nm,
                "selection_box_size_pixels": box_size_pixels,
            },
            "region_statistics": region_stats,
            "total_selected_localisations": total_locs_selected,
            "memory_optimized": memory_optimize,
            "rejection_reasons": {
                "too_few_localisations": rejected_count,
                "accepted": len(selected_puncta),
            },
        }

        return selected_puncta, metadata

    def identify_real_fiducials_with_clustering(
        self,
        selected_puncta: List[np.recarray],
        retention_percentage: float = 0.9,
        min_samples_factor: float = 0.7,
        frame_count: int = 100000,
        pixelsize: float = None,  # nm; None → self.pixel_size * 1000
        output_figure_path: Optional[str] = None,
        title: str = "Fiducial Gaussian Fitting Analysis",
        create_plot: bool = True,
    ) -> Tuple[List[np.recarray], Dict[str, Any]]:
        """Identify real fiducials from selected puncta using single Gaussian distribution fitting."""
        if pixelsize is None:
            pixelsize = self.pixel_size * 1000  # µm → nm

        validated_fiducials = []
        clustering_metadata = []

        if retention_percentage <= 0 or retention_percentage >= 1:
            raise ValueError("retention_percentage must be between 0 and 1")

        radial_threshold_factor = np.sqrt(-2 * np.log(1 - retention_percentage))
        logger.info(f"Using radial threshold factor: {radial_threshold_factor:.3f} for {retention_percentage*100:.1f}% retention")

        for region_id, puncta_locs in enumerate(selected_puncta):
            n_locs = len(puncta_locs)

            if n_locs < 10:
                continue

            X = np.vstack([puncta_locs["xc"], puncta_locs["yc"]]).T

            min_samples = max(
                int(min_samples_factor * frame_count / 1000), 5
            )

            if n_locs < min_samples:
                logger.info(f"  Region {region_id}: Too few points ({n_locs}) < min_samples ({min_samples}), skipping")
                continue

            try:
                from sklearn.mixture import GaussianMixture

                logger.info(f"  Fitting single Gaussian to {n_locs} points in region {region_id}")

                gm = GaussianMixture(n_components=1, random_state=0)
                gm.fit(X)

                mean = gm.means_[0]
                covariance = gm.covariances_[0]

                eigenvals = np.linalg.eigvals(covariance)
                sigma_pixels = np.sqrt(np.mean(eigenvals))
                sigma_nm = sigma_pixels * pixelsize

                dx = X[:, 0] - mean[0]
                dy = X[:, 1] - mean[1]
                radial_distances_pixels = np.sqrt(dx**2 + dy**2)
                radial_distances_nm = radial_distances_pixels * pixelsize

                r_threshold_pixels = sigma_pixels * radial_threshold_factor
                r_threshold_nm = r_threshold_pixels * pixelsize

                kept_mask = radial_distances_pixels <= r_threshold_pixels
                n_kept = np.sum(kept_mask)

                if n_kept >= min_samples:
                    validated_locs = puncta_locs[kept_mask]
                    validated_fiducials.append(validated_locs)

                    gaussian_metadata = {
                        "region_id": region_id,
                        "original_n_locs": n_locs,
                        "validated_n_locs": n_kept,
                        "retention_rate": n_kept / n_locs,
                        "gaussian_centre_x": mean[0],
                        "gaussian_centre_y": mean[1],
                        "gaussian_centre_x_nm": mean[0] * pixelsize,
                        "gaussian_centre_y_nm": mean[1] * pixelsize,
                        "gaussian_sigma_pixels": sigma_pixels,
                        "gaussian_sigma_nm": sigma_nm,
                        "radial_threshold_pixels": r_threshold_pixels,
                        "radial_threshold_nm": r_threshold_nm,
                        "fitting_method": "Single Gaussian",
                        "retention_percentage": retention_percentage,
                        "min_samples_factor": min_samples_factor,
                        "min_samples_used": min_samples,
                        "pixelsize": pixelsize,
                        "kept_mask": kept_mask,
                        "radial_distances_pixels": radial_distances_pixels,
                        "radial_distances_nm": radial_distances_nm,
                    }
                    clustering_metadata.append(gaussian_metadata)

                    if create_plot:
                        self._plot_single_gaussian_validation(
                            puncta_locs,
                            validated_locs,
                            kept_mask,
                            radial_distances_pixels,
                            region_id,
                            gaussian_metadata,
                            output_figure_path,
                            title,
                            r_threshold_pixels,
                        )

                    del validated_locs
                else:
                    logger.info(f"  Region {region_id}: Not enough kept points ({n_kept}) < min_samples ({min_samples}), discarding")

                del X, kept_mask, radial_distances_pixels, radial_distances_nm
                gc.collect()

                if n_locs > 10000:
                    gc.collect()

            except Exception as e:
                logger.warning(f"Warning: Gaussian fitting failed for region {region_id}: {e}")
                continue

        if create_plot and len(validated_fiducials) > 0:
            if self.plotter is not None:
                self.plotter.plot_clustering_summary_only(
                    selected_puncta,
                    validated_fiducials,
                    clustering_metadata,
                    output_figure_path,
                    title,
                )
            else:
                logger.warning("⚠️ DriftPlotter not available, skipping clustering summary plots")

        summary_metadata = {
            "n_input_regions": len(selected_puncta),
            "n_validated_fiducials": len(validated_fiducials),
            "validation_rate": (
                len(validated_fiducials) / len(selected_puncta)
                if selected_puncta
                else 0
            ),
            "fitting_parameters": {
                "retention_percentage": retention_percentage,
                "min_samples_factor": min_samples_factor,
                "frame_count": frame_count,
                "radial_threshold_factor": radial_threshold_factor,
            },
            "region_details": clustering_metadata,
            "total_input_locs": sum(len(puncta) for puncta in selected_puncta),
            "total_validated_locs": sum(
                len(fiducial) for fiducial in validated_fiducials
            ),
        }

        return validated_fiducials, summary_metadata

    def _plot_single_gaussian_validation(
        self,
        original_puncta: np.recarray,
        validated_locs: np.recarray,
        kept_mask: np.ndarray,
        radial_distances: np.ndarray,
        region_id: int,
        metadata: Dict[str, Any],
        output_figure_path: Optional[str],
        title: str,
        r_threshold: float,
        display: bool = True,
    ) -> None:
        """Plot individual Gaussian validation results showing kept vs discarded points."""
        try:
            from PlottingBase import PublicationPlotter
            import matplotlib.pyplot as plt
            import numpy as np

            plotter = PublicationPlotter(poster=False)
        except ImportError:
            logger.warning("PlottingBase not available, skipping Gaussian plot")
            return

        fig, ax = plotter.one_column_plot(width=3.5, height=3.5)

        fig.suptitle(
            f"{title} - Region {region_id+1} Gaussian Validation",
            fontsize=9,
        )

        data_arrays = []
        colors = []
        labels = []

        discarded_mask = ~kept_mask
        if np.any(discarded_mask):
            data_arrays.append(original_puncta[discarded_mask])
            colors.append("grey")
            labels.append(f"Discarded ({np.sum(discarded_mask):,})")

        if np.any(kept_mask):
            data_arrays.append(original_puncta[kept_mask])
            colors.append("red")
            labels.append(f"Kept ({np.sum(kept_mask):,})")

        if data_arrays:
            if self.plotter is not None:
                self.plotter.plot_region_data_with_datashader(
                    ax, data_arrays, colors, labels
                )
            else:
                for i, data in enumerate(data_arrays):
                    color = colors[i % len(colors)] if colors else "blue"
                    ax.plot(
                        data["xc"],
                        data["yc"],
                        ".",
                        color=color,
                        markersize=2,
                        alpha=0.6,
                    )

            from matplotlib.patches import Patch

            legend_elements = [
                Patch(facecolor=color, label=label)
                for color, label in zip(colors, labels)
            ]
            ax.legend(handles=legend_elements, loc="upper right")

        ax.set_xlabel("X Position (pixels)")
        ax.set_ylabel("Y Position (pixels)")
        ax.set_title(f"Gaussian Fitting - Region {region_id+1}")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

        stats_text = f"Stats:\n"
        stats_text += f"Original: {metadata['original_n_locs']:,}\n"
        stats_text += f"Kept: {metadata['validated_n_locs']:,}\n"
        stats_text += f"Retention: {100*metadata['retention_rate']:.1f}%\n"
        stats_text += f"Gaussian σ: {metadata['gaussian_sigma_nm']:.1f} nm\n"
        stats_text += f"Threshold: {metadata['radial_threshold_nm']:.1f} nm"

        retention_rate = metadata["retention_rate"]
        if 0.8 <= retention_rate <= 0.95:
            quality_color = "lightgreen"
        elif 0.7 <= retention_rate < 0.8 or 0.95 < retention_rate <= 1.0:
            quality_color = "lightyellow"
        else:
            quality_color = "lightcoral"

        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=6,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=quality_color, alpha=0.7),
        )

        if output_figure_path:
            base_path = (
                output_figure_path.rsplit(".", 1)[0]
                if "." in output_figure_path
                else output_figure_path
            )
            gaussian_filename = f"{base_path}_gaussian_region_{region_id+1:02d}.png"
            fig.savefig(gaussian_filename, dpi=300, bbox_inches="tight")
            logger.info(f"Saved Gaussian plot: {gaussian_filename}")

        if display:
            plt.show()
        plt.close(fig)

    def _filter_fiducials_fast(
        self,
        all_corrected_x,
        all_corrected_y,
        variance_threshold=3.0,
        rms_threshold=2.0,
    ):
        """Fast filtering of fiducial traces using variance ratio and RMS distance."""
        n_frames, n_fiducials = all_corrected_x.shape
        valid_fiducials = np.ones(n_fiducials, dtype=bool)

        logger.debug("Filtering by variance ratio...")

        x_variances = np.nanvar(all_corrected_x, axis=0)
        y_variances = np.nanvar(all_corrected_y, axis=0)
        combined_variances = x_variances + y_variances

        finite_variances = combined_variances[~np.isnan(combined_variances)]
        if len(finite_variances) == 0:
            logger.info("No valid variances found")
            return np.zeros(n_fiducials, dtype=bool), {}

        median_variance = np.median(finite_variances)
        threshold_variance = variance_threshold * median_variance

        variance_mask = combined_variances <= threshold_variance
        n_removed_variance = np.sum(~variance_mask)
        valid_fiducials &= variance_mask

        logger.debug(f"\rRemoved {n_removed_variance} high-variance fiducials.    ")
        logger.debug("\rFiltering by RMS distance...")

        rms_distances = np.sqrt(
            np.nanmean(all_corrected_x**2 + all_corrected_y**2, axis=0)
        )

        finite_rms = rms_distances[~np.isnan(rms_distances)]
        if len(finite_rms) == 0:
            logger.info("No valid RMS distances found")
            return valid_fiducials, {
                "n_variance_filtered": n_removed_variance,
                "n_rms_filtered": 0,
                "median_variance": median_variance,
                "median_rms": np.nan,
            }

        median_rms = np.median(finite_rms)
        threshold_rms = rms_threshold * median_rms

        rms_mask = rms_distances <= threshold_rms
        n_removed_rms = np.sum(~rms_mask)
        valid_fiducials &= rms_mask

        logger.debug(f"\rRemoved {n_removed_rms} high-RMS fiducials.    ")

        n_total_removed = n_removed_variance + n_removed_rms
        n_final = np.sum(valid_fiducials)
        logger.info(f"\rFinal: {n_final}/{n_fiducials} fiducials retained ({n_total_removed} removed)    ")

        return valid_fiducials, {
            "n_variance_filtered": n_removed_variance,
            "n_rms_filtered": n_removed_rms,
            "median_variance": median_variance,
            "median_rms": median_rms,
            "variance_threshold_used": variance_threshold * median_variance,
            "rms_threshold_used": rms_threshold * median_rms,
        }

    def apply_validated_fiducial_drift_correction(
        self,
        locs: np.recarray,
        validated_fiducials: List[np.recarray],
        x_err_field: str = "xc_err",
        y_err_field: str = "yc_err",
    ) -> Tuple[np.recarray, Dict[str, np.ndarray]]:
        """Apply drift correction using validated fiducials."""
        if not validated_fiducials:
            raise ValueError("No validated fiducials provided")

        sample_fiducial = validated_fiducials[0]
        has_x_err = x_err_field in sample_fiducial.dtype.names
        has_y_err = y_err_field in sample_fiducial.dtype.names

        min_frame = int(locs.frame.min())
        max_frame = int(locs.frame.max())

        if not has_x_err or not has_y_err:
            logger.warning(f"Warning: Error fields '{x_err_field}' or '{y_err_field}' not found. Using uniform weights.")
            has_x_err = has_y_err = False

        unique_frames = np.unique(locs.frame)
        frame_to_idx = {frame: i for i, frame in enumerate(unique_frames)}

        all_corrected_x = np.full(
            [len(unique_frames), len(validated_fiducials)], np.nan
        )
        all_corrected_y = np.full(
            [len(unique_frames), len(validated_fiducials)], np.nan
        )
        all_fiducial_weights_x = np.full(
            [len(unique_frames), len(validated_fiducials)], np.nan
        )
        all_fiducial_weights_y = np.full(
            [len(unique_frames), len(validated_fiducials)], np.nan
        )

        for i, fiducial_cluster in enumerate(validated_fiducials):
            if len(fiducial_cluster) == 0:
                continue

            median_x = np.median(fiducial_cluster.xc)
            median_y = np.median(fiducial_cluster.yc)

            corrected_x = fiducial_cluster.xc - median_x
            corrected_y = fiducial_cluster.yc - median_y
            frames = np.asarray(fiducial_cluster.frame, dtype=np.int_)

            if len(frames) == len(np.unique(frames)):
                frame_indices = [frame_to_idx[frame] for frame in frames]

                all_corrected_x[frame_indices, i] = corrected_x
                all_corrected_y[frame_indices, i] = corrected_y

                if has_x_err and has_y_err:
                    all_fiducial_weights_x[frame_indices, i] = 1.0 / (
                        1e-10 + fiducial_cluster[x_err_field]
                    )
                    all_fiducial_weights_y[frame_indices, i] = 1.0 / (
                        1e-10 + fiducial_cluster[y_err_field]
                    )
                else:
                    all_fiducial_weights_x[frame_indices, i] = 1.0
                    all_fiducial_weights_y[frame_indices, i] = 1.0
            else:
                logger.debug(f"\rWarning: Fiducial cluster {i} has multiple localisations in the same frame. Skipping this cluster.    ")
                continue

        if np.all(np.isnan(all_corrected_x)):
            raise ValueError("No valid fiducials found after median subtraction")

        valid_fiducials, _ = self._filter_fiducials_fast(
            all_corrected_x, all_corrected_y
        )

        all_corrected_x = all_corrected_x[:, valid_fiducials]
        all_corrected_y = all_corrected_y[:, valid_fiducials]
        all_fiducial_weights_x = all_fiducial_weights_x[:, valid_fiducials]
        all_fiducial_weights_y = all_fiducial_weights_y[:, valid_fiducials]

        ma_x = np.ma.MaskedArray(all_corrected_x, mask=np.isnan(all_corrected_x))
        ma_y = np.ma.MaskedArray(all_corrected_y, mask=np.isnan(all_corrected_y))
        ma_x_err = np.ma.MaskedArray(
            all_fiducial_weights_x, mask=np.isnan(all_fiducial_weights_x)
        )
        ma_y_err = np.ma.MaskedArray(
            all_fiducial_weights_y, mask=np.isnan(all_fiducial_weights_y)
        )

        drift_x = np.ma.average(ma_x, weights=ma_x_err, axis=1)
        drift_y = np.ma.average(ma_y, weights=ma_y_err, axis=1)

        if isinstance(drift_x, np.ma.MaskedArray):
            mask_x = np.ma.getmaskarray(drift_x)
        else:
            mask_x = np.zeros(len(drift_x), dtype=bool)

        if isinstance(drift_y, np.ma.MaskedArray):
            mask_y = np.ma.getmaskarray(drift_y)
        else:
            mask_y = np.zeros(len(drift_y), dtype=bool)

        valid_frame_mask = np.logical_not(mask_x | mask_y)

        logger.info(f"\nDEBUG: Drift calculation results:")
        logger.info(f"  - Total unique frames in locs: {len(unique_frames)}")
        logger.info(f"  - drift_x type: {type(drift_x)}, shape: {np.shape(drift_x)}")
        logger.info(f"  - drift_y type: {type(drift_y)}, shape: {np.shape(drift_y)}")
        logger.info(f"  - mask_x sum (masked frames): {np.sum(mask_x)}")
        logger.info(f"  - mask_y sum (masked frames): {np.sum(mask_y)}")
        logger.info(f"  - valid_frame_mask sum: {np.sum(valid_frame_mask)}")
        if len(drift_x) > 0:
            logger.info(f"  - drift_x first 5 values: {drift_x[:5]}")
            logger.info(f"  - drift_y first 5 values: {drift_y[:5]}")
        if np.sum(valid_frame_mask) == 0:
            logger.info(f"  ⚠️ WARNING: No valid frames! All drift values are masked/NaN")
            logger.info(f"  - Checking all_corrected arrays:")
            logger.info(f"    - all_corrected_x shape: {all_corrected_x.shape}")
            logger.info(f"    - all_corrected_y shape: {all_corrected_y.shape}")
            logger.info(f"    - Non-NaN entries in all_corrected_x: {np.sum(~np.isnan(all_corrected_x))}")
            logger.info(f"    - Non-NaN entries in all_corrected_y: {np.sum(~np.isnan(all_corrected_y))}")

        valid_frame_numbers = unique_frames[valid_frame_mask]
        valid_drift_x = np.asarray(drift_x[valid_frame_mask])
        valid_drift_y = np.asarray(drift_y[valid_frame_mask])

        if np.isscalar(valid_frame_numbers):
            valid_frame_numbers = np.array([valid_frame_numbers])
        if np.isscalar(valid_drift_x):
            valid_drift_x = np.array([valid_drift_x])
        if np.isscalar(valid_drift_y):
            valid_drift_y = np.array([valid_drift_y])

        frame_mask = np.isin(locs.frame, valid_frame_numbers)
        corrected_locs = locs[frame_mask].copy()

        frame_nums = (
            valid_frame_numbers.flatten()
            if valid_frame_numbers.ndim > 1
            else valid_frame_numbers
        )
        drift_x_vals = (
            valid_drift_x.flatten() if valid_drift_x.ndim > 1 else valid_drift_x
        )
        drift_y_vals = (
            valid_drift_y.flatten() if valid_drift_y.ndim > 1 else valid_drift_y
        )

        drift_lookup_x = dict(zip(frame_nums, drift_x_vals))
        drift_lookup_y = dict(zip(frame_nums, drift_y_vals))

        fiducial_positions = set()
        for fiducial_cluster in validated_fiducials:
            for fiducial in fiducial_cluster:
                fiducial_positions.add((fiducial.xc, fiducial.yc, fiducial.frame))

        is_fiducial_flags = np.zeros(len(corrected_locs), dtype=bool)
        for i in range(len(corrected_locs)):
            frame = corrected_locs[i].frame

            original_pos = (corrected_locs[i].xc, corrected_locs[i].yc, frame)
            is_fiducial_flags[i] = original_pos in fiducial_positions

            corrected_locs[i].xc -= drift_lookup_x[frame]
            corrected_locs[i].yc -= drift_lookup_y[frame]

        from numpy.lib import recfunctions as rfn

        final_corrected_locs = rfn.append_fields(
            corrected_locs,
            "is_fiducial",
            is_fiducial_flags,
            dtypes=bool,
            asrecarray=True,
            usemask=False,
        )

        drift_info = {
            "frames": valid_frame_numbers,
            "drift_x": valid_drift_x,
            "drift_y": valid_drift_y,
            "n_fiducials_per_frame": np.sum(~np.isnan(all_corrected_x), axis=1)[
                valid_frame_mask
            ],
        }

        logger.info(f"Drift correction applied to {len(final_corrected_locs)} localisations")
        logger.info(f"Used {len(valid_frame_numbers)} frames with fiducials (out of {max_frame - min_frame + 1} total frames)")
        logger.info(f"Average {np.mean(drift_info['n_fiducials_per_frame']):.1f} fiducials per frame")

        return final_corrected_locs, drift_info
