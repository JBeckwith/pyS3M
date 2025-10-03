#!/usr/bin/env python3
"""
Interactive Threshold Tuner for Nile Red Analysis - 20250930
Bacteria with Nile Red experiment

This script helps determine optimal spot detection parameters (pfa, sigma, fraction_true,
temporal_median_window) for spot detection in Nile Red stained bacteria datasets. It
interactively loads 3 test frames (1, 10, 20) from each folder in the Nile Red analysis
workflow, displays detection results in a column layout, and saves the optimized parameters
for use by 20250930_NileRedAnalysis.sh.

Usage:
    python superres_notebooks/20250930_NileRedAnalysisTuner.py

Output:
    20250930_nile_red_threshold_parameters.txt - Parameter file for 20250930_NileRedAnalysis.sh

Author: Claude Code (Anthropic)
Date: September 30, 2025
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tifffile
from typing import Dict, List, Tuple, Optional, Union
import json

# Add src to path for imports
sys.path.append("../src")

# Global flag for display mode
INTERACTIVE_DISPLAY = False

try:
    from SpotDetectionFunctions import SpotDetection_Functions
    from IOFunctions import IO_Functions
    from CalibrationFunctions import Calibration_Functions
    from PlottingFunctions import Plotter
    from HelperFunctions import Helper_Functions
    from sCMOSFunctions import sCMOS_Functions
    from SR_Functions import SuperRes_Functions, TemporalMedianMode
    from ImageAnalysisFunctions import Image_Analysis_Functions
    from MaskFunctions import Mask_Functions
    import matplotlib

    # Try to use interactive backend, fall back gracefully
    try:
        import tkinter

        matplotlib.use("TkAgg")
        INTERACTIVE_DISPLAY = True
        print("✓ Interactive matplotlib backend enabled")
    except ImportError:
        print("⚠ tkinter not available - using file-based display mode")
        print("  Images will be saved as PNG files for viewing")
        print("  To enable interactive display:")
        print("  sudo apt-get install python3-tk")
        matplotlib.use("Agg")
        INTERACTIVE_DISPLAY = False

    import matplotlib.pyplot as plt

except ImportError as e:
    print(f"Error importing pyBayerSMLM modules: {e}")
    print("Please ensure the virtual environment is activated:")
    print("source /home/jbeckwith/.virtualenvs/pyBayerSMLM/bin/activate")
    sys.exit(1)


class NileRedThresholdTuner:
    """Interactive tool for determining optimal spot detection thresholds for Nile Red bacteria analysis"""

    def __init__(self):
        self.sdf = SpotDetection_Functions()
        self.iof = IO_Functions()
        self.cf = Calibration_Functions()
        self.pf = Plotter()
        self.hf = Helper_Functions()
        self.scmos = sCMOS_Functions()
        self.srf = SuperRes_Functions()
        self.iaf = Image_Analysis_Functions()
        self.mf = Mask_Functions()

        # Default parameters optimized for Nile Red staining
        self.default_pfa = 1e-3
        self.default_sigma = 1.5
        self.default_true_fraction = 0.1
        self.default_wavelength = 0.650  # 650nm
        self.default_use_variance_aware = True  # Default to variance-aware demosaicing
        self.default_temporal_median_mode = TemporalMedianMode.FITTING_ONLY  # Default: fitting only
        self.default_temporal_median_window = 500  # Default window size in frames

        # Results storage
        self.threshold_results = {}

        # Load camera calibration data
        self.camera_data = self._load_camera_data()

        # ROI information (will be set per folder)
        self.roi_info = None

        # Folder lists for Nile Red analysis
        self.folder_lists = self._get_nile_red_folder_lists()

    def _load_camera_data(self) -> Dict:
        """Load camera calibration data"""
        try:
            # Go up from superres_notebooks to project root, then to Camera_Calibrations
            project_root = Path(__file__).parent.parent
            data_folder = project_root / "Camera_Calibrations" / "Ximea_Camera"

            camera_data = {
                "gain": self.iof.read_tiff(str(data_folder / "gain.tif")),
                "offset": self.iof.read_tiff(str(data_folder / "offset.tif")),
                "variance": self.iof.read_tiff(str(data_folder / "variance.tif")),
                "readnoise": self.iof.read_tiff(str(data_folder / "readnoise.tif")),
                "rqe": self.iof.read_tiff(str(data_folder / "rqe.tif")),
            }
            print("✓ Camera calibration data loaded successfully")
            return camera_data
        except Exception as e:
            print(f"⚠ Warning: Could not load camera calibration data: {e}")
            print("  Variance-aware demosaicing will not be available")
            return {}

    def _get_nile_red_folder_lists(self) -> Dict[str, List[str]]:
        """Get folder lists for Nile Red analysis experiments"""
        return {
            "NILE_RED_FOLDERS": [
                "/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/JSB/20250930_BacteriaNR4ASina",
            ]
        }

    def find_leaf_directories(self, base_dir: str) -> List[str]:
        """Find leaf directories (directories with no subdirectories) in a hierarchical structure"""
        leaf_dirs = []

        if not os.path.isdir(base_dir):
            print(f"Warning: Directory not found: {base_dir}")
            return []

        for root, dirs, files in os.walk(base_dir):
            # If no subdirectories, it's a leaf
            if not dirs:
                leaf_dirs.append(root)

        return leaf_dirs

    def _get_roi_info(self, folder_path: str) -> Optional[Dict]:
        """Extract ROI information from metadata files if available"""
        try:
            metadata_files = self.hf.file_search(folder_path, "metadata", "")
            if metadata_files:
                start_x, start_y, width, height = self.iof.metadata_reader_imageJ(metadata_files[0])
                return {
                    "start_x": start_x,
                    "start_y": start_y,
                    "width": width,
                    "height": height
                }
        except Exception as e:
            print(f"Could not extract ROI info from {folder_path}: {e}")
        return None

    def _crop_camera_data_to_roi(self, camera_data: Dict, roi_info: Dict) -> Dict:
        """Crop camera calibration data to match ROI dimensions"""
        if not roi_info:
            return camera_data

        cropped_data = {}
        start_x, start_y = roi_info["start_x"], roi_info["start_y"]
        width, height = roi_info["width"], roi_info["height"]

        for key, data in camera_data.items():
            if isinstance(data, np.ndarray) and data.ndim >= 2:
                cropped_data[key] = data[start_x:start_x + width, start_y:start_y + height]
            else:
                cropped_data[key] = data

        return cropped_data

    def get_all_processing_folders(self) -> List[Tuple[str, str, float]]:
        """Get all folders that will be processed by 20250919_NileRedAnalysis.sh with their parameters"""
        all_folders = []

        # Nile Red folders (direct)
        for folder in self.folder_lists["NILE_RED_FOLDERS"]:
            if os.path.isdir(folder):
                # Check if this is a leaf directory or has subdirectories
                leaf_dirs = self.find_leaf_directories(folder)
                if leaf_dirs:
                    # Has subdirectories, process leaf directories
                    for leaf_dir in leaf_dirs:
                        all_folders.append((leaf_dir, "imaging", 0.700))
                else:
                    # No subdirectories, process this folder directly
                    all_folders.append((folder, "imaging", 0.700))

        return all_folders

    def find_tiff_files(self, folder_path: str) -> List[str]:
        """Find all TIFF files in a folder using helper function for proper sorting"""
        try:
            # Use helper function to get sorted TIFF files
            tiff_files = self.hf.file_search(folder_path, ".tif", "")
            if not tiff_files:
                # Try alternative extension
                tiff_files = self.hf.file_search(folder_path, ".tiff", "")
            return tiff_files if tiff_files else []
        except Exception as e:
            print(f"Error finding TIFF files in {folder_path}: {e}")
            return []

    def load_frame_stacks_for_temporal_median(
        self, folder_path: str, temporal_median_window: int
    ) -> Optional[Dict[str, any]]:
        """Load frame stacks for temporal median preview.

        Returns a dict containing:
        - 'stacks': List of 3 frame stacks (each stack has temporal_median_window frames)
        - 'test_frame_indices': List of 3 indices within each stack for the test frame
        - 'display_frames': List of 3 frames to display (the middle frame of each stack)
        """
        tiff_files = self.find_tiff_files(folder_path)

        if not tiff_files:
            print(f"No TIFF files found in {folder_path}")
            return None

        # Check for metadata files to get ROI information
        self.roi_info = self._get_roi_info(folder_path)

        # Determine how many frames to load for each stack
        # Use the temporal median window (don't cap - user controls this)
        stack_size = temporal_median_window
        half_window = stack_size // 2

        print(f"Temporal median window: {temporal_median_window} frames (±{half_window} from center)")

        try:
            result = {
                'stacks': [],
                'test_frame_indices': [],
                'display_frames': []
            }

            if len(tiff_files) == 1:
                # Single file: load 3 stacks centered at frames 50, 200, 400
                print(f"Single TIFF file detected: {os.path.basename(tiff_files[0])}")
                with tifffile.TiffFile(tiff_files[0]) as tif:
                    total_frames = len(tif.pages)
                    print(f"Total frames available: {total_frames}")
                    print(f"Loading stacks of {stack_size} frames for temporal median preview")

                    # Target center frames for the stacks
                    target_centers = [50, 200, 400]

                    for center_idx in target_centers:
                        # Adjust if we're near the end of the file
                        center_idx = min(center_idx, total_frames - 1)

                        # Calculate stack boundaries
                        start_idx = max(0, center_idx - half_window)
                        end_idx = min(total_frames, center_idx + half_window)

                        # Load the stack
                        stack_frames = []
                        for idx in range(start_idx, end_idx):
                            frame = tif.pages[idx].asarray().astype(np.float64)
                            stack_frames.append(frame)

                        if len(stack_frames) > 0:
                            # Index of the display frame within this stack
                            display_idx = center_idx - start_idx
                            display_idx = min(display_idx, len(stack_frames) - 1)

                            result['stacks'].append(np.array(stack_frames))
                            result['test_frame_indices'].append(display_idx)
                            result['display_frames'].append(stack_frames[display_idx])

                            print(f"  Loaded stack: frames {start_idx+1}-{end_idx} ({len(stack_frames)} frames), display frame at index {display_idx}")

            else:
                # Multiple files: load stack from 10% position in first 3 files
                print(f"Multiple TIFF files detected: {len(tiff_files)} files")
                files_to_process = tiff_files[:3]

                for i, tiff_file in enumerate(files_to_process):
                    print(f"Processing file {i+1}: {os.path.basename(tiff_file)}")
                    with tifffile.TiffFile(tiff_file) as tif:
                        total_frames = len(tif.pages)

                        # Center at 10% of stack
                        center_idx = int(total_frames * 0.1)

                        # Calculate stack boundaries
                        start_idx = max(0, center_idx - half_window)
                        end_idx = min(total_frames, center_idx + half_window)

                        # Load the stack
                        stack_frames = []
                        for idx in range(start_idx, end_idx):
                            frame = tif.pages[idx].asarray().astype(np.float64)
                            stack_frames.append(frame)

                        if len(stack_frames) > 0:
                            # Index of the display frame within this stack
                            display_idx = center_idx - start_idx
                            display_idx = min(display_idx, len(stack_frames) - 1)

                            result['stacks'].append(np.array(stack_frames))
                            result['test_frame_indices'].append(display_idx)
                            result['display_frames'].append(stack_frames[display_idx])

                            print(f"  Loaded stack: frames {start_idx+1}-{end_idx} ({len(stack_frames)} frames), display frame at index {display_idx}")

            if len(result['stacks']) == 0:
                print("No frame stacks could be loaded")
                return None

            print(f"Successfully loaded {len(result['stacks'])} frame stacks for temporal median preview")
            return result

        except Exception as e:
            print(f"Error loading frame stacks: {e}")
            import traceback
            traceback.print_exc()
            return None

    def load_test_frames(self, folder_path: str) -> Optional[List[np.ndarray]]:
        """Load 3 test frames with improved logic for single vs multiple files (legacy mode without temporal median)"""
        tiff_files = self.find_tiff_files(folder_path)

        if not tiff_files:
            print(f"No TIFF files found in {folder_path}")
            return None

        # Check for metadata files to get ROI information
        self.roi_info = self._get_roi_info(folder_path)

        try:
            selected_frames = []

            if len(tiff_files) == 1:
                # Single file: use frames 1, 10, 20 as before
                print(f"Single TIFF file detected: {os.path.basename(tiff_files[0])}")
                with tifffile.TiffFile(tiff_files[0]) as tif:
                    total_frames = len(tif.pages)
                    print(f"Total frames available: {total_frames}")

                    # Select frame indices: 1, 10, 20 (0-based: 0, 9, 19)
                    target_frames = [0, 9, 19]  # 0-based indexing
                    actual_indices = []

                    for i, target_idx in enumerate(target_frames):
                        frame_idx = min(target_idx, total_frames - 1)
                        if frame_idx < 0:
                            frame_idx = 0

                        # Avoid duplicate indices
                        if frame_idx not in actual_indices:
                            actual_indices.append(frame_idx)
                            frame = tif.pages[frame_idx].asarray()
                            selected_frames.append(frame.astype(np.float64))
                            print(f"Loaded frame {frame_idx + 1}/{total_frames} (shape: {frame.shape})")

            else:
                # Multiple files: take frame at 10% of each file from first three files
                print(f"Multiple TIFF files detected: {len(tiff_files)} files")
                files_to_process = tiff_files[:3]  # Take first 3 files

                for i, tiff_file in enumerate(files_to_process):
                    print(f"Processing file {i+1}: {os.path.basename(tiff_file)}")
                    with tifffile.TiffFile(tiff_file) as tif:
                        total_frames = len(tif.pages)

                        # Get frame at 10% of the stack (ensuring it's an integer)
                        target_frame_idx = int(total_frames * 0.1)
                        # Ensure we don't go beyond available frames
                        target_frame_idx = min(target_frame_idx, total_frames - 1)
                        if target_frame_idx < 0:
                            target_frame_idx = 0

                        frame = tif.pages[target_frame_idx].asarray()
                        selected_frames.append(frame.astype(np.float64))
                        print(f"  Loaded frame {target_frame_idx + 1}/{total_frames} (10% of stack, shape: {frame.shape})")

            if len(selected_frames) == 0:
                print("No frames could be loaded")
                return None

            # If we have fewer than 3 frames, duplicate the last one
            while len(selected_frames) < 3 and selected_frames:
                selected_frames.append(selected_frames[-1].copy())
                print("Duplicated frame for display (insufficient frames available)")

            if selected_frames:
                print(f"Successfully loaded {len(selected_frames)} test frames")
                for i, frame in enumerate(selected_frames):
                    print(f"Frame {i+1}: Intensity range {frame.min():.0f} - {frame.max():.0f}")
                return selected_frames
            else:
                print("No frames could be loaded")
                return None

        except Exception as e:
            print(f"Error loading frames: {e}")
            return None

    def test_spot_detection_with_temporal_median(
        self,
        frame_stack: np.ndarray,
        test_frame_index: int,
        pfa: float,
        sigma: float,
        fraction_true: float,
        wavelength: float,
        use_variance_aware: bool = True,
        temporal_median_window: int = 100,
    ) -> Tuple[np.ndarray, int, np.ndarray]:
        """Test spot detection with temporal median subtraction.

        Args:
            frame_stack: 3D array of frames (n_frames, height, width)
            test_frame_index: Index of the frame to test within the stack
            pfa: False alarm probability
            sigma: Sigma parameter
            fraction_true: Fraction true parameter
            wavelength: Peak wavelength
            use_variance_aware: Use variance-aware demosaicing
            temporal_median_window: Window size for temporal median

        Returns:
            (detected_spots, num_spots, processed_frame): spots array, count, and the processed frame after temporal median
        """
        try:
            # Calculate temporal median
            print(f"  Computing temporal median from {len(frame_stack)} frames...")
            temporal_median = np.median(frame_stack, axis=0).astype(np.float64)

            # Subtract temporal median from test frame
            test_frame = frame_stack[test_frame_index].astype(np.float64)
            frame_after_median = test_frame - temporal_median

            # Clip negative values to zero (can't have negative photon counts)
            frame_after_median = np.maximum(frame_after_median, 0)

            print(f"  Frame before temporal median: {test_frame.min():.0f} - {test_frame.max():.0f}")
            print(f"  Temporal median: {temporal_median.min():.0f} - {temporal_median.max():.0f}")
            print(f"  Frame after subtraction: {frame_after_median.min():.0f} - {frame_after_median.max():.0f}")

            # Apply demosaicing based on setting
            if use_variance_aware and self.camera_data:
                # Get camera data cropped to ROI if available
                camera_data_to_use = (
                    self._crop_camera_data_to_roi(self.camera_data, self.roi_info)
                    if self.roi_info
                    else self.camera_data
                )

                # Check if calibration data needs to be resized to match image
                image_shape = frame_after_median.shape
                variance_shape = camera_data_to_use["variance"].shape

                if image_shape != variance_shape:
                    print(f"Warning: Image shape {image_shape} != calibration data shape {variance_shape}")
                    print("Falling back to standard demosaicing")
                    demosaiced_image = self.scmos.bayer_demosaic_stack_grayscale(frame_after_median)
                else:
                    # Use variance-aware demosaicing
                    demosaiced_image = self.scmos.variance_aware_malvar_demosaic(
                        frame_after_median,
                        variance_map=camera_data_to_use["variance"],
                        offset_map=camera_data_to_use["offset"],
                        gain=camera_data_to_use["gain"],
                        grayscale=True
                    )
            else:
                # Use standard grayscale demosaicing
                demosaiced_image = self.scmos.bayer_demosaic_stack_grayscale(frame_after_median)

            detected_spots = self.sdf.detect_puncta_in_image(
                image=demosaiced_image,
                pfa=pfa,
                wavelength=wavelength,
                sigma=sigma,
                fraction_true=fraction_true,
                pixel_size=0.069,  # Standard pixel size
                NA=1.49,  # Standard NA
                mf_factor=3.0,  # Standard match filter factor
                local_factor=3.0,  # Standard local factor
            )
            return detected_spots, len(detected_spots), frame_after_median

        except Exception as e:
            print(f"Error in spot detection with temporal median: {e}")
            import traceback
            traceback.print_exc()
            return np.array([]), 0, frame_stack[test_frame_index]

    def fit_detected_spots(
        self,
        raw_frame: np.ndarray,
        detected_spots: np.ndarray,
        smoothing_function=None,
        ROI_size: int = 16,
    ) -> np.ndarray:
        """Fit detected spots using ROI extraction and fitting.

        Args:
            raw_frame: Raw frame data (2D)
            detected_spots: Detected spot coordinates (Nx2 or Nx3 array)
            smoothing_function: Optional smoothing function
            ROI_size: Size of ROI to extract around each spot

        Returns:
            Array of fitted spot coordinates (x, y) or empty array if no fits succeed
        """
        if len(detected_spots) == 0:
            return np.array([])

        try:
            # Get camera data cropped to ROI if available
            camera_data_to_use = (
                self._crop_camera_data_to_roi(self.camera_data, self.roi_info)
                if self.roi_info and self.camera_data
                else self.camera_data
            )

            # Create masks for the ROI
            height, width = raw_frame.shape
            if self.roi_info:
                masks = self.mf.get_ROI_mask(
                    ROI_x_start=self.roi_info["start_x"],
                    ROI_y_start=self.roi_info["start_y"],
                    width=self.roi_info["width"],
                    height=self.roi_info["height"],
                    mosaic_unit=np.array([["B", "G"], ["G", "R"]])
                )
            else:
                masks = self.mf.get_ROI_mask(
                    ROI_x_start=0,
                    ROI_y_start=0,
                    width=width,
                    height=height,
                    mosaic_unit=np.array([["B", "G"], ["G", "R"]])
                )
            masks = np.dstack([masks[x] for x in masks.keys()])

            # Get calibration maps
            gain_map = camera_data_to_use.get("gain", 1.0) if camera_data_to_use else 1.0
            offset_map = camera_data_to_use.get("offset", 0.0) if camera_data_to_use else 0.0
            rqe = camera_data_to_use.get("rqe", 1.0) if camera_data_to_use else 1.0
            read_noise = camera_data_to_use.get("readnoise", 1.6) if camera_data_to_use else 1.6

            # Collect all ROIs first (for parallel fitting)
            puncta_tofit = []
            smoothed_puncta_tofit = []
            weights_tofit = []
            masks_tofit = []
            relative_coords = []

            for i in range(len(detected_spots)):
                result = self.srf._process_roi(
                    raw_frame,
                    detected_spots,
                    i,
                    width,
                    height,
                    ROI_size,
                    smoothing_function,
                    read_noise,
                    masks,
                    gain_map=gain_map,
                    offset_map=offset_map,
                    rqe=rqe,
                    frame_offset=0,
                    is_multi_frame=False,
                )

                if result is not None:
                    photoelectron_roi, smoothed_roi, weights_roi, mask_roi, coords, _ = result
                    puncta_tofit.append(photoelectron_roi)
                    smoothed_puncta_tofit.append(smoothed_roi)
                    weights_tofit.append(weights_roi)
                    masks_tofit.append(mask_roi)
                    relative_coords.append(coords)

            # Fit all ROIs in parallel (much faster than one-by-one)
            if len(puncta_tofit) > 0:
                from ImageAnalysisFunctions import FittingStrategy
                fit_results, _ = self.iaf.fit_puncta_parallel_method(
                    puncta_tofit,
                    smoothed_puncta_tofit,
                    weights_tofit,
                    relative_coords,
                    [0] * len(puncta_tofit),
                    FittingStrategy.STANDARD,
                    masks=masks_tofit,
                )

                # Extract x, y coordinates from fit results
                fitted_coords = []
                for i in range(len(fit_results)):
                    xc, yc = fit_results[i][0], fit_results[i][1]  # xc, yc are first two columns
                    fitted_coords.append([xc, yc])

                return np.array(fitted_coords)
            else:
                return np.array([])

        except Exception as e:
            print(f"Error in fitting detected spots: {e}")
            import traceback
            traceback.print_exc()
            return np.array([])

    def test_spot_detection(
        self,
        image: np.ndarray,
        pfa: float,
        sigma: float,
        fraction_true: float,
        wavelength: float,
        use_variance_aware: bool = True,
    ) -> Tuple[np.ndarray, int]:
        """Test spot detection with given parameters on demosaiced image (no temporal median)"""
        try:
            # Apply demosaicing based on setting
            if use_variance_aware and self.camera_data:
                # Get camera data cropped to ROI if available
                camera_data_to_use = (
                    self._crop_camera_data_to_roi(self.camera_data, self.roi_info)
                    if self.roi_info
                    else self.camera_data
                )

                # Check if calibration data needs to be resized to match image
                image_shape = image.shape
                variance_shape = camera_data_to_use["variance"].shape

                if image_shape != variance_shape:
                    print(f"Warning: Image shape {image_shape} != calibration data shape {variance_shape}")
                    print("Falling back to standard demosaicing")
                    demosaiced_image = self.scmos.bayer_demosaic_stack_grayscale(image)
                else:
                    # Use variance-aware demosaicing
                    demosaiced_image = self.scmos.variance_aware_malvar_demosaic(
                        image,
                        variance_map=camera_data_to_use["variance"],
                        offset_map=camera_data_to_use["offset"],
                        gain=camera_data_to_use["gain"],
                        grayscale=True
                    )
            else:
                # Use standard grayscale demosaicing
                demosaiced_image = self.scmos.bayer_demosaic_stack_grayscale(image)

            detected_spots = self.sdf.detect_puncta_in_image(
                image=demosaiced_image,
                pfa=pfa,
                wavelength=wavelength,
                sigma=sigma,
                fraction_true=fraction_true,
                pixel_size=0.069,  # Standard pixel size
                NA=1.49,  # Standard NA
                mf_factor=3.0,  # Standard match filter factor
                local_factor=3.0,  # Standard local factor
            )
            return detected_spots, len(detected_spots)

        except Exception as e:
            print(f"Error in spot detection: {e}")
            return np.array([]), 0

    def test_spot_detection_multi_frame(
        self,
        frames: List[np.ndarray],
        pfa: float,
        sigma: float,
        fraction_true: float,
        wavelength: float,
        use_variance_aware: bool = True,
    ) -> List[Tuple[np.ndarray, int]]:
        """Test spot detection on multiple frames with given parameters"""
        results = []

        for i, frame in enumerate(frames):
            try:
                # Apply demosaicing based on setting
                if use_variance_aware and self.camera_data:
                    # Get camera data cropped to ROI if available
                    camera_data_to_use = (
                        self._crop_camera_data_to_roi(self.camera_data, self.roi_info)
                        if self.roi_info
                        else self.camera_data
                    )

                    # Check if calibration data needs to be resized to match frame
                    frame_shape = frame.shape
                    variance_shape = camera_data_to_use["variance"].shape

                    if frame_shape != variance_shape:
                        print(f"Warning: Frame shape {frame_shape} != calibration data shape {variance_shape}")
                        print("Falling back to standard demosaicing")
                        demosaiced_frame = self.scmos.bayer_demosaic_stack_grayscale(frame)
                    else:
                        # Use variance-aware demosaicing
                        demosaiced_frame = self.scmos.variance_aware_malvar_demosaic(
                            frame,
                            variance_map=camera_data_to_use["variance"],
                            offset_map=camera_data_to_use["offset"],
                            gain=camera_data_to_use["gain"],
                            grayscale=True
                        )
                else:
                    # Use standard grayscale demosaicing
                    demosaiced_frame = self.scmos.bayer_demosaic_stack_grayscale(frame)

                detected_spots = self.sdf.detect_puncta_in_image(
                    image=demosaiced_frame,
                    pfa=pfa,
                    wavelength=wavelength,
                    sigma=sigma,
                    fraction_true=fraction_true,
                    pixel_size=0.069,  # Standard pixel size
                    NA=1.49,  # Standard NA
                    mf_factor=3.0,  # Standard match filter factor
                    local_factor=3.0,  # Standard local factor
                )
                num_spots = len(detected_spots) if detected_spots is not None else 0
                valid_spots = (
                    detected_spots if detected_spots is not None else np.array([])
                )
                results.append((valid_spots, num_spots))

            except Exception as e:
                print(f"Error in spot detection for frame {i+1}: {e}")
                results.append((np.array([]), 0))

        return results

    def plot_detection_results(
        self,
        image: np.ndarray,
        spots: np.ndarray,
        pfa: float,
        sigma: float,
        fraction_true: float,
        folder_name: str,
    ):
        """Plot the detection results using PlottingFunctions (interactive or file-based)"""
        global INTERACTIVE_DISPLAY

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Original image (left panel)
        self.pf.image_plot(
            axs=ax1,
            data=image,
            vmin=np.percentile(image, 1),
            vmax=np.percentile(image, 99),
            cmap="gist_gray",
            cbar="off",
        )
        ax1.set_title(f"Original Image\n{folder_name}")
        ax1.axis("off")

        # Image with detected spots using PlottingFunctions (right panel)
        if len(spots) > 0:
            self.pf.image_scatter_plot(
                ax2,
                data=image,
                xdata=spots[:, 0],
                ydata=spots[:, 1],
                vmin=float(np.percentile(image, 1)),
                vmax=float(np.percentile(image, 99)),
                cmap="gist_gray",
                cbar="off",  # Disable colorbar for cleaner display
                scattercolor="red",
                s=50,  # Marker size
                alpha=1.0,
                scatteralpha=0.8,
            )
        else:
            # No spots detected, just show image
            self.pf.image_plot(
                axs=ax2,
                data=image,
                vmin=np.percentile(image, 1),
                vmax=np.percentile(image, 99),
                cmap="gist_gray",
                cbar="off",
            )

        ax2.set_title(
            f"Detected Spots ({len(spots)} found)\n"
            f"PFA: {pfa:.0e}, σ: {sigma}, fraction_true: {fraction_true}"
        )
        ax2.axis("off")

        plt.tight_layout()

        if INTERACTIVE_DISPLAY:
            plt.show(block=False)
            return fig
        else:
            # Save to file and notify user
            safe_name = "".join(
                c for c in folder_name if c.isalnum() or c in (" ", "-", "_")
            ).strip()
            filename = f"nile_red_threshold_preview_{safe_name}.png"
            plt.savefig(filename, dpi=150, bbox_inches="tight")
            print(f"📸 Detection preview saved: {filename}")
            plt.close(fig)
            return filename

    def plot_detection_results_multi_frame(
        self,
        frames: List[np.ndarray],
        detection_results: List[Tuple[np.ndarray, int]],
        pfa: float,
        sigma: float,
        fraction_true: float,
        folder_name: str,
    ):
        """Plot the detection results for multiple frames using PlottingFunctions column layout"""
        global INTERACTIVE_DISPLAY

        # Create a 3-row column plot using PlottingFunctions
        fig, axs = self.pf.one_column_plot(npanels=3, ratios=[1, 1, 1], height=15)

        # Ensure axs is always a list for consistent indexing
        if not isinstance(axs, (list, np.ndarray)):
            axs = [axs]

        frame_labels = ["Frame 1", "Frame 10", "Frame 20"]
        total_spots = sum(num_spots for _, num_spots in detection_results)

        for i, (frame, (spots, num_spots)) in enumerate(zip(frames, detection_results)):
            if i >= len(axs):  # Safety check
                break

            # Plot image with detected spots overlaid
            if len(spots) > 0:
                self.pf.image_scatter_plot(
                    axs=axs[i],
                    data=frame,
                    xdata=spots[:, 0],
                    ydata=spots[:, 1],
                    vmin=float(np.percentile(frame, 1)),
                    vmax=float(np.percentile(frame, 99)),
                    s=30,  # Smaller spots for cleaner display
                    scattercolor="red",
                    cmap="gist_gray",
                    cbar="off",
                    scatteralpha=0.8,
                )
            else:
                # No spots detected, just show image
                self.pf.image_plot(
                    axs=axs[i],
                    data=frame,
                    vmin=float(np.percentile(frame, 1)),
                    vmax=float(np.percentile(frame, 99)),
                    cmap="gist_gray",
                    cbar="off",
                )

            axs[i].set_title(f"{frame_labels[i]} - {num_spots} spots")
            axs[i].axis("off")

        # Add overall title with parameters
        fig.suptitle(
            f"{os.path.basename(folder_name)} - Total: {total_spots} spots\n"
            f"PFA: {pfa:.0e}, σ: {sigma}, fraction_true: {fraction_true}",
            fontsize=14,
            y=0.98,
        )

        if INTERACTIVE_DISPLAY:
            plt.tight_layout()
            plt.show(block=False)
            return fig
        else:
            # Save to file
            safe_name = (
                os.path.basename(folder_name).replace(" ", "_").replace("/", "_")
            )
            filename = f"nile_red_threshold_test_{safe_name}_multiframe.png"
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"📸 Multi-frame detection preview saved: {filename}")
            return filename

    def plot_detection_and_fitting_results(
        self,
        detection_frames: List[np.ndarray],
        fitting_frames: List[np.ndarray],
        detection_results: List[Tuple[np.ndarray, int]],
        fitting_results: List[np.ndarray],
        pfa: float,
        sigma: float,
        fraction_true: float,
        folder_name: str,
        use_temporal_median: bool,
    ):
        """Plot detection and fitting results in two separate windows.

        Args:
            detection_frames: Frames used for detection (original/demosaiced)
            fitting_frames: Frames used for fitting (median-subtracted or original)
            detection_results: List of (detected_spots, num_spots) tuples
            fitting_results: List of fitted spot coordinates arrays
            pfa, sigma, fraction_true: Detection parameters
            folder_name: Name of folder being processed
            use_temporal_median: Whether temporal median was used

        Returns:
            Tuple of (detection_fig, fitting_fig) or filenames if not interactive
        """
        global INTERACTIVE_DISPLAY

        # Create figure 1: Detection results
        fig1, axs1 = self.pf.one_column_plot(npanels=3, ratios=[1, 1, 1], height=15)
        if not isinstance(axs1, (list, np.ndarray)):
            axs1 = [axs1]

        frame_labels = ["Frame 1", "Frame 10", "Frame 20"]
        total_detected = sum(num_spots for _, num_spots in detection_results)

        for i, (frame, (spots, num_spots)) in enumerate(zip(detection_frames, detection_results)):
            if i >= len(axs1):
                break

            if len(spots) > 0:
                self.pf.image_scatter_plot(
                    axs=axs1[i],
                    data=frame,
                    xdata=spots[:, 0],
                    ydata=spots[:, 1],
                    vmin=float(np.percentile(frame, 1)),
                    vmax=float(np.percentile(frame, 99)),
                    s=30,
                    scattercolor="red",
                    cmap="gist_gray",
                    cbar="off",
                    scatteralpha=0.8,
                )
            else:
                self.pf.image_plot(
                    axs=axs1[i],
                    data=frame,
                    vmin=float(np.percentile(frame, 1)),
                    vmax=float(np.percentile(frame, 99)),
                    cmap="gist_gray",
                    cbar="off",
                )
            axs1[i].set_title(f"{frame_labels[i]} - {num_spots} spots detected")
            axs1[i].axis("off")

        fig1.suptitle(
            f"{os.path.basename(folder_name)} - DETECTION\n"
            f"Total detected: {total_detected} spots | PFA: {pfa:.0e}, σ: {sigma}, fraction_true: {fraction_true}",
            fontsize=14,
            y=0.98,
        )

        # Create figure 2: Fitting results
        fig2, axs2 = self.pf.one_column_plot(npanels=3, ratios=[1, 1, 1], height=15)
        if not isinstance(axs2, (list, np.ndarray)):
            axs2 = [axs2]

        total_fitted = sum(len(fitted) for fitted in fitting_results)
        data_type = "Temporal Median Subtracted" if use_temporal_median else "Original"

        for i, (frame, fitted_spots) in enumerate(zip(fitting_frames, fitting_results)):
            if i >= len(axs2):
                break

            if len(fitted_spots) > 0:
                self.pf.image_scatter_plot(
                    axs=axs2[i],
                    data=frame,
                    xdata=fitted_spots[:, 0],
                    ydata=fitted_spots[:, 1],
                    vmin=float(np.percentile(frame, 1)),
                    vmax=float(np.percentile(frame, 99)),
                    s=30,
                    scattercolor="lime",
                    cmap="gist_gray",
                    cbar="off",
                    scatteralpha=0.8,
                )
            else:
                self.pf.image_plot(
                    axs=axs2[i],
                    data=frame,
                    vmin=float(np.percentile(frame, 1)),
                    vmax=float(np.percentile(frame, 99)),
                    cmap="gist_gray",
                    cbar="off",
                )
            axs2[i].set_title(f"{frame_labels[i]} - {len(fitted_spots)} spots fitted")
            axs2[i].axis("off")

        fig2.suptitle(
            f"{os.path.basename(folder_name)} - FITTING ({data_type})\n"
            f"Total fitted: {total_fitted} spots",
            fontsize=14,
            y=0.98,
        )

        if INTERACTIVE_DISPLAY:
            plt.figure(fig1.number)
            plt.tight_layout()
            plt.show(block=False)
            plt.figure(fig2.number)
            plt.tight_layout()
            plt.show(block=False)
            return (fig1, fig2)
        else:
            safe_name = os.path.basename(folder_name).replace(" ", "_").replace("/", "_")
            filename1 = f"nile_red_detection_{safe_name}.png"
            filename2 = f"nile_red_fitting_{safe_name}.png"

            fig1.tight_layout()
            fig1.savefig(filename1, dpi=300, bbox_inches="tight")
            plt.close(fig1)
            print(f"📸 Detection preview saved: {filename1}")

            fig2.tight_layout()
            fig2.savefig(filename2, dpi=300, bbox_inches="tight")
            plt.close(fig2)
            print(f"📸 Fitting preview saved: {filename2}")

            return (filename1, filename2)

    def interactive_parameter_tuning(
        self, folder_path: str, folder_type: str, default_wavelength: float
    ) -> Union[Dict, None, str]:
        """Interactive parameter tuning for a single folder"""
        folder_name = os.path.basename(folder_path)
        print(f"\n{'='*80}")
        print(f"PROCESSING: {folder_name}")
        print(f"Path: {folder_path}")
        print(f"Type: {folder_type}, Wavelength: {default_wavelength}")
        print(f"{'='*80}")

        # Start with default parameters
        current_pfa = self.default_pfa
        current_sigma = self.default_sigma
        current_fraction_true = self.default_true_fraction
        current_wavelength = default_wavelength
        current_use_variance_aware = self.default_use_variance_aware
        current_temporal_median_mode = self.default_temporal_median_mode
        current_temporal_median_window = self.default_temporal_median_window

        # Load data based on temporal median mode
        frames = None
        frame_stack_data = None

        if current_temporal_median_mode != TemporalMedianMode.NONE:
            print(f"\nTemporal median mode: {current_temporal_median_mode.name} - loading frame stacks...")
            frame_stack_data = self.load_frame_stacks_for_temporal_median(
                folder_path, current_temporal_median_window
            )
            if frame_stack_data is None:
                print("Could not load frame stacks, skipping...")
                return None
        else:
            print("\nTemporal median disabled - loading individual test frames...")
            frames = self.load_test_frames(folder_path)
            if frames is None:
                print("Could not load test frames, skipping...")
                return None

        fig_or_file = None

        while True:
            # Reload data if temporal median mode changed
            if current_temporal_median_mode != TemporalMedianMode.NONE and frame_stack_data is None:
                print(f"\nTemporal median mode: {current_temporal_median_mode.name} - loading frame stacks...")
                frame_stack_data = self.load_frame_stacks_for_temporal_median(
                    folder_path, current_temporal_median_window
                )
                if frame_stack_data is None:
                    print("Could not load frame stacks, disabling temporal median...")
                    current_temporal_median_mode = TemporalMedianMode.NONE

            if current_temporal_median_mode == TemporalMedianMode.NONE and frames is None:
                print("\nLoading individual test frames...")
                frames = self.load_test_frames(folder_path)
                if frames is None:
                    print("Could not load test frames, skipping folder...")
                    return None

            # Process all frames based on mode
            detection_results = []
            detection_frames_for_plot = []
            fitting_frames_for_plot = []
            original_frames_for_fitting = []

            if current_temporal_median_mode != TemporalMedianMode.NONE and frame_stack_data is not None:
                # Use temporal median processing
                print(f"\nPerforming detection and fitting with temporal median mode: {current_temporal_median_mode.name}")

                for i, (stack, test_idx, display_frame) in enumerate(zip(
                    frame_stack_data['stacks'],
                    frame_stack_data['test_frame_indices'],
                    frame_stack_data['display_frames']
                )):
                    print(f"Processing stack {i+1}/3...")

                    # Compute temporal median subtracted data
                    half_window = current_temporal_median_window // 2
                    start_idx = max(0, test_idx - half_window)
                    end_idx = min(len(stack), test_idx + half_window + 1)
                    window_frames = stack[start_idx:end_idx]
                    temporal_median = np.median(window_frames, axis=0).astype(np.float64)
                    median_subtracted = display_frame.astype(np.float64) - temporal_median
                    median_subtracted = np.maximum(median_subtracted, 0)  # Clip negatives

                    print(f"  Using {len(window_frames)} frames (indices {start_idx}-{end_idx-1}) for median")

                    # Determine which data to use for detection and fitting based on mode
                    if current_temporal_median_mode == TemporalMedianMode.DETECTION_AND_FITTING:
                        detection_frame = median_subtracted
                        fitting_frame = median_subtracted
                        print(f"  Mode: BOTH detection and fitting use temporal median")
                    elif current_temporal_median_mode == TemporalMedianMode.FITTING_ONLY:
                        detection_frame = display_frame
                        fitting_frame = median_subtracted
                        print(f"  Mode: Detection uses original, fitting uses temporal median")
                    else:  # NONE (shouldn't reach here, but handle anyway)
                        detection_frame = display_frame
                        fitting_frame = display_frame

                    # Perform detection
                    spots, num_spots = self.test_spot_detection(
                        detection_frame,
                        current_pfa,
                        current_sigma,
                        current_fraction_true,
                        current_wavelength,
                        current_use_variance_aware,
                    )
                    detection_results.append((spots, num_spots))
                    detection_frames_for_plot.append(detection_frame)
                    fitting_frames_for_plot.append(fitting_frame)
                    original_frames_for_fitting.append(display_frame)

            else:
                # No temporal median
                print("\nPerforming detection and fitting without temporal median...")
                detection_results = self.test_spot_detection_multi_frame(
                    frames,
                    current_pfa,
                    current_sigma,
                    current_fraction_true,
                    current_wavelength,
                    current_use_variance_aware,
                )
                detection_frames_for_plot = frames
                fitting_frames_for_plot = frames
                original_frames_for_fitting = frames

            # Perform fitting on detected spots
            print("\nFitting detected spots...")
            fitting_results = []
            for i, ((spots, num_spots), orig_frame, fit_frame) in enumerate(zip(
                detection_results,
                original_frames_for_fitting,
                fitting_frames_for_plot
            )):
                print(f"  Fitting frame {i+1}/3 ({num_spots} spots)...")
                if num_spots > 0:
                    fitted_coords = self.fit_detected_spots(
                        fit_frame,
                        spots,
                        smoothing_function=None,
                        ROI_size=16,
                    )
                    fitting_results.append(fitted_coords)
                    print(f"    Successfully fitted {len(fitted_coords)}/{num_spots} spots")
                else:
                    fitting_results.append(np.array([]))

            # Calculate total spots
            total_detected = sum(num_spots for _, num_spots in detection_results)
            total_fitted = sum(len(fitted) for fitted in fitting_results)

            # Close previous plots if interactive mode
            if INTERACTIVE_DISPLAY and fig_or_file is not None:
                if isinstance(fig_or_file, tuple):
                    for fig in fig_or_file:
                        if fig is not None:
                            plt.close(fig)
                else:
                    plt.close(fig_or_file)

            # Plot results using dual window display
            use_temporal_median = current_temporal_median_mode != TemporalMedianMode.NONE
            fig_or_file = self.plot_detection_and_fitting_results(
                detection_frames_for_plot,
                fitting_frames_for_plot,
                detection_results,
                fitting_results,
                current_pfa,
                current_sigma,
                current_fraction_true,
                folder_name,
                use_temporal_median,
            )

            variance_aware_status = "enabled" if current_use_variance_aware else "disabled"
            temporal_median_mode_name = current_temporal_median_mode.name
            print(f"\nCurrent parameters:")
            print(f"  PFA (probability of false alarm): {current_pfa:.0e}")
            print(f"  Sigma : {current_sigma}")
            print(f"  Fraction true : {current_fraction_true}")
            print(f"  Wavelength: {current_wavelength}")
            print(f"  Variance-aware demosaicing: {variance_aware_status}")
            print(f"  Temporal median mode: {temporal_median_mode_name}")
            if current_temporal_median_mode != TemporalMedianMode.NONE:
                print(f"  Temporal median window: {current_temporal_median_window} frames")
            print(f"\nResults:")
            print(f"  Detected spots: {total_detected} (across 3 frames)")
            print(f"  Fitted spots: {total_fitted} (across 3 frames)")

            print(f"\nOptions:")
            print(f"  1. Adjust PFA (current: {current_pfa:.0e})")
            print(f"  2. Adjust sigma (current: {current_sigma} pixels)")
            print(f"  3. Adjust Fraction true (current: {current_fraction_true})")
            print(f"  4. Adjust wavelength (current: {current_wavelength})")
            print(f"  5. Toggle variance-aware demosaicing (current: {variance_aware_status})")
            print(f"  6. Change temporal median mode (current: {temporal_median_mode_name})")
            print(f"  7. Adjust temporal median window (current: {current_temporal_median_window} frames)")
            print(f"  8. Accept current parameters")
            print(f"  9. Skip this folder")
            print(f"  q. Quit")

            choice = input("Enter choice (1-9 or q): ").strip()

            if choice == "1":
                try:
                    new_pfa = float(
                        input(f"Enter new PFA (current: {current_pfa:.0e}): ").strip()
                    )
                    current_pfa = new_pfa
                except ValueError:
                    print("Invalid input, keeping current value")

            elif choice == "2":
                try:
                    new_sigma = float(
                        input(f"Enter new sigma (current: {current_sigma}): ").strip()
                    )
                    if 0.1 <= new_sigma <= 10.0:
                        current_sigma = new_sigma
                    else:
                        print(
                            "Sigma must be between 0.1 and 10.0 (PSF standard deviation in pixels)"
                        )
                except ValueError:
                    print("Invalid input, keeping current value")

            elif choice == "3":
                try:
                    new_fraction_true = float(
                        input(
                            f"Enter new Fraction True (current: {current_fraction_true}): "
                        ).strip()
                    )
                    current_fraction_true = new_fraction_true
                except ValueError:
                    print("Invalid input, keeping current value")

            elif choice == "4":
                try:
                    new_wavelength = float(
                        input(
                            f"Enter new wavelength (current: {current_wavelength}): "
                        ).strip()
                    )
                    current_wavelength = new_wavelength
                except ValueError:
                    print("Invalid input, keeping current value")

            elif choice == "5":
                # Toggle variance-aware demosaicing
                current_use_variance_aware = not current_use_variance_aware
                new_status = "enabled" if current_use_variance_aware else "disabled"
                print(f"Variance-aware demosaicing {new_status}")
                if not current_use_variance_aware:
                    print("  Note: Using standard grayscale demosaicing")
                elif not self.camera_data:
                    print("  Warning: Camera calibration data not available - will use standard demosaicing")

            elif choice == "6":
                # Cycle through temporal median modes
                mode_options = [TemporalMedianMode.NONE, TemporalMedianMode.FITTING_ONLY, TemporalMedianMode.DETECTION_AND_FITTING]
                current_idx = mode_options.index(current_temporal_median_mode)
                next_idx = (current_idx + 1) % len(mode_options)
                current_temporal_median_mode = mode_options[next_idx]

                print(f"Temporal median mode changed to: {current_temporal_median_mode.name}")
                if current_temporal_median_mode == TemporalMedianMode.NONE:
                    print("  No temporal median subtraction")
                    frames = None
                    frame_stack_data = None
                elif current_temporal_median_mode == TemporalMedianMode.FITTING_ONLY:
                    print(f"  Temporal median for FITTING only (window={current_temporal_median_window} frames)")
                    frame_stack_data = None
                    frames = None
                elif current_temporal_median_mode == TemporalMedianMode.DETECTION_AND_FITTING:
                    print(f"  Temporal median for BOTH detection and fitting (window={current_temporal_median_window} frames)")
                    frame_stack_data = None
                    frames = None

            elif choice == "7":
                # Adjust temporal median window
                try:
                    new_window = int(
                        input(f"Enter new temporal median window (current: {current_temporal_median_window} frames): ").strip()
                    )
                    if 10 <= new_window <= 500:
                        old_window = current_temporal_median_window
                        current_temporal_median_window = new_window
                        print(f"Temporal median window changed from {old_window} to {new_window} frames")
                        # Need to reload frame stacks with new window size
                        if current_temporal_median_mode != TemporalMedianMode.NONE:
                            frame_stack_data = None  # Force reload on next iteration
                    else:
                        print("Window must be between 10 and 500 frames")
                except ValueError:
                    print("Invalid input, keeping current value")

            elif choice == "8":
                # Accept parameters
                if INTERACTIVE_DISPLAY and fig_or_file is not None:
                    if isinstance(fig_or_file, tuple):
                        for fig in fig_or_file:
                            if fig is not None:
                                plt.close(fig)
                    else:
                        plt.close(fig_or_file)
                return {
                    "folder_path": folder_path,
                    "folder_type": folder_type,
                    "pfa": current_pfa,
                    "sigma": current_sigma,
                    "fraction_true": current_fraction_true,
                    "wavelength": current_wavelength,
                    "use_variance_aware": current_use_variance_aware,
                    "temporal_median_mode": current_temporal_median_mode.value,  # Save as integer
                    "temporal_median_window": current_temporal_median_window,
                    "detected_spots": total_detected,
                }

            elif choice == "9":
                # Skip folder
                if INTERACTIVE_DISPLAY and fig_or_file is not None:
                    if isinstance(fig_or_file, tuple):
                        for fig in fig_or_file:
                            if fig is not None:
                                plt.close(fig)
                    else:
                        plt.close(fig_or_file)
                return None

            elif choice.lower() == "q":
                # Quit
                if INTERACTIVE_DISPLAY and fig_or_file is not None:
                    if isinstance(fig_or_file, tuple):
                        for fig in fig_or_file:
                            if fig is not None:
                                plt.close(fig)
                    else:
                        plt.close(fig_or_file)
                return "quit"

            else:
                print("Invalid choice, please try again")

    def save_threshold_parameters(self, output_file: str = "20250930_nile_red_threshold_parameters.txt"):
        """Save threshold parameters to file for 20250930_NileRedAnalysis.sh"""
        output_path = Path(output_file)

        # Save as JSON for easy parsing
        json_output = output_path.with_suffix(".json")
        with open(json_output, "w") as f:
            json.dump(self.threshold_results, f, indent=2)

        # Also save as text file for batch script
        with open(output_path, "w") as f:
            f.write("# Threshold parameters for Nile Red Analysis - 20250930\n")
            f.write("# Generated by 20250930_NileRedAnalysisTuner.py\n")
            f.write("# Format: folder_path|pfa|sigma|fraction_true|wavelength|use_variance_aware|temporal_median_mode|temporal_median_window\n")
            f.write("# temporal_median_mode: 0=NONE, 1=FITTING_ONLY, 2=DETECTION_AND_FITTING\n")
            f.write("#\n")

            for folder_path, params in self.threshold_results.items():
                use_variance_aware_str = "true" if params.get('use_variance_aware', True) else "false"
                temporal_median_mode = params.get('temporal_median_mode', 1)  # Default to FITTING_ONLY
                temporal_median_window = params.get('temporal_median_window', 500)
                f.write(
                    f"{folder_path}|{params['pfa']:.0e}|{params['sigma']:.1f}|{params['fraction_true']:.1f}|{params['wavelength']:.3f}|{use_variance_aware_str}|{temporal_median_mode}|{temporal_median_window}\n"
                )

        print(f"\nNile Red threshold parameters saved to:")
        print(f"  JSON format: {json_output}")
        print(f"  Text format: {output_path}")
        print(f"  Total folders configured: {len(self.threshold_results)}")

    def run(self):
        """Main interactive loop"""
        print("=" * 80)
        print("Nile Red Interactive Threshold Tuner - 20250930")
        print("Bacteria with Nile Red experiment")
        print("=" * 80)
        print("This tool helps determine optimal spot detection parameters")
        print("(including temporal median settings) for Nile Red stained bacteria.\n")

        # Get all folders to process
        all_folders = self.get_all_processing_folders()
        print(f"Found {len(all_folders)} folders to process")

        # Filter to only folders that exist
        existing_folders = [(f, t, w) for f, t, w in all_folders if os.path.isdir(f)]
        missing_folders = [f for f, t, w in all_folders if not os.path.isdir(f)]

        if missing_folders:
            print(f"Warning: {len(missing_folders)} folders not found (skipping):")
            for folder in missing_folders[:5]:  # Show first 5
                print(f"  {folder}")
            if len(missing_folders) > 5:
                print(f"  ... and {len(missing_folders) - 5} more")

        print(f"\nWill process {len(existing_folders)} existing folders")

        # Ask user if they want to continue
        if existing_folders:
            response = input("\nProceed with parameter tuning? (y/n): ").strip().lower()
            if response != "y":
                print("Aborted by user")
                return
        else:
            print("No folders found to process!")
            return

        # Process each folder
        for i, (folder_path, folder_type, default_wavelength) in enumerate(
            existing_folders
        ):
            print(f"\nProgress: {i+1}/{len(existing_folders)}")

            result = self.interactive_parameter_tuning(
                folder_path, folder_type, default_wavelength
            )

            if result == "quit":
                print("\nQuitting by user request")
                break
            elif result is not None:
                # Store results
                self.threshold_results[folder_path] = result
                print(f"Parameters saved for {os.path.basename(folder_path)}")
            else:
                print(f"Skipped {os.path.basename(folder_path)}")

        # Save results
        if self.threshold_results:
            self.save_threshold_parameters()

            # Summary
            print(f"\n{'='*80}")
            print("SUMMARY")
            print(f"{'='*80}")
            print(
                f"Folders processed: {len(self.threshold_results)}/{len(existing_folders)}"
            )

            # Show parameter distribution
            pfa_values = [params["pfa"] for params in self.threshold_results.values()]
            sigma_values = [
                params["sigma"] for params in self.threshold_results.values()
            ]
            fraction_true_values = [
                params["fraction_true"] for params in self.threshold_results.values()
            ]

            print(f"PFA range: {min(pfa_values):.0e} - {max(pfa_values):.0e}")
            print(f"Sigma range: {min(sigma_values):.1f} - {max(sigma_values):.1f}")
            print(
                f"Fraction true range: {min(fraction_true_values):.3f} - {max(fraction_true_values):.3f}"
            )

        else:
            print("\nNo parameters were saved.")


if __name__ == "__main__":
    # Check if virtual environment is activated
    if "/pyBayerSMLM/" not in sys.executable:
        print("Warning: pyBayerSMLM virtual environment may not be activated")
        print(
            "Please run: source /home/jbeckwith/.virtualenvs/pyBayerSMLM/bin/activate"
        )

    tuner = NileRedThresholdTuner()
    tuner.run()