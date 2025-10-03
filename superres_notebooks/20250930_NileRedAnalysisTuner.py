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

        # Default parameters optimized for Nile Red staining
        self.default_pfa = 1e-4
        self.default_sigma = 1.5
        self.default_true_fraction = 0.2
        self.default_wavelength = 0.700  # 700nm - near-infrared region typical for Nile Red
        self.default_use_variance_aware = True  # Default to variance-aware demosaicing
        self.default_use_temporal_median = True  # Default to using temporal median (ON by default)
        self.default_temporal_median_window = 100  # Default window size in frames

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

    def load_test_frames(self, folder_path: str) -> Optional[List[np.ndarray]]:
        """Load 3 test frames with improved logic for single vs multiple files"""
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

    def test_spot_detection(
        self,
        image: np.ndarray,
        pfa: float,
        sigma: float,
        fraction_true: float,
        wavelength: float,
        use_variance_aware: bool = True,
    ) -> Tuple[np.ndarray, int]:
        """Test spot detection with given parameters on demosaiced image"""
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

        # Load test frames
        frames = self.load_test_frames(folder_path)
        if frames is None:
            print("Could not load test frames, skipping...")
            return None

        # Start with default parameters
        current_pfa = self.default_pfa
        current_sigma = self.default_sigma
        current_fraction_true = self.default_true_fraction
        current_wavelength = default_wavelength
        current_use_variance_aware = self.default_use_variance_aware
        current_use_temporal_median = self.default_use_temporal_median
        current_temporal_median_window = self.default_temporal_median_window

        fig_or_file = None

        while True:
            # Test current parameters on all frames
            detection_results = self.test_spot_detection_multi_frame(
                frames,
                current_pfa,
                current_sigma,
                current_fraction_true,
                current_wavelength,
                current_use_variance_aware,
            )

            # Calculate total spots across all frames
            total_spots = sum(num_spots for _, num_spots in detection_results)

            # Close previous plot if interactive mode
            if INTERACTIVE_DISPLAY and fig_or_file is not None:
                plt.close(fig_or_file)

            # Plot results using multi-frame display
            fig_or_file = self.plot_detection_results_multi_frame(
                frames,
                detection_results,
                current_pfa,
                current_sigma,
                current_fraction_true,
                folder_name,
            )

            variance_aware_status = "enabled" if current_use_variance_aware else "disabled"
            temporal_median_status = "ON" if current_use_temporal_median else "OFF"
            print(f"\nCurrent parameters:")
            print(f"  PFA (probability of false alarm): {current_pfa:.0e}")
            print(f"  Sigma : {current_sigma}")
            print(f"  Fraction true : {current_fraction_true}")
            print(f"  Wavelength: {current_wavelength}")
            print(f"  Variance-aware demosaicing: {variance_aware_status}")
            print(f"  Temporal median: {temporal_median_status}")
            if current_use_temporal_median:
                print(f"  Temporal median window: {current_temporal_median_window} frames")
            print(f"  Detected spots: {total_spots} (across 3 frames)")

            print(f"\nOptions:")
            print(f"  1. Adjust PFA (current: {current_pfa:.0e})")
            print(f"  2. Adjust sigma (current: {current_sigma} pixels)")
            print(f"  3. Adjust Fraction true (current: {current_fraction_true})")
            print(f"  4. Adjust wavelength (current: {current_wavelength})")
            print(f"  5. Toggle variance-aware demosaicing (current: {variance_aware_status})")
            print(f"  6. Toggle temporal median (current: {temporal_median_status})")
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
                # Toggle temporal median
                current_use_temporal_median = not current_use_temporal_median
                new_status = "ON" if current_use_temporal_median else "OFF"
                print(f"Temporal median subtraction {new_status}")
                if current_use_temporal_median:
                    print(f"  Will use moving median window of {current_temporal_median_window} frames")
                else:
                    print("  Note: No temporal median subtraction will be applied")

            elif choice == "7":
                # Adjust temporal median window
                try:
                    new_window = int(
                        input(f"Enter new temporal median window (current: {current_temporal_median_window} frames): ").strip()
                    )
                    if 10 <= new_window <= 500:
                        current_temporal_median_window = new_window
                        print(f"Temporal median window set to {new_window} frames")
                    else:
                        print("Window must be between 10 and 500 frames")
                except ValueError:
                    print("Invalid input, keeping current value")

            elif choice == "8":
                # Accept parameters
                if INTERACTIVE_DISPLAY and fig_or_file is not None:
                    plt.close(fig_or_file)
                return {
                    "folder_path": folder_path,
                    "folder_type": folder_type,
                    "pfa": current_pfa,
                    "sigma": current_sigma,
                    "fraction_true": current_fraction_true,
                    "wavelength": current_wavelength,
                    "use_variance_aware": current_use_variance_aware,
                    "use_temporal_median": current_use_temporal_median,
                    "temporal_median_window": current_temporal_median_window,
                    "detected_spots": total_spots,
                }

            elif choice == "9":
                # Skip folder
                if INTERACTIVE_DISPLAY and fig_or_file is not None:
                    plt.close(fig_or_file)
                return None

            elif choice.lower() == "q":
                # Quit
                if INTERACTIVE_DISPLAY and fig_or_file is not None:
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
            f.write("# Format: folder_path|pfa|sigma|fraction_true|wavelength|use_variance_aware|use_temporal_median|temporal_median_window\n")
            f.write("#\n")

            for folder_path, params in self.threshold_results.items():
                use_variance_aware_str = "true" if params.get('use_variance_aware', True) else "false"
                use_temporal_median_str = "true" if params.get('use_temporal_median', True) else "false"
                temporal_median_window = params.get('temporal_median_window', 100)
                f.write(
                    f"{folder_path}|{params['pfa']:.0e}|{params['sigma']:.1f}|{params['fraction_true']:.1f}|{params['wavelength']:.3f}|{use_variance_aware_str}|{use_temporal_median_str}|{temporal_median_window}\n"
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