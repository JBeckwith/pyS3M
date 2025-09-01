#!/usr/bin/env python3
"""
Spot Detection Validation Test Suite

Comprehensive validation framework for spot detection functions to quantify
false positive and false negative rates using simulated ground truth data.

This script simulates 10x10 grids of puncta with known positions, processes them
through the complete imaging pipeline (simulation → conversion → detection), 
and evaluates detection performance with detailed metrics.

Author: pyBayerSMLM
Date: September 2025
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import time
from copy import deepcopy
import warnings
from scipy.spatial.distance import cdist

# Add module directory to path
module_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(module_dir)

# Import project modules
import IOFunctions
import PSFFunctions
import sCMOSFunctions
import SpotDetectionFunctions
import Multicolour_Simulation_Functions
import SpectralFunctions
import PlottingFunctions

# Initialise function classes
IO = IOFunctions.IO_Functions()
PSF = PSFFunctions.PSF_Functions()
sCMOS = sCMOSFunctions.sCMOS_Functions()
SD = SpotDetectionFunctions.SpotDetection_Functions()
MSF = Multicolour_Simulation_Functions.MultiC_Sim_Funcs()
SF = SpectralFunctions.Spectral_Funcs()
plotter = PlottingFunctions.Plotter()


@dataclass
class ValidationConfig:
    """Configuration parameters for spot detection validation."""

    # Grid simulation parameters
    grid_size: int = 10  # 10x10 grid
    grid_spacing_microns: float = 1.0  # 1 micron spacing
    pixel_size_nm: float = 69.0  # Pixel size in nanometers
    image_size_pixels: int = 200  # Total image size in pixels

    # Photon and noise parameters
    n_photons_range: Tuple[int, int] = (1000, 5000)  # Photon count range
    background_photons: float = 40.0  # Background photons per pixel

    # Bootstrap sampling
    n_bootstrap: int = 50  # Number of bootstrap samples
    camera_region_size: int = 50  # Size of camera regions to sample

    # Detection parameters
    pfa: float = 1e-4  # Probability of false alarm
    wavelength: float = 0.55  # Wavelength in microns
    NA: float = 1.49  # Numerical aperture
    mf_factor: float = 3.0  # Matched filter factor
    local_factor: float = 3.0  # Local maximum factor
    test_bayer_processing: bool = True  # Test both bayer_image=True/False

    # Analysis parameters
    detection_tolerance_nm: float = 150.0  # Maximum distance for true positive


@dataclass
class DetectionMetrics:
    """Container for detection performance metrics."""

    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    true_positive_rate: float
    false_positive_rate: float
    detection_positions: np.ndarray
    ground_truth_positions: np.ndarray
    bayer_processing: bool = None  # Whether Bayer averaging was used


class SpotDetectionValidator:
    """
    Comprehensive validation framework for spot detection functions.

    This class provides methods to:
    1. Generate realistic 10x10 grids of puncta with ground truth positions
    2. Simulate camera images with realistic noise characteristics
    3. Convert simulated images to photoelectron counts
    4. Apply spot detection algorithms
    5. Evaluate detection performance with detailed metrics
    """

    def __init__(self, config: ValidationConfig):
        """Initialise validator with configuration parameters."""
        self.config = config
        self.camera_params = None
        self.spectral_data = None

    def load_camera_calibration(self, camera_path: str) -> Dict[str, Any]:
        """
        Load camera calibration parameters from file system.

        Args:
            camera_path: Path to camera calibration directory

        Returns:
            Dictionary containing camera calibration parameters
        """
        print(f"Loading camera calibration from: {camera_path}")

        # Load calibration files
        gain = IO.read_tiff(os.path.join(camera_path, "gain.tif"))
        offset = IO.read_tiff(os.path.join(camera_path, "offset.tif"))
        variance = IO.read_tiff(os.path.join(camera_path, "variance.tif"))
        readnoise = IO.read_tiff(os.path.join(camera_path, "readnoise.tif"))
        rqe = IO.read_tiff(os.path.join(camera_path, "rqe.tif"))

        # Get image dimensions from gain map
        full_height, full_width = gain.shape
        print(f"Full camera dimensions: {full_height} x {full_width}")

        # Create camera parameter structure
        camera_params = {
            "gain": gain,
            "offset": offset,
            "variance": variance,
            "readnoise": readnoise,
            "rqe": rqe,
            "full_dimensions": (full_height, full_width),
        }

        self.camera_params = camera_params
        return camera_params

    def extract_camera_region(self, start_row: int, start_col: int) -> Dict[str, Any]:
        """
        Extract a subregion of camera parameters for simulation.

        Args:
            start_row: Starting row index
            start_col: Starting column index

        Returns:
            Dictionary with extracted camera parameters
        """
        if self.camera_params is None:
            raise ValueError(
                "Camera calibration not loaded. Call load_camera_calibration() first."
            )

        size = self.config.image_size_pixels

        # Extract regions with bounds checking
        def extract_region(array, start_row, start_col, size):
            max_row, max_col = array.shape
            end_row = min(start_row + size, max_row)
            end_col = min(start_col + size, max_col)
            return array[start_row:end_row, start_col:end_col]

        region_params = {}
        for key in ["gain", "offset", "variance", "readnoise", "rqe"]:
            region_params[key] = extract_region(
                self.camera_params[key], start_row, start_col, size
            )

        # Generate Bayer masks for the region
        region_params["masks"] = MSF.get_masks(MSF.mosaic_unit, size, size)

        return region_params

    def generate_grid_positions(self) -> np.ndarray:
        """
        Generate ground truth positions for 10x10 grid of puncta.

        Returns:
            Array of shape (100, 2) with [x, y] positions in nanometers
        """
        grid_spacing_nm = self.config.grid_spacing_microns * 1000  # Convert to nm
        pixel_size = self.config.pixel_size_nm
        image_size_nm = self.config.image_size_pixels * pixel_size

        # Calculate grid positions centered in image
        grid_positions = []
        start_offset = (
            image_size_nm - (self.config.grid_size - 1) * grid_spacing_nm
        ) / 2

        for i in range(self.config.grid_size):
            for j in range(self.config.grid_size):
                x = start_offset + i * grid_spacing_nm
                y = start_offset + j * grid_spacing_nm
                grid_positions.append([x, y])

        return np.array(grid_positions)

    def simulate_camera_image(
        self,
        ground_truth_positions: np.ndarray,
        camera_region: Dict[str, Any],
        n_photons: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate camera image with ground truth puncta positions.

        Args:
            ground_truth_positions: Array of [x, y] positions in nm
            camera_region: Camera calibration for this region
            n_photons: Number of photons per punctum

        Returns:
            Tuple of (bayer_image, photoelectron_image)
        """
        # Set up spectral data (using ATTO550 as representative dye)
        if self.spectral_data is None:
            self.spectral_data = self._load_spectral_data()

        (
            wavelength,
            dye_spectrum,
            absolute_QYs,
            average_emission_wavelength,
            dye_pixel_efficiency,
        ) = self.spectral_data

        # Convert positions to pixels and create position dictionary
        positions_pixels = ground_truth_positions / self.config.pixel_size_nm

        # Create input format for simulation
        n_photons_dict = {"dye": n_photons}
        x0y0_dict = {"dye": np.array([positions_pixels[:, 0], positions_pixels[:, 1]])}

        # Generate camera image
        try:
            ground_truth_image, bayer_image = MSF.gen_camera_image_stack(
                camera_region,
                wavelength,
                average_emission_wavelength,
                dye_pixel_efficiency,
                n_photons_dict,
                x0y0_dict,
                sCMOS.var_weighted_uniform_filter,
                background_photons=self.config.background_photons,
                NA=self.config.NA,
                pixel_size=self.config.pixel_size_nm,
            )
        except Exception as e:
            print(f"Error in camera image simulation: {e}")
            raise

        # Convert to photoelectron counts as would be done for real data
        photoelectron_image = self._convert_to_photoelectrons(
            bayer_image, camera_region
        )

        return bayer_image, photoelectron_image

    def _load_spectral_data(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
        """Load spectral data for simulation using SpectralFunctions."""
        print("Loading spectral data using SpectralFunctions...")

        # Load camera quantum efficiency using SpectralFunctions
        try:
            # Get camera QE data (filters) from database
            wavelength, camera_qe_data = SF.get_dye_or_filter_data(
                ["CS505CU_B", "CS505CU_G", "CS505CU_R"], is_dye=False
            )

            # Restructure to match expected format [B, G, R]
            absolute_QYs = camera_qe_data.T  # Shape: (3, n_wavelengths) for B, G, R

        except Exception as e:
            print(f"Warning: Could not load camera QE from database: {e}")
            print("Falling back to direct CSV loading...")

            # Fallback to direct loading
            from Camera_QE import getpixelefficiency

            gpe = getpixelefficiency.GPE()
            R, G, B, wavelength = gpe.getpixelefficiency("Camera_QE/CS505CU_QE.csv")
            absolute_QYs = np.vstack([B, G, R])

        # Load ATTO550 dye spectrum using SpectralFunctions
        try:
            dye_wavelength, dye_emission_data = SF.get_dye_or_filter_data(
                ["ATTO550"], is_dye=True
            )

            # Ensure wavelength grids match
            if not np.array_equal(wavelength, dye_wavelength):
                print("Interpolating dye spectrum to camera wavelength grid...")
                dye_spectrum_interp = np.interp(
                    wavelength, dye_wavelength, dye_emission_data[0, :]
                )
            else:
                dye_spectrum_interp = dye_emission_data[0, :]

        except Exception as e:
            print(f"Warning: Could not load ATTO550 from database: {e}")
            print("Using fallback synthetic spectrum...")

            # Create a synthetic spectrum centered around 550nm
            center_wl = 550.0
            width = 30.0
            dye_spectrum_interp = np.exp(-0.5 * ((wavelength - center_wl) / width) ** 2)

        # Normalise dye spectrum
        dye_spectrum_interp = dye_spectrum_interp / np.sum(dye_spectrum_interp)
        dye_spectrum = np.array([dye_spectrum_interp])

        # Calculate average emission wavelength
        average_emission_wavelength = np.trapz(
            y=wavelength * (dye_spectrum.T / np.trapz(x=wavelength, y=dye_spectrum)).T,
            x=wavelength,
        )[0]

        # Calculate dye pixel efficiency
        dye_pixel_efficiency = np.dot(dye_spectrum, absolute_QYs.T)

        print(
            f"  Loaded wavelength range: {wavelength.min():.0f}-{wavelength.max():.0f} nm"
        )
        print(f"  Average emission wavelength: {average_emission_wavelength:.1f} nm")
        print(f"  Dye pixel efficiency shape: {dye_pixel_efficiency.shape}")

        return (
            wavelength,
            dye_spectrum,
            absolute_QYs,
            average_emission_wavelength,
            dye_pixel_efficiency,
        )

    def _convert_to_photoelectrons(
        self, bayer_image: np.ndarray, camera_region: Dict[str, Any]
    ) -> np.ndarray:
        """
        Convert simulated camera image to photoelectron counts.

        This mimics the process of reading real TIFF data and converting
        to photoelectron counts for analysis.

        Args:
            bayer_image: Simulated camera image with noise
            camera_region: Camera calibration parameters

        Returns:
            Photoelectron image ready for spot detection
        """
        # Apply inverse camera calibration: (counts - offset) / (gain * rqe)
        photoelectron_image = np.divide(
            np.divide(
                np.subtract(bayer_image, camera_region["offset"]), camera_region["gain"]
            ),
            camera_region["rqe"],
        )

        # Ensure non-negative values
        photoelectron_image[photoelectron_image < 0] = 0

        return photoelectron_image

    def detect_spots(
        self,
        photoelectron_image: np.ndarray,
        camera_region: Dict[str, Any],
        bayer_processing: bool = True,
    ) -> np.ndarray:
        """
        Apply spot detection algorithm to photoelectron image.

        Args:
            photoelectron_image: Image in photoelectron counts
            camera_region: Camera calibration parameters
            bayer_processing: If True, apply Bayer averaging before detection

        Returns:
            Array of detected spot positions [x, y] in pixels
        """
        # Add image as stack with single frame
        image_stack = photoelectron_image[np.newaxis, :, :]

        # Apply spot detection with Bayer processing option
        detected_spots = SD.detect_puncta_in_stack_parallel(
            image_stack,
            variance=camera_region["variance"],
            pfa=self.config.pfa,
            wavelength=self.config.wavelength,
            pixel_size=self.config.pixel_size_nm / 1000,  # Convert to microns
            NA=self.config.NA,
            mf_factor=self.config.mf_factor,
            local_factor=self.config.local_factor,
            bayer_image=bayer_processing,
        )

        # Extract positions (remove frame information)
        if len(detected_spots) > 0 and detected_spots.shape[1] == 3:
            # Format: [x, y, frame] -> extract [x, y]
            spot_positions = detected_spots[:, :2]
        else:
            spot_positions = np.empty((0, 2))

        return spot_positions

    def evaluate_detection_performance(
        self,
        detected_positions: np.ndarray,
        ground_truth_positions: np.ndarray,
        bayer_processing: bool = None,
    ) -> DetectionMetrics:
        """
        Evaluate detection performance by matching detected spots to ground truth.

        Args:
            detected_positions: Array of detected [x, y] positions in pixels
            ground_truth_positions: Array of ground truth [x, y] positions in nm

        Returns:
            DetectionMetrics object with performance statistics
        """
        # Convert ground truth to pixels for comparison
        gt_positions_pixels = ground_truth_positions / self.config.pixel_size_nm
        tolerance_pixels = (
            self.config.detection_tolerance_nm / self.config.pixel_size_nm
        )

        n_ground_truth = len(gt_positions_pixels)
        n_detected = len(detected_positions)

        if n_detected == 0:
            # No detections - all ground truth are false negatives
            return DetectionMetrics(
                true_positives=0,
                false_positives=0,
                false_negatives=n_ground_truth,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                true_positive_rate=0.0,
                false_positive_rate=0.0,
                detection_positions=detected_positions,
                ground_truth_positions=ground_truth_positions,
                bayer_processing=bayer_processing,
            )

        if n_ground_truth == 0:
            # No ground truth - all detections are false positives
            return DetectionMetrics(
                true_positives=0,
                false_positives=n_detected,
                false_negatives=0,
                precision=0.0,
                recall=0.0,  # Undefined, but set to 0
                f1_score=0.0,
                true_positive_rate=0.0,
                false_positive_rate=1.0,
                detection_positions=detected_positions,
                ground_truth_positions=ground_truth_positions,
                bayer_processing=bayer_processing,
            )

        # Calculate distances between all detected and ground truth positions
        distances = cdist(detected_positions, gt_positions_pixels)

        # Find matches using Hungarian algorithm approach
        true_positives = 0
        matched_gt = set()
        matched_det = set()

        # Greedy matching: assign each detection to closest ground truth within tolerance
        for det_idx in range(n_detected):
            min_dist_idx = np.argmin(distances[det_idx, :])
            min_dist = distances[det_idx, min_dist_idx]

            if min_dist <= tolerance_pixels and min_dist_idx not in matched_gt:
                true_positives += 1
                matched_gt.add(min_dist_idx)
                matched_det.add(det_idx)

        false_positives = n_detected - true_positives
        false_negatives = n_ground_truth - true_positives

        # Calculate performance metrics
        precision = true_positives / n_detected if n_detected > 0 else 0.0
        recall = true_positives / n_ground_truth if n_ground_truth > 0 else 0.0
        f1_score = (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        true_positive_rate = recall  # Same as recall
        false_positive_rate = (
            false_positives / (false_positives + true_positives)
            if (false_positives + true_positives) > 0
            else 0.0
        )

        return DetectionMetrics(
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            true_positive_rate=true_positive_rate,
            false_positive_rate=false_positive_rate,
            detection_positions=detected_positions,
            ground_truth_positions=ground_truth_positions,
            bayer_processing=bayer_processing,
        )

    def run_validation_bootstrap(
        self, camera_path: str, save_folder: str
    ) -> pd.DataFrame:
        """
        Run complete validation with bootstrap sampling across camera regions.

        Args:
            camera_path: Path to camera calibration directory
            save_folder: Directory to save results

        Returns:
            DataFrame with detailed results for each bootstrap sample
        """
        print("Starting spot detection validation...")

        # Load camera calibration
        self.load_camera_calibration(camera_path)

        # Generate ground truth grid
        ground_truth_positions = self.generate_grid_positions()
        print(f"Generated {len(ground_truth_positions)} ground truth puncta positions")

        # Prepare results storage
        results = []

        # Bootstrap sampling parameters
        full_height, full_width = self.camera_params["full_dimensions"]
        max_start_row = max(0, full_height - self.config.image_size_pixels)
        max_start_col = max(0, full_width - self.config.image_size_pixels)

        print(f"Running {self.config.n_bootstrap} bootstrap samples...")

        start_time = time.time()

        # Determine which Bayer processing conditions to test
        bayer_conditions = (
            [True, False] if self.config.test_bayer_processing else [True]
        )
        total_samples = self.config.n_bootstrap * len(bayer_conditions)
        sample_count = 0

        for bootstrap_idx in range(self.config.n_bootstrap):
            # Randomly sample camera region
            start_row = np.random.randint(0, max_start_row + 1)
            start_col = np.random.randint(0, max_start_col + 1)

            # Extract camera region
            camera_region = self.extract_camera_region(start_row, start_col)

            # Random photon count within range
            n_photons = np.random.randint(
                self.config.n_photons_range[0], self.config.n_photons_range[1] + 1
            )

            # Simulate camera image once for both processing conditions
            try:
                bayer_image, photoelectron_image = self.simulate_camera_image(
                    ground_truth_positions, camera_region, n_photons
                )
            except Exception as e:
                print(f"Error in bootstrap sample {bootstrap_idx}: {e}")
                continue

            # Test each Bayer processing condition
            for bayer_processing in bayer_conditions:
                try:
                    # Detect spots with current Bayer processing setting
                    detected_positions = self.detect_spots(
                        photoelectron_image, camera_region, bayer_processing
                    )

                    # Evaluate performance
                    metrics = self.evaluate_detection_performance(
                        detected_positions, ground_truth_positions, bayer_processing
                    )

                    # Store results
                    result = {
                        "bootstrap_idx": bootstrap_idx,
                        "bayer_processing": bayer_processing,
                        "camera_start_row": start_row,
                        "camera_start_col": start_col,
                        "n_photons": n_photons,
                        "n_ground_truth": len(ground_truth_positions),
                        "n_detected": len(detected_positions),
                        "true_positives": metrics.true_positives,
                        "false_positives": metrics.false_positives,
                        "false_negatives": metrics.false_negatives,
                        "precision": metrics.precision,
                        "recall": metrics.recall,
                        "f1_score": metrics.f1_score,
                        "true_positive_rate": metrics.true_positive_rate,
                        "false_positive_rate": metrics.false_positive_rate,
                    }
                    results.append(result)

                    sample_count += 1

                    # Progress update
                    if sample_count % 10 == 0:
                        elapsed_time = time.time() - start_time
                        avg_time_per_sample = elapsed_time / sample_count
                        remaining_time = avg_time_per_sample * (
                            total_samples - sample_count
                        )
                        print(
                            f"Completed {sample_count}/{total_samples} samples "
                            f"(ETA: {remaining_time:.1f}s)"
                        )

                except Exception as e:
                    print(
                        f"Error in bootstrap sample {bootstrap_idx} with bayer_processing={bayer_processing}: {e}"
                    )
                    # Store failed result
                    result = {
                        "bootstrap_idx": bootstrap_idx,
                        "bayer_processing": bayer_processing,
                        "camera_start_row": start_row,
                        "camera_start_col": start_col,
                        "n_photons": n_photons,
                        "n_ground_truth": len(ground_truth_positions),
                        "n_detected": 0,
                        "true_positives": 0,
                        "false_positives": 0,
                        "false_negatives": len(ground_truth_positions),
                        "precision": 0.0,
                        "recall": 0.0,
                        "f1_score": 0.0,
                        "true_positive_rate": 0.0,
                        "false_positive_rate": 0.0,
                        "error": str(e),
                    }
                    results.append(result)
                    continue

        # Convert to DataFrame and save
        results_df = pd.DataFrame(results)

        # Save detailed results
        os.makedirs(save_folder, exist_ok=True)
        results_path = os.path.join(
            save_folder, "spot_detection_validation_results.csv"
        )
        results_df.to_csv(results_path, index=False)
        print(f"Detailed results saved to: {results_path}")

        # Generate and save summary statistics
        self._generate_summary_report(results_df, save_folder)

        print(f"Validation completed in {time.time() - start_time:.1f} seconds")
        return results_df

    def _generate_summary_report(self, results_df: pd.DataFrame, save_folder: str):
        """Generate summary statistics and visualisation."""

        # Check if we tested both Bayer processing conditions
        has_bayer_comparison = (
            "bayer_processing" in results_df.columns
            and len(results_df["bayer_processing"].unique()) > 1
        )

        if has_bayer_comparison:
            # Calculate summary statistics for both conditions
            bayer_true = results_df[results_df["bayer_processing"] == True]
            bayer_false = results_df[results_df["bayer_processing"] == False]

            summary_stats = {
                "total_samples": len(results_df),
                "successful_samples": len(results_df[results_df["n_detected"].notna()]),
                "bayer_processing_tested": True,
                # Overall statistics
                "overall_mean_precision": results_df["precision"].mean(),
                "overall_std_precision": results_df["precision"].std(),
                "overall_mean_recall": results_df["recall"].mean(),
                "overall_std_recall": results_df["recall"].std(),
                "overall_mean_f1_score": results_df["f1_score"].mean(),
                "overall_std_f1_score": results_df["f1_score"].std(),
                "overall_mean_false_positive_rate": results_df[
                    "false_positive_rate"
                ].mean(),
                "overall_std_false_positive_rate": results_df[
                    "false_positive_rate"
                ].std(),
                # Bayer averaging (bayer_processing=True) statistics
                "bayer_true_mean_precision": bayer_true["precision"].mean(),
                "bayer_true_std_precision": bayer_true["precision"].std(),
                "bayer_true_mean_recall": bayer_true["recall"].mean(),
                "bayer_true_std_recall": bayer_true["recall"].std(),
                "bayer_true_mean_f1_score": bayer_true["f1_score"].mean(),
                "bayer_true_std_f1_score": bayer_true["f1_score"].std(),
                "bayer_true_mean_false_positive_rate": bayer_true[
                    "false_positive_rate"
                ].mean(),
                "bayer_true_std_false_positive_rate": bayer_true[
                    "false_positive_rate"
                ].std(),
                "bayer_true_median_false_positives": bayer_true[
                    "false_positives"
                ].median(),
                "bayer_true_median_false_negatives": bayer_true[
                    "false_negatives"
                ].median(),
                # No Bayer averaging (bayer_processing=False) statistics
                "bayer_false_mean_precision": bayer_false["precision"].mean(),
                "bayer_false_std_precision": bayer_false["precision"].std(),
                "bayer_false_mean_recall": bayer_false["recall"].mean(),
                "bayer_false_std_recall": bayer_false["recall"].std(),
                "bayer_false_mean_f1_score": bayer_false["f1_score"].mean(),
                "bayer_false_std_f1_score": bayer_false["f1_score"].std(),
                "bayer_false_mean_false_positive_rate": bayer_false[
                    "false_positive_rate"
                ].mean(),
                "bayer_false_std_false_positive_rate": bayer_false[
                    "false_positive_rate"
                ].std(),
                "bayer_false_median_false_positives": bayer_false[
                    "false_positives"
                ].median(),
                "bayer_false_median_false_negatives": bayer_false[
                    "false_negatives"
                ].median(),
            }
        else:
            # Calculate summary statistics (single condition)
            summary_stats = {
                "total_samples": len(results_df),
                "successful_samples": len(results_df[results_df["n_detected"].notna()]),
                "bayer_processing_tested": False,
                "mean_precision": results_df["precision"].mean(),
                "std_precision": results_df["precision"].std(),
                "mean_recall": results_df["recall"].mean(),
                "std_recall": results_df["recall"].std(),
                "mean_f1_score": results_df["f1_score"].mean(),
                "std_f1_score": results_df["f1_score"].std(),
                "mean_false_positive_rate": results_df["false_positive_rate"].mean(),
                "std_false_positive_rate": results_df["false_positive_rate"].std(),
                "median_false_positives": results_df["false_positives"].median(),
                "median_false_negatives": results_df["false_negatives"].median(),
            }

        # Save summary statistics
        summary_df = pd.DataFrame([summary_stats])
        summary_path = os.path.join(
            save_folder, "spot_detection_validation_summary.csv"
        )
        summary_df.to_csv(summary_path, index=False)

        # Print key results
        print("\n" + "=" * 70)
        print("SPOT DETECTION VALIDATION SUMMARY")
        print("=" * 70)
        print(f"Total bootstrap samples: {summary_stats['total_samples']}")
        print(f"Successful samples: {summary_stats['successful_samples']}")

        if has_bayer_comparison:
            print(f"Bayer processing comparison: ENABLED")
            print(f"")
            print(f"📊 BAYER PROCESSING COMPARISON:")
            print(f"")
            print(f"  🔶 WITH Bayer Averaging (bayer_image=True):")
            print(
                f"    Precision: {summary_stats['bayer_true_mean_precision']:.3f} ± {summary_stats['bayer_true_std_precision']:.3f}"
            )
            print(
                f"    Recall: {summary_stats['bayer_true_mean_recall']:.3f} ± {summary_stats['bayer_true_std_recall']:.3f}"
            )
            print(
                f"    F1-Score: {summary_stats['bayer_true_mean_f1_score']:.3f} ± {summary_stats['bayer_true_std_f1_score']:.3f}"
            )
            print(
                f"    False Positive Rate: {summary_stats['bayer_true_mean_false_positive_rate']:.4f} ± {summary_stats['bayer_true_std_false_positive_rate']:.4f}"
            )
            print(
                f"    Median False Positives: {summary_stats['bayer_true_median_false_positives']:.1f}"
            )
            print(
                f"    Median False Negatives: {summary_stats['bayer_true_median_false_negatives']:.1f}"
            )
            print(f"")
            print(f"  🔷 WITHOUT Bayer Averaging (bayer_image=False):")
            print(
                f"    Precision: {summary_stats['bayer_false_mean_precision']:.3f} ± {summary_stats['bayer_false_std_precision']:.3f}"
            )
            print(
                f"    Recall: {summary_stats['bayer_false_mean_recall']:.3f} ± {summary_stats['bayer_false_std_recall']:.3f}"
            )
            print(
                f"    F1-Score: {summary_stats['bayer_false_mean_f1_score']:.3f} ± {summary_stats['bayer_false_std_f1_score']:.3f}"
            )
            print(
                f"    False Positive Rate: {summary_stats['bayer_false_mean_false_positive_rate']:.4f} ± {summary_stats['bayer_false_std_false_positive_rate']:.4f}"
            )
            print(
                f"    Median False Positives: {summary_stats['bayer_false_median_false_positives']:.1f}"
            )
            print(
                f"    Median False Negatives: {summary_stats['bayer_false_median_false_negatives']:.1f}"
            )
            print(f"")
            print(f"  📈 PERFORMANCE DIFFERENCES:")
            precision_diff = (
                summary_stats["bayer_true_mean_precision"]
                - summary_stats["bayer_false_mean_precision"]
            )
            recall_diff = (
                summary_stats["bayer_true_mean_recall"]
                - summary_stats["bayer_false_mean_recall"]
            )
            fp_rate_diff = (
                summary_stats["bayer_true_mean_false_positive_rate"]
                - summary_stats["bayer_false_mean_false_positive_rate"]
            )
            print(f"    Precision difference (True - False): {precision_diff:+.3f}")
            print(f"    Recall difference (True - False): {recall_diff:+.3f}")
            print(
                f"    False Positive Rate difference (True - False): {fp_rate_diff:+.4f}"
            )
            if fp_rate_diff < 0:
                print(
                    f"    → Bayer averaging REDUCES false positives by {abs(fp_rate_diff):.4f}"
                )
            else:
                print(
                    f"    → Bayer averaging INCREASES false positives by {fp_rate_diff:.4f}"
                )
        else:
            print(f"")
            print(f"Performance Metrics (mean ± std):")
            print(
                f"  Precision: {summary_stats['mean_precision']:.3f} ± {summary_stats['std_precision']:.3f}"
            )
            print(
                f"  Recall: {summary_stats['mean_recall']:.3f} ± {summary_stats['std_recall']:.3f}"
            )
            print(
                f"  F1-Score: {summary_stats['mean_f1_score']:.3f} ± {summary_stats['std_f1_score']:.3f}"
            )
            print(f"")
            print(f"Error Rates:")
            print(
                f"  False Positive Rate: {summary_stats['mean_false_positive_rate']:.4f} ± {summary_stats['std_false_positive_rate']:.4f}"
            )
            print(
                f"  Median False Positives per Image: {summary_stats['median_false_positives']:.1f}"
            )
            print(
                f"  Median False Negatives per Image: {summary_stats['median_false_negatives']:.1f}"
            )
        print("=" * 70)

        # Create visualisation plots
        self._create_validation_plots(results_df, save_folder)

    def _create_validation_plots(self, results_df: pd.DataFrame, save_folder: str):
        """Create validation performance plots."""

        # Check if we have Bayer processing comparison
        has_bayer_comparison = (
            "bayer_processing" in results_df.columns
            and len(results_df["bayer_processing"].unique()) > 1
        )

        if has_bayer_comparison:
            # Create comparison plots
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))

            bayer_true = results_df[results_df["bayer_processing"] == True]
            bayer_false = results_df[results_df["bayer_processing"] == False]

            # Plot 1: Precision vs Recall scatter comparison
            axes[0, 0].scatter(
                bayer_true["recall"],
                bayer_true["precision"],
                alpha=0.6,
                label="Bayer Averaging",
                color="orange",
                s=30,
            )
            axes[0, 0].scatter(
                bayer_false["recall"],
                bayer_false["precision"],
                alpha=0.6,
                label="No Bayer Averaging",
                color="blue",
                s=30,
            )
            axes[0, 0].set_xlabel("Recall (True Positive Rate)")
            axes[0, 0].set_ylabel("Precision")
            axes[0, 0].set_title("Precision vs Recall Comparison")
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # Plot 2: False Positive Rate comparison
            axes[0, 1].hist(
                [bayer_true["false_positive_rate"], bayer_false["false_positive_rate"]],
                bins=15,
                alpha=0.7,
                label=["Bayer Averaging", "No Bayer Averaging"],
                color=["orange", "blue"],
                edgecolor="black",
            )
            axes[0, 1].set_xlabel("False Positive Rate")
            axes[0, 1].set_ylabel("Frequency")
            axes[0, 1].set_title("False Positive Rate Comparison")
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            # Plot 3: Detection counts vs photons comparison
            axes[0, 2].scatter(
                bayer_true["n_photons"],
                bayer_true["n_detected"],
                alpha=0.6,
                label="Bayer Averaging",
                color="orange",
                s=30,
            )
            axes[0, 2].scatter(
                bayer_false["n_photons"],
                bayer_false["n_detected"],
                alpha=0.6,
                label="No Bayer Averaging",
                color="blue",
                s=30,
            )
            axes[0, 2].axhline(
                y=100, color="red", linestyle="--", label="Ground Truth (100)"
            )
            axes[0, 2].set_xlabel("Number of Photons per Punctum")
            axes[0, 2].set_ylabel("Number of Detected Spots")
            axes[0, 2].set_title("Detection Count vs Photon Number")
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)

            # Plot 4: F1-Score comparison
            axes[1, 0].hist(
                [bayer_true["f1_score"], bayer_false["f1_score"]],
                bins=15,
                alpha=0.7,
                label=["Bayer Averaging", "No Bayer Averaging"],
                color=["orange", "blue"],
                edgecolor="black",
            )
            axes[1, 0].set_xlabel("F1-Score")
            axes[1, 0].set_ylabel("Frequency")
            axes[1, 0].set_title("F1-Score Comparison")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            # Plot 5: Box plot comparison for key metrics
            import numpy as np

            metrics_data = [
                [bayer_true["precision"], bayer_false["precision"]],
                [bayer_true["recall"], bayer_false["recall"]],
                [bayer_true["false_positive_rate"], bayer_false["false_positive_rate"]],
            ]
            metric_names = ["Precision", "Recall", "FP Rate"]

            bp = axes[1, 1].boxplot(
                metrics_data[0], labels=["Bayer Avg", "No Bayer"], patch_artist=True
            )
            bp["boxes"][0].set_facecolor("orange")
            bp["boxes"][1].set_facecolor("blue")
            axes[1, 1].set_ylabel("Precision")
            axes[1, 1].set_title("Precision Box Plot Comparison")
            axes[1, 1].grid(True, alpha=0.3)

            # Plot 6: Performance difference scatter
            fp_diff = (
                bayer_true["false_positive_rate"].values
                - bayer_false["false_positive_rate"].values
            )
            precision_diff = (
                bayer_true["precision"].values - bayer_false["precision"].values
            )
            axes[1, 2].scatter(fp_diff, precision_diff, alpha=0.6, color="green", s=30)
            axes[1, 2].axhline(y=0, color="black", linestyle="--", alpha=0.5)
            axes[1, 2].axvline(x=0, color="black", linestyle="--", alpha=0.5)
            axes[1, 2].set_xlabel("FP Rate Difference (Bayer - No Bayer)")
            axes[1, 2].set_ylabel("Precision Difference (Bayer - No Bayer)")
            axes[1, 2].set_title("Performance Trade-off Analysis")
            axes[1, 2].grid(True, alpha=0.3)

        else:
            # Create standard plots for single condition
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            # Plot 1: Precision vs Recall scatter
            axes[0, 0].scatter(results_df["recall"], results_df["precision"], alpha=0.6)
            axes[0, 0].set_xlabel("Recall (True Positive Rate)")
            axes[0, 0].set_ylabel("Precision")
            axes[0, 0].set_title("Precision vs Recall")
            axes[0, 0].grid(True, alpha=0.3)

            # Plot 2: False Positive Rate histogram
            axes[0, 1].hist(
                results_df["false_positive_rate"], bins=20, alpha=0.7, edgecolor="black"
            )
            axes[0, 1].set_xlabel("False Positive Rate")
            axes[0, 1].set_ylabel("Frequency")
            axes[0, 1].set_title("False Positive Rate Distribution")
            axes[0, 1].grid(True, alpha=0.3)

            # Plot 3: Detection counts vs photons
            axes[1, 0].scatter(
                results_df["n_photons"],
                results_df["n_detected"],
                alpha=0.6,
                label="Detected",
            )
            axes[1, 0].axhline(
                y=100, color="red", linestyle="--", label="Ground Truth (100)"
            )
            axes[1, 0].set_xlabel("Number of Photons per Punctum")
            axes[1, 0].set_ylabel("Number of Detected Spots")
            axes[1, 0].set_title("Detection Count vs Photon Number")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            # Plot 4: F1-Score histogram
            axes[1, 1].hist(
                results_df["f1_score"], bins=20, alpha=0.7, edgecolor="black"
            )
            axes[1, 1].set_xlabel("F1-Score")
            axes[1, 1].set_ylabel("Frequency")
            axes[1, 1].set_title("F1-Score Distribution")
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plot_filename = (
            "spot_detection_bayer_comparison_plots.png"
            if has_bayer_comparison
            else "spot_detection_validation_plots.png"
        )
        plot_path = os.path.join(save_folder, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Validation plots saved to: {plot_path}")


def main():
    """Main function to run spot detection validation."""

    # Configuration
    config = ValidationConfig(
        grid_size=10,
        grid_spacing_microns=1.0,
        image_size_pixels=200,
        n_photons_range=(1000, 5000),
        n_bootstrap=500,
        pfa=1e-4,
        detection_tolerance_nm=150.0,
        test_bayer_processing=True,  # Enable Bayer processing comparison
    )

    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    camera_path = os.path.join(base_dir, "Camera_Calibrations", "Ximea_Camera")
    save_folder = os.path.join(base_dir, "validation_results", "spot_detection")

    # Create validator and run
    validator = SpotDetectionValidator(config)
    results = validator.run_validation_bootstrap(camera_path, save_folder)

    return results


if __name__ == "__main__":
    results = main()
