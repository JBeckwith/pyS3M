#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation test comparing Bayer-aware spot detection vs demosaic-then-detect.

Simulates ground truth spot positions using Multicolour_Simulation_Functions,
then compares two detection pipelines:
1. Old pipeline: Demosaic RGB → spot detection
2. New pipeline: Raw Bayer → per-channel detection

Metrics: Precision, Recall, F1 score, position accuracy

Created: December 19, 2025
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
from scipy.spatial import distance_matrix

# Add src to path
module_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, module_dir)

import Multicolour_Simulation_Functions as MSF
import SpotDetectionFunctions
import BayerSpotDetection
import MaskFunctions
import sCMOSFunctions
import SpectralFunctions
import IOFunctions


class SpotDetectionValidator:
    """Compare Bayer-aware vs demosaic spot detection on synthetic data."""

    def __init__(self, image_size=(512, 512), pixel_size=69, NA=1.49):
        """Initialize validator.

        Args:
            image_size: Tuple of (height, width) in pixels
            pixel_size: Pixel size in nm
            NA: Numerical aperture
        """
        self.image_size = image_size
        self.pixel_size = pixel_size
        self.NA = NA

        # Initialize modules
        self.scmos = sCMOSFunctions.sCMOS_Functions()
        self.spectral = SpectralFunctions.Spectral_Functions()
        self.io = IOFunctions.IO_Functions()
        self.mask_gen = MaskFunctions.Mask_Functions()
        self.spot_detector = SpotDetectionFunctions.SpotDetection_Functions()

        # Load camera calibration
        print("Loading camera calibration...")
        self.camera_params = self._load_camera_calibration()

    def _load_camera_calibration(self) -> Dict:
        """Load camera calibration for simulation."""
        # Use default camera calibration file
        camera_cal_file = os.path.join(
            module_dir, '..', 'Camera_Calibration_Data',
            'BayerCMOSCamera_21-09-15.mat'
        )

        if not os.path.exists(camera_cal_file):
            raise FileNotFoundError(
                f"Camera calibration file not found: {camera_cal_file}\n"
                f"Please provide a valid camera calibration file."
            )

        camera_params = self.io.matfile_to_dict(camera_cal_file)

        # Get mosaic unit (RGGB pattern)
        mosaic_unit = np.array([['R', 'G'], ['G', 'B']])

        # Generate masks
        H, W = self.image_size
        masks = self.mask_gen.get_masks(H, W, mosaic_unit)
        camera_params['masks'] = masks

        return camera_params

    def generate_random_spots(
        self,
        n_spots: int = 100,
        n_frames: int = 10,
        border: int = 50,
        wavelength: float = 0.580,  # Green emission
        photons_mean: int = 1000,
        photons_std: int = 200
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate random spot positions and create simulated images.

        Args:
            n_spots: Number of spots per frame
            n_frames: Number of frames
            border: Minimum distance from image edge (pixels)
            wavelength: Emission wavelength in microns
            photons_mean: Mean photon count per spot
            photons_std: Std deviation of photon count

        Returns:
            ground_truth: Array of shape (n_total_spots, 3) with [y, x, frame]
            bayer_stack: Simulated Bayer image stack (n_frames, H, W)
            rgb_stack: Demosaiced RGB image stack (n_frames, H, W, 3)
        """
        print(f"\nGenerating {n_spots} spots per frame across {n_frames} frames...")

        H, W = self.image_size

        # Generate random positions for each frame
        ground_truth_list = []
        x0y0_all_frames = []
        photons_all_frames = []

        for frame_idx in range(n_frames):
            # Random positions (avoiding borders)
            x_pos = np.random.uniform(border, W - border, n_spots)
            y_pos = np.random.uniform(border, H - border, n_spots)

            # Store ground truth [y, x, frame]
            for x, y in zip(x_pos, y_pos):
                ground_truth_list.append([y, x, frame_idx])

            # For simulation: x0y0 format (nm coordinates)
            x0y0_frame = np.column_stack([x_pos * self.pixel_size, y_pos * self.pixel_size])
            x0y0_all_frames.append(x0y0_frame)

            # Random photon counts
            photons = np.random.normal(photons_mean, photons_std, n_spots).astype(int)
            photons = np.maximum(photons, 100)  # Minimum 100 photons
            photons_all_frames.append(photons)

        ground_truth = np.array(ground_truth_list)

        # Prepare data for simulation
        # Stack all frames into single arrays
        x0y0_stacked = np.vstack(x0y0_all_frames)
        photons_stacked = np.hstack(photons_all_frames)

        # Create frame indices for simulation
        frame_indices = np.repeat(np.arange(n_frames), n_spots)

        # Prepare inputs for gen_camera_image_stack
        # We'll simulate as a single "dye" with all spots
        x0y0_dict = {'dye1': x0y0_stacked}
        photons_dict = {'dye1': photons_stacked}

        # Get spectral data for wavelength
        wavelengths_nm = np.arange(400, 801, 1)  # 400-800 nm

        # Create emission spectrum (Gaussian centered at wavelength)
        emission = np.exp(-0.5 * ((wavelengths_nm - wavelength * 1000) / 30) ** 2)
        emission = emission / np.sum(emission)

        # Get pixel efficiency
        pixel_QYs = self.camera_params['pixel_QYs']
        dye_pixel_efficiency = np.dot(emission, pixel_QYs)

        print("Simulating Bayer images...")

        # Initialize simulator
        simulator = MSF.MultiC_Sim_Funcs()

        # Generate images
        bayer_stack, smoothed_stack, _ = simulator.gen_camera_image_stack(
            camera_calibration=self.camera_params,
            wavelength=wavelengths_nm,
            average_emission_wavelengths=wavelength,
            dye_pixel_efficiency=dye_pixel_efficiency,
            n_photons=photons_dict,
            x0y0=x0y0_dict,
            smoothing_function=self.scmos.gaussian_filter_stack,
            background_photons=40,
            background_colour=[1, 1, 1],
            NA=self.NA,
            pixel_size=self.pixel_size,
            return_normal_image=False,
            use_vectorized_photoelectrons=True
        )

        # Demosaic for traditional pipeline
        print("Demosaicing images...")
        rgb_stack = self._demosaic_stack(bayer_stack)

        return ground_truth, bayer_stack, rgb_stack

    def _demosaic_stack(self, bayer_stack: np.ndarray) -> np.ndarray:
        """Demosaic Bayer stack to RGB using simple bilinear interpolation.

        Args:
            bayer_stack: Bayer image stack (n_frames, H, W)

        Returns:
            rgb_stack: RGB image stack (n_frames, H, W, 3)
        """
        from scipy.ndimage import convolve

        n_frames = bayer_stack.shape[0]
        H, W = bayer_stack.shape[1:]
        rgb_stack = np.zeros((n_frames, H, W, 3), dtype=np.float32)

        # Simple bilinear demosaicing kernels (RGGB pattern)
        # R channel kernel (top-left corners)
        kernel_r = np.array([[1, 2, 1],
                             [2, 4, 2],
                             [1, 2, 1]], dtype=np.float32) / 4

        # G channel kernel (checkerboard with 2× density)
        kernel_g = np.array([[0, 1, 0],
                             [1, 4, 1],
                             [0, 1, 0]], dtype=np.float32) / 4

        # B channel kernel (bottom-right corners)
        kernel_b = kernel_r.copy()

        for i in range(n_frames):
            img = bayer_stack[i].astype(np.float32)

            # Extract channels from Bayer pattern (RGGB)
            r_raw = np.zeros_like(img)
            g_raw = np.zeros_like(img)
            b_raw = np.zeros_like(img)

            # RGGB pattern:
            # R at [0::2, 0::2]
            # G at [0::2, 1::2] and [1::2, 0::2]
            # B at [1::2, 1::2]
            r_raw[0::2, 0::2] = img[0::2, 0::2]
            g_raw[0::2, 1::2] = img[0::2, 1::2]
            g_raw[1::2, 0::2] = img[1::2, 0::2]
            b_raw[1::2, 1::2] = img[1::2, 1::2]

            # Interpolate each channel
            rgb_stack[i, :, :, 0] = convolve(r_raw, kernel_r, mode='reflect')
            rgb_stack[i, :, :, 1] = convolve(g_raw, kernel_g, mode='reflect')
            rgb_stack[i, :, :, 2] = convolve(b_raw, kernel_b, mode='reflect')

        return rgb_stack

    def detect_bayer_aware(
        self,
        bayer_stack: np.ndarray,
        pfa: float = 1e-4,
        sigma: float = 1.5,
        wavelength: float = 0.580,
        **kwargs
    ) -> np.ndarray:
        """Run Bayer-aware per-channel detection.

        Args:
            bayer_stack: Raw Bayer stack (n_frames, H, W)
            pfa: Probability of false alarm
            sigma: Threshold multiplier
            wavelength: Emission wavelength (microns)
            **kwargs: Additional params for detection

        Returns:
            detections: Array of [y, x, frame, ...]
        """
        print("\n=== Bayer-Aware Detection (Raw Per-Channel) ===")

        detections, metadata = BayerSpotDetection.detect_spots_bayer_multichannel(
            bayer_stack,
            spot_detector=self.spot_detector,
            pattern='RGGB',
            pfa=pfa,
            sigma=sigma,
            variance=None,
            channels=['red', 'green', 'blue'],
            merge_distance=2.0,
            wavelength=wavelength,
            pixel_size=self.pixel_size / 1000,  # Convert nm to microns
            NA=self.NA,
            **kwargs
        )

        print(f"\nBayer-aware detection: {len(detections)} spots")
        print(f"  Per-channel counts: {metadata['n_detections']}")
        print(f"  Duplicates removed: {metadata['n_duplicates_removed']}")

        return detections

    def detect_demosaic_then_detect(
        self,
        rgb_stack: np.ndarray,
        pfa: float = 1e-4,
        sigma: float = 1.5,
        wavelength: float = 0.580,
        **kwargs
    ) -> np.ndarray:
        """Run traditional demosaic-then-detect pipeline.

        Args:
            rgb_stack: Demosaiced RGB stack (n_frames, H, W, 3)
            pfa: Probability of false alarm
            sigma: Threshold multiplier
            wavelength: Emission wavelength (microns)
            **kwargs: Additional params for detection

        Returns:
            detections: Array of [y, x, frame, ...]
        """
        print("\n=== Traditional Detection (Demosaic → Detect) ===")

        # Convert RGB to grayscale (simple average)
        gray_stack = np.mean(rgb_stack, axis=3).astype(np.float32)

        print(f"Detecting on demosaiced grayscale images (shape: {gray_stack.shape})...")

        detections = self.spot_detector.detect_puncta_in_stack_parallel(
            gray_stack,
            variance=None,
            pfa=pfa,
            sigma=sigma,
            wavelength=wavelength,
            pixel_size=self.pixel_size / 1000,  # Convert nm to microns
            NA=self.NA,
            **kwargs
        )

        print(f"\nDemosaic-then-detect: {len(detections)} spots")

        return detections

    def evaluate_detections(
        self,
        ground_truth: np.ndarray,
        detections: np.ndarray,
        match_threshold: float = 2.0,
        method_name: str = "Method"
    ) -> Dict:
        """Evaluate detection performance against ground truth.

        Args:
            ground_truth: Ground truth positions [y, x, frame]
            detections: Detected positions [y, x, frame, ...]
            match_threshold: Distance threshold for matching (pixels)
            method_name: Name for reporting

        Returns:
            metrics: Dict with precision, recall, F1, etc.
        """
        if len(detections) == 0:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'n_true_positives': 0,
                'n_false_positives': 0,
                'n_false_negatives': len(ground_truth),
                'mean_position_error': np.nan,
                'std_position_error': np.nan
            }

        # Match detections to ground truth frame-by-frame
        n_true_positives = 0
        position_errors = []
        matched_gt_indices = set()
        matched_det_indices = set()

        frames_gt = np.unique(ground_truth[:, 2])

        for frame in frames_gt:
            gt_frame = ground_truth[ground_truth[:, 2] == frame]
            det_frame = detections[detections[:, 2] == frame]

            if len(det_frame) == 0:
                continue

            # Compute distance matrix
            gt_coords = gt_frame[:, :2]  # [y, x]
            det_coords = det_frame[:, :2]

            dist_mat = distance_matrix(gt_coords, det_coords)

            # Greedy matching: find closest pairs under threshold
            while True:
                if dist_mat.size == 0:
                    break

                min_dist = np.min(dist_mat)
                if min_dist > match_threshold:
                    break

                # Find indices of minimum distance
                gt_idx, det_idx = np.unravel_index(np.argmin(dist_mat), dist_mat.shape)

                # Record match
                n_true_positives += 1
                position_errors.append(min_dist)

                # Mark as matched (use frame-specific indices)
                gt_global_idx = np.where(
                    (ground_truth[:, 2] == frame) &
                    (ground_truth[:, 0] == gt_coords[gt_idx, 0]) &
                    (ground_truth[:, 1] == gt_coords[gt_idx, 1])
                )[0][0]
                det_global_idx = np.where(
                    (detections[:, 2] == frame) &
                    (detections[:, 0] == det_coords[det_idx, 0]) &
                    (detections[:, 1] == det_coords[det_idx, 1])
                )[0][0]

                matched_gt_indices.add(gt_global_idx)
                matched_det_indices.add(det_global_idx)

                # Remove matched pair from distance matrix
                dist_mat = np.delete(dist_mat, gt_idx, axis=0)
                dist_mat = np.delete(dist_mat, det_idx, axis=1)
                gt_coords = np.delete(gt_coords, gt_idx, axis=0)
                det_coords = np.delete(det_coords, det_idx, axis=0)

        n_false_positives = len(detections) - n_true_positives
        n_false_negatives = len(ground_truth) - n_true_positives

        precision = n_true_positives / len(detections) if len(detections) > 0 else 0
        recall = n_true_positives / len(ground_truth) if len(ground_truth) > 0 else 0
        f1_score = (2 * precision * recall / (precision + recall)
                    if (precision + recall) > 0 else 0)

        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'n_true_positives': n_true_positives,
            'n_false_positives': n_false_positives,
            'n_false_negatives': n_false_negatives,
            'mean_position_error': np.mean(position_errors) if position_errors else np.nan,
            'std_position_error': np.std(position_errors) if position_errors else np.nan
        }

        print(f"\n{'='*60}")
        print(f"{method_name} Performance:")
        print(f"{'='*60}")
        print(f"True Positives:  {n_true_positives:5d}")
        print(f"False Positives: {n_false_positives:5d}")
        print(f"False Negatives: {n_false_negatives:5d}")
        print(f"Precision:       {precision:5.1%}")
        print(f"Recall:          {recall:5.1%}")
        print(f"F1 Score:        {f1_score:5.1%}")
        if not np.isnan(metrics['mean_position_error']):
            print(f"Position Error:  {metrics['mean_position_error']:.2f} ± {metrics['std_position_error']:.2f} px")
        print(f"{'='*60}")

        return metrics

    def plot_comparison(
        self,
        ground_truth: np.ndarray,
        bayer_detections: np.ndarray,
        demosaic_detections: np.ndarray,
        bayer_stack: np.ndarray,
        frame_idx: int = 0,
        save_path: str = None
    ):
        """Plot visual comparison of detection methods.

        Args:
            ground_truth: Ground truth positions
            bayer_detections: Bayer-aware detections
            demosaic_detections: Demosaic detections
            bayer_stack: Raw Bayer images
            frame_idx: Frame to visualize
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Get data for this frame
        gt_frame = ground_truth[ground_truth[:, 2] == frame_idx]
        bayer_det_frame = bayer_detections[bayer_detections[:, 2] == frame_idx]
        demosaic_det_frame = demosaic_detections[demosaic_detections[:, 2] == frame_idx]

        # Plot ground truth
        ax = axes[0]
        ax.imshow(bayer_stack[frame_idx], cmap='gray', vmin=0, vmax=200)
        ax.scatter(gt_frame[:, 1], gt_frame[:, 0],
                  s=100, facecolors='none', edgecolors='lime', linewidths=2,
                  label=f'Ground Truth ({len(gt_frame)})')
        ax.set_title('Ground Truth', fontsize=14, fontweight='bold')
        ax.legend()
        ax.axis('off')

        # Plot Bayer-aware detections
        ax = axes[1]
        ax.imshow(bayer_stack[frame_idx], cmap='gray', vmin=0, vmax=200)
        ax.scatter(gt_frame[:, 1], gt_frame[:, 0],
                  s=100, facecolors='none', edgecolors='lime', linewidths=1.5,
                  alpha=0.5, label='Ground Truth')
        ax.scatter(bayer_det_frame[:, 1], bayer_det_frame[:, 0],
                  s=60, c='red', marker='x', linewidths=2,
                  label=f'Detected ({len(bayer_det_frame)})')
        ax.set_title('Bayer-Aware Detection', fontsize=14, fontweight='bold')
        ax.legend()
        ax.axis('off')

        # Plot demosaic detections
        ax = axes[2]
        ax.imshow(bayer_stack[frame_idx], cmap='gray', vmin=0, vmax=200)
        ax.scatter(gt_frame[:, 1], gt_frame[:, 0],
                  s=100, facecolors='none', edgecolors='lime', linewidths=1.5,
                  alpha=0.5, label='Ground Truth')
        ax.scatter(demosaic_det_frame[:, 1], demosaic_det_frame[:, 0],
                  s=60, c='blue', marker='+', linewidths=2,
                  label=f'Detected ({len(demosaic_det_frame)})')
        ax.set_title('Demosaic-Then-Detect', fontsize=14, fontweight='bold')
        ax.legend()
        ax.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nSaved comparison plot to: {save_path}")

        plt.show()

    def run_comparison(
        self,
        n_spots: int = 100,
        n_frames: int = 10,
        photons_mean: int = 1000,
        plot_frame: int = 0,
        save_plot: bool = True
    ) -> Dict:
        """Run full comparison between detection methods.

        Args:
            n_spots: Number of spots per frame
            n_frames: Number of frames to simulate
            photons_mean: Mean photon count
            plot_frame: Frame index to plot
            save_plot: Whether to save comparison plot

        Returns:
            results: Dict with metrics for both methods
        """
        # Generate synthetic data
        ground_truth, bayer_stack, rgb_stack = self.generate_random_spots(
            n_spots=n_spots,
            n_frames=n_frames,
            photons_mean=photons_mean
        )

        # Run both detection methods
        bayer_detections = self.detect_bayer_aware(bayer_stack)
        demosaic_detections = self.detect_demosaic_then_detect(rgb_stack)

        # Evaluate both methods
        bayer_metrics = self.evaluate_detections(
            ground_truth, bayer_detections,
            method_name="Bayer-Aware Detection"
        )

        demosaic_metrics = self.evaluate_detections(
            ground_truth, demosaic_detections,
            method_name="Demosaic-Then-Detect"
        )

        # Plot comparison
        if save_plot:
            save_path = os.path.join(
                os.path.dirname(__file__),
                'bayer_detection_comparison.png'
            )
        else:
            save_path = None

        self.plot_comparison(
            ground_truth, bayer_detections, demosaic_detections,
            bayer_stack, frame_idx=plot_frame, save_path=save_path
        )

        # Summary comparison
        print("\n" + "="*60)
        print("SUMMARY COMPARISON")
        print("="*60)
        print(f"{'Metric':<25} {'Bayer-Aware':>15} {'Demosaic':>15}")
        print("-"*60)
        print(f"{'Precision':<25} {bayer_metrics['precision']:>14.1%} {demosaic_metrics['precision']:>15.1%}")
        print(f"{'Recall':<25} {bayer_metrics['recall']:>14.1%} {demosaic_metrics['recall']:>15.1%}")
        print(f"{'F1 Score':<25} {bayer_metrics['f1_score']:>14.1%} {demosaic_metrics['f1_score']:>15.1%}")
        print(f"{'Position Error (px)':<25} {bayer_metrics['mean_position_error']:>14.2f} {demosaic_metrics['mean_position_error']:>15.2f}")
        print("="*60)

        # Calculate improvement
        if demosaic_metrics['f1_score'] > 0:
            f1_improvement = ((bayer_metrics['f1_score'] - demosaic_metrics['f1_score']) /
                             demosaic_metrics['f1_score'] * 100)
            print(f"\nBayer-aware F1 improvement: {f1_improvement:+.1f}%")

        return {
            'bayer_aware': bayer_metrics,
            'demosaic': demosaic_metrics,
            'ground_truth': ground_truth,
            'bayer_detections': bayer_detections,
            'demosaic_detections': demosaic_detections
        }


if __name__ == '__main__':
    print("="*60)
    print("Bayer Spot Detection Validation")
    print("="*60)

    # Initialize validator
    validator = SpotDetectionValidator(image_size=(512, 512))

    # Run comparison
    results = validator.run_comparison(
        n_spots=100,
        n_frames=10,
        photons_mean=1000,
        plot_frame=0,
        save_plot=True
    )

    print("\nValidation complete!")
