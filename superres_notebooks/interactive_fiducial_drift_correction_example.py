#!/usr/bin/env python3
"""
Interactive Fiducial Drift Correction Example Script

This script demonstrates interactive fiducial marker selection and drift correction
using the new DriftCorrectionFunctions.py module. It provides both automatic and
manual fiducial selection workflows.

Fiducial-based drift correction is ideal for:
- Experiments with gold nanoparticles, fluorescent beads, or other stationary markers
- High precision drift correction requirements
- Data where correlation-based methods fail due to sparse labeling
- Multi-color experiments requiring precise registration

Features:
- Interactive plotting for fiducial visualization and selection
- Automatic fiducial detection with parameter tuning
- Manual fiducial picking with visual confirmation
- Quality assessment and comparison tools
- Comprehensive result export and analysis

Author: Claude Code Assistant
Created: September 3, 2025
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import warnings
from typing import Tuple, Dict, Any, List, Optional
from dataclasses import dataclass

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Try to enable interactive plotting
try:
    import matplotlib
    # Try to use interactive backend
    matplotlib.use('TkAgg')  # or 'Qt5Agg'
    interactive_available = True
except:
    # Fall back to Agg backend for headless environments
    matplotlib.use('Agg')
    interactive_available = False

try:
    import DriftCorrectionFunctions as DCF
    import IOFunctions
    import render  # For creating images from localizations
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the pyBayerSMLM directory with src/ available")
    sys.exit(1)

@dataclass
class FiducialSelectionResult:
    """Results from interactive fiducial selection."""
    picked_coordinates: List[Tuple[float, float]]
    selected_localizations: np.recarray
    n_fiducials: int
    selection_method: str
    parameters_used: Dict[str, Any]


class InteractiveFiducialSelector:
    """Interactive tool for selecting fiducial markers from localization data."""
    
    def __init__(self, locs: np.recarray, info: List[dict]):
        """
        Initialize the fiducial selector.
        
        Args:
            locs: Localization data
            info: Metadata list
        """
        self.locs = locs
        self.info = info
        self.picked_coordinates = []
        self.fig = None
        self.ax = None
        self.circles = []
        
        # Extract metadata
        meta = DCF.CoordinateProcessor.extract_metadata(info)
        self.width = meta["width"]
        self.height = meta["height"] 
        self.pixelsize = meta.get("pixelsize", 0.1)  # Default 100nm pixels
        
        print(f"Interactive Fiducial Selector initialized")
        print(f"  Image dimensions: {self.width} x {self.height} pixels")
        print(f"  Pixel size: {self.pixelsize*1000:.0f} nm")
        print(f"  Total localizations: {len(locs)}")
        
    def create_localization_image(self, blur_method: str = "gaussian") -> np.ndarray:
        """
        Create a rendered image from localizations for fiducial selection.
        
        Args:
            blur_method: Rendering blur method
            
        Returns:
            Rendered image array
        """
        print("Rendering localizations to image...")
        
        try:
            # Use render module to create image
            image = render.render(
                locs=self.locs,
                info=self.info,
                blur_method=blur_method,
                min_blur_width=1.0
            )[1]  # Take the rendered image, not the info
            
            print(f"✅ Image rendered: {image.shape} pixels")
            return image
            
        except Exception as e:
            print(f"❌ Error rendering image: {e}")
            # Fallback: create simple histogram image
            return self._create_histogram_image()
            
    def _create_histogram_image(self) -> np.ndarray:
        """Create a simple 2D histogram as fallback rendering method."""
        print("Creating histogram image as fallback...")
        
        # Convert coordinates to pixel indices
        x_pixels = np.clip((self.locs.xc / self.pixelsize).astype(int), 0, self.width-1)
        y_pixels = np.clip((self.locs.yc / self.pixelsize).astype(int), 0, self.height-1)
        
        # Create 2D histogram
        image = np.zeros((self.height, self.width), dtype=np.float32)
        
        # Add localizations (with photon weighting if available)
        if hasattr(self.locs, 'photons'):
            weights = self.locs.photons
        else:
            weights = np.ones(len(self.locs))
            
        for x, y, w in zip(x_pixels, y_pixels, weights):
            image[y, x] += w
            
        return image
        
    def automatic_fiducial_detection(
        self,
        threshold_percentile: float = 95.0,
        box_size_nm: float = 900.0,
        min_frames_fraction: float = 0.8,
        histogram_bins: int = 256,
        preview: bool = True
    ) -> FiducialSelectionResult:
        """
        Automatically detect fiducial markers using built-in detection.
        
        Args:
            threshold_percentile: Histogram percentile threshold (0-100)
            box_size_nm: Detection box size in nanometers
            min_frames_fraction: Minimum fraction of frames for valid fiducial
            histogram_bins: Number of histogram bins
            preview: Whether to show preview of detected fiducials
            
        Returns:
            FiducialSelectionResult with detected fiducials
        """
        print("\n" + "="*60)
        print("AUTOMATIC FIDUCIAL DETECTION")
        print("="*60)
        
        print(f"Detection parameters:")
        print(f"  - Threshold percentile: {threshold_percentile}%")
        print(f"  - Box size: {box_size_nm} nm")
        print(f"  - Min frames fraction: {min_frames_fraction}")
        print(f"  - Histogram bins: {histogram_bins}")
        
        # Use the built-in undrift_with_fiducial_detection method
        try:
            drift_corrector = DCF.Drift_Correction_Functions()
            corrected_locs, drift_result, detection_info = drift_corrector.undrift_with_fiducial_detection(
                locs=self.locs,
                info=self.info,
                threshold_percentile=threshold_percentile,
                box_size_nm=box_size_nm,
                min_frames_fraction=min_frames_fraction,
                histogram_bins=histogram_bins
            )
            
            print(f"✅ Automatic detection completed")
            print(f"   Found {detection_info['n_fiducials']} fiducials")
            print(f"   Frames per fiducial: {detection_info['frames_per_fiducial']}")
            
            # Extract fiducial coordinates from the group field
            picked_coords = self._extract_fiducial_coordinates(corrected_locs)
            
            result = FiducialSelectionResult(
                picked_coordinates=picked_coords,
                selected_localizations=corrected_locs,
                n_fiducials=detection_info['n_fiducials'],
                selection_method="automatic",
                parameters_used={
                    "threshold_percentile": threshold_percentile,
                    "box_size_nm": box_size_nm,
                    "min_frames_fraction": min_frames_fraction,
                    "histogram_bins": histogram_bins
                }
            )
            
            if preview and interactive_available:
                self.preview_fiducials(result)
                
            return result
            
        except Exception as e:
            print(f"❌ Automatic detection failed: {e}")
            print("Try adjusting parameters or using manual selection")
            raise
            
    def _extract_fiducial_coordinates(self, locs_with_groups: np.recarray) -> List[Tuple[float, float]]:
        """Extract fiducial center coordinates from localizations with group field."""
        coordinates = []
        
        if hasattr(locs_with_groups, 'group'):
            unique_groups = np.unique(locs_with_groups.group)
            
            for group_id in unique_groups:
                if group_id >= 0:  # Valid fiducial groups
                    group_locs = locs_with_groups[locs_with_groups.group == group_id]
                    if len(group_locs) > 0:
                        # Calculate center of mass for this fiducial
                        center_x = np.mean(group_locs.xc)
                        center_y = np.mean(group_locs.yc)
                        coordinates.append((center_x, center_y))
                        
        return coordinates
        
    def manual_fiducial_selection(
        self,
        pick_radius_nm: float = 500.0,
        min_localizations: int = 50
    ) -> FiducialSelectionResult:
        """
        Manually select fiducial markers by clicking on the image.
        
        Args:
            pick_radius_nm: Selection radius in nanometers
            min_localizations: Minimum localizations required per fiducial
            
        Returns:
            FiducialSelectionResult with manually selected fiducials
        """
        if not interactive_available:
            print("❌ Interactive selection not available (no interactive backend)")
            print("Try using automatic_fiducial_detection() instead")
            return None
            
        print("\n" + "="*60)
        print("MANUAL FIDUCIAL SELECTION")
        print("="*60)
        
        print(f"Selection parameters:")
        print(f"  - Pick radius: {pick_radius_nm} nm")
        print(f"  - Min localizations per fiducial: {min_localizations}")
        print("\nInstructions:")
        print("  - Click on fiducial markers in the plot")
        print("  - Press 'r' to remove the last selection")
        print("  - Press 'q' or close the window when finished")
        
        # Create localization image
        image = self.create_localization_image()
        
        # Setup interactive plot
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        
        # Display the image
        extent = [0, self.width * self.pixelsize, 0, self.height * self.pixelsize]
        self.ax.imshow(image, extent=extent, origin='lower', cmap='hot', interpolation='nearest')
        self.ax.set_xlabel('X (μm)')
        self.ax.set_ylabel('Y (μm)')
        self.ax.set_title('Click to select fiducial markers\n(Press "q" when finished, "r" to remove last)')
        
        # Connect event handlers
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        
        # Show plot and wait for user interaction
        plt.show()
        
        # Process the picked coordinates
        picked_locs_list = []
        valid_coordinates = []
        
        pick_radius_pixels = pick_radius_nm / (self.pixelsize * 1000)  # Convert to pixels
        
        for i, (x, y) in enumerate(self.picked_coordinates):
            # Find localizations within pick radius
            distances = np.sqrt((self.locs.xc - x)**2 + (self.locs.yc - y)**2)
            within_radius = self.locs[distances <= pick_radius_pixels]
            
            if len(within_radius) >= min_localizations:
                picked_locs_list.append(within_radius)
                valid_coordinates.append((x, y))
                print(f"✅ Fiducial {i+1}: {len(within_radius)} localizations at ({x:.2f}, {y:.2f})")
            else:
                print(f"⚠️ Fiducial {i+1}: Only {len(within_radius)} localizations (< {min_localizations})")
                
        if len(valid_coordinates) == 0:
            print("❌ No valid fiducials selected")
            return None
            
        # Create localization array with group field
        selected_locs = self._create_grouped_localizations(picked_locs_list)
        
        result = FiducialSelectionResult(
            picked_coordinates=valid_coordinates,
            selected_localizations=selected_locs,
            n_fiducials=len(valid_coordinates),
            selection_method="manual",
            parameters_used={
                "pick_radius_nm": pick_radius_nm,
                "min_localizations": min_localizations
            }
        )
        
        print(f"✅ Manual selection completed: {len(valid_coordinates)} fiducials")
        return result
        
    def _on_click(self, event):
        """Handle mouse click events for fiducial selection."""
        if event.inaxes != self.ax:
            return
            
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
            
        # Add to picked coordinates
        self.picked_coordinates.append((x, y))
        
        # Draw circle to show selection
        circle = Circle((x, y), 0.5, fill=False, color='cyan', linewidth=2)
        self.ax.add_patch(circle)
        self.circles.append(circle)
        
        # Add text label
        text = self.ax.text(x, y+0.6, f'{len(self.picked_coordinates)}', 
                           color='cyan', ha='center', va='bottom', fontweight='bold')
        self.circles.append(text)
        
        self.fig.canvas.draw()
        print(f"Selected fiducial {len(self.picked_coordinates)} at ({x:.2f}, {y:.2f})")
        
    def _on_key_press(self, event):
        """Handle key press events."""
        if event.key == 'r' and len(self.picked_coordinates) > 0:
            # Remove last selection
            self.picked_coordinates.pop()
            
            # Remove last two graphics (circle and text)
            if len(self.circles) >= 2:
                self.circles[-1].remove()  # text
                self.circles[-2].remove()  # circle
                self.circles = self.circles[:-2]
                
            self.fig.canvas.draw()
            print(f"Removed last selection. {len(self.picked_coordinates)} fiducials remaining")
            
        elif event.key == 'q':
            # Quit selection
            plt.close(self.fig)
            print("Selection finished")
            
    def _create_grouped_localizations(self, picked_locs_list: List[np.recarray]) -> np.recarray:
        """Create localization array with group field from picked localizations."""
        # Create group field array
        all_locs_list = []
        
        # Add non-selected localizations with group = -1
        remaining_locs = self.locs.copy()
        group_field = np.full(len(remaining_locs), -1, dtype=np.int32)
        
        # Mark selected localizations with group IDs
        for group_id, fiducial_locs in enumerate(picked_locs_list):
            for fid_loc in fiducial_locs:
                # Find matching localizations
                matches = (
                    (remaining_locs.frame == fid_loc.frame) &
                    (np.abs(remaining_locs.xc - fid_loc.xc) < 0.01) &
                    (np.abs(remaining_locs.yc - fid_loc.yc) < 0.01)
                )
                group_field[matches] = group_id
                
        # Create new dtype with group field
        original_dtype = remaining_locs.dtype
        group_dtype = np.dtype(original_dtype.descr + [("group", "i4")])
        
        # Create new recarray
        new_locs = np.empty(len(remaining_locs), dtype=group_dtype)
        
        # Copy original data
        for field in original_dtype.names:
            new_locs[field] = remaining_locs[field]
            
        # Add group data
        new_locs["group"] = group_field
        
        return new_locs.view(np.recarray)
        
    def preview_fiducials(self, selection_result: FiducialSelectionResult):
        """Show preview of selected fiducials."""
        if not interactive_available:
            # Save static preview instead
            self._save_fiducial_preview(selection_result, '/tmp/fiducial_preview.png')
            return
            
        print(f"\nPreviewing {selection_result.n_fiducials} selected fiducials...")
        
        # Create image
        image = self.create_localization_image()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Display image
        extent = [0, self.width * self.pixelsize, 0, self.height * self.pixelsize]
        ax.imshow(image, extent=extent, origin='lower', cmap='hot', alpha=0.7)
        
        # Overlay fiducial selections
        for i, (x, y) in enumerate(selection_result.picked_coordinates):
            circle = Circle((x, y), 0.5, fill=False, color='cyan', linewidth=3)
            ax.add_patch(circle)
            ax.text(x, y+0.7, f'F{i+1}', color='cyan', ha='center', 
                   va='bottom', fontweight='bold', fontsize=12)
                   
        ax.set_xlabel('X (μm)')
        ax.set_ylabel('Y (μm)')
        ax.set_title(f'Selected Fiducials ({selection_result.selection_method})\n'
                    f'{selection_result.n_fiducials} fiducials detected')
        
        plt.tight_layout()
        plt.show()
        
    def _save_fiducial_preview(self, selection_result: FiducialSelectionResult, filename: str):
        """Save static fiducial preview for headless environments."""
        print(f"Saving fiducial preview to {filename}")
        
        image = self.create_localization_image()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        extent = [0, self.width * self.pixelsize, 0, self.height * self.pixelsize]
        ax.imshow(image, extent=extent, origin='lower', cmap='hot', alpha=0.7)
        
        for i, (x, y) in enumerate(selection_result.picked_coordinates):
            circle = Circle((x, y), 0.5, fill=False, color='cyan', linewidth=3)
            ax.add_patch(circle)
            ax.text(x, y+0.7, f'F{i+1}', color='cyan', ha='center', 
                   va='bottom', fontweight='bold', fontsize=12)
                   
        ax.set_xlabel('X (μm)')
        ax.set_ylabel('Y (μm)')
        ax.set_title(f'Selected Fiducials ({selection_result.selection_method})')
        
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ Preview saved to {filename}")


def load_example_data() -> Tuple[np.recarray, List[dict]]:
    """
    Load or create example data with fiducial markers for demonstration.
    
    Returns:
        Tuple of (localizations, metadata_info)
    """
    print("Creating example data with simulated fiducials...")
    
    np.random.seed(42)
    
    # Simulation parameters
    n_frames = 1000
    image_size = 128  # 128x128 pixel image
    pixel_size = 0.1  # 100nm pixels
    
    # Create 4 fiducial markers at known positions
    fiducial_positions = [
        (20, 20),   # Bottom-left
        (108, 20),  # Bottom-right  
        (20, 108),  # Top-left
        (108, 108)  # Top-right
    ]
    
    # Simulate drift (smaller for fiducials to remain identifiable)
    frames = np.arange(n_frames)
    true_drift_x = 0.5 * np.sin(frames / 100) + 0.002 * frames
    true_drift_y = 0.3 * np.cos(frames / 80) + 0.001 * frames
    
    all_locs = []
    
    # Generate fiducial localizations (high density, stable)
    for fid_id, (base_x, base_y) in enumerate(fiducial_positions):
        n_locs_per_frame = 20  # Dense fiducials
        
        for frame in range(n_frames):
            if np.random.random() > 0.1:  # 90% probability of detection per frame
                n_this_frame = np.random.poisson(n_locs_per_frame)
                
                # Base positions with small random scatter
                x_coords = base_x + np.random.normal(0, 0.3, n_this_frame)
                y_coords = base_y + np.random.normal(0, 0.3, n_this_frame) 
                
                # Add drift
                x_coords += true_drift_x[frame]
                y_coords += true_drift_y[frame]
                
                # Add localization precision noise
                x_coords += np.random.normal(0, 0.02, n_this_frame)
                y_coords += np.random.normal(0, 0.02, n_this_frame)
                
                # Create localization records
                for x, y in zip(x_coords, y_coords):
                    all_locs.append({
                        'xc': x * pixel_size,  # Convert to micrometers
                        'yc': y * pixel_size,
                        'frame': frame,
                        'photons': np.random.exponential(2000),
                        'is_fiducial': True,
                        'fiducial_id': fid_id
                    })
    
    # Generate background cellular structures (lower density, more variable)
    n_bg_structures = 6
    
    for struct_id in range(n_bg_structures):
        # Random positions away from fiducials
        while True:
            base_x = np.random.uniform(30, 98)
            base_y = np.random.uniform(30, 98)
            
            # Check distance from fiducials
            min_dist = min([np.sqrt((base_x - fx)**2 + (base_y - fy)**2) 
                          for fx, fy in fiducial_positions])
            if min_dist > 15:  # At least 15 pixels from fiducials
                break
                
        for frame in range(n_frames):
            if np.random.random() > 0.7:  # 30% probability per frame
                n_this_frame = np.random.poisson(5)  # Sparse background
                
                # Larger scatter for biological structures
                x_coords = base_x + np.random.normal(0, 3, n_this_frame)
                y_coords = base_y + np.random.normal(0, 3, n_this_frame)
                
                # Add drift
                x_coords += true_drift_x[frame] 
                y_coords += true_drift_y[frame]
                
                # Add precision noise
                x_coords += np.random.normal(0, 0.05, n_this_frame)
                y_coords += np.random.normal(0, 0.05, n_this_frame)
                
                for x, y in zip(x_coords, y_coords):
                    all_locs.append({
                        'xc': x * pixel_size,
                        'yc': y * pixel_size,
                        'frame': frame,
                        'photons': np.random.exponential(800),
                        'is_fiducial': False,
                        'fiducial_id': -1
                    })
    
    # Convert to structured array
    locs_df = pd.DataFrame(all_locs)
    locs = np.rec.fromrecords(
        locs_df[['xc', 'yc', 'frame', 'photons']].values,
        names=['xc', 'yc', 'frame', 'photons']
    )
    
    # Create metadata
    info = [{
        'Width': image_size,
        'Height': image_size,
        'Frames': n_frames,
        'Pixelsize': pixel_size
    }]
    
    print(f"✅ Created example data:")
    print(f"   - {len(locs)} total localizations")
    print(f"   - {len(fiducial_positions)} simulated fiducials") 
    print(f"   - {n_frames} frames")
    print(f"   - Image size: {image_size}x{image_size} pixels ({pixel_size*1000:.0f} nm/pixel)")
    print(f"   - Simulated drift range: X=[{true_drift_x.min():.3f}, {true_drift_x.max():.3f}], Y=[{true_drift_y.min():.3f}, {true_drift_y.max():.3f}] pixels")
    
    return locs, info


def perform_fiducial_drift_correction(
    selection_result: FiducialSelectionResult,
    info: List[dict]
) -> Tuple[np.recarray, DCF.DriftResult]:
    """
    Perform drift correction using selected fiducials.
    
    Args:
        selection_result: Result from fiducial selection
        info: Metadata list
        
    Returns:
        Tuple of (corrected_localizations, drift_result)
    """
    print("\n" + "="*60)
    print("PERFORMING FIDUCIAL DRIFT CORRECTION")
    print("="*60)
    
    print(f"Using {selection_result.n_fiducials} fiducials from {selection_result.selection_method} selection")
    print(f"Selection parameters: {selection_result.parameters_used}")
    
    # Initialize drift corrector
    drift_corrector = DCF.Drift_Correction_Functions()
    
    # Apply fiducial drift correction directly (data already has group field)
    corrected_locs, drift_result = drift_corrector.undrift(
        locs=selection_result.selected_localizations,
        info=info,
        method="fiducial"
    )
    
    print(f"✅ Fiducial drift correction completed!")
    print(f"   Method: {drift_result.method}")
    print(f"   Fiducials used: {drift_result.metadata.get('n_fiducials', 'unknown')}")
    print(f"   X drift range: {drift_result.drift_x.min():.3f} to {drift_result.drift_x.max():.3f} pixels")
    print(f"   Y drift range: {drift_result.drift_y.min():.3f} to {drift_result.drift_y.max():.3f} pixels")
    
    return corrected_locs, drift_result


def analyze_fiducial_quality(
    selection_result: FiducialSelectionResult,
    drift_result: DCF.DriftResult
) -> Dict[str, Any]:
    """
    Analyze the quality of fiducial-based drift correction.
    
    Args:
        selection_result: Fiducial selection results
        drift_result: Drift correction results
        
    Returns:
        Dictionary with quality metrics
    """
    print("\n" + "="*60)
    print("ANALYZING FIDUCIAL DRIFT CORRECTION QUALITY")
    print("="*60)
    
    # Extract fiducial localizations
    locs_with_groups = selection_result.selected_localizations
    fiducial_locs = locs_with_groups[locs_with_groups.group >= 0]
    
    if len(fiducial_locs) == 0:
        print("❌ No fiducial localizations found for analysis")
        return {}
    
    # Calculate per-fiducial statistics
    unique_groups = np.unique(fiducial_locs.group)
    fiducial_stats = []
    
    for group_id in unique_groups:
        group_locs = fiducial_locs[fiducial_locs.group == group_id]
        
        # Calculate center and spread
        center_x = np.mean(group_locs.xc)
        center_y = np.mean(group_locs.yc)
        std_x = np.std(group_locs.xc)
        std_y = np.std(group_locs.yc)
        
        fiducial_stats.append({
            'group_id': group_id,
            'n_localizations': len(group_locs),
            'center_x': center_x,
            'center_y': center_y,
            'std_x': std_x,
            'std_y': std_y,
            'frames_present': len(np.unique(group_locs.frame))
        })
    
    # Overall quality metrics
    all_std_x = [f['std_x'] for f in fiducial_stats]
    all_std_y = [f['std_y'] for f in fiducial_stats]
    
    metrics = {
        'n_fiducials': len(fiducial_stats),
        'total_fiducial_localizations': len(fiducial_locs),
        'mean_localizations_per_fiducial': np.mean([f['n_localizations'] for f in fiducial_stats]),
        'mean_std_x': np.mean(all_std_x),
        'mean_std_y': np.mean(all_std_y),
        'max_drift_magnitude': np.sqrt(drift_result.drift_x**2 + drift_result.drift_y**2).max(),
        'drift_x_range': drift_result.drift_x.max() - drift_result.drift_x.min(),
        'drift_y_range': drift_result.drift_y.max() - drift_result.drift_y.min(),
        'fiducial_stats': fiducial_stats,
        'selection_method': selection_result.selection_method
    }
    
    # Print quality summary
    print(f"Fiducial Quality Summary:")
    print(f"  - Number of fiducials: {metrics['n_fiducials']}")
    print(f"  - Total fiducial localizations: {metrics['total_fiducial_localizations']}")
    print(f"  - Average localizations per fiducial: {metrics['mean_localizations_per_fiducial']:.1f}")
    print(f"  - Mean fiducial precision: X={metrics['mean_std_x']*1000:.1f} nm, Y={metrics['mean_std_y']*1000:.1f} nm")
    
    print(f"\nDrift Correction Results:")
    print(f"  - Maximum drift magnitude: {metrics['max_drift_magnitude']:.3f} pixels")
    print(f"  - X drift range: {metrics['drift_x_range']:.3f} pixels")
    print(f"  - Y drift range: {metrics['drift_y_range']:.3f} pixels")
    
    print(f"\nPer-Fiducial Statistics:")
    for i, fid_stat in enumerate(fiducial_stats):
        print(f"  Fiducial {fid_stat['group_id']+1}: {fid_stat['n_localizations']} locs, "
              f"precision = ({fid_stat['std_x']*1000:.1f}, {fid_stat['std_y']*1000:.1f}) nm, "
              f"frames = {fid_stat['frames_present']}")
    
    return metrics


def save_fiducial_results(
    selection_result: FiducialSelectionResult,
    corrected_locs: np.recarray,
    drift_result: DCF.DriftResult,
    quality_metrics: Dict[str, Any],
    output_base: str = "/tmp/fiducial_drift_correction"
):
    """
    Save all fiducial drift correction results.
    
    Args:
        selection_result: Fiducial selection results
        corrected_locs: Corrected localizations
        drift_result: Drift correction results
        quality_metrics: Quality analysis results
        output_base: Base path for output files
    """
    print(f"\n" + "="*60)
    print("SAVING FIDUCIAL DRIFT CORRECTION RESULTS")
    print("="*60)
    
    # Save corrected localizations
    corrected_df = pd.DataFrame(corrected_locs)
    corrected_path = f"{output_base}_corrected_locs.csv"
    corrected_df.to_csv(corrected_path, index=False)
    print(f"✅ Corrected localizations: {corrected_path}")
    
    # Save drift trace
    drift_df = pd.DataFrame({
        'frame': np.arange(len(drift_result.drift_x)),
        'drift_x_pixels': drift_result.drift_x,
        'drift_y_pixels': drift_result.drift_y
    })
    drift_path = f"{output_base}_drift_trace.csv"
    drift_df.to_csv(drift_path, index=False)
    print(f"✅ Drift trace: {drift_path}")
    
    # Save fiducial selection info
    selection_info = {
        'selection_method': selection_result.selection_method,
        'n_fiducials': selection_result.n_fiducials,
        'parameters_used': selection_result.parameters_used,
        'picked_coordinates': selection_result.picked_coordinates
    }
    
    import json
    selection_path = f"{output_base}_selection_info.json"
    with open(selection_path, 'w') as f:
        json.dump(selection_info, f, indent=2, default=str)
    print(f"✅ Selection info: {selection_path}")
    
    # Save quality metrics
    quality_path = f"{output_base}_quality_metrics.json"
    with open(quality_path, 'w') as f:
        json.dump(quality_metrics, f, indent=2, default=str)
    print(f"✅ Quality metrics: {quality_path}")
    
    return corrected_path, drift_path, selection_path, quality_path


def plot_fiducial_drift_results(
    original_locs: np.recarray,
    corrected_locs: np.recarray,
    selection_result: FiducialSelectionResult,
    drift_result: DCF.DriftResult,
    save_path: str = "/tmp/fiducial_drift_comparison.png"
):
    """
    Create comprehensive plots showing fiducial drift correction results.
    
    Args:
        original_locs: Original localization data
        corrected_locs: Drift-corrected data
        selection_result: Fiducial selection results
        drift_result: Drift correction results
        save_path: Path to save the plot
    """
    fig = plt.figure(figsize=(16, 12))
    
    # Layout: 2x3 grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. Original data with fiducial markers
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(original_locs.xc, original_locs.yc, s=0.5, alpha=0.3, c='red', label='All locs')
    
    # Highlight fiducial regions
    for i, (x, y) in enumerate(selection_result.picked_coordinates):
        circle = Circle((x, y), 0.5, fill=False, color='cyan', linewidth=2)
        ax1.add_patch(circle)
        ax1.text(x, y+0.6, f'F{i+1}', color='cyan', ha='center', va='bottom', fontweight='bold')
    
    ax1.set_title('Original Data + Fiducials')
    ax1.set_xlabel('X (μm)')
    ax1.set_ylabel('Y (μm)')
    ax1.set_aspect('equal')
    ax1.legend()
    
    # 2. Corrected data
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(corrected_locs.xc, corrected_locs.yc, s=0.5, alpha=0.3, c='blue', label='Corrected')
    ax2.set_title('After Fiducial Correction')
    ax2.set_xlabel('X (μm)')
    ax2.set_ylabel('Y (μm)')
    ax2.set_aspect('equal')
    ax2.legend()
    
    # 3. Drift traces
    ax3 = fig.add_subplot(gs[0, 2])
    frames = np.arange(len(drift_result.drift_x))
    ax3.plot(frames, drift_result.drift_x, 'b-', linewidth=1, label='X drift')
    ax3.plot(frames, drift_result.drift_y, 'r-', linewidth=1, label='Y drift')
    ax3.set_title('Measured Drift')
    ax3.set_xlabel('Frame')
    ax3.set_ylabel('Drift (pixels)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Fiducial localizations only (before correction)
    ax4 = fig.add_subplot(gs[1, 0])
    if hasattr(selection_result.selected_localizations, 'group'):
        fiducial_locs = selection_result.selected_localizations[
            selection_result.selected_localizations.group >= 0
        ]
        
        # Color by fiducial group
        unique_groups = np.unique(fiducial_locs.group)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_groups)))
        
        for i, group_id in enumerate(unique_groups):
            group_locs = fiducial_locs[fiducial_locs.group == group_id]
            ax4.scatter(group_locs.xc, group_locs.yc, s=2, alpha=0.6, 
                       c=[colors[i]], label=f'Fiducial {group_id+1}')
    
    ax4.set_title('Fiducials Before Correction')
    ax4.set_xlabel('X (μm)')
    ax4.set_ylabel('Y (μm)')
    ax4.legend()
    ax4.set_aspect('equal')
    
    # 5. Fiducial localizations after correction
    ax5 = fig.add_subplot(gs[1, 1])
    corrected_fiducial_locs = corrected_locs[corrected_locs.group >= 0] if hasattr(corrected_locs, 'group') else []
    
    if len(corrected_fiducial_locs) > 0:
        unique_groups = np.unique(corrected_fiducial_locs.group)
        for i, group_id in enumerate(unique_groups):
            group_locs = corrected_fiducial_locs[corrected_fiducial_locs.group == group_id]
            ax5.scatter(group_locs.xc, group_locs.yc, s=2, alpha=0.6,
                       c=[colors[i]], label=f'Fiducial {group_id+1}')
    
    ax5.set_title('Fiducials After Correction')
    ax5.set_xlabel('X (μm)')
    ax5.set_ylabel('Y (μm)')
    ax5.legend()
    ax5.set_aspect('equal')
    
    # 6. Drift magnitude over time
    ax6 = fig.add_subplot(gs[1, 2])
    drift_magnitude = np.sqrt(drift_result.drift_x**2 + drift_result.drift_y**2)
    ax6.plot(frames, drift_magnitude, 'g-', linewidth=1)
    ax6.set_title('Drift Magnitude')
    ax6.set_xlabel('Frame')
    ax6.set_ylabel('Drift Magnitude (pixels)')
    ax6.grid(True, alpha=0.3)
    
    # Overall title
    fig.suptitle(f'Fiducial Drift Correction Results\n'
                f'Method: {selection_result.selection_method}, '
                f'{selection_result.n_fiducials} fiducials, '
                f'Max drift: {drift_magnitude.max():.3f} pixels', 
                fontsize=14, fontweight='bold')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Results plot saved: {save_path}")
    
    if interactive_available:
        plt.show()
    else:
        plt.close(fig)


def main():
    """
    Main function demonstrating interactive fiducial drift correction workflow.
    """
    print("Interactive Fiducial Drift Correction Example")
    print("="*60)
    
    if not interactive_available:
        print("⚠️ Interactive plotting not available - using automatic detection only")
    
    # 1. Load example data  
    locs, info = load_example_data()
    
    # 2. Create fiducial selector
    selector = InteractiveFiducialSelector(locs, info)
    
    # 3. Choose selection method
    print(f"\nFiducial Selection Methods:")
    print(f"1. Automatic detection (recommended)")
    print(f"2. Manual selection (requires interactive display)")
    
    if interactive_available:
        choice = input("\nChoose method (1 or 2, default=1): ").strip() or "1"
    else:
        choice = "1"
        print("Using automatic detection (interactive not available)")
    
    if choice == "2" and interactive_available:
        # Manual selection
        selection_result = selector.manual_fiducial_selection(
            pick_radius_nm=800.0,  # 800nm selection radius
            min_localizations=30    # Minimum 30 localizations per fiducial
        )
    else:
        # Automatic selection
        selection_result = selector.automatic_fiducial_detection(
            threshold_percentile=90.0,  # Lower threshold for more candidates
            box_size_nm=1000.0,        # 1μm detection box
            min_frames_fraction=0.5,   # At least 50% of frames
            histogram_bins=256,
            preview=True
        )
    
    if selection_result is None:
        print("❌ No fiducials selected. Exiting.")
        return
    
    # 4. Perform fiducial drift correction
    corrected_locs, drift_result = perform_fiducial_drift_correction(selection_result, info)
    
    # 5. Analyze correction quality
    quality_metrics = analyze_fiducial_quality(selection_result, drift_result)
    
    # 6. Save results
    file_paths = save_fiducial_results(
        selection_result, corrected_locs, drift_result, quality_metrics
    )
    
    # 7. Create comprehensive plots
    plot_fiducial_drift_results(
        locs, corrected_locs, selection_result, drift_result
    )
    
    # 8. Summary
    print(f"\n" + "="*60)
    print("FIDUCIAL DRIFT CORRECTION COMPLETE")
    print("="*60)
    print(f"✅ Selection method: {selection_result.selection_method}")
    print(f"✅ Fiducials used: {selection_result.n_fiducials}")
    print(f"✅ Original data: {len(locs)} localizations")
    print(f"✅ Corrected data: {len(corrected_locs)} localizations")
    print(f"✅ Max drift corrected: {quality_metrics.get('max_drift_magnitude', 0):.3f} pixels")
    print(f"✅ Mean fiducial precision: {quality_metrics.get('mean_std_x', 0)*1000:.1f} nm (X), {quality_metrics.get('mean_std_y', 0)*1000:.1f} nm (Y)")
    print(f"✅ Results saved to: /tmp/fiducial_drift_correction_*")


if __name__ == "__main__":
    # Usage notes:
    #
    # For automatic fiducial detection:
    # python interactive_fiducial_drift_correction_example.py
    # (Choose option 1)
    #
    # For manual selection:
    # python interactive_fiducial_drift_correction_example.py  
    # (Choose option 2, then click on fiducials in the plot)
    #
    # For your own data, replace load_example_data() with:
    # def load_your_data():
    #     io = IOFunctions.IO_Functions()
    #     locs = io.read_localisations("your_data.csv")
    #     info = [{"Width": 256, "Height": 256, "Frames": 10000, "Pixelsize": 0.1}]
    #     return locs, info
    
    main()