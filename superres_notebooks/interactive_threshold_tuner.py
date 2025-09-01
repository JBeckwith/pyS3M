#!/usr/bin/env python3
"""
Interactive Threshold Tuner for pyBayerSMLM Batch Analysis

This script helps determine optimal intensity thresholds (pfa and perc_threshold) 
for spot detection across different datasets. It interactively loads one frame 
from each folder in the batch analysis workflow, allows the user to test different 
parameters, and saves the results for use by batch_analysis.sh.

Usage:
    python claude/interactive_threshold_tuner.py

Output:
    threshold_parameters.txt - Parameter file for batch_analysis.sh
    
Author: Claude Code (Anthropic)
Date: September 1, 2025
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
sys.path.append('/home/jbeckwith/Documents/pCloud/Chemistry/Lee/Code/Python/pyBayerSMLM/src')

try:
    from SpotDetectionFunctions import SpotDetection_Functions
    from IOFunctions import IO_Functions
    from CalibrationFunctions import Calibration_Functions
    import matplotlib
    # Try to use interactive backend
    try:
        import tkinter
        matplotlib.use('TkAgg')
        print("Using interactive matplotlib backend")
    except ImportError:
        print("tkinter not available, installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tk"])
        import tkinter
        matplotlib.use('TkAgg')
        print("tkinter installed and interactive backend enabled")
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"Error importing pyBayerSMLM modules: {e}")
    print("Please ensure the virtual environment is activated:")
    print("source /home/jbeckwith/.virtualenvs/pyBayerSMLM/bin/activate")
    sys.exit(1)


class InteractiveThresholdTuner:
    """Interactive tool for determining optimal spot detection thresholds"""
    
    def __init__(self):
        self.sdf = SpotDetection_Functions()
        self.iof = IO_Functions()
        self.cf = Calibration_Functions()
        
        # Default parameters
        self.default_pfa = 1e-4
        self.default_perc_threshold = 98.0
        self.default_wavelength = 0.638
        
        # Results storage
        self.threshold_results = {}
        
        # Folder lists from batch_analysis.sh
        self.folder_lists = self._get_folder_lists()
        
    def _get_folder_lists(self) -> Dict[str, List[str]]:
        """Get folder lists exactly as defined in batch_analysis.sh"""
        return {
            'SM_DATA_DIRS': [
                '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250819_TetraspeckCalibration',
                '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250717_BiotinDyes/ATTO488_50PM_PCA_PCD',
                '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250725 biotinylated dyes/ATTO514_50pM_PCAPCDTx',
                '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250725 biotinylated dyes/ATTO520_50pM_PCAPCDTx',
                '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250725 biotinylated dyes/ATTORho6G_50pM_PCAPCDTx',
                '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250714_BiotinylatedDyes/Atto565_PCA_PCD_Tx_50pMDye',
                '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250714_BiotinylatedDyes/Atto620_PCA_PCD_Tx_50pMDye',
                '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250711 Biotinylated Dyes/Atto633_PCA_PCD_Tx_100pMDye',
                '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250714_BiotinylatedDyes/Atto647N_PCA_PCD_Tx_20pMDye',
                '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250717_BiotinDyes/ATTO655_50PM_PCA_PCD',
                '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250717_BiotinDyes/ATTO700_50PM_PCA_PCD',
                '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/JSB/20250609_dyes/data',
                '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250714_BiotinylatedDyes/Atto594_PCA_PCD_Tx_50pMDye',
                '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250715_HollidayJunctions/60pM_HollidayJunction_50mMMgCl2/40perc561_NF_SP785_30ms_1',
                '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250715_HollidayJunctions/60pM_HollidayJunction_50mMMgCl2/100perc561_NF_SP785_5ms_1',
                '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250715_HollidayJunctions/60pM_HollidayJunction_50mMMgCl2/100perc561_NF_SP785_10ms_1',
                '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250715_HollidayJunctions/60pM_HollidayJunction_50mMMgCl2/100perc561_NF_SP785_50ms_1'
            ],
            'HELA_FOLDERS': [
                '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250523_HeLa_STORM/Cell3_HILO_190mW_638_ximea638_setting/Lp638_190_mw_40ms_exosure_HILO_1',
                '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250523_HeLa_STORM/Cell4_HILO_190mW_638_ximea638_setting/Lp638_190_mw_40ms_exosure_HILO_1',
                '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250523_HeLa_STORM/Cell2_HILO_190mW_638_ximea638_setting/Lp638_190_mw_40ms_exosure_HILO_1',
                '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250523_HeLa_STORM/Cell1_HILO_190mW_638_ximea638_setting/Lp638_190_mw_40ms_exosure_HILO_2'
            ],
            'IMAGING_FOLDERS': [
                '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/JSB/20250717_Origami/F1F2F3F4Cy3B500pM/10perc561_LP561_BP586-64_1',
                '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/JSB/20250717_Origami/F1F2F3F4Cy3B500pM_LowConcOrigami/10perc561_LP561_BP586-64_1',
                '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/JSB/20250514_DNANanoruler/data/DNANanoRuler_10perc561_30mW488_50mW638/F1CF640CF550R_F2ATTO488AF647_F3ATTO565ATTO655_F4Cy3BCF488A_MultiNotch_488LP_758SP_1',
                '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/JSB/20250514_DNANanoruler/data/DNANanoRuler_10perc561_30mW488_50mW638/F1CF640CF550R_F2ATTO488AF647_F3ATTO565ATTO655_F4Cy3BCF488A_MultiNotch_488LP_758SP_1nM_1',
                '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250730 single colour origami/AlexaFluor647_2nM_strands/30mWboth638_NF_785SP_488LP_1',
                '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250730 single colour origami/CF488A_2nM_strands/20mW488_NF_785SP_488LP_1',
                '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250730 single colour origami/CF550R_2nM_strands_adjusteddichroic/30p561_NF_785SP_488LP_1',
                '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250730 single colour origami/CF640R_2nM_strands/30mWboth638_NF_785SP_488LP_1',
                '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250723 DNA Origami/FourColour_F1AF647_F2ATTO565_F3Cy3B_F4ATTO655_500pMEach/15percent_561_40mWEach_638_NotchFilter_785SP_1',
                '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250723 DNA Origami/FourColour_F1AF647_F2ATTO565_F3Cy3B_F4ATTO655_500pMEach/15percent_561_100mWEach_638_NotchFilter_785SP_1',
                '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250723 DNA Origami/FourColour_F1AF647_F2ATTO565_F3Cy3B_F4CF488A_500pMEach/30mW_488_15percent_561_100mWEach_638_NotchFilter_785SP_1',
                '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250723 DNA Origami/FourColour_F1CF550R_F2ATTO565_F3Cy3B_F4CF488A_500pMEach/30mW_488_15percent_561_NotchFilter_785SP_1',
                '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/JSB/20250716_iPSCJamesEvans/40mW488_30perc561_50mW638_NF_488LP_785SP_1',
                '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/JSB/20250716_iPSCJamesEvans/250pMCy3B_250pM565_250pMCF550_250pM647/20perc561_40mW638_NF_488LP_785SP_1'
            ],
            'HIERARCHICAL_DIRS': [
                '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/JSB/20250414_CellPAINT/data',
                '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250404_Ximea_AsynNRThX/data',
                '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250818_DNAOrigami'
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
    
    def get_all_processing_folders(self) -> List[Tuple[str, str, float]]:
        """Get all folders that will be processed by batch_analysis.sh with their parameters"""
        all_folders = []
        
        # SM data directories (hierarchical)
        for base_dir in self.folder_lists['SM_DATA_DIRS']:
            leaf_dirs = self.find_leaf_directories(base_dir)
            for folder in leaf_dirs:
                all_folders.append((folder, 'sm', 0.638))
        
        # HeLa folders (direct)
        for folder in self.folder_lists['HELA_FOLDERS']:
            if os.path.isdir(folder):
                all_folders.append((folder, 'imaging', 0.647))
        
        # Imaging folders (direct)  
        for folder in self.folder_lists['IMAGING_FOLDERS']:
            if os.path.isdir(folder):
                all_folders.append((folder, 'imaging', 0.55))
        
        # Hierarchical imaging directories
        for base_dir in self.folder_lists['HIERARCHICAL_DIRS']:
            leaf_dirs = self.find_leaf_directories(base_dir)
            for folder in leaf_dirs:
                all_folders.append((folder, 'imaging', 0.55))
        
        return all_folders
    
    def find_first_tiff_file(self, folder_path: str) -> Optional[str]:
        """Find the first TIFF file in a folder"""
        folder = Path(folder_path)
        
        # Common TIFF extensions
        tiff_patterns = ['*.tif', '*.tiff', '*.TIF', '*.TIFF']
        
        for pattern in tiff_patterns:
            tiff_files = list(folder.glob(pattern))
            if tiff_files:
                return str(tiff_files[0])
                
        # Also try subdirectories
        for pattern in tiff_patterns:
            tiff_files = list(folder.rglob(pattern))
            if tiff_files:
                return str(tiff_files[0])
        
        return None
    
    def load_test_frame(self, folder_path: str) -> Optional[np.ndarray]:
        """Load one frame from the first TIFF file found in the folder"""
        tiff_file = self.find_first_tiff_file(folder_path)
        
        if not tiff_file:
            print(f"No TIFF files found in {folder_path}")
            return None
        
        try:
            # Load the TIFF file
            with tifffile.TiffFile(tiff_file) as tif:
                # Get the first frame
                frame = tif.pages[0].asarray()
                print(f"Loaded frame from: {tiff_file}")
                print(f"Frame shape: {frame.shape}, dtype: {frame.dtype}")
                print(f"Intensity range: {frame.min()} - {frame.max()}")
                return frame.astype(np.float64)
                
        except Exception as e:
            print(f"Error loading {tiff_file}: {e}")
            return None
    
    def test_spot_detection(self, image: np.ndarray, pfa: float, 
                          perc_threshold: float, wavelength: float) -> Tuple[np.ndarray, int]:
        """Test spot detection with given parameters"""
        try:
            detected_spots = self.sdf.detect_puncta_in_image(
                image=image,
                pfa=pfa,
                wavelength=wavelength,
                perc_threshold=perc_threshold,
                pixel_size=0.069,  # Standard pixel size
                NA=1.49,          # Standard NA
                mf_factor=3.0,    # Standard match filter factor
                local_factor=3.0  # Standard local factor
            )
            return detected_spots, len(detected_spots)
        
        except Exception as e:
            print(f"Error in spot detection: {e}")
            return np.array([]), 0
    
    def plot_detection_results(self, image: np.ndarray, spots: np.ndarray, 
                             pfa: float, perc_threshold: float, folder_name: str,
                             output_path: str = 'current_detection.png'):
        """Plot and save the detection results"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Original image
        ax1.imshow(image, cmap='gray', vmin=np.percentile(image, 1), 
                  vmax=np.percentile(image, 99))
        ax1.set_title(f'Original Image\n{folder_name}')
        ax1.axis('off')
        
        # Image with detected spots
        ax2.imshow(image, cmap='gray', vmin=np.percentile(image, 1), 
                  vmax=np.percentile(image, 99))
        
        if len(spots) > 0:
            ax2.plot(spots[:, 0], spots[:, 1], 'ro', markersize=8, 
                    markerfacecolor='none', markeredgewidth=2)
        
        ax2.set_title(f'Detected Spots ({len(spots)} found)\n'
                     f'PFA: {pfa:.0e}, Perc Threshold: {perc_threshold}%')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Detection plot saved to: {output_path}")
        return output_path
    
    def interactive_parameter_tuning(self, folder_path: str, folder_type: str, 
                                   default_wavelength: float) -> Union[Dict, None, str]:
        """Interactive parameter tuning for a single folder"""
        folder_name = os.path.basename(folder_path)
        print(f"\n{'='*80}")
        print(f"PROCESSING: {folder_name}")
        print(f"Path: {folder_path}")
        print(f"Type: {folder_type}, Wavelength: {default_wavelength}")
        print(f"{'='*80}")
        
        # Load test frame
        frame = self.load_test_frame(folder_path)
        if frame is None:
            print("Could not load test frame, skipping...")
            return None
        
        # Start with default parameters
        current_pfa = self.default_pfa
        current_perc_threshold = self.default_perc_threshold
        current_wavelength = default_wavelength
        
        fig = None
        
        while True:
            # Test current parameters
            spots, num_spots = self.test_spot_detection(
                frame, current_pfa, current_perc_threshold, current_wavelength)
            
            # Close previous plot
            if fig is not None:
                plt.close(fig)
            
            # Plot results
            fig = self.plot_detection_results(
                frame, spots, current_pfa, current_perc_threshold, folder_name)
            
            print(f"\nCurrent parameters:")
            print(f"  PFA (probability of false alarm): {current_pfa:.0e}")
            print(f"  Percentile threshold: {current_perc_threshold}%")
            print(f"  Wavelength: {current_wavelength}")
            print(f"  Detected spots: {num_spots}")
            
            print(f"\nOptions:")
            print(f"  1. Adjust PFA (current: {current_pfa:.0e})")
            print(f"  2. Adjust percentile threshold (current: {current_perc_threshold}%)")
            print(f"  3. Adjust wavelength (current: {current_wavelength})")
            print(f"  4. Accept current parameters")
            print(f"  5. Skip this folder")
            print(f"  q. Quit")
            
            choice = input("Enter choice (1-5 or q): ").strip()
            
            if choice == '1':
                try:
                    new_pfa = float(input(f"Enter new PFA (current: {current_pfa:.0e}): ").strip())
                    current_pfa = new_pfa
                except ValueError:
                    print("Invalid input, keeping current value")
                    
            elif choice == '2':
                try:
                    new_perc = float(input(f"Enter new percentile threshold (current: {current_perc_threshold}%): ").strip())
                    if 0 <= new_perc <= 100:
                        current_perc_threshold = new_perc
                    else:
                        print("Percentile must be between 0 and 100")
                except ValueError:
                    print("Invalid input, keeping current value")
                    
            elif choice == '3':
                try:
                    new_wavelength = float(input(f"Enter new wavelength (current: {current_wavelength}): ").strip())
                    current_wavelength = new_wavelength
                except ValueError:
                    print("Invalid input, keeping current value")
                    
            elif choice == '4':
                # Accept parameters
                plt.close(fig)
                return {
                    'folder_path': folder_path,
                    'folder_type': folder_type,
                    'pfa': current_pfa,
                    'perc_threshold': current_perc_threshold,
                    'wavelength': current_wavelength,
                    'detected_spots': num_spots
                }
                
            elif choice == '5':
                # Skip folder
                plt.close(fig)
                return None
                
            elif choice.lower() == 'q':
                # Quit
                plt.close(fig)
                return 'quit'
                
            else:
                print("Invalid choice, please try again")
    
    def save_threshold_parameters(self, output_file: str = 'threshold_parameters.txt'):
        """Save threshold parameters to file for batch_analysis.sh"""
        output_path = Path(output_file)
        
        # Save as JSON for easy parsing
        json_output = output_path.with_suffix('.json')
        with open(json_output, 'w') as f:
            json.dump(self.threshold_results, f, indent=2)
        
        # Also save as text file for batch script
        with open(output_path, 'w') as f:
            f.write("# Threshold parameters for pyBayerSMLM batch analysis\n")
            f.write("# Generated by interactive_threshold_tuner.py\n")
            f.write("# Format: folder_path|pfa|perc_threshold|wavelength\n")
            f.write("#\n")
            
            for folder_path, params in self.threshold_results.items():
                f.write(f"{folder_path}|{params['pfa']:.0e}|{params['perc_threshold']:.1f}|{params['wavelength']:.3f}\n")
        
        print(f"\nThreshold parameters saved to:")
        print(f"  JSON format: {json_output}")
        print(f"  Text format: {output_path}")
        print(f"  Total folders configured: {len(self.threshold_results)}")
    
    def run(self):
        """Main interactive loop"""
        print("="*80)
        print("pyBayerSMLM Interactive Threshold Tuner")
        print("="*80)
        print("This tool helps determine optimal spot detection parameters")
        print("for each folder in your batch analysis workflow.\n")
        
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
            if response != 'y':
                print("Aborted by user")
                return
        else:
            print("No folders found to process!")
            return
        
        # Process each folder
        for i, (folder_path, folder_type, default_wavelength) in enumerate(existing_folders):
            print(f"\nProgress: {i+1}/{len(existing_folders)}")
            
            result = self.interactive_parameter_tuning(folder_path, folder_type, default_wavelength)
            
            if result == 'quit':
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
            print(f"Folders processed: {len(self.threshold_results)}/{len(existing_folders)}")
            
            # Show parameter distribution
            pfa_values = [params['pfa'] for params in self.threshold_results.values()]
            perc_values = [params['perc_threshold'] for params in self.threshold_results.values()]
            
            print(f"PFA range: {min(pfa_values):.0e} - {max(pfa_values):.0e}")
            print(f"Percentile threshold range: {min(perc_values):.1f}% - {max(perc_values):.1f}%")
            
        else:
            print("\nNo parameters were saved.")


if __name__ == "__main__":
    # Check if virtual environment is activated
    if '/pyBayerSMLM/' not in sys.executable:
        print("Warning: pyBayerSMLM virtual environment may not be activated")
        print("Please run: source /home/jbeckwith/.virtualenvs/pyBayerSMLM/bin/activate")
    
    tuner = InteractiveThresholdTuner()
    tuner.run()