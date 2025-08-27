#!/usr/bin/env python3
"""
File Descriptor Safe Analysis Script
Date: August 27, 2025
Author: Claude Code Analysis

Specifically addresses file descriptor limit crashes that cause terminal exits.
Key fixes:
1. Monitors and increases file descriptor limits
2. Closes files explicitly after each operation
3. Sequential processing only (no multiprocessing)
4. File descriptor leak detection and cleanup
5. Enhanced logging for crash diagnosis
"""

import sys
import os
import gc
import logging
import traceback
import resource
from datetime import datetime
import time

# CRITICAL: Increase file descriptor limit immediately
try:
    # Get current limits
    fd_soft, fd_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    print(f"Initial FD limit: soft={fd_soft}, hard={fd_hard}")
    
    # Set soft limit to a reasonable value (but not exceeding hard limit)
    new_soft = min(65536, fd_hard)
    resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, fd_hard))
    
    # Verify the change
    fd_soft_new, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    print(f"Updated FD limit: soft={fd_soft_new}")
    
    if fd_soft_new < 4096:
        print(f"WARNING: FD limit still low ({fd_soft_new}). May cause crashes!")
        
except Exception as e:
    print(f"ERROR: Could not increase FD limit: {e}")
    print("This may cause terminal crashes. Consider running 'ulimit -n 65536' first.")

# Set matplotlib to non-interactive backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Setup logging with file descriptor monitoring
log_filename = f"analysis_fdsafe_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'
)

def log_and_flush(message, level='info'):
    """Log with immediate flush"""
    try:
        getattr(logging, level.lower())(message)
        # Also print to console for real-time monitoring
        print(f"[{level.upper()}] {message}")
        for handler in logging.getLogger().handlers:
            handler.flush()
    except Exception as e:
        print(f"Logging error: {e}")

def monitor_file_descriptors():
    """Monitor file descriptor usage - CRITICAL for preventing crashes"""
    try:
        import psutil
        process = psutil.Process()
        
        try:
            num_fds = process.num_fds()
            fd_soft, fd_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            
            usage_percent = (num_fds / fd_soft) * 100
            
            log_and_flush(f"File Descriptors: {num_fds}/{fd_soft} ({usage_percent:.1f}%)")
            
            if usage_percent > 80:
                log_and_flush(f"CRITICAL: High FD usage {num_fds}/{fd_soft}!", 'error')
                return False
            elif usage_percent > 60:
                log_and_flush(f"WARNING: Moderate FD usage {num_fds}/{fd_soft}", 'warning')
            
            return True
            
        except AttributeError:
            # num_fds() not available on this system
            log_and_flush("FD monitoring not available on this system", 'warning')
            return True
            
    except Exception as e:
        log_and_flush(f"FD monitoring error: {e}", 'warning')
        return True

def cleanup_resources():
    """Enhanced cleanup with FD leak prevention"""
    try:
        # Close matplotlib figures
        plt.close('all')
        plt.clf()
        
        # Force garbage collection
        for _ in range(3):
            gc.collect()
        
        # Log FD usage after cleanup
        monitor_file_descriptors()
        
    except Exception as e:
        log_and_flush(f"Cleanup error: {e}", 'warning')

def fd_safe_process_folder(folder_info, functions, camera_data, smoothing_function):
    """File descriptor safe folder processing"""
    
    # Handle folder info tuple
    if isinstance(folder_info, tuple):
        if len(folder_info) == 2:
            folder_type, folder_path = folder_info
            peak_wavelength = 0.55
        else:
            folder_type, folder_path, peak_wavelength = folder_info
    else:
        folder_type = 'auto'
        folder_path = folder_info
        peak_wavelength = 0.55
    
    log_and_flush(f"FD-SAFE: Processing {folder_path} (type: {folder_type})")
    
    # Monitor FDs before processing
    if not monitor_file_descriptors():
        cleanup_resources()
        if not monitor_file_descriptors():
            return f"FD_LIMIT_ERROR: {folder_path}"
    
    try:
        # Check folder exists
        if not os.path.exists(folder_path):
            return f"NOT_FOUND: {folder_path}"
        
        # Check if already analyzed (skip recent .h5 files)
        h5_files = [f for f in os.listdir(folder_path) if f.endswith('.h5')]
        if h5_files:
            cutoff_time = datetime(2025, 8, 26, 10, 0, 0)
            for h5_file in h5_files:
                file_path = os.path.join(folder_path, h5_file)
                file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                if file_mtime >= cutoff_time:
                    return f"SKIPPED: {folder_path}"
        
        # Setup scratch with better error handling
        scratch_folder = f"/scratch2/jsb92/{os.path.basename(folder_path)}_{os.getpid()}_{int(time.time())}"
        
        try:
            # Get list of files to copy
            try:
                all_files = os.listdir(folder_path)
                files_to_copy = [
                    os.path.join(folder_path, f) 
                    for f in all_files 
                    if not f.endswith('.h5')
                ]
            except OSError as e:
                log_and_flush(f"Failed to list files in {folder_path}: {e}", 'error')
                return f"LIST_ERROR: {folder_path}"
            
            log_and_flush(f"Copying {len(files_to_copy)} files to scratch")
            
            # Create scratch directory
            try:
                os.makedirs(scratch_folder, exist_ok=True)
            except OSError as e:
                log_and_flush(f"Failed to create scratch folder {scratch_folder}: {e}", 'error')
                return f"SCRATCH_CREATE_ERROR: {folder_path}"
            
            # Copy files one by one with FD monitoring
            for i, source_file in enumerate(files_to_copy):
                try:
                    dest_file = os.path.join(scratch_folder, os.path.basename(source_file))
                    
                    # Use explicit file operations to control FDs
                    with open(source_file, 'rb') as src:
                        with open(dest_file, 'wb') as dst:
                            # Copy in chunks to manage memory
                            while True:
                                chunk = src.read(1024 * 1024)  # 1MB chunks
                                if not chunk:
                                    break
                                dst.write(chunk)
                    
                    # Monitor FDs every 50 files
                    if (i + 1) % 50 == 0:
                        log_and_flush(f"Copied {i+1}/{len(files_to_copy)} files")
                        if not monitor_file_descriptors():
                            cleanup_resources()
                            
                except Exception as e:
                    log_and_flush(f"Failed to copy {source_file}: {e}", 'error')
                    return f"FILE_COPY_ERROR: {folder_path}"
            
            # Monitor FDs before analysis
            if not monitor_file_descriptors():
                cleanup_resources()
            
            # Run analysis (NO MULTIPROCESSING)
            try:
                SupRes_F = functions['SupRes_F']
                
                log_and_flush(f"Starting analysis for {folder_path}")
                
                if folder_type == 'sm' or (folder_type == 'auto' and ('dyes' in folder_path.lower() or 'biotinylated' in folder_path.lower())):
                    log_and_flush(f"Running SM analysis (wavelength: {peak_wavelength})")
                    SupRes_F.fit_SM_data(
                        scratch_folder,
                        smoothing_function,
                        camera_data['gain'],
                        camera_data['offset'],
                        camera_data['rqe'],
                        camera_data['readnoise'],
                        variance=camera_data['variance'],
                        pfa=1e-4,
                        ROI_size=12,
                        peak_wavelength=peak_wavelength,
                        NA=1.49,
                        pixel_size=0.069,
                        image_type=".tif",
                    )
                else:
                    log_and_flush(f"Running imaging analysis (wavelength: {peak_wavelength})")
                    SupRes_F.fit_imaging_data(
                        scratch_folder,
                        smoothing_function,
                        camera_data['gain'],
                        camera_data['offset'],
                        camera_data['rqe'],
                        camera_data['readnoise'],
                        variance=camera_data['variance'],
                        pfa=1e-4,
                        ROI_size=12,
                        peak_wavelength=peak_wavelength,
                        NA=1.49,
                        pixel_size=0.069,
                        image_type=".tif",
                    )
                
                log_and_flush(f"Analysis completed for {folder_path}")
                
            except Exception as e:
                log_and_flush(f"Analysis failed for {folder_path}: {e}", 'error')
                log_and_flush(f"Analysis traceback: {traceback.format_exc()}", 'error')
                return f"ANALYSIS_ERROR: {folder_path}"
            
            # Monitor FDs after analysis
            monitor_file_descriptors()
            
            # Copy results back
            try:
                result_files = []
                for f in os.listdir(scratch_folder):
                    if f.endswith('.h5'):
                        result_files.append(f)
                
                log_and_flush(f"Copying {len(result_files)} result files back")
                
                for result_file in result_files:
                    source = os.path.join(scratch_folder, result_file)
                    dest = os.path.join(folder_path, result_file)
                    
                    # Use explicit file operations
                    with open(source, 'rb') as src:
                        with open(dest, 'wb') as dst:
                            while True:
                                chunk = src.read(1024 * 1024)
                                if not chunk:
                                    break
                                dst.write(chunk)
                
            except Exception as e:
                log_and_flush(f"Result copy failed for {folder_path}: {e}", 'error')
                return f"RESULT_COPY_ERROR: {folder_path}"
            
            return f"SUCCESS: {folder_path}"
            
        finally:
            # Always cleanup scratch
            try:
                if os.path.exists(scratch_folder):
                    import shutil
                    shutil.rmtree(scratch_folder)
                    log_and_flush(f"Cleaned up scratch: {scratch_folder}")
            except Exception as e:
                log_and_flush(f"Scratch cleanup failed: {e}", 'warning')
    
    except Exception as e:
        log_and_flush(f"Critical error processing {folder_path}: {e}", 'error')
        log_and_flush(f"Critical traceback: {traceback.format_exc()}", 'error')
        return f"CRITICAL_ERROR: {folder_path}"
    
    finally:
        # Always cleanup resources
        cleanup_resources()

def get_all_folders():
    """Get all folders - same as other scripts"""
    
    # Define SM data folders (dye experiments)
    sm_data_base_dirs = [
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250819_TetraspeckCalibration',
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250717_BiotinDyes/ATTO488_50PM_PCA_PCD',
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250717_BiotinDyes/ATTO655_50PM_PCA_PCD',
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250717_BiotinDyes/ATTO700_50PM_PCA_PCD',
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250725 biotinylated dyes/ATTO514_50pM_PCAPCDTx',
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250725 biotinylated dyes/ATTO520_50pM_PCAPCDTx',
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250725 biotinylated dyes/ATTORho6G_50pM_PCAPCDTx',
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250714_BiotinylatedDyes/Atto565_PCA_PCD_Tx_50pMDye',
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250714_BiotinylatedDyes/Atto620_PCA_PCD_Tx_50pMDye',
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250711 Biotinylated Dyes/Atto633_PCA_PCD_Tx_100pMDye',
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250714_BiotinylatedDyes/Atto647N_PCA_PCD_Tx_20pMDye',
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/JSB/20250609_dyes/data',
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250714_BiotinylatedDyes/Atto594_PCA_PCD_Tx_50pMDye'
    ]
    
    # Define individual HeLa imaging folders
    hela_folders = [
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250523_HeLa_STORM/Cell3_HILO_190mW_638_ximea638_setting/Lp638_190_mw_40ms_exosure_HILO_1',
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250523_HeLa_STORM/Cell4_HILO_190mW_638_ximea638_setting/Lp638_190_mw_40ms_exosure_HILO_1',
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250523_HeLa_STORM/Cell2_HILO_190mW_638_ximea638_setting/Lp638_190_mw_40ms_exosure_HILO_1',
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250523_HeLa_STORM/Cell1_HILO_190mW_638_ximea638_setting/Lp638_190_mw_40ms_exosure_HILO_2'
    ]
    
    # Define origami/DNA imaging folders
    imaging_folders = [
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/JSB/20250717_Origami/F1F2F3F4Cy3B500pM/10perc561_LP561_BP586-64_1',
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/JSB/20250717_Origami/F1F2F3F4Cy3B500pM_LowConcOrigami/10perc561_LP561_BP586-64_1',
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/JSB/20250514_DNANanoruler/data/DNANanoRuler_10perc561_30mW488_50mW638/F1CF640CF550R_F2ATTO488AF647_F3ATTO565ATTO655_F4Cy3BCF488A_MultiNotch_488LP_758SP_1',
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/JSB/20250514_DNANanoruler/data/DNANanoRuler_10perc561_30mW488_50mW638/F1CF640CF550R_F2ATTO488AF647_F3ATTO565ATTO655_F4Cy3BCF488A_MultiNotch_488LP_758SP_1nM_1',
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250730 single colour origami/AlexaFluor647_2nM_strands/30mWboth638_NF_785SP_488LP_1',
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250730 single colour origami/CF488A_2nM_strands/20mW488_NF_785SP_488LP_1',
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250730 single colour origami/CF550R_2nM_strands_adjusteddichroic/30p561_NF_785SP_488LP_1',
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250730 single colour origami/CF640R_2nM_strands/30mWboth638_NF_785SP_488LP_1',
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250723 DNA Origami/FourColour_F1AF647_F2ATTO565_F3Cy3B_F4ATTO655_500pMEach/15percent_561_40mWEach_638_NotchFilter_785SP_1',
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250723 DNA Origami/FourColour_F1AF647_F2ATTO565_F3Cy3B_F4ATTO655_500pMEach/15percent_561_100mWEach_638NotchFilter_785SP_1',
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250723 DNA Origami/FourColour_F1AF647_F2ATTO565_F3Cy3B_F4CF488A_500pMEach/30mW_488_15percent_561_100mWEach_638_NotchFilter_785SP_1',
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250723 DNA Origami/FourColour_F1CF550R_F2ATTO565_F3Cy3B_F4CF488A_500pMEach/30mW_488_15percent_561_NotchFilter_785SP_1',
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/JSB/20250716_iPSCJamesEvans/40mW488_30perc561_50mW638_NF_488LP_785SP_1',
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/JSB/20250716_iPSCJamesEvans/250pMCy3B_250pM565_250pMCF550_250pM647/20perc561_40mW638_NF_488LP_785SP_1'
    ]
    
    # Define hierarchical directory bases (that need walking)
    hierarchical_bases = [
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/JSB/20250414_CellPAINT/data',
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250404_Ximea_AsynNRThX/data',
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250818_DNAOrigami'
    ]
    
    all_folders = []
    
    # Add SM data folders (walk hierarchies)
    log_and_flush(f"Walking {len(sm_data_base_dirs)} SM data hierarchies...")
    for base_dir in sm_data_base_dirs:
        if os.path.exists(base_dir):
            for root, dirs, _ in os.walk(base_dir):
                if not dirs:  # Leaf directory
                    all_folders.append(('sm', root))
        else:
            log_and_flush(f"SM directory not found: {base_dir}", 'warning')
    
    # Add HeLa folders (647nm)
    log_and_flush(f"Adding {len(hela_folders)} HeLa folders...")
    for folder in hela_folders:
        if os.path.exists(folder):
            all_folders.append(('imaging', folder, 0.647))
        else:
            log_and_flush(f"HeLa folder not found: {folder}", 'warning')
    
    # Add imaging folders (550nm)
    log_and_flush(f"Adding {len(imaging_folders)} imaging folders...")
    for folder in imaging_folders:
        if os.path.exists(folder):
            all_folders.append(('imaging', folder, 0.55))
        else:
            log_and_flush(f"Imaging folder not found: {folder}", 'warning')
    
    # Add hierarchical imaging bases
    log_and_flush(f"Walking {len(hierarchical_bases)} hierarchical bases...")
    for base_dir in hierarchical_bases:
        if os.path.exists(base_dir):
            for root, dirs, _ in os.walk(base_dir):
                if not dirs:  # Leaf directory
                    all_folders.append(('imaging', root, 0.55))
        else:
            log_and_flush(f"Hierarchical directory not found: {base_dir}", 'warning')
    
    log_and_flush(f"Total folders found: {len(all_folders)}")
    return sorted(all_folders, key=lambda x: x[1])

def main():
    """File descriptor safe main function"""
    
    log_and_flush("="*60)
    log_and_flush("FILE DESCRIPTOR SAFE ANALYSIS STARTING")
    log_and_flush("="*60)
    
    # Initial FD monitoring
    monitor_file_descriptors()
    
    try:
        # Import modules
        log_and_flush("Importing modules...")
        sys.path.append("..")
        
        from src import (IOFunctions, Multicolour_Simulation_Functions, 
                        PlottingFunctions, ImageAnalysisFunctions, 
                        sCMOSFunctions, PSFFunctions, SpectralFunctions,
                        MaskFunctions, SpotDetectionFunctions, SR_Functions)
        import HelperFunctions
        
        # Initialize functions
        log_and_flush("Initializing functions...")
        functions = {
            'IO': IOFunctions.IO_Functions(),
            'MSF': Multicolour_Simulation_Functions.MultiC_Sim_Funcs(),
            'plotter': PlottingFunctions.Plotter(),
            'I_AF': ImageAnalysisFunctions.Image_Analysis_Functions(),
            'sCMOS': sCMOSFunctions.sCMOS_Functions(),
            'PSF': PSFFunctions.PSF_Functions(),
            'S_F': SpectralFunctions.Spectral_Funcs(),
            'M_F': MaskFunctions.Mask_Functions(),
            'SD_F': SpotDetectionFunctions.SpotDetection_Functions(),
            'SupRes_F': SR_Functions.SuperRes_Functions(),
            'H_F': HelperFunctions.Helper_Functions()
        }
        
        # Monitor FDs after module loading
        monitor_file_descriptors()
        
        # Load camera parameters
        log_and_flush("Loading camera parameters...")
        data_folder = '../Camera_Calibrations/Ximea_Camera'
        
        camera_data = {
            'gain': functions['IO'].read_tiff(os.path.join(data_folder, "gain.tif")),
            'offset': functions['IO'].read_tiff(os.path.join(data_folder, "offset.tif")),
            'variance': functions['IO'].read_tiff(os.path.join(data_folder, "variance.tif")),
            'readnoise': functions['IO'].read_tiff(os.path.join(data_folder, "readnoise.tif")),
            'rqe': functions['IO'].read_tiff(os.path.join(data_folder, "rqe.tif"))
        }
        
        # Monitor FDs after loading camera data
        monitor_file_descriptors()
        
        # Setup smoothing function
        import types
        smoothing_function = types.SimpleNamespace()
        smoothing_function.args = {"sigma": 1.5}
        smoothing_function.extent = 1.5
        smoothing_function.smoothing_function = functions['sCMOS'].gaussian_filter_stack
        smoothing_function.data_arg = "image"
        
        log_and_flush("Setup completed")
        
        # Get folders
        folders = get_all_folders()
        log_and_flush(f"Found {len(folders)} folders to process")
        
        # Process folders sequentially with FD monitoring
        results = []
        
        for i, folder_info in enumerate(folders):
            folder_path = folder_info[1] if isinstance(folder_info, tuple) else folder_info
            
            log_and_flush(f"Processing folder {i+1}/{len(folders)}: {folder_path}")
            
            # Monitor FDs before each folder
            monitor_file_descriptors()
            
            result = fd_safe_process_folder(folder_info, functions, camera_data, smoothing_function)
            results.append(result)
            
            log_and_flush(f"Folder {i+1} result: {result}")
            
            # Progress update
            print(f"\rProgress: {i+1}/{len(folders)} - {result.split(':')[0]}", end='', flush=True)
            
            # Periodic FD check
            if (i + 1) % 5 == 0:
                log_and_flush(f"Completed {i+1}/{len(folders)} folders")
                monitor_file_descriptors()
        
        print()  # New line
        
        # Summary
        log_and_flush("="*60)
        log_and_flush("FD-SAFE PROCESSING COMPLETE")
        log_and_flush("="*60)
        
        success_count = len([r for r in results if r.startswith('SUCCESS')])
        skip_count = len([r for r in results if r.startswith('SKIPPED')])
        error_count = len([r for r in results if 'ERROR' in r])
        
        log_and_flush(f"Total folders: {len(folders)}")
        log_and_flush(f"Successful: {success_count}")
        log_and_flush(f"Skipped: {skip_count}")
        log_and_flush(f"Errors: {error_count}")
        
        # Final FD check
        monitor_file_descriptors()
        
        return 0
        
    except Exception as e:
        log_and_flush(f"CRITICAL ERROR: {e}", 'error')
        log_and_flush(f"Traceback: {traceback.format_exc()}", 'error')
        return 1
    
    finally:
        cleanup_resources()

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)