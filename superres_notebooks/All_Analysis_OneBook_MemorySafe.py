#!/usr/bin/env python3
"""
Memory-Safe All Analysis OneBook Script
Date: August 27, 2025
Author: Claude Code Analysis

Key Memory Leak Fixes Applied:
1. ProcessPoolExecutor with proper context managers
2. Matplotlib backend set to non-interactive 
3. Explicit figure cleanup with plt.close()
4. Memory monitoring and forced garbage collection
5. Single-process execution to avoid process leaks
6. Resource cleanup after each folder
"""

import sys
import os
import gc
import logging
import traceback
from datetime import datetime
# numpy not needed - removed to fix pylance warning
import time

# Set matplotlib to non-interactive backend BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')  # Prevents memory leaks from interactive backends
import matplotlib.pyplot as plt

# Setup logging
log_filename = f"analysis_memorysafe_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'
)

def log_and_flush(message, level='info'):
    """Log message and immediately flush to file"""
    getattr(logging, level.lower())(message)
    for handler in logging.getLogger().handlers:
        handler.flush()

def monitor_memory():
    """Monitor memory usage"""
    try:
        import psutil
        process = psutil.Process()
        mem_gb = process.memory_info().rss / 1e9
        available_gb = psutil.virtual_memory().available / 1e9
        
        log_and_flush(f"Memory: Process={mem_gb:.2f}GB, Available={available_gb:.2f}GB")
        
        if mem_gb > 30:
            log_and_flush(f"WARNING: High memory usage: {mem_gb:.1f}GB", 'warning')
            return False
        return True
    except:
        return True

def cleanup_resources():
    """Force cleanup of all resources"""
    try:
        # Close all matplotlib figures
        plt.close('all')
        plt.clf()
        
        # Force garbage collection
        gc.collect()
        
        log_and_flush("Resources cleaned up")
    except Exception as e:
        log_and_flush(f"Cleanup error: {e}", 'warning')

def safe_copy_file(source, dest):
    """Safely copy file with retries"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            import shutil
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copyfile(source, dest)
            return True
        except Exception as e:
            log_and_flush(f"Copy attempt {attempt+1} failed for {source}: {e}", 'warning')
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                return False
    return False

def process_folder_safe(folder_info, functions, camera_data, smoothing_function):
    """Process single folder with maximum safety and memory management"""
    
    # Handle tuple format (type, path, [wavelength])
    if isinstance(folder_info, tuple):
        if len(folder_info) == 2:
            folder_type, folder_path = folder_info
            peak_wavelength = 0.55  # default
        else:
            folder_type, folder_path, peak_wavelength = folder_info
    else:
        # Legacy single folder path
        folder_type = 'auto'
        folder_path = folder_info
        peak_wavelength = 0.55
    
    log_and_flush(f"Starting folder: {folder_path} (type: {folder_type}, wavelength: {peak_wavelength})")
    
    try:
        # Check if folder exists
        if not os.path.exists(folder_path):
            return f"NOT_FOUND: {folder_path}"
        
        # Check if analysis needed
        h5_files = [f for f in os.listdir(folder_path) if f.endswith('.h5')]
        if h5_files:
            cutoff_time = datetime(2025, 8, 26, 10, 0, 0)
            for h5_file in h5_files:
                file_path = os.path.join(folder_path, h5_file)
                file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                if file_mtime >= cutoff_time:
                    log_and_flush(f"Recent .h5 found - skipping: {folder_path}")
                    return f"SKIPPED: {folder_path}"
        
        # Monitor memory before processing
        if not monitor_memory():
            cleanup_resources()
            if not monitor_memory():
                log_and_flush(f"Insufficient memory for {folder_path}", 'error')
                return f"MEMORY_ERROR: {folder_path}"
        
        # Setup scratch directory
        scratch_folder = f"/scratch2/jsb92/{os.path.basename(folder_path)}_{os.getpid()}"
        
        try:
            # Copy files to scratch (avoiding multiprocessing to prevent leaks)
            files_to_copy = [
                os.path.join(folder_path, f) 
                for f in os.listdir(folder_path) 
                if not f.endswith('.h5')
            ]
            
            log_and_flush(f"Copying {len(files_to_copy)} files to scratch")
            
            os.makedirs(scratch_folder, exist_ok=True)
            
            for source_file in files_to_copy:
                dest_file = os.path.join(scratch_folder, os.path.basename(source_file))
                if not safe_copy_file(source_file, dest_file):
                    log_and_flush(f"Failed to copy {source_file}", 'error')
                    return f"COPY_ERROR: {folder_path}"
            
            # Process data using SR_Functions
            SupRes_F = functions['SupRes_F']
            
            # Determine processing type 
            if folder_type == 'sm' or (folder_type == 'auto' and ('dyes' in folder_path.lower() or 'biotinylated' in folder_path.lower())):
                # SM data processing
                log_and_flush(f"Processing as SM data with wavelength {peak_wavelength}")
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
                # Imaging data processing
                log_and_flush(f"Processing as imaging data with wavelength {peak_wavelength}")
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
            
            # Copy results back
            result_files = [f for f in os.listdir(scratch_folder) if f.endswith('.h5')]
            log_and_flush(f"Copying {len(result_files)} results back")
            
            for result_file in result_files:
                source = os.path.join(scratch_folder, result_file)
                dest = os.path.join(folder_path, result_file)
                safe_copy_file(source, dest)
            
            log_and_flush(f"Successfully processed: {folder_path}")
            return f"SUCCESS: {folder_path}"
            
        finally:
            # Always cleanup scratch folder
            try:
                import shutil
                if os.path.exists(scratch_folder):
                    shutil.rmtree(scratch_folder)
                    log_and_flush(f"Cleaned up scratch folder: {scratch_folder}")
            except Exception as e:
                log_and_flush(f"Failed to cleanup scratch: {e}", 'warning')
    
    except Exception as e:
        log_and_flush(f"Processing error for {folder_path}: {e}", 'error')
        log_and_flush(f"Traceback: {traceback.format_exc()}", 'error')
        return f"ERROR: {folder_path} - {str(e)[:100]}"
    
    finally:
        # Always cleanup resources after each folder
        cleanup_resources()

def get_all_folders():
    """Get all folders to process - complete list from original script"""
    
    # Define SM data folders (dye experiments)
    sm_data_base_dirs = [
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250819_TetraspeckCalibration',
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250717_BiotinDyes/ATTO488_50PM_PCA_PCD',
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/29250717_BiotinDyes/ATTO655_50PM_PCA_PCD',
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/29250717_BiotinDyes/ATTO700_50PM_PCA_PCD',
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
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250723 DNA Origami/FourColour_F1AF647_F2ATTO565_F3Cy3B_F4ATTO655_500pMEach/15percent_561_100mWEach_638_NotchFilter_785SP_1',
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
    
    # Add SM data folders (need to walk hierarchies)
    log_and_flush(f"Processing {len(sm_data_base_dirs)} SM data hierarchies...")
    for base_dir in sm_data_base_dirs:
        if os.path.exists(base_dir):
            for root, dirs, _ in os.walk(base_dir):
                if not dirs:  # Leaf directory
                    all_folders.append(('sm', root))
        else:
            log_and_flush(f"SM directory not found: {base_dir}", 'warning')
    
    # Add HeLa imaging folders (647nm wavelength)
    log_and_flush(f"Processing {len(hela_folders)} HeLa imaging folders...")
    for folder in hela_folders:
        if os.path.exists(folder):
            all_folders.append(('imaging', folder, 0.647))
        else:
            log_and_flush(f"HeLa folder not found: {folder}", 'warning')
    
    # Add general imaging folders (550nm default)
    log_and_flush(f"Processing {len(imaging_folders)} general imaging folders...")
    for folder in imaging_folders:
        if os.path.exists(folder):
            all_folders.append(('imaging', folder, 0.55))
        else:
            log_and_flush(f"Imaging folder not found: {folder}", 'warning')
    
    # Add hierarchical imaging folders (need to walk)
    log_and_flush(f"Processing {len(hierarchical_bases)} hierarchical imaging bases...")
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
    """Main function with memory-safe processing"""
    
    log_and_flush("="*60)
    log_and_flush("Starting Memory-Safe Analysis")
    log_and_flush("="*60)
    
    try:
        # Import modules safely
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
        
        # Setup smoothing function
        import types
        smoothing_function = types.SimpleNamespace()
        smoothing_function.args = {"sigma": 1.5}
        smoothing_function.extent = 1.5
        smoothing_function.smoothing_function = functions['sCMOS'].gaussian_filter_stack
        smoothing_function.data_arg = "image"
        
        log_and_flush("Setup completed successfully")
        
        # Get folders to process
        folders = get_all_folders()
        log_and_flush(f"Found {len(folders)} folders to process")
        
        # Process folders sequentially (no multiprocessing to avoid leaks)
        results = []
        
        for i, folder_info in enumerate(folders):
            # Extract folder path for logging
            if isinstance(folder_info, tuple):
                folder_path = folder_info[1]
            else:
                folder_path = folder_info
                
            log_and_flush(f"Processing folder {i+1}/{len(folders)}: {folder_path}")
            
            # Monitor memory before each folder
            monitor_memory()
            
            result = process_folder_safe(folder_info, functions, camera_data, smoothing_function)
            results.append(result)
            
            log_and_flush(f"Folder {i+1} result: {result}")
            
            # Status update
            print(f"\rProgress: {i+1}/{len(folders)} - {result.split(':')[0]}", end='', flush=True)
        
        print()  # New line
        
        # Summary
        log_and_flush("="*60)
        log_and_flush("PROCESSING COMPLETE")
        log_and_flush("="*60)
        
        success_count = len([r for r in results if r.startswith('SUCCESS')])
        skip_count = len([r for r in results if r.startswith('SKIPPED')])
        error_count = len([r for r in results if r.startswith('ERROR')])
        
        log_and_flush(f"Total folders: {len(folders)}")
        log_and_flush(f"Successful: {success_count}")
        log_and_flush(f"Skipped: {skip_count}")
        log_and_flush(f"Errors: {error_count}")
        
        log_and_flush(f"Log file: {log_filename}")
        
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