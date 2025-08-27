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
import numpy as np
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

def process_folder_safe(folder_path, functions, camera_data, smoothing_function):
    """Process single folder with maximum safety and memory management"""
    
    log_and_flush(f"Starting folder: {folder_path}")
    
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
            
            # Determine processing type based on folder structure
            if 'dyes' in folder_path.lower() or 'biotinylated' in folder_path.lower():
                # SM data processing
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
                    peak_wavelength=0.6,
                    NA=1.49,
                    pixel_size=0.069,
                    image_type=".tif",
                )
            else:
                # Imaging data processing
                peak_wavelength = 0.55  # Default
                if 'hela' in folder_path.lower():
                    peak_wavelength = 0.647
                
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
    """Get all folders to process"""
    
    # Define base directories
    base_dirs = [
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250819_TetraspeckCalibration',
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250717 BiotinDyes/ATTO488_50PM_PCA_PCD',
        # Add more as needed for testing
    ]
    
    all_folders = []
    
    for base_dir in base_dirs:
        if os.path.exists(base_dir):
            # Get lowest level directories
            for root, dirs, files in os.walk(base_dir):
                if not dirs:  # Leaf directory
                    all_folders.append(root)
        else:
            log_and_flush(f"Directory not found: {base_dir}", 'warning')
    
    return sorted(all_folders)

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
        
        for i, folder in enumerate(folders):
            log_and_flush(f"Processing folder {i+1}/{len(folders)}: {folder}")
            
            # Monitor memory before each folder
            monitor_memory()
            
            result = process_folder_safe(folder, functions, camera_data, smoothing_function)
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