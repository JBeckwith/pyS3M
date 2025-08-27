#!/usr/bin/env python3
"""
Optimized All Analysis OneBook Script with Memory Management and 28-Core Parallelization
Date: August 27, 2025
Author: Claude Code Analysis

Features:
- Memory cleanup with garbage collection
- 28-core parallelization
- Proper logging with flush statements  
- Preserves essential recursion for server communication
- Compact, organized structure
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl
import tifffile as tiff
from tifffile import imwrite, imread
from copy import deepcopy
import os
import sys
import time
import pandas as pd
from datetime import datetime, timedelta
import shutil
import types
import gc
import logging
from multiprocessing import Pool, cpu_count
from functools import partial
from io import BytesIO
from PIL import Image

# Setup logging with proper flushing
log_filename = f"analysis_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'
)

def log_and_flush(message, level='info'):
    """Log message and immediately flush to file"""
    if level.lower() == 'info':
        logging.info(message)
    elif level.lower() == 'error':
        logging.error(message)
    elif level.lower() == 'warning':
        logging.warning(message)
    
    # Force flush
    for handler in logging.getLogger().handlers:
        handler.flush()

def print_status(message):
    """Print single status line with carriage return (overwrites previous line)"""
    print(f"\r{message}", end='', flush=True)

# Add path and import modules
sys.path.append("..")
log_and_flush("Initializing modules...")

from src import IOFunctions, Multicolour_Simulation_Functions, PlottingFunctions
from src import ImageAnalysisFunctions, sCMOSFunctions, PSFFunctions
from src import SpectralFunctions, MaskFunctions, SpotDetectionFunctions, SR_Functions
import HelperFunctions

# Initialize function objects
IO = IOFunctions.IO_Functions()
MSF = Multicolour_Simulation_Functions.MultiC_Sim_Funcs()
plotter = PlottingFunctions.Plotter()
I_AF = ImageAnalysisFunctions.Image_Analysis_Functions()
sCMOS = sCMOSFunctions.sCMOS_Functions()
PSF = PSFFunctions.PSF_Functions()
S_F = SpectralFunctions.Spectral_Funcs()
M_F = MaskFunctions.Mask_Functions()
SD_F = SpotDetectionFunctions.SpotDetection_Functions()
SupRes_F = SR_Functions.SuperRes_Functions()
H_F = HelperFunctions.Helper_Functions()

log_and_flush("Modules initialized successfully")

# Load camera parameters
log_and_flush("Loading camera parameters...")
data_folder = '../Camera_Calibrations/Ximea_Camera'
gain_map = IO.read_tiff(os.path.join(data_folder, "gain.tif"))
offset_map = IO.read_tiff(os.path.join(data_folder, "offset.tif"))
variance = IO.read_tiff(os.path.join(data_folder, "variance.tif"))
read_noise = IO.read_tiff(os.path.join(data_folder, "readnoise.tif"))
rqe = IO.read_tiff(os.path.join(data_folder, "rqe.tif"))
R, G, B, wavelength = S_F.getpixelefficiency()

pixel_QYs = np.vstack([B, G, R])
camera_parameters = {
    "pixel_QYs": pixel_QYs,
    "pixel_order": ['B', 'G', 'R'],
    "pixel_order_indices": [0, 1, 2]
}

# Setup smoothing function
smoothing_function = types.SimpleNamespace()
smoothing_function.args = {"sigma": 1.5}
smoothing_function.extent = 1.5
smoothing_function.smoothing_function = sCMOS.gaussian_filter_stack
smoothing_function.data_arg = "image"

log_and_flush("Camera parameters loaded successfully")

def copy_file_to_scratch(file, new_folder):
    """Copy file to scratch with essential recursion for server communication"""
    try:
        os.makedirs(new_folder, exist_ok=True)
        new_file = os.path.join(new_folder, os.path.split(file)[-1])
        shutil.copyfile(file, new_file)
    except Exception as e:
        log_and_flush(f"Copy error for {file}: {e}, retrying in 10s", 'warning')
        time.sleep(10)
        copy_file_to_scratch(file, new_folder)  # Essential recursion for server communication

def copy_file_from_scratch(file, new_file):
    """Copy file from scratch with essential recursion for server communication"""
    try:
        os.makedirs(os.path.split(new_file)[0], exist_ok=True)
        shutil.copyfile(file, new_file)
    except Exception as e:
        log_and_flush(f"Copy from scratch error for {file}: {e}, retrying in 10s", 'warning')
        time.sleep(10)
        copy_file_from_scratch(file, new_file)  # Essential recursion for server communication

def delete_folder(folder):
    """Delete folder with essential recursion for server communication"""
    try:
        shutil.rmtree(folder)
    except Exception as e:
        log_and_flush(f"Delete error for {folder}: {e}, retrying in 10s", 'warning')
        time.sleep(10)
        delete_folder(folder)  # Essential recursion for server communication

def copy_folder_to_scratch(files, new_folder):
    """Copy folder to scratch using 28-core parallelization"""
    print_status(f"Copying {len(files)} files to {new_folder} using 28 cores")
    log_and_flush(f"Copying {len(files)} files to {new_folder} using 28 cores")
    
    # Use 28 cores for copying
    copy_func = partial(copy_file_to_scratch, new_folder=new_folder)
    
    with Pool(processes=28) as pool:
        pool.map(copy_func, files)
    
    print_status(f"Completed copying {len(files)} files")
    log_and_flush(f"Completed copying {len(files)} files")

def copy_from_scratch(new_folder, folder, filetype='.h5'):
    """Copy results from scratch back to original location"""
    files = np.sort([x for x in os.listdir(new_folder) if filetype in x])
    print_status(f"Copying back {len(files)} {filetype} files from scratch")
    log_and_flush(f"Copying back {len(files)} {filetype} files from scratch")
    
    for file in files:
        copy_file_from_scratch(
            os.path.join(new_folder, file), 
            os.path.join(folder, file)
        )

def should_analyse_folder(folder_path, cutoff_time=None):
    """
    Check if folder should be analysed based on .h5 file timestamps.
    
    Args:
        folder_path (str): Path to the folder to check
        cutoff_time (datetime, optional): Cutoff time. If None, uses 10am on August 26th, 2025
        
    Returns:
        bool: True if folder should be analysed
    """
    if cutoff_time is None:
        cutoff_time = datetime(2025, 8, 26, 10, 0, 0)
    
    h5_files = [f for f in os.listdir(folder_path) if f.endswith('.h5')]
    
    if not h5_files:
        log_and_flush(f"No .h5 files found in {folder_path} - proceeding with analysis")
        return True
    
    for h5_file in h5_files:
        file_path = os.path.join(folder_path, h5_file)
        file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
        
        if file_mtime >= cutoff_time:
            log_and_flush(f"Found .h5 file {h5_file} from {file_mtime} (after {cutoff_time}) - skipping analysis")
            return False
    
    log_and_flush(f"All .h5 files in {folder_path} are from before {cutoff_time} - proceeding with analysis")
    return True

def process_folder_sm_data(folder_args):
    """Process single molecule data folder with memory cleanup"""
    image_folder = folder_args
    
    try:
        if not should_analyse_folder(image_folder):
            return f"Skipped: {image_folder}"
            
        log_and_flush(f"Processing SM data: {image_folder}")
        
        new_folder = os.path.join('/scratch2/jsb92', os.path.split(image_folder)[-1])
        files_in_folder = [
            os.path.join(image_folder, x) 
            for x in os.listdir(image_folder) 
            if '.h5' not in x
        ]
        
        copy_folder_to_scratch(files_in_folder, new_folder)
        
        SupRes_F.fit_SM_data(
            new_folder,
            smoothing_function,
            gain_map,
            offset_map,
            rqe,
            read_noise,
            variance=variance,
            pfa=1e-4,
            ROI_size=12,
            peak_wavelength=0.6,
            NA=1.49,
            pixel_size=0.069,
            image_type=".tif",
        )
        
        copy_from_scratch(new_folder, image_folder, filetype='.h5')
        delete_folder(new_folder)
        
        # Memory cleanup
        gc.collect()
        
        log_and_flush(f"Completed SM data: {image_folder}")
        return f"Success: {image_folder}"
        
    except Exception as e:
        log_and_flush(f"Error processing {image_folder}: {e}", 'error')
        return f"Error: {image_folder} - {e}"

def process_folder_imaging_data(folder_args):
    """Process imaging data folder with memory cleanup"""
    image_folder, peak_wavelength = folder_args
    
    try:
        if not should_analyse_folder(image_folder):
            return f"Skipped: {image_folder}"
            
        log_and_flush(f"Processing imaging data: {image_folder}")
        
        new_folder = os.path.join('/scratch2/jsb92', os.path.split(image_folder)[-1])
        files_in_folder = [
            os.path.join(image_folder, x) 
            for x in os.listdir(image_folder) 
            if '.h5' not in x
        ]
        
        copy_folder_to_scratch(files_in_folder, new_folder)
        
        SupRes_F.fit_imaging_data(
            new_folder,
            smoothing_function,
            gain_map,
            offset_map,
            rqe,
            read_noise,
            variance=variance,
            pfa=1e-4,
            ROI_size=12,
            peak_wavelength=peak_wavelength,
            NA=1.49,
            pixel_size=0.069,
            image_type=".tif",
        )
        
        copy_from_scratch(new_folder, image_folder, filetype='.h5')
        delete_folder(new_folder)
        
        # Memory cleanup
        gc.collect()
        
        log_and_flush(f"Completed imaging data: {image_folder}")
        return f"Success: {image_folder}"
        
    except Exception as e:
        log_and_flush(f"Error processing {image_folder}: {e}", 'error')
        return f"Error: {image_folder} - {e}"

def get_lowest_dirs(starting_directory):
    """Get all lowest level directories"""
    lowest_dirs = []
    for root, dirs, files in os.walk(starting_directory):
        if not dirs:
            lowest_dirs.append(root)
    return np.sort(lowest_dirs)

def main():
    """Main analysis pipeline with 28-core parallelization and memory management"""
    
    log_and_flush("=" * 50)
    log_and_flush("Starting All_Analysis_OneBook_Optimized")
    log_and_flush("=" * 50)
    
    # Define all folder sets
    dye_folders = np.array([
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250819_TetraspeckCalibration',
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250717 BiotinDyes/ATTO488_50PM_PCA_PCD',
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250725 biotinylated dyes/ATTO514_50pM_PCAPCDTx',
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250725 biotinylated dyes/ATTO520_50pM_PCAPCDTx',
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250725 biotinylated dyes/ATTORho6G_50pM_PCAPCDTx',
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250714_BiotinylatedDyes/Atto565_PCA_PCD_Tx_50pMDye',
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250714_BiotinylatedDyes/Atto620_PCA_PCD_Tx_50pMDye',
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250711 Biotinylated Dyes/Atto633_PCA_PCD_Tx_100pMDye',
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250714_BiotinylatedDyes/Atto647N_PCA_PCD_Tx_20pMDye',
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/29250717 BiotinDyes/ATTO655_50PM _PCA_PCD',
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/29250717 BiotinDyes/ATTO700_50PM _PCA_PCD',
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/JSB/20250609_dyes/data',
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250714_BiotinylatedDyes/Atto594_PCA_PCD_Tx_50pMDye'
    ])
    
    hela_folders = [
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250523_HeLa_STORM/Cell3_HILO_190mW_638_ximea638_setting/Lp638_190_mw_40ms_exosure_HILO_1',
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250523_HeLa_STORM/Cell4_HILO_190mW_638_ximea638_setting/Lp638_190_mw_40ms_exosure_HILO_1',
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250523_HeLa_STORM/Cell2_HILO_190mW_638_ximea638_setting/Lp638_190_mw_40ms_exosure_HILO_1',
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250523_HeLa_STORM/Cell1_HILO_190mW_638_ximea638_setting/Lp638_190_mw_40ms_exosure_HILO_2'
    ]
    
    origami_folders = np.array([
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
    ])
    
    # Process 1: Dye folders with SM data
    log_and_flush(f"Processing {len(dye_folders)} dye folder hierarchies for SM data...")
    all_sm_folders = []
    for starting_directory in dye_folders:
        all_sm_folders.extend(get_lowest_dirs(starting_directory))
    
    log_and_flush(f"Found {len(all_sm_folders)} SM folders to process")
    
    # Use sequential processing to avoid memory overload, but with parallelized copying
    results = []
    for folder in all_sm_folders:
        result = process_folder_sm_data(folder)
        results.append(result)
        print_status(f"SM Progress: {len(results)}/{len(all_sm_folders)}")
    
    log_and_flush("SM data processing complete")
    
    # Process 2: HeLa folders (imaging data, wavelength 0.647)
    log_and_flush(f"Processing {len(hela_folders)} HeLa folders...")
    for folder in hela_folders:
        result = process_folder_imaging_data((folder, 0.647))
        log_and_flush(f"HeLa result: {result}")
    
    # Process 3: Origami folders (imaging data, wavelength 0.55)  
    log_and_flush(f"Processing {len(origami_folders)} origami folders...")
    for folder in origami_folders:
        result = process_folder_imaging_data((folder, 0.55))
        log_and_flush(f"Origami result: {result}")
    
    # Process 4: CellPAINT data
    log_and_flush("Processing CellPAINT data...")
    cellpaint_dir = '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/JSB/20250414_CellPAINT/data'
    cellpaint_folders = get_lowest_dirs(cellpaint_dir)
    cellpaint_folders = [x for x in cellpaint_folders if 'WL_image' not in x and 'WL_Image' not in x]
    
    for folder in cellpaint_folders:
        result = process_folder_imaging_data((folder, 0.6))
        log_and_flush(f"CellPAINT result: {result}")
    
    # Process 5: Ximea data  
    log_and_flush("Processing Ximea data...")
    ximea_dir = '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250404_Ximea_AsynNRThX/data'
    ximea_folders = get_lowest_dirs(ximea_dir)
    
    for folder in ximea_folders:
        result = process_folder_imaging_data((folder, 0.6))
        log_and_flush(f"Ximea result: {result}")
    
    # Process 6: DNA Origami data (note: original had bug using wrong starting_directory)
    log_and_flush("Processing additional DNA Origami data...")
    origami_dir = '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250818_DNAOrigami'
    if os.path.exists(origami_dir):
        origami2_folders = get_lowest_dirs(origami_dir)
        
        for folder in origami2_folders:
            result = process_folder_imaging_data((folder, 0.6))
            log_and_flush(f"DNA Origami 2 result: {result}")
    else:
        log_and_flush(f"DNA Origami directory {origami_dir} not found", 'warning')
    
    # Final memory cleanup
    gc.collect()
    
    # Clear status line and show completion
    print()  # Move to new line after status updates
    log_and_flush("=" * 50)
    log_and_flush("All_Analysis_OneBook_Optimized completed successfully")
    log_and_flush(f"Log file: {log_filename}")
    log_and_flush("=" * 50)

if __name__ == "__main__":
    main()