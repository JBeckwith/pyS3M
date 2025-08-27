#!/usr/bin/env python3
"""
Direct Analysis Script - Process files in original locations
No copying to scratch, work directly on source folders
Clean up existing .h5 files before processing
"""

import logging
import datetime
import os
import sys
import gc
import psutil
import glob
import time

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_dir = os.path.join(project_root, 'src')

if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Setup logging - reduce console spam
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"direct_analysis_{timestamp}.txt"

# File handler for detailed logging
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Console handler for essential messages only
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)  # Only warnings and errors to console
console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))

logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, console_handler]
)

def log_and_flush(message):
    """Log message to file and ensure it's written immediately"""
    logging.info(message)
    for handler in logging.root.handlers:
        handler.flush()

def print_status(message):
    """Print status message to console with carriage return (single line updates)"""
    print(f"\r{message}", end='', flush=True)

def print_and_log(message):
    """Print to console and log to file"""
    print(message)
    logging.info(message)
    for handler in logging.root.handlers:
        handler.flush()

def monitor_memory():
    """Monitor system memory usage"""
    process = psutil.Process()
    memory_info = psutil.virtual_memory()
    log_and_flush(f"Memory: Process={process.memory_info().rss / 1024**3:.2f}GB, Available={memory_info.available / 1024**3:.2f}GB")

def clean_existing_h5_files(folder_path):
    """Remove any existing .h5 files in the folder"""
    h5_files = glob.glob(os.path.join(folder_path, "*.h5"))
    
    if h5_files:
        log_and_flush(f"Found {len(h5_files)} existing .h5 files, removing...")
        for h5_file in h5_files:
            try:
                os.remove(h5_file)
                log_and_flush(f"Removed: {os.path.basename(h5_file)}")
            except Exception as e:
                log_and_flush(f"Failed to remove {h5_file}: {e}")
    else:
        log_and_flush("No existing .h5 files found")

def determine_folder_type_and_wavelength(folder_path):
    """Determine if folder is SM or imaging data and peak wavelength"""
    folder_name = os.path.basename(folder_path).lower()
    
    # SM data indicators
    if any(indicator in folder_name for indicator in ['dyes', 'biotinylated', 'sm_']):
        return 'sm', 0.638  # Default wavelength for SM data
    
    # Extract wavelength from folder name if possible
    if '488' in folder_name:
        return 'imaging', 0.488
    elif '561' in folder_name:
        return 'imaging', 0.561
    elif '638' in folder_name or '640' in folder_name:
        return 'imaging', 0.638
    elif '405' in folder_name:
        return 'imaging', 0.405
    else:
        return 'imaging', 0.55  # Default wavelength

def process_folder_direct(folder_path, functions, camera_data, smoothing_function, folder_type=None, peak_wavelength=None):
    """Process folder directly without copying files"""
    
    try:
        log_and_flush(f"Starting folder: {folder_path}")
        monitor_memory()
        
        # Clean existing H5 files
        clean_existing_h5_files(folder_path)
        
        # Check if folder has the required files
        if not any(f.endswith('.tif') for f in os.listdir(folder_path)):
            return f"SKIP: No .tif files found in {folder_path}"
        
        metadata_files = [f for f in os.listdir(folder_path) if 'metadata' in f.lower()]
        if not metadata_files:
            return f"SKIP: No metadata files found in {folder_path}"
        
        # Determine processing type and wavelength if not provided
        if folder_type is None or peak_wavelength is None:
            folder_type, peak_wavelength = determine_folder_type_and_wavelength(folder_path)
        log_and_flush(f"Processing as {folder_type} data with wavelength {peak_wavelength}")
        
        # Process data using SR_Functions
        SupRes_F = functions['SupRes_F']
        
        if folder_type == 'sm':
            # SM data processing
            try:
                log_and_flush("About to call SupRes_F.fit_SM_data...")
                SupRes_F.fit_SM_data(
                    folder_path,  # Work directly on original folder
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
                log_and_flush("SupRes_F.fit_SM_data completed successfully")
            except Exception as e:
                import traceback
                log_and_flush(f"ERROR in fit_SM_data: {str(e)}")
                log_and_flush(f"Exception type: {type(e).__name__}")
                log_and_flush(f"Full traceback: {traceback.format_exc()}")
                return f"ERROR: {str(e)}"
        else:
            # Imaging data processing
            try:
                log_and_flush("About to call SupRes_F.fit_imaging_data...")
                SupRes_F.fit_imaging_data(
                    folder_path,  # Work directly on original folder
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
                log_and_flush("SupRes_F.fit_imaging_data completed successfully")
            except Exception as e:
                import traceback
                log_and_flush(f"ERROR in fit_imaging_data: {str(e)}")
                log_and_flush(f"Exception type: {type(e).__name__}")
                log_and_flush(f"Full traceback: {traceback.format_exc()}")
                return f"ERROR: {str(e)}"
        
        # Check if output files were created
        h5_files = glob.glob(os.path.join(folder_path, "*.h5"))
        if h5_files:
            log_and_flush(f"Analysis completed - created {len(h5_files)} .h5 files")
            return f"SUCCESS: Created {len(h5_files)} .h5 files"
        else:
            log_and_flush("WARNING: No .h5 files created")
            return "WARNING: No output files created"
        
    except Exception as e:
        import traceback
        log_and_flush(f"FOLDER ERROR: {str(e)}")
        log_and_flush(f"Full traceback: {traceback.format_exc()}")
        return f"FOLDER ERROR: {str(e)}"
    finally:
        # Force garbage collection after each folder
        gc.collect()

def get_all_folders():
    """Get all folders to process - exactly from MemorySafe script"""
    
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
            log_and_flush(f"SM directory not found: {base_dir}")
    
    # Add HeLa imaging folders (647nm wavelength)
    log_and_flush(f"Processing {len(hela_folders)} HeLa imaging folders...")
    for folder in hela_folders:
        if os.path.exists(folder):
            all_folders.append(('imaging', folder, 0.647))
        else:
            log_and_flush(f"HeLa folder not found: {folder}")
    
    # Add general imaging folders (550nm default)
    log_and_flush(f"Processing {len(imaging_folders)} general imaging folders...")
    for folder in imaging_folders:
        if os.path.exists(folder):
            all_folders.append(('imaging', folder, 0.55))
        else:
            log_and_flush(f"Imaging folder not found: {folder}")
    
    # Add hierarchical imaging folders (need to walk)
    log_and_flush(f"Processing {len(hierarchical_bases)} hierarchical imaging bases...")
    for base_dir in hierarchical_bases:
        if os.path.exists(base_dir):
            for root, dirs, _ in os.walk(base_dir):
                if not dirs:  # Leaf directory
                    all_folders.append(('imaging', root, 0.55))
        else:
            log_and_flush(f"Hierarchical directory not found: {base_dir}")
    
    log_and_flush(f"Total folders found: {len(all_folders)}")
    return sorted(all_folders, key=lambda x: x[1])

def main():
    # Set NumExpr threads to avoid warning spam
    os.environ['NUMEXPR_MAX_THREADS'] = '24'
    
    print_and_log("="*60)
    print_and_log("Starting Direct Analysis (No File Copying)")
    print_and_log("="*60)
    
    try:
        # Import modules
        print_and_log("Importing modules...")
        import IOFunctions
        import sCMOSFunctions
        import SpectralFunctions
        import MaskFunctions
        import SpotDetectionFunctions
        import SR_Functions
        import ImageAnalysisFunctions
        import HelperFunctions
        
        # Initialize functions
        print_and_log("Initializing functions...")
        functions = {
            'IO': IOFunctions.IO_Functions(),
            'sCMOS': sCMOSFunctions.sCMOS_Functions(),
            'S_F': SpectralFunctions.Spectral_Funcs(),
            'M_F': MaskFunctions.Mask_Functions(),
            'SD_F': SpotDetectionFunctions.SpotDetection_Functions(),
            'SupRes_F': SR_Functions.SuperRes_Functions(),
            'I_AF': ImageAnalysisFunctions.Image_Analysis_Functions(),
            'H_F': HelperFunctions.Helper_Functions()
        }
        
        # Load camera parameters
        print_and_log("Loading camera parameters...")
        data_folder = os.path.join(project_root, "Camera_Calibrations", "Ximea_Camera")
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
        
        print_and_log("Setup completed successfully")
        
        # Get folders to process
        folders = get_all_folders()
        print_and_log(f"Found {len(folders)} folders to process")
        
        # Process each folder (handle tuple format from MemorySafe)
        results = []
        for i, folder_info in enumerate(folders):
            # Handle tuple format (type, path, [wavelength])
            if isinstance(folder_info, tuple):
                if len(folder_info) == 2:
                    folder_type, folder_path = folder_info
                    peak_wavelength = 0.638 if folder_type == 'sm' else 0.55  # Default wavelengths
                else:
                    folder_type, folder_path, peak_wavelength = folder_info
            else:
                folder_path = folder_info
                folder_type, peak_wavelength = determine_folder_type_and_wavelength(folder_path)
            
            log_and_flush(f"Processing folder {i+1}/{len(folders)}: {folder_path}")
            log_and_flush(f"Type: {folder_type}, Wavelength: {peak_wavelength}")
            
            # Monitor memory before each folder
            monitor_memory()
            
            result = process_folder_direct(folder_path, functions, camera_data, smoothing_function, folder_type, peak_wavelength)
            results.append(result)
            
            log_and_flush(f"Folder {i+1} result: {result}")
            
            # Clean status update to console
            print_status(f"Progress: {i+1}/{len(folders)} - {result.split(':')[0]}")
        
        print()  # New line after progress updates
        
        # Summary
        print_and_log("="*60)
        print_and_log("DIRECT ANALYSIS COMPLETE")
        print_and_log("="*60)
        
        successes = sum(1 for r in results if r.startswith("SUCCESS"))
        errors = sum(1 for r in results if r.startswith("ERROR"))
        skipped = sum(1 for r in results if r.startswith("SKIP"))
        warnings = sum(1 for r in results if r.startswith("WARNING"))
        
        print_and_log(f"Summary:")
        print_and_log(f"  Total folders: {len(folders)}")
        print_and_log(f"  Successful: {successes}")
        print_and_log(f"  Errors: {errors}")
        print_and_log(f"  Skipped: {skipped}")
        print_and_log(f"  Warnings: {warnings}")
        print_and_log(f"Log saved to: {log_file}")
        
    except Exception as e:
        log_and_flush(f"FATAL ERROR: {str(e)}")
        import traceback
        log_and_flush(f"Full traceback: {traceback.format_exc()}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)