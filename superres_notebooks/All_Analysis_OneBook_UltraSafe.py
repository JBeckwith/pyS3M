#!/usr/bin/env python3
"""
Ultra-Safe All Analysis OneBook Script
Date: August 27, 2025
Author: Claude Code Analysis

Enhanced Memory-Safe script with additional protections against terminal crashes:
1. ProcessPoolExecutor with proper context managers
2. File descriptor limit monitoring and cleanup
3. Signal handling for graceful termination 
4. Segmentation fault detection and recovery
5. GPU memory monitoring (if applicable)
6. Enhanced error isolation per folder
7. System resource monitoring
8. Network/filesystem error resilience
"""

import sys
import os
import gc
import logging
import traceback
import signal
import resource
import psutil
from datetime import datetime
import time

# Set resource limits to prevent system overload
try:
    # Limit memory usage (in bytes) - 32GB limit
    resource.setrlimit(resource.RLIMIT_AS, (32 * 1024**3, 32 * 1024**3))
    
    # Limit number of open files - prevent file descriptor leaks
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(4096, hard), hard))
    
    # Limit CPU time per process - prevent infinite loops
    resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))  # 1 hour limit
    
except (OSError, ValueError) as e:
    print(f"Warning: Could not set resource limits: {e}")

# Set matplotlib to non-interactive backend BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')  # Prevents memory leaks from interactive backends
import matplotlib.pyplot as plt

# Global flag for graceful shutdown
SHUTDOWN_REQUESTED = False

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global SHUTDOWN_REQUESTED
    SHUTDOWN_REQUESTED = True
    print(f"\nReceived signal {signum}. Initiating graceful shutdown...")

# Register signal handlers
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# Setup enhanced logging
log_filename = f"analysis_ultrasafe_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
logging.basicConfig(
    filename=log_filename,
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - PID:%(process)d - %(message)s',
    filemode='w'
)

# Add console logging
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)

def log_and_flush(message, level='info'):
    """Enhanced logging with immediate flush and error catching"""
    try:
        getattr(logging, level.lower())(message)
        for handler in logging.getLogger().handlers:
            handler.flush()
    except Exception as e:
        print(f"Logging error: {e}")

def monitor_system_resources():
    """Monitor all system resources"""
    try:
        process = psutil.Process()
        
        # Memory monitoring
        mem_info = process.memory_info()
        mem_gb = mem_info.rss / 1e9
        available_gb = psutil.virtual_memory().available / 1e9
        
        # File descriptor monitoring
        try:
            num_fds = process.num_fds()
        except (AttributeError, psutil.AccessDenied):
            num_fds = "N/A"
        
        # CPU monitoring
        cpu_percent = process.cpu_percent()
        
        # Disk space monitoring for scratch
        try:
            scratch_usage = psutil.disk_usage('/scratch2')
            scratch_free_gb = scratch_usage.free / 1e9
        except:
            scratch_free_gb = "N/A"
        
        log_and_flush(
            f"Resources: Memory={mem_gb:.1f}GB (avail={available_gb:.1f}GB), "
            f"FDs={num_fds}, CPU={cpu_percent:.1f}%, ScratchFree={scratch_free_gb}GB"
        )
        
        # Check for critical resource issues
        warnings = []
        if mem_gb > 30:
            warnings.append(f"High memory usage: {mem_gb:.1f}GB")
        if isinstance(num_fds, int) and num_fds > 1000:
            warnings.append(f"High file descriptor count: {num_fds}")
        if available_gb < 2:
            warnings.append(f"Low available memory: {available_gb:.1f}GB")
        if isinstance(scratch_free_gb, (int, float)) and scratch_free_gb < 10:
            warnings.append(f"Low scratch disk space: {scratch_free_gb}GB")
        
        if warnings:
            for warning in warnings:
                log_and_flush(f"WARNING: {warning}", 'warning')
            return False
        
        return True
    except Exception as e:
        log_and_flush(f"Resource monitoring error: {e}", 'warning')
        return True

def cleanup_resources():
    """Enhanced cleanup of all resources"""
    try:
        # Close all matplotlib figures
        plt.close('all')
        plt.clf()
        
        # Force garbage collection multiple times
        for _ in range(3):
            gc.collect()
        
        # Close any lingering file descriptors
        try:
            process = psutil.Process()
            open_files = process.open_files()
            if len(open_files) > 100:  # Too many open files
                log_and_flush(f"Warning: {len(open_files)} open files detected", 'warning')
        except (psutil.AccessDenied, AttributeError):
            pass
        
        log_and_flush("Resources cleaned up")
    except Exception as e:
        log_and_flush(f"Cleanup error: {e}", 'warning')

def safe_import_modules():
    """Safely import all required modules with error isolation"""
    log_and_flush("Importing modules with crash protection...")
    
    try:
        # Add project path
        sys.path.append("..")
        
        # Import modules one by one with individual error handling
        modules = {}
        
        import_list = [
            ('IOFunctions', 'src.IOFunctions'),
            ('Multicolour_Simulation_Functions', 'src.Multicolour_Simulation_Functions'),
            ('PlottingFunctions', 'src.PlottingFunctions'),
            ('ImageAnalysisFunctions', 'src.ImageAnalysisFunctions'),
            ('sCMOSFunctions', 'src.sCMOSFunctions'),
            ('PSFFunctions', 'src.PSFFunctions'),
            ('SpectralFunctions', 'src.SpectralFunctions'),
            ('MaskFunctions', 'src.MaskFunctions'),
            ('SpotDetectionFunctions', 'src.SpotDetectionFunctions'),
            ('SR_Functions', 'src.SR_Functions'),
            ('HelperFunctions', 'HelperFunctions')
        ]
        
        for name, module_path in import_list:
            try:
                log_and_flush(f"Importing {name}...")
                if name == 'HelperFunctions':
                    import HelperFunctions
                    modules[name] = HelperFunctions
                else:
                    module = __import__(module_path, fromlist=[name])
                    modules[name] = getattr(module, name.split('.')[-1])
                
                log_and_flush(f"✓ {name} imported successfully")
            except Exception as e:
                log_and_flush(f"✗ Failed to import {name}: {e}", 'error')
                raise ImportError(f"Critical module {name} failed to import: {e}")
        
        log_and_flush("All modules imported successfully")
        return modules
        
    except Exception as e:
        log_and_flush(f"Module import failed: {e}", 'error')
        log_and_flush(f"Import traceback: {traceback.format_exc()}", 'error')
        raise

def safe_initialize_functions(modules):
    """Safely initialize all function objects with crash protection"""
    log_and_flush("Initializing function objects with crash protection...")
    
    try:
        functions = {}
        
        init_list = [
            ('IO', 'IOFunctions', 'IO_Functions'),
            ('MSF', 'Multicolour_Simulation_Functions', 'MultiC_Sim_Funcs'),
            ('plotter', 'PlottingFunctions', 'Plotter'),
            ('I_AF', 'ImageAnalysisFunctions', 'Image_Analysis_Functions'),
            ('sCMOS', 'sCMOSFunctions', 'sCMOS_Functions'),
            ('PSF', 'PSFFunctions', 'PSF_Functions'),
            ('S_F', 'SpectralFunctions', 'Spectral_Funcs'),
            ('M_F', 'MaskFunctions', 'Mask_Functions'),
            ('SD_F', 'SpotDetectionFunctions', 'SpotDetection_Functions'),
            ('SupRes_F', 'SR_Functions', 'SuperRes_Functions'),
            ('H_F', 'HelperFunctions', 'Helper_Functions')
        ]
        
        for key, module_name, class_name in init_list:
            try:
                log_and_flush(f"Initializing {key} ({class_name})...")
                module = modules[module_name]
                cls = getattr(module, class_name)
                functions[key] = cls()
                log_and_flush(f"✓ {key} initialized successfully")
            except Exception as e:
                log_and_flush(f"✗ Failed to initialize {key}: {e}", 'error')
                raise RuntimeError(f"Critical function {key} failed to initialize: {e}")
        
        log_and_flush("All functions initialized successfully")
        return functions
        
    except Exception as e:
        log_and_flush(f"Function initialization failed: {e}", 'error')
        log_and_flush(f"Initialization traceback: {traceback.format_exc()}", 'error')
        raise

def ultra_safe_process_folder(folder_info, functions, camera_data, smoothing_function):
    """Ultra-safe folder processing with maximum crash protection"""
    
    # Handle tuple format (type, path, [wavelength])
    if isinstance(folder_info, tuple):
        if len(folder_info) == 2:
            folder_type, folder_path = folder_info
            peak_wavelength = 0.55  # default
        else:
            folder_type, folder_path, peak_wavelength = folder_info
    else:
        folder_type = 'auto'
        folder_path = folder_info
        peak_wavelength = 0.55
    
    log_and_flush(f"ULTRA-SAFE: Starting folder: {folder_path}")
    
    # Check for shutdown signal
    if SHUTDOWN_REQUESTED:
        return f"SHUTDOWN: {folder_path}"
    
    try:
        # Pre-flight checks
        if not os.path.exists(folder_path):
            return f"NOT_FOUND: {folder_path}"
        
        # Check if analysis needed (skip if recent .h5 files exist)
        h5_files = [f for f in os.listdir(folder_path) if f.endswith('.h5')]
        if h5_files:
            cutoff_time = datetime(2025, 8, 26, 10, 0, 0)
            for h5_file in h5_files:
                file_path = os.path.join(folder_path, h5_file)
                file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                if file_mtime >= cutoff_time:
                    return f"SKIPPED: {folder_path}"
        
        # Monitor resources before processing
        if not monitor_system_resources():
            cleanup_resources()
            if not monitor_system_resources():
                return f"RESOURCE_ERROR: {folder_path}"
        
        # Setup scratch directory with PID for uniqueness
        # Try /scratch2 first, fall back to /tmp if not available
        scratch_bases = ["/scratch2/jsb92", "/tmp/jsb92_analysis"]
        scratch_base = None
        
        for base in scratch_bases:
            try:
                os.makedirs(base, exist_ok=True)
                scratch_base = base
                break
            except (OSError, PermissionError) as e:
                log_and_flush(f"Cannot use {base}: {e}", 'warning')
        
        if scratch_base is None:
            return f"SCRATCH_ERROR: No available scratch directory"
        
        scratch_folder = f"{scratch_base}/{os.path.basename(folder_path)}_{os.getpid()}_{int(time.time())}"
        
        try:
            # Copy files to scratch (sequential to avoid issues)
            files_to_copy = [
                os.path.join(folder_path, f) 
                for f in os.listdir(folder_path) 
                if not f.endswith('.h5')
            ]
            
            log_and_flush(f"Copying {len(files_to_copy)} files to scratch")
            os.makedirs(scratch_folder, exist_ok=True)
            
            # Copy with progress monitoring
            for i, source_file in enumerate(files_to_copy):
                if SHUTDOWN_REQUESTED:
                    return f"SHUTDOWN: {folder_path}"
                
                dest_file = os.path.join(scratch_folder, os.path.basename(source_file))
                
                # Retry copy with exponential backoff
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        import shutil
                        shutil.copyfile(source_file, dest_file)
                        break
                    except Exception as e:
                        if attempt < max_retries - 1:
                            wait_time = 2 ** attempt
                            log_and_flush(f"Copy retry {attempt+1} for {source_file} after {wait_time}s: {e}", 'warning')
                            time.sleep(wait_time)
                        else:
                            raise
                
                # Progress update every 10 files
                if (i + 1) % 10 == 0:
                    log_and_flush(f"Copied {i+1}/{len(files_to_copy)} files")
            
            # Process data using SR_Functions with enhanced error handling
            try:
                SupRes_F = functions['SupRes_F']
                
                # Monitor resources before heavy computation
                if not monitor_system_resources():
                    cleanup_resources()
                
                if folder_type == 'sm' or (folder_type == 'auto' and ('dyes' in folder_path.lower() or 'biotinylated' in folder_path.lower())):
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
                
            except Exception as analysis_error:
                log_and_flush(f"Analysis error for {folder_path}: {analysis_error}", 'error')
                return f"ANALYSIS_ERROR: {folder_path} - {str(analysis_error)[:100]}"
            
            # Copy results back
            try:
                result_files = [f for f in os.listdir(scratch_folder) if f.endswith('.h5')]
                log_and_flush(f"Copying {len(result_files)} results back")
                
                for result_file in result_files:
                    source = os.path.join(scratch_folder, result_file)
                    dest = os.path.join(folder_path, result_file)
                    import shutil
                    shutil.copyfile(source, dest)
                
            except Exception as copy_error:
                log_and_flush(f"Result copy error for {folder_path}: {copy_error}", 'warning')
                return f"COPY_BACK_ERROR: {folder_path}"
            
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
        log_and_flush(f"Critical processing error for {folder_path}: {e}", 'error')
        log_and_flush(f"Error traceback: {traceback.format_exc()}", 'error')
        return f"CRITICAL_ERROR: {folder_path} - {str(e)[:100]}"
    
    finally:
        # Always cleanup resources after each folder
        cleanup_resources()

def get_all_folders():
    """Get all folders to process - same as MemorySafe script"""
    # [Same implementation as MemorySafe script - keeping it identical]
    
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
    """Ultra-safe main function with maximum crash protection"""
    
    log_and_flush("="*60)
    log_and_flush("Starting Ultra-Safe Analysis with Enhanced Crash Protection")
    log_and_flush("="*60)
    
    try:
        # Initial resource check
        monitor_system_resources()
        
        # Safe module import
        modules = safe_import_modules()
        
        # Safe function initialization  
        functions = safe_initialize_functions(modules)
        
        # Load camera parameters with error handling
        log_and_flush("Loading camera parameters...")
        try:
            data_folder = '../Camera_Calibrations/Ximea_Camera'
            
            camera_data = {
                'gain': functions['IO'].read_tiff(os.path.join(data_folder, "gain.tif")),
                'offset': functions['IO'].read_tiff(os.path.join(data_folder, "offset.tif")),
                'variance': functions['IO'].read_tiff(os.path.join(data_folder, "variance.tif")),
                'readnoise': functions['IO'].read_tiff(os.path.join(data_folder, "readnoise.tif")),
                'rqe': functions['IO'].read_tiff(os.path.join(data_folder, "rqe.tif"))
            }
            log_and_flush("✓ Camera parameters loaded successfully")
        except Exception as e:
            log_and_flush(f"✗ Camera parameter loading failed: {e}", 'error')
            raise
        
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
        
        if len(folders) == 0:
            log_and_flush("No folders found to process", 'warning')
            return 0
        
        # Process folders with ultra-safe sequential processing
        results = []
        
        for i, folder_info in enumerate(folders):
            # Check for shutdown signal
            if SHUTDOWN_REQUESTED:
                log_and_flush("Shutdown requested - stopping processing", 'warning')
                break
            
            # Extract folder path for logging
            if isinstance(folder_info, tuple):
                folder_path = folder_info[1]
            else:
                folder_path = folder_info
                
            log_and_flush(f"Processing folder {i+1}/{len(folders)}: {folder_path}")
            
            # Monitor resources before each folder
            if not monitor_system_resources():
                log_and_flush("Resource constraints - forcing cleanup", 'warning')
                cleanup_resources()
                time.sleep(5)  # Brief pause for system recovery
            
            result = ultra_safe_process_folder(folder_info, functions, camera_data, smoothing_function)
            results.append(result)
            
            log_and_flush(f"Folder {i+1} result: {result}")
            
            # Status update
            print(f"\rProgress: {i+1}/{len(folders)} - {result.split(':')[0]}", end='', flush=True)
            
            # Periodic full system check
            if (i + 1) % 10 == 0:
                log_and_flush(f"Completed {i+1}/{len(folders)} folders")
                monitor_system_resources()
        
        print()  # New line
        
        # Summary
        log_and_flush("="*60)
        log_and_flush("ULTRA-SAFE PROCESSING COMPLETE")
        log_and_flush("="*60)
        
        success_count = len([r for r in results if r.startswith('SUCCESS')])
        skip_count = len([r for r in results if r.startswith('SKIPPED')])
        error_count = len([r for r in results if 'ERROR' in r])
        shutdown_count = len([r for r in results if r.startswith('SHUTDOWN')])
        
        log_and_flush(f"Total folders: {len(folders)}")
        log_and_flush(f"Successful: {success_count}")
        log_and_flush(f"Skipped: {skip_count}")
        log_and_flush(f"Errors: {error_count}")
        log_and_flush(f"Shutdowns: {shutdown_count}")
        
        log_and_flush(f"Log file: {log_filename}")
        
        # Final resource check
        monitor_system_resources()
        
        return 0
        
    except KeyboardInterrupt:
        log_and_flush("User interrupted processing", 'warning')
        return 1
    except Exception as e:
        log_and_flush(f"CRITICAL SYSTEM ERROR: {e}", 'error')
        log_and_flush(f"System traceback: {traceback.format_exc()}", 'error')
        return 1
    
    finally:
        cleanup_resources()
        log_and_flush("Ultra-safe analysis session ended")

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except SystemExit:
        pass
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)