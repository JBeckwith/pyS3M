#!/usr/bin/env python
# coding: utf-8
"""
Optimized All Analysis Script - Memory Efficient Version
Fixes memory issues, dangerous exception handling, and implements robust processing
"""

import numpy as np
import os
import gc
import time
import logging
import shutil
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import types
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analysis_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import pyBayerSMLM modules
import sys
sys.path.append("..")

try:
    from src import (IOFunctions, Multicolour_Simulation_Functions, PlottingFunctions,
                     ImageAnalysisFunctions, sCMOSFunctions, PSFFunctions, 
                     SpectralFunctions, MaskFunctions, SpotDetectionFunctions, SR_Functions)
    import HelperFunctions
    logger.info("All modules imported successfully")
except ImportError as e:
    logger.error(f"Import error: {e}")
    raise


@dataclass 
class AnalysisConfig:
    """Configuration for analysis parameters"""
    pfa: float = 1e-4
    roi_size: int = 12
    na: float = 1.49
    pixel_size: float = 0.069
    image_type: str = ".tif"
    scratch_dir: str = '/scratch2/jsb92'
    cutoff_time: Optional[datetime] = None
    max_retries: int = 3
    retry_delay: int = 2
    
    def __post_init__(self):
        if self.cutoff_time is None:
            self.cutoff_time = datetime(2025, 8, 26, 10, 0, 0)


@dataclass
class DatasetConfig:
    """Configuration for individual datasets"""
    folders: List[str]
    peak_wavelength: float
    analysis_type: str = "imaging"  # "imaging" or "sm"
    name: str = "unnamed"


class AnalysisProcessor:
    """Memory-efficient analysis processor with proper resource management"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.setup_modules()
        self.setup_camera_parameters()
        
    def setup_modules(self):
        """Initialize all required modules"""
        logger.info("Initializing modules...")
        
        self.IO = IOFunctions.IO_Functions()
        self.MSF = Multicolour_Simulation_Functions.MultiC_Sim_Funcs()
        self.plotter = PlottingFunctions.Plotter()
        self.I_AF = ImageAnalysisFunctions.Image_Analysis_Functions()
        self.sCMOS = sCMOSFunctions.sCMOS_Functions()
        self.PSF = PSFFunctions.PSF_Functions()
        self.S_F = SpectralFunctions.Spectral_Funcs()
        self.M_F = MaskFunctions.Mask_Functions()
        self.SD_F = SpotDetectionFunctions.SpotDetection_Functions()
        self.SupRes_F = SR_Functions.SuperRes_Functions()
        self.H_F = HelperFunctions.Helper_Functions()
        
        # Setup smoothing function
        self.smoothing_function = types.SimpleNamespace()
        self.smoothing_function.args = {"sigma": 1.5}
        self.smoothing_function.extent = 1.5
        self.smoothing_function.smoothing_function = self.sCMOS.gaussian_filter_stack
        self.smoothing_function.data_arg = "image"
        
        logger.info("Modules initialized successfully")
    
    def setup_camera_parameters(self):
        """Load camera calibration parameters"""
        logger.info("Loading camera parameters...")
        
        data_folder = Path('../Camera_Calibrations/Ximea_Camera')
        
        try:
            self.gain_map = self.IO.read_tiff(data_folder / "gain.tif")
            self.offset_map = self.IO.read_tiff(data_folder / "offset.tif") 
            self.variance = self.IO.read_tiff(data_folder / "variance.tif")
            self.read_noise = self.IO.read_tiff(data_folder / "readnoise.tif")
            self.rqe = self.IO.read_tiff(data_folder / "rqe.tif")
            
            R, G, B, wavelength = self.S_F.getpixelefficiency()
            self.pixel_QYs = np.vstack([B, G, R])
            
            self.camera_parameters = {
                "pixel_QYs": self.pixel_QYs,
                "pixel_order": ['B', 'G', 'R'],
                "pixel_order_indices": [0, 1, 2]
            }
            
            logger.info("Camera parameters loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load camera parameters: {e}")
            raise
    
    def check_disk_space(self, required_gb: float = 10.0) -> bool:
        """Check available disk space in scratch directory"""
        try:
            statvfs = os.statvfs(self.config.scratch_dir)
            free_gb = (statvfs.f_frsize * statvfs.f_bavail) / (1024**3)
            
            if free_gb < required_gb:
                logger.warning(f"Low disk space: {free_gb:.1f} GB available, {required_gb:.1f} GB required")
                return False
            
            logger.info(f"Disk space OK: {free_gb:.1f} GB available")
            return True
            
        except Exception as e:
            logger.error(f"Could not check disk space: {e}")
            return False
    
    def should_analyse_folder(self, folder_path: str) -> bool:
        """Check if folder should be analysed based on .h5 file timestamps"""
        try:
            folder_path_obj = Path(folder_path)
            h5_files = list(folder_path_obj.glob('*.h5'))
            
            if not h5_files:
                logger.info(f"No .h5 files found in {folder_path} - proceeding")
                return True
            
            for h5_file in h5_files:
                file_mtime = datetime.fromtimestamp(h5_file.stat().st_mtime)
                if file_mtime >= self.config.cutoff_time:
                    logger.info(f"Found recent .h5 file {h5_file.name} from {file_mtime} - skipping")
                    return False
            
            logger.info(f"All .h5 files in {folder_path} are old - proceeding")
            return True
            
        except Exception as e:
            logger.error(f"Error checking folder {folder_path}: {e}")
            return False
    
    def safe_copy_file(self, src: str, dst_dir: str, max_retries: Optional[int] = None) -> bool:
        """Safely copy file with retry logic and proper error handling"""
        max_retries = max_retries or self.config.max_retries
        src_path = Path(src)
        dst_dir_path = Path(dst_dir)
        dst_file = dst_dir_path / src_path.name
        
        # Create destination directory
        try:
            dst_dir_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Could not create directory {dst_dir_path}: {e}")
            return False
        
        # Copy file with retries
        for attempt in range(max_retries):
            try:
                shutil.copy2(src, dst_file)
                logger.debug(f"Copied {src_path.name} successfully")
                return True
                
            except (IOError, OSError, shutil.Error) as e:
                logger.warning(f"Copy attempt {attempt + 1}/{max_retries} failed for {src_path.name}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(self.config.retry_delay)
                else:
                    logger.error(f"Failed to copy {src} after {max_retries} attempts")
                    
        return False
    
    def safe_remove_directory(self, directory: str) -> bool:
        """Safely remove directory with proper error handling"""
        try:
            if os.path.exists(directory):
                shutil.rmtree(directory)
                logger.debug(f"Removed directory {directory}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove directory {directory}: {e}")
            return False
    
    def copy_files_to_scratch(self, files: List[str], scratch_dir: str) -> bool:
        """Copy files to scratch directory efficiently"""
        logger.info(f"Copying {len(files)} files to {scratch_dir}")
        
        success_count = 0
        for file_path in files:
            if self.safe_copy_file(file_path, scratch_dir):
                success_count += 1
            else:
                logger.warning(f"Failed to copy {file_path}")
        
        logger.info(f"Successfully copied {success_count}/{len(files)} files")
        return success_count == len(files)
    
    def copy_results_back(self, scratch_dir: str, target_dir: str, file_pattern: str = "*.h5") -> bool:
        """Copy analysis results back to original location"""
        try:
            scratch_path = Path(scratch_dir)
            target_path = Path(target_dir)
            
            result_files = list(scratch_path.glob(file_pattern))
            if not result_files:
                logger.warning(f"No result files found in {scratch_dir}")
                return True  # Not an error if no results
            
            target_path.mkdir(parents=True, exist_ok=True)
            
            for result_file in result_files:
                target_file = target_path / result_file.name
                shutil.copy2(result_file, target_file)
                logger.debug(f"Copied result {result_file.name}")
            
            logger.info(f"Copied {len(result_files)} result files back")
            return True
            
        except Exception as e:
            logger.error(f"Failed to copy results back: {e}")
            return False
    
    def process_single_folder(self, folder_path: str, peak_wavelength: float, analysis_type: str = "imaging") -> bool:
        """Process a single folder with proper memory management"""
        folder_path_obj = Path(folder_path)
        
        if not self.should_analyse_folder(folder_path):
            return True  # Skip but not an error
        
        if not self.check_disk_space():
            logger.error("Insufficient disk space")
            return False
        
        scratch_dir = Path(self.config.scratch_dir) / folder_path_obj.name
        
        try:
            logger.info(f"Processing {folder_path} -> {scratch_dir}")
            
            # Get files to copy (exclude .h5 files)
            files_to_copy = [
                str(f) for f in folder_path_obj.iterdir() 
                if f.is_file() and not f.suffix == '.h5'
            ]
            
            if not files_to_copy:
                logger.warning(f"No files to process in {folder_path}")
                return True
            
            # Copy files to scratch
            if not self.copy_files_to_scratch(files_to_copy, str(scratch_dir)):
                logger.error("Failed to copy files to scratch")
                return False
            
            # Run analysis
            try:
                if analysis_type == "sm":
                    self.SupRes_F.fit_SM_data(
                        str(scratch_dir),
                        self.smoothing_function,
                        self.gain_map,
                        self.offset_map,
                        self.rqe,
                        self.read_noise,
                        variance=self.variance,
                        pfa=self.config.pfa,
                        ROI_size=self.config.roi_size,
                        peak_wavelength=peak_wavelength,
                        NA=self.config.na,
                        pixel_size=self.config.pixel_size,
                        image_type=self.config.image_type,
                    )
                else:  # imaging
                    self.SupRes_F.fit_imaging_data(
                        str(scratch_dir),
                        self.smoothing_function,
                        self.gain_map,
                        self.offset_map,
                        self.rqe,
                        self.read_noise,
                        variance=self.variance,
                        pfa=self.config.pfa,
                        ROI_size=self.config.roi_size,
                        peak_wavelength=peak_wavelength,
                        NA=self.config.na,
                        pixel_size=self.config.pixel_size,
                        image_type=self.config.image_type,
                    )
                
                logger.info(f"Analysis completed for {folder_path.name}")
                
            except Exception as e:
                logger.error(f"Analysis failed for {folder_path}: {e}")
                return False
            
            # Copy results back
            if not self.copy_results_back(str(scratch_dir), folder_path):
                logger.error("Failed to copy results back")
                return False
            
            return True
            
        finally:
            # Always clean up scratch directory and force garbage collection
            self.safe_remove_directory(str(scratch_dir))
            gc.collect()  # Force memory cleanup
    
    def get_all_leaf_directories(self, root_directory: str) -> List[str]:
        """Get all leaf directories (directories with no subdirectories)"""
        leaf_dirs = []
        root_path = Path(root_directory)
        
        if not root_path.exists():
            logger.warning(f"Directory does not exist: {root_directory}")
            return leaf_dirs
        
        for root, dirs, files in os.walk(root_path):
            if not dirs:  # No subdirectories = leaf directory
                leaf_dirs.append(root)
        
        return sorted(leaf_dirs)
    
    def process_dataset(self, dataset: DatasetConfig) -> Dict[str, Any]:
        """Process an entire dataset with progress tracking"""
        logger.info(f"Starting dataset: {dataset.name}")
        
        results = {
            'name': dataset.name,
            'total_folders': 0,
            'processed': 0,
            'skipped': 0,
            'failed': 0,
            'start_time': datetime.now()
        }
        
        # Expand directories to leaf directories if needed
        all_folders = []
        for folder in dataset.folders:
            if '*' in folder or folder.endswith('/'):
                # Get leaf directories 
                all_folders.extend(self.get_all_leaf_directories(folder))
            else:
                all_folders.append(folder)
        
        results['total_folders'] = len(all_folders)
        logger.info(f"Found {len(all_folders)} folders to process")
        
        for i, folder in enumerate(all_folders, 1):
            folder_name = Path(folder).name
            logger.info(f"Processing folder {i}/{len(all_folders)}: {folder_name}")
            
            try:
                if self.should_analyse_folder(folder):
                    if self.process_single_folder(folder, dataset.peak_wavelength, dataset.analysis_type):
                        results['processed'] += 1
                        logger.info(f"✅ Completed: {folder_name}")
                    else:
                        results['failed'] += 1
                        logger.error(f"❌ Failed: {folder_name}")
                else:
                    results['skipped'] += 1
                    logger.info(f"⏭️  Skipped: {folder_name}")
                    
            except Exception as e:
                logger.error(f"Unexpected error processing {folder}: {e}")
                results['failed'] += 1
            
            # Progress update
            if i % 5 == 0 or i == len(all_folders):
                elapsed = datetime.now() - results['start_time']
                logger.info(f"Progress: {i}/{len(all_folders)} folders, "
                           f"P:{results['processed']}, S:{results['skipped']}, F:{results['failed']}, "
                           f"Time: {elapsed}")
        
        results['end_time'] = datetime.now()
        results['duration'] = results['end_time'] - results['start_time']
        
        logger.info(f"Dataset {dataset.name} completed: "
                   f"Processed: {results['processed']}, "
                   f"Skipped: {results['skipped']}, "
                   f"Failed: {results['failed']}, "
                   f"Duration: {results['duration']}")
        
        return results


def create_datasets() -> List[DatasetConfig]:
    """Create dataset configurations"""
    
    datasets = []
    
    # Dye characterization datasets
    dye_folders = [
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
    ]
    
    datasets.append(DatasetConfig(
        folders=dye_folders,
        peak_wavelength=0.6,
        analysis_type="sm",
        name="Dye_Characterization"
    ))
    
    # HeLa STORM datasets  
    hela_folders = [
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250523_HeLa_STORM/Cell3_HILO_190mW_638_ximea638_setting/Lp638_190_mw_40ms_exosure_HILO_1',
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250523_HeLa_STORM/Cell4_HILO_190mW_638_ximea638_setting/Lp638_190_mw_40ms_exosure_HILO_1',
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250523_HeLa_STORM/Cell2_HILO_190mW_638_ximea638_setting/Lp638_190_mw_40ms_exosure_HILO_1',
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250523_HeLa_STORM/Cell1_HILO_190mW_638_ximea638_setting/Lp638_190_mw_40ms_exosure_HILO_2'
    ]
    
    datasets.append(DatasetConfig(
        folders=hela_folders,
        peak_wavelength=0.647,
        analysis_type="imaging",
        name="HeLa_STORM"
    ))
    
    # DNA Origami and multi-color datasets
    origami_folders = [
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
    
    datasets.append(DatasetConfig(
        folders=origami_folders,
        peak_wavelength=0.55,
        analysis_type="imaging", 
        name="DNA_Origami_MultiColor"
    ))
    
    # Cell PAINT datasets (use directory expansion)
    datasets.append(DatasetConfig(
        folders=['/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/JSB/20250414_CellPAINT/data/'],
        peak_wavelength=0.6,
        analysis_type="imaging",
        name="CellPAINT"
    ))
    
    # Asyn datasets
    datasets.append(DatasetConfig(
        folders=['/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250404_Ximea_AsynNRThX/data/'],
        peak_wavelength=0.6,
        analysis_type="imaging",
        name="AsynNRThX"
    ))
    
    # DNA Origami 2025 (note: original script has bug - uses wrong starting_directory variable)
    datasets.append(DatasetConfig(
        folders=['/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250818_DNAOrigami/'],
        peak_wavelength=0.6,
        analysis_type="imaging",
        name="DNAOrigami_2025"
    ))
    
    return datasets


def main():
    """Main execution function"""
    logger.info("=== Starting Optimized Analysis Script ===")
    
    # Create configuration
    config = AnalysisConfig()
    logger.info(f"Configuration: {config}")
    
    # Initialize processor
    try:
        processor = AnalysisProcessor(config)
    except Exception as e:
        logger.error(f"Failed to initialize processor: {e}")
        return 1
    
    # Create datasets
    datasets = create_datasets()
    logger.info(f"Created {len(datasets)} datasets")
    
    # Process all datasets
    overall_results = {
        'start_time': datetime.now(),
        'datasets': [],
        'total_processed': 0,
        'total_failed': 0,
        'total_skipped': 0
    }
    
    for dataset in datasets:
        try:
            result = processor.process_dataset(dataset)
            overall_results['datasets'].append(result)
            overall_results['total_processed'] += result['processed']
            overall_results['total_failed'] += result['failed']
            overall_results['total_skipped'] += result['skipped']
            
        except Exception as e:
            logger.error(f"Dataset {dataset.name} failed completely: {e}")
            overall_results['total_failed'] += len(dataset.folders)
    
    overall_results['end_time'] = datetime.now()
    overall_results['total_duration'] = overall_results['end_time'] - overall_results['start_time']
    
    # Final summary
    logger.info("=== FINAL SUMMARY ===")
    logger.info(f"Total Duration: {overall_results['total_duration']}")
    logger.info(f"Total Processed: {overall_results['total_processed']}")
    logger.info(f"Total Skipped: {overall_results['total_skipped']}")
    logger.info(f"Total Failed: {overall_results['total_failed']}")
    
    for result in overall_results['datasets']:
        logger.info(f"Dataset {result['name']}: "
                   f"P:{result['processed']}, S:{result['skipped']}, F:{result['failed']}")
    
    success_rate = (overall_results['total_processed'] / 
                   (overall_results['total_processed'] + overall_results['total_failed'])) * 100 if overall_results['total_processed'] + overall_results['total_failed'] > 0 else 0
    
    logger.info(f"Success Rate: {success_rate:.1f}%")
    
    return 0 if overall_results['total_failed'] == 0 else 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)