#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-hoc normalisation of error terms in all .h5 files from batch analysis.

This script goes through all the folders analysed in batch_analysis.sh,
finds all .h5 files, and normalises the error terms following the same
logic as IOFunctions.py lines 72-80 and 98-106:
- A_R_err, A_G_err, A_B_err terms normalised by photons
- bg_R_err, bg_G_err, bg_B_err terms normalised by background_photons

Created for pyBayerSMLM post-hoc error term normalisation
Author: Claude Code Assistant
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from typing import List, Tuple
import logging

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('post-hoc_normalisation.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def get_batch_analysis_folders() -> List[str]:
    """
    Get all folder paths that are processed by batch_analysis.sh.
    
    Returns:
        List of folder paths from all categories in batch_analysis.sh
    """
    # SM Data directories (base directories that get processed hierarchically)
    sm_data_dirs = [
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
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250715_HollidayJunctions/60pM_HollidayJunction_50mMMgCl2/100perc561_NF_SP785_50ms_1',
    ]
    
    # HeLa folders (direct folders)
    hela_folders = [
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250523_HeLa_STORM/Cell3_HILO_190mW_638_ximea638_setting/Lp638_190_mw_40ms_exosure_HILO_1',
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250523_HeLa_STORM/Cell4_HILO_190mW_638_ximea638_setting/Lp638_190_mw_40ms_exosure_HILO_1',
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250523_HeLa_STORM/Cell2_HILO_190mW_638_ximea638_setting/Lp638_190_mw_40ms_exosure_HILO_1',
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250523_HeLa_STORM/Cell1_HILO_190mW_638_ximea638_setting/Lp638_190_mw_40ms_exosure_HILO_2',
    ]
    
    # Imaging folders (direct folders)
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
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/JSB/20250716_iPSCJamesEvans/250pMCy3B_250pM565_250pMCF550_250pM647/20perc561_40mW638_NF_488LP_785SP_1',
    ]
    
    # Hierarchical directories (base directories that get processed hierarchically)
    hierarchical_dirs = [
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/JSB/20250414_CellPAINT/data',
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250404_Ximea_AsynNRThX/data',
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250818_DNAOrigami',
    ]
    
    # Combine all folders
    all_folders = sm_data_dirs + hela_folders + imaging_folders + hierarchical_dirs
    
    return all_folders

def find_leaf_directories(base_dir: str) -> List[str]:
    """
    Find all leaf directories (directories with no subdirectories) in a hierarchical structure.
    This replicates the hierarchical processing logic from batch_analysis.sh.
    
    Args:
        base_dir: Base directory to search
        
    Returns:
        List of leaf directory paths
    """
    leaf_dirs = []
    
    if not os.path.exists(base_dir):
        return leaf_dirs
    
    for root, dirs, files in os.walk(base_dir):
        # Skip the base directory itself
        if root == base_dir:
            continue
            
        # If this directory has no subdirectories, it's a leaf
        if not dirs:
            leaf_dirs.append(root)
    
    return leaf_dirs

def find_all_h5_files(folder_paths: List[str]) -> List[Tuple[str, str]]:
    """
    Find all .h5 files in the given folder paths.
    
    Args:
        folder_paths: List of directory paths to search
        
    Returns:
        List of tuples (folder_path, h5_file_path)
    """
    h5_files = []
    logger = logging.getLogger(__name__)
    
    for folder_path in folder_paths:
        if not os.path.exists(folder_path):
            logger.warning(f"Folder not found: {folder_path}")
            continue
            
        # Find all .h5 files in this folder
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.h5'):
                    h5_file_path = os.path.join(root, file)
                    h5_files.append((folder_path, h5_file_path))
                    
    logger.info(f"Found {len(h5_files)} .h5 files across all folders")
    return h5_files

def normalise_error_terms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise error terms in the dataframe following IOFunctions.py logic.
    
    This replicates the normalisation from IOFunctions.py lines 72-80 and 98-106:
    - A_R_err, A_G_err, A_B_err normalised by photons
    - bg_R_err, bg_G_err, bg_B_err normalised by background_photons
    
    Args:
        df: DataFrame to normalise
        
    Returns:
        Normalised DataFrame
    """
    logger = logging.getLogger(__name__)
    original_shape = df.shape
    
    # Check if we have the required columns for amplitude error normalisation
    amplitude_error_cols = ['A_B_err', 'A_G_err', 'A_R_err']
    has_amplitude_errors = all(col in df.columns for col in amplitude_error_cols)
    
    # Check if we have photons column
    has_photons = 'photons' in df.columns
    
    if has_amplitude_errors and has_photons:
        # Normalise amplitude error terms by photons (avoid division by zero)
        mask = df['photons'] > 0
        n_to_normalise = mask.sum()
        
        if n_to_normalise > 0:
            df.loc[mask, 'A_B_err'] = df.loc[mask, 'A_B_err'] / df.loc[mask, 'photons']
            df.loc[mask, 'A_G_err'] = df.loc[mask, 'A_G_err'] / df.loc[mask, 'photons']
            df.loc[mask, 'A_R_err'] = df.loc[mask, 'A_R_err'] / df.loc[mask, 'photons']
            logger.info(f"Normalised amplitude errors for {n_to_normalise} rows by photons")
        else:
            logger.warning("No rows with photons > 0 found for amplitude error normalisation")
    else:
        missing_cols = []
        if not has_amplitude_errors:
            missing_cols.extend([col for col in amplitude_error_cols if col not in df.columns])
        if not has_photons:
            missing_cols.append('photons')
        logger.info(f"Skipping amplitude error normalisation - missing columns: {missing_cols}")
    
    # Check if we have background error columns
    background_error_cols = ['bg_B_err', 'bg_G_err', 'bg_R_err']
    has_background_errors = all(col in df.columns for col in background_error_cols)
    
    # Check if we have background columns to calculate background_photons
    background_cols = ['bg_B', 'bg_G', 'bg_R']
    has_background_cols = all(col in df.columns for col in background_cols)
    
    if has_background_errors and has_background_cols:
        # Calculate background_photons (same logic as IOFunctions.py line 84)
        df['background_photons'] = df['bg_B'] + df['bg_G'] + df['bg_R']
        
        # Normalise background error terms by background_photons (avoid division by zero)
        mask = df['background_photons'] > 0
        n_to_normalise = mask.sum()
        
        if n_to_normalise > 0:
            df.loc[mask, 'bg_B_err'] = df.loc[mask, 'bg_B_err'] / df.loc[mask, 'background_photons']
            df.loc[mask, 'bg_G_err'] = df.loc[mask, 'bg_G_err'] / df.loc[mask, 'background_photons']
            df.loc[mask, 'bg_R_err'] = df.loc[mask, 'bg_R_err'] / df.loc[mask, 'background_photons']
            logger.info(f"Normalised background errors for {n_to_normalise} rows by background_photons")
        else:
            logger.warning("No rows with background_photons > 0 found for background error normalisation")
    else:
        missing_cols = []
        if not has_background_errors:
            missing_cols.extend([col for col in background_error_cols if col not in df.columns])
        if not has_background_cols:
            missing_cols.extend([col for col in background_cols if col not in df.columns])
        logger.info(f"Skipping background error normalisation - missing columns: {missing_cols}")
    
    logger.info(f"Processed DataFrame: {original_shape} -> {df.shape}")
    return df

def process_h5_file(folder_path: str, h5_file_path: str, dry_run: bool = False) -> bool:
    """
    Process a single .h5 file to normalise error terms.
    
    Args:
        folder_path: Original folder path (for logging)
        h5_file_path: Path to the .h5 file
        dry_run: If True, don't actually modify files
        
    Returns:
        True if successful, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Read the HDF5 file
        logger.info(f"Processing: {h5_file_path}")
        
        # Try to read with pandas
        df = pd.read_hdf(h5_file_path, key='df')
        original_shape = df.shape
        
        # Check if error terms are already normalised by looking for typical pre-normalisation values
        # Original error terms are typically much larger than normalised ones
        amplitude_cols = ['A_B_err', 'A_G_err', 'A_R_err']
        background_cols = ['bg_B_err', 'bg_G_err', 'bg_R_err']
        
        # Heuristic check: if error terms are very large (>100), they're likely not normalised
        needs_normalisation = False
        
        if all(col in df.columns for col in amplitude_cols):
            max_amp_err = df[amplitude_cols].max().max()
            if not pd.isna(max_amp_err) and max_amp_err > 100:
                needs_normalisation = True
                logger.info(f"Max amplitude error: {max_amp_err:.2f} - appears to need normalisation")
        
        if all(col in df.columns for col in background_cols):
            max_bg_err = df[background_cols].max().max()
            if not pd.isna(max_bg_err) and max_bg_err > 100:
                needs_normalisation = True
                logger.info(f"Max background error: {max_bg_err:.2f} - appears to need normalisation")
        
        if not needs_normalisation:
            logger.info("Error terms appear already normalised - skipping")
            return True
        
        # Normalise error terms
        df_normalised = normalise_error_terms(df.copy())
        
        if not dry_run:
            # Create backup
            backup_path = h5_file_path + '.backup'
            if not os.path.exists(backup_path):
                logger.info(f"Creating backup: {backup_path}")
                import shutil
                shutil.copy2(h5_file_path, backup_path)
            
            # Save normalised data back to the file
            df_normalised.to_hdf(h5_file_path, key='df', mode='w', format='table')
            logger.info(f"Saved normalised data to: {h5_file_path}")
        else:
            logger.info(f"DRY RUN: Would normalise and save {original_shape} -> {df_normalised.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing {h5_file_path}: {str(e)}")
        return False

def main():
    """Main function to process all .h5 files."""
    logger = setup_logging()
    
    # Parse command line arguments
    dry_run = '--dry-run' in sys.argv
    if dry_run:
        logger.info("=== DRY RUN MODE - No files will be modified ===")
    
    logger.info("Starting post-hoc error term normalisation for batch analysis .h5 files")
    
    # Get all folder paths from batch_analysis.sh
    logger.info("Getting folder paths from batch analysis configuration...")
    base_folders = get_batch_analysis_folders()
    logger.info(f"Found {len(base_folders)} base folders from batch analysis")
    
    # For hierarchical directories, we need to find leaf directories
    # (matching the logic in batch_analysis.sh process_hierarchical function)
    hierarchical_base_dirs = [
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/JSB/20250414_CellPAINT/data',
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250404_Ximea_AsynNRThX/data',
        '/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250818_DNAOrigami',
    ]
    
    all_folders_to_process = []
    
    # Add direct folders (non-hierarchical)
    for folder in base_folders:
        if folder not in hierarchical_base_dirs:
            all_folders_to_process.append(folder)
    
    # Add leaf directories from hierarchical bases
    for base_dir in hierarchical_base_dirs:
        if base_dir in base_folders:  # Only process if it's in our batch analysis list
            leaf_dirs = find_leaf_directories(base_dir)
            all_folders_to_process.extend(leaf_dirs)
            logger.info(f"Found {len(leaf_dirs)} leaf directories in {base_dir}")
    
    logger.info(f"Total folders to process: {len(all_folders_to_process)}")
    
    # Find all .h5 files
    logger.info("Scanning for .h5 files...")
    h5_files = find_all_h5_files(all_folders_to_process)
    
    if not h5_files:
        logger.warning("No .h5 files found!")
        return
    
    # Process each .h5 file
    logger.info(f"Processing {len(h5_files)} .h5 files...")
    
    success_count = 0
    error_count = 0
    
    for i, (folder_path, h5_file_path) in enumerate(h5_files, 1):
        logger.info(f"[{i}/{len(h5_files)}] Processing: {os.path.basename(h5_file_path)}")
        
        if process_h5_file(folder_path, h5_file_path, dry_run=dry_run):
            success_count += 1
        else:
            error_count += 1
    
    # Final summary
    logger.info("=" * 60)
    logger.info("POST-HOC NORMALISATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total .h5 files processed: {len(h5_files)}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Errors: {error_count}")
    
    if dry_run:
        logger.info("DRY RUN - No files were actually modified")
        logger.info("Remove --dry-run flag to perform actual normalisation")
    else:
        logger.info("Backup files created with .backup extension")
    
    logger.info("Check post-hoc_normalisation.log for detailed processing information")

if __name__ == "__main__":
    main()