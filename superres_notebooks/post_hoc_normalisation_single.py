#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single folder post-hoc normalisation function for bash script integration.

This module provides a simple function to normalise error terms in all .h5 files
within a single folder. Designed to be called from post-hoc_normalisation.sh.

Created for pyBayerSMLM post-hoc error term normalisation
Author: Claude Code Assistant
"""

import pandas as pd
import os
import sys
from typing import List

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

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
    df = df.copy()  # Avoid modifying original dataframe
    
    # Check if we have the required columns for amplitude error normalisation
    amplitude_error_cols = ['A_B_err', 'A_G_err', 'A_R_err']
    has_amplitude_errors = all(col in df.columns for col in amplitude_error_cols)
    
    # Check if we have photons column
    has_photons = 'photons' in df.columns
    
    if has_amplitude_errors and has_photons:
        # Normalise amplitude error terms by photons (avoid division by zero)
        mask = df['photons'] > 0
        if mask.sum() > 0:
            df.loc[mask, 'A_B_err'] = df.loc[mask, 'A_B_err'] / df.loc[mask, 'photons']
            df.loc[mask, 'A_G_err'] = df.loc[mask, 'A_G_err'] / df.loc[mask, 'photons']
            df.loc[mask, 'A_R_err'] = df.loc[mask, 'A_R_err'] / df.loc[mask, 'photons']
    
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
        if mask.sum() > 0:
            df.loc[mask, 'bg_B_err'] = df.loc[mask, 'bg_B_err'] / df.loc[mask, 'background_photons']
            df.loc[mask, 'bg_G_err'] = df.loc[mask, 'bg_G_err'] / df.loc[mask, 'background_photons']
            df.loc[mask, 'bg_R_err'] = df.loc[mask, 'bg_R_err'] / df.loc[mask, 'background_photons']
    
    return df

def find_h5_files(folder_path: str) -> List[str]:
    """Find all .h5 files in the given folder."""
    h5_files = []
    
    if not os.path.exists(folder_path):
        return h5_files
        
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.h5'):
                h5_files.append(os.path.join(root, file))
                
    return h5_files

def process_h5_file(h5_file_path: str) -> bool:
    """
    Process a single .h5 file to normalise error terms.
    
    Args:
        h5_file_path: Path to the .h5 file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Read the HDF5 file
        df = pd.read_hdf(h5_file_path)
        
        # Check if error terms are already normalised by looking for typical pre-normalisation values
        amplitude_cols = ['A_B_err', 'A_G_err', 'A_R_err']
        background_cols = ['bg_B_err', 'bg_G_err', 'bg_R_err']
        
        # Heuristic check: if error terms are very large (>100), they're likely not normalised
        needs_normalisation = False
        
        if all(col in df.columns for col in amplitude_cols):
            max_amp_err = df[amplitude_cols].max().max()
            if not pd.isna(max_amp_err) and max_amp_err > 100:
                needs_normalisation = True
        
        if all(col in df.columns for col in background_cols):
            max_bg_err = df[background_cols].max().max()
            if not pd.isna(max_bg_err) and max_bg_err > 100:
                needs_normalisation = True
        
        if not needs_normalisation:
            return True
        
        # Normalise error terms
        df_normalised = normalise_error_terms(df)
        
        # No backup needed - we're working on copies in scratch folder
        # Original files on /scratch serve as the backup
        
        # Use IOFunctions to write the database properly
        from IOFunctions import IO_Functions
        io_func = IO_Functions()
        io_func._write_h5_database(df_normalised, h5_file_path, append=False, normalise_photons=False)
        
        return True
        
    except Exception as e:
        print(f"Error processing {h5_file_path}: {str(e)}", file=sys.stderr)
        return False

def normalise_folder(folder_path: str) -> bool:
    """
    Normalise all .h5 files in a folder.
    
    Args:
        folder_path: Path to the folder to process
        
    Returns:
        True if all files processed successfully, False otherwise
    """
    print(f"Normalising .h5 files in: {folder_path}", flush=True)
    
    # Find all .h5 files
    h5_files = find_h5_files(folder_path)
    
    if not h5_files:
        print("No .h5 files found", flush=True)
        return True
    
    print(f"Found {len(h5_files)} .h5 files to process", flush=True)
    
    success_count = 0
    error_count = 0
    
    for i, h5_file_path in enumerate(h5_files, 1):
        filename = os.path.basename(h5_file_path)
        print(f"[{i}/{len(h5_files)}] Processing: {filename}...", end=" ", flush=True)
        
        if process_h5_file(h5_file_path):
            print("✅", flush=True)
            success_count += 1
        else:
            print("❌", flush=True)
            error_count += 1
    
    print(f"Normalisation complete: {success_count} success, {error_count} errors", flush=True)
    
    return error_count == 0

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 post_hoc_normalisation_single.py <folder_path>", file=sys.stderr)
        sys.exit(1)
    
    folder_path = sys.argv[1]
    success = normalise_folder(folder_path)
    sys.exit(0 if success else 1)