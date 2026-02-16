#!/usr/bin/env python3
"""
Debug frame numbering in HDF5 files.

This script investigates the frame numbering issue by:
1. Reading HDF5 file
2. Analyzing frame number distribution
3. Checking if frames are in order
4. Comparing DataFrame index vs frame column
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def debug_h5_frame_numbering(h5_filepath):
    """Debug frame numbering issues in HDF5 file."""
    
    print(f"=== Debugging HDF5 Frame Numbering ===")
    print(f"File: {h5_filepath}")
    
    if not os.path.exists(h5_filepath):
        print(f"ERROR: File not found: {h5_filepath}")
        return
    
    # Read the HDF5 file
    try:
        with pd.HDFStore(h5_filepath, mode='r') as store:
            if 'data' not in store:
                print("ERROR: No 'data' key found in HDF5 file")
                return
            
            df = store['data']
            print(f"Total localizations: {len(df):,}")
            
    except Exception as e:
        print(f"ERROR reading HDF5 file: {e}")
        return
    
    # Check if frame column exists
    if 'frame' not in df.columns:
        print("ERROR: No 'frame' column found")
        print(f"Available columns: {list(df.columns)}")
        return
    
    # Analyze frame numbers
    frames = df['frame'].values
    print(f"\n=== Frame Analysis ===")
    print(f"Frame range: {frames.min()} to {frames.max()}")
    print(f"Total frames: {frames.max() - frames.min() + 1}")
    print(f"Unique frames: {len(np.unique(frames))}")
    
    # Check if frames are in order
    is_sorted = np.all(frames[:-1] <= frames[1:])
    print(f"Frames in ascending order: {is_sorted}")
    
    if not is_sorted:
        print("\n=== ORDER ANALYSIS ===")
        # Find where frame numbers decrease
        decreases = np.where(frames[:-1] > frames[1:])[0]
        print(f"Found {len(decreases)} positions where frame number decreases:")
        
        for i, pos in enumerate(decreases[:5]):  # Show first 5
            print(f"  Position {pos}: frame {frames[pos]} → {frames[pos+1]}")
            if i >= 4 and len(decreases) > 5:
                print(f"  ... and {len(decreases)-5} more")
                break
    
    # Analyze frame distribution
    frame_counts = pd.Series(frames).value_counts().sort_index()
    print(f"\n=== Frame Distribution ===")
    print(f"Average localizations per frame: {len(frames) / len(frame_counts):.1f}")
    print(f"Min localizations per frame: {frame_counts.min()}")
    print(f"Max localizations per frame: {frame_counts.max()}")
    
    # Check for gaps in frame numbering
    expected_frames = set(range(int(frames.min()), int(frames.max()) + 1))
    actual_frames = set(frames)
    missing_frames = expected_frames - actual_frames
    
    if missing_frames:
        print(f"Missing frames: {len(missing_frames)} (e.g., {sorted(missing_frames)[:10]})")
    else:
        print("No missing frames found")
    
    # Show first and last 20 frame numbers to see pattern
    print(f"\n=== Frame Sequence Preview ===")
    print(f"First 20 frames: {frames[:20].tolist()}")
    print(f"Last 20 frames: {frames[-20:].tolist()}")
    
    # Compare DataFrame index vs frame values
    print(f"\n=== Index vs Frame Comparison ===")
    df_with_index = df.reset_index()
    index_vs_frame_diff = df_with_index.index - df_with_index['frame']
    
    if np.all(index_vs_frame_diff == index_vs_frame_diff[0]):
        print(f"Constant offset between index and frame: {index_vs_frame_diff[0]}")
    else:
        print("Variable offset between index and frame - this indicates the issue!")
        print(f"Index-Frame difference range: {index_vs_frame_diff.min()} to {index_vs_frame_diff.max()}")
    
    # Generate plots if requested
    try:
        # Plot 1: Frame vs row position to see jumps
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(frames, 'b-', linewidth=0.5, alpha=0.7)
        plt.xlabel('Row Position (DataFrame Index)')
        plt.ylabel('Frame Number')
        plt.title('Frame Number vs Row Position')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Localizations per frame
        plt.subplot(1, 2, 2)
        frame_counts.plot(kind='line', linewidth=0.5)
        plt.xlabel('Frame Number')
        plt.ylabel('Number of Localizations')
        plt.title('Localizations per Frame')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = h5_filepath.replace('.h5', '_frame_debug.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nDiagnostic plots saved: {output_path}")
        plt.show()
        
    except Exception as e:
        print(f"Could not generate plots: {e}")
    
    return df

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python debug_frame_numbering.py <path_to_h5_file>")
        print("Example: python debug_frame_numbering.py /path/to/results.h5")
        sys.exit(1)
    
    h5_file = sys.argv[1]
    df = debug_h5_frame_numbering(h5_file)
    
    if df is not None:
        print(f"\n=== Summary ===")
        print("If frames are not in order, use IOFunctions.sort_h5_by_frame() to fix")
        print("If you're plotting and seeing jumps to zero, make sure you're plotting df['frame'], not df.index")