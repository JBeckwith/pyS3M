#!/usr/bin/env python3
"""
Test script for HDF5 compatibility fix.
Tests the scenario where int16 frame columns cause append issues.
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from IOFunctions import IO_Functions

def test_hdf5_compatibility():
    """Test HDF5 compatibility handling."""
    
    IO = IO_Functions()
    test_file = "test_compatibility.h5"
    
    # Clean up any existing test file
    if os.path.exists(test_file):
        os.remove(test_file)
    
    print("=== Testing HDF5 Compatibility Fix ===")
    
    # Step 1: Create initial data with int16 frame numbers (simulate old data)
    print("Step 1: Creating initial data with int16 frames...")
    initial_data = pd.DataFrame({
        'xc': np.random.rand(100) * 100,
        'yc': np.random.rand(100) * 100, 
        'frame': np.arange(1, 101, dtype='int16'),  # Explicitly int16
        'photons': np.random.randint(500, 2000, 100),
        'A_B': np.random.rand(100) * 0.2,
        'A_G': np.random.rand(100) * 0.3, 
        'A_R': np.random.rand(100) * 0.5
    })
    
    print(f"Initial data frame dtype: {initial_data['frame'].dtype}")
    
    # Write initial data
    IO._write_h5_database(initial_data, test_file, append=False)
    print("✓ Initial data written")
    
    # Step 2: Try to append new data with int32 frame numbers (the problem case)
    print("\nStep 2: Appending new data with int32 frames...")
    new_data = pd.DataFrame({
        'xc': np.random.rand(50) * 100,
        'yc': np.random.rand(50) * 100,
        'frame': np.arange(101, 151, dtype='int32'),  # Explicitly int32
        'photons': np.random.randint(500, 2000, 50),
        'A_B': np.random.rand(50) * 0.2,
        'A_G': np.random.rand(50) * 0.3,
        'A_R': np.random.rand(50) * 0.5
    })
    
    print(f"New data frame dtype: {new_data['frame'].dtype}")
    
    try:
        # This should now work with our fix
        IO._write_h5_database(new_data, test_file, append=True)
        print("✓ Append operation successful!")
        
        # Step 3: Verify the combined data
        print("\nStep 3: Verifying combined data...")
        combined_data = pd.read_hdf(test_file, key="data")
        print(f"Combined data shape: {combined_data.shape}")
        print(f"Combined data frame dtype: {combined_data['frame'].dtype}")
        print(f"Frame range: {combined_data['frame'].min()} to {combined_data['frame'].max()}")
        
        # Check all frames are present
        expected_frames = set(range(1, 151))
        actual_frames = set(combined_data['frame'])
        if expected_frames == actual_frames:
            print("✓ All frames present and correct!")
        else:
            print(f"✗ Frame mismatch. Missing: {expected_frames - actual_frames}")
        
        return True
        
    except Exception as e:
        print(f"✗ Append operation failed: {e}")
        return False
        
    finally:
        # Cleanup
        if os.path.exists(test_file):
            os.remove(test_file)

def test_large_frame_numbers():
    """Test with realistically large frame numbers."""
    
    IO = IO_Functions()
    test_file = "test_large_frames.h5"
    
    # Clean up any existing test file
    if os.path.exists(test_file):
        os.remove(test_file)
        
    print("\n=== Testing Large Frame Numbers ===")
    
    # Test with frame numbers > int16 max (32767)
    large_frame_data = pd.DataFrame({
        'xc': np.random.rand(100) * 100,
        'yc': np.random.rand(100) * 100,
        'frame': np.arange(50000, 50100),  # Definitely > int16 max
        'photons': np.random.randint(500, 2000, 100),
        'A_B': np.random.rand(100) * 0.2,
        'A_G': np.random.rand(100) * 0.3,
        'A_R': np.random.rand(100) * 0.5
    })
    
    print(f"Large frame numbers: {large_frame_data['frame'].min()} to {large_frame_data['frame'].max()}")
    print(f"Frame dtype: {large_frame_data['frame'].dtype}")
    
    try:
        IO._write_h5_database(large_frame_data, test_file, append=False)
        
        # Read back and verify
        read_data = pd.read_hdf(test_file, key="data")
        print(f"Read back frame dtype: {read_data['frame'].dtype}")
        print(f"Read back frame range: {read_data['frame'].min()} to {read_data['frame'].max()}")
        
        if read_data['frame'].dtype in ['int32', 'int64']:
            print("✓ Large frame numbers handled correctly!")
            return True
        else:
            print(f"✗ Frame dtype should be int32/int64, got {read_data['frame'].dtype}")
            return False
            
    except Exception as e:
        print(f"✗ Large frame test failed: {e}")
        return False
        
    finally:
        # Cleanup
        if os.path.exists(test_file):
            os.remove(test_file)

if __name__ == "__main__":
    success1 = test_hdf5_compatibility()
    success2 = test_large_frame_numbers()
    
    print("\n" + "="*50)
    if success1 and success2:
        print("🎉 All tests passed! HDF5 compatibility fix working correctly.")
    else:
        print("❌ Some tests failed. Check the output above.")