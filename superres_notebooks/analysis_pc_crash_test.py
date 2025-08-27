#!/usr/bin/env python3
"""
Comprehensive test script for analysis PC to identify MemorySafe script crash
Run this on the analysis PC to diagnose the exact issue.
"""

import sys
import os
import traceback

# Add paths - adjust for running from superres_notebooks
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Go up one level from superres_notebooks
src_dir = os.path.join(project_root, 'src')

if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
    
print(f"Added to Python path: {src_dir}")

def test_step(step_name, test_func):
    """Helper to run tests with clear output"""
    print(f"Testing: {step_name}")
    try:
        result = test_func()
        print(f"   ✅ PASS: {step_name}")
        return True, result
    except Exception as e:
        print(f"   ❌ FAIL: {step_name}")
        print(f"      Error: {str(e)}")
        print(f"      Type: {type(e).__name__}")
        traceback.print_exc()
        return False, None

def main():
    print("="*60)
    print("ANALYSIS PC CRASH DIAGNOSTIC")
    print("="*60)
    
    # Test folder that crashes
    crash_folder = "/scratch/sycamore-asap/ASAP_Members_Other_Imaging_Data/Brendan/20250404_Ximea_AsynNRThX/data/10umAsyn_AuNPs_5nMNR_200nMThX_488LP785SP_100mW488_1"
    
    print(f"Target folder: {crash_folder}")
    print(f"Folder exists: {os.path.exists(crash_folder)}")
    
    if os.path.exists(crash_folder):
        files = os.listdir(crash_folder)
        print(f"Files in folder ({len(files)}): {files[:10]}")  # Show first 10
    
    # Test 1: Basic imports
    def test_imports():
        import SR_Functions
        import HelperFunctions
        import IOFunctions
        import MaskFunctions
        import numpy as np
        return "All imports successful"
    
    success, _ = test_step("Basic imports", test_imports)
    if not success:
        return
    
    # Test 2: Function initialization
    def test_function_init():
        import SR_Functions
        import HelperFunctions
        import IOFunctions
        import MaskFunctions
        
        sr_f = SR_Functions.SuperRes_Functions()
        h_f = HelperFunctions.Helper_Functions()
        io_f = IOFunctions.IO_Functions()
        m_f = MaskFunctions.Mask_Functions()
        
        return "Function instances created"
    
    success, _ = test_step("Function initialization", test_function_init)
    if not success:
        return
    
    # Test 3: File search operations
    def test_file_search():
        from HelperFunctions import Helper_Functions
        H_F = Helper_Functions()
        
        # Test with the actual crash folder
        if os.path.exists(crash_folder):
            image_files = H_F.file_search(crash_folder, ".tif", "")
            metadata_files = H_F.file_search(crash_folder, "metadata", "")
            
            return f"Found {len(image_files)} images, {len(metadata_files)} metadata files"
        else:
            # Test with a known existing folder
            test_folder = "/tmp"
            image_files = H_F.file_search(test_folder, ".tmp", "")
            return f"Test search on /tmp returned {type(image_files)} with {len(image_files)} results"
    
    success, result = test_step("File search operations", test_file_search)
    print(f"      Result: {result}")
    
    # Test 4: Camera data loading  
    def test_camera_data():
        import tifffile
        import numpy as np
        
        cam_dir = os.path.join(project_root, "Camera_Calibrations", "Ximea_Camera")
        gain_path = os.path.join(cam_dir, "gain.tif")
        
        if os.path.exists(gain_path):
            gain = tifffile.imread(gain_path).astype(np.float32)
            return f"Camera data loaded: shape {gain.shape}, dtype {gain.dtype}"
        else:
            return "Camera calibration files not found"
    
    success, result = test_step("Camera data loading", test_camera_data)
    print(f"      Result: {result}")
    
    # Test 5: Metadata reading (if folder exists)
    if os.path.exists(crash_folder):
        def test_metadata_reading():
            from HelperFunctions import Helper_Functions
            from IOFunctions import IO_Functions
            
            H_F = Helper_Functions()
            IO = IO_Functions()
            
            metadata_files = H_F.file_search(crash_folder, "metadata", "")
            
            if len(metadata_files) == 0:
                return "ERROR: No metadata files found - this is the crash cause!"
            
            # Try to read the metadata
            start_x, start_y, width, height = IO.metadata_reader_imageJ(metadata_files[0])
            return f"Metadata read: x={start_x}, y={start_y}, w={width}, h={height}"
        
        success, result = test_step("Metadata reading", test_metadata_reading)
        print(f"      Result: {result}")
        
        if not success or "ERROR" in str(result):
            print("\n🎯 CRASH CAUSE IDENTIFIED:")
            print("   The folder has no metadata files!")
            print("   SR_Functions.fit_imaging_data() crashes when accessing metadatafiles[0]")
            print("   Fix: Add validation for empty file arrays")
            return
    
    # Test 6: Simulate the exact SR_Functions call
    def test_sr_functions_call():
        import SR_Functions
        import numpy as np
        
        # Create minimal test data
        sr_f = SR_Functions.SuperRes_Functions()
        
        # Load actual camera parameters like the real script
        import tifffile
        cam_dir = os.path.join(project_root, "Camera_Calibrations", "Ximea_Camera")
        
        gain = tifffile.imread(os.path.join(cam_dir, "gain.tif")).astype(np.float32)
        offset = tifffile.imread(os.path.join(cam_dir, "offset.tif")).astype(np.float32)
        variance = tifffile.imread(os.path.join(cam_dir, "variance.tif")).astype(np.float32)
        readnoise = tifffile.imread(os.path.join(cam_dir, "readnoise.tif")).astype(np.float32)
        rqe = tifffile.imread(os.path.join(cam_dir, "rqe.tif")).astype(np.float32)
        
        # Setup smoothing function like the real script
        import types
        import sCMOSFunctions
        
        scmos_f = sCMOSFunctions.sCMOS_Functions()
        
        smoothing_function = types.SimpleNamespace()
        smoothing_function.args = {"sigma": 1.5}
        smoothing_function.extent = 1.5
        smoothing_function.smoothing_function = scmos_f.gaussian_filter_stack
        smoothing_function.data_arg = "image"
        
        # Try the call (will likely fail due to missing files)
        if os.path.exists(crash_folder):
            sr_f.fit_imaging_data(
                crash_folder,
                smoothing_function,  # Now properly configured
                gain, offset, rqe, readnoise,  # camera params
                variance=variance,
                pfa=1e-4, ROI_size=12,
                peak_wavelength=0.55,
                NA=1.49, pixel_size=0.069,
                image_type=".tif"
            )
        
        return "SR_Functions call completed"
    
    success, result = test_step("SR_Functions call simulation", test_sr_functions_call)
    print(f"      Result: {result}")
    
    print("\n" + "="*60)
    print("DIAGNOSTIC COMPLETE")
    print("="*60)
    
    if not success:
        print("🎯 The crash occurs during the SR_Functions call.")
        print("   Most likely causes:")
        print("   1. Missing metadata files in the target folder")
        print("   2. Invalid file paths or permissions")
        print("   3. Array indexing errors in file operations")

if __name__ == "__main__":
    main()