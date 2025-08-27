#!/usr/bin/env python3
"""
Simple Crash Test Script
Identifies what's causing terminal crashes in the analysis pipeline
"""

import sys
import os
import traceback
import logging
from datetime import datetime

# Setup logging
log_file = f"crash_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

def test_step(step_name, test_func):
    """Test a step and log results"""
    try:
        logging.info(f"TESTING: {step_name}")
        result = test_func()
        if result:
            logging.info(f"✓ PASSED: {step_name}")
            return True
        else:
            logging.error(f"✗ FAILED: {step_name}")
            return False
    except Exception as e:
        logging.error(f"✗ CRASHED: {step_name} - {e}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        return False

def test_basic_imports():
    """Test basic Python imports"""
    try:
        import numpy as np
        import pandas as pd
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import tifffile
        logging.info("Basic imports successful")
        return True
    except Exception as e:
        logging.error(f"Basic import failed: {e}")
        return False

def test_path_setup():
    """Test path setup"""
    try:
        sys.path.append("..")
        logging.info(f"Added .. to path. Current path: {sys.path[-3:]}")
        return True
    except Exception as e:
        logging.error(f"Path setup failed: {e}")
        return False

def test_project_imports():
    """Test project module imports"""
    try:
        from src import IOFunctions
        logging.info("✓ IOFunctions imported")
        
        from src import Multicolour_Simulation_Functions
        logging.info("✓ Multicolour_Simulation_Functions imported")
        
        from src import SR_Functions
        logging.info("✓ SR_Functions imported")
        
        return True
    except Exception as e:
        logging.error(f"Project import failed: {e}")
        return False

def test_function_initialization():
    """Test function object creation"""
    try:
        from src import IOFunctions, SR_Functions
        
        IO = IOFunctions.IO_Functions()
        logging.info("✓ IO_Functions initialized")
        
        SupRes_F = SR_Functions.SuperRes_Functions()
        logging.info("✓ SuperRes_Functions initialized")
        
        return True
    except Exception as e:
        logging.error(f"Function initialization failed: {e}")
        return False

def test_camera_calibration():
    """Test camera calibration loading"""
    try:
        from src import IOFunctions
        IO = IOFunctions.IO_Functions()
        
        data_folder = '../Camera_Calibrations/Ximea_Camera'
        
        if not os.path.exists(data_folder):
            logging.error(f"Camera calibration folder not found: {data_folder}")
            return False
        
        gain = IO.read_tiff(os.path.join(data_folder, "gain.tif"))
        logging.info(f"✓ Loaded gain map: {gain.shape}")
        
        return True
    except Exception as e:
        logging.error(f"Camera calibration failed: {e}")
        return False

def test_multiprocessing():
    """Test multiprocessing functionality"""
    try:
        from multiprocessing import Pool
        import time
        
        def simple_task(x):
            time.sleep(0.1)
            return x * 2
        
        # Test with context manager (safe)
        with Pool(processes=2) as pool:
            results = pool.map(simple_task, [1, 2, 3, 4])
        
        logging.info(f"✓ Multiprocessing with context manager: {results}")
        
        # Test without context manager (unsafe - this might cause issues)
        pool = Pool(processes=2)
        results = pool.map(simple_task, [1, 2, 3])
        pool.close()
        pool.join()
        
        logging.info(f"✓ Multiprocessing manual cleanup: {results}")
        
        return True
    except Exception as e:
        logging.error(f"Multiprocessing test failed: {e}")
        return False

def test_memory_usage():
    """Test memory monitoring"""
    try:
        import psutil
        import numpy as np
        
        # Create large array to test memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1e6  # MB
        
        # Allocate 100MB array
        large_array = np.random.random((10000, 1000))
        current_memory = process.memory_info().rss / 1e6  # MB
        
        logging.info(f"Memory: Initial={initial_memory:.1f}MB, After allocation={current_memory:.1f}MB")
        
        # Delete and force garbage collection
        del large_array
        import gc
        gc.collect()
        
        final_memory = process.memory_info().rss / 1e6  # MB
        logging.info(f"Memory after cleanup: {final_memory:.1f}MB")
        
        return True
    except Exception as e:
        logging.error(f"Memory test failed: {e}")
        return False

def test_matplotlib_cleanup():
    """Test matplotlib memory management"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Create and close multiple figures
        for i in range(5):
            fig, ax = plt.subplots(figsize=(10, 8))
            x = np.random.random(1000)
            y = np.random.random(1000)
            ax.scatter(x, y)
            
            # Test both cleanup methods
            if i < 3:
                plt.close(fig)  # Explicit close
            else:
                plt.close('all')  # Close all
        
        logging.info("✓ Matplotlib cleanup test passed")
        return True
    except Exception as e:
        logging.error(f"Matplotlib test failed: {e}")
        return False

def main():
    """Run all crash tests"""
    
    logging.info("="*60)
    logging.info("CRASH TEST STARTING")
    logging.info("="*60)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Path Setup", test_path_setup),
        ("Project Imports", test_project_imports),
        ("Function Initialization", test_function_initialization),
        ("Camera Calibration", test_camera_calibration),
        ("Multiprocessing", test_multiprocessing),
        ("Memory Usage", test_memory_usage),
        ("Matplotlib Cleanup", test_matplotlib_cleanup),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        success = test_step(test_name, test_func)
        results.append((test_name, success))
        
        if not success:
            logging.error(f"STOPPING: Test '{test_name}' failed")
            break
    
    logging.info("="*60)
    logging.info("CRASH TEST RESULTS")
    logging.info("="*60)
    
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        logging.info(f"{status}: {test_name}")
    
    logging.info(f"Log saved to: {log_file}")
    
    failed_tests = [name for name, success in results if not success]
    if failed_tests:
        logging.error(f"FAILED TESTS: {', '.join(failed_tests)}")
        return 1
    else:
        logging.info("ALL TESTS PASSED")
        return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)