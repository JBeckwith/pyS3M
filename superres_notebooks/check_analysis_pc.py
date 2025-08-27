#!/usr/bin/env python3
"""
Check analysis PC configuration for multiprocessing issues
"""

import multiprocessing
import resource
import psutil
import os
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import traceback

def main():
    print("ANALYSIS PC CONFIGURATION CHECK")
    print("="*60)
    
    # CPU information
    cpu_count = multiprocessing.cpu_count()
    print(f"CPU cores: {cpu_count}")
    
    # Calculate what the analysis code would use
    spot_workers = min(60, max(1, int(0.9 * cpu_count)))
    image_workers = min(60, max(1, int(0.75 * cpu_count)))
    
    print(f"SpotDetection would use: {spot_workers} workers")
    print(f"ImageAnalysis would use: {image_workers} workers")
    
    # File descriptor limits
    fd_soft, fd_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    print(f"\nFile descriptor limits:")
    print(f"  Soft limit: {fd_soft}")
    print(f"  Hard limit: {fd_hard}")
    
    # Memory info
    memory = psutil.virtual_memory()
    print(f"\nMemory:")
    print(f"  Total: {memory.total / (1024**3):.1f} GB")
    print(f"  Available: {memory.available / (1024**3):.1f} GB")
    print(f"  Used: {memory.percent:.1f}%")
    
    # Process limits
    try:
        proc_soft, proc_hard = resource.getrlimit(resource.RLIMIT_NPROC)
        print(f"\nProcess limits:")
        print(f"  Soft limit: {proc_soft}")
        print(f"  Hard limit: {proc_hard}")
    except:
        print(f"\nProcess limits: Not available")
    
    # Estimated resource usage
    print(f"\nEstimated resource usage with parallel processing:")
    print(f"  Max workers: {spot_workers}")
    print(f"  Est. memory per worker: ~100MB")
    print(f"  Total memory needed: ~{spot_workers * 0.1:.1f} GB")
    
    # Check for problematic configurations
    warnings = []
    
    if spot_workers > 32:
        warnings.append(f"Very high worker count: {spot_workers}")
    
    if fd_soft < 4096:
        warnings.append(f"Low file descriptor limit: {fd_soft}")
    
    if memory.available < (spot_workers * 0.1 * 1024**3):
        warnings.append(f"May not have enough memory for {spot_workers} workers")
    
    if warnings:
        print(f"\n⚠️  POTENTIAL ISSUES:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print(f"\n✅ Configuration looks reasonable")
    
    # System info
    print(f"\nSystem info:")
    print(f"  Python version: {sys.version}")
    print(f"  Platform: {sys.platform}")
    
    # Check if running in virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print(f"  Virtual environment: Active")
    else:
        print(f"  Virtual environment: Not detected")
    
    # Test multiprocessing functionality
    print(f"\n" + "="*60)
    print("MULTIPROCESSING FUNCTIONALITY TEST")
    print("="*60)
    
    multiprocessing_ok = test_multiprocessing_fix()
    
    if multiprocessing_ok:
        print("✅ Multiprocessing fix working correctly")
    else:
        print("❌ Multiprocessing issues detected")
        return False
    
    return True


def simple_task(x):
    """Simple standalone function for multiprocessing test."""
    return x * 2


def test_array_processing(data):
    """Test array processing similar to analysis functions."""
    # Simulate some array operations like puncta fitting
    result = np.sum(data) + np.mean(data)
    return result


def test_multiprocessing_fix():
    """Test that multiprocessing works with the fix."""
    print("Testing multiprocessing functionality...")
    
    try:
        # Test 1: Basic ProcessPoolExecutor functionality
        print("  Test 1: Basic ProcessPoolExecutor...")
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(simple_task, i) for i in range(10)]
            results = [f.result() for f in futures]
        
        expected = [i * 2 for i in range(10)]
        if results == expected:
            print("    ✅ Basic ProcessPoolExecutor working")
        else:
            print("    ❌ Basic ProcessPoolExecutor failed")
            return False
        
        # Test 2: Array processing (similar to actual analysis workload)
        print("  Test 2: Array processing...")
        test_arrays = [np.random.randn(50, 50).astype(np.float32) for _ in range(8)]
        
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(test_array_processing, arr) for arr in test_arrays]
            results = [f.result() for f in futures]
        
        if len(results) == 8 and all(isinstance(r, (int, float, np.number)) for r in results):
            print("    ✅ Array processing working")
        else:
            print("    ❌ Array processing failed")
            return False
        
        # Test 3: Stress test with realistic worker count
        spot_workers = min(60, max(1, int(0.9 * multiprocessing.cpu_count())))
        print(f"  Test 3: Stress test with {spot_workers} workers...")
        
        # Create tasks similar to spot detection workload
        n_tasks = min(100, spot_workers * 4)  # Reasonable number of tasks
        
        with ProcessPoolExecutor(max_workers=spot_workers) as executor:
            futures = [executor.submit(simple_task, i) for i in range(n_tasks)]
            results = [f.result() for f in futures]
        
        expected = [i * 2 for i in range(n_tasks)]
        if results == expected:
            print(f"    ✅ Stress test with {spot_workers} workers successful")
        else:
            print(f"    ❌ Stress test failed")
            return False
        
        # Test 4: Memory handling test
        print("  Test 4: Memory handling...")
        large_arrays = [np.random.randn(100, 100).astype(np.float32) for _ in range(20)]
        
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(test_array_processing, arr) for arr in large_arrays]
            results = [f.result() for f in futures]
        
        if len(results) == 20:
            print("    ✅ Memory handling test successful")
        else:
            print("    ❌ Memory handling test failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"    ❌ Multiprocessing test failed with error: {e}")
        print(f"    Error details: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)