#!/usr/bin/env python3
"""
Diagnostic Script for Terminal Crashes
Identifies potential causes of terminal exits beyond memory usage
"""

import sys
import os
import resource
import signal
import psutil
import time
import traceback

def check_system_limits():
    """Check system resource limits that could cause crashes"""
    
    print("="*60)
    print("SYSTEM RESOURCE LIMITS ANALYSIS")
    print("="*60)
    
    # Check memory limits
    try:
        mem_soft, mem_hard = resource.getrlimit(resource.RLIMIT_AS)
        print(f"Memory limit: soft={mem_soft}, hard={mem_hard}")
        if mem_soft != resource.RLIM_INFINITY:
            print(f"  → Memory limit active: {mem_soft / (1024**3):.1f}GB")
    except:
        print("  → Could not check memory limits")
    
    # Check file descriptor limits  
    try:
        fd_soft, fd_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        print(f"File descriptor limit: soft={fd_soft}, hard={fd_hard}")
        if fd_soft < 1024:
            print(f"  ⚠️  Low file descriptor limit: {fd_soft}")
    except:
        print("  → Could not check file descriptor limits")
    
    # Check CPU time limits
    try:
        cpu_soft, cpu_hard = resource.getrlimit(resource.RLIMIT_CPU)
        print(f"CPU time limit: soft={cpu_soft}, hard={cpu_hard}")
        if cpu_soft != resource.RLIM_INFINITY:
            print(f"  → CPU time limit active: {cpu_soft}s")
    except:
        print("  → Could not check CPU time limits")
    
    # Check process limits
    try:
        proc_soft, proc_hard = resource.getrlimit(resource.RLIMIT_NPROC)
        print(f"Process limit: soft={proc_soft}, hard={proc_hard}")
    except:
        print("  → Could not check process limits")

def check_filesystem_issues():
    """Check for filesystem issues that could cause crashes"""
    
    print("\n" + "="*60)
    print("FILESYSTEM ANALYSIS")  
    print("="*60)
    
    # Check /scratch2 availability
    try:
        if os.path.exists('/scratch2'):
            stat = os.statvfs('/scratch2')
            free_gb = (stat.f_frsize * stat.f_available) / (1024**3)
            total_gb = (stat.f_frsize * stat.f_blocks) / (1024**3)
            print(f"/scratch2: {free_gb:.1f}GB free / {total_gb:.1f}GB total")
            if free_gb < 10:
                print(f"  ⚠️  Low disk space on /scratch2: {free_gb:.1f}GB")
        else:
            print("❌ /scratch2 does not exist")
    except Exception as e:
        print(f"❌ /scratch2 access error: {e}")
    
    # Check /tmp space
    try:
        stat = os.statvfs('/tmp')
        free_gb = (stat.f_frsize * stat.f_available) / (1024**3)
        print(f"/tmp: {free_gb:.1f}GB free")
        if free_gb < 1:
            print(f"  ⚠️  Low disk space on /tmp: {free_gb:.1f}GB")
    except Exception as e:
        print(f"❌ /tmp access error: {e}")

def check_import_safety():
    """Test if imports cause crashes"""
    
    print("\n" + "="*60)
    print("IMPORT SAFETY TEST")
    print("="*60)
    
    # Test imports one by one
    imports_to_test = [
        ('numpy', 'import numpy as np'),
        ('matplotlib', 'import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt'),
        ('scipy', 'import scipy'),
        ('scikit-image', 'import skimage'),
        ('opencv', 'import cv2'),
        ('tifffile', 'import tifffile'),
        ('polars', 'import polars'),
        ('pandas', 'import pandas'),
        ('psutil', 'import psutil'),
        ('numba', 'import numba'),
    ]
    
    failed_imports = []
    
    for name, import_cmd in imports_to_test:
        try:
            print(f"Testing {name}...", end=' ')
            exec(import_cmd)
            print("✓")
        except Exception as e:
            print(f"❌ {e}")
            failed_imports.append(name)
    
    if failed_imports:
        print(f"\n⚠️  Failed imports: {', '.join(failed_imports)}")
        print("These could cause segmentation faults or crashes")
    else:
        print("\n✅ All core imports successful")

def check_signal_handling():
    """Test signal handling"""
    
    print("\n" + "="*60)
    print("SIGNAL HANDLING TEST")
    print("="*60)
    
    # Check which signals are available
    available_signals = []
    for sig_name in ['SIGTERM', 'SIGINT', 'SIGKILL', 'SIGQUIT', 'SIGABRT']:
        if hasattr(signal, sig_name):
            available_signals.append(sig_name)
    
    print(f"Available signals: {', '.join(available_signals)}")
    
    # Test signal handler registration
    def test_handler(signum, frame):
        print(f"Caught signal {signum}")
    
    try:
        signal.signal(signal.SIGTERM, test_handler)
        signal.signal(signal.SIGINT, test_handler)
        print("✅ Signal handlers registered successfully")
    except Exception as e:
        print(f"❌ Signal handler registration failed: {e}")

def check_multiprocessing_safety():
    """Test multiprocessing for crashes"""
    
    print("\n" + "="*60)
    print("MULTIPROCESSING SAFETY TEST")
    print("="*60)
    
    try:
        from concurrent.futures import ProcessPoolExecutor
        import time
        
        def simple_task(x):
            time.sleep(0.1)
            return x * 2
        
        print("Testing ProcessPoolExecutor (safe)...")
        with ProcessPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(simple_task, i) for i in range(4)]
            results = [f.result() for f in futures]
        
        print(f"✅ ProcessPoolExecutor test passed: {results}")
        
        # Test without context manager (unsafe)
        print("Testing ProcessPoolExecutor (unsafe)...")
        executor = ProcessPoolExecutor(max_workers=2)
        futures = [executor.submit(simple_task, i) for i in range(4)]
        results = [f.result() for f in futures]
        executor.shutdown(wait=True)
        
        print(f"✅ Unsafe ProcessPoolExecutor test passed: {results}")
        print("  (This could still cause resource leaks over time)")
        
    except Exception as e:
        print(f"❌ ProcessPoolExecutor test failed: {e}")
        traceback.print_exc()

def check_gpu_memory():
    """Check for GPU memory issues"""
    
    print("\n" + "="*60)
    print("GPU MEMORY CHECK")
    print("="*60)
    
    try:
        # Check if NVIDIA GPU is present
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ NVIDIA GPU detected")
            # Look for GPU memory usage
            lines = result.stdout.split('\n')
            for line in lines:
                if 'MiB' in line and ('/' in line):
                    print(f"GPU Memory: {line.strip()}")
        else:
            print("No NVIDIA GPU detected (this is normal for CPU-only systems)")
    except subprocess.TimeoutExpired:
        print("❌ nvidia-smi timeout")
    except FileNotFoundError:
        print("No NVIDIA drivers installed")
    except Exception as e:
        print(f"GPU check error: {e}")

def monitor_process_creation():
    """Monitor for runaway process creation"""
    
    print("\n" + "="*60)
    print("PROCESS MONITORING TEST")
    print("="*60)
    
    try:
        current_process = psutil.Process()
        initial_children = len(current_process.children(recursive=True))
        
        print(f"Initial child processes: {initial_children}")
        
        # Test creating a subprocess
        import subprocess
        result = subprocess.run(['echo', 'test'], capture_output=True)
        
        final_children = len(current_process.children(recursive=True))
        print(f"Final child processes: {final_children}")
        
        if final_children > initial_children:
            print(f"⚠️  Process count increased by {final_children - initial_children}")
        else:
            print("✅ No process leak detected")
            
    except Exception as e:
        print(f"❌ Process monitoring failed: {e}")

def main():
    """Run all diagnostic tests"""
    
    print("TERMINAL CRASH DIAGNOSTIC SCRIPT")
    print("Checking for potential causes of terminal exits...")
    
    # Run all diagnostic checks
    check_system_limits()
    check_filesystem_issues() 
    check_import_safety()
    check_signal_handling()
    check_multiprocessing_safety()
    check_gpu_memory()
    monitor_process_creation()
    
    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)
    
    print("Potential crash causes to investigate:")
    print("1. File descriptor limits (check RLIMIT_NOFILE)")
    print("2. /scratch2 disk space or permissions")  
    print("3. Segmentation faults from native libraries (NumPy, OpenCV, etc.)")
    print("4. GPU memory exhaustion (if using CUDA)")
    print("5. System memory limits (RLIMIT_AS)")
    print("6. Process/thread limit exhaustion")
    print("7. Network filesystem issues with /scratch")
    print("8. Signal handling conflicts")
    
    print("\nRecommendations:")
    print("- Use UltraSafe script with enhanced resource monitoring")
    print("- Set resource limits proactively")
    print("- Monitor file descriptor usage")
    print("- Use sequential processing instead of multiprocessing")
    print("- Add signal handlers for graceful shutdown")
    print("- Monitor system resources in real-time")

if __name__ == "__main__":
    main()