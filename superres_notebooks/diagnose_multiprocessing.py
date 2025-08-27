#!/usr/bin/env python3
"""
Diagnose specific multiprocessing issues causing terminal crashes
"""

import sys
import os
import multiprocessing
import resource
import psutil
import time
import traceback

def check_multiprocessing_config():
    """Check multiprocessing configuration that might cause crashes"""
    
    print("="*60)
    print("MULTIPROCESSING CONFIGURATION ANALYSIS")
    print("="*60)
    
    # Check CPU count and worker calculations
    cpu_count = multiprocessing.cpu_count()
    print(f"CPU count: {cpu_count}")
    
    # Check what the analysis code would use
    spot_detection_workers = min(60, max(1, int(0.9 * cpu_count)))
    image_analysis_workers = min(60, max(1, int(0.75 * cpu_count)))
    postprocess_workers = min(60, max(1, int(0.75 * cpu_count)))
    
    print(f"SpotDetection workers would use: {spot_detection_workers}")
    print(f"ImageAnalysis workers would use: {image_analysis_workers}")  
    print(f"PostProcess workers would use: {postprocess_workers}")
    
    # Check task calculations (these could be huge!)
    print(f"\nTask calculations for typical datasets:")
    for n_frames in [1000, 10000, 50000]:
        n_tasks = min(100 * spot_detection_workers, n_frames)
        print(f"  {n_frames} frames → {n_tasks} tasks ({100 * spot_detection_workers} max)")
    
    # Check if too many workers/tasks
    warnings = []
    if spot_detection_workers > 32:
        warnings.append(f"SpotDetection: {spot_detection_workers} workers (may be too many)")
    if image_analysis_workers > 32:
        warnings.append(f"ImageAnalysis: {image_analysis_workers} workers (may be too many)")
    
    if warnings:
        print(f"\n⚠️  Potential issues:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print(f"\n✅ Worker counts seem reasonable")

def test_process_creation():
    """Test if process creation itself causes issues"""
    
    print("\n" + "="*60)
    print("PROCESS CREATION TEST")
    print("="*60)
    
    try:
        from concurrent.futures import ProcessPoolExecutor
        
        def simple_task(x):
            import time
            time.sleep(0.01)
            return x * 2
        
        # Test with different worker counts
        for workers in [1, 2, 4, 8, 16]:
            print(f"Testing {workers} workers...", end=" ")
            try:
                with ProcessPoolExecutor(max_workers=workers) as executor:
                    futures = [executor.submit(simple_task, i) for i in range(workers * 2)]
                    results = [f.result() for f in futures]
                print(f"✓ ({len(results)} results)")
            except Exception as e:
                print(f"❌ {e}")
                if workers <= 4:  # If even small counts fail, it's a fundamental issue
                    print("CRITICAL: Even small worker counts fail!")
                    return False
        
        return True
        
    except Exception as e:
        print(f"❌ Process creation test failed: {e}")
        traceback.print_exc()
        return False

def test_resource_limits_under_load():
    """Test if resource limits cause crashes under multiprocessing load"""
    
    print("\n" + "="*60)
    print("RESOURCE LIMITS UNDER LOAD TEST")
    print("="*60)
    
    try:
        # Monitor file descriptors during process creation
        process = psutil.Process()
        
        print("Monitoring file descriptors during ProcessPoolExecutor creation...")
        
        from concurrent.futures import ProcessPoolExecutor
        
        def dummy_task(x):
            return x
        
        initial_fds = process.num_fds() if hasattr(process, 'num_fds') else "N/A"
        print(f"Initial FDs: {initial_fds}")
        
        # Create executor and monitor FD usage
        with ProcessPoolExecutor(max_workers=8) as executor:
            during_fds = process.num_fds() if hasattr(process, 'num_fds') else "N/A"
            print(f"FDs with executor: {during_fds}")
            
            # Submit tasks
            futures = [executor.submit(dummy_task, i) for i in range(100)]
            during_tasks_fds = process.num_fds() if hasattr(process, 'num_fds') else "N/A"
            print(f"FDs with 100 tasks: {during_tasks_fds}")
            
            # Get results
            results = [f.result() for f in futures]
            
        final_fds = process.num_fds() if hasattr(process, 'num_fds') else "N/A"
        print(f"Final FDs: {final_fds}")
        
        # Check if FD count increased dramatically
        if isinstance(initial_fds, int) and isinstance(during_tasks_fds, int):
            fd_increase = during_tasks_fds - initial_fds
            if fd_increase > 50:
                print(f"⚠️  Large FD increase: +{fd_increase}")
                return False
        
        print("✅ File descriptor usage seems normal")
        return True
        
    except Exception as e:
        print(f"❌ Resource limits test failed: {e}")
        traceback.print_exc()
        return False

def test_pickling_issues():
    """Test for pickling issues that cause crashes"""
    
    print("\n" + "="*60)
    print("PICKLING ISSUES TEST")
    print("="*60)
    
    try:
        from concurrent.futures import ProcessPoolExecutor
        import pickle
        
        # Test pickling different types of functions/objects that the analysis might use
        test_objects = [
            ("simple function", lambda x: x * 2),
            ("numpy array", __import__('numpy').array([1, 2, 3])),
            ("nested function", lambda: lambda x: x * 2),
        ]
        
        for name, obj in test_objects:
            print(f"Testing pickle of {name}...", end=" ")
            try:
                pickled = pickle.dumps(obj)
                unpickled = pickle.loads(pickled)
                print("✓")
            except Exception as e:
                print(f"❌ {e}")
        
        # Test actual ProcessPoolExecutor with potential problem functions
        print("\nTesting ProcessPoolExecutor with different function types...")
        
        # Global function (should work)
        def global_task(x):
            return x * 2
        
        with ProcessPoolExecutor(max_workers=2) as executor:
            future = executor.submit(global_task, 5)
            result = future.result()
            print(f"Global function: ✓ (result: {result})")
        
        print("✅ Pickling tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Pickling test failed: {e}")
        traceback.print_exc()
        return False

def check_system_overcommit():
    """Check if system overcommit is causing issues"""
    
    print("\n" + "="*60)
    print("SYSTEM OVERCOMMIT CHECK")
    print("="*60)
    
    try:
        # Check memory overcommit settings
        try:
            with open('/proc/sys/vm/overcommit_memory', 'r') as f:
                overcommit = f.read().strip()
            print(f"Memory overcommit setting: {overcommit}")
            
            overcommit_meanings = {
                '0': 'Heuristic overcommit (default)',
                '1': 'Always overcommit',
                '2': 'Don\'t overcommit'
            }
            
            meaning = overcommit_meanings.get(overcommit, 'Unknown')
            print(f"  → {meaning}")
            
            if overcommit == '2':
                print("⚠️  No overcommit mode may cause fork() failures with many processes")
                
        except (FileNotFoundError, PermissionError):
            print("Cannot read overcommit settings (may not be Linux)")
        
        # Check available memory vs what processes might need
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        
        # Estimate memory usage with many workers
        cpu_count = multiprocessing.cpu_count()
        workers = min(60, max(1, int(0.9 * cpu_count)))
        
        # Each Python process might use ~50MB baseline + analysis data
        estimated_memory_gb = workers * 0.05  # 50MB per worker
        
        print(f"Available memory: {available_gb:.1f} GB")
        print(f"Estimated memory for {workers} workers: {estimated_memory_gb:.1f} GB")
        
        if estimated_memory_gb > available_gb * 0.8:
            print("⚠️  High memory usage expected - may cause crashes")
            return False
        
        print("✅ Memory availability looks OK")
        return True
        
    except Exception as e:
        print(f"❌ System overcommit check failed: {e}")
        return True  # Don't fail on this check

def main():
    """Run all multiprocessing diagnostics"""
    
    print("MULTIPROCESSING CRASH DIAGNOSTIC")
    print("Investigating why ProcessPoolExecutor causes terminal crashes...")
    
    results = []
    
    # Run all tests
    tests = [
        ("Multiprocessing Configuration", check_multiprocessing_config),
        ("Process Creation", test_process_creation),
        ("Resource Limits Under Load", test_resource_limits_under_load),
        ("Pickling Issues", test_pickling_issues),
        ("System Overcommit", check_system_overcommit),
    ]
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print('='*60)
        
        try:
            if test_name == "Multiprocessing Configuration":
                test_func()  # This one doesn't return a result
                results.append((test_name, True))
            else:
                result = test_func()
                results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("MULTIPROCESSING DIAGNOSTIC SUMMARY")
    print("="*60)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    failed_tests = [name for name, result in results if not result]
    
    if failed_tests:
        print(f"\n⚠️  Failed tests: {', '.join(failed_tests)}")
        print("\nLikely crash causes:")
        if "Process Creation" in failed_tests:
            print("- ProcessPoolExecutor fundamentally broken on this system")
        if "Resource Limits Under Load" in failed_tests:
            print("- File descriptor or memory limits too low")
        if "Pickling Issues" in failed_tests:
            print("- Function serialization problems")
        if "System Overcommit" in failed_tests:
            print("- System memory overcommit issues")
    else:
        print("\n✅ All tests passed - multiprocessing should work")
        print("The crashes may be caused by:")
        print("- Specific analysis functions that can't be pickled")
        print("- Too many simultaneous processes for your system")
        print("- Analysis code creating nested multiprocessing")

if __name__ == "__main__":
    main()