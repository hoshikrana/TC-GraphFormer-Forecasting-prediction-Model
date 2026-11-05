#!/usr/bin/env python3
"""
Pre-flight verification script for TC-GraphFormer environment.
Validates Python, PyTorch, CUDA devices, essential packages, and project directories.
"""

import os
import sys
import platform

def check_system():
    print("=" * 65)
    print("TC-GRAPHFORMER SYSTEM PRE-FLIGHT VERIFICATION")
    print("=" * 65)
    
    # 1. Platform and Python
    py_ver = sys.version.split()[0]
    print(f"OS Platform:       {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Python Version:    {py_ver} (Executable: {sys.executable})")
    
    # 2. PyTorch and Hardware Acceleration
    try:
        import torch
        print(f"PyTorch Version:   {torch.__version__}")
        cuda_avail = torch.cuda.is_available()
        print(f"CUDA Available:    {cuda_avail}")
        if cuda_avail:
            print(f"CUDA Device Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  Device {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB VRAM)")
        else:
            print("  [INFO] Running on CPU device. For heavy training, CUDA is recommended.")
    except ImportError:
        print("[FAIL] PyTorch is not installed!")
        return False
        
    # 3. Essential Libraries
    required_packages = ["numpy", "pandas", "netCDF4", "scipy", "yaml", "networkx"]
    missing = []
    for pkg in required_packages:
        try:
            __import__(pkg)
            print(f"Package {pkg:<12}: Available")
        except ImportError:
            print(f"Package {pkg:<12}: [MISSING]")
            missing.append(pkg)
            
    if missing:
        print(f"\n[FAIL] Missing required packages: {missing}")
        print("Run: pip install -r requirements.txt")
        return False
        
    # 4. Project Directory Layout
    print("\nVerifying Project Directory Structure:")
    required_dirs = ["configs", "src", "tcprimed_pipeline", "tests", "reports"]
    for d in required_dirs:
        if os.path.isdir(d):
            print(f"Directory {d:<20}: Found")
        else:
            print(f"Directory {d:<20}: [NOT FOUND]")
            return False
            
    print("\n[SUCCESS] Environment verification passed cleanly.")
    print("=" * 65)
    return True

if __name__ == "__main__":
    success = check_system()
    sys.exit(0 if success else 1)
