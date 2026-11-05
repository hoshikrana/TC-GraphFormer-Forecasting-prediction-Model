"""
Hardware Discovery Module for TC PRIMED Two-Machine Setup.
Discovers and records CPU, RAM, OS, Python, PyTorch/CUDA, GPU, and disk space
for both Machine A (workstation) and Machine B (remote GPU node).
"""

import os
import sys
import json
import shutil
import platform
from typing import Dict, Any


def discover_hardware(role: str = "workstation") -> Dict[str, Any]:
    """
    Query system resources and return standardized hardware inventory dictionary.
    Role: 'workstation' or 'gpu_node'
    """
    # 1. Operating System & Platform
    os_name = platform.system()
    os_release = platform.release()
    os_version = platform.version()
    machine = platform.machine()
    processor = platform.processor()

    # 2. Python environment
    py_ver = sys.version.split()[0]
    executable = sys.executable

    # 3. CPU & Memory
    cpu_count = os.cpu_count() or 1
    # Physical memory estimation
    total_ram_gb = None
    try:
        import psutil
        total_ram_gb = round(psutil.virtual_memory().total / (1024**3), 2)
        avail_ram_gb = round(psutil.virtual_memory().available / (1024**3), 2)
    except ImportError:
        # Fallback on Windows
        avail_ram_gb = None
        if os_name == "Windows":
            try:
                import ctypes
                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]
                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
                total_ram_gb = round(stat.ullTotalPhys / (1024**3), 2)
                avail_ram_gb = round(stat.ullAvailPhys / (1024**3), 2)
            except Exception:
                pass

    # 4. Storage / Drives
    disks = {}
    if os_name == "Windows":
        for letter in ["C:\\", "D:\\", "E:\\", "F:\\"]:
            if os.path.exists(letter):
                try:
                    tot, used, free = shutil.disk_usage(letter)
                    disks[letter[0]] = {
                        "total_gb": round(tot / (1024**3), 2),
                        "used_gb": round(used / (1024**3), 2),
                        "free_gb": round(free / (1024**3), 2),
                    }
                except Exception:
                    pass
    else:
        tot, used, free = shutil.disk_usage("/")
        disks["root"] = {
            "total_gb": round(tot / (1024**3), 2),
            "used_gb": round(used / (1024**3), 2),
            "free_gb": round(free / (1024**3), 2),
        }

    # 5. PyTorch & CUDA / GPU
    torch_installed = False
    torch_version = None
    cuda_available = False
    cuda_version = None
    gpu_devices = []

    try:
        import torch
        torch_installed = True
        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda if cuda_available else None
        if cuda_available:
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpu_devices.append({
                    "device_index": i,
                    "device_name": props.name,
                    "total_vram_gb": round(props.total_memory / (1024**3), 2),
                    "multi_processor_count": props.multi_processor_count,
                })
    except ImportError:
        pass

    inventory = {
        "compute_profile": {
            "role": role,
            "description": "Primary Workstation (Data Preprocessing, QC, Evaluation)"
            if role == "workstation"
            else "Remote GPU Compute Node (Self-Supervised & Multi-Task Training)",
        },
        "operating_system": {
            "system": os_name,
            "release": os_release,
            "version": os_version,
            "machine": machine,
            "processor": processor,
        },
        "cpu_and_memory": {
            "logical_cpu_cores": cpu_count,
            "total_physical_ram_gb": total_ram_gb,
            "available_physical_ram_gb": avail_ram_gb,
            "recommended_dataloader_workers": min(4, cpu_count // 2) if cpu_count > 2 else 0,
        },
        "storage_drives": disks,
        "python_environment": {
            "python_version": py_ver,
            "python_executable": executable,
            "torch_installed": torch_installed,
            "torch_version": torch_version,
        },
        "gpu_acceleration": {
            "cuda_available": cuda_available,
            "cuda_version": cuda_version,
            "gpu_count": len(gpu_devices),
            "gpu_devices": gpu_devices,
            "infrastructure_note": (
                "Workstation does not have local CUDA acceleration. "
                "AnyDesk provides remote desktop GUI streaming only and does NOT expose remote GPUs "
                "to local Python sessions. All heavy GPU training must execute natively on Machine B."
            ) if not cuda_available and role == "workstation" else (
                "GPU Node environment verified." if cuda_available else "GPU Node pending CUDA driver initialization."
            ),
        },
    }

    return inventory


def export_inventory(role: str = "workstation", output_path: str = None) -> str:
    """Discover hardware and save inventory JSON."""
    inv = discover_hardware(role)
    if output_path is None:
        output_path = f"reports/hardware_inventory_{role}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(inv, f, indent=2)
    return output_path


if __name__ == "__main__":
    r = sys.argv[1] if len(sys.argv) > 1 else "workstation"
    p = sys.argv[2] if len(sys.argv) > 2 else None
    out = export_inventory(r, p)
    print(f"[OK] Hardware inventory saved to: {out}")

