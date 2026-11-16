# Machine B Hardware & Environment Preflight Report

**Project:** TC-GraphFormer — Tropical Cyclone Spatiotemporal Forecasting Framework  
**Host Designation:** Machine B (Target GPU Compute Node / Training Host)  
**Preflight Timestamp:** 2026-11-16T12:58:30+05:30  
**Inspection Type:** Non-Destructive, Read-Only Diagnostics  
**Overall Readiness Verdict:** **NOT READY** *(for native CUDA GPU acceleration)* / **READY WITH CONDITIONS** *(for offline CPU pipeline & smoke-testing)*

---

## 1. Executive Summary & Verdict

A comprehensive, non-destructive preflight audit was executed on the current machine environment designated as Machine B. 

```
+-----------------------------------------------------------------------------------------+
|                                 PREFLIGHT VERDICT                                       |
+-----------------------------------------------------------------------------------------+
| • Native CUDA GPU Training:   NOT READY (Zero NVIDIA GPUs, No CUDA Driver, PyTorch CPU) |
| • Offline CPU Smoke Testing:  READY WITH CONDITIONS (All Python dependencies met)       |
| • Pilot Data Storage:         READY (Raw pilot & processed data already present on D:)  |
+-----------------------------------------------------------------------------------------+
```

### Primary Blockers for GPU Workloads
1. **Zero NVIDIA GPUs Detected:** The host only has an integrated **Intel(R) Iris(R) Xe Graphics** adapter. No discrete NVIDIA GPU is installed or accessible.
2. **Missing NVIDIA Driver & CUDA Toolkit:** Neither `nvidia-smi` nor `nvcc` is installed. `CUDA_PATH` is unset.
3. **CPU-Only PyTorch Build:** Installed PyTorch is `2.15.0+cpu` with `torch.cuda.is_available() == False`.

---

## 2. Hardware Inventory

### 2.1 Central Processing Unit (CPU)
- **Processor Model:** 12th Gen Intel(R) Core(TM) i5-1235U
- **Processor Identifier:** Intel64 Family 6 Model 154 Stepping 4, GenuineIntel
- **Base Frequency:** ~2496 MHz (up to 4.40 GHz Max Turbo)
- **Physical Cores:** 10 cores (2 Performance-cores with Hyper-Threading + 8 Efficient-cores, verified via `GetLogicalProcessorInformation`)
- **Logical Processors / Threads:** 12 threads
- **Architecture:** AMD64 (x86_64, 64-bit WindowsPE)

```text
Evidence:
CPU Model: 12th Gen Intel(R) Core(TM) i5-1235U
Physical Core count via GetLogicalProcessorInformation: 10
Logical CPU Cores (os.cpu_count): 12
NUMBER_OF_PROCESSORS env: 12
PROCESSOR_ARCHITECTURE env: AMD64
```

### 2.2 System Memory (RAM) & Paging
- **Total Physical RAM:** 15.73 GB (16,890,978,304 bytes)
- **Available Physical RAM:** 6.01 GB (6,458,544,128 bytes)
- **Current Memory Load:** 61%
- **Total Pagefile / Commit Limit:** 23.73 GB (25,480,912,896 bytes)
- **Available Pagefile / Commit Available:** 10.66 GB (11,445,776,384 bytes)
- **Swap / Paging Configuration:** Managed Windows paging file on `C:\` (commit limit: physical RAM + 8.00 GB swap)

```text
Evidence (GlobalMemoryStatusEx):
Total Physical RAM: 15.73 GB (16,890,978,304 bytes)
Available Physical RAM: 6.01 GB (6,458,544,128 bytes)
Memory Load: 61%
Total Pagefile (Commit Limit): 23.73 GB (25,480,912,896 bytes)
Available Pagefile (Commit Available): 10.66 GB (11,445,776,384 bytes)
```

### 2.3 Graphics Processing Unit (GPU)
- **Discrete NVIDIA GPUs:** 0
- **Display Adapters Detected:** 1 (Integrated Intel)
  - **Adapter Name:** `Intel(R) Iris(R) Xe Graphics`
  - **Provider:** Intel Corporation
  - **Driver Version:** 31.0.101.4502 (Date: 7-13-2023)
- **Dedicated VRAM:** 0 MB dedicated (shared host memory only)
- **`nvidia-smi` Status:** Not found on PATH or standard directories (`C:\Program Files\NVIDIA Corporation\NVSMI`, `C:\Windows\System32`).
- **GPU Utilization & Temperature:** Unavailable (requires NVIDIA GPU / driver).

```text
Evidence:
nvidia-smi : The term 'nvidia-smi' is not recognized
Adapter [0000]: Intel(R) Iris(R) Xe Graphics (Driver 31.0.101.4502)
CUDA_PATH env: None
nvcc not found in PATH.
No CUDA toolkit directory at: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA
```

### 2.4 Storage Inventory & Filesystems
All mounted volumes are local NTFS partitions:

| Volume | Label | Filesystem | Total Capacity | Used Space | Free Space | Percent Free | Status / Recommendation |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| `C:\` | `Windows-SSD` | NTFS | 396.59 GB | 345.65 GB | **50.94 GB** | 12.8% | System OS & Temp drive |
| `D:\` | `New Volume` | NTFS | 78.12 GB | 46.98 GB | **31.15 GB** | 39.9% | **Current Workspace & Data Volume** |

**Storage Assessment:**
- Drive `D:\` hosts the repository (`D:\Project\Capstone_project`), the existing raw pilot dataset (19.31 GB), and the processed pilot dataset (1.41 GB).
- Drive `D:\` has **31.15 GB** of remaining free space.
- Drive `C:\` has **50.94 GB** of remaining free space.
- **Pilot Dataset Fit:** Sufficient. The 20 GB raw pilot dataset and processed tensors already reside on `D:\`.
- **100 GB Scale Warning:** Neither volume has sufficient headroom to host the upcoming 100 GB dataset (requires ~115–125 GB free space for raw + processed tensors). An external or expanded dedicated data volume (>= 200 GB) will be mandatory before Phase 3.

---

## 3. Software Environment

### 3.1 Operating System & Shell
- **OS:** Microsoft Windows 11 Home / Pro (Build 10.0.26300, Release 11)
- **OS Platform:** `Windows-11-10.0.26300-SP0`
- **Shell:** Windows PowerShell 5.1 / Core

### 3.2 Python & Package Environment
- **Python Executable:** `D:\Project\Capstone_project\.venv\Scripts\python.exe`
- **Python Version:** `3.13.12 (tags/v3.13.12:1cbe481, Feb 3 2026, 18:22:25) [MSC v.1944 64 bit (AMD64)]`
- **Pip Version:** `25.3`
- **Core Scientific Libraries Installed:**
  - `torch`: `2.15.0+cpu`
  - `numpy`: `2.5.4`
  - `scipy`: `1.18.1`
  - `pandas`: `3.1.1`
  - `netCDF4`: `1.7.5.1.post1`
  - `h5py`: `3.16.0`
  - `cftime`: `1.6.6.1`
  - `PyYAML`: `6.0.3`
  - `matplotlib`: `3.11.2`
  - `networkx`: `3.7`
  - `awscli`: `1.46.2`
  - `tc-graphformer`: `1.0.0` (editable mode in `.venv`)

### 3.3 PyTorch & CUDA Diagnostics
- **`torch.__version__`:** `2.15.0+cpu`
- **`torch.cuda.is_available()`:** `False`
- **`torch.version.cuda`:** `None`
- **`torch.backends.cudnn.is_available()`:** `False`
- **PyTorch Visible CUDA Devices:** `0`
- **PyTorch GPU Names:** None (`No CUDA devices accessible to PyTorch`)

### 3.4 CUDA Separation Audit
To prevent common configuration misconceptions, the three layers of CUDA capability were audited independently:
1. **NVIDIA Driver Layer:** **ABSENT**. No NVIDIA display driver exists on this machine.
2. **CUDA Toolkit Layer (`nvcc`):** **ABSENT**. Neither `nvcc.exe` nor `CUDA_PATH` exists.
3. **PyTorch CUDA Runtime Layer:** **ABSENT**. PyTorch was compiled without CUDA support (`+cpu` wheel).

### 3.5 System Archive & Transfer Utilities
- **`git`:** Available (`git version 2.43.0.windows.1`)
- **`tar`:** Available (`bsdtar 3.8.9 - libarchive 3.8.9`, supports `.tar.gz`, `.tar.zst`, `.tar.bz2`)
- **`curl`:** Available (`curl 8.22.0 (Windows)`)
- **`ssh`:** Available (Windows OpenSSH Client)
- **`scp`:** Available (Windows OpenSSH Client)
- **`robocopy`:** Available (Robust File Copy for Windows)
- **`rsync`:** **Not Found** (Windows does not bundle rsync natively)
- **`7z`:** **Not Found**

---

## 4. Project Compatibility & Repository Status

### 4.1 Git Repository Status
- **Current Branch:** `fix/phase2e4-4-3-final-audit-and-metric-closure`
- **Commit SHA:** `fe31a582ad94efd8ebd201ff77fe334909dcd39e`
- **Working Tree State:** Clean (only untracked `reports/selection/` preserved per project policy)
- **Remote Tracking:** Up to date with `origin/fix/phase2e4-4-3-final-audit-and-metric-closure`

### 4.2 Training Configuration Audit
- **`configs/gpu_node.yaml`:**
  - `device: "cuda"`
  - `cuda_device_id: 0`
  - `precision: "fp16"` (with AMP enabled)
  - `num_workers: 8`
  - `batch_size: 32`
- **Compatibility Conflict:**
  - `configs/gpu_node.yaml` cannot be run as-is on this machine because `--device cuda` will fail due to `torch.cuda.is_available() == False`.
  - However, `src/training/train_multitask.py` includes automatic fallback:
    ```python
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ```
  - When running on `cpu`, automatic mixed precision (AMP) is automatically disabled (`use_amp = False`) and `pin_memory = False`.

---

## 5. Dataset Storage Readiness

Audit of local dataset files on `D:\Project\Capstone_project`:

| Dataset Category | Path | File Count | Size on Disk | Manifest Reference |
|:---|:---|:---:|:---:|:---|
| **Raw Pilot NetCDF** | `data/raw/pilot` | 2,694 files | 19.31 GB | 2,679 records (18.63 GB uncompressed) |
| **Processed Tensors** | `data/processed/pilot` | 80 storm containers | 1.41 GB | 80 storms in `storm_split_manifest.csv` |
| **Sample Index** | `reports/pipeline/sample_index.csv` | 2,599 samples | 1.01 MB | 2,599 valid multi-modal sequence windows |
| **Checkpoints** | `checkpoints/` | 12 files | 158.8 MB | Pretrained backbone & baseline models |
| **Staging Area** | `staging/` | 1 file | 35.1 KB | `staging_manifest.json` ready for packaging |

- **Dataset Transfer Necessity for Pilot:** **Zero transfer required.** The entire pilot dataset (19.31 GB raw + 1.41 GB processed) already resides locally on Drive `D:\`.
- **Available Storage Headroom on `D:\`:** 31.15 GB available.

---

## 6. Blockers, Risks & Smallest Corrective Actions

| Item | Severity | Root Cause | Impact | Smallest Corrective Action |
|:---|:---:|:---|:---|:---|
| **No NVIDIA GPU Hardware** | **CRITICAL (for GPU)** | Host is a laptop / workstation equipped only with Intel Core i5-1235U and Intel Iris Xe Graphics. | Cannot execute CUDA kernels, FP16 AMP, or high-throughput GPU training. | Use this machine strictly for CPU smoke-testing / evaluation, or connect to a dedicated GPU server/instance (e.g., cloud A100/RTX 4090 or separate GPU desktop) for Machine B training. |
| **PyTorch CPU Build** | **CRITICAL (for GPU)** | `torch 2.15.0+cpu` installed. | PyTorch cannot see or utilize CUDA devices even if an external eGPU were connected. | When migrating to a GPU host, install CUDA PyTorch wheel (e.g. `pip install torch --index-url https://download.pytorch.org/whl/cu121`). |
| **Dataloader Worker Count** | **MODERATE** | `configs/gpu_node.yaml` requests `num_workers: 8`. | 10 physical cores / 12 threads with only 6 GB available RAM may cause memory paging under 8 workers. | For local testing, override `--num-workers 2` or `--num-workers 4`. |
| **Storage Capacity for 100 GB** | **FUTURE BLOCKER** | Drive `D:\` has 31.15 GB free; Drive `C:\` has 50.94 GB free. | Sufficient for 20 GB pilot, but will exhaust disk space during 100 GB dataset expansion. | Attach an external SSD (>= 256 GB) or provision high-capacity cloud volume for Phase 3. |

---

## 7. Operational Guidance for Next Step

### Can we run Pilot Smoke-Testing on this machine right now?
**YES (in CPU Mode).**  
Because all pilot raw and processed data are already present on disk, a CPU smoke test can be executed immediately:
```bash
.venv\Scripts\python.exe src/training/train_multitask.py \
    --device cpu \
    --precision fp32 \
    --no-amp \
    --max-train-samples 4 \
    --max-val-samples 4 \
    --epochs 1
```

### Can we run GPU Smoke-Testing or GPU Training on this machine?
**NO.**  
This host does not possess an NVIDIA discrete GPU or CUDA runtime. True GPU smoke-testing and multi-task GPU training must be executed on a machine equipped with an NVIDIA GPU.
