# TC PRIMED Phase 2D — Infrastructure and Two-Machine Readiness Audit

## 1. Executive Summary

Scaling the tropical cyclone intelligence system from the current 20 GB development pilot to a 500 GB curated archive (2000–2025) requires a functional two-machine distributed pipeline. This audit evaluates the actual physical compute, storage capacity, transport protocols, and hardware inventory records across both machines.

The audit reveals two major infrastructure blockers:
1. **Severe Internal Storage Exhaustion on Machine A**: Internal drives possess only 70.19 GB of total free space, creating an immediate 538.4 GB deficit against the required 608.6 GB footprint.
2. **Unverified GPU Compute on Machine B**: The existing GPU hardware report (`hardware_inventory_gpu_node.json`) is a local placeholder. No headless CUDA execution environment has been established on the professor's computer.

---

## 2. Machine A (Friend's Laptop / Workstation) Audit

- **Operating System**: Windows 11 AMD64 (Build 10.0.26300)
- **CPU & Memory**: 12 Logical Cores (Intel Core i7/i9 12th Gen, Model 154 Stepping 4); 15.73 GB Physical RAM (3.88 GB available).
- **Python Environment**: Python 3.13.12 (Virtualenv: `D:\Project\Capstone_project\.venv\Scripts\python.exe`); PyTorch 2.15.0+cpu (No local CUDA).
- **Storage Profile**:
  - Drive `C:`: 396.59 GB Total, 357.73 GB Used, **38.86 GB Free (90.2% full)**.
  - Drive `D:`: 78.12 GB Total, 46.80 GB Used, **31.33 GB Free (59.9% full)**.
  - Total Workstation Free Space: **70.19 GB**.

**Verdict**: Machine A has sufficient CPU compute and RAM to run data extraction, quality control, and preprocessing. However, its internal storage is completely incapable of hosting the 500 GB raw dataset.

---

## 3. Machine B (Professor's Computer / Remote GPU Node) Audit

### 3.1 Investigation of `reports/hardware_inventory_gpu_node.json`
An audit of `reports/hardware_inventory_gpu_node.json` revealed that its contents are an exact duplicate of `reports/hardware_inventory_workstation.json`:
- Same processor: `Intel64 Family 6 Model 154 Stepping 4`
- Same RAM: `15.73 GB`
- Same drive capacities: `C: 396.59 GB, D: 78.12 GB`
- Same PyTorch: `2.15.0+cpu`, `cuda_available: false`

**Finding**: `hardware_inventory_gpu_node.json` was generated locally on Machine A as a placeholder or testing stub. The professor's remote computer was never probed directly by Python or the hardware discovery script. Its actual GPU architecture (e.g. RTX 3090, RTX 4090, A100), VRAM capacity, CUDA toolkit version, driver state, and available storage remain **entirely unverified**.

### 3.2 The AnyDesk Fallacy
AnyDesk is a desktop video-streaming tool. It does not provide network port forwarding, remote CUDA context sharing, or distributed PyTorch execution. Running Python scripts on Machine A while viewing Machine B via AnyDesk executes 100% of the computation on Machine A's local CPU. A real headless environment (SSH server + Git + Conda/Venv + CUDA 12.x) must be deployed on Machine B.

---

## 4. Staging Integrity & File Count Reconciliation

The staging manifest `staging/staging_manifest.json` tracks **82 files**:
- **80 Processed Cyclone Containers**: `data/processed/pilot/.../*_processed.nc` (HDF5 format, 1,509,537,971 bytes).
- **2 Metadata CSV Files**:
  1. `staging/sample_index.csv` (1,012,700 bytes, 2,599 observation rows).
  2. `staging/storm_summary.csv` (4,523 bytes, 80 storm summary rows).
- Total Staged Volume: **1,510,555,194 bytes (1.41 GiB / 1.51 GB)**.
- Cryptographic Verification: **82 out of 82 files match SHA-256 hashes with 100.00% bitwise integrity**.

**Reconciliation**: The 80 containers represent the 80 individual storm HDF5 files; the additional 2 files are the supervisory metadata manifests required for training batch collation on Machine B.

---

## 5. Storage Arithmetic Discrepancy Resolution

Previous documentation contained an apparent arithmetic discrepancy:
$$\text{Estimate A: } 500\text{ GB (raw)} + 37.7\text{ GB (processed)} + 20\text{ GB (reserve)} = \mathbf{557.7\text{ GB}}$$
$$\text{Estimate B (Phase 2C Comprehensive Budget): } \mathbf{608.6\text{ GB}}$$
$$\text{Discrepancy: } 608.6 - 557.7 = \mathbf{50.9\text{ GB}}$$

### Itemized Reconciliation of the Additional 50.9 GB:
1. **Interleaved Shards & Fast Staging Cache (+40.0 GB)**:  
   `src/infrastructure/export_shards.py` pre-packs HDF5 observations into sequential binary shards to eliminate random seek latency across thousands of files on Machine B. For 64,800 overpasses, this staging cache requires ~40 GB.
2. **Model Checkpoints, Optimizer States & Run Logs (+10.0 GB)**:  
   A full TC-GraphFormer model checkpoint including AdamW optimizer states is ~17.5 MB. A 50-epoch multi-task training run with periodic checkpoints, best model saves, fine-tuning checkpoints, and TensorBoard telemetry consumes ~10 GB.
3. **Container Storage Rounding Adjustment (+0.9 GB)**:  
   Based on the pilot's empirical average of 580.8 KB per processed overpass (64,800 overpasses $\times 580.8\text{ KB} = 38.6\text{ GB}$ vs. nominal $37.7\text{ GB}$).

$$\text{Total Required Storage} = 500.0 + 38.6 + 40.0 + 10.0 + 20.0 = \mathbf{608.6\text{ GB}}$$

---

## 6. Infrastructure Readiness Conclusion & Procurement Mandate

```
+----------------------------------------------------------------------------------------------------+
| INFRASTRUCTURE VERDICT: NOT READY FOR 500 GB DOWNLOAD                                              |
+----------------------------------------------------------------------------------------------------+
| 1. Workstation Internal Storage Deficit: 70.19 GB available vs 608.6 GB required (-538.4 GB deficit)|
| 2. Remote GPU Status: Unverified; placeholder JSON detected; AnyDesk cannot execute CUDA jobs.     |
+----------------------------------------------------------------------------------------------------+
| MANDATORY PREREQUISITE:                                                                            |
| Procure and mount an External High-Performance NVMe SSD (1 TB or 2 TB capacity, USB 3.2 Gen 2).    |
| Deploy real SSH server and CUDA PyTorch environment natively on Machine B.                         |
+----------------------------------------------------------------------------------------------------+
```

