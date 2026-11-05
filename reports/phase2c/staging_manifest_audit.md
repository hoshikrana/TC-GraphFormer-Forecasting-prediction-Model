# TC PRIMED Phase 2C — Two-Machine Staging Manifest and Transport Protocol Audit

## 1. Executive Summary & Audit Objective

The Phase 2B specification designed a two-machine research framework:
- **Machine A (Workstation/Friend's Laptop)**: Responsible for AWS S3 inventory, raw NetCDF downloads, preprocessing, quality control, metadata splitting, and staging.
- **Machine B (Remote Compute / GPU)**: Responsible for self-supervised pretraining, multi-task training, and hyperparameter search.

This audit evaluates:
1. Cryptographic integrity of the generated staging manifest (`staging/staging_manifest.json`).
2. Bitwise preservation of staged HDF5 containers and metadata CSVs.
3. Transport protocol reality: dissecting the AnyDesk GUI assumption versus production distributed data staging.
4. Completeness of staging metadata (sample counts and split assignments).

---

## 2. Cryptographic SHA-256 Hash Audit

The staging manifest `staging/staging_manifest.json` was inspected and verified against the actual files on disk. Every file was loaded and re-hashed using SHA-256:

```
Total Staged Files Tracked:               82
  - Processed Storm HDF5 Containers:      80
  - Metadata / Sample Index CSVs:          2 (sample_index.csv, storm_summary.csv)
Total Staged Volume:                      1,510,555,194 bytes (1.41 GiB / 1.51 GB)

Verification Audit Outcome:
  - Exact Cryptographic Matches:          82 / 82 (100.00%)
  - Checksum Mismatches:                   0 (0.00%)
  - Missing Staged Files:                  0 (0.00%)
```

**Conclusion**: The staging pipeline generated valid, bit-exact cryptographic manifests with zero data corruption.

---

## 3. Two-Machine Transport Protocol Critique: AnyDesk vs. Production Staging

In the Phase 2B implementation notes, AnyDesk was cited as the connection mechanism between Machine A and Machine B. We conducted a critical architectural review of this assumption:

### The AnyDesk Fallacy
- **What AnyDesk Is**: A proprietary Remote Desktop Protocol (RDP) application optimized for streaming compressed H.264 video frames of a graphical user interface (GUI) and forwarding mouse/keyboard events.
- **What AnyDesk Is NOT**: It is not a distributed computing orchestrator, network filesystem, or high-throughput data transport pipeline.
- **Technical Failure Modes**:
  1. AnyDesk does not expose a remote CUDA device to local Python processes (running `python train.py` on Machine A executes strictly on Machine A's local CPU, regardless of AnyDesk connections).
  2. AnyDesk file transfer is manual, single-threaded, lacks automated checksum validation, and throttles over WAN connections.
  3. Attempting to transfer 40–500 GB over AnyDesk file transfer is fragile, unresumable, and unsuited for scientific reproducibility.

### Production Two-Machine Protocol for Phase 3
To reliably stage the forthcoming 2000–2025 dataset between Machine A and Machine B, the project must implement a standard scientific distributed protocol:

```
[Machine A: Workstation]                             [Machine B: Dedicated GPU Server]
  1. Download Raw NetCDF                                1. SSH Server Running
  2. Run Preprocessing Pipeline                         2. CUDA 12.x / PyTorch GPU
  3. Generate HDF5 + Shards                             3. Local Fast NVMe Cache
  4. Generate Staging Manifest (SHA-256)
             |                                                    ^
             +==================== rsync / rclone ================+
             |             (Automated SSH Tunnel / LAN)           |
             v                                                    |
  5. Machine B executes: bash scripts/verify_and_train.sh --------+
```

1. **Transport Mechanism**: `rsync -avzP --checksum` over SSH (or `rclone` for cloud object storage / S3 / MinIO).
2. **Automated Verification**: Machine B runs a pre-flight verification script that computes SHA-256 hashes against `staging_manifest.json` before triggering training.
3. **Execution Separation**: Code is tracked via Git; Machine B clones the repository and executes Python natively within its own CUDA virtual environment.

---

## 4. Metadata Completeness Audit

While the file sizes and SHA-256 hashes in `staging/staging_manifest.json` are exact, our inspection revealed missing metadata fields inside the manifest:

```json
{
  "filename": "CP011987_processed.nc",
  "relative_path": "data/processed/pilot/1987/CP/CP011987/CP011987_processed.nc",
  "storm_id": "CP011987_processed",
  "split": "unknown",
  "num_samples": 0,
  "size_bytes": 1714151,
  "sha256": "9dfd71c7cfd4dba12e685f5b9cd81d50efa59b0b67b19e93c42ea1fc6550676a"
}
```

- **Issue**: `split` is populated as `"unknown"` and `num_samples` is `0` for all 80 storm entries.
- **Summary Section**: Reports `'split_breakdown': {'train_storms': 0, 'validation_storms': 0, 'test_storms': 0}`.
- **Root Cause**: In `src/infrastructure/staging.py`, the staging script scanned the directory files and computed checksums but did not join with `reports/pipeline/sample_index.csv` to resolve partition labels (`TRAIN`, `VAL`, `TEST`) or query the observation count per container.
- **Impact**: Non-blocking for data transfer, but prevents Machine B from creating split-aware staging caches without reading `sample_index.csv`.

**Action Item for Phase 3**: Update `src/infrastructure/staging.py` to cross-reference `reports/pipeline/sample_index.csv` and populate true partition labels and observation counts.

---

## 5. Audit Conclusion
The staged data containers are 100% intact and cryptographically verified. The transport protocol must transition from AnyDesk to automated `rsync`/SSH, and staging metadata must be enriched with split assignments.

