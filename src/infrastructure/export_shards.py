"""
Export Staging Shards for Remote GPU Compute Node Transfer.
Computes cryptographic SHA-256 checksums and creates staging_manifest.json
for any configurable processed directory and manifest targets.
"""

import os
import sys
import json
import hashlib
import shutil
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional


def compute_sha256(filepath: str, block_size: int = 65536) -> str:
    """Compute hex SHA-256 digest of a file in streaming chunks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(block_size):
            h.update(chunk)
    return h.hexdigest()


def create_staging_manifest(
    processed_dir: str = "data/processed/pilot",
    split_manifest_path: str = "reports/pipeline/storm_split_manifest.csv",
    sample_index_path: str = "reports/pipeline/sample_index.csv",
    output_dir: str = "staging",
    manifest_name: str = "staging_manifest.json",
    dataset_name: Optional[str] = None,
) -> str:
    """
    Index all processed storm NetCDF files, compute SHA-256, and produce staging manifest.
    Supports arbitrary processed and metadata roots without hardcoding pilot directories.
    """
    os.makedirs(output_dir, exist_ok=True)
    processed_path = Path(processed_dir)
    proc_posix = processed_path.as_posix().rstrip("/")

    # Load storm split mapping if available
    storm_to_split = {}
    if os.path.exists(split_manifest_path):
        import pandas as pd
        df_split = pd.read_csv(split_manifest_path)
        for _, row in df_split.iterrows():
            storm_to_split[str(row["storm_id"])] = row.get("split", "unknown")

    # Load sample count per storm if available
    storm_sample_counts = {}
    if os.path.exists(sample_index_path):
        import pandas as pd
        df_samples = pd.read_csv(sample_index_path)
        counts = df_samples.groupby("storm_id").size().to_dict()
        for k, v in counts.items():
            storm_sample_counts[str(k)] = int(v)

    files_manifest: Dict[str, Any] = {}
    total_bytes = 0

    nc_files = sorted(list(processed_path.rglob("*.nc")))
    print(f"[EXPORT] Found {len(nc_files)} processed NetCDF storm containers in {processed_dir}")

    for idx, nc_file in enumerate(nc_files, 1):
        rel_path = nc_file.relative_to(processed_path).as_posix()
        full_rel_path = f"{proc_posix}/{rel_path}"
        size_bytes = nc_file.stat().st_size
        total_bytes += size_bytes
        digest = compute_sha256(str(nc_file))
        storm_id = nc_file.stem
        split_name = storm_to_split.get(storm_id, "unknown")
        sample_count = storm_sample_counts.get(storm_id, 0)

        files_manifest[full_rel_path] = {
            "filename": nc_file.name,
            "relative_path": full_rel_path,
            "storm_id": storm_id,
            "split": split_name,
            "num_samples": sample_count,
            "size_bytes": size_bytes,
            "sha256": digest,
        }

        if idx % 20 == 0 or idx == len(nc_files):
            print(f"[EXPORT] Indexed {idx}/{len(nc_files)} files: {nc_file.name} (SHA-256: {digest[:12]}...)")

    # Also include metadata / index files in staging manifest
    aux_files = [split_manifest_path, sample_index_path]
    for aux in aux_files:
        if os.path.exists(aux):
            aux_p = Path(aux)
            aux_size = aux_p.stat().st_size
            total_bytes += aux_size
            digest = compute_sha256(str(aux_p))
            files_manifest[aux_p.as_posix()] = {
                "filename": aux_p.name,
                "relative_path": aux_p.as_posix(),
                "storm_id": "METADATA",
                "split": "ALL",
                "num_samples": 0,
                "size_bytes": aux_size,
                "sha256": digest,
            }

    d_name = dataset_name or f"NOAA TC PRIMED v01r01-final ({processed_path.name} Staging)"

    staging_data = {
        "manifest_version": "1.1.0",
        "dataset_name": d_name,
        "source_machine": "Machine A (Workstation)",
        "target_machine": "Machine B (Remote GPU Node)",
        "processed_dir": proc_posix,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_files": len(files_manifest),
            "total_storm_containers": len(nc_files),
            "total_bytes": total_bytes,
            "total_megabytes": round(total_bytes / (1024**2), 2),
            "total_gigabytes": round(total_bytes / (1024**3), 3),
            "split_breakdown": {
                "train_storms": sum(1 for f in files_manifest.values() if f.get("split") == "train"),
                "validation_storms": sum(1 for f in files_manifest.values() if f.get("split") == "validation"),
                "test_storms": sum(1 for f in files_manifest.values() if f.get("split") == "test"),
            }
        },
        "transfer_instructions": [
            "1. AnyDesk note: AnyDesk is a desktop streaming client; it does not connect Python directly to remote GPUs.",
            f"2. Transfer this {output_dir}/ folder and {proc_posix}/ to Machine B via AnyDesk File Transfer, SFTP/SCP, or external SSD.",
            f"3. On Machine B, run: python src/infrastructure/verify_transfer.py --manifest {os.path.join(output_dir, manifest_name)}",
            "4. verify_transfer.py recalculates SHA-256 checksums to guarantee 100% bitwise data integrity before initiating GPU training.",
        ],
        "files": files_manifest,
    }

    manifest_output_path = os.path.join(output_dir, manifest_name)
    with open(manifest_output_path, "w", encoding="utf-8") as f:
        json.dump(staging_data, f, indent=2)

    print(f"\n[OK] Staging manifest written: {manifest_output_path}")
    print(f"     Indexed: {len(files_manifest)} files ({round(total_bytes / (1024**3), 3)} GB)")
    return manifest_output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export staging manifest with SHA-256 hashes.")
    parser.add_argument("--processed-dir", default="data/processed/pilot", help="Processed containers directory")
    parser.add_argument("--split-manifest", default="reports/pipeline/storm_split_manifest.csv", help="Storm split CSV")
    parser.add_argument("--sample-index", default="reports/pipeline/sample_index.csv", help="Sample index CSV")
    parser.add_argument("--output-dir", default="staging", help="Output staging directory")
    parser.add_argument("--manifest-name", default="staging_manifest.json", help="Manifest JSON filename")
    parser.add_argument("--dataset-name", default=None, help="Descriptive name for dataset")
    args = parser.parse_args()

    create_staging_manifest(
        processed_dir=args.processed_dir,
        split_manifest_path=args.split_manifest,
        sample_index_path=args.sample_index,
        output_dir=args.output_dir,
        manifest_name=args.manifest_name,
        dataset_name=args.dataset_name,
    )
