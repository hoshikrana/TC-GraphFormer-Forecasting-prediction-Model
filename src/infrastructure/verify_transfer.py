"""
Verification utility for transferred dataset shards on Machine B.
Validates existence, file sizes, and cryptographic SHA-256 digests.
Outputs reports/transfer_validation_report.json.
"""

import os
import sys
import json
import hashlib
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any


def compute_sha256(filepath: str, block_size: int = 65536) -> str:
    """Compute hex SHA-256 digest of a file in streaming chunks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(block_size):
            h.update(chunk)
    return h.hexdigest()


def verify_staging(
    manifest_path: str = "staging/staging_manifest.json",
    base_dir: str = ".",
    report_output: str = "reports/transfer_validation_report.json"
) -> bool:
    """
    Verify all files in manifest against local files.
    """
    if not os.path.exists(manifest_path):
        print(f"[ERROR] Staging manifest not found: {manifest_path}")
        return False

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    files = manifest_data.get("files", {})
    print(f"[VERIFY] Loaded manifest with {len(files)} entries from {manifest_path}")

    total_files = len(files)
    passed_files = 0
    failed_files = 0
    missing_files = 0
    size_mismatches = 0
    hash_mismatches = 0

    verification_results = {}

    for idx, (rel_path, info) in enumerate(files.items(), 1):
        # Resolve target path relative to base_dir
        expected_size = info["size_bytes"]
        expected_hash = info["sha256"]
        
        # Check relative path directly or under base_dir
        candidate_paths = [
            os.path.join(base_dir, rel_path),
            rel_path,
            os.path.join(base_dir, info.get("filename", "")),
            os.path.join(base_dir, "data", rel_path),
        ]
        
        actual_path = None
        for cand in candidate_paths:
            if os.path.exists(cand):
                actual_path = cand
                break

        if actual_path is None:
            missing_files += 1
            verification_results[rel_path] = {
                "status": "MISSING",
                "expected_size": expected_size,
                "expected_sha256": expected_hash,
                "actual_path": None,
                "error": "File does not exist at expected destination",
            }
            continue

        actual_size = os.path.getsize(actual_path)
        if actual_size != expected_size:
            size_mismatches += 1
            verification_results[rel_path] = {
                "status": "SIZE_MISMATCH",
                "expected_size": expected_size,
                "actual_size": actual_size,
                "expected_sha256": expected_hash,
                "actual_path": actual_path,
                "error": f"Size mismatch: expected {expected_size}, got {actual_size}",
            }
            continue

        actual_hash = compute_sha256(actual_path)
        if actual_hash.lower() != expected_hash.lower():
            hash_mismatches += 1
            verification_results[rel_path] = {
                "status": "HASH_MISMATCH",
                "expected_size": expected_size,
                "expected_sha256": expected_hash,
                "actual_sha256": actual_hash,
                "actual_path": actual_path,
                "error": "SHA-256 digest mismatch",
            }
            continue

        passed_files += 1
        verification_results[rel_path] = {
            "status": "VALID",
            "size_bytes": actual_size,
            "sha256": actual_hash,
            "actual_path": actual_path,
        }

        if idx % 20 == 0 or idx == total_files:
            print(f"[VERIFY] Verified {idx}/{total_files}: {os.path.basename(actual_path)} [PASS]")

    all_passed = (passed_files == total_files)
    failed_files = missing_files + size_mismatches + hash_mismatches

    summary = {
        "verification_status": "SUCCESS" if all_passed else "FAILURE",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": manifest_path,
        "total_files": total_files,
        "passed_files": passed_files,
        "failed_files": failed_files,
        "missing_files": missing_files,
        "size_mismatches": size_mismatches,
        "hash_mismatches": hash_mismatches,
        "pass_rate_pct": round((passed_files / total_files) * 100, 2) if total_files > 0 else 0,
        "results": verification_results,
    }

    os.makedirs(os.path.dirname(report_output), exist_ok=True)
    with open(report_output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"VERIFICATION {'PASSED' if all_passed else 'FAILED'}")
    print(f"Passed: {passed_files}/{total_files} ({summary['pass_rate_pct']}%)")
    print(f"Failed: {failed_files} (Missing: {missing_files}, Size: {size_mismatches}, Hash: {hash_mismatches})")
    print(f"Report saved: {report_output}")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify dataset transfer with SHA-256 digests.")
    parser.add_argument("--manifest", default="staging/staging_manifest.json", help="Path to staging manifest JSON")
    parser.add_argument("--base-dir", default=".", help="Base directory where staged data is located")
    parser.add_argument("--output", default="reports/transfer_validation_report.json", help="Path to output validation JSON")
    args = parser.parse_args()

    success = verify_staging(args.manifest, args.base_dir, args.output)
    sys.exit(0 if success else 1)

