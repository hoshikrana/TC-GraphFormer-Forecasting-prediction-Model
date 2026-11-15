"""
Verification utility for transferred dataset shards on Machine B.
Validates existence, file sizes, and cryptographic SHA-256 digests.
Detects unexpected extra files within configured transfer roots.
Outputs reports/transfer_validation_report.json.
"""

import os
import sys
import json
import hashlib
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Union


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
    report_output: str = "reports/transfer_validation_report.json",
    transfer_roots: Optional[Union[str, List[str]]] = None,
) -> bool:
    """
    Verify all files in manifest against local files and scan transfer roots for extra files.
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
    missing_files = 0
    size_mismatches = 0
    hash_mismatches = 0
    extra_files = 0

    verification_results: Dict[str, Any] = {}
    matched_actual_paths = set()

    is_base_workspace = (os.path.abspath(base_dir) == os.path.abspath("."))
    processed_dir_prefix = manifest_data.get("processed_dir", "").replace("\\", "/").strip("/")

    for idx, (rel_path, info) in enumerate(files.items(), 1):
        expected_size = info["size_bytes"]
        expected_hash = info["sha256"]

        candidate_paths = [
            os.path.normpath(os.path.join(base_dir, rel_path)),
        ]
        if os.path.isabs(rel_path):
            candidate_paths.append(os.path.normpath(rel_path))

        # If processed_dir_prefix is specified in manifest, check under base_dir/processed_dir/rel_path
        # or strip prefix if base_dir already points to processed_dir
        norm_rel = rel_path.replace("\\", "/").strip("/")
        if processed_dir_prefix:
            candidate_paths.append(os.path.normpath(os.path.join(base_dir, processed_dir_prefix, norm_rel)))
            if norm_rel.startswith(processed_dir_prefix + "/"):
                sub_rel = norm_rel[len(processed_dir_prefix) + 1:]
                candidate_paths.append(os.path.normpath(os.path.join(base_dir, sub_rel)))

        # If base_dir is a dedicated transfer folder (not workspace root .), allow matching filename inside base_dir
        if not is_base_workspace:
            fname = info.get("filename") or os.path.basename(rel_path)
            candidate_paths.append(os.path.normpath(os.path.join(base_dir, fname)))

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
        matched_actual_paths.add(os.path.abspath(actual_path))
        verification_results[rel_path] = {
            "status": "VALID",
            "size_bytes": actual_size,
            "sha256": actual_hash,
            "actual_path": actual_path,
        }

        if idx % 20 == 0 or idx == total_files:
            print(f"[VERIFY] Verified {idx}/{total_files}: {os.path.basename(actual_path)} [PASS]")

    # Track all expected manifest files found on disk
    manifest_actual_paths = {
        os.path.abspath(res["actual_path"])
        for res in verification_results.values()
        if res.get("actual_path")
    }

    # Identify transfer roots to scan for unexpected extra files
    roots_to_scan: List[str] = []
    if transfer_roots:
        if isinstance(transfer_roots, str):
            transfer_roots = [transfer_roots]
        for tr in transfer_roots:
            p = os.path.normpath(tr)
            if os.path.isdir(p) and os.path.abspath(p) != os.path.abspath("."):
                roots_to_scan.append(p)
    elif processed_dir_prefix:
        # Check manifest processed_dir
        cand = os.path.normpath(os.path.join(base_dir, processed_dir_prefix))
        if os.path.isdir(cand) and os.path.abspath(cand) != os.path.abspath("."):
            roots_to_scan.append(cand)
    elif not is_base_workspace and os.path.isdir(base_dir):
        # If base_dir is non-default dedicated folder and no processed_dir, scan base_dir
        roots_to_scan.append(os.path.normpath(base_dir))

    # Deduplicate scan roots
    unique_roots = []
    seen_roots = set()
    for r in roots_to_scan:
        r_abs = os.path.abspath(r)
        if r_abs not in seen_roots:
            seen_roots.add(r_abs)
            unique_roots.append(r)

    # Scan for extra files in transfer roots
    extra_results: Dict[str, Any] = {}
    for r in unique_roots:
        for dirpath, _, filenames in os.walk(r):
            for fn in filenames:
                fpath = os.path.normpath(os.path.join(dirpath, fn))
                fpath_abs = os.path.abspath(fpath)
                if fpath_abs not in manifest_actual_paths:
                    extra_files += 1
                    try:
                        rel_extra = os.path.relpath(fpath, base_dir).replace("\\", "/")
                    except ValueError:
                        # Handle cross-drive paths on Windows
                        rel_extra = fpath.replace("\\", "/")
                    extra_results[rel_extra] = {
                        "status": "EXTRA",
                        "actual_path": fpath,
                        "size_bytes": os.path.getsize(fpath),
                        "error": "Unexpected file in transfer root not present in manifest",
                    }
                    print(f"[VERIFY] Detected EXTRA file: {rel_extra}")

    all_passed = (passed_files == total_files) and (extra_files == 0)
    failed_files = missing_files + size_mismatches + hash_mismatches + extra_files

    summary = {
        "verification_status": "SUCCESS" if all_passed else "FAILURE",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": manifest_path,
        "transfer_roots": unique_roots,
        "total_files": total_files,
        "passed_files": passed_files,
        "failed_files": failed_files,
        "missing_files": missing_files,
        "size_mismatches": size_mismatches,
        "hash_mismatches": hash_mismatches,
        "extra_files": extra_files,
        "pass_rate_pct": round((passed_files / max(1, total_files + extra_files)) * 100, 2) if total_files > 0 else 0,
        "results": {**verification_results, **extra_results},
    }

    report_dir = os.path.dirname(report_output)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    with open(report_output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"VERIFICATION {'PASSED' if all_passed else 'FAILED'}")
    print(f"Passed: {passed_files}/{total_files} ({summary['pass_rate_pct']}%)")
    print(f"Failed: {failed_files} (Missing: {missing_files}, Size: {size_mismatches}, Hash: {hash_mismatches}, Extra: {extra_files})")
    print(f"Report saved: {report_output}")
    print("=" * 60)

    return all_passed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify dataset transfer with SHA-256 digests.")
    parser.add_argument("manifest_pos", nargs="?", default=None, help="Optional positional path to staging manifest JSON")
    parser.add_argument("--manifest", default="staging/staging_manifest.json", help="Path to staging manifest JSON")
    parser.add_argument("--base-dir", default=".", help="Base directory where staged data is located")
    parser.add_argument("--transfer-root", action="append", dest="transfer_roots", default=None, help="Root directory to scan for unexpected extra files")
    parser.add_argument("--output", default="reports/transfer_validation_report.json", help="Path to output validation JSON")
    return parser


def parse_args(args=None):
    return build_parser().parse_args(args)


def main():
    args = parse_args()

    target_manifest = args.manifest_pos if args.manifest_pos is not None else args.manifest
    success = verify_staging(
        manifest_path=target_manifest,
        base_dir=args.base_dir,
        report_output=args.output,
        transfer_roots=args.transfer_roots,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
