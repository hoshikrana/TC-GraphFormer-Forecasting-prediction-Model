"""
01_build_catalogue.py - Phase 1: master catalogue of the WHOLE TC PRIMED archive.

Step A (fast):  walk the folder, parse every file NAME  ->  file_index.csv
Step B (slow, resumable): open each overpass file once, read ONLY small metadata / label
                variables (never the big image arrays)  ->  overpass_meta.csv

Usage:
    pip install netCDF4 numpy pandas
    python 01_build_catalogue.py <root> --out <catalogue_dir>
    python 01_build_catalogue.py <root> --out <catalogue_dir> --index-only
    python 01_build_catalogue.py <root> --out <catalogue_dir> --basins IO --limit 300   # quick trial
Re-running resumes: files already in overpass_meta.csv are skipped.

Tip: on Colab, files read from a mounted Drive are slow. If it crawls, run this on the
university machine, or copy a chunk of the archive to local disk first.
"""
import argparse
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from netCDF4 import Dataset

NAME_RE = re.compile(r"^TCPRIMED_(?P<ver>[^_]+)_(?P<storm>[A-Z]{2}\d{6})_(?P<rest>.+)\.nc$")
PERIODS = [-24, -18, -12, -6, 0, 6, 12, 18, 24]          # intensity_change_periods (hours)
DI_COLS = [f"dI_{p:+d}h" for p in PERIODS]

META_COLUMNS = (
    ["rel", "ok", "error", "groups", "has_overpass_meta", "basin", "season", "cyclone_number",
     "instrument_name", "platform_name", "coverage_fraction", "obs_time_utc",
     "storm_lat", "storm_lon", "wind_kt", "pressure_hpa", "stage", "speed_ms", "heading_deg",
     "dist_to_land_km"] + DI_COLS +
    ["has_ir_group", "ir_flag", "ir_shape", "gprof_flag", "radar_flag",
     "pmw_swaths", "pmw_channels", "pmw_n_channels", "pmw_shapes"]
)


# ----------------------------------------------------------------------------- Step A: names
def parse_name(fname):
    m = NAME_RE.match(fname)
    if not m:
        return {"storm": None, "kind": "unknown"}
    storm, parts = m["storm"], m["rest"].split("_")
    # ASSUMPTION: environmental files have a name part starting with "env". Verified later by groups.
    kind = "environmental" if parts[0].lower().startswith("env") else "overpass"
    rec = {"version": m["ver"], "storm": storm, "basin": storm[:2],
           "cyclone_number": int(storm[2:4]), "season": int(storm[4:]), "kind": kind}
    if kind == "overpass":
        rec["instrument"] = parts[0]
        rec["platform"] = parts[1] if len(parts) > 1 else None
        rec["granule"] = parts[2] if len(parts) > 2 else None
        if re.fullmatch(r"\d{14}", parts[-1]):
            rec["file_time_utc"] = pd.to_datetime(parts[-1], format="%Y%m%d%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
    return rec


def build_index(root, out_csv):
    rows, t0 = [], time.time()
    for dirpath, _, files in os.walk(root):
        for f in files:
            if not f.endswith(".nc"):
                continue
            full = os.path.join(dirpath, f)
            rec = parse_name(f)
            rec["rel"] = os.path.relpath(full, root)
            try:
                rec["size_mb"] = round(os.path.getsize(full) / 1e6, 3)
            except OSError:
                rec["size_mb"] = None
            rows.append(rec)
        if len(rows) and len(rows) % 20000 < len(files):
            print(f"  indexed {len(rows):,} files ({time.time() - t0:.0f}s)")
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"\nfile_index.csv: {len(df):,} files, {df['size_mb'].sum() / 1e3:,.1f} GB, "
          f"{df['storm'].nunique():,} storms")
    print(df["kind"].value_counts().to_string())
    return df


# ----------------------------------------------------------------------------- Step B: metadata
def clean(v):
    """netCDF value -> plain python scalar, or None if masked / missing sentinel (<= -900)."""
    if v is None or np.ma.is_masked(v):
        return None
    v = np.ma.getdata(v)
    if isinstance(v, np.ndarray):
        if v.size == 0:
            return None
        v = v.ravel()[0]
    if isinstance(v, bytes):
        return v.decode(errors="ignore")
    if isinstance(v, (np.floating, float)):
        return None if (not np.isfinite(v) or v <= -900) else float(v)
    if isinstance(v, (np.integer, int)):
        return None if v <= -900 else int(v)
    return str(v)


def getv(group, name):
    if group is None or name not in group.variables:
        return None
    return clean(group.variables[name][:])


def getarr(group, name):
    if group is None or name not in group.variables:
        return None
    a = np.ma.filled(np.ma.asarray(group.variables[name][:], dtype="float64"), np.nan).ravel()
    a[a <= -900] = np.nan
    return a


def harvest_one(task):
    root, rel = task
    row = {"rel": rel, "ok": True, "error": ""}
    try:
        with Dataset(os.path.join(root, rel)) as nc:
            row["groups"] = ";".join(nc.groups)
            m, s = nc.groups.get("overpass_metadata"), nc.groups.get("overpass_storm_metadata")
            row["has_overpass_meta"] = m is not None
            for k in ("basin", "season", "cyclone_number", "instrument_name",
                      "platform_name", "coverage_fraction"):
                row[k] = getv(m, k)
            t = getv(m, "time")
            row["obs_time_utc"] = (pd.to_datetime(t, unit="s", utc=True).strftime("%Y-%m-%d %H:%M:%S")
                                   if t is not None else None)
            for k, out in [("storm_latitude", "storm_lat"), ("storm_longitude", "storm_lon"),
                           ("intensity", "wind_kt"), ("central_min_pressure", "pressure_hpa"),
                           ("development_level", "stage"), ("storm_speed", "speed_ms"),
                           ("storm_heading", "heading_deg"), ("distance_to_land", "dist_to_land_km")]:
                row[out] = getv(s, k)
            per, ic = getarr(s, "intensity_change_periods"), getarr(s, "intensity_change")
            if per is not None and ic is not None and len(per) == len(ic):
                for p, v in zip(per, ic):
                    row[f"dI_{int(p):+d}h"] = None if np.isnan(v) else float(v)

            ir = nc.groups.get("infrared")
            row["has_ir_group"] = ir is not None
            row["ir_flag"] = getv(ir, "infrared_availability_flag")
            row["ir_shape"] = ("x".join(map(str, ir.variables["IRWIN"].shape))
                               if ir is not None and "IRWIN" in ir.variables else None)
            row["gprof_flag"] = getv(nc.groups.get("GPROF"), "availability_flag")
            row["radar_flag"] = getv(nc.groups.get("radar_radiometer"), "availability_flag")

            swaths, chans, shapes = [], [], []
            pmw = nc.groups.get("passive_microwave")
            if pmw is not None:
                for sw, g in pmw.groups.items():
                    swaths.append(sw)
                    chans += [n[3:] for n in g.variables if n.startswith("TB_")]
                    if "latitude" in g.variables:
                        shapes.append(f"{sw}:" + "x".join(map(str, g.variables["latitude"].shape)))
            row.update(pmw_swaths=";".join(swaths), pmw_channels=";".join(chans),
                       pmw_n_channels=len(chans), pmw_shapes=";".join(shapes))
    except Exception as e:                                   # keep going; record the failure
        row["ok"], row["error"] = False, f"{type(e).__name__}: {e}"[:200]
    return row


def harvest(root, index, out_csv, workers, batch, limit, basins):
    todo = index[index["kind"] == "overpass"]
    if basins:
        todo = todo[todo["basin"].isin(basins)]
    done = set(pd.read_csv(out_csv, usecols=["rel"])["rel"]) if os.path.exists(out_csv) else set()
    todo = todo[~todo["rel"].isin(done)]
    if limit:
        todo = todo.head(limit)
    n = len(todo)
    print(f"\nharvesting {n:,} overpass files ({len(done):,} already done) with {workers} workers")
    if n == 0:
        return
    t0, rels = time.time(), todo["rel"].tolist()
    with ProcessPoolExecutor(workers) as pool:
        for i in range(0, n, batch):
            chunk = [(root, r) for r in rels[i:i + batch]]
            rows = list(pool.map(harvest_one, chunk, chunksize=8))
            pd.DataFrame(rows).reindex(columns=META_COLUMNS).to_csv(
                out_csv, mode="a", header=not os.path.exists(out_csv), index=False)
            finished = min(i + batch, n)
            rate = finished / (time.time() - t0)
            print(f"  {finished:,}/{n:,}  {rate:.1f} files/s  ETA {(n - finished) / rate / 60:.0f} min")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default="catalogue")
    ap.add_argument("--index-only", action="store_true")
    ap.add_argument("--reindex", action="store_true", help="rebuild file_index.csv (use after adding files)")
    ap.add_argument("--limit", type=int, default=0, help="max files to harvest this run")
    ap.add_argument("--basins", nargs="*", help="e.g. IO to harvest only the North Indian Ocean first")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 2))
    ap.add_argument("--batch", type=int, default=400)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    idx_csv = os.path.join(a.out, "file_index.csv")
    index = (pd.read_csv(idx_csv) if os.path.exists(idx_csv) and not a.reindex
             else build_index(a.root, idx_csv))
    if not a.index_only:
        harvest(a.root, index, os.path.join(a.out, "overpass_meta.csv"),
                a.workers, a.batch, a.limit, a.basins)
    print("\ndone ->", os.path.abspath(a.out))


if __name__ == "__main__":
    main()
