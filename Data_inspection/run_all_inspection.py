"""
run_all_inspection.py - Master inspection script for TC PRIMED sample data.

Runs on all .nc files in Data_inspection/, produces:
  1. Console output: complete NetCDF structure, metadata, labels
  2. Visualization PNGs per file (IR, microwave, GPROF, radar panels)
  3. A summary CSV with QC flags

Usage:
    python run_all_inspection.py
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from netCDF4 import Dataset
from scipy.spatial import cKDTree

# â”€â”€â”€ paths â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(DATA_DIR, "inspection_output")
os.makedirs(OUT_DIR, exist_ok=True)

CATEGORICAL = {"mainprecipitationType", "surfaceType", "surface_type", "surfaceTypeIndex"}
SKIP_2D = {"latitude", "longitude", "x", "y", "qualityFlag"}


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# PART 1: Deep NetCDF Structure Inspector
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def walk_groups(group, prefix=""):
    for name, subgroup in group.groups.items():
        path = f"{prefix}/{name}" if prefix else name
        yield path, subgroup
        yield from walk_groups(subgroup, path)


def inspect_netcdf(file_path):
    """Print complete structure of a TC PRIMED NetCDF file."""
    print("\n" + "=" * 100)
    print(f"FILE: {os.path.basename(file_path)}")
    print(f"SIZE: {os.path.getsize(file_path) / 1e6:.2f} MB")
    print("=" * 100)

    with Dataset(file_path, "r") as nc:
        # Root attributes
        print("\n--- ROOT ATTRIBUTES ---")
        for attr in nc.ncattrs():
            val = getattr(nc, attr)
            # Truncate long strings for readability
            s = str(val)
            if len(s) > 200:
                s = s[:200] + "..."
            print(f"  {attr}: {s}")

        print(f"\nâ”€â”€â”€ TOP-LEVEL GROUPS: {list(nc.groups.keys())} â”€â”€â”€")

        # Walk every group recursively
        for group_path, group in walk_groups(nc):
            print(f"\n{'â”€' * 80}")
            print(f"GROUP: {group_path}")
            if group.dimensions:
                print("  Dimensions:")
                for dname, dim in group.dimensions.items():
                    print(f"    {dname} = {len(dim)}")
            if group.variables:
                print("  Variables:")
                for vname, var in group.variables.items():
                    attrs = {}
                    for a in ["long_name", "units", "frequency", "polarization",
                              "valid_range", "_FillValue", "missing_value", "scale_factor"]:
                        if a in var.ncattrs():
                            attrs[a] = getattr(var, a)
                    print(f"    {vname}: shape={var.shape}  dtype={var.dtype}  {attrs}")

    print("\n" + "=" * 100 + "\n")


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# PART 2: Metadata & Label Extraction
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
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


def extract_metadata(file_path):
    """Extract key metadata and labels from one overpass file."""
    row = {"file": os.path.basename(file_path)}
    try:
        with Dataset(file_path) as nc:
            m = nc.groups.get("overpass_metadata")
            s = nc.groups.get("overpass_storm_metadata")
            ir = nc.groups.get("infrared")

            row["groups"] = list(nc.groups.keys())
            row["basin"] = getv(m, "basin")
            row["season"] = getv(m, "season")
            row["cyclone_number"] = getv(m, "cyclone_number")
            row["instrument"] = getv(m, "instrument_name")
            row["platform"] = getv(m, "platform_name")
            row["coverage_pct"] = getv(m, "coverage_fraction")

            t = getv(m, "time")
            if t is not None:
                row["obs_time_utc"] = pd.to_datetime(t, unit="s", utc=True).strftime("%Y-%m-%d %H:%M:%S")

            row["storm_lat"] = getv(s, "storm_latitude")
            row["storm_lon"] = getv(s, "storm_longitude")
            row["wind_kt"] = getv(s, "intensity")
            row["pressure_hpa"] = getv(s, "central_min_pressure")
            row["stage"] = getv(s, "development_level")
            row["dist_to_land_km"] = getv(s, "distance_to_land")
            row["storm_speed_ms"] = getv(s, "storm_speed")
            row["storm_heading_deg"] = getv(s, "storm_heading")

            # Intensity change
            if s is not None and "intensity_change" in s.variables:
                ic = np.ma.filled(np.ma.asarray(s.variables["intensity_change"][:], dtype="float64"), np.nan).ravel()
                ic[ic <= -900] = np.nan
                per = np.ma.filled(np.ma.asarray(s.variables["intensity_change_periods"][:], dtype="float64"), np.nan).ravel()
                per[per <= -900] = np.nan
                for p, v in zip(per, ic):
                    if np.isfinite(p):
                        row[f"dI_{int(p):+d}h"] = None if np.isnan(v) else float(v)

            # IR info
            row["has_ir"] = ir is not None and "IRWIN" in ir.variables
            if row["has_ir"]:
                row["ir_shape"] = str(ir.variables["IRWIN"].shape)
                row["ir_flag"] = getv(ir, "infrared_availability_flag")

            # PMW info
            pmw = nc.groups.get("passive_microwave")
            if pmw:
                swaths = list(pmw.groups.keys())
                row["pmw_swaths"] = swaths
                channels = []
                for sw in swaths:
                    g = pmw.groups[sw]
                    channels += [n for n in g.variables if n.startswith("TB_")]
                row["pmw_channels"] = channels
                row["pmw_n_channels"] = len(channels)

            # GPROF
            gp = nc.groups.get("GPROF")
            row["gprof_flag"] = getv(gp, "availability_flag") if gp else None

            # Radar
            rr = nc.groups.get("radar_radiometer")
            row["radar_flag"] = getv(rr, "availability_flag") if rr else None

    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"

    return row


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# PART 3: Visualization (storm-centred panels)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def read_var(group, name):
    arr = np.ma.masked_invalid(np.ma.asarray(group.variables[name][:], dtype="float32"))
    return np.ma.masked_less_equal(arr, -900.0)


def scalar(group, name, default=None):
    if group is None or name not in group.variables:
        return default
    v = group.variables[name][:]
    if np.ma.is_masked(v) or v.size == 0:
        return default
    v = np.ma.getdata(v).ravel()[0]
    return v.item() if hasattr(v, "item") else v


def regrid(x, y, data, half_width_km=750.0, res_km=5.0):
    n = int(round(2 * half_width_km / res_km))
    centres = (np.arange(n) + 0.5) * res_km - half_width_km
    gx, gy = np.meshgrid(centres, centres)
    out = np.full((n, n), np.nan, dtype="float32")

    ok = ~(np.ma.getmaskarray(data) | np.ma.getmaskarray(x) | np.ma.getmaskarray(y))
    if ok.sum() < 4:
        return out, [-half_width_km, half_width_km] * 2
    sx = np.ma.getdata(x)[ok].astype("float64")
    sy = np.ma.getdata(y)[ok].astype("float64")
    sv = np.ma.getdata(data)[ok].astype("float32")

    tree = cKDTree(np.c_[sx, sy])
    d_nn, _ = tree.query(np.c_[sx, sy], k=2)
    spacing = float(np.median(d_nn[:, 1]))
    dist, idx = tree.query(np.c_[gx.ravel(), gy.ravel()],
                           distance_upper_bound=max(1.5 * spacing, res_km))
    hit = np.isfinite(dist)
    flat = out.ravel()
    flat[hit] = sv[idx[hit]]
    return flat.reshape(n, n), [-half_width_km, half_width_km, -half_width_km, half_width_km]


def disc_coverage(grid, half_width_km, res_km, radius_km):
    n = grid.shape[0]
    c = (np.arange(n) + 0.5) * res_km - half_width_km
    gx, gy = np.meshgrid(c, c)
    inside = np.hypot(gx, gy) <= radius_km
    return 100.0 * np.isfinite(grid[inside]).mean()


def draw_panel(ax, grid, extent, title, cover, categorical=False, cmap="turbo"):
    ax.set_facecolor("#d9d9d9")
    vals = grid[np.isfinite(grid)]
    if vals.size == 0:
        ax.text(0.5, 0.5, "no valid data", ha="center", va="center", transform=ax.transAxes)
    elif categorical:
        levels = np.unique(vals)
        cm = ListedColormap(plt.get_cmap("tab10")(np.linspace(0, 1, max(len(levels), 2))))
        edges = np.r_[levels - 0.5, levels[-1] + 0.5]
        im = ax.imshow(grid, origin="lower", extent=extent, cmap=cm,
                       norm=BoundaryNorm(edges, cm.N), interpolation="nearest")
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.set_ticks(levels[:12])
    else:
        lo, hi = np.nanpercentile(vals, [1, 99])
        if hi <= lo:
            hi = lo + 1e-6
        im = ax.imshow(grid, origin="lower", extent=extent, cmap=cmap, vmin=lo, vmax=hi,
                       interpolation="nearest")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    for r in (250, 500, 1000):
        ax.add_patch(plt.Circle((0, 0), r, fill=False, ls=":", lw=0.8, color="k", alpha=0.6))
    ax.plot(0, 0, "k+", ms=10)
    ax.set_aspect("equal")
    ax.set_title(f"{title}\ncover(500km)={cover:.0f}%", fontsize=8)
    ax.tick_params(labelsize=7)


def collect_fields(nc):
    ir = nc.groups.get("infrared")
    if ir is not None and "IRWIN" in ir.variables:
        yield "IR  IRWIN (K)", read_var(ir, "x"), read_var(ir, "y"), read_var(ir, "IRWIN"), False

    pmw = nc.groups.get("passive_microwave")
    if pmw is not None:
        for path, g in walk_groups(pmw, "passive_microwave"):
            if "x" not in g.variables:
                continue
            x, y = read_var(g, "x"), read_var(g, "y")
            for name in g.variables:
                if name.startswith("TB_"):
                    yield f"{path.split('/')[-1]}  {name} (K)", x, y, read_var(g, name), False

    gp = nc.groups.get("GPROF")
    if gp is not None:
        for path, g in walk_groups(gp, "GPROF"):
            if "x" not in g.variables:
                continue
            x, y = read_var(g, "x"), read_var(g, "y")
            for name in ("surfacePrecipitation", "rainWaterPath", "iceWaterPath"):
                if name in g.variables and g.variables[name].ndim == 2:
                    yield f"GPROF  {name}", x, y, read_var(g, name), False

    rr = nc.groups.get("radar_radiometer")
    if rr is not None:
        for path, g in walk_groups(rr, "radar_radiometer"):
            if "x" not in g.variables:
                continue
            x, y = read_var(g, "x"), read_var(g, "y")
            for name, var in g.variables.items():
                if name in SKIP_2D or var.ndim != 2 or var.shape != g.variables["x"].shape:
                    continue
                if name in ("nearSurfPrecipTotRate", "mainprecipitationType", "surfaceType"):
                    yield (f"RADAR  {name}", x, y, read_var(g, name), name in CATEGORICAL)


def visualize_file(file_path, out_dir, res_km=5.0):
    """Create a multi-panel overview PNG for one overpass file."""
    base = os.path.splitext(os.path.basename(file_path))[0]
    short = base.replace("TCPRIMED_v01r01-final_", "")

    with Dataset(file_path) as nc:
        meta = nc.groups.get("overpass_metadata")
        storm = nc.groups.get("overpass_storm_metadata")

        wind = scalar(storm, "intensity")
        basin = scalar(meta, "basin")
        instrument = scalar(meta, "instrument_name")
        lat = scalar(storm, "storm_latitude")
        lon = scalar(storm, "storm_longitude")
        stage = scalar(storm, "development_level")
        dist = scalar(storm, "distance_to_land")

        fields = [(l, x, y, d, c) for l, x, y, d, c in collect_fields(nc) if x.shape == d.shape]

        # Auto-size the viewing window
        ext = 0.0
        for _, x, y, data, _ in fields:
            ok = ~(np.ma.getmaskarray(data) | np.ma.getmaskarray(x) | np.ma.getmaskarray(y))
            if ok.any():
                ext = max(ext, float(np.abs(np.ma.getdata(x)[ok]).max()),
                          float(np.abs(np.ma.getdata(y)[ok]).max()))
        half_width_km = float(np.clip(np.ceil(ext / 100.0) * 100.0, 500.0, 3000.0))

        panels = []
        for label, x, y, data, cat in fields:
            grid, extent = regrid(x, y, data, half_width_km, res_km)
            cover = disc_coverage(grid, half_width_km, res_km, 500.0)
            panels.append((label, grid, extent, cover, cat))

    if not panels:
        print(f"  âš  No plottable fields found in {short}")
        return

    ncols = 4
    nrows = max(1, int(np.ceil(len(panels) / ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.3 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (label, grid, extent, cover, cat) in zip(axes.ravel(), panels):
        ax.axis("on")
        draw_panel(ax, grid, extent, label, cover, categorical=cat)

    fig.suptitle(
        f"{short}\n{basin} | {instrument} | centre {lat}Â°N {lon}Â°E | "
        f"{wind} kt | {stage} | land dist {dist} km",
        fontsize=11
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path = os.path.join(out_dir, f"{short}_overview.png")
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"  âœ“ Saved: {out_path}")
    return out_path


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# MAIN
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
def main():
    nc_files = sorted(glob.glob(os.path.join(DATA_DIR, "TCPRIMED*.nc")))
    # Skip duplicates (files with " (1)" in name)
    nc_files = [f for f in nc_files if " (1)" not in f]

    if not nc_files:
        print("ERROR: No TCPRIMED*.nc files found in", DATA_DIR)
        sys.exit(1)

    print(f"\n{'â•' * 80}")
    print(f"TC PRIMED Data Inspection - {len(nc_files)} files")
    print(f"{'â•' * 80}")

    # â”€â”€â”€ STEP 1: Structure Inspection â”€â”€â”€
    print("\n\nâ•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—")
    print("â•‘  STEP 1: NetCDF Structure Inspection â•‘")
    print("â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•")
    for f in nc_files:
        inspect_netcdf(f)

    # â”€â”€â”€ STEP 2: Metadata & Label Extraction â”€â”€â”€
    print("\n\nâ•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—")
    print("â•‘  STEP 2: Metadata & Label Extraction â•‘")
    print("â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•")
    rows = []
    for f in nc_files:
        row = extract_metadata(f)
        rows.append(row)
        print(f"\n--- {row['file']} ---")
        for k, v in row.items():
            if k != "file":
                print(f"  {k}: {v}")

    meta_df = pd.DataFrame(rows)
    meta_csv = os.path.join(OUT_DIR, "sample_metadata.csv")
    meta_df.to_csv(meta_csv, index=False)
    print(f"\nâœ“ Metadata saved to: {meta_csv}")

    # â”€â”€â”€ STEP 3: Summary table â”€â”€â”€
    print("\n\nâ•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—")
    print("â•‘  STEP 3: Summary Table       â•‘")
    print("â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•")
    summary_cols = ["file", "basin", "instrument", "wind_kt", "pressure_hpa",
                    "stage", "storm_lat", "storm_lon", "dist_to_land_km",
                    "has_ir", "pmw_n_channels", "gprof_flag", "radar_flag"]
    avail_cols = [c for c in summary_cols if c in meta_df.columns]
    print(meta_df[avail_cols].to_string(index=False))

    # Intensity change summary
    di_cols = [c for c in meta_df.columns if c.startswith("dI_")]
    if di_cols:
        print("\nâ”€â”€â”€ Intensity Change Labels â”€â”€â”€")
        print(meta_df[["file"] + di_cols].to_string(index=False))

    # â”€â”€â”€ STEP 4: Visualization â”€â”€â”€
    print("\n\nâ•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—")
    print("â•‘  STEP 4: Storm-Centred Visualization â•‘")
    print("â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•")
    for f in nc_files:
        print(f"\nVisualizing: {os.path.basename(f)}")
        visualize_file(f, OUT_DIR)

    print(f"\n\n{'â•' * 80}")
    print(f"INSPECTION COMPLETE")
    print(f"Output directory: {OUT_DIR}")
    print(f"{'â•' * 80}")


if __name__ == "__main__":
    main()

