# %%
"""
02_report_catalogue.py - print + plot everything the catalogue says about the dataset.

Cell format ("# %%") so you can paste cells into a notebook one by one, or just run:
    python 02_report_catalogue.py <catalogue_dir>

Reads  : file_index.csv, overpass_meta.csv   (made by 01_build_catalogue.py)
Writes : storm_table.csv (one row per storm, with a train/val/test/NIO split), catalogue_overview.png
"""
import os
import sys
import zlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %% ------------------------------------------------------------------ config
CAT_DIR = sys.argv[1] if len(sys.argv) > 1 else "/content/drive/MyDrive/tcprimed_catalogue"
MIN_WIND_KT = 34          # tropical-storm strength and above
NIO_BASIN = "IO"          # TC PRIMED code for the North Indian Ocean (Bay of Bengal + Arabian Sea)
pd.set_option("display.width", 220, "display.max_columns", 40, "display.max_rows", 100)


def show(obj, title=None):
    if title:
        print(f"\n=== {title} ===")
    try:
        from IPython.display import display
        display(obj)
    except Exception:
        print(obj.to_string() if hasattr(obj, "to_string") else obj)


# %% ------------------------------------------------------------------ load + merge
idx = pd.read_csv(os.path.join(CAT_DIR, "file_index.csv"))
meta = pd.read_csv(os.path.join(CAT_DIR, "overpass_meta.csv"))
df = idx[idx["kind"] == "overpass"].merge(meta, on="rel", how="inner", suffixes=("", "_m"))
bad = df[~df["ok"].astype(bool)]
df = df[df["ok"].astype(bool)].copy()

df["year"] = df["season"].astype("Int64")
df["has_ir"] = df["ir_flag"].fillna(0) > 0
df["over_water"] = df["dist_to_land_km"] > 0
df["strong_enough"] = df["wind_kt"] >= MIN_WIND_KT
df["usable"] = df["over_water"] & df["strong_enough"]
for h in (6, 12, 18, 24):
    df[f"has_dI{h}"] = df[f"dI_+{h}h"].notna()

bins = [0, 34, 64, 83, 96, 113, 137, 400]
names = ["TD (<34)", "TS (34-63)", "Cat1 (64-82)", "Cat2 (83-95)", "Cat3 (96-112)", "Cat4 (113-136)", "Cat5 (137+)"]
df["category"] = pd.cut(df["wind_kt"], bins=bins, labels=names, right=False)

# %% ------------------------------------------------------------------ 1. what is in the archive
print(f"Files indexed: {len(idx):,}  | size: {idx['size_mb'].sum() / 1e3:,.1f} GB")
show(idx.groupby("kind").agg(files=("rel", "size"), GB=("size_mb", lambda s: s.sum() / 1e3),
                             mean_MB=("size_mb", "mean")).round(2), "files by kind")
print(f"Overpass files with metadata harvested: {len(df):,} of {(idx['kind'] == 'overpass').sum():,}"
      f"  (failed to open: {len(bad):,})")
if len(bad):
    show(bad[["rel", "error"]].head(10), "first failures")
if len(df) < 0.9 * (idx["kind"] == "overpass").sum():
    print("NOTE: harvest is incomplete - all numbers below describe the harvested subset only.")

# %% ------------------------------------------------------------------ 2. basins, instruments, years
basin = df.groupby("basin").agg(storms=("storm", "nunique"), overpasses=("rel", "size"),
                                max_wind_kt=("wind_kt", "max"), usable=("usable", "sum"))
basin["overpasses_per_storm"] = (basin["overpasses"] / basin["storms"]).round(1)
show(basin, "per basin")
show(df.groupby(["instrument", "platform"]).size().rename("overpasses").sort_values(ascending=False),
     "per instrument / platform")
show(df.pivot_table(index=(df["year"] // 5 * 5), columns="basin", values="rel", aggfunc="size", fill_value=0),
     "overpasses per 5-year block x basin")

# %% ------------------------------------------------------------------ 3. labels: how complete?
lab = pd.DataFrame({
    "wind_kt": df["wind_kt"].notna().mean(), "pressure_hpa": df["pressure_hpa"].notna().mean(),
    "stage": df["stage"].notna().mean(), "dI +6h": df["has_dI6"].mean(), "dI +12h": df["has_dI12"].mean(),
    "dI +18h": df["has_dI18"].mean(), "dI +24h": df["has_dI24"].mean()}, index=["fraction available"]).T
show((lab * 100).round(1), "label availability (% of overpasses)")
show((df.groupby("basin")[["wind_kt", "pressure_hpa", "dI_+24h"]].agg(lambda s: s.notna().mean()) * 100).round(1),
     "label availability by basin (%)")
show(df["stage"].value_counts(dropna=False).rename("overpasses"), "development_level codes")
show(df["category"].value_counts().reindex(names).rename("overpasses"), "intensity categories (wind_kt)")
show(pd.crosstab(df["basin"], df["category"]), "intensity category x basin")

# %% ------------------------------------------------------------------ 4. modalities
show(df.groupby("basin").agg(ir_available_pct=("has_ir", lambda s: 100 * s.mean()),
                             gprof_pct=("gprof_flag", lambda s: 100 * (s.fillna(0) > 0).mean()),
                             radar_pct=("radar_flag", lambda s: 100 * (s.fillna(0) > 0).mean())).round(1),
     "modality availability by basin (%)")
show(df.groupby(["basin", "ir_flag"]).size().rename("overpasses"), "ir_flag by basin (meaning of 1 vs 2: check docs)")
show(df["ir_shape"].value_counts(dropna=False).head(10).rename("overpasses"), "IR grid shapes")
show(df["pmw_channels"].value_counts().head(10).rename("overpasses"), "microwave channel sets (top 10)")
show(df["coverage_fraction"].describe().round(1), "S1 swath coverage within 750 km (%)")

# %% ------------------------------------------------------------------ 5. what is trainable?
print(f"Over water AND >= {MIN_WIND_KT} kt: {df['usable'].sum():,} of {len(df):,} overpasses "
      f"({100 * df['usable'].mean():.1f}%)")
print(f"Over land: {(~df['over_water'].fillna(True)).sum():,} | weaker than {MIN_WIND_KT} kt: "
      f"{(~df['strong_enough'].fillna(True)).sum():,}")

# %% ------------------------------------------------------------------ 6. storm table + split
storm = df.groupby("storm").agg(
    basin=("basin", "first"), season=("year", "first"), n_overpass=("rel", "size"),
    max_wind_kt=("wind_kt", "max"), n_usable=("usable", "sum"), n_with_ir=("has_ir", "sum"),
    n_dI24=("has_dI24", "sum"), instruments=("instrument", "nunique"),
    first_obs=("obs_time_utc", "min"), last_obs=("obs_time_utc", "max")).reset_index()


def assign_split(row):
    if row["basin"] == NIO_BASIN:
        return "holdout_NIO"                      # whole basin kept out for the cross-basin test
    h = zlib.crc32(row["storm"].encode()) % 100   # deterministic: same storm -> same split, always
    return "train" if h < 70 else "val" if h < 85 else "test"


storm["split"] = storm.apply(assign_split, axis=1)
storm.to_csv(os.path.join(CAT_DIR, "storm_table.csv"), index=False)
show(storm.groupby(["split"]).agg(storms=("storm", "size"), overpasses=("n_overpass", "sum"),
                                  usable=("n_usable", "sum")), "split summary (by STORM, never by image)")
show(storm.groupby(["basin", "split"]).size().unstack(fill_value=0), "storms per basin x split")

nio = storm[storm["basin"] == NIO_BASIN].sort_values(["season", "storm"])
show(nio[["storm", "season", "n_overpass", "max_wind_kt", "n_usable", "n_with_ir", "n_dI24"]],
     f"North Indian Ocean storms ({len(nio)})")
print("Strong NIO storms (>=64 kt):", int((nio["max_wind_kt"] >= 64).sum()),
      "| NIO usable overpasses:", int(nio["n_usable"].sum()))

# %% ------------------------------------------------------------------ 7. plots
fig, ax = plt.subplots(2, 3, figsize=(18, 9))
df.pivot_table(index="year", columns="basin", values="rel", aggfunc="size", fill_value=0).plot.bar(
    stacked=True, ax=ax[0, 0], width=0.9, legend=True)
ax[0, 0].set_title("overpasses per year"); ax[0, 0].tick_params(labelsize=6)
basin[["storms"]].plot.bar(ax=ax[0, 1], legend=False); ax[0, 1].set_title("storms per basin")
for b, g in df.groupby("basin"):
    ax[0, 2].hist(g["wind_kt"].dropna(), bins=30, histtype="step", density=True, label=b)
ax[0, 2].set_title("wind (kt) distribution per basin"); ax[0, 2].legend()
(lab * 100).plot.barh(ax=ax[1, 0], legend=False); ax[1, 0].set_title("label availability (%)")
df["category"].value_counts().reindex(names).plot.bar(ax=ax[1, 1]); ax[1, 1].set_title("intensity categories")
ax[1, 1].tick_params(labelsize=7)
ax[1, 2].hist(df["dist_to_land_km"].dropna().clip(-1500, 4000), bins=60)
ax[1, 2].axvline(0, color="r"); ax[1, 2].set_title("distance to land (km), <0 = over land")
plt.tight_layout()
plt.savefig(os.path.join(CAT_DIR, "catalogue_overview.png"), dpi=110)
plt.show()
