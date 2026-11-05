# %%
"""
03_show_viz_results.py - print the QC table and display the overview images from tcprimed_viz.py.

Run cells in a notebook, or:  python 03_show_viz_results.py <viz_out_dir>
"""
import glob
import os
import sys

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd

# %% ------------------------------------------------------------------ config
VIZ_DIR = sys.argv[1] if len(sys.argv) > 1 else "/content/viz_out"
pd.set_option("display.width", 220, "display.max_columns", 40)

# %% ------------------------------------------------------------------ QC table
summary = pd.read_csv(os.path.join(VIZ_DIR, "summary.csv"))
cols = ["file", "basin", "instrument", "wind_kt", "pressure_hpa", "stage", "dist_to_land_km",
        "cover500_IR_pct", "cover500_85GHz_pct", "centre_in_85GHz_swath",
        "flag_overland", "flag_weak", "flag_partial_swath", "flag_ok_for_training"]
print(summary[[c for c in cols if c in summary.columns]].round(1).to_string(index=False))
print(f"\n{int(summary['flag_ok_for_training'].sum())} of {len(summary)} samples pass the QC filter")

# %% ------------------------------------------------------------------ show every overview image
for path in sorted(glob.glob(os.path.join(VIZ_DIR, "*_overview.png"))):
    img = mpimg.imread(path)
    plt.figure(figsize=(18, 18 * img.shape[0] / img.shape[1]))
    plt.imshow(img)
    plt.axis("off")
    plt.title(os.path.basename(path), fontsize=11)
    plt.tight_layout()
    plt.show()
