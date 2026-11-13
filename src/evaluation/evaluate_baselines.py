"""
Comprehensive Baseline and Model Evaluation Engine for Phase 2C.
Evaluates:
1. Persistence Baselines (Wind, Pressure, Intensity Change = 0, Track = Last Known)
2. Statistical / Tabular Linear Baselines (Fitted on TRAIN partition only)
3. TC-GraphFormer Neural Model (Loaded from checkpoints/best_checkpoint.pt)
Generates:
- reports/phase2c/baseline_comparison.csv
- reports/phase2c/task_metrics_by_horizon.csv
- reports/phase2c/basin_evaluation.csv
"""

import os
import sys
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.dataset import TCPRIMEDDataset, STAGE_MAPPING
from src.data.collate import collate_multimodal_batch
from src.models.tc_graphformer import TCGraphFormer


def haversine_distance_km(lat1, lon1, lat2, lon2):
    """Compute great-circle distance between two geographic coordinates in km."""
    R = 6371.0 # Earth radius in km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0)**2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c


def evaluate_all(
    sample_index_csv: str = "reports/pipeline/sample_index.csv",
    checkpoint_path: str = "checkpoints/best_checkpoint.pt",
    d_model: int = 128,
    output_dir: str = "reports/phase2e",
):
    df = pd.read_csv(sample_index_csv)
    train_df = df[df["partition"] == "TRAIN"].copy()
    val_df = df[df["partition"] == "VAL"].copy()
    test_df = df[df["partition"] == "TEST"].copy()

    print(f"[EVAL] Train samples: {len(train_df)}, Val samples: {len(val_df)}, Test samples: {len(test_df)}")

    # -------------------------------------------------------------
    # 1. Fit Tabular Linear Models on TRAIN data only
    # -------------------------------------------------------------
    # Wind linear model
    train_v_valid = train_df.dropna(subset=["intensity_vmax_kt", "storm_speed_mps"])
    X_v_train = np.column_stack([
        np.ones(len(train_v_valid)),
        train_v_valid["storm_speed_mps"].values,
        train_v_valid["distance_to_land_km"].fillna(500.0).values
    ])
    y_v_train = train_v_valid["intensity_vmax_kt"].values
    # Ordinary Least Squares
    beta_v, _, _, _ = np.linalg.lstsq(X_v_train, y_v_train, rcond=None)

    # Pressure linear model from intensity
    train_p_valid = train_df.dropna(subset=["central_min_pressure_hpa", "intensity_vmax_kt"])
    X_p_train = np.column_stack([
        np.ones(len(train_p_valid)),
        train_p_valid["intensity_vmax_kt"].values
    ])
    y_p_train = train_p_valid["central_min_pressure_hpa"].values
    beta_p, _, _, _ = np.linalg.lstsq(X_p_train, y_p_train, rcond=None)

    # Stage majority class
    stage_counts = train_df["development_stage_raw"].map(lambda x: STAGE_MAPPING.get(str(x).strip().upper(), 5)).value_counts()
    majority_stage = stage_counts.idxmax()

    # -------------------------------------------------------------
    # 2. Load TC-GraphFormer
    # -------------------------------------------------------------
    device = "cpu"
    model = TCGraphFormer(d_model=d_model).to(device)
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        model_dict = model.state_dict()
        matched = {k: v for k, v in state.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(matched)
        model.load_state_dict(model_dict)
        print(f"[EVAL] Loaded TC-GraphFormer ({len(matched)}/{len(model_dict)} tensors matched) from {checkpoint_path}")
    model.eval()

    # -------------------------------------------------------------
    # 3. Evaluate on Validation & Test Partitions
    # -------------------------------------------------------------
    partitions = [("VAL", val_df), ("TEST", test_df)]
    baseline_records = []
    horizon_records = []
    basin_records = []

    for part_name, part_df in partitions:
        print(f"\n--- Evaluating Partition: {part_name} ({len(part_df)} samples) ---")

        # DataLoader for neural model evaluation
        ds = TCPRIMEDDataset(sample_index_csv=sample_index_csv, partition=part_name, grid_size=64)
        loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate_multimodal_batch)

        nn_intensity_preds = []
        nn_pressure_preds = []
        nn_stage_preds = []
        nn_dv_preds = []
        nn_ri_probs = []

        with torch.no_grad():
            for b_idx, batch in enumerate(loader):
                if b_idx % 10 == 0:
                    print(f"  [{part_name}] Evaluating batch {b_idx+1}/{len(loader)}...", flush=True)
                ir = batch["ir"].to(device)
                pmw = batch["pmw"].to(device)
                gprof = batch["gprof"].to(device)
                radar = batch["radar"].to(device)
                env = batch["env"].to(device)
                masks = batch["modality_masks"].to(device)

                preds = model(ir, pmw, gprof, radar, env, masks)
                nn_intensity_preds.extend(preds["intensity"].numpy().tolist())
                nn_pressure_preds.extend(preds["pressure"].numpy().tolist())
                nn_stage_preds.extend(preds["stage_logits"].argmax(dim=-1).numpy().tolist())
                nn_dv_preds.extend(preds["intensity_change"].numpy().tolist())
                nn_ri_probs.extend(torch.sigmoid(preds["ri_logits"]).numpy().tolist())

        part_df["nn_intensity"] = nn_intensity_preds
        part_df["nn_pressure"] = nn_pressure_preds
        part_df["nn_stage"] = nn_stage_preds
        part_df["nn_ri_prob"] = nn_ri_probs

        # --- A. Intensity (Wind Speed) ---
        v_true = part_df["intensity_vmax_kt"].dropna()
        idx_v = v_true.index
        # Persistence: predict training mean (or lagged wind)
        v_mean_train = train_df["intensity_vmax_kt"].mean()
        v_pred_persist = np.full_like(v_true, v_mean_train)
        
        # Linear Model
        X_v_part = np.column_stack([
            np.ones(len(idx_v)),
            part_df.loc[idx_v, "storm_speed_mps"].fillna(5.0).values,
            part_df.loc[idx_v, "distance_to_land_km"].fillna(500.0).values
        ])
        v_pred_linear = X_v_part @ beta_v
        v_pred_nn = part_df.loc[idx_v, "nn_intensity"].values

        for m_name, preds_arr in [("Climatology (Mean)", v_pred_persist), ("Linear Regression", v_pred_linear), ("TC-GraphFormer", v_pred_nn)]:
            err = preds_arr - v_true.values
            baseline_records.append({
                "partition": part_name,
                "task": "Current Intensity (v_max)",
                "model": m_name,
                "n_samples": len(v_true),
                "mae": round(float(np.mean(np.abs(err))), 2),
                "rmse": round(float(np.sqrt(np.mean(err**2))), 2),
                "bias": round(float(np.mean(err)), 2),
                "notes": "kt units"
            })

        # --- B. Central Minimum Pressure ---
        p_true = part_df["central_min_pressure_hpa"].dropna()
        idx_p = p_true.index
        p_mean_train = train_df["central_min_pressure_hpa"].mean()
        p_pred_persist = np.full_like(p_true, p_mean_train)

        X_p_part = np.column_stack([
            np.ones(len(idx_p)),
            part_df.loc[idx_p, "intensity_vmax_kt"].fillna(v_mean_train).values
        ])
        p_pred_linear = X_p_part @ beta_p
        p_pred_nn = part_df.loc[idx_p, "nn_pressure"].values

        for m_name, preds_arr in [("Climatology (Mean)", p_pred_persist), ("Linear Pressure-Wind Fit", p_pred_linear), ("TC-GraphFormer", p_pred_nn)]:
            err = preds_arr - p_true.values
            baseline_records.append({
                "partition": part_name,
                "task": "Central Pressure (p_min)",
                "model": m_name,
                "n_samples": len(p_true),
                "mae": round(float(np.mean(np.abs(err))), 2),
                "rmse": round(float(np.sqrt(np.mean(err**2))), 2),
                "bias": round(float(np.mean(err)), 2),
                "notes": "hPa units (Note: TC-GraphFormer initial bias unadjusted without extended training)"
            })

        # --- C. Development Stage Classification ---
        s_true = part_df["development_stage_raw"].map(lambda x: STAGE_MAPPING.get(str(x).strip().upper(), 5)).values
        s_pred_maj = np.full_like(s_true, majority_stage)
        s_pred_nn = part_df["nn_stage"].values

        acc_maj = float(np.mean(s_pred_maj == s_true)) * 100
        acc_nn = float(np.mean(s_pred_nn == s_true)) * 100

        baseline_records.append({
            "partition": part_name,
            "task": "Development Stage",
            "model": "Majority Class (TS)",
            "n_samples": len(s_true),
            "mae": round(100 - acc_maj, 2), # error rate
            "rmse": 0.0,
            "bias": 0.0,
            "notes": f"Accuracy: {acc_maj:.2f}%"
        })
        baseline_records.append({
            "partition": part_name,
            "task": "Development Stage",
            "model": "TC-GraphFormer",
            "n_samples": len(s_true),
            "mae": round(100 - acc_nn, 2),
            "rmse": 0.0,
            "bias": 0.0,
            "notes": f"Accuracy: {acc_nn:.2f}%"
        })

        # --- D. Multi-Horizon Intensity Change ---
        nn_dv_matrix = np.array(nn_dv_preds) # (N, 4)
        for h_idx, h in enumerate([6, 12, 18, 24]):
            col = f"intensity_change_{h}h_kt"
            valid_mask = part_df[col].notna()
            dv_true = part_df.loc[valid_mask, col].values
            
            # Persistence: predict 0.0 kt (no intensity change)
            dv_pred_zero = np.zeros_like(dv_true)
            mae_zero = float(np.mean(np.abs(dv_pred_zero - dv_true)))
            rmse_zero = float(np.sqrt(np.mean((dv_pred_zero - dv_true)**2)))

            # Neural model
            dv_pred_nn_h = nn_dv_matrix[valid_mask.values, h_idx]
            mae_nn = float(np.mean(np.abs(dv_pred_nn_h - dv_true)))
            rmse_nn = float(np.sqrt(np.mean((dv_pred_nn_h - dv_true)**2)))

            horizon_records.append({
                "partition": part_name,
                "horizon_hours": h,
                "n_valid_targets": len(dv_true),
                "persistence_mae_kt": round(mae_zero, 2),
                "persistence_rmse_kt": round(rmse_zero, 2),
                "tc_graphformer_mae_kt": round(mae_nn, 2),
                "tc_graphformer_rmse_kt": round(rmse_nn, 2),
                "improvement_kt": round(mae_zero - mae_nn, 2)
            })

        # --- E. Future Track Persistence ---
        import netCDF4 as nc
        # Cache current positions for part_df
        cur_coords = {}
        for _, row in part_df[part_df["future_lat_6h"].notna()].iterrows():
            obs_id = row["sample_id"]
            if obs_id not in cur_coords:
                nc_p = row["processed_container_path"].replace("\\", "/")
                grp_p = row["container_group_path"].split("/")[-1]
                try:
                    with nc.Dataset(nc_p, "r") as ds:
                        tgt = ds.groups["overpasses"].groups[grp_p].groups["targets"]
                        cur_coords[obs_id] = (float(tgt.variables["storm_latitude"][:]), float(tgt.variables["storm_longitude"][:]))
                except Exception:
                    cur_coords[obs_id] = (np.nan, np.nan)

        for h in [6, 12, 18, 24]:
            lat_col, lon_col = f"future_lat_{h}h", f"future_lon_{h}h"
            valid_track = part_df[part_df[lat_col].notna()].copy()
            if len(valid_track) > 0:
                cur_lats = np.array([cur_coords.get(s, (np.nan, np.nan))[0] for s in valid_track["sample_id"]])
                cur_lons = np.array([cur_coords.get(s, (np.nan, np.nan))[1] for s in valid_track["sample_id"]])
                fut_lats = valid_track[lat_col].values
                fut_lons = valid_track[lon_col].values

                valid_coords = ~np.isnan(cur_lats) & ~np.isnan(cur_lons)
                if np.any(valid_coords):
                    dist_km = haversine_distance_km(cur_lats[valid_coords], cur_lons[valid_coords], fut_lats[valid_coords], fut_lons[valid_coords])
                    mean_dist = float(np.mean(dist_km))
                    median_dist = float(np.median(dist_km))
                else:
                    mean_dist, median_dist = 0.0, 0.0
            else:
                mean_dist, median_dist = 0.0, 0.0

            horizon_records.append({
                "partition": part_name,
                "horizon_hours": f"Track_{h}h",
                "n_valid_targets": len(valid_track),
                "persistence_mae_kt": round(mean_dist, 2), # mean position error in km
                "persistence_rmse_kt": round(median_dist, 2), # median error in km
                "tc_graphformer_mae_kt": 0.0,
                "tc_graphformer_rmse_kt": 0.0,
                "improvement_kt": 0.0
            })

        # --- F. Basin-wise Breakdown ---
        for b, b_df in part_df.groupby("basin"):
            v_b = b_df["intensity_vmax_kt"].dropna()
            if len(v_b) > 0:
                nn_v_b = b_df.loc[v_b.index, "nn_intensity"].values
                mae_b = float(np.mean(np.abs(nn_v_b - v_b.values)))
            else:
                mae_b = 0.0

            basin_records.append({
                "partition": part_name,
                "basin": b,
                "total_overpasses": len(b_df),
                "valid_intensity_targets": len(v_b),
                "tc_graphformer_wind_mae_kt": round(mae_b, 2),
                "mean_wind_kt": round(float(v_b.mean()), 2) if len(v_b) > 0 else 0.0,
            })

    # Save outputs
    df_base = pd.DataFrame(baseline_records)
    df_horiz = pd.DataFrame(horizon_records)
    df_basin = pd.DataFrame(basin_records)

    out_base = os.path.join(output_dir, "baseline_comparison.csv")
    out_horiz = os.path.join(output_dir, "task_metrics_by_horizon.csv")
    out_basin = os.path.join(output_dir, "basin_evaluation.csv")

    os.makedirs(output_dir, exist_ok=True)
    df_base.to_csv(out_base, index=False)
    df_horiz.to_csv(out_horiz, index=False)
    df_basin.to_csv(out_basin, index=False)

    print(f"\n[OK] Baseline Comparison saved to: {out_base}")
    print(f"[OK] Task Metrics by Horizon saved to: {out_horiz}")
    print(f"[OK] Basin Evaluation saved to: {out_basin}")

    return df_base, df_horiz, df_basin


if __name__ == "__main__":
    evaluate_all()
