"""
Comprehensive Baseline and Model Evaluation Engine for TC-GraphFormer (Phase 2E.4.1).
Evaluates:
1. Climatology and Persistence Baselines (Wind, Pressure, Stage, Intensity Change, RI Base Rate, Track)
2. Tabular Linear Baselines (Fitted strictly on TRAIN partition)
3. TC-GraphFormer Neural Model (in historical sequence or single-obs mode)

Outputs:
- reports/phase2e4/baseline_comparison.csv
- reports/phase2e4/task_metrics_by_horizon.csv
- reports/phase2e4/basin_evaluation.csv
- reports/phase2e4/evaluation_summary.json

Enforces:
- Geodesic / Euclidean track displacement errors in kilometers (km)
- Persistence RMSE is strictly root-mean-squared error (median error reported separately)
- Track climatology baseline fitted strictly on TRAIN valid displacement vectors (never 0.0 placeholder)
- Missing targets (n=0) reported as null/None rather than 0.0
- Strict checkpoint validation: explicitly requires --smoke-test for untrained or partial checkpoints
- Intensity errors in knots (kt), pressure errors in hectopascals (hPa)
"""

import os
import sys
import math
import json
import argparse
from typing import Tuple, Dict, Any, List, Optional
import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.dataset import (
    TCPRIMEDDataset,
    TCPRIMEDSequenceDataset,
    STAGE_MAPPING_CANONICAL,
    compute_tangent_plane_displacement_km,
)
from src.data.collate import collate_multimodal_batch, collate_sequence_batch
from src.models.tc_graphformer import TCGraphFormer


def haversine_distance_km(lat1, lon1, lat2, lon2):
    """Compute great-circle distance between two geographic coordinates in km."""
    R = 6371.0  # Earth radius in km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(np.clip(a, 0.0, 1.0)), np.sqrt(np.clip(1.0 - a, 0.0, 1.0)))
    return R * c


def compute_pr_auc_and_brier(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Computes PR-AUC, Brier score, precision at default threshold 0.5, and recall at threshold 0.5
    using pure NumPy without external sklearn dependencies.
    """
    if len(y_true) == 0:
        return 0.0, 0.0, 0.0, 0.0

    brier = float(np.mean((y_prob - y_true) ** 2))
    pred_binary = (y_prob >= 0.5).astype(float)
    tp = float(np.sum((pred_binary == 1.0) & (y_true == 1.0)))
    fp = float(np.sum((pred_binary == 1.0) & (y_true == 0.0)))
    fn = float(np.sum((pred_binary == 0.0) & (y_true == 1.0)))
    prec_at_50 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec_at_50 = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    total_pos = float(np.sum(y_true))
    if total_pos == 0:
        return 0.0, round(brier, 4), round(prec_at_50, 4), round(rec_at_50, 4)

    order = np.argsort(-y_prob)
    y_true_sorted = y_true[order]
    tps = np.cumsum(y_true_sorted)
    fps = np.cumsum(1.0 - y_true_sorted)

    recalls = tps / (total_pos + 1e-6)
    precisions = tps / (tps + fps + 1e-6)

    r_padded = np.concatenate([[0.0], recalls])
    p_padded = np.concatenate([[precisions[0]], precisions])
    pr_auc = float(np.sum((r_padded[1:] - r_padded[:-1]) * p_padded[1:]))
    return round(pr_auc, 4), round(brier, 4), round(prec_at_50, 4), round(rec_at_50, 4)


def compute_train_track_climatology(
    sample_index_csv: str,
) -> Dict[int, Optional[Tuple[float, float]]]:
    """
    Compute mean valid training displacement vectors (mean_dx, mean_dy) in km
    for each horizon [6, 12, 18, 24] strictly from the TRAIN partition.
    Returns None for horizons where training targets are unavailable.
    """
    if not os.path.exists(sample_index_csv):
        return {h: None for h in [6, 12, 18, 24]}

    df = pd.read_csv(sample_index_csv)
    train_df = df[df["partition"] == "TRAIN"]
    if len(train_df) == 0:
        return {h: None for h in [6, 12, 18, 24]}

    clim_displacements = {h: [] for h in [6, 12, 18, 24]}

    # Check if coordinates are in CSV or need NetCDF lookup
    has_cur_coords = ("storm_latitude" in train_df.columns and "storm_longitude" in train_df.columns)

    if has_cur_coords:
        for h in [6, 12, 18, 24]:
            lat_col, lon_col = f"future_lat_{h}h", f"future_lon_{h}h"
            valid_m = (
                train_df[lat_col].notna()
                & train_df[lon_col].notna()
                & train_df["storm_latitude"].notna()
                & train_df["storm_longitude"].notna()
            )
            for _, r in train_df[valid_m].iterrows():
                dx, dy = compute_tangent_plane_displacement_km(
                    float(r["storm_latitude"]), float(r["storm_longitude"]),
                    float(r[lat_col]), float(r[lon_col]),
                )
                clim_displacements[h].append((dx, dy))
    else:
        # Resolve from NetCDF containers if available
        import netCDF4 as nc
        container_groups = train_df.groupby("processed_container_path")
        for nc_p, grp_df in container_groups:
            if not os.path.exists(nc_p):
                continue
            try:
                with nc.Dataset(nc_p, "r") as root:
                    for _, row in grp_df.iterrows():
                        g_name = row["container_group_path"].split("/")[-1]
                        if "overpasses" in root.groups and g_name in root.groups["overpasses"].groups:
                            tgt = root.groups["overpasses"].groups[g_name].groups["targets"]
                            c_lat = float(tgt.variables["storm_latitude"][:])
                            c_lon = float(tgt.variables["storm_longitude"][:])
                            for h in [6, 12, 18, 24]:
                                f_lat = row.get(f"future_lat_{h}h")
                                f_lon = row.get(f"future_lon_{h}h")
                                if (
                                    not pd.isna(f_lat)
                                    and not pd.isna(f_lon)
                                    and not np.isnan(c_lat)
                                    and not np.isnan(c_lon)
                                ):
                                    dx, dy = compute_tangent_plane_displacement_km(
                                        c_lat, c_lon, float(f_lat), float(f_lon)
                                    )
                                    clim_displacements[h].append((dx, dy))
            except Exception:
                pass

    climatology_vectors = {}
    for h in [6, 12, 18, 24]:
        arr = clim_displacements[h]
        if len(arr) > 0:
            arr_np = np.array(arr)
            climatology_vectors[h] = (float(arr_np[:, 0].mean()), float(arr_np[:, 1].mean()))
        else:
            climatology_vectors[h] = None

    return climatology_vectors


def compute_track_errors(
    true_vec: np.ndarray,
    pred_vec: np.ndarray,
    clim_vec: Optional[Tuple[float, float]] = None,
    vel_vec: Optional[np.ndarray] = None,
) -> Dict[str, Optional[float]]:
    """
    Compute rigorous Euclidean error metrics for future track displacement (km).
    Calculates true mathematical RMSE (sqrt of mean squared euclidean error),
    MAE, median error for persistence baseline, training climatology baseline,
    observation-time causal velocity extrapolation baseline, and TC-GraphFormer predictions.
    """
    if len(true_vec) == 0:
        return {
            "persist_mae": None, "persist_rmse": None, "persist_median": None,
            "clim_mae": None, "clim_rmse": None,
            "vel_mae": None, "vel_rmse": None,
            "nn_mae": None, "nn_rmse": None,
            "impr_persist": None, "impr_clim": None, "impr_vel": None,
        }

    # 1. Persistence baseline: assumes zero displacement [0, 0]
    persist_err_km = np.sqrt(true_vec[:, 0] ** 2 + true_vec[:, 1] ** 2)
    persist_mae = round(float(np.mean(persist_err_km)), 2)
    persist_rmse = round(float(np.sqrt(np.mean(persist_err_km ** 2))), 2)
    persist_median = round(float(np.median(persist_err_km)), 2)

    # 2. Climatology baseline: training mean displacement vector (clim_dx, clim_dy)
    if clim_vec is not None:
        clim_dx, clim_dy = clim_vec
        clim_err_km = np.sqrt((true_vec[:, 0] - clim_dx) ** 2 + (true_vec[:, 1] - clim_dy) ** 2)
        clim_mae = round(float(np.mean(clim_err_km)), 2)
        clim_rmse = round(float(np.sqrt(np.mean(clim_err_km ** 2))), 2)
    else:
        clim_mae, clim_rmse = None, None

    # 3. Observation-time strictly causal velocity extrapolation baseline
    if vel_vec is not None and len(vel_vec) == len(true_vec):
        vel_err_km = np.sqrt((true_vec[:, 0] - vel_vec[:, 0]) ** 2 + (true_vec[:, 1] - vel_vec[:, 1]) ** 2)
        vel_mae = round(float(np.mean(vel_err_km)), 2)
        vel_rmse = round(float(np.sqrt(np.mean(vel_err_km ** 2))), 2)
    else:
        vel_mae, vel_rmse = None, None

    # 4. Model displacement vector error
    diff_vec = pred_vec - true_vec
    nn_err_km = np.sqrt(diff_vec[:, 0] ** 2 + diff_vec[:, 1] ** 2)
    nn_mae = round(float(np.mean(nn_err_km)), 2)
    nn_rmse = round(float(np.sqrt(np.mean(nn_err_km ** 2))), 2)

    impr_persist = round(persist_mae - nn_mae, 2)
    impr_clim = round(clim_mae - nn_mae, 2) if clim_mae is not None else None
    impr_vel = round(vel_mae - nn_mae, 2) if vel_mae is not None else None

    return {
        "persist_mae": persist_mae,
        "persist_rmse": persist_rmse,
        "persist_median": persist_median,
        "clim_mae": clim_mae,
        "clim_rmse": clim_rmse,
        "vel_mae": vel_mae,
        "vel_rmse": vel_rmse,
        "nn_mae": nn_mae,
        "nn_rmse": nn_rmse,
        "impr_persist": impr_persist,
        "impr_clim": impr_clim,
        "impr_vel": impr_vel,
    }


def load_model_for_evaluation(
    checkpoint_path: Optional[str] = None,
    d_model: int = 128,
    strict: bool = True,
    smoke_test: bool = False,
    device: str = "cpu",
) -> Tuple[TCGraphFormer, Dict[str, Any]]:
    """
    Load TCGraphFormer model and validate checkpoint compatibility and training provenance.
    Fails loudly if checkpoint is untrained or incomplete unless smoke_test=True.
    """
    model = TCGraphFormer(d_model=d_model, num_stages=5).to(device)
    model.eval()

    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        if not smoke_test:
            raise FileNotFoundError(
                f"Trained checkpoint not found at: '{checkpoint_path}'. "
                f"Valid trained weights are required for model evaluation. "
                f"Pass --smoke-test to execute a forward-pass test with random initialization."
            )
        status = {
            "is_trained": False,
            "checkpoint_path": checkpoint_path,
            "status": "smoke_test_missing_checkpoint",
            "model_label": "TC-GraphFormer (Smoke-Test Untrained - Not Forecasting Skill)",
            "message": "No checkpoint supplied or file not found. Running random initialization for verification only.",
            "matched_keys": 0,
            "total_keys": len(model.state_dict()),
        }
        print(f"[EVAL WARN] {status['message']}")
        return model, status

    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state.keys())

    matched = {}
    mismatches = []
    for k, v in state.items():
        if k in model_keys:
            if model.state_dict()[k].shape == v.shape:
                matched[k] = v
            else:
                mismatches.append(k)

    missing = sorted(list(model_keys - set(matched.keys())))
    is_compatible = (len(missing) == 0 and len(mismatches) == 0)

    # Verifiable training provenance check (strictly requiring supervised_multitask_training and epoch >= 1)
    has_training_provenance = (
        isinstance(ckpt.get("provenance"), dict)
        and (ckpt.get("is_trained") is True)
        and (ckpt.get("provenance", {}).get("workflow") == "supervised_multitask_training")
        and (ckpt.get("provenance", {}).get("is_trained") is True)
        and (ckpt.get("epoch", 0) >= 1)
    )

    is_qualified_trained = is_compatible and has_training_provenance

    if strict and not is_qualified_trained and not smoke_test:
        raise ValueError(
            f"Checkpoint '{checkpoint_path}' does not qualify as a valid trained model! "
            f"Compatible parameters: {is_compatible} ({len(matched)}/{len(model_keys)}), "
            f"Training provenance: {has_training_provenance}. "
            f"Pass --smoke-test to run pipeline verification with an untrained or partial checkpoint."
        )

    model.load_state_dict(matched, strict=False)

    if is_qualified_trained:
        status_name = "trained_checkpoint"
        model_label = "TC-GraphFormer"
    elif is_compatible:
        status_name = "smoke_test_untrained"
        model_label = "TC-GraphFormer (Smoke-Test Untrained - Not Forecasting Skill)"
    else:
        status_name = "smoke_test_partial"
        model_label = "TC-GraphFormer (Smoke-Test Partial - Not Forecasting Skill)"

    status = {
        "is_trained": is_qualified_trained,
        "checkpoint_path": checkpoint_path,
        "status": status_name,
        "model_label": model_label,
        "matched_keys": len(matched),
        "total_keys": len(model_keys),
        "missing_keys_count": len(missing),
        "shape_mismatches_count": len(mismatches),
        "epoch": ckpt.get("epoch", None),
        "metrics": ckpt.get("metrics", {}),
    }

    if not is_qualified_trained:
        print(f"[EVAL WARN] Running in SMOKE-TEST mode with {status_name}. Results must NOT be interpreted as forecasting skill.")
    else:
        print(f"[EVAL] Checkpoint '{checkpoint_path}' validated as trained model. All {len(model_keys)} parameters loaded.")

    return model, status


def evaluate_all(
    sample_index_csv: str = "reports/pipeline/sample_index.csv",
    checkpoint_path: Optional[str] = "checkpoints/best_checkpoint.pt",
    d_model: int = 128,
    output_dir: str = "reports/phase2e4",
    use_sequences: bool = True,
    max_history_len: int = 4,
    batch_size: int = 16,
    device: str = "cpu",
    strict_checkpoint: bool = True,
    smoke_test: bool = False,
    max_eval_samples: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Run baseline and model evaluations across Validation and Test partitions.
    """
    df = pd.read_csv(sample_index_csv)
    train_df = df[df["partition"] == "TRAIN"].copy()
    val_df = df[df["partition"] == "VAL"].copy()
    test_df = df[df["partition"] == "TEST"].copy()

    print("=" * 70)
    print("TC-GRAPHFORMER BASELINE & INFERENCE EVALUATION (PHASE 2E.4.1)")
    print(f"Sample Index: {sample_index_csv}")
    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}, Test samples: {len(test_df)}")
    print(f"Evaluation Mode: {'Historical Sequences' if use_sequences else 'Single Overpass'}")
    print(f"Strict Checkpoint Mode: {strict_checkpoint} | Smoke-Test Override: {smoke_test}")
    print("=" * 70)

    # 1. Fit Tabular Linear Models on TRAIN partition ONLY
    train_v_valid = train_df.dropna(subset=["intensity_vmax_kt", "storm_speed_mps"])
    X_v_train = np.column_stack([
        np.ones(len(train_v_valid)),
        train_v_valid["storm_speed_mps"].values,
        train_v_valid["distance_to_land_km"].fillna(500.0).values,
    ])
    y_v_train = train_v_valid["intensity_vmax_kt"].values
    beta_v, _, _, _ = np.linalg.lstsq(X_v_train, y_v_train, rcond=None)

    train_p_valid = train_df.dropna(subset=["central_min_pressure_hpa", "intensity_vmax_kt"])
    X_p_train = np.column_stack([
        np.ones(len(train_p_valid)),
        train_p_valid["intensity_vmax_kt"].values,
    ])
    y_p_train = train_p_valid["central_min_pressure_hpa"].values
    beta_p, _, _, _ = np.linalg.lstsq(X_p_train, y_p_train, rcond=None)

    stage_series = train_df["development_stage_raw"].map(lambda x: STAGE_MAPPING_CANONICAL.get(str(x).strip().upper(), 0))
    majority_stage = int(stage_series.value_counts().idxmax()) if len(stage_series) > 0 else 0

    v_mean_train = float(train_df["intensity_vmax_kt"].dropna().mean()) if len(train_df["intensity_vmax_kt"].dropna()) > 0 else 45.0
    p_mean_train = float(train_df["central_min_pressure_hpa"].dropna().mean()) if len(train_df["central_min_pressure_hpa"].dropna()) > 0 else 1000.0
    ri_mean_train = float(train_df["is_rapid_intensification_24h"].dropna().mean()) if len(train_df["is_rapid_intensification_24h"].dropna()) > 0 else 0.03

    # Climatology mean intensity changes by horizon
    clim_mean_dv = {}
    for h in [6, 12, 18, 24]:
        s = train_df[f"intensity_change_{h}h_kt"].dropna()
        clim_mean_dv[h] = float(s.mean()) if len(s) > 0 else None

    # Track climatology displacement vectors from TRAIN
    train_track_clim = compute_train_track_climatology(sample_index_csv)
    for h, vec in train_track_clim.items():
        if vec is not None:
            print(f"[CLIMATOLOGY] Track {h}h training mean displacement: dx={vec[0]:.2f} km, dy={vec[1]:.2f} km")
        else:
            print(f"[CLIMATOLOGY] Track {h}h training mean displacement unavailable.")

    # 2. Model Loading
    model, ckpt_info = load_model_for_evaluation(
        checkpoint_path=checkpoint_path,
        d_model=d_model,
        strict=strict_checkpoint,
        smoke_test=smoke_test,
        device=device,
    )
    model_label = ckpt_info["model_label"]

    # 3. Partition Evaluations
    partitions = [("VAL", val_df), ("TEST", test_df)]
    baseline_records = []
    horizon_records = []
    basin_records = []
    val_calibrated_ri_threshold = 0.5

    for part_name, part_df in partitions:
        print(f"\n--- Evaluating Partition: {part_name} ({len(part_df)} samples) ---")

        if use_sequences:
            dataset = TCPRIMEDSequenceDataset(
                sample_index_csv=sample_index_csv,
                partition=part_name,
                grid_size=64,
                max_history_len=max_history_len,
                max_sequences=max_eval_samples,
            )
            collate_fn = collate_sequence_batch
        else:
            dataset = TCPRIMEDDataset(
                sample_index_csv=sample_index_csv,
                partition=part_name,
                grid_size=64,
                max_samples=max_eval_samples,
            )
            collate_fn = collate_multimodal_batch

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        v_true_all, v_pred_nn_all, v_mask_all = [], [], []
        p_true_all, p_pred_nn_all, p_mask_all = [], [], []
        s_true_all, s_pred_nn_all, s_mask_all = [], [], []
        dv_true_all, dv_pred_nn_all, dv_mask_all = [], [], []
        ri_true_all, ri_prob_nn_all, ri_mask_all = [], [], []
        tr_true_all, tr_pred_nn_all, tr_mask_all = [], [], []
        kin_anchor_all = []
        sample_ids_all, storm_ids_all = [], []

        with torch.no_grad():
            for b_idx, batch in enumerate(loader):
                if (b_idx + 1) % 10 == 0 or b_idx == len(loader) - 1:
                    print(f"  [{part_name}] Evaluating batch {b_idx + 1}/{len(loader)}...", flush=True)

                ir = batch["ir"].to(device)
                pmw = batch["pmw"].to(device)
                gprof = batch["gprof"].to(device)
                radar = batch["radar"].to(device)
                env = batch["env"].to(device)
                masks = batch["modality_masks"].to(device)

                dt = batch.get("delta_times_hours")
                if dt is not None:
                    dt = dt.to(device)
                pad_m = batch.get("seq_pad_mask")
                if pad_m is not None:
                    pad_m = pad_m.to(device)

                ch_m = {k: v.to(device) for k, v in batch["channel_masks"].items()} if "channel_masks" in batch and batch["channel_masks"] else None
                px_m = {k: v.to(device) for k, v in batch["pixel_masks"].items()} if "pixel_masks" in batch and batch["pixel_masks"] else None
                kin_s = batch.get("kinematic_state")
                extra_eval_kwargs = {}
                if kin_s is not None and hasattr(model, "_encode_kinematic_context"):
                    extra_eval_kwargs["kinematic_state"] = kin_s.to(device)

                anchor_kin_b = batch.get("anchor_kinematic_state")
                if anchor_kin_b is None and kin_s is not None and kin_s.dim() == 2:
                    anchor_kin_b = kin_s
                if anchor_kin_b is not None:
                    kin_anchor_all.extend(anchor_kin_b.cpu().numpy().tolist())
                else:
                    kin_anchor_all.extend([[0.0] * 10 for _ in range(ir.size(0))])

                preds = model(
                    ir=ir,
                    pmw=pmw,
                    gprof=gprof,
                    radar=radar,
                    env=env,
                    modality_masks=masks,
                    delta_times_hours=dt,
                    seq_pad_mask=pad_m,
                    channel_masks=ch_m,
                    pixel_masks=px_m,
                    **extra_eval_kwargs,
                )

                targets = batch["targets"]

                v_true_all.extend(targets["intensity"].cpu().numpy().tolist())
                v_pred_nn_all.extend(preds["intensity"].cpu().numpy().tolist())
                v_mask_all.extend((targets["intensity_mask"].cpu().numpy() > 0.5).tolist())

                p_true_all.extend(targets["pressure"].cpu().numpy().tolist())
                p_pred_nn_all.extend(preds["pressure"].cpu().numpy().tolist())
                p_mask_all.extend((targets["pressure_mask"].cpu().numpy() > 0.5).tolist())

                s_true_all.extend(targets["stage"].cpu().numpy().tolist())
                s_pred_nn_all.extend(preds["stage_logits"].argmax(dim=-1).cpu().numpy().tolist())
                s_mask_all.extend((targets["stage_mask"].cpu().numpy() > 0.5).tolist())

                dv_true_all.extend(targets["intensity_change"].cpu().numpy().tolist())
                dv_pred_nn_all.extend(preds["intensity_change"].cpu().numpy().tolist())
                dv_mask_all.extend((targets["intensity_change_mask"].cpu().numpy() > 0.5).tolist())

                ri_true_all.extend(targets["rapid_intensification"].cpu().numpy().tolist())
                ri_prob_nn_all.extend(torch.sigmoid(preds["ri_logits"]).cpu().numpy().tolist())
                ri_mask_all.extend((targets["ri_mask"].cpu().numpy() > 0.5).tolist())

                tr_true_all.extend(targets["track_displacement"].cpu().numpy().tolist())  # (B, 4, 2)
                tr_pred_nn_all.extend(preds["track_delta"].view(-1, 4, 2).cpu().numpy().tolist())  # (B, 4, 2)
                tr_mask_all.extend((targets["track_mask"].cpu().numpy() > 0.5).tolist())  # (B, 4)

                sample_ids_all.extend(batch["sample_ids"])
                storm_ids_all.extend(batch["storm_ids"])

        # Convert to numpy arrays
        v_true = np.array(v_true_all)
        v_pred_nn = np.array(v_pred_nn_all)
        v_mask = np.array(v_mask_all)

        p_true = np.array(p_true_all)
        p_pred_nn = np.array(p_pred_nn_all)
        p_mask = np.array(p_mask_all)

        s_true = np.array(s_true_all)
        s_pred_nn = np.array(s_pred_nn_all)
        s_mask = np.array(s_mask_all)

        dv_true = np.array(dv_true_all)
        dv_pred_nn = np.array(dv_pred_nn_all)
        dv_mask = np.array(dv_mask_all)

        ri_true = np.array(ri_true_all)
        ri_prob_nn = np.array(ri_prob_nn_all)
        ri_mask = np.array(ri_mask_all)

        tr_true = np.array(tr_true_all)
        tr_pred_nn = np.array(tr_pred_nn_all)
        tr_mask = np.array(tr_mask_all)
        kin_anchor = np.array(kin_anchor_all)

        sub_df = part_df.set_index("sample_id").reindex(sample_ids_all)

        # --- A. Current Intensity (Wind Speed kt) ---
        if np.any(v_mask):
            vt = v_true[v_mask]
            vp_nn = v_pred_nn[v_mask]
            vp_clim = np.full_like(vt, v_mean_train)

            speed_val = sub_df.loc[sub_df.index[v_mask], "storm_speed_mps"].fillna(5.0).values
            dist_val = sub_df.loc[sub_df.index[v_mask], "distance_to_land_km"].fillna(500.0).values
            X_v = np.column_stack([np.ones(len(vt)), speed_val, dist_val])
            vp_lin = X_v @ beta_v

            for m_name, p_arr in [
                ("Climatology (Train Mean)", vp_clim),
                ("Linear Regression", vp_lin),
                (model_label, vp_nn),
            ]:
                err = p_arr - vt
                baseline_records.append({
                    "partition": part_name,
                    "task_category": "current_state_estimation",
                    "task": "Current Intensity (v_max)",
                    "model": m_name,
                    "n_samples": len(vt),
                    "mae": round(float(np.mean(np.abs(err))), 2),
                    "rmse": round(float(np.sqrt(np.mean(err ** 2))), 2),
                    "bias": round(float(np.mean(err)), 2),
                    "units": "kt",
                    "checkpoint_status": ckpt_info["status"],
                })

        # --- B. Central Minimum Pressure (hPa) ---
        if np.any(p_mask):
            pt = p_true[p_mask]
            pp_nn = p_pred_nn[p_mask]
            pp_clim = np.full_like(pt, p_mean_train)

            v_for_p = sub_df.loc[sub_df.index[p_mask], "intensity_vmax_kt"].fillna(v_mean_train).values
            X_p = np.column_stack([np.ones(len(pt)), v_for_p])
            pp_lin = X_p @ beta_p

            for m_name, p_arr in [
                ("Climatology (Train Mean)", pp_clim),
                ("Linear Pressure-Wind Fit", pp_lin),
                (model_label, pp_nn),
            ]:
                err = p_arr - pt
                baseline_records.append({
                    "partition": part_name,
                    "task_category": "current_state_estimation",
                    "task": "Central Pressure (p_min)",
                    "model": m_name,
                    "n_samples": len(pt),
                    "mae": round(float(np.mean(np.abs(err))), 2),
                    "rmse": round(float(np.sqrt(np.mean(err ** 2))), 2),
                    "bias": round(float(np.mean(err)), 2),
                    "units": "hPa",
                    "checkpoint_status": ckpt_info["status"],
                })

        # --- C. Development Stage Classification ---
        if np.any(s_mask):
            st = s_true[s_mask]
            sp_nn = s_pred_nn[s_mask]
            sp_maj = np.full_like(st, majority_stage)

            acc_maj = float(np.mean(sp_maj == st)) * 100.0
            acc_nn = float(np.mean(sp_nn == st)) * 100.0

            baseline_records.append({
                "partition": part_name,
                "task_category": "current_state_estimation",
                "task": "Development Stage Classification",
                "model": "Majority Class (TS)",
                "n_samples": len(st),
                "mae": round(100.0 - acc_maj, 2),
                "rmse": 0.0,
                "bias": 0.0,
                "units": "error_rate_pct",
                "notes": f"Accuracy: {acc_maj:.2f}%",
                "checkpoint_status": "baseline_climatology",
            })
            baseline_records.append({
                "partition": part_name,
                "task_category": "current_state_estimation",
                "task": "Development Stage Classification",
                "model": model_label,
                "n_samples": len(st),
                "mae": round(100.0 - acc_nn, 2),
                "rmse": 0.0,
                "bias": 0.0,
                "units": "error_rate_pct",
                "notes": f"Accuracy: {acc_nn:.2f}%",
                "checkpoint_status": ckpt_info["status"],
            })

        # --- D. Multi-Horizon Intensity Change (kt) ---
        for h_idx, h in enumerate([6, 12, 18, 24]):
            m_h = dv_mask[:, h_idx]
            n_targets = int(np.sum(m_h))

            if n_targets > 0:
                dvt = dv_true[m_h, h_idx]
                dv_persist = np.zeros_like(dvt)
                dv_nn = dv_pred_nn[m_h, h_idx]

                mae_persist = round(float(np.mean(np.abs(dv_persist - dvt))), 2)
                rmse_persist = round(float(np.sqrt(np.mean((dv_persist - dvt) ** 2))), 2)

                if clim_mean_dv[h] is not None:
                    dv_clim = np.full_like(dvt, clim_mean_dv[h])
                    mae_clim = round(float(np.mean(np.abs(dv_clim - dvt))), 2)
                    rmse_clim = round(float(np.sqrt(np.mean((dv_clim - dvt) ** 2))), 2)
                    impr_clim = round(mae_clim - float(np.mean(np.abs(dv_nn - dvt))), 2)
                else:
                    mae_clim, rmse_clim, impr_clim = None, None, None

                mae_nn = round(float(np.mean(np.abs(dv_nn - dvt))), 2)
                rmse_nn = round(float(np.sqrt(np.mean((dv_nn - dvt) ** 2))), 2)
                impr_persist = round(mae_persist - mae_nn, 2)
            else:
                mae_persist, rmse_persist, impr_persist = None, None, None
                mae_clim, rmse_clim, impr_clim = None, None, None
                mae_nn, rmse_nn = None, None

            horizon_records.append({
                "partition": part_name,
                "task_category": "future_forecasting",
                "horizon_hours": f"Intensity_{h}h",
                "n_valid_targets": n_targets,
                "persistence_mae": mae_persist,
                "persistence_rmse": rmse_persist,
                "persistence_median_error": round(float(np.median(np.abs(dv_persist - dvt))), 2) if n_targets > 0 else None,
                "climatology_mae": mae_clim,
                "climatology_rmse": rmse_clim,
                "velocity_baseline_mae": None,
                "velocity_baseline_rmse": None,
                "tc_graphformer_mae": mae_nn,
                "tc_graphformer_rmse": rmse_nn,
                "improvement_over_persist": impr_persist,
                "improvement_over_clim": impr_clim,
                "improvement_over_velocity": None,
                "units": "kt",
                "checkpoint_status": ckpt_info["status"],
            })

        # --- E. Rapid Intensification (24h RI >= 30 kt) ---
        if np.any(ri_mask):
            rit = ri_true[ri_mask]
            rip_nn = ri_prob_nn[ri_mask]
            rip_clim = np.full_like(rit, ri_mean_train)

            pr_auc_nn, brier_nn, prec_50, rec_50 = compute_pr_auc_and_brier(rit, rip_nn)
            pr_auc_clim, brier_clim, clim_p50, clim_r50 = compute_pr_auc_and_brier(rit, rip_clim)

            n_pos = int(np.sum(rit == 1.0))
            n_neg = int(np.sum(rit == 0.0))

            # Calibrate threshold strictly on VAL only; apply frozen threshold to TEST
            if part_name == "VAL" and n_pos > 0:
                best_f1_val = -1.0
                best_th_val = 0.5
                for th_cand in np.linspace(0.02, 0.90, 89):
                    pr_b = (rip_nn >= th_cand).astype(float)
                    tp_c = float(np.sum((pr_b == 1.0) & (rit == 1.0)))
                    fp_c = float(np.sum((pr_b == 1.0) & (rit == 0.0)))
                    fn_c = float(np.sum((pr_b == 0.0) & (rit == 1.0)))
                    p_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
                    r_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
                    f1_c = (2 * p_c * r_c) / (p_c + r_c) if (p_c + r_c) > 0 else 0.0
                    if f1_c > best_f1_val:
                        best_f1_val = f1_c
                        best_th_val = float(round(th_cand, 2))
                val_calibrated_ri_threshold = best_th_val

            pred_cal = (rip_nn >= val_calibrated_ri_threshold).astype(float)
            tp_cal = float(np.sum((pred_cal == 1.0) & (rit == 1.0)))
            fp_cal = float(np.sum((pred_cal == 1.0) & (rit == 0.0)))
            fn_cal = float(np.sum((pred_cal == 0.0) & (rit == 1.0)))
            prec_cal = tp_cal / (tp_cal + fp_cal) if (tp_cal + fp_cal) > 0 else 0.0
            rec_cal = tp_cal / (tp_cal + fn_cal) if (tp_cal + fn_cal) > 0 else 0.0
            f1_cal = (2 * prec_cal * rec_cal) / (prec_cal + rec_cal) if (prec_cal + rec_cal) > 0 else 0.0
            uncertainty_note = f" [LOW_POS_UNCERTAINTY: n_pos={n_pos}<5]" if n_pos < 5 else ""

            baseline_records.append({
                "partition": part_name,
                "task_category": "future_forecasting",
                "task": "Rapid Intensification (24h >= 30kt)",
                "model": "Training Climatology (Base Rate)",
                "n_samples": len(rit),
                "mae": round(brier_clim, 4),
                "rmse": round(float(np.sqrt(brier_clim)), 4),
                "bias": 0.0,
                "units": "brier_score",
                "notes": f"PR-AUC: {pr_auc_clim:.4f}, Base Rate: {ri_mean_train:.3f}, Pos: {n_pos}, Neg: {n_neg}{uncertainty_note}",
                "checkpoint_status": "baseline_climatology",
            })
            baseline_records.append({
                "partition": part_name,
                "task_category": "future_forecasting",
                "task": "Rapid Intensification (24h >= 30kt)",
                "model": model_label,
                "n_samples": len(rit),
                "mae": round(brier_nn, 4),
                "rmse": round(float(np.sqrt(brier_nn)), 4),
                "bias": 0.0,
                "units": "brier_score",
                "notes": (
                    f"PR-AUC: {pr_auc_nn:.4f}, F1@{val_calibrated_ri_threshold:.2f}: {f1_cal:.4f} "
                    f"(Prec: {prec_cal:.4f}, Rec: {rec_cal:.4f}), Pos: {n_pos}, Neg: {n_neg}{uncertainty_note}"
                ),
                "checkpoint_status": ckpt_info["status"],
            })

        # --- F. Future Track Displacement Error (km) ---
        for h_idx, h in enumerate([6, 12, 18, 24]):
            m_tr = tr_mask[:, h_idx]
            n_valid_tr = int(np.sum(m_tr))

            if n_valid_tr > 0:
                true_vec = tr_true[m_tr, h_idx]  # (M, 2) in km
                pred_vec = tr_pred_nn[m_tr, h_idx]  # (M, 2) in km
                c_vec = train_track_clim.get(h)
                # Observation-time strictly causal velocity extrapolation baseline: (u_kmh * h, v_kmh * h)
                # Falls back to training climatology vector when causal prior fix is unavailable (has_kin == 0)
                kin_sub = kin_anchor[m_tr]
                has_k = (kin_sub[:, 9:10] > 0.5).astype(np.float64)
                raw_vel_vec = np.column_stack([kin_sub[:, 0] * float(h), kin_sub[:, 1] * float(h)])
                if c_vec is not None:
                    c_arr = np.array([[c_vec[0], c_vec[1]]], dtype=np.float64)
                    vel_vec = has_k * raw_vel_vec + (1.0 - has_k) * c_arr
                else:
                    vel_vec = raw_vel_vec
                track_errs = compute_track_errors(true_vec, pred_vec, clim_vec=c_vec, vel_vec=vel_vec)
                persist_mae = track_errs["persist_mae"]
                persist_rmse = track_errs["persist_rmse"]
                persist_median = track_errs["persist_median"]
                clim_mae = track_errs["clim_mae"]
                clim_rmse = track_errs["clim_rmse"]
                vel_mae = track_errs["vel_mae"]
                vel_rmse = track_errs["vel_rmse"]
                nn_mae = track_errs["nn_mae"]
                nn_rmse = track_errs["nn_rmse"]
                impr_persist = track_errs["impr_persist"]
                impr_clim = track_errs["impr_clim"]
                impr_vel = track_errs["impr_vel"]
            else:
                persist_mae, persist_rmse, persist_median = None, None, None
                clim_mae, clim_rmse = None, None
                vel_mae, vel_rmse = None, None
                nn_mae, nn_rmse = None, None
                impr_persist, impr_clim, impr_vel = None, None, None

            horizon_records.append({
                "partition": part_name,
                "task_category": "future_forecasting",
                "horizon_hours": f"Track_{h}h",
                "n_valid_targets": n_valid_tr,
                "persistence_mae": persist_mae,
                "persistence_rmse": persist_rmse,
                "persistence_median_error": persist_median,
                "climatology_mae": clim_mae,
                "climatology_rmse": clim_rmse,
                "velocity_baseline_mae": vel_mae,
                "velocity_baseline_rmse": vel_rmse,
                "tc_graphformer_mae": nn_mae,
                "tc_graphformer_rmse": nn_rmse,
                "improvement_over_persist": impr_persist,
                "improvement_over_clim": impr_clim,
                "improvement_over_velocity": impr_vel,
                "units": "km",
                "checkpoint_status": ckpt_info["status"],
            })

        # --- G. Basin Breakdown ---
        if np.any(v_mask):
            vt_all = v_true[v_mask]
            vp_all = v_pred_nn[v_mask]
            basins_all = sub_df.loc[sub_df.index[v_mask], "basin"].fillna("UNKNOWN").values

            for b in np.unique(basins_all):
                b_mask = (basins_all == b)
                if np.sum(b_mask) > 0:
                    b_err = np.abs(vp_all[b_mask] - vt_all[b_mask])
                    b_mae = float(np.mean(b_err))
                    b_mean_v = float(np.mean(vt_all[b_mask]))
                else:
                    b_mae, b_mean_v = 0.0, 0.0

                basin_records.append({
                    "partition": part_name,
                    "basin": b,
                    "valid_intensity_targets": int(np.sum(b_mask)),
                    "tc_graphformer_wind_mae_kt": round(b_mae, 2),
                    "mean_wind_kt": round(b_mean_v, 2),
                    "checkpoint_status": ckpt_info["status"],
                })

    # Save outputs
    df_base = pd.DataFrame(baseline_records)
    df_horiz = pd.DataFrame(horizon_records)
    df_basin = pd.DataFrame(basin_records)

    out_base = os.path.join(output_dir, "baseline_comparison.csv")
    out_horiz = os.path.join(output_dir, "task_metrics_by_horizon.csv")
    out_basin = os.path.join(output_dir, "basin_evaluation.csv")
    out_summary = os.path.join(output_dir, "evaluation_summary.json")

    os.makedirs(output_dir, exist_ok=True)
    df_base.to_csv(out_base, index=False)
    df_horiz.to_csv(out_horiz, index=False)
    df_basin.to_csv(out_basin, index=False)

    summary_dict = {
        "evaluation_timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "sample_index": sample_index_csv,
        "checkpoint": ckpt_info,
        "use_sequences": use_sequences,
        "max_history_len": max_history_len,
        "train_track_climatology": {str(k): list(v) if v is not None else None for k, v in train_track_clim.items()},
        "val_calibrated_ri_threshold": val_calibrated_ri_threshold,
        "partitions_evaluated": [p[0] for p in partitions],
        "total_baseline_records": len(baseline_records),
        "total_horizon_records": len(horizon_records),
        "total_basin_records": len(basin_records),
    }
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)

    print(f"\n[OK] Baseline Comparison saved to: {out_base}")
    print(f"[OK] Task Metrics by Horizon saved to: {out_horiz}")
    print(f"[OK] Basin Evaluation saved to: {out_basin}")
    print(f"[OK] Evaluation Summary saved to: {out_summary}")

    return df_base, df_horiz, df_basin, summary_dict


def build_parser():
    """Build argument parser for baseline evaluation CLI."""
    parser = argparse.ArgumentParser(description="TC-GraphFormer Baseline & Inference Evaluation Engine")
    parser.add_argument("--config", default=None, help="Path to GPU node or pipeline YAML configuration")
    parser.add_argument("--sample-index", default=None, help="Sample index CSV")
    parser.add_argument("--checkpoint", default=None, help="Path to model checkpoint")
    parser.add_argument("--d-model", type=int, default=None, help="Model hidden dimension")
    parser.add_argument("--output-dir", default=None, help="Output directory for reports")
    parser.add_argument("--single-obs", dest="use_sequences", action="store_false", default=None, help="Use single-observation mode instead of sequences")
    parser.add_argument("--use-sequences", dest="use_sequences", action="store_true", default=None, help="Use sequences mode")
    parser.add_argument("--max-history-len", type=int, default=None, help="Historical sequence length")
    parser.add_argument("--batch-size", type=int, default=None, help="Evaluation batch size")
    parser.add_argument("--device", default=None, help="Device to evaluate on ('cpu' or 'cuda')")
    parser.add_argument("--strict-checkpoint", dest="strict_checkpoint", action="store_true", default=None, help="Enforce trained-checkpoint validation")
    parser.add_argument("--no-strict-checkpoint", dest="strict_checkpoint", action="store_false", help="Disable strict checkpoint check")
    parser.add_argument("--smoke-test", dest="smoke_test", action="store_true", default=None, help="Explicitly permit forward-pass smoke test with untrained or partial checkpoints")
    parser.add_argument("--no-smoke-test", dest="smoke_test", action="store_false", help="Disable smoke test")
    parser.add_argument("--max-eval-samples", type=int, default=None, help="Limit samples for fast smoke testing")
    return parser


def resolve_evaluation_config(args, cfg=None):
    """
    Resolve evaluation parameters across CLI explicit args, YAML configuration, and defaults.
    Priority: CLI explicit flag > YAML config > fallback default.
    """
    if cfg is None:
        cfg = {}
        config_path = getattr(args, "config", None)
        if config_path:
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}

    paths_cfg = cfg.get("paths", {})
    eval_cfg = cfg.get("evaluation", {})
    mach_cfg = cfg.get("machine", {})
    mod_cfg = cfg.get("model", {})
    tr_cfg = cfg.get("training", {})

    def resolve_val(cli_val, yaml_val, default):
        if cli_val is not None:
            return cli_val
        if yaml_val is not None:
            return yaml_val
        return default

    resolved_sample_index = resolve_val(getattr(args, "sample_index", None), paths_cfg.get("sample_index_path"), "reports/pipeline/sample_index.csv")
    yaml_ckpt = os.path.join(paths_cfg["checkpoints_dir"], "best_checkpoint.pt") if "checkpoints_dir" in paths_cfg else None
    resolved_checkpoint = resolve_val(getattr(args, "checkpoint", None), yaml_ckpt, "checkpoints/best_checkpoint.pt")
    resolved_output_dir = resolve_val(getattr(args, "output_dir", None), eval_cfg.get("output_dir") or paths_cfg.get("evaluation_output_dir"), "reports/phase2e4")
    resolved_device = resolve_val(getattr(args, "device", None), mach_cfg.get("device"), "cpu")
    resolved_d_model = resolve_val(getattr(args, "d_model", None), mod_cfg.get("d_model"), 128)
    resolved_batch_size = resolve_val(getattr(args, "batch_size", None), eval_cfg.get("batch_size") or tr_cfg.get("batch_size"), 16)
    resolved_max_hist_len = resolve_val(getattr(args, "max_history_len", None), eval_cfg.get("max_history_len") or tr_cfg.get("max_history_len"), 4)

    yaml_use_seq = eval_cfg.get("use_sequences")
    if yaml_use_seq is None:
        yaml_use_seq = tr_cfg.get("use_sequences")
    resolved_use_sequences = resolve_val(getattr(args, "use_sequences", None), yaml_use_seq, True)

    resolved_strict_checkpoint = resolve_val(getattr(args, "strict_checkpoint", None), eval_cfg.get("strict_checkpoint"), True)
    resolved_smoke_test = resolve_val(getattr(args, "smoke_test", None), eval_cfg.get("smoke_test"), False)
    resolved_max_eval_samples = resolve_val(getattr(args, "max_eval_samples", None), eval_cfg.get("max_eval_samples"), None)

    return {
        "sample_index": resolved_sample_index,
        "checkpoint": resolved_checkpoint,
        "output_dir": resolved_output_dir,
        "device": resolved_device,
        "d_model": resolved_d_model,
        "batch_size": resolved_batch_size,
        "max_history_len": resolved_max_hist_len,
        "use_sequences": resolved_use_sequences,
        "strict_checkpoint": resolved_strict_checkpoint,
        "smoke_test": resolved_smoke_test,
        "max_eval_samples": resolved_max_eval_samples,
    }


def main():
    parser = build_parser()
    args = parser.parse_args()
    resolved = resolve_evaluation_config(args)

    evaluate_all(
        sample_index_csv=resolved["sample_index"],
        checkpoint_path=resolved["checkpoint"],
        d_model=resolved["d_model"],
        output_dir=resolved["output_dir"],
        use_sequences=resolved["use_sequences"],
        max_history_len=resolved["max_history_len"],
        batch_size=resolved["batch_size"],
        device=resolved["device"],
        strict_checkpoint=resolved["strict_checkpoint"],
        smoke_test=resolved["smoke_test"],
        max_eval_samples=resolved["max_eval_samples"],
    )


if __name__ == "__main__":
    main()
