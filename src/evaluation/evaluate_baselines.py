"""
Comprehensive Baseline and Model Evaluation Engine for TC-GraphFormer (Phase 2E.4).
Evaluates:
1. Climatology and Persistence Baselines (Wind, Pressure, Stage, Intensity Change, RI Base Rate, Track)
2. Tabular Linear Baselines (Fitted strictly on TRAIN partition)
3. TC-GraphFormer Neural Model (in historical sequence or single-obs mode)

Outputs:
- reports/phase2e/baseline_comparison.csv
- reports/phase2e/task_metrics_by_horizon.csv
- reports/phase2e/basin_evaluation.csv
- reports/phase2e/evaluation_summary.json

Enforces:
- Geodesic / Euclidean track displacement errors in kilometers (km)
- Intensity errors in knots (kt)
- Pressure errors in hectopascals (hPa)
- Strict checkpoint validation: explicitly marks whether weights are trained or random smoke-test
- Masked loss and target handling for all prediction horizons
"""

import os
import sys
import math
import json
import argparse
from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.dataset import TCPRIMEDDataset, TCPRIMEDSequenceDataset, STAGE_MAPPING_CANONICAL
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


def load_model_for_evaluation(
    checkpoint_path: Optional[str] = None,
    d_model: int = 128,
    strict: bool = False,
    device: str = "cpu",
) -> Tuple[TCGraphFormer, Dict[str, Any]]:
    """
    Load TCGraphFormer model and validate checkpoint compatibility.
    Never silently assumes an untrained model is trained.
    """
    model = TCGraphFormer(d_model=d_model, num_stages=5).to(device)
    model.eval()

    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        status = {
            "is_trained": False,
            "checkpoint_path": checkpoint_path,
            "status": "untrained_random_init",
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

    if strict and (len(missing) > 0 or len(mismatches) > 0):
        raise ValueError(
            f"Strict checkpoint validation failed for {checkpoint_path}! "
            f"Missing keys: {len(missing)}, Shape mismatches: {len(mismatches)}"
        )

    model.load_state_dict(matched, strict=False)
    is_trained = (len(matched) == len(model_keys))

    status = {
        "is_trained": is_trained,
        "checkpoint_path": checkpoint_path,
        "status": "trained_checkpoint" if is_trained else "partially_matched_checkpoint",
        "matched_keys": len(matched),
        "total_keys": len(model_keys),
        "missing_keys_count": len(missing),
        "shape_mismatches_count": len(mismatches),
        "epoch": ckpt.get("epoch", None),
        "metrics": ckpt.get("metrics", {}),
    }
    print(f"[EVAL] Checkpoint '{checkpoint_path}' loaded: {len(matched)}/{len(model_keys)} parameters matched. Status: {status['status']}")
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
    strict_checkpoint: bool = False,
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
    print("TC-GRAPHFORMER BASELINE & INFERENCE EVALUATION (PHASE 2E.4)")
    print(f"Sample Index: {sample_index_csv}")
    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}, Test samples: {len(test_df)}")
    print(f"Evaluation Mode: {'Historical Sequences' if use_sequences else 'Single Overpass'}")
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
        clim_mean_dv[h] = float(s.mean()) if len(s) > 0 else 0.0

    # 2. Model Loading
    model, ckpt_info = load_model_for_evaluation(
        checkpoint_path=checkpoint_path,
        d_model=d_model,
        strict=strict_checkpoint,
        device=device,
    )
    is_trained = ckpt_info["is_trained"]

    # 3. Partition Evaluations
    partitions = [("VAL", val_df), ("TEST", test_df)]
    baseline_records = []
    horizon_records = []
    basin_records = []

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

        # Batch accumulators
        v_true_all, v_pred_nn_all, v_mask_all = [], [], []
        p_true_all, p_pred_nn_all, p_mask_all = [], [], []
        s_true_all, s_pred_nn_all, s_mask_all = [], [], []
        dv_true_all, dv_pred_nn_all, dv_mask_all = [], [], []
        ri_true_all, ri_prob_nn_all, ri_mask_all = [], [], []
        tr_true_all, tr_pred_nn_all, tr_mask_all = [], [], []
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
                )

                targets = batch["targets"]

                # Extract and accumulate
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

        model_label = "TC-GraphFormer" if is_trained else "TC-GraphFormer (Untrained Smoke-Test)"

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

        dv_true = np.array(dv_true_all)  # (N, 4)
        dv_pred_nn = np.array(dv_pred_nn_all)  # (N, 4)
        dv_mask = np.array(dv_mask_all)  # (N, 4)

        ri_true = np.array(ri_true_all)
        ri_prob_nn = np.array(ri_prob_nn_all)
        ri_mask = np.array(ri_mask_all)

        tr_true = np.array(tr_true_all)  # (N, 4, 2)
        tr_pred_nn = np.array(tr_pred_nn_all)  # (N, 4, 2)
        tr_mask = np.array(tr_mask_all)  # (N, 4)

        # Align with metadata from part_df for linear baseline evaluation
        sub_df = part_df.set_index("sample_id").reindex(sample_ids_all)

        # --- A. Current Intensity (Wind Speed kt) ---
        if np.any(v_mask):
            vt = v_true[v_mask]
            vp_nn = v_pred_nn[v_mask]
            vp_clim = np.full_like(vt, v_mean_train)

            # Tabular linear fit
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
                "task": "Development Stage Classification",
                "model": "Majority Class (TS)",
                "n_samples": len(st),
                "mae": round(100.0 - acc_maj, 2),  # error rate
                "rmse": 0.0,
                "bias": 0.0,
                "units": "error_rate_pct",
                "notes": f"Accuracy: {acc_maj:.2f}%",
                "checkpoint_status": "baseline_climatology",
            })
            baseline_records.append({
                "partition": part_name,
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
                dv_clim = np.full_like(dvt, clim_mean_dv[h])
                dv_nn = dv_pred_nn[m_h, h_idx]

                mae_persist = float(np.mean(np.abs(dv_persist - dvt)))
                rmse_persist = float(np.sqrt(np.mean((dv_persist - dvt) ** 2)))

                mae_clim = float(np.mean(np.abs(dv_clim - dvt)))
                rmse_clim = float(np.sqrt(np.mean((dv_clim - dvt) ** 2)))

                mae_nn = float(np.mean(np.abs(dv_nn - dvt)))
                rmse_nn = float(np.sqrt(np.mean((dv_nn - dvt) ** 2)))
            else:
                mae_persist, rmse_persist = 0.0, 0.0
                mae_clim, rmse_clim = 0.0, 0.0
                mae_nn, rmse_nn = 0.0, 0.0

            horizon_records.append({
                "partition": part_name,
                "horizon_hours": f"Intensity_{h}h",
                "n_valid_targets": n_targets,
                "persistence_mae": round(mae_persist, 2),
                "persistence_rmse": round(rmse_persist, 2),
                "climatology_mae": round(mae_clim, 2),
                "climatology_rmse": round(rmse_clim, 2),
                "tc_graphformer_mae": round(mae_nn, 2),
                "tc_graphformer_rmse": round(rmse_nn, 2),
                "improvement_over_persist": round(mae_persist - mae_nn, 2),
                "improvement_over_clim": round(mae_clim - mae_nn, 2),
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

            baseline_records.append({
                "partition": part_name,
                "task": "Rapid Intensification (24h >= 30kt)",
                "model": "Training Climatology (Base Rate)",
                "n_samples": len(rit),
                "mae": round(brier_clim, 4),  # Brier score
                "rmse": round(float(np.sqrt(brier_clim)), 4),
                "bias": 0.0,
                "units": "brier_score",
                "notes": f"PR-AUC: {pr_auc_clim:.4f}, Base Rate: {ri_mean_train:.3f}",
                "checkpoint_status": "baseline_climatology",
            })
            baseline_records.append({
                "partition": part_name,
                "task": "Rapid Intensification (24h >= 30kt)",
                "model": model_label,
                "n_samples": len(rit),
                "mae": round(brier_nn, 4),
                "rmse": round(float(np.sqrt(brier_nn)), 4),
                "bias": 0.0,
                "units": "brier_score",
                "notes": f"PR-AUC: {pr_auc_nn:.4f}, Prec@0.5: {prec_50:.4f}, Rec@0.5: {rec_50:.4f}",
                "checkpoint_status": ckpt_info["status"],
            })

        # --- F. Future Track Displacement Error (km) ---
        for h_idx, h in enumerate([6, 12, 18, 24]):
            m_tr = tr_mask[:, h_idx]
            n_valid_tr = int(np.sum(m_tr))

            if n_valid_tr > 0:
                true_vec = tr_true[m_tr, h_idx]  # (M, 2) in km
                pred_vec = tr_pred_nn[m_tr, h_idx]  # (M, 2) in km

                # Persistence error: storm assumes zero displacement [0, 0]
                persist_err_km = np.sqrt(true_vec[:, 0] ** 2 + true_vec[:, 1] ** 2)
                persist_mean_km = float(np.mean(persist_err_km))
                persist_median_km = float(np.median(persist_err_km))

                # TC-GraphFormer displacement vector error
                diff_vec = pred_vec - true_vec
                nn_err_km = np.sqrt(diff_vec[:, 0] ** 2 + diff_vec[:, 1] ** 2)
                nn_mean_km = float(np.mean(nn_err_km))
                nn_rmse_km = float(np.sqrt(np.mean(nn_err_km ** 2)))
            else:
                persist_mean_km, persist_median_km = 0.0, 0.0
                nn_mean_km, nn_rmse_km = 0.0, 0.0

            horizon_records.append({
                "partition": part_name,
                "horizon_hours": f"Track_{h}h",
                "n_valid_targets": n_valid_tr,
                "persistence_mae": round(persist_mean_km, 2),
                "persistence_rmse": round(persist_median_km, 2),
                "climatology_mae": 0.0,
                "climatology_rmse": 0.0,
                "tc_graphformer_mae": round(nn_mean_km, 2),
                "tc_graphformer_rmse": round(nn_rmse_km, 2),
                "improvement_over_persist": round(persist_mean_km - nn_mean_km, 2),
                "improvement_over_clim": 0.0,
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


def main():
    parser = argparse.ArgumentParser(description="TC-GraphFormer Baseline & Inference Evaluation Engine")
    parser.add_argument("--sample-index", default="reports/pipeline/sample_index.csv", help="Sample index CSV")
    parser.add_argument("--checkpoint", default="checkpoints/best_checkpoint.pt", help="Path to model checkpoint")
    parser.add_argument("--d-model", type=int, default=128, help="Model hidden dimension")
    parser.add_argument("--output-dir", default="reports/phase2e4", help="Output directory for reports")
    parser.add_argument("--single-obs", action="store_true", help="Use single-observation mode instead of sequences")
    parser.add_argument("--max-history-len", type=int, default=4, help="Historical sequence length")
    parser.add_argument("--batch-size", type=int, default=16, help="Evaluation batch size")
    parser.add_argument("--device", default="cpu", help="Device to evaluate on ('cpu' or 'cuda')")
    parser.add_argument("--strict-checkpoint", action="store_true", help="Raise error if checkpoint parameters do not match 100%")
    parser.add_argument("--max-eval-samples", type=int, default=None, help="Limit samples for fast smoke testing")
    args = parser.parse_args()

    evaluate_all(
        sample_index_csv=args.sample_index,
        checkpoint_path=args.checkpoint,
        d_model=args.d_model,
        output_dir=args.output_dir,
        use_sequences=not args.single_obs,
        max_history_len=args.max_history_len,
        batch_size=args.batch_size,
        device=args.device,
        strict_checkpoint=args.strict_checkpoint,
        max_eval_samples=args.max_eval_samples,
    )


if __name__ == "__main__":
    main()
