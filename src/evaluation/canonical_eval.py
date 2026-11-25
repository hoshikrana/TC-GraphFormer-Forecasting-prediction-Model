"""
Canonical Single-Path Evaluation & Storm-Cluster Bootstrap Engine for TC-GraphFormer.

Provides:
1. `evaluate(checkpoint_path, split="val", use_ema=False)`:
   A single, deterministic evaluation code path using identical valid target masks,
   strictly prohibited from accessing FINAL_TEST (`split="test"` is blocked),
   returning:
   - Global V1-reference loss (`val_loss_global`, plus per-task raw & weighted V1 losses)
   - Wind MAE and bias (kt)
   - Major-hurricane MAE and bias (>= 96 kt)
   - Central pressure MAE (hPa)
   - Balanced stage accuracy (%) and overall stage accuracy (%)
   - Multi-horizon intensity-change (dV) MAE at 6/12/18/24 h (kt)
   - Rapid intensification (RI) PR-AUC, Brier score, and positive count
   - +24h track error (km) and skill vs. the strictly causal observation-time velocity baseline
   - Per-storm breakdown tables
   - Raw per-observation prediction arrays suitable for `.npz` serialization and paired storm bootstrap

2. `storm_bootstrap(preds_a, preds_b, n=2000, seed=0)`:
   Cluster bootstrap that resamples STORMS (not observations) with replacement,
   uses the exact same resampled storms for both models `preds_a` and `preds_b`,
   and explicitly handles RI resamples with zero positive cases by dropping them
   and counting `dropped_zero_pos_resamples`. Returns paired deltas (`preds_a - preds_b`)
   with 95% confidence intervals.
"""

import os
import sys
import math
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.dataset import TCPRIMEDSequenceDataset
from src.data.collate import collate_sequence_batch
from src.models.tc_graphformer import TCGraphFormer
from src.training.losses import MultiTaskLoss


# Frozen TRAIN-partition 24h track climatology fallback vector (dx_east_km, dy_north_km)
# used by the strictly causal velocity baseline when an observation has no prior fix (has_kin == 0).
# Verified from reports/phase2e4_100gb/evaluation_summary.json (`compute_train_track_climatology`).
TRAIN_TRACK_CLIMATOLOGY_KM: Dict[int, Tuple[float, float]] = {
    6: (-38.05391880880047, 28.564956702040792),
    12: (-77.01912847261264, 56.40528355872715),
    18: (-112.50772742636995, 85.07030046999444),
    24: (-154.43646259182725, 112.17986236385286),
}

DEFAULT_SAMPLE_INDEX_100GB = "reports/pipeline_100gb/sample_index.csv"

CANONICAL_CHECKPOINTS: Dict[str, str] = {
    "Original_Ep5": "checkpoints/tcprimed_100gb/best_checkpoint.pt",
    "C1_Ep8": "checkpoints/tcprimed_100gb_exp_c1/best_checkpoint.pt",
    "T6_Ep6": "checkpoints/tcprimed_100gb_hpo_screen/T6_lr5.0e-5_wd1e-3_decoupled/screen_epoch_006.pt",
    "S24_Ep7": "checkpoints/tcprimed_100gb_exp_s24/best_mt_rel_err_checkpoint.pt",
    "S24_Ep10": "checkpoints/tcprimed_100gb_exp_s24/best_checkpoint.pt",
}

_DATASET_LOADER_CACHE: Dict[Tuple[str, str, int], Tuple[TCPRIMEDSequenceDataset, DataLoader]] = {}


def _compute_canonical_pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Exact step-wise average precision (PR-AUC) matching TCGraphFormerTrainer.evaluate:
    sorts descending by predicted probability with stable mergesort and averages precision
    at every positive rank.
    """
    pos_mask = (y_true == 1.0)
    n_pos = int(np.sum(pos_mask))
    if n_pos == 0 or len(y_true) == 0:
        return float("nan")
    order = np.argsort(-y_prob, kind="mergesort")
    y_sorted = y_true[order]
    tps = np.cumsum(y_sorted == 1.0)
    fps = np.cumsum(y_sorted == 0.0)
    prec_curve = tps / np.maximum(1.0, tps + fps)
    return float(np.sum(prec_curve[y_sorted == 1.0]) / float(n_pos))


def _compute_balanced_stage_acc_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute macro-averaged recall (balanced accuracy %) across all present true classes."""
    if len(y_true) == 0:
        return float("nan")
    classes = np.unique(y_true)
    recalls = []
    for c in classes:
        m_c = (y_true == c)
        if np.sum(m_c) > 0:
            recalls.append(float(np.mean(y_pred[m_c] == c)))
    return float(np.mean(recalls) * 100.0) if recalls else float("nan")


def _get_cached_dataloader(
    sample_index_csv: str,
    partition: str,
    batch_size: int = 32,
) -> Tuple[TCPRIMEDSequenceDataset, DataLoader]:
    key = (os.path.abspath(sample_index_csv), partition, int(batch_size))
    if key not in _DATASET_LOADER_CACHE:
        dataset = TCPRIMEDSequenceDataset(
            sample_index_csv=sample_index_csv,
            partition=partition,
            grid_size=64,
            max_history_len=4,
            max_history_hours=72.0,
            pmw_schema_version="v2.3",
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_sequence_batch,
            pin_memory=torch.cuda.is_available(),
        )
        _DATASET_LOADER_CACHE[key] = (dataset, loader)
    return _DATASET_LOADER_CACHE[key]


def evaluate(
    checkpoint_path: Union[str, Path],
    split: str = "val",
    use_ema: bool = False,
) -> Dict[str, Any]:
    """
    Canonical single-code-path evaluation for TC-GraphFormer checkpoints.

    Args:
        checkpoint_path: Path to saved PyTorch checkpoint (.pt).
        split: Dataset partition ("val" or "train"). "test" / "final_test" is strictly blocked.
        use_ema: If True, load exponential moving average weights (`ema_state_dict` / `model_ema_state_dict`)
                 when present in the checkpoint (raises ValueError if requested but absent).

    Returns:
        Dictionary containing canonical metrics, per-storm table, and raw prediction arrays.
    """
    part_upper = str(split).strip().upper()
    if part_upper in ("TEST", "FINAL_TEST"):
        raise PermissionError(
            "Strictly forbidden: FINAL_TEST ('test') must NEVER be accessed during development or canonical VAL benchmarking."
        )
    if part_upper not in ("VAL", "TRAIN"):
        raise ValueError(f"Unsupported split '{split}'; expected 'val' or 'train'.")

    ckpt_path_str = str(checkpoint_path)
    if not os.path.exists(ckpt_path_str):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path_str}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path_str, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Expected dict checkpoint at {ckpt_path_str}, got {type(ckpt)}")

    d_model = int(ckpt.get("d_model", 128))
    model = TCGraphFormer(d_model=d_model, num_stages=5, head_dropout=0.0).to(device)

    if use_ema:
        ema_state = ckpt.get("ema_state_dict") or ckpt.get("model_ema_state_dict") or ckpt.get("ema_model_state_dict")
        if ema_state is None:
            raise ValueError(f"use_ema=True requested, but checkpoint '{ckpt_path_str}' does not contain EMA weights.")
        model.load_state_dict(ema_state, strict=True)
    else:
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict, strict=True)

    model.eval()

    # Canonical V1-Reference MultiTaskLoss
    loss_fn = MultiTaskLoss(
        weight_intensity=1.0,
        weight_pressure=0.5,
        weight_stage=0.5,
        weight_intensity_change=1.0,
        weight_ri=2.0,
        weight_track=1.0,
        ri_pos_weight=18.1,
        regime_max_extra_weight=2.5,
        regime_power=2.0,
        ri_prior_shift_factor=0.85,
        intensity_huber_beta=8.0,
        intensity_std_scale=1.0,
    ).to(device)

    sample_index_csv = DEFAULT_SAMPLE_INDEX_100GB
    if not os.path.exists(sample_index_csv):
        repo_root = Path(__file__).resolve().parents[2]
        sample_index_csv = str(repo_root / DEFAULT_SAMPLE_INDEX_100GB)

    _, loader = _get_cached_dataloader(sample_index_csv=sample_index_csv, partition=part_upper, batch_size=32)

    unreduced_sums = {
        "intensity": 0.0,
        "pressure": 0.0,
        "stage": 0.0,
        "intensity_change": 0.0,
        "ri": 0.0,
        "track": 0.0,
    }
    valid_counts = {
        "intensity": 0,
        "pressure": 0,
        "stage": 0,
        "intensity_change": 0,
        "ri": 0,
        "track": 0,
    }
    batch_loss_sum = 0.0
    total_samples = 0

    sample_ids_all: List[str] = []
    storm_ids_all: List[str] = []
    timestamps_all: List[int] = []

    v_pred_all, v_true_all, v_mask_all = [], [], []
    p_pred_all, p_true_all, p_mask_all = [], [], []
    s_pred_all, s_true_all, s_mask_all = [], [], []
    dv_pred_all, dv_true_all, dv_mask_all = [], [], []
    ri_prob_all, ri_logit_all, ri_true_all, ri_mask_all = [], [], [], []
    tr_pred_all, tr_true_all, tr_mask_all = [], [], []
    kin_anchor_all = []

    with torch.no_grad():
        for batch in loader:
            ir = batch["ir"].to(device)
            pmw = batch["pmw"].to(device)
            gprof = batch["gprof"].to(device)
            radar = batch["radar"].to(device)
            env = batch["env"].to(device)
            masks = batch["modality_masks"].to(device)
            dt = batch["delta_times_hours"].to(device) if batch.get("delta_times_hours") is not None else None
            pad_m = batch["seq_pad_mask"].to(device) if batch.get("seq_pad_mask") is not None else None
            ch_m = {k: v.to(device) for k, v in batch["channel_masks"].items()} if batch.get("channel_masks") else None
            px_m = {k: v.to(device) for k, v in batch["pixel_masks"].items()} if batch.get("pixel_masks") else None
            kin_s = batch["kinematic_state"].to(device) if batch.get("kinematic_state") is not None else None

            targets = {k: v.to(device) for k, v in batch["targets"].items() if isinstance(v, torch.Tensor)}

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
                kinematic_state=kin_s,
            )

            loss, loss_dict = loss_fn(preds, targets)
            B = ir.size(0)
            total_samples += B
            batch_loss_sum += float(loss.item()) * B

            for t_key in unreduced_sums:
                u_key = f"unreduced_loss_{t_key}"
                c_key = "valid_track_components_count" if t_key == "track" else f"valid_{t_key}_count"
                unreduced_sums[t_key] += float(loss_dict[u_key])
                valid_counts[t_key] += int(round(float(loss_dict[c_key])))

            sample_ids_all.extend(batch["sample_ids"])
            storm_ids_all.extend(batch["storm_ids"])
            timestamps_all.extend(batch["timestamps"].cpu().numpy().tolist())

            anchor_kin_b = batch.get("anchor_kinematic_state")
            if anchor_kin_b is None and kin_s is not None:
                anchor_kin_b = kin_s[:, -1, :].cpu() if kin_s.dim() == 3 else kin_s.cpu()
            kin_anchor_all.append(anchor_kin_b.cpu().numpy())

            v_pred_all.append(preds["intensity"].cpu().numpy())
            v_true_all.append(targets["intensity"].cpu().numpy())
            v_mask_all.append(targets["intensity_mask"].cpu().numpy() > 0.5)

            p_pred_all.append(preds["pressure"].cpu().numpy())
            p_true_all.append(targets["pressure"].cpu().numpy())
            p_mask_all.append(targets["pressure_mask"].cpu().numpy() > 0.5)

            s_pred_all.append(preds["stage_logits"].argmax(dim=-1).cpu().numpy())
            s_true_all.append(targets["stage"].cpu().numpy())
            s_mask_all.append(targets["stage_mask"].cpu().numpy() > 0.5)

            dv_pred_all.append(preds["intensity_change"].cpu().numpy())
            dv_true_all.append(targets["intensity_change"].cpu().numpy())
            dv_mask_all.append(targets["intensity_change_mask"].cpu().numpy() > 0.5)

            ri_logits_np = preds["ri_logits"].cpu().numpy()
            ri_prob_np = torch.sigmoid(preds["ri_logits"]).cpu().numpy()
            ri_logit_all.append(ri_logits_np)
            ri_prob_all.append(ri_prob_np)
            ri_true_all.append(targets["rapid_intensification"].cpu().numpy())
            ri_mask_all.append(targets["ri_mask"].cpu().numpy() > 0.5)

            tr_pred_all.append(preds["track_delta"].view(-1, 4, 2).cpu().numpy())
            tr_true_all.append(targets["track_displacement"].cpu().numpy())
            tr_mask_all.append(targets["track_mask"].cpu().numpy() > 0.5)

    # Concatenate arrays across all batches
    sample_ids_arr = np.asarray(sample_ids_all, dtype=str)
    storm_ids_arr = np.asarray(storm_ids_all, dtype=str)
    timestamps_arr = np.asarray(timestamps_all, dtype=np.int64)
    kin_anchor = np.concatenate(kin_anchor_all, axis=0)  # (N, 10)

    v_pred = np.concatenate(v_pred_all, axis=0)
    v_true = np.concatenate(v_true_all, axis=0)
    v_mask = np.concatenate(v_mask_all, axis=0)
    maj_mask = v_mask & (v_true >= 96.0)

    p_pred = np.concatenate(p_pred_all, axis=0)
    p_true = np.concatenate(p_true_all, axis=0)
    p_mask = np.concatenate(p_mask_all, axis=0)

    s_pred = np.concatenate(s_pred_all, axis=0)
    s_true = np.concatenate(s_true_all, axis=0)
    s_mask = np.concatenate(s_mask_all, axis=0)

    dv_pred = np.concatenate(dv_pred_all, axis=0)  # (N, 4)
    dv_true = np.concatenate(dv_true_all, axis=0)  # (N, 4)
    dv_mask = np.concatenate(dv_mask_all, axis=0)  # (N, 4)

    ri_logit = np.concatenate(ri_logit_all, axis=0)
    ri_prob = np.concatenate(ri_prob_all, axis=0)
    ri_true = np.concatenate(ri_true_all, axis=0)
    ri_mask = np.concatenate(ri_mask_all, axis=0)

    tr_pred = np.concatenate(tr_pred_all, axis=0)  # (N, 4, 2)
    tr_true = np.concatenate(tr_true_all, axis=0)  # (N, 4, 2)
    tr_mask = np.concatenate(tr_mask_all, axis=0)  # (N, 4)

    # Precompute causal velocity baseline track displacement across horizons 6/12/18/24h
    vel_pred = np.zeros_like(tr_pred, dtype=np.float64)
    has_k = (kin_anchor[:, 9:10] > 0.5).astype(np.float64)
    for h_idx, h in enumerate([6, 12, 18, 24]):
        raw_vel = np.column_stack([kin_anchor[:, 0] * float(h), kin_anchor[:, 1] * float(h)])
        c_dx, c_dy = TRAIN_TRACK_CLIMATOLOGY_KM[h]
        c_arr = np.array([[c_dx, c_dy]], dtype=np.float64)
        vel_pred[:, h_idx, :] = has_k * raw_vel + (1.0 - has_k) * c_arr

    # 1. Global V1-Reference Loss
    w_map = loss_fn.task_weights
    task_losses_raw = {}
    task_losses_weighted = {}
    v1_global_loss = 0.0
    for t_key, w_val in w_map.items():
        raw_t = unreduced_sums[t_key] / max(1, valid_counts[t_key]) if valid_counts[t_key] > 0 else 0.0
        w_t = float(w_val) * raw_t
        task_losses_raw[f"loss_{t_key}"] = float(raw_t)
        task_losses_weighted[f"weighted_loss_{t_key}"] = float(w_t)
        v1_global_loss += w_t

    # 2. Wind MAE & Bias
    e_v = v_pred[v_mask] - v_true[v_mask]
    wind_mae_kt = float(np.mean(np.abs(e_v)))
    wind_rmse_kt = float(np.sqrt(np.mean(e_v ** 2)))
    wind_bias_kt = float(np.mean(e_v))

    # 3. Major-Hurricane MAE & Bias (>= 96 kt)
    e_maj = v_pred[maj_mask] - v_true[maj_mask]
    major_wind_mae_kt = float(np.mean(np.abs(e_maj))) if len(e_maj) > 0 else float("nan")
    major_wind_rmse_kt = float(np.sqrt(np.mean(e_maj ** 2))) if len(e_maj) > 0 else float("nan")
    major_wind_bias_kt = float(np.mean(e_maj)) if len(e_maj) > 0 else float("nan")

    # 4. Pressure MAE
    e_p = p_pred[p_mask] - p_true[p_mask]
    pressure_mae_hpa = float(np.mean(np.abs(e_p)))
    pressure_rmse_hpa = float(np.sqrt(np.mean(e_p ** 2)))
    pressure_bias_hpa = float(np.mean(e_p))

    # 5. Balanced Stage Accuracy & Overall Stage Accuracy
    stage_acc_pct = float(np.mean(s_pred[s_mask] == s_true[s_mask]) * 100.0)
    stage_balanced_acc_pct = _compute_balanced_stage_acc_pct(s_true[s_mask], s_pred[s_mask])

    # 6. dV MAE at 6/12/18/24h
    dv_mae_by_h = {}
    dv_rmse_by_h = {}
    for h_idx, h in enumerate([6, 12, 18, 24]):
        m_h = dv_mask[:, h_idx]
        e_dv = dv_pred[m_h, h_idx] - dv_true[m_h, h_idx]
        dv_mae_by_h[h] = float(np.mean(np.abs(e_dv)))
        dv_rmse_by_h[h] = float(np.sqrt(np.mean(e_dv ** 2)))

    # 7. RI PR-AUC, Brier, and Positive Count
    y_ri = ri_true[ri_mask]
    p_ri = ri_prob[ri_mask]
    ri_pr_auc = _compute_canonical_pr_auc(y_ri, p_ri)
    ri_brier = float(np.mean((p_ri - y_ri) ** 2))
    ri_pos_count = int(np.sum(y_ri == 1.0))
    ri_neg_count = int(np.sum(y_ri == 0.0))

    pred_ri_05 = (p_ri >= 0.5).astype(float)
    tp05 = float(np.sum((pred_ri_05 == 1.0) & (y_ri == 1.0)))
    fp05 = float(np.sum((pred_ri_05 == 1.0) & (y_ri == 0.0)))
    fn05 = float(np.sum((pred_ri_05 == 0.0) & (y_ri == 1.0)))
    prec05 = tp05 / max(1e-5, tp05 + fp05)
    rec05 = tp05 / max(1e-5, tp05 + fn05)
    ri_f1_05 = (2.0 * prec05 * rec05) / max(1e-5, prec05 + rec05)

    # 8. Track errors & +24h skill vs causal velocity baseline
    track_err_nn = np.sqrt(np.sum((tr_pred - tr_true) ** 2, axis=-1))   # (N, 4)
    track_err_vel = np.sqrt(np.sum((vel_pred - tr_true) ** 2, axis=-1)) # (N, 4)

    track_mae_by_h = {}
    vel_mae_by_h = {}
    track_skill_vs_vel_by_h = {}
    for h_idx, h in enumerate([6, 12, 18, 24]):
        m_tr = tr_mask[:, h_idx]
        nn_m = float(np.mean(track_err_nn[m_tr, h_idx]))
        vel_m = float(np.mean(track_err_vel[m_tr, h_idx]))
        track_mae_by_h[h] = nn_m
        vel_mae_by_h[h] = vel_m
        # Positive skill = how many km better (lower error) than causal velocity baseline
        track_skill_vs_vel_by_h[h] = float(vel_m - nn_m)

    # Composite VAL_MT_REL_ERR (using exact rounded metric conventions from trainer.py)
    rel_wind = round(wind_mae_kt, 2) / 24.71
    rel_press = round(pressure_mae_hpa, 2) / 18.92
    rel_stage = (100.0 - round(stage_acc_pct, 2)) / 78.05
    rel_dv = float(np.mean([round(dv_mae_by_h[h], 2) for h in [6, 12, 18, 24]])) / 8.1075
    rel_ri = 0.5 * ((1.0 - round(ri_pr_auc, 4)) / 0.9564 + (round(ri_brier, 4) / 0.0550))
    rel_track = float(np.mean([round(track_mae_by_h[h], 2) for h in [6, 12, 18, 24]])) / 119.2525
    val_mt_rel_err = float((rel_wind + rel_press + rel_stage + rel_dv + rel_ri + rel_track) / 6.0)

    # 9. Per-Storm Table
    unique_storms = sorted(np.unique(storm_ids_arr).tolist())
    per_storm_rows = []
    for st_id in unique_storms:
        st_m = (storm_ids_arr == st_id)
        st_v_m = st_m & v_mask
        st_maj_m = st_m & maj_mask
        st_p_m = st_m & p_mask
        st_s_m = st_m & s_mask
        st_dv24_m = st_m & dv_mask[:, 3]
        st_ri_m = st_m & ri_mask
        st_tr24_m = st_m & tr_mask[:, 3]

        st_e_v = v_pred[st_v_m] - v_true[st_v_m]
        st_e_maj = v_pred[st_maj_m] - v_true[st_maj_m]
        st_e_p = p_pred[st_p_m] - p_true[st_p_m]
        st_e_dv24 = dv_pred[st_dv24_m, 3] - dv_true[st_dv24_m, 3]

        st_ri_pos = int(np.sum(ri_true[st_ri_m] == 1.0)) if np.any(st_ri_m) else 0
        st_ri_brier = float(np.mean((ri_prob[st_ri_m] - ri_true[st_ri_m]) ** 2)) if np.any(st_ri_m) else float("nan")
        st_ri_prauc = _compute_canonical_pr_auc(ri_true[st_ri_m], ri_prob[st_ri_m]) if st_ri_pos > 0 else float("nan")

        st_tr24_nn = float(np.mean(track_err_nn[st_tr24_m, 3])) if np.any(st_tr24_m) else float("nan")
        st_tr24_vel = float(np.mean(track_err_vel[st_tr24_m, 3])) if np.any(st_tr24_m) else float("nan")
        st_tr24_skill = float(st_tr24_vel - st_tr24_nn) if np.any(st_tr24_m) else float("nan")

        per_storm_rows.append({
            "storm_id": st_id,
            "n_samples": int(np.sum(st_m)),
            "max_wind_kt": float(np.max(v_true[st_v_m])) if np.any(st_v_m) else float("nan"),
            "n_major_hurr": int(np.sum(st_maj_m)),
            "wind_mae_kt": float(np.mean(np.abs(st_e_v))) if len(st_e_v) > 0 else float("nan"),
            "wind_bias_kt": float(np.mean(st_e_v)) if len(st_e_v) > 0 else float("nan"),
            "major_wind_mae_kt": float(np.mean(np.abs(st_e_maj))) if len(st_e_maj) > 0 else float("nan"),
            "major_wind_bias_kt": float(np.mean(st_e_maj)) if len(st_e_maj) > 0 else float("nan"),
            "pressure_mae_hpa": float(np.mean(np.abs(st_e_p))) if len(st_e_p) > 0 else float("nan"),
            "stage_acc_pct": float(np.mean(s_pred[st_s_m] == s_true[st_s_m]) * 100.0) if np.any(st_s_m) else float("nan"),
            "stage_balanced_acc_pct": _compute_balanced_stage_acc_pct(s_true[st_s_m], s_pred[st_s_m]) if np.any(st_s_m) else float("nan"),
            "dv_24h_mae_kt": float(np.mean(np.abs(st_e_dv24))) if len(st_e_dv24) > 0 else float("nan"),
            "ri_valid_count": int(np.sum(st_ri_m)),
            "ri_pos_count": st_ri_pos,
            "ri_pr_auc": st_ri_prauc,
            "ri_brier": st_ri_brier,
            "track_24h_valid_count": int(np.sum(st_tr24_m)),
            "track_24h_err_km": st_tr24_nn,
            "track_24h_vel_baseline_km": st_tr24_vel,
            "track_24h_skill_vs_vel_km": st_tr24_skill,
        })

    per_storm_df = pd.DataFrame(per_storm_rows)

    predictions_payload = {
        "sample_ids": sample_ids_arr,
        "storm_ids": storm_ids_arr,
        "timestamps": timestamps_arr,
        "v_pred": v_pred,
        "v_true": v_true,
        "v_mask": v_mask,
        "maj_mask": maj_mask,
        "p_pred": p_pred,
        "p_true": p_true,
        "p_mask": p_mask,
        "s_pred": s_pred,
        "s_true": s_true,
        "s_mask": s_mask,
        "dv_pred": dv_pred,
        "dv_true": dv_true,
        "dv_mask": dv_mask,
        "ri_logit": ri_logit,
        "ri_prob": ri_prob,
        "ri_true": ri_true,
        "ri_mask": ri_mask,
        "tr_pred": tr_pred,
        "tr_true": tr_true,
        "tr_mask": tr_mask,
        "vel_pred": vel_pred,
        "track_err_nn": track_err_nn,
        "track_err_vel": track_err_vel,
    }

    return {
        "checkpoint_path": ckpt_path_str,
        "split": part_upper,
        "use_ema": bool(use_ema),
        "epoch": ckpt.get("epoch"),
        "total_samples": int(total_samples),
        "global_v1_reference_loss": round(float(v1_global_loss), 4),
        "batch_avg_v1_loss": round(float(batch_loss_sum / max(1, total_samples)), 4),
        "val_mt_rel_err": round(float(val_mt_rel_err), 4),
        "wind_mae_kt": round(float(wind_mae_kt), 2),
        "wind_rmse_kt": round(float(wind_rmse_kt), 2),
        "wind_bias_kt": round(float(wind_bias_kt), 2),
        "major_hurricane_mae_kt": round(float(major_wind_mae_kt), 2),
        "major_hurricane_rmse_kt": round(float(major_wind_rmse_kt), 2),
        "major_hurricane_bias_kt": round(float(major_wind_bias_kt), 2),
        "pressure_mae_hpa": round(float(pressure_mae_hpa), 2),
        "pressure_rmse_hpa": round(float(pressure_rmse_hpa), 2),
        "pressure_bias_hpa": round(float(pressure_bias_hpa), 2),
        "balanced_stage_acc_pct": round(float(stage_balanced_acc_pct), 2),
        "stage_acc_pct": round(float(stage_acc_pct), 2),
        "dv_6h_mae_kt": round(float(dv_mae_by_h[6]), 2),
        "dv_12h_mae_kt": round(float(dv_mae_by_h[12]), 2),
        "dv_18h_mae_kt": round(float(dv_mae_by_h[18]), 2),
        "dv_24h_mae_kt": round(float(dv_mae_by_h[24]), 2),
        "ri_pr_auc": round(float(ri_pr_auc), 4),
        "ri_brier": round(float(ri_brier), 4),
        "ri_f1_at_05": round(float(ri_f1_05), 4),
        "ri_positive_count": int(ri_pos_count),
        "ri_negative_count": int(ri_neg_count),
        "track_6h_km": round(float(track_mae_by_h[6]), 2),
        "track_12h_km": round(float(track_mae_by_h[12]), 2),
        "track_18h_km": round(float(track_mae_by_h[18]), 2),
        "track_24h_km": round(float(track_mae_by_h[24]), 2),
        "track_24h_causal_vel_baseline_km": round(float(vel_mae_by_h[24]), 2),
        "track_24h_skill_vs_causal_vel_km": round(float(track_skill_vs_vel_by_h[24]), 2),
        "valid_counts": {
            "wind": int(np.sum(v_mask)),
            "major_hurricane": int(np.sum(maj_mask)),
            "pressure": int(np.sum(p_mask)),
            "stage": int(np.sum(s_mask)),
            "dv_6h": int(np.sum(dv_mask[:, 0])),
            "dv_12h": int(np.sum(dv_mask[:, 1])),
            "dv_18h": int(np.sum(dv_mask[:, 2])),
            "dv_24h": int(np.sum(dv_mask[:, 3])),
            "ri": int(np.sum(ri_mask)),
            "ri_pos": int(ri_pos_count),
            "track_6h": int(np.sum(tr_mask[:, 0])),
            "track_12h": int(np.sum(tr_mask[:, 1])),
            "track_18h": int(np.sum(tr_mask[:, 2])),
            "track_24h": int(np.sum(tr_mask[:, 3])),
        },
        "task_losses_raw": {k: round(v, 4) for k, v in task_losses_raw.items()},
        "task_losses_weighted": {k: round(v, 4) for k, v in task_losses_weighted.items()},
        "per_storm_table": per_storm_df,
        "predictions": predictions_payload,
    }


def _extract_predictions_dict(preds_obj: Union[Dict[str, Any], str, Path]) -> Dict[str, np.ndarray]:
    """Accept output of `evaluate()`, raw `predictions` dict, or path to saved `.npz` file."""
    if isinstance(preds_obj, (str, Path)):
        with np.load(str(preds_obj), allow_pickle=False) as data:
            return {k: data[k] for k in data.files}
    if isinstance(preds_obj, dict):
        if "predictions" in preds_obj and isinstance(preds_obj["predictions"], dict):
            return preds_obj["predictions"]
        return preds_obj
    raise TypeError(f"Unsupported predictions object type: {type(preds_obj)}")


def _compute_resample_metrics(p: Dict[str, np.ndarray], idx: np.ndarray) -> Dict[str, float]:
    """Compute canonical metrics on a resampled observation index array `idx`."""
    out: Dict[str, float] = {}

    # 1. Wind MAE & Bias
    vm = p["v_mask"][idx]
    if np.any(vm):
        ev = p["v_pred"][idx][vm] - p["v_true"][idx][vm]
        out["wind_mae_kt"] = float(np.mean(np.abs(ev)))
        out["wind_bias_kt"] = float(np.mean(ev))
    else:
        out["wind_mae_kt"] = float("nan")
        out["wind_bias_kt"] = float("nan")

    # 2. Major-Hurricane MAE & Bias (>= 96 kt)
    mm = p["maj_mask"][idx]
    if np.any(mm):
        em = p["v_pred"][idx][mm] - p["v_true"][idx][mm]
        out["major_hurricane_mae_kt"] = float(np.mean(np.abs(em)))
        out["major_hurricane_bias_kt"] = float(np.mean(em))
    else:
        out["major_hurricane_mae_kt"] = float("nan")
        out["major_hurricane_bias_kt"] = float("nan")

    # 3. Pressure MAE
    pm = p["p_mask"][idx]
    if np.any(pm):
        ep = p["p_pred"][idx][pm] - p["p_true"][idx][pm]
        out["pressure_mae_hpa"] = float(np.mean(np.abs(ep)))
    else:
        out["pressure_mae_hpa"] = float("nan")

    # 4. Balanced Stage Accuracy & Overall Stage Accuracy
    sm = p["s_mask"][idx]
    if np.any(sm):
        st = p["s_true"][idx][sm]
        sp = p["s_pred"][idx][sm]
        out["stage_acc_pct"] = float(np.mean(sp == st) * 100.0)
        out["balanced_stage_acc_pct"] = _compute_balanced_stage_acc_pct(st, sp)
    else:
        out["stage_acc_pct"] = float("nan")
        out["balanced_stage_acc_pct"] = float("nan")

    # 5. dV MAE at 6/12/18/24h
    for h_idx, h in enumerate([6, 12, 18, 24]):
        dvm = p["dv_mask"][idx, h_idx]
        if np.any(dvm):
            edv = p["dv_pred"][idx, h_idx][dvm] - p["dv_true"][idx, h_idx][dvm]
            out[f"dv_{h}h_mae_kt"] = float(np.mean(np.abs(edv)))
        else:
            out[f"dv_{h}h_mae_kt"] = float("nan")

    # 6. RI PR-AUC and Brier (explicitly drop resamples with 0 positive RI observations)
    rim = p["ri_mask"][idx]
    if np.any(rim):
        y_ri = p["ri_true"][idx][rim]
        p_ri = p["ri_prob"][idx][rim]
        n_pos = int(np.sum(y_ri == 1.0))
        out["ri_pos_count_in_resample"] = float(n_pos)
        if n_pos > 0:
            out["ri_pr_auc"] = _compute_canonical_pr_auc(y_ri, p_ri)
            out["ri_brier"] = float(np.mean((p_ri - y_ri) ** 2))
        else:
            out["ri_pr_auc"] = float("nan")
            out["ri_brier"] = float("nan")
    else:
        out["ri_pos_count_in_resample"] = 0.0
        out["ri_pr_auc"] = float("nan")
        out["ri_brier"] = float("nan")

    # 7. +24h Track Error (km) & Skill vs. Causal Velocity Baseline
    trm24 = p["tr_mask"][idx, 3]
    if np.any(trm24):
        nn_err24 = float(np.mean(p["track_err_nn"][idx, 3][trm24]))
        vel_err24 = float(np.mean(p["track_err_vel"][idx, 3][trm24]))
        out["track_24h_km"] = nn_err24
        out["track_24h_skill_vs_causal_vel_km"] = float(vel_err24 - nn_err24)
    else:
        out["track_24h_km"] = float("nan")
        out["track_24h_skill_vs_causal_vel_km"] = float("nan")

    return out


def storm_bootstrap(
    preds_a: Union[Dict[str, Any], str, Path],
    preds_b: Union[Dict[str, Any], str, Path],
    n: int = 2000,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Paired storm-cluster bootstrap comparing model A (`preds_a`) and model B (`preds_b`).

    - Resamples STORMS (not observations) with replacement (`size = n_storms`).
    - Uses the exact same resampled storms in each iteration for both `preds_a` and `preds_b`.
    - Handles RI resamples with zero positives (`ri_pos == 0`) by dropping them from RI metric
      distributions and counting `ri_zero_positive_resamples_dropped`.
    - Returns paired deltas (`delta = metric(preds_a) - metric(preds_b)`) with mean, std, median,
      and 95% confidence interval (`[ci95_low, ci95_high]`).
    """
    pa = _extract_predictions_dict(preds_a)
    pb = _extract_predictions_dict(preds_b)

    if not np.array_equal(pa["sample_ids"], pb["sample_ids"]):
        raise ValueError("preds_a and preds_b do not have identical sample_ids ordering!")
    if not np.array_equal(pa["storm_ids"], pb["storm_ids"]):
        raise ValueError("preds_a and preds_b do not have identical storm_ids ordering!")

    storms = sorted(np.unique(pa["storm_ids"]).tolist())
    n_storms = len(storms)
    storm_to_indices = {s: np.where(pa["storm_ids"] == s)[0] for s in storms}

    rng = np.random.default_rng(seed)

    metric_keys = [
        "wind_mae_kt",
        "wind_bias_kt",
        "major_hurricane_mae_kt",
        "major_hurricane_bias_kt",
        "pressure_mae_hpa",
        "balanced_stage_acc_pct",
        "stage_acc_pct",
        "dv_6h_mae_kt",
        "dv_12h_mae_kt",
        "dv_18h_mae_kt",
        "dv_24h_mae_kt",
        "ri_pr_auc",
        "ri_brier",
        "track_24h_km",
        "track_24h_skill_vs_causal_vel_km",
    ]

    deltas: Dict[str, List[float]] = {k: [] for k in metric_keys}
    vals_a: Dict[str, List[float]] = {k: [] for k in metric_keys}
    vals_b: Dict[str, List[float]] = {k: [] for k in metric_keys}

    ri_zero_positive_resamples_dropped = 0
    major_hurricane_zero_resamples_dropped = 0

    for _ in range(int(n)):
        sampled_storms = rng.choice(storms, size=n_storms, replace=True)
        idx_cat = np.concatenate([storm_to_indices[s] for s in sampled_storms])

        ma = _compute_resample_metrics(pa, idx_cat)
        mb = _compute_resample_metrics(pb, idx_cat)

        if ma["ri_pos_count_in_resample"] == 0:
            ri_zero_positive_resamples_dropped += 1
        if math.isnan(ma["major_hurricane_mae_kt"]):
            major_hurricane_zero_resamples_dropped += 1

        for k in metric_keys:
            va = ma[k]
            vb = mb[k]
            if not (math.isnan(va) or math.isnan(vb)):
                vals_a[k].append(va)
                vals_b[k].append(vb)
                deltas[k].append(va - vb)

    # Full-sample point estimates
    all_idx = np.arange(len(pa["sample_ids"]))
    point_a = _compute_resample_metrics(pa, all_idx)
    point_b = _compute_resample_metrics(pb, all_idx)

    paired_summary: Dict[str, Any] = {}
    for k in metric_keys:
        arr_d = np.asarray(deltas[k], dtype=np.float64)
        arr_a = np.asarray(vals_a[k], dtype=np.float64)
        arr_b = np.asarray(vals_b[k], dtype=np.float64)
        if len(arr_d) > 0:
            paired_summary[k] = {
                "point_a": round(float(point_a[k]), 4),
                "point_b": round(float(point_b[k]), 4),
                "point_delta_a_minus_b": round(float(point_a[k] - point_b[k]), 4),
                "boot_mean_delta": round(float(np.mean(arr_d)), 4),
                "boot_std_delta": round(float(np.std(arr_d)), 4),
                "boot_median_delta": round(float(np.median(arr_d)), 4),
                "ci95_low": round(float(np.percentile(arr_d, 2.5)), 4),
                "ci95_high": round(float(np.percentile(arr_d, 97.5)), 4),
                "model_a_ci95": [round(float(np.percentile(arr_a, 2.5)), 4), round(float(np.percentile(arr_a, 97.5)), 4)],
                "model_b_ci95": [round(float(np.percentile(arr_b, 2.5)), 4), round(float(np.percentile(arr_b, 97.5)), 4)],
                "valid_resamples": int(len(arr_d)),
                "dropped_resamples": int(n - len(arr_d)),
            }
        else:
            paired_summary[k] = {
                "point_a": None,
                "point_b": None,
                "point_delta_a_minus_b": None,
                "boot_mean_delta": None,
                "boot_std_delta": None,
                "boot_median_delta": None,
                "ci95_low": None,
                "ci95_high": None,
                "valid_resamples": 0,
                "dropped_resamples": int(n),
            }

    return {
        "n_resamples_requested": int(n),
        "seed": int(seed),
        "n_storms": int(n_storms),
        "storms": storms,
        "ri_zero_positive_resamples_dropped": int(ri_zero_positive_resamples_dropped),
        "ri_valid_resamples_used": int(n - ri_zero_positive_resamples_dropped),
        "major_hurricane_zero_resamples_dropped": int(major_hurricane_zero_resamples_dropped),
        "paired_deltas_a_minus_b": paired_summary,
    }


def save_canonical_predictions_npz(eval_result: Dict[str, Any], npz_path: Union[str, Path]) -> str:
    """Save per-observation predictions and masks from `evaluate()` to a compressed `.npz` file."""
    out_p = Path(npz_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    preds = eval_result["predictions"]
    np.savez_compressed(str(out_p), **preds)
    return str(out_p)


def run_canonical_val_benchmark(output_dir: Union[str, Path] = "reports/canonical") -> Dict[str, Any]:
    """
    Evaluate all 5 reference checkpoints (`Original_Ep5`, `C1_Ep8`, `T6_Ep6`, `S24_Ep7`, `S24_Ep10`)
    on `VAL` using the single `evaluate()` path, save `reports/canonical/val_table.csv`,
    per-checkpoint `.npz` files, per-storm tables, and `storm_bootstrap` paired comparisons.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_results: Dict[str, Dict[str, Any]] = {}
    table_rows: List[Dict[str, Any]] = []

    for model_name, rel_ckpt_path in CANONICAL_CHECKPOINTS.items():
        print(f"[CANONICAL EVAL] Evaluating {model_name} ({rel_ckpt_path}) on VAL...", flush=True)
        res = evaluate(rel_ckpt_path, split="val", use_ema=False)
        eval_results[model_name] = res

        npz_file = out_dir / f"{model_name.lower()}_val_predictions.npz"
        save_canonical_predictions_npz(res, npz_file)

        storm_csv = out_dir / f"{model_name.lower()}_val_per_storm.csv"
        res["per_storm_table"].to_csv(storm_csv, index=False)

        table_rows.append({
            "model": model_name,
            "checkpoint_path": rel_ckpt_path,
            "epoch": res["epoch"],
            "global_v1_reference_loss": res["global_v1_reference_loss"],
            "batch_avg_v1_loss": res["batch_avg_v1_loss"],
            "val_mt_rel_err": res["val_mt_rel_err"],
            "wind_mae_kt": res["wind_mae_kt"],
            "wind_bias_kt": res["wind_bias_kt"],
            "major_hurricane_mae_kt": res["major_hurricane_mae_kt"],
            "major_hurricane_bias_kt": res["major_hurricane_bias_kt"],
            "pressure_mae_hpa": res["pressure_mae_hpa"],
            "balanced_stage_acc_pct": res["balanced_stage_acc_pct"],
            "stage_acc_pct": res["stage_acc_pct"],
            "dv_6h_mae_kt": res["dv_6h_mae_kt"],
            "dv_12h_mae_kt": res["dv_12h_mae_kt"],
            "dv_18h_mae_kt": res["dv_18h_mae_kt"],
            "dv_24h_mae_kt": res["dv_24h_mae_kt"],
            "ri_pr_auc": res["ri_pr_auc"],
            "ri_brier": res["ri_brier"],
            "ri_positive_count": res["ri_positive_count"],
            "track_24h_km": res["track_24h_km"],
            "track_24h_causal_vel_baseline_km": res["track_24h_causal_vel_baseline_km"],
            "track_24h_skill_vs_causal_vel_km": res["track_24h_skill_vs_causal_vel_km"],
            "weighted_loss_intensity": res["task_losses_weighted"]["weighted_loss_intensity"],
            "weighted_loss_pressure": res["task_losses_weighted"]["weighted_loss_pressure"],
            "weighted_loss_stage": res["task_losses_weighted"]["weighted_loss_stage"],
            "weighted_loss_intensity_change": res["task_losses_weighted"]["weighted_loss_intensity_change"],
            "weighted_loss_ri": res["task_losses_weighted"]["weighted_loss_ri"],
            "weighted_loss_track": res["task_losses_weighted"]["weighted_loss_track"],
        })

    val_table_df = pd.DataFrame(table_rows)
    val_table_csv = out_dir / "val_table.csv"
    val_table_df.to_csv(val_table_csv, index=False)

    # Combined per-storm table across all 5 checkpoints
    combined_storms = []
    for model_name, res in eval_results.items():
        df_s = res["per_storm_table"].copy()
        df_s.insert(0, "model", model_name)
        combined_storms.append(df_s)
    combined_storms_df = pd.concat(combined_storms, ignore_index=True)
    combined_storms_df.to_csv(out_dir / "per_storm_val_all_models.csv", index=False)

    # Run paired storm_bootstrap (n=2000, seed=0) for key comparisons
    bootstrap_comparisons = {
        "S24_Ep10_minus_Original_Ep5": storm_bootstrap(eval_results["S24_Ep10"], eval_results["Original_Ep5"], n=2000, seed=0),
        "S24_Ep10_minus_C1_Ep8": storm_bootstrap(eval_results["S24_Ep10"], eval_results["C1_Ep8"], n=2000, seed=0),
        "S24_Ep10_minus_T6_Ep6": storm_bootstrap(eval_results["S24_Ep10"], eval_results["T6_Ep6"], n=2000, seed=0),
        "S24_Ep7_minus_Original_Ep5": storm_bootstrap(eval_results["S24_Ep7"], eval_results["Original_Ep5"], n=2000, seed=0),
        "S24_Ep7_minus_C1_Ep8": storm_bootstrap(eval_results["S24_Ep7"], eval_results["C1_Ep8"], n=2000, seed=0),
        "T6_Ep6_minus_Original_Ep5": storm_bootstrap(eval_results["T6_Ep6"], eval_results["Original_Ep5"], n=2000, seed=0),
        "C1_Ep8_minus_Original_Ep5": storm_bootstrap(eval_results["C1_Ep8"], eval_results["Original_Ep5"], n=2000, seed=0),
    }

    summary_json_path = out_dir / "canonical_val_summary.json"
    summary_payload = {
        "val_table": table_rows,
        "valid_counts": eval_results["Original_Ep5"]["valid_counts"],
        "storm_bootstrap_n2000_seed0": bootstrap_comparisons,
    }
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print(f"[CANONICAL EVAL] Saved {val_table_csv} and {summary_json_path}", flush=True)
    return summary_payload


if __name__ == "__main__":
    run_canonical_val_benchmark()

