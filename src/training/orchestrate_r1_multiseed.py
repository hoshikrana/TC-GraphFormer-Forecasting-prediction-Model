"""
Orchestration Script: Train R1 Seeds 43 & 44, Evaluate Single Seeds & Ensembles,
Perform Uncertainty Calibration and Storm Bootstrap Analysis.
"""

import os
import sys
import time
import math
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.evaluation.canonical_eval import (
    evaluate,
    storm_bootstrap,
    save_canonical_predictions_npz,
    _compute_resample_metrics,
    _compute_balanced_stage_acc_pct,
    _compute_canonical_pr_auc,
)
from src.training.losses import MultiTaskLoss

REPO_ROOT = Path("D:/Enjula uchoi/TC-GraphFormer-verified")
CANONICAL_DIR = REPO_ROOT / "reports/canonical"
REPORTS_DIR = REPO_ROOT / "reports/phase2e4_100gb"
CONFIGS_DIR = REPO_ROOT / "configs"
PYTHON_EXE = sys.executable


def run_training_parallel():
    """Launch Seed 43 and Seed 44 training in parallel and monitor until completion."""
    cmd_seed43 = [PYTHON_EXE, "src/training/train_multitask.py", "--config", str(CONFIGS_DIR / "exp_r1_seed43.yaml")]
    cmd_seed44 = [PYTHON_EXE, "src/training/train_multitask.py", "--config", str(CONFIGS_DIR / "exp_r1_seed44.yaml")]

    log_s43_path = REPORTS_DIR / "train_seed43.log"
    log_s44_path = REPORTS_DIR / "train_seed44.log"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    log_s43 = open(log_s43_path, "w", encoding="utf-8")
    log_s44 = open(log_s44_path, "w", encoding="utf-8")

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Launching Seed 43 and Seed 44 in parallel...", flush=True)
    p43 = subprocess.Popen(cmd_seed43, stdout=log_s43, stderr=subprocess.STDOUT, cwd=str(REPO_ROOT))
    p44 = subprocess.Popen(cmd_seed44, stdout=log_s44, stderr=subprocess.STDOUT, cwd=str(REPO_ROOT))

    csv_s43 = REPORTS_DIR / "exp_r1_seed43_epochs.csv"
    csv_s44 = REPORTS_DIR / "exp_r1_seed44_epochs.csv"

    start_time = time.time()
    last_reported_ep43 = 0
    last_reported_ep44 = 0

    while True:
        ret43 = p43.poll()
        ret44 = p44.poll()

        ep43 = 0
        if csv_s43.exists():
            try:
                df43 = pd.read_csv(csv_s43)
                if not df43.empty:
                    ep43 = int(df43["epoch"].iloc[-1])
            except Exception:
                pass

        ep44 = 0
        if csv_s44.exists():
            try:
                df44 = pd.read_csv(csv_s44)
                if not df44.empty:
                    ep44 = int(df44["epoch"].iloc[-1])
            except Exception:
                pass

        elapsed_min = (time.time() - start_time) / 60.0
        if ep43 != last_reported_ep43 or ep44 != last_reported_ep44 or int(elapsed_min) % 15 == 0:
            print(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Elapsed: {elapsed_min:.1f}m | "
                f"Seed 43: Epoch {ep43}/24 (active={ret43 is None}) | "
                f"Seed 44: Epoch {ep44}/24 (active={ret44 is None})",
                flush=True,
            )
            last_reported_ep43 = ep43
            last_reported_ep44 = ep44

        if ret43 is not None and ret44 is not None:
            break

        time.sleep(30)

    log_s43.close()
    log_s44.close()

    if ret43 != 0:
        raise RuntimeError(f"Seed 43 training failed with code {ret43}! Check {log_s43_path}")
    if ret44 != 0:
        raise RuntimeError(f"Seed 44 training failed with code {ret44}! Check {log_s44_path}")

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Both seeds completed successfully in {(time.time() - start_time)/60.0:.1f} minutes!", flush=True)


def compute_v1_loss_and_metrics_for_ensemble(ens_preds: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Compute exact global V1 loss, per-task losses, and canonical metrics for an ensemble."""
    idx = np.arange(len(ens_preds["sample_ids"]))
    m = _compute_resample_metrics(ens_preds, idx)

    # Compute unreduced V1 losses via MultiTaskLoss formula
    v_true = ens_preds["v_true"]
    v_pred = ens_preds["v_pred"]
    v_mask = ens_preds["v_mask"].astype(bool)

    # 1. Intensity (Huber beta=8.0, regime weighting)
    v_excess = np.clip((v_true - 50.0) / 50.0, 0.0, 2.0) * v_mask
    w_v_regime = 1.0 + 2.5 * (v_excess ** 2.0)
    diff_v = np.abs(v_pred - v_true)
    huber_v = np.where(diff_v < 8.0, 0.5 * (diff_v ** 2) / 8.0, diff_v - 4.0) * w_v_regime
    loss_v = float(np.sum(huber_v[v_mask]) / max(1, np.sum(v_mask)))

    # 2. Pressure (Huber beta=2.5 on p/10)
    p_true = ens_preds["p_true"] / 10.0
    p_pred = ens_preds["p_pred"] / 10.0
    p_mask = ens_preds["p_mask"].astype(bool)
    diff_p = np.abs(p_pred - p_true)
    huber_p = np.where(diff_p < 2.5, 0.5 * (diff_p ** 2) / 2.5, diff_p - 1.25)
    loss_p = float(np.sum(huber_p[p_mask]) / max(1, np.sum(p_mask)))

    # 3. Stage Cross-Entropy
    s_true = ens_preds["s_true"].astype(int)
    s_prob = ens_preds["s_prob"]
    s_mask = ens_preds["s_mask"].astype(bool)
    # pick true class probability
    n_s = len(s_true)
    s_prob_true = s_prob[np.arange(n_s), np.clip(s_true, 0, 4)]
    ce_s = -np.log(np.clip(s_prob_true, 1e-12, 1.0))
    loss_s = float(np.sum(ce_s[s_mask]) / max(1, np.sum(s_mask)))

    # 4. Intensity Change (Huber beta=4.0)
    dv_true = ens_preds["dv_true"]
    dv_pred = ens_preds["dv_pred"]
    dv_mask = ens_preds["dv_mask"].astype(bool)
    diff_dv = np.abs(dv_pred - dv_true)
    huber_dv = np.where(diff_dv < 4.0, 0.5 * (diff_dv ** 2) / 4.0, diff_dv - 2.0)
    loss_dv = float(np.sum(huber_dv[dv_mask]) / max(1, np.sum(dv_mask)))

    # 5. RI (BCE with pos_weight=18.10)
    ri_true = ens_preds["ri_true"]
    ri_prob = np.clip(ens_preds["ri_prob"], 1e-7, 1.0 - 1e-7)
    ri_mask = ens_preds["ri_mask"].astype(bool)
    # BCE = - [18.10 * y * log(p) + (1-y) * log(1-p)]
    ri_bce = -(18.10 * ri_true * np.log(ri_prob) + (1.0 - ri_true) * np.log(1.0 - ri_prob))
    loss_ri = float(np.sum(ri_bce[ri_mask]) / max(1, np.sum(ri_mask)))

    # 6. Track (Smooth L1 on dx/100, dy/100 with beta=0.1)
    tr_true = ens_preds["tr_true"] / 100.0  # (N, 4, 2)
    tr_pred = ens_preds["tr_pred"] / 100.0
    tr_mask = ens_preds["tr_mask"].astype(bool)  # (N, 4)
    diff_tr = np.abs(tr_pred - tr_true)  # (N, 4, 2)
    huber_tr = np.where(diff_tr < 0.1, 0.5 * (diff_tr ** 2) / 0.1, diff_tr - 0.05)
    loss_tr = float(np.sum(huber_tr[tr_mask]) / max(1, 2 * np.sum(tr_mask)))

    weighted_v = 1.0 * loss_v
    weighted_p = 0.5 * loss_p
    weighted_s = 0.5 * loss_s
    weighted_dv = 1.0 * loss_dv
    weighted_ri = 2.0 * loss_ri
    weighted_tr = 0.2 * loss_tr
    v1_global_loss = weighted_v + weighted_p + weighted_s + weighted_dv + weighted_ri + weighted_tr

    # Composite VAL_MT_REL_ERR
    rel_wind = round(m["wind_mae_kt"], 2) / 24.71
    rel_press = round(m["pressure_mae_hpa"], 2) / 18.92
    rel_stage = (100.0 - round(m["stage_acc_pct"], 2)) / 78.05
    rel_dv = float(np.mean([round(m[f"dv_{h}h_mae_kt"], 2) for h in [6, 12, 18, 24]])) / 8.1075
    rel_ri = 0.5 * ((1.0 - round(m["ri_pr_auc"], 4)) / 0.9564 + (round(m["ri_brier"], 4) / 0.0550))
    # Track across 4 horizons
    track_maes = []
    for h_idx, h in enumerate([6, 12, 18, 24]):
        m_tr = ens_preds["tr_mask"][:, h_idx].astype(bool)
        track_maes.append(float(np.mean(ens_preds["track_err_nn"][m_tr, h_idx])))
    rel_track = float(np.mean([round(x, 2) for x in track_maes])) / 119.2525
    val_mt_rel_err = float((rel_wind + rel_press + rel_stage + rel_dv + rel_ri + rel_track) / 6.0)

    res = {
        "global_v1_reference_loss": round(v1_global_loss, 4),
        "val_mt_rel_err": round(val_mt_rel_err, 4),
        "wind_mae_kt": round(m["wind_mae_kt"], 2),
        "wind_bias_kt": round(m["wind_bias_kt"], 2),
        "major_hurricane_mae_kt": round(m["major_hurricane_mae_kt"], 2),
        "major_hurricane_bias_kt": round(m["major_hurricane_bias_kt"], 2),
        "pressure_mae_hpa": round(m["pressure_mae_hpa"], 2),
        "balanced_stage_acc_pct": round(m["balanced_stage_acc_pct"], 2),
        "stage_acc_pct": round(m["stage_acc_pct"], 2),
        "dv_6h_mae_kt": round(m["dv_6h_mae_kt"], 2),
        "dv_12h_mae_kt": round(m["dv_12h_mae_kt"], 2),
        "dv_18h_mae_kt": round(m["dv_18h_mae_kt"], 2),
        "dv_24h_mae_kt": round(m["dv_24h_mae_kt"], 2),
        "ri_pr_auc": round(m["ri_pr_auc"], 4),
        "ri_brier": round(m["ri_brier"], 4),
        "track_24h_km": round(m["track_24h_km"], 2),
        "track_24h_skill_vs_causal_vel_km": round(m["track_24h_skill_vs_causal_vel_km"], 2),
        "task_losses_weighted": {
            "weighted_loss_intensity": round(weighted_v, 4),
            "weighted_loss_pressure": round(weighted_p, 4),
            "weighted_loss_stage": round(weighted_s, 4),
            "weighted_loss_intensity_change": round(weighted_dv, 4),
            "weighted_loss_ri": round(weighted_ri, 4),
            "weighted_loss_track": round(weighted_tr, 4),
        },
    }
    return res


def evaluate_and_ensemble():
    """Evaluate individual seeds and form ensembles."""
    ckpt_s42 = REPO_ROOT / "checkpoints/tcprimed_100gb_exp_r1/best_checkpoint.pt"
    ckpt_s43 = REPO_ROOT / "checkpoints/tcprimed_100gb_exp_r1_seed43/best_checkpoint.pt"
    ckpt_s44 = REPO_ROOT / "checkpoints/tcprimed_100gb_exp_r1_seed44/best_checkpoint.pt"
    ckpt_s24_ep7 = REPO_ROOT / "checkpoints/tcprimed_100gb_exp_s24/best_mt_rel_err_checkpoint.pt"
    ckpt_s24_ep10 = REPO_ROOT / "checkpoints/tcprimed_100gb_exp_s24/best_checkpoint.pt"

    print("\n[EVAL] Running canonical evaluation on all models...", flush=True)
    res_s42 = evaluate(ckpt_s42, split="val", use_ema=False)
    save_canonical_predictions_npz(res_s42, CANONICAL_DIR / "r1_seed42_val_predictions.npz")

    res_s43 = evaluate(ckpt_s43, split="val", use_ema=False)
    save_canonical_predictions_npz(res_s43, CANONICAL_DIR / "r1_seed43_val_predictions.npz")

    res_s44 = evaluate(ckpt_s44, split="val", use_ema=False)
    save_canonical_predictions_npz(res_s44, CANONICAL_DIR / "r1_seed44_val_predictions.npz")

    res_s24_ep7 = evaluate(ckpt_s24_ep7, split="val", use_ema=False)
    save_canonical_predictions_npz(res_s24_ep7, CANONICAL_DIR / "s24_ep7_val_predictions.npz")

    res_s24_ep10 = evaluate(ckpt_s24_ep10, split="val", use_ema=False)
    save_canonical_predictions_npz(res_s24_ep10, CANONICAL_DIR / "s24_ep10_val_predictions.npz")

    p42 = res_s42["predictions"]
    p43 = res_s43["predictions"]
    p44 = res_s44["predictions"]
    p_ep7 = res_s24_ep7["predictions"]
    p_ep10 = res_s24_ep10["predictions"]

    # =========================================================================
    # Ensemble (ii): 3-Seed R1 Mean-Prediction Ensemble (Seed 42 + Seed 43 + Seed 44)
    # =========================================================================
    print("[ENSEMBLE] Building 3-Seed R1 Ensemble (42 + 43 + 44)...", flush=True)
    ens_3seed = dict(p42)
    ens_3seed["v_pred"] = (p42["v_pred"] + p43["v_pred"] + p44["v_pred"]) / 3.0
    ens_3seed["p_pred"] = (p42["p_pred"] + p43["p_pred"] + p44["p_pred"]) / 3.0
    ens_3seed["dv_pred"] = (p42["dv_pred"] + p43["dv_pred"] + p44["dv_pred"]) / 3.0
    ens_3seed["tr_pred"] = (p42["tr_pred"] + p43["tr_pred"] + p44["tr_pred"]) / 3.0
    ens_3seed["track_err_nn"] = np.sqrt(np.sum((ens_3seed["tr_pred"] - ens_3seed["tr_true"]) ** 2, axis=-1))
    ens_3seed["ri_prob"] = (p42["ri_prob"] + p43["ri_prob"] + p44["ri_prob"]) / 3.0
    ens_3seed["s_prob"] = (p42["s_prob"] + p43["s_prob"] + p44["s_prob"]) / 3.0
    ens_3seed["s_pred"] = np.argmax(ens_3seed["s_prob"], axis=-1)

    np.savez_compressed(str(CANONICAL_DIR / "ensemble_r1_3seeds_val_predictions.npz"), **ens_3seed)
    res_ens_3seed = compute_v1_loss_and_metrics_for_ensemble(ens_3seed)

    # =========================================================================
    # Ensemble (iii): Cross-Architecture Ensemble (R1 Seed 42 + S24 Ep7 + S24 Ep10)
    # =========================================================================
    print("[ENSEMBLE] Building Cross-Architecture Ensemble (R1-42 + S24-Ep7 + S24-Ep10)...", flush=True)
    ens_cross = dict(p42)
    ens_cross["v_pred"] = (p42["v_pred"] + p_ep7["v_pred"] + p_ep10["v_pred"]) / 3.0
    ens_cross["p_pred"] = (p42["p_pred"] + p_ep7["p_pred"] + p_ep10["p_pred"]) / 3.0
    ens_cross["dv_pred"] = (p42["dv_pred"] + p_ep7["dv_pred"] + p_ep10["dv_pred"]) / 3.0
    ens_cross["tr_pred"] = (p42["tr_pred"] + p_ep7["tr_pred"] + p_ep10["tr_pred"]) / 3.0
    ens_cross["track_err_nn"] = np.sqrt(np.sum((ens_cross["tr_pred"] - ens_cross["tr_true"]) ** 2, axis=-1))
    ens_cross["ri_prob"] = (p42["ri_prob"] + p_ep7["ri_prob"] + p_ep10["ri_prob"]) / 3.0
    ens_cross["s_prob"] = (p42["s_prob"] + p_ep7["s_prob"] + p_ep10["s_prob"]) / 3.0
    ens_cross["s_pred"] = np.argmax(ens_cross["s_prob"], axis=-1)

    np.savez_compressed(str(CANONICAL_DIR / "ensemble_r1_s24_val_predictions.npz"), **ens_cross)
    res_ens_cross = compute_v1_loss_and_metrics_for_ensemble(ens_cross)

    # =========================================================================
    # Single-Seed Spread: Mean ± Std Across Seeds 42, 43, 44
    # =========================================================================
    metrics_to_track = [
        "global_v1_reference_loss",
        "val_mt_rel_err",
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

    single_seed_stats = {}
    for m in metrics_to_track:
        vals = [res_s42[m], res_s43[m], res_s44[m]]
        single_seed_stats[m] = {
            "seed_42": vals[0],
            "seed_43": vals[1],
            "seed_44": vals[2],
            "mean": round(float(np.mean(vals)), 4),
            "std": round(float(np.std(vals, ddof=1)), 4),
            "spread_max_minus_min": round(float(max(vals) - min(vals)), 4),
        }

    # =========================================================================
    # Paired Storm Bootstrap (2000 resamples, seed=0)
    # =========================================================================
    print("\n[BOOTSTRAP] Running 2000 paired storm resamples against S24 Ep10...", flush=True)
    bs_3seed_vs_s24ep10 = storm_bootstrap(ens_3seed, p_ep10, n=2000, seed=0)
    bs_cross_vs_s24ep10 = storm_bootstrap(ens_cross, p_ep10, n=2000, seed=0)
    bs_3seed_vs_seed42 = storm_bootstrap(ens_3seed, p42, n=2000, seed=0)
    bs_cross_vs_seed42 = storm_bootstrap(ens_cross, p42, n=2000, seed=0)

    # =========================================================================
    # Uncertainty Proxy Calibration: Spearman Correlation (Spread vs Absolute Error)
    # =========================================================================
    print("[CALIBRATION] Computing uncertainty proxy calibration (Spearman correlation)...", flush=True)

    # 1. 3-Seed R1 Ensemble
    # Wind:
    v_stack_3s = np.stack([p42["v_pred"], p43["v_pred"], p44["v_pred"]], axis=0)  # (3, N)
    v_spread_3s = np.std(v_stack_3s, axis=0, ddof=1)  # (N,)
    v_abs_err_3s = np.abs(ens_3seed["v_pred"] - ens_3seed["v_true"])
    v_mask = ens_3seed["v_mask"].astype(bool)
    spearman_wind_3s, pval_wind_3s = spearmanr(v_spread_3s[v_mask], v_abs_err_3s[v_mask])

    # Track +24h:
    # vector displacement spread: std along x and y components
    tr_stack_3s = np.stack([p42["tr_pred"], p43["tr_pred"], p44["tr_pred"]], axis=0)  # (3, N, 4, 2)
    # 24h is index 3
    tr24_stack_3s = tr_stack_3s[:, :, 3, :]  # (3, N, 2)
    # spread = sqrt(var(dx) + var(dy))
    dx_var_3s = np.var(tr24_stack_3s[:, :, 0], axis=0, ddof=1)
    dy_var_3s = np.var(tr24_stack_3s[:, :, 1], axis=0, ddof=1)
    tr24_spread_3s = np.sqrt(dx_var_3s + dy_var_3s)
    tr24_abs_err_3s = ens_3seed["track_err_nn"][:, 3]
    tr24_mask = ens_3seed["tr_mask"][:, 3].astype(bool)
    spearman_tr24_3s, pval_tr24_3s = spearmanr(tr24_spread_3s[tr24_mask], tr24_abs_err_3s[tr24_mask])

    # 2. Cross-Architecture Ensemble (R1-42 + S24-Ep7 + S24-Ep10)
    v_stack_cross = np.stack([p42["v_pred"], p_ep7["v_pred"], p_ep10["v_pred"]], axis=0)
    v_spread_cross = np.std(v_stack_cross, axis=0, ddof=1)
    v_abs_err_cross = np.abs(ens_cross["v_pred"] - ens_cross["v_true"])
    spearman_wind_cross, pval_wind_cross = spearmanr(v_spread_cross[v_mask], v_abs_err_cross[v_mask])

    tr_stack_cross = np.stack([p42["tr_pred"], p_ep7["tr_pred"], p_ep10["tr_pred"]], axis=0)
    tr24_stack_cross = tr_stack_cross[:, :, 3, :]
    dx_var_cross = np.var(tr24_stack_cross[:, :, 0], axis=0, ddof=1)
    dy_var_cross = np.var(tr24_stack_cross[:, :, 1], axis=0, ddof=1)
    tr24_spread_cross = np.sqrt(dx_var_cross + dy_var_cross)
    tr24_abs_err_cross = ens_cross["track_err_nn"][:, 3]
    spearman_tr24_cross, pval_tr24_cross = spearmanr(tr24_spread_cross[tr24_mask], tr24_abs_err_cross[tr24_mask])

    calibration_results = {
        "ensemble_r1_3seeds": {
            "wind": {
                "spearman_correlation": round(float(spearman_wind_3s), 4),
                "p_value": float(pval_wind_3s),
                "mean_spread_kt": round(float(np.mean(v_spread_3s[v_mask])), 3),
                "mean_abs_error_kt": round(float(np.mean(v_abs_err_3s[v_mask])), 3),
            },
            "track_24h": {
                "spearman_correlation": round(float(spearman_tr24_3s), 4),
                "p_value": float(pval_tr24_3s),
                "mean_spread_km": round(float(np.mean(tr24_spread_3s[tr24_mask])), 3),
                "mean_abs_error_km": round(float(np.mean(tr24_abs_err_3s[tr24_mask])), 3),
            },
        },
        "ensemble_cross_architecture": {
            "wind": {
                "spearman_correlation": round(float(spearman_wind_cross), 4),
                "p_value": float(pval_wind_cross),
                "mean_spread_kt": round(float(np.mean(v_spread_cross[v_mask])), 3),
                "mean_abs_error_kt": round(float(np.mean(v_abs_err_cross[v_mask])), 3),
            },
            "track_24h": {
                "spearman_correlation": round(float(spearman_tr24_cross), 4),
                "p_value": float(pval_tr24_cross),
                "mean_spread_km": round(float(np.mean(tr24_spread_cross[tr24_mask])), 3),
                "mean_abs_error_km": round(float(np.mean(tr24_abs_err_cross[tr24_mask])), 3),
            },
        },
    }

    # =========================================================================
    # Build Consolidated CSV and JSON Reports
    # =========================================================================
    table_rows = [
        {
            "model": "S24_Ep10 (Best Single Reference)",
            "global_v1_loss": res_s24_ep10["global_v1_reference_loss"],
            "val_mt_rel_err": res_s24_ep10["val_mt_rel_err"],
            "wind_mae_kt": res_s24_ep10["wind_mae_kt"],
            "wind_bias_kt": res_s24_ep10["wind_bias_kt"],
            "major_hurricane_mae_kt": res_s24_ep10["major_hurricane_mae_kt"],
            "major_hurricane_bias_kt": res_s24_ep10["major_hurricane_bias_kt"],
            "pressure_mae_hpa": res_s24_ep10["pressure_mae_hpa"],
            "balanced_stage_acc_pct": res_s24_ep10["balanced_stage_acc_pct"],
            "stage_acc_pct": res_s24_ep10["stage_acc_pct"],
            "dv_24h_mae_kt": res_s24_ep10["dv_24h_mae_kt"],
            "ri_pr_auc": res_s24_ep10["ri_pr_auc"],
            "ri_brier": res_s24_ep10["ri_brier"],
            "track_24h_km": res_s24_ep10["track_24h_km"],
            "track_24h_skill_vs_vel_km": res_s24_ep10["track_24h_skill_vs_causal_vel_km"],
        },
        {
            "model": "S24_Ep7 (Best Composite/RI Reference)",
            "global_v1_loss": res_s24_ep7["global_v1_reference_loss"],
            "val_mt_rel_err": res_s24_ep7["val_mt_rel_err"],
            "wind_mae_kt": res_s24_ep7["wind_mae_kt"],
            "wind_bias_kt": res_s24_ep7["wind_bias_kt"],
            "major_hurricane_mae_kt": res_s24_ep7["major_hurricane_mae_kt"],
            "major_hurricane_bias_kt": res_s24_ep7["major_hurricane_bias_kt"],
            "pressure_mae_hpa": res_s24_ep7["pressure_mae_hpa"],
            "balanced_stage_acc_pct": res_s24_ep7["balanced_stage_acc_pct"],
            "stage_acc_pct": res_s24_ep7["stage_acc_pct"],
            "dv_24h_mae_kt": res_s24_ep7["dv_24h_mae_kt"],
            "ri_pr_auc": res_s24_ep7["ri_pr_auc"],
            "ri_brier": res_s24_ep7["ri_brier"],
            "track_24h_km": res_s24_ep7["track_24h_km"],
            "track_24h_skill_vs_vel_km": res_s24_ep7["track_24h_skill_vs_causal_vel_km"],
        },
        {
            "model": "R1_Seed42",
            "global_v1_loss": res_s42["global_v1_reference_loss"],
            "val_mt_rel_err": res_s42["val_mt_rel_err"],
            "wind_mae_kt": res_s42["wind_mae_kt"],
            "wind_bias_kt": res_s42["wind_bias_kt"],
            "major_hurricane_mae_kt": res_s42["major_hurricane_mae_kt"],
            "major_hurricane_bias_kt": res_s42["major_hurricane_bias_kt"],
            "pressure_mae_hpa": res_s42["pressure_mae_hpa"],
            "balanced_stage_acc_pct": res_s42["balanced_stage_acc_pct"],
            "stage_acc_pct": res_s42["stage_acc_pct"],
            "dv_24h_mae_kt": res_s42["dv_24h_mae_kt"],
            "ri_pr_auc": res_s42["ri_pr_auc"],
            "ri_brier": res_s42["ri_brier"],
            "track_24h_km": res_s42["track_24h_km"],
            "track_24h_skill_vs_vel_km": res_s42["track_24h_skill_vs_causal_vel_km"],
        },
        {
            "model": "R1_Seed43",
            "global_v1_loss": res_s43["global_v1_reference_loss"],
            "val_mt_rel_err": res_s43["val_mt_rel_err"],
            "wind_mae_kt": res_s43["wind_mae_kt"],
            "wind_bias_kt": res_s43["wind_bias_kt"],
            "major_hurricane_mae_kt": res_s43["major_hurricane_mae_kt"],
            "major_hurricane_bias_kt": res_s43["major_hurricane_bias_kt"],
            "pressure_mae_hpa": res_s43["pressure_mae_hpa"],
            "balanced_stage_acc_pct": res_s43["balanced_stage_acc_pct"],
            "stage_acc_pct": res_s43["stage_acc_pct"],
            "dv_24h_mae_kt": res_s43["dv_24h_mae_kt"],
            "ri_pr_auc": res_s43["ri_pr_auc"],
            "ri_brier": res_s43["ri_brier"],
            "track_24h_km": res_s43["track_24h_km"],
            "track_24h_skill_vs_vel_km": res_s43["track_24h_skill_vs_causal_vel_km"],
        },
        {
            "model": "R1_Seed44",
            "global_v1_loss": res_s44["global_v1_reference_loss"],
            "val_mt_rel_err": res_s44["val_mt_rel_err"],
            "wind_mae_kt": res_s44["wind_mae_kt"],
            "wind_bias_kt": res_s44["wind_bias_kt"],
            "major_hurricane_mae_kt": res_s44["major_hurricane_mae_kt"],
            "major_hurricane_bias_kt": res_s44["major_hurricane_bias_kt"],
            "pressure_mae_hpa": res_s44["pressure_mae_hpa"],
            "balanced_stage_acc_pct": res_s44["balanced_stage_acc_pct"],
            "stage_acc_pct": res_s44["stage_acc_pct"],
            "dv_24h_mae_kt": res_s44["dv_24h_mae_kt"],
            "ri_pr_auc": res_s44["ri_pr_auc"],
            "ri_brier": res_s44["ri_brier"],
            "track_24h_km": res_s44["track_24h_km"],
            "track_24h_skill_vs_vel_km": res_s44["track_24h_skill_vs_causal_vel_km"],
        },
        {
            "model": "R1_Single_Seed_Mean±Std",
            "global_v1_loss": f"{single_seed_stats['global_v1_reference_loss']['mean']} ± {single_seed_stats['global_v1_reference_loss']['std']}",
            "val_mt_rel_err": f"{single_seed_stats['val_mt_rel_err']['mean']} ± {single_seed_stats['val_mt_rel_err']['std']}",
            "wind_mae_kt": f"{single_seed_stats['wind_mae_kt']['mean']} ± {single_seed_stats['wind_mae_kt']['std']}",
            "wind_bias_kt": f"{single_seed_stats['wind_bias_kt']['mean']} ± {single_seed_stats['wind_bias_kt']['std']}",
            "major_hurricane_mae_kt": f"{single_seed_stats['major_hurricane_mae_kt']['mean']} ± {single_seed_stats['major_hurricane_mae_kt']['std']}",
            "major_hurricane_bias_kt": f"{single_seed_stats['major_hurricane_bias_kt']['mean']} ± {single_seed_stats['major_hurricane_bias_kt']['std']}",
            "pressure_mae_hpa": f"{single_seed_stats['pressure_mae_hpa']['mean']} ± {single_seed_stats['pressure_mae_hpa']['std']}",
            "balanced_stage_acc_pct": f"{single_seed_stats['balanced_stage_acc_pct']['mean']} ± {single_seed_stats['balanced_stage_acc_pct']['std']}",
            "stage_acc_pct": f"{single_seed_stats['stage_acc_pct']['mean']} ± {single_seed_stats['stage_acc_pct']['std']}",
            "dv_24h_mae_kt": f"{single_seed_stats['dv_24h_mae_kt']['mean']} ± {single_seed_stats['dv_24h_mae_kt']['std']}",
            "ri_pr_auc": f"{single_seed_stats['ri_pr_auc']['mean']} ± {single_seed_stats['ri_pr_auc']['std']}",
            "ri_brier": f"{single_seed_stats['ri_brier']['mean']} ± {single_seed_stats['ri_brier']['std']}",
            "track_24h_km": f"{single_seed_stats['track_24h_km']['mean']} ± {single_seed_stats['track_24h_km']['std']}",
            "track_24h_skill_vs_vel_km": f"{single_seed_stats['track_24h_skill_vs_causal_vel_km']['mean']} ± {single_seed_stats['track_24h_skill_vs_causal_vel_km']['std']}",
        },
        {
            "model": "Ensemble_R1_3Seeds (42+43+44)",
            "global_v1_loss": res_ens_3seed["global_v1_reference_loss"],
            "val_mt_rel_err": res_ens_3seed["val_mt_rel_err"],
            "wind_mae_kt": res_ens_3seed["wind_mae_kt"],
            "wind_bias_kt": res_ens_3seed["wind_bias_kt"],
            "major_hurricane_mae_kt": res_ens_3seed["major_hurricane_mae_kt"],
            "major_hurricane_bias_kt": res_ens_3seed["major_hurricane_bias_kt"],
            "pressure_mae_hpa": res_ens_3seed["pressure_mae_hpa"],
            "balanced_stage_acc_pct": res_ens_3seed["balanced_stage_acc_pct"],
            "stage_acc_pct": res_ens_3seed["stage_acc_pct"],
            "dv_24h_mae_kt": res_ens_3seed["dv_24h_mae_kt"],
            "ri_pr_auc": res_ens_3seed["ri_pr_auc"],
            "ri_brier": res_ens_3seed["ri_brier"],
            "track_24h_km": res_ens_3seed["track_24h_km"],
            "track_24h_skill_vs_vel_km": res_ens_3seed["track_24h_skill_vs_causal_vel_km"],
        },
        {
            "model": "Ensemble_Cross_Arch (R1-42 + S24-Ep7 + S24-Ep10)",
            "global_v1_loss": res_ens_cross["global_v1_reference_loss"],
            "val_mt_rel_err": res_ens_cross["val_mt_rel_err"],
            "wind_mae_kt": res_ens_cross["wind_mae_kt"],
            "wind_bias_kt": res_ens_cross["wind_bias_kt"],
            "major_hurricane_mae_kt": res_ens_cross["major_hurricane_mae_kt"],
            "major_hurricane_bias_kt": res_ens_cross["major_hurricane_bias_kt"],
            "pressure_mae_hpa": res_ens_cross["pressure_mae_hpa"],
            "balanced_stage_acc_pct": res_ens_cross["balanced_stage_acc_pct"],
            "stage_acc_pct": res_ens_cross["stage_acc_pct"],
            "dv_24h_mae_kt": res_ens_cross["dv_24h_mae_kt"],
            "ri_pr_auc": res_ens_cross["ri_pr_auc"],
            "ri_brier": res_ens_cross["ri_brier"],
            "track_24h_km": res_ens_cross["track_24h_km"],
            "track_24h_skill_vs_vel_km": res_ens_cross["track_24h_skill_vs_causal_vel_km"],
        },
    ]

    out_csv = CANONICAL_DIR / "r1_seeds_and_ensembles_val_table.csv"
    pd.DataFrame(table_rows).to_csv(out_csv, index=False)
    print(f"\n[OK] Saved consolidated metrics table -> {out_csv}", flush=True)

    def _clean_model_res(res_dict):
        clean = {}
        for k, v in res_dict.items():
            if k in ("per_storm_table", "predictions"):
                continue
            if isinstance(v, (np.floating, float)):
                clean[k] = round(float(v), 4)
            elif isinstance(v, (np.integer, int)):
                clean[k] = int(v)
            elif isinstance(v, dict):
                clean[k] = _clean_model_res(v)
            else:
                clean[k] = v
        return clean

    summary_payload = {
        "models": {
            "s24_ep10": _clean_model_res(res_s24_ep10),
            "s24_ep7": _clean_model_res(res_s24_ep7),
            "r1_seed42": _clean_model_res(res_s42),
            "r1_seed43": _clean_model_res(res_s43),
            "r1_seed44": _clean_model_res(res_s44),
            "ensemble_r1_3seeds": _clean_model_res(res_ens_3seed),
            "ensemble_cross_architecture": _clean_model_res(res_ens_cross),
        },
        "single_seed_stats_mean_std": single_seed_stats,
        "bootstrap_vs_s24_ep10": {
            "ensemble_r1_3seeds_minus_s24ep10": bs_3seed_vs_s24ep10,
            "ensemble_cross_minus_s24ep10": bs_cross_vs_s24ep10,
            "ensemble_r1_3seeds_minus_r1_seed42": bs_3seed_vs_seed42,
            "ensemble_cross_minus_r1_seed42": bs_cross_vs_seed42,
        },
        "uncertainty_calibration_spearman": calibration_results,
    }

    out_json = CANONICAL_DIR / "r1_ensembles_bootstrap_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)
    print(f"[OK] Saved ensemble & bootstrap summary -> {out_json}", flush=True)

    return summary_payload


if __name__ == "__main__":
    # 1. Train Seed 43 and Seed 44
    ckpt_s43 = REPO_ROOT / "checkpoints/tcprimed_100gb_exp_r1_seed43/best_checkpoint.pt"
    ckpt_s44 = REPO_ROOT / "checkpoints/tcprimed_100gb_exp_r1_seed44/best_checkpoint.pt"

    if not (ckpt_s43.exists() and ckpt_s44.exists()):
        run_training_parallel()
    else:
        print("[INFO] Checkpoints for Seed 43 and Seed 44 already exist, skipping training.", flush=True)

    # 2. Evaluate and Ensemble
    evaluate_and_ensemble()

