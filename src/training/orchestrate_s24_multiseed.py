"""
Orchestration script for S24 multi-seed training, canonical evaluation,
and ensembling:
1. Concurrently trains S24 seeds 43 and 44 on GPU using exact recipe (hash c451dd564f7d, patience 5 on VAL loss)
2. Runs canonical_eval on each best-global-loss checkpoint
3. Builds the 3-seed S24 ensemble (Seed 42 Ep10 + Seed 43 + Seed 44)
4. Uses the exact canonical MultiTaskLoss path (with prior_shift and exact weights)
5. Computes single-seed spread (mean ± std across the 3 seeds)
6. Performs paired cluster storm bootstrap (N=2000) comparing the 3-seed S24 ensemble vs S24 Ep10
7. Exports reports/canonical/s24_seeds_and_ensembles_val_table.csv and summary JSON
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.canonical_eval import (
    evaluate,
    storm_bootstrap,
    save_canonical_predictions_npz,
    _compute_resample_metrics,
)
from src.training.losses import MultiTaskLoss

CANONICAL_DIR = REPO_ROOT / "reports/canonical"
PHASE_DIR = REPO_ROOT / "reports/phase2e4_100gb"
PYTHON_EXE = sys.executable


def compute_canonical_v1_loss_and_metrics_for_ensemble(ens_preds: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """
    Computes exact single-path canonical MultiTaskLoss and metrics for an ensemble.
    - Continuous outputs (v_pred, p_pred, dv_pred, tr_pred) are evaluated directly with Huber/SmoothL1.
    - Stage probabilities (s_prob) are inverted to logits via log(s_prob) for exact PyTorch CrossEntropy.
    - RI probabilities (ri_prob) are inverted to logits via log(p / (1-p)) for exact PyTorch BCEWithLogits,
      preserving the canonical prior_shift offset (0.85 * ln(18.10) ≈ 2.4615) and pos_weight (18.10).
    - Track loss is weighted by the canonical 1.0 factor (not 0.2).
    """
    idx = np.arange(len(ens_preds["sample_ids"]))
    m = _compute_resample_metrics(ens_preds, idx)

    loss_fn = MultiTaskLoss(
        weight_intensity=1.0,
        weight_pressure=0.5,
        weight_stage=0.5,
        weight_intensity_change=1.0,
        weight_ri=2.0,
        weight_track=1.0,
        ri_pos_weight=18.10,
        regime_max_extra_weight=2.5,
        regime_power=2.0,
        ri_prior_shift_factor=0.85,
        intensity_huber_beta=8.0,
        intensity_std_scale=1.0,
    )

    preds = {}
    preds["intensity"] = torch.from_numpy(ens_preds["v_pred"]).float()
    preds["pressure"] = torch.from_numpy(ens_preds["p_pred"]).float()
    
    # Invert stage probabilities to logits
    s_p = np.clip(ens_preds["s_prob"], 1e-12, 1.0)
    preds["stage_logits"] = torch.from_numpy(np.log(s_p)).float()
    
    preds["intensity_change"] = torch.from_numpy(ens_preds["dv_pred"]).float()
    
    # Invert RI probabilities to logits
    ri_p = np.clip(ens_preds["ri_prob"], 1e-7, 1.0 - 1e-7)
    preds["ri_logits"] = torch.from_numpy(np.log(ri_p / (1.0 - ri_p))).float()
    
    preds["track_delta"] = torch.from_numpy(ens_preds["tr_pred"]).float().view(-1, 8)

    targets = {}
    targets["intensity"] = torch.from_numpy(ens_preds["v_true"]).float()
    targets["intensity_mask"] = torch.from_numpy(ens_preds["v_mask"].astype(np.float32)).float()
    targets["pressure"] = torch.from_numpy(ens_preds["p_true"]).float()
    targets["pressure_mask"] = torch.from_numpy(ens_preds["p_mask"].astype(np.float32)).float()
    targets["stage"] = torch.from_numpy(ens_preds["s_true"]).long()
    targets["stage_mask"] = torch.from_numpy(ens_preds["s_mask"].astype(np.float32)).float()
    targets["intensity_change"] = torch.from_numpy(ens_preds["dv_true"]).float()
    targets["intensity_change_mask"] = torch.from_numpy(ens_preds["dv_mask"].astype(np.float32)).float()
    targets["rapid_intensification"] = torch.from_numpy(ens_preds["ri_true"]).float()
    targets["ri_mask"] = torch.from_numpy(ens_preds["ri_mask"].astype(np.float32)).float()
    targets["track_displacement"] = torch.from_numpy(ens_preds["tr_true"]).float()
    targets["track_mask"] = torch.from_numpy(ens_preds["tr_mask"].astype(np.float32)).float()

    total_loss, loss_dict = loss_fn(preds, targets)
    v1_global_loss = total_loss.item()

    # Composite VAL_MT_REL_ERR
    rel_wind = round(m["wind_mae_kt"], 2) / 24.71
    rel_press = round(m["pressure_mae_hpa"], 2) / 18.92
    rel_stage = (100.0 - round(m["stage_acc_pct"], 2)) / 78.05
    rel_dv = float(np.mean([round(m[f"dv_{h}h_mae_kt"], 2) for h in [6, 12, 18, 24]])) / 8.1075
    rel_ri = 0.5 * ((1.0 - round(m["ri_pr_auc"], 4)) / 0.9564 + (round(m["ri_brier"], 4) / 0.0550))
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
            "weighted_loss_intensity": round(float(loss_dict.get("weighted_loss_intensity", 0.0)), 4),
            "weighted_loss_pressure": round(float(loss_dict.get("weighted_loss_pressure", 0.0)), 4),
            "weighted_loss_stage": round(float(loss_dict.get("weighted_loss_stage", 0.0)), 4),
            "weighted_loss_intensity_change": round(float(loss_dict.get("weighted_loss_intensity_change", 0.0)), 4),
            "weighted_loss_ri": round(float(loss_dict.get("weighted_loss_ri", 0.0)), 4),
            "weighted_loss_track": round(float(loss_dict.get("weighted_loss_track", 0.0)), 4),
        },
    }
    return res


def run_training_parallel():
    """Spawn Seeds 43 and 44 concurrently and stream status."""
    cfg_s43 = REPO_ROOT / "configs/exp_s24_seed43.yaml"
    cfg_s44 = REPO_ROOT / "configs/exp_s24_seed44.yaml"
    log_s43_path = PHASE_DIR / "exp_s24_seed43_train.log"
    log_s44_path = PHASE_DIR / "exp_s24_seed44_train.log"

    print("\n" + "=" * 70)
    print("CONCURRENT TRAINING: S24 SEED 43 & SEED 44")
    print(f"Seed 43 Config: {cfg_s43}")
    print(f"Seed 44 Config: {cfg_s44}")
    print(f"Python: {PYTHON_EXE}")
    print("=" * 70, flush=True)

    log_s43 = open(log_s43_path, "w", encoding="utf-8")
    log_s44 = open(log_s44_path, "w", encoding="utf-8")

    cmd43 = [PYTHON_EXE, "-u", "src/training/train_multitask.py", "--config", str(cfg_s43)]
    cmd44 = [PYTHON_EXE, "-u", "src/training/train_multitask.py", "--config", str(cfg_s44)]

    p43 = subprocess.Popen(cmd43, stdout=log_s43, stderr=subprocess.STDOUT, cwd=str(REPO_ROOT))
    p44 = subprocess.Popen(cmd44, stdout=log_s44, stderr=subprocess.STDOUT, cwd=str(REPO_ROOT))

    start_time = time.time()
    last_reported_ep43 = -1
    last_reported_ep44 = -1

    epochs_csv43 = PHASE_DIR / "exp_s24_seed43_epochs.csv"
    epochs_csv44 = PHASE_DIR / "exp_s24_seed44_epochs.csv"

    while True:
        ret43 = p43.poll()
        ret44 = p44.poll()

        ep43 = 0
        ep44 = 0
        if epochs_csv43.exists():
            try:
                df43 = pd.read_csv(epochs_csv43)
                if not df43.empty:
                    ep43 = int(df43["epoch"].iloc[-1])
            except Exception:
                pass

        if epochs_csv44.exists():
            try:
                df44 = pd.read_csv(epochs_csv44)
                if not df44.empty:
                    ep44 = int(df44["epoch"].iloc[-1])
            except Exception:
                pass

        elapsed_min = (time.time() - start_time) / 60.0
        if ep43 != last_reported_ep43 or ep44 != last_reported_ep44 or int(elapsed_min) % 5 == 0:
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

        time.sleep(20)

    log_s43.close()
    log_s44.close()

    if ret43 != 0:
        raise RuntimeError(f"Seed 43 training failed with exit code {ret43}! Check {log_s43_path}")
    if ret44 != 0:
        raise RuntimeError(f"Seed 44 training failed with exit code {ret44}! Check {log_s44_path}")

    print(f"\n[DONE] Both S24 seeds completed successfully in {(time.time() - start_time)/60.0:.1f} minutes!", flush=True)


def evaluate_and_ensemble():
    """Evaluate individual seeds, form the 3-seed ensemble, and run paired bootstrap."""
    ckpt_s42 = REPO_ROOT / "checkpoints/tcprimed_100gb_exp_s24/best_checkpoint.pt"
    ckpt_s43 = REPO_ROOT / "checkpoints/tcprimed_100gb_exp_s24_seed43/best_checkpoint.pt"
    ckpt_s44 = REPO_ROOT / "checkpoints/tcprimed_100gb_exp_s24_seed44/best_checkpoint.pt"

    print("\n" + "=" * 70)
    print("CANONICAL EVALUATION OF S24 SEEDS")
    print("=" * 70)

    print(f"\n[EVAL] Running canonical evaluation on S24 Seed 43 ({ckpt_s43})...")
    res_s43 = evaluate(ckpt_s43, split="val", use_ema=False)
    save_canonical_predictions_npz(res_s43, CANONICAL_DIR / "s24_seed43_val_predictions.npz")

    print(f"\n[EVAL] Running canonical evaluation on S24 Seed 44 ({ckpt_s44})...")
    res_s44 = evaluate(ckpt_s44, split="val", use_ema=False)
    save_canonical_predictions_npz(res_s44, CANONICAL_DIR / "s24_seed44_val_predictions.npz")

    # Load frozen S24 Ep 10 predictions
    p42 = dict(np.load(CANONICAL_DIR / "s24_ep10_val_predictions.npz"))
    p43 = res_s43["predictions"]
    p44 = res_s44["predictions"]

    # Load res_s42 from canonical table if possible
    val_table = pd.read_csv(CANONICAL_DIR / "val_table.csv")
    s24_ep10_row = val_table[val_table["model"] == "S24_Ep10"].iloc[0].to_dict()

    # Build 3-Seed S24 Ensemble (Seed 42 Ep10 + Seed 43 + Seed 44)
    print("\n[ENSEMBLE] Building 3-Seed S24 Ensemble (Seed 42 Ep10 + Seed 43 + Seed 44)...", flush=True)
    ens_3s = dict(p42)
    ens_3s["v_pred"] = (p42["v_pred"] + p43["v_pred"] + p44["v_pred"]) / 3.0
    ens_3s["p_pred"] = (p42["p_pred"] + p43["p_pred"] + p44["p_pred"]) / 3.0
    ens_3s["dv_pred"] = (p42["dv_pred"] + p43["dv_pred"] + p44["dv_pred"]) / 3.0
    ens_3s["tr_pred"] = (p42["tr_pred"] + p43["tr_pred"] + p44["tr_pred"]) / 3.0
    ens_3s["track_err_nn"] = np.sqrt(np.sum((ens_3s["tr_pred"] - ens_3s["tr_true"]) ** 2, axis=-1))
    ens_3s["ri_prob"] = (p42["ri_prob"] + p43["ri_prob"] + p44["ri_prob"]) / 3.0
    ens_3s["s_prob"] = (p42["s_prob"] + p43["s_prob"] + p44["s_prob"]) / 3.0
    ens_3s["s_pred"] = np.argmax(ens_3s["s_prob"], axis=-1)

    np.savez_compressed(str(CANONICAL_DIR / "ensemble_s24_3seeds_val_predictions.npz"), **ens_3s)
    res_ens = compute_canonical_v1_loss_and_metrics_for_ensemble(ens_3s)

    # Compute Single-Seed Spread: Mean ± Std Across Seeds 42, 43, 44
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
        v42 = float(s24_ep10_row.get(m, 0.0))
        v43 = float(res_s43[m])
        v44 = float(res_s44[m])
        vals = [v42, v43, v44]
        single_seed_stats[m] = {
            "seed_42_ep10": v42,
            "seed_43": v43,
            "seed_44": v44,
            "mean": round(float(np.mean(vals)), 4),
            "std": round(float(np.std(vals, ddof=1)), 4),
            "spread_max_minus_min": round(float(max(vals) - min(vals)), 4),
        }

    # Run Paired Cluster Storm Bootstrap (N=2000, seed=0) vs S24 Ep10
    print("\n[BOOTSTRAP] Running 2000 paired storm resamples (Ensemble vs S24 Ep10)...", flush=True)
    bs_ens_vs_ep10 = storm_bootstrap(ens_3s, p42, n=2000, seed=0)

    # Save summary table
    table_rows = [
        {
            "model": "S24_Seed42_Ep10 (Reference)",
            "global_v1_loss": s24_ep10_row["global_v1_reference_loss"],
            "val_mt_rel_err": s24_ep10_row["val_mt_rel_err"],
            "wind_mae_kt": s24_ep10_row["wind_mae_kt"],
            "wind_bias_kt": s24_ep10_row["wind_bias_kt"],
            "major_hurricane_mae_kt": s24_ep10_row["major_hurricane_mae_kt"],
            "major_hurricane_bias_kt": s24_ep10_row["major_hurricane_bias_kt"],
            "pressure_mae_hpa": s24_ep10_row["pressure_mae_hpa"],
            "balanced_stage_acc_pct": s24_ep10_row["balanced_stage_acc_pct"],
            "stage_acc_pct": s24_ep10_row["stage_acc_pct"],
            "dv_24h_mae_kt": s24_ep10_row["dv_24h_mae_kt"],
            "ri_pr_auc": s24_ep10_row["ri_pr_auc"],
            "ri_brier": s24_ep10_row["ri_brier"],
            "track_24h_km": s24_ep10_row["track_24h_km"],
            "track_24h_skill_vs_vel_km": s24_ep10_row["track_24h_skill_vs_causal_vel_km"],
        },
        {
            "model": "S24_Seed43 (Best Global Loss)",
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
            "model": "S24_Seed44 (Best Global Loss)",
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
            "model": "S24_Single_Seed_Mean±Std",
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
            "model": "Ensemble_S24_3Seeds (42+43+44)",
            "global_v1_loss": res_ens["global_v1_reference_loss"],
            "val_mt_rel_err": res_ens["val_mt_rel_err"],
            "wind_mae_kt": res_ens["wind_mae_kt"],
            "wind_bias_kt": res_ens["wind_bias_kt"],
            "major_hurricane_mae_kt": res_ens["major_hurricane_mae_kt"],
            "major_hurricane_bias_kt": res_ens["major_hurricane_bias_kt"],
            "pressure_mae_hpa": res_ens["pressure_mae_hpa"],
            "balanced_stage_acc_pct": res_ens["balanced_stage_acc_pct"],
            "stage_acc_pct": res_ens["stage_acc_pct"],
            "dv_24h_mae_kt": res_ens["dv_24h_mae_kt"],
            "ri_pr_auc": res_ens["ri_pr_auc"],
            "ri_brier": res_ens["ri_brier"],
            "track_24h_km": res_ens["track_24h_km"],
            "track_24h_skill_vs_vel_km": res_ens["track_24h_skill_vs_causal_vel_km"],
        },
    ]

    df_out = pd.DataFrame(table_rows)
    df_out.to_csv(CANONICAL_DIR / "s24_seeds_and_ensembles_val_table.csv", index=False)
    print("\n[TABLE] Saved reports/canonical/s24_seeds_and_ensembles_val_table.csv:")
    print(df_out.to_string())

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    # Save summary JSON
    summary_payload = {
        "models": {
            "s24_seed42_ep10": s24_ep10_row,
            "s24_seed43_best": {k: v for k, v in res_s43.items() if not isinstance(v, (np.ndarray, pd.DataFrame)) and k != "predictions"},
            "s24_seed44_best": {k: v for k, v in res_s44.items() if not isinstance(v, (np.ndarray, pd.DataFrame)) and k != "predictions"},
            "ensemble_s24_3seeds": res_ens,
        },
        "single_seed_stats_mean_std": single_seed_stats,
        "bootstrap_vs_s24_ep10": bs_ens_vs_ep10["paired_deltas_a_minus_b"],
    }

    with open(CANONICAL_DIR / "s24_ensembles_bootstrap_summary.json", "w") as f:
        json.dump(summary_payload, f, indent=2, cls=NumpyEncoder)
    print("\n[JSON] Saved reports/canonical/s24_ensembles_bootstrap_summary.json")


def main():
    run_training_parallel()
    evaluate_and_ensemble()


if __name__ == "__main__":
    main()

