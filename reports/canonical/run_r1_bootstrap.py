import json
from pathlib import Path
import numpy as np
import pandas as pd

from src.evaluation.canonical_eval import evaluate, storm_bootstrap, save_canonical_predictions_npz

out_dir = Path("reports/canonical")

# Checkpoint paths
ckpt_r1 = "checkpoints/tcprimed_100gb_exp_r1/best_checkpoint.pt"

# Re-run or load evaluate on R1 to get complete task loss dict
print("Evaluating R1 best_checkpoint.pt...", flush=True)
res_r1 = evaluate(ckpt_r1, split="val", use_ema=False)
npz_path = out_dir / "r1_ep24_val_predictions.npz"
save_canonical_predictions_npz(res_r1, npz_path)

storm_csv = out_dir / "r1_ep24_val_per_storm.csv"
res_r1["per_storm_table"].to_csv(storm_csv, index=False)

# Load prediction npz dicts for baseline models
preds_r1 = dict(np.load(out_dir / "r1_ep24_val_predictions.npz"))
preds_s24_ep10 = dict(np.load(out_dir / "s24_ep10_val_predictions.npz"))
preds_s24_ep7 = dict(np.load(out_dir / "s24_ep7_val_predictions.npz"))
preds_orig_ep5 = dict(np.load(out_dir / "original_ep5_val_predictions.npz"))

print("Running paired storm bootstrap (n=2000, seed=0)...", flush=True)
bs_r1_vs_s24_ep10 = storm_bootstrap(preds_r1, preds_s24_ep10, n=2000, seed=0)
bs_r1_vs_s24_ep7 = storm_bootstrap(preds_r1, preds_s24_ep7, n=2000, seed=0)
bs_r1_vs_orig_ep5 = storm_bootstrap(preds_r1, preds_orig_ep5, n=2000, seed=0)

bootstrap_results = {
    "R1_Ep24_minus_S24_Ep10": bs_r1_vs_s24_ep10,
    "R1_Ep24_minus_S24_Ep7": bs_r1_vs_s24_ep7,
    "R1_Ep24_minus_Original_Ep5": bs_r1_vs_orig_ep5,
}

print("\n=== Bootstrap: R1 Ep24 minus S24 Ep10 ===")
for m in ["wind_mae_kt", "major_hurricane_mae_kt", "pressure_mae_hpa", "balanced_stage_acc_pct", "dv_24h_mae_kt", "ri_pr_auc", "ri_brier", "track_24h_km"]:
    v = bs_r1_vs_s24_ep10["paired_deltas_a_minus_b"][m]
    print(f"  {m}: delta = {v['boot_mean_delta']:+.3f} (95% CI [{v['ci95_low']:+.3f}, {v['ci95_high']:+.3f}])")

print("\n=== Bootstrap: R1 Ep24 minus S24 Ep7 ===")
for m in ["wind_mae_kt", "major_hurricane_mae_kt", "pressure_mae_hpa", "balanced_stage_acc_pct", "dv_24h_mae_kt", "ri_pr_auc", "ri_brier", "track_24h_km"]:
    v = bs_r1_vs_s24_ep7["paired_deltas_a_minus_b"][m]
    print(f"  {m}: delta = {v['boot_mean_delta']:+.3f} (95% CI [{v['ci95_low']:+.3f}, {v['ci95_high']:+.3f}])")

print("\n=== Bootstrap: R1 Ep24 minus Original Ep5 ===")
for m in ["wind_mae_kt", "major_hurricane_mae_kt", "pressure_mae_hpa", "balanced_stage_acc_pct", "dv_24h_mae_kt", "ri_pr_auc", "ri_brier", "track_24h_km"]:
    v = bs_r1_vs_orig_ep5["paired_deltas_a_minus_b"][m]
    print(f"  {m}: delta = {v['boot_mean_delta']:+.3f} (95% CI [{v['ci95_low']:+.3f}, {v['ci95_high']:+.3f}])")

# Append/Update R1_Ep24 in val_table.csv
val_table_csv = out_dir / "val_table.csv"
val_df = pd.read_csv(val_table_csv)

r1_row = {
    "model": "R1_Ep24_Regularized",
    "checkpoint_path": ckpt_r1,
    "epoch": res_r1["epoch"],
    "global_v1_reference_loss": res_r1["global_v1_reference_loss"],
    "batch_avg_v1_loss": res_r1["batch_avg_v1_loss"],
    "val_mt_rel_err": res_r1["val_mt_rel_err"],
    "wind_mae_kt": res_r1["wind_mae_kt"],
    "wind_bias_kt": res_r1["wind_bias_kt"],
    "major_hurricane_mae_kt": res_r1["major_hurricane_mae_kt"],
    "major_hurricane_bias_kt": res_r1["major_hurricane_bias_kt"],
    "pressure_mae_hpa": res_r1["pressure_mae_hpa"],
    "balanced_stage_acc_pct": res_r1["balanced_stage_acc_pct"],
    "stage_acc_pct": res_r1["stage_acc_pct"],
    "dv_6h_mae_kt": res_r1["dv_6h_mae_kt"],
    "dv_12h_mae_kt": res_r1["dv_12h_mae_kt"],
    "dv_18h_mae_kt": res_r1["dv_18h_mae_kt"],
    "dv_24h_mae_kt": res_r1["dv_24h_mae_kt"],
    "ri_pr_auc": res_r1["ri_pr_auc"],
    "ri_brier": res_r1["ri_brier"],
    "ri_positive_count": res_r1["ri_positive_count"],
    "track_24h_km": res_r1["track_24h_km"],
    "track_24h_causal_vel_baseline_km": res_r1["track_24h_causal_vel_baseline_km"],
    "track_24h_skill_vs_causal_vel_km": res_r1["track_24h_skill_vs_causal_vel_km"],
    "weighted_loss_intensity": res_r1["task_losses_weighted"]["weighted_loss_intensity"],
    "weighted_loss_pressure": res_r1["task_losses_weighted"]["weighted_loss_pressure"],
    "weighted_loss_stage": res_r1["task_losses_weighted"]["weighted_loss_stage"],
    "weighted_loss_intensity_change": res_r1["task_losses_weighted"]["weighted_loss_intensity_change"],
    "weighted_loss_ri": res_r1["task_losses_weighted"]["weighted_loss_ri"],
    "weighted_loss_track": res_r1["task_losses_weighted"]["weighted_loss_track"],
}

# Replace if existing or append
val_df = val_df[val_df["model"] != "R1_Ep24_Regularized"]
val_df = pd.concat([val_df, pd.DataFrame([r1_row])], ignore_index=True)
val_df.to_csv(val_table_csv, index=False)
print(f"\nUpdated {val_table_csv} with R1_Ep24_Regularized.")

# Update canonical_val_summary.json
sum_json_path = out_dir / "canonical_val_summary.json"
if sum_json_path.exists():
    with open(sum_json_path, "r", encoding="utf-8") as f:
        summary_data = json.load(f)

    # Update val_table list
    summary_data["val_table"] = val_df.to_dict(orient="records")
    for k, v in bootstrap_results.items():
        summary_data["storm_bootstrap_n2000_seed0"][k] = v

    with open(sum_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print("Updated canonical_val_summary.json successfully.")

# Save standalone bootstrap comparison JSON
out_bs_file = out_dir / "r1_ep24_bootstrap_comparisons.json"
with open(out_bs_file, "w", encoding="utf-8") as f:
    json.dump(bootstrap_results, f, indent=2)
print(f"Saved {out_bs_file}")
