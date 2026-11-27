"""
Build uniform weight averages (SWA) for S24:
  W1: epochs 7-12 (6 checkpoints)
  W2: epochs 6-15 (10 checkpoints)
Recomputes BatchNorm running statistics on TRAIN partition (no labels).
Runs canonical evaluation and paired storm bootstrap vs S24 Ep10 and S24 Ep7.
"""

import os
import sys
import copy
import time
import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.tc_graphformer import TCGraphFormer
from src.data.dataset import TCPRIMEDSequenceDataset
from src.training.train_multitask import collate_sequence_batch
from src.evaluation.canonical_eval import evaluate, storm_bootstrap, save_canonical_predictions_npz

S24_CKPT_DIR = REPO_ROOT / "checkpoints/tcprimed_100gb_exp_s24"
SWA_OUTPUT_DIR = REPO_ROOT / "checkpoints/tcprimed_100gb_exp_s24_swa"
CANONICAL_DIR = REPO_ROOT / "reports/canonical"
SAMPLE_INDEX_CSV = REPO_ROOT / "reports/pipeline_100gb/sample_index.csv"


def build_uniform_weight_average(checkpoint_paths: List[Path]) -> Dict[str, torch.Tensor]:
    """Compute uniform arithmetic average of model_state_dict weights across checkpoints."""
    print(f"\n[SWA] Averaging {len(checkpoint_paths)} checkpoints:")
    for p in checkpoint_paths:
        print(f"  - {p.name}")

    avg_state_dict: Dict[str, torch.Tensor] = {}
    n_ckpts = len(checkpoint_paths)

    for idx, ckpt_path in enumerate(checkpoint_paths):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt["model_state_dict"]

        for k, v in sd.items():
            if idx == 0:
                avg_state_dict[k] = v.clone().float() if v.is_floating_point() else v.clone()
            else:
                if v.is_floating_point():
                    avg_state_dict[k] += v.float()
                else:
                    # Non-floating point tensors (e.g. num_batches_tracked)
                    avg_state_dict[k] = v.clone()

    for k, v in avg_state_dict.items():
        if v.is_floating_point():
            avg_state_dict[k] = (v / float(n_ckpts)).to(sd[k].dtype)

    print(f"[SWA] Averaged {len(avg_state_dict)} tensor keys successfully.")
    return avg_state_dict


def recompute_batchnorm_stats(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
) -> None:
    """
    Recompute running_mean and running_var for all BatchNorm layers on TRAIN inputs.
    Uses cumulative moving average (momentum=None) over the full training set without labels.
    """
    bn_modules = [m for m in model.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))]
    print(f"[BN] Found {len(bn_modules)} BatchNorm modules to update.")

    if not bn_modules:
        print("[BN] No BatchNorm modules found; skipping.")
        return

    # Reset and store momenta
    saved_momenta = {}
    for m in bn_modules:
        m.running_mean = torch.zeros_like(m.running_mean)
        m.running_var = torch.ones_like(m.running_var)
        saved_momenta[m] = m.momentum
        m.momentum = None
        if hasattr(m, "num_batches_tracked") and m.num_batches_tracked is not None:
            m.num_batches_tracked.zero_()

    model.train()
    print("[BN] Running forward pass on TRAIN partition to recompute running statistics...", flush=True)
    t0 = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(train_loader):
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

            extra_kwargs = {}
            if kin_s is not None and hasattr(model, "_encode_kinematic_context"):
                extra_kwargs["kinematic_state"] = kin_s

            # Forward pass only, no target computation or loss
            _ = model(
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
                **extra_kwargs,
            )

            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"  [BN Pass] Batch {batch_idx + 1}/{len(train_loader)} ({time.time() - t0:.1f}s)", flush=True)

    # Restore original momenta
    for m in bn_modules:
        m.momentum = saved_momenta[m]

    model.eval()
    print(f"[BN] BatchNorm statistics successfully recomputed in {time.time() - t0:.1f}s.")


def main():
    SWA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CANONICAL_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INIT] Using device: {device}")

    # 1. Define pre-declared windows
    # Window 1: Epochs 7-12 (6 checkpoints)
    w1_epochs = list(range(7, 13))
    w1_paths = [S24_CKPT_DIR / f"s24_multitask_epoch_{ep:03d}.pt" for ep in w1_epochs]

    # Window 2: Epochs 6-15 (10 checkpoints)
    w2_epochs = list(range(6, 16))
    w2_paths = [S24_CKPT_DIR / f"s24_multitask_epoch_{ep:03d}.pt" for ep in w2_epochs]

    for p in w1_paths + w2_paths:
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint file missing: {p}")

    # 2. Build dataloader for TRAIN partition (batch size 32)
    print("\n[DATASET] Loading TRAIN sequence dataset for BatchNorm recomputation...")
    train_dataset = TCPRIMEDSequenceDataset(
        sample_index_csv=str(SAMPLE_INDEX_CSV),
        partition="TRAIN",
        grid_size=64,
        max_history_len=4,
        max_history_hours=72.0,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_sequence_batch,
        pin_memory=torch.cuda.is_available(),
    )

    # 3. Process Window W1 (epochs 7-12)
    print("\n" + "=" * 70)
    print("PROCESSING WINDOW W1: EPOCHS 7-12 (UNIFORM SWA)")
    print("=" * 70)
    w1_avg_sd = build_uniform_weight_average(w1_paths)
    model_w1 = TCGraphFormer(d_model=128).to(device)
    model_w1.load_state_dict(w1_avg_sd)
    recompute_batchnorm_stats(model_w1, train_loader, device)

    ckpt_w1_path = SWA_OUTPUT_DIR / "swa_w1_ep7_12.pt"
    torch.save(
        {
            "model_state_dict": model_w1.state_dict(),
            "d_model": 128,
            "window": "epochs 7-12",
            "epochs_averaged": w1_epochs,
            "num_epochs": len(w1_epochs),
            "is_swa": True,
            "bn_recomputed": True,
            "provenance": "S24 checkpoints 7-12 uniform SWA with TRAIN BatchNorm update",
        },
        ckpt_w1_path,
    )
    print(f"[SAVE] Saved W1 SWA checkpoint: {ckpt_w1_path}")

    # 4. Process Window W2 (epochs 6-15)
    print("\n" + "=" * 70)
    print("PROCESSING WINDOW W2: EPOCHS 6-15 (UNIFORM SWA)")
    print("=" * 70)
    w2_avg_sd = build_uniform_weight_average(w2_paths)
    model_w2 = TCGraphFormer(d_model=128).to(device)
    model_w2.load_state_dict(w2_avg_sd)
    recompute_batchnorm_stats(model_w2, train_loader, device)

    ckpt_w2_path = SWA_OUTPUT_DIR / "swa_w2_ep6_15.pt"
    torch.save(
        {
            "model_state_dict": model_w2.state_dict(),
            "d_model": 128,
            "window": "epochs 6-15",
            "epochs_averaged": w2_epochs,
            "num_epochs": len(w2_epochs),
            "is_swa": True,
            "bn_recomputed": True,
            "provenance": "S24 checkpoints 6-15 uniform SWA with TRAIN BatchNorm update",
        },
        ckpt_w2_path,
    )
    print(f"[SAVE] Saved W2 SWA checkpoint: {ckpt_w2_path}")

    # 5. Canonical Evaluation on VAL
    print("\n" + "=" * 70)
    print("CANONICAL EVALUATION ON VAL PARTITION")
    print("=" * 70)
    print("\n[EVAL] Running canonical_eval for W1 (Epochs 7-12)...")
    res_w1 = evaluate(ckpt_w1_path, split="val", use_ema=False)
    save_canonical_predictions_npz(res_w1, CANONICAL_DIR / "s24_swa_w1_val_predictions.npz")

    print("\n[EVAL] Running canonical_eval for W2 (Epochs 6-15)...")
    res_w2 = evaluate(ckpt_w2_path, split="val", use_ema=False)
    save_canonical_predictions_npz(res_w2, CANONICAL_DIR / "s24_swa_w2_val_predictions.npz")

    # Load frozen reference predictions
    p_s24_ep10 = dict(np.load(CANONICAL_DIR / "s24_ep10_val_predictions.npz"))
    p_s24_ep7 = dict(np.load(CANONICAL_DIR / "s24_ep7_val_predictions.npz"))

    p_w1 = res_w1["predictions"]
    p_w2 = res_w2["predictions"]

    # 6. Paired Cluster Storm Bootstrap (N=2000, seed=0)
    print("\n" + "=" * 70)
    print("PAIRED CLUSTER STORM BOOTSTRAP (N=2000, Seed=0)")
    print("=" * 70)
    print("\n[BOOTSTRAP] W1 vs S24 Ep10...")
    bs_w1_vs_ep10 = storm_bootstrap(p_w1, p_s24_ep10, n=2000, seed=0)

    print("\n[BOOTSTRAP] W1 vs S24 Ep7...")
    bs_w1_vs_ep7 = storm_bootstrap(p_w1, p_s24_ep7, n=2000, seed=0)

    print("\n[BOOTSTRAP] W2 vs S24 Ep10...")
    bs_w2_vs_ep10 = storm_bootstrap(p_w2, p_s24_ep10, n=2000, seed=0)

    print("\n[BOOTSTRAP] W2 vs S24 Ep7...")
    bs_w2_vs_ep7 = storm_bootstrap(p_w2, p_s24_ep7, n=2000, seed=0)

    print("\n[BOOTSTRAP] W1 vs W2...")
    bs_w1_vs_w2 = storm_bootstrap(p_w1, p_w2, n=2000, seed=0)

    # 7. Print consolidated table
    summary_data = {
        "W1_SWA_Ep7_12": {
            "global_v1_loss": res_w1["global_v1_reference_loss"],
            "val_mt_rel_err": res_w1["val_mt_rel_err"],
            "wind_mae_kt": res_w1["wind_mae_kt"],
            "wind_bias_kt": res_w1["wind_bias_kt"],
            "major_mae_kt": res_w1["major_hurricane_mae_kt"],
            "major_bias_kt": res_w1["major_hurricane_bias_kt"],
            "pressure_mae_hpa": res_w1["pressure_mae_hpa"],
            "stage_acc_pct": res_w1["stage_acc_pct"],
            "balanced_stage_acc_pct": res_w1["balanced_stage_acc_pct"],
            "ri_pr_auc": res_w1["ri_pr_auc"],
            "ri_brier": res_w1["ri_brier"],
            "track_24h_km": res_w1["track_24h_km"],
            "track_24h_skill_vs_vel_km": res_w1["track_24h_skill_vs_causal_vel_km"],
        },
        "W2_SWA_Ep6_15": {
            "global_v1_loss": res_w2["global_v1_reference_loss"],
            "val_mt_rel_err": res_w2["val_mt_rel_err"],
            "wind_mae_kt": res_w2["wind_mae_kt"],
            "wind_bias_kt": res_w2["wind_bias_kt"],
            "major_mae_kt": res_w2["major_hurricane_mae_kt"],
            "major_bias_kt": res_w2["major_hurricane_bias_kt"],
            "pressure_mae_hpa": res_w2["pressure_mae_hpa"],
            "stage_acc_pct": res_w2["stage_acc_pct"],
            "balanced_stage_acc_pct": res_w2["balanced_stage_acc_pct"],
            "ri_pr_auc": res_w2["ri_pr_auc"],
            "ri_brier": res_w2["ri_brier"],
            "track_24h_km": res_w2["track_24h_km"],
            "track_24h_skill_vs_vel_km": res_w2["track_24h_skill_vs_causal_vel_km"],
        },
    }

    # Save summary JSON
    bootstrap_out = {
        "models": summary_data,
        "bootstrap": {
            "w1_vs_s24_ep10": bs_w1_vs_ep10["paired_deltas_a_minus_b"],
            "w1_vs_s24_ep7": bs_w1_vs_ep7["paired_deltas_a_minus_b"],
            "w2_vs_s24_ep10": bs_w2_vs_ep10["paired_deltas_a_minus_b"],
            "w2_vs_s24_ep7": bs_w2_vs_ep7["paired_deltas_a_minus_b"],
            "w1_vs_w2": bs_w1_vs_w2["paired_deltas_a_minus_b"],
        },
    }

    with open(CANONICAL_DIR / "s24_swa_evaluation_summary.json", "w") as f:
        json.dump(bootstrap_out, f, indent=2)

    print("\n[DONE] SWA execution, evaluation, and bootstrap analysis complete!")


if __name__ == "__main__":
    main()
