"""
Task-Specific Fine-Tuning and Domain Adaptation Workflow for TC-GraphFormer.
Supports:
1. Intensity Change & Rapid Intensification focused fine-tuning.
2. North Indian Ocean (NIO / basin='IO') domain adaptation.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import TCPRIMEDDataset
from src.data.collate import collate_multimodal_batch
from src.models.tc_graphformer import TCGraphFormer
from src.training.losses import MultiTaskLoss
from src.training.trainer import TCGraphFormerTrainer


def run_finetuning(
    base_checkpoint: str = "checkpoints/best_checkpoint.pt",
    task: str = "intensity_change", # "intensity_change" or "nio_adaptation"
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 5e-5,
    d_model: int = 128,
    max_samples: int = None,
    freeze_encoders: bool = True,
    sample_index_csv: str = "reports/pipeline/sample_index.csv",
    checkpoint_dir: str = "checkpoints",
    output_report: str = None,
    device: str = None,
) -> Dict[str, Any]:
    """
    Execute task-specific fine-tuning or basin adaptation.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if output_report is None:
        output_report = f"reports/finetuning_{task}_summary.json"

    print("=" * 60)
    print(f"TC-GRAPHFORMER FINE-TUNING: Task = '{task}'")
    print(f"Base Checkpoint: {base_checkpoint}")
    print(f"Device: {device} | Epochs: {epochs} | LR: {lr} | Freeze Encoders: {freeze_encoders}")
    print("=" * 60)

    # 1. Dataset filtering based on task
    df = pd.read_csv(sample_index_csv)

    if task == "nio_adaptation":
        # Domain adaptation for North Indian Ocean (basin == 'IO')
        df_train = df[(df["basin"] == "IO") & (df["partition"] == "TRAIN")].copy()
        df_val = df[(df["basin"] == "IO") & (df["partition"] == "VAL")].copy()
        print(f"[DATA] NIO Domain Adaptation: {len(df_train)} train samples, {len(df_val)} val samples")
    else:
        # Standard partitions
        df_train = df[df["partition"] == "TRAIN"].copy()
        df_val = df[df["partition"] == "VAL"].copy()

    if max_samples is not None and max_samples > 0:
        df_train = df_train.iloc[:max_samples].copy()
        df_val = df_val.iloc[:max(4, max_samples // 4)].copy()

    # Save temporary filtered indices if needed, or instantiate dataset directly
    train_ds = TCPRIMEDDataset(sample_index_csv=sample_index_csv, partition="TRAIN")
    train_ds.df = df_train.reset_index(drop=True)

    val_ds = TCPRIMEDDataset(sample_index_csv=sample_index_csv, partition="VAL")
    val_ds.df = df_val.reset_index(drop=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_multimodal_batch)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_multimodal_batch)

    # 2. Model Initialization & Checkpoint Loading
    model = TCGraphFormer(d_model=d_model).to(device)

    if os.path.exists(base_checkpoint):
        print(f"[CKPT] Loading base checkpoint from: {base_checkpoint}")
        ckpt = torch.load(base_checkpoint, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        # Load compatible keys
        model.load_state_dict(state_dict, strict=False)
        print("[CKPT] Checkpoint weights loaded successfully.")
    else:
        print(f"[WARN] Base checkpoint {base_checkpoint} not found. Training from scratch.")

    # 3. Parameter Freezing
    if freeze_encoders:
        print("[FREEZE] Freezing lower convolutional modality encoders (IR, PMW, GPROF, Radar, Env)...")
        for enc in [model.ir_encoder, model.pmw_encoder, model.gprof_encoder, model.radar_encoder, model.env_encoder]:
            for param in enc.parameters():
                param.requires_grad = False

    # 4. Task-Specific Loss Weights
    if task == "intensity_change":
        # Heavily weight intensity change and rapid intensification
        loss_fn = MultiTaskLoss(
            weight_intensity=0.2,
            weight_pressure=0.1,
            weight_stage=0.2,
            weight_intensity_change=3.0,
            weight_ri=5.0,
        ).to(device)
    else:
        loss_fn = MultiTaskLoss().to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)

    trainer = TCGraphFormerTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        checkpoint_dir=checkpoint_dir,
    )

    history = []
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        print(f"\n--- Fine-Tuning Epoch {epoch}/{epochs} ---")
        train_metrics = trainer.train_epoch(train_loader, epoch=epoch)
        val_metrics = trainer.evaluate(val_loader)

        val_loss = val_metrics["val_loss"]
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        fn = f"finetuned_{task}_epoch_{epoch:03d}.pt"
        saved_path = trainer.save_checkpoint(epoch, val_metrics, is_best=is_best, filename=fn)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics, "checkpoint": saved_path})

        print(f"[Train] Loss: {train_metrics['loss_total']:.4f} | dV Loss: {train_metrics['loss_intensity_change']:.4f} | RI Loss: {train_metrics['loss_ri']:.4f}")
        print(f"[Val]   Loss: {val_loss:.4f} | dV MAE: {val_metrics['val_dv_mae_kt']} kt | RI Acc: {val_metrics['val_ri_acc_pct']}%")

    summary = {
        "task": task,
        "base_checkpoint": base_checkpoint,
        "epochs": epochs,
        "learning_rate": lr,
        "freeze_encoders": freeze_encoders,
        "best_val_loss": round(best_val_loss, 4),
        "history": history,
    }

    os.makedirs(os.path.dirname(output_report), exist_ok=True)
    with open(output_report, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[OK] Fine-tuning complete. Report saved to: {output_report}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TC-GraphFormer Fine-Tuning")
    parser.add_argument("--checkpoint", default="checkpoints/best_checkpoint.pt", help="Base checkpoint")
    parser.add_argument("--task", default="intensity_change", choices=["intensity_change", "nio_adaptation"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-freeze", action="store_true")
    parser.add_argument("--output-report", default=None)
    args = parser.parse_args()

    run_finetuning(
        base_checkpoint=args.checkpoint,
        task=args.task,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        d_model=args.d_model,
        max_samples=args.max_samples,
        freeze_encoders=(not args.no_freeze),
        output_report=args.output_report,
    )
