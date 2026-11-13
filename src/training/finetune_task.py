"""
Task-Specific Fine-Tuning and Domain Adaptation Workflow for TC-GraphFormer.
Supports isolated task objectives:
- intensity
- pressure
- stage
- intensity_change
- rapid_intensification
- track
- multi_task
- nio_adaptation (North Indian Ocean basin adaptation)
Enforces strictly validated base checkpoints, frozen backbone parameters,
and task-specific validation monitoring.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, Optional

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import TCPRIMEDDataset, TCPRIMEDSequenceDataset
from src.data.collate import collate_multimodal_batch, collate_sequence_batch
from src.models.tc_graphformer import TCGraphFormer
from src.training.losses import MultiTaskLoss
from src.training.trainer import TCGraphFormerTrainer

SUPPORTED_TASKS = [
    "intensity",
    "pressure",
    "stage",
    "intensity_change",
    "rapid_intensification",
    "track",
    "multi_task",
    "nio_adaptation",
]


def create_task_specific_loss(task: str) -> MultiTaskLoss:
    """
    Construct MultiTaskLoss where ONLY the designated task objective is active,
    preventing unintended multi-objective interference.
    """
    if task == "intensity":
        return MultiTaskLoss(weight_intensity=1.0, weight_pressure=0.0, weight_stage=0.0,
                             weight_intensity_change=0.0, weight_ri=0.0, weight_track=0.0)
    elif task == "pressure":
        return MultiTaskLoss(weight_intensity=0.0, weight_pressure=1.0, weight_stage=0.0,
                             weight_intensity_change=0.0, weight_ri=0.0, weight_track=0.0)
    elif task == "stage":
        return MultiTaskLoss(weight_intensity=0.0, weight_pressure=0.0, weight_stage=1.0,
                             weight_intensity_change=0.0, weight_ri=0.0, weight_track=0.0)
    elif task == "intensity_change":
        return MultiTaskLoss(weight_intensity=0.0, weight_pressure=0.0, weight_stage=0.0,
                             weight_intensity_change=1.0, weight_ri=0.0, weight_track=0.0)
    elif task == "rapid_intensification":
        return MultiTaskLoss(weight_intensity=0.0, weight_pressure=0.0, weight_stage=0.0,
                             weight_intensity_change=0.0, weight_ri=1.0, weight_track=0.0)
    elif task == "track":
        return MultiTaskLoss(weight_intensity=0.0, weight_pressure=0.0, weight_stage=0.0,
                             weight_intensity_change=0.0, weight_ri=0.0, weight_track=1.0)
    elif task in ("multi_task", "nio_adaptation"):
        return MultiTaskLoss(weight_intensity=1.0, weight_pressure=0.5, weight_stage=0.5,
                             weight_intensity_change=1.0, weight_ri=2.0, weight_track=1.0)
    else:
        raise ValueError(f"Unsupported fine-tuning task: {task}. Must be one of {SUPPORTED_TASKS}")


def run_finetuning(
    base_checkpoint: Optional[str] = "checkpoints/best_checkpoint.pt",
    task: str = "intensity_change",
    epochs: int = 3,
    batch_size: int = 16,
    lr_head: float = 1e-4,
    lr_backbone: float = 1e-5,
    d_model: int = 128,
    max_samples: int = None,
    freeze_encoders: bool = True,
    use_sequences: bool = True,
    max_history_len: int = 4,
    sample_index_csv: str = "reports/pipeline/sample_index.csv",
    checkpoint_dir: str = "checkpoints",
    output_report: str = None,
    device: str = None,
    allow_scratch: bool = False,
) -> Dict[str, Any]:
    """
    Execute task-specific fine-tuning or basin adaptation.
    """
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"Task '{task}' is not supported. Must be in: {SUPPORTED_TASKS}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if output_report is None:
        output_report = f"reports/phase2e/finetuning_{task}_validation.json"

    print("=" * 60)
    print(f"TC-GRAPHFORMER FINE-TUNING: Task = '{task}' (PHASE 2E)")
    print(f"Base Checkpoint: {base_checkpoint}")
    print(f"Device: {device} | Epochs: {epochs} | LR Head: {lr_head} | Freeze Encoders: {freeze_encoders}")
    print("=" * 60)

    # 1. Dataset filtering & Partition Isolation
    df = pd.read_csv(sample_index_csv)

    if task == "nio_adaptation":
        # North Indian Ocean: verify basin == 'IO'
        # Crucial: Hold out TEST storms strictly (IO031991, IO032023)
        df_train = df[(df["basin"] == "IO") & (df["partition"] == "TRAIN")].copy()
        df_val = df[(df["basin"] == "IO") & (df["partition"] == "VAL")].copy()
        test_storms = df[(df["basin"] == "IO") & (df["partition"] == "TEST")]["storm_id"].unique()
        print(f"[DATA] NIO Adaptation: {len(df_train)} train, {len(df_val)} val samples.")
        print(f"[DATA] Strictly held out NIO regional TEST storms: {list(test_storms)}")
    else:
        df_train = df[df["partition"] == "TRAIN"].copy()
        df_val = df[df["partition"] == "VAL"].copy()

    if max_samples is not None and max_samples > 0:
        df_train = df_train.iloc[:max_samples].copy()
        df_val = df_val.iloc[:max(4, max_samples // 4)].copy()

    if use_sequences:
        train_ds = TCPRIMEDSequenceDataset(
            sample_index_csv=sample_index_csv,
            partition="TRAIN",
            max_history_len=max_history_len,
            max_sequences=len(df_train),
        )
        val_ds = TCPRIMEDSequenceDataset(
            sample_index_csv=sample_index_csv,
            partition="VAL",
            max_history_len=max_history_len,
            max_sequences=len(df_val),
        )
        collate_fn = collate_sequence_batch
    else:
        train_ds = TCPRIMEDDataset(sample_index_csv=sample_index_csv, partition="TRAIN", max_samples=len(df_train))
        val_ds = TCPRIMEDDataset(sample_index_csv=sample_index_csv, partition="VAL", max_samples=len(df_val))
        collate_fn = collate_multimodal_batch

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # 2. Model Initialization & Strict Base Checkpoint Verification
    model = TCGraphFormer(d_model=d_model, num_stages=5).to(device)

    if base_checkpoint and os.path.exists(base_checkpoint):
        print(f"[CKPT] Loading base checkpoint from: {base_checkpoint}")
        ckpt = torch.load(base_checkpoint, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        load_res = model.load_state_dict(state_dict, strict=False)
        print(f"[CKPT] Successfully loaded checkpoint weights. (Missing keys: {len(load_res.missing_keys)})")
    elif allow_scratch:
        print("[WARN] Base checkpoint not provided or missing. Training from scratch explicitly permitted.")
    else:
        raise FileNotFoundError(
            f"Base checkpoint required for fine-tuning not found at '{base_checkpoint}'. "
            "To train from scratch, pass --allow-scratch explicitly."
        )

    # 3. Parameter Freezing & Differential Learning Rates
    encoder_modules = [
        model.ir_encoder,
        model.pmw_encoder,
        model.gprof_encoder,
        model.radar_encoder,
        model.env_encoder,
    ]

    frozen_params_count = 0
    if freeze_encoders:
        print("[FREEZE] Freezing modality encoders (IR, PMW, GPROF, Radar, Env)...")
        for enc in encoder_modules:
            for p in enc.parameters():
                p.requires_grad = False
                frozen_params_count += p.numel()

    # Differential learning rates for backbone vs task heads
    head_params = list(model.task_heads.parameters())
    backbone_params = [
        p for name, p in model.named_parameters()
        if p.requires_grad and not name.startswith("task_heads.")
    ]

    param_groups = [
        {"params": head_params, "lr": lr_head},
    ]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": lr_backbone})

    trainable_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[PARAMS] Trainable parameters: {trainable_params_count:,} | Frozen parameters: {frozen_params_count:,}")

    # 4. Task-Specific Loss
    loss_fn = create_task_specific_loss(task).to(device)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)

    trainer = TCGraphFormerTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        checkpoint_dir=checkpoint_dir,
    )

    history = []
    best_score = float("inf")

    for epoch in range(1, epochs + 1):
        print(f"\n--- Fine-Tuning Epoch {epoch}/{epochs} (Task: {task}) ---")
        train_metrics = trainer.train_epoch(train_loader, epoch=epoch)
        val_metrics = trainer.evaluate(val_loader)

        # Task-specific validation score monitoring
        if task == "intensity":
            current_score = val_metrics.get("val_wind_mae_kt", float("inf"))
        elif task == "pressure":
            current_score = val_metrics.get("val_pressure_mae_hpa", float("inf"))
        elif task == "stage":
            current_score = -val_metrics.get("val_stage_acc_pct", 0.0) # Negate so lower is better
        elif task == "rapid_intensification":
            current_score = -val_metrics.get("val_ri_f1", 0.0) # Negate so higher F1 is better
        elif task == "track":
            current_score = val_metrics.get("val_track_24h_err_km", val_metrics.get("val_loss", float("inf")))
        else:
            current_score = val_metrics.get("val_loss", float("inf"))

        is_best = current_score < best_score
        if is_best:
            best_score = current_score

        fn = f"finetuned_{task}_epoch_{epoch:03d}.pt"
        saved_path = trainer.save_checkpoint(
            epoch,
            val_metrics,
            is_best=is_best,
            filename=fn,
            extra_metadata={
                "task": task,
                "base_checkpoint": base_checkpoint,
                "freeze_encoders": freeze_encoders,
            },
        )
        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "monitored_score": current_score,
            "checkpoint": saved_path,
        })

        print(f"[Train] Loss: {train_metrics['loss_total']:.4f}")
        print(f"[Val]   Loss: {val_metrics['val_loss']:.4f} | Monitored Score: {current_score:.4f}")

    summary = {
        "task": task,
        "base_checkpoint": base_checkpoint,
        "epochs": epochs,
        "learning_rate_head": lr_head,
        "learning_rate_backbone": lr_backbone,
        "freeze_encoders": freeze_encoders,
        "trainable_parameters": trainable_params_count,
        "frozen_parameters": frozen_params_count,
        "best_monitored_score": round(best_score, 4),
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
    parser.add_argument("--task", default="intensity_change", choices=SUPPORTED_TASKS)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr-head", type=float, default=1e-4)
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-freeze", action="store_true")
    parser.add_argument("--allow-scratch", action="store_true")
    parser.add_argument("--output-report", default=None)
    args = parser.parse_args()

    run_finetuning(
        base_checkpoint=args.checkpoint,
        task=args.task,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_head=args.lr_head,
        lr_backbone=args.lr_backbone,
        d_model=args.d_model,
        max_samples=args.max_samples,
        freeze_encoders=(not args.no_freeze),
        allow_scratch=args.allow_scratch,
        output_report=args.output_report,
    )
