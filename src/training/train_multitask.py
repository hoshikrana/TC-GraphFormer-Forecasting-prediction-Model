"""
Supervised Multi-Task Training Workflow for TC-GraphFormer.
Trains jointly on wind speed, central pressure, development stage,
intensity changes, and rapid intensification.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from torch.utils.data import DataLoader

from src.data.dataset import TCPRIMEDDataset
from src.data.collate import collate_multimodal_batch
from src.models.tc_graphformer import TCGraphFormer
from src.training.losses import MultiTaskLoss
from src.training.trainer import TCGraphFormerTrainer


def run_multitask_training(
    train_index: str = "reports/pipeline/sample_index.csv",
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    d_model: int = 128,
    max_train_samples: int = None,
    max_val_samples: int = None,
    checkpoint_dir: str = "checkpoints",
    summary_report: str = "reports/multitask_training_summary.json",
    device: str = None,
) -> Dict[str, Any]:
    """
    Execute multi-task training and validation.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("TC-GRAPHFORMER SUPERVISED MULTI-TASK TRAINING")
    print(f"Device: {device} | d_model: {d_model} | Epochs: {epochs} | Batch size: {batch_size}")
    print("=" * 60)

    # 1. Datasets & DataLoaders
    train_dataset = TCPRIMEDDataset(
        sample_index_csv=train_index,
        partition="TRAIN",
        grid_size=64,
        max_samples=max_train_samples,
    )
    val_dataset = TCPRIMEDDataset(
        sample_index_csv=train_index,
        partition="VAL",
        grid_size=64,
        max_samples=max_val_samples,
    )

    num_workers = 0 # Safe default across OS/Windows
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_multimodal_batch,
        pin_memory=(device.startswith("cuda")),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_multimodal_batch,
        pin_memory=(device.startswith("cuda")),
    )

    # 2. Model & Loss & Optimizer
    model = TCGraphFormer(d_model=d_model)
    loss_fn = MultiTaskLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=1e-6)

    trainer = TCGraphFormerTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        checkpoint_dir=checkpoint_dir,
        use_amp=(device.startswith("cuda")),
        grad_clip_norm=1.0,
    )

    training_history = []
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")
        train_metrics = trainer.train_epoch(train_loader, epoch=epoch)
        val_metrics = trainer.evaluate(val_loader)
        scheduler.step()

        val_loss = val_metrics["val_loss"]
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        # Save checkpoint
        saved_ckpt = trainer.save_checkpoint(epoch, val_metrics, is_best=is_best)

        epoch_record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "is_best": is_best,
            "checkpoint_path": saved_ckpt,
        }
        training_history.append(epoch_record)

        print(f"[Train] Loss: {train_metrics['loss_total']:.4f} (v: {train_metrics['loss_intensity']:.2f}, p: {train_metrics['loss_pressure']:.2f}, stage: {train_metrics['loss_stage']:.2f})")
        print(f"[Val]   Loss: {val_loss:.4f} | Wind MAE: {val_metrics['val_wind_mae_kt']} kt | Pressure MAE: {val_metrics['val_pressure_mae_hpa']} hPa | Stage Acc: {val_metrics['val_stage_acc_pct']}%")

    summary = {
        "model_name": "TC-GraphFormer",
        "d_model": d_model,
        "device": device,
        "epochs_trained": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "best_val_loss": round(best_val_loss, 4),
        "checkpoints_directory": checkpoint_dir,
        "latest_checkpoint": os.path.join(checkpoint_dir, "latest_checkpoint.pt"),
        "best_checkpoint": os.path.join(checkpoint_dir, "best_checkpoint.pt"),
        "history": training_history,
    }

    os.makedirs(os.path.dirname(summary_report), exist_ok=True)
    with open(summary_report, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"[OK] Training complete. Best Val Loss: {best_val_loss:.4f}")
    print(f"     Summary written to: {summary_report}")
    print("=" * 60)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TC-GraphFormer Supervised Training")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--d-model", type=int, default=128, help="Model hidden dimension")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Limit train samples (for smoke test)")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Limit val samples (for smoke test)")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--output-report", default="reports/multitask_training_summary.json", help="Summary JSON output")
    args = parser.parse_args()

    run_multitask_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        d_model=args.d_model,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        checkpoint_dir=args.checkpoint_dir,
        summary_report=args.output_report,
    )
