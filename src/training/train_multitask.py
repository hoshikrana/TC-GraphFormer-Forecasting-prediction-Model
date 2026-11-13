"""
Supervised Multi-Task Training Workflow for TC-GraphFormer.
Trains jointly on wind speed, central pressure, development stage,
intensity changes, rapid intensification, and track displacements.
Supports both single-observation and causal historical sequence training,
with verifiable pretrained backbone weight loading.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, Optional

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from torch.utils.data import DataLoader

from src.data.dataset import TCPRIMEDDataset, TCPRIMEDSequenceDataset
from src.data.collate import collate_multimodal_batch, collate_sequence_batch
from src.models.tc_graphformer import TCGraphFormer
from src.training.losses import MultiTaskLoss
from src.training.trainer import TCGraphFormerTrainer


def load_pretrained_backbone(
    model: TCGraphFormer, checkpoint_path: str, device: str = "cpu"
) -> Dict[str, Any]:
    """
    Explicitly validate and transfer pretrained encoder and fusion weights.
    Fails loudly if checkpoint is missing or corrupt.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Pretrained checkpoint not found at: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)

    # Filter for backbone components (modality encoders and fusion)
    backbone_prefixes = ("ir_encoder.", "pmw_encoder.", "gprof_encoder.", "radar_encoder.", "env_encoder.", "fusion.")
    model_keys = set(model.state_dict().keys())

    matched_weights = {}
    missing_in_ckpt = []
    unexpected_in_ckpt = []

    for k, v in state.items():
        if any(k.startswith(p) for p in backbone_prefixes):
            if k in model_keys:
                if model.state_dict()[k].shape == v.shape:
                    matched_weights[k] = v
                else:
                    print(f"[WARN] Shape mismatch for {k}: model {model.state_dict()[k].shape} vs ckpt {v.shape}")
            else:
                unexpected_in_ckpt.append(k)

    for k in model_keys:
        if any(k.startswith(p) for p in backbone_prefixes):
            if k not in matched_weights:
                missing_in_ckpt.append(k)

    msg = model.load_state_dict(matched_weights, strict=False)
    print(f"[TRANSFER] Loaded {len(matched_weights)} backbone parameter tensors from {checkpoint_path}.")
    if missing_in_ckpt:
        print(f"[TRANSFER] Unmatched backbone keys in model: {len(missing_in_ckpt)}")

    return {
        "checkpoint_path": checkpoint_path,
        "loaded_keys_count": len(matched_weights),
        "missing_keys_count": len(missing_in_ckpt),
        "loaded_keys_sample": list(matched_weights.keys())[:5],
    }


def run_multitask_training(
    train_index: str = "reports/pipeline/sample_index.csv",
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    d_model: int = 128,
    max_train_samples: int = None,
    max_val_samples: int = None,
    use_sequences: bool = True,
    max_history_len: int = 4,
    pretrained_checkpoint: Optional[str] = None,
    checkpoint_dir: str = "checkpoints",
    summary_report: str = "reports/phase2e/multitask_training_summary.json",
    device: str = None,
) -> Dict[str, Any]:
    """
    Execute multi-task training and validation with full provenance tracking.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("TC-GRAPHFORMER SUPERVISED MULTI-TASK TRAINING (PHASE 2E)")
    print(f"Device: {device} | d_model: {d_model} | Epochs: {epochs} | Batch size: {batch_size}")
    print(f"Mode: {'Historical Sequences' if use_sequences else 'Single Overpass'}")
    print(f"Pretrained Init: {pretrained_checkpoint or 'Random Initialization (Scratch)'}")
    print("=" * 60)

    # 1. Datasets & DataLoaders
    if use_sequences:
        train_dataset = TCPRIMEDSequenceDataset(
            sample_index_csv=train_index,
            partition="TRAIN",
            grid_size=64,
            max_history_len=max_history_len,
            max_sequences=max_train_samples,
        )
        val_dataset = TCPRIMEDSequenceDataset(
            sample_index_csv=train_index,
            partition="VAL",
            grid_size=64,
            max_history_len=max_history_len,
            max_sequences=max_val_samples,
        )
        collate_fn = collate_sequence_batch
    else:
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
        collate_fn = collate_multimodal_batch

    num_workers = 0 # Safe across Windows processes
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.startswith("cuda")),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.startswith("cuda")),
    )

    # 2. Model & Checkpoint Loading
    model = TCGraphFormer(d_model=d_model, num_stages=5)
    init_provenance = "random_scratch"
    transfer_info = {}

    if pretrained_checkpoint:
        transfer_info = load_pretrained_backbone(model, pretrained_checkpoint, device=device)
        init_provenance = f"pretrained_backbone ({pretrained_checkpoint})"

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
        scheduler.step()

        val_metrics = trainer.evaluate(val_loader)
        val_loss = val_metrics["val_loss"]

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        fn = f"multitask_epoch_{epoch:03d}.pt"
        saved_ckpt = trainer.save_checkpoint(
            epoch,
            val_metrics,
            is_best=is_best,
            filename=fn,
            extra_metadata={
                "initialization": init_provenance,
                "transfer_info": transfer_info,
                "use_sequences": use_sequences,
            },
        )

        training_history.append({
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "saved_checkpoint": saved_ckpt,
        })

        print(f"[Train] Loss: {train_metrics['loss_total']:.4f} | Wind: {train_metrics['loss_intensity']:.4f} | Press: {train_metrics['loss_pressure']:.4f}")
        print(f"[Val]   Loss: {val_loss:.4f} | Wind MAE: {val_metrics['val_wind_mae_kt']} kt | Press MAE: {val_metrics['val_pressure_mae_hpa']} hPa | Stage Acc: {val_metrics['val_stage_acc_pct']}%")

    summary = {
        "workflow": "Supervised Multi-Task Training (Phase 2E)",
        "initialization": init_provenance,
        "transfer_info": transfer_info,
        "use_sequences": use_sequences,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "best_val_loss": round(best_val_loss, 4),
        "history": training_history,
    }

    os.makedirs(os.path.dirname(summary_report), exist_ok=True)
    with open(summary_report, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[OK] Multi-task training complete. Summary saved to: {summary_report}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TC-GraphFormer Supervised Multi-Task Training")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--d-model", type=int, default=128, help="Model hidden dimension")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Max train samples")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Max validation samples")
    parser.add_argument("--pretrained-checkpoint", default=None, help="Path to pretrained backbone checkpoint")
    parser.add_argument("--single-obs", action="store_true", help="Use single-observation mode instead of sequences")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Directory for model checkpoints")
    parser.add_argument("--summary-report", default="reports/phase2e/multitask_training_summary.json", help="Summary report path")
    args = parser.parse_args()

    run_multitask_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        d_model=args.d_model,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        use_sequences=(not args.single_obs),
        pretrained_checkpoint=args.pretrained_checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        summary_report=args.summary_report,
    )
