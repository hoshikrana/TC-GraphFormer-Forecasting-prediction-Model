"""
Supervised Multi-Task Training Workflow for TC-GraphFormer.
Trains jointly on wind speed, central pressure, development stage,
intensity changes, rapid intensification, and track displacements.
Supports both single-observation and causal historical sequence training,
with verifiable pretrained backbone weight loading, YAML configuration parsing,
and strict parameter and checkpoint validation.
"""

import os
import sys
import json
import random
import argparse
from typing import Dict, Any, Optional, Tuple
import yaml
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from torch.utils.data import DataLoader

from src.data.dataset import TCPRIMEDDataset, TCPRIMEDSequenceDataset
from src.data.collate import collate_multimodal_batch, collate_sequence_batch
from src.models.tc_graphformer import TCGraphFormer
from src.training.losses import MultiTaskLoss
from src.training.trainer import TCGraphFormerTrainer


def validate_checkpoint_compatibility(
    checkpoint_path: str,
    model: TCGraphFormer,
    strict: bool = True,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Validate that a checkpoint matches the model architecture.
    Returns detail on matched, missing, and unexpected parameter keys.
    Fails loudly if strict=True and any parameter is missing or shape-mismatched.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)

    model_state = model.state_dict()
    model_keys = set(model_state.keys())
    ckpt_keys = set(state.keys())

    matched = {}
    mismatched_shapes = []

    for k in ckpt_keys:
        if k in model_keys:
            if model_state[k].shape == state[k].shape:
                matched[k] = state[k]
            else:
                mismatched_shapes.append({
                    "key": k,
                    "model_shape": list(model_state[k].shape),
                    "ckpt_shape": list(state[k].shape),
                })

    missing = sorted(list(model_keys - set(matched.keys())))
    unexpected = sorted(list(ckpt_keys - model_keys))

    is_compatible = (len(missing) == 0 and len(mismatched_shapes) == 0)

    if strict and not is_compatible:
        err_msg = (
            f"Checkpoint {checkpoint_path} is incompatible with model! "
            f"Matched: {len(matched)}/{len(model_keys)} keys. "
            f"Missing: {len(missing)}, Shape mismatches: {len(mismatched_shapes)}, "
            f"Unexpected: {len(unexpected)}."
        )
        raise ValueError(err_msg)

    return {
        "checkpoint_path": checkpoint_path,
        "is_compatible": is_compatible,
        "total_model_keys": len(model_keys),
        "matched_keys_count": len(matched),
        "missing_keys": missing,
        "shape_mismatches": mismatched_shapes,
        "unexpected_keys": unexpected,
        "epoch": ckpt.get("epoch", None),
        "metrics": ckpt.get("metrics", {}),
    }


def load_pretrained_backbone(
    model: TCGraphFormer, checkpoint_path: str, device: str = "cpu"
) -> Dict[str, Any]:
    """
    Explicitly validate and transfer pretrained encoder and fusion weights.
    Fails loudly if checkpoint is missing, corrupt, or contains 0 matching weights.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Pretrained checkpoint not found at: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)

    # Filter for backbone components (modality encoders and fusion)
    backbone_prefixes = ("ir_encoder.", "pmw_encoder.", "gprof_encoder.", "radar_encoder.", "env_encoder.", "fusion.")
    model_state = model.state_dict()
    model_keys = set(model_state.keys())

    matched_weights = {}
    missing_in_ckpt = []
    unexpected_in_ckpt = []
    shape_mismatches = []

    for k, v in state.items():
        if any(k.startswith(p) for p in backbone_prefixes):
            if k in model_keys:
                if model_state[k].shape == v.shape:
                    matched_weights[k] = v
                else:
                    shape_mismatches.append((k, model_state[k].shape, v.shape))
                    print(f"[WARN] Shape mismatch for {k}: model {model_state[k].shape} vs ckpt {v.shape}")
            else:
                unexpected_in_ckpt.append(k)

    for k in model_keys:
        if any(k.startswith(p) for p in backbone_prefixes):
            if k not in matched_weights:
                missing_in_ckpt.append(k)

    if len(matched_weights) == 0:
        raise ValueError(
            f"Pretrained checkpoint '{checkpoint_path}' contains 0 compatible backbone parameters "
            f"for TCGraphFormer! Checked prefixes: {backbone_prefixes}"
        )

    model.load_state_dict(matched_weights, strict=False)
    print(f"[TRANSFER] Loaded {len(matched_weights)} backbone parameter tensors from {checkpoint_path}.")
    if missing_in_ckpt:
        print(f"[TRANSFER] Unmatched backbone keys in model: {len(missing_in_ckpt)}")
    if shape_mismatches:
        print(f"[TRANSFER] Shape mismatches encountered: {len(shape_mismatches)}")

    return {
        "checkpoint_path": checkpoint_path,
        "loaded_keys_count": len(matched_weights),
        "missing_keys_count": len(missing_in_ckpt),
        "shape_mismatches_count": len(shape_mismatches),
        "loaded_keys_sample": list(matched_weights.keys())[:5],
    }


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load and parse YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def run_multitask_training(
    config_path: Optional[str] = None,
    train_index: str = "reports/pipeline/sample_index.csv",
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    d_model: int = 128,
    grad_clip_norm: float = 1.0,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    use_sequences: bool = True,
    max_history_len: int = 4,
    max_history_hours: float = 72.0,
    num_workers: int = 0,
    pretrained_checkpoint: Optional[str] = None,
    resume_checkpoint: Optional[str] = None,
    checkpoint_dir: str = "checkpoints",
    summary_report: str = "reports/phase2e4/multitask_training_summary.json",
    device: Optional[str] = None,
    use_amp: Optional[bool] = None,
    seed: int = 42,
    loss_weights: Optional[Dict[str, float]] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Execute multi-task training and validation with full configuration validation,
    provenance tracking, sequence dataloading, and loss weighting.
    """
    # 0. Reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 1. Validation of parameters
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if lr <= 0:
        raise ValueError(f"learning_rate must be positive, got {lr}")
    if d_model <= 0:
        raise ValueError(f"d_model must be positive, got {d_model}")
    if grad_clip_norm <= 0:
        raise ValueError(f"grad_clip_norm must be positive, got {grad_clip_norm}")
    if max_history_len <= 0:
        raise ValueError(f"max_history_len must be positive, got {max_history_len}")
    if max_history_hours <= 0:
        raise ValueError(f"max_history_hours must be positive, got {max_history_hours}")

    if not os.path.exists(train_index):
        raise FileNotFoundError(f"Sample index CSV not found: {train_index}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if use_amp is None:
        use_amp = device.startswith("cuda")
    else:
        use_amp = use_amp and device.startswith("cuda")

    lw = loss_weights or {}
    ri_pos_weight = float(lw.get("ri_pos_weight", 33.53))
    if ri_pos_weight <= 0:
        raise ValueError(f"ri_pos_weight must be positive, got {ri_pos_weight}")

    print("=" * 70)
    print("TC-GRAPHFORMER SUPERVISED MULTI-TASK TRAINING (PHASE 2E.4)")
    print(f"Device: {device} | AMP: {use_amp} | d_model: {d_model} | Epochs: {epochs} | Batch size: {batch_size}")
    print(f"Mode: {'Historical Sequences' if use_sequences else 'Single Overpass'}")
    if use_sequences:
        print(f"History Config: max_history_len={max_history_len}, max_window={max_history_hours}h")
    print(f"Pretrained Init: {pretrained_checkpoint or 'Random Initialization (Scratch)'}")
    if resume_checkpoint:
        print(f"Resume Checkpoint: {resume_checkpoint}")
    print(f"Loss weights: intensity={lw.get('weight_intensity', 1.0)}, pressure={lw.get('weight_pressure', 0.5)}, "
          f"stage={lw.get('weight_stage', 0.5)}, dv={lw.get('weight_intensity_change', 1.0)}, "
          f"ri={lw.get('weight_ri', 2.0)}, track={lw.get('weight_track', 1.0)}, ri_pos_weight={ri_pos_weight}")
    print("=" * 70)

    # 2. Datasets & DataLoaders
    if use_sequences:
        train_dataset = TCPRIMEDSequenceDataset(
            sample_index_csv=train_index,
            partition="TRAIN",
            grid_size=64,
            max_history_len=max_history_len,
            max_history_hours=max_history_hours,
            max_sequences=max_train_samples,
        )
        val_dataset = TCPRIMEDSequenceDataset(
            sample_index_csv=train_index,
            partition="VAL",
            grid_size=64,
            max_history_len=max_history_len,
            max_history_hours=max_history_hours,
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

    pin_mem = device.startswith("cuda")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_mem,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_mem,
    )

    # 3. Model & Checkpoint Loading
    model = TCGraphFormer(d_model=d_model, num_stages=5)
    init_provenance = "random_scratch"
    transfer_info = {}

    if pretrained_checkpoint:
        transfer_info = load_pretrained_backbone(model, pretrained_checkpoint, device=device)
        init_provenance = f"pretrained_backbone ({pretrained_checkpoint})"

    loss_fn = MultiTaskLoss(
        weight_intensity=float(lw.get("weight_intensity", 1.0)),
        weight_pressure=float(lw.get("weight_pressure", 0.5)),
        weight_stage=float(lw.get("weight_stage", 0.5)),
        weight_intensity_change=float(lw.get("weight_intensity_change", 1.0)),
        weight_ri=float(lw.get("weight_ri", 2.0)),
        weight_track=float(lw.get("weight_track", 1.0)),
        ri_pos_weight=ri_pos_weight,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=1e-6)

    start_epoch = 1
    best_val_loss = float("inf")
    training_history = []

    if resume_checkpoint:
        compat = validate_checkpoint_compatibility(resume_checkpoint, model, strict=True, device=device)
        ckpt = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("metrics", {}).get("val_loss", float("inf"))
        init_provenance = f"resumed_checkpoint ({resume_checkpoint}, start_epoch={start_epoch})"
        print(f"[RESUME] Restored model and optimizer state from {resume_checkpoint} (resuming at epoch {start_epoch})")

    trainer = TCGraphFormerTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        checkpoint_dir=checkpoint_dir,
        use_amp=use_amp,
        grad_clip_norm=grad_clip_norm,
    )

    if dry_run:
        print("[DRY-RUN] Validating data loading, batch collation, and single forward-backward execution...")
        if len(train_loader) > 0:
            sample_batch = next(iter(train_loader))
            preds = trainer._forward_batch(sample_batch)
            targets = {k: v.to(device) for k, v in sample_batch["targets"].items() if isinstance(v, torch.Tensor)}
            loss, loss_dict = loss_fn(preds, targets)
            loss.backward()
            optimizer.zero_grad()
            print(f"[DRY-RUN] Forward & backward pass verified successfully! Loss: {loss.item():.4f}")
        return {
            "status": "dry_run_success",
            "device": device,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
        }

    # 4. Training Loop
    for epoch in range(start_epoch, epochs + 1):
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
                "config": {
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": lr,
                    "weight_decay": weight_decay,
                    "d_model": d_model,
                    "grad_clip_norm": grad_clip_norm,
                    "max_history_len": max_history_len,
                    "max_history_hours": max_history_hours,
                    "seed": seed,
                    "loss_weights": lw,
                },
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
        "workflow": "Supervised Multi-Task Training (Phase 2E.4)",
        "initialization": init_provenance,
        "transfer_info": transfer_info,
        "use_sequences": use_sequences,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "d_model": d_model,
        "grad_clip_norm": grad_clip_norm,
        "loss_weights": lw,
        "best_val_loss": round(best_val_loss, 4),
        "history": training_history,
    }

    os.makedirs(os.path.dirname(summary_report), exist_ok=True)
    with open(summary_report, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[OK] Multi-task training complete. Summary saved to: {summary_report}")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="TC-GraphFormer Supervised Multi-Task Training")
    parser.add_argument("--config", default=None, help="Path to YAML training configuration file")
    parser.add_argument("--train-index", default=None, help="Path to sample index CSV")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", "--learning-rate", type=float, default=None, dest="lr", help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=None, help="Weight decay")
    parser.add_argument("--d-model", type=int, default=None, help="Model hidden dimension")
    parser.add_argument("--grad-clip-norm", type=float, default=None, help="Max gradient clipping norm")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Max train samples (for smoke-testing)")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Max validation samples (for smoke-testing)")
    parser.add_argument("--pretrained-checkpoint", default=None, help="Path to pretrained backbone checkpoint")
    parser.add_argument("--resume-checkpoint", default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--single-obs", action="store_true", default=None, help="Use single-observation mode instead of sequences")
    parser.add_argument("--max-history-len", type=int, default=None, help="Max historical sequence length")
    parser.add_argument("--max-history-hours", type=float, default=None, help="Max historical lookback window in hours")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader worker processes")
    parser.add_argument("--checkpoint-dir", default=None, help="Directory for model checkpoints")
    parser.add_argument("--summary-report", default=None, help="Summary report JSON path")
    parser.add_argument("--device", default=None, help="Device to train on ('cuda', 'cpu')")
    parser.add_argument("--use-amp", action="store_true", default=None, help="Enable automatic mixed precision")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--dry-run", action="store_true", help="Perform smoke test of data loading and 1 forward/backward pass without full training")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load configuration file if provided
    cfg = {}
    if args.config:
        cfg = load_yaml_config(args.config)

    # Extract sub-sections
    tr_cfg = cfg.get("training", {})
    loss_weights = cfg.get("loss_weights", {})
    mach_cfg = cfg.get("machine", {})
    paths_cfg = cfg.get("paths", {})

    # Helper to resolve parameter priority: CLI explicit > YAML config > fallback default
    def resolve_val(cli_val, yaml_val, default):
        if cli_val is not None:
            return cli_val
        if yaml_val is not None:
            return yaml_val
        return default

    # Sequence mode logic: CLI --single-obs forces False, else check YAML, default True
    if args.single_obs is True:
        use_sequences = False
    elif "use_sequences" in tr_cfg:
        use_sequences = bool(tr_cfg["use_sequences"])
    else:
        use_sequences = True

    resolved_epochs = resolve_val(args.epochs, tr_cfg.get("epochs") or tr_cfg.get("max_epochs"), 5)
    resolved_batch_size = resolve_val(args.batch_size, tr_cfg.get("batch_size"), 16)
    resolved_lr = resolve_val(args.lr, tr_cfg.get("learning_rate") or tr_cfg.get("lr"), 1e-4)
    resolved_weight_decay = resolve_val(args.weight_decay, tr_cfg.get("weight_decay"), 1e-4)
    resolved_d_model = resolve_val(args.d_model, tr_cfg.get("d_model"), 128)
    resolved_grad_clip = resolve_val(args.grad_clip_norm, tr_cfg.get("grad_clip_norm") or tr_cfg.get("gradient_clip_val"), 1.0)
    resolved_max_hist_len = resolve_val(args.max_history_len, tr_cfg.get("max_history_len"), 4)
    resolved_max_hist_hrs = resolve_val(args.max_history_hours, tr_cfg.get("max_history_hours"), 72.0)
    resolved_workers = resolve_val(args.num_workers, tr_cfg.get("num_workers") or mach_cfg.get("num_workers"), 0)
    resolved_ckpt_dir = resolve_val(args.checkpoint_dir, tr_cfg.get("checkpoint_dir") or paths_cfg.get("checkpoints_dir"), "checkpoints")
    resolved_report = resolve_val(args.summary_report, tr_cfg.get("output_report") or tr_cfg.get("summary_report"), "reports/phase2e4/multitask_training_summary.json")
    resolved_device = resolve_val(args.device, mach_cfg.get("device"), None)
    resolved_amp = resolve_val(args.use_amp, tr_cfg.get("mixed_precision"), None)
    resolved_seed = resolve_val(args.seed, tr_cfg.get("seed"), 42)
    resolved_train_index = resolve_val(args.train_index, paths_cfg.get("sample_index_path"), "reports/pipeline/sample_index.csv")
    resolved_pretrained = resolve_val(args.pretrained_checkpoint, tr_cfg.get("pretrained_checkpoint"), None)
    resolved_resume = resolve_val(args.resume_checkpoint, tr_cfg.get("resume_checkpoint"), None)

    run_multitask_training(
        config_path=args.config,
        train_index=resolved_train_index,
        epochs=resolved_epochs,
        batch_size=resolved_batch_size,
        lr=resolved_lr,
        weight_decay=resolved_weight_decay,
        d_model=resolved_d_model,
        grad_clip_norm=resolved_grad_clip,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        use_sequences=use_sequences,
        max_history_len=resolved_max_hist_len,
        max_history_hours=resolved_max_hist_hrs,
        num_workers=resolved_workers,
        pretrained_checkpoint=resolved_pretrained,
        resume_checkpoint=resolved_resume,
        checkpoint_dir=resolved_ckpt_dir,
        summary_report=resolved_report,
        device=resolved_device,
        use_amp=resolved_amp,
        seed=resolved_seed,
        loss_weights=loss_weights,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
