"""
Supervised Multi-Task Training Workflow for TC-GraphFormer.
Trains jointly on wind speed, central pressure, development stage,
intensity changes, rapid intensification, and track displacements.
Supports both single-observation and causal historical sequence training,
with verifiable pretrained backbone weight loading, YAML configuration parsing,
gradient accumulation, reliable full checkpoint resume (optimizer, scheduler, scaler),
and strict parameter and checkpoint provenance validation.
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
from src.training.trainer import TCGraphFormerTrainer, restore_checkpoint_state


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

    # Check for verifiable training provenance (strictly requiring supervised_multitask_training and epoch >= 1)
    is_trained = (
        is_compatible
        and (ckpt.get("is_trained") is True)
        and isinstance(ckpt.get("provenance"), dict)
        and (ckpt.get("provenance", {}).get("workflow") == "supervised_multitask_training")
        and (ckpt.get("provenance", {}).get("is_trained") is True)
        and (ckpt.get("epoch", 0) >= 1)
    )

    return {
        "checkpoint_path": checkpoint_path,
        "is_compatible": is_compatible,
        "is_trained": is_trained,
        "total_model_keys": len(model_keys),
        "matched_keys_count": len(matched),
        "missing_keys": missing,
        "shape_mismatches": mismatched_shapes,
        "unexpected_keys": unexpected,
        "epoch": ckpt.get("epoch", None),
        "best_val_loss": ckpt.get("best_val_loss", None),
        "metrics": ckpt.get("metrics", {}),
        "has_scheduler_state": ckpt.get("scheduler_state_dict") is not None,
        "has_scaler_state": ckpt.get("scaler_state_dict") is not None,
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
    accumulate_grad_batches: int = 1,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    use_sequences: bool = True,
    max_history_len: int = 4,
    max_history_hours: float = 72.0,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
    persistent_workers: bool = False,
    pretrained_checkpoint: Optional[str] = None,
    resume_checkpoint: Optional[str] = None,
    allow_legacy_resume: bool = False,
    checkpoint_dir: str = "checkpoints",
    summary_report: str = "reports/phase2e4/multitask_training_summary.json",
    device: Optional[str] = None,
    cuda_device_id: Optional[int] = None,
    use_amp: Optional[bool] = None,
    precision: str = "fp16",
    seed: int = 42,
    loss_weights: Optional[Dict[str, float]] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Execute multi-task training and validation with full configuration validation,
    provenance tracking, sequence dataloading, loss weighting, gradient accumulation,
    and complete resume recovery.
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
    if accumulate_grad_batches <= 0:
        raise ValueError(f"accumulate_grad_batches must be positive, got {accumulate_grad_batches}")
    if max_history_len <= 0:
        raise ValueError(f"max_history_len must be positive, got {max_history_len}")
    if max_history_hours <= 0:
        raise ValueError(f"max_history_hours must be positive, got {max_history_hours}")

    if not os.path.exists(train_index):
        raise FileNotFoundError(f"Sample index CSV not found: {train_index}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device.startswith("cuda") and cuda_device_id is not None and torch.cuda.is_available():
        torch.cuda.set_device(cuda_device_id)

    if use_amp is None:
        use_amp = device.startswith("cuda")
    else:
        use_amp = use_amp and device.startswith("cuda")

    if pin_memory is None:
        pin_memory = device.startswith("cuda")

    lw = loss_weights or {}
    ri_pos_weight = float(lw.get("ri_pos_weight", 33.53))
    if ri_pos_weight <= 0:
        raise ValueError(f"ri_pos_weight must be positive, got {ri_pos_weight}")

    print("=" * 70)
    print("TC-GRAPHFORMER SUPERVISED MULTI-TASK TRAINING (PHASE 2E.4.1)")
    print(f"Device: {device} | AMP: {use_amp} | d_model: {d_model} | Epochs: {epochs} | Batch size: {batch_size}")
    print(f"Grad Accumulation: {accumulate_grad_batches} batches | Effective Batch Size: {batch_size * accumulate_grad_batches}")
    print(f"Mode: {'Historical Sequences' if use_sequences else 'Single Overpass'}")
    if use_sequences:
        print(f"History Config: max_history_len={max_history_len}, max_window={max_history_hours}h")
    print(f"Pretrained Init: {pretrained_checkpoint or 'Random Initialization (Scratch)'}")
    if resume_checkpoint:
        print(f"Resume Checkpoint: {resume_checkpoint} (allow_legacy_resume={allow_legacy_resume})")
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=(persistent_workers and num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=(persistent_workers and num_workers > 0),
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

    trainer = TCGraphFormerTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=checkpoint_dir,
        use_amp=use_amp,
        precision=precision,
        grad_clip_norm=grad_clip_norm,
        accumulate_grad_batches=accumulate_grad_batches,
        best_val_loss=best_val_loss,
    )

    if resume_checkpoint:
        res = trainer.restore_checkpoint(resume_checkpoint, allow_legacy_resume=allow_legacy_resume)
        start_epoch = res["next_epoch"]
        best_val_loss = res["best_val_loss"]
        init_provenance = f"resumed_checkpoint ({resume_checkpoint}, start_epoch={start_epoch}, best_val_loss={best_val_loss:.4f})"
        print(f"[RESUME] Checkpoint verified. Resuming at epoch {start_epoch} (historical best_val_loss={best_val_loss:.4f})")

    if dry_run:
        print("[DRY-RUN] Validating data loading, batch collation, and train_epoch execution (with accumulation & optimizer step)...")
        dry_batches = []
        for i, b in enumerate(train_loader):
            dry_batches.append(b)
            if i + 1 >= max(4, accumulate_grad_batches * 2):
                break
        train_metrics = {}
        if dry_batches:
            train_metrics = trainer.train_epoch(dry_batches, epoch=1)
            print(f"[DRY-RUN] train_epoch verified successfully! Loss: {train_metrics.get('loss', 0.0):.4f}, "
                  f"Optimizer Steps: {train_metrics.get('optimizer_steps', 0)}")
        return {
            "status": "dry_run_success",
            "device": device,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "dry_batches_evaluated": len(dry_batches),
            "train_metrics": train_metrics,
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
            trainer.best_val_loss = best_val_loss

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
                    "accumulate_grad_batches": accumulate_grad_batches,
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
        "workflow": "Supervised Multi-Task Training (Phase 2E.4.1)",
        "initialization": init_provenance,
        "transfer_info": transfer_info,
        "use_sequences": use_sequences,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "d_model": d_model,
        "grad_clip_norm": grad_clip_norm,
        "accumulate_grad_batches": accumulate_grad_batches,
        "loss_weights": lw,
        "best_val_loss": round(best_val_loss, 4),
        "history": training_history,
    }

    os.makedirs(os.path.dirname(summary_report), exist_ok=True)
    with open(summary_report, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[OK] Multi-task training complete. Summary saved to: {summary_report}")
    return summary


def build_parser():
    """Build argument parser for multi-task training CLI."""
    parser = argparse.ArgumentParser(description="TC-GraphFormer Supervised Multi-Task Training")
    parser.add_argument("--config", default=None, help="Path to YAML training configuration file")
    parser.add_argument("--train-index", default=None, help="Path to sample index CSV")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", "--learning-rate", type=float, default=None, dest="lr", help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=None, help="Weight decay")
    parser.add_argument("--d-model", type=int, default=None, help="Model hidden dimension")
    parser.add_argument("--grad-clip-norm", type=float, default=None, help="Max gradient clipping norm")
    parser.add_argument("--accumulate-grad-batches", type=int, default=None, help="Number of batches to accumulate gradients")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Max train samples (for smoke-testing)")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Max validation samples (for smoke-testing)")
    parser.add_argument("--pretrained-checkpoint", default=None, help="Path to pretrained backbone checkpoint")
    parser.add_argument("--resume-checkpoint", default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--allow-legacy-resume", dest="allow_legacy_resume", action="store_true", default=None, help="Allow resume from checkpoint lacking scheduler state")
    parser.add_argument("--no-allow-legacy-resume", dest="allow_legacy_resume", action="store_false", help="Disallow resume from legacy checkpoint")
    parser.add_argument("--single-obs", dest="use_sequences", action="store_false", default=None, help="Use single-observation mode instead of sequences")
    parser.add_argument("--use-sequences", dest="use_sequences", action="store_true", default=None, help="Use sequences mode")
    parser.add_argument("--max-history-len", type=int, default=None, help="Max historical sequence length")
    parser.add_argument("--max-history-hours", type=float, default=None, help="Max historical lookback window in hours")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader worker processes")
    parser.add_argument("--pin-memory", dest="pin_memory", action="store_true", default=None, help="Pin memory for DataLoader")
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false", help="Disable pinned memory")
    parser.add_argument("--persistent-workers", dest="persistent_workers", action="store_true", default=None, help="Use persistent workers for DataLoader")
    parser.add_argument("--no-persistent-workers", dest="persistent_workers", action="store_false", help="Disable persistent workers")
    parser.add_argument("--checkpoint-dir", default=None, help="Directory for model checkpoints")
    parser.add_argument("--summary-report", default=None, help="Summary report JSON path")
    parser.add_argument("--device", default=None, help="Device to train on ('cuda', 'cpu')")
    parser.add_argument("--cuda-device-id", type=int, default=None, help="CUDA device index")
    parser.add_argument("--precision", default=None, choices=["fp16", "bf16", "fp32"], help="Floating-point precision ('fp16', 'bf16', 'fp32')")
    parser.add_argument("--use-amp", dest="use_amp", action="store_true", default=None, help="Enable automatic mixed precision")
    parser.add_argument("--no-amp", dest="use_amp", action="store_false", help="Disable automatic mixed precision")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--dry-run", action="store_true", help="Perform smoke test of data loading and 1 forward/backward pass without full training")
    return parser


def parse_args():
    return build_parser().parse_args()


def resolve_training_config(args, cfg=None):
    """
    Resolve training parameters across CLI explicit args, YAML configuration, and defaults.
    Priority: CLI explicit flag > YAML config > fallback default.
    """
    if cfg is None:
        cfg = {}
        config_path = getattr(args, "config", None)
        if config_path:
            cfg = load_yaml_config(config_path)

    tr_cfg = cfg.get("training", {})
    loss_weights = cfg.get("loss_weights", {})
    mach_cfg = cfg.get("machine", {})
    paths_cfg = cfg.get("paths", {})

    def resolve_val(cli_val, yaml_val, default):
        if cli_val is not None:
            return cli_val
        if yaml_val is not None:
            return yaml_val
        return default

    # Sequence mode logic: CLI explicit > YAML > default True
    yaml_use_seq = tr_cfg.get("use_sequences", True)
    use_sequences = resolve_val(getattr(args, "use_sequences", None), yaml_use_seq, True)

    resolved_epochs = resolve_val(getattr(args, "epochs", None), tr_cfg.get("epochs") or tr_cfg.get("max_epochs"), 5)
    resolved_batch_size = resolve_val(getattr(args, "batch_size", None), tr_cfg.get("batch_size"), 16)
    resolved_lr = resolve_val(getattr(args, "lr", None), tr_cfg.get("learning_rate") or tr_cfg.get("lr"), 1e-4)
    resolved_weight_decay = resolve_val(getattr(args, "weight_decay", None), tr_cfg.get("weight_decay"), 1e-4)
    resolved_d_model = resolve_val(getattr(args, "d_model", None), tr_cfg.get("d_model"), 128)
    resolved_grad_clip = resolve_val(getattr(args, "grad_clip_norm", None), tr_cfg.get("grad_clip_norm") or tr_cfg.get("gradient_clip_val"), 1.0)
    resolved_accum = resolve_val(getattr(args, "accumulate_grad_batches", None), tr_cfg.get("accumulate_grad_batches"), 1)
    resolved_max_hist_len = resolve_val(getattr(args, "max_history_len", None), tr_cfg.get("max_history_len"), 4)
    resolved_max_hist_hrs = resolve_val(getattr(args, "max_history_hours", None), tr_cfg.get("max_history_hours"), 72.0)

    # Worker count resolution: 0 must not be overridden by mach_cfg
    tr_workers = tr_cfg.get("num_workers")
    yaml_workers = tr_workers if tr_workers is not None else mach_cfg.get("num_workers")
    resolved_workers = resolve_val(getattr(args, "num_workers", None), yaml_workers, 0)

    resolved_pin_mem = resolve_val(getattr(args, "pin_memory", None), tr_cfg.get("pin_memory"), None)
    resolved_persist = resolve_val(getattr(args, "persistent_workers", None), tr_cfg.get("persistent_workers"), False)
    resolved_ckpt_dir = resolve_val(getattr(args, "checkpoint_dir", None), tr_cfg.get("checkpoint_dir") or paths_cfg.get("checkpoints_dir"), "checkpoints")
    resolved_report = resolve_val(getattr(args, "summary_report", None), tr_cfg.get("output_report") or tr_cfg.get("summary_report"), "reports/phase2e4/multitask_training_summary.json")
    resolved_device = resolve_val(getattr(args, "device", None), mach_cfg.get("device"), None)
    resolved_cuda_id = resolve_val(getattr(args, "cuda_device_id", None), mach_cfg.get("cuda_device_id"), None)
    resolved_precision = resolve_val(getattr(args, "precision", None), mach_cfg.get("precision") or tr_cfg.get("precision"), "fp16")
    resolved_amp = resolve_val(getattr(args, "use_amp", None), tr_cfg.get("mixed_precision"), None)
    if resolved_amp is None and resolved_precision is not None:
        resolved_amp = (resolved_precision in ("fp16", "bf16"))
    elif resolved_precision == "fp32":
        resolved_amp = False
    resolved_seed = resolve_val(getattr(args, "seed", None), tr_cfg.get("seed"), 42)
    resolved_train_index = resolve_val(getattr(args, "train_index", None), paths_cfg.get("sample_index_path"), "reports/pipeline/sample_index.csv")
    resolved_pretrained = resolve_val(getattr(args, "pretrained_checkpoint", None), tr_cfg.get("pretrained_checkpoint"), None)
    resolved_resume = resolve_val(getattr(args, "resume_checkpoint", None), tr_cfg.get("resume_checkpoint"), None)
    resolved_allow_legacy = resolve_val(getattr(args, "allow_legacy_resume", None), tr_cfg.get("allow_legacy_resume"), False)

    return {
        "config_path": getattr(args, "config", None),
        "train_index": resolved_train_index,
        "epochs": resolved_epochs,
        "batch_size": resolved_batch_size,
        "lr": resolved_lr,
        "weight_decay": resolved_weight_decay,
        "d_model": resolved_d_model,
        "grad_clip_norm": resolved_grad_clip,
        "accumulate_grad_batches": resolved_accum,
        "max_train_samples": getattr(args, "max_train_samples", None),
        "max_val_samples": getattr(args, "max_val_samples", None),
        "use_sequences": use_sequences,
        "max_history_len": resolved_max_hist_len,
        "max_history_hours": resolved_max_hist_hrs,
        "num_workers": resolved_workers,
        "pin_memory": resolved_pin_mem,
        "persistent_workers": resolved_persist,
        "pretrained_checkpoint": resolved_pretrained,
        "resume_checkpoint": resolved_resume,
        "allow_legacy_resume": resolved_allow_legacy,
        "checkpoint_dir": resolved_ckpt_dir,
        "summary_report": resolved_report,
        "device": resolved_device,
        "cuda_device_id": resolved_cuda_id,
        "use_amp": resolved_amp,
        "precision": resolved_precision,
        "seed": resolved_seed,
        "loss_weights": loss_weights,
        "dry_run": getattr(args, "dry_run", False),
    }


def main():
    args = parse_args()
    resolved = resolve_training_config(args)

    run_multitask_training(
        config_path=resolved["config_path"],
        train_index=resolved["train_index"],
        epochs=resolved["epochs"],
        batch_size=resolved["batch_size"],
        lr=resolved["lr"],
        weight_decay=resolved["weight_decay"],
        d_model=resolved["d_model"],
        grad_clip_norm=resolved["grad_clip_norm"],
        accumulate_grad_batches=resolved["accumulate_grad_batches"],
        max_train_samples=resolved["max_train_samples"],
        max_val_samples=resolved["max_val_samples"],
        use_sequences=resolved["use_sequences"],
        max_history_len=resolved["max_history_len"],
        max_history_hours=resolved["max_history_hours"],
        num_workers=resolved["num_workers"],
        pin_memory=resolved["pin_memory"],
        persistent_workers=resolved["persistent_workers"],
        pretrained_checkpoint=resolved["pretrained_checkpoint"],
        resume_checkpoint=resolved["resume_checkpoint"],
        allow_legacy_resume=resolved["allow_legacy_resume"],
        checkpoint_dir=resolved["checkpoint_dir"],
        summary_report=resolved["summary_report"],
        device=resolved["device"],
        cuda_device_id=resolved["cuda_device_id"],
        use_amp=resolved["use_amp"],
        precision=resolved["precision"],
        seed=resolved["seed"],
        loss_weights=resolved["loss_weights"],
        dry_run=resolved["dry_run"],
    )


if __name__ == "__main__":
    main()
