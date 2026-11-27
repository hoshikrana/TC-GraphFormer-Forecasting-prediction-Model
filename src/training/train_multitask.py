"""
Supervised Multi-Task Training Workflow for TC-GraphFormer.
Trains jointly on wind speed, central pressure, development stage,
intensity changes, rapid intensification, and track displacements.
Supports both single-observation and causal historical sequence training,
with verifiable pretrained backbone weight loading, YAML configuration parsing,
gradient accumulation, reliable full checkpoint resume (optimizer, scheduler, scaler),
strict parameter and checkpoint provenance validation, and feature-flagged R1
regularization (Weight EMA, train-only whole-modality dropout, fusion sector-token dropout,
linearly scheduled stochastic depth DropPath, mild storm-balanced sampler, differential
backbone/head learning rates with decoupled AdamW weight decay, dual best-composite /
best-global-loss checkpoint preservation, and eval-mode TRAIN_EVAL / VAL global loss tracking).
"""

import os
import sys
import json
import time
import platform
import hashlib
import random
import argparse
from typing import Dict, Any, Optional, Tuple, List
import yaml
import numpy as np
import pandas as pd

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset

from src.data.dataset import TCPRIMEDDataset, TCPRIMEDSequenceDataset
from src.data.collate import collate_multimodal_batch, collate_sequence_batch
from src.models.tc_graphformer import TCGraphFormer
from src.training.losses import MultiTaskLoss
from src.training.trainer import (
    TCGraphFormerTrainer,
    restore_checkpoint_state,
    build_storm_balanced_weights,
    build_decoupled_param_groups,
)


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

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
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

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
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


def _extract_anchor_storm_ids_and_intensities(train_dataset: Any) -> Tuple[List[str], np.ndarray]:
    """Extract anchor observation storm_ids and wind speeds (kt) for storm-balanced sampler construction."""
    if isinstance(train_dataset, TCPRIMEDSequenceDataset):
        df = train_dataset.base_dataset.df
        anchor_indices = [seq[-1] for seq in train_dataset.sequence_index]
        sub = df.iloc[anchor_indices]
    elif isinstance(train_dataset, TCPRIMEDDataset):
        sub = train_dataset.df
    else:
        raise TypeError(f"Unsupported dataset type for storm-balanced sampler: {type(train_dataset)}")
    storm_ids = sub["storm_id"].astype(str).tolist()
    intensities = pd.to_numeric(sub["intensity_vmax_kt"], errors="coerce").to_numpy(dtype=np.float64)
    return storm_ids, intensities


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
    # Feature-flagged R1 regularization options (all default to OFF for 100% backward compatibility)
    use_ema: bool = False,
    ema_decay: float = 0.999,
    modality_dropout_prob: float = 0.0,
    sector_token_dropout: float = 0.0,
    drop_path_rate: float = 0.0,
    head_dropout: float = 0.0,
    use_storm_balanced_sampler: bool = False,
    storm_balance_exponent: float = 0.5,
    storm_balance_min_major_ratio: float = 0.80,
    storm_balance_min_exponent: float = 0.25,
    backbone_lr_scale: float = 1.0,
    decoupled_weight_decay: bool = False,
    cosine_t_max: Optional[int] = None,
    eta_min: float = 1e-6,
    early_stop_patience: Optional[int] = None,
    selection_metric: str = "val_loss",
    epoch_csv_report: Optional[str] = None,
    train_eval_subset_size: int = 0,
    full_train_eval_on_best: bool = False,
    experiment_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute multi-task training and validation with full configuration validation,
    provenance tracking, sequence dataloading, loss weighting, gradient accumulation,
    complete resume recovery, and feature-flagged R1 regularizers.
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

    t_max_sched = int(cosine_t_max) if (cosine_t_max is not None and int(cosine_t_max) > 0) else max(1, epochs)

    config_payload_for_hash = {
        "experiment_name": experiment_name or "supervised_multitask_training",
        "epochs": epochs,
        "cosine_t_max": t_max_sched,
        "eta_min": float(eta_min),
        "batch_size": batch_size,
        "accumulate_grad_batches": accumulate_grad_batches,
        "lr": float(lr),
        "backbone_lr_scale": float(backbone_lr_scale),
        "weight_decay": float(weight_decay),
        "decoupled_weight_decay": bool(decoupled_weight_decay),
        "d_model": int(d_model),
        "head_dropout": float(head_dropout),
        "grad_clip_norm": float(grad_clip_norm),
        "precision": precision,
        "seed": int(seed),
        "use_ema": bool(use_ema),
        "ema_decay": float(ema_decay) if use_ema else None,
        "modality_dropout_prob": float(modality_dropout_prob),
        "sector_token_dropout": float(sector_token_dropout),
        "drop_path_rate": float(drop_path_rate),
        "use_storm_balanced_sampler": bool(use_storm_balanced_sampler),
        "storm_balance_exponent": float(storm_balance_exponent) if use_storm_balanced_sampler else None,
        "selection_metric": selection_metric,
        "early_stop_patience": early_stop_patience,
        "loss_weights": lw,
    }
    config_hash = hashlib.sha256(json.dumps(config_payload_for_hash, sort_keys=True).encode("utf-8")).hexdigest()[:12]

    software_versions = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda if torch.cuda.is_available() else None,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "gpu_name": torch.cuda.get_device_name(0) if (device.startswith("cuda") and torch.cuda.is_available()) else "CPU",
    }

    print("=" * 70)
    print(f"TC-GRAPHFORMER SUPERVISED MULTI-TASK TRAINING ({experiment_name or 'PHASE 2E.4.1'})")
    print(f"Config Hash: {config_hash} | Seed: {seed} | Python: {software_versions['python']} | Torch: {software_versions['torch']}")
    print(f"Device: {device} ({software_versions['gpu_name']}) | AMP: {use_amp} ({precision}) | d_model: {d_model}")
    print(f"Epochs: {epochs} (Cosine T_max={t_max_sched}) | Batch size: {batch_size} x Accum: {accumulate_grad_batches} (Eff={batch_size * accumulate_grad_batches})")
    print(f"Head LR: {lr} | Backbone LR Scale: {backbone_lr_scale} (Backbone LR={lr * backbone_lr_scale}) | WD: {weight_decay} (Decoupled={decoupled_weight_decay})")
    print(f"R1 Flags: EMA={use_ema} (decay={ema_decay}) | ModDrop={modality_dropout_prob} | TokDrop={sector_token_dropout} | DropPath={drop_path_rate} | StormSampler={use_storm_balanced_sampler} (exp={storm_balance_exponent})")
    print(f"Pretrained Init: {pretrained_checkpoint or 'Random Initialization (Scratch)'}")
    print("=" * 70, flush=True)

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

    sampler_info: Dict[str, Any] = {"enabled": False}
    train_sampler = None
    active_sampler_exponent = float(storm_balance_exponent)

    if use_storm_balanced_sampler:
        tr_storm_ids, tr_intensities = _extract_anchor_storm_ids_and_intensities(train_dataset)
        w_np, s_info = build_storm_balanced_weights(
            storm_ids=tr_storm_ids,
            intensities_kt=tr_intensities,
            exponent=storm_balance_exponent,
            min_major_ratio=storm_balance_min_major_ratio,
            min_exponent=storm_balance_min_exponent,
        )
        active_sampler_exponent = float(s_info["active_exponent"])
        sampler_gen = torch.Generator()
        sampler_gen.manual_seed(seed)
        train_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(w_np, dtype=torch.double),
            num_samples=len(train_dataset),
            replacement=True,
            generator=sampler_gen,
        )
        sampler_info = {"enabled": True, **s_info}
        print(
            f"[SAMPLER] Storm-balanced sampler active: exp={active_sampler_exponent:.4f}, "
            f"unbalanced_major_share={100.0 * (s_info['unbalanced_major_hurricane_share'] or 0.0):.2f}%, "
            f"expected_major_share={100.0 * (s_info['expected_major_hurricane_share'] or 0.0):.2f}% "
            f"(ratio={100.0 * (s_info['expected_to_unbalanced_major_ratio'] or 1.0):.2f}%)",
            flush=True,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
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

    # Optional deterministic TRAIN_EVAL loaders (subset for every epoch, full for best epochs)
    train_eval_subset_loader = None
    train_eval_full_loader = None
    if train_eval_subset_size and int(train_eval_subset_size) > 0:
        n_tr = len(train_dataset)
        if int(train_eval_subset_size) < n_tr:
            rng_sub = np.random.default_rng(seed)
            sub_idx = sorted(rng_sub.choice(n_tr, size=int(train_eval_subset_size), replace=False).tolist())
            train_eval_subset_loader = DataLoader(
                Subset(train_dataset, sub_idx),
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=pin_memory,
            )
        else:
            train_eval_subset_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=pin_memory,
            )
    if full_train_eval_on_best:
        train_eval_full_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
        )

    # 3. Model & Checkpoint Loading
    model = TCGraphFormer(
        d_model=d_model,
        num_stages=5,
        head_dropout=head_dropout,
        modality_dropout_prob=modality_dropout_prob,
        sector_token_dropout=sector_token_dropout,
        drop_path_rate=drop_path_rate,
    )
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
        regime_max_extra_weight=float(lw.get("regime_max_extra_weight", 2.5)),
        regime_power=float(lw.get("regime_power", 2.0)),
        ri_prior_shift_factor=float(lw.get("ri_prior_shift_factor", 0.85)),
    ).to(device)

    param_group_counts = {}
    if decoupled_weight_decay or float(backbone_lr_scale) != 1.0:
        param_groups, param_group_counts = build_decoupled_param_groups(
            model=model,
            head_lr=lr,
            backbone_lr_scale=backbone_lr_scale,
            weight_decay=weight_decay,
        )
        optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max_sched, eta_min=float(eta_min))

    start_epoch = 1
    best_val_loss = float("inf")
    best_val_loss_global = float("inf")
    best_val_mt_rel_err = float("inf")
    best_loss_epoch = 0
    best_composite_epoch = 0
    training_history = []
    epoch_csv_rows = []

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
        use_ema=use_ema,
        ema_decay=ema_decay,
    )

    if resume_checkpoint:
        res = trainer.restore_checkpoint(resume_checkpoint, allow_legacy_resume=allow_legacy_resume)
        start_epoch = res["next_epoch"]
        best_val_loss = res["best_val_loss"]
        best_val_loss_global = res["best_val_loss"]
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
            print(f"[DRY-RUN] train_epoch verified successfully! Loss: {train_metrics.get('loss_total', 0.0):.4f}, "
                  f"Optimizer Steps: {train_metrics.get('optimizer_steps', 0)}")
        return {
            "status": "dry_run_success",
            "device": device,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "dry_batches_evaluated": len(dry_batches),
            "train_metrics": train_metrics,
        }

    epochs_since_improvement = 0
    early_stopped = False
    stop_reason = None

    # 4. Training Loop
    for epoch in range(start_epoch, epochs + 1):
        lrs_now = [round(float(pg["lr"]), 8) for pg in optimizer.param_groups]
        head_lr_now = lrs_now[-1]
        backbone_lr_now = lrs_now[0]
        print(f"\n--- Epoch {epoch}/{epochs} (backbone_lr={backbone_lr_now:.2e}, head_lr={head_lr_now:.2e}) ---", flush=True)

        train_metrics = trainer.train_epoch(train_loader, epoch=epoch)
        scheduler.step()

        # Check major-hurricane share guardrail if storm-balanced sampler is enabled
        unbal_maj = sampler_info.get("unbalanced_major_hurricane_share")
        obs_maj = train_metrics.get("major_hurricane_share", 0.0)
        if use_storm_balanced_sampler and unbal_maj and unbal_maj > 0:
            maj_ratio_epoch = obs_maj / float(unbal_maj)
            train_metrics["major_hurricane_share_ratio_vs_unbalanced"] = round(maj_ratio_epoch, 4)
            if maj_ratio_epoch < float(storm_balance_min_major_ratio) and active_sampler_exponent > float(storm_balance_min_exponent):
                new_exp = max(float(storm_balance_min_exponent), active_sampler_exponent - 0.125)
                print(
                    f"[SAMPLER GUARDRAIL] Observed major-hurricane share {100.0 * obs_maj:.2f}% is < 80% of unbalanced "
                    f"({100.0 * float(unbal_maj):.2f}%). Adjusting exponent {active_sampler_exponent:.3f} -> {new_exp:.3f}.",
                    flush=True,
                )
                active_sampler_exponent = new_exp
                tr_storm_ids, tr_intensities = _extract_anchor_storm_ids_and_intensities(train_dataset)
                w_np, _ = build_storm_balanced_weights(
                    storm_ids=tr_storm_ids,
                    intensities_kt=tr_intensities,
                    exponent=active_sampler_exponent,
                    min_major_ratio=storm_balance_min_major_ratio,
                    min_exponent=storm_balance_min_exponent,
                )
                train_loader.sampler.weights = torch.as_tensor(w_np, dtype=torch.double)

        # Evaluate VAL using EMA weights when use_ema=True
        val_metrics = trainer.evaluate(val_loader)
        val_loss_batch = float(val_metrics["val_loss"])
        val_loss_global = float(val_metrics.get("val_loss_global", val_loss_batch))
        val_mt_rel_err = float(val_metrics.get("val_mt_rel_err", float("inf")))

        # Evaluate TRAIN_EVAL in identical eval() mode (using EMA weights when use_ema=True)
        train_eval_metrics = None
        train_eval_loss_global = None
        if train_eval_subset_loader is not None:
            train_eval_metrics = trainer.evaluate(train_eval_subset_loader)
            train_eval_loss_global = float(train_eval_metrics.get("val_loss_global", train_eval_metrics["val_loss"]))

        is_best_loss = val_loss_global < best_val_loss_global
        if is_best_loss:
            best_val_loss_global = val_loss_global
            best_val_loss = val_loss_global
            best_loss_epoch = epoch
            trainer.best_val_loss = best_val_loss_global

        is_best_composite = val_mt_rel_err < best_val_mt_rel_err
        if is_best_composite:
            best_val_mt_rel_err = val_mt_rel_err
            best_composite_epoch = epoch

        if str(selection_metric).lower() in ("val_mt_rel_err", "composite", "mt_rel_err"):
            improved = is_best_composite or is_best_loss
        else:
            improved = is_best_loss

        if improved:
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        fn = f"multitask_epoch_{epoch:03d}.pt"
        saved_ckpt = trainer.save_checkpoint(
            epoch,
            val_metrics,
            is_best=is_best_loss,
            is_best_composite=is_best_composite,
            filename=fn,
            extra_metadata={
                "experiment": experiment_name or "supervised_multitask_training",
                "config_hash": config_hash,
                "initialization": init_provenance,
                "transfer_info": transfer_info,
                "use_sequences": use_sequences,
                "is_best_loss": is_best_loss,
                "is_best_composite": is_best_composite,
                "best_val_loss_global": round(best_val_loss_global, 4),
                "best_val_mt_rel_err": round(best_val_mt_rel_err, 4),
                "train_eval_loss_global_subset": train_eval_loss_global,
                "config": config_payload_for_hash,
                "software_versions": software_versions,
            },
        )

        gen_gap_subset = round(val_loss_global - train_eval_loss_global, 4) if train_eval_loss_global is not None else None

        epoch_record = {
            "epoch": epoch,
            "backbone_lr": backbone_lr_now,
            "head_lr": head_lr_now,
            "train": train_metrics,
            "train_eval_subset": train_eval_metrics,
            "train_eval_loss_global": train_eval_loss_global,
            "val": val_metrics,
            "val_loss_global": val_loss_global,
            "val_loss_batch": val_loss_batch,
            "val_mt_rel_err": val_mt_rel_err,
            "generalization_gap_global": gen_gap_subset,
            "is_best_loss": is_best_loss,
            "is_best_composite": is_best_composite,
            "epochs_since_improvement": epochs_since_improvement,
            "saved_checkpoint": saved_ckpt,
        }
        training_history.append(epoch_record)

        epoch_csv_rows.append({
            "epoch": epoch,
            "backbone_lr": backbone_lr_now,
            "head_lr": head_lr_now,
            "train_opt_loss": train_metrics.get("loss_total"),
            "train_eval_loss_global": train_eval_loss_global,
            "val_loss_global": val_loss_global,
            "val_loss_batch": val_loss_batch,
            "generalization_gap_global": gen_gap_subset,
            "val_mt_rel_err": val_mt_rel_err,
            "val_wind_mae_kt": val_metrics.get("val_wind_mae_kt"),
            "val_wind_bias_kt": val_metrics.get("val_wind_bias_kt"),
            "val_major_wind_mae_kt": val_metrics.get("val_major_wind_mae_kt"),
            "val_major_wind_bias_kt": val_metrics.get("val_major_wind_bias_kt"),
            "val_pressure_mae_hpa": val_metrics.get("val_pressure_mae_hpa"),
            "val_stage_balanced_acc_pct": val_metrics.get("val_stage_balanced_acc_pct"),
            "val_stage_acc_pct": val_metrics.get("val_stage_acc_pct"),
            "val_dv_6h_mae_kt": val_metrics.get("val_dv_6h_mae_kt"),
            "val_dv_12h_mae_kt": val_metrics.get("val_dv_12h_mae_kt"),
            "val_dv_18h_mae_kt": val_metrics.get("val_dv_18h_mae_kt"),
            "val_dv_24h_mae_kt": val_metrics.get("val_dv_24h_mae_kt"),
            "val_ri_pr_auc": val_metrics.get("val_ri_pr_auc"),
            "val_ri_brier": val_metrics.get("val_ri_brier"),
            "val_ri_f1": val_metrics.get("val_ri_f1"),
            "val_track_24h_err_km": val_metrics.get("val_track_24h_err_km"),
            "train_major_hurricane_share": train_metrics.get("major_hurricane_share"),
            "grad_norm_preclip_mean": train_metrics.get("grad_norm_preclip_mean"),
            "grad_clipped_fraction_pct": train_metrics.get("grad_clipped_fraction_pct"),
            "is_best_loss": is_best_loss,
            "is_best_composite": is_best_composite,
            "epoch_time_sec": train_metrics.get("epoch_time_sec"),
        })

        if epoch_csv_report:
            os.makedirs(os.path.dirname(epoch_csv_report), exist_ok=True)
            pd.DataFrame(epoch_csv_rows).to_csv(epoch_csv_report, index=False)

        tags = []
        if is_best_loss:
            tags.append("★BEST_LOSS")
        if is_best_composite:
            tags.append("★BEST_COMPOSITE")
        tag_str = (" " + " ".join(tags)) if tags else ""

        print(
            f"[Train] OptLoss: {train_metrics['loss_total']:.4f} | TrainEval: {train_eval_loss_global if train_eval_loss_global is not None else -1.0:.4f} | "
            f"MajShare: {100.0 * train_metrics.get('major_hurricane_share', 0.0):.2f}% | GradMean: {train_metrics.get('grad_norm_preclip_mean', 0.0):.2f}",
            flush=True,
        )
        print(
            f"[Val]   GlobalLoss: {val_loss_global:.4f} (Batch: {val_loss_batch:.4f}) | MT_REL_ERR: {val_mt_rel_err:.4f} | "
            f"Wind: {val_metrics['val_wind_mae_kt']} kt (Maj: {val_metrics.get('val_major_wind_mae_kt')} / Bias {val_metrics.get('val_major_wind_bias_kt')}) | "
            f"Press: {val_metrics['val_pressure_mae_hpa']} hPa | Stage Bal/Acc: {val_metrics.get('val_stage_balanced_acc_pct')}%/{val_metrics['val_stage_acc_pct']}% | "
            f"dV24: {val_metrics.get('val_dv_24h_mae_kt')} kt | RI PR-AUC/Brier: {val_metrics.get('val_ri_pr_auc')}/{val_metrics.get('val_ri_brier')} | "
            f"Track24: {val_metrics.get('val_track_24h_err_km')} km{tag_str}",
            flush=True,
        )

        # Save rolling summary report after every epoch
        if summary_report:
            os.makedirs(os.path.dirname(summary_report), exist_ok=True)
            with open(summary_report, "w", encoding="utf-8") as f:
                json.dump({
                    "workflow": "Supervised Multi-Task Training (Phase 2E.4.1)",
                    "experiment": experiment_name or "supervised_multitask_training",
                    "config_hash": config_hash,
                    "software_versions": software_versions,
                    "config": config_payload_for_hash,
                    "sampler_info": sampler_info,
                    "param_group_counts": param_group_counts,
                    "best_loss_epoch": best_loss_epoch,
                    "best_val_loss_global": round(best_val_loss_global, 4),
                    "best_composite_epoch": best_composite_epoch,
                    "best_val_mt_rel_err": round(best_val_mt_rel_err, 4),
                    "history": training_history,
                }, f, indent=2)

        if early_stop_patience is not None and int(early_stop_patience) > 0 and epochs_since_improvement >= int(early_stop_patience):
            early_stopped = True
            stop_reason = (
                f"Early stopping triggered at Epoch {epoch}: no improvement in VAL_MT_REL_ERR "
                f"(best Ep {best_composite_epoch}: {best_val_mt_rel_err:.4f}) or VAL_LOSS_GLOBAL "
                f"(best Ep {best_loss_epoch}: {best_val_loss_global:.4f}) for {early_stop_patience} consecutive epochs."
            )
            print(f"\n[EARLY STOP] {stop_reason}", flush=True)
            break

    # 5. Full 10,545-sample TRAIN_EVAL passes on the best-loss and best-composite checkpoints
    full_train_eval_results = {}
    if full_train_eval_on_best and train_eval_full_loader is not None:
        best_ckpts_to_eval = {}
        best_loss_p = os.path.join(checkpoint_dir, "best_checkpoint.pt")
        best_comp_p = os.path.join(checkpoint_dir, "best_mt_rel_err_checkpoint.pt")
        if os.path.exists(best_loss_p):
            best_ckpts_to_eval["best_global_loss_checkpoint"] = (best_loss_epoch, best_loss_p)
        if os.path.exists(best_comp_p):
            if best_composite_epoch == best_loss_epoch and "best_global_loss_checkpoint" in best_ckpts_to_eval:
                best_ckpts_to_eval["best_composite_checkpoint"] = (best_composite_epoch, best_comp_p)
            else:
                best_ckpts_to_eval["best_composite_checkpoint"] = (best_composite_epoch, best_comp_p)

        evaluated_by_epoch = {}
        for label, (ep_num, ck_p) in best_ckpts_to_eval.items():
            if ep_num in evaluated_by_epoch:
                full_train_eval_results[label] = evaluated_by_epoch[ep_num]
                continue
            print(f"\n[FULL TRAIN_EVAL] Evaluating full TRAIN partition (10,545 sequences) on {label} (Epoch {ep_num})...", flush=True)
            ck_loaded = torch.load(ck_p, map_location=device, weights_only=False)
            trainer.model.load_state_dict(ck_loaded["model_state_dict"], strict=True)
            tr_full_m = trainer.evaluate(train_eval_full_loader, use_ema=False)
            val_m_for_ep = next((h["val"] for h in training_history if h["epoch"] == ep_num), {})
            val_g = float(val_m_for_ep.get("val_loss_global", best_val_loss_global))
            tr_g = float(tr_full_m.get("val_loss_global", tr_full_m["val_loss"]))
            res_entry = {
                "epoch": ep_num,
                "checkpoint": ck_p,
                "train_eval_loss_global_full": round(tr_g, 4),
                "val_eval_loss_global": round(val_g, 4),
                "generalization_gap_global_full": round(val_g - tr_g, 4),
                "train_eval_metrics_full": tr_full_m,
                "val_eval_metrics": val_m_for_ep,
            }
            evaluated_by_epoch[ep_num] = res_entry
            full_train_eval_results[label] = res_entry

    summary = {
        "workflow": "Supervised Multi-Task Training (Phase 2E.4.1)",
        "experiment": experiment_name or "supervised_multitask_training",
        "config_hash": config_hash,
        "software_versions": software_versions,
        "initialization": init_provenance,
        "transfer_info": transfer_info,
        "use_sequences": use_sequences,
        "epochs": epochs,
        "cosine_t_max": t_max_sched,
        "batch_size": batch_size,
        "learning_rate": lr,
        "backbone_lr_scale": backbone_lr_scale,
        "weight_decay": weight_decay,
        "decoupled_weight_decay": decoupled_weight_decay,
        "d_model": d_model,
        "grad_clip_norm": grad_clip_norm,
        "accumulate_grad_batches": accumulate_grad_batches,
        "loss_weights": lw,
        "r1_regularization": {
            "use_ema": use_ema,
            "ema_decay": ema_decay,
            "modality_dropout_prob": modality_dropout_prob,
            "sector_token_dropout": sector_token_dropout,
            "drop_path_rate": drop_path_rate,
            "use_storm_balanced_sampler": use_storm_balanced_sampler,
            "sampler_info": sampler_info,
            "param_group_counts": param_group_counts,
        },
        "best_val_loss": round(best_val_loss_global, 4),
        "best_val_loss_global": round(best_val_loss_global, 4),
        "best_loss_epoch": best_loss_epoch,
        "best_val_mt_rel_err": round(best_val_mt_rel_err, 4),
        "best_composite_epoch": best_composite_epoch,
        "early_stopped": early_stopped,
        "stop_reason": stop_reason,
        "full_train_eval_on_best": full_train_eval_results,
        "history": training_history,
    }

    if summary_report:
        os.makedirs(os.path.dirname(summary_report), exist_ok=True)
        with open(summary_report, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    print(f"\n[OK] Multi-task training complete. Summary saved to: {summary_report}", flush=True)
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
    parser.add_argument("--epoch-csv-report", default=None, help="Per-epoch CSV log path")
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
    exp_cfg = cfg.get("experiment", {})
    loss_weights = cfg.get("loss_weights", {})
    mach_cfg = cfg.get("machine", {})
    paths_cfg = cfg.get("paths", {})

    def resolve_val(cli_val, yaml_val, default):
        if cli_val is not None:
            return cli_val
        if yaml_val is not None:
            return yaml_val
        return default

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

    tr_workers = tr_cfg.get("num_workers")
    yaml_workers = tr_workers if tr_workers is not None else mach_cfg.get("num_workers")
    resolved_workers = resolve_val(getattr(args, "num_workers", None), yaml_workers, 0)

    resolved_pin_mem = resolve_val(getattr(args, "pin_memory", None), tr_cfg.get("pin_memory"), None)
    resolved_persist = resolve_val(getattr(args, "persistent_workers", None), tr_cfg.get("persistent_workers"), False)
    resolved_ckpt_dir = resolve_val(getattr(args, "checkpoint_dir", None), tr_cfg.get("checkpoint_dir") or paths_cfg.get("checkpoints_dir"), "checkpoints")
    resolved_report = resolve_val(getattr(args, "summary_report", None), tr_cfg.get("output_report") or tr_cfg.get("summary_report") or paths_cfg.get("training_summary_path"), "reports/phase2e4/multitask_training_summary.json")
    resolved_epoch_csv = resolve_val(getattr(args, "epoch_csv_report", None), tr_cfg.get("epoch_csv_report") or paths_cfg.get("epoch_csv_path"), None)
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
        "epoch_csv_report": resolved_epoch_csv,
        "device": resolved_device,
        "cuda_device_id": resolved_cuda_id,
        "use_amp": resolved_amp,
        "precision": resolved_precision,
        "seed": resolved_seed,
        "loss_weights": loss_weights,
        "dry_run": getattr(args, "dry_run", False),
        # R1 feature-flagged options
        "use_ema": bool(tr_cfg.get("use_ema", False)),
        "ema_decay": float(tr_cfg.get("ema_decay", 0.999)),
        "modality_dropout_prob": float(tr_cfg.get("modality_dropout_prob", 0.0)),
        "sector_token_dropout": float(tr_cfg.get("sector_token_dropout", 0.0)),
        "drop_path_rate": float(tr_cfg.get("drop_path_rate", 0.0)),
        "head_dropout": float(tr_cfg.get("head_dropout", 0.0)),
        "use_storm_balanced_sampler": bool(tr_cfg.get("use_storm_balanced_sampler", False)),
        "storm_balance_exponent": float(tr_cfg.get("storm_balance_exponent", 0.5)),
        "storm_balance_min_major_ratio": float(tr_cfg.get("storm_balance_min_major_ratio", 0.80)),
        "storm_balance_min_exponent": float(tr_cfg.get("storm_balance_min_exponent", 0.25)),
        "backbone_lr_scale": float(tr_cfg.get("backbone_lr_scale", 1.0)),
        "decoupled_weight_decay": bool(tr_cfg.get("decoupled_weight_decay", False)),
        "cosine_t_max": tr_cfg.get("cosine_t_max"),
        "eta_min": float(tr_cfg.get("eta_min", 1e-6)),
        "early_stop_patience": tr_cfg.get("early_stop_patience"),
        "selection_metric": str(tr_cfg.get("selection_metric", "val_loss")),
        "train_eval_subset_size": int(tr_cfg.get("train_eval_subset_size", 0)),
        "full_train_eval_on_best": bool(tr_cfg.get("full_train_eval_on_best", False)),
        "experiment_name": exp_cfg.get("name") or tr_cfg.get("experiment_name"),
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
        epoch_csv_report=resolved["epoch_csv_report"],
        device=resolved["device"],
        cuda_device_id=resolved["cuda_device_id"],
        use_amp=resolved["use_amp"],
        precision=resolved["precision"],
        seed=resolved["seed"],
        loss_weights=resolved["loss_weights"],
        dry_run=resolved["dry_run"],
        use_ema=resolved["use_ema"],
        ema_decay=resolved["ema_decay"],
        modality_dropout_prob=resolved["modality_dropout_prob"],
        sector_token_dropout=resolved["sector_token_dropout"],
        drop_path_rate=resolved["drop_path_rate"],
        head_dropout=resolved["head_dropout"],
        use_storm_balanced_sampler=resolved["use_storm_balanced_sampler"],
        storm_balance_exponent=resolved["storm_balance_exponent"],
        storm_balance_min_major_ratio=resolved["storm_balance_min_major_ratio"],
        storm_balance_min_exponent=resolved["storm_balance_min_exponent"],
        backbone_lr_scale=resolved["backbone_lr_scale"],
        decoupled_weight_decay=resolved["decoupled_weight_decay"],
        cosine_t_max=resolved["cosine_t_max"],
        eta_min=resolved["eta_min"],
        early_stop_patience=resolved["early_stop_patience"],
        selection_metric=resolved["selection_metric"],
        train_eval_subset_size=resolved["train_eval_subset_size"],
        full_train_eval_on_best=resolved["full_train_eval_on_best"],
        experiment_name=resolved["experiment_name"],
    )


if __name__ == "__main__":
    main()
