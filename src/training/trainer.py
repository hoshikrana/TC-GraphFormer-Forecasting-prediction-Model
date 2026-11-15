"""
Hardware-Conscious Training Engine for TC-GraphFormer.
Supports CPU and CUDA/AMP mixed-precision execution, gradient clipping,
checkpoint serialization, and scientifically rigorous masked multi-task validation metrics.
"""

import os
import time
import math
from collections import defaultdict
from typing import Dict, Any, Optional, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .losses import MultiTaskLoss


class TCGraphFormerTrainer:
    """
    Standard Trainer for TC-GraphFormer model supporting single-overpasses and sequences.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Optional[MultiTaskLoss] = None,
        scheduler: Optional[Any] = None,
        device: str = "cpu",
        checkpoint_dir: str = "checkpoints",
        use_amp: bool = False,
        precision: str = "fp16",
        grad_clip_norm: float = 1.0,
        accumulate_grad_batches: int = 1,
        best_val_loss: float = float("inf"),
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn if loss_fn is not None else MultiTaskLoss().to(device)
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.precision = precision.lower() if precision else "fp16"
        self.use_amp = use_amp and (device.startswith("cuda"))
        self.amp_dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16
        self.grad_clip_norm = grad_clip_norm
        self.accumulate_grad_batches = max(1, int(accumulate_grad_batches))
        # GradScaler is primarily used for fp16; bf16 has wider dynamic range
        scaler_enabled = self.use_amp and (self.precision != "bf16")
        self.scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

        os.makedirs(checkpoint_dir, exist_ok=True)
        self.best_val_loss = best_val_loss
        self.history = []

    def _forward_batch(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Route single-observation or sequence batch to model forward pass."""
        ir = batch["ir"].to(self.device)
        pmw = batch["pmw"].to(self.device)
        gprof = batch["gprof"].to(self.device)
        radar = batch["radar"].to(self.device)
        env = batch["env"].to(self.device)
        masks = batch["modality_masks"].to(self.device)

        ch_masks = None
        if "channel_masks" in batch and batch["channel_masks"]:
            ch_masks = {k: v.to(self.device) for k, v in batch["channel_masks"].items()}

        pix_masks = None
        if "pixel_masks" in batch and batch["pixel_masks"]:
            pix_masks = {k: v.to(self.device) for k, v in batch["pixel_masks"].items()}

        delta_t = batch.get("delta_times_hours")
        if delta_t is not None:
            delta_t = delta_t.to(self.device)

        seq_pad = batch.get("seq_pad_mask")
        if seq_pad is not None:
            seq_pad = seq_pad.to(self.device)

        preds = self.model(
            ir=ir,
            pmw=pmw,
            gprof=gprof,
            radar=radar,
            env=env,
            modality_masks=masks,
            delta_times_hours=delta_t,
            seq_pad_mask=seq_pad,
            channel_masks=ch_masks,
            pixel_masks=pix_masks,
        )
        return preds

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Execute one training epoch with sample-weighted gradient accumulation and norm tracking."""
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        loss_accum = {}
        task_loss_sums = defaultdict(float)
        task_valid_counts = defaultdict(float)

        start_time = time.time()
        self.optimizer.zero_grad()
        num_batches = len(dataloader)
        optimizer_steps = 0

        # Import MultiTaskLoss check
        from src.training.losses import MultiTaskLoss

        group_batches = []
        for batch_idx, batch in enumerate(dataloader):
            group_batches.append(batch)
            is_last_batch = (batch_idx + 1 == num_batches)

            if len(group_batches) == self.accumulate_grad_batches or is_last_batch:
                group_total_samples = sum(b["ir"].size(0) for b in group_batches)

                # Precompute valid target counts across group for MultiTaskLoss
                group_target_counts = None
                if isinstance(self.loss_fn, MultiTaskLoss):
                    group_target_counts = {}
                    for b in group_batches:
                        tgts = b.get("targets", {})
                        if "intensity_mask" in tgts:
                            group_target_counts["intensity"] = group_target_counts.get("intensity", 0.0) + float(tgts["intensity_mask"].sum().item())
                        elif "intensity" in tgts:
                            group_target_counts["intensity"] = group_target_counts.get("intensity", 0.0) + float(tgts["intensity"].size(0))

                        if "pressure_mask" in tgts:
                            group_target_counts["pressure"] = group_target_counts.get("pressure", 0.0) + float(tgts["pressure_mask"].sum().item())
                        elif "pressure" in tgts:
                            group_target_counts["pressure"] = group_target_counts.get("pressure", 0.0) + float(tgts["pressure"].size(0))

                        if "stage_mask" in tgts:
                            group_target_counts["stage"] = group_target_counts.get("stage", 0.0) + float(tgts["stage_mask"].sum().item())
                        elif "stage" in tgts:
                            group_target_counts["stage"] = group_target_counts.get("stage", 0.0) + float(tgts["stage"].size(0))

                        if "intensity_change_mask" in tgts:
                            group_target_counts["intensity_change"] = group_target_counts.get("intensity_change", 0.0) + float(tgts["intensity_change_mask"].sum().item())
                        elif "intensity_change" in tgts:
                            group_target_counts["intensity_change"] = group_target_counts.get("intensity_change", 0.0) + float(tgts["intensity_change"].numel())

                        if "ri_mask" in tgts:
                            group_target_counts["rapid_intensification"] = group_target_counts.get("rapid_intensification", 0.0) + float(tgts["ri_mask"].sum().item())
                        elif "rapid_intensification" in tgts:
                            group_target_counts["rapid_intensification"] = group_target_counts.get("rapid_intensification", 0.0) + float(tgts["rapid_intensification"].size(0))

                        if "track_mask" in tgts:
                            group_target_counts["track_displacement"] = group_target_counts.get("track_displacement", 0.0) + float(tgts["track_mask"].sum().item() * 2.0)
                        elif "track_displacement" in tgts:
                            group_target_counts["track_displacement"] = group_target_counts.get("track_displacement", 0.0) + float(tgts["track_displacement"].numel())

                for b in group_batches:
                    targets = {k: v.to(self.device) for k, v in b["targets"].items() if isinstance(v, torch.Tensor)}
                    B = b["ir"].size(0)

                    with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                        preds = self._forward_batch(b)
                        if isinstance(self.loss_fn, MultiTaskLoss):
                            loss, loss_dict = self.loss_fn(preds, targets, group_target_counts=group_target_counts)
                            loss_scaled = loss
                        else:
                            loss, loss_dict = self.loss_fn(preds, targets)
                            loss_scaled = loss * (B / max(1, group_total_samples))

                    if self.use_amp:
                        self.scaler.scale(loss_scaled).backward()
                    else:
                        loss_scaled.backward()

                    total_samples += B
                    total_loss += loss.item() * B
                    for k, v in loss_dict.items():
                        loss_accum[k] = loss_accum.get(k, 0.0) + v * B

                    if isinstance(self.loss_fn, MultiTaskLoss):
                        task_loss_sums["intensity"] += loss_dict.get("unreduced_loss_intensity", 0.0)
                        task_valid_counts["intensity"] += loss_dict.get("valid_intensity_count", loss_dict.get("valid_v_count", 0.0))

                        task_loss_sums["pressure"] += loss_dict.get("unreduced_loss_pressure", 0.0)
                        task_valid_counts["pressure"] += loss_dict.get("valid_pressure_count", loss_dict.get("valid_p_count", 0.0))

                        task_loss_sums["stage"] += loss_dict.get("unreduced_loss_stage", 0.0)
                        task_valid_counts["stage"] += loss_dict.get("valid_stage_count", loss_dict.get("valid_s_count", 0.0))

                        task_loss_sums["intensity_change"] += loss_dict.get("unreduced_loss_intensity_change", 0.0)
                        task_valid_counts["intensity_change"] += loss_dict.get("valid_intensity_change_count", loss_dict.get("valid_dv_count", 0.0))

                        task_loss_sums["ri"] += loss_dict.get("unreduced_loss_ri", 0.0)
                        task_valid_counts["ri"] += loss_dict.get("valid_ri_count", 0.0)

                        task_loss_sums["track"] += loss_dict.get("unreduced_loss_track", 0.0)
                        task_valid_counts["track"] += loss_dict.get("valid_track_count", 0.0)

                # Step optimizer once for the entire accumulation group
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    optimizer_steps += 1
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    optimizer_steps += 1

                group_batches = []

        if isinstance(self.loss_fn, MultiTaskLoss):
            task_weights = getattr(self.loss_fn, "task_weights", {
                "intensity": self.loss_fn.w_v,
                "pressure": self.loss_fn.w_p,
                "stage": self.loss_fn.w_s,
                "intensity_change": self.loss_fn.w_dv,
                "ri": self.loss_fn.w_ri,
                "track": self.loss_fn.w_track,
            })
            metrics = {}
            weighted_total = 0.0
            for task_name, weight in task_weights.items():
                valid_c = task_valid_counts[task_name]
                task_epoch_loss = (task_loss_sums[task_name] / max(1.0, valid_c)) if valid_c > 0 else 0.0
                metrics[f"loss_{task_name}"] = round(task_epoch_loss, 4)
                weighted_total += weight * task_epoch_loss

            metrics["loss_total"] = round(weighted_total, 4)
            for k, v in loss_accum.items():
                if k not in metrics and not k.startswith("unreduced_") and not k.startswith("valid_"):
                    metrics[k] = round(v / max(1, total_samples), 4)
        else:
            metrics = {k: round(v / max(1, total_samples), 4) for k, v in loss_accum.items()}

        metrics["epoch"] = epoch
        metrics["optimizer_steps"] = optimizer_steps
        metrics["total_batches"] = num_batches
        metrics["accumulate_grad_batches"] = self.accumulate_grad_batches
        metrics["epoch_time_sec"] = round(time.time() - start_time, 2)
        return metrics

    def evaluate(self, dataloader: DataLoader) -> Dict[str, Any]:
        """
        Evaluate model on validation or test partition with strict target masking.
        Computes regression MAE/RMSE/bias, classification Macro-F1/balanced accuracy,
        and binary RI PR metrics.
        """
        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        # Intensity accumulators
        v_errs, v_sq_errs = [], []
        # Pressure accumulators
        p_errs, p_sq_errs = [], []
        # Stage metrics
        stage_true_list, stage_pred_list = [], []
        # Multi-horizon intensity change
        dv_errs_by_h = {h: [] for h in [6, 12, 18, 24]}
        dv_sq_errs_by_h = {h: [] for h in [6, 12, 18, 24]}
        # RI metrics
        ri_true_list, ri_prob_list = [], []
        # Track displacement errors (km)
        tr_errs_by_h = {h: [] for h in [6, 12, 18, 24]}

        with torch.no_grad():
            for batch in dataloader:
                targets = {k: v.to(self.device) for k, v in batch["targets"].items() if isinstance(v, torch.Tensor)}

                preds = self._forward_batch(batch)
                loss, _ = self.loss_fn(preds, targets)

                B = batch["ir"].size(0)
                total_samples += B
                total_loss += loss.item() * B

                # 1. Intensity (Wind Speed kt)
                v_pred = preds["intensity"].cpu().numpy()
                v_true = targets["intensity"].cpu().numpy()
                v_mask = targets["intensity_mask"].cpu().numpy() > 0.5
                if np.any(v_mask):
                    e_v = v_pred[v_mask] - v_true[v_mask]
                    v_errs.extend(e_v.tolist())
                    v_sq_errs.extend((e_v ** 2).tolist())

                # 2. Central Minimum Pressure (hPa)
                p_pred = preds["pressure"].cpu().numpy()
                p_true = targets["pressure"].cpu().numpy()
                p_mask = targets["pressure_mask"].cpu().numpy() > 0.5
                if np.any(p_mask):
                    e_p = p_pred[p_mask] - p_true[p_mask]
                    p_errs.extend(e_p.tolist())
                    p_sq_errs.extend((e_p ** 2).tolist())

                # 3. Development Stage Classification
                s_pred = preds["stage_logits"].argmax(dim=-1).cpu().numpy()
                s_true = targets["stage"].cpu().numpy()
                s_mask = targets["stage_mask"].cpu().numpy() > 0.5
                if np.any(s_mask):
                    stage_true_list.extend(s_true[s_mask].tolist())
                    stage_pred_list.extend(s_pred[s_mask].tolist())

                # 4. Multi-Horizon Intensity Change
                dv_pred = preds["intensity_change"].cpu().numpy() # (B, 4)
                dv_true = targets["intensity_change"].cpu().numpy()
                dv_mask = targets["intensity_change_mask"].cpu().numpy() > 0.5
                for h_idx, h in enumerate([6, 12, 18, 24]):
                    m_h = dv_mask[:, h_idx]
                    if np.any(m_h):
                        e_dv = dv_pred[m_h, h_idx] - dv_true[m_h, h_idx]
                        dv_errs_by_h[h].extend(e_dv.tolist())
                        dv_sq_errs_by_h[h].extend((e_dv ** 2).tolist())

                # 5. Rapid Intensification (RI)
                ri_prob = torch.sigmoid(preds["ri_logits"]).cpu().numpy()
                ri_true = targets["rapid_intensification"].cpu().numpy()
                ri_mask = targets["ri_mask"].cpu().numpy() > 0.5
                if np.any(ri_mask):
                    ri_true_list.extend(ri_true[ri_mask].tolist())
                    ri_prob_list.extend(ri_prob[ri_mask].tolist())

                # 6. Future Track Displacement Error (km)
                tr_pred = preds["track_delta"].view(-1, 4, 2).cpu().numpy() # (B, 4, 2)
                tr_true = targets["track_displacement"].cpu().numpy()       # (B, 4, 2)
                tr_mask = targets["track_mask"].cpu().numpy() > 0.5         # (B, 4)
                for h_idx, h in enumerate([6, 12, 18, 24]):
                    m_tr = tr_mask[:, h_idx]
                    if np.any(m_tr):
                        dx_err = tr_pred[m_tr, h_idx, 0] - tr_true[m_tr, h_idx, 0]
                        dy_err = tr_pred[m_tr, h_idx, 1] - tr_true[m_tr, h_idx, 1]
                        dist_err = np.sqrt(dx_err ** 2 + dy_err ** 2)
                        tr_errs_by_h[h].extend(dist_err.tolist())

        # Compile Metrics
        val_metrics: Dict[str, Any] = {
            "val_loss": round(total_loss / max(1, total_samples), 4),
            "total_val_samples": total_samples,
        }

        # Wind stats
        if v_errs:
            val_metrics["val_wind_mae_kt"] = round(float(np.mean(np.abs(v_errs))), 2)
            val_metrics["val_wind_rmse_kt"] = round(float(np.sqrt(np.mean(v_sq_errs))), 2)
            val_metrics["val_wind_bias_kt"] = round(float(np.mean(v_errs)), 2)
            val_metrics["valid_wind_samples"] = len(v_errs)
        else:
            val_metrics["val_wind_mae_kt"] = 0.0

        # Pressure stats
        if p_errs:
            val_metrics["val_pressure_mae_hpa"] = round(float(np.mean(np.abs(p_errs))), 2)
            val_metrics["val_pressure_rmse_hpa"] = round(float(np.sqrt(np.mean(p_sq_errs))), 2)
            val_metrics["val_pressure_bias_hpa"] = round(float(np.mean(p_errs)), 2)
            val_metrics["valid_pressure_samples"] = len(p_errs)
        else:
            val_metrics["val_pressure_mae_hpa"] = 0.0

        # Stage classification metrics (Macro-F1, balanced accuracy)
        if stage_true_list:
            y_s_true = np.array(stage_true_list)
            y_s_pred = np.array(stage_pred_list)
            val_metrics["val_stage_acc_pct"] = round(float(np.mean(y_s_true == y_s_pred)) * 100.0, 2)

            # Per-class recalls for balanced accuracy
            classes = np.unique(y_s_true)
            recalls = []
            for c in classes:
                mask_c = y_s_true == c
                if np.sum(mask_c) > 0:
                    rec_c = float(np.mean(y_s_pred[mask_c] == c))
                    recalls.append(rec_c)
            val_metrics["val_stage_balanced_acc_pct"] = round(float(np.mean(recalls)) * 100.0, 2) if recalls else 0.0
            val_metrics["valid_stage_samples"] = len(stage_true_list)
        else:
            val_metrics["val_stage_acc_pct"] = 0.0

        # Intensity change stats
        for h in [6, 12, 18, 24]:
            if dv_errs_by_h[h]:
                val_metrics[f"val_dv_{h}h_mae_kt"] = round(float(np.mean(np.abs(dv_errs_by_h[h]))), 2)
                val_metrics[f"val_dv_{h}h_rmse_kt"] = round(float(np.sqrt(np.mean(dv_sq_errs_by_h[h]))), 2)
                val_metrics[f"valid_dv_{h}h_samples"] = len(dv_errs_by_h[h])

        # RI binary classification metrics
        if ri_true_list:
            y_ri = np.array(ri_true_list)
            p_ri = np.array(ri_prob_list)
            pred_ri = (p_ri >= 0.5).astype(float)

            tp = float(np.sum((pred_ri == 1.0) & (y_ri == 1.0)))
            fp = float(np.sum((pred_ri == 1.0) & (y_ri == 0.0)))
            fn = float(np.sum((pred_ri == 0.0) & (y_ri == 1.0)))
            tn = float(np.sum((pred_ri == 0.0) & (y_ri == 0.0)))

            precision = tp / max(1e-5, (tp + fp))
            recall = tp / max(1e-5, (tp + fn))
            f1 = (2 * precision * recall) / max(1e-5, (precision + recall))
            far = fp / max(1e-5, (tp + fp)) # False alarm ratio

            val_metrics["val_ri_precision"] = round(precision, 4)
            val_metrics["val_ri_recall"] = round(recall, 4)
            val_metrics["val_ri_f1"] = round(f1, 4)
            val_metrics["val_ri_far"] = round(far, 4)
            val_metrics["val_ri_valid_pos"] = int(np.sum(y_ri == 1.0))
            val_metrics["val_ri_valid_neg"] = int(np.sum(y_ri == 0.0))
        else:
            val_metrics["val_ri_f1"] = 0.0

        # Track error stats (km)
        for h in [6, 12, 18, 24]:
            if tr_errs_by_h[h]:
                val_metrics[f"val_track_{h}h_err_km"] = round(float(np.mean(tr_errs_by_h[h])), 2)
                val_metrics[f"valid_track_{h}h_samples"] = len(tr_errs_by_h[h])

        return val_metrics

    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, Any],
        is_best: bool = False,
        filename: str = None,
        extra_metadata: Dict[str, Any] = None,
    ) -> str:
        """
        Serialize model weights, optimizer state, scheduler state, AMP scaler state,
        historical best validation loss, metrics, and explicit training provenance.
        """
        if filename is None:
            filename = f"checkpoint_epoch_{epoch:03d}.pt"

        if is_best and "val_loss" in metrics:
            self.best_val_loss = float(metrics["val_loss"])

        filepath = os.path.join(self.checkpoint_dir, filename)
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            "scaler_state_dict": self.scaler.state_dict() if (self.use_amp and self.scaler is not None) else None,
            "best_val_loss": self.best_val_loss,
            "metrics": metrics,
            "d_model": getattr(self.model, "d_model", 128),
            "device": self.device,
            "precision": self.precision,
            "is_trained": True,
            "provenance": {
                "workflow": "supervised_multitask_training",
                "is_trained": True,
                "epoch": epoch,
                "best_val_loss": self.best_val_loss,
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "has_scheduler_state": self.scheduler is not None,
                "has_scaler_state": (self.use_amp and self.scaler is not None),
            },
            "extra_metadata": extra_metadata or {},
        }
        torch.save(state, filepath)

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_checkpoint.pt")
            torch.save(state, best_path)

        latest_path = os.path.join(self.checkpoint_dir, "latest_checkpoint.pt")
        torch.save(state, latest_path)

        return filepath

    def restore_checkpoint(self, checkpoint_path: str, allow_legacy_resume: bool = False) -> Dict[str, Any]:
        """Restore model, optimizer, scheduler, and AMP scaler state from checkpoint."""
        res = restore_checkpoint_state(
            checkpoint_path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            use_amp=self.use_amp,
            precision=self.precision,
            allow_legacy_resume=allow_legacy_resume,
            device=self.device,
        )
        self.best_val_loss = res["best_val_loss"]
        return res


def restore_checkpoint_state(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    use_amp: bool = False,
    precision: str = "fp16",
    allow_legacy_resume: bool = False,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Safely restore complete checkpoint state for model training resumption.
    Strictly requires and restores:
      - Model parameters (strictly compatible)
      - Optimizer state (unless allow_legacy_resume=True)
      - Learning-rate scheduler state (unless allow_legacy_resume=True)
      - Historical best validation loss (unless allow_legacy_resume=True)
      - AMP GradScaler state when using fp16 AMP (unless allow_legacy_resume=True)
      - Training epoch and provenance metadata

    Raises ValueError if any required state is missing, preventing silent loss
    of training momentum, learning rate progression, or historical best records.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Invalid checkpoint format in '{checkpoint_path}': expected dict, got {type(ckpt)}")

    is_legacy_resume = False

    # 1. Model State
    state = ckpt.get("model_state_dict", ckpt)
    model_state = model.state_dict()
    model_keys = set(model_state.keys())
    ckpt_keys = set(state.keys())

    matched_keys = []
    mismatched_shapes = []
    for k in ckpt_keys:
        if k in model_keys:
            if model_state[k].shape == state[k].shape:
                matched_keys.append(k)
            else:
                mismatched_shapes.append(k)

    missing_keys = sorted(list(model_keys - set(matched_keys)))
    if missing_keys or mismatched_shapes:
        raise ValueError(
            f"Checkpoint '{checkpoint_path}' is incompatible with model! "
            f"Missing keys: {len(missing_keys)}, Shape mismatches: {len(mismatched_shapes)}."
        )
    model.load_state_dict(state, strict=True)

    # 2. Training Epoch
    epoch = ckpt.get("epoch")
    if epoch is None:
        if not allow_legacy_resume:
            raise ValueError(
                f"Checkpoint '{checkpoint_path}' lacks 'epoch' required for safe resume! "
                f"Pass allow_legacy_resume=True to override."
            )
        epoch = 0
        is_legacy_resume = True

    # 3. Optimizer State
    opt_state = ckpt.get("optimizer_state_dict")
    has_valid_opt = (
        opt_state is not None
        and isinstance(opt_state, dict)
        and "param_groups" in opt_state
    )
    if optimizer is not None:
        if has_valid_opt:
            optimizer.load_state_dict(opt_state)
        elif not allow_legacy_resume:
            raise ValueError(
                f"Checkpoint '{checkpoint_path}' lacks 'optimizer_state_dict' required for safe resume! "
                f"Resuming without optimizer state loses momentum and variance buffers. "
                f"Pass allow_legacy_resume=True to override."
            )
        else:
            is_legacy_resume = True
            print(f"[WARN] Checkpoint '{checkpoint_path}' lacks optimizer state; continuing with fresh optimizer under legacy resume.")

    # 4. Scheduler State
    sched_state = ckpt.get("scheduler_state_dict")
    has_valid_sched = (
        sched_state is not None
        and isinstance(sched_state, dict)
        and len(sched_state) > 0
    )
    if scheduler is not None:
        if has_valid_sched:
            scheduler.load_state_dict(sched_state)
        elif not allow_legacy_resume:
            raise ValueError(
                f"Checkpoint '{checkpoint_path}' lacks 'scheduler_state_dict' required for safe resume! "
                f"Resuming without scheduler state resets the learning rate progression. "
                f"Pass allow_legacy_resume=True to override."
            )
        else:
            is_legacy_resume = True
            print(f"[WARN] Checkpoint '{checkpoint_path}' lacks scheduler state; continuing with fresh scheduler under legacy resume.")

    # 5. Historical Best Validation Loss
    best_val_loss = ckpt.get("best_val_loss")
    if best_val_loss is not None:
        best_val_loss = float(best_val_loss)
    elif not allow_legacy_resume:
        raise ValueError(
            f"Checkpoint '{checkpoint_path}' lacks historical 'best_val_loss' required for safe resume! "
            f"Falling back to current validation loss or infinity can overwrite earlier best models. "
            f"Pass allow_legacy_resume=True to override."
        )
    else:
        is_legacy_resume = True
        if "metrics" in ckpt and "val_loss" in ckpt["metrics"]:
            best_val_loss = float(ckpt["metrics"]["val_loss"])
            print(f"[WARN] Checkpoint lacks 'best_val_loss'; falling back to current checkpoint val_loss ({best_val_loss:.4f}) under legacy resume.")
        else:
            best_val_loss = float("inf")
            print(f"[WARN] Checkpoint lacks 'best_val_loss'; falling back to infinity under legacy resume.")

    # 6. AMP GradScaler State
    scaler_state = ckpt.get("scaler_state_dict")
    requires_scaler = use_amp and (precision.lower() == "fp16")
    if scaler is not None and requires_scaler:
        if scaler_state is not None:
            scaler.load_state_dict(scaler_state)
        elif not allow_legacy_resume:
            raise ValueError(
                f"Checkpoint '{checkpoint_path}' lacks 'scaler_state_dict' required for fp16 AMP resume! "
                f"Pass allow_legacy_resume=True to override."
            )
        else:
            is_legacy_resume = True
            print(f"[WARN] Checkpoint lacks scaler state; continuing with fresh scaler under legacy resume.")

    return {
        "epoch": int(epoch),
        "next_epoch": int(epoch) + 1,
        "best_val_loss": best_val_loss,
        "is_legacy_resume": is_legacy_resume,
        "metrics": ckpt.get("metrics", {}),
        "provenance": ckpt.get("provenance", {}),
        "precision": ckpt.get("precision", precision),
    }
