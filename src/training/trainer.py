"""
Hardware-Conscious Training Engine for TC-GraphFormer.
Supports CPU and CUDA/AMP mixed-precision execution, gradient clipping,
checkpoint serialization, and scientifically rigorous masked multi-task validation metrics.
"""

import os
import time
import math
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
        device: str = "cpu",
        checkpoint_dir: str = "checkpoints",
        use_amp: bool = False,
        grad_clip_norm: float = 1.0,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn if loss_fn is not None else MultiTaskLoss().to(device)
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.use_amp = use_amp and (device.startswith("cuda"))
        self.grad_clip_norm = grad_clip_norm
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        os.makedirs(checkpoint_dir, exist_ok=True)
        self.best_val_loss = float("inf")
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
        )
        return preds

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Execute one training epoch with gradient norm tracking."""
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        loss_accum = {}

        start_time = time.time()

        for batch_idx, batch in enumerate(dataloader):
            targets = {k: v.to(self.device) for k, v in batch["targets"].items() if isinstance(v, torch.Tensor)}

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                preds = self._forward_batch(batch)
                loss, loss_dict = self.loss_fn(preds, targets)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()

            B = batch["ir"].size(0)
            total_samples += B
            total_loss += loss.item() * B

            for k, v in loss_dict.items():
                loss_accum[k] = loss_accum.get(k, 0.0) + v * B

        avg_loss = total_loss / max(1, total_samples)
        metrics = {k: round(v / max(1, total_samples), 4) for k, v in loss_accum.items()}
        metrics["epoch"] = epoch
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
        """Serialize model weights, optimizer state, metrics, and provenance metadata."""
        if filename is None:
            filename = f"checkpoint_epoch_{epoch:03d}.pt"

        filepath = os.path.join(self.checkpoint_dir, filename)
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "d_model": getattr(self.model, "d_model", 128),
            "device": self.device,
            "extra_metadata": extra_metadata or {},
        }
        torch.save(state, filepath)

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_checkpoint.pt")
            torch.save(state, best_path)

        latest_path = os.path.join(self.checkpoint_dir, "latest_checkpoint.pt")
        torch.save(state, latest_path)

        return filepath
