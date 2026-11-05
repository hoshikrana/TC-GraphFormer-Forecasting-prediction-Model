"""
Hardware-Conscious Training Engine for TC-GraphFormer.
Supports CPU and CUDA/AMP mixed-precision execution, gradient clipping,
checkpoint serialization, and multi-task validation metrics.
"""

import os
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional

from .losses import MultiTaskLoss


class TCGraphFormerTrainer:
    """
    Standard Trainer for TC-GraphFormer model.
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

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Execute one training epoch."""
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        loss_accum = {}

        start_time = time.time()

        for batch_idx, batch in enumerate(dataloader):
            ir = batch["ir"].to(self.device)
            pmw = batch["pmw"].to(self.device)
            gprof = batch["gprof"].to(self.device)
            radar = batch["radar"].to(self.device)
            env = batch["env"].to(self.device)
            masks = batch["modality_masks"].to(self.device)

            targets = {k: v.to(self.device) for k, v in batch["targets"].items()}

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                preds = self.model(ir, pmw, gprof, radar, env, masks)
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

            B = ir.size(0)
            total_samples += B
            total_loss += loss.item() * B

            for k, v in loss_dict.items():
                loss_accum[k] = loss_accum.get(k, 0.0) + v * B

        avg_loss = total_loss / max(1, total_samples)
        metrics = {k: v / max(1, total_samples) for k, v in loss_accum.items()}
        metrics["epoch"] = epoch
        metrics["epoch_time_sec"] = round(time.time() - start_time, 2)
        return metrics

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model on validation or test partition."""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        v_mae_sum = 0.0
        p_mae_sum = 0.0
        stage_correct = 0
        dv_mae_sum = 0.0
        dv_count = 0
        ri_correct = 0
        ri_count = 0

        with torch.no_grad():
            for batch in dataloader:
                ir = batch["ir"].to(self.device)
                pmw = batch["pmw"].to(self.device)
                gprof = batch["gprof"].to(self.device)
                radar = batch["radar"].to(self.device)
                env = batch["env"].to(self.device)
                masks = batch["modality_masks"].to(self.device)

                targets = {k: v.to(self.device) for k, v in batch["targets"].items()}

                preds = self.model(ir, pmw, gprof, radar, env, masks)
                loss, _ = self.loss_fn(preds, targets)

                B = ir.size(0)
                total_samples += B
                total_loss += loss.item() * B

                # Intensity MAE (kt)
                v_mae_sum += torch.abs(preds["intensity"] - targets["intensity"]).sum().item()

                # Pressure MAE (hPa)
                p_mae_sum += torch.abs(preds["pressure"] - targets["pressure"]).sum().item()

                # Stage Accuracy
                stage_preds = preds["stage_logits"].argmax(dim=-1)
                stage_correct += (stage_preds == targets["stage"]).sum().item()

                # Intensity Change MAE (kt)
                diff_dv = torch.abs(preds["intensity_change"] - targets["intensity_change"])
                mask_dv = targets["intensity_change_mask"]
                dv_mae_sum += (diff_dv * mask_dv).sum().item()
                dv_count += mask_dv.sum().item()

                # RI Accuracy
                ri_probs = torch.sigmoid(preds["ri_logits"])
                ri_pred_bin = (ri_probs > 0.5).float()
                ri_mask = targets["ri_mask"]
                ri_correct += ((ri_pred_bin == targets["rapid_intensification"]) * ri_mask).sum().item()
                ri_count += ri_mask.sum().item()

        metrics = {
            "val_loss": round(total_loss / max(1, total_samples), 4),
            "val_wind_mae_kt": round(v_mae_sum / max(1, total_samples), 2),
            "val_pressure_mae_hpa": round(p_mae_sum / max(1, total_samples), 2),
            "val_stage_acc_pct": round((stage_correct / max(1, total_samples)) * 100, 2),
            "val_dv_mae_kt": round(dv_mae_sum / max(1, dv_count), 2) if dv_count > 0 else 0.0,
            "val_ri_acc_pct": round((ri_correct / max(1, ri_count)) * 100, 2) if ri_count > 0 else 0.0,
            "total_val_samples": total_samples,
        }
        return metrics

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False, filename: str = None) -> str:
        """Serialize model weights, optimizer state, and training metadata."""
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
        }
        torch.save(state, filepath)

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_checkpoint.pt")
            torch.save(state, best_path)

        latest_path = os.path.join(self.checkpoint_dir, "latest_checkpoint.pt")
        torch.save(state, latest_path)

        return filepath

