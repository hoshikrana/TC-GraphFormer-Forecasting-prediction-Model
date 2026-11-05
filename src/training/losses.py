"""
Multi-Task Loss Module for TC-GraphFormer.
Computes task-specific losses with masking for missing target horizons
and positive weight scaling for rare Rapid Intensification events.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class MultiTaskLoss(nn.Module):
    """
    Combines regression, classification, and masked horizon losses.
    """

    def __init__(
        self,
        weight_intensity: float = 1.0,
        weight_pressure: float = 0.5,
        weight_stage: float = 0.5,
        weight_intensity_change: float = 1.0,
        weight_ri: float = 2.0,
        ri_pos_weight: float = 10.0, # accounts for ~3% RI class imbalance
    ):
        super().__init__()
        self.w_v = weight_intensity
        self.w_p = weight_pressure
        self.w_s = weight_stage
        self.w_dv = weight_intensity_change
        self.w_ri = weight_ri

        self.loss_huber = nn.SmoothL1Loss(reduction="mean", beta=1.0)
        self.loss_ce = nn.CrossEntropyLoss(reduction="mean")
        self.register_buffer("ri_weight", torch.tensor([ri_pos_weight], dtype=torch.float32))

    def forward(self, preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        preds: output of TCGraphFormer
        targets: target dict from batch
        Returns:
            total_loss: torch.Tensor
            loss_components: Dict[str, float]
        """
        # 1. Current Intensity Loss (Huber)
        loss_v = self.loss_huber(preds["intensity"], targets["intensity"])

        # 2. Central Pressure Loss (Huber, scaled down by 10 to balance gradient magnitudes)
        loss_p = self.loss_huber(preds["pressure"] / 10.0, targets["pressure"] / 10.0)

        # 3. Development Stage Classification Loss (Cross Entropy)
        loss_s = self.loss_ce(preds["stage_logits"], targets["stage"])

        # 4. Multi-Horizon Intensity Change Loss (Masked Smooth L1)
        dv_pred = preds["intensity_change"] # (B, 4)
        dv_true = targets["intensity_change"] # (B, 4)
        dv_mask = targets["intensity_change_mask"] # (B, 4)

        diff = F.smooth_l1_loss(dv_pred, dv_true, reduction="none", beta=1.0) # (B, 4)
        if dv_mask.sum() > 0:
            loss_dv = (diff * dv_mask).sum() / dv_mask.sum().clamp(min=1.0)
        else:
            loss_dv = torch.tensor(0.0, device=diff.device)

        # 5. Rapid Intensification Loss (Masked BCE with Logits)
        ri_pred = preds["ri_logits"] # (B,)
        ri_true = targets["rapid_intensification"] # (B,)
        ri_mask = targets["ri_mask"] # (B,)

        ri_bce = F.binary_cross_entropy_with_logits(
            ri_pred, ri_true, pos_weight=self.ri_weight, reduction="none"
        )
        if ri_mask.sum() > 0:
            loss_ri = (ri_bce * ri_mask).sum() / ri_mask.sum().clamp(min=1.0)
        else:
            loss_ri = torch.tensor(0.0, device=ri_bce.device)

        # Total Weighted Multi-Task Loss
        total_loss = (
            self.w_v * loss_v +
            self.w_p * loss_p +
            self.w_s * loss_s +
            self.w_dv * loss_dv +
            self.w_ri * loss_ri
        )

        loss_dict = {
            "loss_total": float(total_loss.detach().item()),
            "loss_intensity": float(loss_v.detach().item()),
            "loss_pressure": float(loss_p.detach().item()),
            "loss_stage": float(loss_s.detach().item()),
            "loss_intensity_change": float(loss_dv.detach().item()),
            "loss_ri": float(loss_ri.detach().item()),
        }
        return total_loss, loss_dict

