"""
Multi-Task Loss Module for TC-GraphFormer.
Computes strictly mask-aware multi-task losses with valid-target denominators:
- Current Wind Speed (kt)
- Central Minimum Pressure (hPa)
- Development Stage Classification (Canonical 5 classes)
- Multi-Horizon Intensity Changes (+6h, +12h, +18h, +24h kt)
- Rapid Intensification (24h binary flag with TRAIN-fitted pos_weight)
- Track Displacements (Delta x, Delta y in km across +6h, +12h, +18h, +24h)
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskLoss(nn.Module):
    """
    Combines regression, classification, horizon changes, and geodesic track losses
    with independent validity masks.
    """

    def __init__(
        self,
        weight_intensity: float = 1.0,
        weight_pressure: float = 0.5,
        weight_stage: float = 0.5,
        weight_intensity_change: float = 1.0,
        weight_ri: float = 2.0,
        weight_track: float = 1.0,
        ri_pos_weight: float = 33.53, # Fitted strictly on TRAIN partition (1207 neg / 36 pos)
    ):
        super().__init__()
        self.w_v = weight_intensity
        self.w_p = weight_pressure
        self.w_s = weight_stage
        self.w_dv = weight_intensity_change
        self.w_ri = weight_ri
        self.w_track = weight_track

        self.register_buffer("ri_weight", torch.tensor([ri_pos_weight], dtype=torch.float32))

    def forward(
        self, preds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        preds: output of TCGraphFormer
        targets: target dict from batch
        Returns:
            total_loss: torch.Tensor
            loss_components: Dict[str, float]
        """
        # 1. Current Intensity Loss (Huber, masked)
        v_pred = preds["intensity"]
        v_true = targets["intensity"]
        v_mask = targets.get("intensity_mask", torch.ones_like(v_true))
        diff_v = F.smooth_l1_loss(v_pred, v_true, reduction="none", beta=1.0)
        n_v = v_mask.sum()
        loss_v = (diff_v * v_mask).sum() / n_v.clamp(min=1.0)

        # 2. Central Pressure Loss (Huber, scaled down by 10 to balance gradient magnitudes, masked)
        p_pred = preds["pressure"] / 10.0
        p_true = targets["pressure"] / 10.0
        p_mask = targets.get("pressure_mask", torch.ones_like(p_true))
        diff_p = F.smooth_l1_loss(p_pred, p_true, reduction="none", beta=1.0)
        n_p = p_mask.sum()
        loss_p = (diff_p * p_mask).sum() / n_p.clamp(min=1.0)

        # 3. Development Stage Classification Loss (Cross Entropy, masked for valid stages)
        stage_logits = preds["stage_logits"] # (B, 5) or (B, 6)
        s_true = targets["stage"]
        s_mask = targets.get("stage_mask", torch.ones_like(s_true, dtype=torch.float32))
        ce_s = F.cross_entropy(stage_logits, s_true, reduction="none")
        n_s = s_mask.sum()
        loss_s = (ce_s * s_mask).sum() / n_s.clamp(min=1.0)

        # 4. Multi-Horizon Intensity Change Loss (Masked Smooth L1)
        dv_pred = preds["intensity_change"] # (B, 4)
        dv_true = targets["intensity_change"] # (B, 4)
        dv_mask = targets["intensity_change_mask"] # (B, 4)
        diff_dv = F.smooth_l1_loss(dv_pred, dv_true, reduction="none", beta=1.0)
        n_dv = dv_mask.sum()
        loss_dv = (diff_dv * dv_mask).sum() / n_dv.clamp(min=1.0)

        # 5. Rapid Intensification Loss (Masked BCE with Logits)
        ri_pred = preds["ri_logits"] # (B,)
        ri_true = targets["rapid_intensification"] # (B,)
        ri_mask = targets["ri_mask"] # (B,)
        ri_bce = F.binary_cross_entropy_with_logits(
            ri_pred, ri_true, pos_weight=self.ri_weight, reduction="none"
        )
        n_ri = ri_mask.sum()
        loss_ri = (ri_bce * ri_mask).sum() / n_ri.clamp(min=1.0)

        # 6. Future Track Displacement Loss (Masked Smooth L1 on [dx, dy] in km scaled by 100km)
        if "track_delta" in preds and "track_displacement" in targets:
            track_pred = preds["track_delta"].view(-1, 4, 2) / 100.0 # (B, 4, 2)
            track_true = targets["track_displacement"] / 100.0       # (B, 4, 2)
            track_mask = targets.get("track_mask", torch.ones_like(track_pred[:, :, 0])).unsqueeze(-1) # (B, 4, 1)
            diff_tr = F.smooth_l1_loss(track_pred, track_true, reduction="none", beta=1.0)
            n_tr = track_mask.sum() * 2.0
            loss_track = (diff_tr * track_mask).sum() / n_tr.clamp(min=1.0)
            valid_tr_count = float(track_mask.sum().item())
        else:
            loss_track = torch.tensor(0.0, device=preds["intensity"].device)
            valid_tr_count = 0.0

        # Total Weighted Multi-Task Loss
        total_loss = (
            self.w_v * loss_v +
            self.w_p * loss_p +
            self.w_s * loss_s +
            self.w_dv * loss_dv +
            self.w_ri * loss_ri +
            self.w_track * loss_track
        )

        loss_dict = {
            "loss_total": float(total_loss.detach().item()),
            "loss_intensity": float(loss_v.detach().item()),
            "loss_pressure": float(loss_p.detach().item()),
            "loss_stage": float(loss_s.detach().item()),
            "loss_intensity_change": float(loss_dv.detach().item()),
            "loss_ri": float(loss_ri.detach().item()),
            "loss_track": float(loss_track.detach().item()),
            "valid_v_count": float(n_v.item()),
            "valid_p_count": float(n_p.item()),
            "valid_s_count": float(n_s.item()),
            "valid_dv_count": float(n_dv.item()),
            "valid_ri_count": float(n_ri.item()),
            "valid_tr_count": valid_tr_count,
        }
        return total_loss, loss_dict
