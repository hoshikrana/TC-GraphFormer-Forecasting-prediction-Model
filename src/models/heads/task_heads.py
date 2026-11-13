"""
Multi-Task Prediction Heads for TC-GraphFormer.
Decodes shared latent spatiotemporal representation into:
- Current Wind Speed (kt)
- Central Minimum Pressure (hPa)
- Development Stage (6 classes)
- Intensity Changes (+6h, +12h, +18h, +24h)
- Rapid Intensification (24h binary flag)
- Track Displacements (Delta lat, Delta lon across horizons)
"""

import torch
import torch.nn as nn
from typing import Dict


class TaskPredictionHeads(nn.Module):
    """
    Multi-task prediction heads operating on shared latent representation.
    """

    def __init__(self, d_model: int = 128, num_stages: int = 5):
        super().__init__()
        self.d_model = d_model
        self.num_stages = num_stages

        # 1. Current Intensity Head (Wind Speed in knots)
        self.head_intensity = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        # 2. Central Minimum Pressure Head (hPa)
        self.head_pressure = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        # 3. Development Stage Classification Head (6 classes)
        self.head_stage = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, num_stages),
        )

        # 4. Multi-Horizon Intensity Change Head (+6h, +12h, +18h, +24h kt)
        self.head_intensity_change = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 4),
        )

        # 5. Rapid Intensification Binary Classification Head (>= 30 kt / 24h)
        self.head_ri = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        # 6. Track Displacement Head (Delta lat, Delta lon for 4 horizons = 8 values)
        self.head_track = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 8),
        )

    def forward(self, latent: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        latent: (B, d_model)
        Returns dictionary of predictions:
            intensity: (B,)
            pressure: (B,)
            stage_logits: (B, 6)
            intensity_change: (B, 4)
            ri_logits: (B,)
            track_delta: (B, 8)
        """
        v_pred = self.head_intensity(latent).squeeze(-1) # (B,)
        p_pred = self.head_pressure(latent).squeeze(-1)   # (B,)
        stage_logits = self.head_stage(latent)            # (B, 6)
        dv_pred = self.head_intensity_change(latent)      # (B, 4)
        ri_logit = self.head_ri(latent).squeeze(-1)       # (B,)
        track_delta = self.head_track(latent)             # (B, 8)

        return {
            "intensity": v_pred,
            "pressure": p_pred,
            "stage_logits": stage_logits,
            "intensity_change": dv_pred,
            "ri_logits": ri_logit,
            "track_delta": track_delta,
        }

