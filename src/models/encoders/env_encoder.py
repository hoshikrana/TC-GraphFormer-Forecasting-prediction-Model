"""
Environmental Synoptic Modality Encoder for TC-GraphFormer.
Encodes ERA5 atmospheric fields (precipitable water, sst, msl, t2m, u10, v10)
into spatial tokens aligned with the cyclone center, with channel-level validity masking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .ir_encoder import ConvBlock


class EnvEncoder(nn.Module):
    """
    Encodes synoptic environmental fields into spatial tokens.
    Canonical 6 channels:
      0: precipitable_water (kg/m^2)
      1: sea_surface_temperature (K)
      2: mean_sea_level_pressure (Pa or hPa)
      3: 2-metre_temperature (K)
      4: 10-metre_u_wind (m/s)
      5: 10-metre_v_wind (m/s)
    """

    def __init__(self, in_channels: int = 6, d_model: int = 128):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.layer1 = ConvBlock(32, 64, stride=2)
        self.layer2 = ConvBlock(64, d_model, stride=2)

        self.pool = nn.AdaptiveAvgPool2d((4, 4)) # 16 tokens
        self.token_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        channel_mask: torch.Tensor = None,
        pixel_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        x: (B, C, H, W)
        mask: (B,)
        channel_mask: (B, C) optional channel-level validity mask
        pixel_mask: (B, C, H, W) or (B, 1, H, W) pixel validity mask
        Returns: (B, 16, d_model)
        """
        B = x.size(0)
        # Support backwards-compatible single-channel input
        if x.size(1) < self.in_channels:
            pad = torch.zeros(B, self.in_channels - x.size(1), *x.shape[2:], device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)

        # Apply channel-level mask if provided
        if channel_mask is not None and channel_mask.dim() == 2:
            x = x * channel_mask.view(B, -1, 1, 1)

        if pixel_mask is not None:
            # Enforce pixel-level masking: invalid pixels cannot contribute to features
            x = x * pixel_mask
            pm_collapsed = pixel_mask.mean(dim=1, keepdim=True) if pixel_mask.size(1) > 1 else pixel_mask
            patch_validity = (F.adaptive_avg_pool2d(pm_collapsed.float(), (4, 4)) > 0.0).float()
        else:
            patch_validity = torch.ones((B, 1, 4, 4), dtype=torch.float32, device=x.device)

        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)

        pooled = self.pool(out) # (B, d_model, 4, 4)
        pooled = pooled * patch_validity
        tokens = pooled.flatten(2).transpose(1, 2)
        tokens = self.token_proj(tokens)

        mask_expanded = mask.view(B, 1, 1)
        return tokens * mask_expanded


