"""
Precipitation Radar Modality Encoder for TC-GraphFormer.
Processes 3D volumetric reflectivity profiles (Ku/Ka bands across vertical bins)
with 3D convolutions and spatial pooling. Handles sparse availability cleanly.
"""

import torch
import torch.nn as nn


class RadarEncoder(nn.Module):
    """
    Encodes 3D radar reflectivity volume (C, D, H, W) into spatial tokens.
    """

    def __init__(self, in_channels: int = 1, d_model: int = 128):
        super().__init__()
        self.d_model = d_model
        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.GELU(),
        )
        self.conv3d_2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool3d((1, 4, 4)) # collapse vertical depth to 1, spatial 4x4
        self.proj = nn.Linear(32, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, D, H, W)
        mask: (B,)
        Returns: (B, 16, d_model)
        """
        B = x.size(0)
        out = self.conv3d_1(x)
        out = self.conv3d_2(out)
        out = self.pool(out) # (B, 32, 1, 4, 4)

        tokens = out.squeeze(2).flatten(2).transpose(1, 2) # (B, 16, 32)
        tokens = self.proj(tokens) # (B, 16, d_model)

        mask_expanded = mask.view(B, 1, 1)
        return tokens * mask_expanded

