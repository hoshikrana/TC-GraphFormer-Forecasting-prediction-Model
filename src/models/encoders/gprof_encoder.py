"""
GPROF Precipitation Modality Encoder for TC-GraphFormer.
Encodes surface precipitation rates and convective hydrometeor structures into spatial tokens.
"""

import torch
import torch.nn as nn
from .ir_encoder import ConvBlock


class GPROFEncoder(nn.Module):
    """
    Encodes GPROF surface precipitation fields into spatial tokens.
    """

    def __init__(self, in_channels: int = 1, d_model: int = 128):
        super().__init__()
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

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, H, W)
        mask: (B,)
        Returns: (B, 16, d_model)
        """
        B = x.size(0)
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)

        pooled = self.pool(out) # (B, d_model, 4, 4)
        tokens = pooled.flatten(2).transpose(1, 2)
        tokens = self.token_proj(tokens)

        mask_expanded = mask.view(B, 1, 1)
        return tokens * mask_expanded

