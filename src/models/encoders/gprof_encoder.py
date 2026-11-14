"""
GPROF Precipitation Modality Encoder for TC-GraphFormer.
Encodes surface precipitation rates, convective components, and hydrometeor water paths
into spatial feature maps and 16 polar sector tokens with channel-level validity masking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .ir_encoder import ConvBlock
from ..graph.spatial_graph import PolarSectorPooling


class GPROFEncoder(nn.Module):
    """
    Encodes GPROF precipitation and hydrometeor fields into spatial feature maps and 16 polar sector tokens.
    Canonical 4 channels:
      0: surfacePrecipitation (mm/hr)
      1: convectivePrecipitation (mm/hr)
      2: rainWaterPath (kg/m^2)
      3: iceWaterPath (kg/m^2)
    """

    def __init__(self, in_channels: int = 4, d_model: int = 128):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),  # (B, 32, H/2, W/2)
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.layer1 = ConvBlock(32, 64, stride=1)       # (B, 64, H/2, W/2)
        self.layer2 = ConvBlock(64, d_model, stride=1)  # (B, d_model, H/2, W/2)

        # Polar Sector Pooling: maps 2D spatial feature map to 16 polar sectors
        self.polar_pool = PolarSectorPooling(in_channels=d_model, d_model=d_model, bias=False)

    def extract_features(
        self,
        x: torch.Tensor,
        channel_mask: torch.Tensor = None,
        pixel_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Extracts 2D spatial feature map with channel and pixel masking.
        Returns: (B, d_model, H/2, W/2)
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
            x = x * pixel_mask

        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)

        if pixel_mask is not None:
            pm = pixel_mask.mean(dim=1, keepdim=True) if pixel_mask.size(1) > 1 else pixel_mask
            pm_res = F.interpolate(pm.float(), size=out.shape[-2:], mode="nearest")
            out = out * (pm_res > 0.0).float()
        return out

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        channel_mask: torch.Tensor = None,
        pixel_mask: torch.Tensor = None,
        return_spatial_map: bool = False,
        return_validity: bool = False,
    ):
        """
        x: (B, C, H, W)
        mask: (B,)
        channel_mask: (B, C) optional channel-level validity mask
        pixel_mask: (B, C, H, W) or (B, 1, H, W) pixel validity mask
        return_spatial_map: if True, returns 2D feature map; if False, returns 16 polar tokens.
        return_validity: if True, returns (tokens, token_validity) where token_validity is (B, 16).
        Returns: (B, 16, d_model) or (B, d_model, H/2, W/2) or ((B, 16, d_model), (B, 16))
        """
        feat = self.extract_features(x, channel_mask=channel_mask, pixel_mask=pixel_mask)
        if return_spatial_map:
            return feat
        tokens, token_validity = self.polar_pool(feat, pixel_mask=pixel_mask)
        mask_expanded = mask.view(x.size(0), 1, 1)
        token_validity = token_validity * mask_expanded
        tokens = tokens * token_validity
        if return_validity:
            return tokens, token_validity.squeeze(-1)
        return tokens
