"""
Infrared (IR) Modality Encoder for TC-GraphFormer.
Processes 2D high-resolution brightness temperature fields with deep residual convolutions
and outputs geometry-aware polar sector tokens with validity masking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..graph.spatial_graph import PolarSectorPooling


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x) + self.shortcut(x))


class IREncoder(nn.Module):
    """
    Encodes Infrared (IRWIN) 2D fields into spatial feature maps and 16 polar sector tokens.
    """

    def __init__(self, in_channels: int = 1, d_model: int = 128):
        super().__init__()
        self.d_model = d_model
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),  # (B, 32, H/2, W/2)
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.layer1 = ConvBlock(32, 64, stride=1)       # (B, 64, H/2, W/2)
        self.layer2 = ConvBlock(64, 128, stride=1)      # (B, 128, H/2, W/2)
        self.layer3 = ConvBlock(128, d_model, stride=1) # (B, d_model, H/2, W/2)

        # Polar Sector Pooling: maps 2D spatial feature map to 16 polar sectors
        self.polar_pool = PolarSectorPooling(in_channels=d_model, d_model=d_model, bias=False)

    def extract_features(
        self,
        x: torch.Tensor,
        pixel_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Extracts 2D spatial feature map from IR field with pixel masking.
        Returns: (B, d_model, H/2, W/2)
        """
        if pixel_mask is not None:
            x = x * pixel_mask
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        if pixel_mask is not None:
            pm_res = F.interpolate(pixel_mask.float(), size=out.shape[-2:], mode="nearest")
            out = out * (pm_res > 0.0).float()
        return out

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        pixel_mask: torch.Tensor = None,
        return_spatial_map: bool = False,
        return_validity: bool = False,
    ):
        """
        x: (B, 1, H, W)
        mask: (B,) binary mask where 1 = modality present, 0 = absent.
        pixel_mask: (B, 1, H, W) binary valid pixel mask where 1 = valid, 0 = invalid.
        return_spatial_map: if True, returns 2D feature map; if False, returns 16 polar tokens.
        return_validity: if True, returns (tokens, token_validity) where token_validity is (B, 16).
        Returns: (B, 16, d_model) or (B, d_model, H/2, W/2) or ((B, 16, d_model), (B, 16))
        """
        feat = self.extract_features(x, pixel_mask=pixel_mask)
        if return_spatial_map:
            return feat
        tokens, token_validity = self.polar_pool(feat, pixel_mask=pixel_mask)
        # Apply validity after token projection and combine with modality presence
        mask_expanded = mask.view(x.size(0), 1, 1)
        token_validity = token_validity * mask_expanded  # (B, 16, 1)
        tokens = tokens * token_validity
        if return_validity:
            return tokens, token_validity.squeeze(-1)  # (B, 16)
        return tokens
