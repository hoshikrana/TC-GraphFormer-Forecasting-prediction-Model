"""
Infrared (IR) Modality Encoder for TC-GraphFormer.
Processes 2D high-resolution brightness temperature fields with deep residual convolutions
and outputs geometry-aware feature tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    Encodes Infrared (IRWIN) 2D fields into spatial tokens of dimension d_model.
    """

    def __init__(self, in_channels: int = 1, d_model: int = 128):
        super().__init__()
        self.d_model = d_model
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2, bias=False), # (B, 32, H/2, W/2)
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.layer1 = ConvBlock(32, 64, stride=2)   # (B, 64, H/4, W/4)
        self.layer2 = ConvBlock(64, 128, stride=2)  # (B, 128, H/8, W/8)
        self.layer3 = ConvBlock(128, d_model, stride=2) # (B, d_model, H/16, W/16)

        # Global context and token projection
        self.pool = nn.AdaptiveAvgPool2d((4, 4)) # produces 16 spatial tokens
        self.token_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        pixel_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        x: (B, 1, H, W)
        mask: (B,) binary mask where 1 = modality present, 0 = absent.
        pixel_mask: (B, 1, H, W) binary valid pixel mask where 1 = valid, 0 = invalid.
        Returns: (B, num_tokens, d_model)
        """
        B = x.size(0)
        if pixel_mask is not None:
            # Enforce pixel-level masking: invalid pixels cannot contribute to features
            x = x * pixel_mask
            patch_validity = (F.adaptive_avg_pool2d(pixel_mask.float(), (4, 4)) > 0.0).float()
        else:
            patch_validity = torch.ones((B, 1, 4, 4), dtype=torch.float32, device=x.device)

        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        pooled = self.pool(out) # (B, d_model, 4, 4)
        pooled = pooled * patch_validity

        tokens = pooled.flatten(2).transpose(1, 2) # (B, 16, d_model)
        tokens = self.token_proj(tokens)

        # Zero out tokens if modality is masked out
        mask_expanded = mask.view(B, 1, 1)
        tokens = tokens * mask_expanded
        return tokens

