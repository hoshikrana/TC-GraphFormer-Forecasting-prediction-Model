"""
Precipitation Radar Modality Encoder for TC-GraphFormer.
Processes 3D volumetric reflectivity profiles (Ku/Ka bands across vertical bins)
with 3D convolutions and polar spatial pooling. Handles sparse availability cleanly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..graph.spatial_graph import PolarSectorPooling


class RadarEncoder(nn.Module):
    """
    Encodes 3D radar reflectivity volume (C, D, H, W) into spatial feature maps and 16 polar sector tokens.
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
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.GELU(),
        )
        self.proj_2d = nn.Sequential(
            nn.Conv2d(32, d_model, kernel_size=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.GELU(),
        )

        # Polar Sector Pooling: maps 2D spatial feature map to 16 polar sectors
        self.polar_pool = PolarSectorPooling(in_channels=d_model, d_model=d_model, bias=False)

    def extract_features(
        self,
        x: torch.Tensor,
        voxel_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Extracts 2D spatial feature map from 3D radar volume with voxel masking.
        Returns: (B, d_model, H, W)
        """
        if voxel_mask is not None:
            x = x * voxel_mask

        out = self.conv3d_1(x)
        out = self.conv3d_2(out)

        # Collapse vertical depth dimension to 2D composite
        out_2d = out.mean(dim=2)  # (B, 32, H, W)
        feat_2d = self.proj_2d(out_2d)  # (B, d_model, H, W)

        if voxel_mask is not None:
            if voxel_mask.dim() == 5:
                vm_2d = (voxel_mask.mean(dim=(1, 2)) > 0.0).float().unsqueeze(1)
            elif voxel_mask.dim() == 4:
                vm_2d = voxel_mask
            else:
                vm_2d = voxel_mask.unsqueeze(1)
            vm_res = F.interpolate(vm_2d, size=feat_2d.shape[-2:], mode="nearest")
            feat_2d = feat_2d * (vm_res > 0.0).float()
        return feat_2d

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        voxel_mask: torch.Tensor = None,
        return_spatial_map: bool = False,
    ) -> torch.Tensor:
        """
        x: (B, C, D, H, W)
        mask: (B,)
        voxel_mask: (B, C, D, H, W) or (B, 1, D, H, W)
        return_spatial_map: if True, returns 2D feature map; if False, returns 16 polar tokens.
        Returns: (B, 16, d_model) or (B, d_model, H, W)
        """
        feat = self.extract_features(x, voxel_mask=voxel_mask)
        if return_spatial_map:
            return feat

        # 2D pixel mask collapsed across depth for polar pooling
        pix_mask_2d = None
        if voxel_mask is not None:
            if voxel_mask.dim() == 5:
                pix_mask_2d = (voxel_mask.mean(dim=(1, 2)) > 0.0).float().unsqueeze(1)
            else:
                pix_mask_2d = voxel_mask

        tokens, _ = self.polar_pool(feat, pixel_mask=pix_mask_2d)
        mask_expanded = mask.view(x.size(0), 1, 1)
        return tokens * mask_expanded
