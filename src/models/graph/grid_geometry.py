"""
Unified Physical Grid and Storm-Relative Polar Geometry for TC-GraphFormer.

Provides mathematically exact and physically consistent coordinate systems:
1. Canonical Storm Domain: [-500.0, +500.0] km in zonal (X, East+) and meridional (Y, North+) axes.
2. Input Grid (64x64): Cell centers spaced at exactly 15.625 km spanning [-492.1875, +492.1875] km.
3. Feature Grid (32x32): Exact receptive-field centers of stride-2, kernel-3, padding-1 convolutions
   spanning [-492.1875, +476.5625] km (offset from naive 32-cell grids by 7.8125 km).
4. Canonical Polar Partition:
   - 4 Concentric Radial Rings: [0, 75, 175, 300, 500] km
     Nominal ring centers: [37.5, 125.0, 237.5, 400.0] km
   - 4 Azimuthal Quadrants (Mathematical Polar Angle):
     Sector 0: NE [0, pi/2)     (x >= 0, y >= 0)
     Sector 1: NW [pi/2, pi)    (x < 0,  y >= 0)
     Sector 2: SW [pi, 3pi/2)   (x <= 0, y < 0)
     Sector 3: SE [3pi/2, 2pi)  (x > 0,  y < 0)
   - Node ordering: p = ring_idx * 4 + sector_idx  (16 nodes: 0 to 15)
"""

import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch


# Canonical Polar Definitions
CANONICAL_DOMAIN_RADIUS_KM: float = 500.0
CANONICAL_RING_BOUNDS_KM: Tuple[float, ...] = (0.0, 75.0, 175.0, 300.0, 500.0)
CANONICAL_RING_CENTERS_KM: Tuple[float, ...] = (37.5, 125.0, 237.5, 400.0)
CANONICAL_SECTOR_NAMES: Tuple[str, ...] = ("NE", "NW", "SW", "SE")
CANONICAL_SECTOR_ANGLES_DEG: Tuple[float, ...] = (45.0, 135.0, 225.0, 315.0)

# Exact geometric areas of each quadrant sector by ring (km^2)
# Area = (pi / 4) * (r_outer^2 - r_inner^2)
CANONICAL_SECTOR_AREAS_KM2: Tuple[float, ...] = tuple(
    (math.pi / 4.0) * (CANONICAL_RING_BOUNDS_KM[i + 1] ** 2 - CANONICAL_RING_BOUNDS_KM[i] ** 2)
    for i in range(4)
)


class GridGeometry:
    """
    Centralized physical coordinate and polar geometry provider.
    Ensures identical spatial grounding across data loading, radar resampling,
    encoder convolutions, polar sector pooling, MFM masking, and graph message passing.
    """

    def __init__(self, domain_radius_km: float = CANONICAL_DOMAIN_RADIUS_KM):
        self.domain_radius_km = float(domain_radius_km)
        self.ring_bounds = CANONICAL_RING_BOUNDS_KM
        self.ring_centers = CANONICAL_RING_CENTERS_KM
        self.sector_areas_km2 = CANONICAL_SECTOR_AREAS_KM2
        self._cached_torch_masks: Dict[Tuple[int, int, bool, str], torch.Tensor] = {}

    # -------------------------------------------------------------------------
    # 1. 1D and 2D Coordinate Generators
    # -------------------------------------------------------------------------
    def get_input_coords_64(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Input grid (64x64) cell centers.
        Cell spacing: 1000 km / 64 = 15.625 km.
        x: West to East [-492.1875, +492.1875] km
        y: North to South [+492.1875, -492.1875] km (row 0 is North)
        """
        dx = (2.0 * self.domain_radius_km) / 64.0  # 15.625 km
        x = -self.domain_radius_km + (np.arange(64, dtype=np.float32) + 0.5) * dx
        y = self.domain_radius_km - (np.arange(64, dtype=np.float32) + 0.5) * dx
        return x, y

    def get_feature_coords_32(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Feature grid (32x32) receptive-field centers after stride-2, kernel-3, pad-1 conv.
        Output index (r, c) corresponds to input cell (2r, 2c):
          x(c) = -500.0 + (2c + 0.5) * 15.625 = -492.1875 + c * 31.25 km
          y(r) = +500.0 - (2r + 0.5) * 15.625 = +492.1875 - r * 31.25 km
        """
        dx_in = (2.0 * self.domain_radius_km) / 64.0  # 15.625 km
        c = np.arange(32, dtype=np.float32)
        r = np.arange(32, dtype=np.float32)
        x = -self.domain_radius_km + (2.0 * c + 0.5) * dx_in
        y = self.domain_radius_km - (2.0 * r + 0.5) * dx_in
        return x, y

    def get_feature_mesh_32(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        2D meshgrid (X, Y) of 32x32 convolution feature receptive-field centers in km.
        Returns:
            X: (32, 32)
            Y: (32, 32)
        """
        x, y = self.get_feature_coords_32()
        return np.meshgrid(x, y)

    def get_coords(self, H: int, W: int, is_feature_grid: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        General coordinate generator.
        If H=32, W=32 and is_feature_grid is True, returns convolution feature centers.
        If H=64, W=64, returns input cell centers.
        Otherwise, returns standard cell centers over [-domain, +domain].
        """
        if H == 32 and W == 32 and is_feature_grid:
            return self.get_feature_coords_32()
        if H == 64 and W == 64:
            return self.get_input_coords_64()

        dx = (2.0 * self.domain_radius_km) / float(W)
        dy = (2.0 * self.domain_radius_km) / float(H)
        x = -self.domain_radius_km + (np.arange(W, dtype=np.float32) + 0.5) * dx
        y = self.domain_radius_km - (np.arange(H, dtype=np.float32) + 0.5) * dy
        return x, y

    # -------------------------------------------------------------------------
    # 2. Polar Sector Membership Masks
    # -------------------------------------------------------------------------
    def get_sector_masks_torch(
        self, H: int, W: int, device: torch.device, is_feature_grid: bool = True
    ) -> torch.Tensor:
        """
        Returns binary sector masks of shape (16, H, W) on device.
        Each mask corresponds to node p = ring_idx * 4 + sector_idx.
        """
        key = (H, W, is_feature_grid, str(device))
        if key in self._cached_torch_masks:
            return self._cached_torch_masks[key]

        x_np, y_np = self.get_coords(H, W, is_feature_grid=is_feature_grid)
        x_t = torch.tensor(x_np, dtype=torch.float32, device=device)
        y_t = torch.tensor(y_np, dtype=torch.float32, device=device)
        yy, xx = torch.meshgrid(y_t, x_t, indexing="ij")

        r = torch.sqrt(xx**2 + yy**2)
        theta = (torch.atan2(yy, xx) + 2.0 * math.pi) % (2.0 * math.pi)

        # 4 azimuthal sectors:
        # Sector 0: NE [0, pi/2), Sector 1: NW [pi/2, pi), Sector 2: SW [pi, 3pi/2), Sector 3: SE [3pi/2, 2pi)
        sector_bounds = [0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi, 2.0 * math.pi]

        masks = torch.zeros((16, H, W), dtype=torch.float32, device=device)
        for i in range(4):
            r_m = (r >= self.ring_bounds[i]) & (r < self.ring_bounds[i + 1])
            for j in range(4):
                th_m = (theta >= sector_bounds[j]) & (theta < sector_bounds[j + 1])
                masks[i * 4 + j] = (r_m & th_m).float()

        # Handle edge cases where small grid might have empty sector: assign nearest cell
        for p in range(16):
            if masks[p].sum() == 0.0:
                ri = self.ring_centers[p // 4]
                thi = math.radians(CANONICAL_SECTOR_ANGLES_DEG[p % 4])
                xi, yi = ri * math.cos(thi), ri * math.sin(thi)
                dist_grid = (xx - xi) ** 2 + (yy - yi) ** 2
                min_idx = torch.argmin(dist_grid)
                masks[p].view(-1)[min_idx] = 1.0

        self._cached_torch_masks[key] = masks
        return masks

    # -------------------------------------------------------------------------
    # 3. Spatial Impulse & Node Mapping Helpers
    # -------------------------------------------------------------------------
    def point_to_polar_node(self, x_km: float, y_km: float) -> Tuple[int, int, int]:
        """
        Maps a physical coordinate (x_km, y_km) to:
        (ring_idx, sector_idx, node_idx) where node_idx = ring_idx * 4 + sector_idx.
        Returns (-1, -1, -1) if outside domain (r >= 500 km).
        """
        r = math.sqrt(x_km**2 + y_km**2)
        if r >= self.domain_radius_km:
            return -1, -1, -1

        # Ring index
        ring_idx = -1
        for i in range(4):
            if self.ring_bounds[i] <= r < self.ring_bounds[i + 1]:
                ring_idx = i
                break
        if ring_idx == -1:
            return -1, -1, -1

        # Azimuthal sector index
        theta = (math.atan2(y_km, x_km) + 2.0 * math.pi) % (2.0 * math.pi)
        sector_bounds = [0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi, 2.0 * math.pi]
        sector_idx = -1
        for j in range(4):
            if sector_bounds[j] <= theta < sector_bounds[j + 1]:
                sector_idx = j
                break
        if sector_idx == -1:
            sector_idx = 0  # 2*pi boundary edge case -> Sector 0 (NE)

        node_idx = ring_idx * 4 + sector_idx
        return ring_idx, sector_idx, node_idx

    def get_sector_area_km2(self, node_idx: int) -> float:
        """
        Returns exact geometric area (km^2) of the polar sector corresponding to node_idx.
        """
        ring_idx = node_idx // 4
        return self.sector_areas_km2[ring_idx]


# Singleton default geometry
GLOBAL_GRID_GEOMETRY = GridGeometry()
