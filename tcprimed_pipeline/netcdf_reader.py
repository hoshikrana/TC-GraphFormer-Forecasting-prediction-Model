"""
Safe NetCDF4 reader module for TC PRIMED.
Handles hierarchical groups, attributes, and masked arrays without double-decoding scale/offset.
"""

import os
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Generator
import numpy as np
import netCDF4 as nc


def clean_scalar(v: Any) -> Any:
    """
    Extract a clean Python primitive (int, float, str, None) from a scalar or 0-d/1-element array.
    Guarantees no numpy/MaskedConstant leaks.
    """
    if v is None:
        return None
    if isinstance(v, (np.ma.core.MaskedConstant, np.ma.MaskedArray)):
        if np.ma.is_masked(v):
            return None
        v = v.item() if hasattr(v, "item") else v
    elif hasattr(v, "item"):
        try:
            v = v.item()
        except (ValueError, TypeError):
            pass
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    return v


def get_array_with_fill(var: nc.Variable, fill_value: float = np.nan) -> Tuple[np.ndarray, np.ndarray]:
    """
    Safely extract a numpy array and its boolean valid mask from a NetCDF variable.
    NetCDF-4 in Python automatically decodes scale_factor and add_offset upon slicing.
    This function guarantees:
      1. Scale/offset is NOT applied a second time.
      2. Masked elements are populated with `fill_value`.
      3. A boolean mask of valid (unmasked) elements is returned.
    
    Returns:
        (data_array, valid_mask):
            data_array: numpy float32/float64 array with masked positions set to fill_value.
            valid_mask: boolean numpy array (True where valid, False where masked/fill).
    """
    # By default, NetCDF4 auto_maskandscale is True
    raw_data = var[:]

    if isinstance(raw_data, np.ma.MaskedArray):
        valid_mask = ~raw_data.mask
        # If all masked
        if not np.any(valid_mask):
            out_arr = np.full(raw_data.shape, fill_value, dtype=np.float32)
            return out_arr, valid_mask
        
        # Fill masked elements with fill_value
        out_arr = raw_data.filled(fill_value)
    else:
        # Check for NaN / Inf
        if np.issubdtype(raw_data.dtype, np.floating):
            valid_mask = ~np.isnan(raw_data) & ~np.isinf(raw_data)
        else:
            valid_mask = np.ones(raw_data.shape, dtype=bool)
        out_arr = np.array(raw_data, copy=True)

    return out_arr, valid_mask


@contextmanager
def open_netcdf(filepath: str, mode: str = "r") -> Generator[nc.Dataset, None, None]:
    """
    Context manager for safely opening and closing a NetCDF dataset.
    """
    if not os.path.exists(filepath) and mode == "r":
        raise FileNotFoundError(f"NetCDF file not found: {filepath}")
    
    ds = nc.Dataset(filepath, mode)
    try:
        yield ds
    finally:
        ds.close()


class NetCDFReader:
    """Helper class for querying hierarchical NetCDF-4 structures."""

    @staticmethod
    def read_root_attrs(ds: nc.Dataset) -> Dict[str, Any]:
        """Extract all global attributes from the root group."""
        return {attr: ds.getncattr(attr) for attr in ds.ncattrs()}

    @staticmethod
    def get_group(ds: nc.Dataset, group_path: str) -> Optional[nc.Group]:
        """
        Navigate a slash-separated group path (e.g. 'passive_microwave/S1').
        Returns None if any group in the path does not exist.
        """
        parts = [p for p in group_path.strip("/").split("/") if p]
        curr: Any = ds
        for part in parts:
            if part not in curr.groups:
                return None
            curr = curr.groups[part]
        return curr

    @staticmethod
    def read_variable_scalar(group: nc.Group, var_name: str) -> Any:
        """Read a scalar variable from a group, returning a clean Python type."""
        if var_name not in group.variables:
            return None
        val = group.variables[var_name][:]
        return clean_scalar(val[0] if hasattr(val, "__len__") and len(val) > 0 else val)

    @staticmethod
    def read_variable_1d(group: nc.Group, var_name: str) -> Optional[np.ndarray]:
        """Read a 1D variable array."""
        if var_name not in group.variables:
            return None
        arr, _ = get_array_with_fill(group.variables[var_name])
        return arr

    @staticmethod
    def read_variable_2d(group: nc.Group, var_name: str, fill_value: float = np.nan) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Read a 2D variable array and its valid mask."""
        if var_name not in group.variables:
            return None, None
        return get_array_with_fill(group.variables[var_name], fill_value=fill_value)

