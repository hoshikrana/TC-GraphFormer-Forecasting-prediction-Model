"""
Infrastructure, hardware discovery, data staging, and shard export.
"""

from .hardware_info import discover_hardware
from .export_shards import compute_sha256

__all__ = ["discover_hardware", "compute_sha256"]

