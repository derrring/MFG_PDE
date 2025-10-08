"""
I/O utilities for MFG_PDE solver data persistence.

This module provides file format support for saving and loading MFG solver results:
- HDF5: Primary format for solver solutions (recommended)
- Checkpoint/resume capability for long-running computations
"""

from __future__ import annotations

# Check HDF5 availability
try:
    from .hdf5_utils import (
        HDF5_AVAILABLE,
        get_hdf5_info,
        load_checkpoint,
        load_hdf5,
        load_solution,
        save_checkpoint,
        save_hdf5,
        save_solution,
    )
except ImportError:
    HDF5_AVAILABLE = False

    def _hdf5_unavailable(*args, **kwargs):
        raise ImportError("HDF5 support requires h5py. Install with: pip install h5py")

    save_solution = _hdf5_unavailable
    load_solution = _hdf5_unavailable
    save_checkpoint = _hdf5_unavailable
    load_checkpoint = _hdf5_unavailable
    get_hdf5_info = _hdf5_unavailable
    save_hdf5 = _hdf5_unavailable
    load_hdf5 = _hdf5_unavailable

__all__ = [
    "HDF5_AVAILABLE",
    "get_hdf5_info",
    "load_checkpoint",
    "load_hdf5",
    "load_solution",
    "save_checkpoint",
    "save_hdf5",
    "save_solution",
]
