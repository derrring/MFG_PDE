"""
Data utilities for MFG_PDE.

This module provides data handling and validation tools:
- polars_integration: Polars DataFrame integration
- validation: Data validation utilities
"""

from __future__ import annotations

# Re-export from submodules
from .validation import safe_solution_return, validate_mfg_solution

__all__ = ["safe_solution_return", "validate_mfg_solution"]

# Optional polars integration
try:
    from .polars_integration import MFGDataFrame  # noqa: F401

    __all__.append("MFGDataFrame")
except ImportError:
    pass
