"""
Performance utilities for MFG_PDE.

This module provides performance monitoring and optimization tools:
- monitoring: Performance profiling and timing utilities
- optimization: Memory and computational optimization helpers
"""

from __future__ import annotations

# Re-export from submodules
from .monitoring import PerformanceMonitor
from .optimization import optimize_array_operations

__all__ = ["PerformanceMonitor", "optimize_array_operations"]
