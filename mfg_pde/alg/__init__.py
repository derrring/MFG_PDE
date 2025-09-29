"""
New algorithm structure for MFG_PDE.

This module implements the reorganized algorithm structure with paradigm-based organization:
- numerical: Classical numerical analysis methods
- optimization: Direct optimization approaches
- neural: Neural network-based methods
- reinforcement: Reinforcement learning paradigm

The structure emphasizes interconnections between paradigms and supports
hybrid methods that combine multiple approaches.
"""

from __future__ import annotations

# Import paradigm modules
from . import neural, numerical, optimization, reinforcement

__all__ = [
    "numerical",
    "optimization",
    "neural",
    "reinforcement",
]
