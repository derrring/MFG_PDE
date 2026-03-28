"""
Backward-compatible re-export module.

All definitions have been moved to:
- protocols.py: Base classes, protocols, enums, handler protocols
- calculators.py: Calculator implementations, topology classes
- ghost_cells.py: Ghost cell formula functions

This module re-exports everything for backward compatibility.
"""

from .calculators import *  # noqa: F403

# Reconstruct __all__ from sub-modules
from .calculators import __all__ as _calculators_all
from .ghost_cells import *  # noqa: F403
from .ghost_cells import __all__ as _ghost_cells_all
from .protocols import *  # noqa: F403
from .protocols import __all__ as _protocols_all

__all__ = list(_protocols_all) + list(_calculators_all) + list(_ghost_cells_all)
