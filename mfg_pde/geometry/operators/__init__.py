"""
DEPRECATED: Geometric operators for MFG problems.

**Status**: DEPRECATED as of v0.18.0 (2026-01-24)
**Removal**: Scheduled for v0.20.0

This module has been moved to `mfg_pde.operators`. Please update imports:

Migration Guide:
    OLD (deprecated):
        >>> from mfg_pde.geometry.operators import LaplacianOperator, AdvectionOperator
        >>> from mfg_pde.geometry.operators.laplacian import LaplacianOperator

    NEW (preferred):
        >>> from mfg_pde.operators import LaplacianOperator, AdvectionOperator
        >>> from mfg_pde.operators.differential.laplacian import LaplacianOperator

All operators are re-exported here for backward compatibility but will emit
deprecation warnings in development mode.

Created: Original location for operators (Issue #595)
Deprecated: 2026-01-24 (Operator module separation)
Scheduled Removal: v0.20.0
"""

import warnings

# Re-export from new location for backward compatibility
from mfg_pde.operators.differential.advection import AdvectionOperator
from mfg_pde.operators.differential.divergence import DivergenceOperator
from mfg_pde.operators.differential.gradient import (
    GradientComponentOperator,
    create_gradient_operators,
)
from mfg_pde.operators.differential.interface_jump import InterfaceJumpOperator
from mfg_pde.operators.differential.laplacian import LaplacianOperator
from mfg_pde.operators.interpolation.interpolation import InterpolationOperator
from mfg_pde.operators.interpolation.projection import (
    GeometryProjector,
    ProjectionRegistry,
)

# Emit deprecation warning on module import
warnings.warn(
    "mfg_pde.geometry.operators is deprecated. Use mfg_pde.operators instead. This module will be removed in v0.20.0.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    # Projection
    "GeometryProjector",
    "ProjectionRegistry",
    # Differential operators
    "LaplacianOperator",
    "GradientComponentOperator",
    "DivergenceOperator",
    "AdvectionOperator",
    "InterpolationOperator",
    "InterfaceJumpOperator",
    # Factory functions
    "create_gradient_operators",
]
