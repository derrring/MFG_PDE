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

All operators are re-exported here for backward compatibility. Deprecation warnings
are emitted when classes are instantiated (not on import).

Created: Original location for operators (Issue #595)
Deprecated: 2026-01-24 (Operator module separation)
Scheduled Removal: v0.20.0
"""

# Import actual implementations from new location
from mfg_pde.operators.differential.advection import (
    AdvectionOperator as _AdvectionOperator,
)
from mfg_pde.operators.differential.divergence import (
    DivergenceOperator as _DivergenceOperator,
)
from mfg_pde.operators.differential.gradient import (
    GradientComponentOperator as _GradientComponentOperator,
)
from mfg_pde.operators.differential.gradient import (
    create_gradient_operators as _create_gradient_operators,
)
from mfg_pde.operators.differential.interface_jump import (
    InterfaceJumpOperator as _InterfaceJumpOperator,
)
from mfg_pde.operators.differential.laplacian import (
    LaplacianOperator as _LaplacianOperator,
)
from mfg_pde.operators.interpolation.interpolation import (
    InterpolationOperator as _InterpolationOperator,
)
from mfg_pde.operators.interpolation.projection import (
    GeometryProjector as _GeometryProjector,
)
from mfg_pde.operators.interpolation.projection import (
    ProjectionRegistry as _ProjectionRegistry,
)
from mfg_pde.utils.deprecation import deprecated_alias

# Create deprecated aliases (warn on instantiation, not import)
LaplacianOperator = deprecated_alias(
    "mfg_pde.geometry.operators.LaplacianOperator",
    _LaplacianOperator,
    since="v0.18.0",
)
GradientComponentOperator = deprecated_alias(
    "mfg_pde.geometry.operators.GradientComponentOperator",
    _GradientComponentOperator,
    since="v0.18.0",
)
DivergenceOperator = deprecated_alias(
    "mfg_pde.geometry.operators.DivergenceOperator",
    _DivergenceOperator,
    since="v0.18.0",
)
AdvectionOperator = deprecated_alias(
    "mfg_pde.geometry.operators.AdvectionOperator",
    _AdvectionOperator,
    since="v0.18.0",
)
InterpolationOperator = deprecated_alias(
    "mfg_pde.geometry.operators.InterpolationOperator",
    _InterpolationOperator,
    since="v0.18.0",
)
InterfaceJumpOperator = deprecated_alias(
    "mfg_pde.geometry.operators.InterfaceJumpOperator",
    _InterfaceJumpOperator,
    since="v0.18.0",
)
GeometryProjector = deprecated_alias(
    "mfg_pde.geometry.operators.GeometryProjector",
    _GeometryProjector,
    since="v0.18.0",
)
ProjectionRegistry = deprecated_alias(
    "mfg_pde.geometry.operators.ProjectionRegistry",
    _ProjectionRegistry,
    since="v0.18.0",
)
create_gradient_operators = deprecated_alias(
    "mfg_pde.geometry.operators.create_gradient_operators",
    _create_gradient_operators,
    since="v0.18.0",
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
