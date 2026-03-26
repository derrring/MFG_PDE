"""
DEPRECATED: Geometric operators for MFG problems.

**Status**: DEPRECATED as of v0.18.0 (2026-01-24)
**Removal**: Scheduled for v0.20.0

This module has been moved to `mfgarchon.operators`. Please update imports:

Migration Guide:
    OLD (deprecated):
        >>> from mfgarchon.geometry.operators import LaplacianOperator, AdvectionOperator
        >>> from mfgarchon.geometry.operators.laplacian import LaplacianOperator

    NEW (preferred):
        >>> from mfgarchon.operators import LaplacianOperator, AdvectionOperator
        >>> from mfgarchon.operators.differential.laplacian import LaplacianOperator

All operators are re-exported here for backward compatibility. Deprecation warnings
are emitted when classes are instantiated (not on import).

Created: Original location for operators (Issue #595)
Deprecated: 2026-01-24 (Operator module separation)
Scheduled Removal: v0.20.0
"""

# Import actual implementations from new location
from mfgarchon.operators.differential.advection import (
    AdvectionOperator as _AdvectionOperator,
)
from mfgarchon.operators.differential.divergence import (
    DivergenceOperator as _DivergenceOperator,
)
from mfgarchon.operators.differential.gradient import (
    PartialDerivOperator as _PartialDerivOperator,
)
from mfgarchon.operators.differential.interface_jump import (
    InterfaceJumpOperator as _InterfaceJumpOperator,
)
from mfgarchon.operators.differential.laplacian import (
    LaplacianOperator as _LaplacianOperator,
)
from mfgarchon.operators.interpolation.interpolation import (
    InterpolationOperator as _InterpolationOperator,
)
from mfgarchon.operators.interpolation.projection import (
    GeometryProjector as _GeometryProjector,
)
from mfgarchon.operators.interpolation.projection import (
    ProjectionRegistry as _ProjectionRegistry,
)
from mfgarchon.utils.deprecation import deprecated_alias

# Create deprecated aliases (warn on instantiation, not import)
LaplacianOperator = deprecated_alias(
    "mfgarchon.geometry.operators.LaplacianOperator",
    _LaplacianOperator,
    since="v0.18.0",
)
GradientComponentOperator = deprecated_alias(
    "mfgarchon.geometry.operators.GradientComponentOperator",
    _PartialDerivOperator,  # Points to new name
    since="v0.18.0",
)
PartialDerivOperator = deprecated_alias(
    "mfgarchon.geometry.operators.PartialDerivOperator",
    _PartialDerivOperator,
    since="v0.18.0",
)
DivergenceOperator = deprecated_alias(
    "mfgarchon.geometry.operators.DivergenceOperator",
    _DivergenceOperator,
    since="v0.18.0",
)
AdvectionOperator = deprecated_alias(
    "mfgarchon.geometry.operators.AdvectionOperator",
    _AdvectionOperator,
    since="v0.18.0",
)
InterpolationOperator = deprecated_alias(
    "mfgarchon.geometry.operators.InterpolationOperator",
    _InterpolationOperator,
    since="v0.18.0",
)
InterfaceJumpOperator = deprecated_alias(
    "mfgarchon.geometry.operators.InterfaceJumpOperator",
    _InterfaceJumpOperator,
    since="v0.18.0",
)
GeometryProjector = deprecated_alias(
    "mfgarchon.geometry.operators.GeometryProjector",
    _GeometryProjector,
    since="v0.18.0",
)
ProjectionRegistry = deprecated_alias(
    "mfgarchon.geometry.operators.ProjectionRegistry",
    _ProjectionRegistry,
    since="v0.18.0",
)

__all__ = [
    # Projection
    "GeometryProjector",
    "ProjectionRegistry",
    # Differential operators
    "LaplacianOperator",
    "PartialDerivOperator",
    "GradientComponentOperator",  # Deprecated alias for PartialDerivOperator
    "DivergenceOperator",
    "AdvectionOperator",
    "InterpolationOperator",
    "InterfaceJumpOperator",
]
