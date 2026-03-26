"""
DEPRECATED: Gradient schemes moved to mfgarchon.operators.reconstruction.

**Status**: DEPRECATED as of v0.18.0 (2026-01-24)
**Removal**: Scheduled for v0.20.0

Migration Guide:
    OLD (deprecated):
        >>> from mfgarchon.geometry.operators.schemes.weno5 import compute_weno5_derivative_1d

    NEW (preferred):
        >>> from mfgarchon.operators.reconstruction import compute_weno5_derivative_1d

Created: 2026-01-18 (Issue #606 - WENO5 Operator Refactoring)
Deprecated: 2026-01-24 (Operator module separation)
"""

# Import actual implementation from new location
from mfgarchon.operators.reconstruction.weno import (
    compute_weno5_derivative_1d as _compute_weno5_derivative_1d,
)
from mfgarchon.utils.deprecation import deprecated_alias

# Create deprecated alias (warns on call)
compute_weno5_derivative_1d = deprecated_alias(
    "mfgarchon.geometry.operators.schemes.compute_weno5_derivative_1d",
    _compute_weno5_derivative_1d,
    since="v0.18.0",
)

__all__ = ["compute_weno5_derivative_1d"]
