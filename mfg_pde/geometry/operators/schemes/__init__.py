"""
DEPRECATED: Gradient schemes moved to mfg_pde.operators.reconstruction.

**Status**: DEPRECATED as of v0.18.0 (2026-01-24)
**Removal**: Scheduled for v0.20.0

Migration Guide:
    OLD (deprecated):
        >>> from mfg_pde.geometry.operators.schemes.weno5 import compute_weno5_derivative_1d

    NEW (preferred):
        >>> from mfg_pde.operators.reconstruction import compute_weno5_derivative_1d

Created: 2026-01-18 (Issue #606 - WENO5 Operator Refactoring)
Deprecated: 2026-01-24 (Operator module separation)
"""

import warnings

# Re-export from new location
from mfg_pde.operators.reconstruction.weno import compute_weno5_derivative_1d

# Emit deprecation warning
warnings.warn(
    "mfg_pde.geometry.operators.schemes is deprecated. "
    "Use mfg_pde.operators.reconstruction instead. "
    "This module will be removed in v0.20.0.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["compute_weno5_derivative_1d"]
