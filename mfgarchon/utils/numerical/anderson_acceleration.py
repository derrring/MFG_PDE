"""
DEPRECATED: This module has moved to mfgarchon.alg.numerical.coupling.anderson_acceleration.

This shim re-exports all public names for backward compatibility.
Will be removed in v0.21.0 (3 versions after v0.18.0).
"""

from __future__ import annotations

import warnings

warnings.warn(
    "Importing from mfgarchon.utils.numerical.anderson_acceleration is deprecated. "
    "Use mfgarchon.alg.numerical.coupling.anderson_acceleration instead. "
    "Will be removed in v0.21.0.",
    DeprecationWarning,
    stacklevel=2,
)

from mfgarchon.alg.numerical.coupling.anderson_acceleration import (  # noqa: E402, F401
    AndersonAccelerator,
    create_anderson_accelerator,
)
