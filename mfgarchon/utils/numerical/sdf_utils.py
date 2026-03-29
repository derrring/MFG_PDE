"""
DEPRECATED: This module has moved to mfgarchon.geometry.implicit.sdf_utils.

This shim re-exports all public names for backward compatibility.
Will be removed in v0.21.0 (3 versions after v0.18.0).
"""

from __future__ import annotations

import warnings

warnings.warn(
    "Importing from mfgarchon.utils.numerical.sdf_utils is deprecated. "
    "Use mfgarchon.geometry.implicit.sdf_utils instead. "
    "Will be removed in v0.21.0.",
    DeprecationWarning,
    stacklevel=2,
)

from mfgarchon.geometry.implicit.sdf_utils import (  # noqa: E402, F401
    sdf_box,
    sdf_complement,
    sdf_difference,
    sdf_gradient,
    sdf_intersection,
    sdf_smooth_intersection,
    sdf_smooth_union,
    sdf_sphere,
    sdf_union,
)
