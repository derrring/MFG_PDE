#!/usr/bin/env python3
"""
DEPRECATED: Use mfg_pde.geometry.protocol instead.

This module is maintained for backward compatibility only.
All new code should import from .protocol directly.

Will be removed in v0.12.0.
"""

from __future__ import annotations

import warnings

# Re-export everything from protocol for backward compatibility
from .protocol import (
    GeometryProtocol,
    GeometryType,
    detect_geometry_type,
    is_geometry_compatible,
    validate_geometry,
)

# Issue deprecation warning after imports to comply with E402
warnings.warn(
    "mfg_pde.geometry.geometry_protocol is deprecated. "
    "Use 'from mfg_pde.geometry.protocol import ...' instead. "
    "This module will be removed in v0.12.0.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "GeometryProtocol",
    "GeometryType",
    "detect_geometry_type",
    "is_geometry_compatible",
    "validate_geometry",
]
