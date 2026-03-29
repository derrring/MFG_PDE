"""
DEPRECATED: This module has moved to mfgarchon.geometry.meshes.mesh_distances.

This shim re-exports all public names for backward compatibility.
Will be removed in v0.21.0 (3 versions after v0.18.0).
"""

from __future__ import annotations

import warnings

warnings.warn(
    "Importing from mfgarchon.utils.numerical.mesh_distances is deprecated. "
    "Use mfgarchon.geometry.meshes.mesh_distances instead. "
    "Will be removed in v0.21.0.",
    DeprecationWarning,
    stacklevel=2,
)

from mfgarchon.geometry.meshes.mesh_distances import (  # noqa: E402, F401
    MeshDistances,
    compute_distances_for_eoc_study,
    compute_mesh_distances,
)
