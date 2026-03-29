"""
DEPRECATED: This module has moved to mfgarchon.alg.numerical.gfdm_components.gfdm_strategies.

This shim re-exports all public names for backward compatibility.
Will be removed in v0.21.0 (3 versions after v0.18.0).
"""

from __future__ import annotations

import warnings

warnings.warn(
    "Importing from mfgarchon.utils.numerical.gfdm_strategies is deprecated. "
    "Use mfgarchon.alg.numerical.gfdm_components.gfdm_strategies instead. "
    "Will be removed in v0.21.0.",
    DeprecationWarning,
    stacklevel=2,
)

from mfgarchon.alg.numerical.gfdm_components.gfdm_strategies import (  # noqa: E402, F401
    BCConfig,
    BoundaryHandler,
    DifferentialOperator,
    DirectCollocationHandler,
    GhostNodeHandler,
    LocalRBFOperator,
    OperatorConfig,
    TaylorOperator,
    UpwindOperator,
    compute_adaptive_gfdm_params,
    compute_gfdm_parameters,
    create_bc_handler,
    create_operator,
    wendland_c2_effective_ratio,
)
