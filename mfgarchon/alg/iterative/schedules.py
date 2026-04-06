"""DEPRECATED: Use mfgarchon.utils.convergence.schedules instead.

This module is a compatibility shim. Will be removed in v0.20.0.
"""

import warnings

warnings.warn(
    "Importing from mfgarchon.alg.iterative.schedules is deprecated. "
    "Use mfgarchon.utils.convergence.schedules instead. "
    "Will be removed in v0.20.0.",
    DeprecationWarning,
    stacklevel=2,
)

from mfgarchon.utils.convergence.schedules import *  # noqa: E402, F403
