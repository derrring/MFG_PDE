"""
Backward compatibility re-exports from mfgarchon.utils.convergence.

DEPRECATED: This module has been moved to mfgarchon.utils.convergence.
Import from there instead:

    # Old (deprecated)
    from mfgarchon.utils.numerical.convergence import ConvergenceConfig

    # New (preferred)
    from mfgarchon.utils.convergence import ConvergenceConfig

This re-export will be removed in v2.0.0.
"""

from __future__ import annotations

import warnings

# Re-export everything from the new location
from mfgarchon.utils.convergence import (
    AdaptiveConvergenceWrapper,
    AdvancedConvergenceMonitor,
    ConvergenceChecker,
    ConvergenceConfig,
    ConvergenceWrapper,
    DistributionComparator,
    DistributionConvergenceMonitor,
    FPConvergenceChecker,
    HJBConvergenceChecker,
    MFGConvergenceChecker,
    OscillationDetector,
    ParticleMethodDetector,
    RollingConvergenceMonitor,
    SolverTypeDetector,
    StochasticConvergenceMonitor,
    _ErrorHistoryTracker,
    adaptive_convergence,
    calculate_error,
    calculate_l2_convergence_metrics,
    compute_norm,
    create_default_monitor,
    create_distribution_monitor,
    create_rolling_monitor,
    create_stochastic_monitor,
    test_particle_detection,
    wrap_solver_with_adaptive_convergence,
)

warnings.warn(
    "mfgarchon.utils.numerical.convergence is deprecated. Import from mfgarchon.utils.convergence instead. Will be removed in v1.0.0.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    # General-purpose
    "DistributionComparator",
    "RollingConvergenceMonitor",
    "calculate_error",
    "calculate_l2_convergence_metrics",
    "ConvergenceConfig",
    "create_rolling_monitor",
    # MFG-specific
    "DistributionConvergenceMonitor",
    "SolverTypeDetector",
    "ConvergenceWrapper",
    "_ErrorHistoryTracker",
    "create_distribution_monitor",
    "adaptive_convergence",
    "wrap_solver_with_adaptive_convergence",
    "test_particle_detection",
    # Convergence checkers
    "ConvergenceChecker",
    "HJBConvergenceChecker",
    "FPConvergenceChecker",
    "MFGConvergenceChecker",
    "compute_norm",
    # Backward compatibility aliases
    "StochasticConvergenceMonitor",
    "create_stochastic_monitor",
    "OscillationDetector",
    "AdvancedConvergenceMonitor",
    "ParticleMethodDetector",
    "AdaptiveConvergenceWrapper",
    "create_default_monitor",
]
