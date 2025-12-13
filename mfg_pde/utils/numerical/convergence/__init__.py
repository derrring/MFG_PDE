#!/usr/bin/env python3
"""
Convergence monitoring utilities for PDE solvers.

This package provides convergence monitoring utilities organized into:

- **convergence_metrics.py**: General-purpose utilities for any PDE
  - DistributionComparator: Wasserstein, KL divergence, moments
  - RollingConvergenceMonitor: Window-based statistical convergence
  - calculate_error: Unified L1/L2/Linf error computation
  - ConvergenceConfig: Configuration dataclass for solver integration

- **convergence_monitors.py**: MFG-specific monitors
  - DistributionConvergenceMonitor: Multi-criteria (Wasserstein, KL, moments)
  - SolverTypeDetector: Detect particle vs grid-based solvers
  - ConvergenceWrapper: Adaptive convergence based on solver type

Usage:
    # General-purpose convergence checking
    from mfg_pde.utils.numerical.convergence import calculate_error, RollingConvergenceMonitor

    error = calculate_error(u_new, u_old, dx=0.01, norm='l2')
    monitor = RollingConvergenceMonitor(window_size=10)

    # MFG-specific monitoring
    from mfg_pde.utils.numerical.convergence import DistributionConvergenceMonitor

    monitor = DistributionConvergenceMonitor(wasserstein_tol=1e-4)
    diagnostics = monitor.update(u_current, u_previous, m_current, x_grid)

    # Adaptive convergence wrapper
    from mfg_pde.utils.numerical.convergence import adaptive_convergence

    @adaptive_convergence()
    class MySolver(BaseMFGSolver):
        ...
"""

from __future__ import annotations

import warnings

# =============================================================================
# GENERAL-PURPOSE IMPORTS (convergence_metrics.py)
# =============================================================================
from .convergence_metrics import (
    ConvergenceConfig,
    # Core utilities
    DistributionComparator,
    RollingConvergenceMonitor,
    # Backward compatibility
    StochasticConvergenceMonitor,
    calculate_error,
    calculate_l2_convergence_metrics,
    # Factory
    create_rolling_monitor,
    create_stochastic_monitor,
)

# =============================================================================
# MFG-SPECIFIC IMPORTS (convergence_monitors.py)
# =============================================================================
from .convergence_monitors import (
    AdaptiveConvergenceWrapper,
    AdvancedConvergenceMonitor,
    ConvergenceWrapper,
    # Core monitors
    DistributionConvergenceMonitor,
    # Backward compatibility
    OscillationDetector,
    ParticleMethodDetector,
    SolverTypeDetector,
    # Internal (exposed for testing)
    _ErrorHistoryTracker,
    adaptive_convergence,
    create_default_monitor,
    # Factory and decorators
    create_distribution_monitor,
    test_particle_detection,
    wrap_solver_with_adaptive_convergence,
)

# =============================================================================
# DEPRECATION HELPERS
# =============================================================================


def _warn_deprecated(old_name: str, new_name: str) -> None:
    """Issue deprecation warning for renamed classes."""
    warnings.warn(
        f"{old_name} is deprecated and will be removed in v1.0.0. Use {new_name} instead.",
        DeprecationWarning,
        stacklevel=3,
    )


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # General-purpose (convergence_metrics.py)
    "DistributionComparator",
    "RollingConvergenceMonitor",
    "calculate_error",
    "calculate_l2_convergence_metrics",
    "ConvergenceConfig",
    "create_rolling_monitor",
    # MFG-specific (convergence_monitors.py)
    "DistributionConvergenceMonitor",
    "SolverTypeDetector",
    "ConvergenceWrapper",
    "_ErrorHistoryTracker",
    "create_distribution_monitor",
    "adaptive_convergence",
    "wrap_solver_with_adaptive_convergence",
    "test_particle_detection",
    # Backward compatibility aliases (deprecated)
    "StochasticConvergenceMonitor",  # -> RollingConvergenceMonitor
    "create_stochastic_monitor",  # -> create_rolling_monitor
    "OscillationDetector",  # -> _ErrorHistoryTracker
    "AdvancedConvergenceMonitor",  # -> DistributionConvergenceMonitor
    "ParticleMethodDetector",  # -> SolverTypeDetector
    "AdaptiveConvergenceWrapper",  # -> ConvergenceWrapper
    "create_default_monitor",  # -> create_distribution_monitor
]
