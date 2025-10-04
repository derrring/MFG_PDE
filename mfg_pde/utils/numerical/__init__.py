"""
Numerical utilities for MFG_PDE.

This module provides numerical computation utilities:
- anderson_acceleration: Anderson acceleration for fixed-point iteration
- convergence: Convergence analysis and criteria
- functional_calculus: Functional calculus operations
- integration: Numerical integration helpers
- monte_carlo: Monte Carlo methods
- mcmc: Markov Chain Monte Carlo methods
"""

from __future__ import annotations

# Re-export from submodules
from .anderson_acceleration import AndersonAccelerator
from .convergence import AdvancedConvergenceMonitor
from .integration import get_integration_info, trapezoid
from .monte_carlo import monte_carlo_sample

__all__ = [
    "AndersonAccelerator",
    "AdvancedConvergenceMonitor",
    "get_integration_info",
    "trapezoid",
    "monte_carlo_sample",
]
