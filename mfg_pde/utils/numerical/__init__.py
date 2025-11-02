"""
Numerical utilities for MFG_PDE.

This module provides numerical computation utilities:
- anderson_acceleration: Anderson acceleration for fixed-point iteration
- convergence: Convergence analysis and criteria
- functional_calculus: Functional calculus operations
- integration: Numerical integration helpers
- monte_carlo: Monte Carlo methods
- mcmc: Markov Chain Monte Carlo methods
- nonlinear_solvers: Fundamental solvers for nonlinear systems
- hjb_policy_iteration: Policy iteration utilities for HJB-MFG problems
"""

from __future__ import annotations

# Re-export from submodules
from .anderson_acceleration import AndersonAccelerator
from .convergence import AdvancedConvergenceMonitor
from .hjb_policy_iteration import (
    LQPolicyIterationHelper,
    create_lq_policy_problem,
    policy_iteration_hjb,
)
from .integration import get_integration_info, trapezoid
from .monte_carlo import monte_carlo_integrate
from .nonlinear_solvers import (
    FixedPointSolver,
    NewtonSolver,
    PolicyIterationSolver,
    SolverInfo,
)

__all__ = [
    "AdvancedConvergenceMonitor",
    "AndersonAccelerator",
    "FixedPointSolver",
    "LQPolicyIterationHelper",
    "NewtonSolver",
    "PolicyIterationSolver",
    "SolverInfo",
    "create_lq_policy_problem",
    "get_integration_info",
    "monte_carlo_integrate",
    "policy_iteration_hjb",
    "trapezoid",
]
