"""
Iterative solvers and patterns for MFG systems.

This module provides:

1. **Abstract Iteration Patterns** (paradigm-agnostic):
   - PicardPattern: Fixed-point iteration protocol
   - AveragingPattern: Fictitious Play protocol
   - FixedPointIteratorBase: Base class for Picard iterators
   - AveragingIterator: Base class for averaging iterators

2. **Learning Rate Schedules**:
   - harmonic_schedule: 1/(k+1) - standard fictitious play
   - sqrt_schedule: 1/sqrt(k+1) - faster initial progress
   - polynomial_schedule: 1/(k+1)^p - configurable decay

3. **Convergence Utilities**:
   - check_convergence: Generic convergence checking
   - ConvergenceTracker: Track error history
   - apply_damping_generic: Generic damping
   - Re-exports from utils/convergence: calculate_error, RollingConvergenceMonitor

4. **Concrete Solvers** (for backward compatibility):
   - FixedPointSolver: High-level solver with hooks
   - MultiPopulationFixedPointSolver: Multi-population extension

Example (Abstract Pattern):
    from mfg_pde.alg.iterative.patterns import FixedPointIteratorBase

    class MyIterator(FixedPointIteratorBase[MyState]):
        def forward_step(self, state):
            return compute_next(state)
        # ... implement other methods

Example (Concrete Solver):
    from mfg_pde.alg.iterative import FixedPointSolver

    solver = FixedPointSolver()
    result = solver.solve(problem)
"""

# Abstract patterns (Issue #630)
# Concrete solvers (backward compatibility)
from .base import BaseIterativeSolver, BaseSolver

# Convergence utilities (local)
# Re-exports from utils/convergence for PDE-aware metrics (DRY - Issue #630)
from .convergence import (
    ConvergenceChecker,
    ConvergenceConfig,
    ConvergenceResult,
    ConvergenceTracker,
    RollingConvergenceMonitor,
    apply_damping_arrays,
    apply_damping_generic,
    calculate_error,
    check_convergence,
    check_convergence_simple,
    compute_absolute_change,
    compute_relative_change,
)
from .fixed_point import FixedPointResult, FixedPointSolver
from .multi_population import MultiPopulationFixedPointSolver
from .patterns import (
    AveragingIterator,
    AveragingPattern,
    FixedPointIteratorBase,
    PicardPattern,
)

# Learning rate schedules
from .schedules import (
    LEARNING_RATE_SCHEDULES,
    constant_schedule,
    get_schedule,
    harmonic_schedule,
    polynomial_schedule,
    sqrt_schedule,
)

__all__ = [
    # Abstract patterns
    "PicardPattern",
    "AveragingPattern",
    "FixedPointIteratorBase",
    "AveragingIterator",
    # Learning rate schedules
    "harmonic_schedule",
    "sqrt_schedule",
    "polynomial_schedule",
    "constant_schedule",
    "get_schedule",
    "LEARNING_RATE_SCHEDULES",
    # Convergence utilities (local)
    "ConvergenceResult",
    "ConvergenceTracker",
    "check_convergence",
    "check_convergence_simple",
    "apply_damping_generic",
    "apply_damping_arrays",
    "compute_relative_change",
    "compute_absolute_change",
    # Re-exports from utils/convergence (DRY)
    "calculate_error",
    "RollingConvergenceMonitor",
    "ConvergenceChecker",
    "ConvergenceConfig",
    # Base classes
    "BaseIterativeSolver",
    "BaseSolver",  # Backward compatibility alias
    # Concrete solvers
    "FixedPointSolver",
    "FixedPointResult",
    "MultiPopulationFixedPointSolver",
]
