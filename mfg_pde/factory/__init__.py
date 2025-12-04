"""
MFG Solver Factory Module

Provides factory patterns for easy creation of problems and solver configurations.

Problem factories:
- create_mfg_problem() - Main factory for all problem types
- create_standard_problem() - Standard HJB-FP MFG
- create_network_problem() - Network/Graph MFG
- create_variational_problem() - Variational/Lagrangian MFG
- create_stochastic_problem() - Stochastic MFG with common noise
- create_highdim_problem() - High-dimensional MFG (d > 3)
- create_lq_problem() - Linear-Quadratic MFG
- create_crowd_problem() - Crowd dynamics MFG

Solver:
- SolverFactory - Main solver factory class
- create_solver() - Convenience function for creating solvers

For most use cases, prefer problem.solve() over create_solver().
"""

# Backend support
from .backend_factory import BackendFactory, create_backend_for_problem, print_backend_info

# General purpose factory
from .general_mfg_factory import GeneralMFGFactory, create_general_mfg_problem, get_general_factory

# Problem factories
from .problem_factories import (
    create_crowd_problem,
    create_highdim_problem,
    create_lq_problem,
    create_mfg_problem,
    create_network_problem,
    create_standard_problem,
    create_stochastic_problem,
    create_variational_problem,
)

# Solver factory
from .solver_factory import (
    SolverFactory,
    create_solver,
)

__all__ = [
    # Problem factories
    "create_mfg_problem",
    "create_standard_problem",
    "create_network_problem",
    "create_variational_problem",
    "create_stochastic_problem",
    "create_highdim_problem",
    "create_lq_problem",
    "create_crowd_problem",
    # Solver
    "SolverFactory",
    "create_solver",
    # Backend support
    "BackendFactory",
    "create_backend_for_problem",
    "print_backend_info",
    # General purpose factory
    "GeneralMFGFactory",
    "create_general_mfg_problem",
    "get_general_factory",
]
