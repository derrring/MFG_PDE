"""
MFG Solver Factory Module

Provides factory patterns for easy creation of optimized solver configurations.
"""

from .backend_factory import (
    BackendFactory,
    create_backend_for_problem,
    print_backend_info,
)
from .solver_factory import (
    create_accurate_solver,
    create_fast_solver,
    create_monitored_solver,
    create_research_solver,
    create_solver,
    SolverFactory,
)

# Removed specific model factory - use GeneralMFGFactory instead

__all__ = [
    "SolverFactory",
    "create_solver",
    "create_fast_solver",
    "create_accurate_solver",
    "create_research_solver",
    "create_monitored_solver",
    "BackendFactory",
    "create_backend_for_problem",
    "print_backend_info",
]
