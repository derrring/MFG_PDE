"""
MFG Solver Factory Module

Provides factory patterns for easy creation of optimized solver configurations.
"""

from .solver_factory import (
    SolverFactory,
    create_solver,
    create_fast_solver,
    create_accurate_solver,
    create_research_solver,
    create_monitored_solver
)

__all__ = [
    "SolverFactory",
    "create_solver",
    "create_fast_solver", 
    "create_accurate_solver",
    "create_research_solver",
    "create_monitored_solver"
]