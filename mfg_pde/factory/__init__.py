"""
MFG Solver Factory Module

Provides factory patterns for easy creation of problems and solver configurations.

New Unified API (Phase 3.3)
---------------------------
Problem factories with dual-output support:
- create_mfg_problem() - Main factory for all problem types
- create_standard_problem() - Standard HJB-FP MFG
- create_network_problem() - Network/Graph MFG
- create_variational_problem() - Variational/Lagrangian MFG
- create_stochastic_problem() - Stochastic MFG with common noise
- create_highdim_problem() - High-dimensional MFG (d > 3)
- create_lq_problem() - Linear-Quadratic MFG
- create_crowd_problem() - Crowd dynamics MFG

Solver factories (updated for Phase 3.2 SolverConfig):
- create_solver() - Main solver factory
- create_fast_solver() - Fast configuration
- create_accurate_solver() - Accurate configuration
- create_research_solver() - Research configuration

Legacy API (backward compatibility):
- All factories support use_unified parameter
- Old specialized problem classes still available
"""

# Problem factories (NEW - Phase 3.3)
# Backend support
from .backend_factory import BackendFactory, create_backend_for_problem, print_backend_info

# General purpose factory (existing)
from .general_mfg_factory import GeneralMFGFactory, create_general_mfg_problem, get_general_factory
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

# Solver factories (existing, will be updated for Phase 3.2 SolverConfig)
from .solver_factory import (
    SolverFactory,
    create_accurate_solver,
    create_amr_solver,
    create_basic_solver,
    create_fast_solver,
    create_research_solver,
    create_semi_lagrangian_solver,
    create_solver,
    create_standard_solver,
)

__all__ = [  # noqa: RUF022 - Intentionally organized by category, not alphabetically
    # =========================================================================
    # NEW UNIFIED PROBLEM FACTORIES (Phase 3.3 - RECOMMENDED)
    # =========================================================================
    "create_mfg_problem",  # Main factory for all types
    "create_standard_problem",  # Standard HJB-FP
    "create_network_problem",  # Network/Graph MFG
    "create_variational_problem",  # Variational/Lagrangian
    "create_stochastic_problem",  # Stochastic with common noise
    "create_highdim_problem",  # High-dimensional (d > 3)
    "create_lq_problem",  # Linear-Quadratic MFG
    "create_crowd_problem",  # Crowd dynamics
    # =========================================================================
    # SOLVER FACTORIES (will be updated for Phase 3.2 SolverConfig)
    # =========================================================================
    "SolverFactory",
    "create_solver",  # Main solver factory
    "create_fast_solver",
    "create_accurate_solver",
    "create_research_solver",
    "create_standard_solver",
    "create_basic_solver",
    "create_semi_lagrangian_solver",
    "create_amr_solver",
    # =========================================================================
    # BACKEND SUPPORT
    # =========================================================================
    "BackendFactory",
    "create_backend_for_problem",
    "print_backend_info",
    # =========================================================================
    # GENERAL PURPOSE FACTORY (existing)
    # =========================================================================
    "GeneralMFGFactory",
    "create_general_mfg_problem",
    "get_general_factory",
]
