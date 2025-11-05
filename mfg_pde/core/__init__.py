from mfg_pde.geometry import BoundaryConditions

# New dimension-agnostic infrastructure
from .base_problem import BaseMFGProblem, MFGProblemProtocol
from .mfg_problem import MFGProblem

# Legacy imports for backward compatibility
from .mfg_problem_legacy import ExampleMFGProblem, MFGComponents, MFGProblemBuilder, create_mfg_problem

# Specialized MFG problems
from .network_mfg_problem import (
    NetworkMFGComponents,
    NetworkMFGProblem,
    create_grid_mfg_problem,
    create_random_mfg_problem,
    create_scale_free_mfg_problem,
)
from .variational_mfg_problem import (
    VariationalMFGComponents,
    VariationalMFGProblem,
    create_obstacle_variational_mfg,
    create_quadratic_variational_mfg,
)

__all__ = [
    # Geometry
    "BoundaryConditions",
    # New dimension-agnostic base
    "BaseMFGProblem",
    "MFGProblemProtocol",
    "MFGProblem",
    # Legacy components (backward compatibility)
    "ExampleMFGProblem",
    "MFGComponents",
    "MFGProblemBuilder",
    # Network MFG
    "NetworkMFGComponents",
    "NetworkMFGProblem",
    # Variational MFG
    "VariationalMFGComponents",
    "VariationalMFGProblem",
    "create_grid_mfg_problem",
    "create_mfg_problem",
    "create_obstacle_variational_mfg",
    "create_quadratic_variational_mfg",
    "create_random_mfg_problem",
    "create_scale_free_mfg_problem",
]
