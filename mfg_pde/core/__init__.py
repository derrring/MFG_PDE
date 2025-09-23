from mfg_pde.geometry import BoundaryConditions

from .variational_mfg_problem import (
    VariationalMFGComponents,
    VariationalMFGProblem,
    create_obstacle_variational_mfg,
    create_quadratic_variational_mfg,
)
from .mfg_problem import ExampleMFGProblem, MFGComponents, MFGProblem, MFGProblemBuilder, create_mfg_problem
from .network_mfg_problem import (
    NetworkMFGComponents,
    NetworkMFGProblem,
    create_grid_mfg_problem,
    create_random_mfg_problem,
    create_scale_free_mfg_problem,
)

__all__ = [
    # Geometry
    "BoundaryConditions",
    # Core MFG components
    "ExampleMFGProblem",
    "MFGComponents",
    "MFGProblem",
    "MFGProblemBuilder",
    "create_mfg_problem",
    # Variational MFG
    "VariationalMFGComponents",
    "VariationalMFGProblem",
    "create_obstacle_variational_mfg",
    "create_quadratic_variational_mfg",
    # Network MFG
    "NetworkMFGComponents",
    "NetworkMFGProblem",
    "create_grid_mfg_problem",
    "create_random_mfg_problem",
    "create_scale_free_mfg_problem",
]
