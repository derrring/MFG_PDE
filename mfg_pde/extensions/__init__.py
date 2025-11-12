"""
Framework extensions for MFG problems.

Extensions provide optional problem formulations beyond the core MFGProblem:
- Network MFG: Problems on graph topologies
- Variational MFG: Obstacle avoidance and state constraints
- Multi-population MFG: Heterogeneous K-population games

Import from here for extended functionality:
    from mfg_pde.extensions import NetworkMFGProblem, MultiPopulationMFGProblem
"""

from .multi_population import MultiPopulationMFGProblem, MultiPopulationMFGProtocol
from .network import (
    NetworkMFGComponents,
    NetworkMFGProblem,
    create_grid_mfg_problem,
    create_random_mfg_problem,
    create_scale_free_mfg_problem,
)
from .variational import (
    VariationalMFGComponents,
    VariationalMFGProblem,
    create_obstacle_variational_mfg,
    create_quadratic_variational_mfg,
)

__all__ = [
    # Multi-population MFG
    "MultiPopulationMFGProblem",
    "MultiPopulationMFGProtocol",
    # Network MFG
    "NetworkMFGComponents",
    "NetworkMFGProblem",
    "create_grid_mfg_problem",
    "create_random_mfg_problem",
    "create_scale_free_mfg_problem",
    # Variational MFG
    "VariationalMFGComponents",
    "VariationalMFGProblem",
    "create_obstacle_variational_mfg",
    "create_quadratic_variational_mfg",
]
