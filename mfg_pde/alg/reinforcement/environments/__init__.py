"""
MFG-RL Environments

Environment infrastructure for reinforcement learning approaches to Mean Field Games.
Provides maze generators, MFG-specific environments, and utilities for RL experiments.

Modules:
- maze_generator: Perfect maze generation algorithms for RL experiments
- maze_config: Comprehensive configuration for maze parameters
- position_placement: Strategies for start/goal position placement
- recursive_division: Variable-width mazes with rooms and open spaces
"""

from mfg_pde.alg.reinforcement.environments.maze_config import (
    MazeConfig,
    MazeTopology,
    PhysicalDimensions,
    PlacementStrategy,
    create_continuous_maze_config,
    create_default_config,
    create_multi_goal_config,
)
from mfg_pde.alg.reinforcement.environments.maze_generator import (
    Cell,
    Grid,
    MazeAlgorithm,
    PerfectMazeGenerator,
    generate_maze,
    verify_perfect_maze,
)
from mfg_pde.alg.reinforcement.environments.position_placement import (
    compute_position_metrics,
    place_positions,
)
from mfg_pde.alg.reinforcement.environments.recursive_division import (
    RecursiveDivisionConfig,
    RecursiveDivisionGenerator,
    SplitOrientation,
    add_loops,
    create_room_based_config,
)

__all__ = [
    # Perfect maze generation
    "MazeAlgorithm",
    "PerfectMazeGenerator",
    "Grid",
    "Cell",
    "generate_maze",
    "verify_perfect_maze",
    # Configuration
    "MazeConfig",
    "MazeTopology",
    "PhysicalDimensions",
    "PlacementStrategy",
    "create_default_config",
    "create_continuous_maze_config",
    "create_multi_goal_config",
    # Position placement
    "place_positions",
    "compute_position_metrics",
    # Recursive Division (variable-width mazes)
    "RecursiveDivisionConfig",
    "RecursiveDivisionGenerator",
    "SplitOrientation",
    "add_loops",
    "create_room_based_config",
]
