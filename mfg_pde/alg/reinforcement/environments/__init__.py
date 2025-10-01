"""
MFG-RL Environments

Environment infrastructure for reinforcement learning approaches to Mean Field Games.
Provides maze generators, MFG-specific environments, and utilities for RL experiments.

Modules:
- maze_generator: Perfect maze generation algorithms for RL experiments
- maze_config: Comprehensive configuration for maze parameters
- position_placement: Strategies for start/goal position placement
- recursive_division: Variable-width mazes with rooms and open spaces
- cellular_automata: Organic, cave-like mazes using cellular automata
- mfg_maze_env: Gymnasium-compatible MFG maze environments with population dynamics
"""

from mfg_pde.alg.reinforcement.environments.cellular_automata import (
    CellularAutomataConfig,
    CellularAutomataGenerator,
    create_preset_ca_config,
)
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

# Conditional import for MFG environment (requires Gymnasium)
try:
    from mfg_pde.alg.reinforcement.environments.mfg_maze_env import (
        ActionType,
        MFGMazeConfig,
        MFGMazeEnvironment,
        PopulationState,
        RewardType,
    )

    MFG_ENV_AVAILABLE = True
except ImportError:
    MFG_ENV_AVAILABLE = False
    ActionType = None  # type: ignore
    MFGMazeConfig = None  # type: ignore
    MFGMazeEnvironment = None  # type: ignore
    PopulationState = None  # type: ignore
    RewardType = None  # type: ignore

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
    # Cellular Automata (organic mazes)
    "CellularAutomataConfig",
    "CellularAutomataGenerator",
    "create_preset_ca_config",
    # Availability flags
    "MFG_ENV_AVAILABLE",
]

# Add MFG environment exports if available
if MFG_ENV_AVAILABLE:
    __all__.extend(
        [
            "MFGMazeEnvironment",
            "MFGMazeConfig",
            "PopulationState",
            "ActionType",
            "RewardType",
        ]
    )
