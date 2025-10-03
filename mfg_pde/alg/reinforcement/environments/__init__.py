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
from mfg_pde.alg.reinforcement.environments.hybrid_maze import (
    AlgorithmSpec,
    HybridMazeConfig,
    HybridMazeGenerator,
    HybridStrategy,
    create_campus_hybrid,
    create_museum_hybrid,
    create_office_hybrid,
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
from mfg_pde.alg.reinforcement.environments.voronoi_maze import (
    VoronoiMazeConfig,
    VoronoiMazeGenerator,
)

# Maze utilities (connectivity analysis)
try:
    from mfg_pde.alg.reinforcement.environments.maze_utils import (
        analyze_maze_connectivity,
        compute_adaptive_door_width,
        connect_regions_adaptive,
        find_disconnected_regions,
        find_region_boundary,
    )

    MAZE_UTILS_AVAILABLE = True
except ImportError:
    MAZE_UTILS_AVAILABLE = False
    analyze_maze_connectivity = None  # type: ignore
    compute_adaptive_door_width = None  # type: ignore
    connect_regions_adaptive = None  # type: ignore
    find_disconnected_regions = None  # type: ignore
    find_region_boundary = None  # type: ignore

# Maze post-processing (smoothing, enhancement, refinement)
try:
    from mfg_pde.alg.reinforcement.environments.maze_postprocessing import (
        adaptive_door_carving,
        enhance_organic_maze,
        normalize_wall_thickness,
        smooth_walls_combined,
        smooth_walls_gaussian,
        smooth_walls_morphological,
    )

    MAZE_POSTPROCESSING_AVAILABLE = True
except ImportError:
    MAZE_POSTPROCESSING_AVAILABLE = False
    adaptive_door_carving = None  # type: ignore
    enhance_organic_maze = None  # type: ignore
    normalize_wall_thickness = None  # type: ignore
    smooth_walls_combined = None  # type: ignore
    smooth_walls_gaussian = None  # type: ignore
    smooth_walls_morphological = None  # type: ignore

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

# Conditional import for continuous MFG base class (requires Gymnasium)
try:
    from mfg_pde.alg.reinforcement.environments.continuous_mfg_env_base import ContinuousMFGEnvBase

    CONTINUOUS_MFG_AVAILABLE = True
except ImportError:
    CONTINUOUS_MFG_AVAILABLE = False
    ContinuousMFGEnvBase = None  # type: ignore

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
    # Voronoi Diagram (room-based mazes)
    "VoronoiMazeConfig",
    "VoronoiMazeGenerator",
    # Hybrid Mazes (multi-algorithm combinations)
    "HybridMazeConfig",
    "HybridMazeGenerator",
    "HybridStrategy",
    "AlgorithmSpec",
    "create_museum_hybrid",
    "create_office_hybrid",
    "create_campus_hybrid",
    # Availability flags
    "MFG_ENV_AVAILABLE",
    "MAZE_UTILS_AVAILABLE",
    "MAZE_POSTPROCESSING_AVAILABLE",
    "CONTINUOUS_MFG_AVAILABLE",
]

# Add maze utilities exports if available
if MAZE_UTILS_AVAILABLE:
    __all__.extend(
        [
            "connect_regions_adaptive",
            "analyze_maze_connectivity",
            "find_disconnected_regions",
            "find_region_boundary",
            "compute_adaptive_door_width",
        ]
    )

# Add maze post-processing exports if available
if MAZE_POSTPROCESSING_AVAILABLE:
    __all__.extend(
        [
            "smooth_walls_morphological",
            "smooth_walls_gaussian",
            "smooth_walls_combined",
            "normalize_wall_thickness",
            "adaptive_door_carving",
            "enhance_organic_maze",
        ]
    )

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

# Add continuous MFG base class if available
if CONTINUOUS_MFG_AVAILABLE:
    __all__.extend(["ContinuousMFGEnvBase"])
