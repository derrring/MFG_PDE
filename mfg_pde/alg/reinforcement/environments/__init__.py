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
    analyze_maze_connectivity = None
    compute_adaptive_door_width = None
    connect_regions_adaptive = None
    find_disconnected_regions = None
    find_region_boundary = None

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
    adaptive_door_carving = None
    enhance_organic_maze = None
    normalize_wall_thickness = None
    smooth_walls_combined = None
    smooth_walls_gaussian = None
    smooth_walls_morphological = None

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

# Conditional import for LQ-MFG environment (requires Gymnasium)
try:
    from mfg_pde.alg.reinforcement.environments.lq_mfg_env import LQMFGEnv

    LQ_MFG_AVAILABLE = True
except ImportError:
    LQ_MFG_AVAILABLE = False
    LQMFGEnv = None  # type: ignore

# Conditional import for Crowd Navigation environment (requires Gymnasium)
try:
    from mfg_pde.alg.reinforcement.environments.crowd_navigation_env import CrowdNavigationEnv

    CROWD_NAV_AVAILABLE = True
except ImportError:
    CROWD_NAV_AVAILABLE = False
    CrowdNavigationEnv = None  # type: ignore

# Conditional import for Price Formation environment (requires Gymnasium)
try:
    from mfg_pde.alg.reinforcement.environments.price_formation_env import PriceFormationEnv

    PRICE_FORMATION_AVAILABLE = True
except ImportError:
    PRICE_FORMATION_AVAILABLE = False
    PriceFormationEnv = None  # type: ignore

# Conditional import for Resource Allocation environment (requires Gymnasium)
try:
    from mfg_pde.alg.reinforcement.environments.resource_allocation_env import ResourceAllocationEnv

    RESOURCE_ALLOCATION_AVAILABLE = True
except ImportError:
    RESOURCE_ALLOCATION_AVAILABLE = False
    ResourceAllocationEnv = None  # type: ignore

# Conditional import for Traffic Flow environment (requires Gymnasium)
try:
    from mfg_pde.alg.reinforcement.environments.traffic_flow_env import TrafficFlowEnv

    TRAFFIC_FLOW_AVAILABLE = True
except ImportError:
    TRAFFIC_FLOW_AVAILABLE = False
    TrafficFlowEnv = None  # type: ignore

__all__ = [
    "CONTINUOUS_MFG_AVAILABLE",
    "CROWD_NAV_AVAILABLE",
    "LQ_MFG_AVAILABLE",
    "MAZE_POSTPROCESSING_AVAILABLE",
    "MAZE_UTILS_AVAILABLE",
    # Availability flags
    "MFG_ENV_AVAILABLE",
    "PRICE_FORMATION_AVAILABLE",
    "RESOURCE_ALLOCATION_AVAILABLE",
    "TRAFFIC_FLOW_AVAILABLE",
    "AlgorithmSpec",
    "Cell",
    # Cellular Automata (organic mazes)
    "CellularAutomataConfig",
    "CellularAutomataGenerator",
    "Grid",
    # Hybrid Mazes (multi-algorithm combinations)
    "HybridMazeConfig",
    "HybridMazeGenerator",
    "HybridStrategy",
    # Perfect maze generation
    "MazeAlgorithm",
    # Configuration
    "MazeConfig",
    "MazeTopology",
    "PerfectMazeGenerator",
    "PhysicalDimensions",
    "PlacementStrategy",
    # Recursive Division (variable-width mazes)
    "RecursiveDivisionConfig",
    "RecursiveDivisionGenerator",
    "SplitOrientation",
    # Voronoi Diagram (room-based mazes)
    "VoronoiMazeConfig",
    "VoronoiMazeGenerator",
    "add_loops",
    "compute_position_metrics",
    "create_campus_hybrid",
    "create_continuous_maze_config",
    "create_default_config",
    "create_multi_goal_config",
    "create_museum_hybrid",
    "create_office_hybrid",
    "create_preset_ca_config",
    "create_room_based_config",
    "generate_maze",
    # Position placement
    "place_positions",
    "verify_perfect_maze",
]

# Add maze utilities exports if available
if MAZE_UTILS_AVAILABLE:
    __all__.extend(
        [
            "analyze_maze_connectivity",
            "compute_adaptive_door_width",
            "connect_regions_adaptive",
            "find_disconnected_regions",
            "find_region_boundary",
        ]
    )

# Add maze post-processing exports if available
if MAZE_POSTPROCESSING_AVAILABLE:
    __all__.extend(
        [
            "adaptive_door_carving",
            "enhance_organic_maze",
            "normalize_wall_thickness",
            "smooth_walls_combined",
            "smooth_walls_gaussian",
            "smooth_walls_morphological",
        ]
    )

# Add MFG environment exports if available
if MFG_ENV_AVAILABLE:
    __all__.extend(
        [
            "ActionType",
            "MFGMazeConfig",
            "MFGMazeEnvironment",
            "PopulationState",
            "RewardType",
        ]
    )

# Add continuous MFG base class if available
if CONTINUOUS_MFG_AVAILABLE:
    __all__.extend(["ContinuousMFGEnvBase"])

# Add LQ-MFG environment if available
if LQ_MFG_AVAILABLE:
    __all__.extend(["LQMFGEnv"])

# Add Crowd Navigation environment if available
if CROWD_NAV_AVAILABLE:
    __all__.extend(["CrowdNavigationEnv"])

# Add Price Formation environment if available
if PRICE_FORMATION_AVAILABLE:
    __all__.extend(["PriceFormationEnv"])

# Add Resource Allocation environment if available
if RESOURCE_ALLOCATION_AVAILABLE:
    __all__.extend(["ResourceAllocationEnv"])

# Add Traffic Flow environment if available
if TRAFFIC_FLOW_AVAILABLE:
    __all__.extend(["TrafficFlowEnv"])
