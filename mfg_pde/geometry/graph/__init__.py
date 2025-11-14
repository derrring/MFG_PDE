"""
Graph-based geometries for MFG problems.

This module provides graph and network geometry support for MFG problems,
including network topologies and maze-based domains.
"""

# Network geometry
# Maze generation
from .cellular_automata import CellularAutomataConfig, CellularAutomataGenerator
from .hybrid_maze import HybridMazeGenerator
from .maze_config import MazeConfig
from .maze_generator import MazeAlgorithm, PerfectMazeGenerator
from .network import (
    BaseNetworkGeometry,
    GridNetwork,
    NetworkData,
    NetworkType,
    RandomNetwork,
    ScaleFreeNetwork,
    compute_network_statistics,
    create_network,
)
from .network_backend import (
    NetworkBackendType,
    OperationType,
    get_backend_manager,
    set_preferred_backend,
)
from .recursive_division import RecursiveDivisionConfig, RecursiveDivisionGenerator
from .voronoi_maze import VoronoiMazeGenerator

# Add maze_ prefixes for clarity
maze_PerfectMazeGenerator = PerfectMazeGenerator
maze_Algorithm = MazeAlgorithm
maze_Config = MazeConfig
maze_CellularAutomataConfig = CellularAutomataConfig
maze_CellularAutomataGenerator = CellularAutomataGenerator
maze_HybridGenerator = HybridMazeGenerator
maze_RecursiveDivisionConfig = RecursiveDivisionConfig
maze_RecursiveDivisionGenerator = RecursiveDivisionGenerator
maze_VoronoiGenerator = VoronoiMazeGenerator

__all__ = [
    # Network geometry
    "NetworkType",
    "NetworkData",
    "BaseNetworkGeometry",
    "GridNetwork",
    "RandomNetwork",
    "ScaleFreeNetwork",
    "compute_network_statistics",
    "create_network",
    # Network backend
    "NetworkBackendType",
    "OperationType",
    "get_backend_manager",
    "set_preferred_backend",
    # Maze generation (with maze_ prefix)
    "maze_PerfectMazeGenerator",
    "maze_Algorithm",
    "maze_Config",
    "maze_CellularAutomataConfig",
    "maze_CellularAutomataGenerator",
    "maze_HybridGenerator",
    "maze_RecursiveDivisionConfig",
    "maze_RecursiveDivisionGenerator",
    "maze_VoronoiGenerator",
    # Original names (for backward compatibility)
    "PerfectMazeGenerator",
    "MazeAlgorithm",
    "MazeConfig",
    "CellularAutomataConfig",
    "CellularAutomataGenerator",
    "HybridMazeGenerator",
    "RecursiveDivisionConfig",
    "RecursiveDivisionGenerator",
    "VoronoiMazeGenerator",
]
