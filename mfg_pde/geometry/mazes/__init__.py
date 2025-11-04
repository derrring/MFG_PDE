"""
Maze generation and integration for MFG geometry.

This module provides maze generation algorithms that can be used to create
structured domains with obstacles for MFG problems. Maze-based geometries are
useful for modeling:
- Crowd dynamics in complex buildings
- Evacuation scenarios with corridors
- Multi-zone heterogeneous environments
- Realistic path planning problems

The maze generators can be used with all solver types (classical PDE, particle
methods, neural networks, and reinforcement learning).

Examples
--------
>>> from mfg_pde.geometry.mazes import MazeGenerator, MazeConfig
>>> config = MazeConfig(width=20, height=20, algorithm="recursive_backtracking")
>>> maze = MazeGenerator.generate(config)
>>> # Use maze to define obstacles in domain
"""

from .cellular_automata import CellularAutomataConfig, CellularAutomataGenerator
from .hybrid_maze import HybridMazeGenerator
from .maze_config import MazeConfig
from .maze_generator import MazeAlgorithm, PerfectMazeGenerator
from .recursive_division import RecursiveDivisionConfig, RecursiveDivisionGenerator
from .voronoi_maze import VoronoiMazeGenerator

# Utility and postprocessing modules available but not imported by default
# from .maze_utils import analyze_maze_connectivity, find_disconnected_regions, etc.
# from .maze_postprocessing import smooth_walls_morphological, etc.

__all__ = [
    # Core generation
    "PerfectMazeGenerator",
    "MazeAlgorithm",
    "MazeConfig",
    # Specialized generators
    "CellularAutomataConfig",
    "CellularAutomataGenerator",
    "HybridMazeGenerator",
    "RecursiveDivisionConfig",
    "RecursiveDivisionGenerator",
    "VoronoiMazeGenerator",
]
