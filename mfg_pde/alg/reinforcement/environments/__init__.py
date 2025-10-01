"""
MFG-RL Environments

Environment infrastructure for reinforcement learning approaches to Mean Field Games.
Provides maze generators, MFG-specific environments, and utilities for RL experiments.

Modules:
- maze_generator: Perfect maze generation algorithms for RL experiments
"""

from mfg_pde.alg.reinforcement.environments.maze_generator import (
    Cell,
    Grid,
    MazeAlgorithm,
    PerfectMazeGenerator,
    generate_maze,
    verify_perfect_maze,
)

__all__ = [
    # Maze generation
    "MazeAlgorithm",
    "PerfectMazeGenerator",
    "Grid",
    "Cell",
    "generate_maze",
    "verify_perfect_maze",
]
