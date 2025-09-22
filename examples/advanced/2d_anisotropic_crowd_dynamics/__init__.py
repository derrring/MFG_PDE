"""
2D Anisotropic Crowd Dynamics Experiment Package

This package provides a complete implementation of the 2D Mean Field Games
experiment with non-separable Hamiltonians for crowd evacuation scenarios.

The experiment demonstrates how spatial coupling between movement directions
creates complex evacuation patterns that cannot be captured by separable
formulations, enhanced with realistic architectural barriers.

Main Components:
- anisotropic_2d_problem: Core 2D MFG problem implementation with barriers
- solver_config: Optimized solver configurations
- experiment_runner: Main experiment orchestration
- numerical_demo: Numerically stable implementation
- validation/: Convergence studies and validation
- analysis/: Comprehensive visualization tools

Usage:
    python experiment_runner.py --mode single --barrier-config anisotropy_aligned
    python experiment_runner.py --mode comparative
    python experiment_runner.py --mode convergence
    python numerical_demo.py   # Numerically stable demo
"""

from .anisotropic_2d_problem import (
    AnisotropicMFGProblem2D,
    CircularBarrier,
    LinearBarrier,
    RectangularBarrier,
    create_anisotropic_problem,
)
from .solver_config import (
    create_anisotropic_solver_config,
    create_experiment_solver,
    create_performance_optimized_config,
)

__version__ = "1.0.0"
__author__ = "MFG_PDE Team"
__description__ = "2D Anisotropic Crowd Dynamics with Barriers"

__all__ = [
    # Main classes
    "AnisotropicMFGProblem2D",
    "create_anisotropic_problem",
    # Barrier types
    "CircularBarrier",
    "LinearBarrier",
    "RectangularBarrier",
    # Solver configuration
    "create_anisotropic_solver_config",
    "create_experiment_solver",
    "create_performance_optimized_config",
    # Package info
    "__version__",
    "__author__",
    "__description__",
]
