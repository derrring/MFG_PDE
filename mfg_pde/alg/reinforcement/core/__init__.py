"""
Core infrastructure for Reinforcement Learning paradigm in MFG.

This module provides the foundational components for MFG-RL solvers:
- Base classes for MFRL solvers
- MFG environment wrappers
- Population state representations
- Training utilities and configuration

The core infrastructure enables the development of agent-based approaches
to Mean Field Games using modern reinforcement learning techniques.
"""

from mfg_pde.alg.base_solver import BaseRLSolver

# Check for RL dependencies at core level
try:
    import gymnasium as gym

    import numpy as np

    CORE_DEPENDENCIES_AVAILABLE = True
except ImportError:
    CORE_DEPENDENCIES_AVAILABLE = False

if CORE_DEPENDENCIES_AVAILABLE:
    from .base_mfrl import (
        BaseMFRLSolver,
        RLSolverConfig,
        RLSolverResult,
    )
    from .environments import (
        ContinuousMFGEnv,
        MFGEnvironment,
        NetworkMFGEnv,
    )
    from .population_state import (
        PopulationMetrics,
        PopulationState,
        PopulationTracker,
    )
    from .training_loops import (
        MFRLTrainingLoop,
        PopulationTrainingManager,
    )

    __all__ = [
        "BaseRLSolver",
        # Base MFRL Components
        "BaseMFRLSolver",
        "RLSolverConfig",
        "RLSolverResult",
        # Environment Components
        "MFGEnvironment",
        "ContinuousMFGEnv",
        "NetworkMFGEnv",
        # Population State Management
        "PopulationState",
        "PopulationTracker",
        "PopulationMetrics",
        # Training Infrastructure
        "MFRLTrainingLoop",
        "PopulationTrainingManager",
    ]

    # Component categories
    BASE_COMPONENTS = ["BaseMFRLSolver", "RLSolverConfig", "RLSolverResult"]
    ENVIRONMENT_COMPONENTS = ["MFGEnvironment", "ContinuousMFGEnv", "NetworkMFGEnv"]
    POPULATION_COMPONENTS = ["PopulationState", "PopulationTracker", "PopulationMetrics"]
    TRAINING_COMPONENTS = ["MFRLTrainingLoop", "PopulationTrainingManager"]

else:
    import warnings

    warnings.warn(
        "RL core infrastructure requires gymnasium and numpy. " "Install with: pip install mfg_pde[rl]",
        ImportWarning,
    )

    __all__ = [
        "BaseRLSolver",
    ]

    # Empty component categories
    BASE_COMPONENTS = []
    ENVIRONMENT_COMPONENTS = []
    POPULATION_COMPONENTS = []
    TRAINING_COMPONENTS = []

# Always export availability info
__all__.extend(["CORE_DEPENDENCIES_AVAILABLE"])
