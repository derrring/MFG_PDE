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
    import gymnasium as gym  # noqa: F401

    import numpy as np  # noqa: F401

    CORE_DEPENDENCIES_AVAILABLE = True
except ImportError:
    CORE_DEPENDENCIES_AVAILABLE = False

if CORE_DEPENDENCIES_AVAILABLE:
    from .base_mfrl import (
        BaseMFRLSolver,
        RLSolverConfig,
        RLSolverResult,
    )

    try:  # pragma: no cover - optional module
        from .environments import (
            ContinuousMFGEnv,
            MFGEnvironment,
            NetworkMFGEnv,
        )
    except ImportError:
        ContinuousMFGEnv = None  # type: ignore[assignment]
        MFGEnvironment = None  # type: ignore[assignment]
        NetworkMFGEnv = None  # type: ignore[assignment]

    try:  # pragma: no cover - optional module
        from .population_state import (
            PopulationMetrics,
            PopulationState,
            PopulationTracker,
        )
    except ImportError:
        PopulationMetrics = None  # type: ignore[assignment]
        PopulationState = None  # type: ignore[assignment]
        PopulationTracker = None  # type: ignore[assignment]

    try:  # pragma: no cover - optional module
        from .training_loops import (
            MFRLTrainingLoop,
            PopulationTrainingManager,
        )
    except ImportError:
        MFRLTrainingLoop = None  # type: ignore[assignment]
        PopulationTrainingManager = None  # type: ignore[assignment]

    __all__ = [
        # Base MFRL Components
        "BaseMFRLSolver",
        "BaseRLSolver",
        "ContinuousMFGEnv",
        # Environment Components
        "MFGEnvironment",
        # Training Infrastructure
        "MFRLTrainingLoop",
        "NetworkMFGEnv",
        "PopulationMetrics",
        # Population State Management
        "PopulationState",
        "PopulationTracker",
        "PopulationTrainingManager",
        "RLSolverConfig",
        "RLSolverResult",
    ]

    # Component categories
    BASE_COMPONENTS = ["BaseMFRLSolver", "RLSolverConfig", "RLSolverResult"]
    ENVIRONMENT_COMPONENTS = [
        name for name in ("MFGEnvironment", "ContinuousMFGEnv", "NetworkMFGEnv") if globals().get(name) is not None
    ]
    POPULATION_COMPONENTS = [
        name
        for name in ("PopulationState", "PopulationTracker", "PopulationMetrics")
        if globals().get(name) is not None
    ]
    TRAINING_COMPONENTS = [
        name for name in ("MFRLTrainingLoop", "PopulationTrainingManager") if globals().get(name) is not None
    ]

else:
    import warnings

    warnings.warn(
        "RL core infrastructure requires gymnasium and numpy. Install with: pip install mfg_pde[rl]",
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
