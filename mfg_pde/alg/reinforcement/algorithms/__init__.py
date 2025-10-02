"""
Reinforcement Learning Algorithms for Mean Field Games.

This module contains RL algorithms specifically designed for MFG problems,
where agents learn policies and value functions that depend on both individual
state and population state (mean field).

Available Algorithms:
- MeanFieldQLearning: Value-based learning with Q(s,a,m)
- MeanFieldActorCritic: Policy gradient with actor-critic architecture
"""

from __future__ import annotations

# Check for PyTorch availability
try:
    import torch  # noqa: F401

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    from .mean_field_actor_critic import (
        ActorNetwork,
        CriticNetwork,
        MeanFieldActorCritic,
    )
    from .mean_field_q_learning import MeanFieldQLearning, MeanFieldQNetwork

    __all__ = [
        "MeanFieldQLearning",
        "MeanFieldQNetwork",
        "MeanFieldActorCritic",
        "ActorNetwork",
        "CriticNetwork",
    ]
else:
    __all__ = []
