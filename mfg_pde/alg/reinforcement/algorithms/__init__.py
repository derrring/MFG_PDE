"""
Reinforcement Learning Algorithms for Mean Field Games.

This module contains RL algorithms specifically designed for MFG problems,
where agents learn policies and value functions that depend on both individual
state and population state (mean field).

Available Algorithms:

Discrete Action Spaces:
- MeanFieldQLearning: Value-based learning with Q(s,a,m)
- MeanFieldActorCritic: Policy gradient with actor-critic architecture

Continuous Action Spaces:
- MeanFieldDDPG: Deep Deterministic Policy Gradient for continuous control
- MeanFieldTD3: Twin Delayed DDPG with target policy smoothing
- MeanFieldSAC: Soft Actor-Critic with maximum entropy objective
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
    from .mean_field_ddpg import MeanFieldDDPG
    from .mean_field_q_learning import MeanFieldQLearning, MeanFieldQNetwork
    from .mean_field_sac import MeanFieldSAC
    from .mean_field_td3 import MeanFieldTD3

    __all__ = [
        "MeanFieldQLearning",
        "MeanFieldQNetwork",
        "MeanFieldActorCritic",
        "ActorNetwork",
        "CriticNetwork",
        "MeanFieldDDPG",
        "MeanFieldTD3",
        "MeanFieldSAC",
    ]
else:
    __all__ = []
