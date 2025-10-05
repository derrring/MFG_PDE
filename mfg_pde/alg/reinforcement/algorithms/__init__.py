"""
Reinforcement Learning Algorithms for Mean Field Games.

This module contains RL algorithms specifically designed for MFG problems,
where agents learn policies and value functions that depend on both individual
state and population state (mean field).

Available Algorithms:

Single-Population Algorithms:

Discrete Action Spaces:
- MeanFieldQLearning: Value-based learning with Q(s,a,m)
- MeanFieldActorCritic: Policy gradient with actor-critic architecture

Continuous Action Spaces:
- MeanFieldDDPG: Deep Deterministic Policy Gradient for continuous control
- MeanFieldTD3: Twin Delayed DDPG with target policy smoothing
- MeanFieldSAC: Soft Actor-Critic with maximum entropy objective

Multi-Population Algorithms (N ≥ 2 interacting populations):

Continuous Action Spaces:
- MultiPopulationDDPG: DDPG for heterogeneous populations
- MultiPopulationTD3: TD3 for heterogeneous populations with twin critics
- MultiPopulationSAC: SAC for heterogeneous populations with entropy regularization
"""

from __future__ import annotations

# Check for PyTorch availability
try:
    import torch  # noqa: F401

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    # Single-population algorithms
    from .mean_field_actor_critic import (
        ActorNetwork,
        CriticNetwork,
        MeanFieldActorCritic,
    )
    from .mean_field_ddpg import MeanFieldDDPG
    from .mean_field_q_learning import MeanFieldQLearning, MeanFieldQNetwork
    from .mean_field_sac import MeanFieldSAC
    from .mean_field_td3 import MeanFieldTD3

    # Multi-population algorithms
    from .multi_population_ddpg import MultiPopulationDDPG
    from .multi_population_sac import MultiPopulationSAC
    from .multi_population_td3 import MultiPopulationTD3

    __all__ = [
        "ActorNetwork",
        "CriticNetwork",
        "MeanFieldActorCritic",
        "MeanFieldDDPG",
        # Single-population
        "MeanFieldQLearning",
        "MeanFieldQNetwork",
        "MeanFieldSAC",
        "MeanFieldTD3",
        # Multi-population
        "MultiPopulationDDPG",
        "MultiPopulationSAC",
        "MultiPopulationTD3",
    ]
else:
    __all__ = []
