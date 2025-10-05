"""
Multi-Population Mean Field Games with Continuous Control.

This module implements heterogeneous multi-population MFG systems where:
- 2-5 populations interact through coupled mean field distributions
- Each population can have different state/action dimensions
- Populations can use different algorithms (DDPG, TD3, SAC)
- Nash equilibrium is achieved through simultaneous training

Components:
- population_config: Configuration for individual populations
- base_environment: Multi-population MFG environment base
- networks: Joint encoders and multi-population actor/critic
- multi_ddpg: Multi-population DDPG implementation
- multi_td3: Multi-population TD3 implementation
- multi_sac: Multi-population SAC implementation
- trainer: Training orchestrator for multi-population systems

Author: MFG_PDE Team
Date: October 2025
"""

from __future__ import annotations

# Base environment
from .base_environment import MultiPopulationMFGEnvironment, SimpleMultiPopulationEnv

# Algorithms
from .multi_ddpg import MultiPopulationDDPG
from .multi_sac import MultiPopulationSAC
from .multi_td3 import MultiPopulationTD3

# Neural networks
from .networks import (
    JointPopulationEncoder,
    MultiPopulationActor,
    MultiPopulationCritic,
    MultiPopulationStochasticActor,
)

# Population configuration
from .population_config import (
    PopulationConfig,
    create_asymmetric_coupling,
    create_symmetric_coupling,
    validate_population_set,
)

# Training
from .trainer import MultiPopulationTrainer

__all__ = [
    # Networks
    "JointPopulationEncoder",
    "MultiPopulationActor",
    "MultiPopulationCritic",
    # Algorithms
    "MultiPopulationDDPG",
    # Environment
    "MultiPopulationMFGEnvironment",
    "MultiPopulationSAC",
    "MultiPopulationStochasticActor",
    "MultiPopulationTD3",
    # Training
    "MultiPopulationTrainer",
    # Configuration
    "PopulationConfig",
    "SimpleMultiPopulationEnv",
    "create_asymmetric_coupling",
    "create_symmetric_coupling",
    "validate_population_set",
]
