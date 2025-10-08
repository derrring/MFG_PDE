"""
Backend-aware strategy selection for algorithm implementations.

This module provides intelligent strategy selection based on backend capabilities,
problem size, and runtime characteristics. Strategies enable automatic selection
of optimal implementation approaches (CPU, GPU, hybrid) without solver code
needing to know about backend details.

Key Components
--------------
ParticleStrategy : Abstract base class
    Strategy interface for particle-based FP solvers
CPUParticleStrategy : Concrete strategy
    NumPy + scipy implementation
GPUParticleStrategy : Concrete strategy
    GPU-accelerated implementation with internal KDE
HybridParticleStrategy : Concrete strategy
    Selective GPU usage for medium problems
StrategySelector : Selection logic
    Intelligent automatic strategy selection
AdaptiveStrategySelector : Advanced selector
    Runtime learning and adaptation (future)

Examples
--------
Automatic strategy selection:

>>> from mfg_pde.backends import create_backend
>>> from mfg_pde.backends.strategies import StrategySelector
>>> backend = create_backend("torch", device="mps")
>>> selector = StrategySelector()
>>> strategy = selector.select_strategy(
...     backend,
...     problem_size=(50000, 50, 50),  # (N, Nx, Nt)
...     strategy_hint="auto"
... )
>>> print(strategy.name)
'gpu-mps'

Manual override:

>>> strategy = selector.select_strategy(
...     backend,
...     problem_size=(5000, 50, 50),
...     strategy_hint="cpu"  # Force CPU even with GPU available
... )
>>> print(strategy.name)
'cpu'
"""

from .particle_strategies import (
    CPUParticleStrategy,
    GPUParticleStrategy,
    HybridParticleStrategy,
    ParticleStrategy,
)
from .strategy_selector import AdaptiveStrategySelector, StrategySelector

__all__ = [
    "AdaptiveStrategySelector",
    "CPUParticleStrategy",
    "GPUParticleStrategy",
    "HybridParticleStrategy",
    "ParticleStrategy",
    "StrategySelector",
]
