"""
Abstract Iteration Patterns.

Provides Protocol classes for common iteration patterns used across
different solving paradigms:

- PicardPattern: Fixed-point iteration with constant damping
- AveragingPattern: Fictitious Play with decaying learning rate
- NewtonPattern: Newton's method with quadratic convergence

These patterns are algorithm-agnostic and define the iteration structure
without coupling to specific problem types (PDE, RL, optimization).

Implementations:
- numerical/coupling/: PDE system coupling (HJB-FP)
- neural/: PINN-based methods (future)
- reinforcement/: Policy iteration (future)
"""

from .averaging import AveragingIterator, AveragingPattern
from .picard import FixedPointIteratorBase, PicardPattern

__all__ = [
    # Protocols
    "PicardPattern",
    "AveragingPattern",
    # Base classes
    "FixedPointIteratorBase",
    "AveragingIterator",
]
