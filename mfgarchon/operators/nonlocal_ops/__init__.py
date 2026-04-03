"""
Non-local operators for integro-differential equations.

This module provides operators for jump-diffusion and Lévy-driven MFG,
complementing the local differential operators in ``operators/differential/``.

- LevyIntegroDiffOperator: Non-local jump operator J[v] for HJB/FP
- LevyMeasure protocol + concrete implementations (Gaussian, compound Poisson)

Issue #923: Part of Layer 1 (Generalized PDE & Institutional MFG Plan).
"""

from .levy_integro_diff import LevyIntegroDiffOperator
from .levy_measures import CompoundPoissonJumps, GaussianJumps, LevyMeasure

__all__ = [
    "LevyIntegroDiffOperator",
    "LevyMeasure",
    "GaussianJumps",
    "CompoundPoissonJumps",
]
