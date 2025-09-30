"""
Physics-Informed Neural Networks (PINNs) for MFG problems.

This module implements advanced PINN methods specifically designed for
Mean Field Games, building on the neural paradigm foundation with
sophisticated physics-informed learning techniques.

Key Features:
- Physics-Informed Loss Functions: Automatic differentiation for PDE residuals
- Multi-Task Learning: Simultaneous learning of u(t,x) and m(t,x)
- Adaptive Sampling: Physics-guided point selection and refinement
- Transfer Learning: Pre-trained models for related MFG problems
- Uncertainty Quantification: Bayesian PINNs with uncertainty estimates

Mathematical Framework:
- Physics Loss: L_physics = ||∂u/∂t + H(∇u,m)||² + ||∂m/∂t - div(m∇H_p)||²
- Boundary Loss: L_BC = ||u - u_BC||² + ||m - m_BC||²
- Initial Loss: L_IC = ||u(0,x) - u₀(x)||² + ||m(0,x) - m₀(x)||²
- Total Loss: L = w₁L_physics + w₂L_BC + w₃L_IC

Components:
- base_pinn: Abstract PINN solver framework
- mfg_pinn_solver: Complete MFG PINN implementation
- adaptive_training: Physics-guided adaptive training strategies
- uncertainty: Bayesian PINNs and uncertainty quantification
- transfer_learning: Pre-trained models and transfer learning utilities
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Check PyTorch availability
try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
    logger.debug("PyTorch available for PINN methods")

except ImportError:
    TORCH_AVAILABLE = False
    logger.info("PyTorch not available - PINN methods unavailable")


if TORCH_AVAILABLE:
    from .adaptive_training import AdaptiveTrainingStrategy, PhysicsGuidedSampler
    from .base_pinn import BasePINNSolver, PINNConfig, PINNResult
    from .mfg_pinn_solver import MFGPINNSolver

    __all__ = [
        "AdaptiveTrainingStrategy",
        "BasePINNSolver",
        "MFGPINNSolver",
        "PINNConfig",
        "PINNResult",
        "PhysicsGuidedSampler",
    ]

else:
    # Provide informative error classes when PyTorch unavailable
    class BasePINNSolver:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for PINN methods. Install with: pip install torch")

    class MFGPINNSolver:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for PINN methods. Install with: pip install torch")

    __all__ = ["BasePINNSolver", "MFGPINNSolver"]


# Version information
__version__ = "1.0.0"
__author__ = "MFG_PDE Development Team"
