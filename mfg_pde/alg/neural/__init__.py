"""
Neural paradigm for MFG problems.

This module contains neural network-based approaches for solving Mean Field Games:
- pinn_solvers: Physics-Informed Neural Networks for MFG systems
- core: Shared neural network infrastructure and utilities

Neural methods use automatic differentiation and optimization to parameterize
PDE solutions as neural networks, providing flexible approaches for complex
geometries and high-dimensional problems.

Note: Neural solvers require PyTorch installation.
"""

from mfg_pde.alg.base_solver import BaseNeuralSolver

# Conditional imports based on PyTorch availability
try:
    import importlib.util

    torch_spec = importlib.util.find_spec("torch")
    if torch_spec is not None:
        TORCH_AVAILABLE = True
    else:
        raise ImportError("torch not available")
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    # Import PINN solvers
    # Import core neural components
    from .core import (
        BoundaryLoss,
        DataLoss,
        FeedForwardNetwork,
        MFGLossFunction,
        ModifiedMLP,
        NetworkArchitecture,
        PhysicsLoss,
        PINNLossFunction,
        ResidualNetwork,
        TrainingManager,
        auto_differentiation,
        compute_gradients,
        create_mfg_networks,
        neural_network_utils,
        sample_points,
    )
    from .pinn_solvers import (
        FPPINNSolver,
        HJBPINNSolver,
        MFGPINNSolver,
        PINNBase,
        PINNConfig,
    )

    __all__ = [
        "BaseNeuralSolver",
        # Core Neural Components
        "BoundaryLoss",
        "DataLoss",
        # PINN Solvers
        "FPPINNSolver",
        "FeedForwardNetwork",
        "HJBPINNSolver",
        "MFGLossFunction",
        "MFGPINNSolver",
        "ModifiedMLP",
        "NetworkArchitecture",
        "PINNBase",
        "PINNConfig",
        "PINNLossFunction",
        "PhysicsLoss",
        "ResidualNetwork",
        "TrainingManager",
        "auto_differentiation",
        "compute_gradients",
        "create_mfg_networks",
        "neural_network_utils",
        "sample_points",
    ]
else:
    import warnings

    warnings.warn(
        "PyTorch is required for neural network solvers. "
        "Install with: pip install mfg_pde[neural] or pip install torch",
        ImportWarning,
    )

    __all__ = [
        "BaseNeuralSolver",
    ]

# Always export availability info
__all__.extend(["TORCH_AVAILABLE"])
