"""
Neural Network Solvers for MFG_PDE.

This module provides Physics-Informed Neural Networks (PINN) and other neural
network-based solvers for Mean Field Games, integrated within the algorithm
framework of MFG_PDE.

Neural solvers represent a modern algorithmic approach to solving PDEs by:
1. Parameterizing solutions as neural networks
2. Using automatic differentiation for PDE residual computation
3. Optimizing network parameters to minimize physics-informed loss functions
4. Supporting complex geometries and high-dimensional problems

Key Components:
- PINNBase: Abstract base class for physics-informed neural networks
- MFGPINNSolver: Complete MFG system solver using coupled neural networks
- HJBPINNSolver: Hamilton-Jacobi-Bellman equation PINN solver
- FPPINNSolver: Fokker-Planck equation PINN solver
- NetworkArchitectures: Customizable neural network architectures
- TrainingStrategies: Advanced training methods and optimization

Integration with MFG_PDE:
- Follows same solver interface as other algorithms (FDM, FEM, etc.)
- Compatible with existing factory patterns and configuration system
- Supports standard MFGProblem interface and geometry management
- Integrates with visualization and analysis tools

Mathematical Foundation:
For MFG system:
- HJB: ∂u/∂t + H(∇u, x, m) = 0
- FP: ∂m/∂t - div(m∇H_p(∇u, x, m)) - σ²/2 Δm = 0

PINN Loss = PDE Residual + Boundary Conditions + Initial Conditions + Data Term
"""

from __future__ import annotations

import warnings

# PyTorch availability check
try:
    import torch
    import torch.cuda

    TORCH_AVAILABLE = True
    TORCH_VERSION = torch.__version__
    CUDA_AVAILABLE = torch.cuda.is_available()
    CUDA_VERSION = torch.version.cuda if CUDA_AVAILABLE else None
    GPU_COUNT = torch.cuda.device_count() if CUDA_AVAILABLE else 0
except ImportError:
    TORCH_AVAILABLE = False
    TORCH_VERSION = None
    CUDA_AVAILABLE = False
    CUDA_VERSION = None
    GPU_COUNT = 0


# Conditional imports - only import if PyTorch is available
if TORCH_AVAILABLE:
    # Core PINN components
    from .base_pinn import PINNBase, PINNConfig
    from .fp_pinn_solver import FPPINNSolver
    from .hjb_pinn_solver import HJBPINNSolver

    # Loss functions
    from .loss_functions import (
        BoundaryLoss,
        DataLoss,
        MFGLossFunction,
        PhysicsLoss,
        PINNLossFunction,
    )
    from .mfg_pinn_solver import MFGPINNSolver

    # Network architectures
    from .networks import (
        FeedForwardNetwork,
        ModifiedMLP,
        NetworkArchitecture,
        ResidualNetwork,
        create_mfg_networks,
    )

    # Training components
    from .training import (
        AdaptiveSampling,
        CurriculumLearning,
        OptimizationScheduler,
        TrainingManager,
    )

    # Utilities
    from .utils import (
        auto_differentiation,
        compute_gradients,
        neural_network_utils,
        sample_points,
    )

    __all__ = [
        "AdaptiveSampling",
        "BoundaryLoss",
        "CurriculumLearning",
        "DataLoss",
        "FPPINNSolver",
        "FeedForwardNetwork",
        "HJBPINNSolver",
        "MFGLossFunction",
        "MFGPINNSolver",
        "ModifiedMLP",
        "NetworkArchitecture",
        "OptimizationScheduler",
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
    # PyTorch not available - provide warning
    import warnings

    warnings.warn(
        "PyTorch is required for neural network solvers. " "Install with: pip install torch torchvision",
        ImportWarning,
    )

    __all__ = []

# Export availability information regardless of PyTorch availability
__all__.extend(["CUDA_AVAILABLE", "CUDA_VERSION", "GPU_COUNT", "TORCH_AVAILABLE", "TORCH_VERSION"])


def get_system_info() -> dict:
    """Get system information for neural network solvers."""
    info = {
        "torch_available": TORCH_AVAILABLE,
        "torch_version": TORCH_VERSION,
        "cuda_available": CUDA_AVAILABLE,
        "cuda_version": CUDA_VERSION,
        "gpu_count": GPU_COUNT,
    }

    return info


def print_system_info() -> None:
    """Print system information for neural network solvers."""
    info = get_system_info()

    print("=== Neural Network Solver System Information ===")
    print(f"PyTorch Available: {info['torch_available']}")
    if info["torch_available"]:
        print(f"PyTorch Version: {info['torch_version']}")
        print(f"CUDA Available: {info['cuda_available']}")
        if info["cuda_available"]:
            print(f"CUDA Version: {info['cuda_version']}")
            print(f"GPU Count: {info['gpu_count']}")
    else:
        print("Install PyTorch: pip install torch torchvision")
    print("=" * 50)


# Version information
__version__ = "1.0.0"

# Ensure these utility functions are always available
__all__.extend(["get_system_info", "print_system_info", "__version__"])
