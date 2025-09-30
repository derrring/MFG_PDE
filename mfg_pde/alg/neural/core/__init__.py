"""
Core neural infrastructure for MFG methods.

This module provides shared components for all neural approaches:
- Network architectures (feed-forward, residual, modified MLPs)
- Loss functions (physics, boundary, data terms)
- Training utilities and optimization
- Neural network utilities and differentiation
"""

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
    # Import loss functions
    # Import network architectures from nn module
    from ..nn import (
        FeedForwardNetwork,
        ModifiedMLP,
        ResidualNetwork,
        create_mfg_networks,
    )
    from .loss_functions import (
        BoundaryLoss,
        DataLoss,
        MFGLossFunction,
        PhysicsLoss,
        PINNLossFunction,
    )

    # Import factory class from core networks
    from .networks import (
        NetworkArchitecture,
    )

    # Import training components
    from .training import (
        AdaptiveSampling,
        CurriculumLearning,
        OptimizationScheduler,
        TrainingManager,
    )

    # Import utilities
    from .utils import (
        auto_differentiation,
        compute_gradients,
        neural_network_utils,
        sample_points,
    )

    __all__ = [
        # Loss functions
        "BoundaryLoss",
        "DataLoss",
        "MFGLossFunction",
        "PINNLossFunction",
        "PhysicsLoss",
        # Network architectures
        "FeedForwardNetwork",
        "ModifiedMLP",
        "NetworkArchitecture",
        "ResidualNetwork",
        "create_mfg_networks",
        # Training components
        "AdaptiveSampling",
        "CurriculumLearning",
        "OptimizationScheduler",
        "TrainingManager",
        # Utilities
        "auto_differentiation",
        "compute_gradients",
        "neural_network_utils",
        "sample_points",
    ]
else:
    import warnings

    warnings.warn(
        "PyTorch is required for neural core components. "
        "Install with: pip install mfg_pde[neural] or pip install torch",
        ImportWarning,
    )

    __all__ = []

# Always export availability info
__all__.extend(["TORCH_AVAILABLE"])
