"""
Neural network architectures for Physics-Informed Neural Networks (PINNs).

This module follows PyTorch's nn convention and provides specialized neural
network architectures optimized for solving Mean Field Games, including:

- Feed-forward networks with physics-aware initialization
- Residual networks for deep architectures
- Modified MLPs with skip connections
- Specialized architectures for HJB and Fokker-Planck equations

Note: Named 'nn' to follow PyTorch convention and avoid confusion with
network-based MFGs (graph-structured problems).
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
    from .feedforward import FeedForwardNetwork
    from .mfg_networks import create_mfg_networks
    from .modified_mlp import ModifiedMLP
    from .residual import ResidualNetwork

    __all__ = [
        "FeedForwardNetwork",
        "ModifiedMLP",
        "ResidualNetwork",
        "create_mfg_networks",
    ]

else:
    import warnings

    warnings.warn(
        "PyTorch is required for neural network architectures. Install with: pip install torch",
        ImportWarning,
    )

    __all__ = []

# Always export availability info
__all__.extend(["TORCH_AVAILABLE"])
