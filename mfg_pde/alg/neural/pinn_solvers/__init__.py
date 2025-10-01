"""
Physics-Informed Neural Network (PINN) solvers for MFG problems.

This module provides PINN-based approaches for solving Mean Field Games:
- Individual equation solvers (HJB, Fokker-Planck)
- Complete MFG system solvers
- Base classes and configuration

PINN solvers parameterize PDE solutions as neural networks and use
automatic differentiation to compute physics-based loss functions.
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
    # Import PINN solver implementations
    try:
        from .adaptive_training import AdaptiveTrainingConfig, AdaptiveTrainingStrategy, PhysicsGuidedSampler
        from .base_pinn import PINNBase, PINNConfig
        from .fp_pinn_solver import FPPINNSolver
        from .hjb_pinn_solver import HJBPINNSolver
        from .mfg_pinn_solver import MFGPINNSolver

        # Individual equation solvers
        INDIVIDUAL_PINN_SOLVERS = [
            "HJBPINNSolver",
            "FPPINNSolver",
        ]

        # Coupled system solvers
        COUPLED_PINN_SOLVERS = [
            "MFGPINNSolver",
        ]

        # All PINN solvers
        ALL_PINN_SOLVERS = INDIVIDUAL_PINN_SOLVERS + COUPLED_PINN_SOLVERS

        __all__ = [
            "AdaptiveTrainingConfig",
            "AdaptiveTrainingStrategy",
            "FPPINNSolver",
            "HJBPINNSolver",
            "MFGPINNSolver",
            "PINNBase",
            "PINNConfig",
            "PhysicsGuidedSampler",
        ]

    except ImportError as e:
        # PINN solvers not fully implemented yet
        import warnings

        warnings.warn(
            f"PINN solvers are currently under development: {e}. Use numerical or optimization paradigms for now.",
            UserWarning,
        )

        __all__ = []
        INDIVIDUAL_PINN_SOLVERS = []
        COUPLED_PINN_SOLVERS = []
        ALL_PINN_SOLVERS = []

else:
    import warnings

    warnings.warn(
        "PyTorch is required for PINN solvers. Install with: pip install mfg_pde[neural] or pip install torch",
        ImportWarning,
    )

    __all__ = []
    INDIVIDUAL_PINN_SOLVERS = []
    COUPLED_PINN_SOLVERS = []
    ALL_PINN_SOLVERS = []

# Always export availability info
__all__.extend(["TORCH_AVAILABLE"])
