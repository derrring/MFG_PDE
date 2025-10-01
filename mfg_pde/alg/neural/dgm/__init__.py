"""
Deep Galerkin Methods (DGM) for high-dimensional MFG problems.

This module implements Deep Galerkin Methods adapted for Mean Field Games,
enabling efficient solution of MFG systems in high dimensions (d > 5).

DGM reformulates PDE systems as optimization problems over neural network
function approximations, using Monte Carlo sampling for high-dimensional
integration and variance reduction techniques for computational efficiency.

Key Components:
- base_dgm: Base DGM solver with Monte Carlo sampling
- mfg_dgm_solver: MFG-specific DGM implementation
- sampling: High-dimensional sampling strategies
- variance_reduction: Monte Carlo variance reduction techniques
- architectures: Deep neural network architectures for DGM

Mathematical Framework:
- Replace PDE grid methods with neural function approximation
- Use Monte Carlo sampling for high-dimensional integration
- Minimize physics residuals over neural network parameters
- Enable solution in dimensions d = 5, 10, 15, 20+

Note: DGM requires PyTorch and is recommended for d > 5 dimensional problems.
"""

from mfg_pde.alg.base_solver import BaseNeuralSolver

# Check for DGM dependencies (same as neural paradigm)
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
    # Import DGM solver implementations
    try:
        # Deep network architectures for DGM
        from .architectures import (
            DeepGalerkinNetwork,
            HighDimMLP,
            ResidualDGMNetwork,
        )
        from .base_dgm import BaseDGMSolver, DGMConfig, DGMResult
        from .mfg_dgm_solver import MFGDGMSolver

        # High-dimensional sampling utilities
        from .sampling import (
            HighDimSampler,
            MonteCarloSampler,
            QuasiMonteCarloSampler,
            adaptive_sampling,
        )

        # Variance reduction techniques
        from .variance_reduction import (
            ControlVariates,
            ImportanceSampling,
            MultilevelMonteCarlo,
        )

        __all__ = [
            # Core DGM Framework
            "BaseDGMSolver",
            "BaseNeuralSolver",
            # Variance Reduction
            "ControlVariates",
            "DGMConfig",
            "DGMResult",
            # DGM Architectures
            "DeepGalerkinNetwork",
            "HighDimMLP",
            # High-Dimensional Sampling
            "HighDimSampler",
            "ImportanceSampling",
            "MFGDGMSolver",
            "MonteCarloSampler",
            "MultilevelMonteCarlo",
            "QuasiMonteCarloSampler",
            "ResidualDGMNetwork",
            "adaptive_sampling",
        ]

        # Solver categories for factory selection
        DGM_SOLVERS = ["MFGDGMSolver"]
        HIGH_DIM_SOLVERS = DGM_SOLVERS  # For d > 5 problems

    except ImportError as e:
        # DGM components not yet fully implemented
        import warnings

        warnings.warn(
            f"DGM solvers are currently under development: {e}. "
            f"Basic PINN solvers are available in the neural paradigm.",
            UserWarning,
        )

        __all__ = [
            "BaseNeuralSolver",
        ]

        DGM_SOLVERS = []
        HIGH_DIM_SOLVERS = []

else:
    import warnings

    warnings.warn(
        "PyTorch is required for DGM solvers. Install with: pip install mfg_pde[neural] or pip install torch",
        ImportWarning,
    )

    __all__ = [
        "BaseNeuralSolver",
    ]

    DGM_SOLVERS = []
    HIGH_DIM_SOLVERS = []

# Always export availability info
__all__.extend(["TORCH_AVAILABLE"])
