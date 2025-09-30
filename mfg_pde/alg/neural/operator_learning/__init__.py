"""
Neural Operator Learning for MFG Problems.

This module implements neural operator methods for learning parameter-to-solution
mappings in Mean Field Games. These methods enable rapid parameter studies and
real-time control applications by learning operators that map problem parameters
directly to solutions.

Key Operator Types:
- Fourier Neural Operators (FNO): Spectral methods for parameter-to-solution mapping
- DeepONet: Deep operator networks for operator learning
- Multi-Fidelity Operators: Combine low/high fidelity data for efficient learning

Mathematical Framework:
- Operator Learning: G: P -> U where P is parameter space, U is solution space
- Parameter-to-Solution: Given parameters theta, rapidly compute u(theta) without solving PDE
- Real-Time Applications: Enable control and optimization with fast evaluation

Research Applications:
- Parameter Studies: Explore parameter space 100x faster than traditional methods
- Uncertainty Quantification: Monte Carlo over parameter distributions
- Real-Time Control: Dynamic MFG applications with fast response requirements
- Multi-Query Optimization: Gradient-based optimization with fast evaluations
"""

from __future__ import annotations

from typing import Any

# Import with availability checking
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
    from torch.fft import fft, ifft, irfft, rfft

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    from .base_operator import BaseNeuralOperator, OperatorConfig, OperatorResult
    from .deeponet import DeepONet, DeepONetConfig
    from .fourier_neural_operator import FNOConfig, FourierNeuralOperator
    from .operator_training import OperatorDataset, OperatorTrainingManager, TrainingConfig

    __all__ = [
        # Base Classes
        "BaseNeuralOperator",
        "OperatorConfig",
        "OperatorResult",
        # Fourier Neural Operator
        "FourierNeuralOperator",
        "FNOConfig",
        # DeepONet
        "DeepONet",
        "DeepONetConfig",
        # Training Infrastructure
        "OperatorTrainingManager",
        "OperatorDataset",
        "TrainingConfig",
        # Utility Functions
        "create_mfg_operator",
        "TORCH_AVAILABLE",
    ]

    def create_mfg_operator(operator_type: str, config: dict[str, Any]) -> Any:
        """
        Factory function for creating MFG neural operators.

        Args:
            operator_type: Type of operator ("fno" or "deeponet")
            config: Configuration dictionary for the operator

        Returns:
            Configured neural operator instance

        Raises:
            ImportError: If PyTorch is not available
            ValueError: If operator_type is not recognized
        """
        if not TORCH_AVAILABLE:
            raise ImportError("Neural operators require PyTorch. Install with: pip install mfg_pde[neural]")

        operator_type = operator_type.lower()

        if operator_type == "fno":
            operator_config = FNOConfig(**config)
            return FourierNeuralOperator(operator_config)
        elif operator_type == "deeponet":
            operator_config = DeepONetConfig(**config)
            return DeepONet(operator_config)
        else:
            raise ValueError(f"Unknown operator type: {operator_type}. Supported types: 'fno', 'deeponet'")

else:
    # Placeholder functions when PyTorch is not available
    def create_mfg_operator(*args, **kwargs):
        raise ImportError("Neural operators require PyTorch. Install with: pip install mfg_pde[neural]")

    __all__ = [
        "TORCH_AVAILABLE",
        "create_mfg_operator",
    ]

# Always export availability info
__all__.extend(["TORCH_AVAILABLE"])
