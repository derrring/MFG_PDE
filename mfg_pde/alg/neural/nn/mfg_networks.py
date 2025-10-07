"""
Specialized network factory for Mean Field Games applications.

This module provides factory functions to create network architectures
specifically optimized for different types of MFG problems and equations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import torch.nn as nn

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    from .feedforward import FeedForwardNetwork
    from .modified_mlp import ModifiedMLP
    from .residual import ResidualNetwork


def create_mfg_networks(
    network_type: Literal["feedforward", "modified_mlp", "residual"] = "feedforward",
    input_dim: int = 2,  # (t, x) for 1D problems
    output_dim: int = 1,  # Single value function or density
    hidden_layers: list[int] | None = None,
    activation: Literal["tanh", "relu", "sigmoid", "elu"] = "tanh",
    problem_type: Literal["hjb", "fp", "coupled"] = "hjb",
    **kwargs: Any,
) -> nn.Module:
    """
    Create neural networks optimized for Mean Field Games.

    Args:
        network_type: Type of network architecture
        input_dim: Input dimension (typically 2 for (t,x) or 3 for (t,x,y))
        output_dim: Output dimension (1 for scalar functions)
        hidden_layers: List of hidden layer sizes
        activation: Activation function
        problem_type: Type of MFG problem (affects default parameters)
        **kwargs: Additional network-specific parameters

    Returns:
        Configured neural network module

    Raises:
        ImportError: If PyTorch is not available
        ValueError: If invalid network type specified
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for MFG neural networks")

    # Default hidden layer configurations based on problem type
    if hidden_layers is None:
        if problem_type == "hjb":
            hidden_layers = [50, 50, 50, 50]  # Deep for value functions
        elif problem_type == "fp":
            hidden_layers = [40, 40, 40]  # Moderate for densities
        else:  # coupled
            hidden_layers = [60, 60, 60, 60]  # Deeper for coupled systems

    # Create network based on type
    if network_type == "feedforward":
        return FeedForwardNetwork(
            input_dim=input_dim, hidden_layers=hidden_layers, output_dim=output_dim, activation=activation, **kwargs
        )

    elif network_type == "modified_mlp":
        return ModifiedMLP(
            input_dim=input_dim, hidden_layers=hidden_layers, output_dim=output_dim, activation=activation, **kwargs
        )

    elif network_type == "residual":
        # For residual networks, use consistent hidden dimension
        hidden_dim = hidden_layers[0] if hidden_layers else 50
        num_blocks = len(hidden_layers) if len(set(hidden_layers)) == 1 else 4

        return ResidualNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_blocks=num_blocks,
            activation=activation,
            **kwargs,
        )

    else:
        raise ValueError(f"Unknown network type: {network_type}")


def create_hjb_network(
    input_dim: int = 2,
    complexity: Literal["simple", "moderate", "complex"] = "moderate",
    activation: str = "tanh",
    **kwargs: Any,
) -> nn.Module:
    """
    Create network specifically for HJB equation solving.

    Args:
        input_dim: Input dimension
        complexity: Problem complexity level
        activation: Activation function
        **kwargs: Additional parameters

    Returns:
        Network optimized for HJB problems
    """
    complexity_configs = {
        "simple": {"hidden_layers": [32, 32, 32], "network_type": "feedforward"},
        "moderate": {"hidden_layers": [50, 50, 50, 50], "network_type": "modified_mlp"},
        "complex": {"hidden_layers": [64, 64, 64, 64], "network_type": "residual"},
    }

    config = complexity_configs[complexity]

    return create_mfg_networks(
        input_dim=input_dim,
        output_dim=1,  # Single value function
        problem_type="hjb",
        **config,
        activation=activation,
        **kwargs,
    )


def create_fp_network(
    input_dim: int = 2,
    ensure_positivity: bool = True,
    complexity: Literal["simple", "moderate", "complex"] = "moderate",
    activation: str = "tanh",
    **kwargs: Any,
) -> nn.Module:
    """
    Create network specifically for Fokker-Planck equation solving.

    Args:
        input_dim: Input dimension
        ensure_positivity: Whether to ensure positive density output
        complexity: Problem complexity level
        activation: Activation function
        **kwargs: Additional parameters

    Returns:
        Network optimized for FP problems
    """
    complexity_configs = {
        "simple": {"hidden_layers": [32, 32], "network_type": "feedforward"},
        "moderate": {"hidden_layers": [40, 40, 40], "network_type": "modified_mlp"},
        "complex": {"hidden_layers": [50, 50, 50, 50], "network_type": "residual"},
    }

    config = complexity_configs[complexity]

    # Add final activation for positivity if requested
    if ensure_positivity:
        if config["network_type"] == "modified_mlp":
            kwargs["final_activation"] = "sigmoid"
        else:
            # For other types, we'll apply softplus in the solver
            pass

    return create_mfg_networks(
        input_dim=input_dim,
        output_dim=1,  # Single density function
        problem_type="fp",
        **config,
        activation=activation,
        **kwargs,
    )


def create_coupled_mfg_networks(
    input_dim: int = 2,
    share_backbone: bool = False,
    complexity: Literal["simple", "moderate", "complex"] = "moderate",
    activation: Literal["tanh", "relu", "sigmoid", "elu"] = "tanh",
    **kwargs: Any,
) -> dict[str, nn.Module]:
    """
    Create paired networks for coupled MFG system solving.

    Args:
        input_dim: Input dimension
        share_backbone: Whether to share feature extraction layers
        complexity: Problem complexity level
        activation: Activation function
        **kwargs: Additional parameters

    Returns:
        Dictionary with 'hjb' and 'fp' networks
    """
    if share_backbone:
        # Create shared backbone with separate heads
        complexity_configs = {
            "simple": {"hidden_layers": [40, 40], "backbone_dim": 32},
            "moderate": {"hidden_layers": [60, 60], "backbone_dim": 50},
            "complex": {"hidden_layers": [80, 80], "backbone_dim": 64},
        }

        config = complexity_configs[complexity]
        backbone_layers = config["hidden_layers"]
        backbone_dim = config["backbone_dim"]

        # Shared backbone
        backbone = FeedForwardNetwork(
            input_dim=input_dim,
            hidden_layers=backbone_layers,
            output_dim=backbone_dim,
            activation=activation,
        )

        # Separate heads
        hjb_head = FeedForwardNetwork(
            input_dim=backbone_dim,
            hidden_layers=[backbone_dim // 2],
            output_dim=1,
            activation=activation,
        )

        fp_head = FeedForwardNetwork(
            input_dim=backbone_dim,
            hidden_layers=[backbone_dim // 2],
            output_dim=1,
            activation=activation,
        )

        # Combined networks
        class SharedBackboneNetwork(nn.Module):
            def __init__(self, backbone: nn.Module, head: nn.Module) -> None:
                super().__init__()
                self.backbone = backbone
                self.head = head

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                features = self.backbone(x)
                return self.head(features)

        return {
            "hjb": SharedBackboneNetwork(backbone, hjb_head),
            "fp": SharedBackboneNetwork(backbone, fp_head),
            "shared_backbone": backbone,
        }

    else:
        # Create separate networks
        return {
            "hjb": create_hjb_network(input_dim=input_dim, complexity=complexity, activation=activation, **kwargs),
            "fp": create_fp_network(input_dim=input_dim, complexity=complexity, activation=activation, **kwargs),
        }


def get_network_info(network: nn.Module) -> dict[str, object]:
    """
    Get comprehensive information about a neural network.

    Args:
        network: Neural network module

    Returns:
        Dictionary with network information
    """

    def count_parameters(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def get_layer_info(model: nn.Module) -> list[dict[str, object]]:
        layers: list[dict[str, object]] = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                layers.append(
                    {
                        "name": name,
                        "type": "Linear",
                        "in_features": module.in_features,
                        "out_features": module.out_features,
                        "parameters": module.in_features * module.out_features
                        + (module.out_features if module.bias is not None else 0),
                    }
                )
        return layers

    return {
        "type": network.__class__.__name__,
        "total_parameters": count_parameters(network),
        "trainable_parameters": sum(p.numel() for p in network.parameters() if p.requires_grad),
        "layers": get_layer_info(network),
        "device": next(network.parameters()).device.type if list(network.parameters()) else "cpu",
    }
