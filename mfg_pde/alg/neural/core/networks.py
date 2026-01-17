"""
Neural Network Architectures for Physics-Informed Neural Networks.

This module provides various neural network architectures optimized for
solving partial differential equations, particularly Mean Field Games.
Each architecture is designed with specific considerations for:

1. Smooth function approximation (important for PDEs)
2. Gradient computation accuracy (critical for PINN residuals)
3. Training stability and convergence
4. Scalability to high-dimensional problems

Key Features:
- Multiple activation functions (Tanh, GELU, Swish, Sine)
- Advanced initialization schemes
- Residual connections for deep networks
- Modified MLPs with enhanced expressivity
- Specialized architectures for MFG u(t,x) and m(t,x)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

# PyTorch imports with fallback
try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Mock classes for type hints
    nn = None
    F = None


class SineActivation(nn.Module):
    """
    Sine activation function: f(x) = sin(w * x + b)

    Sine activations are particularly effective for periodic functions
    and can provide better spectral properties for PINN training.
    """

    def __init__(self, w0: float = 1.0):
        """
        Initialize sine activation.

        Args:
            w0: Frequency parameter for sine function
        """
        super().__init__()
        self.w0 = w0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * x)


class SwishActivation(nn.Module):
    """
    Swish activation function: f(x) = x * sigmoid(x)

    Swish provides smooth gradients and has shown good performance
    in deep learning applications, including PINNs.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


def get_activation_function(activation_name: str, **kwargs: Any) -> nn.Module:
    """
    Get activation function by name.

    Args:
        activation_name: Name of activation function
        **kwargs: Additional parameters for activation

    Returns:
        Activation function module

    Raises:
        ValueError: If activation function is not supported
    """
    activation_name = activation_name.lower()

    if activation_name == "tanh":
        return nn.Tanh()
    elif activation_name == "relu":
        return nn.ReLU()
    elif activation_name == "gelu":
        return nn.GELU()
    elif activation_name == "elu":
        return nn.ELU()
    elif activation_name == "swish":
        return SwishActivation()
    elif activation_name == "sine":
        w0 = kwargs.get("w0", 1.0)
        return SineActivation(w0=w0)
    else:
        raise ValueError(f"Unsupported activation function: {activation_name}")


class FeedForwardNetwork(nn.Module):
    """
    Standard feedforward neural network for PINN applications.

    This is the most basic architecture for PINNs, consisting of
    fully connected layers with specified activation functions.
    Includes options for normalization and regularization.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: list[int],
        output_dim: int,
        activation: str = "tanh",
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
        dropout_rate: float = 0.0,
        initialization: str = "xavier_normal",
    ):
        """
        Initialize feedforward network.

        Args:
            input_dim: Input dimension (typically 2 for t,x)
            hidden_layers: List of hidden layer sizes
            output_dim: Output dimension (1 for scalar functions)
            activation: Activation function name
            use_batch_norm: Whether to use batch normalization
            use_layer_norm: Whether to use layer normalization
            dropout_rate: Dropout probability (0 = no dropout)
            initialization: Weight initialization scheme
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation_name = activation

        # Build layer list
        layer_sizes = [input_dim, *hidden_layers, output_dim]
        self.layers = nn.ModuleList()

        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

        # Activation functions (all hidden layers use same activation)
        self.activation = get_activation_function(activation)

        # Normalization layers
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm

        if use_batch_norm:
            self.batch_norms = nn.ModuleList([nn.BatchNorm1d(size) for size in hidden_layers])

        if use_layer_norm:
            self.layer_norms = nn.ModuleList([nn.LayerNorm(size) for size in hidden_layers])

        # Dropout
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

        # Initialize weights
        self._initialize_weights(initialization)

    def _initialize_weights(self, initialization: str) -> None:
        """Initialize network weights according to specified scheme."""
        for layer in self.layers[:-1]:  # Don't initialize output layer specially
            if initialization == "xavier_normal":
                nn.init.xavier_normal_(layer.weight)
            elif initialization == "xavier_uniform":
                nn.init.xavier_uniform_(layer.weight)
            elif initialization == "kaiming":
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            else:
                raise ValueError(f"Unknown initialization: {initialization}")

            # Initialize biases to zero
            nn.init.zeros_(layer.bias)

        # Special initialization for sine activation
        if self.activation_name == "sine":
            with torch.no_grad():
                # First layer should have uniform distribution
                if len(self.layers) > 0:
                    self.layers[0].weight.uniform_(-1 / self.input_dim, 1 / self.input_dim)

                # Other layers should be initialized carefully for sine
                for layer in self.layers[1:]:
                    bound = np.sqrt(6 / layer.weight.size(1)) / 30  # Scaled for sine
                    layer.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)

            # Apply normalization
            if self.use_batch_norm and i < len(getattr(self, "batch_norms", [])):
                x = self.batch_norms[i](x)
            elif self.use_layer_norm and i < len(getattr(self, "layer_norms", [])):
                x = self.layer_norms[i](x)

            # Apply activation
            x = self.activation(x)

            # Apply dropout
            if self.dropout is not None:
                x = self.dropout(x)

        # Output layer (no activation)
        x = self.layers[-1](x)

        return x


class ResidualNetwork(nn.Module):
    """
    Residual neural network with skip connections.

    Residual connections help with training deeper networks and
    can improve gradient flow, which is crucial for PINN training
    where gradients are computed through the network.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: list[int],
        output_dim: int,
        activation: str = "tanh",
        residual_frequency: int = 2,
        initialization: str = "xavier_normal",
    ):
        """
        Initialize residual network.

        Args:
            input_dim: Input dimension
            hidden_layers: List of hidden layer sizes
            output_dim: Output dimension
            activation: Activation function name
            residual_frequency: Add residual connection every N layers
            initialization: Weight initialization scheme
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.residual_frequency = residual_frequency

        # Build layers
        layer_sizes = [input_dim, *hidden_layers]
        self.hidden_layers = nn.ModuleList()

        for i in range(len(layer_sizes) - 1):
            self.hidden_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

        # Output layer
        self.output_layer = nn.Linear(hidden_layers[-1], output_dim)

        # Activation
        self.activation = get_activation_function(activation)

        # Skip connection adapters (for dimension matching)
        self.skip_adapters = nn.ModuleList()
        for i in range(0, len(self.hidden_layers), residual_frequency):
            if i + residual_frequency < len(self.hidden_layers):
                in_dim = layer_sizes[i]
                out_dim = layer_sizes[i + residual_frequency]
                if in_dim != out_dim:
                    self.skip_adapters.append(nn.Linear(in_dim, out_dim))
                else:
                    self.skip_adapters.append(nn.Identity())

        # Initialize weights
        self._initialize_weights(initialization)

    def _initialize_weights(self, initialization: str) -> None:
        """Initialize network weights."""
        for layer in [*list(self.hidden_layers), self.output_layer]:
            if initialization == "xavier_normal":
                nn.init.xavier_normal_(layer.weight)
            elif initialization == "xavier_uniform":
                nn.init.xavier_uniform_(layer.weight)
            elif initialization == "kaiming":
                nn.init.kaiming_normal_(layer.weight)

            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections."""
        residual = x
        skip_idx = 0

        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            x = self.activation(x)

            # Add residual connection
            if (i + 1) % self.residual_frequency == 0 and skip_idx < len(self.skip_adapters):
                adapted_residual = self.skip_adapters[skip_idx](residual)
                x = x + adapted_residual
                residual = x
                skip_idx += 1

        # Output layer
        x = self.output_layer(x)

        return x


class ModifiedMLP(nn.Module):
    """
    Modified MLP with enhanced features for PINN applications.

    This architecture includes several modifications that can improve
    PINN training:
    - Fourier feature mapping for better high-frequency representation
    - Adaptive activation scaling
    - Skip connections
    - Gradient-friendly design
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: list[int],
        output_dim: int,
        activation: str = "tanh",
        use_fourier_features: bool = False,
        fourier_scale: float = 1.0,
        num_fourier_features: int = 256,
        initialization: str = "xavier_normal",
    ):
        """
        Initialize modified MLP.

        Args:
            input_dim: Input dimension
            hidden_layers: List of hidden layer sizes
            output_dim: Output dimension
            activation: Activation function name
            use_fourier_features: Whether to use Fourier feature mapping
            fourier_scale: Scale for Fourier features
            num_fourier_features: Number of Fourier features to use
            initialization: Weight initialization scheme
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_fourier_features = use_fourier_features

        # Fourier feature mapping
        if use_fourier_features:
            self.fourier_scale = fourier_scale
            self.num_fourier_features = num_fourier_features

            # Random Fourier feature matrix (fixed during training)
            self.register_buffer("fourier_matrix", fourier_scale * torch.randn(input_dim, num_fourier_features))

            # Effective input dimension after Fourier mapping
            effective_input_dim = 2 * num_fourier_features
        else:
            effective_input_dim = input_dim

        # Build main network
        self.main_net = FeedForwardNetwork(
            input_dim=effective_input_dim,
            hidden_layers=hidden_layers,
            output_dim=output_dim,
            activation=activation,
            initialization=initialization,
        )

    def fourier_feature_mapping(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Fourier feature mapping to input."""
        # x: [batch_size, input_dim]
        # fourier_matrix: [input_dim, num_fourier_features]

        projected = torch.matmul(x, self.fourier_matrix)  # [batch_size, num_fourier_features]

        # Apply sine and cosine
        sine_features = torch.sin(2 * math.pi * projected)
        cosine_features = torch.cos(2 * math.pi * projected)

        # Concatenate sine and cosine features
        fourier_features = torch.cat([sine_features, cosine_features], dim=-1)

        return fourier_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through modified MLP."""
        if self.use_fourier_features:
            x = self.fourier_feature_mapping(x)

        return self.main_net(x)


class NetworkArchitecture:
    """
    Factory class for creating neural network architectures.

    Provides convenient methods for creating commonly used
    network architectures for PINN applications.
    """

    @staticmethod
    def create_standard_pinn(
        input_dim: int = 2,
        hidden_layers: list[int] | None = None,
        output_dim: int = 1,
        activation: str = "tanh",
        **kwargs: Any,
    ) -> nn.Module:
        """
        Create standard PINN architecture.

        Args:
            input_dim: Input dimension (default 2 for t,x)
            hidden_layers: Hidden layer sizes (default [50, 50, 50])
            output_dim: Output dimension (default 1)
            activation: Activation function
            **kwargs: Additional arguments

        Returns:
            Neural network module
        """
        if hidden_layers is None:
            hidden_layers = [50, 50, 50]

        return FeedForwardNetwork(
            input_dim=input_dim, hidden_layers=hidden_layers, output_dim=output_dim, activation=activation, **kwargs
        )

    @staticmethod
    def create_deep_pinn(
        input_dim: int = 2,
        hidden_size: int = 100,
        num_layers: int = 8,
        output_dim: int = 1,
        activation: str = "tanh",
        **kwargs: Any,
    ) -> nn.Module:
        """
        Create deep PINN with residual connections.

        Args:
            input_dim: Input dimension
            hidden_size: Size of each hidden layer
            num_layers: Number of hidden layers
            output_dim: Output dimension
            activation: Activation function
            **kwargs: Additional arguments

        Returns:
            Residual neural network
        """
        hidden_layers = [hidden_size] * num_layers

        return ResidualNetwork(
            input_dim=input_dim, hidden_layers=hidden_layers, output_dim=output_dim, activation=activation, **kwargs
        )

    @staticmethod
    def create_fourier_pinn(
        input_dim: int = 2,
        hidden_layers: list[int] | None = None,
        output_dim: int = 1,
        num_fourier_features: int = 256,
        fourier_scale: float = 1.0,
        activation: str = "relu",
        **kwargs: Any,
    ) -> nn.Module:
        """
        Create PINN with Fourier feature mapping.

        Args:
            input_dim: Input dimension
            hidden_layers: Hidden layer sizes
            output_dim: Output dimension
            num_fourier_features: Number of Fourier features
            fourier_scale: Scale for Fourier mapping
            activation: Activation function
            **kwargs: Additional arguments

        Returns:
            Modified MLP with Fourier features
        """
        if hidden_layers is None:
            hidden_layers = [256, 256, 256]

        return ModifiedMLP(
            input_dim=input_dim,
            hidden_layers=hidden_layers,
            output_dim=output_dim,
            activation=activation,
            use_fourier_features=True,
            num_fourier_features=num_fourier_features,
            fourier_scale=fourier_scale,
            **kwargs,
        )


def create_mfg_networks(
    architecture_type: str = "standard", separate_networks: bool = True, **kwargs: Any
) -> dict[str, nn.Module]:
    """
    Create neural networks specifically for MFG problems.

    Args:
        architecture_type: Type of architecture ("standard", "deep", "fourier")
        separate_networks: Whether to use separate networks for u and m
        **kwargs: Additional arguments for network creation

    Returns:
        Dictionary of neural networks with keys 'u_net' and optionally 'm_net'
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for neural networks")

    # Default parameters for MFG
    default_kwargs = {
        "input_dim": 2,  # (t, x)
        "output_dim": 1,  # scalar function
        "activation": "tanh",
    }
    default_kwargs.update(kwargs)

    networks = {}

    if architecture_type == "standard":
        creator = NetworkArchitecture.create_standard_pinn
    elif architecture_type == "deep":
        creator = NetworkArchitecture.create_deep_pinn
    elif architecture_type == "fourier":
        creator = NetworkArchitecture.create_fourier_pinn
    else:
        raise ValueError(f"Unknown architecture type: {architecture_type}")

    # Create network for value function u(t,x)
    networks["u_net"] = creator(**default_kwargs)

    # Create separate network for density m(t,x) if requested
    if separate_networks:
        # Density network might benefit from different parameters
        m_kwargs = default_kwargs.copy()
        m_kwargs.update(kwargs.get("m_net_kwargs", {}))
        networks["m_net"] = creator(**m_kwargs)

    return networks


# Utility functions for network analysis and debugging


def count_parameters(network: nn.Module) -> int:
    """Count total number of trainable parameters in network."""
    return sum(p.numel() for p in network.parameters() if p.requires_grad)


def analyze_network_gradients(network: nn.Module) -> dict[str, float]:
    """Analyze gradient statistics for debugging training issues."""
    grad_stats: dict[str, float] = {
        "grad_norm": 0.0,
        "grad_mean": 0.0,
        "grad_std": 0.0,
        "num_zero_grads": 0,
        "total_params": 0,
    }

    all_grads: list[torch.Tensor] = []
    zero_count = 0

    for param in network.parameters():
        if param.grad is not None:
            grad = param.grad.data
            all_grads.append(grad.flatten())
            zero_count += (grad == 0).sum().item()
        grad_stats["total_params"] += param.numel()

    if all_grads:
        all_grads_tensor = torch.cat(all_grads)
        grad_stats["grad_norm"] = torch.norm(all_grads_tensor).item()
        grad_stats["grad_mean"] = torch.mean(all_grads_tensor).item()
        grad_stats["grad_std"] = torch.std(all_grads_tensor).item()
        grad_stats["num_zero_grads"] = zero_count

    return grad_stats


def print_network_info(network: nn.Module, name: str = "Network") -> None:
    """Print detailed information about network architecture."""
    print(f"{name} Architecture:")
    print(f"  Total parameters: {count_parameters(network):,}")
    print("  Network structure:")

    for _i, (name, module) in enumerate(network.named_modules()):
        if len(list(module.children())) == 0:  # Leaf modules only
            # Backend compatibility - PyTorch module introspection (Issue #543 acceptable)
            # Some modules (Linear, Conv2d) have weights, others (ReLU, Dropout) don't
            if hasattr(module, "weight") and module.weight is not None:
                weight_shape = tuple(module.weight.shape)
                print(f"    {name}: {module.__class__.__name__} {weight_shape}")
            else:
                print(f"    {name}: {module.__class__.__name__}")
    print()
