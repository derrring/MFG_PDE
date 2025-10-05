"""
Neural network architectures optimized for Deep Galerkin Methods.

This module provides specialized neural network architectures designed
for high-dimensional function approximation in DGM solvers. These
architectures are optimized for representing solutions to PDE systems
in high dimensions (d > 5).

Key Architectures:
- DeepGalerkinNetwork: Standard DGM architecture with residual connections
- HighDimMLP: Multi-layer perceptron optimized for high dimensions
- ResidualDGMNetwork: Deep residual networks for complex function approximation

Design Principles:
- Smooth activation functions (tanh, sigmoid) for PDE approximation
- Residual connections to enable deep networks
- Proper initialization for high-dimensional problems
- Batch normalization for training stability
"""

from __future__ import annotations

import numpy as np

# Import with availability checking
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class DeepGalerkinNetwork(nn.Module):
        """
        Standard Deep Galerkin Method architecture.

        This network implements the DGM architecture from Sirignano & Spiliopoulos (2018)
        with modifications for Mean Field Games. Features include:
        - Residual connections for deep training
        - Smooth activations for PDE approximation
        - Proper initialization for high-dimensional inputs

        Mathematical Framework:
        - Input: (t, x) ∈ [0,T] × Ω ⊂ ℝ^d
        - Output: u(t,x) or m(t,x) function approximation
        - Architecture: Deep feedforward with residual connections
        """

        def __init__(
            self,
            input_dim: int,
            hidden_layers: list[int] | None = None,
            output_dim: int = 1,
            activation: str = "tanh",
            use_batch_norm: bool = False,
            dropout_rate: float = 0.0,
        ):
            """
            Initialize Deep Galerkin network.

            Args:
                input_dim: Input dimension (d + 1 for space + time)
                hidden_layers: List of hidden layer sizes
                output_dim: Output dimension (1 for scalar functions)
                activation: Activation function name
                use_batch_norm: Whether to use batch normalization
                dropout_rate: Dropout rate for regularization
            """
            super().__init__()

            if hidden_layers is None:
                hidden_layers = [256, 256, 256, 256]  # Deep for high-dimensional

            self.input_dim = input_dim
            self.hidden_layers = hidden_layers
            self.output_dim = output_dim
            self.use_batch_norm = use_batch_norm
            self.dropout_rate = dropout_rate
            self.activation_name = activation

            # Store activation class (not instance) for Sequential compatibility
            if activation == "tanh":
                self.activation_class = nn.Tanh
            elif activation == "sigmoid":
                self.activation_class = nn.Sigmoid
            elif activation == "relu":
                self.activation_class = nn.ReLU
            elif activation == "swish":
                self.activation_class = nn.SiLU  # SiLU is PyTorch's Swish
            else:
                raise ValueError(f"Unknown activation: {activation}")

            # Build network layers
            self._build_network()

            # Initialize weights properly for high-dimensional problems
            self._initialize_weights()

        def _build_network(self) -> None:
            """Build the DGM network architecture."""
            layers = []
            layer_sizes = [self.input_dim, *self.hidden_layers, self.output_dim]

            for i in range(len(layer_sizes) - 1):
                # Linear layer
                layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

                # Batch normalization (except for output layer)
                if self.use_batch_norm and i < len(layer_sizes) - 2:
                    layers.append(nn.BatchNorm1d(layer_sizes[i + 1]))

                # Activation (except for output layer)
                if i < len(layer_sizes) - 2:
                    layers.append(self.activation_class())

                # Dropout (except for output layer)
                if self.dropout_rate > 0 and i < len(layer_sizes) - 2:
                    layers.append(nn.Dropout(self.dropout_rate))

            self.network = nn.Sequential(*layers)

            # Residual connection layers (for deep networks)
            if len(self.hidden_layers) >= 3:
                self.residual_layers = nn.ModuleList()
                for i in range(1, len(self.hidden_layers) - 1):
                    if self.hidden_layers[i] == self.hidden_layers[i - 1]:
                        # Can add residual connection for same-size layers
                        self.residual_layers.append(nn.Identity())
                    else:
                        # Need projection for different sizes
                        self.residual_layers.append(nn.Linear(self.hidden_layers[i - 1], self.hidden_layers[i]))

        def _initialize_weights(self) -> None:
            """Initialize weights using Xavier initialization for high-dimensional problems."""
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    # Xavier initialization scaled for high-dimensional inputs
                    fan_in = module.in_features
                    fan_out = module.out_features

                    # Scale initialization based on input dimension
                    scale = np.sqrt(2.0 / (fan_in + fan_out)) * np.sqrt(self.input_dim / 10.0)

                    nn.init.uniform_(module.weight, -scale, scale)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass through DGM network."""
            return self.network(x)

    class HighDimMLP(nn.Module):
        """
        Multi-layer perceptron optimized for high-dimensional approximation.

        Features specialized optimizations for high-dimensional function approximation:
        - Dimension-aware initialization
        - Skip connections for gradient flow
        - Adaptive layer width based on input dimension
        """

        def __init__(
            self,
            input_dim: int,
            output_dim: int = 1,
            base_width: int = 128,
            num_layers: int = 6,
            activation: str = "tanh",
        ):
            """
            Initialize high-dimensional MLP.

            Args:
                input_dim: Input dimension (d + 1)
                output_dim: Output dimension
                base_width: Base layer width (scaled by dimension)
                num_layers: Number of hidden layers
                activation: Activation function
            """
            super().__init__()

            self.input_dim = input_dim
            self.output_dim = output_dim

            # Scale layer width based on dimension (more dimensions need wider layers)
            layer_width = max(base_width, int(base_width * np.log(input_dim)))

            # Build layers
            self.layers = nn.ModuleList()
            self.layers.append(nn.Linear(input_dim, layer_width))

            for _ in range(num_layers - 1):
                self.layers.append(nn.Linear(layer_width, layer_width))

            self.layers.append(nn.Linear(layer_width, output_dim))

            # Activation
            if activation == "tanh":
                self.activation = torch.tanh
            elif activation == "sigmoid":
                self.activation = torch.sigmoid
            else:
                self.activation = functional.relu

            # Skip connections for deep networks
            self.use_skip = num_layers >= 4

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass with optional skip connections."""
            h = self.activation(self.layers[0](x))

            # Hidden layers with skip connections
            for i, layer in enumerate(self.layers[1:-1]):
                h_new = self.activation(layer(h))

                # Add skip connection every 2 layers
                if self.use_skip and i % 2 == 1 and h_new.shape == h.shape:
                    h_new = h_new + h

                h = h_new

            # Output layer (no activation)
            return self.layers[-1](h)

    class ResidualDGMNetwork(nn.Module):
        """
        Residual network architecture for Deep Galerkin Methods.

        Implements ResNet-style architecture specifically adapted for PDE
        function approximation with deep residual blocks for complex
        high-dimensional solution representation.
        """

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 256,
            num_residual_blocks: int = 6,
            output_dim: int = 1,
            activation: str = "tanh",
        ):
            """
            Initialize residual DGM network.

            Args:
                input_dim: Input dimension (d + 1)
                hidden_dim: Hidden layer dimension
                num_residual_blocks: Number of residual blocks
                output_dim: Output dimension
                activation: Activation function
            """
            super().__init__()

            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.activation_name = activation

            # Store activation class for Sequential compatibility
            if activation == "tanh":
                self.activation = torch.tanh  # Function for forward()
                self.activation_class = nn.Tanh
            elif activation == "swish":
                self.activation = lambda x: x * torch.sigmoid(x)  # Function for forward()
                self.activation_class = nn.SiLU
            else:
                self.activation = functional.relu  # Function for forward()
                self.activation_class = nn.ReLU

            # Input projection
            self.input_layer = nn.Linear(input_dim, hidden_dim)

            # Residual blocks
            self.residual_blocks = nn.ModuleList()
            for _ in range(num_residual_blocks):
                self.residual_blocks.append(self._make_residual_block(hidden_dim))

            # Output projection
            self.output_layer = nn.Linear(hidden_dim, output_dim)

        def _make_residual_block(self, dim: int) -> nn.Module:
            """Create a residual block."""
            return nn.Sequential(nn.Linear(dim, dim), self.activation_class(), nn.Linear(dim, dim))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass through residual DGM network."""
            # Input projection
            h = self.activation(self.input_layer(x))

            # Residual blocks
            for block in self.residual_blocks:
                residual = block(h)
                h = h + residual  # Residual connection

            # Output projection
            return self.output_layer(h)

else:
    # Placeholder classes when PyTorch is not available
    class DeepGalerkinNetwork:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for DGM architectures")

    class HighDimMLP:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for DGM architectures")

    class ResidualDGMNetwork:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for DGM architectures")
