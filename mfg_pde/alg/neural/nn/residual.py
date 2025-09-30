"""
Residual neural network architecture for deep PINN applications.

This module implements ResNet-style architectures optimized for physics-informed
learning, enabling very deep networks for complex Mean Field Games.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import torch.nn as nn
    from torch import Tensor

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ResidualBlock(nn.Module):
    """Basic residual block for deep neural networks."""

    def __init__(
        self,
        dim: int,
        activation: Literal["tanh", "relu", "swish"] = "tanh",
        use_batch_norm: bool = False,
        dropout_rate: float = 0.0,
    ):
        """
        Initialize residual block.

        Args:
            dim: Feature dimension
            activation: Activation function
            use_batch_norm: Whether to use batch normalization
            dropout_rate: Dropout rate
        """
        super().__init__()

        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)

        self.batch_norm1 = nn.BatchNorm1d(dim) if use_batch_norm else None
        self.batch_norm2 = nn.BatchNorm1d(dim) if use_batch_norm else None

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

        # Activation function
        activations = {
            "tanh": torch.tanh,
            "relu": functional.relu,
            "swish": functional.silu,
        }
        self.activation = activations[activation]

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for residual learning."""
        # Standard initialization for first layer
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)

        # Small initialization for second layer to start with identity mapping
        nn.init.xavier_uniform_(self.linear2.weight, gain=0.1)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through residual block."""
        residual = x

        # First transformation
        out = self.linear1(x)
        if self.batch_norm1 is not None:
            out = self.batch_norm1(out)
        out = self.activation(out)

        if self.dropout is not None:
            out = self.dropout(out)

        # Second transformation
        out = self.linear2(out)
        if self.batch_norm2 is not None:
            out = self.batch_norm2(out)

        # Residual connection
        out = out + residual
        out = self.activation(out)

        return out


class ResidualNetwork(nn.Module):
    """
    Deep residual network for physics-informed learning.

    Enables training of very deep networks by using residual connections
    to combat vanishing gradients in physics-based loss optimization.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_blocks: int = 4,
        activation: Literal["tanh", "relu", "swish"] = "tanh",
        use_batch_norm: bool = False,
        dropout_rate: float = 0.0,
        use_input_projection: bool = True,
    ):
        """
        Initialize residual network.

        Args:
            input_dim: Number of input features
            hidden_dim: Hidden layer dimension (kept constant)
            output_dim: Number of output features
            num_blocks: Number of residual blocks
            activation: Activation function
            use_batch_norm: Whether to use batch normalization
            dropout_rate: Dropout rate
            use_input_projection: Whether to project input to hidden_dim
        """
        super().__init__()

        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for ResidualNetwork")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_blocks = num_blocks
        self.activation_name = activation
        self.use_input_projection = use_input_projection

        # Input projection
        if use_input_projection and input_dim != hidden_dim:
            self.input_projection = nn.Linear(input_dim, hidden_dim)
        else:
            self.input_projection = None

        # Residual blocks
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(
                    dim=hidden_dim,
                    activation=activation,
                    use_batch_norm=use_batch_norm,
                    dropout_rate=dropout_rate,
                )
                for _ in range(num_blocks)
            ]
        )

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # Activation function
        activations = {
            "tanh": torch.tanh,
            "relu": functional.relu,
            "swish": functional.silu,
        }
        self.activation = activations[activation]

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights."""
        if self.input_projection is not None:
            nn.init.xavier_uniform_(self.input_projection.weight)
            nn.init.zeros_(self.input_projection.bias)

        # Output layer
        nn.init.xavier_uniform_(self.output_layer.weight, gain=0.1)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through residual network.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Input projection
        if self.input_projection is not None:
            current = self.input_projection(x)
            current = self.activation(current)
        else:
            current = x

        # Residual blocks
        for block in self.blocks:
            current = block(current)

        # Output layer
        output = self.output_layer(current)

        return output

    def get_effective_depth(self) -> int:
        """Get effective network depth considering residual connections."""
        return 1 + (2 * self.num_blocks) + 1  # input + blocks + output

    def compute_deep_features(self, x: Tensor) -> list[Tensor]:
        """
        Compute intermediate features from each residual block.

        Useful for analyzing what the network learns at different depths.

        Args:
            x: Input tensor

        Returns:
            List of feature tensors from each block
        """
        features = []

        # Input features
        if self.input_projection is not None:
            current = self.input_projection(x)
            current = self.activation(current)
        else:
            current = x

        features.append(current)

        # Features from each block
        for block in self.blocks:
            current = block(current)
            features.append(current)

        return features

    def compute_gradients_with_depth_analysis(self, x: Tensor) -> dict[str, Tensor]:
        """
        Compute gradients and analyze gradient flow through depth.

        Args:
            x: Input tensor with requires_grad=True

        Returns:
            Dictionary with gradient information and depth analysis
        """
        if not x.requires_grad:
            x = x.requires_grad_(True)

        # Get features at each depth
        features = self.compute_deep_features(x)

        # Forward pass to output
        output = self.output_layer(features[-1])

        # Compute gradients
        grad_outputs = torch.ones_like(output)
        gradients = torch.autograd.grad(
            outputs=output,
            inputs=x,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
        )[0]

        # Gradient norms at each layer (for analysis)
        gradient_norms = []
        for _i, feature in enumerate(features):
            if feature.requires_grad:
                feature_grad = torch.autograd.grad(
                    outputs=output,
                    inputs=feature,
                    grad_outputs=grad_outputs,
                    create_graph=True,
                    retain_graph=True,
                )[0]
                gradient_norms.append(torch.norm(feature_grad, dim=1, keepdim=True))

        return {
            "u": output,
            "u_x": gradients,
            "features": features,
            "gradient_norms": gradient_norms,
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ResidualNetwork("
            f"input_dim={self.input_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"output_dim={self.output_dim}, "
            f"num_blocks={self.num_blocks}, "
            f"activation='{self.activation_name}'"
            f")"
        )
