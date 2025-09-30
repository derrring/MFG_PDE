"""
Feed-forward neural network architecture for PINN solvers.

This module implements a flexible feed-forward network with physics-aware
initialization and activation functions optimized for Mean Field Games.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import torch.nn as nn
    from torch import Tensor

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class FeedForwardNetwork(nn.Module):
    """
    Feed-forward neural network for PINN applications.

    Optimized for physics-informed learning with:
    - Xavier/Glorot initialization for stable gradients
    - Multiple activation function choices
    - Optional bias terms
    - Batch normalization support
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: list[int],
        output_dim: int,
        activation: Literal["tanh", "relu", "sigmoid", "elu"] = "tanh",
        use_bias: bool = True,
        use_batch_norm: bool = False,
        dropout_rate: float = 0.0,
    ):
        """
        Initialize feed-forward network.

        Args:
            input_dim: Number of input features
            hidden_layers: List of hidden layer sizes
            output_dim: Number of output features
            activation: Activation function name
            use_bias: Whether to use bias terms
            use_batch_norm: Whether to use batch normalization
            dropout_rate: Dropout rate (0.0 = no dropout)
        """
        super().__init__()

        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for FeedForwardNetwork")

        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        self.activation_name = activation
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate

        # Build network layers
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        self.dropouts = nn.ModuleList() if dropout_rate > 0 else None

        # Input layer
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            self.layers.append(nn.Linear(prev_dim, hidden_dim, bias=use_bias))

            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

            if dropout_rate > 0:
                self.dropouts.append(nn.Dropout(dropout_rate))

            prev_dim = hidden_dim

        # Output layer
        self.output_layer = nn.Linear(prev_dim, output_dim, bias=use_bias)

        # Initialize weights
        self._initialize_weights()

        # Set activation function
        self.activation = self._get_activation_function(activation)

    def _get_activation_function(self, activation: str):
        """Get activation function by name."""
        activations = {
            "tanh": torch.tanh,
            "relu": F.relu,
            "sigmoid": torch.sigmoid,
            "elu": F.elu,
        }

        if activation not in activations:
            raise ValueError(f"Unsupported activation: {activation}")

        return activations[activation]

    def _initialize_weights(self):
        """Initialize network weights using Xavier/Glorot initialization."""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        # Output layer initialization
        nn.init.xavier_uniform_(self.output_layer.weight)
        if self.output_layer.bias is not None:
            nn.init.zeros_(self.output_layer.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        current = x

        # Hidden layers
        for i, layer in enumerate(self.layers):
            current = layer(current)

            if self.use_batch_norm and self.batch_norms is not None:
                current = self.batch_norms[i](current)

            current = self.activation(current)

            if self.dropout_rate > 0 and self.dropouts is not None:
                current = self.dropouts[i](current)

        # Output layer (no activation)
        output = self.output_layer(current)

        return output

    def get_gradients(self, x: Tensor, create_graph: bool = True) -> dict[str, Tensor]:
        """
        Compute gradients of the network output with respect to inputs.

        Args:
            x: Input tensor with requires_grad=True
            create_graph: Whether to create computation graph for higher-order derivatives

        Returns:
            Dictionary containing gradient information
        """
        if not x.requires_grad:
            x = x.requires_grad_(True)

        # Forward pass
        u = self.forward(x)

        # Compute first-order gradients
        grad_outputs = torch.ones_like(u)
        gradients = torch.autograd.grad(
            outputs=u,
            inputs=x,
            grad_outputs=grad_outputs,
            create_graph=create_graph,
            retain_graph=True,
        )[0]

        return {
            "u": u,
            "u_x": gradients,
        }

    def compute_laplacian(self, x: Tensor) -> Tensor:
        """
        Compute Laplacian (second derivatives) for diffusion terms.

        Args:
            x: Input tensor with requires_grad=True

        Returns:
            Laplacian tensor
        """
        if not x.requires_grad:
            x = x.requires_grad_(True)

        # Get first derivatives
        grad_info = self.get_gradients(x, create_graph=True)
        u_x = grad_info["u_x"]

        # Compute second derivatives
        laplacian = torch.zeros_like(u_x[:, 0:1])  # Initialize for output

        for i in range(x.shape[1]):  # For each spatial dimension
            u_xi = u_x[:, i : i + 1]
            u_xi_xi = torch.autograd.grad(
                outputs=u_xi,
                inputs=x,
                grad_outputs=torch.ones_like(u_xi),
                create_graph=True,
                retain_graph=True,
            )[0][:, i : i + 1]

            laplacian += u_xi_xi

        return laplacian

    def __repr__(self) -> str:
        """String representation of the network."""
        return (
            f"FeedForwardNetwork("
            f"input_dim={self.input_dim}, "
            f"hidden_layers={self.hidden_layers}, "
            f"output_dim={self.output_dim}, "
            f"activation='{self.activation_name}'"
            f")"
        )
