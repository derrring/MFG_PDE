"""
Modified Multi-Layer Perceptron with skip connections and specialized features.

This module implements enhanced MLP architectures specifically designed for
Mean Field Games, including skip connections and physics-aware modifications.
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


class ModifiedMLP(nn.Module):
    """
    Modified Multi-Layer Perceptron with skip connections.

    Features:
    - Skip connections for improved gradient flow
    - Adaptive depth based on problem complexity
    - Physics-aware initialization
    - Support for multiple output heads
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: list[int],
        output_dim: int,
        activation: Literal["tanh", "relu", "sigmoid", "swish"] = "tanh",
        use_skip_connections: bool = True,
        skip_connection_freq: int = 2,
        final_activation: str | None = None,
    ):
        """
        Initialize Modified MLP.

        Args:
            input_dim: Number of input features
            hidden_layers: List of hidden layer sizes
            output_dim: Number of output features
            activation: Activation function for hidden layers
            use_skip_connections: Whether to use skip connections
            skip_connection_freq: Frequency of skip connections (every N layers)
            final_activation: Optional final activation function
        """
        super().__init__()

        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for ModifiedMLP")

        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        self.activation_name = activation
        self.use_skip_connections = use_skip_connections
        self.skip_connection_freq = skip_connection_freq
        self.final_activation_name = final_activation

        # Build network
        self.layers = nn.ModuleList()
        self.skip_projections = nn.ModuleList()

        # Track dimensions for skip connections
        self.layer_dims = [input_dim, *hidden_layers]

        # Hidden layers
        for i, (in_dim, out_dim) in enumerate(zip(self.layer_dims[:-1], self.layer_dims[1:], strict=False)):
            self.layers.append(nn.Linear(in_dim, out_dim))

            # Add skip projection if needed
            if use_skip_connections and i > 0 and (i + 1) % skip_connection_freq == 0 and in_dim != out_dim:
                self.skip_projections.append(nn.Linear(self.layer_dims[i - skip_connection_freq + 1], out_dim))
            else:
                self.skip_projections.append(None)

        # Output layer
        self.output_layer = nn.Linear(self.layer_dims[-1], output_dim)

        # Activation functions
        self.activation = self._get_activation_function(activation)
        self.final_activation = self._get_activation_function(final_activation) if final_activation else None

        # Initialize weights
        self._initialize_weights()

    def _get_activation_function(self, activation: str) -> object:
        """Get activation function by name."""
        activations = {
            "tanh": torch.tanh,
            "relu": functional.relu,
            "sigmoid": torch.sigmoid,
            "swish": functional.silu,  # SiLU is the same as Swish
            "elu": functional.elu,
        }

        if activation not in activations:
            raise ValueError(f"Unsupported activation: {activation}")

        return activations[activation]

    def _initialize_weights(self) -> None:
        """Initialize weights with physics-aware scheme."""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                # Xavier uniform for hidden layers
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        # Skip projection initialization
        for proj in self.skip_projections:
            if proj is not None:
                nn.init.xavier_uniform_(proj.weight)
                if proj.bias is not None:
                    nn.init.zeros_(proj.bias)

        # Output layer: smaller initialization for stability
        nn.init.xavier_uniform_(self.output_layer.weight, gain=0.1)
        if self.output_layer.bias is not None:
            nn.init.zeros_(self.output_layer.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with skip connections.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        current = x
        skip_inputs = [x]  # Store inputs for skip connections

        # Hidden layers with skip connections
        for i, layer in enumerate(self.layers):
            current = layer(current)

            # Add skip connection if applicable
            if (
                self.use_skip_connections
                and i >= self.skip_connection_freq - 1
                and (i + 1) % self.skip_connection_freq == 0
            ):
                skip_idx = i - self.skip_connection_freq + 1
                if skip_idx < len(skip_inputs):
                    skip_input = skip_inputs[skip_idx]

                    # Project skip connection if dimensions don't match
                    proj = self.skip_projections[i]
                    if proj is not None:
                        skip_connection = proj(skip_input)
                    else:
                        skip_connection = skip_input

                    current = current + skip_connection

            current = self.activation(current)
            skip_inputs.append(current)

        # Output layer
        output = self.output_layer(current)

        # Apply final activation if specified
        if self.final_activation is not None:
            output = self.final_activation(output)

        return output

    def get_network_depth(self) -> int:
        """Get effective network depth."""
        return len(self.layers)

    def get_parameter_count(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def compute_physics_loss_components(self, x: Tensor) -> dict[str, Tensor]:
        """
        Compute components needed for physics-based loss functions.

        Args:
            x: Input tensor with requires_grad=True

        Returns:
            Dictionary with physics loss components
        """
        if not x.requires_grad:
            x = x.requires_grad_(True)

        # Forward pass
        u = self.forward(x)

        # Compute gradients
        grad_outputs = torch.ones_like(u)
        u_x = torch.autograd.grad(
            outputs=u,
            inputs=x,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
        )[0]

        # Compute gradient magnitude for Hamiltonian
        grad_magnitude_squared = torch.sum(u_x**2, dim=1, keepdim=True)

        return {
            "u": u,
            "u_x": u_x,
            "grad_magnitude_squared": grad_magnitude_squared,
        }

    def __repr__(self) -> str:
        """String representation."""
        skip_info = f", skip_freq={self.skip_connection_freq}" if self.use_skip_connections else ""
        return (
            f"ModifiedMLP("
            f"input_dim={self.input_dim}, "
            f"hidden_layers={self.hidden_layers}, "
            f"output_dim={self.output_dim}, "
            f"activation='{self.activation_name}'"
            f"{skip_info}"
            f")"
        )
