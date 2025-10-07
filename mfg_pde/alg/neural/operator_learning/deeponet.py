"""
DeepONet (Deep Operator Networks) for MFG Problems.

This module implements Deep Operator Networks for learning parameter-to-solution
operators in Mean Field Games. DeepONet uses a branch-trunk architecture to
learn operators that map from parameter functions to solution functions.

Mathematical Framework:
- Branch Network: Encodes input parameter functions
- Trunk Network: Encodes evaluation coordinates
- Operator Learning: G(u)(y) = Σᵢ bᵢ(u) * tᵢ(y)
- Universal Approximation: Can approximate any continuous operator

Key Architecture Components:
- Branch Net: Processes parameter functions/data
- Trunk Net: Processes spatial-temporal coordinates
- Dot Product: Combines branch and trunk outputs
- Bias Network: Optional bias term for improved accuracy

Applications:
- Parameter-to-Solution Mapping: θ → u(x,t) for MFG problems
- Multi-Fidelity Learning: Combine low/high fidelity data
- Uncertainty Quantification: Bayesian operator learning
- Real-Time Control: Fast operator evaluation

References:
- Lu et al. "Learning nonlinear operators via DeepONet based on the universal approximation theorem" (2021)
- Wang et al. "Learning the solution operator of parametric partial differential equations" (2021)
"""

from __future__ import annotations

from dataclasses import dataclass

# Import with availability checking
try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    from .base_operator import BaseNeuralOperator, OperatorConfig

    @dataclass
    class DeepONetConfig(OperatorConfig):
        """Configuration for DeepONet."""

        # Architecture parameters
        branch_depth: int = 6  # Depth of branch network
        trunk_depth: int = 6  # Depth of trunk network
        branch_width: int = 128  # Width of branch network
        trunk_width: int = 128  # Width of trunk network

        # Output dimension
        latent_dim: int = 64  # Dimension of latent space (p in paper)

        # Input specifications
        sensor_points: int = 100  # Number of sensor points for branch network
        coordinate_dim: int = 2  # Dimension of coordinates (space + time)

        # Activation functions
        branch_activation: str = "relu"  # Branch network activation
        trunk_activation: str = "tanh"  # Trunk network activation

        # Normalization
        use_batch_norm: bool = False  # Use batch normalization
        use_layer_norm: bool = True  # Use layer normalization

        # Bias network
        use_bias_net: bool = True  # Use bias network
        bias_depth: int = 2  # Depth of bias network

        # Advanced features
        use_attention: bool = False  # Use attention mechanism in branch
        residual_connections: bool = True  # Use residual connections

        def __post_init__(self):
            """Set DeepONet-specific defaults."""
            super().__post_init__()
            self.operator_type = "deeponet"

    class BranchNetwork(nn.Module):
        """Branch network for encoding parameter functions."""

        def __init__(self, config: DeepONetConfig):
            """
            Initialize branch network.

            Args:
                config: DeepONet configuration
            """
            super().__init__()
            self.config = config

            # Get activation function
            if config.branch_activation == "relu":
                self.activation = nn.ReLU()
            elif config.branch_activation == "tanh":
                self.activation = nn.Tanh()
            elif config.branch_activation == "gelu":
                self.activation = nn.GELU()
            else:
                self.activation = nn.ReLU()

            # Build branch network layers
            layers = []

            # Input layer
            layers.append(nn.Linear(config.sensor_points, config.branch_width))
            if config.use_layer_norm:
                layers.append(nn.LayerNorm(config.branch_width))
            layers.append(self.activation)

            # Hidden layers
            for i in range(config.branch_depth - 2):
                layers.append(nn.Linear(config.branch_width, config.branch_width))
                if config.use_batch_norm:
                    layers.append(nn.BatchNorm1d(config.branch_width))
                elif config.use_layer_norm:
                    layers.append(nn.LayerNorm(config.branch_width))
                layers.append(self.activation)

                # Add residual connection option
                if config.residual_connections and i > 0:
                    # Store layer for residual connection
                    pass

            # Output layer
            layers.append(nn.Linear(config.branch_width, config.latent_dim))

            self.network = nn.Sequential(*layers)

            # Attention mechanism (optional)
            if config.use_attention:
                self.attention = nn.MultiheadAttention(embed_dim=config.branch_width, num_heads=8, batch_first=True)

        def forward(self, u: torch.Tensor) -> torch.Tensor:
            """
            Forward pass through branch network.

            Args:
                u: Input parameter function values [batch, sensor_points]

            Returns:
                Branch network output [batch, latent_dim]
            """
            # Apply attention if enabled
            if self.config.use_attention and hasattr(self, "attention"):
                # Reshape for attention
                u_expanded = u.unsqueeze(1)  # [batch, 1, sensor_points]
                u_attended, _ = self.attention(u_expanded, u_expanded, u_expanded)
                u = u_attended.squeeze(1)

            # Pass through network
            branch_output = self.network(u)
            return branch_output

    class TrunkNetwork(nn.Module):
        """Trunk network for encoding evaluation coordinates."""

        def __init__(self, config: DeepONetConfig):
            """
            Initialize trunk network.

            Args:
                config: DeepONet configuration
            """
            super().__init__()
            self.config = config

            # Get activation function
            if config.trunk_activation == "tanh":
                self.activation = nn.Tanh()
            elif config.trunk_activation == "relu":
                self.activation = nn.ReLU()
            elif config.trunk_activation == "gelu":
                self.activation = nn.GELU()
            else:
                self.activation = nn.Tanh()

            # Build trunk network layers
            layers = []

            # Input layer
            layers.append(nn.Linear(config.coordinate_dim, config.trunk_width))
            if config.use_layer_norm:
                layers.append(nn.LayerNorm(config.trunk_width))
            layers.append(self.activation)

            # Hidden layers
            for _i in range(config.trunk_depth - 2):
                layers.append(nn.Linear(config.trunk_width, config.trunk_width))
                if config.use_batch_norm:
                    layers.append(nn.BatchNorm1d(config.trunk_width))
                elif config.use_layer_norm:
                    layers.append(nn.LayerNorm(config.trunk_width))
                layers.append(self.activation)

            # Output layer
            layers.append(nn.Linear(config.trunk_width, config.latent_dim))

            self.network = nn.Sequential(*layers)

        def forward(self, y: torch.Tensor) -> torch.Tensor:
            """
            Forward pass through trunk network.

            Args:
                y: Evaluation coordinates [batch, num_points, coordinate_dim]

            Returns:
                Trunk network output [batch, num_points, latent_dim]
            """
            # Handle different input shapes
            if len(y.shape) == 3:
                # Reshape for processing: [batch, num_points, coord_dim] -> [batch*num_points, coord_dim]
                batch_size, num_points, coord_dim = y.shape
                y_flat = y.view(-1, coord_dim)

                # Process through network
                trunk_output = self.network(y_flat)

                # Reshape back: [batch*num_points, latent_dim] -> [batch, num_points, latent_dim]
                trunk_output = trunk_output.view(batch_size, num_points, self.config.latent_dim)
            else:
                # Single point evaluation
                trunk_output = self.network(y)

            return trunk_output

    class BiasNetwork(nn.Module):
        """Bias network for DeepONet (optional)."""

        def __init__(self, config: DeepONetConfig):
            """
            Initialize bias network.

            Args:
                config: DeepONet configuration
            """
            super().__init__()
            self.config = config

            # Simple bias network
            layers = []
            layers.append(nn.Linear(config.coordinate_dim, config.trunk_width // 2))
            layers.append(nn.Tanh())

            for _ in range(config.bias_depth - 2):
                layers.append(nn.Linear(config.trunk_width // 2, config.trunk_width // 2))
                layers.append(nn.Tanh())

            layers.append(nn.Linear(config.trunk_width // 2, 1))

            self.network = nn.Sequential(*layers)

        def forward(self, y: torch.Tensor) -> torch.Tensor:
            """
            Forward pass through bias network.

            Args:
                y: Evaluation coordinates

            Returns:
                Bias values
            """
            return self.network(y)

    class DeepONet(BaseNeuralOperator):
        """
        Deep Operator Network for MFG parameter-to-solution mapping.

        DeepONet learns operators using a branch-trunk architecture:
        - Branch Network: Encodes parameter functions
        - Trunk Network: Encodes evaluation coordinates
        - Operator: G(u)(y) = Σᵢ bᵢ(u) * tᵢ(y) + b₀(y)

        Key Features:
        - Universal operator approximation capability
        - Flexible input/output dimensions
        - Multi-fidelity learning support
        - Efficient evaluation at arbitrary coordinates

        Mathematical Framework:
        - Input Function: u(x) ∈ L²(D) (parameter function)
        - Output Function: G(u)(y) ∈ ℝ (solution at coordinate y)
        - Operator: G: L²(D) → L²(D') learned via neural networks
        """

        def __init__(self, config: DeepONetConfig):
            """
            Initialize DeepONet.

            Args:
                config: DeepONet configuration
            """
            super().__init__(config)
            self.config: DeepONetConfig = config

            self._build_network()

        def _build_network(self) -> None:
            """Build the DeepONet architecture."""
            # Branch network (encodes parameter functions)
            self.branch_net = BranchNetwork(self.config)

            # Trunk network (encodes evaluation coordinates)
            self.trunk_net = TrunkNetwork(self.config)

            # Bias network (optional)
            if self.config.use_bias_net:
                self.bias_net = BiasNetwork(self.config)

            # Move to device
            self.to(self.device)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass compatible with base class.

            Args:
                x: Combined input [batch, total_input_dim]
                   First sensor_points values are parameter function
                   Remaining values are flattened coordinates

            Returns:
                Predicted solutions [batch, output_dim]
            """
            batch_size = x.size(0)

            # Split input into branch and trunk components
            branch_input = x[:, : self.config.sensor_points]  # [batch, sensor_points]

            # Remaining input should be coordinates
            coord_input = x[:, self.config.sensor_points :]  # [batch, remaining]

            # Reshape coordinates for trunk network
            num_points = coord_input.size(1) // self.config.coordinate_dim
            trunk_input = coord_input.view(
                batch_size, num_points, self.config.coordinate_dim
            )  # [batch, num_points, coordinate_dim]

            # Forward pass
            output = self.forward_operator(branch_input, trunk_input)
            return output.view(batch_size, -1)  # [batch, output_dim]

        def forward_operator(self, branch_input: torch.Tensor, trunk_input: torch.Tensor) -> torch.Tensor:
            """
            Forward pass through the operator (renamed to avoid conflict).

            Args:
                branch_input: Parameter function values [batch, sensor_points]
                trunk_input: Evaluation coordinates [batch, num_points, coordinate_dim]

            Returns:
                Predicted function values [batch, num_points]
            """
            # Branch network: encode parameter function
            branch_output = self.branch_net(branch_input)  # [batch, latent_dim]

            # Trunk network: encode evaluation coordinates
            trunk_output = self.trunk_net(trunk_input)  # [batch, num_points, latent_dim]

            # Dot product between branch and trunk outputs
            branch_expanded = branch_output.unsqueeze(1)  # [batch, 1, latent_dim]

            # Element-wise multiplication and sum over latent dimension
            operator_output = torch.sum(branch_expanded * trunk_output, dim=-1)  # [batch, num_points]

            # Add bias if enabled
            if self.config.use_bias_net and hasattr(self, "bias_net"):
                bias_output = self.bias_net(trunk_input).squeeze(-1)  # [batch, num_points]
                operator_output = operator_output + bias_output

            return operator_output

        def evaluate_at_coordinates(self, parameter_function: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
            """
            Evaluate operator at specific coordinates.

            Args:
                parameter_function: Parameter function values [batch, sensor_points]
                coordinates: Evaluation coordinates [batch, num_points, coordinate_dim]

            Returns:
                Function values at coordinates [batch, num_points]
            """
            self.eval()
            with torch.no_grad():
                return self.forward_operator(parameter_function, coordinates)

        def train_deeponet(
            self,
            branch_data: torch.Tensor,
            trunk_data: torch.Tensor,
            solution_data: torch.Tensor,
            validation_split: float = 0.2,
        ) -> None:
            """
            Train DeepONet on operator data.

            Args:
                branch_data: Parameter function data [num_samples, sensor_points]
                trunk_data: Coordinate data [num_samples, num_points, coordinate_dim]
                solution_data: Solution data [num_samples, num_points]
                validation_split: Fraction for validation set
            """
            from torch.utils.data import TensorDataset

            # Combine inputs for compatibility with base class
            # Flatten trunk coordinates
            batch_size, _num_points, _coord_dim = trunk_data.shape
            trunk_flat = trunk_data.view(batch_size, -1)

            # Combine branch and trunk data
            combined_input = torch.cat([branch_data, trunk_flat], dim=1)
            solution_flat = solution_data.view(batch_size, -1)

            # Create dataset
            dataset = TensorDataset(combined_input, solution_flat)

            # Split into train/validation
            val_size = int(len(dataset) * validation_split)
            train_size = len(dataset) - val_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

            # Train the operator
            self.train_operator(train_dataset, val_dataset)

        def multi_fidelity_training(
            self,
            low_fidelity_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            high_fidelity_data: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            fidelity_weight: float = 0.1,
        ) -> None:
            """
            Train DeepONet with multi-fidelity data.

            Args:
                low_fidelity_data: (branch, trunk, solution) low-fidelity data
                high_fidelity_data: (branch, trunk, solution) high-fidelity data
                fidelity_weight: Weight for low-fidelity data in loss
            """
            # Combine datasets with different weights
            low_branch, low_trunk, low_solution = low_fidelity_data
            high_branch, high_trunk, high_solution = high_fidelity_data

            # Create weighted dataset (implementation depends on specific needs)
            # This is a simplified version - full implementation would require
            # custom loss functions and training loops

            # For now, combine datasets
            all_branch = torch.cat([low_branch, high_branch], dim=0)
            all_trunk = torch.cat([low_trunk, high_trunk], dim=0)
            all_solution = torch.cat([low_solution, high_solution], dim=0)

            self.train_deeponet(all_branch, all_trunk, all_solution)

else:
    # Placeholder classes when PyTorch is not available
    class DeepONetConfig:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("DeepONet requires PyTorch")

    class DeepONet:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("DeepONet requires PyTorch")
