"""
Fourier Neural Operators (FNO) for MFG Problems.

This module implements Fourier Neural Operators for learning parameter-to-solution
mappings in Mean Field Games. FNOs leverage spectral methods in Fourier space
to efficiently learn operators that map problem parameters to solutions.

Mathematical Framework:
- Fourier Transform: Convert spatial functions to frequency domain
- Spectral Convolution: Learn in Fourier space for global receptive field
- Inverse Transform: Map back to spatial domain
- Parameter-to-Solution: G: θ → u where θ are problem parameters, u is MFG solution

Key Advantages:
- Resolution Invariant: Train on one resolution, evaluate on any resolution
- Global Receptive Field: Fourier transforms capture global dependencies
- Efficient Training: Spectral methods scale well with problem size
- Fast Evaluation: Parameter-to-solution mapping in milliseconds

Applications:
- Parameter Studies: Rapid exploration of parameter spaces
- Real-Time Control: Fast MFG evaluation for dynamic applications
- Uncertainty Quantification: Monte Carlo over parameter distributions
- Multi-Query Optimization: Efficient gradient-based optimization

References:
- Li et al. "Fourier Neural Operator for Parametric Partial Differential Equations" (2021)
- Lu et al. "Learning nonlinear operators via DeepONet based on the universal approximation theorem" (2021)
"""

from __future__ import annotations

from dataclasses import dataclass

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
    from .base_operator import BaseNeuralOperator, OperatorConfig

    @dataclass
    class FNOConfig(OperatorConfig):
        """Configuration for Fourier Neural Operators."""

        # FNO-specific parameters
        modes: int = 16  # Number of Fourier modes to keep
        width: int = 64  # Channel width
        num_layers: int = 4  # Number of FNO layers

        # Architecture parameters
        lifting_size: int = 256  # Lifting layer size
        projection_size: int = 128  # Projection layer size

        # Spatial resolution
        input_resolution: int = 64  # Input spatial resolution
        output_resolution: int = 64  # Output spatial resolution

        # Parameter embedding
        param_embed_dim: int = 32  # Parameter embedding dimension

        # Fourier layer configuration
        use_complex_weights: bool = True  # Use complex-valued weights
        fourier_activation: str = "gelu"  # Activation in Fourier layers

        def __post_init__(self):
            """Set FNO-specific defaults."""
            super().__post_init__()
            self.operator_type = "fno"

    class SpectralConv1d(nn.Module):
        """1D Spectral convolution layer for FNO."""

        def __init__(self, in_channels: int, out_channels: int, modes: int):
            """
            Initialize spectral convolution layer.

            Args:
                in_channels: Input channels
                out_channels: Output channels
                modes: Number of Fourier modes
            """
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.modes = modes

            # Fourier weights (complex-valued)
            scale = 1 / (in_channels * out_channels)
            self.weights = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes, dtype=torch.complex64))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass through spectral convolution.

            Args:
                x: Input tensor [batch, channels, spatial_dim]

            Returns:
                Output after spectral convolution
            """
            batch_size = x.shape[0]

            # Fourier transform to frequency domain
            x_ft = torch.fft.rfft(x)

            # Extract the relevant modes
            out_ft = torch.zeros(
                batch_size, self.out_channels, x.size(-1) // 2 + 1, dtype=torch.complex64, device=x.device
            )

            # Multiply relevant Fourier modes
            out_ft[:, :, : self.modes] = torch.einsum("bix,iox->box", x_ft[:, :, : self.modes], self.weights)

            # Inverse Fourier transform back to spatial domain
            x = torch.fft.irfft(out_ft, n=x.size(-1))
            return x

    class SpectralConv2d(nn.Module):
        """2D Spectral convolution layer for FNO."""

        def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
            """
            Initialize 2D spectral convolution layer.

            Args:
                in_channels: Input channels
                out_channels: Output channels
                modes1: Number of Fourier modes in first dimension
                modes2: Number of Fourier modes in second dimension
            """
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.modes1 = modes1
            self.modes2 = modes2

            scale = 1 / (in_channels * out_channels)
            self.weights1 = nn.Parameter(
                scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.complex64)
            )
            self.weights2 = nn.Parameter(
                scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.complex64)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass through 2D spectral convolution.

            Args:
                x: Input tensor [batch, channels, height, width]

            Returns:
                Output after spectral convolution
            """
            batch_size = x.shape[0]

            # Fourier transform to frequency domain
            x_ft = torch.fft.rfft2(x)

            # Extract spatial dimensions
            out_ft = torch.zeros(
                batch_size, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.complex64, device=x.device
            )

            # Multiply relevant Fourier modes
            out_ft[:, :, : self.modes1, : self.modes2] = torch.einsum(
                "bixy,ioyx->boxy", x_ft[:, :, : self.modes1, : self.modes2], self.weights1
            )

            out_ft[:, :, -self.modes1 :, : self.modes2] = torch.einsum(
                "bixy,ioyx->boxy", x_ft[:, :, -self.modes1 :, : self.modes2], self.weights2
            )

            # Inverse Fourier transform back to spatial domain
            x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
            return x

    class FNOLayer(nn.Module):
        """Fourier Neural Operator layer."""

        def __init__(self, width: int, modes: int, spatial_dim: int = 1):
            """
            Initialize FNO layer.

            Args:
                width: Channel width
                modes: Number of Fourier modes
                spatial_dim: Spatial dimension (1 or 2)
            """
            super().__init__()
            self.width = width
            self.modes = modes
            self.spatial_dim = spatial_dim

            # Spectral convolution
            if spatial_dim == 1:
                self.fourier_layer = SpectralConv1d(width, width, modes)
            elif spatial_dim == 2:
                self.fourier_layer = SpectralConv2d(width, width, modes, modes)
            else:
                raise ValueError(f"Spatial dimension {spatial_dim} not supported")

            # Local convolution (skip connection)
            if spatial_dim == 1:
                self.local_layer = nn.Conv1d(width, width, 1)
            elif spatial_dim == 2:
                self.local_layer = nn.Conv2d(width, width, 1)

            # Activation
            self.activation = nn.GELU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass through FNO layer.

            Args:
                x: Input tensor

            Returns:
                Output after FNO layer processing
            """
            # Spectral convolution
            x1 = self.fourier_layer(x)

            # Local convolution (skip connection)
            x2 = self.local_layer(x)

            # Combine and activate
            out = self.activation(x1 + x2)
            return out

    class FourierNeuralOperator(BaseNeuralOperator):
        """
        Fourier Neural Operator for MFG parameter-to-solution mapping.

        FNO learns operators in Fourier space, enabling efficient learning of
        parameter-to-solution mappings for Mean Field Games. The spectral approach
        provides global receptive fields and resolution invariance.

        Architecture:
        1. Parameter Embedding: Map problem parameters to high-dimensional space
        2. Lifting: Project to FNO channel space
        3. FNO Layers: Series of Fourier convolution layers
        4. Projection: Map back to solution space
        5. Spatial Reconstruction: Reshape to spatial solution format

        Mathematical Framework:
        - Input: θ ∈ ℝᵈ (problem parameters)
        - Output: u(x) ∈ ℝᴺ (discretized MFG solution)
        - Operator: u = G(θ) learned via FNO
        """

        def __init__(self, config: FNOConfig):
            """
            Initialize Fourier Neural Operator.

            Args:
                config: FNO configuration
            """
            super().__init__(config)
            self.config: FNOConfig = config

            # Determine spatial dimension from output shape
            self.spatial_dim = 1 if config.output_resolution**2 != config.output_dim else 2

            self._build_network()

        def _build_network(self) -> None:
            """Build the FNO architecture."""
            # Parameter embedding layer
            self.param_embedding = nn.Sequential(
                nn.Linear(self.config.input_dim, self.config.param_embed_dim),
                nn.GELU(),
                nn.Linear(self.config.param_embed_dim, self.config.param_embed_dim),
            )

            # Lifting layer: map parameters to spatial function
            if self.spatial_dim == 1:
                self.lifting = nn.Linear(
                    self.config.param_embed_dim + 1,  # +1 for spatial coordinate
                    self.config.width,
                )
            else:  # 2D
                self.config.output_resolution * self.config.output_resolution
                self.lifting = nn.Linear(
                    self.config.param_embed_dim + 2,  # +2 for spatial coordinates
                    self.config.width,
                )

            # FNO layers
            self.fno_layers = nn.ModuleList(
                [
                    FNOLayer(self.config.width, self.config.modes, self.spatial_dim)
                    for _ in range(self.config.num_layers)
                ]
            )

            # Projection layers
            self.projection = nn.Sequential(
                nn.Linear(self.config.width, self.config.projection_size),
                nn.GELU(),
                nn.Linear(self.config.projection_size, 1),  # Output single value per spatial point
            )

            # Move to device
            self.to(self.device)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass through FNO.

            Args:
                x: Input parameters [batch_size, input_dim]

            Returns:
                Predicted solutions [batch_size, output_dim]
            """
            batch_size = x.size(0)

            # Parameter embedding
            param_embed = self.param_embedding(x)  # [batch, param_embed_dim]

            # Create spatial grid
            if self.spatial_dim == 1:
                # 1D spatial grid
                grid = torch.linspace(0, 1, self.config.output_resolution, device=x.device)
                grid = grid.unsqueeze(0).repeat(batch_size, 1)  # [batch, resolution]

                # Combine parameters with spatial coordinates
                param_expand = param_embed.unsqueeze(1).repeat(1, self.config.output_resolution, 1)
                grid_expand = grid.unsqueeze(-1)

                # Lifting: [batch, resolution, param_embed_dim + 1] -> [batch, resolution, width]
                lifted_input = torch.cat([param_expand, grid_expand], dim=-1)
                u = self.lifting(lifted_input)  # [batch, resolution, width]
                u = u.permute(0, 2, 1)  # [batch, width, resolution]

            else:  # 2D
                # 2D spatial grid
                h, w = self.config.output_resolution, self.config.output_resolution
                y = torch.linspace(0, 1, h, device=x.device)
                x_coord = torch.linspace(0, 1, w, device=x.device)
                yy, xx = torch.meshgrid(y, x_coord, indexing="ij")
                grid = torch.stack([yy, xx], dim=-1)  # [h, w, 2]
                grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # [batch, h, w, 2]

                # Combine parameters with spatial coordinates
                param_expand = param_embed.view(batch_size, 1, 1, -1).repeat(1, h, w, 1)

                # Lifting: [batch, h, w, param_embed_dim + 2] -> [batch, h, w, width]
                lifted_input = torch.cat([param_expand, grid], dim=-1)
                u = self.lifting(lifted_input)  # [batch, h, w, width]
                u = u.permute(0, 3, 1, 2)  # [batch, width, h, w]

            # FNO layers
            for fno_layer in self.fno_layers:
                u = fno_layer(u)

            # Project back to solution space
            if self.spatial_dim == 1:
                u = u.permute(0, 2, 1)  # [batch, resolution, width]
                output = self.projection(u).squeeze(-1)  # [batch, resolution]
            else:  # 2D
                u = u.permute(0, 2, 3, 1)  # [batch, h, w, width]
                output = self.projection(u).squeeze(-1)  # [batch, h, w]
                output = output.view(batch_size, -1)  # [batch, h*w]

            return output

        def train_fno(
            self, parameter_data: torch.Tensor, solution_data: torch.Tensor, validation_split: float = 0.2
        ) -> None:
            """
            Train FNO on parameter-solution pairs.

            Args:
                parameter_data: Training parameters [num_samples, input_dim]
                solution_data: Training solutions [num_samples, output_dim]
                validation_split: Fraction for validation set
            """
            from torch.utils.data import TensorDataset

            # Create dataset
            dataset = TensorDataset(parameter_data, solution_data)

            # Split into train/validation
            val_size = int(len(dataset) * validation_split)
            train_size = len(dataset) - val_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

            # Train the operator
            self.train_operator(train_dataset, val_dataset)

        def evaluate_speedup(
            self, test_parameters: torch.Tensor, traditional_solver_func, num_evaluations: int = 100
        ) -> dict:
            """
            Evaluate speedup compared to traditional solver.

            Args:
                test_parameters: Test parameters for evaluation
                traditional_solver_func: Function to solve MFG traditionally
                num_evaluations: Number of evaluations for timing

            Returns:
                Speedup analysis results
            """
            import time

            # FNO evaluation time
            start_time = time.time()
            for i in range(num_evaluations):
                with torch.no_grad():
                    _ = self.predict(test_parameters[i : i + 1])
            fno_time = time.time() - start_time

            # Traditional solver time (sample a few)
            sample_size = min(10, num_evaluations)
            start_time = time.time()
            for i in range(sample_size):
                _ = traditional_solver_func(test_parameters[i])
            traditional_time = (time.time() - start_time) * (num_evaluations / sample_size)

            # Calculate speedup
            speedup = traditional_time / fno_time

            return {
                "fno_time": fno_time,
                "traditional_time": traditional_time,
                "speedup_factor": speedup,
                "evaluations_per_second": num_evaluations / fno_time,
            }

else:
    # Placeholder classes when PyTorch is not available
    class FNOConfig:
        def __init__(self, *args, **kwargs):
            raise ImportError("FNO requires PyTorch")

    class FourierNeuralOperator:
        def __init__(self, *args, **kwargs):
            raise ImportError("FNO requires PyTorch")
