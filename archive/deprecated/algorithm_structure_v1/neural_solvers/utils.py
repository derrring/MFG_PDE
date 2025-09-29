"""
Utility functions for Physics-Informed Neural Networks.

This module provides common utilities for PINN implementations including
automatic differentiation helpers, point sampling strategies, and neural
network analysis tools.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

# PyTorch imports with fallback
try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


def auto_differentiation(
    network: nn.Module,
    inputs: torch.Tensor,
    output_index: int = 0,
    derivative_order: int = 1,
    input_index: int = 0,
) -> torch.Tensor:
    """
    Compute derivatives using automatic differentiation.

    Args:
        network: Neural network to differentiate
        inputs: Input tensor [N, input_dim]
        output_index: Index of output to differentiate (for multi-output networks)
        derivative_order: Order of derivative (1 or 2)
        input_index: Index of input variable to differentiate with respect to

    Returns:
        Derivative tensor [N, 1]
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for automatic differentiation")

    # Ensure input requires gradients
    inputs_copy = inputs.clone().detach()
    inputs_copy.requires_grad_(True)

    # Forward pass
    outputs = network(inputs_copy)

    # Handle multi-output networks
    if len(outputs.shape) > 1 and outputs.shape[1] > 1:
        output = outputs[:, output_index : output_index + 1]
    else:
        output = outputs

    # First derivative
    grad = torch.autograd.grad(
        outputs=output,
        inputs=inputs_copy,
        grad_outputs=torch.ones_like(output),
        create_graph=derivative_order > 1,
        retain_graph=derivative_order > 1,
    )[0]

    if derivative_order == 1:
        return grad[:, input_index : input_index + 1]

    elif derivative_order == 2:
        # Second derivative
        first_deriv = grad[:, input_index : input_index + 1]
        second_deriv = torch.autograd.grad(
            outputs=first_deriv,
            inputs=inputs_copy,
            grad_outputs=torch.ones_like(first_deriv),
            create_graph=True,
            retain_graph=True,
        )[0]
        return second_deriv[:, input_index : input_index + 1]

    else:
        raise ValueError(f"Derivative order {derivative_order} not supported")


def compute_gradients(u: torch.Tensor, coords: torch.Tensor, create_graph: bool = True) -> list[torch.Tensor]:
    """
    Compute all first-order partial derivatives of u with respect to coordinates.

    Args:
        u: Function values [N, 1]
        coords: Coordinate tensor [N, d] where d is dimension
        create_graph: Whether to create computational graph for higher-order derivatives

    Returns:
        List of gradient tensors, one for each coordinate dimension
    """
    if not coords.requires_grad:
        coords = coords.clone().detach().requires_grad_(True)

    gradients = []
    for i in range(coords.shape[1]):
        grad = torch.autograd.grad(
            outputs=u, inputs=coords, grad_outputs=torch.ones_like(u), create_graph=create_graph, retain_graph=True
        )[0]
        gradients.append(grad[:, i : i + 1])

    return gradients


def sample_points(
    domain_bounds: list[tuple[float, float]],
    n_points: int,
    sampling_method: str = "random",
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Sample points from a multi-dimensional domain.

    Args:
        domain_bounds: List of (min, max) pairs for each dimension
        n_points: Number of points to sample
        sampling_method: "random", "uniform", "sobol", or "halton"
        device: PyTorch device
        dtype: PyTorch data type

    Returns:
        Sampled points [n_points, n_dims]
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for point sampling")

    n_dims = len(domain_bounds)

    if sampling_method == "random":
        # Standard random sampling
        points = torch.rand(n_points, n_dims, device=device, dtype=dtype)

    elif sampling_method == "uniform":
        # Uniform grid sampling
        n_per_dim = int(np.ceil(n_points ** (1.0 / n_dims)))
        coords_1d = [torch.linspace(0, 1, n_per_dim, device=device, dtype=dtype) for _ in range(n_dims)]

        # Create meshgrid
        mesh = torch.meshgrid(coords_1d, indexing="ij")
        points = torch.stack([m.flatten() for m in mesh], dim=-1)

        # Subsample if needed
        if len(points) > n_points:
            indices = torch.randperm(len(points))[:n_points]
            points = points[indices]

    elif sampling_method == "sobol":
        # Sobol sequence (requires scipy)
        try:
            from scipy.stats import qmc

            sobol_sampler = qmc.Sobol(d=n_dims, scramble=True)
            points_np = sobol_sampler.random(n_points)
            points = torch.from_numpy(points_np).to(device=device, dtype=dtype)
        except ImportError:
            warnings.warn("SciPy not available. Falling back to random sampling.")
            points = torch.rand(n_points, n_dims, device=device, dtype=dtype)

    else:
        raise ValueError(f"Unknown sampling method: {sampling_method}")

    # Scale to domain bounds
    for i, (min_val, max_val) in enumerate(domain_bounds):
        points[:, i] = points[:, i] * (max_val - min_val) + min_val

    return points


def sample_boundary_points(
    domain_bounds: list[tuple[float, float]], n_points: int, device: str = "cpu", dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample points on domain boundary.

    Args:
        domain_bounds: List of (min, max) pairs for each dimension
        n_points: Total number of boundary points to sample
        device: PyTorch device
        dtype: PyTorch data type

    Returns:
        Tuple of (boundary_points, boundary_normals)
    """
    n_dims = len(domain_bounds)
    points_per_face = n_points // (2 * n_dims)

    boundary_points = []
    boundary_normals = []

    for dim in range(n_dims):
        for face in [0, 1]:  # min and max face for this dimension
            # Sample points on this face
            n_face_points = points_per_face
            face_points = torch.rand(n_face_points, n_dims, device=device, dtype=dtype)

            # Set the boundary coordinate
            boundary_value = domain_bounds[dim][face]
            face_points[:, dim] = boundary_value

            # Scale other coordinates to domain bounds
            for other_dim in range(n_dims):
                if other_dim != dim:
                    min_val, max_val = domain_bounds[other_dim]
                    face_points[:, other_dim] = face_points[:, other_dim] * (max_val - min_val) + min_val

            # Create normal vectors
            normal = torch.zeros(n_face_points, n_dims, device=device, dtype=dtype)
            normal[:, dim] = 1.0 if face == 1 else -1.0

            boundary_points.append(face_points)
            boundary_normals.append(normal)

    # Concatenate all boundary points
    all_boundary_points = torch.cat(boundary_points, dim=0)
    all_boundary_normals = torch.cat(boundary_normals, dim=0)

    return all_boundary_points, all_boundary_normals


def create_time_space_grid(
    time_bounds: tuple[float, float], space_bounds: list[tuple[float, float]], nt: int, nx: int | list[int]
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """
    Create structured grid in time-space domain.

    Args:
        time_bounds: (t_min, t_max)
        space_bounds: List of (x_min, x_max) for each spatial dimension
        nt: Number of time points
        nx: Number of spatial points (int for uniform, list for per-dimension)

    Returns:
        Tuple of (time_grid, space_grids)
    """
    t_min, t_max = time_bounds
    t_grid = torch.linspace(t_min, t_max, nt)

    space_grids = []
    if isinstance(nx, int):
        nx_list = [nx] * len(space_bounds)
    else:
        nx_list = nx

    for _i, ((x_min, x_max), nx_dim) in enumerate(zip(space_bounds, nx_list, strict=False)):
        x_grid = torch.linspace(x_min, x_max, nx_dim)
        space_grids.append(x_grid)

    return t_grid, space_grids


def compute_pde_residual_statistics(residuals: torch.Tensor) -> dict[str, float]:
    """
    Compute statistics of PDE residuals for monitoring training.

    Args:
        residuals: PDE residual tensor [N, 1] or [N]

    Returns:
        Dictionary of residual statistics
    """
    residuals_flat = residuals.flatten()
    abs_residuals = torch.abs(residuals_flat)

    stats = {
        "mean_absolute": float(torch.mean(abs_residuals)),
        "max_absolute": float(torch.max(abs_residuals)),
        "std_absolute": float(torch.std(abs_residuals)),
        "l2_norm": float(torch.norm(residuals_flat)),
        "relative_l2": float(torch.norm(residuals_flat) / (torch.norm(residuals_flat) + 1e-12)),
    }

    # Percentiles
    sorted_abs = torch.sort(abs_residuals)[0]
    n = len(sorted_abs)
    stats["median_absolute"] = float(sorted_abs[n // 2])
    stats["p90_absolute"] = float(sorted_abs[int(0.9 * n)])
    stats["p99_absolute"] = float(sorted_abs[int(0.99 * n)])

    return stats


def monitor_training_progress(
    epoch: int, losses: dict[str, float], log_interval: int = 100, detailed: bool = False
) -> None:
    """
    Monitor and log training progress.

    Args:
        epoch: Current epoch number
        losses: Dictionary of loss values
        log_interval: Logging interval
        detailed: Whether to show detailed loss breakdown
    """
    if epoch % log_interval == 0:
        total_loss = losses.get("total_loss", 0.0)
        print(f"Epoch {epoch:5d}: Total Loss = {total_loss:.6f}")

        if detailed:
            for loss_name, loss_value in losses.items():
                if loss_name != "total_loss":
                    print(f"    {loss_name}: {loss_value:.6f}")
            print()


def neural_network_utils():
    """Collection of neural network utility functions."""

    class NetworkUtils:
        @staticmethod
        def freeze_network(network: nn.Module) -> None:
            """Freeze all parameters in network."""
            for param in network.parameters():
                param.requires_grad = False

        @staticmethod
        def unfreeze_network(network: nn.Module) -> None:
            """Unfreeze all parameters in network."""
            for param in network.parameters():
                param.requires_grad = True

        @staticmethod
        def count_parameters(network: nn.Module) -> dict[str, int]:
            """Count parameters in network."""
            total = sum(p.numel() for p in network.parameters())
            trainable = sum(p.numel() for p in network.parameters() if p.requires_grad)
            return {"total": total, "trainable": trainable}

        @staticmethod
        def get_network_memory_usage(network: nn.Module) -> dict[str, float]:
            """Estimate network memory usage in MB."""
            param_size = sum(p.numel() * p.element_size() for p in network.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in network.buffers())

            return {
                "parameters_mb": param_size / 1024**2,
                "buffers_mb": buffer_size / 1024**2,
                "total_mb": (param_size + buffer_size) / 1024**2,
            }

        @staticmethod
        def check_gradients(network: nn.Module) -> dict[str, bool | float]:
            """Check gradient health."""
            has_gradients = any(p.grad is not None for p in network.parameters())

            if has_gradients:
                grad_norms = [torch.norm(p.grad).item() for p in network.parameters() if p.grad is not None]
                total_grad_norm = sum(norm**2 for norm in grad_norms) ** 0.5
                max_grad_norm = max(grad_norms) if grad_norms else 0.0

                return {
                    "has_gradients": True,
                    "total_grad_norm": total_grad_norm,
                    "max_grad_norm": max_grad_norm,
                    "num_zero_grads": sum(1 for norm in grad_norms if norm == 0.0),
                }
            else:
                return {"has_gradients": False}

    return NetworkUtils()


def create_adaptive_sampler(
    domain_bounds: list[tuple[float, float]],
    initial_points: int = 1000,
    refinement_factor: float = 1.5,
    max_points: int = 10000,
) -> Callable:
    """
    Create adaptive point sampling function.

    Args:
        domain_bounds: Domain boundaries
        initial_points: Initial number of points
        refinement_factor: Factor by which to increase points
        max_points: Maximum number of points

    Returns:
        Sampling function that takes (residual_fn, threshold) -> points
    """

    def adaptive_sample(residual_fn: Callable, threshold: float = 0.1) -> torch.Tensor:
        """
        Adaptively sample points based on residual magnitude.

        Args:
            residual_fn: Function that computes residuals at given points
            threshold: Residual threshold for refinement

        Returns:
            Adaptively sampled points
        """
        current_points = initial_points

        while current_points <= max_points:
            # Sample points
            points = sample_points(domain_bounds, current_points, "sobol")

            # Compute residuals
            with torch.no_grad():
                residuals = residual_fn(points)
                max_residual = torch.max(torch.abs(residuals))

            # Check convergence
            if max_residual < threshold:
                break

            # Increase points
            current_points = int(current_points * refinement_factor)
            if current_points > max_points:
                current_points = max_points
                break

        return points

    return adaptive_sample


def validate_pinn_solution(
    network: nn.Module,
    test_points: torch.Tensor,
    analytical_solution: Callable | None = None,
    pde_residual_fn: Callable | None = None,
) -> dict[str, float]:
    """
    Validate PINN solution quality.

    Args:
        network: Trained PINN network
        test_points: Test points for validation
        analytical_solution: Analytical solution function (if available)
        pde_residual_fn: Function to compute PDE residuals

    Returns:
        Dictionary of validation metrics
    """
    metrics = {}

    with torch.no_grad():
        # Network predictions
        predictions = network(test_points)

        # Solution statistics
        metrics["solution_mean"] = float(torch.mean(predictions))
        metrics["solution_std"] = float(torch.std(predictions))
        metrics["solution_min"] = float(torch.min(predictions))
        metrics["solution_max"] = float(torch.max(predictions))

        # Analytical comparison (if available)
        if analytical_solution is not None:
            if torch.is_tensor(test_points):
                test_np = test_points.cpu().numpy()
            else:
                test_np = test_points

            analytical_values = analytical_solution(test_np)
            if not torch.is_tensor(analytical_values):
                analytical_values = torch.from_numpy(analytical_values).to(predictions.device)

            l2_error = float(torch.norm(predictions.flatten() - analytical_values.flatten()))
            relative_l2 = l2_error / (float(torch.norm(analytical_values.flatten())) + 1e-12)

            metrics["l2_error"] = l2_error
            metrics["relative_l2_error"] = relative_l2
            metrics["max_absolute_error"] = float(
                torch.max(torch.abs(predictions.flatten() - analytical_values.flatten()))
            )

        # PDE residual validation (if available)
        if pde_residual_fn is not None:
            residuals = pde_residual_fn(test_points)
            metrics.update({f"pde_{k}": v for k, v in compute_pde_residual_statistics(residuals).items()})

    return metrics


# Export main utility functions
__all__ = [
    "auto_differentiation",
    "compute_gradients",
    "compute_pde_residual_statistics",
    "create_adaptive_sampler",
    "create_time_space_grid",
    "monitor_training_progress",
    "neural_network_utils",
    "sample_boundary_points",
    "sample_points",
    "validate_pinn_solution",
]
