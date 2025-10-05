"""
PyTorch utility functions for MFG_PDE accelerated solvers.

This module provides common utilities for PyTorch-based implementations,
including numerical operations, device management, and KDE.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Check PyTorch availability
try:
    import torch

    HAS_TORCH = True
    HAS_CUDA = torch.cuda.is_available()
    HAS_MPS = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
except ImportError:
    torch = None
    HAS_TORCH = False
    HAS_CUDA = False
    HAS_MPS = False


def ensure_torch_available():
    """Ensure PyTorch is available, raise error if not."""
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for accelerated solvers. Install with: pip install torch")


def get_default_device() -> str:
    """Get default PyTorch device (cuda > mps > cpu)."""
    if HAS_CUDA:
        return "cuda"
    elif HAS_MPS:
        return "mps"
    else:
        return "cpu"


def to_tensor(array: np.ndarray | Any, device: str | None = None, dtype: torch.dtype | None = None) -> torch.Tensor:
    """
    Convert numpy array to PyTorch tensor.

    Args:
        array: Input array
        device: Target device (auto-detected if None)
        dtype: Target dtype (float32 if None)

    Returns:
        PyTorch tensor
    """
    ensure_torch_available()

    if device is None:
        device = get_default_device()

    if dtype is None:
        dtype = torch.float32

    if isinstance(array, torch.Tensor):
        return array.to(device=device, dtype=dtype)

    return torch.tensor(array, device=device, dtype=dtype)


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert PyTorch tensor to numpy array.

    Args:
        tensor: Input tensor

    Returns:
        NumPy array
    """
    ensure_torch_available()

    if isinstance(tensor, np.ndarray):
        return tensor

    return tensor.detach().cpu().numpy()


class GaussianKDE:
    """
    PyTorch-based Gaussian Kernel Density Estimation.

    Replacement for scipy.stats.gaussian_kde with GPU acceleration.
    Compatible with CUDA, MPS (Apple Silicon), and CPU.

    Example:
        >>> particles = np.array([0.1, 0.2, 0.15, 0.3])
        >>> kde = GaussianKDE(particles, bw_method=0.1, device="cuda")
        >>> x_eval = np.linspace(0, 1, 100)
        >>> density = kde(x_eval)
    """

    def __init__(
        self,
        dataset: np.ndarray | torch.Tensor,
        bw_method: float | str = "scott",
        device: str | None = None,
    ):
        """
        Initialize Gaussian KDE.

        Args:
            dataset: 1D array of data points
            bw_method: Bandwidth selection method:
                - float: Fixed bandwidth
                - "scott": Scott's rule (default)
                - "silverman": Silverman's rule
            device: PyTorch device (auto-detected if None)
        """
        ensure_torch_available()

        self.device = device if device is not None else get_default_device()

        # Convert dataset to tensor
        if isinstance(dataset, np.ndarray):
            self.dataset = torch.tensor(dataset, device=self.device, dtype=torch.float32)
        else:
            self.dataset = dataset.to(device=self.device, dtype=torch.float32)

        # Ensure 1D
        if self.dataset.ndim != 1:
            raise ValueError(f"Dataset must be 1D, got shape {self.dataset.shape}")

        self.n_points = len(self.dataset)

        # Compute bandwidth
        self.bw_method = bw_method
        self.bandwidth = self._compute_bandwidth()

    def _compute_bandwidth(self) -> float:
        """Compute bandwidth using specified method."""
        if isinstance(self.bw_method, int | float):
            return float(self.bw_method)

        # Compute standard deviation
        std = torch.std(self.dataset, unbiased=True).item()

        if self.bw_method == "scott":
            # Scott's rule: n^(-1/5) * std
            return float(std * (self.n_points ** (-1.0 / 5.0)))

        elif self.bw_method == "silverman":
            # Silverman's rule: (n * 3/4)^(-1/5) * std
            return float(std * ((self.n_points * 3.0 / 4.0) ** (-1.0 / 5.0)))

        else:
            raise ValueError(f"Unknown bw_method: {self.bw_method}")

    def __call__(self, x_eval: np.ndarray | torch.Tensor) -> np.ndarray:
        """
        Evaluate KDE at given points.

        Args:
            x_eval: Points to evaluate density at

        Returns:
            Density values (NumPy array)
        """
        # Convert evaluation points to tensor
        if isinstance(x_eval, np.ndarray):
            x_tensor = torch.tensor(x_eval, device=self.device, dtype=torch.float32)
            return_numpy = True
        else:
            x_tensor = x_eval.to(device=self.device, dtype=torch.float32)
            return_numpy = False

        # Ensure 1D
        if x_tensor.ndim != 1:
            raise ValueError(f"Evaluation points must be 1D, got shape {x_tensor.shape}")

        # Compute Gaussian kernel:
        # K(x, x_i) = (1 / (sqrt(2π) * h)) * exp(-0.5 * ((x - x_i) / h)^2)
        #
        # For numerical stability and GPU efficiency, we compute in batches:
        # density(x) = (1/n) * sum_i K(x, x_i)

        # Reshape for broadcasting: (n_eval, 1) - (1, n_points)
        x_eval_expanded = x_tensor.unsqueeze(1)  # Shape: (n_eval, 1)
        dataset_expanded = self.dataset.unsqueeze(0)  # Shape: (1, n_points)

        # Compute differences: (n_eval, n_points)
        diff = x_eval_expanded - dataset_expanded

        # Compute Gaussian kernel
        normalization = 1.0 / (np.sqrt(2 * np.pi) * self.bandwidth)
        exponent = -0.5 * (diff / self.bandwidth) ** 2
        kernel_values = normalization * torch.exp(exponent)

        # Sum over data points and average
        density = torch.mean(kernel_values, dim=1)

        # Return as NumPy if input was NumPy
        if return_numpy:
            return density.detach().cpu().numpy()
        else:
            return density


def tridiagonal_solve(
    a: np.ndarray | torch.Tensor,
    b: np.ndarray | torch.Tensor,
    c: np.ndarray | torch.Tensor,
    d: np.ndarray | torch.Tensor,
    device: str | None = None,
) -> np.ndarray:
    """
    Solve tridiagonal system using Thomas algorithm (PyTorch implementation).

    Args:
        a: Lower diagonal (size n-1)
        b: Main diagonal (size n)
        c: Upper diagonal (size n-1)
        d: Right-hand side (size n)
        device: PyTorch device (auto-detected if None)

    Returns:
        Solution vector (NumPy array)
    """
    ensure_torch_available()

    if device is None:
        device = get_default_device()

    # Convert to tensors
    a_t = to_tensor(a, device=device)
    b_t = to_tensor(b, device=device)
    c_t = to_tensor(c, device=device)
    d_t = to_tensor(d, device=device)

    n = len(b_t)

    # Pad a and c to size n for easier indexing
    a_padded = torch.cat([torch.zeros(1, device=device, dtype=a_t.dtype), a_t])
    c_padded = torch.cat([c_t, torch.zeros(1, device=device, dtype=c_t.dtype)])

    # Clone b and d to avoid modifying inputs
    b_work = b_t.clone()
    d_work = d_t.clone()

    # Forward elimination
    for i in range(1, n):
        w = a_padded[i] / b_work[i - 1]
        b_work[i] = b_work[i] - w * c_padded[i - 1]
        d_work[i] = d_work[i] - w * d_work[i - 1]

    # Back substitution
    x = torch.zeros(n, device=device, dtype=d_t.dtype)
    x[n - 1] = d_work[n - 1] / b_work[n - 1]

    for i in range(n - 2, -1, -1):
        x[i] = (d_work[i] - c_padded[i] * x[i + 1]) / b_work[i]

    return to_numpy(x)


__all__ = [
    "HAS_CUDA",
    "HAS_MPS",
    "HAS_TORCH",
    "GaussianKDE",
    "ensure_torch_available",
    "get_default_device",
    "to_numpy",
    "to_tensor",
    "tridiagonal_solve",
]
