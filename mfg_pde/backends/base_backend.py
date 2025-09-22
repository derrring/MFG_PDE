"""
Base Backend Interface for MFG_PDE

Defines the abstract interface that all computational backends must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseBackend(ABC):
    """
    Abstract base class for computational backends.

    All backends must implement these methods to provide consistent
    interfaces for MFG computations across different numerical libraries.
    """

    def __init__(self, device: str = "auto", precision: str = "float64", **kwargs):
        """
        Initialize backend with configuration.

        Args:
            device: Device to use ("cpu", "gpu", or "auto")
            precision: Numerical precision ("float32" or "float64")
            **kwargs: Backend-specific options
        """
        self.device = device
        self.precision = precision
        self.config = kwargs
        self._setup_backend()

    @abstractmethod
    def _setup_backend(self):
        """Backend-specific initialization."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name identifier."""

    @property
    @abstractmethod
    def array_module(self):
        """The array module (numpy, jax.numpy, etc.)."""

    # Array Operations
    @abstractmethod
    def array(self, data, dtype=None):
        """Create backend-specific array."""

    @abstractmethod
    def zeros(self, shape, dtype=None):
        """Create array of zeros."""

    @abstractmethod
    def ones(self, shape, dtype=None):
        """Create array of ones."""

    @abstractmethod
    def linspace(self, start, stop, num):
        """Create linearly spaced array."""

    @abstractmethod
    def meshgrid(self, *arrays, indexing="xy"):
        """Create coordinate arrays from arrays."""

    # Mathematical Operations
    @abstractmethod
    def grad(self, func, argnum=0):
        """Compute gradient of function."""

    @abstractmethod
    def trapezoid(self, y, x=None, dx=1.0, axis=-1):
        """Trapezoidal integration."""

    @abstractmethod
    def diff(self, a, n=1, axis=-1):
        """Discrete difference."""

    @abstractmethod
    def interp(self, x, xp, fp):
        """1-D linear interpolation."""

    # Linear Algebra
    @abstractmethod
    def solve(self, A, b):
        """Solve linear system Ax = b."""

    @abstractmethod
    def eig(self, a):
        """Compute eigenvalues and eigenvectors."""

    # Statistics
    @abstractmethod
    def mean(self, a, axis=None):
        """Compute mean along axis."""

    @abstractmethod
    def std(self, a, axis=None):
        """Compute standard deviation."""

    @abstractmethod
    def max(self, a, axis=None):
        """Maximum values along axis."""

    @abstractmethod
    def min(self, a, axis=None):
        """Minimum values along axis."""

    # MFG-Specific Operations
    @abstractmethod
    def compute_hamiltonian(self, x, p, m, problem_params):
        """Compute Hamiltonian H(x, p, m)."""

    @abstractmethod
    def compute_optimal_control(self, x, p, m, problem_params):
        """Compute optimal control a*(x, p, m)."""

    @abstractmethod
    def hjb_step(self, U, M, dt, dx, problem_params):
        """Single Hamilton-Jacobi-Bellman time step."""

    @abstractmethod
    def fpk_step(self, M, U, dt, dx, problem_params):
        """Single Fokker-Planck-Kolmogorov time step."""

    # Performance and Compilation
    def compile_function(self, func, *args, **kwargs):
        """
        Compile function for performance (JIT compilation for JAX).
        Default implementation returns the function unchanged.
        """
        return func

    def vectorize(self, func, signature=None):
        """
        Vectorize function for element-wise operations.
        Default implementation uses numpy's vectorize.
        """
        return np.vectorize(func, signature=signature)

    # Device Management
    def to_device(self, array):
        """Move array to backend's target device."""
        return array  # Default: no-op

    def from_device(self, array):
        """Move array from backend's device to CPU/numpy."""
        return np.asarray(array)

    # Type Conversion
    def to_numpy(self, array) -> np.ndarray:
        """Convert backend array to numpy array."""
        return np.asarray(array)

    def from_numpy(self, array: np.ndarray):
        """Convert numpy array to backend array."""
        return self.array(array)

    # Backend Information
    def get_device_info(self) -> dict:
        """Get information about current device."""
        return {
            "backend": self.name,
            "device": self.device,
            "precision": self.precision,
        }

    def memory_usage(self) -> dict | None:
        """Get memory usage information if available."""
        return None  # Override in specific backends

    # Context Management
    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
