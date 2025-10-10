"""
Automatic Differentiation Backend Selection.

Provides a clean enum-based API for selecting autodiff backends
instead of using multiple boolean flags.
"""

from enum import Enum


class AutoDiffBackend(str, Enum):
    """
    Backend selection for automatic differentiation.

    This enum provides a clean API for selecting which automatic differentiation
    framework to use for gradient computations.

    Attributes:
        NUMPY: Use NumPy with finite differences (no autodiff)
        JAX: Use JAX automatic differentiation
        PYTORCH: Use PyTorch autograd

    Example:
        >>> from mfg_pde.utils.numerical.autodiff import AutoDiffBackend
        >>> config = FunctionalDerivativeConfig(backend=AutoDiffBackend.JAX)
        >>> # Instead of: use_jax=True, use_pytorch=False
    """

    NUMPY = "numpy"  # Default: finite differences, no autodiff
    JAX = "jax"  # JAX automatic differentiation
    PYTORCH = "pytorch"  # PyTorch autograd

    @property
    def is_numpy(self) -> bool:
        """Check if backend is NumPy (finite differences)."""
        return self == AutoDiffBackend.NUMPY

    @property
    def is_jax(self) -> bool:
        """Check if backend is JAX."""
        return self == AutoDiffBackend.JAX

    @property
    def is_pytorch(self) -> bool:
        """Check if backend is PyTorch."""
        return self == AutoDiffBackend.PYTORCH

    @property
    def requires_dependency(self) -> bool:
        """Check if backend requires optional dependency."""
        return self in (AutoDiffBackend.JAX, AutoDiffBackend.PYTORCH)

    def get_dependency_name(self) -> str | None:
        """
        Get the name of the required dependency for this backend.

        Returns:
            Package name (e.g., "jax", "torch") or None for NumPy
        """
        if self == AutoDiffBackend.JAX:
            return "jax"
        elif self == AutoDiffBackend.PYTORCH:
            return "torch"
        return None
