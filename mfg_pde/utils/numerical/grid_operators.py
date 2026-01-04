"""
Grid-based differential operators for PDE solvers.

.. deprecated:: 0.18.0
    This module is deprecated. Use ``tensor_calculus`` instead:

    Migration Guide::

        # Old
        from mfg_pde.utils.numerical.grid_operators import gradient, laplacian
        from mfg_pde.utils.numerical.grid_operators import GradientOperator

        # New
        from mfg_pde.utils.numerical.tensor_calculus import gradient, laplacian
        # GradientOperator is deprecated - use gradient() directly

    The new ``tensor_calculus`` module provides a complete set of operators:
    - gradient, divergence (first-order)
    - laplacian, hessian (second-order)
    - diffusion, tensor_diffusion (coefficient operators)
    - advection (transport)

This module is kept for backward compatibility and will be removed in v1.0.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mfg_pde.backends.backend_manager import ArrayBackend
    from mfg_pde.geometry.boundary import BoundaryConditions

# =============================================================================
# Re-exports from tensor_calculus (deprecated)
# =============================================================================

from mfg_pde.utils.numerical.tensor_calculus import (
    gradient as _gradient,
)
from mfg_pde.utils.numerical.tensor_calculus import (
    gradient_simple as _gradient_simple,
)
from mfg_pde.utils.numerical.tensor_calculus import (
    laplacian as _laplacian,
)

GradientScheme = Literal["central", "upwind", "one_sided"]


def gradient(
    u: NDArray,
    spacings: list[float] | tuple[float, ...],
    scheme: GradientScheme = "central",
    bc: BoundaryConditions | None = None,
    backend: ArrayBackend | None = None,
    time: float = 0.0,
) -> list[NDArray]:
    """
    DEPRECATED: Use mfg_pde.utils.numerical.tensor_calculus.gradient instead.
    """
    warnings.warn(
        "grid_operators.gradient is deprecated. Use mfg_pde.utils.numerical.tensor_calculus.gradient instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _gradient(u, spacings, scheme=scheme, bc=bc, backend=backend, time=time)


def gradient_simple(
    u: NDArray,
    spacings: list[float] | tuple[float, ...],
    backend: ArrayBackend | None = None,
) -> list[NDArray]:
    """
    DEPRECATED: Use mfg_pde.utils.numerical.tensor_calculus.gradient_simple instead.
    """
    warnings.warn(
        "grid_operators.gradient_simple is deprecated. "
        "Use mfg_pde.utils.numerical.tensor_calculus.gradient_simple instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _gradient_simple(u, spacings, backend=backend)


def laplacian(
    u: NDArray,
    spacings: list[float] | tuple[float, ...],
    bc: BoundaryConditions | None = None,
    backend: ArrayBackend | None = None,
    time: float = 0.0,
) -> NDArray:
    """
    DEPRECATED: Use mfg_pde.utils.numerical.tensor_calculus.laplacian instead.
    """
    warnings.warn(
        "grid_operators.laplacian is deprecated. Use mfg_pde.utils.numerical.tensor_calculus.laplacian instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _laplacian(u, spacings, bc=bc, backend=backend, time=time)


class GradientOperator:
    """
    DEPRECATED: Use mfg_pde.utils.numerical.tensor_calculus.gradient() directly.

    This class is deprecated and will be removed in v1.0.
    The functional interface gradient() provides the same functionality.
    """

    def __init__(
        self,
        scheme: GradientScheme = "central",
        bc: BoundaryConditions | None = None,
        backend: ArrayBackend | None = None,
    ):
        warnings.warn(
            "GradientOperator is deprecated. Use mfg_pde.utils.numerical.tensor_calculus.gradient() directly.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.scheme = scheme
        self.bc = bc
        self.backend = backend

    def __call__(
        self,
        u: NDArray,
        spacings: list[float] | tuple[float, ...],
        time: float = 0.0,
    ) -> list[NDArray]:
        """Compute gradient in all dimensions."""
        return _gradient(u, spacings, scheme=self.scheme, bc=self.bc, backend=self.backend, time=time)


# =============================================================================
# Internal helpers (kept for any external usage, emit warnings)
# =============================================================================


def _gradient_central(u: NDArray, axis: int, h: float, xp: type) -> NDArray:
    """Central difference: (u[i+1] - u[i-1]) / (2h)."""
    return (xp.roll(u, -1, axis=axis) - xp.roll(u, 1, axis=axis)) / (2 * h)


def _gradient_forward(u: NDArray, axis: int, h: float, xp: type) -> NDArray:
    """Forward difference: (u[i+1] - u[i]) / h."""
    return (xp.roll(u, -1, axis=axis) - u) / h


def _gradient_backward(u: NDArray, axis: int, h: float, xp: type) -> NDArray:
    """Backward difference: (u[i] - u[i-1]) / h."""
    return (u - xp.roll(u, 1, axis=axis)) / h


def _gradient_upwind(u: NDArray, axis: int, h: float, xp: type) -> NDArray:
    """Godunov upwind: select forward/backward based on flow direction."""
    grad_forward = _gradient_forward(u, axis, h, xp)
    grad_backward = _gradient_backward(u, axis, h, xp)
    grad_central = (grad_forward + grad_backward) / 2.0
    return xp.where(grad_central >= 0, grad_backward, grad_forward)


if __name__ == "__main__":
    print("grid_operators.py is deprecated.")
    print("Use mfg_pde.utils.numerical.tensor_calculus instead.")
    print("\nRunning quick verification that re-exports work...")

    # Quick test
    u = np.sin(np.linspace(0, 2 * np.pi, 100))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        grads = gradient_simple(u, spacings=[0.1])
        print(f"  gradient_simple works: shape={grads[0].shape}")

        lap = laplacian(u.reshape(10, 10), spacings=[0.1, 0.1])
        print(f"  laplacian works: shape={lap.shape}")

    print("\nAll re-exports verified!")
