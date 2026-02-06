"""
Tensor Diffusion Operators for Anisotropic PDEs.

.. deprecated:: 0.18.0
    This module is deprecated and will be removed in v1.0.0. Use ``tensor_calculus`` instead:

    Migration Guide::

        # Old
        from mfg_pde.utils.numerical.tensor_operators import (
            divergence_tensor_diffusion_2d,
            divergence_tensor_diffusion_nd,
        )

        # New
        from mfg_pde.utils.numerical.tensor_calculus import diffusion

        # diffusion() auto-dispatches to 1D/2D/nD implementations
        # and handles isotropic/anisotropic/spatially-varying cases

    The new ``tensor_calculus`` module provides:
    - Unified API: diffusion(u, Sigma, spacings) for all cases
    - Consistent BC handling across operators
    - Complete tensor calculus: gradient, divergence, laplacian, hessian, advection

This module is kept for backward compatibility and will be removed in v1.0.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mfg_pde.geometry import BoundaryConditions

# Import the canonical implementation
from mfg_pde.utils.numerical.tensor_calculus import diffusion as _diffusion


def divergence_tensor_diffusion_2d(
    m: NDArray,
    Sigma: NDArray,
    dx: float,
    dy: float,
    boundary_conditions: BoundaryConditions | None = None,
    bc_type: Literal["periodic", "dirichlet", "neumann", "no_flux"] = "no_flux",
) -> NDArray:
    """
    DEPRECATED: Use mfg_pde.utils.numerical.tensor_calculus.diffusion instead.

    Compute ∇·(Σ∇m) for 2D anisotropic diffusion.

    Migration::

        # Old
        result = divergence_tensor_diffusion_2d(m, Sigma, dx, dy, bc)

        # New
        from mfg_pde.utils.numerical.tensor_calculus import diffusion
        result = diffusion(m, Sigma, [dx, dy], bc=bc)
    """
    warnings.warn(
        "divergence_tensor_diffusion_2d is deprecated. Use mfg_pde.utils.numerical.tensor_calculus.diffusion instead. Will be removed in v1.0.0.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _diffusion(m, Sigma, [dx, dy], bc=boundary_conditions)


def divergence_diagonal_diffusion_2d(
    m: NDArray,
    sigma_diag: NDArray,
    dx: float,
    dy: float,
    boundary_conditions: BoundaryConditions | None = None,
) -> NDArray:
    """
    DEPRECATED: Use mfg_pde.utils.numerical.tensor_calculus.diffusion instead.

    Compute ∇·(Σ∇m) for 2D diagonal diffusion tensor.

    Migration::

        # Old
        result = divergence_diagonal_diffusion_2d(m, sigma_diag, dx, dy, bc)

        # New
        from mfg_pde.utils.numerical.tensor_calculus import diffusion
        Sigma = np.diag(sigma_diag)  # or pass diagonal directly
        result = diffusion(m, Sigma, [dx, dy], bc=bc)
    """
    warnings.warn(
        "divergence_diagonal_diffusion_2d is deprecated. "
        "Use mfg_pde.utils.numerical.tensor_calculus.diffusion instead. Will be removed in v1.0.0.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Convert diagonal to full tensor
    Sigma = np.diag(sigma_diag)
    return _diffusion(m, Sigma, [dx, dy], bc=boundary_conditions)


def divergence_tensor_diffusion_nd(
    m: NDArray,
    sigma_tensor: NDArray,
    dx: tuple[float, ...] | list[float],
    boundary_conditions: BoundaryConditions | None = None,
) -> NDArray:
    """
    DEPRECATED: Use mfg_pde.utils.numerical.tensor_calculus.diffusion instead.

    Compute ∇·(Σ∇m) for nD anisotropic diffusion.

    Migration::

        # Old
        result = divergence_tensor_diffusion_nd(m, Sigma, dx, bc)

        # New
        from mfg_pde.utils.numerical.tensor_calculus import diffusion
        result = diffusion(m, Sigma, list(dx), bc=bc)
    """
    warnings.warn(
        "divergence_tensor_diffusion_nd is deprecated. Use mfg_pde.utils.numerical.tensor_calculus.diffusion instead. Will be removed in v1.0.0.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _diffusion(m, sigma_tensor, list(dx), bc=boundary_conditions)


if __name__ == "__main__":
    print("tensor_operators.py is deprecated.")
    print("Use mfg_pde.utils.numerical.tensor_calculus.diffusion instead.")
    print("\nRunning quick verification that re-exports work...")

    # Quick test
    m = np.random.rand(10, 10)
    Sigma = np.array([[0.1, 0.0], [0.0, 0.05]])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)

        # Test 2D
        result = divergence_tensor_diffusion_2d(m, Sigma, 0.1, 0.1)
        print(f"  divergence_tensor_diffusion_2d works: shape={result.shape}")

        # Test diagonal
        result_diag = divergence_diagonal_diffusion_2d(m, np.array([0.1, 0.05]), 0.1, 0.1)
        print(f"  divergence_diagonal_diffusion_2d works: shape={result_diag.shape}")

        # Test nD
        m3 = np.random.rand(5, 5, 5)
        Sigma3 = np.eye(3) * 0.1
        result_nd = divergence_tensor_diffusion_nd(m3, Sigma3, (0.1, 0.1, 0.1))
        print(f"  divergence_tensor_diffusion_nd works: shape={result_nd.shape}")

    print("\nAll re-exports verified!")
