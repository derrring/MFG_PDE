"""
Differential Utilities - DEPRECATED.

Most functions in this module are deprecated in favor of:
- scipy.optimize.approx_fprime for function gradients
- scipy.optimize.check_grad for gradient verification
- mfg_pde.utils.numerical.grid_operators for grid-based operators

This module is kept for backward compatibility and will be removed in v1.0.

Migration Guide:
    # Old: from mfg_pde.utils.numerical.differential_utils import gradient_fd
    # New: from scipy.optimize import approx_fprime
    #      grad = approx_fprime(x, f, epsilon=1e-7)

    # Old: from mfg_pde.utils.numerical.differential_utils import gradient_grid_nd
    # New: from mfg_pde.utils.numerical.tensor_calculus import gradient_simple

    # Old: from mfg_pde.utils.numerical.differential_utils import compute_dH_dp
    # New: Use scipy.optimize.approx_fprime directly in your solver
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.optimize import approx_fprime

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


def gradient_fd(
    f: Callable[[NDArray], float],
    x: NDArray,
    eps: float = 1e-7,
    method: Literal["forward", "central"] = "forward",
) -> NDArray:
    """
    DEPRECATED: Use scipy.optimize.approx_fprime instead.

    Finite difference gradient of scalar function f: R^n -> R.
    """
    warnings.warn(
        "gradient_fd is deprecated. Use scipy.optimize.approx_fprime instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Delegate to scipy
    return approx_fprime(np.asarray(x, dtype=np.float64), f, eps)


def gradient_grid_nd(
    U: NDArray,
    spacings: list[float] | tuple[float, ...],
    backend: object | None = None,
) -> list[NDArray]:
    """
    DEPRECATED: Use mfg_pde.utils.numerical.grid_operators.gradient_simple instead.
    """
    warnings.warn(
        "gradient_grid_nd is deprecated. Use mfg_pde.utils.numerical.tensor_calculus.gradient_simple instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from mfg_pde.utils.numerical.tensor_calculus import gradient_simple

    return gradient_simple(U, spacings, backend=backend)


def compute_dH_dp(
    H_func: Callable,
    x_idx: int,
    m: float,
    p: NDArray,
    hess: NDArray | None = None,
    eps: float = 1e-7,
    method: Literal["forward", "central"] = "forward",
) -> NDArray:
    """
    DEPRECATED: Use scipy.optimize.approx_fprime directly.

    Compute dH/dp for Hamiltonian via finite difference.
    """
    warnings.warn(
        "compute_dH_dp is deprecated. Use scipy.optimize.approx_fprime directly.",
        DeprecationWarning,
        stacklevel=2,
    )
    from mfg_pde.core.derivatives import DerivativeTensors

    p = np.asarray(p)
    dim = len(p)

    if hess is None:
        hess = np.zeros((dim, dim))

    def H_of_p(p_vec: NDArray) -> float:
        derivs = DerivativeTensors.from_arrays(grad=p_vec, hess=hess)
        return H_func(x_idx, m, derivs=derivs)

    return approx_fprime(p, H_of_p, eps)


# =============================================================================
# Smoke Tests
# =============================================================================

if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning)

    print("Testing differential_utils (deprecated module)...")

    # Test gradient_fd
    def f_quadratic(x):
        return np.sum(x**2)

    x = np.array([1.0, 2.0, 3.0])
    grad = gradient_fd(f_quadratic, x)
    expected = 2 * x
    print(f"  gradient_fd error: {np.linalg.norm(grad - expected):.2e}")
    assert np.linalg.norm(grad - expected) < 1e-5

    # Test gradient_grid_nd
    U = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    grads = gradient_grid_nd(U, spacings=[1.0, 1.0])
    assert len(grads) == 2
    print(f"  gradient_grid_nd: {len(grads)} components")

    print("\nAll tests passed! (Note: this module is deprecated)")
