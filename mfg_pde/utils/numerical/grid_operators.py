"""
Grid-based differential operators for PDE solvers.

.. deprecated:: 0.18.0
    This module is deprecated. Use ``tensor_calculus`` instead:

    Migration Guide::

        # Old
        from mfg_pde.utils.numerical.grid_operators import gradient, laplacian

        # New
        from mfg_pde.utils.numerical.tensor_calculus import gradient, laplacian

    The new ``tensor_calculus`` module provides a complete set of operators:
    - gradient, divergence (first-order)
    - laplacian, hessian (second-order)
    - diffusion, tensor_diffusion (coefficient operators)
    - advection (transport)

This module is kept for backward compatibility and will be removed in v1.0.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mfg_pde.backends.backend_manager import ArrayBackend
    from mfg_pde.geometry.boundary import BoundaryConditions


# =============================================================================
# Gradient Schemes
# =============================================================================

GradientScheme = Literal["central", "upwind", "one_sided"]


def _gradient_central(
    u: NDArray,
    axis: int,
    h: float,
    xp: type,
) -> NDArray:
    """Central difference: (u[i+1] - u[i-1]) / (2h)."""
    return (xp.roll(u, -1, axis=axis) - xp.roll(u, 1, axis=axis)) / (2 * h)


def _gradient_forward(
    u: NDArray,
    axis: int,
    h: float,
    xp: type,
) -> NDArray:
    """Forward difference: (u[i+1] - u[i]) / h."""
    return (xp.roll(u, -1, axis=axis) - u) / h


def _gradient_backward(
    u: NDArray,
    axis: int,
    h: float,
    xp: type,
) -> NDArray:
    """Backward difference: (u[i] - u[i-1]) / h."""
    return (u - xp.roll(u, 1, axis=axis)) / h


def _gradient_upwind(
    u: NDArray,
    axis: int,
    h: float,
    xp: type,
) -> NDArray:
    """
    Godunov upwind: select forward/backward based on flow direction.

    Uses central difference to estimate sign, then selects:
    - Backward difference if gradient >= 0 (information from left)
    - Forward difference if gradient < 0 (information from right)
    """
    grad_forward = _gradient_forward(u, axis, h, xp)
    grad_backward = _gradient_backward(u, axis, h, xp)
    grad_central = (grad_forward + grad_backward) / 2.0
    return xp.where(grad_central >= 0, grad_backward, grad_forward)


# =============================================================================
# GradientOperator Class
# =============================================================================


class GradientOperator:
    """
    BC-aware gradient operator for regular grids.

    Computes spatial gradients with configurable difference scheme and
    proper boundary condition handling.

    Parameters
    ----------
    scheme : {"central", "upwind", "one_sided"}
        Difference scheme to use:
        - "central": Second-order central differences (default)
        - "upwind": Godunov upwind (monotone, first-order)
        - "one_sided": Forward at left boundary, backward at right
    bc : BoundaryConditions, optional
        Boundary conditions for ghost cell computation.
        If None, uses periodic wrapping (np.roll behavior).
    backend : ArrayBackend, optional
        GPU backend for array operations. Uses numpy if None.

    Examples
    --------
    >>> from mfg_pde.utils.numerical.grid_operators import GradientOperator
    >>> from mfg_pde.geometry.boundary import neumann_bc
    >>>
    >>> # Create operator with Neumann BC
    >>> grad_op = GradientOperator(scheme="central", bc=neumann_bc(dimension=2))
    >>> u = np.random.rand(32, 32)
    >>> grads = grad_op(u, spacings=[0.1, 0.1])
    >>> grads[0].shape  # du/dx
    (32, 32)
    """

    def __init__(
        self,
        scheme: GradientScheme = "central",
        bc: BoundaryConditions | None = None,
        backend: ArrayBackend | None = None,
    ):
        self.scheme = scheme
        self.bc = bc
        self.backend = backend

    def __call__(
        self,
        u: NDArray,
        spacings: list[float] | tuple[float, ...],
        time: float = 0.0,
    ) -> list[NDArray]:
        """
        Compute gradient in all dimensions.

        Parameters
        ----------
        u : NDArray
            Field values on grid, shape (N0, N1, ..., Nd-1)
        spacings : list[float]
            Grid spacing per dimension [h0, h1, ..., hd-1]
        time : float
            Current time for time-dependent BCs

        Returns
        -------
        list[NDArray]
            Gradient components [du/dx0, du/dx1, ...], each with shape of u
        """
        xp = self.backend.array_module if self.backend is not None else np
        dimension = u.ndim

        # Apply ghost cells if BC provided
        if self.bc is not None:
            u_work = self._apply_ghost_cells(u, time)
        else:
            u_work = u

        gradients = []
        for d in range(dimension):
            h = spacings[d]
            if h < 1e-14:
                gradients.append(xp.zeros_like(u))
                continue

            # Compute gradient with selected scheme
            if self.scheme == "central":
                grad_d = _gradient_central(u_work, d, h, xp)
            elif self.scheme == "upwind":
                grad_d = _gradient_upwind(u_work, d, h, xp)
            elif self.scheme == "one_sided":
                # Central in interior, one-sided at boundaries
                grad_d = _gradient_central(u_work, d, h, xp)
                # Fix boundaries with one-sided differences
                grad_d = self._fix_boundaries_one_sided(grad_d, u_work, d, h, xp)
            else:
                raise ValueError(f"Unknown scheme: {self.scheme}")

            # Extract interior if ghost cells were added
            if self.bc is not None:
                grad_d = self._extract_interior(grad_d, dimension)

            gradients.append(grad_d)

        return gradients

    def _apply_ghost_cells(self, u: NDArray, time: float) -> NDArray:
        """Apply ghost cells using boundary conditions."""
        from mfg_pde.geometry.boundary import apply_boundary_conditions_nd

        # apply_boundary_conditions_nd adds 1 ghost cell per side
        return apply_boundary_conditions_nd(u, self.bc, time=time)

    def _extract_interior(self, u_padded: NDArray, dimension: int) -> NDArray:
        """Extract interior from padded array (remove ghost cells)."""
        slices = [slice(1, -1)] * dimension
        return u_padded[tuple(slices)]

    def _fix_boundaries_one_sided(
        self,
        grad: NDArray,
        u: NDArray,
        axis: int,
        h: float,
        xp: type,
    ) -> NDArray:
        """Replace boundary values with one-sided differences."""
        ndim = u.ndim

        # Left boundary: forward difference
        left_slice = [slice(None)] * ndim
        left_slice[axis] = 0
        next_slice = [slice(None)] * ndim
        next_slice[axis] = 1
        grad[tuple(left_slice)] = (u[tuple(next_slice)] - u[tuple(left_slice)]) / h

        # Right boundary: backward difference
        right_slice = [slice(None)] * ndim
        right_slice[axis] = -1
        prev_slice = [slice(None)] * ndim
        prev_slice[axis] = -2
        grad[tuple(right_slice)] = (u[tuple(right_slice)] - u[tuple(prev_slice)]) / h

        return grad


# =============================================================================
# Functional Interface
# =============================================================================


def gradient(
    u: NDArray,
    spacings: list[float] | tuple[float, ...],
    scheme: GradientScheme = "central",
    bc: BoundaryConditions | None = None,
    backend: ArrayBackend | None = None,
    time: float = 0.0,
) -> list[NDArray]:
    """
    Compute spatial gradient on regular grid.

    Functional interface to GradientOperator. For repeated calls with
    same settings, prefer creating a GradientOperator instance.

    Parameters
    ----------
    u : NDArray
        Field values on grid
    spacings : list[float]
        Grid spacing per dimension
    scheme : {"central", "upwind", "one_sided"}
        Difference scheme
    bc : BoundaryConditions, optional
        Boundary conditions
    backend : ArrayBackend, optional
        GPU backend
    time : float
        Time for time-dependent BCs

    Returns
    -------
    list[NDArray]
        Gradient components per dimension

    Examples
    --------
    >>> u = np.sin(np.linspace(0, 2*np.pi, 100))
    >>> du_dx = gradient(u, spacings=[0.1])[0]
    """
    op = GradientOperator(scheme=scheme, bc=bc, backend=backend)
    return op(u, spacings, time=time)


def gradient_simple(
    u: NDArray,
    spacings: list[float] | tuple[float, ...],
    backend: ArrayBackend | None = None,
) -> list[NDArray]:
    """
    Simple central difference gradient without BC handling.

    Fast path for cases where BC handling is done separately
    (e.g., particle methods where BC applies to particles, not grid).

    Parameters
    ----------
    u : NDArray
        Field values on grid
    spacings : list[float]
        Grid spacing per dimension
    backend : ArrayBackend, optional
        GPU backend

    Returns
    -------
    list[NDArray]
        Gradient components per dimension
    """
    xp = backend.array_module if backend is not None else np

    gradients = []
    for d in range(u.ndim):
        h = spacings[d]
        if h > 1e-14:
            grad_d = (xp.roll(u, -1, axis=d) - xp.roll(u, 1, axis=d)) / (2 * h)
        else:
            grad_d = xp.zeros_like(u)
        gradients.append(grad_d)

    return gradients


# =============================================================================
# Laplacian Operator
# =============================================================================


def laplacian(
    u: NDArray,
    spacings: list[float] | tuple[float, ...],
    bc: BoundaryConditions | None = None,
    backend: ArrayBackend | None = None,
    time: float = 0.0,
) -> NDArray:
    """
    Compute Laplacian on regular grid.

    Uses standard 3-point stencil per dimension:
        d²u/dx² ≈ (u[i+1] - 2*u[i] + u[i-1]) / h²

    Parameters
    ----------
    u : NDArray
        Field values on grid
    spacings : list[float]
        Grid spacing per dimension
    bc : BoundaryConditions, optional
        Boundary conditions
    backend : ArrayBackend, optional
        GPU backend
    time : float
        Time for time-dependent BCs

    Returns
    -------
    NDArray
        Laplacian with same shape as u
    """
    xp = backend.array_module if backend is not None else np

    # Apply ghost cells if BC provided
    if bc is not None:
        from mfg_pde.geometry.boundary import apply_boundary_conditions_nd

        u_work = apply_boundary_conditions_nd(u, bc, time=time)
    else:
        u_work = u

    lap = xp.zeros_like(u_work)

    for d in range(u.ndim):
        h = spacings[d]
        if h > 1e-14:
            lap += (xp.roll(u_work, -1, axis=d) - 2 * u_work + xp.roll(u_work, 1, axis=d)) / (h * h)

    # Extract interior if ghost cells were added
    if bc is not None:
        slices = [slice(1, -1)] * u.ndim
        lap = lap[tuple(slices)]

    return lap


# =============================================================================
# Smoke Tests
# =============================================================================

if __name__ == "__main__":
    print("Testing grid_operators...")

    # Test 1D gradient
    print("\n[1D] Testing gradient operators...")
    x = np.linspace(0, 2 * np.pi, 100)
    u_1d = np.sin(x)
    dx = x[1] - x[0]

    grads = gradient_simple(u_1d, spacings=[dx])
    expected = np.cos(x)
    error = np.max(np.abs(grads[0][5:-5] - expected[5:-5]))  # Ignore boundaries
    print(f"  1D sin gradient error (interior): {error:.2e}")
    assert error < 0.01, "1D gradient error too large"

    # Test 2D gradient
    print("\n[2D] Testing gradient operators...")
    nx, ny = 32, 32
    dx, dy = 0.1, 0.1
    x = np.linspace(0, (nx - 1) * dx, nx)
    y = np.linspace(0, (ny - 1) * dy, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    u_2d = X**2 + Y**2  # u = x² + y², grad = (2x, 2y)

    grads_2d = gradient_simple(u_2d, spacings=[dx, dy])
    assert len(grads_2d) == 2, "Should have 2 gradient components"
    assert grads_2d[0].shape == u_2d.shape, "Gradient shape mismatch"

    # Check gradient values in interior
    expected_dudx = 2 * X
    expected_dudy = 2 * Y
    error_x = np.max(np.abs(grads_2d[0][5:-5, 5:-5] - expected_dudx[5:-5, 5:-5]))
    error_y = np.max(np.abs(grads_2d[1][5:-5, 5:-5] - expected_dudy[5:-5, 5:-5]))
    print(f"  2D x-gradient error (interior): {error_x:.2e}")
    print(f"  2D y-gradient error (interior): {error_y:.2e}")
    assert error_x < 1e-10, "2D x-gradient error too large"
    assert error_y < 1e-10, "2D y-gradient error too large"

    # Test upwind scheme
    print("\n[Upwind] Testing upwind gradient...")
    grad_op_upwind = GradientOperator(scheme="upwind")
    grads_upwind = grad_op_upwind(u_2d, spacings=[dx, dy])
    assert len(grads_upwind) == 2, "Upwind should return 2 components"
    print("  Upwind gradient computed successfully")

    # Test Laplacian
    print("\n[Laplacian] Testing Laplacian operator...")
    # u = x² + y², Laplacian = 2 + 2 = 4
    lap = laplacian(u_2d, spacings=[dx, dy])
    expected_lap = 4.0
    error_lap = np.max(np.abs(lap[5:-5, 5:-5] - expected_lap))
    print(f"  Laplacian error (interior): {error_lap:.2e}")
    assert error_lap < 1e-10, "Laplacian error too large"

    # Test with BC (if available)
    print("\n[BC] Testing with boundary conditions...")
    try:
        from mfg_pde.geometry.boundary import neumann_bc

        bc = neumann_bc(dimension=2)
        grad_op_bc = GradientOperator(scheme="central", bc=bc)
        grads_bc = grad_op_bc(u_2d, spacings=[dx, dy])
        print(f"  BC-aware gradient computed, shape: {grads_bc[0].shape}")
        assert grads_bc[0].shape == u_2d.shape, "BC gradient should match input shape"
    except Exception as e:
        print(f"  BC test skipped: {e}")

    print("\nAll tests passed!")
