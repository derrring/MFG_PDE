"""
Temporary callable wrappers for operators not yet refactored to LinearOperator.

These wrappers provide a consistent interface while we gradually migrate to
LinearOperator classes (Issue #595). Once divergence, advection, and interpolation
are refactored, this module can be deprecated.

Created: 2026-01-17 (Issue #595 - Gradual Operator Refactoring)
Status: TEMPORARY - will be replaced by LinearOperator classes in Phase 2
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import numpy as np
    from numpy.typing import NDArray

    from mfg_pde.geometry.boundary import BoundaryConditions


def create_divergence_operator(
    spacings: Sequence[float],
    field_shape: tuple[int, ...],
    bc: BoundaryConditions | None = None,
    time: float = 0.0,
) -> Callable[[NDArray], NDArray]:
    """
    Create divergence operator as callable (temporary wrapper).

    Args:
        spacings: Grid spacing per dimension [h₀, h₁, ..., hd₋₁]
        field_shape: Shape of field arrays (Nx, Ny, ...)
        bc: Boundary conditions (None for periodic)
        time: Time for time-dependent BCs

    Returns:
        Callable div where div(F) computes ∇·F
        Input: F of shape (dimension, Nx, Ny, ...)
        Output: ∇·F of shape (Nx, Ny, ...)

    Example:
        >>> div_op = create_divergence_operator(spacings=[0.1, 0.1], field_shape=(50, 50))
        >>> F = np.random.rand(2, 50, 50)  # Vector field (Fx, Fy)
        >>> div_F = div_op(F)  # Shape: (50, 50)

    Note:
        This is a temporary wrapper. Will be replaced by DivergenceOperator
        LinearOperator class in Issue #595 Phase 2.
    """
    from mfg_pde.utils.numerical.tensor_calculus import divergence

    def divergence_callable(F: NDArray) -> NDArray:
        """Apply divergence to vector field F."""
        return divergence(F, spacings, bc=bc, time=time)

    return divergence_callable


def create_advection_operator(
    velocity_field: NDArray,
    spacings: Sequence[float],
    field_shape: tuple[int, ...],
    scheme: Literal["upwind", "centered"] = "upwind",
    form: Literal["divergence", "gradient"] = "divergence",
    bc: BoundaryConditions | None = None,
    time: float = 0.0,
) -> Callable[[NDArray], NDArray]:
    """
    Create advection operator for given velocity field (temporary wrapper).

    Args:
        velocity_field: Velocity/drift field, shape (dimension, Nx, Ny, ...)
        spacings: Grid spacing per dimension [h₀, h₁, ..., hd₋₁]
        field_shape: Shape of field arrays (Nx, Ny, ...)
        scheme: Advection scheme
            - "upwind": 1st-order upwind (stable, dissipative)
            - "centered": Centered differences (2nd-order, may oscillate)
        form: Formulation type
            - "divergence": ∇·(vm) (conservative, mass-conserving, use for FP)
            - "gradient": v·∇m (non-conservative, use for HJB)
        bc: Boundary conditions (None for periodic)
        time: Time for time-dependent BCs

    Returns:
        Callable adv where adv(m) computes advection term
        For divergence form: ∇·(vm)
        For gradient form: v·∇m

    Example:
        >>> v = np.random.rand(2, 50, 50)  # Velocity field
        >>> adv_op = create_advection_operator(
        ...     velocity_field=v,
        ...     spacings=[0.1, 0.1],
        ...     field_shape=(50, 50),
        ...     scheme="upwind",
        ...     form="divergence"
        ... )
        >>> m = np.random.rand(50, 50)  # Density
        >>> div_mv = adv_op(m)  # Conservative advection

    Note:
        This is a temporary wrapper. Will be replaced by AdvectionOperator
        LinearOperator class in Issue #595 Phase 2.
    """
    from mfg_pde.utils.numerical.tensor_calculus import advection

    # Convert velocity_field from stacked array to list of arrays
    # tensor_calculus expects v as list [vx, vy, ...]
    dimension = velocity_field.shape[0]
    v_list = [velocity_field[d] for d in range(dimension)]

    def advection_callable(m: NDArray) -> NDArray:
        """Apply advection operator to field m."""
        return advection(m, v_list, spacings, form=form, method=scheme, bc=bc, time=time)

    return advection_callable


def create_interpolation_operator(
    grid_points: tuple[NDArray, ...],
    query_points: NDArray,
    order: int = 1,
    extrapolation_mode: Literal["constant", "nearest", "boundary"] = "boundary",
) -> Callable[[NDArray], NDArray]:
    """
    Create interpolation operator for given query points (temporary wrapper).

    Args:
        grid_points: Tuple of 1D arrays defining grid (x, y, z, ...)
        query_points: Points at which to interpolate, shape (num_query, dimension)
        order: Interpolation order
            - 1: Linear (trilinear in 3D)
            - 2: Quadratic
            - 3: Cubic
        extrapolation_mode: How to handle points outside domain
            - "constant": Use fill_value (default: NaN)
            - "nearest": Use nearest boundary value
            - "boundary": Project to boundary and use boundary value

    Returns:
        Callable interp where interp(u) returns interpolated values at query_points
        Input: u of shape (Nx, Ny, ...)
        Output: interpolated values of shape (num_query,)

    Example:
        >>> x = np.linspace(0, 1, 50)
        >>> y = np.linspace(0, 1, 50)
        >>> grid_points = (x, y)
        >>> # Query points from semi-Lagrangian foot points
        >>> foot_points = np.random.rand(100, 2)
        >>> interp = create_interpolation_operator(grid_points, foot_points, order=1)
        >>> u = np.random.rand(50, 50)
        >>> u_foot = interp(u)  # Shape: (100,)

    Note:
        This is a temporary wrapper. Will be replaced by InterpolationOperator
        LinearOperator class in Issue #595 Phase 2.
    """
    from scipy.interpolate import RegularGridInterpolator

    # Map extrapolation_mode to scipy's fill_value parameter
    if extrapolation_mode == "constant":
        fill_value = None  # Will use NaN by default
        bounds_error = False
    elif extrapolation_mode == "nearest" or extrapolation_mode == "boundary":
        fill_value = None
        bounds_error = False
    else:
        raise ValueError(f"Unknown extrapolation_mode: {extrapolation_mode}")

    # Create scipy interpolator
    # Note: scipy's RegularGridInterpolator uses "linear" for order=1, "cubic" for order=3
    method_map = {1: "linear", 3: "cubic"}
    if order not in method_map:
        raise ValueError(f"Order {order} not supported. Use 1 (linear) or 3 (cubic).")

    def interpolation_callable(u: NDArray) -> NDArray:
        """Interpolate field u at query points."""
        # Create interpolator (cannot be cached due to different field values)
        interpolator = RegularGridInterpolator(
            grid_points,
            u,
            method=method_map[order],
            bounds_error=bounds_error,
            fill_value=fill_value,
        )

        # Interpolate at query points
        return interpolator(query_points)

    return interpolation_callable


if __name__ == "__main__":
    """Smoke test for operator wrappers."""
    import numpy as np

    print("Testing operator wrappers...")

    # Test 2D divergence
    print("\n[Divergence Wrapper]")
    Nx, Ny = 50, 50
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    dx, dy = x[1] - x[0], y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Test on F = (x, y), div(F) = ∂x/∂x + ∂y/∂y = 1 + 1 = 2
    F = np.stack([X, Y], axis=0)  # Shape: (2, Nx, Ny)

    div_op = create_divergence_operator(spacings=[dx, dy], field_shape=(Nx, Ny))
    div_F = div_op(F)

    print(f"  Vector field shape: {F.shape}")
    print(f"  Divergence shape: {div_F.shape}")
    print(f"  div(x, y) interior mean: {np.mean(div_F[10:-10, 10:-10]):.3f} (expected = 2.0)")
    assert div_F.shape == (Nx, Ny)
    assert 1.5 < np.mean(div_F[10:-10, 10:-10]) < 2.5
    print("  ✓ Divergence wrapper works")

    # Test 2D advection
    print("\n[Advection Wrapper]")
    v = np.ones((2, Nx, Ny))  # Constant velocity (1, 1)
    m = X**2  # Density = x²

    adv_op = create_advection_operator(
        velocity_field=v,
        spacings=[dx, dy],
        field_shape=(Nx, Ny),
        scheme="upwind",
        form="gradient",  # v·∇m (non-conservative form)
    )
    v_dot_grad_m = adv_op(m)

    print(f"  Velocity field shape: {v.shape}")
    print(f"  Density shape: {m.shape}")
    print(f"  v·∇m shape: {v_dot_grad_m.shape}")
    # v·∇(x²) = (1,1)·(2x, 0) = 2x
    expected = 2 * X
    error = np.max(np.abs(v_dot_grad_m[10:-10, 10:-10] - expected[10:-10, 10:-10]))
    print(f"  v·∇(x²) interior error: {error:.2e} (expected = 2x)")
    assert error < 0.1
    print("  ✓ Advection wrapper works")

    # Test 2D interpolation
    print("\n[Interpolation Wrapper]")
    grid_points = (x, y)
    query_points = np.array([[0.5, 0.5], [0.25, 0.75]])  # Two query points

    u = X + Y  # Simple linear function

    interp_op = create_interpolation_operator(grid_points, query_points, order=1)
    u_interp = interp_op(u)

    print(f"  Grid points: {len(grid_points)} dimensions")
    print(f"  Query points shape: {query_points.shape}")
    print(f"  Interpolated values: {u_interp}")
    # At (0.5, 0.5): x + y = 1.0
    # At (0.25, 0.75): x + y = 1.0
    expected_interp = np.array([1.0, 1.0])
    error = np.max(np.abs(u_interp - expected_interp))
    print(f"  Interpolation error: {error:.2e}")
    assert error < 1e-10
    print("  ✓ Interpolation wrapper works")

    print("\n✅ All operator wrapper tests passed!")
