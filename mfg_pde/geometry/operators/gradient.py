"""
Gradient operator for tensor product grids.

This module provides LinearOperator implementation of the discrete gradient
for structured grids, wrapping the tensor_calculus infrastructure.

Mathematical Background:
    Gradient operator: ∇u = (∂u/∂x₁, ∂u/∂x₂, ..., ∂u/∂xd)

    Discretization (2nd-order central differences):
        ∂u/∂x ≈ (u[i+1] - u[i-1]) / (2h)

    Returns d operators, one for each spatial direction.

References:
    - LeVeque (2007): Finite Difference Methods for ODEs and PDEs
    - Strang (2007): Computational Science and Engineering

Created: 2026-01-17 (Issue #595 - Operator Refactoring)
Part of: Issue #590 Phase 1.2 - TensorProductGrid Operator Traits
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.sparse.linalg import LinearOperator

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from mfg_pde.geometry.boundary import BoundaryConditions


class GradientComponentOperator(LinearOperator):
    """
    Single component of gradient operator: ∂u/∂xᵢ.

    This represents one directional derivative operator.
    The full gradient is a tuple of these operators.

    Attributes:
        direction: Spatial direction (0=x, 1=y, 2=z, ...)
        spacings: Grid spacing per dimension [h₀, h₁, ..., hd₋₁]
        field_shape: Shape of input field (Nx, Ny, ...)
        scheme: Difference scheme ("central", "upwind", "one_sided")
        bc: Boundary conditions
        shape: Operator shape (N, N) where N = ∏field_shape
        dtype: Data type (float64)
    """

    def __init__(
        self,
        direction: int,
        spacings: Sequence[float],
        field_shape: tuple[int, ...],
        scheme: Literal["central", "upwind", "one_sided"] = "central",
        bc: BoundaryConditions | None = None,
        time: float = 0.0,
    ):
        """
        Initialize gradient component operator.

        Args:
            direction: Spatial direction (0=x, 1=y, 2=z, ...)
            spacings: Grid spacing per dimension [h₀, h₁, ..., hd₋₁]
            field_shape: Shape of field arrays (Nx, Ny, ...)
            scheme: Difference scheme
                - "central": 2nd-order central differences (default)
                - "upwind": Godunov upwind (monotone, 1st-order)
                - "one_sided": Forward at left, backward at right
            bc: Boundary conditions (None for periodic)
            time: Time for time-dependent BCs (default 0.0)

        Raises:
            ValueError: If direction >= len(field_shape)
        """
        self.direction = direction
        self.spacings = list(spacings)
        self.field_shape = tuple(field_shape)
        self.scheme = scheme
        self.bc = bc
        self.time = time

        # Validate
        if direction >= len(field_shape):
            raise ValueError(f"direction {direction} >= dimension {len(field_shape)}")

        if len(spacings) != len(field_shape):
            raise ValueError(f"spacings length {len(spacings)} != field_shape dimensions {len(field_shape)}")

        # Compute operator shape
        N = int(np.prod(field_shape))
        super().__init__(shape=(N, N), dtype=np.float64)

    def _matvec(self, u_flat: NDArray) -> NDArray:
        """
        Apply ∂u/∂xᵢ to flattened field.

        Args:
            u_flat: Flattened field array, shape (N,)

        Returns:
            Directional derivative, flattened, shape (N,)
        """
        from mfg_pde.utils.numerical.tensor_calculus import gradient

        # Reshape to field
        u = u_flat.reshape(self.field_shape)

        # Compute full gradient (all components)
        grad_components = gradient(u, self.spacings, scheme=self.scheme, bc=self.bc, time=self.time)

        # Extract requested component
        du_dxi = grad_components[self.direction]

        # Return flattened
        return du_dxi.ravel()

    def __call__(self, u: NDArray) -> NDArray:
        """
        Apply ∂u/∂xᵢ to field (preserves shape).

        Args:
            u: Field array, shape field_shape or (N,)

        Returns:
            Directional derivative, same shape as input
        """
        # Handle already-flattened input
        if u.ndim == 1:
            return self._matvec(u)

        # Handle field input
        if u.shape != self.field_shape:
            raise ValueError(f"Input shape {u.shape} doesn't match field_shape {self.field_shape}")

        du_flat = self._matvec(u.ravel())
        return du_flat.reshape(self.field_shape)

    def __repr__(self) -> str:
        """String representation for debugging."""
        dim_names = ["x", "y", "z", "w"]
        dim_name = dim_names[self.direction] if self.direction < len(dim_names) else f"x{self.direction}"
        bc_str = f"bc={self.bc.bc_type.value}" if self.bc else "bc=periodic"

        return (
            f"GradientComponentOperator(∂/∂{dim_name},\n"
            f"  field_shape={self.field_shape},\n"
            f"  spacings={self.spacings},\n"
            f"  scheme='{self.scheme}',\n"
            f"  {bc_str},\n"
            f"  shape={self.shape}\n"
            f")"
        )


def create_gradient_operators(
    spacings: Sequence[float],
    field_shape: tuple[int, ...] | int,
    scheme: Literal["central", "upwind", "one_sided"] = "central",
    bc: BoundaryConditions | None = None,
    time: float = 0.0,
) -> tuple[GradientComponentOperator, ...]:
    """
    Create gradient operators for all spatial dimensions.

    Returns a tuple of operators (∂/∂x₀, ∂/∂x₁, ..., ∂/∂xd₋₁).

    Args:
        spacings: Grid spacing per dimension [h₀, h₁, ..., hd₋₁]
        field_shape: Shape of field arrays (Nx, Ny, ...) or Nx for 1D
        scheme: Difference scheme ("central", "upwind", "one_sided")
        bc: Boundary conditions (None for periodic)
        time: Time for time-dependent BCs

    Returns:
        Tuple of GradientComponentOperator, one per dimension

    Example:
        >>> # 2D gradient
        >>> grad_x, grad_y = create_gradient_operators(
        ...     spacings=[0.1, 0.1],
        ...     field_shape=(50, 50),
        ...     scheme="central"
        ... )
        >>> u = np.random.rand(50, 50)
        >>> du_dx = grad_x(u)
        >>> du_dy = grad_y(u)
        >>>
        >>> # Can also use @ syntax
        >>> du_dx_flat = grad_x @ u.ravel()
    """
    # Handle 1D shape
    if isinstance(field_shape, int):
        field_shape = (field_shape,)
    else:
        field_shape = tuple(field_shape)

    dimension = len(field_shape)

    # Create operator for each direction
    operators = tuple(
        GradientComponentOperator(
            direction=d,
            spacings=spacings,
            field_shape=field_shape,
            scheme=scheme,
            bc=bc,
            time=time,
        )
        for d in range(dimension)
    )

    return operators


if __name__ == "__main__":
    """Smoke test for GradientOperator."""
    print("Testing GradientOperator...")

    # Test 1D
    print("\n[1D Gradient]")

    # Test on u = sin(x), du/dx = cos(x)
    x = np.linspace(0, 2 * np.pi, 100)
    dx = x[1] - x[0]
    u_1d = np.sin(x)

    (grad_x,) = create_gradient_operators(spacings=[dx], field_shape=100, scheme="central")
    print(f"  Operator shape: {grad_x.shape}")
    print(f"  Field shape: {grad_x.field_shape}")
    print(f"  Grid spacing: dx={dx:.4f}")
    print(f"  Direction: {grad_x.direction} (x)")

    du_dx = grad_x(u_1d)
    expected = np.cos(x)

    print(f"  Input shape: {u_1d.shape}, Output shape: {du_dx.shape}")
    assert du_dx.shape == u_1d.shape

    # Check interior (boundaries may have larger error)
    error = np.max(np.abs(du_dx[10:-10] - expected[10:-10]))
    print(f"  ∂sin(x)/∂x interior error: {error:.2e} (expected = cos(x))")
    assert error < 0.01, f"Error too large: {error}"

    # Test @ syntax
    du_dx_matvec = grad_x @ u_1d.ravel()
    assert np.allclose(du_dx.ravel(), du_dx_matvec)
    print("  ✓ grad_x(u) == grad_x @ u.ravel()")

    # Test 2D
    print("\n[2D Gradient]")
    Nx, Ny = 50, 50
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    dx, dy = x[1] - x[0], y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Test on u = x² + y³, ∂u/∂x = 2x, ∂u/∂y = 3y²
    u_2d = X**2 + Y**3

    grad_x, grad_y = create_gradient_operators(
        spacings=[dx, dy],
        field_shape=(Nx, Ny),
        scheme="central",
    )

    print(f"  Operator shapes: {grad_x.shape}, {grad_y.shape}")
    print(f"  Field shape: {grad_x.field_shape}")

    du_dx = grad_x(u_2d)
    du_dy = grad_y(u_2d)

    print(f"  Input shape: {u_2d.shape}")
    print(f"  ∂u/∂x shape: {du_dx.shape}, ∂u/∂y shape: {du_dy.shape}")
    assert du_dx.shape == u_2d.shape
    assert du_dy.shape == u_2d.shape

    # Check interior
    expected_dx = 2 * X
    expected_dy = 3 * Y**2

    error_dx = np.max(np.abs(du_dx[10:-10, 10:-10] - expected_dx[10:-10, 10:-10]))
    error_dy = np.max(np.abs(du_dy[10:-10, 10:-10] - expected_dy[10:-10, 10:-10]))

    print(f"  ∂u/∂x interior error: {error_dx:.2e} (expected = 2x)")
    print(f"  ∂u/∂y interior error: {error_dy:.2e} (expected = 3y²)")
    # Quadratic function (x²) should have near-machine-precision error
    assert error_dx < 1e-10, f"∂u/∂x error too large: {error_dx}"
    # Cubic function (y³) has O(h²) discretization error
    assert error_dy < 1e-3, f"∂u/∂y error too large: {error_dy}"
    print("  ✓ Gradient accuracy check passed")

    # Test scipy compatibility
    print("\n[scipy compatibility]")
    assert isinstance(grad_x, LinearOperator)
    assert isinstance(grad_y, LinearOperator)
    print("  ✓ isinstance(grad_*, scipy.sparse.linalg.LinearOperator)")

    # Test repr
    print("\n[String representation]")
    print(grad_x)

    print("\n✅ All GradientOperator tests passed!")
