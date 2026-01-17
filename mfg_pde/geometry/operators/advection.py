"""
Advection operator for tensor product grids.

This module provides LinearOperator implementation of discrete advection
for structured grids, wrapping the tensor_calculus infrastructure.

Mathematical Background:
    Advection operator models transport by a velocity field v:

    1. **Gradient form** (non-conservative): A(m) = v·∇m
       - Used in HJB equations: ∂u/∂t + H(∇u) = 0 where H(p) = max_α[α·p - L(α)]
       - Non-conservative: doesn't preserve integral ∫m dx

    2. **Divergence form** (conservative): A(m) = ∇·(vm)
       - Used in Fokker-Planck equations: ∂m/∂t + ∇·(αm) = 0
       - Conservative: preserves mass ∫m dx = const

    Relationship:
        ∇·(vm) = v·∇m + m(∇·v)  (product rule)

        If ∇·v = 0 (incompressible flow), both forms are equivalent.

    Discretization Schemes:
        - **Upwind**: 1st-order, stable, dissipative (numerical viscosity)
        - **Centered**: 2nd-order, may oscillate without stabilization

References:
    - LeVeque (2002): Finite Volume Methods for Hyperbolic Problems
    - Godunov & Ryabenkii (1987): Difference Schemes
    - Achdou & Capuzzo-Dolcetta (2010): Mean Field Games

Created: 2026-01-17 (Issue #595 Phase 2 - Operator Refactoring)
Part of: Geometry Operator LinearOperator Migration
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.sparse.linalg import LinearOperator

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from mfg_pde.geometry.boundary import BoundaryConditions


class AdvectionOperator(LinearOperator):
    """
    Discrete advection operator for tensor product grids.

    Implements scipy.sparse.linalg.LinearOperator interface for compatibility
    with iterative solvers and operator composition.

    The operator wraps mfg_pde.utils.numerical.tensor_calculus.advection with
    grid-specific parameters and velocity field curried into the operator object.

    **Mathematical Context**:
        Advection describes transport by a velocity/drift field:
            - Gradient form: v·∇m (HJB equations)
            - Divergence form: ∇·(vm) (Fokker-Planck equations)

    **MFG Applications**:
        - **HJB equation**: ∂u/∂t + v·∇u = 0 (use form="gradient")
        - **FP equation**: ∂m/∂t + ∇·(vm) = 0 (use form="divergence")

    **Operator Shape**:
        Input:  Scalar field m flattened to shape (N,)
        Output: Advection term flattened to shape (N,)
        Operator shape: (N, N)

        where N = ∏field_shape (total number of grid points)

    Attributes:
        velocity_field: Drift/velocity field, shape (dimension, Nx, Ny, ...)
        spacings: Grid spacing per dimension [h₀, h₁, ..., hd₋₁]
        field_shape: Shape of scalar fields (Nx, Ny, ...) or (Nx,) for 1D
        scheme: "upwind" (1st-order) or "centered" (2nd-order)
        form: "gradient" (v·∇m) or "divergence" (∇·(vm))
        bc: Boundary conditions (None for periodic)
        time: Time for time-dependent BCs
        shape: Operator shape (N, N)
        dtype: Data type (float64)

    Usage:
        >>> # Create operator with upwind scheme
        >>> v = np.ones((2, 50, 50))  # Constant velocity (1, 1)
        >>> adv_op = AdvectionOperator(
        ...     velocity_field=v,
        ...     spacings=[0.1, 0.1],
        ...     field_shape=(50, 50),
        ...     scheme="upwind",
        ...     form="divergence",
        ...     bc=bc
        ... )
        >>>
        >>> # Apply via matrix-vector product
        >>> m_flat = m.ravel()
        >>> adv_m_flat = adv_op @ m_flat
        >>>
        >>> # Apply via callable (preserves field shape)
        >>> adv_m = adv_op(m)  # Input/Output: (50, 50)

    Example:
        >>> import numpy as np
        >>> from mfg_pde.geometry.boundary import no_flux_bc
        >>>
        >>> # Test conservative advection: ∇·(vm) with constant v
        >>> Nx, Ny = 100, 100
        >>> x = np.linspace(0, 1, Nx)
        >>> y = np.linspace(0, 1, Ny)
        >>> X, Y = np.meshgrid(x, y, indexing='ij')
        >>>
        >>> # Linear density m = x
        >>> m = X
        >>> # Constant velocity v = (1, 0)
        >>> v = np.zeros((2, Nx, Ny))
        >>> v[0] = 1.0
        >>>
        >>> bc = no_flux_bc(dimension=2)
        >>> adv_op = AdvectionOperator(
        ...     velocity_field=v,
        ...     spacings=[0.01, 0.01],
        ...     field_shape=(Nx, Ny),
        ...     scheme="upwind",
        ...     form="gradient",  # v·∇m
        ...     bc=bc
        ... )
        >>>
        >>> # v·∇(x) = (1,0)·(1,0) = 1.0
        >>> adv_m = adv_op(m)
        >>> print(f"Mean advection: {np.mean(adv_m):.3f}")  # ~1.0
    """

    def __init__(
        self,
        velocity_field: NDArray,
        spacings: Sequence[float],
        field_shape: tuple[int, ...] | int,
        scheme: Literal["upwind", "centered"] = "upwind",
        form: Literal["gradient", "divergence"] = "divergence",
        bc: BoundaryConditions | None = None,
        time: float = 0.0,
    ):
        """
        Initialize advection operator.

        Args:
            velocity_field: Velocity/drift field, shape (dimension, Nx, Ny, ...)
            spacings: Grid spacing per dimension [h₀, h₁, ..., hd₋₁]
            field_shape: Shape of scalar field arrays (Nx, Ny, ...) or Nx for 1D
            scheme: Discretization scheme
                - "upwind": 1st-order upwind (stable, dissipative)
                - "centered": 2nd-order centered (may oscillate)
            form: Advection form
                - "gradient": v·∇m (non-conservative, for HJB)
                - "divergence": ∇·(vm) (conservative, for FP)
            bc: Boundary conditions (None for periodic/wrap)
            time: Time for time-dependent BCs (default 0.0)

        Raises:
            ValueError: If scheme or form invalid
            ValueError: If velocity_field shape incompatible with field_shape
        """
        # Handle 1D shape
        if isinstance(field_shape, int):
            field_shape = (field_shape,)
        else:
            field_shape = tuple(field_shape)

        # Validate velocity_field shape
        expected_v_shape = (len(field_shape), *field_shape)
        if velocity_field.shape != expected_v_shape:
            raise ValueError(
                f"velocity_field shape {velocity_field.shape} doesn't match "
                f"expected {expected_v_shape} for field_shape={field_shape}"
            )

        # Validate scheme and form
        if scheme not in ("upwind", "centered"):
            raise ValueError(f"Unknown scheme: {scheme}. Use 'upwind' or 'centered'.")

        if form not in ("gradient", "divergence"):
            raise ValueError(f"Unknown form: {form}. Use 'gradient' or 'divergence'.")

        self.velocity_field = velocity_field
        self.spacings = list(spacings)
        self.field_shape = field_shape
        self.scheme = scheme
        self.form = form
        self.bc = bc
        self.time = time
        self.dimension = len(field_shape)

        # Validate
        if len(self.spacings) != self.dimension:
            raise ValueError(f"spacings length {len(self.spacings)} != field_shape dimensions {self.dimension}")

        # Convert velocity_field to list format expected by tensor_calculus
        # tensor_calculus.advection expects v as list [vx, vy, ...]
        self._v_list = [self.velocity_field[d] for d in range(self.dimension)]

        # Compute operator shape (N, N) - scalar field to scalar field
        N = int(np.prod(field_shape))
        super().__init__(shape=(N, N), dtype=np.float64)

    def _matvec(self, m_flat: NDArray) -> NDArray:
        """
        Apply advection to flattened scalar field.

        This is the core LinearOperator method required by scipy.

        Args:
            m_flat: Flattened scalar field, shape (N,)

        Returns:
            Advection term, flattened, shape (N,)
                - Gradient form: v·∇m
                - Divergence form: ∇·(vm)
        """
        from mfg_pde.utils.numerical.tensor_calculus import advection

        # Reshape to field
        m = m_flat.reshape(self.field_shape)

        # Apply advection
        adv_m = advection(
            m,
            self._v_list,
            self.spacings,
            form=self.form,
            method=self.scheme,
            bc=self.bc,
            time=self.time,
        )

        # Return flattened
        return adv_m.ravel()

    def __call__(self, m: NDArray) -> NDArray:
        """
        Apply advection to scalar field (convenience method).

        This preserves the field shape, unlike matrix-vector product which
        operates on flattened arrays.

        Args:
            m: Scalar field, shape (Nx, Ny, ...)

        Returns:
            Advection term, shape (Nx, Ny, ...)
                - Gradient form: v·∇m
                - Divergence form: ∇·(vm)

        Example:
            >>> adv_op = AdvectionOperator(v, spacings=[0.1, 0.1], field_shape=(50, 50))
            >>> m = np.random.rand(50, 50)
            >>> adv_m = adv_op(m)  # Shape: (50, 50)
        """
        # Validate input shape
        if m.shape != self.field_shape:
            raise ValueError(f"Field shape {m.shape} doesn't match expected {self.field_shape}")

        # Apply via _matvec
        adv_m_flat = self._matvec(m.ravel())
        return adv_m_flat.reshape(self.field_shape)

    def __repr__(self) -> str:
        """String representation of operator."""
        return (
            f"AdvectionOperator(\n"
            f"  dimension={self.dimension},\n"
            f"  field_shape={self.field_shape},\n"
            f"  spacings={self.spacings},\n"
            f"  scheme='{self.scheme}',\n"
            f"  form='{self.form}',\n"
            f"  bc={'None (periodic)' if self.bc is None else type(self.bc).__name__},\n"
            f"  operator_shape={self.shape}\n"
            f")"
        )


if __name__ == "__main__":
    """Smoke test for AdvectionOperator."""
    import numpy as np

    from mfg_pde.geometry.boundary import neumann_bc

    print("Testing AdvectionOperator...")

    # Test 2D gradient form: v·∇m with m = x, v = (1, 0)
    print("\n[Test 1: Gradient form v·∇m]")
    Nx, Ny = 50, 50
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    dx, dy = x[1] - x[0], y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing="ij")

    m = X  # m = x
    v = np.zeros((2, Nx, Ny))
    v[0] = 1.0  # v = (1, 0)

    bc = neumann_bc(dimension=2)
    adv_op = AdvectionOperator(
        velocity_field=v,
        spacings=[dx, dy],
        field_shape=(Nx, Ny),
        scheme="upwind",
        form="gradient",
        bc=bc,
    )

    print(f"  Operator: {adv_op.shape}")
    print(f"  Velocity field shape: {v.shape}")
    print(f"  Density shape: {m.shape}")

    # Test callable interface
    # v·∇(x) = (1,0)·(1,0) = 1.0
    adv_m = adv_op(m)
    print(f"  Advection shape: {adv_m.shape}")
    mean_adv = np.mean(adv_m[5:-5, 5:-5])  # Interior points
    print(f"  Mean v·∇(x) = {mean_adv:.3f} (expected ≈ 1.0)")
    assert adv_m.shape == (Nx, Ny)
    assert 0.8 < mean_adv < 1.2, f"Expected ~1.0, got {mean_adv}"
    print("  ✓ Gradient form works")

    # Test divergence form: ∇·(vm)
    print("\n[Test 2: Divergence form ∇·(vm)]")
    adv_op_div = AdvectionOperator(
        velocity_field=v,
        spacings=[dx, dy],
        field_shape=(Nx, Ny),
        scheme="upwind",
        form="divergence",
        bc=bc,
    )

    # ∇·(x * (1,0)) = ∂(x)/∂x = 1.0 (since ∂v/∂y = 0 and v_y = 0)
    adv_m_div = adv_op_div(m)
    mean_adv_div = np.mean(adv_m_div[5:-5, 5:-5])
    print(f"  Mean ∇·(vm) = {mean_adv_div:.3f} (expected ≈ 1.0)")
    assert 0.8 < mean_adv_div < 1.2
    print("  ✓ Divergence form works")

    # Test LinearOperator interface
    print("\n[Test 3: LinearOperator interface]")
    m_flat = m.ravel()
    adv_m_flat = adv_op @ m_flat

    print(f"  Input (flattened): {m_flat.shape}")
    print(f"  Output (flattened): {adv_m_flat.shape}")
    adv_m_reshaped = adv_m_flat.reshape(Nx, Ny)
    error = np.max(np.abs(adv_m_reshaped - adv_m))
    print(f"  Consistency check: max|matvec - callable| = {error:.2e}")
    assert error < 1e-12
    print("  ✓ LinearOperator interface works")

    # Test 1D case
    print("\n[Test 4: 1D advection]")
    Nx_1d = 100
    x_1d = np.linspace(0, 1, Nx_1d)
    dx_1d = x_1d[1] - x_1d[0]

    m_1d = x_1d**2  # m = x²
    v_1d = np.ones((1, Nx_1d))  # v = 1

    adv_op_1d = AdvectionOperator(
        velocity_field=v_1d,
        spacings=[dx_1d],
        field_shape=(Nx_1d,),
        scheme="centered",  # Test centered scheme
        form="gradient",
    )

    # v·∇(x²) = 1 · 2x = 2x
    adv_m_1d = adv_op_1d(m_1d)
    expected_1d = 2 * x_1d
    error_1d = np.max(np.abs(adv_m_1d[5:-5] - expected_1d[5:-5]))
    print(f"  Interior error: {error_1d:.2e} (expected = 2x)")
    assert error_1d < 0.05
    print("  ✓ 1D advection works")

    print("\n✅ All AdvectionOperator tests passed!")
