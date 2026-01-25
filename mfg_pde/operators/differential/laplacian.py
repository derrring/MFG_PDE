"""
Laplacian operator for tensor product grids.

This module provides LinearOperator implementation of the discrete Laplacian
for structured grids, wrapping the tensor_calculus infrastructure.

Mathematical Background:
    Laplacian operator: Δu = ∇²u = ∑ᵢ ∂²u/∂xᵢ²

    Discretization (2nd-order central differences):
        ∂²u/∂x² ≈ (u[i+1] - 2u[i] + u[i-1]) / h²

    For multi-dimensional grids:
        Δu = Δₓu + Δᵧu + Δᵧu + ...

References:
    - LeVeque (2007): Finite Difference Methods for ODEs and PDEs
    - Strang (2007): Computational Science and Engineering

Created: 2026-01-17 (Issue #595 - Operator Refactoring)
Part of: Issue #590 Phase 1.2 - TensorProductGrid Operator Traits
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sparse  # noqa: TC002
from scipy.sparse.linalg import LinearOperator

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from mfg_pde.geometry.boundary import BoundaryConditions


class LaplacianOperator(LinearOperator):
    """
    Discrete Laplacian operator Δu = ∇²u for tensor product grids.

    Implements scipy.sparse.linalg.LinearOperator interface for compatibility
    with iterative solvers (gmres, cg, etc.) and operator composition.

    The operator wraps mfg_pde.utils.numerical.tensor_calculus.laplacian with
    grid-specific parameters (spacings, BC) curried into the operator object.

    Attributes:
        spacings: Grid spacing per dimension [h₀, h₁, ..., hd₋₁]
        field_shape: Shape of input field (Nx, Ny, ...) or (Nx,) for 1D
        bc: Boundary conditions (None for periodic)
        order: Discretization order (2, 4, 6, ...)
        shape: Operator shape (N, N) where N = ∏field_shape
        dtype: Data type (float64)

    Usage:
        >>> # Create operator
        >>> L = LaplacianOperator(spacings=[0.1, 0.1], field_shape=(50, 50), bc=bc)
        >>>
        >>> # Apply via matrix-vector product (scipy-compatible)
        >>> u_flat = u.ravel()
        >>> Lu_flat = L @ u_flat
        >>>
        >>> # Apply via callable (preserves field shape)
        >>> Lu = L(u)  # Same as (L @ u.ravel()).reshape(u.shape)
        >>>
        >>> # Use with iterative solvers
        >>> from scipy.sparse.linalg import gmres
        >>> u_solution, info = gmres(L, b)

    Example:
        >>> import numpy as np
        >>> from mfg_pde.geometry.boundary import neumann_bc
        >>>
        >>> # 2D Poisson problem: Δu = f with Neumann BC
        >>> bc = neumann_bc(dimension=2)
        >>> L = LaplacianOperator(spacings=[0.01, 0.01], field_shape=(100, 100), bc=bc)
        >>>
        >>> # Right-hand side
        >>> f = np.ones((100, 100))
        >>>
        >>> # Solve Δu = f
        >>> u_flat, info = gmres(L, f.ravel(), tol=1e-6)
        >>> u = u_flat.reshape(100, 100)
    """

    def __init__(
        self,
        spacings: Sequence[float],
        field_shape: tuple[int, ...] | int,
        bc: BoundaryConditions | None = None,
        order: int = 2,
        time: float = 0.0,
    ):
        """
        Initialize Laplacian operator.

        Args:
            spacings: Grid spacing per dimension [h₀, h₁, ..., hd₋₁]
            field_shape: Shape of field arrays (Nx, Ny, ...) or Nx for 1D
            bc: Boundary conditions (None for periodic/wrap)
            order: Discretization order (currently only order=2 supported)
            time: Time for time-dependent BCs (default 0.0)

        Raises:
            ValueError: If order != 2 (higher orders not yet implemented)
        """
        # Handle 1D shape
        if isinstance(field_shape, int):
            field_shape = (field_shape,)
        else:
            field_shape = tuple(field_shape)

        self.spacings = list(spacings)
        self.field_shape = field_shape
        self.bc = bc
        self.order = order
        self.time = time

        # Validate
        if len(self.spacings) != len(self.field_shape):
            raise ValueError(f"spacings length {len(self.spacings)} != field_shape dimensions {len(self.field_shape)}")

        if order != 2:
            raise ValueError(f"Only order=2 currently supported, got {order}")

        # Compute operator shape
        N = int(np.prod(field_shape))
        super().__init__(shape=(N, N), dtype=np.float64)

    def _matvec(self, u_flat: NDArray) -> NDArray:
        """
        Apply Laplacian to flattened field.

        This is the core LinearOperator method required by scipy.

        Args:
            u_flat: Flattened field array, shape (N,)

        Returns:
            Laplacian of u, flattened, shape (N,)
        """
        # Issue #625: Migrated from tensor_calculus to stencils
        from mfg_pde.operators.stencils.finite_difference import laplacian_with_bc

        # Reshape to field
        u = u_flat.reshape(self.field_shape)

        # Apply Laplacian using stencil with BC handling
        Lu = laplacian_with_bc(u, self.spacings, bc=self.bc, time=self.time)

        # Return flattened
        return Lu.ravel()

    def __call__(self, u: NDArray) -> NDArray:
        """
        Apply Laplacian to field (preserves shape).

        This method allows using the operator as L(u) in addition to L @ u.ravel().
        Automatically handles reshaping.

        Args:
            u: Field array, shape field_shape or (N,)

        Returns:
            Laplacian of u, same shape as input

        Example:
            >>> L = LaplacianOperator(spacings=[0.1, 0.1], field_shape=(50, 50))
            >>> u = np.random.rand(50, 50)
            >>> Lu = L(u)  # Returns (50, 50) array
        """
        # Handle already-flattened input
        if u.ndim == 1:
            return self._matvec(u)

        # Handle field input
        if u.shape != self.field_shape:
            raise ValueError(f"Input shape {u.shape} doesn't match field_shape {self.field_shape}")

        Lu_flat = self._matvec(u.ravel())
        return Lu_flat.reshape(self.field_shape)

    def as_scipy_sparse(self) -> sparse.spmatrix:
        """
        Export Laplacian as scipy sparse matrix (Issue #597 Milestone 2).

        This method enables using the Laplacian operator in implicit time-stepping
        schemes that require matrix representations (e.g., FP solver).

        Implementation uses direct sparse assembly with correct one-sided stencils
        at Neumann boundaries to match coefficient folding behavior.

        Returns:
            Sparse CSR matrix representation of Laplacian operator

        Example:
            >>> L_op = LaplacianOperator(spacings=[0.1, 0.1], field_shape=(50, 50), bc=bc)
            >>> L_matrix = L_op.as_scipy_sparse()
            >>> # Use in implicit scheme: (I - dt*D*L) @ u = rhs
            >>> import scipy.sparse as sp
            >>> A = sp.eye(2500) - 0.01 * 0.5 * L_matrix

        Notes:
            - Returns CSR format for efficient matrix-vector products
            - BC handling via direct sparse assembly (matches coefficient folding)
            - Neumann BC: Uses one-sided stencils at boundaries
            - Periodic BC: Uses wrapped indices
            - For large grids (N > 100k), raises ValueError

        Raises:
            ValueError: If grid too large (N > 100k points)
        """

        N = int(np.prod(self.field_shape))

        # Threshold for sparse construction
        if N > 100_000:
            raise ValueError(
                f"Grid size {N} too large for current as_scipy_sparse() implementation. "
                f"Use matrix-free methods (LinearOperator interface) instead."
            )

        # Use direct sparse assembly for correct BC handling
        return self._build_sparse_laplacian_direct()

    def _build_sparse_laplacian_direct(self) -> sparse.spmatrix:
        """
        Build sparse Laplacian via direct assembly with correct BC handling.

        This method implements the fix for Issue #597 Milestone 2, ensuring
        that Neumann boundary conditions use one-sided stencils to match
        coefficient folding behavior in FP solver.

        Returns:
            Sparse CSR matrix with correct boundary stencils
        """
        import scipy.sparse as sparse

        N = int(np.prod(self.field_shape))
        ndim = len(self.field_shape)

        # COO format for efficient construction
        row_indices = []
        col_indices = []
        data_values = []

        # Determine BC type
        bc_type = None
        if self.bc is not None:
            try:
                bc_type = self.bc.bc_type.value if hasattr(self.bc.bc_type, "value") else str(self.bc.bc_type)
            except AttributeError:
                bc_type = None

        # Build matrix by iterating over grid points
        for flat_idx in range(N):
            # Convert flat index to multi-index
            multi_idx = np.unravel_index(flat_idx, self.field_shape)

            # Iterate over dimensions
            for d in range(ndim):
                h = self.spacings[d]
                n_d = self.field_shape[d]
                i_d = multi_idx[d]

                # Determine if boundary point in dimension d
                at_min = i_d == 0
                at_max = i_d == n_d - 1
                is_boundary = at_min or at_max

                # Get neighbor indices in dimension d
                im1 = i_d - 1  # Left neighbor
                ip1 = i_d + 1  # Right neighbor

                # Handle BC
                if bc_type in ("neumann", "no_flux") and is_boundary:
                    # Neumann BC: Use one-sided stencil
                    # At min boundary (i=0): Δu ≈ (u[1] - u[0]) / h²
                    # At max boundary (i=n-1): Δu ≈ (u[n-2] - u[n-1]) / h²
                    if at_min:
                        # Diagonal contribution: -1/h²
                        row_indices.append(flat_idx)
                        col_indices.append(flat_idx)
                        data_values.append(-1.0 / (h**2))

                        # Right neighbor contribution: +1/h²
                        neighbor_idx = list(multi_idx)
                        neighbor_idx[d] = ip1
                        neighbor_flat = np.ravel_multi_index(tuple(neighbor_idx), self.field_shape)
                        row_indices.append(flat_idx)
                        col_indices.append(neighbor_flat)
                        data_values.append(+1.0 / (h**2))

                    elif at_max:
                        # Diagonal contribution: -1/h²
                        row_indices.append(flat_idx)
                        col_indices.append(flat_idx)
                        data_values.append(-1.0 / (h**2))

                        # Left neighbor contribution: +1/h²
                        neighbor_idx = list(multi_idx)
                        neighbor_idx[d] = im1
                        neighbor_flat = np.ravel_multi_index(tuple(neighbor_idx), self.field_shape)
                        row_indices.append(flat_idx)
                        col_indices.append(neighbor_flat)
                        data_values.append(+1.0 / (h**2))

                elif bc_type == "periodic" or bc_type is None:
                    # Periodic BC: Wrap indices
                    # Interior standard stencil: -2/h² diagonal, +1/h² off-diagonal

                    # Diagonal contribution: -2/h²
                    row_indices.append(flat_idx)
                    col_indices.append(flat_idx)
                    data_values.append(-2.0 / (h**2))

                    # Left neighbor
                    im1_wrapped = (im1 + n_d) % n_d
                    neighbor_idx_left = list(multi_idx)
                    neighbor_idx_left[d] = im1_wrapped
                    neighbor_flat_left = np.ravel_multi_index(tuple(neighbor_idx_left), self.field_shape)
                    row_indices.append(flat_idx)
                    col_indices.append(neighbor_flat_left)
                    data_values.append(+1.0 / (h**2))

                    # Right neighbor
                    ip1_wrapped = ip1 % n_d
                    neighbor_idx_right = list(multi_idx)
                    neighbor_idx_right[d] = ip1_wrapped
                    neighbor_flat_right = np.ravel_multi_index(tuple(neighbor_idx_right), self.field_shape)
                    row_indices.append(flat_idx)
                    col_indices.append(neighbor_flat_right)
                    data_values.append(+1.0 / (h**2))

                elif bc_type == "dirichlet":
                    # Dirichlet BC: Homogeneous (u=0 at boundary)
                    # Coefficient folding with LinearConstraint(weights={}, bias=0.0):
                    # - Empty weights means no folding occurs
                    # - Ghost contribution is simply omitted
                    # - Leaves full centered stencil diagonal: -2/dx²
                    # - Only one neighbor contributes: +1/dx²
                    if is_boundary:
                        # Diagonal contribution: -2/dx² (full centered)
                        row_indices.append(flat_idx)
                        col_indices.append(flat_idx)
                        data_values.append(-2.0 / (h**2))

                        # Only interior neighbor contributes
                        if at_min:
                            neighbor_idx = list(multi_idx)
                            neighbor_idx[d] = ip1
                            neighbor_flat = np.ravel_multi_index(tuple(neighbor_idx), self.field_shape)
                            row_indices.append(flat_idx)
                            col_indices.append(neighbor_flat)
                            data_values.append(+1.0 / (h**2))

                        elif at_max:
                            neighbor_idx = list(multi_idx)
                            neighbor_idx[d] = im1
                            neighbor_flat = np.ravel_multi_index(tuple(neighbor_idx), self.field_shape)
                            row_indices.append(flat_idx)
                            col_indices.append(neighbor_flat)
                            data_values.append(+1.0 / (h**2))
                    else:
                        # Interior: Standard centered stencil
                        row_indices.append(flat_idx)
                        col_indices.append(flat_idx)
                        data_values.append(-2.0 / (h**2))

                        neighbor_idx_left = list(multi_idx)
                        neighbor_idx_left[d] = im1
                        neighbor_flat_left = np.ravel_multi_index(tuple(neighbor_idx_left), self.field_shape)
                        row_indices.append(flat_idx)
                        col_indices.append(neighbor_flat_left)
                        data_values.append(+1.0 / (h**2))

                        neighbor_idx_right = list(multi_idx)
                        neighbor_idx_right[d] = ip1
                        neighbor_flat_right = np.ravel_multi_index(tuple(neighbor_idx_right), self.field_shape)
                        row_indices.append(flat_idx)
                        col_indices.append(neighbor_flat_right)
                        data_values.append(+1.0 / (h**2))

                else:
                    # Interior or unsupported BC: Standard centered stencil
                    if not is_boundary:
                        # Interior: -2/h² diagonal, +1/h² off-diagonal
                        row_indices.append(flat_idx)
                        col_indices.append(flat_idx)
                        data_values.append(-2.0 / (h**2))

                        neighbor_idx_left = list(multi_idx)
                        neighbor_idx_left[d] = im1
                        neighbor_flat_left = np.ravel_multi_index(tuple(neighbor_idx_left), self.field_shape)
                        row_indices.append(flat_idx)
                        col_indices.append(neighbor_flat_left)
                        data_values.append(+1.0 / (h**2))

                        neighbor_idx_right = list(multi_idx)
                        neighbor_idx_right[d] = ip1
                        neighbor_flat_right = np.ravel_multi_index(tuple(neighbor_idx_right), self.field_shape)
                        row_indices.append(flat_idx)
                        col_indices.append(neighbor_flat_right)
                        data_values.append(+1.0 / (h**2))

        # Build sparse matrix
        L_sparse = sparse.coo_matrix((data_values, (row_indices, col_indices)), shape=(N, N)).tocsr()

        return L_sparse

    def __repr__(self) -> str:
        """String representation for debugging."""
        bc_str = f"bc={self.bc.bc_type.value}" if self.bc else "bc=periodic"
        return (
            f"LaplacianOperator(\n"
            f"  field_shape={self.field_shape},\n"
            f"  spacings={self.spacings},\n"
            f"  {bc_str},\n"
            f"  order={self.order},\n"
            f"  shape={self.shape}\n"
            f")"
        )


if __name__ == "__main__":
    """Smoke test for LaplacianOperator."""
    print("Testing LaplacianOperator...")

    # Test 1D
    print("\n[1D Laplacian]")
    L_1d = LaplacianOperator(spacings=[0.1], field_shape=100)
    print(f"  Operator shape: {L_1d.shape}")
    print(f"  Field shape: {L_1d.field_shape}")

    u_1d = np.sin(np.linspace(0, 2 * np.pi, 100))
    Lu_1d = L_1d(u_1d)
    print(f"  Input shape: {u_1d.shape}, Output shape: {Lu_1d.shape}")
    assert Lu_1d.shape == u_1d.shape

    # Also test @ syntax
    Lu_1d_matvec = L_1d @ u_1d.ravel()
    assert np.allclose(Lu_1d.ravel(), Lu_1d_matvec)
    print("  ✓ L(u) == L @ u.ravel()")

    # Test 2D
    print("\n[2D Laplacian]")
    from mfg_pde.geometry.boundary import neumann_bc

    bc = neumann_bc(dimension=2)

    # Test on quadratic function: u = x² + y², Δu = 4
    Nx, Ny = 50, 50
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    dx, dy = x[1] - x[0], y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing="ij")
    u_2d = X**2 + Y**2

    L_2d = LaplacianOperator(spacings=[dx, dy], field_shape=(Nx, Ny), bc=bc)
    print(f"  Operator shape: {L_2d.shape}")
    print(f"  Field shape: {L_2d.field_shape}")
    print(f"  Grid spacing: dx={dx:.4f}, dy={dy:.4f}")
    print(f"  BC: {bc.bc_type.value}")

    Lu_2d = L_2d(u_2d)
    print(f"  Input shape: {u_2d.shape}, Output shape: {Lu_2d.shape}")
    assert Lu_2d.shape == u_2d.shape

    # Check interior values (should be ~4.0 for continuous case)
    # Analytical: Δ(x²+y²) = 2 + 2 = 4
    interior = Lu_2d[10:-10, 10:-10]
    mean_val = np.mean(interior)
    std_val = np.std(interior)
    print(f"  Δ(x²+y²) interior: mean={mean_val:.3f}, std={std_val:.3f} (expected = 4.0)")
    # Should be very close to 4.0 with correct spacing
    assert 3.5 < mean_val < 4.5, f"Mean value {mean_val} outside expected range [3.5, 4.5]"
    print(f"  ✓ Laplacian accuracy check passed (error = {abs(mean_val - 4.0):.3e})")

    # Test isinstance check with scipy
    print("\n[scipy compatibility]")
    assert isinstance(L_2d, LinearOperator)
    print("  ✓ isinstance(L, scipy.sparse.linalg.LinearOperator)")

    # Test repr
    print("\n[String representation]")
    print(L_2d)

    print("\n✅ All LaplacianOperator tests passed!")
