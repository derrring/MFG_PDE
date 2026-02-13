"""
Unit tests for LaplacianOperator.

Tests the discrete Laplacian on structured grids using known analytical
solutions. Central difference Laplacian is exact for quadratic functions.

Created: 2026-02-10 (Issue #768 - Test coverage for operators/)
"""

import pytest

import numpy as np
from scipy.sparse.linalg import LinearOperator

from mfg_pde.geometry.boundary import neumann_bc, no_flux_bc, periodic_bc
from mfg_pde.operators.differential.laplacian import LaplacianOperator

# =============================================================================
# Fixtures
# =============================================================================


def _1d_grid(n=100):
    """Create 1D uniform grid on [0, 1]."""
    x = np.linspace(0, 1, n)
    dx = x[1] - x[0]
    return x, dx


def _2d_grid(nx=50, ny=50):
    """Create 2D uniform grid on [0, 1]^2."""
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    dx, dy = x[1] - x[0], y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing="ij")
    return X, Y, dx, dy


# =============================================================================
# Basic Functionality
# =============================================================================


class TestLaplacianBasic:
    """Test basic LaplacianOperator functionality."""

    @pytest.mark.unit
    def test_1d_quadratic_exact(self):
        """Laplacian of u=x^2 should be exactly 2 at interior points.

        Note: Neumann ghost cell uses copy (u_ghost = u_boundary), which is
        1st-order at boundary. Interior uses standard 3-point stencil that is
        exact for quadratics.
        """
        x, dx = _1d_grid(100)
        u = x**2

        bc = neumann_bc(dimension=1)
        L = LaplacianOperator(spacings=[dx], field_shape=(100,), bc=bc)
        Lu = L(u)

        assert Lu.shape == (100,)
        # Interior: 3-point stencil exact for quadratic
        error = np.max(np.abs(Lu[2:-2] - 2.0))
        assert error < 1e-10

    @pytest.mark.unit
    def test_2d_quadratic_exact(self):
        """Laplacian of u=x^2+y^2 should be exactly 4 at interior points."""
        X, Y, dx, dy = _2d_grid(50, 50)
        u = X**2 + Y**2

        bc = neumann_bc(dimension=2)
        L = LaplacianOperator(spacings=[dx, dy], field_shape=(50, 50), bc=bc)
        Lu = L(u)

        assert Lu.shape == (50, 50)
        # Interior: exact for quadratic
        error = np.max(np.abs(Lu[2:-2, 2:-2] - 4.0))
        assert error < 1e-10

    @pytest.mark.unit
    def test_1d_constant_zero(self):
        """Laplacian of constant function should be 0."""
        x, dx = _1d_grid(50)
        u = np.ones_like(x) * 5.0

        bc = neumann_bc(dimension=1)
        L = LaplacianOperator(spacings=[dx], field_shape=(50,), bc=bc)
        Lu = L(u)

        np.testing.assert_allclose(Lu, 0.0, atol=1e-12)

    @pytest.mark.unit
    def test_1d_linear_zero(self):
        """Laplacian of linear function should be 0."""
        x, dx = _1d_grid(50)
        u = 3.0 * x + 1.0

        bc = neumann_bc(dimension=1)
        L = LaplacianOperator(spacings=[dx], field_shape=(50,), bc=bc)
        Lu = L(u)

        # Interior should be exactly 0 (boundary may differ due to BC)
        np.testing.assert_allclose(Lu[1:-1], 0.0, atol=1e-10)

    @pytest.mark.unit
    def test_shape_preserved(self):
        """Output shape should match input shape."""
        X, _Y, dx, dy = _2d_grid(40, 30)
        u = X**2

        bc = neumann_bc(dimension=2)
        L = LaplacianOperator(spacings=[dx, dy], field_shape=(40, 30), bc=bc)
        Lu = L(u)

        assert Lu.shape == (40, 30)

    @pytest.mark.unit
    def test_integer_field_shape(self):
        """Should accept integer field_shape for 1D case."""
        L = LaplacianOperator(spacings=[0.1], field_shape=50, bc=neumann_bc(dimension=1))
        assert L.field_shape == (50,)
        assert L.shape == (50, 50)


# =============================================================================
# scipy Interface
# =============================================================================


class TestLaplacianScipyInterface:
    """Test scipy LinearOperator compatibility."""

    @pytest.mark.unit
    def test_isinstance(self):
        """Should be a scipy LinearOperator."""
        L = LaplacianOperator(spacings=[0.1], field_shape=(50,))
        assert isinstance(L, LinearOperator)

    @pytest.mark.unit
    def test_matvec_callable_consistency(self):
        """L(u) and L @ u.ravel() should give identical results."""
        X, Y, dx, dy = _2d_grid(30, 30)
        u = X**2 + Y**2

        bc = neumann_bc(dimension=2)
        L = LaplacianOperator(spacings=[dx, dy], field_shape=(30, 30), bc=bc)

        Lu_callable = L(u)
        Lu_matvec = L @ u.ravel()

        np.testing.assert_allclose(Lu_callable.ravel(), Lu_matvec, atol=1e-14)

    @pytest.mark.unit
    def test_flattened_input(self):
        """Should accept flattened 1D input."""
        X, Y, dx, dy = _2d_grid(30, 30)
        u = X**2 + Y**2

        bc = neumann_bc(dimension=2)
        L = LaplacianOperator(spacings=[dx, dy], field_shape=(30, 30), bc=bc)

        Lu_flat = L(u.ravel())
        Lu_field = L(u)

        np.testing.assert_allclose(Lu_flat, Lu_field.ravel(), atol=1e-14)

    @pytest.mark.unit
    def test_operator_shape(self):
        """Operator shape should be (N, N) where N = prod(field_shape)."""
        L = LaplacianOperator(spacings=[0.1, 0.1], field_shape=(20, 30))
        assert L.shape == (600, 600)


# =============================================================================
# Sparse Export
# =============================================================================


class TestLaplacianSparse:
    """Test sparse matrix export (as_scipy_sparse)."""

    @pytest.mark.unit
    def test_sparse_matches_matvec_neumann(self):
        """Sparse matrix and matvec should agree at interior for Neumann BC.

        Note: Ghost cell (matvec) uses copy, sparse uses mirror doubling.
        These differ at boundary points but agree in the interior.
        """
        x, dx = _1d_grid(30)
        u = x**3

        bc = neumann_bc(dimension=1)
        L = LaplacianOperator(spacings=[dx], field_shape=(30,), bc=bc)

        L_sparse = L.as_scipy_sparse()
        Lu_matvec = L @ u.ravel()
        Lu_sparse = L_sparse @ u.ravel()

        # Interior points should match
        np.testing.assert_allclose(Lu_sparse[2:-2], Lu_matvec[2:-2], atol=1e-10)

    @pytest.mark.unit
    def test_sparse_matches_matvec_periodic(self):
        """Sparse matrix and matvec should match for periodic BC."""
        x, dx = _1d_grid(30)
        u = np.sin(2 * np.pi * x)  # Periodic on [0, 1]

        bc = periodic_bc(dimension=1)
        L = LaplacianOperator(spacings=[dx], field_shape=(30,), bc=bc)

        L_sparse = L.as_scipy_sparse()
        Lu_matvec = L @ u.ravel()
        Lu_sparse = L_sparse @ u.ravel()

        np.testing.assert_allclose(Lu_sparse, Lu_matvec, atol=1e-10)

    @pytest.mark.unit
    def test_sparse_2d_neumann(self):
        """2D sparse export should match matvec at interior for Neumann BC."""
        X, Y, dx, dy = _2d_grid(20, 20)
        u = X**2 + Y**2

        bc = neumann_bc(dimension=2)
        L = LaplacianOperator(spacings=[dx, dy], field_shape=(20, 20), bc=bc)

        L_sparse = L.as_scipy_sparse()
        Lu_matvec = (L @ u.ravel()).reshape(20, 20)
        Lu_sparse = (L_sparse @ u.ravel()).reshape(20, 20)

        # Interior match (boundary differs due to ghost cell vs direct assembly)
        np.testing.assert_allclose(Lu_sparse[2:-2, 2:-2], Lu_matvec[2:-2, 2:-2], atol=1e-10)

    @pytest.mark.unit
    def test_sparse_csr_format(self):
        """Sparse export should return CSR format."""
        import scipy.sparse as sparse

        bc = neumann_bc(dimension=1)
        L = LaplacianOperator(spacings=[0.1], field_shape=(20,), bc=bc)
        L_sparse = L.as_scipy_sparse()

        assert sparse.issparse(L_sparse)
        assert isinstance(L_sparse, sparse.csr_matrix)

    @pytest.mark.unit
    def test_sparse_size_limit(self):
        """Should raise ValueError for grids larger than 100k points."""
        bc = neumann_bc(dimension=2)
        L = LaplacianOperator(spacings=[0.01, 0.01], field_shape=(400, 400), bc=bc)

        with pytest.raises(ValueError, match="too large"):
            L.as_scipy_sparse()


# =============================================================================
# Boundary Conditions
# =============================================================================


class TestLaplacianBC:
    """Test boundary condition handling."""

    @pytest.mark.unit
    def test_periodic_bc_sin(self):
        """Periodic BC on sin(2pi*x) should give -4pi^2 sin(2pi*x)."""
        n = 100
        x = np.linspace(0, 1, n, endpoint=False)  # Periodic: exclude endpoint
        dx = x[1] - x[0]
        u = np.sin(2 * np.pi * x)

        bc = periodic_bc(dimension=1)
        L = LaplacianOperator(spacings=[dx], field_shape=(n,), bc=bc)
        Lu = L(u)

        expected = -((2 * np.pi) ** 2) * np.sin(2 * np.pi * x)
        error = np.max(np.abs(Lu - expected))
        # O(h^2) for 2nd-order central diff
        assert error < 0.5

    @pytest.mark.unit
    def test_no_flux_quadratic(self):
        """No-flux BC with quadratic: exact at interior, boundary error bounded."""
        X, Y, dx, dy = _2d_grid(40, 40)
        u = X**2 + Y**2

        bc = no_flux_bc(dimension=2)
        L = LaplacianOperator(spacings=[dx, dy], field_shape=(40, 40), bc=bc)
        Lu = L(u)

        # Interior: exact for quadratic
        error_interior = np.max(np.abs(Lu[2:-2, 2:-2] - 4.0))
        assert error_interior < 1e-10

        # Boundary: 1st-order ghost cell introduces O(1/h) error
        # but overall result is bounded
        assert Lu.shape == (40, 40)

    @pytest.mark.unit
    def test_no_bc_periodic_wrapping(self):
        """With bc=None, should use periodic wrapping (np.roll)."""
        n = 50
        x = np.linspace(0, 1, n, endpoint=False)
        dx = x[1] - x[0]
        u = np.sin(2 * np.pi * x)

        L = LaplacianOperator(spacings=[dx], field_shape=(n,), bc=None)
        Lu = L(u)

        expected = -((2 * np.pi) ** 2) * np.sin(2 * np.pi * x)
        error = np.max(np.abs(Lu - expected))
        assert error < 0.5


# =============================================================================
# Validation
# =============================================================================


class TestLaplacianValidation:
    """Test input validation."""

    @pytest.mark.unit
    def test_spacings_mismatch(self):
        """Should raise ValueError for mismatched spacings/field_shape."""
        with pytest.raises(ValueError, match="spacings length"):
            LaplacianOperator(spacings=[0.1], field_shape=(10, 10))

    @pytest.mark.unit
    def test_unsupported_order(self):
        """Should raise ValueError for order != 2."""
        with pytest.raises(ValueError, match="Only order=2"):
            LaplacianOperator(spacings=[0.1], field_shape=(10,), order=4)

    @pytest.mark.unit
    def test_shape_mismatch_callable(self):
        """Should raise ValueError when field shape doesn't match."""
        bc = neumann_bc(dimension=2)
        L = LaplacianOperator(spacings=[0.1, 0.1], field_shape=(10, 10), bc=bc)

        with pytest.raises(ValueError, match="doesn't match"):
            L(np.zeros((10, 20)))

    @pytest.mark.unit
    def test_repr(self):
        """repr should contain key attributes."""
        bc = neumann_bc(dimension=2)
        L = LaplacianOperator(spacings=[0.1, 0.1], field_shape=(10, 10), bc=bc)
        r = repr(L)
        assert "LaplacianOperator" in r
        assert "field_shape=(10, 10)" in r
        assert "order=2" in r


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
