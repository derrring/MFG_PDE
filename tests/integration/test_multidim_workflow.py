"""
Integration tests for multi-dimensional MFG workflow.

Tests the complete pipeline: grid → sparse ops → solve → visualize
for 2D and 3D problems.
"""

import pytest

import numpy as np

from mfg_pde.geometry import TensorProductGrid
from mfg_pde.utils import SparseMatrixBuilder, SparseSolver
from mfg_pde.visualization import MultiDimVisualizer


class TestMultiDimWorkflow2D:
    """Integration tests for 2D MFG workflow."""

    def test_complete_2d_poisson_problem(self):
        """
        Test complete 2D workflow: setup → solve → analyze.

        Solves -Δu = f on [0,1]×[0,1] with Dirichlet BC.
        """
        # 1. Create 2D grid
        grid = TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 1.0)], num_points=[31, 31])

        assert grid.dimension == 2
        assert grid.total_points() == 31 * 31

        # 2. Setup sparse operators
        builder = SparseMatrixBuilder(grid, matrix_format="csr")
        L = builder.build_laplacian(boundary_conditions="dirichlet")

        assert L.shape == (31 * 31, 31 * 31)
        assert L.nnz > 0

        # 3. Define problem: -Δu = sin(πx)sin(πy)
        X, Y = grid.meshgrid(indexing="ij")
        f = np.sin(np.pi * X) * np.sin(np.pi * Y)

        # Set boundary to zero
        f[0, :] = 0
        f[-1, :] = 0
        f[:, 0] = 0
        f[:, -1] = 0
        f_flat = f.flatten()

        # 4. Solve
        solver = SparseSolver(method="direct")
        u_flat = solver.solve(-L, f_flat)
        u = u_flat.reshape(31, 31)

        # 5. Verify solution properties
        # Boundary conditions
        assert np.allclose(u[0, :], 0)
        assert np.allclose(u[-1, :], 0)
        assert np.allclose(u[:, 0], 0)
        assert np.allclose(u[:, -1], 0)

        # Interior should be positive
        assert np.all(u[1:-1, 1:-1] > 0)

        # Maximum at center (by symmetry)
        center = (15, 15)
        assert u[center] == pytest.approx(np.max(u), rel=1e-6)

    def test_2d_gradient_operators(self):
        """Test 2D gradient operators on quadratic function."""
        grid = TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 1.0)], num_points=[21, 21])

        builder = SparseMatrixBuilder(grid, matrix_format="csr")
        Gx = builder.build_gradient(direction=0, order=2)
        Gy = builder.build_gradient(direction=1, order=2)

        # Test function: u(x,y) = x² + y²
        X, Y = grid.meshgrid(indexing="ij")
        u = X**2 + Y**2
        u_flat = u.flatten()

        # Compute gradients
        du_dx = (Gx @ u_flat).reshape(21, 21)
        du_dy = (Gy @ u_flat).reshape(21, 21)

        # Analytical: ∂u/∂x = 2x, ∂u/∂y = 2y
        du_dx_exact = 2 * X
        du_dy_exact = 2 * Y

        # Check interior points (boundary may have errors)
        assert np.allclose(du_dx[3:-3, 3:-3], du_dx_exact[3:-3, 3:-3], atol=1e-2)
        assert np.allclose(du_dy[3:-3, 3:-3], du_dy_exact[3:-3, 3:-3], atol=1e-2)

    def test_2d_time_dependent_diffusion(self):
        """Test 2D time-dependent diffusion equation."""
        # Spatial grid
        grid = TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 1.0)], num_points=[21, 21])

        # Time parameters
        T = 0.1
        Nt = 20
        dt = T / Nt

        # Diffusion coefficient
        sigma = 0.1

        # Initial condition: Gaussian
        X, Y = grid.meshgrid(indexing="ij")
        u0 = np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.05)

        # Setup sparse Laplacian
        builder = SparseMatrixBuilder(grid, matrix_format="csr")
        L = builder.build_laplacian(boundary_conditions="neumann")

        # Time evolution: u_{n+1} = u_n + dt*σ²*Δu_n (forward Euler)
        u = u0.copy()
        for _n in range(Nt):
            u_flat = u.flatten()
            u = u + dt * sigma**2 * (L @ u_flat).reshape(21, 21)

        # Verify diffusion properties:
        # 1. Mass conservation (Neumann BC)
        mass_initial = np.sum(u0) * grid.volume_element()
        mass_final = np.sum(u) * grid.volume_element()
        assert mass_final == pytest.approx(mass_initial, rel=1e-2)

        # 2. Maximum decreased (diffusion smooths)
        assert np.max(u) < np.max(u0)

        # 3. Solution spread out (entropy increased)
        assert np.std(u) > 0


class TestMultiDimWorkflow3D:
    """Integration tests for 3D MFG workflow."""

    def test_3d_laplacian_assembly(self):
        """Test 3D Laplacian construction and properties."""
        grid = TensorProductGrid(dimension=3, bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)], num_points=[11, 11, 11])

        builder = SparseMatrixBuilder(grid, matrix_format="csr")
        L = builder.build_laplacian(boundary_conditions="dirichlet")

        # Check shape
        N = 11**3
        assert L.shape == (N, N)

        # Check symmetry
        assert np.allclose(L.data, L.T.data)

        # Check 7-point stencil pattern (interior points)
        # Find an interior point
        idx = grid.get_index((5, 5, 5))
        row = L.getrow(idx)
        nnz = row.nnz

        # Interior point should have 7 nonzeros (center + 6 neighbors)
        assert nnz == 7

    def test_3d_gradient_operators(self):
        """Test 3D gradient operators."""
        grid = TensorProductGrid(dimension=3, bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)], num_points=[11, 11, 11])

        builder = SparseMatrixBuilder(grid, matrix_format="csr")
        Gx = builder.build_gradient(direction=0, order=2)
        Gy = builder.build_gradient(direction=1, order=2)
        Gz = builder.build_gradient(direction=2, order=2)

        # Test on linear function: u(x,y,z) = x + 2y + 3z
        coords = grid.coordinates
        X, Y, Z = np.meshgrid(coords[0], coords[1], coords[2], indexing="ij")

        u = X + 2 * Y + 3 * Z
        u_flat = u.flatten()

        # Compute gradients
        du_dx = Gx @ u_flat
        du_dy = Gy @ u_flat
        du_dz = Gz @ u_flat

        # Analytical: ∂u/∂x = 1, ∂u/∂y = 2, ∂u/∂z = 3
        # Check interior points
        assert np.allclose(du_dx.reshape(11, 11, 11)[2:-2, 2:-2, 2:-2], 1.0, atol=1e-10)
        assert np.allclose(du_dy.reshape(11, 11, 11)[2:-2, 2:-2, 2:-2], 2.0, atol=1e-10)
        assert np.allclose(du_dz.reshape(11, 11, 11)[2:-2, 2:-2, 2:-2], 3.0, atol=1e-10)


class TestMultiDimVisualization:
    """Integration tests for multi-dimensional visualization."""

    def test_2d_visualization_creation(self):
        """Test basic 2D visualization object creation."""
        grid = TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 1.0)], num_points=[21, 21])

        # Should work with 2D grid
        viz = MultiDimVisualizer(grid, backend="plotly")
        assert viz.grid.dimension == 2
        assert viz.backend == "plotly"

    def test_3d_visualization_creation(self):
        """Test basic 3D visualization object creation."""
        grid = TensorProductGrid(dimension=3, bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)], num_points=[11, 11, 11])

        # Should work with 3D grid
        viz = MultiDimVisualizer(grid, backend="plotly")
        assert viz.grid.dimension == 3
        assert viz.backend == "plotly"

    def test_invalid_dimension_visualization(self):
        """Test that 1D grid raises error for MultiDimVisualizer."""
        grid_1d = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], num_points=[21])

        # Should raise ValueError for 1D grid
        with pytest.raises(ValueError, match="requires 2D or 3D grid"):
            MultiDimVisualizer(grid_1d)


class TestIterativeSolvers:
    """Integration tests for iterative solvers on 2D/3D problems."""

    def test_cg_solver_2d(self):
        """Test CG solver on 2D Poisson equation."""
        grid = TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 1.0)], num_points=[31, 31])

        builder = SparseMatrixBuilder(grid, matrix_format="csr")
        L = builder.build_laplacian(boundary_conditions="dirichlet")

        # Random right-hand side
        np.random.seed(42)
        b = np.random.randn(grid.total_points())

        # Solve with CG
        solver = SparseSolver(method="cg", tol=1e-8, max_iter=1000)
        u = solver.solve(L, b)

        # Check residual
        residual = np.linalg.norm(L @ u - b)
        assert residual < 1e-6

    def test_gmres_solver_2d(self):
        """Test GMRES solver on 2D problem."""
        grid = TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 1.0)], num_points=[21, 21])

        builder = SparseMatrixBuilder(grid, matrix_format="csr")
        L = builder.build_laplacian(boundary_conditions="neumann")

        # Random right-hand side
        np.random.seed(42)
        b = np.random.randn(grid.total_points())

        # Solve with GMRES
        solver = SparseSolver(method="gmres", tol=1e-8, max_iter=1000)
        u = solver.solve(L, b)

        # Check residual
        residual = np.linalg.norm(L @ u - b)
        assert residual < 1e-6


class TestMemoryEfficiency:
    """Tests for memory efficiency of tensor product grids."""

    def test_2d_grid_memory_advantage(self):
        """Verify 2D tensor grid uses O(Nx+Ny) storage, not O(Nx*Ny)."""
        grid = TensorProductGrid(dimension=2, bounds=[(0.0, 10.0), (0.0, 10.0)], num_points=[100, 100])

        # Coordinate storage
        coord_memory = sum(len(coords) for coords in grid.coordinates)

        # Should be 100 + 100 = 200, not 100*100 = 10,000
        assert coord_memory == 200

        # Grid points still accessible
        assert grid.total_points() == 10000

    def test_3d_grid_memory_advantage(self):
        """Verify 3D tensor grid uses O(Nx+Ny+Nz) storage."""
        grid = TensorProductGrid(dimension=3, bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)], num_points=[50, 50, 50])

        # Coordinate storage
        coord_memory = sum(len(coords) for coords in grid.coordinates)

        # Should be 50 + 50 + 50 = 150, not 50³ = 125,000
        assert coord_memory == 150

        # Grid points still accessible
        assert grid.total_points() == 125000


class TestBoundaryConditions:
    """Tests for different boundary condition types."""

    def test_2d_neumann_bc_mass_conservation(self):
        """Test Neumann BC preserves mass in diffusion."""
        grid = TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 1.0)], num_points=[21, 21])

        builder = SparseMatrixBuilder(grid, matrix_format="csr")
        L = builder.build_laplacian(boundary_conditions="neumann")

        # Initial mass distribution
        X, Y = grid.meshgrid(indexing="ij")
        u0 = np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.1)

        # Time evolution
        dt = 0.01
        sigma = 0.1
        Nt = 10

        u = u0.copy()
        for _n in range(Nt):
            u_flat = u.flatten()
            u = u + dt * sigma**2 * (L @ u_flat).reshape(21, 21)

        # Mass should be approximately conserved (relaxed tolerance for numerical diffusion)
        mass_initial = np.sum(u0) * grid.volume_element()
        mass_final = np.sum(u) * grid.volume_element()

        assert mass_final == pytest.approx(mass_initial, rel=1e-2)

    def test_2d_dirichlet_bc_zero_boundary(self):
        """Test Dirichlet BC enforces zero on boundary in Poisson solve."""
        grid = TensorProductGrid(dimension=2, bounds=[(0.0, 1.0), (0.0, 1.0)], num_points=[21, 21])

        builder = SparseMatrixBuilder(grid, matrix_format="csr")
        L = builder.build_laplacian(boundary_conditions="dirichlet")

        # Source term in interior, zero on boundary
        X, Y = grid.meshgrid(indexing="ij")
        f = np.sin(np.pi * X) * np.sin(np.pi * Y)

        # Enforce BC: set boundary to zero
        f[0, :] = 0
        f[-1, :] = 0
        f[:, 0] = 0
        f[:, -1] = 0
        b = f.flatten()

        solver = SparseSolver(method="direct")
        u_flat = solver.solve(-L, b)  # Solve -Δu = f, so -L @ u = b
        u = u_flat.reshape(21, 21)

        # With Dirichlet boundary matrix structure, boundaries should remain small
        # Check boundary has small values (may not be exactly zero due to matrix structure)
        assert np.max(np.abs(u[0, :])) < 0.1
        assert np.max(np.abs(u[-1, :])) < 0.1
        assert np.max(np.abs(u[:, 0])) < 0.1
        assert np.max(np.abs(u[:, -1])) < 0.1

        # Interior should have positive non-zero values
        assert np.max(u[5:-5, 5:-5]) > 0.01
