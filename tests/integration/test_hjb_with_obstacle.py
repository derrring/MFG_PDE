"""
Integration tests for HJB solvers with obstacle constraints.

Tests that HJB FDM solver correctly handles ObstacleConstraint (Tier 2 BCs)
through complete solve workflows.

Created: 2026-01-18 (Issue #594 Phase 5.3)
"""

import pytest

import numpy as np

from mfg_pde import MFGProblem
from mfg_pde.alg.numerical.hjb_solvers import HJBFDMSolver
from mfg_pde.geometry import TensorProductGrid
from mfg_pde.geometry.boundary import ObstacleConstraint, neumann_bc


class TestHJBWithLowerObstacle:
    """Test HJB solver with lower obstacle constraint (u ≥ ψ)."""

    def test_1d_parabolic_obstacle_convergence(self):
        """Test HJB solver converges with parabolic obstacle."""
        # Setup
        x_min, x_max = 0.0, 1.0
        Nx = 100
        T = 1.0
        Nt = 50
        sigma = 0.1
        kappa = 0.5

        grid = TensorProductGrid(dimension=1, bounds=[(x_min, x_max)], Nx=[Nx])
        bc = neumann_bc(dimension=1)

        def running_cost(x_coords, alpha=None):
            return 0.5 * (x_coords[0] - 0.5) ** 2

        def terminal_cost(x_coords):
            return (x_coords[0] - 0.5) ** 2

        problem = MFGProblem(
            geometry=grid,
            T=T,
            Nt=Nt,
            diffusion=sigma,
            bc=bc,
            running_cost=running_cost,
            terminal_cost=terminal_cost,
        )

        # Obstacle: ψ(x) = -κ(x - 0.5)²
        x = grid.coordinates[0]
        psi = -kappa * (x - 0.5) ** 2
        obstacle = ObstacleConstraint(lower_bound=psi, upper_bound=None)

        # Solve
        solver = HJBFDMSolver(problem, constraint=obstacle, tolerance=1e-6, max_iterations=100)
        result = solver.solve()

        # Assertions
        assert result.converged, "Solver should converge with obstacle"
        assert result.iterations < 100, "Should converge in reasonable iterations"

        # Check constraint satisfaction
        u_final = result.U[0, :]
        assert np.all(u_final >= psi - 1e-8), "Solution must satisfy u ≥ ψ"

        # Check active set exists
        active_set = np.abs(u_final - psi) < 1e-3
        assert np.sum(active_set) > 0, "Active set should be non-empty"

    def test_2d_obstacle_solver_integration(self):
        """Test HJB solver with 2D obstacle."""
        Nx, Ny = 30, 30
        T = 0.5
        Nt = 25
        sigma = 0.05

        grid = TensorProductGrid(dimension=2, bounds=[(0, 1), (0, 1)], Nx=[Nx, Ny])
        bc = neumann_bc(dimension=2)

        def terminal_cost(x_coords):
            return (x_coords[0] - 0.5) ** 2 + (x_coords[1] - 0.5) ** 2

        problem = MFGProblem(geometry=grid, T=T, Nt=Nt, diffusion=sigma, bc=bc, terminal_cost=terminal_cost)

        # Obstacle: Bowl-shaped
        X, Y = grid.meshgrid()
        psi = -0.2 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2)
        obstacle = ObstacleConstraint(lower_bound=psi.ravel(), upper_bound=None)

        # Solve
        solver = HJBFDMSolver(problem, constraint=obstacle, tolerance=1e-5, max_iterations=80)
        result = solver.solve()

        # Assertions
        assert result.converged, "2D solver should converge"
        u_final = result.U[0, :].reshape(Nx + 1, Ny + 1)
        assert np.all(u_final >= psi - 1e-7), "2D solution must respect obstacle"

    def test_obstacle_without_constraint_comparison(self):
        """Compare solution with and without obstacle constraint."""
        Nx = 80
        T = 0.8
        Nt = 40
        sigma = 0.08
        kappa = 0.3

        grid = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx=[Nx])
        bc = neumann_bc(dimension=1)

        def terminal_cost(x_coords):
            return (x_coords[0] - 0.5) ** 2

        problem = MFGProblem(geometry=grid, T=T, Nt=Nt, diffusion=sigma, bc=bc, terminal_cost=terminal_cost)

        x = grid.coordinates[0]
        psi = -kappa * (x - 0.5) ** 2

        # Solve without obstacle
        solver_free = HJBFDMSolver(problem, tolerance=1e-6)
        result_free = solver_free.solve()

        # Solve with obstacle
        obstacle = ObstacleConstraint(lower_bound=psi)
        solver_constrained = HJBFDMSolver(problem, constraint=obstacle, tolerance=1e-6)
        result_constrained = solver_constrained.solve()

        # Compare
        u_free = result_free.U[0, :]
        u_constrained = result_constrained.U[0, :]

        # Constrained solution should be ≥ free solution (obstacle raises floor)
        assert np.all(u_constrained >= u_free - 1e-6), "Obstacle should raise solution"

        # Should differ meaningfully where obstacle is active
        difference = np.abs(u_constrained - u_free)
        assert np.max(difference) > 0.01, "Obstacle should have visible effect"


class TestHJBWithUpperObstacle:
    """Test HJB solver with upper obstacle constraint (u ≤ ψ_upper)."""

    def test_1d_upper_ceiling(self):
        """Test HJB solver with upper obstacle (ceiling)."""
        Nx = 100
        T = 1.0
        Nt = 50
        sigma = 0.1

        grid = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx=[Nx])
        bc = neumann_bc(dimension=1)

        def terminal_cost(x_coords):
            return (x_coords[0] - 0.5) ** 2

        problem = MFGProblem(geometry=grid, T=T, Nt=Nt, diffusion=sigma, bc=bc, terminal_cost=terminal_cost)

        # Upper obstacle: ceiling at ψ_upper = 0.3
        x = grid.coordinates[0]
        psi_upper = 0.3 + 0.0 * x  # Constant ceiling
        obstacle = ObstacleConstraint(lower_bound=None, upper_bound=psi_upper)

        # Solve
        solver = HJBFDMSolver(problem, constraint=obstacle, tolerance=1e-6)
        result = solver.solve()

        # Assertions
        assert result.converged, "Solver should converge with upper obstacle"
        u_final = result.U[0, :]
        assert np.all(u_final <= psi_upper + 1e-8), "Solution must satisfy u ≤ ψ_upper"

        # Active set at ceiling
        active_ceiling = np.abs(u_final - psi_upper) < 1e-3
        assert np.sum(active_ceiling) > 0, "Should have active ceiling constraint"


class TestHJBWithBilateralObstacle:
    """Test HJB solver with bilateral obstacle (ψ_lower ≤ u ≤ ψ_upper)."""

    def test_1d_corridor_constraint(self):
        """Test bilateral obstacle creating solution corridor."""
        Nx = 100
        T = 1.0
        Nt = 50
        sigma = 0.1

        grid = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx=[Nx])
        bc = neumann_bc(dimension=1)

        def terminal_cost(x_coords):
            return 0.5 * (x_coords[0] - 0.5) ** 2

        problem = MFGProblem(geometry=grid, T=T, Nt=Nt, diffusion=sigma, bc=bc, terminal_cost=terminal_cost)

        # Bilateral obstacle: corridor between -0.2 and 0.3
        x = grid.coordinates[0]
        psi_lower = -0.2 + 0.0 * x
        psi_upper = 0.3 + 0.0 * x
        obstacle = ObstacleConstraint(lower_bound=psi_lower, upper_bound=psi_upper)

        # Solve
        solver = HJBFDMSolver(problem, constraint=obstacle, tolerance=1e-6)
        result = solver.solve()

        # Assertions
        assert result.converged, "Bilateral obstacle should converge"
        u_final = result.U[0, :]

        # Check both constraints
        assert np.all(u_final >= psi_lower - 1e-8), "Must satisfy lower bound"
        assert np.all(u_final <= psi_upper + 1e-8), "Must satisfy upper bound"

        # Solution should be inside corridor
        assert np.all((u_final >= psi_lower) & (u_final <= psi_upper)), "Solution in corridor"


class TestObstacleConvergenceProperties:
    """Test convergence properties with obstacles."""

    def test_tolerance_scaling(self):
        """Test that tighter tolerance improves accuracy."""
        Nx = 60
        T = 0.5
        Nt = 30
        sigma = 0.08
        kappa = 0.4

        grid = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx=[Nx])
        bc = neumann_bc(dimension=1)

        def terminal_cost(x_coords):
            return (x_coords[0] - 0.5) ** 2

        problem = MFGProblem(geometry=grid, T=T, Nt=Nt, diffusion=sigma, bc=bc, terminal_cost=terminal_cost)

        x = grid.coordinates[0]
        psi = -kappa * (x - 0.5) ** 2
        obstacle = ObstacleConstraint(lower_bound=psi)

        # Solve with loose tolerance
        solver_loose = HJBFDMSolver(problem, constraint=obstacle, tolerance=1e-4)
        result_loose = solver_loose.solve()

        # Solve with tight tolerance
        solver_tight = HJBFDMSolver(problem, constraint=obstacle, tolerance=1e-7)
        result_tight = solver_tight.solve()

        # Both should converge
        assert result_loose.converged, "Loose tolerance should converge"
        assert result_tight.converged, "Tight tolerance should converge"

        # Tight tolerance should use more iterations
        assert result_tight.iterations >= result_loose.iterations

        # Final errors should differ
        assert result_tight.error_history_U[-1] < result_loose.error_history_U[-1]

    def test_complementarity_satisfaction(self):
        """Test complementarity condition: (u - ψ)·residual ≈ 0."""
        Nx = 80
        T = 0.8
        Nt = 40
        sigma = 0.1
        kappa = 0.5

        grid = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx=[Nx])
        bc = neumann_bc(dimension=1)

        def running_cost(x_coords, alpha=None):
            return 0.3 * (x_coords[0] - 0.5) ** 2

        def terminal_cost(x_coords):
            return (x_coords[0] - 0.5) ** 2

        problem = MFGProblem(
            geometry=grid,
            T=T,
            Nt=Nt,
            diffusion=sigma,
            bc=bc,
            running_cost=running_cost,
            terminal_cost=terminal_cost,
        )

        x = grid.coordinates[0]
        psi = -kappa * (x - 0.5) ** 2
        obstacle = ObstacleConstraint(lower_bound=psi)

        # Solve
        solver = HJBFDMSolver(problem, constraint=obstacle, tolerance=1e-6)
        result = solver.solve()
        u_final = result.U[0, :]

        # Compute HJB residual (simplified: just diffusion term)
        laplacian_op = grid.get_laplacian_operator(order=2, bc=bc)
        Lu = laplacian_op @ u_final
        residual = -(sigma**2 / 2) * Lu - running_cost(grid.coordinates)

        # Complementarity: (u - ψ) * residual ≈ 0
        complementarity = (u_final - psi) * residual
        max_violation = np.abs(complementarity).max()

        # Should be small (within solver tolerance order)
        assert max_violation < 1e-3, f"Complementarity violation: {max_violation}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
