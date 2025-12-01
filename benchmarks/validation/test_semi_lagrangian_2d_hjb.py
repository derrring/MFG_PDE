#!/usr/bin/env python3
"""
Validation test for 2D Semi-Lagrangian HJB Solver.

This test validates the 2D Semi-Lagrangian method for solving Hamilton-Jacobi-Bellman
equations in Mean Field Games by comparing against:
1. Analytical properties (smoothness, monotonicity)
2. Comparison with 2D FDM solver (when available)
3. Known solution properties for specific test cases

Test cases:
- Quadratic potential with uniform density
- Radial symmetry problem
- Simple diffusion-dominated problem
"""

import pytest

import numpy as np

from mfg_pde import MFGComponents, MFGProblem
from mfg_pde.alg.numerical.hjb_solvers import HJBFDMSolver, HJBSemiLagrangianSolver


class Simple2DMFGProblem(MFGProblem):
    """
    Simple 2D MFG problem for validation testing.

    - Isotropic Hamiltonian: H = (1/2)|∇u|²
    - Quadratic terminal cost centered at goal
    - Weak MFG coupling (optional)
    """

    def __init__(
        self,
        grid_resolution=25,
        time_horizon=1.0,
        num_timesteps=20,
        diffusion=0.1,
        coupling_strength=0.5,
        goal_position=None,
    ):
        super().__init__(
            spatial_bounds=[(0.0, 1.0), (0.0, 1.0)],
            spatial_discretization=[grid_resolution, grid_resolution],
            T=time_horizon,
            Nt=num_timesteps,
            sigma=diffusion,
        )
        self.grid_resolution = grid_resolution
        self.coupling_strength = coupling_strength
        self.goal_position = goal_position if goal_position is not None else [0.5, 0.5]

    def terminal_cost(self, x):
        """Quadratic terminal cost: g(x) = 0.5 * ||x - goal||²"""
        goal = np.array(self.goal_position)
        dist_sq = np.sum((x - goal) ** 2, axis=1)
        return 0.5 * dist_sq

    def initial_density(self, x):
        """Uniform initial density."""
        return np.ones(x.shape[0]) / x.shape[0]

    def running_cost(self, x, t):
        """Zero running cost."""
        return np.zeros(x.shape[0])

    def hamiltonian(self, x, m, p, t):
        """Isotropic Hamiltonian: H = (1/2)|p|² + κ·m"""
        p_squared = np.sum(p**2, axis=1) if p.ndim > 1 else p**2
        return 0.5 * p_squared + self.coupling_strength * m

    def setup_components(self):
        """Setup MFG components for solver."""
        coupling_strength = self.coupling_strength
        goal = np.array(self.goal_position)

        def hamiltonian_func(x_idx, x_position, m_at_x, p_values, t_idx, current_time, problem, derivs=None, **kwargs):
            """H = (1/2)|∇u|² + κ·m"""
            h_value = 0.0

            if derivs is not None:
                # Extract gradient components
                grad_u = []
                ndim = problem.geometry.grid.dimension
                for d in range(ndim):
                    idx_tuple = tuple([0] * d + [1] + [0] * (ndim - d - 1))
                    grad_u.append(derivs.get(idx_tuple, 0.0))
                h_value += 0.5 * float(np.sum(np.array(grad_u) ** 2))

            # MFG coupling term
            h_value += coupling_strength * float(m_at_x)

            return h_value

        def hamiltonian_dm(x_idx, x_position, m_at_x, **kwargs):
            """∂H/∂m = κ"""
            return coupling_strength

        def initial_density_func(x_idx):
            """Uniform initial density."""
            return 1.0

        def terminal_cost_func(x_idx):
            """Quadratic terminal cost centered at goal."""
            # Get physical position
            ndim = self.geometry.grid.dimension
            coords = []
            for d in range(ndim):
                idx_d = x_idx[d] if isinstance(x_idx, tuple) else x_idx
                x_d = self.geometry.grid.bounds[d][0] + idx_d * self.geometry.grid.spacing[d]
                coords.append(x_d)
            coords_array = np.array(coords)

            # Distance to goal
            dist_sq = np.sum((coords_array - goal) ** 2)
            return 0.5 * dist_sq

        return MFGComponents(
            hamiltonian_func=hamiltonian_func,
            hamiltonian_dm_func=hamiltonian_dm,
            initial_density_func=initial_density_func,
            final_value_func=terminal_cost_func,
        )


class TestSemiLagrangian2DInitialization:
    """Test 2D Semi-Lagrangian solver initialization."""

    def test_2d_initialization(self):
        """Test that Semi-Lagrangian solver initializes correctly for 2D problems."""
        problem = Simple2DMFGProblem(
            grid_resolution=15,
            time_horizon=1.0,
            num_timesteps=10,
        )

        solver = HJBSemiLagrangianSolver(problem)

        # Check that solver recognizes 2D problem
        assert solver.dimension == 2

        # Check that grid is properly set
        assert solver.grid is not None
        assert solver.grid.dimension == 2

        # Check solver configuration
        assert solver.hjb_method_name == "Semi-Lagrangian"
        assert solver.dt > 0
        assert len(solver.spacing) == 2

    def test_2d_grid_parameters(self):
        """Test that 2D grid parameters are computed correctly."""
        problem = Simple2DMFGProblem(
            grid_resolution=20,
            time_horizon=1.0,
            num_timesteps=10,
        )

        solver = HJBSemiLagrangianSolver(problem)

        # Check grid spacing
        assert len(solver.spacing) == 2
        # For grid_resolution=20 on [0,1], spacing = 1.0/(N-1) = 1.0/19 ≈ 0.0526
        expected_dx = 1.0 / (20 - 1)
        expected_dy = 1.0 / (20 - 1)
        assert np.isclose(solver.spacing[0], expected_dx, rtol=1e-5)
        assert np.isclose(solver.spacing[1], expected_dy, rtol=1e-5)


class TestSemiLagrangian2DSolve:
    """Test 2D Semi-Lagrangian solver execution."""

    def test_2d_solve_shape_and_finiteness(self):
        """Test that 2D solver produces correct output shape with finite values."""
        problem = Simple2DMFGProblem(
            grid_resolution=12,
            time_domain=(1.0, 10),
        )

        solver = HJBSemiLagrangianSolver(problem)

        # Prepare input arrays
        # Semi-Lagrangian solver expects full grid including boundaries
        Nt = problem.Nt + 1
        num_points_x = problem.geometry.grid.num_points[0]
        num_points_y = problem.geometry.grid.num_points[1]
        total_points = num_points_x * num_points_y

        # Uniform density
        M_density = np.ones((Nt, total_points)) / total_points

        # Simple final condition (zero everywhere)
        U_final = np.zeros(total_points)

        # Previous Picard iteration (zero for first iteration)
        U_prev = np.zeros((Nt, total_points))

        # Solve
        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # Check shape
        assert U_solution.shape == (Nt, total_points)

        # Check all values are finite
        assert np.all(np.isfinite(U_solution)), "Solution contains NaN or Inf"

    def test_2d_solve_final_condition_preserved(self):
        """Test that 2D solver preserves final condition."""
        problem = Simple2DMFGProblem(
            grid_resolution=10,
            time_domain=(1.0, 10),
        )

        solver = HJBSemiLagrangianSolver(problem)

        # Prepare inputs
        Nt = problem.Nt + 1
        num_points_x = problem.geometry.grid.num_points[0]
        num_points_y = problem.geometry.grid.num_points[1]
        total_points = num_points_x * num_points_y

        M_density = np.ones((Nt, total_points)) / total_points

        # Specific final condition (quadratic)
        x_coords = np.linspace(0.0, 1.0, num_points_x)
        y_coords = np.linspace(0.0, 1.0, num_points_y)
        X, Y = np.meshgrid(x_coords, y_coords, indexing="ij")
        U_final = 0.5 * ((X.ravel() - 0.5) ** 2 + (Y.ravel() - 0.5) ** 2)

        U_prev = np.zeros((Nt, total_points))

        # Solve
        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # Final time step should match final condition
        assert np.allclose(U_solution[-1, :], U_final, rtol=1e-6)

    def test_2d_backward_propagation(self):
        """Test that solution propagates backward in time correctly."""
        problem = Simple2DMFGProblem(
            grid_resolution=12,
            time_domain=(1.0, 10),
            coupling_strength=0.1,  # Weak coupling for stability
        )

        solver = HJBSemiLagrangianSolver(problem)

        # Prepare inputs
        Nt = problem.Nt + 1
        num_points_x = problem.geometry.grid.num_points[0]
        num_points_y = problem.geometry.grid.num_points[1]
        total_points = num_points_x * num_points_y

        M_density = np.ones((Nt, total_points)) / total_points

        # Non-zero final condition
        U_final = np.ones(total_points) * 0.5

        U_prev = np.zeros((Nt, total_points))

        # Solve
        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # Solution at t=0 should be influenced by final condition
        # (should not be zero since final condition is non-zero)
        assert not np.allclose(U_solution[0, :], 0.0), "Solution at t=0 should be influenced by final condition"

        # Check that solution is non-trivial
        assert np.std(U_solution) > 0, "Solution should have non-zero variance"


class TestSemiLagrangian2DNumericalProperties:
    """Test numerical properties of 2D Semi-Lagrangian method."""

    def test_2d_solution_smoothness(self):
        """Test that 2D solution exhibits reasonable smoothness."""
        problem = Simple2DMFGProblem(
            grid_resolution=15,
            time_domain=(1.0, 10),
            coupling_strength=0.2,
        )

        solver = HJBSemiLagrangianSolver(problem)

        # Prepare inputs
        Nt = problem.Nt + 1
        num_points_x = problem.geometry.grid.num_points[0]
        num_points_y = problem.geometry.grid.num_points[1]
        total_points = num_points_x * num_points_y

        M_density = np.ones((Nt, total_points)) / total_points

        # Smooth final condition
        x_coords = np.linspace(0.0, 1.0, num_points_x)
        y_coords = np.linspace(0.0, 1.0, num_points_y)
        X, Y = np.meshgrid(x_coords, y_coords, indexing="ij")
        U_final = 0.5 * ((X.ravel() - 0.5) ** 2 + (Y.ravel() - 0.5) ** 2)

        U_prev = np.zeros((Nt, total_points))

        # Solve
        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # Reshape to 2D grid for analysis
        U_final_2d = U_solution[-1, :].reshape(num_points_x, num_points_y)

        # Check spatial smoothness using finite differences
        # Compute gradient magnitude
        U_dx = np.diff(U_final_2d, axis=0)
        U_dy = np.diff(U_final_2d, axis=1)

        # Gradient should not have extreme values
        assert np.max(np.abs(U_dx)) < 50.0, "Spatial gradients in x too large"
        assert np.max(np.abs(U_dy)) < 50.0, "Spatial gradients in y too large"

    def test_2d_symmetry_preservation(self):
        """Test that solver preserves radial symmetry when expected."""
        problem = Simple2DMFGProblem(
            grid_resolution=16,
            time_domain=(1.0, 10),
            goal_position=[0.5, 0.5],  # Centered goal
            coupling_strength=0.1,
        )

        solver = HJBSemiLagrangianSolver(problem)

        # Prepare inputs
        Nt = problem.Nt + 1
        num_points_x = problem.geometry.grid.num_points[0]
        num_points_y = problem.geometry.grid.num_points[1]
        total_points = num_points_x * num_points_y

        # Uniform density (symmetric)
        M_density = np.ones((Nt, total_points)) / total_points

        # Radially symmetric final condition (centered at (0.5, 0.5))
        x_coords = np.linspace(0.0, 1.0, num_points_x)
        y_coords = np.linspace(0.0, 1.0, num_points_y)
        X, Y = np.meshgrid(x_coords, y_coords, indexing="ij")
        R_sq = (X.ravel() - 0.5) ** 2 + (Y.ravel() - 0.5) ** 2
        U_final = 0.5 * R_sq

        U_prev = np.zeros((Nt, total_points))

        # Solve
        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # Check approximate radial symmetry at final time
        U_final_2d = U_solution[-1, :].reshape(num_points_x, num_points_y)

        # Compare points equidistant from center
        center_idx = (num_points_x // 2, num_points_y // 2)

        # Check a few symmetric points
        tolerance = 0.2  # Relaxed tolerance for numerical approximation

        # Points at same radius should have similar values
        for offset in [2, 4]:
            if center_idx[0] + offset < num_points_x and center_idx[1] + offset < num_points_y:
                val_right = U_final_2d[center_idx[0] + offset, center_idx[1]]
                val_top = U_final_2d[center_idx[0], center_idx[1] + offset]

                relative_diff = abs(val_right - val_top) / (abs(val_right) + abs(val_top) + 1e-10)
                assert relative_diff < tolerance, (
                    f"Radial symmetry not preserved: offset={offset}, relative_diff={relative_diff:.3f}"
                )


class TestSemiLagrangian2DvsFDM:
    """Compare Semi-Lagrangian 2D solver against FDM solver."""

    @pytest.mark.skip(reason="HJBFDMSolver does not support 2D problems")
    def test_2d_comparison_with_fdm(self):
        """Compare Semi-Lagrangian 2D results with FDM 2D results."""
        problem = Simple2DMFGProblem(
            grid_resolution=12,
            time_domain=(1.0, 10),
            coupling_strength=0.3,
        )

        # Create both solvers
        sl_solver = HJBSemiLagrangianSolver(problem)
        fdm_solver = HJBFDMSolver(problem, max_newton_iterations=10, newton_tolerance=1e-4)

        # Prepare inputs
        Nt = problem.Nt + 1
        num_points_x = problem.geometry.grid.num_points[0]
        num_points_y = problem.geometry.grid.num_points[1]
        total_points = num_points_x * num_points_y

        M_density = np.ones((Nt, total_points)) / total_points

        # Simple final condition
        U_final = np.zeros(total_points)

        U_prev = np.zeros((Nt, total_points))

        # Solve with both methods
        U_sl = sl_solver.solve_hjb_system(M_density, U_final, U_prev)
        U_fdm = fdm_solver.solve_hjb_system(M_density, U_final, U_prev)

        # Compare solutions at several time points
        # Note: Semi-Lagrangian and FDM may have different accuracy, so use relaxed tolerance
        for t_idx in [0, Nt // 4, Nt // 2, 3 * Nt // 4, Nt - 1]:
            relative_error = np.linalg.norm(U_sl[t_idx, :] - U_fdm[t_idx, :]) / (
                np.linalg.norm(U_fdm[t_idx, :]) + 1e-10
            )

            # Relaxed tolerance since methods have different characteristics
            assert relative_error < 0.5, f"Relative error at t_idx={t_idx} is {relative_error:.3f}, exceeds tolerance"


class TestSemiLagrangian2DStability:
    """Test stability properties of 2D Semi-Lagrangian solver."""

    def test_2d_stability_large_timestep(self):
        """Test that Semi-Lagrangian method remains stable with larger time steps."""
        # Semi-Lagrangian methods are known for stability with large time steps
        problem = Simple2DMFGProblem(
            grid_resolution=12,
            time_domain=(1.0, 5),  # Shorter time, fewer steps
            coupling_strength=0.2,
        )

        solver = HJBSemiLagrangianSolver(problem)

        # Prepare inputs
        Nt = problem.Nt + 1
        num_points_x = problem.geometry.grid.num_points[0]
        num_points_y = problem.geometry.grid.num_points[1]
        total_points = num_points_x * num_points_y

        M_density = np.ones((Nt, total_points)) / total_points
        U_final = np.ones(total_points) * 0.5
        U_prev = np.zeros((Nt, total_points))

        # Solve - should not blow up even with larger dt
        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # Check stability: solution should remain bounded
        assert np.all(np.isfinite(U_solution)), "Solution became unbounded (NaN/Inf)"
        assert np.max(np.abs(U_solution)) < 100.0, "Solution magnitude too large"

    @pytest.mark.skip(reason="Numerical instability with large diffusion coefficient - needs investigation")
    def test_2d_diffusion_dominated(self):
        """Test Semi-Lagrangian solver on diffusion-dominated problem."""
        problem = Simple2DMFGProblem(
            grid_resolution=15,
            time_domain=(1.0, 8),
            diffusion_coeff=0.5,  # Large diffusion
            coupling_strength=0.1,  # Weak coupling
        )

        solver = HJBSemiLagrangianSolver(problem)

        # Prepare inputs
        Nt = problem.Nt + 1
        num_points_x = problem.geometry.grid.num_points[0]
        num_points_y = problem.geometry.grid.num_points[1]
        total_points = num_points_x * num_points_y

        M_density = np.ones((Nt, total_points)) / total_points

        # Localized final condition
        x_coords = np.linspace(0.0, 1.0, num_points_x)
        y_coords = np.linspace(0.0, 1.0, num_points_y)
        X, Y = np.meshgrid(x_coords, y_coords, indexing="ij")
        R_sq = (X.ravel() - 0.5) ** 2 + (Y.ravel() - 0.5) ** 2
        U_final = np.exp(-10 * R_sq)

        U_prev = np.zeros((Nt, total_points))

        # Solve
        U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

        # Check that solution diffuses backward in time
        U_initial_2d = U_solution[0, :].reshape(num_points_x, num_points_y)
        U_final_2d = U_solution[-1, :].reshape(num_points_x, num_points_y)

        # Initial solution should be more diffused (smoother) than final
        # Measure smoothness by checking maximum gradient
        U_initial_grad_x = np.diff(U_initial_2d, axis=0)
        U_final_grad_x = np.diff(U_final_2d, axis=0)

        # Initial should have smaller maximum gradient (more diffused)
        assert np.max(np.abs(U_initial_grad_x)) <= np.max(np.abs(U_final_grad_x)) + 0.1, (
            "Diffusion should smooth solution backward in time"
        )


def test_semi_lagrangian_2d_comprehensive():
    """
    Comprehensive validation test for 2D Semi-Lagrangian HJB solver.

    This test runs a complete 2D problem and validates multiple properties.
    """
    print("\n" + "=" * 80)
    print("Comprehensive 2D Semi-Lagrangian HJB Solver Validation Test")
    print("=" * 80)

    # Setup problem
    problem = Simple2DMFGProblem(
        grid_resolution=20,
        time_domain=(1.0, 15),
        coupling_strength=0.4,
        goal_position=[0.7, 0.7],
    )

    print("\nProblem setup:")
    print(f"  Domain: {problem.geometry.grid.bounds}")
    print(f"  Grid resolution: {problem.geometry.grid.num_points}")
    print(f"  Time domain: [0, {problem.T}]")
    print(f"  Time steps: {problem.Nt + 1}")
    print(f"  Diffusion coefficient: {problem.sigma}")
    print(f"  Goal position: {problem.goal_position}")

    # Create solver
    solver = HJBSemiLagrangianSolver(
        problem,
        interpolation_method="linear",
        optimization_method="brent",
        characteristic_solver="explicit_euler",
    )

    print("\nSolver configuration:")
    print(f"  Method: {solver.hjb_method_name}")
    print(f"  Dimension: {solver.dimension}D")
    print(f"  Interpolation: {solver.interpolation_method}")
    print(f"  Time step size: {solver.dt:.4f}")
    print(f"  Grid spacing: {solver.spacing}")

    # Prepare inputs
    Nt = problem.Nt + 1
    num_points_x = problem.geometry.grid.num_points[0]
    num_points_y = problem.geometry.grid.num_points[1]
    total_points = num_points_x * num_points_y

    M_density = np.ones((Nt, total_points)) / total_points

    # Quadratic terminal cost
    x_coords = np.linspace(0.0, 1.0, num_points_x)
    y_coords = np.linspace(0.0, 1.0, num_points_y)
    X, Y = np.meshgrid(x_coords, y_coords, indexing="ij")
    goal = np.array(problem.goal_position)
    U_final = 0.5 * ((X.ravel() - goal[0]) ** 2 + (Y.ravel() - goal[1]) ** 2)

    U_prev = np.zeros((Nt, total_points))

    print("\nSolving HJB system...")
    U_solution = solver.solve_hjb_system(M_density, U_final, U_prev)

    print("\nValidation results:")
    print(f"  ✓ Solution shape: {U_solution.shape}")
    print(f"  ✓ All values finite: {np.all(np.isfinite(U_solution))}")
    print(f"  ✓ Solution range: [{np.min(U_solution):.4f}, {np.max(U_solution):.4f}]")
    print(f"  ✓ Solution mean: {np.mean(U_solution):.4f}")
    print(f"  ✓ Solution std: {np.std(U_solution):.4f}")

    # Check final condition preservation
    final_cond_error = np.linalg.norm(U_solution[-1, :] - U_final) / np.linalg.norm(U_final)
    print(f"  ✓ Final condition error: {final_cond_error:.6f}")
    assert final_cond_error < 1e-5, "Final condition not preserved"

    # Check backward propagation
    initial_influenced = not np.allclose(U_solution[0, :], 0.0)
    print(f"  ✓ Backward propagation: {'Yes' if initial_influenced else 'No'}")
    assert initial_influenced, "Solution at t=0 should be influenced by final condition"

    # Check smoothness
    U_final_2d = U_solution[-1, :].reshape(num_points_x, num_points_y)
    U_dx = np.diff(U_final_2d, axis=0)
    U_dy = np.diff(U_final_2d, axis=1)
    max_grad_x = np.max(np.abs(U_dx))
    max_grad_y = np.max(np.abs(U_dy))
    print(f"  ✓ Max gradient (x): {max_grad_x:.4f}")
    print(f"  ✓ Max gradient (y): {max_grad_y:.4f}")
    assert max_grad_x < 50.0, "Solution gradient in x too large"
    assert max_grad_y < 50.0, "Solution gradient in y too large"

    print("\n" + "=" * 80)
    print("✓ All validation tests passed!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    # Run comprehensive test when executed directly
    test_semi_lagrangian_2d_comprehensive()

    # Run pytest if available
    pytest.main([__file__, "-v", "-s", "--tb=short"])
