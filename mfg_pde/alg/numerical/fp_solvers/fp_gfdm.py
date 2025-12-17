"""
GFDM-based Fokker-Planck Solver for Meshfree Density Evolution.

This module provides a Fokker-Planck solver using Generalized Finite Difference
Method (GFDM) for spatial derivatives on scattered collocation points.

The solver is suitable for:
- Unstructured/scattered point distributions
- Complex domain geometries
- Meshfree discretizations

Mathematical Formulation:
    Continuity equation: dm/dt + div(m * alpha) = sigma^2/2 * Laplacian(m)

    where:
    - m(t,x): density at collocation points
    - alpha(t,x) = -grad U: drift from value function
    - sigma: diffusion coefficient

Author: MFG_PDE Development Team
Created: 2025-12-12
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mfg_pde.alg.numerical.fp_solvers.base_fp import BaseFPSolver
from mfg_pde.utils.numerical.gfdm_operators import GFDMOperator

if TYPE_CHECKING:
    from collections.abc import Callable

    from mfg_pde.core.mfg_problem import MFGProblem


class FPGFDMSolver(BaseFPSolver):
    """
    Fokker-Planck solver using GFDM on collocation points.

    Solves the continuity equation:
        dm/dt + div(m * alpha) = sigma^2/2 * Laplacian(m)

    using Generalized Finite Difference Method for spatial derivatives
    and forward Euler for time stepping.

    Attributes:
        collocation_points: Scattered points for density evolution, shape (N, d)
        gfdm_operator: Precomputed GFDM operator for spatial derivatives
        delta: Neighborhood radius for GFDM

    Example:
        >>> from mfg_pde import MFGProblem
        >>> from mfg_pde.alg.numerical.fp_solvers import FPGFDMSolver
        >>>
        >>> problem = MFGProblem(Nx=30, Nt=20, T=1.0, diffusion=0.1)
        >>> points = np.random.rand(100, 1)  # Scattered 1D points
        >>> solver = FPGFDMSolver(problem, collocation_points=points)
        >>>
        >>> m_init = np.exp(-10 * (points[:, 0] - 0.5)**2)
        >>> U_drift = np.zeros((21, 100))  # Zero drift
        >>> M = solver.solve_fp_system(m_init, drift_field=U_drift)
    """

    def __init__(
        self,
        problem: MFGProblem,
        collocation_points: np.ndarray,
        delta: float | None = None,
        taylor_order: int = 2,
        weight_function: str = "wendland",
        boundary_indices: set[int] | np.ndarray | None = None,
        domain_bounds: list[tuple[float, float]] | None = None,
        boundary_type: str | None = None,
    ):
        """
        Initialize GFDM-based FP solver.

        Args:
            problem: MFG problem definition
            collocation_points: Scattered points for density evolution, shape (N, d)
            delta: Neighborhood radius for GFDM. If None, computed adaptively
                   as 2x median nearest neighbor distance.
            taylor_order: Order of Taylor expansion (1 or 2)
            weight_function: Weight function type ("wendland", "gaussian", "uniform")
            boundary_indices: Set/array of indices of boundary points (optional)
            domain_bounds: List of (min, max) tuples for each dimension (optional)
            boundary_type: Type of boundary condition ("no_flux" or None)
        """
        super().__init__(problem)
        self.fp_method_name = "GFDM"

        # Store collocation points
        self.collocation_points = np.asarray(collocation_points)
        if self.collocation_points.ndim == 1:
            self.collocation_points = self.collocation_points.reshape(-1, 1)

        self.n_points = self.collocation_points.shape[0]
        self.dimension = self.collocation_points.shape[1]

        # Compute adaptive delta if not provided
        if delta is None:
            delta = self._compute_adaptive_delta()
        self.delta = delta

        # Create GFDM operator with optional boundary condition support
        # GFDMOperator handles ghost particles for no-flux BC
        self.gfdm_operator = GFDMOperator(
            self.collocation_points,
            delta=self.delta,
            taylor_order=taylor_order,
            weight_function=weight_function,
            boundary_indices=boundary_indices,
            domain_bounds=domain_bounds,
            boundary_type=boundary_type,
        )

    def _compute_adaptive_delta(self) -> float:
        """Compute adaptive delta based on point spacing."""
        if self.n_points <= 1:
            return 0.1

        from scipy.spatial import cKDTree

        tree = cKDTree(self.collocation_points)
        # Query 2 nearest neighbors (first is self)
        distances, _ = tree.query(self.collocation_points, k=2)
        # Use 2x median nearest neighbor distance
        return 2.0 * float(np.median(distances[:, 1]))

    def solve_fp_system(
        self,
        m_initial_condition: np.ndarray,
        drift_field: np.ndarray | Callable | None = None,
        diffusion_field: float | np.ndarray | Callable | None = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Solve FP system on collocation points using GFDM.

        Solves: dm/dt + div(m * alpha) = D * Laplacian(m)

        where alpha = -grad(U) is computed from drift_field.

        Args:
            m_initial_condition: Initial density at collocation points, shape (N,)
            drift_field: Value function U for drift computation, shape (Nt+1, N).
                        Drift is computed as alpha = -grad(U).
                        If None, zero drift (pure diffusion).
            diffusion_field: Diffusion coefficient. If None, uses problem.sigma.
                            Currently only scalar diffusion supported.
            show_progress: Display progress bar (not yet implemented)

        Returns:
            Density evolution M(t,x) at collocation points, shape (Nt+1, N)
        """
        # Time discretization
        n_time_points = self.problem.Nt + 1
        dt = self.problem.T / self.problem.Nt

        # Diffusion coefficient
        if diffusion_field is None:
            sigma = self.problem.sigma
        elif isinstance(diffusion_field, (int, float)):
            sigma = float(diffusion_field)
        else:
            raise NotImplementedError("Only scalar diffusion currently supported")

        diffusion_coeff = 0.5 * sigma**2

        # Validate inputs
        m_init = np.asarray(m_initial_condition).ravel()
        if m_init.shape[0] != self.n_points:
            raise ValueError(f"m_initial_condition length {m_init.shape[0]} must match n_points {self.n_points}")

        # Handle drift field
        if drift_field is None:
            U_solution = np.zeros((n_time_points, self.n_points))
        else:
            U_solution = np.asarray(drift_field)
            if U_solution.shape != (n_time_points, self.n_points):
                raise ValueError(f"drift_field shape {U_solution.shape} must be ({n_time_points}, {self.n_points})")

        # Storage for density evolution
        M_solution = np.zeros((n_time_points, self.n_points))
        M_solution[0, :] = m_init.copy()

        # Time stepping loop (forward Euler)
        for t_idx in range(n_time_points - 1):
            m_current = M_solution[t_idx, :]
            U_current = U_solution[t_idx, :]

            # Compute drift: alpha = -grad(U)
            grad_U = self.gfdm_operator.gradient(U_current)
            drift = -grad_U  # Shape: (N, d)

            # Advection term: div(m * alpha)
            advection = self.gfdm_operator.divergence(drift, m_current)

            # Diffusion term: D * Laplacian(m)
            laplacian = self.gfdm_operator.laplacian(m_current)
            diffusion = diffusion_coeff * laplacian

            # Forward Euler update: dm/dt = -div(m*alpha) + D*Laplacian(m)
            dm_dt = -advection + diffusion
            M_solution[t_idx + 1, :] = m_current + dt * dm_dt

            # Physical constraints
            M_solution[t_idx + 1, :] = np.maximum(M_solution[t_idx + 1, :], 0.0)

            # Mass conservation (renormalize)
            mass_current = np.sum(M_solution[t_idx + 1, :])
            if mass_current > 0:
                mass_initial = np.sum(m_init)
                M_solution[t_idx + 1, :] *= mass_initial / mass_current

        return M_solution


# =============================================================================
# Smoke Tests
# =============================================================================

if __name__ == "__main__":
    """Smoke test for FPGFDMSolver."""
    print("Testing FPGFDMSolver...")

    from mfg_pde import MFGProblem

    # Test 1D problem
    print("\n[1D] Testing 1D GFDM FP solver...")
    problem = MFGProblem(Nx=30, Nt=20, T=1.0, diffusion=0.1)

    # Create 1D collocation points
    points_1d = np.linspace(0, 1, 50).reshape(-1, 1)
    solver = FPGFDMSolver(problem, collocation_points=points_1d)

    assert solver.fp_method_name == "GFDM"
    assert solver.n_points == 50
    assert solver.dimension == 1
    print(f"     Delta: {solver.delta:.4f}")

    # Initial Gaussian density
    m_init = np.exp(-50 * (points_1d[:, 0] - 0.5) ** 2)
    m_init = m_init / np.sum(m_init)  # Normalize

    # Zero drift (pure diffusion)
    U_drift = np.zeros((problem.Nt + 1, 50))

    M_solution = solver.solve_fp_system(m_init, drift_field=U_drift)

    assert M_solution.shape == (problem.Nt + 1, 50)
    assert not np.any(np.isnan(M_solution))
    assert not np.any(np.isinf(M_solution))
    assert np.all(M_solution >= 0)
    print(f"     M range: [{M_solution.min():.4f}, {M_solution.max():.4f}]")
    print("     1D test passed!")

    # Test 2D problem
    print("\n[2D] Testing 2D GFDM FP solver...")
    from mfg_pde.geometry import TensorProductGrid

    geometry_2d = TensorProductGrid(
        dimension=2,
        bounds=[(0.0, 1.0), (0.0, 1.0)],
        Nx_points=[10, 10],
    )
    problem_2d = MFGProblem(geometry=geometry_2d, Nt=10, T=0.5, diffusion=0.1)

    # Create 2D scattered points
    np.random.seed(42)
    points_2d = np.random.rand(100, 2)
    solver_2d = FPGFDMSolver(problem_2d, collocation_points=points_2d)

    assert solver_2d.n_points == 100
    assert solver_2d.dimension == 2
    print(f"     Delta: {solver_2d.delta:.4f}")

    # Initial Gaussian density
    r2 = (points_2d[:, 0] - 0.5) ** 2 + (points_2d[:, 1] - 0.5) ** 2
    m_init_2d = np.exp(-20 * r2)
    m_init_2d = m_init_2d / np.sum(m_init_2d)

    # Zero drift
    U_drift_2d = np.zeros((problem_2d.Nt + 1, 100))

    M_solution_2d = solver_2d.solve_fp_system(m_init_2d, drift_field=U_drift_2d)

    assert M_solution_2d.shape == (problem_2d.Nt + 1, 100)
    assert not np.any(np.isnan(M_solution_2d))
    assert not np.any(np.isinf(M_solution_2d))
    assert np.all(M_solution_2d >= 0)
    print(f"     M range: [{M_solution_2d.min():.4f}, {M_solution_2d.max():.4f}]")
    print("     2D test passed!")

    print("\nFPGFDMSolver smoke tests passed!")
