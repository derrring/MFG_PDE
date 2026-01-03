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
        upwind_scheme: str = "none",
        upwind_strength: float = 0.5,
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
            upwind_scheme: Upwind stabilization scheme ("none", "exponential", "linear")
            upwind_strength: Upwind bias parameter β (typically 0.3-1.0)
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

        # Store upwind parameters
        self.upwind_scheme = upwind_scheme
        self.upwind_strength = upwind_strength

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

    def _compute_upwind_divergence(
        self,
        drift_field: np.ndarray,
        density: np.ndarray,
    ) -> np.ndarray:
        """
        Compute div(m * α) with streamline-biased weighted finite differences.

        ACTUAL Implementation (径向有限差分 + 流线加权):
        =========================================================
        This is NOT full GFDM Taylor expansion - it's a simplified approach:

        1. Use radial finite differences: div F ≈ (F_j - F_i) · r_ij / ||r_ij||²
        2. Weight each neighbor by:
           - Distance: w_dist = 1 / ||r_ij||  (inverse distance)
           - Streamline bias: w_upwind = exp(β cos θ)  where cos θ = (α · r) / (||α|| · ||r||)
        3. Compute: div F = Σ_j [w_dist * w_upwind * (F_j - F_i) · r_ij / ||r_ij||²] / Σ_j w_total

        Why NOT Full GFDM Taylor Expansion:
        ------------------------------------
        Full GFDM would require:
        1. Build weighted LS matrix: A_ij = [1, r_x, r_y, r_x², r_xy, r_y², ...]
        2. Solve: (A^T W A) c = A^T W b  → get derivative coefficients
        3. Use coefficients to compute ∇·F

        Problem: Modifying weights requires rebuilding operator for EACH point
        at EACH time step → computationally expensive.

        Current Approach (Simplified):
        -------------------------------
        - Uses SPH-style radial gradient: (F_j - F_i) · r_ij / ||r_ij||²
        - Faster than full GFDM rebuild
        - Still maintains key property: streamline-biased weighting
        - Accuracy: lower than full GFDM Taylor (O(h) vs O(h²)), but stable

        Weight Formula:
        ---------------
        w_combined = w_distance * w_upwind

        where:
        - w_distance = 1 / ||r_ij||  (closer neighbors → higher weight)
        - w_upwind = exp(β cos θ)  (downstream → exp(+β·cos), upstream → exp(-β·|cos|))
        - cos θ = (α · r) / (||α|| · ||r||)  (streamline alignment)
        - β = upwind_strength  (typically 0.5-1.0)

        Key Properties:
        ---------------
        ✓ No dimension splitting (single unified computation)
        ✓ Rotation invariant (no grid orientation effect)
        ✓ All neighbors kept (no hard selection → stable)
        ✓ Streamline-aware (not axis-aware)

        ✗ Lower order accuracy than full GFDM (O(h) radial FD vs O(h²) Taylor)
        ✗ Not using precomputed GFDM operator (rebuilds each call)

        Args:
            drift_field: Drift α = -∇U at each point, shape (N, d)
            density: Density m at each point, shape (N,)

        Returns:
            divergence: div(m*α) with streamline-biased weighting, shape (N,)

        References:
            - SPH radial gradient: Monaghan (2005), Rep. Prog. Phys.
            - Streamline upwinding: Oñate et al. (1996), FPM
        """
        N = len(self.collocation_points)
        divergence = np.zeros(N)

        # Compute flux: F = m * α
        flux = density[:, np.newaxis] * drift_field  # Shape: (N, d)

        # For each point, compute div(F) with streamline-biased weights
        for i in range(N):
            neighborhood = self.gfdm_operator.neighborhoods[i]
            neighbors = neighborhood["indices"]

            # Filter ghost particles
            real_neighbors = neighbors[neighbors >= 0]
            if len(real_neighbors) == 0:
                divergence[i] = 0.0
                continue

            drift_i = drift_field[i]
            drift_norm = np.linalg.norm(drift_i)

            # Compute streamline-biased weights for neighbors
            upwind_weights = []
            neighbor_points = []

            for j in real_neighbors:
                r_ij = self.collocation_points[j] - self.collocation_points[i]
                r_norm = np.linalg.norm(r_ij)

                if r_norm < 1e-12:
                    continue  # Skip coincident points

                # Streamline alignment: cos θ = (α · r) / (||α|| · ||r||)
                cos_theta = np.dot(drift_i, r_ij) / (drift_norm * r_norm)

                # Upwind weight modification
                if self.upwind_scheme == "exponential":
                    # Scharfetter-Gummel style: exp(β cos θ)
                    # Upstream (cos<0): weight < 1
                    # Downstream (cos>0): weight > 1
                    weight_factor = np.exp(self.upwind_strength * cos_theta)
                elif self.upwind_scheme == "linear":
                    # Linear bias: 1 + β*cos θ
                    weight_factor = 1.0 + self.upwind_strength * cos_theta
                else:
                    weight_factor = 1.0

                upwind_weights.append(weight_factor)
                neighbor_points.append(j)

            if len(neighbor_points) == 0:
                divergence[i] = 0.0
                continue

            upwind_weights = np.array(upwind_weights)
            neighbor_points = np.array(neighbor_points)

            # Build local GFDM operator with modified weights
            # We need to manually compute divergence using upwind-weighted Taylor expansion

            # Get neighbor positions and flux values
            X_neighbors = self.collocation_points[neighbor_points]  # Shape: (K, d)
            F_neighbors = flux[neighbor_points]  # Shape: (K, d)
            F_i = flux[i]  # Shape: (d,)

            # Relative positions
            dX = X_neighbors - self.collocation_points[i]  # Shape: (K, d)

            # Build weighted least-squares system for divergence
            # We want: div(F) ≈ Σ_k w_k * ∇·F|_k
            # Using finite differences: ∇·F ≈ (F_k - F_i) · r_k / ||r_k||²

            div_i = 0.0
            total_weight = 0.0

            for idx, _j in enumerate(neighbor_points):
                r_ij = dX[idx]
                r_norm_sq = np.dot(r_ij, r_ij)

                if r_norm_sq < 1e-12:
                    continue

                # Flux difference
                dF = F_neighbors[idx] - F_i

                # Divergence approximation: (dF · r) / ||r||²
                div_contrib = np.dot(dF, r_ij) / r_norm_sq

                # Weight by distance (GFDM-style) AND upwind bias
                dist_weight = 1.0 / (np.sqrt(r_norm_sq) + 1e-12)
                combined_weight = dist_weight * upwind_weights[idx]

                div_i += combined_weight * div_contrib
                total_weight += combined_weight

            if total_weight > 1e-12:
                divergence[i] = div_i / total_weight
            else:
                divergence[i] = 0.0

        return divergence

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
            if self.upwind_scheme != "none":
                # Use upwind-stabilized divergence
                advection = self._compute_upwind_divergence(drift, m_current)
            else:
                # Use standard GFDM divergence (central differences)
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
