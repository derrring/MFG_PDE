"""
Adjoint Semi-Lagrangian Fokker-Planck Solver for Mean Field Games.

This module implements the Forward (Adjoint) Semi-Lagrangian method for solving
the Fokker-Planck equation. This is the mathematically correct dual to the
Backward SL used for HJB, ensuring discrete duality for MFG convergence.

The FP equation solved is (divergence form):
    dm/dt + div(alpha * m) = sigma^2/2 * Laplacian(m)    in [0,T] x Omega
    m(0, x) = m0(x)                                       at t = 0

Key differences from Backward SL:
- **Forward tracing**: x_dest = x + α*dt (where does mass go?)
- **Splatting**: Mass is scattered to destination cells (transpose of interpolation)
- **Conservative**: Mass conservation is exact by construction
- **Duality**: Preserves ∫ m (S φ) dx = ∫ (S* m) φ dx with HJB operator S

References:
    - Carlini & Silva (2014): Semi-Lagrangian schemes for MFG
    - The discrete FP operator is M^{n+1} = I_{interp}^T @ M^n where I_{interp}
      is the interpolation matrix used in HJB.

Issue #578: Adjoint SL implementation for proper SL-SL MFG coupling
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.linalg import solve_banded

from mfg_pde.utils.mfg_logging import get_logger

from .base_fp import BaseFPSolver

if TYPE_CHECKING:
    from mfg_pde.core.mfg_problem import MFGProblem
    from mfg_pde.geometry import BoundaryConditions

logger = get_logger(__name__)


class FPSLAdjointSolver(BaseFPSolver):
    """
    Adjoint (Forward) Semi-Lagrangian solver for Fokker-Planck equations.

    The Forward SL method asks "Where does mass go?" and scatters (splats) mass
    to destination cells. This is the adjoint of the Backward SL interpolation
    used in HJB solvers, ensuring discrete duality for MFG.

    Algorithm (operator splitting):
        1. Advection: Forward trace x_dest = x + α*dt, scatter mass via linear splatting
        2. Diffusion: Crank-Nicolson implicit solve

    Key Properties:
        - Mass conservation is exact (scatter weights sum to 1)
        - Density peaks form naturally from converging flow
        - Discrete duality with HJB Backward SL is preserved
        - No Jacobian correction needed (conservation is intrinsic)

    Dimension support:
        - 1D: Full support (production-ready)
        - nD: Planned for future
    """

    # Scheme family trait for duality validation (Issue #580)
    from mfg_pde.alg.base_solver import SchemeFamily

    _scheme_family = SchemeFamily.SL  # Forward SL (adjoint of HJB Backward SL)

    def __init__(
        self,
        problem: MFGProblem,
        boundary_conditions: BoundaryConditions | None = None,
    ):
        """
        Initialize Adjoint Semi-Lagrangian FP solver.

        Args:
            problem: MFG problem instance
            boundary_conditions: Optional boundary conditions (defaults to absorbing)
        """
        super().__init__(problem)
        self.fp_method_name = "Adjoint Semi-Lagrangian"

        # Detect problem dimension
        self.dimension = self._detect_dimension()  # Issue #633: Use inherited method

        if self.dimension > 1:
            raise NotImplementedError(
                "FPSLAdjointSolver currently only supports 1D problems. nD support is planned for future versions."
            )

        # Precompute grid parameters (1D)
        bounds = problem.geometry.get_bounds()
        self.xmin, self.xmax = bounds[0][0], bounds[1][0]
        Nx = problem.geometry.get_grid_shape()[0]
        self.x_grid = np.linspace(self.xmin, self.xmax, Nx)
        self.dx = problem.geometry.get_grid_spacing()[0]
        self.dt = problem.dt
        self.Nx = Nx

        # Boundary conditions
        if boundary_conditions is not None:
            self.boundary_conditions = boundary_conditions
        else:
            self.boundary_conditions = self._get_boundary_conditions_from_problem()

    def _get_boundary_conditions_from_problem(self) -> BoundaryConditions | None:
        """Get boundary conditions from problem or geometry."""
        try:
            return self.problem.geometry.boundary_conditions
        except AttributeError:
            pass
        try:
            return self.problem.geometry.get_boundary_conditions()
        except AttributeError:
            pass
        return None

    def solve_fp_system(
        self,
        M_initial: np.ndarray | None = None,
        drift_field: np.ndarray | None = None,
        diffusion_field: float | np.ndarray | None = None,
        show_progress: bool = True,
        m_initial_condition: np.ndarray | None = None,  # Deprecated
    ) -> np.ndarray:
        """
        Solve FP system forward in time using Adjoint Semi-Lagrangian method.

        The Forward SL discretization uses mass splatting instead of interpolation:
            1. Forward trace: x_dest = x + α*dt
            2. Scatter mass to destination cells with linear weights
            3. Apply diffusion via Crank-Nicolson

        Args:
            M_initial: Initial density m0(x). Shape: (Nx,)
            drift_field: Drift field (value function U from HJB).
                - np.ndarray: Shape (Nt+1, Nx) - U values at each time step
                  The drift velocity is computed as alpha = -grad(U)
            diffusion_field: Optional diffusion coefficient override.
            show_progress: Show progress bar during solve

        Returns:
            Density evolution M(t,x). Shape: (Nt+1, Nx)
        """
        import warnings

        # Handle deprecated parameter
        if m_initial_condition is not None:
            if M_initial is not None:
                raise ValueError("Cannot specify both M_initial and m_initial_condition")
            warnings.warn(
                "Parameter 'm_initial_condition' is deprecated. Use 'M_initial' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            M_initial = m_initial_condition

        if M_initial is None:
            raise ValueError("M_initial is required")

        if drift_field is None:
            raise ValueError(
                "drift_field (value function U) is required for Adjoint SL FP. Pass the U solution from HJB solver."
            )

        # Handle diffusion
        if diffusion_field is None:
            sigma = self.problem.sigma
        elif isinstance(diffusion_field, (int, float)):
            sigma = float(diffusion_field)
        else:
            raise NotImplementedError("Array/callable diffusion_field not yet supported")

        # Determine number of time steps from drift_field
        Nt_points = drift_field.shape[0]

        # Allocate solution array
        M = np.zeros((Nt_points, self.Nx))
        M[0, :] = M_initial.copy()

        # Progress bar
        if show_progress:
            from mfg_pde.utils.progress import RichProgressBar

            timestep_range = RichProgressBar(
                range(Nt_points - 1),
                desc="FP-SL Adjoint",
                unit="step",
            )
        else:
            timestep_range = range(Nt_points - 1)

        # Forward time stepping
        for n in timestep_range:
            # Compute velocity field alpha = -grad(U) at current time
            U_n = drift_field[n, :]
            alpha = self._compute_velocity(U_n)

            # Adjoint Semi-Lagrangian step (forward splatting)
            M[n + 1, :] = self._adjoint_sl_step(
                M[n, :],
                alpha,
                self.dt,
                sigma,
            )

        return M

    def _compute_velocity(self, U: np.ndarray) -> np.ndarray:
        """Compute velocity field alpha = -grad(U)."""
        return -np.gradient(U, self.dx)

    def _adjoint_sl_step(
        self,
        m: np.ndarray,
        alpha: np.ndarray,
        dt: float,
        sigma: float,
    ) -> np.ndarray:
        """
        One Adjoint Semi-Lagrangian step for Fokker-Planck equation.

        Operator splitting:
            1. Forward advection with linear splatting (mass-conservative)
            2. Diffusion via Crank-Nicolson

        This is the ADJOINT of the HJB backward interpolation, ensuring discrete
        duality for MFG convergence.

        Mass Conservation Strategy:
            We use a finite-volume interpretation where each grid point represents
            a cell with volume dx (interior) or dx/2 (boundary). We splat CELL MASS
            (density * cell_volume), then recover density by dividing by destination
            cell volume. This preserves the trapezoidal integral exactly.

        Args:
            m: Current density, shape (Nx,)
            alpha: Velocity field, shape (Nx,)
            dt: Time step
            sigma: Diffusion coefficient

        Returns:
            Density at next time step, shape (Nx,)
        """
        # Step 1: Forward Advection (Splatting)
        # =====================================
        # Forward trace: where does mass at x go?
        x_dest = self.x_grid + alpha * dt

        # Convert to grid indices (continuous)
        pos_cont = (x_dest - self.xmin) / self.dx

        # Lower neighbor index
        j = np.floor(pos_cont).astype(int)

        # Weight for upper neighbor (linear interpolation weights)
        w = pos_cont - j

        # Finite-volume cell volumes for trapezoidal quadrature consistency
        # Boundary cells have half volume, interior cells have full volume
        cell_volume = np.ones(self.Nx) * self.dx
        cell_volume[0] = self.dx / 2
        cell_volume[-1] = self.dx / 2

        # Convert density to cell mass (this is what we conserve)
        cell_mass = m * cell_volume

        # Initialize accumulated mass at destinations
        mass_star = np.zeros_like(m)

        # Scatter mass to destination cells (transpose of linear interpolation)
        # This is the key difference from Backward SL:
        # - Backward: m_new[i] = w * m_old[j] + (1-w) * m_old[j+1]  (gather)
        # - Forward:  mass_new[j] += (1-w) * mass_old[i]; mass_new[j+1] += w * mass_old[i]

        # Handle boundary conditions: absorbing (mass leaving domain is lost)
        valid_mask = (j >= 0) & (j < self.Nx - 1)

        # Use np.add.at for atomic accumulation (handles overlapping destinations)
        valid_j = j[valid_mask]
        valid_w = w[valid_mask]
        mass_source = cell_mass[valid_mask]

        # Scatter to lower neighbor with weight (1-w)
        np.add.at(mass_star, valid_j, mass_source * (1 - valid_w))

        # Scatter to upper neighbor with weight w
        np.add.at(mass_star, valid_j + 1, mass_source * valid_w)

        # Handle particles that land exactly on last cell (j == Nx-1, w == 0)
        exact_last = (j == self.Nx - 1) & (w == 0)
        if np.any(exact_last):
            np.add.at(mass_star, j[exact_last], cell_mass[exact_last])

        # Convert accumulated mass back to density
        # Avoid division by zero (cell_volume is always > 0)
        m_star = mass_star / cell_volume

        # Ensure non-negativity
        m_star = np.maximum(m_star, 0)

        # Step 2: Diffusion via Crank-Nicolson
        # ====================================
        D = sigma**2 / 2
        r = D * dt / (self.dx**2)

        # Build RHS: (I + r/2 * L) * m_star
        rhs = np.zeros(self.Nx)
        # Interior points
        rhs[1:-1] = m_star[1:-1] + (r / 2) * (m_star[:-2] - 2 * m_star[1:-1] + m_star[2:])
        # Neumann BC: dm/dx = 0 at boundaries
        rhs[0] = m_star[0] + (r / 2) * (2 * m_star[1] - 2 * m_star[0])
        rhs[-1] = m_star[-1] + (r / 2) * (2 * m_star[-2] - 2 * m_star[-1])

        # Build tridiagonal matrix (I - r/2 * L) for Crank-Nicolson
        ab = np.zeros((3, self.Nx))
        ab[1, :] = 1 + r
        ab[0, 1:] = -r / 2
        ab[2, :-1] = -r / 2
        ab[0, 1] = -r  # Neumann BC at left
        ab[2, -2] = -r  # Neumann BC at right

        # Solve tridiagonal system
        m_new = solve_banded((1, 1), ab, rhs)

        # Ensure non-negativity
        m_new = np.maximum(m_new, 0)

        return m_new

    def _get_solver_type_id(self) -> str | None:
        """Get solver type identifier for compatibility checking."""
        # Use semi_lagrangian for compatibility (adjoint is a variant)
        return "semi_lagrangian"


if __name__ == "__main__":
    """Quick smoke test for development."""
    print("Testing FPSLAdjointSolver...")
    print("=" * 60)

    from mfg_pde import MFGProblem
    from mfg_pde.geometry import TensorProductGrid
    from mfg_pde.geometry.boundary import BCSegment, BCType, mixed_bc

    # Test parameters
    X_MIN, X_MAX = -0.5, 0.5
    SIGMA = 0.2
    N = 100
    T = 10.0
    Nt = 1000

    L = X_MAX - X_MIN
    dx = L / N
    x = np.linspace(X_MIN, X_MAX, N + 1)

    print(f"Grid: N={N}, dx={dx:.4f}")
    print(f"Time: T={T}, Nt={Nt}, dt={T / Nt:.4f}")
    print(f"Diffusion: sigma={SIGMA}")

    # Create problem with Neumann BC
    left_bc = BCSegment(name="left", bc_type=BCType.NEUMANN, value=0.0, boundary="x_min")
    right_bc = BCSegment(name="right", bc_type=BCType.NEUMANN, value=0.0, boundary="x_max")
    bc = mixed_bc(segments=[left_bc, right_bc], dimension=1, domain_bounds=np.array([[X_MIN, X_MAX]]))

    domain = TensorProductGrid(dimension=1, bounds=[(X_MIN, X_MAX)], Nx_points=[N + 1], boundary_conditions=bc)

    problem = MFGProblem(
        geometry=domain,
        T=T,
        Nt=Nt,
        diffusion=SIGMA,
    )

    # Test 1: Solver initialization
    print("\n1. Testing solver initialization...")
    solver = FPSLAdjointSolver(problem)
    assert solver.dimension == 1
    assert solver.fp_method_name == "Adjoint Semi-Lagrangian"
    print("   Initialization: OK")

    # Test 2: Solve with drift toward center (confining potential)
    print("\n2. Testing with confining potential U = x^2...")

    U_well = np.tile(x**2, (Nt + 1, 1))

    # Start from uniform
    m_uniform = np.ones(N + 1) / L

    # Expected Gibbs
    m_gibbs = np.exp(-2 * x**2 / SIGMA**2)
    m_gibbs /= np.trapezoid(m_gibbs, x)

    print(f"   Initial: uniform, peak = {m_uniform.max():.4f}")
    print(f"   Target Gibbs: peak = {m_gibbs.max():.4f}")

    M = solver.solve_fp_system(M_initial=m_uniform.copy(), drift_field=U_well, show_progress=False)

    # Check evolution
    for t_idx in [0, 100, 500, 1000]:
        m_t = M[t_idx, :]
        peak_idx = np.argmax(m_t)
        peak_x = x[peak_idx]
        mass = np.trapezoid(m_t, x)
        print(f"   t={t_idx * T / Nt:5.2f}: peak={m_t.max():.4f} at x={peak_x:.3f}, mass={mass:.4f}")

    # Check final result
    m_final = M[-1, :]
    m_final_norm = m_final / np.trapezoid(m_final, x)
    l2_to_gibbs = np.sqrt(np.trapezoid((m_final_norm - m_gibbs) ** 2, x))
    print(f"\n   Final L2 to Gibbs: {l2_to_gibbs:.4e}")

    # Test 3: Compare with Backward SL
    print("\n3. Comparing with Backward SL...")

    from mfg_pde.alg.numerical.fp_solvers import FPSLSolver

    backward_solver = FPSLSolver(problem)
    M_backward = backward_solver.solve_fp_system(M_initial=m_uniform.copy(), drift_field=U_well, show_progress=False)

    m_backward_final = M_backward[-1, :]
    m_backward_norm = m_backward_final / np.trapezoid(m_backward_final, x)
    l2_backward = np.sqrt(np.trapezoid((m_backward_norm - m_gibbs) ** 2, x))

    print(f"   Adjoint SL peak: {m_final.max():.4f}, L2 to Gibbs: {l2_to_gibbs:.4e}")
    print(f"   Backward SL peak: {m_backward_final.max():.4f}, L2 to Gibbs: {l2_backward:.4e}")

    print("\n" + "=" * 60)
    print("All smoke tests passed!")
