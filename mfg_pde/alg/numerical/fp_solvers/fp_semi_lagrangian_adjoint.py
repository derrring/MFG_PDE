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

from mfg_pde.alg.numerical.hjb_solvers.hjb_sl_characteristics import (
    apply_boundary_conditions_1d,
)
from mfg_pde.geometry.boundary.bc_utils import (
    bc_type_to_geometric_operation,
    get_bc_type_string,
)
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
            boundary_conditions: Optional boundary conditions override.
                If None, uses boundary conditions from problem.geometry.
                The advection step uses reflecting BC for mass conservation.
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

    def _get_bc_operation_type(self) -> str:
        """
        Get boundary operation type from boundary conditions.

        Issue #702: Uses centralized bc_utils for consistent BC handling.

        Returns:
            Geometric operation: 'reflect', 'clamp', or 'periodic'
        """
        bc_type = get_bc_type_string(self.boundary_conditions)
        return bc_type_to_geometric_operation(bc_type)

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

        Mass Conservation (Issue #708):
            The SL adjoint naturally conserves sum(m), not the trapezoidal integral.
            This follows from the mathematical structure:
            - HJB interpolation matrix P has row sums = 1
            - FP splatting matrix P^T has column sums = 1
            - Therefore sum(P^T @ m) = sum(m) is exact

            Combined with Crank-Nicolson diffusion (which also preserves sum(m)
            with Neumann BC), total mass sum(m) is conserved by construction.

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

        # Issue #702: Apply boundary conditions based on problem BC type
        # This ensures adjoint consistency with HJB-SL solver
        # Uses shared BC operations from hjb_sl_characteristics module
        bc_op = self._get_bc_operation_type()
        apply_bc = np.vectorize(lambda x: apply_boundary_conditions_1d(x, self.xmin, self.xmax, bc_op))
        x_dest_bounded = apply_bc(x_dest)

        # Convert to grid indices (continuous)
        pos_cont = (x_dest_bounded - self.xmin) / self.dx

        # Lower neighbor index
        j = np.floor(pos_cont).astype(int)

        # Clamp j to valid range [0, Nx-2] for interior splatting
        # (j and j+1 must both be valid indices)
        j = np.clip(j, 0, self.Nx - 2)

        # Weight for upper neighbor (linear interpolation weights)
        w = pos_cont - j
        w = np.clip(w, 0, 1)  # Safety clamp

        # Scatter density to destination cells (transpose of linear interpolation)
        # Issue #708: Use sum(m) as mass measure (consistent with SL theory)
        #
        # The SL adjoint naturally conserves sum(m), not trapezoidal integral.
        # From adjoint_discretization_mfg.md Section 6.4:
        # - Interpolation matrix P has row sums = 1
        # - Splatting matrix P^T has column sums = 1
        # - Therefore sum(P^T @ m) = sum(m) is conserved exactly
        #
        # This is the key difference from Backward SL:
        # - Backward (HJB): m_new[i] = (1-w)*m_old[j] + w*m_old[j+1]  (gather/interpolate)
        # - Forward (FP):   m_new[j] += (1-w)*m_old[i]; m_new[j+1] += w*m_old[i]  (scatter)

        m_star = np.zeros_like(m)

        # Use np.add.at for atomic accumulation (handles overlapping destinations)
        np.add.at(m_star, j, m * (1 - w))  # Scatter to lower neighbor
        np.add.at(m_star, j + 1, m * w)  # Scatter to upper neighbor

        # Ensure non-negativity
        m_star = np.maximum(m_star, 0)

        # Step 2: Diffusion via Crank-Nicolson (Finite Volume formulation)
        # ================================================================
        # Issue #708: Use zero-flux (finite volume) boundary stencil for mass conservation.
        #
        # The standard ghost-point method (m[-1] = m[1]) is Strong Form and breaks
        # mass conservation. The correct approach is Flux Form (finite volume):
        #
        # Physical interpretation:
        #   - Boundary flux J_{-1/2} = 0 (zero-flux BC)
        #   - Interior flux J_{i+1/2} = -D * (m[i+1] - m[i]) / dx
        #   - dm[i]/dt = -(J_{i+1/2} - J_{i-1/2}) / dx
        #
        # This gives boundary stencil: L[0] = (m[1] - m[0])/dx^2
        # The resulting matrix has column sums = 0, ensuring mass conservation.
        #
        D = sigma**2 / 2
        r = D * dt / (self.dx**2)

        # Build RHS: (I + r/2 * L_fv) * m_star
        # L_fv uses zero-flux (one-sided) boundary stencil
        rhs = np.zeros(self.Nx)
        # Interior points: standard 3-point stencil
        rhs[1:-1] = m_star[1:-1] + (r / 2) * (m_star[:-2] - 2 * m_star[1:-1] + m_star[2:])
        # Boundary points: zero-flux (finite volume) stencil
        # L[0] = (m[1] - m[0])/dx^2, so (I + r/2*L)[0] = m[0] + r/2*(m[1] - m[0])
        rhs[0] = m_star[0] + (r / 2) * (m_star[1] - m_star[0])
        rhs[-1] = m_star[-1] + (r / 2) * (m_star[-2] - m_star[-1])

        # Build tridiagonal matrix (I - r/2 * L_fv) for Crank-Nicolson
        # Interior: (I - r/2*L) has [r/2, 1+r, r/2] pattern
        # Boundary: zero-flux gives [-1, 1]/dx^2, so (I - r/2*L) has [1+r/2, -r/2]
        ab = np.zeros((3, self.Nx))
        # Main diagonal
        ab[1, :] = 1 + r  # Interior: 1 + r
        ab[1, 0] = 1 + r / 2  # Left boundary: 1 + r/2
        ab[1, -1] = 1 + r / 2  # Right boundary: 1 + r/2
        # Upper diagonal (superdiagonal)
        ab[0, 1:] = -r / 2  # Interior
        ab[0, 1] = -r / 2  # Left boundary (same coefficient)
        # Lower diagonal (subdiagonal)
        ab[2, :-1] = -r / 2  # Interior
        ab[2, -2] = -r / 2  # Right boundary (same coefficient)

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
    from mfg_pde.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
    from mfg_pde.core.mfg_problem import MFGComponents
    from mfg_pde.geometry import TensorProductGrid
    from mfg_pde.geometry.boundary import BCSegment, BCType, BoundaryConditions

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
    bc = BoundaryConditions(segments=[left_bc, right_bc])

    domain = TensorProductGrid(bounds=[(X_MIN, X_MAX)], Nx_points=[N + 1], boundary_conditions=bc)

    # Create Hamiltonian and components
    H = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: 0.0,
        coupling_dm=lambda m: 0.0,
    )
    components = MFGComponents(
        hamiltonian=H,
        u_final=lambda x: 0.0,
        m_initial=lambda x: 1.0,
    )

    problem = MFGProblem(
        geometry=domain,
        T=T,
        Nt=Nt,
        diffusion=SIGMA,
        components=components,
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

    # Test 3: Mass conservation (Issue #708)
    print("\n3. Testing mass conservation (Issue #708 fix)...")

    # sum(m) is the conserved quantity for SL adjoint
    sum_m_initial = np.sum(M[0])
    sum_m_final = np.sum(M[-1])
    sum_m_error = abs(sum_m_final - sum_m_initial) / sum_m_initial

    print(f"   sum(m) initial: {sum_m_initial:.6f}")
    print(f"   sum(m) final:   {sum_m_final:.6f}")
    print(f"   sum(m) error:   {sum_m_error:.2e}")

    assert sum_m_error < 1e-10, f"Mass conservation failed: error={sum_m_error:.2e}"
    print("   Mass conservation: OK (error < 1e-10)")

    # Test 4: Compare with Backward SL
    print("\n4. Comparing with Backward SL...")

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
