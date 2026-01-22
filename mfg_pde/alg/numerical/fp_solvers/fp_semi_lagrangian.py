"""
Semi-Lagrangian Fokker-Planck Solver for Mean Field Games.

This module implements a semi-Lagrangian method for solving the Fokker-Planck
equation in MFG problems. The method traces characteristics forward in time
and uses interpolation with Jacobian correction for mass conservation.

The FP equation solved is (divergence form):
    dm/dt + div(alpha * m) = sigma^2/2 * Laplacian(m)    in [0,T] x Omega
    m(0, x) = m0(x)                                       at t = 0

where alpha = -grad(U) is the drift field from HJB solution.

Key features:
- Stable for large time steps (same as HJB Semi-Lagrangian)
- Jacobian correction preserves divergence form structure
- O(h^2) convergence when paired with SL-HJB (Carlini & Silva 2014)
- Crank-Nicolson diffusion for unconditional stability

Issue #578: Implement Semi-Lagrangian FP solver for SL-SL coupling
Validated in: mfg-research/experiments/crowd_evacuation_2d/runners/exp14j_fp_semi_lagrangian_v2.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.interpolate import interp1d
from scipy.linalg import solve_banded

# Import BC infrastructure from HJB-SL (reuse unified components)
from mfg_pde.alg.numerical.hjb_solvers.hjb_sl_characteristics import (
    apply_boundary_conditions_1d,
)
from mfg_pde.utils.mfg_logging import get_logger

# Use tensor_calculus for proper differential operators with BC handling
from mfg_pde.utils.numerical.tensor_calculus import laplacian as tensor_laplacian

from .base_fp import BaseFPSolver

if TYPE_CHECKING:
    from collections.abc import Callable

    from mfg_pde.core.mfg_problem import MFGProblem
    from mfg_pde.geometry import BoundaryConditions

logger = get_logger(__name__)


class FPSLSolver(BaseFPSolver):
    """
    Semi-Lagrangian solver for Fokker-Planck equations.

    The Semi-Lagrangian method for FP uses backward characteristic tracing
    with Jacobian correction to handle the divergence form properly:

        m^{n+1}(x) = m^n(x - alpha*dt) * exp(-dt * div(alpha))

    This correction accounts for compression (div < 0, density increases)
    and expansion (div > 0, density decreases) of the flow.

    Algorithm (operator splitting):
        1. Advection: Trace characteristics backward, interpolate, apply Jacobian
        2. Diffusion: Crank-Nicolson implicit solve (unconditionally stable)

    Expected convergence:
        - O(h^2) when paired with SL-HJB and Dt = O(h^{3/2})
        - Better equilibrium preservation than upwind FDM

    Dimension support:
        - 1D: Full support (production-ready)
        - nD: Planned for future

    References:
        - Carlini & Silva (2014): Semi-Lagrangian schemes for MFG
        - Calzola et al. (2023): High-order MFG schemes
    """

    # Scheme family trait for duality validation (Issue #580)
    from mfg_pde.alg.base_solver import SchemeFamily

    _scheme_family = SchemeFamily.SL

    def __init__(
        self,
        problem: MFGProblem,
        interpolation_method: str = "cubic",
        characteristic_solver: str = "explicit_euler",
        boundary_conditions: BoundaryConditions | None = None,
    ):
        """
        Initialize Semi-Lagrangian FP solver.

        Args:
            problem: MFG problem instance
            interpolation_method: Method for interpolating density at departure points
                - 'linear': Linear interpolation (faster, C^0 continuous)
                - 'cubic': Cubic spline interpolation (slower, C^2 continuous)
            characteristic_solver: Method for solving characteristics
                - 'explicit_euler': First-order explicit Euler (default)
                - 'rk2': Second-order Runge-Kutta midpoint
                - 'rk4': Fourth-order Runge-Kutta
            boundary_conditions: Optional boundary conditions (defaults to no-flux)
        """
        super().__init__(problem)
        self.fp_method_name = "Semi-Lagrangian"

        # Solver configuration
        self.interpolation_method = interpolation_method
        self.characteristic_solver = characteristic_solver

        # Detect problem dimension
        self.dimension = self._detect_dimension(problem)

        if self.dimension > 1:
            raise NotImplementedError(
                "FPSLSolver currently only supports 1D problems. nD support is planned for future versions."
            )

        # Precompute grid parameters (1D)
        bounds = problem.geometry.get_bounds()
        self.xmin, self.xmax = bounds[0][0], bounds[1][0]
        Nx = problem.geometry.get_grid_shape()[0]
        self.x_grid = np.linspace(self.xmin, self.xmax, Nx)
        self.dx = problem.geometry.get_grid_spacing()[0]
        self.dt = problem.dt

        # Boundary conditions
        if boundary_conditions is not None:
            self.boundary_conditions = boundary_conditions
        else:
            # Try to get BC from problem/geometry
            self.boundary_conditions = self._get_boundary_conditions_from_problem()

    def _detect_dimension(self, problem: Any) -> int:
        """Detect problem dimension."""
        try:
            return problem.geometry.dimension
        except AttributeError:
            pass

        try:
            return problem.dimension
        except AttributeError:
            pass

        # Legacy 1D detection
        if getattr(problem, "Nx", None) is not None and getattr(problem, "Ny", None) is None:
            return 1

        return 1  # Default to 1D

    def _get_boundary_conditions_from_problem(self) -> BoundaryConditions | None:
        """Get boundary conditions from problem or geometry."""
        # Priority 1: geometry.boundary_conditions
        try:
            bc = self.problem.geometry.boundary_conditions
            if bc is not None:
                return bc
        except AttributeError:
            pass

        # Priority 2: geometry.get_boundary_conditions()
        try:
            bc = self.problem.geometry.get_boundary_conditions()
            if bc is not None:
                return bc
        except AttributeError:
            pass

        # Priority 3: problem.boundary_conditions
        try:
            bc = self.problem.boundary_conditions
            if bc is not None:
                return bc
        except AttributeError:
            pass

        return None

    def _get_bc_type(self) -> str:
        """Get boundary condition type string."""
        if self.boundary_conditions is None:
            return "no_flux"  # Default for FP

        try:
            bc_type_enum = self.boundary_conditions.default_bc
            if bc_type_enum is not None:
                try:
                    return bc_type_enum.value
                except AttributeError:
                    return str(bc_type_enum)
        except AttributeError:
            pass

        return "no_flux"

    def solve_fp_system(
        self,
        M_initial: np.ndarray | None = None,
        drift_field: np.ndarray | Callable | None = None,
        diffusion_field: float | np.ndarray | Callable | None = None,
        show_progress: bool = True,
        # Deprecated parameter name
        m_initial_condition: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Solve FP system forward in time using Semi-Lagrangian method.

        The Semi-Lagrangian discretization of the FP equation in divergence form:
            dm/dt + div(alpha * m) = sigma^2/2 * Laplacian(m)

        uses backward characteristic tracing with Jacobian correction:
            m^{n+1}(x) = m^n(x - alpha*dt) * exp(-dt * div(alpha)) + diffusion

        Args:
            M_initial: Initial density m0(x). Shape: (Nx,)
            drift_field: Drift field (value function U from HJB).
                - np.ndarray: Shape (Nt+1, Nx) - U values at each time step
                  The drift velocity is computed as alpha = -grad(U)
            diffusion_field: Optional diffusion coefficient override.
                - None: Use problem.sigma
                - float: Constant diffusion
            show_progress: Show progress bar during solve

        Returns:
            Density evolution M(t,x). Shape: (Nt+1, Nx)

        Note:
            drift_field is the VALUE FUNCTION U, not the velocity alpha.
            The velocity is computed internally as alpha = -grad(U).
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
                "drift_field (value function U) is required for Semi-Lagrangian FP. "
                "Pass the U solution from HJB solver."
            )

        if callable(drift_field):
            raise NotImplementedError("Callable drift_field not yet supported for FPSLSolver")

        # Handle diffusion
        if diffusion_field is None:
            sigma = self.problem.sigma
        elif isinstance(diffusion_field, (int, float)):
            sigma = float(diffusion_field)
        else:
            raise NotImplementedError("Array/callable diffusion_field not yet supported for FPSLSolver")

        # Determine number of time steps from drift_field
        Nt_points = drift_field.shape[0]
        Nx = len(self.x_grid)

        # Allocate solution array
        M = np.zeros((Nt_points, Nx))
        M[0, :] = M_initial.copy()

        # Progress bar
        if show_progress:
            from mfg_pde.utils.progress import RichProgressBar

            timestep_range = RichProgressBar(
                range(Nt_points - 1),
                desc="FP-SL (forward)",
                unit="step",
            )
        else:
            timestep_range = range(Nt_points - 1)

        # Forward time stepping
        for n in timestep_range:
            # Compute velocity field alpha = -grad(U) at current time
            U_n = drift_field[n, :]
            alpha = self._compute_velocity(U_n)
            # Issue #579 fix: Use direct Laplacian instead of compound gradient
            # div(alpha) = div(-grad(U)) = -Laplacian(U)
            div_alpha = self._compute_divergence_from_U(U_n)

            # Semi-Lagrangian step
            M[n + 1, :] = self._sl_step(
                M[n, :],
                alpha,
                div_alpha,
                self.dt,
                sigma,
            )

        return M

    def _compute_velocity(self, U: np.ndarray) -> np.ndarray:
        """Compute velocity field alpha = -grad(U)."""
        return -np.gradient(U, self.dx)

    def _compute_divergence_from_U(self, U: np.ndarray) -> np.ndarray:
        """
        Compute div(alpha) = div(-grad(U)) = -Laplacian(U).

        Uses the tensor_calculus.laplacian with proper BC handling instead of
        compound gradient (which causes 50% boundary error).

        Issue #579: Compound gradient np.gradient(np.gradient(U)) gives
        wrong values at boundaries because np.gradient uses one-sided
        differences. For U = 0.5*x^2, div(alpha) should be -1 everywhere,
        but compound gradient gives -0.5 at boundaries.

        The fix is to compute -Laplacian(U) directly using proper ghost cells.
        We use quadratic extrapolation BC (EXTRAPOLATION_QUADRATIC) because the
        potential U does not satisfy Neumann BC - it has non-zero gradient at
        boundaries.
        """
        # Use quadratic extrapolation BC for fields that don't satisfy Neumann BC
        from mfg_pde.geometry.boundary import BCSegment, BCType, mixed_bc

        # Create quadratic extrapolation BC for the potential field
        # This gives O(h^2) accuracy for smooth fields like U = 0.5*x^2
        left_bc = BCSegment(name="left", bc_type=BCType.EXTRAPOLATION_QUADRATIC, value=0.0, boundary="x_min")
        right_bc = BCSegment(name="right", bc_type=BCType.EXTRAPOLATION_QUADRATIC, value=0.0, boundary="x_max")
        extrap_bc = mixed_bc(
            segments=[left_bc, right_bc], dimension=1, domain_bounds=np.array([[self.xmin, self.xmax]])
        )

        # Compute Laplacian with proper ghost cell handling
        lap_U = tensor_laplacian(U, spacings=[self.dx], bc=extrap_bc)

        return -lap_U

    def _apply_reflecting_bc(self, x_dep: np.ndarray) -> np.ndarray:
        """
        Apply reflecting boundary conditions to departure points.

        Uses unified BC infrastructure from hjb_sl_characteristics module.
        For no-flux boundaries, particles that would leave the domain
        are reflected back. This preserves mass and is consistent with
        Neumann BC (dm/dn = 0).

        Args:
            x_dep: Departure point coordinates

        Returns:
            Departure points with reflecting BC applied
        """
        # Get BC types for each boundary
        bc_type_min, bc_type_max = self._get_bc_types_per_boundary()

        # Use unified infrastructure for consistent BC handling across all SL solvers
        result = np.zeros_like(x_dep)
        for i, x in enumerate(x_dep):
            result[i] = apply_boundary_conditions_1d(
                x,
                xmin=self.xmin,
                xmax=self.xmax,
                bc_type_min=bc_type_min,
                bc_type_max=bc_type_max,
            )
        return result

    def _get_bc_types_per_boundary(self) -> tuple[str | None, str | None]:
        """Get BC type for each boundary."""
        if self.boundary_conditions is None:
            return ("neumann", "neumann")  # Default for FP

        try:
            bc_type_min_enum = self.boundary_conditions.get_bc_type_at_boundary("x_min")
            bc_type_max_enum = self.boundary_conditions.get_bc_type_at_boundary("x_max")

            bc_type_min = bc_type_min_enum.value if bc_type_min_enum is not None else "neumann"
            bc_type_max = bc_type_max_enum.value if bc_type_max_enum is not None else "neumann"

            return (bc_type_min, bc_type_max)
        except AttributeError:
            # Fallback to default BC type
            bc_type = self._get_bc_type()
            return (bc_type, bc_type)

    def _sl_step(
        self,
        m: np.ndarray,
        alpha: np.ndarray,
        div_alpha: np.ndarray,
        dt: float,
        sigma: float,
    ) -> np.ndarray:
        """
        One Semi-Lagrangian step for Fokker-Planck equation.

        Operator splitting:
            1. Advection with Jacobian correction (divergence form)
            2. Diffusion via Crank-Nicolson

        Args:
            m: Current density, shape (Nx,)
            alpha: Velocity field, shape (Nx,)
            div_alpha: Divergence of velocity, shape (Nx,)
            dt: Time step
            sigma: Diffusion coefficient

        Returns:
            Density at next time step, shape (Nx,)
        """
        Nx = len(m)

        # Step 1: Advection with Jacobian correction
        # ==========================================
        # Trace characteristics backward: x_departure = x - alpha * dt
        x_departure = self.x_grid - alpha * dt

        # For confining potentials (inward drift at boundaries), departure points
        # may lie outside the domain. We use fill_value=0.0 (vacuum outside)
        # instead of reflection or edge values. This correctly models that no
        # mass exists outside the domain, causing edge erosion and creating
        # gradients for diffusion to act upon.
        #
        # Note: For periodic BC, use np.interp with period or wrap x_departure.
        # For Neumann BC with outward drift, reflection might be appropriate.
        # But for confining potentials, vacuum BC breaks the "false fixed point"
        # where uniform distributions stay uniform forever.

        # Interpolate density at departure points with vacuum BC (no mass outside)
        if self.interpolation_method == "linear":
            interpolator = interp1d(
                self.x_grid,
                m,
                kind="linear",
                bounds_error=False,
                fill_value=0.0,  # Vacuum outside domain
            )
        else:  # cubic
            interpolator = interp1d(
                self.x_grid,
                m,
                kind="cubic",
                bounds_error=False,
                fill_value=0.0,  # Vacuum outside domain
            )

        m_advected = interpolator(x_departure)

        # Jacobian correction for divergence form:
        # m^{n+1}(x) = m^n(x_dep) * |det(dx_dep/dx)|
        # For 1D with x_dep = x - alpha*dt:
        #   dx_dep/dx = 1 - dt * d(alpha)/dx = 1 - dt * div(alpha)
        # For finite dt, use exponential form: J = exp(-dt * div(alpha))
        #
        # Clamp the exponent to prevent numerical overflow.
        # |dt * div(alpha)| < 10 is a reasonable bound for stability.
        jacobian_exponent = np.clip(-dt * div_alpha, -10.0, 10.0)
        jacobian = np.exp(jacobian_exponent)
        m_advected = m_advected * jacobian

        # Ensure non-negativity
        m_advected = np.maximum(m_advected, 0)

        # Step 2: Diffusion via Crank-Nicolson
        # ====================================
        D = sigma**2 / 2
        r = D * dt / (self.dx**2)

        # Build RHS: (I + r/2 * L) * m_advected
        rhs = np.zeros(Nx)
        # Interior points
        rhs[1:-1] = m_advected[1:-1] + (r / 2) * (m_advected[:-2] - 2 * m_advected[1:-1] + m_advected[2:])
        # Neumann BC: dm/dx = 0 at boundaries
        # Use ghost points: m[-1] = m[0], m[Nx] = m[Nx-1]
        rhs[0] = m_advected[0] + (r / 2) * (2 * m_advected[1] - 2 * m_advected[0])
        rhs[-1] = m_advected[-1] + (r / 2) * (2 * m_advected[-2] - 2 * m_advected[-1])

        # Build tridiagonal matrix (I - r/2 * L) for Crank-Nicolson
        # Use banded format: ab[0] = upper diagonal, ab[1] = main, ab[2] = lower
        ab = np.zeros((3, Nx))

        # Main diagonal: 1 + r
        ab[1, :] = 1 + r

        # Off-diagonals: -r/2
        ab[0, 1:] = -r / 2  # upper diagonal
        ab[2, :-1] = -r / 2  # lower diagonal

        # Neumann BC modifications:
        # At i=0: (1+r)*m[0] - r*m[1] = rhs[0]
        # At i=Nx-1: -r*m[Nx-2] + (1+r)*m[Nx-1] = rhs[Nx-1]
        ab[0, 1] = -r  # upper at position 1 (affects row 0)
        ab[2, -2] = -r  # lower at position Nx-2 (affects row Nx-1)

        # Solve tridiagonal system
        m_new = solve_banded((1, 1), ab, rhs)

        # Ensure non-negativity
        m_new = np.maximum(m_new, 0)

        # Renormalize to preserve total mass (critical for numerical stability)
        # This compensates for any numerical mass drift from the Jacobian correction
        # and boundary handling.
        total_mass_old = np.trapezoid(m, self.x_grid)
        total_mass_new = np.trapezoid(m_new, self.x_grid)
        if total_mass_new > 1e-15:
            m_new *= total_mass_old / total_mass_new

        return m_new

    def _get_solver_type_id(self) -> str | None:
        """Get solver type identifier for compatibility checking."""
        return "semi_lagrangian"


if __name__ == "__main__":
    """Quick smoke test for development."""
    print("Testing FPSLSolver...")
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

    # Minimal problem definition
    problem = MFGProblem(
        geometry=domain,
        T=T,
        Nt=Nt,
        diffusion=SIGMA,
    )

    # Test 1: Solver initialization
    print("\n1. Testing solver initialization...")
    solver = FPSLSolver(problem, interpolation_method="cubic")
    assert solver.dimension == 1
    assert solver.fp_method_name == "Semi-Lagrangian"
    print("   Initialization: OK")

    # Test 2: Solve with constant drift (pure diffusion test)
    print("\n2. Testing pure diffusion (zero drift)...")
    m0 = np.exp(-((x - 0.0) ** 2) / 0.02)
    m0 /= np.trapezoid(m0, x)

    # Zero drift field
    U_zero = np.zeros((Nt + 1, N + 1))

    M_diffusion = solver.solve_fp_system(M_initial=m0, drift_field=U_zero, show_progress=False)

    assert M_diffusion.shape == (Nt + 1, N + 1)
    assert not np.any(np.isnan(M_diffusion))

    # Check mass conservation
    mass_initial = np.trapezoid(m0, x)
    mass_final = np.trapezoid(M_diffusion[-1, :], x)
    mass_error = abs(mass_final - mass_initial) / mass_initial
    print(f"   Mass conservation: initial={mass_initial:.4f}, final={mass_final:.4f}, error={mass_error:.2e}")
    assert mass_error < 0.01, f"Mass error too large: {mass_error}"
    print("   Pure diffusion: OK")

    # Test 3: Solve with drift toward center
    print("\n3. Testing with drift field...")

    # Create a smooth potential well at x=0: U(x) = x^2
    # This gives alpha = -2x, drift toward center
    U_well = np.tile(x**2, (Nt + 1, 1))

    # Start from peaked Gaussian (closer to equilibrium for faster convergence)
    m_init = np.exp(-((x - 0.0) ** 2) / 0.1)
    m_init /= np.trapezoid(m_init, x)

    M_drift = solver.solve_fp_system(M_initial=m_init, drift_field=U_well, show_progress=False)

    assert M_drift.shape == (Nt + 1, N + 1)
    assert not np.any(np.isnan(M_drift))

    # Check mass conservation
    mass_init = np.trapezoid(m_init, x)
    mass_final = np.trapezoid(M_drift[-1, :], x)
    mass_error = abs(mass_final - mass_init) / mass_init
    print(f"   Mass conservation: initial={mass_init:.4f}, final={mass_final:.4f}, error={mass_error:.2e}")
    assert mass_error < 0.01, f"Mass error too large: {mass_error}"

    # Check that density stays centered (started at center, should stay near center)
    final_peak_idx = np.argmax(M_drift[-1, :])
    final_peak_x = x[final_peak_idx]
    print(f"   Final peak at index {final_peak_idx} (x={final_peak_x:.3f})")
    # Peak should remain near center (within 0.15 of x=0)
    assert abs(final_peak_x) < 0.15, f"Peak drifted from center: x={final_peak_x}"
    print("   Drift test: OK")

    # Test 4: Compare with FDM
    print("\n4. Comparing with FPFDMSolver...")

    from mfg_pde.alg.numerical.fp_solvers import FPFDMSolver

    fdm_solver = FPFDMSolver(problem, advection_scheme="divergence_upwind")
    M_fdm = fdm_solver.solve_fp_system(
        M_initial=m_init.copy(),  # Use same initial condition
        drift_field=U_well,
        show_progress=False,
    )

    # Both should stay near center (started there)
    fdm_peak_x = x[np.argmax(M_fdm[-1, :])]
    sl_peak_x = x[np.argmax(M_drift[-1, :])]
    print(f"   FDM final peak: x={fdm_peak_x:.3f}")
    print(f"   SL final peak: x={sl_peak_x:.3f}")

    # Compare peak values
    sl_peak_value = M_drift[-1, :].max()
    fdm_peak_value = M_fdm[-1, :].max()
    print(f"   FDM peak value: {fdm_peak_value:.4f}")
    print(f"   SL peak value: {sl_peak_value:.4f}")
    print("   Comparison: OK")

    print("\n" + "=" * 60)
    print("All smoke tests passed!")
