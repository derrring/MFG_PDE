#!/usr/bin/env python3
"""
Semi-Lagrangian HJB Solver for Mean Field Games

This module implements a semi-Lagrangian method for solving the Hamilton-Jacobi-Bellman
equation in MFG problems. The method follows characteristics backward in time and uses
interpolation to compute values at departure points.

The HJB equation solved is:
    ∂u/∂t + H(x, ∇u, m) - σ²/2 Δu = 0    in [0,T) × Ω
    u(T, x) = g(x)                         at t = T

where H is the Hamiltonian and the semi-Lagrangian scheme discretizes this as:
    (u^{n+1} - û^n) / Δt + H(x, ∇û^n, m^{n+1}) - σ²/2 Δû^n = 0

where û^n is the value interpolated at the departure point of the characteristic.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from mfg_pde.utils.pde_coefficients import check_adi_compatibility

from .base_hjb import BaseHJBSolver
from .hjb_sl_adi import (
    adi_diffusion_step,
    solve_crank_nicolson_diffusion_1d,
)
from .hjb_sl_characteristics import (
    apply_boundary_conditions_1d,
    apply_boundary_conditions_nd,
    trace_characteristic_backward_1d,
    trace_characteristic_backward_nd,
)
from .hjb_sl_interpolation import (
    interpolate_nearest_neighbor,
    interpolate_value_1d,
    interpolate_value_nd,
    interpolate_value_rbf_fallback,
)

if TYPE_CHECKING:
    from mfg_pde.core.mfg_problem import MFGProblem

logger = logging.getLogger(__name__)

try:
    import jax.numpy as jnp
    from jax import jit

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


class HJBSemiLagrangianSolver(BaseHJBSolver):
    """
    Semi-Lagrangian method for solving Hamilton-Jacobi-Bellman equations.

    The semi-Lagrangian method discretizes the HJB equation by following
    characteristics backward in time and interpolating values at departure points.
    This approach is particularly stable for convection-dominated problems.

    Key features:
    - Stable for large time steps
    - Handles discontinuous solutions well
    - Natural upwind discretization
    - Monotone and conservative

    Dimension support:
    - 1D: Full support (production-ready)
    - nD (2D/3D/4D+): Full support (2025-11-02)
      - Interpolation: RegularGridInterpolator (complete)
      - Diffusion: nD Laplacian (complete)
      - Characteristic tracing: Vector form (complete)
      - Optimal control: Vector optimization (complete)
    """

    def __init__(
        self,
        problem: MFGProblem,
        interpolation_method: str = "linear",
        optimization_method: str = "brent",
        characteristic_solver: str = "explicit_euler",
        use_rbf_fallback: bool = True,
        rbf_kernel: str = "thin_plate_spline",
        use_jax: bool | None = None,
        tolerance: float = 1e-8,
        max_char_iterations: int = 100,
        check_cfl: bool = True,
        enable_adaptive_substepping: bool = True,
        max_substeps: int = 100,
        cfl_target: float = 0.9,
    ):
        """
        Initialize semi-Lagrangian HJB solver.

        Args:
            problem: MFG problem instance
            interpolation_method: Method for interpolating values
                - 'linear': Linear interpolation (fastest, C⁰ continuous)
                - 'cubic': Cubic spline interpolation (slower, C² continuous)
                - 'quintic': Quintic interpolation (slowest, highest accuracy, nD only)
                - 'nearest': Nearest neighbor (for debugging)
            optimization_method: Method for Hamiltonian optimization ('brent', 'golden')
            characteristic_solver: Method for solving characteristics
                - 'explicit_euler': First-order explicit Euler (fastest, least accurate)
                - 'rk2': Second-order Runge-Kutta midpoint method
                - 'rk4': Fourth-order Runge-Kutta via scipy.solve_ivp (most accurate)
            use_rbf_fallback: Use RBF interpolation as fallback for boundary cases
            rbf_kernel: RBF kernel function
                - 'thin_plate_spline': Smooth, no free parameters (recommended)
                - 'multiquadric': Good for scattered data
                - 'gaussian': Localized influence
            use_jax: Whether to use JAX acceleration (auto-detect if None)
            tolerance: Convergence tolerance for optimization
            max_char_iterations: Maximum iterations for characteristic solving
            check_cfl: Whether to check CFL condition and issue warnings (default: True).
                CFL = max|grad(u)| * dt / dx. Warns if CFL > 1.0.
            enable_adaptive_substepping: Whether to automatically subdivide time steps
                when CFL > 1.0 to maintain stability (default: True). When enabled,
                the solver will use smaller internal time steps while preserving the
                overall time discretization.
            max_substeps: Maximum number of substeps per time step when adaptive
                substepping is enabled (default: 100). If more substeps are needed,
                a warning is issued and the solver proceeds with max_substeps.
            cfl_target: Target CFL number for adaptive substepping (default: 0.9).
                When CFL > 1.0, the time step is subdivided to achieve CFL ≤ cfl_target.
        """
        super().__init__(problem)
        self.hjb_method_name = "Semi-Lagrangian"

        # Solver configuration
        self.interpolation_method = interpolation_method
        self.optimization_method = optimization_method
        self.characteristic_solver = characteristic_solver
        self.use_rbf_fallback = use_rbf_fallback
        self.rbf_kernel = rbf_kernel
        self.tolerance = tolerance
        self.max_char_iterations = max_char_iterations
        self.check_cfl = check_cfl
        self.enable_adaptive_substepping = enable_adaptive_substepping
        self.max_substeps = max_substeps
        self.cfl_target = cfl_target

        # JAX acceleration
        self.use_jax = use_jax if use_jax is not None else JAX_AVAILABLE
        if self.use_jax and not JAX_AVAILABLE:
            logger.warning("JAX not available, falling back to NumPy")
            self.use_jax = False

        # Detect problem dimension
        self.dimension = self._detect_dimension(problem)

        # Precompute grid and time parameters (dimension-agnostic)
        if self.dimension == 1:
            # 1D problem: Use legacy attributes
            self.x_grid = np.linspace(problem.xmin, problem.xmax, problem.Nx + 1)
            self.dt = problem.dt
            self.dx = problem.dx
            self.grid = None  # No TensorProductGrid for 1D
        else:
            # nD problem: Use CartesianGrid interface
            from mfg_pde.geometry.base import CartesianGrid

            if not isinstance(problem.geometry, CartesianGrid):
                raise ValueError(
                    f"Multi-dimensional problem must have CartesianGrid geometry (TensorProductGrid). "
                    f"Got dimension={self.dimension}"
                )
            self.grid = problem.geometry  # Geometry IS the grid
            self.dt = problem.dt
            # Grid spacing: vector of spacings in each dimension
            self.spacing = np.array(self.grid.get_grid_spacing())
            # Grid shape: use get_grid_shape() for CartesianGrid interface compatibility
            self._grid_shape = tuple(self.grid.get_grid_shape())
            self._num_points_total = int(np.prod(self._grid_shape))
            self.x_grid = None  # Not used for nD

            # Check ADI compatibility for nD diffusion
            adi_ok, adi_msg = check_adi_compatibility(problem.sigma)
            self._adi_compatible = adi_ok
            if not adi_ok:
                logger.warning(
                    f"Diffusion tensor not ADI-compatible: {adi_msg}. "
                    f"ADI scheme may be inaccurate. Consider using more timesteps "
                    f"or implementing Craig-Sneyd scheme for mixed derivatives."
                )
            else:
                logger.info(f"ADI diffusion enabled for nD solve: {adi_msg}")

        # Setup JAX functions if available
        if self.use_jax:
            self._setup_jax_functions()

    def _setup_jax_functions(self):
        """Setup JAX-accelerated functions for performance."""
        if not self.use_jax:
            return

        @jit
        def jax_interpolate_linear(x_points, y_values, x_query):
            """JAX-accelerated linear interpolation."""
            return jnp.interp(x_query, x_points, y_values)

        @jit
        def jax_solve_characteristic_euler(x_current, p_optimal, dt):
            """JAX-accelerated characteristic solving using Euler method."""
            return x_current - p_optimal * dt

        self._jax_interpolate = jax_interpolate_linear
        self._jax_solve_characteristic = jax_solve_characteristic_euler

    def _detect_dimension(self, problem) -> int:
        """
        Detect the dimension of the problem.

        Args:
            problem: MFGProblem instance (supports 1D, 2D, 3D, nD)

        Returns:
            dimension: 1 for 1D problems, 2 for 2D, 3 for 3D, etc.

        Raises:
            ValueError: If dimension cannot be determined
        """
        # Try geometry.dimension first (unified interface)
        if hasattr(problem, "geometry") and hasattr(problem.geometry, "dimension"):
            return problem.geometry.dimension

        # Fall back to problem.dimension
        if hasattr(problem, "dimension"):
            return problem.dimension

        # Legacy 1D detection
        if getattr(problem, "Nx", None) is not None and getattr(problem, "Ny", None) is None:
            return 1

        # If we can't determine dimension, raise error
        raise ValueError("Cannot determine problem dimension. Use MFGProblem with spatial_bounds for nD support.")

    def _compute_gradient(self, u_values: np.ndarray, check_cfl: bool = True) -> np.ndarray | tuple[np.ndarray, ...]:
        """
        Compute gradient ∇u for optimal control using np.gradient.

        For standard MFG with quadratic control cost, the optimal control is:
            α*(x,t) = ∇u(x,t)

        Args:
            u_values: Value function array
                - 1D: shape (Nx+1,)
                - nD: shape (Nx1+1, Nx2+1, ..., Nxd+1)
            check_cfl: Whether to check CFL condition (default: True)

        Returns:
            gradient: Gradient array(s)
                - 1D: shape (Nx+1,) - scalar gradient at each point
                - nD: tuple of d arrays, each shape (Nx1+1, ..., Nxd+1)

        Note:
            Uses np.gradient with edge_order=2 for accurate boundary gradients.
            Issues CFL warning if max|∇u|·dt/dx > 1.
        """
        if self.dimension == 1:
            # 1D gradient computation
            grad_u = np.gradient(u_values, self.dx, edge_order=2)

            # CFL check
            if check_cfl and self.check_cfl:
                max_grad = np.max(np.abs(grad_u))
                cfl = max_grad * self.dt / self.dx
                if cfl > 1.0:
                    logger.warning(
                        f"CFL condition violated: max|∇u|·dt/dx = {cfl:.3f} > 1.0. "
                        f"Consider reducing dt or increasing dx. "
                        f"max|∇u| = {max_grad:.3f}, dt = {self.dt:.6f}, dx = {self.dx:.6f}"
                    )

            return grad_u

        else:
            # nD gradient computation
            grad_components = []
            for axis in range(self.dimension):
                grad_axis = np.gradient(u_values, self.spacing[axis], axis=axis, edge_order=2)
                grad_components.append(grad_axis)

            # CFL check
            if check_cfl and self.check_cfl:
                grad = np.stack(grad_components, axis=0)
                magnitude = np.sqrt(np.sum(grad**2, axis=0))
                max_grad = np.max(magnitude)
                min_spacing = np.min(self.spacing)
                cfl = max_grad * self.dt / min_spacing
                if cfl > 1.0:
                    logger.warning(
                        f"CFL condition violated: max|∇u|·dt/dx_min = {cfl:.3f} > 1.0. "
                        f"Consider reducing dt or increasing grid spacing. "
                        f"max|∇u| = {max_grad:.3f}, dt = {self.dt:.6f}, dx_min = {min_spacing:.6f}"
                    )

            # Return as tuple of arrays (one per dimension)
            return tuple(grad_components)

    def _compute_cfl_and_substeps(self, u_values: np.ndarray, dt_target: float) -> tuple[float, int, float]:
        """
        Compute CFL number and determine optimal number of substeps.

        When the CFL condition (CFL = max|grad(u)| * dt / dx) exceeds 1.0,
        this method computes how many substeps are needed to maintain
        CFL <= cfl_target (default 0.9).

        Args:
            u_values: Current value function array
            dt_target: Target time step (full time step to subdivide)

        Returns:
            Tuple of (cfl_number, n_substeps, dt_substep):
                - cfl_number: The CFL number with the target dt
                - n_substeps: Number of substeps needed (1 if CFL <= 1.0)
                - dt_substep: Time step to use for each substep
        """
        if self.dimension == 1:
            # 1D CFL computation
            grad_u = np.gradient(u_values, self.dx, edge_order=2)
            max_grad = np.max(np.abs(grad_u))
            cfl = max_grad * dt_target / self.dx
            dx_eff = self.dx
        else:
            # nD CFL computation
            grad_components = []
            for axis in range(self.dimension):
                grad_axis = np.gradient(u_values, self.spacing[axis], axis=axis, edge_order=2)
                grad_components.append(grad_axis)

            grad = np.stack(grad_components, axis=0)
            magnitude = np.sqrt(np.sum(grad**2, axis=0))
            max_grad = np.max(magnitude)
            dx_eff = np.min(self.spacing)
            cfl = max_grad * dt_target / dx_eff

        # Determine substeps needed
        if cfl <= 1.0 or not self.enable_adaptive_substepping:
            return cfl, 1, dt_target

        # Compute substeps to achieve CFL <= cfl_target
        n_substeps = int(np.ceil(cfl / self.cfl_target))
        n_substeps = min(n_substeps, self.max_substeps)

        if n_substeps >= self.max_substeps:
            logger.warning(
                f"CFL = {cfl:.2f} requires {int(np.ceil(cfl / self.cfl_target))} substeps, "
                f"capped at max_substeps={self.max_substeps}. "
                f"Stability may be compromised. Consider reducing dt or increasing grid resolution."
            )

        dt_substep = dt_target / n_substeps
        actual_cfl = max_grad * dt_substep / dx_eff

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Adaptive substepping: CFL={cfl:.2f} -> {actual_cfl:.2f} ({n_substeps} substeps, dt={dt_substep:.6f})"
            )

        return cfl, n_substeps, dt_substep

    def solve_hjb_system(
        self,
        M_density: np.ndarray | None = None,
        U_terminal: np.ndarray | None = None,
        U_coupling_prev: np.ndarray | None = None,
        diffusion_field: float | np.ndarray | None = None,
        # Deprecated parameter names for backward compatibility
        M_density_evolution_from_FP: np.ndarray | None = None,
        U_final_condition_at_T: np.ndarray | None = None,
        U_from_prev_picard: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Solve the HJB system using semi-Lagrangian method.

        The semi-Lagrangian discretization of the HJB equation:
            ∂u/∂t + H(x, ∇u, m) - σ²/2 Δu = 0

        is solved by following characteristics backward in time:
            1. For each grid point x_i at time t^{n+1}
            2. Find optimal control p* that minimizes H(x_i, p, m^{n+1})
            3. Trace characteristic backward: X(t^n) = x_i - p* Δt
            4. Interpolate u^n at departure point X(t^n)
            5. Update: u^{n+1}_i = û^n(X(t^n)) - Δt[H(...) - σ²/2 Δu]

        Args:
            M_density: (Nt, *spatial_shape) density from FP solver
            U_terminal: (*spatial_shape,) terminal condition u(T, x)
            U_coupling_prev: (Nt, *spatial_shape) previous coupling iteration estimate
            diffusion_field: Optional diffusion coefficient override

        Returns:
            (Nt, *grid_shape) solution array for value function
        """
        # Handle deprecated parameter names with warnings
        if M_density_evolution_from_FP is not None:
            if M_density is not None:
                raise ValueError("Cannot specify both 'M_density' and deprecated 'M_density_evolution_from_FP'")
            warnings.warn(
                "Parameter 'M_density_evolution_from_FP' is deprecated. Use 'M_density' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            M_density = M_density_evolution_from_FP

        if U_final_condition_at_T is not None:
            if U_terminal is not None:
                raise ValueError("Cannot specify both 'U_terminal' and deprecated 'U_final_condition_at_T'")
            warnings.warn(
                "Parameter 'U_final_condition_at_T' is deprecated. Use 'U_terminal' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            U_terminal = U_final_condition_at_T

        if U_from_prev_picard is not None:
            if U_coupling_prev is not None:
                raise ValueError("Cannot specify both 'U_coupling_prev' and deprecated 'U_from_prev_picard'")
            warnings.warn(
                "Parameter 'U_from_prev_picard' is deprecated. Use 'U_coupling_prev' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            U_coupling_prev = U_from_prev_picard

        # Validate required parameters
        if M_density is None:
            raise ValueError("M_density is required")
        if U_terminal is None:
            raise ValueError("U_terminal is required")
        if U_coupling_prev is None:
            raise ValueError("U_coupling_prev is required")

        # Handle multi-dimensional grids
        shape = M_density.shape
        Nt = shape[0]
        grid_shape = shape[1:]  # Remaining dimensions

        # Output shape: (Nt + 1, *grid_shape) to include terminal condition at t=T
        U_solution = np.zeros((Nt + 1, *grid_shape))

        # Set final condition at t=T (index Nt)
        U_solution[Nt] = U_terminal

        total_points = np.prod(grid_shape)
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                f"Starting semi-Lagrangian HJB solve: {Nt + 1} time steps (including terminal), {total_points} spatial points ({grid_shape})"
            )

        # Solve backward in time using semi-Lagrangian method (from Nt-1 down to 0)
        total_substeps_used = 0
        for n in range(Nt - 1, -1, -1):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Solving time step {n}/{Nt - 1}")

            # For density and coupling, use min(n+1, Nt-1) to handle edge case at n=Nt-1
            # where M_density and U_coupling_prev don't have index Nt
            m_idx = min(n + 1, Nt - 1) if n + 1 >= Nt else n + 1
            u_prev_idx = min(n, Nt - 1)

            # Compute CFL and determine substeps needed for this time step
            cfl, n_substeps, dt_substep = self._compute_cfl_and_substeps(U_solution[n + 1], self.dt)
            total_substeps_used += n_substeps

            if n_substeps == 1:
                # No substepping needed - use standard time step
                U_solution[n] = self._solve_timestep_semi_lagrangian(
                    U_solution[n + 1],  # u^{n+1} (from output array, always valid)
                    M_density[m_idx],  # m^{n+1} or last available density
                    U_coupling_prev[u_prev_idx],  # u_k^n for coupling terms
                    n,  # time index
                )
            else:
                # Adaptive substepping: subdivide the time step
                U_current = U_solution[n + 1].copy()
                for substep in range(n_substeps):
                    U_current = self._solve_timestep_semi_lagrangian_with_dt(
                        U_current,
                        M_density[m_idx],
                        U_coupling_prev[u_prev_idx],
                        n,
                        dt_substep,
                    )
                    # Check for numerical issues after each substep
                    if np.any(np.isnan(U_current) | np.isinf(U_current)):
                        error_msg = (
                            f"Semi-Lagrangian solver failed at time step {n}/{Nt - 1}, "
                            f"substep {substep + 1}/{n_substeps} with NaN/Inf values. "
                            f"CFL was {cfl:.2f}, using {n_substeps} substeps with dt={dt_substep:.6f}"
                        )
                        logger.error(error_msg)
                        raise ValueError(error_msg)
                U_solution[n] = U_current
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Time step {n}: used {n_substeps} substeps (CFL={cfl:.2f})")

            # Check for numerical issues
            if np.any(np.isnan(U_solution[n]) | np.isinf(U_solution[n])):
                error_msg = (
                    f"Semi-Lagrangian solver failed at time step {n}/{Nt - 1} with NaN/Inf values. "
                    "Possible causes:\n"
                    "  1. CFL condition violated (try smaller dt or enable adaptive_substepping=True)\n"
                    "  2. Grid too coarse for solution features\n"
                    "  3. Hamiltonian evaluation issues\n"
                    "  4. Interpolation errors near boundaries"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

        if logger.isEnabledFor(logging.INFO):
            final_residual = np.linalg.norm(U_solution[1] - U_solution[0])
            logger.info(f"Semi-Lagrangian HJB solve completed. Final residual: {final_residual:.2e}")

        return U_solution

    def _solve_timestep_semi_lagrangian(
        self,
        U_next: np.ndarray,
        M_next: np.ndarray,
        U_prev_picard: np.ndarray,
        time_idx: int,
    ) -> np.ndarray:
        """
        Solve one timestep using semi-Lagrangian method (supports 1D and nD).

        Args:
            U_next: Value function at next time step
                - 1D: shape (Nx,)
                - nD: shape matching grid.num_points
            M_next: Density at next time step (same shape as U_next)
            U_prev_picard: Value from previous Picard iteration (for coupling)
            time_idx: Current time index

        Returns:
            Value function at current time step (same shape as U_next)
        """
        if self.dimension == 1:
            # 1D solve with operator splitting: characteristics + Crank-Nicolson diffusion
            Nx = len(U_next)
            U_star = np.zeros(Nx)  # Intermediate solution after advection

            # Compute gradient for optimal control: α* = ∇u
            grad_u = self._compute_gradient(U_next, check_cfl=True)

            # Step 1: Advection along characteristics (explicit)
            for i in range(Nx):
                x_current = self.x_grid[i]
                m_current = M_next[i]

                try:
                    # Optimal control from gradient
                    p_optimal = grad_u[i]
                    x_departure = self._trace_characteristic_backward(x_current, p_optimal, self.dt)
                    u_departure = self._interpolate_value(U_next, x_departure)
                    hamiltonian_value = self._evaluate_hamiltonian(x_current, p_optimal, m_current, time_idx)

                    # Advection step: u* = u(X(t-dt)) - dt * H(x, p*, m)
                    U_star[i] = u_departure - self.dt * hamiltonian_value

                except Exception as e:
                    logger.warning(f"Error at grid point {i}: {e}")
                    U_star[i] = U_next[i]

            # Step 2: Diffusion (Crank-Nicolson for unconditional stability)
            # Solve: (I - 0.5*dt*sigma^2*L) u^n = (I + 0.5*dt*sigma^2*L) u*
            # where L is the Laplacian operator
            U_current = self._solve_crank_nicolson_diffusion(U_star, self.dt, self.problem.sigma)

            return U_current

        else:
            # nD solve with operator splitting: advection + ADI diffusion
            # Reshape arrays to grid shape for easier indexing
            if U_next.ndim == 1:
                # Infer grid shape from array size (handles both full grid and interior points)
                total_points = U_next.size
                expected_full = int(np.prod(self._grid_shape))

                if total_points == expected_full:
                    grid_shape = tuple(self._grid_shape)
                else:
                    # Interior points only (num_points - 1 in each dimension)
                    grid_shape = tuple(n - 1 for n in self._grid_shape)

                U_next_shaped = U_next.reshape(grid_shape)
                M_next_shaped = M_next.reshape(grid_shape)
            else:
                U_next_shaped = U_next
                M_next_shaped = M_next
                grid_shape = U_next_shaped.shape

            # Step 1: Advection pass - compute u_star for all points
            # u_star = u(X(t-dt)) - dt * H(x, p*, m)
            U_star = np.zeros_like(U_next_shaped)

            # Compute gradient for optimal control: alpha* = grad(u)
            # Returns tuple of gradient components, each with shape grid_shape
            grad_components = self._compute_gradient(U_next_shaped, check_cfl=True)

            # Track errors for diagnostics
            error_count = 0
            total_points = int(np.prod(grid_shape))

            # Iterate over all grid points for advection
            for multi_idx in np.ndindex(grid_shape):
                # Get spatial coordinates for this grid point
                x_current = np.array([self.grid.coordinates[d][multi_idx[d]] for d in range(self.dimension)])
                m_current = M_next_shaped[multi_idx]

                # Extract optimal control from gradient (vector for nD)
                p_optimal = np.array([grad_components[d][multi_idx] for d in range(self.dimension)])

                # Trace characteristic backward (vector operation)
                x_departure = self._trace_characteristic_backward(x_current, p_optimal, self.dt)

                # Interpolate at departure point
                u_departure = self._interpolate_value(U_next_shaped, x_departure)

                # Evaluate Hamiltonian
                hamiltonian_value = self._evaluate_hamiltonian(x_current, p_optimal, m_current, time_idx)

                # Advection step: u_star = u(X(t-dt)) - dt * H(x, p*, m)
                u_star_val = u_departure - self.dt * hamiltonian_value

                # Check for numerical issues
                if np.isnan(u_star_val) or np.isinf(u_star_val):
                    error_count += 1
                    if error_count <= 5:
                        logger.warning(
                            f"NaN/Inf at grid point {multi_idx}: "
                            f"u_departure={u_departure:.3e}, H={hamiltonian_value:.3e}"
                        )
                    U_star[multi_idx] = U_next_shaped[multi_idx]  # Fallback
                else:
                    U_star[multi_idx] = u_star_val

            # Report error summary if any occurred in advection
            if error_count > 0:
                error_pct = 100 * error_count / total_points
                if error_pct > 10:
                    raise ValueError(
                        f"Semi-Lagrangian advection failed: {error_count}/{total_points} points ({error_pct:.1f}%) "
                        f"had NaN/Inf values at time step {time_idx}. Check grid resolution and time step."
                    )
                else:
                    logger.warning(
                        f"Semi-Lagrangian advection: {error_count}/{total_points} points ({error_pct:.1f}%) "
                        f"had NaN/Inf values at time step {time_idx}"
                    )

            # Step 2: ADI diffusion pass
            # Apply implicit diffusion using ADI (Peaceman-Rachford)
            U_current_shaped = self._adi_diffusion_step(U_star, self.dt)

            # Return flattened if input was flattened
            if U_next.ndim == 1:
                return U_current_shaped.ravel()
            else:
                return U_current_shaped

    def _solve_timestep_semi_lagrangian_with_dt(
        self,
        U_next: np.ndarray,
        M_next: np.ndarray,
        U_prev_picard: np.ndarray,
        time_idx: int,
        dt: float,
    ) -> np.ndarray:
        """
        Solve one timestep using semi-Lagrangian method with custom time step.

        This is the same as _solve_timestep_semi_lagrangian but allows specifying
        a custom dt for adaptive substepping.

        Args:
            U_next: Value function at next time step
            M_next: Density at next time step
            U_prev_picard: Value from previous Picard iteration
            time_idx: Current time index
            dt: Time step to use (allows custom dt for substepping)

        Returns:
            Value function at current time step
        """
        if self.dimension == 1:
            # 1D solve with operator splitting
            Nx = len(U_next)
            U_star = np.zeros(Nx)

            # Compute gradient for optimal control
            grad_u = self._compute_gradient(U_next, check_cfl=False)

            # Step 1: Advection along characteristics
            for i in range(Nx):
                x_current = self.x_grid[i]
                m_current = M_next[i]

                try:
                    p_optimal = grad_u[i]
                    x_departure = self._trace_characteristic_backward(x_current, p_optimal, dt)
                    u_departure = self._interpolate_value(U_next, x_departure)
                    hamiltonian_value = self._evaluate_hamiltonian(x_current, p_optimal, m_current, time_idx)

                    U_star[i] = u_departure - dt * hamiltonian_value

                except Exception as e:
                    logger.warning(f"Error at grid point {i}: {e}")
                    U_star[i] = U_next[i]

            # Step 2: Diffusion with custom dt
            U_current = self._solve_crank_nicolson_diffusion(U_star, dt, self.problem.sigma)
            return U_current

        else:
            # nD solve with operator splitting
            if U_next.ndim == 1:
                total_points = U_next.size
                expected_full = int(np.prod(self._grid_shape))

                if total_points == expected_full:
                    grid_shape = tuple(self._grid_shape)
                else:
                    grid_shape = tuple(n - 1 for n in self._grid_shape)

                U_next_shaped = U_next.reshape(grid_shape)
                M_next_shaped = M_next.reshape(grid_shape)
            else:
                U_next_shaped = U_next
                M_next_shaped = M_next
                grid_shape = U_next_shaped.shape

            U_star = np.zeros_like(U_next_shaped)
            grad_components = self._compute_gradient(U_next_shaped, check_cfl=False)

            error_count = 0
            total_points = int(np.prod(grid_shape))

            for multi_idx in np.ndindex(grid_shape):
                x_current = np.array([self.grid.coordinates[d][multi_idx[d]] for d in range(self.dimension)])
                m_current = M_next_shaped[multi_idx]
                p_optimal = np.array([grad_components[d][multi_idx] for d in range(self.dimension)])

                x_departure = self._trace_characteristic_backward(x_current, p_optimal, dt)
                u_departure = self._interpolate_value(U_next_shaped, x_departure)
                hamiltonian_value = self._evaluate_hamiltonian(x_current, p_optimal, m_current, time_idx)

                u_star_val = u_departure - dt * hamiltonian_value

                if np.isnan(u_star_val) or np.isinf(u_star_val):
                    error_count += 1
                    U_star[multi_idx] = U_next_shaped[multi_idx]
                else:
                    U_star[multi_idx] = u_star_val

            if error_count > 0:
                error_pct = 100 * error_count / total_points
                if error_pct > 10:
                    raise ValueError(
                        f"Semi-Lagrangian advection failed: {error_count}/{total_points} points ({error_pct:.1f}%) "
                        f"had NaN/Inf values at time step {time_idx}."
                    )

            # ADI diffusion with custom dt
            U_current_shaped = self._adi_diffusion_step(U_star, dt)

            if U_next.ndim == 1:
                return U_current_shaped.ravel()
            else:
                return U_current_shaped

    def _find_optimal_control(self, x: np.ndarray | float, m: float, time_idx: int) -> np.ndarray | float:
        """
        Find optimal control p* that minimizes H(x, p, m) (supports 1D and nD).

        For the standard MFG Hamiltonian H(x, p, m) = |p|²/2 + V(x) + C(x,m),
        the optimal control is p* = 0 in all dimensions.

        For general Hamiltonians:
        - 1D: Uses scalar optimization (minimize_scalar with Brent/Golden search)
        - nD: Uses vector optimization (scipy.optimize.minimize with L-BFGS-B)

        Args:
            x: Spatial position
                - 1D: scalar float
                - nD: array of shape (dimension,)
            m: Density value
            time_idx: Time index

        Returns:
            Optimal control value p*
                - 1D: scalar float
                - nD: array of shape (dimension,)
        """
        if self.dimension == 1:
            # 1D optimal control
            x_scalar = float(x) if np.ndim(x) > 0 else x

            # For standard MFG problems, analytical solution
            if hasattr(self.problem, "coupling_coefficient"):
                return 0.0

            # For general Hamiltonians, use numerical optimization
            def hamiltonian_objective(p):
                derivs = {(0,): 0.0, (1,): p}
                try:
                    x_idx = int((x_scalar - self.problem.xmin) / self.dx)
                    x_idx = np.clip(x_idx, 0, self.problem.Nx)
                    return self.problem.H(x_idx, m, derivs=derivs, t_idx=time_idx)
                except Exception:
                    return np.inf

            try:
                if self.optimization_method == "brent":
                    result = minimize_scalar(
                        hamiltonian_objective,
                        bounds=(-10.0, 10.0),
                        method="bounded",
                        options={"xatol": self.tolerance},
                    )
                else:
                    result = minimize_scalar(
                        hamiltonian_objective,
                        bounds=(-10.0, 10.0),
                        method="golden",
                        options={"xtol": self.tolerance},
                    )

                return result.x if result.success else 0.0

            except Exception as e:
                logger.debug(f"Optimization failed at x={x_scalar}: {e}")
                return 0.0

        else:
            # nD optimal control
            # For standard quadratic Hamiltonian: H = |p|²/2 + ..., optimal p* = 0
            if hasattr(self.problem, "coupling_coefficient"):
                return np.zeros(self.dimension)

            # Vector optimization using scipy.optimize.minimize
            # Objective: minimize H(x, p, m) over p ∈ ℝ^d
            def hamiltonian_objective(p_vec):
                """Objective function for vector optimization: H(x, p, m)"""
                try:
                    # Call problem's Hamiltonian function if available
                    if hasattr(self.problem, "hamiltonian"):
                        t_value = time_idx * self.problem.T / self.problem.Nt if time_idx is not None else 0.0
                        return self.problem.hamiltonian(x, m, p_vec, t_value)
                    else:
                        # Fallback: standard quadratic Hamiltonian H = |p|²/2 + C*m
                        p_norm_sq = np.sum(p_vec**2)
                        coef_CT = getattr(self.problem, "coupling_coefficient", 0.5)
                        return 0.5 * p_norm_sq + coef_CT * m
                except Exception:
                    # Fallback: quadratic in p
                    return 0.5 * np.sum(p_vec**2)

            # Initial guess: zero vector
            p0 = np.zeros(self.dimension)

            try:
                # Use L-BFGS-B for smooth, unconstrained optimization
                result = minimize(
                    hamiltonian_objective,
                    p0,
                    method="L-BFGS-B",
                    options={"ftol": self.tolerance, "maxiter": 100},
                )

                if result.success:
                    return result.x
                else:
                    logger.debug(f"Vector optimization did not converge: {result.message}")
                    return p0

            except Exception as e:
                logger.debug(f"Vector optimization failed at x={x}: {e}")
                return p0

    def _trace_characteristic_backward(
        self, x_current: np.ndarray | float, p_optimal: np.ndarray | float, dt: float
    ) -> np.ndarray | float:
        """
        Trace characteristic backward in time to find departure point (supports 1D and nD).

        Delegates to hjb_sl_characteristics module functions.

        Args:
            x_current: Current spatial position
                - 1D: scalar float
                - nD: array of shape (dimension,), e.g., [x, y] for 2D
            p_optimal: Optimal control value
                - 1D: scalar float
                - nD: array of shape (dimension,), e.g., [px, py] for 2D
            dt: Time step size

        Returns:
            Departure point X(t-dt)
                - 1D: scalar float
                - nD: array of shape (dimension,)
        """
        if self.dimension == 1:
            # 1D characteristic tracing
            jax_fn = self._jax_solve_characteristic if self.use_jax else None
            x_departure = trace_characteristic_backward_1d(
                x_current,
                p_optimal,
                dt,
                method=self.characteristic_solver,
                use_jax=self.use_jax,
                jax_solve_fn=jax_fn,
            )

            # Apply boundary conditions
            bc_type = None
            if hasattr(self.problem, "boundary_conditions"):
                bc = getattr(self.problem, "boundary_conditions", None)
                if bc is not None and hasattr(bc, "type"):
                    bc_type = bc.type

            return apply_boundary_conditions_1d(
                x_departure,
                xmin=self.problem.xmin,
                xmax=self.problem.xmax,
                bc_type=bc_type,
            )

        else:
            # nD characteristic tracing
            x_departure = trace_characteristic_backward_nd(
                x_current,
                p_optimal,
                dt,
                dimension=self.dimension,
                method=self.characteristic_solver,
            )

            # Apply boundary conditions
            return apply_boundary_conditions_nd(
                x_departure,
                bounds=self.grid.bounds,
                bc_type=None,  # nD clamping only for now
            )

    def _interpolate_value(self, U_values: np.ndarray, x_query: np.ndarray | float) -> float:
        """
        Interpolate value function at query point (supports 1D and nD).

        Delegates to hjb_sl_interpolation module functions.

        Args:
            U_values: Value function on grid
                - 1D: shape (Nx,)
                - nD: shape matching grid.num_points, e.g., (Nx, Ny) for 2D
            x_query: Query point for interpolation
                - 1D: scalar float
                - nD: array of shape (dimension,), e.g., [x, y] for 2D

        Returns:
            Interpolated value at query point
        """
        if self.dimension == 1:
            # 1D interpolation
            jax_fn = self._jax_interpolate if self.use_jax else None
            return interpolate_value_1d(
                U_values,
                x_query,
                self.x_grid,
                method=self.interpolation_method,
                xmin=self.problem.xmin,
                xmax=self.problem.xmax,
                use_jax=self.use_jax,
                jax_interpolate_fn=jax_fn,
            )

        else:
            # nD interpolation
            grid_coords = tuple(self.grid.coordinates)
            grid_shape = tuple(self._grid_shape)

            try:
                return interpolate_value_nd(
                    U_values,
                    x_query,
                    grid_coords,
                    grid_shape,
                    method=self.interpolation_method,
                )
            except Exception as e:
                logger.debug(f"nD interpolation failed at x={x_query}: {e}")

                # Try RBF fallback if enabled
                if self.use_rbf_fallback:
                    try:
                        return interpolate_value_rbf_fallback(
                            U_values,
                            x_query,
                            grid_coords,
                            grid_shape,
                            rbf_kernel=self.rbf_kernel,
                        )
                    except Exception as rbf_error:
                        logger.debug(f"RBF fallback failed: {rbf_error}")

                # Final fallback: nearest neighbor
                return interpolate_nearest_neighbor(
                    U_values,
                    x_query,
                    grid_coords,
                    grid_shape,
                )

    def _compute_diffusion_term(self, U_values: np.ndarray, idx: int | tuple) -> float:
        """
        Compute discrete Laplacian (diffusion term) at grid point (supports 1D and nD).

        1D: Uses standard finite difference (U[i+1] - 2*U[i] + U[i-1]) / dx²
        nD: Computes Laplacian as sum over dimensions: Δu = Σ_d ∂²u/∂x_d²

        Args:
            U_values: Value function array
                - 1D: shape (Nx,)
                - nD: shape matching grid.num_points
            idx: Grid point index
                - 1D: scalar integer i
                - nD: tuple of indices, e.g., (i, j) for 2D

        Returns:
            Discrete Laplacian value
        """
        if self.dimension == 1:
            # 1D Laplacian: Use existing logic
            i = int(idx)
            Nx = len(U_values)

            if Nx <= 2:
                return 0.0

            # Handle boundary points
            if i == 0:
                if hasattr(self.problem, "boundary_conditions"):
                    bc = getattr(self.problem, "boundary_conditions", None)
                    if bc is not None and hasattr(bc, "type") and bc.type == "periodic":
                        laplacian = (U_values[1] - 2 * U_values[0] + U_values[-1]) / self.dx**2
                    else:
                        laplacian = (U_values[1] - U_values[0]) / self.dx**2
                else:
                    laplacian = (U_values[1] - U_values[0]) / self.dx**2

            elif i == Nx - 1:
                if hasattr(self.problem, "boundary_conditions"):
                    bc = getattr(self.problem, "boundary_conditions", None)
                    if bc is not None and hasattr(bc, "type") and bc.type == "periodic":
                        laplacian = (U_values[0] - 2 * U_values[-1] + U_values[-2]) / self.dx**2
                    else:
                        laplacian = (U_values[-1] - U_values[-2]) / self.dx**2
                else:
                    laplacian = (U_values[-1] - U_values[-2]) / self.dx**2

            else:
                # Central difference for interior points
                laplacian = (U_values[i + 1] - 2 * U_values[i] + U_values[i - 1]) / self.dx**2

            return laplacian

        else:
            # nD Laplacian: Sum of second derivatives in each dimension
            # Δu = ∂²u/∂x₁² + ∂²u/∂x₂² + ...

            # Ensure U_values is reshaped to grid shape
            if U_values.ndim == 1:
                U_shaped = U_values.reshape(self._grid_shape)
            else:
                U_shaped = U_values

            # Get multi-index
            if isinstance(idx, (tuple, list)):
                multi_idx = tuple(idx)
            else:
                # Convert flat index to multi-index
                multi_idx = self.grid.get_multi_index(int(idx))

            laplacian = 0.0

            # Compute second derivative in each dimension
            for d in range(self.dimension):
                # Check if we're at a boundary in this dimension
                at_lower_bound = multi_idx[d] == 0
                at_upper_bound = multi_idx[d] == self._grid_shape[d] - 1

                # Create index tuples for neighbors
                idx_center = list(multi_idx)
                idx_plus = list(multi_idx)
                idx_minus = list(multi_idx)

                if at_lower_bound or at_upper_bound:
                    # Boundary: use one-sided difference (assume Neumann BC)
                    if at_lower_bound:
                        idx_plus[d] = multi_idx[d] + 1
                        u_center = U_shaped[tuple(idx_center)]
                        u_plus = U_shaped[tuple(idx_plus)]
                        # One-sided: (u_plus - u_center) / dx²
                        second_deriv = (u_plus - u_center) / self.spacing[d] ** 2
                    else:  # at_upper_bound
                        idx_minus[d] = multi_idx[d] - 1
                        u_center = U_shaped[tuple(idx_center)]
                        u_minus = U_shaped[tuple(idx_minus)]
                        # One-sided: (u_center - u_minus) / dx²
                        second_deriv = (u_center - u_minus) / self.spacing[d] ** 2

                else:
                    # Interior: central difference
                    idx_plus[d] = multi_idx[d] + 1
                    idx_minus[d] = multi_idx[d] - 1

                    u_center = U_shaped[tuple(idx_center)]
                    u_plus = U_shaped[tuple(idx_plus)]
                    u_minus = U_shaped[tuple(idx_minus)]

                    # Central: (u_plus - 2*u_center + u_minus) / dx²
                    second_deriv = (u_plus - 2 * u_center + u_minus) / self.spacing[d] ** 2

                laplacian += second_deriv

            return float(laplacian)

    def _evaluate_hamiltonian(self, x: np.ndarray | float, p: np.ndarray | float, m: float, time_idx: int) -> float:
        """
        Evaluate Hamiltonian H(x, p, m) at given point (supports 1D and nD).

        Args:
            x: Spatial position
                - 1D: scalar float
                - nD: array of shape (dimension,)
            p: Control/momentum value
                - 1D: scalar float
                - nD: array of shape (dimension,)
            m: Density value
            time_idx: Time index

        Returns:
            Hamiltonian value
        """
        if self.dimension == 1:
            # 1D Hamiltonian evaluation
            x_scalar = float(x) if np.ndim(x) > 0 else x
            p_scalar = float(p) if np.ndim(p) > 0 else p

            try:
                # Try problem.hamiltonian() first (nD MFGProblem)
                if hasattr(self.problem, "hamiltonian"):
                    t_value = time_idx * self.problem.T / self.problem.Nt if time_idx is not None else 0.0
                    return self.problem.hamiltonian(x_scalar, m, p_scalar, t_value)
                # Fall back to problem.H() for legacy MFGProblem
                elif hasattr(self.problem, "H"):
                    x_idx = int((x_scalar - self.problem.xmin) / self.dx)
                    x_idx = np.clip(x_idx, 0, self.problem.Nx)
                    derivs = {(0,): 0.0, (1,): p_scalar}
                    return self.problem.H(x_idx, m, derivs=derivs, t_idx=time_idx)
                else:
                    # Fallback: standard quadratic Hamiltonian
                    return 0.5 * p_scalar**2 + getattr(self.problem, "coupling_coefficient", 0.5) * m
            except Exception as e:
                logger.debug(f"1D Hamiltonian evaluation failed: {e}, using fallback")
                return 0.5 * p_scalar**2 + getattr(self.problem, "coupling_coefficient", 0.5) * m

        else:
            # nD Hamiltonian evaluation
            p_vec = np.atleast_1d(p)

            try:
                # Call problem's Hamiltonian function if available
                if hasattr(self.problem, "hamiltonian"):
                    x_vec = np.atleast_1d(x)
                    t_value = time_idx * self.problem.T / self.problem.Nt if time_idx is not None else 0.0
                    return self.problem.hamiltonian(x_vec, m, p_vec, t_value)
                else:
                    # Fallback: standard quadratic Hamiltonian H = |p|²/2 + C*m
                    p_norm_sq = np.sum(p_vec**2)
                    coef_CT = getattr(self.problem, "coupling_coefficient", 0.5)
                    return 0.5 * p_norm_sq + coef_CT * m

            except Exception as e:
                logger.debug(f"nD Hamiltonian evaluation failed: {e}, using fallback")
                p_norm_sq = np.sum(p_vec**2)
                return 0.5 * p_norm_sq + getattr(self.problem, "coupling_coefficient", 0.5) * m

    def _solve_crank_nicolson_diffusion(self, U_star: np.ndarray, dt: float, sigma: float) -> np.ndarray:
        """
        Solve diffusion step using Crank-Nicolson (unconditionally stable).

        Delegates to hjb_sl_adi.solve_crank_nicolson_diffusion_1d.

        Args:
            U_star: Intermediate solution after advection step
            dt: Time step size
            sigma: Diffusion coefficient

        Returns:
            Solution after implicit diffusion step
        """
        return solve_crank_nicolson_diffusion_1d(U_star, dt, sigma, self.x_grid)

    def _adi_diffusion_step(self, U_star: np.ndarray, dt: float) -> np.ndarray:
        """
        Apply ADI (Alternating Direction Implicit) diffusion for nD grids.

        Delegates to hjb_sl_adi.adi_diffusion_step.

        Args:
            U_star: Intermediate solution after advection step, shape (N1, N2, ..., Nd)
            dt: Time step size

        Returns:
            Solution after ADI diffusion step, same shape as U_star
        """
        if self.dimension == 1:
            # For 1D, use standard Crank-Nicolson
            return self._solve_crank_nicolson_diffusion(U_star, dt, self.problem.sigma)

        return adi_diffusion_step(
            U_star,
            dt,
            self.problem.sigma,
            self.spacing,
            tuple(self._grid_shape),
        )

    def get_solver_info(self) -> dict[str, Any]:
        """Return solver configuration information."""
        return {
            "method": "Semi-Lagrangian",
            "interpolation": self.interpolation_method,
            "optimization": self.optimization_method,
            "characteristic_solver": self.characteristic_solver,
            "use_jax": self.use_jax,
            "tolerance": self.tolerance,
            "max_iterations": self.max_char_iterations,
            "adaptive_substepping": self.enable_adaptive_substepping,
            "max_substeps": self.max_substeps,
            "cfl_target": self.cfl_target,
        }


if __name__ == "__main__":
    """Quick smoke test for development."""
    print("Testing HJBSemiLagrangianSolver...")

    from mfg_pde import MFGProblem

    # Test 1: Solver initialization
    print("\n1. Testing solver initialization...")
    problem = MFGProblem(Nx=50, Nt=100, T=1.0, sigma=0.1)
    solver = HJBSemiLagrangianSolver(problem, interpolation_method="linear", optimization_method="brent")

    assert solver.dimension == 1
    assert solver.hjb_method_name == "Semi-Lagrangian"
    assert solver.interpolation_method == "linear"
    print("   1D solver initialization: OK")

    # Test 2: 1D Crank-Nicolson diffusion (used by 1D solver)
    print("\n2. Testing 1D Crank-Nicolson diffusion...")
    # Create a smooth test function (Gaussian)
    x = np.linspace(0, 1, 51)
    U_test = np.exp(-50 * (x - 0.5) ** 2)

    # Apply diffusion for one timestep
    dt = 0.01
    sigma = 0.1
    U_diffused = solver._solve_crank_nicolson_diffusion(U_test, dt, sigma)

    assert U_diffused.shape == U_test.shape
    assert not np.any(np.isnan(U_diffused))
    assert not np.any(np.isinf(U_diffused))
    # Diffusion should smooth the peak
    assert U_diffused.max() < U_test.max()
    print(f"   Peak before diffusion: {U_test.max():.4f}")
    print(f"   Peak after diffusion: {U_diffused.max():.4f}")
    print("   1D Crank-Nicolson: OK")

    # Test 3: 2D solver initialization with ADI compatibility check
    print("\n3. Testing 2D solver with ADI...")

    problem_2d = MFGProblem(
        spatial_bounds=[(0.0, 1.0), (0.0, 1.0)],
        spatial_discretization=[20, 20],
        T=0.5,
        Nt=50,
        sigma=0.1,
    )

    solver_2d = HJBSemiLagrangianSolver(problem_2d, interpolation_method="linear")

    assert solver_2d.dimension == 2
    assert hasattr(solver_2d, "_adi_compatible")
    assert solver_2d._adi_compatible  # Scalar sigma should be ADI compatible
    print(f"   ADI compatible: {solver_2d._adi_compatible}")
    print("   2D solver initialization: OK")

    # Test 4: ADI diffusion step directly
    print("\n4. Testing ADI diffusion step...")
    # Create 2D Gaussian test function
    grid_shape = tuple(solver_2d._grid_shape)
    x = np.linspace(0, 1, grid_shape[0])
    y = np.linspace(0, 1, grid_shape[1])
    X, Y = np.meshgrid(x, y, indexing="ij")
    U_2d_test = np.exp(-50 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2))

    # Apply ADI diffusion
    U_2d_diffused = solver_2d._adi_diffusion_step(U_2d_test, dt=0.01)

    assert U_2d_diffused.shape == U_2d_test.shape
    assert not np.any(np.isnan(U_2d_diffused))
    assert not np.any(np.isinf(U_2d_diffused))
    # Diffusion should smooth the peak
    assert U_2d_diffused.max() < U_2d_test.max()
    print(f"   Peak before ADI diffusion: {U_2d_test.max():.4f}")
    print(f"   Peak after ADI diffusion: {U_2d_diffused.max():.4f}")
    print("   ADI diffusion step: OK")

    # Test 5: ADI preserves mass (integral)
    print("\n5. Testing ADI mass conservation...")
    dx_2d = solver_2d.spacing
    mass_before = np.sum(U_2d_test) * dx_2d[0] * dx_2d[1]
    mass_after = np.sum(U_2d_diffused) * dx_2d[0] * dx_2d[1]
    mass_error = abs(mass_after - mass_before) / mass_before
    print(f"   Mass before: {mass_before:.6f}")
    print(f"   Mass after: {mass_after:.6f}")
    print(f"   Relative error: {mass_error:.2e}")
    # With Neumann BC, mass should be approximately conserved
    assert mass_error < 0.05, f"Mass error too large: {mass_error}"
    print("   Mass conservation: OK")

    # Test 6: ADI with anisotropic sigma (diagonal tensor)
    print("\n6. Testing ADI with anisotropic diffusion...")
    problem_2d_aniso = MFGProblem(
        spatial_bounds=[(0.0, 1.0), (0.0, 1.0)],
        spatial_discretization=[20, 20],
        T=0.5,
        Nt=50,
        sigma=np.array([0.15, 0.05]),  # Different sigma in x and y
    )
    solver_2d_aniso = HJBSemiLagrangianSolver(problem_2d_aniso)
    assert solver_2d_aniso._adi_compatible  # Diagonal is ADI compatible
    print(f"   Anisotropic ADI compatible: {solver_2d_aniso._adi_compatible}")
    print("   Anisotropic sigma: OK")

    print("\nAll smoke tests passed!")
