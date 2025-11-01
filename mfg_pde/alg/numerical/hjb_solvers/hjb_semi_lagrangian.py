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
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.optimize import minimize_scalar

from .base_hjb import BaseHJBSolver

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
    - nD (2D/3D): Partial support (interpolation and diffusion complete)
      - Interpolation: RegularGridInterpolator (complete)
      - Diffusion: nD Laplacian (complete)
      - Characteristic tracing: Needs vector form
      - Optimal control: Needs vector optimization
    """

    def __init__(
        self,
        problem: MFGProblem,
        interpolation_method: str = "linear",
        optimization_method: str = "brent",
        characteristic_solver: str = "explicit_euler",
        use_jax: bool | None = None,
        tolerance: float = 1e-8,
        max_char_iterations: int = 100,
    ):
        """
        Initialize semi-Lagrangian HJB solver.

        Args:
            problem: MFG problem instance
            interpolation_method: Method for interpolating values ('linear', 'cubic')
            optimization_method: Method for Hamiltonian optimization ('brent', 'golden')
            characteristic_solver: Method for solving characteristics ('explicit_euler', 'rk2')
            use_jax: Whether to use JAX acceleration (auto-detect if None)
            tolerance: Convergence tolerance for optimization
            max_char_iterations: Maximum iterations for characteristic solving
        """
        super().__init__(problem)
        self.hjb_method_name = "Semi-Lagrangian"

        # Solver configuration
        self.interpolation_method = interpolation_method
        self.optimization_method = optimization_method
        self.characteristic_solver = characteristic_solver
        self.tolerance = tolerance
        self.max_char_iterations = max_char_iterations

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
            self.dt = problem.Dt
            self.dx = problem.Dx
            self.grid = None  # No TensorProductGrid for 1D
        else:
            # nD problem: Use TensorProductGrid
            if not hasattr(problem, "geometry") or not hasattr(problem.geometry, "grid"):
                raise ValueError(
                    f"Multi-dimensional problem must have geometry.grid (TensorProductGrid). "
                    f"Got dimension={self.dimension}"
                )
            self.grid = problem.geometry.grid
            self.dt = problem.dt
            # Grid spacing: vector of spacings in each dimension
            self.spacing = np.array(self.grid.spacing)
            self.x_grid = None  # Not used for nD

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
            problem: MFGProblem or GridBasedMFGProblem instance

        Returns:
            dimension: 1 for 1D problems, 2 for 2D, 3 for 3D, etc.

        Raises:
            ValueError: If dimension cannot be determined
        """
        # Check if it's a GridBasedMFGProblem with explicit dimension
        if hasattr(problem, "geometry") and hasattr(problem.geometry, "grid"):
            if hasattr(problem.geometry.grid, "dimension"):
                return problem.geometry.grid.dimension

        # Check for 1D MFGProblem (has Nx but not Ny)
        if hasattr(problem, "Nx") and not hasattr(problem, "Ny"):
            return 1

        # Check for explicit dimension attribute
        if hasattr(problem, "dimension"):
            return problem.dimension

        # If we can't determine dimension, raise error
        raise ValueError(
            "Cannot determine problem dimension. Problem must be either 1D MFGProblem or GridBasedMFGProblem."
        )

    def solve_hjb_system(
        self,
        M_density_evolution_from_FP: np.ndarray,
        U_final_condition_at_T: np.ndarray,
        U_from_prev_picard: np.ndarray,
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
            M_density_evolution_from_FP: (Nt, Nx) density evolution from FP equation
            U_final_condition_at_T: (Nx,) final condition u(T, x)
            U_from_prev_picard: (Nt, Nx) previous Picard iteration for coupling terms

        Returns:
            (Nt, *grid_shape) solution array for value function
        """
        # Handle multi-dimensional grids
        shape = M_density_evolution_from_FP.shape
        Nt = shape[0]
        grid_shape = shape[1:]  # Remaining dimensions

        U_solution = np.zeros_like(M_density_evolution_from_FP)

        # Set final condition
        U_solution[Nt - 1] = U_final_condition_at_T

        total_points = np.prod(grid_shape)
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                f"Starting semi-Lagrangian HJB solve: {Nt} time steps, {total_points} spatial points ({grid_shape})"
            )

        # Solve backward in time using semi-Lagrangian method
        for n in range(Nt - 2, -1, -1):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Solving time step {n}/{Nt - 2}")

            U_solution[n] = self._solve_timestep_semi_lagrangian(
                U_solution[n + 1],  # u^{n+1}
                M_density_evolution_from_FP[n + 1],  # m^{n+1}
                U_from_prev_picard[n],  # u_k^n for coupling terms
                n,  # time index
            )

            # Check for numerical issues
            if np.any(np.isnan(U_solution[n]) | np.isinf(U_solution[n])):
                logger.warning(f"Numerical issues detected at time step {n}")
                # Use simple backward Euler as fallback
                U_solution[n] = self._fallback_backward_euler(U_solution[n + 1], M_density_evolution_from_FP[n + 1], n)

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
            # 1D solve: Use existing logic
            Nx = len(U_next)
            U_current = np.zeros(Nx)

            for i in range(Nx):
                x_current = self.x_grid[i]
                m_current = M_next[i]

                try:
                    p_optimal = self._find_optimal_control(x_current, m_current, time_idx)
                    x_departure = self._trace_characteristic_backward(x_current, p_optimal, self.dt)
                    u_departure = self._interpolate_value(U_next, x_departure)
                    diffusion_term = self._compute_diffusion_term(U_next, i)
                    hamiltonian_value = self._evaluate_hamiltonian(x_current, p_optimal, m_current, time_idx)

                    U_current[i] = u_departure - self.dt * (
                        hamiltonian_value - 0.5 * self.problem.sigma**2 * diffusion_term
                    )

                except Exception as e:
                    logger.warning(f"Error at grid point {i}: {e}")
                    U_current[i] = U_next[i]

            return U_current

        else:
            # nD solve: Iterate over all grid points
            # Reshape arrays to grid shape for easier indexing
            if U_next.ndim == 1:
                # Infer grid shape from array size (handles both full grid and interior points)
                total_points = U_next.size
                expected_full = int(np.prod(self.grid.num_points))

                if total_points == expected_full:
                    grid_shape = tuple(self.grid.num_points)
                else:
                    # Interior points only (num_points - 1 in each dimension)
                    grid_shape = tuple(n - 1 for n in self.grid.num_points)

                U_next_shaped = U_next.reshape(grid_shape)
                M_next_shaped = M_next.reshape(grid_shape)
            else:
                U_next_shaped = U_next
                M_next_shaped = M_next
                grid_shape = U_next_shaped.shape

            U_current_shaped = np.zeros_like(U_next_shaped)

            # Iterate over all grid points using the actual array shape
            for multi_idx in np.ndindex(grid_shape):
                try:
                    # Get spatial coordinates for this grid point
                    x_current = np.array([self.grid.coordinates[d][multi_idx[d]] for d in range(self.dimension)])
                    m_current = M_next_shaped[multi_idx]

                    # Step 1: Find optimal control (returns vector for nD)
                    p_optimal = self._find_optimal_control(x_current, m_current, time_idx)

                    # Step 2: Trace characteristic backward (vector operation)
                    x_departure = self._trace_characteristic_backward(x_current, p_optimal, self.dt)

                    # Step 3: Interpolate at departure point
                    u_departure = self._interpolate_value(U_next_shaped, x_departure)

                    # Step 4: Compute diffusion term (nD Laplacian)
                    diffusion_term = self._compute_diffusion_term(U_next_shaped, multi_idx)

                    # Step 5: Evaluate Hamiltonian
                    hamiltonian_value = self._evaluate_hamiltonian(x_current, p_optimal, m_current, time_idx)

                    # Step 6: Semi-Lagrangian update
                    U_current_shaped[multi_idx] = u_departure - self.dt * (
                        hamiltonian_value - 0.5 * self.problem.sigma**2 * diffusion_term
                    )

                except Exception as e:
                    logger.warning(f"Error at grid point {multi_idx}: {e}")
                    U_current_shaped[multi_idx] = U_next_shaped[multi_idx]

            # Return flattened if input was flattened
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
        - 1D: Uses scalar optimization (minimize_scalar)
        - nD: Currently returns zero vector (TODO: implement vector optimization)

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
            if hasattr(self.problem, "coefCT"):
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
            if hasattr(self.problem, "coefCT"):
                return np.zeros(self.dimension)

            # TODO: Implement vector optimization using scipy.optimize.minimize
            # For now, return zero vector (works for standard MFG)
            logger.debug(
                "nD optimal control optimization not yet implemented. "
                "Using p* = 0 (valid for standard quadratic Hamiltonians)."
            )
            return np.zeros(self.dimension)

    def _trace_characteristic_backward(
        self, x_current: np.ndarray | float, p_optimal: np.ndarray | float, dt: float
    ) -> np.ndarray | float:
        """
        Trace characteristic backward in time to find departure point (supports 1D and nD).

        The characteristic equation is dx/dt = -∇_p H(x, p, m).
        For standard MFG: dx/dt = -p, so X(t-dt) = x - p*dt.

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
            x_scalar = float(x_current) if np.ndim(x_current) > 0 else x_current
            p_scalar = float(p_optimal) if np.ndim(p_optimal) > 0 else p_optimal

            if self.use_jax:
                return float(self._jax_solve_characteristic(x_scalar, p_scalar, dt))

            if self.characteristic_solver == "explicit_euler":
                x_departure = x_scalar - p_scalar * dt
            elif self.characteristic_solver == "rk2":
                k1 = -p_scalar
                x_scalar + 0.5 * dt * k1
                k2 = -p_scalar
                x_departure = x_scalar + dt * k2
            else:
                x_departure = x_scalar - p_scalar * dt

            # Apply boundary conditions
            if hasattr(self.problem, "boundary_conditions"):
                x_departure = self._apply_boundary_conditions(x_departure)
            else:
                x_departure = np.clip(x_departure, self.problem.xmin, self.problem.xmax)

            return x_departure

        else:
            # nD characteristic tracing: X(t-dt) = X(t) - P*dt
            x_vec = np.atleast_1d(x_current)
            p_vec = np.atleast_1d(p_optimal)

            if len(x_vec) != self.dimension or len(p_vec) != self.dimension:
                raise ValueError(
                    f"x_current and p_optimal must have {self.dimension} components, got {len(x_vec)} and {len(p_vec)}"
                )

            if self.characteristic_solver == "explicit_euler":
                # Vector Euler: X(t-dt) = X(t) - P*dt
                x_departure = x_vec - p_vec * dt
            elif self.characteristic_solver == "rk2":
                # Vector RK2 (simplified: assume constant p)
                k1 = -p_vec
                k2 = -p_vec
                x_departure = x_vec + dt * k2
            else:
                x_departure = x_vec - p_vec * dt

            # Apply boundary conditions (clamp to domain bounds)
            for d in range(self.dimension):
                x_departure[d] = np.clip(x_departure[d], self.grid.bounds[d][0], self.grid.bounds[d][1])

            return x_departure

    def _apply_boundary_conditions(self, x: float) -> float:
        """Apply boundary conditions to ensure x is in valid domain."""
        if hasattr(self.problem, "boundary_conditions"):
            bc = getattr(self.problem, "boundary_conditions", None)
            if bc is None:
                return x
            if hasattr(bc, "type") and bc.type == "periodic":
                # Periodic boundary conditions
                length = self.problem.xmax - self.problem.xmin
                while x < self.problem.xmin:
                    x += length
                while x > self.problem.xmax:
                    x -= length
                return x

        # Default: clamp to domain
        return np.clip(x, self.problem.xmin, self.problem.xmax)

    def _interpolate_value(self, U_values: np.ndarray, x_query: np.ndarray | float) -> float:
        """
        Interpolate value function at query point (supports 1D and nD).

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
            # 1D interpolation: Use existing logic
            x_query_scalar = float(x_query) if np.ndim(x_query) > 0 else x_query

            if self.use_jax and self.interpolation_method == "linear":
                return float(self._jax_interpolate(self.x_grid, U_values, x_query_scalar))

            # Handle boundary cases
            if x_query_scalar <= self.problem.xmin:
                return U_values[0]
            if x_query_scalar >= self.problem.xmax:
                return U_values[-1]

            try:
                if self.interpolation_method == "linear":
                    interpolator = interp1d(
                        self.x_grid,
                        U_values,
                        kind="linear",
                        bounds_error=False,
                        fill_value="extrapolate",
                    )
                elif self.interpolation_method == "cubic":
                    interpolator = interp1d(
                        self.x_grid,
                        U_values,
                        kind="cubic",
                        bounds_error=False,
                        fill_value="extrapolate",
                    )
                else:
                    interpolator = interp1d(
                        self.x_grid,
                        U_values,
                        kind="linear",
                        bounds_error=False,
                        fill_value="extrapolate",
                    )

                return float(interpolator(x_query_scalar))

            except Exception as e:
                logger.debug(f"Interpolation failed at x={x_query_scalar}: {e}")
                idx = np.argmin(np.abs(self.x_grid - x_query_scalar))
                return U_values[idx]

        else:
            # nD interpolation: Use RegularGridInterpolator
            try:
                # Ensure query point is the right shape
                x_query_vec = np.atleast_1d(x_query)
                if len(x_query_vec) != self.dimension:
                    raise ValueError(f"Query point must have {self.dimension} coordinates, got {len(x_query_vec)}")

                # Create nD interpolator
                # TensorProductGrid.coordinates returns list of 1D grids for each dimension
                grid_axes = tuple(self.grid.coordinates)  # (grid_x, grid_y) for 2D

                # Reshape U_values to grid shape if needed
                if U_values.ndim == 1:
                    U_values_reshaped = U_values.reshape(self.grid.num_points)
                else:
                    U_values_reshaped = U_values

                interpolator = RegularGridInterpolator(
                    grid_axes,
                    U_values_reshaped,
                    method="linear",
                    bounds_error=False,
                    fill_value=None,  # Extrapolate using nearest
                )

                # Query at point (must be shape (1, dimension) for single point)
                result = interpolator(x_query_vec.reshape(1, -1))
                return float(result[0])

            except Exception as e:
                logger.debug(f"nD interpolation failed at x={x_query}: {e}")
                # Fallback to nearest neighbor
                # Find closest grid point in each dimension
                multi_idx = []
                for d in range(self.dimension):
                    # Find nearest index in this dimension
                    distances = np.abs(self.grid.coordinates[d] - x_query_vec[d])
                    nearest_idx = np.argmin(distances)
                    multi_idx.append(nearest_idx)

                # Convert to flat index and return value
                if U_values.ndim == 1:
                    flat_idx = self.grid.get_index(multi_idx)
                    return U_values[flat_idx]
                else:
                    return U_values[tuple(multi_idx)]

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
                U_shaped = U_values.reshape(self.grid.num_points)
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
                at_upper_bound = multi_idx[d] == self.grid.num_points[d] - 1

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
                x_idx = int((x_scalar - self.problem.xmin) / self.dx)
                x_idx = np.clip(x_idx, 0, self.problem.Nx)
                derivs = {(0,): 0.0, (1,): p_scalar}
                return self.problem.H(x_idx, m, derivs=derivs, t_idx=time_idx)
            except Exception as e:
                logger.debug(f"Hamiltonian evaluation failed: {e}")
                return 0.5 * p_scalar**2 + getattr(self.problem, "coefCT", 0.5) * m

        else:
            # nD Hamiltonian evaluation
            p_vec = np.atleast_1d(p)

            try:
                # For standard quadratic Hamiltonian: H = |p|²/2 + C*m + V(x)
                # This is the most common case for MFG problems
                p_norm_sq = np.sum(p_vec**2)
                coef_CT = getattr(self.problem, "coefCT", 0.5)

                # Terminal cost V(x) - would need problem-specific evaluation
                # For now, assume V(x) is incorporated elsewhere
                return 0.5 * p_norm_sq + coef_CT * m

            except Exception as e:
                logger.debug(f"nD Hamiltonian evaluation failed: {e}")
                p_norm_sq = np.sum(p_vec**2)
                return 0.5 * p_norm_sq + getattr(self.problem, "coefCT", 0.5) * m

    def _fallback_backward_euler(self, U_next: np.ndarray, M_next: np.ndarray, time_idx: int) -> np.ndarray:
        """
        Fallback solver using simple backward Euler when semi-Lagrangian fails.

        Args:
            U_next: Value at next time step
            M_next: Density at next time step
            time_idx: Time index

        Returns:
            Value at current time step using backward Euler
        """
        Nx = len(U_next)
        U_current = np.zeros(Nx)

        for i in range(Nx):
            # Simple backward Euler: u^n = u^{n+1} - dt * H(x, 0, m)
            try:
                hamiltonian = self._evaluate_hamiltonian(self.x_grid[i], 0.0, M_next[i], time_idx)
                diffusion = self._compute_diffusion_term(U_next, i)

                U_current[i] = U_next[i] - self.dt * (hamiltonian - 0.5 * self.problem.sigma**2 * diffusion)
            except (ValueError, IndexError, FloatingPointError, Exception):
                # Numerical issues or user Hamiltonian errors - keep previous value
                U_current[i] = U_next[i]

        return U_current

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
        }
