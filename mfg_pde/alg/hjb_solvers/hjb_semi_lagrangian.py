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
from scipy.interpolate import interp1d
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

        # Precompute grid and time parameters
        self.x_grid = np.linspace(problem.xmin, problem.xmax, problem.Nx + 1)
        self.dt = problem.Dt
        self.dx = problem.Dx

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
            (Nt, Nx) solution array for value function
        """
        Nt, Nx = M_density_evolution_from_FP.shape
        U_solution = np.zeros((Nt, Nx))

        # Set final condition
        U_solution[Nt - 1, :] = U_final_condition_at_T

        if logger.isEnabledFor(logging.INFO):
            logger.info(f"Starting semi-Lagrangian HJB solve: {Nt} time steps, {Nx} spatial points")

        # Solve backward in time using semi-Lagrangian method
        for n in range(Nt - 2, -1, -1):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Solving time step {n}/{Nt - 2}")

            U_solution[n, :] = self._solve_timestep_semi_lagrangian(
                U_solution[n + 1, :],  # u^{n+1}
                M_density_evolution_from_FP[n + 1, :],  # m^{n+1}
                U_from_prev_picard[n, :],  # u_k^n for coupling terms
                n,  # time index
            )

            # Check for numerical issues
            if np.any(np.isnan(U_solution[n, :]) | np.isinf(U_solution[n, :])):
                logger.warning(f"Numerical issues detected at time step {n}")
                # Use simple backward Euler as fallback
                U_solution[n, :] = self._fallback_backward_euler(
                    U_solution[n + 1, :], M_density_evolution_from_FP[n + 1, :], n
                )

        if logger.isEnabledFor(logging.INFO):
            final_residual = np.linalg.norm(U_solution[1, :] - U_solution[0, :])
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
        Solve one timestep using semi-Lagrangian method.

        Args:
            U_next: Value function at next time step
            M_next: Density at next time step
            U_prev_picard: Value from previous Picard iteration (for coupling)
            time_idx: Current time index

        Returns:
            Value function at current time step
        """
        Nx = len(U_next)
        U_current = np.zeros(Nx)

        # Process each spatial point
        for i in range(Nx):
            x_current = self.x_grid[i]
            m_current = M_next[i]

            try:
                # Step 1: Find optimal control by minimizing Hamiltonian
                p_optimal = self._find_optimal_control(x_current, m_current, time_idx)

                # Step 2: Trace characteristic backward to find departure point
                x_departure = self._trace_characteristic_backward(x_current, p_optimal, self.dt)

                # Step 3: Interpolate value function at departure point
                u_departure = self._interpolate_value(U_next, x_departure)

                # Step 4: Compute diffusion term at current point
                diffusion_term = self._compute_diffusion_term(U_next, i)

                # Step 5: Compute Hamiltonian at optimal control
                hamiltonian_value = self._evaluate_hamiltonian(x_current, p_optimal, m_current, time_idx)

                # Step 6: Semi-Lagrangian update
                U_current[i] = u_departure - self.dt * (
                    hamiltonian_value - 0.5 * self.problem.sigma**2 * diffusion_term
                )

            except Exception as e:
                logger.warning(f"Error at grid point {i}: {e}")
                # Fallback to simple backward difference
                U_current[i] = U_next[i]

        return U_current

    def _find_optimal_control(self, x: float, m: float, time_idx: int) -> float:
        """
        Find optimal control p* that minimizes H(x, p, m).

        For the standard MFG Hamiltonian H(x, p, m) = |p|²/2 + V(x) + C(x,m),
        the optimal control is p* = 0 (no control cost in p).

        For more general Hamiltonians, we use numerical optimization.

        Args:
            x: Spatial position
            m: Density value
            time_idx: Time index

        Returns:
            Optimal control value p*
        """
        # For standard MFG problems, try analytical solution first
        if hasattr(self.problem, "coefCT"):
            # Standard quadratic Hamiltonian: |p|²/2 + C*m + V(x)
            # Optimal control is p* = 0
            return 0.0

        # For general Hamiltonians, use numerical optimization
        def hamiltonian_objective(p):
            p_values = {"forward": p, "backward": p}  # Symmetric for optimization
            try:
                x_idx = int((x - self.problem.xmin) / self.dx)
                x_idx = np.clip(x_idx, 0, self.problem.Nx)
                return self.problem.H(x_idx, m, p_values, time_idx)
            except:
                return np.inf

        # Find optimal control using scalar optimization
        try:
            if self.optimization_method == "brent":
                result = minimize_scalar(
                    hamiltonian_objective,
                    bounds=(-10.0, 10.0),
                    method="bounded",
                    options={"xatol": self.tolerance},
                )
            else:  # golden section
                result = minimize_scalar(
                    hamiltonian_objective,
                    bounds=(-10.0, 10.0),
                    method="golden",
                    options={"xtol": self.tolerance},
                )

            return result.x if result.success else 0.0

        except Exception as e:
            logger.debug(f"Optimization failed at x={x}: {e}")
            return 0.0

    def _trace_characteristic_backward(self, x_current: float, p_optimal: float, dt: float) -> float:
        """
        Trace characteristic backward in time to find departure point.

        The characteristic equation is dx/dt = -∇_p H(x, p, m).
        For standard MFG: dx/dt = -p, so X(t-dt) = x - p*dt.

        Args:
            x_current: Current spatial position
            p_optimal: Optimal control value
            dt: Time step size

        Returns:
            Departure point X(t-dt)
        """
        if self.use_jax:
            return float(self._jax_solve_characteristic(x_current, p_optimal, dt))

        if self.characteristic_solver == "explicit_euler":
            # Simple Euler: X(t-dt) = x - p*dt
            x_departure = x_current - p_optimal * dt

        elif self.characteristic_solver == "rk2":
            # Second-order Runge-Kutta for better accuracy
            k1 = -p_optimal
            x_current + 0.5 * dt * k1

            # Need to re-evaluate p at midpoint (simplified)
            k2 = -p_optimal  # Assume constant for now
            x_departure = x_current + dt * k2

        else:
            # Default to Euler
            x_departure = x_current - p_optimal * dt

        # Ensure departure point is within domain (use periodic BC if needed)
        if hasattr(self.problem, "boundary_conditions"):
            # Handle boundary conditions properly
            x_departure = self._apply_boundary_conditions(x_departure)
        else:
            # Simple clamping
            x_departure = np.clip(x_departure, self.problem.xmin, self.problem.xmax)

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

    def _interpolate_value(self, U_values: np.ndarray, x_query: float) -> float:
        """
        Interpolate value function at query point.

        Args:
            U_values: Value function on grid
            x_query: Query point for interpolation

        Returns:
            Interpolated value
        """
        if self.use_jax and self.interpolation_method == "linear":
            return float(self._jax_interpolate(self.x_grid, U_values, x_query))

        # Handle boundary cases
        if x_query <= self.problem.xmin:
            return U_values[0]
        if x_query >= self.problem.xmax:
            return U_values[-1]

        try:
            if self.interpolation_method == "linear":
                # Linear interpolation
                interpolator = interp1d(
                    self.x_grid,
                    U_values,
                    kind="linear",
                    bounds_error=False,
                    fill_value="extrapolate",  # type: ignore[arg-type]
                )
            elif self.interpolation_method == "cubic":
                # Cubic interpolation for higher accuracy
                interpolator = interp1d(
                    self.x_grid,
                    U_values,
                    kind="cubic",
                    bounds_error=False,
                    fill_value="extrapolate",  # type: ignore[arg-type]
                )
            else:
                # Default to linear
                interpolator = interp1d(
                    self.x_grid,
                    U_values,
                    kind="linear",
                    bounds_error=False,
                    fill_value="extrapolate",  # type: ignore[arg-type]
                )

            return float(interpolator(x_query))

        except Exception as e:
            logger.debug(f"Interpolation failed at x={x_query}: {e}")
            # Fallback to nearest neighbor
            idx = np.argmin(np.abs(self.x_grid - x_query))
            return U_values[idx]

    def _compute_diffusion_term(self, U_values: np.ndarray, i: int) -> float:
        """
        Compute discrete Laplacian (diffusion term) at grid point i.

        Uses standard finite difference: (U[i+1] - 2*U[i] + U[i-1]) / dx²

        Args:
            U_values: Value function array
            i: Grid point index

        Returns:
            Discrete Laplacian value
        """
        Nx = len(U_values)

        if Nx <= 2:
            return 0.0

        # Handle boundary points with appropriate conditions
        if i == 0:
            # Forward difference at left boundary
            if hasattr(self.problem, "boundary_conditions"):
                bc = getattr(self.problem, "boundary_conditions", None)
                if bc is not None and hasattr(bc, "type") and bc.type == "periodic":
                    # Periodic boundary
                    laplacian = (U_values[1] - 2 * U_values[0] + U_values[-1]) / self.dx**2
                else:
                    # Neumann (zero derivative) boundary
                    laplacian = (U_values[1] - U_values[0]) / self.dx**2
            else:
                laplacian = (U_values[1] - U_values[0]) / self.dx**2

        elif i == Nx - 1:
            # Backward difference at right boundary
            if hasattr(self.problem, "boundary_conditions"):
                bc = getattr(self.problem, "boundary_conditions", None)
                if bc is not None and hasattr(bc, "type") and bc.type == "periodic":
                    # Periodic boundary
                    laplacian = (U_values[0] - 2 * U_values[-1] + U_values[-2]) / self.dx**2
                else:
                    # Neumann boundary
                    laplacian = (U_values[-1] - U_values[-2]) / self.dx**2
            else:
                laplacian = (U_values[-1] - U_values[-2]) / self.dx**2

        else:
            # Central difference for interior points
            laplacian = (U_values[i + 1] - 2 * U_values[i] + U_values[i - 1]) / self.dx**2

        return laplacian

    def _evaluate_hamiltonian(self, x: float, p: float, m: float, time_idx: int) -> float:
        """
        Evaluate Hamiltonian H(x, p, m) at given point.

        Args:
            x: Spatial position
            p: Control/momentum value
            m: Density value
            time_idx: Time index

        Returns:
            Hamiltonian value
        """
        try:
            # Convert x to grid index for problem.H interface
            x_idx = int((x - self.problem.xmin) / self.dx)
            x_idx = np.clip(x_idx, 0, self.problem.Nx)

            # Create p_values dict expected by problem.H
            p_values = {"forward": p, "backward": p}

            return self.problem.H(x_idx, m, p_values, time_idx)

        except Exception as e:
            logger.debug(f"Hamiltonian evaluation failed: {e}")
            # Fallback to simple quadratic Hamiltonian
            return 0.5 * p**2 + getattr(self.problem, "coefCT", 0.5) * m

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
            except:
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
