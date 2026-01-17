"""
Finite Difference Method (FDM) for HJB Equation - All Dimensions.

Supports:
    - 1D: Optimized Newton solver from base_hjb
    - 2D/3D/nD: Uses centralized nonlinear solvers

References:
    - Evans (2010): Partial Differential Equations, Ch. 10
    - Achdou & Capuzzo-Dolcetta (2010): Mean field games: numerical methods
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from mfg_pde.core.derivatives import DerivativeTensors, to_multi_index_dict
from mfg_pde.geometry.boundary import pad_array_with_ghosts
from mfg_pde.utils.numerical import FixedPointSolver, NewtonSolver
from mfg_pde.utils.pde_coefficients import CoefficientField

from . import base_hjb
from .base_hjb import BaseHJBSolver

# Type alias for HJB advection schemes (gradient form only - HJB is not a conservation law)
HJBAdvectionScheme = Literal["gradient_centered", "gradient_upwind"]


def is_diagonal_tensor(Sigma: NDArray, rtol: float = 1e-10) -> bool:
    """
    Check if tensor is diagonal (off-diagonal elements near zero).

    Args:
        Sigma: Tensor array, either (d, d) or (*shape, d, d)
        rtol: Relative tolerance for off-diagonal elements

    Returns:
        True if diagonal, False otherwise
    """
    # Handle both single tensor and spatially-varying tensors
    if Sigma.ndim == 2:
        # Single (d, d) tensor
        d = Sigma.shape[0]
        off_diag_sum = np.sum(np.abs(Sigma)) - np.sum(np.abs(np.diag(Sigma)))
        diag_sum = np.sum(np.abs(np.diag(Sigma)))
        return off_diag_sum < rtol * diag_sum if diag_sum > 0 else off_diag_sum < rtol
    else:
        # Spatially-varying (*shape, d, d)
        d = Sigma.shape[-1]
        diag_mask = np.eye(d, dtype=bool)
        off_diag_elements = Sigma[..., ~diag_mask]
        diag_elements = Sigma[..., diag_mask]
        off_diag_norm = np.linalg.norm(off_diag_elements)
        diag_norm = np.linalg.norm(diag_elements)
        return off_diag_norm < rtol * diag_norm if diag_norm > 0 else off_diag_norm < rtol


if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mfg_pde.core.mfg_problem import MFGProblem


class HJBFDMSolver(BaseHJBSolver):
    """
    Finite Difference Method solver for HJB equation (all dimensions).

    Automatically handles 1D, 2D, 3D, and higher-dimensional problems:
        - 1D: Uses optimized Newton solver from base_hjb
        - nD: Uses centralized FixedPointSolver or NewtonSolver

    Recommended: d ≤ 3 due to O(N^d) complexity
    """

    # Scheme family trait for duality validation (Issue #580)
    from mfg_pde.alg.base_solver import SchemeFamily

    _scheme_family = SchemeFamily.FDM

    def __init__(
        self,
        problem: MFGProblem,
        solver_type: Literal["fixed_point", "newton"] = "newton",
        advection_scheme: HJBAdvectionScheme = "gradient_upwind",
        damping_factor: float = 1.0,
        max_newton_iterations: int | None = None,
        newton_tolerance: float | None = None,
        bc_mode: Literal["standard", "adjoint_consistent"] = "standard",
        # Deprecated parameters
        NiterNewton: int | None = None,
        l2errBoundNewton: float | None = None,
        backend: str | None = None,
    ):
        """
        Initialize FDM solver.

        Args:
            problem: MFG problem (1D or MFGProblem with spatial_bounds for nD)
            solver_type: 'fixed_point' or 'newton' (nD only, 1D always uses Newton)
            advection_scheme: Discretization scheme for advection term:
                - 'gradient_upwind': Godunov upwind (default, monotone, first-order)
                - 'gradient_centered': Central differences (second-order, may oscillate)
                For MFG coupling, use 'gradient_upwind' with FP 'divergence_upwind'.
            damping_factor: Damping ω ∈ (0,1] for fixed-point (recommend 0.5-0.8)
            max_newton_iterations: Max iterations per timestep
            newton_tolerance: Convergence tolerance
            bc_mode: Boundary condition mode for reflecting boundaries (Issue #574):
                - 'standard': Classical Neumann BC (∂U/∂n = 0)
                - 'adjoint_consistent': Coupled BC (∂U/∂n = -σ²/2 · ∂ln(m)/∂n)
                Use 'adjoint_consistent' for better equilibrium consistency when
                stall point is at domain boundary. Default: 'standard' (backward compat).
            backend: 'numpy', 'torch', or None
        """
        import warnings

        super().__init__(problem)

        # Initialize backend
        from mfg_pde.backends import create_backend

        self.backend = create_backend(backend or "numpy")

        # Validate and store advection scheme
        valid_schemes = {"gradient_centered", "gradient_upwind"}
        if advection_scheme not in valid_schemes:
            raise ValueError(f"Invalid advection_scheme: '{advection_scheme}'. Valid options: {sorted(valid_schemes)}")
        self.advection_scheme = advection_scheme
        self.use_upwind = advection_scheme == "gradient_upwind"

        # Handle deprecated parameters
        if NiterNewton is not None:
            warnings.warn(
                "Parameter 'NiterNewton' is deprecated. Use 'max_newton_iterations' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            max_newton_iterations = max_newton_iterations or NiterNewton

        if l2errBoundNewton is not None:
            warnings.warn(
                "Parameter 'l2errBoundNewton' is deprecated. Use 'newton_tolerance' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            newton_tolerance = newton_tolerance or l2errBoundNewton

        # Set defaults (use None check to avoid treating 0 as falsy)
        self.max_newton_iterations = max_newton_iterations if max_newton_iterations is not None else 30
        self.newton_tolerance = newton_tolerance if newton_tolerance is not None else 1e-6
        self.solver_type = solver_type
        self.damping_factor = damping_factor
        self.bc_mode = bc_mode

        # Validate
        if self.max_newton_iterations < 1:
            raise ValueError(f"max_newton_iterations must be >= 1, got {self.max_newton_iterations}")
        if self.newton_tolerance <= 0:
            raise ValueError(f"newton_tolerance must be > 0, got {self.newton_tolerance}")
        if not 0 < damping_factor <= 1.0:
            raise ValueError(f"damping_factor must be in (0,1], got {damping_factor}")
        if bc_mode not in ("standard", "adjoint_consistent"):
            raise ValueError(f"bc_mode must be 'standard' or 'adjoint_consistent', got '{bc_mode}'")

        # Backward compatibility: Store Newton config
        self._newton_config = {
            "max_iterations": self.max_newton_iterations,
            "tolerance": self.newton_tolerance,
        }

        # Detect dimension
        self.dimension = self._detect_dimension(problem)
        # Backward compatibility: 1D uses "FDM", nD uses "FDM-{d}D-{solver_type}"
        if self.dimension == 1:
            self.hjb_method_name = "FDM"
        else:
            self.hjb_method_name = f"FDM-{self.dimension}D-{solver_type}"

        # For nD, extract grid info and create nonlinear solver
        if self.dimension > 1:
            # Import at runtime to avoid circular dependency
            from mfg_pde.geometry.base import CartesianGrid

            if not isinstance(problem.geometry, CartesianGrid):
                raise ValueError("nD FDM requires problem with CartesianGrid geometry (TensorProductGrid)")

            self.grid = problem.geometry  # Geometry IS the grid
            self.shape = tuple(self.grid.get_grid_shape())
            self.spacing = self.grid.get_grid_spacing()
            self.N_total = int(np.prod(self.shape))
            self.dt = problem.dt

            if self.dimension > 3:
                warnings.warn(
                    f"FDM solver in {self.dimension}D requires {self.N_total:,} grid points. "
                    f"Consider GFDM or sparse methods for d>3.",
                    UserWarning,
                    stacklevel=2,
                )

            # Create nonlinear solver
            if solver_type == "fixed_point":
                self.nonlinear_solver = FixedPointSolver(
                    damping_factor=damping_factor,
                    max_iterations=self.max_newton_iterations,
                    tolerance=self.newton_tolerance,
                )
            else:  # newton
                self.nonlinear_solver = NewtonSolver(
                    max_iterations=self.max_newton_iterations,
                    tolerance=self.newton_tolerance,
                    sparse=True,
                    jacobian=None,  # Use automatic finite differences
                )

            # Create BC applicator for enforcing boundary values (Issue #542)
            from mfg_pde.geometry.boundary.applicator_fdm import FDMApplicator

            self.bc_applicator = FDMApplicator(dimension=self.dimension)

        # Initialize warning flags (Issue #545 - NO hasattr pattern)
        self._bc_warning_emitted: bool = False

    def _detect_dimension(self, problem) -> int:
        """Detect spatial dimension from geometry (unified interface)."""
        # Primary: Use geometry.dimension (standard for all modern problems)
        # Issue #545: Use try/except instead of hasattr
        try:
            return problem.geometry.dimension
        except AttributeError:
            pass

        # Fallback: problem.dimension attribute
        # Note: We prioritize explicit geometry protocol but allow problem.dimension
        # if it's explicitly set (e.g. legacy 1D init sets self.dimension)
        try:
            return problem.dimension
        except AttributeError:
            pass

        raise ValueError(
            "Cannot determine problem dimension. "
            "Ensure problem has 'geometry' with 'dimension' attribute or 'dimension' property."
        )

    def solve_hjb_system(
        self,
        M_density: NDArray | None = None,
        U_terminal: NDArray | None = None,
        U_coupling_prev: NDArray | None = None,
        diffusion_field: float | NDArray | None = None,
        tensor_diffusion_field: NDArray | None = None,
        bc_values: dict[str, float] | None = None,
        # Deprecated parameter names for backward compatibility
        M_density_evolution_from_FP: NDArray | None = None,
        U_final_condition_at_T: NDArray | None = None,
        U_from_prev_picard: NDArray | None = None,
        M_density_evolution: NDArray | None = None,
        U_final_condition: NDArray | None = None,
    ) -> NDArray:
        """
        Solve HJB system backward in time.

        Automatically routes to 1D or nD solver based on dimension.

        Args:
            M_density: Density field from FP solver
            U_terminal: Terminal condition u(T,x)
            U_coupling_prev: Previous coupling iteration estimate
            diffusion_field: Diffusion coefficient (None uses problem.sigma)
            tensor_diffusion_field: Tensor diffusion (Phase 3.0, not yet fully implemented)
            bc_values: Per-boundary BC values for adjoint-consistent mode (Issue #574).
                Dict mapping boundary names to gradient values:
                {"x_min": value_left, "x_max": value_right}
                Only used when bc_mode="adjoint_consistent". Default: None (use standard BC).
        """
        import warnings

        # Handle deprecated parameter names (oldest first)
        if M_density_evolution is not None:
            if M_density is not None or M_density_evolution_from_FP is not None:
                raise ValueError("Cannot specify M_density_evolution with M_density or M_density_evolution_from_FP")
            warnings.warn(
                "Parameter 'M_density_evolution' is deprecated. Use 'M_density' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            M_density = M_density_evolution

        if M_density_evolution_from_FP is not None:
            if M_density is not None:
                raise ValueError("Cannot specify both 'M_density' and deprecated 'M_density_evolution_from_FP'")
            warnings.warn(
                "Parameter 'M_density_evolution_from_FP' is deprecated. Use 'M_density' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            M_density = M_density_evolution_from_FP

        if U_final_condition is not None:
            if U_terminal is not None or U_final_condition_at_T is not None:
                raise ValueError("Cannot specify U_final_condition with U_terminal or U_final_condition_at_T")
            warnings.warn(
                "Parameter 'U_final_condition' is deprecated. Use 'U_terminal' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            U_terminal = U_final_condition

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
        # Validate mutual exclusivity
        if diffusion_field is not None and tensor_diffusion_field is not None:
            raise ValueError(
                "Cannot specify both diffusion_field and tensor_diffusion_field. "
                "Use diffusion_field for scalar or tensor_diffusion_field for anisotropic."
            )

        # Check tensor type and warn if non-diagonal (not fully implemented yet)
        if tensor_diffusion_field is not None:
            import warnings

            # Check if diagonal
            if callable(tensor_diffusion_field):
                # Cannot easily check callable tensors without evaluation
                warnings.warn(
                    "Callable tensor_diffusion_field in HJB solver is not yet fully implemented. "
                    "If the tensor is diagonal, it will be handled correctly. "
                    "Full tensor (non-diagonal) support requires Hamiltonian refactoring.",
                    UserWarning,
                    stacklevel=2,
                )
            elif not is_diagonal_tensor(tensor_diffusion_field):
                warnings.warn(
                    "Full tensor (non-diagonal) diffusion in HJB solver is not yet implemented. "
                    "The parameter is accepted for API compatibility but only diagonal tensors "
                    "are currently supported. Full tensor support requires Hamiltonian refactoring.",
                    UserWarning,
                    stacklevel=2,
                )

        if self.dimension == 1:
            # Extract BC from geometry for Issue #542 fix
            # Issue #527: Use centralized get_boundary_conditions() from BaseMFGSolver
            bc = self.get_boundary_conditions()

            # Extract domain bounds from geometry
            domain_bounds = None
            try:
                bounds = self.problem.geometry.get_bounds()
                # Convert to (1, 2) array for 1D
                domain_bounds = np.array([[bounds[0][0], bounds[1][0]]])
            except AttributeError:
                pass

            # Issue #574: Compute adjoint-consistent BC values if needed
            if self.bc_mode == "adjoint_consistent" and bc_values is None:
                from mfg_pde.geometry.boundary import compute_coupled_hjb_bc_values

                # Get grid spacing
                try:
                    dx = self.problem.geometry.get_grid_spacing()[0]
                except (AttributeError, IndexError):
                    dx = self.problem.dx  # Fallback for legacy API

                # Get diffusion coefficient
                sigma = self.problem.sigma if diffusion_field is None else diffusion_field
                if not isinstance(sigma, (int, float)):
                    raise ValueError(
                        "bc_mode='adjoint_consistent' requires scalar diffusion. "
                        f"Got diffusion_field type: {type(sigma)}"
                    )

                # Compute coupled BC values from current density
                # Use time-averaged density or final time slice
                m_for_bc = M_density[-1, :] if M_density.ndim == 2 else M_density
                bc_values = compute_coupled_hjb_bc_values(
                    m=m_for_bc,
                    dx=dx,
                    sigma=sigma,
                )

            # Debug: Log BC being passed (Issue #542 investigation)
            import logging
            from contextlib import suppress

            logger = logging.getLogger(__name__)
            logger.info(f"[DEBUG Issue #542] BC passed to solve_hjb_system_backward: {bc}")
            # Log segment count if BC has segments attribute (Issue #545: use contextlib.suppress)
            if bc is not None:
                with suppress(AttributeError):
                    logger.info(f"[DEBUG Issue #542] BC has {len(bc.segments)} segments")

            # Use optimized 1D solver with BC-aware computation (Issue #542 fix)
            return base_hjb.solve_hjb_system_backward(
                M_density_from_prev_picard=M_density,
                U_final_condition_at_T=U_terminal,
                U_from_prev_picard=U_coupling_prev,
                problem=self.problem,
                max_newton_iterations=self.max_newton_iterations,
                newton_tolerance=self.newton_tolerance,
                backend=self.backend,
                diffusion_field=diffusion_field,
                use_upwind=self.use_upwind,
                bc=bc,
                domain_bounds=domain_bounds,
                bc_values=bc_values,
            )
        else:
            # Use nD solver with centralized nonlinear solver
            return self._solve_hjb_nd(
                M_density,
                U_terminal,
                U_coupling_prev,
                diffusion_field,
                tensor_diffusion_field,
            )

    def _solve_hjb_nd(
        self,
        M_density: NDArray,
        U_final: NDArray,
        U_prev: NDArray,
        diffusion_field: float | NDArray | None = None,
        tensor_diffusion_field: NDArray | None = None,
    ) -> NDArray:
        """Solve nD HJB using centralized nonlinear solvers with variable diffusion support.

        Supports scalar, array, and callable diffusion coefficients.

        Note: tensor_diffusion_field is accepted but not yet fully implemented.
        """
        # Validate shapes
        # n_time_points = problem.Nt + 1 (number of time knots including t=0 and t=T)
        # problem.Nt = number of time intervals
        n_time_points = self.problem.Nt + 1
        expected_shape = (n_time_points, *self.shape)
        if M_density.shape != expected_shape:
            raise ValueError(f"M_density shape {M_density.shape} != {expected_shape}")
        if U_final.shape != self.shape:
            raise ValueError(f"U_final shape {U_final.shape} != {self.shape}")

        # Allocate solution array with shape (n_time_points, *spatial)
        U_solution = np.zeros(expected_shape, dtype=np.float64)
        # Set terminal condition at t=T (last time index)
        U_solution[n_time_points - 1] = U_final.copy()

        if n_time_points <= 1:
            return U_solution

        # Progress bar for backward time stepping
        from mfg_pde.utils.progress import RichProgressBar

        # Backward time loop: problem.Nt steps from index (n_time_points-2) down to 0
        timestep_range = RichProgressBar(
            range(n_time_points - 2, -1, -1),
            desc=f"HJB {self.dimension}D-FDM ({self.solver_type})",
            unit="step",
        )

        # Backward time loop
        for n in timestep_range:
            U_next = U_solution[n + 1]
            M_next = M_density[n + 1]
            U_guess = U_prev[n]

            # Extract or evaluate diffusion using CoefficientField abstraction
            if tensor_diffusion_field is not None:
                # Evaluate tensor at current timestep
                if callable(tensor_diffusion_field):
                    # Callable: Σ(t, x, m)
                    t = n * self.problem.dt
                    Sigma_at_n = np.zeros((*self.shape, self.dimension, self.dimension))
                    for idx in np.ndindex(self.shape):
                        x_coords = np.array([self.grid.coordinates[d][idx[d]] for d in range(self.dimension)])
                        m_at_point = M_next[idx]
                        Sigma_at_n[idx] = tensor_diffusion_field(t, x_coords, m_at_point)
                else:
                    # Constant or spatially-varying
                    Sigma_at_n = tensor_diffusion_field
            else:
                Sigma_at_n = None

            # Handle scalar diffusion_field
            if diffusion_field is not None or Sigma_at_n is None:
                diffusion = CoefficientField(
                    diffusion_field, self.problem.sigma, "diffusion_field", dimension=self.dimension
                )
                sigma_at_n = diffusion.evaluate_at(
                    timestep_idx=n, grid=self.grid.coordinates, density=M_next, dt=self.problem.dt
                )
            else:
                sigma_at_n = None

            # Compute current time for time-dependent BCs
            t_current = n * self.dt

            # Solve nonlinear system using centralized solver
            U_solution[n] = self._solve_single_timestep(U_next, M_next, U_guess, sigma_at_n, Sigma_at_n, time=t_current)

        return U_solution

    def _solve_single_timestep(
        self,
        U_next: NDArray,
        M_next: NDArray,
        U_guess: NDArray,
        sigma_at_n: float | NDArray | None = None,
        Sigma_at_n: NDArray | None = None,
        time: float = 0.0,
    ) -> NDArray:
        """
        Solve single HJB timestep using centralized nonlinear solver.

        HJB equation: -∂u/∂t + H(∇u, m) - (σ²/2) Δu = 0

        For fixed-point: Solves u = G(u) where G(u) = u_next - dt·(H - (σ²/2)Δu)
        For Newton: Solves F(u) = 0 where F(u) = (u - u_next)/dt + H - (σ²/2)Δu

        Args:
            U_next: Value function at next timestep
            M_next: Density at next timestep
            U_guess: Initial guess for current timestep
            sigma_at_n: Scalar diffusion coefficient at current timestep (None, float, or array)
            Sigma_at_n: Tensor diffusion coefficient at current timestep (None or tensor array)
            time: Current time for time-dependent BC values
        """
        if self.solver_type == "fixed_point":
            # Define fixed-point map G: u → u
            # HJB uses H which includes viscosity term (σ²/2)|∇u|²
            # Fixed-point iteration: u_n = u_{n+1} - dt·H(∇u_n, m)
            def G(U: NDArray) -> NDArray:
                gradients = self._compute_gradients_nd(U, time=time)
                H_values = self._evaluate_hamiltonian_nd(U, M_next, gradients, sigma_at_n, Sigma_at_n)
                return U_next - self.dt * H_values

            U_solution, info = self.nonlinear_solver.solve(G, U_guess)

        else:  # newton
            # Define residual F: u → residual
            # F(u) = (u - u_next)/dt + H(∇u, m) = 0
            def F(U: NDArray) -> NDArray:
                gradients = self._compute_gradients_nd(U, time=time)
                H_values = self._evaluate_hamiltonian_nd(U, M_next, gradients, sigma_at_n, Sigma_at_n)
                return (U - U_next) / self.dt + H_values

            U_solution, info = self.nonlinear_solver.solve(F, U_guess)

        # Warn if not converged
        if not info.converged:
            import warnings

            warnings.warn(
                f"{self.solver_type} did not converge (residual: {info.residual:.2e})",
                UserWarning,
                stacklevel=2,
            )

        # Enforce BC on solution (Issue #542 - nD extension, Issue #527 - centralized BC access)
        # BC-aware gradients use ghost cells for derivatives, but boundary values must be explicitly set
        # This extends the 1D fix from base_hjb.py to nD FDM solver path
        # Refactored to use FDMApplicator.enforce_values() for proper separation of concerns
        bc = self.get_boundary_conditions()
        if bc is not None:
            U_solution = self.bc_applicator.enforce_values(
                field=U_solution,
                boundary_conditions=bc,
                spacing=self.spacing,
                time=time,
            )

        return U_solution

    def _compute_gradients_nd(self, U: NDArray, time: float = 0.0) -> dict[int, NDArray]:
        """Compute gradients using selected advection scheme with proper BC handling.

        Supports two schemes:
        - gradient_centered: Central differences (second-order accurate)
        - gradient_upwind: Godunov upwind (first-order, monotone)

        Uses ghost values from boundary conditions for proper gradient computation
        at domain boundaries, ensuring upwind schemes respect BCs.

        Args:
            U: Value function at current timestep
            time: Current time for time-dependent BC values

        Returns:
            Dict mapping dimension index to gradient array for that dimension.
            Key 0 = ∂u/∂x₀, Key 1 = ∂u/∂x₁, etc.
            Also includes special key -1 for the function value U itself.
        """
        # Store gradients by dimension index for efficient access
        gradients: dict[int, NDArray] = {-1: U}  # -1 = function value

        # Get ghost values from boundary conditions (if available)
        ghost_values = self._get_ghost_values(U, time=time)

        for d in range(self.dimension):
            h = self.spacing[d]

            if self.use_upwind:
                # Godunov upwind scheme: choose based on characteristic direction
                # Forward difference: (U[i+1] - U[i]) / h
                # Backward difference: (U[i] - U[i-1]) / h
                # Select backward if p >= 0, forward if p < 0

                # Compute forward and backward differences
                grad_forward = np.zeros_like(U)
                grad_backward = np.zeros_like(U)

                # Forward difference: D^+ U = (U[i+1] - U[i]) / h
                # Interior points (not including right boundary)
                slices_curr = [slice(None)] * self.dimension
                slices_next = [slice(None)] * self.dimension
                slices_curr[d] = slice(None, -1)
                slices_next[d] = slice(1, None)
                grad_forward[tuple(slices_curr)] = (U[tuple(slices_next)] - U[tuple(slices_curr)]) / h

                # Right boundary: use ghost value if available
                slices_right = [slice(None)] * self.dimension
                slices_right[d] = slice(-1, None)
                slices_prev = [slice(None)] * self.dimension
                slices_prev[d] = slice(-2, -1)
                if ghost_values is not None and (d, 1) in ghost_values:
                    # Use ghost value: (U_ghost - U[-1]) / h
                    u_ghost_right = ghost_values[(d, 1)]
                    grad_forward[tuple(slices_right)] = (
                        np.expand_dims(u_ghost_right, axis=d) - U[tuple(slices_right)]
                    ) / h
                else:
                    # Fallback: backward difference at right boundary
                    grad_forward[tuple(slices_right)] = (U[tuple(slices_right)] - U[tuple(slices_prev)]) / h

                # Backward difference: D^- U = (U[i] - U[i-1]) / h
                # Interior points (not including left boundary)
                slices_curr = [slice(None)] * self.dimension
                slices_prev = [slice(None)] * self.dimension
                slices_curr[d] = slice(1, None)
                slices_prev[d] = slice(None, -1)
                grad_backward[tuple(slices_curr)] = (U[tuple(slices_curr)] - U[tuple(slices_prev)]) / h

                # Left boundary: use ghost value if available
                slices_left = [slice(None)] * self.dimension
                slices_left[d] = slice(0, 1)
                slices_next[d] = slice(1, 2)
                if ghost_values is not None and (d, 0) in ghost_values:
                    # Use ghost value: (U[0] - U_ghost) / h
                    u_ghost_left = ghost_values[(d, 0)]
                    grad_backward[tuple(slices_left)] = (
                        U[tuple(slices_left)] - np.expand_dims(u_ghost_left, axis=d)
                    ) / h
                else:
                    # Fallback: forward difference at left boundary
                    grad_backward[tuple(slices_left)] = (U[tuple(slices_next)] - U[tuple(slices_left)]) / h

                # Godunov upwind selection: use central estimate for sign
                grad_central = (grad_forward + grad_backward) / 2.0
                grad_d = np.where(grad_central >= 0, grad_backward, grad_forward)

            else:
                # Central difference scheme with ghost cell BC handling
                slices = [slice(None)] * self.dimension

                # Interior: central difference
                slices[d] = slice(1, -1)
                slices_fwd = slices.copy()
                slices_fwd[d] = slice(2, None)
                slices_bwd = slices.copy()
                slices_bwd[d] = slice(None, -2)

                grad_interior = (U[tuple(slices_fwd)] - U[tuple(slices_bwd)]) / (2 * h)

                # Left boundary: use ghost value if available
                slices_left = [slice(None)] * self.dimension
                slices_left[d] = slice(0, 1)
                slices_next = [slice(None)] * self.dimension
                slices_next[d] = slice(1, 2)
                if ghost_values is not None and (d, 0) in ghost_values:
                    # Central diff: (U[1] - U_ghost) / (2*h)
                    u_ghost_left = ghost_values[(d, 0)]
                    grad_left = (U[tuple(slices_next)] - np.expand_dims(u_ghost_left, axis=d)) / (2 * h)
                else:
                    # Fallback: one-sided difference
                    grad_left = (U[tuple(slices_next)] - U[tuple(slices_left)]) / h

                # Right boundary: use ghost value if available
                slices_right = [slice(None)] * self.dimension
                slices_right[d] = slice(-1, None)
                slices_prev = [slice(None)] * self.dimension
                slices_prev[d] = slice(-2, -1)
                if ghost_values is not None and (d, 1) in ghost_values:
                    # Central diff: (U_ghost - U[-2]) / (2*h)
                    u_ghost_right = ghost_values[(d, 1)]
                    grad_right = (np.expand_dims(u_ghost_right, axis=d) - U[tuple(slices_prev)]) / (2 * h)
                else:
                    # Fallback: one-sided difference
                    grad_right = (U[tuple(slices_right)] - U[tuple(slices_prev)]) / h

                # Concatenate
                grad_d = np.concatenate([grad_left, grad_interior, grad_right], axis=d)

            # Store with dimension index as key
            gradients[d] = grad_d

        return gradients

    def _get_ghost_values(self, U: NDArray, time: float = 0.0) -> dict[tuple[int, int], NDArray] | None:
        """Get ghost values from geometry boundary conditions.

        Returns ghost values for each boundary, or None if BCs not available.
        Ghost values are computed once per timestep to ensure upwind schemes
        respect boundary conditions at domain boundaries.

        Uses the unified BC infrastructure from geometry/boundary/applicator_fdm.py.
        When BCs are not available, gradient computation falls back to one-sided
        stencils at boundaries (with a warning on first occurrence).

        Args:
            U: Value function at current timestep
            time: Current time for time-dependent BC values

        Returns:
            Dictionary mapping (dimension, side) to ghost value arrays,
            or None if geometry doesn't have boundary conditions.
        """
        # Try to get boundary conditions from geometry
        # Issue #545: self.grid initialized in __init__ for nD, check directly
        try:
            if self.grid is None:
                self._warn_no_bc_once("grid not available")
                return None
        except AttributeError:
            # 1D case: self.grid doesn't exist
            self._warn_no_bc_once("grid not available")
            return None

        # Issue #527: Use centralized get_boundary_conditions() from BaseMFGSolver
        bc = self.get_boundary_conditions()

        if bc is None:
            self._warn_no_bc_once("no boundary_conditions attribute")
            return None

        # Compute ghost values using unified BC infrastructure (Issue #577)
        try:
            # Pad array with ghost cells
            U_padded = pad_array_with_ghosts(U, bc, ghost_depth=1, time=time)

            # Extract ghost values for each boundary
            # Ghost values should match the shape of boundary slices (excluding ghost corners)
            ghost_dict = {}
            for d in range(U.ndim):
                # Left boundary ghost (d, 0) - select interior in other dimensions
                slices_left = [slice(1, -1)] * U.ndim  # Interior in all dims
                slices_left[d] = 0  # Ghost cell in this dim
                ghost_dict[(d, 0)] = U_padded[tuple(slices_left)]

                # Right boundary ghost (d, 1) - select interior in other dimensions
                slices_right = [slice(1, -1)] * U.ndim  # Interior in all dims
                slices_right[d] = -1  # Ghost cell in this dim
                ghost_dict[(d, 1)] = U_padded[tuple(slices_right)]

            return ghost_dict
        except (ValueError, TypeError) as e:
            # BC not compatible with ghost value computation
            self._warn_no_bc_once(f"BC computation failed: {e}")
            return None

    def _warn_no_bc_once(self, reason: str) -> None:
        """Emit warning about missing BC (once per solver instance)."""
        # Issue #545: _bc_warning_emitted initialized in __init__, no hasattr needed
        if not self._bc_warning_emitted:
            import warnings

            warnings.warn(
                f"HJB FDM nD solver: Boundary conditions not available ({reason}). "
                f"Using one-sided stencil fallback at boundaries. "
                f"For proper BC handling, attach BoundaryConditions to geometry or problem.",
                UserWarning,
                stacklevel=4,
            )
            self._bc_warning_emitted = True

    def _evaluate_hamiltonian_nd(
        self,
        U: NDArray,
        M: NDArray,
        gradients: dict[int, NDArray],
        sigma_at_n: float | NDArray | None = None,
        Sigma_at_n: NDArray | None = None,
    ) -> NDArray:
        """Evaluate Hamiltonian at all grid points with variable diffusion support.

        Tries vectorized evaluation first (fast), falls back to point-by-point (compatible).
        Supports scalar and diagonal tensor diffusion.

        Args:
            U: Value function at current timestep
            M: Density at current timestep
            gradients: Dictionary mapping dimension index to gradient arrays.
                       Key d = ∂u/∂xd, Key -1 = function value U.
            sigma_at_n: Scalar diffusion coefficient (None uses problem.sigma, float is constant, array is spatially varying)
            Sigma_at_n: Tensor diffusion coefficient (None or tensor array). If provided and diagonal, computes
                        H_viscosity = (1/2) Σᵢ σᵢ² pᵢ² separately and adds to running cost.
        """
        # Try vectorized evaluation first (10-50x faster)
        try:
            return self._evaluate_hamiltonian_vectorized(U, M, gradients, sigma_at_n, Sigma_at_n)
        except (TypeError, AttributeError, ValueError):
            # Fall back to point-by-point evaluation for compatibility
            pass

        # Point-by-point evaluation (fallback for non-vectorized Hamiltonians)
        H_values = np.zeros(self.shape, dtype=np.float64)

        # Determine which diffusion mode to use
        use_tensor_diffusion = Sigma_at_n is not None

        if not use_tensor_diffusion:
            # Scalar diffusion mode
            if sigma_at_n is None:
                sigma_base = self.problem.sigma
            else:
                sigma_base = sigma_at_n

        for multi_idx in np.ndindex(self.shape):
            x_coords = np.array([self.grid.coordinates[d][multi_idx[d]] for d in range(self.dimension)])
            m_at_point = M[multi_idx]

            # Build gradient vector p from gradients dict
            p = np.array([gradients[d][multi_idx] for d in range(self.dimension)])

            # Build DerivativeTensors for Hamiltonian call
            derivs = DerivativeTensors.from_gradient(p)

            if use_tensor_diffusion:
                # Tensor diffusion mode
                # Get tensor at this point
                if Sigma_at_n.ndim == 2:
                    # Constant tensor
                    Sigma_point = Sigma_at_n
                else:
                    # Spatially-varying tensor
                    Sigma_point = Sigma_at_n[multi_idx]

                # Check if diagonal (should be, given warning in solve_hjb_system)
                if is_diagonal_tensor(Sigma_point):
                    # Extract diagonal elements: σᵢ²
                    sigma_squared = np.diag(Sigma_point)

                    # Compute viscosity term: H_viscosity = (1/2) Σᵢ σᵢ² pᵢ²
                    H_viscosity = 0.5 * np.sum(sigma_squared * p**2)

                    # Compute running cost manually without viscosity term
                    # Standard MFG: H = (coupling/2)|p|² + (sigma²/2)|p|² + V(x) + F(m)
                    # We want:      H_running = (coupling/2)|p|² + V(x) + F(m)
                    #               H_total = H_viscosity + H_running

                    # Control cost using DerivativeTensors
                    H_control = 0.5 * self.problem.coupling_coefficient * derivs.grad_norm_squared

                    # Coupling term F(m) - typically G(m) where G'(m) = g(m)
                    # For standard MFG: F(m) = 0 (coupling is only through g(m) in FP)
                    # If custom components have coupling, it would be included
                    H_coupling_m = 0.0

                    # Potential V(x) if present
                    # For now, assume V=0 unless custom components specify
                    H_potential = 0.0

                    H_running = H_control + H_coupling_m + H_potential
                    H_values[multi_idx] = H_viscosity + H_running
                else:
                    # Non-diagonal tensor - not fully supported yet
                    # Fall back to treating as diagonal (ignoring off-diagonal terms)
                    import warnings

                    warnings.warn(
                        "Non-diagonal tensor detected during HJB evaluation. "
                        "Full tensor support not implemented. Using diagonal approximation (ignoring off-diagonal terms).",
                        UserWarning,
                        stacklevel=2,
                    )
                    # Extract diagonal elements and ignore off-diagonal
                    sigma_squared = np.diag(Sigma_point)

                    # Compute viscosity with diagonal only: H_viscosity = (1/2) Σᵢ σᵢ² pᵢ²
                    H_viscosity = 0.5 * np.sum(sigma_squared * p**2)

                    # Compute running cost using DerivativeTensors
                    H_control = 0.5 * self.problem.coupling_coefficient * derivs.grad_norm_squared
                    H_coupling_m = 0.0
                    H_potential = 0.0

                    H_running = H_control + H_coupling_m + H_potential
                    H_values[multi_idx] = H_viscosity + H_running

            else:
                # Scalar diffusion mode (original code)
                # Get diffusion at this point
                if isinstance(sigma_base, (int, float)):
                    sigma_at_point = sigma_base
                elif isinstance(sigma_base, np.ndarray):
                    sigma_at_point = sigma_base[multi_idx]
                else:
                    sigma_at_point = self.problem.sigma

                # Temporarily override problem.sigma for Hamiltonian evaluation
                original_sigma = self.problem.sigma
                self.problem.sigma = sigma_at_point

                try:
                    # Call problem Hamiltonian (try both interfaces)
                    # Issue #545: Use try/except instead of hasattr
                    try:
                        H_values[multi_idx] = self.problem.hamiltonian(x_coords, m_at_point, p, t=0.0)
                    except AttributeError:
                        # Use new DerivativeTensors format
                        # Legacy support via explicit exception handling
                        try:
                            H_values[multi_idx] = self.problem.H(multi_idx, m_at_point, derivs=derivs)
                        except TypeError:
                            # Legacy: convert to multi-index dict format
                            legacy_derivs = to_multi_index_dict(derivs)
                            H_values[multi_idx] = self.problem.H(multi_idx, m_at_point, derivs=legacy_derivs)
                        except AttributeError:
                            raise AttributeError("Problem must have 'hamiltonian' or 'H' method") from None
                finally:
                    # Restore original sigma
                    self.problem.sigma = original_sigma

        return H_values

    def _evaluate_hamiltonian_vectorized(
        self, U: NDArray, M: NDArray, gradients: dict[int, NDArray], sigma_at_n=None, Sigma_at_n=None
    ) -> NDArray:
        """
        Vectorized Hamiltonian evaluation across all grid points (10-50x faster than point-by-point).

        Args:
            U: Value function at current timestep
            M: Density at current timestep
            gradients: Dictionary mapping dimension index to gradient arrays.
                       Key d = ∂u/∂xd.
            sigma_at_n: Scalar diffusion coefficient
            Sigma_at_n: Tensor diffusion coefficient

        Returns:
            H_values: Hamiltonian evaluated at all grid points

        Raises:
            TypeError: If problem.hamiltonian doesn't support vectorized evaluation
            AttributeError: If required attributes are missing
            ValueError: If array shapes are incompatible
        """
        # Build x_grid: (N_total_points, dimension)
        # Use meshgrid to get coordinates at all points, then flatten
        coord_grids = np.meshgrid(*self.grid.coordinates, indexing="ij")
        x_grid = np.stack([g.ravel() for g in coord_grids], axis=-1)  # (N_total, d)

        # Build p_grid: (N_total_points, dimension)
        # Extract first-order derivatives for each spatial dimension using new format
        p_components = []
        for d in range(self.dimension):
            if d in gradients:
                p_components.append(gradients[d].ravel())
            else:
                p_components.append(np.zeros(x_grid.shape[0]))
        p_grid = np.stack(p_components, axis=-1)  # (N_total, d)

        # Flatten density
        m_grid = M.ravel()  # (N_total,)

        # Handle diffusion coefficient
        use_tensor_diffusion = Sigma_at_n is not None

        if use_tensor_diffusion:
            # Tensor diffusion mode
            if Sigma_at_n.ndim == 2:
                # Constant tensor - check if diagonal
                if not is_diagonal_tensor(Sigma_at_n):
                    raise ValueError(
                        "Vectorized Hamiltonian evaluation only supports diagonal tensor diffusion. "
                        "Non-diagonal tensors require point-by-point evaluation."
                    )
                # Extract diagonal: σᵢ²
                sigma_squared = np.diag(Sigma_at_n)  # (d,)
                # Broadcast to all points: (N_total, d)
                sigma_squared_grid = np.tile(sigma_squared, (x_grid.shape[0], 1))
            else:
                # Spatially-varying tensor - check if diagonal
                if not is_diagonal_tensor(Sigma_at_n):
                    raise ValueError(
                        "Vectorized Hamiltonian evaluation only supports diagonal tensor diffusion. "
                        "Non-diagonal tensors require point-by-point evaluation."
                    )
                # Extract diagonal elements at each point: (*shape, d)
                sigma_squared_grid = np.diagonal(Sigma_at_n, axis1=-2, axis2=-1).reshape(-1, self.dimension)

            # Compute viscosity term: H_viscosity = (1/2) Σᵢ σᵢ² pᵢ²
            H_viscosity = 0.5 * np.sum(sigma_squared_grid * p_grid**2, axis=1)  # (N_total,)

            # Compute running cost without viscosity: |∇u|² = Σᵢ pᵢ²
            p_squared_norm = np.sum(p_grid**2, axis=1)  # (N_total,)
            H_control = 0.5 * self.problem.coupling_coefficient * p_squared_norm
            H_coupling_m = 0.0  # Standard MFG has no direct m-coupling in H
            H_potential = 0.0  # Assume V=0 unless custom components

            H_running = H_control + H_coupling_m + H_potential
            H_values_flat = H_viscosity + H_running

        else:
            # Scalar diffusion mode
            if sigma_at_n is None:
                sigma_val = self.problem.sigma
            else:
                sigma_val = sigma_at_n

            # Prepare sigma for Hamiltonian call
            if isinstance(sigma_val, (int, float)):
                # Constant sigma - no need to pass spatially-varying array
                sigma_for_call = sigma_val
            elif isinstance(sigma_val, np.ndarray):
                # Spatially-varying sigma - flatten it
                sigma_for_call = sigma_val.ravel()  # (N_total,)
            else:
                sigma_for_call = self.problem.sigma

            # Call problem.hamiltonian with vectorized inputs
            # Try both possible signatures
            # Issue #545: Use try/except instead of hasattr
            try:
                # Try calling with vectorized inputs
                # Signature: hamiltonian(x, m, p, t=0.0, sigma=...)
                # This will raise TypeError if not vectorized
                H_values_flat = self.problem.hamiltonian(x_grid, m_grid, p_grid, t=0.0, sigma=sigma_for_call)
            except AttributeError:
                # Old interface - doesn't support vectorization
                # If neither hamiltonian nor H exists, raise clear error
                try:
                    # Check if H exists to provide specific error
                    _ = self.problem.H
                    raise TypeError("Problem.H interface does not support vectorized evaluation")
                except AttributeError:
                    raise AttributeError("Problem must have 'hamiltonian' or 'H' method") from None

        # Reshape back to grid shape
        H_values = H_values_flat.reshape(self.shape)
        return H_values


if __name__ == "__main__":
    """Quick smoke test for development."""
    print("Testing HJBFDMSolver...")

    # Test 1D problem
    from mfg_pde import MFGProblem
    from mfg_pde.geometry import TensorProductGrid

    geometry_1d = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[31])
    problem_1d = MFGProblem(geometry=geometry_1d, T=1.0, Nt=20, diffusion=0.1)
    solver_1d = HJBFDMSolver(problem_1d, solver_type="newton")

    # Test solver initialization
    assert solver_1d.dimension == 1
    assert solver_1d.solver_type == "newton"
    assert solver_1d.hjb_method_name == "FDM"

    # Test solve_hjb_system
    import numpy as np

    M_test = np.ones((problem_1d.Nt + 1, problem_1d.Nx + 1)) * 0.5
    U_final = np.zeros(problem_1d.Nx + 1)
    U_prev = np.zeros((problem_1d.Nt + 1, problem_1d.Nx + 1))

    U_solution = solver_1d.solve_hjb_system(
        M_density_evolution_from_FP=M_test,
        U_final_condition_at_T=U_final,
        U_from_prev_picard=U_prev,
    )

    assert U_solution.shape == (problem_1d.Nt + 1, problem_1d.Nx + 1)
    assert not np.any(np.isnan(U_solution))
    assert not np.any(np.isinf(U_solution))

    print("  1D solver converged")
    print(f"  U range: [{U_solution.min():.3f}, {U_solution.max():.3f}]")

    print("All smoke tests passed!")
