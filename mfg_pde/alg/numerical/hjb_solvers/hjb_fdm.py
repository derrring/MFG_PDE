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
from mfg_pde.geometry import TensorProductGrid  # Required to check geometry type
from mfg_pde.utils.mfg_logging import get_logger
from mfg_pde.utils.numerical import FixedPointSolver, NewtonSolver
from mfg_pde.utils.pde_coefficients import CoefficientField

from . import base_hjb
from .base_hjb import BaseHJBSolver

logger = get_logger(__name__)

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
    from collections.abc import Callable

    import scipy.sparse as sparse
    from numpy.typing import NDArray

    from mfg_pde.core.mfg_problem import MFGProblem
    from mfg_pde.geometry.boundary import ConstraintProtocol


class HJBFDMSolver(BaseHJBSolver):
    """
    Finite Difference Method solver for HJB equation (all dimensions).

    Automatically handles 1D, 2D, 3D, and higher-dimensional problems:
        - 1D: Uses optimized Newton solver from base_hjb
        - nD: Uses centralized FixedPointSolver or NewtonSolver

    Recommended: d ≤ 3 due to O(N^d) complexity

    Required Geometry Traits (Issue #596 Phase 2.1):
        - SupportsGradient: Provides ∇U operator for Hamiltonian evaluation H(x, ∇U, m)

    Compatible Geometries:
        - TensorProductGrid (structured grids)
        - ImplicitDomain (SDF-based domains)
        - Any geometry implementing SupportsGradient trait

    Example:
        >>> from mfg_pde import MFGProblem
        >>> from mfg_pde.geometry import TensorProductGrid
        >>>
        >>> grid = TensorProductGrid(bounds=[(0,1), (0,1)], Nx=[50, 50])
        >>> problem = MFGProblem(geometry=grid, ...)
        >>> solver = HJBFDMSolver(problem, advection_scheme="gradient_upwind")
        >>> U_solution = solver.solve_hjb_system(M_density, U_final)

    Note:
        Solver uses trait-based operators for gradient computation, eliminating
        manual stencil code and enabling geometry-agnostic algorithm design.
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
        constraint: ConstraintProtocol | None = None,
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
            bc_mode: DEPRECATED (Issue #625). Use BCValueProvider in BoundaryConditions.
                Boundary condition mode for reflecting boundaries (Issue #574):
                - 'standard': Classical Neumann BC (∂U/∂n = 0)
                - 'adjoint_consistent': Coupled BC (∂U/∂n = -σ²/2 · ∂ln(m)/∂n)
                Migration: Use AdjointConsistentProvider in BCSegment.value instead.
            constraint: Variational inequality constraint (Issue #591):
                - ObstacleConstraint: u ≥ ψ or u ≤ ψ (capacity limits, running cost floor)
                - BilateralConstraint: ψ_lower ≤ u ≤ ψ_upper (bounded controls)
                - None: No constraints (default)
                Applied after each timestep solve via projection P_K(u).
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

        # Issue #625: Deprecate bc_mode in favor of BCValueProvider
        if bc_mode != "standard":
            warnings.warn(
                "Parameter 'bc_mode' is deprecated since v0.18.0. "
                "Use BCValueProvider in BoundaryConditions instead:\n\n"
                "  from mfg_pde.geometry.boundary import (\n"
                "      BCSegment, BCType, mixed_bc, AdjointConsistentProvider\n"
                "  )\n\n"
                "  bc = mixed_bc([\n"
                "      BCSegment(\n"
                "          name='left_ac',\n"
                "          bc_type=BCType.ROBIN,\n"
                "          alpha=0.0, beta=1.0,\n"
                "          value=AdjointConsistentProvider(side='left', sigma=sigma),\n"
                "          boundary='x_min',\n"
                "      ),\n"
                "      # ... right boundary similarly\n"
                "  ])\n"
                "  grid = TensorProductGrid(..., boundary_conditions=bc)\n\n"
                "The bc_mode parameter will be removed in v1.0.0. "
                "See Issue #625 for migration details.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Set defaults (use None check to avoid treating 0 as falsy)
        self.max_newton_iterations = max_newton_iterations if max_newton_iterations is not None else 30
        self.newton_tolerance = newton_tolerance if newton_tolerance is not None else 1e-6
        self.solver_type = solver_type
        self.damping_factor = damping_factor
        self.bc_mode = bc_mode
        self.constraint = constraint  # Variational inequality constraint (Issue #591)

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

        # Detect dimension (inherited from BaseNumericalSolver, Issue #633)
        self.dimension = self._detect_dimension()
        # Backward compatibility: 1D uses "FDM", nD uses "FDM-{d}D-{solver_type}"
        if self.dimension == 1:
            self.hjb_method_name = "FDM"
        else:
            self.hjb_method_name = f"FDM-{self.dimension}D-{solver_type}"

        # Validate geometry capabilities (Issue #596 Phase 2.1)
        # HJB solver requires gradient operator for Hamiltonian evaluation
        from mfg_pde.geometry.protocols import SupportsGradient

        if not isinstance(problem.geometry, SupportsGradient):
            raise TypeError(
                f"HJB FDM solver requires geometry with SupportsGradient trait for ∇U computation. "
                f"{type(problem.geometry).__name__} does not implement this trait. "
                f"Compatible geometries: TensorProductGrid, ImplicitDomain."
            )

        # For nD, extract grid info and create nonlinear solver
        if self.dimension > 1:
            # We already imported TensorProductGrid at the top.
            if not isinstance(problem.geometry, TensorProductGrid):
                raise ValueError("nD FDM requires problem with TensorProductGrid geometry")

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

            # Create BC applicator using FDMApplicator (Issue #516)
            from mfg_pde.geometry.boundary.applicator_fdm import FDMApplicator

            self.bc_applicator = FDMApplicator(dimension=self.dimension)

            # Get gradient operators from geometry (Issue #596 Phase 2.1)
            # Operators automatically inherit BC from geometry
            scheme = "upwind" if self.use_upwind else "central"
            self._gradient_operators = problem.geometry.get_gradient_operator(scheme=scheme)

        # Initialize warning flags (Issue #545 - NO hasattr pattern)
        self._bc_warning_emitted: bool = False

    # _detect_dimension() inherited from BaseNumericalSolver (Issue #633)

    def solve_hjb_system(
        self,
        M_density: NDArray | None = None,
        U_terminal: NDArray | None = None,
        U_coupling_prev: NDArray | None = None,
        diffusion_field: float | NDArray | None = None,
        tensor_diffusion_field: NDArray | None = None,
        bc_values: dict[str, float] | None = None,
        progress_callback: Callable[[int], None] | None = None,  # Issue #640
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
            bc_values: DEPRECATED. No longer used (kept for backward compatibility).
                Adjoint-consistent BC now handled automatically via bc_mode parameter.

        Note:
            When bc_mode="adjoint_consistent", the solver automatically creates
            proper Robin BC using create_adjoint_consistent_bc_1d() from the
            current FP density. This integrates with the existing BC framework
            and works with all solver backends.
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

        # Warn about deprecated bc_values parameter
        if bc_values is not None:
            warnings.warn(
                "Parameter 'bc_values' is deprecated and no longer used. "
                "Adjoint-consistent BC is now handled automatically via bc_mode='adjoint_consistent'.",
                DeprecationWarning,
                stacklevel=2,
            )

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

            # Issue #574: Create adjoint-consistent BC if needed
            if self.bc_mode == "adjoint_consistent":
                from mfg_pde.geometry.boundary import create_adjoint_consistent_bc_1d

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

                # Create adjoint-consistent Robin BC from current density
                # Use final time slice if time-dependent
                m_for_bc = M_density[-1, :] if M_density.ndim == 2 else M_density

                # Create BoundaryConditions object with Robin BC segments
                bc = create_adjoint_consistent_bc_1d(
                    m_current=m_for_bc,
                    dx=dx,
                    sigma=sigma,
                    domain_bounds=domain_bounds,
                )

            # Debug: Log BC being passed (Issue #542 investigation)
            # Changed from logger.info to logger.debug to reduce verbosity (Issue #623)
            from contextlib import suppress

            logger.debug(f"[DEBUG Issue #542] BC passed to solve_hjb_system_backward: {bc}")
            # Log segment count if BC has segments attribute (Issue #545: use contextlib.suppress)
            if bc is not None:
                with suppress(AttributeError):
                    logger.debug(f"[DEBUG Issue #542] BC has {len(bc.segments)} segments")

            # Use optimized 1D solver with BC-aware computation (Issue #542 fix)
            U_solution = base_hjb.solve_hjb_system_backward(
                M_density_from_prev_picard=M_density,
                U_final_condition_at_T=U_terminal,
                U_from_prev_picard=U_coupling_prev,
                problem=self.problem,
                max_newton_iterations=self.max_newton_iterations,
                newton_tolerance=self.newton_tolerance,
                backend=self.backend,
                diffusion_field=diffusion_field,
                use_upwind=self.use_upwind,
                bc=bc,  # Now uses proper Robin BC for adjoint-consistent mode (Issue #574)
                domain_bounds=domain_bounds,
            )

            # Apply variational inequality constraint via projection (Issue #591)
            # For 1D path, apply constraint to all timesteps after solving
            if self.constraint is not None:
                for n in range(U_solution.shape[0]):
                    U_solution[n] = self.constraint.project(U_solution[n])

            return U_solution
        else:
            # Use nD solver with centralized nonlinear solver
            return self._solve_hjb_nd(
                M_density,
                U_terminal,
                U_coupling_prev,
                diffusion_field,
                tensor_diffusion_field,
                progress_callback=progress_callback,
            )

    def _solve_hjb_nd(
        self,
        M_density: NDArray,
        U_final: NDArray,
        U_prev: NDArray,
        diffusion_field: float | NDArray | None = None,
        tensor_diffusion_field: NDArray | None = None,
        progress_callback: Callable[[int], None] | None = None,  # Issue #640
    ) -> NDArray:
        """Solve nD HJB using centralized nonlinear solvers with variable diffusion support.

        Supports scalar, array, and callable diffusion coefficients.

        Args:
            progress_callback: Optional callback called after each timestep.
                If provided, internal progress bar is suppressed.
                Typically a subtask.advance callable from HierarchicalProgress.

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

        # Issue #640: Use progress_callback if provided, else show own progress bar
        use_external_progress = progress_callback is not None

        if use_external_progress:
            # External progress manager - just iterate, callback handles updates
            timestep_iter = range(n_time_points - 2, -1, -1)
        else:
            # Standalone mode - show own progress bar
            from mfg_pde.utils.progress import RichProgressBar

            timestep_iter = RichProgressBar(
                range(n_time_points - 2, -1, -1),
                desc=f"HJB {self.dimension}D-FDM ({self.solver_type})",
                unit="step",
            )

        # Backward time loop
        for n in timestep_iter:
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
            U_solution[n] = self._solve_single_timestep(
                U_next, M_next, U_guess, sigma_at_n, Sigma_at_n, time=t_current, constraint=self.constraint
            )

            # Issue #640: Update external progress if callback provided
            if progress_callback is not None:
                progress_callback(1)

        return U_solution

    def _solve_single_timestep(
        self,
        U_next: NDArray,
        M_next: NDArray,
        U_guess: NDArray,
        sigma_at_n: float | NDArray | None = None,
        Sigma_at_n: NDArray | None = None,
        time: float = 0.0,
        constraint: ConstraintProtocol | None = None,
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
            constraint: Variational inequality constraint (Issue #591)
                - ObstacleConstraint: u ≥ ψ or u ≤ ψ
                - BilateralConstraint: ψ_lower ≤ u ≤ ψ_upper
                - None: No constraints
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
        bc = self.get_boundary_conditions()
        if bc is not None:
            U_solution = self.bc_applicator.enforce_values(
                field=U_solution,
                boundary_conditions=bc,
                spacing=self.spacing,
                time=time,
            )

        # Apply variational inequality constraint via projection (Issue #591)
        # Order: 1) Solve PDE → 2) Enforce BC → 3) Project onto constraint set K
        # This ensures the solution satisfies both BC and constraints
        if constraint is not None:
            U_solution = constraint.project(U_solution)

        return U_solution

    def _compute_gradients_nd(self, U: NDArray, time: float = 0.0) -> dict[int, NDArray]:
        """Compute gradients using trait-based geometry operators (Issue #596 Phase 2.1).

        Uses geometry.get_gradient_operator() which automatically handles:
        - Boundary conditions via ghost cells
        - Scheme selection (upwind vs central)
        - Multi-dimensional stencils

        This replaces ~130 lines of manual gradient computation with a clean trait-based interface.

        Args:
            U: Value function at current timestep
            time: Current time for time-dependent BC values

        Returns:
            Dict mapping dimension index to gradient array for that dimension.
            Key 0 = ∂u/∂x₀, Key 1 = ∂u/∂x₁, etc.
            Also includes special key -1 for the function value U itself.

        Note:
            For time-dependent BCs, operators are created per-timestep with correct time.
            For time-independent BCs, cached operators from __init__() are reused.
        """
        # Store gradients by dimension index for efficient access
        gradients: dict[int, NDArray] = {-1: U}  # -1 = function value

        # Check if we need time-dependent operators
        # For now, always create operators with current time to ensure proper BC handling
        # TODO (Issue #596): Optimize by caching operators for time-independent BCs
        scheme = "upwind" if self.use_upwind else "central"
        grad_ops = self.problem.geometry.get_gradient_operator(scheme=scheme, time=time)

        # Apply gradient operators in each direction
        for d in range(self.dimension):
            # Operator call syntax: grad_op(U) applies stencil with BC handling
            gradients[d] = grad_ops[d](U)

        return gradients

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

    # =========================================================================
    # Strict Adjoint Mode (Issue #622)
    # =========================================================================

    def build_advection_matrix(
        self,
        U: NDArray,
        coupling_coefficient: float | None = None,
        time: float = 0.0,
    ) -> sparse.csr_matrix:
        """
        Build upwind advection matrix from value function gradient.

        This matrix encodes the drift velocity v = -coupling_coefficient * ∇U
        using the same upwind discretization that would be used internally.

        For strict adjoint mode (Issue #622), this matrix A_HJB is passed to the
        FP solver which uses A_HJB^T, guaranteeing exact adjoint consistency:
            L_FP = L_HJB^T

        Mathematical Background:
            For MFG with Hamiltonian H = (coupling/2)|∇u|², the optimal control is:
                α* = -coupling_coefficient * ∇u

            The HJB advection term is: α* · ∇u = -coupling * ∇u · ∇u
            This is discretized with upwind differences for stability.

        Args:
            U: Value function at current timestep, shape (*spatial_shape)
            coupling_coefficient: Drift coupling coefficient.
                If None, uses problem.coupling_coefficient
            time: Current time for time-dependent BCs

        Returns:
            Sparse CSR matrix A_advection of shape (N_total, N_total) where
            N_total = prod(spatial_shape). The matrix encodes the advection
            operator using velocity-based linear upwind discretization.

        Example:
            >>> # Build matrix for strict adjoint coupling
            >>> A_hjb = hjb_solver.build_advection_matrix(U_current)
            >>> # FP solver uses transpose
            >>> A_fp = A_hjb.T  # Exact adjoint!

        Note:
            The matrix uses velocity-based linear upwind (same as FP's
            divergence_upwind mode). This ensures the transpose relationship
            holds for mass conservation.

        See Also:
            - Issue #622: Strict Achdou adjoint mode implementation
            - solve_hjb_step_with_matrix(): Uses externally provided matrix
            - FPFDMSolver.solve_fp_step_adjoint_mode(): Uses A^T from this method
        """
        if self.dimension == 1:
            return self._build_advection_matrix_1d(U, coupling_coefficient, time)
        else:
            return self._build_advection_matrix_nd(U, coupling_coefficient, time)

    def _build_advection_matrix_1d(
        self,
        U: NDArray,
        coupling_coefficient: float | None = None,
        time: float = 0.0,
    ) -> sparse.csr_matrix:
        """Build 1D upwind advection matrix.

        Uses velocity-based linear upwind discretization matching FP divergence form.

        For velocity v = -coupling * ∂U/∂x at point i:
        - If v_i > 0 (flow to right): use backward difference (info from left)
          A[i,i] += v_i/dx, A[i,i-1] -= v_i/dx
        - If v_i < 0 (flow to left): use forward difference (info from right)
          A[i,i] -= v_i/dx, A[i,i+1] += v_i/dx

        This matches the "divergence_upwind" scheme in FP solver.
        """
        import scipy.sparse as sparse

        # Get grid info
        Nx = self.problem.geometry.get_grid_shape()[0]
        dx = self.problem.geometry.get_grid_spacing()[0]

        # Get coupling coefficient
        if coupling_coefficient is None:
            coupling_coefficient = getattr(self.problem, "coupling_coefficient", 1.0)

        # Compute gradient ∂U/∂x using BC-aware computation
        bc = self.get_boundary_conditions()
        grad_U = base_hjb._compute_gradient_array_1d(U, dx, bc=bc, upwind=True, time=time)

        # Compute velocity: v = -coupling * ∂U/∂x
        velocity = -coupling_coefficient * grad_U

        # Build sparse matrix using COO format
        row_indices = []
        col_indices = []
        data_values = []

        for i in range(Nx):
            v_i = velocity[i]

            if abs(v_i) < 1e-14:
                # Zero velocity - no advection contribution
                continue

            if v_i > 0:
                # Flow to right: backward difference (upwind from left)
                # (v * m)' ≈ v_i * (m_i - m_{i-1}) / dx
                # Matrix form: A[i,i] = v_i/dx, A[i,i-1] = -v_i/dx
                row_indices.append(i)
                col_indices.append(i)
                data_values.append(v_i / dx)

                if i > 0:
                    row_indices.append(i)
                    col_indices.append(i - 1)
                    data_values.append(-v_i / dx)
                # Boundary: for i=0, no left neighbor (no-flux BC handles this)
            else:
                # Flow to left: forward difference (upwind from right)
                # (v * m)' ≈ v_i * (m_{i+1} - m_i) / dx
                # Matrix form: A[i,i] = -v_i/dx, A[i,i+1] = v_i/dx
                row_indices.append(i)
                col_indices.append(i)
                data_values.append(-v_i / dx)

                if i < Nx - 1:
                    row_indices.append(i)
                    col_indices.append(i + 1)
                    data_values.append(v_i / dx)
                # Boundary: for i=Nx-1, no right neighbor (no-flux BC handles this)

        # Assemble sparse matrix
        A = sparse.coo_matrix(
            (data_values, (row_indices, col_indices)),
            shape=(Nx, Nx),
        ).tocsr()

        return A

    def _build_advection_matrix_nd(
        self,
        U: NDArray,
        coupling_coefficient: float | None = None,
        time: float = 0.0,
    ) -> sparse.csr_matrix:
        """Build nD upwind advection matrix.

        Extends 1D logic to multiple dimensions using the same upwind principle
        applied independently in each direction.

        For each dimension d with velocity v_d = -coupling * ∂U/∂x_d:
        - Upwind from appropriate neighbor based on flow direction
        - Matrix entries are summed across all dimensions
        """
        import scipy.sparse as sparse

        # Get coupling coefficient
        if coupling_coefficient is None:
            coupling_coefficient = getattr(self.problem, "coupling_coefficient", 1.0)

        # Compute gradients using trait-based operators
        gradients = self._compute_gradients_nd(U, time=time)

        # Compute velocity in each direction: v_d = -coupling * ∂U/∂x_d
        velocities = {}
        for d in range(self.dimension):
            velocities[d] = -coupling_coefficient * gradients[d]

        # Build sparse matrix using COO format
        row_indices = []
        col_indices = []
        data_values = []

        N_total = int(np.prod(self.shape))

        for flat_idx in range(N_total):
            multi_idx = self.grid.get_multi_index(flat_idx)

            # Process each dimension
            for d in range(self.dimension):
                dx_d = self.spacing[d]
                v_d = velocities[d][multi_idx]

                if abs(v_d) < 1e-14:
                    continue

                if v_d > 0:
                    # Flow in +x_d direction: backward difference
                    row_indices.append(flat_idx)
                    col_indices.append(flat_idx)
                    data_values.append(v_d / dx_d)

                    # Check if neighbor exists
                    if multi_idx[d] > 0:
                        neighbor_idx = list(multi_idx)
                        neighbor_idx[d] -= 1
                        neighbor_flat = np.ravel_multi_index(tuple(neighbor_idx), self.shape)
                        row_indices.append(flat_idx)
                        col_indices.append(neighbor_flat)
                        data_values.append(-v_d / dx_d)
                else:
                    # Flow in -x_d direction: forward difference
                    row_indices.append(flat_idx)
                    col_indices.append(flat_idx)
                    data_values.append(-v_d / dx_d)

                    # Check if neighbor exists
                    if multi_idx[d] < self.shape[d] - 1:
                        neighbor_idx = list(multi_idx)
                        neighbor_idx[d] += 1
                        neighbor_flat = np.ravel_multi_index(tuple(neighbor_idx), self.shape)
                        row_indices.append(flat_idx)
                        col_indices.append(neighbor_flat)
                        data_values.append(v_d / dx_d)

        # Assemble sparse matrix
        A = sparse.coo_matrix(
            (data_values, (row_indices, col_indices)),
            shape=(N_total, N_total),
        ).tocsr()

        return A


if __name__ == "__main__":
    """Quick smoke test for development."""
    print("Testing HJBFDMSolver...")

    # Test 1D problem
    from mfg_pde import MFGProblem
    from mfg_pde.geometry import TensorProductGrid

    geometry_1d = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[31])
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

    # Test build_advection_matrix (Issue #622 Phase 1)
    print("\nTesting build_advection_matrix (Issue #622)...")
    U_slice = U_solution[10]  # Use a middle timestep
    A_hjb = solver_1d.build_advection_matrix(U_slice)

    import scipy.sparse as sparse

    assert sparse.issparse(A_hjb), "build_advection_matrix should return sparse matrix"
    assert A_hjb.shape == (problem_1d.Nx + 1, problem_1d.Nx + 1), f"Matrix shape mismatch: {A_hjb.shape}"
    assert not np.any(np.isnan(A_hjb.data)), "Matrix contains NaN"
    assert not np.any(np.isinf(A_hjb.data)), "Matrix contains Inf"

    # Verify transpose property: A^T should have same sparsity pattern
    A_hjb_T = A_hjb.T.tocsr()
    assert A_hjb_T.shape == A_hjb.shape, "Transpose should have same shape"

    print(f"  1D advection matrix: shape={A_hjb.shape}, nnz={A_hjb.nnz}")
    print("  build_advection_matrix (1D) passed!")

    # Test 2D problem
    print("\nTesting 2D solver...")
    geometry_2d = TensorProductGrid(bounds=[(0.0, 1.0), (0.0, 1.0)], Nx_points=[11, 11])
    problem_2d = MFGProblem(geometry=geometry_2d, T=1.0, Nt=5, diffusion=0.1)
    solver_2d = HJBFDMSolver(problem_2d, solver_type="newton")

    # Quick 2D build_advection_matrix test
    U_2d = np.zeros((11, 11))
    U_2d[5, 5] = 1.0  # Point source
    A_hjb_2d = solver_2d.build_advection_matrix(U_2d)

    assert sparse.issparse(A_hjb_2d), "2D build_advection_matrix should return sparse matrix"
    assert A_hjb_2d.shape == (11 * 11, 11 * 11), f"2D matrix shape mismatch: {A_hjb_2d.shape}"
    print(f"  2D advection matrix: shape={A_hjb_2d.shape}, nnz={A_hjb_2d.nnz}")
    print("  build_advection_matrix (2D) passed!")

    print("\nAll smoke tests passed!")
