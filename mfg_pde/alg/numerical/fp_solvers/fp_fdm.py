from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sparse

from mfg_pde.backends.compat import has_nan_or_inf
from mfg_pde.geometry import BoundaryConditions
from mfg_pde.utils.aux_func import npart, ppart

from .base_fp import BaseFPSolver
from .fp_fdm_time_stepping import (
    solve_fp_nd_full_system as _solve_fp_nd_full_system,
)

if TYPE_CHECKING:
    from collections.abc import Callable


class FPFDMSolver(BaseFPSolver):
    """
    Finite Difference Method (FDM) solver for Fokker-Planck equations.

    Supports general FP equation: ∂m/∂t + ∇·(α m) = σ²/2 Δm

    Discretization Modes:
        - **Conservative (Flux FDM)** [default, recommended]:
          Discretizes div(αm) as flux differences at cell interfaces.
          Mass conserved exactly (column sums = 1/dt).

        - **Non-conservative (Gradient FDM)**:
          Discretizes α·∇m at cell centers.
          May lose 1-5% mass per unit time. Use for prototyping only.

    Equation Types:
        1. Advection-diffusion (σ>0, α≠0): Standard MFG (stable)
        2. Pure diffusion (σ>0, α=0): Heat equation (stable)
        3. Pure advection (σ=0, α≠0): Transport equation (works but may be unstable
           for sharp fronts; consider WENO/SL solvers for better stability)

    Numerical Scheme:
        - Implicit timestepping for stability
        - Upwind scheme for advection terms
        - Central differences for diffusion terms
        - Supports periodic, Dirichlet, and no-flux boundary conditions

    When to Use Each Mode:
        - conservative=True: Production runs, long time horizons, coupled MFG
        - conservative=False: Quick prototyping, Dirichlet BC (mass loss expected)

    Note:
        FPFDMSolver handles σ=0 (pure advection) algebraically correctly, but
        upwind discretization may exhibit numerical diffusion or instability for
        advection-dominated flows. For pure advection problems, WENO or
        Semi-Lagrangian solvers provide better accuracy and stability.
    """

    def __init__(
        self,
        problem: Any,
        boundary_conditions: BoundaryConditions | None = None,
        conservative: bool = True,
    ) -> None:
        """
        Initialize FDM solver for Fokker-Planck equations.

        Parameters
        ----------
        problem : Any
            MFG problem definition
        boundary_conditions : BoundaryConditions | None
            Boundary condition specification (default: no-flux)
        conservative : bool
            Discretization mode for advection term (default: True).

            True (Flux FDM) [RECOMMENDED]:
                - Discretizes div(alpha * m) as flux differences
                - Velocities computed at cell interfaces
                - Column sums = 1/dt (mass conserved exactly)
                - Use for: production, long runs, coupled MFG

            False (Gradient FDM):
                - Discretizes alpha . grad(m) directly
                - Velocities at cell centers
                - Column sums != 1/dt (1-5% mass loss typical)
                - Use for: quick prototyping, Dirichlet BC only
        """
        super().__init__(problem)
        self.fp_method_name = "FDM"
        self.conservative = conservative

        # Detect problem dimension first (needed for BC creation)
        self.dimension = self._detect_dimension(problem)

        # Boundary condition resolution hierarchy:
        # 1. Explicit boundary_conditions parameter (highest priority)
        # 2. Problem components BC (if available)
        # 3. Grid geometry boundary handler (if available)
        # 4. Default no-flux BC (fallback)
        if boundary_conditions is not None:
            self.boundary_conditions = boundary_conditions
        elif hasattr(problem, "components") and problem.components is not None:
            if problem.components.boundary_conditions is not None:
                self.boundary_conditions = problem.components.boundary_conditions
            else:
                # No BC in components, use default
                from mfg_pde.geometry.boundary import no_flux_bc

                self.boundary_conditions = no_flux_bc(dimension=self.dimension)
        elif hasattr(problem, "geometry") and hasattr(problem.geometry, "get_boundary_handler"):
            # Try to get BC from grid geometry (Phase 2 integration)
            try:
                self.boundary_conditions = problem.geometry.get_boundary_handler()
            except Exception:
                # Fallback if geometry BC retrieval fails
                from mfg_pde.geometry.boundary import no_flux_bc

                self.boundary_conditions = no_flux_bc(dimension=self.dimension)
        else:
            # Default to no-flux boundaries for mass conservation
            from mfg_pde.geometry.boundary import no_flux_bc

            self.boundary_conditions = no_flux_bc(dimension=self.dimension)

    def _detect_dimension(self, problem: Any) -> int:
        """
        Detect the dimension of the problem.

        Returns
        -------
        int
            Problem dimension (1, 2, 3, ...)
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

        # Default to 1D for backward compatibility
        return 1

    def solve_fp_system(
        self,
        M_initial: np.ndarray | None = None,
        drift_field: np.ndarray | Callable | None = None,
        diffusion_field: float | np.ndarray | Callable | None = None,
        tensor_diffusion_field: np.ndarray | Callable | None = None,
        show_progress: bool = True,
        # Deprecated parameter name for backward compatibility
        m_initial_condition: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Solve FP system forward in time with general drift and diffusion support.

        Implements BaseFPSolver unified API for both drift and diffusion.
        Automatically routes to 1D or nD solver based on problem dimension.

        Parameters
        ----------
        M_initial : np.ndarray
            Initial density m₀(x). Shape: (Nx+1,) for 1D or (N1-1, N2-1, ...) for nD
        m_initial_condition : np.ndarray
            DEPRECATED, use M_initial
        drift_field : np.ndarray or callable, optional
            Drift field specification:
            - None: Zero drift (pure diffusion)
            - np.ndarray: Precomputed drift α(t,x), e.g., -∇U/λ for MFG
            - Callable: Function α(t, x, m) -> drift (Phase 2)
            Default: None
        diffusion_field : float, np.ndarray, or callable, optional
            Diffusion specification:
            - None: Use problem.sigma (backward compatible)
            - float: Constant isotropic diffusion
            - np.ndarray: Spatially/temporally varying diffusion
              Shape: (Nx,) for spatial only, (Nt, Nx) for spatiotemporal
            - Callable: State-dependent diffusion D(t, x, m) -> float | ndarray
              Signature: (t: float, x: ndarray, m: ndarray) -> float | ndarray
              Evaluated per timestep using bootstrap strategy
            Default: None
            Note: Cannot be used with tensor_diffusion_field
        tensor_diffusion_field : np.ndarray or callable, optional
            Tensor diffusion specification (Phase 3.0):
            - None: Use scalar diffusion_field instead
            - np.ndarray: Constant or spatially-varying tensor
              Shape: (d, d) for constant, (Ny, Nx, d, d) for 2D spatially-varying
            - Callable: State-dependent tensor Σ(t, x, m) -> (d, d) array
              Signature: (t: float, x: ndarray, m: ndarray) -> (d, d) ndarray
              Must return symmetric positive semi-definite matrix
            Default: None
            Note: Cannot be used with diffusion_field (mutually exclusive)
        show_progress : bool
            Whether to show progress bar

        Returns
        -------
        np.ndarray
            Density evolution. Shape: (Nt+1, Nx+1) for 1D or
            (Nt+1, N1-1, N2-1, ...) for nD

        Examples
        --------
        Pure diffusion (heat equation):
        >>> M = solver.solve_fp_system(m0)

        MFG optimal control:
        >>> drift = -problem.compute_gradient(U_hjb) / problem.control_cost
        >>> M = solver.solve_fp_system(m0, drift_field=drift)

        Custom diffusion coefficient:
        >>> M = solver.solve_fp_system(m0, drift_field=drift, diffusion_field=0.5)

        Spatially varying diffusion (higher at boundaries):
        >>> x_grid = np.linspace(0, 1, problem.Nx + 1)
        >>> diffusion_array = 0.1 + 0.2 * np.abs(x_grid - 0.5)
        >>> M = solver.solve_fp_system(m0, drift_field=drift, diffusion_field=diffusion_array)

        Spatiotemporal diffusion (time and space dependent):
        >>> Nt, Nx = problem.Nt + 1, problem.Nx + 1
        >>> diffusion_field = np.zeros((Nt, Nx))
        >>> for t in range(Nt):
        ...     diffusion_field[t, :] = 0.1 * (1 + 0.5 * t / Nt)  # Increasing over time
        >>> M = solver.solve_fp_system(m0, drift_field=drift, diffusion_field=diffusion_field)

        State-dependent diffusion (porous medium equation):
        >>> def porous_medium(t, x, m):
        ...     return 0.1 * m  # Diffusion proportional to density
        >>> M = solver.solve_fp_system(m0, diffusion_field=porous_medium)

        Density-dependent diffusion with drift:
        >>> def crowd_diffusion(t, x, m):
        ...     return 0.05 + 0.15 * (1 - m / np.max(m))  # Lower diffusion in crowds
        >>> M = solver.solve_fp_system(m0, drift_field=drift, diffusion_field=crowd_diffusion)

        Pure advection (zero diffusion):
        >>> M = solver.solve_fp_system(m0, drift_field=drift, diffusion_field=0.0)

        Anisotropic tensor diffusion (Phase 3.0):
        >>> # Diagonal tensor: faster horizontal diffusion
        >>> Sigma = np.diag([0.2, 0.05])  # σ_x=0.2, σ_y=0.05
        >>> M = solver.solve_fp_system(m0, drift_field=drift, tensor_diffusion_field=Sigma)

        Full tensor with cross-diffusion:
        >>> # 2x2 symmetric tensor
        >>> Sigma = np.array([[0.2, 0.05], [0.05, 0.1]])
        >>> M = solver.solve_fp_system(m0, drift_field=drift, tensor_diffusion_field=Sigma)

        State-dependent tensor diffusion:
        >>> def anisotropic_crowd(t, x, m):
        ...     # Reduce perpendicular diffusion in crowds
        ...     sigma_parallel = 0.2
        ...     sigma_perp = 0.05 * (1 - m / np.max(m))
        ...     return np.diag([sigma_parallel, sigma_perp])
        >>> M = solver.solve_fp_system(m0, tensor_diffusion_field=anisotropic_crowd)
        """
        import warnings

        # Handle deprecated parameter name
        if m_initial_condition is not None:
            if M_initial is not None:
                raise ValueError(
                    "Cannot specify both M_initial and m_initial_condition. "
                    "Use M_initial (m_initial_condition is deprecated)."
                )
            warnings.warn(
                "Parameter 'm_initial_condition' is deprecated. Use 'M_initial' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            M_initial = m_initial_condition

        # Validate required parameter
        if M_initial is None:
            raise ValueError("M_initial is required")

        # Handle drift_field parameter
        if drift_field is None:
            # Zero drift (pure diffusion): create zero U field for internal use
            if hasattr(self.problem, "Nt"):
                Nt = self.problem.Nt + 1
            else:
                raise ValueError("Cannot infer time steps. Ensure problem has Nt attribute.")

            # Create zero U field with appropriate shape
            if self.dimension == 1:
                Nx_val = getattr(self.problem, "Nx", None)
                Nx = Nx_val + 1 if Nx_val is not None else self.problem.geometry.get_grid_shape()[0]
                effective_U = np.zeros((Nt, Nx))
            else:
                grid_shape = self.problem.geometry.get_grid_shape()
                effective_U = np.zeros((Nt, *grid_shape))

        elif isinstance(drift_field, np.ndarray):
            # Precomputed drift field (including MFG drift = -∇U/λ)
            # For FDM, we interpret scalar drift as -∇U and use existing upwind scheme
            effective_U = drift_field
        elif callable(drift_field):
            # Custom drift function - Phase 2
            raise NotImplementedError(
                "FPFDMSolver does not yet support callable drift_field. "
                "Pass precomputed drift as np.ndarray. "
                "Support for callable drift coming in Phase 2."
            )
        else:
            raise TypeError(f"drift_field must be None, np.ndarray, or Callable, got {type(drift_field)}")

        # Validate mutual exclusivity of diffusion_field and tensor_diffusion_field
        if diffusion_field is not None and tensor_diffusion_field is not None:
            raise ValueError(
                "Cannot specify both diffusion_field and tensor_diffusion_field. "
                "Use diffusion_field for scalar diffusion or tensor_diffusion_field for anisotropic diffusion."
            )

        # Handle tensor_diffusion_field (Phase 3.0)
        if tensor_diffusion_field is not None:
            # Tensor diffusion path - only supported for nD
            if self.dimension == 1:
                raise NotImplementedError(
                    "Tensor diffusion not yet implemented for 1D problems. "
                    "Use diffusion_field for scalar diffusion in 1D."
                )

            # Validate and pass to nD solver
            # Note: PSD validation will be done by CoefficientField in nD solver
            if not isinstance(tensor_diffusion_field, (np.ndarray, type(lambda: None))) and not callable(
                tensor_diffusion_field
            ):
                raise TypeError(
                    f"tensor_diffusion_field must be np.ndarray or Callable, got {type(tensor_diffusion_field)}"
                )

            # Route to nD solver with tensor diffusion
            return _solve_fp_nd_full_system(
                m_initial_condition=M_initial,
                U_solution_for_drift=effective_U,
                problem=self.problem,
                boundary_conditions=self.boundary_conditions,
                show_progress=show_progress,
                backend=self.backend,
                diffusion_field=None,
                tensor_diffusion_field=tensor_diffusion_field,
                conservative=self.conservative,
            )

        # Handle diffusion_field parameter (scalar diffusion)
        if diffusion_field is None:
            # Use problem.sigma (backward compatible)
            effective_sigma = self.problem.sigma
        elif isinstance(diffusion_field, (int, float)):
            # Constant isotropic diffusion
            effective_sigma = float(diffusion_field)
        elif isinstance(diffusion_field, np.ndarray):
            # Spatially/temporally varying diffusion - Phase 2.1
            # Shape: (Nt, Nx) for spatiotemporal or broadcastable
            effective_sigma = diffusion_field
        elif callable(diffusion_field):
            # State-dependent diffusion - Phase 2.2/2.4
            # For 1D with conservative=False, use legacy callable solver
            # For 1D with conservative=True or nD, use unified nD solver
            if self.dimension == 1 and not self.conservative:
                return self._solve_fp_1d_with_callable(
                    m_initial_condition=M_initial,
                    drift_field=effective_U,
                    diffusion_callable=diffusion_field,
                    show_progress=show_progress,
                )
            else:
                # nD or 1D conservative: use unified nD solver with callable diffusion
                effective_sigma = diffusion_field
        else:
            raise TypeError(
                f"diffusion_field must be None, float, np.ndarray, or Callable, got {type(diffusion_field)}"
            )

        # Route to appropriate solver based on dimension, conservative mode, and BC type
        #
        # Unified nD solver path: Used for:
        # - All dimensions >= 2
        # - 1D with conservative=True (unified flux-based discretization)
        # - 1D with no_flux BC (can use nD path, more maintainable)
        #
        # Legacy 1D path: Used for backward compatibility with:
        # - 1D with periodic or dirichlet BC (special handling needed)
        # - 1D with conservative=False (legacy gradient-based discretization)

        use_nd_path = (
            self.dimension > 1
            or self.conservative  # Conservative mode always uses unified nD solver
            or self.boundary_conditions.type == "no_flux"  # no_flux works in nD solver
        )

        if use_nd_path:
            # Unified nD solver (works for any dimension including 1D)
            return _solve_fp_nd_full_system(
                m_initial_condition=M_initial,
                U_solution_for_drift=effective_U,
                problem=self.problem,
                boundary_conditions=self.boundary_conditions,
                show_progress=show_progress,
                backend=self.backend,
                diffusion_field=effective_sigma if diffusion_field is not None else None,
                conservative=self.conservative,
            )
        else:
            # Legacy 1D solver for periodic/dirichlet BC with non-conservative mode
            original_sigma = self.problem.sigma
            if diffusion_field is not None and not callable(diffusion_field):
                self.problem.sigma = effective_sigma

            try:
                return self._solve_fp_1d(M_initial, effective_U, show_progress)
            finally:
                self.problem.sigma = original_sigma

    def _solve_fp_1d(
        self, m_initial_condition: np.ndarray, U_solution_for_drift: np.ndarray, show_progress: bool = True
    ) -> np.ndarray:
        """Original 1D FP solver implementation."""
        # Handle both old 1D interface and new geometry-based interface
        if getattr(self.problem, "Nx", None) is not None:
            # Old 1D interface
            Nx = self.problem.Nx + 1
            Dx = self.problem.dx
            Dt = self.problem.dt
        else:
            # Geometry-based interface (CartesianGrid)
            Nx = self.problem.geometry.get_grid_shape()[0]
            Dx = self.problem.geometry.get_grid_spacing()[0]
            Dt = self.problem.dt

        # Infer number of time points from U_solution shape, not problem.Nt
        # n_time_points = number of time knots (including t=0 and t=T)
        # This allows tests to pass edge cases like n_time_points=0 or n_time_points=1
        n_time_points = U_solution_for_drift.shape[0]
        sigma_base = self.problem.sigma  # Base diffusion (scalar or array)
        coupling_coefficient = getattr(self.problem, "coupling_coefficient", 1.0)

        if n_time_points == 0:
            if self.backend is not None:
                return self.backend.zeros((0, Nx))
            return np.zeros((0, Nx))
        if n_time_points == 1:
            if self.backend is not None:
                m_sol = self.backend.zeros((1, Nx))
            else:
                m_sol = np.zeros((1, Nx))
            m_sol[0, :] = m_initial_condition
            m_sol[0, :] = np.maximum(m_sol[0, :], 0)
            # Apply boundary conditions
            if self.boundary_conditions.type == "dirichlet":
                m_sol[0, 0] = self.boundary_conditions.left_value
                m_sol[0, -1] = self.boundary_conditions.right_value
            return m_sol

        if self.backend is not None:
            m = self.backend.zeros((n_time_points, Nx))
        else:
            m = np.zeros((n_time_points, Nx))
        m[0, :] = m_initial_condition
        m[0, :] = np.maximum(m[0, :], 0)
        # Apply boundary conditions to initial condition
        if self.boundary_conditions.type == "dirichlet":
            m[0, 0] = self.boundary_conditions.left_value
            m[0, -1] = self.boundary_conditions.right_value

        # Pre-allocate lists for COO format, then convert to CSR
        row_indices: list[int] = []
        col_indices: list[int] = []
        data_values: list[float] = []

        # Progress bar for forward timesteps
        # Forward FP loop: (n_time_points - 1) steps from index 0 to (n_time_points - 2)
        from mfg_pde.utils.progress import RichProgressBar

        timestep_range = range(n_time_points - 1)
        if show_progress:
            timestep_range = RichProgressBar(
                timestep_range,
                desc="FP (forward)",
                unit="step",
                disable=False,
            )

        for k_idx_fp in timestep_range:
            if Dt < 1e-14:
                m[k_idx_fp + 1, :] = m[k_idx_fp, :]
                continue
            if Dx < 1e-14 and Nx > 1:
                m[k_idx_fp + 1, :] = m[k_idx_fp, :]
                continue

            u_at_tk = U_solution_for_drift[k_idx_fp, :]

            # Extract diffusion coefficient at current timestep
            # Handle scalar (constant) or array (spatially/temporally varying)
            if isinstance(sigma_base, np.ndarray):
                # Array diffusion: shape (Nt, Nx) or broadcastable
                if sigma_base.ndim == 1:
                    # Spatially varying only: sigma.shape = (Nx,)
                    sigma_at_k = sigma_base
                elif sigma_base.ndim == 2:
                    # Spatiotemporal: sigma.shape = (Nt, Nx)
                    sigma_at_k = sigma_base[k_idx_fp, :]
                else:
                    raise ValueError(
                        f"diffusion_field array must be 1D (Nx,) or 2D (Nt, Nx), got shape {sigma_base.shape}"
                    )
            else:
                # Scalar diffusion (constant)
                sigma_at_k = sigma_base

            row_indices.clear()
            col_indices.clear()
            data_values.clear()

            # Handle different boundary conditions
            if self.boundary_conditions.type == "periodic":
                # Original periodic boundary implementation
                for i in range(Nx):
                    # Get diffusion at point i (scalar or array)
                    sigma_i = sigma_at_k[i] if isinstance(sigma_at_k, np.ndarray) else sigma_at_k

                    # Diagonal term for m_i^{k+1}
                    val_A_ii = 1.0 / Dt
                    if Nx > 1:
                        val_A_ii += sigma_i**2 / Dx**2
                        # Advection part of diagonal (outflow from cell i)
                        ip1 = (i + 1) % Nx
                        im1 = (i - 1 + Nx) % Nx
                        val_A_ii += float(
                            coupling_coefficient
                            * (npart(u_at_tk[ip1] - u_at_tk[i]) + ppart(u_at_tk[i] - u_at_tk[im1]))
                            / Dx**2
                        )

                    row_indices.append(i)
                    col_indices.append(i)
                    data_values.append(val_A_ii)

                    if Nx > 1:
                        # Lower diagonal term
                        im1 = (i - 1 + Nx) % Nx  # Previous cell index (periodic)
                        val_A_i_im1 = -(sigma_i**2) / (2 * Dx**2)
                        val_A_i_im1 += float(-coupling_coefficient * npart(u_at_tk[i] - u_at_tk[im1]) / Dx**2)
                        row_indices.append(i)
                        col_indices.append(im1)
                        data_values.append(val_A_i_im1)

                        # Upper diagonal term
                        ip1 = (i + 1) % Nx  # Next cell index (periodic)
                        val_A_i_ip1 = -(sigma_i**2) / (2 * Dx**2)
                        val_A_i_ip1 += float(-coupling_coefficient * ppart(u_at_tk[ip1] - u_at_tk[i]) / Dx**2)
                        row_indices.append(i)
                        col_indices.append(ip1)
                        data_values.append(val_A_i_ip1)

            elif self.boundary_conditions.type == "dirichlet":
                # Dirichlet boundary conditions: m[0] = left_value, m[Nx-1] = right_value
                for i in range(Nx):
                    if i == 0 or i == Nx - 1:
                        # Boundary points: identity equation m[i] = boundary_value
                        row_indices.append(i)
                        col_indices.append(i)
                        data_values.append(1.0)
                    else:
                        # Get diffusion at point i (scalar or array)
                        sigma_i = sigma_at_k[i] if isinstance(sigma_at_k, np.ndarray) else sigma_at_k

                        # Interior points: standard FDM discretization
                        val_A_ii = 1.0 / Dt
                        if Nx > 1:
                            val_A_ii += sigma_i**2 / Dx**2
                            # Advection part (no wrapping for interior points)
                            if i > 0 and i < Nx - 1:
                                val_A_ii += float(
                                    coupling_coefficient
                                    * (npart(u_at_tk[i + 1] - u_at_tk[i]) + ppart(u_at_tk[i] - u_at_tk[i - 1]))
                                    / Dx**2
                                )

                        row_indices.append(i)
                        col_indices.append(i)
                        data_values.append(val_A_ii)

                        if Nx > 1 and i > 0:
                            # Lower diagonal term (flux from left)
                            val_A_i_im1 = -(sigma_i**2) / (2 * Dx**2)
                            val_A_i_im1 += float(-coupling_coefficient * npart(u_at_tk[i] - u_at_tk[i - 1]) / Dx**2)
                            row_indices.append(i)
                            col_indices.append(i - 1)
                            data_values.append(val_A_i_im1)

                        if Nx > 1 and i < Nx - 1:
                            # Upper diagonal term (flux from right)
                            val_A_i_ip1 = -(sigma_i**2) / (2 * Dx**2)
                            val_A_i_ip1 += float(-coupling_coefficient * ppart(u_at_tk[i + 1] - u_at_tk[i]) / Dx**2)
                            row_indices.append(i)
                            col_indices.append(i + 1)
                            data_values.append(val_A_i_ip1)

            elif self.boundary_conditions.type == "no_flux":
                # Two discretization modes:
                # - conservative=True: Flux FDM with interface velocities (mass-preserving)
                # - conservative=False: Gradient FDM (original, may lose mass)

                if self.conservative:
                    # Conservative Flux FDM: discretize div(alpha * m) as flux differences
                    # Interface velocity: alpha_{i+1/2} = -coupling * (u[i+1] - u[i]) / Dx
                    # Upwind flux: F_{i+1/2} = alpha * m_upwind
                    # Column sums = 0 for advection part -> exact mass conservation

                    for i in range(Nx):
                        sigma_i = sigma_at_k[i] if isinstance(sigma_at_k, np.ndarray) else sigma_at_k

                        # Start with time derivative and diffusion
                        val_A_ii = 1.0 / Dt + sigma_i**2 / Dx**2
                        val_A_i_im1 = 0.0
                        val_A_i_ip1 = 0.0

                        # Diffusion coupling (symmetric, standard centered)
                        if i > 0:
                            val_A_i_im1 -= sigma_i**2 / (2 * Dx**2)
                        if i < Nx - 1:
                            val_A_i_ip1 -= sigma_i**2 / (2 * Dx**2)

                        # Right interface: F_{i+1/2}
                        if i < Nx - 1:
                            # Interface velocity at x_{i+1/2}
                            alpha_right = -coupling_coefficient * (u_at_tk[i + 1] - u_at_tk[i]) / Dx

                            if alpha_right >= 0:
                                # Flow to the right: upwind from m_i
                                # F_{i+1/2} = alpha_right * m_i
                                # In row i: +alpha_right/Dx (outflow from cell i)
                                val_A_ii += alpha_right / Dx
                            else:
                                # Flow to the left: upwind from m_{i+1}
                                # F_{i+1/2} = alpha_right * m_{i+1}
                                # In row i: coefficient on m_{i+1}
                                val_A_i_ip1 += alpha_right / Dx

                        # Left interface: -F_{i-1/2}
                        if i > 0:
                            # Interface velocity at x_{i-1/2}
                            alpha_left = -coupling_coefficient * (u_at_tk[i] - u_at_tk[i - 1]) / Dx

                            if alpha_left >= 0:
                                # Flow to the right: upwind from m_{i-1}
                                # F_{i-1/2} = alpha_left * m_{i-1}
                                # In row i: -alpha_left/Dx (inflow to cell i)
                                val_A_i_im1 -= alpha_left / Dx
                            else:
                                # Flow to the left: upwind from m_i
                                # F_{i-1/2} = alpha_left * m_i
                                # In row i: -alpha_left/Dx * m_i (outflow from cell i)
                                val_A_ii -= alpha_left / Dx

                        # Boundary treatment: F at domain boundary = 0 (no flux)
                        # This is automatic: we simply don't add flux terms at boundaries
                        # i=0: no left interface flux, only right interface
                        # i=Nx-1: no right interface flux, only left interface

                        # Add matrix entries
                        row_indices.append(i)
                        col_indices.append(i)
                        data_values.append(val_A_ii)

                        if i > 0 and abs(val_A_i_im1) > 1e-15:
                            row_indices.append(i)
                            col_indices.append(i - 1)
                            data_values.append(val_A_i_im1)

                        if i < Nx - 1 and abs(val_A_i_ip1) > 1e-15:
                            row_indices.append(i)
                            col_indices.append(i + 1)
                            data_values.append(val_A_i_ip1)

                else:
                    # Non-conservative Gradient FDM (original implementation)
                    # Bug #8 Fix: No-flux boundaries WITH advection
                    # Previous "partial fix" dropped advection at boundaries → mass leaked
                    # New strategy: Include advection with one-sided stencils
                    # Accept ~1-2% FDM discretization error as normal

                    for i in range(Nx):
                        # Get diffusion at point i (scalar or array)
                        sigma_i = sigma_at_k[i] if isinstance(sigma_at_k, np.ndarray) else sigma_at_k

                        if i == 0:
                            # Left boundary: include both diffusion AND advection
                            # Use one-sided (forward) stencil for velocity gradient

                            # Diagonal term: time + diffusion + advection (upwind)
                            val_A_ii = 1.0 / Dt + sigma_i**2 / Dx**2

                            # Add advection contribution (one-sided upwind scheme)
                            # For left boundary, use forward difference for velocity
                            # Only positive part contributes (flux out of domain)
                            if Nx > 1:
                                val_A_ii += float(coupling_coefficient * ppart(u_at_tk[i + 1] - u_at_tk[i]) / Dx**2)

                            row_indices.append(i)
                            col_indices.append(i)
                            data_values.append(val_A_ii)

                            # Coupling to m[1]: diffusion + advection
                            val_A_i_ip1 = -(sigma_i**2) / Dx**2
                            if Nx > 1:
                                val_A_i_ip1 += float(-coupling_coefficient * ppart(u_at_tk[i + 1] - u_at_tk[i]) / Dx**2)

                            row_indices.append(i)
                            col_indices.append(i + 1)
                            data_values.append(val_A_i_ip1)

                        elif i == Nx - 1:
                            # Right boundary: include both diffusion AND advection
                            # Use one-sided (backward) stencil for velocity gradient

                            # Diagonal term: time + diffusion + advection (upwind)
                            val_A_ii = 1.0 / Dt + sigma_i**2 / Dx**2

                            # Add advection contribution (one-sided upwind scheme)
                            # For right boundary, use backward difference for velocity
                            # Only negative part contributes (flux out of domain)
                            if Nx > 1:
                                val_A_ii += float(coupling_coefficient * npart(u_at_tk[i] - u_at_tk[i - 1]) / Dx**2)

                            row_indices.append(i)
                            col_indices.append(i)
                            data_values.append(val_A_ii)

                            # Coupling to m[N-2]: diffusion + advection
                            val_A_i_im1 = -(sigma_i**2) / Dx**2
                            if Nx > 1:
                                val_A_i_im1 += float(-coupling_coefficient * npart(u_at_tk[i] - u_at_tk[i - 1]) / Dx**2)

                            row_indices.append(i)
                            col_indices.append(i - 1)
                            data_values.append(val_A_i_im1)

                        else:
                            # Interior points: standard conservative FDM discretization
                            val_A_ii = 1.0 / Dt + sigma_i**2 / Dx**2

                            val_A_ii += float(
                                coupling_coefficient
                                * (npart(u_at_tk[i + 1] - u_at_tk[i]) + ppart(u_at_tk[i] - u_at_tk[i - 1]))
                                / Dx**2
                            )

                            row_indices.append(i)
                            col_indices.append(i)
                            data_values.append(val_A_ii)

                            # Lower diagonal term
                            val_A_i_im1 = -(sigma_i**2) / (2 * Dx**2)
                            val_A_i_im1 += float(-coupling_coefficient * npart(u_at_tk[i] - u_at_tk[i - 1]) / Dx**2)
                            row_indices.append(i)
                            col_indices.append(i - 1)
                            data_values.append(val_A_i_im1)

                            # Upper diagonal term
                            val_A_i_ip1 = -(sigma_i**2) / (2 * Dx**2)
                            val_A_i_ip1 += float(-coupling_coefficient * ppart(u_at_tk[i + 1] - u_at_tk[i]) / Dx**2)
                            row_indices.append(i)
                            col_indices.append(i + 1)
                            data_values.append(val_A_i_ip1)

            A_matrix = sparse.coo_matrix((data_values, (row_indices, col_indices)), shape=(Nx, Nx)).tocsr()

            # Set up right-hand side
            b_rhs = m[k_idx_fp, :] / Dt

            # Apply boundary conditions to RHS
            if self.boundary_conditions.type == "dirichlet":
                b_rhs[0] = self.boundary_conditions.left_value  # Left boundary
                b_rhs[-1] = self.boundary_conditions.right_value  # Right boundary
            elif self.boundary_conditions.type == "no_flux":
                # For no-flux boundaries, RHS remains as m[k]/Dt
                # The no-flux condition is enforced through the matrix coefficients
                pass

            if self.backend is not None:
                m_next_step_raw = self.backend.zeros((Nx,))
            else:
                m_next_step_raw = np.zeros(Nx, dtype=np.float64)
            try:
                if not A_matrix.nnz > 0 and Nx > 0:
                    m_next_step_raw[:] = m[k_idx_fp, :]
                else:
                    solution = sparse.linalg.spsolve(A_matrix, b_rhs)
                    m_next_step_raw[:] = solution

                if has_nan_or_inf(m_next_step_raw, self.backend):
                    m_next_step_raw[:] = m[k_idx_fp, :]
            except Exception:
                m_next_step_raw[:] = m[k_idx_fp, :]

            m[k_idx_fp + 1, :] = m_next_step_raw

            # Ensure boundary conditions are satisfied
            if self.boundary_conditions.type == "dirichlet":
                m[k_idx_fp + 1, 0] = self.boundary_conditions.left_value
                m[k_idx_fp + 1, -1] = self.boundary_conditions.right_value
            elif self.boundary_conditions.type == "no_flux":
                # No additional enforcement needed - no-flux is built into the discretization
                # Ensure non-negativity
                m[k_idx_fp + 1, :] = np.maximum(m[k_idx_fp + 1, :], 0)

        return m

    def _validate_callable_output(
        self,
        output: np.ndarray | float,
        expected_shape: tuple,
        param_name: str,
        timestep: int | None = None,
    ) -> np.ndarray:
        """
        Validate callable coefficient output.

        Parameters
        ----------
        output : np.ndarray or float
            Output from callable (diffusion or drift)
        expected_shape : tuple
            Expected shape for spatial array
        param_name : str
            Parameter name for error messages
        timestep : int, optional
            Current timestep (for error messages)

        Returns
        -------
        np.ndarray
            Validated array (converts scalar to array if needed)

        Raises
        ------
        ValueError
            If output shape is incorrect or contains NaN/Inf
        TypeError
            If output type is incorrect
        """
        # Convert scalar to array
        if isinstance(output, (int, float)):
            output = np.full(expected_shape, float(output))
        elif isinstance(output, np.ndarray):
            # Validate shape
            if output.shape != expected_shape:
                raise ValueError(
                    f"{param_name} callable returned array with shape {output.shape}, "
                    f"expected {expected_shape} at timestep {timestep}"
                )
        else:
            raise TypeError(
                f"{param_name} callable must return float or np.ndarray, got {type(output)} at timestep {timestep}"
            )

        # Check for NaN/Inf
        if has_nan_or_inf(output, self.backend):
            raise ValueError(f"{param_name} callable returned NaN or Inf at timestep {timestep}")

        return output

    def _solve_fp_1d_with_callable(
        self,
        m_initial_condition: np.ndarray,
        drift_field: np.ndarray | None,
        diffusion_callable: callable,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Solve 1D FP equation with callable (state-dependent) diffusion.

        Uses bootstrap strategy: evaluate callable at each timestep using
        the already-computed density m[k] to solve for m[k+1].

        Parameters
        ----------
        m_initial_condition : np.ndarray
            Initial density, shape (Nx,)
        drift_field : np.ndarray or None
            Precomputed drift field, shape (Nt, Nx), or None for zero drift
        diffusion_callable : callable
            Function D(t, x, m) -> diffusion coefficient
            Signature: (float, np.ndarray, np.ndarray) -> float | np.ndarray
        show_progress : bool
            Show progress bar

        Returns
        -------
        np.ndarray
            Density evolution, shape (Nt, Nx)
        """
        from mfg_pde.types.pde_coefficients import DiffusionCallable

        # Validate callable signature using protocol
        if not isinstance(diffusion_callable, DiffusionCallable):
            raise TypeError(
                "diffusion_field callable does not match DiffusionCallable protocol. "
                "Expected signature: (t: float, x: ndarray, m: ndarray) -> float | ndarray"
            )

        # Get problem dimensions
        if getattr(self.problem, "Nx", None) is not None:
            Nx = self.problem.Nx + 1
            Dt = self.problem.dt
            xmin, xmax = self.problem.xmin, self.problem.xmax
        else:
            Nx = self.problem.geometry.get_grid_shape()[0]
            Dt = self.problem.dt
            domain = self.problem.geometry.domain
            xmin, xmax = domain[0][0], domain[0][1]

        # Infer Nt from drift_field if provided, else use problem.Nt
        if drift_field is not None:
            Nt = drift_field.shape[0]
        else:
            Nt = self.problem.Nt + 1

        # Create spatial grid for callable evaluation
        x_grid = np.linspace(xmin, xmax, Nx)

        # Allocate solution array
        if self.backend is not None:
            m_solution = self.backend.zeros((Nt, Nx))
        else:
            m_solution = np.zeros((Nt, Nx))

        m_solution[0, :] = m_initial_condition
        m_solution[0, :] = np.maximum(m_solution[0, :], 0)

        # Apply boundary conditions to initial condition
        if self.boundary_conditions.type == "dirichlet":
            m_solution[0, 0] = self.boundary_conditions.left_value
            m_solution[0, -1] = self.boundary_conditions.right_value

        # Progress bar for forward timesteps with callable diffusion
        # n_time_points - 1 steps to go from t=0 to t=T
        from mfg_pde.utils.progress import RichProgressBar

        timestep_range = range(Nt - 1)
        if show_progress:
            timestep_range = RichProgressBar(
                timestep_range,
                desc="FP (callable diffusion)",
                unit="step",
                disable=False,
            )

        # Bootstrap forward iteration: use m[k] to evaluate callable and compute m[k+1]
        for k in timestep_range:
            t_current = k * Dt
            m_current = m_solution[k, :]

            # Evaluate diffusion callable at current state
            diffusion_at_k = diffusion_callable(t_current, x_grid, m_current)

            # Validate callable output
            diffusion_at_k = self._validate_callable_output(
                diffusion_at_k,
                expected_shape=(Nx,),
                param_name="diffusion_field",
                timestep=k,
            )

            # Temporarily set sigma to evaluated diffusion for this timestep
            original_sigma = self.problem.sigma
            self.problem.sigma = diffusion_at_k

            try:
                # Get drift at current timestep (or zero)
                if drift_field is not None:
                    U_at_k = drift_field[k, :]
                else:
                    U_at_k = np.zeros(Nx)

                # Solve single timestep using _solve_fp_1d machinery
                # Create temporary arrays for single-step solve
                m_temp = np.zeros((2, Nx))
                m_temp[0, :] = m_current
                U_temp = np.zeros((2, Nx))
                U_temp[0, :] = U_at_k

                # Call _solve_fp_1d for single timestep (Nt=2 gives one step)
                # This reuses all the boundary condition logic
                m_result = self._solve_fp_1d(
                    m_initial_condition=m_current,
                    U_solution_for_drift=U_temp,
                    show_progress=False,
                )

                # Extract result at next timestep
                m_solution[k + 1, :] = m_result[1, :]

            finally:
                # Restore original sigma
                self.problem.sigma = original_sigma

        return m_solution


if __name__ == "__main__":
    """Quick smoke test for development."""
    print("Testing FPFDMSolver...")
    print("=" * 60)

    from mfg_pde import MFGProblem
    from mfg_pde.geometry import TensorProductGrid
    from mfg_pde.geometry.boundary import no_flux_bc

    # Test 1D problem using geometry-based API (unified with nD solver)
    print("\n1. Testing 1D FDM (conservative vs non-conservative)...")

    # Create 1D grid with TensorProductGrid
    Nx = 40  # Number of cells (grid points = Nx + 1)
    grid_1d = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], num_points=[Nx + 1])
    dx = grid_1d.get_mesh_spacing()[0]
    x_points = grid_1d.get_spatial_grid()

    problem_1d = MFGProblem(
        geometry=grid_1d,
        Nt=25,
        T=1.0,
        sigma=0.1,
        coupling_coefficient=1.0,
    )

    # Create initial density (Gaussian) and drift field
    x = np.array(x_points)
    m_init_1d = np.exp(-((x - 0.5) ** 2) / 0.05)
    m_init_1d /= m_init_1d.sum() * dx  # Normalize using sum*dx (consistent with 2D)

    # Create drift pushing mass to right (advection test)
    Nt = problem_1d.Nt + 1
    U_test_1d = np.zeros((Nt, Nx + 1))
    for t in range(Nt):
        U_test_1d[t] = -x  # Drift to the right (alpha = -dU/dx = +1)

    # Test non-conservative 1D solver
    solver_1d_nc = FPFDMSolver(problem_1d, boundary_conditions=no_flux_bc(dimension=1), conservative=False)
    assert solver_1d_nc.dimension == 1
    assert solver_1d_nc.fp_method_name == "FDM"
    assert solver_1d_nc.conservative is False

    M_1d_nc = solver_1d_nc.solve_fp_system(m_init_1d, U_test_1d, show_progress=False)
    assert M_1d_nc.shape == (Nt, Nx + 1)
    assert not has_nan_or_inf(M_1d_nc)

    # Test conservative 1D solver
    solver_1d_c = FPFDMSolver(problem_1d, boundary_conditions=no_flux_bc(dimension=1), conservative=True)
    assert solver_1d_c.conservative is True

    M_1d_c = solver_1d_c.solve_fp_system(m_init_1d, U_test_1d, show_progress=False)
    assert M_1d_c.shape == (Nt, Nx + 1)
    assert not has_nan_or_inf(M_1d_c)

    # Calculate mass drift for both (using sum*dx for consistency with 2D)
    initial_mass_1d = m_init_1d.sum() * dx
    final_mass_1d_nc = M_1d_nc[-1].sum() * dx
    final_mass_1d_c = M_1d_c[-1].sum() * dx
    mass_drift_1d_nc = abs(final_mass_1d_nc - initial_mass_1d) / initial_mass_1d
    mass_drift_1d_c = abs(final_mass_1d_c - initial_mass_1d) / initial_mass_1d

    print(f"   Initial mass: {initial_mass_1d:.6f}")
    print(f"   Non-conservative: final={final_mass_1d_nc:.6f}, drift={mass_drift_1d_nc:.2%}")
    print(f"   Conservative:     final={final_mass_1d_c:.6f}, drift={mass_drift_1d_c:.2%}")

    # Test 2D problem with conservative mode
    print("\n2. Testing 2D FDM (conservative vs non-conservative)...")

    # Create 2D problem
    grid_2d = TensorProductGrid(
        dimension=2,
        bounds=[(0.0, 1.0), (0.0, 1.0)],  # [(xmin, xmax), (ymin, ymax)]
        num_points=[11, 11],  # (nx+1, ny+1) grid points
    )
    problem_2d = MFGProblem(
        geometry=grid_2d,
        Nt=20,
        T=0.5,
        sigma=0.2,
        coupling_coefficient=1.0,
    )

    # Create Gaussian initial density
    x, y = grid_2d.coordinates
    X, Y = np.meshgrid(x, y, indexing="ij")
    dx, dy = grid_2d.get_grid_spacing()
    cell_volume = dx * dy
    m_init_2d = np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 0.05)
    m_init_2d /= m_init_2d.sum() * cell_volume

    # Create a drift field that pushes mass to corner (advection test)
    Nt = problem_2d.Nt + 1
    U_drift = np.zeros((Nt, *grid_2d.get_grid_shape()))
    # Potential U = -x - y (drift to upper-right corner)
    for t in range(Nt):
        U_drift[t] = -(X + Y)

    # Test non-conservative solver
    solver_nc = FPFDMSolver(
        problem_2d,
        boundary_conditions=no_flux_bc(dimension=2),
        conservative=False,
    )
    M_nc = solver_nc.solve_fp_system(m_init_2d, U_drift, show_progress=False)

    # Test conservative solver
    solver_c = FPFDMSolver(
        problem_2d,
        boundary_conditions=no_flux_bc(dimension=2),
        conservative=True,
    )
    M_c = solver_c.solve_fp_system(m_init_2d, U_drift, show_progress=False)

    # Calculate mass drift for both (dx, dy already computed above)
    initial_mass_2d = m_init_2d.sum() * cell_volume

    final_mass_nc = M_nc[-1].sum() * cell_volume
    mass_drift_nc = abs(final_mass_nc - initial_mass_2d) / initial_mass_2d

    final_mass_c = M_c[-1].sum() * cell_volume
    mass_drift_c = abs(final_mass_c - initial_mass_2d) / initial_mass_2d

    print(f"   Initial mass: {initial_mass_2d:.6f}")
    print(f"   Non-conservative: final={final_mass_nc:.6f}, drift={mass_drift_nc:.2%}")
    print(f"   Conservative:     final={final_mass_c:.6f}, drift={mass_drift_c:.2%}")

    # Conservative should have better mass conservation
    # (Note: with no-flux BC, both should be reasonable, but conservative is exact)
    assert not has_nan_or_inf(M_nc), "Non-conservative solution has NaN/Inf"
    assert not has_nan_or_inf(M_c), "Conservative solution has NaN/Inf"
    assert np.all(M_nc >= -1e-10), "Non-conservative: density should be non-negative"
    assert np.all(M_c >= -1e-10), "Conservative: density should be non-negative"

    # Verify conservative flag is properly set
    assert solver_nc.conservative is False
    assert solver_c.conservative is True

    print("\n" + "=" * 60)
    print("All smoke tests passed!")
