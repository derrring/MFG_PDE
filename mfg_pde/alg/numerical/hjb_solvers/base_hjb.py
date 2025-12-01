from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sparse

from mfg_pde.alg.base_solver import BaseNumericalSolver
from mfg_pde.backends.compat import backend_aware_assign, backend_aware_copy, has_nan_or_inf
from mfg_pde.compat.gradient_notation import derivs_to_p_values_1d
from mfg_pde.utils.pde_coefficients import CoefficientField, get_spatial_grid

if TYPE_CHECKING:
    from mfg_pde.backends.base_backend import BaseBackend
    from mfg_pde.config import BaseConfig  # type: ignore[attr-defined]
    from mfg_pde.core.mfg_problem import MFGProblem

    # from mfg_pde.utils.aux_func import npart, ppart # Not needed here if problem provides jacobian parts

# Clipping limit for p_values ONLY when using numerical FD for Jacobian H-part (fallback)
P_VALUE_CLIP_LIMIT_FD_JAC = 1e6


class BaseHJBSolver(BaseNumericalSolver):
    """Base class for Hamilton-Jacobi-Bellman equation solvers."""

    def __init__(self, problem: MFGProblem, config: BaseConfig | None = None) -> None:
        # Maintain backward compatibility - if no config provided, create a minimal one
        if config is None:
            config = type("MinimalConfig", (), {})()

        super().__init__(problem, config)
        self.hjb_method_name = "BaseHJB"

        # Validate solver compatibility if problem supports it (Phase 3.1.5)
        self._validate_problem_compatibility()

    def _validate_problem_compatibility(self) -> None:
        """
        Validate that this solver is compatible with the problem.

        This method checks if the problem has solver compatibility detection
        (Phase 3.1 unified interface) and validates compatibility if available.
        For older problems without this feature, validation is skipped.
        """
        # Only validate if problem has the new unified interface
        if not hasattr(self.problem, "validate_solver_type"):
            return  # Backward compatibility: skip validation for old problems

        # Get solver type identifier from subclass
        solver_type = self._get_solver_type_id()
        if solver_type is None:
            return  # Solver doesn't specify type, skip validation

        # Validate compatibility
        try:
            self.problem.validate_solver_type(solver_type)
        except ValueError as e:
            # Re-raise with solver class information
            raise ValueError(f"Cannot use {self.__class__.__name__} with this problem.\n\n{e!s}") from e

    def _get_solver_type_id(self) -> str | None:
        """
        Get solver type identifier for compatibility checking.

        Subclasses should override this to return their type identifier.
        Returns None if solver type cannot be determined (skips validation).
        """
        # Map class names to solver type IDs
        class_name = self.__class__.__name__
        type_mapping = {
            "HJBFDMSolver": "fdm",
            "HJBSemiLagrangianSolver": "semi_lagrangian",
            "HJBWENOSolver": "semi_lagrangian",  # WENO is a semi-Lagrangian variant
            "HJBGFDMSolver": "gfdm",
            "HJBNetworkSolver": "network_solver",
            "HJBDGMSolver": "dgm",
            "HJBPINNSolver": "pinn",
        }
        return type_mapping.get(class_name)

    # Implementation of BaseMFGSolver abstract methods
    def solve(self) -> np.ndarray:
        """
        Solve standalone HJB equation (single-agent optimal control).

        This method solves the HJB equation in standalone mode without Mean Field Game
        coupling. It uses uniform density (no population effects) and is suitable for:
        - Single-agent optimal control problems
        - HJB solver comparison and benchmarking
        - Algorithm testing and validation

        For Mean Field Games with population coupling, use through MFG fixed-point
        solver which calls solve_hjb_system() with density from Fokker-Planck equation.

        Returns:
            np.ndarray: Value function U(t,x) of shape (Nt+1, Nx+1) for 1D problems
                       or (Nt+1, *spatial_shape) for nD problems.

        Example:
            >>> from mfg_pde.core.mfg_problem import MFGProblem
            >>> from mfg_pde.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver
            >>> problem = MFGProblem(Nx=100, xmin=0.0, xmax=1.0, Nt=50, T=1.0)
            >>> solver = HJBFDMSolver(problem)
            >>> U = solver.solve()  # Standalone HJB solution
            >>> print(U.shape)  # (51, 101)
        """
        # Create uniform density for standalone mode (no MFG coupling)
        if hasattr(self.problem, "Nx") and self.problem.Nx is not None:
            # 1D problem
            Nt, Nx = self.problem.Nt + 1, self.problem.Nx + 1
            m_uniform = np.ones((Nt, Nx)) / (self.problem.xmax - self.problem.xmin)

            # Get terminal condition from problem
            U_terminal = self.problem.get_final_u()

            # Initial guess for nonlinear solver: repeat terminal condition
            U_prev_picard = np.tile(U_terminal, (Nt, 1))

        elif hasattr(self.problem, "spatial_shape") and self.problem.spatial_shape is not None:
            # nD problem (2D, 3D, etc.)
            Nt = self.problem.Nt + 1
            spatial_shape = self.problem.spatial_shape

            # Compute domain volume for normalization
            if hasattr(self.problem, "spatial_bounds") and self.problem.spatial_bounds:
                volume = 1.0
                for bounds in self.problem.spatial_bounds:
                    volume *= bounds[1] - bounds[0]
            else:
                # Fallback if bounds not available
                volume = 1.0

            # Create uniform density: shape (Nt, *spatial_shape)
            full_shape = (Nt, *spatial_shape)
            m_uniform = np.ones(full_shape) / volume

            # Get terminal condition from problem
            U_terminal = self.problem.get_final_u()

            # Initial guess for nonlinear solver: repeat terminal condition across time
            U_prev_picard = np.tile(U_terminal, (Nt,) + (1,) * len(spatial_shape))

        else:
            # Complex geometry or network - not yet supported
            raise NotImplementedError(
                "Standalone solve() for geometry/network problems not yet implemented. "
                "Use solve_hjb_system() directly with uniform density."
            )

        # Solve using the specific solver's implementation
        return self.solve_hjb_system(m_uniform, U_terminal, U_prev_picard)

    def validate_solution(self) -> dict[str, float]:
        """Validate the HJB solution."""
        # Basic validation - can be extended by specific solvers
        return {
            "hjb_residual": 0.0,  # Placeholder
            "boundary_error": 0.0,  # Placeholder
            "terminal_condition_error": 0.0,  # Placeholder
        }

    def discretize(self) -> None:
        """Set up spatial and temporal discretization for HJB equation."""
        # Most HJB solvers will discretize based on the problem geometry
        # This is a placeholder that can be overridden

    @abstractmethod
    def solve_hjb_system(
        self,
        M_density_evolution_from_FP: np.ndarray,
        U_final_condition_at_T: np.ndarray,
        U_from_prev_picard: np.ndarray,
        diffusion_field: float | np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Solve the HJB system given density evolution and boundary conditions.

        This is the main method that specific HJB solvers must implement.

        The HJB equation is:
            -∂u/∂t + H(∇u, x, t, m) - σ²(x,t)/2 Δu = 0

        where σ²(x,t) is the diffusion coefficient that can be:
        - Constant (scalar): σ² = constant (classical MFG)
        - Spatially varying: σ²(x,t) varies in space/time
        - None: Use problem.sigma (backward compatible)

        Args:
            M_density_evolution_from_FP: Density field m(t,x) from Fokker-Planck equation
                                        For MFG: from previous Picard iteration
                                        For standalone: uniform density
            U_final_condition_at_T: Terminal condition u(T,x) at final time
            U_from_prev_picard: Value function from previous Picard iteration
                               For MFG: actual previous iterate U^{k-1}
                               For standalone: initial guess (zeros, terminal condition, etc.)
            diffusion_field: Diffusion coefficient specification (optional):
                - None: Use problem.sigma (backward compatible)
                - float: Constant diffusion σ²
                - np.ndarray: Spatially/temporally varying diffusion σ²(t,x)
                  Shape: (Nt+1, Nx+1) for 1D, (Nt+1, Nx+1, Ny+1) for 2D, etc.
                Default: None

        Returns:
            np.ndarray: Value function U(t,x) solution

        Note:
            For MFG consistency, the same diffusion_field should be used in both
            HJB and FP solvers. The coupling solver handles this synchronization.
        """


def _calculate_derivatives(
    U_array: np.ndarray,
    i: int,
    Dx: float,
    Nx: int,
    clip: bool = False,
    clip_limit: float = P_VALUE_CLIP_LIMIT_FD_JAC,
    upwind: bool = False,
) -> dict[tuple[int], float]:
    """
    Calculate derivatives using standard tuple multi-index notation.

    This function computes 1D derivatives and returns them in tuple notation:
    - derivs[(0,)] = u (function value)
    - derivs[(1,)] = ∂u/∂x (first derivative)

    Supports both central difference (default) and upwind discretization:
    - Central: p = (p_forward + p_backward) / 2 (second-order accurate)
    - Upwind: Godunov scheme based on characteristic direction (monotone)

    Args:
        U_array: Solution array
        i: Spatial index
        Dx: Spatial grid spacing
        Nx: Number of spatial points
        clip: Whether to clip derivative values
        clip_limit: Maximum absolute value for clipping
        upwind: If True, use Godunov upwind discretization for HJB stability

    Returns:
        Dictionary with tuple keys: {(0,): u, (1,): p}

    See:
        - docs/gradient_notation_standard.md
        - mfg_pde/compat/gradient_notation.py
    """
    # Extract function value
    if hasattr(U_array[i], "item"):
        u_i = U_array[i].item()
    else:
        u_i = float(U_array[i])

    # Handle edge cases
    if Nx == 1:
        return {(0,): u_i, (1,): 0.0}

    if abs(Dx) < 1e-14:
        return {(0,): u_i, (1,): np.nan}

    # Extract neighbor values
    if hasattr(U_array[(i + 1) % Nx], "item"):
        u_ip1 = U_array[(i + 1) % Nx].item()
    else:
        u_ip1 = float(U_array[(i + 1) % Nx])

    if hasattr(U_array[(i - 1 + Nx) % Nx], "item"):
        u_im1 = U_array[(i - 1 + Nx) % Nx].item()
    else:
        u_im1 = float(U_array[(i - 1 + Nx) % Nx])

    # Check for NaN/Inf in values
    if np.isinf(u_i) or np.isinf(u_ip1) or np.isinf(u_im1) or np.isnan(u_i) or np.isnan(u_ip1) or np.isnan(u_im1):
        return {(0,): u_i, (1,): np.nan}

    # Compute forward and backward differences
    p_forward = (u_ip1 - u_i) / Dx
    p_backward = (u_i - u_im1) / Dx

    # Check for NaN/Inf in derivatives
    if np.isinf(p_forward) or np.isnan(p_forward):
        p_forward = np.nan
    if np.isinf(p_backward) or np.isnan(p_backward):
        p_backward = np.nan

    # Choose discretization based on upwind flag
    if np.isnan(p_forward) or np.isnan(p_backward):
        p_value = np.nan
    elif upwind:
        # Godunov upwind: choose based on characteristic direction
        # For typical MFG Hamiltonian H = |p|²/(2σ²) + V(x,m), characteristic velocity = p/σ²
        # Use backward difference if p ≥ 0 (info from left), forward if p < 0 (info from right)
        p_central_for_sign = (p_forward + p_backward) / 2.0
        if p_central_for_sign >= 0:
            p_value = p_backward  # Characteristic from left
        else:
            p_value = p_forward  # Characteristic from right
    else:
        # Central difference (default, second-order accurate)
        p_value = (p_forward + p_backward) / 2.0

    # Clip if requested
    if clip and not np.isnan(p_value):
        p_value = np.clip(p_value, -clip_limit, clip_limit)

    return {(0,): u_i, (1,): p_value}


def _calculate_p_values(
    U_array: np.ndarray,
    i: int,
    Dx: float,
    Nx: int,
    clip: bool = False,
    clip_limit: float = P_VALUE_CLIP_LIMIT_FD_JAC,
) -> dict[str, float]:
    """
    Legacy wrapper for _calculate_derivatives() using string keys.

    DEPRECATED: Use _calculate_derivatives() with tuple notation instead.

    This function maintains backward compatibility for code expecting
    string keys {"forward": ..., "backward": ...}.

    Returns:
        Dictionary with string keys {" forward": p, "backward": p}

    See:
        - _calculate_derivatives() for new tuple notation
        - mfg_pde/compat/gradient_notation.py for conversion utilities
    """
    # Call new tuple-based function
    derivs = _calculate_derivatives(U_array, i, Dx, Nx, clip=clip, clip_limit=clip_limit)

    # Convert to legacy format
    return derivs_to_p_values_1d(derivs)


def _clip_p_values(p_values: dict[str, float], clip_limit: float) -> dict[str, float]:  # Helper for FD Jac
    clipped_p_values = {}
    for key, p_val in p_values.items():
        if np.isnan(p_val) or np.isinf(p_val):
            clipped_p_values[key] = np.nan
        else:
            clipped_p_values[key] = np.clip(p_val, -clip_limit, clip_limit)
    return clipped_p_values


def compute_hjb_residual(
    U_n_current_newton_iterate: np.ndarray,  # U_kp1_n in notebook's getFnU_withM
    U_n_plus_1_from_hjb_step: np.ndarray,  # U_kp1_np1 in notebook
    M_density_at_n_plus_1: np.ndarray,  # M_k_np1 in notebook
    problem: MFGProblem,
    t_idx_n: int,  # Time index for U_n
    backend=None,  # Backend for MPS/CUDA support
    sigma_at_n: float | np.ndarray | None = None,  # Diffusion at time t_n
) -> np.ndarray:
    Nx = problem.Nx + 1
    dx = problem.dx
    dt = problem.dt

    # Handle diffusion field - NumPy will broadcast scalar automatically
    if sigma_at_n is None:
        sigma = problem.sigma  # Backward compatible (scalar)
    elif isinstance(sigma_at_n, (int, float)):
        sigma = sigma_at_n  # Keep as scalar (not float()) for broadcasting
    else:
        # Spatially varying diffusion array
        sigma = sigma_at_n

    if backend is not None:
        Phi_U = backend.zeros((Nx,))
    else:
        Phi_U = np.zeros(Nx)

    if has_nan_or_inf(U_n_current_newton_iterate, backend):
        if backend is not None:
            return backend.full((Nx,), float("nan"))
        return np.full(Nx, np.nan)

    # Time derivative: (U_n_current - U_{n+1})/dt
    # Notebook FnU[i] += -(Ukp1_np1[i] - Ukp1_n[i])/dt;  (U_n - U_{n+1})/dt
    if abs(dt) < 1e-14:
        if not np.allclose(U_n_current_newton_iterate, U_n_plus_1_from_hjb_step, rtol=1e-9, atol=1e-9):
            pass
    else:
        time_deriv_term = (U_n_current_newton_iterate - U_n_plus_1_from_hjb_step) / dt
        if has_nan_or_inf(time_deriv_term, backend):
            if backend is not None:
                return backend.full((Nx,), float("nan"))
            Phi_U[:] = np.nan
            return Phi_U
        Phi_U += time_deriv_term

    # Diffusion term: -(sigma^2/2) * (U_n_current)_xx
    # Notebook FnU[i] += - ((sigma**2)/2.) * (Ukp1_n[i+1]-2*Ukp1_n[i]+Ukp1_n[i-1])/(dx**2)
    if abs(dx) > 1e-14 and Nx > 1:
        # Use backend-aware roll operation
        if backend is not None and hasattr(U_n_current_newton_iterate, "roll"):
            # PyTorch tensors have .roll() method
            U_xx = (
                U_n_current_newton_iterate.roll(-1)
                - 2 * U_n_current_newton_iterate
                + U_n_current_newton_iterate.roll(1)
            ) / dx**2
        else:
            # NumPy arrays use np.roll()
            U_xx = (
                np.roll(U_n_current_newton_iterate, -1)
                - 2 * U_n_current_newton_iterate
                + np.roll(U_n_current_newton_iterate, 1)
            ) / dx**2
        if has_nan_or_inf(U_xx, backend):
            if backend is not None:
                return backend.full((Nx,), float("nan"))
            Phi_U[:] = np.nan
            return Phi_U

        # Apply diffusion term (NumPy broadcasts scalar automatically)
        # Works for both constant σ (scalar) and σ(x,t) (array)
        Phi_U += -(sigma**2 / 2.0) * U_xx

    # For m-coupling term, original notebook passed gradUkn, gradUknim1 (from prev Picard iter)
    # but mdmH_withM itself didn't use them. We'll pass an empty dict for now.
    U_n_derivatives_for_m_coupling: dict[str, Any] = {}  # Not used by MFGProblem's term

    for i in range(Nx):
        # Get scalar value for nan check (works for both NumPy and PyTorch)
        phi_val = Phi_U[i].item() if hasattr(Phi_U[i], "item") else float(Phi_U[i])
        if np.isnan(phi_val):
            continue

        # For Hamiltonian, use unclipped p_values derived from U_n_current_newton_iterate
        # Calculate derivatives using tuple notation (Phase 3 migration)
        # Use upwind=True for HJB FDM stability (Godunov upwind discretization)
        derivs = _calculate_derivatives(U_n_current_newton_iterate, i, dx, Nx, clip=False, upwind=True)

        if np.any(np.isnan(list(derivs.values()))):
            Phi_U[i] = float("nan")
            continue

        # Hamiltonian term H(x_i, M_{n+1,i}, (Du_n_current)_i, t_n)
        # Notebook: FnU[i] += H_withM(...)
        # Get M value as scalar for H function
        m_val = (
            M_density_at_n_plus_1[i].item()
            if hasattr(M_density_at_n_plus_1[i], "item")
            else float(M_density_at_n_plus_1[i])
        )
        # Call H() with tuple notation directly (Phase 3: no more conversion to legacy format)
        hamiltonian_val = problem.H(x_idx=i, m_at_x=m_val, derivs=derivs, t_idx=t_idx_n)
        if np.isnan(hamiltonian_val) or np.isinf(hamiltonian_val):
            Phi_U[i] = float("nan")
            continue
        else:
            Phi_U[i] += hamiltonian_val

        # Problem-specific m-coupling term (like mdmH_withM from notebook)
        # Notebook: FnU[i] += mdmH_withM(...)
        m_coupling_term = problem.get_hjb_residual_m_coupling_term(
            M_density_at_n_plus_1, U_n_derivatives_for_m_coupling, i, t_idx_n
        )
        if m_coupling_term is not None:
            if np.isnan(m_coupling_term) or np.isinf(m_coupling_term):
                Phi_U[i] = float("nan")
                continue
            Phi_U[i] += m_coupling_term

    return Phi_U


def compute_hjb_jacobian(
    U_n_current_newton_iterate: np.ndarray,  # U_new_n_tmp in notebook Newton step
    U_k_n_from_prev_picard: np.ndarray,  # U_k_n (or Uoldn) in notebook Jacobian
    M_density_at_n_plus_1: np.ndarray,
    problem: MFGProblem,
    t_idx_n: int,
    backend=None,  # Backend for MPS/CUDA support
    sigma_at_n: float | np.ndarray | None = None,  # Diffusion at time t_n
) -> sparse.csr_matrix:
    Nx = problem.Nx + 1
    dx = problem.dx
    dt = problem.dt
    eps = 1e-7

    # Handle diffusion field - NumPy will broadcast scalar automatically
    if sigma_at_n is None:
        sigma = problem.sigma  # Backward compatible (scalar)
    elif isinstance(sigma_at_n, (int, float)):
        sigma = sigma_at_n  # Keep as scalar (not float()) for broadcasting
    else:
        # Spatially varying diffusion array
        sigma = sigma_at_n

    # For Jacobian, we always need NumPy arrays for scipy.sparse
    # Convert backend arrays to NumPy if needed
    if backend is not None:
        from mfg_pde.backends.compat import to_numpy

        U_n_np = to_numpy(U_n_current_newton_iterate, backend)
    else:
        U_n_np = U_n_current_newton_iterate

    J_D = np.zeros(Nx)
    J_L = np.zeros(Nx)
    J_U = np.zeros(Nx)

    if has_nan_or_inf(U_n_current_newton_iterate, backend):
        return sparse.diags([np.full(Nx, np.nan)], [0], shape=(Nx, Nx)).tocsr()

    # Time derivative part: d/dU_n_current[j] of (U_n_current[i] - U_{n+1}[i])/dt
    if abs(dt) > 1e-14:
        J_D += 1.0 / dt

    # Diffusion part: d/dU_n_current[j] of -(sigma^2/2) * (U_n_current)_xx[i]
    # NumPy broadcasts scalar σ automatically to match array shapes
    if abs(dx) > 1e-14 and Nx > 1:
        # Diagonal: ∂/∂U[i] of -(σ²/2)(U[i-1] - 2U[i] + U[i+1])/dx² = σ²/dx²
        J_D += sigma**2 / dx**2
        # Off-diagonal: coefficient for U[i±1] terms
        val_off_diag_diff = -(sigma**2) / (2 * dx**2)
        J_L += val_off_diag_diff
        J_U += val_off_diag_diff
        # Note: For spatially varying σ(x), this assumes σ is smooth.
        # More accurate treatment would include ∂σ/∂x terms (Phase 3 extension)

    # Hamiltonian part & m-coupling term's Jacobian contribution
    # Try to get analytical/specific Jacobian contributions from the problem for H-part
    # Crucially, pass U_k_n_from_prev_picard for MFGProblem's specific Jacobian
    hamiltonian_jac_contrib = problem.get_hjb_hamiltonian_jacobian_contrib(
        U_k_n_from_prev_picard,
        t_idx_n,  # This is Uoldn from original notebook
    )

    if hamiltonian_jac_contrib is not None:
        J_D += hamiltonian_jac_contrib.diagonal
        J_L += hamiltonian_jac_contrib.lower
        J_U += hamiltonian_jac_contrib.upper
    else:
        # Fallback to numerical Jacobian for H-part, using NumPy version
        for i in range(Nx):
            U_perturbed_p_i = U_n_np.copy()
            U_perturbed_p_i[i] += eps
            U_perturbed_m_i = U_n_np.copy()
            U_perturbed_m_i[i] -= eps

            # Use tuple notation for Jacobian (Phase 3 migration)
            # Use upwind=True for consistency with residual computation
            derivs_p_i = _calculate_derivatives(
                U_perturbed_p_i,
                i,
                dx,
                Nx,
                clip=True,
                clip_limit=P_VALUE_CLIP_LIMIT_FD_JAC,
                upwind=True,
            )
            derivs_m_i = _calculate_derivatives(
                U_perturbed_m_i,
                i,
                dx,
                Nx,
                clip=True,
                clip_limit=P_VALUE_CLIP_LIMIT_FD_JAC,
                upwind=True,
            )

            H_p_i = np.nan
            H_m_i = np.nan
            if not (np.any(np.isnan(list(derivs_p_i.values())))):
                H_p_i = problem.H(i, M_density_at_n_plus_1[i], derivs=derivs_p_i, t_idx=t_idx_n)
            if not (np.any(np.isnan(list(derivs_m_i.values())))):
                H_m_i = problem.H(i, M_density_at_n_plus_1[i], derivs=derivs_m_i, t_idx=t_idx_n)

            if not (np.isnan(H_p_i) or np.isnan(H_m_i) or np.isinf(H_p_i) or np.isinf(H_m_i)):
                J_D[i] += (H_p_i - H_m_i) / (2 * eps)
            else:
                J_D[i] += 0

            if Nx > 1:
                # ... (numerical FD for J_L_H and J_U_H as before, using U_n_current_newton_iterate for perturbations)
                im1 = (i - 1 + Nx) % Nx
                U_perturbed_p_im1 = U_n_current_newton_iterate.copy()
                U_perturbed_p_im1[im1] += eps
                U_perturbed_m_im1 = U_n_current_newton_iterate.copy()
                U_perturbed_m_im1[im1] -= eps
                # Use tuple notation for Jacobian (Phase 3 migration)
                derivs_p_im1 = _calculate_derivatives(
                    U_perturbed_p_im1,
                    i,
                    dx,
                    Nx,
                    clip=True,
                    clip_limit=P_VALUE_CLIP_LIMIT_FD_JAC,
                    upwind=True,
                )
                derivs_m_im1 = _calculate_derivatives(
                    U_perturbed_m_im1,
                    i,
                    dx,
                    Nx,
                    clip=True,
                    clip_limit=P_VALUE_CLIP_LIMIT_FD_JAC,
                    upwind=True,
                )
                H_p_im1 = np.nan
                H_m_im1 = np.nan
                if not (np.any(np.isnan(list(derivs_p_im1.values())))):
                    H_p_im1 = problem.H(i, M_density_at_n_plus_1[i], derivs=derivs_p_im1, t_idx=t_idx_n)
                if not (np.any(np.isnan(list(derivs_m_im1.values())))):
                    H_m_im1 = problem.H(i, M_density_at_n_plus_1[i], derivs=derivs_m_im1, t_idx=t_idx_n)
                if not (np.isnan(H_p_im1) or np.isnan(H_m_im1) or np.isinf(H_p_im1) or np.isinf(H_m_im1)):
                    J_L[i] += (H_p_im1 - H_m_im1) / (2 * eps)
                else:
                    J_L[i] += 0

                ip1 = (i + 1) % Nx
                U_perturbed_p_ip1 = U_n_current_newton_iterate.copy()
                U_perturbed_p_ip1[ip1] += eps
                U_perturbed_m_ip1 = U_n_current_newton_iterate.copy()
                U_perturbed_m_ip1[ip1] -= eps
                # Use tuple notation for Jacobian (Phase 3 migration)
                derivs_p_ip1 = _calculate_derivatives(
                    U_perturbed_p_ip1,
                    i,
                    dx,
                    Nx,
                    clip=True,
                    clip_limit=P_VALUE_CLIP_LIMIT_FD_JAC,
                    upwind=True,
                )
                derivs_m_ip1 = _calculate_derivatives(
                    U_perturbed_m_ip1,
                    i,
                    dx,
                    Nx,
                    clip=True,
                    clip_limit=P_VALUE_CLIP_LIMIT_FD_JAC,
                    upwind=True,
                )
                H_p_ip1 = np.nan
                H_m_ip1 = np.nan
                if not (np.any(np.isnan(list(derivs_p_ip1.values())))):
                    H_p_ip1 = problem.H(i, M_density_at_n_plus_1[i], derivs=derivs_p_ip1, t_idx=t_idx_n)
                if not (np.any(np.isnan(list(derivs_m_ip1.values())))):
                    H_m_ip1 = problem.H(i, M_density_at_n_plus_1[i], derivs=derivs_m_ip1, t_idx=t_idx_n)
                if not (np.isnan(H_p_ip1) or np.isnan(H_m_ip1) or np.isinf(H_p_ip1) or np.isinf(H_m_ip1)):
                    J_U[i] += (H_p_ip1 - H_m_ip1) / (2 * eps)
                else:
                    J_U[i] += 0

    # Assemble sparse Jacobian
    # The original notebook used rolled J_L and J_U for its spdiags call:
    # spdiags([np.roll(ML,-1),MD,np.roll(MU,1)],[-1,0,1],Nx,Nx)
    # where ML[j] was dRes[j]/dU[j-1] and MU[j] was dRes[j]/dU[j+1]
    # So, np.roll(J_L, -1)[k] = J_L[k+1] = dRes[k+1]/dU[k] (for offset -1)
    # And np.roll(J_U, 1)[k] = J_U[k-1] = dRes[k-1]/dU[k] (for offset 1)
    # This matches the spdiags convention if J_L and J_U are the direct band contributions.

    J_L_for_spdiags = np.roll(J_L, -1) if Nx > 1 else J_L
    J_U_for_spdiags = np.roll(J_U, 1) if Nx > 1 else J_U

    diagonals_data = [J_L_for_spdiags, J_D, J_U_for_spdiags] if Nx > 1 else [J_D]
    offsets = [-1, 0, 1] if Nx > 1 else [0]

    try:
        Jac = sparse.spdiags(diagonals_data, offsets, Nx, Nx, format="csr")
    except ValueError:
        fallback_diag = np.ones(Nx) * (1.0 / dt if abs(dt) > 1e-14 else 1.0)
        Jac = sparse.diags([fallback_diag], [0], shape=(Nx, Nx)).tocsr()

    return Jac.tocsr()


def newton_hjb_step(
    U_n_current_newton_iterate: np.ndarray,  # U_new_n_tmp in notebook
    U_n_plus_1_from_hjb_step: np.ndarray,  # U_new_np1 in notebook
    U_k_n_from_prev_picard: np.ndarray,  # U_k_n in notebook
    M_density_at_n_plus_1: np.ndarray,
    problem: MFGProblem,
    t_idx_n: int,
    backend=None,  # Add backend parameter for MPS/CUDA support
    sigma_at_n: float | np.ndarray | None = None,  # Diffusion at time t_n
) -> tuple[np.ndarray, float]:
    dx_norm = problem.dx if abs(problem.dx) > 1e-12 else 1.0

    if has_nan_or_inf(U_n_current_newton_iterate, backend):
        return U_n_current_newton_iterate, np.inf

    residual_F_U = compute_hjb_residual(
        U_n_current_newton_iterate,
        U_n_plus_1_from_hjb_step,
        M_density_at_n_plus_1,
        problem,
        t_idx_n,
        backend,
        sigma_at_n,
    )
    if has_nan_or_inf(residual_F_U, backend):
        return U_n_current_newton_iterate, np.inf

    # Jacobian uses U_k_n_from_prev_picard for its H-part if problem provides specific terms
    jacobian_J_U = compute_hjb_jacobian(
        U_n_current_newton_iterate,  # For time/diffusion deriv
        U_k_n_from_prev_picard,  # For specific H-deriv
        M_density_at_n_plus_1,
        problem,
        t_idx_n,
        backend,
        sigma_at_n,
    )
    if np.any(np.isnan(jacobian_J_U.data)) or np.any(np.isinf(jacobian_J_U.data)):
        return U_n_current_newton_iterate, np.inf

    delta_U = np.zeros_like(U_n_current_newton_iterate)
    l2_error_of_step = np.inf
    try:
        if not jacobian_J_U.nnz > 0 and problem.Nx > 0:
            pass
        else:
            # Original notebook's RHS for Newton solve was effectively:
            # b = Jac * U_current_newton_iterate - Residual(U_current_newton_iterate)
            # And it solved Jac * U_next_newton_iterate = b
            # This is equivalent to Jac * delta_U = -Residual
            # where delta_U = U_next_newton_iterate - U_current_newton_iterate
            delta_U = sparse.linalg.spsolve(jacobian_J_U, -residual_F_U)

        if np.any(np.isnan(delta_U)) or np.any(np.isinf(delta_U)):
            delta_U = np.zeros_like(U_n_current_newton_iterate)
        else:
            l2_error_of_step = np.linalg.norm(delta_U) * np.sqrt(dx_norm)

    except Exception:
        pass

    max_delta_u_norm = 1e2
    current_delta_u_norm = np.linalg.norm(delta_U) * np.sqrt(dx_norm)
    if current_delta_u_norm > max_delta_u_norm and current_delta_u_norm > 1e-9:
        delta_U = delta_U * (max_delta_u_norm / current_delta_u_norm)
        l2_error_of_step = np.linalg.norm(delta_U) * np.sqrt(dx_norm)

    U_n_next_newton_iterate = U_n_current_newton_iterate + delta_U

    return U_n_next_newton_iterate, l2_error_of_step


def solve_hjb_timestep_newton(
    U_n_plus_1_from_hjb_step: np.ndarray,  # U_new[n+1] in notebook
    U_k_n_from_prev_picard: np.ndarray,  # U_k[n] in notebook
    M_density_at_n_plus_1: np.ndarray,  # M_k[n+1] in notebook
    problem: MFGProblem,
    max_newton_iterations: int | None = None,
    newton_tolerance: float | None = None,
    t_idx_n: int | None = None,  # time index for U_n being solved
    # Deprecated parameters for backward compatibility
    NiterNewton: int | None = None,
    l2errBoundNewton: float | None = None,
    backend: BaseBackend | None = None,
    sigma_at_n: float | np.ndarray | None = None,  # Diffusion at time t_n
) -> np.ndarray:
    """
    Solve HJB timestep using Newton's method.

    Args:
        U_n_plus_1_from_hjb_step: Solution at next time step
        U_k_n_from_prev_picard: Solution from previous Picard iteration
        M_density_at_n_plus_1: Density at next time step
        problem: MFG problem instance
        max_newton_iterations: Maximum Newton iterations (new parameter name)
        newton_tolerance: Newton convergence tolerance (new parameter name)
        t_idx_n: Time index for current solution
        NiterNewton: DEPRECATED - use max_newton_iterations
        l2errBoundNewton: DEPRECATED - use newton_tolerance
    """
    import warnings

    # Handle backward compatibility
    if NiterNewton is not None:
        warnings.warn(
            "Parameter 'NiterNewton' is deprecated. Use 'max_newton_iterations' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if max_newton_iterations is None:
            max_newton_iterations = NiterNewton

    if l2errBoundNewton is not None:
        warnings.warn(
            "Parameter 'l2errBoundNewton' is deprecated. Use 'newton_tolerance' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if newton_tolerance is None:
            newton_tolerance = l2errBoundNewton

    # Set defaults if still None
    if max_newton_iterations is None:
        max_newton_iterations = 30
    if newton_tolerance is None:
        newton_tolerance = 1e-6

    # Initial guess for Newton for U_n is U_{n+1} (from current HJB backward step)
    # Use backend-aware copy from compatibility layer
    U_n_current_newton_iterate = backend_aware_copy(U_n_plus_1_from_hjb_step, backend)

    if has_nan_or_inf(U_n_current_newton_iterate, backend):
        return U_n_current_newton_iterate

    final_l2_error = np.inf
    converged = False

    for iiter in range(max_newton_iterations):
        # Ensure t_idx_n is not None
        if t_idx_n is None:
            t_idx_n = 0  # Default time index

        U_n_next_newton_iterate, l2_error = newton_hjb_step(
            U_n_current_newton_iterate,
            U_n_plus_1_from_hjb_step,
            U_k_n_from_prev_picard,  # Pass U from prev Picard for Jacobian
            M_density_at_n_plus_1,
            problem,
            t_idx_n,
            backend,  # Pass backend for MPS/CUDA support
            sigma_at_n,  # Pass diffusion field
        )

        if has_nan_or_inf(U_n_next_newton_iterate, backend):
            break

        if iiter > 0 and l2_error > final_l2_error * 0.9999 and l2_error > newton_tolerance:
            break

        U_n_current_newton_iterate = U_n_next_newton_iterate
        final_l2_error = l2_error

        if l2_error < newton_tolerance:
            converged = True
            break

    if not converged and max_newton_iterations > 0 and not (np.isnan(final_l2_error) or np.isinf(final_l2_error)):
        pass

    return U_n_current_newton_iterate


def solve_hjb_system_backward(
    M_density_from_prev_picard: np.ndarray,  # M_k in notebook
    U_final_condition_at_T: np.ndarray,
    U_from_prev_picard: np.ndarray,  # U_k in notebook
    problem: MFGProblem,
    max_newton_iterations: int | None = None,
    newton_tolerance: float | None = None,
    # Deprecated parameters for backward compatibility
    NiterNewton: int | None = None,
    l2errBoundNewton: float | None = None,
    backend: BaseBackend | None = None,
    diffusion_field: float | np.ndarray | None = None,  # Diffusion field
) -> np.ndarray:
    """
    Solve HJB system backward in time using Newton's method.

    Args:
        M_density_from_prev_picard: Density from previous Picard iteration
        U_final_condition_at_T: Terminal condition for value function
        U_from_prev_picard: Value function from previous Picard iteration
        problem: MFG problem instance
        max_newton_iterations: Maximum Newton iterations (new parameter name)
        newton_tolerance: Newton convergence tolerance (new parameter name)
        NiterNewton: DEPRECATED - use max_newton_iterations
        l2errBoundNewton: DEPRECATED - use newton_tolerance
    """
    import warnings

    # Handle backward compatibility
    if NiterNewton is not None:
        warnings.warn(
            "Parameter 'NiterNewton' is deprecated. Use 'max_newton_iterations' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if max_newton_iterations is None:
            max_newton_iterations = NiterNewton

    if l2errBoundNewton is not None:
        warnings.warn(
            "Parameter 'l2errBoundNewton' is deprecated. Use 'newton_tolerance' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if newton_tolerance is None:
            newton_tolerance = l2errBoundNewton

    # Set defaults if still None
    if max_newton_iterations is None:
        max_newton_iterations = 30
    if newton_tolerance is None:
        newton_tolerance = 1e-6

    Nt = problem.Nt + 1
    Nx = problem.Nx + 1

    # Use backend.zeros() instead of xp.zeros() to ensure correct device
    if backend is not None:
        U_solution_this_picard_iter = backend.zeros((Nt, Nx))
    else:
        U_solution_this_picard_iter = np.zeros((Nt, Nx))  # U_new in notebook
    if Nt == 0:
        return U_solution_this_picard_iter

    # Use backend-aware nan/inf checking and assignment
    if has_nan_or_inf(U_final_condition_at_T, backend):
        backend_aware_assign(U_solution_this_picard_iter, (Nt - 1, slice(None)), float("nan"), backend)
    else:
        backend_aware_assign(U_solution_this_picard_iter, (Nt - 1, slice(None)), U_final_condition_at_T, backend)

    if Nt == 1:
        return U_solution_this_picard_iter

    for n_idx_hjb in range(Nt - 2, -1, -1):  # Solves for U_solution_this_picard_iter at t_idx_n = n_idx_hjb
        U_n_plus_1_current_picard = U_solution_this_picard_iter[n_idx_hjb + 1, :]

        # Backend-aware nan/inf checking
        if has_nan_or_inf(U_n_plus_1_current_picard, backend):
            backend_aware_assign(
                U_solution_this_picard_iter, (n_idx_hjb, slice(None)), U_n_plus_1_current_picard, backend
            )
            continue

        # BUG #7 FIX: Use M at the same time index as we're solving for U
        # Previously used M[n+1], now correctly uses M[n] to match HJB equation structure
        M_n_prev_picard = M_density_from_prev_picard[n_idx_hjb, :]  # M_k[n] - FIXED from n_idx_hjb+1
        if has_nan_or_inf(M_n_prev_picard, backend):
            backend_aware_assign(U_solution_this_picard_iter, (n_idx_hjb, slice(None)), float("nan"), backend)
            continue

        U_n_prev_picard = U_from_prev_picard[n_idx_hjb, :]  # U_k[n] from notebook
        if has_nan_or_inf(U_n_prev_picard, backend):
            backend_aware_assign(U_solution_this_picard_iter, (n_idx_hjb, slice(None)), float("nan"), backend)
            continue

        # Extract or evaluate diffusion using CoefficientField abstraction
        diffusion = CoefficientField(diffusion_field, problem.sigma, "diffusion_field", dimension=1)
        grid = get_spatial_grid(problem)
        sigma_at_n = diffusion.evaluate_at(timestep_idx=n_idx_hjb, grid=grid, density=M_n_prev_picard, dt=problem.dt)

        # Handle backend compatibility for NaN/Inf checking in callable results
        if diffusion.is_callable() and isinstance(sigma_at_n, np.ndarray):
            if has_nan_or_inf(sigma_at_n, backend):
                raise ValueError(f"Callable diffusion_field returned NaN/Inf at timestep {n_idx_hjb}")

        U_new_n = solve_hjb_timestep_newton(
            U_n_plus_1_current_picard,  # U_new[n+1]
            U_n_prev_picard,  # U_k[n] (for Jacobian)
            M_n_prev_picard,  # M_k[n] - FIXED: now uses correct time index
            problem,
            max_newton_iterations=max_newton_iterations,
            newton_tolerance=newton_tolerance,
            t_idx_n=n_idx_hjb,
            backend=backend,  # Pass backend for acceleration
            sigma_at_n=sigma_at_n,  # Pass diffusion at time n
        )
        backend_aware_assign(U_solution_this_picard_iter, (n_idx_hjb, slice(None)), U_new_n, backend)

        # Debug check for NaN introduction
        if has_nan_or_inf(U_solution_this_picard_iter[n_idx_hjb, :], backend) and not has_nan_or_inf(
            U_n_plus_1_current_picard, backend
        ):
            print(f"SYS_DEBUG: U_solution became NaN after Newton step for t_idx_n={n_idx_hjb}.")

    return U_solution_this_picard_iter


if __name__ == "__main__":
    """Quick smoke test for development."""
    print("Testing BaseHJBSolver...")

    # Test base class and helper functions availability
    assert BaseHJBSolver is not None
    assert compute_hjb_residual is not None
    assert compute_hjb_jacobian is not None
    assert newton_hjb_step is not None
    assert solve_hjb_timestep_newton is not None
    assert solve_hjb_system_backward is not None
    print("  Base HJB solver class and helpers available")

    # Test that BaseHJBSolver is abstract
    from mfg_pde import MFGProblem

    problem = MFGProblem(Nx=10, Nt=5, T=1.0, sigma=0.1)

    try:
        base_solver = BaseHJBSolver(problem)
        # Should fail because solve_hjb_system is abstract
        base_solver.solve_hjb_system(None, None, None)
        raise AssertionError("Should have raised NotImplementedError")
    except (TypeError, NotImplementedError):
        print("  BaseHJBSolver correctly abstract")

    print("Smoke tests passed!")
