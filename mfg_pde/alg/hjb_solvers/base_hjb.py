from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, TYPE_CHECKING

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg

if TYPE_CHECKING:
    from mfg_pde.core.mfg_problem import MFGProblem

    # from mfg_pde.utils.aux_func import npart, ppart # Not needed here if problem provides jacobian parts

# Clipping limit for p_values ONLY when using numerical FD for Jacobian H-part (fallback)
P_VALUE_CLIP_LIMIT_FD_JAC = 1e6


class BaseHJBSolver(ABC):
    def __init__(self, problem: "MFGProblem"):
        self.problem = problem
        self.hjb_method_name = "BaseHJB"

    @abstractmethod
    def solve_hjb_system(
        self,
        M_density_evolution_from_FP: np.ndarray,
        U_final_condition_at_T: np.ndarray,
        U_from_prev_picard: np.ndarray,  # Added: U from previous Picard iteration
    ) -> np.ndarray:
        pass


def _calculate_p_values(
    U_array: np.ndarray,
    i: int,
    Dx: float,
    Nx: int,
    clip: bool = False,
    clip_limit: float = P_VALUE_CLIP_LIMIT_FD_JAC,
) -> Dict[str, float]:
    # (Implementation from base_hjb_v4 - no p-value clipping by default)
    p_forward, p_backward = 0.0, 0.0
    if Nx > 1 and abs(Dx) > 1e-14:
        u_i = U_array[i]
        u_ip1 = U_array[(i + 1) % Nx]
        u_im1 = U_array[(i - 1 + Nx) % Nx]

        if (
            np.isinf(u_i)
            or np.isinf(u_ip1)
            or np.isinf(u_im1)
            or np.isnan(u_i)
            or np.isnan(u_ip1)
            or np.isnan(u_im1)
        ):
            return {"forward": np.nan, "backward": np.nan}

        p_forward = (u_ip1 - u_i) / Dx
        p_backward = (u_i - u_im1) / Dx

        if np.isinf(p_forward) or np.isnan(p_forward):
            p_forward = np.nan
        if np.isinf(p_backward) or np.isnan(p_backward):
            p_backward = np.nan

    elif Nx == 1:
        return {"forward": 0.0, "backward": 0.0}
    else:
        return {"forward": np.nan, "backward": np.nan}

    raw_p_values = {"forward": p_forward, "backward": p_backward}
    return _clip_p_values(raw_p_values, clip_limit) if clip else raw_p_values


def _clip_p_values(
    p_values: Dict[str, float], clip_limit: float
) -> Dict[str, float]:  # Helper for FD Jac
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
    problem: "MFGProblem",
    t_idx_n: int,  # Time index for U_n
) -> np.ndarray:
    Nx = problem.Nx + 1
    Dx = problem.Dx
    Dt = problem.Dt
    sigma = problem.sigma
    Phi_U = np.zeros(Nx)

    if np.any(np.isnan(U_n_current_newton_iterate)) or np.any(
        np.isinf(U_n_current_newton_iterate)
    ):
        return np.full(Nx, np.nan)

    # Time derivative: (U_n_current - U_{n+1})/Dt
    # Notebook FnU[i] += -(Ukp1_np1[i] - Ukp1_n[i])/Dt;  (U_n - U_{n+1})/Dt
    if abs(Dt) < 1e-14:
        if not np.allclose(
            U_n_current_newton_iterate, U_n_plus_1_from_hjb_step, rtol=1e-9, atol=1e-9
        ):
            pass
    else:
        time_deriv_term = (U_n_current_newton_iterate - U_n_plus_1_from_hjb_step) / Dt
        if np.any(np.isinf(time_deriv_term)) or np.any(np.isnan(time_deriv_term)):
            Phi_U[:] = np.nan
            return Phi_U
        Phi_U += time_deriv_term

    # Diffusion term: -(sigma^2/2) * (U_n_current)_xx
    # Notebook FnU[i] += - ((sigma**2)/2.) * (Ukp1_n[i+1]-2*Ukp1_n[i]+Ukp1_n[i-1])/(Dx**2)
    if abs(Dx) > 1e-14 and Nx > 1:
        U_xx = (
            np.roll(U_n_current_newton_iterate, -1)
            - 2 * U_n_current_newton_iterate
            + np.roll(U_n_current_newton_iterate, 1)
        ) / Dx**2
        if np.any(np.isinf(U_xx)) or np.any(np.isnan(U_xx)):
            Phi_U[:] = np.nan
            return Phi_U
        Phi_U += -(sigma**2 / 2.0) * U_xx

    # For m-coupling term, original notebook passed gradUkn, gradUknim1 (from prev Picard iter)
    # but mdmH_withM itself didn't use them. We'll pass an empty dict for now.
    U_n_derivatives_for_m_coupling = {}  # Not used by ExampleMFGProblem's term

    for i in range(Nx):
        if np.isnan(Phi_U[i]):
            continue

        # For Hamiltonian, use unclipped p_values derived from U_n_current_newton_iterate
        p_values = _calculate_p_values(
            U_n_current_newton_iterate, i, Dx, Nx, clip=False
        )

        if np.any(np.isnan(list(p_values.values()))):
            Phi_U[i] = np.nan
            continue

        # Hamiltonian term H(x_i, M_{n+1,i}, (Du_n_current)_i, t_n)
        # Notebook: FnU[i] += H_withM(...)
        hamiltonian_val = problem.H(
            x_idx=i, m_at_x=M_density_at_n_plus_1[i], p_values=p_values, t_idx=t_idx_n
        )
        if np.isnan(hamiltonian_val) or np.isinf(hamiltonian_val):
            Phi_U[i] = np.nan
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
                Phi_U[i] = np.nan
                continue
            Phi_U[i] += m_coupling_term

    return Phi_U


def compute_hjb_jacobian(
    U_n_current_newton_iterate: np.ndarray,  # U_new_n_tmp in notebook Newton step
    U_k_n_from_prev_picard: np.ndarray,  # U_k_n (or Uoldn) in notebook Jacobian
    M_density_at_n_plus_1: np.ndarray,
    problem: "MFGProblem",
    t_idx_n: int,
) -> sparse.csr_matrix:
    Nx = problem.Nx + 1
    Dx = problem.Dx
    Dt = problem.Dt
    sigma = problem.sigma
    eps = 1e-7

    J_D = np.zeros(Nx)
    J_L = np.zeros(Nx)
    J_U = np.zeros(Nx)

    if np.any(np.isnan(U_n_current_newton_iterate)) or np.any(
        np.isinf(U_n_current_newton_iterate)
    ):
        return sparse.diags([np.full(Nx, np.nan)], [0], shape=(Nx, Nx), format="csr")

    # Time derivative part: d/dU_n_current[j] of (U_n_current[i] - U_{n+1}[i])/Dt
    if abs(Dt) > 1e-14:
        J_D += 1.0 / Dt

    # Diffusion part: d/dU_n_current[j] of -(sigma^2/2) * (U_n_current)_xx[i]
    if abs(Dx) > 1e-14 and Nx > 1:
        J_D += sigma**2 / Dx**2
        val_off_diag_diff = -(sigma**2) / (2 * Dx**2)
        J_L += val_off_diag_diff
        J_U += val_off_diag_diff

    # Hamiltonian part & m-coupling term's Jacobian contribution
    # Try to get analytical/specific Jacobian contributions from the problem for H-part
    # Crucially, pass U_k_n_from_prev_picard for ExampleMFGProblem's specific Jacobian
    hamiltonian_jac_contrib = problem.get_hjb_hamiltonian_jacobian_contrib(
        U_k_n_from_prev_picard, t_idx_n  # This is Uoldn from original notebook
    )

    if hamiltonian_jac_contrib is not None:
        J_D_H, J_L_H, J_U_H = hamiltonian_jac_contrib
        J_D += J_D_H
        J_L += J_L_H
        J_U += J_U_H
    else:
        # Fallback to numerical Jacobian for H-part, using U_n_current_newton_iterate
        for i in range(Nx):
            U_perturbed_p_i = U_n_current_newton_iterate.copy()
            U_perturbed_p_i[i] += eps
            U_perturbed_m_i = U_n_current_newton_iterate.copy()
            U_perturbed_m_i[i] -= eps

            pv_p_i = _calculate_p_values(
                U_perturbed_p_i,
                i,
                Dx,
                Nx,
                clip=True,
                clip_limit=P_VALUE_CLIP_LIMIT_FD_JAC,
            )
            pv_m_i = _calculate_p_values(
                U_perturbed_m_i,
                i,
                Dx,
                Nx,
                clip=True,
                clip_limit=P_VALUE_CLIP_LIMIT_FD_JAC,
            )

            H_p_i = np.nan
            H_m_i = np.nan
            if not (np.any(np.isnan(list(pv_p_i.values())))):
                H_p_i = problem.H(i, M_density_at_n_plus_1[i], pv_p_i, t_idx_n)
            if not (np.any(np.isnan(list(pv_m_i.values())))):
                H_m_i = problem.H(i, M_density_at_n_plus_1[i], pv_m_i, t_idx_n)

            if not (
                np.isnan(H_p_i) or np.isnan(H_m_i) or np.isinf(H_p_i) or np.isinf(H_m_i)
            ):
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
                pv_p_im1 = _calculate_p_values(
                    U_perturbed_p_im1,
                    i,
                    Dx,
                    Nx,
                    clip=True,
                    clip_limit=P_VALUE_CLIP_LIMIT_FD_JAC,
                )
                pv_m_im1 = _calculate_p_values(
                    U_perturbed_m_im1,
                    i,
                    Dx,
                    Nx,
                    clip=True,
                    clip_limit=P_VALUE_CLIP_LIMIT_FD_JAC,
                )
                H_p_im1 = np.nan
                H_m_im1 = np.nan
                if not (np.any(np.isnan(list(pv_p_im1.values())))):
                    H_p_im1 = problem.H(i, M_density_at_n_plus_1[i], pv_p_im1, t_idx_n)
                if not (np.any(np.isnan(list(pv_m_im1.values())))):
                    H_m_im1 = problem.H(i, M_density_at_n_plus_1[i], pv_m_im1, t_idx_n)
                if not (
                    np.isnan(H_p_im1)
                    or np.isnan(H_m_im1)
                    or np.isinf(H_p_im1)
                    or np.isinf(H_m_im1)
                ):
                    J_L[i] += (H_p_im1 - H_m_im1) / (2 * eps)
                else:
                    J_L[i] += 0

                ip1 = (i + 1) % Nx
                U_perturbed_p_ip1 = U_n_current_newton_iterate.copy()
                U_perturbed_p_ip1[ip1] += eps
                U_perturbed_m_ip1 = U_n_current_newton_iterate.copy()
                U_perturbed_m_ip1[ip1] -= eps
                pv_p_ip1 = _calculate_p_values(
                    U_perturbed_p_ip1,
                    i,
                    Dx,
                    Nx,
                    clip=True,
                    clip_limit=P_VALUE_CLIP_LIMIT_FD_JAC,
                )
                pv_m_ip1 = _calculate_p_values(
                    U_perturbed_m_ip1,
                    i,
                    Dx,
                    Nx,
                    clip=True,
                    clip_limit=P_VALUE_CLIP_LIMIT_FD_JAC,
                )
                H_p_ip1 = np.nan
                H_m_ip1 = np.nan
                if not (np.any(np.isnan(list(pv_p_ip1.values())))):
                    H_p_ip1 = problem.H(i, M_density_at_n_plus_1[i], pv_p_ip1, t_idx_n)
                if not (np.any(np.isnan(list(pv_m_ip1.values())))):
                    H_m_ip1 = problem.H(i, M_density_at_n_plus_1[i], pv_m_ip1, t_idx_n)
                if not (
                    np.isnan(H_p_ip1)
                    or np.isnan(H_m_ip1)
                    or np.isinf(H_p_ip1)
                    or np.isinf(H_m_ip1)
                ):
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
    except ValueError as e:
        fallback_diag = np.ones(Nx) * (1.0 / Dt if abs(Dt) > 1e-14 else 1.0)
        Jac = sparse.diags([fallback_diag], [0], shape=(Nx, Nx), format="csr")

    return Jac.tocsr()


def newton_hjb_step(
    U_n_current_newton_iterate: np.ndarray,  # U_new_n_tmp in notebook
    U_n_plus_1_from_hjb_step: np.ndarray,  # U_new_np1 in notebook
    U_k_n_from_prev_picard: np.ndarray,  # U_k_n in notebook
    M_density_at_n_plus_1: np.ndarray,
    problem: "MFGProblem",
    t_idx_n: int,
) -> tuple[np.ndarray, float]:
    Dx_norm = problem.Dx if abs(problem.Dx) > 1e-12 else 1.0

    if np.any(np.isnan(U_n_current_newton_iterate)) or np.any(
        np.isinf(U_n_current_newton_iterate)
    ):
        return U_n_current_newton_iterate, np.inf

    residual_F_U = compute_hjb_residual(
        U_n_current_newton_iterate,
        U_n_plus_1_from_hjb_step,
        M_density_at_n_plus_1,
        problem,
        t_idx_n,
    )
    if np.any(np.isnan(residual_F_U)) or np.any(np.isinf(residual_F_U)):
        return U_n_current_newton_iterate, np.inf

    # Jacobian uses U_k_n_from_prev_picard for its H-part if problem provides specific terms
    jacobian_J_U = compute_hjb_jacobian(
        U_n_current_newton_iterate,  # For time/diffusion deriv
        U_k_n_from_prev_picard,  # For specific H-deriv
        M_density_at_n_plus_1,
        problem,
        t_idx_n,
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
            l2_error_of_step = np.linalg.norm(delta_U) * np.sqrt(Dx_norm)

    except Exception as e:
        pass

    max_delta_u_norm = 1e2
    current_delta_u_norm = np.linalg.norm(delta_U) * np.sqrt(Dx_norm)
    if current_delta_u_norm > max_delta_u_norm and current_delta_u_norm > 1e-9:
        delta_U = delta_U * (max_delta_u_norm / current_delta_u_norm)
        l2_error_of_step = np.linalg.norm(delta_U) * np.sqrt(Dx_norm)

    U_n_next_newton_iterate = U_n_current_newton_iterate + delta_U

    return U_n_next_newton_iterate, l2_error_of_step


def solve_hjb_timestep_newton(
    U_n_plus_1_from_hjb_step: np.ndarray,  # U_new[n+1] in notebook
    U_k_n_from_prev_picard: np.ndarray,  # U_k[n] in notebook
    M_density_at_n_plus_1: np.ndarray,  # M_k[n+1] in notebook
    problem: "MFGProblem",
    max_newton_iterations: int = None,
    newton_tolerance: float = None,
    t_idx_n: int = None,  # time index for U_n being solved
    # Deprecated parameters for backward compatibility
    NiterNewton: int = None,
    l2errBoundNewton: float = None,
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
    U_n_current_newton_iterate = U_n_plus_1_from_hjb_step.copy()

    if np.any(np.isnan(U_n_current_newton_iterate)) or np.any(
        np.isinf(U_n_current_newton_iterate)
    ):
        return U_n_current_newton_iterate

    final_l2_error = np.inf
    converged = False

    for iiter in range(max_newton_iterations):
        U_n_next_newton_iterate, l2_error = newton_hjb_step(
            U_n_current_newton_iterate,
            U_n_plus_1_from_hjb_step,
            U_k_n_from_prev_picard,  # Pass U from prev Picard for Jacobian
            M_density_at_n_plus_1,
            problem,
            t_idx_n,
        )

        if np.any(np.isnan(U_n_next_newton_iterate)) or np.any(
            np.isinf(U_n_next_newton_iterate)
        ):
            break

        if (
            iiter > 0
            and l2_error > final_l2_error * 0.9999
            and l2_error > newton_tolerance
        ):
            break

        U_n_current_newton_iterate = U_n_next_newton_iterate
        final_l2_error = l2_error

        if l2_error < newton_tolerance:
            converged = True
            break

    if (
        not converged
        and max_newton_iterations > 0
        and not (np.isnan(final_l2_error) or np.isinf(final_l2_error))
    ):
        pass

    return U_n_current_newton_iterate


def solve_hjb_system_backward(
    M_density_from_prev_picard: np.ndarray,  # M_k in notebook
    U_final_condition_at_T: np.ndarray,
    U_from_prev_picard: np.ndarray,  # U_k in notebook
    problem: "MFGProblem",
    max_newton_iterations: int = None,
    newton_tolerance: float = None,
    # Deprecated parameters for backward compatibility
    NiterNewton: int = None,
    l2errBoundNewton: float = None,
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

    U_solution_this_picard_iter = np.zeros((Nt, Nx))  # U_new in notebook
    if Nt == 0:
        return U_solution_this_picard_iter

    if np.any(np.isnan(U_final_condition_at_T)) or np.any(
        np.isinf(U_final_condition_at_T)
    ):
        U_solution_this_picard_iter[Nt - 1, :] = np.nan
    else:
        U_solution_this_picard_iter[Nt - 1, :] = U_final_condition_at_T

    if Nt == 1:
        return U_solution_this_picard_iter

    for n_idx_hjb in range(
        Nt - 2, -1, -1
    ):  # Solves for U_solution_this_picard_iter at t_idx_n = n_idx_hjb
        U_n_plus_1_current_picard = U_solution_this_picard_iter[n_idx_hjb + 1, :]

        if np.any(np.isnan(U_n_plus_1_current_picard)) or np.any(
            np.isinf(U_n_plus_1_current_picard)
        ):
            U_solution_this_picard_iter[n_idx_hjb, :] = U_n_plus_1_current_picard
            continue

        M_n_plus_1_prev_picard = M_density_from_prev_picard[n_idx_hjb + 1, :]
        if np.any(np.isnan(M_n_plus_1_prev_picard)) or np.any(
            np.isinf(M_n_plus_1_prev_picard)
        ):
            U_solution_this_picard_iter[n_idx_hjb, :] = np.nan
            continue

        U_n_prev_picard = U_from_prev_picard[n_idx_hjb, :]  # U_k[n] from notebook
        if np.any(np.isnan(U_n_prev_picard)) or np.any(np.isinf(U_n_prev_picard)):
            U_solution_this_picard_iter[n_idx_hjb, :] = np.nan
            continue

        U_solution_this_picard_iter[n_idx_hjb, :] = solve_hjb_timestep_newton(
            U_n_plus_1_current_picard,  # U_new[n+1]
            U_n_prev_picard,  # U_k[n] (for Jacobian)
            M_n_plus_1_prev_picard,  # M_k[n+1]
            problem,
            max_newton_iterations=max_newton_iterations,
            newton_tolerance=newton_tolerance,
            t_idx_n=n_idx_hjb,
        )
        if np.any(np.isnan(U_solution_this_picard_iter[n_idx_hjb, :])) and not np.any(
            np.isnan(U_n_plus_1_current_picard)
        ):
            print(
                f"SYS_DEBUG: U_solution became NaN after Newton step for t_idx_n={n_idx_hjb}."
            )

    return U_solution_this_picard_iter
