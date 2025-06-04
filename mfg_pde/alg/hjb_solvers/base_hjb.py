from abc import ABC, abstractmethod
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg

# Assuming MFGProblem and ExampleMFGProblem are accessible, e.g.
from mfg_pde.core.mfg_problem import MFGProblem,ExampleMFGProblem # Adjust path as per actual structure


class BaseHJBSolver(ABC):
    def __init__(self, problem: "MFGProblem"):  # Use forward reference for MFGProblem
        self.problem = problem
        self.hjb_method_name = "BaseHJB"  # Concrete solvers will override this

    @abstractmethod
    def solve_hjb_system(
        self,
        M_density_evolution_from_FP: np.ndarray,
        U_final_condition_at_T: np.ndarray,
    ) -> np.ndarray:
        """
        Solves the full HJB system by marching backward in time.

        Args:
            M_density_evolution_from_FP (np.ndarray): (Nt+1, Nx) array of density m(t,x).
            U_final_condition_at_T (np.array): (Nx,) array for U(T,x).

        Returns:
            np.array: U_solution (Nt+1, Nx)
        """
        pass


def compute_hjb_residual(
    U_n_current_guess: np.ndarray,
    U_n_plus_1_known: np.ndarray,
    M_density_at_n_plus_1: np.ndarray,  # Density m( (n+1)*Dt, x )
    problem: "MFGProblem",
    t_idx_n_plus_1: int,  # Time index for M_density
) -> np.ndarray:
    """
    Calculates the residual of the HJB equation for the current guess U_n_current_guess.
    Equation form (discretized at time step n*Dt):
    (U_n - U_{n+1})/Dt - (sigma^2/2) * U_n_xx + H(x, m_{n+1}, Du_n, t_n) = 0

    Args:
        U_n_current_guess (np.ndarray): Current guess for U at time step n.
        U_n_plus_1_known (np.ndarray): Known U at time step n+1 (from previous HJB step).
        M_density_at_n_plus_1 (np.ndarray): Density distribution m to be used in H, typically M_old[n+1].
        problem (MFGProblem): The MFG problem instance.
        t_idx_n_plus_1 (int): The time index corresponding to M_density_at_n_plus_1 and U_n_plus_1_known.
                                The Hamiltonian might be evaluated at t_n or t_{n+1}.
                                Let's assume H is evaluated using m at n+1, and p from U_n.
                                The time argument for H itself could be t_n.
    """
    Nx = problem.Nx
    Dx = problem.Dx
    Dt = problem.Dt
    sigma = problem.sigma

    Phi_U = np.zeros(Nx)

    # Time derivative term: (U_n_current_guess - U_n_plus_1_known) / Dt
    Phi_U += (U_n_current_guess - U_n_plus_1_known) / Dt

    # Diffusion term: -(sigma^2 / 2.0) * d^2(U_n_current_guess)/dx^2 (Laplacian of U_n)
    # Periodic boundary conditions assumed for U_xx
    U_xx = (
        np.roll(U_n_current_guess, -1)
        - 2 * U_n_current_guess
        + np.roll(U_n_current_guess, 1)
    ) / Dx**2
    Phi_U += -(sigma**2 / 2.0) * U_xx

    # Hamiltonian H(x, m, p, t)
    # p is Du_n (gradient of U at current time step n)
    # m is M_density_at_n_plus_1 (density from previous fixed point iteration at time n+1)
    # t for H could be t_n (index t_idx_n_plus_1 - 1) or t_{n+1} (index t_idx_n_plus_1)
    # Let's assume t_idx_n_plus_1 - 1 for time t_n if H is time-dependent.
    t_idx_for_H = t_idx_n_plus_1 - 1 if t_idx_n_plus_1 > 0 else 0

    for i in range(Nx):
        # Calculate momentum approximations (Du) at spatial point i, for U_n_current_guess
        # Periodic boundary conditions for derivatives
        ip1 = (i + 1) % Nx
        im1 = (i - 1 + Nx) % Nx  # Ensures positive index

        p_forward = (U_n_current_guess[ip1] - U_n_current_guess[i]) / Dx
        p_backward = (U_n_current_guess[i] - U_n_current_guess[im1]) / Dx
        # p_centered = (U_n_current_guess[ip1] - U_n_current_guess[im1]) / (2 * Dx) # If needed

        p_values = {
            "forward": p_forward,
            "backward": p_backward,
        }  # Add 'centered': p_centered if used by H

        # Call the problem's Hamiltonian
        # M_density_at_n_plus_1[i] is m(x_i, t_{n+1})
        hamiltonian_val = problem.H(
            x_idx=i,
            m_at_x=M_density_at_n_plus_1[i],
            p_values=p_values,
            t_idx=t_idx_for_H,
        )  # Pass t_n index
        Phi_U[i] += hamiltonian_val

    return Phi_U


def compute_hjb_jacobian(
    U_n_current_guess: np.ndarray,
    M_density_at_n_plus_1: np.ndarray,
    problem: "MFGProblem",
    t_idx_n_plus_1: int,
) -> sparse.csr_matrix:
    """
    Calculates the Jacobian matrix for Newton's method for the HJB equation: d(Residual)/d(U_n_current_guess).
    The Hamiltonian's dependency on m is treated explicitly in the fixed point iteration,
    so m is considered fixed when taking derivatives w.r.t. U_n.

    Args:
        U_n_current_guess (np.array): Current guess for U at time step n.
        M_density_at_n_plus_1 (np.array): Density distribution m to be used in H, M_old[n+1].
        problem (MFGProblem): The MFG problem instance.
        t_idx_n_plus_1 (int): Time index for M_density and U_n_plus_1.
    """
    Nx = problem.Nx
    Dx = problem.Dx
    Dt = problem.Dt
    sigma = problem.sigma

    # For the ExampleMFGProblem, coefCT is used inside its H method.
    # If a more general Jacobian is needed, it might require dH/dp from the problem.
    # The current ExampleMFGProblem's H is non-linear in p_values.
    # A numerical Jacobian or a specific analytical one for ExampleMFGProblem would be more robust.
    # For now, let's replicate the structure of the original Jacobian from MFG-FDM-particle2.py,
    # which was specific to that H form.

    A_L = np.zeros(Nx)  # Lower diag: d(Res[i])/d(U[i-1])
    A_D = np.zeros(Nx)  # Main diag:  d(Res[i])/d(U[i])
    A_U = np.zeros(Nx)  # Upper diag: d(Res[i])/d(U[i+1])

    # 1. Derivative of time term (U_n - U_{n+1})/Dt w.r.t U_n[i]
    A_D += 1.0 / Dt

    # 2. Derivative of diffusion term -(sigma^2/2) * U_xx w.r.t U_n
    A_D += sigma**2 / Dx**2  # for -2*U_n[i] component
    A_L_val_diff = -(sigma**2) / (2 * Dx**2)  # for U_n[i-1] component
    A_U_val_diff = -(sigma**2) / (2 * Dx**2)  # for U_n[i+1] component

    # Fill for periodic: A_L affects main diagonal via roll for i=0, A_U for i=Nx-1
    # A_L[i] is dRes[i]/dU[i-1]. A_U[i] is dRes[i]/dU[i+1]
    # For i=0, dRes[0]/dU[Nx-1] goes to A_L[0] (if rolled) or corner.
    # For i=Nx-1, dRes[Nx-1]/dU[0] goes to A_U[Nx-1] (if rolled) or corner.

    # This is where the Jacobian becomes specific to ExampleMFGProblem's H
    # H = 0.5 * coefCT * (npart(p_fwd)^2 + ppart(p_bwd)^2) - V - m^2
    # p_fwd_i = (U[i+1]-U[i])/Dx
    # p_bwd_i = (U[i]-U[i-1])/Dx
    # Need dH_i / dU[i], dH_i / dU[i-1], dH_i / dU[i+1]
    # This is complex due to npart/ppart.
    # The original Jacobian in MFG-FDM-particle2.py was:
    # A_D[1:Nx-1] += coefCT * (problem._npart(U_old_n[2:Nx]-U_old_n[1:Nx-1]) + problem._ppart(U_old_n[1:Nx-1]-U_old_n[0:Nx-2]))/ (Dx**2)
    # A_L[1:Nx] += -coefCT * problem._ppart(U_old_n[1:Nx] - U_old_n[0:Nx-1]) / Dx**2
    # A_U[0:Nx-1] += coefCT * (-problem._npart(U_old_n[1:Nx]-U_old_n[0:Nx-1])) / (Dx**2)
    # This looks like a Jacobian for a *different* H, possibly related to finite volume or specific upwinding.
    # For the H in ExampleMFGProblem, a finite difference approximation of Jacobian might be safer
    # or an analytical derivation which will be more involved.

    # Let's use a simplified Jacobian assuming H is approximately quadratic in Du,
    # or use the structure from the original code if ExampleMFGProblem is the target.
    # Since ExampleMFGProblem is the target for now, we can try to adapt that structure.
    # The original Jacobian structure seems to be for an HJB where the Hamiltonian's derivative terms are explicit.

    # If we assume problem is ExampleMFGProblem for this Jacobian:
    if isinstance(
        problem, eval("ExampleMFGProblem")
    ):  # eval to avoid import error if run standalone
        coefCT = problem.coefCT
        # Diagonal contribution from H's dependence on U_n[i] (via p_forward[i-1], p_backward[i], p_forward[i], p_backward[i+1])
        # This is where the Jacobian from MFG-FDM-particle2.py was used. It's specific.
        # For U_n_current_guess (aliased as U_n below for brevity)
        U_n = U_n_current_guess

        # dH_i / dU_n[i]
        # p_fwd at i: (U[i+1]-U[i])/Dx -> d/dU[i] = -1/Dx * coefCT * npart(p_fwd) * npart'(p_fwd)
        # p_bwd at i: (U[i]-U[i-1])/Dx -> d/dU[i] =  1/Dx * coefCT * ppart(p_bwd) * ppart'(p_bwd)
        # This requires derivatives of npart and ppart (Heaviside/Dirac delta functions).
        # The Jacobian in MFG-FDM-particle2 was likely for a specific linearized or upwind scheme.

        # Given the difficulty of a general analytical Jacobian for the npart/ppart Hamiltonian,
        # and the specificity of the Jacobian in MFG-FDM-particle2.py,
        # for robustness, one might use a numerical Jacobian (e.g., with small perturbations).
        # However, to proceed with the structure provided previously:
        # The terms from MFG-FDM-particle2.py's Jacobian:
        A_D_H = np.zeros(Nx)
        A_L_H = np.zeros(Nx)
        A_U_H = np.zeros(Nx)

        for i in range(Nx):
            ip1 = (i + 1) % Nx
            im1 = (i - 1 + Nx) % Nx

            # Approx dH[i]/dU[i]
            # This is simplified. True derivative is more complex.
            # The original Jacobian terms were:
            # A_D[i] += coefCT * (problem._npart(U_n[ip1]-U_n[i]) + problem._ppart(U_n[i]-U_n[im1])) / (Dx**2) # This is not dH/dU

            # Let's use finite difference for Jacobian of H for now for simplicity, though less efficient
            eps = 1e-7
            p_fwd_base = (U_n[ip1] - U_n[i]) / Dx
            p_bwd_base = (U_n[i] - U_n[im1]) / Dx
            p_values_base = {"forward": p_fwd_base, "backward": p_bwd_base}
            H_base = problem.H(
                i,
                M_density_at_n_plus_1[i],
                p_values_base,
                t_idx_n_plus_1 - 1 if t_idx_n_plus_1 > 0 else 0,
            )

            # dH[i]/dU[i]
            U_n_eps_i = U_n.copy()
            U_n_eps_i[i] += eps
            p_fwd_eps_i = (U_n_eps_i[ip1] - U_n_eps_i[i]) / Dx
            p_bwd_eps_i = (U_n_eps_i[i] - U_n_eps_i[im1]) / Dx
            p_values_eps_i = {"forward": p_fwd_eps_i, "backward": p_bwd_eps_i}
            H_eps_i = problem.H(
                i,
                M_density_at_n_plus_1[i],
                p_values_eps_i,
                t_idx_n_plus_1 - 1 if t_idx_n_plus_1 > 0 else 0,
            )
            A_D_H[i] = (H_eps_i - H_base) / eps

            # dH[i]/dU[i-1] (affects A_L[i])
            if Nx > 1:  # Avoid if only one point
                U_n_eps_im1 = U_n.copy()
                U_n_eps_im1[im1] += eps
                # H depends on U[i-1] via p_bwd[i] and p_fwd[i-1]
                # For Res[i], we need dH[i]/dU[i-1]
                p_fwd_eps_im1_at_i = (
                    U_n[ip1] - U_n[i]
                ) / Dx  # p_fwd at i is not affected by U[i-1]
                p_bwd_eps_im1_at_i = (U_n[i] - U_n_eps_im1[im1]) / Dx
                p_values_eps_im1 = {
                    "forward": p_fwd_eps_im1_at_i,
                    "backward": p_bwd_eps_im1_at_i,
                }
                H_eps_im1 = problem.H(
                    i,
                    M_density_at_n_plus_1[i],
                    p_values_eps_im1,
                    t_idx_n_plus_1 - 1 if t_idx_n_plus_1 > 0 else 0,
                )
                A_L_H[i] = (H_eps_im1 - H_base) / eps  # This is dH_i / dU_{i-1}

            # dH[i]/dU[i+1] (affects A_U[i])
            if Nx > 1:
                U_n_eps_ip1 = U_n.copy()
                U_n_eps_ip1[ip1] += eps
                # H depends on U[i+1] via p_fwd[i] and p_bwd[i+1]
                # For Res[i], we need dH[i]/dU[i+1]
                p_fwd_eps_ip1_at_i = (U_n_eps_ip1[ip1] - U_n[i]) / Dx
                p_bwd_eps_ip1_at_i = (
                    U_n[i] - U_n[im1]
                ) / Dx  # p_bwd at i is not affected by U[i+1]
                p_values_eps_ip1 = {
                    "forward": p_fwd_eps_ip1_at_i,
                    "backward": p_bwd_eps_ip1_at_i,
                }
                H_eps_ip1 = problem.H(
                    i,
                    M_density_at_n_plus_1[i],
                    p_values_eps_ip1,
                    t_idx_n_plus_1 - 1 if t_idx_n_plus_1 > 0 else 0,
                )
                A_U_H[i] = (H_eps_ip1 - H_base) / eps  # This is dH_i / dU_{i+1}

        A_D += A_D_H
        A_L_combined = np.zeros(Nx)
        A_U_combined = np.zeros(Nx)
        A_L_combined[1:Nx] = A_L_val_diff  # Diffusion part for U[i-1] affecting Res[i]
        A_L_combined[0] = A_L_val_diff  # Periodic: dRes[0]/dU[Nx-1]
        A_L_combined += A_L_H  # Hamiltonian part for U[i-1] affecting Res[i]

        A_U_combined[0 : Nx - 1] = (
            A_U_val_diff  # Diffusion part for U[i+1] affecting Res[i]
        )
        A_U_combined[Nx - 1] = A_U_val_diff  # Periodic: dRes[Nx-1]/dU[0]
        A_U_combined += A_U_H  # Hamiltonian part for U[i+1] affecting Res[i]

        # Construct sparse matrix with periodic boundary conditions
        # A_L_combined[i] is dRes[i]/dU[i-1] (or dRes[0]/dU[Nx-1] for i=0)
        # A_U_combined[i] is dRes[i]/dU[i+1] (or dRes[Nx-1]/dU[0] for i=Nx-1)

        diagonals = [
            A_D,
            np.roll(A_L_combined, 1),
            np.roll(A_U_combined, -1),
        ]  # D, L (rolled), U (rolled)
        offsets = [0, -1, 1]

        # Add corner elements for periodic BC if not handled by roll for specific structure
        # For standard tridiagonal from spdiags([main, lower, upper], [0, -1, 1])
        # Jac_ij = D_i if j=i
        # Jac_ij = L_i if j=i-1
        # Jac_ij = U_i if j=i+1
        # For periodic, we need Jac[0,Nx-1] and Jac[Nx-1,0]

        # Corrected spdiags construction for standard indexing:
        # A_L_std[i] = dRes[i]/dU[i-1] -> goes on k=-1 diagonal at row i, col i-1
        # A_U_std[i] = dRes[i]/dU[i+1] -> goes on k=1 diagonal at row i, col i+1

        A_L_final = A_L_combined  # A_L_final[i] is dRes[i]/dU[i-1]
        A_U_final = A_U_combined  # A_U_final[i] is dRes[i]/dU[i+1]

        # For spdiags, the k-th diagonal has main_diag[j] at (j, j+k)
        # k=0: main_diag (A_D)
        # k=-1: lower_diag (A_L_final)
        # k=1: upper_diag (A_U_final)

        Jac = sparse.spdiags(
            [A_D, A_L_final, A_U_final], [0, -1, 1], Nx, Nx, format="csr"
        )

        # Manually set periodic corners if the spdiags doesn't handle it as desired
        if Nx > 1:
            Jac[0, Nx - 1] += A_L_final[0]  # dRes[0]/dU[Nx-1]
            Jac[Nx - 1, 0] += A_U_final[Nx - 1]  # dRes[Nx-1]/dU[0]

    else:  # Fallback for unknown problem type, or if a generic analytical Jacobian is too hard
        # This is a placeholder. A robust solution would require dH/dp from the problem.
        # Or use numerical differentiation for the H part of Jacobian more carefully.
        print(
            "Warning: Using a simplified Jacobian for HJB. May affect Newton convergence for complex Hamiltonians."
        )
        A_L_combined = np.zeros(Nx)
        A_L_combined += A_L_val_diff
        A_U_combined = np.zeros(Nx)
        A_U_combined += A_U_val_diff
        Jac = sparse.spdiags(
            [A_D, A_L_combined, A_U_combined], [0, -1, 1], Nx, Nx, format="csr"
        )
        if Nx > 1:  # Basic periodic for diffusion
            Jac[0, Nx - 1] += A_L_val_diff
            Jac[Nx - 1, 0] += A_U_val_diff

    return Jac.tocsr()


def newton_hjb_step(
    U_n_current_guess: np.ndarray,
    U_n_plus_1_known: np.ndarray,
    M_density_at_n_plus_1: np.ndarray,
    problem: "MFGProblem",
    t_idx_n_plus_1: int,
) -> tuple[np.ndarray, float]:
    """Performs one step of Newton's method for the HJB equation at a time slice."""
    Dx = problem.Dx

    residual_F_U = compute_hjb_residual(
        U_n_current_guess,
        U_n_plus_1_known,
        M_density_at_n_plus_1,
        problem,
        t_idx_n_plus_1,
    )

    jacobian_J_U = compute_hjb_jacobian(
        U_n_current_guess, M_density_at_n_plus_1, problem, t_idx_n_plus_1
    )

    try:
        delta_U = sparse.linalg.spsolve(jacobian_J_U, -residual_F_U)
    except Exception as e:
        print(f"Error solving Jacobian system in HJB Newton step: {e}")
        # Fallback: no update or a small step in direction of negative gradient (not implemented here)
        delta_U = np.zeros_like(U_n_current_guess)

    U_n_next_iteration = U_n_current_guess + delta_U
    # Error based on update step size, or norm of residual
    l2_error_of_step = np.linalg.norm(delta_U) * np.sqrt(Dx if Dx > 0 else 1.0)
    # l2_error_of_step = np.linalg.norm(residual_F_U) * np.sqrt(Dx if Dx > 0 else 1.0)

    return U_n_next_iteration, l2_error_of_step


def solve_hjb_timestep_newton(
    U_n_plus_1_known: np.ndarray,
    M_density_at_n_plus_1: np.ndarray,
    problem: "MFGProblem",
    NiterNewton: int,
    l2errBoundNewton: float,
    t_idx_n_plus_1: int,  # time index for U_{n+1} and M_{n+1}
) -> np.ndarray:
    """Solves HJB for one time slice U_n using Newton's method."""

    U_n_current_iteration = U_n_plus_1_known.copy()  # Initial guess for U_n

    converged = False
    final_l2_error = -1.0
    for iiter in range(NiterNewton):
        U_n_next_iteration, l2_error = newton_hjb_step(
            U_n_current_iteration,
            U_n_plus_1_known,
            M_density_at_n_plus_1,
            problem,
            t_idx_n_plus_1,
        )
        U_n_current_iteration = U_n_next_iteration
        final_l2_error = l2_error

        if l2_error < l2errBoundNewton:
            converged = True
            # print(f"Newton for HJB at t_idx={t_idx_n_plus_1-1} converged in {iiter+1} iterations. Error: {l2_error:.2e}")
            break

    if not converged:
        print(
            f"Warning: Newton method for HJB at t_idx={t_idx_n_plus_1-1} did NOT converge after {NiterNewton} iter. Final Error: {final_l2_error:.2e} (Bound: {l2errBoundNewton:.2e})"
        )

    return U_n_current_iteration


def solve_hjb_system_backward(
    M_density_evolution_from_FP: np.ndarray,
    U_final_condition_at_T: np.ndarray,
    problem: "MFGProblem",
    NiterNewton: int,
    l2errBoundNewton: float,
) -> np.ndarray:
    """
    Solves the full HJB system by marching backward in time.
    """
    Nt = problem.Nt
    Nx = problem.Nx

    U_solution = np.zeros(
        (Nt, Nx)
    )  # Corrected: Nt time points if Nt is number of knots
    # If Nt is number of intervals, then Nt+1 points.
    # User's mfg_problem.py has Nt as number of knots.
    # So U_solution should be (problem.Nt, problem.Nx)

    U_solution[problem.Nt - 1] = U_final_condition_at_T  # U(T,x) is at index Nt-1

    # Loop backward in time: n from Nt-2 down to 0 (for U_solution[n])
    # U_solution[n] is U at time t_n = n*Dt
    # U_solution[n+1] is U at time t_{n+1} = (n+1)*Dt
    # M_density_evolution_from_FP[n+1] is m at time t_{n+1}
    for n_idx_hjb in range(problem.Nt - 2, -1, -1):  # n_idx_hjb for U_n
        U_n_plus_1 = U_solution[n_idx_hjb + 1]

        # Density m to be used in H(x, Du_n, m)
        # M_density_evolution_from_FP has Nt time points.
        # M_density_evolution_from_FP[n_idx_hjb + 1] is m at t_{n+1}
        M_for_H_at_this_step = M_density_evolution_from_FP[n_idx_hjb + 1]

        U_solution[n_idx_hjb] = solve_hjb_timestep_newton(
            U_n_plus_1,
            M_for_H_at_this_step,
            problem,
            NiterNewton,
            l2errBoundNewton,
            t_idx_n_plus_1=n_idx_hjb
            + 1,  # This is the time index for U_{n+1} and M_{n+1}
        )
    return U_solution
