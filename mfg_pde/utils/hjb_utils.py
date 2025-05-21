# mfg_pde/utils/hjb_utils.py
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg

def compute_hjb_residual(U_n_current_guess, U_n_plus_1_known, M_density_for_H,
                        problem):
    """
    Calculates the residual of the HJB equation for the current guess U_n_current_guess.
    Equation form (discretized):
    (U_n - U_{n+1})/Dt - (sigma^2/2) * U_n_xx + H(x, Du_n, m_for_H) = 0
    where m_for_H is typically m from the previous fixed-point iteration, evaluated at time n or n+1.
    The original code used M_k[n+1] from the previous fixed point iteration.

    Args:
        U_n_current_guess (np.array): Current guess for U at time step n.
        U_n_plus_1_known (np.array): Known U at time step n+1 (from previous HJB step).
        M_density_for_H (np.array): Density distribution m to be used in H.
                                In the original code, this was M_old[n+1].
        problem (MFGProblem): The MFG problem instance.
    """
    Nx = problem.Nx
    Dx = problem.Dx
    Dt = problem.Dt
    sigma = problem.sigma

    Phi_U = np.zeros(Nx)

    # Time derivative term: (U_n_current_guess - U_n_plus_1_known) / Dt
    # This corresponds to -(U_n_plus_1_known - U_n_current_guess) / Dt in original
    Phi_U += (U_n_current_guess - U_n_plus_1_known) / Dt

    # Diffusion term: -(sigma^2 / 2.0) * d^2(U_n_current_guess)/dx^2
    diff_centered_U_n = (np.roll(U_n_current_guess, -1) - 2 * U_n_current_guess + np.roll(U_n_current_guess, 1)) / Dx**2
    Phi_U += - (sigma**2 / 2.0) * diff_centered_U_n

    # Hamiltonian H(x, p, m) and Coupling Term F(x,m)
    # problem.H already includes H_control(p) - F(x,m)
    # where F(x,m) = problem.f_potential[i] + M_density_for_H[i]**2
    for i in range(Nx):
        ip1 = (i + 1) % Nx
        im1 = (i - 1 + Nx) % Nx
        
        p1_U_n = (U_n_current_guess[ip1] - U_n_current_guess[i]) / Dx # Forward difference
        p2_U_n = (U_n_current_guess[i] - U_n_current_guess[im1]) / Dx # Backward difference
        
        # M_density_for_H is the full spatial distribution. problem.H takes this.
        Phi_U[i] += problem.H(M_density_for_H, i, p1_U_n, p2_U_n)
        
    return Phi_U

def compute_hjb_jacobian(U_n_current_guess, problem):
    """
    Calculates the Jacobian matrix for Newton's method for the HJB equation.
    The Jacobian is d(Residual)/d(U_n_current_guess).
    M_density_for_H is not used in the original Jacobian for H, as H's m-dependency
    is treated explicitly in the fixed point iteration.

    Args:
        U_n_current_guess (np.array): Current guess for U at time step n.
        problem (MFGProblem): The MFG problem instance.
    """
    Nx = problem.Nx
    Dx = problem.Dx
    Dt = problem.Dt
    sigma = problem.sigma
    coefCT = problem.coefCT

    A_L = np.zeros(Nx) # Lower diag
    A_D = np.zeros(Nx) # Main diag
    A_U = np.zeros(Nx) # Upper diag

    # Derivative of time term (U_n_current_guess - U_n_plus_1_known)/Dt w.r.t U_n_current_guess[i]
    A_D += 1.0 / Dt
    
    # Derivative of diffusion term -(sigma^2/2) * U_xx w.r.t U_n_current_guess
    A_D += sigma**2 / Dx**2 # for -2*U_n_current_guess[i] component
    A_L_val = -sigma**2 / (2 * Dx**2)
    A_U_val = -sigma**2 / (2 * Dx**2)
    
    A_L[1:] += A_L_val   # For U_n_current_guess[i-1] component (A_L[0] is for Jac[0,-1] if periodic)
    A_U[:-1] += A_U_val # For U_n_current_guess[i+1] component (A_U[Nx-1] is for Jac[Nx-1,Nx] if periodic)

    # For periodic BC, these off-diagonal elements affect the corners.
    # The original implementation folded these into A_D for i=0, Nx-1 or handled
    # them implicitly by the structure of the Hamiltonian derivative terms.
    # The Hamiltonian part's Jacobian from original code:
    U_for_H_jac = U_n_current_guess # Alias for clarity
    
    # Diagonal contributions from H
    A_D[1:Nx-1] += coefCT * (problem._npart(U_for_H_jac[2:Nx]-U_for_H_jac[1:Nx-1]) + \
                        problem._ppart(U_for_H_jac[1:Nx-1]-U_for_H_jac[0:Nx-2])) / (Dx**2)
    A_D[Nx-1] += coefCT * (problem._npart(U_for_H_jac[0]-U_for_H_jac[Nx-1]) + \
                        problem._ppart(U_for_H_jac[Nx-1]-U_for_H_jac[Nx-2])) / (Dx**2) # Periodic
    A_D[0] += coefCT * (problem._npart(U_for_H_jac[1]-U_for_H_jac[0]) + \
                        problem._ppart(U_for_H_jac[0]-U_for_H_jac[Nx-1])) / (Dx**2) # Periodic

    # Off-diagonal contributions from H (LHS: U[i-1], RHS: U[i+1])
    # dH/d(U[i-1]) terms add to A_L[i]
    A_L[1:Nx] += -coefCT * problem._ppart(U_for_H_jac[1:Nx] - U_for_H_jac[0:Nx-1]) / Dx**2
    # dH/d(U[i+1]) terms add to A_U[i]
    A_U[0:Nx-1] += coefCT * (-problem._npart(U_for_H_jac[1:Nx]-U_for_H_jac[0:Nx-1])) / (Dx**2)

    # Construct sparse matrix (tridiagonal based on original HJB Jacobian structure)
    # For truly periodic Jacobian from Hamiltonian, corners Jac[0,Nx-1] and Jac[Nx-1,0] would also be needed.
    # The original HJB Jacobian appears to effectively create a tridiagonal matrix by how A_D, A_L, A_U are populated.
    # A_L[0] and A_U[Nx-1] correspond to Jac[0,Nx-1] and Jac[Nx-1,0] in a periodic sense if using offsets [-1,0,1] and rolling.
    # However, the original did not explicitly make the Jacobian matrix periodic with corner elements in the spdiags call for HJB.
    # It used: sparse.spdiags([np.roll(A_L, -1), A_D, np.roll(A_U, 1)], [-1, 0, 1], Nx, Nx, format='csr')
    # Let's use the direct A_L, A_U for spdiags [-1,0,1] as this is standard for tridiagonal.
    # The values at A_L[0] and A_U[Nx-1] will be used if the matrix is considered periodic and constructed with specific corner terms.
    # The formulation of A_D[0] and A_D[Nx-1] already incorporates the periodic U terms.

    Jac = sparse.spdiags([A_L, A_D, A_U], [-1, 0, 1], Nx, Nx, format='csr')
    
    return Jac

def newton_hjb_step(U_n_current_guess, U_n_plus_1_known, M_density_for_H,
                    problem):
    """Performs one step of Newton's method for the HJB equation at a time slice."""
    Dx = problem.Dx
    
    residual_F_U = compute_hjb_residual(U_n_current_guess, U_n_plus_1_known,
                                        M_density_for_H, problem)
    
    jacobian_J_U = compute_hjb_jacobian(U_n_current_guess, problem)
    
    try:
        delta_U = sparse.linalg.spsolve(jacobian_J_U, -residual_F_U)
    except Exception as e:
        print(f"Error solving Jacobian system in HJB Newton step: {e}")
        # Fallback: no update or a small step in direction of negative gradient (not implemented here)
        delta_U = np.zeros_like(U_n_current_guess)
        
    U_n_next_iteration = U_n_current_guess + delta_U
    l2_error_of_step = np.linalg.norm(delta_U) * np.sqrt(Dx) # Error based on update step size
    
    return U_n_next_iteration, l2_error_of_step

def solve_hjb_timestep_newton(U_n_plus_1_known, M_density_for_H,
                            problem, NiterNewton, l2errBoundNewton):
    """Solves HJB for one time slice U_n using Newton's method."""
    
    # Initial guess for U_n (value at current time step n)
    # Original code used U_n_plus_1_known.copy() as the initial guess.
    U_n_current_iteration = U_n_plus_1_known.copy()
    
    converged = False
    for iiter in range(NiterNewton):
        U_n_next_iteration, l2_error = newton_hjb_step(U_n_current_iteration,
                                                    U_n_plus_1_known,
                                                    M_density_for_H,
                                                    problem)
        U_n_current_iteration = U_n_next_iteration
        
        if l2_error < l2errBoundNewton:
            converged = True
            break
            
    if not converged:
        print(f"Warning: Newton method for HJB did not converge at this time slice. Error: {l2_error:.2e} (Bound: {l2errBoundNewton:.2e})")
        
    return U_n_current_iteration

def solve_hjb_system_backward(M_density_evolution_from_FP, U_final_condition_at_T,
                            problem, NiterNewton, l2errBoundNewton):
    """
    Solves the full HJB system by marching backward in time.

    Args:
        M_density_evolution_from_FP (np.array): (Nt+1, Nx) array of density m(t,x)
                                                from the previous fixed-point iteration.
        U_final_condition_at_T (np.array): (Nx,) array for U(T,x).
        problem (MFGProblem): The MFG problem instance.
        NiterNewton (int): Max iterations for Newton's method per time slice.
        l2errBoundNewton (float): Convergence tolerance for Newton's method.
    """
    Nt = problem.Nt
    Nx = problem.Nx

    U_solution = np.zeros((Nt + 1, Nx))
    U_solution[Nt] = U_final_condition_at_T # Set U(T,x)

    # Loop backward in time: n from Nt-1 down to 0
    for n in range(Nt - 1, -1, -1):
        U_n_plus_1 = U_solution[n + 1] # This is U((n+1)*Dt, x)
        
        # Density m to be used in H(x, Du_n, m)
        # Original code used M_old[n+1] (density at time (n+1)*Dt from previous FP iter)
        M_for_H_at_this_step = M_density_evolution_from_FP[n + 1]
        
        U_solution[n] = solve_hjb_timestep_newton(U_n_plus_1,
                                                M_for_H_at_this_step,
                                                problem,
                                                NiterNewton,
                                                l2errBoundNewton)
    return U_solution