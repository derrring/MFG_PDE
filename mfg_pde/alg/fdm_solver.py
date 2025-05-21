import numpy as np
import scipy as sc
import scipy.sparse as sparse
import scipy.sparse.linalg
import time
from ..core.base_solver import MFGSolver
from ..utils import hjb_utils  # Path to the new HJB utilities


class FDMSolver(MFGSolver):
    def __init__(self, problem, thetaUM=0.5, NiterNewton=30, l2errBoundNewton=1e-6):
        super().__init__(problem)
        self.thetaUM = thetaUM
        self.NiterNewton = NiterNewton  # Retain for passing to hjb_utils
        self.l2errBoundNewton = l2errBoundNewton  # Retain for passing
        self.U = None
        self.M = None
        self.l2distu = None
        self.l2distm = None
        self.l2disturel = None
        self.l2distmrel = None
        self.iterations_run = 0

    def _solveFP(self, m_init, u):  # m_init is initial density, u is current U solution
        # This function (Fokker-Planck solver) remains unchanged as it was unique.
        # Make sure all references like self.problem._npart still work.
        print("****** Solving FP (FDM)")
        Nx = self.problem.Nx
        Nt = self.problem.Nt  # Note: FP original loops 1 to Nt, M has Nt+1 entries
        Dx = self.problem.Dx
        Dt = self.problem.Dt
        sigma = self.problem.sigma
        coefCT = self.problem.coefCT

        m = np.zeros((Nt + 1, Nx))  # M needs Nt+1 time points (0 to T)
        m[0] = m_init

        # Original loop was range(1, Nt+1) filling m[k] from m[k-1] and u[k-1]
        # This means m has Nt+1 elements, from m[0] to m[Nt]
        for k_idx_fp in range(Nt):  # k_idx_fp from 0 to Nt-1 for time steps
            # m[k_idx_fp+1] will be density at t=(k_idx_fp+1)*Dt
            # u_km1 corresponds to u at time t=k_idx_fp*Dt
            u_at_tk = u[k_idx_fp]

            A_L = np.zeros(Nx)
            A_D = np.zeros(Nx)
            A_U = np.zeros(Nx)
            A_upcorner = np.zeros(Nx)
            A_lowcorner = np.zeros(Nx)

            A_D += 1.0 / Dt
            A_D += sigma**2 / Dx**2

            # Using u_at_tk for drifts
            A_D[1 : Nx - 1] += (
                coefCT
                * (
                    self.problem._npart(u_at_tk[2:Nx] - u_at_tk[1 : Nx - 1])
                    + self.problem._ppart(u_at_tk[1 : Nx - 1] - u_at_tk[0 : Nx - 2])
                )
                / Dx**2
            )
            A_D[0] += (
                coefCT
                * (
                    self.problem._npart(u_at_tk[1] - u_at_tk[0])
                    + self.problem._ppart(u_at_tk[0] - u_at_tk[Nx - 1])
                )
                / Dx**2
            )
            A_D[Nx - 1] += (
                coefCT
                * (
                    self.problem._npart(u_at_tk[0] - u_at_tk[Nx - 1])
                    + self.problem._ppart(u_at_tk[Nx - 1] - u_at_tk[Nx - 2])
                )
                / Dx**2
            )

            A_L_val_diff = -(sigma**2) / (2 * Dx**2)
            A_U_val_diff = -(sigma**2) / (2 * Dx**2)

            A_L[1:Nx] += A_L_val_diff
            A_L[1:Nx] += (
                -coefCT
                * self.problem._npart(u_at_tk[1:Nx] - u_at_tk[0 : Nx - 1])
                / Dx**2
            )

            A_U[0 : Nx - 1] += A_U_val_diff
            A_U[0 : Nx - 1] += (
                -coefCT
                * self.problem._ppart(u_at_tk[1:Nx] - u_at_tk[0 : Nx - 1])
                / Dx**2
            )

            # Periodic Corners (from original _solveFP)
            A_upcorner[0] = (
                A_U_val_diff
                - coefCT * self.problem._ppart(u_at_tk[1] - u_at_tk[0]) / Dx**2
            )
            A_lowcorner[Nx - 1] = (
                A_L_val_diff
                - coefCT * self.problem._npart(u_at_tk[0] - u_at_tk[Nx - 1]) / Dx**2
            )

            A_L_sp = np.roll(A_L, -1)
            A_U_sp = np.roll(A_U, 1)

            A_matrix = sparse.spdiags(
                [A_lowcorner, A_L_sp, A_D, A_U_sp, A_upcorner],
                [-(Nx - 1), -1, 0, 1, Nx - 1],
                Nx,
                Nx,
                format="csr",
            )  # Corrected offsets for periodic

            b_rhs = m[k_idx_fp] / Dt

            try:
                m[k_idx_fp + 1] = sparse.linalg.spsolve(A_matrix, b_rhs)
            except Exception as e:
                print(
                    f"Error solving linear system for FP at time step {k_idx_fp + 1}: {e}"
                )
                m[k_idx_fp + 1] = m[k_idx_fp]  # Fallback
        return m

    # REMOVE _getPhi_U, _getJacobianU, _solveHJB_NewtonStep, _solveHJBTimeStep (original methods)

    def _solveHJB(self, M_old_from_FP_iter):  # M_old_from_FP_iter is M_k from original
        print("****** Solving HJB (FDM via hjb_utils)")
        U_new_solution = hjb_utils.solve_hjb_system_backward(
            M_density_evolution_from_FP=M_old_from_FP_iter,
            U_final_condition_at_T=self.problem.get_final_u(),
            problem=self.problem,
            NiterNewton=self.NiterNewton,
            l2errBoundNewton=self.l2errBoundNewton,
        )
        return U_new_solution

    def solve(self, Niter, l2errBoundPicard=1e-5):
        print(
            f"\n________________ Solving MFG with FDM (T={self.problem.T}) _______________"
        )
        Nx = self.problem.Nx
        Nt = self.problem.Nt  # Total number of temporal knots for U and M (0 to Nt)
        Dx = self.problem.Dx
        Dt = self.problem.Dt

        # Initialization
        # U and M arrays should have Nt+1 time points (0, Dt, ..., T)
        self.U = np.zeros((Nt + 1, Nx))
        self.M = np.zeros((Nt + 1, Nx))

        initial_m_dist = self.problem.get_initial_m()
        self.M[0] = initial_m_dist  # M at t=0

        # Initialize U with final cost, M with initial density for t > 0 (as per original)
        final_u_cost = self.problem.get_final_u()
        for n_time_idx in range(Nt + 1):
            self.U[n_time_idx] = final_u_cost
            if n_time_idx > 0:
                self.M[n_time_idx] = initial_m_dist

        self.l2distu = np.ones(Niter)
        self.l2distm = np.ones(Niter)
        self.l2disturel = np.ones(Niter)
        self.l2distmrel = np.ones(Niter)
        self.iterations_run = 0

        for iiter in range(Niter):
            start_time_iter = time.time()
            print(
                f"\n******************** FDM Fixed-Point Iteration = {iiter + 1} / {Niter}"
            )

            U_old_iter = self.U.copy()
            M_old_iter = self.M.copy()

            # Solve HJB backward using M_old_iter
            U_new_tmp_hjb = self._solveHJB(M_old_iter)

            # Apply damping to U update
            self.U = self.thetaUM * U_new_tmp_hjb + (1 - self.thetaUM) * U_old_iter

            # Solve FP forward using the newly computed U
            # _solveFP expects m_init (at t=0) and the full U solution
            M_new_tmp_fp = self._solveFP(initial_m_dist, self.U)

            # Apply damping to M update
            self.M = self.thetaUM * M_new_tmp_fp + (1 - self.thetaUM) * M_old_iter

            # Convergence metrics
            self.l2distu[iiter] = np.linalg.norm(self.U - U_old_iter) * np.sqrt(Dx * Dt)
            norm_U_iter = np.linalg.norm(self.U) * np.sqrt(Dx * Dt)
            self.l2disturel[iiter] = (
                self.l2distu[iiter] / norm_U_iter
                if norm_U_iter > 1e-9
                else self.l2distu[iiter]
            )

            self.l2distm[iiter] = np.linalg.norm(self.M - M_old_iter) * np.sqrt(Dx * Dt)
            norm_M_iter = np.linalg.norm(self.M) * np.sqrt(Dx * Dt)
            self.l2distmrel[iiter] = (
                self.l2distm[iiter] / norm_M_iter
                if norm_M_iter > 1e-9
                else self.l2distm[iiter]
            )

            elapsed_time_iter = time.time() - start_time_iter
            print(
                f" === END Iteration {iiter+1}: ||u_new - u_old||_2 = {self.l2distu[iiter]:.4e} (rel: {self.l2disturel[iiter]:.4e})"
            )
            print(
                f" === END Iteration {iiter+1}: ||m_new - m_old||_2 = {self.l2distm[iiter]:.4e} (rel: {self.l2distmrel[iiter]:.4e})"
            )
            print(f" === Time for iteration = {elapsed_time_iter:.2f} s")

            self.iterations_run = iiter + 1
            if (
                self.l2disturel[iiter] < l2errBoundPicard
                and self.l2distmrel[iiter] < l2errBoundPicard
            ):
                print(f"Convergence reached after {iiter + 1} iterations.")
                break

        # Trim convergence arrays if needed
        self.l2distu = self.l2distu[: self.iterations_run]
        self.l2distm = self.l2distm[: self.iterations_run]
        self.l2disturel = self.l2disturel[: self.iterations_run]
        self.l2distmrel = self.l2distmrel[: self.iterations_run]

        return self.U, self.M, self.iterations_run, self.l2distu, self.l2distm

    def get_results(self):
        return self.U, self.M

    def get_convergence_data(self):
        return (
            self.iterations_run,
            self.l2distu,
            self.l2distm,
            self.l2disturel,
            self.l2distmrel,
        )
