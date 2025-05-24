import numpy as np
import time
from .base_mfg_solver import MFGSolver 


class FixedPointIterator(MFGSolver):
    def __init__(self, problem, hjb_solver, fp_solver, thetaUM=0.5):
        super().__init__(problem) # MFGSolver init takes problem
        self.hjb_solver = hjb_solver
        self.fp_solver = fp_solver
        self.thetaUM = thetaUM

        # Construct a descriptive name
        self.name = f"HJB-{self.hjb_solver.hjb_method_name}_FP-{self.fp_solver.fp_method_name}"

        self.U = None
        self.M = None
        self.l2distu = None
        self.l2distm = None
        self.l2disturel = None
        self.l2distmrel = None
        self.iterations_run = 0

    def solve(self, Niter, l2errBoundPicard=1e-5):
        print(f"\n________________ Solving MFG with {self.name} (T={self.problem.T}) _______________")
        Nx = self.problem.Nx
        Nt = self.problem.Nt
        Dx = self.problem.Dx
        Dt = self.problem.Dt

        self.U = np.zeros((Nt + 1, Nx))
        self.M = np.zeros((Nt + 1, Nx))

        initial_m_dist = self.problem.get_initial_m()
        self.M[0] = initial_m_dist
        final_u_cost = self.problem.get_final_u()
        for n_time_idx in range(Nt + 1):
            self.U[n_time_idx] = final_u_cost
            if n_time_idx > 0:
                self.M[n_time_idx] = initial_m_dist # Initialize M for t>0

        self.l2distu = np.ones(Niter)
        self.l2distm = np.ones(Niter)
        self.l2disturel = np.ones(Niter)
        self.l2distmrel = np.ones(Niter)
        self.iterations_run = 0

        for iiter in range(Niter):
            start_time_iter = time.time()
            print(f"\n******************** {self.name} Fixed-Point Iteration = {iiter + 1} / {Niter}")

            U_old_iter = self.U.copy()
            M_old_iter = self.M.copy()

            # Solve HJB backward using M_old_iter
            U_new_tmp_hjb = self.hjb_solver.solve_hjb(M_old_iter, final_u_cost)
            self.U = self.thetaUM * U_new_tmp_hjb + (1 - self.thetaUM) * U_old_iter

            # Solve FP forward using the newly computed U
            M_new_tmp_fp = self.fp_solver.solve_fp_system(initial_m_dist, self.U)
            self.M = self.thetaUM * M_new_tmp_fp + (1 - self.thetaUM) * M_old_iter

            # Convergence metrics (copied from FDMSolver.solve)
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
            print(f" === END Iteration {iiter+1}: ||u_new - u_old||_2 = {self.l2distu[iiter]:.4e} (rel: {self.l2disturel[iiter]:.4e})")
            print(f" === END Iteration {iiter+1}: ||m_new - m_old||_2 = {self.l2distm[iiter]:.4e} (rel: {self.l2distmrel[iiter]:.4e})")
            print(f" === Time for iteration = {elapsed_time_iter:.2f} s")

            self.iterations_run = iiter + 1
            if (self.l2disturel[iiter] < l2errBoundPicard and 
                self.l2distmrel[iiter] < l2errBoundPicard):
                print(f"Convergence reached after {iiter + 1} iterations.")
                break

        self.l2distu = self.l2distu[: self.iterations_run]
        self.l2distm = self.l2distm[: self.iterations_run]
        self.l2disturel = self.l2disturel[: self.iterations_run]
        self.l2distmrel = self.l2distmrel[: self.iterations_run]

        return self.U, self.M, self.iterations_run, self.l2distu, self.l2distm

    def get_results(self):
        return self.U, self.M

    def get_convergence_data(self):
        return (self.iterations_run, self.l2distu, self.l2distm,
                self.l2disturel, self.l2distmrel)