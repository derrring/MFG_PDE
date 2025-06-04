import numpy as np
import time

from .base_mfg_solver import MFGSolver # Assuming MFGSolver is a base for this iterator
from ..core.mfg_problem import MFGProblem # For type hinting
from .hjb_solvers.base_hjb import BaseHJBSolver
from .fp_solvers.base_fp import BaseFPSolver


# To make this runnable standalone for checking, we might need to define dummy base classes
class MFGSolver:  # Dummy for standalone
    def __init__(self, problem):
        self.problem = problem


class BaseHJBSolver:  # Dummy
    def __init__(self, problem):
        self.problem = problem

    def solve_hjb_system(self, M, U_final):
        return np.zeros_like(M)


class BaseFPSolver:  # Dummy
    def __init__(self, problem):
        self.problem = problem

    def solve_fp_system(self, m_init, U):
        return np.zeros_like(U)


class FixedPointIterator(MFGSolver):
    def __init__(
        self,
        problem: "MFGProblem",
        hjb_solver: "BaseHJBSolver",
        fp_solver: "BaseFPSolver",
        thetaUM: float = 0.5,
    ):
        super().__init__(problem)
        self.hjb_solver = hjb_solver
        self.fp_solver = fp_solver
        self.thetaUM = thetaUM  # Damping factor

        # Construct a descriptive name based on actual solver components
        hjb_name = getattr(hjb_solver, "hjb_method_name", "UnknownHJB")
        fp_name = getattr(fp_solver, "fp_method_name", "UnknownFP")
        self.name = f"HJB-{hjb_name}_FP-{fp_name}"

        self.U: np.ndarray  # Stores the value function solution U(t,x)
        self.M: np.ndarray  # Stores the density solution M(t,x)

        self.l2distu: np.ndarray
        self.l2distm: np.ndarray
        self.l2disturel: np.ndarray
        self.l2distmrel: np.ndarray
        self.iterations_run: int = 0

    def solve(
        self, Niter_max: int, l2errBoundPicard: float = 1e-5
    ) -> tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]:
        print(
            f"\n________________ Solving MFG with {self.name} (T={self.problem.T}) _______________"
        )
        Nx = self.problem.Nx
        Nt = self.problem.Nt  # Number of time points
        Dx = self.problem.Dx
        Dt = self.problem.Dt

        # Initialize U and M arrays (Nt time points, Nx spatial points)
        self.U = np.zeros((Nt, Nx))
        self.M = np.zeros((Nt, Nx))

        initial_m_dist = self.problem.get_initial_m()  # Shape (Nx,)
        final_u_cost = self.problem.get_final_u()  # Shape (Nx,)

        # Initialize M at t=0 and U at t=T (which is index Nt-1)
        self.M[0, :] = initial_m_dist
        self.U[Nt - 1, :] = final_u_cost

        # Initialize U for t < T with final_u_cost, and M for t > 0 with initial_m_dist
        for n_time_idx in range(Nt - 1):  # 0 to Nt-2
            self.U[n_time_idx, :] = (
                final_u_cost  # Initialize U backward (can be improved)
            )
        for n_time_idx in range(1, Nt):  # 1 to Nt-1
            self.M[n_time_idx, :] = initial_m_dist  # Initialize M forward

        self.l2distu = np.ones(Niter_max)
        self.l2distm = np.ones(Niter_max)
        self.l2disturel = np.ones(Niter_max)
        self.l2distmrel = np.ones(Niter_max)
        self.iterations_run = 0

        for iiter in range(Niter_max):
            start_time_iter = time.time()
            print(
                f"\n******************** {self.name} Fixed-Point Iteration = {iiter + 1} / {Niter_max}"
            )

            U_old_iter = self.U.copy()
            M_old_iter = self.M.copy()

            # 1. Solve HJB backward using M_old_iter
            # The HJB solver (e.g., FdmHJBSolver) will use M_old_iter and problem.get_final_u()
            # Its output U_new_tmp_hjb will be (Nt, Nx)
            U_new_tmp_hjb = self.hjb_solver.solve_hjb_system(M_old_iter, final_u_cost)

            # Apply damping to U update
            self.U = self.thetaUM * U_new_tmp_hjb + (1 - self.thetaUM) * U_old_iter

            # 2. Solve FP forward using the newly computed U
            # The FP solver (e.g., FdmFPSolver or ParticleFPSolver) will use initial_m_dist and self.U
            # Its output M_new_tmp_fp will be (Nt, Nx)
            M_new_tmp_fp = self.fp_solver.solve_fp_system(initial_m_dist, self.U)

            # Apply damping to M update
            self.M = self.thetaUM * M_new_tmp_fp + (1 - self.thetaUM) * M_old_iter

            # Convergence metrics
            # Ensure Dx and Dt are not zero if Nx or Nt is 1
            norm_factor = np.sqrt(max(Dx, 1e-9) * max(Dt, 1e-9))  # Avoid sqrt(0)

            self.l2distu[iiter] = np.linalg.norm(self.U - U_old_iter) * norm_factor
            norm_U_iter = np.linalg.norm(self.U) * norm_factor
            self.l2disturel[iiter] = (
                self.l2distu[iiter] / norm_U_iter
                if norm_U_iter > 1e-9
                else self.l2distu[iiter]
            )

            self.l2distm[iiter] = np.linalg.norm(self.M - M_old_iter) * norm_factor
            norm_M_iter = np.linalg.norm(self.M) * norm_factor
            self.l2distmrel[iiter] = (
                self.l2distm[iiter] / norm_M_iter
                if norm_M_iter > 1e-9
                else self.l2distm[iiter]
            )

            elapsed_time_iter = time.time() - start_time_iter
            print(
                f" === Iter {iiter+1}: ||U_new-U_old||_rel = {self.l2disturel[iiter]:.2e}, ||M_new-M_old||_rel = {self.l2distmrel[iiter]:.2e}"
            )
            print(f" === Time for iteration = {elapsed_time_iter:.2f} s")

            self.iterations_run = iiter + 1
            if (
                self.l2disturel[iiter] < l2errBoundPicard
                and self.l2distmrel[iiter] < l2errBoundPicard
            ):
                print(f"Convergence reached after {iiter + 1} iterations.")
                break
        else:  # Loop finished without break
            print(f"Warning: Max iterations ({Niter_max}) reached without convergence.")

        # Trim convergence arrays
        self.l2distu = self.l2distu[: self.iterations_run]
        self.l2distm = self.l2distm[: self.iterations_run]
        self.l2disturel = self.l2disturel[: self.iterations_run]
        self.l2distmrel = self.l2distmrel[: self.iterations_run]

        return (
            self.U,
            self.M,
            self.iterations_run,
            self.l2disturel,
            self.l2distmrel,
        )  # Return relative errors

    def get_results(self) -> tuple[np.ndarray, np.ndarray]:
        if self.U is None or self.M is None:
            raise ValueError("Solver has not been run yet. Call solve() first.")
        return self.U, self.M

    def get_convergence_data(
        self,
    ) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.iterations_run == 0:
            raise ValueError("Solver has not been run yet. Call solve() first.")
        return (
            self.iterations_run,
            self.l2distu,
            self.l2distm,
            self.l2disturel,
            self.l2distmrel,
        )
