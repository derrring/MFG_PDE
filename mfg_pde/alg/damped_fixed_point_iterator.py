import numpy as np
import time
from typing import TYPE_CHECKING, Tuple  # Added Tuple

if TYPE_CHECKING:
    from ..core.mfg_problem import MFGProblem
    from .base_mfg_solver import MFGSolver
    from .hjb_solvers.base_hjb import BaseHJBSolver
    from .fp_solvers.base_fp import BaseFPSolver


# Dummy base classes for standalone checking if imports are tricky
if not TYPE_CHECKING:

    class MFGSolver:
        def __init__(self, problem):
            self.problem = problem

    class BaseHJBSolver:
        def __init__(self, problem):
            self.problem = problem

        hjb_method_name = "DummyHJB"

        def solve_hjb_system(self, M, U_final, U_prev_picard):
            return np.zeros_like(M)  # Added U_prev_picard

    class BaseFPSolver:
        def __init__(self, problem):
            self.problem = problem

        fp_method_name = "DummyFP"

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
        self.thetaUM = thetaUM

        hjb_name = getattr(hjb_solver, "hjb_method_name", "UnknownHJB")
        fp_name = getattr(fp_solver, "fp_method_name", "UnknownFP")
        self.name = f"HJB-{hjb_name}_FP-{fp_name}"

        self.U: np.ndarray
        self.M: np.ndarray

        self.l2distu_abs: np.ndarray
        self.l2distm_abs: np.ndarray
        self.l2distu_rel: np.ndarray
        self.l2distm_rel: np.ndarray
        self.iterations_run: int = 0

    def solve(
        self, Niter_max: int, l2errBoundPicard: float = 1e-5
    ) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]:
        print(
            f"\n________________ Solving MFG with {self.name} (T={self.problem.T}) _______________"
        )
        Nx = self.problem.Nx
        Nt = self.problem.Nt
        Dx = self.problem.Dx if abs(self.problem.Dx) > 1e-12 else 1.0
        Dt = self.problem.Dt if abs(self.problem.Dt) > 1e-12 else 1.0

        self.U = np.zeros((Nt, Nx))
        self.M = np.zeros((Nt, Nx))

        initial_m_dist = self.problem.get_initial_m()
        final_u_cost = self.problem.get_final_u()

        if Nt > 0:
            self.M[0, :] = initial_m_dist
            self.U[Nt - 1, :] = final_u_cost
            for n_time_idx in range(Nt - 1):
                self.U[n_time_idx, :] = final_u_cost
            for n_time_idx in range(1, Nt):
                self.M[n_time_idx, :] = initial_m_dist
        elif Nt == 0:
            print("Warning: Nt=0, cannot initialize U and M.")
            return self.U, self.M, 0, np.array([]), np.array([])

        self.l2distu_abs = np.ones(Niter_max)
        self.l2distm_abs = np.ones(Niter_max)
        self.l2distu_rel = np.ones(Niter_max)
        self.l2distm_rel = np.ones(Niter_max)
        self.iterations_run = 0

        U_picard_prev = (
            self.U.copy()
        )  # Initialize U from previous Picard (k-1) with initial U for k=0

        for iiter in range(Niter_max):
            start_time_iter = time.time()
            print(f"--- {self.name} Picard Iteration = {iiter + 1} / {Niter_max} ---")

            U_old_current_picard_iter = self.U.copy()  # U_k
            M_old_current_picard_iter = self.M.copy()  # M_k

            # 1. Solve HJB backward using M_old_current_picard_iter (M_k)
            #    and U_picard_prev (U_{k-1}) for the Jacobian if needed by the specific problem type
            U_new_tmp_hjb = self.hjb_solver.solve_hjb_system(
                M_old_current_picard_iter,  # M_k
                final_u_cost,
                U_picard_prev,  # U_{k-1} (this is U_old from notebook's solveHJB_withM(sigma, U, M))
            )

            # Apply damping to U update: U_{k+1} = theta * U_tmp + (1-theta) * U_k
            self.U = (
                self.thetaUM * U_new_tmp_hjb
                + (1 - self.thetaUM) * U_old_current_picard_iter
            )

            # 2. Solve FP forward using the newly computed U (U_{k+1})
            M_new_tmp_fp = self.fp_solver.solve_fp_system(initial_m_dist, self.U)

            # Apply damping to M update: M_{k+1} = theta * M_tmp + (1-theta) * M_k
            self.M = (
                self.thetaUM * M_new_tmp_fp
                + (1 - self.thetaUM) * M_old_current_picard_iter
            )

            # Update U_picard_prev for the next iteration's Jacobian calculation
            U_picard_prev = (
                U_old_current_picard_iter.copy()
            )  # U_k becomes U_{k-1} for next iter

            """
            # Ensure M remains non-negative and normalized
            self.M = np.maximum(self.M, 0)
            for t_step in range(Nt):
                current_mass = np.sum(self.M[t_step, :]) * Dx
                if current_mass > 1e-9:
                    self.M[t_step, :] /= current_mass
                else:
                    pass
            """

            norm_factor = np.sqrt(Dx * Dt)

            self.l2distu_abs[iiter] = (
                np.linalg.norm(self.U - U_old_current_picard_iter) * norm_factor
            )
            norm_U_iter = np.linalg.norm(self.U) * norm_factor
            self.l2distu_rel[iiter] = (
                self.l2distu_abs[iiter] / norm_U_iter
                if norm_U_iter > 1e-12
                else self.l2distu_abs[iiter]
            )

            self.l2distm_abs[iiter] = (
                np.linalg.norm(self.M - M_old_current_picard_iter) * norm_factor
            )
            norm_M_iter = np.linalg.norm(self.M) * norm_factor
            self.l2distm_rel[iiter] = (
                self.l2distm_abs[iiter] / norm_M_iter
                if norm_M_iter > 1e-12
                else self.l2distm_abs[iiter]
            )

            elapsed_time_iter = time.time() - start_time_iter
            print(
                f"  Iter {iiter+1}: Rel Err U={self.l2distu_rel[iiter]:.2e}, M={self.l2distm_rel[iiter]:.2e}. Abs Err U={self.l2distu_abs[iiter]:.2e}, M={self.l2distm_abs[iiter]:.2e}. Time: {elapsed_time_iter:.2f}s"
            )

            self.iterations_run = iiter + 1
            if (
                self.l2distu_rel[iiter] < l2errBoundPicard
                and self.l2distm_rel[iiter] < l2errBoundPicard
            ):
                print(f"Convergence reached after {iiter + 1} iterations.")
                break
        else:
            print(f"Warning: Max iterations ({Niter_max}) reached without convergence.")

        self.l2distu_abs = self.l2distu_abs[: self.iterations_run]
        self.l2distm_abs = self.l2distm_abs[: self.iterations_run]
        self.l2distu_rel = self.l2distu_rel[: self.iterations_run]
        self.l2distm_rel = self.l2distm_rel[: self.iterations_run]

        return self.U, self.M, self.iterations_run, self.l2distu_rel, self.l2distm_rel

    def get_results(self) -> Tuple[np.ndarray, np.ndarray]:
        if not hasattr(self, "U") or not hasattr(self, "M") or self.iterations_run == 0:
            raise ValueError(
                "Solver has not been run or did not produce results. Call solve() first."
            )
        return self.U, self.M

    def get_convergence_data(
        self,
    ) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.iterations_run == 0:
            raise ValueError("Solver has not been run yet. Call solve() first.")
        return (
            self.iterations_run,
            self.l2distu_abs,
            self.l2distm_abs,
            self.l2distu_rel,
            self.l2distm_rel,
        )
