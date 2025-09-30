"""
Damped Fixed Point Iterator for Mean Field Games.

This is the classic fixed point iterator with damping, providing the foundational
approach for solving coupled MFG systems using numerical methods.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np

from .base_mfg import BaseMFGSolver

if TYPE_CHECKING:
    from mfg_pde.alg.numerical.fp_solvers.base_fp import BaseFPSolver
    from mfg_pde.alg.numerical.hjb_solvers.base_hjb import BaseHJBSolver
    from mfg_pde.core.mfg_problem import MFGProblem
    from mfg_pde.utils.solver_result import SolverResult


class FixedPointIterator(BaseMFGSolver):
    """
    Classic Fixed Point Iterator with damping for MFG systems.

    This solver implements the foundational Picard iteration approach for
    coupled Mean Field Games systems, alternating between HJB and FP solutions
    with optional damping for stability.

    Features:
    - Classic Picard iteration with configurable damping
    - Comprehensive convergence monitoring
    - Warm start capability
    - Enhanced error handling and reporting
    - Structured result objects with rich metadata
    - Full backward compatibility

    Note: This is the classic implementation maintained for compatibility
    and comparison with newer configuration-aware approaches.
    """

    def __init__(
        self,
        problem: MFGProblem,
        hjb_solver: BaseHJBSolver,
        fp_solver: BaseFPSolver,
        thetaUM: float = 0.5,
    ):
        """
        Initialize the Fixed Point Iterator.

        Args:
            problem: MFG problem instance
            hjb_solver: Hamilton-Jacob-Bellman solver
            fp_solver: Fokker-Planck solver
            thetaUM: Damping parameter (0.5 = balanced damping)
        """
        super().__init__(problem)
        self.hjb_solver = hjb_solver
        self.fp_solver = fp_solver
        self.thetaUM = thetaUM

        hjb_name = getattr(hjb_solver, "hjb_method_name", "UnknownHJB")
        fp_name = getattr(fp_solver, "fp_method_name", "UnknownFP")
        self.name = f"HJB-{hjb_name}_FP-{fp_name}"

        # Initialize solution arrays
        self.U: np.ndarray
        self.M: np.ndarray

        # Convergence tracking
        self.l2distu_abs: np.ndarray
        self.l2distm_abs: np.ndarray
        self.l2distu_rel: np.ndarray
        self.l2distm_rel: np.ndarray
        self.iterations_run: int = 0

    def solve(  # type: ignore[override]
        self,
        max_iterations: int | None = None,
        tolerance: float | None = None,
        # Alias parameters for specific solver compatibility
        max_picard_iterations: int | None = None,
        picard_tolerance: float | None = None,
        # Deprecated parameters for backward compatibility
        Niter_max: int | None = None,
        l2errBoundPicard: float | None = None,
        # New parameter for result format
        return_structured: bool = False,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray] | SolverResult:
        """
        Solve the MFG system using fixed point iteration.

        Args:
            max_iterations: Maximum number of Picard iterations
            tolerance: Convergence tolerance
            max_picard_iterations: Alias for max_iterations
            picard_tolerance: Alias for tolerance
            Niter_max: DEPRECATED - use max_iterations
            l2errBoundPicard: DEPRECATED - use tolerance
            return_structured: Return SolverResult object instead of tuple
            **kwargs: Additional parameters

        Returns:
            Tuple (U, M, iterations, err_u, err_m) or SolverResult object
        """
        import warnings

        # Handle parameter precedence: standardized > specific > deprecated
        # Priority: max_iterations/tolerance > max_picard_iterations/picard_tolerance > Niter_max/l2errBoundPicard

        final_max_iterations = None
        final_tolerance = None

        # Process max_iterations with precedence
        if max_iterations is not None:
            final_max_iterations = max_iterations
        elif max_picard_iterations is not None:
            final_max_iterations = max_picard_iterations
        elif Niter_max is not None:
            warnings.warn(
                "Parameter 'Niter_max' is deprecated. Use 'max_iterations' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            final_max_iterations = Niter_max
        else:
            final_max_iterations = 20  # Default

        # Process tolerance with precedence
        if tolerance is not None:
            final_tolerance = tolerance
        elif picard_tolerance is not None:
            final_tolerance = picard_tolerance
        elif l2errBoundPicard is not None:
            warnings.warn(
                "Parameter 'l2errBoundPicard' is deprecated. Use 'tolerance' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            final_tolerance = l2errBoundPicard
        else:
            final_tolerance = 1e-5  # Default

        # Validate parameters with enhanced error messages
        from mfg_pde.utils.exceptions import validate_parameter_value

        validate_parameter_value(
            final_max_iterations,
            "max_iterations",
            int,
            (1, 1000),
            solver_name=f"{self.name} (Fixed Point Iterator)",
        )
        validate_parameter_value(
            final_tolerance,
            "tolerance",
            (int, float),  # type: ignore[arg-type]
            (1e-12, 1e-1),
            solver_name=f"{self.name} (Fixed Point Iterator)",
        )

        # Track execution time for structured results
        solve_start_time = time.time()

        print(f"\n________________ Solving MFG with {self.name} (T={self.problem.T}) _______________")
        Nx = self.problem.Nx + 1
        Nt = self.problem.Nt + 1
        Dx = self.problem.Dx if abs(self.problem.Dx) > 1e-12 else 1.0
        Dt = self.problem.Dt if abs(self.problem.Dt) > 1e-12 else 1.0

        # Try warm start initialization first
        warm_start_init = self._get_warm_start_initialization()
        if warm_start_init is not None:
            U_init, M_init = warm_start_init
            self.U = U_init.copy()
            self.M = M_init.copy()
            print("   Using warm start initialization from previous solution")
        else:
            # Cold start - default initialization
            self.U = np.zeros((Nt, Nx))
            self.M = np.zeros((Nt, Nx))

        initial_m_dist = self.problem.get_initial_m()
        final_u_cost = self.problem.get_final_u()

        if Nt > 0:
            # Always enforce boundary conditions (even with warm start)
            self.M[0, :] = initial_m_dist
            self.U[Nt - 1, :] = final_u_cost

            # For cold start, initialize interior with boundary conditions
            if warm_start_init is None:
                for n_time_idx in range(Nt - 1):
                    self.U[n_time_idx, :] = final_u_cost
                for n_time_idx in range(1, Nt):
                    self.M[n_time_idx, :] = initial_m_dist
        elif Nt == 0:
            print("Warning: Nt=0, cannot initialize U and M.")
            return self.U, self.M, 0, np.array([]), np.array([])

        self.l2distu_abs = np.ones(final_max_iterations)
        self.l2distm_abs = np.ones(final_max_iterations)
        self.l2distu_rel = np.ones(final_max_iterations)
        self.l2distm_rel = np.ones(final_max_iterations)
        self.iterations_run = 0

        U_picard_prev = self.U.copy()  # Initialize U from previous Picard (k-1) with initial U for k=0

        for iiter in range(final_max_iterations):
            start_time_iter = time.time()
            print(f"--- {self.name} Picard Iteration = {iiter + 1} / {final_max_iterations} ---")

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
            self.U = self.thetaUM * U_new_tmp_hjb + (1 - self.thetaUM) * U_old_current_picard_iter

            # 2. Solve FP forward using the newly computed U (U_{k+1})
            M_new_tmp_fp = self.fp_solver.solve_fp_system(initial_m_dist, self.U)

            # Apply damping to M update: M_{k+1} = theta * M_tmp + (1-theta) * M_k
            self.M = self.thetaUM * M_new_tmp_fp + (1 - self.thetaUM) * M_old_current_picard_iter

            # Update U_picard_prev for the next iteration's Jacobian calculation
            U_picard_prev = U_old_current_picard_iter.copy()  # U_k becomes U_{k-1} for next iter

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

            self.l2distu_abs[iiter] = np.linalg.norm(self.U - U_old_current_picard_iter) * norm_factor
            norm_U_iter = np.linalg.norm(self.U) * norm_factor
            self.l2distu_rel[iiter] = (
                self.l2distu_abs[iiter] / norm_U_iter if norm_U_iter > 1e-12 else self.l2distu_abs[iiter]
            )

            self.l2distm_abs[iiter] = np.linalg.norm(self.M - M_old_current_picard_iter) * norm_factor
            norm_M_iter = np.linalg.norm(self.M) * norm_factor
            self.l2distm_rel[iiter] = (
                self.l2distm_abs[iiter] / norm_M_iter if norm_M_iter > 1e-12 else self.l2distm_abs[iiter]
            )

            elapsed_time_iter = time.time() - start_time_iter
            print(
                f"  Iter {iiter + 1}: Rel Err U={self.l2distu_rel[iiter]:.2e}, "
                f"M={self.l2distm_rel[iiter]:.2e}. "
                f"Abs Err U={self.l2distu_abs[iiter]:.2e}, M={self.l2distm_abs[iiter]:.2e}. "
                f"Time: {elapsed_time_iter:.2f}s"
            )

            self.iterations_run = iiter + 1
            if self.l2distu_rel[iiter] < final_tolerance and self.l2distm_rel[iiter] < final_tolerance:
                print(f"Convergence reached after {iiter + 1} iterations.")
                break
        else:
            # Enhanced convergence failure reporting
            final_error_u = self.l2distu_rel[self.iterations_run - 1] if self.iterations_run > 0 else float("inf")
            final_error_m = self.l2distm_rel[self.iterations_run - 1] if self.iterations_run > 0 else float("inf")
            final_error = max(final_error_u, final_error_m)

            convergence_history = list(self.l2distu_rel[: self.iterations_run]) + list(
                self.l2distm_rel[: self.iterations_run]
            )

            from mfg_pde.utils.exceptions import ConvergenceError

            # Use strict error handling mode by default (no config available)
            strict_mode = True

            if strict_mode:
                # Strict mode: Always raise convergence errors
                conv_error = ConvergenceError(
                    iterations_used=self.iterations_run,
                    max_iterations=final_max_iterations,
                    final_error=final_error,
                    tolerance=final_tolerance,
                    solver_name=self.name,
                    convergence_history=convergence_history,
                )
                raise conv_error
            else:
                # Permissive mode: Log warning with detailed analysis
                print(f"WARNING:  Convergence Warning: Max iterations ({final_max_iterations}) reached")

                try:
                    conv_error = ConvergenceError(
                        iterations_used=self.iterations_run,
                        max_iterations=final_max_iterations,
                        final_error=final_error,
                        tolerance=final_tolerance,
                        solver_name=self.name,
                        convergence_history=convergence_history,
                    )
                    print(f" {conv_error.suggested_action}")
                    # Store error for later analysis
                    self._convergence_warning = conv_error
                except Exception as e:
                    # Fallback if error analysis fails
                    print(" Suggestion: Try increasing max_picard_iterations or relaxing picard_tolerance")
                    print(f"WARNING:  Error analysis failed: {e}")

        self.l2distu_abs = self.l2distu_abs[: self.iterations_run]
        self.l2distm_abs = self.l2distm_abs[: self.iterations_run]
        self.l2distu_rel = self.l2distu_rel[: self.iterations_run]
        self.l2distm_rel = self.l2distm_rel[: self.iterations_run]

        # Mark solution as computed for warm start capability
        self._solution_computed = True

        # Calculate total execution time
        execution_time = time.time() - solve_start_time

        # Return structured result if requested, otherwise maintain backward compatibility
        if return_structured:
            from mfg_pde.utils.solver_result import create_solver_result

            return create_solver_result(
                U=self.U,
                M=self.M,
                iterations=self.iterations_run,
                error_history_U=self.l2distu_rel,
                error_history_M=self.l2distm_rel,
                solver_name=self.name,
                convergence_achieved=(
                    (self.l2distu_rel[-1] < final_tolerance and self.l2distm_rel[-1] < final_tolerance)
                    if self.iterations_run > 0
                    else False
                ),
                tolerance=final_tolerance,
                execution_time=execution_time,
                # Additional metadata
                damping_parameter=self.thetaUM,
                problem_parameters={
                    "T": self.problem.T,
                    "Nx": self.problem.Nx,
                    "Nt": self.problem.Nt,
                    "Dx": getattr(self.problem, "Dx", None),
                    "Dt": getattr(self.problem, "Dt", None),
                },
                absolute_errors_U=self.l2distu_abs,
                absolute_errors_M=self.l2distm_abs,
            )
        else:
            # Backward compatible tuple return
            return (
                self.U,
                self.M,
                self.iterations_run,
                self.l2distu_rel,
                self.l2distm_rel,
            )

    def _get_warm_start_initialization(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Get warm start initialization data."""
        return self.get_warm_start_data()

    def get_results(self) -> tuple[np.ndarray, np.ndarray]:
        """Get computed U and M solutions."""
        from mfg_pde.utils.exceptions import validate_solver_state

        validate_solver_state(self, "get_results")
        return self.U, self.M

    def get_convergence_data(
        self,
    ) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get convergence information."""
        from mfg_pde.utils.exceptions import validate_solver_state

        validate_solver_state(self, "get_convergence_data")
        return (
            self.iterations_run,
            self.l2distu_abs,
            self.l2distm_abs,
            self.l2distu_rel,
            self.l2distm_rel,
        )
