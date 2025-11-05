"""
Fixed Point Iterator

Modern fixed-point iterator for MFG systems with full feature support:
- Config-based parameter management with backward compatibility
- Anderson acceleration for faster convergence
- Backend support (GPU/CPU)
- Structured SolverResult output (tuple output for legacy compatibility)
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np

from mfg_pde.utils.solver_result import SolverResult

from .base_mfg import BaseMFGSolver
from .fixed_point_utils import (
    check_convergence_criteria,
    initialize_cold_start,
    preserve_boundary_conditions,
)

if TYPE_CHECKING:
    from mfg_pde.alg.numerical.fp_solvers.base_fp import BaseFPSolver
    from mfg_pde.alg.numerical.hjb_solvers.base_hjb import BaseHJBSolver
    from mfg_pde.config import MFGSolverConfig
    from mfg_pde.problem.base_mfg_problem import MFGProblem


class FixedPointIterator(BaseMFGSolver):
    """
    Fixed-point iterator for MFG systems with full feature support.

    Features:
    - Config-based parameter management with backward compatibility
    - Optional Anderson acceleration for faster convergence
    - GPU/CPU backend support
    - Structured SolverResult output (tuple output for legacy compatibility)
    - Warm start support

    Args:
        problem: MFG problem definition
        hjb_solver: HJB solver instance
        fp_solver: FP solver instance
        config: Configuration object (preferred modern approach)
        damping_factor: Damping parameter (legacy parameter, overridden by config)
        use_anderson: Enable Anderson acceleration
        anderson_depth: Anderson acceleration memory depth
        anderson_beta: Anderson acceleration mixing parameter
        backend: Backend name ('numpy', 'torch', 'jax', etc.)
    """

    def __init__(
        self,
        problem: MFGProblem,
        hjb_solver: BaseHJBSolver,
        fp_solver: BaseFPSolver,
        config: MFGSolverConfig | None = None,
        damping_factor: float = 0.5,  # Renamed from thetaUM
        use_anderson: bool = False,
        anderson_depth: int = 5,
        anderson_beta: float = 1.0,
        backend: str | None = None,
    ):
        super().__init__(problem)
        self.backend = backend
        self.hjb_solver = hjb_solver
        self.fp_solver = fp_solver
        self.config = config

        # Anderson acceleration support
        self.use_anderson = use_anderson
        self.anderson_accelerator = None
        if use_anderson:
            from mfg_pde.utils.numerical.anderson_acceleration import AndersonAccelerator

            self.anderson_accelerator = AndersonAccelerator(depth=anderson_depth, beta=anderson_beta)

        # Damping parameter (overridden by config if provided)
        self.damping_factor = damping_factor

        # State arrays (initialized in solve)
        self.U: np.ndarray | None = None
        self.M: np.ndarray | None = None

        # Convergence tracking
        self.l2distu_abs: np.ndarray | None = None
        self.l2distm_abs: np.ndarray | None = None
        self.l2distu_rel: np.ndarray | None = None
        self.l2distm_rel: np.ndarray | None = None
        self.iterations_run = 0

        # Warm start support
        self._warm_start_U: np.ndarray | None = None
        self._warm_start_M: np.ndarray | None = None

    def solve(
        self,
        config: MFGSolverConfig | None = None,
        max_iterations: int | None = None,
        tolerance: float | None = None,
        return_tuple: bool = False,
        **kwargs: Any,
    ) -> SolverResult | tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]:
        """
        Solve coupled MFG system using fixed-point iteration.

        Args:
            config: Solver configuration (overrides instance config)
            max_iterations: Maximum iterations (legacy parameter)
            tolerance: Convergence tolerance (legacy parameter)
            return_tuple: Return legacy tuple format instead of SolverResult
            **kwargs: Additional parameters for backward compatibility

        Returns:
            SolverResult object (or tuple if return_tuple=True)
        """
        # Use provided config or fall back to instance config
        solve_config = config or self.config

        # Parameter resolution (config > explicit args > instance defaults)
        if solve_config is not None:
            final_max_iterations = solve_config.picard.max_iterations
            final_tolerance = solve_config.picard.tolerance
            final_damping_factor = solve_config.picard.damping_factor
            verbose = solve_config.picard.verbose
        else:
            # Legacy parameter precedence
            final_max_iterations = (
                max_iterations or kwargs.get("max_picard_iterations") or kwargs.get("Niter_max") or 100
            )
            final_tolerance = tolerance or kwargs.get("picard_tolerance") or kwargs.get("l2errBoundPicard") or 1e-6
            final_damping_factor = self.damping_factor
            verbose = True

        # Get problem dimensions - handle both old 1D and new nD interfaces
        num_time_steps = self.problem.Nt + 1  # Renamed from Nt

        # Detect problem shape
        if hasattr(self.problem, "Nx"):
            # Old 1D interface
            shape = (self.problem.Nx + 1,)
            grid_spacing = self.problem.Dx  # Renamed from Dx
            time_step = self.problem.Dt  # Renamed from Dt
        elif hasattr(self.problem, "geometry") and hasattr(self.problem.geometry, "grid"):
            # New GridBasedMFGProblem interface (nD)
            shape = tuple(self.problem.geometry.grid.num_points)
            grid_spacing = self.problem.geometry.grid.spacing[0]  # For compatibility
            time_step = self.problem.dt
        else:
            raise ValueError("Problem must have either (Nx, Dx, Dt) or (geometry.grid) attributes")

        # Initialize arrays (cold start or warm start)
        warm_start = self.get_warm_start_data()
        if warm_start is not None:
            self.U, self.M = warm_start
        else:
            # Cold start initialization
            if self.backend is not None:
                self.U = self.backend.zeros((num_time_steps, *shape))
                self.M = self.backend.zeros((num_time_steps, *shape))
            else:
                self.U = np.zeros((num_time_steps, *shape))
                self.M = np.zeros((num_time_steps, *shape))

            # Get initial density and final cost - handle both interfaces
            if hasattr(self.problem, "get_initial_m"):
                # Old 1D interface
                initial_m_dist = self.problem.get_initial_m()
                final_u_cost = self.problem.get_final_u()
            else:
                # New nD interface - evaluate on grid
                x_grid = self.problem.geometry.grid.flatten()
                initial_m_dist = self.problem.initial_density(x_grid).reshape(shape)
                final_u_cost = self.problem.terminal_cost(x_grid).reshape(shape)
                # Note: initial_density() should already return normalized density
                # (integral = 1), so no renormalization needed here

            if num_time_steps > 0:
                # Set boundary conditions
                if len(shape) == 1:
                    self.M[0, :] = initial_m_dist
                    self.U[num_time_steps - 1, :] = final_u_cost
                else:
                    self.M[0] = initial_m_dist
                    self.U[num_time_steps - 1] = final_u_cost

                # Initialize interior with boundary conditions
                self.U, self.M = initialize_cold_start(self.U, self.M, initial_m_dist, final_u_cost, num_time_steps)

        # Initialize error tracking
        self.l2distu_abs = np.ones(final_max_iterations)
        self.l2distm_abs = np.ones(final_max_iterations)
        self.l2distu_rel = np.ones(final_max_iterations)
        self.l2distm_rel = np.ones(final_max_iterations)
        self.iterations_run = 0

        # Reset Anderson accelerator if using it
        if self.anderson_accelerator is not None:
            self.anderson_accelerator.reset()

        # Main fixed-point iteration loop
        converged = False
        convergence_reason = "Maximum iterations reached"
        # initial_m_dist already computed above

        # Progress bar for Picard iterations
        from mfg_pde.utils.progress import tqdm

        picard_range = range(final_max_iterations)
        if verbose:
            picard_range = tqdm(
                picard_range,
                desc="MFG Picard",
                unit="iter",
                disable=False,
            )

        for iiter in picard_range:
            iter_start = time.time()

            U_old = self.U.copy()
            M_old = self.M.copy()

            # 1. Solve HJB backward with current M (disable inner progress bar when verbose)
            show_hjb_progress = not verbose
            if hasattr(self.hjb_solver, "solve_hjb_system"):
                # Check if solve_hjb_system accepts show_progress parameter
                import inspect

                sig = inspect.signature(self.hjb_solver.solve_hjb_system)
                if "show_progress" in sig.parameters:
                    U_new = self.hjb_solver.solve_hjb_system(
                        M_old, final_u_cost, U_old, show_progress=show_hjb_progress
                    )
                else:
                    U_new = self.hjb_solver.solve_hjb_system(M_old, final_u_cost, U_old)
            else:
                U_new = self.hjb_solver.solve_hjb_system(M_old, final_u_cost, U_old)

            # 2. Solve FP forward with new U (disable inner progress bar when verbose)
            show_fp_progress = not verbose
            if hasattr(self.fp_solver, "solve_fp_system"):
                # Check if solve_fp_system accepts show_progress parameter
                import inspect

                sig = inspect.signature(self.fp_solver.solve_fp_system)
                if "show_progress" in sig.parameters:
                    M_new = self.fp_solver.solve_fp_system(initial_m_dist, U_new, show_progress=show_fp_progress)
                else:
                    M_new = self.fp_solver.solve_fp_system(initial_m_dist, U_new)
            else:
                M_new = self.fp_solver.solve_fp_system(initial_m_dist, U_new)

            # 3. Apply damping or Anderson acceleration
            if self.use_anderson and self.anderson_accelerator is not None:
                # Anderson acceleration on U only (M uses standard damping for positivity)
                x_current_U = U_old.flatten()
                f_current_U = U_new.flatten()
                x_next_U = self.anderson_accelerator.update(x_current_U, f_current_U, method="type1")
                self.U = x_next_U.reshape(U_old.shape)

                # Standard damping for M (guarantees non-negativity and mass conservation)
                self.M = final_damping_factor * M_new + (1 - final_damping_factor) * M_old
            else:
                # Standard damping for both
                self.U = final_damping_factor * U_new + (1 - final_damping_factor) * U_old
                self.M = final_damping_factor * M_new + (1 - final_damping_factor) * M_old

            # Preserve initial boundary condition
            self.M = preserve_boundary_conditions(self.M, initial_m_dist)

            # Calculate convergence metrics
            from mfg_pde.utils.numerical.convergence import calculate_l2_convergence_metrics

            metrics = calculate_l2_convergence_metrics(self.U, U_old, self.M, M_old, grid_spacing, time_step)
            self.l2distu_abs[iiter] = metrics["l2distu_abs"]
            self.l2distu_rel[iiter] = metrics["l2distu_rel"]
            self.l2distm_abs[iiter] = metrics["l2distm_abs"]
            self.l2distm_rel[iiter] = metrics["l2distm_rel"]

            iter_time = time.time() - iter_start
            self.iterations_run = iiter + 1

            # Update progress bar with convergence metrics
            if verbose and hasattr(picard_range, "set_postfix"):
                accel_tag = "A" if self.use_anderson else ""
                picard_range.set_postfix(
                    U_err=f"{self.l2distu_rel[iiter]:.2e}",
                    M_err=f"{self.l2distm_rel[iiter]:.2e}",
                    t=f"{iter_time:.1f}s",
                    acc=accel_tag,
                )
            elif not verbose:
                # Print traditional output when not using progress bar
                accel_tag = " (Anderson)" if self.use_anderson else ""
                print(f"--- Picard Iteration {iiter + 1}/{final_max_iterations}{accel_tag} ---")
                print(f"  Errors: U={self.l2distu_rel[iiter]:.2e}, M={self.l2distm_rel[iiter]:.2e}")
                print(f"  Time: {iter_time:.3f}s")

            # Check convergence
            converged, convergence_reason = check_convergence_criteria(
                self.l2distu_rel[iiter],
                self.l2distm_rel[iiter],
                self.l2distu_abs[iiter],
                self.l2distm_abs[iiter],
                final_tolerance,
            )

            if converged:
                if verbose and hasattr(picard_range, "write"):
                    picard_range.write(convergence_reason)
                elif not verbose:
                    print(convergence_reason)
                break

        # Construct result
        result = SolverResult(
            U=self.U,
            M=self.M,
            iterations=self.iterations_run,
            error_history_U=self.l2distu_abs[: self.iterations_run],
            error_history_M=self.l2distm_abs[: self.iterations_run],
            solver_name=self.name,
            converged=converged,
            metadata={
                "convergence_reason": convergence_reason,
                "l2distu_rel": self.l2distu_rel[: self.iterations_run],
                "l2distm_rel": self.l2distm_rel[: self.iterations_run],
                "anderson_used": self.use_anderson,
            },
        )

        # Return tuple for backward compatibility if requested
        if return_tuple:
            return (
                self.U,
                self.M,
                self.iterations_run,
                self.l2distu_rel[: self.iterations_run],
                self.l2distm_rel[: self.iterations_run],
            )
        else:
            return result

    @property
    def name(self) -> str:
        """Solver name for diagnostics."""
        return "FixedPointIterator"

    def get_convergence_data(self) -> dict[str, np.ndarray]:
        """Get convergence diagnostics."""
        return {
            "l2distu_abs": self.l2distu_abs[: self.iterations_run] if self.l2distu_abs is not None else np.array([]),
            "l2distm_abs": self.l2distm_abs[: self.iterations_run] if self.l2distm_abs is not None else np.array([]),
            "l2distu_rel": self.l2distu_rel[: self.iterations_run] if self.l2distu_rel is not None else np.array([]),
            "l2distm_rel": self.l2distm_rel[: self.iterations_run] if self.l2distm_rel is not None else np.array([]),
        }

    def set_warm_start_data(self, U_init: np.ndarray, M_init: np.ndarray) -> None:
        """Set warm start initialization data."""
        self._warm_start_U = U_init
        self._warm_start_M = M_init

    def get_warm_start_data(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Get warm start initialization data."""
        if self._warm_start_U is not None and self._warm_start_M is not None:
            return (self._warm_start_U, self._warm_start_M)
        return None

    def clear_warm_start_data(self) -> None:
        """Clear warm start data."""
        self._warm_start_U = None
        self._warm_start_M = None

    def get_results(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the computed solution arrays.

        Returns:
            Tuple of (U, M) solution arrays

        Raises:
            RuntimeError: If no solution has been computed yet
        """
        if self.U is None or self.M is None:
            raise RuntimeError("No solution computed. Call solve() first.")
        return self.U, self.M
