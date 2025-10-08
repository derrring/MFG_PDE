"""
Configuration-aware Fixed Point Iterator for Mean Field Games.

This is an enhanced version of the FixedPointIterator that uses the new
configuration system for improved parameter management and user experience.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np

from mfg_pde.config.solver_config import MFGSolverConfig, extract_legacy_parameters

from .base_mfg import BaseMFGSolver

if TYPE_CHECKING:
    from mfg_pde.alg.numerical.fp_solvers.base_fp import BaseFPSolver
    from mfg_pde.alg.numerical.hjb_solvers.base_hjb import BaseHJBSolver
    from mfg_pde.core.mfg_problem import MFGProblem
    from mfg_pde.utils.solver_result import SolverResult


class ConfigAwareFixedPointIterator(BaseMFGSolver):
    """
    Fixed Point Iterator with structured configuration support.

    This enhanced version uses the MFGSolverConfig system for better parameter
    management, validation, and user experience while maintaining full backward
    compatibility with the original API.

    Features:
    - Structured configuration objects for all parameters
    - Automatic parameter validation and sensible defaults
    - Factory methods for common use cases (fast, accurate, research)
    - Enhanced error handling and convergence analysis
    - Structured result objects with rich metadata
    - Full backward compatibility with legacy parameter names

    Note: This class maintains backward compatibility with the original interface
    while being part of the new numerical methods paradigm.
    """

    def __init__(
        self,
        problem: MFGProblem,
        hjb_solver: BaseHJBSolver,
        fp_solver: BaseFPSolver,
        config: MFGSolverConfig | None = None,
        thetaUM: float = 0.5,  # Legacy parameter for backward compatibility
    ):
        """
        Initialize the configuration-aware Fixed Point Iterator.

        Args:
            problem: MFG problem instance
            hjb_solver: Hamilton-Jacob-Bellman solver
            fp_solver: Fokker-Planck solver
            config: Structured configuration object (uses default if None)
            thetaUM: Legacy damping parameter (superseded by config.picard.damping_factor)
        """
        super().__init__(problem)
        self.hjb_solver = hjb_solver
        self.fp_solver = fp_solver

        # Use provided config or create default
        self.config = config or MFGSolverConfig()

        # Handle legacy damping parameter
        if thetaUM != 0.5:  # Non-default value provided
            import warnings

            warnings.warn(
                "Parameter 'thetaUM' is deprecated. Use config.picard.damping_factor instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.config.picard.damping_factor = thetaUM

        self.thetaUM = self.config.picard.damping_factor  # For backward compatibility

        # Generate solver name
        hjb_name = getattr(hjb_solver, "hjb_method_name", "UnknownHJB")
        fp_name = getattr(fp_solver, "fp_method_name", "UnknownFP")
        self.name = f"HJB-{hjb_name}_FP-{fp_name} (Config-Aware)"

        # Backend support - initialize as None, will be set via property
        self._backend = None

        # Initialize solution arrays
        self.U: np.ndarray
        self.M: np.ndarray

        # Convergence tracking
        self.l2distu_abs: np.ndarray
        self.l2distm_abs: np.ndarray
        self.l2distu_rel: np.ndarray
        self.l2distm_rel: np.ndarray
        self.iterations_run: int = 0

    def solve(self, config: MFGSolverConfig | None = None, **kwargs: Any) -> SolverResult:  # type: ignore[override]
        """
        Solve the MFG system using structured configuration.

        Args:
            config: Configuration object (uses instance config if None)
            **kwargs: Legacy parameters for backward compatibility

        Returns:
            SolverResult object with solution arrays and metadata.
            Note: For backward compatibility, SolverResult supports tuple unpacking:
                  U, M, iterations, err_u, err_m = solver.solve()
        """
        # Use provided config or instance config
        solve_config = config or self.config

        # Handle legacy parameters for backward compatibility
        kwargs = extract_legacy_parameters(solve_config, **kwargs)

        # Validate configuration
        solve_config.picard.__post_init__()  # Trigger validation

        if solve_config.picard.verbose:
            print(f"\n{'=' * 80}")
            print(f" {self.name}")
            print(f"{'=' * 80}")
            print(" Configuration:")
            print(f"   • Picard iterations: {solve_config.picard.max_iterations}")
            print(f"   • Picard tolerance: {solve_config.picard.tolerance:.2e}")
            print(f"   • Damping factor: {solve_config.picard.damping_factor}")
            print(f"   • Return format: {'Structured' if solve_config.return_structured else 'Tuple'}")
            print(f"   • Warm start: {'Enabled' if solve_config.warm_start else 'Disabled'}")

        # Track execution time
        solve_start_time = time.time()

        # Get problem dimensions
        Nx = self.problem.Nx + 1
        Nt = self.problem.Nt + 1
        Dx = self.problem.Dx if abs(self.problem.Dx) > 1e-12 else 1.0
        Dt = self.problem.Dt if abs(self.problem.Dt) > 1e-12 else 1.0

        # Initialize solution arrays
        if solve_config.warm_start and self.has_warm_start_data:
            warm_start_init = self._get_warm_start_initialization()
            if warm_start_init is not None:
                self.U, self.M = warm_start_init
                if solve_config.picard.verbose:
                    print(" Using warm start initialization")
            else:
                self._cold_start_initialization(Nt, Nx)
                if solve_config.picard.verbose:
                    print("Cold start initialization (warm start failed)")
        else:
            self._cold_start_initialization(Nt, Nx)
            if solve_config.picard.verbose:
                print("Cold start initialization")

        # Initialize convergence tracking - use backend if available
        if self.backend is not None:
            self.l2distu_abs = self.backend.ones((solve_config.picard.max_iterations,))
            self.l2distm_abs = self.backend.ones((solve_config.picard.max_iterations,))
            self.l2distu_rel = self.backend.ones((solve_config.picard.max_iterations,))
            self.l2distm_rel = self.backend.ones((solve_config.picard.max_iterations,))
        else:
            self.l2distu_abs = np.ones(solve_config.picard.max_iterations)
            self.l2distm_abs = np.ones(solve_config.picard.max_iterations)
            self.l2distu_rel = np.ones(solve_config.picard.max_iterations)
            self.l2distm_rel = np.ones(solve_config.picard.max_iterations)
        self.iterations_run = 0

        # Picard iteration with enhanced monitoring
        # Use .clone() for PyTorch tensors, .copy() for NumPy arrays
        U_picard_prev = self.U.clone() if hasattr(self.U, "clone") else self.U.copy()
        convergence_achieved = False

        for iiter in range(solve_config.picard.max_iterations):
            iter_start_time = time.time()

            if solve_config.picard.verbose:
                print(f"\n Picard Iteration {iiter + 1}/{solve_config.picard.max_iterations}")

            # Store previous iteration
            # Use .clone() for PyTorch tensors, .copy() for NumPy arrays
            U_old = self.U.clone() if hasattr(self.U, "clone") else self.U.copy()
            M_old = self.M.clone() if hasattr(self.M, "clone") else self.M.copy()

            # Solve HJB system
            final_u_cond = self.problem.get_final_u() if hasattr(self.problem, "get_final_u") else np.zeros(Nx)
            if self.backend is not None:
                final_u_cond = self.backend.from_numpy(final_u_cond)
            U_new_tmp = self.hjb_solver.solve_hjb_system(
                M_old,
                final_u_cond,
                U_picard_prev,
            )

            # Apply damping to U
            self.U = solve_config.picard.damping_factor * U_new_tmp + (1 - solve_config.picard.damping_factor) * U_old

            # Solve FP system
            initial_m_cond = self.problem.get_initial_m() if hasattr(self.problem, "get_initial_m") else np.ones(Nx)
            if self.backend is not None:
                initial_m_cond = self.backend.from_numpy(initial_m_cond)
            M_new_tmp = self.fp_solver.solve_fp_system(
                initial_m_cond,
                self.U,
            )

            # Apply damping to M
            self.M = solve_config.picard.damping_factor * M_new_tmp + (1 - solve_config.picard.damping_factor) * M_old

            # Preserve initial condition (boundary condition in time)
            # The damping step above may modify M[0,:], but initial condition is fixed
            initial_m_dist = self.problem.get_initial_m() if hasattr(self.problem, "get_initial_m") else np.ones(Nx)
            if self.backend is not None:
                initial_m_dist = self.backend.from_numpy(initial_m_dist)
            self.M[0, :] = initial_m_dist

            # Update U_picard_prev for next iteration
            U_picard_prev = U_old.clone() if hasattr(U_old, "clone") else U_old.copy()

            # Compute convergence metrics
            norm_factor = np.sqrt(Dx * Dt)

            self.l2distu_abs[iiter] = np.linalg.norm(self.U - U_old) * norm_factor
            norm_U = np.linalg.norm(self.U) * norm_factor
            self.l2distu_rel[iiter] = self.l2distu_abs[iiter] / norm_U if norm_U > 1e-12 else self.l2distu_abs[iiter]

            self.l2distm_abs[iiter] = np.linalg.norm(self.M - M_old) * norm_factor
            norm_M = np.linalg.norm(self.M) * norm_factor
            self.l2distm_rel[iiter] = self.l2distm_abs[iiter] / norm_M if norm_M > 1e-12 else self.l2distm_abs[iiter]

            iter_time = time.time() - iter_start_time

            if solve_config.picard.verbose:
                print(f"   Time: {iter_time:.3f}s")
                print(f"    Errors: U={self.l2distu_rel[iiter]:.2e}, M={self.l2distm_rel[iiter]:.2e}")

            self.iterations_run = iiter + 1

            # Check convergence
            if (iiter + 1) % solve_config.picard.convergence_check_frequency == 0 and (
                self.l2distu_rel[iiter] < solve_config.picard.tolerance
                and self.l2distm_rel[iiter] < solve_config.picard.tolerance
            ):
                convergence_achieved = True
                if solve_config.picard.verbose:
                    print(f"SUCCESS: Convergence achieved after {iiter + 1} iterations!")
                break

        if not convergence_achieved and solve_config.picard.verbose:
            print(f"WARNING:  Max iterations ({solve_config.picard.max_iterations}) reached")
            final_error = max(
                self.l2distu_rel[self.iterations_run - 1],
                self.l2distm_rel[self.iterations_run - 1],
            )
            if final_error > solve_config.picard.tolerance * 10:
                print(" Consider: reducing time step, better initialization, or more iterations")

        # Trim convergence arrays
        self.l2distu_abs = self.l2distu_abs[: self.iterations_run]
        self.l2distm_abs = self.l2distm_abs[: self.iterations_run]
        self.l2distu_rel = self.l2distu_rel[: self.iterations_run]
        self.l2distm_rel = self.l2distm_rel[: self.iterations_run]

        # Mark solution as computed
        self._solution_computed = True
        execution_time = time.time() - solve_start_time

        if solve_config.picard.verbose:
            print(f"\n🏁 Solve completed in {execution_time:.3f}s")

        # Always return SolverResult for type safety and consistency
        # Note: return_structured flag is now deprecated but kept for compatibility warnings
        if not solve_config.return_structured and solve_config.picard.verbose:
            import warnings

            warnings.warn(
                "return_structured=False is deprecated. SolverResult is now always returned, "
                "but it supports tuple unpacking for backward compatibility: "
                "U, M, iterations, err_u, err_m = solver.solve()",
                DeprecationWarning,
                stacklevel=2,
            )

        from mfg_pde.utils.solver_result import create_solver_result

        return create_solver_result(
            U=self.U,
            M=self.M,
            iterations=self.iterations_run,
            error_history_U=self.l2distu_rel,
            error_history_M=self.l2distm_rel,
            solver_name=self.name,
            convergence_achieved=convergence_achieved,
            tolerance=solve_config.picard.tolerance,
            execution_time=execution_time,
            # Rich metadata from configuration
            solver_config=solve_config.to_dict(),
            damping_parameter=solve_config.picard.damping_factor,
            problem_parameters={
                "T": self.problem.T,
                "Nx": self.problem.Nx,
                "Nt": self.problem.Nt,
                "Dx": getattr(self.problem, "Dx", None),
                "Dt": getattr(self.problem, "Dt", None),
            },
        )

    def _cold_start_initialization(self, Nt: int, Nx: int) -> None:
        """Initialize with cold start (default initialization)."""
        # Use backend-native arrays if backend is available
        if self.backend is not None:
            self.U = self.backend.zeros((Nt, Nx))
            self.M = self.backend.zeros((Nt, Nx))
        else:
            self.U = np.zeros((Nt, Nx))
            self.M = np.zeros((Nt, Nx))

        # Set boundary conditions - convert to backend array if needed
        if hasattr(self.problem, "get_initial_m"):
            initial_m_dist = self.problem.get_initial_m()
            if self.backend is not None:
                initial_m_dist = self.backend.from_numpy(initial_m_dist)
            self.M[0, :] = initial_m_dist
            for t in range(1, Nt):
                self.M[t, :] = initial_m_dist

        if hasattr(self.problem, "get_final_u"):
            final_u_cost = self.problem.get_final_u()
            if self.backend is not None:
                final_u_cost = self.backend.from_numpy(final_u_cost)
            self.U[Nt - 1, :] = final_u_cost
            for t in range(Nt - 1):
                self.U[t, :] = final_u_cost

    def _get_warm_start_initialization(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Get warm start initialization data."""
        return self.get_warm_start_data()

    def get_results(self) -> tuple[np.ndarray, np.ndarray]:
        """Get computed U and M solutions."""
        from mfg_pde.utils.exceptions import validate_solver_state

        validate_solver_state(self, "get_results")
        return self.U, self.M

    def get_convergence_data(self) -> tuple:
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

    @property
    def backend(self):
        """Get backend."""
        return self._backend

    @backend.setter
    def backend(self, value):
        """Set backend and propagate to sub-solvers."""
        self._backend = value
        # Propagate to HJB and FP solvers
        if self.hjb_solver is not None:
            self.hjb_solver.backend = value
        if self.fp_solver is not None:
            self.fp_solver.backend = value

    def get_config(self) -> MFGSolverConfig:
        """Get current configuration."""
        return self.config

    def update_config(self, **kwargs: Any) -> None:
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            elif hasattr(self.config.picard, key):
                setattr(self.config.picard, key, value)
            elif hasattr(self.config.hjb, key) or hasattr(self.config.fp, key):
                raise NotImplementedError(f"Updating {key} not yet implemented")
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")

    # Factory methods for common configurations

    @classmethod
    def create_fast(
        cls,
        problem: MFGProblem,
        hjb_solver: BaseHJBSolver,
        fp_solver: BaseFPSolver,
    ) -> ConfigAwareFixedPointIterator:
        """Create iterator optimized for speed."""
        from mfg_pde.config.solver_config import create_fast_config

        return cls(problem, hjb_solver, fp_solver, create_fast_config())

    @classmethod
    def create_accurate(
        cls,
        problem: MFGProblem,
        hjb_solver: BaseHJBSolver,
        fp_solver: BaseFPSolver,
    ) -> ConfigAwareFixedPointIterator:
        """Create iterator optimized for accuracy."""
        from mfg_pde.config.solver_config import create_accurate_config

        return cls(problem, hjb_solver, fp_solver, create_accurate_config())

    @classmethod
    def create_research(
        cls,
        problem: MFGProblem,
        hjb_solver: BaseHJBSolver,
        fp_solver: BaseFPSolver,
    ) -> ConfigAwareFixedPointIterator:
        """Create iterator optimized for research with detailed monitoring."""
        from mfg_pde.config.solver_config import create_research_config

        return cls(problem, hjb_solver, fp_solver, create_research_config())
