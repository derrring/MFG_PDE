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
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from mfg_pde.utils.mfg_logging import get_logger
from mfg_pde.utils.solver_result import SolverResult

from .base_mfg import BaseMFGSolver
from .fixed_point_utils import (
    check_convergence_criteria,
    initialize_cold_start,
    preserve_initial_condition,
    preserve_terminal_condition,
)

logger = get_logger(__name__)

if TYPE_CHECKING:
    from mfg_pde.alg.numerical.fp_solvers.base_fp import BaseFPSolver
    from mfg_pde.alg.numerical.hjb_solvers.base_hjb import BaseHJBSolver
    from mfg_pde.config import MFGSolverConfig
    from mfg_pde.problem.base_mfg_problem import MFGProblem

# Type alias for iteration callback (Issue #614)
# Signature: callback(iteration, U, M, error_U, error_M) -> bool
IterationCallback = Callable[[int, np.ndarray, np.ndarray, float, float], bool | None]


class FixedPointIterator(BaseMFGSolver):
    """
    Fixed-point iterator for MFG systems with full feature support.

    Features:
    - Config-based parameter management with backward compatibility
    - Optional Anderson acceleration for faster convergence
    - GPU/CPU backend support
    - Structured SolverResult output (tuple output for legacy compatibility)
    - Warm start support
    - State-dependent coefficients (Phase 2.3)

    Required Geometry Traits (Issue #596 Phase 2.3):
        This coupling solver requires trait-validated HJB and FP component solvers:
        - HJB solver must use geometry with SupportsGradient trait
        - FP solver must use geometry with SupportsLaplacian trait

        Trait validation occurs in component solvers, not at coupling layer.
        See HJBFDMSolver and FPFDMSolver docstrings for trait details.

    Args:
        problem: MFG problem definition
        hjb_solver: HJB solver instance (must be trait-validated)
        fp_solver: FP solver instance (must be trait-validated)
        config: Configuration object (preferred modern approach)
        damping_factor: Damping parameter (legacy parameter, overridden by config)
        use_anderson: Enable Anderson acceleration
        anderson_depth: Anderson acceleration memory depth
        anderson_beta: Anderson acceleration mixing parameter
        backend: Backend name ('numpy', 'torch', 'jax', etc.)
        diffusion_field: Optional diffusion override (float, array, or callable)
            - None: Use problem.sigma (default)
            - float: Constant diffusion
            - ndarray: Spatially/temporally varying diffusion
            - Callable: State-dependent diffusion D(t, x, m) -> float | ndarray
        drift_field: Optional drift override for non-MFG problems (array or callable)
            - None: Use MFG drift (default, drift from U)
            - ndarray: Precomputed drift field
            - Callable: State-dependent drift α(t, x, m) -> ndarray
    """

    def __init__(
        self,
        problem: MFGProblem,
        hjb_solver: BaseHJBSolver,
        fp_solver: BaseFPSolver,
        config: MFGSolverConfig | None = None,
        damping_factor: float = 0.5,  # Renamed from thetaUM
        damping_factor_M: float | None = None,  # Issue #719: Per-variable damping
        use_anderson: bool = False,
        anderson_depth: int = 5,
        anderson_beta: float = 1.0,
        backend: str | None = None,
        diffusion_field: float | np.ndarray | Any | None = None,  # Phase 2.3
        drift_field: np.ndarray | Any | None = None,  # Phase 2.3
    ):
        """
        Args:
            damping_factor: Damping for U (theta_U). Default 0.5.
            damping_factor_M: Damping for M (theta_M). If None, uses damping_factor.
                Issue #719: Per-variable damping support.
                Recommended for MFG: damping_factor=1.0, damping_factor_M=0.2
                (U adapts fully, M filters particle noise)
        """
        super().__init__(problem)
        self.backend = backend
        self.hjb_solver = hjb_solver
        self.fp_solver = fp_solver
        self.config = config

        # PDE coefficient overrides (Phase 2.3)
        self.diffusion_field = diffusion_field
        self.drift_field = drift_field

        # Anderson acceleration support
        self.use_anderson = use_anderson
        self.anderson_accelerator = None
        if use_anderson:
            from mfg_pde.utils.numerical.anderson_acceleration import AndersonAccelerator

            self.anderson_accelerator = AndersonAccelerator(depth=anderson_depth, beta=anderson_beta)

        # Damping parameters (overridden by config if provided)
        # Issue #719: Per-variable damping support
        self.damping_factor = damping_factor
        self.damping_factor_M = damping_factor_M  # None = use damping_factor for both

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

        # Cache signature info for solver interfaces (performance optimization)
        self._hjb_sig_params: set[str] | None = None
        self._fp_sig_params: set[str] | None = None
        self._cache_solver_signatures()

    def _cache_solver_signatures(self) -> None:
        """
        Cache solver method signatures to avoid repeated inspect calls.

        Issue #543 Phase 2: Eliminates hasattr checks for solver methods.
        """
        import inspect

        # Try to get HJB solver signature
        try:
            sig = inspect.signature(self.hjb_solver.solve_hjb_system)
            self._hjb_sig_params = set(sig.parameters.keys())
        except AttributeError:
            # Solver doesn't have solve_hjb_system method
            self._hjb_sig_params = None

        # Try to get FP solver signature
        try:
            sig = inspect.signature(self.fp_solver.solve_fp_system)
            self._fp_sig_params = set(sig.parameters.keys())
        except AttributeError:
            # Solver doesn't have solve_fp_system method
            self._fp_sig_params = None

    def _get_initial_and_terminal_conditions(self, shape: tuple) -> tuple[np.ndarray, np.ndarray]:
        """
        Retrieve initial density and terminal value function from problem.

        Issue #543 Phase 2: Centralizes initial/terminal condition retrieval
        with 4-priority cascade (eliminates 8 hasattr checks).

        Args:
            shape: Spatial grid shape

        Returns:
            (M_initial, U_terminal): Initial density and terminal value function

        Priority order:
            1. get_m_init() / get_u_fin() methods (preferred modern API)
            2. m_init / u_fin attributes (legacy direct access)
            3. get_initial_m() / get_final_u() methods (alternate modern API)
            4. initial_density() / terminal_cost() callables (functional API)
        """
        # Priority 1: get_m_init() / get_u_fin() methods
        try:
            M_initial = self.problem.get_m_init()
            if M_initial.shape != shape:
                M_initial = M_initial.reshape(shape)

            try:
                U_terminal = self.problem.get_u_fin()
            except AttributeError:
                # No terminal condition - use zeros
                U_terminal = np.zeros(shape)

            if U_terminal.shape != shape:
                U_terminal = U_terminal.reshape(shape)

            return M_initial, U_terminal
        except AttributeError:
            pass  # Try next priority

        # Priority 2: m_initial / u_final attributes (Issue #670: unified naming)
        try:
            M_initial = self.problem.m_initial
            if M_initial is not None:
                if M_initial.shape != shape:
                    M_initial = M_initial.reshape(shape)

                try:
                    U_terminal = self.problem.u_final
                except AttributeError:
                    U_terminal = np.zeros(shape)

                if U_terminal.shape != shape:
                    U_terminal = U_terminal.reshape(shape)

                return M_initial, U_terminal
        except AttributeError:
            pass  # Try next priority

        # Priority 3: get_initial_m() / get_final_u() methods
        try:
            M_initial = self.problem.get_initial_m()
            U_terminal = self.problem.get_final_u()
            return M_initial, U_terminal
        except AttributeError:
            pass  # Try next priority

        # Priority 4: initial_density() / terminal_cost() callables
        try:
            x_grid = self.problem.geometry.get_spatial_grid()
            M_initial = self.problem.initial_density(x_grid).reshape(shape)
            U_terminal = self.problem.terminal_cost(x_grid).reshape(shape)
            return M_initial, U_terminal
        except AttributeError as e:
            raise ValueError(
                "Problem must provide initial/terminal conditions via one of:\n"
                "  1. get_m_init()/get_u_fin() methods (preferred)\n"
                "  2. m_init/u_fin attributes\n"
                "  3. get_initial_m()/get_final_u() methods\n"
                "  4. initial_density()/terminal_cost() callables"
            ) from e

    def solve(
        self,
        config: MFGSolverConfig | None = None,
        max_iterations: int | None = None,
        tolerance: float | None = None,
        return_tuple: bool = False,
        iteration_callback: IterationCallback | None = None,
        **kwargs: Any,
    ) -> SolverResult | tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]:
        """
        Solve coupled MFG system using fixed-point iteration.

        Args:
            config: Solver configuration (overrides instance config)
            max_iterations: Maximum iterations (legacy parameter)
            tolerance: Convergence tolerance (legacy parameter)
            return_tuple: Return legacy tuple format instead of SolverResult
            iteration_callback: Optional callback called after each Picard iteration.
                Signature: callback(iteration, U, M, error_U, error_M) -> bool
                Return True to continue, False to stop early.
                If None (default), no callback is invoked.
            **kwargs: Additional parameters for backward compatibility

        Returns:
            SolverResult object (or tuple if return_tuple=True)

        Example:
            >>> def monitor(i, U, M, err_U, err_M):
            ...     print(f"Iteration {i}: err_U={err_U:.2e}, err_M={err_M:.2e}")
            ...     # Save checkpoint every 10 iterations
            ...     if i % 10 == 0:
            ...         np.save(f"checkpoint_{i}.npy", {"U": U, "M": M})
            ...     return True  # Continue
            >>> result = solver.solve(iteration_callback=monitor)
        """
        # Use provided config or fall back to instance config
        solve_config = config or self.config

        # Parameter resolution (config > explicit args > instance defaults)
        if solve_config is not None:
            final_max_iterations = solve_config.picard.max_iterations
            final_tolerance = solve_config.picard.tolerance
            final_damping_factor = solve_config.picard.damping_factor
            # Issue #719: Per-variable damping - config doesn't have this yet, use instance
            final_damping_factor_M = self.damping_factor_M
            verbose = solve_config.picard.verbose
        else:
            # Legacy parameter precedence
            final_max_iterations = (
                max_iterations or kwargs.get("max_picard_iterations") or kwargs.get("Niter_max") or 100
            )
            final_tolerance = tolerance or kwargs.get("picard_tolerance") or kwargs.get("l2errBoundPicard") or 1e-6
            final_damping_factor = self.damping_factor
            final_damping_factor_M = self.damping_factor_M  # Issue #719
            verbose = True

        # Get problem dimensions - handle both old 1D and new nD interfaces
        num_time_steps = self.problem.Nt + 1  # Renamed from Nt

        # Detect problem shape using geometry API
        from mfg_pde.geometry.base import CartesianGrid

        # Issue #543 Phase 2: Replace hasattr with try/except
        try:
            geometry = self.problem.geometry
        except AttributeError as e:
            raise ValueError("Problem must have 'geometry' attribute") from e

        if geometry is None:
            raise ValueError("Problem geometry cannot be None")

        if not isinstance(geometry, CartesianGrid):
            raise ValueError("Problem geometry must be CartesianGrid (TensorProductGrid)")

        shape = tuple(self.problem.geometry.get_grid_shape())
        grid_spacing = self.problem.geometry.get_grid_spacing()[0]  # For compatibility
        time_step = self.problem.dt

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

            # Get initial density and terminal condition
            # Issue #543 Phase 2: Use centralized helper (eliminates 8 hasattr checks)
            M_initial, U_terminal = self._get_initial_and_terminal_conditions(shape)

            if num_time_steps > 0:
                # Set boundary conditions
                if len(shape) == 1:
                    self.M[0, :] = M_initial
                    self.U[num_time_steps - 1, :] = U_terminal
                else:
                    self.M[0] = M_initial
                    self.U[num_time_steps - 1] = U_terminal

                # Initialize interior with boundary conditions
                self.U, self.M = initialize_cold_start(self.U, self.M, M_initial, U_terminal, num_time_steps)

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
        # M_initial already computed above

        # Hierarchical progress for Picard iterations (Issue #614)
        from mfg_pde.utils.progress import HierarchicalProgress

        with HierarchicalProgress(verbose=verbose) as progress:
            # Add main Picard task with initial metrics
            picard_task = progress.add_task(
                "MFG Picard",
                total=final_max_iterations,
                iter=f"0/{final_max_iterations}",
                err_U=0.0,
                err_M=0.0,
            )

            for iiter in range(final_max_iterations):
                iter_start = time.time()

                U_old = self.U.copy()
                M_old = self.M.copy()

                # Build iteration state for BC provider resolution (Issue #625)
                # This state is passed to BCValueProvider.compute() for dynamic BCs
                bc_resolution_state = {
                    "m_current": M_old,
                    "U_current": U_old,
                    "geometry": self.problem.geometry,
                    "sigma": getattr(self.problem, "sigma", None),
                    "iteration": iiter,
                }

                # 1. Solve HJB backward with current M (transient subtask)
                # Issue #614: Use hierarchical subtask for inner solver visibility
                # Issue #625: Resolve BC providers before HJB solve
                with (
                    progress.subtask("HJB", total=num_time_steps) as hjb_subtask,
                    self.problem.using_resolved_bc(bc_resolution_state),
                ):
                    # Issue #543 Phase 2: Use cached signature instead of hasattr
                    if self._hjb_sig_params is not None:
                        # Method exists and signature is cached
                        params = self._hjb_sig_params
                        kwargs = {}
                        if "show_progress" in params:
                            kwargs["show_progress"] = False  # Subtask replaces inner bar
                        if "diffusion_field" in params and self.diffusion_field is not None:
                            kwargs["diffusion_field"] = self.diffusion_field
                        # Issue #640: Pass progress callback for incremental updates
                        if "progress_callback" in params:
                            kwargs["progress_callback"] = hjb_subtask.advance

                        U_new = self.hjb_solver.solve_hjb_system(M_old, U_terminal, U_old, **kwargs)
                    else:
                        # No signature cached - use basic call
                        U_new = self.hjb_solver.solve_hjb_system(M_old, U_terminal, U_old)
                        # Fallback: advance all at once if no callback support
                        hjb_subtask.advance(num_time_steps)

                # 2. Solve FP forward with new U (transient subtask)
                # Issue #614: Use hierarchical subtask for inner solver visibility
                with progress.subtask("FP", total=num_time_steps) as fp_subtask:
                    if self._fp_sig_params is not None:
                        # Standard mode: FP builds its own matrix
                        # Issue #543 Phase 2: Use cached signature instead of hasattr
                        # Method exists and signature is cached
                        params = self._fp_sig_params
                        kwargs = {}
                        if "show_progress" in params:
                            kwargs["show_progress"] = False  # Subtask replaces inner bar

                        # Determine drift field: override or MFG drift from U
                        if self.drift_field is not None:
                            # User-provided drift override (for non-MFG problems)
                            effective_drift = self.drift_field
                        else:
                            # Standard MFG: drift from U
                            effective_drift = U_new

                        if "drift_field" in params:
                            kwargs["drift_field"] = effective_drift
                        else:
                            # Legacy: positional argument for U
                            # solve_fp_system(m_initial, U_drift, **kwargs)
                            pass  # Will use positional argument below

                        if "diffusion_field" in params and self.diffusion_field is not None:
                            kwargs["diffusion_field"] = self.diffusion_field

                        # Issue #640: Pass progress callback for incremental updates
                        if "progress_callback" in params:
                            kwargs["progress_callback"] = fp_subtask.advance

                        # Call with appropriate arguments
                        if "drift_field" in params:
                            M_new = self.fp_solver.solve_fp_system(M_initial, **kwargs)
                        else:
                            # Legacy interface: second positional arg is U for drift
                            M_new = self.fp_solver.solve_fp_system(M_initial, effective_drift, **kwargs)

                        # If no callback support, advance all at once
                        if "progress_callback" not in params:
                            fp_subtask.advance(num_time_steps)
                    else:
                        # No signature cached - use basic call
                        M_new = self.fp_solver.solve_fp_system(M_initial, U_new)
                        # Fallback: advance all at once
                        fp_subtask.advance(num_time_steps)

                # 3. Apply damping or Anderson acceleration
                # Issue #719: Per-variable damping support
                theta_M = final_damping_factor_M if final_damping_factor_M is not None else final_damping_factor

                if self.use_anderson and self.anderson_accelerator is not None:
                    # Anderson acceleration on U only (M uses standard damping for positivity)
                    x_current_U = U_old.flatten()
                    f_current_U = U_new.flatten()
                    x_next_U = self.anderson_accelerator.update(x_current_U, f_current_U, method="type1")
                    self.U = x_next_U.reshape(U_old.shape)

                    # Standard damping for M (guarantees non-negativity and mass conservation)
                    self.M = theta_M * M_new + (1 - theta_M) * M_old
                else:
                    # Standard damping for both - Issue #719: separate factors
                    self.U = final_damping_factor * U_new + (1 - final_damping_factor) * U_old
                    self.M = theta_M * M_new + (1 - theta_M) * M_old

                # Preserve boundary conditions
                self.M = preserve_initial_condition(self.M, M_initial)
                self.U = preserve_terminal_condition(self.U, U_terminal)

                # Issue #688: Early termination on NaN/Inf (runtime safety)
                if not np.all(np.isfinite(self.U)) or not np.all(np.isfinite(self.M)):
                    convergence_reason = "diverged_nan"
                    logger.warning(
                        "NaN/Inf detected in iteration %d. Terminating early.",
                        iiter + 1,
                    )
                    self.iterations_run = iiter + 1
                    break

                # Calculate convergence metrics
                from mfg_pde.utils.convergence import calculate_l2_convergence_metrics

                metrics = calculate_l2_convergence_metrics(self.U, U_old, self.M, M_old, grid_spacing, time_step)
                self.l2distu_abs[iiter] = metrics["l2distu_abs"]
                self.l2distu_rel[iiter] = metrics["l2distu_rel"]
                self.l2distm_abs[iiter] = metrics["l2distm_abs"]
                self.l2distm_rel[iiter] = metrics["l2distm_rel"]

                iter_time = time.time() - iter_start
                self.iterations_run = iiter + 1

                # Update main task progress with metrics (Issue #614)
                progress.update(
                    picard_task,
                    iter=f"{iiter + 1}/{final_max_iterations}",
                    err_U=self.l2distu_rel[iiter],
                    err_M=self.l2distm_rel[iiter],
                    time=f"{iter_time:.1f}s",
                )

                # Issue #614: Invoke user callback if provided
                if iteration_callback is not None:
                    should_continue = iteration_callback(
                        iiter,
                        self.U,
                        self.M,
                        self.l2distu_rel[iiter],
                        self.l2distm_rel[iiter],
                    )
                    if should_continue is False:
                        converged = True
                        convergence_reason = "callback_stopped"
                        break

                # Check convergence
                converged, convergence_reason = check_convergence_criteria(
                    self.l2distu_rel[iiter],
                    self.l2distm_rel[iiter],
                    self.l2distu_abs[iiter],
                    self.l2distm_abs[iiter],
                    final_tolerance,
                )

                if converged:
                    break

        # Build metadata
        metadata = {
            "convergence_reason": convergence_reason,
            "l2distu_rel": self.l2distu_rel[: self.iterations_run],
            "l2distm_rel": self.l2distm_rel[: self.iterations_run],
            "anderson_used": self.use_anderson,
        }

        # Issue #688: Validate final solver output
        from mfg_pde.utils.validation.runtime import validate_solver_output

        output_validation = validate_solver_output(
            self.U,
            self.M,
            check_finite=True,
            check_density_positive=True,
        )
        if not output_validation.is_valid:
            metadata["output_validation"] = {
                "is_valid": False,
                "issues": [str(issue) for issue in output_validation.issues],
            }
            logger.warning(
                "Solver output validation failed: %s",
                "; ".join(str(i) for i in output_validation.issues),
            )

        # Construct result
        result = SolverResult(
            U=self.U,
            M=self.M,
            iterations=self.iterations_run,
            error_history_U=self.l2distu_abs[: self.iterations_run],
            error_history_M=self.l2distm_abs[: self.iterations_run],
            solver_name=self.name,
            converged=converged,
            metadata=metadata,
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


if __name__ == "__main__":
    """Quick smoke test for development."""
    print("Testing FixedPointIterator...")

    # Test class availability
    assert FixedPointIterator is not None
    print("  FixedPointIterator class available")

    # Full smoke test requires complete solver setup
    # See examples/basic/ for usage examples

    print("Smoke tests passed!")
