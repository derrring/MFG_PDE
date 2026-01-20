"""
Block Iteration Solvers for MFG Systems.

Issue #492 Phase 2: Block methods for coupled HJB-FP systems.

Provides two classical block iteration methods:
- Block Jacobi: Update U and M using values from previous iteration
- Block Gauss-Seidel: Use updated U immediately when computing M

Mathematical Background:
    The MFG system at equilibrium satisfies:
        U = HJB_solve(M)  (value function)
        M = FP_solve(U)   (density)

    Block Jacobi iteration (parallel updates):
        U^{k+1} = HJB_solve(M^k)
        M^{k+1} = FP_solve(U^k)  <- uses OLD U

    Block Gauss-Seidel iteration (sequential updates):
        U^{k+1} = HJB_solve(M^k)
        M^{k+1} = FP_solve(U^{k+1})  <- uses NEW U

    Gauss-Seidel typically converges faster due to using fresh information,
    but Jacobi can be parallelized across HJB and FP solves.

References:
    - Achdou & Capuzzo-Dolcetta (2010): Mean field games: Numerical methods
    - Saad (2003): Iterative Methods for Sparse Linear Systems
"""

from __future__ import annotations

import time
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

from mfg_pde.utils.mfg_logging import get_logger
from mfg_pde.utils.solver_result import SolverResult

from .base_mfg import BaseMFGSolver
from .fixed_point_utils import (
    apply_damping,
    check_convergence_criteria,
    initialize_cold_start,
    preserve_initial_condition,
    preserve_terminal_condition,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mfg_pde.alg.numerical.fp_solvers.base_fp import BaseFPSolver
    from mfg_pde.alg.numerical.hjb_solvers.base_hjb import BaseHJBSolver
    from mfg_pde.core.mfg_problem import MFGProblem

logger = get_logger(__name__)


class BlockMethod(Enum):
    """Block iteration method selection."""

    JACOBI = "jacobi"
    GAUSS_SEIDEL = "gauss_seidel"


class BlockIterator(BaseMFGSolver):
    """
    Block iteration solver for coupled MFG systems.

    Provides unified implementation of Jacobi and Gauss-Seidel block methods
    with optional damping for robustness.

    Required Geometry Traits (Issue #596 Phase 2.3):
        This coupling solver requires trait-validated HJB and FP component solvers:
        - HJB solver must use geometry with SupportsGradient trait
        - FP solver must use geometry with SupportsLaplacian trait

        Trait validation occurs in component solvers, not at coupling layer.
        See HJBFDMSolver and FPFDMSolver docstrings for trait details.

    Args:
        problem: MFG problem definition
        hjb_solver: HJB solver instance
        fp_solver: FP solver instance
        method: Block iteration method ('jacobi' or 'gauss_seidel')
        damping_factor: Damping parameter in (0, 1] (default: 1.0 = no damping)
        diffusion_field: Optional diffusion override
        drift_field: Optional drift override for non-MFG problems

    Example:
        >>> # Block Gauss-Seidel (faster convergence)
        >>> solver = BlockIterator(problem, hjb_solver, fp_solver, method='gauss_seidel')
        >>> result = solver.solve(max_iterations=50, tolerance=1e-5)
        >>>
        >>> # Block Jacobi (can be parallelized)
        >>> solver = BlockIterator(problem, hjb_solver, fp_solver, method='jacobi')
        >>> result = solver.solve(max_iterations=100, tolerance=1e-5)
    """

    def __init__(
        self,
        problem: MFGProblem,
        hjb_solver: BaseHJBSolver,
        fp_solver: BaseFPSolver,
        *,
        method: str | BlockMethod = BlockMethod.GAUSS_SEIDEL,
        damping_factor: float = 1.0,
        diffusion_field: float | NDArray | Any | None = None,
        drift_field: NDArray | Any | None = None,
    ):
        """Initialize block iterator."""
        super().__init__(problem)

        self.hjb_solver = hjb_solver
        self.fp_solver = fp_solver

        # Parse method
        if isinstance(method, str):
            method = BlockMethod(method.lower())
        self.method = method

        # Iteration parameters
        self.damping_factor = damping_factor

        # PDE coefficient overrides
        self.diffusion_field = diffusion_field
        self.drift_field = drift_field

        # Solution state
        self.U: NDArray | None = None
        self.M: NDArray | None = None

        # Convergence tracking
        self.error_history_U: list[float] = []
        self.error_history_M: list[float] = []
        self.iterations_run = 0

        # Cache solver signatures
        self._hjb_sig_params: set[str] | None = None
        self._fp_sig_params: set[str] | None = None
        self._cache_solver_signatures()

        # Cache initial/terminal conditions
        self._M_initial: NDArray | None = None
        self._U_terminal: NDArray | None = None

    def _cache_solver_signatures(self) -> None:
        """Cache solver method signatures for parameter passing."""
        import inspect

        try:
            sig = inspect.signature(self.hjb_solver.solve_hjb_system)
            self._hjb_sig_params = set(sig.parameters.keys())
        except AttributeError:
            self._hjb_sig_params = None

        try:
            sig = inspect.signature(self.fp_solver.solve_fp_system)
            self._fp_sig_params = set(sig.parameters.keys())
        except AttributeError:
            self._fp_sig_params = None

    def _initialize_conditions(self, shape: tuple[int, ...]) -> tuple[NDArray, NDArray]:
        """Initialize initial density and terminal value from problem."""
        # Try get_m_init() / get_u_fin() methods (preferred)
        try:
            M_initial = self.problem.get_m_init()
            if M_initial.shape != shape:
                M_initial = M_initial.reshape(shape)
        except AttributeError:
            try:
                M_initial = self.problem.m_init
                if M_initial is not None and M_initial.shape != shape:
                    M_initial = M_initial.reshape(shape)
            except AttributeError:
                M_initial = np.ones(shape) / np.prod(shape)
                logger.warning("No initial density found, using uniform")

        try:
            U_terminal = self.problem.get_u_fin()
            if U_terminal.shape != shape:
                U_terminal = U_terminal.reshape(shape)
        except AttributeError:
            try:
                U_terminal = self.problem.u_fin
                if U_terminal is not None and U_terminal.shape != shape:
                    U_terminal = U_terminal.reshape(shape)
            except AttributeError:
                U_terminal = np.zeros(shape)

        return M_initial, U_terminal

    def _solve_hjb(
        self,
        M: NDArray,
        U_terminal: NDArray,
        U_prev: NDArray,
    ) -> NDArray:
        """Solve HJB equation with given density."""
        kwargs: dict[str, Any] = {}

        if self._hjb_sig_params is not None:
            if "show_progress" in self._hjb_sig_params:
                kwargs["show_progress"] = False
            if "diffusion_field" in self._hjb_sig_params and self.diffusion_field is not None:
                kwargs["diffusion_field"] = self.diffusion_field

        return self.hjb_solver.solve_hjb_system(M, U_terminal, U_prev, **kwargs)

    def _solve_fp(self, M_initial: NDArray, U: NDArray) -> NDArray:
        """Solve FP equation with given value function."""
        kwargs: dict[str, Any] = {}

        if self._fp_sig_params is not None:
            if "show_progress" in self._fp_sig_params:
                kwargs["show_progress"] = False

            effective_drift = self.drift_field if self.drift_field is not None else U

            if "drift_field" in self._fp_sig_params:
                kwargs["drift_field"] = effective_drift
                if "diffusion_field" in self._fp_sig_params and self.diffusion_field is not None:
                    kwargs["diffusion_field"] = self.diffusion_field
                return self.fp_solver.solve_fp_system(M_initial, **kwargs)
            else:
                return self.fp_solver.solve_fp_system(M_initial, effective_drift, **kwargs)
        else:
            return self.fp_solver.solve_fp_system(M_initial, U)

    def _jacobi_step(
        self,
        U_old: NDArray,
        M_old: NDArray,
        M_initial: NDArray,
        U_terminal: NDArray,
    ) -> tuple[NDArray, NDArray]:
        """
        Perform one Jacobi block iteration.

        Both HJB and FP use values from the previous iteration.

        Returns:
            (U_new, M_new): Updated value function and density
        """
        # Both solves use OLD values (can be parallelized)
        U_new = self._solve_hjb(M_old, U_terminal, U_old)
        M_new = self._solve_fp(M_initial, U_old)  # Uses U_old, not U_new

        return U_new, M_new

    def _gauss_seidel_step(
        self,
        U_old: NDArray,
        M_old: NDArray,
        M_initial: NDArray,
        U_terminal: NDArray,
    ) -> tuple[NDArray, NDArray]:
        """
        Perform one Gauss-Seidel block iteration.

        FP uses the freshly computed U from HJB.

        Returns:
            (U_new, M_new): Updated value function and density
        """
        # First solve HJB with old M
        U_new = self._solve_hjb(M_old, U_terminal, U_old)

        # Then solve FP with NEW U (sequential dependency)
        M_new = self._solve_fp(M_initial, U_new)  # Uses U_new

        return U_new, M_new

    def solve(
        self,
        max_iterations: int = 100,
        tolerance: float = 1e-5,
        verbose: bool = True,
        **kwargs: Any,
    ) -> SolverResult | tuple[NDArray, NDArray, dict[str, Any]]:
        """
        Solve MFG system using block iteration.

        Args:
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance
            verbose: Print progress information
            **kwargs: Additional parameters

        Returns:
            SolverResult with solution and convergence info
        """
        start_time = time.time()

        # Get problem dimensions
        num_time_steps = self.problem.Nt + 1
        shape = tuple(self.problem.geometry.get_grid_shape())
        grid_spacing = self.problem.geometry.get_grid_spacing()[0]
        time_step = self.problem.dt

        # Initialize conditions
        self._M_initial, self._U_terminal = self._initialize_conditions(shape)

        # Initialize solution arrays
        warm_start = self.get_warm_start_data()
        if warm_start is not None:
            self.U, self.M = warm_start
            if verbose:
                print(f"Using warm start ({self.method.value})")
        else:
            self.U = np.zeros((num_time_steps, *shape))
            self.M = np.zeros((num_time_steps, *shape))

            # Set boundary conditions
            self.M[0] = self._M_initial
            self.U[-1] = self._U_terminal

            # Initialize interior
            self.U, self.M = initialize_cold_start(self.U, self.M, self._M_initial, self._U_terminal, num_time_steps)

        # Reset tracking
        self.error_history_U = []
        self.error_history_M = []
        self.iterations_run = 0

        # Select iteration step function
        if self.method == BlockMethod.JACOBI:
            step_fn = self._jacobi_step
            method_name = "Block Jacobi"
        else:
            step_fn = self._gauss_seidel_step
            method_name = "Block Gauss-Seidel"

        if verbose:
            print(f"Starting {method_name} iteration (damping={self.damping_factor})")

        # Progress bar (Issue #587 Protocol pattern)
        from mfg_pde.utils.progress import create_progress_bar

        iter_range = create_progress_bar(
            range(max_iterations),
            verbose=verbose,
            desc=method_name,
        )

        converged = False
        convergence_reason = "Maximum iterations reached"

        for iiter in iter_range:
            iter_start = time.time()

            U_old = self.U.copy()
            M_old = self.M.copy()

            # Perform block iteration step
            U_new, M_new = step_fn(U_old, M_old, self._M_initial, self._U_terminal)

            # Apply damping
            self.U, self.M = apply_damping(U_old, U_new, M_old, M_new, self.damping_factor)

            # Preserve boundary conditions
            self.M = preserve_initial_condition(self.M, self._M_initial)
            self.U = preserve_terminal_condition(self.U, self._U_terminal)

            # Calculate convergence metrics
            from mfg_pde.utils.convergence import calculate_l2_convergence_metrics

            metrics = calculate_l2_convergence_metrics(self.U, U_old, self.M, M_old, grid_spacing, time_step)

            self.error_history_U.append(metrics["l2distu_rel"])
            self.error_history_M.append(metrics["l2distm_rel"])

            iter_time = time.time() - iter_start
            self.iterations_run = iiter + 1

            # Update progress metrics (Issue #587 Protocol - no hasattr needed)
            iter_range.update_metrics(
                U_err=f"{metrics['l2distu_rel']:.2e}",
                M_err=f"{metrics['l2distm_rel']:.2e}",
                t=f"{iter_time:.1f}s",
            )

            # Check convergence
            converged, convergence_reason = check_convergence_criteria(
                metrics["l2distu_rel"],
                metrics["l2distm_rel"],
                metrics["l2distu_abs"],
                metrics["l2distm_abs"],
                tolerance,
            )

            if converged:
                # Log convergence message (Issue #587 Protocol - no hasattr needed)
                iter_range.log(convergence_reason)
                break

        elapsed = time.time() - start_time

        if verbose:
            status = "converged" if converged else "not converged"
            print(f"{method_name} {status} in {self.iterations_run} iterations ({elapsed:.2f}s)")

        self._solution_computed = True

        return SolverResult(
            U=self.U,
            M=self.M,
            iterations=self.iterations_run,
            error_history_U=np.array(self.error_history_U),
            error_history_M=np.array(self.error_history_M),
            solver_name=self.name,
            converged=converged,
            metadata={
                "method": self.method.value,
                "damping_factor": self.damping_factor,
                "convergence_reason": convergence_reason,
                "elapsed_time": elapsed,
            },
        )

    @property
    def name(self) -> str:
        """Solver name for diagnostics."""
        return f"BlockIterator({self.method.value})"

    def get_results(self) -> tuple[NDArray, NDArray]:
        """Get computed solution arrays."""
        if self.U is None or self.M is None:
            raise RuntimeError("Solver has not been run yet. Call solve() first.")
        return self.U, self.M


# Convenience aliases
class BlockJacobiIterator(BlockIterator):
    """
    Block Jacobi iterator for MFG systems.

    Updates U and M using values from the previous iteration only.
    Can be parallelized since HJB and FP solves are independent.

    Example:
        >>> solver = BlockJacobiIterator(problem, hjb_solver, fp_solver)
        >>> result = solver.solve(max_iterations=100)
    """

    def __init__(
        self,
        problem: MFGProblem,
        hjb_solver: BaseHJBSolver,
        fp_solver: BaseFPSolver,
        *,
        damping_factor: float = 0.5,
        **kwargs: Any,
    ):
        super().__init__(
            problem,
            hjb_solver,
            fp_solver,
            method=BlockMethod.JACOBI,
            damping_factor=damping_factor,
            **kwargs,
        )


class BlockGaussSeidelIterator(BlockIterator):
    """
    Block Gauss-Seidel iterator for MFG systems.

    Uses freshly computed U when solving FP (sequential dependency).
    Typically converges faster than Jacobi.

    Example:
        >>> solver = BlockGaussSeidelIterator(problem, hjb_solver, fp_solver)
        >>> result = solver.solve(max_iterations=50)
    """

    def __init__(
        self,
        problem: MFGProblem,
        hjb_solver: BaseHJBSolver,
        fp_solver: BaseFPSolver,
        *,
        damping_factor: float = 0.5,
        **kwargs: Any,
    ):
        super().__init__(
            problem,
            hjb_solver,
            fp_solver,
            method=BlockMethod.GAUSS_SEIDEL,
            damping_factor=damping_factor,
            **kwargs,
        )


if __name__ == "__main__":
    """Quick smoke test for development."""
    # Fix imports for direct script execution
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    print("Testing BlockIterator...")

    from mfg_pde.alg.numerical.coupling.base_mfg import BaseMFGSolver
    from mfg_pde.alg.numerical.coupling.block_iterators import (
        BlockGaussSeidelIterator,
        BlockJacobiIterator,
    )
    from mfg_pde.alg.numerical.fp_solvers import FPFDMSolver
    from mfg_pde.alg.numerical.hjb_solvers import HJBFDMSolver
    from mfg_pde.core.mfg_problem import MFGProblem
    from mfg_pde.geometry import TensorProductGrid

    # Create simple 1D problem
    geometry = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[21])
    problem = MFGProblem(geometry=geometry, T=0.5, Nt=10, diffusion=0.2)

    # Create solvers
    hjb_solver = HJBFDMSolver(problem)
    fp_solver = FPFDMSolver(problem)

    # Test Block Gauss-Seidel
    print("\nTesting Block Gauss-Seidel...")
    gs_solver = BlockGaussSeidelIterator(problem, hjb_solver, fp_solver, damping_factor=0.5)
    result_gs = gs_solver.solve(max_iterations=10, tolerance=1e-4, verbose=True)
    print(f"  Converged: {result_gs.converged}, iterations: {result_gs.iterations}")

    # Test Block Jacobi
    print("\nTesting Block Jacobi...")
    jacobi_solver = BlockJacobiIterator(problem, hjb_solver, fp_solver, damping_factor=0.5)
    result_jacobi = jacobi_solver.solve(max_iterations=10, tolerance=1e-4, verbose=True)
    print(f"  Converged: {result_jacobi.converged}, iterations: {result_jacobi.iterations}")

    # Verify shapes
    assert result_gs.U.shape == (11, 21)
    assert result_gs.M.shape == (11, 21)
    assert np.all(np.isfinite(result_gs.U))
    assert np.all(np.isfinite(result_gs.M))

    print("\nBlockIterator smoke tests passed!")
