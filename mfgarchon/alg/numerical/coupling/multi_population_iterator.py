"""
Multi-population Picard iterator for K-population MFG.

Issue #910 Phase 2: Coordinates K single-population solves in the
Picard fixed-point loop. Each iteration:
  1. Solve K HJB equations (each sees all current densities)
  2. Extract K drift fields via H_k.optimal_control()
  3. Solve K FP equations
  4. Damp all K density fields

Reuses standard HJB/FP solvers — this class only coordinates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mfgarchon.utils.mfg_logging import get_logger

if TYPE_CHECKING:
    from mfgarchon.alg.numerical.fp_solvers.base_fp import BaseFPSolver
    from mfgarchon.alg.numerical.hjb_solvers.base_hjb import BaseHJBSolver
    from mfgarchon.core.multi_population import MultiPopulationProblem

logger = get_logger(__name__)


class MultiPopulationIterator:
    """Picard iteration for K-population MFG.

    Parameters
    ----------
    multi_problem : MultiPopulationProblem
        Container holding K single-population problems.
    hjb_solvers : list[BaseHJBSolver]
        One HJB solver per population.
    fp_solvers : list[BaseFPSolver]
        One FP solver per population.
    damping_factor : float
        Picard damping parameter omega in (0, 1]. Default 0.5.

    Examples
    --------
    >>> iterator = MultiPopulationIterator(
    ...     multi_problem=multi,
    ...     hjb_solvers=[hjb_A, hjb_B],
    ...     fp_solvers=[fp_A, fp_B],
    ... )
    >>> result = iterator.solve(max_iterations=50, tolerance=1e-6)
    >>> result.U  # list of K value functions
    >>> result.M  # list of K density fields
    """

    def __init__(
        self,
        multi_problem: MultiPopulationProblem,
        hjb_solvers: list[BaseHJBSolver],
        fp_solvers: list[BaseFPSolver],
        damping_factor: float = 0.5,
    ):
        self.multi_problem = multi_problem
        self.hjb_solvers = hjb_solvers
        self.fp_solvers = fp_solvers
        self.damping_factor = damping_factor
        K = multi_problem.K

        if len(hjb_solvers) != K:
            raise ValueError(f"Need {K} HJB solvers, got {len(hjb_solvers)}")
        if len(fp_solvers) != K:
            raise ValueError(f"Need {K} FP solvers, got {len(fp_solvers)}")

    def solve(
        self,
        max_iterations: int = 50,
        tolerance: float = 1e-6,
    ) -> MultiPopulationResult:
        """Run Picard iteration over K populations.

        Returns
        -------
        MultiPopulationResult
            Contains U (list of value functions), M (list of densities),
            iterations, and convergence info.
        """
        K = self.multi_problem.K
        omega = self.damping_factor

        # Initialize from each population's problem
        M = []
        U = []
        for k in range(K):
            prob_k = self.multi_problem.get_population(k)
            Nt = prob_k.Nt
            grid_shape = prob_k.geometry.get_grid_shape()
            Nx = grid_shape[0] if grid_shape else getattr(prob_k, "num_nodes", 10)

            # Initial density from problem (or uniform fallback)
            try:
                m0_k = prob_k.get_initial_m()
            except Exception:
                m0_k = np.ones(Nx) / Nx

            M_k = np.zeros((Nt + 1, Nx))
            M_k[0] = m0_k
            for n in range(1, Nt + 1):
                M_k[n] = m0_k
            M.append(M_k)

            # Terminal value from problem (or zero fallback)
            try:
                U_terminal_k = prob_k.get_final_u()
            except Exception:
                U_terminal_k = np.zeros(Nx)

            U_k = np.zeros((Nt + 1, Nx))
            U_k[-1] = U_terminal_k
            U.append(U_k)

        # Picard iteration
        converged = False
        for iteration in range(max_iterations):
            M_old = [m.copy() for m in M]

            # Step 1: Solve K HJB equations
            for k in range(K):
                prob_k = self.multi_problem.get_population(k)
                # Pass current density of all populations
                # For now, pass only population k's density to its HJB solver
                # (cross-coupling enters through H_k's coupling function)
                U_terminal_k = U[k][-1]  # terminal condition
                try:
                    U[k] = self.hjb_solvers[k].solve_hjb_system(M[k], U_terminal_k, U[k])
                except Exception as e:
                    logger.warning(f"HJB solver for population {k} failed: {e}")

            # Step 2: Solve K FP equations with drift from H_k
            for k in range(K):
                prob_k = self.multi_problem.get_population(k)
                m0_k = M[k][0]

                # Compute drift via H_k.optimal_control (Issue #896 pattern)
                H_k = prob_k.hamiltonian_class
                if H_k is not None:
                    effective_drift = self._compute_drift_field(U[k], M[k], H_k, prob_k)
                else:
                    effective_drift = U[k]

                try:
                    M_new_k = self.fp_solvers[k].solve_fp_system(m0_k, drift_field=effective_drift, show_progress=False)
                    # Damp
                    M[k] = (1 - omega) * M_old[k] + omega * M_new_k
                except Exception as e:
                    logger.warning(f"FP solver for population {k} failed: {e}")

            # Check convergence
            errors = []
            for k in range(K):
                err_k = np.max(np.abs(M[k] - M_old[k]))
                errors.append(err_k)
            max_error = max(errors)

            logger.info(
                f"Multi-pop iter {iteration + 1}/{max_iterations}: "
                f"max_err={max_error:.4e}, per_pop={[f'{e:.2e}' for e in errors]}"
            )

            if max_error < tolerance:
                converged = True
                break

        return MultiPopulationResult(
            U=U,
            M=M,
            iterations=iteration + 1,
            converged=converged,
            errors=errors,
            population_names=self.multi_problem.population_names,
        )

    @staticmethod
    def _compute_drift_field(U, M, H_class, problem):
        """Compute drift field from H.optimal_control.

        Dispatches on problem type:
        - Network (spatial_dimension == 0): return U directly (FP network
          solver handles drift extraction internally)
        - Continuous: synthetic U whose differences reproduce alpha*
        """
        # Issue #915: dispatch on problem type
        spatial_dim = getattr(problem, "spatial_dimension", None)
        if spatial_dim == 0:
            # Network problem: pass U directly.
            # FPNetworkSolver computes flows from U differences.
            # TODO (#913): FPNetworkSolver should use H.optimal_control()
            return U

        # For smooth separable H, the FP solver's internal drift extraction
        # (-coupling_coefficient * ∇U) is already correct. Skip synthetic-U.
        from mfgarchon.core.hamiltonian import SeparableHamiltonian

        if isinstance(H_class, SeparableHamiltonian) and H_class.control_cost.is_smooth():
            return U

        # Non-smooth H, 1D only: synthetic U approach
        if U.ndim > 2:
            return U  # nD non-smooth: deferred

        # Continuous problem: synthetic U approach (same as FixedPointIterator)
        geometry = problem.geometry
        grid_spacing = geometry.get_grid_spacing()
        dx = grid_spacing[0]
        dt = problem.dt
        Nt = U.shape[0]
        Nx = U.shape[-1]
        coupling_coefficient = getattr(problem, "coupling_coefficient", 1.0)

        grad_U = np.gradient(U, dx, axis=-1)
        bounds = geometry.get_bounds()
        x_grid = np.linspace(bounds[0][0], bounds[1][0], Nx).reshape(-1, 1)

        alpha_field = np.zeros_like(grad_U)
        for n in range(Nt):
            p = grad_U[n]
            m_n = M[n] if n < M.shape[0] else M[-1]
            alpha_field[n] = H_class.optimal_control(x_grid, m_n, p.reshape(-1, 1), t=n * dt).ravel()

        alpha_mid = 0.5 * (alpha_field[:, :-1] + alpha_field[:, 1:])
        increments = -alpha_mid * dx / coupling_coefficient
        U_synthetic = np.zeros_like(U)
        U_synthetic[:, 1:] = np.cumsum(increments, axis=-1)
        return U_synthetic


class MultiPopulationResult:
    """Result of multi-population MFG solve.

    Attributes
    ----------
    U : list[np.ndarray]
        Value functions, one per population. Each shape (Nt+1, Nx).
    M : list[np.ndarray]
        Density fields, one per population. Each shape (Nt+1, Nx).
    iterations : int
        Number of Picard iterations performed.
    converged : bool
        Whether tolerance was reached.
    errors : list[float]
        Final per-population errors.
    population_names : list[str]
        Names of populations.
    """

    def __init__(self, U, M, iterations, converged, errors, population_names):
        self.U = U
        self.M = M
        self.iterations = iterations
        self.converged = converged
        self.errors = errors
        self.population_names = population_names

    def __repr__(self):
        status = "converged" if self.converged else "not converged"
        return f"MultiPopulationResult({self.K} populations, {self.iterations} iterations, {status})"

    @property
    def K(self) -> int:
        return len(self.U)
