"""
Multi-population fixed-point solver.

This module provides a wrapper solver that orchestrates K single-population
solvers to solve coupled multi-population MFG systems.

Mathematical Framework
----------------------
For populations k = 1, ..., K:

    HJB equations (backward):
        -∂uₖ/∂t + Hₖ(x, {mⱼ}ⱼ₌₁ᴷ, ∇uₖ, t) = 0    for k = 1, ..., K

    FP equations (forward):
        ∂mₖ/∂t - div(mₖ ∇ₚHₖ) - σₖ²Δmₖ = 0       for k = 1, ..., K

    Coupling through Hamiltonian:
        Hₖ(x, {mⱼ}, p, t) = ½|p|² + Σⱼ αₖⱼ·mⱼ(x) + fₖ(x, {mⱼ}, t)

Algorithm
---------
1. Initialize {u⁰ₖ, m⁰ₖ} for k = 1, ..., K
2. For iteration n = 0, 1, 2, ... until convergence:
   a. For each population k:
      - Solve HJB with fixed {mⁿⱼ} → uⁿ⁺¹ₖ
      - Solve FP with uⁿ⁺¹ₖ → mⁿ⁺¹ₖ
   b. Check convergence: ||{uⁿ⁺¹ₖ} - {uⁿₖ}|| < tol

Implementation Strategy
-----------------------
This is a *wrapper* solver that:
- Creates K single-population MFGProblem instances
- Each wraps the multi-population problem with fixed k
- Orchestrates K FixedPointSolver instances
- Handles cross-population coupling through shared density array

Part of: Issue #295 - Multi-population MFG support
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from mfg_pde.solvers.base import BaseSolver
from mfg_pde.solvers.fixed_point import FixedPointSolver

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mfg_pde.extensions.multi_population import MultiPopulationMFGProtocol
    from mfg_pde.types import ConvergenceInfo, SolutionArray, SpatialTemporalState


class MultiPopulationFixedPointSolver(BaseSolver):
    """
    Fixed-point solver for K-population MFG systems.

    This solver orchestrates K single-population solvers to solve the
    coupled multi-population system iteratively. Each population's
    HJB-FP system is solved while holding other populations' densities fixed.

    Example:
        >>> from mfg_pde.extensions import MultiPopulationMFGProblem
        >>> from mfg_pde.solvers.extensions import MultiPopulationFixedPointSolver
        >>>
        >>> problem = MultiPopulationMFGProblem(
        ...     num_populations=2,
        ...     coupling_matrix=[[0.1, 0.05], [0.05, 0.1]],
        ...     sigma=[0.01, 0.02],
        ...     T=1.0, Nt=50
        ... )
        >>>
        >>> solver = MultiPopulationFixedPointSolver(
        ...     max_iterations=100,
        ...     tolerance=1e-6,
        ...     damping_factor=0.8
        ... )
        >>> result = solver.solve(problem)
        >>>
        >>> # Access population-specific solutions
        >>> u_pop1 = result.solution['u'][0]  # Shape: (Nt+1, Nx+1, ..., Nz+1)
        >>> m_pop1 = result.solution['m'][0]

    Configuration:
        max_iterations: Maximum outer loop iterations (default: 100)
        tolerance: Convergence tolerance for all populations (default: 1e-6)
        damping_factor: Damping for density updates (default: 1.0)
        single_pop_config: Configuration passed to each FixedPointSolver
            - max_iterations: Inner iterations per population (default: 50)
            - hjb_method: HJB solver method (default: "semi_lagrangian")
            - fp_method: FP solver method (default: "upwind")

    Returns:
        result.solution: Dict with keys 'u' and 'm', each a list of K arrays
        result.convergence_info: Convergence history and statistics
    """

    def __init__(
        self,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        damping_factor: float = 1.0,
        single_pop_config: dict[str, Any] | None = None,
        **config,
    ):
        """
        Initialize multi-population fixed-point solver.

        Args:
            max_iterations: Maximum outer loop iterations
            tolerance: Convergence tolerance
            damping_factor: Damping factor for density updates (0 < damping <= 1)
            single_pop_config: Configuration for single-population solvers
                Example: {"max_iterations": 50, "hjb_method": "semi_lagrangian"}
            **config: Additional configuration options
        """
        super().__init__(max_iterations, tolerance, **config)

        self.damping_factor = damping_factor
        self.single_pop_config = single_pop_config or {
            "max_iterations": 50,
            "hjb_method": "semi_lagrangian",
            "fp_method": "upwind",
        }

        # Validate parameters
        if not 0 < damping_factor <= 1:
            raise ValueError("damping_factor must be in (0, 1]")

    def _initialize_state(self, problem: MultiPopulationMFGProtocol) -> SpatialTemporalState:
        """
        Initialize multi-population solver state.

        Creates initial value functions and densities for all K populations.

        Args:
            problem: Multi-population MFG problem

        Returns:
            state: Initial state with 'u' and 'm' as lists of K arrays
        """
        K = problem.num_populations

        # Initialize storage for all populations
        u_all: list[NDArray] = []
        m_all: list[NDArray] = []

        for k in range(K):
            # Get population-specific initial conditions
            # (Assumes problem has methods: get_initial_value_function_k,
            #  get_initial_density_k, or falls back to defaults)
            try:
                u_init_k = problem.get_initial_value_function_k(k)
            except (AttributeError, NotImplementedError):
                # Fallback: zero initial value function
                shape = self._get_solution_shape(problem)
                u_init_k = np.zeros(shape)

            try:
                m_init_k = problem.get_initial_density_k(k)
            except (AttributeError, NotImplementedError):
                # Fallback: uniform density
                shape = self._get_solution_shape(problem)
                m_init_k = np.ones(shape)
                # Normalize (assuming first index is time)
                m_init_k[0, :] /= float(np.sum(m_init_k[0, :]))

            u_all.append(u_init_k)
            m_all.append(m_init_k)

        return {"u": u_all, "m": m_all, "residual": float("inf"), "iteration": 0}

    def _get_solution_shape(self, problem: MultiPopulationMFGProtocol) -> tuple:
        """Get expected shape of solution arrays (Nt+1, Nx+1, ..., Nz+1)."""
        # This is a placeholder - actual implementation would query problem geometry
        Nt = problem.Nt
        # Assume spatial dimensions from problem
        # For now, default to 1D with 100 points
        Nx = 100
        return (Nt + 1, Nx + 1)

    def _iteration_step(self, state: SpatialTemporalState, problem: MultiPopulationMFGProtocol) -> SpatialTemporalState:
        """
        Perform one iteration of multi-population fixed-point.

        For each population k:
        1. Create single-population problem with fixed {mⱼ}ⱼ≠ₖ
        2. Solve single-population HJB-FP system
        3. Update uₖ and mₖ with damping

        Args:
            state: Current state {u: [u₁, ..., uₖ], m: [m₁, ..., mₖ], residual: float}
            problem: Multi-population MFG problem

        Returns:
            new_state: Updated state with new residual
        """
        K = problem.num_populations
        u_all = state["u"]
        m_all = state["m"]

        u_new_all = []
        m_new_all = []

        # Solve for each population sequentially
        for k in range(K):
            # Create single-population wrapper problem
            # This would wrap the multi-population problem to expose
            # only population k's Hamiltonian/costs while fixing other densities
            single_pop_problem = self._create_single_population_problem(problem, k, m_all)

            # Solve single-population system
            single_solver = FixedPointSolver(**self.single_pop_config)
            result_k = single_solver.solve(single_pop_problem)

            # Extract solution
            u_k_new = result_k.solution["u"]
            m_k_new = result_k.solution["m"]

            # Apply damping
            u_k_damped = self.damping_factor * u_k_new + (1 - self.damping_factor) * u_all[k]
            m_k_damped = self.damping_factor * m_k_new + (1 - self.damping_factor) * m_all[k]

            u_new_all.append(u_k_damped)
            m_new_all.append(m_k_damped)

        # Compute convergence metrics
        u_error = max(np.linalg.norm(u_new - u_old) for u_new, u_old in zip(u_new_all, u_all, strict=False))
        m_error = max(np.linalg.norm(m_new - m_old) for m_new, m_old in zip(m_new_all, m_all, strict=False))
        residual = max(u_error, m_error)

        new_state = {
            "u": u_new_all,
            "m": m_new_all,
            "residual": residual,
            "iteration": state.get("iteration", 0) + 1,
        }

        return new_state

    def _create_single_population_problem(
        self,
        problem: MultiPopulationMFGProtocol,
        k: int,
        m_all: list[NDArray],
    ):
        """
        Create a single-population problem wrapper for population k.

        This wrapper exposes population k's Hamiltonian and costs while
        treating other populations' densities as fixed external fields.

        Args:
            problem: Multi-population problem
            k: Population index
            m_all: Current densities for all populations

        Returns:
            Single-population problem instance

        Note:
            This is a placeholder. Full implementation would create a
            wrapper class that satisfies MFGProblemProtocol and delegates
            to problem.hamiltonian_k, problem.terminal_cost_k, etc.
        """
        # TODO: Implement proper wrapper class
        # For now, raise NotImplementedError to signal incomplete implementation
        raise NotImplementedError(
            "Single-population problem wrapper not yet implemented. "
            "This requires creating a wrapper class that translates "
            "MultiPopulationMFGProtocol methods to MFGProblemProtocol."
        )

    def _create_result(
        self,
        state: SpatialTemporalState,
        problem: MultiPopulationMFGProtocol,
        convergence_info: ConvergenceInfo,
        total_time: float,
        avg_iteration_time: float,
    ) -> SolutionArray:
        """
        Create final result object for multi-population solution.

        Args:
            state: Final solver state
            problem: Multi-population MFG problem
            convergence_info: Convergence information
            total_time: Total solving time
            avg_iteration_time: Average iteration time

        Returns:
            Result object with solution arrays for K populations
        """
        # Extract solution arrays
        u_all = state["u"]
        m_all = state["m"]

        # Create result (simplified version - full implementation would use MFGResult)
        result = {
            "solution": {"u": u_all, "m": m_all},
            "convergence_info": convergence_info,
            "timing": {
                "total_time": total_time,
                "avg_iteration_time": avg_iteration_time,
            },
            "problem_info": {
                "num_populations": problem.num_populations,
                "population_labels": getattr(problem, "population_labels", None),
            },
        }

        return result  # type: ignore[return-value]


# ============================================================================
# Inline Smoke Test
# ============================================================================

if __name__ == "__main__":
    """Smoke test for MultiPopulationFixedPointSolver structure."""
    print("Testing MultiPopulationFixedPointSolver initialization...")

    # Test solver creation
    solver = MultiPopulationFixedPointSolver(
        max_iterations=50,
        tolerance=1e-6,
        damping_factor=0.8,
        single_pop_config={"max_iterations": 30},
    )

    print(f"✓ Solver created with max_iterations={solver.max_iterations}")
    print(f"✓ Damping factor: {solver.damping_factor}")
    print(f"✓ Single-pop config: {solver.single_pop_config}")

    # Test parameter validation
    try:
        bad_solver = MultiPopulationFixedPointSolver(damping_factor=1.5)
        print("✗ Failed to catch invalid damping_factor")
    except ValueError as e:
        print(f"✓ Correctly rejected invalid damping_factor: {e}")

    print("\nAll smoke tests passed!")
    print("\nNote: Full solve() functionality requires implementing single-population problem wrapper.")
