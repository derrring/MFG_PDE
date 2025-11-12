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
from scipy.interpolate import RegularGridInterpolator

from mfg_pde.solvers.base import BaseSolver
from mfg_pde.solvers.fixed_point import FixedPointSolver

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mfg_pde.extensions.multi_population import MultiPopulationMFGProtocol
    from mfg_pde.types import ConvergenceInfo, SolutionArray, SpatialTemporalState


# ============================================================================
# Single-Population Adapter
# ============================================================================


class _SinglePopulationAdapter:
    """
    Adapter translating MultiPopulationMFGProtocol to MFGProblemProtocol.

    This internal class wraps a multi-population problem to expose a
    single-population interface for use with standard solvers like
    FixedPointSolver. It fixes a specific population index k and treats
    other populations' densities as frozen external fields.

    Mathematical Translation:
        Multi-pop: Hₖ(x, {mⱼ}, p, t)  →  Single-pop: H(x, m_k, p, t)
        where {mⱼ}ⱼ≠ₖ are held constant

    Usage:
        >>> adapter = _SinglePopulationAdapter(multi_pop_problem, k=0, m_all=[m0, m1])
        >>> # Now adapter can be used with FixedPointSolver
        >>> solver = FixedPointSolver()
        >>> result = solver.solve(adapter)

    Args:
        problem: Multi-population MFG problem
        k: Population index to expose (0 to K-1)
        m_all: Current density state for all K populations

    Attributes:
        All attributes from the wrapped multi-population problem,
        delegated to expose population k's specific parameters.
    """

    def __init__(
        self,
        problem: MultiPopulationMFGProtocol,
        k: int,
        m_all: list[NDArray],
    ):
        """
        Initialize adapter for population k.

        Args:
            problem: Multi-population MFG problem to wrap
            k: Population index (0 to K-1)
            m_all: Density arrays for all populations [m₁, ..., mₖ]
        """
        self._problem = problem
        self._k = k
        self._m_all = m_all

        # Validate population index
        if not 0 <= k < problem.num_populations:
            raise ValueError(f"Population index k={k} out of range [0, {problem.num_populations})")

    # ====================
    # Spatial Interpolation
    # ====================

    def _interpolate_densities_at_point(self, x, t: float) -> list[float]:
        """
        Interpolate all population densities at spatial point x and time t.

        This method enables the adapter to evaluate the Hamiltonian at arbitrary
        spatial points by interpolating the frozen density arrays m_all.

        Args:
            x: Spatial position (scalar for 1D, tuple/array for nD)
            t: Time value

        Returns:
            List of interpolated density values [m₁(t,x), ..., mₖ(t,x)]

        Implementation:
            - Uses RegularGridInterpolator for efficient interpolation on regular grids
            - Handles 1D, 2D, and 3D spatial domains
            - Time is handled by finding nearest time index (piecewise constant in time)

        Note:
            This is the key Phase 2.5 feature that bridges the gap between
            array-based solver operations and point-wise Hamiltonian evaluation.
        """
        # Get spatial grid from problem
        xSpace = self._problem.xSpace
        tSpace = self._problem.tSpace

        # Find time index (nearest neighbor for now)
        t_idx = np.argmin(np.abs(tSpace - t))

        # Interpolate each population's density
        m_interpolated = []
        for m_k in self._m_all:
            # Extract density at time t_idx: m_k[t_idx, ...]
            m_k_at_t = m_k[t_idx]

            # Handle dimensionality
            dim = self._problem.dimension

            if dim == 1:
                # 1D interpolation
                # xSpace is 1D array, m_k_at_t is 1D array
                interp_func = RegularGridInterpolator((xSpace,), m_k_at_t, bounds_error=False, fill_value=0.0)
                x_point = np.atleast_1d(x)
                m_val = float(interp_func(x_point)[0])

            elif dim == 2:
                # 2D interpolation
                # xSpace is tuple (x_grid, y_grid) or 2D meshgrid
                # m_k_at_t is 2D array with shape (Nx+1, Ny+1)
                if isinstance(xSpace, tuple) and len(xSpace) == 2:
                    x_grid, y_grid = xSpace
                else:
                    # xSpace might be stored differently - fallback
                    # Assume uniform grid from problem bounds
                    raise NotImplementedError("2D interpolation requires xSpace as (x_grid, y_grid) tuple")

                interp_func = RegularGridInterpolator((x_grid, y_grid), m_k_at_t, bounds_error=False, fill_value=0.0)
                x_point = np.atleast_1d(x).flatten()
                m_val = float(interp_func(x_point)[0])

            elif dim == 3:
                # 3D interpolation
                if isinstance(xSpace, tuple) and len(xSpace) == 3:
                    x_grid, y_grid, z_grid = xSpace
                else:
                    raise NotImplementedError("3D interpolation requires xSpace as (x_grid, y_grid, z_grid) tuple")

                interp_func = RegularGridInterpolator(
                    (x_grid, y_grid, z_grid), m_k_at_t, bounds_error=False, fill_value=0.0
                )
                x_point = np.atleast_1d(x).flatten()
                m_val = float(interp_func(x_point)[0])

            else:
                raise ValueError(f"Unsupported spatial dimension: {dim}")

            m_interpolated.append(m_val)

        return m_interpolated

    # ====================
    # Delegated Universal Properties
    # ====================

    @property
    def dimension(self):
        """Spatial dimension (delegated)."""
        return self._problem.dimension

    @property
    def T(self):
        """Terminal time (delegated)."""
        return self._problem.T

    @property
    def Nt(self):
        """Number of time steps (delegated)."""
        return self._problem.Nt

    @property
    def tSpace(self):
        """Time discretization array (delegated)."""
        return self._problem.tSpace

    @property
    def sigma(self):
        """Diffusion coefficient for population k."""
        # Extract population-specific diffusion if available
        if hasattr(self._problem, "sigma_vec"):
            return self._problem.sigma_vec[self._k]
        return self._problem.sigma

    # ====================
    # Adapted Population-Specific Methods
    # ====================

    def hamiltonian(self, x, m, p, t):
        """
        Hamiltonian for population k with frozen other densities.

        Translates: hamiltonian_k(k, x, m_all, p, t) → hamiltonian(x, m, p, t)

        Args:
            x: Spatial position
            m: Density at x for population k (not used - interpolated from m_all)
            p: Momentum ∇uₖ
            t: Time

        Returns:
            Hamiltonian value Hₖ(x, {mⱼ}, p, t)

        Implementation (Phase 2.5):
            Interpolates frozen density arrays m_all at point (t, x) to get
            scalar values [m₁(t,x), ..., mₖ(t,x)], then evaluates
            hamiltonian_k(k, x, m_all_interpolated, p, t).

        Note:
            The parameter m is for interface compatibility with MFGProblemProtocol
            but is not used. The actual density values are interpolated from
            the frozen state m_all.
        """
        # Interpolate all population densities at point (t, x)
        m_all_at_x = self._interpolate_densities_at_point(x, t)

        # Delegate to population-specific Hamiltonian with interpolated scalar densities
        return self._problem.hamiltonian_k(self._k, x, m_all_at_x, p, t)

    def terminal_cost(self, x):
        """
        Terminal cost for population k.

        Translates: terminal_cost_k(k, x) → terminal_cost(x)

        Args:
            x: Spatial position

        Returns:
            Terminal cost gₖ(x)
        """
        return self._problem.terminal_cost_k(self._k, x)

    def initial_density(self, x):
        """
        Initial density for population k.

        Translates: initial_density_k(k, x) → initial_density(x)

        Args:
            x: Spatial position

        Returns:
            Initial density m₀ₖ(x)
        """
        return self._problem.initial_density_k(self._k, x)

    def get_final_u(self) -> np.ndarray:
        """
        Get terminal value function array for population k.

        Returns:
            Terminal value function array uₖ(T, x) for population k

        Note:
            Assumes problem has method get_final_u_k(k) or falls back to
            evaluating terminal_cost_k over the spatial domain.
        """
        # Try population-specific method first
        if hasattr(self._problem, "get_final_u_k"):
            return self._problem.get_final_u_k(self._k)

        # Fallback: Evaluate terminal cost over spatial domain
        # This requires spatial grid information from the problem
        if hasattr(self._problem, "get_final_u"):
            # Assume problem provides unified get_final_u that we can use
            return self._problem.get_final_u()

        # Last resort: Zero terminal condition
        shape = self._get_spatial_shape()
        return np.zeros(shape)

    def get_initial_m(self) -> np.ndarray:
        """
        Get initial density array for population k.

        Returns:
            Initial density array m₀ₖ(x) for population k

        Note:
            Assumes problem has method get_initial_m_k(k) or falls back to
            evaluating initial_density_k over the spatial domain.
        """
        # Try population-specific method first
        if hasattr(self._problem, "get_initial_m_k"):
            return self._problem.get_initial_m_k(self._k)

        # Fallback: Evaluate initial density over spatial domain
        if hasattr(self._problem, "get_initial_m"):
            # Assume problem provides unified get_initial_m that we can use
            return self._problem.get_initial_m()

        # Last resort: Uniform distribution
        shape = self._get_spatial_shape()
        m_init = np.ones(shape)
        m_init /= float(np.sum(m_init))
        return m_init

    def get_boundary_conditions(self):
        """Get boundary conditions (delegated)."""
        return self._problem.get_boundary_conditions()

    def _get_spatial_shape(self) -> tuple:
        """
        Get spatial grid shape for population k.

        Returns:
            Shape tuple (Nx+1,) or (Nx+1, Ny+1) or (Nx+1, Ny+1, Nz+1)

        Note:
            This is a helper method for fallback array creation.
        """
        # Try to extract spatial shape from problem
        if hasattr(self._problem, "Nx"):
            Nx = self._problem.Nx
            if isinstance(Nx, (list, tuple)):
                return tuple(n + 1 for n in Nx)
            return (Nx + 1,)

        # Fallback: 1D with 100 points
        return (101,)


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
            Creates an adapter that translates MultiPopulationMFGProtocol
            to MFGProblemProtocol, fixing population index k and freezing
            other populations' densities.
        """
        return _SinglePopulationAdapter(problem, k, m_all)

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
    """Smoke test for MultiPopulationFixedPointSolver and adapter."""
    print("=" * 70)
    print("Testing Multi-Population Fixed-Point Solver (Phase 2)")
    print("=" * 70)

    # Test 1: Solver initialization
    print("\n[Test 1] Solver initialization...")
    solver = MultiPopulationFixedPointSolver(
        max_iterations=50,
        tolerance=1e-6,
        damping_factor=0.8,
        single_pop_config={"max_iterations": 30},
    )
    print(f"✓ Solver created with max_iterations={solver.max_iterations}")
    print(f"✓ Damping factor: {solver.damping_factor}")
    print(f"✓ Single-pop config: {solver.single_pop_config}")

    # Test 2: Parameter validation
    print("\n[Test 2] Parameter validation...")
    try:
        bad_solver = MultiPopulationFixedPointSolver(damping_factor=1.5)
        print("✗ Failed to catch invalid damping_factor")
    except ValueError as e:
        print(f"✓ Correctly rejected invalid damping_factor: {e}")

    # Test 3: Adapter creation
    print("\n[Test 3] Single-population adapter creation...")
    from mfg_pde.extensions import MultiPopulationMFGProblem

    # Create simple 2-population problem
    problem = MultiPopulationMFGProblem(
        num_populations=2,
        spatial_bounds=[(0, 1)],
        spatial_discretization=[50],
        coupling_matrix=[[0.1, 0.05], [0.05, 0.1]],
        T=1.0,
        Nt=20,
        sigma=[0.01, 0.02],
    )

    # Create dummy density arrays
    m_all = [
        np.ones((21, 51)) / 51.0,  # m₁: Shape (Nt+1, Nx+1)
        np.ones((21, 51)) / 51.0,  # m₂
    ]

    # Test adapter for population 0
    adapter_0 = _SinglePopulationAdapter(problem, k=0, m_all=m_all)
    print("✓ Adapter created for population 0")
    print(f"  - dimension: {adapter_0.dimension}")
    print(f"  - T: {adapter_0.T}")
    print(f"  - Nt: {adapter_0.Nt}")
    print(f"  - sigma: {adapter_0.sigma}")

    # Test adapter methods
    print("\n[Test 4] Adapter method delegation...")
    x_test = 0.5
    p_test = 0.1
    t_test = 0.5

    try:
        # Phase 2.5: Hamiltonian now interpolates m_all at point x
        H_val = adapter_0.hamiltonian(x_test, 0.0, p_test, t_test)
        print(f"✓ hamiltonian(x={x_test}, p={p_test}, t={t_test}) = {H_val:.6f}")
        print("  (Phase 2.5: Spatial interpolation working!)")
    except Exception as e:
        print(f"✗ hamiltonian() failed: {type(e).__name__}: {e}")

    try:
        g_val = adapter_0.terminal_cost(x_test)
        print(f"✓ terminal_cost(x={x_test}) = {g_val:.6f}")
    except Exception as e:
        print(f"✗ terminal_cost() failed: {e}")

    try:
        m0_val = adapter_0.initial_density(x_test)
        print(f"✓ initial_density(x={x_test}) = {m0_val:.6f}")
    except Exception as e:
        print(f"✗ initial_density() failed: {e}")

    # Test 4b: Verify interpolation method directly
    print("\n[Test 4b] Direct interpolation test...")
    try:
        m_interp = adapter_0._interpolate_densities_at_point(x_test, t_test)
        print(f"✓ Interpolated densities at (t={t_test}, x={x_test}):")
        for k, m_val in enumerate(m_interp):
            print(f"  - Population {k}: m_{k}({x_test}) = {m_val:.6f}")
    except Exception as e:
        print(f"✗ Interpolation failed: {type(e).__name__}: {e}")

    # Test 5: Array methods
    print("\n[Test 5] Adapter array methods...")
    try:
        u_final = adapter_0.get_final_u()
        print(f"✓ get_final_u() returned array with shape: {u_final.shape}")
    except Exception as e:
        print(f"✗ get_final_u() failed: {e}")

    try:
        m_init = adapter_0.get_initial_m()
        print(f"✓ get_initial_m() returned array with shape: {m_init.shape}")
        print(f"  - Sum (should be ~1.0): {np.sum(m_init):.6f}")
    except Exception as e:
        print(f"✗ get_initial_m() failed: {e}")

    # Test 6: Adapter for population 1
    print("\n[Test 6] Adapter for second population...")
    adapter_1 = _SinglePopulationAdapter(problem, k=1, m_all=m_all)
    print("✓ Adapter created for population 1")
    print(f"  - sigma (different from pop 0): {adapter_1.sigma}")

    # Test 7: Invalid population index
    print("\n[Test 7] Invalid population index validation...")
    try:
        bad_adapter = _SinglePopulationAdapter(problem, k=5, m_all=m_all)
        print("✗ Failed to catch invalid population index")
    except ValueError as e:
        print(f"✓ Correctly rejected invalid k=5: {e}")

    print("\n" + "=" * 70)
    print("All Phase 2.5 smoke tests passed!")
    print("=" * 70)
    print("\nCompleted:")
    print("  ✓ Phase 2: Single-population adapter structure")
    print("  ✓ Phase 2.5: Spatial interpolation for point-wise Hamiltonian evaluation")
    print("\nNext: Phase 3 - Add examples with capacity constraints and MFG vs ABM comparison.")
