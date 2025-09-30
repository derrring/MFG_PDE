"""
Wasserstein Distance-Based MFG Solver.

This module implements Mean Field Game solvers using optimal transport theory
and Wasserstein distance formulations. The approach reformulates MFG problems
in the space of probability measures with Wasserstein geometry.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from mfg_pde.alg.base_solver import BaseOptimizationSolver

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mfg_pde.core.mfg_problem import MFGProblem

logger = logging.getLogger(__name__)


@dataclass
class WassersteinSolverConfig:
    """Configuration for Wasserstein MFG solver."""

    # Optimization parameters
    max_iterations: int = 1000
    tolerance: float = 1e-6
    step_size: float = 0.01

    # Wasserstein parameters
    regularization: float = 0.1  # Entropic regularization parameter
    ground_metric: str = "euclidean"  # Ground metric for optimal transport
    sinkhorn_iterations: int = 100  # Sinkhorn algorithm iterations

    # Discretization
    num_time_steps: int = 50
    num_spatial_points: int = 100

    # Convergence criteria
    relative_tolerance: bool = True
    history_length: int = 10


@dataclass
class WassersteinSolverResult:
    """Result container for Wasserstein MFG solver."""

    # Optimal solution
    optimal_densities: NDArray | None = None  # m(t,x) over time
    optimal_velocities: NDArray | None = None  # v(t,x) velocity field
    transport_plans: list[NDArray] | None = None  # Optimal transport plans

    # Cost and convergence
    final_cost: float = np.inf
    wasserstein_distances: list[float] | None = None  # W2 distances over time
    cost_history: list[float] | None = None
    converged: bool = False
    num_iterations: int = 0

    # Transport information
    total_transport_cost: float = np.inf
    average_displacement: float = 0.0


class WassersteinMFGSolver(BaseOptimizationSolver):
    """
    Wasserstein distance-based solver for Mean Field Games.

    This solver reformulates MFG problems using optimal transport theory,
    working directly in the Wasserstein space of probability measures.

    Mathematical Framework:
    - State space: W2(Ω) (Wasserstein space of probability measures)
    - Dynamics: Gradient flow in Wasserstein space
    - Cost: Integrated transport cost + running cost functional
    - Equilibrium: Minimizes total cost in Wasserstein metric

    Key Features:
    - Geometric approach via optimal transport
    - Natural handling of probability constraints
    - Displacement interpolation for dynamics
    - Wasserstein distance-based convergence criteria
    """

    def __init__(
        self,
        problem: MFGProblem,
        config: WassersteinSolverConfig | None = None,
        **kwargs: Any,
    ):
        """
        Initialize Wasserstein MFG solver.

        Args:
            problem: MFG problem instance
            config: Solver configuration
            **kwargs: Additional solver arguments
        """
        super().__init__(problem, **kwargs)
        self.config = config or WassersteinSolverConfig()
        self.logger = self._get_logger()

        # Validate dependencies
        self._check_dependencies()

        # Initialize discretization
        self._setup_discretization()

        # Initialize transport infrastructure
        self._setup_optimal_transport()

    def _check_dependencies(self) -> None:
        """Check that required dependencies are available."""
        try:
            import ot

            import scipy.optimize

            self.ot = ot
            self.scipy_opt = scipy.optimize
        except ImportError as e:
            raise ImportError(
                f"Wasserstein solver requires POT and scipy: {e}. Install with: pip install mfg_pde[optimization]"
            ) from e

    def _setup_discretization(self) -> None:
        """Setup temporal and spatial discretization."""
        self.time_grid = np.linspace(0, self.problem.T, self.config.num_time_steps)
        self.dt = self.time_grid[1] - self.time_grid[0]

        # Spatial grid (assuming 1D for simplicity, extend to multi-D)
        self.spatial_grid = np.linspace(self.problem.domain[0], self.problem.domain[1], self.config.num_spatial_points)
        self.dx = self.spatial_grid[1] - self.spatial_grid[0]

        self.logger.info(f"Discretization: {len(self.time_grid)} time steps, {len(self.spatial_grid)} spatial points")

    def _setup_optimal_transport(self) -> None:
        """Setup optimal transport infrastructure."""
        # Ground cost matrix (Euclidean distance)
        if self.config.ground_metric == "euclidean":
            x_grid, y_grid = np.meshgrid(self.spatial_grid, self.spatial_grid)
            self.ground_cost = (x_grid - y_grid) ** 2
        else:
            raise NotImplementedError(f"Ground metric {self.config.ground_metric} not implemented")

        self.logger.info(f"Ground cost matrix shape: {self.ground_cost.shape}")

    def solve(self) -> WassersteinSolverResult:
        """
        Solve MFG problem using Wasserstein optimal transport approach.

        Returns:
            WassersteinSolverResult: Solution with optimal densities and transport
        """
        self.logger.info("Starting Wasserstein MFG solver")

        # Initialize result container
        result = WassersteinSolverResult()

        try:
            # Initialize density evolution
            densities = self._initialize_densities()

            # Main optimization loop
            cost_history = []
            wasserstein_distances = []

            for iteration in range(self.config.max_iterations):
                self.logger.debug(f"Iteration {iteration + 1}/{self.config.max_iterations}")

                # Compute transport plans between consecutive time steps
                transport_plans = self._compute_transport_sequence(densities)

                # Update densities using transport-based dynamics
                new_densities = self._update_densities_via_transport(densities, transport_plans)

                # Compute total cost
                current_cost = self._compute_total_cost(new_densities, transport_plans)
                cost_history.append(current_cost)

                # Compute Wasserstein distances for convergence
                w2_distance = self._compute_wasserstein_distance(densities, new_densities)
                wasserstein_distances.append(w2_distance)

                # Check convergence
                if self._check_convergence(cost_history, wasserstein_distances):
                    self.logger.info(f"Converged after {iteration + 1} iterations")
                    result.converged = True
                    break

                densities = new_densities

                # Progress logging
                if (iteration + 1) % 50 == 0:
                    self.logger.info(f"  Iteration {iteration + 1}: Cost={current_cost:.6e}, W2={w2_distance:.6e}")

            else:
                self.logger.warning(f"Did not converge after {self.config.max_iterations} iterations")

            # Finalize results
            result.optimal_densities = densities
            result.transport_plans = transport_plans
            result.final_cost = cost_history[-1] if cost_history else np.inf
            result.cost_history = cost_history
            result.wasserstein_distances = wasserstein_distances
            result.num_iterations = len(cost_history)
            result.total_transport_cost = sum(self._compute_transport_cost(plan) for plan in transport_plans)

            # Compute optimal velocities from transport plans
            result.optimal_velocities = self._compute_velocities_from_transport(transport_plans)

            self.logger.info(f"Wasserstein solver completed: Cost={result.final_cost:.6e}")

        except Exception as e:
            self.logger.error(f"Wasserstein solver failed: {e}")
            raise

        return result

    def _initialize_densities(self) -> NDArray:
        """Initialize density evolution with boundary conditions."""
        densities = np.zeros((len(self.time_grid), len(self.spatial_grid)))

        # Initial density (t=0)
        if hasattr(self.problem, "initial_density") and self.problem.initial_density is not None:
            # Use problem-specific initial density
            for i, x in enumerate(self.spatial_grid):
                densities[0, i] = self.problem.initial_density(x)
        else:
            # Default: Gaussian initial density
            center = (self.problem.domain[0] + self.problem.domain[1]) / 2
            width = (self.problem.domain[1] - self.problem.domain[0]) / 6
            densities[0, :] = np.exp(-((self.spatial_grid - center) ** 2) / (2 * width**2))

        # Normalize initial density
        densities[0, :] /= np.trapz(densities[0, :], self.spatial_grid)

        return densities

    def _compute_transport_sequence(self, densities: NDArray) -> list[NDArray]:
        """Compute optimal transport plans between consecutive time steps."""
        transport_plans = []

        for t in range(len(self.time_grid) - 1):
            source = densities[t, :]
            target = densities[t + 1, :] if t + 1 < len(densities) else densities[t, :]

            # Ensure positive masses
            source = np.maximum(source, 1e-12)
            target = np.maximum(target, 1e-12)

            # Normalize
            source /= np.sum(source)
            target /= np.sum(target)

            # Compute optimal transport plan using Sinkhorn
            transport_plan = self.ot.sinkhorn(
                source,
                target,
                self.ground_cost,
                reg=self.config.regularization,
                numItermax=self.config.sinkhorn_iterations,
            )

            transport_plans.append(transport_plan)

        return transport_plans

    def _update_densities_via_transport(self, densities: NDArray, transport_plans: list[NDArray]) -> NDArray:
        """Update density evolution using optimal transport dynamics."""
        new_densities = densities.copy()

        for t in range(1, len(self.time_grid)):
            if t - 1 < len(transport_plans):
                # Update density using transport plan
                transport_plan = transport_plans[t - 1]
                new_densities[t, :] = np.sum(transport_plan, axis=0)

        return new_densities

    def _compute_total_cost(self, densities: NDArray, transport_plans: list[NDArray]) -> float:
        """Compute total MFG cost including transport and running costs."""
        total_cost = 0.0

        # Transport cost
        for plan in transport_plans:
            total_cost += np.sum(plan * self.ground_cost) * self.dt

        # Running cost (simplified)
        if hasattr(self.problem, "running_cost"):
            for t, density in enumerate(densities):
                time = self.time_grid[t] if t < len(self.time_grid) else self.time_grid[-1]
                for i, x in enumerate(self.spatial_grid):
                    total_cost += self.problem.running_cost(time, x, density[i]) * self.dx * self.dt

        return total_cost

    def _compute_wasserstein_distance(self, densities1: NDArray, densities2: NDArray) -> float:
        """Compute Wasserstein-2 distance between density evolutions."""
        total_distance = 0.0

        for t in range(min(len(densities1), len(densities2))):
            d1 = np.maximum(densities1[t, :], 1e-12)
            d2 = np.maximum(densities2[t, :], 1e-12)

            # Normalize
            d1 /= np.sum(d1)
            d2 /= np.sum(d2)

            # Compute W2 distance
            w2_dist = self.ot.wasserstein_1d(self.spatial_grid, self.spatial_grid, d1, d2, p=2)
            total_distance += w2_dist**2

        return np.sqrt(total_distance)

    def _check_convergence(self, cost_history: list[float], w2_distances: list[float]) -> bool:
        """Check convergence based on cost and Wasserstein distance."""
        if len(cost_history) < 2:
            return False

        # Cost-based convergence
        cost_change = abs(cost_history[-1] - cost_history[-2])
        cost_converged = cost_change < self.config.tolerance

        # Wasserstein distance convergence
        w2_converged = w2_distances[-1] < self.config.tolerance if w2_distances else False

        return cost_converged and w2_converged

    def _compute_transport_cost(self, transport_plan: NDArray) -> float:
        """Compute cost of a single transport plan."""
        return np.sum(transport_plan * self.ground_cost)

    def _compute_velocities_from_transport(self, transport_plans: list[NDArray]) -> NDArray:
        """Compute velocity field from optimal transport plans."""
        velocities = np.zeros((len(self.time_grid), len(self.spatial_grid)))

        for t, plan in enumerate(transport_plans):
            if t + 1 < len(velocities):
                # Compute displacement field from transport plan
                for i in range(len(self.spatial_grid)):
                    displacement = 0.0
                    total_mass = np.sum(plan[i, :])
                    if total_mass > 1e-12:
                        for j in range(len(self.spatial_grid)):
                            displacement += plan[i, j] * (self.spatial_grid[j] - self.spatial_grid[i])
                        velocities[t + 1, i] = displacement / (total_mass * self.dt)

        return velocities
