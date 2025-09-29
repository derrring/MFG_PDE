"""
Base solver classes for the new algorithm structure.

This module defines the foundational classes that all algorithm paradigms inherit from,
ensuring consistent interfaces while allowing paradigm-specific customization.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mfg_pde.config import BaseConfig
    from mfg_pde.core import MFGProblem


class BaseMFGSolver(ABC):
    """
    Abstract base class for all MFG solvers across paradigms.

    This class defines the common interface that all solvers must implement,
    regardless of their mathematical approach (numerical, neural, RL, optimization).
    """

    def __init__(self, problem: MFGProblem, config: BaseConfig) -> None:
        """
        Initialize the solver with a problem and configuration.

        Args:
            problem: The MFG problem to solve
            config: Solver-specific configuration
        """
        self.problem = problem
        self.config = config
        self._is_solved = False
        self._solution: Any | None = None

    @abstractmethod
    def solve(self) -> Any:
        """
        Solve the MFG problem.

        Returns:
            Solution object containing u(t,x), m(t,x) and metadata
        """

    @abstractmethod
    def validate_solution(self) -> dict[str, float]:
        """
        Validate the computed solution.

        Returns:
            Dictionary of validation metrics (Nash gap, mass conservation, etc.)
        """

    @property
    def is_solved(self) -> bool:
        """Check if the solver has computed a solution."""
        return self._is_solved

    @property
    def solution(self) -> Any:
        """Get the computed solution."""
        if not self._is_solved:
            raise RuntimeError("Solver has not been run. Call solve() first.")
        return self._solution


class BaseNumericalSolver(BaseMFGSolver):
    """Base class for numerical methods (FDM, FEM, spectral, etc.)."""

    def __init__(self, problem: MFGProblem, config: BaseConfig) -> None:
        super().__init__(problem, config)
        self.convergence_history: list[float] = []

    @abstractmethod
    def discretize(self) -> None:
        """Set up the spatial and temporal discretization."""

    def get_convergence_info(self) -> dict[str, Any]:
        """Get convergence information."""
        return {
            "iterations": len(self.convergence_history),
            "final_error": self.convergence_history[-1] if self.convergence_history else None,
            "convergence_rate": self._estimate_convergence_rate(),
        }

    def _estimate_convergence_rate(self) -> float | None:
        """Estimate convergence rate from history."""
        if len(self.convergence_history) < 3:
            return None

        errors = self.convergence_history[-3:]
        if errors[-1] == 0 or errors[-2] == 0:
            return None

        return abs(errors[-1] / errors[-2])


class BaseOptimizationSolver(BaseMFGSolver):
    """Base class for optimization methods (variational, optimal transport, etc.)."""

    def __init__(self, problem: MFGProblem, config: BaseConfig) -> None:
        super().__init__(problem, config)
        self.objective_history: list[float] = []

    @abstractmethod
    def compute_objective(self, variables: Any) -> float:
        """Compute the objective function value."""

    @abstractmethod
    def compute_gradient(self, variables: Any) -> Any:
        """Compute the gradient of the objective function."""


class BaseNeuralSolver(BaseMFGSolver):
    """Base class for neural network methods (PINN, neural operators, etc.)."""

    def __init__(self, problem: MFGProblem, config: BaseConfig) -> None:
        super().__init__(problem, config)
        self.training_history: dict[str, list[float]] = {}

    @abstractmethod
    def build_networks(self) -> None:
        """Build the neural network architectures."""

    @abstractmethod
    def compute_loss(self, *args, **kwargs) -> dict[str, float]:
        """Compute the total loss and its components."""

    @abstractmethod
    def train_step(self) -> dict[str, float]:
        """Perform one training step."""


class BaseRLSolver(BaseMFGSolver):
    """Base class for reinforcement learning methods."""

    def __init__(self, problem: MFGProblem, config: BaseConfig) -> None:
        super().__init__(problem, config)
        self.training_metrics: dict[str, list[float]] = {}
        self.population_size: int | None = getattr(config, "population_size", None)

    @abstractmethod
    def create_environment(self) -> Any:
        """Create the MFG environment for RL agents."""

    @abstractmethod
    def create_agents(self) -> Any:
        """Create the RL agents."""

    @abstractmethod
    def train_agents(self) -> dict[str, float]:
        """Train the RL agents to reach Nash equilibrium."""

    def evaluate_nash_gap(self) -> float:
        """Evaluate the Nash equilibrium gap."""
        # Default implementation - should be overridden
        return 0.0

    def scale_to_mean_field(self) -> Any:
        """Convert finite-population solution to mean field limit."""
        if self.population_size is None or self.population_size == float("inf"):
            return self.solution
        # Default implementation - should be overridden by specific solvers
        return self.solution
