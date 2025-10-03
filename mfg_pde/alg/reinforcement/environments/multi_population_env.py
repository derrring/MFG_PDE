"""
Multi-Population Mean Field Game Environment Base Class.

Provides foundation for MFG environments with multiple interacting populations,
where different populations may have:
- Heterogeneous action spaces
- Different objectives and reward structures
- Distinct dynamics
- Cross-population interactions

Mathematical Framework:
- N populations with policies π₁, π₂, ..., πₙ
- Population densities m₁, m₂, ..., mₙ
- Coupled HJB equations: ∂uᵢ/∂t + Hᵢ(x, ∇uᵢ, m₁, ..., mₙ) = 0
- Coupled FP equations: ∂mᵢ/∂t - div(mᵢ ∇ₚHᵢ) - Δmᵢ = 0

Author: MFG_PDE Team
Date: October 2025
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class MultiPopulationMFGEnv(ABC):
    """
    Abstract base class for multi-population Mean Field Game environments.

    Key Features:
    - Multiple interacting populations (N ≥ 2)
    - Heterogeneous action spaces per population
    - Population-specific reward functions
    - Cross-population coupling through mean fields
    - Nash equilibrium as solution concept

    State Space:
    - Individual state: sᵢ ∈ Sᵢ for population i
    - Population states: (m₁, m₂, ..., mₙ) where mᵢ ∈ P(Sᵢ)

    Action Space:
    - Population-specific: aᵢ ∈ Aᵢ (can be heterogeneous)

    Dynamics:
    - State transition: s'ᵢ = f_i(sᵢ, aᵢ, m₁, ..., mₙ) + noise
    - Population evolution: mᵢ_{t+1} = T_i(mᵢ_t, πᵢ, m_{-i})
    """

    def __init__(
        self,
        num_populations: int,
        state_dims: list[int] | int,
        action_specs: list[dict[str, Any]],
        population_sizes: list[int] | int,
        dt: float = 0.1,
        max_steps: int = 100,
    ):
        """
        Initialize multi-population MFG environment.

        Args:
            num_populations: Number of interacting populations (N ≥ 2)
            state_dims: State dimension for each population (list) or shared (int)
            action_specs: Action space specification for each population
                Each dict contains:
                - 'type': 'discrete' or 'continuous'
                - 'dim': action dimension
                - 'bounds': (min, max) for continuous actions
            population_sizes: Number of agents per population (for discretization)
            dt: Time step size
            max_steps: Maximum episode length
        """
        if num_populations < 2:
            raise ValueError(f"Multi-population requires N ≥ 2, got {num_populations}")

        self.num_populations = num_populations

        # Handle state dimensions
        if isinstance(state_dims, int):
            self.state_dims = [state_dims] * num_populations
        else:
            if len(state_dims) != num_populations:
                raise ValueError(f"state_dims length {len(state_dims)} != num_populations {num_populations}")
            self.state_dims = state_dims

        # Validate action specs
        if len(action_specs) != num_populations:
            raise ValueError(f"action_specs length {len(action_specs)} != num_populations {num_populations}")
        self.action_specs = action_specs

        # Handle population sizes
        if isinstance(population_sizes, int):
            self.population_sizes = [population_sizes] * num_populations
        else:
            if len(population_sizes) != num_populations:
                raise ValueError(
                    f"population_sizes length {len(population_sizes)} != num_populations {num_populations}"
                )
            self.population_sizes = population_sizes

        self.dt = dt
        self.max_steps = max_steps

        # Episode tracking
        self.current_step = 0
        self.current_states: dict[int, NDArray[np.floating[Any]]] = {}
        self.population_distributions: dict[int, NDArray[np.floating[Any]]] = {}

        # Initialize populations
        self._initialize_populations()

    def _initialize_populations(self) -> None:
        """Initialize population states and distributions."""
        for pop_id in range(self.num_populations):
            # Initialize individual states for each population
            self.current_states[pop_id] = self._sample_initial_state(pop_id)

            # Initialize population distributions
            self.population_distributions[pop_id] = self._compute_population_distribution(pop_id)

    @abstractmethod
    def _sample_initial_state(self, pop_id: int) -> NDArray[np.floating[Any]]:
        """
        Sample initial state for population pop_id.

        Args:
            pop_id: Population identifier (0 to N-1)

        Returns:
            Initial state sᵢ ∈ ℝ^{state_dims[pop_id]}
        """

    @abstractmethod
    def _compute_population_distribution(self, pop_id: int) -> NDArray[np.floating[Any]]:
        """
        Compute population distribution for population pop_id.

        Args:
            pop_id: Population identifier

        Returns:
            Population distribution mᵢ ∈ ℝ^{population_sizes[pop_id]}
            Should be normalized probability distribution
        """

    @abstractmethod
    def _dynamics(
        self,
        pop_id: int,
        state: NDArray[np.floating[Any]],
        action: NDArray[np.floating[Any]] | int,
        population_states: dict[int, NDArray[np.floating[Any]]],
    ) -> NDArray[np.floating[Any]]:
        """
        State transition dynamics for population pop_id.

        Args:
            pop_id: Population identifier
            state: Current state sᵢ
            action: Chosen action aᵢ
            population_states: All population distributions {0: m₀, 1: m₁, ...}

        Returns:
            Next state s'ᵢ = f_i(sᵢ, aᵢ, m₁, ..., mₙ) + noise
        """

    @abstractmethod
    def _reward(
        self,
        pop_id: int,
        state: NDArray[np.floating[Any]],
        action: NDArray[np.floating[Any]] | int,
        next_state: NDArray[np.floating[Any]],
        population_states: dict[int, NDArray[np.floating[Any]]],
    ) -> float:
        """
        Reward function for population pop_id.

        Args:
            pop_id: Population identifier
            state: Current state sᵢ
            action: Chosen action aᵢ
            next_state: Next state s'ᵢ
            population_states: All population distributions

        Returns:
            Immediate reward rᵢ(sᵢ, aᵢ, m₁, ..., mₙ)
        """

    def step(
        self, actions: dict[int, NDArray[np.floating[Any]] | int]
    ) -> tuple[
        dict[int, NDArray[np.floating[Any]]],
        dict[int, float],
        dict[int, bool],
        dict[int, bool],
        dict[int, dict[str, Any]],
    ]:
        """
        Execute one time step with actions from all populations.

        Args:
            actions: Dictionary mapping pop_id → action for each population

        Returns:
            Tuple of (next_states, rewards, terminated, truncated, info) where:
            - next_states: {pop_id: s'ᵢ} next state for each population
            - rewards: {pop_id: rᵢ} reward for each population
            - terminated: {pop_id: bool} episode termination per population
            - truncated: {pop_id: bool} time limit truncation per population
            - info: {pop_id: dict} additional information per population
        """
        if len(actions) != self.num_populations:
            raise ValueError(f"Expected actions for {self.num_populations} populations, " f"got {len(actions)}")

        next_states: dict[int, NDArray[np.floating[Any]]] = {}
        rewards: dict[int, float] = {}
        terminated: dict[int, bool] = {}
        truncated: dict[int, bool] = {}
        info: dict[int, dict[str, Any]] = {}

        # Update all populations
        for pop_id in range(self.num_populations):
            if pop_id not in actions:
                raise ValueError(f"Missing action for population {pop_id}")

            # Apply dynamics
            current_state = self.current_states[pop_id]
            action = actions[pop_id]
            next_state = self._dynamics(pop_id, current_state, action, self.population_distributions)

            # Compute reward
            reward = self._reward(pop_id, current_state, action, next_state, self.population_distributions)

            # Check termination
            is_terminated = self._is_terminated(pop_id, next_state)
            is_truncated = self.current_step >= self.max_steps - 1

            # Store results
            next_states[pop_id] = next_state
            rewards[pop_id] = reward
            terminated[pop_id] = is_terminated
            truncated[pop_id] = is_truncated
            info[pop_id] = self._get_info(pop_id)

        # Update current states
        self.current_states = next_states

        # Update population distributions
        for pop_id in range(self.num_populations):
            self.population_distributions[pop_id] = self._compute_population_distribution(pop_id)

        self.current_step += 1

        return next_states, rewards, terminated, truncated, info

    def _is_terminated(self, pop_id: int, state: NDArray[np.floating[Any]]) -> bool:
        """
        Check if population pop_id has reached terminal state.

        Args:
            pop_id: Population identifier
            state: Current state

        Returns:
            True if episode should terminate for this population
        """
        # Default: no early termination
        return False

    def _get_info(self, pop_id: int) -> dict[str, Any]:
        """
        Get additional information for population pop_id.

        Args:
            pop_id: Population identifier

        Returns:
            Dictionary with additional information
        """
        return {
            "step": self.current_step,
            "population_id": pop_id,
            "population_mass": np.sum(self.population_distributions[pop_id]),
        }

    def reset(self, seed: int | None = None) -> tuple[dict[int, NDArray[np.floating[Any]]], dict[int, dict[str, Any]]]:
        """
        Reset environment to initial state.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Tuple of (initial_states, info) where:
            - initial_states: {pop_id: sᵢ} initial state for each population
            - info: {pop_id: dict} additional information per population
        """
        if seed is not None:
            np.random.seed(seed)

        self.current_step = 0
        self._initialize_populations()

        info = {pop_id: self._get_info(pop_id) for pop_id in range(self.num_populations)}

        return self.current_states.copy(), info

    def get_population_states(self) -> dict[int, NDArray[np.floating[Any]]]:
        """
        Get current population distributions for all populations.

        Returns:
            Dictionary {pop_id: mᵢ} of population distributions
        """
        return self.population_distributions.copy()

    def get_action_dim(self, pop_id: int) -> int:
        """Get action dimension for population pop_id."""
        return self.action_specs[pop_id]["dim"]

    def get_state_dim(self, pop_id: int) -> int:
        """Get state dimension for population pop_id."""
        return self.state_dims[pop_id]

    def get_action_bounds(self, pop_id: int) -> tuple[float, float] | None:
        """
        Get action bounds for population pop_id.

        Returns:
            (min, max) tuple for continuous actions, None for discrete
        """
        spec = self.action_specs[pop_id]
        if spec["type"] == "continuous":
            return spec.get("bounds", (-1.0, 1.0))
        return None

    def is_continuous_action(self, pop_id: int) -> bool:
        """Check if population pop_id has continuous action space."""
        return self.action_specs[pop_id]["type"] == "continuous"
