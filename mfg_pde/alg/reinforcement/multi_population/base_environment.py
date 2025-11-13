"""
Base environment for multi-population Mean Field Games.

Implements the core interface for heterogeneous multi-population MFG systems
where 2-5 populations with different state/action spaces interact through
coupled mean field distributions.

Author: MFG_PDE Team
Date: October 2025
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from mfg_pde.geometry.grids.grid_1d import SimpleGrid1D
    from mfg_pde.geometry.mesh_2d import Mesh2D

    from .population_config import PopulationConfig


class MultiPopulationMFGEnvironment(ABC):
    """
    Base environment for multi-population Mean Field Games.

    Supports 2-5 heterogeneous populations with:
    - Different state dimensions: S_i ⊆ ℝ^{d_i^s}
    - Different action dimensions: A_i ⊆ ℝ^{d_i^a}
    - Coupled dynamics: f_i(s_i, a_i, m_1, ..., m_N)
    - Coupled rewards: r_i(s_i, a_i, m_1, ..., m_N)

    The environment manages:
    - Individual agent states per population
    - Population distributions m_i(t,x) on spatial domain
    - Coupled state evolution through mean field interactions
    - Population-specific termination conditions
    """

    def __init__(
        self,
        populations: dict[str, PopulationConfig],
        coupling_dynamics: Callable | None = None,
        domain: SimpleGrid1D | Mesh2D | None = None,
        time_horizon: float = 1.0,
        dt: float = 0.01,
    ):
        """
        Initialize multi-population MFG environment.

        Args:
            populations: {pop_id: PopulationConfig} for 2-5 populations
            coupling_dynamics: Function computing coupled state evolution
            domain: Spatial domain for population distributions
            time_horizon: Total time horizon T
            dt: Time step size

        Raises:
            ValueError: If populations configuration is invalid
        """
        from .population_config import validate_population_set

        validate_population_set(populations)

        self.populations = populations
        self.coupling_dynamics = coupling_dynamics
        self.domain = domain
        self.time_horizon = time_horizon
        self.dt = dt

        # State tracking
        self.current_time: float = 0.0
        self.population_states: dict[str, NDArray] = {}
        self.population_distributions: dict[str, NDArray] = {}

        # Episode tracking
        self.episode_step: int = 0
        self.max_steps: int = int(time_horizon / dt)

    @property
    def num_populations(self) -> int:
        """Number of populations."""
        return len(self.populations)

    @property
    def population_ids(self) -> list[str]:
        """List of population IDs."""
        return list(self.populations.keys())

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, NDArray], dict[str, Any]]:
        """
        Reset environment to initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional reset options

        Returns:
            states: {pop_id: initial_state} for each population
            info: {pop_id: metadata} for each population
        """
        if seed is not None:
            np.random.seed(seed)

        self.current_time = 0.0
        self.episode_step = 0

        states = {}
        info = {}

        for pop_id, config in self.populations.items():
            # Sample initial state
            if config.initial_distribution is not None:
                states[pop_id] = config.sample_initial_state()
            else:
                # Default: zero state
                states[pop_id] = np.zeros(config.state_dim)

            info[pop_id] = {
                "population_id": pop_id,
                "population_size": config.population_size,
                "time": self.current_time,
            }

        self.population_states = states
        self._update_distributions()

        return states, info

    def step(
        self, actions: dict[str, NDArray]
    ) -> tuple[
        dict[str, NDArray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, Any],
    ]:
        """
        Execute one time step for all populations.

        Args:
            actions: {pop_id: action_array} for each population

        Returns:
            next_states: {pop_id: next_state}
            rewards: {pop_id: reward_value}
            terminated: {pop_id: episode_done}
            truncated: {pop_id: time_limit_reached}
            info: {pop_id: metadata}

        Raises:
            ValueError: If actions don't match expected populations
        """
        # Validate actions
        if set(actions.keys()) != set(self.populations.keys()):
            msg = f"Actions keys {set(actions.keys())} don't match populations {set(self.populations.keys())}"
            raise ValueError(msg)

        # Compute coupled dynamics
        next_states = self._compute_coupled_dynamics(
            states=self.population_states,
            actions=actions,
            distributions=self.population_distributions,
        )

        # Compute population-specific rewards
        rewards = {}
        for pop_id in self.populations:
            rewards[pop_id] = self._compute_reward(
                pop_id=pop_id,
                state=self.population_states[pop_id],
                action=actions[pop_id],
                next_state=next_states[pop_id],
                distributions=self.population_distributions,
            )

        # Update state
        self.population_states = next_states
        self._update_distributions()

        # Time advancement
        self.current_time += self.dt
        self.episode_step += 1

        # Check termination
        terminated = {pop_id: self._is_terminated(pop_id) for pop_id in self.populations}
        truncated = dict.fromkeys(self.populations, self.episode_step >= self.max_steps)

        # Info
        info = {
            pop_id: {
                "time": self.current_time,
                "episode_step": self.episode_step,
                "distribution_mass": self.population_distributions[pop_id].sum()
                if self.population_distributions[pop_id] is not None
                else 0.0,
            }
            for pop_id in self.populations
        }

        return next_states, rewards, terminated, truncated, info

    def get_population_state(self, pop_id: str) -> NDArray:
        """
        Get distribution for specific population.

        Args:
            pop_id: Population identifier

        Returns:
            Population distribution array

        Raises:
            KeyError: If pop_id is invalid
        """
        if pop_id not in self.populations:
            msg = f"Unknown population ID: '{pop_id}'"
            raise KeyError(msg)

        return self.population_distributions[pop_id]

    def get_all_population_states(self) -> dict[str, NDArray]:
        """
        Get all population distributions.

        Returns:
            {pop_id: distribution_array} for all populations
        """
        return self.population_distributions.copy()

    def _compute_coupled_dynamics(
        self,
        states: dict[str, NDArray],
        actions: dict[str, NDArray],
        distributions: dict[str, NDArray],
    ) -> dict[str, NDArray]:
        """
        Compute coupled state evolution for all populations.

        Args:
            states: Current states {pop_id: state}
            actions: Current actions {pop_id: action}
            distributions: Current distributions {pop_id: m_i}

        Returns:
            Next states {pop_id: next_state}
        """
        if self.coupling_dynamics is not None:
            return self.coupling_dynamics(states, actions, distributions)
        else:
            # Default: use population-specific dynamics
            next_states = {}
            for pop_id in self.populations:
                next_states[pop_id] = self._compute_single_dynamics(
                    pop_id=pop_id,
                    state=states[pop_id],
                    action=actions[pop_id],
                    distributions=distributions,
                )
            return next_states

    @abstractmethod
    def _compute_single_dynamics(
        self,
        pop_id: str,
        state: NDArray,
        action: NDArray,
        distributions: dict[str, NDArray],
    ) -> NDArray:
        """
        Compute next state for single population.

        Must be implemented by concrete environment.

        Args:
            pop_id: Population identifier
            state: Current state
            action: Current action
            distributions: All population distributions

        Returns:
            Next state
        """

    @abstractmethod
    def _compute_reward(
        self,
        pop_id: str,
        state: NDArray,
        action: NDArray,
        next_state: NDArray,
        distributions: dict[str, NDArray],
    ) -> float:
        """
        Compute reward for specific population.

        Must be implemented by concrete environment.

        Args:
            pop_id: Population identifier
            state: Current state
            action: Current action
            next_state: Next state
            distributions: All population distributions

        Returns:
            Reward value
        """

    def _update_distributions(self) -> None:
        """
        Update population distributions from current states.

        Converts individual states to spatial distributions on domain.
        """
        for pop_id, state in self.population_states.items():
            self.population_distributions[pop_id] = self._state_to_distribution(pop_id=pop_id, state=state)

    @abstractmethod
    def _state_to_distribution(self, pop_id: str, state: NDArray) -> NDArray:
        """
        Convert individual state to population distribution.

        Must be implemented by concrete environment.

        Args:
            pop_id: Population identifier
            state: Individual state

        Returns:
            Distribution array (histogram on domain)
        """

    def _is_terminated(self, pop_id: str) -> bool:
        """
        Check if population has reached terminal state.

        Can be overridden by concrete environment.

        Args:
            pop_id: Population identifier

        Returns:
            True if terminated
        """
        return False  # Default: no early termination

    def render(self, mode: str = "human") -> None:  # noqa: B027
        """
        Render current environment state.

        Can be overridden by concrete environment.

        Args:
            mode: Rendering mode
        """

    def close(self) -> None:  # noqa: B027
        """Clean up environment resources."""


class SimpleMultiPopulationEnv(MultiPopulationMFGEnvironment):
    """
    Simple multi-population environment for testing.

    Linear dynamics: s' = s + a * dt
    Quadratic cost: r = -||a||² - coupling_cost
    """

    def _compute_single_dynamics(
        self,
        pop_id: str,
        state: NDArray,
        action: NDArray,
        distributions: dict[str, NDArray],
    ) -> NDArray:
        """Simple linear dynamics."""
        return state + action * self.dt

    def _compute_reward(
        self,
        pop_id: str,
        state: NDArray,
        action: NDArray,
        next_state: NDArray,
        distributions: dict[str, NDArray],
    ) -> float:
        """Quadratic action cost with coupling."""
        action_cost = -np.sum(action**2)

        # Coupling cost based on other populations
        config = self.populations[pop_id]
        coupling_cost = 0.0

        for other_id, weight in config.coupling_weights.items():
            if other_id in distributions:
                # Simple L2 distance between states
                other_state = self.population_states[other_id]
                coupling_cost += weight * np.sum((state - other_state) ** 2)

        return float(action_cost - coupling_cost)

    def _state_to_distribution(self, pop_id: str, state: NDArray) -> NDArray:
        """Convert state to delta distribution."""
        # For testing: just return the state itself
        return state
