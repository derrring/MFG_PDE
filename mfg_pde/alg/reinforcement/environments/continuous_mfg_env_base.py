"""
Base class for continuous action Mean Field Game environments.

Provides Gymnasium-compatible API for single-population MFG environments with:
- Continuous action spaces
- Population distribution tracking via histograms
- Mean field coupling in rewards
- Standardized observation/action spaces

Mathematical Framework:
- State space: $x \\in \\mathcal{X} \\subset \\mathbb{R}^d$
- Action space: $u \\in \\mathcal{U} \\subset \\mathbb{R}^m$
- Population distribution: $m(x) \\in \\mathcal{P}(\\mathcal{X})$
- Dynamics: $dx = f(x, u, m) dt + \\sigma dW$
- Reward: $r(x, u, m) = r_0(x, u) + r_{\text{MF}}(x, m)$

Author: MFG_PDE Team
Date: October 2025
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

try:
    import gymnasium as gym
    from gymnasium import spaces

    GYMNASIUM_AVAILABLE = True
except ImportError:
    GYMNASIUM_AVAILABLE = False
    # Fallback for when gymnasium not available
    gym = None  # type: ignore
    spaces = None  # type: ignore


class ContinuousMFGEnvBase(ABC):
    """
    Base class for continuous action MFG environments.

    Gymnasium-compatible interface for single-population Mean Field Games with:
    - Continuous state and action spaces
    - Population distribution tracking
    - Mean field coupling in dynamics and rewards
    - Stochastic dynamics with Brownian noise

    Key Design:
    - Each agent observes: (individual_state, population_histogram)
    - Agent acts in continuous action space
    - Reward depends on state, action, and population distribution
    - Population evolves according to all agents' policies

    Mathematical Components:
    1. Individual Dynamics:
       $x_{t+1} = x_t + f(x_t, u_t, m_t) \\cdot dt + \\sigma \\sqrt{dt} \\cdot \\epsilon$

    2. Population Evolution:
       $m_{t+1}(x) = $ empirical distribution of all agents at time $t+1$

    3. Reward Structure:
       $r_t = r_0(x_t, u_t) + r_{\text{MF}}(x_t, m_t)$
    """

    def __init__(
        self,
        num_agents: int,
        state_dim: int,
        action_dim: int,
        action_bounds: tuple[float, float] = (-1.0, 1.0),
        population_bins: int = 100,
        dt: float = 0.01,
        max_steps: int = 200,
        noise_std: float = 0.1,
    ):
        """
        Initialize continuous MFG environment.

        Args:
            num_agents: Number of agents in population (for discretization)
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            action_bounds: (min, max) bounds for actions
            population_bins: Number of bins for population histogram
            dt: Time step size
            max_steps: Maximum episode length
            noise_std: Standard deviation of Brownian noise
        """
        if not GYMNASIUM_AVAILABLE:
            raise ImportError("gymnasium required for continuous MFG environments. Install with: pip install gymnasium")

        if num_agents < 1:
            raise ValueError(f"num_agents must be >= 1, got {num_agents}")
        if state_dim < 1:
            raise ValueError(f"state_dim must be >= 1, got {state_dim}")
        if action_dim < 1:
            raise ValueError(f"action_dim must be >= 1, got {action_dim}")
        if population_bins < 1:
            raise ValueError(f"population_bins must be >= 1, got {population_bins}")

        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bounds = action_bounds
        self.population_bins = population_bins
        self.dt = dt
        self.max_steps = max_steps
        self.noise_std = noise_std

        # Gymnasium spaces
        state_low, state_high = self._get_state_bounds()
        self.observation_space = spaces.Box(low=state_low, high=state_high, shape=(state_dim,), dtype=np.float32)

        action_low = np.full(action_dim, action_bounds[0], dtype=np.float32)
        action_high = np.full(action_dim, action_bounds[1], dtype=np.float32)
        self.action_space = spaces.Box(low=action_low, high=action_high, shape=(action_dim,), dtype=np.float32)

        # Episode state
        self.current_step = 0
        self.agent_states: NDArray[np.floating[Any]] | None = None
        self.population_histogram: NDArray[np.floating[Any]] | None = None

        # Random number generator
        self.rng = np.random.default_rng()

    @abstractmethod
    def _get_state_bounds(self) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
        r"""
        Get bounds for state space.

        Returns:
            Tuple (low, high) where each is array of shape (state_dim,)
            Defines the state space bounds: $x \in [low, high]$
        """

    @abstractmethod
    def _sample_initial_states(self) -> NDArray[np.floating[Any]]:
        """
        Sample initial states for all agents.

        Returns:
            Array of shape (num_agents, state_dim) with initial states
            Should implement problem-specific initialization distribution
        """

    @abstractmethod
    def _drift(
        self, state: NDArray[np.floating[Any]], action: NDArray[np.floating[Any]], population: NDArray[np.floating[Any]]
    ) -> NDArray[np.floating[Any]]:
        r"""
        Compute drift term in dynamics: $f(x, u, m)$

        Args:
            state: Current state $x \in \mathbb{R}^{state\_dim}$
            action: Current action $u \in \mathbb{R}^{action\_dim}$
            population: Population histogram $m \in \mathbb{R}^{population\_bins}$

        Returns:
            Drift vector $f(x, u, m) \in \mathbb{R}^{state\_dim}$
        """

    @abstractmethod
    def compute_mean_field_coupling(
        self, state: NDArray[np.floating[Any]], population: NDArray[np.floating[Any]]
    ) -> float:
        """
        Compute mean field interaction term for reward.

        This captures how the population distribution affects an agent's reward.
        Common examples:
        - Congestion: penalty proportional to local density
        - Coordination: reward for matching population mean
        - Repulsion: penalty for proximity to other agents

        Args:
            state: Current state $x$
            population: Population histogram $m$

        Returns:
            Mean field coupling term $r_{\text{MF}}(x, m)$
        """

    def _individual_reward(
        self, state: NDArray[np.floating[Any]], action: NDArray[np.floating[Any]], next_state: NDArray[np.floating[Any]]
    ) -> float:
        """
        Compute individual reward term: $r_0(x, u)$

        Default implementation: zero individual reward (pure MF coupling).
        Override for problem-specific individual costs.

        Args:
            state: Current state $x$
            action: Current action $u$
            next_state: Next state $x'$

        Returns:
            Individual reward $r_0(x, u)$
        """
        return 0.0

    def _compute_reward(
        self,
        state: NDArray[np.floating[Any]],
        action: NDArray[np.floating[Any]],
        next_state: NDArray[np.floating[Any]],
        population: NDArray[np.floating[Any]],
    ) -> float:
        """
        Compute total reward: $r = r_0(x, u) + r_{\text{MF}}(x, m)$

        Args:
            state: Current state
            action: Current action
            next_state: Next state
            population: Population histogram

        Returns:
            Total reward
        """
        individual = self._individual_reward(state, action, next_state)
        mean_field = self.compute_mean_field_coupling(state, population)
        return individual + mean_field

    def get_population_state(self) -> NDArray[np.floating[Any]]:
        """
        Get current population distribution as histogram.

        Computes empirical distribution:
        $m(x) \approx \frac{1}{N} \\sum_{i=1}^N \\delta_{x_i}(x)$

        Returns:
            Population histogram of shape (population_bins,)
            Normalized to sum to 1.0
        """
        if self.agent_states is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        # For now, return uniform distribution as placeholder
        # Subclasses should implement proper binning based on state space structure
        histogram = np.ones(self.population_bins, dtype=np.float32) / self.population_bins
        return histogram

    def step(
        self, action: NDArray[np.floating[Any]]
    ) -> tuple[NDArray[np.floating[Any]], float, bool, bool, dict[str, Any]]:
        r"""
        Execute one timestep of the MFG dynamics.

        This represents a single agent's interaction. In a full MFG setting,
        all agents would act, but here we track one representative agent.

        Args:
            action: Action $u \in \mathbb{R}^{action\_dim}$

        Returns:
            Tuple (observation, reward, terminated, truncated, info):
            - observation: Next state $x'$
            - reward: Reward $r(x, u, m)$
            - terminated: Whether episode ended naturally
            - truncated: Whether episode hit time limit
            - info: Additional diagnostic information
        """
        if self.agent_states is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        # Clip action to bounds
        action = np.clip(action, self.action_bounds[0], self.action_bounds[1])

        # Get representative agent state (first agent)
        state = self.agent_states[0]

        # Get current population distribution
        population = self.get_population_state()

        # Apply dynamics: x' = x + f(x, u, m) * dt + sigma * sqrt(dt) * noise
        drift = self._drift(state, action, population)
        noise = self.rng.normal(0, self.noise_std * np.sqrt(self.dt), size=state.shape)
        next_state = state + drift * self.dt + noise

        # Clip to state bounds
        state_low, state_high = self._get_state_bounds()
        next_state = np.clip(next_state, state_low, state_high)

        # Compute reward
        reward = self._compute_reward(state, action, next_state, population)

        # Update agent state
        self.agent_states[0] = next_state

        # Check termination
        terminated = self._is_terminated(next_state)
        truncated = self.current_step >= self.max_steps - 1

        self.current_step += 1

        info = {
            "step": self.current_step,
            "population_mass": np.sum(population),
        }

        return next_state.astype(np.float32), float(reward), terminated, truncated, info

    def _is_terminated(self, state: NDArray[np.floating[Any]]) -> bool:
        """
        Check if episode should terminate.

        Default: no early termination.
        Override for problem-specific terminal conditions.

        Args:
            state: Current state

        Returns:
            True if episode should end
        """
        return False

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.floating[Any]], dict[str, Any]]:
        """
        Reset environment to initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)

        Returns:
            Tuple (observation, info):
            - observation: Initial state $x_0$
            - info: Additional information
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.current_step = 0

        # Sample initial states for all agents
        self.agent_states = self._sample_initial_states()

        # Compute initial population distribution
        self.population_histogram = self.get_population_state()

        # Return initial state of first agent (representative)
        initial_state = self.agent_states[0]

        info = {
            "step": 0,
            "population_mass": np.sum(self.population_histogram),
        }

        return initial_state.astype(np.float32), info

    def render(self) -> None:  # noqa: B027
        """
        Render environment state.

        Default: no rendering.
        Override for problem-specific visualization.
        """

    def close(self) -> None:  # noqa: B027
        """Clean up resources."""
