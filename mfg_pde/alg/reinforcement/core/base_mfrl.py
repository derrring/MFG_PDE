"""
Base classes for Mean Field Reinforcement Learning (MFRL) solvers.

This module provides the abstract base classes and configuration structures
for all MFRL approaches to Mean Field Games. It establishes the interface
between classical MFG problems and reinforcement learning algorithms.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from mfg_pde.alg.base_solver import BaseRLSolver

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mfg_pde.core.mfg_problem import MFGProblem

logger = logging.getLogger(__name__)


@dataclass
class RLSolverConfig:
    """Configuration for MFRL solvers."""

    # RL algorithm parameters
    learning_rate: float = 3e-4
    discount_factor: float = 0.99
    exploration_rate: float = 0.1
    exploration_decay: float = 0.995

    # Training parameters
    max_episodes: int = 10000
    max_steps_per_episode: int = 1000
    batch_size: int = 64
    replay_buffer_size: int = 100000

    # Population parameters
    population_size: int = 1000
    population_update_frequency: int = 10
    population_sample_size: int = 100

    # Network architecture
    hidden_layers: list[int] | None = None
    activation: str = "relu"
    use_dueling: bool = False

    # Convergence criteria
    convergence_tolerance: float = 1e-4
    convergence_window: int = 100

    # Logging and monitoring
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5000

    def __post_init__(self) -> None:
        """Set default hidden layers if not provided."""
        if self.hidden_layers is None:
            self.hidden_layers = [256, 256]


@dataclass
class RLSolverResult:
    """Result container for MFRL solvers."""

    # Learned policy and value functions
    learned_policy: Any = None  # Policy network or function
    learned_value_function: Any = None  # Value network or function
    population_state_evolution: NDArray | None = None  # m(t,x) evolution

    # Training performance
    episode_rewards: list[float] | None = None
    episode_lengths: list[int] | None = None
    loss_history: list[float] | None = None

    # Population dynamics
    population_convergence: list[float] | None = None
    nash_equilibrium_error: float = np.inf
    population_stability: float = 0.0

    # Solution quality
    final_episode_reward: float = -np.inf
    average_reward_last_100: float = -np.inf
    policy_convergence: bool = False

    # Training statistics
    total_episodes: int = 0
    total_timesteps: int = 0
    training_time: float = 0.0

    # MFG-specific metrics
    value_function_approximation: NDArray | None = None  # u(t,x) approximation
    density_approximation: NDArray | None = None  # m(t,x) approximation
    nash_equilibrium_achieved: bool = False


class BaseMFRLSolver(BaseRLSolver):
    """
    Abstract base class for Mean Field Reinforcement Learning solvers.

    This class provides the common interface for all MFRL approaches to
    solving Mean Field Games. It handles the integration between classical
    MFG problem formulations and modern RL algorithms.

    Mathematical Framework:
    - Individual Agent Problem: max E[∑ r(s_t, a_t, m_t) | π, m]
    - Population Consistency: m_t = E[φ(s_t) | π]
    - Nash Equilibrium: π* ∈ BR(m*), m* = μ(π*)

    Key Responsibilities:
    - Environment setup and management
    - Population state tracking and updates
    - Policy learning with population feedback
    - Convergence monitoring and Nash equilibrium detection
    """

    def __init__(
        self,
        problem: MFGProblem,
        config: RLSolverConfig | None = None,
        **kwargs: Any,
    ):
        """
        Initialize MFRL solver base class.

        Args:
            problem: MFG problem instance
            config: RL solver configuration
            **kwargs: Additional solver arguments
        """
        super().__init__(problem, **kwargs)
        self.config = config or RLSolverConfig()
        self.logger = self._get_logger()

        # Initialize RL infrastructure
        self._setup_environment()
        self._setup_population_tracking()
        self._setup_policy_learning()

    @abstractmethod
    def _setup_environment(self) -> None:
        """Setup MFG environment for RL training."""

    @abstractmethod
    def _setup_population_tracking(self) -> None:
        """Setup population state tracking and updates."""

    @abstractmethod
    def _setup_policy_learning(self) -> None:
        """Setup policy learning algorithm and networks."""

    @abstractmethod
    def _update_population_state(self, agent_states: NDArray) -> NDArray:
        """
        Update population state based on current agent distribution.

        Args:
            agent_states: Current states of all agents

        Returns:
            Updated population state (mean field)
        """

    @abstractmethod
    def _compute_individual_reward(self, state: NDArray, action: NDArray, population_state: NDArray) -> float:
        """
        Compute reward for individual agent given population state.

        Args:
            state: Agent's current state
            action: Agent's action
            population_state: Current population state (mean field)

        Returns:
            Individual reward
        """

    @abstractmethod
    def _check_nash_equilibrium(self) -> tuple[bool, float]:
        """
        Check if current policy constitutes a Nash equilibrium.

        Returns:
            Tuple of (is_nash_equilibrium, nash_error)
        """

    def solve(self) -> RLSolverResult:
        """
        Solve MFG problem using reinforcement learning.

        Returns:
            RLSolverResult with learned policy and population dynamics
        """
        self.logger.info("Starting MFRL solver training")

        result = RLSolverResult()

        try:
            # Training loop
            episode_rewards = []
            episode_lengths = []
            population_errors = []

            for episode in range(self.config.max_episodes):
                # Run episode
                episode_reward, episode_length = self._run_episode(episode)
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)

                # Update population state periodically
                if episode % self.config.population_update_frequency == 0:
                    population_error = self._update_population_and_check_convergence()
                    population_errors.append(population_error)

                # Check convergence
                if self._check_training_convergence(episode_rewards):
                    self.logger.info(f"Training converged after {episode + 1} episodes")
                    result.policy_convergence = True
                    break

                # Periodic logging
                if (episode + 1) % self.config.log_interval == 0:
                    avg_reward = np.mean(episode_rewards[-self.config.log_interval :])
                    self.logger.info(f"Episode {episode + 1}: Avg reward = {avg_reward:.3f}")

            # Finalize results
            result.episode_rewards = episode_rewards
            result.episode_lengths = episode_lengths
            result.population_convergence = population_errors
            result.total_episodes = len(episode_rewards)
            result.final_episode_reward = episode_rewards[-1] if episode_rewards else -np.inf

            # Compute final metrics
            if len(episode_rewards) >= 100:
                result.average_reward_last_100 = np.mean(episode_rewards[-100:])

            # Check Nash equilibrium
            result.nash_equilibrium_achieved, result.nash_equilibrium_error = self._check_nash_equilibrium()

            # Extract learned policy and value function
            result.learned_policy = self._extract_policy()
            result.learned_value_function = self._extract_value_function()

            self.logger.info(f"MFRL training completed: {result.total_episodes} episodes")

        except Exception as e:
            self.logger.error(f"MFRL solver failed: {e}")
            raise

        return result

    @abstractmethod
    def _run_episode(self, episode_num: int) -> tuple[float, int]:
        """
        Run a single training episode.

        Args:
            episode_num: Current episode number

        Returns:
            Tuple of (episode_reward, episode_length)
        """

    @abstractmethod
    def _update_population_and_check_convergence(self) -> float:
        """
        Update population state and check convergence.

        Returns:
            Population convergence error
        """

    @abstractmethod
    def _extract_policy(self) -> Any:
        """Extract learned policy for result."""

    @abstractmethod
    def _extract_value_function(self) -> Any:
        """Extract learned value function for result."""

    def _check_training_convergence(self, episode_rewards: list[float]) -> bool:
        """Check if training has converged based on reward stability."""
        if len(episode_rewards) < self.config.convergence_window:
            return False

        recent_rewards = episode_rewards[-self.config.convergence_window :]
        reward_std = np.std(recent_rewards)

        return reward_std < self.config.convergence_tolerance

    def _get_current_population_state(self) -> NDArray:
        """Get current population state representation."""
        # This will be implemented by specific solvers
        raise NotImplementedError("Subclasses must implement population state access")

    def _validate_mfg_problem(self) -> None:
        """Validate that MFG problem is suitable for RL approach."""
        # Check if problem has required components for RL
        # Use getattr instead of hasattr for validation (Issue #543)
        required_attrs = ["T", "domain"]
        for attr in required_attrs:
            if getattr(self.problem, attr, None) is None:
                raise ValueError(f"MFG problem missing required attribute: {attr}")

        self.logger.info("MFG problem validation passed for RL approach")
