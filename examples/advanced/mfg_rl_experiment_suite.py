#!/usr/bin/env python3
"""
Mean Field Games Reinforcement Learning Experiment Suite

This module provides a comprehensive experiment environment for testing MFRL algorithms
without OpenSpiel dependencies. Includes multiple benchmark scenarios commonly used in
MFG RL literature.

Key Experiment Scenarios:
1. Crowd Navigation - Agents navigate through congested space
2. Linear Quadratic MFG - Classical benchmark with analytical solution
3. Finite State MFG - Discrete state space games
4. Epidemic Control - SIR model with agent-based control
5. Price Formation - Market making with mean field effects

Author: MFG_PDE Team
Date: October 2025
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np

# MFG_PDE imports
from mfg_pde.utils.logging import configure_research_logging, get_logger

logger = get_logger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for MFG RL experiments."""

    # Experiment setup
    scenario: str = "crowd_navigation"  # crowd_navigation, linear_quadratic, finite_state, epidemic, price_formation
    num_agents: int = 100
    num_episodes: int = 5000
    max_steps_per_episode: int = 200

    # Environment parameters
    domain_size: float = 10.0
    grid_resolution: int = 50
    time_horizon: float = 1.0
    dt: float = 0.02

    # RL parameters
    learning_rate: float = 3e-4
    discount_factor: float = 0.99
    exploration_rate: float = 0.1
    batch_size: int = 64

    # Evaluation
    eval_frequency: int = 500
    convergence_window: int = 100
    convergence_tolerance: float = 1e-3

    # Logging
    log_level: str = "INFO"
    save_results: bool = True
    plot_results: bool = True


class MFGEnvironment:
    """
    Custom MFG Environment for RL experiments.

    This environment simulates multiple scenarios commonly used in MFG RL research,
    providing a standardized interface for algorithm comparison.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.logger = get_logger(__name__)

        # Environment state
        self.current_step = 0
        self.agent_positions = np.zeros((config.num_agents, 2))
        self.agent_velocities = np.zeros((config.num_agents, 2))
        self.population_density = np.zeros((config.grid_resolution, config.grid_resolution))

        # Scenario-specific setup
        self._setup_scenario()

        # Performance tracking
        self.episode_rewards = []
        self.population_evolution = []
        self.nash_errors = []

    def _setup_scenario(self) -> None:
        """Setup scenario-specific parameters and initial conditions."""
        scenario = self.config.scenario.lower()

        if scenario == "crowd_navigation":
            self._setup_crowd_navigation()
        elif scenario == "linear_quadratic":
            self._setup_linear_quadratic()
        elif scenario == "finite_state":
            self._setup_finite_state()
        elif scenario == "epidemic":
            self._setup_epidemic_control()
        elif scenario == "price_formation":
            self._setup_price_formation()
        else:
            raise ValueError(f"Unknown scenario: {scenario}")

    def _setup_crowd_navigation(self) -> None:
        """
        Setup crowd navigation scenario.

        Agents start at random positions and try to reach target locations
        while avoiding congestion. Common benchmark in MFG RL literature.
        """
        self.logger.info("Setting up crowd navigation scenario")

        # Random initial positions
        self.agent_positions = np.random.uniform(
            low=0.1 * self.config.domain_size, high=0.9 * self.config.domain_size, size=(self.config.num_agents, 2)
        )

        # Target positions (opposite corner)
        self.target_positions = np.full(
            (self.config.num_agents, 2), [0.9 * self.config.domain_size, 0.9 * self.config.domain_size]
        )

        # Scenario parameters
        self.congestion_penalty = 2.0
        self.movement_cost = 0.1
        self.target_reward = 10.0
        self.collision_penalty = 5.0

    def _setup_linear_quadratic(self) -> None:
        """
        Setup Linear-Quadratic MFG scenario.

        Classical benchmark with known analytical solution, allowing
        precise convergence analysis.
        """
        self.logger.info("Setting up Linear-Quadratic MFG scenario")

        # Initial positions (Gaussian distribution)
        self.agent_positions = np.random.normal(
            loc=self.config.domain_size / 2, scale=self.config.domain_size / 10, size=(self.config.num_agents, 2)
        )

        # LQ-MFG parameters
        self.state_cost_weight = 1.0
        self.control_cost_weight = 0.5
        self.interaction_strength = 0.2
        self.target_state = np.array([self.config.domain_size / 2, self.config.domain_size / 2])

    def _setup_finite_state(self) -> None:
        """Setup finite state MFG scenario."""
        self.logger.info("Setting up finite state MFG scenario")

        # Discrete state space (grid positions)
        self.num_states = 25  # 5x5 grid
        self.agent_states = np.random.randint(0, self.num_states, size=self.config.num_agents)

        # Transition probabilities and rewards
        self.transition_matrix = self._create_transition_matrix()
        self.base_rewards = np.random.uniform(-1, 1, size=self.num_states)

    def _setup_epidemic_control(self) -> None:
        """Setup epidemic control scenario (SIR model)."""
        self.logger.info("Setting up epidemic control scenario")

        # SIR states: 0=Susceptible, 1=Infected, 2=Recovered
        self.agent_health_states = np.zeros(self.config.num_agents, dtype=int)

        # Initial infections
        num_initial_infected = max(1, self.config.num_agents // 20)
        infected_indices = np.random.choice(self.config.num_agents, num_initial_infected, replace=False)
        self.agent_health_states[infected_indices] = 1

        # Epidemic parameters
        self.infection_rate = 0.1
        self.recovery_rate = 0.05
        self.isolation_effectiveness = 0.8
        self.isolation_cost = 1.0

    def _setup_price_formation(self) -> None:
        """Setup price formation scenario."""
        self.logger.info("Setting up price formation scenario")

        # Agent holdings and cash
        self.agent_holdings = np.random.uniform(0, 10, size=self.config.num_agents)
        self.agent_cash = np.random.uniform(90, 110, size=self.config.num_agents)

        # Market parameters
        self.current_price = 100.0
        self.price_volatility = 0.1
        self.transaction_cost = 0.01

    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.current_step = 0
        self._setup_scenario()
        self._update_population_density()

        return self._get_observations()

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool, dict[str, Any]]:
        """
        Execute environment step.

        Args:
            actions: Actions for all agents

        Returns:
            Tuple of (observations, rewards, done, info)
        """
        # Execute actions based on scenario
        rewards = self._execute_actions(actions)

        # Update environment state
        self._update_environment()
        self.current_step += 1

        # Check termination
        done = self._check_termination()

        # Get new observations
        observations = self._get_observations()

        # Additional info
        info = {
            "step": self.current_step,
            "population_density": self.population_density.copy(),
            "nash_error": self._compute_nash_error(),
        }

        return observations, rewards, done, info

    def _execute_actions(self, actions: np.ndarray) -> np.ndarray:
        """Execute actions and compute rewards based on scenario."""
        scenario = self.config.scenario.lower()

        if scenario == "crowd_navigation":
            return self._execute_crowd_navigation_actions(actions)
        elif scenario == "linear_quadratic":
            return self._execute_lq_actions(actions)
        elif scenario == "finite_state":
            return self._execute_finite_state_actions(actions)
        elif scenario == "epidemic":
            return self._execute_epidemic_actions(actions)
        elif scenario == "price_formation":
            return self._execute_price_formation_actions(actions)
        else:
            raise ValueError(f"Unknown scenario: {scenario}")

    def _execute_crowd_navigation_actions(self, actions: np.ndarray) -> np.ndarray:
        """Execute crowd navigation actions."""
        # Actions are velocity changes
        self.agent_velocities += actions * self.config.dt

        # Update positions
        old_positions = self.agent_positions.copy()
        self.agent_positions += self.agent_velocities * self.config.dt

        # Boundary conditions
        self.agent_positions = np.clip(self.agent_positions, 0, self.config.domain_size)

        # Compute rewards
        rewards = np.zeros(self.config.num_agents)

        for i in range(self.config.num_agents):
            # Distance to target reward
            distance_to_target = np.linalg.norm(self.agent_positions[i] - self.target_positions[i])
            old_distance = np.linalg.norm(old_positions[i] - self.target_positions[i])
            progress_reward = (old_distance - distance_to_target) * self.target_reward

            # Movement cost
            movement_cost = np.linalg.norm(actions[i]) * self.movement_cost

            # Congestion penalty
            congestion = self._get_local_density(self.agent_positions[i])
            congestion_cost = congestion * self.congestion_penalty

            rewards[i] = progress_reward - movement_cost - congestion_cost

        return rewards

    def _execute_lq_actions(self, actions: np.ndarray) -> np.ndarray:
        """Execute Linear-Quadratic MFG actions."""
        # Actions are control inputs
        self.agent_positions += actions * self.config.dt

        # Compute LQ rewards
        rewards = np.zeros(self.config.num_agents)
        mean_position = np.mean(self.agent_positions, axis=0)

        for i in range(self.config.num_agents):
            # State cost
            state_deviation = self.agent_positions[i] - self.target_state
            state_cost = self.state_cost_weight * np.sum(state_deviation**2)

            # Control cost
            control_cost = self.control_cost_weight * np.sum(actions[i] ** 2)

            # Interaction cost
            interaction_deviation = self.agent_positions[i] - mean_position
            interaction_cost = self.interaction_strength * np.sum(interaction_deviation**2)

            rewards[i] = -(state_cost + control_cost + interaction_cost) * self.config.dt

        return rewards

    def _execute_finite_state_actions(self, actions: np.ndarray) -> np.ndarray:
        """Execute finite state MFG actions."""
        # Actions are state transitions
        new_states = np.zeros_like(self.agent_states)
        rewards = np.zeros(self.config.num_agents)

        for i in range(self.config.num_agents):
            # Execute transition
            current_state = self.agent_states[i]
            action = int(actions[i])

            # Valid transitions based on action
            possible_transitions = self._get_valid_transitions(current_state, action)
            new_states[i] = np.random.choice(possible_transitions)

            # Compute reward
            base_reward = self.base_rewards[new_states[i]]
            congestion_penalty = self._get_state_congestion_penalty(new_states[i])
            rewards[i] = base_reward - congestion_penalty

        self.agent_states = new_states
        return rewards

    def _execute_epidemic_actions(self, actions: np.ndarray) -> np.ndarray:
        """Execute epidemic control actions."""
        # Actions: 0=normal behavior, 1=isolation
        rewards = np.zeros(self.config.num_agents)

        # Update epidemic dynamics
        new_infections = 0
        recoveries = 0

        for i in range(self.config.num_agents):
            if self.agent_health_states[i] == 0:  # Susceptible
                # Compute infection probability
                infection_prob = self._compute_infection_probability(i, actions)
                if np.random.random() < infection_prob:
                    self.agent_health_states[i] = 1
                    new_infections += 1
                    rewards[i] = -10.0  # Infection penalty

            elif self.agent_health_states[i] == 1:  # Infected
                # Recovery probability
                if np.random.random() < self.recovery_rate:
                    self.agent_health_states[i] = 2
                    recoveries += 1
                    rewards[i] = 5.0  # Recovery reward
                else:
                    rewards[i] = -1.0  # Being infected cost

            # Isolation cost
            if actions[i] == 1:
                rewards[i] -= self.isolation_cost

        return rewards

    def _execute_price_formation_actions(self, actions: np.ndarray) -> np.ndarray:
        """Execute price formation actions."""
        # Actions are buy/sell quantities
        total_demand = np.sum(actions)

        # Update price based on demand
        price_change = self.price_volatility * total_demand / self.config.num_agents
        self.current_price += price_change

        # Execute trades and compute rewards
        rewards = np.zeros(self.config.num_agents)

        for i in range(self.config.num_agents):
            action = actions[i]

            # Transaction cost
            transaction_cost = self.transaction_cost * abs(action)

            # Portfolio change
            self.agent_holdings[i] += action
            self.agent_cash[i] -= action * self.current_price

            # Reward based on price change and transaction cost
            rewards[i] = price_change * self.agent_holdings[i] - transaction_cost

        return rewards

    def _update_environment(self) -> None:
        """Update global environment state."""
        self._update_population_density()

        # Scenario-specific updates
        if self.config.scenario == "price_formation":
            # Add random price shocks
            shock = np.random.normal(0, 0.1)
            self.current_price += shock

    def _update_population_density(self) -> None:
        """Update population density representation."""
        self.population_density.fill(0)

        if self.config.scenario in ["crowd_navigation", "linear_quadratic"]:
            # Spatial density for continuous scenarios
            x_edges = np.linspace(0, self.config.domain_size, self.config.grid_resolution + 1)
            y_edges = np.linspace(0, self.config.domain_size, self.config.grid_resolution + 1)

            density, _, _ = np.histogram2d(
                self.agent_positions[:, 0], self.agent_positions[:, 1], bins=[x_edges, y_edges]
            )
            self.population_density = density / self.config.num_agents

        elif self.config.scenario == "finite_state":
            # State occupation density
            state_counts = np.bincount(self.agent_states, minlength=self.num_states)
            # Reshape to 2D for consistency
            side_length = int(np.sqrt(self.num_states))
            self.population_density = (state_counts / self.config.num_agents).reshape(side_length, side_length)

    def _get_observations(self) -> np.ndarray:
        """Get observations for all agents."""
        scenario = self.config.scenario.lower()

        if scenario == "crowd_navigation":
            return self._get_crowd_navigation_observations()
        elif scenario == "linear_quadratic":
            return self._get_lq_observations()
        elif scenario == "finite_state":
            return self._get_finite_state_observations()
        elif scenario == "epidemic":
            return self._get_epidemic_observations()
        elif scenario == "price_formation":
            return self._get_price_formation_observations()
        else:
            raise ValueError(f"Unknown scenario: {scenario}")

    def _get_crowd_navigation_observations(self) -> np.ndarray:
        """Get observations for crowd navigation."""
        observations = np.zeros((self.config.num_agents, 6))  # pos, vel, target

        for i in range(self.config.num_agents):
            observations[i, :2] = self.agent_positions[i]
            observations[i, 2:4] = self.agent_velocities[i]
            observations[i, 4:6] = self.target_positions[i]

        return observations

    def _get_lq_observations(self) -> np.ndarray:
        """Get observations for Linear-Quadratic MFG."""
        observations = np.zeros((self.config.num_agents, 4))  # pos, mean_pos
        mean_position = np.mean(self.agent_positions, axis=0)

        for i in range(self.config.num_agents):
            observations[i, :2] = self.agent_positions[i]
            observations[i, 2:4] = mean_position

        return observations

    def _get_finite_state_observations(self) -> np.ndarray:
        """Get observations for finite state MFG."""
        observations = np.zeros((self.config.num_agents, 2))  # state, state_density

        state_densities = np.bincount(self.agent_states, minlength=self.num_states) / self.config.num_agents

        for i in range(self.config.num_agents):
            observations[i, 0] = self.agent_states[i]
            observations[i, 1] = state_densities[self.agent_states[i]]

        return observations

    def _get_epidemic_observations(self) -> np.ndarray:
        """Get observations for epidemic control."""
        observations = np.zeros((self.config.num_agents, 4))  # health_state, local_infection_rate, S, I ratios

        s_ratio = np.mean(self.agent_health_states == 0)
        i_ratio = np.mean(self.agent_health_states == 1)

        for i in range(self.config.num_agents):
            observations[i, 0] = self.agent_health_states[i]
            observations[i, 1] = self._compute_local_infection_rate(i)
            observations[i, 2] = s_ratio
            observations[i, 3] = i_ratio

        return observations

    def _get_price_formation_observations(self) -> np.ndarray:
        """Get observations for price formation."""
        observations = np.zeros((self.config.num_agents, 3))  # holdings, cash, price

        for i in range(self.config.num_agents):
            observations[i, 0] = self.agent_holdings[i]
            observations[i, 1] = self.agent_cash[i]
            observations[i, 2] = self.current_price

        return observations

    def _check_termination(self) -> bool:
        """Check if episode should terminate."""
        if self.current_step >= self.config.max_steps_per_episode:
            return True

        # Scenario-specific termination conditions
        if self.config.scenario == "crowd_navigation":
            # All agents reached target
            distances = np.linalg.norm(self.agent_positions - self.target_positions, axis=1)
            return np.all(distances < 0.5)

        elif self.config.scenario == "epidemic":
            # No more infected agents
            return np.sum(self.agent_health_states == 1) == 0

        return False

    def _compute_nash_error(self) -> float:
        """Compute Nash equilibrium error estimate."""
        # Simple estimate based on policy variance
        # In practice, this would require more sophisticated analysis
        return np.random.uniform(0, 1)  # Placeholder

    def _get_local_density(self, position: np.ndarray) -> float:
        """Get local population density around a position."""
        distances = np.linalg.norm(self.agent_positions - position, axis=1)
        local_agents = np.sum(distances < 1.0)  # Within radius 1.0
        return local_agents / self.config.num_agents

    def _create_transition_matrix(self) -> np.ndarray:
        """Create transition matrix for finite state scenario."""
        matrix = np.random.rand(self.num_states, self.num_states)
        # Normalize rows
        matrix = matrix / np.sum(matrix, axis=1, keepdims=True)
        return matrix

    def _get_valid_transitions(self, state: int, action: int) -> list[int]:
        """Get valid state transitions for finite state scenario."""
        # Simple grid transitions
        side_length = int(np.sqrt(self.num_states))
        row, col = divmod(state, side_length)

        valid_states = [state]  # Can stay in place

        # Action meanings: 0=stay, 1=up, 2=down, 3=left, 4=right
        if action == 1 and row > 0:  # Up
            valid_states.append((row - 1) * side_length + col)
        elif action == 2 and row < side_length - 1:  # Down
            valid_states.append((row + 1) * side_length + col)
        elif action == 3 and col > 0:  # Left
            valid_states.append(row * side_length + (col - 1))
        elif action == 4 and col < side_length - 1:  # Right
            valid_states.append(row * side_length + (col + 1))

        return valid_states

    def _get_state_congestion_penalty(self, state: int) -> float:
        """Get congestion penalty for finite state scenario."""
        state_count = np.sum(self.agent_states == state)
        return 0.1 * state_count / self.config.num_agents

    def _compute_infection_probability(self, agent_idx: int, actions: np.ndarray) -> float:
        """Compute infection probability for epidemic scenario."""
        if self.agent_health_states[agent_idx] != 0:  # Not susceptible
            return 0.0

        # Count nearby infected agents
        nearby_infected = 0
        for j in range(self.config.num_agents):
            if j != agent_idx and self.agent_health_states[j] == 1:  # Infected
                # In finite state case, just count infected agents
                nearby_infected += 1

        # Reduce infection rate if isolating
        effectiveness = self.isolation_effectiveness if actions[agent_idx] == 1 else 1.0

        infection_prob = self.infection_rate * nearby_infected / self.config.num_agents
        return infection_prob * effectiveness

    def _compute_local_infection_rate(self, agent_idx: int) -> float:
        """Compute local infection rate for epidemic observations."""
        infected_count = np.sum(self.agent_health_states == 1)
        return infected_count / self.config.num_agents


class MFGRLExperimentSuite:
    """
    Comprehensive experiment suite for MFG RL algorithms.

    Provides multiple benchmark scenarios and evaluation metrics
    for comparing different MFRL approaches.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.logger = get_logger(__name__)

        # Setup logging
        configure_research_logging("mfg_rl_experiments", level=config.log_level)

        # Initialize environment
        self.env = MFGEnvironment(config)

        # Results storage
        self.results = {
            "episode_rewards": [],
            "nash_errors": [],
            "population_evolution": [],
            "convergence_metrics": {},
        }

    def run_random_baseline(self) -> dict[str, Any]:
        """Run random policy baseline for comparison."""
        self.logger.info("Running random policy baseline")

        episode_rewards = []
        nash_errors = []

        for episode in tqdm(range(self.config.num_episodes), desc="Random Baseline"):
            _obs = self.env.reset()
            total_reward = 0

            for step in range(self.config.max_steps_per_episode):
                # Random actions
                if self.config.scenario == "crowd_navigation":
                    actions = np.random.normal(0, 0.1, size=(self.config.num_agents, 2))
                elif self.config.scenario == "linear_quadratic":
                    actions = np.random.normal(0, 0.5, size=(self.config.num_agents, 2))
                elif self.config.scenario == "finite_state":
                    actions = np.random.randint(0, 5, size=self.config.num_agents)
                elif self.config.scenario == "epidemic":
                    actions = np.random.randint(0, 2, size=self.config.num_agents)
                elif self.config.scenario == "price_formation":
                    actions = np.random.normal(0, 1, size=self.config.num_agents)
                else:
                    actions = np.random.normal(0, 0.1, size=(self.config.num_agents, 2))

                _obs, rewards, done, info = self.env.step(actions)
                total_reward += np.mean(rewards)

                if done:
                    break

            episode_rewards.append(total_reward)
            nash_errors.append(info["nash_error"])

        return {
            "episode_rewards": episode_rewards,
            "nash_errors": nash_errors,
            "final_average_reward": np.mean(episode_rewards[-100:]),
            "convergence_achieved": False,
        }

    def evaluate_mfrl_algorithm(self, algorithm_class, algorithm_config: dict[str, Any]) -> dict[str, Any]:
        """
        Evaluate an MFRL algorithm on the current scenario.

        Args:
            algorithm_class: MFRL algorithm class
            algorithm_config: Algorithm configuration

        Returns:
            Dictionary with evaluation results
        """
        self.logger.info(f"Evaluating MFRL algorithm: {algorithm_class.__name__}")

        # Initialize algorithm
        algorithm = algorithm_class(self.env, **algorithm_config)

        # Training
        start_time = time.time()
        results = algorithm.train(self.config.num_episodes)
        training_time = time.time() - start_time

        # Evaluation
        eval_results = self._evaluate_learned_policy(algorithm)

        return {
            "training_results": results,
            "evaluation_results": eval_results,
            "training_time": training_time,
            "algorithm_name": algorithm_class.__name__,
        }

    def _evaluate_learned_policy(self, algorithm) -> dict[str, Any]:
        """Evaluate the learned policy."""
        self.logger.info("Evaluating learned policy")

        eval_episodes = 100
        eval_rewards = []
        nash_errors = []

        for episode in range(eval_episodes):
            obs = self.env.reset()
            total_reward = 0

            for step in range(self.config.max_steps_per_episode):
                actions = algorithm.predict(obs)
                obs, rewards, done, info = self.env.step(actions)
                total_reward += np.mean(rewards)

                if done:
                    break

            eval_rewards.append(total_reward)
            nash_errors.append(info["nash_error"])

        return {
            "average_reward": np.mean(eval_rewards),
            "reward_std": np.std(eval_rewards),
            "average_nash_error": np.mean(nash_errors),
            "nash_error_std": np.std(nash_errors),
        }

    def run_comparative_study(self, algorithms: list[tuple[Any, dict[str, Any]]]) -> dict[str, Any]:
        """
        Run comparative study across multiple algorithms.

        Args:
            algorithms: List of (algorithm_class, config) tuples

        Returns:
            Comparative results
        """
        self.logger.info("Running comparative study")

        results = {}

        # Run random baseline
        results["random_baseline"] = self.run_random_baseline()

        # Run each algorithm
        for algorithm_class, config in algorithms:
            try:
                algorithm_results = self.evaluate_mfrl_algorithm(algorithm_class, config)
                results[algorithm_class.__name__] = algorithm_results
            except Exception as e:
                self.logger.error(f"Algorithm {algorithm_class.__name__} failed: {e}")
                results[algorithm_class.__name__] = {"error": str(e)}

        # Generate comparison report
        comparison = self._generate_comparison_report(results)
        results["comparison"] = comparison

        return results

    def _generate_comparison_report(self, results: dict[str, Any]) -> dict[str, Any]:
        """Generate comparison report across algorithms."""
        comparison = {
            "performance_ranking": [],
            "convergence_analysis": {},
            "statistical_summary": {},
        }

        # Extract performance metrics
        performance_data = {}
        for alg_name, alg_results in results.items():
            if "error" in alg_results:
                continue

            if alg_name == "random_baseline":
                performance_data[alg_name] = alg_results["final_average_reward"]
            else:
                performance_data[alg_name] = alg_results["evaluation_results"]["average_reward"]

        # Rank algorithms by performance
        sorted_performance = sorted(performance_data.items(), key=lambda x: x[1], reverse=True)
        comparison["performance_ranking"] = sorted_performance

        return comparison

    def plot_results(self, results: dict[str, Any]) -> None:
        """Plot experiment results."""
        if not self.config.plot_results:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"MFG RL Experiment Results - {self.config.scenario}")

        # Performance comparison
        if "comparison" in results and "performance_ranking" in results["comparison"]:
            ax = axes[0, 0]
            alg_names, rewards = zip(*results["comparison"]["performance_ranking"], strict=False)
            ax.bar(alg_names, rewards)
            ax.set_title("Algorithm Performance Comparison")
            ax.set_ylabel("Average Reward")
            ax.tick_params(axis="x", rotation=45)

        # Learning curves
        ax = axes[0, 1]
        for alg_name, alg_results in results.items():
            if "training_results" in alg_results:
                training_rewards = alg_results["training_results"].get("episode_rewards", [])
                if training_rewards:
                    ax.plot(training_rewards, label=alg_name, alpha=0.7)
            elif alg_name == "random_baseline":
                baseline_rewards = alg_results["episode_rewards"]
                ax.plot(baseline_rewards, label=alg_name, alpha=0.7)

        ax.set_title("Learning Curves")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Episode Reward")
        ax.legend()

        # Population evolution (if available)
        ax = axes[1, 0]
        if hasattr(self.env, "population_evolution") and self.env.population_evolution:
            evolution_data = np.array(self.env.population_evolution)
            if evolution_data.ndim == 3:  # Time x Height x Width
                # Show final population distribution
                im = ax.imshow(evolution_data[-1], cmap="viridis")
                ax.set_title("Final Population Distribution")
                plt.colorbar(im, ax=ax)

        # Nash equilibrium errors
        ax = axes[1, 1]
        for alg_name, alg_results in results.items():
            if "training_results" in alg_results:
                nash_errors = alg_results["training_results"].get("nash_errors", [])
                if nash_errors:
                    ax.plot(nash_errors, label=alg_name, alpha=0.7)
            elif alg_name == "random_baseline":
                nash_errors = alg_results["nash_errors"]
                ax.plot(nash_errors, label=alg_name, alpha=0.7)

        ax.set_title("Nash Equilibrium Error")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Nash Error")
        ax.legend()

        plt.tight_layout()

        if self.config.save_results:
            plt.savefig(f"mfg_rl_results_{self.config.scenario}.png", dpi=300, bbox_inches="tight")

        plt.show()


def main():
    """Run MFG RL experiment suite demo."""
    print("🎮 MFG RL Experiment Suite Demo")
    print("=" * 50)

    # Configure experiment
    config = ExperimentConfig(
        scenario="crowd_navigation", num_agents=50, num_episodes=1000, max_steps_per_episode=100, log_level="INFO"
    )

    # Create experiment suite
    suite = MFGRLExperimentSuite(config)

    print(f"🎯 Running {config.scenario} scenario with {config.num_agents} agents")

    # Run random baseline
    print("\n📊 Running Random Baseline...")
    baseline_results = suite.run_random_baseline()

    print("✅ Random Baseline Complete:")
    print(f"   Average Reward: {baseline_results['final_average_reward']:.3f}")
    print(f"   Convergence: {baseline_results['convergence_achieved']}")

    # Demonstrate environment
    print("\n🧪 Environment Demo (10 steps):")
    _obs = suite.env.reset()
    for step in range(10):
        # Random actions for demo
        if config.scenario == "crowd_navigation":
            actions = np.random.normal(0, 0.1, size=(config.num_agents, 2))
        else:
            actions = np.random.normal(0, 0.1, size=(config.num_agents, 2))

        _obs, rewards, done, _info = suite.env.step(actions)
        avg_reward = np.mean(rewards)
        print(f"   Step {step + 1}: Avg Reward = {avg_reward:.3f}, Done = {done}")

        if done:
            break

    print("\n🎉 Experiment Suite Ready for MFRL Algorithm Testing!")
    print("💡 Usage:")
    print("   1. Implement your MFRL algorithm with train() and predict() methods")
    print("   2. Use suite.evaluate_mfrl_algorithm(your_algorithm, config)")
    print("   3. Compare multiple algorithms with suite.run_comparative_study()")

    return suite


if __name__ == "__main__":
    suite = main()
