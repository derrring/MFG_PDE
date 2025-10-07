"""
MFG Maze Environment for Reinforcement Learning.

Gymnasium-compatible environment for Mean Field Games in maze settings.
Integrates maze generation with MFG population dynamics, providing
a testbed for multi-agent reinforcement learning algorithms.

Mathematical Framework:
- State space: Agent position (discrete or continuous) + population density
- Action space: Movement directions (4-connected or 8-connected)
- Reward: Distance to goal + population interaction + collision penalties
- Population state: Density distribution over maze cells

Author: MFG_PDE Team
Date: October 2025
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

try:
    from scipy.ndimage import gaussian_filter

    SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    SCIPY_AVAILABLE = False

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Check for Gymnasium availability
try:
    import gymnasium as gym
    from gymnasium import spaces

    GYMNASIUM_AVAILABLE = True
except ImportError:
    GYMNASIUM_AVAILABLE = False
    # Provide dummy types for type checking
    gym = None
    spaces = None


class ActionType(Enum):
    """Action space types for maze navigation."""

    FOUR_CONNECTED = "4-connected"  # Up, Down, Left, Right
    EIGHT_CONNECTED = "8-connected"  # Add diagonals
    CONTINUOUS = "continuous"  # Continuous velocity control


class RewardType(Enum):
    """Reward function types for MFG maze environments."""

    SPARSE = "sparse"  # Only reward at goal
    DENSE = "dense"  # Distance-based shaping
    MFG_STANDARD = "mfg_standard"  # Classical MFG running cost + terminal reward
    CONGESTION = "congestion"  # Population-dependent costs


@dataclass
class MFGMazeConfig:
    """Configuration for MFG Maze Environment."""

    # Maze structure
    maze_array: NDArray  # 1 = wall, 0 = open
    start_positions: list[tuple[int, int]] | None = None
    goal_positions: list[tuple[int, int]] | None = None

    # Population parameters
    population_size: int = 100
    population_update_frequency: int = 10
    population_smoothing: float = 0.1  # Kernel bandwidth for KDE
    num_agents: int = 1

    # Action space
    action_type: ActionType = ActionType.FOUR_CONNECTED
    move_cost: float = 0.01  # Cost per movement

    # Reward structure
    reward_type: RewardType = RewardType.MFG_STANDARD
    goal_reward: float = 1.0
    collision_penalty: float = -1.0
    congestion_weight: float = 0.1  # Weight for population interaction

    # Episode parameters
    max_episode_steps: int = 1000
    time_penalty: float = -0.001  # Penalty per timestep

    # Observation space
    include_population_in_obs: bool = True
    population_obs_radius: int = 3  # Local neighborhood for population

    # Physical dimensions (optional)
    cell_size: float = 1.0
    use_continuous_dynamics: bool = False

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.maze_array is None:
            raise ValueError("maze_array must be provided")
        if self.population_size < 1:
            raise ValueError("population_size must be >= 1")
        if self.num_agents < 1:
            raise ValueError("num_agents must be >= 1")


class PopulationState:
    """
    Efficient representation of population distribution in maze.

    Tracks agent density across maze cells using histogram approximation
    and provides KDE-based smooth density estimates when needed.
    """

    def __init__(
        self,
        maze_shape: tuple[int, int],
        smoothing: float = 0.1,
    ):
        """
        Initialize population state tracker.

        Args:
            maze_shape: (rows, cols) shape of maze
            smoothing: Kernel bandwidth for density smoothing
        """
        self.maze_shape = maze_shape
        self.smoothing = smoothing

        # Histogram representation (fast)
        self.density_histogram = np.zeros(maze_shape, dtype=np.float32)
        self.smoothed_density = np.zeros_like(self.density_histogram)

        # Agent position tracking
        self.agent_positions: list[tuple[int, int]] = []
        self.num_agents = 0

    def update(self, agent_positions: list[tuple[int, int]]) -> None:
        """
        Update population density from agent positions.

        Args:
            agent_positions: List of (row, col) positions
        """
        self.agent_positions = agent_positions
        self.num_agents = len(agent_positions)

        # Reset histogram
        self.density_histogram.fill(0)

        # Count agents per cell
        for row, col in agent_positions:
            if 0 <= row < self.maze_shape[0] and 0 <= col < self.maze_shape[1]:
                self.density_histogram[row, col] += 1

        # Normalize to probability distribution
        if self.num_agents > 0:
            self.density_histogram /= self.num_agents

        self._apply_smoothing()

    def get_density_at(self, position: tuple[int, int]) -> float:
        """
        Get population density at a specific position.

        Args:
            position: (row, col) position

        Returns:
            Density value at position
        """
        row, col = position
        if 0 <= row < self.maze_shape[0] and 0 <= col < self.maze_shape[1]:
            return float(self.density_histogram[row, col])
        return 0.0

    def get_local_density(
        self,
        position: tuple[int, int],
        radius: int = 3,
    ) -> NDArray:
        """
        Get local density neighborhood around position.

        Args:
            position: (row, col) center position
            radius: Neighborhood radius

        Returns:
            Local density patch (2*radius+1, 2*radius+1)
        """
        row, col = position
        local_patch = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.float32)

        source = self.smoothed_density if self.smoothing > 0 else self.density_histogram

        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                r, c = row + dr, col + dc
                if 0 <= r < self.maze_shape[0] and 0 <= c < self.maze_shape[1]:
                    local_patch[dr + radius, dc + radius] = source[r, c]

        return local_patch

    def get_full_density(self) -> NDArray:
        """
        Get full population density field.

        Returns:
            Density array (rows, cols)
        """
        source = self.smoothed_density if self.smoothing > 0 else self.density_histogram
        return source.copy()

    def _apply_smoothing(self) -> None:
        """Apply optional smoothing to the density histogram."""
        if self.smoothing <= 0 or not SCIPY_AVAILABLE:
            self.smoothed_density = self.density_histogram.copy()
            return

        sigma = max(self.smoothing, 1e-3)
        smoothed = gaussian_filter(self.density_histogram, sigma=sigma, mode="constant")
        total = smoothed.sum()
        if total > 0:
            smoothed /= total
        self.smoothed_density = smoothed.astype(np.float32)


if GYMNASIUM_AVAILABLE:

    class MFGMazeEnvironment(gym.Env):
        """Gymnasium-compatible Mean Field Games maze environment."""

        metadata: dict[str, Any] = {"render_modes": ["human", "rgb_array"], "render_fps": 10}  # noqa: RUF012

        def __init__(self, config: MFGMazeConfig, render_mode: str | None = None):
            super().__init__()

            self.config = config
            self.render_mode = render_mode

            self.maze = config.maze_array
            self.rows, self.cols = self.maze.shape
            self.num_agents = config.num_agents

            if self.num_agents > np.prod(self.maze.shape):
                raise ValueError("num_agents exceeds number of available maze cells")

            self.population_state = PopulationState(
                maze_shape=(self.rows, self.cols),
                smoothing=config.population_smoothing,
            )

            self._action_deltas = self._build_action_deltas(config.action_type)
            self._single_action_space = spaces.Discrete(len(self._action_deltas))
            if self.num_agents == 1:
                self.action_space = self._single_action_space
            else:
                self.action_space = spaces.MultiDiscrete(
                    np.full(self.num_agents, self._single_action_space.n, dtype=np.int64)
                )

            self._obs_dim = self._compute_observation_dim()
            if self.num_agents == 1:
                self.observation_space = spaces.Box(
                    low=np.zeros(self._obs_dim, dtype=np.float32),
                    high=np.ones(self._obs_dim, dtype=np.float32),
                    dtype=np.float32,
                )
            else:
                self.observation_space = spaces.Box(
                    low=np.zeros((self.num_agents, self._obs_dim), dtype=np.float32),
                    high=np.ones((self.num_agents, self._obs_dim), dtype=np.float32),
                    dtype=np.float32,
                )

            self._open_cells = np.argwhere(self.maze == 0)
            if self._open_cells.size == 0:
                raise ValueError("Maze must contain at least one open cell")

            self.agent_positions = np.zeros((self.num_agents, 2), dtype=np.int32)
            self.agent_goals = np.zeros((self.num_agents, 2), dtype=np.int32)
            self.agent_done = np.zeros(self.num_agents, dtype=bool)
            self.episode_rewards = np.zeros(self.num_agents, dtype=np.float32)
            self.current_step = 0

        # ------------------------------------------------------------------
        # Gym API
        # ------------------------------------------------------------------
        def reset(
            self,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
        ) -> tuple[NDArray[np.float32], dict[str, Any]]:
            super().reset(seed=seed)

            self.current_step = 0
            self.agent_done.fill(False)
            self.episode_rewards.fill(0.0)

            self.agent_positions = self._select_positions(self.config.start_positions)
            self.agent_goals = self._select_positions(self.config.goal_positions)

            self.population_state.update([tuple(pos) for pos in self.agent_positions])

            observations = self._get_observation()
            info = self._build_info()
            return observations, info

        def step(
            self,
            actions: int | NDArray[np.int64] | list[int],
        ) -> tuple[NDArray[np.float32], NDArray[np.float32], bool, bool, dict[str, Any]]:
            self.current_step += 1

            action_array = self._prepare_actions(actions)
            rewards = np.zeros(self.num_agents, dtype=np.float32)
            terminated_agents = np.zeros(self.num_agents, dtype=bool)

            for idx in range(self.num_agents):
                if self.agent_done[idx]:
                    continue

                delta_row, delta_col = self._action_deltas[action_array[idx]]
                new_position = (
                    int(self.agent_positions[idx, 0] + delta_row),
                    int(self.agent_positions[idx, 1] + delta_col),
                )

                if self._is_wall(new_position):
                    rewards[idx] += self.config.collision_penalty
                    self.agent_done[idx] = True
                    terminated_agents[idx] = True
                    continue

                self.agent_positions[idx] = np.array(new_position, dtype=np.int32)
                rewards[idx] += self._calculate_movement_reward(idx)

                if tuple(new_position) == tuple(self.agent_goals[idx]):
                    rewards[idx] += self.config.goal_reward
                    self.agent_done[idx] = True
                    terminated_agents[idx] = True

            self.population_state.update([tuple(pos) for pos in self.agent_positions])

            observations = self._get_observation()
            self.episode_rewards += rewards

            terminated = bool(np.all(self.agent_done))
            truncated = self.current_step >= self.config.max_episode_steps

            done_mask = self.agent_done.copy()
            if truncated:
                done_mask |= True

            info = self._build_info()
            info.update(
                {
                    "agents_done": done_mask,
                    "terminated_agents": terminated_agents,
                    "rewards": rewards.copy(),
                }
            )

            return observations, rewards, terminated, truncated, info

        # ------------------------------------------------------------------
        # Helper methods
        # ------------------------------------------------------------------
        def _build_action_deltas(self, action_type: ActionType) -> list[tuple[int, int]]:
            if action_type == ActionType.FOUR_CONNECTED:
                return [(-1, 0), (1, 0), (0, -1), (0, 1)]
            if action_type == ActionType.EIGHT_CONNECTED:
                return [
                    (-1, 0),
                    (1, 0),
                    (0, -1),
                    (0, 1),
                    (-1, -1),
                    (-1, 1),
                    (1, -1),
                    (1, 1),
                ]
            raise NotImplementedError("Continuous action space not yet implemented")

        def _compute_observation_dim(self) -> int:
            base_dim = 5  # position (2), goal (2), time remaining (1)
            if self.config.include_population_in_obs:
                radius = self.config.population_obs_radius
                base_dim += (2 * radius + 1) ** 2
            return base_dim

        def _select_positions(self, configured: list[tuple[int, int]] | None) -> NDArray[np.int32]:
            if configured:
                candidates = [pos for pos in configured if not self._is_wall(pos)]
                if len(candidates) < self.num_agents:
                    raise ValueError("Not enough valid configured positions for the requested number of agents")
                indices = self.np_random.choice(len(candidates), size=self.num_agents, replace=False)
                return np.array([candidates[i] for i in indices], dtype=np.int32)

            if self._open_cells.shape[0] < self.num_agents:
                raise ValueError("Maze does not have enough open cells for all agents")

            indices = self.np_random.choice(self._open_cells.shape[0], size=self.num_agents, replace=False)
            return self._open_cells[indices].astype(np.int32)

        def _prepare_actions(self, actions: int | NDArray[np.int64] | list[int]) -> NDArray[np.int64]:
            if np.isscalar(actions):
                return np.full(self.num_agents, int(actions), dtype=np.int64)

            action_array = np.asarray(actions, dtype=np.int64)
            if action_array.shape == (self.num_agents,):
                return action_array
            if action_array.shape == (self.num_agents, 1):
                return action_array.reshape(self.num_agents)

            raise ValueError(
                f"Actions must be scalar or array of shape ({self.num_agents},), got shape {action_array.shape}"
            )

        def _calculate_movement_reward(self, agent_idx: int) -> float:
            reward = -self.config.move_cost + self.config.time_penalty

            if self.config.reward_type == RewardType.CONGESTION:
                density = self.population_state.get_density_at(tuple(self.agent_positions[agent_idx]))
                reward -= self.config.congestion_weight * density

            if self.config.reward_type == RewardType.DENSE:
                distance = self._manhattan_distance(self.agent_positions[agent_idx], self.agent_goals[agent_idx])
                reward -= 0.001 * float(distance)

            return float(reward)

        def _get_observation(self) -> NDArray[np.float32]:
            obs = np.zeros((self.num_agents, self._obs_dim), dtype=np.float32)
            for idx in range(self.num_agents):
                obs[idx] = self._build_agent_observation(idx)
            return obs if self.num_agents > 1 else obs.reshape(self._obs_dim)

        def _build_agent_observation(self, idx: int) -> NDArray[np.float32]:
            row, col = self.agent_positions[idx]
            goal_row, goal_col = self.agent_goals[idx]
            components: list[float] = [
                row / max(self.rows - 1, 1),
                col / max(self.cols - 1, 1),
                goal_row / max(self.rows - 1, 1),
                goal_col / max(self.cols - 1, 1),
                1.0 - self.current_step / max(self.config.max_episode_steps, 1),
            ]

            if self.config.include_population_in_obs:
                density_patch = self.population_state.get_local_density(
                    (int(row), int(col)), radius=self.config.population_obs_radius
                )
                components.extend(density_patch.astype(np.float32).flatten())

            return np.asarray(components, dtype=np.float32)

        def _build_info(self) -> dict[str, Any]:
            return {
                "positions": self.agent_positions.copy(),
                "goals": self.agent_goals.copy(),
                "current_step": self.current_step,
                "episode_rewards": self.episode_rewards.copy(),
                "nash_error": 0.0,
            }

        def _is_wall(self, position: tuple[int, int]) -> bool:
            row, col = position
            if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
                return True
            return bool(self.maze[row, col] == 1)

        def _manhattan_distance(self, pos1: NDArray[np.int32], pos2: NDArray[np.int32]) -> int:
            return int(abs(int(pos1[0]) - int(pos2[0])) + abs(int(pos1[1]) - int(pos2[1])))

        # ------------------------------------------------------------------
        # Rendering helpers
        # ------------------------------------------------------------------
        def render(self) -> NDArray | None:
            if self.render_mode == "rgb_array":
                return self._render_rgb_array()
            if self.render_mode == "human":
                print(self._render_ascii())
                return None
            return None

        def _render_ascii(self) -> str:
            agent_map = {tuple(pos): idx for idx, pos in enumerate(self.agent_positions)}
            goal_map = {tuple(goal): True for goal in self.agent_goals}
            lines: list[str] = []
            for r in range(self.rows):
                line_chars: list[str] = []
                for c in range(self.cols):
                    key = (r, c)
                    if key in agent_map:
                        idx = agent_map[key]
                        symbol = "A" if idx == 0 else chr(65 + (idx % 26))
                        line_chars.append(symbol)
                    elif key in goal_map:
                        line_chars.append("G")
                    elif self.maze[r, c] == 1:
                        line_chars.append("#")
                    else:
                        density = self.population_state.get_density_at(key)
                        line_chars.append("." if density > 0.1 else " ")
                lines.append("".join(line_chars))
            return "\n".join(lines)

        def _render_rgb_array(self) -> NDArray:
            img = np.ones((self.rows, self.cols, 3), dtype=np.uint8) * 255
            img[self.maze == 1] = [0, 0, 0]

            density = self.population_state.get_full_density()
            density_normalized = (density * 255).astype(np.uint8)
            img[:, :, 2] = np.minimum(img[:, :, 2], 255 - density_normalized)

            for goal in self.agent_goals:
                img[tuple(goal)] = [0, 255, 0]

            for idx, pos in enumerate(self.agent_positions):
                color = [255, 0, 0] if idx == 0 else [255, 165, 0]
                img[tuple(pos)] = color

            return img

else:

    class MFGMazeEnvironment:  # type: ignore[no-redef]  # pragma: no cover - graceful degradation
        """Placeholder when Gymnasium is unavailable."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("Gymnasium is required for MFGMazeEnvironment. Install with: pip install mfg_pde[rl]")
