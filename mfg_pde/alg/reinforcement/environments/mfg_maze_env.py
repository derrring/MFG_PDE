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
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

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
    gym = None  # type: ignore
    spaces = None  # type: ignore


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

        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                r, c = row + dr, col + dc
                if 0 <= r < self.maze_shape[0] and 0 <= c < self.maze_shape[1]:
                    local_patch[dr + radius, dc + radius] = self.density_histogram[r, c]

        return local_patch

    def get_full_density(self) -> NDArray:
        """
        Get full population density field.

        Returns:
            Density array (rows, cols)
        """
        return self.density_histogram.copy()


if GYMNASIUM_AVAILABLE:

    class MFGMazeEnvironment(gym.Env):
        """
        Gymnasium environment for Mean Field Games in mazes.

        This environment integrates maze navigation with population dynamics,
        providing a testbed for MFG reinforcement learning algorithms.

        Observation Space:
        - Agent position: (row, col)
        - Goal position: (row, col) or (goal_id,)
        - Population state: Local density neighborhood
        - Time remaining: Scalar

        Action Space:
        - Discrete: 4 or 8 movement directions
        - Continuous: Velocity vector (future)

        Reward Structure:
        - Goal reward: Large positive reward for reaching goal
        - Movement cost: Small negative reward per step
        - Collision penalty: Large negative reward for hitting walls
        - Congestion cost: Population-dependent cost (MFG interaction)
        - Time penalty: Small negative reward per timestep
        """

        metadata: ClassVar[dict[str, Any]] = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

        def __init__(
            self,
            config: MFGMazeConfig,
            render_mode: str | None = None,
        ):
            """
            Initialize MFG Maze Environment.

            Args:
                config: Environment configuration
                render_mode: Rendering mode ("human" or "rgb_array")
            """
            super().__init__()

            self.config = config
            self.render_mode = render_mode

            # Maze structure
            self.maze = config.maze_array
            self.rows, self.cols = self.maze.shape

            # Start and goal positions
            self.start_positions = config.start_positions or [(1, 1)]
            self.goal_positions = config.goal_positions or [(self.rows - 2, self.cols - 2)]

            # Population state
            self.population_state = PopulationState(
                maze_shape=(self.rows, self.cols),
                smoothing=config.population_smoothing,
            )

            # Action space
            if config.action_type == ActionType.FOUR_CONNECTED:
                self.action_space = spaces.Discrete(4)
                self.action_deltas = [
                    (-1, 0),  # Up
                    (1, 0),  # Down
                    (0, -1),  # Left
                    (0, 1),  # Right
                ]
            elif config.action_type == ActionType.EIGHT_CONNECTED:
                self.action_space = spaces.Discrete(8)
                self.action_deltas = [
                    (-1, 0),  # Up
                    (1, 0),  # Down
                    (0, -1),  # Left
                    (0, 1),  # Right
                    (-1, -1),  # Up-Left
                    (-1, 1),  # Up-Right
                    (1, -1),  # Down-Left
                    (1, 1),  # Down-Right
                ]
            else:
                raise NotImplementedError("Continuous action space not yet implemented")

            # Observation space
            obs_dict = {
                "position": spaces.Box(
                    low=np.array([0, 0]),
                    high=np.array([self.rows - 1, self.cols - 1]),
                    dtype=np.int32,
                ),
                "goal": spaces.Box(
                    low=np.array([0, 0]),
                    high=np.array([self.rows - 1, self.cols - 1]),
                    dtype=np.int32,
                ),
                "time_remaining": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            }

            if config.include_population_in_obs:
                radius = config.population_obs_radius
                obs_dict["local_density"] = spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(2 * radius + 1, 2 * radius + 1),
                    dtype=np.float32,
                )

            self.observation_space = spaces.Dict(obs_dict)

            # Episode state
            self.agent_position: tuple[int, int] | None = None
            self.goal_position: tuple[int, int] | None = None
            self.current_step = 0
            self.episode_reward = 0.0

        def reset(
            self,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
        ) -> tuple[dict[str, NDArray], dict[str, Any]]:
            """
            Reset environment to initial state.

            Args:
                seed: Random seed
                options: Additional options

            Returns:
                observation: Initial observation
                info: Additional information
            """
            super().reset(seed=seed)

            # Sample start and goal positions
            self.agent_position = self.start_positions[self.np_random.integers(len(self.start_positions))]
            self.goal_position = self.goal_positions[self.np_random.integers(len(self.goal_positions))]

            # Reset episode state
            self.current_step = 0
            self.episode_reward = 0.0

            # Initialize population (single agent for now)
            self.population_state.update([self.agent_position])

            observation = self._get_observation()
            info = self._get_info()

            return observation, info

        def step(
            self,
            action: int,
        ) -> tuple[dict[str, NDArray], float, bool, bool, dict[str, Any]]:
            """
            Execute one step in environment.

            Args:
                action: Action to take

            Returns:
                observation: Next observation
                reward: Reward for this step
                terminated: Episode terminated (goal reached or collision)
                truncated: Episode truncated (max steps)
                info: Additional information
            """
            self.current_step += 1

            # Execute action
            delta = self.action_deltas[action]
            new_position = (
                self.agent_position[0] + delta[0],
                self.agent_position[1] + delta[1],
            )

            # Check for collision
            if self._is_wall(new_position):
                # Collision with wall
                reward = self.config.collision_penalty
                terminated = True
                truncated = False
            elif new_position == self.goal_position:
                # Reached goal
                self.agent_position = new_position
                reward = self.config.goal_reward
                terminated = True
                truncated = False
            else:
                # Valid move
                self.agent_position = new_position

                # Calculate reward
                reward = self._calculate_reward()

                terminated = False
                truncated = self.current_step >= self.config.max_episode_steps

            # Update population state
            self.population_state.update([self.agent_position])

            self.episode_reward += reward

            observation = self._get_observation()
            info = self._get_info()

            return observation, reward, terminated, truncated, info

        def _is_wall(self, position: tuple[int, int]) -> bool:
            """Check if position is a wall or out of bounds."""
            row, col = position
            if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
                return True
            return bool(self.maze[row, col] == 1)

        def _calculate_reward(self) -> float:
            """Calculate reward for current state."""
            reward = 0.0

            # Movement cost
            reward -= self.config.move_cost

            # Time penalty
            reward += self.config.time_penalty

            # Congestion cost (population interaction)
            if self.config.reward_type == RewardType.CONGESTION:
                density = self.population_state.get_density_at(self.agent_position)
                reward -= self.config.congestion_weight * density

            # Distance-based shaping (dense rewards)
            if self.config.reward_type == RewardType.DENSE:
                distance = self._manhattan_distance(self.agent_position, self.goal_position)
                reward -= 0.001 * distance

            return reward

        def _manhattan_distance(self, pos1: tuple[int, int], pos2: tuple[int, int]) -> int:
            """Calculate Manhattan distance between two positions."""
            return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

        def _get_observation(self) -> dict[str, NDArray]:
            """Get current observation."""
            obs = {
                "position": np.array(self.agent_position, dtype=np.int32),
                "goal": np.array(self.goal_position, dtype=np.int32),
                "time_remaining": np.array(
                    [1.0 - self.current_step / self.config.max_episode_steps],
                    dtype=np.float32,
                ),
            }

            if self.config.include_population_in_obs:
                obs["local_density"] = self.population_state.get_local_density(
                    self.agent_position,
                    radius=self.config.population_obs_radius,
                )

            return obs

        def _get_info(self) -> dict[str, Any]:
            """Get additional information."""
            return {
                "position": self.agent_position,
                "goal": self.goal_position,
                "distance_to_goal": self._manhattan_distance(self.agent_position, self.goal_position),
                "current_step": self.current_step,
                "episode_reward": self.episode_reward,
            }

        def render(self) -> NDArray | None:
            """
            Render the environment.

            Returns:
                RGB array if render_mode is "rgb_array", else None
            """
            if self.render_mode == "rgb_array":
                return self._render_rgb_array()
            elif self.render_mode == "human":
                print(self._render_ascii())
                return None
            return None

        def _render_ascii(self) -> str:
            """Render ASCII representation of maze."""
            lines = []
            for r in range(self.rows):
                line = []
                for c in range(self.cols):
                    if (r, c) == self.agent_position:
                        line.append("A")
                    elif (r, c) == self.goal_position:
                        line.append("G")
                    elif self.maze[r, c] == 1:
                        line.append("#")
                    else:
                        density = self.population_state.get_density_at((r, c))
                        if density > 0.1:
                            line.append(".")
                        else:
                            line.append(" ")
                lines.append("".join(line))
            return "\n".join(lines)

        def _render_rgb_array(self) -> NDArray:
            """Render RGB array representation."""
            # Create RGB image
            img = np.ones((self.rows, self.cols, 3), dtype=np.uint8) * 255

            # Draw walls (black)
            img[self.maze == 1] = [0, 0, 0]

            # Draw population density (blue gradient)
            density = self.population_state.get_full_density()
            density_normalized = (density * 255).astype(np.uint8)
            img[:, :, 2] = np.minimum(img[:, :, 2], 255 - density_normalized)

            # Draw goal (green)
            if self.goal_position:
                img[self.goal_position] = [0, 255, 0]

            # Draw agent (red)
            if self.agent_position:
                img[self.agent_position] = [255, 0, 0]

            return img

else:
    # Provide dummy class when Gymnasium not available
    class MFGMazeEnvironment:
        """Dummy class when Gymnasium is not available."""

        def __init__(self, *args, **kwargs):
            raise ImportError("Gymnasium is required for MFGMazeEnvironment. " "Install with: pip install gymnasium")
