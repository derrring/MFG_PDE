"""
Multi-Population MFG Maze Environment.

Extends MFGMazeEnvironment to support heterogeneous agent types with:
- Different objectives (reward functions)
- Different capabilities (action spaces, speeds)
- Strategic interactions through multi-population state

Mathematical Framework:
- K agent types with type-specific policies π^k(a|s,m)
- Multi-population state m = (m^1, ..., m^K)
- Nash equilibrium: Each type plays best response to other populations

Example Applications:
- Predator-prey dynamics
- Traffic with cars/trucks/motorcycles
- Epidemic models (S-I-R)

Author: MFG_PDE Team
Date: October 2025
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from mfg_pde.alg.reinforcement.environments.mfg_maze_env import (
    ActionType,
    RewardType,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Check for Gymnasium availability
try:
    import gymnasium as gym
    from gymnasium import spaces

    GYMNASIUM_AVAILABLE = True
except ImportError:  # pragma: no cover
    GYMNASIUM_AVAILABLE = False
    gym = None
    spaces = None


@dataclass
class AgentTypeConfig:
    """Configuration for a single agent type in multi-population MFG."""

    # Identification
    type_id: str  # Unique identifier (e.g., "predator", "prey")
    type_index: int  # Numeric index (0, 1, ..., K-1)

    # Capabilities
    action_type: ActionType = ActionType.FOUR_CONNECTED
    speed_multiplier: float = 1.0  # Relative speed (1.0 = normal)

    # Objectives
    reward_type: RewardType = RewardType.MFG_STANDARD
    goal_reward: float = 10.0
    collision_penalty: float = -1.0
    move_cost: float = 0.01

    # Population interaction
    congestion_weight: float = 0.1  # Weight for own-population congestion
    cross_population_weights: dict[str, float] = field(default_factory=dict)  # Weights for other populations

    # Positions
    start_positions: list[tuple[int, int]] | None = None
    goal_positions: list[tuple[int, int]] | None = None

    # Number of agents
    num_agents: int = 5


@dataclass
class MultiPopulationMazeConfig:
    """Configuration for multi-population MFG maze environment."""

    # Maze structure (shared by all types)
    maze_array: NDArray  # 1 = wall, 0 = open

    # Agent types (K populations)
    agent_types: dict[str, AgentTypeConfig]  # {type_id: config}

    # Population dynamics
    population_smoothing: float = 0.1
    population_update_frequency: int = 10

    # Episode parameters
    max_episode_steps: int = 1000
    time_penalty: float = -0.001

    # Observation
    include_population_in_obs: bool = True
    population_obs_radius: int = 3

    def __post_init__(self):
        """Validate configuration."""
        assert len(self.agent_types) >= 2, "Multi-population requires at least 2 agent types"

        # Assign type indices
        for idx, (_type_id, config) in enumerate(self.agent_types.items()):
            config.type_index = idx


class MultiPopulationState:
    """
    Multi-population state representation for heterogeneous MFG.

    Maintains density distributions for K agent types:
        m = (m^1, m^2, ..., m^K)

    Each m^k is a discrete distribution over maze cells.
    """

    def __init__(self, maze_shape: tuple[int, int], agent_types: dict[str, AgentTypeConfig]):
        """
        Initialize multi-population state.

        Args:
            maze_shape: (height, width) of maze
            agent_types: Dict of agent type configurations
        """
        self.maze_shape = maze_shape
        self.agent_types = agent_types
        self.K = len(agent_types)

        # Population densities for each type
        # distributions[type_id] = density over maze cells [height, width]
        self.distributions: dict[str, NDArray] = {
            type_id: np.zeros(maze_shape, dtype=np.float32) for type_id in agent_types
        }

    def update_from_positions(self, positions: dict[str, list[tuple[int, int]]], smoothing: float = 0.1) -> None:
        """
        Update population distributions from agent positions.

        Args:
            positions: Dict {type_id: [(x1, y1), (x2, y2), ...]}
            smoothing: KDE bandwidth for density smoothing
        """
        from mfg_pde.alg.reinforcement.environments.mfg_maze_env import PopulationState

        for type_id, agent_positions in positions.items():
            if len(agent_positions) == 0:
                self.distributions[type_id] = np.zeros(self.maze_shape, dtype=np.float32)
                continue

            # Create temporary single-population state for this type
            temp_state = PopulationState(self.maze_shape, smoothing=smoothing)
            temp_state.update(agent_positions)

            # Copy density
            self.distributions[type_id] = temp_state.density_histogram.copy()

    def get_density_field(self, type_id: str) -> NDArray:
        """
        Get density field for specific agent type.

        Args:
            type_id: Agent type identifier

        Returns:
            Density distribution [height, width]
        """
        return self.distributions[type_id]

    def get_all_densities(self) -> dict[str, NDArray]:
        """
        Get all population densities.

        Returns:
            Dict {type_id: density_field}
        """
        return self.distributions.copy()

    def get_local_densities(self, position: tuple[int, int], radius: int = 3) -> dict[str, NDArray]:
        """
        Get local population densities around a position.

        Args:
            position: (row, col) center position
            radius: Neighborhood radius

        Returns:
            Dict {type_id: local_density} where local_density is flattened window
        """
        row, col = position
        height, width = self.maze_shape

        # Extract local window for each type
        local_densities = {}

        for type_id, density_field in self.distributions.items():
            # Compute window bounds
            r_min = max(0, row - radius)
            r_max = min(height, row + radius + 1)
            c_min = max(0, col - radius)
            c_max = min(width, col + radius + 1)

            # Extract and pad
            local_window = density_field[r_min:r_max, c_min:c_max]

            # Pad to fixed size (2*radius+1, 2*radius+1)
            target_size = 2 * radius + 1
            padded = np.zeros((target_size, target_size), dtype=np.float32)

            # Compute offsets for centering
            offset_r = radius - (row - r_min)
            offset_c = radius - (col - c_min)

            padded[
                offset_r : offset_r + local_window.shape[0],
                offset_c : offset_c + local_window.shape[1],
            ] = local_window

            local_densities[type_id] = padded.flatten()

        return local_densities

    def get_flattened_state(self) -> NDArray:
        """
        Get flattened multi-population state for neural network input.

        Returns:
            Concatenated densities [K * height * width]
        """
        # Sort by type_index for consistent ordering
        sorted_types = sorted(self.agent_types.items(), key=lambda x: x[1].type_index)

        flattened = []
        for type_id, _ in sorted_types:
            flattened.append(self.distributions[type_id].flatten())

        return np.concatenate(flattened, dtype=np.float32)


class MultiPopulationMazeEnvironment:
    """
    Multi-population Mean Field Game environment for maze navigation.

    Supports K heterogeneous agent types with:
    - Different reward functions r^k(s, a, m)
    - Different action spaces A^k
    - Strategic interactions through multi-population state m = (m^1, ..., m^K)

    Implements Gymnasium interface for each agent type.
    """

    def __init__(self, config: MultiPopulationMazeConfig):
        """
        Initialize multi-population MFG maze environment.

        Args:
            config: Multi-population environment configuration
        """
        if not GYMNASIUM_AVAILABLE:
            raise ImportError("Gymnasium is required for MFG environments. Install with: pip install gymnasium")

        self.config = config
        self.maze_array = config.maze_array
        self.height, self.width = config.maze_array.shape

        # Agent types
        self.agent_types = config.agent_types
        self.K = len(config.agent_types)

        # Multi-population state
        self.multi_pop_state = MultiPopulationState(
            maze_shape=(self.height, self.width), agent_types=config.agent_types
        )

        # Agent positions for each type
        # positions[type_id] = [(row1, col1), (row2, col2), ...]
        self.positions: dict[str, list[tuple[int, int]]] = {}

        # Episode tracking
        self.current_step = 0
        self.episode_count = 0

        # Define observation and action spaces for each type
        self._setup_spaces()

    def _setup_spaces(self):
        """Setup observation and action spaces for each agent type."""
        self.observation_spaces: dict[str, spaces.Space] = {}
        self.action_spaces: dict[str, spaces.Space] = {}

        for type_id, type_config in self.agent_types.items():
            # Action space (discrete for now)
            if type_config.action_type == ActionType.FOUR_CONNECTED:
                action_dim = 4
            elif type_config.action_type == ActionType.EIGHT_CONNECTED:
                action_dim = 8
            else:
                raise NotImplementedError(f"Action type {type_config.action_type} not supported")

            self.action_spaces[type_id] = spaces.Discrete(action_dim)

            # Observation space: [position (2) + K local population densities]
            if self.config.include_population_in_obs:
                radius = self.config.population_obs_radius
                local_pop_size = (2 * radius + 1) ** 2

                # Position (2) + local densities for K types
                obs_dim = 2 + self.K * local_pop_size
            else:
                obs_dim = 2  # Just position

            self.observation_spaces[type_id] = spaces.Box(
                low=-1.0, high=float(max(self.height, self.width)), shape=(obs_dim,), dtype=np.float32
            )

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, NDArray], dict[str, Any]]:
        """
        Reset environment for all agent types.

        Returns:
            observations: Dict {type_id: observations_array}
                where observations_array is [num_agents_of_type, obs_dim]
            info: Dict with auxiliary information
        """
        if seed is not None:
            np.random.seed(seed)

        self.current_step = 0
        self.episode_count += 1

        # Initialize positions for each type
        self.positions = {}

        for type_id, type_config in self.agent_types.items():
            # Get valid start positions
            if type_config.start_positions:
                valid_starts = type_config.start_positions
            else:
                # Use all non-wall positions
                valid_starts = list(zip(*np.where(self.maze_array == 0), strict=False))

            # Sample positions for agents of this type
            num_agents = type_config.num_agents
            if len(valid_starts) < num_agents:
                raise ValueError(
                    f"Not enough valid start positions for type {type_id}: need {num_agents}, have {len(valid_starts)}"
                )

            indices = np.random.choice(len(valid_starts), size=num_agents, replace=False)
            self.positions[type_id] = [valid_starts[i] for i in indices]

        # Update multi-population state
        self.multi_pop_state.update_from_positions(self.positions, smoothing=self.config.population_smoothing)

        # Get observations for each type
        observations = self._get_observations()

        info = {
            "episode": self.episode_count,
            "step": self.current_step,
            "num_types": self.K,
        }

        return observations, info

    def step(
        self, actions: dict[str, NDArray]
    ) -> tuple[dict[str, NDArray], dict[str, NDArray], bool, bool, dict[str, Any]]:
        """
        Execute actions for all agent types.

        Args:
            actions: Dict {type_id: actions_array}
                where actions_array is [num_agents_of_type] integer actions

        Returns:
            observations: Dict {type_id: obs_array}
            rewards: Dict {type_id: rewards_array}
            terminated: Episode ended (all agents reached goals)
            truncated: Episode exceeded max steps
            info: Auxiliary information
        """
        self.current_step += 1

        # Execute actions for each type
        new_positions: dict[str, list[tuple[int, int]]] = {}
        rewards: dict[str, NDArray] = {}

        for type_id, type_config in self.agent_types.items():
            type_actions = actions[type_id]
            num_agents = len(self.positions[type_id])

            assert len(type_actions) == num_agents, (
                f"Action count mismatch for {type_id}: expected {num_agents}, got {len(type_actions)}"
            )

            # Compute new positions
            new_pos = []
            type_rewards = np.zeros(num_agents, dtype=np.float32)

            for i, action in enumerate(type_actions):
                old_pos = self.positions[type_id][i]
                next_pos = self._apply_action(old_pos, action, type_config)
                new_pos.append(next_pos)

                # Compute reward for this agent
                type_rewards[i] = self._compute_reward(old_pos, next_pos, action, type_id, type_config)

            new_positions[type_id] = new_pos
            rewards[type_id] = type_rewards

        # Update positions
        self.positions = new_positions

        # Update multi-population state
        if self.current_step % self.config.population_update_frequency == 0:
            self.multi_pop_state.update_from_positions(self.positions, smoothing=self.config.population_smoothing)

        # Get observations
        observations = self._get_observations()

        # Check termination
        terminated = self._check_termination()
        truncated = self.current_step >= self.config.max_episode_steps

        info = {
            "step": self.current_step,
            "multi_population_state": self.multi_pop_state.get_all_densities(),
        }

        return observations, rewards, terminated, truncated, info

    def _apply_action(self, position: tuple[int, int], action: int, type_config: AgentTypeConfig) -> tuple[int, int]:
        """
        Apply action to get next position.

        Args:
            position: Current (row, col)
            action: Integer action
            type_config: Agent type configuration

        Returns:
            Next position (row, col)
        """
        row, col = position

        # Action directions (4-connected or 8-connected)
        if type_config.action_type == ActionType.FOUR_CONNECTED:
            # 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT
            deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        elif type_config.action_type == ActionType.EIGHT_CONNECTED:
            # 0-3: cardinal, 4-7: diagonal
            deltas = [
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
                (-1, -1),
                (-1, 1),
                (1, -1),
                (1, 1),
            ]
        else:
            raise NotImplementedError(f"Action type {type_config.action_type}")

        # Apply speed multiplier (for now, just affects reward)
        # In future, could allow multi-step moves
        dr, dc = deltas[action]
        new_row = row + dr
        new_col = col + dc

        # Check bounds and walls
        if 0 <= new_row < self.height and 0 <= new_col < self.width and self.maze_array[new_row, new_col] == 0:
            return (new_row, new_col)
        else:
            # Collision - stay in place
            return position

    def _compute_reward(
        self,
        old_pos: tuple[int, int],
        new_pos: tuple[int, int],
        action: int,
        type_id: str,
        type_config: AgentTypeConfig,
    ) -> float:
        """
        Compute reward for single agent of given type.

        Args:
            old_pos: Previous position
            new_pos: New position
            action: Action taken
            type_id: Agent type ID
            type_config: Agent type configuration

        Returns:
            Reward value
        """
        reward = 0.0

        # Time penalty
        reward += self.config.time_penalty

        # Movement cost
        if new_pos != old_pos:
            reward -= type_config.move_cost * type_config.speed_multiplier

        # Collision penalty
        if new_pos == old_pos and self._is_collision(old_pos, action, type_config):
            reward += type_config.collision_penalty

        # Goal reward
        if type_config.goal_positions and new_pos in type_config.goal_positions:
            reward += type_config.goal_reward

        # Population interaction (congestion)
        if type_config.reward_type == RewardType.CONGESTION:
            # Own-population congestion
            own_density = self.multi_pop_state.get_density_field(type_id)[new_pos]
            reward -= type_config.congestion_weight * own_density

            # Cross-population interaction
            for other_type_id, weight in type_config.cross_population_weights.items():
                other_density = self.multi_pop_state.get_density_field(other_type_id)[new_pos]
                reward -= weight * other_density

        return reward

    def _is_collision(self, position: tuple[int, int], action: int, type_config: AgentTypeConfig) -> bool:
        """Check if action would result in collision with wall."""
        row, col = position

        if type_config.action_type == ActionType.FOUR_CONNECTED:
            deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        else:
            deltas = [
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
                (-1, -1),
                (-1, 1),
                (1, -1),
                (1, 1),
            ]

        dr, dc = deltas[action]
        new_row, new_col = row + dr, col + dc

        # Collision if out of bounds or wall
        if new_row < 0 or new_row >= self.height or new_col < 0 or new_col >= self.width:
            return True

        return self.maze_array[new_row, new_col] == 1

    def _get_observations(self) -> dict[str, NDArray]:
        """
        Get observations for all agents of all types.

        Returns:
            Dict {type_id: observations} where observations is [num_agents, obs_dim]
        """
        observations = {}

        for type_id, positions in self.positions.items():
            obs_list = []

            for pos in positions:
                # Position
                obs = np.array([pos[0], pos[1]], dtype=np.float32)

                # Population state (local densities for all types)
                if self.config.include_population_in_obs:
                    local_densities = self.multi_pop_state.get_local_densities(
                        pos, radius=self.config.population_obs_radius
                    )

                    # Concatenate in consistent order (sorted by type_index)
                    sorted_types = sorted(self.agent_types.items(), key=lambda x: x[1].type_index)
                    for other_type_id, _ in sorted_types:
                        obs = np.concatenate([obs, local_densities[other_type_id]])

                obs_list.append(obs)

            observations[type_id] = np.array(obs_list, dtype=np.float32)

        return observations

    def _check_termination(self) -> bool:
        """
        Check if episode should terminate.

        For now, simple criterion: all agents of all types reached their goals.
        """
        for type_id, type_config in self.agent_types.items():
            if not type_config.goal_positions:
                continue  # No goals for this type

            # Check if all agents of this type reached goal
            positions = self.positions[type_id]
            all_at_goal = all(pos in type_config.goal_positions for pos in positions)

            if not all_at_goal:
                return False  # At least one agent not at goal

        # All agents of all types at their goals
        return True

    def get_multi_population_state(self) -> MultiPopulationState:
        """Get current multi-population state."""
        return self.multi_pop_state

    def render(self, mode: str = "human") -> NDArray | None:
        """
        Render the multi-population environment.

        Args:
            mode: Rendering mode
                - "human": Print ASCII visualization to console
                - "rgb_array": Return RGB numpy array

        Returns:
            RGB array if mode="rgb_array", None otherwise
        """
        if mode == "rgb_array":
            return self._render_rgb_array()
        if mode == "human":
            print(self._render_ascii())
            return None
        return None

    def _render_ascii(self) -> str:
        """
        Render ASCII visualization with multi-population agents.

        Agent symbols by type:
        - Type 0: A, B, C, ... (uppercase letters)
        - Type 1: a, b, c, ... (lowercase letters)
        - Type 2+: 0, 1, 2, ... (numbers)

        Other symbols:
        - '#': Wall
        - 'G': Goal (if single goal for all types)
        - '.': High population density area
        - ' ': Empty space
        """
        # Map positions to (type_id, agent_idx)
        agent_map: dict[tuple[int, int], tuple[str, int]] = {}
        for type_id, positions in self.positions.items():
            for agent_idx, pos in enumerate(positions):
                agent_map[tuple(pos)] = (type_id, agent_idx)

        # Map goal positions by type
        goal_map: dict[tuple[int, int], set[str]] = {}
        for type_id, type_config in self.agent_types.items():
            if type_config.goal_positions:
                for goal_pos in type_config.goal_positions:
                    if tuple(goal_pos) not in goal_map:
                        goal_map[tuple(goal_pos)] = set()
                    goal_map[tuple(goal_pos)].add(type_id)

        lines: list[str] = []
        for r in range(self.height):
            line_chars: list[str] = []
            for c in range(self.width):
                key = (r, c)

                # Check for agents (highest priority)
                if key in agent_map:
                    type_id, agent_idx = agent_map[key]
                    type_index = self.agent_types[type_id].type_index

                    # Different symbol schemes for different types
                    if type_index == 0:
                        # First type: uppercase letters A-Z
                        symbol = chr(65 + (agent_idx % 26))
                    elif type_index == 1:
                        # Second type: lowercase letters a-z
                        symbol = chr(97 + (agent_idx % 26))
                    else:
                        # Additional types: numbers 0-9
                        symbol = str(agent_idx % 10)

                    line_chars.append(symbol)

                # Check for goals
                elif key in goal_map:
                    # If all types share this goal, use 'G'
                    # Otherwise use type-specific marker
                    goal_types = goal_map[key]
                    if len(goal_types) == self.K:
                        line_chars.append("G")
                    else:
                        # Use marker for first type at this goal
                        first_type_id = next(iter(goal_types))
                        type_index = self.agent_types[first_type_id].type_index
                        line_chars.append("*" if type_index > 0 else "G")

                # Check for walls
                elif self.maze_array[r, c] == 1:
                    line_chars.append("#")

                # Show population density
                else:
                    # Check if any population has significant density here
                    max_density = max(self.multi_pop_state.distributions[type_id][r, c] for type_id in self.agent_types)
                    line_chars.append("." if max_density > 0.1 else " ")

            lines.append("".join(line_chars))

        return "\n".join(lines)

    def _render_rgb_array(self) -> NDArray:
        """
        Render RGB array visualization with multi-population agents.

        Color scheme:
        - Walls: Black [0, 0, 0]
        - Empty space: White [255, 255, 255]
        - Population densities: Blended colors per type
        - Goals: Green [0, 255, 0]
        - Agents: Type-specific colors
          - Type 0: Red [255, 0, 0]
          - Type 1: Blue [0, 0, 255]
          - Type 2+: Orange, Purple, Cyan, etc.

        Returns:
            RGB array of shape (height, width, 3)
        """
        # Start with white background
        img = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255

        # Draw walls as black
        img[self.maze_array == 1] = [0, 0, 0]

        # Define colors for different agent types
        type_colors = {
            0: np.array([255, 0, 0], dtype=np.uint8),  # Red
            1: np.array([0, 0, 255], dtype=np.uint8),  # Blue
            2: np.array([255, 165, 0], dtype=np.uint8),  # Orange
            3: np.array([128, 0, 128], dtype=np.uint8),  # Purple
            4: np.array([0, 255, 255], dtype=np.uint8),  # Cyan
            5: np.array([255, 255, 0], dtype=np.uint8),  # Yellow
        }

        # Overlay population densities (blend colors)
        for type_id, type_config in self.agent_types.items():
            type_index = type_config.type_index
            density = self.multi_pop_state.distributions[type_id]

            # Get color for this type
            color = type_colors.get(type_index, np.array([128, 128, 128], dtype=np.uint8))

            # Normalize density to [0, 1]
            density_normalized = np.clip(density, 0, 1)

            # Blend density color with background (only on non-wall cells)
            for r in range(self.height):
                for c in range(self.width):
                    if self.maze_array[r, c] == 0 and density_normalized[r, c] > 0.01:
                        # Blend: img = (1-α)*white + α*color
                        alpha = density_normalized[r, c] * 0.5  # Scale for visibility
                        img[r, c] = (1 - alpha) * img[r, c] + alpha * color

        # Draw goal positions (green)
        for _type_id, type_config in self.agent_types.items():
            if type_config.goal_positions:
                for goal_pos in type_config.goal_positions:
                    img[tuple(goal_pos)] = [0, 255, 0]

        # Draw agents on top (use type-specific colors)
        for type_id, positions in self.positions.items():
            type_index = self.agent_types[type_id].type_index
            color = type_colors.get(type_index, np.array([128, 128, 128], dtype=np.uint8))

            for pos in positions:
                img[tuple(pos)] = color

        return img
