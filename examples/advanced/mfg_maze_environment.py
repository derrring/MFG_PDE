#!/usr/bin/env python3
"""
MFG Maze/Labyrinth Environment

This module implements a maze navigation environment for Mean Field Games RL experiments.
Agents must navigate through a maze while avoiding congestion, creating natural mean field
interactions through spatial bottlenecks and congestion costs.

Key Features:
- Procedural maze generation with configurable complexity
- Multiple agents navigating simultaneously
- Congestion-based mean field interactions
- Configurable start/goal positions
- Visualization capabilities
- Multiple maze types (random, structured, custom)

Environment Dynamics:
- Agents start at designated positions and navigate to goals
- Movement costs increase with local agent density (congestion)
- Walls block movement and create natural bottlenecks
- Rewards based on progress toward goal minus congestion penalty

Author: MFG_PDE Team
Date: October 2025
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from mfg_pde.utils.logging import get_logger

logger = get_logger(__name__)


class CellType(Enum):
    """Types of cells in the maze."""

    WALL = 0
    EMPTY = 1
    START = 2
    GOAL = 3
    AGENT = 4


@dataclass
class MazeConfig:
    """Configuration for maze environment."""

    # Maze dimensions
    width: int = 21  # Should be odd for proper maze generation
    height: int = 21  # Should be odd for proper maze generation

    # Agent parameters
    num_agents: int = 50
    max_episode_steps: int = 500

    # Maze generation
    maze_type: str = "random"  # random, empty, corridors, rooms, custom
    wall_density: float = 0.3  # For random mazes
    corridor_width: int = 1  # For structured mazes

    # Goal configuration
    goal_mode: str = "single"  # single, multiple, opposite_corners
    num_goals: int = 1

    # Rewards and costs
    goal_reward: float = 100.0
    step_cost: float = -0.1
    wall_collision_cost: float = -1.0
    congestion_penalty_weight: float = 2.0
    distance_reward_weight: float = 0.1

    # Congestion parameters
    congestion_radius: float = 1.5  # Radius for congestion calculation
    max_congestion_agents: int = 10  # Saturation point for congestion

    # Rendering
    cell_size: float = 1.0
    show_agent_trails: bool = False
    trail_length: int = 10


class MazeGenerator:
    """Generates various types of mazes for MFG experiments."""

    def __init__(self, config: MazeConfig):
        self.config = config
        self.logger = get_logger(__name__)

    def generate_maze(self) -> np.ndarray:
        """Generate maze based on configuration."""
        maze_type = self.config.maze_type.lower()

        if maze_type == "random":
            return self._generate_random_maze()
        elif maze_type == "empty":
            return self._generate_empty_maze()
        elif maze_type == "corridors":
            return self._generate_corridor_maze()
        elif maze_type == "rooms":
            return self._generate_room_maze()
        elif maze_type == "classic":
            return self._generate_classic_maze()
        else:
            self.logger.warning(f"Unknown maze type: {maze_type}, using random")
            return self._generate_random_maze()

    def _generate_random_maze(self) -> np.ndarray:
        """Generate random maze with specified wall density."""
        maze = np.ones((self.config.height, self.config.width), dtype=int)

        # Add random walls
        for i in range(1, self.config.height - 1):
            for j in range(1, self.config.width - 1):
                if random.random() < self.config.wall_density:
                    maze[i, j] = CellType.WALL.value

        # Ensure borders are walls
        maze[0, :] = CellType.WALL.value
        maze[-1, :] = CellType.WALL.value
        maze[:, 0] = CellType.WALL.value
        maze[:, -1] = CellType.WALL.value

        # Ensure connectivity by creating some guaranteed paths
        self._ensure_connectivity(maze)

        return maze

    def _generate_empty_maze(self) -> np.ndarray:
        """Generate empty maze (just walls around border)."""
        maze = np.ones((self.config.height, self.config.width), dtype=int)

        # Only border walls
        maze[0, :] = CellType.WALL.value
        maze[-1, :] = CellType.WALL.value
        maze[:, 0] = CellType.WALL.value
        maze[:, -1] = CellType.WALL.value

        return maze

    def _generate_corridor_maze(self) -> np.ndarray:
        """Generate maze with corridor structure."""
        maze = np.zeros((self.config.height, self.config.width), dtype=int)

        # Start with all walls
        maze.fill(CellType.WALL.value)

        # Create horizontal corridors
        for i in range(2, self.config.height - 2, 4):
            for j in range(1, self.config.width - 1):
                maze[i, j] = CellType.EMPTY.value

        # Create vertical corridors
        for j in range(2, self.config.width - 2, 4):
            for i in range(1, self.config.height - 1):
                maze[i, j] = CellType.EMPTY.value

        # Add some random openings
        for _ in range(self.config.width * self.config.height // 20):
            i = random.randint(1, self.config.height - 2)
            j = random.randint(1, self.config.width - 2)
            if random.random() < 0.3:
                maze[i, j] = CellType.EMPTY.value

        return maze

    def _generate_room_maze(self) -> np.ndarray:
        """Generate maze with room structure."""
        maze = np.zeros((self.config.height, self.config.width), dtype=int)
        maze.fill(CellType.WALL.value)

        # Create rooms
        room_size = 4
        for i in range(2, self.config.height - room_size, room_size + 1):
            for j in range(2, self.config.width - room_size, room_size + 1):
                # Clear room interior
                for di in range(room_size):
                    for dj in range(room_size):
                        if i + di < self.config.height - 1 and j + dj < self.config.width - 1:
                            maze[i + di, j + dj] = CellType.EMPTY.value

                # Add room connections
                # Horizontal connection
                if j + room_size + 1 < self.config.width - 1:
                    maze[i + room_size // 2, j + room_size] = CellType.EMPTY.value

                # Vertical connection
                if i + room_size + 1 < self.config.height - 1:
                    maze[i + room_size, j + room_size // 2] = CellType.EMPTY.value

        return maze

    def _generate_classic_maze(self) -> np.ndarray:
        """Generate classic maze using recursive backtracking."""
        # Initialize maze with all walls
        maze = np.zeros((self.config.height, self.config.width), dtype=int)
        maze.fill(CellType.WALL.value)

        # Recursive backtracking maze generation
        def carve_path(x, y):
            maze[y, x] = CellType.EMPTY.value

            # Randomize directions
            directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
            random.shuffle(directions)

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (
                    1 <= nx < self.config.width - 1
                    and 1 <= ny < self.config.height - 1
                    and maze[ny, nx] == CellType.WALL.value
                ):
                    # Carve path between current and next cell
                    maze[y + dy // 2, x + dx // 2] = CellType.EMPTY.value
                    carve_path(nx, ny)

        # Start carving from (1, 1)
        if self.config.width > 2 and self.config.height > 2:
            carve_path(1, 1)

        return maze

    def _ensure_connectivity(self, maze: np.ndarray):
        """Ensure maze has paths connecting different areas."""
        # Simple approach: clear some horizontal and vertical lines
        mid_row = self.config.height // 2
        mid_col = self.config.width // 2

        # Clear middle row and column
        for j in range(1, self.config.width - 1):
            if j % 3 == 0:  # Every third cell to maintain some walls
                maze[mid_row, j] = CellType.EMPTY.value

        for i in range(1, self.config.height - 1):
            if i % 3 == 0:
                maze[i, mid_col] = CellType.EMPTY.value


class MFGMazeEnvironment:
    """
    Mean Field Games Maze Environment.

    Agents navigate through a maze while experiencing congestion costs based on
    local agent density. This creates natural mean field interactions through
    spatial bottlenecks.
    """

    def __init__(self, config: MazeConfig):
        self.config = config
        self.logger = get_logger(__name__)

        # Generate maze
        self.maze_generator = MazeGenerator(config)
        self.maze = self.maze_generator.generate_maze()

        # Environment state
        self.agent_positions = np.zeros((config.num_agents, 2), dtype=int)
        self.agent_goals = np.zeros((config.num_agents, 2), dtype=int)
        self.agent_trails = [[] for _ in range(config.num_agents)]

        # Episode tracking
        self.current_step = 0
        self.episode_rewards = []
        self.agents_reached_goal = np.zeros(config.num_agents, dtype=bool)

        # Initialize start and goal positions
        self._setup_start_goal_positions()

        # Visualization
        self.fig = None
        self.ax = None

        self.logger.info(f"Created MFG Maze Environment: {config.width}x{config.height}, {config.num_agents} agents")

    def _setup_start_goal_positions(self):
        """Setup start and goal positions based on configuration."""
        empty_cells = self._get_empty_cells()

        if len(empty_cells) < self.config.num_agents + self.config.num_goals:
            raise ValueError("Not enough empty cells for agents and goals")

        # Setup goals
        goal_positions = self._setup_goals(empty_cells)

        # Setup agent start positions and assign goals
        self._setup_agent_starts(empty_cells, goal_positions)

    def _get_empty_cells(self) -> list[tuple[int, int]]:
        """Get list of empty cell coordinates."""
        empty_cells = []
        for i in range(self.config.height):
            for j in range(self.config.width):
                if self.maze[i, j] == CellType.EMPTY.value:
                    empty_cells.append((i, j))
        return empty_cells

    def _setup_goals(self, empty_cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Setup goal positions."""
        goal_mode = self.config.goal_mode.lower()

        if goal_mode == "single":
            # Single goal at bottom-right corner (or nearest empty cell)
            goal_candidates = [
                (i, j) for i, j in empty_cells if i > self.config.height * 0.7 and j > self.config.width * 0.7
            ]
            if not goal_candidates:
                goal_candidates = empty_cells[-10:]  # Last 10 empty cells
            goal_positions = [random.choice(goal_candidates)]

        elif goal_mode == "multiple":
            # Multiple goals spread across the maze
            goal_positions = random.sample(empty_cells, min(self.config.num_goals, len(empty_cells) // 2))

        elif goal_mode == "opposite_corners":
            # Goals in opposite corners
            corner_candidates = [
                [
                    (i, j) for i, j in empty_cells if i < self.config.height * 0.3 and j > self.config.width * 0.7
                ],  # Top-right
                [
                    (i, j) for i, j in empty_cells if i > self.config.height * 0.7 and j < self.config.width * 0.3
                ],  # Bottom-left
                [
                    (i, j) for i, j in empty_cells if i > self.config.height * 0.7 and j > self.config.width * 0.7
                ],  # Bottom-right
            ]

            goal_positions = []
            for candidates in corner_candidates:
                if candidates and len(goal_positions) < self.config.num_goals:
                    goal_positions.append(random.choice(candidates))

        else:
            # Default to single goal
            goal_positions = [random.choice(empty_cells)]

        # Mark goals in maze
        for goal_pos in goal_positions:
            self.maze[goal_pos] = CellType.GOAL.value

        return goal_positions

    def _setup_agent_starts(self, empty_cells: list[tuple[int, int]], goal_positions: list[tuple[int, int]]):
        """Setup agent starting positions and assign goals."""
        # Filter out goal positions from potential start positions
        goal_set = set(goal_positions)
        start_candidates = [pos for pos in empty_cells if pos not in goal_set]

        if len(start_candidates) < self.config.num_agents:
            raise ValueError("Not enough empty cells for agent starting positions")

        # Prefer starting positions in top-left area
        preferred_starts = [
            (i, j) for i, j in start_candidates if i < self.config.height * 0.3 and j < self.config.width * 0.3
        ]

        if len(preferred_starts) >= self.config.num_agents:
            start_positions = random.sample(preferred_starts, self.config.num_agents)
        else:
            # Use preferred starts and fill with random positions
            remaining_needed = self.config.num_agents - len(preferred_starts)
            other_starts = [pos for pos in start_candidates if pos not in preferred_starts]
            additional_starts = random.sample(other_starts, min(remaining_needed, len(other_starts)))
            start_positions = preferred_starts + additional_starts

            # If still not enough, allow some overlap
            while len(start_positions) < self.config.num_agents:
                start_positions.append(random.choice(start_candidates))

        # Assign start positions to agents
        for i, start_pos in enumerate(start_positions):
            self.agent_positions[i] = [start_pos[0], start_pos[1]]

        # Assign goals to agents
        for i in range(self.config.num_agents):
            if self.config.goal_mode == "single":
                # All agents go to the same goal
                self.agent_goals[i] = [goal_positions[0][0], goal_positions[0][1]]
            else:
                # Assign nearest goal or random goal
                agent_pos = self.agent_positions[i]
                goal_distances = [abs(agent_pos[0] - gp[0]) + abs(agent_pos[1] - gp[1]) for gp in goal_positions]
                nearest_goal_idx = np.argmin(goal_distances)
                self.agent_goals[i] = [goal_positions[nearest_goal_idx][0], goal_positions[nearest_goal_idx][1]]

    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        # Regenerate maze if needed
        if hasattr(self.config, "regenerate_maze_each_episode") and self.config.regenerate_maze_each_episode:
            self.maze = self.maze_generator.generate_maze()
            self._setup_start_goal_positions()
        else:
            # Reset agent positions to start
            self._setup_start_goal_positions()

        # Reset episode state
        self.current_step = 0
        self.agents_reached_goal.fill(False)
        self.agent_trails = [[] for _ in range(self.config.num_agents)]

        return self._get_observations()

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool, dict]:
        """
        Execute environment step.

        Args:
            actions: Movement actions for each agent (0=stay, 1=up, 2=down, 3=left, 4=right)

        Returns:
            observations, rewards, done, info
        """
        # Store previous positions for trail tracking
        if self.config.show_agent_trails:
            for i in range(self.config.num_agents):
                self.agent_trails[i].append(tuple(self.agent_positions[i]))
                if len(self.agent_trails[i]) > self.config.trail_length:
                    self.agent_trails[i].pop(0)

        # Execute actions and compute rewards
        rewards = self._execute_actions(actions)

        # Update step counter
        self.current_step += 1

        # Check termination conditions
        done = self._check_termination()

        # Get new observations
        observations = self._get_observations()

        # Additional info
        info = {
            "step": self.current_step,
            "agents_at_goal": np.sum(self.agents_reached_goal),
            "congestion_map": self._compute_congestion_map(),
            "mean_distance_to_goal": self._compute_mean_distance_to_goal(),
        }

        return observations, rewards, done, info

    def _execute_actions(self, actions: np.ndarray) -> np.ndarray:
        """Execute agent actions and compute rewards."""
        rewards = np.zeros(self.config.num_agents)

        # Define action mappings (0=stay, 1=up, 2=down, 3=left, 4=right)
        action_map = {
            0: (0, 0),  # stay
            1: (-1, 0),  # up
            2: (1, 0),  # down
            3: (0, -1),  # left
            4: (0, 1),  # right
        }

        for i in range(self.config.num_agents):
            if self.agents_reached_goal[i]:
                # Agent already reached goal, no movement or cost
                rewards[i] = 0
                continue

            # Get current position and action
            current_pos = self.agent_positions[i].copy()
            action = int(actions[i]) if actions[i] < len(action_map) else 0
            delta = action_map[action]

            # Calculate new position
            new_pos = current_pos + np.array(delta)

            # Check bounds and walls
            if (
                0 <= new_pos[0] < self.config.height
                and 0 <= new_pos[1] < self.config.width
                and self.maze[new_pos[0], new_pos[1]] != CellType.WALL.value
            ):
                # Valid move
                self.agent_positions[i] = new_pos
                move_cost = 0
            else:
                # Invalid move (wall collision or out of bounds)
                move_cost = self.config.wall_collision_cost

            # Compute reward components
            reward = 0

            # Step cost
            reward += self.config.step_cost

            # Movement cost (wall collision)
            reward += move_cost

            # Distance-based reward
            old_distance = self._manhattan_distance(current_pos, self.agent_goals[i])
            new_distance = self._manhattan_distance(self.agent_positions[i], self.agent_goals[i])
            distance_improvement = old_distance - new_distance
            reward += distance_improvement * self.config.distance_reward_weight

            # Congestion penalty
            congestion = self._compute_local_congestion(self.agent_positions[i])
            congestion_penalty = congestion * self.config.congestion_penalty_weight
            reward -= congestion_penalty

            # Goal reached reward
            if (
                self.agent_positions[i][0] == self.agent_goals[i][0]
                and self.agent_positions[i][1] == self.agent_goals[i][1]
            ):
                if not self.agents_reached_goal[i]:
                    reward += self.config.goal_reward
                    self.agents_reached_goal[i] = True

            rewards[i] = reward

        return rewards

    def _compute_local_congestion(self, position: np.ndarray) -> float:
        """Compute local congestion around a position."""
        congestion = 0

        for j in range(self.config.num_agents):
            other_pos = self.agent_positions[j]
            distance = np.linalg.norm(position - other_pos)

            if distance <= self.config.congestion_radius and distance > 0:
                # Weight by inverse distance
                congestion += 1.0 / (distance + 0.1)

        # Normalize by maximum possible congestion
        normalized_congestion = min(congestion / self.config.max_congestion_agents, 1.0)
        return normalized_congestion

    def _compute_congestion_map(self) -> np.ndarray:
        """Compute congestion map for the entire maze."""
        congestion_map = np.zeros((self.config.height, self.config.width))

        for i in range(self.config.height):
            for j in range(self.config.width):
                if self.maze[i, j] != CellType.WALL.value:
                    congestion_map[i, j] = self._compute_local_congestion(np.array([i, j]))

        return congestion_map

    def _manhattan_distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """Compute Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _compute_mean_distance_to_goal(self) -> float:
        """Compute mean distance of all agents to their goals."""
        distances = []
        for i in range(self.config.num_agents):
            if not self.agents_reached_goal[i]:
                distance = self._manhattan_distance(self.agent_positions[i], self.agent_goals[i])
                distances.append(distance)

        return np.mean(distances) if distances else 0.0

    def _get_observations(self) -> np.ndarray:
        """Get observations for all agents."""
        observations = np.zeros((self.config.num_agents, 8))

        for i in range(self.config.num_agents):
            obs = observations[i]

            # Agent position (normalized)
            obs[0] = self.agent_positions[i][0] / self.config.height
            obs[1] = self.agent_positions[i][1] / self.config.width

            # Goal position (normalized)
            obs[2] = self.agent_goals[i][0] / self.config.height
            obs[3] = self.agent_goals[i][1] / self.config.width

            # Distance to goal (normalized)
            distance = self._manhattan_distance(self.agent_positions[i], self.agent_goals[i])
            max_distance = self.config.height + self.config.width
            obs[4] = distance / max_distance

            # Local congestion
            obs[5] = self._compute_local_congestion(self.agent_positions[i])

            # Goal reached flag
            obs[6] = float(self.agents_reached_goal[i])

            # Progress (1 - normalized distance to goal)
            obs[7] = 1.0 - obs[4]

        return observations

    def _check_termination(self) -> bool:
        """Check if episode should terminate."""
        # Time limit reached
        if self.current_step >= self.config.max_episode_steps:
            return True

        # All agents reached their goals
        return bool(np.all(self.agents_reached_goal))

    def render(
        self, mode: str = "human", show_congestion: bool = True, show_trails: bool | None = None
    ) -> np.ndarray | None:
        """
        Render the maze environment.

        Args:
            mode: Rendering mode ("human" for display, "rgb_array" for image)
            show_congestion: Whether to show congestion heatmap
            show_trails: Whether to show agent trails (overrides config)
        """
        if show_trails is None:
            show_trails = self.config.show_agent_trails

        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 10))

        self.ax.clear()

        # Create visualization array
        vis_array = self.maze.copy().astype(float)

        # Add congestion heatmap if requested
        if show_congestion:
            congestion_map = self._compute_congestion_map()
            # Overlay congestion on empty cells
            mask = vis_array == CellType.EMPTY.value
            vis_array[mask] = 1 + congestion_map[mask]  # Offset to distinguish from empty

        # Create colormap
        colors = ["black", "white", "lightblue", "green", "red", "orange"]
        if show_congestion:
            # Add gradient for congestion
            colors = ["black", "white", "yellow", "orange", "red", "darkred", "green", "blue"]

        cmap = ListedColormap(colors[: int(np.max(vis_array)) + 1])

        # Display maze
        self.ax.imshow(vis_array, cmap=cmap, interpolation="nearest")

        # Add agent trails
        if show_trails:
            for i, trail in enumerate(self.agent_trails):
                if len(trail) > 1:
                    trail_array = np.array(trail)
                    alpha_values = np.linspace(0.2, 0.8, len(trail))
                    for j in range(len(trail) - 1):
                        self.ax.plot(
                            [trail_array[j, 1], trail_array[j + 1, 1]],
                            [trail_array[j, 0], trail_array[j + 1, 0]],
                            "b-",
                            alpha=alpha_values[j],
                            linewidth=1,
                        )

        # Add agents
        for i in range(self.config.num_agents):
            pos = self.agent_positions[i]
            color = "blue" if not self.agents_reached_goal[i] else "lightgreen"
            circle = patches.Circle((pos[1], pos[0]), 0.3, color=color, alpha=0.8)
            self.ax.add_patch(circle)

            # Add agent ID for small numbers of agents
            if self.config.num_agents <= 20:
                self.ax.text(pos[1], pos[0], str(i), ha="center", va="center", fontsize=8, fontweight="bold")

        # Add goal markers
        goal_positions = set()
        for i in range(self.config.num_agents):
            goal_pos = tuple(self.agent_goals[i])
            goal_positions.add(goal_pos)

        for goal_pos in goal_positions:
            star = patches.RegularPolygon((goal_pos[1], goal_pos[0]), 5, 0.4, color="gold", alpha=0.9)
            self.ax.add_patch(star)

        # Add title and info
        agents_at_goal = np.sum(self.agents_reached_goal)
        mean_distance = self._compute_mean_distance_to_goal()

        title = f"MFG Maze - Step {self.current_step} | "
        title += f"Agents at Goal: {agents_at_goal}/{self.config.num_agents} | "
        title += f"Mean Distance: {mean_distance:.1f}"

        self.ax.set_title(title)
        self.ax.set_xlim(-0.5, self.config.width - 0.5)
        self.ax.set_ylim(-0.5, self.config.height - 0.5)
        self.ax.set_aspect("equal")

        # Remove axis ticks
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        if mode == "human":
            plt.pause(0.1)
            return None
        elif mode == "rgb_array":
            self.fig.canvas.draw()
            buf = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape((*self.fig.canvas.get_width_height()[::-1], 3))
            return buf

    def close(self):
        """Close rendering."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None

    def get_maze_info(self) -> dict:
        """Get information about the maze structure."""
        empty_cells = len(self._get_empty_cells())
        total_cells = self.config.height * self.config.width
        wall_density = 1.0 - (empty_cells / total_cells)

        return {
            "dimensions": (self.config.height, self.config.width),
            "total_cells": total_cells,
            "empty_cells": empty_cells,
            "wall_density": wall_density,
            "maze_type": self.config.maze_type,
            "num_agents": self.config.num_agents,
            "goal_mode": self.config.goal_mode,
        }


def create_maze_config(maze_type: str = "random", size: str = "medium", difficulty: str = "medium") -> MazeConfig:
    """
    Factory function to create maze configurations.

    Args:
        maze_type: Type of maze ("random", "empty", "corridors", "rooms", "classic")
        size: Size of maze ("small", "medium", "large")
        difficulty: Difficulty level ("easy", "medium", "hard")

    Returns:
        Configured MazeConfig
    """
    # Size configurations
    size_configs = {
        "small": {"width": 15, "height": 15, "num_agents": 20},
        "medium": {"width": 21, "height": 21, "num_agents": 50},
        "large": {"width": 31, "height": 31, "num_agents": 100},
    }

    # Difficulty configurations
    difficulty_configs = {
        "easy": {
            "wall_density": 0.2,
            "congestion_penalty_weight": 1.0,
            "max_episode_steps": 300,
        },
        "medium": {
            "wall_density": 0.3,
            "congestion_penalty_weight": 2.0,
            "max_episode_steps": 500,
        },
        "hard": {
            "wall_density": 0.4,
            "congestion_penalty_weight": 3.0,
            "max_episode_steps": 700,
        },
    }

    # Base configuration
    config = MazeConfig(maze_type=maze_type)

    # Apply size settings
    if size in size_configs:
        for key, value in size_configs[size].items():
            setattr(config, key, value)

    # Apply difficulty settings
    if difficulty in difficulty_configs:
        for key, value in difficulty_configs[difficulty].items():
            setattr(config, key, value)

    return config


def demo_maze_environment():
    """Demonstrate the maze environment with various configurations."""
    print("🎮 MFG Maze Environment Demo")
    print("=" * 40)

    # Test different maze types
    maze_types = ["empty", "random", "corridors", "classic"]

    for maze_type in maze_types:
        print(f"\n🏗️  Testing {maze_type} maze...")

        # Create configuration
        config = create_maze_config(maze_type=maze_type, size="small", difficulty="easy")

        # Create environment
        env = MFGMazeEnvironment(config)

        # Print maze info
        info = env.get_maze_info()
        print(f"   Dimensions: {info['dimensions']}")
        print(f"   Wall density: {info['wall_density']:.2f}")
        print(f"   Agents: {info['num_agents']}")

        # Run a few steps with random actions
        _obs = env.reset()
        total_reward = 0

        for step in range(20):
            # Random actions
            actions = np.random.randint(0, 5, size=config.num_agents)
            _obs, rewards, done, info = env.step(actions)

            step_reward = np.mean(rewards)
            total_reward += step_reward

            agents_at_goal = info["agents_at_goal"]
            mean_distance = info["mean_distance_to_goal"]

            if step % 5 == 0:
                print(
                    f"   Step {step}: Reward={step_reward:.3f}, "
                    f"At Goal={agents_at_goal}, Distance={mean_distance:.1f}"
                )

            if done:
                print(f"   Episode completed at step {step}")
                break

        print(f"   Total reward: {total_reward:.3f}")

        # Render final state (comment out for automated testing)
        # env.render()
        # plt.show()

        env.close()


if __name__ == "__main__":
    demo_maze_environment()
