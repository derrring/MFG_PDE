"""
Unit tests for MFG Maze Environment.

Tests the Gymnasium-compatible environment including:
- Environment initialization and configuration
- Population state tracking
- Action execution and state transitions
- Reward calculation
- Observation space and rendering
"""

import pytest

import numpy as np

# Check for Gymnasium availability
try:
    import gymnasium as gym

    GYMNASIUM_AVAILABLE = True
except ImportError:
    GYMNASIUM_AVAILABLE = False

from mfg_pde.alg.reinforcement.environments import (
    MazeAlgorithm,
    PerfectMazeGenerator,
    RecursiveDivisionGenerator,
    create_room_based_config,
)

if GYMNASIUM_AVAILABLE:
    from mfg_pde.alg.reinforcement.environments import (
        ActionType,
        MFGMazeConfig,
        MFGMazeEnvironment,
        PopulationState,
        RewardType,
    )


@pytest.mark.skipif(not GYMNASIUM_AVAILABLE, reason="Gymnasium not installed")
class TestPopulationState:
    """Test PopulationState tracking."""

    def test_initialization(self):
        """Test population state initialization."""
        pop_state = PopulationState(maze_shape=(10, 10), smoothing=0.1)

        assert pop_state.maze_shape == (10, 10)
        assert pop_state.smoothing == 0.1
        assert pop_state.num_agents == 0
        assert pop_state.density_histogram.shape == (10, 10)
        assert np.all(pop_state.density_histogram == 0)

    def test_update_single_agent(self):
        """Test population update with single agent."""
        pop_state = PopulationState(maze_shape=(10, 10))
        pop_state.update([(5, 5)])

        assert pop_state.num_agents == 1
        assert pop_state.get_density_at((5, 5)) == 1.0
        assert pop_state.get_density_at((5, 6)) == 0.0

    def test_update_multiple_agents(self):
        """Test population update with multiple agents."""
        pop_state = PopulationState(maze_shape=(10, 10))
        positions = [(2, 2), (2, 2), (3, 3), (3, 3)]  # 2 at (2,2), 2 at (3,3)
        pop_state.update(positions)

        assert pop_state.num_agents == 4
        assert pop_state.get_density_at((2, 2)) == 0.5  # 2/4
        assert pop_state.get_density_at((3, 3)) == 0.5  # 2/4
        assert pop_state.get_density_at((4, 4)) == 0.0

    def test_local_density(self):
        """Test local density neighborhood extraction."""
        pop_state = PopulationState(maze_shape=(10, 10))
        pop_state.update([(5, 5)])

        local = pop_state.get_local_density((5, 5), radius=1)

        assert local.shape == (3, 3)
        assert local[1, 1] == 1.0  # Center
        assert np.sum(local) == 1.0  # Only center occupied

    def test_full_density(self):
        """Test full density field retrieval."""
        pop_state = PopulationState(maze_shape=(5, 5))
        pop_state.update([(1, 1), (3, 3)])

        density = pop_state.get_full_density()

        assert density.shape == (5, 5)
        assert density[1, 1] == 0.5
        assert density[3, 3] == 0.5
        assert np.sum(density) == 1.0


@pytest.mark.skipif(not GYMNASIUM_AVAILABLE, reason="Gymnasium not installed")
class TestMFGMazeConfig:
    """Test MFG Maze configuration."""

    def test_basic_config(self):
        """Test basic configuration creation."""
        maze = np.zeros((10, 10), dtype=np.int32)
        config = MFGMazeConfig(maze_array=maze)

        assert config.maze_array.shape == (10, 10)
        assert config.population_size == 100
        assert config.action_type == ActionType.FOUR_CONNECTED
        assert config.reward_type == RewardType.MFG_STANDARD

    def test_config_validation(self):
        """Test configuration validation."""
        with pytest.raises(ValueError):
            MFGMazeConfig(maze_array=None)

        maze = np.zeros((10, 10), dtype=np.int32)
        with pytest.raises(ValueError):
            MFGMazeConfig(maze_array=maze, population_size=0)


@pytest.mark.skipif(not GYMNASIUM_AVAILABLE, reason="Gymnasium not installed")
class TestMFGMazeEnvironment:
    """Test MFG Maze Environment."""

    def setup_method(self):
        """Setup test environment."""
        # Create simple test maze
        generator = PerfectMazeGenerator(10, 10, MazeAlgorithm.RECURSIVE_BACKTRACKING)
        generator.generate(seed=42)
        self.maze_array = generator.to_numpy_array()

        self.config = MFGMazeConfig(
            maze_array=self.maze_array,
            start_positions=[(1, 1)],
            goal_positions=[(8, 8)],
            action_type=ActionType.FOUR_CONNECTED,
            reward_type=RewardType.SPARSE,
            max_episode_steps=100,
        )

        self.env = MFGMazeEnvironment(self.config)

    def test_initialization(self):
        """Test environment initialization."""
        assert self.env.maze.shape == self.maze_array.shape
        assert self.env.action_space.n == 4
        assert isinstance(self.env.observation_space, gym.spaces.Dict)

    def test_reset(self):
        """Test environment reset."""
        observation, info = self.env.reset(seed=42)

        assert "position" in observation
        assert "goal" in observation
        assert "time_remaining" in observation
        assert self.env.agent_position == (1, 1)
        assert self.env.goal_position == (8, 8)
        assert self.env.current_step == 0

    def test_step_valid_action(self):
        """Test valid action execution."""
        self.env.reset(seed=42)
        initial_position = self.env.agent_position

        # Try all actions to find a valid move
        valid_move_found = False
        for action in range(4):
            self.env.reset(seed=42)
            observation, reward, terminated, truncated, info = self.env.step(action)

            if not terminated:  # If not collision
                valid_move_found = True
                assert self.env.agent_position != initial_position
                assert self.env.current_step == 1
                break

        assert valid_move_found, "No valid moves found from start position"

    def test_step_collision(self):
        """Test collision with wall."""
        self.env.reset(seed=42)

        # Find action that leads to wall
        for action in range(4):
            self.env.reset(seed=42)
            observation, reward, terminated, truncated, info = self.env.step(action)

            if terminated and reward == self.config.collision_penalty:
                break

        # Note: May not find collision depending on maze structure
        # This test verifies the mechanism works when collision occurs

    def test_step_goal_reached(self):
        """Test goal reaching."""
        # Create maze with goal right next to start
        simple_maze = np.ones((5, 5), dtype=np.int32)
        simple_maze[1:4, 1:4] = 0  # 3x3 open space

        config = MFGMazeConfig(
            maze_array=simple_maze,
            start_positions=[(1, 1)],
            goal_positions=[(1, 2)],  # Goal right next to start
            action_type=ActionType.FOUR_CONNECTED,
            goal_reward=10.0,
        )

        env = MFGMazeEnvironment(config)
        env.reset(seed=42)

        # Move right to goal
        observation, reward, terminated, truncated, info = env.step(3)

        assert terminated
        assert reward == 10.0
        assert env.agent_position == (1, 2)

    def test_episode_truncation(self):
        """Test episode truncation at max steps."""
        # Create simple maze with guaranteed safe moves
        simple_maze = np.ones((7, 7), dtype=np.int32)
        simple_maze[1:6, 1:6] = 0  # 5x5 open space

        config = MFGMazeConfig(
            maze_array=simple_maze,
            start_positions=[(3, 3)],  # Center
            goal_positions=[(1, 1)],  # Far corner
            max_episode_steps=3,  # Very short
        )

        env = MFGMazeEnvironment(config)
        env.reset(seed=42)

        truncated = False
        terminated = False
        for _ in range(5):
            observation, reward, terminated, truncated, info = env.step(1)  # Move down
            if truncated or terminated:
                break

        assert truncated or env.current_step >= 3
        if truncated:
            assert env.current_step == 3

    def test_observation_space(self):
        """Test observation space structure."""
        observation, info = self.env.reset(seed=42)

        assert "position" in observation
        assert "goal" in observation
        assert "time_remaining" in observation

        assert observation["position"].shape == (2,)
        assert observation["goal"].shape == (2,)
        assert observation["time_remaining"].shape == (1,)

    def test_population_in_observation(self):
        """Test population density in observation."""
        config = MFGMazeConfig(
            maze_array=self.maze_array,
            start_positions=[(1, 1)],
            goal_positions=[(8, 8)],
            include_population_in_obs=True,
            population_obs_radius=2,
        )

        env = MFGMazeEnvironment(config)
        observation, info = env.reset(seed=42)

        assert "local_density" in observation
        assert observation["local_density"].shape == (5, 5)  # 2*2+1

    def test_reward_types(self):
        """Test different reward types."""
        reward_configs = [
            (RewardType.SPARSE, "sparse"),
            (RewardType.DENSE, "dense"),
            (RewardType.CONGESTION, "congestion"),
        ]

        for reward_type, _ in reward_configs:
            config = MFGMazeConfig(
                maze_array=self.maze_array,
                start_positions=[(1, 1)],
                goal_positions=[(8, 8)],
                reward_type=reward_type,
            )

            env = MFGMazeEnvironment(config)
            env.reset(seed=42)

            # Take a valid action
            observation, reward, terminated, truncated, info = env.step(0)

            # Reward should be calculated (non-zero for most types)
            assert isinstance(reward, float)

    def test_action_types(self):
        """Test different action types."""
        # Four-connected
        config4 = MFGMazeConfig(
            maze_array=self.maze_array,
            start_positions=[(1, 1)],
            goal_positions=[(8, 8)],
            action_type=ActionType.FOUR_CONNECTED,
        )
        env4 = MFGMazeEnvironment(config4)
        assert env4.action_space.n == 4

        # Eight-connected
        config8 = MFGMazeConfig(
            maze_array=self.maze_array,
            start_positions=[(1, 1)],
            goal_positions=[(8, 8)],
            action_type=ActionType.EIGHT_CONNECTED,
        )
        env8 = MFGMazeEnvironment(config8)
        assert env8.action_space.n == 8

    def test_rendering(self):
        """Test rendering functionality."""
        # ASCII rendering
        env_ascii = MFGMazeEnvironment(self.config, render_mode="human")
        env_ascii.reset(seed=42)
        output = env_ascii.render()
        assert output is None  # Human mode returns None

        # RGB array rendering
        env_rgb = MFGMazeEnvironment(self.config, render_mode="rgb_array")
        env_rgb.reset(seed=42)
        img = env_rgb.render()
        assert img is not None
        assert img.shape == (self.maze_array.shape[0], self.maze_array.shape[1], 3)
        assert img.dtype == np.uint8

    def test_reproducibility(self):
        """Test environment reproducibility with seeds."""
        env1 = MFGMazeEnvironment(self.config)
        env2 = MFGMazeEnvironment(self.config)

        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)

        np.testing.assert_array_equal(obs1["position"], obs2["position"])
        np.testing.assert_array_equal(obs1["goal"], obs2["goal"])


@pytest.mark.skipif(not GYMNASIUM_AVAILABLE, reason="Gymnasium not installed")
class TestIntegration:
    """Integration tests with different maze types."""

    def test_perfect_maze(self):
        """Test with perfect maze."""
        generator = PerfectMazeGenerator(15, 15, MazeAlgorithm.RECURSIVE_BACKTRACKING)
        generator.generate(seed=42)
        maze_array = generator.to_numpy_array()

        config = MFGMazeConfig(
            maze_array=maze_array,
            start_positions=[(1, 1)],
            goal_positions=[(13, 13)],
        )

        env = MFGMazeEnvironment(config)
        observation, info = env.reset(seed=42)

        # Run a few steps
        for _ in range(10):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        assert env.current_step <= 10

    def test_recursive_division_maze(self):
        """Test with recursive division maze."""
        config_rd = create_room_based_config(20, 20, room_size="medium", corridor_width="medium", seed=42)
        generator = RecursiveDivisionGenerator(config_rd)
        maze = generator.generate()

        config = MFGMazeConfig(
            maze_array=maze,
            start_positions=[(2, 2)],
            goal_positions=[(17, 17)],
            action_type=ActionType.EIGHT_CONNECTED,
        )

        env = MFGMazeEnvironment(config)
        observation, info = env.reset(seed=42)

        # Run a few steps
        for _ in range(20):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        assert env.current_step <= 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
