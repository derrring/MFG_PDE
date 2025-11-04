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

pytestmark = pytest.mark.environment


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
    """Test the Gymnasium-compatible MFG maze environment."""

    def setup_method(self) -> None:
        # Create simple predictable maze for testing (not randomly generated)
        # 1 = wall, 0 = free space
        self.maze_array = np.ones((10, 10), dtype=np.int32)
        self.maze_array[1:9, 1:9] = 0  # Open interior, walls on borders

        self.config = MFGMazeConfig(
            maze_array=self.maze_array,
            start_positions=[(1, 1)],
            goal_positions=[(8, 8)],
            action_type=ActionType.FOUR_CONNECTED,
            reward_type=RewardType.SPARSE,
            max_episode_steps=100,
            num_agents=1,
        )

        self.env = MFGMazeEnvironment(self.config)

    def test_initialization(self) -> None:
        assert self.env.maze.shape == self.maze_array.shape
        assert self.env.action_space.n == 4
        assert isinstance(self.env.observation_space, gym.spaces.Box)

    def test_reset(self) -> None:
        observation, info = self.env.reset(seed=42)
        assert observation.shape == self.env.observation_space.shape
        np.testing.assert_array_equal(info["positions"], np.array([[1, 1]], dtype=np.int32))
        np.testing.assert_array_equal(info["goals"], np.array([[8, 8]], dtype=np.int32))

    def test_step_valid_action(self) -> None:
        self.env.reset(seed=42)
        for action in range(self.env.action_space.n):
            _, rewards, terminated, _, _ = self.env.step(action)
            if not terminated:
                assert rewards.shape == (1,)
                assert self.env.current_step == 1
                break

    def test_step_goal_reached(self) -> None:
        simple_maze = np.ones((5, 5), dtype=np.int32)
        simple_maze[1:4, 1:4] = 0

        config = MFGMazeConfig(
            maze_array=simple_maze,
            start_positions=[(1, 1)],
            goal_positions=[(1, 2)],
            action_type=ActionType.FOUR_CONNECTED,
            reward_type=RewardType.SPARSE,  # Use sparse rewards for exact goal reward
            goal_reward=10.0,
            num_agents=1,
        )

        env = MFGMazeEnvironment(config)
        env.reset(seed=0)
        _, rewards, terminated, _, _ = env.step(3)

        assert terminated
        # Goal reward minus small move_cost and time_penalty
        assert rewards[0] == pytest.approx(10.0, abs=0.02)
        np.testing.assert_array_equal(env.agent_positions, np.array([[1, 2]], dtype=np.int32))

    def test_episode_truncation(self) -> None:
        simple_maze = np.ones((7, 7), dtype=np.int32)
        simple_maze[1:6, 1:6] = 0
        config = MFGMazeConfig(
            maze_array=simple_maze,
            start_positions=[(3, 3)],
            goal_positions=[(1, 1)],
            max_episode_steps=3,
            num_agents=1,
        )
        env = MFGMazeEnvironment(config)
        env.reset(seed=0)

        truncated = False
        for _ in range(5):
            _, _, terminated, truncated, _ = env.step(0)
            if terminated or truncated:
                break

        assert truncated or env.current_step >= 3

    def test_observation_space(self) -> None:
        observation, _ = self.env.reset(seed=7)
        assert observation.shape == self.env.observation_space.shape

    def test_population_in_observation(self) -> None:
        radius = 2
        config = MFGMazeConfig(
            maze_array=self.maze_array,
            start_positions=[(1, 1)],
            goal_positions=[(8, 8)],
            include_population_in_obs=True,
            population_obs_radius=radius,
            num_agents=1,
        )
        env = MFGMazeEnvironment(config)
        observation, _ = env.reset(seed=1)
        expected_dim = 5 + (2 * radius + 1) ** 2
        assert observation.shape == (expected_dim,)

    def test_reward_types(self) -> None:
        for reward_type in (RewardType.SPARSE, RewardType.DENSE, RewardType.CONGESTION):
            config = MFGMazeConfig(
                maze_array=self.maze_array,
                start_positions=[(1, 1)],
                goal_positions=[(8, 8)],
                reward_type=reward_type,
                num_agents=1,
            )
            env = MFGMazeEnvironment(config)
            env.reset(seed=5)
            _, rewards, _, _, _ = env.step(0)
            assert rewards.shape == (1,)

    def test_action_types(self) -> None:
        config4 = MFGMazeConfig(
            maze_array=self.maze_array,
            start_positions=[(1, 1)],
            goal_positions=[(8, 8)],
            action_type=ActionType.FOUR_CONNECTED,
            num_agents=1,
        )
        assert MFGMazeEnvironment(config4).action_space.n == 4

        config8 = MFGMazeConfig(
            maze_array=self.maze_array,
            start_positions=[(1, 1)],
            goal_positions=[(8, 8)],
            action_type=ActionType.EIGHT_CONNECTED,
            num_agents=1,
        )
        assert MFGMazeEnvironment(config8).action_space.n == 8

    def test_multi_agent_support(self) -> None:
        config = MFGMazeConfig(
            maze_array=self.maze_array,
            start_positions=[(1, 1), (1, 2), (1, 3)],
            goal_positions=[(8, 8), (8, 7), (8, 6)],
            num_agents=3,
        )
        env = MFGMazeEnvironment(config)
        observation, _info = env.reset(seed=9)
        assert observation.shape[0] == 3
        assert isinstance(env.action_space, gym.spaces.MultiDiscrete)
        _, rewards, _, _, _ = env.step(np.zeros(3, dtype=int))
        assert rewards.shape == (3,)

    def test_rendering(self) -> None:
        env_ascii = MFGMazeEnvironment(self.config, render_mode="human")
        env_ascii.reset(seed=11)
        assert env_ascii.render() is None

        env_rgb = MFGMazeEnvironment(self.config, render_mode="rgb_array")
        env_rgb.reset(seed=11)
        img = env_rgb.render()
        assert img.shape == (self.maze_array.shape[0], self.maze_array.shape[1], 3)

    def test_reproducibility(self) -> None:
        env1 = MFGMazeEnvironment(self.config)
        env2 = MFGMazeEnvironment(self.config)
        obs1, _ = env1.reset(seed=123)
        obs2, _ = env2.reset(seed=123)
        np.testing.assert_array_equal(obs1, obs2)


@pytest.mark.skipif(not GYMNASIUM_AVAILABLE, reason="Gymnasium not installed")
class TestIntegration:
    """Integration tests with different maze generators."""

    def test_perfect_maze(self) -> None:
        generator = PerfectMazeGenerator(15, 15, MazeAlgorithm.RECURSIVE_BACKTRACKING)
        generator.generate(seed=21)
        maze_array = generator.to_numpy_array()
        config = MFGMazeConfig(
            maze_array=maze_array,
            start_positions=[(1, 1)],
            goal_positions=[(13, 13)],
            num_agents=1,
        )
        env = MFGMazeEnvironment(config)
        env.reset(seed=21)
        for _ in range(10):
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        assert env.current_step <= 10

    def test_recursive_division_maze(self) -> None:
        config_rd = create_room_based_config(20, 20, room_size="medium", corridor_width="medium", seed=33)
        generator = RecursiveDivisionGenerator(config_rd)
        maze = generator.generate()
        config = MFGMazeConfig(
            maze_array=maze,
            start_positions=[(2, 2)],
            goal_positions=[(17, 17)],
            action_type=ActionType.EIGHT_CONNECTED,
            num_agents=1,
        )
        env = MFGMazeEnvironment(config)
        env.reset(seed=33)
        for _ in range(20):
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        assert env.current_step <= 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
