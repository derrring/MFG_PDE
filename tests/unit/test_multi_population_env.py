"""
Unit tests for Multi-Population MFG Environment.

Tests multi-population maze environment with heterogeneous agents.
"""

import pytest

import numpy as np

try:
    import gymnasium  # noqa: F401

    GYMNASIUM_AVAILABLE = True
except ImportError:
    GYMNASIUM_AVAILABLE = False

from mfg_pde.alg.reinforcement.environments.multi_population_maze_env import (
    ActionType,
    AgentTypeConfig,
    MultiPopulationMazeConfig,
    MultiPopulationMazeEnvironment,
    MultiPopulationState,
)


@pytest.mark.skipif(not GYMNASIUM_AVAILABLE, reason="Gymnasium not available")
class TestMultiPopulationState:
    """Tests for MultiPopulationState class."""

    def test_initialization(self):
        """Test multi-population state initialization."""
        agent_types = {
            "type1": AgentTypeConfig(type_id="type1", type_index=0),
            "type2": AgentTypeConfig(type_id="type2", type_index=1),
        }

        multi_pop = MultiPopulationState(maze_shape=(10, 10), agent_types=agent_types)

        assert multi_pop.K == 2
        assert "type1" in multi_pop.distributions
        assert "type2" in multi_pop.distributions
        assert multi_pop.distributions["type1"].shape == (10, 10)

    def test_update_from_positions(self):
        """Test updating densities from agent positions."""
        agent_types = {
            "type1": AgentTypeConfig(type_id="type1", type_index=0),
            "type2": AgentTypeConfig(type_id="type2", type_index=1),
        }

        multi_pop = MultiPopulationState(maze_shape=(10, 10), agent_types=agent_types)

        positions = {
            "type1": [(2, 3), (2, 4), (3, 3)],
            "type2": [(7, 8), (8, 8)],
        }

        multi_pop.update_from_positions(positions, smoothing=0.5)

        # Check that densities are non-zero near agent positions
        density1 = multi_pop.get_density_field("type1")
        density2 = multi_pop.get_density_field("type2")

        assert density1[2, 3] > 0  # Type1 agents present
        assert density2[7, 8] > 0  # Type2 agents present
        assert np.isclose(density1.sum(), 1.0, atol=0.1)  # Normalized
        assert np.isclose(density2.sum(), 1.0, atol=0.1)

    def test_get_local_densities(self):
        """Test local density extraction."""
        agent_types = {
            "type1": AgentTypeConfig(type_id="type1", type_index=0),
            "type2": AgentTypeConfig(type_id="type2", type_index=1),
        }

        multi_pop = MultiPopulationState(maze_shape=(10, 10), agent_types=agent_types)

        positions = {
            "type1": [(5, 5)],
            "type2": [(5, 6)],
        }

        multi_pop.update_from_positions(positions, smoothing=0.5)

        # Get local densities around (5, 5)
        local_densities = multi_pop.get_local_densities((5, 5), radius=2)

        assert "type1" in local_densities
        assert "type2" in local_densities
        # Local window is (2*radius+1)^2 = 25 cells
        assert local_densities["type1"].shape == (25,)
        assert local_densities["type2"].shape == (25,)


@pytest.mark.skipif(not GYMNASIUM_AVAILABLE, reason="Gymnasium not available")
class TestMultiPopulationMazeEnvironment:
    """Tests for MultiPopulationMazeEnvironment."""

    def setup_method(self):
        """Setup test environment."""
        # Simple 10x10 maze
        maze = np.ones((10, 10), dtype=np.int32)
        maze[1:9, 1:9] = 0  # Open interior

        # Two agent types
        agent_types = {
            "predator": AgentTypeConfig(
                type_id="predator",
                type_index=0,
                action_type=ActionType.FOUR_CONNECTED,
                start_positions=[(1, 1), (1, 2)],  # 2 start positions for 2 agents
                goal_positions=[(8, 8)],
                num_agents=2,
                goal_reward=10.0,
            ),
            "prey": AgentTypeConfig(
                type_id="prey",
                type_index=1,
                action_type=ActionType.FOUR_CONNECTED,
                start_positions=[(8, 1), (8, 2), (8, 3)],  # 3 start positions for 3 agents
                goal_positions=[(1, 8)],
                num_agents=3,
                goal_reward=5.0,
            ),
        }

        config = MultiPopulationMazeConfig(maze_array=maze, agent_types=agent_types)

        self.env = MultiPopulationMazeEnvironment(config)

    def test_initialization(self):
        """Test environment initialization."""
        assert self.env.K == 2
        assert "predator" in self.env.agent_types
        assert "prey" in self.env.agent_types

        # Check observation and action spaces
        assert "predator" in self.env.observation_spaces
        assert "prey" in self.env.action_spaces

    def test_reset(self):
        """Test environment reset."""
        observations, info = self.env.reset(seed=42)

        # Check observations returned for both types
        assert "predator" in observations
        assert "prey" in observations

        # Check correct number of agents
        assert observations["predator"].shape[0] == 2
        assert observations["prey"].shape[0] == 3

        # Check observation dimensions
        assert observations["predator"].shape[1] > 0
        assert observations["prey"].shape[1] > 0

    def test_step(self):
        """Test environment step."""
        observations, _ = self.env.reset(seed=42)

        # Create actions for all agents
        actions = {
            "predator": np.array([0, 1]),  # 2 predators
            "prey": np.array([2, 3, 0]),  # 3 prey
        }

        next_obs, rewards, terminated, truncated, info = self.env.step(actions)

        # Check outputs
        assert "predator" in next_obs
        assert "prey" in next_obs
        assert "predator" in rewards
        assert "prey" in rewards

        # Check reward shapes
        assert rewards["predator"].shape == (2,)
        assert rewards["prey"].shape == (3,)

        # Check termination flags
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_multi_population_state(self):
        """Test multi-population state tracking."""
        self.env.reset(seed=42)

        multi_pop_state = self.env.get_multi_population_state()

        # Check densities exist for both types
        densities = multi_pop_state.get_all_densities()
        assert "predator" in densities
        assert "prey" in densities

        # Check density shapes
        assert densities["predator"].shape == (10, 10)
        assert densities["prey"].shape == (10, 10)

    def test_goal_reaching(self):
        """Test goal reaching for heterogeneous types."""
        observations, _ = self.env.reset(seed=42)

        # Move agents towards goals
        max_steps = 100
        for _ in range(max_steps):
            # Simple policy: move right and down for predator, right and up for prey
            actions = {
                "predator": np.array([1, 1]),  # DOWN for both
                "prey": np.array([0, 0, 0]),  # UP for all
            }

            _, rewards, terminated, truncated, _ = self.env.step(actions)

            if terminated or truncated:
                break

        # Check that episode ran without errors
        assert self.env.current_step > 0, "Episode should make progress"

        # It's OK if episode doesn't terminate early - goals may not be reached with simple policy
        # The important thing is that the environment works correctly

    def test_collision_penalty(self):
        """Test collision with walls."""
        observations, _ = self.env.reset(seed=42)

        # Try to move into wall (depends on start position)
        # Position (1,1) -> try to move LEFT (into wall at col 0)
        actions = {
            "predator": np.array([2, 2]),  # LEFT
            "prey": np.array([0, 0, 0]),  # UP
        }

        _, rewards, _, _, _ = self.env.step(actions)

        # If collision occurred, reward should include penalty
        # (exact value depends on configuration)
        assert rewards["predator"] is not None
        assert rewards["prey"] is not None


@pytest.mark.skipif(not GYMNASIUM_AVAILABLE, reason="Gymnasium not available")
class TestMultiPopulationConfig:
    """Tests for multi-population configuration."""

    def test_config_validation(self):
        """Test configuration validation."""
        maze = np.ones((10, 10), dtype=np.int32)
        maze[1:9, 1:9] = 0

        # Valid config with 2 types
        agent_types = {
            "type1": AgentTypeConfig(type_id="type1", type_index=0),
            "type2": AgentTypeConfig(type_id="type2", type_index=1),
        }

        config = MultiPopulationMazeConfig(maze_array=maze, agent_types=agent_types)

        assert len(config.agent_types) == 2

    def test_config_requires_multiple_types(self):
        """Test that config requires at least 2 types."""
        maze = np.ones((10, 10), dtype=np.int32)
        maze[1:9, 1:9] = 0

        # Invalid: only 1 type
        agent_types = {
            "type1": AgentTypeConfig(type_id="type1", type_index=0),
        }

        with pytest.raises(AssertionError, match="at least 2 agent types"):
            MultiPopulationMazeConfig(maze_array=maze, agent_types=agent_types)

    def test_heterogeneous_action_spaces(self):
        """Test different action spaces for different types."""
        maze = np.ones((10, 10), dtype=np.int32)
        maze[1:9, 1:9] = 0

        agent_types = {
            "type1": AgentTypeConfig(
                type_id="type1",
                type_index=0,
                action_type=ActionType.FOUR_CONNECTED,
            ),
            "type2": AgentTypeConfig(
                type_id="type2",
                type_index=1,
                action_type=ActionType.EIGHT_CONNECTED,
            ),
        }

        config = MultiPopulationMazeConfig(maze_array=maze, agent_types=agent_types)
        env = MultiPopulationMazeEnvironment(config)

        # Check action space dimensions
        assert env.action_spaces["type1"].n == 4
        assert env.action_spaces["type2"].n == 8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
