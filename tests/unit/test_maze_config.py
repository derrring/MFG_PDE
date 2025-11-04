"""
Unit tests for maze configuration and position placement.

Tests comprehensive parameter control including:
- Physical dimensions (continuous vs discrete)
- Position placement strategies
- Multi-goal configurations
"""

import pytest

from mfg_pde.alg.reinforcement.environments import (
    MazeConfig,
    PerfectMazeGenerator,
    PhysicalDimensions,
    PlacementStrategy,
    compute_position_metrics,
    create_continuous_maze_config,
    create_default_config,
    create_multi_goal_config,
    place_positions,
)

pytestmark = pytest.mark.environment


class TestMazeConfig:
    """Test maze configuration."""

    def test_default_config(self):
        """Test default configuration creation."""
        config = create_default_config(20, 30)

        assert config.rows == 20
        assert config.cols == 30
        assert config.num_starts == 1
        assert config.num_goals == 1
        assert config.algorithm == "recursive_backtracking"

    def test_physical_dimensions(self):
        """Test configuration with physical dimensions."""
        physical_dims = PhysicalDimensions(width=10.0, height=15.0, cell_size=0.5)
        config = MazeConfig(rows=20, cols=30, physical_dims=physical_dims)

        assert config.physical_dims.width == 10.0
        assert config.physical_dims.height == 15.0
        assert config.physical_dims.cell_size == 0.5

    def test_continuous_maze_config(self):
        """Test continuous maze configuration."""
        config = create_continuous_maze_config(10.0, 10.0, cell_density=20)

        assert config.rows == 200
        assert config.cols == 200
        assert config.physical_dims is not None
        assert config.physical_dims.width == 10.0
        assert config.physical_dims.height == 10.0

    def test_cell_to_continuous_conversion(self):
        """Test conversion from cell to continuous coordinates."""
        config = create_continuous_maze_config(10.0, 10.0, cell_density=10)

        x, y = config.cell_to_continuous(5, 5)

        assert x is not None
        assert y is not None
        assert 0.0 <= x <= 10.0
        assert 0.0 <= y <= 10.0

    def test_continuous_to_cell_conversion(self):
        """Test conversion from continuous to cell coordinates."""
        config = create_continuous_maze_config(10.0, 10.0, cell_density=10)

        row, col = config.continuous_to_cell(5.0, 5.0)

        assert row is not None
        assert col is not None
        assert 40 < row < 60
        assert 40 < col < 60

    def test_multi_goal_config(self):
        """Test multi-goal configuration."""
        config = create_multi_goal_config(30, 30, num_goals=4, goal_strategy="corners")

        assert config.num_goals == 4
        assert config.placement_strategy == PlacementStrategy.CORNERS

    def test_invalid_dimensions(self):
        """Test that invalid dimensions raise error."""
        with pytest.raises(ValueError):
            MazeConfig(rows=1, cols=10)

        with pytest.raises(ValueError):
            MazeConfig(rows=10, cols=0)

    def test_pixel_dimensions(self):
        """Test pixel dimension calculation."""
        config = MazeConfig(rows=10, cols=20, wall_thickness=1)

        height, width = config.get_pixel_dimensions()

        assert height == 10 * 3 + 1
        assert width == 20 * 3 + 1


class TestPositionPlacement:
    """Test position placement strategies."""

    def setup_method(self):
        """Setup test maze."""
        from mfg_pde.alg.reinforcement.environments import MazeAlgorithm

        generator = PerfectMazeGenerator(10, 10, MazeAlgorithm.RECURSIVE_BACKTRACKING)
        self.grid = generator.generate(seed=42)

    def test_random_placement(self):
        """Test random position placement."""
        positions = place_positions(self.grid, num_positions=5, strategy=PlacementStrategy.RANDOM, seed=42)

        assert len(positions) == 5
        assert all(0 <= r < 10 and 0 <= c < 10 for r, c in positions)
        assert len(set(positions)) == 5

    def test_corner_placement(self):
        """Test corner position placement."""
        positions = place_positions(self.grid, num_positions=4, strategy=PlacementStrategy.CORNERS, seed=42)

        assert len(positions) == 4
        expected_corners = {(0, 0), (0, 9), (9, 0), (9, 9)}
        assert set(positions) == expected_corners

    def test_edge_placement(self):
        """Test edge position placement."""
        positions = place_positions(self.grid, num_positions=10, strategy=PlacementStrategy.EDGES, seed=42)

        assert len(positions) == 10

        for r, c in positions:
            assert r == 0 or r == 9 or c == 0 or c == 9

    def test_farthest_placement(self):
        """Test farthest position placement."""
        positions = place_positions(self.grid, num_positions=3, strategy=PlacementStrategy.FARTHEST, seed=42)

        assert len(positions) == 3

        metrics = compute_position_metrics(self.grid, positions)
        assert metrics["min_distance"] > 0

    def test_clustered_placement(self):
        """Test clustered position placement."""
        positions = place_positions(self.grid, num_positions=5, strategy=PlacementStrategy.CLUSTERED, seed=42)

        assert len(positions) == 5
        assert len(set(positions)) == 5

    def test_custom_placement(self):
        """Test custom position placement."""
        custom = [(1, 1), (2, 2), (3, 3)]
        positions = place_positions(
            self.grid,
            num_positions=3,
            strategy=PlacementStrategy.CUSTOM,
            custom_positions=custom,
        )

        assert positions == custom

    def test_custom_placement_validation(self):
        """Test custom placement validates position count."""
        with pytest.raises(ValueError):
            place_positions(
                self.grid,
                num_positions=3,
                strategy=PlacementStrategy.CUSTOM,
                custom_positions=[(1, 1)],
            )


class TestPositionMetrics:
    """Test position metric computation."""

    def setup_method(self):
        """Setup test maze."""
        from mfg_pde.alg.reinforcement.environments import MazeAlgorithm

        generator = PerfectMazeGenerator(10, 10, MazeAlgorithm.RECURSIVE_BACKTRACKING)
        self.grid = generator.generate(seed=42)

    def test_metrics_two_positions(self):
        """Test metrics for two positions."""
        positions = [(0, 0), (9, 9)]
        metrics = compute_position_metrics(self.grid, positions)

        assert metrics["total_pairs"] == 1
        assert metrics["min_distance"] == metrics["max_distance"]
        assert metrics["min_distance"] > 0

    def test_metrics_multiple_positions(self):
        """Test metrics for multiple positions."""
        positions = [(0, 0), (0, 9), (9, 0), (9, 9)]
        metrics = compute_position_metrics(self.grid, positions)

        assert metrics["total_pairs"] == 6
        assert metrics["min_distance"] <= metrics["avg_distance"]
        assert metrics["avg_distance"] <= metrics["max_distance"]

    def test_metrics_single_position(self):
        """Test metrics for single position."""
        positions = [(5, 5)]
        metrics = compute_position_metrics(self.grid, positions)

        assert metrics["total_pairs"] == 0
        assert metrics["min_distance"] == 0
        assert metrics["max_distance"] == 0


class TestIntegration:
    """Integration tests combining configuration and generation."""

    def test_continuous_maze_with_positions(self):
        """Test continuous maze with position placement."""
        config = create_continuous_maze_config(5.0, 5.0, cell_density=10, num_goals=3)

        assert config.rows == 50
        assert config.cols == 50
        assert config.num_goals == 3

        continuous_coords = config.cell_to_continuous(25, 25)
        assert continuous_coords is not None

        cell_coords = config.continuous_to_cell(2.5, 2.5)
        assert cell_coords is not None

    def test_multi_goal_with_farthest_strategy(self):
        """Test multi-goal configuration with farthest placement."""
        from mfg_pde.alg.reinforcement.environments import MazeAlgorithm

        config = create_multi_goal_config(20, 20, num_goals=5, goal_strategy="farthest")

        generator = PerfectMazeGenerator(config.rows, config.cols, MazeAlgorithm(config.algorithm))
        grid = generator.generate(seed=42)

        positions = place_positions(grid, config.num_goals, config.placement_strategy, seed=42)

        assert len(positions) == 5

        metrics = compute_position_metrics(grid, positions)
        assert metrics["min_distance"] > 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
