"""
Unit tests for hybrid maze generation.

Tests hybrid maze creation, connectivity verification, and algorithm combinations.
"""

import pytest

import numpy as np

from mfg_pde.geometry.mazes.hybrid_maze import (
    AlgorithmSpec,
    HybridMazeConfig,
    HybridMazeGenerator,
    HybridStrategy,
    create_campus_hybrid,
    create_museum_hybrid,
    create_office_hybrid,
)

pytestmark = pytest.mark.environment


class TestAlgorithmSpec:
    """Test AlgorithmSpec dataclass."""

    def test_basic_creation(self):
        """Test creating algorithm spec."""
        spec = AlgorithmSpec("voronoi", {"num_points": 10})
        assert spec.algorithm == "voronoi"
        assert spec.config == {"num_points": 10}
        assert spec.region is None

    def test_with_region(self):
        """Test algorithm spec with region."""
        spec = AlgorithmSpec("recursive_division", {}, region="zone_0")
        assert spec.region == "zone_0"

    def test_default_config(self):
        """Test algorithm spec with default empty config."""
        spec = AlgorithmSpec("perfect")
        assert spec.config == {}


class TestHybridMazeConfig:
    """Test HybridMazeConfig dataclass."""

    def test_basic_creation(self):
        """Test creating basic hybrid config."""
        config = HybridMazeConfig(
            rows=50,
            cols=60,
            strategy=HybridStrategy.SPATIAL_SPLIT,
            algorithms=[
                AlgorithmSpec("voronoi", {"num_points": 10}),
                AlgorithmSpec("cellular_automata", {}),
            ],
        )
        assert config.rows == 50
        assert config.cols == 60
        assert config.strategy == HybridStrategy.SPATIAL_SPLIT
        assert len(config.algorithms) == 2

    def test_default_parameters(self):
        """Test default parameter values."""
        config = HybridMazeConfig(
            rows=50,
            cols=60,
            strategy=HybridStrategy.SPATIAL_SPLIT,
            algorithms=[AlgorithmSpec("voronoi"), AlgorithmSpec("cellular_automata")],
        )
        assert config.blend_ratio == 0.5
        assert config.split_axis == "vertical"
        assert config.num_zones == 4
        assert config.seed is None
        assert config.ensure_connectivity is True

    def test_invalid_dimensions(self):
        """Test validation of invalid dimensions."""
        with pytest.raises(ValueError, match="Rows and cols must be positive"):
            HybridMazeConfig(
                rows=0,
                cols=50,
                strategy=HybridStrategy.SPATIAL_SPLIT,
                algorithms=[AlgorithmSpec("voronoi"), AlgorithmSpec("cellular_automata")],
            )

    def test_invalid_blend_ratio(self):
        """Test validation of blend ratio."""
        with pytest.raises(ValueError, match="blend_ratio must be in"):
            HybridMazeConfig(
                rows=50,
                cols=60,
                strategy=HybridStrategy.SPATIAL_SPLIT,
                algorithms=[AlgorithmSpec("voronoi"), AlgorithmSpec("cellular_automata")],
                blend_ratio=1.5,
            )

    def test_no_algorithms(self):
        """Test validation requires at least one algorithm."""
        with pytest.raises(ValueError, match="Must specify at least one algorithm"):
            HybridMazeConfig(
                rows=50,
                cols=60,
                strategy=HybridStrategy.SPATIAL_SPLIT,
                algorithms=[],
            )

    def test_spatial_split_requires_two_algorithms(self):
        """Test SPATIAL_SPLIT requires at least 2 algorithms."""
        with pytest.raises(ValueError, match="SPATIAL_SPLIT requires at least 2"):
            HybridMazeConfig(
                rows=50,
                cols=60,
                strategy=HybridStrategy.SPATIAL_SPLIT,
                algorithms=[AlgorithmSpec("voronoi")],
            )

    def test_invalid_num_zones(self):
        """Test validation of num_zones."""
        with pytest.raises(ValueError, match="num_zones must be at least 2"):
            HybridMazeConfig(
                rows=50,
                cols=60,
                strategy=HybridStrategy.HIERARCHICAL,
                algorithms=[AlgorithmSpec("voronoi"), AlgorithmSpec("cellular_automata")],
                num_zones=1,
            )


class TestHybridMazeGenerator:
    """Test HybridMazeGenerator class."""

    def test_initialization(self):
        """Test generator initialization."""
        config = HybridMazeConfig(
            rows=50,
            cols=60,
            strategy=HybridStrategy.SPATIAL_SPLIT,
            algorithms=[AlgorithmSpec("voronoi"), AlgorithmSpec("cellular_automata")],
            seed=42,
        )
        generator = HybridMazeGenerator(config)
        assert generator.config == config
        assert generator.maze is None
        assert generator.zone_map is None

    def test_vertical_split_generation(self):
        """Test vertical spatial split generation."""
        config = HybridMazeConfig(
            rows=40,
            cols=60,
            strategy=HybridStrategy.SPATIAL_SPLIT,
            algorithms=[
                AlgorithmSpec("voronoi", {"num_points": 5}),
                AlgorithmSpec("cellular_automata", {"initial_wall_prob": 0.45}),
            ],
            blend_ratio=0.5,
            split_axis="vertical",
            seed=42,
        )
        generator = HybridMazeGenerator(config)
        maze = generator.generate()

        # Check maze dimensions
        assert maze.shape == (40, 60)

        # Check binary values
        assert np.all((maze == 0) | (maze == 1))

        # Check zone map exists
        assert generator.zone_map is not None
        assert generator.zone_map.shape == (40, 60)

        # Check two zones created
        assert np.any(generator.zone_map == 0)
        assert np.any(generator.zone_map == 1)

        # Check split location (approximately at 50%)
        assert np.all(generator.zone_map[:, :30] == 0)  # Left half
        assert np.all(generator.zone_map[:, 30:] == 1)  # Right half

    def test_horizontal_split_generation(self):
        """Test horizontal spatial split generation."""
        config = HybridMazeConfig(
            rows=60,
            cols=40,
            strategy=HybridStrategy.SPATIAL_SPLIT,
            algorithms=[
                AlgorithmSpec("recursive_division", {"min_room_width": 4}),
                AlgorithmSpec("voronoi", {"num_points": 6}),
            ],
            blend_ratio=0.6,
            split_axis="horizontal",
            seed=123,
        )
        generator = HybridMazeGenerator(config)
        maze = generator.generate()

        # Check maze dimensions
        assert maze.shape == (60, 40)

        # Check zone map
        assert generator.zone_map is not None
        split_row = int(60 * 0.6)
        assert np.all(generator.zone_map[:split_row, :] == 0)  # Top
        assert np.all(generator.zone_map[split_row:, :] == 1)  # Bottom

    def test_quadrant_split_generation(self):
        """Test quadrant (both axes) spatial split generation."""
        config = HybridMazeConfig(
            rows=80,
            cols=80,
            strategy=HybridStrategy.SPATIAL_SPLIT,
            algorithms=[
                AlgorithmSpec("recursive_division", {"min_room_width": 5}),
                AlgorithmSpec("voronoi", {"num_points": 8}),
                AlgorithmSpec("perfect", {}),
                AlgorithmSpec("cellular_automata", {"initial_wall_prob": 0.42}),
            ],
            split_axis="both",
            seed=999,
        )
        generator = HybridMazeGenerator(config)
        maze = generator.generate()

        # Check maze dimensions
        assert maze.shape == (80, 80)

        # Check four zones created
        assert generator.zone_map is not None
        assert np.any(generator.zone_map == 0)  # NW
        assert np.any(generator.zone_map == 1)  # NE
        assert np.any(generator.zone_map == 2)  # SW
        assert np.any(generator.zone_map == 3)  # SE

    def test_connectivity_verification(self):
        """Test that connectivity verification ensures global connectivity."""
        config = HybridMazeConfig(
            rows=50,
            cols=60,
            strategy=HybridStrategy.SPATIAL_SPLIT,
            algorithms=[
                AlgorithmSpec("voronoi", {"num_points": 6}),
                AlgorithmSpec("cellular_automata", {"initial_wall_prob": 0.45}),
            ],
            ensure_connectivity=True,
            seed=777,
        )
        generator = HybridMazeGenerator(config)
        maze = generator.generate()

        # Use flood fill to verify single connected component
        visited = np.zeros_like(maze, dtype=bool)
        regions_count = 0

        for i in range(maze.shape[0]):
            for j in range(maze.shape[1]):
                if maze[i, j] == 0 and not visited[i, j]:
                    regions_count += 1
                    self._flood_fill_count(maze, visited, i, j)

        # Should have exactly one connected region
        assert regions_count == 1, f"Found {regions_count} disconnected regions"

    @staticmethod
    def _flood_fill_count(maze, visited, start_row, start_col):
        """Helper flood fill for connectivity testing."""
        stack = [(start_row, start_col)]

        while stack:
            row, col = stack.pop()

            if row < 0 or row >= maze.shape[0] or col < 0 or col >= maze.shape[1]:
                continue

            if visited[row, col] or maze[row, col] == 1:
                continue

            visited[row, col] = True

            stack.extend([(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)])

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same maze."""
        config = HybridMazeConfig(
            rows=40,
            cols=50,
            strategy=HybridStrategy.SPATIAL_SPLIT,
            algorithms=[
                AlgorithmSpec("voronoi", {"num_points": 6}),
                AlgorithmSpec("cellular_automata", {}),
            ],
            seed=12345,
        )

        generator1 = HybridMazeGenerator(config)
        maze1 = generator1.generate()

        generator2 = HybridMazeGenerator(config)
        maze2 = generator2.generate()

        assert np.array_equal(maze1, maze2)

    def test_unknown_strategy_raises_error(self):
        """Test that unknown strategy raises error."""
        config = HybridMazeConfig(
            rows=40,
            cols=50,
            strategy=HybridStrategy.HIERARCHICAL,  # Not implemented yet
            algorithms=[AlgorithmSpec("voronoi"), AlgorithmSpec("cellular_automata")],
        )
        generator = HybridMazeGenerator(config)

        with pytest.raises(NotImplementedError):
            generator.generate()

    def test_unknown_algorithm_raises_error(self):
        """Test that unknown algorithm raises error."""
        config = HybridMazeConfig(
            rows=40,
            cols=50,
            strategy=HybridStrategy.SPATIAL_SPLIT,
            algorithms=[
                AlgorithmSpec("unknown_algo", {}),  # type: ignore
                AlgorithmSpec("voronoi", {}),
            ],
        )
        generator = HybridMazeGenerator(config)

        with pytest.raises(ValueError, match="Unknown algorithm"):
            generator.generate()

    def test_has_passages_not_all_walls(self):
        """Test that generated maze has passages (not all walls)."""
        config = HybridMazeConfig(
            rows=40,
            cols=50,
            strategy=HybridStrategy.SPATIAL_SPLIT,
            algorithms=[
                AlgorithmSpec("voronoi", {"num_points": 6}),
                AlgorithmSpec("cellular_automata", {"initial_wall_prob": 0.45}),
            ],
            seed=555,
        )
        generator = HybridMazeGenerator(config)
        maze = generator.generate()

        # Should have significant passage area
        passage_ratio = np.sum(maze == 0) / maze.size
        assert passage_ratio > 0.2, "Maze should have at least 20% passages"


class TestPresetConfigurations:
    """Test preset hybrid maze configurations."""

    def test_museum_hybrid(self):
        """Test museum preset configuration."""
        config = create_museum_hybrid(rows=80, cols=100, seed=42)

        assert config.rows == 80
        assert config.cols == 100
        assert config.strategy == HybridStrategy.SPATIAL_SPLIT
        assert len(config.algorithms) == 2
        assert config.algorithms[0].algorithm == "voronoi"
        assert config.algorithms[1].algorithm == "cellular_automata"
        assert config.blend_ratio == 0.6
        assert config.split_axis == "vertical"

        # Generate and verify
        generator = HybridMazeGenerator(config)
        maze = generator.generate()
        assert maze.shape == (80, 100)

    def test_office_hybrid(self):
        """Test office preset configuration."""
        config = create_office_hybrid(rows=80, cols=100, seed=123)

        assert config.rows == 80
        assert config.cols == 100
        assert config.strategy == HybridStrategy.SPATIAL_SPLIT
        assert len(config.algorithms) == 2
        assert config.algorithms[0].algorithm == "recursive_division"
        assert config.algorithms[1].algorithm == "perfect"
        assert config.blend_ratio == 0.7
        assert config.split_axis == "horizontal"

        # Generate and verify
        generator = HybridMazeGenerator(config)
        maze = generator.generate()
        assert maze.shape == (80, 100)

    def test_campus_hybrid(self):
        """Test campus preset configuration."""
        config = create_campus_hybrid(rows=120, cols=120, seed=999)

        assert config.rows == 120
        assert config.cols == 120
        assert config.strategy == HybridStrategy.SPATIAL_SPLIT
        assert len(config.algorithms) == 4
        assert config.algorithms[0].algorithm == "recursive_division"
        assert config.algorithms[1].algorithm == "voronoi"
        assert config.algorithms[2].algorithm == "perfect"
        assert config.algorithms[3].algorithm == "cellular_automata"
        assert config.split_axis == "both"

        # Generate and verify
        generator = HybridMazeGenerator(config)
        maze = generator.generate()
        assert maze.shape == (120, 120)
