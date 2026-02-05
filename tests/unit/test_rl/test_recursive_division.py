"""
Unit tests for Recursive Division maze generation.

Tests variable-width mazes, room creation, door placement,
and loop addition (braided mazes).
"""

import pytest

import numpy as np

from mfg_pde.alg.reinforcement.environments import (
    RecursiveDivisionConfig,
    RecursiveDivisionGenerator,
    add_loops,
    create_room_based_config,
)


class TestRecursiveDivisionConfig:
    """Test configuration for Recursive Division."""

    def test_valid_config(self):
        """Test valid configuration creation."""
        config = RecursiveDivisionConfig(
            rows=40,
            cols=60,
            min_room_width=5,
            min_room_height=5,
            door_width=2,
        )

        assert config.rows == 40
        assert config.cols == 60
        assert config.door_width == 2

    def test_invalid_dimensions(self):
        """Test that invalid dimensions raise error."""
        with pytest.raises(ValueError):
            RecursiveDivisionConfig(rows=10, cols=20, min_room_height=15)

        with pytest.raises(ValueError):
            RecursiveDivisionConfig(rows=20, cols=10, min_room_width=15)

    def test_invalid_door_width(self):
        """Test that invalid door width raises error."""
        with pytest.raises(ValueError):
            RecursiveDivisionConfig(rows=20, cols=20, door_width=0)

    def test_invalid_split_bias(self):
        """Test that invalid split bias raises error."""
        with pytest.raises(ValueError):
            RecursiveDivisionConfig(rows=20, cols=20, split_bias=1.5)

        with pytest.raises(ValueError):
            RecursiveDivisionConfig(rows=20, cols=20, split_bias=-0.1)


class TestRecursiveDivisionGenerator:
    """Test Recursive Division maze generation."""

    def test_basic_generation(self):
        """Test basic maze generation."""
        config = RecursiveDivisionConfig(rows=30, cols=40)
        generator = RecursiveDivisionGenerator(config)
        maze = generator.generate(seed=42)

        assert maze.shape == (30, 40)
        assert maze.dtype == np.int32
        assert np.all((maze == 0) | (maze == 1))

    def test_boundary_walls(self):
        """Test that boundary walls are present."""
        config = RecursiveDivisionConfig(rows=20, cols=30)
        generator = RecursiveDivisionGenerator(config)
        maze = generator.generate(seed=42)

        assert np.all(maze[0, :] == 1)
        assert np.all(maze[-1, :] == 1)
        assert np.all(maze[:, 0] == 1)
        assert np.all(maze[:, -1] == 1)

    def test_has_open_spaces(self):
        """Test that maze has open spaces (not all walls)."""
        config = RecursiveDivisionConfig(rows=30, cols=40)
        generator = RecursiveDivisionGenerator(config)
        maze = generator.generate(seed=42)

        open_cells = np.sum(maze == 0)
        assert open_cells > 0
        assert open_cells > maze.size * 0.3

    def test_reproducibility(self):
        """Test that same seed produces same maze."""
        config = RecursiveDivisionConfig(rows=20, cols=30)

        gen1 = RecursiveDivisionGenerator(config)
        maze1 = gen1.generate(seed=42)

        gen2 = RecursiveDivisionGenerator(config)
        maze2 = gen2.generate(seed=42)

        np.testing.assert_array_equal(maze1, maze2)

    def test_different_seeds(self):
        """Test that different seeds produce different mazes."""
        config = RecursiveDivisionConfig(rows=20, cols=30)

        gen1 = RecursiveDivisionGenerator(config)
        maze1 = gen1.generate(seed=42)

        gen2 = RecursiveDivisionGenerator(config)
        maze2 = gen2.generate(seed=123)

        assert not np.array_equal(maze1, maze2)

    @pytest.mark.parametrize("door_width", [1, 2, 3])
    def test_door_widths(self, door_width):
        """Test different door widths."""
        config = RecursiveDivisionConfig(rows=30, cols=40, door_width=door_width, min_room_width=5, min_room_height=5)
        generator = RecursiveDivisionGenerator(config)
        maze = generator.generate(seed=42)

        assert maze.shape == (30, 40)
        open_cells = np.sum(maze == 0)
        assert open_cells > 0

    @pytest.mark.parametrize("split_bias", [0.0, 0.5, 1.0])
    def test_split_bias(self, split_bias):
        """Test different split biases."""
        config = RecursiveDivisionConfig(rows=30, cols=40, split_bias=split_bias)
        generator = RecursiveDivisionGenerator(config)
        maze = generator.generate(seed=42)

        assert maze.shape == (30, 40)

    def test_min_room_size(self):
        """Test minimum room size constraint."""
        config = RecursiveDivisionConfig(rows=30, cols=40, min_room_width=10, min_room_height=10)
        generator = RecursiveDivisionGenerator(config)
        maze = generator.generate(seed=42)

        assert maze.shape == (30, 40)


class TestLoopAddition:
    """Test loop addition (braided mazes)."""

    def test_add_loops_basic(self):
        """Test basic loop addition."""
        config = RecursiveDivisionConfig(rows=20, cols=30)
        generator = RecursiveDivisionGenerator(config)
        original_maze = generator.generate(seed=42)

        braided_maze = add_loops(original_maze, loop_density=0.1, seed=42)

        assert braided_maze.shape == original_maze.shape
        # Braided maze should have more open space
        assert np.sum(braided_maze == 0) >= np.sum(original_maze == 0)

    def test_loop_density_zero(self):
        """Test that zero density doesn't change maze."""
        config = RecursiveDivisionConfig(rows=20, cols=30)
        generator = RecursiveDivisionGenerator(config)
        original_maze = generator.generate(seed=42)

        braided_maze = add_loops(original_maze, loop_density=0.0, seed=42)

        np.testing.assert_array_equal(original_maze, braided_maze)

    def test_loop_density_variations(self):
        """Test different loop densities."""
        config = RecursiveDivisionConfig(rows=20, cols=30)
        generator = RecursiveDivisionGenerator(config)
        original_maze = generator.generate(seed=42)

        original_open = np.sum(original_maze == 0)

        for density in [0.1, 0.2, 0.3]:
            braided = add_loops(original_maze, loop_density=density, seed=42)
            braided_open = np.sum(braided == 0)

            # More density → more open space
            assert braided_open >= original_open

    def test_loop_reproducibility(self):
        """Test loop addition is reproducible."""
        config = RecursiveDivisionConfig(rows=20, cols=30)
        generator = RecursiveDivisionGenerator(config)
        maze = generator.generate(seed=42)

        braided1 = add_loops(maze, loop_density=0.15, seed=100)
        braided2 = add_loops(maze, loop_density=0.15, seed=100)

        np.testing.assert_array_equal(braided1, braided2)


class TestRoomBasedConfig:
    """Test preset room-based configurations."""

    def test_small_rooms_narrow_corridors(self):
        """Test small rooms with narrow corridors."""
        config = create_room_based_config(30, 40, room_size="small", corridor_width="narrow")

        assert config.min_room_width == 3
        assert config.min_room_height == 3
        assert config.door_width == 1

    def test_medium_rooms_medium_corridors(self):
        """Test medium rooms with medium corridors."""
        config = create_room_based_config(30, 40, room_size="medium", corridor_width="medium")

        assert config.min_room_width == 5
        assert config.min_room_height == 5
        assert config.door_width == 2

    def test_large_rooms_wide_corridors(self):
        """Test large rooms with wide corridors."""
        config = create_room_based_config(40, 60, room_size="large", corridor_width="wide")

        assert config.min_room_width == 8
        assert config.min_room_height == 8
        assert config.door_width == 3

    def test_preset_generation(self):
        """Test that presets generate valid mazes."""
        for room_size in ["small", "medium", "large"]:
            for corridor_width in ["narrow", "medium", "wide"]:
                config = create_room_based_config(30, 40, room_size=room_size, corridor_width=corridor_width)
                generator = RecursiveDivisionGenerator(config)
                maze = generator.generate(seed=42)

                assert maze.shape == (30, 40)
                assert np.sum(maze == 0) > 0


class TestIntegration:
    """Integration tests combining features."""

    def test_room_based_with_loops(self):
        """Test room-based config with loop addition."""
        config = create_room_based_config(40, 60, room_size="medium", corridor_width="medium")
        generator = RecursiveDivisionGenerator(config)
        maze = generator.generate(seed=42)

        braided_maze = add_loops(maze, loop_density=0.2, seed=42)

        assert braided_maze.shape == (40, 60)
        assert np.sum(braided_maze == 0) >= np.sum(maze == 0)

    def test_various_sizes(self):
        """Test generation for various maze sizes."""
        sizes = [(20, 20), (30, 40), (50, 50), (20, 60)]

        for rows, cols in sizes:
            config = RecursiveDivisionConfig(rows=rows, cols=cols)
            generator = RecursiveDivisionGenerator(config)
            maze = generator.generate(seed=42)

            assert maze.shape == (rows, cols)
            assert np.all((maze == 0) | (maze == 1))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
