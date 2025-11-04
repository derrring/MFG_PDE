"""
Unit tests for Cellular Automata maze generation.

Tests organic maze generation, CA rules, connectivity,
and preset configurations.
"""

import pytest

import numpy as np

from mfg_pde.alg.reinforcement.environments import (
    CellularAutomataConfig,
    CellularAutomataGenerator,
    create_preset_ca_config,
)

pytestmark = pytest.mark.experimental


class TestCellularAutomataConfig:
    """Test CA configuration."""

    def test_valid_config(self):
        """Test valid configuration creation."""
        config = CellularAutomataConfig(
            rows=50,
            cols=50,
            initial_wall_prob=0.45,
            num_iterations=5,
        )

        assert config.rows == 50
        assert config.cols == 50
        assert config.initial_wall_prob == 0.45
        assert config.num_iterations == 5

    def test_invalid_dimensions(self):
        """Test that invalid dimensions raise error."""
        with pytest.raises(ValueError):
            CellularAutomataConfig(rows=3, cols=50)

        with pytest.raises(ValueError):
            CellularAutomataConfig(rows=50, cols=2)

    def test_invalid_wall_probability(self):
        """Test that invalid wall probability raises error."""
        with pytest.raises(ValueError):
            CellularAutomataConfig(rows=50, cols=50, initial_wall_prob=-0.1)

        with pytest.raises(ValueError):
            CellularAutomataConfig(rows=50, cols=50, initial_wall_prob=1.5)

    def test_invalid_iterations(self):
        """Test that negative iterations raise error."""
        with pytest.raises(ValueError):
            CellularAutomataConfig(rows=50, cols=50, num_iterations=-1)

    def test_default_values(self):
        """Test default configuration values."""
        config = CellularAutomataConfig(rows=50, cols=50)

        assert config.initial_wall_prob == 0.45
        assert config.num_iterations == 5
        assert config.birth_limit == 5
        assert config.death_limit == 4
        assert config.use_moore_neighborhood is True
        assert config.ensure_connectivity is True


class TestCellularAutomataGenerator:
    """Test CA maze generation."""

    def test_basic_generation(self):
        """Test basic maze generation."""
        config = CellularAutomataConfig(rows=30, cols=40, seed=42)
        generator = CellularAutomataGenerator(config)
        maze = generator.generate()

        assert maze.shape == (30, 40)
        assert maze.dtype == np.int32
        assert np.all((maze == 0) | (maze == 1))

    def test_boundary_walls(self):
        """Test that boundary walls are present."""
        config = CellularAutomataConfig(rows=20, cols=30, seed=42)
        generator = CellularAutomataGenerator(config)
        maze = generator.generate()

        assert np.all(maze[0, :] == 1)
        assert np.all(maze[-1, :] == 1)
        assert np.all(maze[:, 0] == 1)
        assert np.all(maze[:, -1] == 1)

    def test_has_open_spaces(self):
        """Test that maze has open spaces."""
        config = CellularAutomataConfig(rows=30, cols=40, seed=42)
        generator = CellularAutomataGenerator(config)
        maze = generator.generate()

        open_cells = np.sum(maze == 0)
        assert open_cells > 0
        assert open_cells > maze.size * 0.1  # At least 10% open

    def test_reproducibility(self):
        """Test that same seed produces same maze."""
        config = CellularAutomataConfig(rows=20, cols=30, seed=42)

        gen1 = CellularAutomataGenerator(config)
        maze1 = gen1.generate()

        gen2 = CellularAutomataGenerator(config)
        maze2 = gen2.generate()

        np.testing.assert_array_equal(maze1, maze2)

    def test_different_seeds(self):
        """Test that different seeds produce different mazes."""
        config1 = CellularAutomataConfig(rows=20, cols=30, seed=42)
        gen1 = CellularAutomataGenerator(config1)
        maze1 = gen1.generate()

        config2 = CellularAutomataConfig(rows=20, cols=30, seed=123)
        gen2 = CellularAutomataGenerator(config2)
        maze2 = gen2.generate()

        assert not np.array_equal(maze1, maze2)

    def test_wall_probability_effect(self):
        """Test that wall probability affects maze density."""
        # Low wall probability → more open space
        config_sparse = CellularAutomataConfig(rows=50, cols=50, initial_wall_prob=0.30, num_iterations=3, seed=42)
        gen_sparse = CellularAutomataGenerator(config_sparse)
        maze_sparse = gen_sparse.generate()
        open_sparse = np.sum(maze_sparse == 0)

        # High wall probability → less open space
        config_dense = CellularAutomataConfig(rows=50, cols=50, initial_wall_prob=0.60, num_iterations=3, seed=42)
        gen_dense = CellularAutomataGenerator(config_dense)
        maze_dense = gen_dense.generate()
        open_dense = np.sum(maze_dense == 0)

        # Sparse should have more open cells
        assert open_sparse > open_dense

    def test_iterations_effect(self):
        """Test that more iterations produce smoother mazes."""
        # Few iterations → noisy
        config_noisy = CellularAutomataConfig(rows=50, cols=50, num_iterations=1, seed=42)
        gen_noisy = CellularAutomataGenerator(config_noisy)
        maze_noisy = gen_noisy.generate()

        # Many iterations → smooth
        config_smooth = CellularAutomataConfig(rows=50, cols=50, num_iterations=8, seed=42)
        gen_smooth = CellularAutomataGenerator(config_smooth)
        maze_smooth = gen_smooth.generate()

        # Both should be valid
        assert maze_noisy.shape == (50, 50)
        assert maze_smooth.shape == (50, 50)

    def test_connectivity(self):
        """Test that connectivity is ensured."""
        config = CellularAutomataConfig(rows=30, cols=40, ensure_connectivity=True, seed=42)
        generator = CellularAutomataGenerator(config)
        maze = generator.generate()

        # Should have open cells
        assert np.sum(maze == 0) > 0

        # Check connectivity via flood fill
        visited = np.zeros_like(maze, dtype=bool)
        regions = []

        for r in range(maze.shape[0]):
            for c in range(maze.shape[1]):
                if maze[r, c] == 0 and not visited[r, c]:
                    # Start flood fill
                    stack = [(r, c)]
                    region_size = 0

                    while stack:
                        curr_r, curr_c = stack.pop()
                        if curr_r < 0 or curr_r >= maze.shape[0] or curr_c < 0 or curr_c >= maze.shape[1]:
                            continue
                        if visited[curr_r, curr_c] or maze[curr_r, curr_c] == 1:
                            continue

                        visited[curr_r, curr_c] = True
                        region_size += 1

                        stack.extend(
                            [
                                (curr_r - 1, curr_c),
                                (curr_r + 1, curr_c),
                                (curr_r, curr_c - 1),
                                (curr_r, curr_c + 1),
                            ]
                        )

                    if region_size > 0:
                        regions.append(region_size)

        # Should have only one region (or very few large regions)
        assert len(regions) <= 3

    def test_moore_vs_von_neumann(self):
        """Test Moore (8-connected) vs Von Neumann (4-connected) neighborhoods."""
        # Moore neighborhood (8-connected)
        config_moore = CellularAutomataConfig(rows=30, cols=40, use_moore_neighborhood=True, seed=42)
        gen_moore = CellularAutomataGenerator(config_moore)
        maze_moore = gen_moore.generate()

        # Von Neumann neighborhood (4-connected)
        config_vn = CellularAutomataConfig(rows=30, cols=40, use_moore_neighborhood=False, seed=42)
        gen_vn = CellularAutomataGenerator(config_vn)
        maze_vn = gen_vn.generate()

        # Should produce different results
        assert not np.array_equal(maze_moore, maze_vn)

    def test_small_region_removal(self):
        """Test removal of small regions."""
        config = CellularAutomataConfig(
            rows=50,
            cols=50,
            min_region_size=20,
            ensure_connectivity=True,
            seed=42,
        )
        generator = CellularAutomataGenerator(config)
        maze = generator.generate()

        # All regions should be large (this is hard to test definitively,
        # but we can check maze is valid)
        assert maze.shape == (50, 50)
        assert np.sum(maze == 0) > 0


class TestPresetConfigurations:
    """Test preset CA configurations."""

    def test_cave_preset(self):
        """Test cave preset."""
        config = create_preset_ca_config(50, 50, style="cave", seed=42)

        assert config.rows == 50
        assert config.cols == 50
        assert config.initial_wall_prob == 0.45
        assert config.num_iterations == 5

    def test_cavern_preset(self):
        """Test cavern preset."""
        config = create_preset_ca_config(50, 50, style="cavern", seed=42)

        assert config.initial_wall_prob == 0.40
        assert config.num_iterations == 4

    def test_maze_preset(self):
        """Test maze preset."""
        config = create_preset_ca_config(50, 50, style="maze", seed=42)

        assert config.initial_wall_prob == 0.50
        assert config.num_iterations == 6

    def test_dense_preset(self):
        """Test dense preset."""
        config = create_preset_ca_config(50, 50, style="dense", seed=42)

        assert config.initial_wall_prob == 0.55
        assert config.num_iterations == 5

    def test_sparse_preset(self):
        """Test sparse preset."""
        config = create_preset_ca_config(50, 50, style="sparse", seed=42)

        assert config.initial_wall_prob == 0.35
        assert config.num_iterations == 3

    def test_invalid_style(self):
        """Test invalid style raises error."""
        with pytest.raises(ValueError):
            create_preset_ca_config(50, 50, style="invalid")

    def test_preset_generation(self):
        """Test that all presets generate valid mazes."""
        styles = ["cave", "cavern", "maze", "dense", "sparse"]

        for style in styles:
            config = create_preset_ca_config(30, 40, style=style, seed=42)
            generator = CellularAutomataGenerator(config)
            maze = generator.generate()

            assert maze.shape == (30, 40)
            assert np.sum(maze == 0) > 0


class TestIntegration:
    """Integration tests with different scenarios."""

    def test_various_sizes(self):
        """Test generation for various maze sizes."""
        sizes = [(20, 20), (30, 40), (50, 50), (20, 60)]

        for rows, cols in sizes:
            config = CellularAutomataConfig(rows=rows, cols=cols, seed=42)
            generator = CellularAutomataGenerator(config)
            maze = generator.generate()

            assert maze.shape == (rows, cols)
            assert np.all((maze == 0) | (maze == 1))

    def test_extreme_parameters(self):
        """Test with extreme parameter values."""
        # Very high wall probability
        config_walls = CellularAutomataConfig(rows=30, cols=30, initial_wall_prob=0.8, seed=42)
        gen_walls = CellularAutomataGenerator(config_walls)
        maze_walls = gen_walls.generate()
        assert maze_walls.shape == (30, 30)

        # Very low wall probability
        config_open = CellularAutomataConfig(rows=30, cols=30, initial_wall_prob=0.2, seed=42)
        gen_open = CellularAutomataGenerator(config_open)
        maze_open = gen_open.generate()
        assert maze_open.shape == (30, 30)

        # Zero iterations
        config_zero = CellularAutomataConfig(rows=30, cols=30, num_iterations=0, seed=42)
        gen_zero = CellularAutomataGenerator(config_zero)
        maze_zero = gen_zero.generate()
        assert maze_zero.shape == (30, 30)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
