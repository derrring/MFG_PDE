"""
Unit tests for perfect maze generation.

Tests maze generation algorithms for correctness, reproducibility,
and perfect maze properties (connectivity, acyclicity).
"""

import pytest

import numpy as np

from mfg_pde.alg.reinforcement.environments import (
    MazeAlgorithm,
    PerfectMazeGenerator,
    generate_maze,
    verify_perfect_maze,
)


class TestPerfectMazeGenerator:
    """Test perfect maze generation algorithms."""

    @pytest.mark.parametrize(
        "algorithm",
        [MazeAlgorithm.RECURSIVE_BACKTRACKING, MazeAlgorithm.WILSONS],
    )
    def test_maze_is_perfect(self, algorithm):
        """Test that generated mazes are perfect (connected, no loops)."""
        generator = PerfectMazeGenerator(10, 10, algorithm)
        grid = generator.generate(seed=42)

        verification = verify_perfect_maze(grid)

        assert verification["is_perfect"], f"Maze is not perfect: {verification}"
        assert verification["is_connected"], "Maze is not fully connected"
        assert verification["is_no_loops"], "Maze has loops"

    @pytest.mark.parametrize(
        "algorithm",
        [MazeAlgorithm.RECURSIVE_BACKTRACKING, MazeAlgorithm.WILSONS],
    )
    def test_maze_reproducibility(self, algorithm):
        """Test that same seed produces same maze."""
        gen1 = PerfectMazeGenerator(10, 10, algorithm)
        gen1.generate(seed=42)
        maze1 = gen1.to_numpy_array()

        gen2 = PerfectMazeGenerator(10, 10, algorithm)
        gen2.generate(seed=42)
        maze2 = gen2.to_numpy_array()

        np.testing.assert_array_equal(maze1, maze2)

    @pytest.mark.parametrize(
        "algorithm",
        [MazeAlgorithm.RECURSIVE_BACKTRACKING, MazeAlgorithm.WILSONS],
    )
    def test_maze_dimensions(self, algorithm):
        """Test that maze has correct dimensions."""
        rows, cols = 15, 20
        generator = PerfectMazeGenerator(rows, cols, algorithm)
        generator.generate(seed=42)
        maze = generator.to_numpy_array(wall_thickness=1)

        expected_height = rows * 3 + 1
        expected_width = cols * 3 + 1

        assert maze.shape == (expected_height, expected_width)

    def test_passage_count(self):
        """Test that perfect maze has exactly (n-1) passages for n cells."""
        generator = PerfectMazeGenerator(10, 10, MazeAlgorithm.RECURSIVE_BACKTRACKING)
        grid = generator.generate(seed=42)

        verification = verify_perfect_maze(grid)

        assert verification["passage_count"] == verification["expected_passages"]
        assert verification["passage_count"] == 99  # 100 cells - 1

    def test_connectivity(self):
        """Test that all cells are reachable from any starting cell."""
        generator = PerfectMazeGenerator(10, 10, MazeAlgorithm.RECURSIVE_BACKTRACKING)
        grid = generator.generate(seed=42)

        verification = verify_perfect_maze(grid)

        assert verification["visited_cells"] == verification["total_cells"]
        assert verification["visited_cells"] == 100

    @pytest.mark.parametrize(("rows", "cols"), [(5, 5), (10, 15), (20, 20), (3, 50)])
    def test_various_sizes(self, rows, cols):
        """Test maze generation for various grid sizes."""
        generator = PerfectMazeGenerator(rows, cols, MazeAlgorithm.RECURSIVE_BACKTRACKING)
        grid = generator.generate(seed=42)

        verification = verify_perfect_maze(grid)

        assert verification["is_perfect"]
        assert verification["total_cells"] == rows * cols


class TestGenerateMazeFunction:
    """Test high-level generate_maze() function."""

    def test_generate_maze_recursive_backtracking(self):
        """Test generate_maze with recursive backtracking."""
        maze = generate_maze(10, 10, algorithm="recursive_backtracking", seed=42)

        assert isinstance(maze, np.ndarray)
        assert maze.dtype == np.int32
        assert np.all((maze == 0) | (maze == 1))

    def test_generate_maze_wilsons(self):
        """Test generate_maze with Wilson's algorithm."""
        maze = generate_maze(10, 10, algorithm="wilsons", seed=42)

        assert isinstance(maze, np.ndarray)
        assert maze.dtype == np.int32
        assert np.all((maze == 0) | (maze == 1))

    def test_generate_maze_invalid_algorithm(self):
        """Test that invalid algorithm raises error."""
        with pytest.raises(ValueError):
            generate_maze(10, 10, algorithm="invalid_algorithm")

    def test_generate_maze_reproducibility(self):
        """Test reproducibility of generate_maze()."""
        maze1 = generate_maze(10, 10, algorithm="recursive_backtracking", seed=42)
        maze2 = generate_maze(10, 10, algorithm="recursive_backtracking", seed=42)

        np.testing.assert_array_equal(maze1, maze2)


class TestMazeAlgorithmComparison:
    """Test differences between maze generation algorithms."""

    def test_algorithms_produce_different_mazes(self):
        """Test that different algorithms produce different mazes (with same seed)."""
        maze_rb = generate_maze(20, 20, algorithm="recursive_backtracking", seed=42)
        maze_w = generate_maze(20, 20, algorithm="wilsons", seed=42)

        assert not np.array_equal(maze_rb, maze_w)

    def test_both_algorithms_are_perfect(self):
        """Test that both algorithms produce perfect mazes."""
        for algorithm in ["recursive_backtracking", "wilsons"]:
            generator = PerfectMazeGenerator(15, 15, MazeAlgorithm(algorithm))
            grid = generator.generate(seed=42)
            verification = verify_perfect_maze(grid)

            assert verification["is_perfect"]
            assert verification["is_connected"]
            assert verification["is_no_loops"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
