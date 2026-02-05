"""
Unit tests for Voronoi diagram maze generation.

Tests Voronoi-based maze generation with Delaunay triangulation,
spanning tree connectivity, and room-based layouts.
"""

import pytest

import numpy as np

from mfg_pde.alg.reinforcement.environments import (
    VoronoiMazeConfig,
    VoronoiMazeGenerator,
)

pytestmark = pytest.mark.environment

# Check scipy availability
scipy_available = True
try:
    from scipy.spatial import Delaunay, Voronoi
except ImportError:
    scipy_available = False


@pytest.mark.skipif(not scipy_available, reason="scipy not available")
class TestVoronoiMazeConfig:
    """Test Voronoi maze configuration."""

    def test_valid_config(self):
        """Test valid configuration creation."""
        config = VoronoiMazeConfig(
            rows=50,
            cols=50,
            num_points=15,
            wall_thickness=1,
            door_width=2,
        )

        assert config.rows == 50
        assert config.cols == 50
        assert config.num_points == 15
        assert config.wall_thickness == 1
        assert config.door_width == 2

    def test_invalid_dimensions(self):
        """Test that small dimensions raise error."""
        with pytest.raises(ValueError):
            VoronoiMazeConfig(rows=10, cols=50)

        with pytest.raises(ValueError):
            VoronoiMazeConfig(rows=50, cols=10)

    def test_invalid_num_points(self):
        """Test that too few points raise error."""
        with pytest.raises(ValueError):
            VoronoiMazeConfig(rows=50, cols=50, num_points=2)

    def test_invalid_wall_thickness(self):
        """Test that invalid wall thickness raises error."""
        with pytest.raises(ValueError):
            VoronoiMazeConfig(rows=50, cols=50, wall_thickness=0)


@pytest.mark.skipif(not scipy_available, reason="scipy not available")
class TestUnionFind:
    """Test Union-Find data structure."""

    def test_initialization(self):
        """Test UnionFind initialization."""
        from mfg_pde.geometry.graph.maze_voronoi import UnionFind

        uf = UnionFind(5)

        # Each element should be its own parent initially
        for i in range(5):
            assert uf.find(i) == i

    def test_union_operation(self):
        """Test union of two sets."""
        from mfg_pde.geometry.graph.maze_voronoi import UnionFind

        uf = UnionFind(5)

        # Union 0 and 1
        result = uf.union(0, 1)
        assert result is True  # Successfully merged

        # They should now have same root
        assert uf.find(0) == uf.find(1)

    def test_union_already_connected(self):
        """Test union of already connected elements."""
        from mfg_pde.geometry.graph.maze_voronoi import UnionFind

        uf = UnionFind(5)

        uf.union(0, 1)
        result = uf.union(0, 1)  # Try again

        assert result is False  # Already connected

    def test_path_compression(self):
        """Test path compression optimization."""
        from mfg_pde.geometry.graph.maze_voronoi import UnionFind

        uf = UnionFind(5)

        # Create chain: 0 <- 1 <- 2 <- 3
        uf.union(0, 1)
        uf.union(1, 2)
        uf.union(2, 3)

        # All should have same root
        root = uf.find(3)
        assert uf.find(0) == root
        assert uf.find(1) == root
        assert uf.find(2) == root


@pytest.mark.skipif(not scipy_available, reason="scipy not available")
class TestVoronoiMazeGenerator:
    """Test Voronoi maze generation."""

    def test_basic_generation(self):
        """Test basic maze generation."""
        config = VoronoiMazeConfig(rows=40, cols=40, num_points=10, seed=42)
        generator = VoronoiMazeGenerator(config)

        maze = generator.generate()

        assert maze.shape == (40, 40)
        assert maze.dtype == np.int32
        assert np.all((maze == 0) | (maze == 1))

    def test_reproducibility(self):
        """Test that same seed produces same maze."""
        config = VoronoiMazeConfig(rows=30, cols=30, num_points=8, seed=42)

        gen1 = VoronoiMazeGenerator(config)
        maze1 = gen1.generate()

        gen2 = VoronoiMazeGenerator(config)
        maze2 = gen2.generate()

        assert np.array_equal(maze1, maze2)

    def test_different_seeds(self):
        """Test that different seeds produce different mazes."""
        config1 = VoronoiMazeConfig(rows=30, cols=30, num_points=8, seed=42)
        config2 = VoronoiMazeConfig(rows=30, cols=30, num_points=8, seed=123)

        gen1 = VoronoiMazeGenerator(config1)
        maze1 = gen1.generate()

        gen2 = VoronoiMazeGenerator(config2)
        maze2 = gen2.generate()

        assert not np.array_equal(maze1, maze2)

    def test_has_open_space(self):
        """Test that maze has significant open space."""
        config = VoronoiMazeConfig(rows=50, cols=50, num_points=12, seed=42)
        generator = VoronoiMazeGenerator(config)

        maze = generator.generate()

        open_cells = np.sum(maze == 0)
        total_cells = maze.size
        open_percentage = 100 * open_cells / total_cells

        # Voronoi mazes should have substantial open space
        assert open_percentage > 30.0

    def test_has_walls(self):
        """Test that maze has walls."""
        config = VoronoiMazeConfig(rows=50, cols=50, num_points=12, seed=42)
        generator = VoronoiMazeGenerator(config)

        maze = generator.generate()

        wall_cells = np.sum(maze == 1)

        # Should have some walls
        assert wall_cells > 0

    def test_seed_points_generation(self):
        """Test seed point generation."""
        config = VoronoiMazeConfig(rows=50, cols=60, num_points=10, padding=5, seed=42)
        generator = VoronoiMazeGenerator(config)

        points = generator._generate_seed_points()

        assert points.shape == (10, 2)
        assert np.all(points[:, 0] >= 5)  # x >= padding
        assert np.all(points[:, 0] <= 55)  # x <= cols - padding
        assert np.all(points[:, 1] >= 5)  # y >= padding
        assert np.all(points[:, 1] <= 45)  # y <= rows - padding

    def test_lloyds_relaxation(self):
        """Test Lloyd's relaxation for uniform distribution."""
        config = VoronoiMazeConfig(rows=50, cols=50, num_points=10, seed=42)
        generator = VoronoiMazeGenerator(config)

        # Generate initial random points
        points = generator._generate_seed_points()

        # Apply relaxation
        relaxed = generator._lloyds_relaxation(points, iterations=3)

        assert relaxed.shape == points.shape
        # Points should move (not identical)
        assert not np.allclose(points, relaxed)

    def test_kruskals_spanning_tree(self):
        """Test Kruskal's spanning tree algorithm."""
        config = VoronoiMazeConfig(rows=40, cols=40, num_points=10, seed=42)
        generator = VoronoiMazeGenerator(config)

        # Generate infrastructure
        generator.points = generator._generate_seed_points()
        generator.voronoi = Voronoi(generator.points)
        generator.delaunay = Delaunay(generator.points)

        # Find spanning tree
        spanning_edges = generator._kruskals_spanning_tree()

        # Spanning tree should have n-1 edges
        assert len(spanning_edges) == config.num_points - 1

        # All edges should be valid point indices
        for i, j in spanning_edges:
            assert 0 <= i < config.num_points
            assert 0 <= j < config.num_points
            assert i != j

    def test_spanning_tree_connectivity(self):
        """Test that spanning tree connects all points."""
        from mfg_pde.geometry.graph.maze_voronoi import UnionFind

        config = VoronoiMazeConfig(rows=40, cols=40, num_points=10, seed=42)
        generator = VoronoiMazeGenerator(config)

        generator.points = generator._generate_seed_points()
        generator.voronoi = Voronoi(generator.points)
        generator.delaunay = Delaunay(generator.points)
        spanning_edges = generator._kruskals_spanning_tree()

        # Verify connectivity using UnionFind
        uf = UnionFind(config.num_points)
        for i, j in spanning_edges:
            uf.union(i, j)

        # All points should have same root (fully connected)
        root = uf.find(0)
        for i in range(config.num_points):
            assert uf.find(i) == root

    def test_varying_num_points(self):
        """Test maze generation with different numbers of points."""
        for num_points in [5, 10, 20, 30]:
            config = VoronoiMazeConfig(rows=60, cols=60, num_points=num_points, seed=42)
            generator = VoronoiMazeGenerator(config)

            maze = generator.generate()

            assert maze.shape == (60, 60)
            assert np.sum(maze == 0) > 0  # Has open space
            assert np.sum(maze == 1) > 0  # Has walls

    def test_relaxation_effect(self):
        """Test that relaxation affects distribution."""
        config_no_relax = VoronoiMazeConfig(rows=50, cols=50, num_points=15, relaxation_iterations=0, seed=42)
        gen_no_relax = VoronoiMazeGenerator(config_no_relax)
        maze_no_relax = gen_no_relax.generate()

        config_relax = VoronoiMazeConfig(rows=50, cols=50, num_points=15, relaxation_iterations=3, seed=42)
        gen_relax = VoronoiMazeGenerator(config_relax)
        maze_relax = gen_relax.generate()

        # Mazes should differ due to relaxation
        assert not np.array_equal(maze_no_relax, maze_relax)

    def test_wall_thickness_variations(self):
        """Test different wall thicknesses."""
        for thickness in [1, 2, 3]:
            config = VoronoiMazeConfig(rows=40, cols=40, num_points=10, wall_thickness=thickness, seed=42)
            generator = VoronoiMazeGenerator(config)

            maze = generator.generate()

            wall_cells = np.sum(maze == 1)

            # Thicker walls should have more wall cells
            assert wall_cells > 0

    def test_door_width_variations(self):
        """Test different door widths."""
        for door_width in [1, 2, 3]:
            config = VoronoiMazeConfig(rows=40, cols=40, num_points=10, door_width=door_width, seed=42)
            generator = VoronoiMazeGenerator(config)

            maze = generator.generate()

            # Wider doors should have more open space
            open_cells = np.sum(maze == 0)
            assert open_cells > 0


@pytest.mark.skipif(not scipy_available, reason="scipy not available")
class TestIntegration:
    """Integration tests for Voronoi maze generation."""

    def test_full_generation_pipeline(self):
        """Test complete generation pipeline."""
        config = VoronoiMazeConfig(rows=50, cols=50, num_points=15, relaxation_iterations=2, seed=42)
        generator = VoronoiMazeGenerator(config)

        # Generate maze
        maze = generator.generate()

        # Verify all components were created
        assert generator.points is not None
        assert generator.voronoi is not None
        assert generator.delaunay is not None
        assert generator.spanning_edges is not None
        assert generator.maze is not None

        # Verify maze properties
        assert maze.shape == (50, 50)
        assert len(generator.spanning_edges) == config.num_points - 1

    def test_various_sizes(self):
        """Test maze generation with various grid sizes."""
        sizes = [(30, 30), (40, 60), (60, 40), (50, 50)]

        for rows, cols in sizes:
            config = VoronoiMazeConfig(rows=rows, cols=cols, num_points=12, seed=42)
            generator = VoronoiMazeGenerator(config)

            maze = generator.generate()

            assert maze.shape == (rows, cols)
            assert np.sum(maze == 0) > 0
            assert np.sum(maze == 1) > 0

    def test_comparison_with_recursive_division(self):
        """Compare Voronoi maze with Recursive Division."""
        from mfg_pde.alg.reinforcement.environments import RecursiveDivisionConfig, RecursiveDivisionGenerator

        # Voronoi maze
        voronoi_config = VoronoiMazeConfig(rows=50, cols=50, num_points=15, seed=42)
        voronoi_gen = VoronoiMazeGenerator(voronoi_config)
        voronoi_maze = voronoi_gen.generate()

        # Recursive division maze
        rd_config = RecursiveDivisionConfig(rows=50, cols=50, min_room_width=5, seed=42)
        rd_gen = RecursiveDivisionGenerator(rd_config)
        rd_maze = rd_gen.generate()

        # Both should have similar open space ratios
        voronoi_open = 100 * np.sum(voronoi_maze == 0) / voronoi_maze.size
        rd_open = 100 * np.sum(rd_maze == 0) / rd_maze.size

        # Both should be in reasonable range (30-90% open)
        # Voronoi tends to have more open space due to room-based structure
        assert 30 < voronoi_open < 90
        assert 30 < rd_open < 90


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
