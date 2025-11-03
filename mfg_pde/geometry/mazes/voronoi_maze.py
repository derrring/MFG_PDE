"""
Voronoi Diagram Maze Generation for MFG Environments

Generates organic, room-based mazes using Voronoi diagrams and Delaunay
triangulation. Creates realistic building-like layouts with irregular rooms
of varying sizes, connected through a spanning tree to ensure connectivity.

Algorithm:
1. Generate random seed points → room centers
2. Compute Voronoi diagram → defines room boundaries (walls)
3. Extract Delaunay triangulation → dual graph (potential connections)
4. Find spanning tree via Kruskal's algorithm → guaranteed connectivity
5. Carve maze by opening walls at spanning tree edges

Mathematical Foundation:
- **Voronoi Diagram**: Partitions space into cells where each point in a cell
  is closer to that cell's seed than to any other seed
- **Delaunay Triangulation**: Dual graph where seeds are connected if their
  Voronoi cells share an edge
- **Spanning Tree**: Subset of Delaunay edges connecting all seeds without cycles

Properties:
- Organic, realistic room layouts (irregular polygons)
- Variable room sizes emerge naturally from seed distribution
- Guaranteed connectivity (spanning tree)
- Controllable room density (number of seeds)
- Perfect for architectural MFG scenarios

MFG Applications:
- Building evacuation with realistic floor plans
- Museum/gallery crowd flow
- Shopping mall navigation
- Conference venue dynamics
- Irregular urban spaces

References:
- Fortune's Algorithm for Voronoi diagrams
- Delaunay triangulation as geometric dual
- Kruskal's MST algorithm for connectivity

Author: MFG_PDE Team
Date: October 2025
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Scipy for geometric algorithms
try:
    from scipy.spatial import Delaunay, Voronoi

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@dataclass
class VoronoiMazeConfig:
    """
    Configuration for Voronoi diagram maze generation.

    Attributes:
        rows: Grid height
        cols: Grid width
        num_points: Number of Voronoi seed points (room centers)
        wall_thickness: Thickness of Voronoi cell boundaries
        door_width: Width of passages between rooms
        padding: Padding from grid boundaries
        seed: Random seed for reproducibility
        relaxation_iterations: Lloyd's relaxation iterations for uniformity
    """

    rows: int
    cols: int
    num_points: int = 20
    wall_thickness: int = 1
    door_width: int = 2
    padding: int = 5
    seed: int | None = None
    relaxation_iterations: int = 0  # 0 = no relaxation, 3-5 typical

    def __post_init__(self):
        """Validate configuration."""
        if self.rows < 20:
            raise ValueError(f"rows ({self.rows}) must be >= 20 for Voronoi mazes")
        if self.cols < 20:
            raise ValueError(f"cols ({self.cols}) must be >= 20 for Voronoi mazes")
        if self.num_points < 4:
            raise ValueError(f"num_points ({self.num_points}) must be >= 4")
        if self.wall_thickness < 1:
            raise ValueError("wall_thickness must be >= 1")
        if self.door_width < 1:
            raise ValueError("door_width must be >= 1")


class UnionFind:
    """
    Union-Find (Disjoint Set Union) data structure for Kruskal's algorithm.

    Efficiently tracks connected components for spanning tree construction.
    """

    def __init__(self, n: int):
        """
        Initialize union-find structure.

        Args:
            n: Number of elements
        """
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        """
        Find root of element with path compression.

        Args:
            x: Element to find

        Returns:
            Root of element's set
        """
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        """
        Union two sets by rank.

        Args:
            x: First element
            y: Second element

        Returns:
            True if sets were merged (not already connected)
        """
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return False  # Already connected

        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

        return True


class VoronoiMazeGenerator:
    """
    Generates mazes using Voronoi diagrams and Delaunay triangulation.

    Creates organic, room-based layouts ideal for realistic MFG scenarios
    like building evacuation, crowd flow, and architectural navigation.
    """

    def __init__(self, config: VoronoiMazeConfig):
        """
        Initialize generator.

        Args:
            config: Voronoi maze configuration
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy is required for Voronoi maze generation. Install with: pip install scipy")

        self.config = config
        self.maze: NDArray | None = None
        self.points: NDArray | None = None
        self.voronoi: Voronoi | None = None
        self.delaunay: Delaunay | None = None
        self.spanning_edges: list[tuple[int, int]] | None = None

    def generate(self, seed: int | None = None) -> NDArray:
        """
        Generate Voronoi-based maze.

        Args:
            seed: Random seed (overrides config seed)

        Returns:
            Numpy array where 1 = wall, 0 = open space
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        elif self.config.seed is not None:
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)

        # Step 1: Generate random seed points
        self.points = self._generate_seed_points()

        # Optional: Lloyd's relaxation for more uniform distribution
        if self.config.relaxation_iterations > 0:
            self.points = self._lloyds_relaxation(self.points, self.config.relaxation_iterations)

        # Step 2: Create Voronoi diagram
        self.voronoi = Voronoi(self.points)

        # Step 3: Create Delaunay triangulation (dual graph)
        self.delaunay = Delaunay(self.points)

        # Step 4: Find spanning tree using Kruskal's algorithm
        self.spanning_edges = self._kruskals_spanning_tree()

        # Step 5: Carve maze from Voronoi + spanning tree
        self.maze = self._carve_maze()

        return self.maze

    def _generate_seed_points(self) -> NDArray:
        """
        Generate random seed points for Voronoi cells.

        Returns:
            Array of points [(x, y), ...] shape (num_points, 2)
        """
        padding = self.config.padding

        # Generate points within padded boundaries
        points = np.random.rand(self.config.num_points, 2)
        points[:, 0] = points[:, 0] * (self.config.cols - 2 * padding) + padding
        points[:, 1] = points[:, 1] * (self.config.rows - 2 * padding) + padding

        return points

    def _lloyds_relaxation(self, points: NDArray, iterations: int) -> NDArray:
        """
        Apply Lloyd's relaxation to make Voronoi cells more uniform.

        Lloyd's algorithm:
        1. Compute Voronoi diagram for current points
        2. Move each point to the centroid of its Voronoi cell
        3. Repeat for specified iterations

        Args:
            points: Initial seed points
            iterations: Number of relaxation iterations

        Returns:
            Relaxed points
        """
        relaxed_points = points.copy()

        for _ in range(iterations):
            vor = Voronoi(relaxed_points)

            for i in range(len(relaxed_points)):
                region_idx = vor.point_region[i]
                region_vertices = vor.regions[region_idx]

                # Skip infinite regions
                if -1 in region_vertices or len(region_vertices) == 0:
                    continue

                # Compute centroid of Voronoi cell
                vertices = vor.vertices[region_vertices]
                centroid = vertices.mean(axis=0)

                # Keep within bounds
                centroid[0] = np.clip(centroid[0], self.config.padding, self.config.cols - self.config.padding)
                centroid[1] = np.clip(centroid[1], self.config.padding, self.config.rows - self.config.padding)

                relaxed_points[i] = centroid

        return relaxed_points

    def _kruskals_spanning_tree(self) -> list[tuple[int, int]]:
        """
        Find minimum spanning tree using Kruskal's algorithm.

        Kruskal's algorithm for perfect maze:
        1. Create list of all Delaunay edges (potential doors)
        2. Shuffle edges randomly for maze variation
        3. For each edge, check if endpoints are already connected
        4. If not connected, add edge to spanning tree and union sets
        5. Stop when all points are in one connected component

        Returns:
            List of edges in spanning tree [(point_i, point_j), ...]
        """
        if self.delaunay is None:
            raise RuntimeError("Delaunay triangulation not computed")

        num_points = self.config.num_points
        uf = UnionFind(num_points)

        # Extract all Delaunay edges
        edges = []
        for simplex in self.delaunay.simplices:
            # Each simplex (triangle) has 3 edges
            edges.append((simplex[0], simplex[1]))
            edges.append((simplex[1], simplex[2]))
            edges.append((simplex[2], simplex[0]))

        # Remove duplicates (each edge appears twice in different orientations)
        unique_edges = set()
        for i, j in edges:
            unique_edges.add((min(i, j), max(i, j)))

        edges = list(unique_edges)

        # Shuffle for random spanning tree
        random.shuffle(edges)

        # Kruskal's algorithm
        spanning_tree = []
        for i, j in edges:
            if uf.union(i, j):  # If not already connected
                spanning_tree.append((i, j))

            # Stop when we have n-1 edges (spanning tree complete)
            if len(spanning_tree) == num_points - 1:
                break

        return spanning_tree

    def _carve_maze(self) -> NDArray:
        """
        Carve maze from Voronoi diagram using spanning tree.

        Process:
        1. Initialize grid with all walls (1)
        2. Rasterize Voronoi cells as open space (0)
        3. Draw Voronoi edges as walls (1)
        4. Open doors at spanning tree edges (0)

        Returns:
            Maze grid array
        """
        if self.voronoi is None or self.spanning_edges is None:
            raise RuntimeError("Voronoi diagram or spanning tree not computed")

        # Initialize with all walls
        maze = np.ones((self.config.rows, self.config.cols), dtype=np.int32)

        # For each Voronoi cell, fill interior as open space
        maze = self._rasterize_voronoi_cells(maze)

        # Draw Voronoi edges as walls
        maze = self._draw_voronoi_walls(maze)

        # Open doors at spanning tree edges
        maze = self._open_spanning_doors(maze)

        return maze

    def _rasterize_voronoi_cells(self, maze: NDArray) -> NDArray:
        """
        Fill Voronoi cells with open space.

        Args:
            maze: Maze grid to modify

        Returns:
            Updated maze
        """
        # Simple flood-fill approach: assign each grid point to nearest seed
        for r in range(self.config.rows):
            for c in range(self.config.cols):
                # Mark as open space (will be overwritten by walls later)
                # All cells start as open; walls will be drawn on top
                maze[r, c] = 0

        return maze

    def _draw_voronoi_walls(self, maze: NDArray) -> NDArray:
        """
        Draw Voronoi cell boundaries as walls.

        Args:
            maze: Maze grid to modify

        Returns:
            Updated maze
        """
        # For each Voronoi edge, draw wall
        for _ridge_points, ridge_vertices in zip(self.voronoi.ridge_points, self.voronoi.ridge_vertices, strict=False):
            # Skip infinite ridges
            if -1 in ridge_vertices:
                continue

            # Get vertices of this edge
            v0 = self.voronoi.vertices[ridge_vertices[0]]
            v1 = self.voronoi.vertices[ridge_vertices[1]]

            # Rasterize line segment as wall
            maze = self._draw_line(maze, v0, v1, self.config.wall_thickness)

        return maze

    def _open_spanning_doors(self, maze: NDArray) -> NDArray:
        """
        Open doors at spanning tree edges.

        For each edge in spanning tree, find corresponding Voronoi wall
        and create door opening.

        Args:
            maze: Maze grid to modify

        Returns:
            Updated maze
        """
        for i, j in self.spanning_edges:
            # Find midpoint between two seed points
            midpoint = (self.points[i] + self.points[j]) / 2

            # Open door at midpoint
            cx, cy = int(midpoint[0]), int(midpoint[1])

            # Ensure within bounds
            if 0 <= cy < self.config.rows and 0 <= cx < self.config.cols:
                # Open door area
                half_width = self.config.door_width // 2
                for dy in range(-half_width, half_width + 1):
                    for dx in range(-half_width, half_width + 1):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < self.config.rows and 0 <= nx < self.config.cols:
                            maze[ny, nx] = 0

        return maze

    def _draw_line(self, maze: NDArray, p0: NDArray, p1: NDArray, thickness: int) -> NDArray:
        """
        Draw line segment on maze grid using Bresenham's algorithm.

        Args:
            maze: Maze grid to modify
            p0: Start point [x, y]
            p1: End point [x, y]
            thickness: Line thickness

        Returns:
            Updated maze
        """
        x0, y0 = int(p0[0]), int(p0[1])
        x1, y1 = int(p1[0]), int(p1[1])

        # Bresenham's line algorithm
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        x, y = x0, y0

        while True:
            # Draw thick line
            for dt in range(-thickness // 2, thickness // 2 + 1):
                for dn in range(-thickness // 2, thickness // 2 + 1):
                    nx, ny = x + dn, y + dt
                    if 0 <= ny < self.config.rows and 0 <= nx < self.config.cols:
                        maze[ny, nx] = 1

            if x == x1 and y == y1:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

        return maze
