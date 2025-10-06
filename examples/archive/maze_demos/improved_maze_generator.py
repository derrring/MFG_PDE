#!/usr/bin/env python3
"""
Improved Maze Generator for MFG RL Experiments

This module implements proper maze generation principles:
1. Guaranteed connectivity (all cells reachable)
2. Flexible entrance/exit placement on perimeter
3. Strategic bottlenecks for mean field interactions
4. Multiple path options for interesting dynamics

Author: MFG_PDE Team
Date: October 2025
"""

import random
from collections import deque

from mfg_maze_environment import CellType

import matplotlib.pyplot as plt
import numpy as np

from mfg_pde.utils.logging import get_logger

logger = get_logger(__name__)


class ProperMazeGenerator:
    """
    Generates properly connected mazes following maze design principles.
    """

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.maze = np.zeros((height, width), dtype=int)
        self.logger = get_logger(__name__)

    def generate_connected_maze(
        self, algorithm: str = "recursive_backtrack", density: float = 0.4, num_entrances: int = 2
    ) -> np.ndarray:
        """
        Generate a properly connected maze.

        Args:
            algorithm: Generation algorithm ("recursive_backtrack", "random_walk", "cellular_automata")
            density: Wall density (0.0 = empty, 1.0 = full walls)
            num_entrances: Number of entrance/exit points

        Returns:
            Connected maze array
        """
        if algorithm == "recursive_backtrack":
            maze = self._generate_recursive_backtrack()
        elif algorithm == "random_walk":
            maze = self._generate_random_walk(density)
        elif algorithm == "cellular_automata":
            maze = self._generate_cellular_automata(density)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        # Ensure connectivity
        maze = self._ensure_full_connectivity(maze)

        # Add entrances/exits
        maze = self._add_perimeter_entrances(maze, num_entrances)

        # Validate final maze
        self._validate_maze(maze)

        return maze

    def _generate_recursive_backtrack(self) -> np.ndarray:
        """
        Generate maze using recursive backtracking (guaranteed connected).
        Classic algorithm that creates perfect mazes (exactly one path between any two points).
        """
        # Initialize with all walls
        maze = np.zeros((self.height, self.width), dtype=int)

        # Start with border walls
        maze[0, :] = CellType.WALL.value
        maze[-1, :] = CellType.WALL.value
        maze[:, 0] = CellType.WALL.value
        maze[:, -1] = CellType.WALL.value

        # Use iterative approach instead of recursive to avoid stack issues
        visited = set()
        stack = []

        # Start from center
        start_r = self.height // 2
        start_c = self.width // 2
        stack.append((start_r, start_c))

        def get_unvisited_neighbors(r: int, c: int) -> list[tuple[int, int]]:
            neighbors = []
            for dr, dc in [(-2, 0), (2, 0), (0, -2), (0, 2)]:  # Step by 2 for maze cells
                nr, nc = r + dr, c + dc
                if 1 < nr < self.height - 1 and 1 < nc < self.width - 1 and (nr, nc) not in visited:
                    neighbors.append((nr, nc))
            return neighbors

        while stack:
            current_r, current_c = stack[-1]

            if (current_r, current_c) not in visited:
                visited.add((current_r, current_c))
                maze[current_r, current_c] = CellType.EMPTY.value

            neighbors = get_unvisited_neighbors(current_r, current_c)

            if neighbors:
                # Choose random neighbor
                next_r, next_c = random.choice(neighbors)

                # Carve wall between current and next
                wall_r = current_r + (next_r - current_r) // 2
                wall_c = current_c + (next_c - current_c) // 2
                maze[wall_r, wall_c] = CellType.EMPTY.value

                stack.append((next_r, next_c))
            else:
                stack.pop()

        return maze

    def _generate_random_walk(self, density: float) -> np.ndarray:
        """
        Generate maze using constrained random walk to ensure connectivity.
        """
        # Start with walls everywhere
        maze = np.zeros((self.height, self.width), dtype=int)

        # Create border of walls
        maze[0, :] = CellType.WALL.value
        maze[-1, :] = CellType.WALL.value
        maze[:, 0] = CellType.WALL.value
        maze[:, -1] = CellType.WALL.value

        # Random walk to create connected paths
        num_paths = max(3, int((1 - density) * self.width * self.height / 20))

        for _ in range(num_paths):
            # Start random walk from random interior position
            start_r = random.randint(1, self.height - 2)
            start_c = random.randint(1, self.width - 2)

            current_r, current_c = start_r, start_c
            walk_length = random.randint(10, 50)

            for _ in range(walk_length):
                maze[current_r, current_c] = CellType.EMPTY.value

                # Choose random direction
                directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                random.shuffle(directions)

                for dr, dc in directions:
                    new_r, new_c = current_r + dr, current_c + dc
                    if 1 <= new_r < self.height - 1 and 1 <= new_c < self.width - 1:
                        current_r, current_c = new_r, new_c
                        break

        # Add some random empty cells based on density
        for i in range(1, self.height - 1):
            for j in range(1, self.width - 1):
                if random.random() > density:
                    maze[i, j] = CellType.EMPTY.value

        return maze

    def _generate_cellular_automata(self, density: float) -> np.ndarray:
        """
        Generate maze using cellular automata with connectivity post-processing.
        """
        # Initialize randomly
        maze = np.random.choice(
            [CellType.WALL.value, CellType.EMPTY.value], size=(self.height, self.width), p=[density, 1 - density]
        )

        # Ensure borders are walls
        maze[0, :] = CellType.WALL.value
        maze[-1, :] = CellType.WALL.value
        maze[:, 0] = CellType.WALL.value
        maze[:, -1] = CellType.WALL.value

        # Apply cellular automata rules
        for iteration in range(5):
            new_maze = maze.copy()

            for i in range(1, self.height - 1):
                for j in range(1, self.width - 1):
                    # Count wall neighbors
                    wall_count = 0
                    for di in range(-1, 2):
                        for dj in range(-1, 2):
                            if maze[i + di, j + dj] == CellType.WALL.value:
                                wall_count += 1

                    # Cellular automata rule
                    if wall_count >= 5:
                        new_maze[i, j] = CellType.WALL.value
                    else:
                        new_maze[i, j] = CellType.EMPTY.value

            maze = new_maze

        return maze

    def _ensure_full_connectivity(self, maze: np.ndarray) -> np.ndarray:
        """
        Ensure all empty cells are connected by adding paths where needed.
        """
        # Find all connected components
        components = self._find_connected_components(maze)

        if len(components) <= 1:
            return maze  # Already fully connected

        self.logger.info(f"Found {len(components)} disconnected components, connecting them...")

        # Connect components by adding paths
        main_component = max(components, key=len)  # Largest component

        for component in components:
            if component == main_component:
                continue

            # Find shortest path to connect this component to main component
            self._connect_components(maze, component, main_component)

        # Verify connectivity
        final_components = self._find_connected_components(maze)
        self.logger.info(f"After connection: {len(final_components)} components")

        return maze

    def _find_connected_components(self, maze: np.ndarray) -> list[set[tuple[int, int]]]:
        """Find all connected components of empty cells."""
        visited = set()
        components = []

        def get_empty_neighbors(r: int, c: int) -> list[tuple[int, int]]:
            neighbors = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.height and 0 <= nc < self.width and maze[nr, nc] != CellType.WALL.value:
                    neighbors.append((nr, nc))
            return neighbors

        for i in range(self.height):
            for j in range(self.width):
                if maze[i, j] != CellType.WALL.value and (i, j) not in visited:
                    # BFS to find connected component
                    component = set()
                    queue = deque([(i, j)])

                    while queue:
                        r, c = queue.popleft()
                        if (r, c) not in visited:
                            visited.add((r, c))
                            component.add((r, c))

                            for nr, nc in get_empty_neighbors(r, c):
                                if (nr, nc) not in visited:
                                    queue.append((nr, nc))

                    if component:
                        components.append(component)

        return components

    def _connect_components(self, maze: np.ndarray, component1: set[tuple[int, int]], component2: set[tuple[int, int]]):
        """Connect two components by carving a path between them."""
        # Find closest pair of cells between components
        min_distance = float("inf")
        best_pair = None

        for r1, c1 in component1:
            for r2, c2 in component2:
                distance = abs(r1 - r2) + abs(c1 - c2)
                if distance < min_distance:
                    min_distance = distance
                    best_pair = ((r1, c1), (r2, c2))

        if best_pair is None:
            return

        # Carve straight line path between the closest cells
        (r1, c1), (r2, c2) = best_pair

        # Carve horizontal then vertical (L-shaped path)
        current_r, current_c = r1, c1

        # Move horizontally first
        while current_c != c2:
            maze[current_r, current_c] = CellType.EMPTY.value
            current_c += 1 if c2 > current_c else -1

        # Then move vertically
        while current_r != r2:
            maze[current_r, current_c] = CellType.EMPTY.value
            current_r += 1 if r2 > current_r else -1

        # Ensure final cell is empty
        maze[r2, c2] = CellType.EMPTY.value

    def _add_perimeter_entrances(self, maze: np.ndarray, num_entrances: int) -> np.ndarray:
        """
        Add entrances/exits on the perimeter, avoiding connections to internal walls.
        """
        # Find valid perimeter positions (not adjacent to internal walls)
        valid_positions = []

        # Top and bottom edges
        for j in range(1, self.width - 1):
            # Top edge
            if maze[1, j] == CellType.EMPTY.value:  # Internal cell is empty
                valid_positions.append((0, j))
            # Bottom edge
            if maze[self.height - 2, j] == CellType.EMPTY.value:
                valid_positions.append((self.height - 1, j))

        # Left and right edges
        for i in range(1, self.height - 1):
            # Left edge
            if maze[i, 1] == CellType.EMPTY.value:
                valid_positions.append((i, 0))
            # Right edge
            if maze[i, self.width - 2] == CellType.EMPTY.value:
                valid_positions.append((i, self.width - 1))

        # Select random entrance positions
        if len(valid_positions) < num_entrances:
            self.logger.warning(
                f"Only {len(valid_positions)} valid entrance positions found, requested {num_entrances}"
            )
            num_entrances = len(valid_positions)

        entrance_positions = random.sample(valid_positions, num_entrances)

        # Mark entrances as empty
        for r, c in entrance_positions:
            maze[r, c] = CellType.EMPTY.value

        self.logger.info(f"Added {len(entrance_positions)} entrances at positions: {entrance_positions}")

        return maze

    def _validate_maze(self, maze: np.ndarray):
        """Validate that the maze follows all design principles."""
        # Check connectivity
        components = self._find_connected_components(maze)

        if len(components) > 1:
            raise ValueError(f"Maze is not fully connected! Found {len(components)} components")

        # Check that there are some empty cells
        empty_cells = np.sum(maze != CellType.WALL.value)
        if empty_cells < 0.1 * self.height * self.width:
            raise ValueError("Maze has too few empty cells")

        # Check perimeter has some entrances
        perimeter_empty = 0
        perimeter_empty += np.sum(maze[0, :] == CellType.EMPTY.value)
        perimeter_empty += np.sum(maze[-1, :] == CellType.EMPTY.value)
        perimeter_empty += np.sum(maze[:, 0] == CellType.EMPTY.value)
        perimeter_empty += np.sum(maze[:, -1] == CellType.EMPTY.value)

        if perimeter_empty == 0:
            raise ValueError("Maze has no perimeter entrances")

        self.logger.info("✅ Maze validation passed")


def demo_proper_maze_generation():
    """Demonstrate proper maze generation with different algorithms."""
    print("🏗️  Proper Maze Generation Demo")
    print("=" * 50)

    algorithms = ["recursive_backtrack", "random_walk", "cellular_automata"]

    _fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, algorithm in enumerate(algorithms):
        print(f"\n🎯 Testing {algorithm} algorithm...")

        # Generate maze
        generator = ProperMazeGenerator(21, 21)
        maze = generator.generate_connected_maze(algorithm=algorithm, density=0.4, num_entrances=4)

        # Analyze connectivity
        components = generator._find_connected_components(maze)
        empty_cells = np.sum(maze != 0)
        total_cells = maze.shape[0] * maze.shape[1]

        print(f"   Dimensions: {maze.shape}")
        print(f"   Empty cells: {empty_cells}/{total_cells} ({empty_cells/total_cells:.1%})")
        print(f"   Connected components: {len(components)}")
        print(f"   Largest component: {len(components[0]) if components else 0} cells")

        # Visualize original maze
        ax = axes[i]
        ax.imshow(maze, cmap="binary", interpolation="nearest")
        ax.set_title(
            f'{algorithm.replace("_", " ").title()}\n{empty_cells} empty cells, {len(components)} component(s)'
        )
        ax.set_xticks([])
        ax.set_yticks([])

        # Add entrance markers
        entrances = []
        height, width = maze.shape

        # Find entrances on perimeter
        for j in range(width):
            if maze[0, j] == CellType.EMPTY.value:
                entrances.append((0, j))
            if maze[height - 1, j] == CellType.EMPTY.value:
                entrances.append((height - 1, j))

        for i in range(height):
            if maze[i, 0] == CellType.EMPTY.value:
                entrances.append((i, 0))
            if maze[i, width - 1] == CellType.EMPTY.value:
                entrances.append((i, width - 1))

        # Mark entrances with red circles
        for r, c in entrances:
            circle = plt.Circle((c, r), 0.4, color="red", alpha=0.8)
            ax.add_patch(circle)

        # Visualize connectivity analysis
        ax2 = axes[i + len(algorithms)]

        # Create connectivity visualization
        connectivity_maze = maze.copy().astype(float)

        # Color different components differently
        for comp_idx, component in enumerate(components):
            for r, c in component:
                connectivity_maze[r, c] = comp_idx + 1

        if len(components) > 1:
            cmap = "tab10"
        else:
            cmap = "binary"

        ax2.imshow(connectivity_maze, cmap=cmap, interpolation="nearest")
        ax2.set_title(
            f'Connectivity Analysis\n{"✅ Fully Connected" if len(components) == 1 else f"❌ {len(components)} Components"}'
        )
        ax2.set_xticks([])
        ax2.set_yticks([])

        # Mark entrances on connectivity plot too
        for r, c in entrances:
            circle = plt.Circle((c, r), 0.4, color="orange", alpha=0.9)
            ax2.add_patch(circle)

    plt.tight_layout()
    plt.suptitle("Proper Maze Generation: Connectivity & Entrance Analysis", fontsize=16, y=0.98)
    plt.show()


def create_custom_research_maze(width: int = 25, height: int = 25) -> np.ndarray:
    """
    Create a custom maze following all proper design principles.

    Returns:
        Properly connected maze suitable for MFG RL research
    """
    print(f"🎨 Creating custom {height}×{width} research maze...")

    generator = ProperMazeGenerator(width, height)

    # Use cellular automata for interesting structure
    maze = generator.generate_connected_maze(
        algorithm="cellular_automata",
        density=0.45,  # Moderate density for strategic choices
        num_entrances=6,  # Multiple entrances for flexibility
    )

    # Verify it meets our requirements
    components = generator._find_connected_components(maze)
    empty_cells = np.sum(maze != 0)

    print(f"✅ Created maze with {empty_cells} empty cells, {len(components)} component(s)")

    return maze


if __name__ == "__main__":
    demo_proper_maze_generation()

    print("\n" + "=" * 60)
    print("🧪 Creating Custom Research Maze...")
    custom_maze = create_custom_research_maze(21, 21)

    # Visualize the custom maze
    plt.figure(figsize=(10, 10))
    plt.imshow(custom_maze, cmap="binary", interpolation="nearest")
    plt.title("Custom Research Maze\n(Guaranteed Connected with Multiple Entrances)", fontsize=14, fontweight="bold")
    plt.xticks([])
    plt.yticks([])

    # Mark entrances
    height, width = custom_maze.shape
    entrances = []

    for j in range(width):
        if custom_maze[0, j] == 1:
            entrances.append((0, j))
        if custom_maze[height - 1, j] == 1:
            entrances.append((height - 1, j))

    for i in range(height):
        if custom_maze[i, 0] == 1:
            entrances.append((i, 0))
        if custom_maze[i, width - 1] == 1:
            entrances.append((i, width - 1))

    for r, c in entrances:
        circle = plt.Circle((c, r), 0.4, color="red", alpha=0.8)
        plt.gca().add_patch(circle)

    plt.text(
        width // 2, -1.5, f"Red circles = Entrances ({len(entrances)} total)", ha="center", fontsize=12, color="red"
    )

    plt.tight_layout()
    plt.show()
