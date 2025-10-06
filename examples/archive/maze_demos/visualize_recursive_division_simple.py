#!/usr/bin/env python3
"""
Simple ASCII visualization of Recursive Division maze generation.
Shows the characteristic room-based structure with variable-width corridors.
"""

import random

import numpy as np


def generate_recursive_division_maze(rows, cols, min_room_size=5, door_width=1, seed=None):
    """
    Generate maze using Recursive Division algorithm.

    Args:
        rows: Height of maze
        cols: Width of maze
        min_room_size: Minimum dimension before stopping subdivision
        door_width: Width of doors in walls
        seed: Random seed

    Returns:
        Maze array (1=wall, 0=open)
    """
    if seed is not None:
        random.seed(seed)

    # Start with completely open space
    maze = np.zeros((rows, cols), dtype=int)
    # Add border walls
    maze[0, :] = 1
    maze[-1, :] = 1
    maze[:, 0] = 1
    maze[:, -1] = 1

    def divide(x, y, width, height):
        """Recursively divide space and add walls with doors."""
        if width < min_room_size or height < min_room_size:
            return

        # Choose split orientation
        horizontal = (
            random.choice([True, False]) if width > min_room_size and height > min_room_size else width < height
        )

        if horizontal:
            # Add horizontal wall
            wall_y = y + random.randint(min_room_size // 2, height - min_room_size // 2)
            for col in range(x, x + width):
                maze[wall_y, col] = 1

            # Add door
            door_x = x + random.randint(1, width - 2)
            for i in range(min(door_width, width - 2)):
                maze[wall_y, door_x + i] = 0

            # Recurse on subdivisions
            divide(x, y, width, wall_y - y)
            divide(x, wall_y + 1, width, y + height - wall_y - 1)
        else:
            # Add vertical wall
            wall_x = x + random.randint(min_room_size // 2, width - min_room_size // 2)
            for row in range(y, y + height):
                maze[row, wall_x] = 1

            # Add door
            door_y = y + random.randint(1, height - 2)
            for i in range(min(door_width, height - 2)):
                maze[door_y + i, wall_x] = 0

            # Recurse on subdivisions
            divide(x, y, wall_x - x, height)
            divide(wall_x + 1, y, x + width - wall_x - 1, height)

    # Start recursive division
    divide(1, 1, cols - 2, rows - 2)

    return maze


def print_maze_ascii(maze):
    """Print maze as ASCII art."""
    for row in maze:
        line = ""
        for cell in row:
            line += "█" if cell == 1 else " "
        print(line)


def main():
    print("=" * 60)
    print("Recursive Division Maze Generation")
    print("=" * 60)
    print()

    # Generate maze
    maze = generate_recursive_division_maze(rows=30, cols=50, min_room_size=5, door_width=2, seed=42)

    # Print statistics
    num_open = np.sum(maze == 0)
    num_walls = np.sum(maze == 1)
    total = maze.size
    print(f"Maze Size: {maze.shape[0]}x{maze.shape[1]} = {total} cells")
    print(f"Open Space: {num_open} cells ({100*num_open/total:.1f}%)")
    print(f"Walls: {num_walls} cells ({100*num_walls/total:.1f}%)")
    print()
    print("Key Features:")
    print("  - Starts with empty space, adds walls recursively")
    print("  - Creates variable-width rooms and corridors")
    print("  - Door width: 2 cells (configurable bottlenecks)")
    print("  - Minimum room size: 5x5 (controllable subdivision)")
    print()
    print("=" * 60)
    print()

    # Print maze
    print_maze_ascii(maze)

    print()
    print("=" * 60)
    print("Notice the rectangular rooms and variable-width passages")
    print("Perfect for MFG crowd dynamics and building evacuation!")
    print("=" * 60)


if __name__ == "__main__":
    main()
