"""
Capacity field computation for maze-based MFG problems.

This module provides spatially-varying capacity fields C(x) representing
the local "carrying capacity" of corridors in maze environments. The capacity
is used to model congestion effects in population-level mean field games.

Key Features:
- Distance-transform-based capacity from maze geometry
- Regularized capacity to prevent division-by-zero singularities
- Interpolation methods for both grid and particle solvers
- Particle-in-Cell (PIC) projection for efficient particle-grid coupling

Mathematical Background:
    In capacity-constrained MFG, the Hamiltonian includes a congestion term:

    H(x, m, ∇u) = (1/2)|∇u|² + γ·g(m(x)/C(x))

    where:
    - m(x): agent density at position x
    - C(x): corridor capacity at position x
    - g(): convex congestion cost function
    - γ: congestion weight parameter

    As C(x) → 0 near walls, the cost g(m/C) → ∞ creates a "soft wall" effect.
    However, this requires careful regularization to avoid numerical instability.

References:
    - Hughes, R. L. (2002). "A continuum theory for the flow of pedestrians."
      Transportation Research Part B, 36(6), 507-535.
    - Di Francesco, M., & Fagioli, S. (2016). "Measure solutions for non-local
      interaction PDEs with two distinct species." DCDS-B, 18(6).

Created: 2025-11-12
Author: MFG_PDE Team
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates

if TYPE_CHECKING:
    from numpy.typing import NDArray


class CapacityField:
    """
    Spatially-varying capacity field for maze navigation MFG.

    The capacity field C(x) represents the local corridor width or carrying
    capacity at each spatial position. It is computed from the maze geometry
    using a distance transform from walls.

    Attributes:
        capacity: Regularized capacity array (N_x, N_y) or (N_x,)
        epsilon: Minimum capacity threshold (prevents singularities)
        cell_size: Physical size of each grid cell
        bounds: Bounding box ((x_min, x_max), (y_min, y_max))

    Examples:
        >>> from mfg_pde.geometry.graph import create_perfect_maze
        >>> maze = create_perfect_maze(rows=20, cols=20, wall_thickness=3)
        >>> maze_array = maze.to_numpy_array(wall_thickness=3)
        >>> capacity = CapacityField.from_maze_geometry(maze_array, wall_thickness=3)
        >>> print(f"Max capacity: {capacity.max_capacity:.3f}")
        >>> print(f"Mean capacity: {capacity.mean_capacity:.3f}")
    """

    def __init__(
        self,
        capacity: NDArray[np.floating],
        epsilon: float = 1e-3,
        cell_size: float = 1.0,
        bounds: tuple[tuple[float, float], ...] | None = None,
    ):
        """
        Initialize capacity field.

        Args:
            capacity: Raw capacity array (will be regularized)
            epsilon: Minimum capacity threshold (prevents C(x) = 0)
            cell_size: Physical size of grid cells
            bounds: Physical bounds ((x_min, x_max), (y_min, y_max), ...)

        Raises:
            ValueError: If capacity has invalid shape or epsilon <= 0
        """
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        self.capacity = np.clip(capacity, epsilon, None)
        self.epsilon = epsilon
        self.cell_size = cell_size

        # Infer bounds if not provided
        if bounds is None:
            shape = self.capacity.shape
            self.bounds = tuple((0.0, float(n * cell_size)) for n in shape)
        else:
            self.bounds = bounds

    @classmethod
    def from_maze_geometry(
        cls,
        maze_array: NDArray[np.integer],
        wall_thickness: float = 1.0,
        epsilon: float = 1e-3,
        decay_factor: float = 0.5,
        normalization: Literal["max", "width", "none"] = "max",
    ) -> CapacityField:
        """
        Compute capacity field from maze geometry using distance transform.

        Algorithm:
            1. Compute Euclidean Distance Transform (EDT) from walls
            2. Normalize distances to [0, 1] range
            3. Apply optional decay function for smooth falloff near walls
            4. Regularize: C_eff(x) = max(C(x), epsilon)

        Args:
            maze_array: Binary maze (0=wall, 1=passage) or (1=wall, 0=passage)
            wall_thickness: Physical wall thickness (for scaling)
            epsilon: Minimum capacity (prevents singularities)
            decay_factor: Exponential decay rate near walls (0=linear, 1=sharp)
            normalization: How to normalize distances
                - "max": Scale by maximum distance (corridor centers = 1.0)
                - "width": Scale by wall_thickness (physical units)
                - "none": Use raw EDT distances

        Returns:
            CapacityField instance with regularized capacity

        Notes:
            The distance transform computes the Euclidean distance from each
            passage cell to the nearest wall cell. Corridor centers have
            maximum distance, walls have zero distance.

            Regularization ensures C(x) ≥ epsilon everywhere, preventing
            division-by-zero in the congestion term g(m/C).

        Examples:
            >>> maze_array = np.array([[1,1,1], [1,0,1], [1,1,1]])  # 0=passage
            >>> capacity = CapacityField.from_maze_geometry(maze_array)
            >>> assert capacity.capacity[1, 1] > capacity.epsilon
        """
        # Detect passage cells (0=wall or 1=passage convention)
        if np.mean(maze_array) > 0.5:
            # 1=passage, 0=wall
            passages = maze_array > 0
        else:
            # 0=passage, 1=wall
            passages = maze_array == 0

        # Compute distance to nearest wall
        distance = distance_transform_edt(passages)

        # Normalize capacity
        if normalization == "max":
            max_dist = distance.max()
            if max_dist > 0:
                capacity_raw = distance / max_dist
            else:
                capacity_raw = distance
        elif normalization == "width":
            # Scale by expected corridor width
            capacity_raw = distance / (wall_thickness + epsilon)
        else:  # "none"
            capacity_raw = distance

        # Apply decay function (optional smoothing near walls)
        if decay_factor > 0 and decay_factor != 1.0:
            capacity_raw = capacity_raw**decay_factor

        # Regularize: Prevent zero capacity
        capacity_regularized = np.clip(capacity_raw, epsilon, None)

        # Set walls to epsilon (safety)
        capacity_regularized[~passages] = epsilon

        return cls(
            capacity=capacity_regularized,
            epsilon=epsilon,
            cell_size=1.0,  # Maze units
        )

    @classmethod
    def from_custom_field(
        cls,
        capacity_array: NDArray[np.floating],
        epsilon: float = 1e-3,
        validate: bool = True,
    ) -> CapacityField:
        """
        Create capacity field from user-specified array.

        Useful for heterogeneous corridor widths or custom geometries.

        Args:
            capacity_array: User-defined capacity values
            epsilon: Minimum capacity threshold
            validate: Check capacity_array >= 0 and apply regularization

        Returns:
            CapacityField instance

        Raises:
            ValueError: If validation fails (negative capacities)
        """
        if validate:
            if np.any(capacity_array < 0):
                raise ValueError("Capacity must be non-negative everywhere")
            capacity_array = np.clip(capacity_array, epsilon, None)

        return cls(capacity=capacity_array, epsilon=epsilon)

    def interpolate_at_positions(
        self,
        positions: NDArray[np.floating],
        method: Literal["nearest", "linear", "cubic"] = "linear",
    ) -> NDArray[np.floating]:
        """
        Interpolate capacity at arbitrary positions (for particle solvers).

        Args:
            positions: Query positions (N, dimension) in grid coordinates
            method: Interpolation method
                - "nearest": Nearest-neighbor (fast, discontinuous)
                - "linear": Bilinear/trilinear (smooth, O(2^d))
                - "cubic": Bicubic (smoother, O(4^d))

        Returns:
            Capacity values at query positions (N,)

        Notes:
            Positions are assumed to be in grid index coordinates,
            not physical coordinates. For physical coordinates, first
            convert using: grid_coords = (phys_coords - bounds_min) / cell_size

        Examples:
            >>> capacity = CapacityField(np.random.rand(10, 10), epsilon=0.01)
            >>> positions = np.array([[5.5, 5.5], [3.2, 7.8]])  # Fractional indices
            >>> c_interp = capacity.interpolate_at_positions(positions)
        """
        # Convert method to scipy order
        order_map = {"nearest": 0, "linear": 1, "cubic": 3}
        order = order_map.get(method, 1)

        # Transpose positions for map_coordinates (expects (dimension, N))
        coords = positions.T

        # Interpolate using scipy
        capacity_values = map_coordinates(
            self.capacity,
            coords,
            order=order,
            mode="nearest",  # Clamp at boundaries
            cval=self.epsilon,  # Fill value outside domain
        )

        return capacity_values

    def particle_in_cell_projection(
        self,
        particle_positions: NDArray[np.floating],
        particle_masses: NDArray[np.floating],
        grid_shape: tuple[int, ...],
    ) -> NDArray[np.floating]:
        """
        Project particle density onto grid using Particle-in-Cell (PIC) method.

        This is the efficient O(N) alternative to KDE for particle-grid coupling.

        Algorithm:
            1. Bin particles onto grid cells (histogramming)
            2. Weight by particle masses
            3. Normalize by cell volume

        Args:
            particle_positions: Particle positions (N, dimension) in grid coords
            particle_masses: Particle masses/weights (N,)
            grid_shape: Target grid dimensions

        Returns:
            Density field on grid (grid_shape)

        Notes:
            This implements first-order (NGP - Nearest Grid Point) PIC.
            For smoother results, use CIC (Cloud-in-Cell) with linear
            weighting across neighboring cells.

        References:
            - Hockney & Eastwood (1988). "Computer Simulation Using Particles."
        """
        # Ensure positions are within grid bounds
        positions_clamped = np.clip(
            particle_positions,
            0,
            np.array(grid_shape) - 1,
        )

        # Round to nearest grid point
        grid_indices = np.round(positions_clamped).astype(int)

        # Bin particles onto grid
        density_grid = np.zeros(grid_shape)

        # Accumulate masses at grid points
        for idx, mass in zip(grid_indices, particle_masses, strict=True):
            density_grid[tuple(idx)] += mass

        return density_grid

    @property
    def dimension(self) -> int:
        """Spatial dimension."""
        return len(self.capacity.shape)

    @property
    def shape(self) -> tuple[int, ...]:
        """Grid shape."""
        return self.capacity.shape

    @property
    def max_capacity(self) -> float:
        """Maximum capacity value."""
        return float(np.max(self.capacity))

    @property
    def mean_capacity(self) -> float:
        """Mean capacity value."""
        return float(np.mean(self.capacity))

    @property
    def min_capacity(self) -> float:
        """Minimum capacity value (should be epsilon)."""
        return float(np.min(self.capacity))

    def get_congestion_ratio(self, density: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Compute congestion ratio ρ(x) = m(x) / C(x).

        Args:
            density: Agent density field (same shape as capacity)

        Returns:
            Congestion ratio array

        Notes:
            Values > 1 indicate overcrowding (density exceeds capacity).
            Values < 1 indicate free flow (capacity available).
        """
        return density / self.capacity

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CapacityField(shape={self.shape}, "
            f"epsilon={self.epsilon:.1e}, "
            f"capacity_range=[{self.min_capacity:.3f}, {self.max_capacity:.3f}])"
        )


def visualize_capacity_field(
    capacity: CapacityField,
    maze_array: NDArray[np.integer] | None = None,
    figsize: tuple[int, int] = (12, 5),
) -> None:
    """
    Visualize capacity field with optional maze overlay.

    Args:
        capacity: CapacityField instance
        maze_array: Optional maze geometry for overlay
        figsize: Figure size (width, height)

    Examples:
        >>> from mfg_pde.geometry.graph import create_perfect_maze
        >>> maze = create_perfect_maze(rows=20, cols=20, wall_thickness=3)
        >>> maze_array = maze.to_numpy_array(wall_thickness=3)
        >>> capacity = CapacityField.from_maze_geometry(maze_array)
        >>> visualize_capacity_field(capacity, maze_array)
    """
    import matplotlib.pyplot as plt

    _fig, axes = plt.subplots(1, 2 if maze_array is not None else 1, figsize=figsize)
    if maze_array is None:
        axes = [axes]

    # Capacity field
    im = axes[0].imshow(capacity.capacity.T, origin="lower", cmap="viridis", interpolation="nearest")
    axes[0].set_title("Capacity Field C(x)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    plt.colorbar(im, ax=axes[0], label="Corridor Capacity")

    # Maze geometry (if provided)
    if maze_array is not None:
        axes[1].imshow(maze_array.T, origin="lower", cmap="binary", interpolation="nearest")
        axes[1].set_title("Maze Geometry")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")

    plt.tight_layout()
    plt.show()


__all__ = [
    "CapacityField",
    "visualize_capacity_field",
]


if __name__ == "__main__":
    """Smoke tests for CapacityField module."""
    print("Running CapacityField smoke tests...")

    # Test 1: Create simple maze and compute capacity
    print("\n1. Distance-transform capacity from simple maze...")
    maze_array = np.array(
        [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 1, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ]
    )

    capacity = CapacityField.from_maze_geometry(maze_array, wall_thickness=1.0)
    print(f"   Shape: {capacity.shape}")
    print(f"   Capacity range: [{capacity.min_capacity:.3f}, {capacity.max_capacity:.3f}]")
    print(f"   Mean capacity: {capacity.mean_capacity:.3f}")
    assert capacity.min_capacity >= capacity.epsilon, "Capacity should be >= epsilon"
    assert capacity.max_capacity <= 1.0, "Normalized capacity should be <= 1.0"
    print("   ✓ Capacity field computed")

    # Test 2: Interpolation at fractional positions
    print("\n2. Interpolation at fractional positions...")
    positions = np.array([[2.5, 2.5], [1.5, 1.5], [2.0, 3.0]])
    capacity_interp = capacity.interpolate_at_positions(positions, method="linear")
    print(f"   Interpolated values: {capacity_interp}")
    assert len(capacity_interp) == 3, "Should have 3 interpolated values"
    assert np.all(capacity_interp >= capacity.epsilon), "Interpolated values should be >= epsilon"
    print("   ✓ Interpolation working")

    # Test 3: Particle-in-Cell projection
    print("\n3. Particle-in-Cell density projection...")
    np.random.seed(42)
    num_particles = 100
    particle_positions = np.random.uniform(1, 4, (num_particles, 2))  # In passage region
    particle_masses = np.ones(num_particles) / num_particles

    density_grid = capacity.particle_in_cell_projection(particle_positions, particle_masses, grid_shape=(5, 5))
    print(f"   Total mass: {np.sum(density_grid):.3f} (should be ~1.0)")
    assert 0.8 < np.sum(density_grid) < 1.2, "Mass should be approximately conserved"
    print("   ✓ PIC projection working")

    # Test 4: Congestion ratio computation
    print("\n4. Congestion ratio computation...")
    test_density = 0.5 * capacity.capacity
    congestion_ratio = capacity.get_congestion_ratio(test_density)
    print(f"   Congestion ratio range: [{np.min(congestion_ratio):.3f}, {np.max(congestion_ratio):.3f}]")
    assert np.allclose(congestion_ratio, 0.5, atol=0.01), "Ratio should be ~0.5"
    print("   ✓ Congestion ratio correct")

    # Test 5: Custom capacity field
    print("\n5. Custom capacity field...")
    custom_array = np.random.uniform(0.1, 1.0, (10, 10))
    custom_capacity = CapacityField.from_custom_field(custom_array, epsilon=0.01)
    print(f"   Custom capacity shape: {custom_capacity.shape}")
    print(f"   Min capacity: {custom_capacity.min_capacity:.3f}")
    assert custom_capacity.min_capacity >= 0.01, "Should respect epsilon"
    print("   ✓ Custom field created")

    print("\n✅ All CapacityField smoke tests passed!")
