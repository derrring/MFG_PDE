"""
Hyperrectangle Domain (Axis-Aligned Box in n-D)

Hyperrectangle: Cartesian product of intervals in each dimension
    D = [a₁, b₁] × [a₂, b₂] × ... × [aₐ, bₐ]

Most common domain type in MFG applications:
- 2D: Rectangle (e.g., [0,1] × [0,1] for unit square)
- 3D: Box (e.g., [0,1]³ for unit cube)
- 4D: Hypercube (e.g., [0,1]⁴ for unit tesseract)
- nD: General hyperrectangle

Advantages:
- Exact signed distance function (O(d) complexity)
- Efficient sampling (no rejection needed!)
- Natural for grid-based problems
- Easy periodic boundary conditions

References:
- TECHNICAL_REFERENCE_HIGH_DIMENSIONAL_MFG.md Section 4.1
"""

import numpy as np
from numpy.typing import NDArray

from .implicit_domain import ImplicitDomain


class Hyperrectangle(ImplicitDomain):
    """
    Axis-aligned hyperrectangle in n dimensions.

    Domain: D = [a₁, b₁] × [a₂, b₂] × ... × [aₐ, bₐ]

    Signed distance function (exact):
        φ(x) = max(max(a - x), max(x - b))

    where a = (a₁, ..., aₐ) and b = (b₁, ..., bₐ).

    Attributes:
        bounds: Array of shape (d, 2) where bounds[i] = [min_i, max_i]
        dimension: Spatial dimension

    Example:
        >>> # 2D unit square
        >>> domain = Hyperrectangle(np.array([[0, 1], [0, 1]]))
        >>> domain.contains(np.array([0.5, 0.5]))  # True

        >>> # 4D unit hypercube
        >>> domain_4d = Hyperrectangle(np.array([[0, 1]] * 4))
        >>> particles = domain_4d.sample_uniform(10000)
        >>> assert particles.shape == (10000, 4)
    """

    def __init__(self, bounds: NDArray):
        """
        Initialize hyperrectangle domain.

        Args:
            bounds: Array of shape (d, 2) where bounds[i] = [min_i, max_i]

        Raises:
            ValueError: If bounds are invalid (min >= max)

        Example:
            >>> # 2D rectangle [0,2] × [0,1]
            >>> domain = Hyperrectangle(np.array([[0, 2], [0, 1]]))

            >>> # 3D box [-1,1]³
            >>> domain_3d = Hyperrectangle(np.array([[-1, 1]] * 3))
        """
        bounds = np.asarray(bounds, dtype=float)

        if bounds.ndim != 2 or bounds.shape[1] != 2:
            raise ValueError(f"bounds must have shape (d, 2), got {bounds.shape}")

        if np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError("bounds must have min < max for all dimensions")

        self.bounds = bounds
        self._dimension = bounds.shape[0]

    @property
    def dimension(self) -> int:
        """Spatial dimension of the hyperrectangle."""
        return self._dimension

    def signed_distance(self, x: NDArray) -> float | NDArray:
        """
        Compute exact signed distance to hyperrectangle.

        For a point x and hyperrectangle [a, b]:
            φ(x) = max(max(a - x), max(x - b))

        This gives exact Euclidean distance to the boundary.

        Args:
            x: Point(s) - shape (d,) or (N, d)

        Returns:
            Signed distance(s) - scalar or shape (N,)

        Complexity:
            O(d) per point

        Example:
            >>> domain = Hyperrectangle(np.array([[0, 1], [0, 1]]))
            >>> domain.signed_distance(np.array([0.5, 0.5]))  # -0.5 (inside)
            >>> domain.signed_distance(np.array([1.5, 0.5]))  #  0.5 (outside)
            >>> domain.signed_distance(np.array([1.0, 0.5]))  #  0.0 (boundary)
        """
        x = np.asarray(x, dtype=float)
        is_single = x.ndim == 1

        if is_single:
            x = x.reshape(1, -1)

        if x.shape[1] != self.dimension:
            raise ValueError(f"Point dimension {x.shape[1]} does not match domain dimension {self.dimension}")

        # Distance to lower bounds: a - x (positive if x < a)
        lower_dist = self.bounds[:, 0] - x

        # Distance to upper bounds: x - b (positive if x > b)
        upper_dist = x - self.bounds[:, 1]

        # Maximum distance along each dimension
        # If inside on dimension i: max(lower_dist[i], upper_dist[i]) < 0
        max_dist_per_dim = np.maximum(lower_dist, upper_dist)

        # Signed distance is max over all dimensions
        sd = np.max(max_dist_per_dim, axis=1)

        return sd[0] if is_single else sd

    def get_bounding_box(self) -> NDArray:
        """
        Get bounding box (returns self for hyperrectangle).

        Returns:
            bounds: Array of shape (d, 2) where bounds[i] = [min_i, max_i]
        """
        return self.bounds.copy()

    def sample_uniform(self, n_samples: int, max_attempts: int = 100, seed: int | None = None) -> NDArray:
        """
        Sample particles uniformly from hyperrectangle.

        Efficient implementation: no rejection sampling needed!

        Args:
            n_samples: Number of particles
            max_attempts: Unused (kept for interface compatibility)
            seed: Random seed

        Returns:
            particles: Array of shape (n_samples, d)

        Complexity:
            O(n_samples * d) - no rejection!

        Example:
            >>> domain = Hyperrectangle(np.array([[0, 1], [0, 1]]))
            >>> particles = domain.sample_uniform(1000, seed=42)
            >>> assert np.all(particles >= 0) and np.all(particles <= 1)
        """
        if seed is not None:
            np.random.seed(seed)

        # Sample uniformly from each dimension independently
        particles = np.random.uniform(low=self.bounds[:, 0], high=self.bounds[:, 1], size=(n_samples, self.dimension))

        return particles

    def apply_boundary_conditions(self, particles: NDArray, bc_type: str = "reflecting") -> NDArray:
        """
        Apply boundary conditions (specialized for hyperrectangle).

        Args:
            particles: Particle positions - shape (N, d)
            bc_type: Boundary condition type
                - "reflecting": Reflect particles at boundaries
                - "absorbing": Remove particles outside domain
                - "periodic": Wrap particles to opposite side

        Returns:
            Updated particle positions

        Example:
            >>> domain = Hyperrectangle(np.array([[0, 1], [0, 1]]))
            >>> particles = np.array([[1.2, 0.5], [-0.1, 0.5]])
            >>> particles_reflected = domain.apply_boundary_conditions(particles, "reflecting")
            >>> # [1.2, 0.5] → [0.8, 0.5] (reflected from right wall)
            >>> # [-0.1, 0.5] → [0.1, 0.5] (reflected from left wall)
        """
        particles = particles.copy()

        if bc_type == "reflecting":
            # Reflect particles at each boundary
            for dim in range(self.dimension):
                # Reflect from lower boundary
                below = particles[:, dim] < self.bounds[dim, 0]
                particles[below, dim] = 2 * self.bounds[dim, 0] - particles[below, dim]

                # Reflect from upper boundary
                above = particles[:, dim] > self.bounds[dim, 1]
                particles[above, dim] = 2 * self.bounds[dim, 1] - particles[above, dim]

            return particles

        elif bc_type == "absorbing":
            # Remove particles outside domain
            inside = np.all(
                (particles >= self.bounds[:, 0]) & (particles <= self.bounds[:, 1]),
                axis=1,
            )
            return particles[inside]

        elif bc_type == "periodic":
            # Wrap particles to opposite side
            for dim in range(self.dimension):
                width = self.bounds[dim, 1] - self.bounds[dim, 0]

                # Particles below lower bound
                below = particles[:, dim] < self.bounds[dim, 0]
                particles[below, dim] += width

                # Particles above upper bound
                above = particles[:, dim] > self.bounds[dim, 1]
                particles[above, dim] -= width

            return particles

        else:
            raise ValueError(f"Unknown boundary condition type: {bc_type}")

    def compute_volume(self, n_monte_carlo: int = 100000) -> float:
        """
        Compute exact volume (no Monte Carlo needed for hyperrectangle).

        Volume = ∏ᵢ (bᵢ - aᵢ)

        Args:
            n_monte_carlo: Unused (kept for interface compatibility)

        Returns:
            Exact volume

        Example:
            >>> domain = Hyperrectangle(np.array([[0, 2], [0, 3]]))
            >>> domain.compute_volume()  # 2 * 3 = 6
        """
        return np.prod(self.bounds[:, 1] - self.bounds[:, 0])

    def get_center(self) -> NDArray:
        """
        Get center point of hyperrectangle.

        Returns:
            center: Array of shape (d,)

        Example:
            >>> domain = Hyperrectangle(np.array([[0, 2], [0, 4]]))
            >>> domain.get_center()  # array([1, 2])
        """
        return (self.bounds[:, 0] + self.bounds[:, 1]) / 2

    def get_vertices(self) -> NDArray:
        """
        Get all vertices (corners) of hyperrectangle.

        Returns:
            vertices: Array of shape (2^d, d) with all corners

        Warning:
            Grows exponentially with dimension! Use only for low dimensions.

        Example:
            >>> domain = Hyperrectangle(np.array([[0, 1], [0, 1]]))  # 2D square
            >>> vertices = domain.get_vertices()
            >>> # Returns: [[0,0], [0,1], [1,0], [1,1]]
        """
        if self.dimension > 10:
            raise ValueError(
                f"get_vertices() with dimension={self.dimension} would create "
                f"2^{self.dimension} = {2**self.dimension} vertices. Use d≤10."
            )

        # Generate all combinations of (min, max) for each dimension
        from itertools import product

        vertices = []
        for combo in product(*[(0, 1)] * self.dimension):
            vertex = np.array([self.bounds[i, combo[i]] for i in range(self.dimension)])
            vertices.append(vertex)

        return np.array(vertices)

    def __repr__(self) -> str:
        """String representation."""
        bounds_str = ", ".join([f"[{self.bounds[i, 0]}, {self.bounds[i, 1]}]" for i in range(min(self.dimension, 3))])
        if self.dimension > 3:
            bounds_str += ", ..."
        return f"Hyperrectangle({bounds_str})"
