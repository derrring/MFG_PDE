"""
Implicit Domain Infrastructure for Meshfree Methods

This module provides dimension-agnostic implicit domain representation via
signed distance functions (SDFs). This enables:

- No mesh generation required (O(d) storage instead of O(N^d))
- Natural obstacle representation via CSG operations
- Dimension-agnostic code (works for 2D, 3D, 4D, ..., 100D)
- Particle-friendly boundary conditions

Key concept: A domain D ⊂ ℝ^d is represented implicitly by a function:
    φ: ℝ^d → ℝ  where  φ(x) < 0 ⟺ x ∈ D

References:
- Osher & Fedkiw (2003): Level Set Methods and Dynamic Implicit Surfaces
- TECHNICAL_REFERENCE_HIGH_DIMENSIONAL_MFG.md Section 4
"""

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
from numpy.typing import NDArray


class ImplicitDomain(ABC):
    """
    Abstract base class for n-dimensional implicit domains.

    An implicit domain D ⊂ ℝ^d is defined by a signed distance function φ:
        x ∈ D  ⟺  φ(x) < 0   (interior)
        x ∈ ∂D ⟺  φ(x) = 0   (boundary)
        x ∉ D  ⟺  φ(x) > 0   (exterior)

    Advantages over explicit mesh representation:
    - Memory: O(d) vs O(N^d) for mesh
    - Obstacles: Free with CSG operations
    - Dimension-agnostic: Same code for any d
    - Particle methods: Natural boundary handling

    Subclasses must implement:
    - signed_distance(x): Core SDF
    - get_bounding_box(): For efficient sampling

    Example:
        >>> domain = Hyperrectangle(np.array([[0, 1], [0, 1]]))  # [0,1]²
        >>> domain.contains(np.array([0.5, 0.5]))  # True (interior)
        >>> domain.contains(np.array([1.5, 0.5]))  # False (exterior)
        >>> particles = domain.sample_uniform(1000)  # Sample particles
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Spatial dimension of the domain."""

    @abstractmethod
    def signed_distance(self, x: NDArray[np.float64]) -> float | NDArray[np.float64]:
        """
        Compute signed distance function φ(x).

        Convention:
            φ(x) < 0  ⟺  x inside domain
            φ(x) = 0  ⟺  x on boundary
            φ(x) > 0  ⟺  x outside domain

        Args:
            x: Point(s) to evaluate - shape (d,) or (N, d)

        Returns:
            Signed distance(s) - scalar float or array of shape (N,)

        Notes:
            For exact SDFs, |φ(x)| = distance to boundary.
            For approximations, only sign matters for containment.
        """

    @abstractmethod
    def get_bounding_box(self) -> NDArray[np.float64]:
        """
        Get axis-aligned bounding box containing the domain.

        Returns:
            bounds: Array of shape (d, 2) where bounds[i] = [min_i, max_i]

        Example:
            >>> domain = Hypersphere(center=[0, 0], radius=1.0)
            >>> bounds = domain.get_bounding_box()
            >>> bounds  # array([[-1, 1], [-1, 1]])
        """

    def contains(self, x: NDArray[np.float64], tol: float = 0.0) -> bool | NDArray[np.bool_]:
        """
        Check if point(s) are inside the domain.

        Args:
            x: Point(s) to test - shape (d,) or (N, d)
            tol: Tolerance for boundary (default 0)
                 tol > 0 makes domain slightly larger
                 tol < 0 makes domain slightly smaller

        Returns:
            Boolean or boolean array indicating containment

        Example:
            >>> domain = Hyperrectangle(np.array([[0, 1], [0, 1]]))
            >>> domain.contains(np.array([0.5, 0.5]))  # True
            >>> domain.contains(np.array([[0.5, 0.5], [1.5, 0.5]]))  # [True, False]
        """
        return self.signed_distance(x) <= tol

    def sample_uniform(self, n_samples: int, max_attempts: int = 100, seed: int | None = None) -> NDArray[np.float64]:
        """
        Sample particles uniformly inside the domain via rejection sampling.

        Args:
            n_samples: Number of particles to sample
            max_attempts: Maximum rejection sampling attempts per particle
            seed: Random seed for reproducibility

        Returns:
            particles: Array of shape (n_samples, d) with particles inside domain

        Raises:
            RuntimeError: If rejection sampling fails to find enough valid particles

        Algorithm:
            1. Get bounding box
            2. Sample uniformly from bounding box
            3. Reject samples outside domain
            4. Repeat until n_samples accepted

        Example:
            >>> domain = Hypersphere(center=[0, 0], radius=1.0)
            >>> particles = domain.sample_uniform(1000, seed=42)
            >>> assert particles.shape == (1000, 2)
            >>> assert np.all(domain.contains(particles))
        """
        if seed is not None:
            np.random.seed(seed)

        bounds = self.get_bounding_box()
        dim = self.dimension

        particles = []
        attempts = 0
        max_total_attempts = n_samples * max_attempts

        while len(particles) < n_samples and attempts < max_total_attempts:
            # Sample from bounding box
            batch_size = min(n_samples - len(particles), 1000)
            candidates = np.random.uniform(low=bounds[:, 0], high=bounds[:, 1], size=(batch_size, dim))

            # Accept particles inside domain
            inside = self.contains(candidates)
            if np.isscalar(inside):
                inside = np.array([inside])

            valid = candidates[inside]
            particles.append(valid)

            attempts += batch_size

        if len(particles) == 0 or sum(len(p) for p in particles) < n_samples:
            raise RuntimeError(
                f"Rejection sampling failed: only found {sum(len(p) for p in particles)}/{n_samples} "
                f"valid particles after {attempts} attempts. Domain may be too small or complex."
            )

        all_particles = np.vstack(particles)
        return all_particles[:n_samples]

    def project_to_domain(
        self, x: NDArray[np.float64], method: Literal["simple", "gradient"] = "simple"
    ) -> NDArray[np.float64]:
        """
        Project point(s) outside domain back inside.

        Args:
            x: Point(s) - shape (d,) or (N, d)
            method: Projection method
                - "simple": Scale toward centroid of bounding box
                - "gradient": Use SDF gradient (requires smooth SDF)

        Returns:
            Projected point(s) - same shape as x

        Example:
            >>> domain = Hypersphere(center=[0, 0], radius=1.0)
            >>> outside = np.array([2.0, 0.0])
            >>> inside = domain.project_to_domain(outside)
            >>> assert domain.contains(inside)
        """
        is_single = x.ndim == 1
        if is_single:
            x = x.reshape(1, -1)

        # Check which points need projection
        sd = self.signed_distance(x)
        if np.isscalar(sd):
            sd = np.array([sd])

        outside = sd > 0

        if not np.any(outside):
            return x[0] if is_single else x

        # Simple projection: scale toward bounding box center
        if method == "simple":
            bounds = self.get_bounding_box()
            center = bounds.mean(axis=1)

            x_projected = x.copy()
            for i in np.where(outside)[0]:
                # Move point toward center until inside
                direction = center - x[i]
                alpha = 0.0
                step = 0.1

                for _ in range(100):
                    candidate = x[i] + alpha * direction
                    if self.signed_distance(candidate) <= 0:
                        x_projected[i] = candidate
                        break
                    alpha += step
                else:
                    # Fallback: use center
                    x_projected[i] = center

            return x_projected[0] if is_single else x_projected

        else:
            raise ValueError(f"Unknown projection method: {method}")

    def apply_boundary_conditions(
        self,
        particles: NDArray[np.float64],
        bc_type: Literal["reflecting", "absorbing", "periodic"] = "reflecting",
    ) -> NDArray[np.float64]:
        """
        Apply boundary conditions to particles that left the domain.

        Args:
            particles: Particle positions - shape (N, d)
            bc_type: Boundary condition type
                - "reflecting": Reflect particles back into domain
                - "absorbing": Remove particles outside domain
                - "periodic": Wrap particles to opposite side (box domains only)

        Returns:
            Updated particle positions (may have fewer particles for "absorbing")

        Example:
            >>> domain = Hyperrectangle(np.array([[0, 1], [0, 1]]))
            >>> particles = np.array([[0.5, 0.5], [1.2, 0.5]])  # One outside
            >>> particles_updated = domain.apply_boundary_conditions(particles, "reflecting")
            >>> assert np.all(domain.contains(particles_updated))
        """
        if bc_type == "reflecting":
            return self.project_to_domain(particles, method="simple")

        elif bc_type == "absorbing":
            inside = self.contains(particles)
            if np.isscalar(inside):
                inside = np.array([inside])
            return particles[inside]

        elif bc_type == "periodic":
            # Periodic BC requires special handling (only works for hyperrectangles)
            raise NotImplementedError("Periodic BC requires Hyperrectangle-specific implementation")

        else:
            raise ValueError(f"Unknown boundary condition type: {bc_type}")

    def compute_volume(self, n_monte_carlo: int = 100000) -> float:
        """
        Estimate domain volume via Monte Carlo integration.

        Args:
            n_monte_carlo: Number of samples for Monte Carlo

        Returns:
            Estimated volume

        Example:
            >>> sphere = Hypersphere(center=[0, 0], radius=1.0)
            >>> vol = sphere.compute_volume(n_monte_carlo=1000000)
            >>> assert np.abs(vol - np.pi) < 0.01  # π for unit circle in 2D
        """
        bounds = self.get_bounding_box()
        bbox_volume = np.prod(bounds[:, 1] - bounds[:, 0])

        # Sample uniformly from bounding box
        samples = np.random.uniform(low=bounds[:, 0], high=bounds[:, 1], size=(n_monte_carlo, self.dimension))

        # Fraction inside domain
        inside = self.contains(samples)
        if np.isscalar(inside):
            inside = np.array([inside])

        fraction_inside = np.mean(inside)

        return bbox_volume * fraction_inside

    def __repr__(self) -> str:
        """String representation of the domain."""
        return f"{self.__class__.__name__}(dimension={self.dimension})"
