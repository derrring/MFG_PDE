"""
Particle Density Query for Efficient HJB-FP Coupling.

Provides spatial indexing and density estimation for particle-based Fokker-Planck
solvers, enabling efficient queries at specific points rather than full-grid KDE.

Performance Benefits:
    - Full-grid KDE: O(N_particles × N_grid) per timestep
    - Direct query: O(N_particles × log N_particles + N_query × log N_particles)
    - Speedup: 10-100× when N_query << N_grid

Applications:
    - Semi-Lagrangian HJB: Query density along characteristics
    - Policy iteration: Query at policy evaluation points
    - Adaptive mesh: Query only refined regions

Mathematical Background:
    Kernel Density Estimation at point x:
        m(x) = (1/N) ∑ᵢ K((x - Xᵢ)/h) / h^d

    k-NN Density Estimation:
        m(x) = k / (N · Volume(k-NN ball))

    Hybrid (kernel on k-NN subset):
        m(x) = ∑ᵢ∈kNN K((x - Xᵢ)/h) / (N · h^d)

Periodic Topology Support (Issue #714):
    For periodic (torus) domains, KDTree uses Euclidean distance which fails
    to find correct neighbors near domain boundaries. Solution: ghost particle
    augmentation - replicate particles near boundaries into shifted copies,
    then map neighbor indices back to original particles.

Boundary Correction Support (Issue #709):
    For reflecting/no-flux boundaries, standard KDE underestimates density at
    boundaries by ~50% because the kernel extends into "empty" space. Solution:
    reflection ghost particles - mirror particles about boundaries so the kernel
    "sees" reflected density, correcting the bias.

References:
    - Silverman (1986): Density Estimation for Statistics and Data Analysis
    - Scott (1992): Multivariate Density Estimation
    - scipy.spatial.KDTree documentation

Created: 2026-01-18 (Issue #489 Phase 1 - Core Query Infrastructure)
Updated: 2026-02-05 (Issue #714 - Periodic topology support)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.spatial import KDTree

from mfg_pde.utils.mfg_logging import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = get_logger(__name__)


class ParticleDensityQuery:
    """
    Efficient particle density estimation using spatial indexing.

    Uses KD-tree for O(log N) nearest neighbor queries, enabling fast density
    estimation at arbitrary points without full-grid KDE computation.

    Examples
    --------
    >>> # Create query object from particles
    >>> particles = np.random.rand(10000, 2)  # 10k particles in 2D
    >>> query = ParticleDensityQuery(particles, bandwidth=0.1)
    >>>
    >>> # Query density at specific points
    >>> query_points = np.array([[0.5, 0.5], [0.3, 0.7]])
    >>> densities = query.query_density(query_points, method="kernel")
    >>>
    >>> # Batch query along characteristic
    >>> characteristics = np.linspace([0, 0], [1, 1], 50).reshape(-1, 2)
    >>> densities_path = query.query_density(characteristics)

    Notes
    -----
    **Method Comparison**:

    - "kernel": Gaussian kernel over all particles
      - Smooth, matches full KDE exactly
      - Cost: O(N_particles) per query

    - "knn": k-nearest neighbors
      - Fast, adaptive bandwidth
      - Cost: O(log N_particles) per query
      - Can produce discontinuities

    - "hybrid": Kernel over k-NN subset
      - Balanced: smooth + fast
      - Cost: O(k × log N_particles) per query
      - Recommended for most applications

    **Bandwidth Selection**:

    - Fixed: User-specified h
    - Scott's rule: h = N^(-1/(d+4)) × std(X)
    - Silverman's rule: h = (4/(d+2))^(1/(d+4)) × N^(-1/(d+4)) × std(X)
    """

    def __init__(
        self,
        particles: NDArray[np.float64],
        bandwidth: float | None = None,
        bandwidth_rule: Literal["fixed", "scott", "silverman"] = "fixed",
        periodic_bounds: list[tuple[float, float]] | None = None,
        periodic_dims: tuple[int, ...] | None = None,
        reflect_bounds: list[tuple[float, float]] | None = None,
        reflect_dims: tuple[int, ...] | None = None,
        reflect_margin: float | None = None,
    ):
        """
        Initialize particle density query with spatial index.

        Parameters
        ----------
        particles : NDArray
            Particle positions, shape (N_particles, dimension).
        bandwidth : float, optional
            Kernel bandwidth. If None, computed from bandwidth_rule.
        bandwidth_rule : str, default="fixed"
            Bandwidth selection rule:
            - "fixed": Use provided bandwidth (required if bandwidth is None)
            - "scott": Scott's rule of thumb
            - "silverman": Silverman's rule of thumb
        periodic_bounds : list of (float, float), optional
            Domain bounds for periodic topology, e.g. [(0, 1), (0, 1)].
            If provided with periodic_dims, enables ghost particle augmentation
            for correct neighbor search across periodic boundaries (Issue #714).
        periodic_dims : tuple of int, optional
            Dimensions with periodic topology. If None and periodic_bounds is
            provided, all dimensions are treated as periodic.
            Example: (0,) for cylinder (periodic in x only).
        reflect_bounds : list of (float, float), optional
            Domain bounds for reflecting boundaries (no-flux/Neumann BC).
            Enables reflection ghost particles to correct boundary bias in KDE
            (Issue #709). Cannot be used together with periodic_bounds.
        reflect_dims : tuple of int, optional
            Dimensions with reflecting boundaries. If None and reflect_bounds
            is provided, all dimensions have reflecting boundaries.
        reflect_margin : float, optional
            Distance from boundary to include particles for reflection.
            Default: 3 * bandwidth. Particles within this margin of a boundary
            are reflected about that boundary.

        Raises
        ------
        ValueError
            If bandwidth is None and bandwidth_rule is "fixed".
            If both periodic_bounds and reflect_bounds are provided.
        """
        self.particles = particles
        self.N_particles, self.dimension = particles.shape
        self.periodic_bounds = periodic_bounds
        self.periodic_dims = periodic_dims
        self.reflect_bounds = reflect_bounds
        self.reflect_dims = reflect_dims

        # Validate: can't have both periodic and reflecting
        if periodic_bounds is not None and reflect_bounds is not None:
            raise ValueError(
                "Cannot specify both periodic_bounds and reflect_bounds. "
                "Use periodic for torus topology, reflect for no-flux boundaries."
            )

        # Bandwidth selection (needed before reflection for margin calculation)
        if bandwidth is None:
            if bandwidth_rule == "fixed":
                raise ValueError("bandwidth must be provided when bandwidth_rule='fixed'")
            self.bandwidth = self._compute_bandwidth(bandwidth_rule)
            logger.debug(f"Auto-selected bandwidth: h={self.bandwidth:.4f} ({bandwidth_rule} rule)")
        else:
            self.bandwidth = bandwidth
            logger.debug(f"Using fixed bandwidth: h={bandwidth:.4f}")

        # Issue #714: Handle periodic topology via ghost particle augmentation
        if periodic_bounds is not None:
            from mfg_pde.geometry.boundary.periodic import create_periodic_ghost_points

            # Default: all dimensions periodic if bounds provided but dims not specified
            if periodic_dims is None:
                periodic_dims = tuple(range(self.dimension))
                self.periodic_dims = periodic_dims

            # Create augmented point cloud with ghost copies
            self._particles_augmented, self._original_indices = create_periodic_ghost_points(
                particles, periodic_bounds, periodic_dims
            )
            # Build KD-tree on augmented points
            self.tree = KDTree(self._particles_augmented)
            n_ghosts = len(self._particles_augmented) - self.N_particles
            logger.debug(
                f"Built periodic KD-tree: {self.N_particles} particles + "
                f"{n_ghosts} ghosts = {len(self._particles_augmented)} total"
            )

        # Issue #709: Handle reflecting boundaries via reflection ghost particles
        elif reflect_bounds is not None:
            from mfg_pde.geometry.boundary.periodic import create_reflection_ghost_points

            # Default: all dimensions reflecting if bounds provided but dims not specified
            if reflect_dims is None:
                reflect_dims = tuple(range(self.dimension))
                self.reflect_dims = reflect_dims

            # Default margin: 3 * bandwidth (captures 99.7% of Gaussian kernel)
            margin = reflect_margin if reflect_margin is not None else 3.0 * self.bandwidth

            # Create augmented point cloud with reflected copies near boundaries
            self._particles_augmented, self._original_indices = create_reflection_ghost_points(
                particles, reflect_bounds, margin, reflect_dims
            )
            # Build KD-tree on augmented points
            self.tree = KDTree(self._particles_augmented)
            n_ghosts = len(self._particles_augmented) - self.N_particles
            logger.debug(
                f"Built reflecting KD-tree: {self.N_particles} particles + "
                f"{n_ghosts} ghosts = {len(self._particles_augmented)} total "
                f"(margin={margin:.4f})"
            )

        else:
            # Standard case (no periodic or reflecting boundaries)
            self._particles_augmented = particles
            self._original_indices = np.arange(self.N_particles, dtype=np.int64)
            self.tree = KDTree(particles)
            logger.debug(f"Built KD-tree for {self.N_particles} particles in {self.dimension}D")

    def _compute_bandwidth(self, rule: str) -> float:
        """
        Compute bandwidth using rule of thumb.

        Parameters
        ----------
        rule : str
            Bandwidth selection rule ("scott" or "silverman").

        Returns
        -------
        h : float
            Computed bandwidth.
        """
        std = np.std(self.particles, axis=0).mean()  # Average std across dimensions

        if rule == "scott":
            # Scott's rule: h = N^(-1/(d+4)) × std
            h = self.N_particles ** (-1.0 / (self.dimension + 4)) * std
        elif rule == "silverman":
            # Silverman's rule: h = (4/(d+2))^(1/(d+4)) × N^(-1/(d+4)) × std
            factor = (4.0 / (self.dimension + 2)) ** (1.0 / (self.dimension + 4))
            h = factor * self.N_particles ** (-1.0 / (self.dimension + 4)) * std
        else:
            raise ValueError(f"Unknown bandwidth rule: {rule}")

        return h

    def query_density(
        self,
        query_points: NDArray[np.float64],
        method: Literal["kernel", "knn", "hybrid"] = "hybrid",
        k: int = 50,
    ) -> NDArray[np.float64]:
        """
        Estimate density at query points.

        Parameters
        ----------
        query_points : NDArray
            Query point locations, shape (N_query, dimension) or (dimension,) for single point.
        method : str, default="hybrid"
            Density estimation method:
            - "kernel": Gaussian kernel over all particles (exact, slow)
            - "knn": k-nearest neighbors (fast, discontinuous)
            - "hybrid": Kernel over k-NN subset (recommended)
        k : int, default=50
            Number of nearest neighbors for "knn" and "hybrid" methods.

        Returns
        -------
        densities : NDArray
            Estimated density at each query point, shape (N_query,).

        Examples
        --------
        >>> # Single point query
        >>> density = query.query_density(np.array([0.5, 0.5]))
        >>>
        >>> # Batch query
        >>> points = np.random.rand(100, 2)
        >>> densities = query.query_density(points, method="hybrid", k=30)
        """
        # Handle single point query
        query_points = np.atleast_2d(query_points)

        if query_points.shape[1] != self.dimension:
            raise ValueError(f"Query points dimension {query_points.shape[1]} != particle dimension {self.dimension}")

        if method == "kernel":
            return self._query_kernel(query_points)
        elif method == "knn":
            return self._query_knn(query_points, k)
        elif method == "hybrid":
            return self._query_hybrid(query_points, k)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _query_kernel(self, query_points: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Full kernel density estimation (exact but slow).

        Evaluates Gaussian kernel at all particles for each query point.
        Cost: O(N_query × N_particles)

        Issue #714: For periodic domains, we compute distances to AUGMENTED
        particles (including ghosts) to capture periodic wraparound contributions.
        """
        N_query = query_points.shape[0]
        densities = np.zeros(N_query)

        h = self.bandwidth
        normalization = 1.0 / (self.N_particles * h**self.dimension)

        for i, x in enumerate(query_points):
            # Compute distances to all particles (augmented if periodic)
            distances = np.linalg.norm(self._particles_augmented - x, axis=1)

            # Gaussian kernel: K(r) = exp(-r²/2)
            kernel_values = np.exp(-0.5 * (distances / h) ** 2)

            # Density estimate (normalize by ORIGINAL count, not augmented)
            densities[i] = normalization * np.sum(kernel_values)

        return densities

    def _query_knn(self, query_points: NDArray[np.float64], k: int) -> NDArray[np.float64]:
        """
        k-nearest neighbors density estimation.

        Density proportional to inverse volume containing k neighbors.
        Cost: O(N_query × log N_particles)

        Issue #714: For periodic domains, KDTree on augmented particles gives
        correct geodesic distances (torus metric) via ghost point copies.
        """
        N_query = query_points.shape[0]
        densities = np.zeros(N_query)

        for i, x in enumerate(query_points):
            # Find k nearest neighbors (in augmented cloud if periodic)
            distances, _ = self.tree.query(x, k=k)

            # Radius to k-th neighbor
            radius = distances[-1]

            # Volume of d-dimensional ball
            if self.dimension == 1:
                volume = 2 * radius
            elif self.dimension == 2:
                volume = np.pi * radius**2
            elif self.dimension == 3:
                volume = (4.0 / 3.0) * np.pi * radius**3
            else:
                # General formula: V_d = π^(d/2) / Γ(d/2 + 1) × r^d
                # Approximation for high dimensions
                from scipy.special import gamma

                volume = (np.pi ** (self.dimension / 2.0)) / gamma(self.dimension / 2.0 + 1) * radius**self.dimension

            # Density: k neighbors in volume (normalize by ORIGINAL count)
            densities[i] = k / (self.N_particles * volume) if volume > 1e-12 else 0.0

        return densities

    def _query_hybrid(self, query_points: NDArray[np.float64], k: int) -> NDArray[np.float64]:
        """
        Hybrid: Gaussian kernel over k-nearest neighbors.

        Balances smoothness (kernel) with efficiency (k-NN subset).
        Cost: O(N_query × (log N_particles + k))

        Issue #714: For periodic domains, KDTree is built on augmented points
        (original + ghosts). Distances are correct because ghosts are physical
        copies. We use distances directly - no index mapping needed for kernel.
        """
        N_query = query_points.shape[0]
        densities = np.zeros(N_query)

        h = self.bandwidth
        normalization = 1.0 / (self.N_particles * h**self.dimension)

        for i, x in enumerate(query_points):
            # Find k nearest neighbors (in augmented cloud if periodic)
            distances, _indices = self.tree.query(x, k=k)

            # Gaussian kernel on nearby particles only
            # Note: distances are correct even for periodic case because
            # ghost particles are actual copies at shifted positions
            kernel_values = np.exp(-0.5 * (distances / h) ** 2)

            # Density estimate (scaled by ORIGINAL population, not augmented)
            densities[i] = normalization * np.sum(kernel_values)

        return densities


if __name__ == "__main__":
    """Smoke test for ParticleDensityQuery."""
    print("Testing ParticleDensityQuery...")

    # Test 1: 1D particle distribution
    print("\n[Test 1: 1D Gaussian Particle Distribution]")
    np.random.seed(42)
    particles_1d = np.random.randn(1000, 1) * 0.2 + 0.5  # Gaussian centered at 0.5

    query = ParticleDensityQuery(particles_1d, bandwidth=0.1)
    print(f"  N_particles: {query.N_particles}")
    print(f"  Dimension: {query.dimension}")
    print(f"  Bandwidth: {query.bandwidth:.4f}")

    # Query at center
    x_center = np.array([[0.5]])
    density_center = query.query_density(x_center, method="hybrid", k=50)
    print(f"  Density at x=0.5: {density_center[0]:.4f}")

    # Query at edge
    x_edge = np.array([[0.0]])
    density_edge = query.query_density(x_edge, method="hybrid", k=50)
    print(f"  Density at x=0.0: {density_edge[0]:.4f}")

    assert density_center[0] > density_edge[0], "Center density should be higher than edge"
    print("  ✓ 1D density estimation working!")

    # Test 2: 2D uniform distribution
    print("\n[Test 2: 2D Uniform Distribution]")
    particles_2d = np.random.rand(5000, 2)  # Uniform in [0,1]²

    query_2d = ParticleDensityQuery(particles_2d, bandwidth=0.05)

    # Query multiple points
    query_points = np.array([[0.5, 0.5], [0.1, 0.1], [0.9, 0.9]])
    densities = query_2d.query_density(query_points, method="hybrid", k=30)

    print(f"  Densities at 3 points: {densities}")
    print(f"  Mean density: {densities.mean():.4f}")
    print(f"  Std density: {densities.std():.4f}")

    # For uniform distribution, all densities should be similar
    assert densities.std() < 0.5, "Uniform distribution should have low density variance"
    print("  ✓ 2D density estimation working!")

    # Test 3: Method comparison
    print("\n[Test 3: Method Comparison]")
    x_test = np.array([[0.5, 0.5]])

    density_kernel = query_2d.query_density(x_test, method="kernel")[0]
    density_knn = query_2d.query_density(x_test, method="knn", k=50)[0]
    density_hybrid = query_2d.query_density(x_test, method="hybrid", k=50)[0]

    print(f"  Kernel (exact): {density_kernel:.4f}")
    print(f"  k-NN (k=50): {density_knn:.4f}")
    print(f"  Hybrid (k=50): {density_hybrid:.4f}")

    # All methods should give reasonable density estimates
    assert 0.1 < density_kernel < 10.0, "Kernel density should be reasonable"
    assert 0.1 < density_knn < 10.0, "k-NN density should be reasonable"
    assert 0.1 < density_hybrid < 10.0, "Hybrid density should be reasonable"
    print("  ✓ All methods produce reasonable estimates!")

    # Test 4: Bandwidth selection
    print("\n[Test 4: Automatic Bandwidth Selection]")
    query_scott = ParticleDensityQuery(particles_2d, bandwidth_rule="scott")
    query_silverman = ParticleDensityQuery(particles_2d, bandwidth_rule="silverman")

    print(f"  Scott's rule: h={query_scott.bandwidth:.4f}")
    print(f"  Silverman's rule: h={query_silverman.bandwidth:.4f}")

    assert 0.01 < query_scott.bandwidth < 1.0, "Scott bandwidth should be reasonable"
    assert 0.01 < query_silverman.bandwidth < 1.0, "Silverman bandwidth should be reasonable"
    print("  ✓ Automatic bandwidth selection working!")

    # Test 5: Periodic topology (Issue #714)
    print("\n[Test 5: Periodic Topology (Issue #714)]")
    # Create particles clustered near boundaries of [0,1]²
    # On a torus, points at x=0.05 and x=0.95 are very close (distance ~0.1)
    # Without periodic handling, KDTree sees them as far apart (distance ~0.9)
    np.random.seed(123)
    # Particles near left boundary
    particles_left = np.column_stack(
        [
            np.random.uniform(0.0, 0.1, 50),  # x near 0
            np.random.uniform(0.4, 0.6, 50),  # y near 0.5
        ]
    )
    # Particles near right boundary
    particles_right = np.column_stack(
        [
            np.random.uniform(0.9, 1.0, 50),  # x near 1
            np.random.uniform(0.4, 0.6, 50),  # y near 0.5
        ]
    )
    particles_periodic = np.vstack([particles_left, particles_right])

    # Query point just inside left boundary
    query_periodic = np.array([[0.02, 0.5]])

    # Without periodic: only sees left particles
    query_non_periodic = ParticleDensityQuery(particles_periodic, bandwidth=0.15)
    density_non_periodic = query_non_periodic.query_density(query_periodic, method="hybrid", k=30)

    # With periodic: sees both left and right (wrapped) particles
    periodic_bounds = [(0.0, 1.0), (0.0, 1.0)]
    query_with_periodic = ParticleDensityQuery(
        particles_periodic,
        bandwidth=0.15,
        periodic_bounds=periodic_bounds,
        periodic_dims=(0, 1),  # Full torus
    )
    density_with_periodic = query_with_periodic.query_density(query_periodic, method="hybrid", k=30)

    print("  Query at x=0.02 (near left boundary):")
    print(f"    Non-periodic density: {density_non_periodic[0]:.4f}")
    print(f"    Periodic density: {density_with_periodic[0]:.4f}")

    # Periodic should see more neighbors -> higher density
    assert density_with_periodic[0] > density_non_periodic[0], "Periodic should find more neighbors near boundary"
    print("  Periodic density > non-periodic: passed")

    # Test cylinder topology (periodic only in x)
    query_cylinder = ParticleDensityQuery(
        particles_periodic,
        bandwidth=0.15,
        periodic_bounds=periodic_bounds,
        periodic_dims=(0,),  # Cylinder: periodic in x only
    )
    density_cylinder = query_cylinder.query_density(query_periodic, method="hybrid", k=30)
    print(f"    Cylinder density: {density_cylinder[0]:.4f}")
    # Cylinder should also see wrapped neighbors in x direction
    assert density_cylinder[0] > density_non_periodic[0], "Cylinder should also find periodic neighbors in x"
    print("  Cylinder topology: passed")
    print("  Periodic topology support: OK")

    # Test 6: Reflection boundary correction (Issue #709)
    print("\n[Test 6: Reflection Boundary Correction (Issue #709)]")
    # Create uniform particles in [0, 1]
    # Standard KDE underestimates density at boundaries by ~50%
    # Reflection correction should fix this
    np.random.seed(456)
    particles_uniform = np.random.uniform(0.0, 1.0, (2000, 1))

    # Without reflection: boundary density will be ~50% of center
    query_no_reflect = ParticleDensityQuery(particles_uniform, bandwidth=0.1)
    density_boundary_no_reflect = query_no_reflect.query_density(np.array([[0.0]]), method="kernel")[0]
    density_center_no_reflect = query_no_reflect.query_density(np.array([[0.5]]), method="kernel")[0]

    print("  Without reflection (uniform particles):")
    print(f"    Boundary (x=0) density: {density_boundary_no_reflect:.4f}")
    print(f"    Center (x=0.5) density: {density_center_no_reflect:.4f}")
    boundary_center_ratio_before = density_boundary_no_reflect / density_center_no_reflect
    print(f"    Boundary/center ratio: {boundary_center_ratio_before:.2f} (expected ~0.5)")

    # With reflection: boundary density should be close to center
    reflect_bounds = [(0.0, 1.0)]
    query_with_reflect = ParticleDensityQuery(
        particles_uniform,
        bandwidth=0.1,
        reflect_bounds=reflect_bounds,
    )
    density_boundary_reflect = query_with_reflect.query_density(np.array([[0.0]]), method="kernel")[0]
    density_center_reflect = query_with_reflect.query_density(np.array([[0.5]]), method="kernel")[0]

    print("  With reflection:")
    print(f"    Boundary (x=0) density: {density_boundary_reflect:.4f}")
    print(f"    Center (x=0.5) density: {density_center_reflect:.4f}")
    boundary_center_ratio_after = density_boundary_reflect / density_center_reflect
    print(f"    Boundary/center ratio: {boundary_center_ratio_after:.2f} (expected ~1.0)")

    # Reflection should significantly improve boundary density
    boundary_improvement = density_boundary_reflect / density_boundary_no_reflect
    print(f"    Boundary improvement: {boundary_improvement:.2f}x")

    assert boundary_improvement > 1.5, (
        f"Reflection should improve boundary density by >1.5x, got {boundary_improvement:.2f}x"
    )
    # Key test: boundary density should match center density after reflection
    # (for uniform distribution, they should be equal)
    assert 0.8 < boundary_center_ratio_after < 1.2, (
        f"Reflected boundary should match center density, ratio={boundary_center_ratio_after:.2f}"
    )
    print("  Reflection boundary correction: OK")

    print("\nAll ParticleDensityQuery tests passed!")
    print("\nImplementation Status:")
    print("  - KD-tree spatial indexing")
    print("  - Kernel density estimation (exact)")
    print("  - k-NN density estimation (fast)")
    print("  - Hybrid method (recommended)")
    print("  - Automatic bandwidth selection (Scott, Silverman)")
    print("  - Batch query support")
    print("  - Periodic topology support (Issue #714)")
    print("  - Reflection boundary correction (Issue #709)")
