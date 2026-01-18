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

References:
    - Silverman (1986): Density Estimation for Statistics and Data Analysis
    - Scott (1992): Multivariate Density Estimation
    - scipy.spatial.KDTree documentation

Created: 2026-01-18 (Issue #489 Phase 1 - Core Query Infrastructure)
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

        Raises
        ------
        ValueError
            If bandwidth is None and bandwidth_rule is "fixed".
        """
        self.particles = particles
        self.N_particles, self.dimension = particles.shape

        # Build KD-tree for fast nearest neighbor queries
        self.tree = KDTree(particles)
        logger.debug(f"Built KD-tree for {self.N_particles} particles in {self.dimension}D")

        # Bandwidth selection
        if bandwidth is None:
            if bandwidth_rule == "fixed":
                raise ValueError("bandwidth must be provided when bandwidth_rule='fixed'")
            self.bandwidth = self._compute_bandwidth(bandwidth_rule)
            logger.info(f"Auto-selected bandwidth: h={self.bandwidth:.4f} ({bandwidth_rule} rule)")
        else:
            self.bandwidth = bandwidth
            logger.debug(f"Using fixed bandwidth: h={bandwidth:.4f}")

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
        """
        N_query = query_points.shape[0]
        densities = np.zeros(N_query)

        h = self.bandwidth
        normalization = 1.0 / (self.N_particles * h**self.dimension)

        for i, x in enumerate(query_points):
            # Compute distances to all particles
            distances = np.linalg.norm(self.particles - x, axis=1)

            # Gaussian kernel: K(r) = exp(-r²/2)
            kernel_values = np.exp(-0.5 * (distances / h) ** 2)

            # Density estimate
            densities[i] = normalization * np.sum(kernel_values)

        return densities

    def _query_knn(self, query_points: NDArray[np.float64], k: int) -> NDArray[np.float64]:
        """
        k-nearest neighbors density estimation.

        Density proportional to inverse volume containing k neighbors.
        Cost: O(N_query × log N_particles)
        """
        N_query = query_points.shape[0]
        densities = np.zeros(N_query)

        for i, x in enumerate(query_points):
            # Find k nearest neighbors
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

            # Density: k neighbors in volume
            densities[i] = k / (self.N_particles * volume) if volume > 1e-12 else 0.0

        return densities

    def _query_hybrid(self, query_points: NDArray[np.float64], k: int) -> NDArray[np.float64]:
        """
        Hybrid: Gaussian kernel over k-nearest neighbors.

        Balances smoothness (kernel) with efficiency (k-NN subset).
        Cost: O(N_query × (log N_particles + k))
        """
        N_query = query_points.shape[0]
        densities = np.zeros(N_query)

        h = self.bandwidth
        normalization = 1.0 / (self.N_particles * h**self.dimension)

        for i, x in enumerate(query_points):
            # Find k nearest neighbors
            distances, _indices = self.tree.query(x, k=k)

            # Gaussian kernel on nearby particles only
            kernel_values = np.exp(-0.5 * (distances / h) ** 2)

            # Density estimate (scaled by full population)
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

    print("\n✅ All ParticleDensityQuery tests passed!")
    print("\n📊 Implementation Status:")
    print("  ✓ KD-tree spatial indexing")
    print("  ✓ Kernel density estimation (exact)")
    print("  ✓ k-NN density estimation (fast)")
    print("  ✓ Hybrid method (recommended)")
    print("  ✓ Automatic bandwidth selection (Scott, Silverman)")
    print("  ✓ Batch query support")
