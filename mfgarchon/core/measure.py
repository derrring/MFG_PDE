"""
Measure representations for MFG on measure space (Layer 2).

Provides the ``MeasureRepresentation`` Protocol and concrete implementations
for representing probability measures mu in P_2(R^d). These are the building
blocks for:
- Master Equation: v(x, mu, t) where mu is a measure argument
- Common Noise PDE: conditional measures mu^theta evolving in measure space
- Lions derivative: delta U / delta m computed via measure perturbation

Design reference:
    Joplin Dev: "Generalized PDE & Institutional MFG Plan" Layer 2 section
    Joplin Dev: "Layer 2 Readiness Assessment"
    Issue #956: Layer 2 alignment

Current implementations:
    - ParticleMeasure: empirical measure m_N = (1/N) sum_i w_i delta_{y_i}

Planned (not yet implemented):
    - BasisExpansionMeasure: density expanded in basis (d <= 2)
    - NeuralMeasure: neural network parametrization (high-dimensional)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@runtime_checkable
class MeasureRepresentation(Protocol):
    """Protocol for probability measure representations.

    Any object satisfying this protocol can be used as a measure argument
    in MFG solvers operating on measure space (Layer 2).

    Methods:
        to_density: Project measure onto a spatial grid as density array.
        total_mass: Return total mass (should be 1.0 for probability measures).
        dimension: Spatial dimension of the measure support.
    """

    def to_density(self, grid_points: NDArray) -> NDArray:
        """Project measure onto spatial grid as density array.

        Args:
            grid_points: Spatial grid, shape (N,) for 1D or (N, d) for dD.

        Returns:
            Density values at grid points, shape (N,).
        """
        ...

    def total_mass(self) -> float:
        """Return total mass of the measure."""
        ...

    @property
    def dimension(self) -> int:
        """Spatial dimension of the measure support."""
        ...


class ParticleMeasure:
    """Empirical measure: m_N = sum_i w_i delta_{y_i}.

    Represents a probability measure as a weighted sum of Dirac masses
    at particle positions. This is the natural representation for:
    - Monte Carlo methods (common noise sampling)
    - N-player game convergence
    - Particle-based FP solvers
    - Functional derivative computation via position perturbation

    Parameters
    ----------
    positions : NDArray
        Particle positions, shape (N,) for 1D or (N, d) for dD.
    weights : NDArray | None
        Particle weights, shape (N,). Default: uniform 1/N.

    Example
    -------
    >>> mu = ParticleMeasure(np.array([0.2, 0.5, 0.8]))
    >>> density = mu.to_density(np.linspace(0, 1, 101))
    >>> mu.total_mass()  # 1.0
    """

    def __init__(self, positions: NDArray, weights: NDArray | None = None):
        self._positions = np.asarray(positions, dtype=float)
        if self._positions.ndim == 1:
            self._positions = self._positions.reshape(-1, 1)
        self._n_particles = self._positions.shape[0]
        self._dim = self._positions.shape[1]

        if weights is None:
            self._weights = np.ones(self._n_particles) / self._n_particles
        else:
            self._weights = np.asarray(weights, dtype=float)
            self._weights = self._weights / self._weights.sum()

    @property
    def positions(self) -> NDArray:
        """Particle positions, shape (N, d)."""
        return self._positions

    @property
    def weights(self) -> NDArray:
        """Particle weights, shape (N,), sum to 1."""
        return self._weights

    @property
    def n_particles(self) -> int:
        """Number of particles."""
        return self._n_particles

    @property
    def dimension(self) -> int:
        """Spatial dimension."""
        return self._dim

    def total_mass(self) -> float:
        """Total mass (should be 1.0 for probability measure)."""
        return float(self._weights.sum())

    def to_density(self, grid_points: NDArray, bandwidth: float | None = None) -> NDArray:
        """Project onto spatial grid via kernel density estimation.

        Uses Gaussian KDE with bandwidth selected by Scott's rule if not specified.

        Args:
            grid_points: Grid points, shape (M,) for 1D or (M, d) for dD.
            bandwidth: KDE bandwidth. None = Scott's rule: h = N^{-1/(d+4)} * std.

        Returns:
            Density values at grid points, shape (M,).
        """
        grid = np.asarray(grid_points, dtype=float)
        if grid.ndim == 1:
            grid = grid.reshape(-1, 1)

        if self._dim == 1:
            return self._kde_1d(grid[:, 0], bandwidth)

        # General dD KDE
        return self._kde_nd(grid, bandwidth)

    def _kde_1d(self, x: NDArray, bandwidth: float | None) -> NDArray:
        """1D kernel density estimation."""
        positions = self._positions[:, 0]
        if bandwidth is None:
            std = np.std(positions)
            bandwidth = std * self._n_particles ** (-1 / 5) if std > 0 else 0.1

        # Gaussian kernel: K(u) = (1/sqrt(2*pi)) * exp(-u^2/2)
        # density(x) = sum_i w_i * K((x - y_i) / h) / h
        diff = x[:, None] - positions[None, :]  # (M, N)
        kernel_vals = np.exp(-0.5 * (diff / bandwidth) ** 2) / (bandwidth * np.sqrt(2 * np.pi))
        return kernel_vals @ self._weights  # (M,)

    def _kde_nd(self, x: NDArray, bandwidth: float | None) -> NDArray:
        """General dD kernel density estimation (product kernel)."""
        d = self._dim
        if bandwidth is None:
            std = np.std(self._positions, axis=0).mean()
            bandwidth = std * self._n_particles ** (-1 / (d + 4)) if std > 0 else 0.1

        M = x.shape[0]
        density = np.zeros(M)
        for i in range(self._n_particles):
            diff = x - self._positions[i]  # (M, d)
            sq_dist = np.sum(diff**2, axis=1)  # (M,)
            kernel_vals = np.exp(-0.5 * sq_dist / bandwidth**2) / (bandwidth * np.sqrt(2 * np.pi)) ** d
            density += self._weights[i] * kernel_vals
        return density

    def __repr__(self) -> str:
        return f"ParticleMeasure(n={self._n_particles}, d={self._dim})"
