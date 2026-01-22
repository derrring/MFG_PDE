"""
High-dimensional sampling strategies for Deep Galerkin Methods.

This module provides DGM-specific sampling methods built on top of the
centralized Monte Carlo utilities. It focuses on PDE-specific sampling
patterns like boundary conditions, initial conditions, and physics points.

Sampling Strategies:
- Physics-Informed Sampling: Interior domain sampling for PDE residuals
- Boundary Sampling: Spatial boundary condition enforcement
- Initial Condition Sampling: Time t=0 sampling for initial conditions
- Adaptive Sampling: Residual-based refinement

Mathematical Foundation:
- Physics Loss: L_physics = E[|PDE_residual(t,x)|²]
- Boundary Loss: L_BC = E[|BC_residual(x)|²]
- Initial Loss: L_IC = E[|IC_residual(x)|²]
- Adaptive: Add points where |residual(x)| > threshold
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from mfg_pde.utils.mfg_logging import get_logger

# Import centralized Monte Carlo utilities
from mfg_pde.utils.numerical.particle.sampling import (
    MCConfig,
    MCSampler,
    QuasiMCSampler,
    UniformMCSampler,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = get_logger(__name__)


class HighDimSampler(ABC):
    """Abstract base class for high-dimensional domain sampling."""

    def __init__(self, domain_bounds: list[tuple[float, float]], dimension: int):
        """
        Initialize high-dimensional sampler.

        Args:
            domain_bounds: List of (min, max) bounds for each dimension
            dimension: Spatial dimension of the problem
        """
        self.domain_bounds = domain_bounds
        self.dimension = dimension
        self.total_dimension = dimension + 1  # Including time dimension

        # Validate bounds
        if len(domain_bounds) != dimension:
            raise ValueError(f"Expected {dimension} domain bounds, got {len(domain_bounds)}")

    @abstractmethod
    def sample_interior(self, num_points: int, time_bounds: tuple[float, float]) -> NDArray:
        """
        Sample points in the interior domain.

        Args:
            num_points: Number of points to sample
            time_bounds: (t_min, t_max) time domain

        Returns:
            Sample points of shape (num_points, dimension + 1)
        """

    @abstractmethod
    def sample_boundary(self, num_points: int, time_bounds: tuple[float, float]) -> NDArray:
        """
        Sample points on the spatial boundary.

        Args:
            num_points: Number of points to sample
            time_bounds: (t_min, t_max) time domain

        Returns:
            Boundary points of shape (num_points, dimension + 1)
        """

    def sample_initial(self, num_points: int) -> NDArray:
        """
        Sample points at t=0 for initial conditions.

        Args:
            num_points: Number of points to sample

        Returns:
            Initial points of shape (num_points, dimension)
        """
        # Sample uniformly in spatial domain at t=0
        points = np.zeros((num_points, self.dimension))

        for i, (min_val, max_val) in enumerate(self.domain_bounds):
            points[:, i] = np.random.uniform(min_val, max_val, num_points)

        return points


class MonteCarloSampler(HighDimSampler):
    """Standard Monte Carlo sampling using centralized utilities."""

    def __init__(self, domain_bounds: list[tuple[float, float]], dimension: int, seed: int | None = None):
        """
        Initialize Monte Carlo sampler.

        Args:
            domain_bounds: Domain bounds for each dimension
            dimension: Spatial dimension
            seed: Random seed for reproducibility
        """
        super().__init__(domain_bounds, dimension)

        # Create MC config for centralized utilities
        self.mc_config = MCConfig(seed=seed)

        # Create random number generator for boundary sampling
        self.rng = np.random.default_rng(seed)

        # Initialize space-time domain for centralized sampler
        self.spacetime_domain = None  # Will be set when time bounds are known

    def sample_interior(self, num_points: int, time_bounds: tuple[float, float]) -> NDArray:
        """Sample uniformly in the space-time domain using centralized utilities."""
        # Create space-time domain
        spacetime_domain = [time_bounds, *self.domain_bounds]

        # Use centralized uniform sampler
        sampler = UniformMCSampler(spacetime_domain, self.mc_config)
        points = sampler.sample(num_points)

        return points

    def sample_boundary(self, num_points: int, time_bounds: tuple[float, float]) -> NDArray:
        """Sample on spatial boundary over time."""
        # For simplicity, sample on faces of hypercube boundary
        boundary_points = []
        points_per_face = num_points // (2 * self.dimension)  # Points per boundary face

        t_min, t_max = time_bounds

        for dim in range(self.dimension):
            for boundary_val in [self.domain_bounds[dim][0], self.domain_bounds[dim][1]]:
                face_points = np.zeros((points_per_face, self.total_dimension))

                # Time dimension
                face_points[:, 0] = self.rng.uniform(t_min, t_max, points_per_face)

                # Spatial dimensions
                for i, (min_val, max_val) in enumerate(self.domain_bounds):
                    if i == dim:
                        face_points[:, i + 1] = boundary_val  # On boundary face
                    else:
                        face_points[:, i + 1] = self.rng.uniform(min_val, max_val, points_per_face)

                boundary_points.append(face_points)

        return np.vstack(boundary_points)


class QuasiMonteCarloSampler(HighDimSampler):
    """Quasi-Monte Carlo sampling using centralized utilities."""

    def __init__(self, domain_bounds: list[tuple[float, float]], dimension: int, sequence_type: str = "sobol"):
        """
        Initialize Quasi-Monte Carlo sampler.

        Args:
            domain_bounds: Domain bounds for each dimension
            dimension: Spatial dimension
            sequence_type: "sobol" | "halton" | "latin_hypercube"
        """
        super().__init__(domain_bounds, dimension)
        self.sequence_type = sequence_type

        # Create MC config for centralized utilities
        self.mc_config = MCConfig(sampling_method=sequence_type)

        # Create random number generator for fallback methods
        self.rng = np.random.default_rng()

    def sample_interior(self, num_points: int, time_bounds: tuple[float, float]) -> NDArray:
        """Sample using low-discrepancy sequence via centralized utilities."""
        # Create space-time domain
        spacetime_domain = [time_bounds, *self.domain_bounds]

        # Use centralized quasi-MC sampler
        sampler: MCSampler
        try:
            sampler = QuasiMCSampler(spacetime_domain, self.mc_config, self.sequence_type)
            points = sampler.sample(num_points)
            return points
        except (ImportError, ValueError, NotImplementedError, RuntimeError) as e:
            # Issue #547: Quasi-MC can fail if scipy.stats.qmc unavailable or sequence type not supported
            logger.warning(
                "Quasi-MC sampling failed (%s): %s. Using uniform fallback (performance may be degraded).",
                type(e).__name__,
                e,
            )
            sampler = UniformMCSampler(spacetime_domain, self.mc_config)
            return sampler.sample(num_points)

    def sample_boundary(self, num_points: int, time_bounds: tuple[float, float]) -> NDArray:
        """Sample boundary using stratified approach."""
        # Simplified boundary sampling (can be enhanced with QMC)
        return self._monte_carlo_boundary_fallback(num_points, time_bounds)

    def _monte_carlo_fallback(self, num_points: int, time_bounds: tuple[float, float]) -> NDArray:
        """Fallback to standard Monte Carlo if QMC unavailable."""
        points = np.zeros((num_points, self.total_dimension))

        t_min, t_max = time_bounds
        points[:, 0] = self.rng.uniform(t_min, t_max, num_points)

        for i, (min_val, max_val) in enumerate(self.domain_bounds):
            points[:, i + 1] = self.rng.uniform(min_val, max_val, num_points)

        return points

    def _monte_carlo_boundary_fallback(self, num_points: int, time_bounds: tuple[float, float]) -> NDArray:
        """Fallback boundary sampling."""
        # Use simple Monte Carlo boundary sampling
        mc_sampler = MonteCarloSampler(self.domain_bounds, self.dimension)
        return mc_sampler.sample_boundary(num_points, time_bounds)


def adaptive_sampling(
    residual_function: callable,
    current_points: NDArray,
    threshold: float,
    max_new_points: int,
    domain_bounds: list[tuple[float, float]],
) -> NDArray:
    """
    Generate additional sample points based on residual analysis.

    Args:
        residual_function: Function that computes residuals at points
        current_points: Existing sample points
        threshold: Residual threshold for adding points
        max_new_points: Maximum number of new points to add
        domain_bounds: Domain bounds for new point generation

    Returns:
        New sample points for high-residual regions
    """
    # Evaluate residuals at current points
    residuals = residual_function(current_points)
    high_residual_indices = np.where(np.abs(residuals) > threshold)[0]

    if len(high_residual_indices) == 0:
        logger.info("No high-residual points found, no adaptive sampling needed")
        return np.array([]).reshape(0, current_points.shape[1])

    # Generate new points around high-residual regions
    high_residual_points = current_points[high_residual_indices]
    num_new_points = min(max_new_points, len(high_residual_points) * 5)

    # Add noise around high-residual points
    len(domain_bounds)
    new_points = []

    for point in high_residual_points[: num_new_points // 5]:
        # Generate 5 points around each high-residual point
        for _ in range(5):
            new_point = point.copy()

            # Add small random perturbation
            for i, (min_val, max_val) in enumerate(domain_bounds):
                noise_scale = (max_val - min_val) * 0.01  # 1% of domain size
                new_point[i + 1] += np.random.normal(0, noise_scale)

                # Ensure bounds are respected
                new_point[i + 1] = np.clip(new_point[i + 1], min_val, max_val)

            new_points.append(new_point)

    logger.info(f"Adaptive sampling added {len(new_points)} points near high-residual regions")
    return np.array(new_points) if new_points else np.array([]).reshape(0, current_points.shape[1])
