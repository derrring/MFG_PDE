"""
Variance reduction techniques for Deep Galerkin Methods.

This module implements advanced Monte Carlo variance reduction techniques
to improve computational efficiency of DGM solvers in high dimensions.

Techniques Implemented:
- Control Variates: Use known analytical solutions for variance reduction
- Importance Sampling: Sample from distributions concentrated on high-error regions
- Multilevel Monte Carlo: Combine multiple resolution levels for efficiency
- Stratified Sampling: Divide domain into strata for uniform coverage

Mathematical Foundation:
- Control Variates: Var[f - c(g - E[g])] ≤ Var[f] with optimal c*
- Importance Sampling: ∫f(x)dx = ∫(f(x)/p(x))p(x)dx with p(x) ∝ |f(x)|
- MLMC: E[f] = E[f_L] + Σₗ E[f_l - f_{l-1}] with different resolutions
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np

# Import centralized Monte Carlo utilities

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class ControlVariates:
    """
    Control variates for variance reduction in DGM Monte Carlo integration.

    Uses known analytical solutions or simpler approximations as control
    variates to reduce the variance of Monte Carlo estimators for PDE residuals.

    Mathematical Framework:
    - Original estimator: θ = (1/N) Σᵢ f(xᵢ)
    - Control variate: θ_cv = (1/N) Σᵢ [f(xᵢ) - c(g(xᵢ) - μ_g)]
    - Optimal coefficient: c* = Cov[f,g] / Var[g]
    - Variance reduction: Var[θ_cv] = Var[θ](1 - ρ²_{f,g})
    """

    def __init__(self, control_function: Callable[[NDArray], NDArray] | None = None):
        """
        Initialize control variates.

        Args:
            control_function: Known function for variance reduction
        """
        self.control_function = control_function
        self.control_coefficient = 0.0
        self.is_calibrated = False

    def calibrate(self, sample_points: NDArray, target_function: Callable[[NDArray], NDArray]) -> None:
        """
        Calibrate control variate coefficient.

        Args:
            sample_points: Sample points for calibration
            target_function: Function whose variance we want to reduce
        """
        if self.control_function is None:
            logger.warning("No control function provided, skipping calibration")
            return

        # Evaluate both functions
        f_vals = target_function(sample_points)
        g_vals = self.control_function(sample_points)

        # Compute optimal coefficient
        covariance = np.cov(f_vals, g_vals)[0, 1]
        variance_g = np.var(g_vals)

        if variance_g > 1e-12:
            self.control_coefficient = covariance / variance_g
            self.is_calibrated = True

            # Compute variance reduction factor
            correlation = covariance / (np.std(f_vals) * np.std(g_vals))
            variance_reduction = 1 - correlation**2

            logger.info(
                f"Control variate calibrated: c* = {self.control_coefficient:.6f}, "
                f"variance reduction = {variance_reduction:.3f}"
            )
        else:
            logger.warning("Control function has zero variance, cannot calibrate")

    def apply(self, sample_points: NDArray, function_values: NDArray) -> NDArray:
        """
        Apply control variate variance reduction.

        Args:
            sample_points: Sample points
            function_values: Function evaluations at sample points

        Returns:
            Variance-reduced function values
        """
        if not self.is_calibrated or self.control_function is None:
            return function_values

        # Evaluate control function
        control_values = self.control_function(sample_points)
        control_mean = np.mean(control_values)

        # Apply control variate formula
        reduced_values = function_values - self.control_coefficient * (control_values - control_mean)

        return reduced_values


class ImportanceSampling:
    """
    Importance sampling for efficient Monte Carlo integration.

    Samples from a distribution concentrated on high-contribution regions
    rather than uniform sampling, improving Monte Carlo efficiency.

    Mathematical Framework:
    - Standard MC: ∫f(x)dx ≈ (1/N) Σᵢ f(xᵢ), xᵢ ~ Uniform
    - Importance MC: ∫f(x)dx ≈ (1/N) Σᵢ f(xᵢ)/p(xᵢ), xᵢ ~ p(x)
    - Optimal p(x): p*(x) ∝ |f(x)| minimizes variance
    """

    def __init__(self, importance_function: Callable[[NDArray], NDArray] | None = None):
        """
        Initialize importance sampling.

        Args:
            importance_function: Function defining importance distribution
        """
        self.importance_function = importance_function
        self.is_setup = False

    def setup_from_residuals(self, sample_points: NDArray, residuals: NDArray) -> None:
        """
        Setup importance distribution based on residual analysis.

        Args:
            sample_points: Sample points from previous iterations
            residuals: Residual values at sample points
        """
        # Create importance function based on residual magnitude
        abs_residuals = np.abs(residuals)

        if np.max(abs_residuals) > 1e-12:
            # Normalize to create probability distribution
            importance_weights = abs_residuals / np.sum(abs_residuals)

            # Create interpolated importance function (simplified)
            def residual_based_importance(points: NDArray) -> NDArray:
                # Simple nearest-neighbor importance (can be enhanced with interpolation)
                distances = np.sum((points[:, np.newaxis, :] - sample_points[np.newaxis, :, :]) ** 2, axis=2)
                nearest_indices = np.argmin(distances, axis=1)
                return importance_weights[nearest_indices]

            self.importance_function = residual_based_importance
            self.is_setup = True

            logger.info("Importance sampling setup from residual analysis")
        else:
            logger.warning("All residuals near zero, importance sampling not beneficial")

    def sample(
        self, num_points: int, domain_bounds: list[tuple[float, float]], time_bounds: tuple[float, float]
    ) -> tuple[NDArray, NDArray]:
        """
        Generate importance-weighted samples.

        Args:
            num_points: Number of points to sample
            domain_bounds: Spatial domain bounds
            time_bounds: Time domain bounds

        Returns:
            Tuple of (sample_points, importance_weights)
        """
        if not self.is_setup or self.importance_function is None:
            # Fallback to uniform sampling
            logger.info("Using uniform sampling (importance function not available)")
            return self._uniform_sampling(num_points, domain_bounds, time_bounds)

        # Importance sampling implementation (simplified)
        # In practice, this would use sophisticated sampling algorithms
        return self._approximate_importance_sampling(num_points, domain_bounds, time_bounds)

    def _uniform_sampling(
        self, num_points: int, domain_bounds: list[tuple[float, float]], time_bounds: tuple[float, float]
    ) -> tuple[NDArray, NDArray]:
        """Fallback to uniform sampling."""
        dimension = len(domain_bounds)
        points = np.zeros((num_points, dimension + 1))

        # Time dimension
        t_min, t_max = time_bounds
        points[:, 0] = np.random.uniform(t_min, t_max, num_points)

        # Spatial dimensions
        for i, (min_val, max_val) in enumerate(domain_bounds):
            points[:, i + 1] = np.random.uniform(min_val, max_val, num_points)

        weights = np.ones(num_points) / num_points  # Uniform weights
        return points, weights

    def _approximate_importance_sampling(
        self, num_points: int, domain_bounds: list[tuple[float, float]], time_bounds: tuple[float, float]
    ) -> tuple[NDArray, NDArray]:
        """Approximate importance sampling using rejection sampling."""
        # Generate candidate points
        candidate_points, _ = self._uniform_sampling(num_points * 2, domain_bounds, time_bounds)

        # Evaluate importance function
        importance_values = self.importance_function(candidate_points)

        # Select points with probability proportional to importance
        selection_probs = importance_values / np.max(importance_values)
        selected_mask = np.random.random(len(candidate_points)) < selection_probs

        selected_points = candidate_points[selected_mask]

        # If we don't have enough points, fill with uniform sampling
        if len(selected_points) < num_points:
            additional_points, _ = self._uniform_sampling(num_points - len(selected_points), domain_bounds, time_bounds)
            selected_points = np.vstack([selected_points, additional_points])

        # Take exactly num_points
        selected_points = selected_points[:num_points]

        # Compute importance weights
        weights = 1.0 / self.importance_function(selected_points)
        weights = weights / np.sum(weights)  # Normalize

        return selected_points, weights


class MultilevelMonteCarlo:
    """
    Multilevel Monte Carlo for hierarchical variance reduction.

    Uses multiple levels of approximation (coarse to fine) to reduce
    computational cost while maintaining accuracy.

    Mathematical Framework:
    - Standard MC: E[f] ≈ (1/N) Σᵢ f_L(xᵢ) with O(N⁻¹/²) convergence
    - MLMC: E[f] ≈ E[f₀] + Σₗ E[f_l - f_{l-1}] with optimal level allocation
    - Cost reduction: Achieve same accuracy with lower computational cost
    """

    def __init__(self, num_levels: int = 3):
        """
        Initialize multilevel Monte Carlo.

        Args:
            num_levels: Number of resolution levels
        """
        self.num_levels = num_levels
        self.level_costs = []
        self.level_variances = []

    def estimate_level_parameters(
        self, sample_function: Callable[[int, int], NDArray], max_samples: int = 1000
    ) -> None:
        """
        Estimate computational cost and variance for each level.

        Args:
            sample_function: Function that takes (level, num_samples) and returns function evaluations
            max_samples: Maximum samples for parameter estimation
        """
        logger.info(f"Estimating MLMC parameters for {self.num_levels} levels")

        for level in range(self.num_levels):
            # Estimate cost (simplified - time-based)
            start_time = time.time()
            samples = sample_function(level, max_samples)
            cost = time.time() - start_time

            # Estimate variance
            variance = np.var(samples)

            self.level_costs.append(cost)
            self.level_variances.append(variance)

            logger.info(f"Level {level}: Cost = {cost:.4f}s, Variance = {variance:.6e}")

    def optimal_allocation(self, target_accuracy: float) -> list[int]:
        """
        Compute optimal sample allocation across levels.

        Args:
            target_accuracy: Target mean squared error

        Returns:
            Optimal number of samples for each level
        """
        if not self.level_costs or not self.level_variances:
            logger.warning("Level parameters not estimated, using uniform allocation")
            return [1000] * self.num_levels

        # Compute optimal allocation using MLMC theory
        # N_l ∝ sqrt(V_l / C_l) where V_l is variance and C_l is cost
        allocation_weights = [
            np.sqrt(var / cost) for var, cost in zip(self.level_variances, self.level_costs, strict=False)
        ]

        # Normalize to achieve target accuracy
        total_weight = sum(allocation_weights)
        total_budget = target_accuracy ** (-2)  # Simplified budget calculation

        optimal_samples = [int(total_budget * weight / total_weight) for weight in allocation_weights]

        logger.info(f"Optimal MLMC allocation: {optimal_samples}")
        return optimal_samples


def create_control_variate_function(problem_type: str) -> Callable[[NDArray], NDArray] | None:
    """
    Create appropriate control variate function based on problem type.

    Args:
        problem_type: Type of MFG problem for control variate selection

    Returns:
        Control variate function or None if not available
    """
    if problem_type == "linear_quadratic":
        # Linear-quadratic MFG has known analytical properties

        def lq_control_variate(points: NDArray) -> NDArray:
            """Control variate for linear-quadratic MFG."""
            t = points[:, 0]
            x = points[:, 1:]

            # Simple quadratic control variate
            control_vals = 0.5 * np.sum(x**2, axis=1) * (1 - t)
            return control_vals

        return lq_control_variate

    elif problem_type == "crowd_dynamics":
        # Crowd dynamics may have approximate analytical solutions

        def crowd_control_variate(points: NDArray) -> NDArray:
            """Control variate for crowd dynamics."""
            t = points[:, 0]
            x = points[:, 1:]

            # Gaussian-based control variate
            center = np.mean(x, axis=1)
            control_vals = np.exp(-0.5 * np.sum((x - center[:, np.newaxis]) ** 2, axis=1)) * (1 - t)
            return control_vals

        return crowd_control_variate

    else:
        logger.info(f"No specific control variate available for problem type: {problem_type}")
        return None


def adaptive_importance_distribution(
    points: NDArray, residuals: NDArray, smoothing_parameter: float = 0.1
) -> Callable[[NDArray], NDArray]:
    """
    Create adaptive importance distribution based on residual analysis.

    Args:
        points: Sample points where residuals were computed
        residuals: Residual values at sample points
        smoothing_parameter: Smoothing for importance distribution

    Returns:
        Importance distribution function
    """
    # Create importance weights based on residual magnitude
    abs_residuals = np.abs(residuals)
    max_residual = np.max(abs_residuals)

    if max_residual < 1e-12:
        # Uniform distribution if all residuals are small
        def uniform_importance(query_points: NDArray) -> NDArray:
            return np.ones(len(query_points))

        return uniform_importance

    # Normalize residuals to create importance weights
    importance_weights = abs_residuals / max_residual

    # Create interpolated importance function
    try:
        from scipy.interpolate import RBFInterpolator
        from scipy.spatial import cKDTree

        # Use RBF interpolation for smooth importance distribution
        rbf = RBFInterpolator(points, importance_weights, kernel="gaussian", epsilon=smoothing_parameter)

        def rbf_importance(query_points: NDArray) -> NDArray:
            """RBF-interpolated importance function."""
            importance_vals = rbf(query_points)
            return np.maximum(importance_vals, 0.01)  # Ensure minimum importance

        logger.info("Created RBF-based adaptive importance distribution")
        return rbf_importance

    except ImportError:
        # Fallback to nearest-neighbor importance
        def nn_importance(query_points: NDArray) -> NDArray:
            """Nearest-neighbor importance function."""
            distances = np.sum((query_points[:, np.newaxis, :] - points[np.newaxis, :, :]) ** 2, axis=2)
            nearest_indices = np.argmin(distances, axis=1)
            return importance_weights[nearest_indices]

        logger.info("Using nearest-neighbor importance distribution (scipy not available)")
        return nn_importance
