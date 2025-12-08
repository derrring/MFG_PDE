"""
Monte Carlo utilities for MFG computations.

This module provides reusable Monte Carlo methods used across different
paradigms (numerical, neural, optimization, RL) for high-dimensional
integration, sampling, and variance reduction.

Key Components:
- Sampling strategies (uniform, stratified, importance, adaptive)
- Variance reduction techniques (control variates, antithetic variables)
- Low-discrepancy sequences (Sobol, Halton, Latin hypercube)
- High-dimensional integration methods
- Convergence diagnostics and error estimation

Mathematical Foundation:
- Standard MC: ∫f(x)dx ≈ (1/N) Σᵢ f(xᵢ), xᵢ ~ p(x)
- Importance MC: ∫f(x)dx = ∫(f(x)/p(x))p(x)dx
- Control Variates: Var[f - c(g - E[g])] ≤ Var[f]
- Stratified: Divide domain Ω = ∪Ωᵢ, sample within each stratum
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# Type definitions for Monte Carlo integration
MCIntegrand = Callable[[NDArray], NDArray]
MCWeightFunction = Callable[[NDArray], NDArray]
MCDomain = list[tuple[float, float]]


@dataclass
class MCConfig:
    """Configuration for Monte Carlo computations."""

    # Sampling parameters
    num_samples: int = 10000
    sampling_method: str = "uniform"  # "uniform" | "stratified" | "importance" | "sobol" | "halton"
    seed: int | None = None

    # Variance reduction
    use_control_variates: bool = False
    use_antithetic_variables: bool = False
    control_function: Callable[[NDArray], NDArray] | None = None

    # Adaptive sampling
    adaptive: bool = False
    error_tolerance: float = 1e-3
    max_adaptive_samples: int = 100000

    # Stratification (for stratified sampling)
    num_strata_per_dim: int = 10

    # Importance sampling
    importance_function: Callable[[NDArray], NDArray] | None = None
    proposal_distribution: str = "uniform"  # "uniform" | "gaussian" | "custom"

    # Convergence monitoring
    batch_size: int = 1000
    confidence_level: float = 0.95

    # Computational
    parallel: bool = False
    chunk_size: int = 10000


@dataclass
class MCResult:
    """Result container for Monte Carlo computations."""

    # Estimation results
    estimate: float | NDArray
    standard_error: float | NDArray
    confidence_interval: tuple[float, float] | tuple[NDArray, NDArray]

    # Sampling information
    num_samples_used: int
    effective_sample_size: int
    variance_reduction_factor: float = 1.0

    # Convergence diagnostics
    converged: bool = False
    final_error: float = np.inf
    iterations: int = 0

    # Performance metrics
    computation_time: float = 0.0
    samples_per_second: float = 0.0


class MCSampler(ABC):
    """Abstract base class for Monte Carlo samplers."""

    def __init__(self, domain: MCDomain, config: MCConfig):
        """
        Initialize Monte Carlo sampler.

        Args:
            domain: List of (min, max) bounds for each dimension
            config: Monte Carlo configuration
        """
        self.domain = domain
        self.config = config
        self.dimension = len(domain)

        # Initialize random number generator
        self.rng = np.random.RandomState(config.seed)

        # Precompute domain properties
        self.domain_bounds = np.array(domain)
        self.domain_volume = np.prod(self.domain_bounds[:, 1] - self.domain_bounds[:, 0])

    @abstractmethod
    def sample(self, num_samples: int) -> NDArray:
        """
        Generate sample points in the domain.

        Args:
            num_samples: Number of samples to generate

        Returns:
            Sample points of shape (num_samples, dimension)
        """

    def get_weights(self, points: NDArray) -> NDArray:
        """
        Get importance weights for sampled points.

        Args:
            points: Sample points

        Returns:
            Importance weights (default: uniform)
        """
        return np.ones(len(points)) / len(points)


class UniformMCSampler(MCSampler):
    """Uniform Monte Carlo sampler."""

    def sample(self, num_samples: int) -> NDArray:
        """Generate uniform random samples."""
        samples = np.zeros((num_samples, self.dimension))

        for i, (min_val, max_val) in enumerate(self.domain):
            samples[:, i] = self.rng.uniform(min_val, max_val, num_samples)

        return samples


class StratifiedMCSampler(MCSampler):
    """Stratified Monte Carlo sampler for variance reduction."""

    def sample(self, num_samples: int) -> NDArray:
        """Generate stratified samples."""
        strata_per_dim = self.config.num_strata_per_dim
        total_strata = strata_per_dim**self.dimension
        samples_per_stratum = max(1, num_samples // total_strata)

        samples = []

        # Generate stratified samples
        for stratum_idx in range(total_strata):
            # Convert linear index to multi-dimensional stratum coordinates
            stratum_coords = np.unravel_index(stratum_idx, (strata_per_dim,) * self.dimension)

            # Sample within this stratum
            stratum_samples = np.zeros((samples_per_stratum, self.dimension))

            for dim in range(self.dimension):
                min_val, max_val = self.domain[dim]
                stratum_width = (max_val - min_val) / strata_per_dim

                stratum_min = min_val + stratum_coords[dim] * stratum_width
                stratum_max = stratum_min + stratum_width

                stratum_samples[:, dim] = self.rng.uniform(stratum_min, stratum_max, samples_per_stratum)

            samples.append(stratum_samples)

        all_samples = np.vstack(samples)

        # Truncate to exact number if needed
        return all_samples[:num_samples]


class QuasiMCSampler(MCSampler):
    """Quasi-Monte Carlo sampler using low-discrepancy sequences."""

    def __init__(self, domain: MCDomain, config: MCConfig, sequence_type: str = "sobol"):
        """
        Initialize quasi-Monte Carlo sampler.

        Args:
            domain: Domain bounds
            config: MC configuration
            sequence_type: "sobol" | "halton" | "latin_hypercube"
        """
        super().__init__(domain, config)
        self.sequence_type = sequence_type
        self._init_sequence_generator()

    def _init_sequence_generator(self):
        """Initialize low-discrepancy sequence generator."""
        try:
            from scipy.stats import qmc

            if self.sequence_type == "sobol":
                self.qmc_sampler = qmc.Sobol(d=self.dimension, scramble=True, seed=self.config.seed)
            elif self.sequence_type == "halton":
                self.qmc_sampler = qmc.Halton(d=self.dimension, scramble=True, seed=self.config.seed)
            elif self.sequence_type == "latin_hypercube":
                self.qmc_sampler = qmc.LatinHypercube(d=self.dimension, seed=self.config.seed)
            else:
                raise ValueError(f"Unknown sequence type: {self.sequence_type}")

            self.scipy_available = True

        except ImportError:
            logger.warning("SciPy QMC not available, falling back to uniform sampling")
            self.scipy_available = False

    def sample(self, num_samples: int) -> NDArray:
        """Generate quasi-random samples."""
        if not self.scipy_available:
            # Fallback to uniform sampling
            return UniformMCSampler(self.domain, self.config).sample(num_samples)

        # Generate unit cube samples
        unit_samples = self.qmc_sampler.random(n=num_samples)

        # Transform to actual domain
        samples = np.zeros_like(unit_samples)
        for i, (min_val, max_val) in enumerate(self.domain):
            samples[:, i] = min_val + (max_val - min_val) * unit_samples[:, i]

        return samples


class ImportanceMCSampler(MCSampler):
    """Importance sampling Monte Carlo sampler."""

    def __init__(self, domain: MCDomain, config: MCConfig, importance_function: Callable[[NDArray], NDArray]):
        """
        Initialize importance sampling.

        Args:
            domain: Domain bounds
            config: MC configuration
            importance_function: Importance distribution p(x)
        """
        super().__init__(domain, config)
        self.importance_function = importance_function

    def sample(self, num_samples: int) -> NDArray:
        """Generate importance-weighted samples using rejection sampling."""
        # Use rejection sampling (simplified implementation)
        # In practice, more sophisticated methods would be used

        candidate_samples = num_samples * 3  # Generate extra candidates
        candidates = UniformMCSampler(self.domain, self.config).sample(candidate_samples)

        # Evaluate importance function
        importance_values = self.importance_function(candidates)
        max_importance = np.max(importance_values)

        # Rejection sampling
        accept_probs = importance_values / max_importance
        accepted_mask = self.rng.random(candidate_samples) < accept_probs

        accepted_samples = candidates[accepted_mask]

        # If not enough samples, fill with uniform
        if len(accepted_samples) < num_samples:
            additional_needed = num_samples - len(accepted_samples)
            additional_samples = UniformMCSampler(self.domain, self.config).sample(additional_needed)
            accepted_samples = np.vstack([accepted_samples, additional_samples])

        return accepted_samples[:num_samples]

    def get_weights(self, points: NDArray) -> NDArray:
        """Get importance weights (inverse of importance function)."""
        importance_vals = self.importance_function(points)
        weights = 1.0 / np.maximum(importance_vals, 1e-12)
        return weights / np.sum(weights)  # Normalize


class ControlVariates:
    """Control variates for variance reduction."""

    def __init__(self, control_function: Callable[[NDArray], NDArray]):
        """
        Initialize control variates.

        Args:
            control_function: Known function g(x) with known expectation
        """
        self.control_function = control_function
        self.control_coefficient = 0.0
        self.control_mean = 0.0
        self.is_calibrated = False

    def calibrate(self, sample_points: NDArray, target_values: NDArray, control_expectation: float | None = None):
        """
        Calibrate control variate coefficient.

        Args:
            sample_points: Sample points for calibration
            target_values: Target function values f(x)
            control_expectation: Known E[g] (estimated if None)
        """
        control_values = self.control_function(sample_points)

        if control_expectation is None:
            self.control_mean = np.mean(control_values)
        else:
            self.control_mean = control_expectation

        # Compute optimal coefficient: c* = Cov[f,g] / Var[g]
        covariance = np.cov(target_values, control_values)[0, 1]
        control_variance = np.var(control_values)

        if control_variance > 1e-12:
            self.control_coefficient = covariance / control_variance
            self.is_calibrated = True

            # Compute variance reduction factor
            correlation = covariance / (np.std(target_values) * np.std(control_values))
            variance_reduction = max(0, 1 - correlation**2)

            logger.info(f"Control variate calibrated: c* = {self.control_coefficient:.6f}")
            logger.info(f"Expected variance reduction: {variance_reduction:.3f}")
        else:
            logger.warning("Control function has zero variance")

    def apply(self, sample_points: NDArray, target_values: NDArray) -> NDArray:
        """Apply control variate variance reduction."""
        if not self.is_calibrated:
            return target_values

        control_values = self.control_function(sample_points)
        reduced_values = target_values - self.control_coefficient * (control_values - self.control_mean)

        return reduced_values


def monte_carlo_integrate(integrand: MCIntegrand, domain: MCDomain, config: MCConfig | None = None) -> MCResult:
    """
    Monte Carlo integration of a function over a domain.

    Args:
        integrand: Function f(x) to integrate
        domain: Integration domain as list of (min, max) bounds
        config: Monte Carlo configuration

    Returns:
        MCResult with integration estimate and diagnostics
    """
    if config is None:
        config = MCConfig()

    import time

    start_time = time.time()

    # Choose sampler based on configuration
    sampler: MCSampler
    if config.sampling_method == "uniform":
        sampler = UniformMCSampler(domain, config)
    elif config.sampling_method == "stratified":
        sampler = StratifiedMCSampler(domain, config)
    elif config.sampling_method in ["sobol", "halton", "latin_hypercube"]:
        sampler = QuasiMCSampler(domain, config, config.sampling_method)
    else:
        logger.warning(f"Unknown sampling method: {config.sampling_method}, using uniform")
        sampler = UniformMCSampler(domain, config)

    # Sample points
    sample_points = sampler.sample(config.num_samples)
    weights = sampler.get_weights(sample_points)

    # Evaluate integrand
    function_values = integrand(sample_points)

    # Apply variance reduction if configured
    if config.use_control_variates and config.control_function is not None:
        cv = ControlVariates(config.control_function)
        cv.calibrate(sample_points, function_values)
        function_values = cv.apply(sample_points, function_values)

    # Apply antithetic variables if configured
    if config.use_antithetic_variables:
        function_values = _apply_antithetic_variables(function_values, sampler, integrand)

    # Compute estimate
    domain_volume = np.prod([max_val - min_val for min_val, max_val in domain])
    weighted_values = function_values * weights * len(weights)
    estimate = domain_volume * np.mean(weighted_values)

    # Compute error estimates
    standard_error = domain_volume * np.std(weighted_values) / np.sqrt(len(weighted_values))

    # Confidence interval
    z_score = 1.96 if config.confidence_level == 0.95 else 2.576  # Simplified
    ci_lower = estimate - z_score * standard_error
    ci_upper = estimate + z_score * standard_error

    # Create result
    result = MCResult(
        estimate=estimate,
        standard_error=standard_error,
        confidence_interval=(ci_lower, ci_upper),
        num_samples_used=config.num_samples,
        effective_sample_size=config.num_samples,
        converged=standard_error < config.error_tolerance,
        final_error=standard_error,
        computation_time=time.time() - start_time,
        samples_per_second=config.num_samples / max(time.time() - start_time, 1e-6),
    )

    return result


def _apply_antithetic_variables(function_values: NDArray, sampler: MCSampler, integrand: MCIntegrand) -> NDArray:
    """Apply antithetic variables variance reduction."""
    # Generate antithetic samples (simplified)
    # For uniform distribution: if U ~ Uniform(0,1), then 1-U is antithetic
    # This is a simplified implementation
    return function_values  # Placeholder


def adaptive_monte_carlo(
    integrand: MCIntegrand,
    domain: MCDomain,
    target_error: float = 1e-3,
    initial_samples: int = 1000,
    max_samples: int = 100000,
    batch_size: int = 1000,
) -> MCResult:
    """
    Adaptive Monte Carlo integration with error control.

    Args:
        integrand: Function to integrate
        domain: Integration domain
        target_error: Target error tolerance
        initial_samples: Initial number of samples
        max_samples: Maximum total samples
        batch_size: Batch size for adaptive sampling

    Returns:
        MCResult with adaptive sampling statistics
    """
    config = MCConfig(num_samples=initial_samples)

    # Initial estimation
    result = monte_carlo_integrate(integrand, domain, config)

    total_samples = initial_samples
    estimates = [result.estimate]
    errors = [result.standard_error]

    # Adaptive refinement
    while result.standard_error > target_error and total_samples < max_samples:
        # Add more samples
        additional_samples = min(batch_size, max_samples - total_samples)
        config.num_samples = additional_samples

        batch_result = monte_carlo_integrate(integrand, domain, config)

        # Combine estimates (simplified combination)
        total_samples += additional_samples
        weight_old = len(estimates) * initial_samples
        weight_new = additional_samples

        combined_estimate = (weight_old * result.estimate + weight_new * batch_result.estimate) / (
            weight_old + weight_new
        )

        # Update error estimate (simplified)
        result.estimate = combined_estimate
        result.standard_error *= np.sqrt(len(estimates) * initial_samples / total_samples)
        result.num_samples_used = total_samples

        estimates.append(combined_estimate)
        errors.append(result.standard_error)

        logger.info(f"Adaptive MC: {total_samples} samples, error = {result.standard_error:.6e}")

    result.converged = result.standard_error <= target_error
    result.iterations = len(estimates)

    return result


# Convenience functions for common use cases
def integrate_gaussian_quadrature_mc(
    mean: NDArray, cov: NDArray, integrand: MCIntegrand, num_samples: int = 10000
) -> MCResult:
    """Monte Carlo integration with Gaussian distribution."""
    # Create domain (6-sigma bounds) - vectorized
    sigma = np.sqrt(np.diag(cov))
    lower_bounds = mean - 6 * sigma
    upper_bounds = mean + 6 * sigma
    domain = list(zip(lower_bounds.tolist(), upper_bounds.tolist(), strict=True))

    # Gaussian importance function
    def gaussian_importance(x):
        diff = x - mean
        return np.exp(-0.5 * np.sum(diff @ np.linalg.inv(cov) * diff, axis=1))

    config = MCConfig(num_samples=num_samples, sampling_method="importance", importance_function=gaussian_importance)

    return monte_carlo_integrate(integrand, domain, config)


def estimate_expectation(samples: NDArray, weights: NDArray | None = None) -> tuple[float, float]:
    """
    Estimate expectation and standard error from samples.

    Args:
        samples: Sample values
        weights: Sample weights (uniform if None)

    Returns:
        Tuple of (expectation, standard_error)
    """
    if weights is None:
        weights = np.ones(len(samples)) / len(samples)

    expectation = np.average(samples, weights=weights)
    variance = np.average((samples - expectation) ** 2, weights=weights)
    standard_error = np.sqrt(variance / len(samples))

    return expectation, standard_error
