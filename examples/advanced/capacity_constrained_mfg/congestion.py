"""
Congestion cost models for capacity-constrained MFG problems.

This module provides density-dependent cost functions g(m/C) that model
congestion effects when agent density m(x) approaches capacity C(x).

Mathematical Framework:
    In capacity-constrained MFG, the running cost includes a congestion term:

    L(x, m) = L_0(x) + γ·g(m(x)/C(x))

    where:
    - m(x): agent density at position x
    - C(x): spatial capacity at x
    - g(ρ): congestion cost function of congestion ratio ρ = m/C
    - γ: congestion weight (trade-off parameter)

    The Hamiltonian becomes:
    H(x, m, ∇u) = H_0(∇u) + γ·g(m/C)

    For well-posedness, g(ρ) should satisfy:
    1. Monotonicity: g'(ρ) > 0 (cost increases with density)
    2. Convexity: g''(ρ) ≥ 0 (marginal cost increasing)
    3. Boundary: g(0) = 0, g(ρ) → ∞ as ρ → ρ_max

References:
    - Hughes, R. L. (2002). "A continuum theory for the flow of pedestrians."
    - Maury, B., & Venel, J. (2008). "A mathematical framework for a crowd
      motion model." Comptes Rendus Mathematique, 346(23-24), 1245-1250.
    - Di Francesco, M., Markowich, P. A., Pietschmann, J. F., & Wolfram, M. T.
      (2011). "On the Hughes' model for pedestrian flow."

Created: 2025-11-12
Author: MFG_PDE Team
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class CongestionModel(ABC):
    """
    Abstract base class for congestion cost functions.

    All congestion models implement:
    - cost(ρ): Running cost g(ρ) where ρ = m/C
    - derivative(ρ): First derivative g'(ρ) for HJB coupling ∂H/∂m
    - second_derivative(ρ): Second derivative g''(ρ) for convexity checks

    Subclasses must ensure:
    - g(0) = 0 (no cost at zero density)
    - g'(ρ) > 0 for ρ > 0 (monotonicity)
    - g''(ρ) ≥ 0 (convexity for well-posedness)
    """

    @abstractmethod
    def cost(self, density: NDArray[np.floating], capacity: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Compute congestion cost g(m/C).

        Args:
            density: Agent density m(x) ≥ 0
            capacity: Corridor capacity C(x) > 0

        Returns:
            Congestion cost array (same shape as input)
        """

    @abstractmethod
    def derivative(self, density: NDArray[np.floating], capacity: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Compute derivative ∂g/∂m needed for HJB equation.

        The HJB coupling term is:
            ∂H/∂m = γ · (∂g/∂m) = γ · g'(m/C) · (1/C)

        Args:
            density: Agent density m(x)
            capacity: Corridor capacity C(x)

        Returns:
            Derivative array g'(m/C) / C
        """

    def second_derivative(self, density: NDArray[np.floating], capacity: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Compute second derivative g''(m/C) for convexity checks.

        Args:
            density: Agent density m(x)
            capacity: Corridor capacity C(x)

        Returns:
            Second derivative array
        """
        raise NotImplementedError("Second derivative not implemented for this model")

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}()"


class QuadraticCongestion(CongestionModel):
    """
    Quadratic congestion cost: g(ρ) = ρ².

    This is the standard choice in Hughes-type models, providing smooth
    congestion penalties with g'(ρ) = 2ρ.

    Properties:
    - Moderate penalties: Cost grows quadratically
    - Smooth: C² differentiable
    - Well-posed: Convex with g''(ρ) = 2

    Examples:
        >>> model = QuadraticCongestion()
        >>> density = np.array([0.5, 1.0, 1.5])
        >>> capacity = np.array([1.0, 1.0, 1.0])
        >>> cost = model.cost(density, capacity)
        >>> # cost = [0.25, 1.0, 2.25]
    """

    def cost(self, density: NDArray[np.floating], capacity: NDArray[np.floating]) -> NDArray[np.floating]:
        """g(ρ) = ρ² = (m/C)²"""
        ratio = density / capacity
        return ratio**2

    def derivative(self, density: NDArray[np.floating], capacity: NDArray[np.floating]) -> NDArray[np.floating]:
        """∂g/∂m = 2ρ / C = 2m / C²"""
        return 2.0 * density / (capacity**2)

    def second_derivative(self, density: NDArray[np.floating], capacity: NDArray[np.floating]) -> NDArray[np.floating]:
        """g''(ρ) = 2 / C²"""
        return 2.0 * np.ones_like(density) / (capacity**2)


class ExponentialCongestion(CongestionModel):
    """
    Exponential congestion cost: g(ρ) = exp(ρ) - 1.

    This model imposes sharp penalties near capacity, creating stronger
    congestion avoidance than quadratic costs.

    Properties:
    - Sharp penalties: Cost grows exponentially
    - Smooth: C∞ differentiable
    - Well-posed: Convex with g''(ρ) = exp(ρ)

    Attributes:
        scale: Scaling factor for ρ (controls sharpness)

    Examples:
        >>> model = ExponentialCongestion(scale=2.0)
        >>> # g(ρ) = exp(2ρ) - 1
    """

    def __init__(self, scale: float = 1.0):
        """
        Initialize exponential congestion model.

        Args:
            scale: Scaling factor for congestion ratio (default 1.0)
                   Higher scale = sharper penalties
        """
        if scale <= 0:
            raise ValueError(f"Scale must be positive, got {scale}")
        self.scale = scale

    def cost(self, density: NDArray[np.floating], capacity: NDArray[np.floating]) -> NDArray[np.floating]:
        """g(ρ) = exp(scale · ρ) - 1"""
        ratio = density / capacity
        return np.exp(self.scale * ratio) - 1.0

    def derivative(self, density: NDArray[np.floating], capacity: NDArray[np.floating]) -> NDArray[np.floating]:
        """∂g/∂m = scale · exp(scale · ρ) / C"""
        ratio = density / capacity
        return self.scale * np.exp(self.scale * ratio) / capacity

    def second_derivative(self, density: NDArray[np.floating], capacity: NDArray[np.floating]) -> NDArray[np.floating]:
        """g''(ρ) = scale² · exp(scale · ρ) / C²"""
        ratio = density / capacity
        return (self.scale**2) * np.exp(self.scale * ratio) / (capacity**2)

    def __repr__(self) -> str:
        return f"ExponentialCongestion(scale={self.scale})"


class LogBarrierCongestion(CongestionModel):
    """
    Log-barrier congestion cost: g(ρ) = -log(1 - ρ) for ρ < 1.

    This model enforces a hard capacity constraint: as ρ → 1⁻, cost → ∞.
    Agents cannot exceed capacity without infinite cost.

    Properties:
    - Hard constraint: g(ρ) → ∞ as ρ → 1⁻
    - Smooth: C∞ for ρ < 1
    - Well-posed: Convex with g''(ρ) = 1/(1-ρ)²

    Numerical Stability:
        If m > C during numerical iterations (discretization error), the log
        returns NaN. This implementation uses a "soft clamp" with piecewise
        linear extension beyond threshold to maintain stability.

    Attributes:
        threshold: Safety threshold ρ_max < 1 (default 0.95)
        penalty_slope: Linear penalty slope for ρ > threshold

    Examples:
        >>> model = LogBarrierCongestion(threshold=0.95)
        >>> density = np.array([0.5, 0.9, 0.99])
        >>> capacity = np.array([1.0, 1.0, 1.0])
        >>> cost = model.cost(density, capacity)
        >>> # For ρ=0.99 > threshold, uses linear extension
    """

    def __init__(self, threshold: float = 0.95, penalty_slope: float = 10.0):
        """
        Initialize log-barrier congestion model.

        Args:
            threshold: Maximum safe congestion ratio (default 0.95)
            penalty_slope: Linear penalty slope for ρ > threshold

        Raises:
            ValueError: If threshold not in (0, 1)
        """
        if not 0 < threshold < 1:
            raise ValueError(f"Threshold must be in (0, 1), got {threshold}")
        if penalty_slope <= 0:
            raise ValueError(f"Penalty slope must be positive, got {penalty_slope}")

        self.threshold = threshold
        self.penalty_slope = penalty_slope

        # Precompute extension parameters for continuity
        # At threshold: g(ρ_th) = -log(1 - ρ_th)
        # Extension: g(ρ) = g(ρ_th) + slope·(ρ - ρ_th) for ρ > ρ_th
        self._barrier_at_threshold = -np.log(1.0 - threshold)
        self._deriv_at_threshold = 1.0 / (1.0 - threshold)

    def cost(self, density: NDArray[np.floating], capacity: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        g(ρ) = -log(1 - ρ)  for ρ < threshold
             = g(threshold) + penalty_slope·(ρ - threshold)  for ρ ≥ threshold
        """
        ratio = density / capacity
        cost = np.zeros_like(ratio)

        # Safe region: ρ < threshold
        safe_mask = ratio < self.threshold
        cost[safe_mask] = -np.log(1.0 - ratio[safe_mask])

        # Extension region: ρ ≥ threshold (linear penalty)
        unsafe_mask = ~safe_mask
        excess = ratio[unsafe_mask] - self.threshold
        cost[unsafe_mask] = self._barrier_at_threshold + self.penalty_slope * excess

        return cost

    def derivative(self, density: NDArray[np.floating], capacity: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        ∂g/∂m = g'(ρ) / C
              = 1 / [(1-ρ)·C]  for ρ < threshold
              = penalty_slope / C  for ρ ≥ threshold
        """
        ratio = density / capacity
        deriv = np.zeros_like(ratio)

        # Safe region
        safe_mask = ratio < self.threshold
        deriv[safe_mask] = 1.0 / ((1.0 - ratio[safe_mask]) * capacity[safe_mask])

        # Extension region (constant slope)
        unsafe_mask = ~safe_mask
        deriv[unsafe_mask] = self.penalty_slope / capacity[unsafe_mask]

        return deriv

    def second_derivative(self, density: NDArray[np.floating], capacity: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        g''(ρ) = 1 / [(1-ρ)² · C²]  for ρ < threshold
               = 0  for ρ ≥ threshold (piecewise linear extension)
        """
        ratio = density / capacity
        second_deriv = np.zeros_like(ratio)

        safe_mask = ratio < self.threshold
        second_deriv[safe_mask] = 1.0 / ((1.0 - ratio[safe_mask]) ** 2 * capacity[safe_mask] ** 2)

        return second_deriv

    def __repr__(self) -> str:
        return f"LogBarrierCongestion(threshold={self.threshold}, penalty_slope={self.penalty_slope})"


class PiecewiseCongestion(CongestionModel):
    """
    Piecewise congestion cost with free-flow and congested regimes.

    This model captures the empirical observation that congestion costs
    are negligible at low density, then increase sharply past a threshold.

    Model:
        g(ρ) = 0                    for ρ < ρ_free
             = k·(ρ - ρ_free)^p     for ρ ≥ ρ_free

    where:
    - ρ_free: Free-flow threshold (no congestion below)
    - k: Cost scaling factor
    - p: Power law exponent (p=2 is quadratic)

    Attributes:
        free_flow_threshold: Congestion ratio below which cost = 0
        cost_scale: Scaling factor k
        power: Exponent p (default 2)

    Examples:
        >>> model = PiecewiseCongestion(free_flow_threshold=0.3, power=2)
        >>> # No cost for ρ < 0.3, quadratic cost for ρ ≥ 0.3
    """

    def __init__(self, free_flow_threshold: float = 0.3, cost_scale: float = 1.0, power: float = 2.0):
        """
        Initialize piecewise congestion model.

        Args:
            free_flow_threshold: Congestion ratio for free flow (default 0.3)
            cost_scale: Scaling factor k (default 1.0)
            power: Exponent p (default 2.0)

        Raises:
            ValueError: If parameters out of valid range
        """
        if not 0 < free_flow_threshold < 1:
            raise ValueError(f"Free flow threshold must be in (0, 1), got {free_flow_threshold}")
        if cost_scale <= 0:
            raise ValueError(f"Cost scale must be positive, got {cost_scale}")
        if power < 1:
            raise ValueError(f"Power must be ≥ 1 for convexity, got {power}")

        self.free_flow_threshold = free_flow_threshold
        self.cost_scale = cost_scale
        self.power = power

    def cost(self, density: NDArray[np.floating], capacity: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        g(ρ) = 0                          for ρ < ρ_free
             = k·(ρ - ρ_free)^p           for ρ ≥ ρ_free
        """
        ratio = density / capacity
        cost = np.zeros_like(ratio)

        congested_mask = ratio >= self.free_flow_threshold
        excess = ratio[congested_mask] - self.free_flow_threshold
        cost[congested_mask] = self.cost_scale * (excess**self.power)

        return cost

    def derivative(self, density: NDArray[np.floating], capacity: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        ∂g/∂m = 0                                      for ρ < ρ_free
              = k·p·(ρ - ρ_free)^(p-1) / C             for ρ ≥ ρ_free
        """
        ratio = density / capacity
        deriv = np.zeros_like(ratio)

        congested_mask = ratio >= self.free_flow_threshold
        excess = ratio[congested_mask] - self.free_flow_threshold
        deriv[congested_mask] = self.cost_scale * self.power * (excess ** (self.power - 1.0)) / capacity[congested_mask]

        return deriv

    def second_derivative(self, density: NDArray[np.floating], capacity: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        g''(ρ) = 0                                        for ρ < ρ_free
               = k·p·(p-1)·(ρ - ρ_free)^(p-2) / C²       for ρ ≥ ρ_free
        """
        ratio = density / capacity
        second_deriv = np.zeros_like(ratio)

        congested_mask = ratio >= self.free_flow_threshold
        excess = ratio[congested_mask] - self.free_flow_threshold
        second_deriv[congested_mask] = (
            self.cost_scale
            * self.power
            * (self.power - 1.0)
            * (excess ** (self.power - 2.0))
            / (capacity[congested_mask] ** 2)
        )

        return second_deriv

    def __repr__(self) -> str:
        return f"PiecewiseCongestion(free_flow={self.free_flow_threshold}, scale={self.cost_scale}, power={self.power})"


def create_congestion_model(
    model_type: str = "quadratic",
    **kwargs,
) -> CongestionModel:
    """
    Factory function for congestion models.

    Args:
        model_type: Type of congestion model
            - "quadratic": QuadraticCongestion (default)
            - "exponential": ExponentialCongestion
            - "log_barrier": LogBarrierCongestion
            - "piecewise": PiecewiseCongestion
        **kwargs: Model-specific parameters

    Returns:
        CongestionModel instance

    Examples:
        >>> model = create_congestion_model("quadratic")
        >>> model = create_congestion_model("exponential", scale=2.0)
        >>> model = create_congestion_model("log_barrier", threshold=0.95)
        >>> model = create_congestion_model("piecewise", free_flow_threshold=0.3)
    """
    models = {
        "quadratic": QuadraticCongestion,
        "exponential": ExponentialCongestion,
        "log_barrier": LogBarrierCongestion,
        "piecewise": PiecewiseCongestion,
    }

    if model_type not in models:
        raise ValueError(f"Unknown model type '{model_type}'. Choose from: {list(models.keys())}")

    return models[model_type](**kwargs)


__all__ = [
    "CongestionModel",
    "QuadraticCongestion",
    "ExponentialCongestion",
    "LogBarrierCongestion",
    "PiecewiseCongestion",
    "create_congestion_model",
]


if __name__ == "__main__":
    """Smoke tests for CongestionModel module."""
    print("Running CongestionModel smoke tests...")

    # Test densities and capacities
    density = np.array([0.3, 0.5, 0.7, 0.9, 0.95])
    capacity = np.ones_like(density)

    # Test 1: Quadratic congestion
    print("\n1. QuadraticCongestion...")
    quad = QuadraticCongestion()
    cost = quad.cost(density, capacity)
    deriv = quad.derivative(density, capacity)
    print(f"   Costs: {cost}")
    print(f"   Derivatives: {deriv}")
    # Verify monotonicity
    assert np.all(np.diff(cost) > 0), "Cost should be monotonically increasing"
    assert np.all(deriv > 0), "Derivative should be positive"
    print("   ✓ Quadratic model working")

    # Test 2: Exponential congestion
    print("\n2. ExponentialCongestion...")
    exp_model = ExponentialCongestion(scale=2.0)
    cost = exp_model.cost(density, capacity)
    deriv = exp_model.derivative(density, capacity)
    print(f"   Costs: {cost}")
    # Verify sharp growth
    assert cost[-1] > 5 * cost[0], "Exponential should grow sharply"
    assert np.all(deriv > 0), "Derivative should be positive"
    print("   ✓ Exponential model working")

    # Test 3: LogBarrier congestion (safe region)
    print("\n3. LogBarrierCongestion (safe region)...")
    log_model = LogBarrierCongestion(threshold=0.95, penalty_slope=10.0)
    cost = log_model.cost(density, capacity)
    deriv = log_model.derivative(density, capacity)
    print(f"   Costs: {cost}")
    # Verify divergence near threshold
    assert cost[-1] > cost[-2], "Cost should grow rapidly near threshold"
    assert np.all(np.isfinite(cost)), "Costs should be finite"
    assert np.all(np.isfinite(deriv)), "Derivatives should be finite"
    print("   ✓ LogBarrier model working (safe)")

    # Test 4: LogBarrier with overcapacity (numerical stability)
    print("\n4. LogBarrierCongestion (overcapacity test)...")
    overcrowded_density = np.array([0.96, 0.98, 1.0, 1.05])  # Exceeds threshold
    overcrowded_capacity = np.ones_like(overcrowded_density)
    cost_over = log_model.cost(overcrowded_density, overcrowded_capacity)
    deriv_over = log_model.derivative(overcrowded_density, overcrowded_capacity)
    print(f"   Costs (overcapacity): {cost_over}")
    assert np.all(np.isfinite(cost_over)), "Should handle overcapacity without NaN"
    assert np.all(np.isfinite(deriv_over)), "Derivatives should be finite"
    print("   ✓ LogBarrier stable under overcapacity")

    # Test 5: Piecewise congestion
    print("\n5. PiecewiseCongestion...")
    piecewise = PiecewiseCongestion(free_flow_threshold=0.3, cost_scale=1.0, power=2.0)
    cost = piecewise.cost(density, capacity)
    deriv = piecewise.derivative(density, capacity)
    print(f"   Costs: {cost}")
    # Verify free-flow region
    assert np.isclose(cost[0], 0.0, atol=1e-10), "Should be zero in free-flow"
    assert cost[-1] > 0, "Should have cost in congested regime"
    print("   ✓ Piecewise model working")

    # Test 6: Factory function
    print("\n6. Factory function...")
    models = [
        create_congestion_model("quadratic"),
        create_congestion_model("exponential", scale=1.5),
        create_congestion_model("log_barrier", threshold=0.9),
        create_congestion_model("piecewise", free_flow_threshold=0.2),
    ]
    for model in models:
        cost = model.cost(np.array([0.5]), np.array([1.0]))
        assert np.isfinite(cost[0]), f"{model} should produce finite cost"
    print(f"   Created {len(models)} models via factory")
    print("   ✓ Factory working")

    # Test 7: Convexity checks (second derivative)
    print("\n7. Convexity checks...")
    test_density = np.linspace(0.1, 0.8, 10)
    test_capacity = np.ones_like(test_density)

    for name, model in [
        ("Quadratic", QuadraticCongestion()),
        ("Exponential", ExponentialCongestion(scale=1.0)),
    ]:
        second_deriv = model.second_derivative(test_density, test_capacity)
        assert np.all(second_deriv >= 0), f"{name} should be convex (g'' ≥ 0)"
        print(f"   ✓ {name} is convex")

    print("\n✅ All CongestionModel smoke tests passed!")
