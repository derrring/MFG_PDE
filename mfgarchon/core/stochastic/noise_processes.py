"""
Common noise processes for stochastic Mean Field Games.

This module provides standard stochastic processes used to model common noise
in Mean Field Games applications (finance, epidemiology, robotics).

Mathematical Background:
    Common noise processes θ_t affect all agents simultaneously, representing
    shared environmental uncertainty such as:
    - Market indices (VIX, interest rates) in finance
    - Epidemic intensity levels in epidemiology
    - Shared sensor measurements in robotics

References:
    - Carmona & Delarue (2018): Probabilistic Theory of Mean Field Games
    - Karatzas & Shreve (1991): Brownian Motion and Stochastic Calculus
"""

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class NoiseProcess(Protocol):
    """
    Protocol for common noise processes in stochastic MFG.

    A noise process defines a stochastic differential equation:
        dθ_t = μ(θ_t, t) dt + σ(θ_t, t) dB_t

    where B_t is a standard Brownian motion.
    """

    def drift(self, theta: np.ndarray, t: float) -> np.ndarray:
        """
        Drift coefficient μ(θ, t) in the SDE.

        Args:
            theta: Current state of the process
            t: Current time

        Returns:
            Drift term μ(θ, t)
        """
        ...

    def diffusion(self, theta: np.ndarray, t: float) -> np.ndarray:
        """
        Diffusion coefficient σ(θ, t) in the SDE.

        Args:
            theta: Current state of the process
            t: Current time

        Returns:
            Diffusion term σ(θ, t)
        """
        ...

    def sample_path(
        self,
        T: float,
        Nt: int,
        theta0: float | np.ndarray | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        """
        Generate a sample path of the noise process.

        Args:
            T: Terminal time
            Nt: Number of time steps
            theta0: Initial condition (default: 0)
            seed: Random seed for reproducibility

        Returns:
            Array of shape (Nt+1,) containing the sample path θ_0, ..., θ_T
        """
        ...


class OrnsteinUhlenbeckProcess:
    """
    Ornstein-Uhlenbeck mean-reverting process.

    Mathematical Form:
        dθ_t = κ(μ - θ_t) dt + σ dB_t

    where:
        - κ > 0: Mean reversion speed
        - μ: Long-term mean level
        - σ > 0: Volatility

    Properties:
        - Mean-reverting: Process tends toward μ over time
        - Gaussian: Stationary distribution is N(μ, σ²/(2κ))
        - Always defined: Can take any real value

    Applications:
        - Interest rate models (Vasicek model)
        - Volatility processes
        - Temperature models
        - Any mean-reverting quantity

    Example:
        >>> # Model VIX volatility index (mean ~20, vol ~8)
        >>> vix_process = OrnsteinUhlenbeckProcess(
        ...     kappa=2.0,  # Fast mean reversion
        ...     mu=20.0,    # Long-term VIX level
        ...     sigma=8.0   # VIX volatility
        ... )
        >>> path = vix_process.sample_path(T=1.0, Nt=252)  # 1 year, daily
    """

    def __init__(self, kappa: float, mu: float, sigma: float):
        """
        Initialize Ornstein-Uhlenbeck process.

        Args:
            kappa: Mean reversion speed (κ > 0)
            mu: Long-term mean level
            sigma: Volatility (σ > 0)

        Raises:
            ValueError: If kappa ≤ 0 or sigma ≤ 0
        """
        if kappa <= 0:
            raise ValueError(f"Mean reversion speed must be positive, got kappa={kappa}")
        if sigma <= 0:
            raise ValueError(f"Volatility must be positive, got sigma={sigma}")

        self.kappa = kappa
        self.mu = mu
        self.sigma = sigma

    def drift(self, theta: np.ndarray, t: float) -> np.ndarray:
        """Drift coefficient: μ(θ,t) = κ(μ - θ)."""
        return self.kappa * (self.mu - theta)

    def diffusion(self, theta: np.ndarray, t: float) -> np.ndarray:
        """Diffusion coefficient: σ(θ,t) = σ (constant)."""
        return self.sigma * np.ones_like(theta)

    def sample_path(
        self,
        T: float,
        Nt: int,
        theta0: float | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        """
        Generate sample path using Euler-Maruyama scheme.

        Uses exact solution for better accuracy:
            θ_{t+dt} = θ_t e^{-κ dt} + μ(1 - e^{-κ dt}) + σ√((1-e^{-2κ dt})/(2κ)) Z

        where Z ~ N(0,1).
        """
        if seed is not None:
            np.random.seed(seed)

        if theta0 is None:
            theta0 = self.mu  # Start at long-term mean

        dt = T / Nt
        path = np.zeros(Nt + 1)
        path[0] = theta0

        # Precompute coefficients for exact solution
        exp_kdt = np.exp(-self.kappa * dt)
        mean_coef = 1 - exp_kdt
        std_coef = self.sigma * np.sqrt((1 - np.exp(-2 * self.kappa * dt)) / (2 * self.kappa))

        for i in range(Nt):
            path[i + 1] = path[i] * exp_kdt + self.mu * mean_coef + std_coef * np.random.randn()

        return path

    def __repr__(self) -> str:
        return f"OrnsteinUhlenbeckProcess(kappa={self.kappa}, mu={self.mu}, sigma={self.sigma})"


class CoxIngersollRossProcess:
    """
    Cox-Ingersoll-Ross (CIR) process - always positive mean-reverting process.

    Mathematical Form:
        dθ_t = κ(μ - θ_t) dt + σ√θ_t dB_t

    where:
        - κ > 0: Mean reversion speed
        - μ > 0: Long-term mean level
        - σ > 0: Volatility
        - Feller condition: 2κμ ≥ σ² ensures θ_t > 0 always

    Properties:
        - Always positive: θ_t > 0 for all t (under Feller condition)
        - Mean-reverting: Tends toward μ
        - Square-root diffusion: Volatility proportional to √θ

    Applications:
        - Interest rate models (Cox-Ingersoll-Ross model)
        - Variance processes (Heston model)
        - Epidemic intensity (always positive)
        - Any positive mean-reverting quantity

    Example:
        >>> # Model 3-month treasury rate
        >>> interest_rate = CoxIngersollRossProcess(
        ...     kappa=0.5,   # Moderate mean reversion
        ...     mu=0.03,     # 3% long-term rate
        ...     sigma=0.05   # 5% volatility
        ... )
        >>> # Verify Feller condition: 2*0.5*0.03 = 0.03 ≥ 0.05² = 0.0025 ✓
        >>> path = interest_rate.sample_path(T=10.0, Nt=120)  # 10 years
    """

    def __init__(self, kappa: float, mu: float, sigma: float):
        """
        Initialize Cox-Ingersoll-Ross process.

        Args:
            kappa: Mean reversion speed (κ > 0)
            mu: Long-term mean level (μ > 0)
            sigma: Volatility (σ > 0)

        Raises:
            ValueError: If parameters are non-positive
            Warning: If Feller condition not satisfied
        """
        if kappa <= 0:
            raise ValueError(f"Mean reversion speed must be positive, got kappa={kappa}")
        if mu <= 0:
            raise ValueError(f"Long-term mean must be positive, got mu={mu}")
        if sigma <= 0:
            raise ValueError(f"Volatility must be positive, got sigma={sigma}")

        # Check Feller condition: 2κμ ≥ σ² ensures process stays positive
        if 2 * kappa * mu < sigma**2:
            import warnings

            warnings.warn(
                f"Feller condition not satisfied: 2κμ={2 * kappa * mu:.6f} < σ²={sigma**2:.6f}. Process may hit zero.",
                UserWarning,
            )

        self.kappa = kappa
        self.mu = mu
        self.sigma = sigma

    def drift(self, theta: np.ndarray, t: float) -> np.ndarray:
        """Drift coefficient: μ(θ,t) = κ(μ - θ)."""
        return self.kappa * (self.mu - theta)

    def diffusion(self, theta: np.ndarray, t: float) -> np.ndarray:
        """Diffusion coefficient: σ(θ,t) = σ√θ."""
        return self.sigma * np.sqrt(np.maximum(theta, 0))  # Ensure non-negative

    def sample_path(
        self,
        T: float,
        Nt: int,
        theta0: float | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        """
        Generate sample path using Euler-Maruyama with reflection at zero.

        Uses Euler-Maruyama with reflection to maintain positivity:
            θ_{t+dt} = max(0, θ_t + κ(μ - θ_t)dt + σ√θ_t √dt Z)

        where Z ~ N(0,1).
        """
        if seed is not None:
            np.random.seed(seed)

        if theta0 is None:
            theta0 = self.mu  # Start at long-term mean

        dt = T / Nt
        sqrt_dt = np.sqrt(dt)
        path = np.zeros(Nt + 1)
        path[0] = theta0

        for i in range(Nt):
            theta_current = path[i]

            # Euler-Maruyama step
            drift_term = self.kappa * (self.mu - theta_current) * dt
            diffusion_term = self.sigma * np.sqrt(max(theta_current, 0)) * sqrt_dt * np.random.randn()

            # Reflect at zero to maintain positivity
            path[i + 1] = max(0, theta_current + drift_term + diffusion_term)

        return path

    def __repr__(self) -> str:
        return f"CoxIngersollRossProcess(kappa={self.kappa}, mu={self.mu}, sigma={self.sigma})"


class GeometricBrownianMotion:
    """
    Geometric Brownian Motion - standard model for asset prices.

    Mathematical Form:
        dθ_t = μ θ_t dt + σ θ_t dB_t

    Equivalent to:
        θ_t = θ_0 exp((μ - σ²/2)t + σB_t)

    where:
        - μ: Drift (expected return)
        - σ > 0: Volatility
        - θ_0 > 0: Initial value

    Properties:
        - Always positive: θ_t > 0 for all t
        - Log-normal distribution: log(θ_t) ~ N(log(θ_0) + (μ - σ²/2)t, σ²t)
        - Multiplicative: Percentage changes are independent of level

    Applications:
        - Stock prices (Black-Scholes model)
        - Market indices (S&P 500, VIX)
        - Foreign exchange rates
        - Commodity prices

    Example:
        >>> # Model S&P 500 index
        >>> sp500 = GeometricBrownianMotion(
        ...     mu=0.10,      # 10% expected annual return
        ...     sigma=0.20    # 20% annual volatility
        ... )
        >>> path = sp500.sample_path(T=1.0, Nt=252, theta0=4000)  # 1 year from 4000
    """

    def __init__(self, mu: float, sigma: float):
        """
        Initialize Geometric Brownian Motion.

        Args:
            mu: Drift coefficient (expected return)
            sigma: Volatility (σ > 0)

        Raises:
            ValueError: If sigma ≤ 0
        """
        if sigma <= 0:
            raise ValueError(f"Volatility must be positive, got sigma={sigma}")

        self.mu = mu
        self.sigma = sigma

    def drift(self, theta: np.ndarray, t: float) -> np.ndarray:
        """Drift coefficient: μ(θ,t) = μθ."""
        return self.mu * theta

    def diffusion(self, theta: np.ndarray, t: float) -> np.ndarray:
        """Diffusion coefficient: σ(θ,t) = σθ."""
        return self.sigma * theta

    def sample_path(
        self,
        T: float,
        Nt: int,
        theta0: float | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        """
        Generate sample path using exact solution.

        Uses exact solution for maximum accuracy:
            θ_t = θ_0 exp((μ - σ²/2)t + σB_t)

        where B_t = √t Z with Z ~ N(0,1).
        """
        if seed is not None:
            np.random.seed(seed)

        if theta0 is None:
            theta0 = 1.0  # Default starting value

        if theta0 <= 0:
            raise ValueError(f"Initial value must be positive, got theta0={theta0}")

        dt = T / Nt
        path = np.zeros(Nt + 1)
        path[0] = theta0

        # Exact solution: θ_{t+dt} = θ_t exp((μ - σ²/2)dt + σ√dt Z)
        drift_correction = (self.mu - 0.5 * self.sigma**2) * dt
        diffusion_factor = self.sigma * np.sqrt(dt)

        for i in range(Nt):
            path[i + 1] = path[i] * np.exp(drift_correction + diffusion_factor * np.random.randn())

        return path

    def __repr__(self) -> str:
        return f"GeometricBrownianMotion(mu={self.mu}, sigma={self.sigma})"


class JumpDiffusionProcess:
    """
    Jump diffusion process (Merton model) with discontinuous jumps.

    Mathematical Form:
        dθ_t = μ dt + σ dB_t + J_t dN_t

    where:
        - μ: Continuous drift
        - σ > 0: Continuous volatility (Brownian component)
        - N_t: Poisson process with intensity λ
        - J_t: Jump size (typically log-normal)

    Properties:
        - Combines continuous and jump components
        - Captures sudden events (market crashes, policy changes)
        - Heavy tails compared to pure Brownian motion

    Applications:
        - Market crashes and sudden drops
        - Epidemic outbreaks (sudden jumps in infections)
        - Policy interventions (sudden regulatory changes)
        - System failures or recoveries

    Example:
        >>> # Model stock index with crash risk
        >>> index_with_jumps = JumpDiffusionProcess(
        ...     mu=0.10,           # 10% drift
        ...     sigma=0.15,        # 15% continuous volatility
        ...     jump_intensity=2,  # 2 jumps per year on average
        ...     jump_mean=-0.05,   # -5% average jump (crashes)
        ...     jump_std=0.03      # 3% jump size volatility
        ... )
        >>> path = index_with_jumps.sample_path(T=1.0, Nt=252)
    """

    def __init__(
        self,
        mu: float,
        sigma: float,
        jump_intensity: float,
        jump_mean: float,
        jump_std: float,
    ):
        """
        Initialize jump diffusion process.

        Args:
            mu: Drift coefficient
            sigma: Continuous volatility (σ > 0)
            jump_intensity: Poisson intensity λ > 0 (jumps per unit time)
            jump_mean: Mean jump size
            jump_std: Jump size standard deviation (> 0)

        Raises:
            ValueError: If sigma, jump_intensity, or jump_std are non-positive
        """
        if sigma <= 0:
            raise ValueError(f"Volatility must be positive, got sigma={sigma}")
        if jump_intensity <= 0:
            raise ValueError(f"Jump intensity must be positive, got jump_intensity={jump_intensity}")
        if jump_std <= 0:
            raise ValueError(f"Jump std must be positive, got jump_std={jump_std}")

        self.mu = mu
        self.sigma = sigma
        self.jump_intensity = jump_intensity
        self.jump_mean = jump_mean
        self.jump_std = jump_std

    def drift(self, theta: np.ndarray, t: float) -> np.ndarray:
        """Drift coefficient: μ(θ,t) = μ (continuous component only)."""
        return self.mu * np.ones_like(theta)

    def diffusion(self, theta: np.ndarray, t: float) -> np.ndarray:
        """Diffusion coefficient: σ(θ,t) = σ (continuous component only)."""
        return self.sigma * np.ones_like(theta)

    def sample_path(
        self,
        T: float,
        Nt: int,
        theta0: float | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        """
        Generate sample path with both continuous and jump components.

        Discretization:
            θ_{t+dt} = θ_t + μ dt + σ√dt Z + ∑_{jumps in (t,t+dt]} J_i

        where:
            - Z ~ N(0,1) for Brownian component
            - Number of jumps ~ Poisson(λ dt)
            - Jump sizes J_i ~ N(jump_mean, jump_std²)
        """
        if seed is not None:
            np.random.seed(seed)

        if theta0 is None:
            theta0 = 0.0  # Default starting value

        dt = T / Nt
        sqrt_dt = np.sqrt(dt)
        path = np.zeros(Nt + 1)
        path[0] = theta0

        for i in range(Nt):
            # Continuous component (Euler-Maruyama)
            continuous_drift = self.mu * dt
            continuous_diffusion = self.sigma * sqrt_dt * np.random.randn()

            # Jump component (Compound Poisson)
            num_jumps = np.random.poisson(self.jump_intensity * dt)
            jump_component = 0.0
            if num_jumps > 0:
                jump_sizes = np.random.normal(self.jump_mean, self.jump_std, size=num_jumps)
                jump_component = np.sum(jump_sizes)

            path[i + 1] = path[i] + continuous_drift + continuous_diffusion + jump_component

        return path

    def __repr__(self) -> str:
        return (
            f"JumpDiffusionProcess(mu={self.mu}, sigma={self.sigma}, "
            f"jump_intensity={self.jump_intensity}, jump_mean={self.jump_mean}, "
            f"jump_std={self.jump_std})"
        )
