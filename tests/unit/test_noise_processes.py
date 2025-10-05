"""
Unit tests for stochastic noise processes.

Tests mathematical properties and numerical accuracy of noise process implementations.
"""

import pytest

import numpy as np

from mfg_pde.core.stochastic import (
    CoxIngersollRossProcess,
    GeometricBrownianMotion,
    JumpDiffusionProcess,
    OrnsteinUhlenbeckProcess,
)


class TestOrnsteinUhlenbeckProcess:
    """Test Ornstein-Uhlenbeck process properties."""

    def test_initialization_valid(self):
        """Test OU process initialization with valid parameters."""
        process = OrnsteinUhlenbeckProcess(kappa=2.0, mu=0.5, sigma=0.1)
        assert process.kappa == 2.0
        assert process.mu == 0.5
        assert process.sigma == 0.1

    def test_initialization_invalid_kappa(self):
        """Test that non-positive kappa raises ValueError."""
        with pytest.raises(ValueError, match="Mean reversion speed must be positive"):
            OrnsteinUhlenbeckProcess(kappa=0.0, mu=0.5, sigma=0.1)

        with pytest.raises(ValueError, match="Mean reversion speed must be positive"):
            OrnsteinUhlenbeckProcess(kappa=-1.0, mu=0.5, sigma=0.1)

    def test_initialization_invalid_sigma(self):
        """Test that non-positive sigma raises ValueError."""
        with pytest.raises(ValueError, match="Volatility must be positive"):
            OrnsteinUhlenbeckProcess(kappa=2.0, mu=0.5, sigma=0.0)

        with pytest.raises(ValueError, match="Volatility must be positive"):
            OrnsteinUhlenbeckProcess(kappa=2.0, mu=0.5, sigma=-0.1)

    def test_drift_formula(self):
        """Test drift coefficient κ(μ - θ)."""
        process = OrnsteinUhlenbeckProcess(kappa=2.0, mu=0.5, sigma=0.1)

        theta = np.array([0.0, 0.5, 1.0])
        expected_drift = 2.0 * (0.5 - theta)  # κ(μ - θ)

        drift = process.drift(theta, t=0.0)
        np.testing.assert_allclose(drift, expected_drift)

    def test_diffusion_constant(self):
        """Test diffusion coefficient is constant σ."""
        process = OrnsteinUhlenbeckProcess(kappa=2.0, mu=0.5, sigma=0.1)

        theta = np.array([0.0, 0.5, 1.0])
        expected_diffusion = 0.1 * np.ones(3)

        diffusion = process.diffusion(theta, t=0.0)
        np.testing.assert_allclose(diffusion, expected_diffusion)

    def test_sample_path_shape(self):
        """Test sample path has correct shape."""
        process = OrnsteinUhlenbeckProcess(kappa=2.0, mu=0.5, sigma=0.1)
        path = process.sample_path(T=1.0, Nt=100, seed=42)

        assert path.shape == (101,)  # Nt+1 points

    def test_sample_path_initial_condition(self):
        """Test sample path starts at specified initial value."""
        process = OrnsteinUhlenbeckProcess(kappa=2.0, mu=0.5, sigma=0.1)

        # Default: starts at long-term mean
        path_default = process.sample_path(T=1.0, Nt=100, seed=42)
        assert path_default[0] == 0.5

        # Custom initial condition
        path_custom = process.sample_path(T=1.0, Nt=100, theta0=1.0, seed=42)
        assert path_custom[0] == 1.0

    def test_mean_reversion(self):
        """Test long-term convergence to mean μ."""
        process = OrnsteinUhlenbeckProcess(kappa=2.0, mu=0.5, sigma=0.1)

        # Generate long path starting far from mean
        path = process.sample_path(T=10.0, Nt=10000, theta0=2.0, seed=42)

        # Check convergence: last 20% of path should be close to μ=0.5
        long_term_mean = path[-2000:].mean()
        assert abs(long_term_mean - 0.5) < 0.2  # Within 2 standard deviations

    def test_reproducibility(self):
        """Test that same seed produces same path."""
        process = OrnsteinUhlenbeckProcess(kappa=2.0, mu=0.5, sigma=0.1)

        path1 = process.sample_path(T=1.0, Nt=100, seed=42)
        path2 = process.sample_path(T=1.0, Nt=100, seed=42)

        np.testing.assert_array_equal(path1, path2)

    def test_repr(self):
        """Test string representation."""
        process = OrnsteinUhlenbeckProcess(kappa=2.0, mu=0.5, sigma=0.1)
        repr_str = repr(process)

        assert "OrnsteinUhlenbeckProcess" in repr_str
        assert "kappa=2.0" in repr_str
        assert "mu=0.5" in repr_str
        assert "sigma=0.1" in repr_str


class TestCoxIngersollRossProcess:
    """Test Cox-Ingersoll-Ross process properties."""

    def test_initialization_valid(self):
        """Test CIR process initialization with valid parameters."""
        process = CoxIngersollRossProcess(kappa=0.5, mu=0.03, sigma=0.05)
        assert process.kappa == 0.5
        assert process.mu == 0.03
        assert process.sigma == 0.05

    def test_initialization_invalid_parameters(self):
        """Test that non-positive parameters raise ValueError."""
        with pytest.raises(ValueError, match="Mean reversion speed must be positive"):
            CoxIngersollRossProcess(kappa=0.0, mu=0.03, sigma=0.05)

        with pytest.raises(ValueError, match="Long-term mean must be positive"):
            CoxIngersollRossProcess(kappa=0.5, mu=0.0, sigma=0.05)

        with pytest.raises(ValueError, match="Volatility must be positive"):
            CoxIngersollRossProcess(kappa=0.5, mu=0.03, sigma=0.0)

    def test_feller_condition_warning(self):
        """Test warning when Feller condition not satisfied."""
        # Feller condition: 2κμ ≥ σ²
        # Violate: 2*0.1*0.01 = 0.002 < 0.1² = 0.01
        with pytest.warns(UserWarning, match="Feller condition not satisfied"):
            CoxIngersollRossProcess(kappa=0.1, mu=0.01, sigma=0.1)

    def test_drift_formula(self):
        """Test drift coefficient κ(μ - θ)."""
        process = CoxIngersollRossProcess(kappa=0.5, mu=0.03, sigma=0.05)

        theta = np.array([0.01, 0.03, 0.05])
        expected_drift = 0.5 * (0.03 - theta)

        drift = process.drift(theta, t=0.0)
        np.testing.assert_allclose(drift, expected_drift)

    def test_diffusion_square_root(self):
        """Test diffusion coefficient σ√θ."""
        process = CoxIngersollRossProcess(kappa=0.5, mu=0.03, sigma=0.05)

        theta = np.array([0.01, 0.04, 0.09])
        expected_diffusion = 0.05 * np.sqrt(theta)

        diffusion = process.diffusion(theta, t=0.0)
        np.testing.assert_allclose(diffusion, expected_diffusion, rtol=1e-10)

    def test_diffusion_zero_protection(self):
        """Test diffusion handles negative values gracefully."""
        process = CoxIngersollRossProcess(kappa=0.5, mu=0.03, sigma=0.05)

        # Should not raise error for negative (non-physical) values
        theta = np.array([-0.01, 0.0, 0.01])
        diffusion = process.diffusion(theta, t=0.0)

        # Negative treated as zero
        expected = 0.05 * np.sqrt(np.maximum(theta, 0))
        np.testing.assert_allclose(diffusion, expected)

    def test_sample_path_positivity(self):
        """Test sample path stays positive."""
        # Use parameters satisfying Feller condition
        process = CoxIngersollRossProcess(kappa=2.0, mu=0.05, sigma=0.1)

        path = process.sample_path(T=1.0, Nt=1000, theta0=0.05, seed=42)

        assert np.all(path >= 0), "CIR process violated positivity"

    def test_mean_reversion(self):
        """Test long-term convergence to mean μ."""
        process = CoxIngersollRossProcess(kappa=2.0, mu=0.05, sigma=0.1)

        # Generate long path
        path = process.sample_path(T=10.0, Nt=10000, theta0=0.1, seed=42)

        # Check long-term mean
        long_term_mean = path[-2000:].mean()
        assert abs(long_term_mean - 0.05) < 0.03


class TestGeometricBrownianMotion:
    """Test Geometric Brownian Motion properties."""

    def test_initialization_valid(self):
        """Test GBM initialization with valid parameters."""
        process = GeometricBrownianMotion(mu=0.10, sigma=0.20)
        assert process.mu == 0.10
        assert process.sigma == 0.20

    def test_initialization_invalid_sigma(self):
        """Test that non-positive sigma raises ValueError."""
        with pytest.raises(ValueError, match="Volatility must be positive"):
            GeometricBrownianMotion(mu=0.10, sigma=0.0)

    def test_drift_proportional(self):
        """Test drift coefficient μθ."""
        process = GeometricBrownianMotion(mu=0.10, sigma=0.20)

        theta = np.array([1.0, 2.0, 3.0])
        expected_drift = 0.10 * theta

        drift = process.drift(theta, t=0.0)
        np.testing.assert_allclose(drift, expected_drift)

    def test_diffusion_proportional(self):
        """Test diffusion coefficient σθ."""
        process = GeometricBrownianMotion(mu=0.10, sigma=0.20)

        theta = np.array([1.0, 2.0, 3.0])
        expected_diffusion = 0.20 * theta

        diffusion = process.diffusion(theta, t=0.0)
        np.testing.assert_allclose(diffusion, expected_diffusion)

    def test_sample_path_positivity(self):
        """Test sample path stays positive."""
        process = GeometricBrownianMotion(mu=0.10, sigma=0.20)

        path = process.sample_path(T=1.0, Nt=252, theta0=100.0, seed=42)

        assert np.all(path > 0), "GBM violated positivity"

    def test_sample_path_initial_condition_error(self):
        """Test that non-positive initial value raises error."""
        process = GeometricBrownianMotion(mu=0.10, sigma=0.20)

        with pytest.raises(ValueError, match="Initial value must be positive"):
            process.sample_path(T=1.0, Nt=100, theta0=0.0)

    def test_expected_growth(self):
        """Test average growth rate matches drift μ."""
        process = GeometricBrownianMotion(mu=0.10, sigma=0.20)

        # Generate many paths and compute average growth
        num_paths = 1000
        T = 1.0
        theta0 = 100.0

        final_values = []
        for seed in range(num_paths):
            path = process.sample_path(T=T, Nt=252, theta0=theta0, seed=seed)
            final_values.append(path[-1])

        # E[θ_T / θ_0] = e^{μT}
        average_ratio = np.mean(final_values) / theta0
        expected_ratio = np.exp(process.mu * T)

        # Should be close (within 2 standard errors)
        assert abs(average_ratio - expected_ratio) / expected_ratio < 0.1


class TestJumpDiffusionProcess:
    """Test Jump Diffusion Process properties."""

    def test_initialization_valid(self):
        """Test jump diffusion initialization with valid parameters."""
        process = JumpDiffusionProcess(
            mu=0.10,
            sigma=0.15,
            jump_intensity=2.0,
            jump_mean=-0.05,
            jump_std=0.03,
        )
        assert process.mu == 0.10
        assert process.sigma == 0.15
        assert process.jump_intensity == 2.0
        assert process.jump_mean == -0.05
        assert process.jump_std == 0.03

    def test_initialization_invalid_parameters(self):
        """Test that invalid parameters raise ValueError."""
        with pytest.raises(ValueError, match="Volatility must be positive"):
            JumpDiffusionProcess(mu=0.10, sigma=0.0, jump_intensity=2.0, jump_mean=-0.05, jump_std=0.03)

        with pytest.raises(ValueError, match="Jump intensity must be positive"):
            JumpDiffusionProcess(mu=0.10, sigma=0.15, jump_intensity=0.0, jump_mean=-0.05, jump_std=0.03)

        with pytest.raises(ValueError, match="Jump std must be positive"):
            JumpDiffusionProcess(mu=0.10, sigma=0.15, jump_intensity=2.0, jump_mean=-0.05, jump_std=0.0)

    def test_drift_constant(self):
        """Test drift is constant μ."""
        process = JumpDiffusionProcess(
            mu=0.10,
            sigma=0.15,
            jump_intensity=2.0,
            jump_mean=-0.05,
            jump_std=0.03,
        )

        theta = np.array([0.0, 1.0, 2.0])
        expected_drift = 0.10 * np.ones(3)

        drift = process.drift(theta, t=0.0)
        np.testing.assert_allclose(drift, expected_drift)

    def test_diffusion_constant(self):
        """Test diffusion is constant σ."""
        process = JumpDiffusionProcess(
            mu=0.10,
            sigma=0.15,
            jump_intensity=2.0,
            jump_mean=-0.05,
            jump_std=0.03,
        )

        theta = np.array([0.0, 1.0, 2.0])
        expected_diffusion = 0.15 * np.ones(3)

        diffusion = process.diffusion(theta, t=0.0)
        np.testing.assert_allclose(diffusion, expected_diffusion)

    def test_sample_path_shape(self):
        """Test sample path has correct shape."""
        process = JumpDiffusionProcess(
            mu=0.10,
            sigma=0.15,
            jump_intensity=2.0,
            jump_mean=-0.05,
            jump_std=0.03,
        )

        path = process.sample_path(T=1.0, Nt=100, seed=42)
        assert path.shape == (101,)

    def test_jump_detection(self):
        """Test that jumps occur with expected frequency."""
        process = JumpDiffusionProcess(
            mu=0.0,  # No drift
            sigma=0.01,  # Small continuous component
            jump_intensity=10.0,  # 10 jumps per unit time expected
            jump_mean=-0.5,  # Large negative jumps
            jump_std=0.1,
        )

        # Generate path
        np.random.seed(42)
        T = 10.0
        Nt = 10000
        path = process.sample_path(T=T, Nt=Nt, theta0=0.0, seed=42)

        # Count significant drops (likely jumps)
        increments = np.diff(path)

        # Jumps should be significantly negative
        large_drops = np.sum(increments < -0.3)  # Threshold for jump detection

        # Expected: λT = 10*10 = 100 jumps
        # With small continuous volatility, most large drops are jumps
        assert 50 < large_drops < 150, f"Expected ~100 jumps, found {large_drops}"

    def test_repr(self):
        """Test string representation."""
        process = JumpDiffusionProcess(
            mu=0.10,
            sigma=0.15,
            jump_intensity=2.0,
            jump_mean=-0.05,
            jump_std=0.03,
        )
        repr_str = repr(process)

        assert "JumpDiffusionProcess" in repr_str
        assert "mu=0.1" in repr_str
        assert "sigma=0.15" in repr_str
        assert "jump_intensity=2.0" in repr_str
