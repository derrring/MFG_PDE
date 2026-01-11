"""
Integration tests for Common Noise MFG solver.

Tests the CommonNoiseMFGSolver with simple analytical cases where solutions
can be verified or compared with deterministic limits.
"""

import pytest

import numpy as np

from mfg_pde.core.stochastic import OrnsteinUhlenbeckProcess, StochasticMFGProblem
from mfg_pde.geometry import TensorProductGrid


@pytest.mark.slow
class TestCommonNoiseMFGSolver:
    """Test suite for Common Noise MFG solver."""

    def test_solver_initialization(self):
        """Test solver can be initialized with valid problem."""
        from mfg_pde.alg.numerical.stochastic import CommonNoiseMFGSolver

        # Create simple stochastic problem
        noise_process = OrnsteinUhlenbeckProcess(kappa=1.0, mu=0.0, sigma=0.1)

        def simple_hamiltonian(x, p, m, theta):
            return 0.5 * p**2 + 0.1 * m

        geometry = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[22])  # Nx=21 -> 22 points
        problem = StochasticMFGProblem(
            geometry=geometry,
            T=0.5,
            Nt=11,
            noise_process=noise_process,
            conditional_hamiltonian=simple_hamiltonian,
        )

        # Create solver
        solver = CommonNoiseMFGSolver(problem, num_noise_samples=10, variance_reduction=False, parallel=False)

        assert solver.K == 10
        assert not solver.variance_reduction
        assert not solver.parallel

    def test_solver_requires_common_noise(self):
        """Test that solver raises error if problem has no common noise."""
        from mfg_pde.alg.numerical.stochastic import CommonNoiseMFGSolver
        from mfg_pde.core import MFGProblem

        # Regular MFG problem without noise
        geometry = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[22])  # Nx=21 -> 22 points
        problem = MFGProblem(geometry=geometry, T=0.5, Nt=11)

        # Should raise ValueError
        with pytest.raises(ValueError, match="must have common noise"):
            CommonNoiseMFGSolver(problem, num_noise_samples=10)

    def test_noise_path_sampling(self):
        """Test noise path sampling with variance reduction."""
        from mfg_pde.alg.numerical.stochastic import CommonNoiseMFGSolver

        noise_process = OrnsteinUhlenbeckProcess(kappa=1.0, mu=0.0, sigma=0.1)

        def simple_hamiltonian(x, p, m, theta):
            return 0.5 * p**2

        geometry = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[22])  # Nx=21 -> 22 points
        problem = StochasticMFGProblem(
            geometry=geometry,
            T=0.5,
            Nt=11,
            noise_process=noise_process,
            conditional_hamiltonian=simple_hamiltonian,
        )

        # Test with variance reduction (quasi-MC)
        solver_qmc = CommonNoiseMFGSolver(problem, num_noise_samples=20, variance_reduction=True, seed=42)
        paths_qmc = solver_qmc._sample_noise_paths()

        assert len(paths_qmc) == 20
        assert all(path.shape == (12,) for path in paths_qmc)  # Nt+1 = 12

        # Test without variance reduction (standard MC)
        solver_mc = CommonNoiseMFGSolver(problem, num_noise_samples=20, variance_reduction=False, seed=42)
        paths_mc = solver_mc._sample_noise_paths()

        assert len(paths_mc) == 20
        assert all(path.shape == (12,) for path in paths_mc)

    @pytest.mark.parametrize("variance_reduction", [True, False])
    def test_zero_noise_recovers_deterministic(self, variance_reduction):
        """Test that zero noise recovers deterministic MFG solution."""
        pytest.skip("Requires conditional MFG solver implementation - placeholder test")

        # This test would verify that when noise variance → 0,
        # the common noise solution converges to deterministic solution

    def test_result_structure(self):
        """Test that solver result has correct structure."""
        pytest.skip("Requires conditional MFG solver implementation - placeholder test")

        # This test would verify CommonNoiseMFGResult structure:
        # - u_mean, m_mean shapes
        # - u_std, m_std shapes
        # - MC error estimates
        # - Confidence intervals


@pytest.mark.unit
class TestCommonNoiseMFGResult:
    """Test suite for CommonNoiseMFGResult class."""

    def test_confidence_interval_computation(self):
        """Test confidence interval calculation."""
        from mfg_pde.alg.numerical.stochastic import CommonNoiseMFGResult

        # Create mock result
        Nt, Nx = 11, 21
        K = 100

        u_samples = [np.random.randn(Nt, Nx) for _ in range(K)]
        m_samples = [np.random.randn(Nt, Nx) for _ in range(K)]

        u_mean = np.mean(u_samples, axis=0)
        m_mean = np.mean(m_samples, axis=0)
        u_std = np.std(u_samples, axis=0, ddof=1)
        m_std = np.std(m_samples, axis=0, ddof=1)

        result = CommonNoiseMFGResult(
            u_mean=u_mean,
            m_mean=m_mean,
            u_std=u_std,
            m_std=m_std,
            u_samples=u_samples,
            m_samples=m_samples,
            noise_paths=[],
            num_noise_samples=K,
            mc_error_u=0.01,
            mc_error_m=0.01,
        )

        # Test confidence interval calculation
        lower_u, upper_u = result.get_confidence_interval_u(confidence=0.95)
        lower_m, upper_m = result.get_confidence_interval_m(confidence=0.95)

        # Check shapes
        assert lower_u.shape == u_mean.shape
        assert upper_u.shape == u_mean.shape
        assert lower_m.shape == m_mean.shape
        assert upper_m.shape == m_mean.shape

        # Check bounds contain mean
        assert np.all(lower_u <= u_mean)
        assert np.all(upper_u >= u_mean)
        assert np.all(lower_m <= m_mean)
        assert np.all(upper_m >= m_mean)

    def test_result_attributes(self):
        """Test CommonNoiseMFGResult has all required attributes."""
        from mfg_pde.alg.numerical.stochastic import CommonNoiseMFGResult

        # Create minimal result
        result = CommonNoiseMFGResult(
            u_mean=np.zeros((10, 20)),
            m_mean=np.zeros((10, 20)),
            u_std=np.zeros((10, 20)),
            m_std=np.zeros((10, 20)),
            u_samples=[],
            m_samples=[],
            noise_paths=[],
            num_noise_samples=50,
            mc_error_u=0.001,
            mc_error_m=0.001,
        )

        # Check all attributes exist
        assert hasattr(result, "u_mean")
        assert hasattr(result, "m_mean")
        assert hasattr(result, "u_std")
        assert hasattr(result, "m_std")
        assert hasattr(result, "u_samples")
        assert hasattr(result, "m_samples")
        assert hasattr(result, "noise_paths")
        assert hasattr(result, "num_noise_samples")
        assert hasattr(result, "mc_error_u")
        assert hasattr(result, "mc_error_m")
        assert hasattr(result, "variance_reduction_factor")
        assert hasattr(result, "computation_time")
        assert hasattr(result, "converged")
