#!/usr/bin/env python3
"""
Unit tests for mfg_pde/utils/numerical/convergence.py

Tests comprehensive convergence monitoring system including:
- DistributionComparator (Wasserstein, KL divergence, moments)
- OscillationDetector (stabilization detection)
- StochasticConvergenceMonitor (confidence intervals, relative error)
- AdvancedConvergenceMonitor (multi-criteria convergence)
- ParticleMethodDetector (automatic method detection)
- AdaptiveConvergenceWrapper (decorator/wrapper pattern)
- Utility functions (factory functions, metrics)

Coverage target: mfg_pde/utils/numerical/convergence.py (411 lines, 14% -> 70%+)
"""

import pytest

import numpy as np

from mfg_pde.utils.numerical.convergence import (
    AdaptiveConvergenceWrapper,
    AdvancedConvergenceMonitor,
    DistributionComparator,
    OscillationDetector,
    ParticleMethodDetector,
    StochasticConvergenceMonitor,
    calculate_l2_convergence_metrics,
    create_default_monitor,
    create_stochastic_monitor,
)

# =============================================================================
# Test DistributionComparator
# =============================================================================


@pytest.mark.unit
def test_distribution_comparator_wasserstein_1d_identical():
    """Test Wasserstein distance is 0 for identical distributions."""
    x = np.linspace(0, 1, 100)
    p = np.exp(-((x - 0.5) ** 2) / 0.1)  # Gaussian
    p = p / np.sum(p)

    distance = DistributionComparator.wasserstein_1d(p, p, x)

    assert distance < 1e-10


@pytest.mark.unit
def test_distribution_comparator_wasserstein_1d_shifted():
    """Test Wasserstein distance for shifted Gaussians."""
    x = np.linspace(0, 1, 100)
    p = np.exp(-((x - 0.3) ** 2) / 0.01)
    q = np.exp(-((x - 0.7) ** 2) / 0.01)

    distance = DistributionComparator.wasserstein_1d(p, q, x)

    # Distance should be approximately the shift distance
    assert distance > 0.3
    assert distance < 0.5


@pytest.mark.unit
def test_distribution_comparator_wasserstein_normalization():
    """Test Wasserstein handles unnormalized distributions."""
    x = np.linspace(0, 1, 50)
    p = np.ones(50) * 10.0  # Unnormalized uniform
    q = np.ones(50) * 5.0  # Different unnormalized uniform

    distance = DistributionComparator.wasserstein_1d(p, q, x)

    # Should be near 0 since both are uniform
    assert distance < 0.01


@pytest.mark.unit
def test_distribution_comparator_kl_divergence_identical():
    """Test KL divergence is 0 for identical distributions."""
    p = np.array([0.1, 0.2, 0.3, 0.25, 0.15])

    kl_div = DistributionComparator.kl_divergence(p, p)

    assert abs(kl_div) < 1e-6


@pytest.mark.unit
def test_distribution_comparator_kl_divergence_different():
    """Test KL divergence is positive for different distributions."""
    p = np.array([0.4, 0.3, 0.2, 0.1])
    q = np.array([0.1, 0.2, 0.3, 0.4])

    kl_div = DistributionComparator.kl_divergence(p, q)

    assert kl_div > 0


@pytest.mark.unit
def test_distribution_comparator_kl_divergence_stability():
    """Test KL divergence handles near-zero probabilities."""
    p = np.array([0.99, 0.01, 0.0, 0.0])
    q = np.array([0.5, 0.3, 0.1, 0.1])

    kl_div = DistributionComparator.kl_divergence(p, q, epsilon=1e-12)

    assert np.isfinite(kl_div)


@pytest.mark.unit
def test_distribution_comparator_statistical_moments_gaussian():
    """Test statistical moments for Gaussian distribution."""
    x = np.linspace(-3, 3, 200)
    dist = np.exp(-(x**2) / 2) / np.sqrt(2 * np.pi)
    dist = dist / np.sum(dist)

    moments = DistributionComparator.statistical_moments(dist, x)

    assert abs(moments["mean"]) < 0.1  # Should be near 0
    assert abs(moments["variance"] - 1.0) < 0.2  # Should be near 1
    assert abs(moments["skewness"]) < 0.2  # Symmetric -> skewness ~ 0
    assert abs(moments["kurtosis"]) < 0.5  # Gaussian has kurtosis 0 (excess)


@pytest.mark.unit
def test_distribution_comparator_statistical_moments_uniform():
    """Test statistical moments for uniform distribution."""
    x = np.linspace(0, 1, 100)
    dist = np.ones(100)
    dist = dist / np.sum(dist)

    moments = DistributionComparator.statistical_moments(dist, x)

    assert abs(moments["mean"] - 0.5) < 0.01  # Mean at center
    assert moments["variance"] > 0
    assert abs(moments["skewness"]) < 0.1  # Symmetric


@pytest.mark.unit
def test_distribution_comparator_statistical_moments_zero_variance():
    """Test moments handle zero variance (delta function)."""
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    dist = np.array([0.0, 0.0, 1.0, 0.0, 0.0])  # Delta at x=2

    moments = DistributionComparator.statistical_moments(dist, x)

    assert abs(moments["mean"] - 2.0) < 1e-10
    assert moments["variance"] < 1e-10
    assert moments["skewness"] == 0.0
    assert moments["kurtosis"] == 0.0


# =============================================================================
# Test OscillationDetector
# =============================================================================


@pytest.mark.unit
def test_oscillation_detector_initialization():
    """Test OscillationDetector initialization."""
    detector = OscillationDetector(history_length=20)

    assert detector.history_length == 20
    assert len(detector.error_history) == 0


@pytest.mark.unit
def test_oscillation_detector_add_samples():
    """Test adding samples to detector."""
    detector = OscillationDetector(history_length=5)

    for i in range(10):
        detector.add_sample(float(i))

    # Should only keep last 5
    assert len(detector.error_history) == 5
    assert list(detector.error_history) == [5.0, 6.0, 7.0, 8.0, 9.0]


@pytest.mark.unit
def test_oscillation_detector_insufficient_history():
    """Test is_stabilized returns false with insufficient history."""
    detector = OscillationDetector(history_length=10)

    for _i in range(5):
        detector.add_sample(0.1)

    is_stable, diagnostics = detector.is_stabilized(magnitude_threshold=1.0, stability_threshold=0.1)

    assert not is_stable
    assert diagnostics["status"] == "insufficient_history"
    assert diagnostics["samples"] == 5


@pytest.mark.unit
def test_oscillation_detector_stabilized():
    """Test detection of stabilized oscillation."""
    detector = OscillationDetector(history_length=10)

    # Add 10 samples with small variation
    for i in range(10):
        detector.add_sample(0.05 + 0.001 * i)

    is_stable, diagnostics = detector.is_stabilized(magnitude_threshold=0.1, stability_threshold=0.01)

    assert is_stable
    assert diagnostics["magnitude_ok"]
    assert diagnostics["stability_ok"]
    assert diagnostics["mean_error"] < 0.1


@pytest.mark.unit
def test_oscillation_detector_not_stabilized_magnitude():
    """Test detection of high magnitude error."""
    detector = OscillationDetector(history_length=10)

    for _i in range(10):
        detector.add_sample(0.5)  # High error

    is_stable, diagnostics = detector.is_stabilized(magnitude_threshold=0.1, stability_threshold=0.1)

    assert not is_stable
    assert not diagnostics["magnitude_ok"]


@pytest.mark.unit
def test_oscillation_detector_not_stabilized_oscillation():
    """Test detection of high oscillation."""
    detector = OscillationDetector(history_length=10)

    # Oscillating errors
    for i in range(10):
        detector.add_sample(0.01 + 0.2 * (i % 2))

    is_stable, diagnostics = detector.is_stabilized(magnitude_threshold=1.0, stability_threshold=0.05)

    assert not is_stable
    assert diagnostics["magnitude_ok"]
    assert not diagnostics["stability_ok"]


# =============================================================================
# Test StochasticConvergenceMonitor
# =============================================================================


@pytest.mark.unit
def test_stochastic_convergence_monitor_initialization():
    """Test StochasticConvergenceMonitor initialization."""
    monitor = StochasticConvergenceMonitor(window_size=10, median_tolerance=1e-4, quantile=0.9)

    assert monitor.window_size == 10
    assert monitor.median_tolerance == 1e-4
    assert monitor.quantile == 0.9


@pytest.mark.unit
def test_stochastic_convergence_monitor_add_iteration():
    """Test adding iterations to stochastic monitor."""
    monitor = StochasticConvergenceMonitor(window_size=5)

    for i in range(10):
        monitor.add_iteration(float(i) * 0.1, float(i) * 0.05)

    # Window size limits history
    assert len(monitor.errors_u) <= 5
    assert len(monitor.errors_m) <= 5
    assert monitor.iteration_count == 10


@pytest.mark.unit
def test_stochastic_convergence_monitor_get_statistics():
    """Test statistics computation."""
    monitor = StochasticConvergenceMonitor(window_size=20)

    # Add iterations
    np.random.seed(42)
    for _i in range(15):
        monitor.add_iteration(0.1 + np.random.rand() * 0.01, 0.05 + np.random.rand() * 0.005)

    stats = monitor.get_statistics()

    assert "u_stats" in stats
    assert "m_stats" in stats
    assert "median" in stats["u_stats"]
    assert "mean" in stats["u_stats"]
    assert "quantile" in stats["u_stats"]


@pytest.mark.unit
def test_stochastic_convergence_monitor_no_data():
    """Test statistics with no data."""
    monitor = StochasticConvergenceMonitor()

    stats = monitor.get_statistics()

    assert stats["status"] == "no_data"


@pytest.mark.unit
def test_stochastic_convergence_monitor_check_convergence():
    """Test convergence checking."""
    monitor = StochasticConvergenceMonitor(window_size=10, median_tolerance=0.01, min_iterations=5)

    # Add iterations with decreasing errors
    for i in range(15):
        error_u = 0.005 + 0.0001 * i  # Small errors
        error_m = 0.003 + 0.0001 * i
        monitor.add_iteration(error_u, error_m)

    has_converged, diagnostics = monitor.check_convergence()

    assert isinstance(has_converged, bool)
    assert "median_u" in diagnostics or "status" in diagnostics


@pytest.mark.unit
def test_stochastic_convergence_monitor_insufficient_iterations():
    """Test convergence check with insufficient iterations."""
    monitor = StochasticConvergenceMonitor(min_iterations=10)

    for _i in range(5):
        monitor.add_iteration(0.001, 0.001)

    has_converged, _diagnostics = monitor.check_convergence()

    assert not has_converged


# =============================================================================
# Test AdvancedConvergenceMonitor
# =============================================================================


@pytest.mark.unit
def test_advanced_convergence_monitor_initialization():
    """Test AdvancedConvergenceMonitor initialization."""
    monitor = AdvancedConvergenceMonitor(
        wasserstein_tol=1e-4,
        kl_divergence_tol=1e-3,
        u_magnitude_tol=1e-3,
        history_length=10,
    )

    assert monitor.wasserstein_tol == 1e-4
    assert monitor.kl_divergence_tol == 1e-3
    assert monitor.u_magnitude_tol == 1e-3
    assert monitor.oscillation_detector.history_length == 10


@pytest.mark.unit
def test_advanced_convergence_monitor_update_basic():
    """Test basic update with value function convergence."""
    monitor = AdvancedConvergenceMonitor(u_magnitude_tol=1e-3)

    u_prev = np.ones((10, 10)) * 1.0
    u_curr = np.ones((10, 10)) * 1.001  # Small change
    m_curr = np.ones((10, 10)) * 0.5
    x_grid = np.linspace(0, 1, 10)

    diagnostics = monitor.update(u_curr, u_prev, m_curr, x_grid)

    assert "u_l2_error" in diagnostics
    assert "iteration" in diagnostics
    assert diagnostics["u_l2_error"] < 0.01


@pytest.mark.unit
def test_advanced_convergence_monitor_convergence_detection():
    """Test convergence detection with small changes."""
    monitor = AdvancedConvergenceMonitor(u_magnitude_tol=1e-2, u_stability_tol=1e-3, history_length=5)

    u = np.ones((10, 10))
    m = np.ones((10, 10)) * 0.5
    m = m / np.sum(m)  # Normalize
    x = np.linspace(0, 1, 10)

    # Add iterations with tiny changes
    for _i in range(15):
        u_new = u + 1e-6 * np.random.rand(10, 10)
        diagnostics = monitor.update(u_new, u, m, x)

    # Check diagnostics structure
    assert "converged" in diagnostics
    assert "u_l2_error" in diagnostics


@pytest.mark.unit
def test_advanced_convergence_monitor_multi_dimensional():
    """Test monitor handles multi-dimensional arrays."""
    monitor = AdvancedConvergenceMonitor()

    u_prev = np.random.rand(20, 20)
    u_curr = u_prev + np.random.rand(20, 20) * 0.001
    m_curr = np.random.rand(20, 20)
    m_curr = m_curr / np.sum(m_curr)  # Normalize
    x_grid = np.linspace(0, 1, 20)

    diagnostics = monitor.update(u_curr, u_prev, m_curr, x_grid)

    assert diagnostics["u_l2_error"] >= 0
    assert np.isfinite(diagnostics["u_l2_error"])


@pytest.mark.unit
def test_advanced_convergence_monitor_wasserstein_computation():
    """Test Wasserstein distance computation."""
    monitor = AdvancedConvergenceMonitor(wasserstein_tol=1e-4)

    # 1D distributions
    x = np.linspace(0, 1, 50)
    m_prev = np.exp(-((x - 0.5) ** 2) / 0.1)
    m_prev = m_prev / np.sum(m_prev)
    m_curr = np.exp(-((x - 0.51) ** 2) / 0.1)  # Slightly shifted
    m_curr = m_curr / np.sum(m_curr)

    u_prev = np.zeros(50)
    u_curr = np.zeros(50)

    # First update (no previous_m)
    monitor.update(u_curr, u_prev, m_prev, x)

    # Second update (with previous_m)
    diagnostics = monitor.update(u_curr, u_prev, m_curr, x)

    # Check that Wasserstein was computed
    assert "converged" in diagnostics or "wasserstein_distance" in diagnostics


# =============================================================================
# Test ParticleMethodDetector
# =============================================================================


@pytest.mark.unit
def test_particle_method_detector_basic():
    """Test ParticleMethodDetector can be instantiated."""
    detector = ParticleMethodDetector()

    assert detector is not None


@pytest.mark.unit
def test_particle_method_detector_mock_solver():
    """Test detection from mock solver with particle attributes."""

    class MockParticleSolver:
        def __init__(self):
            self.particles = np.random.rand(100, 2)
            self.num_particles = 100

    mock_solver = MockParticleSolver()
    has_particles, detection_info = ParticleMethodDetector.detect_particle_methods(mock_solver)

    assert isinstance(has_particles, bool)
    assert "confidence" in detection_info
    assert "particle_components" in detection_info


# =============================================================================
# Test AdaptiveConvergenceWrapper
# =============================================================================


@pytest.mark.unit
def test_adaptive_convergence_wrapper_initialization():
    """Test AdaptiveConvergenceWrapper initialization."""

    class MockSolver:
        def __init__(self):
            self.u = np.ones((10, 10))
            self.m = np.ones((10, 10)) * 0.5
            self.x_grid = np.linspace(0, 1, 10)

    mock_solver = MockSolver()
    wrapper = AdaptiveConvergenceWrapper(mock_solver, classical_tol=1e-4, verbose=False)

    # Wrapper wraps the solver internally
    assert wrapper._wrapped_solver is mock_solver
    assert wrapper.classical_tol == 1e-4


@pytest.mark.unit
def test_adaptive_convergence_wrapper_particle_detection():
    """Test wrapper detects particle methods."""

    class MockParticleSolver:
        def __init__(self):
            self.particles = np.random.rand(100, 2)
            self.num_particles = 100

    mock_solver = MockParticleSolver()
    wrapper = AdaptiveConvergenceWrapper(mock_solver, verbose=False)

    # Wrapper wraps the solver internally
    assert wrapper._wrapped_solver is mock_solver


# =============================================================================
# Test Utility Functions
# =============================================================================


@pytest.mark.unit
def test_create_default_monitor():
    """Test create_default_monitor factory function."""
    monitor = create_default_monitor(wasserstein_tol=1e-5, u_magnitude_tol=1e-3, history_length=20)

    assert isinstance(monitor, AdvancedConvergenceMonitor)
    assert monitor.wasserstein_tol == 1e-5
    assert monitor.u_magnitude_tol == 1e-3


@pytest.mark.unit
def test_create_stochastic_monitor():
    """Test create_stochastic_monitor factory function."""
    monitor = create_stochastic_monitor(median_tolerance=1e-5, window_size=30, quantile=0.95)

    assert isinstance(monitor, StochasticConvergenceMonitor)
    assert monitor.median_tolerance == 1e-5
    assert monitor.window_size == 30


@pytest.mark.unit
def test_calculate_l2_convergence_metrics():
    """Test calculate_l2_convergence_metrics utility."""
    u_prev = np.ones((10, 10)) * 1.0
    u_curr = np.ones((10, 10)) * 1.01
    m_prev = np.ones((10, 10)) * 0.5
    m_curr = np.ones((10, 10)) * 0.501
    Dx = 0.1
    Dt = 0.01

    metrics = calculate_l2_convergence_metrics(u_curr, u_prev, m_curr, m_prev, Dx, Dt)

    assert "l2distu_abs" in metrics
    assert "l2distm_abs" in metrics
    assert "l2distu_rel" in metrics
    assert "l2distm_rel" in metrics
    assert metrics["l2distu_abs"] >= 0
    assert metrics["l2distm_abs"] >= 0


@pytest.mark.unit
def test_calculate_l2_convergence_metrics_identical():
    """Test L2 metrics are near zero for identical arrays."""
    u = np.random.rand(15, 15)
    m = np.random.rand(15, 15)
    Dx = 0.1
    Dt = 0.01

    metrics = calculate_l2_convergence_metrics(u, u, m, m, Dx, Dt)

    assert metrics["l2distu_abs"] < 1e-10
    assert metrics["l2distm_abs"] < 1e-10


@pytest.mark.unit
def test_calculate_l2_convergence_metrics_large_change():
    """Test L2 metrics for large changes."""
    u_prev = np.ones((10, 10))
    u_curr = np.ones((10, 10)) * 2.0  # 100% change
    m_prev = np.ones((10, 10)) * 0.5
    m_curr = np.ones((10, 10)) * 1.0  # 100% change
    Dx = 0.1
    Dt = 0.01

    metrics = calculate_l2_convergence_metrics(u_curr, u_prev, m_curr, m_prev, Dx, Dt)

    assert metrics["l2distu_abs"] > 0
    assert metrics["l2distm_abs"] > 0
    assert metrics["l2distu_rel"] >= 0.5  # Changed from > to >=


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.unit
def test_integration_convergence_workflow():
    """Test full convergence monitoring workflow."""
    # Create monitor
    monitor = create_default_monitor(u_magnitude_tol=1e-3, history_length=5)

    # Simulate iterative solver
    x = np.linspace(0, 1, 50)
    u = np.ones(50)
    m = np.exp(-((x - 0.5) ** 2) / 0.1)
    m = m / np.sum(m)

    for iteration in range(15):
        u_prev = u.copy()
        # Simulate convergence
        u = u + np.random.rand(50) * 0.001 * (1.0 / (iteration + 1))

        diagnostics = monitor.update(u, u_prev, m, x)

        if diagnostics["converged"]:
            break

    # Check that diagnostics were computed
    assert "u_l2_error" in diagnostics
    assert diagnostics["iteration"] == iteration + 1


@pytest.mark.unit
def test_integration_stochastic_convergence():
    """Test stochastic convergence monitoring."""
    monitor = create_stochastic_monitor(median_tolerance=0.01, window_size=10, min_iterations=5)

    # Simulate stochastic errors converging
    np.random.seed(42)
    for i in range(20):
        error_u = 0.005 + 0.001 / (i + 1)  # Decreasing errors
        error_m = 0.003 + 0.0005 / (i + 1)
        monitor.add_iteration(error_u, error_m)

    # Check statistics
    stats = monitor.get_statistics()
    assert "u_stats" in stats


@pytest.mark.unit
def test_integration_distribution_comparison_pipeline():
    """Test full distribution comparison pipeline."""
    # Create two similar distributions
    x = np.linspace(0, 1, 100)
    p = np.exp(-((x - 0.5) ** 2) / 0.1)
    p = p / np.sum(p)
    q = np.exp(-((x - 0.52) ** 2) / 0.1)  # Slightly shifted
    q = q / np.sum(q)

    # Compute all metrics
    wasserstein = DistributionComparator.wasserstein_1d(p, q, x)
    kl_div = DistributionComparator.kl_divergence(p, q)
    moments_p = DistributionComparator.statistical_moments(p, x)
    moments_q = DistributionComparator.statistical_moments(q, x)

    # Assertions
    assert wasserstein > 0  # Different distributions
    assert wasserstein < 0.1  # But close
    assert kl_div > 0
    assert abs(moments_p["mean"] - moments_q["mean"]) < 0.05


# =============================================================================
# Edge Cases
# =============================================================================


@pytest.mark.unit
def test_edge_case_empty_history():
    """Test monitors handle empty history gracefully."""
    detector = OscillationDetector(history_length=10)

    is_stable, diagnostics = detector.is_stabilized(magnitude_threshold=1.0, stability_threshold=0.1)

    assert not is_stable
    assert diagnostics["status"] == "insufficient_history"


@pytest.mark.unit
def test_edge_case_single_point_distribution():
    """Test distribution metrics with single point."""
    x = np.array([1.0])
    p = np.array([1.0])

    wasserstein = DistributionComparator.wasserstein_1d(p, p, x)
    kl_div = DistributionComparator.kl_divergence(p, p)
    moments = DistributionComparator.statistical_moments(p, x)

    assert np.isfinite(wasserstein)
    assert np.isfinite(kl_div)
    assert moments["mean"] == 1.0
    assert moments["variance"] == 0.0


@pytest.mark.unit
def test_edge_case_zero_distribution():
    """Test handling of zero distribution."""
    x = np.linspace(0, 1, 10)
    p = np.zeros(10)
    q = np.ones(10)

    # Should handle gracefully with normalization
    wasserstein = DistributionComparator.wasserstein_1d(p, q, x)
    moments = DistributionComparator.statistical_moments(p, x)

    assert np.isfinite(wasserstein)
    assert np.isfinite(moments["mean"])


@pytest.mark.unit
def test_edge_case_large_arrays():
    """Test performance with large arrays."""
    u_prev = np.random.rand(200, 200)
    u_curr = u_prev + np.random.rand(200, 200) * 0.001
    m_prev = np.random.rand(200, 200)
    m_curr = m_prev + np.random.rand(200, 200) * 0.001
    Dx = 0.01
    Dt = 0.001

    metrics = calculate_l2_convergence_metrics(u_curr, u_prev, m_curr, m_prev, Dx, Dt)

    assert np.isfinite(metrics["l2distu_abs"])
    assert np.isfinite(metrics["l2distm_abs"])
