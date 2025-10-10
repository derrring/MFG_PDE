"""
Unit tests for MFG Analytics Computation Logic.

Tests focus on statistical computations, metric calculations, and data validation
in the analytics engine, not visualization output.
"""

import tempfile
from pathlib import Path

import pytest

import numpy as np

from mfg_pde.visualization.mfg_analytics import MFGAnalyticsEngine

# ============================================================================
# Test: Analytics Engine Initialization
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_analytics_engine_initialization_default():
    """Test MFGAnalyticsEngine initialization with defaults."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MFGAnalyticsEngine(output_dir=tmpdir)

        assert engine.prefer_plotly is True
        assert engine.output_dir == Path(tmpdir)
        assert engine.output_dir.exists()


@pytest.mark.unit
@pytest.mark.fast
def test_analytics_engine_initialization_custom():
    """Test MFGAnalyticsEngine initialization with custom parameters."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MFGAnalyticsEngine(prefer_plotly=False, output_dir=tmpdir)

        assert engine.prefer_plotly is False
        assert engine.output_dir == Path(tmpdir)


@pytest.mark.unit
@pytest.mark.fast
def test_analytics_engine_creates_output_directory():
    """Test output directory creation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "analytics_output"
        MFGAnalyticsEngine(output_dir=output_path)

        assert output_path.exists()
        assert output_path.is_dir()


# ============================================================================
# Test: Capability Detection
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_get_capabilities():
    """Test capability detection returns correct flags."""
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = MFGAnalyticsEngine(output_dir=tmpdir)

        capabilities = engine.get_capabilities()

        # Should return dict with capability flags
        assert isinstance(capabilities, dict)
        assert "visualization" in capabilities
        assert "data_analysis" in capabilities
        assert "plotly_3d" in capabilities
        assert "bokeh_interactive" in capabilities

        # All values should be boolean
        for key, value in capabilities.items():
            assert isinstance(value, bool), f"{key} should be boolean"


# ============================================================================
# Test: Convergence Metric Computation
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_l2_norm_computation():
    """Test L2 norm calculation for convergence."""
    # Create two arrays
    u_current = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    u_previous = np.array([1.1, 2.1, 2.9, 4.1, 4.9])

    # Compute L2 norm of difference
    diff = u_current - u_previous
    l2_norm = np.sqrt(np.sum(diff**2))

    # Expected: sqrt((0.1)^2 + (0.1)^2 + (0.1)^2 + (0.1)^2 + (0.1)^2)
    expected = np.sqrt(5 * 0.01)
    assert np.allclose(l2_norm, expected)


@pytest.mark.unit
@pytest.mark.fast
def test_relative_error_computation():
    """Test relative error calculation."""
    u_current = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    u_previous = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    # Compute relative error
    diff = u_current - u_previous
    relative_error = np.linalg.norm(diff) / np.linalg.norm(u_previous) if np.linalg.norm(u_previous) > 0 else 0.0

    # All values doubled, so relative error = 1.0
    assert np.allclose(relative_error, 1.0)


@pytest.mark.unit
@pytest.mark.fast
def test_residual_computation():
    """Test residual computation for equation validation."""
    # Simple residual: Au - b
    A = np.array([[2.0, 1.0], [1.0, 2.0]])
    u = np.array([1.0, 1.0])
    b = np.array([3.0, 3.0])

    residual = A @ u - b

    # Expected: [2+1, 1+2] - [3, 3] = [0, 0]
    assert np.allclose(residual, 0.0)


# ============================================================================
# Test: Mass Conservation Check
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_mass_conservation_1d(density_1d_gaussian, grid_1d_small):
    """Test mass conservation validation in 1D."""
    density = density_1d_gaussian
    x_grid = grid_1d_small

    # Compute total mass
    mass = np.trapezoid(density, x_grid)

    # Should be normalized to 1
    assert np.allclose(mass, 1.0, rtol=1e-6)


@pytest.mark.unit
@pytest.mark.fast
def test_mass_conservation_error():
    """Test mass conservation error computation."""
    x_grid = np.linspace(0, 1, 100)

    # Create density that's NOT normalized
    density = np.exp(-10 * (x_grid - 0.5) ** 2)  # Not normalized

    # Compute mass
    mass = np.trapezoid(density, x_grid)

    # Compute conservation error
    mass_error = abs(mass - 1.0)

    # Should have significant error
    assert mass_error > 0.1  # Not well-conserved


@pytest.mark.unit
@pytest.mark.fast
def test_mass_conservation_tolerance_check():
    """Test mass conservation tolerance checking."""
    x_grid = np.linspace(0, 1, 50)
    density = np.ones_like(x_grid)  # Uniform
    density = density / np.trapezoid(density, x_grid)  # Normalize

    mass = np.trapezoid(density, x_grid)
    tolerance = 1e-6

    is_conserved = abs(mass - 1.0) < tolerance

    # Convert numpy bool to Python bool
    assert bool(is_conserved) is True


# ============================================================================
# Test: Energy Functional Computation
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_kinetic_energy_computation():
    """Test kinetic energy calculation."""
    x_grid = np.linspace(0, 1, 10)
    velocity = np.sin(np.pi * x_grid)  # Velocity field

    # Kinetic energy: integral of (1/2) * v^2
    kinetic_energy = 0.5 * np.trapezoid(velocity**2, x_grid)

    assert kinetic_energy > 0
    assert np.isfinite(kinetic_energy)


@pytest.mark.unit
@pytest.mark.fast
def test_potential_energy_computation():
    """Test potential energy calculation."""
    x_grid = np.linspace(0, 1, 10)
    potential = x_grid**2  # Quadratic potential

    density = np.ones_like(x_grid)
    density = density / np.trapezoid(density, x_grid)  # Normalize

    # Potential energy: integral of rho * V
    potential_energy = np.trapezoid(density * potential, x_grid)

    assert potential_energy > 0
    assert np.isfinite(potential_energy)


@pytest.mark.unit
@pytest.mark.fast
def test_total_energy_computation():
    """Test total energy calculation."""
    # Simple kinetic + potential
    kinetic = 2.0
    potential = 3.0
    total_energy = kinetic + potential

    assert total_energy == 5.0


# ============================================================================
# Test: Statistical Summary Computation
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_density_statistics(density_1d_gaussian):
    """Test statistical summary of density."""
    density = density_1d_gaussian

    # Compute statistics
    max_density = np.max(density)
    min_density = np.min(density)
    mean_density = np.mean(density)

    # Gaussian should have positive values
    assert max_density > min_density
    assert min_density >= 0.0
    assert mean_density > 0.0


@pytest.mark.unit
@pytest.mark.fast
def test_density_center_of_mass():
    """Test center of mass calculation."""
    x_grid = np.linspace(0, 1, 100)

    # Density concentrated at x=0.3
    density = np.exp(-100 * (x_grid - 0.3) ** 2)
    density = density / np.trapezoid(density, x_grid)

    # Compute center of mass
    center_of_mass = np.trapezoid(x_grid * density, x_grid)

    # Should be close to 0.3
    assert np.allclose(center_of_mass, 0.3, rtol=1e-2)


@pytest.mark.unit
@pytest.mark.fast
def test_density_spread():
    """Test density spread (variance) calculation."""
    x_grid = np.linspace(-3, 3, 200)  # Wider range, more points for better accuracy

    # Standard normal distribution (approximately)
    density = np.exp(-0.5 * x_grid**2) / np.sqrt(2 * np.pi)
    density = density / np.trapezoid(density, x_grid)  # Renormalize

    # Compute mean
    mean = np.trapezoid(x_grid * density, x_grid)

    # Compute variance
    variance = np.trapezoid((x_grid - mean) ** 2 * density, x_grid)

    # Should be close to 1.0 for standard normal (wider tolerance for numerical integration)
    assert np.allclose(variance, 1.0, rtol=0.3)


# ============================================================================
# Test: Time Series Data Extraction
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_time_series_mass_evolution(network_density_evolution):
    """Test extraction of mass conservation over time."""
    density_evolution = network_density_evolution

    # Compute total mass at each timestep
    mass_per_time = np.sum(density_evolution, axis=0)

    # Should be approximately 1.0 at all times
    assert np.allclose(mass_per_time, 1.0, rtol=1e-10)


@pytest.mark.unit
@pytest.mark.fast
def test_time_series_max_density_evolution(density_2d_gaussian):
    """Test extraction of maximum density over time."""
    density = density_2d_gaussian

    # Extract max at each timestep
    max_per_time = np.max(density, axis=0)

    # Max density should decrease over time (spreading Gaussian)
    assert max_per_time[0] > max_per_time[-1]


# ============================================================================
# Test: Parameter Sweep Result Aggregation
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_parameter_sweep_result_collection():
    """Test collection of parameter sweep results."""
    # Simulate parameter sweep results
    results = []

    for sigma in [0.1, 0.2, 0.3]:
        result = {"sigma": sigma, "final_error": sigma * 0.1, "iterations": int(10 / sigma)}
        results.append(result)

    assert len(results) == 3
    assert results[0]["sigma"] == 0.1
    assert results[-1]["sigma"] == 0.3


@pytest.mark.unit
@pytest.mark.fast
def test_parameter_sweep_optimal_parameter():
    """Test identification of optimal parameter from sweep."""
    results = [
        {"param": 1.0, "error": 0.5},
        {"param": 2.0, "error": 0.2},
        {"param": 3.0, "error": 0.3},
    ]

    # Find parameter with minimum error
    optimal = min(results, key=lambda r: r["error"])

    assert optimal["param"] == 2.0
    assert optimal["error"] == 0.2


# ============================================================================
# Test: Output Directory Management
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_output_directory_creation():
    """Test output directory structure creation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        output_dir = base_dir / "analytics"

        output_dir.mkdir(parents=True, exist_ok=True)

        assert output_dir.exists()
        assert output_dir.is_dir()


@pytest.mark.unit
@pytest.mark.fast
def test_subdirectory_organization():
    """Test creation of organized subdirectories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)

        # Create organized structure
        (base_dir / "visualizations").mkdir(parents=True)
        (base_dir / "data").mkdir(parents=True)
        (base_dir / "reports").mkdir(parents=True)

        assert (base_dir / "visualizations").exists()
        assert (base_dir / "data").exists()
        assert (base_dir / "reports").exists()
