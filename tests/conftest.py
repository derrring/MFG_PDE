"""
Pytest configuration and shared fixtures for MFG_PDE test suite.

This module provides common fixtures, test configuration, and utilities
used across the entire test suite.
"""

import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

import numpy as np

# Import main package components
from mfg_pde import ExampleMFGProblem
from mfg_pde.config.pydantic_config import create_accurate_config, create_fast_config, create_research_config

# =============================================================================
# Test Configuration
# =============================================================================


def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line("markers", "unit: Unit tests (fast, isolated)")
    config.addinivalue_line("markers", "integration: Integration tests (slower, cross-component)")
    config.addinivalue_line("markers", "performance: Performance tests (may be slow)")
    config.addinivalue_line("markers", "mathematical: Mathematical property validation tests")
    config.addinivalue_line("markers", "slow: Slow tests (may take >10 seconds)")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test paths."""
    for item in items:
        # Add markers based on test file paths
        test_path = str(item.fspath)

        if "/unit/" in test_path:
            item.add_marker(pytest.mark.unit)
        elif "/integration/" in test_path:
            item.add_marker(pytest.mark.integration)
        elif "/performance/" in test_path:
            item.add_marker(pytest.mark.performance)
        elif "/mathematical/" in test_path:
            item.add_marker(pytest.mark.mathematical)

        # Mark slow tests based on name patterns
        if "large" in item.name or "slow" in item.name or "benchmark" in item.name:
            item.add_marker(pytest.mark.slow)


# =============================================================================
# Problem Fixtures
# =============================================================================


@pytest.fixture
def tiny_problem():
    """Very small problem for quick tests."""
    return ExampleMFGProblem(Nx=5, Nt=3, T=0.1)


@pytest.fixture
def small_problem():
    """Small problem for unit tests."""
    return ExampleMFGProblem(Nx=10, Nt=5, T=0.5)


@pytest.fixture
def medium_problem():
    """Medium problem for integration tests."""
    return ExampleMFGProblem(Nx=25, Nt=12, T=1.0)


@pytest.fixture
def large_problem():
    """Large problem for performance tests."""
    return ExampleMFGProblem(Nx=50, Nt=25, T=2.0)


@pytest.fixture(
    params=[
        {"Nx": 10, "Nt": 5, "T": 0.5},
        {"Nx": 15, "Nt": 8, "T": 1.0},
        {"Nx": 20, "Nt": 10, "T": 1.5},
    ]
)
def parametrized_problem(request):
    """Parametrized problem fixture for testing multiple configurations."""
    return ExampleMFGProblem(**request.param)


@pytest.fixture(params=[0.1, 0.5, 1.0, 2.0])
def diffusion_coefficient(request):
    """Parametrized diffusion coefficient for testing."""
    return request.param


# =============================================================================
# Configuration Fixtures
# =============================================================================


@pytest.fixture
def fast_config():
    """Fast configuration for testing."""
    return create_fast_config()


@pytest.fixture
def accurate_config():
    """Accurate configuration for testing."""
    return create_accurate_config()


@pytest.fixture
def research_config():
    """Research configuration for testing."""
    return create_research_config()


@pytest.fixture(params=["fast", "accurate", "research"])
def any_config(request):
    """Parametrized fixture returning different configuration types."""
    config_factories = {
        "fast": create_fast_config,
        "accurate": create_accurate_config,
        "research": create_research_config,
    }
    return config_factories[request.param]()


# =============================================================================
# Data Fixtures
# =============================================================================


@pytest.fixture
def deterministic_arrays():
    """Deterministic arrays for reproducible tests."""
    np.random.seed(42)  # Fixed seed for reproducibility
    return {
        'U_small': np.random.rand(6, 11),  # (Nt+1, Nx+1) for small problem
        'M_small': np.random.rand(6, 11),
        'U_medium': np.random.rand(13, 26),  # (Nt+1, Nx+1) for medium problem
        'M_medium': np.random.rand(13, 26),
    }


@pytest.fixture
def valid_test_matrices():
    """Valid test matrices with proper physical properties."""
    np.random.seed(123)

    # Create density matrix with mass conservation
    M = np.random.rand(11, 21)
    M = np.maximum(M, 0)  # Ensure non-negativity
    # Normalize each time slice to conserve mass
    for t in range(M.shape[0]):
        if np.sum(M[t, :]) > 0:
            M[t, :] /= np.sum(M[t, :])

    # Create value function matrix
    U = np.random.rand(11, 21) * 10 - 5  # Range [-5, 5]

    return {'U': U, 'M': M}


@pytest.fixture
def boundary_conditions():
    """Standard boundary condition configurations."""
    return {
        'dirichlet': {'type': 'dirichlet', 'value': 0.0},
        'neumann': {'type': 'neumann', 'derivative': 0.0},
        'periodic': {'type': 'periodic'},
        'mixed': {'type': 'mixed', 'left': 'dirichlet', 'right': 'neumann'},
    }


# =============================================================================
# File System Fixtures
# =============================================================================


@pytest.fixture
def temp_directory():
    """Temporary directory for file operations."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def test_output_dir(temp_directory):
    """Test output directory with subdirectories."""
    output_dir = temp_directory / "test_output"
    output_dir.mkdir()

    # Create subdirectories
    (output_dir / "results").mkdir()
    (output_dir / "plots").mkdir()
    (output_dir / "reports").mkdir()

    return output_dir


# =============================================================================
# Solver Fixtures
# =============================================================================


@pytest.fixture(params=["fixed_point", "particle_collocation"])
def solver_type(request):
    """Parametrized solver type for testing."""
    return request.param


@pytest.fixture
def solver_factory():
    """Factory function for creating solvers."""

    def _create_solver(problem, solver_type="fixed_point", config=None):
        from mfg_pde import create_fast_solver

        if config is None:
            return create_fast_solver(problem, solver_type)
        else:
            from mfg_pde.factory import create_solver

            return create_solver(problem, solver_type, config=config)

    return _create_solver


# =============================================================================
# Mathematical Properties Fixtures
# =============================================================================


@pytest.fixture
def tolerance_levels():
    """Different tolerance levels for testing convergence."""
    return {'strict': 1e-8, 'normal': 1e-6, 'relaxed': 1e-4, 'loose': 1e-2}


@pytest.fixture
def convergence_criteria():
    """Different convergence criteria configurations."""
    return {
        'standard': {'relative_tolerance': True, 'absolute_tolerance': False},
        'absolute': {'relative_tolerance': False, 'absolute_tolerance': True},
        'combined': {'relative_tolerance': True, 'absolute_tolerance': True},
    }


# =============================================================================
# Utility Fixtures
# =============================================================================


@pytest.fixture
def performance_timer():
    """Timer utility for performance testing."""
    import time

    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None

        def start(self):
            self.start_time = time.time()

        def stop(self):
            self.end_time = time.time()
            return self.elapsed

        @property
        def elapsed(self):
            if self.start_time is None:
                return 0
            end = self.end_time or time.time()
            return end - self.start_time

    return Timer()


@pytest.fixture
def memory_tracker():
    """Memory usage tracker for performance testing."""
    import psutil

    class MemoryTracker:
        def __init__(self):
            self.process = psutil.Process()
            self.start_memory = None
            self.peak_memory = 0

        def start(self):
            self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            self.peak_memory = self.start_memory

        def update(self):
            current_memory = self.process.memory_info().rss / 1024 / 1024
            self.peak_memory = max(self.peak_memory, current_memory)
            return current_memory

        @property
        def current_mb(self):
            return self.process.memory_info().rss / 1024 / 1024

        @property
        def increase_mb(self):
            if self.start_memory is None:
                return 0
            return self.current_mb - self.start_memory

    return MemoryTracker()


# =============================================================================
# Mock and Stub Fixtures
# =============================================================================


@pytest.fixture
def mock_convergence_result():
    """Mock convergence result for testing."""
    return {
        'converged': True,
        'iterations': 15,
        'final_error': 1e-7,
        'error_history': [1e-1, 3e-2, 8e-3, 2e-3, 5e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7],
        'convergence_rate': 0.3,
        'execution_time': 2.5,
    }


@pytest.fixture
def mock_failed_convergence():
    """Mock failed convergence result for testing error handling."""
    return {
        'converged': False,
        'iterations': 50,
        'final_error': 1e-3,
        'error_history': [1e-1, 5e-2, 2e-2, 1e-2, 5e-3, 2e-3, 1e-3],
        'convergence_rate': None,
        'execution_time': 10.0,
        'failure_reason': 'Maximum iterations reached',
    }


# =============================================================================
# Parameterized Test Data
# =============================================================================


@pytest.fixture(
    params=[
        (10, 5, 0.5),  # Small problem
        (20, 10, 1.0),  # Medium problem
        (30, 15, 1.5),  # Large problem
    ]
)
def problem_dimensions(request):
    """Parametrized problem dimensions (Nx, Nt, T)."""
    return request.param


@pytest.fixture(
    params=[
        {'max_iterations': 10, 'tolerance': 1e-4},
        {'max_iterations': 20, 'tolerance': 1e-6},
        {'max_iterations': 50, 'tolerance': 1e-8},
    ]
)
def newton_parameters(request):
    """Parametrized Newton solver parameters."""
    return request.param


# =============================================================================
# Session-Scoped Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def reference_solutions():
    """Reference solutions for validation (computed once per session)."""
    # This would load or compute reference solutions
    # For now, return empty dict - implement as needed
    return {}


@pytest.fixture(scope="session")
def performance_baselines():
    """Performance baselines for regression testing."""
    return {
        'small_problem_time': 1.0,  # seconds
        'medium_problem_time': 5.0,  # seconds
        'memory_per_dof': 1e-6,  # MB per degree of freedom
    }


# =============================================================================
# Cleanup Utilities
# =============================================================================


@pytest.fixture(autouse=True)
def cleanup_numpy_state():
    """Automatically cleanup numpy random state after each test."""
    yield
    # Reset numpy random state to avoid test interference
    np.random.seed(None)


@pytest.fixture(autouse=True)
def suppress_warnings():
    """Suppress known warnings during testing."""
    import warnings

    # Suppress specific warnings that are expected during testing
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="mfg_pde.*")
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)

    yield

    # Reset warning filters
    warnings.resetwarnings()


# =============================================================================
# Test Data Validation
# =============================================================================


def validate_mfg_solution(U, M, problem):
    """Validate that U and M arrays represent a valid MFG solution."""
    # Check dimensions
    expected_shape = (problem.Nt + 1, problem.Nx + 1)
    assert U.shape == expected_shape, f"U shape {U.shape} != expected {expected_shape}"
    assert M.shape == expected_shape, f"M shape {M.shape} != expected {expected_shape}"

    # Check for NaN/Inf
    assert not np.any(np.isnan(U)), "U contains NaN values"
    assert not np.any(np.isnan(M)), "M contains NaN values"
    assert not np.any(np.isinf(U)), "U contains Inf values"
    assert not np.any(np.isinf(M)), "M contains Inf values"

    # Check physical properties
    assert np.all(M >= -1e-10), f"M contains negative values: min={np.min(M)}"

    # Check mass conservation (approximately)
    initial_mass = np.sum(problem.m_init) * problem.Dx
    for t in range(problem.Nt + 1):
        current_mass = np.sum(M[t, :]) * problem.Dx
        mass_error = abs(current_mass - initial_mass)
        assert mass_error < 0.1, f"Mass conservation violated at t={t}: error={mass_error}"


# Export validation function for use in tests
__all__ = ['validate_mfg_solution']
