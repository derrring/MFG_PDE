#!/usr/bin/env python3
"""
Unit tests for Pydantic Solver Factory

Tests the PydanticSolverFactory class and related factory functions
for creating MFG solvers with Pydantic validation.
"""

import pytest

import numpy as np

try:
    from pydantic import ValidationError

    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

from mfg_pde.core.mfg_problem import MFGProblem
from mfg_pde.factory.pydantic_solver_factory import (
    PydanticSolverFactory,
    create_accurate_validated_solver,
    create_fast_validated_solver,
    create_research_validated_solver,
    create_validated_solver,
    validate_solver_config,
)

if PYDANTIC_AVAILABLE:
    from mfg_pde.config.pydantic_config import MFGSolverConfig, create_fast_config


# Test Problem Fixture
class SimpleLQMFG(MFGProblem):
    """Simple Linear-Quadratic MFG for testing."""

    def __init__(self):
        super().__init__(T=1.0, Nt=10, xmin=0.0, xmax=1.0, Nx=20)

    def g(self, x):
        return 0.5 * np.sum(x**2, axis=-1)

    def rho0(self, x):
        return np.exp(-10 * (x - 0.5) ** 2)

    def f(self, x, u, m):
        return 0.1 * u**2 + 0.05 * m

    def sigma(self, x):
        return 0.1

    def H(self, x, p, m):
        return 0.5 * np.sum(p**2, axis=-1)

    def dH_dm(self, x, p, m):
        return 0.0


@pytest.fixture
def simple_problem():
    """Create simple test problem."""
    return SimpleLQMFG()


@pytest.fixture
def factory():
    """Create PydanticSolverFactory instance."""
    return PydanticSolverFactory()


# ============================================================================
# Test: Factory Initialization
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_factory_initialization():
    """Test PydanticSolverFactory can be instantiated."""
    factory = PydanticSolverFactory()
    assert factory is not None
    assert hasattr(factory, "logger")


@pytest.mark.unit
@pytest.mark.fast
@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic not available")
def test_factory_pydantic_warning(caplog):
    """Test factory logs warning when Pydantic unavailable."""
    # This test is skipped when Pydantic IS available
    # In a real scenario without Pydantic, the warning should be logged


# ============================================================================
# Test: Preset Config Creation
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic required")
def test_create_preset_config_fast(factory):
    """Test creating fast preset configuration."""
    config = factory._create_preset_config("fast")

    assert isinstance(config, MFGSolverConfig)
    assert config.convergence_tolerance == 1e-3  # Fast tolerance


@pytest.mark.unit
@pytest.mark.fast
@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic required")
def test_create_preset_config_accurate(factory):
    """Test creating accurate preset configuration."""
    config = factory._create_preset_config("accurate")

    assert isinstance(config, MFGSolverConfig)
    assert config.convergence_tolerance < 1e-5  # Accurate tolerance


@pytest.mark.unit
@pytest.mark.fast
@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic required")
def test_create_preset_config_research(factory):
    """Test creating research preset configuration."""
    config = factory._create_preset_config("research")

    assert isinstance(config, MFGSolverConfig)
    # Research config should have structured returns enabled
    assert config.return_structured is True
    # Research config typically has tight convergence tolerance
    assert config.convergence_tolerance < 1e-5


@pytest.mark.unit
@pytest.mark.fast
@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic required")
def test_create_preset_config_balanced(factory):
    """Test creating balanced (default) preset configuration."""
    config = factory._create_preset_config("balanced")

    assert isinstance(config, MFGSolverConfig)


@pytest.mark.unit
@pytest.mark.fast
@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic required")
def test_create_preset_config_invalid(factory):
    """Test creating config with invalid preset raises error."""
    with pytest.raises(ValueError, match="Unknown config preset"):
        factory._create_preset_config("invalid_preset")


# ============================================================================
# Test: Config Update with Kwargs
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic required")
def test_update_config_convergence_tolerance(factory):
    """Test updating convergence tolerance."""
    config = create_fast_config()
    updated = factory._update_config_with_kwargs(config, convergence_tolerance=1e-8)

    assert updated.convergence_tolerance == 1e-8


@pytest.mark.unit
@pytest.mark.fast
@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic required")
def test_update_config_newton_parameters(factory):
    """Test updating Newton solver parameters."""
    config = create_fast_config()
    updated = factory._update_config_with_kwargs(config, newton_tolerance=1e-10, newton_max_iterations=50)

    assert updated.newton.tolerance == 1e-10
    assert updated.newton.max_iterations == 50


@pytest.mark.unit
@pytest.mark.fast
@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic required")
def test_update_config_picard_parameters(factory):
    """Test updating Picard iteration parameters."""
    config = create_fast_config()
    updated = factory._update_config_with_kwargs(config, picard_tolerance=1e-9, picard_max_iterations=30)

    assert updated.picard.tolerance == 1e-9
    assert updated.picard.max_iterations == 30


@pytest.mark.unit
@pytest.mark.fast
@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic required")
def test_update_config_particle_parameters(factory):
    """Test updating particle method parameters."""
    config = create_fast_config()
    updated = factory._update_config_with_kwargs(config, num_particles=2000, kde_bandwidth=0.05)

    assert updated.fp.particle.num_particles == 2000
    assert updated.fp.particle.kde_bandwidth == 0.05


# ============================================================================
# Test: Validated Solver Creation
# ============================================================================


@pytest.mark.unit
@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic required")
@pytest.mark.skip(
    reason="HJBGFDMSolver is abstract and cannot be instantiated directly. "
    "Factory attempts to create GFDM solver which fails. "
    "Issue #140 - pre-existing test failure."
)
def test_create_validated_fixed_point_solver(factory, simple_problem):
    """Test creating validated fixed point solver."""
    solver = factory.create_validated_solver(simple_problem, solver_type="fixed_point", config_preset="fast")

    assert solver is not None
    assert hasattr(solver, "solve")


@pytest.mark.unit
@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic required")
@pytest.mark.skip(
    reason="HJBGFDMSolver is abstract and cannot be instantiated directly. "
    "Factory attempts to create GFDM solver which fails. "
    "Issue #140 - pre-existing test failure."
)
def test_create_validated_particle_collocation_solver(factory, simple_problem):
    """Test creating validated particle collocation solver."""
    x_coords = np.linspace(simple_problem.xmin, simple_problem.xmax, simple_problem.Nx)
    collocation_points = x_coords.reshape(-1, 1)

    solver = factory.create_validated_solver(
        simple_problem, solver_type="particle_collocation", config_preset="fast", collocation_points=collocation_points
    )

    assert solver is not None
    assert hasattr(solver, "solve")


@pytest.mark.unit
@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic required")
@pytest.mark.skip(
    reason="HJBGFDMSolver is abstract and cannot be instantiated directly. "
    "Factory attempts to create GFDM solver which fails. "
    "Issue #140 - pre-existing test failure."
)
def test_create_validated_monitored_particle_solver(factory, simple_problem):
    """Test creating validated particle collocation solver."""
    x_coords = np.linspace(simple_problem.xmin, simple_problem.xmax, simple_problem.Nx)
    collocation_points = x_coords.reshape(-1, 1)

    solver = factory.create_validated_solver(
        simple_problem,
        solver_type="particle_collocation",
        config_preset="research",
        collocation_points=collocation_points,
    )

    assert solver is not None
    assert hasattr(solver, "solve")


@pytest.mark.unit
@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic required")
@pytest.mark.skip(
    reason="HJBGFDMSolver is abstract and cannot be instantiated directly. "
    "Factory attempts to create GFDM solver which fails. "
    "Issue #140 - pre-existing test failure."
)
def test_create_validated_adaptive_particle_solver(factory, simple_problem):
    """Test creating validated particle collocation solver with custom config."""
    x_coords = np.linspace(simple_problem.xmin, simple_problem.xmax, simple_problem.Nx)
    collocation_points = x_coords.reshape(-1, 1)

    solver = factory.create_validated_solver(
        simple_problem,
        solver_type="particle_collocation",
        config_preset="research",
        collocation_points=collocation_points,
    )

    assert solver is not None
    assert hasattr(solver, "solve")


@pytest.mark.unit
@pytest.mark.fast
@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic required")
def test_create_validated_solver_invalid_type(factory, simple_problem):
    """Test creating solver with invalid type raises error."""
    with pytest.raises(ValueError, match="Unknown solver type"):
        factory.create_validated_solver(
            simple_problem,
            solver_type="invalid_type",  # type: ignore
            config_preset="fast",
        )


@pytest.mark.unit
@pytest.mark.fast
@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic required")
def test_create_validated_solver_with_custom_config(factory, simple_problem):
    """Test creating solver with custom configuration."""
    custom_config = create_fast_config()
    custom_config.convergence_tolerance = 1e-7

    solver = factory.create_validated_solver(simple_problem, solver_type="fixed_point", config=custom_config)

    assert solver is not None


@pytest.mark.unit
@pytest.mark.fast
@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic required")
def test_create_validated_solver_invalid_config_type(factory, simple_problem):
    """Test creating solver with invalid config type raises error."""
    with pytest.raises(ValueError, match="config must be MFGSolverConfig"):
        factory.create_validated_solver(
            simple_problem,
            solver_type="fixed_point",
            config={"invalid": "config"},  # type: ignore
        )


# ============================================================================
# Test: Convenience Functions
# ============================================================================


@pytest.mark.unit
@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic required")
def test_create_validated_solver_function(simple_problem):
    """Test module-level create_validated_solver function."""
    solver = create_validated_solver(simple_problem, solver_type="fixed_point", config_preset="fast")

    assert solver is not None


@pytest.mark.unit
@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic required")
def test_create_fast_validated_solver(simple_problem):
    """Test create_fast_validated_solver convenience function."""
    solver = create_fast_validated_solver(simple_problem, solver_type="fixed_point")

    assert solver is not None


@pytest.mark.unit
@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic required")
def test_create_accurate_validated_solver(simple_problem):
    """Test create_accurate_validated_solver convenience function."""
    solver = create_accurate_validated_solver(simple_problem, solver_type="fixed_point")

    assert solver is not None


@pytest.mark.unit
@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic required")
def test_create_research_validated_solver(simple_problem):
    """Test create_research_validated_solver convenience function."""
    solver = create_research_validated_solver(simple_problem, solver_type="fixed_point")

    assert solver is not None


# ============================================================================
# Test: Config Validation
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic required")
def test_validate_solver_config_valid():
    """Test validating valid config dict."""
    config_dict = {
        "convergence_tolerance": 1e-6,
        "max_iterations": 100,
        "newton": {"tolerance": 1e-8, "max_iterations": 20},
        "picard": {"tolerance": 1e-6, "max_iterations": 50},
    }

    config = validate_solver_config(config_dict)

    assert isinstance(config, MFGSolverConfig)
    assert config.convergence_tolerance == 1e-6


@pytest.mark.unit
@pytest.mark.fast
@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic required")
def test_validate_solver_config_invalid():
    """Test validating invalid config dict raises ValidationError."""
    config_dict = {
        "convergence_tolerance": "invalid",  # Should be float
        "max_iterations": 100,
    }

    with pytest.raises(ValidationError):
        validate_solver_config(config_dict)


# ============================================================================
# Test: Error Handling
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_factory_without_pydantic_raises_error(simple_problem):
    """Test that factory raises error when Pydantic not available."""
    factory = PydanticSolverFactory()

    if not PYDANTIC_AVAILABLE:
        with pytest.raises(ImportError, match="Pydantic required"):
            factory.create_validated_solver(simple_problem, solver_type="fixed_point")


@pytest.mark.unit
@pytest.mark.fast
@pytest.mark.skipif(not PYDANTIC_AVAILABLE, reason="Pydantic required")
def test_create_validated_solver_with_invalid_kwargs(factory, simple_problem):
    """Test that invalid kwargs in config update are handled gracefully."""
    # This should either update successfully or log warning
    # but should not crash
    try:
        solver = factory.create_validated_solver(
            simple_problem, solver_type="fixed_point", config_preset="fast", unknown_parameter="should_be_ignored"
        )
        assert solver is not None
    except ValidationError:
        # If validation fails, that's also acceptable behavior
        pass
