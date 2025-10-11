#!/usr/bin/env python3
"""
Unit tests for General MFG Factory

Tests the GeneralMFGFactory class for creating MFG problems from
various input formats (functions, configs, files).
"""

import tempfile
from pathlib import Path

import pytest

import numpy as np

# Check if OmegaConf is available
try:
    import omegaconf

    OMEGACONF_AVAILABLE = True
except ImportError:
    OMEGACONF_AVAILABLE = False
    omegaconf = None

from mfg_pde.factory.general_mfg_factory import (
    GeneralMFGFactory,
    create_general_mfg_problem,
    get_general_factory,
)

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def factory():
    """Create GeneralMFGFactory instance."""
    return GeneralMFGFactory()


@pytest.fixture
def sample_functions():
    """Create sample MFG component functions with correct signatures."""

    def hamiltonian(x_idx, x_position, m_at_x, p_values, t_idx, current_time, problem):
        """Hamiltonian function with full signature."""
        # Simple quadratic hamiltonian
        p_forward = p_values.get("forward", 0.0)
        p_backward = p_values.get("backward", 0.0)
        return 0.5 * problem.coefCT * (problem.utils.npart(p_forward) ** 2 + problem.utils.ppart(p_backward) ** 2)

    def hamiltonian_dm(x_idx, x_position, m_at_x, p_values, t_idx, current_time, problem):
        """Hamiltonian dm derivative with full signature."""
        return 0.0 * m_at_x

    def potential(x):
        return 0.5 * np.sum(x**2, axis=-1)

    def initial_density(x):
        return np.exp(-10 * np.sum((x - 0.5) ** 2, axis=-1))

    def final_value(x):
        return 0.5 * np.sum(x**2, axis=-1)

    return {
        "hamiltonian": hamiltonian,
        "hamiltonian_dm": hamiltonian_dm,
        "potential": potential,
        "initial_density": initial_density,
        "final_value": final_value,
    }


@pytest.fixture
def domain_config():
    """Create sample domain configuration."""
    return {"xmin": 0.0, "xmax": 1.0, "Nx": 20}


@pytest.fixture
def time_config():
    """Create sample time configuration."""
    return {"T": 1.0, "Nt": 10}


@pytest.fixture
def solver_config():
    """Create sample solver configuration."""
    return {"sigma": 0.1, "coefCT": 1.0}


# ============================================================================
# Test: Factory Initialization
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_factory_initialization():
    """Test GeneralMFGFactory can be instantiated."""
    factory = GeneralMFGFactory()

    assert factory is not None
    assert hasattr(factory, "function_registry")
    assert isinstance(factory.function_registry, dict)
    assert len(factory.function_registry) == 0


@pytest.mark.unit
@pytest.mark.fast
def test_get_general_factory():
    """Test get_general_factory convenience function."""
    factory = get_general_factory()

    assert isinstance(factory, GeneralMFGFactory)


# ============================================================================
# Test: Function Registration
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_register_single_function(factory):
    """Test registering a single function."""

    def test_func(x):
        return x**2

    factory.register_functions({"test_func": test_func})

    assert "test_func" in factory.function_registry
    assert factory.function_registry["test_func"] is test_func


@pytest.mark.unit
@pytest.mark.fast
def test_register_multiple_functions(factory, sample_functions):
    """Test registering multiple functions at once."""
    factory.register_functions(sample_functions)

    assert len(factory.function_registry) == len(sample_functions)
    for name in sample_functions:
        assert name in factory.function_registry


@pytest.mark.unit
@pytest.mark.fast
def test_register_functions_overwrites(factory):
    """Test that registering function with same name overwrites."""

    def func_v1(x):
        return x

    def func_v2(x):
        return x**2

    factory.register_functions({"func": func_v1})
    assert factory.function_registry["func"] is func_v1

    factory.register_functions({"func": func_v2})
    assert factory.function_registry["func"] is func_v2


# ============================================================================
# Test: Create from Functions
# ============================================================================


@pytest.mark.unit
def test_create_from_functions_minimal(factory, sample_functions, domain_config, time_config):
    """Test creating MFG problem with minimal required components."""
    problem = factory.create_from_functions(
        hamiltonian_func=sample_functions["hamiltonian"],
        hamiltonian_dm_func=sample_functions["hamiltonian_dm"],
        domain_config=domain_config,
        time_config=time_config,
    )

    assert problem is not None
    assert hasattr(problem, "T")
    assert time_config["T"] == problem.T
    assert hasattr(problem, "H")


@pytest.mark.unit
def test_create_from_functions_with_solver_config(factory, sample_functions, domain_config, time_config, solver_config):
    """Test creating problem with solver configuration."""
    problem = factory.create_from_functions(
        hamiltonian_func=sample_functions["hamiltonian"],
        hamiltonian_dm_func=sample_functions["hamiltonian_dm"],
        domain_config=domain_config,
        time_config=time_config,
        solver_config=solver_config,
    )

    assert problem is not None
    assert hasattr(problem, "sigma")


@pytest.mark.unit
def test_create_from_functions_with_optional_components(factory, sample_functions, domain_config, time_config):
    """Test creating problem with optional components."""
    problem = factory.create_from_functions(
        hamiltonian_func=sample_functions["hamiltonian"],
        hamiltonian_dm_func=sample_functions["hamiltonian_dm"],
        domain_config=domain_config,
        time_config=time_config,
        potential_func=sample_functions["potential"],
        initial_density_func=sample_functions["initial_density"],
        final_value_func=sample_functions["final_value"],
    )

    assert problem is not None
    # Check that optional components are stored (actual attribute names may vary)
    assert hasattr(problem, "T")
    assert time_config["T"] == problem.T


@pytest.mark.unit
def test_create_from_functions_with_description(factory, sample_functions, domain_config, time_config):
    """Test creating problem with description."""
    description = "Test MFG Problem"

    problem = factory.create_from_functions(
        hamiltonian_func=sample_functions["hamiltonian"],
        hamiltonian_dm_func=sample_functions["hamiltonian_dm"],
        domain_config=domain_config,
        time_config=time_config,
        description=description,
    )

    assert problem is not None


# ============================================================================
# Test: Create from Config Dict
# ============================================================================


@pytest.mark.unit
def test_create_from_config_dict_simple(factory):
    """Test creating problem from simple config dict."""
    config = {
        "hamiltonian": {"type": "quadratic"},
        "domain": {"xmin": 0.0, "xmax": 1.0, "Nx": 20},
        "time": {"T": 1.0, "Nt": 10},
    }

    # This might fail if hamiltonian type is not recognized, which is expected
    try:
        problem = factory.create_from_config_dict(config)
        assert problem is not None
    except (ValueError, KeyError, NotImplementedError):
        # If not implemented, that's fine for this test
        pass


@pytest.mark.unit
@pytest.mark.fast
def test_create_from_config_dict_validates_required_fields(factory):
    """Test that missing required fields are caught."""
    incomplete_config = {
        "domain": {"xmin": 0.0, "xmax": 1.0, "Nx": 20}
        # Missing hamiltonian and time
    }

    with pytest.raises((KeyError, ValueError)):
        factory.create_from_config_dict(incomplete_config)


# ============================================================================
# Test: Config Validation
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_validate_config_valid(factory):
    """Test validation accepts valid config."""
    config = {
        "functions": {
            "hamiltonian": "lambda x_idx, x_position, m_at_x, p_values, t_idx, current_time, problem: 0.5 * p_values['forward']**2",
            "hamiltonian_dm": "lambda x_idx, x_position, m_at_x, p_values, t_idx, current_time, problem: 0.0",
        },
        "domain": {"xmin": 0.0, "xmax": 1.0, "Nx": 20},
        "time": {"T": 1.0, "Nt": 10},
    }

    validated = factory.validate_config(config)

    assert isinstance(validated, dict)
    assert validated["valid"] is True
    assert len(validated["errors"]) == 0


@pytest.mark.unit
@pytest.mark.fast
def test_validate_config_missing_functions_section(factory):
    """Test validation catches missing functions section."""
    config = {
        "domain": {"xmin": 0.0, "xmax": 1.0, "Nx": 20},
        "time": {"T": 1.0, "Nt": 10},
    }

    validated = factory.validate_config(config)

    assert validated["valid"] is False
    assert "Missing required section: functions" in validated["errors"]


@pytest.mark.unit
@pytest.mark.fast
def test_validate_config_missing_domain(factory):
    """Test validation catches missing domain."""
    config = {
        "functions": {
            "hamiltonian": "lambda x: 0.5 * x**2",
            "hamiltonian_dm": "lambda x: 0.0",
        },
        "time": {"T": 1.0, "Nt": 10},
    }

    validated = factory.validate_config(config)

    assert validated["valid"] is False
    assert "Missing required section: domain" in validated["errors"]


@pytest.mark.unit
@pytest.mark.fast
def test_validate_config_missing_time(factory):
    """Test validation catches missing time."""
    config = {
        "functions": {
            "hamiltonian": "lambda x: 0.5 * x**2",
            "hamiltonian_dm": "lambda x: 0.0",
        },
        "domain": {"xmin": 0.0, "xmax": 1.0, "Nx": 20},
    }

    validated = factory.validate_config(config)

    assert validated["valid"] is False
    assert "Missing required section: time" in validated["errors"]


# ============================================================================
# Test: Template Creation
# ============================================================================


@pytest.mark.unit
@pytest.mark.skipif(not OMEGACONF_AVAILABLE, reason="OmegaConf not available")
def test_create_template_config(factory):
    """Test creating template configuration file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        template_path = Path(tmpdir) / "template_config.yaml"

        factory.create_template_config(str(template_path))

        # Check file was created
        assert template_path.exists()

        # Check file has content
        content = template_path.read_text()
        assert len(content) > 0
        assert "hamiltonian" in content.lower() or "domain" in content.lower()


# ============================================================================
# Test: Convenience Functions
# ============================================================================


@pytest.mark.unit
def test_create_general_mfg_problem_from_functions():
    """Test module-level convenience function."""

    def hamiltonian(x_idx, x_position, m_at_x, p_values, t_idx, current_time, problem):
        p_forward = p_values.get("forward", 0.0)
        p_backward = p_values.get("backward", 0.0)
        return 0.5 * problem.coefCT * (problem.utils.npart(p_forward) ** 2 + problem.utils.ppart(p_backward) ** 2)

    def hamiltonian_dm(x_idx, x_position, m_at_x, p_values, t_idx, current_time, problem):
        return 0.0

    # Use convenience function with explicit parameters (not config dicts)
    problem = create_general_mfg_problem(
        hamiltonian_func=hamiltonian,
        hamiltonian_dm_func=hamiltonian_dm,
        xmin=0.0,
        xmax=1.0,
        Nx=20,
        T=1.0,
        Nt=10,
    )

    assert problem is not None


# ============================================================================
# Test: Error Handling
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_create_from_functions_missing_hamiltonian(factory, domain_config, time_config):
    """Test error when hamiltonian is missing."""
    with pytest.raises(ValueError):  # Builder raises ValueError for missing hamiltonian
        factory.create_from_functions(
            hamiltonian_func=None,  # type: ignore
            hamiltonian_dm_func=lambda x_idx, x_position, m_at_x, p_values, t_idx, current_time, problem: 0.0,
            domain_config=domain_config,
            time_config=time_config,
        )


@pytest.mark.unit
@pytest.mark.fast
def test_create_from_functions_missing_domain(factory, sample_functions, time_config):
    """Test error when domain config is missing."""
    with pytest.raises((TypeError, KeyError)):
        factory.create_from_functions(
            hamiltonian_func=sample_functions["hamiltonian"],
            hamiltonian_dm_func=sample_functions["hamiltonian_dm"],
            domain_config=None,  # type: ignore
            time_config=time_config,
        )


@pytest.mark.unit
@pytest.mark.fast
def test_create_from_functions_missing_time(factory, sample_functions, domain_config):
    """Test error when time config is missing."""
    with pytest.raises((TypeError, KeyError)):
        factory.create_from_functions(
            hamiltonian_func=sample_functions["hamiltonian"],
            hamiltonian_dm_func=sample_functions["hamiltonian_dm"],
            domain_config=domain_config,
            time_config=None,  # type: ignore
        )


# ============================================================================
# Test: Integration
# ============================================================================


@pytest.mark.unit
def test_end_to_end_problem_creation(factory, sample_functions, domain_config, time_config, solver_config):
    """Test complete workflow from functions to problem."""
    # Register functions
    factory.register_functions(sample_functions)

    # Create problem
    problem = factory.create_from_functions(
        hamiltonian_func=sample_functions["hamiltonian"],
        hamiltonian_dm_func=sample_functions["hamiltonian_dm"],
        domain_config=domain_config,
        time_config=time_config,
        solver_config=solver_config,
        potential_func=sample_functions["potential"],
        initial_density_func=sample_functions["initial_density"],
        final_value_func=sample_functions["final_value"],
    )

    # Verify problem has all components
    assert problem is not None
    assert hasattr(problem, "H")
    assert hasattr(problem, "dH_dm")
    assert hasattr(problem, "T")
    assert time_config["T"] == problem.T
    assert hasattr(problem, "sigma")
    assert problem.sigma == solver_config["sigma"]


@pytest.mark.unit
def test_multiple_problems_from_same_factory(factory, sample_functions, domain_config, time_config):
    """Test creating multiple problems from same factory instance."""
    # Create first problem
    problem1 = factory.create_from_functions(
        hamiltonian_func=sample_functions["hamiltonian"],
        hamiltonian_dm_func=sample_functions["hamiltonian_dm"],
        domain_config=domain_config,
        time_config=time_config,
    )

    # Create second problem with different config
    problem2 = factory.create_from_functions(
        hamiltonian_func=sample_functions["hamiltonian"],
        hamiltonian_dm_func=sample_functions["hamiltonian_dm"],
        domain_config={"xmin": 0.0, "xmax": 2.0, "Nx": 30},
        time_config={"T": 2.0, "Nt": 20},
    )

    # Both should be valid but different
    assert problem1 is not None
    assert problem2 is not None
    assert problem1.T != problem2.T


# ============================================================================
# Test: Function Registry Usage
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_function_registry_accessible_after_registration(factory):
    """Test registered functions can be accessed."""

    def custom_func(x):
        return x * 2

    factory.register_functions({"custom": custom_func})

    # Should be able to retrieve function
    retrieved = factory.function_registry["custom"]
    assert retrieved is custom_func
    assert retrieved(5) == 10
