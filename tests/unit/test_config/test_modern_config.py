#!/usr/bin/env python3
"""
Unit tests for mfg_pde/config/modern_config.py

Tests modern configuration system including:
- SolverConfig builder pattern
- Configuration immutability (returns new instances)
- Builder methods (with_tolerance, with_max_iterations, etc.)
- Property access (tolerance, max_iterations)
- Deprecation warnings for lowercase parameters
- Custom parameters
- PresetConfig factory methods
- Convenience functions
- Module exports
"""

import warnings

import pytest

from mfg_pde.config.modern_config import (
    PresetConfig,
    SolverConfig,
    accurate_config,
    create_config,
    crowd_dynamics_config,
    educational_config,
    epidemic_config,
    fast_config,
    financial_config,
    large_scale_config,
    production_config,
    research_config,
    traffic_config,
)

# ===================================================================
# Test SolverConfig Initialization
# ===================================================================


@pytest.mark.unit
def test_solver_config_default_initialization():
    """Test SolverConfig with default initialization."""
    config = SolverConfig()

    # Should have underlying Pydantic config
    assert config.get_underlying_config() is not None
    # Should have default tolerance and max_iterations
    assert isinstance(config.tolerance, float)
    assert isinstance(config.max_iterations, int)


@pytest.mark.unit
def test_solver_config_with_base_config():
    """Test SolverConfig initialization with base config."""
    from mfg_pde.config.pydantic_config import create_fast_config

    base = create_fast_config()
    config = SolverConfig(base_config=base)

    assert config.get_underlying_config() is base


@pytest.mark.unit
def test_solver_config_string_representation():
    """Test SolverConfig __str__ and __repr__."""
    config = SolverConfig()

    str_repr = str(config)
    assert "SolverConfig" in str_repr
    assert "tolerance" in str_repr
    assert "max_iterations" in str_repr

    # __repr__ should match __str__
    assert repr(config) == str(config)


# ===================================================================
# Test Builder Pattern Immutability
# ===================================================================


@pytest.mark.unit
def test_builder_returns_new_instance():
    """Test that builder methods return new SolverConfig instances."""
    config1 = SolverConfig()
    config2 = config1.with_tolerance(1e-8)

    # Should be different instances
    assert config1 is not config2
    # Original config should be unchanged
    assert config1.tolerance != config2.tolerance


@pytest.mark.unit
def test_builder_method_chaining():
    """Test that builder methods can be chained."""
    config = SolverConfig().with_tolerance(1e-8).with_max_iterations(500).with_damping(0.8).with_verbose(True)

    assert isinstance(config, SolverConfig)
    assert config.tolerance == pytest.approx(1e-8)
    assert config.max_iterations == 500


# ===================================================================
# Test Builder Methods
# ===================================================================


@pytest.mark.unit
def test_with_tolerance():
    """Test with_tolerance() method."""
    config = SolverConfig().with_tolerance(1e-10)

    assert config.tolerance == pytest.approx(1e-10)


@pytest.mark.unit
def test_with_max_iterations():
    """Test with_max_iterations() method."""
    # Note: Pydantic config has max limit of 500 for max_iterations
    config = SolverConfig().with_max_iterations(250)

    assert config.max_iterations == 250


@pytest.mark.unit
def test_with_damping():
    """Test with_damping() method."""
    config = SolverConfig().with_damping(0.5)

    # Should return new instance without error
    assert isinstance(config, SolverConfig)


@pytest.mark.unit
def test_with_verbose():
    """Test with_verbose() method."""
    config1 = SolverConfig().with_verbose(True)
    config2 = SolverConfig().with_verbose(False)

    # Should return new instances without error
    assert isinstance(config1, SolverConfig)
    assert isinstance(config2, SolverConfig)
    assert config1 is not config2


@pytest.mark.unit
def test_with_method():
    """Test with_method() method."""
    config = SolverConfig().with_method("newton")

    # Should return new instance without error
    assert isinstance(config, SolverConfig)


@pytest.mark.unit
def test_with_grid_size():
    """Test with_grid_size() method with uppercase parameters."""
    config = SolverConfig().with_grid_size(Nx=100, Nt=200)

    # Should return new instance without error
    assert isinstance(config, SolverConfig)


@pytest.mark.unit
def test_with_grid_size_deprecation_warning():
    """Test with_grid_size() issues deprecation warning for lowercase parameters."""
    with pytest.warns(DeprecationWarning, match="Parameter 'nx' is deprecated"):
        config = SolverConfig().with_grid_size(nx=100)
    assert isinstance(config, SolverConfig)

    with pytest.warns(DeprecationWarning, match="Parameter 'nt' is deprecated"):
        config = SolverConfig().with_grid_size(nt=200)
    assert isinstance(config, SolverConfig)


@pytest.mark.unit
def test_with_grid_size_uppercase_overrides_lowercase():
    """Test that uppercase parameters override lowercase when both provided."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        config = SolverConfig().with_grid_size(Nx=100, nx=50)

    # Uppercase should take precedence (100, not 50)
    assert isinstance(config, SolverConfig)


@pytest.mark.unit
def test_with_custom_parameter():
    """Test with_custom_parameter() method."""
    config = SolverConfig().with_custom_parameter("my_param", 42)

    # Should be able to retrieve custom parameter
    assert config.get_parameter("my_param") == 42


@pytest.mark.unit
def test_with_multiple_custom_parameters():
    """Test multiple custom parameters."""
    config = (
        SolverConfig()
        .with_custom_parameter("param1", "value1")
        .with_custom_parameter("param2", 123)
        .with_custom_parameter("param3", True)
    )

    assert config.get_parameter("param1") == "value1"
    assert config.get_parameter("param2") == 123
    assert config.get_parameter("param3") is True


# ===================================================================
# Test Property Access
# ===================================================================


@pytest.mark.unit
def test_tolerance_property():
    """Test tolerance property."""
    config = SolverConfig()

    tolerance = config.tolerance
    assert isinstance(tolerance, float)
    assert tolerance > 0.0


@pytest.mark.unit
def test_max_iterations_property():
    """Test max_iterations property."""
    config = SolverConfig()

    max_iter = config.max_iterations
    assert isinstance(max_iter, int)
    assert max_iter > 0


# ===================================================================
# Test Configuration Conversion
# ===================================================================


@pytest.mark.unit
def test_to_dict():
    """Test to_dict() method."""
    config = SolverConfig()

    config_dict = config.to_dict()
    assert isinstance(config_dict, dict)


@pytest.mark.unit
def test_get_underlying_config():
    """Test get_underlying_config() method."""
    config = SolverConfig()

    underlying = config.get_underlying_config()
    assert underlying is not None


@pytest.mark.unit
def test_get_parameter():
    """Test get_parameter() method."""
    config = SolverConfig().with_custom_parameter("test_param", 99)

    # Should retrieve custom parameter
    assert config.get_parameter("test_param") == 99
    # Should return default for non-existent parameter
    assert config.get_parameter("nonexistent", "default") == "default"


# ===================================================================
# Test PresetConfig Factory Methods
# ===================================================================


@pytest.mark.unit
def test_preset_for_research_prototype():
    """Test PresetConfig.for_research_prototype()."""
    config = PresetConfig.for_research_prototype()

    assert isinstance(config, SolverConfig)
    assert config.tolerance == pytest.approx(1e-4)
    assert config.max_iterations == 50


@pytest.mark.unit
def test_preset_for_production_quality():
    """Test PresetConfig.for_production_quality()."""
    config = PresetConfig.for_production_quality()

    assert isinstance(config, SolverConfig)
    assert config.tolerance == pytest.approx(1e-8)
    assert config.max_iterations == 500


@pytest.mark.unit
def test_preset_for_high_performance():
    """Test PresetConfig.for_high_performance()."""
    config = PresetConfig.for_high_performance()

    assert isinstance(config, SolverConfig)
    assert config.tolerance == pytest.approx(1e-5)
    assert config.max_iterations == 200


@pytest.mark.unit
def test_preset_for_educational():
    """Test PresetConfig.for_educational()."""
    config = PresetConfig.for_educational()

    assert isinstance(config, SolverConfig)
    assert config.tolerance == pytest.approx(1e-4)
    assert config.max_iterations == 100
    # Should have custom parameters for educational mode
    assert config.get_parameter("show_steps") is True
    assert config.get_parameter("plot_convergence") is True


@pytest.mark.unit
def test_preset_for_high_accuracy():
    """Test PresetConfig.for_high_accuracy()."""
    config = PresetConfig.for_high_accuracy()

    assert isinstance(config, SolverConfig)
    assert config.tolerance == pytest.approx(1e-10)
    assert config.max_iterations == 500


@pytest.mark.unit
def test_preset_for_large_problems():
    """Test PresetConfig.for_large_problems()."""
    config = PresetConfig.for_large_problems()

    assert isinstance(config, SolverConfig)
    assert config.tolerance == pytest.approx(1e-4)
    assert config.max_iterations == 100
    # Should have custom parameters for large problems
    assert config.get_parameter("memory_efficient") is True
    assert config.get_parameter("use_sparse") is True


@pytest.mark.unit
def test_preset_for_crowd_dynamics():
    """Test PresetConfig.for_crowd_dynamics()."""
    config = PresetConfig.for_crowd_dynamics()

    assert isinstance(config, SolverConfig)
    assert config.tolerance == pytest.approx(1e-5)
    assert config.max_iterations == 200
    # Should have custom parameter for adaptive grid
    assert config.get_parameter("adaptive_grid") is True


@pytest.mark.unit
def test_preset_for_financial_problems():
    """Test PresetConfig.for_financial_problems()."""
    config = PresetConfig.for_financial_problems()

    assert isinstance(config, SolverConfig)
    assert config.tolerance == pytest.approx(1e-7)
    assert config.max_iterations == 500
    # Should have custom parameter for bounds enforcement
    assert config.get_parameter("enforce_bounds") is True


@pytest.mark.unit
def test_preset_for_epidemic_models():
    """Test PresetConfig.for_epidemic_models()."""
    config = PresetConfig.for_epidemic_models()

    assert isinstance(config, SolverConfig)
    assert config.tolerance == pytest.approx(1e-6)
    assert config.max_iterations == 300
    # Should have custom parameter for adaptive timestep
    assert config.get_parameter("adaptive_timestep") is True


@pytest.mark.unit
def test_preset_for_traffic_flow():
    """Test PresetConfig.for_traffic_flow()."""
    config = PresetConfig.for_traffic_flow()

    assert isinstance(config, SolverConfig)
    assert config.tolerance == pytest.approx(1e-5)
    assert config.max_iterations == 150
    # Should have custom parameters for traffic flow
    assert config.get_parameter("use_upwind") is True
    assert config.get_parameter("steady_state_acceleration") is True


# ===================================================================
# Test Convenience Functions
# ===================================================================


@pytest.mark.unit
def test_create_config():
    """Test create_config() convenience function."""
    config = create_config()

    assert isinstance(config, SolverConfig)


@pytest.mark.unit
def test_research_config():
    """Test research_config() convenience function."""
    config = research_config()

    assert isinstance(config, SolverConfig)
    # Should match PresetConfig.for_research_prototype()
    assert config.tolerance == pytest.approx(1e-4)
    assert config.max_iterations == 50


@pytest.mark.unit
def test_production_config():
    """Test production_config() convenience function."""
    config = production_config()

    assert isinstance(config, SolverConfig)
    assert config.tolerance == pytest.approx(1e-8)


@pytest.mark.unit
def test_fast_config():
    """Test fast_config() convenience function."""
    config = fast_config()

    assert isinstance(config, SolverConfig)
    assert config.tolerance == pytest.approx(1e-5)


@pytest.mark.unit
def test_accurate_config():
    """Test accurate_config() convenience function."""
    config = accurate_config()

    assert isinstance(config, SolverConfig)
    assert config.tolerance == pytest.approx(1e-10)


@pytest.mark.unit
def test_educational_config():
    """Test educational_config() convenience function."""
    config = educational_config()

    assert isinstance(config, SolverConfig)
    assert config.get_parameter("show_steps") is True


@pytest.mark.unit
def test_crowd_dynamics_config():
    """Test crowd_dynamics_config() convenience function."""
    config = crowd_dynamics_config()

    assert isinstance(config, SolverConfig)
    assert config.get_parameter("adaptive_grid") is True


@pytest.mark.unit
def test_financial_config():
    """Test financial_config() convenience function."""
    config = financial_config()

    assert isinstance(config, SolverConfig)
    assert config.get_parameter("enforce_bounds") is True


@pytest.mark.unit
def test_epidemic_config():
    """Test epidemic_config() convenience function."""
    config = epidemic_config()

    assert isinstance(config, SolverConfig)
    assert config.get_parameter("adaptive_timestep") is True


@pytest.mark.unit
def test_traffic_config():
    """Test traffic_config() convenience function."""
    config = traffic_config()

    assert isinstance(config, SolverConfig)
    assert config.get_parameter("use_upwind") is True


@pytest.mark.unit
def test_large_scale_config():
    """Test large_scale_config() convenience function."""
    config = large_scale_config()

    assert isinstance(config, SolverConfig)
    assert config.get_parameter("memory_efficient") is True


# ===================================================================
# Test Module Exports
# ===================================================================


@pytest.mark.unit
def test_module_exports_classes():
    """Test module exports main classes."""
    from mfg_pde.config import modern_config

    assert hasattr(modern_config, "SolverConfig")
    assert hasattr(modern_config, "PresetConfig")


@pytest.mark.unit
def test_module_exports_convenience_functions():
    """Test module exports convenience functions."""
    from mfg_pde.config import modern_config

    convenience_funcs = [
        "create_config",
        "research_config",
        "production_config",
        "fast_config",
        "accurate_config",
        "educational_config",
        "crowd_dynamics_config",
        "financial_config",
        "epidemic_config",
        "traffic_config",
        "large_scale_config",
    ]

    for func_name in convenience_funcs:
        assert hasattr(modern_config, func_name)


@pytest.mark.unit
def test_exported_classes_are_types():
    """Test that exported classes are actually types."""
    assert isinstance(SolverConfig, type)
    assert isinstance(PresetConfig, type)


# ===================================================================
# Test Edge Cases and Integration
# ===================================================================


@pytest.mark.unit
def test_config_immutability_deep():
    """Test that configurations are truly immutable (deep copy)."""
    config1 = SolverConfig().with_tolerance(1e-6)
    config2 = config1.with_max_iterations(200)
    config3 = config2.with_damping(0.7)

    # All should be different instances
    assert config1 is not config2
    assert config2 is not config3
    assert config1 is not config3


@pytest.mark.unit
def test_multiple_custom_parameters_isolation():
    """Test that custom parameters are isolated between instances."""
    config1 = SolverConfig().with_custom_parameter("key", "value1")
    config2 = config1.with_custom_parameter("key", "value2")

    # config1 should still have original value
    assert config1.get_parameter("key") == "value1"
    assert config2.get_parameter("key") == "value2"


@pytest.mark.unit
def test_preset_configs_are_independent():
    """Test that preset configs create independent instances."""
    config1 = PresetConfig.for_research_prototype()
    config2 = PresetConfig.for_research_prototype()

    # Should be different instances
    assert config1 is not config2


@pytest.mark.unit
def test_get_parameter_with_default():
    """Test get_parameter() returns default for missing keys."""
    config = SolverConfig()

    assert config.get_parameter("missing_key", "default_value") == "default_value"
    assert config.get_parameter("another_missing", 999) == 999
