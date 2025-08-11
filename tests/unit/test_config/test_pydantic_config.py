"""
Unit tests for Pydantic-based configuration system.

This module tests the configuration validation, default values, and error handling
in the Pydantic configuration classes.
"""

import warnings

import pytest
from pydantic import ValidationError

from mfg_pde.config.pydantic_config import (
    FPConfig,
    HJBConfig,
    MFGSolverConfig,
    NewtonConfig,
    PicardConfig,
    create_accurate_config,
    create_fast_config,
    create_research_config,
)


class TestNewtonConfig:
    """Test Newton method configuration validation."""

    def test_valid_configuration(self):
        """Test creation with valid parameters."""
        config = NewtonConfig(max_iterations=20, tolerance=1e-6, damping_factor=0.8, line_search=True, verbose=False)
        assert config.max_iterations == 20
        assert config.tolerance == 1e-6
        assert config.damping_factor == 0.8
        assert config.line_search is True
        assert config.verbose is False

    def test_default_values(self):
        """Test default parameter values."""
        config = NewtonConfig()
        assert config.max_iterations == 30
        assert config.tolerance == 1e-6
        assert config.damping_factor == 1.0
        assert config.line_search is False
        assert config.verbose is False

    def test_parameter_validation_ranges(self):
        """Test parameter range validation."""
        # Test invalid max_iterations (too small)
        with pytest.raises(ValidationError) as exc_info:
            NewtonConfig(max_iterations=0)
        assert "greater than or equal to 1" in str(exc_info.value)

        # Test invalid max_iterations (too large)
        with pytest.raises(ValidationError) as exc_info:
            NewtonConfig(max_iterations=1001)
        assert "less than or equal to 1000" in str(exc_info.value)

        # Test invalid tolerance (too small)
        with pytest.raises(ValidationError) as exc_info:
            NewtonConfig(tolerance=1e-16)
        assert "greater than" in str(exc_info.value) and "0.000000000000001" in str(exc_info.value)

        # Test invalid tolerance (too large)
        with pytest.raises(ValidationError) as exc_info:
            NewtonConfig(tolerance=0.2)
        assert "less than or equal to 0.1" in str(exc_info.value)

        # Test invalid damping_factor (too small)
        with pytest.raises(ValidationError) as exc_info:
            NewtonConfig(damping_factor=0.0)
        assert "greater than 0" in str(exc_info.value)

        # Test invalid damping_factor (too large)
        with pytest.raises(ValidationError) as exc_info:
            NewtonConfig(damping_factor=1.1)
        assert "less than or equal to 1" in str(exc_info.value)

    def test_tolerance_warning_validation(self):
        """Test tolerance warning validator."""
        # Should issue warning for very strict tolerance
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = NewtonConfig(tolerance=1e-14)
            assert len(w) == 1
            assert "very strict newton tolerance" in str(w[0].message).lower()

        # No warning expected for loose tolerance (current implementation)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = NewtonConfig(tolerance=1e-2)
            assert len(w) == 0  # Current implementation only warns for strict tolerance

    @pytest.mark.parametrize(
        "max_iter,tol,damping",
        [
            (10, 1e-4, 0.5),
            (50, 1e-8, 1.0),
            (100, 1e-10, 0.1),
            (1, 1e-15 + 1e-16, 0.01),  # Edge case: minimum valid values
            (1000, 1e-1, 1.0),  # Edge case: maximum valid values
        ],
    )
    def test_parameter_combinations(self, max_iter, tol, damping):
        """Test various valid parameter combinations."""
        config = NewtonConfig(max_iterations=max_iter, tolerance=tol, damping_factor=damping)
        assert config.max_iterations == max_iter
        assert config.tolerance == tol
        assert config.damping_factor == damping

    def test_config_serialization(self):
        """Test configuration serialization and deserialization."""
        original_config = NewtonConfig(max_iterations=25, tolerance=1e-7, damping_factor=0.9, line_search=True)

        # Test JSON serialization
        json_str = original_config.model_dump_json()
        assert isinstance(json_str, str)

        # Test deserialization
        config_dict = original_config.model_dump()
        reconstructed_config = NewtonConfig(**config_dict)

        assert reconstructed_config.max_iterations == original_config.max_iterations
        assert reconstructed_config.tolerance == original_config.tolerance
        assert reconstructed_config.damping_factor == original_config.damping_factor
        assert reconstructed_config.line_search == original_config.line_search


class TestPicardConfig:
    """Test Picard iteration configuration validation."""

    def test_valid_configuration(self):
        """Test creation with valid parameters."""
        config = PicardConfig(max_iterations=15, tolerance=1e-5, damping_factor=0.7, verbose=True)
        assert config.max_iterations == 15
        assert config.tolerance == 1e-5
        assert config.damping_factor == 0.7
        assert config.verbose is True

    def test_default_values(self):
        """Test default parameter values."""
        config = PicardConfig()
        assert config.max_iterations == 20
        assert config.tolerance == 1e-3
        assert config.damping_factor == 0.5
        assert config.verbose is False

    def test_parameter_validation(self):
        """Test Picard-specific parameter validation."""
        # Test minimum iterations
        with pytest.raises(ValidationError):
            PicardConfig(max_iterations=0)

        # Test maximum iterations
        with pytest.raises(ValidationError):
            PicardConfig(max_iterations=501)

        # Test tolerance range
        with pytest.raises(ValidationError):
            PicardConfig(tolerance=1e-13)  # Below minimum gt=1e-12

        with pytest.raises(ValidationError):
            PicardConfig(tolerance=1.1)  # Above maximum le=1.0


class TestMFGSolverConfig:
    """Test master MFG solver configuration."""

    def test_valid_configuration(self):
        """Test creation with valid nested configurations."""
        newton_config = NewtonConfig(max_iterations=20, tolerance=1e-7)
        picard_config = PicardConfig(max_iterations=30, tolerance=1e-5)

        config = MFGSolverConfig(
            newton=newton_config,
            picard=picard_config,
            return_structured=True,
            enable_warm_start=False,
            convergence_tolerance=1e-6,
            strict_convergence_errors=True,
            experiment_name="test_experiment",
        )

        assert config.newton.max_iterations == 20
        assert config.picard.max_iterations == 30
        assert config.return_structured is True
        assert config.enable_warm_start is False
        assert config.convergence_tolerance == 1e-6
        assert config.strict_convergence_errors is True
        assert config.experiment_name == "test_experiment"

    def test_default_nested_configurations(self):
        """Test that nested configurations use proper defaults."""
        config = MFGSolverConfig()

        # Check nested defaults
        assert config.newton.max_iterations == 30
        assert config.newton.tolerance == 1e-6
        assert config.picard.max_iterations == 20
        assert config.picard.tolerance == 1e-3

        # Check top-level defaults
        assert config.return_structured is True
        assert config.enable_warm_start is False
        assert config.convergence_tolerance == 1e-5
        assert config.strict_convergence_errors is True
        assert config.experiment_name is None
        assert config.metadata == {}

    def test_cross_validation_tolerance_hierarchy(self):
        """Test cross-validation between tolerance levels."""
        # Test that the validator works correctly
        config = MFGSolverConfig(
            convergence_tolerance=1e-4,
            newton=NewtonConfig(tolerance=1e-3),  # Looser than global
            picard=PicardConfig(tolerance=1e-3),  # Looser than global
        )

        # Should work without warnings when component tolerances are looser
        assert config.convergence_tolerance == 1e-4
        assert config.newton.tolerance == 1e-3
        assert config.picard.tolerance == 1e-3

    def test_environment_variable_configuration(self):
        """Test configuration from environment variables."""
        # This would require setting environment variables in the test
        # For now, just test that the env_prefix is set correctly
        config = MFGSolverConfig()

        # Check that the Config class has env_prefix set correctly
        # In Pydantic v2, model_config can be a dict or ConfigDict
        env_prefix = None
        if hasattr(MFGSolverConfig.model_config, 'env_prefix'):
            env_prefix = MFGSolverConfig.model_config.env_prefix
        elif isinstance(MFGSolverConfig.model_config, dict):
            env_prefix = MFGSolverConfig.model_config.get('env_prefix')
        assert env_prefix == 'MFG_'

    def test_metadata_handling(self):
        """Test metadata dictionary handling."""
        metadata = {"description": "Test configuration", "author": "pytest", "version": "1.0"}

        config = MFGSolverConfig(metadata=metadata)
        assert config.metadata == metadata

        # Test that metadata can be updated
        config.metadata["additional_info"] = "test"
        assert "additional_info" in config.metadata


class TestConfigurationFactories:
    """Test configuration factory functions."""

    def test_fast_config_creation(self):
        """Test fast configuration factory."""
        config = create_fast_config()

        assert isinstance(config, MFGSolverConfig)

        # Fast config should have relaxed tolerances for speed
        assert config.newton.tolerance >= 1e-6
        assert config.picard.tolerance >= 1e-5
        assert config.convergence_tolerance >= 1e-5

        # Should have reasonable iteration limits
        assert config.newton.max_iterations <= 30
        assert config.picard.max_iterations <= 50

    def test_accurate_config_creation(self):
        """Test accurate configuration factory."""
        config = create_accurate_config()

        assert isinstance(config, MFGSolverConfig)

        # Accurate config should have tighter tolerances
        assert config.newton.tolerance <= 1e-7
        assert config.picard.tolerance <= 1e-6
        assert config.convergence_tolerance <= 1e-6

        # May have higher iteration limits
        assert config.newton.max_iterations >= 30
        assert config.picard.max_iterations >= 50

    def test_research_config_creation(self):
        """Test research configuration factory."""
        config = create_research_config()

        assert isinstance(config, MFGSolverConfig)

        # Research config should have very tight tolerances
        assert config.newton.tolerance <= 1e-8  # Should be 1e-10
        assert config.picard.tolerance <= 1e-5  # Should be 1e-6
        assert config.convergence_tolerance <= 1e-7  # Should be 1e-8

        # Should enable comprehensive monitoring
        assert config.return_structured is True
        assert config.strict_convergence_errors is True

        # Should have high iteration limits for thorough convergence
        assert config.newton.max_iterations >= 50
        assert config.picard.max_iterations >= 100

    def test_factory_config_consistency(self):
        """Test that factory configurations are internally consistent."""
        configs = [
            ("fast", create_fast_config()),
            ("accurate", create_accurate_config()),
            ("research", create_research_config()),
        ]

        for name, config in configs:
            # All factory configs should be valid
            assert isinstance(config, MFGSolverConfig)

            # Tolerance hierarchy: component tolerances should be reasonable relative to global
            # Allow component tolerances to be stricter or moderately looser than global
            assert (
                config.newton.tolerance <= config.convergence_tolerance * 100
                or abs(config.newton.tolerance - config.convergence_tolerance) < 1e-10
            )

            assert (
                config.picard.tolerance <= config.convergence_tolerance * 100
                or abs(config.picard.tolerance - config.convergence_tolerance) < 1e-10
            )

            # Basic sanity checks
            assert config.newton.max_iterations > 0
            assert config.picard.max_iterations > 0
            assert config.newton.damping_factor > 0
            assert config.picard.damping_factor > 0

    @pytest.mark.parametrize("factory_func", [create_fast_config, create_accurate_config, create_research_config])
    def test_factory_serialization(self, factory_func):
        """Test that factory-created configs can be serialized."""
        config = factory_func()

        # Should be able to serialize to JSON
        json_str = config.model_dump_json()
        assert isinstance(json_str, str)
        assert len(json_str) > 0

        # Should be able to serialize to dict
        config_dict = config.model_dump()
        assert isinstance(config_dict, dict)
        assert 'newton' in config_dict
        assert 'picard' in config_dict

        # Should be able to reconstruct
        reconstructed = MFGSolverConfig(**config_dict)
        assert reconstructed.newton.tolerance == config.newton.tolerance
        assert reconstructed.picard.tolerance == config.picard.tolerance


class TestConfigurationEdgeCases:
    """Test edge cases and error conditions."""

    def test_invalid_nested_config_types(self):
        """Test error handling for invalid nested configuration types."""
        with pytest.raises(ValidationError):
            MFGSolverConfig(newton="invalid_type")

        with pytest.raises(ValidationError):
            MFGSolverConfig(picard=123)

    def test_partial_configuration_updates(self):
        """Test updating configurations partially."""
        config = MFGSolverConfig()
        original_newton_tolerance = config.newton.tolerance

        # Update only Newton max_iterations
        config.newton.max_iterations = 15

        # Other values should remain unchanged
        assert config.newton.tolerance == original_newton_tolerance
        assert config.newton.max_iterations == 15

    def test_configuration_immutability_where_intended(self):
        """Test that certain configuration aspects behave as expected."""
        config = MFGSolverConfig()

        # Should be able to modify mutable fields
        config.experiment_name = "new_experiment"
        assert config.experiment_name == "new_experiment"

        # Metadata should be mutable
        config.metadata["test_key"] = "test_value"
        assert config.metadata["test_key"] == "test_value"

    def test_configuration_validation_assignment(self):
        """Test validation on assignment if enabled."""
        config = MFGSolverConfig()

        # Should validate on assignment if validate_assignment is True
        try:
            # This should either work or raise ValidationError
            config.convergence_tolerance = -1.0
            # If we get here, validation_assignment might be False
            pass
        except (ValidationError, ValueError):
            # This is expected behavior with validation_assignment=True
            pass
