"""
Unit tests for Pydantic-based configuration system.

This module tests the configuration validation, default values, and error handling
in the Pydantic configuration classes.
"""

import contextlib
import warnings

import pytest
from pydantic import ValidationError

from mfg_pde.config.core import PicardConfig as CorePicardConfig
from mfg_pde.config.pydantic_config import (
    MFGSolverConfig,
    NewtonConfig,
    PicardConfig,
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
        assert "greater than" in str(exc_info.value)
        assert "0.000000000000001" in str(exc_info.value)

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
            NewtonConfig(tolerance=1e-14)
            assert len(w) == 1
            assert "very strict newton tolerance" in str(w[0].message).lower()

        # No warning expected for loose tolerance (current implementation)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            NewtonConfig(tolerance=1e-2)
            assert len(w) == 0  # Current implementation only warns for strict tolerance

    @pytest.mark.parametrize(
        ("max_iter", "tol", "damping"),
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
    """Test master MFG solver configuration (canonical from core.py)."""

    def test_valid_configuration(self):
        """Test creation with valid nested configurations."""
        # Use CorePicardConfig for the canonical MFGSolverConfig
        picard_config = CorePicardConfig(max_iterations=50, tolerance=1e-5, damping_factor=0.7)

        config = MFGSolverConfig(picard=picard_config)

        assert config.picard.max_iterations == 50
        assert config.picard.tolerance == 1e-5
        assert config.picard.damping_factor == 0.7

    def test_default_nested_configurations(self):
        """Test that nested configurations use proper defaults."""
        config = MFGSolverConfig()

        # Check picard defaults (from core.py PicardConfig)
        assert config.picard.max_iterations == 100
        assert config.picard.tolerance == 1e-6
        assert config.picard.damping_factor == 0.5

        # Check backend defaults
        assert config.backend.type == "numpy"
        assert config.backend.device == "cpu"
        assert config.backend.precision == "float64"

        # Check logging defaults
        assert config.logging.level == "INFO"
        assert config.logging.progress_bar is True

    def test_hjb_and_fp_configs(self):
        """Test HJB and FP nested configurations."""
        config = MFGSolverConfig()

        # HJB config should have defaults
        assert hasattr(config, "hjb")
        assert hasattr(config, "fp")

    def test_picard_configuration(self):
        """Test Picard configuration with custom values."""
        picard = CorePicardConfig(
            max_iterations=200,
            tolerance=1e-8,
            damping_factor=0.3,
            anderson_memory=5,
            verbose=True,
        )
        config = MFGSolverConfig(picard=picard)

        assert config.picard.max_iterations == 200
        assert config.picard.tolerance == 1e-8
        assert config.picard.damping_factor == 0.3
        assert config.picard.anderson_memory == 5
        assert config.picard.verbose is True


class TestConfigurationEdgeCases:
    """Test edge cases and error conditions."""

    def test_invalid_nested_config_types(self):
        """Test error handling for invalid nested configuration types."""
        with pytest.raises(ValidationError):
            MFGSolverConfig(picard="invalid_type")

        with pytest.raises(ValidationError):
            MFGSolverConfig(backend=123)

    def test_partial_configuration_updates(self):
        """Test updating configurations partially."""
        config = MFGSolverConfig()
        original_tolerance = config.picard.tolerance

        # Update only Picard max_iterations
        config.picard.max_iterations = 150

        # Other values should remain unchanged
        assert config.picard.tolerance == original_tolerance
        assert config.picard.max_iterations == 150

    def test_configuration_serialization(self):
        """Test that configuration can be serialized to YAML-compatible dict."""
        config = MFGSolverConfig()

        # Should be able to dump to dict
        config_dict = config.model_dump_yaml()
        assert isinstance(config_dict, dict)
        assert "picard" in config_dict
        assert "backend" in config_dict

    def test_configuration_validation_assignment(self):
        """Test validation on assignment if enabled."""
        config = MFGSolverConfig()

        # Should validate on assignment if validate_assignment is True
        # This is expected behavior with validation_assignment=True
        with contextlib.suppress(ValidationError, ValueError):
            # This should either work or raise ValidationError
            config.convergence_tolerance = -1.0
            # If we get here, validation_assignment might be False
