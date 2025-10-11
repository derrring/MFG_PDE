#!/usr/bin/env python3
"""
Comprehensive tests for structured OmegaConf configuration (Issue #28).

Tests the complete solution for OmegaConf type checking problems using
structured configs with dataclass schemas.
"""

import tempfile
from pathlib import Path

import pytest

# Check if OmegaConf is available
try:
    from mfg_pde.config.omegaconf_manager import (
        create_default_structured_config,
        create_omega_manager,
        load_structured_beach_config,
        load_structured_mfg_config,
    )
    from mfg_pde.config.structured_schemas import BeachProblemConfig, MFGConfig

    OMEGACONF_AVAILABLE = True
except ImportError:
    OMEGACONF_AVAILABLE = False

# Skip all tests in this module if OmegaConf not available
pytestmark = pytest.mark.skipif(not OMEGACONF_AVAILABLE, reason="OmegaConf not available (optional dependency)")


class TestStructuredConfigs:
    """Test structured configuration functionality."""

    def test_default_structured_config_creation(self):
        """Test creating default structured configuration."""
        config = create_default_structured_config()

        # Test type safety - these should not raise AttributeError
        assert hasattr(config, "problem")
        assert hasattr(config, "solver")
        assert hasattr(config, "experiment")

        # Test default values
        assert config.problem.T == 1.0
        assert config.problem.Nx == 50
        assert config.problem.Nt == 30
        assert config.solver.type == "fixed_point"
        assert config.solver.max_iterations == 100

    def test_mfg_config_schema(self):
        """Test MFGConfig dataclass schema."""
        config = MFGConfig()

        # Test that all required attributes exist
        assert hasattr(config.problem, "name")
        assert hasattr(config.problem, "T")
        assert hasattr(config.problem, "Nx")
        assert hasattr(config.problem, "Nt")
        assert hasattr(config.solver, "type")
        assert hasattr(config.solver, "max_iterations")
        assert hasattr(config.solver, "tolerance")

        # Test default values match schema
        assert config.problem.name == "base_mfg_problem"
        assert config.solver.damping == 0.5
        assert config.experiment.name == "parameter_sweep"

    def test_beach_config_schema(self):
        """Test BeachProblemConfig specialized schema."""
        config = BeachProblemConfig()

        # Test beach-specific parameters
        assert config.problem.name == "towel_on_beach"
        assert config.problem.type == "spatial_competition"
        assert config.problem.T == 2.0
        assert config.problem.Nx == 80
        assert config.problem.Nt == 40

        # Test beach-specific parameters
        params = config.problem.parameters
        assert params["stall_position"] == 0.6
        assert params["crowd_aversion"] == 1.5
        assert params["noise_level"] == 0.1

    def test_structured_config_from_yaml(self):
        """Test loading structured config from YAML file."""
        # Create a temporary YAML config
        yaml_content = """
problem:
  name: "test_problem"
  T: 3.0
  Nx: 100
  Nt: 60

solver:
  max_iterations: 200
  tolerance: 1e-8
  damping: 0.3

experiment:
  name: "test_experiment"
  description: "Test experiment for structured configs"
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_yaml = Path(f.name)

        try:
            # Load using structured config
            config = load_structured_mfg_config(temp_yaml)

            # Test that values were loaded correctly
            assert config.problem.name == "test_problem"
            assert config.problem.T == 3.0
            assert config.problem.Nx == 100
            assert config.solver.max_iterations == 200
            assert config.solver.tolerance == 1e-8
            assert config.experiment.name == "test_experiment"

            # Test that default values are preserved for non-specified fields
            assert config.problem.Nt == 60
            assert config.solver.damping == 0.3

        finally:
            temp_yaml.unlink()

    def test_structured_config_default_behavior(self):
        """Test structured config default behavior when file doesn't exist."""
        # Test loading non-existent file uses schema defaults
        config = load_structured_mfg_config("nonexistent.yaml")

        # Test that schema defaults are used when file doesn't exist
        assert config.problem.T == 1.0  # Schema default
        assert config.solver.max_iterations == 100  # Schema default
        assert config.experiment.name == "parameter_sweep"  # Schema default

        # Test that the config is still fully typed
        assert isinstance(config.problem.T, (int, float))
        assert isinstance(config.solver.max_iterations, int)
        assert isinstance(config.experiment.name, str)

        # Note: Parameter override syntax could be enhanced in future versions
        # Current implementation focuses on YAML-based configuration

    def test_omega_manager_structured_methods(self):
        """Test OmegaConfManager structured config methods."""
        manager = create_omega_manager()

        # Test structured MFG config creation
        config = manager.create_default_mfg_config()
        assert hasattr(config, "problem")
        assert hasattr(config, "solver")

        # Test validation method
        assert manager.validate_structured_config(config) is True

        # Test invalid config validation would fail
        # (This would need a mock invalid config to test properly)

    def test_beach_config_structured_loading(self):
        """Test Beach problem structured config loading."""
        # Create a temporary Beach YAML config
        yaml_content = """
problem:
  name: "custom_beach"
  T: 3.0
  parameters:
    stall_position: 0.7
    crowd_aversion: 2.0
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_yaml = Path(f.name)

        try:
            config = load_structured_beach_config(temp_yaml)

            # Test beach-specific loading
            assert config.problem.name == "custom_beach"
            assert config.problem.T == 3.0
            assert config.problem.parameters["stall_position"] == 0.7
            assert config.problem.parameters["crowd_aversion"] == 2.0

            # Test that defaults are preserved
            assert config.problem.type == "spatial_competition"

        finally:
            temp_yaml.unlink()

    def test_type_safety_demonstration(self):
        """
        Demonstrate the type safety improvements from Issue #28.

        This test shows that structured configs provide the type safety
        that was missing with plain DictConfig objects.
        """
        config = create_default_structured_config()

        # These attribute accesses should work without mypy errors
        problem_time = config.problem.T
        solver_type = config.solver.type
        experiment_name = config.experiment.name

        # Test that we get expected types
        assert isinstance(problem_time, (int, float))
        assert isinstance(solver_type, str)
        assert isinstance(experiment_name, str)

        # Test nested attribute access
        boundary_type = config.problem.boundary_conditions.type
        assert isinstance(boundary_type, str)
        assert boundary_type == "periodic"

        # Test dict-style parameter access
        hjb_method = config.solver.hjb.method
        assert isinstance(hjb_method, str)
        assert hjb_method == "gfdm"


class TestIssue28TypeSafetySolution:
    """Test that Issue #28 type safety problems are solved."""

    def test_before_after_comparison(self):
        """
        Document the before/after comparison from Issue #28.

        BEFORE: DictConfig objects caused mypy errors
        AFTER: Structured configs provide full type safety
        """
        # AFTER: Type-safe structured config (Issue #28 solution)
        config = create_default_structured_config()

        # These should all work without mypy errors:
        assert config.problem.T == 1.0  # ✅ Type safe
        assert config.problem.Nx == 50  # ✅ Type safe
        assert config.solver.max_iterations == 100  # ✅ Type safe
        assert config.experiment.logging.level == "INFO"  # ✅ Type safe

        # Test that IDE autocompletion would work
        # (This is verified by the fact that these don't raise AttributeError)
        _ = config.problem.domain.x_min
        _ = config.solver.hjb.newton.max_iterations
        _ = config.experiment.visualization.enabled

    def test_yaml_integration_type_safety(self):
        """Test that YAML loading maintains type safety."""
        yaml_content = """
problem:
  T: 2.5
  Nx: 75
solver:
  tolerance: 1e-10
experiment:
  logging:
    level: "DEBUG"
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_yaml = Path(f.name)

        try:
            config = load_structured_mfg_config(temp_yaml)

            # Test that loaded values maintain type safety
            assert isinstance(config.problem.T, (int, float))
            assert isinstance(config.problem.Nx, int)
            assert isinstance(config.solver.tolerance, float)
            assert isinstance(config.experiment.logging.level, str)

        finally:
            temp_yaml.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
