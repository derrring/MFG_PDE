"""
Unit tests for dataclass-based solver configuration system.

This module tests the dataclass configuration classes, validation logic,
factory functions, and backward compatibility handling.
"""

import warnings
from dataclasses import asdict

import pytest

from mfg_pde.config.solver_config import (
    FPConfig,
    GFDMConfig,
    HJBConfig,
    MFGSolverConfig,
    NewtonConfig,
    ParticleConfig,
    PicardConfig,
    create_accurate_config,
    create_default_config,
    create_fast_config,
    create_production_config,
    create_research_config,
    extract_legacy_parameters,
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

    def test_post_init_validation_max_iterations(self):
        """Test __post_init__ validates max_iterations."""
        with pytest.raises(ValueError, match="max_iterations must be >= 1"):
            NewtonConfig(max_iterations=0)

        with pytest.raises(ValueError, match="max_iterations must be >= 1"):
            NewtonConfig(max_iterations=-5)

    def test_post_init_validation_tolerance(self):
        """Test __post_init__ validates tolerance."""
        with pytest.raises(ValueError, match="tolerance must be > 0"):
            NewtonConfig(tolerance=0.0)

        with pytest.raises(ValueError, match="tolerance must be > 0"):
            NewtonConfig(tolerance=-1e-6)

    def test_post_init_validation_damping_factor(self):
        """Test __post_init__ validates damping_factor range."""
        with pytest.raises(ValueError, match="damping_factor must be in"):
            NewtonConfig(damping_factor=0.0)

        with pytest.raises(ValueError, match="damping_factor must be in"):
            NewtonConfig(damping_factor=1.1)

        with pytest.raises(ValueError, match="damping_factor must be in"):
            NewtonConfig(damping_factor=-0.5)

    def test_factory_fast(self):
        """Test fast configuration factory."""
        config = NewtonConfig.fast()
        assert config.max_iterations == 10
        assert config.tolerance == 1e-4
        assert config.damping_factor == 0.8

    def test_factory_accurate(self):
        """Test accurate configuration factory."""
        config = NewtonConfig.accurate()
        assert config.max_iterations == 50
        assert config.tolerance == 1e-8
        assert config.damping_factor == 1.0

    def test_edge_case_minimum_valid_values(self):
        """Test edge case with minimum valid parameters."""
        config = NewtonConfig(max_iterations=1, tolerance=1e-15, damping_factor=1e-10)
        assert config.max_iterations == 1
        assert config.tolerance == 1e-15
        assert config.damping_factor > 0

    def test_serialization_to_dict(self):
        """Test conversion to dictionary via asdict."""
        config = NewtonConfig(max_iterations=25, tolerance=1e-7, damping_factor=0.9)
        config_dict = asdict(config)

        assert config_dict["max_iterations"] == 25
        assert config_dict["tolerance"] == 1e-7
        assert config_dict["damping_factor"] == 0.9


class TestPicardConfig:
    """Test Picard iteration configuration validation."""

    def test_valid_configuration(self):
        """Test creation with valid parameters."""
        config = PicardConfig(
            max_iterations=15, tolerance=1e-5, damping_factor=0.7, convergence_check_frequency=2, verbose=True
        )
        assert config.max_iterations == 15
        assert config.tolerance == 1e-5
        assert config.damping_factor == 0.7
        assert config.convergence_check_frequency == 2
        assert config.verbose is True

    def test_default_values(self):
        """Test default parameter values."""
        config = PicardConfig()
        assert config.max_iterations == 20
        assert config.tolerance == 1e-5
        assert config.damping_factor == 0.5
        assert config.convergence_check_frequency == 1
        assert config.verbose is True

    def test_post_init_validation_max_iterations(self):
        """Test __post_init__ validates max_iterations."""
        with pytest.raises(ValueError, match="max_iterations must be >= 1"):
            PicardConfig(max_iterations=0)

    def test_post_init_validation_tolerance(self):
        """Test __post_init__ validates tolerance."""
        with pytest.raises(ValueError, match="tolerance must be > 0"):
            PicardConfig(tolerance=0.0)

    def test_post_init_validation_damping_factor(self):
        """Test __post_init__ validates damping_factor."""
        with pytest.raises(ValueError, match="damping_factor must be in"):
            PicardConfig(damping_factor=0.0)

        with pytest.raises(ValueError, match="damping_factor must be in"):
            PicardConfig(damping_factor=1.5)

    def test_post_init_validation_convergence_check_frequency(self):
        """Test __post_init__ validates convergence_check_frequency."""
        with pytest.raises(ValueError, match="convergence_check_frequency must be >= 1"):
            PicardConfig(convergence_check_frequency=0)

    def test_factory_fast(self):
        """Test fast configuration factory."""
        config = PicardConfig.fast()
        assert config.max_iterations == 10
        assert config.tolerance == 1e-3
        assert config.damping_factor == 0.7

    def test_factory_accurate(self):
        """Test accurate configuration factory."""
        config = PicardConfig.accurate()
        assert config.max_iterations == 50
        assert config.tolerance == 1e-7
        assert config.damping_factor == 0.3


class TestGFDMConfig:
    """Test GFDM solver configuration validation."""

    def test_valid_configuration(self):
        """Test creation with valid parameters."""
        config = GFDMConfig(
            delta=0.15, taylor_order=3, weight_function="wendland", weight_scale=2.0, use_qp_constraints=True
        )
        assert config.delta == 0.15
        assert config.taylor_order == 3
        assert config.weight_function == "wendland"
        assert config.weight_scale == 2.0
        assert config.use_qp_constraints is True

    def test_default_values(self):
        """Test default parameter values."""
        config = GFDMConfig()
        assert config.delta == 0.1
        assert config.taylor_order == 2
        assert config.weight_function == "gaussian"
        assert config.weight_scale == 1.0
        assert config.use_qp_constraints is False
        assert config.boundary_method == "dirichlet"

    def test_post_init_validation_delta(self):
        """Test __post_init__ validates delta."""
        with pytest.raises(ValueError, match="delta must be > 0"):
            GFDMConfig(delta=0.0)

        with pytest.raises(ValueError, match="delta must be > 0"):
            GFDMConfig(delta=-0.1)

    def test_post_init_validation_taylor_order(self):
        """Test __post_init__ validates taylor_order."""
        with pytest.raises(ValueError, match="taylor_order must be >= 1"):
            GFDMConfig(taylor_order=0)

    def test_post_init_validation_weight_scale(self):
        """Test __post_init__ validates weight_scale."""
        with pytest.raises(ValueError, match="weight_scale must be > 0"):
            GFDMConfig(weight_scale=0.0)

    def test_weight_function_valid_values(self):
        """Test that all weight function types are valid."""
        valid_functions = ["gaussian", "inverse_distance", "uniform", "wendland"]
        for func in valid_functions:
            config = GFDMConfig(weight_function=func)
            assert config.weight_function == func

    def test_boundary_method_valid_values(self):
        """Test that all boundary methods are valid."""
        valid_methods = ["dirichlet", "neumann", "extrapolation"]
        for method in valid_methods:
            config = GFDMConfig(boundary_method=method)
            assert config.boundary_method == method

    def test_factory_fast(self):
        """Test fast configuration factory."""
        config = GFDMConfig.fast()
        assert config.delta == 0.15
        assert config.taylor_order == 1
        assert config.weight_function == "uniform"

    def test_factory_accurate(self):
        """Test accurate configuration factory."""
        config = GFDMConfig.accurate()
        assert config.delta == 0.05
        assert config.taylor_order == 3
        assert config.weight_function == "wendland"
        assert config.use_qp_constraints is True


class TestParticleConfig:
    """Test particle solver configuration validation."""

    def test_valid_configuration(self):
        """Test creation with valid parameters."""
        config = ParticleConfig(
            num_particles=10000, kde_bandwidth=0.02, normalize_output=True, boundary_handling="reflecting"
        )
        assert config.num_particles == 10000
        assert config.kde_bandwidth == 0.02
        assert config.normalize_output is True
        assert config.boundary_handling == "reflecting"

    def test_default_values(self):
        """Test default parameter values."""
        config = ParticleConfig()
        assert config.num_particles == 5000
        assert config.kde_bandwidth == "scott"
        assert config.normalize_output is True
        assert config.boundary_handling == "absorbing"
        assert config.random_seed is None

    def test_post_init_validation_num_particles(self):
        """Test __post_init__ validates num_particles."""
        with pytest.raises(ValueError, match="num_particles must be >= 10"):
            ParticleConfig(num_particles=5)

    def test_post_init_validation_kde_bandwidth_numeric(self):
        """Test __post_init__ validates numeric kde_bandwidth."""
        with pytest.raises(ValueError, match="kde_bandwidth must be > 0"):
            ParticleConfig(kde_bandwidth=0.0)

        with pytest.raises(ValueError, match="kde_bandwidth must be > 0"):
            ParticleConfig(kde_bandwidth=-0.01)

    def test_kde_bandwidth_string_values(self):
        """Test that string kde_bandwidth values are valid."""
        valid_strings = ["scott", "silverman"]
        for bw in valid_strings:
            config = ParticleConfig(kde_bandwidth=bw)
            assert config.kde_bandwidth == bw

    def test_boundary_handling_valid_values(self):
        """Test that all boundary handling types are valid."""
        valid_types = ["absorbing", "reflecting", "periodic"]
        for bh_type in valid_types:
            config = ParticleConfig(boundary_handling=bh_type)
            assert config.boundary_handling == bh_type

    def test_factory_fast(self):
        """Test fast configuration factory."""
        config = ParticleConfig.fast()
        assert config.num_particles == 1000
        assert config.kde_bandwidth == "scott"
        assert config.normalize_output is False

    def test_factory_accurate(self):
        """Test accurate configuration factory."""
        config = ParticleConfig.accurate()
        assert config.num_particles == 10000
        assert config.kde_bandwidth == 0.01
        assert config.normalize_output is True


class TestHJBConfig:
    """Test HJB solver configuration."""

    def test_valid_configuration(self):
        """Test creation with valid nested configurations."""
        newton_config = NewtonConfig(max_iterations=20, tolerance=1e-7)
        gfdm_config = GFDMConfig(delta=0.05, use_qp_constraints=True)

        config = HJBConfig(newton=newton_config, gfdm=gfdm_config, solver_type="gfdm_qp")

        assert config.newton.max_iterations == 20
        assert config.gfdm.delta == 0.05
        assert config.solver_type == "gfdm_qp"

    def test_default_nested_configurations(self):
        """Test that nested configurations use proper defaults."""
        config = HJBConfig()
        assert config.newton.max_iterations == 30
        assert config.gfdm.delta == 0.1
        assert config.solver_type == "gfdm"
        assert config.boundary_conditions is None

    def test_solver_type_valid_values(self):
        """Test that all solver types are valid."""
        valid_types = ["fdm", "gfdm", "gfdm_qp", "semi_lagrangian"]
        for solver_type in valid_types:
            config = HJBConfig(solver_type=solver_type)
            assert config.solver_type == solver_type

    def test_factory_fast(self):
        """Test fast configuration factory."""
        config = HJBConfig.fast()
        assert config.newton.max_iterations == 10
        assert config.gfdm.taylor_order == 1
        assert config.solver_type == "fdm"

    def test_factory_accurate(self):
        """Test accurate configuration factory."""
        config = HJBConfig.accurate()
        assert config.newton.max_iterations == 50
        assert config.gfdm.use_qp_constraints is True
        assert config.solver_type == "gfdm_qp"

    def test_boundary_conditions_dict(self):
        """Test boundary_conditions dictionary."""
        bc = {"type": "dirichlet", "value": 0.0}
        config = HJBConfig(boundary_conditions=bc)
        assert config.boundary_conditions == bc


class TestFPConfig:
    """Test FP solver configuration."""

    def test_valid_configuration(self):
        """Test creation with valid nested configurations."""
        particle_config = ParticleConfig(num_particles=10000, kde_bandwidth=0.01)

        config = FPConfig(particle=particle_config, solver_type="particle")

        assert config.particle.num_particles == 10000
        assert config.solver_type == "particle"

    def test_default_nested_configurations(self):
        """Test that nested configurations use proper defaults."""
        config = FPConfig()
        assert config.particle.num_particles == 5000
        assert config.solver_type == "fdm"
        assert config.boundary_conditions is None

    def test_solver_type_valid_values(self):
        """Test that all solver types are valid."""
        valid_types = ["fdm", "particle"]
        for solver_type in valid_types:
            config = FPConfig(solver_type=solver_type)
            assert config.solver_type == solver_type

    def test_factory_fast(self):
        """Test fast configuration factory."""
        config = FPConfig.fast()
        assert config.particle.num_particles == 1000
        assert config.solver_type == "fdm"

    def test_factory_accurate(self):
        """Test accurate configuration factory."""
        config = FPConfig.accurate()
        assert config.particle.num_particles == 10000
        assert config.solver_type == "particle"


class TestMFGSolverConfig:
    """Test master MFG solver configuration."""

    def test_valid_configuration(self):
        """Test creation with valid nested configurations."""
        picard_config = PicardConfig(max_iterations=30, tolerance=1e-5)
        hjb_config = HJBConfig(solver_type="gfdm_qp")
        fp_config = FPConfig(solver_type="fdm")

        config = MFGSolverConfig(
            picard=picard_config,
            hjb=hjb_config,
            fp=fp_config,
            warm_start=True,
            return_structured=True,
        )

        assert config.picard.max_iterations == 30
        assert config.hjb.solver_type == "gfdm_qp"
        assert config.fp.solver_type == "fdm"
        assert config.warm_start is True
        assert config.return_structured is True

    def test_default_nested_configurations(self):
        """Test that nested configurations use proper defaults."""
        config = MFGSolverConfig()

        assert config.picard.max_iterations == 20
        assert config.hjb.solver_type == "gfdm"
        assert config.fp.solver_type == "fdm"
        assert config.warm_start is False
        assert config.return_structured is False
        assert config.metadata == {}

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = MFGSolverConfig()
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert "picard" in config_dict
        assert "hjb" in config_dict
        assert "fp" in config_dict
        assert "warm_start" in config_dict

    def test_from_dict(self):
        """Test creation from dictionary."""
        config_dict = {
            "picard": {"max_iterations": 25, "tolerance": 1e-6},
            "hjb": {
                "newton": {"max_iterations": 15, "tolerance": 1e-7},
                "gfdm": {"delta": 0.08},
                "solver_type": "gfdm",
            },
            "fp": {
                "particle": {"num_particles": 8000},
                "solver_type": "particle",
            },
            "warm_start": True,
            "return_structured": False,
        }

        config = MFGSolverConfig.from_dict(config_dict)

        assert config.picard.max_iterations == 25
        assert config.hjb.newton.max_iterations == 15
        assert config.hjb.gfdm.delta == 0.08
        assert config.fp.particle.num_particles == 8000
        assert config.warm_start is True

    def test_from_dict_partial(self):
        """Test from_dict with partial configuration."""
        config_dict = {"picard": {"max_iterations": 15}, "warm_start": True}

        config = MFGSolverConfig.from_dict(config_dict)

        assert config.picard.max_iterations == 15
        assert config.warm_start is True
        # Other values should use defaults
        assert config.hjb.solver_type == "gfdm"

    def test_metadata_handling(self):
        """Test metadata dictionary handling."""
        metadata = {"experiment_name": "test", "author": "pytest"}
        config = MFGSolverConfig(metadata=metadata)

        assert config.metadata == metadata

        # Test metadata mutation
        config.metadata["version"] = "1.0"
        assert "version" in config.metadata


class TestConfigurationFactories:
    """Test configuration factory functions."""

    def test_create_default_config(self):
        """Test default configuration factory."""
        config = create_default_config()
        assert isinstance(config, MFGSolverConfig)
        assert config.picard.max_iterations == 20
        assert config.hjb.solver_type == "gfdm"

    def test_create_fast_config(self):
        """Test fast configuration factory."""
        config = create_fast_config()
        assert isinstance(config, MFGSolverConfig)
        assert config.picard.max_iterations == 10
        assert config.hjb.solver_type == "fdm"
        assert config.return_structured is True

    def test_create_accurate_config(self):
        """Test accurate configuration factory."""
        config = create_accurate_config()
        assert isinstance(config, MFGSolverConfig)
        assert config.picard.max_iterations == 50
        assert config.hjb.solver_type == "gfdm_qp"
        assert config.warm_start is True
        assert config.return_structured is True

    def test_create_research_config(self):
        """Test research configuration factory."""
        config = create_research_config()
        assert isinstance(config, MFGSolverConfig)
        assert config.picard.verbose is True
        assert config.hjb.newton.verbose is True
        assert config.metadata["purpose"] == "research"

    def test_create_production_config(self):
        """Test production configuration factory."""
        config = create_production_config()
        assert isinstance(config, MFGSolverConfig)
        assert config.picard.verbose is False
        assert config.hjb.newton.verbose is False
        assert config.metadata["purpose"] == "production"

    def test_factory_consistency(self):
        """Test that all factories produce valid configurations."""
        factories = [
            create_default_config,
            create_fast_config,
            create_accurate_config,
            create_research_config,
            create_production_config,
        ]

        for factory in factories:
            config = factory()
            assert isinstance(config, MFGSolverConfig)
            assert config.picard.max_iterations > 0
            assert config.hjb.newton.tolerance > 0


class TestLegacyParameterExtraction:
    """Test backward compatibility functions."""

    def test_extract_legacy_parameters_picard_max_iterations(self):
        """Test extraction of Picard max_iterations from various names."""
        config = create_default_config()

        # Test max_iterations parameter
        remaining = extract_legacy_parameters(config, max_iterations=50)
        assert config.picard.max_iterations == 50
        assert "max_iterations" not in remaining

    def test_extract_legacy_parameters_picard_tolerance(self):
        """Test extraction of Picard tolerance from various names."""
        config = create_default_config()

        # Test picard_tolerance parameter
        remaining = extract_legacy_parameters(config, picard_tolerance=1e-7)
        assert config.picard.tolerance == 1e-7
        assert "picard_tolerance" not in remaining

    def test_extract_legacy_parameters_deprecated_niter_max(self):
        """Test extraction of deprecated Niter_max parameter."""
        config = create_default_config()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            remaining = extract_legacy_parameters(config, Niter_max=100)

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "Niter_max" in str(w[0].message)

        assert config.picard.max_iterations == 100
        assert "Niter_max" not in remaining

    def test_extract_legacy_parameters_deprecated_l2errBoundPicard(self):
        """Test extraction of deprecated l2errBoundPicard parameter."""
        config = create_default_config()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            extract_legacy_parameters(config, l2errBoundPicard=1e-8)

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "l2errBoundPicard" in str(w[0].message)

        assert config.picard.tolerance == 1e-8

    def test_extract_legacy_parameters_newton_max_iterations(self):
        """Test extraction of Newton max_iterations."""
        config = create_default_config()

        remaining = extract_legacy_parameters(config, max_newton_iterations=40)
        assert config.hjb.newton.max_iterations == 40
        assert "max_newton_iterations" not in remaining

    def test_extract_legacy_parameters_deprecated_NiterNewton(self):
        """Test extraction of deprecated NiterNewton parameter."""
        config = create_default_config()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            extract_legacy_parameters(config, NiterNewton=60)

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)

        assert config.hjb.newton.max_iterations == 60

    def test_extract_legacy_parameters_newton_tolerance(self):
        """Test extraction of Newton tolerance."""
        config = create_default_config()

        extract_legacy_parameters(config, newton_tolerance=1e-9)
        assert config.hjb.newton.tolerance == 1e-9

    def test_extract_legacy_parameters_deprecated_l2errBoundNewton(self):
        """Test extraction of deprecated l2errBoundNewton parameter."""
        config = create_default_config()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            extract_legacy_parameters(config, l2errBoundNewton=1e-10)

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)

        assert config.hjb.newton.tolerance == 1e-10

    def test_extract_legacy_parameters_return_structured(self):
        """Test extraction of return_structured parameter."""
        config = create_default_config()

        remaining = extract_legacy_parameters(config, return_structured=True)
        assert config.return_structured is True
        assert "return_structured" not in remaining

    def test_extract_legacy_parameters_unknown_params(self):
        """Test that unknown parameters are returned."""
        config = create_default_config()

        remaining = extract_legacy_parameters(config, unknown_param=42, another_param="test")
        assert remaining["unknown_param"] == 42
        assert remaining["another_param"] == "test"

    def test_extract_legacy_parameters_mixed(self):
        """Test extraction with mix of known and unknown parameters."""
        config = create_default_config()

        remaining = extract_legacy_parameters(config, max_iterations=25, newton_tolerance=1e-8, custom_param="value")

        assert config.picard.max_iterations == 25
        assert config.hjb.newton.tolerance == 1e-8
        assert "max_iterations" not in remaining
        assert "newton_tolerance" not in remaining
        assert remaining["custom_param"] == "value"

    def test_extract_legacy_parameters_priority(self):
        """Test parameter extraction priority when multiple aliases exist."""
        config = create_default_config()

        # Test that we properly handle parameter name (no longer relevant after v0.10.2)
        remaining = extract_legacy_parameters(config, max_iterations=30)

        # max_iterations should be extracted
        assert config.picard.max_iterations == 30
        # Should not remain in kwargs
        assert "max_iterations" not in remaining
