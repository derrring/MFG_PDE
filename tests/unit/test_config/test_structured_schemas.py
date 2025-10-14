#!/usr/bin/env python3
"""
Unit tests for mfg_pde/config/structured_schemas.py

Tests comprehensive structured configuration schemas for OmegaConf including:
- BoundaryConditionsConfig
- InitialConditionConfig
- DomainConfig
- ProblemConfig
- NewtonConfig
- HJBConfig
- FPConfig
- SolverConfig
- LoggingConfig
- VisualizationConfig
- ExperimentConfig
- MFGConfig (top-level)
- BeachProblemConfig (specialized)
"""

import pytest

from mfg_pde.config.structured_schemas import (
    BeachProblemConfig,
    BoundaryConditionsConfig,
    DomainConfig,
    ExperimentConfig,
    FPConfig,
    HJBConfig,
    InitialConditionConfig,
    LoggingConfig,
    MFGConfig,
    NewtonConfig,
    ProblemConfig,
    SolverConfig,
    StructuredBeachConfig,
    StructuredMFGConfig,
    VisualizationConfig,
)

# ===================================================================
# Test BoundaryConditionsConfig
# ===================================================================


@pytest.mark.unit
def test_boundary_conditions_default():
    """Test BoundaryConditionsConfig default values."""
    config = BoundaryConditionsConfig()
    assert config.type == "periodic"
    assert config.left_value is None
    assert config.right_value is None


@pytest.mark.unit
def test_boundary_conditions_custom():
    """Test BoundaryConditionsConfig with custom values."""
    config = BoundaryConditionsConfig(
        type="dirichlet",
        left_value=0.0,
        right_value=1.0,
    )
    assert config.type == "dirichlet"
    assert config.left_value == 0.0
    assert config.right_value == 1.0


# ===================================================================
# Test InitialConditionConfig
# ===================================================================


@pytest.mark.unit
def test_initial_condition_default():
    """Test InitialConditionConfig default values."""
    config = InitialConditionConfig()
    assert config.type == "gaussian"
    assert isinstance(config.parameters, dict)
    assert len(config.parameters) == 0


@pytest.mark.unit
def test_initial_condition_custom():
    """Test InitialConditionConfig with custom parameters."""
    params = {"mean": 0.5, "std": 0.1}
    config = InitialConditionConfig(type="custom", parameters=params)
    assert config.type == "custom"
    assert config.parameters == params


# ===================================================================
# Test DomainConfig
# ===================================================================


@pytest.mark.unit
def test_domain_config_default():
    """Test DomainConfig default 1D domain."""
    config = DomainConfig()
    assert config.x_min == 0.0
    assert config.x_max == 1.0
    assert config.y_min is None
    assert config.y_max is None
    assert config.z_min is None
    assert config.z_max is None


@pytest.mark.unit
def test_domain_config_custom_1d():
    """Test DomainConfig custom 1D domain."""
    config = DomainConfig(x_min=-1.0, x_max=2.0)
    assert config.x_min == -1.0
    assert config.x_max == 2.0


@pytest.mark.unit
def test_domain_config_2d():
    """Test DomainConfig 2D domain."""
    config = DomainConfig(
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
    )
    assert config.y_min == 0.0
    assert config.y_max == 1.0


@pytest.mark.unit
def test_domain_config_3d():
    """Test DomainConfig 3D domain."""
    config = DomainConfig(
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
        z_min=0.0,
        z_max=1.0,
    )
    assert config.z_min == 0.0
    assert config.z_max == 1.0


# ===================================================================
# Test ProblemConfig
# ===================================================================


@pytest.mark.unit
def test_problem_config_default():
    """Test ProblemConfig default values."""
    config = ProblemConfig()
    assert config.name == "base_mfg_problem"
    assert config.type == "standard"
    assert config.T == 1.0
    assert config.Nx == 50
    assert config.Nt == 30
    assert isinstance(config.domain, DomainConfig)
    assert isinstance(config.initial_condition, InitialConditionConfig)
    assert isinstance(config.boundary_conditions, BoundaryConditionsConfig)
    assert isinstance(config.parameters, dict)


@pytest.mark.unit
def test_problem_config_custom():
    """Test ProblemConfig with custom values."""
    config = ProblemConfig(
        name="custom_problem",
        type="extended",
        T=2.0,
        Nx=100,
        Nt=50,
    )
    assert config.name == "custom_problem"
    assert config.T == 2.0
    assert config.Nx == 100


@pytest.mark.unit
def test_problem_config_nested_objects():
    """Test ProblemConfig nested configuration objects."""
    domain = DomainConfig(x_min=-1.0, x_max=1.0)
    bc = BoundaryConditionsConfig(type="neumann")
    config = ProblemConfig(domain=domain, boundary_conditions=bc)

    assert config.domain.x_min == -1.0
    assert config.boundary_conditions.type == "neumann"


# ===================================================================
# Test NewtonConfig
# ===================================================================


@pytest.mark.unit
def test_newton_config_default():
    """Test NewtonConfig default values."""
    config = NewtonConfig()
    assert config.max_iterations == 20
    assert config.tolerance == 1e-8
    assert config.line_search is True


@pytest.mark.unit
def test_newton_config_custom():
    """Test NewtonConfig with custom values."""
    config = NewtonConfig(
        max_iterations=50,
        tolerance=1e-10,
        line_search=False,
    )
    assert config.max_iterations == 50
    assert config.tolerance == 1e-10
    assert config.line_search is False


# ===================================================================
# Test HJBConfig
# ===================================================================


@pytest.mark.unit
def test_hjb_config_default():
    """Test HJBConfig default values."""
    config = HJBConfig()
    assert config.method == "gfdm"
    assert config.boundary_handling == "penalty"
    assert config.penalty_weight == 1000.0
    assert isinstance(config.newton, NewtonConfig)


@pytest.mark.unit
def test_hjb_config_custom():
    """Test HJBConfig with custom values."""
    newton = NewtonConfig(max_iterations=30)
    config = HJBConfig(
        method="fdm",
        boundary_handling="extrapolation",
        penalty_weight=500.0,
        newton=newton,
    )
    assert config.method == "fdm"
    assert config.newton.max_iterations == 30


# ===================================================================
# Test FPConfig
# ===================================================================


@pytest.mark.unit
def test_fp_config_default():
    """Test FPConfig default values."""
    config = FPConfig()
    assert config.method == "fdm"
    assert config.upwind_scheme == "central"


@pytest.mark.unit
def test_fp_config_custom():
    """Test FPConfig with custom values."""
    config = FPConfig(method="dgm", upwind_scheme="lax_friedrichs")
    assert config.method == "dgm"
    assert config.upwind_scheme == "lax_friedrichs"


# ===================================================================
# Test SolverConfig
# ===================================================================


@pytest.mark.unit
def test_solver_config_default():
    """Test SolverConfig default values."""
    config = SolverConfig()
    assert config.type == "fixed_point"
    assert config.max_iterations == 100
    assert config.tolerance == 1e-6
    assert config.damping == 0.5
    assert config.backend == "numpy"
    assert isinstance(config.hjb, HJBConfig)
    assert isinstance(config.fp, FPConfig)


@pytest.mark.unit
def test_solver_config_custom():
    """Test SolverConfig with custom values."""
    config = SolverConfig(
        type="newton",
        max_iterations=200,
        tolerance=1e-8,
        backend="torch",
    )
    assert config.type == "newton"
    assert config.backend == "torch"


@pytest.mark.unit
def test_solver_config_nested():
    """Test SolverConfig with nested configurations."""
    hjb = HJBConfig(method="spectral")
    fp = FPConfig(method="particle")
    config = SolverConfig(hjb=hjb, fp=fp)

    assert config.hjb.method == "spectral"
    assert config.fp.method == "particle"


# ===================================================================
# Test LoggingConfig
# ===================================================================


@pytest.mark.unit
def test_logging_config_default():
    """Test LoggingConfig default values."""
    config = LoggingConfig()
    assert config.level == "INFO"
    assert config.file is None


@pytest.mark.unit
def test_logging_config_custom():
    """Test LoggingConfig with custom values."""
    config = LoggingConfig(level="DEBUG", file="solver.log")
    assert config.level == "DEBUG"
    assert config.file == "solver.log"


# ===================================================================
# Test VisualizationConfig
# ===================================================================


@pytest.mark.unit
def test_visualization_config_default():
    """Test VisualizationConfig default values."""
    config = VisualizationConfig()
    assert config.enabled is True
    assert config.save_plots is True
    assert config.plot_dir == "plots"
    assert config.formats == ["png", "html"]
    assert config.dpi == 300


@pytest.mark.unit
def test_visualization_config_custom():
    """Test VisualizationConfig with custom values."""
    config = VisualizationConfig(
        enabled=False,
        save_plots=False,
        plot_dir="outputs",
        formats=["pdf", "svg"],
        dpi=600,
    )
    assert config.enabled is False
    assert config.formats == ["pdf", "svg"]
    assert config.dpi == 600


@pytest.mark.unit
def test_visualization_config_formats_mutable():
    """Test VisualizationConfig formats list is independent."""
    config1 = VisualizationConfig()
    config2 = VisualizationConfig()

    config1.formats.append("pdf")

    # config2 should have original default formats
    assert "pdf" not in config2.formats


# ===================================================================
# Test ExperimentConfig
# ===================================================================


@pytest.mark.unit
def test_experiment_config_default():
    """Test ExperimentConfig default values."""
    config = ExperimentConfig()
    assert config.name == "parameter_sweep"
    assert config.description == "Parameter sweep experiment"
    assert config.output_dir == "results"
    assert isinstance(config.logging, LoggingConfig)
    assert isinstance(config.visualization, VisualizationConfig)
    assert isinstance(config.sweeps, dict)


@pytest.mark.unit
def test_experiment_config_custom():
    """Test ExperimentConfig with custom values."""
    config = ExperimentConfig(
        name="convergence_study",
        description="Test convergence rates",
        output_dir="data",
    )
    assert config.name == "convergence_study"
    assert config.output_dir == "data"


@pytest.mark.unit
def test_experiment_config_nested():
    """Test ExperimentConfig with nested configurations."""
    logging = LoggingConfig(level="WARNING")
    viz = VisualizationConfig(enabled=False)
    config = ExperimentConfig(logging=logging, visualization=viz)

    assert config.logging.level == "WARNING"
    assert config.visualization.enabled is False


@pytest.mark.unit
def test_experiment_config_sweeps():
    """Test ExperimentConfig with parameter sweeps."""
    sweeps = {
        "Nx": [50, 100, 200],
        "tolerance": [1e-6, 1e-8, 1e-10],
    }
    config = ExperimentConfig(sweeps=sweeps)

    assert len(config.sweeps) == 2
    assert config.sweeps["Nx"] == [50, 100, 200]


# ===================================================================
# Test MFGConfig (Top-Level)
# ===================================================================


@pytest.mark.unit
def test_mfg_config_default():
    """Test MFGConfig default values."""
    config = MFGConfig()
    assert isinstance(config.problem, ProblemConfig)
    assert isinstance(config.solver, SolverConfig)
    assert isinstance(config.experiment, ExperimentConfig)


@pytest.mark.unit
def test_mfg_config_custom():
    """Test MFGConfig with custom nested configurations."""
    problem = ProblemConfig(name="custom", T=3.0)
    solver = SolverConfig(backend="jax")
    experiment = ExperimentConfig(name="test")

    config = MFGConfig(problem=problem, solver=solver, experiment=experiment)

    assert config.problem.name == "custom"
    assert config.problem.T == 3.0
    assert config.solver.backend == "jax"
    assert config.experiment.name == "test"


@pytest.mark.unit
def test_mfg_config_deep_nesting():
    """Test MFGConfig with deeply nested configurations."""
    config = MFGConfig()

    # Access deeply nested values
    assert config.solver.hjb.newton.max_iterations == 20
    assert config.problem.domain.x_max == 1.0
    assert config.experiment.visualization.dpi == 300


@pytest.mark.unit
def test_mfg_config_modification():
    """Test MFGConfig values can be modified."""
    config = MFGConfig()

    # Modify nested values
    config.problem.T = 5.0
    config.solver.max_iterations = 500
    config.experiment.name = "modified"

    assert config.problem.T == 5.0
    assert config.solver.max_iterations == 500
    assert config.experiment.name == "modified"


# ===================================================================
# Test BeachProblemConfig (Specialized)
# ===================================================================


@pytest.mark.unit
def test_beach_problem_config_default():
    """Test BeachProblemConfig specialized configuration."""
    config = BeachProblemConfig()

    assert isinstance(config.problem, ProblemConfig)
    assert config.problem.name == "towel_on_beach"
    assert config.problem.type == "spatial_competition"
    assert config.problem.T == 2.0
    assert config.problem.Nx == 80
    assert config.problem.Nt == 40


@pytest.mark.unit
def test_beach_problem_config_parameters():
    """Test BeachProblemConfig has specialized parameters."""
    config = BeachProblemConfig()

    params = config.problem.parameters
    assert "stall_position" in params
    assert "crowd_aversion" in params
    assert "noise_level" in params
    assert params["stall_position"] == 0.6
    assert params["crowd_aversion"] == 1.5
    assert params["noise_level"] == 0.1


@pytest.mark.unit
def test_beach_problem_config_has_solver():
    """Test BeachProblemConfig includes solver configuration."""
    config = BeachProblemConfig()
    assert isinstance(config.solver, SolverConfig)
    assert isinstance(config.experiment, ExperimentConfig)


# ===================================================================
# Test Type Aliases
# ===================================================================


@pytest.mark.unit
def test_type_alias_structured_mfg_config():
    """Test StructuredMFGConfig type alias."""
    assert StructuredMFGConfig is MFGConfig


@pytest.mark.unit
def test_type_alias_structured_beach_config():
    """Test StructuredBeachConfig type alias."""
    assert StructuredBeachConfig is BeachProblemConfig


@pytest.mark.unit
def test_type_alias_usage():
    """Test type aliases can be used to create instances."""
    config1 = StructuredMFGConfig()
    config2 = StructuredBeachConfig()

    assert isinstance(config1, MFGConfig)
    assert isinstance(config2, BeachProblemConfig)


# ===================================================================
# Test Dataclass Behavior
# ===================================================================


@pytest.mark.unit
def test_config_is_dataclass():
    """Test configurations are proper dataclasses."""
    import dataclasses

    assert dataclasses.is_dataclass(MFGConfig)
    assert dataclasses.is_dataclass(ProblemConfig)
    assert dataclasses.is_dataclass(SolverConfig)


@pytest.mark.unit
def test_config_equality():
    """Test configuration equality comparison."""
    config1 = ProblemConfig(name="test", T=2.0)
    config2 = ProblemConfig(name="test", T=2.0)
    config3 = ProblemConfig(name="test", T=3.0)

    assert config1 == config2
    assert config1 != config3


@pytest.mark.unit
def test_config_copy():
    """Test configuration can be copied."""
    from dataclasses import replace

    original = ProblemConfig(name="original", T=1.0)
    copied = replace(original, name="copied")

    assert copied.name == "copied"
    assert copied.T == 1.0
    assert original.name == "original"


# ===================================================================
# Test Integration Scenarios
# ===================================================================


@pytest.mark.unit
def test_complete_configuration_scenario():
    """Test creating a complete configuration for a real scenario."""
    # Create a complete MFG configuration
    config = MFGConfig(
        problem=ProblemConfig(
            name="crowding_game",
            T=1.5,
            Nx=100,
            Nt=75,
            domain=DomainConfig(x_min=-2.0, x_max=2.0),
            boundary_conditions=BoundaryConditionsConfig(type="neumann"),
        ),
        solver=SolverConfig(
            type="anderson",
            max_iterations=150,
            tolerance=1e-7,
            backend="torch",
            hjb=HJBConfig(method="weno", boundary_handling="extrapolation"),
        ),
        experiment=ExperimentConfig(
            name="crowding_experiment",
            output_dir="results/crowding",
            logging=LoggingConfig(level="DEBUG", file="crowding.log"),
        ),
    )

    # Verify all configurations are set correctly
    assert config.problem.name == "crowding_game"
    assert config.problem.T == 1.5
    assert config.problem.domain.x_min == -2.0
    assert config.solver.backend == "torch"
    assert config.solver.hjb.method == "weno"
    assert config.experiment.logging.level == "DEBUG"


@pytest.mark.unit
def test_module_docstring():
    """Test module has comprehensive docstring."""
    from mfg_pde.config import structured_schemas

    assert structured_schemas.__doc__ is not None
    assert "OmegaConf" in structured_schemas.__doc__
    assert "Issue #28" in structured_schemas.__doc__
