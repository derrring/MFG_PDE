#!/usr/bin/env python3
"""
Unit tests for mfg_pde/config/structured_schemas.py

Tests comprehensive structured configuration schemas for OmegaConf including:
- BoundaryConditionsSchema
- InitialConditionSchema
- DomainSchema
- ProblemSchema
- NewtonSchema
- HJBSchema
- FPSchema
- SolverSchema
- LoggingSchema
- VisualizationSchema
- ExperimentSchema
- MFGSchema (top-level)
- BeachProblemSchema (specialized)
"""

import pytest

from mfg_pde.config.structured_schemas import (
    BeachProblemSchema,
    BoundaryConditionsSchema,
    DomainSchema,
    ExperimentSchema,
    FPSchema,
    HJBSchema,
    InitialConditionSchema,
    LoggingSchema,
    MFGSchema,
    NewtonSchema,
    ProblemSchema,
    SolverSchema,
    StructuredBeachConfig,
    StructuredMFGConfig,
    VisualizationSchema,
)

# ===================================================================
# Test BoundaryConditionsSchema
# ===================================================================


@pytest.mark.unit
def test_boundary_conditions_default():
    """Test BoundaryConditionsSchema default values."""
    config = BoundaryConditionsSchema()
    assert config.type == "periodic"
    assert config.left_value is None
    assert config.right_value is None


@pytest.mark.unit
def test_boundary_conditions_custom():
    """Test BoundaryConditionsSchema with custom values."""
    config = BoundaryConditionsSchema(
        type="dirichlet",
        left_value=0.0,
        right_value=1.0,
    )
    assert config.type == "dirichlet"
    assert config.left_value == 0.0
    assert config.right_value == 1.0


# ===================================================================
# Test InitialConditionSchema
# ===================================================================


@pytest.mark.unit
def test_initial_condition_default():
    """Test InitialConditionSchema default values."""
    config = InitialConditionSchema()
    assert config.type == "gaussian"
    assert isinstance(config.parameters, dict)
    assert len(config.parameters) == 0


@pytest.mark.unit
def test_initial_condition_custom():
    """Test InitialConditionSchema with custom parameters."""
    params = {"mean": 0.5, "std": 0.1}
    config = InitialConditionSchema(type="custom", parameters=params)
    assert config.type == "custom"
    assert config.parameters == params


# ===================================================================
# Test DomainSchema
# ===================================================================


@pytest.mark.unit
def test_domain_config_default():
    """Test DomainSchema default 1D domain."""
    config = DomainSchema()
    assert config.x_min == 0.0
    assert config.x_max == 1.0
    assert config.y_min is None
    assert config.y_max is None
    assert config.z_min is None
    assert config.z_max is None


@pytest.mark.unit
def test_domain_config_custom_1d():
    """Test DomainSchema custom 1D domain."""
    config = DomainSchema(x_min=-1.0, x_max=2.0)
    assert config.x_min == -1.0
    assert config.x_max == 2.0


@pytest.mark.unit
def test_domain_config_2d():
    """Test DomainSchema 2D domain."""
    config = DomainSchema(
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
    )
    assert config.y_min == 0.0
    assert config.y_max == 1.0


@pytest.mark.unit
def test_domain_config_3d():
    """Test DomainSchema 3D domain."""
    config = DomainSchema(
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
# Test ProblemSchema
# ===================================================================


@pytest.mark.unit
def test_problem_config_default():
    """Test ProblemSchema default values."""
    config = ProblemSchema()
    assert config.name == "base_mfg_problem"
    assert config.type == "standard"
    assert config.T == 1.0
    assert config.Nx == 50
    assert config.Nt == 30
    assert isinstance(config.domain, DomainSchema)
    assert isinstance(config.initial_condition, InitialConditionSchema)
    assert isinstance(config.boundary_conditions, BoundaryConditionsSchema)
    assert isinstance(config.parameters, dict)


@pytest.mark.unit
def test_problem_config_custom():
    """Test ProblemSchema with custom values."""
    config = ProblemSchema(
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
    """Test ProblemSchema nested configuration objects."""
    domain = DomainSchema(x_min=-1.0, x_max=1.0)
    bc = BoundaryConditionsSchema(type="neumann")
    config = ProblemSchema(domain=domain, boundary_conditions=bc)

    assert config.domain.x_min == -1.0
    assert config.boundary_conditions.type == "neumann"


# ===================================================================
# Test NewtonSchema
# ===================================================================


@pytest.mark.unit
def test_newton_config_default():
    """Test NewtonSchema default values."""
    config = NewtonSchema()
    assert config.max_iterations == 20
    assert config.tolerance == 1e-8
    assert config.line_search is True


@pytest.mark.unit
def test_newton_config_custom():
    """Test NewtonSchema with custom values."""
    config = NewtonSchema(
        max_iterations=50,
        tolerance=1e-10,
        line_search=False,
    )
    assert config.max_iterations == 50
    assert config.tolerance == 1e-10
    assert config.line_search is False


# ===================================================================
# Test HJBSchema
# ===================================================================


@pytest.mark.unit
def test_hjb_config_default():
    """Test HJBSchema default values."""
    config = HJBSchema()
    assert config.method == "gfdm"
    assert config.boundary_handling == "penalty"
    assert config.penalty_weight == 1000.0
    assert isinstance(config.newton, NewtonSchema)


@pytest.mark.unit
def test_hjb_config_custom():
    """Test HJBSchema with custom values."""
    newton = NewtonSchema(max_iterations=30)
    config = HJBSchema(
        method="fdm",
        boundary_handling="extrapolation",
        penalty_weight=500.0,
        newton=newton,
    )
    assert config.method == "fdm"
    assert config.newton.max_iterations == 30


# ===================================================================
# Test FPSchema
# ===================================================================


@pytest.mark.unit
def test_fp_config_default():
    """Test FPSchema default values."""
    config = FPSchema()
    assert config.method == "fdm"
    assert config.upwind_scheme == "central"


@pytest.mark.unit
def test_fp_config_custom():
    """Test FPSchema with custom values."""
    config = FPSchema(method="dgm", upwind_scheme="lax_friedrichs")
    assert config.method == "dgm"
    assert config.upwind_scheme == "lax_friedrichs"


# ===================================================================
# Test SolverSchema
# ===================================================================


@pytest.mark.unit
def test_solver_config_default():
    """Test SolverSchema default values."""
    config = SolverSchema()
    assert config.type == "fixed_point"
    assert config.max_iterations == 100
    assert config.tolerance == 1e-6
    assert config.damping == 0.5
    assert config.backend == "numpy"
    assert isinstance(config.hjb, HJBSchema)
    assert isinstance(config.fp, FPSchema)


@pytest.mark.unit
def test_solver_config_custom():
    """Test SolverSchema with custom values."""
    config = SolverSchema(
        type="newton",
        max_iterations=200,
        tolerance=1e-8,
        backend="torch",
    )
    assert config.type == "newton"
    assert config.backend == "torch"


@pytest.mark.unit
def test_solver_config_nested():
    """Test SolverSchema with nested configurations."""
    hjb = HJBSchema(method="spectral")
    fp = FPSchema(method="particle")
    config = SolverSchema(hjb=hjb, fp=fp)

    assert config.hjb.method == "spectral"
    assert config.fp.method == "particle"


# ===================================================================
# Test LoggingSchema
# ===================================================================


@pytest.mark.unit
def test_logging_config_default():
    """Test LoggingSchema default values."""
    config = LoggingSchema()
    assert config.level == "INFO"
    assert config.file is None


@pytest.mark.unit
def test_logging_config_custom():
    """Test LoggingSchema with custom values."""
    config = LoggingSchema(level="DEBUG", file="solver.log")
    assert config.level == "DEBUG"
    assert config.file == "solver.log"


# ===================================================================
# Test VisualizationSchema
# ===================================================================


@pytest.mark.unit
def test_visualization_config_default():
    """Test VisualizationSchema default values."""
    config = VisualizationSchema()
    assert config.enabled is True
    assert config.save_plots is True
    assert config.plot_dir == "plots"
    assert config.formats == ["png", "html"]
    assert config.dpi == 300


@pytest.mark.unit
def test_visualization_config_custom():
    """Test VisualizationSchema with custom values."""
    config = VisualizationSchema(
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
    """Test VisualizationSchema formats list is independent."""
    config1 = VisualizationSchema()
    config2 = VisualizationSchema()

    config1.formats.append("pdf")

    # config2 should have original default formats
    assert "pdf" not in config2.formats


# ===================================================================
# Test ExperimentSchema
# ===================================================================


@pytest.mark.unit
def test_experiment_config_default():
    """Test ExperimentSchema default values."""
    config = ExperimentSchema()
    assert config.name == "parameter_sweep"
    assert config.description == "Parameter sweep experiment"
    assert config.output_dir == "results"
    assert isinstance(config.logging, LoggingSchema)
    assert isinstance(config.visualization, VisualizationSchema)
    assert isinstance(config.sweeps, dict)


@pytest.mark.unit
def test_experiment_config_custom():
    """Test ExperimentSchema with custom values."""
    config = ExperimentSchema(
        name="convergence_study",
        description="Test convergence rates",
        output_dir="data",
    )
    assert config.name == "convergence_study"
    assert config.output_dir == "data"


@pytest.mark.unit
def test_experiment_config_nested():
    """Test ExperimentSchema with nested configurations."""
    logging = LoggingSchema(level="WARNING")
    viz = VisualizationSchema(enabled=False)
    config = ExperimentSchema(logging=logging, visualization=viz)

    assert config.logging.level == "WARNING"
    assert config.visualization.enabled is False


@pytest.mark.unit
def test_experiment_config_sweeps():
    """Test ExperimentSchema with parameter sweeps."""
    sweeps = {
        "Nx": [50, 100, 200],
        "tolerance": [1e-6, 1e-8, 1e-10],
    }
    config = ExperimentSchema(sweeps=sweeps)

    assert len(config.sweeps) == 2
    assert config.sweeps["Nx"] == [50, 100, 200]


# ===================================================================
# Test MFGSchema (Top-Level)
# ===================================================================


@pytest.mark.unit
def test_mfg_config_default():
    """Test MFGSchema default values."""
    config = MFGSchema()
    assert isinstance(config.problem, ProblemSchema)
    assert isinstance(config.solver, SolverSchema)
    assert isinstance(config.experiment, ExperimentSchema)


@pytest.mark.unit
def test_mfg_config_custom():
    """Test MFGSchema with custom nested configurations."""
    problem = ProblemSchema(name="custom", T=3.0)
    solver = SolverSchema(backend="jax")
    experiment = ExperimentSchema(name="test")

    config = MFGSchema(problem=problem, solver=solver, experiment=experiment)

    assert config.problem.name == "custom"
    assert config.problem.T == 3.0
    assert config.solver.backend == "jax"
    assert config.experiment.name == "test"


@pytest.mark.unit
def test_mfg_config_deep_nesting():
    """Test MFGSchema with deeply nested configurations."""
    config = MFGSchema()

    # Access deeply nested values
    assert config.solver.hjb.newton.max_iterations == 20
    assert config.problem.domain.x_max == 1.0
    assert config.experiment.visualization.dpi == 300


@pytest.mark.unit
def test_mfg_config_modification():
    """Test MFGSchema values can be modified."""
    config = MFGSchema()

    # Modify nested values
    config.problem.T = 5.0
    config.solver.max_iterations = 500
    config.experiment.name = "modified"

    assert config.problem.T == 5.0
    assert config.solver.max_iterations == 500
    assert config.experiment.name == "modified"


# ===================================================================
# Test BeachProblemSchema (Specialized)
# ===================================================================


@pytest.mark.unit
def test_beach_problem_config_default():
    """Test BeachProblemSchema specialized configuration."""
    config = BeachProblemSchema()

    assert isinstance(config.problem, ProblemSchema)
    assert config.problem.name == "towel_on_beach"
    assert config.problem.type == "spatial_competition"
    assert config.problem.T == 2.0
    assert config.problem.Nx == 80
    assert config.problem.Nt == 40


@pytest.mark.unit
def test_beach_problem_config_parameters():
    """Test BeachProblemSchema has specialized parameters."""
    config = BeachProblemSchema()

    params = config.problem.parameters
    assert "stall_position" in params
    assert "crowd_aversion" in params
    assert "noise_level" in params
    assert params["stall_position"] == 0.6
    assert params["crowd_aversion"] == 1.5
    assert params["noise_level"] == 0.1


@pytest.mark.unit
def test_beach_problem_config_has_solver():
    """Test BeachProblemSchema includes solver configuration."""
    config = BeachProblemSchema()
    assert isinstance(config.solver, SolverSchema)
    assert isinstance(config.experiment, ExperimentSchema)


# ===================================================================
# Test Type Aliases
# ===================================================================


@pytest.mark.unit
def test_type_alias_structured_mfg_config():
    """Test StructuredMFGConfig type alias."""
    assert StructuredMFGConfig is MFGSchema


@pytest.mark.unit
def test_type_alias_structured_beach_config():
    """Test StructuredBeachConfig type alias."""
    assert StructuredBeachConfig is BeachProblemSchema


@pytest.mark.unit
def test_type_alias_usage():
    """Test type aliases can be used to create instances."""
    config1 = StructuredMFGConfig()
    config2 = StructuredBeachConfig()

    assert isinstance(config1, MFGSchema)
    assert isinstance(config2, BeachProblemSchema)


# ===================================================================
# Test Dataclass Behavior
# ===================================================================


@pytest.mark.unit
def test_config_is_dataclass():
    """Test configurations are proper dataclasses."""
    import dataclasses

    assert dataclasses.is_dataclass(MFGSchema)
    assert dataclasses.is_dataclass(ProblemSchema)
    assert dataclasses.is_dataclass(SolverSchema)


@pytest.mark.unit
def test_config_equality():
    """Test configuration equality comparison."""
    config1 = ProblemSchema(name="test", T=2.0)
    config2 = ProblemSchema(name="test", T=2.0)
    config3 = ProblemSchema(name="test", T=3.0)

    assert config1 == config2
    assert config1 != config3


@pytest.mark.unit
def test_config_copy():
    """Test configuration can be copied."""
    from dataclasses import replace

    original = ProblemSchema(name="original", T=1.0)
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
    config = MFGSchema(
        problem=ProblemSchema(
            name="crowding_game",
            T=1.5,
            Nx=100,
            Nt=75,
            domain=DomainSchema(x_min=-2.0, x_max=2.0),
            boundary_conditions=BoundaryConditionsSchema(type="neumann"),
        ),
        solver=SolverSchema(
            type="anderson",
            max_iterations=150,
            tolerance=1e-7,
            backend="torch",
            hjb=HJBSchema(method="weno", boundary_handling="extrapolation"),
        ),
        experiment=ExperimentSchema(
            name="crowding_experiment",
            output_dir="results/crowding",
            logging=LoggingSchema(level="DEBUG", file="crowding.log"),
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
