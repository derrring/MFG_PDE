#!/usr/bin/env python3
"""
Unit tests for mfg_pde/factory/solver_factory.py

Tests solver factory patterns including:
- SolverFactoryConfig dataclass
- SolverFactory class methods
- Configuration preset validation
- Config update with kwargs
- Convenience factory functions
- Error handling and validation
- Deprecation warnings
"""

from unittest.mock import Mock, patch

import pytest

from mfg_pde.factory.solver_factory import (
    SolverFactory,
    SolverFactoryConfig,
    create_accurate_solver,
    create_amr_solver,
    create_basic_solver,
    create_fast_solver,
    create_research_solver,
    create_semi_lagrangian_solver,
    create_solver,
    create_standard_solver,
)

# ===================================================================
# Mock Problem for Testing
# ===================================================================


class MockMFGProblem:
    """Minimal mock MFG problem for testing solver factory."""

    def __init__(
        self,
        T=1.0,
        Nt=50,
        xmin=0.0,
        xmax=1.0,
        Nx=100,
        sigma=0.1,
        coupling_coefficient=0.5,
    ):
        self.T = T
        self.Nt = Nt
        self.xmin = xmin
        self.xmax = xmax
        self.Nx = Nx
        self.Dx = (xmax - xmin) / (Nx - 1) if Nx > 1 else 0.0
        self.Dt = T / (Nt - 1) if Nt > 1 else 0.0
        self.sigma = sigma
        self.coupling_coefficient = coupling_coefficient


# ===================================================================
# Test SolverFactoryConfig Dataclass
# ===================================================================


@pytest.mark.unit
def test_solver_factory_config_defaults():
    """Test SolverFactoryConfig has correct default values."""
    config = SolverFactoryConfig()

    assert config.solver_type == "fixed_point"
    assert config.config_preset == "balanced"
    assert config.return_structured is True
    assert config.warm_start is False
    assert config.custom_config is None
    assert config.solver_kwargs == {}  # __post_init__ initializes empty dict


@pytest.mark.unit
def test_solver_factory_config_custom_values():
    """Test SolverFactoryConfig with custom values."""
    custom_kwargs = {"max_iterations": 100}
    config = SolverFactoryConfig(
        solver_type="fixed_point",
        config_preset="fast",
        return_structured=False,
        warm_start=True,
        solver_kwargs=custom_kwargs,
    )

    assert config.solver_type == "fixed_point"
    assert config.config_preset == "fast"
    assert config.return_structured is False
    assert config.warm_start is True
    assert config.solver_kwargs == custom_kwargs


@pytest.mark.unit
def test_solver_factory_config_post_init():
    """Test SolverFactoryConfig __post_init__ initializes solver_kwargs."""
    config = SolverFactoryConfig(solver_kwargs=None)
    assert config.solver_kwargs == {}


# ===================================================================
# Test SolverFactory._get_config_by_preset
# ===================================================================


@pytest.mark.unit
def test_get_config_by_preset_fast():
    """Test getting 'fast' preset configuration."""
    config = SolverFactory._get_config_by_preset("fast")

    assert config is not None
    assert hasattr(config, "picard")
    assert hasattr(config, "hjb")
    assert hasattr(config, "fp")


@pytest.mark.unit
def test_get_config_by_preset_accurate():
    """Test getting 'accurate' preset configuration."""
    config = SolverFactory._get_config_by_preset("accurate")

    assert config is not None
    assert hasattr(config, "picard")
    assert hasattr(config, "hjb")
    assert hasattr(config, "fp")


@pytest.mark.unit
def test_get_config_by_preset_research():
    """Test getting 'research' preset configuration."""
    config = SolverFactory._get_config_by_preset("research")

    assert config is not None
    assert hasattr(config, "picard")
    assert hasattr(config, "hjb")
    assert hasattr(config, "fp")


@pytest.mark.unit
def test_get_config_by_preset_balanced():
    """Test getting 'balanced' preset configuration."""
    config = SolverFactory._get_config_by_preset("balanced")

    assert config is not None
    assert hasattr(config, "picard")
    assert hasattr(config, "hjb")
    assert hasattr(config, "fp")
    # Balanced has specific values
    assert config.picard.max_iterations == 25
    assert config.picard.tolerance == 1e-4
    assert config.picard.damping_factor == 0.6


@pytest.mark.unit
def test_get_config_by_preset_invalid():
    """Test getting invalid preset raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        SolverFactory._get_config_by_preset("invalid_preset")

    assert "Unknown config preset" in str(exc_info.value)
    assert "invalid_preset" in str(exc_info.value)
    assert "fast" in str(exc_info.value)
    assert "accurate" in str(exc_info.value)
    assert "research" in str(exc_info.value)
    assert "balanced" in str(exc_info.value)


# ===================================================================
# Test SolverFactory._update_config_with_kwargs
# ===================================================================


@pytest.mark.unit
def test_update_config_with_kwargs_picard():
    """Test updating Picard config parameters."""
    from mfg_pde.config.pydantic_config import create_fast_config

    base_config = create_fast_config()
    updated = SolverFactory._update_config_with_kwargs(
        base_config,
        max_picard_iterations=150,
        picard_tolerance=1e-6,
    )

    assert updated.picard.max_iterations == 150
    assert updated.picard.tolerance == 1e-6
    # Base config should be unchanged (deepcopy)
    assert base_config.picard.max_iterations != 150


@pytest.mark.unit
def test_update_config_with_kwargs_newton():
    """Test updating Newton config parameters."""
    from mfg_pde.config.pydantic_config import create_fast_config

    base_config = create_fast_config()
    updated = SolverFactory._update_config_with_kwargs(
        base_config,
        max_newton_iterations=50,
        newton_tolerance=1e-7,
    )

    assert updated.hjb.newton.max_iterations == 50
    assert updated.hjb.newton.tolerance == 1e-7


@pytest.mark.unit
def test_update_config_with_kwargs_particles():
    """Test updating particle config parameters."""
    from mfg_pde.config.pydantic_config import create_fast_config

    base_config = create_fast_config()
    updated = SolverFactory._update_config_with_kwargs(
        base_config,
        num_particles=8000,
    )

    assert updated.fp.particle.num_particles == 8000


@pytest.mark.unit
def test_update_config_with_kwargs_return_structured():
    """Test updating return_structured flag."""
    from mfg_pde.config.pydantic_config import create_fast_config

    base_config = create_fast_config()
    updated = SolverFactory._update_config_with_kwargs(
        base_config,
        return_structured=False,
    )

    assert updated.return_structured is False


# ===================================================================
# Test SolverFactory.create_solver Validation
# ===================================================================


@pytest.mark.unit
def test_create_solver_none_problem():
    """Test create_solver raises error for None problem."""
    with pytest.raises(ValueError) as exc_info:
        SolverFactory.create_solver(problem=None)

    assert "Problem cannot be None" in str(exc_info.value)
    assert "MFGProblem" in str(exc_info.value)


@pytest.mark.unit
def test_create_solver_invalid_solver_type():
    """Test create_solver raises error for invalid solver type."""
    problem = MockMFGProblem()

    with pytest.raises(ValueError) as exc_info:
        SolverFactory.create_solver(problem=problem, solver_type="invalid_type")

    assert "Unknown solver type" in str(exc_info.value)
    assert "invalid_type" in str(exc_info.value)
    assert "fixed_point" in str(exc_info.value)


@pytest.mark.unit
def test_create_solver_invalid_preset():
    """Test create_solver raises error for invalid preset."""
    problem = MockMFGProblem()

    with pytest.raises(ValueError) as exc_info:
        SolverFactory.create_solver(problem=problem, config_preset="invalid_preset")

    assert "Unknown config preset" in str(exc_info.value)


@pytest.mark.unit
def test_create_solver_fixed_point_missing_solvers():
    """Test fixed_point solver requires hjb_solver and fp_solver."""
    problem = MockMFGProblem()

    with pytest.raises(ValueError) as exc_info:
        SolverFactory.create_solver(problem=problem, solver_type="fixed_point")

    assert "requires both hjb_solver and fp_solver" in str(exc_info.value)


# ===================================================================
# Test SolverFactory.create_solver with Mocks
# ===================================================================


@pytest.mark.unit
def test_create_solver_fixed_point_with_solvers():
    """Test creating fixed_point solver with provided solvers."""
    problem = MockMFGProblem()
    mock_hjb = Mock()
    mock_fp = Mock()

    with patch("mfg_pde.factory.solver_factory.FixedPointIterator") as MockIterator:
        MockIterator.return_value = Mock()

        solver = SolverFactory.create_solver(
            problem=problem,
            solver_type="fixed_point",
            hjb_solver=mock_hjb,
            fp_solver=mock_fp,
        )

        assert solver is not None
        MockIterator.assert_called_once()
        call_kwargs = MockIterator.call_args[1]
        assert call_kwargs["problem"] == problem
        assert call_kwargs["hjb_solver"] == mock_hjb
        assert call_kwargs["fp_solver"] == mock_fp


@pytest.mark.unit
def test_create_solver_custom_config():
    """Test create_solver with custom configuration."""
    from mfg_pde.config.pydantic_config import create_research_config

    problem = MockMFGProblem()
    mock_hjb = Mock()
    mock_fp = Mock()
    custom_config = create_research_config()

    with patch("mfg_pde.factory.solver_factory.FixedPointIterator") as MockIterator:
        MockIterator.return_value = Mock()

        solver = SolverFactory.create_solver(
            problem=problem,
            solver_type="fixed_point",
            hjb_solver=mock_hjb,
            fp_solver=mock_fp,
            custom_config=custom_config,
        )

        assert solver is not None
        call_kwargs = MockIterator.call_args[1]
        # Config should be the custom one (but deepcopied and updated)
        assert call_kwargs["config"] is not None


@pytest.mark.unit
def test_create_solver_amr_warning():
    """Test create_solver warns about AMR being experimental."""
    problem = MockMFGProblem()
    mock_hjb = Mock()
    mock_fp = Mock()

    with patch("mfg_pde.factory.solver_factory.FixedPointIterator") as MockIterator:
        MockIterator.return_value = Mock()

        with pytest.warns(UserWarning, match="AMR enhancement is currently experimental"):
            solver = SolverFactory.create_solver(
                problem=problem,
                solver_type="fixed_point",
                hjb_solver=mock_hjb,
                fp_solver=mock_fp,
                enable_amr=True,
            )

        assert solver is not None


# ===================================================================
# Test Convenience Function: create_solver
# ===================================================================


@pytest.mark.unit
def test_convenience_create_solver():
    """Test convenience create_solver function."""
    problem = MockMFGProblem()
    mock_hjb = Mock()
    mock_fp = Mock()

    with patch("mfg_pde.factory.solver_factory.SolverFactory.create_solver") as mock_create:
        mock_create.return_value = Mock()

        solver = create_solver(
            problem=problem,
            solver_type="fixed_point",
            preset="fast",
            hjb_solver=mock_hjb,
            fp_solver=mock_fp,
        )

        assert solver is not None
        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["problem"] == problem
        assert call_kwargs["solver_type"] == "fixed_point"
        assert call_kwargs["config_preset"] == "fast"


# ===================================================================
# Test Convenience Function: create_basic_solver
# ===================================================================


@pytest.mark.unit
def test_create_basic_solver():
    """Test create_basic_solver creates FDM solver."""
    problem = MockMFGProblem()

    with (
        patch("mfg_pde.alg.numerical.hjb_solvers.hjb_fdm.HJBFDMSolver") as MockHJB,
        patch("mfg_pde.alg.numerical.fp_solvers.fp_fdm.FPFDMSolver") as MockFP,
        patch("mfg_pde.alg.numerical.coupling.FixedPointIterator") as MockIterator,
    ):
        MockHJB.return_value = Mock()
        MockFP.return_value = Mock()
        MockIterator.return_value = Mock()

        solver = create_basic_solver(problem=problem, damping=0.7, max_iterations=150, tolerance=1e-6)

        assert solver is not None
        MockHJB.assert_called_once_with(problem=problem)
        MockFP.assert_called_once_with(problem=problem)
        MockIterator.assert_called_once()
        # Check damping was passed
        call_kwargs = MockIterator.call_args[1]
        assert call_kwargs["damping_factor"] == 0.7


# ===================================================================
# Test Convenience Function: create_standard_solver
# ===================================================================


@pytest.mark.unit
def test_create_standard_solver():
    """Test create_standard_solver creates hybrid solver."""
    problem = MockMFGProblem()

    with (
        patch("mfg_pde.alg.numerical.hjb_solvers.hjb_fdm.HJBFDMSolver") as MockHJB,
        patch("mfg_pde.alg.numerical.fp_solvers.fp_particle.FPParticleSolver") as MockFP,
        patch("mfg_pde.factory.solver_factory.SolverFactory.create_solver") as mock_create,
    ):
        MockHJB.return_value = Mock()
        MockFP.return_value = Mock()
        mock_create.return_value = Mock()

        solver = create_standard_solver(problem=problem)

        assert solver is not None
        # Should create default HJB-FDM and FP-Particle solvers
        MockHJB.assert_called_once()
        MockFP.assert_called_once()
        # FP-Particle should have 5000 particles for standard
        call_kwargs = MockFP.call_args[1]
        assert call_kwargs["num_particles"] == 5000


# ===================================================================
# Test Convenience Function: create_fast_solver (deprecated)
# ===================================================================


@pytest.mark.unit
def test_create_fast_solver_deprecation_warning():
    """Test create_fast_solver raises deprecation warning."""
    problem = MockMFGProblem()

    with patch("mfg_pde.factory.solver_factory.create_standard_solver") as mock_standard:
        mock_standard.return_value = Mock()

        with pytest.warns(DeprecationWarning, match="create_fast_solver.*deprecated"):
            solver = create_fast_solver(problem=problem)

        assert solver is not None
        mock_standard.assert_called_once()


# ===================================================================
# Test Convenience Function: create_semi_lagrangian_solver
# ===================================================================


@pytest.mark.unit
def test_create_semi_lagrangian_solver():
    """Test create_semi_lagrangian_solver creates SL solver."""
    problem = MockMFGProblem()

    with (
        patch("mfg_pde.alg.numerical.hjb_solvers.hjb_semi_lagrangian.HJBSemiLagrangianSolver") as MockHJB,
        patch("mfg_pde.alg.numerical.fp_solvers.fp_fdm.FPFDMSolver") as MockFP,
        patch("mfg_pde.factory.solver_factory.create_standard_solver") as mock_standard,
    ):
        MockHJB.return_value = Mock()
        MockFP.return_value = Mock()
        mock_standard.return_value = Mock()

        solver = create_semi_lagrangian_solver(
            problem=problem,
            interpolation_method="cubic",
            optimization_method="brent",
        )

        assert solver is not None
        MockHJB.assert_called_once()
        call_kwargs = MockHJB.call_args[1]
        assert call_kwargs["interpolation_method"] == "cubic"
        assert call_kwargs["optimization_method"] == "brent"


@pytest.mark.unit
def test_create_semi_lagrangian_solver_particle_fp():
    """Test create_semi_lagrangian_solver with particle FP solver."""
    problem = MockMFGProblem()

    with (
        patch("mfg_pde.alg.numerical.hjb_solvers.hjb_semi_lagrangian.HJBSemiLagrangianSolver") as MockHJB,
        patch("mfg_pde.alg.numerical.fp_solvers.fp_particle.FPParticleSolver") as MockFP,
        patch("mfg_pde.factory.solver_factory.create_standard_solver") as mock_standard,
    ):
        MockHJB.return_value = Mock()
        MockFP.return_value = Mock()
        mock_standard.return_value = Mock()

        solver = create_semi_lagrangian_solver(
            problem=problem,
            fp_solver_type="particle",
        )

        assert solver is not None
        MockFP.assert_called_once_with(problem=problem)


@pytest.mark.unit
def test_create_semi_lagrangian_solver_invalid_fp_type():
    """Test create_semi_lagrangian_solver raises error for invalid FP type."""
    problem = MockMFGProblem()

    with patch("mfg_pde.alg.numerical.hjb_solvers.hjb_semi_lagrangian.HJBSemiLagrangianSolver") as MockHJB:
        MockHJB.return_value = Mock()

        with pytest.raises(ValueError) as exc_info:
            create_semi_lagrangian_solver(
                problem=problem,
                fp_solver_type="invalid_type",
            )

        assert "Unknown FP solver type" in str(exc_info.value)


# ===================================================================
# Test Convenience Function: create_accurate_solver
# ===================================================================


@pytest.mark.unit
def test_create_accurate_solver():
    """Test create_accurate_solver creates high-precision solver."""
    problem = MockMFGProblem()

    with (
        patch("mfg_pde.alg.numerical.hjb_solvers.hjb_fdm.HJBFDMSolver") as MockHJB,
        patch("mfg_pde.alg.numerical.fp_solvers.fp_particle.FPParticleSolver") as MockFP,
        patch("mfg_pde.factory.solver_factory.SolverFactory.create_solver") as mock_create,
    ):
        MockHJB.return_value = Mock()
        MockFP.return_value = Mock()
        mock_create.return_value = Mock()

        solver = create_accurate_solver(problem=problem)

        assert solver is not None
        # FP-Particle should have 10000 particles for accuracy
        call_kwargs = MockFP.call_args[1]
        assert call_kwargs["num_particles"] == 10000
        # Should use accurate preset
        create_kwargs = mock_create.call_args[1]
        assert create_kwargs["config_preset"] == "accurate"


# ===================================================================
# Test Convenience Function: create_research_solver
# ===================================================================


@pytest.mark.unit
def test_create_research_solver():
    """Test create_research_solver creates solver with monitoring."""
    problem = MockMFGProblem()

    with patch("mfg_pde.factory.solver_factory.SolverFactory.create_solver") as mock_create:
        mock_create.return_value = Mock()

        solver = create_research_solver(problem=problem)

        assert solver is not None
        # Should use research preset
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["config_preset"] == "research"


# ===================================================================
# Test Convenience Function: create_amr_solver
# ===================================================================


@pytest.mark.unit
def test_create_amr_solver():
    """Test create_amr_solver returns base solver (AMR experimental)."""
    problem = MockMFGProblem()

    with patch("mfg_pde.factory.solver_factory.SolverFactory.create_solver") as mock_create:
        mock_create.return_value = Mock()

        solver = create_amr_solver(
            problem=problem,
            base_solver_type="fixed_point",
            error_threshold=1e-5,
            max_levels=6,
        )

        assert solver is not None
        # Should use accurate preset
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["config_preset"] == "accurate"


# ===================================================================
# Test Module Exports
# ===================================================================


@pytest.mark.unit
def test_module_exports_solver_factory_class():
    """Test module exports SolverFactory class."""
    from mfg_pde.factory import solver_factory

    assert hasattr(solver_factory, "SolverFactory")
    assert isinstance(solver_factory.SolverFactory, type)


@pytest.mark.unit
def test_module_exports_convenience_functions():
    """Test module exports all convenience functions."""
    from mfg_pde.factory import solver_factory

    expected_functions = [
        "create_solver",
        "create_basic_solver",
        "create_standard_solver",
        "create_fast_solver",
        "create_semi_lagrangian_solver",
        "create_accurate_solver",
        "create_research_solver",
        "create_amr_solver",
    ]

    for func_name in expected_functions:
        assert hasattr(solver_factory, func_name)
        assert callable(getattr(solver_factory, func_name))
