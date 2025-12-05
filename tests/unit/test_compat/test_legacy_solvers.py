#!/usr/bin/env python3
"""
Unit tests for mfg_pde/compat/legacy_solvers.py

Tests legacy solver compatibility wrappers including:
- LegacyMFGSolver base class
- Deprecated solver aliases (EnhancedParticleCollocationSolver, FixedPointIterator, etc.)
- Deprecation warnings
- Config conversion (_convert_config)
- Problem type inference (_infer_problem_type)
"""

# Import directly to avoid legacy_config import errors
import sys
import warnings
from pathlib import Path

import pytest

# Add parent directory to path to import module directly
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "mfg_pde"))

try:
    from mfg_pde.compat.legacy_solvers import (
        AdaptiveMFGSolver,
        DebugMFGSolver,
        EnhancedParticleCollocationSolver,
        FixedPointIterator,
        LegacyMFGSolver,
    )
except (TypeError, ImportError) as e:
    # If there's an import error from legacy_config, skip these tests
    pytest.skip(f"Cannot import legacy_solvers due to dependency issue: {e}", allow_module_level=True)

# ===================================================================
# Test LegacyMFGSolver Base Class
# ===================================================================


@pytest.mark.unit
def test_legacy_mfg_solver_init_warning():
    """Test LegacyMFGSolver initialization raises deprecation warning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        solver = LegacyMFGSolver()

        assert solver is not None  # Check instantiation succeeds
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "deprecated" in str(w[0].message).lower()
        assert "problem.solve()" in str(w[0].message)


@pytest.mark.unit
def test_legacy_mfg_solver_init_with_config():
    """Test LegacyMFGSolver initialization with config dict."""
    config = {"max_iterations": 100, "tolerance": 1e-6}

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        solver = LegacyMFGSolver(config=config)

        assert solver.config == config
        assert solver.config["max_iterations"] == 100


@pytest.mark.unit
def test_legacy_mfg_solver_default_config():
    """Test LegacyMFGSolver default empty config."""
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        solver = LegacyMFGSolver()

        assert solver.config == {}
        assert isinstance(solver.config, dict)


@pytest.mark.unit
def test_legacy_mfg_solver_convert_config():
    """Test _convert_config method."""
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        solver = LegacyMFGSolver()

        legacy_config = {
            "max_iterations": 100,
            "tolerance": 1e-6,
            "damping_parameter": 0.5,
            "backend": "numpy",
            "unknown_param": 42,  # Should be ignored
        }

        new_config = solver._convert_config(legacy_config)

        assert new_config["max_iterations"] == 100
        assert new_config["tolerance"] == 1e-6
        assert new_config["damping"] == 0.5
        assert new_config["backend"] == "numpy"
        assert "unknown_param" not in new_config


@pytest.mark.unit
def test_legacy_mfg_solver_convert_config_partial():
    """Test _convert_config with partial config."""
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        solver = LegacyMFGSolver()

        legacy_config = {"max_iterations": 50}

        new_config = solver._convert_config(legacy_config)

        assert new_config["max_iterations"] == 50
        assert "tolerance" not in new_config
        assert "damping" not in new_config


@pytest.mark.unit
def test_legacy_mfg_solver_convert_config_empty():
    """Test _convert_config with empty config."""
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        solver = LegacyMFGSolver()

        new_config = solver._convert_config({})

        assert new_config == {}


@pytest.mark.unit
def test_legacy_mfg_solver_infer_problem_type_attribute():
    """Test _infer_problem_type with problem_type attribute."""

    class MockProblem:
        problem_type = "crowd_dynamics"

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        solver = LegacyMFGSolver()

        problem = MockProblem()
        problem_type = solver._infer_problem_type(problem)

        assert problem_type == "crowd_dynamics"


@pytest.mark.unit
def test_legacy_mfg_solver_infer_problem_type_crowd():
    """Test _infer_problem_type for crowd dynamics problems."""

    class CrowdDynamicsProblem:
        pass

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        solver = LegacyMFGSolver()

        problem = CrowdDynamicsProblem()
        problem_type = solver._infer_problem_type(problem)

        assert problem_type == "crowd_dynamics"


@pytest.mark.unit
def test_legacy_mfg_solver_infer_problem_type_portfolio():
    """Test _infer_problem_type for portfolio problems."""

    class PortfolioOptimizationProblem:
        pass

    class MertonProblem:
        pass

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        solver = LegacyMFGSolver()

        problem1 = PortfolioOptimizationProblem()
        problem2 = MertonProblem()

        assert solver._infer_problem_type(problem1) == "portfolio_optimization"
        assert solver._infer_problem_type(problem2) == "portfolio_optimization"


@pytest.mark.unit
def test_legacy_mfg_solver_infer_problem_type_traffic():
    """Test _infer_problem_type for traffic flow problems."""

    class TrafficFlowProblem:
        pass

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        solver = LegacyMFGSolver()

        problem = TrafficFlowProblem()
        problem_type = solver._infer_problem_type(problem)

        assert problem_type == "traffic_flow"


@pytest.mark.unit
def test_legacy_mfg_solver_infer_problem_type_epidemic():
    """Test _infer_problem_type for epidemic problems."""

    class EpidemicProblem:
        pass

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        solver = LegacyMFGSolver()

        problem = EpidemicProblem()
        problem_type = solver._infer_problem_type(problem)

        assert problem_type == "epidemic"


@pytest.mark.unit
def test_legacy_mfg_solver_infer_problem_type_unknown():
    """Test _infer_problem_type for unknown problem types."""

    class UnknownProblem:
        pass

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        solver = LegacyMFGSolver()

        problem = UnknownProblem()
        problem_type = solver._infer_problem_type(problem)

        assert problem_type is None


# ===================================================================
# Test Deprecated Solver Aliases
# ===================================================================


@pytest.mark.unit
def test_enhanced_particle_collocation_solver_deprecation():
    """Test EnhancedParticleCollocationSolver deprecation warning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        solver = EnhancedParticleCollocationSolver()

        assert solver is not None  # Check instantiation succeeds
        # Should have deprecation warning
        assert len(w) >= 1
        assert any(issubclass(warning.category, DeprecationWarning) for warning in w)
        assert any("problem.solve()" in str(warning.message) for warning in w)


@pytest.mark.unit
def test_enhanced_particle_collocation_solver_inheritance():
    """Test EnhancedParticleCollocationSolver inherits from LegacyMFGSolver."""
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        solver = EnhancedParticleCollocationSolver()

        assert isinstance(solver, LegacyMFGSolver)
        assert hasattr(solver, "solve")
        assert hasattr(solver, "_convert_config")


@pytest.mark.unit
def test_fixed_point_iterator_deprecation():
    """Test FixedPointIterator deprecation warning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        solver = FixedPointIterator()

        assert solver is not None  # Check instantiation succeeds
        # Should have deprecation warning
        assert len(w) >= 1
        assert any(issubclass(warning.category, DeprecationWarning) for warning in w)
        assert any("problem.solve()" in str(warning.message) for warning in w)


@pytest.mark.unit
def test_fixed_point_iterator_inheritance():
    """Test FixedPointIterator inherits from LegacyMFGSolver."""
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        solver = FixedPointIterator()

        assert isinstance(solver, LegacyMFGSolver)


@pytest.mark.unit
def test_adaptive_mfg_solver_deprecation():
    """Test AdaptiveMFGSolver deprecation warning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        solver = AdaptiveMFGSolver()

        assert solver is not None  # Check instantiation succeeds
        # Should have deprecation warning
        assert len(w) >= 1
        assert any(issubclass(warning.category, DeprecationWarning) for warning in w)
        assert any("problem.solve()" in str(warning.message) for warning in w)


@pytest.mark.unit
def test_adaptive_mfg_solver_inheritance():
    """Test AdaptiveMFGSolver inherits from LegacyMFGSolver."""
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        solver = AdaptiveMFGSolver()

        assert isinstance(solver, LegacyMFGSolver)


@pytest.mark.unit
def test_debug_mfg_solver_deprecation():
    """Test DebugMFGSolver deprecation warning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        solver = DebugMFGSolver()

        assert solver is not None  # Check instantiation succeeds
        # Should have deprecation warning
        assert len(w) >= 1
        assert any(issubclass(warning.category, DeprecationWarning) for warning in w)
        assert any("problem.solve()" in str(warning.message) for warning in w)


@pytest.mark.unit
def test_debug_mfg_solver_inheritance():
    """Test DebugMFGSolver inherits from LegacyMFGSolver."""
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        solver = DebugMFGSolver()

        assert isinstance(solver, LegacyMFGSolver)


# ===================================================================
# Test All Aliases Have Proper Deprecation
# ===================================================================


@pytest.mark.unit
def test_all_deprecated_solvers_emit_warnings():
    """Test that all deprecated solver classes emit deprecation warnings."""
    deprecated_classes = [
        EnhancedParticleCollocationSolver,
        FixedPointIterator,
        AdaptiveMFGSolver,
        DebugMFGSolver,
    ]

    for solver_class in deprecated_classes:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            solver = solver_class()

            assert solver is not None  # Check instantiation succeeds
            # Each class should emit at least one deprecation warning
            assert len(w) >= 1
            assert any(issubclass(warning.category, DeprecationWarning) for warning in w), (
                f"{solver_class.__name__} did not emit DeprecationWarning"
            )


# ===================================================================
# Test Module Exports
# ===================================================================


@pytest.mark.unit
def test_module_exports():
    """Test all legacy solvers are importable."""
    from mfg_pde.compat import legacy_solvers

    assert hasattr(legacy_solvers, "LegacyMFGSolver")
    assert hasattr(legacy_solvers, "EnhancedParticleCollocationSolver")
    assert hasattr(legacy_solvers, "FixedPointIterator")
    assert hasattr(legacy_solvers, "AdaptiveMFGSolver")
    assert hasattr(legacy_solvers, "DebugMFGSolver")


@pytest.mark.unit
def test_module_docstring():
    """Test module has docstring."""
    from mfg_pde.compat import legacy_solvers

    assert legacy_solvers.__doc__ is not None
    assert "Legacy solver compatibility" in legacy_solvers.__doc__
