#!/usr/bin/env python3
"""
Unit tests for Simple API (mfg_pde.simple)

Tests the high-level user-facing API for MFG problem solving.
"""

import pytest

from mfg_pde.simple import (
    create_mfg_problem,
    get_available_problems,
    get_config_recommendation,
    solve_mfg,
    solve_mfg_auto,
    solve_mfg_smart,
    suggest_problem_setup,
    validate_problem_parameters,
)

# ============================================================================
# Test: Problem Type Discovery
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_get_available_problems():
    """Test getting list of available problem types."""
    problems = get_available_problems()

    assert isinstance(problems, dict)
    assert len(problems) > 0

    # Check that common problem types are available
    expected_types = ["crowd_dynamics", "portfolio_optimization", "traffic_flow", "epidemic"]
    for problem_type in expected_types:
        assert problem_type in problems, f"Problem type '{problem_type}' should be available"

    # Check that each problem has metadata
    for _problem_type, metadata in problems.items():
        assert isinstance(metadata, dict)
        assert "description" in metadata or "name" in metadata


@pytest.mark.unit
@pytest.mark.fast
def test_suggest_problem_setup():
    """Test getting suggested setup for problem types."""
    # Test with known problem type
    setup = suggest_problem_setup("crowd_dynamics")

    assert isinstance(setup, dict)
    # Should have some configuration recommendations
    assert len(setup) > 0


@pytest.mark.unit
@pytest.mark.fast
def test_suggest_problem_setup_all_types():
    """Test suggesting setup for all available problem types."""
    problems = get_available_problems()

    for problem_type in problems:
        setup = suggest_problem_setup(problem_type)
        assert isinstance(setup, dict), f"Setup for '{problem_type}' should be a dict"


# ============================================================================
# Test: Config Recommendation
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_get_config_recommendation_crowd_dynamics():
    """Test config recommendation for crowd dynamics."""
    config = get_config_recommendation("crowd_dynamics")

    assert isinstance(config, dict)
    # Should have some configuration parameters
    assert len(config) > 0


@pytest.mark.unit
@pytest.mark.fast
def test_get_config_recommendation_portfolio():
    """Test config recommendation for portfolio optimization."""
    config = get_config_recommendation("portfolio_optimization")

    assert isinstance(config, dict)
    assert len(config) > 0


@pytest.mark.unit
@pytest.mark.fast
def test_get_config_recommendation_with_kwargs():
    """Test config recommendation respects kwargs."""
    config = get_config_recommendation("crowd_dynamics", domain_size=5.0, time_horizon=2.0)

    assert isinstance(config, dict)


# ============================================================================
# Test: Parameter Validation
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_validate_problem_parameters_valid():
    """Test validation with valid parameters."""
    params = validate_problem_parameters("crowd_dynamics", domain_size=1.0, time_horizon=1.0)

    assert isinstance(params, dict)


@pytest.mark.unit
@pytest.mark.fast
def test_validate_problem_parameters_all_types():
    """Test validation works for all problem types."""
    problem_types = ["crowd_dynamics", "portfolio_optimization", "traffic_flow", "epidemic"]

    for problem_type in problem_types:
        params = validate_problem_parameters(problem_type)
        assert isinstance(params, dict), f"Validation for '{problem_type}' should return dict"


# ============================================================================
# Test: Problem Creation
# ============================================================================


@pytest.mark.unit
def test_create_mfg_problem_crowd_dynamics():
    """Test creating crowd dynamics problem."""
    problem = create_mfg_problem("crowd_dynamics", domain_size=1.0, time_horizon=1.0)

    assert problem is not None
    assert hasattr(problem, "T")
    assert hasattr(problem, "g")
    assert hasattr(problem, "rho0")


@pytest.mark.unit
def test_create_mfg_problem_portfolio():
    """Test creating portfolio optimization problem."""
    problem = create_mfg_problem("portfolio_optimization", domain_size=2.0, time_horizon=1.5)

    assert problem is not None
    assert hasattr(problem, "T")
    assert pytest.approx(1.5, rel=0.01) == problem.T


@pytest.mark.unit
def test_create_mfg_problem_traffic():
    """Test creating traffic flow problem."""
    problem = create_mfg_problem("traffic_flow")

    assert problem is not None
    assert hasattr(problem, "T")


@pytest.mark.unit
def test_create_mfg_problem_epidemic():
    """Test creating epidemic problem."""
    problem = create_mfg_problem("epidemic")

    assert problem is not None
    assert hasattr(problem, "T")


@pytest.mark.unit
@pytest.mark.fast
def test_create_mfg_problem_invalid_type():
    """Test creating problem with invalid type raises error."""
    with pytest.raises((ValueError, KeyError)):
        create_mfg_problem("invalid_problem_type")


# ============================================================================
# Test: Solve MFG - Basic
# ============================================================================


@pytest.mark.unit
@pytest.mark.slow
def test_solve_mfg_crowd_dynamics():
    """Test solving crowd dynamics problem."""
    result = solve_mfg("crowd_dynamics", domain_size=1.0, time_horizon=0.5, accuracy="fast")

    assert result is not None
    # Result should have solution arrays
    assert hasattr(result, "u") or hasattr(result, "value_function")
    assert hasattr(result, "m") or hasattr(result, "density")


@pytest.mark.unit
@pytest.mark.slow
def test_solve_mfg_with_fast_flag():
    """Test solving with fast flag enabled."""
    result = solve_mfg("crowd_dynamics", domain_size=0.5, time_horizon=0.5, fast=True)

    assert result is not None


@pytest.mark.unit
@pytest.mark.slow
def test_solve_mfg_with_verbose_flag():
    """Test solving with verbose flag enabled."""
    result = solve_mfg("crowd_dynamics", domain_size=0.5, time_horizon=0.5, verbose=True)

    assert result is not None


@pytest.mark.unit
@pytest.mark.slow
def test_solve_mfg_accuracy_levels():
    """Test all accuracy levels work."""
    accuracy_levels = ["fast", "balanced", "high"]

    for accuracy in accuracy_levels:
        result = solve_mfg("crowd_dynamics", domain_size=0.5, time_horizon=0.5, accuracy=accuracy)

        assert result is not None, f"Accuracy level '{accuracy}' should work"


# ============================================================================
# Test: Solve MFG Smart
# ============================================================================


@pytest.mark.unit
@pytest.mark.slow
def test_solve_mfg_smart():
    """Test smart solver with automatic configuration."""
    result = solve_mfg_smart("crowd_dynamics", domain_size=0.5, time_horizon=0.5)

    assert result is not None


@pytest.mark.unit
@pytest.mark.slow
def test_solve_mfg_smart_auto_tuning():
    """Test smart solver with auto-tuning enabled."""
    result = solve_mfg_smart("crowd_dynamics", domain_size=0.5, time_horizon=0.5, auto_tune=True)

    assert result is not None


# ============================================================================
# Test: Solve MFG Auto
# ============================================================================


@pytest.mark.unit
@pytest.mark.slow
def test_solve_mfg_auto():
    """Test automatic solver selection."""
    result = solve_mfg_auto("crowd_dynamics", domain_size=0.5, time_horizon=0.5)

    assert result is not None


# ============================================================================
# Test: Parameter Combinations
# ============================================================================


@pytest.mark.unit
def test_solve_mfg_custom_domain_size():
    """Test solving with custom domain size."""
    result = solve_mfg("crowd_dynamics", domain_size=2.0, time_horizon=0.5, accuracy="fast")

    assert result is not None


@pytest.mark.unit
def test_solve_mfg_custom_time_horizon():
    """Test solving with custom time horizon."""
    result = solve_mfg("crowd_dynamics", domain_size=1.0, time_horizon=2.0, accuracy="fast")

    assert result is not None


@pytest.mark.unit
@pytest.mark.fast
def test_solve_mfg_portfolio_type():
    """Test solving portfolio optimization problem."""
    # Portfolio problems may have different requirements
    result = solve_mfg("portfolio_optimization", domain_size=1.0, time_horizon=0.5, accuracy="fast")

    assert result is not None


# ============================================================================
# Test: Error Handling
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_solve_mfg_invalid_problem_type():
    """Test solving with invalid problem type raises error."""
    with pytest.raises((ValueError, KeyError)):
        solve_mfg("nonexistent_problem_type")


@pytest.mark.unit
@pytest.mark.fast
def test_solve_mfg_invalid_accuracy_level():
    """Test solving with invalid accuracy level."""
    # Should either work with default or raise descriptive error
    try:
        result = solve_mfg("crowd_dynamics", accuracy="invalid_accuracy")
        # If it doesn't raise, should fall back to default
        assert result is not None
    except ValueError:
        # If it raises ValueError, that's expected for invalid accuracy
        pass


@pytest.mark.unit
@pytest.mark.fast
def test_create_mfg_problem_negative_domain_size():
    """Test creating problem with negative domain size."""
    # Should either work (taking absolute value) or raise error
    try:
        problem = create_mfg_problem("crowd_dynamics", domain_size=-1.0)
        # If it works, domain should be positive
        assert problem.xmax > problem.xmin
    except ValueError:
        # Negative domain size properly rejected
        pass


@pytest.mark.unit
@pytest.mark.fast
def test_create_mfg_problem_zero_time_horizon():
    """Test creating problem with zero time horizon."""
    with pytest.raises((ValueError, AssertionError)):
        create_mfg_problem("crowd_dynamics", time_horizon=0.0)


# ============================================================================
# Test: Result Properties
# ============================================================================


@pytest.mark.unit
def test_solve_mfg_result_has_required_attributes():
    """Test that result has expected attributes."""
    result = solve_mfg("crowd_dynamics", domain_size=0.5, time_horizon=0.5, accuracy="fast")

    # Should have either new-style or legacy attributes
    has_u = hasattr(result, "u") or hasattr(result, "value_function")
    has_m = hasattr(result, "m") or hasattr(result, "density")

    assert has_u, "Result should have value function (u or value_function)"
    assert has_m, "Result should have density (m or density)"


# ============================================================================
# Test: Multiple Problem Types
# ============================================================================


@pytest.mark.unit
@pytest.mark.slow
@pytest.mark.parametrize(
    "problem_type",
    [
        "crowd_dynamics",
        "portfolio_optimization",
        "traffic_flow",
        "epidemic",
    ],
)
def test_solve_mfg_all_problem_types(problem_type):
    """Test solving all available problem types."""
    result = solve_mfg(problem_type, domain_size=0.5, time_horizon=0.5, accuracy="fast")

    assert result is not None
    # Each problem type should return a valid result
    has_solution = (
        hasattr(result, "u") or hasattr(result, "value_function") or hasattr(result, "m") or hasattr(result, "density")
    )
    assert has_solution, f"Problem type '{problem_type}' should return solution"


# ============================================================================
# Test: Problem Creation with Custom Parameters
# ============================================================================


@pytest.mark.unit
def test_create_mfg_problem_with_kwargs():
    """Test problem creation accepts custom kwargs."""
    # Should not crash with extra parameters
    problem = create_mfg_problem("crowd_dynamics", domain_size=1.0, custom_param=0.5)

    assert problem is not None
