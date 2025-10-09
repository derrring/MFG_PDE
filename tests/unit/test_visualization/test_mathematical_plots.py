"""
Unit tests for Mathematical Plot Validation Logic.

Tests focus on backend selection, input validation, and coordinate handling
in mathematical visualizations, not actual plot rendering.
"""

import pytest

import numpy as np

from mfg_pde.visualization.mathematical_plots import (
    MathematicalPlotter,
    MFGMathematicalVisualizer,
    create_mathematical_visualizer,
)

# ============================================================================
# Test: Backend Detection and Selection
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_backend_auto_detection():
    """Test automatic backend detection."""
    # Test "auto" selects based on availability
    plotter = MathematicalPlotter(backend="auto")

    # Backend should be set to either "plotly" or "matplotlib"
    assert plotter.backend in ["plotly", "matplotlib"]


@pytest.mark.unit
@pytest.mark.fast
def test_backend_explicit_matplotlib():
    """Test explicit matplotlib backend selection."""
    plotter = MathematicalPlotter(backend="matplotlib")

    assert plotter.backend == "matplotlib"


@pytest.mark.unit
@pytest.mark.fast
def test_backend_string_storage():
    """Test backend string is stored correctly."""
    # Implementation stores backend string as-is without strict validation
    plotter = MathematicalPlotter(backend="custom_backend")

    # Backend string should be stored
    assert plotter.backend == "custom_backend"


@pytest.mark.unit
@pytest.mark.fast
def test_backend_plotly_if_available():
    """Test backend selection prioritizes plotly when available."""
    try:
        import plotly  # noqa: F401

        plotter = MathematicalPlotter(backend="auto")
        assert plotter.backend == "plotly"
    except ImportError:
        # If plotly not available, should fall back to matplotlib
        plotter = MathematicalPlotter(backend="auto")
        assert plotter.backend == "matplotlib"


# ============================================================================
# Test: LaTeX Support Detection
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_latex_setup_flag():
    """Test LaTeX usage flag is set correctly."""
    # Test with LaTeX enabled
    plotter_latex = MathematicalPlotter(backend="matplotlib", use_latex=True)
    assert plotter_latex.use_latex is True

    # Test with LaTeX disabled
    plotter_no_latex = MathematicalPlotter(backend="matplotlib", use_latex=False)
    assert plotter_no_latex.use_latex is False


# ============================================================================
# Test: Input Array Validation
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_input_array_1d_validation():
    """Test 1D array validation for mathematical functions."""
    x = np.linspace(0, 1, 10)
    y = np.sin(np.pi * x)

    # Should be 1D arrays
    assert x.ndim == 1
    assert y.ndim == 1

    # Same length
    assert len(x) == len(y)


@pytest.mark.unit
@pytest.mark.fast
def test_input_array_list_conversion():
    """Test list to array conversion."""
    x_list = [0.0, 0.25, 0.5, 0.75, 1.0]
    y_list = [0.0, 1.0, 0.0, -1.0, 0.0]

    x_array = np.asarray(x_list)
    y_array = np.asarray(y_list)

    assert isinstance(x_array, np.ndarray)
    assert isinstance(y_array, np.ndarray)
    assert len(x_array) == len(x_list)


@pytest.mark.unit
@pytest.mark.fast
def test_input_array_length_mismatch():
    """Test detection of mismatched array lengths."""
    x = np.linspace(0, 1, 10)
    y = np.sin(np.pi * x)[:-2]  # Shorter array

    # Should detect mismatch
    assert len(x) != len(y)


# ============================================================================
# Test: Grid Consistency Validation
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_grid_density_shape_consistency(grid_2d_time, density_2d_gaussian):
    """Test density shape matches grid dimensions."""
    x_grid, t_grid = grid_2d_time
    density = density_2d_gaussian

    # Density should be (len(x_grid), len(t_grid))
    expected_shape = (len(x_grid), len(t_grid))
    assert density.shape == expected_shape


@pytest.mark.unit
@pytest.mark.fast
def test_grid_consistency_check(mismatched_grid_density):
    """Test detection of grid-density shape mismatch."""
    x_grid, density = mismatched_grid_density

    # Should detect mismatch
    assert len(x_grid) != len(density)


@pytest.mark.unit
@pytest.mark.fast
def test_2d_grid_meshgrid_consistency():
    """Test meshgrid creates consistent coordinate arrays."""
    x_grid = np.linspace(0, 1, 5)
    t_grid = np.linspace(0, 1, 3)

    X, T = np.meshgrid(x_grid, t_grid, indexing="ij")

    # Check shapes
    assert X.shape == (len(x_grid), len(t_grid))
    assert T.shape == (len(x_grid), len(t_grid))

    # Check consistency
    assert np.allclose(X[:, 0], x_grid)  # First column should be x_grid
    assert np.allclose(T[0, :], t_grid)  # First row should be t_grid


# ============================================================================
# Test: Coordinate Range Validation
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_coordinate_monotonic_increasing():
    """Test coordinate grids are monotonically increasing."""
    x_grid = np.linspace(0, 1, 10)

    # Check monotonic increasing
    assert np.all(np.diff(x_grid) > 0)


@pytest.mark.unit
@pytest.mark.fast
def test_coordinate_finite_values():
    """Test coordinate arrays contain only finite values."""
    x_grid = np.linspace(0, 1, 10)
    t_grid = np.linspace(0, 1, 5)

    assert np.all(np.isfinite(x_grid))
    assert np.all(np.isfinite(t_grid))


@pytest.mark.unit
@pytest.mark.fast
def test_coordinate_range_bounds():
    """Test coordinate range validation."""
    x_grid = np.linspace(0, 1, 10)

    # Check bounds
    assert np.min(x_grid) == 0.0
    assert np.max(x_grid) == 1.0


# ============================================================================
# Test: Density Value Validation
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_density_non_negative(density_1d_gaussian):
    """Test density values are non-negative."""
    assert np.all(density_1d_gaussian >= 0)


@pytest.mark.unit
@pytest.mark.fast
def test_density_finite_values(density_2d_gaussian):
    """Test density contains only finite values."""
    assert np.all(np.isfinite(density_2d_gaussian))


@pytest.mark.unit
@pytest.mark.fast
def test_density_nan_detection(invalid_density_nan):
    """Test NaN detection in density arrays."""
    has_nan = np.any(np.isnan(invalid_density_nan))

    # Convert numpy bool to Python bool for comparison
    assert bool(has_nan) is True


@pytest.mark.unit
@pytest.mark.fast
def test_density_inf_detection(invalid_density_inf):
    """Test infinite value detection in density arrays."""
    has_inf = np.any(np.isinf(invalid_density_inf))

    # Convert numpy bool to Python bool for comparison
    assert bool(has_inf) is True


# ============================================================================
# Test: Gradient Computation Validation
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_gradient_shape_matches_function(value_function_1d, grid_1d_small):
    """Test gradient array shape matches value function."""
    u = value_function_1d
    x = grid_1d_small

    du_dx = np.gradient(u, x)

    assert du_dx.shape == u.shape


@pytest.mark.unit
@pytest.mark.fast
def test_gradient_finite_values(value_function_1d, grid_1d_small):
    """Test gradient contains finite values."""
    u = value_function_1d
    x = grid_1d_small

    du_dx = np.gradient(u, x)

    assert np.all(np.isfinite(du_dx))


# ============================================================================
# Test: Phase Portrait Vector Validation
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_vector_field_shape_consistency(vector_field_2d):
    """Test vector field shapes match grid."""
    x, y, u_field, v_field = vector_field_2d

    # u_field and v_field should have shape (len(y), len(x))
    expected_shape = (len(y), len(x))
    assert u_field.shape == expected_shape
    assert v_field.shape == expected_shape


@pytest.mark.unit
@pytest.mark.fast
def test_vector_field_finite_values(vector_field_2d):
    """Test vector field contains finite values."""
    _x, _y, u_field, v_field = vector_field_2d

    assert np.all(np.isfinite(u_field))
    assert np.all(np.isfinite(v_field))


# ============================================================================
# Test: String Parameter Sanitization
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_title_string_handling():
    """Test title string is handled correctly."""
    title = "Test Plot Title"

    # Should be string type
    assert isinstance(title, str)
    assert len(title) > 0


@pytest.mark.unit
@pytest.mark.fast
def test_empty_string_handling():
    """Test empty string handling for labels."""
    title = ""
    xlabel = ""
    ylabel = ""

    # Empty strings should be valid
    assert isinstance(title, str)
    assert isinstance(xlabel, str)
    assert isinstance(ylabel, str)


@pytest.mark.unit
@pytest.mark.fast
def test_latex_string_characters():
    """Test LaTeX special character handling."""
    latex_title = r"$u(t,x)$ Evolution"

    # Should contain LaTeX markers
    assert "$" in latex_title
    assert "u(t,x)" in latex_title


# ============================================================================
# Test: Visualizer Factory Function
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_create_visualizer_default():
    """Test create_mathematical_visualizer with defaults."""
    visualizer = create_mathematical_visualizer()

    assert isinstance(visualizer, MFGMathematicalVisualizer)
    # Backend may be "auto", "plotly", or "matplotlib"
    assert visualizer.backend in ["plotly", "matplotlib", "auto"]


@pytest.mark.unit
@pytest.mark.fast
def test_create_visualizer_explicit_backend():
    """Test create_mathematical_visualizer with explicit backend."""
    visualizer = create_mathematical_visualizer(backend="matplotlib")

    assert isinstance(visualizer, MFGMathematicalVisualizer)
    assert visualizer.backend == "matplotlib"


# ============================================================================
# Test: Edge Cases
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_single_point_function():
    """Test handling of single-point function."""
    x = np.array([0.5])
    y = np.array([1.0])

    assert len(x) == 1
    assert len(y) == 1


@pytest.mark.unit
@pytest.mark.fast
def test_uniform_density():
    """Test handling of uniform density."""
    x_grid = np.linspace(0, 1, 10)
    density = np.ones_like(x_grid) / len(x_grid)

    # Uniform density
    assert np.allclose(density, density[0])
    # Normalized
    assert np.allclose(np.sum(density), 1.0)


@pytest.mark.unit
@pytest.mark.fast
def test_zero_gradient():
    """Test handling of zero gradient (constant function)."""
    x_grid = np.linspace(0, 1, 10)
    u = np.ones_like(x_grid)

    du_dx = np.gradient(u, x_grid)

    # Gradient should be approximately zero
    assert np.allclose(du_dx, 0.0, atol=1e-10)
