"""
Unit tests for Coordinate Transformation Logic.

Tests focus on coordinate extraction, transformation, and mapping logic
used in visualization, not the actual rendering.
"""

import pytest

import numpy as np

# ============================================================================
# Test: Meshgrid Creation and Extraction
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_meshgrid_extraction_basic(grid_2d_time):
    """Test extraction of X, Y from 1D grids via meshgrid."""
    x_grid, t_grid = grid_2d_time

    X, T = np.meshgrid(x_grid, t_grid, indexing="ij")

    # Check shapes
    assert X.shape == (len(x_grid), len(t_grid))
    assert T.shape == (len(x_grid), len(t_grid))

    # Check that X varies along first axis
    assert np.allclose(X[:, 0], x_grid)
    assert np.allclose(X[:, 1], x_grid)

    # Check that T varies along second axis
    assert np.allclose(T[0, :], t_grid)
    assert np.allclose(T[1, :], t_grid)


@pytest.mark.unit
@pytest.mark.fast
def test_meshgrid_indexing_consistency():
    """Test meshgrid indexing='ij' vs 'xy' behavior."""
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([10.0, 20.0])

    # Matrix indexing (ij) - matches (x, y) ordering
    X_ij, Y_ij = np.meshgrid(x, y, indexing="ij")

    assert X_ij.shape == (3, 2)  # (len(x), len(y))
    assert Y_ij.shape == (3, 2)

    # Cartesian indexing (xy) - swaps dimensions
    X_xy, Y_xy = np.meshgrid(x, y, indexing="xy")

    assert X_xy.shape == (2, 3)  # (len(y), len(x)) - swapped!
    assert Y_xy.shape == (2, 3)


# ============================================================================
# Test: Coordinate Scaling and Normalization
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_coordinate_normalization_to_unit_interval():
    """Test normalization of coordinates to [0, 1]."""
    coords = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

    # Normalize to [0, 1]
    min_coord = np.min(coords)
    max_coord = np.max(coords)
    normalized = (coords - min_coord) / (max_coord - min_coord)

    assert np.allclose(normalized[0], 0.0)
    assert np.allclose(normalized[-1], 1.0)
    assert np.all((normalized >= 0.0) & (normalized <= 1.0))


@pytest.mark.unit
@pytest.mark.fast
def test_coordinate_scaling_to_range():
    """Test scaling coordinates to specific range."""
    coords = np.linspace(0, 1, 10)

    # Scale to [5, 15]
    target_min, target_max = 5.0, 15.0
    scaled = coords * (target_max - target_min) + target_min

    assert np.allclose(scaled[0], target_min)
    assert np.allclose(scaled[-1], target_max)
    assert np.all((scaled >= target_min) & (scaled <= target_max))


@pytest.mark.unit
@pytest.mark.fast
def test_coordinate_centering():
    """Test centering coordinates around origin."""
    coords = np.array([5.0, 10.0, 15.0, 20.0, 25.0])

    # Center around 0
    mean_coord = np.mean(coords)
    centered = coords - mean_coord

    assert np.allclose(np.mean(centered), 0.0, atol=1e-10)
    assert np.allclose(np.sum(centered), 0.0, atol=1e-10)


# ============================================================================
# Test: Spatial Index Mapping
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_coordinate_to_index_conversion():
    """Test mapping from coordinate value to grid index."""
    x_grid = np.linspace(0, 1, 11)  # 11 points from 0 to 1

    # Find index of coordinate value 0.5
    target_coord = 0.5
    # Using np.argmin to find closest grid point
    idx = np.argmin(np.abs(x_grid - target_coord))

    assert idx == 5  # Middle index
    assert np.allclose(x_grid[idx], 0.5)


@pytest.mark.unit
@pytest.mark.fast
def test_index_to_coordinate_conversion():
    """Test mapping from grid index to coordinate value."""
    x_grid = np.linspace(0, 10, 21)  # 21 points from 0 to 10

    # Get coordinate at index 10
    idx = 10
    coord = x_grid[idx]

    assert np.allclose(coord, 5.0)  # Middle value


@pytest.mark.unit
@pytest.mark.fast
def test_bilinear_index_mapping():
    """Test 2D index to coordinate mapping."""
    x_grid = np.linspace(0, 1, 5)
    y_grid = np.linspace(0, 2, 3)

    # Map 2D index (i, j) to coordinates
    i, j = 2, 1

    x_coord = x_grid[i]
    y_coord = y_grid[j]

    assert np.allclose(x_coord, 0.5)  # Middle x
    assert np.allclose(y_coord, 1.0)  # Middle y


# ============================================================================
# Test: Bounding Box Computation
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_bounding_box_from_coordinates():
    """Test bounding box calculation from coordinate arrays."""
    x_coords = np.array([1.0, 5.0, 3.0, 7.0, 2.0])
    y_coords = np.array([2.0, 6.0, 1.0, 8.0, 4.0])

    # Compute bounding box
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)

    assert x_min == 1.0
    assert x_max == 7.0
    assert y_min == 1.0
    assert y_max == 8.0


@pytest.mark.unit
@pytest.mark.fast
def test_bounding_box_with_margin():
    """Test bounding box with added margin."""
    coords = np.array([0.0, 10.0])

    min_coord, max_coord = np.min(coords), np.max(coords)

    # Add 10% margin
    margin = 0.1 * (max_coord - min_coord)
    min_with_margin = min_coord - margin
    max_with_margin = max_coord + margin

    assert min_with_margin == -1.0
    assert max_with_margin == 11.0


# ============================================================================
# Test: Aspect Ratio Calculation
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_aspect_ratio_from_data():
    """Test aspect ratio computation from coordinate ranges."""
    x_range = (0.0, 10.0)
    y_range = (0.0, 5.0)

    # Compute aspect ratio
    x_span = x_range[1] - x_range[0]
    y_span = y_range[1] - y_range[0]
    aspect_ratio = x_span / y_span if y_span > 0 else 1.0

    assert aspect_ratio == 2.0  # x is twice as wide as y


@pytest.mark.unit
@pytest.mark.fast
def test_equal_aspect_enforcement():
    """Test enforcement of equal aspect ratio."""
    x_range = (0.0, 10.0)
    y_range = (0.0, 5.0)

    # To enforce equal aspect, expand the smaller dimension
    x_span = x_range[1] - x_range[0]
    y_span = y_range[1] - y_range[0]

    if x_span > y_span:
        # Expand y to match x
        y_center = (y_range[0] + y_range[1]) / 2
        new_y_half_span = x_span / 2
        new_y_range = (y_center - new_y_half_span, y_center + new_y_half_span)

        assert new_y_range[1] - new_y_range[0] == x_span
        assert np.allclose(new_y_range, (-2.5, 7.5))


# ============================================================================
# Test: Network Layout Coordinates
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_circular_layout_coordinates():
    """Test circular layout coordinate generation."""
    n_nodes = 4

    # Generate circular layout
    angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    radius = 1.0

    x_coords = radius * np.cos(angles)
    y_coords = radius * np.sin(angles)

    # Verify positions are on unit circle
    distances = np.sqrt(x_coords**2 + y_coords**2)
    assert np.allclose(distances, radius)


@pytest.mark.unit
@pytest.mark.fast
def test_grid_layout_coordinates():
    """Test grid layout coordinate generation."""
    n_rows, n_cols = 3, 4

    # Generate grid positions
    x_coords = []
    y_coords = []

    for i in range(n_rows):
        for j in range(n_cols):
            x_coords.append(j)
            y_coords.append(i)

    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)

    assert len(x_coords) == n_rows * n_cols
    assert len(y_coords) == n_rows * n_cols

    # Check ranges
    assert np.min(x_coords) == 0
    assert np.max(x_coords) == n_cols - 1
    assert np.min(y_coords) == 0
    assert np.max(y_coords) == n_rows - 1


# ============================================================================
# Test: Edge Midpoint Calculation
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_edge_midpoint_calculation(small_network_positions):
    """Test calculation of edge midpoints for labels."""
    positions = small_network_positions

    # Calculate midpoint between nodes 0 and 1
    node_i, node_j = 0, 1
    pos_i = positions[node_i]
    pos_j = positions[node_j]

    midpoint = (pos_i + pos_j) / 2

    # Verify midpoint properties
    assert midpoint.shape == (2,)
    assert np.allclose(midpoint, [0.5, 0.0])  # Between (0,0) and (1,0)


@pytest.mark.unit
@pytest.mark.fast
def test_edge_midpoint_multiple_edges():
    """Test midpoint calculation for multiple edges."""
    positions = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 2.0]])

    edges = [(0, 1), (1, 2), (2, 0)]

    midpoints = []
    for i, j in edges:
        midpoint = (positions[i] + positions[j]) / 2
        midpoints.append(midpoint)

    midpoints = np.array(midpoints)

    assert midpoints.shape == (3, 2)
    assert np.allclose(midpoints[0], [1.0, 0.0])  # Edge 0-1
    assert np.allclose(midpoints[1], [1.5, 1.0])  # Edge 1-2
    assert np.allclose(midpoints[2], [0.5, 1.0])  # Edge 2-0


# ============================================================================
# Test: Coordinate Transformation Chains
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_coordinate_transformation_chain():
    """Test chained coordinate transformations."""
    # Original data coordinates
    data_coords = np.array([100.0, 200.0, 300.0, 400.0, 500.0])

    # Step 1: Normalize to [0, 1]
    min_val, max_val = np.min(data_coords), np.max(data_coords)
    normalized = (data_coords - min_val) / (max_val - min_val)

    # Step 2: Scale to plot coordinates [0, 800]
    plot_coords = normalized * 800

    # Verify transformation
    assert np.allclose(plot_coords[0], 0.0)
    assert np.allclose(plot_coords[-1], 800.0)
    assert plot_coords[2] == 400.0  # Middle value


@pytest.mark.unit
@pytest.mark.fast
def test_inverse_coordinate_transformation():
    """Test inverse transformation from plot to data coordinates."""
    # Plot coordinates
    plot_coords = np.array([0.0, 200.0, 400.0, 600.0, 800.0])

    # Known data range
    data_min, data_max = 100.0, 500.0

    # Inverse transform: plot -> normalized -> data
    normalized = plot_coords / 800.0
    data_coords = normalized * (data_max - data_min) + data_min

    # Verify inverse transformation
    assert np.allclose(data_coords[0], data_min)
    assert np.allclose(data_coords[-1], data_max)
    assert np.allclose(data_coords[2], 300.0)  # Middle value
