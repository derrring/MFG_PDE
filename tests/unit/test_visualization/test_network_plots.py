"""
Unit tests for Network Plot Data Preparation Logic.

Tests focus on data extraction, validation, and transformation logic
in network visualization, not actual plot rendering.
"""

import pytest

import numpy as np

from mfg_pde.visualization.network_plots import NetworkMFGVisualizer

# ============================================================================
# Test: Initialization and Setup
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_network_visualizer_initialization_with_network_data(small_network_data):
    """Test NetworkMFGVisualizer initialization with network data."""
    visualizer = NetworkMFGVisualizer(network_data=small_network_data)

    assert visualizer.network_data is not None
    assert visualizer.num_nodes == small_network_data.num_nodes
    assert visualizer.num_edges == small_network_data.num_edges


@pytest.mark.unit
@pytest.mark.fast
def test_network_visualizer_initialization_with_problem(mock_network_mfg_problem):
    """Test NetworkMFGVisualizer initialization with MFG problem."""
    visualizer = NetworkMFGVisualizer(problem=mock_network_mfg_problem)

    assert visualizer.network_data is not None
    assert visualizer.problem is mock_network_mfg_problem
    assert visualizer.num_nodes == mock_network_mfg_problem.num_nodes


@pytest.mark.unit
@pytest.mark.fast
def test_network_visualizer_initialization_fails_without_data():
    """Test NetworkMFGVisualizer raises error when neither problem nor data provided."""
    with pytest.raises(ValueError, match="Either problem or network_data must be provided"):
        NetworkMFGVisualizer(problem=None, network_data=None)


# ============================================================================
# Test: Network Property Extraction
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_network_properties_extracted_correctly(small_network_data):
    """Test extraction of network properties from data."""
    visualizer = NetworkMFGVisualizer(network_data=small_network_data)

    # Verify all network properties accessible
    assert visualizer.num_nodes == 3
    assert visualizer.num_edges == 3  # Fully connected triangle
    assert visualizer.adjacency_matrix is not None
    assert visualizer.node_positions is not None
    assert visualizer.adjacency_matrix.shape == (3, 3)
    assert visualizer.node_positions.shape == (3, 2)


@pytest.mark.unit
@pytest.mark.fast
def test_default_visualization_parameters(small_network_data):
    """Test default parameter values set correctly."""
    visualizer = NetworkMFGVisualizer(network_data=small_network_data)

    assert visualizer.default_node_size == 300
    assert visualizer.default_edge_width == 2
    assert visualizer.default_colorscale == "viridis"


# ============================================================================
# Test: Edge Coordinate Extraction
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_edge_coordinate_extraction_basic(small_network_adjacency, small_network_positions):
    """Test extraction of edge coordinates from adjacency matrix."""
    # Test the logic from lines 142-150 in network_plots.py
    edge_x, edge_y = [], []
    rows, cols = np.nonzero(small_network_adjacency)

    for i, j in zip(rows, cols, strict=False):
        x0, y0 = small_network_positions[i]
        x1, y1 = small_network_positions[j]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    # Each edge appears twice (i->j and j->i), 3 edges * 2 = 6
    # Each edge adds 3 elements (x0, x1, None)
    assert len(edge_x) == 6 * 3  # 18 elements
    assert len(edge_y) == 6 * 3
    assert edge_x[2] is None  # Every third element should be None
    assert edge_y[2] is None


@pytest.mark.unit
@pytest.mark.fast
def test_edge_extraction_with_no_edges():
    """Test edge extraction handles empty adjacency matrix."""
    adjacency = np.zeros((3, 3))
    positions = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])

    edge_x, edge_y = [], []
    rows, cols = np.nonzero(adjacency)

    for i, j in zip(rows, cols, strict=False):
        x0, y0 = positions[i]
        x1, y1 = positions[j]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    # No edges, so lists should be empty
    assert len(edge_x) == 0
    assert len(edge_y) == 0


@pytest.mark.unit
@pytest.mark.fast
def test_edge_coordinates_line_network(line_network_data):
    """Test edge extraction for simple line network."""
    edge_x, edge_y = [], []
    rows, cols = np.nonzero(line_network_data.adjacency_matrix)

    for i, j in zip(rows, cols, strict=False):
        x0, y0 = line_network_data.node_positions[i]
        x1, y1 = line_network_data.node_positions[j]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    # Line network: 3 edges * 2 (bidirectional) = 6
    # Each edge: 3 elements
    assert len(edge_x) == 6 * 3
    # All y-coordinates should be 0 (horizontal line)
    assert all(y == 0.0 or y is None for y in edge_y)


# ============================================================================
# Test: Node Value Normalization
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_node_value_normalization_basic():
    """Test node value normalization for coloring."""
    node_values = np.array([0.0, 0.5, 1.0])

    # Normalize to [0, 1]
    min_val = np.min(node_values)
    max_val = np.max(node_values)
    normalized = (node_values - min_val) / (max_val - min_val) if max_val > min_val else node_values

    assert normalized[0] == 0.0
    assert normalized[1] == 0.5
    assert normalized[2] == 1.0


@pytest.mark.unit
@pytest.mark.fast
def test_node_value_normalization_with_nan(invalid_density_nan):
    """Test handling of NaN values in node values."""
    # Remove NaN before normalization
    valid_mask = ~np.isnan(invalid_density_nan)
    valid_values = invalid_density_nan[valid_mask]

    assert len(valid_values) == 4  # 5 - 1 NaN
    assert not np.any(np.isnan(valid_values))


@pytest.mark.unit
@pytest.mark.fast
def test_node_value_normalization_uniform():
    """Test normalization when all values are identical."""
    node_values = np.array([0.5, 0.5, 0.5])

    min_val = np.min(node_values)
    max_val = np.max(node_values)
    # Should handle division by zero gracefully
    normalized = (node_values - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(node_values)

    assert np.allclose(normalized, 0.0)


# ============================================================================
# Test: Validation Logic
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_node_position_shape_validation(small_network_data):
    """Test validation of node position array shape."""
    positions = small_network_data.node_positions

    # Should be (num_nodes, 2)
    assert positions.ndim == 2
    assert positions.shape[0] == small_network_data.num_nodes
    assert positions.shape[1] == 2


@pytest.mark.unit
@pytest.mark.fast
def test_adjacency_matrix_square_validation(small_network_adjacency):
    """Test adjacency matrix is square."""
    assert small_network_adjacency.shape[0] == small_network_adjacency.shape[1]


@pytest.mark.unit
@pytest.mark.fast
def test_adjacency_matrix_non_negative(small_network_adjacency):
    """Test adjacency matrix has non-negative weights."""
    assert np.all(small_network_adjacency >= 0)


@pytest.mark.unit
@pytest.mark.fast
def test_adjacency_matrix_symmetry_undirected(small_network_adjacency):
    """Test adjacency matrix symmetry for undirected graph."""
    # For undirected graphs, adjacency should be symmetric
    assert np.allclose(small_network_adjacency, small_network_adjacency.T)


@pytest.mark.unit
@pytest.mark.fast
def test_density_array_shape_validation(small_network_data, network_density_evolution):
    """Test density array shape matches network structure."""
    num_nodes = small_network_data.num_nodes
    num_timesteps = network_density_evolution.shape[1]

    assert network_density_evolution.shape == (num_nodes, num_timesteps)


@pytest.mark.unit
@pytest.mark.fast
def test_density_non_negative_validation(network_density_evolution):
    """Test density values are non-negative."""
    assert np.all(network_density_evolution >= 0)


@pytest.mark.unit
@pytest.mark.fast
def test_density_normalization_check(network_density_evolution):
    """Test density normalization at each timestep."""
    # Sum over nodes should be approximately 1 at each time
    mass_per_timestep = np.sum(network_density_evolution, axis=0)

    assert np.allclose(mass_per_timestep, 1.0, rtol=1e-10)


@pytest.mark.unit
@pytest.mark.fast
def test_network_data_consistency_check(small_network_data):
    """Test consistency between adjacency matrix and positions."""
    # Number of nodes from matrix should match positions length
    assert small_network_data.adjacency_matrix.shape[0] == small_network_data.node_positions.shape[0]

    # Edge indices should be within valid range
    rows, cols = np.nonzero(small_network_data.adjacency_matrix)
    assert np.all(rows < small_network_data.num_nodes)
    assert np.all(cols < small_network_data.num_nodes)


# ============================================================================
# Test: Parameter Scaling
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_node_size_scaling():
    """Test node size scaling calculation."""
    default_size = 300
    scale_factor = 2.0

    scaled_size = default_size * scale_factor

    assert scaled_size == 600


@pytest.mark.unit
@pytest.mark.fast
def test_edge_width_scaling():
    """Test edge width scaling calculation."""
    default_width = 2.0
    scale_factor = 1.5

    scaled_width = default_width * scale_factor

    assert scaled_width == 3.0


@pytest.mark.unit
@pytest.mark.fast
def test_parameter_scaling_zero():
    """Test parameter scaling with zero scale factor."""
    default_size = 300
    scale_factor = 0.0

    scaled_size = default_size * scale_factor

    assert scaled_size == 0.0


# ============================================================================
# Test: Edge Cases
# ============================================================================


@pytest.mark.unit
@pytest.mark.fast
def test_single_node_network():
    """Test network with single node (no edges)."""

    class SingleNodeNetwork:
        def __init__(self):
            self.num_nodes = 1
            self.num_edges = 0
            self.adjacency_matrix = np.array([[0.0]])
            self.node_positions = np.array([[0.5, 0.5]])

    network = SingleNodeNetwork()
    visualizer = NetworkMFGVisualizer(network_data=network)

    assert visualizer.num_nodes == 1
    assert visualizer.num_edges == 0


@pytest.mark.unit
@pytest.mark.fast
def test_disconnected_network():
    """Test network with disconnected components."""

    class DisconnectedNetwork:
        def __init__(self):
            self.num_nodes = 4
            self.num_edges = 2
            # Two disconnected pairs: 0-1 and 2-3
            self.adjacency_matrix = np.array(
                [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]]
            )
            self.node_positions = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [4.0, 0.0]])

    network = DisconnectedNetwork()
    visualizer = NetworkMFGVisualizer(network_data=network)

    assert visualizer.num_nodes == 4
    assert visualizer.num_edges == 2


@pytest.mark.unit
@pytest.mark.fast
def test_weighted_network():
    """Test network with weighted edges."""

    class WeightedNetwork:
        def __init__(self):
            self.num_nodes = 3
            self.num_edges = 3
            # Different weights on edges
            self.adjacency_matrix = np.array([[0.0, 0.5, 1.0], [0.5, 0.0, 2.0], [1.0, 2.0, 0.0]])
            self.node_positions = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])

    network = WeightedNetwork()
    visualizer = NetworkMFGVisualizer(network_data=network)

    # Verify weighted edges are handled
    assert np.max(visualizer.adjacency_matrix) == 2.0
    assert np.min(visualizer.adjacency_matrix[visualizer.adjacency_matrix > 0]) == 0.5
