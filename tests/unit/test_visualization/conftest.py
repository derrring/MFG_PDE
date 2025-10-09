"""
Shared fixtures for visualization tests.

Provides common test data structures including networks, grids, and densities
for testing visualization module data preparation logic.
"""

import pytest

import numpy as np

try:
    import scipy.sparse as sp

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ============================================================================
# Network Test Data
# ============================================================================


@pytest.fixture
def small_network_adjacency():
    """Create small test network adjacency matrix (3 nodes, fully connected)."""
    adjacency = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    return adjacency


@pytest.fixture
def small_network_positions():
    """Create node positions for small test network (triangle layout)."""
    positions = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
    return positions


@pytest.fixture
def small_network_data(small_network_adjacency, small_network_positions):
    """Create minimal NetworkData-like object for testing."""

    class MockNetworkData:
        def __init__(self, adj, pos):
            self.num_nodes = adj.shape[0]
            self.num_edges = int(np.sum(adj > 0) / 2)  # Undirected
            self.adjacency_matrix = adj
            self.node_positions = pos

    return MockNetworkData(small_network_adjacency, small_network_positions)


@pytest.fixture
def line_network_data():
    """Create simple line network (4 nodes in a row)."""

    class MockNetworkData:
        def __init__(self):
            self.num_nodes = 4
            self.num_edges = 3
            self.adjacency_matrix = np.array(
                [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]]
            )
            self.node_positions = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])

    return MockNetworkData()


@pytest.fixture
def sparse_network_adjacency():
    """Create sparse adjacency matrix if scipy available."""
    if not SCIPY_AVAILABLE:
        pytest.skip("scipy not available")

    # 5 node network with only 3 edges
    data = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    row = [0, 1, 1, 2, 2, 3]
    col = [1, 0, 2, 1, 3, 2]
    return sp.csr_matrix((data, (row, col)), shape=(5, 5))


# ============================================================================
# Grid and Density Test Data
# ============================================================================


@pytest.fixture
def grid_1d_small():
    """Create small 1D grid for quick tests."""
    return np.linspace(0, 1, 10)


@pytest.fixture
def grid_1d_medium():
    """Create medium 1D grid for standard tests."""
    return np.linspace(0, 1, 50)


@pytest.fixture
def grid_2d_time():
    """Create small 2D space-time grid."""
    x_grid = np.linspace(0, 1, 10)
    t_grid = np.linspace(0, 1, 5)
    return x_grid, t_grid


@pytest.fixture
def density_1d_gaussian(grid_1d_small):
    """Create normalized Gaussian density on 1D grid."""
    x = grid_1d_small
    density = np.exp(-10 * (x - 0.5) ** 2)
    # Use trapezoid instead of deprecated trapz
    density = density / np.trapezoid(density, x)
    return density


@pytest.fixture
def density_2d_gaussian(grid_2d_time):
    """Create 2D Gaussian density evolving in time."""
    x_grid, t_grid = grid_2d_time
    X, T = np.meshgrid(x_grid, t_grid, indexing="ij")

    # Gaussian that spreads over time
    sigma_t = 0.1 + 0.2 * T
    density = np.exp(-0.5 * (X - 0.5) ** 2 / sigma_t**2) / (sigma_t * np.sqrt(2 * np.pi))

    return density


@pytest.fixture
def value_function_1d(grid_1d_small):
    """Create simple value function on 1D grid."""
    x = grid_1d_small
    return np.sin(np.pi * x) * (1 - x)


@pytest.fixture
def value_function_2d(grid_2d_time):
    """Create 2D value function decaying in time."""
    x_grid, t_grid = grid_2d_time
    X, T = np.meshgrid(x_grid, t_grid, indexing="ij")
    return np.sin(np.pi * X) * np.exp(-2 * T)


# ============================================================================
# Network Evolution Data
# ============================================================================


@pytest.fixture
def network_density_evolution(small_network_data):
    """Create density evolution on network nodes."""
    num_nodes = small_network_data.num_nodes
    num_timesteps = 5

    # Initialize with concentration at node 0, diffuse over time
    density = np.zeros((num_nodes, num_timesteps))
    density[0, 0] = 1.0

    for t in range(1, num_timesteps):
        # Simple diffusion: average with neighbors
        new_density = density[:, t - 1].copy()
        adj = small_network_data.adjacency_matrix
        for i in range(num_nodes):
            neighbors = np.where(adj[i] > 0)[0]
            if len(neighbors) > 0:
                new_density[i] = 0.5 * density[i, t - 1] + 0.5 * np.mean(density[neighbors, t - 1])
        density[:, t] = new_density / np.sum(new_density)  # Normalize

    return density


@pytest.fixture
def network_value_evolution(small_network_data):
    """Create value function evolution on network nodes."""
    num_nodes = small_network_data.num_nodes
    num_timesteps = 5

    # Simple decaying value function
    value = np.zeros((num_nodes, num_timesteps))
    for i in range(num_nodes):
        for t in range(num_timesteps):
            value[i, t] = (num_nodes - i) * np.exp(-0.5 * t)

    return value


# ============================================================================
# Vector Field Data
# ============================================================================


@pytest.fixture
def vector_field_2d():
    """Create 2D vector field for phase portrait tests."""
    x = np.linspace(-2, 2, 8)
    y = np.linspace(-2, 2, 8)
    X, Y = np.meshgrid(x, y)

    # Simple rotating vector field
    u_field = -Y
    v_field = X

    return x, y, u_field, v_field


# ============================================================================
# Validation Test Data (Edge Cases)
# ============================================================================


@pytest.fixture
def invalid_density_negative():
    """Create invalid density with negative values."""
    return np.array([-0.1, 0.5, 0.3, 0.2, 0.1])


@pytest.fixture
def invalid_density_nan():
    """Create invalid density with NaN values."""
    return np.array([0.2, np.nan, 0.3, 0.2, 0.1])


@pytest.fixture
def invalid_density_inf():
    """Create invalid density with infinite values."""
    return np.array([0.2, 0.3, np.inf, 0.2, 0.1])


@pytest.fixture
def mismatched_grid_density():
    """Create grid and density with mismatched shapes."""
    x_grid = np.linspace(0, 1, 10)
    density = np.ones(15)  # Wrong size!
    return x_grid, density


# ============================================================================
# Mock MFG Problem
# ============================================================================


@pytest.fixture
def mock_network_mfg_problem(small_network_data):
    """Create minimal NetworkMFGProblem-like object for testing."""

    class MockNetworkMFGProblem:
        def __init__(self, network_data):
            self.network_data = network_data
            self.num_nodes = network_data.num_nodes
            self.T = 1.0
            self.Nt = 10

    return MockNetworkMFGProblem(small_network_data)
