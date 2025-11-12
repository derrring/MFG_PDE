"""Tests for mfg_pde.visualization.enhanced_network_plots module."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

import numpy as np


@pytest.fixture
def mock_network_data():
    """Mock NetworkData for testing."""
    network = Mock()
    network.num_nodes = 5
    network.adjacency_matrix = np.array(
        [
            [0, 1, 1, 0, 0],
            [1, 0, 1, 1, 0],
            [1, 1, 0, 1, 1],
            [0, 1, 1, 0, 1],
            [0, 0, 1, 1, 0],
        ]
    )

    # Node positions in 2D
    angles = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    network.node_positions = np.column_stack([np.cos(angles), np.sin(angles)])

    return network


@pytest.fixture
def sample_trajectories():
    """Sample trajectory data."""
    return [
        [0, 1, 2, 4],  # Trajectory 1
        [1, 2, 3],  # Trajectory 2
        [0, 2, 4],  # Trajectory 3
    ]


@pytest.fixture
def sample_solution_data():
    """Sample U and M arrays."""
    Nt, num_nodes = 10, 5
    U = np.random.rand(Nt, num_nodes)
    M = np.random.rand(Nt, num_nodes)
    return U, M


@pytest.fixture
def sample_velocity_field():
    """Sample velocity field data."""
    Nt, num_nodes, velocity_dim = 10, 5, 2
    return np.random.rand(Nt, num_nodes, velocity_dim)


class TestEnhancedNetworkMFGVisualizerInitialization:
    """Test EnhancedNetworkMFGVisualizer initialization."""

    @patch("mfg_pde.visualization.enhanced_network_plots.NetworkMFGVisualizer.__init__")
    def test_initialization_with_network_data(self, mock_super_init, mock_network_data):
        """Test initialization with network data."""
        from mfg_pde.visualization.enhanced_network_plots import EnhancedNetworkMFGVisualizer

        mock_super_init.return_value = None

        viz = EnhancedNetworkMFGVisualizer(network_data=mock_network_data)

        assert hasattr(viz, "trajectory_colors")
        assert hasattr(viz, "flow_arrow_scale")
        assert hasattr(viz, "velocity_field_resolution")
        assert viz.flow_arrow_scale == 1.0
        assert viz.velocity_field_resolution == 20

    @patch("mfg_pde.visualization.enhanced_network_plots.NetworkMFGVisualizer.__init__")
    def test_initialization_without_data(self, mock_super_init):
        """Test initialization without data."""
        from mfg_pde.visualization.enhanced_network_plots import EnhancedNetworkMFGVisualizer

        mock_super_init.return_value = None

        viz = EnhancedNetworkMFGVisualizer()

        assert len(viz.trajectory_colors) == 6
        assert "red" in viz.trajectory_colors


class TestLagrangianTrajectories:
    """Test Lagrangian trajectory plotting."""

    @patch("mfg_pde.visualization.enhanced_network_plots.PLOTLY_AVAILABLE", True)
    @patch("mfg_pde.visualization.enhanced_network_plots.NetworkMFGVisualizer.__init__")
    def test_plot_trajectories_interactive(self, mock_super_init, mock_network_data, sample_trajectories):
        """Test interactive trajectory plotting with Plotly."""
        from mfg_pde.visualization.enhanced_network_plots import EnhancedNetworkMFGVisualizer

        mock_super_init.return_value = None

        viz = EnhancedNetworkMFGVisualizer(network_data=mock_network_data)
        viz.node_positions = mock_network_data.node_positions
        viz.adjacency_matrix = mock_network_data.adjacency_matrix
        viz.num_nodes = mock_network_data.num_nodes

        fig = viz.plot_lagrangian_trajectories(sample_trajectories, title="Test Trajectories", interactive=True)

        # Functional assertion: verify a figure object was returned
        assert fig is not None
        assert hasattr(fig, "data") or hasattr(fig, "layout")

    @patch("mfg_pde.visualization.enhanced_network_plots.PLOTLY_AVAILABLE", False)
    @patch("mfg_pde.visualization.enhanced_network_plots.plt")
    @patch("mfg_pde.visualization.enhanced_network_plots.NetworkMFGVisualizer.__init__")
    def test_plot_trajectories_matplotlib(self, mock_super_init, mock_plt, mock_network_data, sample_trajectories):
        """Test trajectory plotting with matplotlib."""
        from mfg_pde.visualization.enhanced_network_plots import EnhancedNetworkMFGVisualizer

        mock_super_init.return_value = None
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        viz = EnhancedNetworkMFGVisualizer(network_data=mock_network_data)
        viz.node_positions = mock_network_data.node_positions
        viz.adjacency_matrix = mock_network_data.adjacency_matrix
        viz.num_nodes = mock_network_data.num_nodes

        fig = viz.plot_lagrangian_trajectories(sample_trajectories, title="Test Trajectories", interactive=False)

        assert fig == mock_fig
        assert mock_plt.subplots.called
        assert mock_ax.plot.called

    @patch("mfg_pde.visualization.enhanced_network_plots.PLOTLY_AVAILABLE", True)
    @patch("mfg_pde.visualization.enhanced_network_plots.NetworkMFGVisualizer.__init__")
    def test_plot_trajectories_with_solution_data(
        self, mock_super_init, mock_network_data, sample_trajectories, sample_solution_data
    ):
        """Test trajectory plotting with U and M data."""
        from mfg_pde.visualization.enhanced_network_plots import EnhancedNetworkMFGVisualizer

        mock_super_init.return_value = None

        viz = EnhancedNetworkMFGVisualizer(network_data=mock_network_data)
        viz.node_positions = mock_network_data.node_positions
        viz.adjacency_matrix = mock_network_data.adjacency_matrix
        viz.num_nodes = mock_network_data.num_nodes

        U, M = sample_solution_data

        fig = viz.plot_lagrangian_trajectories(sample_trajectories, U=U, M=M, interactive=True)

        # Functional assertion: verify a figure was returned
        assert fig is not None

    @patch("mfg_pde.visualization.enhanced_network_plots.plt")
    @patch("mfg_pde.visualization.enhanced_network_plots.NetworkMFGVisualizer.__init__")
    def test_plot_trajectories_empty_trajectory(self, mock_super_init, mock_plt, mock_network_data):
        """Test handling of empty trajectories."""
        from mfg_pde.visualization.enhanced_network_plots import EnhancedNetworkMFGVisualizer

        mock_super_init.return_value = None
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        viz = EnhancedNetworkMFGVisualizer(network_data=mock_network_data)
        viz.node_positions = mock_network_data.node_positions
        viz.adjacency_matrix = mock_network_data.adjacency_matrix
        viz.num_nodes = mock_network_data.num_nodes

        # Empty trajectory
        viz.plot_lagrangian_trajectories([[]], interactive=False)

        # Should still create figure
        assert mock_plt.subplots.called


class TestVelocityField:
    """Test velocity field visualization."""

    @patch("mfg_pde.visualization.enhanced_network_plots.PLOTLY_AVAILABLE", True)
    @patch("mfg_pde.visualization.enhanced_network_plots.NetworkMFGVisualizer.__init__")
    def test_plot_velocity_field_interactive(self, mock_super_init, mock_network_data, sample_velocity_field):
        """Test interactive velocity field plotting."""
        from mfg_pde.visualization.enhanced_network_plots import EnhancedNetworkMFGVisualizer

        mock_super_init.return_value = None

        viz = EnhancedNetworkMFGVisualizer(network_data=mock_network_data)
        viz.node_positions = mock_network_data.node_positions
        viz.num_nodes = mock_network_data.num_nodes

        fig = viz.plot_velocity_field(sample_velocity_field, title="Test Velocity Field", time_idx=0, interactive=True)

        # Functional assertion: verify a figure was returned
        assert fig is not None
        assert hasattr(fig, "data") or hasattr(fig, "layout")

    @patch("mfg_pde.visualization.enhanced_network_plots.plt")
    @patch("mfg_pde.visualization.enhanced_network_plots.NetworkMFGVisualizer.__init__")
    def test_plot_velocity_field_matplotlib(self, mock_super_init, mock_plt, mock_network_data, sample_velocity_field):
        """Test velocity field plotting with matplotlib."""
        from mfg_pde.visualization.enhanced_network_plots import EnhancedNetworkMFGVisualizer

        mock_super_init.return_value = None
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        viz = EnhancedNetworkMFGVisualizer(network_data=mock_network_data)
        viz.node_positions = mock_network_data.node_positions
        viz.adjacency_matrix = mock_network_data.adjacency_matrix
        viz.num_nodes = mock_network_data.num_nodes

        fig = viz.plot_velocity_field(sample_velocity_field, title="Test Velocity Field", time_idx=0, interactive=False)

        assert fig == mock_fig
        assert mock_plt.subplots.called

    @patch("mfg_pde.visualization.enhanced_network_plots.plt")
    @patch("mfg_pde.visualization.enhanced_network_plots.NetworkMFGVisualizer.__init__")
    def test_plot_velocity_field_with_density(
        self, mock_super_init, mock_plt, mock_network_data, sample_velocity_field, sample_solution_data
    ):
        """Test velocity field with density background."""
        from mfg_pde.visualization.enhanced_network_plots import EnhancedNetworkMFGVisualizer

        mock_super_init.return_value = None
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        viz = EnhancedNetworkMFGVisualizer(network_data=mock_network_data)
        viz.node_positions = mock_network_data.node_positions
        viz.adjacency_matrix = mock_network_data.adjacency_matrix
        viz.num_nodes = mock_network_data.num_nodes

        _U, M = sample_solution_data

        viz.plot_velocity_field(sample_velocity_field, M=M, time_idx=2, interactive=False)

        assert mock_plt.subplots.called

    @patch("mfg_pde.visualization.enhanced_network_plots.plt")
    @patch("mfg_pde.visualization.enhanced_network_plots.NetworkMFGVisualizer.__init__")
    def test_plot_velocity_field_1d(self, mock_super_init, mock_plt, mock_network_data):
        """Test velocity field with 1D velocities."""
        from mfg_pde.visualization.enhanced_network_plots import EnhancedNetworkMFGVisualizer

        mock_super_init.return_value = None
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        viz = EnhancedNetworkMFGVisualizer(network_data=mock_network_data)
        viz.node_positions = mock_network_data.node_positions
        viz.adjacency_matrix = mock_network_data.adjacency_matrix
        viz.num_nodes = 5

        # 1D velocity field
        velocity_field_1d = np.random.rand(10, 5, 1)

        viz.plot_velocity_field(velocity_field_1d, interactive=False)

        assert mock_plt.subplots.called


class TestSchemeComparison:
    """Test discretization scheme comparison."""

    @patch("mfg_pde.visualization.enhanced_network_plots.PLOTLY_AVAILABLE", True)
    @patch("mfg_pde.visualization.enhanced_network_plots.NetworkMFGVisualizer.__init__")
    def test_plot_scheme_comparison_interactive(self, mock_super_init, mock_network_data):
        """Test interactive scheme comparison."""
        from mfg_pde.visualization.enhanced_network_plots import EnhancedNetworkMFGVisualizer

        mock_super_init.return_value = None

        viz = EnhancedNetworkMFGVisualizer(network_data=mock_network_data)
        viz.num_nodes = 5

        # Multiple solutions
        Nt, num_nodes = 10, 5
        solutions = {
            "FDM": (np.random.rand(Nt, num_nodes), np.random.rand(Nt, num_nodes)),
            "FEM": (np.random.rand(Nt, num_nodes), np.random.rand(Nt, num_nodes)),
            "GFDM": (np.random.rand(Nt, num_nodes), np.random.rand(Nt, num_nodes)),
        }
        times = np.linspace(0, 1, Nt)

        fig = viz.plot_scheme_comparison(solutions, times, selected_nodes=[0, 1, 2], interactive=True)

        # Functional assertion
        assert fig is not None
        assert hasattr(fig, "data") or hasattr(fig, "layout")

    @patch("mfg_pde.visualization.enhanced_network_plots.plt")
    @patch("matplotlib.cm")
    @patch("mfg_pde.visualization.enhanced_network_plots.NetworkMFGVisualizer.__init__")
    def test_plot_scheme_comparison_matplotlib(self, mock_super_init, mock_cm, mock_plt, mock_network_data):
        """Test scheme comparison with matplotlib."""
        from mfg_pde.visualization.enhanced_network_plots import EnhancedNetworkMFGVisualizer

        mock_super_init.return_value = None
        mock_fig = Mock()
        mock_axes = np.array([[Mock(), Mock()], [Mock(), Mock()]])
        mock_plt.subplots.return_value = (mock_fig, mock_axes)
        mock_cm.get_cmap.return_value = lambda x: np.array([[1, 0, 0, 1]] * len(x))

        viz = EnhancedNetworkMFGVisualizer(network_data=mock_network_data)
        viz.num_nodes = 5

        Nt, num_nodes = 10, 5
        solutions = {
            "FDM": (np.random.rand(Nt, num_nodes), np.random.rand(Nt, num_nodes)),
            "FEM": (np.random.rand(Nt, num_nodes), np.random.rand(Nt, num_nodes)),
        }
        times = np.linspace(0, 1, Nt)

        fig = viz.plot_scheme_comparison(solutions, times, selected_nodes=[0, 1], interactive=False)

        assert fig == mock_fig
        assert mock_plt.subplots.called

    @patch("mfg_pde.visualization.enhanced_network_plots.plt")
    @patch("matplotlib.cm")
    @patch("mfg_pde.visualization.enhanced_network_plots.NetworkMFGVisualizer.__init__")
    def test_plot_scheme_comparison_default_nodes(self, mock_super_init, mock_cm, mock_plt, mock_network_data):
        """Test scheme comparison with default node selection."""
        from mfg_pde.visualization.enhanced_network_plots import EnhancedNetworkMFGVisualizer

        mock_super_init.return_value = None
        mock_fig = Mock()
        # Create mock axes with sufficient columns for default nodes (up to 5)
        mock_axes = np.array([[Mock(), Mock(), Mock(), Mock(), Mock()], [Mock(), Mock(), Mock(), Mock(), Mock()]])
        mock_plt.subplots.return_value = (mock_fig, mock_axes)
        mock_cm.get_cmap.return_value = lambda x: np.array([[1, 0, 0, 1]] * len(x))

        viz = EnhancedNetworkMFGVisualizer(network_data=mock_network_data)
        viz.num_nodes = 5

        Nt, num_nodes = 10, 5
        solutions = {"FDM": (np.random.rand(Nt, num_nodes), np.random.rand(Nt, num_nodes))}
        times = np.linspace(0, 1, Nt)

        # No selected_nodes specified - should use default
        viz.plot_scheme_comparison(solutions, times, interactive=False)

        assert mock_plt.subplots.called


class TestNetwork3D:
    """Test 3D network visualization."""

    @patch("mfg_pde.visualization.enhanced_network_plots.PLOTLY_AVAILABLE", True)
    @patch("mfg_pde.visualization.enhanced_network_plots.NetworkMFGVisualizer.__init__")
    def test_plot_network_3d_with_values(self, mock_super_init, mock_network_data):
        """Test 3D network plot with node values."""
        from mfg_pde.visualization.enhanced_network_plots import EnhancedNetworkMFGVisualizer

        mock_super_init.return_value = None

        viz = EnhancedNetworkMFGVisualizer(network_data=mock_network_data)
        viz.node_positions = mock_network_data.node_positions
        viz.adjacency_matrix = mock_network_data.adjacency_matrix
        viz.num_nodes = 5

        node_values = np.random.rand(5)

        fig = viz.plot_network_3d(node_values=node_values, title="Test 3D Network", height_scale=2.0)

        # Functional assertion
        assert fig is not None
        assert hasattr(fig, "data") or hasattr(fig, "layout")

    @patch("mfg_pde.visualization.enhanced_network_plots.PLOTLY_AVAILABLE", True)
    @patch("mfg_pde.visualization.enhanced_network_plots.NetworkMFGVisualizer.__init__")
    def test_plot_network_3d_without_values(self, mock_super_init, mock_network_data):
        """Test 3D network plot without node values."""
        from mfg_pde.visualization.enhanced_network_plots import EnhancedNetworkMFGVisualizer

        mock_super_init.return_value = None

        viz = EnhancedNetworkMFGVisualizer(network_data=mock_network_data)
        viz.node_positions = mock_network_data.node_positions
        viz.adjacency_matrix = mock_network_data.adjacency_matrix
        viz.num_nodes = 5

        fig = viz.plot_network_3d(title="Test 3D Network")

        # Functional assertion
        assert fig is not None
        assert hasattr(fig, "data") or hasattr(fig, "layout")

    @patch("mfg_pde.visualization.enhanced_network_plots.PLOTLY_AVAILABLE", True)
    @patch("mfg_pde.visualization.enhanced_network_plots.NetworkMFGVisualizer.__init__")
    def test_plot_network_3d_without_positions(self, mock_super_init, mock_network_data):
        """Test 3D network plot with default circular layout."""
        from mfg_pde.visualization.enhanced_network_plots import EnhancedNetworkMFGVisualizer

        mock_super_init.return_value = None

        viz = EnhancedNetworkMFGVisualizer(network_data=mock_network_data)
        viz.node_positions = None  # No positions
        viz.adjacency_matrix = mock_network_data.adjacency_matrix
        viz.num_nodes = 5

        fig = viz.plot_network_3d()

        # Functional assertion
        assert fig is not None
        assert hasattr(fig, "data") or hasattr(fig, "layout")

    @patch("mfg_pde.visualization.enhanced_network_plots.PLOTLY_AVAILABLE", False)
    @patch("mfg_pde.visualization.enhanced_network_plots.NetworkMFGVisualizer.__init__")
    def test_plot_network_3d_plotly_not_available(self, mock_super_init, mock_network_data):
        """Test 3D network plot raises error without Plotly."""
        from mfg_pde.visualization.enhanced_network_plots import EnhancedNetworkMFGVisualizer

        mock_super_init.return_value = None

        viz = EnhancedNetworkMFGVisualizer(network_data=mock_network_data)
        viz.adjacency_matrix = mock_network_data.adjacency_matrix
        viz.num_nodes = 5

        with pytest.raises(ImportError, match="Plotly is required"):
            viz.plot_network_3d()


class TestHelperMethods:
    """Test helper methods."""

    @patch("mfg_pde.visualization.enhanced_network_plots.plt")
    @patch("mfg_pde.visualization.enhanced_network_plots.NetworkMFGVisualizer.__init__")
    def test_add_network_topology_to_matplotlib_with_values(self, mock_super_init, mock_plt, mock_network_data):
        """Test adding network topology to matplotlib with values."""
        from mfg_pde.visualization.enhanced_network_plots import EnhancedNetworkMFGVisualizer

        mock_super_init.return_value = None
        mock_ax = Mock()

        viz = EnhancedNetworkMFGVisualizer(network_data=mock_network_data)
        viz.node_positions = mock_network_data.node_positions
        viz.adjacency_matrix = mock_network_data.adjacency_matrix

        node_values = np.random.rand(5)

        viz._add_network_topology_to_matplotlib(mock_ax, node_values=node_values)

        assert mock_ax.plot.called
        assert mock_ax.scatter.called

    @patch("mfg_pde.visualization.enhanced_network_plots.plt")
    @patch("mfg_pde.visualization.enhanced_network_plots.NetworkMFGVisualizer.__init__")
    def test_add_network_topology_to_matplotlib_without_values(self, mock_super_init, mock_plt, mock_network_data):
        """Test adding network topology to matplotlib without values."""
        from mfg_pde.visualization.enhanced_network_plots import EnhancedNetworkMFGVisualizer

        mock_super_init.return_value = None
        mock_ax = Mock()

        viz = EnhancedNetworkMFGVisualizer(network_data=mock_network_data)
        viz.node_positions = mock_network_data.node_positions
        viz.adjacency_matrix = mock_network_data.adjacency_matrix

        viz._add_network_topology_to_matplotlib(mock_ax)

        assert mock_ax.plot.called
        assert mock_ax.scatter.called


class TestFactoryFunction:
    """Test factory function."""

    @patch("mfg_pde.visualization.enhanced_network_plots.EnhancedNetworkMFGVisualizer")
    def test_create_enhanced_network_visualizer(self, mock_visualizer_class, mock_network_data):
        """Test factory function."""
        from mfg_pde.visualization.enhanced_network_plots import create_enhanced_network_visualizer

        mock_instance = Mock()
        mock_visualizer_class.return_value = mock_instance

        viz = create_enhanced_network_visualizer(network_data=mock_network_data)

        assert viz == mock_instance
        mock_visualizer_class.assert_called_once_with(problem=None, network_data=mock_network_data)

    @patch("mfg_pde.visualization.enhanced_network_plots.EnhancedNetworkMFGVisualizer")
    def test_create_enhanced_network_visualizer_no_args(self, mock_visualizer_class):
        """Test factory function with no arguments."""
        from mfg_pde.visualization.enhanced_network_plots import create_enhanced_network_visualizer

        mock_instance = Mock()
        mock_visualizer_class.return_value = mock_instance

        viz = create_enhanced_network_visualizer()

        assert viz == mock_instance
        mock_visualizer_class.assert_called_once_with(problem=None, network_data=None)


class TestFileSaving:
    """Test file saving functionality."""

    @patch("mfg_pde.visualization.enhanced_network_plots.PLOTLY_AVAILABLE", True)
    @patch("mfg_pde.visualization.enhanced_network_plots.NetworkMFGVisualizer.__init__")
    def test_save_trajectory_plot(self, mock_super_init, mock_network_data, sample_trajectories, tmp_path):
        """Test saving trajectory plot to file."""
        from mfg_pde.visualization.enhanced_network_plots import EnhancedNetworkMFGVisualizer

        mock_super_init.return_value = None

        viz = EnhancedNetworkMFGVisualizer(network_data=mock_network_data)
        viz.node_positions = mock_network_data.node_positions
        viz.adjacency_matrix = mock_network_data.adjacency_matrix
        viz.num_nodes = 5

        save_path = str(tmp_path / "trajectories.html")

        viz.plot_lagrangian_trajectories(sample_trajectories, interactive=True, save_path=save_path)

        # Functional assertion: verify file was created
        assert (tmp_path / "trajectories.html").exists()

    @patch("mfg_pde.visualization.enhanced_network_plots.PLOTLY_AVAILABLE", True)
    @patch("mfg_pde.visualization.enhanced_network_plots.NetworkMFGVisualizer.__init__")
    def test_save_3d_network(self, mock_super_init, mock_network_data, tmp_path):
        """Test saving 3D network plot."""
        from mfg_pde.visualization.enhanced_network_plots import EnhancedNetworkMFGVisualizer

        mock_super_init.return_value = None

        viz = EnhancedNetworkMFGVisualizer(network_data=mock_network_data)
        viz.node_positions = mock_network_data.node_positions
        viz.adjacency_matrix = mock_network_data.adjacency_matrix
        viz.num_nodes = 5

        save_path = str(tmp_path / "network_3d.html")

        viz.plot_network_3d(save_path=save_path)

        # Functional assertion: verify file was created
        assert (tmp_path / "network_3d.html").exists()


class TestIntegration:
    """Integration tests combining multiple features."""

    @patch("mfg_pde.visualization.enhanced_network_plots.PLOTLY_AVAILABLE", True)
    @patch("mfg_pde.visualization.enhanced_network_plots.NetworkMFGVisualizer.__init__")
    def test_full_workflow_plotly(self, mock_super_init, mock_network_data, sample_trajectories, sample_velocity_field):
        """Test complete workflow with Plotly."""
        from mfg_pde.visualization.enhanced_network_plots import EnhancedNetworkMFGVisualizer

        mock_super_init.return_value = None

        viz = EnhancedNetworkMFGVisualizer(network_data=mock_network_data)
        viz.node_positions = mock_network_data.node_positions
        viz.adjacency_matrix = mock_network_data.adjacency_matrix
        viz.num_nodes = 5

        # Create multiple visualizations
        fig1 = viz.plot_lagrangian_trajectories(sample_trajectories, interactive=True)
        fig2 = viz.plot_velocity_field(sample_velocity_field, interactive=True)
        fig3 = viz.plot_network_3d()

        # Functional assertions: verify all figures were created
        assert fig1 is not None
        assert fig2 is not None
        assert fig3 is not None

    @patch("mfg_pde.visualization.enhanced_network_plots.plt")
    @patch("matplotlib.cm")
    @patch("mfg_pde.visualization.enhanced_network_plots.NetworkMFGVisualizer.__init__")
    def test_full_workflow_matplotlib(
        self, mock_super_init, mock_cm, mock_plt, mock_network_data, sample_trajectories, sample_velocity_field
    ):
        """Test complete workflow with matplotlib."""
        from mfg_pde.visualization.enhanced_network_plots import EnhancedNetworkMFGVisualizer

        mock_super_init.return_value = None
        mock_fig = Mock()
        mock_ax = Mock()

        # Create mock axes with sufficient columns (up to 5)
        mock_axes = np.array([[Mock(), Mock(), Mock(), Mock(), Mock()], [Mock(), Mock(), Mock(), Mock(), Mock()]])

        def subplots_side_effect(*args, **kwargs):
            # Check if nrows=2 in either kwargs or positional args
            if ("nrows" in kwargs and kwargs["nrows"] == 2) or (len(args) > 0 and args[0] == 2):
                return (mock_fig, mock_axes)
            return (mock_fig, mock_ax)

        mock_plt.subplots.side_effect = subplots_side_effect
        mock_cm.get_cmap.return_value = lambda x: np.array([[1, 0, 0, 1]] * len(x))

        viz = EnhancedNetworkMFGVisualizer(network_data=mock_network_data)
        viz.node_positions = mock_network_data.node_positions
        viz.adjacency_matrix = mock_network_data.adjacency_matrix
        viz.num_nodes = 5

        # Create multiple visualizations
        fig1 = viz.plot_lagrangian_trajectories(sample_trajectories, interactive=False)
        fig2 = viz.plot_velocity_field(sample_velocity_field, interactive=False)

        Nt, num_nodes = 10, 5
        solutions = {"FDM": (np.random.rand(Nt, num_nodes), np.random.rand(Nt, num_nodes))}
        times = np.linspace(0, 1, Nt)
        fig3 = viz.plot_scheme_comparison(solutions, times, interactive=False)

        # Functional assertions
        assert fig1 is not None
        assert fig2 is not None
        assert fig3 is not None
        assert mock_plt.subplots.called
