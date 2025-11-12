"""Tests for mfg_pde.visualization.multidim_viz module."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

import numpy as np


@pytest.fixture
def mock_grid_2d():
    """Mock TensorProductGrid for 2D problems."""
    grid = Mock()
    grid.dimension = 2
    grid.coordinates = [
        np.linspace(0, 1, 21),  # x
        np.linspace(0, 1, 21),  # y
    ]
    # Mock meshgrid method
    X, Y = np.meshgrid(grid.coordinates[0], grid.coordinates[1], indexing="ij")
    grid.meshgrid = Mock(return_value=(X, Y))
    return grid


@pytest.fixture
def mock_grid_3d():
    """Mock TensorProductGrid for 3D problems."""
    grid = Mock()
    grid.dimension = 3
    grid.coordinates = [
        np.linspace(0, 1, 11),  # x
        np.linspace(0, 1, 11),  # y
        np.linspace(0, 1, 11),  # z
    ]
    return grid


@pytest.fixture
def sample_2d_data():
    """Sample 2D spatial data."""
    x = np.linspace(0, 1, 21)
    y = np.linspace(0, 1, 21)
    X, Y = np.meshgrid(x, y, indexing="ij")
    Z = np.sin(np.pi * X) * np.cos(np.pi * Y)
    return Z


@pytest.fixture
def sample_3d_temporal_data():
    """Sample 3D temporal data (Nt, Nx, Ny)."""
    Nt, Nx, Ny = 10, 21, 21
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    t = np.linspace(0, 1, Nt)

    data = np.zeros((Nt, Nx, Ny))
    for it in range(Nt):
        X, Y = np.meshgrid(x, y, indexing="ij")
        data[it, :, :] = np.sin(np.pi * X) * np.cos(np.pi * Y) * np.exp(-t[it])

    return data


class TestMultiDimVisualizerInitialization:
    """Test MultiDimVisualizer initialization."""

    @patch("plotly.graph_objects")
    def test_initialization_plotly_backend(self, mock_go, mock_grid_2d):
        """Test initialization with Plotly backend."""
        from mfg_pde.visualization.multidim_viz import MultiDimVisualizer

        viz = MultiDimVisualizer(mock_grid_2d, backend="plotly", colorscale="Viridis")

        assert viz.grid == mock_grid_2d
        assert viz.backend == "plotly"
        assert viz.colorscale == "Viridis"
        assert viz.plotly_available is True

    @patch("matplotlib.pyplot")
    def test_initialization_matplotlib_backend(self, mock_plt, mock_grid_2d):
        """Test initialization with Matplotlib backend."""
        from mfg_pde.visualization.multidim_viz import MultiDimVisualizer

        viz = MultiDimVisualizer(mock_grid_2d, backend="matplotlib", colorscale="Plasma")

        assert viz.grid == mock_grid_2d
        assert viz.backend == "matplotlib"
        assert viz.colorscale == "Plasma"
        assert viz.plotly_available is False

    def test_initialization_invalid_dimension(self, mock_grid_2d):
        """Test initialization fails with 1D grid."""
        from mfg_pde.visualization.multidim_viz import MultiDimVisualizer

        mock_grid_2d.dimension = 1  # Invalid

        with pytest.raises(ValueError, match="requires 2D or 3D grid"):
            MultiDimVisualizer(mock_grid_2d)

    def test_initialization_plotly_not_available(self, mock_grid_2d):
        """Test initialization fails gracefully without Plotly."""
        from mfg_pde.visualization.multidim_viz import MultiDimVisualizer

        # Mock the import to raise ImportError
        with (
            patch("builtins.__import__", side_effect=ImportError("plotly not available")),
            pytest.raises(ImportError, match="Plotly not available"),
        ):
            MultiDimVisualizer(mock_grid_2d, backend="plotly")


class TestSurfacePlot:
    """Test surface plot functionality."""

    @patch("plotly.graph_objects")
    def test_surface_plot_2d_data(self, mock_go, mock_grid_2d, sample_2d_data):
        """Test surface plot with 2D data."""
        from mfg_pde.visualization.multidim_viz import MultiDimVisualizer

        # Setup mock
        mock_fig = Mock()
        mock_go.Figure.return_value = mock_fig
        mock_go.Surface.return_value = Mock()

        viz = MultiDimVisualizer(mock_grid_2d, backend="plotly")
        fig = viz.surface_plot(sample_2d_data, title="Test Surface")

        assert mock_go.Surface.called
        assert mock_go.Figure.called
        assert fig == mock_fig

    @patch("plotly.graph_objects")
    def test_surface_plot_3d_temporal_data(self, mock_go, mock_grid_2d, sample_3d_temporal_data):
        """Test surface plot with 3D temporal data."""
        from mfg_pde.visualization.multidim_viz import MultiDimVisualizer

        mock_fig = Mock()
        mock_go.Figure.return_value = mock_fig
        mock_go.Surface.return_value = Mock()

        viz = MultiDimVisualizer(mock_grid_2d, backend="plotly")
        viz.surface_plot(sample_3d_temporal_data, title="Test", time_index=5)

        assert mock_go.Surface.called

    @patch("plotly.graph_objects")
    def test_surface_plot_invalid_data_shape(self, mock_go, mock_grid_2d):
        """Test surface plot with invalid data shape."""
        from mfg_pde.visualization.multidim_viz import MultiDimVisualizer

        viz = MultiDimVisualizer(mock_grid_2d, backend="plotly")
        invalid_data = np.array([1, 2, 3])  # 1D data

        with pytest.raises(ValueError, match="Expected 2D or 3D data"):
            viz.surface_plot(invalid_data)

    @patch("matplotlib.pyplot")
    def test_surface_plot_matplotlib_backend(self, mock_plt, mock_grid_2d, sample_2d_data):
        """Test surface plot with Matplotlib backend."""
        from mfg_pde.visualization.multidim_viz import MultiDimVisualizer

        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax

        viz = MultiDimVisualizer(mock_grid_2d, backend="matplotlib")
        viz.surface_plot(sample_2d_data, title="Matplotlib Surface")

        assert mock_plt.figure.called
        assert mock_ax.plot_surface.called


class TestContourPlot:
    """Test contour plot functionality."""

    @patch("plotly.graph_objects")
    def test_contour_plot_2d_data(self, mock_go, mock_grid_2d, sample_2d_data):
        """Test contour plot with 2D data."""
        from mfg_pde.visualization.multidim_viz import MultiDimVisualizer

        mock_fig = Mock()
        mock_go.Figure.return_value = mock_fig
        mock_go.Contour.return_value = Mock()

        viz = MultiDimVisualizer(mock_grid_2d, backend="plotly")
        viz.contour_plot(sample_2d_data, title="Test Contour", levels=15)

        assert mock_go.Contour.called
        assert mock_go.Figure.called

    @patch("plotly.graph_objects")
    def test_contour_plot_with_time_index(self, mock_go, mock_grid_2d, sample_3d_temporal_data):
        """Test contour plot with temporal data and time index."""
        from mfg_pde.visualization.multidim_viz import MultiDimVisualizer

        mock_fig = Mock()
        mock_go.Figure.return_value = mock_fig
        mock_go.Contour.return_value = Mock()

        viz = MultiDimVisualizer(mock_grid_2d, backend="plotly")
        viz.contour_plot(sample_3d_temporal_data, time_index=3)

        assert mock_go.Contour.called

    @patch("matplotlib.pyplot")
    def test_contour_plot_matplotlib(self, mock_plt, mock_grid_2d, sample_2d_data):
        """Test contour plot with Matplotlib backend."""
        from mfg_pde.visualization.multidim_viz import MultiDimVisualizer

        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        viz = MultiDimVisualizer(mock_grid_2d, backend="matplotlib")
        viz.contour_plot(sample_2d_data, levels=10)

        assert mock_ax.contourf.called
        assert mock_ax.contour.called


class TestHeatmap:
    """Test heatmap functionality."""

    @patch("plotly.graph_objects")
    def test_heatmap_2d_data(self, mock_go, mock_grid_2d, sample_2d_data):
        """Test heatmap with 2D data."""
        from mfg_pde.visualization.multidim_viz import MultiDimVisualizer

        mock_fig = Mock()
        mock_go.Figure.return_value = mock_fig
        mock_go.Heatmap.return_value = Mock()

        viz = MultiDimVisualizer(mock_grid_2d, backend="plotly")
        viz.heatmap(sample_2d_data, title="Test Heatmap")

        assert mock_go.Heatmap.called

    @patch("plotly.graph_objects")
    def test_heatmap_3d_temporal_default_time(self, mock_go, mock_grid_2d, sample_3d_temporal_data):
        """Test heatmap with 3D data using default time index."""
        from mfg_pde.visualization.multidim_viz import MultiDimVisualizer

        mock_fig = Mock()
        mock_go.Figure.return_value = mock_fig
        mock_go.Heatmap.return_value = Mock()

        viz = MultiDimVisualizer(mock_grid_2d, backend="plotly")
        viz.heatmap(sample_3d_temporal_data)  # Should use time_index=-1

        assert mock_go.Heatmap.called

    @patch("matplotlib.pyplot")
    def test_heatmap_matplotlib(self, mock_plt, mock_grid_2d, sample_2d_data):
        """Test heatmap with Matplotlib backend."""
        from mfg_pde.visualization.multidim_viz import MultiDimVisualizer

        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        viz = MultiDimVisualizer(mock_grid_2d, backend="matplotlib")
        viz.heatmap(sample_2d_data)

        assert mock_ax.imshow.called


class TestSlicePlot:
    """Test slice plot functionality."""

    @patch("plotly.graph_objects")
    def test_slice_plot_2d_data_x_slice(self, mock_go, mock_grid_2d, sample_2d_data):
        """Test slice plot along x dimension."""
        from mfg_pde.visualization.multidim_viz import MultiDimVisualizer

        mock_fig = Mock()
        mock_go.Figure.return_value = mock_fig
        mock_go.Scatter.return_value = Mock()

        viz = MultiDimVisualizer(mock_grid_2d, backend="plotly")
        viz.slice_plot(sample_2d_data, slice_dim=0, slice_index=10)

        assert mock_go.Scatter.called

    @patch("plotly.graph_objects")
    def test_slice_plot_2d_data_y_slice(self, mock_go, mock_grid_2d, sample_2d_data):
        """Test slice plot along y dimension."""
        from mfg_pde.visualization.multidim_viz import MultiDimVisualizer

        mock_fig = Mock()
        mock_go.Figure.return_value = mock_fig
        mock_go.Scatter.return_value = Mock()

        viz = MultiDimVisualizer(mock_grid_2d, backend="plotly")
        viz.slice_plot(sample_2d_data, slice_dim=1, slice_index=5)

        assert mock_go.Scatter.called

    @patch("matplotlib.pyplot")
    def test_slice_plot_matplotlib(self, mock_plt, mock_grid_2d, sample_2d_data):
        """Test slice plot with Matplotlib backend."""
        from mfg_pde.visualization.multidim_viz import MultiDimVisualizer

        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        viz = MultiDimVisualizer(mock_grid_2d, backend="matplotlib")
        viz.slice_plot(sample_2d_data, slice_dim=0, slice_index=10)

        assert mock_ax.plot.called

    @patch("plotly.graph_objects")
    def test_slice_plot_3d_temporal_data(self, mock_go, mock_grid_2d, sample_3d_temporal_data):
        """Test slice plot with 3D temporal data."""
        from mfg_pde.visualization.multidim_viz import MultiDimVisualizer

        mock_fig = Mock()
        mock_go.Figure.return_value = mock_fig
        mock_go.Scatter.return_value = Mock()

        viz = MultiDimVisualizer(mock_grid_2d, backend="plotly")
        viz.slice_plot(sample_3d_temporal_data, slice_dim=0, slice_index=10, time_index=5)

        assert mock_go.Scatter.called


class TestAnimation:
    """Test animation functionality."""

    @patch("plotly.graph_objects")
    def test_animation_creation(self, mock_go, mock_grid_2d, sample_3d_temporal_data):
        """Test animation creation with temporal data."""
        from mfg_pde.visualization.multidim_viz import MultiDimVisualizer

        mock_fig = Mock()
        mock_go.Figure.return_value = mock_fig
        mock_go.Surface.return_value = Mock()
        mock_go.Frame.return_value = Mock()

        viz = MultiDimVisualizer(mock_grid_2d, backend="plotly")
        viz.animation(sample_3d_temporal_data, title="Test Animation", fps=5)

        assert mock_go.Surface.called
        assert mock_go.Frame.called

    @patch("plotly.graph_objects")
    def test_animation_matplotlib_raises_error(self, mock_go, mock_grid_2d, sample_3d_temporal_data):
        """Test animation with Matplotlib backend raises error."""
        from mfg_pde.visualization.multidim_viz import MultiDimVisualizer

        # Create with matplotlib backend
        with patch("mfg_pde.visualization.multidim_viz.plt", create=True):
            viz = MultiDimVisualizer(mock_grid_2d, backend="matplotlib")

        with pytest.raises(ValueError, match="Animation requires Plotly backend"):
            viz.animation(sample_3d_temporal_data)

    @patch("plotly.graph_objects")
    def test_animation_invalid_data_shape(self, mock_go, mock_grid_2d, sample_2d_data):
        """Test animation with invalid data shape."""
        from mfg_pde.visualization.multidim_viz import MultiDimVisualizer

        viz = MultiDimVisualizer(mock_grid_2d, backend="plotly")

        with pytest.raises(ValueError, match="Expected 3D data"):
            viz.animation(sample_2d_data)

    @patch("plotly.graph_objects")
    def test_animation_3d_grid_raises_error(self, mock_go, mock_grid_3d):
        """Test animation with 3D grid raises error."""
        from mfg_pde.visualization.multidim_viz import MultiDimVisualizer

        viz = MultiDimVisualizer(mock_grid_3d, backend="plotly")
        data = np.zeros((10, 11, 11))  # Valid 3D temporal data

        with pytest.raises(ValueError, match="animation requires 2D grid"):
            viz.animation(data)


class TestSaveAndShow:
    """Test save and show functionality."""

    @patch("plotly.graph_objects")
    def test_save_plotly_html(self, mock_go, mock_grid_2d, tmp_path):
        """Test saving Plotly figure as HTML."""
        from mfg_pde.visualization.multidim_viz import MultiDimVisualizer

        mock_fig = Mock()
        mock_go.Figure.return_value = mock_fig

        viz = MultiDimVisualizer(mock_grid_2d, backend="plotly")
        filepath = tmp_path / "test.html"

        viz.save(mock_fig, filepath)

        assert mock_fig.write_html.called
        mock_fig.write_html.assert_called_once_with(str(filepath))

    @patch("matplotlib.pyplot")
    def test_save_matplotlib_image(self, mock_plt, mock_grid_2d, tmp_path):
        """Test saving Matplotlib figure as image."""
        from mfg_pde.visualization.multidim_viz import MultiDimVisualizer

        mock_fig = Mock()

        viz = MultiDimVisualizer(mock_grid_2d, backend="matplotlib")
        filepath = tmp_path / "test.png"

        viz.save(mock_fig, filepath)

        mock_fig.savefig.assert_called_once()

    @patch("plotly.graph_objects")
    def test_show_plotly(self, mock_go, mock_grid_2d):
        """Test showing Plotly figure."""
        from mfg_pde.visualization.multidim_viz import MultiDimVisualizer

        mock_fig = Mock()

        viz = MultiDimVisualizer(mock_grid_2d, backend="plotly")
        viz.show(mock_fig)

        mock_fig.show.assert_called_once()

    @patch("matplotlib.pyplot")
    def test_show_matplotlib(self, mock_plt, mock_grid_2d):
        """Test showing Matplotlib figure."""
        from mfg_pde.visualization.multidim_viz import MultiDimVisualizer

        mock_fig = Mock()

        viz = MultiDimVisualizer(mock_grid_2d, backend="matplotlib")
        viz.show(mock_fig)

        mock_plt.show.assert_called_once()


class TestDimensionValidation:
    """Test dimension validation for different plot types."""

    @patch("plotly.graph_objects")
    def test_surface_plot_requires_2d_grid(self, mock_go, mock_grid_3d, sample_2d_data):
        """Test surface plot requires 2D grid."""
        from mfg_pde.visualization.multidim_viz import MultiDimVisualizer

        viz = MultiDimVisualizer(mock_grid_3d, backend="plotly")

        with pytest.raises(ValueError, match="surface_plot requires 2D grid"):
            viz.surface_plot(sample_2d_data)

    @patch("plotly.graph_objects")
    def test_contour_plot_requires_2d_grid(self, mock_go, mock_grid_3d, sample_2d_data):
        """Test contour plot requires 2D grid."""
        from mfg_pde.visualization.multidim_viz import MultiDimVisualizer

        viz = MultiDimVisualizer(mock_grid_3d, backend="plotly")

        with pytest.raises(ValueError, match="contour_plot requires 2D grid"):
            viz.contour_plot(sample_2d_data)

    @patch("plotly.graph_objects")
    def test_heatmap_requires_2d_grid(self, mock_go, mock_grid_3d, sample_2d_data):
        """Test heatmap requires 2D grid."""
        from mfg_pde.visualization.multidim_viz import MultiDimVisualizer

        viz = MultiDimVisualizer(mock_grid_3d, backend="plotly")

        with pytest.raises(ValueError, match="heatmap requires 2D grid"):
            viz.heatmap(sample_2d_data)


class TestIntegration:
    """Integration tests combining multiple features."""

    @patch("plotly.graph_objects")
    def test_full_workflow_plotly(self, mock_go, mock_grid_2d, sample_3d_temporal_data, tmp_path):
        """Test complete workflow with Plotly backend."""
        from mfg_pde.visualization.multidim_viz import MultiDimVisualizer

        mock_fig = Mock()
        mock_go.Figure.return_value = mock_fig
        mock_go.Surface.return_value = Mock()
        mock_go.Contour.return_value = Mock()
        mock_go.Heatmap.return_value = Mock()

        viz = MultiDimVisualizer(mock_grid_2d, backend="plotly", colorscale="Plasma")

        # Create various plots
        viz.surface_plot(sample_3d_temporal_data, time_index=0)
        viz.contour_plot(sample_3d_temporal_data, time_index=5)
        viz.heatmap(sample_3d_temporal_data, time_index=-1)

        # Save and show
        filepath = tmp_path / "output.html"
        viz.save(mock_fig, filepath)
        viz.show(mock_fig)

        assert mock_go.Surface.called
        assert mock_go.Contour.called
        assert mock_go.Heatmap.called

    @patch("matplotlib.pyplot")
    def test_full_workflow_matplotlib(self, mock_plt, mock_grid_2d, sample_2d_data, tmp_path):
        """Test complete workflow with Matplotlib backend."""
        from mfg_pde.visualization.multidim_viz import MultiDimVisualizer

        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.figure.return_value = mock_fig
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_fig.add_subplot.return_value = mock_ax

        viz = MultiDimVisualizer(mock_grid_2d, backend="matplotlib")

        # Create plots
        viz.surface_plot(sample_2d_data)
        viz.contour_plot(sample_2d_data)
        viz.heatmap(sample_2d_data)

        # Save
        filepath = tmp_path / "output.png"
        viz.save(mock_fig, filepath)

        assert mock_ax.plot_surface.called or mock_ax.contourf.called or mock_ax.imshow.called
