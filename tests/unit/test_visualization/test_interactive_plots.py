"""Tests for mfg_pde.visualization.interactive_plots module."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

import numpy as np

# Import module under test
from mfg_pde.visualization.interactive_plots import (
    BOKEH_AVAILABLE,
    PLOTLY_AVAILABLE,
    create_bokeh_visualizer,
    create_plotly_visualizer,
    create_visualization_manager,
)

# Skip all tests if visualization backends not available
pytestmark = pytest.mark.skipif(
    not PLOTLY_AVAILABLE and not BOKEH_AVAILABLE, reason="Neither plotly nor bokeh available"
)


# Fixtures
@pytest.fixture
def sample_1d_data():
    """Sample 1D MFG solution data."""
    x_grid = np.linspace(0, 1, 50)
    time_grid = np.linspace(0, 1, 20)
    U = np.sin(np.pi * x_grid[None, :]) * np.exp(-time_grid[:, None])
    M = np.exp(-((x_grid[None, :] - 0.5) ** 2) / 0.1) * (1 - time_grid[:, None])
    return {
        "x_grid": x_grid,
        "time_grid": time_grid,
        "U": U,
        "M": M,
    }


@pytest.fixture
def sample_2d_data():
    """Sample 2D spatial data."""
    x = np.linspace(0, 1, 20)
    y = np.linspace(0, 1, 20)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.pi * X) * np.cos(np.pi * Y)
    return {
        "x": x,
        "y": y,
        "X": X,
        "Y": Y,
        "Z": Z,
    }


# Test MFGPlotlyVisualizer
@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not available")
class TestMFGPlotlyVisualizer:
    """Test MFGPlotlyVisualizer class."""

    def test_initialization(self):
        """Test MFGPlotlyVisualizer initialization."""
        from mfg_pde.visualization.interactive_plots import MFGPlotlyVisualizer

        visualizer = MFGPlotlyVisualizer()
        assert visualizer is not None
        assert hasattr(visualizer, "default_config")
        assert isinstance(visualizer.default_config, dict)
        assert "displayModeBar" in visualizer.default_config

    def test_initialization_without_plotly(self):
        """Test initialization fails gracefully without plotly."""
        from mfg_pde.visualization.interactive_plots import MFGPlotlyVisualizer

        with (
            patch("mfg_pde.visualization.interactive_plots.PLOTLY_AVAILABLE", False),
            pytest.raises(ImportError, match="Plotly not available"),
        ):
            MFGPlotlyVisualizer()

    def test_plot_density_evolution_2d(self, sample_1d_data):
        """Test 2D density evolution plot."""
        from mfg_pde.visualization.interactive_plots import MFGPlotlyVisualizer

        visualizer = MFGPlotlyVisualizer()

        # Mock plotly go.Figure to avoid actual plotting
        with patch("mfg_pde.visualization.interactive_plots.go") as mock_go:
            mock_fig = Mock()
            mock_go.Figure.return_value = mock_fig
            mock_go.Heatmap.return_value = Mock()

            visualizer.plot_density_evolution_2d(
                x_grid=sample_1d_data["x_grid"],
                time_grid=sample_1d_data["time_grid"],
                density_history=sample_1d_data["M"],
                title="Test Density",
            )

            # Verify figure was created
            assert mock_go.Figure.called

    def test_plot_value_function_3d(self, sample_1d_data):
        """Test 3D value function surface plot."""
        from mfg_pde.visualization.interactive_plots import MFGPlotlyVisualizer

        visualizer = MFGPlotlyVisualizer()

        with patch("mfg_pde.visualization.interactive_plots.go") as mock_go:
            mock_fig = Mock()
            mock_go.Figure.return_value = mock_fig
            mock_go.Surface.return_value = Mock()

            visualizer.plot_value_function_3d(
                x_grid=sample_1d_data["x_grid"],
                time_grid=sample_1d_data["time_grid"],
                value_history=sample_1d_data["U"],
                title="Test Value Function",
            )

            assert mock_go.Figure.called


# Test MFGBokehVisualizer
@pytest.mark.skipif(not BOKEH_AVAILABLE, reason="Bokeh not available")
class TestMFGBokehVisualizer:
    """Test MFGBokehVisualizer class."""

    def test_initialization(self):
        """Test MFGBokehVisualizer initialization."""
        from mfg_pde.visualization.interactive_plots import MFGBokehVisualizer

        visualizer = MFGBokehVisualizer()
        assert visualizer is not None
        assert hasattr(visualizer, "default_tools")

    def test_initialization_without_bokeh(self):
        """Test initialization fails gracefully without bokeh."""
        from mfg_pde.visualization.interactive_plots import MFGBokehVisualizer

        with (
            patch("mfg_pde.visualization.interactive_plots.BOKEH_AVAILABLE", False),
            pytest.raises(ImportError, match="Bokeh not available"),
        ):
            MFGBokehVisualizer()

    def test_plot_density_heatmap(self, sample_1d_data):
        """Test density heatmap plot."""
        from mfg_pde.visualization.interactive_plots import MFGBokehVisualizer

        visualizer = MFGBokehVisualizer()

        with patch("mfg_pde.visualization.interactive_plots.figure") as mock_figure:
            mock_fig = Mock()
            mock_figure.return_value = mock_fig

            visualizer.plot_density_heatmap(
                x_grid=sample_1d_data["x_grid"],
                time_grid=sample_1d_data["time_grid"],
                density_history=sample_1d_data["M"],
                title="Test Density Heatmap",
            )

            assert mock_figure.called

    def test_create_mfg_dashboard(self, sample_1d_data):
        """Test dashboard creation."""
        from mfg_pde.visualization.interactive_plots import MFGBokehVisualizer

        visualizer = MFGBokehVisualizer()

        with patch("mfg_pde.visualization.interactive_plots.gridplot") as mock_gridplot:
            mock_layout = Mock()
            mock_gridplot.return_value = mock_layout

            # Create mock figures
            with patch("mfg_pde.visualization.interactive_plots.figure"):
                visualizer.create_mfg_dashboard(
                    x_grid=sample_1d_data["x_grid"],
                    density=sample_1d_data["M"][0, :],  # Single time slice
                    value_func=sample_1d_data["U"][0, :],  # Single time slice
                )

                assert mock_gridplot.called


# Test MFGVisualizationManager
class TestMFGVisualizationManager:
    """Test MFGVisualizationManager class."""

    def test_initialization_with_plotly(self):
        """Test manager initialization preferring plotly."""
        from mfg_pde.visualization.interactive_plots import MFGVisualizationManager

        with (
            patch("mfg_pde.visualization.interactive_plots.PLOTLY_AVAILABLE", True),
            patch("mfg_pde.visualization.interactive_plots.MFGPlotlyVisualizer") as mock_plotly,
        ):
            MFGVisualizationManager(prefer_plotly=True)
            assert mock_plotly.called

    def test_initialization_with_bokeh(self):
        """Test manager initialization preferring bokeh."""
        from mfg_pde.visualization.interactive_plots import MFGVisualizationManager

        with (
            patch("mfg_pde.visualization.interactive_plots.BOKEH_AVAILABLE", True),
            patch("mfg_pde.visualization.interactive_plots.MFGBokehVisualizer") as mock_bokeh,
        ):
            MFGVisualizationManager(prefer_plotly=False)
            assert mock_bokeh.called

    def test_initialization_no_backends(self):
        """Test manager initialization fails without backends."""
        from mfg_pde.visualization.interactive_plots import MFGVisualizationManager

        with (
            patch("mfg_pde.visualization.interactive_plots.PLOTLY_AVAILABLE", False),
            patch("mfg_pde.visualization.interactive_plots.BOKEH_AVAILABLE", False),
            pytest.raises(ImportError, match="No visualization libraries available"),
        ):
            MFGVisualizationManager()

    def test_create_2d_density_plot(self, sample_1d_data):
        """Test unified 2D density plot interface."""
        from mfg_pde.visualization.interactive_plots import MFGVisualizationManager

        with (
            patch("mfg_pde.visualization.interactive_plots.PLOTLY_AVAILABLE", True),
            patch("mfg_pde.visualization.interactive_plots.MFGPlotlyVisualizer") as mock_plotly_class,
        ):
            mock_visualizer = Mock()
            mock_plotly_class.return_value = mock_visualizer

            manager = MFGVisualizationManager(prefer_plotly=True)
            manager.create_2d_density_plot(
                x_grid=sample_1d_data["x_grid"],
                time_grid=sample_1d_data["time_grid"],
                density_history=sample_1d_data["M"],
            )

            # Verify visualizer was used
            assert mock_visualizer.plot_density_evolution_2d.called


# Test Factory Functions
class TestFactoryFunctions:
    """Test factory functions."""

    @pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not available")
    def test_create_plotly_visualizer(self):
        """Test create_plotly_visualizer factory."""
        visualizer = create_plotly_visualizer()
        assert visualizer is not None
        from mfg_pde.visualization.interactive_plots import MFGPlotlyVisualizer

        assert isinstance(visualizer, MFGPlotlyVisualizer)

    @pytest.mark.skipif(not BOKEH_AVAILABLE, reason="Bokeh not available")
    def test_create_bokeh_visualizer(self):
        """Test create_bokeh_visualizer factory."""
        visualizer = create_bokeh_visualizer()
        assert visualizer is not None
        from mfg_pde.visualization.interactive_plots import MFGBokehVisualizer

        assert isinstance(visualizer, MFGBokehVisualizer)

    def test_create_visualization_manager_prefer_plotly(self):
        """Test create_visualization_manager with plotly preference."""
        with (
            patch("mfg_pde.visualization.interactive_plots.PLOTLY_AVAILABLE", True),
            patch("mfg_pde.visualization.interactive_plots.MFGPlotlyVisualizer"),
        ):
            manager = create_visualization_manager(prefer_plotly=True)
            assert manager is not None

    def test_create_visualization_manager_prefer_bokeh(self):
        """Test create_visualization_manager with bokeh preference."""
        with (
            patch("mfg_pde.visualization.interactive_plots.BOKEH_AVAILABLE", True),
            patch("mfg_pde.visualization.interactive_plots.MFGBokehVisualizer"),
        ):
            manager = create_visualization_manager(prefer_plotly=False)
            assert manager is not None


# Test Quick Plot Functions
@pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not available")
class TestQuickPlotFunctions:
    """Test quick plotting convenience functions."""

    def test_quick_2d_plot(self, sample_1d_data):
        """Test quick_2d_plot function."""
        from mfg_pde.visualization.interactive_plots import quick_2d_plot

        with patch("mfg_pde.visualization.interactive_plots.MFGPlotlyVisualizer") as mock_class:
            mock_visualizer = Mock()
            mock_class.return_value = mock_visualizer

            quick_2d_plot(
                x_grid=sample_1d_data["x_grid"],
                time_grid=sample_1d_data["time_grid"],
                density_history=sample_1d_data["M"],
                title="Quick 2D Test",
            )

            assert mock_visualizer.plot_density_evolution_2d.called

    def test_quick_3d_plot(self, sample_1d_data):
        """Test quick_3d_plot function."""
        from mfg_pde.visualization.interactive_plots import quick_3d_plot

        with patch("mfg_pde.visualization.interactive_plots.MFGPlotlyVisualizer") as mock_class:
            mock_visualizer = Mock()
            mock_class.return_value = mock_visualizer

            quick_3d_plot(
                x_grid=sample_1d_data["x_grid"],
                time_grid=sample_1d_data["time_grid"],
                data=sample_1d_data["U"],
                data_type="value",
                title="Quick 3D Test",
            )

            assert mock_visualizer.plot_value_function_3d.called


# Integration Tests
class TestIntegration:
    """Integration tests for interactive plotting."""

    @pytest.mark.skipif(not PLOTLY_AVAILABLE, reason="Plotly not available")
    def test_end_to_end_plotly_workflow(self, sample_1d_data):
        """Test complete workflow with Plotly."""
        from mfg_pde.visualization.interactive_plots import MFGPlotlyVisualizer

        visualizer = MFGPlotlyVisualizer()

        # Mock all plotly calls to avoid actual rendering
        with patch("mfg_pde.visualization.interactive_plots.go"):
            # Should not raise any errors
            visualizer.plot_density_evolution_2d(
                sample_1d_data["x_grid"],
                sample_1d_data["time_grid"],
                sample_1d_data["M"],
            )

            # Chain multiple plots
            visualizer.plot_density_evolution_2d(
                sample_1d_data["x_grid"],
                sample_1d_data["time_grid"],
                sample_1d_data["U"],
                title="Value Function",
            )

    @pytest.mark.skipif(not BOKEH_AVAILABLE, reason="Bokeh not available")
    def test_end_to_end_bokeh_workflow(self, sample_1d_data):
        """Test complete workflow with Bokeh."""
        from mfg_pde.visualization.interactive_plots import MFGBokehVisualizer

        visualizer = MFGBokehVisualizer()

        with patch("mfg_pde.visualization.interactive_plots.figure"):
            # Should not raise any errors
            visualizer.plot_density_heatmap(
                sample_1d_data["x_grid"],
                sample_1d_data["time_grid"],
                sample_1d_data["M"],
            )
