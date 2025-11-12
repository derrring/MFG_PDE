"""Tests for mfg_pde.visualization.legacy_plotting module."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

import numpy as np


@pytest.fixture
def sample_grid_data():
    """Create sample grid data for testing."""
    x_grid = np.linspace(0, 1, 11)
    t_grid = np.linspace(0, 1, 6)
    U = np.random.rand(6, 11)
    M = np.random.rand(6, 11)
    M = M / M.sum(axis=1, keepdims=True)  # Normalize density
    return x_grid, t_grid, U, M


@pytest.fixture
def mock_problem():
    """Create mock MFG problem for legacy functions."""
    problem = Mock()
    problem.xSpace = np.linspace(0, 1, 11)
    problem.tSpace = np.linspace(0, 1, 6)
    problem.dx = 0.1
    return problem


class TestLegacyMyplot3d:
    """Test legacy myplot3d function."""

    @patch("mfg_pde.visualization.legacy_plotting.plt")
    def test_myplot3d_basic(self, mock_plt):
        """Test basic 3D surface plot creation."""
        from mfg_pde.visualization.legacy_plotting import myplot3d

        # Setup mocks
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax

        # Create test data
        X = np.linspace(0, 1, 5)
        Y = np.linspace(0, 1, 4)
        Z = np.random.rand(4, 5)

        # Call function
        myplot3d(X, Y, Z, title="Test Plot")

        # Verify calls
        mock_plt.figure.assert_called_once()
        mock_fig.add_subplot.assert_called_once_with(projection="3d")
        mock_ax.plot_surface.assert_called_once()
        mock_ax.set_xlabel.assert_called_once_with("x")
        mock_ax.set_ylabel.assert_called_once_with("time")
        mock_ax.set_title.assert_called_once_with("Test Plot")
        mock_ax.view_init.assert_called_once_with(40, -135)
        mock_plt.show.assert_called_once()

    @patch("mfg_pde.visualization.legacy_plotting.plt")
    def test_myplot3d_default_title(self, mock_plt):
        """Test myplot3d with default title."""
        from mfg_pde.visualization.legacy_plotting import myplot3d

        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax

        X = np.linspace(0, 1, 3)
        Y = np.linspace(0, 1, 3)
        Z = np.random.rand(3, 3)

        myplot3d(X, Y, Z)

        mock_ax.set_title.assert_called_once_with("Surface Plot")

    @patch("mfg_pde.visualization.legacy_plotting.plt")
    def test_myplot3d_formatting(self, mock_plt):
        """Test myplot3d applies proper formatting."""
        from mfg_pde.visualization.legacy_plotting import myplot3d

        mock_fig = Mock()
        mock_ax = Mock()
        mock_zaxis = Mock()
        mock_ax.zaxis = mock_zaxis
        mock_plt.figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax

        X = np.linspace(0, 1, 3)
        Y = np.linspace(0, 1, 3)
        Z = np.random.rand(3, 3)

        myplot3d(X, Y, Z)

        # Verify Z-axis formatting
        mock_zaxis.set_major_locator.assert_called_once()
        mock_zaxis.set_major_formatter.assert_called_once()


class TestLegacyPlotConvergence:
    """Test legacy plot_convergence function."""

    @patch("mfg_pde.visualization.legacy_plotting.plt")
    def test_plot_convergence_basic(self, mock_plt):
        """Test convergence plotting for U and M."""
        from mfg_pde.visualization.legacy_plotting import plot_convergence

        # Create convergence data
        iterations = 5
        l2disturel_u = np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
        l2disturel_m = np.array([2e-1, 2e-2, 2e-3, 2e-4, 2e-5])

        # Call function
        plot_convergence(iterations, l2disturel_u, l2disturel_m, solver_name="TestSolver")

        # Should create 2 figures (one for U, one for M)
        assert mock_plt.figure.call_count == 2

        # Should create 2 semilogy plots
        assert mock_plt.semilogy.call_count == 2

        # Should show 2 plots
        assert mock_plt.show.call_count == 2

        # Verify labels and titles
        xlabel_calls = list(mock_plt.xlabel.call_args_list)
        assert len(xlabel_calls) == 2
        assert all(call[0][0] == "Iteration" for call in xlabel_calls)

    @patch("mfg_pde.visualization.legacy_plotting.plt")
    def test_plot_convergence_default_solver_name(self, mock_plt):
        """Test plot_convergence with default solver name."""
        from mfg_pde.visualization.legacy_plotting import plot_convergence

        iterations = 3
        errors_u = np.array([1e-1, 1e-2, 1e-3])
        errors_m = np.array([1e-1, 1e-2, 1e-3])

        plot_convergence(iterations, errors_u, errors_m)

        # Check that default solver name "Solver" is used in titles
        title_calls = [call[0][0] for call in mock_plt.title.call_args_list]
        assert any("Solver" in title for title in title_calls)

    @patch("mfg_pde.visualization.legacy_plotting.plt")
    def test_plot_convergence_grid_enabled(self, mock_plt):
        """Test that grid is enabled in convergence plots."""
        from mfg_pde.visualization.legacy_plotting import plot_convergence

        iterations = 2
        errors_u = np.array([1e-1, 1e-2])
        errors_m = np.array([1e-1, 1e-2])

        plot_convergence(iterations, errors_u, errors_m)

        # Grid should be called twice (once for each plot)
        assert mock_plt.grid.call_count == 2
        assert all(call[0][0] is True for call in mock_plt.grid.call_args_list)


class TestLegacyPlotResults:
    """Test legacy plot_results function."""

    @patch("mfg_pde.visualization.legacy_plotting.myplot3d")
    @patch("mfg_pde.visualization.legacy_plotting.plt")
    def test_plot_results_basic(self, mock_plt, mock_myplot3d, mock_problem):
        """Test comprehensive results plotting."""
        from mfg_pde.visualization.legacy_plotting import plot_results

        # Create solution data
        u = np.random.rand(6, 11)
        m = np.random.rand(6, 11)
        m = m / m.sum(axis=1, keepdims=True)  # Normalize

        # Call function
        plot_results(mock_problem, u, m, solver_name="TestSolver")

        # Should create 3D plots for U and M
        assert mock_myplot3d.call_count == 2

        # Should create 2 additional matplotlib figures (final density, mass conservation)
        assert mock_plt.figure.call_count == 2

        # Should show 2 plots
        assert mock_plt.show.call_count == 2

    @patch("mfg_pde.visualization.legacy_plotting.myplot3d")
    @patch("mfg_pde.visualization.legacy_plotting.plt")
    def test_plot_results_mass_conservation(self, mock_plt, mock_myplot3d, mock_problem):
        """Test mass conservation plot creation."""
        from mfg_pde.visualization.legacy_plotting import plot_results

        u = np.random.rand(6, 11)
        m = np.random.rand(6, 11)
        m = m / m.sum(axis=1, keepdims=True)

        plot_results(mock_problem, u, m)

        # Verify mass conservation plot elements
        title_calls = [call[0][0] for call in mock_plt.title.call_args_list]
        assert any("Total Mass" in title for title in title_calls)

        # Verify ylim is called for mass conservation plot
        mock_plt.ylim.assert_called_once()

    @patch("mfg_pde.visualization.legacy_plotting.myplot3d")
    @patch("mfg_pde.visualization.legacy_plotting.plt")
    def test_plot_results_final_density(self, mock_plt, mock_myplot3d, mock_problem):
        """Test final density plot creation."""
        from mfg_pde.visualization.legacy_plotting import plot_results

        u = np.random.rand(6, 11)
        m = np.random.rand(6, 11)

        plot_results(mock_problem, u, m, solver_name="TestSolver")

        # Verify final density plot
        title_calls = [call[0][0] for call in mock_plt.title.call_args_list]
        assert any("Final Density" in title for title in title_calls)

        # Should call plot for final density
        plot_calls = list(mock_plt.plot.call_args_list)
        assert len(plot_calls) >= 2  # At least final density and mass conservation

    @patch("mfg_pde.visualization.legacy_plotting.myplot3d")
    @patch("mfg_pde.visualization.legacy_plotting.plt")
    def test_plot_results_subsampling(self, mock_plt, mock_myplot3d, mock_problem):
        """Test that results are subsampled for plotting."""
        from mfg_pde.visualization.legacy_plotting import plot_results

        u = np.random.rand(6, 11)
        m = np.random.rand(6, 11)

        plot_results(mock_problem, u, m)

        # Verify myplot3d was called with subsampled data (not full arrays)
        for call_args in mock_myplot3d.call_args_list:
            args = call_args[0]
            # First two args are x and t grids (subsampled)
            # Check that arrays are smaller than original due to subsampling
            assert len(args[0]) <= len(mock_problem.xSpace)
            assert len(args[1]) <= len(mock_problem.tSpace)


class TestModernPlotMFGSolution:
    """Test modern_plot_mfg_solution wrapper function."""

    @patch("mfg_pde.visualization.interactive_plots.create_visualization_manager")
    def test_modern_plot_mfg_solution_success(self, mock_create_viz, sample_grid_data):
        """Test successful modern MFG solution plotting."""
        from mfg_pde.visualization.legacy_plotting import modern_plot_mfg_solution

        x_grid, t_grid, U, M = sample_grid_data

        # Setup mock visualization manager
        mock_viz_manager = Mock()
        mock_viz_manager.create_2d_density_plot.return_value = Mock()
        mock_create_viz.return_value = mock_viz_manager

        # Call function
        result = modern_plot_mfg_solution(U, M, x_grid, t_grid, title="Test Solution")

        # Verify visualization manager was created
        mock_create_viz.assert_called_once()

        # Verify 2D density plot was created
        mock_viz_manager.create_2d_density_plot.assert_called_once()

        # Result should be the density plot
        assert result is not None

    @patch("mfg_pde.visualization.interactive_plots.create_visualization_manager")
    def test_modern_plot_mfg_solution_with_3d(self, mock_create_viz, sample_grid_data):
        """Test modern plotting with 3D surface support."""
        from mfg_pde.visualization.legacy_plotting import modern_plot_mfg_solution

        x_grid, t_grid, U, M = sample_grid_data

        # Setup mock with 3D support
        mock_viz_manager = Mock()
        mock_viz_manager.create_2d_density_plot.return_value = Mock()
        mock_viz_manager.create_3d_surface_plot = Mock()
        mock_create_viz.return_value = mock_viz_manager

        modern_plot_mfg_solution(U, M, x_grid, t_grid)

        # Should call 3D surface plot if available
        mock_viz_manager.create_3d_surface_plot.assert_called_once()

    @patch("mfg_pde.visualization.legacy_plotting.warnings.warn")
    @patch("mfg_pde.visualization.legacy_plotting.myplot3d")
    @patch("mfg_pde.visualization.interactive_plots.create_visualization_manager")
    def test_modern_plot_mfg_solution_fallback(self, mock_create_viz, mock_myplot3d, mock_warn, sample_grid_data):
        """Test fallback to legacy plotting on ImportError."""
        from mfg_pde.visualization.legacy_plotting import modern_plot_mfg_solution

        x_grid, t_grid, U, M = sample_grid_data

        # Simulate ImportError
        mock_create_viz.side_effect = ImportError("Module not found")

        modern_plot_mfg_solution(U, M, x_grid, t_grid, title="Fallback Test")

        # Should warn about fallback
        mock_warn.assert_called_once()
        assert "legacy matplotlib" in mock_warn.call_args[0][0]

        # Should fall back to myplot3d
        mock_myplot3d.assert_called_once()


class TestModernPlotConvergence:
    """Test modern_plot_convergence wrapper function."""

    @patch("mfg_pde.visualization.legacy_plotting.PLOTLY_AVAILABLE", True)
    @patch("mfg_pde.visualization.legacy_plotting.go")
    @patch("mfg_pde.visualization.legacy_plotting.px")
    @patch("mfg_pde.visualization.interactive_plots.create_visualization_manager")
    def test_modern_plot_convergence_plotly(self, mock_create_viz, mock_px, mock_go):
        """Test modern convergence plotting with Plotly backend."""
        from mfg_pde.visualization.legacy_plotting import modern_plot_convergence

        # Setup convergence data
        convergence_data = {
            "U_error": [1e-1, 1e-2, 1e-3],
            "M_error": [2e-1, 2e-2, 2e-3],
        }

        # Setup mocks
        mock_viz_manager = Mock()
        mock_viz_manager.plotly_viz = True
        mock_create_viz.return_value = mock_viz_manager

        mock_fig = Mock()
        mock_go.Figure.return_value = mock_fig
        mock_px.colors.qualitative.Set1 = ["#e41a1c", "#377eb8"]

        # Call function
        result = modern_plot_convergence(convergence_data, backend="plotly")

        # Should create Plotly figure
        mock_go.Figure.assert_called_once()

        # Should add traces for each metric
        assert mock_fig.add_trace.call_count == 2

        # Should update layout
        mock_fig.update_layout.assert_called_once()

        assert result is mock_fig

    @patch("mfg_pde.visualization.legacy_plotting.PLOTLY_AVAILABLE", True)
    @patch("mfg_pde.visualization.legacy_plotting.go")
    @patch("mfg_pde.visualization.legacy_plotting.px")
    @patch("mfg_pde.visualization.interactive_plots.create_visualization_manager")
    def test_modern_plot_convergence_with_tolerances(self, mock_create_viz, mock_px, mock_go):
        """Test convergence plotting with tolerance lines."""
        from mfg_pde.visualization.legacy_plotting import modern_plot_convergence

        convergence_data = {"error": [1e-1, 1e-2, 1e-3]}
        tolerances = {"error": 1e-4}

        mock_viz_manager = Mock()
        mock_viz_manager.plotly_viz = True
        mock_create_viz.return_value = mock_viz_manager

        mock_fig = Mock()
        mock_go.Figure.return_value = mock_fig
        mock_px.colors.qualitative.Set1 = ["#e41a1c"]

        modern_plot_convergence(convergence_data, tolerances=tolerances, backend="plotly")

        # Should add horizontal line for tolerance
        mock_fig.add_hline.assert_called_once()

    @patch("mfg_pde.visualization.legacy_plotting.PLOTLY_AVAILABLE", True)
    @patch("mfg_pde.visualization.legacy_plotting.go")
    @patch("mfg_pde.visualization.legacy_plotting.px")
    @patch("mfg_pde.visualization.interactive_plots.create_visualization_manager")
    def test_modern_plot_convergence_save_plotly(self, mock_create_viz, mock_px, mock_go, tmp_path):
        """Test saving Plotly convergence plot."""
        from mfg_pde.visualization.legacy_plotting import modern_plot_convergence

        convergence_data = {"error": [1e-1, 1e-2]}

        mock_viz_manager = Mock()
        mock_viz_manager.plotly_viz = True
        mock_create_viz.return_value = mock_viz_manager

        mock_fig = Mock()
        mock_go.Figure.return_value = mock_fig
        mock_px.colors.qualitative.Set1 = ["#e41a1c"]

        save_path = str(tmp_path / "convergence.html")
        modern_plot_convergence(convergence_data, save_path=save_path, backend="plotly")

        # Should call write_html
        mock_fig.write_html.assert_called_once_with(save_path)

    @patch("matplotlib.pyplot")
    def test_modern_plot_convergence_matplotlib_fallback(self, mock_plt):
        """Test matplotlib fallback for convergence plotting."""
        from mfg_pde.visualization.legacy_plotting import modern_plot_convergence

        convergence_data = {
            "U_error": [1e-1, 1e-2, 1e-3],
            "M_error": [2e-1, 2e-2, 2e-3],
        }

        # Setup matplotlib mocks
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        result = modern_plot_convergence(convergence_data, backend="matplotlib")

        # Should create matplotlib figure
        mock_plt.subplots.assert_called_once()

        # Should create semilogy plots
        assert mock_ax.semilogy.call_count == 2

        # Should set labels
        mock_ax.set_xlabel.assert_called_once_with("Iteration")
        mock_ax.set_ylabel.assert_called_once_with("Error/Residual")
        mock_ax.set_title.assert_called_once()

        assert result is mock_fig

    @patch("matplotlib.pyplot")
    def test_modern_plot_convergence_matplotlib_with_tolerances(self, mock_plt):
        """Test matplotlib convergence plotting with tolerance lines."""
        from mfg_pde.visualization.legacy_plotting import modern_plot_convergence

        convergence_data = {"error": [1e-1, 1e-2]}
        tolerances = {"error": 1e-3}

        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        modern_plot_convergence(convergence_data, tolerances=tolerances, backend="matplotlib")

        # Should add horizontal line for tolerance
        mock_ax.axhline.assert_called_once()

    @patch("matplotlib.pyplot")
    def test_modern_plot_convergence_matplotlib_save(self, mock_plt, tmp_path):
        """Test saving matplotlib convergence plot."""
        from mfg_pde.visualization.legacy_plotting import modern_plot_convergence

        convergence_data = {"error": [1e-1, 1e-2]}

        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        save_path = str(tmp_path / "convergence.png")
        modern_plot_convergence(convergence_data, save_path=save_path, backend="matplotlib")

        # Should call savefig
        mock_plt.savefig.assert_called_once()
        assert save_path in mock_plt.savefig.call_args[0]

    @patch("mfg_pde.visualization.legacy_plotting.warnings.warn")
    @patch("matplotlib.pyplot")
    @patch("mfg_pde.visualization.interactive_plots.create_visualization_manager")
    def test_modern_plot_convergence_exception_fallback(self, mock_create_viz, mock_plt, mock_warn):
        """Test fallback to matplotlib on exception."""
        from mfg_pde.visualization.legacy_plotting import modern_plot_convergence

        convergence_data = {"error": [1e-1, 1e-2]}

        # Simulate exception in Plotly path
        mock_create_viz.side_effect = Exception("Test exception")

        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        result = modern_plot_convergence(convergence_data)

        # Should warn about fallback (check first call - matplotlib may issue deprecation warnings)
        assert mock_warn.call_count >= 1
        assert "legacy matplotlib" in str(mock_warn.call_args_list[0])

        # Should create matplotlib figure as fallback
        mock_plt.subplots.assert_called_once()
        assert result is mock_fig


class TestBackwardCompatibilityAliases:
    """Test backward compatibility aliases."""

    def test_legacy_myplot3d_alias(self):
        """Test that legacy_myplot3d is an alias for myplot3d."""
        from mfg_pde.visualization.legacy_plotting import legacy_myplot3d, myplot3d

        assert legacy_myplot3d is myplot3d

    def test_legacy_plot_convergence_alias(self):
        """Test that legacy_plot_convergence is an alias."""
        from mfg_pde.visualization.legacy_plotting import legacy_plot_convergence, plot_convergence

        assert legacy_plot_convergence is plot_convergence

    def test_legacy_plot_results_alias(self):
        """Test that legacy_plot_results is an alias."""
        from mfg_pde.visualization.legacy_plotting import legacy_plot_results, plot_results

        assert legacy_plot_results is plot_results


class TestMatplotlibConfiguration:
    """Test matplotlib configuration settings."""

    def test_matplotlib_rcparams_configured(self):
        """Test that matplotlib rcParams are set properly."""
        # Import module to trigger configuration

        import matplotlib.pyplot as plt

        # Verify LaTeX is disabled for cross-platform compatibility
        assert plt.rcParams["text.usetex"] is False

        # Verify font family is set to sans-serif (matplotlib returns a list)
        assert plt.rcParams["font.family"] == ["sans-serif"]

        # Verify mathtext fontset
        assert plt.rcParams["mathtext.fontset"] == "dejavusans"


class TestIntegration:
    """Integration tests for legacy plotting module."""

    @patch("mfg_pde.visualization.legacy_plotting.plt")
    def test_full_workflow_legacy_functions(self, mock_plt, mock_problem):
        """Test complete workflow using legacy plotting functions."""
        from mfg_pde.visualization.legacy_plotting import plot_convergence, plot_results

        # Create solution data
        u = np.random.rand(6, 11)
        m = np.random.rand(6, 11)
        m = m / m.sum(axis=1, keepdims=True)

        # Convergence data
        iterations = 5
        errors_u = np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
        errors_m = np.array([2e-1, 2e-2, 2e-3, 2e-4, 2e-5])

        # Mock myplot3d to avoid actual plotting
        with patch("mfg_pde.visualization.legacy_plotting.myplot3d"):
            # Plot results
            plot_results(mock_problem, u, m, solver_name="IntegrationTest")

            # Plot convergence
            plot_convergence(iterations, errors_u, errors_m, solver_name="IntegrationTest")

        # Verify all plotting functions were called appropriately
        assert mock_plt.figure.call_count >= 2
        assert mock_plt.show.call_count >= 2
