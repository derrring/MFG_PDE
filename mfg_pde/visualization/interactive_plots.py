"""
Advanced interactive visualization for MFG_PDE using Bokeh and Plotly.

This module provides high-quality interactive visualizations for:
- 2D and 3D MFG solution plots
- Parameter sweep analysis
- Convergence monitoring
- Real-time simulation visualization
- Multi-dimensional data exploration
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Core dependencies
try:
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.offline as offline
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = px = make_subplots = offline = None

try:
    import bokeh.transform as transform
    from bokeh.io import curdoc, push_notebook, show
    from bokeh.layouts import column, gridplot, row
    from bokeh.models import ColorBar, ColumnDataSource, HoverTool, LinearColorMapper
    from bokeh.palettes import Inferno256, Plasma256, Viridis256
    from bokeh.plotting import figure, output_file, save

    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False
    figure = save = output_file = HoverTool = ColorBar = LinearColorMapper = None
    Viridis256 = Plasma256 = Inferno256 = None
    gridplot = column = row = curdoc = push_notebook = show = transform = None
    ColumnDataSource = None

# Optional Polars integration
try:
    from ..utils.polars_integration import MFGDataFrame, POLARS_AVAILABLE
except ImportError:
    POLARS_AVAILABLE = False
    MFGDataFrame = None

logger = logging.getLogger(__name__)


class MFGPlotlyVisualizer:
    """
    Advanced Plotly-based visualizations for MFG problems.

    Features:
    - Interactive 2D/3D surface plots
    - Parameter sweep animations
    - Multi-plot dashboards
    - High-quality export capabilities
    """

    def __init__(self):
        """Initialize Plotly visualizer."""
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly not available. Install with: pip install plotly")

        self.default_config = {
            "displayModeBar": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": ["pan2d", "lasso2d"],
            "toImageButtonOptions": {
                "format": "png",
                "filename": "mfg_plot",
                "height": 800,
                "width": 1200,
                "scale": 2,
            },
        }

    def plot_density_evolution_2d(
        self,
        x_grid: np.ndarray,
        time_grid: np.ndarray,
        density_history: np.ndarray,
        title: str = "MFG Density Evolution",
    ) -> go.Figure:
        """
        Create 2D heatmap of density evolution over time.

        Args:
            x_grid: Spatial grid points
            time_grid: Time grid points
            density_history: Density values [time, space]
            title: Plot title

        Returns:
            Plotly Figure object
        """
        fig = go.Figure(
            data=go.Heatmap(
                z=density_history,
                x=x_grid,
                y=time_grid,
                colorscale="Viridis",
                hovertemplate="Position: %{x:.3f}<br>Time: %{y:.3f}<br>Density: %{z:.4f}<extra></extra>",
                colorbar=dict(title="Density m(t,x)"),
            )
        )

        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis_title="Position x",
            yaxis_title="Time t",
            font=dict(size=12),
            width=800,
            height=600,
        )

        return fig

    def plot_density_surface_3d(
        self,
        x_grid: np.ndarray,
        time_grid: np.ndarray,
        density_history: np.ndarray,
        title: str = "3D MFG Density Surface",
    ) -> go.Figure:
        """
        Create 3D surface plot of density evolution.

        Args:
            x_grid: Spatial grid points
            time_grid: Time grid points
            density_history: Density values [time, space]
            title: Plot title

        Returns:
            Plotly Figure object
        """
        fig = go.Figure(
            data=go.Surface(
                z=density_history,
                x=x_grid,
                y=time_grid,
                colorscale="Viridis",
                hovertemplate="Position: %{x:.3f}<br>Time: %{y:.3f}<br>Density: %{z:.4f}<extra></extra>",
                colorbar=dict(title="Density m(t,x)", x=1.1),
            )
        )

        fig.update_layout(
            title=dict(text=title, x=0.5),
            scene=dict(
                xaxis_title="Position x",
                yaxis_title="Time t",
                zaxis_title="Density m(t,x)",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
            ),
            width=900,
            height=700,
        )

        return fig

    def plot_value_function_3d(
        self,
        x_grid: np.ndarray,
        time_grid: np.ndarray,
        value_history: np.ndarray,
        title: str = "3D Value Function u(t,x)",
    ) -> go.Figure:
        """
        Create 3D surface plot of value function evolution.

        Args:
            x_grid: Spatial grid points
            time_grid: Time grid points
            value_history: Value function values [time, space]
            title: Plot title

        Returns:
            Plotly Figure object
        """
        fig = go.Figure(
            data=go.Surface(
                z=value_history,
                x=x_grid,
                y=time_grid,
                colorscale="Plasma",
                hovertemplate="Position: %{x:.3f}<br>Time: %{y:.3f}<br>Value: %{z:.4f}<extra></extra>",
                colorbar=dict(title="Value u(t,x)", x=1.1),
            )
        )

        fig.update_layout(
            title=dict(text=title, x=0.5),
            scene=dict(
                xaxis_title="Position x",
                yaxis_title="Time t",
                zaxis_title="Value u(t,x)",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
            ),
            width=900,
            height=700,
        )

        return fig

    def create_parameter_sweep_dashboard(
        self,
        sweep_results: List[Dict[str, Any]],
        parameter_name: str = "lambda",
        title: str = "Parameter Sweep Analysis",
    ) -> go.Figure:
        """
        Create interactive dashboard for parameter sweep results.

        Args:
            sweep_results: List of sweep result dictionaries
            parameter_name: Name of parameter being swept
            title: Dashboard title

        Returns:
            Plotly Figure with subplots
        """
        # Extract data
        param_values = [result.get(parameter_name, 0) for result in sweep_results]
        crater_depths = [result.get("crater_depth", 0) for result in sweep_results]
        spatial_spreads = [result.get("spatial_spread", 0) for result in sweep_results]
        equilibrium_types = [
            result.get("equilibrium_type", "Unknown") for result in sweep_results
        ]

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                f"Crater Depth vs {parameter_name}",
                f"Spatial Spread vs {parameter_name}",
                "Equilibrium Type Distribution",
                "Parameter Space Overview",
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"type": "pie"}, {"type": "scatter3d"}],
            ],
        )

        # Crater depth scatter
        fig.add_trace(
            go.Scatter(
                x=param_values,
                y=crater_depths,
                mode="markers+lines",
                name="Crater Depth",
                marker=dict(color="blue", size=8),
                hovertemplate=f"{parameter_name}: %{{x:.3f}}<br>Crater Depth: %{{y:.4f}}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # Spatial spread scatter
        fig.add_trace(
            go.Scatter(
                x=param_values,
                y=spatial_spreads,
                mode="markers+lines",
                name="Spatial Spread",
                marker=dict(color="red", size=8),
                hovertemplate=f"{parameter_name}: %{{x:.3f}}<br>Spatial Spread: %{{y:.4f}}<extra></extra>",
            ),
            row=1,
            col=2,
        )

        # Equilibrium type distribution
        eq_counts = {}
        for eq_type in equilibrium_types:
            eq_counts[eq_type] = eq_counts.get(eq_type, 0) + 1

        fig.add_trace(
            go.Pie(
                labels=list(eq_counts.keys()),
                values=list(eq_counts.values()),
                name="Equilibrium Types",
            ),
            row=2,
            col=1,
        )

        # 3D parameter space
        fig.add_trace(
            go.Scatter3d(
                x=param_values,
                y=crater_depths,
                z=spatial_spreads,
                mode="markers",
                marker=dict(size=5, color=param_values, colorscale="Viridis"),
                name="Parameter Space",
                hovertemplate=f"{parameter_name}: %{{x:.3f}}<br>Crater: %{{y:.4f}}<br>Spread: %{{z:.4f}}<extra></extra>",
            ),
            row=2,
            col=2,
        )

        fig.update_layout(title=dict(text=title, x=0.5), height=800, showlegend=True)

        return fig

    def create_convergence_animation(
        self,
        convergence_history: List[Dict[str, np.ndarray]],
        title: str = "MFG Convergence Animation",
    ) -> go.Figure:
        """
        Create animated convergence visualization.

        Args:
            convergence_history: List of convergence data per iteration
            title: Animation title

        Returns:
            Plotly Figure with animation
        """
        frames = []

        for i, data in enumerate(convergence_history):
            x_grid = data.get("x_grid", np.linspace(0, 1, len(data.get("density", []))))
            density = data.get("density", np.zeros_like(x_grid))
            value_func = data.get("value_function", np.zeros_like(x_grid))

            frame = go.Frame(
                data=[
                    go.Scatter(
                        x=x_grid,
                        y=density,
                        mode="lines",
                        name="Density m(t,x)",
                        line=dict(color="blue", width=3),
                    ),
                    go.Scatter(
                        x=x_grid,
                        y=value_func,
                        mode="lines",
                        name="Value u(t,x)",
                        line=dict(color="red", width=3),
                        yaxis="y2",
                    ),
                ],
                name=str(i),
            )
            frames.append(frame)

        # Initial frame
        initial_data = convergence_history[0] if convergence_history else {}
        x_grid = initial_data.get("x_grid", np.linspace(0, 1, 50))
        density = initial_data.get("density", np.zeros_like(x_grid))
        value_func = initial_data.get("value_function", np.zeros_like(x_grid))

        fig = go.Figure(
            data=[
                go.Scatter(
                    x=x_grid,
                    y=density,
                    mode="lines",
                    name="Density m(t,x)",
                    line=dict(color="blue", width=3),
                ),
                go.Scatter(
                    x=x_grid,
                    y=value_func,
                    mode="lines",
                    name="Value u(t,x)",
                    line=dict(color="red", width=3),
                    yaxis="y2",
                ),
            ],
            frames=frames,
        )

        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis_title="Position x",
            yaxis=dict(title="Density m(t,x)", side="left"),
            yaxis2=dict(title="Value u(t,x)", side="right", overlaying="y"),
            updatemenus=[
                {
                    "buttons": [
                        {
                            "args": [
                                None,
                                {
                                    "frame": {"duration": 100, "redraw": True},
                                    "fromcurrent": True,
                                },
                            ],
                            "label": "Play",
                            "method": "animate",
                        },
                        {
                            "args": [
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                            "label": "Pause",
                            "method": "animate",
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 87},
                    "showactive": False,
                    "type": "buttons",
                    "x": 0.1,
                    "xanchor": "right",
                    "y": 0,
                    "yanchor": "top",
                }
            ],
            sliders=[
                {
                    "active": 0,
                    "yanchor": "top",
                    "xanchor": "left",
                    "currentvalue": {
                        "font": {"size": 20},
                        "prefix": "Iteration:",
                        "visible": True,
                        "xanchor": "right",
                    },
                    "transition": {"duration": 300, "easing": "cubic-in-out"},
                    "pad": {"b": 10, "t": 50},
                    "len": 0.9,
                    "x": 0.1,
                    "y": 0,
                    "steps": [
                        {
                            "args": [
                                [str(i)],
                                {
                                    "frame": {"duration": 300, "redraw": True},
                                    "mode": "immediate",
                                    "transition": {"duration": 300},
                                },
                            ],
                            "label": str(i),
                            "method": "animate",
                        }
                        for i in range(len(frames))
                    ],
                }
            ],
        )

        return fig

    def save_figure(
        self, fig: go.Figure, filepath: Union[str, Path], format: str = "html", **kwargs
    ) -> None:
        """
        Save Plotly figure to file.

        Args:
            fig: Plotly Figure object
            filepath: Output file path
            format: Output format ('html', 'png', 'pdf', 'svg')
            **kwargs: Additional arguments for plotly save functions
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if format.lower() == "html":
            fig.write_html(str(filepath), config=self.default_config, **kwargs)
        elif format.lower() == "png":
            fig.write_image(str(filepath), format="png", **kwargs)
        elif format.lower() == "pdf":
            fig.write_image(str(filepath), format="pdf", **kwargs)
        elif format.lower() == "svg":
            fig.write_image(str(filepath), format="svg", **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Plotly figure saved: {filepath}")


class MFGBokehVisualizer:
    """
    Advanced Bokeh-based visualizations for MFG problems.

    Features:
    - Real-time interactive plots
    - Streaming data visualization
    - Custom dashboard layouts
    - High-performance rendering
    """

    def __init__(self):
        """Initialize Bokeh visualizer."""
        if not BOKEH_AVAILABLE:
            raise ImportError("Bokeh not available. Install with: pip install bokeh")

        self.default_tools = "pan,wheel_zoom,box_zoom,reset,save"
        self.default_width = 800
        self.default_height = 600

    def plot_density_heatmap(
        self,
        x_grid: np.ndarray,
        time_grid: np.ndarray,
        density_history: np.ndarray,
        title: str = "MFG Density Evolution",
    ) -> Any:
        """
        Create Bokeh heatmap of density evolution.

        Args:
            x_grid: Spatial grid points
            time_grid: Time grid points
            density_history: Density values [time, space]
            title: Plot title

        Returns:
            Bokeh Figure object
        """
        # Prepare data for Bokeh heatmap
        x_coords, t_coords = np.meshgrid(x_grid, time_grid)
        x_flat = x_coords.flatten()
        t_flat = t_coords.flatten()
        density_flat = density_history.flatten()

        # Create data source
        source = ColumnDataSource(data=dict(x=x_flat, y=t_flat, density=density_flat))

        # Create color mapper
        color_mapper = LinearColorMapper(
            palette=Viridis256, low=np.min(density_flat), high=np.max(density_flat)
        )

        # Create figure
        p = figure(
            title=title,
            x_axis_label="Position x",
            y_axis_label="Time t",
            width=self.default_width,
            height=self.default_height,
            tools=self.default_tools,
        )

        # Add rectangles for heatmap
        dx = x_grid[1] - x_grid[0] if len(x_grid) > 1 else 0.01
        dt = time_grid[1] - time_grid[0] if len(time_grid) > 1 else 0.01

        p.rect(
            x="x",
            y="y",
            width=dx,
            height=dt,
            color={"field": "density", "transform": color_mapper},
            source=source,
        )

        # Add color bar
        color_bar = ColorBar(color_mapper=color_mapper, width=8, location=(0, 0))
        p.add_layout(color_bar, "right")

        # Add hover tool
        hover = HoverTool(
            tooltips=[
                ("Position", "@x{0.000}"),
                ("Time", "@y{0.000}"),
                ("Density", "@density{0.0000}"),
            ]
        )
        p.add_tools(hover)

        return p

    def plot_convergence_monitoring(
        self,
        iterations: np.ndarray,
        errors: np.ndarray,
        title: str = "Convergence Monitoring",
    ) -> Any:
        """
        Create real-time convergence monitoring plot.

        Args:
            iterations: Iteration numbers
            errors: Error values
            title: Plot title

        Returns:
            Bokeh Figure object
        """
        # Create data source
        source = ColumnDataSource(data=dict(iterations=iterations, errors=errors))

        p = figure(
            title=title,
            x_axis_label="Iteration",
            y_axis_label="Error",
            y_axis_type="log",
            width=self.default_width,
            height=self.default_height,
            tools=self.default_tools,
        )

        # Add line plot
        p.line(
            "iterations", "errors", line_width=2, color="blue", alpha=0.8, source=source
        )
        p.circle("iterations", "errors", size=6, color="blue", alpha=0.6, source=source)

        # Add hover tool
        hover = HoverTool(
            tooltips=[("Iteration", "@iterations"), ("Error", "@errors{0.0000e+00}")]
        )
        p.add_tools(hover)

        return p

    def create_mfg_dashboard(
        self,
        x_grid: np.ndarray,
        density: np.ndarray,
        value_func: np.ndarray,
        convergence_data: Optional[Dict] = None,
        title: str = "MFG Solution Dashboard",
    ) -> Any:
        """
        Create comprehensive MFG solution dashboard.

        Args:
            x_grid: Spatial grid points
            density: Current density distribution
            value_func: Current value function
            convergence_data: Optional convergence data
            title: Dashboard title

        Returns:
            Bokeh layout object
        """
        # Create data sources
        main_source = ColumnDataSource(
            data=dict(x=x_grid, density=density, value_func=value_func)
        )

        # Density plot
        p1 = figure(
            title="Density Distribution m(t,x)",
            x_axis_label="Position x",
            y_axis_label="Density",
            width=400,
            height=300,
            tools=self.default_tools,
        )
        p1.line(
            "x",
            "density",
            line_width=3,
            color="blue",
            legend_label="m(t,x)",
            source=main_source,
        )
        p1.circle("x", "density", size=4, color="blue", alpha=0.6, source=main_source)

        # Value function plot
        p2 = figure(
            title="Value Function u(t,x)",
            x_axis_label="Position x",
            y_axis_label="Value",
            width=400,
            height=300,
            tools=self.default_tools,
        )
        p2.line(
            "x",
            "value_func",
            line_width=3,
            color="red",
            legend_label="u(t,x)",
            source=main_source,
        )
        p2.circle("x", "value_func", size=4, color="red", alpha=0.6, source=main_source)

        plots = [[p1, p2]]

        # Add convergence plot if data provided
        if convergence_data:
            iterations = convergence_data.get("iterations", np.array([]))
            errors = convergence_data.get("errors", np.array([]))

            conv_source = ColumnDataSource(
                data=dict(iterations=iterations, errors=errors)
            )

            p3 = figure(
                title="Convergence History",
                x_axis_label="Iteration",
                y_axis_label="Error",
                y_axis_type="log",
                width=400,
                height=300,
                tools=self.default_tools,
            )
            p3.line(
                "iterations", "errors", line_width=2, color="green", source=conv_source
            )
            p3.circle(
                "iterations",
                "errors",
                size=6,
                color="green",
                alpha=0.6,
                source=conv_source,
            )

            # Phase space plot (if available)
            if "phase_x" in convergence_data and "phase_y" in convergence_data:
                phase_source = ColumnDataSource(
                    data=dict(
                        phase_x=convergence_data["phase_x"],
                        phase_y=convergence_data["phase_y"],
                    )
                )

                p4 = figure(
                    title="Phase Space",
                    x_axis_label="Phase X",
                    y_axis_label="Phase Y",
                    width=400,
                    height=300,
                    tools=self.default_tools,
                )
                p4.line(
                    "phase_x",
                    "phase_y",
                    line_width=2,
                    color="purple",
                    source=phase_source,
                )
                plots.append([p3, p4])
            else:
                plots.append([p3])

        # Create grid layout
        grid = gridplot(plots, sizing_mode="fixed")

        return grid

    def save_plot(self, plot: Any, filepath: Union[str, Path]) -> None:
        """
        Save Bokeh plot to HTML file.

        Args:
            plot: Bokeh plot object
            filepath: Output file path
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        output_file(str(filepath))
        save(plot)
        logger.info(f"Bokeh plot saved: {filepath}")


class MFGVisualizationManager:
    """
    High-level manager for MFG visualizations combining Plotly and Bokeh.

    Provides unified interface for creating publication-quality visualizations
    with automatic fallback between visualization libraries.
    """

    def __init__(self, prefer_plotly: bool = True):
        """
        Initialize visualization manager.

        Args:
            prefer_plotly: Whether to prefer Plotly over Bokeh when both available
        """
        self.prefer_plotly = prefer_plotly
        self.plotly_viz = None
        self.bokeh_viz = None

        # Initialize available visualizers
        if PLOTLY_AVAILABLE:
            try:
                self.plotly_viz = MFGPlotlyVisualizer()
                logger.info("Plotly visualizer initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Plotly visualizer: {e}")

        if BOKEH_AVAILABLE:
            try:
                self.bokeh_viz = MFGBokehVisualizer()
                logger.info("Bokeh visualizer initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Bokeh visualizer: {e}")

        if not self.plotly_viz and not self.bokeh_viz:
            raise ImportError(
                "No visualization libraries available. Install plotly and/or bokeh"
            )

    def get_available_backends(self) -> List[str]:
        """Get list of available visualization backends."""
        backends = []
        if self.plotly_viz:
            backends.append("plotly")
        if self.bokeh_viz:
            backends.append("bokeh")
        return backends

    def create_2d_density_plot(
        self,
        x_grid: np.ndarray,
        time_grid: np.ndarray,
        density_history: np.ndarray,
        backend: str = "auto",
        title: str = "MFG Density Evolution",
    ) -> Union[go.Figure, Any]:
        """
        Create 2D density plot using specified backend.

        Args:
            x_grid: Spatial grid points
            time_grid: Time grid points
            density_history: Density values [time, space]
            backend: Visualization backend ("plotly", "bokeh", "auto")
            title: Plot title

        Returns:
            Plot object (Plotly Figure or Bokeh plot)
        """
        backend = self._select_backend(backend)

        if backend == "plotly" and self.plotly_viz:
            return self.plotly_viz.plot_density_evolution_2d(
                x_grid, time_grid, density_history, title
            )
        elif backend == "bokeh" and self.bokeh_viz:
            return self.bokeh_viz.plot_density_heatmap(
                x_grid, time_grid, density_history, title
            )
        else:
            raise ValueError(f"Backend {backend} not available")

    def create_3d_surface_plot(
        self,
        x_grid: np.ndarray,
        time_grid: np.ndarray,
        data: np.ndarray,
        data_type: str = "density",
        title: str = "3D MFG Surface",
    ) -> go.Figure:
        """
        Create 3D surface plot (Plotly only).

        Args:
            x_grid: Spatial grid points
            time_grid: Time grid points
            data: Data values [time, space]
            data_type: Type of data ("density" or "value")
            title: Plot title

        Returns:
            Plotly Figure object
        """
        if not self.plotly_viz:
            raise ValueError("3D surface plots require Plotly")

        if data_type == "density":
            return self.plotly_viz.plot_density_surface_3d(
                x_grid, time_grid, data, title
            )
        elif data_type == "value":
            return self.plotly_viz.plot_value_function_3d(
                x_grid, time_grid, data, title
            )
        else:
            raise ValueError(f"Unknown data type: {data_type}")

    def create_parameter_sweep_dashboard(
        self,
        sweep_results: List[Dict[str, Any]],
        parameter_name: str = "lambda",
        backend: str = "auto",
    ) -> Union[go.Figure, Any]:
        """
        Create parameter sweep dashboard.

        Args:
            sweep_results: List of sweep result dictionaries
            parameter_name: Name of parameter being swept
            backend: Visualization backend

        Returns:
            Dashboard plot object
        """
        backend = self._select_backend(backend)

        if backend == "plotly" and self.plotly_viz:
            return self.plotly_viz.create_parameter_sweep_dashboard(
                sweep_results, parameter_name
            )
        else:
            # For Bokeh, create simpler dashboard
            if not self.bokeh_viz:
                raise ValueError(f"Backend {backend} not available")

            # Extract basic data for Bokeh plots
            param_values = np.array([r.get(parameter_name, 0) for r in sweep_results])
            crater_depths = np.array([r.get("crater_depth", 0) for r in sweep_results])

            from bokeh.layouts import column
            from bokeh.plotting import figure

            p1 = figure(
                title=f"Crater Depth vs {parameter_name}",
                x_axis_label=parameter_name,
                y_axis_label="Crater Depth",
            )
            p1.circle(param_values, crater_depths, size=8, alpha=0.6)
            p1.line(param_values, crater_depths, line_width=2, alpha=0.8)

            return p1

    def save_plot(
        self,
        plot: Union[go.Figure, Any],
        filepath: Union[str, Path],
        format: str = "html",
        **kwargs,
    ) -> None:
        """
        Save plot to file with automatic format detection.

        Args:
            plot: Plot object
            filepath: Output file path
            format: Output format
            **kwargs: Additional save arguments
        """
        if hasattr(plot, "write_html"):  # Plotly figure
            if self.plotly_viz:
                self.plotly_viz.save_figure(plot, filepath, format, **kwargs)
        else:  # Bokeh plot
            if self.bokeh_viz:
                self.bokeh_viz.save_plot(plot, filepath)

    def _create_line_comparison_plot(
        self,
        plot_data: Dict[str, Dict[str, np.ndarray]],
        title: str = "Comparison Plot",
        xlabel: str = "x",
        ylabel: str = "y",
        ylim: Optional[Tuple[float, float]] = None,
        save_path: Optional[str] = None,
    ) -> Any:
        """
        Create a line comparison plot for multiple data series.

        Args:
            plot_data: Dictionary with series names as keys and {'x': x_data, 'y': y_data} as values
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            ylim: Y-axis limits as (min, max)
            save_path: Optional path to save the plot

        Returns:
            Figure object
        """
        if self.prefer_plotly and self.plotly_viz:
            # Use Plotly for comparison plot
            fig = go.Figure()

            colors = px.colors.qualitative.Set1
            for i, (series_name, data) in enumerate(plot_data.items()):
                fig.add_trace(
                    go.Scatter(
                        x=data["x"],
                        y=data["y"],
                        mode="lines",
                        name=series_name,
                        line=dict(color=colors[i % len(colors)], width=2),
                    )
                )

            fig.update_layout(
                title=title,
                xaxis_title=xlabel,
                yaxis_title=ylabel,
                template="plotly_white",
                hovermode="x unified",
                showlegend=True,
            )

            if ylim:
                fig.update_yaxes(range=ylim)

            if save_path:
                fig.write_html(save_path)

            return fig
        else:
            # Fallback to matplotlib
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 6))

            for series_name, data in plot_data.items():
                ax.plot(data["x"], data["y"], label=series_name, linewidth=2)

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend(loc="best", fontsize="small")
            ax.grid(True, alpha=0.3)

            if ylim:
                ax.set_ylim(ylim)

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")

            plt.tight_layout()
            return fig

    def _select_backend(self, backend: str) -> str:
        """Select appropriate visualization backend."""
        if backend == "auto":
            if self.prefer_plotly and self.plotly_viz:
                return "plotly"
            elif self.bokeh_viz:
                return "bokeh"
            elif self.plotly_viz:
                return "plotly"
            else:
                raise ValueError("No visualization backend available")

        return backend


# Factory functions
def create_plotly_visualizer() -> MFGPlotlyVisualizer:
    """Create Plotly visualizer instance."""
    return MFGPlotlyVisualizer()


def create_bokeh_visualizer() -> MFGBokehVisualizer:
    """Create Bokeh visualizer instance."""
    return MFGBokehVisualizer()


def create_visualization_manager(prefer_plotly: bool = True) -> MFGVisualizationManager:
    """Create visualization manager instance."""
    return MFGVisualizationManager(prefer_plotly)


# Convenience functions for quick visualization
def quick_2d_plot(
    x_grid: np.ndarray,
    time_grid: np.ndarray,
    density_history: np.ndarray,
    title: str = "MFG Density",
    backend: str = "auto",
) -> Union[go.Figure, Any]:
    """Quick 2D density plot creation."""
    viz_manager = create_visualization_manager()
    return viz_manager.create_2d_density_plot(
        x_grid, time_grid, density_history, backend, title
    )


def quick_3d_plot(
    x_grid: np.ndarray,
    time_grid: np.ndarray,
    data: np.ndarray,
    data_type: str = "density",
    title: str = "3D MFG Surface",
) -> go.Figure:
    """Quick 3D surface plot creation."""
    viz_manager = create_visualization_manager()
    return viz_manager.create_3d_surface_plot(x_grid, time_grid, data, data_type, title)
