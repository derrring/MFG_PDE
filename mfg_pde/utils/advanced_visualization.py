#!/usr/bin/env python3
"""
Advanced Visualization Module for MFG_PDE

Provides comprehensive visualization capabilities using both Plotly for interactive
plots and matplotlib for publication-quality static plots. Supports MFG solution
visualization, convergence analysis, and monitoring dashboards.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import warnings

# Plotly imports with fallback
try:
    import plotly.graph_objects as go
    import plotly.subplots as sp
    from plotly.subplots import make_subplots
    import plotly.express as px
    import plotly.offline as offline
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Install with: pip install plotly")

# Matplotlib imports with fallback
try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import rcParams
    
    # Configure matplotlib for cross-platform compatibility
    rcParams['text.usetex'] = False  # Avoid LaTeX dependency
    rcParams['font.family'] = 'sans-serif'  # Use system sans-serif fonts
    rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Helvetica', 'sans-serif']
    rcParams['mathtext.fontset'] = 'dejavusans'  # Use DejaVu for math text
    rcParams['axes.formatter.use_mathtext'] = True
    
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("Matplotlib not available. Install with: pip install matplotlib")

from .logging import get_logger


class MathematicalVisualizationError(Exception):
    """Exception raised for visualization-related errors."""
    pass


class MFGVisualizer:
    """
    Main visualization class for MFG problems and solutions.
    
    Provides both interactive (Plotly) and static (matplotlib) visualization
    capabilities with automatic fallback and consistent styling.
    """
    
    def __init__(self, backend: str = "auto", theme: str = "default"):
        """
        Initialize the MFG visualizer.
        
        Args:
            backend: Visualization backend ("plotly", "matplotlib", or "auto")
            theme: Color theme ("default", "dark", "publication")
        """
        self.logger = get_logger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self.backend = self._validate_backend(backend)
        self.theme = theme
        self._setup_theme()
        
        self.logger.info(f"MFGVisualizer initialized with backend: {self.backend}")
    
    def _validate_backend(self, backend: str) -> str:
        """Validate and set the visualization backend."""
        if backend == "auto":
            if PLOTLY_AVAILABLE:
                return "plotly"
            elif MATPLOTLIB_AVAILABLE:
                return "matplotlib"
            else:
                raise MathematicalVisualizationError("No visualization backend available")
        elif backend == "plotly" and not PLOTLY_AVAILABLE:
            raise MathematicalVisualizationError("Plotly backend requested but not available")
        elif backend == "matplotlib" and not MATPLOTLIB_AVAILABLE:
            raise MathematicalVisualizationError("Matplotlib backend requested but not available")
        
        return backend
    
    def _setup_theme(self):
        """Setup color themes and styling."""
        if self.theme == "dark":
            self.colors = {
                'primary': '#00D4FF',
                'secondary': '#FF6B6B',
                'tertiary': '#4ECDC4',
                'background': '#2F3349',
                'surface': '#383B53'
            }
        elif self.theme == "publication":
            self.colors = {
                'primary': '#1f77b4',
                'secondary': '#ff7f0e',
                'tertiary': '#2ca02c',
                'background': 'white',
                'surface': '#f8f9fa'
            }
        else:  # default
            self.colors = {
                'primary': '#636EFA',
                'secondary': '#EF553B',
                'tertiary': '#00CC96',
                'background': 'white',
                'surface': '#f8f9fa'
            }
    
    def plot_mfg_solution(self, 
                         U: np.ndarray, 
                         M: np.ndarray, 
                         x_grid: np.ndarray, 
                         t_grid: np.ndarray,
                         title: str = "MFG Solution",
                         save_path: Optional[str] = None,
                         show: bool = True) -> Optional[Any]:
        """
        Plot MFG solution (value function U and density M).
        
        Args:
            U: Value function array (Nx, Nt)
            M: Density function array (Nx, Nt)
            x_grid: Spatial grid points
            t_grid: Temporal grid points
            title: Plot title
            save_path: Path to save the plot
            show: Whether to display the plot
            
        Returns:
            Figure object (backend-dependent)
        """
        self.logger.info(f"Creating MFG solution plot with {self.backend} backend")
        
        if self.backend == "plotly":
            return self._plot_mfg_solution_plotly(U, M, x_grid, t_grid, title, save_path, show)
        else:
            return self._plot_mfg_solution_matplotlib(U, M, x_grid, t_grid, title, save_path, show)
    
    def _plot_mfg_solution_plotly(self, U, M, x_grid, t_grid, title, save_path, show):
        """Create MFG solution plot using Plotly."""
        # Create subplot with 3D surfaces
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'surface'}, {'type': 'surface'}]],
            subplot_titles=['Value Function u(t,x)', 'Density m(t,x)'],
            horizontal_spacing=0.1
        )
        
        # Create meshgrids for 3D plotting
        X, T = np.meshgrid(x_grid, t_grid)
        
        # Value function surface
        fig.add_trace(
            go.Surface(
                x=X, y=T, z=U.T,
                colorscale='Viridis',
                name='u(t,x)',
                showscale=True,
                colorbar=dict(x=0.45, len=0.8)
            ),
            row=1, col=1
        )
        
        # Density surface
        fig.add_trace(
            go.Surface(
                x=X, y=T, z=M.T,
                colorscale='Plasma',
                name='m(t,x)',
                showscale=True,
                colorbar=dict(x=1.02, len=0.8)
            ),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=dict(text=title, x=0.5),
            scene=dict(
                xaxis_title="Space (x)",
                yaxis_title="Time (t)",
                zaxis_title="u(t,x)",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            scene2=dict(
                xaxis_title="Space (x)",
                yaxis_title="Time (t)",
                zaxis_title="m(t,x)",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            height=600,
            width=1200
        )
        
        if save_path:
            fig.write_html(save_path)
            self.logger.info(f"Plot saved to {save_path}")
        
        if show:
            fig.show()
        
        return fig
    
    def _plot_mfg_solution_matplotlib(self, U, M, x_grid, t_grid, title, save_path, show):
        """Create MFG solution plot using matplotlib."""
        fig = plt.figure(figsize=(15, 6))
        
        # Create meshgrids
        X, T = np.meshgrid(x_grid, t_grid)
        
        # Value function plot
        ax1 = fig.add_subplot(121, projection='3d')
        surf1 = ax1.plot_surface(X, T, U.T, cmap='viridis', alpha=0.9)
        ax1.set_xlabel('Space (x)')
        ax1.set_ylabel('Time (t)')
        ax1.set_zlabel('u(t,x)')
        ax1.set_title('Value Function u(t,x)')
        fig.colorbar(surf1, ax=ax1, shrink=0.8)
        
        # Density plot
        ax2 = fig.add_subplot(122, projection='3d')
        surf2 = ax2.plot_surface(X, T, M.T, cmap='plasma', alpha=0.9)
        ax2.set_xlabel('Space (x)')
        ax2.set_ylabel('Time (t)')
        ax2.set_zlabel('m(t,x)')
        ax2.set_title('Density m(t,x)')
        fig.colorbar(surf2, ax=ax2, shrink=0.8)
        
        plt.suptitle(title, fontsize=18, y=0.96)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Plot saved to {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_convergence_history(self,
                                convergence_data: Dict[str, List[float]],
                                title: str = "Convergence History",
                                log_scale: bool = True,
                                save_path: Optional[str] = None,
                                show: bool = True) -> Optional[Any]:
        """
        Plot convergence history for different metrics.
        
        Args:
            convergence_data: Dictionary with metric names as keys and lists of values
            title: Plot title
            log_scale: Whether to use logarithmic scale for y-axis
            save_path: Path to save the plot
            show: Whether to display the plot
            
        Returns:
            Figure object
        """
        self.logger.info(f"Creating convergence history plot with {len(convergence_data)} metrics")
        
        if self.backend == "plotly":
            return self._plot_convergence_plotly(convergence_data, title, log_scale, save_path, show)
        else:
            return self._plot_convergence_matplotlib(convergence_data, title, log_scale, save_path, show)
    
    def _plot_convergence_plotly(self, convergence_data, title, log_scale, save_path, show):
        """Create convergence plot using Plotly."""
        fig = go.Figure()
        
        colors = [self.colors['primary'], self.colors['secondary'], self.colors['tertiary']]
        
        for i, (metric_name, values) in enumerate(convergence_data.items()):
            iterations = list(range(1, len(values) + 1))
            
            fig.add_trace(go.Scatter(
                x=iterations,
                y=values,
                mode='lines+markers',
                name=metric_name,
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=4)
            ))
        
        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis_title="Iteration",
            yaxis_title="Error/Residual",
            yaxis_type="log" if log_scale else "linear",
            hovermode='x unified',
            template='plotly_white',
            height=500,
            width=800
        )
        
        if save_path:
            fig.write_html(save_path)
            self.logger.info(f"Convergence plot saved to {save_path}")
        
        if show:
            fig.show()
        
        return fig
    
    def _plot_convergence_matplotlib(self, convergence_data, title, log_scale, save_path, show):
        """Create convergence plot using matplotlib."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = [self.colors['primary'], self.colors['secondary'], self.colors['tertiary']]
        
        for i, (metric_name, values) in enumerate(convergence_data.items()):
            iterations = list(range(1, len(values) + 1))
            ax.plot(iterations, values, 'o-', label=metric_name, 
                   color=colors[i % len(colors)], linewidth=2, markersize=4)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Error/Residual')
        ax.set_title(title)
        
        if log_scale:
            ax.set_yscale('log')
        
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Convergence plot saved to {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_solution_snapshots(self,
                               U: np.ndarray,
                               M: np.ndarray,
                               x_grid: np.ndarray,
                               t_indices: List[int],
                               t_grid: np.ndarray,
                               title: str = "Solution Snapshots",
                               save_path: Optional[str] = None,
                               show: bool = True) -> Optional[Any]:
        """
        Plot solution snapshots at specific time points.
        
        Args:
            U: Value function array
            M: Density array
            x_grid: Spatial grid
            t_indices: Time indices to plot
            t_grid: Time grid
            title: Plot title
            save_path: Save path
            show: Whether to display
            
        Returns:
            Figure object
        """
        self.logger.info(f"Creating solution snapshots at {len(t_indices)} time points")
        
        if self.backend == "plotly":
            return self._plot_snapshots_plotly(U, M, x_grid, t_indices, t_grid, title, save_path, show)
        else:
            return self._plot_snapshots_matplotlib(U, M, x_grid, t_indices, t_grid, title, save_path, show)
    
    def _plot_snapshots_plotly(self, U, M, x_grid, t_indices, t_grid, title, save_path, show):
        """Create snapshots plot using Plotly."""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Value Function u(t,x)', 'Density m(t,x)'],
            vertical_spacing=0.1
        )
        
        colors = px.colors.qualitative.Set1
        
        for i, t_idx in enumerate(t_indices):
            color = colors[i % len(colors)]
            t_val = t_grid[t_idx]
            
            # Value function
            fig.add_trace(
                go.Scatter(
                    x=x_grid, y=U[:, t_idx],
                    mode='lines',
                    name=f'U(x, t={t_val:.2f})',
                    line=dict(color=color, width=2)
                ),
                row=1, col=1
            )
            
            # Density
            fig.add_trace(
                go.Scatter(
                    x=x_grid, y=M[:, t_idx],
                    mode='lines',
                    name=f'M(x, t={t_val:.2f})',
                    line=dict(color=color, width=2, dash='dash')
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            title=dict(text=title, x=0.5),
            height=800,
            width=800,
            template='plotly_white'
        )
        
        fig.update_xaxes(title_text="Space (x)", row=2, col=1)
        fig.update_yaxes(title_text="u(t,x)", row=1, col=1)
        fig.update_yaxes(title_text="m(t,x)", row=2, col=1)
        
        if save_path:
            fig.write_html(save_path)
            self.logger.info(f"Snapshots plot saved to {save_path}")
        
        if show:
            fig.show()
        
        return fig
    
    def _plot_snapshots_matplotlib(self, U, M, x_grid, t_indices, t_grid, title, save_path, show):
        """Create snapshots plot using matplotlib."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(t_indices)))
        
        for i, t_idx in enumerate(t_indices):
            t_val = t_grid[t_idx]
            color = colors[i]
            
            # Value function
            ax1.plot(x_grid, U[:, t_idx], '-', color=color, 
                    label=f't={t_val:.2f}', linewidth=2)
            
            # Density
            ax2.plot(x_grid, M[:, t_idx], '--', color=color, 
                    label=f't={t_val:.2f}', linewidth=2)
        
        ax1.set_ylabel('u(t,x)')
        ax1.set_title('Value Function u(t,x)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.set_xlabel('Space (x)')
        ax2.set_ylabel('m(t,x)')
        ax2.set_title('Density m(t,x)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.suptitle(title, fontsize=18, y=0.96)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Snapshots plot saved to {save_path}")
        
        if show:
            plt.show()
        
        return fig


class SolverMonitoringDashboard:
    """
    Interactive dashboard for monitoring solver progress and performance.
    
    Provides real-time visualization of solver metrics, convergence history,
    and performance indicators.
    """
    
    def __init__(self, update_interval: float = 1.0):
        """
        Initialize the monitoring dashboard.
        
        Args:
            update_interval: Update interval in seconds
        """
        self.logger = get_logger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self.update_interval = update_interval
        self.metrics_history = {}
        self.performance_data = {}
        
        if not PLOTLY_AVAILABLE:
            raise MathematicalVisualizationError("Dashboard requires Plotly. Install with: pip install plotly")
        
        self.logger.info("Solver monitoring dashboard initialized")
    
    def add_metric(self, name: str, value: float, iteration: int):
        """Add a metric value to the monitoring history."""
        if name not in self.metrics_history:
            self.metrics_history[name] = {'iterations': [], 'values': []}
        
        self.metrics_history[name]['iterations'].append(iteration)
        self.metrics_history[name]['values'].append(value)
    
    def add_performance_data(self, operation: str, duration: float, metadata: Dict[str, Any] = None):
        """Add performance data to the monitoring."""
        if operation not in self.performance_data:
            self.performance_data[operation] = []
        
        entry = {'duration': duration, 'timestamp': len(self.performance_data[operation])}
        if metadata:
            entry.update(metadata)
        
        self.performance_data[operation].append(entry)
    
    def create_dashboard(self, save_path: Optional[str] = None, show: bool = True) -> go.Figure:
        """
        Create comprehensive monitoring dashboard.
        
        Args:
            save_path: Path to save dashboard HTML
            show: Whether to display dashboard
            
        Returns:
            Plotly figure object
        """
        self.logger.info("Creating solver monitoring dashboard")
        
        # Calculate subplot layout
        n_metrics = len(self.metrics_history)
        n_performance = len(self.performance_data)
        total_plots = n_metrics + n_performance + 1  # +1 for summary
        
        rows = max(3, (total_plots + 1) // 2)
        
        # Create subplots
        fig = make_subplots(
            rows=rows, cols=2,
            subplot_titles=self._get_subplot_titles(),
            specs=[[{'secondary_y': False}] * 2 for _ in range(rows)],
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )
        
        # Add convergence metrics
        self._add_convergence_plots(fig)
        
        # Add performance metrics
        self._add_performance_plots(fig)
        
        # Add summary statistics
        self._add_summary_plot(fig)
        
        # Update layout
        fig.update_layout(
            title=dict(
                text="MFG Solver Monitoring Dashboard",
                x=0.5,
                font=dict(size=20)
            ),
            height=200 * rows,
            width=1400,
            template='plotly_white',
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
            self.logger.info(f"Dashboard saved to {save_path}")
        
        if show:
            fig.show()
        
        return fig
    
    def _get_subplot_titles(self) -> List[str]:
        """Generate subplot titles."""
        titles = []
        
        # Convergence metrics
        for metric_name in self.metrics_history.keys():
            titles.append(f"Convergence: {metric_name}")
        
        # Performance metrics
        for operation in self.performance_data.keys():
            titles.append(f"Performance: {operation}")
        
        # Summary
        titles.append("Summary Statistics")
        
        # Pad with empty titles if needed
        while len(titles) % 2 != 0:
            titles.append("")
        
        return titles
    
    def _add_convergence_plots(self, fig):
        """Add convergence plots to dashboard."""
        colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']
        
        for i, (metric_name, data) in enumerate(self.metrics_history.items()):
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            fig.add_trace(
                go.Scatter(
                    x=data['iterations'],
                    y=data['values'],
                    mode='lines+markers',
                    name=metric_name,
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=4)
                ),
                row=row, col=col
            )
            
            fig.update_yaxes(type="log", row=row, col=col)
    
    def _add_performance_plots(self, fig):
        """Add performance plots to dashboard."""
        n_metrics = len(self.metrics_history)
        
        for i, (operation, data) in enumerate(self.performance_data.items()):
            row = ((n_metrics + i) // 2) + 1
            col = ((n_metrics + i) % 2) + 1
            
            timestamps = [d['timestamp'] for d in data]
            durations = [d['duration'] for d in data]
            
            fig.add_trace(
                go.Bar(
                    x=timestamps,
                    y=durations,
                    name=operation,
                    marker_color='lightblue'
                ),
                row=row, col=col
            )
    
    def _add_summary_plot(self, fig):
        """Add summary statistics plot."""
        n_plots = len(self.metrics_history) + len(self.performance_data)
        row = ((n_plots) // 2) + 1
        col = ((n_plots) % 2) + 1
        
        # Create summary data
        summary_data = []
        
        for metric_name, data in self.metrics_history.items():
            if data['values']:
                summary_data.append({
                    'Metric': metric_name,
                    'Current': data['values'][-1],
                    'Best': min(data['values']),
                    'Iterations': len(data['values'])
                })
        
        if summary_data:
            metrics = [d['Metric'] for d in summary_data]
            current_values = [d['Current'] for d in summary_data]
            
            fig.add_trace(
                go.Bar(
                    x=metrics,
                    y=current_values,
                    name='Current Values',
                    marker_color='lightgreen'
                ),
                row=row, col=col
            )


class VisualizationUtils:
    """Utility functions for visualization tasks."""
    
    @staticmethod
    def create_animation(U_sequence: List[np.ndarray],
                        x_grid: np.ndarray,
                        title: str = "MFG Solution Animation",
                        save_path: Optional[str] = None) -> Optional[Any]:
        """
        Create animation of solution evolution.
        
        Args:
            U_sequence: List of solution arrays over time
            x_grid: Spatial grid
            title: Animation title
            save_path: Path to save animation
            
        Returns:
            Animation object (matplotlib) or HTML string (plotly)
        """
        logger = get_logger("mfg_pde.utils.visualization")
        
        if PLOTLY_AVAILABLE:
            return VisualizationUtils._create_plotly_animation(
                U_sequence, x_grid, title, save_path)
        elif MATPLOTLIB_AVAILABLE:
            return VisualizationUtils._create_matplotlib_animation(
                U_sequence, x_grid, title, save_path)
        else:
            raise MathematicalVisualizationError("No animation backend available")
    
    @staticmethod
    def _create_plotly_animation(U_sequence, x_grid, title, save_path):
        """Create animation using Plotly."""
        frames = []
        
        for i, U in enumerate(U_sequence):
            frame = go.Frame(
                data=[go.Scatter(x=x_grid, y=U, mode='lines', name=f'Step {i}')],
                name=str(i)
            )
            frames.append(frame)
        
        fig = go.Figure(
            data=[go.Scatter(x=x_grid, y=U_sequence[0], mode='lines')],
            frames=frames
        )
        
        fig.update_layout(
            title=title,
            xaxis_title="Space (x)",
            yaxis_title="U(x)",
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {'label': 'Play', 'method': 'animate', 'args': [None]},
                    {'label': 'Pause', 'method': 'animate', 'args': [[None], {'frame': {'duration': 0}}]}
                ]
            }]
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    @staticmethod
    def _create_matplotlib_animation(U_sequence, x_grid, title, save_path):
        """Create animation using matplotlib."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        line, = ax.plot(x_grid, U_sequence[0], 'b-', linewidth=2)
        ax.set_xlim(x_grid[0], x_grid[-1])
        ax.set_ylim(min(U.min() for U in U_sequence), max(U.max() for U in U_sequence))
        ax.set_xlabel('Space (x)')
        ax.set_ylabel('U(x)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        def animate(frame):
            line.set_ydata(U_sequence[frame])
            return line,
        
        anim = FuncAnimation(fig, animate, frames=len(U_sequence), 
                           interval=100, blit=True, repeat=True)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=10)
        
        return anim
    
    @staticmethod
    def save_interactive_report(results: Dict[str, Any],
                              output_path: str,
                              title: str = "MFG Analysis Report") -> str:
        """
        Save comprehensive interactive HTML report.
        
        Args:
            results: Dictionary containing analysis results
            output_path: Path to save HTML report
            title: Report title
            
        Returns:
            Path to saved report
        """
        logger = get_logger("mfg_pde.utils.visualization")
        
        if not PLOTLY_AVAILABLE:
            raise MathematicalVisualizationError("Interactive reports require Plotly")
        
        # Create HTML report template
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .section {{ margin: 20px 0; }}
                .plot-container {{ margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{title}</h1>
                <p>Generated by MFG_PDE Advanced Visualization Module</p>
            </div>
        """
        
        # Add plots from results
        plot_id = 0
        for section, data in results.items():
            html_content += f'<div class="section"><h2>{section}</h2>'
            
            if isinstance(data, dict) and 'figure' in data:
                plot_id += 1
                html_content += f'<div id="plot{plot_id}" class="plot-container"></div>'
                html_content += f"""
                <script>
                    Plotly.newPlot('plot{plot_id}', {data['figure'].to_json()});
                </script>
                """
            
            html_content += '</div>'
        
        html_content += """
        </body>
        </html>
        """
        
        # Save report
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Interactive report saved to {output_path}")
        return output_path


# Convenience functions for quick plotting
def quick_plot_solution(U: np.ndarray,
                       M: np.ndarray,
                       x_grid: np.ndarray,
                       t_grid: np.ndarray,
                       backend: str = "auto",
                       save_path: Optional[str] = None) -> Any:
    """
    Quick function to plot MFG solution.
    
    Args:
        U: Value function
        M: Density function
        x_grid: Spatial grid
        t_grid: Temporal grid
        backend: Visualization backend
        save_path: Save path
        
    Returns:
        Figure object
    """
    visualizer = MFGVisualizer(backend=backend)
    return visualizer.plot_mfg_solution(U, M, x_grid, t_grid, save_path=save_path)


def quick_plot_convergence(convergence_data: Dict[str, List[float]],
                          backend: str = "auto",
                          save_path: Optional[str] = None) -> Any:
    """
    Quick function to plot convergence history.
    
    Args:
        convergence_data: Convergence metrics
        backend: Visualization backend
        save_path: Save path
        
    Returns:
        Figure object
    """
    visualizer = MFGVisualizer(backend=backend)
    return visualizer.plot_convergence_history(convergence_data, save_path=save_path)


# Example and demonstration functions
def demo_visualization_capabilities():
    """Demonstrate the visualization capabilities."""
    logger = get_logger("mfg_pde.utils.visualization")
    logger.info("Starting visualization capabilities demonstration")
    
    # Create sample data
    x_grid = np.linspace(0, 1, 50)
    t_grid = np.linspace(0, 1, 20)
    X, T = np.meshgrid(x_grid, t_grid)
    
    # Sample MFG solution
    U = np.sin(np.pi * X) * np.exp(-T)
    M = np.exp(-10 * (X - 0.5)**2) * (1 + 0.5 * T)
    
    # Sample convergence data
    convergence_data = {
        'L2_error': [1e-1 * (0.8 ** i) for i in range(25)],
        'residual': [5e-2 * (0.85 ** i) for i in range(25)],
        'mass_conservation': [1e-3 * (0.9 ** i) for i in range(25)]
    }
    
    try:
        # Test both backends if available
        for backend in ['plotly', 'matplotlib']:
            try:
                logger.info(f"Testing {backend} backend")
                visualizer = MFGVisualizer(backend=backend)
                
                # Test solution plot
                fig1 = visualizer.plot_mfg_solution(
                    U.T, M.T, x_grid, t_grid,
                    title=f"MFG Solution ({backend})",
                    show=False
                )
                
                # Test convergence plot
                fig2 = visualizer.plot_convergence_history(
                    convergence_data,
                    title=f"Convergence History ({backend})",
                    show=False
                )
                
                logger.info(f"{backend} backend working correctly")
                
            except MathematicalVisualizationError as e:
                logger.warning(f"{backend} backend not available: {e}")
        
        # Test dashboard (Plotly only)
        if PLOTLY_AVAILABLE:
            dashboard = SolverMonitoringDashboard()
            
            # Add some sample data
            for i in range(20):
                dashboard.add_metric('error', 1e-2 * (0.9 ** i), i)
                dashboard.add_metric('residual', 5e-3 * (0.85 ** i), i)
                dashboard.add_performance_data('iteration', 0.1 + 0.01 * i)
            
            dashboard_fig = dashboard.create_dashboard(show=False)
            logger.info("Dashboard creation working correctly")
        
        logger.info("Visualization demonstration completed successfully")
        
    except Exception as e:
        logger.error(f"Error in visualization demonstration: {e}")
        raise


if __name__ == "__main__":
    demo_visualization_capabilities()