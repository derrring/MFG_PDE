"""
Mathematical Visualization for MFG_PDE.

This module provides advanced mathematical plotting capabilities with LaTeX support,
migrated and enhanced from utils/mathematical_visualization.py.

Features:
- Professional mathematical notation
- Cross-platform LaTeX compatibility
- Publication-quality output
- Integration with modern visualization system
"""

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Plotly imports with LaTeX support
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Matplotlib imports with LaTeX configuration
try:
    from mpl_toolkits.mplot3d import Axes3D

    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    # Configure matplotlib for cross-platform compatibility
    rcParams["text.usetex"] = False  # Avoid LaTeX dependency
    rcParams["font.family"] = "sans-serif"
    rcParams["font.sans-serif"] = [
        "Arial",
        "DejaVu Sans",
        "Liberation Sans",
        "Helvetica",
        "sans-serif",
    ]
    rcParams["mathtext.fontset"] = "dejavusans"
    rcParams["axes.formatter.use_mathtext"] = True

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class MathematicalPlotter:
    """
    Mathematical plotting class with professional notation support.

    Enhanced and migrated from utils/mathematical_visualization.py
    """

    def __init__(self, backend: str = "auto", use_latex: bool = False):
        """
        Initialize mathematical plotter.

        Args:
            backend: Visualization backend ("plotly", "matplotlib", "auto")
            use_latex: Whether to use LaTeX rendering (matplotlib only)
        """
        self.backend = self._validate_backend(backend)
        self.use_latex = use_latex and MATPLOTLIB_AVAILABLE

        if self.use_latex:
            self._setup_latex()

    def _validate_backend(self, backend: str) -> str:
        """Validate and set visualization backend."""
        if backend == "auto":
            if PLOTLY_AVAILABLE:
                return "plotly"
            elif MATPLOTLIB_AVAILABLE:
                return "matplotlib"
            else:
                raise ImportError("No visualization backend available")
        return backend

    def _setup_latex(self):
        """Setup LaTeX rendering for matplotlib."""
        try:
            rcParams["text.usetex"] = True
            rcParams["font.family"] = "serif"
            rcParams["font.serif"] = ["Computer Modern Roman"]
        except:
            warnings.warn("LaTeX setup failed, falling back to mathtext")
            self.use_latex = False

    def plot_mathematical_function(
        self,
        x: np.ndarray,
        y: np.ndarray,
        title: str = "Mathematical Function",
        xlabel: str = r"$x$",
        ylabel: str = r"$f(x)$",
        save_path: Optional[str] = None,
    ) -> Any:
        """
        Plot mathematical function with proper notation.

        Args:
            x: Independent variable values
            y: Function values
            title: Plot title with LaTeX support
            xlabel: X-axis label with LaTeX
            ylabel: Y-axis label with LaTeX
            save_path: Path to save the plot

        Returns:
            Figure object
        """
        if self.backend == "plotly":
            return self._plot_function_plotly(x, y, title, xlabel, ylabel, save_path)
        else:
            return self._plot_function_matplotlib(x, y, title, xlabel, ylabel, save_path)

    def _plot_function_plotly(self, x, y, title, xlabel, ylabel, save_path):
        """Plot function using Plotly with LaTeX support."""
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(width=2, color="blue"), name="f(x)"))

        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=16)),
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            template="plotly_white",
            width=800,
            height=600,
            font=dict(size=14),
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def _plot_function_matplotlib(self, x, y, title, xlabel, ylabel, save_path):
        """Plot function using matplotlib with LaTeX support."""
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(x, y, "b-", linewidth=2, label="f(x)")
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_title(title, fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_mfg_density_evolution(
        self,
        x_grid: np.ndarray,
        t_grid: np.ndarray,
        density: np.ndarray,
        title: str = "Density Evolution $m(t,x)$",
        save_path: Optional[str] = None,
    ) -> Any:
        """
        Plot MFG density evolution with mathematical notation.

        Args:
            x_grid: Spatial grid points
            t_grid: Time grid points
            density: Density values [time, space]
            title: Plot title
            save_path: Save path

        Returns:
            Figure object
        """
        if self.backend == "plotly":
            return self._plot_density_plotly(x_grid, t_grid, density, title, save_path)
        else:
            return self._plot_density_matplotlib(x_grid, t_grid, density, title, save_path)

    def _plot_density_plotly(self, x_grid, t_grid, density, title, save_path):
        """Plot density evolution using Plotly."""
        fig = go.Figure(
            data=go.Heatmap(
                z=density,
                x=x_grid,
                y=t_grid,
                colorscale="Viridis",
                colorbar=dict(title="m(t,x)"),
            )
        )

        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis_title="Position x",
            yaxis_title="Time t",
            width=800,
            height=600,
        )

        if save_path:
            fig.write_html(save_path)

        return fig

    def _plot_density_matplotlib(self, x_grid, t_grid, density, title, save_path):
        """Plot density evolution using matplotlib."""
        fig, ax = plt.subplots(figsize=(10, 8))

        X, T = np.meshgrid(x_grid, t_grid)

        im = ax.contourf(X, T, density, levels=50, cmap="viridis")
        ax.set_xlabel(r"Position $x$", fontsize=14)
        ax.set_ylabel(r"Time $t$", fontsize=14)
        ax.set_title(title, fontsize=16)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(r"$m(t,x)$", fontsize=14)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_phase_portrait(
        self,
        x: np.ndarray,
        y: np.ndarray,
        xlabel: str = r"$x$",
        ylabel: str = r"$\dot{x}$",
        title: str = "Phase Portrait",
        save_path: Optional[str] = None,
    ) -> Any:
        """
        Plot phase portrait with mathematical notation.

        Args:
            x: State variable
            y: State derivative
            xlabel: X-axis label
            ylabel: Y-axis label
            title: Plot title
            save_path: Save path

        Returns:
            Figure object
        """
        if self.backend == "plotly":
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Trajectory"))
            fig.update_layout(
                title=title,
                xaxis_title=xlabel,
                yaxis_title=ylabel,
                template="plotly_white",
            )

            if save_path:
                fig.write_html(save_path)
            return fig
        else:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.plot(x, y, "b-", linewidth=2)
            ax.set_xlabel(xlabel, fontsize=14)
            ax.set_ylabel(ylabel, fontsize=14)
            ax.set_title(title, fontsize=16)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
            return fig


class MFGMathematicalVisualizer:
    """
    Specialized visualizer for MFG mathematical analysis.

    Combines mathematical notation with MFG-specific visualization patterns.
    """

    def __init__(self, backend: str = "auto"):
        """Initialize MFG mathematical visualizer."""
        self.plotter = MathematicalPlotter(backend)
        self.backend = backend

    def visualize_hjb_equation(
        self,
        x_grid: np.ndarray,
        u: np.ndarray,
        du_dx: np.ndarray,
        title: str = "HJB Solution Analysis",
    ) -> Any:
        """
        Visualize Hamilton-Jacobi-Bellman equation solution.

        Args:
            x_grid: Spatial grid
            u: Value function u(x)
            du_dx: Value function gradient ∂u/∂x
            title: Plot title

        Returns:
            Figure object
        """
        if self.backend == "plotly":
            fig = make_subplots(
                rows=2,
                cols=1,
                subplot_titles=[r"Value Function u(x)", r"Gradient ∂u/∂x"],
            )

            fig.add_trace(go.Scatter(x=x_grid, y=u, mode="lines", name="u(x)"), row=1, col=1)
            fig.add_trace(go.Scatter(x=x_grid, y=du_dx, mode="lines", name="∂u/∂x"), row=2, col=1)

            fig.update_layout(title=title, height=800)
            return fig
        else:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

            ax1.plot(x_grid, u, "b-", linewidth=2)
            ax1.set_ylabel(r"$u(x)$", fontsize=14)
            ax1.set_title("Value Function", fontsize=14)
            ax1.grid(True, alpha=0.3)

            ax2.plot(x_grid, du_dx, "r-", linewidth=2)
            ax2.set_xlabel(r"Position $x$", fontsize=14)
            ax2.set_ylabel(r"$\partial u/\partial x$", fontsize=14)
            ax2.set_title("Value Function Gradient", fontsize=14)
            ax2.grid(True, alpha=0.3)

            plt.suptitle(title, fontsize=16)
            plt.tight_layout()
            return fig

    def visualize_fokker_planck_solution(
        self,
        x_grid: np.ndarray,
        t_grid: np.ndarray,
        density: np.ndarray,
        current: np.ndarray,
        title: str = "Fokker-Planck Solution",
    ) -> Any:
        """
        Visualize Fokker-Planck equation solution.

        Args:
            x_grid: Spatial grid
            t_grid: Time grid
            density: Density m(t,x)
            current: Probability current j(t,x)
            title: Plot title

        Returns:
            Figure object
        """
        if self.backend == "plotly":
            fig = make_subplots(rows=1, cols=2, subplot_titles=["Density m(t,x)", "Current j(t,x)"])

            fig.add_trace(
                go.Heatmap(z=density, x=x_grid, y=t_grid, colorscale="Viridis"),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Heatmap(z=current, x=x_grid, y=t_grid, colorscale="RdBu"),
                row=1,
                col=2,
            )

            fig.update_layout(title=title, height=600, width=1200)
            return fig
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            X, T = np.meshgrid(x_grid, t_grid)

            im1 = ax1.contourf(X, T, density, levels=50, cmap="viridis")
            ax1.set_xlabel(r"Position $x$", fontsize=14)
            ax1.set_ylabel(r"Time $t$", fontsize=14)
            ax1.set_title(r"Density $m(t,x)$", fontsize=14)
            plt.colorbar(im1, ax=ax1)

            im2 = ax2.contourf(X, T, current, levels=50, cmap="RdBu")
            ax2.set_xlabel(r"Position $x$", fontsize=14)
            ax2.set_ylabel(r"Time $t$", fontsize=14)
            ax2.set_title(r"Current $j(t,x)$", fontsize=14)
            plt.colorbar(im2, ax=ax2)

            plt.suptitle(title, fontsize=16)
            plt.tight_layout()
            return fig


# Convenience functions
def plot_mathematical_function(
    x: np.ndarray,
    y: np.ndarray,
    title: str = "Mathematical Function",
    backend: str = "auto",
) -> Any:
    """Quick mathematical function plotting."""
    plotter = MathematicalPlotter(backend)
    return plotter.plot_mathematical_function(x, y, title)


def plot_mfg_density(x_grid: np.ndarray, t_grid: np.ndarray, density: np.ndarray, backend: str = "auto") -> Any:
    """Quick MFG density plotting."""
    plotter = MathematicalPlotter(backend)
    return plotter.plot_mfg_density_evolution(x_grid, t_grid, density)


def create_mathematical_visualizer(backend: str = "auto") -> MFGMathematicalVisualizer:
    """Create MFG mathematical visualizer."""
    return MFGMathematicalVisualizer(backend)
