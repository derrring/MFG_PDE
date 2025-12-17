"""
Legacy Plotting Functions for MFG_PDE.

This module contains matplotlib-based plotting functions migrated from utils/
for backward compatibility and basic visualization needs.

Migrated from:
- utils/plot_utils.py
- Basic plotting functions from utils/advanced_visualization.py
"""

from __future__ import annotations

import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter, LinearLocator

# Import optional dependencies from parent module (centralized imports)
from . import PLOTLY_AVAILABLE, go, px

# Configure matplotlib for cross-platform compatibility
plt.rcParams["text.usetex"] = False
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Arial",
    "DejaVu Sans",
    "Liberation Sans",
    "Helvetica",
    "sans-serif",
]
plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams["axes.formatter.use_mathtext"] = True


def myplot3d(X, Y, Z, title="Surface Plot"):
    """
    Create 3D surface plot with matplotlib.

    Legacy function migrated from utils/plot_utils.py
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    X, Y = np.meshgrid(X, Y)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.get_cmap("coolwarm"), linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))
    ax.set_xlabel("x")
    ax.set_ylabel("time")
    ax.set_title(title)
    ax.view_init(40, -135)
    plt.show()


def plot_convergence(iterations_run, l2disturel_u, l2disturel_m, solver_name="Solver"):
    """
    Plot convergence history for U and M.

    Legacy function migrated from utils/plot_utils.py
    """
    iterSpace = np.arange(1, iterations_run + 1)

    plt.figure()
    plt.semilogy(iterSpace, l2disturel_u)
    plt.xlabel("Iteration")
    plt.ylabel("$||u_{new}-u_{old}||_{rel}$")
    plt.title(f"Convergence of U ({solver_name})")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.semilogy(iterSpace, l2disturel_m)
    plt.xlabel("Iteration")
    plt.ylabel("$||m_{new}-m_{old}||_{rel}$")
    plt.title(f"Convergence of M ({solver_name})")
    plt.grid(True)
    plt.show()


def plot_results(problem, u, m, solver_name="Solver", prefix=None):
    """
    Plot MFG results with 3D surfaces and analysis plots.

    Legacy function migrated from utils/plot_utils.py
    """
    # Get spatial grid from geometry (1D), temporal from problem
    xSpace = problem.geometry.coordinates[0]  # 1D spatial coordinates
    tSpace = problem.tSpace  # Temporal grid is on problem, not geometry
    dx = problem.geometry.get_grid_spacing()[0]

    # Subsample for plotting if desired
    kx = 2  # Example subsampling
    kt = 5  # Example subsampling
    xSpacecut = xSpace[::kx]
    tSpacecut = tSpace[::kt]
    ucut = u[::kt, ::kx]
    mcut = m[::kt, ::kx]

    # Plot U
    myplot3d(xSpacecut, tSpacecut, ucut, title=f"Evolution of U ({solver_name})")

    # Plot M
    myplot3d(xSpacecut, tSpacecut, mcut, title=f"Evolution of M ({solver_name})")

    # Final density plot
    plt.figure()
    plt.plot(xSpace, m[-1, :])
    plt.xlabel("x")
    plt.ylabel("m(T,x)")
    plt.title(f"Final Density m(T,x) ({solver_name})")
    plt.grid(True)
    plt.show()

    # Mass conservation plot
    plt.figure()
    mtot = np.sum(m * dx, axis=1)
    plt.plot(tSpace, mtot)
    plt.xlabel("t")
    plt.ylabel("Total Mass $\\int m(t)$")
    plt.title(f"Total Mass ({solver_name})")
    plt.ylim(
        min(0.9, np.min(mtot) * 0.98 if mtot.size > 0 else 0.9),
        max(1.1, np.max(mtot) * 1.02 if mtot.size > 0 else 1.1),
    )
    plt.grid(True)
    plt.show()


# Modern wrapper functions that delegate to the new visualization system
def modern_plot_mfg_solution(
    U: np.ndarray,
    M: np.ndarray,
    x_grid: np.ndarray,
    t_grid: np.ndarray,
    title: str = "MFG Solution",
    backend: str = "auto",
) -> Any:
    """
    Modern wrapper that uses the new visualization system.

    Recommended alternative to legacy functions.
    """
    try:
        from .interactive_plots import create_visualization_manager

        viz_manager = create_visualization_manager()

        # Create 2D density evolution plot
        density_plot = viz_manager.create_2d_density_plot(x_grid, t_grid, M, backend, f"{title} - Density m(t,x)")

        # For 3D surface plots
        if hasattr(viz_manager, "create_3d_surface_plot"):
            viz_manager.create_3d_surface_plot(x_grid, t_grid, M, "density", f"{title} - 3D Surface")

        return density_plot

    except ImportError:
        warnings.warn("Modern visualization system not available, using legacy matplotlib", stacklevel=2)
        # Fall back to legacy plotting
        myplot3d(x_grid, t_grid, M, title)


def modern_plot_convergence(
    convergence_data: dict[str, list[float]],
    title: str = "Convergence History",
    tolerances: dict[str, float] | None = None,
    save_path: str | None = None,
    backend: str = "auto",
) -> Any:
    """
    Modern wrapper for convergence plotting with tolerance lines.

    Args:
        convergence_data: Dictionary with error series names as keys and error lists as values
        title: Plot title
        tolerances: Dictionary with tolerance values for each error series
        save_path: Optional path to save the plot
        backend: Visualization backend to use

    Returns:
        Figure object

    Recommended alternative to legacy plot_convergence.
    """
    try:
        from .interactive_plots import create_visualization_manager

        viz_manager = create_visualization_manager(prefer_plotly=(backend != "matplotlib"))

        # Create a comprehensive convergence plot
        if (
            viz_manager.plotly_viz
            and backend != "matplotlib"
            and PLOTLY_AVAILABLE
            and go is not None
            and px is not None
        ):
            # Use Plotly for advanced interactive plotting
            fig = go.Figure()

            colors = px.colors.qualitative.Set1
            for i, (metric_name, values) in enumerate(convergence_data.items()):
                iterations = list(range(1, len(values) + 1))

                fig.add_trace(
                    go.Scatter(
                        x=iterations,
                        y=values,
                        mode="lines+markers",
                        name=metric_name,
                        line={"color": colors[i % len(colors)], "width": 2},
                        marker={"size": 4},
                    )
                )

                # Add tolerance line if provided
                if tolerances and metric_name in tolerances:
                    fig.add_hline(
                        y=tolerances[metric_name],
                        line_dash="dash",
                        line_color=colors[i % len(colors)],
                        annotation_text=f"{metric_name} tolerance",
                    )

            fig.update_layout(
                title=title,
                xaxis_title="Iteration",
                yaxis_title="Error/Residual",
                yaxis_type="log",
                template="plotly_white",
                hovermode="x unified",
                showlegend=True,
            )

            if save_path:
                fig.write_html(save_path)

            return fig

    except Exception as e:
        warnings.warn(f"Modern visualization failed ({e}), using legacy matplotlib", stacklevel=2)

    # Fall back to matplotlib-based plotting
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = cm.get_cmap("Set1")(np.linspace(0, 1, len(convergence_data)))

    for i, (metric_name, values) in enumerate(convergence_data.items()):
        iterations = list(range(1, len(values) + 1))
        ax.semilogy(
            iterations,
            values,
            "o-",
            label=metric_name,
            color=colors[i],
            linewidth=2,
            markersize=4,
        )

        # Add tolerance line if provided
        if tolerances and metric_name in tolerances:
            ax.axhline(
                y=tolerances[metric_name],
                color=colors[i],
                linestyle="--",
                alpha=0.7,
                label=f"{metric_name} tolerance",
            )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Error/Residual")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.tight_layout()
    return fig


# Backward compatibility aliases
legacy_myplot3d = myplot3d
legacy_plot_convergence = plot_convergence
legacy_plot_results = plot_results
