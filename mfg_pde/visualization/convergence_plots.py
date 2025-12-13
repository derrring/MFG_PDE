#!/usr/bin/env python3
"""
Convergence visualization utilities.

Provides standalone plotting functions for convergence analysis that work
with any PDE solver. Functions accept data arrays directly (not monitors),
making them flexible and testable.

Essential convergence plots:
- plot_error_history: Track L2/L1/Linf errors vs iteration
- plot_distribution_evolution: Visualize m(t,x) convergence
- plot_wasserstein_history: Distribution convergence metric
- plot_mass_history: Track total mass over time
- plot_convergence_rate: Estimate numerical order (log-log)
- plot_convergence_summary: Multi-panel overview
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    Figure = Any  # type: ignore[misc, assignment]


def _check_matplotlib():
    """Check if matplotlib is available."""
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for convergence plotting. Install with: pip install matplotlib")


# =============================================================================
# ERROR HISTORY PLOTS
# =============================================================================


def plot_error_history(
    errors: list[float] | np.ndarray,
    tolerance: float | None = None,
    label: str = "L2 Error",
    title: str = "Convergence History",
    xlabel: str = "Iteration",
    ylabel: str = "Error",
    log_scale: bool = True,
    save_path: str | None = None,
    ax: Any | None = None,
) -> Figure | None:
    """
    Plot error history with optional tolerance line.

    Args:
        errors: List or array of error values per iteration
        tolerance: Optional tolerance line to display
        label: Legend label for the error curve
        title: Plot title
        xlabel, ylabel: Axis labels
        log_scale: Use logarithmic y-axis (recommended for convergence)
        save_path: Optional path to save figure
        ax: Optional matplotlib axes to plot on

    Returns:
        Figure object if ax not provided, else None
    """
    _check_matplotlib()

    created_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.get_figure()

    iterations = np.arange(1, len(errors) + 1)

    if log_scale:
        ax.semilogy(iterations, errors, "b-", linewidth=1.5, label=label)
    else:
        ax.plot(iterations, errors, "b-", linewidth=1.5, label=label)

    if tolerance is not None:
        ax.axhline(y=tolerance, color="r", linestyle="--", linewidth=1, label=f"Tolerance ({tolerance:.0e})")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path and created_fig:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig if created_fig else None


def plot_multi_error_history(
    error_dict: dict[str, list[float] | np.ndarray],
    tolerances: dict[str, float] | None = None,
    title: str = "Convergence History",
    log_scale: bool = True,
    save_path: str | None = None,
) -> Figure:
    """
    Plot multiple error histories on the same axes.

    Args:
        error_dict: Dictionary mapping label -> error array
        tolerances: Optional dictionary mapping label -> tolerance
        title: Plot title
        log_scale: Use logarithmic y-axis
        save_path: Optional path to save figure

    Returns:
        Figure object
    """
    _check_matplotlib()

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]
    tolerances = tolerances or {}

    for i, (label, errors) in enumerate(error_dict.items()):
        iterations = np.arange(1, len(errors) + 1)
        color = colors[i % len(colors)]

        if log_scale:
            ax.semilogy(iterations, errors, "-", color=color, linewidth=1.5, label=label)
        else:
            ax.plot(iterations, errors, "-", color=color, linewidth=1.5, label=label)

        if label in tolerances:
            ax.axhline(y=tolerances[label], color=color, linestyle="--", linewidth=1, alpha=0.7)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Error")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# =============================================================================
# DISTRIBUTION EVOLUTION PLOTS
# =============================================================================


def plot_distribution_evolution(
    distributions: np.ndarray,
    x_grid: np.ndarray,
    iterations: list[int] | None = None,
    title: str = "Distribution Evolution",
    xlabel: str = "x",
    ylabel: str = "m(x)",
    save_path: str | None = None,
) -> Figure:
    """
    Plot evolution of distribution over iterations.

    Args:
        distributions: 2D array (n_iter, n_points) of distributions
        x_grid: Spatial grid points
        iterations: Which iterations to plot (default: 5 evenly spaced)
        title: Plot title
        xlabel, ylabel: Axis labels
        save_path: Optional path to save figure

    Returns:
        Figure object
    """
    _check_matplotlib()

    fig, ax = plt.subplots(figsize=(10, 6))

    n_iter = distributions.shape[0]
    if iterations is None:
        # Select 5 evenly spaced iterations including first and last
        iterations = [0, n_iter // 4, n_iter // 2, 3 * n_iter // 4, n_iter - 1]
        iterations = sorted(set(iterations))  # Remove duplicates

    colors = plt.cm.viridis(np.linspace(0, 1, len(iterations)))  # type: ignore[attr-defined]

    for i, iter_idx in enumerate(iterations):
        if iter_idx < n_iter:
            ax.plot(x_grid, distributions[iter_idx], color=colors[i], linewidth=1.5, label=f"Iter {iter_idx}")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# =============================================================================
# WASSERSTEIN / DISTRIBUTION METRIC PLOTS
# =============================================================================


def plot_wasserstein_history(
    wasserstein_distances: list[float] | np.ndarray,
    tolerance: float | None = None,
    title: str = "Distribution Convergence (Wasserstein)",
    save_path: str | None = None,
) -> Figure:
    """
    Plot Wasserstein distance history.

    Args:
        wasserstein_distances: Wasserstein distances per iteration
        tolerance: Optional tolerance line
        title: Plot title
        save_path: Optional path to save figure

    Returns:
        Figure object
    """
    return plot_error_history(
        wasserstein_distances,
        tolerance=tolerance,
        label="Wasserstein Distance",
        title=title,
        ylabel="Wasserstein Distance",
        save_path=save_path,
    )


# =============================================================================
# MASS TRACKING PLOTS
# =============================================================================


def plot_mass_history(
    mass_values: list[float] | np.ndarray,
    title: str = "Mass History",
    xlabel: str = "Iteration",
    ylabel: str = "Total Mass",
    show_initial: bool = True,
    save_path: str | None = None,
) -> Figure:
    """
    Plot total mass over iterations.

    Args:
        mass_values: Total mass per iteration
        title: Plot title
        xlabel, ylabel: Axis labels
        show_initial: Show horizontal line at initial mass
        save_path: Optional path to save figure

    Returns:
        Figure object
    """
    _check_matplotlib()

    fig, ax = plt.subplots(figsize=(8, 5))

    iterations = np.arange(1, len(mass_values) + 1)
    ax.plot(iterations, mass_values, "g-", linewidth=1.5, label="Total Mass")

    if show_initial and len(mass_values) > 0:
        ax.axhline(y=mass_values[0], color="r", linestyle="--", linewidth=1, alpha=0.7, label="Initial Mass")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# =============================================================================
# CONVERGENCE RATE ANALYSIS
# =============================================================================


def plot_convergence_rate(
    errors: list[float] | np.ndarray,
    grid_sizes: list[float] | np.ndarray | None = None,
    title: str = "Convergence Rate Analysis",
    expected_order: float | None = None,
    save_path: str | None = None,
) -> tuple[Figure, float]:
    """
    Plot convergence rate on log-log scale and estimate order.

    Args:
        errors: Error values at different resolutions
        grid_sizes: Grid spacing (h values). If None, uses iteration index.
        title: Plot title
        expected_order: Optional expected order to show as reference
        save_path: Optional path to save figure

    Returns:
        (Figure, estimated_order)
    """
    _check_matplotlib()

    fig, ax = plt.subplots(figsize=(8, 6))

    errors = np.array(errors)
    if grid_sizes is None:
        grid_sizes = np.arange(1, len(errors) + 1)
    else:
        grid_sizes = np.array(grid_sizes)

    # Filter out zeros/nans
    valid = (errors > 0) & np.isfinite(errors) & (grid_sizes > 0)
    x = np.log(grid_sizes[valid])
    y = np.log(errors[valid])

    # Plot data
    ax.loglog(grid_sizes[valid], errors[valid], "bo-", markersize=8, linewidth=1.5, label="Computed Error")

    # Fit line to estimate order
    if len(x) >= 2:
        coeffs = np.polyfit(x, y, 1)
        estimated_order = coeffs[0]

        # Plot fitted line
        x_fit = np.linspace(x.min(), x.max(), 100)
        y_fit = np.polyval(coeffs, x_fit)
        ax.loglog(np.exp(x_fit), np.exp(y_fit), "r--", linewidth=1, label=f"Fitted (order={estimated_order:.2f})")
    else:
        estimated_order = np.nan

    # Plot expected order reference
    if expected_order is not None and len(x) >= 2:
        y_ref = y[0] + expected_order * (x - x[0])
        ax.loglog(np.exp(x), np.exp(y_ref), "g:", linewidth=1, alpha=0.7, label=f"Expected O(h^{expected_order})")

    ax.set_xlabel("Grid Size (h)")
    ax.set_ylabel("Error")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, estimated_order


# =============================================================================
# SUMMARY PLOTS
# =============================================================================


def plot_convergence_summary(
    u_errors: list[float] | np.ndarray,
    m_errors: list[float] | np.ndarray | None = None,
    wasserstein_distances: list[float] | np.ndarray | None = None,
    mass_values: list[float] | np.ndarray | None = None,
    tolerances: dict[str, float] | None = None,
    title: str = "Convergence Summary",
    save_path: str | None = None,
) -> Figure:
    """
    Create multi-panel convergence summary.

    Args:
        u_errors: Value function errors
        m_errors: Distribution errors (optional)
        wasserstein_distances: Wasserstein distances (optional)
        mass_values: Total mass values (optional)
        tolerances: Dict of tolerance values for horizontal lines
        title: Overall figure title
        save_path: Optional path to save figure

    Returns:
        Figure object
    """
    _check_matplotlib()

    tolerances = tolerances or {}

    # Determine number of subplots
    n_plots = 1  # Always have u_errors
    if m_errors is not None:
        n_plots += 1
    if wasserstein_distances is not None:
        n_plots += 1
    if mass_values is not None:
        n_plots += 1

    # Create subplot grid
    if n_plots <= 2:
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
        if n_plots == 1:
            axes = [axes]
    else:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.flatten()

    plot_idx = 0

    # U errors
    plot_error_history(
        u_errors,
        tolerance=tolerances.get("u"),
        label="U Error",
        title="Value Function Convergence",
        ax=axes[plot_idx],
    )
    plot_idx += 1

    # M errors
    if m_errors is not None:
        plot_error_history(
            m_errors,
            tolerance=tolerances.get("m"),
            label="M Error",
            title="Distribution Convergence",
            ax=axes[plot_idx],
        )
        plot_idx += 1

    # Wasserstein
    if wasserstein_distances is not None:
        # Filter NaN values
        valid_wass = [w for w in wasserstein_distances if not np.isnan(w)]
        if valid_wass:
            plot_error_history(
                valid_wass,
                tolerance=tolerances.get("wasserstein"),
                label="Wasserstein",
                title="Wasserstein Distance",
                ax=axes[plot_idx],
            )
        plot_idx += 1

    # Mass
    if mass_values is not None:
        iterations = np.arange(1, len(mass_values) + 1)
        axes[plot_idx].plot(iterations, mass_values, "g-", linewidth=1.5)
        if len(mass_values) > 0:
            axes[plot_idx].axhline(y=mass_values[0], color="r", linestyle="--", alpha=0.7)
        axes[plot_idx].set_xlabel("Iteration")
        axes[plot_idx].set_ylabel("Total Mass")
        axes[plot_idx].set_title("Mass History")
        axes[plot_idx].grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(plot_idx + 1, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# =============================================================================
# UTILITY FOR MONITOR INTEGRATION
# =============================================================================


def plot_from_monitor(
    monitor: Any,
    save_path: str | None = None,
) -> Figure:
    """
    Create convergence plot from a DistributionConvergenceMonitor.

    Args:
        monitor: DistributionConvergenceMonitor instance
        save_path: Optional path to save figure

    Returns:
        Figure object
    """
    # Get plot data from monitor
    if hasattr(monitor, "get_plot_data"):
        data = monitor.get_plot_data()
    elif hasattr(monitor, "convergence_history"):
        # Fallback for direct attribute access
        data = {
            "iterations": [d["iteration"] for d in monitor.convergence_history],
            "u_errors": [d["u_l2_error"] for d in monitor.convergence_history],
            "wasserstein_distances": [d.get("wasserstein_distance", np.nan) for d in monitor.convergence_history],
        }
    else:
        raise ValueError("Monitor must have get_plot_data() method or convergence_history attribute")

    # Get tolerances
    tolerances = {}
    if hasattr(monitor, "u_magnitude_tol"):
        tolerances["u"] = monitor.u_magnitude_tol
    if hasattr(monitor, "wasserstein_tol"):
        tolerances["wasserstein"] = monitor.wasserstein_tol

    return plot_convergence_summary(
        u_errors=data["u_errors"],
        wasserstein_distances=data.get("wasserstein_distances"),
        tolerances=tolerances,
        title="Convergence History",
        save_path=save_path,
    )
