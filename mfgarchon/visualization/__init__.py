"""
Visualization utilities for MFGarchon.

Provides convergence and solver diagnostics plotting (matplotlib).
For solution visualization, use matplotlib/plotly/pyvista directly.
"""

from .convergence_plots import (
    plot_convergence_rate,
    plot_convergence_summary,
    plot_distribution_evolution,
    plot_error_history,
    plot_from_monitor,
    plot_mass_history,
    plot_multi_error_history,
    plot_wasserstein_history,
)

__all__ = [
    "plot_convergence_rate",
    "plot_convergence_summary",
    "plot_distribution_evolution",
    "plot_error_history",
    "plot_from_monitor",
    "plot_mass_history",
    "plot_multi_error_history",
    "plot_wasserstein_history",
]
