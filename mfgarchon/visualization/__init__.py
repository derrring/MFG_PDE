"""
Visualization utilities for MFGarchon.

Provides convergence plotting and basic MFG solution visualization.
For interactive/3D visualization, use Plotly or pyvista directly.
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
from .legacy_plotting import (
    myplot3d,
    plot_convergence,
    plot_results,
)

__all__ = [
    # Convergence plotting
    "plot_convergence_rate",
    "plot_convergence_summary",
    "plot_distribution_evolution",
    "plot_error_history",
    "plot_from_monitor",
    "plot_mass_history",
    "plot_multi_error_history",
    "plot_wasserstein_history",
    # Basic solution plotting
    "myplot3d",
    "plot_convergence",
    "plot_results",
]
