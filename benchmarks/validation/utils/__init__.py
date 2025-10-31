"""
Utility functions for validation benchmarks.

Provides common utilities for:
- Solver comparison (FDM vs GFDM)
- Metric computation (L² error, mass conservation, convergence rate)
- Visualization (side-by-side plots)
"""

from .comparison import compare_solvers, create_comparison_report
from .metrics import compute_convergence_rate, compute_l2_error, compute_mass_conservation
from .visualization import plot_convergence_comparison, plot_side_by_side

__all__ = [
    "compare_solvers",
    "compute_convergence_rate",
    "compute_l2_error",
    "compute_mass_conservation",
    "create_comparison_report",
    "plot_convergence_comparison",
    "plot_side_by_side",
]
