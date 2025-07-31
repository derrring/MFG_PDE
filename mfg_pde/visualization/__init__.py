"""
Comprehensive visualization module for MFG_PDE.

This module provides state-of-the-art interactive visualizations for Mean Field Games
with full migration of all plotting utilities from utils/ for centralized management.

Features:
- Interactive 2D/3D plotting with Plotly & Bokeh
- Real-time monitoring and dashboards  
- Parameter sweep analysis & dashboards
- Convergence animations & monitoring
- Mathematical plotting with LaTeX support
- Legacy matplotlib compatibility
- Multi-format export capabilities
- Unified visualization management
- Professional analytics engine

Migrated Components:
- utils/plot_utils.py → legacy_plotting.py
- utils/advanced_visualization.py → integrated into interactive_plots.py
- utils/mathematical_visualization.py → mathematical_plots.py
"""

# Enhanced network MFG visualization
from .enhanced_network_plots import (
    create_enhanced_network_visualizer,
    EnhancedNetworkMFGVisualizer,
)

# Core interactive visualization system
from .interactive_plots import (
    BOKEH_AVAILABLE,
    create_bokeh_visualizer,
    create_plotly_visualizer,
    create_visualization_manager,
    MFGBokehVisualizer,
    MFGPlotlyVisualizer,
    MFGVisualizationManager,
    PLOTLY_AVAILABLE,
    quick_2d_plot,
    quick_3d_plot,
)

# Legacy plotting for backward compatibility
from .legacy_plotting import (  # Aliases for backward compatibility
    legacy_myplot3d,
    legacy_plot_convergence,
    legacy_plot_results,
    modern_plot_convergence,
    modern_plot_mfg_solution,
    myplot3d,
    plot_convergence,
    plot_results,
)

# Mathematical plotting with LaTeX support
from .mathematical_plots import (
    create_mathematical_visualizer,
    MathematicalPlotter,
    MFGMathematicalVisualizer,
    plot_mathematical_function,
    plot_mfg_density,
)

# Comprehensive analytics engine
from .mfg_analytics import (
    analyze_mfg_solution_quick,
    analyze_parameter_sweep_quick,
    create_analytics_engine,
    MFGAnalyticsEngine,
)

# Network MFG visualization
from .network_plots import create_network_visualizer, NetworkMFGVisualizer

__all__ = [
    # Core interactive visualization
    "MFGPlotlyVisualizer",
    "MFGBokehVisualizer",
    "MFGVisualizationManager",
    "create_plotly_visualizer",
    "create_bokeh_visualizer",
    "create_visualization_manager",
    "quick_2d_plot",
    "quick_3d_plot",
    # Network MFG visualization
    "NetworkMFGVisualizer",
    "create_network_visualizer",
    "EnhancedNetworkMFGVisualizer",
    "create_enhanced_network_visualizer",
    # Analytics engine
    "MFGAnalyticsEngine",
    "create_analytics_engine",
    "analyze_mfg_solution_quick",
    "analyze_parameter_sweep_quick",
    # Mathematical plotting
    "MathematicalPlotter",
    "MFGMathematicalVisualizer",
    "plot_mathematical_function",
    "plot_mfg_density",
    "create_mathematical_visualizer",
    # Legacy compatibility
    "myplot3d",
    "plot_convergence",
    "plot_results",
    "modern_plot_mfg_solution",
    "modern_plot_convergence",
    # Availability flags
    "PLOTLY_AVAILABLE",
    "BOKEH_AVAILABLE",
]

# Version info
__version__ = "2.0.0"  # Updated for comprehensive migration
__author__ = "MFG_PDE Team"
__description__ = (
    "Comprehensive visualization system for Mean Field Games with full utils/ migration"
)

# Migration completion notice
_MIGRATION_COMPLETE = True
_MIGRATED_MODULES = [
    "utils/plot_utils.py",
    "utils/advanced_visualization.py",
    "utils/mathematical_visualization.py",
]


def get_migration_info():
    """Get information about the completed migration."""
    return {
        "migration_complete": _MIGRATION_COMPLETE,
        "migrated_modules": _MIGRATED_MODULES,
        "new_location": "mfg_pde.visualization",
        "version": __version__,
        "legacy_support": True,
    }
