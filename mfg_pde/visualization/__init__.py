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

from __future__ import annotations

# Centralized optional dependency imports
# All submodules should import from here to avoid duplication

# Plotly imports
try:
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.offline as offline
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    px = None  # type: ignore[assignment]
    go = None  # type: ignore[assignment]
    offline = None  # type: ignore[assignment]
    make_subplots = None  # type: ignore[assignment]

# Bokeh imports
try:
    import bokeh.transform as transform
    from bokeh.io import curdoc, push_notebook, show
    from bokeh.layouts import column, gridplot, row
    from bokeh.models import ColorBar, ColumnDataSource, HoverTool, LinearColorMapper
    from bokeh.models.tools import BoxZoomTool, PanTool, ResetTool, SaveTool, WheelZoomTool
    from bokeh.palettes import Inferno256, Plasma256, Viridis256
    from bokeh.plotting import figure, output_file, save

    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False
    # Set all bokeh imports to None for graceful degradation
    transform = None  # type: ignore[assignment]
    curdoc = None  # type: ignore[assignment]
    push_notebook = None  # type: ignore[assignment]
    show = None  # type: ignore[assignment]
    column = None  # type: ignore[assignment]
    gridplot = None  # type: ignore[assignment]
    row = None  # type: ignore[assignment]
    ColorBar = None  # type: ignore[assignment]
    ColumnDataSource = None  # type: ignore[assignment]
    HoverTool = None  # type: ignore[assignment]
    LinearColorMapper = None  # type: ignore[assignment]
    BoxZoomTool = None  # type: ignore[assignment]
    PanTool = None  # type: ignore[assignment]
    ResetTool = None  # type: ignore[assignment]
    SaveTool = None  # type: ignore[assignment]
    WheelZoomTool = None  # type: ignore[assignment]
    Inferno256 = None  # type: ignore[assignment]
    Plasma256 = None  # type: ignore[assignment]
    Viridis256 = None  # type: ignore[assignment]
    figure = None  # type: ignore[assignment]
    output_file = None  # type: ignore[assignment]
    save = None  # type: ignore[assignment]

# NetworkX imports
try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None  # type: ignore[assignment]

# Enhanced network MFG visualization
# Convergence plotting (standalone functions)
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
from .enhanced_network_plots import EnhancedNetworkMFGVisualizer, create_enhanced_network_visualizer

# Core interactive visualization system
from .interactive_plots import (
    MFGBokehVisualizer,
    MFGPlotlyVisualizer,
    MFGVisualizationManager,
    create_bokeh_visualizer,
    create_plotly_visualizer,
    create_visualization_manager,
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
    MathematicalPlotter,
    MFGMathematicalVisualizer,
    create_mathematical_visualizer,
    plot_mathematical_function,
    plot_mfg_density,
)

# Comprehensive analytics engine
from .mfg_analytics import (
    MFGAnalyticsEngine,
    analyze_mfg_solution_quick,
    analyze_parameter_sweep_quick,
    create_analytics_engine,
)

# Multi-dimensional visualization (2D/3D)
from .multidim_viz import MultiDimVisualizer

# Network MFG visualization
from .network_plots import NetworkMFGVisualizer, create_network_visualizer

__all__ = [
    # Availability flags
    "BOKEH_AVAILABLE",
    "NETWORKX_AVAILABLE",
    "PLOTLY_AVAILABLE",
    "BoxZoomTool",
    "ColorBar",
    "ColumnDataSource",
    # Enhanced network MFG visualization
    "EnhancedNetworkMFGVisualizer",
    "HoverTool",
    "Inferno256",
    "LinearColorMapper",
    # Analytics engine
    "MFGAnalyticsEngine",
    "MFGBokehVisualizer",
    "MFGMathematicalVisualizer",
    # Core interactive visualization
    "MFGPlotlyVisualizer",
    "MFGVisualizationManager",
    # Mathematical plotting
    "MathematicalPlotter",
    # Multi-dimensional visualization
    "MultiDimVisualizer",
    # Network MFG visualization
    "NetworkMFGVisualizer",
    "PanTool",
    "Plasma256",
    "ResetTool",
    "SaveTool",
    "Viridis256",
    "WheelZoomTool",
    "analyze_mfg_solution_quick",
    "analyze_parameter_sweep_quick",
    "column",
    "create_analytics_engine",
    "create_bokeh_visualizer",
    "create_enhanced_network_visualizer",
    "create_mathematical_visualizer",
    "create_network_visualizer",
    "create_plotly_visualizer",
    "create_visualization_manager",
    "curdoc",
    "figure",
    "go",
    "gridplot",
    # Legacy compatibility (including unused legacy functions)
    "legacy_myplot3d",
    "legacy_plot_convergence",
    "legacy_plot_results",
    "make_subplots",
    "modern_plot_convergence",
    "modern_plot_mfg_solution",
    "myplot3d",
    # NetworkX imports for submodules
    "nx",
    "offline",
    "output_file",
    "plot_convergence",
    # Convergence plotting (standalone)
    "plot_convergence_rate",
    "plot_convergence_summary",
    "plot_distribution_evolution",
    "plot_error_history",
    "plot_from_monitor",
    "plot_mass_history",
    "plot_multi_error_history",
    "plot_wasserstein_history",
    "plot_mathematical_function",
    "plot_mfg_density",
    "plot_results",
    "push_notebook",
    # Plotly imports for submodules
    "px",
    "quick_2d_plot",
    "quick_3d_plot",
    "row",
    "save",
    "show",
    # Bokeh imports for submodules
    "transform",
]

# Version info
__version__ = "2.0.0"  # Updated for comprehensive migration
__author__ = "MFG_PDE Team"
__description__ = "Comprehensive visualization system for Mean Field Games with full utils/ migration"

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
