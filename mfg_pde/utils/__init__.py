"""
MFG_PDE Utilities Module.

This module contains core utilities for the MFG_PDE package.
All plotting functionality has been permanently moved to mfg_pde.visualization.

For visualization, use:
    from mfg_pde.visualization import create_visualization_manager
    from mfg_pde.visualization import plot_convergence, plot_results

Organization:
- logging/: Logging utilities and decorators
- performance/: Performance monitoring and optimization
- notebooks/: Jupyter notebook integration
- numerical/: Numerical computation utilities
- data/: Data handling and validation
"""

import warnings as _warnings

# Backward compatibility alias for logging module rename (DEPRECATED - will be removed in v0.13.0)
# The logging module was renamed to mfg_logging to avoid shadowing Python's stdlib logging
from . import mfg_logging as logging  # noqa: F401

# Core utility functions (non-plotting)
from .aux_func import npart, ppart

# Subdirectory imports
from .data.validation import (
    safe_solution_return,
    validate_convergence_parameters,
    validate_mfg_solution,
    validate_solution_array,
)

# Drift field helpers for FP solvers
from .drift_helpers import (
    composite_drift,
    density_dependent_drift,
    optimal_control_drift,
    prescribed_drift,
    zero_drift,
)
from .exceptions import (
    ConfigurationError,
    ConvergenceError,
    DimensionMismatchError,
    MFGSolverError,
    NumericalInstabilityError,
    SolutionNotAvailableError,
    check_numerical_stability,
    validate_array_dimensions,
    validate_parameter_value,
    validate_solver_state,
)
from .geometry import (
    BoxObstacle,
    CircleObstacle,
    Complement,
    Difference,
    Hyperrectangle,
    Hypersphere,
    ImplicitDomain,
    Intersection,
    RectangleObstacle,
    SphereObstacle,
    Union,
    create_box_obstacle,
    create_circle_obstacle,
    create_rectangle_obstacle,
    create_sphere_obstacle,
)
from .hamiltonian_adapter import HamiltonianAdapter, adapt_hamiltonian, create_hamiltonian_adapter
from .mfg_logging.decorators import (
    LoggingMixin,
    add_logging_to_class,
    logged_operation,
    logged_solver_method,
    logged_validation,
    performance_logged,
)
from .mfg_logging.logger import (
    LoggedOperation,
    MFGLogger,
    configure_logging,
    configure_research_logging,
    get_logger,
    log_performance_metric,
    log_solver_completion,
    log_solver_progress,
    log_solver_start,
    log_validation_error,
)
from .numerical.convergence import (
    AdaptiveConvergenceWrapper,
    AdvancedConvergenceMonitor,
    DistributionComparator,
    OscillationDetector,
    ParticleMethodDetector,
    adaptive_convergence,
    create_default_monitor,
    test_particle_detection,
    wrap_solver_with_adaptive_convergence,
)
from .numerical.integration import get_integration_info, trapezoid
from .numerical.particle_interpolation import (
    estimate_kde_bandwidth,
    interpolate_grid_to_particles,
    interpolate_particles_to_grid,
)
from .numerical.qp_utils import QPCache, QPSolver
from .numerical.sdf_utils import (
    sdf_box,
    sdf_complement,
    sdf_difference,
    sdf_gradient,
    sdf_intersection,
    sdf_smooth_intersection,
    sdf_smooth_union,
    sdf_sphere,
    sdf_union,
)

# PDE coefficient handling
from .pde_coefficients import CoefficientField, get_spatial_grid
from .solver_result import ConvergenceResult, MFGSolverResult, SolverResult, create_solver_result
from .sparse_operations import (
    SparseMatrixBuilder,
    SparseSolver,
    estimate_sparsity,
    sparse_matmul,
)

# Backward compatibility alias (DEPRECATED - will be removed in v0.12.0)
adaptive_bandwidth_selection = estimate_kde_bandwidth

# Issue deprecation warning for adaptive_bandwidth_selection alias
_warnings.warn(
    "adaptive_bandwidth_selection is deprecated and will be removed in v0.12.0. "
    "Please use estimate_kde_bandwidth directly:\n"
    "  from mfg_pde.utils import estimate_kde_bandwidth",
    DeprecationWarning,
    stacklevel=2,
)

# Issue deprecation warning for logging module rename
_warnings.warn(
    "Importing from 'mfg_pde.utils.logging' is deprecated and will be removed in v0.13.0. "
    "The module has been renamed to avoid shadowing Python's stdlib. Please update:\n"
    "  from mfg_pde.utils.mfg_logging import get_logger, configure_logging\n"
    "Or import from mfg_pde.utils directly:\n"
    "  from mfg_pde.utils import get_logger, configure_logging",
    DeprecationWarning,
    stacklevel=2,
)

# Optional modules with graceful handling
try:
    from .notebooks.reporting import (  # noqa: F401
        MFGNotebookReporter,
        create_comparative_analysis,
        create_mfg_research_report,
    )

    NOTEBOOK_REPORTING_AVAILABLE = True
except ImportError:
    NOTEBOOK_REPORTING_AVAILABLE = False

try:
    from .data.polars_integration import (
        MFGDataFrame,  # noqa: F401
        benchmark_polars_vs_pandas,  # noqa: F401
        create_data_exporter,  # noqa: F401
        create_mfg_dataframe,  # noqa: F401
        create_parameter_sweep_analyzer,  # noqa: F401
        create_time_series_analyzer,  # noqa: F401
    )

    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

# Utility modules
try:
    from .cli import main as cli_main  # noqa: F401

    CLI_AVAILABLE = True
except ImportError:
    CLI_AVAILABLE = False

try:
    from .experiment_manager import (
        load_experiment_data,  # noqa: F401
        load_experiments_from_dir,  # noqa: F401
        plot_comparison_final_m,  # noqa: F401
        plot_comparison_total_mass,  # noqa: F401
        save_experiment_data,  # noqa: F401
    )

    EXPERIMENT_MANAGER_AVAILABLE = True
except ImportError:
    EXPERIMENT_MANAGER_AVAILABLE = False

try:
    from .memory_management import (
        MemoryMonitor,  # noqa: F401
        check_system_memory_availability,  # noqa: F401
        estimate_problem_memory_requirements,  # noqa: F401
        memory_usage_report,  # noqa: F401
    )

    MEMORY_MANAGEMENT_AVAILABLE = True
except ImportError:
    MEMORY_MANAGEMENT_AVAILABLE = False

try:
    from .performance.monitoring import (
        PerformanceMonitor,  # noqa: F401
        benchmark_solver,  # noqa: F401
        get_performance_report,  # noqa: F401
        performance_tracked,  # noqa: F401
    )

    PERFORMANCE_MONITORING_AVAILABLE = True
except ImportError:
    PERFORMANCE_MONITORING_AVAILABLE = False

# Availability info
AVAILABLE_MODULES = {
    "notebook_reporting": NOTEBOOK_REPORTING_AVAILABLE,
    "polars_integration": POLARS_AVAILABLE,
    "cli": CLI_AVAILABLE,
    "experiment_manager": EXPERIMENT_MANAGER_AVAILABLE,
    "memory_management": MEMORY_MANAGEMENT_AVAILABLE,
    "performance_monitoring": PERFORMANCE_MONITORING_AVAILABLE,
}

# Public API - Core utilities always available
# Keep organized with comments, don't auto-sort
__all__ = [
    # Availability flags
    "AVAILABLE_MODULES",
    "CLI_AVAILABLE",
    "EXPERIMENT_MANAGER_AVAILABLE",
    "MEMORY_MANAGEMENT_AVAILABLE",
    "NOTEBOOK_REPORTING_AVAILABLE",
    "PERFORMANCE_MONITORING_AVAILABLE",
    "POLARS_AVAILABLE",
    # PDE coefficient handling
    "CoefficientField",
    "get_spatial_grid",
    # Drift field helpers for FP solvers
    "zero_drift",
    "optimal_control_drift",
    "prescribed_drift",
    "density_dependent_drift",
    "composite_drift",
    # Convergence monitoring
    "AdaptiveConvergenceWrapper",
    "AdvancedConvergenceMonitor",
    # Geometry utilities (obstacles, SDF)
    "BoxObstacle",
    "CircleObstacle",
    "Complement",
    # Exception handling
    "ConfigurationError",
    "ConvergenceError",
    # Solver results
    "ConvergenceResult",
    # CSG operations
    "Difference",
    "DimensionMismatchError",
    "DistributionComparator",
    # Geometry primitives
    "Hyperrectangle",
    "Hypersphere",
    "ImplicitDomain",
    "Intersection",
    # Hamiltonian signature adapter
    "HamiltonianAdapter",
    "adapt_hamiltonian",
    "create_hamiltonian_adapter",
    # Logging
    "LoggedOperation",
    # Logging decorators
    "LoggingMixin",
    "MFGLogger",
    "MFGSolverError",
    "MFGSolverResult",
    "NumericalInstabilityError",
    "OscillationDetector",
    "ParticleMethodDetector",
    "SolutionNotAvailableError",
    "SolverResult",
    "SparseMatrixBuilder",
    "SparseSolver",
    "adaptive_bandwidth_selection",
    "adaptive_convergence",
    "add_logging_to_class",
    "check_numerical_stability",
    "configure_logging",
    "configure_research_logging",
    "create_default_monitor",
    "create_solver_result",
    "estimate_sparsity",
    # Integration utilities
    "get_integration_info",
    # Particle interpolation
    "interpolate_grid_to_particles",
    "interpolate_particles_to_grid",
    "estimate_kde_bandwidth",
    # Signed distance functions
    "sdf_box",
    "sdf_complement",
    "sdf_difference",
    "sdf_gradient",
    "sdf_intersection",
    "sdf_smooth_intersection",
    "sdf_smooth_union",
    "sdf_sphere",
    "sdf_union",
    # QP utilities
    "QPCache",
    "QPSolver",
    "get_logger",
    "log_performance_metric",
    "log_solver_completion",
    "log_solver_progress",
    "log_solver_start",
    "log_validation_error",
    "logged_operation",
    "logged_solver_method",
    "logged_validation",
    # Auxiliary functions
    "npart",
    "performance_logged",
    "ppart",
    # Geometry aliases
    "RectangleObstacle",
    "SphereObstacle",
    # Validation
    "safe_solution_return",
    "sparse_matmul",
    "test_particle_detection",
    "trapezoid",
    # CSG helpers
    "Union",
    "validate_array_dimensions",
    "validate_convergence_parameters",
    "validate_mfg_solution",
    "validate_parameter_value",
    "validate_solution_array",
    "validate_solver_state",
    "wrap_solver_with_adaptive_convergence",
    # Geometry factory functions
    "create_box_obstacle",
    "create_circle_obstacle",
    "create_rectangle_obstacle",
    "create_sphere_obstacle",
]

# Add optional modules to public API if available
if NOTEBOOK_REPORTING_AVAILABLE:
    __all__.extend(
        [
            "MFGNotebookReporter",
            "create_comparative_analysis",
            "create_mfg_research_report",
        ]
    )

if POLARS_AVAILABLE:
    __all__.extend(
        [
            "MFGDataFrame",
            "benchmark_polars_vs_pandas",
            "create_data_exporter",
            "create_mfg_dataframe",
            "create_parameter_sweep_analyzer",
            "create_time_series_analyzer",
        ]
    )

if CLI_AVAILABLE:
    __all__.extend(
        [
            "cli_main",
        ]
    )

if EXPERIMENT_MANAGER_AVAILABLE:
    __all__.extend(
        [
            "load_experiment_data",
            "load_experiments_from_dir",
            "plot_comparison_final_m",
            "plot_comparison_total_mass",
            "save_experiment_data",
        ]
    )

if MEMORY_MANAGEMENT_AVAILABLE:
    __all__.extend(
        [
            "MemoryMonitor",
            "check_system_memory_availability",
            "estimate_problem_memory_requirements",
            "memory_usage_report",
        ]
    )

if PERFORMANCE_MONITORING_AVAILABLE:
    __all__.extend(
        [
            "PerformanceMonitor",
            "benchmark_solver",
            "get_performance_report",
            "performance_tracked",
        ]
    )
