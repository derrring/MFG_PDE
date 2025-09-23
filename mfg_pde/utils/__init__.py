"""
MFG_PDE Utilities Module.

This module contains core utilities for the MFG_PDE package.
All plotting functionality has been permanently moved to mfg_pde.visualization.

For visualization, use:
    from mfg_pde.visualization import create_visualization_manager
    from mfg_pde.visualization import plot_convergence, plot_results
"""

# Core utility functions (non-plotting)
from .aux_func import npart, ppart
from .convergence import (
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
from .integration import get_integration_info, trapezoid
from .logging import (
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
from .logging_decorators import (
    LoggingMixin,
    add_logging_to_class,
    logged_operation,
    logged_solver_method,
    logged_validation,
    performance_logged,
)
from .solver_result import ConvergenceResult, MFGSolverResult, SolverResult, create_solver_result
from .validation import (
    safe_solution_return,
    validate_convergence_parameters,
    validate_mfg_solution,
    validate_solution_array,
)

# Optional modules with graceful handling
try:
    from .notebook_reporting import (  # noqa: F401
        MFGNotebookReporter,
        create_comparative_analysis,
        create_mfg_research_report,
    )

    NOTEBOOK_REPORTING_AVAILABLE = True
except ImportError:
    NOTEBOOK_REPORTING_AVAILABLE = False

try:
    from .polars_integration import (
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
    from .performance_monitoring import (
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
__all__ = [
    # Convergence monitoring
    "AdaptiveConvergenceWrapper",
    "AdvancedConvergenceMonitor",
    "DistributionComparator",
    "OscillationDetector",
    "ParticleMethodDetector",
    "adaptive_convergence",
    "create_default_monitor",
    "test_particle_detection",
    "wrap_solver_with_adaptive_convergence",
    # Auxiliary functions
    "npart",
    "ppart",
    # Exception handling
    "ConfigurationError",
    "ConvergenceError",
    "DimensionMismatchError",
    "MFGSolverError",
    "NumericalInstabilityError",
    "SolutionNotAvailableError",
    "check_numerical_stability",
    "validate_array_dimensions",
    "validate_parameter_value",
    "validate_solver_state",
    # Integration utilities
    "get_integration_info",
    "trapezoid",
    # Logging
    "LoggedOperation",
    "MFGLogger",
    "configure_logging",
    "configure_research_logging",
    "get_logger",
    "log_performance_metric",
    "log_solver_completion",
    "log_solver_progress",
    "log_solver_start",
    "log_validation_error",
    # Logging decorators
    "LoggingMixin",
    "add_logging_to_class",
    "logged_operation",
    "logged_solver_method",
    "logged_validation",
    "performance_logged",
    # Solver results
    "ConvergenceResult",
    "MFGSolverResult",
    "SolverResult",
    "create_solver_result",
    # Validation
    "safe_solution_return",
    "validate_convergence_parameters",
    "validate_mfg_solution",
    "validate_solution_array",
    # Availability flags
    "AVAILABLE_MODULES",
    "NOTEBOOK_REPORTING_AVAILABLE",
    "POLARS_AVAILABLE",
    "CLI_AVAILABLE",
    "EXPERIMENT_MANAGER_AVAILABLE",
    "MEMORY_MANAGEMENT_AVAILABLE",
    "PERFORMANCE_MONITORING_AVAILABLE",
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
