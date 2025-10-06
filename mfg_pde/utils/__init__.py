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

# Core utility functions (non-plotting)
from .aux_func import npart, ppart

# Subdirectory imports
from .data.validation import (
    safe_solution_return,
    validate_convergence_parameters,
    validate_mfg_solution,
    validate_solution_array,
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
from .logging.decorators import (
    LoggingMixin,
    add_logging_to_class,
    logged_operation,
    logged_solver_method,
    logged_validation,
    performance_logged,
)
from .logging.logger import (
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
from .solver_result import ConvergenceResult, MFGSolverResult, SolverResult, create_solver_result
from .sparse_operations import (
    SparseMatrixBuilder,
    SparseSolver,
    estimate_sparsity,
    sparse_matmul,
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
__all__ = [
    # Availability flags
    "AVAILABLE_MODULES",
    "CLI_AVAILABLE",
    "EXPERIMENT_MANAGER_AVAILABLE",
    "MEMORY_MANAGEMENT_AVAILABLE",
    "NOTEBOOK_REPORTING_AVAILABLE",
    "PERFORMANCE_MONITORING_AVAILABLE",
    "POLARS_AVAILABLE",
    # Convergence monitoring
    "AdaptiveConvergenceWrapper",
    "AdvancedConvergenceMonitor",
    # Exception handling
    "ConfigurationError",
    "ConvergenceError",
    # Solver results
    "ConvergenceResult",
    "DimensionMismatchError",
    "DistributionComparator",
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
    # Validation
    "safe_solution_return",
    "sparse_matmul",
    "test_particle_detection",
    "trapezoid",
    "validate_array_dimensions",
    "validate_convergence_parameters",
    "validate_mfg_solution",
    "validate_parameter_value",
    "validate_solution_array",
    "validate_solver_state",
    "wrap_solver_with_adaptive_convergence",
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
