"""
MFG_PDE Utilities Module.

This module contains core utilities for the MFG_PDE package.
All plotting functionality has been permanently moved to mfg_pde.visualization.

For visualization, use:
    from mfg_pde.visualization import create_visualization_manager
    from mfg_pde.visualization import plot_convergence, plot_results
"""

# Core utility functions (non-plotting)
from .aux_func import *
from .convergence import (
    adaptive_convergence,
    AdaptiveConvergenceWrapper,
    AdvancedConvergenceMonitor,
    create_default_monitor,
    DistributionComparator,
    OscillationDetector,
    ParticleMethodDetector,
    test_particle_detection,
    wrap_solver_with_adaptive_convergence,
)
from .exceptions import (
    check_numerical_stability,
    ConfigurationError,
    ConvergenceError,
    DimensionMismatchError,
    MFGSolverError,
    NumericalInstabilityError,
    SolutionNotAvailableError,
    validate_array_dimensions,
    validate_parameter_value,
    validate_solver_state,
)
from .logging import (
    configure_logging,
    configure_research_logging,
    get_logger,
    log_performance_metric,
    log_solver_completion,
    log_solver_progress,
    log_solver_start,
    log_validation_error,
    LoggedOperation,
    MFGLogger,
)
from .logging_decorators import (
    add_logging_to_class,
    logged_operation,
    logged_solver_method,
    logged_validation,
    LoggingMixin,
    performance_logged,
)
from .math_utils import *
from .integration import trapezoid, get_integration_info
from .solver_result import (
    ConvergenceResult,
    create_solver_result,
    MFGSolverResult,
    SolverResult,
)
from .validation import (
    safe_solution_return,
    validate_convergence_parameters,
    validate_mfg_solution,
    validate_solution_array,
)

# Optional modules with graceful handling
try:
    from .notebook_reporting import (
        create_comparative_analysis,
        create_mfg_research_report,
        MFGNotebookReporter,
    )

    NOTEBOOK_REPORTING_AVAILABLE = True
except ImportError:
    NOTEBOOK_REPORTING_AVAILABLE = False

try:
    from .polars_integration import (
        PolarsDataFrameManager,
        create_polars_manager,
        convert_to_polars,
        validate_polars_dataframe,
    )

    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

# Utility modules
try:
    from .cli import main as cli_main

    CLI_AVAILABLE = True
except ImportError:
    CLI_AVAILABLE = False

try:
    from .experiment_manager import (
        ExperimentManager,
        create_experiment_manager,
    )

    EXPERIMENT_MANAGER_AVAILABLE = True
except ImportError:
    EXPERIMENT_MANAGER_AVAILABLE = False

try:
    from .memory_management import (
        MemoryMonitor,
        get_memory_usage,
        optimize_memory_usage,
    )

    MEMORY_MANAGEMENT_AVAILABLE = True
except ImportError:
    MEMORY_MANAGEMENT_AVAILABLE = False

try:
    from .performance_monitoring import (
        PerformanceMonitor,
        benchmark_solver,
        profile_function,
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
