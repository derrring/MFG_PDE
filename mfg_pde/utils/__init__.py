from .advanced_visualization import (
    MFGVisualizer,
    quick_plot_convergence,
    quick_plot_solution,
    SolverMonitoringDashboard,
    VisualizationUtils,
)
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
from .mathematical_visualization import (
    MFGMathematicalVisualizer,
    quick_fp_analysis,
    quick_hjb_analysis,
    quick_phase_space_analysis,
)
from .plot_utils import plot_convergence, plot_results
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
