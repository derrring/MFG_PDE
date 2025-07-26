from .plot_utils import plot_results, plot_convergence
from .aux_func import *
from .math_utils import *
from .convergence import (
    AdvancedConvergenceMonitor, DistributionComparator, OscillationDetector, create_default_monitor,
    AdaptiveConvergenceWrapper, adaptive_convergence, wrap_solver_with_adaptive_convergence,
    ParticleMethodDetector, test_particle_detection
)
from .validation import validate_solution_array, validate_mfg_solution, validate_convergence_parameters, safe_solution_return
from .exceptions import (
    MFGSolverError, ConvergenceError, ConfigurationError, SolutionNotAvailableError, 
    DimensionMismatchError, NumericalInstabilityError,
    validate_solver_state, validate_array_dimensions, validate_parameter_value, check_numerical_stability
)
from .solver_result import (
    SolverResult, ConvergenceResult, create_solver_result, MFGSolverResult
)
from .logging import (
    get_logger, configure_logging, MFGLogger,
    log_solver_start, log_solver_progress, log_solver_completion,
    log_validation_error, log_performance_metric, LoggedOperation
)
from .logging_decorators import (
    logged_solver_method, logged_operation, logged_validation,
    performance_logged, LoggingMixin, add_logging_to_class
)
from .advanced_visualization import (
    MFGVisualizer, SolverMonitoringDashboard, VisualizationUtils,
    quick_plot_solution, quick_plot_convergence
)
from .mathematical_visualization import (
    MFGMathematicalVisualizer, quick_hjb_analysis, quick_fp_analysis,
    quick_phase_space_analysis
)
