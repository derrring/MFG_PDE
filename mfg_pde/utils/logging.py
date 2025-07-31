#!/usr/bin/env python3
"""
Professional Logging Infrastructure for MFG_PDE

Provides structured logging with configurable levels, formatting, and optional
color support for better debugging and monitoring capabilities.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import colorlog

    COLORLOG_AVAILABLE = True
except ImportError:
    COLORLOG_AVAILABLE = False


class MFGFormatter(logging.Formatter):
    """Custom formatter for MFG_PDE logging with enhanced information."""

    def __init__(self, use_colors: bool = False, include_location: bool = False):
        self.use_colors = use_colors and COLORLOG_AVAILABLE
        self.include_location = include_location

        if self.use_colors:
            # Use colorlog for colored output
            self.colored_formatter = colorlog.ColoredFormatter(
                "%(log_color)s%(asctime)s - %(name)-20s - %(levelname)-8s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                log_colors={
                    "DEBUG": "cyan",
                    "INFO": "green",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "red,bg_white",
                },
            )

        # Standard formatter
        format_str = "%(asctime)s - %(name)-20s - %(levelname)-8s - %(message)s"
        if self.include_location:
            format_str += " [%(filename)s:%(lineno)d]"

        super().__init__(format_str, datefmt="%Y-%m-%d %H:%M:%S")

    def format(self, record):
        if self.use_colors:
            return self.colored_formatter.format(record)
        else:
            return super().format(record)


class MFGLogger:
    """Central logging manager for MFG_PDE with configuration management."""

    _instance = None
    _loggers: Dict[str, logging.Logger] = {}
    _log_level = logging.INFO
    _log_to_file = False
    _log_file_path: Optional[Path] = None
    _use_colors = True
    _include_location = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MFGLogger, cls).__new__(cls)
        return cls._instance

    @classmethod
    def configure(
        cls,
        level: Union[str, int] = "INFO",
        log_to_file: bool = False,
        log_file_path: Optional[Union[str, Path]] = None,
        use_colors: bool = True,
        include_location: bool = False,
        suppress_external: bool = True,
    ):
        """
        Configure global logging settings for MFG_PDE.

        Args:
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_to_file: Whether to log to file
            log_file_path: Path to log file (optional)
            use_colors: Use colored terminal output if available
            include_location: Include file location in log messages
            suppress_external: Suppress verbose logging from external libraries
        """
        # Set log level
        if isinstance(level, str):
            cls._log_level = getattr(logging, level.upper())
        else:
            cls._log_level = level

        cls._log_to_file = log_to_file
        cls._use_colors = use_colors and COLORLOG_AVAILABLE
        cls._include_location = include_location

        # Set up log file path
        if log_to_file:
            if log_file_path is None:
                # Default to logs directory
                log_dir = Path.cwd() / "logs"
                log_dir.mkdir(exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                cls._log_file_path = log_dir / f"mfg_pde_{timestamp}.log"
            else:
                cls._log_file_path = Path(log_file_path)
                cls._log_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Suppress verbose logging from external libraries
        if suppress_external:
            logging.getLogger("matplotlib").setLevel(logging.WARNING)
            logging.getLogger("PIL").setLevel(logging.WARNING)
            logging.getLogger("urllib3").setLevel(logging.WARNING)
            logging.getLogger("requests").setLevel(logging.WARNING)

        # Update existing loggers
        for logger in cls._loggers.values():
            cls._setup_logger(logger)

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Get or create a logger for the specified module/component.

        Args:
            name: Logger name (typically __name__ from calling module)

        Returns:
            Configured logger instance
        """
        if name not in cls._loggers:
            logger = logging.getLogger(name)
            cls._loggers[name] = logger
            cls._setup_logger(logger)

        return cls._loggers[name]

    @classmethod
    def _setup_logger(cls, logger: logging.Logger):
        """Configure individual logger with current settings."""
        # Clear existing handlers
        logger.handlers.clear()
        logger.setLevel(cls._log_level)

        # Create formatter
        formatter = MFGFormatter(
            use_colors=cls._use_colors, include_location=cls._include_location
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(cls._log_level)
        logger.addHandler(console_handler)

        # File handler (if enabled)
        if cls._log_to_file and cls._log_file_path:
            file_handler = logging.FileHandler(cls._log_file_path)
            # File logs always use standard formatting (no colors)
            file_formatter = MFGFormatter(
                use_colors=False, include_location=cls._include_location
            )
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(cls._log_level)
            logger.addHandler(file_handler)

        # Prevent propagation to root logger
        logger.propagate = False


# Convenience functions for easy usage
def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger for the current module.

    Args:
        name: Logger name (if None, uses calling module name)

    Returns:
        Configured logger instance
    """
    if name is None:
        # Get caller's module name
        import inspect

        frame = inspect.currentframe().f_back
        name = frame.f_globals.get("__name__", "mfg_pde")

    return MFGLogger.get_logger(name)


def configure_logging(**kwargs):
    """
    Configure global logging settings.

    Keyword Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to file
        log_file_path: Path to log file
        use_colors: Use colored terminal output
        include_location: Include file location in messages
        suppress_external: Suppress external library logging
    """
    MFGLogger.configure(**kwargs)


# Configuration presets for common use cases
def configure_research_logging(
    experiment_name: str = None, level: str = "INFO", include_debug: bool = False
):
    """
    Configure logging optimized for research sessions.

    Args:
        experiment_name: Name for the experiment (used in filename)
        level: Base logging level
        include_debug: Whether to include debug information

    Returns:
        Path to the log file created
    """
    if include_debug:
        level = "DEBUG"

    # Create research logs directory
    research_dir = Path("research_logs")
    research_dir.mkdir(exist_ok=True)

    # Generate filename
    if experiment_name:
        safe_name = "".join(
            c for c in experiment_name if c.isalnum() or c in (" ", "-", "_")
        ).strip()
        safe_name = safe_name.replace(" ", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = research_dir / f"{safe_name}_{timestamp}.log"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = research_dir / f"research_session_{timestamp}.log"

    configure_logging(
        level=level,
        log_to_file=True,
        log_file_path=log_file,
        use_colors=True,
        include_location=include_debug,
        suppress_external=True,
    )

    # Log session start
    logger = get_logger("mfg_pde.research")
    logger.info(f"Research session started: {experiment_name or 'Unnamed'}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Logging level: {level}")

    return str(log_file)


def configure_development_logging(include_location: bool = True):
    """
    Configure logging optimized for development and debugging.

    Args:
        include_location: Include file:line information
    """
    configure_logging(
        level="DEBUG",
        log_to_file=True,
        use_colors=True,
        include_location=include_location,
        suppress_external=False,  # Show external library logs for debugging
    )

    logger = get_logger("mfg_pde.development")
    logger.info("Development logging enabled - DEBUG level with full details")


def configure_production_logging(log_file: Optional[str] = None):
    """
    Configure logging optimized for production use.

    Args:
        log_file: Custom log file path (optional)
    """
    if log_file is None:
        prod_dir = Path("production_logs")
        prod_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = prod_dir / f"mfg_production_{timestamp}.log"

    configure_logging(
        level="WARNING",  # Only warnings and errors
        log_to_file=True,
        log_file_path=log_file,
        use_colors=False,  # Clean output for production
        include_location=False,
        suppress_external=True,
    )

    logger = get_logger("mfg_pde.production")
    logger.warning("Production logging enabled - WARNING level and above only")


def configure_performance_logging(log_file: Optional[str] = None):
    """
    Configure logging optimized for performance analysis.

    Args:
        log_file: Custom log file path (optional)
    """
    if log_file is None:
        perf_dir = Path("performance_logs")
        perf_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = perf_dir / f"performance_{timestamp}.log"

    configure_logging(
        level="INFO",
        log_to_file=True,
        log_file_path=log_file,
        use_colors=False,
        include_location=False,
        suppress_external=True,
    )

    logger = get_logger("mfg_pde.performance")
    logger.info("Performance logging enabled - focus on timing and metrics")

    return str(log_file)


def log_solver_start(logger: logging.Logger, solver_name: str, config: Dict[str, Any]):
    """Log solver initialization with configuration."""
    logger.info(f"Initializing {solver_name}")
    logger.debug(f"Solver configuration: {config}")


def log_solver_progress(
    logger: logging.Logger,
    iteration: int,
    error: float,
    max_iterations: int,
    additional_info: Dict[str, Any] = None,
):
    """Log solver iteration progress."""
    progress_pct = (iteration / max_iterations) * 100
    msg = f"Iteration {iteration}/{max_iterations} ({progress_pct:.1f}%) - Error: {error:.2e}"

    if additional_info:
        info_str = ", ".join(f"{k}: {v}" for k, v in additional_info.items())
        msg += f" - {info_str}"

    logger.debug(msg)


def log_solver_completion(
    logger: logging.Logger,
    solver_name: str,
    iterations: int,
    final_error: float,
    execution_time: float,
    converged: bool,
):
    """Log solver completion with summary."""
    status = "CONVERGED" if converged else "MAX_ITERATIONS_REACHED"
    logger.info(f"{solver_name} completed - Status: {status}")
    logger.info(
        f"Final results: {iterations} iterations, error: {final_error:.2e}, "
        f"time: {execution_time:.3f}s"
    )


def log_validation_error(
    logger: logging.Logger, component: str, error_msg: str, suggestion: str = None
):
    """Log validation errors with suggestions."""
    logger.error(f"Validation error in {component}: {error_msg}")
    if suggestion:
        logger.info(f"Suggestion: {suggestion}")


def log_performance_metric(
    logger: logging.Logger,
    operation: str,
    duration: float,
    additional_metrics: Dict[str, Any] = None,
):
    """Log performance metrics with enhanced details."""
    msg = f"Performance - {operation}: {duration:.3f}s"
    if additional_metrics:
        metrics_str = ", ".join(f"{k}: {v}" for k, v in additional_metrics.items())
        msg += f" ({metrics_str})"
    logger.info(msg)  # Changed to INFO level for better visibility


def log_memory_usage(
    logger: logging.Logger, operation: str, peak_memory_mb: float = None
):
    """Log memory usage information."""
    try:
        import os

        import psutil

        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        current_mb = memory_info.rss / 1024 / 1024

        msg = f"Memory - {operation}: {current_mb:.1f} MB"
        if peak_memory_mb:
            msg += f" (peak: {peak_memory_mb:.1f} MB)"
        logger.debug(msg)
    except ImportError:
        if peak_memory_mb:
            logger.debug(f"Memory - {operation}: peak {peak_memory_mb:.1f} MB")


def log_solver_configuration(
    logger: logging.Logger,
    solver_name: str,
    config: Dict[str, Any],
    problem_info: Dict[str, Any] = None,
):
    """Enhanced solver configuration logging."""
    logger.info(f"=== {solver_name} Configuration ===")

    # Log solver parameters
    for key, value in config.items():
        if isinstance(value, (int, float, str, bool)):
            logger.info(f"  {key}: {value}")
        elif isinstance(value, dict):
            logger.info(f"  {key}: {len(value)} parameters")
        else:
            logger.info(f"  {key}: {type(value).__name__}")

    # Log problem information
    if problem_info:
        logger.info("=== Problem Information ===")
        for key, value in problem_info.items():
            logger.info(f"  {key}: {value}")


def log_convergence_analysis(
    logger: logging.Logger,
    error_history: List[float],
    final_iterations: int,
    tolerance: float,
    converged: bool,
):
    """Log detailed convergence analysis."""
    logger.info("=== Convergence Analysis ===")
    logger.info(f"  Final status: {'CONVERGED' if converged else 'MAX_ITERATIONS'}")
    logger.info(f"  Iterations: {final_iterations}")
    logger.info(f"  Target tolerance: {tolerance:.2e}")

    if error_history:
        initial_error = error_history[0]
        final_error = error_history[-1]
        logger.info(f"  Initial error: {initial_error:.2e}")
        logger.info(f"  Final error: {final_error:.2e}")

        if len(error_history) > 1:
            reduction_factor = initial_error / final_error
            logger.info(f"  Error reduction: {reduction_factor:.2e}x")

            # Estimate convergence rate
            if len(error_history) > 2:
                ratios = [
                    error_history[i + 1] / error_history[i]
                    for i in range(len(error_history) - 1)
                    if error_history[i] > 0
                ]
                if ratios:
                    avg_ratio = sum(ratios) / len(ratios)
                    logger.info(f"  Average convergence rate: {avg_ratio:.4f}")


def log_mass_conservation(
    logger: logging.Logger, mass_history: List[float], tolerance: float = 1e-6
):
    """Log mass conservation analysis."""
    if not mass_history:
        return

    logger.info("=== Mass Conservation Analysis ===")
    initial_mass = mass_history[0]
    final_mass = mass_history[-1]
    max_deviation = max(abs(m - 1.0) for m in mass_history)
    mass_drift = abs(final_mass - initial_mass)

    logger.info(f"  Initial mass: {initial_mass:.8f}")
    logger.info(f"  Final mass: {final_mass:.8f}")
    logger.info(f"  Mass drift: {mass_drift:.2e}")
    logger.info(f"  Max deviation: {max_deviation:.2e}")

    if max_deviation < tolerance:
        logger.info("  Status: Excellent mass conservation")
    elif max_deviation < 1e-3:
        logger.info("  Status: Good mass conservation")
    elif max_deviation < 1e-2:
        logger.info("  Status: Acceptable mass conservation")
    else:
        logger.warning("  Status: Mass conservation needs improvement")


# Context manager for logging operations
class LoggedOperation:
    """Context manager for logging timed operations."""

    def __init__(
        self, logger: logging.Logger, operation_name: str, log_level: int = logging.INFO
    ):
        self.logger = logger
        self.operation_name = operation_name
        self.log_level = log_level
        self.start_time = None

    def __enter__(self):
        self.logger.log(self.log_level, f"Starting {self.operation_name}")
        import time

        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time

        duration = time.time() - self.start_time

        if exc_type is None:
            self.logger.log(
                self.log_level, f"Completed {self.operation_name} in {duration:.3f}s"
            )
        else:
            self.logger.error(
                f"Failed {self.operation_name} after {duration:.3f}s: {exc_val}"
            )

        return False  # Don't suppress exceptions


# Initialize default logger for the utils module
utils_logger = get_logger(__name__)


# Example usage and testing functions
def demo_logging_capabilities():
    """Demonstrate logging capabilities."""
    # Configure logging
    configure_logging(level="DEBUG", use_colors=True, include_location=True)

    # Get loggers for different components
    solver_logger = get_logger("mfg_pde.solvers")
    config_logger = get_logger("mfg_pde.config")
    utils_logger = get_logger("mfg_pde.utils")

    # Demonstrate different log levels
    solver_logger.debug("Debug message: Detailed solver internals")
    solver_logger.info("Info message: Solver initialized successfully")
    solver_logger.warning("Warning message: Using default parameters")
    solver_logger.error("Error message: Convergence issue detected")

    # Demonstrate structured logging functions
    log_solver_start(
        solver_logger,
        "ParticleCollocationSolver",
        {"max_iterations": 50, "tolerance": 1e-6},
    )

    log_solver_progress(
        solver_logger, 10, 1.5e-4, 50, {"phase": "Picard", "damping": 0.5}
    )

    log_solver_completion(
        solver_logger, "ParticleCollocationSolver", 25, 8.7e-7, 2.34, True
    )

    # Demonstrate context manager
    with LoggedOperation(utils_logger, "Matrix computation"):
        import time

        time.sleep(0.1)  # Simulate work

    print("\nLogging demonstration complete!")


if __name__ == "__main__":
    demo_logging_capabilities()
