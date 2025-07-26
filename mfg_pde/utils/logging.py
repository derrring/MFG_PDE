#!/usr/bin/env python3
"""
Professional Logging Infrastructure for MFG_PDE

Provides structured logging with configurable levels, formatting, and optional
color support for better debugging and monitoring capabilities.
"""

import logging
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any, Union
from datetime import datetime

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
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white',
                }
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
    def configure(cls,
                  level: Union[str, int] = "INFO",
                  log_to_file: bool = False,
                  log_file_path: Optional[Union[str, Path]] = None,
                  use_colors: bool = True,
                  include_location: bool = False,
                  suppress_external: bool = True):
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
            use_colors=cls._use_colors,
            include_location=cls._include_location
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
                use_colors=False,
                include_location=cls._include_location
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
        name = frame.f_globals.get('__name__', 'mfg_pde')
    
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


def log_solver_start(logger: logging.Logger, solver_name: str, config: Dict[str, Any]):
    """Log solver initialization with configuration."""
    logger.info(f"Initializing {solver_name}")
    logger.debug(f"Solver configuration: {config}")


def log_solver_progress(logger: logging.Logger, iteration: int, error: float, 
                       max_iterations: int, additional_info: Dict[str, Any] = None):
    """Log solver iteration progress."""
    progress_pct = (iteration / max_iterations) * 100
    msg = f"Iteration {iteration}/{max_iterations} ({progress_pct:.1f}%) - Error: {error:.2e}"
    
    if additional_info:
        info_str = ", ".join(f"{k}: {v}" for k, v in additional_info.items())
        msg += f" - {info_str}"
    
    logger.debug(msg)


def log_solver_completion(logger: logging.Logger, solver_name: str, 
                         iterations: int, final_error: float, 
                         execution_time: float, converged: bool):
    """Log solver completion with summary."""
    status = "CONVERGED" if converged else "MAX_ITERATIONS_REACHED"
    logger.info(f"{solver_name} completed - Status: {status}")
    logger.info(f"Final results: {iterations} iterations, error: {final_error:.2e}, "
               f"time: {execution_time:.3f}s")


def log_validation_error(logger: logging.Logger, component: str, error_msg: str, 
                        suggestion: str = None):
    """Log validation errors with suggestions."""
    logger.error(f"Validation error in {component}: {error_msg}")
    if suggestion:
        logger.info(f"Suggestion: {suggestion}")


def log_performance_metric(logger: logging.Logger, operation: str, 
                         duration: float, additional_metrics: Dict[str, Any] = None):
    """Log performance metrics."""
    msg = f"Performance - {operation}: {duration:.3f}s"
    if additional_metrics:
        metrics_str = ", ".join(f"{k}: {v}" for k, v in additional_metrics.items())
        msg += f" ({metrics_str})"
    logger.debug(msg)


# Context manager for logging operations
class LoggedOperation:
    """Context manager for logging timed operations."""
    
    def __init__(self, logger: logging.Logger, operation_name: str, 
                 log_level: int = logging.INFO):
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
            self.logger.log(self.log_level, 
                          f"Completed {self.operation_name} in {duration:.3f}s")
        else:
            self.logger.error(
                f"Failed {self.operation_name} after {duration:.3f}s: {exc_val}")
        
        return False  # Don't suppress exceptions


# Initialize default logger for the utils module
utils_logger = get_logger(__name__)


# Example usage and testing functions
def demo_logging_capabilities():
    """Demonstrate logging capabilities."""
    # Configure logging
    configure_logging(
        level="DEBUG",
        use_colors=True,
        include_location=True
    )
    
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
    log_solver_start(solver_logger, "ParticleCollocationSolver", 
                    {"max_iterations": 50, "tolerance": 1e-6})
    
    log_solver_progress(solver_logger, 10, 1.5e-4, 50, 
                       {"phase": "Picard", "damping": 0.5})
    
    log_solver_completion(solver_logger, "ParticleCollocationSolver", 
                         25, 8.7e-7, 2.34, True)
    
    # Demonstrate context manager
    with LoggedOperation(utils_logger, "Matrix computation"):
        import time
        time.sleep(0.1)  # Simulate work
    
    print("\nLogging demonstration complete!")


if __name__ == "__main__":
    demo_logging_capabilities()