# MFG_PDE Logging System Guide

**Date**: July 26, 2025  
**Purpose**: Comprehensive guide for using the professional logging system in MFG_PDE

## Overview

The MFG_PDE logging system provides professional, structured logging capabilities for better debugging, monitoring, and development experience. It follows the established formatting preferences (no emojis, clean text-based output).

## Quick Start

### Basic Setup

```python
from mfg_pde.utils import configure_logging, get_logger

# Configure global logging
configure_logging(
    level="INFO",
    use_colors=True,
    log_to_file=False
)

# Get a logger for your module
logger = get_logger(__name__)

# Use the logger
logger.info("Application started")
logger.debug("Debug information")
logger.warning("Warning message")
logger.error("Error occurred")
```

### File Logging

```python
# Enable file logging
configure_logging(
    level="DEBUG",
    log_to_file=True,
    log_file_path="logs/mfg_pde.log",
    use_colors=True
)
```

## Configuration Options

### Logging Levels
- `DEBUG`: Detailed information for debugging
- `INFO`: General information about program execution
- `WARNING`: Warning messages for potential issues
- `ERROR`: Error messages for failures
- `CRITICAL`: Critical errors that may stop execution

### Global Configuration

```python
configure_logging(
    level="INFO",                    # Logging level
    log_to_file=False,              # Enable file logging
    log_file_path=None,             # Custom log file path
    use_colors=True,                # Colored terminal output
    include_location=False,         # Include file:line in messages
    suppress_external=True          # Suppress verbose external library logs
)
```

## Solver Integration

### Method 1: Structured Logging Functions

```python
from mfg_pde.utils import (
    get_logger, log_solver_start, log_solver_progress, 
    log_solver_completion, LoggedOperation
)

class MySolver:
    def __init__(self):
        self.logger = get_logger(f"{self.__class__.__module__}.{self.__class__.__name__}")
    
    def solve(self, max_iterations=50, tolerance=1e-6):
        # Log solver start
        config = {"max_iterations": max_iterations, "tolerance": tolerance}
        log_solver_start(self.logger, self.__class__.__name__, config)
        
        # Solve with operation logging
        with LoggedOperation(self.logger, "Main solve loop"):
            for i in range(max_iterations):
                error = self._solve_iteration()
                
                # Log progress
                log_solver_progress(self.logger, i+1, error, max_iterations)
                
                if error < tolerance:
                    break
        
        # Log completion
        log_solver_completion(
            self.logger, self.__class__.__name__, 
            i+1, error, 2.5, error < tolerance
        )
```

### Method 2: Logging Decorators

```python
from mfg_pde.utils import logged_solver_method, LoggingMixin

class MySolver(LoggingMixin):
    def __init__(self, max_iterations=50):
        super().__init__()
        self.max_iterations = max_iterations
    
    @logged_solver_method(log_config=True, log_performance=True)
    def solve(self, verbose=True):
        # Your solving logic here
        # Logging is handled automatically
        return result
```

### Method 3: Retrofit Existing Solvers

```python
from mfg_pde.utils import add_logging_to_class

# Enhance existing solver class
LoggedSolver = add_logging_to_class(OriginalSolver, "mfg_solver")

# Use enhanced solver
solver = LoggedSolver()
solver.log_info("Starting computation")
with solver.log_operation("Complex calculation"):
    result = solver.solve()
```

## Advanced Features

### Performance Logging

```python
from mfg_pde.utils import log_performance_metric, performance_logged

# Manual performance logging
logger = get_logger(__name__)
log_performance_metric(logger, "Matrix multiplication", 0.145, 
                      {"size": "1000x1000", "method": "numpy"})

# Automatic performance logging with decorator
@performance_logged("Heavy computation", include_memory=True)
def expensive_function():
    # Your computation here
    pass
```

### Validation Logging

```python
from mfg_pde.utils import log_validation_error, logged_validation

# Manual validation logging
try:
    validate_array(data)
except ValueError as e:
    log_validation_error(logger, "Array validation", str(e), 
                        "Check array dimensions and data types")

# Automatic validation logging
@logged_validation("Input parameters")
def validate_solver_input(problem, config):
    if problem is None:
        raise ValueError("Problem cannot be None")
    return True
```

### Context Manager Logging

```python
from mfg_pde.utils import LoggedOperation

logger = get_logger(__name__)

# Time and log operations automatically
with LoggedOperation(logger, "Database query"):
    results = database.query(sql)

with LoggedOperation(logger, "Data processing", logging.DEBUG):
    processed_data = process(results)
```

## Best Practices

### 1. Logger Naming Convention

```python
# Use module-based naming
logger = get_logger(__name__)

# Or component-based naming
solver_logger = get_logger("mfg_pde.solvers.particle_collocation")
config_logger = get_logger("mfg_pde.config")
```

### 2. Appropriate Log Levels

```python
# DEBUG: Internal details, variable values
logger.debug(f"Iteration {i}: residual = {residual:.2e}")

# INFO: Important events, progress updates
logger.info("Solver converged successfully")

# WARNING: Potential issues, using defaults
logger.warning("Using default tolerance value")

# ERROR: Failures that don't stop execution
logger.error("Validation failed, using fallback method")

# CRITICAL: Fatal errors
logger.critical("Unable to initialize solver")
```

### 3. Structured Information

```python
# Good: Structured information
logger.info(f"Solver completed: {iterations} iterations, "
           f"error: {error:.2e}, time: {time:.3f}s")

# Better: Use structured logging functions
log_solver_completion(logger, "MySolver", iterations, error, time, True)
```

### 4. Exception Logging

```python
try:
    result = risky_operation()
except Exception as e:
    logger.error(f"Operation failed: {str(e)}")
    logger.debug("Full traceback:", exc_info=True)
    raise
```

## Configuration Examples

### Development Setup

```python
# Development: verbose logging with colors and location
configure_logging(
    level="DEBUG",
    use_colors=True,
    include_location=True,
    log_to_file=False
)
```

### Production Setup

```python
# Production: file logging with appropriate level
configure_logging(
    level="INFO",
    use_colors=False,
    log_to_file=True,
    log_file_path="/var/log/mfg_pde/application.log",
    suppress_external=True
)
```

### Research Setup

```python
# Research: detailed logging with performance metrics
configure_logging(
    level="DEBUG",
    use_colors=True,
    log_to_file=True,
    log_file_path="experiments/run_001.log",
    include_location=False
)
```

## Integration with Existing Code

### Minimal Changes

```python
# Add logging to existing solver with minimal changes
class ExistingSolver:
    def __init__(self):
        # Add this line
        self.logger = get_logger(f"{self.__class__.__module__}.{self.__class__.__name__}")
    
    def solve(self):
        # Add logging at key points
        self.logger.info("Starting solve")
        
        # Existing code unchanged
        result = self._existing_solve_logic()
        
        # Add completion logging
        self.logger.info("Solve completed")
        return result
```

### Complete Integration

```python
# Comprehensive logging integration
from mfg_pde.utils import LoggingMixin, logged_solver_method

class EnhancedSolver(LoggingMixin):
    def __init__(self, **kwargs):
        super().__init__()
        self.log_info("Solver initialized")
    
    @logged_solver_method(log_config=True, log_performance=True)
    def solve(self, **kwargs):
        with self.log_operation("Preprocessing"):
            self._preprocess()
        
        with self.log_operation("Main computation"):
            result = self._compute()
        
        with self.log_operation("Postprocessing"):
            self._postprocess(result)
        
        return result
```

## Troubleshooting

### Common Issues

1. **No colored output**: Install colorlog: `pip install colorlog`
2. **Log file not created**: Check directory permissions
3. **Too verbose**: Adjust logging level or suppress external libraries
4. **Performance impact**: Use appropriate logging levels in production

### Debug Logging Configuration

```python
# Check current logging configuration
from mfg_pde.utils.logging import MFGLogger

logger = get_logger("debug")
logger.info("Testing logging configuration")

# List all active loggers
import logging
for name in logging.Logger.manager.loggerDict:
    if name.startswith('mfg_pde'):
        print(f"Logger: {name}")
```

## Examples and Demonstrations

- `examples/logging_integration_example.py` - Basic logging integration
- `examples/retrofit_solver_logging.py` - Retrofitting existing solvers
- Run demos: `python examples/logging_integration_example.py`

## Dependencies

### Required
- Python standard library `logging` module

### Optional
- `colorlog` - For colored terminal output
- `psutil` - For memory usage metrics in performance logging

### Installation
```bash
# For full features
pip install colorlog psutil

# Minimal installation (no colors, no memory metrics)
# No additional dependencies needed
```

## Migration Guide

### From Print Statements

```python
# Old approach
print("Solver started")
print(f"Iteration {i}: error = {error}")
print("Solver completed")

# New approach
logger = get_logger(__name__)
logger.info("Solver started")
logger.debug(f"Iteration {i}: error = {error:.2e}")
logger.info("Solver completed")
```

### From Basic Logging

```python
# Old basic logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# New MFG_PDE logging
from mfg_pde.utils import configure_logging, get_logger
configure_logging(level="INFO", use_colors=True)
logger = get_logger(__name__)
```

---

This logging system provides professional debugging and monitoring capabilities while maintaining the clean, text-based formatting preferences established for the MFG_PDE project.