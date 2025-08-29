# MFG_PDE Logging System Guide

**Date**: August 15, 2025  
**Updated**: Based on comprehensive analysis and existing implementation  
**Quality Score**: 8.7/10 - Excellent scientific computing logging infrastructure  
**Purpose**: Complete guide for using the professional logging system in MFG_PDE

## Overview

The MFG_PDE logging system provides a **sophisticated, multi-layered logging infrastructure** specifically designed for scientific computing and mathematical research workflows. The system demonstrates professional-grade engineering with extensive configurability, performance monitoring integration, and analytical capabilities.

**Key Features**:
- **Multi-destination logging**: Console and file output with independent formatting
- **Colored terminal output**: Professional color-coded log levels using `colorlog`
- **Research session management**: Timestamped log files with automatic directory creation
- **Performance monitoring integration**: Built-in performance tracking and regression detection
- **Mathematical analysis logging**: Specialized functions for convergence, mass conservation, and solver analysis
- **Comprehensive decorator system**: Easy integration with existing solvers
- **Log analysis tools**: Pattern recognition and performance bottleneck identification

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
- `DEBUG`: Detailed information for debugging (solver internals, variable values)
- `INFO`: General information about program execution (solver progress, completion)
- `WARNING`: Warning messages for potential issues (using defaults, performance concerns)
- `ERROR`: Error messages for failures (convergence issues, validation errors)
- `CRITICAL`: Critical errors that may stop execution (fatal initialization failures)

### Four Professional Configuration Presets

#### 1. Research Configuration (Recommended for Scientific Work)
```python
from mfg_pde.utils.logging import configure_research_logging

# Optimized for research sessions with experiment tracking
log_file = configure_research_logging(
    experiment_name="hjb_convergence_study",
    level="INFO",
    include_debug=False
)
# Creates: research_logs/hjb_convergence_study_20250815_142030.log
```

#### 2. Development Configuration (Full Debugging)
```python
from mfg_pde.utils.logging import configure_development_logging

# Full debugging with file:line information
configure_development_logging(include_location=True)
```

#### 3. Production Configuration (Minimal Logging)
```python
from mfg_pde.utils.logging import configure_production_logging

# WARNING level and above only, clean output
configure_production_logging(log_file="/var/log/mfg_pde.log")
```

#### 4. Performance Configuration (Performance Analysis)
```python
from mfg_pde.utils.logging import configure_performance_logging

# Focus on timing and performance metrics
log_file = configure_performance_logging()
```

### Manual Global Configuration

```python
configure_logging(
    level="INFO",                    # Logging level
    log_to_file=False,              # Enable file logging
    log_file_path=None,             # Custom log file path
    use_colors=True,                # Colored terminal output (requires colorlog)
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

### Enhanced Mathematical Analysis Logging

```python
from mfg_pde.utils.logging import (
    log_solver_configuration, log_convergence_analysis, 
    log_mass_conservation, log_performance_metric
)

# Comprehensive solver configuration documentation
log_solver_configuration(
    logger, "HJBSemiLagrangianSolver", 
    config={"max_iterations": 50, "tolerance": 1e-6},
    problem_info={"Nx": 100, "Nt": 50, "T": 1.0}
)

# Detailed convergence analysis with rate calculation
error_history = [1e-2, 5e-3, 1e-4, 2e-6, 8e-7]
log_convergence_analysis(logger, error_history, 25, 1e-6, True)

# Mass conservation monitoring
mass_history = [1.0001, 0.9999, 1.0000, 0.9998]
log_mass_conservation(logger, mass_history, tolerance=1e-6)
```

### Performance Logging with Regression Detection

```python
# Manual performance logging with enhanced metrics
log_performance_metric(
    logger, "Matrix multiplication", 0.145, 
    additional_metrics={
        "size": "1000x1000", 
        "method": "numpy", 
        "memory_mb": 156.7,
        "cpu_percent": 85.2
    }
)

# Memory usage tracking (requires psutil)
from mfg_pde.utils.logging import log_memory_usage
log_memory_usage(logger, "Large array allocation", peak_memory_mb=512.3)
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

### Method 1: Minimal Changes (Existing Code Enhancement)

```python
# Add logging to existing solver with minimal changes
class ExistingSolver:
    def __init__(self):
        # Add this line
        self.logger = get_logger(f"{self.__class__.__module__}.{self.__class__.__name__}")
    
    def solve(self, max_iterations=50, tolerance=1e-6):
        # Enhanced solver lifecycle logging
        config = {"max_iterations": max_iterations, "tolerance": tolerance}
        log_solver_start(self.logger, self.__class__.__name__, config)
        
        with LoggedOperation(self.logger, "Main solve loop"):
            # Existing solve logic with progress logging
            for i in range(max_iterations):
                error = self._solve_iteration()
                
                # Progress logging every 10 iterations
                if i % 10 == 0:
                    log_solver_progress(self.logger, i+1, error, max_iterations)
                
                if error < tolerance:
                    break
        
        # Enhanced completion logging with timing
        converged = error < tolerance
        log_solver_completion(
            self.logger, self.__class__.__name__, 
            i+1, error, self.execution_time, converged
        )
        return result
```

### Method 2: Decorator-Based Integration

```python
from mfg_pde.utils.logging_decorators import logged_solver_method, LoggingMixin

# Easy integration with decorators
class EnhancedSolver(LoggingMixin):
    @logged_solver_method(log_config=True, log_performance=True)
    def solve(self, **kwargs):
        # Automatic logging of method entry, exit, performance, and configuration
        return self._solve_implementation(**kwargs)
```

### Method 3: Retrofit Existing Classes

```python
from mfg_pde.utils.logging_decorators import add_logging_to_class

# Dynamically add logging to existing solver class
LoggedSolver = add_logging_to_class(OriginalSolver, "enhanced_solver")

# Use enhanced solver with full logging capabilities
solver = LoggedSolver()
solver.log_info("Starting enhanced computation")
with solver.log_operation("Complex calculation"):
    result = solver.solve()
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

## Log Analysis and Monitoring

### Built-in Log Analysis Tools

```python
from mfg_pde.utils.log_analysis import LogAnalyzer

# Comprehensive log file analysis
analyzer = LogAnalyzer("research_logs/experiment_20250815.log")

# Performance analysis
performance_report = analyzer.analyze_performance()
print(f"Average solve time: {performance_report['avg_solve_time']:.3f}s")
print(f"Performance bottlenecks: {performance_report['bottlenecks']}")

# Error pattern recognition
error_patterns = analyzer.analyze_error_patterns()
for pattern, count in error_patterns.items():
    print(f"Error pattern '{pattern}': {count} occurrences")

# Convergence analysis
convergence_stats = analyzer.analyze_convergence_behavior()
print(f"Average convergence rate: {convergence_stats['avg_rate']:.4f}")
```

### Performance Monitoring and Regression Detection

```python
from mfg_pde.utils.performance_monitoring import PerformanceMonitor

# Automatic performance baseline tracking
@performance_tracked("matrix_solve", baseline_file="baselines.json")
def solve_matrix(A, b):
    return np.linalg.solve(A, b)

# Performance regression alerts
monitor = PerformanceMonitor()
monitor.check_regression(
    operation="hjb_solve",
    current_time=2.5,
    baseline_time=1.2,
    threshold=2.0  # Alert if 2x slower
)
```

## Real-World Usage Examples

### Research Session Logging

```python
# Complete research session setup
from mfg_pde.utils.logging import configure_research_logging, get_logger

# Start research session
log_file = configure_research_logging("semi_lagrangian_analysis")
logger = get_logger(__name__)

logger.info("Research session: Semi-Lagrangian solver analysis")
logger.info(f"Git commit: {get_git_commit()}")
logger.info(f"Parameters: {experiment_parameters}")

# Your research code here...

logger.info("Research session completed successfully")
```

### Production Deployment

```python
# Production-ready logging setup
configure_production_logging("/var/log/mfg_pde/production.log")

# Minimal logging overhead with critical information only
logger = get_logger("production")
try:
    result = solve_production_problem()
    logger.info(f"Production solve completed: {result.summary}")
except Exception as e:
    logger.critical(f"Production failure: {e}")
    raise
```

## Troubleshooting

### Common Issues and Solutions

1. **No colored output**: 
   ```bash
   pip install colorlog
   # Or check: COLORLOG_AVAILABLE in mfg_pde.utils.logging
   ```

2. **Log file not created**: 
   ```python
   # Check directory permissions and path
   from pathlib import Path
   log_path = Path("logs/test.log")
   log_path.parent.mkdir(parents=True, exist_ok=True)
   ```

3. **Too verbose output**: 
   ```python
   # Adjust level or suppress external libraries
   configure_logging(level="WARNING", suppress_external=True)
   ```

4. **Performance impact**: 
   ```python
   # Use appropriate levels for production
   configure_production_logging()  # WARNING level and above only
   ```

5. **Memory usage monitoring requires psutil**: 
   ```bash
   pip install psutil
   # Or disable memory logging if not available
   ```

### Debug Logging Configuration

```python
# Comprehensive logging system diagnostics
from mfg_pde.utils.logging import MFGLogger

# Test logging configuration
logger = get_logger("debug_test")
logger.debug("Debug level test")
logger.info("Info level test")
logger.warning("Warning level test")
logger.error("Error level test")

# Inspect active loggers
import logging
print("Active MFG_PDE loggers:")
for name in sorted(logging.Logger.manager.loggerDict.keys()):
    if name.startswith('mfg_pde'):
        logger_obj = logging.getLogger(name)
        print(f"  {name}: level={logger_obj.level}, handlers={len(logger_obj.handlers)}")

# Check color support
from mfg_pde.utils.logging import COLORLOG_AVAILABLE
print(f"Color support available: {COLORLOG_AVAILABLE}")
```

## Examples and Demonstrations

- `examples/logging_integration_example.py` - Basic logging integration
- `examples/retrofit_solver_logging.py` - Retrofitting existing solvers
- Run demos: `python examples/logging_integration_example.py`

## Dependencies and Installation

### Required Dependencies
- Python standard library `logging` module (built-in)
- No additional required dependencies for basic functionality

### Optional Dependencies (Recommended)
- `colorlog` - Professional colored terminal output with log level color coding
- `psutil` - Memory usage monitoring and system metrics collection

### Installation Options

```bash
# Full features (recommended for development and research)
pip install colorlog psutil

# Research environment with UV (fastest)
uv add colorlog psutil

# Minimal installation (production environments)
# No additional dependencies needed - graceful fallback to standard logging
```

### Feature Availability Matrix

| Feature | No Dependencies | With colorlog | With psutil | Full Install |
|---------|----------------|---------------|-------------|-------------|
| Basic logging | ✅ | ✅ | ✅ | ✅ |
| File logging | ✅ | ✅ | ✅ | ✅ |
| Colored output | ❌ | ✅ | ❌ | ✅ |
| Memory monitoring | ❌ | ❌ | ✅ | ✅ |
| Performance tracking | ✅ (limited) | ✅ (limited) | ✅ (full) | ✅ (full) |
| Research sessions | ✅ | ✅ | ✅ | ✅ |

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
