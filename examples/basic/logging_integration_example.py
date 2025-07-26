#!/usr/bin/env python3
"""
Logging Integration Example

Demonstrates how to integrate the professional logging system with MFG_PDE
solvers for better debugging, monitoring, and development experience.
"""

import numpy as np
import sys
import os
import time

# Add the parent directory to the path so we can import mfg_pde
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mfg_pde import MFGProblem, create_fast_solver
from mfg_pde.utils.logging import (
    get_logger, configure_logging, LoggedOperation,
    log_solver_start, log_solver_progress, log_solver_completion,
    log_validation_error, log_performance_metric
)


def create_test_problem():
    """Create a simple test MFG problem."""
    
    class SimpleMFGProblem(MFGProblem):
        def __init__(self):
            super().__init__(T=1.0, Nt=20, xmin=0.0, xmax=1.0, Nx=50)
        
        def g(self, x):
            return 0.5 * (x - 0.5)**2
        
        def rho0(self, x):
            return np.exp(-10 * (x - 0.3)**2) + np.exp(-10 * (x - 0.7)**2)
        
        def f(self, x, u, m):
            return 0.1 * u**2 + 0.05 * m
        
        def sigma(self, x):
            return 0.1
        
        def H(self, x, p, m):
            return 0.5 * p**2
        
        def dH_dm(self, x, p, m):
            return 0.0
    
    return SimpleMFGProblem()


def demo_basic_logging():
    """Demonstrate basic logging setup and usage."""
    print("=" * 60)
    print("BASIC LOGGING DEMONSTRATION")
    print("=" * 60)
    
    # Configure logging
    configure_logging(
        level="INFO",
        use_colors=True,
        log_to_file=False
    )
    
    # Get loggers for different components
    main_logger = get_logger(__name__)
    solver_logger = get_logger("mfg_pde.solvers")
    config_logger = get_logger("mfg_pde.config")
    
    # Demonstrate different log levels
    main_logger.info("Starting logging demonstration")
    main_logger.debug("This debug message won't show (level=INFO)")
    
    solver_logger.info("Solver logger initialized")
    config_logger.warning("Using default configuration parameters")
    
    print()


def demo_structured_logging():
    """Demonstrate structured logging functions."""
    print("=" * 60)
    print("STRUCTURED LOGGING DEMONSTRATION")
    print("=" * 60)
    
    # Configure with debug level to see all messages
    configure_logging(
        level="DEBUG",
        use_colors=True,
        include_location=True
    )
    
    logger = get_logger("demo.structured")
    
    # Simulate solver lifecycle logging
    solver_config = {
        "max_iterations": 50,
        "tolerance": 1e-6,
        "num_particles": 1000,
        "solver_type": "monitored_particle"
    }
    
    log_solver_start(logger, "MonitoredParticleCollocationSolver", solver_config)
    
    # Simulate solver iterations
    for i in range(1, 11):
        error = 1e-2 * np.exp(-i * 0.3)  # Decreasing error
        additional_info = {
            "phase": "Picard" if i <= 5 else "Newton",
            "damping": 0.5 if i <= 5 else 1.0
        }
        log_solver_progress(logger, i, error, 10, additional_info)
        time.sleep(0.1)  # Simulate computation time
    
    log_solver_completion(logger, "MonitoredParticleCollocationSolver", 
                         10, 8.7e-7, 1.0, True)
    
    print()


def demo_logged_operations():
    """Demonstrate LoggedOperation context manager."""
    print("=" * 60)
    print("LOGGED OPERATIONS DEMONSTRATION")
    print("=" * 60)
    
    logger = get_logger("demo.operations")
    
    # Successful operation
    with LoggedOperation(logger, "Matrix initialization"):
        time.sleep(0.2)
        matrix = np.random.rand(100, 100)
    
    # Operation with performance metrics
    with LoggedOperation(logger, "Matrix multiplication"):
        time.sleep(0.1)
        result = matrix @ matrix.T
    
    # Demonstrate error handling
    try:
        with LoggedOperation(logger, "Problematic operation"):
            time.sleep(0.1)
            raise ValueError("Simulated error for demonstration")
    except ValueError:
        logger.info("Handled the simulated error gracefully")
    
    print()


def demo_solver_integration():
    """Demonstrate logging integration with actual MFG solvers."""
    print("=" * 60)
    print("SOLVER INTEGRATION DEMONSTRATION")
    print("=" * 60)
    
    # Configure logging with file output
    configure_logging(
        level="INFO",
        use_colors=True,
        log_to_file=True,
        log_file_path="logs/solver_demo.log"
    )
    
    logger = get_logger("demo.solver_integration")
    
    # Create problem and solver
    problem = create_test_problem()
    x_coords = np.linspace(problem.xmin, problem.xmax, problem.Nx)
    collocation_points = x_coords.reshape(-1, 1)
    
    logger.info("Creating MFG problem and solver")
    
    with LoggedOperation(logger, "Solver creation"):
        solver = create_fast_solver(
            problem=problem,
            solver_type="monitored_particle",
            collocation_points=collocation_points,
            num_particles=500
        )
    
    logger.info(f"Solver created: {type(solver).__name__}")
    
    # Log solver configuration
    solver_config = {
        "solver_type": type(solver).__name__,
        "num_particles": getattr(solver, 'num_particles', 'N/A'),
        "problem_T": problem.T,
        "problem_Nt": problem.Nt,
        "problem_Nx": problem.Nx
    }
    
    log_solver_start(logger, type(solver).__name__, solver_config)
    
    # Solve with performance monitoring
    with LoggedOperation(logger, "MFG problem solving"):
        try:
            result = solver.solve(verbose=False)  # Disable solver's own verbose output
            log_solver_completion(
                logger, 
                type(solver).__name__,
                iterations=getattr(result, 'iterations', 'N/A'),
                final_error=getattr(result, 'final_error', 0.0),
                execution_time=1.0,  # Would get from timer in real implementation
                converged=getattr(result, 'converged', True)
            )
        except Exception as e:
            logger.error(f"Solver failed: {str(e)}")
            raise
    
    logger.info("Solver integration demonstration completed")
    print()


def demo_performance_logging():
    """Demonstrate performance logging capabilities."""
    print("=" * 60)
    print("PERFORMANCE LOGGING DEMONSTRATION")
    print("=" * 60)
    
    logger = get_logger("demo.performance")
    
    # Simulate various performance metrics
    operations = [
        ("Matrix assembly", 0.15, {"size": "100x100", "method": "sparse"}),
        ("Linear solve", 0.08, {"solver": "GMRES", "iterations": 12}),
        ("Convergence check", 0.02, {"tolerance": 1e-6, "norm": "L2"}),
        ("Result validation", 0.05, {"arrays": 3, "checks": "dimensions+values"})
    ]
    
    for op_name, duration, metrics in operations:
        log_performance_metric(logger, op_name, duration, metrics)
        time.sleep(0.1)
    
    print()


def demo_validation_logging():
    """Demonstrate validation error logging."""
    print("=" * 60)
    print("VALIDATION LOGGING DEMONSTRATION")
    print("=" * 60)
    
    logger = get_logger("demo.validation")
    
    # Simulate validation scenarios
    validation_cases = [
        ("Array dimensions", "Expected shape (50, 20), got (50, 21)", 
         "Check time discretization parameters"),
        ("Boundary conditions", "No-flux boundary violated at x=0", 
         "Verify boundary condition implementation"),
        ("Mass conservation", "Total mass changed by 0.15%", 
         "Check particle weights or integration scheme")
    ]
    
    for component, error_msg, suggestion in validation_cases:
        log_validation_error(logger, component, error_msg, suggestion)
    
    print()


def demo_file_logging():
    """Demonstrate file logging capabilities."""
    print("=" * 60)
    print("FILE LOGGING DEMONSTRATION")
    print("=" * 60)
    
    # Configure with file logging
    configure_logging(
        level="DEBUG",
        use_colors=True,
        log_to_file=True,
        log_file_path="logs/mfg_pde_demo.log"
    )
    
    logger = get_logger("demo.file_logging")
    
    logger.info("This message goes to both console and file")
    logger.debug("Debug information is logged to both outputs")
    logger.warning("Warnings are highlighted in console but plain in file")
    
    # Check if log file was created
    log_file = "logs/mfg_pde_demo.log"
    if os.path.exists(log_file):
        print(f"Log file created: {log_file}")
        with open(log_file, 'r') as f:
            print("Log file contents:")
            print("-" * 40)
            print(f.read())
            print("-" * 40)
    
    print()


def run_comprehensive_logging_demo():
    """Run complete logging system demonstration."""
    print("[DEMO] MFG_PDE PROFESSIONAL LOGGING SYSTEM DEMONSTRATION")
    print("=" * 80)
    print("This example demonstrates the comprehensive logging capabilities")
    print("for better debugging, monitoring, and development experience.")
    print("=" * 80)
    print()
    
    try:
        demo_basic_logging()
        demo_structured_logging()
        demo_logged_operations()
        demo_performance_logging()
        demo_validation_logging()
        demo_solver_integration()
        demo_file_logging()
        
        print("=" * 80)
        print("[COMPLETED] LOGGING SYSTEM DEMONSTRATION")
        print("=" * 80)
        print()
        print("Key Features Demonstrated:")
        print("• Professional structured logging with configurable levels")
        print("• Optional colored terminal output (when colorlog available)")
        print("• File logging with automatic log rotation")
        print("• Specialized logging functions for solvers and validation")
        print("• Context managers for timed operations")
        print("• Performance metrics logging")
        print("• Integration with existing MFG_PDE components")
        print()
        print("Installation tip: For colored output, install colorlog:")
        print("  pip install colorlog")
        print()
        
    except Exception as e:
        print(f"[ERROR] Error in demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_comprehensive_logging_demo()