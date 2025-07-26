#!/usr/bin/env python3
"""
Retrofit Existing Solvers with Logging

Demonstrates how to add professional logging to existing MFG_PDE solvers
without modifying their core implementation.
"""

import numpy as np
import sys
import os

# Add the parent directory to the path so we can import mfg_pde
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mfg_pde import MFGProblem, create_fast_solver
from mfg_pde.utils.logging import configure_logging, get_logger
from mfg_pde.utils.logging_decorators import (
    logged_solver_method, LoggingMixin, add_logging_to_class
)


def create_test_problem():
    """Create a simple test MFG problem."""
    
    class LoggedMFGProblem(MFGProblem):
        def __init__(self):
            super().__init__(T=1.0, Nt=30, xmin=0.0, xmax=1.0, Nx=60)
            self.logger = get_logger(f"{self.__class__.__module__}.{self.__class__.__name__}")
            self.logger.info("MFG problem initialized")
            self.logger.debug(f"Problem parameters: T={self.T}, Nt={self.Nt}, Nx={self.Nx}")
        
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
    
    return LoggedMFGProblem()


def demo_decorator_retrofit():
    """Demonstrate retrofitting solvers with logging decorators."""
    print("=" * 60)
    print("DECORATOR RETROFIT DEMONSTRATION")
    print("=" * 60)
    
    # Configure logging
    configure_logging(level="INFO", use_colors=True)
    logger = get_logger(__name__)
    
    # Create a mock solver class to demonstrate retrofit
    class MockSolver:
        def __init__(self, max_iterations=50, tolerance=1e-6):
            self.max_iterations = max_iterations
            self.tolerance = tolerance
            self.num_particles = 1000
        
        def solve(self, verbose=True):
            """Original solve method without logging."""
            import time
            
            if verbose:
                print("Starting solve (original verbose output)")
            
            # Simulate solving process
            iterations = 0
            error = 1.0
            
            while iterations < self.max_iterations and error > self.tolerance:
                time.sleep(0.05)  # Simulate computation
                error *= 0.8  # Simulate convergence
                iterations += 1
                
                if verbose and iterations % 10 == 0:
                    print(f"  Iteration {iterations}, error: {error:.2e}")
            
            # Mock result object
            class MockResult:
                def __init__(self, iterations, error, converged):
                    self.iterations = iterations
                    self.final_error = error
                    self.converged = converged
            
            converged = error <= self.tolerance
            return MockResult(iterations, error, converged)
    
    # Method 1: Decorate individual methods
    logger.info("Method 1: Decorating solve method")
    
    # Add logging decorator to the solve method
    MockSolver.solve = logged_solver_method(
        log_config=True,
        log_performance=True,
        log_result=True
    )(MockSolver.solve)
    
    # Test the decorated solver
    solver1 = MockSolver(max_iterations=30, tolerance=1e-5)
    result1 = solver1.solve(verbose=False)  # Disable original verbose output
    
    print()


def demo_mixin_retrofit():
    """Demonstrate retrofitting with LoggingMixin."""
    print("=" * 60)
    print("MIXIN RETROFIT DEMONSTRATION")
    print("=" * 60)
    
    logger = get_logger(__name__)
    
    # Create a solver class that inherits from LoggingMixin
    class LoggedSolver(LoggingMixin):
        def __init__(self, max_iterations=50, tolerance=1e-6):
            super().__init__()
            self.max_iterations = max_iterations
            self.tolerance = tolerance
            self.log_info("LoggedSolver initialized")
            self.log_debug(f"Parameters: max_iter={max_iterations}, tol={tolerance}")
        
        def solve(self, verbose=True):
            """Solve method with integrated logging."""
            self.log_info("Starting solve process")
            
            with self.log_operation("MFG solve"):
                import time
                iterations = 0
                error = 1.0
                
                while iterations < self.max_iterations and error > self.tolerance:
                    time.sleep(0.03)
                    error *= 0.85
                    iterations += 1
                    
                    if iterations % 15 == 0:
                        self.log_debug(f"Iteration {iterations}: error = {error:.2e}")
                
                converged = error <= self.tolerance
                
                if converged:
                    self.log_info(f"Converged in {iterations} iterations (error: {error:.2e})")
                else:
                    self.log_warning(f"Max iterations reached: {iterations} (error: {error:.2e})")
                
                # Mock result
                class MockResult:
                    def __init__(self, iterations, error, converged):
                        self.iterations = iterations
                        self.final_error = error
                        self.converged = converged
                
                return MockResult(iterations, error, converged)
    
    # Test the logged solver
    logger.info("Method 2: Using LoggingMixin")
    solver2 = LoggedSolver(max_iterations=40, tolerance=1e-4)
    result2 = solver2.solve()
    
    print()


def demo_class_enhancement():
    """Demonstrate enhancing existing classes with logging."""
    print("=" * 60)
    print("CLASS ENHANCEMENT DEMONSTRATION")
    print("=" * 60)
    
    logger = get_logger(__name__)
    
    # Original solver class without logging
    class OriginalSolver:
        def __init__(self, preset="balanced"):
            self.preset = preset
            self.max_iterations = {"fast": 20, "balanced": 50, "accurate": 100}[preset]
            self.tolerance = {"fast": 1e-4, "balanced": 1e-6, "accurate": 1e-8}[preset]
        
        def solve(self):
            """Original solve method."""
            import time
            time.sleep(0.2)
            
            # Mock solving
            iterations = min(25, self.max_iterations)
            error = self.tolerance * 0.1
            
            class MockResult:
                def __init__(self, iterations, error):
                    self.iterations = iterations
                    self.final_error = error
                    self.converged = True
            
            return MockResult(iterations, error)
        
        def validate_solution(self, result):
            """Original validation method."""
            if result.final_error > self.tolerance:
                raise ValueError("Solution validation failed")
            return True
    
    # Method 3: Enhance entire class with logging
    logger.info("Method 3: Enhancing entire class")
    
    LoggedOriginalSolver = add_logging_to_class(
        OriginalSolver, 
        logger_name="demo.enhanced_solver"
    )
    
    # Test the enhanced solver
    solver3 = LoggedOriginalSolver(preset="accurate")
    solver3.log_info("Testing enhanced solver class")
    
    with solver3.log_operation("Solving with enhanced class"):
        result3 = solver3.solve()
        solver3.validate_solution(result3)
    
    solver3.log_info("Enhanced solver demonstration completed")
    
    print()


def demo_real_solver_integration():
    """Demonstrate integration with actual MFG_PDE solvers."""
    print("=" * 60)
    print("REAL SOLVER INTEGRATION DEMONSTRATION")
    print("=" * 60)
    
    # Configure with debug level and file logging
    configure_logging(
        level="DEBUG",
        use_colors=True,
        log_to_file=True,
        log_file_path="logs/solver_retrofit.log"
    )
    
    logger = get_logger(__name__)
    
    # Create problem and solver
    problem = create_test_problem()
    x_coords = np.linspace(problem.xmin, problem.xmax, problem.Nx)
    collocation_points = x_coords.reshape(-1, 1)
    
    logger.info("Creating real MFG solver for retrofit demonstration")
    
    try:
        # Create solver
        solver = create_fast_solver(
            problem=problem,
            solver_type="monitored_particle",
            collocation_points=collocation_points,
            num_particles=800
        )
        
        logger.info(f"Created solver: {type(solver).__name__}")
        
        # If the solver has a solve method, we could retrofit it
        if hasattr(solver, 'solve'):
            logger.info("Solver has solve method - could be retrofitted")
            
            # For demonstration, just solve normally
            logger.info("Solving with existing solver (not retrofitted for this demo)")
            result = solver.solve(verbose=False)
            
            # Log result manually
            logger.info("Solver completed successfully")
            if hasattr(result, 'iterations'):
                logger.info(f"Iterations: {result.iterations}")
            if hasattr(result, 'converged'):
                logger.info(f"Converged: {result.converged}")
        
    except Exception as e:
        logger.error(f"Error in real solver integration: {e}")
        logger.debug("This is expected if dependencies are missing")
    
    print()


def run_retrofit_logging_demo():
    """Run complete retrofit logging demonstration."""
    print("[DEMO] RETROFIT EXISTING SOLVERS WITH LOGGING")
    print("=" * 80)
    print("This example shows how to add professional logging to existing")
    print("MFG_PDE solvers without modifying their core implementation.")
    print("=" * 80)
    print()
    
    try:
        demo_decorator_retrofit()
        demo_mixin_retrofit()
        demo_class_enhancement()
        demo_real_solver_integration()
        
        print("=" * 80)
        print("[COMPLETED] RETROFIT LOGGING DEMONSTRATION")
        print("=" * 80)
        print()
        print("Retrofit Methods Demonstrated:")
        print("• Method 1: Decorating individual solver methods")
        print("• Method 2: Inheriting from LoggingMixin for new classes")
        print("• Method 3: Enhancing entire existing classes")
        print("• Integration examples with real MFG_PDE solvers")
        print()
        print("Benefits:")
        print("• No modification of core solver logic required")
        print("• Consistent logging across all solvers")
        print("• Configurable logging levels and output")
        print("• Professional debugging and monitoring capabilities")
        print()
        
    except Exception as e:
        print(f"[ERROR] Error in demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_retrofit_logging_demo()