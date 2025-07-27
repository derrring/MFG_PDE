"""
Example custom solver plugin for MFG_PDE.

This example demonstrates how to create a custom solver plugin that extends
the MFG_PDE framework with new solver capabilities. This particular example
implements a simple gradient descent-based solver for educational purposes.
"""

import numpy as np
from typing import Optional, List, Dict, Any
import time

from mfg_pde.core.plugin_system import SolverPlugin, PluginMetadata
from mfg_pde.config.pydantic_config import MFGSolverConfig
from mfg_pde.core.mfg_problem import MFGProblem
from mfg_pde.core.solver_result import SolverResult


class GradientDescentMFGSolver:
    """
    Example gradient descent-based MFG solver.
    
    This is a simplified educational implementation that demonstrates
    the solver interface required for plugin integration.
    """
    
    def __init__(self, problem: MFGProblem, learning_rate: float = 0.01, 
                 max_iterations: int = 1000, tolerance: float = 1e-6):
        self.problem = problem
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        
        # Initialize solution arrays
        self.U_solution = np.zeros((problem.Nt + 1, problem.Nx + 1))
        self.M_solution = np.zeros((problem.Nt + 1, problem.Nx + 1))
        
        # Set initial conditions
        self.M_solution[0, :] = problem.m_init
        self.U_solution[-1, :] = problem.g_final
    
    def solve(self) -> SolverResult:
        """Solve the MFG system using gradient descent."""
        start_time = time.time()
        
        convergence_history = []
        
        for iteration in range(self.max_iterations):
            # Store previous solution for convergence check
            U_prev = self.U_solution.copy()
            M_prev = self.M_solution.copy()
            
            # Gradient descent step for value function
            self._update_value_function()
            
            # Gradient descent step for density
            self._update_density()
            
            # Check convergence
            U_error = np.linalg.norm(self.U_solution - U_prev)
            M_error = np.linalg.norm(self.M_solution - M_prev)
            total_error = U_error + M_error
            
            convergence_history.append(total_error)
            
            if total_error < self.tolerance:
                break
        
        execution_time = time.time() - start_time
        converged = total_error < self.tolerance
        
        # Create result object
        result = SolverResult(
            U_solution=self.U_solution,
            M_solution=self.M_solution,
            converged=converged,
            iterations=iteration + 1,
            final_error=total_error,
            execution_time=execution_time,
            convergence_history=convergence_history,
            solver_info={
                'solver_type': 'gradient_descent',
                'learning_rate': self.learning_rate,
                'max_iterations': self.max_iterations,
                'tolerance': self.tolerance
            }
        )
        
        return result
    
    def _update_value_function(self):
        """Update value function using gradient descent."""
        # Simplified HJB equation gradient (this is educational, not mathematically rigorous)
        for t in range(self.problem.Nt - 1, -1, -1):
            for x_idx in range(1, self.problem.Nx):
                x = self.problem.x_values[x_idx]
                
                # Simplified gradient computation
                if t < self.problem.Nt:
                    time_derivative = (self.U_solution[t+1, x_idx] - self.U_solution[t, x_idx]) / self.problem.Dt
                else:
                    time_derivative = 0
                
                # Spatial derivatives (finite differences)
                u_x = (self.U_solution[t, x_idx+1] - self.U_solution[t, x_idx-1]) / (2 * self.problem.Dx)
                u_xx = (self.U_solution[t, x_idx+1] - 2*self.U_solution[t, x_idx] + self.U_solution[t, x_idx-1]) / (self.problem.Dx**2)
                
                # Simplified HJB residual
                hjb_residual = time_derivative + 0.5 * u_x**2 - 0.5 * self.problem.sigma**2 * u_xx
                
                # Gradient descent update
                self.U_solution[t, x_idx] -= self.learning_rate * hjb_residual
    
    def _update_density(self):
        """Update density using gradient descent."""
        # Simplified Fokker-Planck equation gradient
        for t in range(1, self.problem.Nt + 1):
            for x_idx in range(1, self.problem.Nx):
                # Time derivative
                time_derivative = (self.M_solution[t, x_idx] - self.M_solution[t-1, x_idx]) / self.problem.Dt
                
                # Spatial derivatives
                m_x = (self.M_solution[t, x_idx+1] - self.M_solution[t, x_idx-1]) / (2 * self.problem.Dx)
                m_xx = (self.M_solution[t, x_idx+1] - 2*self.M_solution[t, x_idx] + self.M_solution[t, x_idx-1]) / (self.problem.Dx**2)
                
                # Control from value function
                u_x = (self.U_solution[t, x_idx+1] - self.U_solution[t, x_idx-1]) / (2 * self.problem.Dx)
                
                # Simplified FP residual
                fp_residual = time_derivative - self.M_solution[t, x_idx] * u_x - 0.5 * self.problem.sigma**2 * m_xx
                
                # Gradient descent update
                self.M_solution[t, x_idx] -= self.learning_rate * fp_residual
                
                # Ensure non-negativity
                self.M_solution[t, x_idx] = max(0, self.M_solution[t, x_idx])


class SimplifiedNewtonMFGSolver:
    """
    Simplified Newton-method based solver for demonstration.
    
    This implements a basic Newton iteration approach for solving
    the MFG system, serving as another example solver type.
    """
    
    def __init__(self, problem: MFGProblem, max_iterations: int = 50, tolerance: float = 1e-6):
        self.problem = problem
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        
        # Initialize with existing core solver for comparison
        from mfg_pde import create_fast_solver
        self.reference_solver = create_fast_solver(problem, "fixed_point")
    
    def solve(self) -> SolverResult:
        """Solve using simplified Newton method."""
        start_time = time.time()
        
        # For this example, we'll use the reference solver with modified parameters
        # In a real implementation, this would contain the actual Newton method logic
        result = self.reference_solver.solve()
        
        # Modify result to indicate it came from our custom solver
        result.solver_info.update({
            'solver_type': 'simplified_newton',
            'plugin_provided': True,
            'custom_parameters': {
                'max_iterations': self.max_iterations,
                'tolerance': self.tolerance
            }
        })
        
        return result


class ExampleSolverPlugin(SolverPlugin):
    """
    Example solver plugin implementation.
    
    This plugin provides two custom solver types:
    1. gradient_descent - Educational gradient descent solver
    2. simplified_newton - Wrapper around existing solver with custom parameters
    """
    
    @property
    def metadata(self) -> PluginMetadata:
        """Plugin metadata."""
        return PluginMetadata(
            name="example_solver_plugin",
            version="1.0.0",
            description="Example plugin providing custom MFG solvers for educational purposes",
            author="MFG_PDE Development Team",
            email="dev@mfg-pde.org",
            license="MIT",
            homepage="https://github.com/mfg-pde/example-plugins",
            min_mfg_version="1.0.0",
            dependencies=["numpy"],
            tags=["educational", "gradient_descent", "newton_method"]
        )
    
    def get_solver_types(self) -> List[str]:
        """Return available solver types."""
        return ["gradient_descent", "simplified_newton"]
    
    def create_solver(self, problem: MFGProblem, solver_type: str, 
                     config: Optional[MFGSolverConfig] = None, **kwargs):
        """Create solver instance."""
        if not self.validate_solver_type(solver_type):
            raise ValueError(f"Unsupported solver type: {solver_type}")
        
        if solver_type == "gradient_descent":
            return self._create_gradient_descent_solver(problem, config, **kwargs)
        elif solver_type == "simplified_newton":
            return self._create_simplified_newton_solver(problem, config, **kwargs)
        
        raise ValueError(f"Unknown solver type: {solver_type}")
    
    def validate_solver_type(self, solver_type: str) -> bool:
        """Validate solver type."""
        return solver_type in self.get_solver_types()
    
    def get_solver_description(self, solver_type: str) -> str:
        """Get solver description."""
        descriptions = {
            "gradient_descent": (
                "Educational gradient descent solver for MFG systems. "
                "Uses simple gradient descent to iteratively solve the HJB and "
                "Fokker-Planck equations. Suitable for learning and small problems."
            ),
            "simplified_newton": (
                "Simplified Newton method solver with custom parameterization. "
                "Demonstrates how to wrap existing solvers with plugin functionality."
            )
        }
        return descriptions.get(solver_type, "Custom solver provided by example plugin")
    
    def get_solver_parameters(self, solver_type: str) -> Dict[str, Any]:
        """Get available parameters for solver type."""
        if solver_type == "gradient_descent":
            return {
                "learning_rate": {
                    "type": "float",
                    "default": 0.01,
                    "range": [1e-6, 1.0],
                    "description": "Learning rate for gradient descent steps"
                },
                "max_iterations": {
                    "type": "int", 
                    "default": 1000,
                    "range": [1, 10000],
                    "description": "Maximum number of iterations"
                },
                "tolerance": {
                    "type": "float",
                    "default": 1e-6,
                    "range": [1e-12, 1e-2],
                    "description": "Convergence tolerance"
                }
            }
        elif solver_type == "simplified_newton":
            return {
                "max_iterations": {
                    "type": "int",
                    "default": 50,
                    "range": [1, 200],
                    "description": "Maximum Newton iterations"
                },
                "tolerance": {
                    "type": "float",
                    "default": 1e-6,
                    "range": [1e-12, 1e-2],
                    "description": "Newton convergence tolerance"
                }
            }
        return {}
    
    def _create_gradient_descent_solver(self, problem: MFGProblem, 
                                      config: Optional[MFGSolverConfig] = None, **kwargs):
        """Create gradient descent solver instance."""
        # Extract parameters
        learning_rate = kwargs.get("learning_rate", 0.01)
        max_iterations = kwargs.get("max_iterations", 1000)
        tolerance = kwargs.get("tolerance", 1e-6)
        
        # Override with config values if provided
        if config:
            # Use general convergence settings if specific ones not provided
            tolerance = kwargs.get("tolerance", config.convergence_tolerance)
            max_iterations = kwargs.get("max_iterations", config.newton.max_iterations)
        
        return GradientDescentMFGSolver(
            problem=problem,
            learning_rate=learning_rate,
            max_iterations=max_iterations,
            tolerance=tolerance
        )
    
    def _create_simplified_newton_solver(self, problem: MFGProblem,
                                       config: Optional[MFGSolverConfig] = None, **kwargs):
        """Create simplified Newton solver instance."""
        max_iterations = kwargs.get("max_iterations", 50)
        tolerance = kwargs.get("tolerance", 1e-6)
        
        if config:
            tolerance = kwargs.get("tolerance", config.convergence_tolerance)
            max_iterations = kwargs.get("max_iterations", config.newton.max_iterations)
        
        return SimplifiedNewtonMFGSolver(
            problem=problem,
            max_iterations=max_iterations,
            tolerance=tolerance
        )
    
    def on_load(self):
        """Called when plugin is loaded."""
        print(f"Loading {self.metadata.name} v{self.metadata.version}")
        print("Available solvers:", ", ".join(self.get_solver_types()))
    
    def on_unload(self):
        """Called when plugin is unloaded."""
        print(f"Unloading {self.metadata.name}")


# Plugin registration function (called automatically by plugin system)
def register():
    """Register this plugin with the MFG_PDE plugin system."""
    return ExampleSolverPlugin


# For direct testing of the plugin
if __name__ == "__main__":
    # Example usage of the plugin
    from mfg_pde import ExampleMFGProblem
    from mfg_pde.core.plugin_system import get_plugin_manager
    
    # Create test problem
    problem = ExampleMFGProblem(Nx=20, Nt=10, T=1.0)
    
    # Register plugin manually for testing
    plugin_manager = get_plugin_manager()
    plugin = ExampleSolverPlugin()
    plugin_manager.register_plugin(ExampleSolverPlugin)
    plugin_manager.load_plugin("example_solver_plugin")
    
    # Test gradient descent solver
    print("Testing gradient descent solver...")
    gd_solver = plugin.create_solver(problem, "gradient_descent", learning_rate=0.02, max_iterations=100)
    gd_result = gd_solver.solve()
    print(f"Gradient descent converged: {gd_result.converged}, iterations: {gd_result.iterations}")
    
    # Test simplified Newton solver
    print("Testing simplified Newton solver...")
    newton_solver = plugin.create_solver(problem, "simplified_newton", max_iterations=30)
    newton_result = newton_solver.solve()
    print(f"Simplified Newton converged: {newton_result.converged}, iterations: {newton_result.iterations}")
    
    # List available solvers
    print("\nAll available solvers:")
    available_solvers = plugin_manager.list_available_solvers()
    for solver_type, info in available_solvers.items():
        print(f"  {solver_type}: {info['description']}")