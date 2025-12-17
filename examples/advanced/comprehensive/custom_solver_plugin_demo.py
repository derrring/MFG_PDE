#!/usr/bin/env python3
"""
Custom Solver Plugin Example for MFG_PDE

Demonstrates how to create custom solver plugins that extend the MFG_PDE
framework with new solver capabilities.

Modern API (Phase 3.3):
- Uses SolverConfig (not deprecated MFGSolverConfig)
- Uses MFGProblem unified interface (not old 1D-specific attributes)
- Returns SolverResult with modern field names (U, M not U_solution, M_solution)
- Supports nD problems through geometry.grid interface

Features Demonstrated:
1. Custom solver implementation (GradientDescentMFGSolver)
2. Plugin registration and metadata
3. Plugin manager integration
4. Parameter validation and defaults
5. Lifecycle hooks (on_load, on_unload)
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np

from mfg_pde.core.plugin_system import PluginMetadata, SolverPlugin
from mfg_pde.utils.solver_result import SolverResult

if TYPE_CHECKING:
    from mfg_pde.config import SolverConfig
    from mfg_pde.core.mfg_problem import MFGProblem


class GradientDescentMFGSolver:
    """
    Educational gradient descent-based MFG solver.

    This is a simplified implementation demonstrating the solver interface
    required for plugin integration. Not intended for production use.

    Mathematical Formulation:
    - HJB: ∂u/∂t + H(x, ∇u, m) = 0 with u(T, x) = g(x)
    - FP:  ∂m/∂t - div(m ∇_p H(x, ∇u, m)) = 0 with m(0, x) = m₀(x)

    Simplified gradient descent approach:
    - Compute residuals for both equations
    - Update u and m in opposite direction of residuals
    - Enforce constraints (boundary conditions, non-negativity)

    Args:
        problem: MFG problem definition (must be 1D for this demo)
        learning_rate: Gradient descent step size
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance (L² norm)
    """

    def __init__(
        self,
        problem: MFGProblem,
        learning_rate: float = 0.01,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
    ):
        self.problem = problem
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        # Get problem dimensions using geometry-first API
        grid_shape = problem.geometry.get_grid_shape()  # (Nt_points, Nx_points)
        Nt_points = grid_shape[0]
        Nx_points = grid_shape[1]

        # Store dimensions (Nt, Nx are counts without +1)
        self.Nt = Nt_points - 1
        self.Nx = Nx_points - 1

        # Get spatial/temporal spacing and diffusion coefficient
        spacing = problem.geometry.get_grid_spacing()
        self.Dt = spacing[0]  # Temporal spacing
        self.Dx = spacing[1]  # Spatial spacing
        self.sigma = problem.diffusion

        # Get initial/final conditions
        self.m_init = problem.get_initial_m()
        self.g_final = problem.get_final_u()

        # Initialize solution arrays
        self.U = np.zeros((Nt_points, Nx_points))
        self.M = np.zeros((Nt_points, Nx_points))

        # Set boundary conditions
        self.M[0, :] = self.m_init
        self.U[-1, :] = self.g_final

    def solve(self) -> SolverResult:
        """
        Solve the MFG system using gradient descent.

        Returns:
            SolverResult with fields:
            - U, M: Solution arrays (Nt+1, Nx+1)
            - converged: Whether tolerance was met
            - iterations: Number of iterations performed
            - error_history_U, error_history_M: Convergence tracking
            - solver_name: "GradientDescentMFG"
            - metadata: Additional solver information
        """
        start_time = time.time()

        error_history_U = []
        error_history_M = []

        for iteration in range(self.max_iterations):
            # Store previous solution for convergence check
            U_prev = self.U.copy()
            M_prev = self.M.copy()

            # Gradient descent step for value function (HJB)
            self._update_value_function()

            # Gradient descent step for density (FP)
            self._update_density()

            # Restore boundary conditions (may drift during updates)
            self.M[0, :] = self.m_init
            self.U[-1, :] = self.g_final

            # Check convergence (L² norms)
            U_error = np.sqrt(np.sum((self.U - U_prev) ** 2) * self.Dx * self.Dt)
            M_error = np.sqrt(np.sum((self.M - M_prev) ** 2) * self.Dx * self.Dt)

            error_history_U.append(U_error)
            error_history_M.append(M_error)

            # Convergence criterion: both errors below tolerance
            if U_error < self.tolerance and M_error < self.tolerance:
                break

        execution_time = time.time() - start_time
        converged = U_error < self.tolerance and M_error < self.tolerance

        # Create result using modern SolverResult format
        result = SolverResult(
            U=self.U,
            M=self.M,
            converged=converged,
            iterations=iteration + 1,
            error_history_U=np.array(error_history_U),
            error_history_M=np.array(error_history_M),
            solver_name="GradientDescentMFG",
            metadata={
                "solver_type": "gradient_descent",
                "learning_rate": self.learning_rate,
                "max_iterations": self.max_iterations,
                "tolerance": self.tolerance,
                "execution_time": execution_time,
                "final_error_U": U_error,
                "final_error_M": M_error,
            },
        )

        return result

    def _update_value_function(self):
        """
        Update value function using gradient descent on HJB residual.

        HJB equation (simplified, 1D):
            ∂u/∂t + (1/2)|∇u|² - (σ²/2)Δu + f(x, m) = 0

        Discretization:
            (u[t+1] - u[t])/Δt + (1/2)(u_x)² - (σ²/2)u_xx = 0

        Gradient descent:
            u[t] ← u[t] - α * residual
        """
        for t in range(self.Nt - 1, -1, -1):
            for x_idx in range(1, self.Nx):
                # Time derivative (backward difference for HJB)
                if t < self.Nt:
                    time_deriv = (self.U[t + 1, x_idx] - self.U[t, x_idx]) / self.Dt
                else:
                    time_deriv = 0.0

                # Spatial derivatives (central differences)
                u_x = (self.U[t, x_idx + 1] - self.U[t, x_idx - 1]) / (2 * self.Dx)
                u_xx = (self.U[t, x_idx + 1] - 2 * self.U[t, x_idx] + self.U[t, x_idx - 1]) / (self.Dx**2)

                # Simplified HJB residual (Hamiltonian = (1/2)|p|²)
                hjb_residual = time_deriv + 0.5 * u_x**2 - 0.5 * self.sigma**2 * u_xx

                # Gradient descent update
                self.U[t, x_idx] -= self.learning_rate * hjb_residual

    def _update_density(self):
        """
        Update density using gradient descent on Fokker-Planck residual.

        Fokker-Planck equation (1D):
            ∂m/∂t - div(m ∇_p H) - (σ²/2)Δm = 0
            where ∇_p H = ∇u for quadratic Hamiltonian

        Discretization:
            (m[t] - m[t-1])/Δt - ∂(m * u_x)/∂x - (σ²/2)m_xx = 0

        Gradient descent:
            m[t] ← m[t] - α * residual
        """
        for t in range(1, self.Nt + 1):
            for x_idx in range(1, self.Nx):
                # Time derivative (forward difference for FP)
                time_deriv = (self.M[t, x_idx] - self.M[t - 1, x_idx]) / self.Dt

                # Spatial derivative (second order for diffusion term)
                m_xx = (self.M[t, x_idx + 1] - 2 * self.M[t, x_idx] + self.M[t, x_idx - 1]) / (self.Dx**2)

                # Control from value function
                u_x = (self.U[t, x_idx + 1] - self.U[t, x_idx - 1]) / (2 * self.Dx)

                # Simplified FP residual (linearized, educational only)
                fp_residual = time_deriv - self.M[t, x_idx] * u_x - 0.5 * self.sigma**2 * m_xx

                # Gradient descent update
                self.M[t, x_idx] -= self.learning_rate * fp_residual

                # Enforce non-negativity constraint (projection)
                self.M[t, x_idx] = max(0.0, self.M[t, x_idx])


class WrapperSolver:
    """
    Wrapper solver demonstrating how to wrap existing solvers.

    This shows how a plugin can provide a custom interface to built-in
    solvers, adding custom preprocessing, postprocessing, or parameter
    selection strategies.

    Args:
        problem: MFG problem definition
        max_iterations: Maximum Picard iterations
        tolerance: Convergence tolerance
    """

    def __init__(self, problem: MFGProblem, max_iterations: int = 50, tolerance: float = 1e-6):
        self.problem = problem
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        # Use fast solver factory for actual computation
        from mfg_pde.factory import create_fast_solver

        self.reference_solver = create_fast_solver(problem, method="fixed_point")

    def solve(self) -> SolverResult:
        """
        Solve using wrapped solver with custom parameters.

        Returns:
            SolverResult from underlying solver with modified metadata
        """
        # Solve using reference solver
        result = self.reference_solver.solve(max_iterations=self.max_iterations, tolerance=self.tolerance)

        # Augment metadata to indicate plugin-provided wrapper
        if result.metadata is None:
            result.metadata = {}

        result.metadata.update(
            {
                "wrapped_by": "WrapperSolver",
                "plugin_provided": True,
                "custom_parameters": {
                    "max_iterations": self.max_iterations,
                    "tolerance": self.tolerance,
                },
            }
        )

        return result


class ExampleSolverPlugin(SolverPlugin):
    """
    Example solver plugin implementation.

    This plugin provides two custom solver types:
    1. gradient_descent - Educational gradient descent solver
    2. wrapped - Wrapper around existing solver with custom parameters

    Plugin Interface Requirements:
    - metadata: PluginMetadata describing plugin
    - get_solver_types(): List available solver types
    - create_solver(): Factory method for solver instances
    - validate_solver_type(): Check if solver type is supported
    - get_solver_description(): Human-readable description
    - get_solver_parameters(): Parameter schema
    """

    @property
    def metadata(self) -> PluginMetadata:
        """Plugin metadata for discovery and version management."""
        return PluginMetadata(
            name="example_solver_plugin",
            version="2.0.0",
            description="Example plugin providing custom MFG solvers (Phase 3.3 compatible)",
            author="MFG_PDE Development Team",
            email="dev@mfg-pde.org",
            license="MIT",
            homepage="https://github.com/mfg-pde/mfg-pde",
            min_mfg_version="0.9.0",
            dependencies=["numpy>=1.20"],
            tags=["educational", "gradient_descent", "wrapper", "example"],
        )

    def get_solver_types(self) -> list[str]:
        """Return list of available solver types provided by this plugin."""
        return ["gradient_descent", "wrapped"]

    def create_solver(
        self,
        problem: MFGProblem,
        solver_type: str,
        config: SolverConfig | None = None,
        **kwargs: Any,
    ):
        """
        Create solver instance of specified type.

        Args:
            problem: MFG problem to solve
            solver_type: Type of solver ("gradient_descent" or "wrapped")
            config: Optional SolverConfig (modern API, replaces MFGSolverConfig)
            **kwargs: Solver-specific parameters (override config values)

        Returns:
            Solver instance with solve() method

        Raises:
            ValueError: If solver_type is not supported
        """
        if not self.validate_solver_type(solver_type):
            raise ValueError(f"Unsupported solver type: {solver_type}")

        if solver_type == "gradient_descent":
            return self._create_gradient_descent_solver(problem, config, **kwargs)
        elif solver_type == "wrapped":
            return self._create_wrapped_solver(problem, config, **kwargs)

        raise ValueError(f"Unknown solver type: {solver_type}")

    def validate_solver_type(self, solver_type: str) -> bool:
        """Validate that solver type is provided by this plugin."""
        return solver_type in self.get_solver_types()

    def get_solver_description(self, solver_type: str) -> str:
        """Get human-readable description of solver type."""
        descriptions = {
            "gradient_descent": (
                "Educational gradient descent solver for 1D MFG systems. "
                "Uses simple gradient descent to iteratively solve the HJB and "
                "Fokker-Planck equations. Suitable for learning and small problems. "
                "Not recommended for production use."
            ),
            "wrapped": (
                "Wrapper around existing fast solver with custom parameterization. "
                "Demonstrates how plugins can provide custom interfaces to built-in solvers."
            ),
        }
        return descriptions.get(solver_type, "Custom solver provided by example plugin")

    def get_solver_parameters(self, solver_type: str) -> dict[str, Any]:
        """
        Get parameter schema for solver type.

        Returns dictionary mapping parameter names to their specifications:
        - type: Parameter type ("float", "int", "bool", "str")
        - default: Default value
        - range: Valid range [min, max] (for numeric types)
        - description: Human-readable description
        """
        if solver_type == "gradient_descent":
            return {
                "learning_rate": {
                    "type": "float",
                    "default": 0.01,
                    "range": [1e-6, 1.0],
                    "description": "Learning rate for gradient descent steps",
                },
                "max_iterations": {
                    "type": "int",
                    "default": 1000,
                    "range": [1, 10000],
                    "description": "Maximum number of iterations",
                },
                "tolerance": {
                    "type": "float",
                    "default": 1e-6,
                    "range": [1e-12, 1e-2],
                    "description": "Convergence tolerance (L² norm)",
                },
            }
        elif solver_type == "wrapped":
            return {
                "max_iterations": {
                    "type": "int",
                    "default": 50,
                    "range": [1, 200],
                    "description": "Maximum Picard iterations",
                },
                "tolerance": {
                    "type": "float",
                    "default": 1e-6,
                    "range": [1e-12, 1e-2],
                    "description": "Picard convergence tolerance",
                },
            }
        return {}

    def _create_gradient_descent_solver(
        self,
        problem: MFGProblem,
        config: SolverConfig | None = None,
        **kwargs: Any,
    ):
        """
        Create gradient descent solver instance.

        Parameter precedence: kwargs > config > defaults
        """
        # Extract parameters with defaults
        learning_rate = kwargs.get("learning_rate", 0.01)
        max_iterations = kwargs.get("max_iterations", 1000)
        tolerance = kwargs.get("tolerance", 1e-6)

        # Override with config values if provided (modern SolverConfig API)
        if config is not None:
            if "tolerance" not in kwargs:
                tolerance = config.picard.tolerance
            if "max_iterations" not in kwargs:
                max_iterations = config.picard.max_iterations

        return GradientDescentMFGSolver(
            problem=problem,
            learning_rate=learning_rate,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )

    def _create_wrapped_solver(
        self,
        problem: MFGProblem,
        config: SolverConfig | None = None,
        **kwargs: Any,
    ):
        """Create wrapper solver instance."""
        max_iterations = kwargs.get("max_iterations", 50)
        tolerance = kwargs.get("tolerance", 1e-6)

        # Override with config if provided
        if config is not None:
            if "tolerance" not in kwargs:
                tolerance = config.picard.tolerance
            if "max_iterations" not in kwargs:
                max_iterations = config.picard.max_iterations

        return WrapperSolver(problem=problem, max_iterations=max_iterations, tolerance=tolerance)

    def on_load(self):
        """Called when plugin is loaded into plugin manager."""
        print(f"Loading {self.metadata.name} v{self.metadata.version}")
        print(f"  Available solvers: {', '.join(self.get_solver_types())}")

    def on_unload(self):
        """Called when plugin is unloaded from plugin manager."""
        print(f"Unloading {self.metadata.name}")


# Plugin registration function (called automatically by plugin system)
def register():
    """
    Register this plugin with the MFG_PDE plugin system.

    Returns:
        Plugin class (not instance) for lazy instantiation
    """
    return ExampleSolverPlugin


# Direct testing and demonstration
if __name__ == "__main__":
    print("=" * 70)
    print("Example Solver Plugin Demonstration")
    print("=" * 70)

    # Create test problem using modern API
    from mfg_pde import MFGProblem
    from mfg_pde.core.plugin_system import get_plugin_manager

    problem = MFGProblem()  # Uses default 1D problem
    grid_shape = problem.geometry.get_grid_shape()
    print(f"\nTest problem: 1D, Nx={grid_shape[1] - 1}, Nt={grid_shape[0] - 1}, T={problem.T}")

    # Register plugin manually for testing
    plugin_manager = get_plugin_manager()
    plugin = ExampleSolverPlugin()

    # Manually register (in production, this happens automatically via plugin discovery)
    plugin_manager.register_plugin(ExampleSolverPlugin)
    plugin_manager.load_plugin("example_solver_plugin")

    # Test 1: Gradient descent solver
    print("\n" + "-" * 70)
    print("Test 1: Gradient Descent Solver")
    print("-" * 70)

    gd_solver = plugin.create_solver(
        problem,
        "gradient_descent",
        learning_rate=0.02,
        max_iterations=100,
        tolerance=1e-4,
    )

    print("Solving with gradient descent...")
    gd_result = gd_solver.solve()

    print(f"  Converged: {gd_result.converged}")
    print(f"  Iterations: {gd_result.iterations}")
    print(f"  Final error (U): {gd_result.error_history_U[-1]:.2e}")
    print(f"  Final error (M): {gd_result.error_history_M[-1]:.2e}")
    print(f"  Execution time: {gd_result.metadata['execution_time']:.2f}s")

    # Test 2: Wrapped solver
    print("\n" + "-" * 70)
    print("Test 2: Wrapped Solver")
    print("-" * 70)

    wrapped_solver = plugin.create_solver(
        problem,
        "wrapped",
        max_iterations=10,
        tolerance=1e-4,
    )

    print("Solving with wrapped solver...")
    wrapped_result = wrapped_solver.solve()

    print(f"  Converged: {wrapped_result.converged}")
    print(f"  Iterations: {wrapped_result.iterations}")
    print(f"  Final error (U): {wrapped_result.error_history_U[-1]:.2e}")
    print(f"  Final error (M): {wrapped_result.error_history_M[-1]:.2e}")
    print(f"  Wrapped by: {wrapped_result.metadata.get('wrapped_by', 'N/A')}")

    # Test 3: Plugin manager integration
    print("\n" + "-" * 70)
    print("Test 3: Plugin Manager Integration")
    print("-" * 70)

    print("\nAll available solvers from plugin manager:")
    available_solvers = plugin_manager.list_available_solvers()
    for solver_type, info in available_solvers.items():
        if "example" in solver_type or solver_type in ["gradient_descent", "wrapped"]:
            print(f"  {solver_type}:")
            print(f"    Description: {info.get('description', 'N/A')}")
            print(f"    Plugin: {info.get('plugin', 'N/A')}")

    print("\n" + "=" * 70)
    print("Plugin demonstration complete!")
    print("=" * 70)
