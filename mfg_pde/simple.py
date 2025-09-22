"""
Simple API for MFG_PDE - Dead Simple Interface

This module provides one-line functions for solving MFG problems
without requiring knowledge of complex types or configurations.

Basic Usage:
    from mfg_pde import solve_mfg

    result = solve_mfg("crowd_dynamics")
    result.plot()

Advanced Usage:
    result = solve_mfg("portfolio_optimization",
                       time_horizon=2.0,
                       domain_size=10.0,
                       accuracy="high")
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import (
    accurate_config,
    crowd_dynamics_config,
    educational_config,
    epidemic_config,
    fast_config,
    financial_config,
    large_scale_config,
    production_config,
    research_config,
    traffic_config,
)
from .hooks import SolverHooks
from .solvers import FixedPointSolver
from .types import MFGResult


def solve_mfg(
    problem_type: str,
    domain_size: float = 1.0,
    time_horizon: float = 1.0,
    accuracy: str = "balanced",
    fast: bool = False,
    verbose: bool = False,
    **kwargs,
) -> MFGResult:
    """
    One-line MFG solver for common problems.

    This function automatically creates the problem, selects the appropriate
    solver, and returns results with sensible defaults.

    Args:
        problem_type: Type of problem to solve:
                     - "crowd_dynamics": Pedestrian/crowd flow
                     - "portfolio_optimization": Financial portfolio
                     - "traffic_flow": Traffic dynamics
                     - "epidemic": Epidemic spreading
                     - "custom": Custom problem (requires additional parameters)

        domain_size: Size of spatial domain (default: 1.0)
        time_horizon: Time horizon T (default: 1.0)

        accuracy: Accuracy level:
                 - "fast": Quick results, moderate accuracy
                 - "balanced": Good balance of speed and accuracy (default)
                 - "high": High accuracy, slower
                 - "research": For research with debugging output
                 - "educational": For teaching/demonstration with detailed output

        fast: If True, prioritize speed over accuracy
        verbose: If True, show detailed solving progress

        **kwargs: Additional problem-specific parameters

    Returns:
        Solution result with u(t,x), m(t,x) and analysis methods

    Examples:
        # Basic usage
        result = solve_mfg("crowd_dynamics")

        # Custom parameters
        result = solve_mfg("portfolio_optimization",
                          domain_size=5.0,
                          time_horizon=2.0,
                          accuracy="high")

        # Quick prototyping
        result = solve_mfg("traffic_flow", fast=True, verbose=True)
    """
    # Create problem based on type
    problem = _create_problem_from_string(problem_type, domain_size, time_horizon, **kwargs)

    # Select configuration based on accuracy requirements
    if fast or accuracy == "fast":
        config = fast_config()
    elif accuracy == "high":
        config = accurate_config()
    elif accuracy == "research":
        config = research_config()
    elif accuracy == "educational":
        config = educational_config()
    else:  # balanced
        config = production_config()

    # Apply user preferences
    if verbose:
        config = config.with_verbose(True)

    # Create and run solver
    solver = FixedPointSolver(max_iterations=config.max_iterations, tolerance=config.tolerance, **kwargs)

    # Add simple hooks if verbose
    hooks = None
    if verbose:
        hooks = _create_simple_debug_hook()

    return solver.solve(problem, hooks=hooks)


def create_mfg_problem(
    problem_type: str, domain: tuple = (0.0, 1.0), time_horizon: float = 1.0, **kwargs
) -> SimpleMFGProblem:
    """
    Create an MFG problem with simple parameters.

    Args:
        problem_type: Type of problem ("crowd_dynamics", "portfolio", etc.)
        domain: Spatial domain as (xmin, xmax)
        time_horizon: Time horizon T
        **kwargs: Problem-specific parameters

    Returns:
        MFG problem ready to solve

    Example:
        problem = create_mfg_problem("crowd_dynamics",
                                   domain=(0, 5),
                                   time_horizon=2.0)
        solver = FixedPointSolver()
        result = solver.solve(problem)
    """
    return _create_problem_from_string(problem_type, domain[1] - domain[0], time_horizon, **kwargs)


def get_config_recommendation(problem_type: str, **kwargs) -> dict[str, Any]:
    """
    Get configuration recommendations for a given problem type.

    This function analyzes the problem type and parameters to provide
    recommendations for optimal solver settings, without actually solving.

    Args:
        problem_type: Type of MFG problem
        **kwargs: Problem-specific parameters

    Returns:
        Dictionary with recommended settings and explanations

    Example:
        rec = get_config_recommendation("crowd_dynamics", crowd_size=500)
        print(f"Recommended tolerance: {rec['tolerance']}")
        print(f"Reason: {rec['explanation']}")
    """
    # Get the auto-selected configuration
    config = auto_config(problem_type, 1.0, 1.0, **kwargs)  # Use default sizes for analysis

    # Build recommendation dict
    recommendation = {
        "tolerance": config.tolerance,
        "max_iterations": config.max_iterations,
        "config_type": "auto-selected",
        "explanation": _explain_config_choice(problem_type, config, **kwargs),
    }

    # Add problem-specific advice
    if problem_type == "crowd_dynamics":
        crowd_size = kwargs.get("crowd_size", 100)
        if crowd_size > 1000:
            recommendation["advice"] = (
                "Large crowd size detected. Consider using domain decomposition for better performance."
            )
        elif crowd_size < 50:
            recommendation["advice"] = (
                "Small crowd size allows for high accuracy. Consider 'accuracy=high' for better results."
            )

    elif problem_type == "portfolio_optimization":
        risk_aversion = kwargs.get("risk_aversion", 0.5)
        if risk_aversion < 0.1:
            recommendation["advice"] = "Low risk aversion can be numerically unstable. Monitor convergence carefully."

    return recommendation


def _explain_config_choice(problem_type: str, config, **kwargs) -> str:
    """Generate explanation for configuration choice."""
    explanations = {
        "crowd_dynamics": "Crowd dynamics problems require stable numerics for high-density regions",
        "portfolio_optimization": "Financial problems need high accuracy and stability for reliable results",
        "traffic_flow": "Traffic flow problems typically converge well and can use faster settings",
        "epidemic": "Epidemic models may have sharp transitions requiring careful convergence criteria",
    }

    base_explanation = explanations.get(problem_type, "Using balanced settings for unknown problem type")

    # Add parameter-specific explanations
    param_notes = []
    if problem_type == "crowd_dynamics" and kwargs.get("crowd_size", 100) > 500:
        param_notes.append("increased iterations for large crowd size")
    elif problem_type == "portfolio_optimization" and kwargs.get("risk_aversion", 0.5) < 0.1:
        param_notes.append("conservative damping for low risk aversion")

    if param_notes:
        base_explanation += f" ({', '.join(param_notes)})"

    return base_explanation


def validate_problem_parameters(problem_type: str, **kwargs) -> dict[str, Any]:
    """
    Validate problem parameters and suggest corrections.

    Args:
        problem_type: Type of MFG problem
        **kwargs: Problem parameters to validate

    Returns:
        Dictionary with validation results and suggestions

    Example:
        validation = validate_problem_parameters("crowd_dynamics", crowd_size=-10)
        if not validation['valid']:
            print("Issues found:", validation['issues'])
            print("Suggestions:", validation['suggestions'])
    """
    issues = []
    suggestions = []
    warnings = []

    # Common validations
    domain_size = kwargs.get("domain_size", 1.0)
    time_horizon = kwargs.get("time_horizon", 1.0)

    if domain_size <= 0:
        issues.append("domain_size must be positive")
        suggestions.append("Set domain_size to a positive value (e.g., 1.0)")

    if time_horizon <= 0:
        issues.append("time_horizon must be positive")
        suggestions.append("Set time_horizon to a positive value (e.g., 1.0)")

    # Problem-specific validations
    if problem_type == "crowd_dynamics":
        crowd_size = kwargs.get("crowd_size", 100)
        if crowd_size <= 0:
            issues.append("crowd_size must be positive")
            suggestions.append("Set crowd_size to a positive integer (e.g., 100)")
        elif crowd_size > 10000:
            warnings.append("Very large crowd_size may lead to slow convergence")
            suggestions.append("Consider using domain decomposition or faster algorithms")

    elif problem_type == "portfolio_optimization":
        risk_aversion = kwargs.get("risk_aversion", 0.5)
        if risk_aversion <= 0:
            issues.append("risk_aversion must be positive")
            suggestions.append("Set risk_aversion to a positive value (typical range: 0.1-2.0)")
        elif risk_aversion < 0.01:
            warnings.append("Very low risk_aversion may cause numerical instability")
            suggestions.append("Consider risk_aversion >= 0.1 for stability")

    elif problem_type == "traffic_flow":
        speed_limit = kwargs.get("speed_limit", 1.0)
        if speed_limit <= 0:
            issues.append("speed_limit must be positive")
            suggestions.append("Set speed_limit to a positive value (e.g., 1.0)")

    elif problem_type == "epidemic":
        infection_rate = kwargs.get("infection_rate", 0.3)
        recovery_rate = kwargs.get("recovery_rate", 0.1)
        if infection_rate < 0 or infection_rate > 1:
            issues.append("infection_rate should be between 0 and 1")
            suggestions.append("Set infection_rate to a value between 0 and 1")
        if recovery_rate < 0 or recovery_rate > 1:
            issues.append("recovery_rate should be between 0 and 1")
            suggestions.append("Set recovery_rate to a value between 0 and 1")

    return {"valid": len(issues) == 0, "issues": issues, "warnings": warnings, "suggestions": suggestions}


def load_example(example_name: str) -> MFGResult:
    """
    Load and solve a predefined example problem.

    Args:
        example_name: Name of example:
                     - "simple_crowd": Basic crowd dynamics
                     - "portfolio_basic": Simple portfolio optimization
                     - "traffic_light": Traffic light problem
                     - "epidemic_basic": Basic epidemic model

    Returns:
        Solved example result

    Example:
        result = load_example("simple_crowd")
        result.plot()
    """
    examples = {
        "simple_crowd": ("crowd_dynamics", {"domain_size": 2.0, "crowd_size": 100}),
        "portfolio_basic": ("portfolio_optimization", {"domain_size": 1.0, "risk_aversion": 0.5}),
        "traffic_light": ("traffic_flow", {"domain_size": 1.0, "signal_period": 1.0}),
        "epidemic_basic": ("epidemic", {"domain_size": 1.0, "infection_rate": 0.3}),
    }

    if example_name not in examples:
        available = ", ".join(examples.keys())
        raise ValueError(f"Unknown example '{example_name}'. Available: {available}")

    problem_type, params = examples[example_name]
    return solve_mfg(problem_type, accuracy="balanced", verbose=True, **params)


def solve_mfg_smart(
    problem_type: str, domain_size: float = 1.0, time_horizon: float = 1.0, validate: bool = True, **kwargs
) -> MFGResult:
    """
    Solve MFG problem with smart defaults, validation, and automatic optimization.

    This is the most intelligent interface that combines parameter validation,
    automatic configuration selection, and adaptive solver behavior.

    Args:
        problem_type: Type of problem to solve
        domain_size: Size of spatial domain (default: 1.0)
        time_horizon: Time horizon T (default: 1.0)
        validate: Whether to validate parameters before solving (default: True)
        **kwargs: Additional problem-specific parameters

    Returns:
        Solution result with automatically optimized settings

    Raises:
        ValueError: If parameter validation fails and validate=True

    Examples:
        # Simplest possible usage - everything automatic
        result = solve_mfg_smart("crowd_dynamics")

        # Automatic validation and optimization
        result = solve_mfg_smart("portfolio_optimization",
                                domain_size=5.0,
                                risk_aversion=0.3)

        # Skip validation for trusted parameters
        result = solve_mfg_smart("traffic_flow", validate=False, speed_limit=2.0)
    """
    # Parameter validation if requested
    if validate:
        validation = validate_problem_parameters(
            problem_type, domain_size=domain_size, time_horizon=time_horizon, **kwargs
        )
        if not validation["valid"]:
            error_msg = "Parameter validation failed:\n"
            for issue in validation["issues"]:
                error_msg += f"  - {issue}\n"
            error_msg += "Suggestions:\n"
            for suggestion in validation["suggestions"]:
                error_msg += f"  - {suggestion}\n"
            raise ValueError(error_msg)

        # Show warnings but continue
        if validation["warnings"]:
            print("Warnings:")
            for warning in validation["warnings"]:
                print(f"  - {warning}")

    # Auto-select best configuration and solve
    return solve_mfg_auto(problem_type, domain_size, time_horizon, **kwargs)


def get_available_problems() -> dict[str, dict[str, Any]]:
    """
    Get information about available problem types and their parameters.

    Returns:
        Dictionary mapping problem types to their descriptions and parameters

    Example:
        problems = get_available_problems()
        for name, info in problems.items():
            print(f"{name}: {info['description']}")
            print(f"  Parameters: {info['parameters']}")
    """
    return {
        "crowd_dynamics": {
            "description": "Pedestrian/crowd flow with congestion effects",
            "parameters": {
                "crowd_size": "Number of individuals (default: 100)",
                "exit_attraction": "Attraction to exit (default: 1.0)",
            },
            "typical_domain_size": 2.0,
            "typical_time_horizon": 1.0,
            "complexity": "medium",
            "recommended_accuracy": "balanced",
        },
        "portfolio_optimization": {
            "description": "Financial portfolio optimization (Merton problem)",
            "parameters": {
                "risk_aversion": "Risk aversion parameter (default: 0.5)",
                "drift": "Expected return rate (default: 0.05)",
            },
            "typical_domain_size": 1.0,
            "typical_time_horizon": 1.0,
            "complexity": high,
            "recommended_accuracy": high,
        },
        "traffic_flow": {
            "description": "Traffic dynamics with congestion",
            "parameters": {
                "speed_limit": "Maximum speed (default: 1.0)",
                "congestion_factor": "Congestion sensitivity (default: 0.5)",
            },
            "typical_domain_size": 1.0,
            "typical_time_horizon": 2.0,
            "complexity": low,
            "recommended_accuracy": fast,
        },
        "epidemic": {
            "description": "Epidemic spreading with control",
            "parameters": {
                "infection_rate": "Rate of infection (default: 0.3)",
                "recovery_rate": "Rate of recovery (default: 0.1)",
            },
            "typical_domain_size": 1.0,
            "typical_time_horizon": 5.0,
            "complexity": "medium",
            "recommended_accuracy": "balanced",
        },
    }


def suggest_problem_setup(problem_type: str) -> dict[str, Any]:
    """
    Suggest optimal problem setup for a given problem type.

    Args:
        problem_type: Type of MFG problem

    Returns:
        Dictionary with suggested parameters and settings

    Example:
        setup = suggest_problem_setup("crowd_dynamics")
        print(f"Suggested domain size: {setup['domain_size']}")
        print(f"Suggested parameters: {setup['parameters']}")
    """
    problems_info = get_available_problems()

    if problem_type not in problems_info:
        available = list(problems_info.keys())
        raise ValueError(f"Unknown problem type '{problem_type}'. Available: {available}")

    info = problems_info[problem_type]

    # Build parameter suggestions
    suggested_params = {}
    if problem_type == "crowd_dynamics":
        suggested_params = {"crowd_size": 100, "exit_attraction": 1.0}
    elif problem_type == "portfolio_optimization":
        suggested_params = {"risk_aversion": 0.5, "drift": 0.05}
    elif problem_type == "traffic_flow":
        suggested_params = {"speed_limit": 1.0, "congestion_factor": 0.5}
    elif problem_type == "epidemic":
        suggested_params = {"infection_rate": 0.3, "recovery_rate": 0.1}

    return {
        "problem_type": problem_type,
        "domain_size": info["typical_domain_size"],
        "time_horizon": info["typical_time_horizon"],
        "accuracy": info["recommended_accuracy"],
        "parameters": suggested_params,
        "description": info["description"],
        "complexity": info["complexity"],
    }


# Internal helper functions


def _create_problem_from_string(
    problem_type: str, domain_size: float, time_horizon: float, **kwargs
) -> SimpleMFGProblem:
    """Create a problem instance from string specification."""
    if problem_type == "crowd_dynamics":
        return _create_crowd_dynamics_problem(domain_size, time_horizon, **kwargs)
    elif problem_type == "portfolio_optimization":
        return _create_portfolio_problem(domain_size, time_horizon, **kwargs)
    elif problem_type == "traffic_flow":
        return _create_traffic_problem(domain_size, time_horizon, **kwargs)
    elif problem_type == "epidemic":
        return _create_epidemic_problem(domain_size, time_horizon, **kwargs)
    elif problem_type == "custom":
        return _create_custom_problem(domain_size, time_horizon, **kwargs)
    else:
        available = ["crowd_dynamics", "portfolio_optimization", "traffic_flow", "epidemic", "custom"]
        raise ValueError(f"Unknown problem type '{problem_type}'. Available: {available}")


class SimpleMFGProblem:
    """
    Simple MFG problem implementation that satisfies the MFGProblem protocol.

    This class provides a straightforward implementation of common MFG problems
    without requiring users to understand the full complexity of the framework.
    """

    def __init__(
        self,
        problem_type: str,
        domain_bounds: tuple,
        time_horizon: float,
        hamiltonian_func: callable,
        initial_density_func: callable,
        terminal_value_func: callable,
        **metadata,
    ):
        self.problem_type = problem_type
        self._domain_bounds = domain_bounds
        self._time_horizon = time_horizon
        self._hamiltonian_func = hamiltonian_func
        self._initial_density_func = initial_density_func
        self._terminal_value_func = terminal_value_func
        self.metadata = metadata

    def get_domain_bounds(self) -> tuple:
        """Get spatial domain bounds (xmin, xmax)."""
        return self._domain_bounds

    def get_time_horizon(self) -> float:
        """Get time horizon T."""
        return self._time_horizon

    def evaluate_hamiltonian(self, x: float, p: float, m: float, t: float) -> float:
        """Evaluate Hamiltonian H(x, p, m, t)."""
        return self._hamiltonian_func(x, p, m, t)

    def get_initial_density(self) -> np.ndarray:
        """Get initial density m_0(x)."""
        xmin, xmax = self._domain_bounds
        x_grid = np.linspace(xmin, xmax, 101)  # Default grid
        density = np.array([self._initial_density_func(x) for x in x_grid])
        # Normalize
        density = density / np.trapz(density, x_grid)
        return density

    def get_initial_value_function(self) -> np.ndarray:
        """Get terminal value function g(x)."""
        xmin, xmax = self._domain_bounds
        x_grid = np.linspace(xmin, xmax, 101)  # Default grid
        return np.array([self._terminal_value_func(x) for x in x_grid])

    def __str__(self) -> str:
        return f"SimpleMFGProblem({self.problem_type}, domain={self._domain_bounds}, T={self._time_horizon})"


def _create_crowd_dynamics_problem(domain_size: float, time_horizon: float, **kwargs) -> SimpleMFGProblem:
    """Create a crowd dynamics problem."""
    crowd_size = kwargs.get("crowd_size", 100)
    exit_attraction = kwargs.get("exit_attraction", 1.0)

    def hamiltonian(x, p, m, t):
        # Simple crowd dynamics Hamiltonian
        kinetic = 0.5 * p**2
        interaction = 0.1 * m  # Congestion cost
        exit_cost = exit_attraction * (x - domain_size) ** 2  # Attraction to exit
        return kinetic + interaction + exit_cost

    def initial_density(x):
        # Gaussian initial distribution
        center = domain_size * 0.2  # Start near entrance
        width = domain_size * 0.1
        return np.exp(-0.5 * ((x - center) / width) ** 2)

    def terminal_value(x):
        # Cost of not reaching exit
        return (x - domain_size) ** 2

    return SimpleMFGProblem(
        "crowd_dynamics",
        (0.0, domain_size),
        time_horizon,
        hamiltonian,
        initial_density,
        terminal_value,
        crowd_size=crowd_size,
        exit_attraction=exit_attraction,
    )


def _create_portfolio_problem(domain_size: float, time_horizon: float, **kwargs) -> SimpleMFGProblem:
    """Create a portfolio optimization problem."""
    risk_aversion = kwargs.get("risk_aversion", 0.5)
    drift = kwargs.get("drift", 0.05)

    def hamiltonian(x, p, m, t):
        # Merton portfolio problem
        kinetic = 0.5 * p**2 / risk_aversion
        market_impact = 0.01 * m * p  # Market impact
        return kinetic + market_impact

    def initial_density(x):
        # Initial wealth distribution
        return np.exp(-((x - domain_size / 2) ** 2) / (0.1 * domain_size))

    def terminal_value(x):
        # Utility of terminal wealth
        return -np.log(x + 0.01)  # Log utility

    return SimpleMFGProblem(
        "portfolio_optimization",
        (0.0, domain_size),
        time_horizon,
        hamiltonian,
        initial_density,
        terminal_value,
        risk_aversion=risk_aversion,
        drift=drift,
    )


def _create_traffic_problem(domain_size: float, time_horizon: float, **kwargs) -> SimpleMFGProblem:
    """Create a traffic flow problem."""
    speed_limit = kwargs.get("speed_limit", 1.0)
    congestion_factor = kwargs.get("congestion_factor", 0.5)

    def hamiltonian(x, p, m, t):
        # Traffic flow Hamiltonian
        desired_speed = speed_limit * (1 - congestion_factor * m)
        cost = 0.5 * (p - desired_speed) ** 2
        return cost

    def initial_density(x):
        # Uniform initial traffic distribution
        return np.ones_like(x) if hasattr(x, "__len__") else 1.0

    def terminal_value(x):
        # Minimal terminal cost
        return 0.1 * x**2

    return SimpleMFGProblem(
        "traffic_flow",
        (0.0, domain_size),
        time_horizon,
        hamiltonian,
        initial_density,
        terminal_value,
        speed_limit=speed_limit,
        congestion_factor=congestion_factor,
    )


def _create_epidemic_problem(domain_size: float, time_horizon: float, **kwargs) -> SimpleMFGProblem:
    """Create an epidemic spreading problem."""
    infection_rate = kwargs.get("infection_rate", 0.3)
    recovery_rate = kwargs.get("recovery_rate", 0.1)

    def hamiltonian(x, p, m, t):
        # Epidemic control Hamiltonian
        control_cost = 0.5 * p**2
        infection_cost = infection_rate * m * (1 - x)  # Cost of being susceptible
        return control_cost + infection_cost

    def initial_density(x):
        # Most population initially susceptible
        return np.exp(-10 * x)  # Concentrated near x=0 (susceptible)

    def terminal_value(x):
        # Prefer being recovered
        return -((x - 1) ** 2)

    return SimpleMFGProblem(
        "epidemic",
        (0.0, domain_size),
        time_horizon,
        hamiltonian,
        initial_density,
        terminal_value,
        infection_rate=infection_rate,
        recovery_rate=recovery_rate,
    )


def _create_custom_problem(domain_size: float, time_horizon: float, **kwargs) -> SimpleMFGProblem:
    """Create a custom problem from user-provided functions."""
    hamiltonian = kwargs.get("hamiltonian")
    initial_density = kwargs.get("initial_density")
    terminal_value = kwargs.get("terminal_value")

    if not all([hamiltonian, initial_density, terminal_value]):
        raise ValueError("Custom problems require 'hamiltonian', 'initial_density', and 'terminal_value' functions")

    return SimpleMFGProblem(
        "custom", (0.0, domain_size), time_horizon, hamiltonian, initial_density, terminal_value, **kwargs
    )


def _create_simple_debug_hook() -> SolverHooks:
    """Create a simple debugging hook for verbose output."""

    class SimpleDebugHook(SolverHooks):
        def on_solve_start(self, initial_state):
            print(f"Starting MFG solve with initial residual: {initial_state.residual:.2e}")

        def on_iteration_end(self, state):
            if state.iteration % 10 == 0:
                print(f"Iteration {state.iteration:3d}: residual = {state.residual:.2e}")

        def on_solve_end(self, result):
            if result.converged:
                print(f"✓ Converged in {result.iterations} iterations")
            else:
                print(f"✗ Did not converge after {result.iterations} iterations")
            return result

    return SimpleDebugHook()


def auto_config(problem_type: str, domain_size: float, time_horizon: float, **kwargs):
    """
    Automatically select the best configuration based on problem characteristics.

    This function analyzes the problem parameters and automatically chooses
    the most appropriate solver configuration using problem-specific presets
    and then fine-tuning based on parameter values.

    Args:
        problem_type: Type of MFG problem
        domain_size: Spatial domain size
        time_horizon: Time horizon
        **kwargs: Additional problem parameters

    Returns:
        Recommended SolverConfig instance
    """
    # Start with problem-specific configuration
    if problem_type == "crowd_dynamics":
        config = crowd_dynamics_config()
        # Fine-tune based on crowd characteristics
        crowd_size = kwargs.get("crowd_size", 100)
        if crowd_size > 1000:
            config = config.with_tolerance(1e-4).with_max_iterations(400)
        elif crowd_size < 50:
            config = config.with_tolerance(1e-6).with_max_iterations(150)

    elif problem_type == "portfolio_optimization":
        config = financial_config()
        # Fine-tune based on financial parameters
        risk_aversion = kwargs.get("risk_aversion", 0.5)
        if risk_aversion < 0.1:
            # Very low risk aversion needs extra stability
            config = config.with_damping(0.6).with_max_iterations(800)
        elif risk_aversion > 2.0:
            # High risk aversion converges more easily
            config = config.with_tolerance(1e-5).with_max_iterations(200)

    elif problem_type == "traffic_flow":
        config = traffic_config()
        # Fine-tune based on traffic parameters
        congestion_factor = kwargs.get("congestion_factor", 0.5)
        if congestion_factor > 0.8:
            # High congestion needs more iterations
            config = config.with_max_iterations(250)

    elif problem_type == "epidemic":
        config = epidemic_config()
        # Fine-tune based on epidemic parameters
        infection_rate = kwargs.get("infection_rate", 0.3)
        if infection_rate > 0.7:
            # High infection rate has sharp transitions
            config = config.with_tolerance(1e-7).with_damping(0.6)

    else:
        # Unknown problem type - use balanced default
        config = production_config()

    # Adjust based on overall problem size
    estimated_complexity = domain_size * time_horizon

    if estimated_complexity > 20.0:
        # Very large problem - switch to large-scale config
        config = large_scale_config()
    elif estimated_complexity > 10.0:
        # Large problem - reduce tolerance slightly for speed
        config = config.with_tolerance(max(config.tolerance * 2, 1e-4))
    elif estimated_complexity < 0.1:
        # Small problem - can afford higher accuracy
        config = config.with_tolerance(min(config.tolerance / 2, 1e-8))

    return config


def solve_mfg_auto(
    problem_type: str, domain_size: float = 1.0, time_horizon: float = 1.0, verbose: bool = False, **kwargs
) -> MFGResult:
    """
    Solve MFG problem with automatic configuration selection.

    This function automatically analyzes the problem parameters and selects
    the most appropriate solver configuration, removing the need for users
    to understand accuracy vs speed tradeoffs.

    Args:
        problem_type: Type of problem ("crowd_dynamics", "portfolio_optimization", etc.)
        domain_size: Size of spatial domain (default: 1.0)
        time_horizon: Time horizon T (default: 1.0)
        verbose: If True, show detailed solving progress
        **kwargs: Additional problem-specific parameters

    Returns:
        Solution result with automatically optimized settings

    Examples:
        # Automatic configuration - just specify the problem
        result = solve_mfg_auto("crowd_dynamics", domain_size=5.0, crowd_size=200)

        # Still supports custom parameters
        result = solve_mfg_auto("portfolio_optimization",
                               time_horizon=2.0,
                               risk_aversion=0.1,
                               verbose=True)
    """
    # Create problem
    problem = _create_problem_from_string(problem_type, domain_size, time_horizon, **kwargs)

    # Auto-select configuration
    config = auto_config(problem_type, domain_size, time_horizon, **kwargs)

    # Apply verbosity preference
    if verbose:
        config = config.with_verbose(True)

    # Create and run solver
    solver = FixedPointSolver(max_iterations=config.max_iterations, tolerance=config.tolerance, **kwargs)

    # Add hooks if verbose
    hooks = None
    if verbose:
        hooks = _create_simple_debug_hook()

    return solver.solve(problem, hooks=hooks)
