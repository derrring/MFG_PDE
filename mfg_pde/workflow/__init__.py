"""
Workflow Management System for MFG_PDE

This module provides a comprehensive workflow management system for reproducible
research, experiment tracking, and collaborative development in mean field games.

Features:
- Reproducible experiment workflows with automatic versioning
- Parameter sweep and batch processing capabilities
- Result comparison and analysis across experiments
- Collaborative workspace management
- Automatic report generation and documentation
- Integration with version control and data provenance

Components:
- WorkflowManager: Core workflow orchestration
- ExperimentTracker: Experiment lifecycle management
- ParameterSweep: Automated parameter space exploration
- ResultAnalyzer: Cross-experiment analysis and comparison
- CollaborativeWorkspace: Multi-user research environment
- ReportGenerator: Automated documentation and visualization
"""

import warnings
from typing import Any, Dict, List, Optional

from .collaborative_workspace import CollaborativeWorkspace, WorkspaceConfig

# Workflow decorators and utilities
from .decorators import experiment, parameter_study, workflow_step
from .experiment_tracker import Experiment, ExperimentResult, ExperimentTracker
from .parameter_sweep import ParameterSweep, SweepConfiguration
from .report_generator import ReportConfig, ReportGenerator
from .result_analyzer import ComparisonReport, ResultAnalyzer
from .utils import create_workspace, load_experiment, save_experiment

# Core workflow components
from .workflow_manager import Workflow, WorkflowManager

# Version and compatibility
__version__ = "1.0.0"
__workflow_api_version__ = "1.0"

# Global workflow manager instance
_global_workflow_manager: Optional[WorkflowManager] = None


def get_workflow_manager() -> WorkflowManager:
    """Get the global workflow manager instance."""
    global _global_workflow_manager
    if _global_workflow_manager is None:
        _global_workflow_manager = WorkflowManager()
    return _global_workflow_manager


def set_workflow_manager(manager: WorkflowManager):
    """Set the global workflow manager instance."""
    global _global_workflow_manager
    _global_workflow_manager = manager


def create_workflow(name: str, description: str = "", **kwargs) -> Workflow:
    """Create a new workflow with the global manager."""
    return get_workflow_manager().create_workflow(name, description, **kwargs)


def create_experiment(
    name: str, workflow: Optional[Workflow] = None, **kwargs
) -> Experiment:
    """Create a new experiment."""
    tracker = ExperimentTracker()
    return tracker.create_experiment(name, workflow=workflow, **kwargs)


def create_parameter_sweep(parameters: Dict[str, Any], **kwargs) -> ParameterSweep:
    """Create a parameter sweep configuration."""
    return ParameterSweep(parameters, **kwargs)


def create_workspace(
    name: str, description: str = "", **kwargs
) -> CollaborativeWorkspace:
    """Create a collaborative workspace."""
    config = WorkspaceConfig(name=name, description=description, **kwargs)
    return CollaborativeWorkspace(config)


def analyze_results(experiments: List[Experiment], **kwargs) -> ComparisonReport:
    """Analyze and compare experimental results."""
    analyzer = ResultAnalyzer()
    return analyzer.compare_experiments(experiments, **kwargs)


def generate_report(
    experiments: List[Experiment], format: str = "html", **kwargs
) -> str:
    """Generate comprehensive report from experiments."""
    generator = ReportGenerator()
    config = ReportConfig(format=format, **kwargs)
    return generator.generate_report(experiments, config)


# Convenience functions for common workflows
def mfg_parameter_study(
    problem_factory,
    parameters: Dict[str, List],
    solver_type: str = "fixed_point",
    **kwargs,
):
    """
    Convenient function for MFG parameter studies.

    Args:
        problem_factory: Function that creates MFG problems
        parameters: Dictionary of parameter names to lists of values
        solver_type: Type of solver to use
        **kwargs: Additional arguments for the workflow

    Returns:
        Workflow with parameter study configured
    """
    workflow = create_workflow(
        name=f"mfg_parameter_study_{solver_type}",
        description=f"Parameter study for MFG system using {solver_type} solver",
    )

    # Create parameter sweep
    sweep = create_parameter_sweep(parameters)

    # Define solve step
    @workflow_step(workflow)
    def solve_mfg_problem(params):
        problem = problem_factory(**params)

        # Import solver dynamically to avoid circular imports
        from mfg_pde import create_fast_solver

        solver = create_fast_solver(problem, solver_type)

        return solver.solve()

    # Execute parameter sweep
    results = sweep.execute(solve_mfg_problem)

    return workflow, results


def convergence_analysis_workflow(
    problem, solver_types: List[str], tolerances: List[float], **kwargs
):
    """
    Workflow for analyzing convergence across different solvers and tolerances.

    Args:
        problem: MFG problem instance
        solver_types: List of solver types to compare
        tolerances: List of tolerance values to test
        **kwargs: Additional workflow arguments

    Returns:
        Workflow with convergence analysis configured
    """
    workflow = create_workflow(
        name="convergence_analysis",
        description="Convergence analysis across solvers and tolerances",
    )

    # Create parameter combinations
    parameters = {"solver_type": solver_types, "tolerance": tolerances}

    sweep = create_parameter_sweep(parameters)

    @workflow_step(workflow)
    def analyze_convergence(params):
        from mfg_pde import create_fast_solver

        solver = create_fast_solver(problem, params["solver_type"])

        # Modify solver tolerance if possible
        if hasattr(solver, "tolerance"):
            solver.tolerance = params["tolerance"]

        result = solver.solve()

        return {
            "solver_type": params["solver_type"],
            "tolerance": params["tolerance"],
            "converged": result.converged,
            "iterations": result.iterations,
            "final_error": result.final_error,
            "execution_time": result.execution_time,
        }

    results = sweep.execute(analyze_convergence)

    return workflow, results


def performance_benchmark_workflow(
    problem_sizes: List[tuple], solver_types: List[str], **kwargs
):
    """
    Workflow for performance benchmarking across problem sizes and solvers.

    Args:
        problem_sizes: List of (Nx, Nt) tuples
        solver_types: List of solver types to benchmark
        **kwargs: Additional workflow arguments

    Returns:
        Workflow with performance benchmarking configured
    """
    workflow = create_workflow(
        name="performance_benchmark",
        description="Performance benchmarking across problem sizes and solvers",
    )

    parameters = {"problem_size": problem_sizes, "solver_type": solver_types}

    sweep = create_parameter_sweep(parameters)

    @workflow_step(workflow)
    def benchmark_performance(params):
        import time

        from mfg_pde import create_fast_solver, ExampleMFGProblem

        Nx, Nt = params["problem_size"]
        problem = ExampleMFGProblem(Nx=Nx, Nt=Nt, T=1.0)

        solver = create_fast_solver(problem, params["solver_type"])

        start_time = time.time()
        result = solver.solve()
        execution_time = time.time() - start_time

        return {
            "problem_size": (Nx, Nt),
            "solver_type": params["solver_type"],
            "grid_points": Nx * Nt,
            "execution_time": execution_time,
            "converged": result.converged,
            "iterations": result.iterations,
            "memory_estimate": Nx * Nt * 16 / 1024**2,  # Rough estimate in MB
        }

    results = sweep.execute(benchmark_performance)

    return workflow, results


# Export public API
__all__ = [
    # Core classes
    "WorkflowManager",
    "Workflow",
    "ExperimentTracker",
    "Experiment",
    "ExperimentResult",
    "ParameterSweep",
    "SweepConfiguration",
    "ResultAnalyzer",
    "ComparisonReport",
    "CollaborativeWorkspace",
    "WorkspaceConfig",
    "ReportGenerator",
    "ReportConfig",
    # Decorators
    "workflow_step",
    "experiment",
    "parameter_study",
    # Factory functions
    "get_workflow_manager",
    "set_workflow_manager",
    "create_workflow",
    "create_experiment",
    "create_parameter_sweep",
    "create_workspace",
    "analyze_results",
    "generate_report",
    # Convenience workflows
    "mfg_parameter_study",
    "convergence_analysis_workflow",
    "performance_benchmark_workflow",
    # Utilities
    "load_experiment",
    "save_experiment",
]


# Initialize workflow system
def _initialize_workflow_system():
    """Initialize the workflow system with default configuration."""
    try:
        # Set up default workspace if needed
        import os

        if not os.getenv("MFG_WORKFLOW_INITIALIZED"):
            manager = get_workflow_manager()
            manager.initialize_default_workspace()
            os.environ["MFG_WORKFLOW_INITIALIZED"] = "true"
    except Exception as e:
        warnings.warn(f"Could not initialize workflow system: {e}")


# Initialize on import
_initialize_workflow_system()
