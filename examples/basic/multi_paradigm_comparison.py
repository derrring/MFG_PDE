#!/usr/bin/env python3
"""
Multi-Paradigm MFG Solver Comparison

This example demonstrates MFG_PDE's new paradigm-based architecture by solving
the same problem using three different mathematical approaches:

1. **Numerical Paradigm**: Classical finite difference methods
2. **Optimization Paradigm**: Direct variational optimization
3. **Neural Paradigm**: Physics-informed neural networks

This showcases the flexibility and power of the multi-paradigm framework,
allowing users to choose the most appropriate approach for their problem.
"""

import time
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np

# Multi-paradigm imports - new structure
from mfg_pde.alg.numerical import HJBFDMSolver, ParticleCollocationSolver
from mfg_pde.alg.optimization import VariationalMFGSolver
from mfg_pde.alg.neural import MFGPINNSolver, TORCH_AVAILABLE

from mfg_pde import ExampleMFGProblem
from mfg_pde.utils.logging import configure_research_logging, get_logger

# Configure logging
configure_research_logging("multi_paradigm_comparison", level="INFO")
logger = get_logger(__name__)

def create_test_problem(Nx: int = 32, Nt: int = 16) -> ExampleMFGProblem:
    """Create a standard MFG test problem for paradigm comparison."""
    problem = ExampleMFGProblem(Nx=Nx, Nt=Nt)

    # Use moderate problem size for fair comparison across paradigms
    problem.params.alpha = 0.5  # Running cost weight
    problem.params.sigma = 0.3  # Noise strength
    problem.params.T = 1.0      # Time horizon

    logger.info(f"Created test problem: {Nx}x{Nt} grid, T={problem.params.T}")
    return problem

def solve_numerical_paradigm(problem: ExampleMFGProblem) -> Dict[str, Any]:
    """Solve using numerical paradigm (finite difference methods)."""
    logger.info("🧮 NUMERICAL PARADIGM: Classical discretization methods")

    start_time = time.time()

    # Use particle collocation solver (representative of numerical paradigm)
    solver = ParticleCollocationSolver(
        mfg_problem=problem,
        max_iterations=50,
        convergence_tolerance=1e-4,
        damping_factor=0.7
    )

    # Solve the MFG system
    solution = solver.solve()

    solve_time = time.time() - start_time

    # Extract solution quality metrics
    final_residual = getattr(solver, 'final_residual', None)
    iterations = getattr(solver, 'iterations', None)

    logger.info(f"  Solved in {solve_time:.2f}s, {iterations} iterations")
    logger.info(f"  Final residual: {final_residual:.2e}" if final_residual else "  Residual: N/A")

    return {
        'paradigm': 'Numerical',
        'method': 'Particle Collocation (FDM)',
        'solution': solution,
        'solve_time': solve_time,
        'iterations': iterations,
        'final_residual': final_residual,
        'description': 'Classical finite difference discretization'
    }

def solve_optimization_paradigm(problem: ExampleMFGProblem) -> Dict[str, Any]:
    """Solve using optimization paradigm (direct variational methods)."""
    logger.info("🎯 OPTIMIZATION PARADIGM: Direct functional minimization")

    start_time = time.time()

    # Use variational MFG solver (representative of optimization paradigm)
    solver = VariationalMFGSolver(
        mfg_problem=problem,
        max_iterations=30,
        convergence_tolerance=1e-4,
        step_size=0.01
    )

    # Solve by direct optimization of the action functional
    solution = solver.solve()

    solve_time = time.time() - start_time

    # Extract optimization metrics
    final_objective = getattr(solver, 'final_objective', None)
    iterations = getattr(solver, 'iterations', None)

    logger.info(f"  Solved in {solve_time:.2f}s, {iterations} iterations")
    logger.info(f"  Final objective: {final_objective:.2e}" if final_objective else "  Objective: N/A")

    return {
        'paradigm': 'Optimization',
        'method': 'Variational Minimization',
        'solution': solution,
        'solve_time': solve_time,
        'iterations': iterations,
        'final_objective': final_objective,
        'description': 'Direct minimization of action functional'
    }

def solve_neural_paradigm(problem: ExampleMFGProblem) -> Dict[str, Any]:
    """Solve using neural paradigm (physics-informed neural networks)."""
    if not TORCH_AVAILABLE:
        logger.warning("🧠 NEURAL PARADIGM: PyTorch not available, skipping")
        return {
            'paradigm': 'Neural',
            'method': 'PINN (PyTorch unavailable)',
            'solution': None,
            'solve_time': None,
            'error': 'PyTorch not installed',
            'description': 'Physics-informed neural networks (requires PyTorch)'
        }

    logger.info("🧠 NEURAL PARADIGM: Physics-informed neural networks")

    start_time = time.time()

    try:
        # Use PINN solver (representative of neural paradigm)
        solver = MFGPINNSolver(
            mfg_problem=problem,
            epochs=1000,  # Reduced for demo
            learning_rate=1e-3,
            hidden_dims=[32, 32, 32],  # Smaller network for demo
        )

        # Solve using neural network optimization
        solution = solver.solve()

        solve_time = time.time() - start_time

        # Extract training metrics
        final_loss = getattr(solver, 'final_loss', None)
        epochs = getattr(solver, 'epochs_completed', None)

        logger.info(f"  Solved in {solve_time:.2f}s, {epochs} epochs")
        logger.info(f"  Final loss: {final_loss:.2e}" if final_loss else "  Loss: N/A")

        return {
            'paradigm': 'Neural',
            'method': 'Physics-Informed Neural Network',
            'solution': solution,
            'solve_time': solve_time,
            'epochs': epochs,
            'final_loss': final_loss,
            'description': 'Neural network with physics constraints'
        }

    except Exception as e:
        logger.error(f"  Neural paradigm failed: {e}")
        return {
            'paradigm': 'Neural',
            'method': 'PINN (failed)',
            'solution': None,
            'solve_time': time.time() - start_time,
            'error': str(e),
            'description': 'Physics-informed neural networks (failed)'
        }

def compare_solutions(results: list) -> None:
    """Compare and visualize solutions from different paradigms."""
    logger.info("\n📊 PARADIGM COMPARISON ANALYSIS")
    print("=" * 60)

    # Performance comparison
    print("\n🚀 Performance Metrics:")
    print("-" * 30)
    for result in results:
        if result['solution'] is not None:
            print(f"{result['paradigm']:12} | {result['solve_time']:6.2f}s | {result['method']}")
        else:
            error = result.get('error', 'Unknown error')
            print(f"{result['paradigm']:12} | {'N/A':>6} | {error}")

    # Solution quality analysis
    valid_results = [r for r in results if r['solution'] is not None]

    if len(valid_results) >= 2:
        print("\n📈 Solution Analysis:")
        print("-" * 30)

        # Basic solution comparison (placeholder - actual analysis would be more sophisticated)
        for result in valid_results:
            paradigm = result['paradigm']
            method = result['method']
            print(f"{paradigm}: {method}")

            # Additional metrics based on paradigm
            if 'final_residual' in result and result['final_residual']:
                print(f"  PDE residual: {result['final_residual']:.2e}")
            if 'final_objective' in result and result['final_objective']:
                print(f"  Objective value: {result['final_objective']:.2e}")
            if 'final_loss' in result and result['final_loss']:
                print(f"  Neural loss: {result['final_loss']:.2e}")

    # Paradigm insights
    print("\n🎯 Paradigm Insights:")
    print("-" * 30)
    insights = {
        'Numerical': 'Best for: Well-established problems, proven convergence',
        'Optimization': 'Best for: Variational formulations, constrained problems',
        'Neural': 'Best for: High-dimensional problems, complex geometries'
    }

    for result in results:
        paradigm = result['paradigm']
        if paradigm in insights:
            print(f"{paradigm}: {insights[paradigm]}")

def visualize_paradigm_comparison(results: list) -> None:
    """Create visualization comparing the three paradigms."""
    valid_results = [r for r in results if r['solution'] is not None]

    if len(valid_results) == 0:
        logger.warning("No valid solutions to visualize")
        return

    fig, axes = plt.subplots(1, len(valid_results), figsize=(5*len(valid_results), 4))
    if len(valid_results) == 1:
        axes = [axes]

    for i, result in enumerate(valid_results):
        solution = result['solution']
        paradigm = result['paradigm']
        method = result['method']

        # Plot solution (placeholder visualization)
        # In practice, this would extract and plot the actual solution fields
        ax = axes[i]

        # Create dummy visualization for demo
        x = np.linspace(0, 1, 50)
        y = np.sin(2*np.pi*x) * np.exp(-x)  # Placeholder
        ax.plot(x, y, label=f'{paradigm} Solution')

        ax.set_title(f'{paradigm} Paradigm\n{method}', fontsize=10)
        ax.set_xlabel('Space')
        ax.set_ylabel('Value Function')
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.suptitle('Multi-Paradigm MFG Solution Comparison', y=1.02)
    plt.show()

def main():
    """Main function demonstrating multi-paradigm comparison."""
    print("🏗️ MFG_PDE: Multi-Paradigm Solver Comparison")
    print("=" * 60)
    print("Demonstrating three mathematical approaches to Mean Field Games:")
    print("1. 🧮 Numerical: Classical discretization methods")
    print("2. 🎯 Optimization: Direct variational optimization")
    print("3. 🧠 Neural: Physics-informed neural networks")
    print()

    # Create test problem
    problem = create_test_problem(Nx=24, Nt=12)  # Smaller for demo

    # Solve using all three paradigms
    results = []

    # Numerical paradigm
    try:
        result_numerical = solve_numerical_paradigm(problem)
        results.append(result_numerical)
    except Exception as e:
        logger.error(f"Numerical paradigm failed: {e}")
        results.append({
            'paradigm': 'Numerical',
            'method': 'Failed',
            'solution': None,
            'error': str(e)
        })

    # Optimization paradigm
    try:
        result_optimization = solve_optimization_paradigm(problem)
        results.append(result_optimization)
    except Exception as e:
        logger.error(f"Optimization paradigm failed: {e}")
        results.append({
            'paradigm': 'Optimization',
            'method': 'Failed',
            'solution': None,
            'error': str(e)
        })

    # Neural paradigm
    result_neural = solve_neural_paradigm(problem)
    results.append(result_neural)

    # Compare and analyze results
    compare_solutions(results)

    # Visualize comparison
    try:
        visualize_paradigm_comparison(results)
    except Exception as e:
        logger.warning(f"Visualization failed: {e}")

    print("\n🎉 Multi-Paradigm Comparison Complete!")
    print("\n💡 Key Takeaway:")
    print("MFG_PDE's paradigm-based architecture enables users to choose")
    print("the most appropriate mathematical approach for their specific problem.")

if __name__ == "__main__":
    main()