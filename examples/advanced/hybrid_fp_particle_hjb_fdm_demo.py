#!/usr/bin/env python3
"""
Hybrid FP-Particle + HJB-FDM Solver Demonstration

This script demonstrates the new hybrid solver that combines:
- Fokker-Planck equation: Particle-based solution
- Hamilton-Jacobi-Bellman equation: Finite Difference Method (FDM)

This specific combination provides excellent mass conservation (from particles)
with stable and accurate HJB solution (from FDM).
"""

import sys
import time
from pathlib import Path

# Add the package to the path for local development
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


from mfg_pde.alg.numerical.coupling.hybrid_fp_particle_hjb_fdm import (
    HybridSolverPresets,
    create_hybrid_fp_particle_hjb_fdm_solver,
)
from mfg_pde.core.mfg_problem import ExampleMFGProblem
from mfg_pde.factory.solver_factory import create_solver
from mfg_pde.utils.logging import configure_research_logging, get_logger

# Configure logging
configure_research_logging("hybrid_fp_particle_hjb_fdm_demo", level="INFO")
logger = get_logger(__name__)


def create_test_problem():
    """Create a test MFG problem for demonstration."""
    problem_params = {
        "xmin": 0.0,
        "xmax": 1.0,
        "Nx": 50,  # Moderate resolution for demo
        "T": 1.0,
        "Nt": 50,
        "sigma": 1.0,
        "coupling_coefficient": 0.5,
    }
    return ExampleMFGProblem(**problem_params)


def demo_basic_hybrid_solver():
    """Demonstrate basic usage of the hybrid solver."""
    logger.info("=== Basic Hybrid FP-Particle + HJB-FDM Solver Demo ===")

    # Create problem
    problem = create_test_problem()
    logger.info(f"Problem: {problem.Nx} spatial points, {problem.Nt} time points")

    # Create hybrid solver using direct constructor
    hybrid_solver = create_hybrid_fp_particle_hjb_fdm_solver(
        mfg_problem=problem,
        num_particles=5000,
        kde_bandwidth="scott",
        hjb_newton_iterations=30,
        hjb_newton_tolerance=1e-7,
    )

    # Get solver info
    solver_info = hybrid_solver.get_solver_info()
    logger.info(f"Solver: {solver_info['solver_name']}")
    logger.info(f"Particles: {solver_info['num_particles']}")
    logger.info(f"KDE bandwidth: {solver_info['kde_bandwidth']}")
    logger.info(f"HJB Newton iterations: {solver_info['hjb_newton_iterations']}")

    # Solve the problem
    start_time = time.time()
    result = hybrid_solver.solve(max_iterations=30, tolerance=1e-4, damping_factor=0.5)
    solve_time = time.time() - start_time

    # Display results
    if result["converged"]:
        logger.info("✅ Hybrid solver converged!")
        logger.info(f"   Iterations: {result['iterations']}")
        logger.info(f"   Solve time: {solve_time:.2f}s")
        logger.info(f"   Final residual: {result['final_residual']:.2e}")
        logger.info(f"   Time per iteration: {solve_time / result['iterations']:.2f}s")
    else:
        logger.warning("⚠️ Hybrid solver did not converge")
        logger.warning(f"   Final residual: {result['final_residual']:.2e}")

    return result


def demo_hybrid_solver_presets():
    """Demonstrate different hybrid solver presets."""
    logger.info("\n=== Hybrid Solver Presets Demo ===")

    problem = create_test_problem()

    # Test different presets
    presets = {
        "fast": HybridSolverPresets.fast_hybrid,
        "accurate": HybridSolverPresets.accurate_hybrid,
        "research": HybridSolverPresets.research_hybrid,
    }

    results = {}

    for preset_name, preset_func in presets.items():
        logger.info(f"\nTesting {preset_name} preset...")

        try:
            # Create solver with preset
            solver = preset_func(problem)
            solver_info = solver.get_solver_info()

            logger.info(f"  Particles: {solver_info['num_particles']}")
            logger.info(f"  HJB Newton iter: {solver_info['hjb_newton_iterations']}")

            # Solve with reduced iterations for demo
            start_time = time.time()
            result = solver.solve(
                max_iterations=20,  # Reduced for demo
                tolerance=1e-3,  # Relaxed for demo
                damping_factor=0.5,
            )
            solve_time = time.time() - start_time

            results[preset_name] = {
                "converged": result["converged"],
                "iterations": result.get("iterations", 0),
                "solve_time": solve_time,
                "final_residual": result.get("final_residual", float("inf")),
                "num_particles": solver_info["num_particles"],
            }

            status = "✅" if result["converged"] else "❌"
            logger.info(f"  {status} Result: {result['iterations']} iter, {solve_time:.2f}s")

        except Exception as e:
            logger.error(f"  ❌ {preset_name} preset failed: {e}")
            results[preset_name] = {"error": str(e)}

    # Summary
    logger.info("\n--- Preset Comparison Summary ---")
    for preset_name, result in results.items():
        if "error" not in result:
            logger.info(
                f"{preset_name:8}: {result['converged']!s:5} | "
                f"{result['iterations']:2d} iter | "
                f"{result['solve_time']:5.2f}s | "
                f"{result['num_particles']:5d} particles"
            )

    return results


def demo_factory_integration():
    """Demonstrate integration with the solver factory."""
    logger.info("\n=== Factory Integration Demo ===")

    problem = create_test_problem()

    # Create hybrid solver using factory
    try:
        hybrid_solver = create_solver(
            problem=problem,
            solver_type="hybrid_fp_particle_hjb_fdm",
            preset="balanced",
            num_particles=3000,  # Custom parameter
        )

        logger.info("✅ Successfully created hybrid solver via factory")

        # Solve using factory-created solver
        result = hybrid_solver.solve(max_iterations=25, tolerance=1e-4, damping_factor=0.6)

        if result["converged"]:
            logger.info(f"✅ Factory solver converged in {result['iterations']} iterations")
        else:
            logger.warning("⚠️ Factory solver did not converge")

        return result

    except Exception as e:
        logger.error(f"❌ Factory integration failed: {e}")
        return None


def demo_particle_scaling_benchmark():
    """Demonstrate particle scaling benchmark capability."""
    logger.info("\n=== Particle Scaling Benchmark Demo ===")

    problem = create_test_problem()

    # Create solver for benchmarking
    hybrid_solver = HybridSolverPresets.fast_hybrid(problem)

    # Run particle scaling benchmark with reduced scope for demo
    particle_counts = [1000, 2000, 3000, 5000]

    try:
        logger.info("Running particle scaling benchmark...")
        benchmark_results = hybrid_solver.benchmark_particle_scaling(
            particle_counts=particle_counts,
            max_iterations=15,  # Reduced for demo
            tolerance=1e-3,  # Relaxed for demo
        )

        # Display results
        logger.info("\n--- Particle Scaling Results ---")
        logger.info("Particles | Converged | Iterations | Time (s) | Particles/s")
        logger.info("-" * 60)

        for result in benchmark_results["results"]:
            if "error" not in result:
                particles_per_sec = result.get("particles_per_second", 0)
                logger.info(
                    f"{result['num_particles']:8d} | "
                    f"{result['converged']!s:9} | "
                    f"{result['iterations']:9d} | "
                    f"{result['solve_time']:7.2f} | "
                    f"{particles_per_sec:10.0f}"
                )
            else:
                logger.info(f"{result['num_particles']:8d} | ERROR: {result['error']}")

        # Analysis
        analysis = benchmark_results["scaling_analysis"]
        optimal_count = benchmark_results["optimal_particle_count"]

        logger.info("\n--- Analysis ---")
        logger.info(f"Optimal particle count: {optimal_count}")
        logger.info(f"Convergence rate: {analysis['convergence_rate']:.1%}")
        logger.info(f"Efficiency trend: {analysis['efficiency_trend']}")

        if analysis["scaling_coefficient"]:
            logger.info(f"Scaling coefficient: {analysis['scaling_coefficient']:.2f}")

        return benchmark_results

    except Exception as e:
        logger.error(f"❌ Particle scaling benchmark failed: {e}")
        return None


def demo_solver_comparison():
    """Compare hybrid solver with standard fixed point solver."""
    logger.info("\n=== Solver Comparison Demo ===")

    problem = create_test_problem()

    # Results storage
    comparison_results = {}

    # Test hybrid solver
    logger.info("Testing Hybrid FP-Particle + HJB-FDM solver...")
    try:
        hybrid_solver = HybridSolverPresets.fast_hybrid(problem)

        start_time = time.time()
        hybrid_result = hybrid_solver.solve(max_iterations=25, tolerance=1e-4, damping_factor=0.5)
        hybrid_time = time.time() - start_time

        comparison_results["hybrid"] = {
            "converged": hybrid_result["converged"],
            "iterations": hybrid_result.get("iterations", 0),
            "solve_time": hybrid_time,
            "final_residual": hybrid_result.get("final_residual", float("inf")),
            "solver_type": "Hybrid FP-Particle + HJB-FDM",
        }

        status = "✅" if hybrid_result["converged"] else "❌"
        logger.info(f"  {status} Hybrid: {hybrid_result['iterations']} iter, {hybrid_time:.2f}s")

    except Exception as e:
        logger.error(f"  ❌ Hybrid solver failed: {e}")
        comparison_results["hybrid"] = {"error": str(e)}

    # Test standard fixed point solver for comparison
    logger.info("Testing standard Fixed Point solver...")
    try:
        standard_solver = create_solver(problem=problem, solver_type="fixed_point", preset="fast")

        start_time = time.time()
        standard_result = standard_solver.solve(max_iterations=25, tolerance=1e-4, damping_factor=0.5)
        standard_time = time.time() - start_time

        comparison_results["standard"] = {
            "converged": standard_result["converged"],
            "iterations": standard_result.get("iterations", 0),
            "solve_time": standard_time,
            "final_residual": standard_result.get("final_residual", float("inf")),
            "solver_type": "Standard Fixed Point",
        }

        status = "✅" if standard_result["converged"] else "❌"
        logger.info(f"  {status} Standard: {standard_result['iterations']} iter, {standard_time:.2f}s")

    except Exception as e:
        logger.error(f"  ❌ Standard solver failed: {e}")
        comparison_results["standard"] = {"error": str(e)}

    # Summary comparison
    logger.info("\n--- Solver Comparison Summary ---")
    logger.info("Solver Type                    | Converged | Iterations | Time (s) | Residual")
    logger.info("-" * 80)

    for solver_name, result in comparison_results.items():
        if "error" not in result:
            logger.info(
                f"{result['solver_type']:30} | "
                f"{result['converged']!s:9} | "
                f"{result['iterations']:9d} | "
                f"{result['solve_time']:7.2f} | "
                f"{result['final_residual']:.2e}"
            )
        else:
            logger.info(f"{solver_name:30} | ERROR: {result['error']}")

    return comparison_results


def main():
    """Run all hybrid solver demonstrations."""
    logger.info("🚀 Starting Hybrid FP-Particle + HJB-FDM Solver Demonstrations")
    logger.info("=" * 80)

    results = {}

    # Run demonstrations
    try:
        results["basic"] = demo_basic_hybrid_solver()
        results["presets"] = demo_hybrid_solver_presets()
        results["factory"] = demo_factory_integration()
        results["scaling"] = demo_particle_scaling_benchmark()
        results["comparison"] = demo_solver_comparison()

    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("🎯 HYBRID SOLVER DEMONSTRATION SUMMARY")
    logger.info("=" * 80)

    demos = [
        ("Basic Hybrid Solver", results.get("basic")),
        ("Preset Configurations", results.get("presets")),
        ("Factory Integration", results.get("factory")),
        ("Particle Scaling", results.get("scaling")),
        ("Solver Comparison", results.get("comparison")),
    ]

    successful_demos = 0
    total_demos = len(demos)

    for demo_name, demo_result in demos:
        if demo_result is not None:
            if isinstance(demo_result, dict) and demo_result.get("converged", True):
                status = "✅ SUCCESS"
                successful_demos += 1
            elif isinstance(demo_result, dict) and not demo_result.get("converged", False):
                status = "⚠️ NO CONVERGENCE"
                successful_demos += 1  # Still successful execution
            else:
                status = "✅ SUCCESS"
                successful_demos += 1
        else:
            status = "❌ FAILED"

        logger.info(f"   {demo_name}: {status}")

    logger.info(f"\nDemonstrations completed: {successful_demos}/{total_demos}")

    if successful_demos == total_demos:
        logger.info("🎉 ALL HYBRID SOLVER DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        logger.info("The Hybrid FP-Particle + HJB-FDM solver is fully operational.")
    else:
        logger.warning(f"⚠️ {total_demos - successful_demos} demonstrations had issues")

    logger.info("=" * 80)

    return successful_demos == total_demos


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
