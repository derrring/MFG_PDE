#!/usr/bin/env python3
"""
Fixed Method Comparison
Properly compares FDM, Hybrid, and Improved QP-Collocation methods
using the correct solver interfaces.
"""

import time
import warnings

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

from mfg_pde.alg.damped_fixed_point_iterator import FixedPointIterator
from mfg_pde.alg.fp_solvers.fdm_fp import FdmFPSolver
from mfg_pde.alg.fp_solvers.particle_fp import ParticleFPSolver
from mfg_pde.alg.hjb_solvers.fdm_hjb import FdmHJBSolver
from mfg_pde.alg.hjb_solvers.optimized_gfdm_hjb_v2 import OptimizedGFDMHJBSolver
from mfg_pde.alg.particle_collocation_solver import ParticleCollocationSolver
from mfg_pde.core.boundaries import BoundaryConditions
from mfg_pde.core.mfg_problem import ExampleMFGProblem


def test_method_comparison():
    """Run fixed comparison of the three methods"""
    print("=" * 80)
    print("FIXED THREE METHOD COMPARISON")
    print("=" * 80)
    print("Methods: Pure FDM, Hybrid Particle-FDM, Improved QP-Collocation")

    # Problem parameters - moderate size for reasonable execution time
    problem_params = {"xmin": 0.0, "xmax": 1.0, "Nx": 20, "T": 1.0, "Nt": 40, "sigma": 0.15, "coefCT": 0.02}

    print(f"Problem: Nx={problem_params['Nx']}, T={problem_params['T']}, Nt={problem_params['Nt']}")
    print(f"Parameters: σ={problem_params['sigma']}, coefCT={problem_params['coefCT']}")

    results = {}

    # Method 1: Pure FDM
    print(f"\n{'-' * 60}")
    print("TESTING PURE FDM METHOD")
    print(f"{'-' * 60}")

    try:
        problem = ExampleMFGProblem(**problem_params)

        # Create FDM HJB solver
        hjb_solver = FdmHJBSolver(problem, NiterNewton=30, l2errBoundNewton=1e-6)

        # Create FDM FP solver
        fp_solver = FdmFPSolver(problem)

        # Create fixed point iterator
        fdm_solver = FixedPointIterator(problem=problem, hjb_solver=hjb_solver, fp_solver=fp_solver, thetaUM=0.5)

        print("Running FDM solver...")
        start_time = time.time()
        U_fdm, M_fdm, iterations_fdm, rel_err_u_fdm, rel_err_m_fdm = fdm_solver.solve(
            Niter_max=10, l2errBoundPicard=1e-3
        )
        fdm_time = time.time() - start_time

        # Calculate mass conservation
        Dx = (problem_params["xmax"] - problem_params["xmin"]) / problem_params["Nx"]
        initial_mass_fdm = np.sum(M_fdm[0, :]) * Dx
        final_mass_fdm = np.sum(M_fdm[-1, :]) * Dx
        mass_error_fdm = abs(final_mass_fdm - initial_mass_fdm) / initial_mass_fdm * 100

        results["fdm"] = {
            "method": "Pure FDM",
            "success": True,
            "time": fdm_time,
            "mass_error": mass_error_fdm,
            "converged": rel_err_u_fdm[-1] < 1e-3 and rel_err_m_fdm[-1] < 1e-3,
            "iterations": iterations_fdm,
            "U": U_fdm,
            "M": M_fdm,
            "final_error": max(rel_err_u_fdm[-1], rel_err_m_fdm[-1]),
        }

        print(f"✓ FDM completed: {fdm_time:.1f}s, mass error: {mass_error_fdm:.2f}%, iterations: {iterations_fdm}")

    except Exception as e:
        print(f"✗ FDM failed: {e}")
        results["fdm"] = {"method": "Pure FDM", "success": False, "error": str(e)}

    # Method 2: Hybrid Particle-FDM
    print(f"\n{'-' * 60}")
    print("TESTING HYBRID PARTICLE-FDM METHOD")
    print(f"{'-' * 60}")

    try:
        problem = ExampleMFGProblem(**problem_params)

        # Create FDM HJB solver
        hjb_solver = FdmHJBSolver(problem, NiterNewton=30, l2errBoundNewton=1e-6)

        # Create Particle FP solver
        fp_solver = ParticleFPSolver(
            problem,
            num_particles=1500,
            kde_bandwidth="scott",
            normalize_kde_output=True,
            boundary_conditions=BoundaryConditions(type="no_flux"),
        )

        # Create fixed point iterator
        hybrid_solver = FixedPointIterator(problem=problem, hjb_solver=hjb_solver, fp_solver=fp_solver, thetaUM=0.3)

        print("Running Hybrid solver...")
        start_time = time.time()
        U_hybrid, M_hybrid, iterations_hybrid, rel_err_u_hybrid, rel_err_m_hybrid = hybrid_solver.solve(
            Niter_max=10, l2errBoundPicard=1e-3
        )
        hybrid_time = time.time() - start_time

        # Calculate mass conservation
        Dx = (problem_params["xmax"] - problem_params["xmin"]) / problem_params["Nx"]
        initial_mass_hybrid = np.sum(M_hybrid[0, :]) * Dx
        final_mass_hybrid = np.sum(M_hybrid[-1, :]) * Dx
        mass_error_hybrid = abs(final_mass_hybrid - initial_mass_hybrid) / initial_mass_hybrid * 100

        results["hybrid"] = {
            "method": "Hybrid Particle-FDM",
            "success": True,
            "time": hybrid_time,
            "mass_error": mass_error_hybrid,
            "converged": rel_err_u_hybrid[-1] < 1e-3 and rel_err_m_hybrid[-1] < 1e-3,
            "iterations": iterations_hybrid,
            "U": U_hybrid,
            "M": M_hybrid,
            "final_error": max(rel_err_u_hybrid[-1], rel_err_m_hybrid[-1]),
        }

        print(
            f"✓ Hybrid completed: {hybrid_time:.1f}s, mass error: {mass_error_hybrid:.2f}%, iterations: {iterations_hybrid}"
        )

    except Exception as e:
        print(f"✗ Hybrid failed: {e}")
        results["hybrid"] = {"method": "Hybrid Particle-FDM", "success": False, "error": str(e)}

    # Method 3: Improved QP-Collocation
    print(f"\n{'-' * 60}")
    print("TESTING IMPROVED QP-COLLOCATION METHOD")
    print(f"{'-' * 60}")

    try:
        problem = ExampleMFGProblem(**problem_params)
        no_flux_bc = BoundaryConditions(type="no_flux")

        # Setup collocation points
        num_collocation_points = 10
        collocation_points = np.linspace(problem.xmin, problem.xmax, num_collocation_points).reshape(-1, 1)

        # Identify boundary points
        boundary_indices = [0, num_collocation_points - 1]

        # Create optimized HJB solver
        optimized_hjb_solver = OptimizedGFDMHJBSolver(
            problem=problem,
            collocation_points=collocation_points,
            delta=0.4,
            taylor_order=2,
            weight_function="wendland",
            NiterNewton=4,
            l2errBoundNewton=1e-3,
            boundary_indices=np.array(boundary_indices),
            boundary_conditions=no_flux_bc,
            use_monotone_constraints=True,
            qp_activation_tolerance=1e-3,
        )

        # Create particle collocation solver
        qp_solver = ParticleCollocationSolver(
            problem=problem,
            collocation_points=collocation_points,
            num_particles=150,
            delta=0.4,
            taylor_order=2,
            weight_function="wendland",
            NiterNewton=4,
            l2errBoundNewton=1e-3,
            kde_bandwidth="scott",
            normalize_kde_output=False,
            boundary_indices=np.array(boundary_indices),
            boundary_conditions=no_flux_bc,
            use_monotone_constraints=True,
        )

        # Replace HJB solver with optimized version
        qp_solver.hjb_solver = optimized_hjb_solver

        print("Running Improved QP-Collocation solver...")
        start_time = time.time()
        U_qp, M_qp, info_qp = qp_solver.solve(Niter=8, l2errBound=1e-3, verbose=True)
        qp_time = time.time() - start_time

        # Calculate mass conservation
        Dx = (problem_params["xmax"] - problem_params["xmin"]) / problem_params["Nx"]
        initial_mass_qp = np.sum(M_qp[0, :]) * Dx
        final_mass_qp = np.sum(M_qp[-1, :]) * Dx
        mass_error_qp = abs(final_mass_qp - initial_mass_qp) / initial_mass_qp * 100

        # Get optimization statistics
        optimization_stats = {}
        if hasattr(optimized_hjb_solver, "get_performance_report"):
            optimization_stats = optimized_hjb_solver.get_performance_report()

        results["qp"] = {
            "method": "Improved QP-Collocation",
            "success": True,
            "time": qp_time,
            "mass_error": mass_error_qp,
            "converged": info_qp.get("converged", False),
            "iterations": info_qp.get("iterations", 0),
            "U": U_qp,
            "M": M_qp,
            "optimization_stats": optimization_stats,
            "final_error": info_qp.get("final_error", 0),
        }

        print(
            f"✓ QP-Collocation completed: {qp_time:.1f}s, mass error: {mass_error_qp:.2f}%, iterations: {info_qp.get('iterations', 0)}"
        )

        # Print optimization statistics
        if optimization_stats:
            print("QP Optimization Statistics:")
            print(f"  QP Activation Rate: {optimization_stats.get('qp_activation_rate', 0):.1%}")
            print(f"  QP Skip Rate: {optimization_stats.get('qp_skip_rate', 0):.1%}")
            print(f"  Total QP Calls: {optimization_stats.get('total_qp_calls', 0)}")

            if optimization_stats.get("qp_skip_rate", 0) > 0:
                estimated_speedup = 1 / (1 - optimization_stats["qp_skip_rate"] * 0.9)
                print(f"  Estimated Speedup from Optimization: {estimated_speedup:.1f}x")

    except Exception as e:
        print(f"✗ QP-Collocation failed: {e}")
        import traceback

        traceback.print_exc()
        results["qp"] = {"method": "Improved QP-Collocation", "success": False, "error": str(e)}

    # Print summary
    print(f"\n{'=' * 80}")
    print("COMPARISON SUMMARY")
    print(f"{'=' * 80}")

    print(f"\n{'Method':<25} {'Success':<8} {'Time(s)':<10} {'Mass Error %':<12} {'Converged':<10} {'Iterations':<10}")
    print("-" * 85)

    for key, result in results.items():
        if result["success"]:
            success_str = "✓"
            time_str = f"{result['time']:.1f}"
            mass_str = f"{result['mass_error']:.2f}"
            conv_str = "✓" if result["converged"] else "✗"
            iter_str = f"{result['iterations']}"
        else:
            success_str = "✗"
            time_str = "N/A"
            mass_str = "N/A"
            conv_str = "N/A"
            iter_str = "N/A"

        print(f"{result['method']:<25} {success_str:<8} {time_str:<10} {mass_str:<12} {conv_str:<10} {iter_str:<10}")

    # Performance analysis
    successful_results = {k: v for k, v in results.items() if v["success"]}

    if len(successful_results) > 1:
        print("\nPERFORMANCE ANALYSIS:")
        print("-" * 40)

        # Find fastest method
        fastest = min(successful_results.items(), key=lambda x: x[1]["time"])
        print(f"Fastest Method: {fastest[1]['method']} ({fastest[1]['time']:.1f}s)")

        # Find most accurate method
        most_accurate = min(successful_results.items(), key=lambda x: x[1]["mass_error"])
        print(f"Best Mass Conservation: {most_accurate[1]['method']} ({most_accurate[1]['mass_error']:.2f}% error)")

        # Calculate relative speedups
        if "fdm" in successful_results:
            fdm_time = successful_results["fdm"]["time"]
            print("\nSpeedup vs FDM:")
            for key, result in successful_results.items():
                if key != "fdm":
                    speedup = fdm_time / result["time"]
                    print(f"  {result['method']}: {speedup:.2f}x")

    # QP Optimization Analysis
    if "qp" in results and results["qp"]["success"] and "optimization_stats" in results["qp"]:
        stats = results["qp"]["optimization_stats"]
        if stats:
            print("\nQP OPTIMIZATION ANALYSIS:")
            print("-" * 40)
            print(f"QP Activation Rate: {stats.get('qp_activation_rate', 0):.1%}")
            print(f"QP Skip Rate: {stats.get('qp_skip_rate', 0):.1%}")
            print(f"Total QP Calls: {stats.get('total_qp_calls', 0)}")
            print(f"QP Calls Skipped: {stats.get('qp_calls_skipped', 0)}")

            if stats.get("qp_skip_rate", 0) > 0:
                estimated_speedup = 1 / (1 - stats["qp_skip_rate"] * 0.9)
                print(f"Estimated Speedup from QP Optimization: {estimated_speedup:.1f}x")

    # Create comparison plots
    create_comparison_plots(results, problem_params)

    return results


def create_comparison_plots(results, problem_params):
    """Create comparison plots"""
    successful_results = {k: v for k, v in results.items() if v["success"]}

    if len(successful_results) == 0:
        print("No successful results to plot")
        return

    _fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Solve Time Comparison
    methods = [r["method"] for r in successful_results.values()]
    times = [r["time"] for r in successful_results.values()]
    colors = ["red", "blue", "green"][: len(methods)]

    bars1 = ax1.bar(methods, times, color=colors, alpha=0.7)
    ax1.set_ylabel("Solve Time (seconds)")
    ax1.set_title("Computational Time Comparison")
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, time_val in zip(bars1, times, strict=False):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, height + 0.5, f"{time_val:.1f}s", ha="center", va="bottom")

    # Plot 2: Mass Conservation Error
    mass_errors = [r["mass_error"] for r in successful_results.values()]

    bars2 = ax2.bar(methods, mass_errors, color=colors, alpha=0.7)
    ax2.set_ylabel("Mass Conservation Error (%)")
    ax2.set_title("Mass Conservation Quality")
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale("log")

    # Add value labels
    for bar, error in zip(bars2, mass_errors, strict=False):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, height * 1.1, f"{error:.2f}%", ha="center", va="bottom")

    # Plot 3: Final Density Profiles
    x_grid = np.linspace(
        problem_params["xmin"],
        problem_params["xmax"],
        successful_results[next(iter(successful_results.keys()))]["M"].shape[1],
    )

    for i, (_key, result) in enumerate(successful_results.items()):
        M = result["M"]
        ax3.plot(x_grid, M[-1, :], label=f"{result['method']} (final)", color=colors[i], linewidth=2)
        ax3.plot(
            x_grid,
            M[0, :],
            label=f"{result['method']} (initial)",
            color=colors[i],
            linewidth=1,
            linestyle="--",
            alpha=0.7,
        )

    ax3.set_xlabel("x")
    ax3.set_ylabel("Density")
    ax3.set_title("Density Evolution Comparison")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: QP Optimization Statistics (if available)
    qp_stats = None
    for result in successful_results.values():
        if result.get("optimization_stats"):
            qp_stats = result["optimization_stats"]
            break

    if qp_stats:
        categories = ["QP Activation Rate", "QP Skip Rate"]
        values = [qp_stats.get("qp_activation_rate", 0) * 100, qp_stats.get("qp_skip_rate", 0) * 100]

        bars4 = ax4.bar(categories, values, color=["orange", "green"], alpha=0.7)
        ax4.set_ylabel("Rate (%)")
        ax4.set_title("QP Optimization Performance")
        ax4.tick_params(axis="x", rotation=45)
        ax4.grid(True, alpha=0.3)

        # Add value labels
        for bar, value in zip(bars4, values, strict=False):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2.0, height + 1, f"{value:.1f}%", ha="center", va="bottom")

        # Add estimated speedup text
        if qp_stats.get("qp_skip_rate", 0) > 0:
            estimated_speedup = 1 / (1 - qp_stats["qp_skip_rate"] * 0.9)
            ax4.text(
                0.5,
                0.9,
                f"Estimated Speedup: {estimated_speedup:.1f}x",
                transform=ax4.transAxes,
                ha="center",
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "yellow", "alpha": 0.5},
            )
    else:
        ax4.text(
            0.5,
            0.5,
            "No QP optimization\nstatistics available",
            transform=ax4.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            style="italic",
        )
        ax4.set_title("QP Optimization Statistics")

    plt.tight_layout()
    plt.savefig(
        "/Users/zvezda/Library/CloudStorage/OneDrive-Personal/code/MFG_PDE/tests/method_comparisons/fixed_method_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    print("\nComparison plots saved to: fixed_method_comparison.png")
    plt.show()


def main():
    """Run the fixed method comparison"""
    print("Starting Fixed Three Method Comparison...")
    print("This compares FDM, Hybrid, and Improved QP-Collocation methods with correct interfaces")
    print("Expected execution time: 10-30 minutes")

    try:
        results = test_method_comparison()

        print(f"\n{'=' * 80}")
        print("FIXED METHOD COMPARISON COMPLETED")
        print(f"{'=' * 80}")
        print("Check the summary above and generated plots for detailed comparison.")

        return results

    except KeyboardInterrupt:
        print("\nComparison interrupted by user.")
        return None
    except Exception as e:
        print(f"\nComparison failed: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()
