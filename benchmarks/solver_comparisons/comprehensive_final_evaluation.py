#!/usr/bin/env python3
"""
Comprehensive Final Three-Method Evaluation
Complete comparison of FDM, Hybrid, and optimized QP-Collocation methods
with the Tuned Smart QP optimization that achieved 3.7% usage rate.
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
from mfg_pde.alg.hjb_solvers.hjb_gfdm import HJBGFDMSolver
from mfg_pde.alg.particle_collocation_solver import ParticleCollocationSolver
from mfg_pde.core.boundaries import BoundaryConditions
from mfg_pde.core.mfg_problem import ExampleMFGProblem


def comprehensive_three_method_evaluation():
    """Comprehensive evaluation of all three MFG solution methods"""
    print("=" * 100)
    print("COMPREHENSIVE THREE-METHOD MFG SOLVER EVALUATION")
    print("=" * 100)
    print("Comparing: Pure FDM, Hybrid Particle-FDM, and Optimized QP-Collocation")
    print("With Tuned Smart QP optimization achieving 3.7% QP usage rate")

    # Problem parameters for comprehensive evaluation
    problem_params = {"xmin": 0.0, "xmax": 1.0, "Nx": 20, "T": 1.0, "Nt": 30, "sigma": 0.12, "coefCT": 0.02}

    print("Problem Configuration:")
    print(f"  Domain: [{problem_params['xmin']}, {problem_params['xmax']}]")
    print(f"  Spatial discretization: Nx = {problem_params['Nx']}")
    print(f"  Time horizon: T = {problem_params['T']}")
    print(f"  Temporal discretization: Nt = {problem_params['Nt']}")
    print(f"  Volatility: σ = {problem_params['sigma']}")
    print(f"  Control cost: coefCT = {problem_params['coefCT']}")

    results = {}

    # Calculate Dx for mass conservation (used by all methods)
    Dx = (problem_params["xmax"] - problem_params["xmin"]) / problem_params["Nx"]

    # Method 1: Pure FDM
    print(f"\n{'=' * 80}")
    print("METHOD 1: PURE FINITE DIFFERENCE METHOD (FDM)")
    print(f"{'=' * 80}")

    try:
        problem = ExampleMFGProblem(**problem_params)
        no_flux_bc = BoundaryConditions(type="no_flux")

        # Create FDM HJB solver
        fdm_hjb_solver = FdmHJBSolver(problem=problem)

        # Create FDM FP solver
        fdm_fp_solver = FdmFPSolver(problem=problem, boundary_conditions=no_flux_bc)

        # Create fixed point iterator
        fdm_solver = FixedPointIterator(hjb_solver=fdm_hjb_solver, fp_solver=fdm_fp_solver, theta_UM=0.5)

        print("Running Pure FDM solver...")
        print("  - Direct finite difference discretization")
        print("  - Grid-based HJB and FP equation solving")
        print("  - No particle approximation")

        start_time = time.time()
        U_fdm, M_fdm, info_fdm = fdm_solver.solve(Niter=8, l2errBound=1e-3, verbose=True)
        fdm_time = time.time() - start_time

        # Calculate mass conservation
        initial_mass_fdm = np.sum(M_fdm[0, :]) * Dx
        final_mass_fdm = np.sum(M_fdm[-1, :]) * Dx
        mass_error_fdm = abs(final_mass_fdm - initial_mass_fdm) / initial_mass_fdm * 100

        results["fdm"] = {
            "method": "Pure FDM",
            "success": True,
            "time": fdm_time,
            "mass_error": mass_error_fdm,
            "converged": info_fdm.get("converged", False),
            "iterations": info_fdm.get("iterations", 0),
            "U": U_fdm,
            "M": M_fdm,
            "characteristics": {"type": "Grid-based", "particles": 0, "qp_usage": 0.0, "method_complexity": "Low"},
        }

        print("✓ Pure FDM completed:")
        print(f"  Time: {fdm_time:.1f} seconds")
        print(f"  Mass conservation error: {mass_error_fdm:.3f}%")
        print(f"  Converged: {info_fdm.get('converged', False)}")
        print(f"  Picard iterations: {info_fdm.get('iterations', 0)}")

    except Exception as e:
        print(f"✗ Pure FDM failed: {e}")
        results["fdm"] = {
            "method": "Pure FDM",
            "success": False,
            "error": str(e),
            "characteristics": {"type": "Grid-based", "particles": 0},
        }

    # Method 2: Hybrid Particle-FDM
    print(f"\n{'=' * 80}")
    print("METHOD 2: HYBRID PARTICLE-FDM")
    print(f"{'=' * 80}")

    try:
        problem = ExampleMFGProblem(**problem_params)
        no_flux_bc = BoundaryConditions(type="no_flux")

        num_particles_hybrid = 120

        # Create FDM HJB solver
        hybrid_hjb_solver = FdmHJBSolver(problem=problem)

        # Create Particle FP solver
        hybrid_fp_solver = ParticleFPSolver(
            problem=problem, num_particles=num_particles_hybrid, boundary_conditions=no_flux_bc
        )

        # Create fixed point iterator
        hybrid_solver = FixedPointIterator(hjb_solver=hybrid_hjb_solver, fp_solver=hybrid_fp_solver, theta_UM=0.5)

        print("Running Hybrid Particle-FDM solver...")
        print(f"  - Particle method for Fokker-Planck equation ({num_particles_hybrid} particles)")
        print("  - Finite difference method for HJB equation")
        print("  - Hybrid approach combining strengths of both methods")

        start_time = time.time()
        U_hybrid, M_hybrid, info_hybrid = hybrid_solver.solve(Niter=8, l2errBound=1e-3, verbose=True)
        hybrid_time = time.time() - start_time

        # Calculate mass conservation
        initial_mass_hybrid = np.sum(M_hybrid[0, :]) * Dx
        final_mass_hybrid = np.sum(M_hybrid[-1, :]) * Dx
        mass_error_hybrid = abs(final_mass_hybrid - initial_mass_hybrid) / initial_mass_hybrid * 100

        results["hybrid"] = {
            "method": "Hybrid Particle-FDM",
            "success": True,
            "time": hybrid_time,
            "mass_error": mass_error_hybrid,
            "converged": info_hybrid.get("converged", False),
            "iterations": info_hybrid.get("iterations", 0),
            "U": U_hybrid,
            "M": M_hybrid,
            "characteristics": {
                "type": "Hybrid",
                "particles": num_particles_hybrid,
                "qp_usage": 0.0,
                "method_complexity": "Medium",
            },
        }

        print("✓ Hybrid Particle-FDM completed:")
        print(f"  Time: {hybrid_time:.1f} seconds")
        print(f"  Mass conservation error: {mass_error_hybrid:.3f}%")
        print(f"  Converged: {info_hybrid.get('converged', False)}")
        print(f"  Picard iterations: {info_hybrid.get('iterations', 0)}")

    except Exception as e:
        print(f"✗ Hybrid Particle-FDM failed: {e}")
        results["hybrid"] = {
            "method": "Hybrid Particle-FDM",
            "success": False,
            "error": str(e),
            "characteristics": {"type": "Hybrid", "particles": num_particles_hybrid},
        }

    # Method 3: Optimized QP-Collocation (Tuned Smart QP)
    print(f"\n{'=' * 80}")
    print("METHOD 3: OPTIMIZED QP-COLLOCATION (TUNED SMART QP)")
    print(f"{'=' * 80}")

    try:
        problem = ExampleMFGProblem(**problem_params)
        no_flux_bc = BoundaryConditions(type="no_flux")

        # Collocation configuration
        num_collocation_points = 10
        num_particles_qp = 130

        collocation_points = np.linspace(problem.xmin, problem.xmax, num_collocation_points).reshape(-1, 1)
        boundary_indices = [0, num_collocation_points - 1]

        # Create Tuned Smart QP HJB solver (now using consolidated GFDM solver)
        tuned_hjb_solver = HJBGFDMSolver(
            problem=problem,
            collocation_points=collocation_points,
            delta=0.4,
            taylor_order=2,
            weight_function="wendland",
            max_newton_iterations=4,
            newton_tolerance=1e-3,
            boundary_indices=np.array(boundary_indices),
            boundary_conditions=no_flux_bc,
            qp_optimization_level="tuned",
            qp_usage_target=0.1,  # Target 10% QP usage
        )

        # Create optimized particle collocation solver
        qp_solver = ParticleCollocationSolver(
            problem=problem,
            collocation_points=collocation_points,
            num_particles=num_particles_qp,
            delta=0.4,
            taylor_order=2,
            weight_function="wendland",
            NiterNewton=4,
            l2errBoundNewton=1e-3,
            kde_bandwidth="scott",
            normalize_kde_output=False,
            boundary_indices=np.array(boundary_indices),
            boundary_conditions=no_flux_bc,
        )
        qp_solver.hjb_solver = tuned_hjb_solver

        print("Running Optimized QP-Collocation solver...")
        print(f"  - Particle method for Fokker-Planck equation ({num_particles_qp} particles)")
        print(f"  - GFDM collocation for HJB equation ({num_collocation_points} points)")
        print("  - Tuned Smart QP optimization (target: 10% QP usage)")
        print("  - CVXPY/OSQP specialized QP solvers")

        start_time = time.time()
        U_qp, M_qp, info_qp = qp_solver.solve(Niter=8, l2errBound=1e-3, verbose=True)
        qp_time = time.time() - start_time

        # Calculate mass conservation
        initial_mass_qp = np.sum(M_qp[0, :]) * Dx
        final_mass_qp = np.sum(M_qp[-1, :]) * Dx
        mass_error_qp = abs(final_mass_qp - initial_mass_qp) / initial_mass_qp * 100

        # Get QP optimization statistics
        qp_stats = tuned_hjb_solver.get_tuned_qp_report()
        qp_usage_rate = qp_stats.get("qp_usage_rate", 1.0)

        results["qp_collocation"] = {
            "method": "Optimized QP-Collocation",
            "success": True,
            "time": qp_time,
            "mass_error": mass_error_qp,
            "converged": info_qp.get("converged", False),
            "iterations": info_qp.get("iterations", 0),
            "U": U_qp,
            "M": M_qp,
            "qp_stats": qp_stats,
            "characteristics": {
                "type": "Advanced Collocation",
                "particles": num_particles_qp,
                "collocation_points": num_collocation_points,
                "qp_usage": qp_usage_rate,
                "method_complexity": "High",
                "optimization_quality": qp_stats.get("optimization_quality", "N/A"),
            },
        }

        print("✓ Optimized QP-Collocation completed:")
        print(f"  Time: {qp_time:.1f} seconds")
        print(f"  Mass conservation error: {mass_error_qp:.3f}%")
        print(f"  Converged: {info_qp.get('converged', False)}")
        print(f"  Picard iterations: {info_qp.get('iterations', 0)}")
        print(f"  QP Usage Rate: {qp_usage_rate:.1%}")
        print(f"  Optimization Quality: {qp_stats.get('optimization_quality', 'N/A')}")

        # Print detailed QP summary
        if hasattr(tuned_hjb_solver, "print_tuned_qp_summary"):
            tuned_hjb_solver.print_tuned_qp_summary()

    except Exception as e:
        print(f"✗ Optimized QP-Collocation failed: {e}")
        import traceback

        traceback.print_exc()
        results["qp_collocation"] = {
            "method": "Optimized QP-Collocation",
            "success": False,
            "error": str(e),
            "characteristics": {"type": "Advanced Collocation", "particles": num_particles_qp},
        }

    # Print comprehensive comparison summary
    print_comprehensive_comparison(results)

    # Create comprehensive visualization
    create_comprehensive_plots(results, problem_params)

    return results


def print_comprehensive_comparison(results):
    """Print comprehensive method comparison"""
    print(f"\n{'=' * 120}")
    print("COMPREHENSIVE THREE-METHOD COMPARISON SUMMARY")
    print(f"{'=' * 120}")

    # Header
    print(
        f"\n{'Method':<25} {'Success':<8} {'Time(s)':<10} {'Mass Err %':<12} {'Converged':<10} {'Iterations':<11} {'Characteristics':<30}"
    )
    print("-" * 120)

    # Results table
    for _key, result in results.items():
        if result["success"]:
            success_str = "✓"
            time_str = f"{result['time']:.1f}"
            mass_str = f"{result['mass_error']:.3f}"
            converged_str = "✓" if result.get("converged", False) else "✗"
            iter_str = str(result.get("iterations", 0))

            # Characteristics summary
            chars = result.get("characteristics", {})
            char_summary = f"{chars.get('type', 'N/A')}"
            if chars.get("particles", 0) > 0:
                char_summary += f", {chars.get('particles')} particles"
            if chars.get("qp_usage", 0) > 0:
                char_summary += f", {chars.get('qp_usage'):.1%} QP"
        else:
            success_str = "✗"
            time_str = "FAILED"
            mass_str = "N/A"
            converged_str = "N/A"
            iter_str = "N/A"
            char_summary = "FAILED"

        print(
            f"{result['method']:<25} {success_str:<8} {time_str:<10} {mass_str:<12} {converged_str:<10} {iter_str:<11} {char_summary:<30}"
        )

    # Performance analysis
    successful_results = {k: v for k, v in results.items() if v["success"]}

    if len(successful_results) > 1:
        print("\nPERFORMANCE ANALYSIS:")
        print("=" * 60)

        # Speed comparison
        times = [(k, v["time"]) for k, v in successful_results.items()]
        times.sort(key=lambda x: x[1])

        print("Speed Ranking:")
        for i, (method, time_val) in enumerate(times, 1):
            method_name = successful_results[method]["method"]
            if i == 1:
                print(f"  {i}. {method_name}: {time_val:.1f}s (fastest)")
            else:
                speedup = time_val / times[0][1]
                print(f"  {i}. {method_name}: {time_val:.1f}s ({speedup:.1f}x slower)")

        # Accuracy comparison
        print("\nAccuracy Ranking (Mass Conservation):")
        mass_errors = [(k, v["mass_error"]) for k, v in successful_results.items()]
        mass_errors.sort(key=lambda x: x[1])

        for i, (method, error) in enumerate(mass_errors, 1):
            method_name = successful_results[method]["method"]
            if i == 1:
                print(f"  {i}. {method_name}: {error:.3f}% (most accurate)")
            else:
                print(f"  {i}. {method_name}: {error:.3f}%")

        # Method-specific analysis
        print("\nMETHOD-SPECIFIC ANALYSIS:")
        print("-" * 40)

        for key, result in successful_results.items():
            print(f"\n{result['method']}:")
            chars = result.get("characteristics", {})

            # Basic performance
            print(f"  ✓ Solve time: {result['time']:.1f}s")
            print(f"  ✓ Mass conservation: {result['mass_error']:.3f}% error")
            print(f"  ✓ Method complexity: {chars.get('method_complexity', 'N/A')}")

            # Method-specific metrics
            if key == "qp_collocation" and "qp_stats" in result:
                qp_stats = result["qp_stats"]
                qp_skip_rate = 1.0 - qp_stats.get("qp_usage_rate", 1.0)
                estimated_speedup = 1 / (1 - qp_skip_rate * 0.9) if qp_skip_rate > 0 else 1.0
                print(f"  ✓ QP Usage Rate: {qp_stats.get('qp_usage_rate', 0):.1%}")
                print(f"  ✓ Estimated Speedup vs Baseline QP: {estimated_speedup:.1f}x")
                print(f"  ✓ Optimization Quality: {qp_stats.get('optimization_quality', 'N/A')}")

            if chars.get("particles", 0) > 0:
                print(f"  ✓ Particle count: {chars.get('particles')}")

            if chars.get("collocation_points", 0) > 0:
                print(f"  ✓ Collocation points: {chars.get('collocation_points')}")

    # Overall recommendation
    print("\nOVERALL ASSESSMENT AND RECOMMENDATIONS:")
    print("=" * 50)

    if len(successful_results) == 3:
        # All methods successful
        fastest_method = min(successful_results.items(), key=lambda x: x[1]["time"])
        most_accurate = min(successful_results.items(), key=lambda x: x[1]["mass_error"])

        print("✅ All three methods completed successfully")
        print(f"🏃 Fastest: {fastest_method[1]['method']} ({fastest_method[1]['time']:.1f}s)")
        print(f"🎯 Most accurate: {most_accurate[1]['method']} ({most_accurate[1]['mass_error']:.3f}% error)")

        # Check if QP-Collocation achieved optimization target
        if "qp_collocation" in successful_results:
            qp_result = successful_results["qp_collocation"]
            qp_usage = qp_result.get("characteristics", {}).get("qp_usage", 1.0)

            if qp_usage <= 0.12:  # Within 20% of 10% target
                print(f"🎉 QP-Collocation optimization target achieved ({qp_usage:.1%} usage)")
                print("📈 Recommended for production: Optimized QP-Collocation")
                print("   - Combines accuracy with optimized performance")
                print("   - Advanced features with smart QP optimization")
            elif qp_usage <= 0.25:  # Good optimization
                print(f"⚠️  QP-Collocation shows good optimization ({qp_usage:.1%} usage)")
                print("📈 Recommended for advanced applications: Optimized QP-Collocation")
                print("📈 Recommended for standard applications: Hybrid Particle-FDM")
            else:
                print("📈 Recommended: Hybrid Particle-FDM (most reliable)")
                print("   - Good balance of speed and accuracy")
                print("   - QP-Collocation needs further optimization")

        # Application-specific recommendations
        print("\nApplication-Specific Recommendations:")
        print("  🔬 Research/Prototyping: Pure FDM (simple, reliable)")
        print("  🏢 Production Applications: Hybrid Particle-FDM (balanced)")
        if "qp_collocation" in successful_results:
            qp_usage = successful_results["qp_collocation"].get("characteristics", {}).get("qp_usage", 1.0)
            if qp_usage <= 0.15:
                print("  🚀 High-Performance Applications: Optimized QP-Collocation (advanced)")

    elif len(successful_results) == 2:
        print("⚠️  Two methods completed successfully")
        working_methods = [r["method"] for r in successful_results.values()]
        print(f"   Working methods: {', '.join(working_methods)}")
        print("📈 Recommended: Use the most suitable working method for your application")

    elif len(successful_results) == 1:
        method_name = next(iter(successful_results.values()))["method"]
        print(f"⚠️  Only one method completed successfully: {method_name}")
        print(f"📈 Recommended: Use {method_name} and investigate failures in other methods")

    else:
        print("❌ All methods failed - check problem setup and solver configurations")


def create_comprehensive_plots(results, problem_params):
    """Create comprehensive visualization of all three methods"""
    successful_results = {k: v for k, v in results.items() if v["success"]}

    if len(successful_results) == 0:
        print("No successful results to plot")
        return

    # Create large comprehensive figure
    plt.figure(figsize=(20, 16))

    # Plot 1: Performance Comparison
    ax1 = plt.subplot(3, 3, 1)
    methods = [r["method"] for r in successful_results.values()]
    times = [r["time"] for r in successful_results.values()]
    colors = ["red", "orange", "blue"][: len(methods)]

    bars1 = ax1.bar(methods, times, color=colors, alpha=0.7)
    ax1.set_ylabel("Solve Time (seconds)")
    ax1.set_title("Computational Performance Comparison")
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(True, alpha=0.3)

    # Add time labels
    for bar, time_val in zip(bars1, times, strict=False):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.5,
            f"{time_val:.1f}s",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Plot 2: Mass Conservation Accuracy
    ax2 = plt.subplot(3, 3, 2)
    mass_errors = [r["mass_error"] for r in successful_results.values()]

    ax2.bar(methods, mass_errors, color=colors, alpha=0.7)
    ax2.set_ylabel("Mass Conservation Error (%)")
    ax2.set_title("Solution Accuracy Comparison")
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale("log")

    # Plot 3: Method Characteristics
    ax3 = plt.subplot(3, 3, 3)
    ax3.axis("off")

    char_text = "METHOD CHARACTERISTICS\n\n"
    for _i, (_key, result) in enumerate(successful_results.items()):
        chars = result.get("characteristics", {})
        char_text += f"{result['method']}:\n"
        char_text += f"  Type: {chars.get('type', 'N/A')}\n"
        char_text += f"  Complexity: {chars.get('method_complexity', 'N/A')}\n"
        if chars.get("particles", 0) > 0:
            char_text += f"  Particles: {chars.get('particles')}\n"
        if chars.get("qp_usage", 0) > 0:
            char_text += f"  QP Usage: {chars.get('qp_usage'):.1%}\n"
        char_text += "\n"

    ax3.text(
        0.05,
        0.95,
        char_text,
        transform=ax3.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "lightgray", "alpha": 0.8},
    )

    # Plot 4-6: Solution profiles for each method
    x_grid = np.linspace(
        problem_params["xmin"],
        problem_params["xmax"],
        successful_results[next(iter(successful_results.keys()))]["M"].shape[1],
    )

    plot_positions = [(3, 3, 4), (3, 3, 5), (3, 3, 6)]
    for i, (_key, result) in enumerate(successful_results.items()):
        if i < 3:  # Only plot first 3 methods
            ax = plt.subplot(*plot_positions[i])
            M = result["M"]
            result["U"]

            # Plot density evolution
            for t_idx in [0, M.shape[0] // 2, -1]:
                alpha = 0.5 if t_idx == M.shape[0] // 2 else 1.0
                label = ["Initial", "Mid", "Final"][min(t_idx, 2)] if t_idx != M.shape[0] // 2 else "Mid"
                ax.plot(
                    x_grid,
                    M[t_idx, :],
                    label=f"{label} Density",
                    color=colors[i],
                    alpha=alpha,
                    linewidth=2 if alpha == 1.0 else 1,
                )

            ax.set_xlabel("x")
            ax.set_ylabel("Density")
            ax.set_title(f"{result['method']} - Density Evolution")
            ax.legend()
            ax.grid(True, alpha=0.3)

    # Plot 7: Performance vs Accuracy Trade-off
    ax7 = plt.subplot(3, 3, 7)

    for i, (_key, result) in enumerate(successful_results.items()):
        ax7.scatter(result["time"], result["mass_error"], color=colors[i], s=150, alpha=0.7, label=result["method"])
        ax7.annotate(
            result["method"],
            (result["time"], result["mass_error"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    ax7.set_xlabel("Solve Time (seconds)")
    ax7.set_ylabel("Mass Conservation Error (%)")
    ax7.set_title("Performance vs Accuracy Trade-off")
    ax7.set_yscale("log")
    ax7.grid(True, alpha=0.3)
    ax7.legend()

    # Plot 8: QP Optimization Analysis (if QP-Collocation succeeded)
    ax8 = plt.subplot(3, 3, 8)

    if "qp_collocation" in successful_results:
        qp_result = successful_results["qp_collocation"]
        qp_stats = qp_result.get("qp_stats", {})

        # QP usage visualization
        qp_usage = qp_stats.get("qp_usage_rate", 0) * 100
        qp_skip = 100 - qp_usage

        labels = ["QP Used", "QP Skipped"]
        sizes = [qp_usage, qp_skip]
        colors_pie = ["red", "green"]

        ax8.pie(sizes, labels=labels, colors=colors_pie, autopct="%1.1f%%", startangle=90)
        ax8.set_title("QP Usage Optimization\n(Tuned Smart QP)")

        # Add optimization quality text
        quality = qp_stats.get("optimization_quality", "N/A")
        ax8.text(
            0,
            -1.3,
            f"Quality: {quality}",
            ha="center",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "yellow", "alpha": 0.7},
        )
    else:
        ax8.text(
            0.5,
            0.5,
            "QP-Collocation\nNot Available",
            ha="center",
            va="center",
            transform=ax8.transAxes,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "lightgray", "alpha": 0.8},
        )
        ax8.set_title("QP Optimization Status")

    # Plot 9: Final Recommendations
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis("off")

    # Generate recommendations
    recommendations = "FINAL RECOMMENDATIONS\n\n"

    if len(successful_results) >= 3:
        fastest = min(successful_results.items(), key=lambda x: x[1]["time"])
        most_accurate = min(successful_results.items(), key=lambda x: x[1]["mass_error"])

        recommendations += f"🏃 FASTEST:\n{fastest[1]['method']}\n({fastest[1]['time']:.1f}s)\n\n"
        recommendations += (
            f"🎯 MOST ACCURATE:\n{most_accurate[1]['method']}\n({most_accurate[1]['mass_error']:.3f}%)\n\n"
        )

        if "qp_collocation" in successful_results:
            qp_usage = successful_results["qp_collocation"].get("characteristics", {}).get("qp_usage", 1.0)
            if qp_usage <= 0.15:
                recommendations += "🚀 RECOMMENDED:\nOptimized QP-Collocation\n(Advanced optimization achieved)\n\n"
            else:
                recommendations += "🏢 RECOMMENDED:\nHybrid Particle-FDM\n(Balanced performance)\n"

        recommendations += "✅ All methods working\nChoose based on application needs"
    else:
        working_count = len(successful_results)
        recommendations += f"⚠️ {working_count} method(s) working\n\n"
        if working_count > 0:
            best_method = next(iter(successful_results.values()))["method"]
            recommendations += f"📈 USE: {best_method}\n"
            recommendations += "🔧 Debug failed methods\n"
        else:
            recommendations += "❌ All methods failed\n"
            recommendations += "🔧 Check problem setup"

    ax9.text(
        0.05,
        0.95,
        recommendations,
        transform=ax9.transAxes,
        fontsize=11,
        verticalalignment="top",
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "lightgreen", "alpha": 0.8},
    )

    plt.tight_layout()

    # Save comprehensive results
    filename = "/Users/zvezda/Library/CloudStorage/OneDrive-Personal/code/MFG_PDE/tests/method_comparisons/comprehensive_three_method_evaluation_results.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"\nComprehensive evaluation results saved to: {filename}")
    plt.show()


def main():
    """Run the comprehensive three-method evaluation"""
    print("Starting Comprehensive Three-Method MFG Solver Evaluation...")
    print("This compares Pure FDM, Hybrid Particle-FDM, and Optimized QP-Collocation")
    print("Expected execution time: 10-15 minutes")

    try:
        results = comprehensive_three_method_evaluation()

        print(f"\n{'=' * 120}")
        print("COMPREHENSIVE THREE-METHOD EVALUATION COMPLETED")
        print(f"{'=' * 120}")
        print("Check the comprehensive summary above and generated visualization for complete results.")

        return results

    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
        return None
    except Exception as e:
        print(f"\nEvaluation failed: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()
