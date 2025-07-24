#!/usr/bin/env python3
"""
Extensive Statistical Analysis
Comprehensive statistical evaluation of all three methods with extensive test cases
to demonstrate robustness, stability, computational cost, and mass conservation.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
import scipy.stats as stats
from scipy import ndimage

warnings.filterwarnings("ignore")

from mfg_pde.core.mfg_problem import ExampleMFGProblem
from mfg_pde.alg.hjb_solvers.fdm_hjb import FdmHJBSolver
from mfg_pde.alg.fp_solvers.fdm_fp import FdmFPSolver
from mfg_pde.alg.fp_solvers.particle_fp import ParticleFPSolver
from mfg_pde.alg.damped_fixed_point_iterator import FixedPointIterator
from mfg_pde.alg.particle_collocation_solver import ParticleCollocationSolver
from mfg_pde.alg.hjb_solvers.tuned_smart_qp_gfdm_hjb import TunedSmartQPGFDMHJBSolver
from mfg_pde.core.boundaries import BoundaryConditions


def generate_extensive_test_cases():
    """Generate extensive test cases for comprehensive statistical analysis"""
    test_cases = []

    # 1. Scale variation tests (15 cases)
    print("Generating scale variation tests...")
    for Nx in [12, 15, 18, 20, 22]:
        for T in [0.4, 0.8, 1.2]:
            test_cases.append(
                {
                    "name": f"Scale_Nx{Nx}_T{T}",
                    "params": {
                        "xmin": 0.0,
                        "xmax": 1.0,
                        "Nx": Nx,
                        "T": T,
                        "Nt": int(T * 25),
                        "sigma": 0.12,
                        "coefCT": 0.02,
                    },
                    "category": "Scale Variation",
                    "difficulty": "Easy" if Nx <= 15 and T <= 0.8 else "Moderate",
                }
            )

    # 2. Volatility sweep tests (12 cases)
    print("Generating volatility sweep tests...")
    for sigma in [0.05, 0.08, 0.12, 0.16, 0.20, 0.25]:
        for coefCT in [0.01, 0.03]:
            test_cases.append(
                {
                    "name": f"Vol_s{sigma:.2f}_c{coefCT:.2f}",
                    "params": {
                        "xmin": 0.0,
                        "xmax": 1.0,
                        "Nx": 18,
                        "T": 0.8,
                        "Nt": 20,
                        "sigma": sigma,
                        "coefCT": coefCT,
                    },
                    "category": "Volatility Sweep",
                    "difficulty": "Easy" if sigma <= 0.12 else "Challenge",
                }
            )

    # 3. Time horizon tests (9 cases)
    print("Generating time horizon tests...")
    for T in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        for Nt_factor in [15, 25]:
            if len([case for case in test_cases if case["params"]["T"] == T]) < 3:
                test_cases.append(
                    {
                        "name": f"Time_T{T}_Nt{int(T*Nt_factor)}",
                        "params": {
                            "xmin": 0.0,
                            "xmax": 1.0,
                            "Nx": 16,
                            "T": T,
                            "Nt": int(T * Nt_factor),
                            "sigma": 0.10,
                            "coefCT": 0.02,
                        },
                        "category": "Time Horizon",
                        "difficulty": "Easy" if T <= 1.0 else "Challenge",
                    }
                )
            if (
                len([case for case in test_cases if case["category"] == "Time Horizon"])
                >= 9
            ):
                break

    # 4. Control cost variation (8 cases)
    print("Generating control cost tests...")
    for coefCT in [0.005, 0.01, 0.02, 0.03, 0.04, 0.05]:
        if len([case for case in test_cases if case["category"] == "Control Cost"]) < 8:
            test_cases.append(
                {
                    "name": f"Control_c{coefCT:.3f}",
                    "params": {
                        "xmin": 0.0,
                        "xmax": 1.0,
                        "Nx": 16,
                        "T": 1.0,
                        "Nt": 25,
                        "sigma": 0.12,
                        "coefCT": coefCT,
                    },
                    "category": "Control Cost",
                    "difficulty": "Easy" if coefCT >= 0.02 else "Moderate",
                }
            )

    # 5. Mixed challenging tests (6 cases)
    print("Generating challenging combination tests...")
    challenging_params = [
        {"Nx": 24, "T": 1.5, "sigma": 0.18, "coefCT": 0.01, "Nt": 30},
        {"Nx": 20, "T": 2.0, "sigma": 0.22, "coefCT": 0.015, "Nt": 40},
        {"Nx": 26, "T": 1.0, "sigma": 0.25, "coefCT": 0.025, "Nt": 25},
        {"Nx": 18, "T": 2.5, "sigma": 0.15, "coefCT": 0.01, "Nt": 50},
        {"Nx": 22, "T": 1.8, "sigma": 0.20, "coefCT": 0.02, "Nt": 36},
        {"Nx": 16, "T": 3.0, "sigma": 0.12, "coefCT": 0.005, "Nt": 60},
    ]

    for i, params in enumerate(challenging_params):
        test_cases.append(
            {
                "name": f"Challenge_{i+1}",
                "params": {
                    "xmin": 0.0,
                    "xmax": 1.0,
                    "Nx": params["Nx"],
                    "T": params["T"],
                    "Nt": params["Nt"],
                    "sigma": params["sigma"],
                    "coefCT": params["coefCT"],
                },
                "category": "Challenging",
                "difficulty": "Extreme",
            }
        )

    print(f"Generated {len(test_cases)} test cases")
    return test_cases


def test_method_with_stats(method_name, solver_func, params, test_name, timeout=600):
    """Test a method and return comprehensive statistics"""
    try:
        start_time = time.time()
        result = solver_func(params, test_name, timeout)
        total_time = time.time() - start_time

        if not result["success"]:
            return {
                "success": False,
                "method": method_name,
                "test_name": test_name,
                "error": result.get("error", "Unknown"),
                "time": total_time,
                "converged": False,
            }

        # Calculate additional statistics
        U, M = result.get("U"), result.get("M")

        # Mass conservation analysis
        Dx = (params["xmax"] - params["xmin"]) / params["Nx"]
        initial_mass = np.sum(M[0, :]) * Dx
        final_mass = np.sum(M[-1, :]) * Dx
        mass_error = (
            abs(final_mass - initial_mass) / initial_mass * 100
            if initial_mass > 1e-9
            else 0
        )

        # Solution stability metrics
        if M.shape[0] > 2:
            temporal_changes = np.linalg.norm(
                M[1:, :] - M[:-1, :], axis=1
            ) / np.linalg.norm(M[:-1, :], axis=1)
            temporal_stability = np.mean(temporal_changes)
        else:
            temporal_stability = 0.0

        return {
            "success": True,
            "method": method_name,
            "test_name": test_name,
            "time": result["time"],
            "total_time": total_time,
            "mass_error": mass_error,
            "converged": result.get("converged", False),
            "iterations": result.get("iterations", 0),
            "temporal_stability": temporal_stability,
        }

    except Exception as e:
        return {
            "success": False,
            "method": method_name,
            "test_name": test_name,
            "error": str(e),
            "time": 0,
            "converged": False,
        }


def test_pure_fdm(problem_params, test_name, timeout=300):
    """Test Pure FDM method"""
    try:
        problem = ExampleMFGProblem(**problem_params)
        no_flux_bc = BoundaryConditions(type="no_flux")

        fdm_hjb_solver = FdmHJBSolver(problem=problem)
        fdm_fp_solver = FdmFPSolver(problem=problem, boundary_conditions=no_flux_bc)
        fdm_solver = FixedPointIterator(
            problem=problem,
            hjb_solver=fdm_hjb_solver,
            fp_solver=fdm_fp_solver,
            thetaUM=0.5,
        )

        start_time = time.time()
        U_fdm, M_fdm, iterations_run, l2distu_rel, l2distm_rel = fdm_solver.solve(
            Niter_max=50, l2errBoundPicard=1e-3
        )
        solve_time = time.time() - start_time

        if solve_time > timeout:
            return {"success": False, "error": "Timeout", "time": solve_time}

        converged = (
            len(l2distu_rel) > 0 and l2distu_rel[-1] < 1e-3 and l2distm_rel[-1] < 1e-3
        )

        return {
            "success": True,
            "time": solve_time,
            "converged": converged,
            "iterations": iterations_run,
            "U": U_fdm,
            "M": M_fdm,
        }

    except Exception as e:
        return {"success": False, "error": str(e), "time": 0}


def test_hybrid_particle_fdm(problem_params, test_name, timeout=300):
    """Test Hybrid Particle-FDM method"""
    try:
        problem = ExampleMFGProblem(**problem_params)
        no_flux_bc = BoundaryConditions(type="no_flux")

        num_particles = min(120, max(60, problem_params["Nx"] * 4))

        hybrid_hjb_solver = FdmHJBSolver(problem=problem)
        hybrid_fp_solver = ParticleFPSolver(
            problem=problem, num_particles=num_particles, boundary_conditions=no_flux_bc
        )
        hybrid_solver = FixedPointIterator(
            problem=problem,
            hjb_solver=hybrid_hjb_solver,
            fp_solver=hybrid_fp_solver,
            thetaUM=0.5,
        )

        start_time = time.time()
        U_hybrid, M_hybrid, iterations_run, l2distu_rel, l2distm_rel = (
            hybrid_solver.solve(Niter_max=50, l2errBoundPicard=1e-3)
        )
        solve_time = time.time() - start_time

        if solve_time > timeout:
            return {"success": False, "error": "Timeout", "time": solve_time}

        converged = (
            len(l2distu_rel) > 0 and l2distu_rel[-1] < 1e-3 and l2distm_rel[-1] < 1e-3
        )

        return {
            "success": True,
            "time": solve_time,
            "converged": converged,
            "iterations": iterations_run,
            "U": U_hybrid,
            "M": M_hybrid,
        }

    except Exception as e:
        return {"success": False, "error": str(e), "time": 0}


def test_optimized_qp_collocation(problem_params, test_name, timeout=600):
    """Test Optimized QP-Collocation method"""
    try:
        problem = ExampleMFGProblem(**problem_params)
        no_flux_bc = BoundaryConditions(type="no_flux")

        num_collocation_points = min(12, max(8, problem_params["Nx"] // 2))
        num_particles = min(150, max(100, problem_params["Nx"] * 5))
        collocation_points = np.linspace(
            problem.xmin, problem.xmax, num_collocation_points
        ).reshape(-1, 1)

        qp_solver = ParticleCollocationSolver(
            problem=problem,
            collocation_points=collocation_points,
            num_particles=num_particles,
            boundary_conditions=no_flux_bc,
        )

        start_time = time.time()
        U_qp, M_qp, info_qp = qp_solver.solve(Niter=50, l2errBound=1e-3, verbose=False)
        solve_time = time.time() - start_time

        if solve_time > timeout:
            return {"success": False, "error": "Timeout", "time": solve_time}

        return {
            "success": True,
            "time": solve_time,
            "converged": info_qp.get("converged", False),
            "iterations": info_qp.get("iterations", 0),
            "U": U_qp,
            "M": M_qp,
        }

    except Exception as e:
        return {"success": False, "error": str(e), "time": 0}


def run_extensive_statistical_analysis():
    """Run extensive statistical analysis"""
    print("=" * 100)
    print("EXTENSIVE STATISTICAL ANALYSIS - THREE METHOD COMPARISON")
    print("=" * 100)

    test_cases = generate_extensive_test_cases()
    total_cases = len(test_cases)

    results = {
        "test_cases": test_cases,
        "fdm_results": [],
        "hybrid_results": [],
        "qp_results": [],
    }

    methods = [
        ("Pure FDM", test_pure_fdm),
        ("Hybrid", test_hybrid_particle_fdm),
        ("QP-Collocation", test_optimized_qp_collocation),
    ]

    total_tests = total_cases * len(methods)
    current_test = 0

    for i, test_case in enumerate(test_cases):
        case_name = test_case["name"]
        params = test_case["params"]
        difficulty = test_case["difficulty"]

        print(
            f"\n--- TEST CASE {i+1}/{total_cases}: {case_name} (Difficulty: {difficulty}) ---"
        )

        for method_name, test_func in methods:
            current_test += 1
            print(
                f"[{current_test:3d}/{total_tests}] Testing {method_name}...",
                end=" ",
                flush=True,
            )

            timeout = 300 if difficulty in ["Easy", "Moderate"] else 900

            result = test_method_with_stats(
                method_name, test_func, params, case_name, timeout
            )
            result["difficulty"] = difficulty

            if method_name == "Pure FDM":
                results["fdm_results"].append(result)
            elif method_name == "Hybrid":
                results["hybrid_results"].append(result)
            else:
                results["qp_results"].append(result)

            if result["success"]:
                print(f"✓ Success ({result['time']:.1f}s)")
            else:
                print(f"✗ Failed ({result.get('error', 'Unknown')[:30]})")

    analyze_and_visualize_results(results)
    return results


def run_extensive_statistical_analysis():
    """Run extensive statistical analysis"""
    test_cases = generate_extensive_test_cases()
    results = {"fdm_results": [], "hybrid_results": [], "qp_results": []}
    methods = [
        ("Pure FDM", test_pure_fdm),
        ("Hybrid", test_hybrid_particle_fdm),
        ("QP-Collocation", test_optimized_qp_collocation),
    ]

    for i, test_case in enumerate(test_cases):
        print(f"\n--- Running Case {i+1}/{len(test_cases)}: {test_case['name']} ---")
        for method_name, test_func in methods:
            result = test_method_with_stats(
                method_name, test_func, test_case["params"], test_case["name"]
            )
            result["difficulty"] = test_case["difficulty"]
            if method_name == "Pure FDM":
                results["fdm_results"].append(result)
            elif method_name == "Hybrid":
                results["hybrid_results"].append(result)
            else:
                results["qp_results"].append(result)

    analyze_and_visualize_results(results)
    return results


def analyze_and_visualize_results(results):
    """Generate comprehensive statistical analysis and visualization"""
    method_names = ["Pure FDM", "Hybrid", "QP-Collocation"]
    method_results_map = {
        "Pure FDM": results["fdm_results"],
        "Hybrid": results["hybrid_results"],
        "QP-Collocation": results["qp_results"],
    }

    overall_stats = {}
    for name in method_names:
        res = method_results_map[name]
        successful = [r for r in res if r["success"]]
        if successful:
            overall_stats[name] = {
                "times": [r["time"] for r in successful],
                "mass_errors": [r["mass_error"] for r in successful],
            }
        else:
            overall_stats[name] = {"times": [], "mass_errors": []}

    create_selected_3_panel_figure(overall_stats, method_results_map)


def create_selected_3_panel_figure(overall_stats, method_results_map):
    """Create a 1x3 figure showing selected analysis plots."""

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle(
        "Selected Performance Analysis of Three Methods", fontsize=18, fontweight="bold"
    )

    methods = ["Pure FDM", "Hybrid", "QP-Collocation"]
    colors = {"Pure FDM": "#348ABD", "Hybrid": "#7A68A6", "QP-Collocation": "#A60628"}

    # --- Plot 1: Mass Conservation Quality ---
    mass_data = [
        stats["mass_errors"] for stats in overall_stats.values() if stats["mass_errors"]
    ]
    if mass_data:
        bp1 = ax1.boxplot(mass_data, patch_artist=True, showfliers=False, widths=0.6)
        for patch, name in zip(bp1["boxes"], methods):
            patch.set_facecolor(colors[name])
            patch.set_alpha(0.8)

    ax1.set_yscale("log")
    ax1.set_title("Mass Conservation Quality", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Mass Conservation Error (%)")
    ax1.set_xticklabels(methods)
    ax1.axhline(5, color="orange", linestyle="--", linewidth=1)
    ax1.axhline(2, color="green", linestyle="--", linewidth=1)
    ax1.text(0.5, 5.2, "Good (<5%)", color="orange", va="bottom", ha="left")
    ax1.text(0.5, 2.1, "Excellent (<2%)", color="green", va="bottom", ha="left")

    # --- Plot 2: Computational Cost ---
    time_data = [stats["times"] for stats in overall_stats.values() if stats["times"]]
    if time_data:
        bp2 = ax2.boxplot(time_data, patch_artist=True, showfliers=False, widths=0.6)
        for patch, name in zip(bp2["boxes"], methods):
            patch.set_facecolor(colors[name])
            patch.set_alpha(0.8)

    ax2.set_yscale("log")
    ax2.set_title("Computational Cost", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Solve Time (seconds)")
    ax2.set_xticklabels(methods)

    # --- Plot 3: Mass Conservation vs Computational Cost ---
    for name in methods:
        res = [r for r in method_results_map[name] if r["success"]]
        if res:
            times = [r["time"] for r in res]
            errors = [r["mass_error"] for r in res]
            ax3.scatter(times, errors, alpha=0.6, label=name, color=colors[name], s=40)

    ax3.set_xscale("log")
    ax3.set_yscale("log")
    ax3.set_title("Mass Conservation vs Cost", fontsize=14, fontweight="bold")
    ax3.set_xlabel("Solve Time (seconds)")
    ax3.set_ylabel("Mass Conservation Error (%)")
    ax3.legend()
    ax3.axhline(5, color="orange", linestyle="--", linewidth=1)
    ax3.axhline(2, color="green", linestyle="--", linewidth=1)

    # --- Final Touches ---
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    filename = "selected_analysis_plots.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"\n📊 Selected analysis plots saved to: {filename}")
    plt.show()


def main():
    """Run extensive statistical analysis"""
    print("Starting Extensive Statistical Analysis...")
    try:
        run_extensive_statistical_analysis()
        print(f"\n{'='*100}")
        print("🎉 ANALYSIS COMPLETED SUCCESSFULLY")
        print(f"{'='*100}")
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"\nAnalysis failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
