"""
Solver comparison utilities for validation benchmarks.
"""

from __future__ import annotations

from typing import Any

from .metrics import (
    compute_l2_error,
    compute_mass_conservation,
    compute_relative_l2_error,
    compute_solution_statistics,
)


def compare_solvers(
    solver1_result: dict[str, Any],
    solver2_result: dict[str, Any],
    dx: float | tuple[float, ...],
    solver1_name: str = "Solver 1",
    solver2_name: str = "Solver 2",
) -> dict[str, Any]:
    """
    Compare two solver results.

    Parameters
    ----------
    solver1_result : dict
        First solver result with keys: 'U', 'M', 'info'
    solver2_result : dict
        Second solver result
    dx : float or tuple
        Grid spacing
    solver1_name : str
        Name of first solver (for reporting)
    solver2_name : str
        Name of second solver

    Returns
    -------
    dict
        Comparison metrics
    """
    U1, M1 = solver1_result["U"], solver1_result["M"]
    U2, M2 = solver2_result["U"], solver2_result["M"]

    # Compute errors
    u_l2_error = compute_l2_error(U1, U2, dx=dx)
    u_rel_error = compute_relative_l2_error(U1, U2, dx=dx)
    m_l2_error = compute_l2_error(M1, M2, dx=dx)
    m_rel_error = compute_relative_l2_error(M1, M2, dx=dx)

    # Mass conservation
    mass1 = compute_mass_conservation(M1[-1], dx=dx, target_mass=1.0)
    mass2 = compute_mass_conservation(M2[-1], dx=dx, target_mass=1.0)

    # Solution statistics
    u1_stats = compute_solution_statistics(U1)
    u2_stats = compute_solution_statistics(U2)
    m1_stats = compute_solution_statistics(M1)
    m2_stats = compute_solution_statistics(M2)

    # Runtime
    runtime1 = solver1_result.get("info", {}).get("runtime", None)
    runtime2 = solver2_result.get("info", {}).get("runtime", None)

    # Convergence
    converged1 = solver1_result.get("info", {}).get("converged", False)
    converged2 = solver2_result.get("info", {}).get("converged", False)
    iterations1 = solver1_result.get("info", {}).get("num_iterations", 0)
    iterations2 = solver2_result.get("info", {}).get("num_iterations", 0)

    return {
        "solvers": {
            "solver1": solver1_name,
            "solver2": solver2_name,
        },
        "errors": {
            "value_function_l2": u_l2_error,
            "value_function_rel_l2": u_rel_error,
            "density_l2": m_l2_error,
            "density_rel_l2": m_rel_error,
        },
        "mass_conservation": {
            "solver1": mass1,
            "solver2": mass2,
        },
        "solution_stats": {
            "value_function": {"solver1": u1_stats, "solver2": u2_stats},
            "density": {"solver1": m1_stats, "solver2": m2_stats},
        },
        "convergence": {
            "solver1": {
                "converged": converged1,
                "iterations": iterations1,
            },
            "solver2": {
                "converged": converged2,
                "iterations": iterations2,
            },
        },
        "runtime": {
            "solver1": runtime1,
            "solver2": runtime2,
            "speedup": runtime1 / runtime2 if (runtime1 and runtime2) else None,
        },
    }


def create_comparison_report(comparison: dict[str, Any]) -> str:
    """
    Create human-readable comparison report.

    Parameters
    ----------
    comparison : dict
        Output from compare_solvers()

    Returns
    -------
    str
        Formatted report
    """
    s1 = comparison["solvers"]["solver1"]
    s2 = comparison["solvers"]["solver2"]

    report = []
    report.append("=" * 70)
    report.append(f"Solver Comparison: {s1} vs {s2}")
    report.append("=" * 70)

    # Errors
    report.append("\n### Solution Errors")
    errors = comparison["errors"]
    report.append(f"Value Function L² Error:     {errors['value_function_l2']:.6e}")
    report.append(f"Value Function Rel L² Error: {errors['value_function_rel_l2']:.4f}")
    report.append(f"Density L² Error:            {errors['density_l2']:.6e}")
    report.append(f"Density Rel L² Error:        {errors['density_rel_l2']:.4f}")

    # Mass conservation
    report.append("\n### Mass Conservation")
    mass = comparison["mass_conservation"]
    report.append(f"{s1}: {mass['solver1'] * 100:.2f}% error")
    report.append(f"{s2}: {mass['solver2'] * 100:.2f}% error")

    # Convergence
    report.append("\n### Convergence")
    conv = comparison["convergence"]
    report.append(f"{s1}: {'✓' if conv['solver1']['converged'] else '✗'} ({conv['solver1']['iterations']} iterations)")
    report.append(f"{s2}: {'✓' if conv['solver2']['converged'] else '✗'} ({conv['solver2']['iterations']} iterations)")

    # Runtime
    report.append("\n### Runtime")
    runtime = comparison["runtime"]
    if runtime["solver1"] and runtime["solver2"]:
        report.append(f"{s1}: {runtime['solver1']:.2f}s")
        report.append(f"{s2}: {runtime['solver2']:.2f}s")
        if runtime["speedup"]:
            report.append(f"Speedup: {runtime['speedup']:.2f}×")
    else:
        report.append("Runtime data not available")

    report.append("=" * 70)

    return "\n".join(report)
