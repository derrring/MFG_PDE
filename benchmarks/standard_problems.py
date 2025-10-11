"""
Standard Benchmark Problems for MFG_PDE

Defines canonical test problems with known characteristics for consistent
performance tracking and comparison across solver implementations.

Problem Categories:
- Small: Quick validation (50x50 grids, <5s expected)
- Medium: Moderate complexity (100x100 grids, 10-30s expected)
- Large: Stress testing (2D problems, >60s expected)

Each problem provides:
- Consistent configuration
- Expected convergence behavior
- Reference timing estimates
- Problem-specific validation criteria
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class BenchmarkProblem:
    """
    Standard benchmark problem specification.

    Attributes:
        name: Unique identifier for this problem
        category: Size category ('small', 'medium', 'large')
        description: Human-readable problem description
        solver_type: Solver to use ('hjb_fdm', 'hjb_collocation', 'particle_fp', etc.)
        problem_config: Configuration dictionary for problem setup
        solver_config: Configuration dictionary for solver
        expected_time_range: (min, max) expected execution time in seconds
        convergence_threshold: Expected final error tolerance
        success_criteria: Dictionary of validation criteria
    """

    name: str
    category: str  # 'small', 'medium', 'large'
    description: str
    solver_type: str
    problem_config: dict[str, Any]
    solver_config: dict[str, Any]
    expected_time_range: tuple[float, float]
    convergence_threshold: float
    success_criteria: dict[str, Any]


# ==================== Small Problems (Quick Validation) ====================


def lq_mfg_small() -> BenchmarkProblem:
    """
    Small Linear-Quadratic MFG problem.

    Fast convergence test for basic HJB-FDM solver functionality.
    Expected runtime: 1-3 seconds on typical hardware.
    """
    return BenchmarkProblem(
        name="LQ-MFG-Small",
        category="small",
        description="Linear-Quadratic MFG with 50x50 grid",
        solver_type="hjb_fdm",
        problem_config={
            "T": 1.0,
            "Nx": 50,
            "Nt": 50,
            "x_min": -2.0,
            "x_max": 2.0,
            "sigma": 0.5,
            "alpha": 1.0,
            "beta": 1.0,
            "problem_type": "lq_mfg",
        },
        solver_config={
            "max_iterations": 100,
            "tolerance": 1e-6,
            "backend": "numpy",
            "solver_mode": "fast",
        },
        expected_time_range=(0.5, 3.0),
        convergence_threshold=1e-6,
        success_criteria={
            "must_converge": True,
            "max_iterations_allowed": 100,
            "mass_conservation_error": 1e-4,
        },
    )


def congestion_small() -> BenchmarkProblem:
    """
    Small congestion problem.

    Tests nonlinear coupling with moderate grid.
    Expected runtime: 2-5 seconds on typical hardware.
    """
    return BenchmarkProblem(
        name="Congestion-Small",
        category="small",
        description="Congestion MFG with 50x50 grid",
        solver_type="hjb_fdm",
        problem_config={
            "T": 1.0,
            "Nx": 50,
            "Nt": 50,
            "x_min": -1.0,
            "x_max": 1.0,
            "sigma": 0.3,
            "congestion_power": 2,
            "problem_type": "congestion",
        },
        solver_config={
            "max_iterations": 150,
            "tolerance": 1e-5,
            "backend": "numpy",
            "solver_mode": "fast",
        },
        expected_time_range=(1.0, 5.0),
        convergence_threshold=1e-5,
        success_criteria={
            "must_converge": True,
            "max_iterations_allowed": 150,
            "mass_conservation_error": 1e-3,
        },
    )


# ==================== Medium Problems (Standard Testing) ====================


def lq_mfg_medium() -> BenchmarkProblem:
    """
    Medium Linear-Quadratic MFG problem.

    Standard complexity for performance comparison.
    Expected runtime: 10-20 seconds on typical hardware.
    """
    return BenchmarkProblem(
        name="LQ-MFG-Medium",
        category="medium",
        description="Linear-Quadratic MFG with 100x100 grid",
        solver_type="hjb_fdm",
        problem_config={
            "T": 1.0,
            "Nx": 100,
            "Nt": 100,
            "x_min": -2.0,
            "x_max": 2.0,
            "sigma": 0.5,
            "alpha": 1.0,
            "beta": 1.0,
            "problem_type": "lq_mfg",
        },
        solver_config={
            "max_iterations": 200,
            "tolerance": 1e-6,
            "backend": "numpy",
            "solver_mode": "balanced",
        },
        expected_time_range=(5.0, 20.0),
        convergence_threshold=1e-6,
        success_criteria={
            "must_converge": True,
            "max_iterations_allowed": 200,
            "mass_conservation_error": 1e-4,
        },
    )


def congestion_medium() -> BenchmarkProblem:
    """
    Medium congestion problem.

    Nonlinear coupling with larger grid for performance testing.
    Expected runtime: 15-30 seconds on typical hardware.
    """
    return BenchmarkProblem(
        name="Congestion-Medium",
        category="medium",
        description="Congestion MFG with 100x100 grid",
        solver_type="hjb_fdm",
        problem_config={
            "T": 1.0,
            "Nx": 100,
            "Nt": 100,
            "x_min": -1.0,
            "x_max": 1.0,
            "sigma": 0.3,
            "congestion_power": 2,
            "problem_type": "congestion",
        },
        solver_config={
            "max_iterations": 250,
            "tolerance": 1e-5,
            "backend": "numpy",
            "solver_mode": "balanced",
        },
        expected_time_range=(10.0, 30.0),
        convergence_threshold=1e-5,
        success_criteria={
            "must_converge": True,
            "max_iterations_allowed": 250,
            "mass_conservation_error": 1e-3,
        },
    )


# ==================== Large Problems (Stress Testing) ====================


def traffic_2d_large() -> BenchmarkProblem:
    """
    Large 2D traffic flow problem.

    Stress test for 2D solvers with significant computational demands.
    Expected runtime: 60-180 seconds on typical hardware.
    """
    return BenchmarkProblem(
        name="Traffic-2D-Large",
        category="large",
        description="2D Traffic Flow with 50x50x100 grid",
        solver_type="hjb_fdm",
        problem_config={
            "T": 1.0,
            "Nx": 50,
            "Ny": 50,
            "Nt": 100,
            "x_min": 0.0,
            "x_max": 10.0,
            "y_min": 0.0,
            "y_max": 10.0,
            "sigma": 0.5,
            "problem_type": "traffic_2d",
        },
        solver_config={
            "max_iterations": 300,
            "tolerance": 1e-4,
            "backend": "numpy",
            "solver_mode": "balanced",
        },
        expected_time_range=(30.0, 180.0),
        convergence_threshold=1e-4,
        success_criteria={
            "must_converge": True,
            "max_iterations_allowed": 300,
            "mass_conservation_error": 1e-2,
        },
    )


# ==================== Problem Registry ====================


def get_all_problems() -> list[BenchmarkProblem]:
    """Get all standard benchmark problems."""
    return [
        lq_mfg_small(),
        congestion_small(),
        lq_mfg_medium(),
        congestion_medium(),
        traffic_2d_large(),
    ]


def get_problems_by_category(category: str) -> list[BenchmarkProblem]:
    """
    Get benchmark problems by category.

    Args:
        category: Problem category ('small', 'medium', 'large', or 'all')

    Returns:
        List of matching BenchmarkProblem instances
    """
    if category == "all":
        return get_all_problems()

    return [p for p in get_all_problems() if p.category == category]


def get_problem_by_name(name: str) -> BenchmarkProblem:
    """
    Get a specific benchmark problem by name.

    Args:
        name: Problem name (e.g., 'LQ-MFG-Small')

    Returns:
        BenchmarkProblem instance

    Raises:
        ValueError: If problem name not found
    """
    problems = {p.name: p for p in get_all_problems()}
    if name not in problems:
        available = ", ".join(problems.keys())
        raise ValueError(f"Unknown problem '{name}'. Available: {available}")
    return problems[name]


def get_problem_info() -> str:
    """
    Get formatted information about all benchmark problems.

    Returns:
        Formatted string with problem details
    """
    problems = get_all_problems()

    lines = ["Standard Benchmark Problems", "=" * 70, ""]

    for category in ["small", "medium", "large"]:
        cat_problems = [p for p in problems if p.category == category]
        if not cat_problems:
            continue

        lines.append(f"\n{category.upper()} Problems:")
        lines.append("-" * 70)

        for p in cat_problems:
            lines.append(f"\n{p.name}")
            lines.append(f"  Description: {p.description}")
            lines.append(f"  Solver: {p.solver_type}")

            # Grid size info
            if "Nx" in p.problem_config and "Nt" in p.problem_config:
                if "Ny" in p.problem_config:
                    grid = f"{p.problem_config['Nx']}x{p.problem_config['Ny']}x{p.problem_config['Nt']}"
                else:
                    grid = f"{p.problem_config['Nx']}x{p.problem_config['Nt']}"
                lines.append(f"  Grid: {grid}")

            lines.append(f"  Expected time: {p.expected_time_range[0]:.1f}-{p.expected_time_range[1]:.1f}s")
            lines.append(f"  Convergence: {p.convergence_threshold:.0e}")

    return "\n".join(lines)


if __name__ == "__main__":
    # Display problem information
    print(get_problem_info())
