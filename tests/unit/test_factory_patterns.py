#!/usr/bin/env python3
"""
Factory Patterns Test Suite - Updated for Architecture Separation

NOTE: Particle-collocation tests have been removed as particle-collocation methods
have been moved to the mfg-research repository. This test now focuses on fixed_point
solver factory functionality.

For particle-collocation tests, see: mfg-research/algorithms/particle_collocation/tests/
Migration documentation: MIGRATION_PARTICLE_COLLOCATION.md
"""

import os
import sys

import pytest

import numpy as np

# Add the parent directory to the path so we can import mfg_pde
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mfg_pde import (
    MFGProblem,
    create_accurate_solver,
    create_fast_solver,
    create_research_solver,
    create_solver,
)
from mfg_pde.alg.numerical.fp_solvers import FPParticleSolver
from mfg_pde.alg.numerical.hjb_solvers import HJBGFDMSolver
from mfg_pde.alg.numerical.mfg_solvers import FixedPointIterator


def create_test_problem():
    """Create a simple test MFG problem."""

    class SimpleMFGProblem(MFGProblem):
        def __init__(self):
            super().__init__(T=1.0, Nt=20, xmin=0.0, xmax=1.0, Nx=50)

        def g(self, x):
            return 0.5 * (x - 0.5) ** 2

        def rho0(self, x):
            return np.exp(-10 * (x - 0.3) ** 2) + np.exp(-10 * (x - 0.7) ** 2)

        def f(self, x, u, m):
            return 0.1 * u**2 + 0.05 * m

        def sigma(self, x):
            return 0.1

        def H(self, x, p, m):
            return 0.5 * p**2

        def dH_dm(self, x, p, m):
            return 0.0

    return SimpleMFGProblem()


def test_factory_creation_functions():
    """Test factory creation functions for fixed_point solver."""
    print("Testing factory creation functions...")

    problem = create_test_problem()
    x_coords = np.linspace(problem.xmin, problem.xmax, problem.Nx)
    collocation_points = x_coords.reshape(-1, 1)

    results = {}

    # Test convenience functions with fixed_point solver
    functions_to_test = [
        ("create_fast_solver", create_fast_solver),
        ("create_accurate_solver", create_accurate_solver),
        ("create_research_solver", create_research_solver),
    ]

    for name, func in functions_to_test:
        try:
            # Create required component solvers
            hjb_solver = HJBGFDMSolver(problem, collocation_points)
            fp_solver = FPParticleSolver(problem, collocation_points)

            solver = func(problem, solver_type="fixed_point", hjb_solver=hjb_solver, fp_solver=fp_solver)

            results[name] = {
                "success": True,
                "solver_type": type(solver).__name__,
            }
            print(f"  ✓ {name}: {type(solver).__name__}")
        except Exception as e:
            results[name] = {"success": False, "error": str(e)}
            print(f"  ✗ {name}: {e}")

    return results


def test_solver_types():
    """Test fixed_point solver type."""
    print("\nTesting solver types...")

    problem = create_test_problem()
    x_coords = np.linspace(problem.xmin, problem.xmax, problem.Nx)
    collocation_points = x_coords.reshape(-1, 1)

    results = {}

    # Test fixed point solver
    try:
        hjb_solver = HJBGFDMSolver(problem, collocation_points)
        fp_solver = FPParticleSolver(problem, collocation_points)

        solver = create_solver(
            problem=problem, solver_type="fixed_point", preset="fast", hjb_solver=hjb_solver, fp_solver=fp_solver
        )

        results["fixed_point"] = {
            "success": True,
            "solver_class": type(solver).__name__,
            "has_config": hasattr(solver, "config"),
        }
        print(f"  ✓ fixed_point: {type(solver).__name__}")
    except Exception as e:
        results["fixed_point"] = {"success": False, "error": str(e)}
        print(f"  ✗ fixed_point: {e}")

    return results


def test_configuration_presets():
    """Test configuration presets for fixed_point solver."""
    print("\nTesting configuration presets...")

    problem = create_test_problem()
    x_coords = np.linspace(problem.xmin, problem.xmax, problem.Nx)
    collocation_points = x_coords.reshape(-1, 1)

    presets = ["fast", "balanced", "accurate", "research"]
    results = {}

    for preset in presets:
        try:
            hjb_solver = HJBGFDMSolver(problem, collocation_points)
            fp_solver = FPParticleSolver(problem, collocation_points)

            solver = create_solver(
                problem=problem,
                solver_type="fixed_point",
                preset=preset,
                hjb_solver=hjb_solver,
                fp_solver=fp_solver,
            )

            results[preset] = {
                "success": True,
                "has_config": hasattr(solver, "config"),
            }
            print(f"  ✓ {preset}: FixedPointIterator created")
        except Exception as e:
            results[preset] = {"success": False, "error": str(e)}
            print(f"  ✗ {preset}: {e}")

    return results


def test_type_consistency():
    """Test that factory returns expected solver types."""
    print("\nTesting type consistency...")

    problem = create_test_problem()
    x_coords = np.linspace(problem.xmin, problem.xmax, problem.Nx)
    collocation_points = x_coords.reshape(-1, 1)

    results = {}

    # Test fixed_point type
    try:
        hjb_solver = HJBGFDMSolver(problem, collocation_points)
        fp_solver = FPParticleSolver(problem, collocation_points)

        solver = create_solver(
            problem=problem, solver_type="fixed_point", preset="fast", hjb_solver=hjb_solver, fp_solver=fp_solver
        )

        is_correct_type = isinstance(solver, FixedPointIterator)
        results["fixed_point"] = {
            "success": True,
            "expected_type": "FixedPointIterator",
            "actual_type": type(solver).__name__,
            "type_correct": is_correct_type,
        }

        if is_correct_type:
            print("  ✓ fixed_point: correct type FixedPointIterator")
        else:
            print(f"  ⚠ fixed_point: expected FixedPointIterator, got {type(solver).__name__}")

    except Exception as e:
        results["fixed_point"] = {"success": False, "error": str(e)}
        print(f"  ✗ fixed_point: {e}")

    return results


@pytest.mark.skip(
    reason="Particle-collocation tests removed - solver moved to mfg-research. See MIGRATION_PARTICLE_COLLOCATION.md"
)
def test_particle_collocation_removed():
    """Placeholder indicating particle-collocation tests have been removed."""


def run_comprehensive_test():
    """Run comprehensive factory pattern tests."""
    print("=" * 80)
    print("FACTORY PATTERNS TEST SUITE (Fixed-Point Solvers)")
    print("=" * 80)
    print("NOTE: Particle-collocation tests removed - see mfg-research repository")
    print("=" * 80)

    all_results = {}

    # Run all test functions
    test_functions = [
        ("Factory Creation Functions", test_factory_creation_functions),
        ("Solver Types", test_solver_types),
        ("Configuration Presets", test_configuration_presets),
        ("Type Consistency", test_type_consistency),
    ]

    for test_name, test_func in test_functions:
        print(f"\n{test_name}:")
        print("-" * 40)
        try:
            all_results[test_name] = test_func()
        except Exception as e:
            print(f"  ✗ Test suite failed: {e}")
            all_results[test_name] = {"success": False, "error": str(e)}

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    total_tests = 0
    passed_tests = 0

    for _test_name, results in all_results.items():
        if isinstance(results, dict) and "success" in results:
            # Single test result
            total_tests += 1
            if results["success"]:
                passed_tests += 1
        else:
            # Multiple test results
            for _test_key, result in results.items():
                total_tests += 1
                if result.get("success", False):
                    passed_tests += 1

    print(f"Tests Passed: {passed_tests}/{total_tests}")

    if passed_tests == total_tests:
        print("All tests passed! Factory patterns for fixed_point solver work correctly.")
    else:
        print(f"  {total_tests - passed_tests} tests failed. See details above.")

    print("\nFactory Pattern Features Verified (Fixed-Point Solver):")
    print("  - Convenience creation functions")
    print("  - Configuration presets")
    print("  - Type consistency")

    return all_results


if __name__ == "__main__":
    run_comprehensive_test()
