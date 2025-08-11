#!/usr/bin/env python3
"""
Test script to verify all package imports work correctly after reorganization.
"""


def test_core_imports():
    """Test core package imports"""
    print("Testing core imports...")

    try:
        from mfg_pde import BoundaryConditions, ExampleMFGProblem, MFGProblem

        print("✓ Core imports successful")
    except ImportError as e:
        print(f"✗ Core imports failed: {e}")
        return False

    return True


def test_algorithm_imports():
    """Test algorithm imports"""
    print("Testing algorithm imports...")

    try:
        from mfg_pde.alg import BaseFPSolver, BaseHJBSolver, FixedPointIterator, ParticleCollocationSolver

        print("✓ Algorithm base imports successful")
    except ImportError as e:
        print(f"✗ Algorithm base imports failed: {e}")
        return False

    return True


def test_hjb_solver_imports():
    """Test HJB solver imports"""
    print("Testing HJB solver imports...")

    try:
        from mfg_pde.alg.hjb_solvers import (
            BaseHJBSolver,
            FdmHJBSolver,
            GFDMHJBSolver,
            SmartQPGFDMHJBSolver,
            TunedSmartQPGFDMHJBSolver,
        )

        print("✓ HJB solver imports successful")
    except ImportError as e:
        print(f"✗ HJB solver imports failed: {e}")
        return False

    return True


def test_fp_solver_imports():
    """Test FP solver imports"""
    print("Testing FP solver imports...")

    try:
        from mfg_pde.alg.fp_solvers import BaseFPSolver, FdmFPSolver, ParticleFPSolver

        print("✓ FP solver imports successful")
    except ImportError as e:
        print(f"✗ FP solver imports failed: {e}")
        return False

    return True


def test_utils_imports():
    """Test utility imports"""
    print("Testing utils imports...")

    try:
        from mfg_pde.utils import plot_convergence, plot_results

        print("✓ Utils imports successful")
    except ImportError as e:
        print(f"✗ Utils imports failed: {e}")
        return False

    return True


def test_production_solvers():
    """Test that our main production solvers can be instantiated"""
    print("Testing production solver instantiation...")

    try:
        from mfg_pde import BoundaryConditions, ExampleMFGProblem
        from mfg_pde.alg.hjb_solvers import TunedSmartQPGFDMHJBSolver
        from mfg_pde.alg.particle_collocation_solver import ParticleCollocationSolver

        # Create a simple problem
        problem = ExampleMFGProblem(xmin=0.0, xmax=1.0, Nx=10, T=0.5, Nt=10, sigma=0.1, coefCT=0.02)
        boundary_conditions = BoundaryConditions(type="no_flux")

        # Test that we can create solver instances (without running them)
        import numpy as np

        collocation_points = np.linspace(0, 1, 6).reshape(-1, 1)

        hjb_solver = TunedSmartQPGFDMHJBSolver(
            problem=problem, collocation_points=collocation_points, delta=0.4, boundary_conditions=boundary_conditions
        )

        print("✓ Production solver instantiation successful")
        return True
    except Exception as e:
        print(f"✗ Production solver instantiation failed: {e}")
        return False


def main():
    """Run all import tests"""
    print("=" * 60)
    print("MFG_PDE Package Import Verification")
    print("=" * 60)

    tests = [
        test_core_imports,
        test_algorithm_imports,
        test_hjb_solver_imports,
        test_fp_solver_imports,
        test_utils_imports,
        test_production_solvers,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            results.append(False)
        print()

    print("=" * 60)
    print("IMPORT TEST SUMMARY")
    print("=" * 60)

    passed = sum(results)
    total = len(results)

    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("🎉 All imports working correctly! Package organization successful.")
        return True
    else:
        print("❌ Some imports failed. Package needs additional fixes.")
        return False


if __name__ == "__main__":
    success = main()
