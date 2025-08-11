#!/usr/bin/env python3
"""
Test script to verify renamed solvers work correctly.
Tests the new equation_method_suffix naming convention.
"""


def test_hjb_solver_imports():
    """Test HJB solver imports with new names"""
    print("Testing HJB solver imports...")

    try:
        from mfg_pde.alg.hjb_solvers import (
            BaseHJBSolver,
            HJBFDMSolver,
            HJBGFDMOptimizedSolver,
            HJBGFDMSmartQPSolver,
            HJBGFDMSolver,
            HJBGFDMTunedSmartQPSolver,
            HJBSemiLagrangianSolver,
        )

        print("✓ All HJB solver imports successful")
        return True
    except ImportError as e:
        print(f"✗ HJB solver imports failed: {e}")
        return False


def test_fp_solver_imports():
    """Test FP solver imports with new names"""
    print("Testing FP solver imports...")

    try:
        from mfg_pde.alg.fp_solvers import BaseFPSolver, FPFDMSolver, FPParticleSolver

        print("✓ All FP solver imports successful")
        return True
    except ImportError as e:
        print(f"✗ FP solver imports failed: {e}")
        return False


def test_solver_instantiation():
    """Test that renamed solvers can be instantiated"""
    print("Testing solver instantiation...")

    try:
        from mfg_pde import BoundaryConditions, ExampleMFGProblem
        from mfg_pde.alg.fp_solvers import FPFDMSolver
        from mfg_pde.alg.hjb_solvers import HJBGFDMTunedSmartQPSolver

        # Create a simple problem
        problem = ExampleMFGProblem(xmin=0.0, xmax=1.0, Nx=10, T=0.5, Nt=10, sigma=0.1, coefCT=0.02)
        boundary_conditions = BoundaryConditions(type="no_flux")

        # Test HJB solver instantiation
        import numpy as np

        collocation_points = np.linspace(0, 1, 6).reshape(-1, 1)

        hjb_solver = HJBGFDMTunedSmartQPSolver(
            problem=problem, collocation_points=collocation_points, delta=0.4, boundary_conditions=boundary_conditions
        )

        # Test FP solver instantiation
        fp_solver = FPFDMSolver(problem=problem, boundary_conditions=boundary_conditions)

        print("✓ Solver instantiation successful")
        print(f"  - HJB Solver: {hjb_solver.__class__.__name__}")
        print(f"  - FP Solver: {fp_solver.__class__.__name__}")
        return True
    except Exception as e:
        print(f"✗ Solver instantiation failed: {e}")
        return False


def test_naming_convention():
    """Verify new naming convention is followed"""
    print("Testing naming convention...")

    try:
        from mfg_pde.alg.fp_solvers import FPFDMSolver, FPParticleSolver
        from mfg_pde.alg.hjb_solvers import HJBFDMSolver, HJBGFDMSolver, HJBGFDMTunedSmartQPSolver

        # Verify naming pattern: Equation_Method_Suffix
        naming_checks = [
            (HJBFDMSolver, "HJB", "FDM"),
            (HJBGFDMSolver, "HJB", "GFDM"),
            (HJBGFDMTunedSmartQPSolver, "HJB", "GFDM"),
            (FPFDMSolver, "FP", "FDM"),
            (FPParticleSolver, "FP", "Particle"),
        ]

        for solver_class, equation, method in naming_checks:
            name = solver_class.__name__
            if not (name.startswith(equation) and method in name):
                print(f"✗ Naming convention violation: {name}")
                return False

        print("✓ Naming convention verification passed")
        print("  Pattern: Equation_Method_Suffix")
        print("  Examples: HJBFDMSolver, HJBGFDMTunedSmartQPSolver, FPParticleSolver")
        return True
    except Exception as e:
        print(f"✗ Naming convention test failed: {e}")
        return False


def test_backward_compatibility_removed():
    """Verify old names are no longer available"""
    print("Testing old names are removed...")

    old_names = [
        "FdmHJBSolver",
        "GFDMHJBSolver",
        "SmartQPGFDMHJBSolver",
        "TunedSmartQPGFDMHJBSolver",
        "FdmFPSolver",
        "ParticleFPSolver",
    ]

    failed_imports = []
    for old_name in old_names:
        try:
            # Try to import from the module where it used to be
            if "HJB" in old_name:
                exec(f"from mfg_pde.alg.hjb_solvers import {old_name}")
            else:
                exec(f"from mfg_pde.alg.fp_solvers import {old_name}")
            failed_imports.append(old_name)
        except ImportError:
            # This is expected - old names should not be importable
            pass

    if failed_imports:
        print(f"✗ Old names still available: {failed_imports}")
        return False
    else:
        print("✓ Old names properly removed")
        return True


def main():
    """Run all renaming tests"""
    print("=" * 60)
    print("MFG_PDE Solver Renaming Verification")
    print("=" * 60)
    print("Testing equation_method_suffix naming convention")

    tests = [
        test_hjb_solver_imports,
        test_fp_solver_imports,
        test_solver_instantiation,
        test_naming_convention,
        test_backward_compatibility_removed,
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
    print("RENAMING TEST SUMMARY")
    print("=" * 60)

    passed = sum(results)
    total = len(results)

    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("🎉 Solver renaming completed successfully!")
        print("✓ New naming convention: equation_method_suffix")
        print("✓ All imports working with new names")
        print("✓ Old names properly removed")
        return True
    else:
        print("❌ Some renaming tests failed. Need additional fixes.")
        return False


if __name__ == "__main__":
    success = main()
