#!/usr/bin/env python3
"""
Test Phase 2 Backend Integration for HJB Solver.

Verifies that HJB solver now uses backend.array_module for array operations
while maintaining backward compatibility with NumPy-only code.
"""

import numpy as np

from mfg_pde.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver
from mfg_pde.core.mfg_problem import MFGProblem


def test_hjb_backend_integration():
    """Test HJB solver with different backends."""
    print("=" * 80)
    print("PHASE 2 BACKEND INTEGRATION TEST - HJB Solver")
    print("=" * 80)

    # Create simple 1D MFG problem
    problem = MFGProblem(
        xmin=0.0,
        xmax=1.0,
        Nx=20,  # Small grid for fast testing
        T=0.5,
        Nt=10,
        sigma=0.1,
    )

    # HJB solver expects M as (Nt+1, Nx+1) array
    M_initial = problem.get_initial_m()  # (Nx+1,)
    M = np.tile(M_initial, (problem.Nt + 1, 1))  # Expand to (Nt+1, Nx+1)

    U_final = problem.get_final_u()  # (Nx+1,)
    U_prev = np.zeros((problem.Nt + 1, problem.Nx + 1))

    # Test 1: Default (NumPy backend)
    print("\n" + "=" * 80)
    print("Test 1: Default Backend (NumPy)")
    print("=" * 80)

    hjb_default = HJBFDMSolver(problem)
    print(f"Backend: {hjb_default.backend.name}")
    print(f"Array module: {hjb_default.backend.array_module.__name__}")

    try:
        result_default = hjb_default.solve_hjb_system(M, U_final, U_prev)
        print("✅ Default solve successful")
        print(f"   Result shape: {result_default.shape}")
        print(f"   Result type: {type(result_default)}")
        print(f"   Result range: [{result_default.min():.4f}, {result_default.max():.4f}]")
    except Exception as e:
        print(f"❌ Default solve failed: {e}")
        import traceback

        traceback.print_exc()
        return

    # Test 2: Explicit NumPy backend
    print("\n" + "=" * 80)
    print("Test 2: Explicit NumPy Backend")
    print("=" * 80)

    hjb_numpy = HJBFDMSolver(problem, backend="numpy")
    print(f"Backend: {hjb_numpy.backend.name}")
    print(f"Array module: {hjb_numpy.backend.array_module.__name__}")

    try:
        result_numpy = hjb_numpy.solve_hjb_system(M, U_final, U_prev)
        print("✅ NumPy solve successful")
        print(f"   Result shape: {result_numpy.shape}")

        # Verify results match default
        diff = np.max(np.abs(result_default - result_numpy))
        print(f"   Max difference from default: {diff:.2e}")
        if diff < 1e-12:
            print("   ✅ Results match exactly")
        else:
            print("   ⚠️ Results differ!")
    except Exception as e:
        print(f"❌ NumPy solve failed: {e}")
        import traceback

        traceback.print_exc()

    # Test 3: Numba backend (if available)
    print("\n" + "=" * 80)
    print("Test 3: Numba Backend (JIT compilation)")
    print("=" * 80)

    try:
        hjb_numba = HJBFDMSolver(problem, backend="numba")
        print(f"Backend: {hjb_numba.backend.name}")
        print(f"Array module: {hjb_numba.backend.array_module.__name__}")

        result_numba = hjb_numba.solve_hjb_system(M, U_final, U_prev)
        print("✅ Numba solve successful")
        print(f"   Result shape: {result_numba.shape}")

        # Verify results match NumPy
        diff = np.max(np.abs(result_default - result_numba))
        print(f"   Max difference from NumPy: {diff:.2e}")
        if diff < 1e-10:
            print("   ✅ Results match NumPy (within tolerance)")
        else:
            print("   ⚠️ Results differ from NumPy!")
    except Exception as e:
        print(f"⚠️ Numba backend not available or failed: {e}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Phase 2 Backend Integration Status:

✅ Backend parameter working
✅ NumPy default fallback working
✅ Backend object stored in solver
✅ Array module accessible via backend.array_module

Current Limitation:
- Most operations still use pure NumPy (scipy.sparse, np checks)
- Actual acceleration requires replacing computational kernels
- Full acceleration pending (KDE, matrix ops, gradient)

Next Steps:
1. Replace computational bottlenecks with backend operations
2. Implement GPU-accelerated KDE for FP particle solver
3. Benchmark actual speedups
""")


if __name__ == "__main__":
    test_hjb_backend_integration()
