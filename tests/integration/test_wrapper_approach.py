#!/usr/bin/env python3
"""
Test Wrapper Approach for Backend Acceleration.

This tests the zero-modification wrapper strategy where existing
solvers get accelerated WITHOUT changing their source code.
"""

import time

import numpy as np

from mfg_pde.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver
from mfg_pde.core.mfg_problem import MFGProblem

# Import wrappers (if they work)
try:
    from mfg_pde.backends.solver_wrapper import BackendContextManager, accelerate_solver

    WRAPPERS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Wrappers not available: {e}")
    WRAPPERS_AVAILABLE = False


def test_basic_wrapper():
    """Test basic wrapper functionality."""
    print("=" * 80)
    print("WRAPPER APPROACH TEST")
    print("=" * 80)

    if not WRAPPERS_AVAILABLE:
        print("\n❌ Wrappers not available, skipping test")
        return

    # Create problem and solver
    problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=50, T=1.0, Nt=50, sigma=1.0)

    # Test 1: NumPy baseline
    print("\n" + "=" * 80)
    print("Test 1: Baseline NumPy Solver")
    print("=" * 80)

    hjb_numpy = HJBFDMSolver(problem)

    M = problem.get_initial_m()
    U_final = problem.get_final_u()
    U_prev = np.zeros((problem.Nt + 1, problem.Nx + 1))

    start = time.time()
    try:
        result_numpy = hjb_numpy.solve_hjb_system(M, U_final, U_prev)
        numpy_time = time.time() - start
        print(f"✅ NumPy solve: {numpy_time:.3f}s")
        print(f"   Result shape: {result_numpy.shape}")
    except Exception as e:
        print(f"❌ NumPy solve failed: {e}")
        return

    # Test 2: Wrapped solver (same instance, accelerated)
    print("\n" + "=" * 80)
    print("Test 2: Wrapper-Accelerated Solver (NumPy backend)")
    print("=" * 80)

    hjb_wrapped = HJBFDMSolver(problem)

    try:
        accelerate_solver(hjb_wrapped, "numpy")  # Use NumPy backend first

        start = time.time()
        result_wrapped = hjb_wrapped.solve_hjb_system(M, U_final, U_prev)
        wrapped_time = time.time() - start

        print(f"✅ Wrapped solve: {wrapped_time:.3f}s")
        print(f"   Result shape: {result_wrapped.shape}")

        # Verify results match
        diff = np.max(np.abs(result_numpy - result_wrapped))
        print(f"   Max difference: {diff:.2e}")

        if diff < 1e-10:
            print("   ✅ Results match!")
        else:
            print("   ⚠️ Results differ!")

    except Exception as e:
        print(f"❌ Wrapped solve failed: {e}")
        import traceback

        traceback.print_exc()

    # Test 3: Context manager
    print("\n" + "=" * 80)
    print("Test 3: Context Manager (Temporary Acceleration)")
    print("=" * 80)

    hjb_context = HJBFDMSolver(problem)

    try:
        # Normal solve
        start = time.time()
        _ = hjb_context.solve_hjb_system(M, U_final, U_prev)
        time1 = time.time() - start
        print(f"Before context: {time1:.3f}s")

        # Accelerated solve
        with BackendContextManager(hjb_context, "numpy"):
            start = time.time()
            _ = hjb_context.solve_hjb_system(M, U_final, U_prev)
            time2 = time.time() - start
            print(f"Inside context: {time2:.3f}s")

        # Back to normal
        start = time.time()
        _ = hjb_context.solve_hjb_system(M, U_final, U_prev)
        time3 = time.time() - start
        print(f"After context:  {time3:.3f}s")

        print("✅ Context manager works!")

    except Exception as e:
        print(f"❌ Context manager failed: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Wrapper Approach Status:

✅ Concept: Wrap solvers without modifying source code
✅ Implementation: solver_wrapper.py created

Current Limitation:
- Wrapping works at the interface level
- But internal solver code still uses NumPy directly
- Wrapper can't intercept internal np.* calls

Why This Approach Has Limits:
1. Solvers use np.* internally (e.g., np.linalg.solve)
2. Wrapper only converts inputs/outputs
3. Internal operations still use NumPy, not backend

Alternative Strategies:
A. **Monkey-patching**: Replace np in solver's namespace (risky)
B. **Source refactoring**: Change np.* to backend.* (Phase 2)
C. **Hybrid**: Wrap critical functions only (partial benefit)

Recommendation: Proceed with Phase 2 refactoring for real acceleration
""")


if __name__ == "__main__":
    test_basic_wrapper()
