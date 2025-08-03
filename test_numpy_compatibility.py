#!/usr/bin/env python3
"""
NumPy compatibility test for MFG_PDE.

Tests that the package works correctly with different NumPy versions
and that the compatibility layer functions properly.
"""

import sys
import numpy as np
import traceback

def test_numpy_compatibility():
    """Test NumPy compatibility across versions."""
    print(f"Testing with NumPy version: {np.__version__}")
    
    try:
        # Test basic NumPy functionality
        print("✓ NumPy imports successfully")
        
        # Test MFG_PDE imports
        import mfg_pde
        print("✓ MFG_PDE imports successfully")
        
        # Test NumPy compatibility layer
        from mfg_pde.utils.numpy_compat import trapezoid, ensure_numpy_compatibility
        print("✓ NumPy compatibility layer imports successfully")
        
        # Test compatibility function
        info = ensure_numpy_compatibility()
        print(f"✓ NumPy compatibility check passed: {info}")
        
        # Test trapezoid function
        x = np.linspace(0, 1, 100)
        y = x**2
        result = trapezoid(y, x)
        expected = 1.0/3.0  # Integral of x^2 from 0 to 1
        
        if abs(result - expected) < 1e-2:
            print(f"✓ Trapezoid integration test passed: {result:.6f} ≈ {expected:.6f}")
        else:
            print(f"✗ Trapezoid integration test failed: {result:.6f} != {expected:.6f}")
            return False
            
        # Test MFG problem creation
        from mfg_pde.core.mfg_problem import MFGProblem
        problem = MFGProblem(xmin=0, xmax=1, T=1.0, Nx=50, Nt=25)
        print("✓ MFG problem creation successful")
        
        # Test factory functions
        from mfg_pde.factory import create_fast_solver
        from mfg_pde.alg.hjb_solvers.hjb_fdm import HJBFDMSolver
        from mfg_pde.alg.fp_solvers.fp_fdm import FPFDMSolver
        
        hjb_solver = HJBFDMSolver(problem=problem)
        fp_solver = FPFDMSolver(problem=problem)
        
        solver = create_fast_solver(
            problem=problem,
            solver_type="fixed_point",
            hjb_solver=hjb_solver,
            fp_solver=fp_solver
        )
        print("✓ Solver creation successful")
        
        print(f"\n🎉 All NumPy compatibility tests passed with NumPy {np.__version__}")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        print(f"Traceback:")
        traceback.print_exc()
        return False

def benchmark_integration_methods():
    """Benchmark different integration methods for performance comparison."""
    print("Benchmarking integration methods...")
    
    try:
        from mfg_pde.utils.numpy_compat import trapezoid
        import time
        
        # Create test data
        x = np.linspace(0, 10, 10000)
        y = np.sin(x) * np.exp(-x/5)
        
        # Benchmark trapezoid method
        start_time = time.time()
        for _ in range(100):
            result = trapezoid(y, x)
        trapezoid_time = time.time() - start_time
        
        print(f"✓ Trapezoid method: {trapezoid_time:.4f}s for 100 integrations")
        print(f"  Result: {result:.6f}")
        
        # Test different array sizes
        sizes = [100, 1000, 10000]
        for size in sizes:
            x_test = np.linspace(0, 1, size)
            y_test = x_test**2
            
            start_time = time.time()
            result = trapezoid(y_test, x_test)
            elapsed = time.time() - start_time
            
            print(f"  Size {size}: {elapsed:.6f}s, result: {result:.6f}")
            
        print("✓ Benchmark completed successfully")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    success = test_numpy_compatibility()
    sys.exit(0 if success else 1)