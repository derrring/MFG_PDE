#!/usr/bin/env python3
"""
Comprehensive NumPy 2.0+ Compatibility Testing Suite

This script thoroughly tests the NumPy compatibility layer across different
scenarios and provides benchmarking for performance validation.
"""

import time
import warnings
from typing import Dict, List, Tuple

import numpy as np


def run_compatibility_tests() -> Dict[str, bool]:
    """Run comprehensive compatibility tests."""
    results = {}
    
    print("🧪 Running NumPy Compatibility Tests")
    print("=" * 40)
    
    # Test 1: Basic import and functionality
    try:
        from mfg_pde.utils.numpy_compat import trapezoid, get_numpy_info
        results["import"] = True
        print("✅ Test 1: Import compatibility - PASSED")
    except Exception as e:
        results["import"] = False
        print(f"❌ Test 1: Import compatibility - FAILED: {e}")
        return results
    
    # Test 2: Function equivalence
    try:
        x = np.linspace(0, 2*np.pi, 100)
        y = np.sin(x)
        
        # Test our trapezoid function
        result_compat = trapezoid(y, x=x)
        
        # Compare with available methods
        if hasattr(np, 'trapezoid'):
            result_numpy = np.trapezoid(y, x=x)
            error = abs(result_compat - result_numpy)
            assert error < 1e-12, f"NumPy trapezoid error: {error}"
        
        if hasattr(np, 'trapz'):
            result_legacy = np.trapz(y, x=x)
            error = abs(result_compat - result_legacy)
            assert error < 1e-12, f"Legacy trapz error: {error}"
        
        results["equivalence"] = True
        print("✅ Test 2: Function equivalence - PASSED")
    except Exception as e:
        results["equivalence"] = False
        print(f"❌ Test 2: Function equivalence - FAILED: {e}")
    
    # Test 3: Different argument patterns
    try:
        test_cases = [
            # (y_data, x_data, dx_value, axis_value, description)
            (np.random.rand(100), None, 0.1, -1, "dx only"),
            (np.random.rand(50, 30), None, 1.0, 0, "2D array, axis=0"),
            (np.random.rand(50, 30), None, 1.0, 1, "2D array, axis=1"),
            (np.random.rand(100), np.linspace(0, 10, 100), None, -1, "with x array"),
        ]
        
        for y, x, dx, axis, desc in test_cases:
            kwargs = {'axis': axis}
            if x is not None:
                kwargs['x'] = x
            if dx is not None:
                kwargs['dx'] = dx
            
            result = trapezoid(y, **kwargs)
            assert np.isfinite(result).all(), f"Non-finite result for {desc}"
        
        results["argument_patterns"] = True
        print("✅ Test 3: Argument patterns - PASSED")
    except Exception as e:
        results["argument_patterns"] = False
        print(f"❌ Test 3: Argument patterns - FAILED: {e}")
    
    # Test 4: Integration with MFG_PDE components
    try:
        from mfg_pde.core.lagrangian_mfg_problem import create_quadratic_lagrangian_mfg
        from mfg_pde.alg.variational_solvers import VariationalMFGSolver
        
        # Create a small problem
        problem = create_quadratic_lagrangian_mfg(xmin=0, xmax=1, Nx=20, T=0.1, Nt=10)
        solver = VariationalMFGSolver(problem)
        
        # Test operations that use trapezoid
        initial_guess = solver.create_initial_guess('gaussian')
        mass_error = solver.compute_mass_conservation_error(initial_guess)
        cost = solver.evaluate_cost_functional(initial_guess)
        
        assert mass_error < 1e-10, f"Mass conservation error too large: {mass_error}"
        assert np.isfinite(cost), f"Cost functional returned non-finite: {cost}"
        
        results["mfg_integration"] = True
        print("✅ Test 4: MFG_PDE integration - PASSED")
    except Exception as e:
        results["mfg_integration"] = False
        print(f"❌ Test 4: MFG_PDE integration - FAILED: {e}")
    
    # Test 5: Performance comparison
    try:
        x = np.linspace(0, 10, 10000)
        y = np.sin(x) * np.exp(-x/5)
        
        # Benchmark our compatibility function
        start_time = time.time()
        for _ in range(100):
            result = trapezoid(y, x=x)
        compat_time = time.time() - start_time
        
        # Compare with native NumPy if available
        if hasattr(np, 'trapezoid'):
            start_time = time.time()
            for _ in range(100):
                result_numpy = np.trapezoid(y, x=x)
            numpy_time = time.time() - start_time
            
            overhead = (compat_time - numpy_time) / numpy_time * 100
            print(f"📊 Performance overhead: {overhead:.1f}%")
        
        results["performance"] = True
        print("✅ Test 5: Performance benchmark - PASSED")
    except Exception as e:
        results["performance"] = False
        print(f"❌ Test 5: Performance benchmark - FAILED: {e}")
    
    return results


def run_version_specific_tests():
    """Run tests specific to different NumPy versions."""
    print("\n🔍 Version-Specific Tests")
    print("=" * 25)
    
    info = {}
    try:
        from mfg_pde.utils.numpy_compat import get_numpy_info
        info = get_numpy_info()
        
        print(f"NumPy Version: {info['numpy_version']}")
        print(f"Tuple: {info['numpy_version_tuple']}")
        print(f"Is 2.0+: {info['is_numpy_2_plus']}")
        print(f"Has trapezoid: {info['has_trapezoid']}")
        print(f"Has trapz: {info['has_trapz']}")
        print(f"Recommended: {info['recommended_method']}")
        
    except Exception as e:
        print(f"❌ Version detection failed: {e}")
        return
    
    # Test deprecation warnings
    if info.get('is_numpy_2_plus', False) and info.get('has_trapz', False):
        print("\n⚠️  Testing deprecation warnings for NumPy 2.0+")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # This should trigger a warning if using trapz on NumPy 2.0+
            if hasattr(np, 'trapz'):
                result = np.trapz([1, 2, 3, 4], x=[0, 1, 2, 3])
            
            if w:
                print(f"✅ Deprecation warning caught: {w[0].message}")
            else:
                print("ℹ️  No deprecation warning (may be suppressed)")


def benchmark_integration_methods():
    """Benchmark different integration methods."""
    print("\n⚡ Integration Method Benchmarks")
    print("=" * 35)
    
    # Create test data
    sizes = [100, 1000, 10000]
    methods = []
    
    # Check available methods
    if hasattr(np, 'trapezoid'):
        methods.append(('np.trapezoid', np.trapezoid))
    if hasattr(np, 'trapz'):
        methods.append(('np.trapz', np.trapz))
    
    try:
        from scipy.integrate import trapezoid as scipy_trapezoid
        methods.append(('scipy.trapezoid', scipy_trapezoid))
    except ImportError:
        pass
    
    from mfg_pde.utils.numpy_compat import trapezoid as compat_trapezoid
    methods.append(('compat.trapezoid', compat_trapezoid))
    
    print(f"Testing {len(methods)} methods on {len(sizes)} data sizes")
    print()
    
    for size in sizes:
        print(f"Data size: {size}")
        x = np.linspace(0, 10, size)
        y = np.sin(x) * np.exp(-x/3)
        
        for name, func in methods:
            try:
                # Warm up
                func(y, x=x)
                
                # Benchmark
                start_time = time.time()
                for _ in range(100 if size <= 1000 else 10):
                    result = func(y, x=x)
                elapsed = time.time() - start_time
                
                print(f"  {name:20}: {elapsed*1000:8.2f} ms")
            except Exception as e:
                print(f"  {name:20}: ERROR - {e}")
        print()


def main():
    """Main test execution."""
    print("🔬 MFG_PDE NumPy Compatibility Test Suite")
    print("=" * 50)
    
    # Run compatibility tests
    results = run_compatibility_tests()
    
    # Run version-specific tests
    run_version_specific_tests()
    
    # Run benchmarks
    benchmark_integration_methods()
    
    # Summary
    print("\n📋 Test Summary")
    print("=" * 15)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success rate: {passed_tests/total_tests*100:.1f}%")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! NumPy compatibility is working perfectly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the output above.")
        return 1


if __name__ == "__main__":
    exit(main())