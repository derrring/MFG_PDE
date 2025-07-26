#!/usr/bin/env python3
"""
Validation Utilities Example

This example demonstrates how to use the centralized validation utilities
to replace common validation patterns scattered throughout the codebase.
"""

import numpy as np
from mfg_pde.utils.validation import (
    validate_solution_array, validate_mfg_solution, 
    safe_solution_return, validate_convergence_parameters
)

def demonstrate_validation_utilities():
    """
    Demonstrate the validation utilities with various scenarios.
    """
    print("=" * 60)
    print("VALIDATION UTILITIES DEMONSTRATION")
    print("=" * 60)
    
    # Create test data
    print("\n1. Creating test MFG solution data...")
    Nt, Nx = 20, 30
    U = np.random.rand(Nt, Nx) * 10.0  # Value function
    M = np.random.rand(Nt, Nx)         # Distribution
    
    # Normalize distribution to have proper mass
    for t in range(Nt):
        M[t, :] = M[t, :] / np.sum(M[t, :])
    
    print(f"   U shape: {U.shape}, range: [{np.min(U):.3f}, {np.max(U):.3f}]")
    print(f"   M shape: {M.shape}, range: [{np.min(M):.3f}, {np.max(M):.3f}]")

    # Test 1: Valid solution validation
    print("\n2. Testing valid solution validation...")
    try:
        validate_solution_array(U, "Value function")
        validate_solution_array(M, "Distribution", min_value=0.0)
        print("   ✓ Individual array validation passed")
        
        result = validate_mfg_solution(U, M, strict=False)
        print(f"   ✓ Complete MFG validation: {result['valid']}")
        print(f"   - Warnings: {len(result['warnings'])}")
        print(f"   - Mass conservation error: {result['diagnostics'].get('mass_conservation_error', 'N/A'):.3f}%")
        
    except Exception as e:
        print(f"   ✗ Validation failed: {e}")

    # Test 2: Invalid solution with NaN
    print("\n3. Testing invalid solution with NaN...")
    U_invalid = U.copy()
    U_invalid[10, 15] = np.nan  # Introduce NaN
    
    try:
        validate_solution_array(U_invalid, "Invalid U")
        print("   ✗ Should have failed but didn't")
    except Exception as e:
        print(f"   ✓ Correctly detected NaN: {e}")

    # Test 3: Invalid solution with negative values
    print("\n4. Testing distribution with negative values...")
    M_invalid = M.copy()
    M_invalid[5, 10] = -0.1  # Introduce negative value
    
    try:
        validate_solution_array(M_invalid, "Invalid M", min_value=0.0)
        print("   ✗ Should have failed but didn't")
    except Exception as e:
        print(f"   ✓ Correctly detected negative value: {e}")

    # Test 4: Safe solution return
    print("\n5. Testing safe solution return wrapper...")
    U_safe, M_safe, info = safe_solution_return(U, M, {'method': 'test'})
    
    print(f"   ✓ Safe return completed")
    print(f"   - Solution status: {info['solution_status']}")
    print(f"   - Original info preserved: {'method' in info}")
    print(f"   - Validation diagnostics added: {'validation' in info}")
    
    if info['validation']['warnings']:
        print(f"   - Warnings: {info['validation']['warnings']}")

    # Test 5: Parameter validation
    print("\n6. Testing convergence parameter validation...")
    try:
        validate_convergence_parameters(30, 1e-6, "Newton")
        print("   ✓ Valid parameters accepted")
        
        validate_convergence_parameters(-5, 1e-6, "Invalid")
        print("   ✗ Should have failed but didn't")
    except ValueError as e:
        print(f"   ✓ Correctly rejected invalid parameters: {e}")

    # Test 6: Real-world usage pattern
    print("\n7. Real-world usage pattern...")
    
    def mock_solver_method():
        """Mock solver method showing how to use validation."""
        # Simulate some computation
        U_result = np.random.rand(10, 15)
        M_result = np.random.rand(10, 15)
        
        # Add some noise to simulate numerical issues
        if np.random.random() < 0.3:  # 30% chance of NaN
            U_result[5, 7] = np.nan
            
        info = {
            'iterations': 15,
            'converged': True,
            'method': 'mock_solver'
        }
        
        # Use safe return instead of manual validation
        return safe_solution_return(U_result, M_result, info)
    
    try:
        U_result, M_result, info_result = mock_solver_method()
        print(f"   ✓ Mock solver returned: status={info_result['solution_status']}")
        
        if info_result['solution_status'] == 'invalid':
            print(f"   - Validation caught issues: {info_result['validation']['errors']}")
        
    except Exception as e:
        print(f"   ✗ Mock solver failed: {e}")

    print("\n" + "=" * 60)
    print("VALIDATION BENEFITS DEMONSTRATED")
    print("=" * 60)
    print("✓ Centralized validation logic")
    print("✓ Consistent error messages")  
    print("✓ Automatic diagnostics collection")
    print("✓ Safe fallback behavior")
    print("✓ Easy integration with existing code")


if __name__ == "__main__":
    demonstrate_validation_utilities()