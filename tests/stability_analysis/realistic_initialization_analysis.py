#!/usr/bin/env python3
"""
Realistic analysis of initialization issues in MFG solvers.
Uses a problem with non-zero final conditions to better demonstrate the issue.
"""

import numpy as np
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

from mfg_pde.alg.particle_collocation_solver import ParticleCollocationSolver
from mfg_pde.core.mfg_problem import ExampleMFGProblem

def create_problem_with_nonzero_final():
    """Create a problem with non-zero final condition."""
    
    # Custom MFG problem with non-zero final condition
    class RealisticMFGProblem(ExampleMFGProblem):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            # Override final condition to be non-zero
            for i in range(self.Nx + 1):
                x_i = self.xSpace[i]
                # Quadratic final cost favoring center
                self.u_fin[i] = 10.0 * (x_i - 0.5)**2
    
    problem_params = {
        "xmin": 0.0,
        "xmax": 1.0,
        "Nx": 10,
        "T": 1.0,
        "Nt": 10,
        "sigma": 1.0,
        "coefCT": 0.5,
    }
    
    return RealisticMFGProblem(**problem_params)

def simulate_realistic_first_iteration(problem):
    """Simulate what happens in the first Picard iteration with realistic values."""
    
    Nt = problem.Nt + 1
    Nx = problem.Nx + 1
    
    print(f"Simulating first Picard iteration:")
    
    # === PARTICLE COLLOCATION INITIALIZATION (current approach) ===
    print(f"\nCurrent Particle-Collocation initialization:")
    U_current = np.zeros((Nt, Nx))
    M_current = np.zeros((Nt, Nx))
    
    # Terminal condition
    U_current[Nt - 1, :] = problem.get_final_u()
    
    # Initial condition  
    M_current[0, :] = problem.get_initial_m()
    
    print(f"  U_current: only final time set, rest are zeros")
    print(f"    Final time U[-1]: range [{U_current[-1].min():.2f}, {U_current[-1].max():.2f}], norm={np.linalg.norm(U_current[-1]):.2f}")
    print(f"    Initial time U[0]: range [{U_current[0].min():.2f}, {U_current[0].max():.2f}], norm={np.linalg.norm(U_current[0]):.2f}")
    print(f"    Total U norm: {np.linalg.norm(U_current):.2f}")
    
    print(f"  M_current: only initial time set, rest are zeros")
    print(f"    Initial time M[0]: range [{M_current[0].min():.2f}, {M_current[0].max():.2f}], norm={np.linalg.norm(M_current[0]):.2f}")
    print(f"    Final time M[-1]: range [{M_current[-1].min():.2f}, {M_current[-1].max():.2f}], norm={np.linalg.norm(M_current[-1]):.2f}")
    print(f"    Total M norm: {np.linalg.norm(M_current):.2f}")
    
    # === SIMULATE AFTER FIRST HJB AND FP SOLVE ===
    print(f"\nAfter first HJB and FP solves (simulated realistic results):")
    
    # Simulate HJB result: backward in time, U should decrease from final cost
    U_new = np.zeros((Nt, Nx))
    for t in range(Nt):
        # Linear interpolation from final cost to some smaller value
        progress = t / (Nt - 1)  # 0 at t=0, 1 at t=T
        U_new[t, :] = problem.get_final_u() * (0.3 + 0.7 * progress)  # Goes from 30% to 100% of final cost
    
    # Simulate FP result: forward in time, M should evolve from initial density
    M_new = np.zeros((Nt, Nx))
    for t in range(Nt):
        # Simulate diffusion: spreads out over time
        center_shift = 0.1 * t / (Nt - 1)  # Slight drift
        for i in range(Nx):
            x = problem.xSpace[i]
            # Gaussian that spreads and shifts slightly
            width = 0.1 + 0.05 * t / (Nt - 1)  # Spreads over time
            center = 0.5 + center_shift
            M_new[t, i] = np.exp(-0.5 * ((x - center) / width)**2)
        # Normalize each time slice
        if np.sum(M_new[t, :]) * problem.Dx > 0:
            M_new[t, :] /= np.sum(M_new[t, :]) * problem.Dx
    
    print(f"  U_new (after HJB solve):")
    print(f"    Initial time U[0]: range [{U_new[0].min():.2f}, {U_new[0].max():.2f}], norm={np.linalg.norm(U_new[0]):.2f}")
    print(f"    Final time U[-1]: range [{U_new[-1].min():.2f}, {U_new[-1].max():.2f}], norm={np.linalg.norm(U_new[-1]):.2f}")
    print(f"    Total U norm: {np.linalg.norm(U_new):.2f}")
    
    print(f"  M_new (after FP solve):")
    print(f"    Initial time M[0]: range [{M_new[0].min():.2f}, {M_new[0].max():.2f}], norm={np.linalg.norm(M_new[0]):.2f}")
    print(f"    Final time M[-1]: range [{M_new[-1].min():.2f}, {M_new[-1].max():.2f}], norm={np.linalg.norm(M_new[-1]):.2f}")
    print(f"    Total M norm: {np.linalg.norm(M_new):.2f}")
    
    # === COMPUTE ERRORS ===
    print(f"\nError computation (as in particle collocation solver):")
    U_prev = U_current
    M_prev = M_current
    
    U_error = np.linalg.norm(U_new - U_prev) / max(np.linalg.norm(U_prev), 1e-10)
    M_error = np.linalg.norm(M_new - M_prev) / max(np.linalg.norm(M_prev), 1e-10)
    total_error = max(U_error, M_error)
    
    print(f"  ||U_new - U_prev|| = {np.linalg.norm(U_new - U_prev):.2f}")
    print(f"  ||U_prev|| = {np.linalg.norm(U_prev):.2f}")
    print(f"  U relative error = {U_error:.2e}")
    
    print(f"  ||M_new - M_prev|| = {np.linalg.norm(M_new - M_prev):.2f}")
    print(f"  ||M_prev|| = {np.linalg.norm(M_prev):.2f}")
    print(f"  M relative error = {M_error:.2e}")
    
    print(f"  Total error = {total_error:.2e}")
    
    # === COMPARE WITH BETTER INITIALIZATION ===
    print(f"\n" + "="*60)
    print(f"COMPARISON: Better initialization (like other solvers)")
    
    # Better initialization: fill all time levels
    U_better = np.zeros((Nt, Nx))
    M_better = np.zeros((Nt, Nx))
    
    # Initialize U everywhere with final cost (common in other solvers)
    for t in range(Nt):
        U_better[t, :] = problem.get_final_u()
    
    # Initialize M everywhere with initial density (common in other solvers)
    for t in range(Nt):
        M_better[t, :] = problem.get_initial_m()
    
    print(f"Better initialization:")
    print(f"  U_better: all times initialized with final cost")
    print(f"    Total U norm: {np.linalg.norm(U_better):.2f}")
    print(f"  M_better: all times initialized with initial density")
    print(f"    Total M norm: {np.linalg.norm(M_better):.2f}")
    
    # Compute errors with better initialization
    U_error_better = np.linalg.norm(U_new - U_better) / max(np.linalg.norm(U_better), 1e-10)
    M_error_better = np.linalg.norm(M_new - M_better) / max(np.linalg.norm(M_better), 1e-10)
    total_error_better = max(U_error_better, M_error_better)
    
    print(f"\nError with better initialization:")
    print(f"  U relative error = {U_error_better:.2e} (was {U_error:.2e})")
    print(f"  M relative error = {M_error_better:.2e} (was {M_error:.2e})")
    print(f"  Total error = {total_error_better:.2e} (was {total_error:.2e})")
    
    if total_error > 0 and total_error_better > 0:
        improvement = total_error / total_error_better
        print(f"  Improvement factor: {improvement:.1f}x smaller error")
    
    return {
        'current_error': total_error,
        'better_error': total_error_better,
        'U_error_current': U_error,
        'U_error_better': U_error_better,
        'M_error_current': M_error,
        'M_error_better': M_error_better
    }

def analyze_realistic_initialization():
    """Main analysis function."""
    print("Realistic Analysis of MFG Solver Initialization Issues")
    print("=" * 60)
    
    problem = create_problem_with_nonzero_final()
    
    print(f"Problem setup:")
    print(f"  Grid: {problem.Nx+1} spatial points, {problem.Nt+1} time points")
    print(f"  Domain: [{problem.xmin}, {problem.xmax}] x [0, {problem.T}]")
    
    # Show initial and final conditions
    m_init = problem.get_initial_m()
    u_final = problem.get_final_u()
    
    print(f"\nBoundary conditions:")
    print(f"  Initial density M(0,x): norm={np.linalg.norm(m_init):.2f}, integral={np.sum(m_init)*problem.Dx:.3f}")
    print(f"  Final value U(T,x): range=[{u_final.min():.2f}, {u_final.max():.2f}], norm={np.linalg.norm(u_final):.2f}")
    
    # Simulate first iteration
    results = simulate_realistic_first_iteration(problem)
    
    # Summary
    print(f"\n" + "="*60)
    print(f"SUMMARY OF FINDINGS:")
    print(f"1. **Root cause**: Particle-collocation initializes most of U and M as zeros")
    print(f"   - Only final time for U and initial time for M are set")
    print(f"   - This gives very small ||U_prev|| and ||M_prev||")
    
    print(f"\n2. **Error amplification**: Relative error = ||new - old|| / max(||old||, 1e-10)")
    print(f"   - When ||old|| is small, denominator ≈ 1e-10 or ||old||")
    print(f"   - Large changes in first iteration get amplified")
    
    print(f"\n3. **Current vs. Better initialization errors**:")
    print(f"   - Current U error: {results['U_error_current']:.2e}")
    print(f"   - Better U error: {results['U_error_better']:.2e}")
    print(f"   - Current M error: {results['M_error_current']:.2e}")
    print(f"   - Better M error: {results['M_error_better']:.2e}")
    
    print(f"\n4. **Fix**: Initialize U and M everywhere with reasonable guesses")
    print(f"   - U: final cost everywhere (U[t,:] = U[T,:] for all t)")
    print(f"   - M: initial density everywhere (M[t,:] = M[0,:] for all t)")
    print(f"   - This matches what other solvers (FDM, damped iterator) do")
    
    print(f"\n5. **Expected benefit**: Smaller initial errors lead to:")
    print(f"   - More meaningful convergence metrics from iteration 0")
    print(f"   - Better assessment of actual solver convergence")
    print(f"   - Potentially faster convergence due to better initial guess")

if __name__ == "__main__":
    analyze_realistic_initialization()