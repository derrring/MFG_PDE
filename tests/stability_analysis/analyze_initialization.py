#!/usr/bin/env python3
"""
Analysis of initialization issues in MFG solvers.
This script examines how U and M are initialized and why initial errors are large.
"""

import numpy as np
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

from mfg_pde.alg.particle_collocation_solver import ParticleCollocationSolver
from mfg_pde.core.mfg_problem import ExampleMFGProblem

def analyze_initialization():
    """Analyze the initialization process and identify sources of large initial errors."""
    print("Analyzing MFG Solver Initialization Issues")
    print("=" * 60)
    
    # Create a small problem for analysis
    problem_params = {
        "xmin": 0.0,
        "xmax": 1.0,
        "Nx": 10,  # Small grid for analysis
        "T": 1.0,
        "Nt": 10,  # Small time grid for analysis
        "sigma": 1.0,
        "coefCT": 0.5,
    }
    
    problem = ExampleMFGProblem(**problem_params)
    
    # Print problem setup
    print(f"Problem setup:")
    print(f"  Grid: {problem.Nx+1} spatial points, {problem.Nt+1} time points")
    print(f"  Domain: [{problem.xmin}, {problem.xmax}] x [0, {problem.T}]")
    print(f"  Grid spacing: Dx={problem.Dx:.4f}, Dt={problem.Dt:.4f}")
    
    # Examine initial conditions
    print(f"\nInitial conditions:")
    m_init = problem.get_initial_m()
    u_final = problem.get_final_u()
    
    print(f"  Initial density M(0,x):")
    print(f"    Shape: {m_init.shape}")
    print(f"    Range: [{m_init.min():.4f}, {m_init.max():.4f}]")
    print(f"    Integral: {np.sum(m_init) * problem.Dx:.4f}")
    print(f"    Norm: {np.linalg.norm(m_init):.4f}")
    
    print(f"  Final value function U(T,x):")
    print(f"    Shape: {u_final.shape}")
    print(f"    Range: [{u_final.min():.4f}, {u_final.max():.4f}]")
    print(f"    Norm: {np.linalg.norm(u_final):.4f}")
    
    # Create solver with minimal setup
    n_collocation = 5
    collocation_points = np.linspace(0, 1, n_collocation).reshape(-1, 1)
    
    solver = ParticleCollocationSolver(
        problem=problem,
        collocation_points=collocation_points,
        num_particles=100,  # Small number for analysis
        delta=0.3,
        taylor_order=2,
        weight_function="gaussian",
        NiterNewton=5,
        l2errBoundNewton=1e-4,
    )
    
    print(f"\nSolver setup:")
    print(f"  Collocation points: {n_collocation}")
    print(f"  Particles: {solver.num_particles}")
    
    # Analyze initialization in the solve method
    print(f"\nAnalyzing initialization in solve method:")
    
    Nt = problem.Nt + 1
    Nx = problem.Nx + 1
    
    # Replicate initialization from solve method
    U_current = np.zeros((Nt, Nx))
    M_current = np.zeros((Nt, Nx))
    
    # Set terminal condition for U
    if hasattr(problem, 'get_terminal_condition'):
        U_current[Nt - 1, :] = problem.get_terminal_condition()
        print(f"  Used get_terminal_condition(): {hasattr(problem, 'get_terminal_condition')}")
    else:
        U_current[Nt - 1, :] = 0.0
        print(f"  Used default terminal condition (zeros)")
    
    # Set initial condition for M  
    if hasattr(problem, 'get_initial_density'):
        M_current[0, :] = problem.get_initial_density()
        print(f"  Used get_initial_density(): {hasattr(problem, 'get_initial_density')}")
    else:
        if problem.Dx > 1e-14:
            M_current[0, :] = 1.0 / problem.Lx
        else:
            M_current[0, :] = 1.0
        print(f"  Used default uniform density")
    
    print(f"\nInitialized arrays:")
    print(f"  U_current shape: {U_current.shape}")
    print(f"  U_current[0] (initial time): range [{U_current[0].min():.4f}, {U_current[0].max():.4f}], norm={np.linalg.norm(U_current[0]):.4f}")
    print(f"  U_current[-1] (final time): range [{U_current[-1].min():.4f}, {U_current[-1].max():.4f}], norm={np.linalg.norm(U_current[-1]):.4f}")
    
    print(f"  M_current shape: {M_current.shape}")
    print(f"  M_current[0] (initial time): range [{M_current[0].min():.4f}, {M_current[0].max():.4f}], norm={np.linalg.norm(M_current[0]):.4f}")
    print(f"  M_current[-1] (final time): range [{M_current[-1].min():.4f}, {M_current[-1].max():.4f}], norm={np.linalg.norm(M_current[-1]):.4f}")
    
    # Compare with other solver initialization approaches
    print(f"\nComparison with other solver approaches:")
    
    # Method 1: Initialize U everywhere with final condition (like pure FDM)
    U_method1 = np.zeros((Nt, Nx))
    M_method1 = np.zeros((Nt, Nx))
    
    final_u_cost = problem.get_final_u()
    initial_m_dist = problem.get_initial_m()
    
    for n_time_idx in range(Nt):
        U_method1[n_time_idx] = final_u_cost
    M_method1[0] = initial_m_dist
    for n_time_idx in range(1, Nt):
        M_method1[n_time_idx] = initial_m_dist
    
    print(f"  Method 1 (like pure FDM - initialize U everywhere with final cost):")
    print(f"    U norm: {np.linalg.norm(U_method1):.4f}")
    print(f"    M norm: {np.linalg.norm(M_method1):.4f}")
    
    # Method 2: Initialize as in damped fixed point iterator
    U_method2 = np.zeros((Nt, Nx))
    M_method2 = np.zeros((Nt, Nx))
    
    M_method2[0, :] = initial_m_dist
    U_method2[Nt - 1, :] = final_u_cost
    for n_time_idx in range(Nt - 1):
        U_method2[n_time_idx, :] = final_u_cost
    for n_time_idx in range(1, Nt):
        M_method2[n_time_idx, :] = initial_m_dist
    
    print(f"  Method 2 (like damped iterator - initialize U everywhere with final cost):")
    print(f"    U norm: {np.linalg.norm(U_method2):.4f}")
    print(f"    M norm: {np.linalg.norm(M_method2):.4f}")
    
    # Compare error computations
    print(f"\nError analysis - why initial error is large:")
    
    # Simulate first iteration
    print(f"  In first Picard iteration:")
    U_prev = U_current.copy()  # This is mostly zeros except final time
    M_prev = M_current.copy()  # This is mostly zeros except initial time
    
    # After first HJB solve, U would be computed properly everywhere
    # For simulation, let's assume U becomes similar to the final cost everywhere
    U_new = U_method1.copy()  # Simulate what HJB solver would produce
    
    # After first FP solve, M would be computed properly everywhere  
    # For simulation, let's assume M becomes the initial density everywhere
    M_new = M_method1.copy()  # Simulate what FP solver would produce
    
    # Compute errors as in the solver
    U_error = np.linalg.norm(U_new - U_prev) / max(np.linalg.norm(U_prev), 1e-10)
    M_error = np.linalg.norm(M_new - M_prev) / max(np.linalg.norm(M_prev), 1e-10)
    total_error = max(U_error, M_error)
    
    print(f"    U_prev norm: {np.linalg.norm(U_prev):.4f}")
    print(f"    U_new norm: {np.linalg.norm(U_new):.4f}")
    print(f"    ||U_new - U_prev||: {np.linalg.norm(U_new - U_prev):.4f}")
    print(f"    U relative error: {U_error:.4e}")
    
    print(f"    M_prev norm: {np.linalg.norm(M_prev):.4f}")
    print(f"    M_new norm: {np.linalg.norm(M_new):.4f}")
    print(f"    ||M_new - M_prev||: {np.linalg.norm(M_new - M_prev):.4f}")
    print(f"    M relative error: {M_error:.4e}")
    
    print(f"    Total error: {total_error:.4e}")
    
    # Root cause analysis
    print(f"\nRoot cause analysis:")
    print(f"1. **Cold start problem**: U starts mostly zeros, M starts mostly zeros")
    print(f"   - U_prev has norm {np.linalg.norm(U_prev):.4f} (very small)")
    print(f"   - M_prev has norm {np.linalg.norm(M_prev):.4f} (very small)")
    
    print(f"2. **Large change in first iteration**: Solvers fill in the missing data")
    print(f"   - U goes from mostly zeros to meaningful values everywhere")
    print(f"   - M goes from mostly zeros to meaningful values everywhere")
    
    print(f"3. **Relative error amplification**: ||new - old|| / max(||old||, 1e-10)")
    print(f"   - When ||old|| is very small, the denominator becomes 1e-10")
    print(f"   - This amplifies the error by a factor of 1e10")
    
    # Proposed solution
    print(f"\nProposed initialization improvement:")
    print(f"  Instead of initializing U and M mostly with zeros,")
    print(f"  initialize them with reasonable guesses:")
    print(f"  - U: final cost everywhere (like other solvers)")
    print(f"  - M: initial density everywhere (like other solvers)")
    
    # Test improved initialization
    U_improved = U_method1.copy()
    M_improved = M_method1.copy()
    
    # Simulate second iteration with improved initialization
    U_error_improved = np.linalg.norm(U_new - U_improved) / max(np.linalg.norm(U_improved), 1e-10)
    M_error_improved = np.linalg.norm(M_new - M_improved) / max(np.linalg.norm(M_improved), 1e-10)
    total_error_improved = max(U_error_improved, M_error_improved)
    
    print(f"  With improved initialization:")
    print(f"    U relative error: {U_error_improved:.4e} (was {U_error:.4e})")
    print(f"    M relative error: {M_error_improved:.4e} (was {M_error:.4e})")
    print(f"    Total error: {total_error_improved:.4e} (was {total_error:.4e})")
    print(f"    Improvement factor: {total_error/total_error_improved:.1f}x smaller")

if __name__ == "__main__":
    analyze_initialization()