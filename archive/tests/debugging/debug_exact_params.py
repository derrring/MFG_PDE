#!/usr/bin/env python3
"""
Test with exact parameters from comparison script
"""

import numpy as np
from mfg_pde.alg.particle_collocation_solver import ParticleCollocationSolver
from mfg_pde.core.mfg_problem import ExampleMFGProblem
from mfg_pde.core.boundaries import BoundaryConditions

def test_exact_params():
    print("=== Testing with Exact Comparison Script Parameters ===")
    
    # Exact parameters from compare_all_no_flux_bc.py
    problem_params = {
        "xmin": 0.0,
        "xmax": 1.0,
        "Nx": 30,
        "T": 0.5,
        "Nt": 25,
        "sigma": 1.0,
        "coefCT": 0.5,
    }
    problem = ExampleMFGProblem(**problem_params)
    
    # Exact solver parameters
    num_collocation_points = 15
    collocation_points = np.linspace(0.0, 1.0, num_collocation_points).reshape(-1, 1)
    boundary_indices = [0, num_collocation_points-1]
    no_flux_bc = BoundaryConditions(type='no_flux')
    num_particles = 300
    
    print(f"Problem: Nx={problem.Nx}, T={problem.T}, Nt={problem.Nt}")
    print(f"Collocation points: {num_collocation_points}")
    print(f"Particles: {num_particles}")
    print(f"Initial mass: {np.sum(problem.m_init * problem.Dx):.6f}")
    
    try:
        solver = ParticleCollocationSolver(
            problem=problem,
            collocation_points=collocation_points,
            num_particles=num_particles,
            delta=0.2,
            taylor_order=2,
            weight_function="gaussian",
            weight_scale=1.0,
            NiterNewton=20,
            l2errBoundNewton=1e-6,
            kde_bandwidth="scott",
            normalize_kde_output=True,
            boundary_indices=np.array(boundary_indices),
            boundary_conditions=no_flux_bc
        )
        
        print("\nRunning solver with max 5 iterations...")
        U, M, info = solver.solve(Niter=5, l2errBound=1e-4, verbose=True)
        
        if M is not None:
            mass_evolution = np.sum(M * problem.Dx, axis=1)
            print(f"\nMass evolution over time:")
            for i, mass in enumerate(mass_evolution[::5]):  # Show every 5th step
                print(f"  t[{i*5}]: {mass:.6f}")
            
            print(f"\nFinal mass: {mass_evolution[-1]:.6f}")
            print(f"Mass change: {mass_evolution[-1] - mass_evolution[0]:.6f}")
            
            # Check if solution is collapsing
            if mass_evolution[-1] < 0.1:
                print("ERROR: Solution is collapsing!")
                print(f"M min/max at final time: [{np.min(M[-1,:]):.6f}, {np.max(M[-1,:]):.6f}]")
            else:
                print("OK: Solution seems stable")
        else:
            print("ERROR: Solver returned None for M")
            
    except Exception as e:
        print(f"Solver failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_exact_params()