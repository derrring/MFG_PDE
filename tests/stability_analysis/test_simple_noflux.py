#!/usr/bin/env python3
"""
Simple test for no-flux boundary conditions with KDE normalization False
"""

import numpy as np
import matplotlib.pyplot as plt
import time

from mfg_pde.alg.particle_collocation_solver import ParticleCollocationSolver
from mfg_pde.core.mfg_problem import ExampleMFGProblem
from mfg_pde.core.boundaries import BoundaryConditions

def test_simple_noflux():
    print("=== Simple No-Flux Test (KDE normalization = False) ===")
    
    # Simple problem parameters
    problem = ExampleMFGProblem(
        xmin=0.0, xmax=1.0,
        Nx=20, T=0.1, Nt=5,
        sigma=0.3, coefCT=0.1
    )
    
    # Simple collocation setup
    num_colloc = 8
    collocation_points = np.linspace(0.0, 1.0, num_colloc).reshape(-1, 1)
    boundary_indices = np.array([0, num_colloc-1])
    
    print(f"Problem: Nx={problem.Nx}, T={problem.T}, Nt={problem.Nt}")
    print(f"Collocation: {num_colloc} points, boundaries at indices {boundary_indices}")
    
    # No-flux boundary conditions
    no_flux_bc = BoundaryConditions(type='no_flux')
    
    try:
        # Create solver with KDE normalization = False
        solver = ParticleCollocationSolver(
            problem=problem,
            collocation_points=collocation_points,
            num_particles=100,
            delta=0.8,
            taylor_order=1,  # Use first order for stability
            weight_function="wendland",
            NiterNewton=8,
            l2errBoundNewton=1e-4,
            kde_bandwidth="scott",
            normalize_kde_output=False,  # As requested
            boundary_indices=boundary_indices,
            boundary_conditions=no_flux_bc
        )
        
        print(f"✓ Solver created successfully")
        
        # Get diagnostics
        decomp_info = solver.hjb_solver.get_decomposition_info()
        print(f"SVD usage: {decomp_info['svd_percentage']:.0f}%")
        print(f"Condition numbers: avg={decomp_info['avg_condition_number']:.1e}")
        
        # Check ghost particle structure
        print("\nGhost particle structure:")
        for b_idx in boundary_indices:
            neighborhood = solver.hjb_solver.neighborhoods[b_idx]
            print(f"  Point {b_idx}: {neighborhood['ghost_count']} ghosts, {neighborhood['size']} total neighbors")
        
        # Run solver
        print(f"\n--- Running solver ---")
        start_time = time.time()
        
        U, M, info = solver.solve(Niter=5, l2errBound=1e-3, verbose=True)
        
        solve_time = time.time() - start_time
        print(f"Solved in {solve_time:.2f}s")
        
        if M is not None:
            # Mass analysis
            mass_evolution = np.sum(M * problem.Dx, axis=1)
            mass_initial = mass_evolution[0]
            mass_final = mass_evolution[-1]
            mass_change = mass_final - mass_initial
            mass_variation = np.max(mass_evolution) - np.min(mass_evolution)
            
            print(f"\n--- Mass Conservation Results ---")
            print(f"Initial mass: {mass_initial:.6f}")
            print(f"Final mass: {mass_final:.6f}")
            print(f"Mass change: {mass_change:.2e}")
            print(f"Mass variation: {mass_variation:.2e}")
            print(f"Relative change: {abs(mass_change)/mass_initial*100:.4f}%")
            
            # Stability assessment
            if abs(mass_change) < 1e-10:
                print("✓ EXCELLENT mass conservation")
            elif abs(mass_change) < 1e-6:
                print("✓ GOOD mass conservation")
            elif abs(mass_change) < 1e-3:
                print("⚠ FAIR mass conservation")
            else:
                print("❌ POOR mass conservation")
            
            # Check for solution stability
            max_U = np.max(np.abs(U)) if U is not None else np.inf
            print(f"Max |U|: {max_U:.1e}")
            
            if max_U < 1e2:
                print("✓ Solution is stable")
            elif max_U < 1e6:
                print("⚠ Solution may be unstable")
            else:
                print("❌ Solution is unstable")
            
            # Particle boundary check
            particles_trajectory = solver.fp_solver.M_particles_trajectory
            if particles_trajectory is not None:
                final_particles = particles_trajectory[-1, :]
                violations = np.sum((final_particles < 0) | (final_particles > 1))
                print(f"Particle boundary violations: {violations}")
                
                if violations == 0:
                    print("✓ All particles stayed in bounds")
                else:
                    print(f"⚠ {violations} particles violated boundaries")
            
            # Quick visualization
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 3, 1)
            plt.plot(problem.tSpace, mass_evolution, 'b-', linewidth=2)
            plt.xlabel('Time')
            plt.ylabel('Total Mass')
            plt.title('Mass Conservation')
            plt.grid(True)
            
            plt.subplot(1, 3, 2)
            plt.contourf(problem.xSpace, problem.tSpace, M, levels=20)
            plt.colorbar(label='Density m(t,x)')
            plt.xlabel('Position x')
            plt.ylabel('Time t')
            plt.title('Density Evolution')
            
            plt.subplot(1, 3, 3)
            if U is not None:
                plt.contourf(problem.xSpace, problem.tSpace, U, levels=20)
                plt.colorbar(label='Value function u(t,x)')
                plt.xlabel('Position x')
                plt.ylabel('Time t')
                plt.title('Value Function')
            
            plt.tight_layout()
            plt.show()
            
        else:
            print("❌ Solver failed: M is None")
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple_noflux()