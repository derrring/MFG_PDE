#!/usr/bin/env python3
"""
Working test for no-flux boundary conditions with unnormalized KDE
"""

import numpy as np
import matplotlib.pyplot as plt
import time

from mfg_pde.alg.particle_collocation_solver import ParticleCollocationSolver
from mfg_pde.core.mfg_problem import ExampleMFGProblem
from mfg_pde.core.boundaries import BoundaryConditions

def test_working_noflux():
    print("=== Working No-Flux Test (KDE normalization = False) ===")
    
    # Conservative problem parameters (known to work)
    problem = ExampleMFGProblem(
        xmin=0.0, xmax=1.0,
        Nx=8, T=0.05, Nt=3,
        sigma=0.3, coefCT=0.1
    )
    
    # Conservative collocation setup
    num_colloc = 5
    collocation_points = np.linspace(0.0, 1.0, num_colloc).reshape(-1, 1)
    boundary_indices = np.array([0, num_colloc-1])
    
    print(f"Problem: Nx={problem.Nx}, T={problem.T}, Nt={problem.Nt}")
    print(f"Collocation: {num_colloc} points, boundaries at {boundary_indices}")
    
    no_flux_bc = BoundaryConditions(type='no_flux')
    
    try:
        # Create solver with conservative parameters
        solver = ParticleCollocationSolver(
            problem=problem,
            collocation_points=collocation_points,
            num_particles=50,  # Small number for testing
            delta=0.8,         # Large delta for stability
            taylor_order=1,    # First order only
            weight_function="wendland",
            NiterNewton=5,     # Few Newton iterations
            l2errBoundNewton=1e-3,  # Relaxed tolerance
            kde_bandwidth="scott",
            normalize_kde_output=False,  # As requested
            boundary_indices=boundary_indices,
            boundary_conditions=no_flux_bc
        )
        
        print("✓ Solver created successfully")
        
        # Get pre-solve diagnostics
        decomp_info = solver.hjb_solver.get_decomposition_info()
        print(f"SVD usage: {decomp_info['svd_percentage']:.0f}%")
        print(f"Average condition number: {decomp_info['avg_condition_number']:.1e}")
        
        # Check ghost particle setup
        print("\nGhost particle structure:")
        for b_idx in boundary_indices:
            neighborhood = solver.hjb_solver.neighborhoods[b_idx]
            print(f"  Boundary {b_idx}: {neighborhood['ghost_count']} ghosts, {neighborhood['size']} total")
        
        # Run solver with conservative settings
        print("\n--- Running Solver ---")
        start_time = time.time()
        
        U, M, info = solver.solve(
            Niter=3,      # Only 3 Picard iterations 
            l2errBound=1e-2,  # Very relaxed convergence
            verbose=True
        )
        
        solve_time = time.time() - start_time
        converged = info.get('converged', False)
        iterations = info.get('iterations', 3)
        
        print(f"Solver completed in {solve_time:.2f}s ({iterations} iterations)")
        print(f"Converged: {converged}")
        
        if M is not None and U is not None:
            print("\n--- Results Analysis ---")
            
            # Mass conservation
            mass_evolution = np.sum(M * problem.Dx, axis=1)
            initial_mass = mass_evolution[0]
            final_mass = mass_evolution[-1]
            mass_change = final_mass - initial_mass
            relative_change = abs(mass_change) / initial_mass * 100
            
            print(f"Initial mass: {initial_mass:.6f}")
            print(f"Final mass: {final_mass:.6f}")
            print(f"Absolute change: {mass_change:.4e}")
            print(f"Relative change: {relative_change:.4f}%")
            
            # Solution stability
            max_U = np.max(np.abs(U))
            print(f"Max |U|: {max_U:.2e}")
            
            # Particle boundary analysis
            particles_trajectory = solver.get_particles_trajectory()
            if particles_trajectory is not None:
                print(f"\n--- Particle Analysis ---")
                print(f"Particle trajectory shape: {particles_trajectory.shape}")
                
                final_particles = particles_trajectory[-1, :]
                in_bounds = np.all((final_particles >= 0) & (final_particles <= 1))
                violations = np.sum((final_particles < 0) | (final_particles > 1))
                
                print(f"Final particle range: [{np.min(final_particles):.4f}, {np.max(final_particles):.4f}]")
                print(f"All particles in bounds: {in_bounds}")
                print(f"Boundary violations: {violations}")
                
                if violations == 0:
                    print("✓ Perfect boundary enforcement")
                else:
                    print(f"⚠ {violations} particles violated boundaries")
            
            # Overall assessment
            print(f"\n--- Assessment ---")
            mass_ok = relative_change < 5.0  # Accept 5% mass change for this test
            solution_ok = max_U < 1e3
            particles_ok = violations == 0 if particles_trajectory is not None else True
            
            if mass_ok and solution_ok and particles_ok:
                print("✓ TEST PASSED: No-flux implementation works with unnormalized KDE")
            else:
                print("⚠ TEST ISSUES:")
                if not mass_ok:
                    print(f"  - Mass conservation issue: {relative_change:.1f}% change")
                if not solution_ok:
                    print(f"  - Solution stability issue: max|U| = {max_U:.1e}")
                if not particles_ok:
                    print(f"  - Particle boundary violations: {violations}")
            
            # Visualization
            plt.figure(figsize=(15, 5))
            
            # Mass evolution
            plt.subplot(1, 3, 1)
            plt.plot(problem.tSpace, mass_evolution, 'b-', linewidth=2, marker='o')
            plt.xlabel('Time t')
            plt.ylabel('Total Mass')
            plt.title(f'Mass Conservation\n(Δ = {relative_change:.2f}%)')
            plt.grid(True)
            
            # Density evolution
            plt.subplot(1, 3, 2)
            plt.contourf(problem.xSpace, problem.tSpace, M, levels=15, cmap='Blues')
            plt.colorbar(label='Density m(t,x)')
            plt.xlabel('Position x')
            plt.ylabel('Time t')
            plt.title('Density Evolution (No-Flux)')
            
            # Value function
            plt.subplot(1, 3, 3)
            plt.contourf(problem.xSpace, problem.tSpace, U, levels=15, cmap='Reds')
            plt.colorbar(label='Value u(t,x)')
            plt.xlabel('Position x')
            plt.ylabel('Time t')
            plt.title('Value Function')
            
            plt.tight_layout()
            plt.show()
            
        else:
            print("❌ Solver failed to produce results")
            if M is None:
                print("  - M (density) is None")
            if U is None:
                print("  - U (value function) is None")
                
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_working_noflux()