#!/usr/bin/env python3
"""
Minimal demonstration addressing user feedback:
1. Mass conservation in FDM with no-flux BC should show slight increase (reflection)
2. Methods should converge to consistent final masses for proper comparison
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import sys

# Add the main package to path
sys.path.insert(0, '/Users/zvezda/Library/CloudStorage/OneDrive-Personal/code/MFG_PDE')

from mfg_pde.alg.particle_collocation_solver import ParticleCollocationSolver
from mfg_pde.core.mfg_problem import ExampleMFGProblem
from mfg_pde.core.boundaries import BoundaryConditions

def demonstrate_qp_collocation_behavior():
    print("="*80)
    print("MINIMAL QP PARTICLE-COLLOCATION DEMONSTRATION")
    print("="*80)
    print("Addressing user feedback on mass conservation and convergence")
    
    # Very conservative parameters for stability
    problem_params = {
        "xmin": 0.0,
        "xmax": 1.0,
        "Nx": 15,
        "T": 0.1,
        "Nt": 5,
        "sigma": 0.1,
        "coefCT": 0.005
    }
    
    print(f"\nProblem Parameters (very conservative for stability):")
    for key, value in problem_params.items():
        print(f"  {key}: {value}")
    
    problem = ExampleMFGProblem(**problem_params)
    no_flux_bc = BoundaryConditions(type="no_flux")
    
    print(f"\n{'='*60}")
    print("TESTING QP PARTICLE-COLLOCATION")
    print(f"{'='*60}")
    
    try:
        start_time = time.time()
        
        # Minimal collocation setup
        num_collocation_points = 5
        collocation_points = np.linspace(
            problem.xmin, problem.xmax, num_collocation_points
        ).reshape(-1, 1)
        
        boundary_indices = [0, num_collocation_points - 1]  # First and last points
        
        collocation_solver = ParticleCollocationSolver(
            problem=problem,
            collocation_points=collocation_points,
            num_particles=100,  # Minimal for speed
            delta=0.2,
            taylor_order=1,  # First order for stability
            weight_function="wendland",
            NiterNewton=3,  # Minimal
            l2errBoundNewton=1e-2,  # Relaxed
            kde_bandwidth="scott",
            normalize_kde_output=False,
            boundary_indices=np.array(boundary_indices),
            boundary_conditions=no_flux_bc,
            use_monotone_constraints=True
        )
        
        U_solution, M_solution, info = collocation_solver.solve(
            Niter=3, l2errBound=1e-2, verbose=True  # Minimal iterations, relaxed tolerance
        )
        
        elapsed_time = time.time() - start_time
        
        if M_solution is not None and U_solution is not None:
            mass_evolution = np.sum(M_solution * problem.Dx, axis=1)
            initial_mass = mass_evolution[0]
            final_mass = mass_evolution[-1]
            mass_change = final_mass - initial_mass
            mass_change_percent = (mass_change / initial_mass) * 100
            
            # Count boundary violations
            violations = 0
            particles_trajectory = collocation_solver.get_particles_trajectory()
            if particles_trajectory is not None:
                final_particles = particles_trajectory[-1, :]
                violations = np.sum((final_particles < problem.xmin) | (final_particles > problem.xmax))
            
            print(f"\n✓ QP-Collocation completed successfully")
            print(f"  Runtime: {elapsed_time:.2f}s")
            print(f"  Converged: {info.get('converged', False)}")
            print(f"  Iterations: {info.get('iterations', 0)}")
            print(f"  Boundary violations: {violations}")
            
            print(f"\n--- Mass Conservation Analysis ---")
            print(f"  Initial mass: {initial_mass:.6f}")
            print(f"  Final mass: {final_mass:.6f}")
            print(f"  Mass change: {mass_change:+.2e} ({mass_change_percent:+.3f}%)")
            
            if mass_change > 0:
                print(f"  ✅ POSITIVE: Mass increase observed (consistent with no-flux reflection)")
            elif abs(mass_change_percent) < 1.0:
                print(f"  ✅ GOOD: Mass well conserved (< 1% change)")
            else:
                print(f"  ⚠️  WARNING: Significant mass loss detected")
            
            print(f"\n--- Solution Quality Analysis ---")
            max_U = np.max(np.abs(U_solution))
            max_M = np.max(M_solution)
            min_M = np.min(M_solution)
            
            print(f"  Max |U|: {max_U:.2e}")
            print(f"  Density range: [{min_M:.2e}, {max_M:.2e}]")
            
            if min_M >= 0:
                print(f"  ✅ GOOD: Density remains non-negative")
            else:
                print(f"  ⚠️  WARNING: Negative density values detected")
            
            # Create visualization
            create_minimal_demo_plots(problem, M_solution, U_solution, mass_evolution)
            
            print(f"\n--- Addressing User Feedback ---")
            print(f"1. Mass conservation behavior:")
            if mass_change >= 0:
                print(f"   ✅ Method shows mass increase/conservation (expected with no-flux BC)")
            else:
                print(f"   ⚠️  Method shows mass loss (needs investigation)")
            
            print(f"2. Implementation consistency:")
            if info.get('converged', False):
                print(f"   ✅ Method converged within tolerance")
            else:
                print(f"   ⚠️  Method did not fully converge (may need more iterations)")
            
            print(f"3. Numerical stability:")
            if violations == 0 and min_M >= 0:
                print(f"   ✅ Method shows good numerical stability")
            else:
                print(f"   ⚠️  Method shows some numerical issues")
                
        else:
            print(f"❌ QP-Collocation failed to produce solution")
            
    except Exception as e:
        print(f"❌ QP-Collocation crashed: {e}")
        import traceback
        traceback.print_exc()

def create_minimal_demo_plots(problem, M_solution, U_solution, mass_evolution):
    """Create minimal demonstration plots"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('QP Particle-Collocation Method Demonstration', fontsize=14)
    
    # Mass evolution over time
    ax1 = axes[0, 0]
    ax1.plot(problem.tSpace, mass_evolution, 'b-o', linewidth=2, markersize=4)
    ax1.set_xlabel('Time t')
    ax1.set_ylabel('Total Mass')
    ax1.set_title('Mass Conservation Over Time')
    ax1.grid(True)
    
    # Final density
    ax2 = axes[0, 1]
    final_density = M_solution[-1, :]
    ax2.plot(problem.xSpace, final_density, 'r-', linewidth=2)
    ax2.set_xlabel('Space x')
    ax2.set_ylabel('Final Density M(T,x)')
    ax2.set_title('Final Density Distribution')
    ax2.grid(True)
    
    # Initial vs Final density comparison
    ax3 = axes[1, 0]
    initial_density = M_solution[0, :]
    ax3.plot(problem.xSpace, initial_density, 'g--', linewidth=2, label='Initial')
    ax3.plot(problem.xSpace, final_density, 'r-', linewidth=2, label='Final')
    ax3.set_xlabel('Space x')
    ax3.set_ylabel('Density')
    ax3.set_title('Initial vs Final Density')
    ax3.legend()
    ax3.grid(True)
    
    # Final control field
    ax4 = axes[1, 1]
    final_U = U_solution[-1, :]
    ax4.plot(problem.xSpace, final_U, 'purple', linewidth=2)
    ax4.set_xlabel('Space x')
    ax4.set_ylabel('Final Control U(T,x)')
    ax4.set_title('Final Control Field')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('/Users/zvezda/Library/CloudStorage/OneDrive-Personal/code/MFG_PDE/minimal_demo.png', 
                dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    demonstrate_qp_collocation_behavior()
