#!/usr/bin/env python3
"""
Test the improved constrained QP implementation with all optimizations
"""

import numpy as np
import time
from mfg_pde.alg.particle_collocation_solver import ParticleCollocationSolver
from mfg_pde.core.mfg_problem import ExampleMFGProblem
from mfg_pde.core.boundaries import BoundaryConditions

def test_improved_qp():
    print("=== Improved Constrained QP Performance Test ===")
    
    # Test with longer simulation to see benefits
    problem = ExampleMFGProblem(
        xmin=0.0, xmax=1.0,
        Nx=40, T=0.5, Nt=20,  # More challenging problem
        sigma=0.3, coefCT=0.1
    )
    
    # Higher resolution setup
    num_colloc = 12
    collocation_points = np.linspace(0.0, 1.0, num_colloc).reshape(-1, 1)
    boundary_indices = np.array([0, num_colloc-1])
    no_flux_bc = BoundaryConditions(type='no_flux')
    
    print(f"Challenging test setup:")
    print(f"  Nx={problem.Nx}, T={problem.T}, Nt={problem.Nt}")
    print(f"  Collocation points: {num_colloc}")
    print(f"  Physics: σ={problem.sigma}, coefCT={problem.coefCT}")
    
    # Common solver parameters
    solver_params = {
        "problem": problem,
        "collocation_points": collocation_points,
        "num_particles": 250,
        "delta": 0.35,
        "taylor_order": 1,
        "weight_function": "wendland",
        "NiterNewton": 10,
        "l2errBoundNewton": 1e-4,
        "kde_bandwidth": "scott",
        "normalize_kde_output": False,
        "boundary_indices": boundary_indices,
        "boundary_conditions": no_flux_bc
    }
    
    print(f"\n{'='*60}")
    print("STANDARD UNCONSTRAINED METHOD")
    print(f"{'='*60}")
    
    # Test standard method
    solver_std = ParticleCollocationSolver(**solver_params, use_monotone_constraints=False)
    
    start_time = time.time()
    U_std, M_std, info_std = solver_std.solve(Niter=8, l2errBound=5e-4, verbose=True)
    time_std = time.time() - start_time
    
    print(f"\n{'='*60}")
    print("IMPROVED CONSTRAINED QP METHOD")
    print(f"{'='*60}")
    
    # Test improved QP method
    solver_qp = ParticleCollocationSolver(**solver_params, use_monotone_constraints=True)
    
    start_time = time.time()
    U_qp, M_qp, info_qp = solver_qp.solve(Niter=8, l2errBound=5e-4, verbose=True)
    time_qp = time.time() - start_time
    
    # Detailed analysis
    print(f"\n{'='*70}")
    print("DETAILED PERFORMANCE ANALYSIS")
    print(f"{'='*70}")
    
    if M_std is not None and M_qp is not None:
        # Mass conservation analysis
        mass_std = np.sum(M_std * problem.Dx, axis=1)
        mass_qp = np.sum(M_qp * problem.Dx, axis=1)
        
        mass_change_std = abs(mass_std[-1] - mass_std[0])
        mass_change_qp = abs(mass_qp[-1] - mass_qp[0])
        
        mass_variation_std = np.max(mass_std) - np.min(mass_std)
        mass_variation_qp = np.max(mass_qp) - np.min(mass_qp)
        
        # Solution stability
        max_U_std = np.max(np.abs(U_std)) if U_std is not None else np.inf
        max_U_qp = np.max(np.abs(U_qp)) if U_qp is not None else np.inf
        
        # Boundary violations
        particles_std = solver_std.get_particles_trajectory()
        particles_qp = solver_qp.get_particles_trajectory()
        
        violations_std = 0
        violations_qp = 0
        
        if particles_std is not None:
            final_particles = particles_std[-1, :]
            violations_std = np.sum((final_particles < 0) | (final_particles > 1))
            
        if particles_qp is not None:
            final_particles = particles_qp[-1, :]
            violations_qp = np.sum((final_particles < 0) | (final_particles > 1))
        
        # Performance metrics
        print(f"{'Metric':<25} {'Standard':<15} {'Improved QP':<15} {'Improvement'}")
        print(f"{'-'*25} {'-'*15} {'-'*15} {'-'*12}")
        
        # Mass conservation metrics
        mass_change_imp = (mass_change_std - mass_change_qp) / mass_change_std * 100 if mass_change_std > 0 else 0
        mass_var_imp = (mass_variation_std - mass_variation_qp) / mass_variation_std * 100 if mass_variation_std > 0 else 0
        
        mass_change_std_str = f"{mass_change_std:.2e}"
        mass_change_qp_str = f"{mass_change_qp:.2e}"
        mass_var_std_str = f"{mass_variation_std:.2e}"
        mass_var_qp_str = f"{mass_variation_qp:.2e}"
        
        print(f"{'Mass change':<25} {mass_change_std_str:<15} {mass_change_qp_str:<15} {mass_change_imp:+.1f}%")
        print(f"{'Mass variation':<25} {mass_var_std_str:<15} {mass_var_qp_str:<15} {mass_var_imp:+.1f}%")
        
        # Solution stability
        stability_imp = (max_U_std - max_U_qp) / max_U_std * 100 if max_U_std > 0 else 0
        max_U_std_str = f"{max_U_std:.1e}"
        max_U_qp_str = f"{max_U_qp:.1e}"
        print(f"{'Max |U|':<25} {max_U_std_str:<15} {max_U_qp_str:<15} {stability_imp:+.1f}%")
        
        # Boundary compliance
        viol_imp = (violations_std - violations_qp) / max(violations_std, 1) * 100
        print(f"{'Boundary violations':<25} {violations_std:<15} {violations_qp:<15} {viol_imp:+.1f}%")
        
        # Performance
        time_overhead = (time_qp - time_std) / time_std * 100
        time_std_str = f"{time_std:.2f}"
        time_qp_str = f"{time_qp:.2f}"
        print(f"{'Runtime (s)':<25} {time_std_str:<15} {time_qp_str:<15} {time_overhead:+.1f}%")
        
        # Convergence
        converged_std = info_std.get('converged', False)
        converged_qp = info_qp.get('converged', False)
        print(f"{'Converged':<25} {converged_std:<15} {converged_qp:<15}")
        
        print(f"\n{'='*70}")
        print("OVERALL ASSESSMENT")
        print(f"{'='*70}")
        
        improvements = 0
        total_metrics = 4
        
        # Count improvements
        if mass_change_imp > 1:
            improvements += 1
            print(f"✓ Mass conservation improved by {mass_change_imp:.1f}%")
        elif mass_change_imp < -5:
            print(f"❌ Mass conservation degraded by {abs(mass_change_imp):.1f}%")
        else:
            print(f"≈ Mass conservation similar ({mass_change_imp:+.1f}%)")
        
        if mass_var_imp > 1:
            improvements += 1
            print(f"✓ Mass variation reduced by {mass_var_imp:.1f}%")
        elif mass_var_imp < -5:
            print(f"❌ Mass variation increased by {abs(mass_var_imp):.1f}%")
        else:
            print(f"≈ Mass variation similar ({mass_var_imp:+.1f}%)")
        
        if stability_imp > 1:
            improvements += 1
            print(f"✓ Solution stability improved by {stability_imp:.1f}%")
        elif stability_imp < -5:
            print(f"❌ Solution stability degraded by {abs(stability_imp):.1f}%")
        else:
            print(f"≈ Solution stability similar ({stability_imp:+.1f}%)")
        
        if viol_imp > 0:
            improvements += 1
            print(f"✓ Boundary violations reduced by {viol_imp:.1f}%")
        elif viol_imp < -10:
            print(f"❌ Boundary violations increased by {abs(viol_imp):.1f}%")
        else:
            print(f"≈ Boundary compliance similar")
        
        print(f"\nComputational overhead: {time_overhead:+.1f}%")
        print(f"Performance improvements: {improvements}/{total_metrics} metrics")
        
        if improvements >= 3:
            print(f"🎉 CONSTRAINED QP WINS: Significant improvements across multiple metrics!")
        elif improvements >= 2:
            print(f"✓ CONSTRAINED QP GOOD: Clear advantages with acceptable overhead")
        elif improvements == 1:
            print(f"⚖ MIXED RESULTS: Some benefits but limited impact")
        else:
            print(f"⚠ STANDARD METHOD PREFERRED: No clear benefits from constraints")
        
        # Specific recommendations
        print(f"\n--- Recommendations ---")
        if time_overhead < 50 and improvements >= 2:
            print("✓ Recommended: Use constrained QP for this problem class")
        elif time_overhead < 100 and improvements >= 1:
            print("⚖ Consider: Use constrained QP for stability-critical applications")
        else:
            print("⚠ Use standard method unless monotonicity is critical")
            
    else:
        print("❌ One or both methods failed - cannot compare")

if __name__ == "__main__":
    test_improved_qp()