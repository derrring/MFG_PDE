#!/usr/bin/env python3
"""
Test constrained QP for monotonicity vs standard unconstrained method
"""

import numpy as np
import time
from mfg_pde.alg.particle_collocation_solver import ParticleCollocationSolver
from mfg_pde.core.mfg_problem import ExampleMFGProblem
from mfg_pde.core.boundaries import BoundaryConditions

def test_monotone_qp_comparison():
    print("=== Constrained QP vs Standard Method Comparison ===")
    
    # Use conservative parameters for fair comparison
    problem = ExampleMFGProblem(
        xmin=0.0, xmax=1.0,
        Nx=30, T=0.3, Nt=12,
        sigma=0.3, coefCT=0.1
    )
    
    # Conservative collocation setup
    num_colloc = 10
    collocation_points = np.linspace(0.0, 1.0, num_colloc).reshape(-1, 1)
    boundary_indices = np.array([0, num_colloc-1])
    no_flux_bc = BoundaryConditions(type='no_flux')
    
    print(f"Problem setup:")
    print(f"  Nx={problem.Nx}, T={problem.T}, Nt={problem.Nt}")
    print(f"  Collocation points: {num_colloc}")
    print(f"  σ={problem.sigma}, coefCT={problem.coefCT}")
    
    # Common solver parameters
    solver_params = {
        "problem": problem,
        "collocation_points": collocation_points,
        "num_particles": 200,
        "delta": 0.4,
        "taylor_order": 1,
        "weight_function": "wendland",
        "NiterNewton": 8,
        "l2errBoundNewton": 1e-4,
        "kde_bandwidth": "scott",
        "normalize_kde_output": False,
        "boundary_indices": boundary_indices,
        "boundary_conditions": no_flux_bc
    }
    
    results = {}
    
    # Test 1: Standard unconstrained method
    print(f"\n{'='*50}")
    print("TEST 1: Standard Unconstrained Method")
    print(f"{'='*50}")
    
    solver_standard = ParticleCollocationSolver(
        **solver_params,
        use_monotone_constraints=False
    )
    
    start_time = time.time()
    U_std, M_std, info_std = solver_standard.solve(Niter=5, l2errBound=1e-3, verbose=True)
    time_std = time.time() - start_time
    
    if M_std is not None:
        mass_evolution_std = np.sum(M_std * problem.Dx, axis=1)
        mass_change_std = abs(mass_evolution_std[-1] - mass_evolution_std[0])
        
        particles_trajectory_std = solver_standard.get_particles_trajectory()
        violations_std = 0
        if particles_trajectory_std is not None:
            final_particles = particles_trajectory_std[-1, :]
            violations_std = np.sum((final_particles < 0) | (final_particles > 1))
        
        max_U_std = np.max(np.abs(U_std)) if U_std is not None else np.inf
        
        results['standard'] = {
            'mass_change': mass_change_std,
            'violations': violations_std,
            'max_U': max_U_std,
            'time': time_std,
            'converged': info_std.get('converged', False)
        }
        
        print(f"Standard method results:")
        print(f"  Mass change: {mass_change_std:.3e}")
        print(f"  Particle violations: {violations_std}")
        print(f"  Max |U|: {max_U_std:.2e}")
        print(f"  Runtime: {time_std:.2f}s")
        print(f"  Converged: {info_std.get('converged', False)}")
    else:
        print("❌ Standard method failed")
        results['standard'] = {'failed': True}
    
    # Test 2: Constrained QP method
    print(f"\n{'='*50}")
    print("TEST 2: Constrained QP Method")
    print(f"{'='*50}")
    
    solver_qp = ParticleCollocationSolver(
        **solver_params,
        use_monotone_constraints=True
    )
    
    start_time = time.time()
    U_qp, M_qp, info_qp = solver_qp.solve(Niter=5, l2errBound=1e-3, verbose=True)
    time_qp = time.time() - start_time
    
    if M_qp is not None:
        mass_evolution_qp = np.sum(M_qp * problem.Dx, axis=1)
        mass_change_qp = abs(mass_evolution_qp[-1] - mass_evolution_qp[0])
        
        particles_trajectory_qp = solver_qp.get_particles_trajectory()
        violations_qp = 0
        if particles_trajectory_qp is not None:
            final_particles = particles_trajectory_qp[-1, :]
            violations_qp = np.sum((final_particles < 0) | (final_particles > 1))
        
        max_U_qp = np.max(np.abs(U_qp)) if U_qp is not None else np.inf
        
        results['qp'] = {
            'mass_change': mass_change_qp,
            'violations': violations_qp,
            'max_U': max_U_qp,
            'time': time_qp,
            'converged': info_qp.get('converged', False)
        }
        
        print(f"Constrained QP results:")
        print(f"  Mass change: {mass_change_qp:.3e}")
        print(f"  Particle violations: {violations_qp}")
        print(f"  Max |U|: {max_U_qp:.2e}")
        print(f"  Runtime: {time_qp:.2f}s")
        print(f"  Converged: {info_qp.get('converged', False)}")
    else:
        print("❌ Constrained QP method failed")
        results['qp'] = {'failed': True}
    
    # Comparison
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    if 'failed' not in results['standard'] and 'failed' not in results['qp']:
        print(f"{'Metric':<20} {'Standard':<15} {'QP':<15} {'Improvement'}")
        print(f"{'-'*20} {'-'*15} {'-'*15} {'-'*12}")
        
        # Mass conservation
        mass_std = results['standard']['mass_change']
        mass_qp = results['qp']['mass_change']
        mass_improvement = (mass_std - mass_qp) / mass_std * 100 if mass_std > 0 else 0
        mass_std_str = f"{mass_std:.2e}"
        mass_qp_str = f"{mass_qp:.2e}"
        print(f"{'Mass change':<20} {mass_std_str:<15} {mass_qp_str:<15} {mass_improvement:+.1f}%")
        
        # Boundary violations
        viol_std = results['standard']['violations']
        viol_qp = results['qp']['violations']
        viol_improvement = (viol_std - viol_qp) / max(viol_std, 1) * 100
        print(f"{'Violations':<20} {viol_std:<15} {viol_qp:<15} {viol_improvement:+.1f}%")
        
        # Solution magnitude
        U_std = results['standard']['max_U']
        U_qp = results['qp']['max_U']
        U_improvement = (U_std - U_qp) / U_std * 100 if U_std > 0 else 0
        U_std_str = f"{U_std:.1e}"
        U_qp_str = f"{U_qp:.1e}"
        print(f"{'Max |U|':<20} {U_std_str:<15} {U_qp_str:<15} {U_improvement:+.1f}%")
        
        # Runtime
        time_std = results['standard']['time']
        time_qp = results['qp']['time']
        time_overhead = (time_qp - time_std) / time_std * 100
        time_std_str = f"{time_std:.2f}"
        time_qp_str = f"{time_qp:.2f}"
        print(f"{'Runtime (s)':<20} {time_std_str:<15} {time_qp_str:<15} {time_overhead:+.1f}%")
        
        print(f"\n--- Assessment ---")
        
        # Overall improvement
        improvements = 0
        total_metrics = 0
        
        if mass_improvement > 0:
            improvements += 1
            print(f"✓ Better mass conservation: {mass_improvement:.1f}% improvement")
        elif mass_improvement < -10:
            print(f"❌ Worse mass conservation: {abs(mass_improvement):.1f}% degradation")
        else:
            print(f"≈ Similar mass conservation")
        total_metrics += 1
        
        if viol_improvement > 0:
            improvements += 1
            print(f"✓ Fewer boundary violations: {viol_improvement:.1f}% improvement")
        elif viol_improvement < -10:
            print(f"❌ More boundary violations: {abs(viol_improvement):.1f}% degradation")
        else:
            print(f"≈ Similar boundary compliance")
        total_metrics += 1
        
        if U_improvement > 0:
            improvements += 1
            print(f"✓ Better solution stability: {U_improvement:.1f}% improvement")
        elif U_improvement < -10:
            print(f"❌ Worse solution stability: {abs(U_improvement):.1f}% degradation")
        else:
            print(f"≈ Similar solution magnitude")
        total_metrics += 1
        
        print(f"\nComputational overhead: {time_overhead:+.1f}%")
        
        if improvements >= 2:
            print(f"🎉 CONSTRAINED QP WINS: {improvements}/{total_metrics} metrics improved!")
        elif improvements == 1:
            print(f"⚖ MIXED RESULTS: {improvements}/{total_metrics} metrics improved")
        else:
            print(f"⚠ STANDARD METHOD PREFERRED: No clear QP advantage")
            
    else:
        if 'failed' in results['standard']:
            print("❌ Standard method failed - cannot compare")
        if 'failed' in results['qp']:
            print("❌ Constrained QP method failed - may need tuning")

if __name__ == "__main__":
    test_monotone_qp_comparison()