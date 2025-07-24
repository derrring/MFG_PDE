#!/usr/bin/env python3
"""
Debug QP instability by comparing standard vs QP methods with identical parameters
"""

import numpy as np
import time
from mfg_pde.alg.particle_collocation_solver import ParticleCollocationSolver
from mfg_pde.core.mfg_problem import ExampleMFGProblem
from mfg_pde.core.boundaries import BoundaryConditions

def debug_qp_instability():
    print("=== Debugging QP Instability ===")
    
    # Use realistic parameters that showed instability
    problem = ExampleMFGProblem(
        xmin=0.0, xmax=1.0,
        Nx=60, T=1.0, Nt=40,
        sigma=0.25, coefCT=0.08
    )
    
    # Collocation setup
    num_colloc = 15
    collocation_points = np.linspace(0.0, 1.0, num_colloc).reshape(-1, 1)
    boundary_indices = np.array([0, num_colloc-1])
    no_flux_bc = BoundaryConditions(type='no_flux')
    
    print(f"Problem: Nx={problem.Nx}, T={problem.T}, Nt={problem.Nt}")
    print(f"Physics: σ={problem.sigma}, coefCT={problem.coefCT}")
    print(f"Collocation: {num_colloc} points")
    
    # Common solver parameters
    solver_params = {
        "problem": problem,
        "collocation_points": collocation_points,
        "num_particles": 400,
        "delta": 0.4,
        "taylor_order": 1,
        "weight_function": "wendland",
        "NiterNewton": 12,
        "l2errBoundNewton": 1e-4,
        "kde_bandwidth": "scott",
        "normalize_kde_output": False,
        "boundary_indices": boundary_indices,
        "boundary_conditions": no_flux_bc
    }
    
    results = {}
    
    # Test 1: Standard unconstrained method
    print(f"\\n{'='*50}")
    print("TEST 1: Standard Method (Baseline)")
    print(f"{'='*50}")
    
    solver_std = ParticleCollocationSolver(**solver_params, use_monotone_constraints=False)
    
    try:
        start_time = time.time()
        U_std, M_std, info_std = solver_std.solve(Niter=8, l2errBound=2e-3, verbose=True)
        time_std = time.time() - start_time
        
        if M_std is not None:
            mass_std = np.sum(M_std * problem.Dx, axis=1)
            mass_change_std = abs(mass_std[-1] - mass_std[0])
            max_U_std = np.max(np.abs(U_std)) if U_std is not None else np.inf
            
            particles_std = solver_std.get_particles_trajectory()
            violations_std = 0
            if particles_std is not None:
                final_particles = particles_std[-1, :]
                violations_std = np.sum((final_particles < 0) | (final_particles > 1))
            
            results['standard'] = {
                'mass_change': mass_change_std,
                'max_U': max_U_std,
                'violations': violations_std,
                'time': time_std,
                'converged': info_std.get('converged', False),
                'success': True
            }
            
            print(f"Standard results:")
            print(f"  Mass change: {mass_change_std:.3e}")
            print(f"  Max |U|: {max_U_std:.2e}")
            print(f"  Violations: {violations_std}")
            print(f"  Time: {time_std:.2f}s")
            print(f"  Converged: {info_std.get('converged', False)}")
            
        else:
            print("❌ Standard method failed")
            results['standard'] = {'success': False}
            
    except Exception as e:
        print(f"❌ Standard method crashed: {e}")
        results['standard'] = {'success': False, 'error': str(e)}
    
    # Test 2: QP method
    print(f"\\n{'='*50}")
    print("TEST 2: QP Constrained Method")
    print(f"{'='*50}")
    
    solver_qp = ParticleCollocationSolver(**solver_params, use_monotone_constraints=True)
    
    try:
        start_time = time.time()
        U_qp, M_qp, info_qp = solver_qp.solve(Niter=8, l2errBound=2e-3, verbose=True)
        time_qp = time.time() - start_time
        
        if M_qp is not None:
            mass_qp = np.sum(M_qp * problem.Dx, axis=1)
            mass_change_qp = abs(mass_qp[-1] - mass_qp[0])
            max_U_qp = np.max(np.abs(U_qp)) if U_qp is not None else np.inf
            
            particles_qp = solver_qp.get_particles_trajectory()
            violations_qp = 0
            if particles_qp is not None:
                final_particles = particles_qp[-1, :]
                violations_qp = np.sum((final_particles < 0) | (final_particles > 1))
            
            results['qp'] = {
                'mass_change': mass_change_qp,
                'max_U': max_U_qp,
                'violations': violations_qp,
                'time': time_qp,
                'converged': info_qp.get('converged', False),
                'success': True
            }
            
            print(f"QP results:")
            print(f"  Mass change: {mass_change_qp:.3e}")
            print(f"  Max |U|: {max_U_qp:.2e}")
            print(f"  Violations: {violations_qp}")
            print(f"  Time: {time_qp:.2f}s")
            print(f"  Converged: {info_qp.get('converged', False)}")
            
        else:
            print("❌ QP method failed")
            results['qp'] = {'success': False}
            
    except Exception as e:
        print(f"❌ QP method crashed: {e}")
        results['qp'] = {'success': False, 'error': str(e)}
        import traceback
        traceback.print_exc()
    
    # Analysis
    print(f"\\n{'='*60}")
    print("INSTABILITY ANALYSIS")
    print(f"{'='*60}")
    
    if results['standard'].get('success', False) and results['qp'].get('success', False):
        std = results['standard']
        qp = results['qp']
        
        print(f"Comparison:")
        print(f"  Standard: mass={std['mass_change']:.2e}, |U|={std['max_U']:.1e}, violations={std['violations']}")
        print(f"  QP:       mass={qp['mass_change']:.2e}, |U|={qp['max_U']:.1e}, violations={qp['violations']}")
        
        if qp['max_U'] > 1e10 and std['max_U'] < 1e10:
            print("\\n🔍 DIAGNOSIS: QP method produces extreme U values")
            print("   → QP constraints may be over-constraining the system")
            print("   → Bounds or constraint formulation needs adjustment")
            
        if qp['violations'] > std['violations'] * 2:
            print("\\n🔍 DIAGNOSIS: QP method increases boundary violations")
            print("   → QP constraints may interfere with boundary handling")
            
        time_overhead = (qp['time'] - std['time']) / std['time'] * 100
        print(f"\\nPerformance overhead: {time_overhead:+.1f}%")
        
    elif not results['standard'].get('success', False):
        print("❌ Standard method failed - problem may be too challenging")
        
    elif not results['qp'].get('success', False):
        print("❌ QP method failed - constraints likely too restrictive")
        print("   Recommendations:")
        print("   1. Relax constraint bounds")
        print("   2. Reduce constraint activation frequency")  
        print("   3. Check constraint mathematical formulation")
        
    else:
        print("Both methods failed - problem setup issue")

if __name__ == "__main__":
    debug_qp_instability()