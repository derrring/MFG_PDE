#!/usr/bin/env python3
"""
Compare the current no-flux implementation with a simple approach
"""

import numpy as np
from mfg_pde.alg.mfg_solvers.particle_collocation_solver import ParticleCollocationSolver
from mfg_pde.core.mfg_problem import ExampleMFGProblem
from mfg_pde.geometry import BoundaryConditions

def test_simple_noflux_comparison():
    print("=== Comparing No-Flux Implementations ===")
    
    # Very simple problem for clear testing
    problem = ExampleMFGProblem(
        xmin=0.0, xmax=1.0, Nx=8, T=0.05, Nt=3, sigma=0.2, coefCT=0.1
    )
    
    num_collocation_points = 5
    collocation_points = np.linspace(0.0, 1.0, num_collocation_points).reshape(-1, 1)
    boundary_indices = np.array([0, num_collocation_points-1])
    
    print(f"Problem: Simple test case")
    print(f"Initial mass: {np.sum(problem.m_init * problem.Dx):.6f}")
    
    # Test 1: Current ghost particle implementation
    print(f"\n{'='*50}")
    print(f"TEST 1: Current Ghost Particle Implementation")
    print(f"{'='*50}")
    
    no_flux_bc = BoundaryConditions(type='no_flux')
    
    try:
        solver1 = ParticleCollocationSolver(
            problem=problem,
            collocation_points=collocation_points,
            num_particles=100,
            delta=0.8,
            taylor_order=1,
            weight_function="wendland",
            NiterNewton=5,
            l2errBoundNewton=1e-4,
            kde_bandwidth="scott",
            normalize_kde_output=True,
            boundary_indices=boundary_indices,
            boundary_conditions=no_flux_bc
        )
        
        # Get SVD diagnostics
        decomp_info = solver1.hjb_solver.get_decomposition_info()
        print(f"SVD coverage: {decomp_info['svd_percentage']:.1f}%")
        print(f"Condition numbers: avg={decomp_info['avg_condition_number']:.1e}")
        
        # Check neighborhood structure
        neighborhood0 = solver1.hjb_solver.neighborhoods[0]
        neighborhood4 = solver1.hjb_solver.neighborhoods[4]
        print(f"Boundary point 0: {neighborhood0['ghost_count']} ghosts, {neighborhood0['size']} total")
        print(f"Boundary point 4: {neighborhood4['ghost_count']} ghosts, {neighborhood4['size']} total")
        
        # Run solver
        U1, M1, info1 = solver1.solve(Niter=2, l2errBound=1e-3, verbose=False)
        
        if M1 is not None:
            mass1 = np.sum(M1 * problem.Dx, axis=1)
            max_U1 = np.max(np.abs(U1)) if U1 is not None else np.inf
            
            print(f"Results:")
            print(f"  Mass change: {mass1[-1] - mass1[0]:.2e}")
            print(f"  Max |U|: {max_U1:.1e}")
            
            if abs(mass1[-1] - mass1[0]) < 1e-10 and max_U1 < 1e3:
                status1 = "✓ EXCELLENT"
            elif mass1[-1] > 0.5:
                status1 = "✓ STABLE"
            else:
                status1 = "❌ COLLAPSED"
            
            print(f"  Status: {status1}")
        else:
            print(f"  ❌ FAILED")
            status1 = "❌ FAILED"
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        status1 = "❌ ERROR"
    
    # Test 2: Dirichlet BC for comparison (should work)
    print(f"\n{'='*50}")
    print(f"TEST 2: Dirichlet BC (Reference)")
    print(f"{'='*50}")
    
    dirichlet_bc = {"type": "dirichlet", "value": 0.0}
    
    try:
        solver2 = ParticleCollocationSolver(
            problem=problem,
            collocation_points=collocation_points,
            num_particles=100,
            delta=0.8,
            taylor_order=1,
            weight_function="wendland",
            NiterNewton=5,
            l2errBoundNewton=1e-4,
            kde_bandwidth="scott",
            normalize_kde_output=True,
            boundary_indices=boundary_indices,
            boundary_conditions=dirichlet_bc
        )
        
        U2, M2, info2 = solver2.solve(Niter=2, l2errBound=1e-3, verbose=False)
        
        if M2 is not None:
            mass2 = np.sum(M2 * problem.Dx, axis=1)
            max_U2 = np.max(np.abs(U2)) if U2 is not None else np.inf
            
            print(f"Results:")
            print(f"  Mass change: {mass2[-1] - mass2[0]:.2e}")
            print(f"  Max |U|: {max_U2:.1e}")
            
            if abs(mass2[-1] - mass2[0]) < 1e-10 and max_U2 < 1e3:
                status2 = "✓ EXCELLENT"
            elif mass2[-1] > 0.5:
                status2 = "✓ STABLE"
            else:
                status2 = "❌ COLLAPSED"
            
            print(f"  Status: {status2}")
        else:
            print(f"  ❌ FAILED")
            status2 = "❌ FAILED"
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        status2 = "❌ ERROR"
    
    # Summary
    print(f"\n{'='*50}")
    print(f"COMPARISON SUMMARY")
    print(f"{'='*50}")
    print(f"{'Method':<25} {'Status'}")
    print(f"{'-'*25} {'-'*15}")
    print(f"{'Ghost Particle No-Flux':<25} {status1}")
    print(f"{'Dirichlet (Reference)':<25} {status2}")
    
    if status1 in ["✓ EXCELLENT", "✓ STABLE"] and status2 in ["✓ EXCELLENT", "✓ STABLE"]:
        print(f"\n✓ Both methods are working - ghost particle no-flux is successful!")
    elif status2 in ["✓ EXCELLENT", "✓ STABLE"]:
        print(f"\n⚠ Dirichlet works but no-flux ghost particle needs improvement")
    else:
        print(f"\n❌ Both methods have issues - check basic implementation")

if __name__ == "__main__":
    test_simple_noflux_comparison()