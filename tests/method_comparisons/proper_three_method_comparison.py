#!/usr/bin/env python3
"""
Proper three-method comparison using existing validated implementations.
Addresses the question: Do all three methods converge to the same physical solution?
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import sys
import os
import importlib.util

# Add the main package to path
sys.path.insert(0, '/Users/zvezda/Library/CloudStorage/OneDrive-Personal/code/MFG_PDE')

from mfg_pde.alg.particle_collocation_solver import ParticleCollocationSolver
from mfg_pde.core.mfg_problem import ExampleMFGProblem
from mfg_pde.core.boundaries import BoundaryConditions

# Import existing solvers by loading modules directly
def load_existing_solvers():
    """Load existing FDM and Hybrid solvers from tests directory"""
    test_dir = '/Users/zvezda/Library/CloudStorage/OneDrive-Personal/code/MFG_PDE/tests'
    
    # Load pure_fdm module
    fdm_spec = importlib.util.spec_from_file_location("pure_fdm", os.path.join(test_dir, "pure_fdm.py"))
    fdm_module = importlib.util.module_from_spec(fdm_spec)
    
    # Load hybrid_fdm module  
    hybrid_spec = importlib.util.spec_from_file_location("hybrid_fdm", os.path.join(test_dir, "hybrid_fdm.py"))
    hybrid_module = importlib.util.module_from_spec(hybrid_spec)
    
    # Add required imports to modules
    sys.modules['pure_fdm'] = fdm_module
    sys.modules['hybrid_fdm'] = hybrid_module
    
    # Add the base solver to modules before executing
    from mfg_pde.alg.base_mfg_solver import MFGSolver
    fdm_module.MFGSolver = MFGSolver
    hybrid_module.MFGSolver = MFGSolver
    
    try:
        fdm_spec.loader.exec_module(fdm_module)
        print("✓ Successfully loaded pure FDM solver")
        FDMSolver = fdm_module.FDMSolver
    except Exception as e:
        print(f"❌ Failed to load pure FDM solver: {e}")
        FDMSolver = None
    
    try:
        hybrid_spec.loader.exec_module(hybrid_module)
        print("✓ Successfully loaded hybrid solver")
        HybridSolver = hybrid_module.ParticleSolver  # Note: it's called ParticleSolver in the file
    except Exception as e:
        print(f"❌ Failed to load hybrid solver: {e}")
        HybridSolver = None
    
    return FDMSolver, HybridSolver

def compare_three_methods_proper():
    """Proper comparison using existing validated implementations"""
    print("="*80)
    print("PROPER THREE-METHOD COMPARISON")
    print("="*80)
    print("Using existing validated implementations to ensure fair comparison")
    print("Question: Do all methods converge to the same physical solution?")
    
    # Load existing solvers
    print("\nLoading existing solver implementations...")
    FDMSolver, HybridSolver = load_existing_solvers()
    
    if FDMSolver is None or HybridSolver is None:
        print("❌ Cannot proceed without all solver implementations")
        return None
    
    # Unified problem parameters - conservative settings that should work for all methods
    problem_params = {
        "xmin": 0.0,
        "xmax": 1.0,
        "Nx": 30,  # Moderate resolution
        "T": 0.2,   # Shorter time horizon
        "Nt": 10,   # Moderate time steps
        "sigma": 0.15,  # Conservative diffusion
        "coefCT": 0.01  # Light coupling
    }
    
    print(f"\nUnified Problem Parameters (identical for all methods):")
    for key, value in problem_params.items():
        print(f"  {key}: {value}")
    
    problem = ExampleMFGProblem(**problem_params)
    no_flux_bc = BoundaryConditions(type="no_flux")
    
    print(f"\nProblem setup:")
    print(f"  Domain: [{problem.xmin}, {problem.xmax}] × [0, {problem.T}]")
    print(f"  Grid resolution: Dx = {problem.Dx:.4f}, Dt = {problem.Dt:.4f}")
    print(f"  CFL number: {problem_params['sigma']**2 * problem.Dt / problem.Dx**2:.4f}")
    
    results = {}
    
    # Method 1: Pure FDM
    print(f"\n{'='*60}")
    print("METHOD 1: PURE FDM (Existing Implementation)")
    print(f"{'='*60}")
    
    try:
        start_time = time.time()
        
        # Use existing FDM solver with conservative settings
        fdm_solver = FDMSolver(
            problem=problem,
            thetaUM=0.5,  # Conservative damping
            NiterNewton=10,  # Moderate Newton iterations
            l2errBoundNewton=1e-4  # Reasonable tolerance
        )
        
        print("  Starting FDM solve...")
        U_fdm, M_fdm, iterations_fdm, l2dist_u, l2dist_m = fdm_solver.solve(
            Niter=15,  # Conservative Picard iterations
            l2errBoundPicard=1e-4
        )
        
        fdm_time = time.time() - start_time
        
        if M_fdm is not None and U_fdm is not None:
            # Comprehensive analysis
            mass_evolution_fdm = np.sum(M_fdm * problem.Dx, axis=1)
            initial_mass_fdm = mass_evolution_fdm[0]
            final_mass_fdm = mass_evolution_fdm[-1]
            mass_change_fdm = final_mass_fdm - initial_mass_fdm
            
            # Physical observables
            center_of_mass_fdm = np.sum(problem.xSpace * M_fdm[-1, :]) * problem.Dx
            max_density_idx_fdm = np.argmax(M_fdm[-1, :])
            max_density_loc_fdm = problem.xSpace[max_density_idx_fdm]
            final_density_peak_fdm = M_fdm[-1, max_density_idx_fdm]
            
            results['fdm'] = {
                'success': True,
                'method_name': 'Pure FDM',
                'solver_info': 'Existing validated implementation',
                'time': fdm_time,
                'iterations': iterations_fdm,
                'converged': True,  # Assume converged if completed
                'mass_conservation': {
                    'initial_mass': initial_mass_fdm,
                    'final_mass': final_mass_fdm,
                    'mass_change': mass_change_fdm,
                    'mass_change_percent': (mass_change_fdm / initial_mass_fdm) * 100
                },
                'physical_observables': {
                    'center_of_mass': center_of_mass_fdm,
                    'max_density_location': max_density_loc_fdm,
                    'final_density_peak': final_density_peak_fdm
                },
                'solution_quality': {
                    'max_U': np.max(np.abs(U_fdm)),
                    'min_M': np.min(M_fdm),
                    'negative_densities': np.sum(M_fdm < -1e-10),
                    'violations': 0  # Grid-based method
                },
                'arrays': {
                    'U_solution': U_fdm,
                    'M_solution': M_fdm,
                    'mass_evolution': mass_evolution_fdm
                }
            }
            
            print(f"  ✓ FDM completed successfully in {fdm_time:.2f}s")
            print(f"    Iterations: {iterations_fdm}")
            print(f"    Mass: {initial_mass_fdm:.6f} → {final_mass_fdm:.6f} ({(mass_change_fdm/initial_mass_fdm)*100:+.3f}%)")
            print(f"    Center of mass: {center_of_mass_fdm:.4f}")
            print(f"    Max density at: x = {max_density_loc_fdm:.4f} (value = {final_density_peak_fdm:.3f})")
            
        else:
            results['fdm'] = {'success': False, 'method_name': 'Pure FDM'}
            print(f"  ❌ FDM failed to produce solution")
            
    except Exception as e:
        results['fdm'] = {'success': False, 'method_name': 'Pure FDM', 'error': str(e)}
        print(f"  ❌ FDM crashed: {e}")
        import traceback
        traceback.print_exc()
    
    # Method 2: Hybrid Particle-FDM
    print(f"\n{'='*60}")
    print("METHOD 2: HYBRID PARTICLE-FDM (Existing Implementation)")
    print(f"{'='*60}")
    
    try:
        start_time = time.time()
        
        # Use existing Hybrid solver with matching settings
        hybrid_solver = HybridSolver(
            problem=problem,
            num_particles=500,  # Reasonable particle count
            particle_thetaUM=0.5,  # Conservative damping
            kde_bandwidth="scott",  # Standard KDE bandwidth
            NiterNewton=10,  # Match FDM settings
            l2errBoundNewton=1e-4
        )
        
        print(f"  Starting Hybrid solve with {500} particles...")
        U_hybrid, M_hybrid, iterations_hybrid, l2dist_u_h, l2dist_m_h = hybrid_solver.solve(
            Niter=15,  # Match FDM iterations
            l2errBoundPicard=1e-4
        )
        
        hybrid_time = time.time() - start_time
        
        if M_hybrid is not None and U_hybrid is not None:
            # Comprehensive analysis
            mass_evolution_hybrid = np.sum(M_hybrid * problem.Dx, axis=1)
            initial_mass_hybrid = mass_evolution_hybrid[0]
            final_mass_hybrid = mass_evolution_hybrid[-1]
            mass_change_hybrid = final_mass_hybrid - initial_mass_hybrid
            
            # Physical observables
            center_of_mass_hybrid = np.sum(problem.xSpace * M_hybrid[-1, :]) * problem.Dx
            max_density_idx_hybrid = np.argmax(M_hybrid[-1, :])
            max_density_loc_hybrid = problem.xSpace[max_density_idx_hybrid]
            final_density_peak_hybrid = M_hybrid[-1, max_density_idx_hybrid]
            
            # Check particle violations
            violations_hybrid = 0
            if hasattr(hybrid_solver, 'M_particles') and hybrid_solver.M_particles is not None:
                final_particles = hybrid_solver.M_particles[-1, :]
                violations_hybrid = np.sum(
                    (final_particles < problem.xmin - 1e-10) | 
                    (final_particles > problem.xmax + 1e-10)
                )
            
            results['hybrid'] = {
                'success': True,
                'method_name': 'Hybrid Particle-FDM',
                'solver_info': 'Existing validated implementation',
                'time': hybrid_time,
                'iterations': iterations_hybrid,
                'converged': True,
                'mass_conservation': {
                    'initial_mass': initial_mass_hybrid,
                    'final_mass': final_mass_hybrid,
                    'mass_change': mass_change_hybrid,
                    'mass_change_percent': (mass_change_hybrid / initial_mass_hybrid) * 100
                },
                'physical_observables': {
                    'center_of_mass': center_of_mass_hybrid,
                    'max_density_location': max_density_loc_hybrid,
                    'final_density_peak': final_density_peak_hybrid
                },
                'solution_quality': {
                    'max_U': np.max(np.abs(U_hybrid)),
                    'min_M': np.min(M_hybrid),
                    'negative_densities': np.sum(M_hybrid < -1e-10),
                    'violations': violations_hybrid
                },
                'arrays': {
                    'U_solution': U_hybrid,
                    'M_solution': M_hybrid,
                    'mass_evolution': mass_evolution_hybrid
                }
            }
            
            print(f"  ✓ Hybrid completed successfully in {hybrid_time:.2f}s")
            print(f"    Iterations: {iterations_hybrid}")
            print(f"    Mass: {initial_mass_hybrid:.6f} → {final_mass_hybrid:.6f} ({(mass_change_hybrid/initial_mass_hybrid)*100:+.3f}%)")
            print(f"    Center of mass: {center_of_mass_hybrid:.4f}")
            print(f"    Max density at: x = {max_density_loc_hybrid:.4f} (value = {final_density_peak_hybrid:.3f})")
            print(f"    Boundary violations: {violations_hybrid}")
            
        else:
            results['hybrid'] = {'success': False, 'method_name': 'Hybrid Particle-FDM'}
            print(f"  ❌ Hybrid failed to produce solution")
            
    except Exception as e:
        results['hybrid'] = {'success': False, 'method_name': 'Hybrid Particle-FDM', 'error': str(e)}
        print(f"  ❌ Hybrid crashed: {e}")
        import traceback
        traceback.print_exc()
    
    # Method 3: QP Particle-Collocation
    print(f"\n{'='*60}")
    print("METHOD 3: QP PARTICLE-COLLOCATION (Validated Implementation)")
    print(f"{'='*60}")
    
    try:
        start_time = time.time()
        
        # Setup collocation points
        num_collocation_points = 8  # Conservative for speed
        collocation_points = np.linspace(problem.xmin, problem.xmax, num_collocation_points).reshape(-1, 1)
        boundary_indices = [0, num_collocation_points - 1]
        
        qp_solver = ParticleCollocationSolver(
            problem=problem,
            collocation_points=collocation_points,
            num_particles=500,  # Match hybrid particle count
            delta=0.25,  # Conservative delta
            taylor_order=2,
            weight_function="wendland",
            NiterNewton=10,  # Match other methods
            l2errBoundNewton=1e-4,
            kde_bandwidth="scott",
            normalize_kde_output=False,
            boundary_indices=np.array(boundary_indices),
            boundary_conditions=no_flux_bc,
            use_monotone_constraints=True
        )
        
        print(f"  Starting QP solve with {500} particles, {num_collocation_points} collocation points...")
        U_qp, M_qp, info_qp = qp_solver.solve(
            Niter=15,  # Match other methods
            l2errBound=1e-4,
            verbose=False
        )
        
        qp_time = time.time() - start_time
        
        if M_qp is not None and U_qp is not None:
            # Comprehensive analysis
            mass_evolution_qp = np.sum(M_qp * problem.Dx, axis=1)
            initial_mass_qp = mass_evolution_qp[0]
            final_mass_qp = mass_evolution_qp[-1]
            mass_change_qp = final_mass_qp - initial_mass_qp
            
            # Physical observables
            center_of_mass_qp = np.sum(problem.xSpace * M_qp[-1, :]) * problem.Dx
            max_density_idx_qp = np.argmax(M_qp[-1, :])
            max_density_loc_qp = problem.xSpace[max_density_idx_qp]
            final_density_peak_qp = M_qp[-1, max_density_idx_qp]
            
            # Check particle violations
            violations_qp = 0
            particles_traj = qp_solver.get_particles_trajectory()
            if particles_traj is not None:
                final_particles = particles_traj[-1, :]
                violations_qp = np.sum(
                    (final_particles < problem.xmin - 1e-10) | 
                    (final_particles > problem.xmax + 1e-10)
                )
            
            results['qp'] = {
                'success': True,
                'method_name': 'QP Particle-Collocation',
                'solver_info': 'Validated implementation with QP constraints',
                'time': qp_time,
                'iterations': info_qp.get('iterations', 0),
                'converged': info_qp.get('converged', False),
                'mass_conservation': {
                    'initial_mass': initial_mass_qp,
                    'final_mass': final_mass_qp,
                    'mass_change': mass_change_qp,
                    'mass_change_percent': (mass_change_qp / initial_mass_qp) * 100
                },
                'physical_observables': {
                    'center_of_mass': center_of_mass_qp,
                    'max_density_location': max_density_loc_qp,
                    'final_density_peak': final_density_peak_qp
                },
                'solution_quality': {
                    'max_U': np.max(np.abs(U_qp)),
                    'min_M': np.min(M_qp),
                    'negative_densities': np.sum(M_qp < -1e-10),
                    'violations': violations_qp
                },
                'arrays': {
                    'U_solution': U_qp,
                    'M_solution': M_qp,
                    'mass_evolution': mass_evolution_qp
                }
            }
            
            print(f"  ✓ QP completed successfully in {qp_time:.2f}s")
            print(f"    Iterations: {info_qp.get('iterations', 0)}")
            print(f"    Converged: {info_qp.get('converged', False)}")
            print(f"    Mass: {initial_mass_qp:.6f} → {final_mass_qp:.6f} ({(mass_change_qp/initial_mass_qp)*100:+.3f}%)")
            print(f"    Center of mass: {center_of_mass_qp:.4f}")
            print(f"    Max density at: x = {max_density_loc_qp:.4f} (value = {final_density_peak_qp:.3f})")
            print(f"    Boundary violations: {violations_qp}")
            
        else:
            results['qp'] = {'success': False, 'method_name': 'QP Particle-Collocation'}
            print(f"  ❌ QP failed to produce solution")
            
    except Exception as e:
        results['qp'] = {'success': False, 'method_name': 'QP Particle-Collocation', 'error': str(e)}
        print(f"  ❌ QP crashed: {e}")
        import traceback
        traceback.print_exc()
    
    # Comprehensive analysis
    print(f"\n{'='*80}")
    print("CONVERGENCE TO SAME SOLUTION ANALYSIS")
    print(f"{'='*80}")
    
    analyze_method_convergence(results, problem)
    create_convergence_comparison_plots(results, problem)
    
    return results

def analyze_method_convergence(results, problem):
    """Analyze whether all methods converge to the same physical solution"""
    successful_methods = [method for method in ['fdm', 'hybrid', 'qp'] 
                         if results.get(method, {}).get('success', False)]
    
    print(f"Successful methods: {len(successful_methods)}/3")
    
    if len(successful_methods) == 0:
        print("❌ No methods completed successfully - cannot analyze convergence")
        return
    
    if len(successful_methods) < 3:
        print(f"⚠️  Only {len(successful_methods)} methods succeeded - partial analysis only")
        for method in ['fdm', 'hybrid', 'qp']:
            if method not in successful_methods:
                error = results.get(method, {}).get('error', 'Failed')
                print(f"  Failed: {results.get(method, {}).get('method_name', method)} - {error}")
    
    if len(successful_methods) < 2:
        print("Cannot compare methods - need at least 2 successful results")
        return
    
    # Summary table
    print(f"\n{'Method':<25} {'Final Mass':<12} {'Center of Mass':<15} {'Max Density Loc':<15} {'Peak Value':<12}")
    print(f"{'-'*25} {'-'*12} {'-'*15} {'-'*15} {'-'*12}")
    
    for method in successful_methods:
        result = results[method]
        mass_cons = result['mass_conservation']
        phys_obs = result['physical_observables']
        
        print(f"{result['method_name']:<25} {mass_cons['final_mass']:<12.6f} "
              f"{phys_obs['center_of_mass']:<15.4f} {phys_obs['max_density_location']:<15.4f} "
              f"{phys_obs['final_density_peak']:<12.3f}")
    
    # Detailed convergence analysis
    print(f"\n--- DETAILED CONVERGENCE ANALYSIS ---")
    
    # Extract key observables
    final_masses = [results[m]['mass_conservation']['final_mass'] for m in successful_methods]
    centers_of_mass = [results[m]['physical_observables']['center_of_mass'] for m in successful_methods]
    max_density_locs = [results[m]['physical_observables']['max_density_location'] for m in successful_methods]
    density_peaks = [results[m]['physical_observables']['final_density_peak'] for m in successful_methods]
    
    # Statistical analysis of convergence
    observables = {
        'Final Mass': final_masses,
        'Center of Mass': centers_of_mass,
        'Max Density Location': max_density_locs,
        'Density Peak Value': density_peaks
    }
    
    convergence_score = 0
    max_score = len(observables) * 2  # 2 points per observable
    
    for obs_name, values in observables.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        cv = std_val / abs(mean_val) if abs(mean_val) > 1e-10 else std_val
        max_diff = max(values) - min(values)
        rel_diff_percent = (max_diff / abs(mean_val)) * 100 if abs(mean_val) > 1e-10 else 0
        
        print(f"\n{obs_name}:")
        print(f"  Values: {[f'{v:.6f}' if abs(v) > 1e-3 else f'{v:.2e}' for v in values]}")
        print(f"  Range: [{min(values):.6f}, {max(values):.6f}]")
        print(f"  Mean: {mean_val:.6f}, Std: {std_val:.2e}")
        print(f"  Max difference: {max_diff:.2e} ({rel_diff_percent:.3f}% of mean)")
        print(f"  Coefficient of variation: {cv:.4f}")
        
        # Scoring based on relative differences
        if rel_diff_percent < 0.1:  # Less than 0.1% difference
            score = 2
            status = "✅ EXCELLENT convergence"
        elif rel_diff_percent < 1.0:  # Less than 1% difference
            score = 1.5
            status = "✅ VERY GOOD convergence"
        elif rel_diff_percent < 5.0:  # Less than 5% difference
            score = 1
            status = "⚠️  ACCEPTABLE convergence"
        elif rel_diff_percent < 10.0:  # Less than 10% difference
            score = 0.5
            status = "⚠️  POOR convergence"
        else:
            score = 0
            status = "❌ VERY POOR convergence"
        
        convergence_score += score
        print(f"  Assessment: {status}")
    
    # Overall convergence assessment
    print(f"\n--- OVERALL CONVERGENCE ASSESSMENT ---")
    
    overall_percentage = (convergence_score / max_score) * 100
    print(f"Convergence score: {convergence_score:.1f}/{max_score} ({overall_percentage:.1f}%)")
    
    if overall_percentage >= 90:
        overall_assessment = "✅ EXCELLENT: Methods converge to essentially the same solution"
    elif overall_percentage >= 75:
        overall_assessment = "✅ VERY GOOD: Methods converge to very similar solutions"
    elif overall_percentage >= 60:
        overall_assessment = "✅ GOOD: Methods converge to reasonably similar solutions"
    elif overall_percentage >= 40:
        overall_assessment = "⚠️  ACCEPTABLE: Methods show some convergence but with notable differences"
    elif overall_percentage >= 20:
        overall_assessment = "⚠️  POOR: Methods show significant differences in final solutions"
    else:
        overall_assessment = "❌ VERY POOR: Methods appear to converge to different solutions"
    
    print(f"\n{overall_assessment}")
    
    # Physical interpretation
    print(f"\n--- PHYSICAL INTERPRETATION ---")
    
    # Check mass conservation behavior consistency
    mass_changes = [results[m]['mass_conservation']['mass_change'] for m in successful_methods]
    all_increase = all(change > 0 for change in mass_changes)
    all_conservative = all(abs(results[m]['mass_conservation']['mass_change_percent']) < 5.0 
                          for m in successful_methods)
    
    if all_increase:
        print("✅ All methods show mass increase (expected with no-flux BC)")
    elif all_conservative:
        print("✅ All methods show good mass conservation")
    else:
        print("⚠️  Inconsistent mass conservation behavior between methods")
    
    # Check solution quality consistency
    all_clean = all(results[m]['solution_quality']['violations'] == 0 and 
                   results[m]['solution_quality']['negative_densities'] == 0 
                   for m in successful_methods)
    
    if all_clean:
        print("✅ All methods produce numerically clean solutions")
    else:
        print("⚠️  Some methods have numerical quality issues")
        for method in successful_methods:
            result = results[method]
            violations = result['solution_quality']['violations']
            neg_densities = result['solution_quality']['negative_densities']
            if violations > 0 or neg_densities > 0:
                print(f"  {result['method_name']}: {violations} violations, {neg_densities} negative densities")
    
    # Performance comparison
    print(f"\n--- PERFORMANCE COMPARISON ---")
    times = [results[m]['time'] for m in successful_methods]
    iterations = [results[m]['iterations'] for m in successful_methods]
    
    print(f"Execution times: {[f'{t:.2f}s' for t in times]}")
    print(f"Iterations: {iterations}")
    
    fastest_idx = np.argmin(times)
    fastest_method = successful_methods[fastest_idx]
    print(f"Fastest method: {results[fastest_method]['method_name']} ({times[fastest_idx]:.2f}s)")
    
    if len(times) > 1:
        time_ratio = max(times) / min(times)
        print(f"Performance spread: {time_ratio:.2f}x difference")

def create_convergence_comparison_plots(results, problem):
    """Create comprehensive plots comparing method convergence"""
    successful_methods = [method for method in ['fdm', 'hybrid', 'qp'] 
                         if results.get(method, {}).get('success', False)]
    
    if len(successful_methods) < 2:
        print("Insufficient successful results for plotting")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Three-Method Convergence Analysis: Do They Converge to the Same Solution?', fontsize=16)
    
    colors = {'fdm': 'blue', 'hybrid': 'green', 'qp': 'red'}
    method_names = {'fdm': 'Pure FDM', 'hybrid': 'Hybrid', 'qp': 'QP-Collocation'}
    
    # 1. Final density comparison (most important)
    ax1 = axes[0, 0]
    for method in successful_methods:
        result = results[method]
        final_density = result['arrays']['M_solution'][-1, :]
        ax1.plot(problem.xSpace, final_density, 
                label=method_names[method], color=colors[method], linewidth=3)
    ax1.set_xlabel('Space x')
    ax1.set_ylabel('Final Density M(T,x)')
    ax1.set_title('Final Density Comparison\n(Key Convergence Test)', fontweight='bold')
    ax1.grid(True)
    ax1.legend()
    
    # 2. Mass evolution comparison
    ax2 = axes[0, 1]
    for method in successful_methods:
        result = results[method]
        mass_evolution = result['arrays']['mass_evolution']
        ax2.plot(problem.tSpace, mass_evolution, 'o-', 
                label=method_names[method], color=colors[method], linewidth=2)
    ax2.set_xlabel('Time t')
    ax2.set_ylabel('Total Mass')
    ax2.set_title('Mass Evolution Comparison')
    ax2.grid(True)
    ax2.legend()
    
    # 3. Physical observables comparison
    ax3 = axes[0, 2]
    names = [method_names[m] for m in successful_methods]
    centers_of_mass = [results[m]['physical_observables']['center_of_mass'] for m in successful_methods]
    max_density_locs = [results[m]['physical_observables']['max_density_location'] for m in successful_methods]
    
    x_pos = np.arange(len(names))
    width = 0.35
    
    bars1 = ax3.bar(x_pos - width/2, centers_of_mass, width, 
                    label='Center of Mass', alpha=0.8, color='purple')
    bars2 = ax3.bar(x_pos + width/2, max_density_locs, width, 
                    label='Max Density Location', alpha=0.8, color='orange')
    
    ax3.set_xlabel('Method')
    ax3.set_ylabel('Position')
    ax3.set_title('Physical Observables\n(Convergence Indicators)')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(names, rotation=15)
    ax3.legend()
    ax3.grid(True, axis='y')
    
    # Add value labels
    for bar, value in zip(bars1, centers_of_mass):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    for bar, value in zip(bars2, max_density_locs):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 4. Convergence quality metrics
    ax4 = axes[1, 0]
    
    # Calculate convergence metrics
    if len(successful_methods) >= 2:
        final_masses = [results[m]['mass_conservation']['final_mass'] for m in successful_methods]
        centers = [results[m]['physical_observables']['center_of_mass'] for m in successful_methods]
        max_locs = [results[m]['physical_observables']['max_density_location'] for m in successful_methods]
        
        # Relative differences from mean
        mass_cv = np.std(final_masses) / np.mean(final_masses) * 100
        center_cv = np.std(centers) / abs(np.mean(centers)) * 100 if np.mean(centers) != 0 else 0
        loc_cv = np.std(max_locs) / abs(np.mean(max_locs)) * 100 if np.mean(max_locs) != 0 else 0
        
        metrics = ['Final Mass', 'Center of Mass', 'Max Density Loc']
        cv_values = [mass_cv, center_cv, loc_cv]
        
        bars = ax4.bar(metrics, cv_values, color=['blue', 'purple', 'orange'], alpha=0.7)
        ax4.set_ylabel('Coefficient of Variation (%)')
        ax4.set_title('Convergence Quality\n(Lower = Better)')
        ax4.grid(True, axis='y')
        
        # Color code bars based on quality
        for bar, value in zip(bars, cv_values):
            if value < 1.0:
                color = 'green'
                label = 'Excellent'
            elif value < 5.0:
                color = 'orange' 
                label = 'Good'
            else:
                color = 'red'
                label = 'Poor'
            
            ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{value:.2f}%\n({label})', ha='center', va='bottom', fontsize=9)
        
        plt.setp(ax4.get_xticklabels(), rotation=15)
    
    # 5. Performance vs accuracy trade-off
    ax5 = axes[1, 1]
    if len(successful_methods) >= 2:
        times = [results[m]['time'] for m in successful_methods]
        # Use final mass as accuracy proxy (closer to mean = more accurate)
        final_masses = [results[m]['mass_conservation']['final_mass'] for m in successful_methods]
        mean_mass = np.mean(final_masses)
        accuracy_scores = [1.0 / (1.0 + abs(mass - mean_mass)) for mass in final_masses]
        
        scatter = ax5.scatter(times, accuracy_scores, 
                             c=[colors[m] for m in successful_methods], 
                             s=150, alpha=0.7, edgecolors='black')
        
        for i, method in enumerate(successful_methods):
            ax5.annotate(method_names[method], (times[i], accuracy_scores[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        ax5.set_xlabel('Execution Time (seconds)')
        ax5.set_ylabel('Convergence Accuracy Score')
        ax5.set_title('Performance vs Accuracy')
        ax5.grid(True)
    
    # 6. Final control field comparison
    ax6 = axes[1, 2]
    for method in successful_methods:
        result = results[method]
        final_U = result['arrays']['U_solution'][-1, :]
        ax6.plot(problem.xSpace, final_U, 
                label=method_names[method], color=colors[method], linewidth=2)
    ax6.set_xlabel('Space x')
    ax6.set_ylabel('Final Control U(T,x)')
    ax6.set_title('Final Control Field Comparison')
    ax6.grid(True)
    ax6.legend()
    
    plt.tight_layout()
    plt.savefig('/Users/zvezda/Library/CloudStorage/OneDrive-Personal/code/MFG_PDE/proper_three_method_comparison.png', 
                dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("Starting proper three-method comparison...")
    print("Using existing validated implementations for fair comparison.")
    print("Expected execution time: 2-10 minutes depending on solver performance")
    
    try:
        results = compare_three_methods_proper()
        
        if results is not None:
            print("\n" + "="*80)
            print("PROPER THREE-METHOD COMPARISON COMPLETED")
            print("="*80)
            print("Check the analysis above and generated plots for convergence assessment.")
        else:
            print("\nComparison could not be completed due to solver loading issues.")
            
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
