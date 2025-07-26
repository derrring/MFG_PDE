#!/usr/bin/env python3
"""
Comprehensive Three-Method Evaluation: FDM vs Hybrid vs QP-Collocation
Systematic evaluation of mass conservation quality, computational cost, and stability success rate.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Import solvers
from mfg_pde.core.mfg_problem import ExampleMFGProblem
from mfg_pde.alg.hjb_solvers.fdm_hjb import FdmHJBSolver
from mfg_pde.alg.fp_solvers.fdm_fp import FdmFPSolver
from mfg_pde.alg.fp_solvers.particle_fp import ParticleFPSolver
from mfg_pde.alg.damped_fixed_point_iterator import FixedPointIterator
from mfg_pde.alg.particle_collocation_solver import ParticleCollocationSolver
from mfg_pde.core.boundaries import BoundaryConditions

def test_pure_fdm_method(problem_params, solver_params, test_label):
    """Test pure FDM method (FDM-HJB + FDM-FP)"""
    try:
        start_time = time.time()
        
        # Create problem
        problem = ExampleMFGProblem(**problem_params)
        no_flux_bc = BoundaryConditions(type="no_flux")
        
        # Setup FDM solvers
        hjb_solver = FdmHJBSolver(
            problem, 
            NiterNewton=solver_params["newton_iterations"], 
            l2errBoundNewton=solver_params["newton_tolerance"]
        )
        
        fp_solver = FdmFPSolver(
            problem,
            boundary_conditions=no_flux_bc,
        )
        
        # Fixed point iterator
        fdm_iterator = FixedPointIterator(
            problem,
            hjb_solver=hjb_solver,
            fp_solver=fp_solver,
            thetaUM=solver_params["thetaUM"],
        )
        
        # Solve
        solve_start = time.time()
        U_fdm, M_fdm, iters, _, _ = fdm_iterator.solve(
            solver_params["max_iterations"], 
            solver_params["convergence_tolerance"]
        )
        solve_time = time.time() - solve_start
        total_time = time.time() - start_time
        
        if U_fdm is not None and M_fdm is not None:
            # Mass analysis
            mass_evolution = np.sum(M_fdm * problem.Dx, axis=1)
            initial_mass = mass_evolution[0]
            final_mass = mass_evolution[-1]
            mass_change_percent = ((final_mass - initial_mass) / initial_mass) * 100
            
            # Quality metrics
            negative_densities = np.sum(M_fdm < -1e-10)
            min_density = np.min(M_fdm)
            max_control = np.max(np.abs(U_fdm))
            
            return {
                'success': True,
                'method': 'Pure FDM',
                'label': test_label,
                'solve_time': solve_time,
                'total_time': total_time,
                'iterations': iters,
                'mass_change_percent': mass_change_percent,
                'final_mass': final_mass,
                'mass_evolution': mass_evolution,
                'negative_densities': negative_densities,
                'min_density': min_density,
                'max_control': max_control,
                'boundary_violations': 0,  # FDM doesn't have particle violations
                'U': U_fdm,
                'M': M_fdm,
                'converged': abs(mass_change_percent) < 5.0  # Consider converged if mass error < 5%
            }
        else:
            return {'success': False, 'method': 'Pure FDM', 'label': test_label, 'error': 'No solution'}
            
    except Exception as e:
        return {'success': False, 'method': 'Pure FDM', 'label': test_label, 'error': str(e)}

def test_hybrid_method(problem_params, solver_params, test_label):
    """Test Hybrid method (FDM-HJB + Particle-FP)"""
    try:
        start_time = time.time()
        
        # Create problem
        problem = ExampleMFGProblem(**problem_params)
        no_flux_bc = BoundaryConditions(type="no_flux")
        
        # Setup Hybrid solvers
        hjb_solver = FdmHJBSolver(
            problem, 
            NiterNewton=solver_params["newton_iterations"], 
            l2errBoundNewton=solver_params["newton_tolerance"]
        )
        
        fp_solver = ParticleFPSolver(
            problem,
            num_particles=solver_params["num_particles"],
            kde_bandwidth="scott",
            normalize_kde_output=False,
            boundary_conditions=no_flux_bc,
        )
        
        # Fixed point iterator
        hybrid_iterator = FixedPointIterator(
            problem,
            hjb_solver=hjb_solver,
            fp_solver=fp_solver,
            thetaUM=solver_params["thetaUM"],
        )
        
        # Solve
        solve_start = time.time()
        U_hybrid, M_hybrid, iters, _, _ = hybrid_iterator.solve(
            solver_params["max_iterations"], 
            solver_params["convergence_tolerance"]
        )
        solve_time = time.time() - solve_start
        total_time = time.time() - start_time
        
        if U_hybrid is not None and M_hybrid is not None:
            # Mass analysis
            mass_evolution = np.sum(M_hybrid * problem.Dx, axis=1)
            initial_mass = mass_evolution[0]
            final_mass = mass_evolution[-1]
            mass_change_percent = ((final_mass - initial_mass) / initial_mass) * 100
            
            # Quality metrics
            negative_densities = np.sum(M_hybrid < -1e-10)
            min_density = np.min(M_hybrid)
            max_control = np.max(np.abs(U_hybrid))
            
            # Particle violations
            violations = 0
            if hasattr(fp_solver, 'M_particles_trajectory') and fp_solver.M_particles_trajectory is not None:
                final_particles = fp_solver.M_particles_trajectory[-1, :]
                violations = np.sum(
                    (final_particles < problem.xmin - 1e-10) | 
                    (final_particles > problem.xmax + 1e-10)
                )
            
            return {
                'success': True,
                'method': 'Hybrid',
                'label': test_label,
                'solve_time': solve_time,
                'total_time': total_time,
                'iterations': iters,
                'mass_change_percent': mass_change_percent,
                'final_mass': final_mass,
                'mass_evolution': mass_evolution,
                'negative_densities': negative_densities,
                'min_density': min_density,
                'max_control': max_control,
                'boundary_violations': violations,
                'U': U_hybrid,
                'M': M_hybrid,
                'converged': abs(mass_change_percent) < 5.0 and violations == 0
            }
        else:
            return {'success': False, 'method': 'Hybrid', 'label': test_label, 'error': 'No solution'}
            
    except Exception as e:
        return {'success': False, 'method': 'Hybrid', 'label': test_label, 'error': str(e)}

def test_qp_collocation_method(problem_params, solver_params, test_label):
    """Test QP-Collocation method"""
    try:
        start_time = time.time()
        
        # Create problem
        problem = ExampleMFGProblem(**problem_params)
        no_flux_bc = BoundaryConditions(type="no_flux")
        
        # Setup collocation points
        num_collocation_points = 12
        collocation_points = np.linspace(
            problem.xmin, problem.xmax, num_collocation_points
        ).reshape(-1, 1)
        
        # Identify boundary points
        boundary_tolerance = 1e-10
        boundary_indices = []
        for i, point in enumerate(collocation_points):
            x = point[0]
            if (abs(x - problem.xmin) < boundary_tolerance or 
                abs(x - problem.xmax) < boundary_tolerance):
                boundary_indices.append(i)
        boundary_indices = np.array(boundary_indices)
        
        # QP solver
        qp_solver = ParticleCollocationSolver(
            problem=problem,
            collocation_points=collocation_points,
            num_particles=solver_params["num_particles"],
            delta=0.35,
            taylor_order=2,
            weight_function="wendland",
            NiterNewton=solver_params["newton_iterations"],
            l2errBoundNewton=solver_params["newton_tolerance"],
            kde_bandwidth="scott",
            normalize_kde_output=False,
            boundary_indices=boundary_indices,
            boundary_conditions=no_flux_bc,
            use_monotone_constraints=True
        )
        
        # Solve
        solve_start = time.time()
        U_qp, M_qp, solve_info = qp_solver.solve(
            Niter=solver_params["max_iterations"],
            l2errBound=solver_params["convergence_tolerance"],
            verbose=False
        )
        solve_time = time.time() - solve_start
        total_time = time.time() - start_time
        
        if U_qp is not None and M_qp is not None:
            # Mass analysis
            mass_evolution = np.sum(M_qp * problem.Dx, axis=1)
            initial_mass = mass_evolution[0]
            final_mass = mass_evolution[-1]
            mass_change_percent = ((final_mass - initial_mass) / initial_mass) * 100
            
            # Quality metrics
            negative_densities = np.sum(M_qp < -1e-10)
            min_density = np.min(M_qp)
            max_control = np.max(np.abs(U_qp))
            
            # Particle violations
            violations = 0
            if hasattr(qp_solver.fp_solver, 'M_particles_trajectory') and qp_solver.fp_solver.M_particles_trajectory is not None:
                final_particles = qp_solver.fp_solver.M_particles_trajectory[-1, :]
                violations = np.sum(
                    (final_particles < problem.xmin - 1e-10) | 
                    (final_particles > problem.xmax + 1e-10)
                )
            
            iterations = solve_info.get("iterations", 0)
            converged_flag = solve_info.get("converged", False)
            
            return {
                'success': True,
                'method': 'QP-Collocation',
                'label': test_label,
                'solve_time': solve_time,
                'total_time': total_time,
                'iterations': iterations,
                'mass_change_percent': mass_change_percent,
                'final_mass': final_mass,
                'mass_evolution': mass_evolution,
                'negative_densities': negative_densities,
                'min_density': min_density,
                'max_control': max_control,
                'boundary_violations': violations,
                'U': U_qp,
                'M': M_qp,
                'converged': converged_flag and abs(mass_change_percent) < 5.0 and violations == 0
            }
        else:
            return {'success': False, 'method': 'QP-Collocation', 'label': test_label, 'error': 'No solution'}
            
    except Exception as e:
        return {'success': False, 'method': 'QP-Collocation', 'label': test_label, 'error': str(e)}

def run_comprehensive_evaluation():
    """Run comprehensive evaluation across multiple test scenarios"""
    print("="*80)
    print("COMPREHENSIVE THREE-METHOD EVALUATION")
    print("="*80)
    print("Evaluating: Pure FDM vs Hybrid vs QP-Collocation")
    print("Metrics: Mass conservation, computational cost, stability success rate")
    
    # Define test scenarios with varying difficulty
    test_scenarios = [
        # Easy scenarios - should all succeed
        {
            'name': 'Easy_Short',
            'params': {'xmin': 0.0, 'xmax': 1.0, 'Nx': 20, 'T': 0.5, 'Nt': 25, 'sigma': 0.1, 'coefCT': 0.01},
            'solver': {'max_iterations': 10, 'convergence_tolerance': 1e-3, 'newton_iterations': 6, 'newton_tolerance': 1e-4, 'num_particles': 200, 'thetaUM': 0.5}
        },
        {
            'name': 'Easy_Medium',
            'params': {'xmin': 0.0, 'xmax': 1.0, 'Nx': 25, 'T': 1.0, 'Nt': 50, 'sigma': 0.15, 'coefCT': 0.02},
            'solver': {'max_iterations': 12, 'convergence_tolerance': 1e-3, 'newton_iterations': 6, 'newton_tolerance': 1e-4, 'num_particles': 300, 'thetaUM': 0.5}
        },
        
        # Moderate scenarios - some may struggle
        {
            'name': 'Moderate_Resolution',
            'params': {'xmin': 0.0, 'xmax': 1.0, 'Nx': 40, 'T': 1.0, 'Nt': 80, 'sigma': 0.2, 'coefCT': 0.03},
            'solver': {'max_iterations': 15, 'convergence_tolerance': 1e-3, 'newton_iterations': 8, 'newton_tolerance': 1e-4, 'num_particles': 400, 'thetaUM': 0.4}
        },
        {
            'name': 'Moderate_Time',
            'params': {'xmin': 0.0, 'xmax': 1.0, 'Nx': 30, 'T': 2.0, 'Nt': 100, 'sigma': 0.15, 'coefCT': 0.02},
            'solver': {'max_iterations': 15, 'convergence_tolerance': 1e-3, 'newton_iterations': 8, 'newton_tolerance': 1e-4, 'num_particles': 500, 'thetaUM': 0.4}
        },
        
        # Challenging scenarios - likely to cause issues
        {
            'name': 'Challenge_HighDiffusion',
            'params': {'xmin': 0.0, 'xmax': 1.0, 'Nx': 30, 'T': 1.5, 'Nt': 75, 'sigma': 0.3, 'coefCT': 0.05},
            'solver': {'max_iterations': 15, 'convergence_tolerance': 1e-3, 'newton_iterations': 10, 'newton_tolerance': 1e-5, 'num_particles': 600, 'thetaUM': 0.3}
        },
        {
            'name': 'Challenge_StrongCoupling',
            'params': {'xmin': 0.0, 'xmax': 1.0, 'Nx': 35, 'T': 1.0, 'Nt': 70, 'sigma': 0.2, 'coefCT': 0.08},
            'solver': {'max_iterations': 15, 'convergence_tolerance': 1e-3, 'newton_iterations': 10, 'newton_tolerance': 1e-5, 'num_particles': 700, 'thetaUM': 0.3}
        },
        
        # Extreme scenarios - expected to fail some methods
        {
            'name': 'Extreme_HighRes',
            'params': {'xmin': 0.0, 'xmax': 1.0, 'Nx': 60, 'T': 1.0, 'Nt': 120, 'sigma': 0.25, 'coefCT': 0.04},
            'solver': {'max_iterations': 20, 'convergence_tolerance': 1e-3, 'newton_iterations': 12, 'newton_tolerance': 1e-5, 'num_particles': 800, 'thetaUM': 0.3}
        },
        {
            'name': 'Extreme_LongTime',
            'params': {'xmin': 0.0, 'xmax': 1.0, 'Nx': 30, 'T': 3.0, 'Nt': 150, 'sigma': 0.2, 'coefCT': 0.03},
            'solver': {'max_iterations': 20, 'convergence_tolerance': 1e-3, 'newton_iterations': 12, 'newton_tolerance': 1e-5, 'num_particles': 600, 'thetaUM': 0.25}
        }
    ]
    
    results = []
    
    print(f"\nTesting {len(test_scenarios)} scenarios across 3 methods = {len(test_scenarios) * 3} total tests")
    
    for i, scenario in enumerate(test_scenarios):
        print(f"\n{'='*60}")
        print(f"SCENARIO {i+1}/{len(test_scenarios)}: {scenario['name']}")
        print(f"{'='*60}")
        
        # Print scenario parameters
        params = scenario['params']
        solver = scenario['solver']
        print(f"Problem: Nx={params['Nx']}, T={params['T']}, sigma={params['sigma']}, coefCT={params['coefCT']}")
        print(f"Solver: particles={solver['num_particles']}, thetaUM={solver['thetaUM']}")
        
        # Test each method
        methods = [
            ('Pure FDM', test_pure_fdm_method),
            ('Hybrid', test_hybrid_method),
            ('QP-Collocation', test_qp_collocation_method)
        ]
        
        for method_name, test_func in methods:
            print(f"\n--- Testing {method_name} ---")
            
            try:
                result = test_func(params, solver, f"{scenario['name']}_{method_name}")
                results.append(result)
                
                if result['success']:
                    print(f"✓ Success: {result['solve_time']:.2f}s, {result['iterations']} iters")
                    print(f"  Mass change: {result['mass_change_percent']:+.3f}%")
                    print(f"  Converged: {result['converged']}")
                    if 'boundary_violations' in result:
                        print(f"  Violations: {result['boundary_violations']}")
                else:
                    print(f"❌ Failed: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"❌ Crashed: {e}")
                results.append({
                    'success': False, 
                    'method': method_name, 
                    'label': f"{scenario['name']}_{method_name}",
                    'error': str(e)
                })
    
    # Analysis and visualization
    print(f"\n{'='*80}")
    print("EVALUATION ANALYSIS")
    print(f"{'='*80}")
    
    analyze_comprehensive_results(results)
    create_comprehensive_visualization(results, test_scenarios)
    
    return results

def analyze_comprehensive_results(results):
    """Analyze comprehensive evaluation results"""
    
    # Separate by method
    methods = ['Pure FDM', 'Hybrid', 'QP-Collocation']
    method_results = {method: [] for method in methods}
    
    for result in results:
        method = result.get('method', 'Unknown')
        if method in method_results:
            method_results[method].append(result)
    
    print("--- SUCCESS RATES ---")
    for method in methods:
        total = len(method_results[method])
        successful = len([r for r in method_results[method] if r.get('success', False)])
        converged = len([r for r in method_results[method] if r.get('converged', False)])
        
        success_rate = (successful / total) * 100 if total > 0 else 0
        convergence_rate = (converged / total) * 100 if total > 0 else 0
        
        print(f"{method:<15}: {successful}/{total} success ({success_rate:.1f}%), {converged}/{total} converged ({convergence_rate:.1f}%)")
    
    print("\n--- MASS CONSERVATION QUALITY ---")
    for method in methods:
        successful_results = [r for r in method_results[method] if r.get('success', False)]
        if successful_results:
            mass_changes = [abs(r['mass_change_percent']) for r in successful_results]
            avg_mass_error = np.mean(mass_changes)
            max_mass_error = np.max(mass_changes)
            excellent = len([r for r in successful_results if abs(r['mass_change_percent']) < 1.0])
            good = len([r for r in successful_results if 1.0 <= abs(r['mass_change_percent']) < 5.0])
            poor = len([r for r in successful_results if abs(r['mass_change_percent']) >= 5.0])
            
            print(f"{method:<15}: Avg {avg_mass_error:.2f}%, Max {max_mass_error:.2f}%")
            print(f"{'':15}  Excellent (<1%): {excellent}, Good (1-5%): {good}, Poor (≥5%): {poor}")
    
    print("\n--- COMPUTATIONAL COST ---")
    for method in methods:
        successful_results = [r for r in method_results[method] if r.get('success', False)]
        if successful_results:
            solve_times = [r['solve_time'] for r in successful_results]
            avg_time = np.mean(solve_times)
            median_time = np.median(solve_times)
            max_time = np.max(solve_times)
            
            print(f"{method:<15}: Avg {avg_time:.2f}s, Median {median_time:.2f}s, Max {max_time:.2f}s")

def create_comprehensive_visualization(results, scenarios):
    """Create comprehensive visualization of evaluation results"""
    
    # Prepare data
    methods = ['Pure FDM', 'Hybrid', 'QP-Collocation']
    method_colors = {'Pure FDM': 'blue', 'Hybrid': 'green', 'QP-Collocation': 'red'}
    
    # Organize data by method
    method_data = {method: {'success': [], 'mass_error': [], 'time': [], 'converged': []} for method in methods}
    
    for result in results:
        method = result.get('method', 'Unknown')
        if method in method_data:
            method_data[method]['success'].append(result.get('success', False))
            if result.get('success', False):
                method_data[method]['mass_error'].append(abs(result.get('mass_change_percent', 100)))
                method_data[method]['time'].append(result.get('solve_time', 0))
                method_data[method]['converged'].append(result.get('converged', False))
            else:
                method_data[method]['mass_error'].append(100)  # Failed case
                method_data[method]['time'].append(0)
                method_data[method]['converged'].append(False)
    
    # Create comprehensive plot
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Success Rate Comparison
    ax1 = plt.subplot(2, 3, 1)
    success_rates = []
    convergence_rates = []
    method_names = []
    
    for method in methods:
        total = len(method_data[method]['success'])
        successful = sum(method_data[method]['success'])
        converged = sum(method_data[method]['converged'])
        
        success_rate = (successful / total) * 100 if total > 0 else 0
        convergence_rate = (converged / total) * 100 if total > 0 else 0
        
        success_rates.append(success_rate)
        convergence_rates.append(convergence_rate)
        method_names.append(method.replace(' ', '\n'))
    
    x_pos = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax1.bar(x_pos - width/2, success_rates, width, label='Success Rate', 
                    color=[method_colors[m] for m in methods], alpha=0.7)
    bars2 = ax1.bar(x_pos + width/2, convergence_rates, width, label='Convergence Rate',
                    color=[method_colors[m] for m in methods], alpha=0.5)
    
    ax1.set_xlabel('Method')
    ax1.set_ylabel('Success Rate (%)')
    ax1.set_title('Success and Convergence Rates')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(method_names)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 105])
    
    # Add percentage labels
    for bar, value in zip(bars1, success_rates):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{value:.1f}%', ha='center', va='bottom', fontsize=9)
    for bar, value in zip(bars2, convergence_rates):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{value:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 2. Mass Conservation Quality
    ax2 = plt.subplot(2, 3, 2)
    mass_error_data = []
    method_labels = []
    colors = []
    
    for method in methods:
        errors = [e for e in method_data[method]['mass_error'] if e < 50]  # Filter out extreme failures
        if errors:
            mass_error_data.append(errors)
            method_labels.append(method.replace(' ', '\n'))
            colors.append(method_colors[method])
    
    if mass_error_data:
        bp = ax2.boxplot(mass_error_data, labels=method_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    ax2.set_ylabel('Mass Conservation Error (%)')
    ax2.set_title('Mass Conservation Quality')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # Add horizontal lines for quality thresholds
    ax2.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Excellent (<1%)')
    ax2.axhline(y=5.0, color='orange', linestyle='--', alpha=0.7, label='Good (<5%)')
    ax2.legend(fontsize=8)
    
    # 3. Computational Cost
    ax3 = plt.subplot(2, 3, 3)
    time_data = []
    time_labels = []
    time_colors = []
    
    for method in methods:
        times = [t for t in method_data[method]['time'] if t > 0]  # Filter out failed cases
        if times:
            time_data.append(times)
            time_labels.append(method.replace(' ', '\n'))
            time_colors.append(method_colors[method])
    
    if time_data:
        bp = ax3.boxplot(time_data, labels=time_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], time_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    ax3.set_ylabel('Solve Time (seconds)')
    ax3.set_title('Computational Cost')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # 4. Mass Conservation vs Computational Cost Scatter
    ax4 = plt.subplot(2, 3, 4)
    
    for method in methods:
        successful_indices = [i for i, success in enumerate(method_data[method]['success']) if success]
        if successful_indices:
            mass_errors = [method_data[method]['mass_error'][i] for i in successful_indices if method_data[method]['mass_error'][i] < 50]
            times = [method_data[method]['time'][i] for i in successful_indices if method_data[method]['mass_error'][i] < 50]
            
            if mass_errors and times:
                ax4.scatter(times, mass_errors, label=method, color=method_colors[method], 
                           alpha=0.7, s=60)
    
    ax4.set_xlabel('Solve Time (seconds)')
    ax4.set_ylabel('Mass Conservation Error (%)')
    ax4.set_title('Mass Conservation vs Computational Cost')
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # Add quality regions
    ax4.axhline(y=1.0, color='green', linestyle='--', alpha=0.3)
    ax4.axhline(y=5.0, color='orange', linestyle='--', alpha=0.3)
    ax4.text(0.02, 0.95, 'Excellent\n(<1%)', transform=ax4.transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.5), fontsize=8)
    ax4.text(0.02, 0.75, 'Good\n(1-5%)', transform=ax4.transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.5), fontsize=8)
    
    # 5. Stability Success Rate by Scenario Difficulty
    ax5 = plt.subplot(2, 3, 5)
    
    # Group scenarios by difficulty
    difficulty_groups = {
        'Easy': ['Easy_Short', 'Easy_Medium'],
        'Moderate': ['Moderate_Resolution', 'Moderate_Time'],
        'Challenge': ['Challenge_HighDiffusion', 'Challenge_StrongCoupling'],
        'Extreme': ['Extreme_HighRes', 'Extreme_LongTime']
    }
    
    difficulty_success = {method: {'Easy': 0, 'Moderate': 0, 'Challenge': 0, 'Extreme': 0} for method in methods}
    difficulty_total = {'Easy': 0, 'Moderate': 0, 'Challenge': 0, 'Extreme': 0}
    
    for result in results:
        method = result.get('method', 'Unknown')
        label = result.get('label', '')
        
        # Find difficulty level
        for difficulty, scenario_names in difficulty_groups.items():
            if any(scenario in label for scenario in scenario_names):
                difficulty_total[difficulty] += 1
                if result.get('success', False) and result.get('converged', False):
                    if method in difficulty_success:
                        difficulty_success[method][difficulty] += 1
                break
    
    # Convert to success rates
    difficulties = list(difficulty_groups.keys())
    x_pos = np.arange(len(difficulties))
    width = 0.25
    
    for i, method in enumerate(methods):
        rates = []
        for diff in difficulties:
            total = difficulty_total[diff] // 3  # 3 methods per scenario
            successful = difficulty_success[method][diff]
            rate = (successful / total * 100) if total > 0 else 0
            rates.append(rate)
        
        ax5.bar(x_pos + i*width, rates, width, label=method, 
                color=method_colors[method], alpha=0.7)
    
    ax5.set_xlabel('Scenario Difficulty')
    ax5.set_ylabel('Success Rate (%)')
    ax5.set_title('Stability by Scenario Difficulty')
    ax5.set_xticks(x_pos + width)
    ax5.set_xticklabels(difficulties)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim([0, 105])
    
    # 6. Overall Performance Radar Chart
    ax6 = plt.subplot(2, 3, 6, projection='polar')
    
    # Performance metrics (normalized to 0-1 scale)
    metrics = ['Success\nRate', 'Mass\nConservation', 'Speed', 'Stability']
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    for method in methods:
        # Calculate normalized scores
        total_tests = len(method_data[method]['success'])
        success_score = sum(method_data[method]['success']) / total_tests if total_tests > 0 else 0
        
        # Mass conservation score (1 - normalized error)
        successful_mass_errors = [e for e in method_data[method]['mass_error'] if e < 50]
        if successful_mass_errors:
            avg_mass_error = np.mean(successful_mass_errors)
            mass_score = max(0, 1 - avg_mass_error / 20)  # Normalize to 0-1
        else:
            mass_score = 0
        
        # Speed score (inverse of time, normalized)
        successful_times = [t for t in method_data[method]['time'] if t > 0]
        if successful_times:
            avg_time = np.mean(successful_times)
            # Normalize by maximum time across all methods
            all_times = []
            for m in methods:
                all_times.extend([t for t in method_data[m]['time'] if t > 0])
            max_time = max(all_times) if all_times else 1
            speed_score = 1 - (avg_time / max_time)
        else:
            speed_score = 0
        
        # Stability score (convergence rate)
        stability_score = sum(method_data[method]['converged']) / total_tests if total_tests > 0 else 0
        
        scores = [success_score, mass_score, speed_score, stability_score]
        scores += scores[:1]  # Complete the circle
        
        ax6.plot(angles, scores, 'o-', linewidth=2, label=method, color=method_colors[method])
        ax6.fill(angles, scores, alpha=0.25, color=method_colors[method])
    
    ax6.set_xticks(angles[:-1])
    ax6.set_xticklabels(metrics)
    ax6.set_ylim(0, 1)
    ax6.set_title('Overall Performance Profile', pad=20)
    ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax6.grid(True)
    
    plt.tight_layout()
    plt.savefig('/Users/zvezda/Library/CloudStorage/OneDrive-Personal/code/MFG_PDE/tests/method_comparisons/comprehensive_three_method_evaluation.png', 
                dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n✅ Comprehensive evaluation visualization saved: comprehensive_three_method_evaluation.png")

if __name__ == "__main__":
    print("Starting Comprehensive Three-Method Evaluation...")
    print("Expected execution time: 15-30 minutes")
    
    try:
        results = run_comprehensive_evaluation()
        print("\n" + "="*80)
        print("COMPREHENSIVE EVALUATION COMPLETED")
        print("="*80)
        print("Check the generated analysis and plots for detailed comparison.")
        
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
    except Exception as e:
        print(f"\nEvaluation failed: {e}")
        import traceback
        traceback.print_exc()