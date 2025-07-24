#!/usr/bin/env python3
"""
Final QP Optimization Test
Comprehensive test of all QP optimization approaches to validate
the deep integration and demonstrate the final optimized performance.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

from mfg_pde.core.mfg_problem import ExampleMFGProblem
from mfg_pde.alg.particle_collocation_solver import ParticleCollocationSolver
from mfg_pde.alg.hjb_solvers.optimized_gfdm_hjb_v2 import OptimizedGFDMHJBSolver
from mfg_pde.alg.hjb_solvers.smart_qp_gfdm_hjb import SmartQPGFDMHJBSolver
from mfg_pde.alg.hjb_solvers.tuned_smart_qp_gfdm_hjb import TunedSmartQPGFDMHJBSolver
from mfg_pde.core.boundaries import BoundaryConditions

def run_comprehensive_qp_optimization_test():
    """Run comprehensive test of all QP optimization approaches"""
    print("="*80)
    print("COMPREHENSIVE QP OPTIMIZATION TEST")
    print("="*80)
    print("Testing: Baseline, Basic Optimized, Smart QP, and Tuned Smart QP approaches")
    
    # Problem parameters
    problem_params = {
        'xmin': 0.0, 'xmax': 1.0, 'Nx': 20, 'T': 1.0, 'Nt': 40,
        'sigma': 0.15, 'coefCT': 0.02
    }
    
    print(f"Problem: Nx={problem_params['Nx']}, T={problem_params['T']}, Nt={problem_params['Nt']}")
    print(f"Parameters: σ={problem_params['sigma']}, coefCT={problem_params['coefCT']}")
    
    results = {}
    
    # Common solver parameters
    num_collocation_points = 10
    solver_params = {
        'num_particles': 150,
        'delta': 0.4,
        'taylor_order': 2,
        'newton_iterations': 4,
        'newton_tolerance': 1e-3,
        'max_picard_iterations': 6,
        'convergence_tolerance': 1e-3
    }
    
    # Test 1: Baseline (No Optimization)
    print(f"\n{'-'*60}")
    print("TESTING BASELINE QP-COLLOCATION (NO OPTIMIZATION)")
    print(f"{'-'*60}")
    
    try:
        problem = ExampleMFGProblem(**problem_params)
        no_flux_bc = BoundaryConditions(type="no_flux")
        
        collocation_points = np.linspace(
            problem.xmin, problem.xmax, num_collocation_points
        ).reshape(-1, 1)
        boundary_indices = [0, num_collocation_points-1]
        
        # Create baseline solver (standard GFDM with QP)
        baseline_solver = ParticleCollocationSolver(
            problem=problem,
            collocation_points=collocation_points,
            num_particles=solver_params['num_particles'],
            delta=solver_params['delta'],
            taylor_order=solver_params['taylor_order'],
            weight_function="wendland",
            NiterNewton=solver_params['newton_iterations'],
            l2errBoundNewton=solver_params['newton_tolerance'],
            kde_bandwidth="scott",
            normalize_kde_output=False,
            boundary_indices=np.array(boundary_indices),
            boundary_conditions=no_flux_bc,
            use_monotone_constraints=True
        )
        
        print("Running Baseline QP-Collocation solver...")
        start_time = time.time()
        U_baseline, M_baseline, info_baseline = baseline_solver.solve(
            Niter=solver_params['max_picard_iterations'], 
            l2errBound=solver_params['convergence_tolerance'], 
            verbose=True
        )
        baseline_time = time.time() - start_time
        
        # Calculate mass conservation
        Dx = (problem_params['xmax'] - problem_params['xmin']) / problem_params['Nx']
        initial_mass = np.sum(M_baseline[0, :]) * Dx
        final_mass = np.sum(M_baseline[-1, :]) * Dx
        mass_error = abs(final_mass - initial_mass) / initial_mass * 100
        
        results['baseline'] = {
            'method': 'Baseline QP-Collocation',
            'success': True,
            'time': baseline_time,
            'mass_error': mass_error,
            'converged': info_baseline.get('converged', False),
            'iterations': info_baseline.get('iterations', 0),
            'qp_usage_rate': 1.0,  # Assume 100% for baseline
            'U': U_baseline,
            'M': M_baseline
        }
        
        print(f"✓ Baseline completed: {baseline_time:.1f}s, mass error: {mass_error:.2f}%")
        
    except Exception as e:
        print(f"✗ Baseline failed: {e}")
        results['baseline'] = {'method': 'Baseline QP-Collocation', 'success': False, 'error': str(e)}
    
    # Test 2: Basic Optimized QP
    print(f"\n{'-'*60}")
    print("TESTING BASIC OPTIMIZED QP-COLLOCATION")
    print(f"{'-'*60}")
    
    try:
        problem = ExampleMFGProblem(**problem_params)
        no_flux_bc = BoundaryConditions(type="no_flux")
        
        collocation_points = np.linspace(
            problem.xmin, problem.xmax, num_collocation_points
        ).reshape(-1, 1)
        boundary_indices = [0, num_collocation_points-1]
        
        basic_hjb_solver = OptimizedGFDMHJBSolver(
            problem=problem,
            collocation_points=collocation_points,
            delta=solver_params['delta'],
            taylor_order=solver_params['taylor_order'],
            weight_function="wendland",
            NiterNewton=solver_params['newton_iterations'],
            l2errBoundNewton=solver_params['newton_tolerance'],
            boundary_indices=np.array(boundary_indices),
            boundary_conditions=no_flux_bc,
            use_monotone_constraints=True,
            qp_activation_tolerance=1e-3
        )
        
        basic_solver = ParticleCollocationSolver(
            problem=problem,
            collocation_points=collocation_points,
            num_particles=solver_params['num_particles'],
            delta=solver_params['delta'],
            taylor_order=solver_params['taylor_order'],
            weight_function="wendland",
            NiterNewton=solver_params['newton_iterations'],
            l2errBoundNewton=solver_params['newton_tolerance'],
            kde_bandwidth="scott",
            normalize_kde_output=False,
            boundary_indices=np.array(boundary_indices),
            boundary_conditions=no_flux_bc,
            use_monotone_constraints=True
        )
        basic_solver.hjb_solver = basic_hjb_solver
        
        print("Running Basic Optimized QP-Collocation solver...")
        start_time = time.time()
        U_basic, M_basic, info_basic = basic_solver.solve(
            Niter=solver_params['max_picard_iterations'], 
            l2errBound=solver_params['convergence_tolerance'], 
            verbose=True
        )
        basic_time = time.time() - start_time
        
        # Calculate mass conservation
        initial_mass = np.sum(M_basic[0, :]) * Dx
        final_mass = np.sum(M_basic[-1, :]) * Dx
        mass_error = abs(final_mass - initial_mass) / initial_mass * 100
        
        # Get optimization statistics
        basic_stats = {}
        if hasattr(basic_hjb_solver, 'get_performance_report'):
            basic_stats = basic_hjb_solver.get_performance_report()
        
        results['basic'] = {
            'method': 'Basic Optimized QP-Collocation',
            'success': True,
            'time': basic_time,
            'mass_error': mass_error,
            'converged': info_basic.get('converged', False),
            'iterations': info_basic.get('iterations', 0),
            'qp_usage_rate': basic_stats.get('qp_activation_rate', 1.0),
            'optimization_stats': basic_stats,
            'U': U_basic,
            'M': M_basic
        }
        
        print(f"✓ Basic Optimized completed: {basic_time:.1f}s, mass error: {mass_error:.2f}%")
        if basic_stats:
            print(f"  QP Usage Rate: {basic_stats.get('qp_activation_rate', 0):.1%}")
        
    except Exception as e:
        print(f"✗ Basic Optimized failed: {e}")
        results['basic'] = {'method': 'Basic Optimized QP-Collocation', 'success': False, 'error': str(e)}
    
    # Test 3: Smart QP
    print(f"\n{'-'*60}")
    print("TESTING SMART QP-COLLOCATION")
    print(f"{'-'*60}")
    
    try:
        problem = ExampleMFGProblem(**problem_params)
        no_flux_bc = BoundaryConditions(type="no_flux")
        
        collocation_points = np.linspace(
            problem.xmin, problem.xmax, num_collocation_points
        ).reshape(-1, 1)
        boundary_indices = [0, num_collocation_points-1]
        
        smart_hjb_solver = SmartQPGFDMHJBSolver(
            problem=problem,
            collocation_points=collocation_points,
            delta=solver_params['delta'],
            taylor_order=solver_params['taylor_order'],
            weight_function="wendland",
            NiterNewton=solver_params['newton_iterations'],
            l2errBoundNewton=solver_params['newton_tolerance'],
            boundary_indices=np.array(boundary_indices),
            boundary_conditions=no_flux_bc,
            use_monotone_constraints=True,
            qp_usage_target=0.1
        )
        
        smart_solver = ParticleCollocationSolver(
            problem=problem,
            collocation_points=collocation_points,
            num_particles=solver_params['num_particles'],
            delta=solver_params['delta'],
            taylor_order=solver_params['taylor_order'],
            weight_function="wendland",
            NiterNewton=solver_params['newton_iterations'],
            l2errBoundNewton=solver_params['newton_tolerance'],
            kde_bandwidth="scott",
            normalize_kde_output=False,
            boundary_indices=np.array(boundary_indices),
            boundary_conditions=no_flux_bc,
            use_monotone_constraints=True
        )
        smart_solver.hjb_solver = smart_hjb_solver
        
        print("Running Smart QP-Collocation solver...")
        start_time = time.time()
        U_smart, M_smart, info_smart = smart_solver.solve(
            Niter=solver_params['max_picard_iterations'], 
            l2errBound=solver_params['convergence_tolerance'], 
            verbose=True
        )
        smart_time = time.time() - start_time
        
        # Calculate mass conservation
        initial_mass = np.sum(M_smart[0, :]) * Dx
        final_mass = np.sum(M_smart[-1, :]) * Dx
        mass_error = abs(final_mass - initial_mass) / initial_mass * 100
        
        # Get smart optimization statistics
        smart_stats = {}
        if hasattr(smart_hjb_solver, 'get_smart_qp_report'):
            smart_stats = smart_hjb_solver.get_smart_qp_report()
        
        results['smart'] = {
            'method': 'Smart QP-Collocation',
            'success': True,
            'time': smart_time,
            'mass_error': mass_error,
            'converged': info_smart.get('converged', False),
            'iterations': info_smart.get('iterations', 0),
            'qp_usage_rate': smart_stats.get('qp_usage_rate', 1.0),
            'smart_stats': smart_stats,
            'U': U_smart,
            'M': M_smart
        }
        
        print(f"✓ Smart QP completed: {smart_time:.1f}s, mass error: {mass_error:.2f}%")
        if smart_stats:
            print(f"  QP Usage Rate: {smart_stats.get('qp_usage_rate', 0):.1%}")
        
    except Exception as e:
        print(f"✗ Smart QP failed: {e}")
        results['smart'] = {'method': 'Smart QP-Collocation', 'success': False, 'error': str(e)}
    
    # Test 4: Tuned Smart QP
    print(f"\n{'-'*60}")
    print("TESTING TUNED SMART QP-COLLOCATION")
    print(f"{'-'*60}")
    
    try:
        problem = ExampleMFGProblem(**problem_params)
        no_flux_bc = BoundaryConditions(type="no_flux")
        
        collocation_points = np.linspace(
            problem.xmin, problem.xmax, num_collocation_points
        ).reshape(-1, 1)
        boundary_indices = [0, num_collocation_points-1]
        
        tuned_hjb_solver = TunedSmartQPGFDMHJBSolver(
            problem=problem,
            collocation_points=collocation_points,
            delta=solver_params['delta'],
            taylor_order=solver_params['taylor_order'],
            weight_function="wendland",
            NiterNewton=solver_params['newton_iterations'],
            l2errBoundNewton=solver_params['newton_tolerance'],
            boundary_indices=np.array(boundary_indices),
            boundary_conditions=no_flux_bc,
            use_monotone_constraints=True,
            qp_usage_target=0.1
        )
        
        tuned_solver = ParticleCollocationSolver(
            problem=problem,
            collocation_points=collocation_points,
            num_particles=solver_params['num_particles'],
            delta=solver_params['delta'],
            taylor_order=solver_params['taylor_order'],
            weight_function="wendland",
            NiterNewton=solver_params['newton_iterations'],
            l2errBoundNewton=solver_params['newton_tolerance'],
            kde_bandwidth="scott",
            normalize_kde_output=False,
            boundary_indices=np.array(boundary_indices),
            boundary_conditions=no_flux_bc,
            use_monotone_constraints=True
        )
        tuned_solver.hjb_solver = tuned_hjb_solver
        
        print("Running Tuned Smart QP-Collocation solver...")
        start_time = time.time()
        U_tuned, M_tuned, info_tuned = tuned_solver.solve(
            Niter=solver_params['max_picard_iterations'], 
            l2errBound=solver_params['convergence_tolerance'], 
            verbose=True
        )
        tuned_time = time.time() - start_time
        
        # Calculate mass conservation
        initial_mass = np.sum(M_tuned[0, :]) * Dx
        final_mass = np.sum(M_tuned[-1, :]) * Dx
        mass_error = abs(final_mass - initial_mass) / initial_mass * 100
        
        # Get tuned optimization statistics
        tuned_stats = {}
        if hasattr(tuned_hjb_solver, 'get_tuned_qp_report'):
            tuned_stats = tuned_hjb_solver.get_tuned_qp_report()
        
        results['tuned'] = {
            'method': 'Tuned Smart QP-Collocation',
            'success': True,
            'time': tuned_time,
            'mass_error': mass_error,
            'converged': info_tuned.get('converged', False),
            'iterations': info_tuned.get('iterations', 0),
            'qp_usage_rate': tuned_stats.get('qp_usage_rate', 1.0),
            'tuned_stats': tuned_stats,
            'U': U_tuned,
            'M': M_tuned
        }
        
        print(f"✓ Tuned Smart QP completed: {tuned_time:.1f}s, mass error: {mass_error:.2f}%")
        if tuned_stats:
            print(f"  QP Usage Rate: {tuned_stats.get('qp_usage_rate', 0):.1%}")
            print(f"  Optimization Quality: {tuned_stats.get('optimization_quality', 'N/A')}")
        
        # Print detailed tuned summary
        if hasattr(tuned_hjb_solver, 'print_tuned_qp_summary'):
            tuned_hjb_solver.print_tuned_qp_summary()
        
    except Exception as e:
        print(f"✗ Tuned Smart QP failed: {e}")
        import traceback
        traceback.print_exc()
        results['tuned'] = {'method': 'Tuned Smart QP-Collocation', 'success': False, 'error': str(e)}
    
    # Print comprehensive summary
    print_comprehensive_summary(results)
    
    # Create comprehensive plots
    create_comprehensive_plots(results, problem_params)
    
    return results

def print_comprehensive_summary(results):
    """Print comprehensive comparison summary"""
    print(f"\n{'='*100}")
    print("COMPREHENSIVE QP OPTIMIZATION COMPARISON SUMMARY")
    print(f"{'='*100}")
    
    print(f"\n{'Method':<35} {'Success':<8} {'Time(s)':<10} {'Mass Err %':<12} {'QP Usage':<12} {'Speedup':<10} {'Status':<15}")
    print("-" * 110)
    
    # Get baseline time for speedup calculation
    baseline_time = None
    if 'baseline' in results and results['baseline']['success']:
        baseline_time = results['baseline']['time']
    
    for key, result in results.items():
        if result['success']:
            success_str = "✓"
            time_str = f"{result['time']:.1f}"
            mass_str = f"{result['mass_error']:.2f}"
            qp_usage_str = f"{result['qp_usage_rate']:.1%}"
            
            # Calculate speedup
            if baseline_time and baseline_time > 0:
                speedup = baseline_time / result['time']
                speedup_str = f"{speedup:.2f}x"
            else:
                speedup_str = "1.00x"
            
            # Status assessment
            qp_rate = result['qp_usage_rate']
            if qp_rate <= 0.15:  # Close to 10% target
                status = "EXCELLENT"
            elif qp_rate <= 0.3:  # Significant improvement
                status = "GOOD"
            elif qp_rate <= 0.5:  # Some improvement
                status = "FAIR"
            elif qp_rate < 1.0:  # Basic improvement
                status = "BASIC"
            else:
                status = "BASELINE"
        else:
            success_str = "✗"
            time_str = "FAILED"
            mass_str = "N/A"
            qp_usage_str = "N/A"
            speedup_str = "N/A"
            status = "FAILED"
        
        print(f"{result['method']:<35} {success_str:<8} {time_str:<10} {mass_str:<12} {qp_usage_str:<12} {speedup_str:<10} {status:<15}")
    
    # Performance analysis
    successful_results = {k: v for k, v in results.items() if v['success']}
    
    if len(successful_results) > 1:
        print(f"\nPERFORMANCE ANALYSIS:")
        print("-" * 60)
        
        # Find best performers
        fastest = min(successful_results.items(), key=lambda x: x[1]['time'])
        most_accurate = min(successful_results.items(), key=lambda x: x[1]['mass_error'])
        lowest_qp_usage = min(successful_results.items(), key=lambda x: x[1]['qp_usage_rate'])
        
        print(f"Fastest Method: {fastest[1]['method']} ({fastest[1]['time']:.1f}s)")
        print(f"Best Mass Conservation: {most_accurate[1]['method']} ({most_accurate[1]['mass_error']:.2f}% error)")
        print(f"Lowest QP Usage: {lowest_qp_usage[1]['method']} ({lowest_qp_usage[1]['qp_usage_rate']:.1%})")
        
        # QP optimization progression
        print(f"\nQP OPTIMIZATION PROGRESSION:")
        print("-" * 40)
        
        for key, result in successful_results.items():
            qp_rate = result['qp_usage_rate']
            if baseline_time:
                speedup = baseline_time / result['time']
                qp_reduction = (1.0 - qp_rate) * 100
                print(f"{result['method']}: {qp_rate:.1%} usage, {qp_reduction:.1f}% reduction, {speedup:.2f}x speedup")
        
        # Overall assessment
        print(f"\nOVERALL ASSESSMENT:")
        print("-" * 30)
        
        best_optimized = None
        best_qp_rate = 1.0
        
        for key, result in successful_results.items():
            if result['qp_usage_rate'] < best_qp_rate and key != 'baseline':
                best_qp_rate = result['qp_usage_rate']
                best_optimized = result
        
        if best_optimized:
            if best_qp_rate <= 0.12:  # Within 20% of 10% target
                print("✓ OPTIMIZATION TARGET ACHIEVED")
                print(f"  Best optimization: {best_optimized['method']} with {best_qp_rate:.1%} QP usage")
            elif best_qp_rate <= 0.2:  # Within 100% of target
                print("⚠️ CLOSE TO OPTIMIZATION TARGET")
                print(f"  Best optimization: {best_optimized['method']} with {best_qp_rate:.1%} QP usage")
            else:
                print("❌ OPTIMIZATION TARGET NOT REACHED")
                print(f"  Best optimization: {best_optimized['method']} with {best_qp_rate:.1%} QP usage")
                print("  Further tuning needed to reach 10% target")

def create_comprehensive_plots(results, problem_params):
    """Create comprehensive comparison plots"""
    successful_results = {k: v for k, v in results.items() if v['success']}
    
    if len(successful_results) == 0:
        print("No successful results to plot")
        return
    
    fig = plt.figure(figsize=(20, 16))
    
    # Plot 1: QP Usage Rate Evolution
    ax1 = plt.subplot(3, 3, 1)
    methods = [r['method'] for r in successful_results.values()]
    qp_rates = [r['qp_usage_rate'] * 100 for r in successful_results.values()]
    colors = ['red', 'orange', 'blue', 'green'][:len(methods)]
    
    bars1 = ax1.bar(methods, qp_rates, color=colors, alpha=0.7)
    ax1.axhline(y=10, color='black', linestyle='--', linewidth=2, label='Target: 10%')
    ax1.set_ylabel('QP Usage Rate (%)')
    ax1.set_title('QP Usage Rate Comparison')
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, rate in zip(bars1, qp_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{rate:.1f}%', ha='center', va='bottom')
    
    # Plot 2: Solve Time Comparison
    ax2 = plt.subplot(3, 3, 2)
    times = [r['time'] for r in successful_results.values()]
    
    bars2 = ax2.bar(methods, times, color=colors, alpha=0.7)
    ax2.set_ylabel('Solve Time (seconds)')
    ax2.set_title('Computational Time Comparison')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, time_val in zip(bars2, times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{time_val:.1f}s', ha='center', va='bottom')
    
    # Plot 3: Speedup vs Baseline
    ax3 = plt.subplot(3, 3, 3)
    baseline_time = successful_results.get('baseline', {}).get('time', None)
    
    if baseline_time:
        speedups = [baseline_time / r['time'] for r in successful_results.values()]
        bars3 = ax3.bar(methods, speedups, color=colors, alpha=0.7)
        ax3.set_ylabel('Speedup Factor')
        ax3.set_title('Speedup vs Baseline')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, speedup in zip(bars3, speedups):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{speedup:.2f}x', ha='center', va='bottom')
    
    # Plot 4: Mass Conservation Quality
    ax4 = plt.subplot(3, 3, 4)
    mass_errors = [r['mass_error'] for r in successful_results.values()]
    
    bars4 = ax4.bar(methods, mass_errors, color=colors, alpha=0.7)
    ax4.set_ylabel('Mass Conservation Error (%)')
    ax4.set_title('Solution Quality Comparison')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    # Plot 5: QP Usage vs Performance Trade-off
    ax5 = plt.subplot(3, 3, 5)
    
    if len(successful_results) > 1:
        qp_usage_rates = [r['qp_usage_rate'] * 100 for r in successful_results.values()]
        solve_times = [r['time'] for r in successful_results.values()]
        
        for i, (method, qp_rate, solve_time) in enumerate(zip(methods, qp_usage_rates, solve_times)):
            ax5.scatter(qp_rate, solve_time, color=colors[i], s=100, alpha=0.7, label=method)
            ax5.annotate(method, (qp_rate, solve_time), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8)
        
        ax5.axvline(x=10, color='red', linestyle='--', alpha=0.5, label='Target QP Rate')
        ax5.set_xlabel('QP Usage Rate (%)')
        ax5.set_ylabel('Solve Time (seconds)')
        ax5.set_title('QP Usage vs Performance Trade-off')
        ax5.grid(True, alpha=0.3)
    
    # Plot 6: Solution Profiles Comparison
    ax6 = plt.subplot(3, 3, 6)
    x_grid = np.linspace(problem_params['xmin'], problem_params['xmax'], 
                        successful_results[list(successful_results.keys())[0]]['M'].shape[1])
    
    for i, (key, result) in enumerate(successful_results.items()):
        M = result['M']
        ax6.plot(x_grid, M[-1, :], label=f"{result['method']} (final)", 
                color=colors[i], linewidth=2)
    
    ax6.set_xlabel('x')
    ax6.set_ylabel('Density')
    ax6.set_title('Final Density Profiles')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Optimization Effectiveness Summary
    ax7 = plt.subplot(3, 3, 7)
    ax7.axis('off')
    
    # Create effectiveness summary
    summary_text = "OPTIMIZATION EFFECTIVENESS\n\n"
    
    for key, result in successful_results.items():
        qp_rate = result['qp_usage_rate']
        if baseline_time:
            speedup = baseline_time / result['time']
        else:
            speedup = 1.0
        
        # Calculate effectiveness score
        qp_score = max(0, (1.0 - qp_rate) * 100)  # Higher is better
        speed_score = max(0, (speedup - 1.0) * 50)  # Higher is better
        overall_score = (qp_score + speed_score) / 2
        
        summary_text += f"{result['method']}:\n"
        summary_text += f"  QP Usage: {qp_rate:.1%}\n"
        summary_text += f"  Speedup: {speedup:.2f}x\n"
        summary_text += f"  Score: {overall_score:.1f}/100\n\n"
    
    ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    # Plot 8: Optimization Progress Timeline  
    ax8 = plt.subplot(3, 3, 8)
    
    # Show the progression of optimization
    optimization_stages = ['Baseline', 'Basic Opt', 'Smart QP', 'Tuned QP']
    stage_qp_rates = []
    stage_times = []
    
    for stage, key in zip(optimization_stages, ['baseline', 'basic', 'smart', 'tuned']):
        if key in successful_results:
            stage_qp_rates.append(successful_results[key]['qp_usage_rate'] * 100)
            stage_times.append(successful_results[key]['time'])
        else:
            stage_qp_rates.append(None)
            stage_times.append(None)
    
    # Plot QP usage progression
    valid_stages = [stage for stage, rate in zip(optimization_stages, stage_qp_rates) if rate is not None]
    valid_rates = [rate for rate in stage_qp_rates if rate is not None]
    
    if len(valid_rates) > 1:
        ax8.plot(range(len(valid_rates)), valid_rates, 'bo-', linewidth=2, markersize=8, label='QP Usage Rate')
        ax8.axhline(y=10, color='red', linestyle='--', alpha=0.7, label='Target: 10%')
        ax8.set_xlabel('Optimization Stage')
        ax8.set_ylabel('QP Usage Rate (%)')
        ax8.set_title('QP Optimization Progress')
        ax8.set_xticks(range(len(valid_rates)))
        ax8.set_xticklabels(valid_stages, rotation=45)
        ax8.legend()
        ax8.grid(True, alpha=0.3)
    
    # Plot 9: Final Assessment
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    # Final assessment text
    best_result = None
    best_qp_rate = 1.0
    
    for key, result in successful_results.items():
        if result['qp_usage_rate'] < best_qp_rate and key != 'baseline':
            best_qp_rate = result['qp_usage_rate']
            best_result = result
    
    assessment_text = "FINAL ASSESSMENT\n\n"
    
    if best_result:
        if best_qp_rate <= 0.12:
            assessment_text += "✓ OPTIMIZATION SUCCESSFUL\n\n"
            assessment_text += f"Target Achieved: {best_qp_rate:.1%} QP usage\n"
            assessment_text += f"Method: {best_result['method']}\n"
            if baseline_time:
                speedup = baseline_time / best_result['time']
                assessment_text += f"Performance: {speedup:.2f}x speedup\n"
        elif best_qp_rate <= 0.2:
            assessment_text += "⚠️ NEAR TARGET\n\n"
            assessment_text += f"Close to Target: {best_qp_rate:.1%} QP usage\n"
            assessment_text += f"Method: {best_result['method']}\n"
        else:
            assessment_text += "❌ NEEDS IMPROVEMENT\n\n"
            assessment_text += f"Current Best: {best_qp_rate:.1%} QP usage\n"
            assessment_text += "Further optimization needed\n"
    
    assessment_text += f"\nRecommendation:\n"
    if best_qp_rate <= 0.15:
        assessment_text += "Use optimized QP-Collocation\nfor production workloads."
    else:
        assessment_text += "Continue using Hybrid method\nuntil QP optimization improves."
    
    ax9.text(0.05, 0.95, assessment_text, transform=ax9.transAxes, fontsize=12,
            verticalalignment='top', weight='bold',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('/Users/zvezda/Library/CloudStorage/OneDrive-Personal/code/MFG_PDE/tests/method_comparisons/comprehensive_qp_optimization_results.png', 
               dpi=300, bbox_inches='tight')
    print(f"\nComprehensive QP optimization results saved to: comprehensive_qp_optimization_results.png")
    plt.show()

def main():
    """Run the comprehensive QP optimization test"""
    print("Starting Comprehensive QP Optimization Test...")
    print("This tests all QP optimization approaches developed")
    print("Expected execution time: 15-30 minutes")
    
    try:
        results = run_comprehensive_qp_optimization_test()
        
        print(f"\n{'='*100}")
        print("COMPREHENSIVE QP OPTIMIZATION TEST COMPLETED")
        print(f"{'='*100}")
        print("Check the summary above and generated plots for comprehensive results.")
        
        return results
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
        return None
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()