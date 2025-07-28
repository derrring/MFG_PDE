#!/usr/bin/env python3
"""
Generate Santa Fe Bar Comparison Results

This script runs both discrete and continuous MFG implementations
for the Santa Fe Bar Problem and saves the results in JSON format
for use in the comparison notebook.
"""

import numpy as np
import json
import time
from pathlib import Path

# Configure matplotlib to avoid font warnings
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# Import MFG implementations
from examples.advanced.santa_fe_bar_discrete_mfg import DiscreteSantaFeBarMFG
from examples.basic.el_farol_simple_working import create_el_farol_problem, analyze_el_farol_solution
from mfg_pde import create_fast_solver
from mfg_pde.utils.logging import get_logger, configure_research_logging


def run_discrete_mfg(settings):
    """Run discrete MFG implementation."""
    logger = get_logger(__name__)
    logger.info(f"Running discrete MFG: {settings['name']}")
    
    start_time = time.time()
    
    # Create and solve discrete problem
    problem = DiscreteSantaFeBarMFG(
        T=settings['T'],
        m_threshold=settings['threshold'],
        payoff_good=settings['payoff_good'],
        payoff_bad=settings['payoff_bad'],
        payoff_home=0.0,
        noise_level=settings['noise_level'],
        initial_m=settings['initial_attendance']
    )
    
    solution = problem.solve(nt=2000)
    analysis = problem.analyze_equilibrium(solution)
    
    solve_time = time.time() - start_time
    
    return {
        'solve_time': float(solve_time),
        'final_values': {
            'u0': float(solution['final_u0']),
            'u1': float(solution['final_u1'])
        },
        'analysis': {
            'steady_state_attendance': float(analysis['steady_state_attendance']),
            'efficiency': float(analysis['efficiency']),
            'regime': str(analysis['regime']),
            'convergence_achieved': bool(analysis['convergence_achieved']),
            'oscillation_amplitude': float(analysis.get('oscillation_amplitude', 0.0))
        }
    }


def run_continuous_mfg(settings):
    """Run continuous MFG implementation."""
    logger = get_logger(__name__)
    logger.info(f"Running continuous MFG: {settings['name']}")
    
    start_time = time.time()
    
    # Create and solve continuous problem
    problem = create_el_farol_problem(
        bar_capacity=settings['threshold'],
        crowd_aversion=settings['crowd_aversion'],
        Nx=settings['Nx'],
        Nt=settings['Nt']
    )
    
    # Override sigma parameter
    problem.sigma = settings['sigma']
    
    solver = create_fast_solver(problem, solver_type="fixed_point")
    result = solver.solve()
    U, M = result.U, result.M
    
    # Analyze results
    analysis = analyze_el_farol_solution(problem, U, M)
    
    solve_time = time.time() - start_time
    
    # Determine regime based on attendance
    attendance = analysis['final_attendance']
    threshold = settings['threshold']
    
    if attendance > threshold * 1.1:
        regime = 'overcrowded'
    elif attendance < threshold * 0.9:
        regime = 'underutilized'
    else:
        regime = 'optimal'
    
    return {
        'solve_time': float(solve_time),
        'analysis': {
            'steady_state_attendance': float(analysis['final_attendance']),
            'efficiency': float(analysis['efficiency']),
            'regime': str(regime),
            'converged': bool(analysis['converged']),
            'attendance_variance': float(np.var(analysis['attendance_evolution'][-10:]))
        }
    }


def generate_comparison_results():
    """Generate complete comparison results for notebook."""
    
    configure_research_logging("santa_fe_comparison", level="INFO")
    logger = get_logger(__name__)
    
    logger.info("🍺 Generating Santa Fe Bar Comparison Results")
    logger.info("=" * 60)
    
    # Define scenarios for balanced comparison
    scenarios = {
        'balanced_coordination': {
            'name': 'Balanced Coordination',
            'description': 'Moderate parameters for realistic coordination behavior',
            'T': 20.0,
            'threshold': 0.6,
            'initial_attendance': 0.3,
            'payoff_good': 10.0,
            'payoff_bad': -5.0,
            'noise_level': 1.0,
            'sigma': 0.15,
            'crowd_aversion': 2.0,
            'Nx': 50,
            'Nt': 50
        },
        'low_uncertainty': {
            'name': 'Low Uncertainty',
            'description': 'Strong preferences lead to deterministic decisions',
            'T': 20.0,
            'threshold': 0.6,
            'initial_attendance': 0.3,
            'payoff_good': 10.0,
            'payoff_bad': -5.0,
            'noise_level': 0.5,
            'sigma': 0.08,
            'crowd_aversion': 3.0,
            'Nx': 50,
            'Nt': 50
        },
        'high_uncertainty': {
            'name': 'High Uncertainty',
            'description': 'Weak preferences lead to exploratory behavior',
            'T': 20.0,
            'threshold': 0.6,
            'initial_attendance': 0.3,
            'payoff_good': 10.0,
            'payoff_bad': -5.0,
            'noise_level': 2.0,
            'sigma': 0.25,
            'crowd_aversion': 1.0,
            'Nx': 50,
            'Nt': 50
        }
    }
    
    results = {}
    
    for scenario_id, settings in scenarios.items():
        logger.info(f"\n📊 Processing scenario: {settings['name']}")
        
        # Run both implementations
        discrete_results = run_discrete_mfg(settings)
        continuous_results = run_continuous_mfg(settings)
        
        results[scenario_id] = {
            'settings': settings,
            'discrete': discrete_results,
            'continuous': continuous_results
        }
        
        # Log comparison
        discrete_time = discrete_results['solve_time']
        continuous_time = continuous_results['solve_time']
        speedup = continuous_time / discrete_time
        
        logger.info(f"  Discrete MFG:   {discrete_time:.3f}s, "
                   f"attendance={discrete_results['analysis']['steady_state_attendance']:.1%}")
        logger.info(f"  Continuous MFG: {continuous_time:.3f}s, "
                   f"attendance={continuous_results['analysis']['steady_state_attendance']:.1%}")
        logger.info(f"  Speedup: {speedup:.1f}× (discrete faster)")
    
    # Save results to JSON file
    output_file = "santa_fe_comparison_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n✅ Comparison results saved to: {output_file}")
    
    # Print summary
    logger.info(f"\n📊 COMPARISON SUMMARY:")
    avg_discrete_time = np.mean([results[s]['discrete']['solve_time'] for s in results.keys()])
    avg_continuous_time = np.mean([results[s]['continuous']['solve_time'] for s in results.keys()])
    overall_speedup = avg_continuous_time / avg_discrete_time
    
    logger.info(f"  • Scenarios analyzed: {len(scenarios)}")
    logger.info(f"  • Average discrete solve time: {avg_discrete_time:.3f}s")
    logger.info(f"  • Average continuous solve time: {avg_continuous_time:.3f}s")
    logger.info(f"  • Overall speedup: {overall_speedup:.1f}× (discrete faster)")
    
    return results


if __name__ == "__main__":
    generate_comparison_results()