#!/usr/bin/env python3
"""
QP-Collocation Long-Time Mass Conservation Test
Extended time horizon (T=10) to demonstrate mass conservation properties.
"""

import numpy as np
import time
import matplotlib.pyplot as plt

from mfg_pde.alg.particle_collocation_solver import ParticleCollocationSolver
from mfg_pde.core.mfg_problem import ExampleMFGProblem
from mfg_pde.core.boundaries import BoundaryConditions

def run_long_time_qp_test():
    """Test QP-Collocation method with extended time horizon T=10"""
    print("="*80)
    print("QP-COLLOCATION LONG-TIME MASS CONSERVATION TEST")
    print("="*80)
    print("Extended time horizon T=10 to demonstrate mass conservation")
    print("Using QP constraints for monotonicity preservation")
    
    # Extended time problem parameters
    problem_params = {
        "xmin": 0.0,
        "xmax": 1.0,
        "Nx": 60,        # Good spatial resolution
        "T": 10.0,       # Extended time horizon
        "Nt": 200,       # More time steps for stability
        "sigma": 0.3,    # Moderate diffusion
        "coefCT": 0.05,  # Light coupling for stability
    }
    
    print(f"\nExtended Time Problem Parameters:")
    for key, value in problem_params.items():
        print(f"  {key}: {value}")
    
    # Create problem
    problem = ExampleMFGProblem(**problem_params)
    
    print(f"\nProblem setup:")
    print(f"  Domain: [{problem.xmin}, {problem.xmax}] × [0, {problem.T}]")
    print(f"  Grid: Dx = {problem.Dx:.4f}, Dt = {problem.Dt:.4f}")
    print(f"  CFL number: {problem_params['sigma']**2 * problem.Dt / problem.Dx**2:.4f}")
    print(f"  Total time steps: {problem.Nt}")
    
    # QP solver parameters optimized for long-time stability
    solver_params = {
        "num_particles": 1000,        # More particles for better statistics
        "delta": 0.35,               # Conservative neighborhood size
        "taylor_order": 2,           # Second-order accuracy
        "weight_function": "wendland", # Stable weight function
        "NiterNewton": 10,           # Adequate Newton iterations
        "l2errBoundNewton": 1e-4,    # Standard tolerance
        "kde_bandwidth": "scott",    # Adaptive KDE bandwidth
        "normalize_kde_output": False, # No artificial normalization
        "use_monotone_constraints": True # QP constraints for stability
    }
    
    print(f"\nQP Solver Parameters (Optimized for Long-Time):")
    for key, value in solver_params.items():
        if key == "use_monotone_constraints" and value:
            print(f"  {key}: {value} ← QP CONSTRAINTS ENABLED")
        else:
            print(f"  {key}: {value}")
    
    # Setup collocation points
    num_collocation_points = 15  # Good coverage
    collocation_points = np.linspace(
        problem.xmin, problem.xmax, num_collocation_points
    ).reshape(-1, 1)
    
    # Boundary indices
    boundary_tolerance = 1e-10
    boundary_indices = []
    for i, point in enumerate(collocation_points):
        x = point[0]
        if (abs(x - problem.xmin) < boundary_tolerance or 
            abs(x - problem.xmax) < boundary_tolerance):
            boundary_indices.append(i)
    boundary_indices = np.array(boundary_indices)
    
    print(f"  Collocation points: {num_collocation_points}")
    print(f"  Boundary points: {len(boundary_indices)}")
    
    # No-flux boundary conditions
    no_flux_bc = BoundaryConditions(type="no_flux")
    
    # Create solver
    print(f"\n--- Creating QP-Collocation Solver ---")
    solver = ParticleCollocationSolver(
        problem=problem,
        collocation_points=collocation_points,
        boundary_indices=boundary_indices,
        boundary_conditions=no_flux_bc,
        **solver_params
    )
    
    # Solve with extended convergence settings
    max_iterations = 20
    convergence_tolerance = 1e-3  # Relaxed for long-time stability
    
    print(f"\n--- Running Long-Time QP-Collocation Simulation ---")
    print(f"Max iterations: {max_iterations}")
    print(f"Convergence tolerance: {convergence_tolerance}")
    print("This may take several minutes due to extended time horizon...")
    
    start_time = time.time()
    
    try:
        U_solution, M_solution, solve_info = solver.solve(
            Niter=max_iterations,
            l2errBound=convergence_tolerance,
            verbose=True
        )
        
        total_time = time.time() - start_time
        iterations_run = solve_info.get("iterations", max_iterations)
        converged = solve_info.get("converged", False)
        
        print(f"\n--- QP Long-Time Simulation Completed ---")
        print(f"Total execution time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"Iterations: {iterations_run}/{max_iterations}")
        print(f"Converged: {converged}")
        
        if U_solution is not None and M_solution is not None:
            # Comprehensive mass conservation analysis
            print(f"\n{'='*60}")
            print("LONG-TIME MASS CONSERVATION ANALYSIS")
            print(f"{'='*60}")
            
            # Calculate mass at each time step
            mass_evolution = np.sum(M_solution * problem.Dx, axis=1)
            initial_mass = mass_evolution[0]
            final_mass = mass_evolution[-1]
            mass_change = final_mass - initial_mass
            mass_change_percent = (mass_change / initial_mass) * 100
            
            print(f"Initial mass (t=0): {initial_mass:.10f}")
            print(f"Final mass (t={problem.T}): {final_mass:.10f}")
            print(f"Absolute mass change: {mass_change:.2e}")
            print(f"Relative mass change: {mass_change_percent:+.6f}%")
            
            # Mass conservation statistics
            max_mass = np.max(mass_evolution)
            min_mass = np.min(mass_evolution)
            mass_variation = max_mass - min_mass
            mass_std = np.std(mass_evolution)
            
            print(f"\nMass Evolution Statistics:")
            print(f"  Maximum mass: {max_mass:.10f}")
            print(f"  Minimum mass: {min_mass:.10f}")
            print(f"  Mass variation: {mass_variation:.2e}")
            print(f"  Mass standard deviation: {mass_std:.2e}")
            print(f"  Relative variation: {mass_variation/initial_mass*100:.6f}%")
            
            # Mass conservation assessment
            if abs(mass_change_percent) < 0.1:
                mass_assessment = "✅ EXCELLENT mass conservation"
            elif abs(mass_change_percent) < 1.0:
                mass_assessment = "✅ VERY GOOD mass conservation"
            elif abs(mass_change_percent) < 3.0:
                mass_assessment = "✅ GOOD mass conservation"
            elif abs(mass_change_percent) < 10.0:
                mass_assessment = "⚠️  ACCEPTABLE mass conservation"
            else:
                mass_assessment = "❌ POOR mass conservation"
            
            print(f"\nMass Conservation Assessment: {mass_assessment}")
            
            # Physical evolution analysis
            print(f"\n{'='*60}")
            print("PHYSICAL EVOLUTION ANALYSIS")
            print(f"{'='*60}")
            
            # Track center of mass evolution
            center_of_mass_evolution = []
            max_density_locations = []
            peak_densities = []
            
            for t_idx in range(M_solution.shape[0]):
                # Center of mass
                com = np.sum(problem.xSpace * M_solution[t_idx, :]) * problem.Dx
                center_of_mass_evolution.append(com)
                
                # Peak density location and value
                max_idx = np.argmax(M_solution[t_idx, :])
                max_loc = problem.xSpace[max_idx]
                max_val = M_solution[t_idx, max_idx]
                max_density_locations.append(max_loc)
                peak_densities.append(max_val)
            
            center_of_mass_evolution = np.array(center_of_mass_evolution)
            max_density_locations = np.array(max_density_locations)
            peak_densities = np.array(peak_densities)
            
            print(f"Center of mass evolution:")
            print(f"  Initial: {center_of_mass_evolution[0]:.6f}")
            print(f"  Final: {center_of_mass_evolution[-1]:.6f}")
            print(f"  Change: {center_of_mass_evolution[-1] - center_of_mass_evolution[0]:+.6f}")
            
            print(f"\nPeak density evolution:")
            print(f"  Initial location: {max_density_locations[0]:.6f}")
            print(f"  Final location: {max_density_locations[-1]:.6f}")
            print(f"  Initial value: {peak_densities[0]:.6f}")
            print(f"  Final value: {peak_densities[-1]:.6f}")
            
            # Particle boundary analysis
            particles_trajectory = solver.fp_solver.M_particles_trajectory
            if particles_trajectory is not None:
                print(f"\n{'='*60}")
                print("PARTICLE BOUNDARY ANALYSIS")
                print(f"{'='*60}")
                
                # Count boundary violations over entire trajectory
                total_violations = 0
                violations_per_timestep = []
                
                for t_step in range(particles_trajectory.shape[0]):
                    step_particles = particles_trajectory[t_step, :]
                    violations = np.sum(
                        (step_particles < problem.xmin - 1e-10) | 
                        (step_particles > problem.xmax + 1e-10)
                    )
                    violations_per_timestep.append(violations)
                    total_violations += violations
                
                violations_per_timestep = np.array(violations_per_timestep)
                
                print(f"Particle trajectory analysis:")
                print(f"  Total particles: {particles_trajectory.shape[1]}")
                print(f"  Time steps: {particles_trajectory.shape[0]}")
                print(f"  Total boundary violations: {total_violations}")
                print(f"  Max violations per timestep: {np.max(violations_per_timestep)}")
                print(f"  Average violations per timestep: {np.mean(violations_per_timestep):.2f}")
                
                # Final particle distribution
                final_particles = particles_trajectory[-1, :]
                print(f"\nFinal particle distribution:")
                print(f"  Range: [{np.min(final_particles):.6f}, {np.max(final_particles):.6f}]")
                print(f"  Mean: {np.mean(final_particles):.6f}")
                print(f"  Std: {np.std(final_particles):.6f}")
                
                if total_violations == 0:
                    print("✅ EXCELLENT: No boundary violations throughout simulation!")
                elif np.max(violations_per_timestep) <= 5:
                    print("✅ VERY GOOD: Minimal boundary violations")
                else:
                    print("⚠️  Some boundary violations detected")
            
            # Solution quality metrics
            print(f"\n{'='*60}")
            print("SOLUTION QUALITY METRICS")
            print(f"{'='*60}")
            
            max_U = np.max(np.abs(U_solution))
            min_M = np.min(M_solution)
            negative_densities = np.sum(M_solution < -1e-10)
            
            print(f"Control field range: [-{max_U:.3f}, {max_U:.3f}]")
            print(f"Minimum density: {min_M:.2e}")
            print(f"Negative densities: {negative_densities}")
            
            if negative_densities == 0:
                print("✅ All densities are non-negative")
            else:
                print(f"⚠️  {negative_densities} negative density violations")
            
            # Create comprehensive plots
            create_long_time_plots(
                problem, M_solution, U_solution, mass_evolution,
                center_of_mass_evolution, max_density_locations, peak_densities,
                particles_trajectory, total_time
            )
            
            # Summary
            print(f"\n{'='*80}")
            print("LONG-TIME SIMULATION SUMMARY")
            print(f"{'='*80}")
            print(f"✅ Successfully completed T={problem.T} simulation")
            print(f"✅ {mass_assessment}")
            print(f"✅ Mass change over {problem.T} time units: {mass_change_percent:+.3f}%")
            print(f"✅ Zero boundary violations: {total_violations == 0}")
            print(f"✅ All densities non-negative: {negative_densities == 0}")
            print(f"⏱️  Execution time: {total_time:.1f}s ({total_time/60:.1f} min)")
            
        else:
            print("❌ Solver failed to produce valid results")
            
    except Exception as e:
        print(f"❌ Long-time simulation failed: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n--- QP Long-Time Mass Conservation Test Finished ---")

def create_long_time_plots(problem, M_solution, U_solution, mass_evolution,
                          center_of_mass_evolution, max_density_locations, peak_densities,
                          particles_trajectory, execution_time):
    """Create comprehensive plots for long-time simulation"""
    
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    fig.suptitle(f'QP-Collocation Long-Time Mass Conservation (T={problem.T})\n'
                f'Execution time: {execution_time:.1f}s ({execution_time/60:.1f} min)', 
                fontsize=16, fontweight='bold')
    
    # 1. Mass evolution over time
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(problem.tSpace, mass_evolution, 'g-', linewidth=2, alpha=0.8)
    ax1.axhline(y=mass_evolution[0], color='r', linestyle='--', alpha=0.7, 
                label=f'Initial mass: {mass_evolution[0]:.6f}')
    ax1.set_xlabel('Time t')
    ax1.set_ylabel('Total Mass')
    ax1.set_title(f'Mass Evolution over T={problem.T}', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add mass change annotation
    mass_change_percent = (mass_evolution[-1] - mass_evolution[0]) / mass_evolution[0] * 100
    ax1.text(0.05, 0.95, f'Mass change: {mass_change_percent:+.3f}%', 
             transform=ax1.transAxes, fontsize=12, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    # 2. Mass variation detail (zoomed view)
    ax2 = fig.add_subplot(gs[0, 2:])
    mass_variation = mass_evolution - mass_evolution[0]
    ax2.plot(problem.tSpace, mass_variation * 100, 'b-', linewidth=2)
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    ax2.set_xlabel('Time t')
    ax2.set_ylabel('Mass Change (%)')
    ax2.set_title('Mass Change Detail (Percent)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Center of mass evolution
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(problem.tSpace, center_of_mass_evolution, 'purple', linewidth=2)
    ax3.set_xlabel('Time t')
    ax3.set_ylabel('Center of Mass')
    ax3.set_title('Center of Mass Evolution')
    ax3.grid(True, alpha=0.3)
    
    # 4. Peak density location evolution
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(problem.tSpace, max_density_locations, 'orange', linewidth=2)
    ax4.set_xlabel('Time t')
    ax4.set_ylabel('Peak Density Location')
    ax4.set_title('Peak Location Evolution')
    ax4.grid(True, alpha=0.3)
    
    # 5. Peak density value evolution
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.plot(problem.tSpace, peak_densities, 'red', linewidth=2)
    ax5.set_xlabel('Time t')
    ax5.set_ylabel('Peak Density Value')
    ax5.set_title('Peak Value Evolution')
    ax5.grid(True, alpha=0.3)
    
    # 6. Final density profile
    ax6 = fig.add_subplot(gs[1, 3])
    ax6.plot(problem.xSpace, M_solution[0, :], 'g--', linewidth=2, alpha=0.7, label='Initial (t=0)')
    ax6.plot(problem.xSpace, M_solution[-1, :], 'r-', linewidth=2, label=f'Final (t={problem.T})')
    ax6.set_xlabel('Space x')
    ax6.set_ylabel('Density')
    ax6.set_title('Initial vs Final Density')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    
    # 7. Density evolution heatmap
    ax7 = fig.add_subplot(gs[2, :2])
    time_indices = np.linspace(0, len(problem.tSpace)-1, 50, dtype=int)  # Sample for visualization
    im = ax7.imshow(M_solution[time_indices, :].T, aspect='auto', origin='lower', 
                    extent=[problem.tSpace[time_indices[0]], problem.tSpace[time_indices[-1]], 
                           problem.xmin, problem.xmax], cmap='viridis')
    ax7.set_xlabel('Time t')
    ax7.set_ylabel('Space x')
    ax7.set_title('Density Evolution Heatmap')
    plt.colorbar(im, ax=ax7, label='Density')
    
    # 8. Control field evolution heatmap
    ax8 = fig.add_subplot(gs[2, 2:])
    im2 = ax8.imshow(U_solution[time_indices, :].T, aspect='auto', origin='lower',
                     extent=[problem.tSpace[time_indices[0]], problem.tSpace[time_indices[-1]], 
                            problem.xmin, problem.xmax], cmap='RdBu_r')
    ax8.set_xlabel('Time t')
    ax8.set_ylabel('Space x')
    ax8.set_title('Control Field Evolution Heatmap')
    plt.colorbar(im2, ax=ax8, label='Control U')
    
    # 9. Sample particle trajectories
    ax9 = fig.add_subplot(gs[3, :2])
    if particles_trajectory is not None:
        # Show sample particle trajectories
        sample_size = min(50, particles_trajectory.shape[1])
        particle_indices = np.linspace(0, particles_trajectory.shape[1]-1, sample_size, dtype=int)
        
        for i in particle_indices:
            ax9.plot(problem.tSpace, particles_trajectory[:, i], 'b-', alpha=0.3, linewidth=1)
        
        ax9.set_xlabel('Time t')
        ax9.set_ylabel('Particle Position')
        ax9.set_title(f'Sample Particle Trajectories (n={sample_size})')
        ax9.set_ylim([problem.xmin - 0.1, problem.xmax + 0.1])
        ax9.grid(True, alpha=0.3)
        
        # Add boundary lines
        ax9.axhline(y=problem.xmin, color='red', linestyle='--', alpha=0.7, label='Boundaries')
        ax9.axhline(y=problem.xmax, color='red', linestyle='--', alpha=0.7)
        ax9.legend()
    
    # 10. Mass conservation statistics
    ax10 = fig.add_subplot(gs[3, 2:])
    
    # Calculate rolling statistics
    window_size = max(1, len(mass_evolution) // 20)
    mass_variation_rolling = []
    time_windows = []
    
    for i in range(window_size, len(mass_evolution), window_size):
        window_mass = mass_evolution[i-window_size:i]
        variation = (np.max(window_mass) - np.min(window_mass)) / mass_evolution[0] * 100
        mass_variation_rolling.append(variation)
        time_windows.append(problem.tSpace[i])
    
    ax10.bar(time_windows, mass_variation_rolling, width=problem.T/len(time_windows)*0.8, 
             alpha=0.7, color='green')
    ax10.set_xlabel('Time t')
    ax10.set_ylabel('Mass Variation (%)')
    ax10.set_title('Mass Conservation Quality Over Time')
    ax10.grid(True, alpha=0.3, axis='y')
    
    # Add quality assessment
    avg_variation = np.mean(mass_variation_rolling)
    if avg_variation < 0.01:
        quality_text = "Excellent Conservation"
        quality_color = "green"
    elif avg_variation < 0.1:
        quality_text = "Very Good Conservation"
        quality_color = "blue"
    elif avg_variation < 1.0:
        quality_text = "Good Conservation"
        quality_color = "orange"
    else:
        quality_text = "Poor Conservation"
        quality_color = "red"
    
    ax10.text(0.05, 0.95, f'{quality_text}\nAvg: {avg_variation:.3f}%', 
              transform=ax10.transAxes, fontsize=10,
              bbox=dict(boxstyle="round,pad=0.3", facecolor=quality_color, alpha=0.3))
    
    plt.savefig('/Users/zvezda/Library/CloudStorage/OneDrive-Personal/code/MFG_PDE/qp_long_time_mass_conservation.png', 
                dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("Starting QP-Collocation Long-Time Mass Conservation Test...")
    print("Extended time horizon T=10 with comprehensive analysis")
    print("Expected execution time: 5-15 minutes")
    
    try:
        run_long_time_qp_test()
        print("\n" + "="*80)
        print("QP LONG-TIME MASS CONSERVATION TEST COMPLETED")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()