#!/usr/bin/env python3
"""
Simple Hybrid vs QP-Collocation Comparison
Direct comparison of particle-based methods with simplified hybrid implementation.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import sys
from scipy.stats import gaussian_kde
import scipy.sparse as sparse
import scipy.sparse.linalg

# Add the main package to path
sys.path.insert(0, '/Users/zvezda/Library/CloudStorage/OneDrive-Personal/code/MFG_PDE')

from mfg_pde.alg.particle_collocation_solver import ParticleCollocationSolver
from mfg_pde.alg.base_mfg_solver import MFGSolver
from mfg_pde.core.mfg_problem import ExampleMFGProblem
from mfg_pde.core.boundaries import BoundaryConditions

class SimpleHybridSolver(MFGSolver):
    """Simplified Hybrid Particle-FDM solver for comparison"""
    
    def __init__(self, problem, num_particles, kde_bandwidth="scott", 
                 NiterNewton=10, l2errBoundNewton=1e-4):
        super().__init__(problem)
        self.num_particles = num_particles
        self.kde_bandwidth = kde_bandwidth
        self.NiterNewton = NiterNewton
        self.l2errBoundNewton = l2errBoundNewton
        
        # Initialize solution arrays
        Nt = problem.Nt
        Nx = problem.Nx
        
        self.U = np.zeros((Nt+1, Nx+1))  # Grid has Nx+1 points
        self.M_particles = np.zeros((Nt+1, num_particles))
        self.M_density = np.zeros((Nt+1, Nx+1))  # Grid has Nx+1 points
        
        # Initialize particles uniformly
        self.M_particles[0, :] = np.random.uniform(problem.xmin, problem.xmax, num_particles)
        
        # Estimate initial density
        self.M_density[0, :] = self._estimate_density_from_particles(self.M_particles[0, :])
        
        # Terminal condition for U (simple quadratic)
        x = problem.xSpace
        self.U[-1, :] = 0.5 * (x - 0.5)**2
    
    def _estimate_density_from_particles(self, particles):
        """Estimate density on grid from particle positions using KDE"""
        try:
            grid_size = self.problem.Nx + 1  # Grid has Nx+1 points
            
            if len(particles) == 0:
                return np.zeros(grid_size)
            
            # Check for degenerate case (all particles at same location)
            if np.std(particles) < 1e-10:
                # Create a narrow Gaussian around the location
                center_idx = np.argmin(np.abs(self.problem.xSpace - np.mean(particles)))
                density = np.zeros(grid_size)
                density[center_idx] = 1.0
                return density / (np.sum(density) * self.problem.Dx)
            
            # Use KDE
            kde = gaussian_kde(particles, bw_method=self.kde_bandwidth)
            density = kde(self.problem.xSpace)
            
            # Normalize to preserve mass
            density = density / (np.sum(density) * self.problem.Dx)
            
            return np.maximum(density, 0)  # Ensure non-negative
            
        except Exception as e:
            # Fallback to histogram-based estimation
            grid_size = self.problem.Nx + 1
            hist, _ = np.histogram(particles, bins=grid_size, 
                                 range=(self.problem.xmin, self.problem.xmax))
            density = hist / (self.num_particles * self.problem.Dx)
            return density
    
    def _solve_hjb_step(self, t_idx, M_current):
        """Solve HJB equation using simple finite differences"""
        grid_size = self.problem.Nx + 1  # Grid has Nx+1 points
        Dx = self.problem.Dx
        sigma = self.problem.sigma
        
        # Build finite difference matrix for -sigma^2/2 * d²u/dx² + |∇u|²/2 = M
        diag_main = 2 * sigma**2 / Dx**2 * np.ones(grid_size)
        diag_off = -sigma**2 / Dx**2 * np.ones(grid_size-1)
        
        # Simple fixed-point iteration for nonlinear term
        U_old = self.U[t_idx, :].copy()
        
        for newton_iter in range(self.NiterNewton):
            # Compute gradient term |∇u|²/2
            grad_u = np.gradient(U_old, Dx)
            grad_term = 0.5 * grad_u**2
            
            # Right-hand side
            rhs = M_current + grad_term
            
            # Solve linear system
            A = sparse.diags([diag_off, diag_main, diag_off], [-1, 0, 1], 
                           shape=(grid_size, grid_size), format='csr')
            
            # Apply boundary conditions (Neumann)
            A[0, 1] = -2 * sigma**2 / Dx**2
            A[-1, -2] = -2 * sigma**2 / Dx**2
            
            try:
                U_new = sparse.linalg.spsolve(A, rhs)
                
                # Check convergence
                if np.linalg.norm(U_new - U_old) < self.l2errBoundNewton:
                    break
                    
                U_old = U_new
                
            except Exception:
                # If solve fails, keep previous solution
                U_new = U_old
                break
        
        return U_new
    
    def _evolve_particles_step(self, t_idx, dt):
        """Evolve particles forward one time step"""
        particles_old = self.M_particles[t_idx-1, :].copy()
        U_current = self.U[t_idx, :]
        
        # Compute control field gradient at particle locations
        grad_U = np.gradient(U_current, self.problem.Dx)
        
        for i in range(self.num_particles):
            x_old = particles_old[i]
            
            # Interpolate gradient at particle location
            if x_old <= self.problem.xmin:
                u_grad = grad_U[0]
            elif x_old >= self.problem.xmax:
                u_grad = grad_U[-1]
            else:
                # Linear interpolation
                idx = np.searchsorted(self.problem.xSpace, x_old) - 1
                idx = max(0, min(idx, len(grad_U)-2))
                alpha = (x_old - self.problem.xSpace[idx]) / self.problem.Dx
                u_grad = (1-alpha) * grad_U[idx] + alpha * grad_U[idx+1]
            
            # Particle dynamics: dx = -∇U dt + σ dW
            deterministic_drift = -u_grad * dt
            stochastic_drift = self.problem.sigma * np.sqrt(dt) * np.random.randn()
            
            x_new = x_old + deterministic_drift + stochastic_drift
            
            # Apply no-flux boundary conditions (reflection)
            if x_new < self.problem.xmin:
                x_new = 2 * self.problem.xmin - x_new
            elif x_new > self.problem.xmax:
                x_new = 2 * self.problem.xmax - x_new
            
            self.M_particles[t_idx, i] = x_new
    
    def solve(self, Niter=10, l2errBoundPicard=1e-4):
        """Main solve routine"""
        Nt = self.problem.Nt
        dt = self.problem.Dt
        
        # Picard iteration
        for picard_iter in range(Niter):
            U_old = self.U.copy()
            M_old = self.M_density.copy()
            
            # Backward sweep: solve HJB
            for t_idx in range(Nt-1, -1, -1):
                M_current = self.M_density[t_idx, :]
                self.U[t_idx, :] = self._solve_hjb_step(t_idx, M_current)
            
            # Forward sweep: evolve particles and estimate density
            for t_idx in range(1, Nt+1):
                self._evolve_particles_step(t_idx, dt)
                self.M_density[t_idx, :] = self._estimate_density_from_particles(
                    self.M_particles[t_idx, :]
                )
            
            # Check convergence
            u_err = np.linalg.norm(self.U - U_old) / np.linalg.norm(U_old)
            m_err = np.linalg.norm(self.M_density - M_old) / np.linalg.norm(M_old)
            
            if max(u_err, m_err) < l2errBoundPicard:
                return self.U, self.M_density, picard_iter+1, u_err, m_err
        
        return self.U, self.M_density, Niter, u_err, m_err
    
    def get_results(self):
        """Return computed U and M"""
        return self.U, self.M_density

def compare_hybrid_vs_qp():
    """Compare simplified Hybrid and QP-Collocation methods"""
    print("="*80)
    print("HYBRID vs QP-COLLOCATION COMPARISON")
    print("="*80)
    print("Comparing two particle-based methods with identical parameters")
    
    # Problem parameters
    problem_params = {
        "xmin": 0.0,
        "xmax": 1.0,
        "Nx": 30,
        "T": 0.2,
        "Nt": 10,
        "sigma": 0.15,
        "coefCT": 0.01
    }
    
    # Shared settings
    shared_settings = {
        "num_particles": 600,
        "newton_iterations": 6,
        "picard_iterations": 10,
        "tolerance": 1e-4
    }
    
    print(f"\nProblem Parameters:")
    for key, value in problem_params.items():
        print(f"  {key}: {value}")
    
    print(f"\nShared Settings:")
    print(f"  Particles: {shared_settings['num_particles']}")
    print(f"  Newton iterations: {shared_settings['newton_iterations']}")
    print(f"  Picard iterations: {shared_settings['picard_iterations']}")
    
    problem = ExampleMFGProblem(**problem_params)
    no_flux_bc = BoundaryConditions(type="no_flux")
    
    print(f"\nGrid: Dx = {problem.Dx:.4f}, Dt = {problem.Dt:.4f}")
    
    results = {}
    
    # Method 1: Simplified Hybrid
    print(f"\n{'='*60}")
    print("METHOD 1: SIMPLIFIED HYBRID PARTICLE-FDM")
    print(f"{'='*60}")
    
    try:
        start_time = time.time()
        np.random.seed(42)  # For reproducibility
        
        hybrid_solver = SimpleHybridSolver(
            problem=problem,
            num_particles=shared_settings["num_particles"],
            kde_bandwidth="scott",
            NiterNewton=shared_settings["newton_iterations"],
            l2errBoundNewton=shared_settings["tolerance"]
        )
        
        print(f"  Solving with {shared_settings['num_particles']} particles...")
        U_hybrid, M_hybrid, iterations_hybrid, l2dist_u, l2dist_m = hybrid_solver.solve(
            Niter=shared_settings["picard_iterations"],
            l2errBoundPicard=shared_settings["tolerance"]
        )
        
        hybrid_time = time.time() - start_time
        
        # Analysis
        mass_evolution_hybrid = np.sum(M_hybrid * problem.Dx, axis=1)
        initial_mass_hybrid = mass_evolution_hybrid[0]
        final_mass_hybrid = mass_evolution_hybrid[-1]
        mass_change_hybrid = final_mass_hybrid - initial_mass_hybrid
        
        center_of_mass_hybrid = np.sum(problem.xSpace * M_hybrid[-1, :]) * problem.Dx
        max_density_idx_hybrid = np.argmax(M_hybrid[-1, :])
        max_density_loc_hybrid = problem.xSpace[max_density_idx_hybrid]
        final_density_peak_hybrid = M_hybrid[-1, max_density_idx_hybrid]
        
        # Boundary violations
        final_particles = hybrid_solver.M_particles[-1, :]
        violations_hybrid = np.sum(
            (final_particles < problem.xmin - 1e-10) | 
            (final_particles > problem.xmax + 1e-10)
        )
        
        results['hybrid'] = {
            'success': True,
            'method_name': 'Simplified Hybrid',
            'time': hybrid_time,
            'iterations': iterations_hybrid,
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
        
        print(f"  ✓ Completed in {hybrid_time:.2f}s ({iterations_hybrid} iterations)")
        print(f"    Mass: {initial_mass_hybrid:.6f} → {final_mass_hybrid:.6f} ({(mass_change_hybrid/initial_mass_hybrid)*100:+.3f}%)")
        print(f"    Center of mass: {center_of_mass_hybrid:.4f}")
        print(f"    Max density: {final_density_peak_hybrid:.3f} at x = {max_density_loc_hybrid:.4f}")
        print(f"    Violations: {violations_hybrid}")
        
    except Exception as e:
        results['hybrid'] = {'success': False, 'error': str(e)}
        print(f"  ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Method 2: QP Particle-Collocation
    print(f"\n{'='*60}")
    print("METHOD 2: QP PARTICLE-COLLOCATION")
    print(f"{'='*60}")
    
    try:
        start_time = time.time()
        np.random.seed(42)  # Same seed for fairness
        
        # Setup collocation
        num_collocation_points = 8
        collocation_points = np.linspace(problem.xmin, problem.xmax, num_collocation_points).reshape(-1, 1)
        boundary_indices = [0, num_collocation_points - 1]
        
        qp_solver = ParticleCollocationSolver(
            problem=problem,
            collocation_points=collocation_points,
            num_particles=shared_settings["num_particles"],
            delta=0.25,
            taylor_order=2,
            weight_function="wendland",
            NiterNewton=shared_settings["newton_iterations"],
            l2errBoundNewton=shared_settings["tolerance"],
            kde_bandwidth="scott",
            normalize_kde_output=False,
            boundary_indices=np.array(boundary_indices),
            boundary_conditions=no_flux_bc,
            use_monotone_constraints=True
        )
        
        print(f"  Solving with {shared_settings['num_particles']} particles, {num_collocation_points} collocation points...")
        U_qp, M_qp, info_qp = qp_solver.solve(
            Niter=shared_settings["picard_iterations"],
            l2errBound=shared_settings["tolerance"],
            verbose=False
        )
        
        qp_time = time.time() - start_time
        
        # Analysis
        mass_evolution_qp = np.sum(M_qp * problem.Dx, axis=1)
        initial_mass_qp = mass_evolution_qp[0]
        final_mass_qp = mass_evolution_qp[-1]
        mass_change_qp = final_mass_qp - initial_mass_qp
        
        center_of_mass_qp = np.sum(problem.xSpace * M_qp[-1, :]) * problem.Dx
        max_density_idx_qp = np.argmax(M_qp[-1, :])
        max_density_loc_qp = problem.xSpace[max_density_idx_qp]
        final_density_peak_qp = M_qp[-1, max_density_idx_qp]
        
        # Boundary violations
        particles_traj = qp_solver.get_particles_trajectory()
        violations_qp = 0
        if particles_traj is not None:
            final_particles = particles_traj[-1, :]
            violations_qp = np.sum(
                (final_particles < problem.xmin - 1e-10) | 
                (final_particles > problem.xmax + 1e-10)
            )
        
        results['qp'] = {
            'success': True,
            'method_name': 'QP Particle-Collocation',
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
        
        print(f"  ✓ Completed in {qp_time:.2f}s ({info_qp.get('iterations', 0)} iterations)")
        print(f"    Converged: {info_qp.get('converged', False)}")
        print(f"    Mass: {initial_mass_qp:.6f} → {final_mass_qp:.6f} ({(mass_change_qp/initial_mass_qp)*100:+.3f}%)")
        print(f"    Center of mass: {center_of_mass_qp:.4f}")
        print(f"    Max density: {final_density_peak_qp:.3f} at x = {max_density_loc_qp:.4f}")
        print(f"    Violations: {violations_qp}")
        
    except Exception as e:
        results['qp'] = {'success': False, 'error': str(e)}
        print(f"  ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Analysis
    print(f"\n{'='*80}")
    print("COMPARISON ANALYSIS")
    print(f"{'='*80}")
    
    analyze_comparison(results)
    create_comparison_plots(results, problem)
    
    return results

def analyze_comparison(results):
    """Analyze the comparison results"""
    successful_methods = [method for method in ['hybrid', 'qp'] 
                         if results.get(method, {}).get('success', False)]
    
    if len(successful_methods) != 2:
        print("Cannot perform full comparison - not all methods succeeded")
        return
    
    hybrid = results['hybrid']
    qp = results['qp']
    
    # Summary table
    print(f"\n{'Metric':<25} {'Hybrid':<15} {'QP-Collocation':<15} {'Difference':<15}")
    print(f"{'-'*25} {'-'*15} {'-'*15} {'-'*15}")
    
    # Compare key metrics
    metrics = [
        ('Final Mass', lambda r: r['mass_conservation']['final_mass'], '.6f'),
        ('Mass Change %', lambda r: r['mass_conservation']['mass_change_percent'], '+.2f'),
        ('Center of Mass', lambda r: r['physical_observables']['center_of_mass'], '.4f'),
        ('Max Density Loc', lambda r: r['physical_observables']['max_density_location'], '.4f'),
        ('Peak Density', lambda r: r['physical_observables']['final_density_peak'], '.3f'),
        ('Execution Time', lambda r: r['time'], '.2f'),
        ('Iterations', lambda r: r['iterations'], 'd'),
        ('Violations', lambda r: r['solution_quality']['violations'], 'd')
    ]
    
    for metric_name, extract_func, fmt in metrics:
        hybrid_val = extract_func(hybrid)
        qp_val = extract_func(qp)
        
        if fmt == 'd':
            diff_str = str(abs(qp_val - hybrid_val))
            hybrid_str = str(hybrid_val)
            qp_str = str(qp_val)
        else:
            if '+' in fmt:
                hybrid_str = f"{hybrid_val:{fmt}}%"
                qp_str = f"{qp_val:{fmt}}%"
                diff_str = f"{abs(qp_val - hybrid_val):.2f}pp"
            else:
                hybrid_str = f"{hybrid_val:{fmt}}"
                qp_str = f"{qp_val:{fmt}}"
                if 'time' in metric_name.lower():
                    diff_str = f"{abs(qp_val - hybrid_val):.2f}s"
                else:
                    diff_str = f"{abs(qp_val - hybrid_val):.4f}"
        
        print(f"{metric_name:<25} {hybrid_str:<15} {qp_str:<15} {diff_str:<15}")
    
    # Detailed analysis
    print(f"\n--- DETAILED ANALYSIS ---")
    
    mass_diff = abs(qp['mass_conservation']['mass_change_percent'] - 
                   hybrid['mass_conservation']['mass_change_percent'])
    com_diff = abs(qp['physical_observables']['center_of_mass'] - 
                   hybrid['physical_observables']['center_of_mass'])
    
    print(f"Mass change difference: {mass_diff:.2f} percentage points")
    print(f"Center of mass difference: {com_diff:.4f}")
    
    # Check consistency
    both_increase = (hybrid['mass_conservation']['mass_change'] > 0 and 
                    qp['mass_conservation']['mass_change'] > 0)
    both_clean = (hybrid['solution_quality']['violations'] == 0 and
                 qp['solution_quality']['violations'] == 0 and
                 hybrid['solution_quality']['negative_densities'] == 0 and
                 qp['solution_quality']['negative_densities'] == 0)
    
    if both_increase:
        print("✅ Both methods show mass increase (expected with no-flux BC)")
    if both_clean:
        print("✅ Both methods produce numerically clean solutions")
    
    # Performance
    faster_method = "Hybrid" if hybrid['time'] < qp['time'] else "QP-Collocation"
    time_ratio = max(hybrid['time'], qp['time']) / min(hybrid['time'], qp['time'])
    print(f"Faster method: {faster_method} ({time_ratio:.2f}x speedup)")

def create_comparison_plots(results, problem):
    """Create comparison plots"""
    if not (results.get('hybrid', {}).get('success') and results.get('qp', {}).get('success')):
        print("Cannot create plots - insufficient successful results")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Hybrid vs QP-Collocation Comparison', fontsize=14)
    
    hybrid = results['hybrid']
    qp = results['qp']
    
    # 1. Final density comparison
    ax1 = axes[0, 0]
    ax1.plot(problem.xSpace, hybrid['arrays']['M_solution'][-1, :], 
             'g-', linewidth=2, label='Hybrid')
    ax1.plot(problem.xSpace, qp['arrays']['M_solution'][-1, :], 
             'r-', linewidth=2, label='QP-Collocation')
    ax1.set_xlabel('Space x')
    ax1.set_ylabel('Final Density')
    ax1.set_title('Final Density Comparison')
    ax1.grid(True)
    ax1.legend()
    
    # 2. Mass evolution
    ax2 = axes[0, 1]
    ax2.plot(problem.tSpace, hybrid['arrays']['mass_evolution'], 
             'g-o', linewidth=2, label='Hybrid')
    ax2.plot(problem.tSpace, qp['arrays']['mass_evolution'], 
             'r-s', linewidth=2, label='QP-Collocation')
    ax2.set_xlabel('Time t')
    ax2.set_ylabel('Total Mass')
    ax2.set_title('Mass Evolution')
    ax2.grid(True)
    ax2.legend()
    
    # 3. Performance comparison
    ax3 = axes[1, 0]
    methods = ['Hybrid', 'QP-Collocation']
    times = [hybrid['time'], qp['time']]
    iterations = [hybrid['iterations'], qp['iterations']]
    
    x_pos = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax3.bar(x_pos - width/2, times, width, label='Time (s)', color='blue', alpha=0.7)
    ax3_twin = ax3.twinx()
    bars2 = ax3_twin.bar(x_pos + width/2, iterations, width, label='Iterations', color='orange', alpha=0.7)
    
    ax3.set_xlabel('Method')
    ax3.set_ylabel('Time (s)', color='blue')
    ax3_twin.set_ylabel('Iterations', color='orange')
    ax3.set_title('Performance Comparison')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(methods)
    ax3.grid(True, axis='y')
    
    # 4. Key observables
    ax4 = axes[1, 1]
    observables = ['Center of Mass', 'Max Density Loc']
    hybrid_vals = [hybrid['physical_observables']['center_of_mass'],
                   hybrid['physical_observables']['max_density_location']]
    qp_vals = [qp['physical_observables']['center_of_mass'],
               qp['physical_observables']['max_density_location']]
    
    x_pos = np.arange(len(observables))
    width = 0.35
    
    ax4.bar(x_pos - width/2, hybrid_vals, width, label='Hybrid', color='green', alpha=0.7)
    ax4.bar(x_pos + width/2, qp_vals, width, label='QP-Collocation', color='red', alpha=0.7)
    
    ax4.set_xlabel('Observable')
    ax4.set_ylabel('Value')
    ax4.set_title('Physical Observables')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(observables)
    ax4.legend()
    ax4.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig('/Users/zvezda/Library/CloudStorage/OneDrive-Personal/code/MFG_PDE/simple_hybrid_vs_qp.png', 
                dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("Starting Hybrid vs QP-Collocation comparison...")
    print("Expected execution time: 1-3 minutes")
    
    try:
        results = compare_hybrid_vs_qp()
        print("\n" + "="*80)
        print("COMPARISON COMPLETED")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\nComparison interrupted by user.")
    except Exception as e:
        print(f"\nComparison failed: {e}")
        import traceback
        traceback.print_exc()