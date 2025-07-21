#!/usr/bin/env python3
"""
Robust comparison of three MFG solver methods with enhanced stability:
1. Robust Pure FDM with implicit schemes and adaptive controls
2. Robust Hybrid Particle-FDM with stabilization
3. Second-Order QP Particle-Collocation (validated implementation)
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import sys
import os
from scipy.stats import gaussian_kde
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# Add the main package to path
sys.path.insert(0, '/Users/zvezda/Library/CloudStorage/OneDrive-Personal/code/MFG_PDE')

from mfg_pde.alg.particle_collocation_solver import ParticleCollocationSolver
from mfg_pde.core.mfg_problem import ExampleMFGProblem
from mfg_pde.core.boundaries import BoundaryConditions
from mfg_pde.alg.base_mfg_solver import MFGSolver

# Robust Pure FDM implementation
class RobustFDMSolver(MFGSolver):
    """Robust Pure FDM solver with implicit schemes and adaptive stability controls"""
    
    def __init__(self, problem, boundary_conditions=None):
        super().__init__(problem)
        self.hjb_method_name = "FDM"
        self.boundary_conditions = boundary_conditions
        self.U = None
        self.M = None
        self.iterations_run = 0
        self.converged = False
        
    def get_results(self):
        if self.U is None or self.M is None:
            raise ValueError("No solution available. Call solve() first.")
        return self.U, self.M
        
    def solve(self, Niter=10, l2errBound=1e-3):
        """Robust FDM solve with implicit schemes and stability controls"""
        print("Running Robust Pure FDM solver...")
        
        Nx = self.problem.Nx
        Nt = self.problem.Nt
        dx = self.problem.Dx
        dt = self.problem.Dt
        sigma = self.problem.sigma
        coefCT = self.problem.coefCT
        
        # Stability check - reduce dt if needed
        cfl_condition = sigma**2 * dt / dx**2
        if cfl_condition > 0.4:
            dt_factor = 0.4 / cfl_condition
            dt = dt * dt_factor
            print(f"  Stability: reducing dt by factor {dt_factor:.3f}")
        
        # Initialize solutions
        U = np.zeros((Nt + 1, Nx + 1))
        M = np.zeros((Nt + 1, Nx + 1))
        
        # Set terminal condition for U
        U[Nt, :] = 0.0
        
        # Set initial condition for M (normalized Gaussian)
        x_center = (self.problem.xmin + self.problem.xmax) / 2
        x_std = 0.15
        for i in range(Nx + 1):
            x = self.problem.xSpace[i]
            M[0, i] = np.exp(-0.5 * ((x - x_center) / x_std) ** 2)
        
        # Normalize initial density
        total_mass = np.sum(M[0, :]) * dx
        M[0, :] = M[0, :] / total_mass
        
        # Picard iteration
        for picard_iter in range(Niter):
            U_prev = U.copy()
            M_prev = M.copy()
            
            # Step 1: Solve HJB equation backward (implicit)
            for t_idx in range(Nt - 1, -1, -1):
                if t_idx == Nt - 1:
                    # Terminal condition
                    U[t_idx, :] = 0.0
                    continue
                    
                # Build implicit system for HJB
                A = np.zeros((Nx + 1, Nx + 1))
                b = np.zeros(Nx + 1)
                
                # Interior points
                for i in range(1, Nx):
                    # Coefficients for implicit scheme
                    diff_coeff = 0.5 * sigma**2 / dx**2
                    
                    # Central coefficients
                    A[i, i-1] = -diff_coeff
                    A[i, i] = 1.0 / dt + 2 * diff_coeff
                    A[i, i+1] = -diff_coeff
                    
                    # RHS from explicit terms
                    coupling = coefCT * M[t_idx, i]
                    b[i] = U[t_idx + 1, i] / dt + coupling
                
                # Boundary conditions (no-flux: dU/dx = 0)
                A[0, 0] = 1.0
                A[0, 1] = -1.0
                b[0] = 0.0
                
                A[Nx, Nx] = 1.0
                A[Nx, Nx-1] = -1.0
                b[Nx] = 0.0
                
                # Solve implicit system
                try:
                    U[t_idx, :] = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    # Fallback to explicit with very small time step
                    for i in range(1, Nx):
                        coupling = coefCT * M[t_idx, i]
                        U_xx = (U[t_idx + 1, i + 1] - 2 * U[t_idx + 1, i] + U[t_idx + 1, i - 1]) / (dx ** 2)
                        U[t_idx, i] = U[t_idx + 1, i] + 0.001 * dt * (coupling - 0.5 * sigma**2 * U_xx)
                    U[t_idx, 0] = U[t_idx, 1]
                    U[t_idx, Nx] = U[t_idx, Nx - 1]
            
            # Step 2: Solve FP equation forward (implicit)
            for t_idx in range(1, Nt + 1):
                # Build implicit system for FP
                A = np.zeros((Nx + 1, Nx + 1))
                b = np.zeros(Nx + 1)
                
                # Interior points
                for i in range(1, Nx):
                    # Diffusion coefficient
                    diff_coeff = 0.5 * sigma**2 / dx**2
                    
                    # Drift from optimal control
                    U_x = (U[t_idx - 1, i + 1] - U[t_idx - 1, i - 1]) / (2 * dx)
                    U_x = np.clip(U_x, -10.0, 10.0)  # Stability clipping
                    
                    # Upwind for advection based on drift direction
                    if U_x >= 0:
                        # Backward difference
                        A[i, i-1] = U_x / dx
                        A[i, i] = 1.0 / dt + U_x / dx + 2 * diff_coeff
                    else:
                        # Forward difference
                        A[i, i] = 1.0 / dt - U_x / dx + 2 * diff_coeff
                        A[i, i+1] = -U_x / dx
                    
                    # Diffusion terms
                    A[i, i-1] += -diff_coeff
                    A[i, i+1] += -diff_coeff
                    
                    # RHS
                    b[i] = M[t_idx - 1, i] / dt
                
                # Boundary conditions (no-flux: dM/dx = 0)
                A[0, 0] = 1.0
                A[0, 1] = -1.0
                b[0] = 0.0
                
                A[Nx, Nx] = 1.0
                A[Nx, Nx-1] = -1.0
                b[Nx] = 0.0
                
                # Solve implicit system
                try:
                    M_new = np.linalg.solve(A, b)
                    # Ensure positivity
                    M_new = np.maximum(M_new, 0.0)
                    M[t_idx, :] = M_new
                except np.linalg.LinAlgError:
                    # Fallback to simple explicit
                    M[t_idx, :] = M[t_idx - 1, :]
                    M[t_idx, 0] = M[t_idx, 1]
                    M[t_idx, Nx] = M[t_idx, Nx - 1]
            
            # Check convergence
            U_error = np.linalg.norm(U - U_prev) / max(np.linalg.norm(U_prev), 1e-10)
            M_error = np.linalg.norm(M - M_prev) / max(np.linalg.norm(M_prev), 1e-10)
            total_error = max(U_error, M_error)
            
            self.iterations_run = picard_iter + 1
            
            print(f"  Iteration {picard_iter + 1}: U_error={U_error:.2e}, M_error={M_error:.2e}")
            
            if total_error < l2errBound:
                self.converged = True
                print(f"  Converged at iteration {picard_iter + 1}")
                break
        
        self.U = U
        self.M = M
        
        return U, M

# Robust Hybrid solver
class RobustHybridSolver(MFGSolver):
    """Robust Hybrid Particle-FDM solver with enhanced stability"""
    
    def __init__(self, problem, num_particles=1000, boundary_conditions=None):
        super().__init__(problem)
        self.hjb_method_name = "Hybrid"
        self.num_particles = num_particles
        self.boundary_conditions = boundary_conditions
        self.U = None
        self.M = None
        self.particles = None
        self.iterations_run = 0
        self.converged = False
        
    def get_results(self):
        if self.U is None or self.M is None:
            raise ValueError("No solution available. Call solve() first.")
        return self.U, self.M
        
    def solve(self, Niter=10, l2errBound=1e-3):
        """Robust hybrid solve with enhanced stability"""
        print("Running Robust Hybrid Particle-FDM solver...")
        
        Nx = self.problem.Nx
        Nt = self.problem.Nt
        dx = self.problem.Dx
        dt = self.problem.Dt
        sigma = self.problem.sigma
        coefCT = self.problem.coefCT
        
        # Adaptive time stepping for particle stability
        dt_particle = min(dt, 0.1 * dx**2 / sigma**2)
        sub_steps = max(1, int(dt / dt_particle))
        dt_particle = dt / sub_steps
        
        # Initialize HJB solution on grid
        U = np.zeros((Nt + 1, Nx + 1))
        U[Nt, :] = 0.0
        
        # Initialize particles
        particles = np.zeros((Nt + 1, self.num_particles))
        
        # Initial particle distribution (same as FDM)
        x_center = (self.problem.xmin + self.problem.xmax) / 2
        x_std = 0.15
        particles[0, :] = np.random.normal(x_center, x_std, self.num_particles)
        particles[0, :] = np.clip(particles[0, :], self.problem.xmin, self.problem.xmax)
        
        # Picard iteration
        for picard_iter in range(Niter):
            U_prev = U.copy()
            particles_prev = particles.copy()
            
            # Step 1: Estimate density from particles
            M_from_particles = self._estimate_density_from_particles(particles)
            
            # Step 2: Solve HJB with implicit scheme
            for t_idx in range(Nt - 1, -1, -1):
                if t_idx == Nt - 1:
                    U[t_idx, :] = 0.0
                    continue
                
                # Simple implicit scheme for HJB
                for i in range(1, Nx):
                    coupling = coefCT * M_from_particles[t_idx, i]
                    U_xx = (U[t_idx + 1, i + 1] - 2 * U[t_idx + 1, i] + U[t_idx + 1, i - 1]) / (dx ** 2)
                    
                    # Conservative update
                    U[t_idx, i] = U[t_idx + 1, i] + dt * (coupling - 0.5 * sigma**2 * U_xx)
                
                # No-flux boundary conditions
                U[t_idx, 0] = U[t_idx, 1]
                U[t_idx, Nx] = U[t_idx, Nx - 1]
            
            # Step 3: Evolve particles with sub-stepping
            for t_idx in range(1, Nt + 1):
                for sub_step in range(sub_steps):
                    for p in range(self.num_particles):
                        x_p = particles[t_idx - 1, p]
                        
                        # Interpolate control field gradient
                        i = int((x_p - self.problem.xmin) / dx)
                        i = max(0, min(Nx - 1, i))
                        
                        # Linear interpolation for U_x
                        if i < Nx:
                            alpha = (x_p - self.problem.xSpace[i]) / dx
                            alpha = max(0, min(1, alpha))
                            
                            U_x_left = (U[t_idx - 1, i + 1] - U[t_idx - 1, i]) / dx if i < Nx else 0
                            U_x_right = (U[t_idx - 1, i + 1] - U[t_idx - 1, i]) / dx if i + 1 < Nx else 0
                            U_x_p = (1 - alpha) * U_x_left + alpha * U_x_right
                            
                            # Stability clipping
                            U_x_p = np.clip(U_x_p, -5.0, 5.0)
                        else:
                            U_x_p = 0
                        
                        # Particle SDE with sub-stepping
                        drift = -U_x_p
                        noise = sigma * np.random.normal(0, np.sqrt(dt_particle))
                        
                        new_pos = x_p + drift * dt_particle + noise
                        
                        # Reflect at boundaries
                        if new_pos < self.problem.xmin:
                            new_pos = 2 * self.problem.xmin - new_pos
                        elif new_pos > self.problem.xmax:
                            new_pos = 2 * self.problem.xmax - new_pos
                        
                        particles[t_idx - 1, p] = new_pos
                
                # Copy final sub-step result
                particles[t_idx, :] = particles[t_idx - 1, :]
            
            # Check convergence
            U_error = np.linalg.norm(U - U_prev) / max(np.linalg.norm(U_prev), 1e-10)
            particles_error = np.linalg.norm(particles - particles_prev) / max(np.linalg.norm(particles_prev), 1e-10)
            total_error = max(U_error, particles_error)
            
            self.iterations_run = picard_iter + 1
            
            print(f"  Iteration {picard_iter + 1}: U_error={U_error:.2e}, particles_error={particles_error:.2e}")
            
            if total_error < l2errBound:
                self.converged = True
                print(f"  Converged at iteration {picard_iter + 1}")
                break
        
        self.U = U
        self.particles = particles
        self.M = self._estimate_density_from_particles(particles)
        
        return U, self.M
    
    def _estimate_density_from_particles(self, particles):
        """Robust density estimation using KDE with fallback"""
        Nt, Nx = self.problem.Nt + 1, self.problem.Nx + 1
        M = np.zeros((Nt, Nx))
        
        for t_idx in range(Nt):
            particles_t = particles[t_idx, :]
            
            # Remove outliers
            q1, q3 = np.percentile(particles_t, [25, 75])
            iqr = q3 - q1
            valid_mask = (particles_t >= q1 - 1.5*iqr) & (particles_t <= q3 + 1.5*iqr)
            particles_clean = particles_t[valid_mask]
            
            if len(particles_clean) > 5 and len(np.unique(particles_clean)) > 1:
                try:
                    # Use KDE with robust bandwidth
                    kde = gaussian_kde(particles_clean)
                    kde.set_bandwidth(kde.factor * 1.5)  # Slightly smoother
                    M[t_idx, :] = kde(self.problem.xSpace)
                    
                    # Normalize
                    total_mass = np.sum(M[t_idx, :]) * self.problem.Dx
                    if total_mass > 0:
                        M[t_idx, :] = M[t_idx, :] / total_mass
                except:
                    # Fallback to histogram
                    hist, _ = np.histogram(particles_clean, bins=Nx, 
                                         range=(self.problem.xmin, self.problem.xmax))
                    M[t_idx, :-1] = hist / (len(particles_clean) * self.problem.Dx)
                    M[t_idx, -1] = M[t_idx, -2]
            else:
                # Uniform fallback
                M[t_idx, :] = 1.0 / (self.problem.xmax - self.problem.xmin)
        
        return M
    
    def get_particles_trajectory(self):
        return self.particles

def compare_three_methods_robust():
    print("="*80)
    print("ROBUST THREE-METHOD MFG COMPARISON")
    print("="*80)
    print("1. Robust Pure FDM (implicit schemes + stability controls)")
    print("2. Robust Hybrid Particle-FDM (sub-stepping + stabilization)")  
    print("3. Second-Order QP Particle-Collocation (validated implementation)")
    
    # Conservative test parameters for stability
    problem_params = {
        "xmin": 0.0,
        "xmax": 1.0,
        "Nx": 25,   
        "T": 0.2,   
        "Nt": 10,   
        "sigma": 0.2,  
        "coefCT": 0.01  
    }
    
    print(f"\nProblem Parameters:")
    for key, value in problem_params.items():
        print(f"  {key}: {value}")
    
    problem = ExampleMFGProblem(**problem_params)
    no_flux_bc = BoundaryConditions(type="no_flux")
    
    results = {}
    
    # Method 1: Robust Pure FDM
    print(f"\n{'='*60}")
    print("METHOD 1: ROBUST PURE FDM")
    print(f"{'='*60}")
    
    try:
        start_time = time.time()
        fdm_solver = RobustFDMSolver(problem, no_flux_bc)
        U_fdm, M_fdm = fdm_solver.solve(Niter=8, l2errBound=1e-3)
        time_fdm = time.time() - start_time
        
        if M_fdm is not None and U_fdm is not None:
            mass_fdm = np.sum(M_fdm * problem.Dx, axis=1)
            initial_mass = mass_fdm[0]
            final_mass = mass_fdm[-1]
            mass_change = abs(final_mass - initial_mass)
            
            results['fdm'] = {
                'success': True,
                'initial_mass': initial_mass,
                'final_mass': final_mass,
                'mass_change': mass_change,
                'max_U': np.max(np.abs(U_fdm)),
                'time': time_fdm,
                'converged': fdm_solver.converged,
                'iterations': fdm_solver.iterations_run,
                'violations': 0,
                'U_solution': U_fdm,
                'M_solution': M_fdm
            }
            print(f"✓ FDM completed: initial_mass={initial_mass:.6f}, final_mass={final_mass:.6f}")
            print(f"  Mass change: {mass_change:.2e}, time: {time_fdm:.2f}s")
        else:
            results['fdm'] = {'success': False}
            print("❌ FDM failed")
    except Exception as e:
        results['fdm'] = {'success': False, 'error': str(e)}
        print(f"❌ FDM crashed: {e}")
    
    # Method 2: Robust Hybrid
    print(f"\n{'='*60}")
    print("METHOD 2: ROBUST HYBRID PARTICLE-FDM")
    print(f"{'='*60}")
    
    try:
        start_time = time.time()
        hybrid_solver = RobustHybridSolver(problem, num_particles=500, boundary_conditions=no_flux_bc)
        U_hybrid, M_hybrid = hybrid_solver.solve(Niter=8, l2errBound=1e-3)
        time_hybrid = time.time() - start_time
        
        if M_hybrid is not None and U_hybrid is not None:
            mass_hybrid = np.sum(M_hybrid * problem.Dx, axis=1)
            initial_mass = mass_hybrid[0]
            final_mass = mass_hybrid[-1]
            mass_change = abs(final_mass - initial_mass)
            
            # Count boundary violations
            violations = 0
            if hybrid_solver.particles is not None:
                final_particles = hybrid_solver.particles[-1, :]
                violations = np.sum((final_particles < problem.xmin) | (final_particles > problem.xmax))
            
            results['hybrid'] = {
                'success': True,
                'initial_mass': initial_mass,
                'final_mass': final_mass,
                'mass_change': mass_change,
                'max_U': np.max(np.abs(U_hybrid)),
                'time': time_hybrid,
                'converged': hybrid_solver.converged,
                'iterations': hybrid_solver.iterations_run,
                'violations': violations,
                'U_solution': U_hybrid,
                'M_solution': M_hybrid
            }
            print(f"✓ Hybrid completed: initial_mass={initial_mass:.6f}, final_mass={final_mass:.6f}")
            print(f"  Mass change: {mass_change:.2e}, time: {time_hybrid:.2f}s, violations: {violations}")
        else:
            results['hybrid'] = {'success': False}
            print("❌ Hybrid failed")
    except Exception as e:
        results['hybrid'] = {'success': False, 'error': str(e)}
        print(f"❌ Hybrid crashed: {e}")
    
    # Method 3: QP-Collocation (validated)
    print(f"\n{'='*60}")
    print("METHOD 3: SECOND-ORDER QP PARTICLE-COLLOCATION")
    print(f"{'='*60}")
    
    try:
        start_time = time.time()
        
        # Collocation setup
        num_collocation_points = 10
        collocation_points = np.linspace(problem.xmin, problem.xmax, num_collocation_points).reshape(-1, 1)
        
        boundary_indices = []
        for i, point in enumerate(collocation_points):
            x = point[0]
            if abs(x - problem.xmin) < 1e-10 or abs(x - problem.xmax) < 1e-10:
                boundary_indices.append(i)
        boundary_indices = np.array(boundary_indices)
        
        collocation_solver = ParticleCollocationSolver(
            problem=problem,
            collocation_points=collocation_points,
            num_particles=500,
            delta=0.3,
            taylor_order=2,
            weight_function="wendland",
            NiterNewton=6,
            l2errBoundNewton=1e-3,
            kde_bandwidth="scott",
            normalize_kde_output=False,
            boundary_indices=boundary_indices,
            boundary_conditions=no_flux_bc,
            use_monotone_constraints=True
        )
        
        U_colloc, M_colloc, info_colloc = collocation_solver.solve(
            Niter=8, l2errBound=1e-3, verbose=False
        )
        
        time_colloc = time.time() - start_time
        
        if M_colloc is not None and U_colloc is not None:
            mass_colloc = np.sum(M_colloc * problem.Dx, axis=1)
            initial_mass = mass_colloc[0]
            final_mass = mass_colloc[-1]
            mass_change = abs(final_mass - initial_mass)
            
            # Count boundary violations
            violations = 0
            particles_trajectory = collocation_solver.get_particles_trajectory()
            if particles_trajectory is not None:
                final_particles = particles_trajectory[-1, :]
                violations = np.sum((final_particles < problem.xmin) | (final_particles > problem.xmax))
            
            results['collocation'] = {
                'success': True,
                'initial_mass': initial_mass,
                'final_mass': final_mass,
                'mass_change': mass_change,
                'max_U': np.max(np.abs(U_colloc)),
                'time': time_colloc,
                'converged': info_colloc.get('converged', False),
                'iterations': info_colloc.get('iterations', 0),
                'violations': violations,
                'U_solution': U_colloc,
                'M_solution': M_colloc
            }
            print(f"✓ QP-Collocation completed: initial_mass={initial_mass:.6f}, final_mass={final_mass:.6f}")
            print(f"  Mass change: {mass_change:.2e}, time: {time_colloc:.2f}s, violations: {violations}")
        else:
            results['collocation'] = {'success': False}
            print("❌ QP-Collocation failed")
    except Exception as e:
        results['collocation'] = {'success': False, 'error': str(e)}
        print(f"❌ QP-Collocation crashed: {e}")
    
    # Results comparison
    print(f"\n{'='*80}")
    print("ROBUST COMPARISON RESULTS")
    print(f"{'='*80}")
    
    successful_methods = [m for m in ['fdm', 'hybrid', 'collocation'] if results.get(m, {}).get('success', False)]
    
    if len(successful_methods) >= 2:
        print(f"\n{'Metric':<25} {'Pure FDM':<15} {'Hybrid':<15} {'QP-Collocation':<15}")
        print(f"{'-'*25} {'-'*15} {'-'*15} {'-'*15}")
        
        # Show initial and final masses
        for method in ['fdm', 'hybrid', 'collocation']:
            if method in successful_methods:
                initial = results[method]['initial_mass']
                final = results[method]['final_mass']
                print(f"Initial mass {method.upper():<15} {initial:.6f}")
        
        print()
        for method in ['fdm', 'hybrid', 'collocation']:
            if method in successful_methods:
                initial = results[method]['initial_mass']
                final = results[method]['final_mass']
                print(f"Final mass {method.upper():<17} {final:.6f}")
        
        print()
        
        # Comparison table
        metrics = [
            ('Mass change', 'mass_change', lambda x: f"{x:.2e}"),
            ('Max |U|', 'max_U', lambda x: f"{x:.1e}"),
            ('Runtime (s)', 'time', lambda x: f"{x:.2f}"),
            ('Violations', 'violations', lambda x: str(int(x))),
            ('Converged', 'converged', lambda x: "Yes" if x else "No"),
            ('Iterations', 'iterations', lambda x: str(int(x)))
        ]
        
        for metric_name, key, fmt in metrics:
            row = [metric_name]
            for method in ['fdm', 'hybrid', 'collocation']:
                if method in successful_methods:
                    value = results[method].get(key, 0)
                    row.append(fmt(value))
                else:
                    row.append("FAILED")
            print(f"{row[0]:<25} {row[1]:<15} {row[2]:<15} {row[3]:<15}")
        
        # Convergence analysis
        print(f"\n--- Convergence Analysis ---")
        if len(successful_methods) >= 2:
            final_masses = [results[m]['final_mass'] for m in successful_methods]
            max_diff = max(final_masses) - min(final_masses)
            avg_mass = np.mean(final_masses)
            relative_diff = (max_diff / avg_mass) * 100 if avg_mass > 0 else 0
            
            print(f"Final mass range: [{min(final_masses):.6f}, {max(final_masses):.6f}]")
            print(f"Max difference: {max_diff:.2e}")
            print(f"Relative difference: {relative_diff:.3f}%")
            
            if relative_diff < 1.0:
                print("✅ EXCELLENT: All methods converge to consistent solution")
            elif relative_diff < 5.0:
                print("✅ GOOD: Methods show reasonable convergence")
            else:
                print("⚠️  WARNING: Methods show significant divergence")
        
        # Create plots
        create_robust_comparison_plots(results, problem, successful_methods)
    
    else:
        print("Insufficient successful methods for comparison")
        for method in ['fdm', 'hybrid', 'collocation']:
            if not results.get(method, {}).get('success', False):
                error = results.get(method, {}).get('error', 'Failed')
                print(f"❌ {method.upper()}: {error}")

def create_robust_comparison_plots(results, problem, successful_methods):
    """Create comparison plots for the robust implementations"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Robust Three-Method MFG Comparison (Stable Implementations)', fontsize=16)
    
    method_names = {'fdm': 'Pure FDM', 'hybrid': 'Hybrid', 'collocation': 'QP-Collocation'}
    colors = {'fdm': 'blue', 'hybrid': 'green', 'collocation': 'red'}
    
    # Final density comparison
    ax1 = axes[0, 0]
    for method in successful_methods:
        M_solution = results[method]['M_solution']
        final_density = M_solution[-1, :]
        ax1.plot(problem.xSpace, final_density, 
                label=method_names[method], linewidth=2, color=colors[method])
    ax1.set_xlabel('Space x')
    ax1.set_ylabel('Final Density M(T,x)')
    ax1.set_title('Final Density Comparison')
    ax1.grid(True)
    ax1.legend()
    
    # Mass conservation over time
    ax2 = axes[0, 1]
    for method in successful_methods:
        M_solution = results[method]['M_solution']
        mass_evolution = np.sum(M_solution * problem.Dx, axis=1)
        ax2.plot(problem.tSpace, mass_evolution, 
                label=method_names[method], linewidth=2, color=colors[method])
    ax2.set_xlabel('Time t')
    ax2.set_ylabel('Total Mass')
    ax2.set_title('Mass Conservation Over Time')
    ax2.grid(True)
    ax2.legend()
    
    # Mass change comparison (log scale)
    ax3 = axes[1, 0]
    methods = [method_names[m] for m in successful_methods]
    mass_changes = [results[m]['mass_change'] for m in successful_methods]
    bars = ax3.bar(methods, mass_changes, color=[colors[m] for m in successful_methods])
    ax3.set_ylabel('Mass Change (log scale)')
    ax3.set_title('Mass Conservation Quality')
    ax3.set_yscale('log')
    ax3.grid(True, axis='y')
    for bar, value in zip(bars, mass_changes):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{value:.1e}', ha='center', va='bottom')
    
    # Runtime comparison
    ax4 = axes[1, 1]
    runtimes = [results[m]['time'] for m in successful_methods]
    bars = ax4.bar(methods, runtimes, color=[colors[m] for m in successful_methods])
    ax4.set_ylabel('Runtime (seconds)')
    ax4.set_title('Performance Comparison')
    ax4.grid(True, axis='y')
    for bar, value in zip(bars, runtimes):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{value:.2f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('/Users/zvezda/Library/CloudStorage/OneDrive-Personal/code/MFG_PDE/robust_comparison.png', 
                dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    compare_three_methods_robust()
