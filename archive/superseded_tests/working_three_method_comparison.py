#!/usr/bin/env python3
"""
Working comparison of three MFG solver methods:
1. Pure FDM (simplified implementation)
2. Hybrid Particle-FDM (simplified implementation)
3. Second-Order QP Particle-Collocation (our full implementation)
"""

import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

# Add the main package to path
sys.path.insert(0, '/Users/zvezda/Library/CloudStorage/OneDrive-Personal/code/MFG_PDE')

from mfg_pde.alg.base_mfg_solver import MFGSolver
from mfg_pde.alg.particle_collocation_solver import ParticleCollocationSolver
from mfg_pde.core.boundaries import BoundaryConditions
from mfg_pde.core.mfg_problem import ExampleMFGProblem


# Simplified Pure FDM implementation for comparison
class SimpleFDMSolver(MFGSolver):
    """Simplified Pure FDM solver for comparison purposes"""

    def __init__(self, problem, boundary_conditions=None):
        super().__init__(problem)
        self.hjb_method_name = "FDM"
        self.boundary_conditions = boundary_conditions
        self.U = None
        self.M = None
        self.iterations_run = 0
        self.converged = False

    def get_results(self):
        """Required abstract method implementation"""
        if self.U is None or self.M is None:
            raise ValueError("No solution available. Call solve() first.")
        return self.U, self.M

    def solve(self, Niter=10, l2errBound=1e-3):
        """Simplified solve method that demonstrates FDM characteristics"""
        print("Running Simplified Pure FDM solver...")

        Nx = self.problem.Nx
        Nt = self.problem.Nt

        # Initialize solutions
        U = np.zeros((Nt + 1, Nx + 1))
        M = np.zeros((Nt + 1, Nx + 1))

        # Set terminal condition for U
        U[Nt, :] = 0.0  # Terminal condition

        # Set initial condition for M (normalized Gaussian)
        x_center = (self.problem.xmin + self.problem.xmax) / 2
        x_std = 0.1
        for i in range(Nx + 1):
            x = self.problem.xSpace[i]
            M[0, i] = np.exp(-0.5 * ((x - x_center) / x_std) ** 2)

        # Normalize initial density
        M[0, :] = M[0, :] / (np.sum(M[0, :]) * self.problem.Dx)

        # Picard iteration (simplified)
        for picard_iter in range(Niter):
            U_prev = U.copy()
            M_prev = M.copy()

            # Simplified backward solve for U (HJB)
            for t_idx in range(Nt - 1, -1, -1):
                for i in range(1, Nx):
                    # Simple finite difference approximation
                    if t_idx < Nt - 1:
                        U_t = (U[t_idx + 1, i] - U[t_idx, i]) / self.problem.Dt
                    else:
                        U_t = 0

                    # Spatial derivatives (central difference)
                    U_x = (U[t_idx, i + 1] - U[t_idx, i - 1]) / (2 * self.problem.Dx)
                    U_xx = (U[t_idx, i + 1] - 2 * U[t_idx, i] + U[t_idx, i - 1]) / (self.problem.Dx**2)

                    # Simplified HJB equation with stability controls
                    hamiltonian = 0.5 * U_x**2
                    coupling = self.problem.coefCT * M[t_idx, i]

                    # Strong stability controls
                    hamiltonian = (
                        min(abs(hamiltonian), 1.0) * np.sign(hamiltonian) if not np.isnan(hamiltonian) else 0.0
                    )
                    coupling = min(abs(coupling), 1.0) * np.sign(coupling) if not np.isnan(coupling) else 0.0
                    U_xx = min(abs(U_xx), 10.0) * np.sign(U_xx) if not np.isnan(U_xx) else 0.0

                    # Update with very conservative time stepping
                    dt_factor = 0.1  # Very conservative
                    update = (
                        -dt_factor * self.problem.Dt * (hamiltonian + coupling - 0.5 * self.problem.sigma**2 * U_xx)
                    )
                    update = min(abs(update), 0.1) * np.sign(update)  # Limit update size
                    U[t_idx, i] = U[t_idx + 1, i] + update

                # No-flux boundary conditions for U
                U[t_idx, 0] = U[t_idx, 1]
                U[t_idx, Nx] = U[t_idx, Nx - 1]

            # Simplified forward solve for M (FP)
            for t_idx in range(Nt):
                for i in range(1, Nx):
                    if t_idx > 0:
                        # Compute drift from U
                        U_x = (U[t_idx, i + 1] - U[t_idx, i - 1]) / (2 * self.problem.Dx)

                        # Simple finite difference for FP
                        M_x = (M[t_idx - 1, i + 1] - M[t_idx - 1, i - 1]) / (2 * self.problem.Dx)
                        M_xx = (M[t_idx - 1, i + 1] - 2 * M[t_idx - 1, i] + M[t_idx - 1, i - 1]) / (self.problem.Dx**2)

                        # FP equation (explicit Euler) with stability controls
                        drift_term = U_x * M_x
                        diffusion_term = 0.5 * self.problem.sigma**2 * M_xx

                        # Apply conservative time stepping and ensure positivity
                        dt_factor = 0.3  # Very conservative for FP
                        M_new = M[t_idx - 1, i] + dt_factor * self.problem.Dt * (drift_term + diffusion_term)
                        M[t_idx, i] = max(M_new, 0.0)  # Ensure positivity

                # No-flux boundary conditions for M
                if t_idx > 0:
                    M[t_idx, 0] = M[t_idx, 1]
                    M[t_idx, Nx] = M[t_idx, Nx - 1]

            # Check convergence
            U_error = np.linalg.norm(U - U_prev) / max(np.linalg.norm(U_prev), 1e-10)
            M_error = np.linalg.norm(M - M_prev) / max(np.linalg.norm(M_prev), 1e-10)
            total_error = max(U_error, M_error)

            self.iterations_run = picard_iter + 1

            if total_error < l2errBound:
                self.converged = True
                break

        self.U = U
        self.M = M

        return U, M


# Simplified Hybrid solver
class SimpleHybridSolver(MFGSolver):
    """Simplified Hybrid Particle-FDM solver for comparison"""

    def __init__(self, problem, num_particles=400, boundary_conditions=None):
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
        """Required abstract method implementation"""
        if self.U is None or self.M is None:
            raise ValueError("No solution available. Call solve() first.")
        return self.U, self.M

    def solve(self, Niter=10, l2errBound=1e-3):
        """Simplified hybrid solve: FDM for HJB + particles for FP"""
        print("Running Simplified Hybrid Particle-FDM solver...")

        Nx = self.problem.Nx
        Nt = self.problem.Nt

        # Initialize HJB solution on grid
        U = np.zeros((Nt + 1, Nx + 1))
        U[Nt, :] = 0.0  # Terminal condition

        # Initialize particles
        particles = np.zeros((Nt + 1, self.num_particles))

        # Initial particle distribution (Gaussian)
        x_center = (self.problem.xmin + self.problem.xmax) / 2
        x_std = 0.1
        particles[0, :] = np.random.normal(x_center, x_std, self.num_particles)

        # Ensure particles are in domain
        particles[0, :] = np.clip(particles[0, :], self.problem.xmin, self.problem.xmax)

        # Picard iteration
        for picard_iter in range(Niter):
            U_prev = U.copy()
            particles_prev = particles.copy()

            # Step 1: Solve HJB with FDM using current particle density
            M_from_particles = self._estimate_density_from_particles(particles)

            # Simplified HJB solve (similar to Pure FDM)
            for t_idx in range(Nt - 1, -1, -1):
                for i in range(1, Nx):
                    if t_idx < Nt - 1:
                        U_t = (U[t_idx + 1, i] - U[t_idx, i]) / self.problem.Dt
                    else:
                        U_t = 0

                    U_x = (U[t_idx, i + 1] - U[t_idx, i - 1]) / (2 * self.problem.Dx)
                    U_xx = (U[t_idx, i + 1] - 2 * U[t_idx, i] + U[t_idx, i - 1]) / (self.problem.Dx**2)

                    hamiltonian = 0.5 * U_x**2
                    coupling = self.problem.coefCT * M_from_particles[t_idx, i]

                    U[t_idx, i] = U[t_idx + 1, i] - self.problem.Dt * (
                        hamiltonian + coupling - 0.5 * self.problem.sigma**2 * U_xx
                    )

                # No-flux boundary conditions
                U[t_idx, 0] = U[t_idx, 1]
                U[t_idx, Nx] = U[t_idx, Nx - 1]

            # Step 2: Evolve particles using updated control
            for t_idx in range(1, Nt + 1):
                for p in range(self.num_particles):
                    # Get particle position
                    x_p = particles[t_idx - 1, p]

                    # Interpolate control field at particle position
                    i = int((x_p - self.problem.xmin) / self.problem.Dx)
                    i = max(0, min(Nx - 1, i))

                    if i < Nx:
                        U_x_p = (U[t_idx - 1, i + 1] - U[t_idx - 1, i]) / self.problem.Dx
                        # Add stability control
                        U_x_p = min(abs(U_x_p), 5.0) * np.sign(U_x_p) if not np.isnan(U_x_p) else 0.0
                    else:
                        U_x_p = 0

                    # Particle SDE: dX = -U_x dt + sigma dW with stability controls
                    drift = -U_x_p * 0.1  # Reduced drift strength
                    noise = self.problem.sigma * np.random.normal(0, np.sqrt(self.problem.Dt))

                    new_pos = x_p + drift * self.problem.Dt + noise

                    # Reflect at boundaries (no-flux)
                    if new_pos < self.problem.xmin:
                        new_pos = 2 * self.problem.xmin - new_pos
                    elif new_pos > self.problem.xmax:
                        new_pos = 2 * self.problem.xmax - new_pos

                    particles[t_idx, p] = new_pos

            # Check convergence
            U_error = np.linalg.norm(U - U_prev) / max(np.linalg.norm(U_prev), 1e-10)
            particles_error = np.linalg.norm(particles - particles_prev) / max(np.linalg.norm(particles_prev), 1e-10)
            total_error = max(U_error, particles_error)

            self.iterations_run = picard_iter + 1

            if total_error < l2errBound:
                self.converged = True
                break

        self.U = U
        self.particles = particles
        self.M = self._estimate_density_from_particles(particles)

        return U, self.M

    def _estimate_density_from_particles(self, particles):
        """Estimate density on grid from particle positions using simple binning"""
        Nt, Nx = self.problem.Nt + 1, self.problem.Nx + 1
        M = np.zeros((Nt, Nx))

        for t_idx in range(Nt):
            hist, _ = np.histogram(particles[t_idx, :], bins=Nx - 1, range=(self.problem.xmin, self.problem.xmax))
            # Properly handle histogram to grid mapping
            M[t_idx, : Nx - 1] = hist / (self.num_particles * self.problem.Dx)
            M[t_idx, -1] = M[t_idx, -2]  # Boundary handling

        return M

    def get_particles_trajectory(self):
        return self.particles


def compare_three_methods():
    print("=" * 80)
    print("WORKING THREE-METHOD MFG COMPARISON")
    print("=" * 80)
    print("1. Simplified Pure FDM")
    print("2. Simplified Hybrid Particle-FDM")
    print("3. Second-Order QP Particle-Collocation (full implementation)")

    # Optimized test parameters with 1000 particles for quality comparison
    problem_params = {
        "xmin": 0.0,
        "xmax": 1.0,
        "Nx": 30,  # Balanced resolution
        "T": 0.3,  # Moderate time horizon
        "Nt": 12,  # Adequate time steps
        "sigma": 0.4,  # Higher diffusion for stability with longer time
        "coefCT": 0.01,  # Light coupling for stability
    }

    print(f"\nProblem Parameters:")
    for key, value in problem_params.items():
        print(f"  {key}: {value}")

    problem = ExampleMFGProblem(**problem_params)
    no_flux_bc = BoundaryConditions(type="no_flux")

    results = {}

    # Method 1: Simplified Pure FDM
    print(f"\n{'='*60}")
    print("METHOD 1: SIMPLIFIED PURE FDM")
    print(f"{'='*60}")

    try:
        start_time = time.time()
        fdm_solver = SimpleFDMSolver(problem, no_flux_bc)
        U_fdm, M_fdm = fdm_solver.solve(Niter=10, l2errBound=1e-3)
        time_fdm = time.time() - start_time

        if M_fdm is not None and U_fdm is not None:
            mass_fdm = np.sum(M_fdm * problem.Dx, axis=1)
            results['fdm'] = {
                'success': True,
                'mass_change': abs(mass_fdm[-1] - mass_fdm[0]),
                'max_U': np.max(np.abs(U_fdm)),
                'time': time_fdm,
                'converged': fdm_solver.converged,
                'iterations': fdm_solver.iterations_run,
                'violations': 0,  # Grid-based, no particle violations
                'U_solution': U_fdm,
                'M_solution': M_fdm,
            }
            print(f"✓ FDM completed: mass_change={results['fdm']['mass_change']:.2e}, time={time_fdm:.2f}s")
        else:
            results['fdm'] = {'success': False}
            print("❌ FDM failed")
    except Exception as e:
        results['fdm'] = {'success': False, 'error': str(e)}
        print(f"❌ FDM crashed: {e}")

    # Method 2: Simplified Hybrid
    print(f"\n{'='*60}")
    print("METHOD 2: SIMPLIFIED HYBRID PARTICLE-FDM")
    print(f"{'='*60}")

    try:
        start_time = time.time()
        hybrid_solver = SimpleHybridSolver(problem, num_particles=1000, boundary_conditions=no_flux_bc)
        U_hybrid, M_hybrid = hybrid_solver.solve(Niter=10, l2errBound=1e-3)
        time_hybrid = time.time() - start_time

        if M_hybrid is not None and U_hybrid is not None:
            mass_hybrid = np.sum(M_hybrid * problem.Dx, axis=1)

            # Count boundary violations
            violations = 0
            if hybrid_solver.particles is not None:
                final_particles = hybrid_solver.particles[-1, :]
                violations = np.sum((final_particles < problem.xmin) | (final_particles > problem.xmax))

            results['hybrid'] = {
                'success': True,
                'mass_change': abs(mass_hybrid[-1] - mass_hybrid[0]),
                'max_U': np.max(np.abs(U_hybrid)),
                'time': time_hybrid,
                'converged': hybrid_solver.converged,
                'iterations': hybrid_solver.iterations_run,
                'violations': violations,
                'U_solution': U_hybrid,
                'M_solution': M_hybrid,
            }
            print(
                f"✓ Hybrid completed: mass_change={results['hybrid']['mass_change']:.2e}, time={time_hybrid:.2f}s, violations={violations}"
            )
        else:
            results['hybrid'] = {'success': False}
            print("❌ Hybrid failed")
    except Exception as e:
        results['hybrid'] = {'success': False, 'error': str(e)}
        print(f"❌ Hybrid crashed: {e}")

    # Method 3: Second-Order QP Particle-Collocation
    print(f"\n{'='*60}")
    print("METHOD 3: SECOND-ORDER QP PARTICLE-COLLOCATION")
    print(f"{'='*60}")

    try:
        start_time = time.time()

        # Collocation setup - better resolution for meaningful comparison
        num_collocation_points = 15
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
            num_particles=1000,  # Same as hybrid for fair comparison
            delta=0.4,
            taylor_order=2,  # SECOND-ORDER
            weight_function="wendland",
            NiterNewton=8,  # Reduced for faster execution
            l2errBoundNewton=2e-4,  # Relaxed tolerance
            kde_bandwidth="scott",
            normalize_kde_output=False,
            boundary_indices=boundary_indices,
            boundary_conditions=no_flux_bc,
            use_monotone_constraints=True,  # QP CONSTRAINTS
        )

        U_colloc, M_colloc, info_colloc = collocation_solver.solve(Niter=10, l2errBound=1e-3, verbose=False)

        time_colloc = time.time() - start_time

        if M_colloc is not None and U_colloc is not None:
            mass_colloc = np.sum(M_colloc * problem.Dx, axis=1)

            # Count boundary violations
            violations = 0
            particles_trajectory = collocation_solver.get_particles_trajectory()
            if particles_trajectory is not None:
                final_particles = particles_trajectory[-1, :]
                violations = np.sum((final_particles < problem.xmin) | (final_particles > problem.xmax))

            results['collocation'] = {
                'success': True,
                'mass_change': abs(mass_colloc[-1] - mass_colloc[0]),
                'max_U': np.max(np.abs(U_colloc)),
                'time': time_colloc,
                'converged': info_colloc.get('converged', False),
                'iterations': info_colloc.get('iterations', 0),
                'violations': violations,
                'U_solution': U_colloc,
                'M_solution': M_colloc,
            }
            print(
                f"✓ QP-Collocation completed: mass_change={results['collocation']['mass_change']:.2e}, time={time_colloc:.2f}s, violations={violations}"
            )
        else:
            results['collocation'] = {'success': False}
            print("❌ QP-Collocation failed")
    except Exception as e:
        results['collocation'] = {'success': False, 'error': str(e)}
        print(f"❌ QP-Collocation crashed: {e}")

    # Results comparison
    print(f"\n{'='*80}")
    print("COMPARISON RESULTS")
    print(f"{'='*80}")

    successful_methods = [m for m in ['fdm', 'hybrid', 'collocation'] if results.get(m, {}).get('success', False)]

    if len(successful_methods) >= 2:
        print(f"\n{'Metric':<20} {'Pure FDM':<15} {'Hybrid':<15} {'QP-Collocation':<15}")
        print(f"{'-'*20} {'-'*15} {'-'*15} {'-'*15}")

        # Comparison table
        metrics = [
            ('Mass change', 'mass_change', lambda x: f"{x:.2e}"),
            ('Max |U|', 'max_U', lambda x: f"{x:.1e}"),
            ('Runtime (s)', 'time', lambda x: f"{x:.2f}"),
            ('Violations', 'violations', lambda x: str(int(x))),
            ('Converged', 'converged', lambda x: "Yes" if x else "No"),
        ]

        for metric_name, key, fmt in metrics:
            row = [metric_name]
            for method in ['fdm', 'hybrid', 'collocation']:
                if method in successful_methods:
                    value = results[method].get(key, 0)
                    row.append(fmt(value))
                else:
                    row.append("FAILED")
            print(f"{row[0]:<20} {row[1]:<15} {row[2]:<15} {row[3]:<15}")

        # Winner analysis
        print(f"\n--- Performance Analysis ---")
        if len(successful_methods) == 3:
            best_mass = min(successful_methods, key=lambda m: results[m]['mass_change'])
            fastest = min(successful_methods, key=lambda m: results[m]['time'])
            best_boundary = min(successful_methods, key=lambda m: results[m]['violations'])

            print(f"🏆 Best mass conservation: {best_mass.upper()}")
            print(f"🏆 Fastest execution: {fastest.upper()}")
            print(f"🏆 Best boundary compliance: {best_boundary.upper()}")

            # Performance overhead
            fdm_time = results['fdm']['time']
            hybrid_overhead = (results['hybrid']['time'] - fdm_time) / fdm_time * 100
            colloc_overhead = (results['collocation']['time'] - fdm_time) / fdm_time * 100

            print(f"\nPerformance overhead vs Pure FDM:")
            print(f"  Hybrid: {hybrid_overhead:+.1f}%")
            print(f"  QP-Collocation: {colloc_overhead:+.1f}%")

        # Create plots
        create_comparison_plots(results, problem, successful_methods)

    else:
        print("Insufficient successful methods for comparison")
        for method in ['fdm', 'hybrid', 'collocation']:
            if not results.get(method, {}).get('success', False):
                error = results.get(method, {}).get('error', 'Failed')
                print(f"❌ {method.upper()}: {error}")


def create_comparison_plots(results, problem, successful_methods):
    """Create comparison plots"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Three-Method MFG Comparison', fontsize=16)

    method_names = {'fdm': 'Pure FDM', 'hybrid': 'Hybrid', 'collocation': 'QP-Collocation'}
    colors = {'fdm': 'blue', 'hybrid': 'green', 'collocation': 'red'}

    # Final density comparison
    ax1 = axes[0, 0]
    for method in successful_methods:
        M_solution = results[method]['M_solution']
        final_density = M_solution[-1, :]
        ax1.plot(problem.xSpace, final_density, label=method_names[method], linewidth=2, color=colors[method])
    ax1.set_xlabel('Space x')
    ax1.set_ylabel('Final Density M(T,x)')
    ax1.set_title('Final Density Comparison')
    ax1.grid(True)
    ax1.legend()

    # Mass conservation
    ax2 = axes[0, 1]
    for method in successful_methods:
        M_solution = results[method]['M_solution']
        mass_evolution = np.sum(M_solution * problem.Dx, axis=1)
        ax2.plot(problem.tSpace, mass_evolution, label=method_names[method], linewidth=2, color=colors[method])
    ax2.set_xlabel('Time t')
    ax2.set_ylabel('Total Mass')
    ax2.set_title('Mass Conservation')
    ax2.grid(True)
    ax2.legend()

    # Performance comparison
    ax3 = axes[1, 0]
    methods = [method_names[m] for m in successful_methods]
    runtimes = [results[m]['time'] for m in successful_methods]
    bars = ax3.bar(methods, runtimes, color=[colors[m] for m in successful_methods])
    ax3.set_ylabel('Runtime (seconds)')
    ax3.set_title('Performance Comparison')
    for bar, value in zip(bars, runtimes):
        ax3.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f'{value:.2f}s', ha='center', va='bottom')

    # Mass change comparison
    ax4 = axes[1, 1]
    mass_changes = [results[m]['mass_change'] for m in successful_methods]
    bars = ax4.bar(methods, mass_changes, color=[colors[m] for m in successful_methods])
    ax4.set_ylabel('Mass Change')
    ax4.set_title('Mass Conservation Quality')
    ax4.set_yscale('log')
    for bar, value in zip(bars, mass_changes):
        ax4.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f'{value:.1e}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    compare_three_methods()
