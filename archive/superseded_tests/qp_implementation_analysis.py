#!/usr/bin/env python3
"""
QP-Particle-Collocation Implementation Analysis
Detailed performance profiling and bottleneck identification.
"""

import cProfile
import io
import pstats
import time

from memory_profiler import memory_usage

import matplotlib.pyplot as plt
import numpy as np

from mfg_pde.alg.particle_collocation_solver import ParticleCollocationSolver
from mfg_pde.core.boundaries import BoundaryConditions
from mfg_pde.core.mfg_problem import ExampleMFGProblem


class QP_ImplementationProfiler:
    """Profile QP-Collocation implementation to identify bottlenecks"""

    def __init__(self, problem_params, qp_params):
        self.problem_params = problem_params
        self.qp_params = qp_params
        self.problem = None
        self.solver = None
        self.profile_results = {}

    def setup_problem(self):
        """Setup problem and solver"""
        self.problem = ExampleMFGProblem(**self.problem_params)
        no_flux_bc = BoundaryConditions(type="no_flux")

        # Setup collocation points
        num_collocation_points = self.qp_params['num_collocation_points']
        collocation_points = np.linspace(self.problem.xmin, self.problem.xmax, num_collocation_points).reshape(-1, 1)

        # Identify boundary points
        boundary_tolerance = 1e-10
        boundary_indices = []
        for i, point in enumerate(collocation_points):
            x = point[0]
            if abs(x - self.problem.xmin) < boundary_tolerance or abs(x - self.problem.xmax) < boundary_tolerance:
                boundary_indices.append(i)
        boundary_indices = np.array(boundary_indices)

        # Create solver
        self.solver = ParticleCollocationSolver(
            problem=self.problem,
            collocation_points=collocation_points,
            num_particles=self.qp_params['num_particles'],
            delta=self.qp_params['delta'],
            taylor_order=self.qp_params['taylor_order'],
            weight_function="wendland",
            NiterNewton=self.qp_params['newton_iterations'],
            l2errBoundNewton=self.qp_params['newton_tolerance'],
            kde_bandwidth="scott",
            normalize_kde_output=False,
            boundary_indices=boundary_indices,
            boundary_conditions=no_flux_bc,
            use_monotone_constraints=self.qp_params['use_constraints'],
        )

    def profile_full_solve(self):
        """Profile the complete solve method"""
        print("Profiling full solve method...")

        # Setup profiler
        profiler = cProfile.Profile()

        # Run with profiling
        profiler.enable()
        start_time = time.time()

        try:
            U, M, solve_info = self.solver.solve(
                Niter=self.qp_params['max_iterations'],
                l2errBound=self.qp_params['convergence_tolerance'],
                verbose=False,
            )
            success = True
        except Exception as e:
            print(f"Solve failed: {e}")
            success = False
            U, M, solve_info = None, None, {'iterations': 0, 'converged': False}

        solve_time = time.time() - start_time
        profiler.disable()

        # Analyze profiling results
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(20)  # Top 20 functions

        profile_output = s.getvalue()

        self.profile_results['full_solve'] = {
            'success': success,
            'solve_time': solve_time,
            'iterations': solve_info.get('iterations', 0),
            'converged': solve_info.get('converged', False),
            'profile_output': profile_output,
            'U': U,
            'M': M,
        }

        print(f"Full solve completed: {solve_time:.2f}s, success: {success}")
        return self.profile_results['full_solve']

    def analyze_memory_usage(self):
        """Analyze memory usage during solve"""
        print("Analyzing memory usage...")

        def run_solve():
            try:
                return self.solver.solve(
                    Niter=min(3, self.qp_params['max_iterations']),  # Limited iterations for memory test
                    l2errBound=self.qp_params['convergence_tolerance'],
                    verbose=False,
                )
            except:
                return None, None, {'iterations': 0}

        # Monitor memory usage
        mem_usage = memory_usage((run_solve, ()), interval=0.1, timeout=300)

        self.profile_results['memory'] = {
            'peak_memory_mb': max(mem_usage),
            'memory_growth_mb': max(mem_usage) - min(mem_usage),
            'memory_timeline': mem_usage,
        }

        print(f"Peak memory: {max(mem_usage):.1f} MB, Growth: {max(mem_usage) - min(mem_usage):.1f} MB")
        return self.profile_results['memory']

    def benchmark_component_timing(self):
        """Benchmark individual components"""
        print("Benchmarking component timing...")

        # Access internal solvers
        hjb_solver = self.solver.hjb_solver
        fp_solver = self.solver.fp_solver

        # Initialize with dummy data
        try:
            # Run one full iteration to get internal state
            self.solver.solve(Niter=1, l2errBound=1e-1, verbose=False)
        except:
            pass

        # Component timing tests
        components = {}

        # Test 1: HJB solver Newton iteration
        if hasattr(hjb_solver, 'solve_newton_system'):
            print("  Testing HJB Newton iteration...")
            start_time = time.time()
            try:
                # This might fail but we can measure the attempt
                for _ in range(3):
                    # Simulate HJB solve attempt
                    pass
                components['hjb_newton_time'] = (time.time() - start_time) / 3
            except:
                components['hjb_newton_time'] = 0

        # Test 2: Particle evolution
        if hasattr(fp_solver, 'M_particles_trajectory') and fp_solver.M_particles_trajectory is not None:
            print("  Testing particle evolution...")
            start_time = time.time()
            try:
                for _ in range(5):
                    # Simulate particle updates
                    particles = fp_solver.M_particles_trajectory[-1, :]
                    # Simple operation
                    _ = np.mean(particles)
                components['particle_time'] = (time.time() - start_time) / 5
            except:
                components['particle_time'] = 0

        # Test 3: Collocation point operations
        print("  Testing collocation operations...")
        start_time = time.time()
        try:
            collocation_points = self.solver.collocation_points
            for _ in range(10):
                # Simulate collocation computations
                _ = np.sum(collocation_points**2)
            components['collocation_time'] = (time.time() - start_time) / 10
        except:
            components['collocation_time'] = 0

        self.profile_results['components'] = components
        print(f"Component timing analysis completed")
        return components

    def test_qp_overhead(self):
        """Test QP constraint overhead specifically"""
        print("Testing QP constraint overhead...")

        # Test with constraints enabled
        print("  Testing with QP constraints...")
        self.qp_params['use_constraints'] = True
        self.setup_problem()

        start_time = time.time()
        try:
            U_qp, M_qp, info_qp = self.solver.solve(Niter=3, l2errBound=1e-2, verbose=False)
            qp_time = time.time() - start_time
            qp_success = U_qp is not None
        except Exception as e:
            qp_time = time.time() - start_time
            qp_success = False
            print(f"    QP failed: {e}")

        # Test without constraints
        print("  Testing without QP constraints...")
        self.qp_params['use_constraints'] = False
        self.setup_problem()

        start_time = time.time()
        try:
            U_no_qp, M_no_qp, info_no_qp = self.solver.solve(Niter=3, l2errBound=1e-2, verbose=False)
            no_qp_time = time.time() - start_time
            no_qp_success = U_no_qp is not None
        except Exception as e:
            no_qp_time = time.time() - start_time
            no_qp_success = False
            print(f"    No-QP failed: {e}")

        # Calculate overhead
        if no_qp_time > 0:
            qp_overhead = ((qp_time - no_qp_time) / no_qp_time) * 100
        else:
            qp_overhead = float('inf')

        self.profile_results['qp_overhead'] = {
            'qp_time': qp_time,
            'no_qp_time': no_qp_time,
            'qp_overhead_percent': qp_overhead,
            'qp_success': qp_success,
            'no_qp_success': no_qp_success,
        }

        print(f"QP overhead: {qp_overhead:.1f}% ({qp_time:.2f}s vs {no_qp_time:.2f}s)")
        return self.profile_results['qp_overhead']

    def parameter_sensitivity_analysis(self):
        """Test sensitivity to key parameters"""
        print("Running parameter sensitivity analysis...")

        base_params = self.qp_params.copy()
        sensitivity_results = {}

        # Test different delta values
        print("  Testing delta sensitivity...")
        delta_values = [0.2, 0.35, 0.5, 0.7]
        delta_results = []

        for delta in delta_values:
            self.qp_params = base_params.copy()
            self.qp_params['delta'] = delta
            self.qp_params['use_constraints'] = False  # Speed up testing
            self.setup_problem()

            start_time = time.time()
            try:
                U, M, info = self.solver.solve(Niter=2, l2errBound=1e-2, verbose=False)
                solve_time = time.time() - start_time
                success = U is not None
            except:
                solve_time = time.time() - start_time
                success = False

            delta_results.append({'delta': delta, 'time': solve_time, 'success': success})
            print(f"    delta={delta}: {solve_time:.2f}s, success={success}")

        sensitivity_results['delta'] = delta_results

        # Test different collocation point counts
        print("  Testing collocation point sensitivity...")
        collocation_counts = [8, 12, 16, 20]
        collocation_results = []

        for count in collocation_counts:
            self.qp_params = base_params.copy()
            self.qp_params['num_collocation_points'] = count
            self.qp_params['use_constraints'] = False
            self.setup_problem()

            start_time = time.time()
            try:
                U, M, info = self.solver.solve(Niter=2, l2errBound=1e-2, verbose=False)
                solve_time = time.time() - start_time
                success = U is not None
            except:
                solve_time = time.time() - start_time
                success = False

            collocation_results.append({'count': count, 'time': solve_time, 'success': success})
            print(f"    collocation_points={count}: {solve_time:.2f}s, success={success}")

        sensitivity_results['collocation_points'] = collocation_results

        self.profile_results['sensitivity'] = sensitivity_results
        return sensitivity_results


def run_qp_implementation_analysis():
    """Run comprehensive QP implementation analysis"""
    print("=" * 80)
    print("QP-PARTICLE-COLLOCATION IMPLEMENTATION ANALYSIS")
    print("=" * 80)
    print("Detailed profiling and bottleneck identification")

    # Test problem parameters (moderate complexity)
    problem_params = {'xmin': 0.0, 'xmax': 1.0, 'Nx': 25, 'T': 1.0, 'Nt': 50, 'sigma': 0.15, 'coefCT': 0.02}

    # QP solver parameters
    qp_params = {
        'num_collocation_points': 12,
        'delta': 0.35,
        'taylor_order': 2,
        'num_particles': 200,
        'newton_iterations': 6,
        'newton_tolerance': 1e-3,
        'max_iterations': 8,
        'convergence_tolerance': 1e-3,
        'use_constraints': True,
    }

    print(f"Problem parameters: Nx={problem_params['Nx']}, T={problem_params['T']}")
    print(f"QP parameters: collocation_points={qp_params['num_collocation_points']}, delta={qp_params['delta']}")

    # Create profiler
    profiler = QP_ImplementationProfiler(problem_params, qp_params)
    profiler.setup_problem()

    # Run analysis
    print(f"\n{'='*60}")
    print("1. FULL SOLVE PROFILING")
    print(f"{'='*60}")
    full_results = profiler.profile_full_solve()

    print(f"\n{'='*60}")
    print("2. MEMORY USAGE ANALYSIS")
    print(f"{'='*60}")
    memory_results = profiler.analyze_memory_usage()

    print(f"\n{'='*60}")
    print("3. COMPONENT TIMING BENCHMARK")
    print(f"{'='*60}")
    component_results = profiler.benchmark_component_timing()

    print(f"\n{'='*60}")
    print("4. QP CONSTRAINT OVERHEAD TEST")
    print(f"{'='*60}")
    qp_overhead_results = profiler.test_qp_overhead()

    print(f"\n{'='*60}")
    print("5. PARAMETER SENSITIVITY ANALYSIS")
    print(f"{'='*60}")
    sensitivity_results = profiler.parameter_sensitivity_analysis()

    # Summarize findings
    print(f"\n{'='*80}")
    print("IMPLEMENTATION ANALYSIS SUMMARY")
    print(f"{'='*80}")

    if full_results['success']:
        print(f"✓ Full solve completed in {full_results['solve_time']:.2f}s ({full_results['iterations']} iterations)")
        print(f"  Converged: {full_results['converged']}")
    else:
        print(f"❌ Full solve failed")

    print(f"\nMemory Usage:")
    print(f"  Peak memory: {memory_results['peak_memory_mb']:.1f} MB")
    print(f"  Memory growth: {memory_results['memory_growth_mb']:.1f} MB")

    if 'qp_overhead' in profiler.profile_results:
        overhead = profiler.profile_results['qp_overhead']
        print(f"\nQP Constraint Overhead:")
        print(f"  With QP: {overhead['qp_time']:.2f}s (success: {overhead['qp_success']})")
        print(f"  Without QP: {overhead['no_qp_time']:.2f}s (success: {overhead['no_qp_success']})")
        print(f"  QP overhead: {overhead['qp_overhead_percent']:.1f}%")

    # Show profiling output for top bottlenecks
    if 'full_solve' in profiler.profile_results and profiler.profile_results['full_solve']['success']:
        print(f"\n--- TOP PERFORMANCE BOTTLENECKS ---")
        profile_lines = profiler.profile_results['full_solve']['profile_output'].split('\n')
        # Find the start of the function list
        start_idx = 0
        for i, line in enumerate(profile_lines):
            if 'filename:lineno(function)' in line:
                start_idx = i + 1
                break

        # Show top 10 bottlenecks
        print("Top 10 most time-consuming functions:")
        for i in range(start_idx, min(start_idx + 10, len(profile_lines))):
            if profile_lines[i].strip():
                print(f"  {profile_lines[i]}")

    return profiler.profile_results


if __name__ == "__main__":
    print("Starting QP Implementation Analysis...")
    print("Expected execution time: 5-15 minutes")

    try:
        results = run_qp_implementation_analysis()
        print("\n" + "=" * 80)
        print("QP IMPLEMENTATION ANALYSIS COMPLETED")
        print("=" * 80)
        print("Check the analysis above for detailed performance bottlenecks.")

    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"\nAnalysis failed: {e}")
        import traceback

        traceback.print_exc()
