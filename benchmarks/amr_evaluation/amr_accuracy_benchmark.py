#!/usr/bin/env python3
"""
AMR Accuracy Benchmarking Suite

This module focuses specifically on solution accuracy comparisons between
AMR-enhanced solvers and uniform grid solvers. Uses manufactured solutions
and high-resolution reference solutions for precise error quantification.

Accuracy Metrics:
- L2 norm errors for U and M
- Maximum pointwise errors
- Convergence rates under refinement
- Error distribution analysis
- Solution feature preservation
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# MFG_PDE imports
from mfg_pde import ExampleMFGProblem
from mfg_pde.factory import create_amr_solver, create_solver
from mfg_pde.geometry import Domain1D, dirichlet_bc, periodic_bc


@dataclass
class AccuracyResult:
    """Container for accuracy benchmark results."""

    problem_name: str
    solver_name: str
    grid_size: Tuple[int, ...]

    # Error metrics
    l2_error_u: float
    l2_error_m: float
    max_error_u: float
    max_error_m: float
    h1_error_u: float = None  # Gradient error

    # Mesh information
    total_elements: int
    effective_resolution: float  # sqrt(elements) for comparison
    refinement_levels: int = 0

    # Computational cost
    solve_time: float
    dofs_per_second: float  # Degrees of freedom processed per second


class AMRAccuracyBenchmark:
    """Specialized accuracy benchmarking for AMR methods."""

    def __init__(self, output_dir: str = "accuracy_benchmarks"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results: List[AccuracyResult] = []

        print("AMR Accuracy Benchmarking Suite")
        print("=" * 50)

    def manufactured_solution_1d(self, x: np.ndarray, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Manufactured solution for 1D MFG with known analytical form.

        U(x,t) = exp(-t) * sin(π*x) * cos(π*x)
        M(x,t) = (1 + 0.5*sin(2*π*x)) * exp(-0.5*t) / Z(t)

        where Z(t) is chosen to ensure ∫M dx = 1
        """
        # Value function with spatial variation
        U = np.exp(-t) * np.sin(np.pi * x) * np.cos(np.pi * x)

        # Density with different spatial structure
        M_unnorm = (1 + 0.5 * np.sin(2 * np.pi * x)) * np.exp(-0.5 * t)

        # Normalize to ensure ∫M dx = 1
        dx = x[1] - x[0] if len(x) > 1 else 1.0
        total_mass = np.trapz(M_unnorm, dx=dx)
        M = M_unnorm / total_mass

        return U, M

    def create_manufactured_problem_1d(self, nx: int = 64) -> ExampleMFGProblem:
        """Create 1D problem with manufactured solution for accuracy testing."""

        domain = Domain1D(0.0, 1.0, periodic_bc())

        # Create problem with moderate parameters
        problem = ExampleMFGProblem(
            T=0.5,  # Shorter time for clearer comparison
            xmin=0.0,
            xmax=1.0,
            Nx=nx,
            Nt=20,
            sigma=0.1,  # Moderate diffusion
            coefCT=0.5,  # Moderate congestion
        )

        problem.domain = domain
        problem.dimension = 1

        # Add reference solution method
        def get_reference_solution(t_val=0.5):
            x = np.linspace(0, 1, 256)  # High resolution reference
            return self.manufactured_solution_1d(x, t_val)

        problem.get_reference_solution = get_reference_solution

        return problem

    def create_sharp_feature_problem_1d(self, nx: int = 64) -> ExampleMFGProblem:
        """Create 1D problem with sharp features to test AMR effectiveness."""

        domain = Domain1D(-2.0, 2.0, periodic_bc())

        problem = ExampleMFGProblem(
            T=1.0,
            xmin=-2.0,
            xmax=2.0,
            Nx=nx,
            Nt=30,
            sigma=0.02,  # Small diffusion → sharp features
            coefCT=3.0,  # Strong congestion → localized dynamics
        )

        problem.domain = domain
        problem.dimension = 1

        return problem

    def compute_solution_errors(
        self,
        computed_solution: Tuple[np.ndarray, np.ndarray],
        reference_solution: Tuple[np.ndarray, np.ndarray],
        x_grid: np.ndarray,
    ) -> Dict[str, float]:
        """Compute comprehensive error metrics between solutions."""

        U_comp, M_comp = computed_solution
        U_ref, M_ref = reference_solution

        # Interpolate to common grid if needed
        if U_comp.shape != U_ref.shape:
            print(f"Warning: Solution shapes don't match: {U_comp.shape} vs {U_ref.shape}")
            # For now, return NaN errors
            return {
                'l2_error_u': np.nan,
                'l2_error_m': np.nan,
                'max_error_u': np.nan,
                'max_error_m': np.nan,
                'h1_error_u': np.nan,
            }

        # L2 errors
        dx = x_grid[1] - x_grid[0] if len(x_grid) > 1 else 1.0

        l2_error_u = np.sqrt(np.trapz((U_comp - U_ref) ** 2, dx=dx))
        l2_error_m = np.sqrt(np.trapz((M_comp - M_ref) ** 2, dx=dx))

        # Maximum errors
        max_error_u = np.max(np.abs(U_comp - U_ref))
        max_error_m = np.max(np.abs(M_comp - M_ref))

        # H1 error (including gradient)
        if len(U_comp) > 2:
            du_comp = np.gradient(U_comp, dx)
            du_ref = np.gradient(U_ref, dx)
            h1_error_u = np.sqrt(np.trapz((U_comp - U_ref) ** 2, dx=dx) + np.trapz((du_comp - du_ref) ** 2, dx=dx))
        else:
            h1_error_u = l2_error_u

        return {
            'l2_error_u': l2_error_u,
            'l2_error_m': l2_error_m,
            'max_error_u': max_error_u,
            'max_error_m': max_error_m,
            'h1_error_u': h1_error_u,
        }

    def benchmark_manufactured_solution(self):
        """Benchmark accuracy using manufactured solutions."""

        print("\n🎯 Manufactured Solution Accuracy Test")
        print("-" * 40)

        # Test different grid sizes
        grid_sizes = [32, 64, 128]

        for nx in grid_sizes:
            print(f"\nTesting grid size: {nx}")

            # Create problem
            problem = self.create_manufactured_problem_1d(nx)

            # Get reference solution
            U_ref, M_ref = problem.get_reference_solution()
            x_ref = np.linspace(0, 1, len(U_ref))

            # Test configurations
            configs = [
                {
                    'name': f'Uniform-{nx}',
                    'create_solver': lambda p: create_solver(p, solver_type='fixed_point', preset='accurate'),
                    'amr_enabled': False,
                },
                {
                    'name': f'AMR-{nx}',
                    'create_solver': lambda p: create_amr_solver(
                        p, base_solver_type='fixed_point', error_threshold=1e-5, max_levels=4
                    ),
                    'amr_enabled': True,
                },
            ]

            for config in configs:
                print(f"  Testing {config['name']}...")

                # Create solver and solve
                solver = config['create_solver'](problem)

                start_time = time.perf_counter()
                result = solver.solve(max_iterations=100, tolerance=1e-7, verbose=False)
                solve_time = time.perf_counter() - start_time

                # Extract solution
                if isinstance(result, dict):
                    U = result['U']
                    M = result['M']
                    total_elements = result.get('mesh_statistics', {}).get('total_intervals', nx)
                    refinement_levels = result.get('mesh_statistics', {}).get('max_level', 0)
                else:
                    U, M, info = result
                    total_elements = nx
                    refinement_levels = 0

                # Compute errors (using problem grid points)
                x_comp = np.linspace(problem.xmin, problem.xmax, len(U))

                # Interpolate reference to computational grid
                U_ref_interp = np.interp(x_comp, x_ref, U_ref)
                M_ref_interp = np.interp(x_comp, x_ref, M_ref)

                errors = self.compute_solution_errors((U, M), (U_ref_interp, M_ref_interp), x_comp)

                # Store result
                accuracy_result = AccuracyResult(
                    problem_name=f"Manufactured_1D_N{nx}",
                    solver_name=config['name'],
                    grid_size=(nx,),
                    l2_error_u=errors['l2_error_u'],
                    l2_error_m=errors['l2_error_m'],
                    max_error_u=errors['max_error_u'],
                    max_error_m=errors['max_error_m'],
                    h1_error_u=errors['h1_error_u'],
                    total_elements=total_elements,
                    effective_resolution=np.sqrt(total_elements),
                    refinement_levels=refinement_levels,
                    solve_time=solve_time,
                    dofs_per_second=total_elements / solve_time,
                )

                self.results.append(accuracy_result)

                # Print summary
                print(f"    L2 Error U: {errors['l2_error_u']:.2e}")
                print(f"    L2 Error M: {errors['l2_error_m']:.2e}")
                print(f"    Elements: {total_elements}")
                print(f"    Time: {solve_time:.3f}s")

    def benchmark_sharp_features(self):
        """Benchmark AMR effectiveness on problems with sharp features."""

        print("\n🔥 Sharp Features Accuracy Test")
        print("-" * 40)

        # Create problem with sharp features
        problem = self.create_sharp_feature_problem_1d(64)

        # Compute high-resolution reference
        print("Computing high-resolution reference...")
        ref_problem = self.create_sharp_feature_problem_1d(512)  # Much finer grid
        ref_solver = create_solver(ref_problem, solver_type='fixed_point', preset='accurate')
        ref_result = ref_solver.solve(max_iterations=150, tolerance=1e-8, verbose=False)

        if isinstance(ref_result, dict):
            U_ref, M_ref = ref_result['U'], ref_result['M']
        else:
            U_ref, M_ref = ref_result[0], ref_result[1]

        x_ref = np.linspace(ref_problem.xmin, ref_problem.xmax, len(U_ref))

        print("Testing different AMR configurations...")

        # Test different AMR aggressiveness levels
        configs = [
            {
                'name': 'Uniform-64',
                'create_solver': lambda p: create_solver(p, solver_type='fixed_point', preset='accurate'),
                'amr_enabled': False,
            },
            {
                'name': 'AMR-Conservative',
                'create_solver': lambda p: create_amr_solver(
                    p, base_solver_type='fixed_point', error_threshold=1e-3, max_levels=3
                ),
                'amr_enabled': True,
            },
            {
                'name': 'AMR-Aggressive',
                'create_solver': lambda p: create_amr_solver(
                    p, base_solver_type='fixed_point', error_threshold=1e-4, max_levels=5
                ),
                'amr_enabled': True,
            },
            {
                'name': 'AMR-VeryAggressive',
                'create_solver': lambda p: create_amr_solver(
                    p, base_solver_type='fixed_point', error_threshold=1e-5, max_levels=6
                ),
                'amr_enabled': True,
            },
        ]

        for config in configs:
            print(f"  Testing {config['name']}...")

            solver = config['create_solver'](problem)

            start_time = time.perf_counter()
            result = solver.solve(max_iterations=100, tolerance=1e-6, verbose=False)
            solve_time = time.perf_counter() - start_time

            # Extract solution
            if isinstance(result, dict):
                U = result['U']
                M = result['M']
                total_elements = result.get('mesh_statistics', {}).get('total_intervals', 64)
                refinement_levels = result.get('mesh_statistics', {}).get('max_level', 0)
            else:
                U, M, info = result
                total_elements = 64
                refinement_levels = 0

            # Compute errors against high-resolution reference
            x_comp = np.linspace(problem.xmin, problem.xmax, len(U))

            U_ref_interp = np.interp(x_comp, x_ref, U_ref)
            M_ref_interp = np.interp(x_comp, x_ref, M_ref)

            errors = self.compute_solution_errors((U, M), (U_ref_interp, M_ref_interp), x_comp)

            # Store result
            accuracy_result = AccuracyResult(
                problem_name="Sharp_Features_1D",
                solver_name=config['name'],
                grid_size=(64,),
                l2_error_u=errors['l2_error_u'],
                l2_error_m=errors['l2_error_m'],
                max_error_u=errors['max_error_u'],
                max_error_m=errors['max_error_m'],
                h1_error_u=errors['h1_error_u'],
                total_elements=total_elements,
                effective_resolution=np.sqrt(total_elements),
                refinement_levels=refinement_levels,
                solve_time=solve_time,
                dofs_per_second=total_elements / solve_time,
            )

            self.results.append(accuracy_result)

            # Print summary
            print(f"    L2 Error U: {errors['l2_error_u']:.2e}")
            print(f"    L2 Error M: {errors['l2_error_m']:.2e}")
            print(f"    Elements: {total_elements} (efficiency: {total_elements/64:.2f})")
            print(f"    Max level: {refinement_levels}")
            print(f"    Time: {solve_time:.3f}s")

    def generate_convergence_plots(self):
        """Generate convergence rate plots."""

        print("\nGenerating convergence analysis plots...")

        if not self.results:
            print("No results to plot!")
            return

        try:
            # Manufactured solution convergence
            manufactured_results = [r for r in self.results if "Manufactured" in r.problem_name]

            if manufactured_results:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

                # Separate uniform and AMR results
                uniform_results = [r for r in manufactured_results if not r.solver_name.startswith('AMR')]
                amr_results = [r for r in manufactured_results if r.solver_name.startswith('AMR')]

                # Plot L2 error vs effective resolution
                if uniform_results:
                    resolutions = [r.effective_resolution for r in uniform_results]
                    errors_u = [r.l2_error_u for r in uniform_results]
                    ax1.loglog(resolutions, errors_u, 'b-o', label='Uniform Grid', markersize=6)

                if amr_results:
                    resolutions = [r.effective_resolution for r in amr_results]
                    errors_u = [r.l2_error_u for r in amr_results]
                    ax1.loglog(resolutions, errors_u, 'r-s', label='AMR', markersize=6)

                ax1.set_xlabel('Effective Resolution (√elements)')
                ax1.set_ylabel('L2 Error in U')
                ax1.set_title('Convergence Rate - Value Function')
                ax1.grid(True, alpha=0.3)
                ax1.legend()

                # Plot computational efficiency (error vs time)
                if uniform_results:
                    times = [r.solve_time for r in uniform_results]
                    errors_u = [r.l2_error_u for r in uniform_results]
                    ax2.loglog(times, errors_u, 'b-o', label='Uniform Grid', markersize=6)

                if amr_results:
                    times = [r.solve_time for r in amr_results]
                    errors_u = [r.l2_error_u for r in amr_results]
                    ax2.loglog(times, errors_u, 'r-s', label='AMR', markersize=6)

                ax2.set_xlabel('Solve Time (s)')
                ax2.set_ylabel('L2 Error in U')
                ax2.set_title('Computational Efficiency')
                ax2.grid(True, alpha=0.3)
                ax2.legend()

                plt.tight_layout()
                plt.savefig(self.output_dir / 'convergence_analysis.png', dpi=150, bbox_inches='tight')
                print(f"Convergence plots saved to {self.output_dir / 'convergence_analysis.png'}")

                # Show plot if possible
                try:
                    plt.show()
                except Exception:
                    print("Display not available, plot saved to file.")

        except Exception as e:
            print(f"Error generating plots: {e}")

    def generate_accuracy_report(self):
        """Generate comprehensive accuracy analysis report."""

        report_file = self.output_dir / "accuracy_report.md"

        with open(report_file, 'w') as f:
            f.write("# AMR Accuracy Benchmark Report\n\n")
            f.write(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}  \n\n")

            # Accuracy comparison table
            f.write("## Accuracy Comparison\n\n")
            f.write("| Problem | Solver | Elements | L2 Error U | L2 Error M | Max Error U | Time (s) |\n")
            f.write("|---------|---------|----------|------------|------------|-------------|----------|\n")

            for result in self.results:
                f.write(
                    f"| {result.problem_name} | {result.solver_name} | "
                    f"{result.total_elements} | {result.l2_error_u:.2e} | "
                    f"{result.l2_error_m:.2e} | {result.max_error_u:.2e} | "
                    f"{result.solve_time:.3f} |\n"
                )

            # Analysis by problem type
            f.write("\n## Analysis by Problem Type\n\n")

            problem_types = set(r.problem_name for r in self.results)

            for problem_type in problem_types:
                f.write(f"### {problem_type}\n\n")

                problem_results = [r for r in self.results if r.problem_name == problem_type]

                # Find best accuracy
                best_accuracy_u = min(r.l2_error_u for r in problem_results if not np.isnan(r.l2_error_u))
                best_solver_u = next(r.solver_name for r in problem_results if r.l2_error_u == best_accuracy_u)

                f.write(f"**Best Accuracy (L2 Error U)**: {best_accuracy_u:.2e} ({best_solver_u})  \n")

                # Find most efficient (best error/time ratio)
                efficiency_ratios = []
                for r in problem_results:
                    if not np.isnan(r.l2_error_u) and r.solve_time > 0:
                        efficiency_ratios.append((r.solver_name, r.l2_error_u / r.solve_time))

                if efficiency_ratios:
                    best_efficiency = min(efficiency_ratios, key=lambda x: x[1])
                    f.write(f"**Most Efficient**: {best_efficiency[0]} (error/time: {best_efficiency[1]:.2e})  \n")

                f.write("\n")

            # AMR effectiveness summary
            amr_results = [r for r in self.results if 'AMR' in r.solver_name]
            uniform_results = [r for r in self.results if 'Uniform' in r.solver_name]

            if amr_results and uniform_results:
                f.write("## AMR Effectiveness Summary\n\n")

                total_amr_elements = sum(r.total_elements for r in amr_results)
                total_uniform_elements = sum(r.total_elements for r in uniform_results)

                avg_efficiency = total_amr_elements / total_uniform_elements if total_uniform_elements > 0 else 1.0

                f.write(
                    f"**Average Mesh Efficiency**: {avg_efficiency:.3f} "
                    f"({100*(1-avg_efficiency):.1f}% element reduction)  \n"
                )

                # Accuracy comparison
                avg_amr_error = np.mean([r.l2_error_u for r in amr_results if not np.isnan(r.l2_error_u)])
                avg_uniform_error = np.mean([r.l2_error_u for r in uniform_results if not np.isnan(r.l2_error_u)])

                if not np.isnan(avg_amr_error) and not np.isnan(avg_uniform_error):
                    accuracy_ratio = avg_uniform_error / avg_amr_error
                    f.write(f"**Average Accuracy Improvement**: {accuracy_ratio:.2f}x better  \n")

                f.write("\n**Recommendation**: ")
                if avg_efficiency < 0.8 and accuracy_ratio > 1.2:
                    f.write("AMR shows significant benefits in both efficiency and accuracy.")
                elif avg_efficiency < 0.8:
                    f.write("AMR shows good mesh efficiency benefits.")
                elif accuracy_ratio > 1.2:
                    f.write("AMR shows good accuracy benefits.")
                else:
                    f.write("AMR benefits are marginal for these problem types.")

        print(f"Accuracy report generated: {report_file}")

    def run_comprehensive_accuracy_benchmark(self):
        """Run the complete accuracy benchmarking suite."""

        print("Starting Comprehensive AMR Accuracy Benchmark")
        print("=" * 60)

        # Run manufactured solution benchmarks
        self.benchmark_manufactured_solution()

        # Run sharp features benchmarks
        self.benchmark_sharp_features()

        # Generate analysis
        self.generate_convergence_plots()
        self.generate_accuracy_report()

        print(f"\n✅ Accuracy benchmarking complete!")
        print(f"Results saved to: {self.output_dir}")

        # Print summary
        if self.results:
            print(f"\nSummary:")
            print(f"  Total benchmarks: {len(self.results)}")

            amr_results = [r for r in self.results if 'AMR' in r.solver_name]
            if amr_results:
                avg_elements = np.mean([r.total_elements for r in amr_results])
                avg_levels = np.mean([r.refinement_levels for r in amr_results])
                print(f"  Average AMR elements: {avg_elements:.0f}")
                print(f"  Average refinement levels: {avg_levels:.1f}")


def main():
    """Run the AMR accuracy benchmark suite."""
    benchmark_suite = AMRAccuracyBenchmark()
    benchmark_suite.run_comprehensive_accuracy_benchmark()


if __name__ == "__main__":
    main()
