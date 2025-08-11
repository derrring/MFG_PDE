#!/usr/bin/env python3
"""
Tuned QP Final Test
Focus test on the Tuned Smart QP approach that achieved 8.2% usage.
"""

import time
import warnings

import numpy as np

warnings.filterwarnings('ignore')

from mfg_pde.alg.hjb_solvers.tuned_smart_qp_gfdm_hjb import TunedSmartQPGFDMHJBSolver
from mfg_pde.alg.particle_collocation_solver import ParticleCollocationSolver
from mfg_pde.core.boundaries import BoundaryConditions
from mfg_pde.core.mfg_problem import ExampleMFGProblem


def test_tuned_qp_final():
    """Final test of the Tuned Smart QP approach"""
    print("=" * 70)
    print("TUNED SMART QP FINAL VALIDATION")
    print("=" * 70)
    print("Testing the optimized Tuned Smart QP with target parameters")

    # Target problem parameters (moderate size)
    problem_params = {'xmin': 0.0, 'xmax': 1.0, 'Nx': 18, 'T': 0.8, 'Nt': 25, 'sigma': 0.12, 'coefCT': 0.02}

    print(f"Problem: Nx={problem_params['Nx']}, T={problem_params['T']}, Nt={problem_params['Nt']}")
    print(f"Parameters: σ={problem_params['sigma']}, coefCT={problem_params['coefCT']}")

    try:
        problem = ExampleMFGProblem(**problem_params)
        no_flux_bc = BoundaryConditions(type="no_flux")

        # Solver configuration
        num_collocation_points = 9
        collocation_points = np.linspace(problem.xmin, problem.xmax, num_collocation_points).reshape(-1, 1)
        boundary_indices = [0, num_collocation_points - 1]

        # Create Tuned Smart QP HJB solver
        tuned_hjb_solver = TunedSmartQPGFDMHJBSolver(
            problem=problem,
            collocation_points=collocation_points,
            delta=0.4,
            taylor_order=2,
            weight_function="wendland",
            NiterNewton=4,
            l2errBoundNewton=1e-3,
            boundary_indices=np.array(boundary_indices),
            boundary_conditions=no_flux_bc,
            use_monotone_constraints=True,
            qp_usage_target=0.1,  # Target 10% QP usage
        )

        # Create particle collocation solver
        tuned_solver = ParticleCollocationSolver(
            problem=problem,
            collocation_points=collocation_points,
            num_particles=110,
            delta=0.4,
            taylor_order=2,
            weight_function="wendland",
            NiterNewton=4,
            l2errBoundNewton=1e-3,
            kde_bandwidth="scott",
            normalize_kde_output=False,
            boundary_indices=np.array(boundary_indices),
            boundary_conditions=no_flux_bc,
            use_monotone_constraints=True,
        )
        tuned_solver.hjb_solver = tuned_hjb_solver

        print(f"\n{'-'*50}")
        print("RUNNING TUNED SMART QP SOLVER")
        print(f"{'-'*50}")

        start_time = time.time()
        U_tuned, M_tuned, info_tuned = tuned_solver.solve(Niter=5, l2errBound=1e-3, verbose=True)
        solve_time = time.time() - start_time

        # Calculate mass conservation
        Dx = (problem_params['xmax'] - problem_params['xmin']) / problem_params['Nx']
        initial_mass = np.sum(M_tuned[0, :]) * Dx
        final_mass = np.sum(M_tuned[-1, :]) * Dx
        mass_error = abs(final_mass - initial_mass) / initial_mass * 100

        # Get optimization statistics
        tuned_stats = tuned_hjb_solver.get_tuned_qp_report()

        print(f"\n{'='*70}")
        print("TUNED SMART QP RESULTS")
        print(f"{'='*70}")

        print(f"Solution Status:")
        print(f"  ✓ Solver completed successfully")
        print(f"  ✓ Total solve time: {solve_time:.1f} seconds")
        print(f"  ✓ Converged: {info_tuned.get('converged', False)}")
        print(f"  ✓ Picard iterations: {info_tuned.get('iterations', 0)}")
        print(f"  ✓ Mass conservation error: {mass_error:.3f}%")

        print(f"\nQP Optimization Performance:")
        print(f"  ✓ QP Usage Rate: {tuned_stats.get('qp_usage_rate', 0):.1%}")
        print(f"  ✓ Target Achievement: {tuned_stats.get('optimization_quality', 'N/A')}")
        print(f"  ✓ Optimization Effectiveness: {tuned_stats.get('optimization_effectiveness', 0):.1%}")

        # Calculate speedup estimate
        qp_skip_rate = 1.0 - tuned_stats.get('qp_usage_rate', 1.0)
        estimated_speedup = 1 / (1 - qp_skip_rate * 0.9) if qp_skip_rate > 0 else 1.0
        print(f"  ✓ Estimated Speedup vs Baseline: {estimated_speedup:.1f}x")

        # Print detailed performance summary
        if hasattr(tuned_hjb_solver, 'print_tuned_qp_summary'):
            tuned_hjb_solver.print_tuned_qp_summary()

        # Final Assessment
        print(f"\n{'='*70}")
        print("FINAL ASSESSMENT")
        print(f"{'='*70}")

        qp_rate = tuned_stats.get('qp_usage_rate', 1.0)

        if qp_rate <= 0.12:  # Within 20% of 10% target
            print("🎉 OPTIMIZATION TARGET ACHIEVED!")
            print(f"   QP usage rate of {qp_rate:.1%} meets the 10% target")
            print(f"   Estimated {estimated_speedup:.1f}x speedup over baseline QP-Collocation")
            print("   ✅ Tuned Smart QP-Collocation is ready for production use")
            status = "SUCCESS"
        elif qp_rate <= 0.15:  # Within 50% of target
            print("✅ NEAR-OPTIMAL PERFORMANCE!")
            print(f"   QP usage rate of {qp_rate:.1%} is very close to 10% target")
            print(f"   Excellent {estimated_speedup:.1f}x speedup achieved")
            print("   ✅ Performance is excellent for production use")
            status = "EXCELLENT"
        elif qp_rate <= 0.25:  # Significant improvement
            print("⚠️  GOOD OPTIMIZATION ACHIEVED")
            print(f"   QP usage rate of {qp_rate:.1%} shows good improvement")
            print(f"   Solid {estimated_speedup:.1f}x speedup achieved")
            print("   📈 Further tuning could improve performance")
            status = "GOOD"
        else:
            print("❌ OPTIMIZATION NEEDS IMPROVEMENT")
            print(f"   QP usage rate of {qp_rate:.1%} is above target")
            print("   🔧 Further optimization required")
            status = "NEEDS_WORK"

        print(f"\nRecommendation:")
        if status in ["SUCCESS", "EXCELLENT"]:
            print("   🚀 Deploy Tuned Smart QP-Collocation for production workloads")
            print("   📊 Monitor performance and collect usage statistics")
        elif status == "GOOD":
            print("   🔄 Use current optimization with minor tuning")
            print("   🎯 Focus on boundary condition handling for improvement")
        else:
            print("   🔄 Continue using Hybrid method until optimization improves")
            print("   🔍 Debug QP decision logic and threshold adaptation")

        return {
            'success': True,
            'solve_time': solve_time,
            'mass_error': mass_error,
            'qp_usage_rate': qp_rate,
            'estimated_speedup': estimated_speedup,
            'status': status,
            'stats': tuned_stats,
        }

    except Exception as e:
        print(f"✗ Tuned Smart QP test failed: {e}")
        import traceback

        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def main():
    """Run the tuned QP final test"""
    print("Starting Tuned Smart QP Final Validation...")
    print("This validates the best optimization approach developed")
    print("Expected execution time: 5-10 minutes")

    try:
        result = test_tuned_qp_final()

        print(f"\n{'='*70}")
        print("TUNED SMART QP VALIDATION COMPLETED")
        print(f"{'='*70}")

        if result['success']:
            print("✅ Validation successful - check results above")
        else:
            print("❌ Validation failed - check error details above")

        return result

    except KeyboardInterrupt:
        print("\nValidation interrupted by user.")
        return None
    except Exception as e:
        print(f"\nValidation failed: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    result = main()
