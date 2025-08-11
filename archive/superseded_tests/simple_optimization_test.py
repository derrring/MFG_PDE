#!/usr/bin/env python3
"""
Simple Optimization Test
Quick validation of the adaptive QP activation optimization.
"""

import os
import sys
import time

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from mfg_pde.alg.hjb_solvers.optimized_gfdm_hjb_v2 import OptimizedGFDMHJBSolver
from mfg_pde.alg.particle_collocation_solver import ParticleCollocationSolver
from mfg_pde.core.boundaries import BoundaryConditions
from mfg_pde.core.mfg_problem import ExampleMFGProblem


def test_adaptive_qp_activation():
    """Test the adaptive QP activation directly"""
    print("=" * 60)
    print("SIMPLE ADAPTIVE QP ACTIVATION TEST")
    print("=" * 60)

    # Create a very simple problem
    problem_params = {'xmin': 0.0, 'xmax': 1.0, 'Nx': 15, 'T': 0.5, 'Nt': 20, 'sigma': 0.1, 'coefCT': 0.01}

    problem = ExampleMFGProblem(**problem_params)
    no_flux_bc = BoundaryConditions(type="no_flux")

    # Simple collocation setup
    num_collocation_points = 6
    collocation_points = np.linspace(problem.xmin, problem.xmax, num_collocation_points).reshape(-1, 1)

    boundary_indices = [0, num_collocation_points - 1]  # Just endpoints

    print(f"Problem setup: Nx={problem_params['Nx']}, T={problem_params['T']}")
    print(f"Collocation points: {num_collocation_points}")

    # Test 1: Create optimized solver and check initialization
    print(f"\n--- Test 1: Solver Initialization ---")
    try:
        optimized_solver = OptimizedGFDMHJBSolver(
            problem=problem,
            collocation_points=collocation_points,
            delta=0.4,
            taylor_order=2,
            NiterNewton=2,
            l2errBoundNewton=1e-3,
            boundary_indices=np.array(boundary_indices),
            boundary_conditions=no_flux_bc,
            use_monotone_constraints=True,
            qp_activation_tolerance=1e-3,
        )
        print("✓ Optimized solver created successfully")

        # Check if _needs_qp_constraints method exists
        if hasattr(optimized_solver, '_needs_qp_constraints'):
            print("✓ Adaptive QP method available")
        else:
            print("✗ Adaptive QP method missing")
            return False

    except Exception as e:
        print(f"✗ Solver creation failed: {e}")
        return False

    # Test 2: Test adaptive QP activation directly
    print(f"\n--- Test 2: Adaptive QP Activation Logic ---")
    try:
        # Test with different solution patterns
        test_solutions = [
            np.array([0.1, 0.2, 0.1, -0.1]),  # Small values - should skip QP
            np.array([1.0, 2.0, 1.5, 0.5]),  # Medium values - should skip QP
            np.array([10.0, 15.0, 20.0, 5.0]),  # Large values - should use QP
            np.array([np.inf, 1.0, 2.0, 0.5]),  # Contains inf - should use QP
            np.array([np.nan, 1.0, 2.0, 0.5]),  # Contains NaN - should use QP
        ]

        qp_needed_count = 0
        qp_skipped_count = 0

        for i, test_sol in enumerate(test_solutions):
            needs_qp = optimized_solver._needs_qp_constraints(test_sol, i)
            if needs_qp:
                qp_needed_count += 1
                status = "QP NEEDED"
            else:
                qp_skipped_count += 1
                status = "QP SKIPPED"

            print(f"  Test solution {i+1}: {status}")

        skip_rate = qp_skipped_count / len(test_solutions) * 100
        print(f"\nSkip Rate: {skip_rate:.1f}% ({qp_skipped_count}/{len(test_solutions)} skipped)")

        if qp_skipped_count > 0:
            print("✓ Adaptive QP activation is working")
        else:
            print("⚠ No QP calls were skipped - check logic")

    except Exception as e:
        print(f"✗ Adaptive QP test failed: {e}")
        return False

    # Test 3: Performance comparison with a single time step
    print(f"\n--- Test 3: Single Time Step Performance ---")
    try:
        # Create baseline solver for comparison
        baseline_solver = ParticleCollocationSolver(
            problem=problem,
            collocation_points=collocation_points,
            num_particles=50,  # Very small for speed
            delta=0.4,
            taylor_order=2,
            weight_function="wendland",
            NiterNewton=2,
            l2errBoundNewton=1e-3,
            kde_bandwidth="scott",
            normalize_kde_output=False,
            boundary_indices=np.array(boundary_indices),
            boundary_conditions=no_flux_bc,
            use_monotone_constraints=True,
        )

        # Create optimized solver
        optimized_particle_solver = ParticleCollocationSolver(
            problem=problem,
            collocation_points=collocation_points,
            num_particles=50,
            delta=0.4,
            taylor_order=2,
            weight_function="wendland",
            NiterNewton=2,
            l2errBoundNewton=1e-3,
            kde_bandwidth="scott",
            normalize_kde_output=False,
            boundary_indices=np.array(boundary_indices),
            boundary_conditions=no_flux_bc,
            use_monotone_constraints=True,
        )

        # Replace HJB solver with optimized version
        optimized_particle_solver.hjb_solver = optimized_solver

        print("Testing baseline solver...")
        start_time = time.time()
        try:
            U_baseline, M_baseline, info_baseline = baseline_solver.solve(Niter=2, l2errBound=1e-2, verbose=False)
            baseline_time = time.time() - start_time
            baseline_success = U_baseline is not None
            print(f"  Baseline: {baseline_time:.2f}s, success={baseline_success}")
        except Exception as e:
            baseline_time = time.time() - start_time
            baseline_success = False
            print(f"  Baseline failed: {e}")

        print("Testing optimized solver...")
        start_time = time.time()
        try:
            U_optimized, M_optimized, info_optimized = optimized_particle_solver.solve(
                Niter=2, l2errBound=1e-2, verbose=False
            )
            optimized_time = time.time() - start_time
            optimized_success = U_optimized is not None
            print(f"  Optimized: {optimized_time:.2f}s, success={optimized_success}")

            # Show performance stats
            if hasattr(optimized_solver, 'get_performance_report'):
                stats = optimized_solver.get_performance_report()
                print(f"  QP Activation Rate: {stats.get('qp_activation_rate', 0):.1%}")
                print(f"  QP Skip Rate: {stats.get('qp_skip_rate', 0):.1%}")

        except Exception as e:
            optimized_time = time.time() - start_time
            optimized_success = False
            print(f"  Optimized failed: {e}")

        # Calculate speedup
        if baseline_success and optimized_success and baseline_time > 0:
            speedup = baseline_time / optimized_time
            print(f"\nSpeedup: {speedup:.2f}x")

            if speedup > 1.0:
                print("✓ Optimization provides speedup")
                return True
            else:
                print("⚠ No speedup achieved")
                return False
        else:
            print("⚠ Cannot calculate speedup - one or both solvers failed")
            return False

    except Exception as e:
        print(f"✗ Performance test failed: {e}")
        return False


def main():
    """Run the simple optimization test"""
    print("Starting Simple Optimization Test...")
    print("This tests the core adaptive QP activation optimization.")

    try:
        success = test_adaptive_qp_activation()

        print(f"\n{'='*60}")
        if success:
            print("SIMPLE OPTIMIZATION TEST: SUCCESS")
            print("The adaptive QP activation optimization is working.")
        else:
            print("SIMPLE OPTIMIZATION TEST: NEEDS IMPROVEMENT")
            print("The optimization needs further tuning.")
        print(f"{'='*60}")

        return success

    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()
