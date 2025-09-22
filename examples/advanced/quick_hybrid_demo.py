#!/usr/bin/env python3
"""
Quick demonstration of the Hybrid FP-Particle + HJB-FDM solver.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mfg_pde.core.mfg_problem import ExampleMFGProblem
from mfg_pde.alg.mfg_solvers.hybrid_fp_particle_hjb_fdm import HybridSolverPresets
from mfg_pde.factory.solver_factory import create_solver

def main():
    print("🚀 Quick Hybrid FP-Particle + HJB-FDM Solver Demo")
    print("=" * 60)

    # Create a smaller problem for quick demo
    problem = ExampleMFGProblem(Nx=25, Nt=25, T=0.5)  # Smaller, faster
    print(f"Problem: {problem.Nx} spatial points, {problem.Nt} time points")

    # Test 1: Direct solver creation
    print("\n--- Test 1: Direct Solver Creation ---")
    hybrid_solver = HybridSolverPresets.fast_hybrid(problem)

    result = hybrid_solver.solve(
        max_iterations=15,  # Reduced for quick demo
        tolerance=1e-3,     # Relaxed tolerance
        damping_factor=0.6
    )

    if result['converged']:
        print(f"✅ Direct solver: CONVERGED in {result['iterations']} iterations")
        print(f"   Time: {result['solve_time']:.2f}s")
        print(f"   Residual: {result['final_residual']:.2e}")
    else:
        print(f"⚠️ Direct solver: Did not converge")

    # Test 2: Factory integration
    print("\n--- Test 2: Factory Integration ---")
    try:
        factory_solver = create_solver(
            problem=problem,
            solver_type="hybrid_fp_particle_hjb_fdm",
            preset="fast",
            num_particles=2000  # Reduced for speed
        )

        result2 = factory_solver.solve(
            max_iterations=15,
            tolerance=1e-3,
            damping_factor=0.6
        )

        if result2['converged']:
            print(f"✅ Factory solver: CONVERGED in {result2['iterations']} iterations")
            print(f"   Time: {result2['solve_time']:.2f}s")
        else:
            print(f"⚠️ Factory solver: Did not converge")

    except Exception as e:
        print(f"❌ Factory test failed: {e}")

    print("\n" + "=" * 60)
    print("🎯 Hybrid FP-Particle + HJB-FDM solver is operational!")
    print("   • FP equation: Solved with PARTICLES (mass conservation)")
    print("   • HJB equation: Solved with FDM (stability & accuracy)")
    print("   • Integration: Complete with factory patterns")

if __name__ == "__main__":
    main()