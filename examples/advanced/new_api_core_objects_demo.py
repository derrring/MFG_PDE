#!/usr/bin/env python3
"""
New API Core Objects Demo - Layer 2 Interface

This example demonstrates the new Layer 2 API that provides clean object-oriented
interfaces with full customization power for advanced users.

Perfect for:
- Research requiring specific solver control
- Custom problem definitions
- Performance optimization
- Method comparison studies
"""

import time

import numpy as np

from mfg_pde import create_mfg_problem
from mfg_pde.config import accurate_config, crowd_dynamics_config
from mfg_pde.hooks import DebugHook, PerformanceHook, ProgressHook
from mfg_pde.solvers import FixedPointSolver


def main():
    print("🔧 MFG_PDE New API Demo - Core Objects Interface")
    print("=" * 55)

    # 1. PROBLEM CREATION AND CUSTOMIZATION
    print("\n1. Creating and customizing problems:")

    # Create problem with explicit control
    problem = create_mfg_problem(
        "crowd_dynamics",
        domain=(0, 8),  # 8-meter corridor
        time_horizon=3.0,  # 3 seconds
        crowd_size=400,  # 400 people
        exit_attraction=1.5,
    )  # Strong exit attraction

    print(f"📝 Created problem: {problem}")
    print(f"   Domain: {problem.get_domain_bounds()}")
    print(f"   Time horizon: {problem.get_time_horizon()}")

    # 2. SOLVER CONFIGURATION
    print("\n2. Solver configuration and presets:")

    # Use problem-specific configuration
    config = crowd_dynamics_config()
    FixedPointSolver.from_config(config)  # Example of preset creation
    print(f"🎛️ Crowd dynamics preset: {config.max_iterations} iterations, tol={config.tolerance}")

    # Chain configuration modifications
    custom_config = accurate_config().with_tolerance(1e-7).with_max_iterations(600).with_damping(0.85)

    solver_custom = FixedPointSolver.from_config(custom_config)
    print(f"⚙️ Custom config: {custom_config.max_iterations} iterations, tol={custom_config.tolerance}")

    # Direct solver creation
    solver_direct = FixedPointSolver(max_iterations=400, tolerance=1e-6, damping=0.8, backend="auto")
    print(f"🔨 Direct solver: {solver_direct.max_iterations} iterations")

    # 3. METHOD CHAINING
    print("\n3. Fluent interface with method chaining:")

    result = FixedPointSolver().with_tolerance(1e-6).with_max_iterations(300).with_damping(0.8).solve(problem)

    print(f"⛓️ Method chaining result: {result.iterations} iterations")

    # 4. HOOKS FOR MONITORING
    print("\n4. Using hooks for monitoring and debugging:")

    # Create multiple hooks
    debug_hook = DebugHook(log_level="INFO", save_intermediate=True)
    progress_hook = ProgressHook(update_frequency=10, show_eta=True)
    perf_hook = PerformanceHook(profile_memory=True)

    # Combine hooks
    from mfg_pde.hooks import HookCollection

    hooks = HookCollection([debug_hook, progress_hook, perf_hook])

    # Solve with hooks
    monitored_solver = FixedPointSolver(max_iterations=200, tolerance=1e-5)
    monitored_result = monitored_solver.solve(problem, hooks=hooks)

    print(f"📊 Monitored solve: {monitored_result.iterations} iterations")
    print(f"   Peak memory: {monitored_result.peak_memory_usage / 1024**2:.1f} MB")

    # 5. ITERATIVE SOLVING WITH STATE ACCESS
    print("\n5. Iterative solving with state access:")

    iterative_solver = FixedPointSolver(max_iterations=100)

    print("🔄 Solving iteratively with custom control:")
    for state in iterative_solver.solve_iteratively(problem):
        if state.iteration % 20 == 0:
            print(f"   Iteration {state.iteration:3d}: residual = {state.residual:.2e}")

        # Custom stopping criteria
        if state.residual < 1e-7:
            print(f"   Early stop at iteration {state.iteration} (reached target accuracy)")
            break

        # Save checkpoints
        if state.iteration % 50 == 0:
            state.save_checkpoint(f"checkpoint_{state.iteration}.pkl")

    final_result = state.to_result()
    print(f"🏁 Final result: {final_result.iterations} iterations")

    # 6. ADVANCED RESULT ANALYSIS
    print("\n6. Advanced result analysis:")

    # Rich analysis capabilities
    result = solver_custom.solve(problem)

    print("📈 Solution quality metrics:")
    print(f"   Converged: {result.converged}")
    print(f"   Final residual: {result.final_residual:.2e}")
    print(f"   Total mass: {result.total_mass:.6f}")
    print(f"   Mass conservation error: {result.mass_conservation_error:.2e}")
    print(f"   Total energy: {result.total_energy:.6f}")

    # Compute derived quantities
    velocity_field = result.compute_velocity_field()
    optimal_trajectory = result.compute_optimal_trajectory(x_start=0.0)

    print(f"   Max velocity: {np.max(np.abs(velocity_field)):.3f}")
    print(f"   Trajectory length: {len(optimal_trajectory)} points")

    # 7. PROBLEM MODIFICATION AND PARAMETER STUDIES
    print("\n7. Problem modification and parameter studies:")

    base_problem = create_mfg_problem("crowd_dynamics", domain=(0, 5), crowd_size=100)

    print("📊 Parameter study results:")
    for crowd_size in [50, 100, 200, 400]:
        modified_problem = base_problem.with_parameters(crowd_size=crowd_size)
        param_result = solver_direct.solve(modified_problem)

        print(
            f"   Crowd size {crowd_size:3d}: "
            f"{param_result.iterations:3d} iterations, "
            f"evacuation time: {param_result.evacuation_time:.2f}s"
        )

    # 8. BACKEND COMPARISON
    print("\n8. Backend performance comparison:")

    backends = ["numpy", "torch"]  # Add 'jax', 'numba' if available
    backend_results = {}

    for backend in backends:
        try:
            backend_solver = FixedPointSolver(backend=backend, max_iterations=100)
            start_time = time.time()
            backend_result = backend_solver.solve(problem)
            solve_time = time.time() - start_time

            backend_results[backend] = {"time": solve_time, "iterations": backend_result.iterations}

            print(f"🖥️ {backend:8s}: {solve_time:6.3f}s ({backend_result.iterations:3d} iterations)")

        except ImportError:
            print(f"⚠️ {backend:8s}: Not available")

    # Find fastest backend
    if backend_results:
        fastest = min(backend_results.items(), key=lambda x: x[1]["time"])
        print(f"🏆 Fastest backend: {fastest[0]} ({fastest[1]['time']:.3f}s)")

    # 9. ERROR HANDLING AND DIAGNOSTICS
    print("\n9. Error handling and diagnostics:")

    try:
        # Try with very strict tolerance
        strict_solver = FixedPointSolver(tolerance=1e-12, max_iterations=50)
        strict_result = strict_solver.solve(problem)
        print(f"✅ Strict solve succeeded: {strict_result.iterations} iterations")

    except Exception as e:
        print(f"❌ Strict solve failed: {e}")

        # Fallback with relaxed settings
        print("🔄 Trying with relaxed settings...")
        relaxed_solver = FixedPointSolver(tolerance=1e-4, max_iterations=200)
        relaxed_result = relaxed_solver.solve(problem)
        print(f"✅ Relaxed solve succeeded: {relaxed_result.iterations} iterations")

    # Diagnostic analysis
    from mfg_pde.diagnostics import diagnose_problem

    diagnosis = diagnose_problem(problem)
    print("🔍 Problem diagnosis:")
    print(f"   Difficulty: {diagnosis.difficulty}")
    print(f"   Recommended tolerance: {diagnosis.recommended_tolerance}")
    print(f"   Estimated iterations: {diagnosis.estimated_iterations}")

    # 10. INTEGRATION WITH SIMPLE API
    print("\n10. Integration with simple API:")

    # Start with simple API
    from mfg_pde import solve_mfg

    simple_result = solve_mfg("crowd_dynamics", crowd_size=200)

    # Extract problem for detailed analysis
    extracted_problem = simple_result.problem

    # Use core objects for detailed control
    detailed_solver = FixedPointSolver().with_tolerance(1e-8).with_adaptive_damping(True)

    detailed_result = detailed_solver.solve(extracted_problem)

    print("🔄 API integration:")
    print(f"   Simple API: {simple_result.iterations} iterations")
    print(f"   Detailed control: {detailed_result.iterations} iterations")
    print(f"   Accuracy improvement: {simple_result.final_residual / detailed_result.final_residual:.1f}x")

    print("\n🎉 Core objects demo completed!")
    print("📖 Next steps:")
    print("   • Try custom problem definitions")
    print("   • Experiment with different solver configurations")
    print("   • Explore the hooks system for algorithm customization")
    print("   • Check out advanced_hooks.md for expert-level control")


if __name__ == "__main__":
    main()
