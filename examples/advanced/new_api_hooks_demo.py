#!/usr/bin/env python3
"""
New API Hooks Demo - Layer 3 Interface

This example demonstrates the new Layer 3 API that provides unprecedented
control over solver algorithms through the hooks system.

Perfect for:
- Algorithm researchers
- Custom method development
- Performance optimization
- Advanced monitoring and analysis
"""

import itertools
import time

import matplotlib.pyplot as plt
import numpy as np

from mfg_pde import create_mfg_problem
from mfg_pde.hooks import (
    DebugHook,
    HookCollection,
    PerformanceHook,
    ProgressHook,
    SolverHooks,
)
from mfg_pde.solvers import FixedPointSolver


class CustomMonitoringHook(SolverHooks):
    """Custom hook for detailed monitoring and analysis."""

    def __init__(self):
        self.iteration_times = []
        self.residual_history = []
        self.mass_conservation = []
        self.energy_history = []

    def on_solve_start(self, initial_state):
        print(f"🚀 Starting solve with {initial_state.grid_size} grid points")
        self.start_time = time.time()

    def on_iteration_start(self, state):
        self.iter_start_time = time.time()

    def on_iteration_end(self, state):
        iter_time = time.time() - self.iter_start_time
        self.iteration_times.append(iter_time)
        self.residual_history.append(state.residual)

        # Check mass conservation
        mass = np.trapz(state.density, state.x_grid)
        self.mass_conservation.append(abs(mass - 1.0))

        # Compute energy
        velocity = -np.gradient(state.value_function, state.dx)
        energy = 0.5 * np.trapz(state.density * velocity**2, state.x_grid)
        self.energy_history.append(energy)

        if state.iteration % 10 == 0:
            print(
                f"Iter {state.iteration:3d}: "
                f"residual={state.residual:.2e}, "
                f"mass_error={self.mass_conservation[-1]:.2e}, "
                f"energy={energy:.4f}, "
                f"time={iter_time:.3f}s"
            )

    def on_solve_end(self, result):
        total_time = time.time() - self.start_time
        avg_iter_time = sum(self.iteration_times) / len(self.iteration_times)

        print(f"✅ Solve completed in {total_time:.2f}s")
        print(f"📊 Average iteration time: {avg_iter_time:.3f}s")
        print(f"📈 Final mass conservation error: {self.mass_conservation[-1]:.2e}")
        print(f"⚡ Final energy: {self.energy_history[-1]:.6f}")

        # Add analysis to result
        result.iteration_times = self.iteration_times
        result.mass_conservation_history = self.mass_conservation
        result.energy_history = self.energy_history
        return result


class AdaptiveDampingHook(SolverHooks):
    """Dynamically adjust damping based on convergence behavior."""

    def __init__(self, initial_damping=0.8, min_damping=0.1, max_damping=0.95):
        self.damping = initial_damping
        self.min_damping = min_damping
        self.max_damping = max_damping
        self.residual_history = []
        self.damping_history = []

    def on_iteration_end(self, state):
        self.residual_history.append(state.residual)

        if len(self.residual_history) >= 3:
            recent_residuals = self.residual_history[-3:]

            if recent_residuals[-1] > recent_residuals[-2]:
                # Residual increased - reduce damping for stability
                old_damping = self.damping
                self.damping = max(self.min_damping, self.damping * 0.9)
                if abs(self.damping - old_damping) > 1e-6:
                    print(f"🔽 Reducing damping to {self.damping:.3f} (stability)")

            elif all(r1 > r2 for r1, r2 in itertools.pairwise(recent_residuals)):
                # Consistent improvement - can increase damping for speed
                old_damping = self.damping
                self.damping = min(self.max_damping, self.damping * 1.05)
                if abs(self.damping - old_damping) > 1e-6:
                    print(f"🔼 Increasing damping to {self.damping:.3f} (acceleration)")

        self.damping_history.append(self.damping)

        # Update solver damping (if supported)
        if hasattr(state.solver, "set_damping"):
            state.solver.set_damping(self.damping)

        return state


class EarlyStoppingHook(SolverHooks):
    """Stop early based on custom criteria."""

    def __init__(self, patience=50, min_improvement=1e-8):
        self.patience = patience
        self.min_improvement = min_improvement
        self.best_residual = float("inf")
        self.wait_count = 0

    def on_convergence_check(self, state):
        if state.residual < self.best_residual - self.min_improvement:
            self.best_residual = state.residual
            self.wait_count = 0
        else:
            self.wait_count += 1

        if self.wait_count >= self.patience:
            print(f"⏰ Early stopping: no improvement for {self.patience} iterations")
            state.should_stop = True

        return state


class DataCollectionHook(SolverHooks):
    """Collect detailed data for post-hoc analysis."""

    def __init__(self, collect_frequency=10):
        self.collect_frequency = collect_frequency
        self.snapshots = []

    def on_iteration_end(self, state):
        if state.iteration % self.collect_frequency == 0:
            snapshot = {
                "iteration": state.iteration,
                "residual": state.residual,
                "density": state.density.copy(),
                "value_function": state.value_function.copy(),
                "velocity_field": self._compute_velocity(state),
                "energy": self._compute_energy(state),
                "entropy": self._compute_entropy(state),
            }
            self.snapshots.append(snapshot)

    def _compute_velocity(self, state):
        """Compute velocity field from value function."""
        return -np.gradient(state.value_function, state.dx)

    def _compute_energy(self, state):
        """Compute total kinetic energy."""
        velocity = self._compute_velocity(state)
        return 0.5 * np.trapz(state.density * velocity**2, state.x_grid)

    def _compute_entropy(self, state):
        """Compute entropy of density distribution."""
        density_safe = np.maximum(state.density, 1e-12)
        return -np.trapz(state.density * np.log(density_safe), state.x_grid)

    def create_analysis(self):
        """Create analysis plots from collected data."""
        iterations = [s["iteration"] for s in self.snapshots]
        energies = [s["energy"] for s in self.snapshots]
        entropies = [s["entropy"] for s in self.snapshots]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        ax1.plot(iterations, energies, "b-", linewidth=2)
        ax1.set_title("Energy Evolution During Solving")
        ax1.set_ylabel("Total Energy")
        ax1.grid(True, alpha=0.3)

        ax2.plot(iterations, entropies, "r-", linewidth=2)
        ax2.set_title("Entropy Evolution During Solving")
        ax2.set_ylabel("Entropy")
        ax2.set_xlabel("Iteration")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


class HJBCustomizationHook(SolverHooks):
    """Customize HJB equation solving."""

    def on_hjb_start(self, state):
        if state.iteration % 20 == 0:
            print(f"🔧 Solving HJB equation at iteration {state.iteration}")

    def on_hjb_step(self, state, x_point, current_value):
        # Custom processing at each spatial point
        if self._needs_special_treatment(x_point):
            return self._apply_custom_logic(x_point, current_value, state)
        return current_value

    def on_hjb_end(self, state):
        # Post-process HJB solution with smoothing
        state.value_function = self._smooth_solution(state.value_function)

    def _needs_special_treatment(self, x_point):
        # Special treatment near boundaries
        return x_point < 0.1 or x_point > 0.9

    def _apply_custom_logic(self, x_point, value, state):
        # Custom boundary behavior
        if x_point < 0.1:
            return value * 0.8  # Reduce value near left boundary
        return value

    def _smooth_solution(self, solution):
        # Apply mild smoothing filter
        from scipy.ndimage import gaussian_filter1d

        return gaussian_filter1d(solution, sigma=0.5)


class PerformanceOptimizationHook(SolverHooks):
    """Optimize performance during solving."""

    def __init__(self, memory_limit_gb=8):
        self.memory_limit = memory_limit_gb * 1024**3
        self.gc_frequency = 20

    def on_iteration_end(self, state):
        if state.iteration % self.gc_frequency == 0:
            import gc

            import psutil

            # Check memory usage
            memory_usage = psutil.Process().memory_info().rss
            if memory_usage > self.memory_limit * 0.8:
                print(f"💾 High memory usage: {memory_usage / 1024**3:.1f}GB - cleaning up")
                gc.collect()

        return state


def main():
    print("🎛️ MFG_PDE New API Demo - Advanced Hooks System")
    print("=" * 60)

    # Create test problem
    problem = create_mfg_problem("crowd_dynamics", domain=(0, 5), time_horizon=2.0, crowd_size=200)

    # 1. BASIC HOOKS USAGE
    print("\n1. Basic hooks for monitoring:")

    basic_hooks = [
        DebugHook(log_level="INFO"),
        ProgressHook(update_frequency=15, show_eta=True),
        PerformanceHook(profile_memory=True),
    ]

    solver = FixedPointSolver(max_iterations=100, tolerance=1e-5)
    result = solver.solve(problem, hooks=HookCollection(basic_hooks))

    print(f"📊 Basic monitoring result: {result.iterations} iterations")

    # 2. CUSTOM MONITORING HOOK
    print("\n2. Custom monitoring hook:")

    custom_monitor = CustomMonitoringHook()
    result_monitored = solver.solve(problem, hooks=custom_monitor)

    # Access custom analysis
    print("📈 Custom analysis available:")
    print(f"   Average iteration time: {np.mean(result_monitored.iteration_times):.3f}s")
    print(f"   Energy variance: {np.var(result_monitored.energy_history):.6f}")

    # 3. ADAPTIVE ALGORITHMS
    print("\n3. Adaptive damping hook:")

    adaptive_hook = AdaptiveDampingHook(initial_damping=0.7)
    adaptive_result = solver.solve(problem, hooks=adaptive_hook)

    print(f"🔄 Adaptive damping result: {adaptive_result.iterations} iterations")
    print(f"   Final damping: {adaptive_hook.damping:.3f}")
    print(f"   Damping changes: {len(set(adaptive_hook.damping_history))} different values")

    # 4. EARLY STOPPING
    print("\n4. Early stopping hook:")

    early_stop_hook = EarlyStoppingHook(patience=30, min_improvement=1e-9)
    large_solver = FixedPointSolver(max_iterations=500, tolerance=1e-8)

    early_result = large_solver.solve(problem, hooks=early_stop_hook)
    print(f"⏰ Early stopping result: {early_result.iterations} iterations")

    # 5. DATA COLLECTION AND ANALYSIS
    print("\n5. Data collection for analysis:")

    data_hook = DataCollectionHook(collect_frequency=5)
    solver.solve(problem, hooks=data_hook)

    print(f"📊 Data collection result: {len(data_hook.snapshots)} snapshots collected")

    # Create analysis plots
    data_hook.create_analysis()
    plt.savefig("hooks_analysis.png", dpi=150, bbox_inches="tight")
    print("📈 Analysis plots saved as 'hooks_analysis.png'")
    plt.close()

    # 6. ALGORITHM CUSTOMIZATION
    print("\n6. HJB customization hook:")

    hjb_hook = HJBCustomizationHook()
    custom_result = solver.solve(problem, hooks=hjb_hook)

    print(f"🔧 HJB customization result: {custom_result.iterations} iterations")

    # 7. HOOK COMPOSITION AND PRIORITIES
    print("\n7. Hook composition with priorities:")

    class CompositeHook(SolverHooks):
        def __init__(self, hooks, priorities=None):
            self.hooks = hooks
            self.priorities = priorities or [1] * len(hooks)

        def on_iteration_end(self, state):
            # Execute hooks in priority order
            hook_methods = list(zip(self.priorities, self.hooks, strict=False))
            hook_methods.sort(key=lambda x: x[0], reverse=True)

            for priority, hook in hook_methods:
                if hasattr(hook, "on_iteration_end"):
                    state = hook.on_iteration_end(state) or state

            return state

    # High priority for adaptive damping, low for monitoring
    composite = CompositeHook(
        [AdaptiveDampingHook(), CustomMonitoringHook(), PerformanceOptimizationHook()], priorities=[3, 1, 2]
    )

    composite_result = solver.solve(problem, hooks=composite)
    print(f"🎭 Composite hooks result: {composite_result.iterations} iterations")

    # 8. CONDITIONAL HOOKS
    print("\n8. Conditional hooks:")

    class ConditionalHook(SolverHooks):
        def __init__(self, hook, condition):
            self.hook = hook
            self.condition = condition

        def on_iteration_end(self, state):
            if self.condition(state):
                return self.hook.on_iteration_end(state)
            return state

    # Only debug if convergence is slow
    def slow_convergence(state):
        return state.iteration > 50 and state.residual > 1e-6

    conditional_debug = ConditionalHook(DebugHook(), slow_convergence)

    conditional_result = solver.solve(problem, hooks=conditional_debug)
    print(f"🎯 Conditional hooks result: {conditional_result.iterations} iterations")

    # 9. RESEARCH DATA COLLECTION
    print("\n9. Advanced research data collection:")

    class ResearchHook(SolverHooks):
        def __init__(self):
            self.solution_evolution = []
            self.hamiltonian_values = []

        def on_iteration_end(self, state):
            if state.iteration % 10 == 0:
                # Store complete solution state
                self.solution_evolution.append(
                    {
                        "iteration": state.iteration,
                        "u": state.value_function.copy(),
                        "m": state.density.copy(),
                        "residual": state.residual,
                    }
                )

                # Compute Hamiltonian values
                u_x = np.gradient(state.value_function, state.dx)
                H_values = 0.5 * u_x**2 + 0.1 * state.density
                self.hamiltonian_values.append(H_values.copy())

        def export_research_data(self, filename):
            import pickle

            data = {"evolution": self.solution_evolution, "hamiltonians": self.hamiltonian_values}
            with open(filename, "wb") as f:
                pickle.dump(data, f)

    research_hook = ResearchHook()
    solver.solve(problem, hooks=research_hook)

    research_hook.export_research_data("research_data.pkl")
    print(f"🔬 Research data exported: {len(research_hook.solution_evolution)} time points")

    # 10. PERFORMANCE COMPARISON
    print("\n10. Performance comparison with and without hooks:")

    # Baseline solve
    baseline_start = time.time()
    baseline_result = solver.solve(problem)
    baseline_time = time.time() - baseline_start

    # With comprehensive hooks
    comprehensive_hooks = [
        CustomMonitoringHook(),
        AdaptiveDampingHook(),
        DataCollectionHook(collect_frequency=5),
        PerformanceOptimizationHook(),
    ]

    hooks_start = time.time()
    hooks_result = solver.solve(problem, hooks=HookCollection(comprehensive_hooks))
    hooks_time = time.time() - hooks_start

    print("⚡ Performance comparison:")
    print(f"   Baseline: {baseline_time:.3f}s ({baseline_result.iterations} iter)")
    print(f"   With hooks: {hooks_time:.3f}s ({hooks_result.iterations} iter)")
    print(f"   Overhead: {(hooks_time / baseline_time - 1) * 100:.1f}%")

    print("\n🎉 Advanced hooks demo completed!")
    print("🔬 The hooks system provides unlimited extensibility:")
    print("   • Monitor and debug algorithm behavior")
    print("   • Implement adaptive algorithms")
    print("   • Collect research data")
    print("   • Optimize performance")
    print("   • Customize solver algorithms")
    print("   • Integrate with external systems")

    print("\n📖 Next steps:")
    print("   • Develop custom hooks for your research")
    print("   • Combine multiple hooks for complex behaviors")
    print("   • Contribute useful hooks back to the community")
    print("   • Explore the full hooks API documentation")


if __name__ == "__main__":
    main()
