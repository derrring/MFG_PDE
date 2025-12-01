#!/usr/bin/env python3
"""
High-Level solve_mfg() Interface Demonstration

Demonstrates the simplified one-line interface for solving MFG problems.

Phase 3.3 Update
----------------
This example now shows the new unified API with Phase 3.2 SolverConfig integration:
- New config parameter with presets
- YAML, Builder, and Preset patterns
- Legacy method parameter (deprecated)

Run:
    python examples/basic/solve_mfg_demo.py
"""

from mfg_pde import MFGProblem, solve_mfg
from mfg_pde.config import ConfigBuilder, presets


def demo_simple_usage():
    """Simplest usage with all defaults."""
    print("\n" + "=" * 60)
    print("Demo 1: Simplest Usage (Auto Configuration)")
    print("=" * 60)

    problem = MFGProblem()

    # One-line solve with new API!
    result = solve_mfg(problem)

    print("\n✓ Solved successfully")
    print(f"  Converged: {result.converged}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Final U error: {result.error_history_U[-1]:.2e}")
    print(f"  Final M error: {result.error_history_M[-1]:.2e}")
    if result.execution_time is not None:
        print(f"  Execution time: {result.execution_time:.3f}s")
    print(f"  Solution shapes: U={result.U.shape}, M={result.M.shape}")


def demo_new_config_api():
    """Demonstrate new Phase 3.2/3.3 config API."""
    print("\n" + "=" * 60)
    print("Demo 2: New Config API (Phase 3.3 - RECOMMENDED)")
    print("=" * 60)

    problem = MFGProblem()

    # Method 1: Preset objects (recommended)
    print("\n1. Using preset objects:")
    print("   from mfg_pde.config import presets")
    print("   result = solve_mfg(problem, config=presets.accurate_solver())")
    result = solve_mfg(problem, config=presets.accurate_solver(), verbose=False)
    print(f"   ✓ Converged in {result.iterations} iterations")
    print(f"   Final error: {result.error_history_U[-1]:.2e}")

    # Method 2: Preset names
    print("\n2. Using preset names:")
    print("   result = solve_mfg(problem, config='fast')")
    result = solve_mfg(problem, config="fast", verbose=False)
    print(f"   ✓ Converged in {result.iterations} iterations")
    print(f"   Final error: {result.error_history_U[-1]:.2e}")

    # Method 3: Builder API
    print("\n3. Using Builder API:")
    print("   from mfg_pde.config import ConfigBuilder")
    print("   config = (")
    print("       ConfigBuilder()")
    print("       .solver_hjb(method='fdm', accuracy_order=2)")
    print("       .solver_fp_particle(num_particles=5000)")
    print("       .picard(max_iterations=100, tolerance=1e-6)")
    print("       .build()")
    print("   )")
    print("   result = solve_mfg(problem, config=config)")

    config = (
        ConfigBuilder()
        .solver_hjb(method="fdm", accuracy_order=2)
        .solver_fp_particle(num_particles=5000)
        .picard(max_iterations=100, tolerance=1e-6)
        .build()
    )
    result = solve_mfg(problem, config=config, verbose=False)
    print(f"   ✓ Converged in {result.iterations} iterations")
    print(f"   Final error: {result.error_history_U[-1]:.2e}")


def demo_preset_comparison():
    """Compare different preset configurations."""
    print("\n" + "=" * 60)
    print("Demo 3: Preset Comparison")
    print("=" * 60)

    problem = MFGProblem()

    # Fast preset
    print("\n1. Fast preset (optimized for speed)")
    result_fast = solve_mfg(problem, config="fast", verbose=False)
    print(f"   Iterations: {result_fast.iterations}")
    if result_fast.execution_time:
        print(f"   Time: {result_fast.execution_time:.3f}s")
    print(f"   Final error: {result_fast.error_history_U[-1]:.2e}")

    # Accurate preset
    print("\n2. Accurate preset (high precision)")
    result_accurate = solve_mfg(problem, config="accurate", verbose=False)
    print(f"   Iterations: {result_accurate.iterations}")
    if result_accurate.execution_time:
        print(f"   Time: {result_accurate.execution_time:.3f}s")
    print(f"   Final error: {result_accurate.error_history_U[-1]:.2e}")

    # Research preset
    print("\n3. Research preset (comprehensive diagnostics)")
    result_research = solve_mfg(problem, config="research", verbose=False)
    print(f"   Iterations: {result_research.iterations}")
    if result_research.execution_time:
        print(f"   Time: {result_research.execution_time:.3f}s")
    print(f"   Final error: {result_research.error_history_U[-1]:.2e}")

    print(
        f"\n   Accuracy improvement (fast → accurate): {result_fast.error_history_U[-1] / result_accurate.error_history_U[-1]:.1f}×"
    )


def demo_custom_parameters():
    """Demonstrate custom parameter overrides."""
    print("\n" + "=" * 60)
    print("Demo 4: Custom Parameters with Presets")
    print("=" * 60)

    problem = MFGProblem()

    # Start with preset, override specific parameters
    result = solve_mfg(
        problem,
        config="accurate",  # Base preset
        max_iterations=200,  # Override max iterations
        tolerance=1e-7,  # Override tolerance
        verbose=False,
    )

    print("\n✓ Solved with custom overrides")
    print("  Base config: 'accurate' preset")
    print("  Overrides:")
    print("    - max_iterations: 200")
    print("    - tolerance: 1e-7")
    print("\nResults:")
    print(f"  Converged: {result.converged}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Final error: {result.error_history_U[-1]:.2e}")


def demo_legacy_api():
    """Show legacy API with deprecation warnings."""
    print("\n" + "=" * 60)
    print("Demo 5: Legacy API (Deprecated)")
    print("=" * 60)

    problem = MFGProblem()

    print("\nLegacy API (still works but deprecated):")
    print("  result = solve_mfg(problem, method='fast')")
    print("\nNew API (recommended):")
    print("  result = solve_mfg(problem, config='fast')")

    print("\nRunning with legacy API (will show deprecation warning):")
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = solve_mfg(problem, method="fast", verbose=False)

        if w:
            print("\n⚠️  Deprecation warning captured:")
            print(f"    {w[0].message}")

    print(f"\n✓ Still works: Converged in {result.iterations} iterations")
    print("   (But please migrate to new API!)")


def demo_migration_guide():
    """Show migration from old to new API."""
    print("\n" + "=" * 60)
    print("Demo 6: Migration Guide")
    print("=" * 60)

    print("\n" + "=" * 40)
    print("OLD API (Deprecated)")
    print("=" * 40)
    print("""
# Using method parameter
result = solve_mfg(problem, method="accurate")

# With custom config
from mfg_pde.config import create_accurate_config
config = create_accurate_config()
# ... modify config ...
solver = create_accurate_solver(problem, custom_config=config)
result = solver.solve()
""")

    print("=" * 40)
    print("NEW API (Recommended)")
    print("=" * 40)
    print("""
# 1. Using preset name (simplest)
result = solve_mfg(problem, config="accurate")

# 2. Using preset object
from mfg_pde.config import presets
result = solve_mfg(problem, config=presets.accurate_solver())

# 3. Using Builder API (most flexible)
from mfg_pde.config import ConfigBuilder
config = (
    ConfigBuilder()
    .solver_hjb(method="fdm", accuracy_order=2)
    .solver_fp_particle(num_particles=5000)
    .picard(max_iterations=100, tolerance=1e-6)
    .build()
)
result = solve_mfg(problem, config=config)

# 4. Using YAML (most reproducible)
from mfg_pde.config import load_solver_config
config = load_solver_config("experiment.yaml")
result = solve_mfg(problem, config=config)
""")

    print("\n✓ Migration complete!")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("solve_mfg() High-Level Interface Demonstration")
    print("Phase 3.3: Unified Factory & Configuration Integration")
    print("=" * 60)

    demo_simple_usage()
    demo_new_config_api()
    demo_preset_comparison()
    demo_custom_parameters()
    demo_legacy_api()
    demo_migration_guide()

    print("\n" + "=" * 60)
    print("All demonstrations complete")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("  1. New API uses 'config' parameter (not 'method')")
    print("  2. Three usage patterns: Presets, Builder, YAML")
    print("  3. Legacy API still works but is deprecated")
    print("  4. Migration is straightforward: method → config")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
