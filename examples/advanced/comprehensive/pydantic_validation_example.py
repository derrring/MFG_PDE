#!/usr/bin/env python3
"""
Configuration System Example for MFG_PDE

Demonstrates the Pydantic-based configuration system with:
- Automatic validation and type checking
- YAML serialization for reproducible experiments
- Config composition and parameter overrides
- Integration with solve_mfg() high-level API

Modern API (Phase 3.3):
- Uses SolverConfig (not deprecated MFGSolverConfig)
- Uses MFGProblem (not GridBasedMFGProblem)
- Shows both YAML-based and programmatic configuration
"""

from pathlib import Path

from mfg_pde import solve_mfg
from mfg_pde.config import (
    BackendConfig,
    FPConfig,
    HJBConfig,
    LoggingConfig,
    PicardConfig,
    SolverConfig,
)
from mfg_pde.core.mfg_problem import MFGProblem


def example_1_basic_config():
    """Example 1: Create and validate basic solver configuration."""
    print("\n" + "=" * 70)
    print("Example 1: Basic Solver Configuration")
    print("=" * 70)

    # Create configuration programmatically
    config = SolverConfig(
        hjb=HJBConfig(method="fdm", accuracy_order=2),
        fp=FPConfig(method="particle", num_particles=5000),
        picard=PicardConfig(max_iterations=50, tolerance=1e-6, damping_factor=0.5),
        backend=BackendConfig(type="numpy", device="cpu", precision="float64"),
        logging=LoggingConfig(level="INFO", progress_bar=True),
    )

    print("\nConfiguration created:")
    print(f"  HJB solver: {config.hjb.method}")
    print(f"  FP solver: {config.fp.method} ({config.fp.num_particles} particles)")
    print(f"  Picard iterations: {config.picard.max_iterations}")
    print(f"  Tolerance: {config.picard.tolerance:.2e}")
    print(f"  Backend: {config.backend.type} ({config.backend.precision})")

    return config


def example_2_yaml_serialization():
    """Example 2: Save and load configuration from YAML."""
    print("\n" + "=" * 70)
    print("Example 2: YAML Serialization")
    print("=" * 70)

    # Create configuration
    config = SolverConfig(
        hjb=HJBConfig(method="fdm", accuracy_order=2),
        fp=FPConfig(method="particle", num_particles=10000),
        picard=PicardConfig(max_iterations=100, tolerance=1e-8),
    )

    # Save to YAML
    yaml_path = Path("example_solver_config.yaml")
    config.to_yaml(yaml_path)
    print(f"\n Configuration saved to: {yaml_path}")

    # Show YAML content
    print("\nYAML content:")
    print(yaml_path.read_text())

    # Reload from YAML
    reloaded_config = SolverConfig.from_yaml(yaml_path)
    print(" Configuration reloaded successfully!")
    print(f"  HJB method: {reloaded_config.hjb.method}")
    print(f"  Picard tolerance: {reloaded_config.picard.tolerance:.2e}")

    # Cleanup
    yaml_path.unlink()

    return reloaded_config


def example_3_validation_errors():
    """Example 3: Demonstrate automatic validation."""
    print("\n" + "=" * 70)
    print("Example 3: Automatic Validation")
    print("=" * 70)

    # Test 1: Invalid tolerance (must be positive)
    print("\n[Test 1] Invalid tolerance (negative):")
    try:
        SolverConfig(picard=PicardConfig(tolerance=-1.0))
        print("  FAILED: Should have raised validation error")
    except ValueError as e:
        print("  PASS: Caught validation error")
        print(f"  Error: {str(e)[:80]}...")

    # Test 2: Invalid damping factor (must be in (0, 1])
    print("\n[Test 2] Invalid damping_factor (>1.0):")
    try:
        SolverConfig(picard=PicardConfig(damping_factor=1.5))
        print("  FAILED: Should have raised validation error")
    except ValueError as e:
        print("  PASS: Caught validation error")
        print(f"  Error: {str(e)[:80]}...")

    # Test 3: Invalid backend (numpy doesn't support GPU)
    print("\n[Test 3] Invalid backend (numpy + gpu):")
    try:
        SolverConfig(backend=BackendConfig(type="numpy", device="gpu"))
        print("  FAILED: Should have raised validation error")
    except ValueError as e:
        print("  PASS: Caught validation error")
        print(f"  Error: {str(e)[:80]}...")

    # Test 4: Valid configuration
    print("\n[Test 4] Valid configuration:")
    try:
        _ = SolverConfig(
            picard=PicardConfig(tolerance=1e-6, damping_factor=0.5),
            backend=BackendConfig(type="numpy", device="cpu"),
        )
        print("  PASS: Valid configuration accepted")
    except ValueError as e:
        print(f"  FAILED: Should not have raised error: {e}")


def example_4_solve_with_config():
    """Example 4: Use configuration with solve_mfg()."""
    print("\n" + "=" * 70)
    print("Example 4: Solve MFG with Custom Configuration")
    print("=" * 70)

    # Create 1D MFG problem
    problem = MFGProblem(
        spatial_bounds=[(0.0, 1.0)],
        spatial_discretization=[51],
        T=1.0,
        Nt=51,
        sigma=0.1,
    )

    # Create custom configuration
    config = SolverConfig(
        hjb=HJBConfig(method="fdm", accuracy_order=2),
        fp=FPConfig(method="particle", num_particles=2000),
        picard=PicardConfig(max_iterations=5, tolerance=1e-4, verbose=False),
    )

    print("\nSolving MFG problem...")
    print(f"  Problem: 1D, 51 grid points, T={problem.T}")
    print(f"  HJB: {config.hjb.method}")
    print(f"  FP: {config.fp.method} ({config.fp.num_particles} particles)")
    print(f"  Picard: max_iter={config.picard.max_iterations}")

    # Solve (using solve_mfg high-level API)
    result = solve_mfg(problem, method="fast", max_iterations=5, tolerance=1e-4, verbose=False)

    print("\n Solution computed:")
    print(f"  Converged: {result.converged}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Final error (U): {result.error_history_U[-1]:.2e}")
    print(f"  Final error (M): {result.error_history_M[-1]:.2e}")
    print(f"  Solution shape: U={result.U.shape}, M={result.M.shape}")


def main():
    """Run all configuration examples."""
    print("\n" + "=" * 70)
    print("MFG_PDE Configuration System Examples")
    print("=" * 70)

    # Run examples
    example_1_basic_config()
    example_2_yaml_serialization()
    example_3_validation_errors()
    example_4_solve_with_config()

    print("\n" + "=" * 70)
    print(" All examples completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
