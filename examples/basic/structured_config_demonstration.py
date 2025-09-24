#!/usr/bin/env python3
"""
Structured Configuration Demonstration (Issue #28 Solution)

This example demonstrates the complete solution to OmegaConf type checking
problems using structured configs with dataclass schemas.

The problem: OmegaConf DictConfig objects cause mypy type errors
The solution: Use structured configs for full type safety
"""

from pathlib import Path
from tempfile import NamedTemporaryFile

# Import the type-safe structured config functions
from mfg_pde.config.omegaconf_manager import (
    create_default_structured_config,
    load_structured_mfg_config,
)


def demonstrate_issue_28_solution():
    """Show the before/after for Issue #28 type checking solution."""
    print("🛡️  === STRUCTURED CONFIG DEMONSTRATION (Issue #28) ===")
    print()

    print("❌ BEFORE - Type checking fails:")
    print("   from omegaconf import OmegaConf")
    print("   conf = OmegaConf.load('config.yaml')")
    print("   print(conf.problem.T)  # ❌ Mypy error: DictConfig has no attribute 'problem'")
    print()

    print("✅ AFTER - Type safe with structured configs:")
    print("   config = load_structured_mfg_config('config.yaml')")
    print("   print(config.problem.T)  # ✅ Type safe, autocompletes!")
    print()

    # Demonstrate the working solution
    print("🔍 Creating type-safe default configuration...")
    config = create_default_structured_config()

    print("✅ Type-safe attribute access:")
    print(f"   Problem time horizon: {config.problem.T}")
    print(f"   Spatial grid points: {config.problem.Nx}")
    print(f"   Time grid points: {config.problem.Nt}")
    print(f"   Solver type: {config.solver.type}")
    print(f"   Max iterations: {config.solver.max_iterations}")
    print(f"   Convergence tolerance: {config.solver.tolerance}")
    print()

    print("✅ Nested attribute access (fully typed):")
    print(f"   Boundary condition type: {config.problem.boundary_conditions.type}")
    print(f"   Initial condition type: {config.problem.initial_condition.type}")
    print(f"   HJB solver method: {config.solver.hjb.method}")
    print(f"   Newton max iterations: {config.solver.hjb.newton.max_iterations}")
    print(f"   Experiment logging level: {config.experiment.logging.level}")
    print()


def demonstrate_yaml_integration():
    """Show how structured configs work with YAML files."""
    print("📄 === YAML INTEGRATION WITH TYPE SAFETY ===")
    print()

    # Create a sample YAML configuration
    yaml_content = """
problem:
  name: "crowd_dynamics_demo"
  type: "spatial_optimization"
  T: 2.5
  Nx: 100
  Nt: 50
  parameters:
    crowd_aversion: 1.2
    noise_level: 0.05

solver:
  type: "semi_lagrangian"
  max_iterations: 200
  tolerance: 1e-8
  damping: 0.4
  hjb:
    method: "gfdm_optimized"
    newton:
      max_iterations: 25
      tolerance: 1e-10

experiment:
  name: "structured_config_demo"
  description: "Demonstrating type-safe YAML configuration loading"
  logging:
    level: "DEBUG"
  visualization:
    enabled: true
    save_plots: true
    formats: ["png", "html", "pdf"]
"""

    print("🔍 Sample YAML configuration:")
    print(yaml_content)

    # Create temporary YAML file
    with NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        temp_yaml_path = Path(f.name)

    try:
        print("🔍 Loading YAML with structured config...")
        config = load_structured_mfg_config(temp_yaml_path)

        print("✅ Type-safe YAML values loaded:")
        print(f"   Problem name: {config.problem.name}")
        print(f"   Problem type: {config.problem.type}")
        print(f"   Time horizon: {config.problem.T}")
        print(f"   Grid size: {config.problem.Nx} x {config.problem.Nt}")
        print(f"   Solver: {config.solver.type}")
        print(f"   HJB method: {config.solver.hjb.method}")
        print(f"   Experiment: {config.experiment.name}")
        print(f"   Logging level: {config.experiment.logging.level}")
        print()

        print("✅ Parameter access (type-safe dictionaries):")
        params = config.problem.parameters
        print(f"   Crowd aversion: {params.get('crowd_aversion', 'N/A')}")
        print(f"   Noise level: {params.get('noise_level', 'N/A')}")
        print()

        print("✅ List access (type-safe):")
        formats = config.experiment.visualization.formats
        print(f"   Visualization formats: {formats}")
        print()

    finally:
        temp_yaml_path.unlink()


def demonstrate_type_safety_benefits():
    """Show the key benefits of structured configs."""
    print("🎯 === TYPE SAFETY BENEFITS ===")
    print()

    config = create_default_structured_config()

    print("✅ Benefits achieved:")
    print("   1. Full mypy compatibility - Zero type checking errors")
    print("   2. IDE autocompletion - Complete attribute suggestions")
    print("   3. Runtime validation - Schema enforcement at load time")
    print("   4. Better maintainability - Clear configuration structure")
    print("   5. Catch errors early - Configuration errors at development time")
    print()

    print("🔍 Schema validation in action:")
    print(f"   Problem T is numeric: {isinstance(config.problem.T, (int, float))}")
    print(f"   Problem Nx is integer: {isinstance(config.problem.Nx, int)}")
    print(f"   Solver type is string: {isinstance(config.solver.type, str)}")
    print(f"   Visualization enabled is boolean: {isinstance(config.experiment.visualization.enabled, bool)}")
    print()

    print("💡 Usage in real MFG code:")
    print("   # This code is now fully type-safe and IDE-friendly:")
    print("   config = load_structured_mfg_config('my_experiment.yaml')")
    print("   time_horizon = config.problem.T  # ✅ Autocompletes, type-checked")
    print("   nx, nt = config.problem.Nx, config.problem.Nt  # ✅ Type-safe")
    print("   solver_settings = config.solver  # ✅ Full schema available")
    print()


def demonstrate_compatibility():
    """Show backward compatibility with existing code."""
    print("🔄 === BACKWARD COMPATIBILITY ===")
    print()

    print("✅ Existing YAML files work unchanged:")
    print("   - No need to modify existing configuration files")
    print("   - Schema provides defaults for missing fields")
    print("   - Gradual migration path available")
    print()

    print("✅ Migration strategy:")
    print("   1. OLD: conf = OmegaConf.load('config.yaml')")
    print("   2. NEW: conf = load_structured_mfg_config('config.yaml')")
    print("   3. Benefit: Full type safety with minimal code changes")
    print()


if __name__ == "__main__":
    demonstrate_issue_28_solution()
    print()
    demonstrate_yaml_integration()
    print()
    demonstrate_type_safety_benefits()
    print()
    demonstrate_compatibility()
    print()
    print("🚀 Issue #28 Complete - Structured configs provide full type safety!")
    print("   Use load_structured_mfg_config() for new code")
    print("   See mfg_pde.config.structured_schemas for all available schemas")
