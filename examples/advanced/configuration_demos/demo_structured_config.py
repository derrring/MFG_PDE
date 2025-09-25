#!/usr/bin/env python3
"""
Demonstration of Structured OmegaConf Configuration (Issue #28)

This script demonstrates how structured configs solve the common
OmegaConf type checking problems by providing static type information
that mypy can understand.

Run this script to see type-safe configuration access in action:
    python demo_structured_config.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

# Import our new structured config schemas
from mfg_pde.config.structured_schemas import BeachProblemConfig, MFGConfig

try:
    from omegaconf import OmegaConf

    OMEGACONF_AVAILABLE = True
except ImportError:
    OMEGACONF_AVAILABLE = False
    print("⚠️  OmegaConf not available, demonstrating schema-only functionality")


def demonstrate_type_safe_config():
    """Demonstrate type-safe configuration access."""
    print("🛡️  === STRUCTURED CONFIG DEMONSTRATION ===")
    print()

    # 1. Create default structured config
    print("1️⃣  Creating default MFG configuration...")
    config = MFGConfig()

    # ✅ This is now fully type-safe with autocompletion
    print(f"   ✅ Problem time horizon: {config.problem.T}")
    print(f"   ✅ Spatial grid points: {config.problem.Nx}")
    print(f"   ✅ Time grid points: {config.problem.Nt}")
    print(f"   ✅ Solver type: {config.solver.type}")
    print(f"   ✅ Max iterations: {config.solver.max_iterations}")
    print(f"   ✅ Tolerance: {config.solver.tolerance}")
    print()

    # 2. Demonstrate Beach problem config
    print("2️⃣  Creating Beach problem configuration...")
    beach_config = BeachProblemConfig()
    print(f"   ✅ Problem name: {beach_config.problem.name}")
    print(f"   ✅ Problem type: {beach_config.problem.type}")
    print(f"   ✅ Stall position: {beach_config.problem.parameters.get('stall_position', 'N/A')}")
    print()

    if OMEGACONF_AVAILABLE:
        # 3. Demonstrate OmegaConf integration
        print("3️⃣  Demonstrating OmegaConf structured config...")

        # Create structured config from schema
        omega_config = OmegaConf.structured(MFGConfig)

        # Create a temporary YAML file
        yaml_content = """
problem:
  name: "demo_problem"
  T: 2.5
  Nx: 100
  Nt: 50

solver:
  max_iterations: 200
  tolerance: 1e-8
  damping: 0.3

experiment:
  name: "type_safe_demo"
  description: "Demonstrating type-safe configuration"
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_yaml = Path(f.name)

        try:
            # Load YAML config
            yaml_config = OmegaConf.load(temp_yaml)

            # Merge with structured schema (provides type safety + validation)
            final_config = OmegaConf.merge(omega_config, yaml_config)

            # ✅ Now this is fully type-safe!
            print(f"   ✅ Loaded problem name: {final_config.problem.name}")
            print(f"   ✅ Loaded time horizon: {final_config.problem.T}")
            print(f"   ✅ Loaded grid size: {final_config.problem.Nx}")
            print(f"   ✅ Solver max iterations: {final_config.solver.max_iterations}")
            print(f"   ✅ Experiment name: {final_config.experiment.name}")
            print()

            # 4. Demonstrate that the config is still a proper dataclass
            print("4️⃣  Configuration object inspection...")
            print(f"   ✅ Type: {type(final_config)}")
            print(f"   ✅ Has problem attr: {hasattr(final_config, 'problem')}")
            print(f"   ✅ Has solver attr: {hasattr(final_config, 'solver')}")
            print(f"   ✅ Problem type: {type(final_config.problem)}")
            print()

        finally:
            # Clean up
            temp_yaml.unlink()

    print("✅ === DEMONSTRATION COMPLETE ===")
    print()
    print("🎯 KEY BENEFITS:")
    print("   • Full mypy type checking compatibility")
    print("   • IDE autocompletion for all configuration attributes")
    print("   • Runtime validation via dataclass schemas")
    print("   • Seamless integration with existing YAML files")
    print("   • Zero configuration access errors")


def demonstrate_old_vs_new():
    """Show the difference between old and new approaches."""
    print()
    print("📊 === OLD vs NEW COMPARISON ===")
    print()

    print("❌ OLD WAY (Type checking fails):")
    print("   conf: DictConfig = OmegaConf.load('config.yaml')")
    print("   print(conf.problem.T)  # ❌ Mypy error: DictConfig has no attribute 'problem'")
    print()

    print("✅ NEW WAY (Type checking passes):")
    print("   schema = OmegaConf.structured(MFGConfig)")
    print("   file_conf = OmegaConf.load('config.yaml')")
    print("   conf: MFGConfig = OmegaConf.merge(schema, file_conf)")
    print("   print(conf.problem.T)  # ✅ Type safe, autocompletes!")
    print()


if __name__ == "__main__":
    demonstrate_type_safe_config()
    demonstrate_old_vs_new()

    print("🚀 Ready to use structured configs in your MFG applications!")
    print("   See mfg_pde.config.structured_schemas for all available schemas")
    print("   Use load_structured_mfg_config() for type-safe config loading")
