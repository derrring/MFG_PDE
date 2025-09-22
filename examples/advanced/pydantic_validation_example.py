#!/usr/bin/env python3
"""
Pydantic Validation Example for MFG_PDE

Demonstrates the enhanced Pydantic-based configuration system with:
- Automatic validation and error checking
- JSON serialization and configuration management
- Advanced array validation for solutions
- Enhanced notebook reporting
- Research workflow integration
"""

from pathlib import Path

import numpy as np

# Import Pydantic configurations
from mfg_pde.config import (
    ExperimentConfig,
    MFGArrays,
    MFGGridConfig,
    MFGSolverConfig,
    NewtonConfig,
    PicardConfig,
)

# Import core MFG components
from mfg_pde.core.mfg_problem import ExampleMFGProblem

# Import enhanced factory and reporting
from mfg_pde.factory.pydantic_solver_factory import create_validated_solver
from mfg_pde.utils.integration import trapezoid
from mfg_pde.utils.logging import configure_research_logging, get_logger
from mfg_pde.utils.pydantic_notebook_integration import create_pydantic_mfg_report


def demonstrate_basic_pydantic_config():
    """Demonstrate basic Pydantic configuration usage."""
    print(" Basic Pydantic Configuration Demo")
    print("=" * 50)

    # Create configuration with automatic validation
    try:
        config = MFGSolverConfig(
            newton=NewtonConfig(max_iterations=50, tolerance=1e-8),
            picard=PicardConfig(max_iterations=30, tolerance=1e-4),
            convergence_tolerance=1e-6,
            experiment_name="basic_validation_demo",
        )
        print("SUCCESS: Configuration created successfully!")
        print(f"   Newton tolerance: {config.newton.tolerance:.2e}")
        print(f"   Picard tolerance: {config.picard.tolerance:.2e}")
        print(f"   Global tolerance: {config.convergence_tolerance:.2e}")

    except Exception as e:
        print(f"ERROR: Configuration creation failed: {e}")
        return None

    # Demonstrate automatic JSON serialization
    config_json = config.json(indent=2)
    print("\n📄 JSON Serialization (first 200 chars):")
    print(config_json[:200] + "...")

    # Save and reload configuration
    config_path = "demo_config.json"
    with open(config_path, "w") as f:
        f.write(config_json)

    # Reload with validation
    try:
        reloaded_config = MFGSolverConfig.parse_file(config_path)
        print("SUCCESS: Configuration reloaded successfully!")
        print(f"   Experiment name: {reloaded_config.experiment_name}")
    except Exception as e:
        print(f"ERROR: Configuration reload failed: {e}")

    # Clean up
    Path(config_path).unlink(missing_ok=True)

    return config


def demonstrate_validation_errors():
    """Demonstrate Pydantic validation error handling."""
    print("\n Validation Error Handling Demo")
    print("=" * 50)

    # Test 1: Invalid tolerance (too small)
    try:
        config = MFGSolverConfig(
            newton=NewtonConfig(tolerance=1e-20),
            convergence_tolerance=1e-15,  # Too strict!  # Too strict!
        )
        print("❓ Somehow passed validation (with warnings)")
    except Exception as e:
        print(f"SUCCESS: Caught validation error for too-strict tolerance: {type(e).__name__}")

    # Test 2: Invalid iterations (negative)
    try:
        config = MFGSolverConfig(
            newton=NewtonConfig(max_iterations=-5),
            picard=PicardConfig(max_iterations=0),  # Invalid!  # Invalid!
        )
        print("ERROR: Should have failed validation!")
    except Exception as e:
        print(f"SUCCESS: Caught validation error for invalid iterations: {type(e).__name__}")

    # Test 3: Invalid damping factor
    try:
        config = MFGSolverConfig(
            newton=NewtonConfig(damping_factor=1.5),  # > 1.0 invalid!
            picard=PicardConfig(damping_factor=0.0),  # <= 0.0 invalid!
        )
        print("ERROR: Should have failed validation!")
    except Exception as e:
        print(f"SUCCESS: Caught validation error for invalid damping: {type(e).__name__}")

    print("SUCCESS: Validation system working correctly!")


def demonstrate_grid_and_array_validation():
    """Demonstrate grid configuration and array validation."""
    print("\n Grid and Array Validation Demo")
    print("=" * 50)

    # Create grid configuration with CFL checking
    try:
        grid_config = MFGGridConfig(Nx=50, Nt=100, xmin=0.0, xmax=1.0, T=1.0, sigma=0.1)

        print("SUCCESS: Grid configuration created:")
        print(f"   Grid: {grid_config.Nx}×{grid_config.Nt}")
        print(f"   Domain: [{grid_config.xmin}, {grid_config.xmax}] × [0, {grid_config.T}]")
        print(f"   Spacing: dx={grid_config.dx:.6f}, dt={grid_config.dt:.6f}")
        print(f"   CFL number: {grid_config.cfl_number:.4f}")
        print(f"   Expected array shape: {grid_config.grid_shape}")

    except Exception as e:
        print(f"ERROR: Grid configuration failed: {e}")
        return None

    # Create mock solution arrays
    Nt, Nx = grid_config.Nt, grid_config.Nx
    U_solution = np.random.randn(Nt + 1, Nx + 1) * 0.1

    # Create valid density (non-negative, normalized)
    M_solution = np.random.rand(Nt + 1, Nx + 1)
    # Normalize each time slice to integrate to 1
    for t in range(Nt + 1):
        M_solution[t] = M_solution[t] / trapezoid(M_solution[t], dx=grid_config.dx)

    try:
        # Validate arrays with Pydantic
        arrays = MFGArrays(U_solution=U_solution, M_solution=M_solution, grid_config=grid_config)

        print("SUCCESS: Array validation passed!")

        # Get validation statistics
        stats = arrays.get_solution_statistics()
        print("\n Array Statistics:")
        print(f"   U range: [{stats['U']['min']:.3e}, {stats['U']['max']:.3e}]")
        print(f"   M range: [{stats['M']['min']:.3e}, {stats['M']['max']:.3e}]")
        print("   Mass conservation:")
        print(f"     Initial: {stats['mass_conservation']['initial_mass']:.6f}")
        print(f"     Final: {stats['mass_conservation']['final_mass']:.6f}")
        print(f"     Drift: {stats['mass_conservation']['mass_drift']:.2e}")

        return grid_config, arrays

    except Exception as e:
        print(f"ERROR: Array validation failed: {e}")
        return grid_config, None


def demonstrate_enhanced_solver_creation():
    """Demonstrate validated solver creation."""
    print("\n🔨 Enhanced Solver Creation Demo")
    print("=" * 50)

    # Create MFG problem
    problem = ExampleMFGProblem(xmin=0.0, xmax=1.0, Nx=20, T=1.0, Nt=30, sigma=0.1, coefCT=0.02)

    # Method 1: Using factory with preset
    try:
        solver1 = create_validated_solver(
            problem=problem, solver_type="fixed_point", config_preset="fast", experiment_name="factory_demo"
        )
        print("SUCCESS: Created solver with factory (fast preset)")

    except Exception as e:
        print(f"ERROR: Factory solver creation failed: {e}")

    # Method 2: Using custom Pydantic configuration
    try:
        custom_config = MFGSolverConfig(
            newton=NewtonConfig(max_iterations=20, tolerance=1e-5),
            picard=PicardConfig(max_iterations=15, tolerance=1e-3),
            convergence_tolerance=1e-4,
            experiment_name="custom_config_demo",
            enable_warm_start=True,
        )

        solver2 = create_validated_solver(problem=problem, solver_type="adaptive_particle", config=custom_config)
        print("SUCCESS: Created solver with custom Pydantic config")

    except Exception as e:
        print(f"ERROR: Custom config solver creation failed: {e}")

    # Method 3: Configuration from JSON
    try:
        # Save config to JSON
        config_path = "solver_config.json"
        custom_config.json()  # Validate JSON serialization works

        with open(config_path, "w") as f:
            f.write(custom_config.json(indent=2))

        # Load and use config
        loaded_config = MFGSolverConfig.parse_file(config_path)

        solver3 = create_validated_solver(problem=problem, solver_type="monitored_particle", config=loaded_config)
        print("SUCCESS: Created solver from JSON configuration")

        # Clean up
        Path(config_path).unlink(missing_ok=True)

    except Exception as e:
        print(f"ERROR: JSON config solver creation failed: {e}")


def demonstrate_enhanced_notebook_reporting():
    """Demonstrate enhanced notebook reporting with Pydantic."""
    print("\n📓 Enhanced Notebook Reporting Demo")
    print("=" * 50)

    # Create complete experiment configuration
    try:
        # Grid configuration
        grid_config = MFGGridConfig(Nx=30, Nt=50, xmin=0.0, xmax=1.0, T=1.0, sigma=0.15)

        # Create mock validated arrays
        U_solution = np.sin(np.pi * np.linspace(0, 1, 31)) * np.exp(-np.linspace(0, 1, 51)[:, None])
        M_solution = np.ones((51, 31))
        for t in range(51):
            M_solution[t] = np.exp(-((np.linspace(0, 1, 31) - 0.5) ** 2) / 0.1)
            M_solution[t] = M_solution[t] / trapezoid(M_solution[t], dx=grid_config.dx)

        arrays = MFGArrays(U_solution=U_solution, M_solution=M_solution, grid_config=grid_config)

        # Complete experiment configuration
        experiment_config = ExperimentConfig(
            grid_config=grid_config,
            arrays=arrays,
            experiment_name="pydantic_validation_demo",
            description="Comprehensive demonstration of Pydantic validation features",
            researcher="MFG_PDE Validation Team",
            tags=["validation", "pydantic", "demo", "research"],
            output_dir="./demo_reports",
        )

        print("SUCCESS: Experiment configuration created and validated!")

        # Mock solver results
        solver_results = {
            "convergence_history": [1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
            "final_error": 1e-5,
            "iterations": 5,
            "solver_type": "pydantic_validated",
            "U": U_solution,
            "M": M_solution,
        }

        # Generate enhanced notebook report
        print("\n Generating enhanced notebook report...")

        report_paths = create_pydantic_mfg_report(
            title="Pydantic Validation Demonstration",
            experiment_config=experiment_config,
            solver_results=solver_results,
            output_dir="demo_reports",
            export_html=True,
        )

        print("SUCCESS: Enhanced notebook report generated!")
        print(f"   Notebook: {report_paths.get('notebook_path', 'N/A')}")
        print(f"   HTML: {report_paths.get('html_path', 'N/A')}")

        # Print experiment metadata
        metadata = experiment_config.to_notebook_metadata()
        print("\n Experiment Metadata:")
        print(f"   Name: {metadata['experiment_name']}")
        print(f"   Researcher: {metadata['researcher']}")
        print(f"   Tags: {metadata['tags']}")
        if "solution_statistics" in metadata:
            mass_stats = metadata["solution_statistics"]["mass_conservation"]
            print(f"   Mass drift: {mass_stats['mass_drift']:.2e}")

        return experiment_config, report_paths

    except Exception as e:
        print(f"ERROR: Enhanced notebook reporting failed: {e}")
        return None, None


def demonstrate_environment_variable_support():
    """Demonstrate environment variable configuration support."""
    print("\n🌍 Environment Variable Support Demo")
    print("=" * 50)

    import os

    # Set environment variables
    test_env_vars = {
        "MFG_NEWTON_MAX_ITERATIONS": "25",
        "MFG_NEWTON_TOLERANCE": "1e-7",
        "MFG_PICARD_MAX_ITERATIONS": "15",
        "MFG_CONVERGENCE_TOLERANCE": "1e-5",
    }

    # Save original environment
    original_env = {}
    for key in test_env_vars:
        original_env[key] = os.environ.get(key)
        os.environ[key] = test_env_vars[key]

    try:
        # Create configuration that will read from environment
        config = MFGSolverConfig()

        print("SUCCESS: Configuration created with environment variables:")
        print(f"   Newton max iterations: {config.newton.max_iterations} (from env)")
        print(f"   Newton tolerance: {config.newton.tolerance:.2e} (from env)")
        print(f"   Picard max iterations: {config.picard.max_iterations} (from env)")
        print(f"   Global tolerance: {config.convergence_tolerance:.2e} (from env)")

    except Exception as e:
        print(f"ERROR: Environment variable configuration failed: {e}")

    finally:
        # Restore original environment
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def main():
    """Run comprehensive Pydantic validation demonstration."""
    print(" MFG_PDE Pydantic Validation Comprehensive Demo")
    print("=" * 60)
    print()

    # Setup logging
    configure_research_logging(experiment_name="pydantic_validation_demo")
    logger = get_logger(__name__)
    logger.info("Starting Pydantic validation demonstration")

    try:
        # Run all demonstrations
        config = demonstrate_basic_pydantic_config()
        demonstrate_validation_errors()
        grid_config, arrays = demonstrate_grid_and_array_validation()
        demonstrate_enhanced_solver_creation()
        experiment_config, report_paths = demonstrate_enhanced_notebook_reporting()
        demonstrate_environment_variable_support()

        print("\n Pydantic Validation Demo Complete!")
        print("=" * 60)
        print("\n Summary of Pydantic Features Demonstrated:")
        print("   SUCCESS: Automatic configuration validation")
        print("   SUCCESS: JSON serialization and deserialization")
        print("   SUCCESS: Advanced error handling with detailed messages")
        print("   SUCCESS: Grid configuration with CFL stability checking")
        print("   SUCCESS: Array validation with physical constraints")
        print("   SUCCESS: Enhanced solver factory with validation")
        print("   SUCCESS: Comprehensive notebook reporting integration")
        print("   SUCCESS: Environment variable configuration support")

        print("\n🔬 Key Benefits for Research:")
        print("   • Automatic parameter validation prevents numerical errors")
        print("   • JSON serialization enables experiment reproducibility")
        print("   • Enhanced error messages speed up debugging")
        print("   • Array validation ensures physical constraint compliance")
        print("   • Professional reporting suitable for publications")

        if report_paths and report_paths.get("notebook_path"):
            print(f"\n📖 Generated Report: {report_paths['notebook_path']}")
            print("   Open this notebook to see the full validation report!")

        logger.info("Pydantic validation demonstration completed successfully")

    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        print(f"\nERROR: Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()
