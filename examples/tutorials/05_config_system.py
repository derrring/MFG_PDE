"""
Tutorial 05: ConfigBuilder System

Learn how to use the ConfigBuilder API to configure MFG solvers.

What you'll learn:
- How to build solver configurations with ConfigBuilder
- How to choose between different solver backends (FDM, GFDM, particles)
- How to configure coupling methods (Picard, Newton)
- How to enable acceleration (JAX, GPU)
- How to save and load configurations

The ConfigBuilder provides a fluent API for creating solver configurations.
It's the recommended way to configure solve_mfg().
"""

from mfg_pde import ExampleMFGProblem, solve_mfg
from mfg_pde.factory import ConfigBuilder

# ==============================================================================
# Step 1: Basic Configuration
# ==============================================================================

print("=" * 70)
print("TUTORIAL 05: ConfigBuilder System")
print("=" * 70)
print()

problem = ExampleMFGProblem(xmin=0.0, xmax=1.0, Nx=50, T=1.0, Nt=50, sigma=0.1, lam=0.5)

print("METHOD 1: Default configuration (implicit)")
print("-" * 70)

# Without config, solve_mfg uses default:
# - Picard iteration
# - FDM for both HJB and FP
# - Standard convergence criteria

result_default = solve_mfg(problem, verbose=False)
print(f"  Converged: {result_default.converged}")
print(f"  Iterations: {result_default.iterations}")
print()

# ==============================================================================
# Step 2: Explicit Configuration
# ==============================================================================

print("METHOD 2: Explicit configuration with ConfigBuilder")
print("-" * 70)
print()

# Build configuration step-by-step
config = (
    ConfigBuilder()
    .picard(  # Coupling method: Picard fixed-point iteration
        max_iterations=30,  # Maximum number of iterations
        tolerance=1e-4,  # Convergence tolerance
    )
    .solver_hjb(  # HJB solver backend
        "fdm",  # Finite Difference Method
        scheme="upwind",  # Upwind scheme for stability
    )
    .solver_fp(  # FP solver backend
        "fdm",  # Finite Difference Method
        scheme="lax_friedrichs",  # Lax-Friedrichs scheme
    )
    .build()  # Create the configuration object
)

print("Configuration built:")
print(f"  Coupling: {config.coupling_config.method}")
print(f"  HJB solver: {config.hjb_config.backend}")
print(f"  FP solver: {config.fp_config.backend}")
print()

result_explicit = solve_mfg(problem, config=config, verbose=False)
print(f"  Converged: {result_explicit.converged}")
print(f"  Iterations: {result_explicit.iterations}")
print()

# ==============================================================================
# Step 3: Advanced Solver Selection
# ==============================================================================

print("=" * 70)
print("ADVANCED SOLVER CONFIGURATIONS")
print("=" * 70)
print()

# Configuration A: GFDM (Generalized Finite Difference Method)
print("Config A: GFDM solvers (higher-order accuracy)")
print("-" * 70)

config_gfdm = ConfigBuilder().picard(max_iterations=30, tolerance=1e-4).solver_hjb("gfdm").solver_fp("gfdm").build()

result_gfdm = solve_mfg(problem, config=config_gfdm, verbose=False)
print(f"  Converged: {result_gfdm.converged} in {result_gfdm.iterations} iterations")
print()

# Configuration B: Particle-based FP
print("Config B: Hybrid (FDM HJB + Particle FP)")
print("-" * 70)

config_hybrid = (
    ConfigBuilder()
    .picard(max_iterations=30, tolerance=1e-4)
    .solver_hjb("fdm")
    .solver_fp("particle", num_particles=3000, kde_bandwidth="scott")
    .build()
)

result_hybrid = solve_mfg(problem, config=config_hybrid, verbose=False)
print(f"  Converged: {result_hybrid.converged} in {result_hybrid.iterations} iterations")
print()

# Configuration C: Policy iteration (for LQ problems)
print("Config C: Policy Iteration (for LQ problems)")
print("-" * 70)

config_policy = ConfigBuilder().policy_iteration(max_iterations=10, tolerance=1e-5).solver_fp("fdm").build()

result_policy = solve_mfg(problem, config=config_policy, verbose=False)
print(f"  Converged: {result_policy.converged} in {result_policy.iterations} iterations")
print()

# ==============================================================================
# Step 4: Acceleration Options
# ==============================================================================

print("=" * 70)
print("ACCELERATION OPTIONS")
print("=" * 70)
print()

print("JAX Acceleration (if available):")
print("-" * 70)

try:
    import jax  # noqa: F401

    config_jax = (
        ConfigBuilder()
        .picard(max_iterations=30, tolerance=1e-4)
        .solver_hjb("fdm")
        .solver_fp("fdm")
        .acceleration("jax")  # Enable JAX acceleration
        .build()
    )

    print("  JAX available - acceleration enabled")
    result_jax = solve_mfg(problem, config=config_jax, verbose=False)
    print(f"  Converged: {result_jax.converged} in {result_jax.iterations} iterations")

except ImportError:
    print("  JAX not available - skipping JAX acceleration")

print()

# ==============================================================================
# Step 5: Configuration Comparison
# ==============================================================================

print("=" * 70)
print("CONFIGURATION COMPARISON")
print("=" * 70)
print()

configurations = {
    "Default (FDM)": result_default,
    "Explicit (FDM)": result_explicit,
    "GFDM": result_gfdm,
    "Hybrid (Particle)": result_hybrid,
    "Policy Iteration": result_policy,
}

print(f"{'Configuration':<25} {'Converged':<12} {'Iterations':<12} {'Residual':<12}")
print("-" * 70)

for name, result in configurations.items():
    print(f"{name:<25} {result.converged!s:<12} {result.iterations:<12} {result.residual:.4e}")

print()

# ==============================================================================
# Step 6: Configuration Best Practices
# ==============================================================================

print("=" * 70)
print("CONFIGURATION BEST PRACTICES")
print("=" * 70)
print()

print("General Guidelines:")
print("  1. START SIMPLE: Use default config first, then customize")
print("  2. MATCH PROBLEM: Choose solvers appropriate for your problem")
print("  3. TUNE PARAMETERS: Adjust tolerances and iterations as needed")
print("  4. TEST CONVERGENCE: Always check result.converged flag")
print()

print("Solver Selection Guide:")
print()
print("  Problem Type              | Recommended Config")
print("  " + "-" * 68)
print("  1D/2D, smooth domain      | FDM (default)")
print("  High-order accuracy       | GFDM")
print("  3D+, complex geometry     | Particle FP + FDM HJB")
print("  Linear-Quadratic          | Policy Iteration")
print("  Stiff problems            | Semi-Lagrangian HJB")
print()

print("Performance Tips:")
print("  - Use JAX acceleration for repeated solves")
print("  - Use GPU backend for particle methods (if available)")
print("  - Reduce grid resolution (Nx, Nt) for prototyping")
print("  - Increase tolerance for faster (but less accurate) solves")
print()

# ==============================================================================
# Step 7: Save and Load Configurations
# ==============================================================================

print("=" * 70)
print("SAVE AND LOAD CONFIGURATIONS")
print("=" * 70)
print()

# Configurations can be saved to JSON for reproducibility
print("Saving configuration to JSON...")

config_to_save = (
    ConfigBuilder()
    .picard(max_iterations=50, tolerance=1e-5)
    .solver_hjb("fdm", scheme="upwind")
    .solver_fp("particle", num_particles=5000, kde_bandwidth="scott")
    .build()
)

# Note: This would require implementing to_dict() method on config
# For now, we document the pattern
print("  Config attributes:")
print(f"    coupling_config.max_iterations = {config_to_save.coupling_config.max_iterations}")
print(f"    coupling_config.tolerance = {config_to_save.coupling_config.tolerance}")
print(f"    hjb_config.backend = {config_to_save.hjb_config.backend}")
print(f"    fp_config.backend = {config_to_save.fp_config.backend}")
print()

print("  (Full save/load functionality requires config.to_dict() implementation)")
print()

# ==============================================================================
# Summary
# ==============================================================================

print("=" * 70)
print("TUTORIAL COMPLETE")
print("=" * 70)
print()
print("What you learned:")
print("  1. How to use ConfigBuilder for creating solver configurations")
print("  2. How to select between different solver backends")
print("  3. How to choose coupling methods (Picard vs Policy Iteration)")
print("  4. How to compare configurations and measure performance")
print("  5. Best practices for configuration selection")
print()
print("Key Takeaways:")
print("  - ConfigBuilder provides a clean, type-safe API")
print("  - Different problems benefit from different solver choices")
print("  - Always validate convergence before trusting results")
print("  - Configuration is about trade-offs: speed vs accuracy")
print()
print("=" * 70)
print("TUTORIAL SERIES COMPLETE!")
print("=" * 70)
print()
print("You've completed all 5 tutorials:")
print("  01: Hello MFG - Basic problem setup")
print("  02: Custom Hamiltonian - Problem definition")
print("  03: 2D Geometry - Multi-dimensional problems")
print("  04: Particle Methods - Alternative solver backends")
print("  05: ConfigBuilder System - Advanced configuration")
print()
print("Next steps:")
print("  - Explore examples/basic/ for single-concept demos")
print("  - Explore examples/advanced/ for complex applications")
print("  - Read docs/ for mathematical theory and API reference")
print("  - Join community discussions on GitHub")
print()
print("Happy MFG solving!")
print("=" * 70)
