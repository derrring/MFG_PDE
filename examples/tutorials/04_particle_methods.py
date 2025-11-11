"""
Tutorial 04: Particle Methods

Learn how to use particle-based solvers for the Fokker-Planck equation.

What you'll learn:
- The difference between grid-based (FDM) and particle-based (FP) solvers
- How to configure particle methods using ConfigBuilder
- When to use particles vs grids
- How Kernel Density Estimation (KDE) works

Mathematical Background:
    Particle methods solve the FP equation by simulating individual agents:
    - Each particle follows: dX_t = -∇_p H(X_t, ∇u, m) dt + σ dW_t
    - Density m is estimated from particles using KDE
    - Advantages: Natural for high dimensions, handles complex geometries
    - Disadvantages: Statistical noise, requires many particles
"""

import numpy as np

from mfg_pde import ExampleMFGProblem, solve_mfg
from mfg_pde.factory import ConfigBuilder

# ==============================================================================
# Step 1: Standard Grid-Based Solution
# ==============================================================================

print("=" * 70)
print("TUTORIAL 04: Particle Methods")
print("=" * 70)
print()

# Create a simple 1D problem
problem = ExampleMFGProblem(
    xmin=0.0,
    xmax=1.0,
    Nx=50,
    T=1.0,
    Nt=50,
    sigma=0.15,
    lam=0.3,
)

print("Solving with GRID-BASED methods (default)...")
print("-" * 70)

# Default configuration: FDM for both HJB and FP
config_grid = ConfigBuilder().picard(max_iterations=30, tolerance=1e-4).solver_hjb("fdm").solver_fp("fdm").build()

result_grid = solve_mfg(problem, config=config_grid, verbose=True)

print()
print(f"Grid-based: Converged in {result_grid.iterations} iterations")
print(f"  Final residual: {result_grid.residual:.6e}")
print()

# ==============================================================================
# Step 2: Particle-Based Solution
# ==============================================================================

print("=" * 70)
print("Solving with PARTICLE METHODS...")
print("-" * 70)
print()

# Configure particle solver for FP equation
# HJB still uses FDM (particles don't solve backward PDE well)
config_particle = (
    ConfigBuilder()
    .picard(max_iterations=30, tolerance=1e-4)
    .solver_hjb("fdm")
    .solver_fp(
        "particle",
        num_particles=5000,  # Number of particles to simulate
        kde_bandwidth="scott",  # Automatic bandwidth selection
    )
    .build()
)

result_particle = solve_mfg(problem, config=config_particle, verbose=True)

print()
print(f"Particle-based: Converged in {result_particle.iterations} iterations")
print(f"  Final residual: {result_particle.residual:.6e}")
print()

# ==============================================================================
# Step 3: Compare Solutions
# ==============================================================================

print("=" * 70)
print("COMPARISON: Particle vs Grid")
print("=" * 70)
print()

# L2 error between solutions
density_error = np.linalg.norm(result_particle.M - result_grid.M) / np.linalg.norm(result_grid.M)
value_error = np.linalg.norm(result_particle.U - result_grid.U) / np.linalg.norm(result_grid.U)

print("Relative L2 errors:")
print(f"  Density (M): {density_error:.4e}")
print(f"  Value (U):   {value_error:.4e}")
print()

# Mass conservation
mass_grid = np.sum(result_grid.M[-1, :]) * problem.dx
mass_particle = np.sum(result_particle.M[-1, :]) * problem.dx

print("Final mass conservation:")
print(f"  Grid:     {mass_grid:.6f}")
print(f"  Particle: {mass_particle:.6f}")
print()

# ==============================================================================
# Step 4: Particle Count Sensitivity
# ==============================================================================

print("=" * 70)
print("PARTICLE COUNT SENSITIVITY")
print("=" * 70)
print()

print("Testing different particle counts...")
print("(This demonstrates the bias-variance tradeoff)")
print()

particle_counts = [500, 2000, 5000]
errors = []

for n_particles in particle_counts:
    print(f"  n_particles = {n_particles:5d}... ", end="", flush=True)

    config = (
        ConfigBuilder()
        .picard(max_iterations=30, tolerance=1e-4)
        .solver_hjb("fdm")
        .solver_fp("particle", num_particles=n_particles, kde_bandwidth="scott")
        .build()
    )

    result = solve_mfg(problem, config=config, verbose=False)

    error = np.linalg.norm(result.M - result_grid.M) / np.linalg.norm(result_grid.M)
    errors.append(error)

    print(f"Error: {error:.4e}")

print()
print("Observation: More particles → Better accuracy, but slower")
print()

# ==============================================================================
# Step 5: When to Use Particle Methods
# ==============================================================================

print("=" * 70)
print("WHEN TO USE PARTICLE METHODS")
print("=" * 70)
print()

print("USE PARTICLES when:")
print("  ✓ High-dimensional problems (d ≥ 3)")
print("  ✓ Complex/irregular geometries")
print("  ✓ Sparse distributions (particles avoid empty regions)")
print("  ✓ Stochastic problems with many noise dimensions")
print()

print("USE GRIDS (FDM) when:")
print("  ✓ Low-dimensional problems (d ≤ 2)")
print("  ✓ High accuracy required")
print("  ✓ Smooth, regular geometries")
print("  ✓ Computational budget is tight (fewer resources)")
print()

print("HYBRID APPROACH:")
print("  ✓ Particle-based FP + Grid-based HJB (most common)")
print("  ✓ Grid-based FP + Particle-based collocation for high-d")
print()

# ==============================================================================
# Step 6: Visualize
# ==============================================================================

try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Final density comparison
    axes[0, 0].plot(problem.xSpace, result_grid.M[-1, :], "b-", linewidth=2, label="Grid (FDM)")
    axes[0, 0].plot(problem.xSpace, result_particle.M[-1, :], "r--", linewidth=2, label="Particle", alpha=0.7)
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("m(T, x)")
    axes[0, 0].set_title("Final Density: Particle vs Grid")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Density evolution (particle)
    X, T = np.meshgrid(problem.xSpace, problem.tSpace)
    contour = axes[0, 1].contourf(X, T, result_particle.M, levels=20, cmap="viridis")
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("t")
    axes[0, 1].set_title(f"Particle Method Density (N={config_particle.fp_config.num_particles})")
    plt.colorbar(contour, ax=axes[0, 1])

    # Plot 3: Error vs particle count
    axes[1, 0].semilogy(particle_counts, errors, "o-", linewidth=2, markersize=8)
    axes[1, 0].set_xlabel("Number of Particles")
    axes[1, 0].set_ylabel("Relative L2 Error")
    axes[1, 0].set_title("Accuracy vs Particle Count")
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Density difference (error heatmap)
    diff = result_particle.M - result_grid.M
    contour_diff = axes[1, 1].contourf(X, T, diff, levels=20, cmap="RdBu_r")
    axes[1, 1].set_xlabel("x")
    axes[1, 1].set_ylabel("t")
    axes[1, 1].set_title("Density Difference: Particle - Grid")
    plt.colorbar(contour_diff, ax=axes[1, 1])

    plt.tight_layout()
    plt.savefig("examples/outputs/tutorials/04_particle_methods.png", dpi=150, bbox_inches="tight")
    print("Saved plot to: examples/outputs/tutorials/04_particle_methods.png")
    print()

except ImportError:
    print("Matplotlib not available - skipping visualization")
    print()

# ==============================================================================
# Summary
# ==============================================================================

print("=" * 70)
print("TUTORIAL COMPLETE")
print("=" * 70)
print()
print("What you learned:")
print("  1. How to configure particle methods with ConfigBuilder")
print("  2. The difference between grid and particle solvers")
print("  3. How particle count affects accuracy")
print("  4. When to use particles vs grids")
print()
print("Key takeaway:")
print("  Particle methods are a powerful tool for high-dimensional MFGs,")
print("  but require careful tuning (num_particles, kde_bandwidth).")
print()
print("Next: Tutorial 05 - ConfigBuilder System")
print("=" * 70)
