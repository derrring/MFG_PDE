"""
Tutorial 05: Problem Variations and Next Steps

Learn how to vary problem parameters and explore the solution space.

What you'll learn:
- How to create problem variations by changing parameters
- How to compare solutions across different settings
- How to analyze MFG solution outputs
- Next steps for advanced usage

This tutorial wraps up the series and points you toward more advanced topics.
"""

import numpy as np

from mfg_pde import MFGProblem

# ==============================================================================
# Step 1: Problem Setup
# ==============================================================================

print("=" * 70)
print("TUTORIAL 05: Problem Variations and Next Steps")
print("=" * 70)
print()

print("We'll explore how changing parameters affects MFG solutions.")
print()

# ==============================================================================
# Step 2: Diffusion Parameter Study
# ==============================================================================

print("=" * 70)
print("DIFFUSION PARAMETER STUDY")
print("=" * 70)
print()

print("Varying sigma (diffusion) changes how agents spread out.")
print()

# Test different diffusion coefficients
sigma_values = [0.05, 0.15, 0.30]
results_sigma = {}

for sigma in sigma_values:
    print(f"Solving with sigma={sigma}...")
    problem = MFGProblem(
        xmin=0.0,
        xmax=1.0,
        Nx=50,
        T=1.0,
        Nt=50,
        sigma=sigma,
        coupling_coefficient=0.5,
    )
    results_sigma[sigma] = problem.solve(verbose=False)
    print(f"  Converged in {results_sigma[sigma].iterations} iterations")

print()

# ==============================================================================
# Step 3: Coupling Strength Study
# ==============================================================================

print("=" * 70)
print("COUPLING STRENGTH STUDY")
print("=" * 70)
print()

print("Varying coupling_coefficient changes congestion effects.")
print()

# Test different coupling strengths
coupling_values = [0.0, 0.5, 2.0]
results_coupling = {}

for coupling in coupling_values:
    print(f"Solving with coupling={coupling}...")
    problem = MFGProblem(
        xmin=0.0,
        xmax=1.0,
        Nx=50,
        T=1.0,
        Nt=50,
        sigma=0.15,
        coupling_coefficient=coupling,
    )
    results_coupling[coupling] = problem.solve(verbose=False)
    print(f"  Converged in {results_coupling[coupling].iterations} iterations")

print()

# ==============================================================================
# Step 4: Grid Resolution Study
# ==============================================================================

print("=" * 70)
print("GRID RESOLUTION STUDY")
print("=" * 70)
print()

print("Finer grids increase accuracy but also computation time.")
print()

# Test different resolutions
Nx_values = [25, 50, 100]
results_grid = {}

for Nx in Nx_values:
    print(f"Solving with Nx={Nx} grid points...")
    problem = MFGProblem(
        xmin=0.0,
        xmax=1.0,
        Nx=Nx,
        T=1.0,
        Nt=Nx,  # Keep dt/dx ratio constant
        diffusion=0.15,
        coupling_coefficient=0.5,
    )
    results_grid[Nx] = problem.solve(verbose=False)
    print(f"  Converged in {results_grid[Nx].iterations} iterations")

print()

# ==============================================================================
# Step 5: Solution Analysis
# ==============================================================================

print("=" * 70)
print("SOLUTION ANALYSIS")
print("=" * 70)
print()

print("Key outputs from any MFG solution:")
print()
print("  Attribute         | Description")
print("  " + "-" * 50)
print("  result.U          | Value function u(t,x)")
print("  result.M          | Density m(t,x)")
print("  result.converged  | Convergence status")
print("  result.iterations | Number of iterations")
print("  result.max_error  | Final convergence error")
print()

# Example analysis: mass conservation
print("Mass Conservation Check (sigma variations):")
for sigma, result in results_sigma.items():
    # Create a problem with matching grid to get dx
    problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=50, T=1.0, Nt=50, diffusion=sigma)
    initial_mass = np.sum(result.M[0, :]) * problem.dx
    final_mass = np.sum(result.M[-1, :]) * problem.dx
    print(
        f"  sigma={sigma}: Initial={initial_mass:.4f}, Final={final_mass:.4f}, Drift={abs(final_mass - initial_mass):.2e}"
    )

print()

# ==============================================================================
# Step 6: Summary Table
# ==============================================================================

print("=" * 70)
print("PARAMETER EFFECTS SUMMARY")
print("=" * 70)
print()

print("Effect of parameters on MFG solutions:")
print()
print("  Parameter              | High Value Effect")
print("  " + "-" * 55)
print("  sigma (diffusion)      | More spreading, smoother density")
print("  coupling_coefficient   | Stronger congestion avoidance")
print("  Nx (grid resolution)   | Higher accuracy, more computation")
print("  T (time horizon)       | More time for dynamics to evolve")
print()

# ==============================================================================
# Step 7: Visualization (Optional)
# ==============================================================================

try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Create reference problem for grid
    ref_problem = MFGProblem(xmin=0.0, xmax=1.0, Nx=50, T=1.0, Nt=50, diffusion=0.15)

    # Plot 1: Diffusion comparison (final density)
    for sigma, result in results_sigma.items():
        axes[0, 0].plot(ref_problem.xSpace, result.M[-1, :], linewidth=2, label=f"sigma={sigma}")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("m(T, x)")
    axes[0, 0].set_title("Effect of Diffusion on Final Density")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Coupling comparison (final density)
    for coupling, result in results_coupling.items():
        axes[0, 1].plot(ref_problem.xSpace, result.M[-1, :], linewidth=2, label=f"coupling={coupling}")
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("m(T, x)")
    axes[0, 1].set_title("Effect of Coupling on Final Density")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Value function comparison (diffusion)
    for sigma, result in results_sigma.items():
        axes[1, 0].plot(ref_problem.xSpace, result.U[0, :], linewidth=2, label=f"sigma={sigma}")
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel("u(0, x)")
    axes[1, 0].set_title("Effect of Diffusion on Initial Value Function")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Value function comparison (coupling)
    for coupling, result in results_coupling.items():
        axes[1, 1].plot(ref_problem.xSpace, result.U[0, :], linewidth=2, label=f"coupling={coupling}")
    axes[1, 1].set_xlabel("x")
    axes[1, 1].set_ylabel("u(0, x)")
    axes[1, 1].set_title("Effect of Coupling on Initial Value Function")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("examples/outputs/tutorials/05_problem_variations.png", dpi=150, bbox_inches="tight")
    print("Saved plot to: examples/outputs/tutorials/05_problem_variations.png")
    print()

except ImportError:
    print("Matplotlib not available - skipping visualization")
    print()

# ==============================================================================
# Step 8: Next Steps
# ==============================================================================

print("=" * 70)
print("NEXT STEPS")
print("=" * 70)
print()

print("You've completed the tutorial series! Here's what to explore next:")
print()
print("1. EXAMPLES")
print("   - examples/basic/     : Single-concept demonstrations")
print("   - examples/advanced/  : Complex applications")
print()
print("2. CUSTOM PROBLEMS")
print("   - Subclass MFGProblem for custom Hamiltonians (Tutorial 02)")
print("   - Define hamiltonian(), terminal_cost(), initial_density()")
print()
print("3. HIGHER DIMENSIONS")
print("   - Use spatial_bounds and spatial_discretization for 2D/3D")
print("   - See Tutorial 03 for 2D examples")
print()
print("4. ADVANCED SOLVERS")
print("   - MFG-Research repository for cutting-edge methods")
print("   - Particle methods, neural networks, GFDM")
print()

# ==============================================================================
# Summary
# ==============================================================================

print("=" * 70)
print("TUTORIAL SERIES COMPLETE!")
print("=" * 70)
print()
print("You've completed all 5 tutorials:")
print()
print("  01: Hello MFG           - Basic problem setup and solving")
print("  02: Custom Hamiltonian  - Problem customization")
print("  03: 2D Geometry         - Multi-dimensional problems")
print("  04: Solver Config       - Advanced configuration")
print("  05: Problem Variations  - Parameter studies and next steps")
print()
print("Key takeaways:")
print("  - Use problem.solve() for the simplest workflow")
print("  - Subclass MFGProblem for custom problems")
print("  - Use SolverConfig for advanced control")
print("  - Always check result.converged before using solutions")
print()
print("Happy MFG solving!")
print("=" * 70)
