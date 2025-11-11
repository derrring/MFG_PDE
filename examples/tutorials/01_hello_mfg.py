"""
Tutorial 01: Hello Mean Field Games

This is the simplest possible MFG example using the Linear-Quadratic (LQ) framework.

What you'll learn:
- How to define an MFG problem using ExampleMFGProblem
- How to solve it using the solve_mfg() function
- How to inspect the solution

Mathematical Problem:
    A Linear-Quadratic Mean Field Game on [0,1] with:
    - Hamiltonian: H(p) = (1/2)|p|^2
    - Coupling: λ * m (congestion cost)
    - Initial density: Gaussian centered at 0.5
    - Terminal cost: Quadratic penalty g(x) = (1/2)(x - 0.5)^2
"""

import numpy as np

from mfg_pde import ExampleMFGProblem, solve_mfg

# ==============================================================================
# Step 1: Create the Problem
# ==============================================================================

print("=" * 70)
print("TUTORIAL 01: Hello Mean Field Games")
print("=" * 70)
print()

# The ExampleMFGProblem is a pre-configured Linear-Quadratic MFG problem
# It's the simplest way to get started with MFG_PDE

problem = ExampleMFGProblem(
    # Spatial domain: [0, 1]
    xmin=0.0,
    xmax=1.0,
    Nx=50,  # Number of spatial grid points
    # Time horizon: [0, 1]
    T=1.0,
    Nt=50,  # Number of time steps
    # Diffusion coefficient (controls agent randomness)
    sigma=0.1,
    # Congestion parameter (controls interaction strength)
    lam=0.5,
)

print("Problem created:")
print(f"  Domain: [{problem.xmin}, {problem.xmax}]")
print(f"  Grid: {problem.Nx} spatial points, {problem.Nt} time steps")
print(f"  Diffusion σ = {problem.sigma}")
print(f"  Congestion λ = {problem.lam}")
print()

# ==============================================================================
# Step 2: Solve the MFG
# ==============================================================================

print("Solving MFG system...")
print()

# The solve_mfg() function solves the coupled HJB-FP system
# using a Picard fixed-point iteration (alternating HJB and FP solves)

result = solve_mfg(problem, verbose=True)

print()
print("Solution completed!")
print()

# ==============================================================================
# Step 3: Inspect the Solution
# ==============================================================================

print("=" * 70)
print("SOLUTION SUMMARY")
print("=" * 70)
print()

# The result object contains:
# - U: Value function (solution to HJB equation)
# - M: Density (solution to Fokker-Planck equation)
# - Convergence information

print(f"Converged: {result.converged}")
print(f"Iterations: {result.iterations}")
print(f"Final residual: {result.residual:.6e}")
print()

# Solution arrays are (Nt+1, Nx) shaped grids
print("Solution arrays:")
print(f"  U (value function): {result.U.shape}")
print(f"  M (density): {result.M.shape}")
print()

# Mass conservation check (important for validating the solution)
total_mass = np.sum(result.M[-1, :]) * problem.dx
print(f"Final mass (should be ≈1.0): {total_mass:.6f}")
print()

# ==============================================================================
# Step 4: Visualize (Optional)
# ==============================================================================

print("=" * 70)
print("VISUALIZATION")
print("=" * 70)
print()

try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Plot value function at final time
    axes[0].plot(problem.xSpace, result.U[-1, :])
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("u(T, x)")
    axes[0].set_title("Terminal Value Function")
    axes[0].grid(True, alpha=0.3)

    # Plot density evolution
    X, T = np.meshgrid(problem.xSpace, problem.tSpace)
    contour = axes[1].contourf(X, T, result.M, levels=20, cmap="viridis")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("t")
    axes[1].set_title("Density Evolution m(t,x)")
    plt.colorbar(contour, ax=axes[1])

    plt.tight_layout()
    plt.savefig("examples/outputs/tutorials/01_hello_mfg.png", dpi=150, bbox_inches="tight")
    print("Saved plot to: examples/outputs/tutorials/01_hello_mfg.png")
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
print("  1. How to create an MFG problem using ExampleMFGProblem")
print("  2. How to solve it with solve_mfg()")
print("  3. How to inspect the solution (U, M, convergence)")
print("  4. How to check mass conservation")
print()
print("Next: Tutorial 02 - Custom Hamiltonian")
print("=" * 70)
