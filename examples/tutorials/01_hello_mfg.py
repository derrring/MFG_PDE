"""
Tutorial 01: Hello Mean Field Games

This is the simplest possible MFG example using the Linear-Quadratic (LQ) framework.

What you'll learn:
- How to define an MFG problem using MFGProblem
- How to solve it using the problem.solve() method
- How to inspect the solution

Mathematical Problem:
    A Linear-Quadratic Mean Field Game on [0,1] with:
    - Hamiltonian: H(p) = (1/2)|p|^2 + coupling * m
    - Initial density: Gaussian centered at 0.5
    - Terminal cost: Quadratic penalty u_T(x) = (x - 0.5)^2
"""

import numpy as np

from mfg_pde import MFGProblem
from mfg_pde.core import MFGComponents
from mfg_pde.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfg_pde.geometry import TensorProductGrid
from mfg_pde.geometry.boundary import no_flux_bc

# ==============================================================================
# Step 1: Define the Domain
# ==============================================================================

print("=" * 70)
print("TUTORIAL 01: Hello Mean Field Games")
print("=" * 70)
print()

# Create a 1D spatial grid
grid = TensorProductGrid(
    bounds=[(0.0, 1.0)],  # Domain [0, 1]
    Nx_points=[51],  # 51 grid points (50 intervals)
    boundary_conditions=no_flux_bc(dimension=1),
)

print("Grid created:")
print("  Domain: [0.0, 1.0]")
print(f"  Points: {grid.num_spatial_points}")
print()

# ==============================================================================
# Step 2: Define the MFG Components
# ==============================================================================

# Physical parameters
diffusion = 0.1  # Controls agent randomness
coupling = 0.5  # Congestion cost strength


# Terminal cost: agents want to be at x = 0.5
def terminal_cost(x):
    """Terminal cost u_T(x) = (x - 0.5)^2"""
    return (x - 0.5) ** 2


# Initial density: Gaussian centered at 0.5
def initial_density(x):
    """Gaussian initial density centered at 0.5"""
    return np.exp(-50 * (x - 0.5) ** 2)


# Hamiltonian: H(p, m) = (1/2)|p|^2 + coupling * m
hamiltonian = SeparableHamiltonian(
    control_cost=QuadraticControlCost(control_cost=1.0),
    coupling=lambda m: coupling * m,
    coupling_dm=lambda m: coupling,
)

# Bundle all components
components = MFGComponents(
    hamiltonian=hamiltonian,
    m_initial=initial_density,
    u_terminal=terminal_cost,
)

# ==============================================================================
# Step 3: Create and Solve the Problem
# ==============================================================================

problem = MFGProblem(
    geometry=grid,
    T=1.0,  # Time horizon
    Nt=50,  # Time steps
    diffusion=diffusion,
    components=components,
)

print("Problem created:")
print(f"  Time horizon: T = {problem.T}")
print(f"  Time steps: {problem.Nt}")
print(f"  Diffusion: {diffusion}")
print()

print("Solving MFG system...")
print()

result = problem.solve(verbose=True)

print()
print("Solution completed!")
print()

# ==============================================================================
# Step 4: Inspect the Solution
# ==============================================================================

print("=" * 70)
print("SOLUTION SUMMARY")
print("=" * 70)
print()

print(f"Converged: {result.converged}")
print(f"Iterations: {result.iterations}")
print(f"Final error: {result.max_error:.6e}")
print()

# Solution arrays
print("Solution arrays:")
print(f"  U (value function): {result.U.shape}")
print(f"  M (density): {result.M.shape}")
print()

# Mass conservation check
dx = grid.get_grid_spacing()[0]
total_mass = np.sum(result.M[-1, :]) * dx
print(f"Final mass (should be ~1.0): {total_mass:.6f}")
print()

# ==============================================================================
# Step 5: Visualize (Optional)
# ==============================================================================

print("=" * 70)
print("VISUALIZATION")
print("=" * 70)
print()

try:
    from pathlib import Path

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    x = grid.coordinates[0]
    t = np.linspace(0, problem.T, problem.Nt + 1)

    # Plot value function at final time
    axes[0].plot(x, result.U[-1, :])
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("u(T, x)")
    axes[0].set_title("Terminal Value Function")
    axes[0].grid(True, alpha=0.3)

    # Plot density evolution
    X, T_grid = np.meshgrid(x, t)
    contour = axes[1].contourf(X, T_grid, result.M, levels=20, cmap="viridis")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("t")
    axes[1].set_title("Density Evolution m(t,x)")
    plt.colorbar(contour, ax=axes[1])

    plt.tight_layout()

    output_dir = Path(__file__).parent.parent / "outputs" / "tutorials"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "01_hello_mfg.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to: {output_path}")
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
print("  1. Create a grid with TensorProductGrid")
print("  2. Define components: hamiltonian, initial density, terminal cost")
print("  3. Bundle components in MFGComponents")
print("  4. Create and solve with MFGProblem")
print("  5. Inspect the solution (U, M, convergence)")
print()
print("Next: Tutorial 02 - Custom Hamiltonian")
print("=" * 70)
