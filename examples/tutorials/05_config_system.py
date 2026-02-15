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
from mfg_pde.core import MFGComponents
from mfg_pde.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfg_pde.geometry import TensorProductGrid
from mfg_pde.geometry.boundary import no_flux_bc

# ==============================================================================
# Helper: Create Standard LQ Components
# ==============================================================================


def create_lq_components(coupling_strength: float = 0.5):
    """
    Create standard Linear-Quadratic MFG components.

    This is the default setup used throughout this tutorial:
    - Hamiltonian: H(p,m) = (1/2)|p|^2 + coupling * m
    - Terminal cost: (x - 0.5)^2 (agents want to be at center)
    - Initial density: Gaussian at center
    """
    hamiltonian = SeparableHamiltonian(
        control_cost=QuadraticControlCost(control_cost=1.0),
        coupling=lambda m: coupling_strength * m,
        coupling_dm=lambda m: coupling_strength,
    )

    return MFGComponents(
        hamiltonian=hamiltonian,
        m_initial=lambda x: np.exp(-50 * (x - 0.5) ** 2),
        u_terminal=lambda x: (x - 0.5) ** 2,
    )


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

print("Varying diffusion changes how agents spread out.")
print()

# Test different diffusion coefficients
diffusion_values = [0.05, 0.15, 0.30]
results_diffusion = {}

for sigma in diffusion_values:
    print(f"Solving with sigma={sigma}...")
    geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51], boundary_conditions=no_flux_bc(dimension=1))
    problem = MFGProblem(
        geometry=geometry,
        T=1.0,
        Nt=50,
        sigma=sigma,
        components=create_lq_components(coupling_strength=0.5),
    )
    results_diffusion[sigma] = problem.solve(verbose=False)
    print(f"  Converged in {results_diffusion[sigma].iterations} iterations")

print()

# ==============================================================================
# Step 3: Coupling Strength Study
# ==============================================================================

print("=" * 70)
print("COUPLING STRENGTH STUDY")
print("=" * 70)
print()

print("Varying coupling strength changes congestion effects.")
print()

# Test different coupling strengths
coupling_values = [0.0, 0.5, 2.0]
results_coupling = {}

for coupling in coupling_values:
    print(f"Solving with coupling={coupling}...")
    geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[51], boundary_conditions=no_flux_bc(dimension=1))
    problem = MFGProblem(
        geometry=geometry,
        T=1.0,
        Nt=50,
        sigma=0.15,
        components=create_lq_components(coupling_strength=coupling),
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
Nx_values = [26, 51, 101]  # Number of grid points
results_grid = {}

for Nx in Nx_values:
    print(f"Solving with Nx={Nx} grid points...")
    geometry = TensorProductGrid(bounds=[(0.0, 1.0)], Nx_points=[Nx], boundary_conditions=no_flux_bc(dimension=1))
    problem = MFGProblem(
        geometry=geometry,
        T=1.0,
        Nt=Nx,  # Keep dt/dx ratio constant
        sigma=0.15,
        components=create_lq_components(coupling_strength=0.5),
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
print("Mass Conservation Check (diffusion variations):")
for sigma, result in results_diffusion.items():
    # Use the stored result's shape to compute dx
    Nx = result.M.shape[1]
    dx = 1.0 / (Nx - 1)
    initial_mass = np.sum(result.M[0, :]) * dx
    final_mass = np.sum(result.M[-1, :]) * dx
    print(
        f"  sigma={sigma}: Initial={initial_mass:.4f}, "
        f"Final={final_mass:.4f}, Drift={abs(final_mass - initial_mass):.2e}"
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
print("  diffusion              | More spreading, smoother density")
print("  coupling_strength      | Stronger congestion avoidance")
print("  Nx (grid resolution)   | Higher accuracy, more computation")
print("  T (time horizon)       | More time for dynamics to evolve")
print()

# ==============================================================================
# Step 7: Visualization
# ==============================================================================

print("=" * 70)
print("VISUALIZATION")
print("=" * 70)
print()

try:
    from pathlib import Path

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Get x coordinates for plotting
    x = np.linspace(0, 1, 51)

    # Row 1: Diffusion study - final densities
    ax = axes[0, 0]
    for sigma, result in results_diffusion.items():
        x_grid = np.linspace(0, 1, result.M.shape[1])
        ax.plot(x_grid, result.M[-1, :], label=f"sigma={sigma}")
    ax.set_xlabel("x")
    ax.set_ylabel("m(T, x)")
    ax.set_title("Final Density: Diffusion Study")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Row 1: Coupling study - final densities
    ax = axes[0, 1]
    for coupling, result in results_coupling.items():
        x_grid = np.linspace(0, 1, result.M.shape[1])
        ax.plot(x_grid, result.M[-1, :], label=f"coup={coupling}")
    ax.set_xlabel("x")
    ax.set_ylabel("m(T, x)")
    ax.set_title("Final Density: Coupling Study")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Row 1: Grid study - final densities
    ax = axes[0, 2]
    for Nx, result in results_grid.items():
        x_grid = np.linspace(0, 1, result.M.shape[1])
        ax.plot(x_grid, result.M[-1, :], label=f"Nx={Nx}")
    ax.set_xlabel("x")
    ax.set_ylabel("m(T, x)")
    ax.set_title("Final Density: Resolution Study")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Row 2: Value functions
    ax = axes[1, 0]
    for sigma, result in results_diffusion.items():
        x_grid = np.linspace(0, 1, result.U.shape[1])
        ax.plot(x_grid, result.U[-1, :], label=f"sigma={sigma}")
    ax.set_xlabel("x")
    ax.set_ylabel("u(T, x)")
    ax.set_title("Terminal Value: Diffusion Study")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    for coupling, result in results_coupling.items():
        x_grid = np.linspace(0, 1, result.U.shape[1])
        ax.plot(x_grid, result.U[-1, :], label=f"coup={coupling}")
    ax.set_xlabel("x")
    ax.set_ylabel("u(T, x)")
    ax.set_title("Terminal Value: Coupling Study")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    for Nx, result in results_grid.items():
        x_grid = np.linspace(0, 1, result.U.shape[1])
        ax.plot(x_grid, result.U[-1, :], label=f"Nx={Nx}")
    ax.set_xlabel("x")
    ax.set_ylabel("u(T, x)")
    ax.set_title("Terminal Value: Resolution Study")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_dir = Path(__file__).parent.parent / "outputs" / "tutorials"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "05_config_system.png"
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
print("  1. Create parameter variations using helper functions")
print("  2. Compare diffusion, coupling, and resolution effects")
print("  3. Analyze mass conservation and convergence")
print("  4. Visualize parameter studies")
print()
print("Next steps:")
print("  - Explore examples/advanced/ for complex geometries")
print("  - See examples/applications/ for real-world problems")
print("  - Check docs/theory/ for mathematical background")
print()
print("=" * 70)
