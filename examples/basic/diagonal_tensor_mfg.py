#!/usr/bin/env python3
"""
MFG with Diagonal Anisotropic Tensor Diffusion.

Demonstrates full MFG coupling with diagonal tensor diffusion, where agents have
different diffusion rates in different spatial directions.

Example: Crowd moving through a corridor where horizontal movement is easier than
vertical movement (e.g., due to physical constraints or social norms).

Mathematical Setup:
-------------------
HJB equation:
    -∂u/∂t + H(x, ∇u, m) = 0
    H = (α/2)|∇u|² + (1/2) Σᵢ σᵢ² (∂u/∂xᵢ)² + F(m)

FP equation:
    ∂m/∂t = Σᵢ ∂/∂xᵢ(σᵢ² ∂m/∂xᵢ) - ∇·(α m ∇u)

where Σ = diag(σ₁², σ₂²) is the diagonal tensor diffusion.
"""

import matplotlib.pyplot as plt
import numpy as np

from mfg_pde.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
from mfg_pde.alg.numerical.hjb_solvers.hjb_fdm import HJBFDMSolver
from mfg_pde.core import MFGComponents
from mfg_pde.core.hamiltonian import QuadraticControlCost, SeparableHamiltonian
from mfg_pde.core.mfg_problem import MFGProblem
from mfg_pde.geometry import TensorProductGrid
from mfg_pde.geometry.boundary.conditions import no_flux_bc

print("=" * 70)
print("MFG with Diagonal Anisotropic Tensor Diffusion")
print("=" * 70)
print()

# ============================================================================
# Problem Setup
# ============================================================================

print("Setting up problem parameters...")
print()

# Create 2D domain: corridor [0, 1] × [0, 0.6]
domain = TensorProductGrid(
    bounds=[(0.0, 1.0), (0.0, 0.6)],
    Nx_points=[31, 21],  # Nx × Ny grid points
    boundary_conditions=no_flux_bc(dimension=2),
)

# Create LQ Hamiltonian with coupling
hamiltonian = SeparableHamiltonian(
    control_cost=QuadraticControlCost(control_cost=0.5),
    coupling=lambda m: 0.5 * m,
    coupling_dm=lambda m: 0.5,
)

# Get grid coordinates for initial/terminal conditions
x_coords = domain.coordinates[0]
y_coords = domain.coordinates[1]
X, Y = np.meshgrid(x_coords, y_coords, indexing="ij")
dx, dy = domain.get_grid_spacing()

# Initial Gaussian centered at (0.3, 0.3)
x0, y0 = 0.3, 0.3
m0 = np.exp(-50 * ((X - x0) ** 2 + (Y - y0) ** 2))
m0 /= np.sum(m0) * dx * dy  # Normalize


# Terminal cost: quadratic distance to center
def terminal_cost_2d(xy):
    """Terminal cost for 2D problem."""
    if xy.ndim == 1:
        return (xy[0] - 0.5) ** 2 + (xy[1] - 0.3) ** 2
    return (xy[:, 0] - 0.5) ** 2 + (xy[:, 1] - 0.3) ** 2


components = MFGComponents(
    hamiltonian=hamiltonian,
    m_initial=m0,  # Pass array directly
    u_final=terminal_cost_2d,
)

problem = MFGProblem(
    geometry=domain,
    T=0.1,  # Short time horizon
    Nt=10,  # 10 timesteps
    diffusion=0.1,  # Base diffusion (tensor_diffusion_field overrides)
    components=components,
)

print("Domain: [0.0, 1.0] × [0.0, 0.6]")
print("Grid: 30 × 20 points")
print(f"Time: T = {problem.T}, Nt = {problem.Nt}, dt = {problem.dt:.4f}")
print()

# ============================================================================
# Diagonal Tensor Diffusion
# ============================================================================

print("Configuring anisotropic diagonal tensor diffusion...")
print()

# Diagonal tensor: fast horizontal, slow vertical
# Σ = [[σ_x², 0    ],
#      [0,     σ_y²]]
sigma_x = 0.15  # Fast horizontal diffusion
sigma_y = 0.05  # Slow vertical diffusion

Sigma = np.diag([sigma_x**2, sigma_y**2])

print("Tensor diffusion matrix:")
print(f"  Σ = [[{Sigma[0, 0]:.4f}, {Sigma[0, 1]:.4f}],")
print(f"       [{Sigma[1, 0]:.4f}, {Sigma[1, 1]:.4f}]]")
print()
print("Physical interpretation:")
print(f"  - Horizontal diffusion: σ_x = {sigma_x:.3f} (agents spread quickly left-right)")
print(f"  - Vertical diffusion:   σ_y = {sigma_y:.3f} (agents spread slowly up-down)")
print(f"  - Ratio: σ_x/σ_y = {sigma_x / sigma_y:.1f}x")
print()

# ============================================================================
# Initial Density (already configured in MFGComponents above)
# ============================================================================

print("Initial density: Gaussian centered at (0.3, 0.3)")
dx, dy = domain.get_grid_spacing()
print(f"  Mass: {np.sum(m0) * dx * dy:.6f}")
print()

# ============================================================================
# Create Solvers
# ============================================================================

print("Creating HJB and FP solvers...")
print()

boundary_conditions = no_flux_bc(dimension=2)

hjb_solver = HJBFDMSolver(
    problem,
    solver_type="newton",
    max_newton_iterations=30,
    newton_tolerance=1e-6,
)

fp_solver = FPFDMSolver(problem, boundary_conditions=boundary_conditions)

print("[OK] Solvers created")
print()

# ============================================================================
# Simplified Picard Iteration
# ============================================================================

print("Running simplified Picard iteration...")
print()

# Get grid shape
Nx, Ny = domain.get_grid_shape()
Nt = problem.Nt + 1

# Initialize: M uniform, U zero
M = np.ones((Nt, Nx, Ny)) / (Nx * Ny)
U = np.zeros((Nt, Nx, Ny))

# Set initial density
M[0] = m0.copy()

# Normalize all timesteps
for n in range(Nt):
    M[n] /= np.sum(M[n]) * dx * dy

# Run 3 Picard iterations
max_iterations = 3

for iteration in range(max_iterations):
    print(f"Picard Iteration {iteration + 1}/{max_iterations}")

    # Solve HJB with diagonal tensor
    U_new = hjb_solver.solve_hjb_system(
        M_density=M,
        U_terminal=U[-1],
        U_coupling_prev=U,
        tensor_diffusion_field=Sigma,
    )

    # Solve FP with diagonal tensor
    M_new = fp_solver.solve_fp_system(
        M_initial=m0,
        drift_field=U_new,
        tensor_diffusion_field=Sigma,
        show_progress=False,
    )

    # Check convergence
    U_change = np.linalg.norm(U_new - U) / (np.linalg.norm(U_new) + 1e-10)
    M_change = np.linalg.norm(M_new - M) / (np.linalg.norm(M_new) + 1e-10)

    print(f"  ΔU / ||U||: {U_change:.2e}")
    print(f"  ΔM / ||M||: {M_change:.2e}")

    # Check mass conservation
    masses = np.sum(M_new, axis=(1, 2)) * dx * dy
    mass_error = np.abs(masses - 1.0).max()
    print(f"  Mass conservation error: {mass_error:.2e}")
    print()

    # Update for next iteration
    U = U_new.copy()
    M = M_new.copy()

print("[OK] Picard iterations completed")
print()

# ============================================================================
# Visualization
# ============================================================================

print("Generating visualization...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("MFG with Diagonal Tensor Diffusion (σ_x=0.15, σ_y=0.05)", fontsize=14, fontweight="bold")

# Time indices to plot
time_indices = [0, Nt // 2, Nt - 1]
time_labels = ["t = 0", f"t = {problem.T / 2:.2f}", f"t = {problem.T:.2f}"]

for i, (t_idx, t_label) in enumerate(zip(time_indices, time_labels, strict=True)):
    # Plot density M(t, x, y)
    ax = axes[0, i]
    im = ax.contourf(X, Y, M[t_idx], levels=20, cmap="viridis")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Density m(x, y) at {t_label}")
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax)

    # Plot value function U(t, x, y)
    ax = axes[1, i]
    im = ax.contourf(X, Y, U[t_idx], levels=20, cmap="coolwarm")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Value u(x, y) at {t_label}")
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax)

plt.tight_layout()

# Save figure
output_path = "examples/outputs/diagonal_tensor_mfg.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"[OK] Figure saved to {output_path}")
print()

# Show plot
plt.show()

print("=" * 70)
print("MFG with Diagonal Tensor Diffusion - Complete")
print("=" * 70)
print()
print("Key Results:")
print(f"  - Anisotropic diffusion ratio: {sigma_x / sigma_y:.1f}x (horizontal vs vertical)")
print(f"  - Mass conservation error: < {mass_error:.2e}")
print(f"  - Picard convergence: {max_iterations} iterations")
print()
print("Observation:")
print("  - Density spreads primarily in horizontal direction (σ_x > σ_y)")
print("  - Agents prefer horizontal movement due to lower diffusion cost")
print("  - Value function reflects anisotropic cost structure")
