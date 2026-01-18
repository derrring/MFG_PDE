"""
2D Stefan Problem - Circular Ice Melting with Level Set Method.

Demonstrates:
- 2D level set evolution for free boundary problems
- Symmetry preservation during evolution
- Energy conservation checks
- Contour plot visualization of moving interface

Problem Setup:
    2D heat equation with circular ice region:
        ∂T/∂t = α·(∂²T/∂x² + ∂²T/∂y²)

    Stefan condition at interface:
        V_n = -k·[∂T/∂n]  (normal velocity from heat flux)

    Level set representation:
        Ice region: {(x,y) : φ(t,x,y) < 0}
        Interface: {(x,y) : φ(t,x,y) = 0}
        Evolution: ∂φ/∂t + V|∇φ| = 0

Mathematical Formulation:
    - Domain: (x,y) ∈ [0,1] × [0,1]
    - Initial ice: circle centered at (0.5, 0.5) with radius R₀ = 0.3
    - Boundary conditions: T = 1.0 on all boundaries (hot)
    - Initial condition: T = 0 inside ice, T = 1 outside
    - Diffusivity: α = 0.01
    - Thermal conductivity: k = 1.0

Expected Behavior:
    - Circle shrinks uniformly (if symmetry preserved)
    - Energy: ∫T dx dy + latent_heat·Area(ice) ≈ const
    - Interface remains approximately circular

Validation Metrics:
    1. Energy conservation: |E(t) - E(0)| / E(0) < 10%
    2. Symmetry: aspect_ratio = max_radius_x / max_radius_y < 1.1
    3. Qualitative: Interface shrinks monotonically

References:
    - Juric & Tryggvason (1996): Front-tracking method for dendritic solidification
    - Chen, Merriman, Osher, Smereka (1997): Simple level set method for incompressible flows

Created: 2026-01-18 (Issue #592 Phase 3.2.2)
Part of: Issue #592 - Level Set Methods for Free Boundary Problems
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mfg_pde.alg.numerical.pde_solvers import ImplicitHeatSolver
from mfg_pde.geometry import TensorProductGrid
from mfg_pde.geometry.boundary import dirichlet_bc
from mfg_pde.geometry.level_set import TimeDependentDomain

# Note: InterfaceJumpOperator not yet used in 2D (requires nD implementation)
# from mfg_pde.geometry.operators import InterfaceJumpOperator

# ========================================
# Problem Parameters
# ========================================
print("=" * 70)
print("2D Stefan Problem - Circular Ice Melting")
print("=" * 70)

# Domain and discretization
x_min, x_max = 0.0, 1.0
y_min, y_max = 0.0, 1.0
Nx, Ny = 60, 60  # Coarser grid for 2D (computational cost)
T_final = 0.5  # Shorter time (ice melts faster in 2D)

# Physics
alpha = 0.01  # Thermal diffusivity
kappa_thermal = 1.0  # Thermal conductivity
T_boundary = 1.0  # Hot boundary temperature
T_ice = 0.0  # Ice temperature

# Initial ice geometry
x_center, y_center = 0.5, 0.5
R0 = 0.3  # Initial radius

# Time stepping
# IMPLICIT SOLVER: Can use CFL >> 1 for 2D heat equation
dx_min = min((x_max - x_min) / Nx, (y_max - y_min) / Ny)
dt_explicit = 0.125 * dx_min**2 / alpha  # CFL = 0.125 (explicit stability, 2D)
cfl_factor = 10  # Use 10× larger timestep
dt = cfl_factor * dt_explicit
Nt = int(T_final / dt)

print("\nImplicit Solver Speedup (2D):")
print(f"  Explicit stable dt: {dt_explicit:.6f} (CFL = 0.125)")
print(f"  Implicit dt: {dt:.6f} (CFL = {cfl_factor * 0.125:.2f})")
print(f"  Speedup factor: {cfl_factor}× ({Nt} steps vs {int(T_final / dt_explicit)} steps)")

print("\nProblem Setup:")
print(f"  Domain: [{x_min},{x_max}] × [{y_min},{y_max}]")
print(f"  Time horizon: T = {T_final}")
print(f"  Initial ice: circle at ({x_center}, {y_center}), radius = {R0}")
print(f"  Grid: Nx = {Nx}, Ny = {Ny}")
print(f"  Time: Nt = {Nt}, dt = {dt:.6f}")
print(f"  Thermal diffusivity: α = {alpha}")
print(f"  Boundary temperature: T = {T_boundary}")

# ========================================
# Numerical Setup
# ========================================

# Create 2D grid
grid = TensorProductGrid(dimension=2, bounds=[(x_min, x_max), (y_min, y_max)], Nx=[Nx, Ny])
X, Y = grid.meshgrid()
dx, dy = grid.spacing

print("\nNumerical Grid:")
print(f"  Points: {grid.Nx_points}")
print(f"  Spacing: dx = {dx:.6f}, dy = {dy:.6f}")
print(f"  CFL (heat): α·dt/dx² = {alpha * dt / dx**2:.4f} (should be < 0.25)")

# Initial level set: φ = ||r - r_center|| - R0
phi0 = np.sqrt((X - x_center) ** 2 + (Y - y_center) ** 2) - R0

# Create time-dependent domain
ls_domain = TimeDependentDomain(phi0, grid, initial_time=0.0, is_signed_distance=True)

print("\nInitialized TimeDependentDomain:")
print(f"  {ls_domain}")

# ========================================
# Heat Equation Solver (Implicit 2D)
# ========================================

# Create Dirichlet boundary conditions for 2D heat equation
bc_heat = dirichlet_bc(dimension=2, value=0.0)  # Will override values manually

# Initialize implicit heat solver (Crank-Nicolson)
heat_solver = ImplicitHeatSolver(
    grid=grid,
    alpha=alpha,
    bc=bc_heat,
    theta=0.5,  # Crank-Nicolson (2nd-order accurate, unconditionally stable)
)

print("\nHeat Solver (2D):")
print(f"  {heat_solver}")
print(f"  CFL number: {heat_solver.get_cfl_number(dt):.2f} (no stability restriction)")


def compute_heat_flux_normal(T: np.ndarray, phi: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Compute normal heat flux at interface: q_n = -k·∂T/∂n.

    Returns velocity field V for level set evolution.
    """
    # Compute temperature gradients
    grad_T_x = np.zeros_like(T)
    grad_T_y = np.zeros_like(T)

    # Central differences
    grad_T_x[1:-1, :] = (T[2:, :] - T[:-2, :]) / (2 * dx)
    grad_T_y[:, 1:-1] = (T[:, 2:] - T[:, :-2]) / (2 * dy)

    # Compute level set normal: n = ∇φ / |∇φ|
    grad_phi_x = np.zeros_like(phi)
    grad_phi_y = np.zeros_like(phi)

    grad_phi_x[1:-1, :] = (phi[2:, :] - phi[:-2, :]) / (2 * dx)
    grad_phi_y[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2 * dy)

    grad_phi_mag = np.sqrt(grad_phi_x**2 + grad_phi_y**2) + 1e-10

    n_x = grad_phi_x / grad_phi_mag
    n_y = grad_phi_y / grad_phi_mag

    # Normal derivative: ∂T/∂n = ∇T · n
    dT_dn = grad_T_x * n_x + grad_T_y * n_y

    # Stefan condition: V = -κ·∂T/∂n
    # For simplicity, use uniform velocity near interface
    V = -kappa_thermal * dT_dn

    # Smoothing: only apply near interface
    interface_mask = np.abs(phi) < 3 * max(dx, dy)
    V_smoothed = np.where(interface_mask, V, 0.0)

    return V_smoothed


# ========================================
# Time Evolution Loop
# ========================================

print("\n" + "=" * 70)
print("Time Evolution")
print("=" * 70)

# Initialize temperature field
# Hot outside ice, cold inside ice
T = np.where(phi0 < 0, T_ice, T_boundary)

# Storage for snapshots
snapshot_interval = max(1, Nt // 5)  # ~5 snapshots
snapshot_times = []
snapshot_T = []
snapshot_phi = []


# Energy tracking
def compute_energy(T: np.ndarray, phi: np.ndarray, dx: float, dy: float) -> dict:
    """Compute total energy: thermal + latent."""
    # Thermal energy: ∫T dx dy
    E_thermal = np.sum(T) * dx * dy

    # Latent heat: proportional to ice area
    ice_area = np.sum(phi < 0) * dx * dy

    # Total (latent heat coefficient = 1 for simplicity)
    E_total = E_thermal + ice_area

    return {"thermal": E_thermal, "latent": ice_area, "total": E_total}


E0 = compute_energy(T, phi0, dx, dy)
energy_history = [E0]
time_history_energy = [0.0]

print("\nInitial Energy:")
print(f"  Thermal: {E0['thermal']:.6f}")
print(f"  Latent (ice area): {E0['latent']:.6f}")
print(f"  Total: {E0['total']:.6f}")

print(f"\nEvolving system for {Nt} timesteps...")

# Save initial snapshot
snapshot_times.append(0.0)
snapshot_T.append(T.copy())
snapshot_phi.append(phi0.copy())

for n in range(Nt):
    t = (n + 1) * dt

    # 1. Solve heat equation (implicit solver)
    phi_current = ls_domain.current_phi
    T = heat_solver.solve_step(T, dt)

    # Enforce Dirichlet boundary conditions
    T[0, :] = T_boundary  # Left
    T[-1, :] = T_boundary  # Right
    T[:, 0] = T_boundary  # Bottom
    T[:, -1] = T_boundary  # Top

    # 2. Compute interface velocity from heat flux
    V_field = compute_heat_flux_normal(T, phi_current, dx, dy)

    # 3. Evolve level set
    # Reinitialize every 10 steps to maintain SDF property
    reinit = n % 10 == 0
    ls_domain.evolve_step(V_field, dt, reinitialize=reinit, save_to_history=True)

    # Track energy
    if n % 10 == 0:
        E = compute_energy(T, ls_domain.current_phi, dx, dy)
        energy_history.append(E)
        time_history_energy.append(t)

    # Save snapshots
    if n % snapshot_interval == 0 or n == Nt - 1:
        snapshot_times.append(t)
        snapshot_T.append(T.copy())
        snapshot_phi.append(ls_domain.current_phi.copy())
        print(f"  t = {t:.4f}: snapshots = {len(snapshot_times)}")

print("\nEvolution complete!")
print(f"  Final time: t = {ls_domain.current_time:.4f}")
print(f"  Total snapshots: {len(snapshot_times)}")

# ========================================
# Validation
# ========================================

print("\n" + "=" * 70)
print("Validation")
print("=" * 70)

# 1. Energy conservation
E_final = energy_history[-1]
energy_change_rel = abs(E_final["total"] - E0["total"]) / E0["total"]

print("\nEnergy Conservation:")
print(f"  Initial total: {E0['total']:.6f}")
print(f"  Final total: {E_final['total']:.6f}")
print(f"  Relative change: {100 * energy_change_rel:.2f}%")

if energy_change_rel < 0.1:  # 10% tolerance
    print("  ✅ Energy reasonably conserved")
else:
    print(f"  ⚠️  Energy change = {100 * energy_change_rel:.1f}% > 10%")

# 2. Symmetry preservation
phi_final = ls_domain.current_phi
interface_final = np.abs(phi_final) < dx

# Find interface extent in x and y
x_interface = X[interface_final]
y_interface = Y[interface_final]

if len(x_interface) > 0:
    x_extent = x_interface.max() - x_interface.min()
    y_extent = y_interface.max() - y_interface.min()
    aspect_ratio = x_extent / (y_extent + 1e-10)

    print("\nSymmetry Check:")
    print(f"  Interface extent: x = {x_extent:.4f}, y = {y_extent:.4f}")
    print(f"  Aspect ratio: {aspect_ratio:.4f}")

    if abs(aspect_ratio - 1.0) < 0.1:
        print("  ✅ Symmetry well preserved (aspect ≈ 1)")
    else:
        print(f"  ⚠️  Asymmetry: aspect ratio = {aspect_ratio:.2f}")

# 3. Interface monotonicity (should shrink)
area_initial = np.sum(phi0 < 0) * dx * dy
area_final = np.sum(phi_final < 0) * dx * dy

print("\nInterface Evolution:")
print(f"  Initial ice area: {area_initial:.6f}")
print(f"  Final ice area: {area_final:.6f}")
print(f"  Change: {area_final - area_initial:.6f} (should be negative)")

if area_final < area_initial:
    print("  ✅ Ice shrinks as expected")
else:
    print("  ⚠️  Ice expanded (unexpected)")

# ========================================
# Visualization
# ========================================

print("\n" + "=" * 70)
print("Creating Visualizations")
print("=" * 70)

# Create output directory
output_dir = Path(__file__).parent.parent / "outputs" / "level_set_methods"
output_dir.mkdir(parents=True, exist_ok=True)

# Create figure with contour plots
n_snapshots = len(snapshot_times)
fig = plt.figure(figsize=(18, 12))

# Top row: Temperature contours
for i in range(min(3, n_snapshots)):
    ax = plt.subplot(3, 3, i + 1)
    T_snap = snapshot_T[i]
    phi_snap = snapshot_phi[i]
    t_snap = snapshot_times[i]

    # Temperature contour
    cs = ax.contourf(X, Y, T_snap, levels=20, cmap="hot")
    # Interface (φ = 0)
    ax.contour(X, Y, phi_snap, levels=[0], colors="cyan", linewidths=2)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Temperature @ t={t_snap:.3f}")
    ax.set_aspect("equal")
    plt.colorbar(cs, ax=ax)

# Middle row: Level set
for i in range(min(3, n_snapshots)):
    ax = plt.subplot(3, 3, 3 + i + 1)
    phi_snap = snapshot_phi[i]
    t_snap = snapshot_times[i]

    # Level set contour
    cs = ax.contourf(X, Y, phi_snap, levels=20, cmap="RdBu_r", vmin=-0.3, vmax=0.3)
    ax.contour(X, Y, phi_snap, levels=[0], colors="k", linewidths=2)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Level Set @ t={t_snap:.3f}")
    ax.set_aspect("equal")
    plt.colorbar(cs, ax=ax)

# Bottom row: Diagnostics
ax7 = plt.subplot(3, 3, 7)
# Energy evolution
E_thermal_arr = [E["thermal"] for E in energy_history]
E_latent_arr = [E["latent"] for E in energy_history]
E_total_arr = [E["total"] for E in energy_history]

ax7.plot(time_history_energy, E_thermal_arr, label="Thermal", linewidth=2)
ax7.plot(time_history_energy, E_latent_arr, label="Latent (ice area)", linewidth=2)
ax7.plot(time_history_energy, E_total_arr, "k--", label="Total", linewidth=2)
ax7.set_xlabel("Time t")
ax7.set_ylabel("Energy")
ax7.set_title("Energy Evolution")
ax7.legend()
ax7.grid(True, alpha=0.3)

ax8 = plt.subplot(3, 3, 8)
# Ice area vs time
ax8.plot(time_history_energy, E_latent_arr, "b-", linewidth=2)
ax8.set_xlabel("Time t")
ax8.set_ylabel("Ice Area")
ax8.set_title("Ice Area Evolution")
ax8.grid(True, alpha=0.3)

ax9 = plt.subplot(3, 3, 9)
# Interface overlay for all times
for i, (phi_snap, t_snap) in enumerate(zip(snapshot_phi, snapshot_times, strict=False)):
    ax9.contour(X, Y, phi_snap, levels=[0], colors=f"C{i}", linewidths=1.5, alpha=0.7)

ax9.set_xlabel("x")
ax9.set_ylabel("y")
ax9.set_title("Interface Evolution (all times)")
ax9.set_aspect("equal")
ax9.grid(True, alpha=0.3)

plt.tight_layout()

# Save figure
fig_path = output_dir / "stefan_problem_2d.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"\n✅ Saved figure: {fig_path}")

plt.show()

print("\n" + "=" * 70)
print("Stefan Problem 2D - Complete")
print("=" * 70)
