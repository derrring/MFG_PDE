"""
1D Obstacle Problem - Heat Equation with Obstacle (Proper Physics).

This demonstrates a realistic obstacle problem using the HEAT EQUATION
as the baseline, which has well-understood physics.

Physical Interpretation: Cooling Rod with Minimum Temperature
------------------------------------------------------------
Consider a heated rod cooling in air with a minimum temperature constraint:
- u(t,x) = temperature at time t, position x
- ψ(x) = minimum allowable temperature (thermostat)
- Constraint: u(t,x) ≥ ψ(x)

The obstacle represents a thermostat maintaining minimum temperature.

Mathematically:
    ∂u/∂t = σ²/2 · ∂²u/∂x² - λu    if u > ψ  (diffusion + cooling)
    u = ψ                            if u = ψ  (thermostat active)

Active Set: Points where temperature hits the minimum bound.

Setup:
    - Initial condition: u(0,x) = 1.0 (uniform hot temperature)
    - Obstacle: ψ(x) = parabolic (higher at center)
    - Physics: Rod cools exponentially (∂u/∂t = diffusion - λu)
    - Thermostat prevents cooling below ψ(x)

Expected Behavior:
    - Early times: Rod is hot everywhere, no constraint active
    - Later times: Rod cools, center hits minimum first (higher ψ)
    - Active set GROWS over time (physically meaningful!)

Created: 2026-01-17 (Issue #591 - Proper Obstacle with Correct Physics)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mfg_pde.geometry import TensorProductGrid
from mfg_pde.geometry.boundary import ObstacleConstraint, no_flux_bc

print("=" * 70)
print("Heat Equation with Minimum Temperature Constraint")
print("=" * 70)

# ========================================
# Parameters
# ========================================
x_min, x_max = 0.0, 1.0
Nx = 50  # Reduced for faster computation
T = 3.5  # Extended to see active set appear
Nt = 220  # Increased proportionally
sigma = 0.15  # Diffusion coefficient
cooling_rate = 0.5  # λ: cooling rate (heat loss to environment)

# Create grid

grid = TensorProductGrid(bounds=[(x_min, x_max)], Nx=[Nx], boundary_conditions=no_flux_bc(dimension=1))
x = grid.coordinates[0]
Nx_actual = len(x)
dx = x[1] - x[0]
dt = T / Nt

print("\nPhysical Setup:")
print("  Heat diffusion in 1D rod")
print("  Minimum temperature constraint ψ(x)")
print(f"  Domain: x ∈ [{x_min}, {x_max}]")
print(f"  Time: t ∈ [0, {T}]")
print(f"  Diffusion: σ² = {sigma**2:.4f}")

# ========================================
# Initial Condition and Obstacle
# ========================================
# Initial temperature: Uniform high temperature (hot rod)
u_initial = np.ones_like(x) * 1.0

# Obstacle: Parabolic minimum temperature (thermostat)
# Higher minimum at center (must stay warm), lower at edges
psi_center = 0.3  # Minimum at center
psi_edge = 0.1  # Lower minimum at edges
psi = psi_center + (psi_edge - psi_center) * (2 * (x - 0.5)) ** 2

print("\nInitial Condition:")
print("  u(0,x) = uniform hot temperature")
print(f"    u(0,x) = {u_initial[0]:.4f} everywhere")

print("\nObstacle (Minimum Temperature Thermostat):")
print("  ψ(x) = parabolic")
print(f"    ψ(0.5) = {psi[Nx_actual // 2]:.4f} (center - must stay warm)")
print(f"    ψ(0, 1) = {psi[0]:.4f} (edges - can cool more)")

print("\nCooling dynamics:")
print(f"  ∂u/∂t = {sigma**2 / 2:.4f}·∂²u/∂x² - {cooling_rate:.2f}·u")
print("  Rod cools exponentially while diffusing")

# Create obstacle constraint
constraint = ObstacleConstraint(psi, constraint_type="lower")

# Check if initial condition satisfies obstacle
if np.all(u_initial >= psi - 1e-10):
    print("\n✓ Initial condition satisfies u ≥ ψ everywhere")
else:
    violations_initial = (psi - u_initial).clip(min=0)
    num_viol = (violations_initial > 1e-10).sum()
    print(f"\n⚠ Initial condition violates obstacle at {num_viol} points")
    print(f"  Max violation: {violations_initial.max():.4f}")
    print("  Projecting initial condition onto feasible set...")
    u_initial = constraint.project(u_initial)
    print(f"  ✓ After projection: u(0,x) ∈ [{u_initial.min():.4f}, {u_initial.max():.4f}]")

# ========================================
# Solve Heat Equation WITHOUT Constraint
# ========================================
print("\n" + "=" * 70)
print("Solving Heat Equation WITHOUT constraint...")
print("=" * 70)

# For heat equation, set Hamiltonian to zero (no advection term)
# Just pure diffusion: ∂u/∂t = σ²/2 ∂²u/∂x²
# We'll solve using forward Euler for simplicity

U_unconstr = np.zeros((Nt + 1, Nx_actual))
U_unconstr[0] = u_initial.copy()

# Forward Euler timestepping
diffusion_coef = sigma**2 / 2
stability_limit = dx**2 / (2 * diffusion_coef)

print("\nNumerical stability:")
print(f"  dt = {dt:.6f}")
print(f"  Stability limit = {stability_limit:.6f}")
print(f"  Ratio dt/limit = {dt / stability_limit:.3f}")

if dt > stability_limit:
    raise ValueError(f"Unstable! dt={dt:.6f} exceeds stability limit {stability_limit:.6f}")
elif dt > 0.8 * stability_limit:
    print("  ⚠ Warning: dt close to stability limit")
else:
    print("  ✓ Stable (dt < 0.8 × stability limit)")

# Simple central differences for Laplacian
for n in range(Nt):
    u_curr = U_unconstr[n]
    # Laplacian with Neumann BC (du/dx = 0 at boundaries)
    laplacian = np.zeros(Nx_actual)
    laplacian[1:-1] = (u_curr[2:] - 2 * u_curr[1:-1] + u_curr[:-2]) / dx**2
    # Neumann BC: ghost points equal interior
    laplacian[0] = (u_curr[1] - u_curr[0]) / dx**2  # One-sided
    laplacian[-1] = (u_curr[-2] - u_curr[-1]) / dx**2

    # Heat equation with cooling: ∂u/∂t = σ²/2 ∆u - λu
    U_unconstr[n + 1] = u_curr + dt * (diffusion_coef * laplacian - cooling_rate * u_curr)

print("Solution evolution (unconstrained):")
for n in [0, Nt // 4, Nt // 2, 3 * Nt // 4, Nt]:
    t = n * dt
    u_min, u_max = U_unconstr[n].min(), U_unconstr[n].max()
    print(f"  t={t:.3f}: u ∈ [{u_min:.4f}, {u_max:.4f}]")

# Check violations
violations_unconstr = np.maximum(0, psi - U_unconstr)
max_viol = violations_unconstr.max()
num_viol_points = (violations_unconstr > 1e-8).sum()

print("\nConstraint check:")
print(f"  Max violation: {max_viol:.4f}")
print(f"  Violating points: {num_viol_points} / {(Nt + 1) * Nx_actual}")

if max_viol > 1e-6:
    print("  → Heat diffusion causes temperature to drop below minimum")
else:
    print("  → Temperature stays above minimum naturally")

# ========================================
# Solve WITH Constraint (Project at Each Step)
# ========================================
print("\n" + "=" * 70)
print("Solving Heat Equation WITH constraint...")
print("=" * 70)

U_constr = np.zeros((Nt + 1, Nx_actual))
U_constr[0] = u_initial.copy()

for n in range(Nt):
    u_curr = U_constr[n]

    # Same heat equation step with cooling
    laplacian = np.zeros(Nx_actual)
    laplacian[1:-1] = (u_curr[2:] - 2 * u_curr[1:-1] + u_curr[:-2]) / dx**2
    laplacian[0] = (u_curr[1] - u_curr[0]) / dx**2
    laplacian[-1] = (u_curr[-2] - u_curr[-1]) / dx**2

    # Heat equation with cooling: ∂u/∂t = σ²/2 ∆u - λu
    u_next = u_curr + dt * (diffusion_coef * laplacian - cooling_rate * u_curr)

    # Apply obstacle constraint (thermostat activates)
    U_constr[n + 1] = constraint.project(u_next)

print("Solution evolution (constrained):")
for n in [0, Nt // 4, Nt // 2, 3 * Nt // 4, Nt]:
    t = n * dt
    u_min, u_max = U_constr[n].min(), U_constr[n].max()
    active = constraint.get_active_set(U_constr[n], tol=1e-6)
    print(f"  t={t:.3f}: u ∈ [{u_min:.4f}, {u_max:.4f}], {active.sum()}/{Nx_actual} active")

# Verify no violations
violations_constr = np.maximum(0, psi - U_constr)
print("\nConstraint satisfaction:")
print(f"  Max violation: {violations_constr.max():.2e}")
print(f"  Status: {'✓ SATISFIED' if violations_constr.max() < 1e-8 else '✗ VIOLATED'}")

# ========================================
# Active Set Analysis
# ========================================
print("\n" + "=" * 70)
print("Active Set Evolution")
print("=" * 70)

active_sets = []
for n in range(Nt + 1):
    active = constraint.get_active_set(U_constr[n], tol=1e-6)
    active_sets.append(active)
active_sets = np.array(active_sets)

print("\nActive set growth:")
for n in [0, Nt // 4, Nt // 2, 3 * Nt // 4, Nt]:
    t = n * dt
    frac = active_sets[n].sum() / Nx_actual
    print(f"  t={t:.3f}: {frac * 100:5.1f}% active")

print("\nPhysical interpretation:")
print("  As rod cools exponentially:")
print("  - Temperature drops everywhere (cooling to environment)")
print("  - Center hits minimum first (higher thermostat setting)")
print("  - More points hit the minimum temperature ψ(x)")
print("  - Active set GROWS (physically correct!)")

# ========================================
# Visualization
# ========================================
print("\n" + "=" * 70)
print("Creating visualizations...")
print("=" * 70)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

t_grid = np.linspace(0, T, Nt + 1)
X_mesh, T_mesh = np.meshgrid(x, t_grid)

# Plot 1: Unconstrained evolution
ax = axes[0, 0]
im = ax.pcolormesh(X_mesh, T_mesh, U_unconstr, shading="auto", cmap="hot")
plt.colorbar(im, ax=ax, label="Temperature u(t,x)")
ax.set_xlabel("Position x")
ax.set_ylabel("Time t")
ax.set_title("Unconstrained Heat Diffusion")

# Plot 2: Constrained evolution
ax = axes[0, 1]
im = ax.pcolormesh(X_mesh, T_mesh, U_constr, shading="auto", cmap="hot")
plt.colorbar(im, ax=ax, label="Temperature u(t,x)")
ax.set_xlabel("Position x")
ax.set_ylabel("Time t")
ax.set_title("Constrained Heat Diffusion")

# Plot 3: Difference
ax = axes[0, 2]
diff = U_constr - U_unconstr
im = ax.pcolormesh(X_mesh, T_mesh, diff, shading="auto", cmap="Greens")
plt.colorbar(im, ax=ax, label="Δu (constraint effect)")
ax.set_xlabel("Position x")
ax.set_ylabel("Time t")
ax.set_title("Temperature Boost from Constraint")

# Plot 4: Temperature snapshots
ax = axes[1, 0]
times = [0, Nt // 4, Nt // 2, 3 * Nt // 4, Nt]
colors = plt.cm.coolwarm(np.linspace(0, 1, len(times)))
for i, n in enumerate(times):
    t = n * dt
    ax.plot(x, U_unconstr[n], "--", color=colors[i], alpha=0.5)
    ax.plot(x, U_constr[n], "-", color=colors[i], linewidth=2, label=f"t={t:.2f}")
ax.plot(x, psi, "k:", linewidth=3, label="ψ(x) (minimum)")
ax.set_xlabel("Position x")
ax.set_ylabel("Temperature u(t,x)")
ax.set_title("Temperature Evolution")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Plot 5: Active set heatmap
ax = axes[1, 1]
im = ax.pcolormesh(X_mesh, T_mesh, active_sets.astype(float), shading="auto", cmap="RdYlGn_r", vmin=0, vmax=1)
plt.colorbar(im, ax=ax, label="At Minimum (1) / Free (0)")
ax.set_xlabel("Position x")
ax.set_ylabel("Time t")
ax.set_title("Active Set (At Minimum Temperature)")

# Plot 6: Active set growth
ax = axes[1, 2]
active_frac = [active_sets[n].sum() / Nx_actual for n in range(Nt + 1)]
ax.plot(t_grid, np.array(active_frac) * 100, "b-", linewidth=2)
ax.set_xlabel("Time t")
ax.set_ylabel("Active Set (%)")
ax.set_title("Fraction at Minimum Temperature")
ax.grid(True, alpha=0.3)

plt.tight_layout()

# Save
output_dir = Path(__file__).parent / "outputs" / "obstacle_problem_1d_heat"
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "heat_with_minimum_temperature.png"
plt.savefig(output_file, dpi=150, bbox_inches="tight")
print(f"✓ Figure saved to {output_file}")
plt.close()

# ========================================
# Summary
# ========================================
print("\n" + "=" * 70)
print("SUMMARY - Cooling Rod with Thermostat")
print("=" * 70)
print("Physical problem: Exponential cooling with minimum temperature constraint")
print("Initial condition: Uniform hot temperature (u=1.0)")
print("Obstacle: Parabolic thermostat (ψ=0.3 at center, 0.1 at edges)")
print("\nResults:")
print(f"  Unconstrained violations: {max_viol:.4f}")
print(f"  Constrained violations: {violations_constr.max():.2e}")
print(f"  Active set: {active_frac[0] * 100:.1f}% → {active_frac[-1] * 100:.1f}%")
print("\n✓ Physically meaningful obstacle problem!")
print("  - Active set GROWS as heat diffuses")
print("  - Constraint prevents temperature drop below minimum")
print("  - Well-posed PDE with correct physics")
print("=" * 70)
