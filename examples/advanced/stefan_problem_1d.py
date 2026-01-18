"""
1D Stefan Problem - Ice Melting with Level Set Method.

Demonstrates:
- Level set evolution for free boundary problems
- Stefan condition: V = -[k·∂T/∂n] (velocity from heat flux jump)
- Coupling: Heat equation ↔ Moving boundary
- Analytical validation (Neumann solution)

Problem Setup:
    Classical Stefan problem for ice melting:
        ∂T/∂t = α·∂²T/∂x²    (heat equation in both regions)

    Stefan condition at interface s(t):
        V = ds/dt = -k·[∂T/∂x]  (interface velocity from heat flux)

    Level set representation:
        Domain: {x : φ(t,x) < 0}  (ice region)
        Interface: {x : φ(t,x) = 0}
        Evolution: ∂φ/∂t + V|∇φ| = 0

Mathematical Formulation:
    - Domain: x ∈ [0, 1]
    - Initial interface: s(0) = 0.5
    - Boundary conditions: T(0) = 1.0 (hot), T(1) = 0.0 (cold)
    - Diffusivity: α = 0.01
    - Thermal conductivity: k = 1.0

Analytical Solution (Neumann):
    Interface position: s(t) = s₀ + λ·√(4αt)
    where λ solves transcendental equation involving erf functions

Expected Behavior:
    - Interface moves left (ice melts from hot boundary)
    - Temperature field smooth on both sides
    - Heat flux discontinuity drives interface motion

References:
    - Crank (1984): Free and Moving Boundary Problems
    - Alexiades & Solomon (1993): Mathematical Modeling of Melting and Freezing
    - Osher & Fedkiw (2003): Level Set Methods, Chapter 8

Created: 2026-01-18 (Issue #592 Phase 3.2.1)
Part of: Issue #592 - Level Set Methods for Free Boundary Problems
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf

from mfg_pde.alg.numerical.pde_solvers import ImplicitHeatSolver
from mfg_pde.geometry import TensorProductGrid
from mfg_pde.geometry.boundary import dirichlet_bc
from mfg_pde.geometry.level_set import TimeDependentDomain

# ========================================
# Problem Parameters
# ========================================
print("=" * 70)
print("1D Stefan Problem - Ice Melting with Level Set Method")
print("=" * 70)

# Domain and discretization
x_min, x_max = 0.0, 1.0
Nx = 400  # Finer grid for better heat flux accuracy
T_final = 2.0  # Final time

# Physics
alpha = 0.01  # Thermal diffusivity (α = k/(ρc))
kappa_thermal = 1.0  # Thermal conductivity
T_hot = 1.0  # Hot boundary temperature
T_cold = 0.0  # Cold boundary temperature
s0 = 0.5  # Initial interface position

# Time stepping
# IMPLICIT SOLVER: Can use CFL >> 1 (unconditionally stable)
# With explicit solver: CFL < 0.5 required → dt_explicit ≈ 0.0003125
# With implicit solver: CFL = 10-50 achievable → 10-100× speedup
dx_val = (x_max - x_min) / Nx
dt_explicit = 0.2 * dx_val**2 / alpha  # CFL = 0.2 (explicit stability limit)
cfl_factor = 20  # Use 20× larger timestep (CFL = 4.0)
dt = cfl_factor * dt_explicit
Nt = int(T_final / dt)

print("\nImplicit Solver Speedup:")
print(f"  Explicit stable dt: {dt_explicit:.6f} (CFL = 0.2)")
print(f"  Implicit dt: {dt:.6f} (CFL = {cfl_factor * 0.2:.1f})")
print(f"  Speedup factor: {cfl_factor}× ({Nt} steps vs {int(T_final / dt_explicit)} steps)")

print("\nProblem Setup:")
print(f"  Domain: x ∈ [{x_min}, {x_max}]")
print(f"  Time horizon: T = {T_final}")
print(f"  Initial interface: s(0) = {s0}")
print(f"  Grid: Nx = {Nx}, Nt = {Nt}, dt = {dt:.4f}")
print(f"  Thermal diffusivity: α = {alpha}")
print(f"  Thermal conductivity: κ = {kappa_thermal}")
print(f"  Boundary temperatures: T(0) = {T_hot}, T(1) = {T_cold}")

# ========================================
# Analytical Solution (Neumann)
# ========================================


def neumann_transcendental(lam: float, T_hot: float = 1.0, T_cold: float = 0.0) -> float:
    """
    Transcendental equation for Neumann solution: f(λ) = 0.

    For one-phase Stefan problem:
        λ·exp(λ²)·erf(λ) = (T_hot - T_melt) / √π

    Simplified for T_melt = 0:
        λ·exp(λ²)·erf(λ) = T_hot / √π
    """
    if abs(lam) < 1e-10:
        return -T_hot / np.sqrt(np.pi)

    lhs = lam * np.exp(lam**2) * erf(lam)
    rhs = T_hot / np.sqrt(np.pi)
    return lhs - rhs


# Solve for λ using bisection
lam_min, lam_max = 0.01, 1.0
for _ in range(50):
    lam_mid = (lam_min + lam_max) / 2
    f_mid = neumann_transcendental(lam_mid, T_hot, T_cold)

    if abs(f_mid) < 1e-8:
        break

    if f_mid * neumann_transcendental(lam_min, T_hot, T_cold) < 0:
        lam_max = lam_mid
    else:
        lam_min = lam_mid

lambda_neumann = lam_mid

print("\nAnalytical Solution:")
print(f"  Neumann λ = {lambda_neumann:.6f} (from transcendental eq)")
print(f"  Interface velocity: V = λ·√(α/t) ≈ {lambda_neumann * np.sqrt(alpha):.6f}·t^(-1/2)")

# ========================================
# Numerical Setup
# ========================================

# Create grid
grid = TensorProductGrid(dimension=1, bounds=[(x_min, x_max)], Nx=[Nx])
x = grid.coordinates[0]
dx = grid.spacing[0]

print("\nNumerical Grid:")
print(f"  Points: {len(x)}")
print(f"  Spacing: dx = {dx:.6f}")

# Initial level set: φ = x - s0 (interface at x = s0)
phi0 = x - s0

# Create time-dependent domain
ls_domain = TimeDependentDomain(phi0, grid, initial_time=0.0, is_signed_distance=True)

print("\nInitialized TimeDependentDomain:")
print(f"  {ls_domain}")

# ========================================
# Heat Equation Solver (Implicit)
# ========================================

# Create Dirichlet boundary conditions for heat equation
# Left boundary: T = T_hot, Right boundary: T = T_cold
bc_heat = dirichlet_bc(dimension=1, value=0.0)  # Will override values manually

# Initialize implicit heat solver (Crank-Nicolson, theta=0.5)
heat_solver = ImplicitHeatSolver(
    grid=grid,
    alpha=alpha,
    bc=bc_heat,
    theta=0.5,  # Crank-Nicolson (2nd-order accurate, unconditionally stable)
)

print("\nHeat Solver:")
print(f"  {heat_solver}")
print(f"  CFL number: {heat_solver.get_cfl_number(dt):.2f} (no stability restriction)")


# ========================================
# Time Evolution Loop
# ========================================

print("\n" + "=" * 70)
print("Time Evolution")
print("=" * 70)

# Initialize temperature field
# CRITICAL: Need ASYMMETRIC gradients to create heat flux jump
# The Stefan problem requires different temperature gradients on each side
#
# Classical Stefan setup:
# - Water (left): Should have gradient ∂T/∂x = -T_hot/s0  (cooling towards interface)
# - Ice (right): Should have gradient ∂T/∂x = 0 (uniformly cold, will evolve)
#
# This creates heat flux jump: κ·[∂T/∂x] = κ·(0 - (-T_hot/s0)) = κ·T_hot/s0 > 0
# Stefan condition: V = -κ·[∂T/∂x] = -κ·T_hot/s0 < 0  (moves left)

T = np.zeros_like(x)
for i, xi in enumerate(x):
    if xi <= s0:
        # Water side: linear from T_hot at x=0 to T_melt=0 at interface
        T[i] = T_hot * (1 - xi / s0)
    else:
        # Ice side: uniform T_cold (zero gradient initially)
        T[i] = T_cold

# Enforce boundary conditions exactly
T[0] = T_hot
T[-1] = T_cold

# Storage for snapshots
snapshot_times = [0.0, 0.5, 1.0, 1.5, 2.0]
snapshot_T = []
snapshot_phi = []
snapshot_s = []

# Track interface position
interface_history = [s0]
time_history = [0.0]

# Evolve
print(f"\nEvolving system for {Nt} timesteps...")
print(f"  Saving snapshots at t = {snapshot_times}")

for n in range(Nt):
    t = n * dt

    # 1. Solve heat equation on current domain (implicit solver)
    phi_current = ls_domain.current_phi
    T = heat_solver.solve_step(T, dt)

    # Enforce Dirichlet boundary conditions (solver may not preserve exactly)
    T[0] = T_hot
    T[-1] = T_cold

    # 2. Compute interface velocity from heat flux jump
    # Stefan condition: V = -κ·[∂T/∂x] at interface
    # Find interface location
    idx_interface = np.argmin(np.abs(phi_current))

    # Compute heat flux using central differences averaged on both sides
    if idx_interface > 1 and idx_interface < len(x) - 2:
        # Average gradient on right side (ice, φ > 0)
        # Use 2 points right of interface
        grad_T_right = (T[idx_interface + 2] - T[idx_interface]) / (2 * dx)

        # Average gradient on left side (water, φ < 0)
        # Use 2 points left of interface
        grad_T_left = (T[idx_interface] - T[idx_interface - 2]) / (2 * dx)

        # Heat flux jump: [∂T/∂x] = grad_right - grad_left
        heat_flux_jump = grad_T_right - grad_T_left

        # Stefan condition: V = -κ·[∂T/∂x]
        # Negative of jump gives melting direction
        V_interface = -kappa_thermal * heat_flux_jump
    else:
        V_interface = 0.0

    # 3. Evolve level set with interface velocity
    # Constant velocity field (interface moves uniformly)
    V_field = V_interface * np.ones_like(phi_current)

    # Evolve (without reinitialization for now - φ stays close to SDF)
    ls_domain.evolve_step(V_field, dt, reinitialize=False, save_to_history=True)

    # Track interface position
    phi_new = ls_domain.current_phi
    idx_interface_new = np.argmin(np.abs(phi_new))
    s_current = x[idx_interface_new]

    interface_history.append(s_current)
    time_history.append((n + 1) * dt)

    # Save snapshots
    if any(abs(t - t_snap) < dt / 2 for t_snap in snapshot_times):
        snapshot_T.append(T.copy())
        snapshot_phi.append(phi_new.copy())
        snapshot_s.append(s_current)
        print(f"  t = {t:.4f}: s = {s_current:.4f}, V = {V_interface:.6f}")

# Final snapshot if not already saved
if len(snapshot_T) < len(snapshot_times):
    snapshot_T.append(T.copy())
    snapshot_phi.append(ls_domain.current_phi.copy())
    snapshot_s.append(interface_history[-1])

print("\nEvolution complete!")
print(f"  Final time: t = {time_history[-1]:.4f}")
print(f"  Final interface: s = {interface_history[-1]:.4f}")

# ========================================
# Validation
# ========================================

print("\n" + "=" * 70)
print("Validation Against Analytical Solution")
print("=" * 70)

# Analytical interface position
# Note: Classical Neumann solution has interface moving INTO solid (positive direction)
# Our setup: hot at x=0, cold at x=1, interface moves LEFT (melting towards hot side)
# Therefore: s(t) = s0 - λ·√(4αt)  (negative sign for leftward motion)
time_array = np.array(time_history)
s_analytical = s0 - lambda_neumann * np.sqrt(4 * alpha * time_array)
s_numerical = np.array(interface_history)

# Compute error
error_abs = np.abs(s_numerical - s_analytical)
error_rel = error_abs / (s0 + 1e-10)  # Relative to initial position

print("\nFinal Time Comparison:")
print(f"  Analytical: s = {s_analytical[-1]:.6f}")
print(f"  Numerical:  s = {s_numerical[-1]:.6f}")
print(f"  Absolute error: {error_abs[-1]:.6f}")
print(f"  Relative error: {100 * error_rel[-1]:.4f}%")

# Maximum error over all time
max_error_abs = np.max(error_abs)
max_error_rel = np.max(error_rel)
print("\nMaximum Error (all time):")
print(f"  Absolute: {max_error_abs:.6f}")
print(f"  Relative: {100 * max_error_rel:.4f}%")

# Check acceptance criterion
if max_error_rel < 0.05:  # 5% error threshold
    print("\n✅ Validation PASSED: Error < 5%")
else:
    print(f"\n⚠️  Validation WARNING: Error = {100 * max_error_rel:.2f}% > 5%")

# ========================================
# Visualization
# ========================================

print("\n" + "=" * 70)
print("Creating Visualizations")
print("=" * 70)

# Create output directory
output_dir = Path(__file__).parent.parent / "outputs" / "level_set_methods"
output_dir.mkdir(parents=True, exist_ok=True)

# 6-panel layout
fig = plt.figure(figsize=(18, 10))

# Row 0: Temperature evolution | Interface position
ax1 = plt.subplot(2, 3, 1)
for i, (T_snap, s_snap, t_snap) in enumerate(
    zip(snapshot_T, snapshot_s, snapshot_times[: len(snapshot_T)], strict=False)
):
    ax1.plot(x, T_snap, label=f"t={t_snap:.1f}", alpha=0.7)
    # Mark interface
    ax1.axvline(s_snap, color=f"C{i}", linestyle="--", alpha=0.5)

ax1.set_xlabel("x")
ax1.set_ylabel("Temperature T(x)")
ax1.set_title("Temperature Evolution")
ax1.legend(loc="upper right")
ax1.grid(True, alpha=0.3)

ax2 = plt.subplot(2, 3, 2)
ax2.plot(time_array, s_numerical, "b-", label="Numerical (Level Set)", linewidth=2)
ax2.plot(time_array, s_analytical, "r--", label="Analytical (Neumann)", linewidth=2)
ax2.set_xlabel("Time t")
ax2.set_ylabel("Interface position s(t)")
ax2.set_title("Interface Tracking")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Row 0, col 3: Error vs time
ax3 = plt.subplot(2, 3, 3)
ax3.semilogy(time_array, error_abs, "k-", linewidth=2)
ax3.set_xlabel("Time t")
ax3.set_ylabel("Absolute Error |s_num - s_ana|")
ax3.set_title(f"Interface Error (max = {max_error_abs:.4f})")
ax3.grid(True, alpha=0.3)

# Row 1: Level set evolution | Phase diagram | Velocity
ax4 = plt.subplot(2, 3, 4)
for i, (phi_snap, t_snap) in enumerate(zip(snapshot_phi, snapshot_times[: len(snapshot_phi)], strict=False)):
    ax4.plot(x, phi_snap, label=f"t={t_snap:.1f}", alpha=0.7)

ax4.axhline(0, color="k", linestyle=":", label="Interface (φ=0)")
ax4.set_xlabel("x")
ax4.set_ylabel("Level set φ(x)")
ax4.set_title("Level Set Evolution")
ax4.legend()
ax4.grid(True, alpha=0.3)

ax5 = plt.subplot(2, 3, 5)
# Compute interface velocity from position history
dt_array = np.diff(time_array)
ds_array = np.diff(s_numerical)
V_numerical = ds_array / dt_array
t_velocity = time_array[1:]  # Velocity defined between timesteps

# Analytical velocity: V = ds/dt = λ·√(α/t)
V_analytical = lambda_neumann * np.sqrt(alpha / (t_velocity + 1e-10))

ax5.plot(t_velocity, V_numerical, "b-", label="Numerical", alpha=0.7)
ax5.plot(t_velocity, V_analytical, "r--", label="Analytical", alpha=0.7)
ax5.set_xlabel("Time t")
ax5.set_ylabel("Interface velocity V(t)")
ax5.set_title("Interface Velocity")
ax5.legend()
ax5.grid(True, alpha=0.3)

ax6 = plt.subplot(2, 3, 6)
# Plot error percentage over time
ax6.plot(time_array, 100 * error_rel, "k-", linewidth=2)
ax6.axhline(5, color="r", linestyle="--", alpha=0.5, label="5% threshold")
ax6.set_xlabel("Time t")
ax6.set_ylabel("Relative Error (%)")
ax6.set_title(f"Relative Error (max = {100 * max_error_rel:.2f}%)")
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()

# Save figure
fig_path = output_dir / "stefan_problem_1d.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"\n✅ Saved figure: {fig_path}")

# Show plot
plt.show()

print("\n" + "=" * 70)
print("Stefan Problem 1D - Complete")
print("=" * 70)
