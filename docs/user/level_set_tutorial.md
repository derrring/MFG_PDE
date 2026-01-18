# Tutorial: Level Set Methods for Free Boundary Problems

**Tutorial Level**: Advanced
**Prerequisites**: PDEs, basic numerical methods, MFG_PDE geometry module
**Estimated Time**: 60-90 minutes
**Version**: 1.0 (2026-01-18)

---

## Table of Contents

1. [Introduction](#introduction)
2. [Level Set Basics](#level-set-basics)
3. [Tutorial 1: 1D Stefan Problem (Ice Melting)](#tutorial-1-1d-stefan-problem-ice-melting)
4. [Tutorial 2: 2D Stefan Problem (Circular Interface)](#tutorial-2-2d-stefan-problem-circular-interface)
5. [Understanding the Results](#understanding-the-results)
6. [Numerical Stability](#numerical-stability)
7. [Common Pitfalls](#common-pitfalls)
8. [Advanced Topics](#advanced-topics)

---

## Introduction

This tutorial teaches you how to solve **free boundary problems** using the **Level Set method**. These problems feature interfaces whose position is **not known a priori** and must be determined as part of the solution.

### What You'll Learn

- Represent moving interfaces using level set functions
- Implement the Level Set evolution equation
- Couple Level Set method with PDEs (Stefan problem)
- Handle CFL stability conditions
- Validate against analytical solutions

### Real-World Applications

- **Phase transitions**: Ice melting, solidification
- **Mean Field Games**: Expanding exit boundaries, crowd-driven domain changes
- **Finance**: American options (free boundary = optimal exercise boundary)
- **Fluid mechanics**: Droplet coalescence, bubble dynamics

---

## Level Set Basics

### Implicit Interface Representation

Instead of tracking interface points explicitly (Lagrangian), we represent the interface **implicitly** using a scalar function $\phi(t, x)$:

```
Interface Γ(t) = {x : φ(t, x) = 0}    (zero level set)
Inside region = {x : φ(t, x) < 0}
Outside region = {x : φ(t, x) > 0}
```

**Example** (1D):
- Interface at $x = s(t)$ → Level set: $\phi(t, x) = x - s(t)$
- If $s(t) = 0.5 + 0.1t$ (moving right), then $\phi(t, x) = x - (0.5 + 0.1t)$

**★ Insight ─────────────────────────────────────**
The sign of $\phi$ determines inside/outside, but the **magnitude** gives distance to interface (if $\phi$ is a signed distance function). This makes geometry computations simple: just evaluate $\phi$ at a point!
**─────────────────────────────────────────────────**

### Hamilton-Jacobi Evolution Equation

If the interface moves with **normal velocity** $V_n(t, x)$, the level set evolves via:

```
∂φ/∂t + V_n|∇φ| = 0
```

**Derivation** (1-minute intuition):
1. Since $\phi(t, x(t)) = 0$ for all $t$ on the interface, differentiate:
   ```
   d/dt[φ(t, x(t))] = ∂φ/∂t + ∇φ · dx/dt = 0
   ```
2. Interface moves with normal velocity: $dx/dt = V_n \cdot n$
3. Normal vector: $n = \nabla\phi / |\nabla\phi|$
4. Substitute: $∂φ/∂t + ∇φ \cdot V_n \cdot (\nabla\phi / |\nabla\phi|) = 0$
5. Simplify: $∂φ/∂t + V_n|\nabla\phi| = 0$ ✓

**Key Property**: This equation holds **everywhere**, not just on the interface. This is why level sets are powerful—you evolve a function on the entire domain, and the zero level set automatically tracks the interface.

### Signed Distance Function (SDF)

A **signed distance function** satisfies:

```
|∇φ| = 1    everywhere
```

**Benefits**:
- $|\phi(x)|$ = actual distance from $x$ to interface
- Normal vector: $n = \nabla\phi$ (no normalization needed)
- Numerical stability (smooth gradients)

**Reinitialization**: During evolution, $|\nabla\phi| = 1$ may degrade. Restore it by solving:

```
∂φ/∂τ + sign(φ₀)(|∇φ| - 1) = 0    (pseudo-time τ)
```

until $\max ||\nabla\phi| - 1| < \epsilon$ (typically 20-50 iterations).

---

## Tutorial 1: 1D Stefan Problem (Ice Melting)

### Problem Statement

We model **ice melting** due to a hot boundary:

```
Heat equation:  ∂T/∂t = α·∂²T/∂x²    (in both ice and water regions)

Stefan condition (interface s(t)):
    Interface velocity: V = -κ·[∂T/∂x]    (heat flux jump drives motion)

Level set representation:
    Interface: φ(t, x) = 0
    Evolution: ∂φ/∂t + V|∂φ/∂x| = 0
```

**Physical Setup**:
- Domain: $x \in [0, 1]$
- Initial interface: $s(0) = 0.5$ (ice on left, water on right)
- Boundaries: $T(0) = 1.0$ (hot), $T(1) = 0.0$ (cold)
- Ice melts and interface moves **left** (toward hot boundary)

**Analytical Solution** (Neumann, 1860):
```
s(t) = s₀ + λ·√(4αt)

where λ solves transcendental equation:
    λ·exp(λ²)·erf(λ) = T_hot / √π
```

For $T_{\text{hot}} = 1, \alpha = 0.01$, the solution is $\lambda \approx 0.62$.

### Step 1: Set Up Grid and Initial Level Set

```python
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf

from mfg_pde.geometry import TensorProductGrid
from mfg_pde.geometry.level_set import TimeDependentDomain

# Parameters
x_min, x_max = 0.0, 1.0
Nx = 400  # Fine grid for accurate gradients
T_final = 2.0
alpha = 0.01  # Thermal diffusivity
kappa_thermal = 1.0  # Thermal conductivity
T_hot, T_cold = 1.0, 0.0
s0 = 0.5  # Initial interface position

# Create grid
grid = TensorProductGrid(dimension=1, bounds=[(x_min, x_max)], Nx=[Nx])
x = grid.coordinates[0]
dx = grid.spacing[0]

# CFL condition for heat equation: α·dt/dx² < 0.5
dt = 0.2 * dx**2 / alpha  # CFL = 0.2 (conservative)
Nt = int(T_final / dt)

print(f"Grid: Nx = {Nx}, dx = {dx:.6f}")
print(f"Time: Nt = {Nt}, dt = {dt:.6f}")
print(f"CFL = α·dt/dx² = {alpha * dt / dx**2:.3f}")
```

**★ Insight ─────────────────────────────────────**
The CFL condition for the **heat equation** is $\alpha \Delta t / \Delta x^2 < 0.5$ (parabolic). This is **stricter** than the Level Set CFL ($V \Delta t / \Delta x < 1$, hyperbolic). Always check both and use the smaller $\Delta t$.
**─────────────────────────────────────────────────**

```python
# Initial level set: φ = x - s0
# φ < 0 for x < s0 (ice region)
# φ > 0 for x > s0 (water region)
phi0 = x - s0

# Create time-dependent domain
ls_domain = TimeDependentDomain(
    phi0,
    grid,
    initial_time=0.0,
    is_signed_distance=True  # φ = x - s0 has |∇φ| = 1
)

print(f"\nInitial interface: s(0) = {s0}")
print(f"Initial φ range: [{phi0.min():.3f}, {phi0.max():.3f}]")
print(f"Zero crossing at: x = {x[np.argmin(np.abs(phi0))]:.3f}")
```

### Step 2: Implement Heat Equation Solver

```python
def solve_heat_equation_1d(
    T_prev: np.ndarray,
    dx: float,
    dt: float,
    alpha: float,
    T_hot: float = 1.0,
    T_cold: float = 0.0,
) -> np.ndarray:
    """
    Solve heat equation ∂T/∂t = α·∂²T/∂x² with Dirichlet BC.

    Uses explicit finite difference (forward Euler).
    """
    # CFL check
    cfl = alpha * dt / dx**2
    if cfl > 0.5:
        raise ValueError(f"CFL = {cfl:.3f} > 0.5, unstable!")

    T_new = T_prev.copy()

    # Interior points: explicit FD
    # T^{n+1}_i = T^n_i + α·dt/dx²·(T^n_{i+1} - 2T^n_i + T^n_{i-1})
    for i in range(1, len(T_new) - 1):
        laplacian = (T_prev[i+1] - 2*T_prev[i] + T_prev[i-1]) / dx**2
        T_new[i] = T_prev[i] + alpha * dt * laplacian

    # Boundary conditions (Dirichlet)
    T_new[0] = T_hot
    T_new[-1] = T_cold

    return T_new
```

### Step 3: Initialize Temperature Field

```python
# Initial temperature: piecewise linear profile
# Ice region (x < s0): linear from T_hot at x=0 to T_melt=0 at x=s0
# Water region (x > s0): T = T_cold = 0
T = np.zeros(Nx)

for i, xi in enumerate(x):
    if xi < s0:
        # Ice region: linear from T_hot to 0
        T[i] = T_hot * (s0 - xi) / s0
    else:
        # Water region: constant T_cold
        T[i] = T_cold

print(f"\nInitial temperature range: [{T.min():.3f}, {T.max():.3f}]")
```

**Why not constant initial temperature?**
- Constant IC → discontinuity at interface → poor heat flux estimate
- Piecewise linear IC → smooth in each region → better Stefan condition

### Step 4: Time Evolution Loop

```python
# Storage for history
time_history = [0.0]
interface_positions = [s0]
phi_history = [phi0.copy()]
T_history = [T.copy()]

# Time stepping
t = 0.0
for n in range(Nt):
    t += dt

    # 1. Solve heat equation on current domain
    T = solve_heat_equation_1d(T, dx, dt, alpha, T_hot, T_cold)

    # 2. Compute interface velocity from Stefan condition
    # V = -κ·[∂T/∂x] (heat flux jump at interface)

    # Find interface location
    phi_current = ls_domain.get_level_set_at_time(t - dt)
    idx_interface = np.argmin(np.abs(phi_current))
    x_interface = x[idx_interface]

    # Compute heat flux using centered difference near interface
    # ∂T/∂x ≈ (T[i+1] - T[i-1]) / (2dx)
    if 1 <= idx_interface < Nx - 1:
        grad_T_interface = (T[idx_interface + 1] - T[idx_interface - 1]) / (2 * dx)
    else:
        grad_T_interface = 0.0  # Fallback (shouldn't happen)

    # Stefan condition: V = -κ·∂T/∂x
    velocity = -kappa_thermal * grad_T_interface

    # 3. Evolve level set
    ls_domain.evolve_step(velocity, dt)

    # 4. Store history (every 10 steps)
    if n % 10 == 0:
        time_history.append(t)
        phi_current = ls_domain.get_level_set_at_time(t)
        phi_history.append(phi_current.copy())
        T_history.append(T.copy())

        # Find interface position (zero crossing)
        idx_interface = np.argmin(np.abs(phi_current))
        interface_positions.append(x[idx_interface])

        if n % 100 == 0:
            print(f"t = {t:.3f}, s(t) = {x[idx_interface]:.4f}, V = {velocity:.4f}")

print(f"\nSimulation complete: t = {t:.3f}")
```

**★ Insight ─────────────────────────────────────**
The **Stefan condition** $V = -\kappa \cdot [\partial T/\partial x]$ is the key coupling. Heat flows from hot to cold, creating a flux. At the phase boundary, this flux melts ice, moving the interface. The sign ensures: positive heat flux (flowing into ice) → ice melts → interface moves in that direction.
**─────────────────────────────────────────────────**

### Step 5: Validate Against Analytical Solution

```python
# Compute analytical solution (Neumann)
def neumann_transcendental(lam):
    """Transcendental equation: λ·exp(λ²)·erf(λ) = T_hot / √π"""
    if abs(lam) < 1e-10:
        return -T_hot / np.sqrt(np.pi)
    return lam * np.exp(lam**2) * erf(lam) - T_hot / np.sqrt(np.pi)

# Solve for λ using bisection
lam_min, lam_max = 0.01, 1.0
for _ in range(50):
    lam_mid = (lam_min + lam_max) / 2
    f_mid = neumann_transcendental(lam_mid)
    if abs(f_mid) < 1e-8:
        break
    if f_mid * neumann_transcendental(lam_min) < 0:
        lam_max = lam_mid
    else:
        lam_min = lam_mid

lambda_neumann = lam_mid

# Analytical interface positions
s_analytical = np.array([s0 + lambda_neumann * np.sqrt(4 * alpha * t) for t in time_history])

# Compare with numerical
s_numerical = np.array(interface_positions)
error = np.abs(s_numerical - s_analytical)
relative_error = error / s_analytical

print("\nValidation:")
print(f"  Neumann λ = {lambda_neumann:.6f}")
print(f"  Final interface position:")
print(f"    Analytical: s({T_final}) = {s_analytical[-1]:.4f}")
print(f"    Numerical:  s({T_final}) = {s_numerical[-1]:.4f}")
print(f"    Error: {error[-1]:.4f} ({100*relative_error[-1]:.2f}%)")
print(f"  Max relative error over time: {100*relative_error.max():.2f}%")
```

**Expected Results**:
- Error < 5% for $Nx = 400, \text{CFL} = 0.2$
- Error decreases with finer grids (first-order convergence)
- Early time error may be higher (initial profile approximation)

### Step 6: Visualize Results

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("1D Stefan Problem - Level Set Method", fontsize=14, fontweight="bold")

# (a) Interface position over time
axes[0, 0].plot(time_history, s_numerical, 'b-', linewidth=2, label="Numerical (Level Set)")
axes[0, 0].plot(time_history, s_analytical, 'r--', linewidth=2, label="Analytical (Neumann)")
axes[0, 0].set_xlabel("Time t")
axes[0, 0].set_ylabel("Interface position s(t)")
axes[0, 0].set_title("(a) Interface Evolution")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# (b) Error over time
axes[0, 1].plot(time_history, 100 * relative_error, 'k-', linewidth=2)
axes[0, 1].set_xlabel("Time t")
axes[0, 1].set_ylabel("Relative error (%)")
axes[0, 1].set_title("(b) Validation Error")
axes[0, 1].grid(True, alpha=0.3)

# (c) Level set evolution
time_indices = [0, len(phi_history)//4, len(phi_history)//2, 3*len(phi_history)//4, -1]
for idx in time_indices:
    t_val = time_history[idx]
    phi_val = phi_history[idx]
    axes[1, 0].plot(x, phi_val, label=f"t={t_val:.2f}")

axes[1, 0].axhline(0, color='k', linestyle='--', linewidth=1, label="Interface (φ=0)")
axes[1, 0].set_xlabel("x")
axes[1, 0].set_ylabel("φ(t, x)")
axes[1, 0].set_title("(c) Level Set Function Evolution")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# (d) Temperature evolution
for idx in time_indices:
    t_val = time_history[idx]
    T_val = T_history[idx]
    axes[1, 1].plot(x, T_val, label=f"t={t_val:.2f}")

# Mark interface positions
for idx in time_indices:
    s_val = interface_positions[idx]
    axes[1, 1].axvline(s_val, color='gray', linestyle=':', alpha=0.5)

axes[1, 1].set_xlabel("x")
axes[1, 1].set_ylabel("Temperature T(t, x)")
axes[1, 1].set_title("(d) Temperature Field Evolution")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**Expected Observations**:
1. **Panel (a)**: Interface moves left with $\sqrt{t}$ behavior (matches analytical)
2. **Panel (b)**: Error typically < 5%, may increase slightly at late times
3. **Panel (c)**: Level set shifts left, zero crossing tracks interface
4. **Panel (d)**: Temperature smooth in each region, discontinuity at interface

### Key Takeaways from Tutorial 1

✅ **Level Set evolution**: $\partial\phi/\partial t + V|\nabla\phi| = 0$ tracks interface implicitly
✅ **Stefan coupling**: Interface velocity from PDE (heat flux jump)
✅ **CFL stability**: Must satisfy **both** heat equation and level set CFL
✅ **Validation**: Neumann analytical solution confirms accuracy

---

## Tutorial 2: 2D Stefan Problem (Circular Interface)

### Problem Statement

We extend to **2D** with a **circular interface** (symmetric melting/freezing):

```
Heat equation: ∂T/∂t = α·(∂²T/∂x² + ∂²T/∂y²)

Stefan condition: V_n = -κ·∇T·n    (normal velocity from heat flux)

Level set: ∂φ/∂t + V_n|∇φ| = 0
```

**Setup**:
- Domain: $(x, y) \in [0, 1] \times [0, 1]$
- Initial interface: Circle at center, radius $R_0 = 0.2$
- Boundary: $T = T_{\text{hot}}$ everywhere (ice melts inward)
- Expected: Circle shrinks uniformly

### Step 1: Set Up 2D Grid and Circular Interface

```python
from mfg_pde.geometry import TensorProductGrid
from mfg_pde.geometry.level_set import TimeDependentDomain

# 2D Parameters
x_min, x_max = 0.0, 1.0
y_min, y_max = 0.0, 1.0
Nx, Ny = 100, 100
T_final = 0.5
alpha = 0.01
T_hot = 1.0
T_cold = 0.0

# Create 2D grid
grid = TensorProductGrid(
    dimension=2,
    bounds=[(x_min, x_max), (y_min, y_max)],
    Nx=[Nx, Ny]
)

x = grid.coordinates[0]
y = grid.coordinates[1]
dx, dy = grid.spacing

# Create meshgrid
X, Y = np.meshgrid(x, y, indexing='ij')

# CFL condition (2D heat equation)
dt = 0.1 * min(dx, dy)**2 / (2 * alpha)  # Factor of 2 for 2D
Nt = int(T_final / dt)

print(f"Grid: {Nx} x {Ny}, dx = {dx:.4f}, dy = {dy:.4f}")
print(f"Time: Nt = {Nt}, dt = {dt:.6f}")
```

```python
# Initial level set: circular interface
center = np.array([0.5, 0.5])
R0 = 0.2  # Initial radius

# φ(x, y) = ||(x,y) - center|| - R0
# φ < 0 inside circle (ice)
# φ > 0 outside circle
phi0 = np.sqrt((X - center[0])**2 + (Y - center[1])**2) - R0

# Create time-dependent domain
ls_domain = TimeDependentDomain(
    phi0,
    grid,
    initial_time=0.0,
    is_signed_distance=True  # Distance to circle
)

print(f"\nInitial interface: Circle at {center}, radius {R0}")
print(f"Interface cells: {np.sum(np.abs(phi0) < 0.01)}")
```

**★ Insight ─────────────────────────────────────**
For a circle, the signed distance function is $\phi(x, y) = \sqrt{(x - x_c)^2 + (y - y_c)^2} - R$. This automatically satisfies $|\nabla\phi| = 1$ (verify: $\nabla\phi = (x - x_c, y - y_c)/r$, so $|\nabla\phi| = r/r = 1$). No reinitialization needed initially!
**─────────────────────────────────────────────────**

### Step 2: Implement 2D Heat Equation Solver

```python
def solve_heat_equation_2d(
    T_prev: np.ndarray,
    dx: float,
    dy: float,
    dt: float,
    alpha: float,
    T_boundary: float = 1.0,
) -> np.ndarray:
    """
    Solve 2D heat equation ∂T/∂t = α·(∂²T/∂x² + ∂²T/∂y²).

    Uses explicit finite difference with Dirichlet BC on all boundaries.
    """
    Nx, Ny = T_prev.shape

    # CFL check
    cfl_x = alpha * dt / dx**2
    cfl_y = alpha * dt / dy**2
    cfl_total = cfl_x + cfl_y
    if cfl_total > 0.5:
        raise ValueError(f"CFL = {cfl_total:.3f} > 0.5, unstable!")

    T_new = T_prev.copy()

    # Interior points
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            # Laplacian: ∂²T/∂x² + ∂²T/∂y²
            laplacian_x = (T_prev[i+1, j] - 2*T_prev[i, j] + T_prev[i-1, j]) / dx**2
            laplacian_y = (T_prev[i, j+1] - 2*T_prev[i, j] + T_prev[i, j-1]) / dy**2
            T_new[i, j] = T_prev[i, j] + alpha * dt * (laplacian_x + laplacian_y)

    # Boundary conditions (Dirichlet: T = T_boundary)
    T_new[0, :] = T_boundary   # Left
    T_new[-1, :] = T_boundary  # Right
    T_new[:, 0] = T_boundary   # Bottom
    T_new[:, -1] = T_boundary  # Top

    return T_new
```

### Step 3: Initialize Temperature and Evolve

```python
# Initial temperature: Hot outside circle, cold inside
T = np.where(phi0 < 0, T_cold, T_hot)

# Storage
time_history = [0.0]
phi_history = [phi0.copy()]
T_history = [T.copy()]

# Time stepping
t = 0.0
for n in range(Nt):
    t += dt

    # 1. Solve heat equation
    T = solve_heat_equation_2d(T, dx, dy, dt, alpha, T_hot)

    # 2. Compute velocity field from heat flux
    # V = -κ·∇T·n, where n = ∇φ/|∇φ|
    # For simplicity, compute velocity at interface cells

    phi_current = ls_domain.get_level_set_at_time(t - dt)

    # Gradient of temperature (central differences)
    grad_T_x = np.zeros_like(T)
    grad_T_y = np.zeros_like(T)
    grad_T_x[1:-1, :] = (T[2:, :] - T[:-2, :]) / (2 * dx)
    grad_T_y[:, 1:-1] = (T[:, 2:] - T[:, :-2]) / (2 * dy)

    # Normal vector: n = ∇φ/|∇φ|
    grad_phi_x = np.gradient(phi_current, dx, axis=0)
    grad_phi_y = np.gradient(phi_current, dy, axis=1)
    grad_phi_mag = np.sqrt(grad_phi_x**2 + grad_phi_y**2) + 1e-10

    normal_x = grad_phi_x / grad_phi_mag
    normal_y = grad_phi_y / grad_phi_mag

    # Normal velocity: V_n = -κ·(∇T·n)
    velocity = -kappa_thermal * (grad_T_x * normal_x + grad_T_y * normal_y)

    # 3. Evolve level set
    ls_domain.evolve_step(velocity, dt)

    # 4. Store history (every 20 steps)
    if n % 20 == 0:
        time_history.append(t)
        phi_current = ls_domain.get_level_set_at_time(t)
        phi_history.append(phi_current.copy())
        T_history.append(T.copy())

        if n % 100 == 0:
            # Compute current radius (average distance of zero level set from center)
            interface_mask = np.abs(phi_current) < 0.02
            if interface_mask.sum() > 0:
                R_current = np.mean(np.sqrt((X[interface_mask] - center[0])**2 +
                                             (Y[interface_mask] - center[1])**2))
                print(f"t = {t:.3f}, R(t) ≈ {R_current:.4f}")

print(f"\nSimulation complete: t = {t:.3f}")
```

### Step 4: Validate Symmetry and Mass Conservation

```python
# Check symmetry: interface should remain circular
phi_final = phi_history[-1]
interface_mask = np.abs(phi_final) < 0.02

# Compute center of mass of interface (should be at (0.5, 0.5))
x_interface = X[interface_mask]
y_interface = Y[interface_mask]
center_computed = np.array([x_interface.mean(), y_interface.mean()])

# Compute aspect ratio (max extent in x vs y)
x_extent = x_interface.max() - x_interface.min()
y_extent = y_interface.max() - y_interface.min()
aspect_ratio = max(x_extent, y_extent) / min(x_extent, y_extent)

print("\nSymmetry Validation:")
print(f"  Interface center: ({center_computed[0]:.4f}, {center_computed[1]:.4f})")
print(f"  Expected center:  ({center[0]}, {center[1]})")
print(f"  Center error: {np.linalg.norm(center_computed - center):.4f}")
print(f"  Aspect ratio: {aspect_ratio:.3f} (should be ≈ 1.0)")

# Energy conservation check
# Total energy: ∫T dx dy + latent_heat·Area(ice)
# (Simplified: just check temperature integral)
energy_initial = T_history[0].sum() * dx * dy
energy_final = T_history[-1].sum() * dx * dy
energy_change = abs(energy_final - energy_initial) / energy_initial

print(f"\nEnergy Conservation:")
print(f"  Initial energy: {energy_initial:.4f}")
print(f"  Final energy: {energy_final:.4f}")
print(f"  Relative change: {100 * energy_change:.2f}%")
```

### Step 5: Visualize 2D Results

```python
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# Row 0: Interface evolution at different times
time_indices = [0, len(phi_history)//3, 2*len(phi_history)//3, -1]
titles = ["(a) t = 0.00", "(b) t ≈ T/3", "(c) t ≈ 2T/3", "(d) t = T"]

for plot_idx, (time_idx, title) in enumerate(zip(time_indices, titles)):
    ax = fig.add_subplot(gs[0, plot_idx // 2])
    if plot_idx >= 2:
        ax = fig.add_subplot(gs[1, (plot_idx - 2)])

    phi_plot = phi_history[time_idx]
    T_plot = T_history[time_idx]

    # Show temperature field as background
    im = ax.contourf(X, Y, T_plot, levels=20, cmap='hot')
    plt.colorbar(im, ax=ax, label="Temperature")

    # Overlay interface (φ = 0 contour)
    ax.contour(X, Y, phi_plot, levels=[0], colors='cyan', linewidths=3)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{title} (R ≈ {R0 - time_history[time_idx]*0.05:.3f})")
    ax.set_aspect('equal')

# Additional plot: Radius evolution
ax_radius = fig.add_subplot(gs[1, 2])
radii = []
for phi_t in phi_history:
    interface_mask = np.abs(phi_t) < 0.02
    if interface_mask.sum() > 0:
        R_t = np.mean(np.sqrt((X[interface_mask] - center[0])**2 +
                               (Y[interface_mask] - center[1])**2))
        radii.append(R_t)
    else:
        radii.append(0.0)

ax_radius.plot(time_history, radii, 'b-', linewidth=2)
ax_radius.set_xlabel("Time t")
ax_radius.set_ylabel("Interface radius R(t)")
ax_radius.set_title("(e) Interface Radius Evolution")
ax_radius.grid(True, alpha=0.3)

plt.suptitle("2D Stefan Problem - Circular Interface", fontsize=16, fontweight="bold")
plt.show()
```

**Expected Observations**:
- Interface shrinks uniformly (symmetric)
- Aspect ratio ≈ 1.0 (circular shape preserved)
- Temperature smooth outside circle, cold inside
- Radius decreases roughly as $R(t) \approx R_0 - V \cdot t$ (constant melting rate for uniform $T_{\text{boundary}}$)

### Key Takeaways from Tutorial 2

✅ **2D Level Set**: Same equation, higher-dimensional gradients
✅ **Symmetry preservation**: SDF maintains circular shape
✅ **Normal velocity**: $V_n = -\kappa \cdot \nabla T \cdot n$ computed from gradient
✅ **CFL in 2D**: $\alpha \Delta t (\frac{1}{\Delta x^2} + \frac{1}{\Delta y^2}) < 0.5$

---

## Understanding the Results

### What is |∇φ| and Why Does It Matter?

The **gradient magnitude** $|\nabla\phi|$ has a geometric meaning:

```
If |∇φ| = 1 (signed distance function):
    φ(x) = ±distance from x to interface

If |∇φ| ≠ 1 (general level set):
    φ(x) = some scalar value (sign still indicates inside/outside)
```

**Why maintain |∇φ| = 1?**
1. **Numerical stability**: Smooth gradients prevent CFL violations
2. **Accurate normals**: $n = \nabla\phi$ without renormalization
3. **Distance queries**: $|\phi(x)|$ = actual distance (useful for narrow band methods)

**When to reinitialize**:
- Check: `max(|np.gradient(phi)| - 1) > 0.2` (>20% deviation)
- Frequency: Every 5-10 time steps, or when solver becomes unstable
- Method: Solve $\partial\phi/\partial\tau + \text{sign}(\phi_0)(|\nabla\phi| - 1) = 0$ for 20-50 pseudo-time steps

### Interpreting Interface Velocity

**Stefan condition**: $V = -\kappa \cdot [\partial T/\partial x]$

**Physical meaning**:
- Heat flows from hot to cold: $\partial T/\partial x < 0$ at interface (decreasing toward ice)
- Negative gradient → positive flux into ice
- Positive flux → melting → interface moves in direction of heat source
- Sign convention: $V > 0$ moves interface in positive $x$ direction

**Numerical computation**:
```python
# Centered difference (2nd order)
grad_T = (T[i+1] - T[i-1]) / (2*dx)

# One-sided at boundaries (1st order)
grad_T_left = (T[i+1] - T[i]) / dx
grad_T_right = (T[i] - T[i-1]) / dx
```

### Mass/Energy Conservation

**Heat equation conserves energy** (with appropriate BC):
```
d/dt ∫T dx = α·∫∂²T/∂x² dx = α·[∂T/∂x]_boundary
```

**Level Set does NOT conserve mass** inherently:
- Interface can "disappear" if it shrinks to zero
- Interface can merge/split (topology changes)
- Use **volume correction** if exact mass needed:
  ```python
  # Compute volume inside φ < 0
  volume = np.sum(phi < 0) * dx * dy
  # Rescale to match target volume (if desired)
  ```

---

## Numerical Stability

### CFL Conditions Summary

| Equation | CFL Condition | Typical Value |
|:---------|:--------------|:--------------|
| **Heat equation** (1D) | $\alpha \frac{\Delta t}{\Delta x^2} < 0.5$ | 0.2-0.4 |
| **Heat equation** (2D) | $\alpha \Delta t (\frac{1}{\Delta x^2} + \frac{1}{\Delta y^2}) < 0.5$ | 0.1-0.2 |
| **Level Set** (1D) | $\frac{V \Delta t}{\Delta x} < 1$ | 0.5-0.9 |
| **Level Set** (2D) | $V \Delta t (\frac{1}{\Delta x} + \frac{1}{\Delta y}) < 1$ | 0.5-0.9 |

**Combined problems**: Use $\Delta t = \min(\Delta t_{\text{heat}}, \Delta t_{\text{LS}})$

**Adaptive time stepping**:
```python
# Compute CFL-limited time step at each iteration
dt_heat = 0.2 * dx**2 / alpha
dt_ls = 0.9 * dx / max(abs(velocity))
dt = min(dt_heat, dt_ls)
```

### Reinitialization Stability

**When to reinitialize**:
```python
# Check gradient magnitude
grad_phi = np.gradient(phi, dx)
deviation = np.abs(np.abs(grad_phi) - 1.0).max()

if deviation > 0.15:  # 15% deviation threshold
    phi = reinitialize(phi, dx, num_iterations=20)
```

**Reinitialization algorithm** (pseudo-time evolution):
```python
def reinitialize(phi_initial, dx, num_iterations=20):
    """Restore |∇φ| = 1 via ∂φ/∂τ + sign(φ₀)(|∇φ| - 1) = 0."""
    phi = phi_initial.copy()
    sign_phi0 = np.sign(phi_initial + 1e-10)  # Avoid sign(0)
    dtau = 0.5 * dx  # Pseudo-time step

    for _ in range(num_iterations):
        # Compute gradient magnitude
        grad_phi = np.gradient(phi, dx)
        grad_mag = np.abs(grad_phi)

        # Update: φ^{k+1} = φ^k - dtau·sign(φ₀)·(|∇φ| - 1)
        phi = phi - dtau * sign_phi0 * (grad_mag - 1.0)

    return phi
```

**Expected result**: $\max ||\nabla\phi| - 1| < 0.1$ after 20 iterations

---

## Common Pitfalls

### 1. Forgetting to Check Both CFL Conditions

**Problem**: Use heat equation CFL, ignore level set CFL → instability when velocity is large.

**Solution**:
```python
# ALWAYS compute both
cfl_heat = alpha * dt / dx**2
cfl_ls = max(abs(velocity)) * dt / dx

print(f"CFL heat: {cfl_heat:.3f}, CFL LS: {cfl_ls:.3f}")
assert cfl_heat < 0.5 and cfl_ls < 1.0, "CFL violation!"
```

### 2. Poor Initial Temperature Profile

**Problem**: Constant initial temperature → huge heat flux spike → interface jumps.

**Solution**: Use smooth initial profile consistent with boundary conditions:
```python
# Bad: T = T_cold everywhere
T = np.full(Nx, T_cold)  # Discontinuity at hot boundary!

# Good: Piecewise linear matching boundaries
T = np.where(x < s0, T_hot * (s0 - x) / s0, T_cold)  # Smooth in each region
```

### 3. Not Storing Interface History

**Problem**: Only store current φ, lose track of interface motion for analysis.

**Solution**: Store φ at regular intervals:
```python
phi_history = []
time_history = []

for n in range(Nt):
    # ... evolution ...
    if n % 10 == 0:  # Every 10 steps
        phi_history.append(phi.copy())  # COPY, not reference!
        time_history.append(t)
```

### 4. Using np.gradient Without Understanding Output

**Problem**: `np.gradient(phi, dx)` returns **actual gradient**, not normalized.

**Correct usage**:
```python
# Gradient magnitude
grad_phi = np.gradient(phi, dx)
grad_mag = np.abs(grad_phi)  # This is |∇φ|, NOT the normal!

# Normal vector (requires normalization)
normal = grad_phi / (grad_mag + 1e-10)  # Add epsilon to avoid /0
```

### 5. Interface Disappearing or Merging

**Problem**: Circle shrinks to zero → φ has no zero level set → tracking fails.

**Solution**: Monitor interface:
```python
interface_exists = np.any((phi[:-1] * phi[1:]) < 0)  # Sign change indicates crossing
if not interface_exists:
    print("Warning: Interface disappeared!")
    break  # Stop simulation
```

---

## Advanced Topics

### 1. Narrow Band Methods

**Idea**: Only evolve φ near interface (save computation).

```python
# Define narrow band: |φ| < bandwidth
bandwidth = 5 * dx
narrow_band_mask = np.abs(phi) < bandwidth

# Evolve only in narrow band
velocity_nb = velocity.copy()
velocity_nb[~narrow_band_mask] = 0

ls_domain.evolve_step(velocity_nb, dt)

# Periodically reinitialize to extend SDF beyond band
if n % 10 == 0:
    phi = reinitialize(phi, dx, num_iterations=20)
```

**Benefit**: 5-10× speedup for 2D/3D problems with small interfaces.

### 2. Higher-Order Schemes (WENO)

**Default**: First-order upwind (Godunov) → numerical diffusion.

**Upgrade**: WENO5 (5th-order Weighted Essentially Non-Oscillatory).

```python
# MFG_PDE currently uses Godunov (Issue #592)
# Future: WENO5 for reduced diffusion

# Expected improvement:
#   - Sharper interfaces
#   - Better mass conservation
#   - Higher accuracy (5th-order vs 1st-order)
```

**When to use**: Long-time simulations where diffusion accumulates.

### 3. Curvature-Driven Flow

**Extension**: Add surface tension term $V = V_{\text{Stefan}} + \gamma \kappa$

```python
# Compute mean curvature κ = ∇·(∇φ/|∇φ|)
from mfg_pde.geometry.level_set import compute_curvature

kappa = compute_curvature(phi, grid)

# Velocity includes curvature
gamma = 0.01  # Surface tension coefficient
velocity_total = velocity_stefan + gamma * kappa
```

**Application**: Droplet coalescence, bubble dynamics, Mullins-Sekerka instability.

### 4. Multiple Level Sets

**Problem**: Track multiple interfaces (e.g., ice-water-vapor).

**Solution**: Use multiple φ functions:
```python
phi_ice_water = x - s1(t)   # Ice-water interface
phi_water_vapor = x - s2(t) # Water-vapor interface

# Region classification
ice_region = (phi_ice_water < 0)
water_region = (phi_ice_water > 0) & (phi_water_vapor < 0)
vapor_region = (phi_water_vapor > 0)
```

---

## Summary and Best Practices

### Quick Start Checklist

1. ✅ **Initialize φ as SDF**: $\phi = \pm \text{distance to interface}$
2. ✅ **Check CFL conditions**: Both heat equation and level set
3. ✅ **Smooth initial data**: Avoid discontinuities in PDE solution
4. ✅ **Evolve coupled system**:
   - Solve PDE (heat equation)
   - Compute velocity from PDE (Stefan condition)
   - Evolve level set with velocity
5. ✅ **Monitor |∇φ|**: Reinitialize if $\max ||\nabla\phi| - 1| > 0.15$
6. ✅ **Store history**: Save φ, T, and interface positions for analysis
7. ✅ **Validate**: Compare with analytical solution or check symmetry

### Best Practices

**DO**:
- ✅ Use fine grids (Nx ≥ 200 for 1D, ≥ 100² for 2D)
- ✅ Start with conservative CFL (0.2 for heat, 0.5 for LS)
- ✅ Visualize φ evolution (not just interface position)
- ✅ Check mass/energy conservation
- ✅ Validate against analytical solutions when available

**DON'T**:
- ❌ Use constant initial temperature (creates flux spike)
- ❌ Ignore level set CFL (instability!)
- ❌ Forget to reinitialize (φ degrades over time)
- ❌ Store only final φ (lose evolution history)
- ❌ Use φ directly for visualization (extract zero level set instead)

### Further Reading

- **Theory**: `docs/theory/level_set_method.md` (complete mathematical derivation)
- **Examples**:
  - `examples/advanced/stefan_problem_1d.py` (ice melting, analytical validation)
  - `examples/advanced/stefan_problem_2d.py` (circular interface)
- **API Reference**: `docs/user/advanced_boundary_conditions.md` (TimeDependentDomain)
- **Textbooks**:
  - Osher & Fedkiw (2003): *Level Set Methods and Dynamic Implicit Surfaces*
  - Sethian (1999): *Level Set Methods and Fast Marching Methods*

---

**Last Updated**: 2026-01-18
**Author**: MFG_PDE Documentation Team
**Related Issues**: #592 (Level Set Methods), #594 (Documentation)
