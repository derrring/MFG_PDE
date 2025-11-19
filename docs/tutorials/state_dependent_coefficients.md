# State-Dependent Coefficients in MFG_PDE

**Tutorial**: Using flexible drift and diffusion fields in Mean Field Games

**Level**: Intermediate
**Version**: 0.13.3+
**Prerequisites**: Basic MFG problem setup

---

## Overview

MFG_PDE provides a **unified, flexible API** for specifying drift and diffusion coefficients in Fokker-Planck equations. This tutorial shows how to use:

1. **Constant coefficients** (classical MFG)
2. **Spatially varying coefficients**
3. **Time-dependent coefficients**
4. **State-dependent coefficients** (depends on density m)
5. **Combined scenarios** (multiple dependencies)

**Key Insight**: The current API already supports all these use cases without requiring a Strategy Pattern (see Issue #335 evaluation).

---

## 1. Quick Reference: Coefficient API

### Supported Types

```python
# Drift field α(t, x, m)
drift_field: None | np.ndarray | Callable = None

# Diffusion field σ(t, x, m)
diffusion_field: None | float | np.ndarray | Callable = None
```

### Type Meanings

| Type | Meaning | Example |
|:-----|:--------|:--------|
| `None` | Zero drift / Default diffusion | Pure diffusion, classical MFG |
| `float` | Constant coefficient | Isotropic diffusion σ² = 0.1 |
| `np.ndarray` | Precomputed field | Optimal control drift, spatially varying |
| `Callable` | State-dependent function | Density-dependent, time-varying |

---

## 2. Pure Diffusion (Zero Drift)

**Use Case**: Passive agent dispersion, heat equation

```python
from mfg_pde import MFGProblem
from mfg_pde.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver

# Setup problem
problem = MFGProblem(Nx=100, xmin=0, xmax=1, Nt=50, T=1.0, sigma=0.1)

# Create solver
solver = FPFDMSolver(problem)

# Solve with zero drift (pure diffusion)
m0 = problem.get_initial_m()
M = solver.solve_fp_system(
    m_initial_condition=m0,
    drift_field=None  # ← Zero drift
)
```

**Equation**: ∂m/∂t = (σ²/2) Δm

---

## 3. Constant Wind (Prescribed Drift)

**Use Case**: Agents moving in constant background flow

### Method 1: Precomputed Array

```python
import numpy as np

# Constant drift field (1D example)
Nt, Nx = 51, 101
drift_constant = np.ones((Nt, Nx)) * 0.5  # Velocity = 0.5

M = solver.solve_fp_system(
    m_initial_condition=m0,
    drift_field=drift_constant
)
```

### Method 2: Callable (More Flexible)

```python
def constant_wind(t: float, x: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Constant rightward drift."""
    return np.full_like(x, 0.5)

M = solver.solve_fp_system(
    m_initial_condition=m0,
    drift_field=constant_wind
)
```

**Equation**: ∂m/∂t + ∇·(0.5 m) = (σ²/2) Δm

---

## 4. Spatially Varying Diffusion

**Use Case**: Heterogeneous medium (e.g., porous material with varying permeability)

```python
def spatially_varying_diffusion(t: float, x: np.ndarray, m: np.ndarray) -> np.ndarray:
    """
    Diffusion varies with position.

    Higher diffusion in center, lower at edges.
    """
    x_center = 0.5
    base_diffusion = 0.05
    peak_diffusion = 0.20

    # Gaussian profile centered at 0.5
    distance_from_center = np.abs(x - x_center)
    diffusion = base_diffusion + (peak_diffusion - base_diffusion) * np.exp(-50 * distance_from_center**2)

    return diffusion

M = solver.solve_fp_system(
    m_initial_condition=m0,
    drift_field=None,
    diffusion_field=spatially_varying_diffusion
)
```

**Equation**: ∂m/∂t = (1/2) ∇·(σ²(x) ∇m)

**Result**: Agents spread faster in the center than at edges.

---

## 5. Time-Dependent Drift

**Use Case**: Time-varying external forces (e.g., tidal flow, traffic signals)

```python
def time_varying_wind(t: float, x: np.ndarray, m: np.ndarray) -> np.ndarray:
    """
    Wind direction oscillates with time.

    Rightward in first half, leftward in second half.
    """
    if t < 0.5:
        return np.full_like(x, 0.3)  # Rightward
    else:
        return np.full_like(x, -0.3)  # Leftward

M = solver.solve_fp_system(
    m_initial_condition=m0,
    drift_field=time_varying_wind
)
```

**Equation**: ∂m/∂t + ∇·(α(t) m) = (σ²/2) Δm

---

## 6. Density-Dependent Diffusion

**Use Case**: Crowding effects (diffusion increases in dense regions)

```python
def density_dependent_diffusion(t: float, x: np.ndarray, m: np.ndarray) -> np.ndarray:
    """
    Diffusion increases with local density.

    Models: Agents spread faster when crowded.
    """
    base_diffusion = 0.05
    crowding_factor = 0.10

    # Diffusion = base + crowding_factor * m(x)
    # Higher density → higher diffusion
    diffusion = base_diffusion + crowding_factor * m

    return diffusion

M = solver.solve_fp_system(
    m_initial_condition=m0,
    drift_field=None,
    diffusion_field=density_dependent_diffusion
)
```

**Equation**: ∂m/∂t = (1/2) ∇·(σ²(m) ∇m)

**Physical Interpretation**: Crowded regions have higher "social pressure" causing faster dispersion.

---

## 7. Repulsive Drift (Anti-Crowding)

**Use Case**: Agents avoid crowded areas

```python
def repulsive_drift(t: float, x: np.ndarray, m: np.ndarray) -> np.ndarray:
    """
    Drift away from high-density regions.

    α(x) = -∇m / (m + ε)
    """
    epsilon = 1e-6  # Regularization to avoid division by zero

    # Compute gradient of m using finite differences
    dm_dx = np.gradient(m, x)

    # Drift proportional to -∇m (away from density gradient)
    drift = -dm_dx / (m + epsilon)

    return drift

M = solver.solve_fp_system(
    m_initial_condition=m0,
    drift_field=repulsive_drift,
    diffusion_field=0.05
)
```

**Equation**: ∂m/∂t + ∇·(-∇m/(m+ε) · m) = (σ²/2) Δm

**Result**: Agents naturally spread out, avoiding clusters.

---

## 8. MFG Optimal Control Drift

**Use Case**: Mean Field Game with optimal control

```python
from mfg_pde.factory import create_standard_solver

# Full MFG problem
problem = MFGProblem(Nx=100, xmin=0, xmax=1, Nt=50, T=1.0, sigma=0.1)

# Create standard MFG solver (HJB + FP coupling)
solver = create_standard_solver(problem)

# Solve full MFG system
result = solver.solve()

# Extract optimal control drift from HJB solution
U = result.U
dx = problem.dx
sigma_sq = problem.sigma ** 2

# Optimal drift: α*(x) = -∇U / σ²
grad_U = np.gradient(U, dx, axis=1)  # Gradient along spatial axis
optimal_drift = -grad_U / sigma_sq

# Use in standalone FP solve (for testing/validation)
from mfg_pde.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver
fp_solver = FPFDMSolver(problem)

M_test = fp_solver.solve_fp_system(
    m_initial_condition=problem.get_initial_m(),
    drift_field=optimal_drift
)
```

**Equation**:
- HJB: -∂u/∂t + H(∇u, m) - (σ²/2)Δu = 0
- FP: ∂m/∂t + ∇·(α* m) = (σ²/2)Δm, where α* = -∇u/σ²

---

## 9. Composite Drift (Multiple Sources)

**Use Case**: Agents subject to multiple forces simultaneously

```python
def composite_drift(t: float, x: np.ndarray, m: np.ndarray) -> np.ndarray:
    """
    Combine multiple drift sources.

    Total drift = external wind + repulsive (anti-crowding) + gravitational
    """
    # 1. External wind (constant)
    wind_drift = 0.2

    # 2. Repulsive drift (density-dependent)
    epsilon = 1e-6
    dm_dx = np.gradient(m, x)
    repulsion_drift = -0.1 * dm_dx / (m + epsilon)

    # 3. Gravitational drift (spatially varying)
    gravity_center = 0.5
    gravity_strength = 0.15
    gravity_drift = gravity_strength * (gravity_center - x)

    # Combine all sources
    total_drift = wind_drift + repulsion_drift + gravity_drift

    return total_drift

M = solver.solve_fp_system(
    m_initial_condition=m0,
    drift_field=composite_drift,
    diffusion_field=0.05
)
```

**Result**: Complex agent dynamics from multiple simultaneous effects.

---

## 10. Anisotropic Diffusion (2D)

**Use Case**: Directional diffusion (e.g., flow in channels, geological formations)

```python
from mfg_pde import MFGProblem
from mfg_pde.alg.numerical.fp_solvers.fp_fdm import FPFDMSolver

# 2D problem
problem = MFGProblem(
    spatial_bounds=[(0, 1), (0, 1)],
    spatial_discretization=[50, 50],
    Nt=50,
    T=1.0
)

def anisotropic_diffusion_2d(t: float, x: np.ndarray, m: np.ndarray) -> np.ndarray:
    """
    Diagonal tensor diffusion: different rates in x and y.

    σ_x² = 0.2 (high diffusion in x-direction)
    σ_y² = 0.05 (low diffusion in y-direction)

    Returns:
        Array of shape (N, 2) with [σ_x², σ_y²] for each point
    """
    N = len(x)
    diffusion_tensor = np.zeros((N, 2))
    diffusion_tensor[:, 0] = 0.2  # x-direction
    diffusion_tensor[:, 1] = 0.05  # y-direction

    return diffusion_tensor

solver = FPFDMSolver(problem)
M = solver.solve_fp_system(
    m_initial_condition=problem.get_initial_m(),
    drift_field=None,
    diffusion_field=anisotropic_diffusion_2d
)
```

**Equation**: ∂m/∂t = (1/2)(∂²m/∂x² · σ_x² + ∂²m/∂y² · σ_y²)

**Result**: Agents spread faster horizontally than vertically.

---

## 11. Complete Example: Traffic Flow with Congestion

**Scenario**: Vehicle traffic with:
1. Desired velocity (rightward drift)
2. Congestion-dependent slowdown (density-dependent drift)
3. Random lane changes (diffusion)

```python
import numpy as np
import matplotlib.pyplot as plt
from mfg_pde import MFGProblem
from mfg_pde.alg.numerical.fp_solvers.fp_particle import FPParticleSolver

# Setup: 1D road from x=0 to x=1
problem = MFGProblem(
    Nx=200,
    xmin=0.0,
    xmax=1.0,
    Nt=100,
    T=2.0,
    sigma=0.05  # Lane change diffusion
)

def traffic_drift(t: float, x: np.ndarray, m: np.ndarray) -> np.ndarray:
    """
    Traffic velocity with congestion slowdown.

    v(m) = v_free * (1 - m/m_jam)

    Args:
        t: Time (not used here)
        x: Positions
        m: Local traffic density

    Returns:
        Velocity field (drift)
    """
    v_free = 0.8       # Free-flow velocity
    m_jam = 2.0        # Jam density (traffic stops)

    # Velocity decreases linearly with density
    velocity = v_free * (1.0 - np.clip(m / m_jam, 0, 1))

    return velocity

# Initial condition: Traffic jam at entrance (x=0)
m0 = problem.get_initial_m()
# Override with localized jam
x_grid = problem.xSpace
m0 = 3.0 * np.exp(-100 * (x_grid - 0.1)**2)  # Jam near entrance
m0 /= np.sum(m0) * problem.dx  # Normalize

# Solve using particle method (better for shocks/discontinuities)
solver = FPParticleSolver(problem, num_particles=5000)
M = solver.solve_fp_system(
    m_initial_condition=m0,
    drift_field=traffic_drift,
    diffusion_field=0.02  # Lane changes
)

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Plot density evolution
for i, t_idx in enumerate([0, 33, 66, 99]):
    ax = axes.flat[i]
    ax.plot(x_grid, M[t_idx, :], 'b-', linewidth=2)
    ax.set_xlabel('Position (road location)')
    ax.set_ylabel('Density (vehicles/unit length)')
    ax.set_title(f't = {t_idx * problem.dt:.2f}')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 3.5])

plt.tight_layout()
plt.savefig('traffic_congestion_evolution.png', dpi=150)
print("Traffic flow simulation complete!")
print(f"Initial jam dissipates over time due to congestion-dependent drift")
```

**Key Features**:
1. **Density-dependent drift**: v(m) = v_free(1 - m/m_jam)
2. **Automatic slowdown**: High density → low velocity
3. **Diffusion**: Random lane changes smooth out shocks

**Output**: Traffic jam gradually dissipates as vehicles spread out.

---

## 12. API Design Philosophy (Issue #335)

### Why Not Strategy Pattern?

**Question**: Should we use explicit Strategy Pattern for drift types?

```python
# Proposed Strategy Pattern (NOT RECOMMENDED)
solver.solve_fp_system(
    m0,
    drift_strategy=OptimalControlDrift(U, problem)
)
```

**Answer**: **No** - Current flexible API is better.

**Rationale**:
1. ✅ **Current API already flexible**: Accepts None | array | Callable
2. ✅ **Simpler**: No extra class hierarchy to learn
3. ✅ **Pythonic**: Functions are first-class citizens
4. ✅ **Composable**: Easy to combine multiple drifts (see Example 9)

### Optional: Helper Functions (Alternative to Strategy Pattern)

If you want **named patterns without class overhead**:

```python
# Future enhancement (not yet implemented):
from mfg_pde.utils.drift import optimal_control_drift, zero_drift, composite_drift

# Pure diffusion
M = solver.solve_fp_system(m0, drift_field=zero_drift())

# MFG optimal control
M = solver.solve_fp_system(m0, drift_field=optimal_control_drift(U, problem))

# Multiple sources
M = solver.solve_fp_system(
    m0,
    drift_field=composite_drift([
        optimal_control_drift(U, problem),
        lambda t, x, m: 0.1  # External wind
    ])
)
```

**Status**: This could be added in future release if useful, but **current API already achieves the goal**.

---

## 13. Best Practices

### DO ✅

1. **Use callables for state-dependence**
   ```python
   def my_drift(t, x, m):
       return compute_drift(t, x, m)
   ```

2. **Add regularization for division by m**
   ```python
   drift = -dm_dx / (m + 1e-10)  # Avoid division by zero
   ```

3. **Return arrays matching input shape**
   ```python
   def my_diffusion(t, x, m):
       return np.full_like(x, 0.1)  # Same shape as x
   ```

4. **Document physical meaning**
   ```python
   def gravity_drift(t, x, m):
       """Gravitational attraction to center at x=0.5."""
       return 0.2 * (0.5 - x)
   ```

### DON'T ❌

1. **Don't modify input arrays**
   ```python
   # BAD
   def bad_drift(t, x, m):
       m[0] = 0  # ❌ Modifies input!
       return x
   ```

2. **Don't use global state**
   ```python
   # BAD
   global_counter = 0
   def bad_drift(t, x, m):
       global global_counter
       global_counter += 1  # ❌ Non-deterministic!
   ```

3. **Don't ignore arguments**
   ```python
   # BAD (if you need state-dependence)
   def bad_drift(t, x, m):
       return 0.5  # ❌ Ignores m when it shouldn't
   ```

---

## 14. Type Hints for Custom Callables

**New in v0.13.3+**: Type protocols for better IDE support

```python
from mfg_pde.types import DriftFieldCallable, DiffusionFieldCallable
import numpy as np

def my_drift(
    t: float,
    x: np.ndarray,
    m: np.ndarray
) -> np.ndarray:
    """Custom drift field with full type hints."""
    return -0.1 * np.gradient(m, x)

# Type checker will validate signature
drift: DriftFieldCallable = my_drift

# Use in solver
M = solver.solve_fp_system(m0, drift_field=drift)
```

**Benefits**:
- ✅ IDE autocomplete
- ✅ MyPy type checking
- ✅ Better documentation

---

## 15. Performance Considerations

### Precompute vs Callable

**Precomputed (Faster)**:
```python
# Compute once
drift_array = compute_optimal_drift(U, problem)

# Reuse many times
M1 = solver.solve_fp_system(m0, drift_field=drift_array)
M2 = solver.solve_fp_system(m0_alt, drift_field=drift_array)  # Same drift
```

**Callable (More Flexible)**:
```python
# Recomputed each timestep
def state_dependent_drift(t, x, m):
    return compute_drift(t, x, m)  # Depends on current m

M = solver.solve_fp_system(m0, drift_field=state_dependent_drift)
```

**Recommendation**:
- Use **precomputed** for MFG optimal control (drift doesn't change during FP solve)
- Use **callable** for true state-dependence (drift depends on evolving m)

---

## Summary

MFG_PDE's coefficient API supports:

| Feature | Type | Example |
|:--------|:-----|:--------|
| Zero drift | `None` | Pure diffusion |
| Constant | `float`, array | Background flow |
| Spatially varying | Array, callable | Heterogeneous medium |
| Time-dependent | Callable | Oscillating forces |
| Density-dependent | Callable | Crowding effects |
| Composite | Callable | Multiple simultaneous effects |
| Anisotropic | Array (2D+) | Directional diffusion |

**Key Insight**: Current flexible API (None | scalar | array | Callable) already achieves all use cases without requiring Strategy Pattern.

---

## Further Reading

- **API Evaluation**: Issue #335 analysis
- **Examples**: `examples/basic/state_dependent_diffusion_simple.py`
- **Type Protocols**: `mfg_pde/types/callable_protocols.py`
- **Theory**: Fokker-Planck equations with state-dependent coefficients

**Next Tutorial**: Tensor diffusion and anisotropic PDEs
