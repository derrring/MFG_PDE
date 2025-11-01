# MFG_PDE Code Naming Conventions

**Last Updated**: 2025-10-31
**Status**: Transition to full English names in progress
**Related**: See `docs/theory/foundations/NOTATION_STANDARDS.md` for mathematical notation

---

## Purpose and Scope

This document defines **Python code parameter naming conventions** for MFG_PDE solvers and problem definitions.

- **This document**: Python variable names (e.g., `damping_factor`, `num_time_steps`)
- **NOTATION_STANDARDS.md**: Mathematical notation in theory docs (e.g., $N_x$, $\Delta t$, $m(t,x)$)

---

## Naming Convention Standards

MFG_PDE uses **full English names** for clarity and readability in code.

### Core Solver Parameters

| Old Name | New Name | Meaning | Math Notation |
|----------|----------|---------|---------------|
| `thetaUM` | `damping_factor` | Picard iteration damping (0-1) | $\theta$ |
| `Niter_max` | `max_iterations` | Maximum Picard iterations | — |
| `l2errBound` | `tolerance` | Convergence tolerance | $\epsilon$ |

### Discretization Parameters

| Old Name | New Name | Meaning | Math Notation |
|----------|----------|---------|---------------|
| `Nt` | `num_time_steps` | Time steps (Nt+1 points) | $N_t$ |
| `Nx`, `Ny`, `Nz` | `num_intervals_x/y/z` | Spatial intervals (see Grid Convention) | $N_x, N_y, N_z$ |
| `Dx`, `dx` | `grid_spacing` | Spatial grid spacing | $\Delta x$ or $h$ |
| `Dt`, `dt` | `time_step` | Temporal step size | $\Delta t$ |
| `sigma` | `diffusion_coeff` | Diffusion coefficient | $\sigma$ |

---

## Grid Convention ⚠️ **CRITICAL**

This follows the standard finite difference discretization defined in `NOTATION_STANDARDS.md` §7.2.

### Mathematical Definition

Domain $[a, b]$ divided into $N$ equal **intervals**:
- Grid spacing: $h = (b-a)/N$
- Grid points: $x_i = a + ih$ for $i = 0, 1, \ldots, N$
- **Array size**: $N+1$ points (includes both boundaries)

### Code Implementation

**Legacy 1D MFGProblem** (standard FDM convention):
```python
Nx = 50                                # 50 intervals
xSpace = np.linspace(xmin, xmax, Nx+1) # 51 grid points
Dx = (xmax - xmin) / Nx                # Interval spacing

# Arrays have shape (Nx+1,)
m_init = np.zeros(Nx + 1)              # Initial density: 51 points
u_fin = np.zeros(Nx + 1)               # Final value: 51 points
```

**GridBasedMFGProblem (nD)** (modern convention):
```python
# User specifies number of grid points directly
grid_resolution = 51                   # Number of points (includes boundaries)
problem = GridBasedMFGProblem(
    domain_bounds=(0.0, 1.0),
    grid_resolution=51,                # 51 points = 50 intervals internally
)

# Arrays have shape matching num_points
m_init = np.zeros(51)                  # 51 points
```

**Adapter Conversion** (bridging conventions):
```python
# In _FPProblem1DAdapter and _HJBProblem1DAdapter:
num_intervals = num_points - 1         # Convert: 51 points → 50 intervals
self.Nx = num_intervals                # For 1D FDM solver compatibility
self.Dx = grid_spacing                 # Spacing unchanged
```

### Multi-Dimensional Grids

- **1D**: Arrays of shape `(Nx+1,)` for $N_x+1$ points
- **2D**: Arrays of shape `(Nx+1, Ny+1)` for $(N_x+1) \times (N_y+1)$ points
- **Time-dependent**: Arrays of shape `(Nt+1, Nx+1, ...)` for time-space

---

### Examples

```python
# Old style (deprecated)
solver = FixedPointIterator(
    problem, hjb_solver, fp_solver,
    thetaUM=0.6  # ❌ Old name
)

# New style (preferred)
solver = FixedPointIterator(
    problem, hjb_solver, fp_solver,
    damping_factor=0.6  # ✅ New name
)
```

## Files Updated

1. `mfg_pde/alg/numerical/mfg_solvers/fixed_point_iterator.py`
   - `thetaUM` → `damping_factor` ✓
   - `Nt` → `num_time_steps` ✓
   - `Dx` → `grid_spacing` ✓
   - `Dt` → `time_step` ✓

2. *(Additional files will be updated as needed)*

## Backward Compatibility

For now, legacy code using old names will continue to work where possible. Full migration to new naming will happen gradually.

## Rationale

Full English names improve:
- **Readability**: `damping_factor` vs `thetaUM` is self-documenting
- **Clarity**: No confusion between conventions (Nx = intervals vs points)
- **Maintainability**: Easier for new contributors to understand

## Migration Guide

When updating code:
1. Replace parameter names in function signatures
2. Update all internal references
3. Update tests and examples
4. Document changes in commit message
