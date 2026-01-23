# MFG_PDE Naming Conventions

**Last Updated**: 2026-01-23
**Status**: Current reference document (v0.17.1+)
**Related**: See "Derivative Tensor Standard" section for the canonical derivative representation

---

## Purpose

This document defines Python code naming conventions for MFG_PDE based on actual codebase standards, not aspirational goals. Use this as the authoritative reference for parameter and variable naming.

---

## Core Principles

### 1. Mathematical Symbols for Direct Algorithm Implementation

**When to use**: When code directly implements a textbook algorithm or specific mathematical formula.

**Examples**:
- `Nx`: Number of spatial intervals per dimension (array, e.g., `[50, 30]`)
- `Nx_points`: Number of spatial grid points per dimension (`Nx + 1`)
- `Nt`: Number of time intervals
- `Nt_points`: Number of time grid points (`Nt + 1`)
- `dx`: Spatial grid spacing Δx
- `dt`: Time step size Δt
- `sigma`: Diffusion coefficient σ
- `T`: Terminal time
- `xSpace`: Spatial grid array (all dimensions)
- `tSpace`: Temporal grid array

**Rationale**: Direct correspondence to mathematical notation makes algorithm validation straightforward.

**Key Convention (v0.16.0+)**: `Nx` = intervals, `Nx_points` = points. This mirrors the `Nt`/`Nt_points` relationship.

### 2. Descriptive English for Configuration Parameters

**When to use**: When variable represents a conceptual parameter, configuration, or setting rather than a formula symbol.

**Examples**:
- `damping_factor` (NOT `thetaUM`) - Picard iteration damping parameter
- `max_iterations` (NOT `Niter_max`) - Maximum iteration count
- `tolerance` - Generic convergence tolerance
- `l2errBound` - Specific L² error bound (accepted terminology)
- `num_spatial_points` - Total spatial grid points
- `spatial_shape` - Shape tuple for nD arrays
- `coupling_coefficient` (NOT `coefCT`) - MFG coupling strength

**Rationale**: Configuration parameters are not formula symbols. `thetaUM` is "magic number" jargon with no clear meaning. `damping_factor` clearly describes its purpose. Use generic names (`tolerance`) for generic concepts, but preserve specific mathematical terminology (`l2errBound`) when appropriate.

---

## Array-Based Notation Standard (v0.10.0+)

### Canonical Standard: Always Use Arrays for Spatial Quantities

**Goal**: Enable dimension-agnostic algorithms that work seamlessly for 1D, 2D, 3D, and nD problems without special cases.

**Principle**: **All spatial quantities are arrays**, even for 1D problems.

### Array Notation for Spatial Parameters

| Parameter | 1D Example | 2D Example | 3D Example | Type |
|-----------|-----------|-----------|-----------|------|
| `Nx` | `[100]` | `[100, 80]` | `[100, 80, 60]` | `list[int]` |
| `xmin` | `[-2.0]` | `[-2.0, -1.0]` | `[-2.0, -1.0, -0.5]` | `list[float]` |
| `xmax` | `[2.0]` | `[2.0, 1.0]` | `[2.0, 1.0, 0.5]` | `list[float]` |
| `dx` | `[0.04]` | `[0.04, 0.025]` | `[0.04, 0.025, 0.017]` | `list[float]` |
| `Lx` | `[4.0]` | `[4.0, 2.0]` | `[4.0, 2.0, 1.0]` | `list[float]` |

**Key points**:
- **1D uses single-element arrays**: `Nx=[100]` not `Nx=100`
- **No separate `Ny`, `Nz`**: All dimensions in one array `Nx`
- **Natural indexing**: Access dimension `i` via `Nx[i]`, `dx[i]`, `xmin[i]`
- **Algorithms work for arbitrary dimensions** without type checking

### Benefits of Array-First Convention

1. **Dimension-agnostic code**: Algorithms work for 1D/2D/3D/nD without special cases
2. **Consistent interface**: `Nx` is always a list, never "sometimes int, sometimes list"
3. **Natural subscripts**: `for i in range(len(Nx)): print(f"Dimension {i}: {Nx[i]} intervals")`
4. **Eliminates type checking**: No need for `if isinstance(Nx, int)` branches
5. **Easier maintenance**: Single code path for all dimensions

### Migration Strategy: Gradual Deprecation

**Current state (v0.10.0 - v0.11.0)**:
- ✅ Array notation is canonical and recommended
- ✅ Scalar notation still works for 1D (backward compatibility)
- ⚠️ Scalar notation emits `DeprecationWarning`
- ✅ Internal normalization via `MFGProblem._normalize_to_array()`

**Future state (v1.0.0)**:
- ❌ Scalar notation will be removed
- ✅ Only array notation will be accepted

### Deprecation Warnings

When you pass scalar values, you'll see:

```python
from mfg_pde import MFGProblem

# This works but warns:
problem = MFGProblem(Nx=100, xmin=0.0, xmax=1.0, T=1.0)
# DeprecationWarning: Passing scalar Nx=100 is deprecated.
# Use array notation Nx=[100] instead.
# Scalar support will be removed in v1.0.0.

# Recommended:
problem = MFGProblem(Nx=[100], xmin=[0.0], xmax=[1.0], T=1.0)
# No warning
```

### Legacy 1D Scalar Compatibility

For backward compatibility only, 1D problems still accept scalar notation:

```python
# ✅ Modern (dimension-agnostic) - RECOMMENDED
problem = MFGProblem(Nx=[50], xmin=[0.0], xmax=[1.0], ...)

# ⚠️ Legacy (1D only) - DEPRECATED
problem = MFGProblem(Nx=50, xmin=0.0, xmax=1.0, ...)  # Warns, normalized to arrays internally
```

**Implementation**: `MFGProblem._normalize_to_array()` converts scalars to 1-element lists with deprecation warnings.

### Dimension-Agnostic Examples

**1D Problem**:
```python
problem = MFGProblem(
    Nx=[100],           # 100 intervals → 101 grid points
    xmin=[-2.0],        # Lower bound
    xmax=[2.0],         # Upper bound
    T=1.0,
    Nt=50
)
```

**2D Problem**:
```python
problem = MFGProblem(
    Nx=[100, 80],                # 100×80 intervals → 101×81 grid
    xmin=[-2.0, -1.0],           # Lower bounds
    xmax=[2.0, 1.0],             # Upper bounds
    T=1.0,
    Nt=50
)
# Grid spacing: dx = [(2-(-2))/100, (1-(-1))/80] = [0.04, 0.025]
```

**3D Problem**:
```python
problem = MFGProblem(
    Nx=[100, 80, 60],            # 100×80×60 intervals
    xmin=[-2.0, -1.0, -0.5],     # Lower bounds
    xmax=[2.0, 1.0, 0.5],        # Upper bounds
    T=1.0,
    Nt=50
)
```

### Alternative: High-Level API

For explicit multi-dimensional problems, use the high-level API:

```python
problem = MFGProblem(
    spatial_bounds=[(-2.0, 2.0), (-1.0, 1.0)],  # 2D bounds
    spatial_discretization=[100, 80],            # 2D discretization
    time_domain=(1.0, 50),                       # (T, Nt)
    diffusion=0.1
)
```

Both APIs normalize to the same internal array representation.

### Migration Guide: Scalar → Array Notation

**Step 1: Identify scalar usage**
```bash
# Find all scalar Nx usage in your code
grep -r "Nx\s*=\s*[0-9]" your_code/
```

**Step 2: Convert to arrays**
```python
# Before (deprecated):
problem = MFGProblem(Nx=100, xmin=-2.0, xmax=2.0, T=1.0, Nt=50)

# After (recommended):
problem = MFGProblem(Nx=[100], xmin=[-2.0], xmax=[2.0], T=1.0, Nt=50)
```

**Step 3: Suppress warnings during migration**
```python
import warnings

# Temporarily suppress during gradual migration
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    problem = MFGProblem(Nx=100, xmin=0.0, xmax=1.0, T=1.0)
```

**Step 4: Test thoroughly**
- Verify array access: `Nx[0]` instead of `Nx`
- Check loops: `for i in range(len(Nx))` works for all dimensions
- Validate outputs match before/after migration

**Common Pitfalls**:
```python
# ❌ Wrong: Mixing scalar and array
problem = MFGProblem(Nx=[100], xmin=0.0, xmax=1.0, T=1.0)
# xmin and xmax should also be arrays

# ✅ Correct: All arrays
problem = MFGProblem(Nx=[100], xmin=[0.0], xmax=[1.0], T=1.0)

# ❌ Wrong: Accessing scalar like array
Nx = 100
dx = (xmax - xmin) / Nx[0]  # IndexError!

# ✅ Correct: Array access
Nx = [100]
dx = (xmax[0] - xmin[0]) / Nx[0]
```

**Timeline**:
- **v0.10.0 - v0.11.0**: Deprecation warnings active, both work
- **v0.12.0+**: Continued deprecation warnings
- **v1.0.0**: Scalar notation removed, arrays required

---

## Spatial Grid Terminology

### `xSpace` - Universal Spatial Grid Array

**`xSpace`**: Spatial coordinate array for all dimensions
- **1D**: `np.ndarray` of shape `(Nx1+1,)`
  - Example: `xSpace = np.linspace(0, 1, 51)`
- **2D**: `np.ndarray` of shape `(Nx1+1, Nx2+1)` or `(Nx1+1, Nx2+1, 2)` for meshgrid
  - Contains spatial coordinates for entire grid
- **nD**: Array or tuple of coordinate arrays
  - Can be flattened or meshgrid format depending on usage

**Usage**:
- Access via `problem.xSpace` or `geometry.get_spatial_grid()`
- All geometry types provide spatial grid through consistent interface
- GeometryProtocol ensures uniform access across implementations

---

## Discretization Parameters

### ⚠️ CRITICAL CONVENTION: `N*` = Intervals, `N*_points` = Points (v0.16.0+)

**Universal Rule**:
- Variables named `N*` (e.g., `Nx`, `Nt`) represent **number of intervals**
- Variables named `N*_points` (e.g., `Nx_points`, `Nt_points`) represent **number of grid points**
- Relationship: `N*_points = N* + 1` (includes both endpoints)

| Dimension | Intervals Variable | Points Variable | Example |
|-----------|-------------------|-----------------|---------|
| Time | `Nt` | `Nt_points` | `Nt=100` → `Nt_points=101` |
| Space (1D) | `Nx=[50]` | `Nx_points=[51]` | 50 intervals → 51 points |
| Space (2D) | `Nx=[50, 30]` | `Nx_points=[51, 31]` | 50×30 intervals → 51×31 points |

**TensorProductGrid API (v0.16.0+)**:
```python
from mfg_pde.geometry import TensorProductGrid

# Option 1: Specify intervals (like Nt)
grid = TensorProductGrid(dimension=2, bounds=[(0,1), (0,1)], Nx=[50, 30])
# grid.Nx = [50, 30]        # intervals
# grid.Nx_points = [51, 31]  # points

# Option 2: Specify points directly
grid = TensorProductGrid(dimension=2, bounds=[(0,1), (0,1)], Nx_points=[51, 31])
# grid.Nx = [50, 30]        # intervals
# grid.Nx_points = [51, 31]  # points

# DEPRECATED: num_points (use Nx_points instead)
grid = TensorProductGrid(dimension=2, bounds=[(0,1), (0,1)], num_points=[51, 31])  # Warns
```

**MFGProblem API**:
```python
problem = MFGProblem(geometry=grid, T=1.0, Nt=100)
problem.Nt         # 100 intervals
problem.Nt_points  # 101 points
```

**Common Mistake**:
```python
# ❌ WRONG: Using Nx to mean number of points
Nx = 51  # 51 points
dx = (xmax - xmin) / (Nx - 1)  # Incorrect!

# ✅ CORRECT: Nx means intervals, Nx_points = Nx + 1
Nx = 50  # 50 intervals → 51 points
dx = (xmax - xmin) / Nx  # Correct!
Nx_points = Nx + 1  # 51 points
```

**Why This Matters**:
- **Consistency**: `Nx` and `Nt` have same semantics (intervals)
- **Clarity**: `Nx_points` explicitly means grid points
- **Grid spacing**: `dx = L / Nx` (NOT `L / (Nx-1)`)
- **Arrays**: Solution arrays have shape `(Nt_points, Nx_points[0], Nx_points[1], ...)`
- **Interoperability**: All MFG_PDE solvers assume this convention

### Spatial Discretization

| Parameter | Type | Meaning | Math | Example |
|-----------|------|---------|------|---------|
| `Nx` | **list[int]** | Number of **intervals** per dimension | N | `[50, 30]` |
| `Nx_points` | **list[int]** | Number of **grid points** per dimension | N+1 | `[51, 31]` |
| `xmin` | **list[float]** | Domain lower bounds | x_min | `[0.0, 0.0]` |
| `xmax` | **list[float]** | Domain upper bounds | x_max | `[1.0, 1.0]` |
| `dx` | **list[float]** | Grid spacing per dimension | Δx | `[0.02, 0.033]` |
| `Lx` | **list[float]** | Domain length per dimension | L | `[1.0, 1.0]` |

**Grid Convention**:
- `Nx[i]` **intervals** → `Nx_points[i] = Nx[i]+1` **grid points** (for each dimension i)
- Grid spacing: `dx[i] = (xmax[i] - xmin[i]) / Nx[i]`
- Arrays include both boundaries
- **Use `Nx` for intervals, `Nx_points` for points** (never confuse them)

**Example (2D)**:
```python
# Nx = [Nx1, Nx2] = [50, 30]
problem = MFGProblem(
    Nx=[50, 30],             # Array notation (recommended)
    xmin=[0.0, 0.0],         # Array notation
    xmax=[1.0, 0.5],         # Array notation
    T=1.0,
    Nt=100
)
# Creates (Nx1+1)×(Nx2+1) = 51×31 = 1581 spatial grid points

# Alternative: High-level API
problem = MFGProblem(
    spatial_bounds=[(0.0, 1.0), (0.0, 0.5)],
    spatial_discretization=[50, 30],
    T=1.0,
    Nt=100
)
```

### Temporal Discretization

| Parameter | Type | Meaning | Math |
|-----------|------|---------|------|
| `Nt` | int | Number of time **intervals** | N_t |
| `Nt_points` | int | Number of time **points** (`Nt + 1`) | N_t + 1 |
| `T` | float | Terminal time | T |
| `dt` | float | Time step size | Δt |
| `tSpace` | ndarray | Time grid array | t_i |

**Grid Convention**:
- `Nt` **intervals** → `Nt_points = Nt + 1` **time points** (includes t=0 and t=T)
- Time step: `dt = T / Nt` (interval length)
- Time grid: `tSpace = [0, dt, 2*dt, ..., T]` with `len(tSpace) = Nt_points`

**Example**:
```python
problem = MFGProblem(geometry=grid, T=1.0, Nt=100)
problem.Nt         # 100 time intervals
problem.Nt_points  # 101 time points
problem.dt         # = 0.01 (interval size)
problem.tSpace     # np.linspace(0, 1.0, 101) - 101 time points
```

---

## Solver Parameters

### Picard Iteration

| Parameter | Type | Meaning | Math |
|-----------|------|---------|------|
| `damping_factor` | float | Picard iteration damping (0-1) | θ |
| `max_iterations` | int | Maximum Picard iterations | N_max |
| `tolerance` | float | Convergence tolerance | ε |

**Example**:
```python
from mfg_pde.solvers import FixedPointIterator

solver = FixedPointIterator(
    problem, hjb_solver, fp_solver,
    damping_factor=0.6,
    max_iterations=100,
    tolerance=1e-6
)
```

---

## Solver Method API Conventions (v0.11.0+)

### MFG Coupling Data Flow

The MFG system couples HJB and FP solvers through an iterative process. Understanding the data flow clarifies parameter naming:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         COUPLING ITERATION                              │
│                                                                         │
│  Initial: M_initial ─────────────────────────────────────────────────┐  │
│                                                                      │  │
│      ┌──────────────┐                      ┌──────────────┐          │  │
│      │              │     M_density        │              │          │  │
│      │  FP Solver   │ ──────────────────>  │  HJB Solver  │          │  │
│      │              │                      │              │          │  │
│      │  solve_fp_   │                      │  solve_hjb_  │          │  │
│      │  system()    │  <──────────────────  │  system()    │          │  │
│      │              │    drift_field       │              │          │  │
│      └──────────────┘    (from ∇U)         └──────────────┘          │  │
│            │                                      │                  │  │
│            v                                      v                  │  │
│        (output)                            U_coupling_prev           │  │
│                                            (for next iter)           │  │
│                                                                      │  │
│  Terminal: U_terminal ───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key Data Transfers**:
- **FP → HJB**: `M_density` (density field computed by FP solver)
- **HJB → FP**: `drift_field` (velocity field derived from ∇U via optimal control)
- **Coupling state**: `U_coupling_prev` (from previous outer iteration)

### HJB Solver API

```python
def solve_hjb_system(
    self,
    M_density: NDArray,              # Density from FP solver
    U_terminal: NDArray,             # Terminal condition u(T,x)
    U_coupling_prev: NDArray,        # Previous coupling iteration estimate
    diffusion_field: float | NDArray | None = None,
) -> NDArray:
    """Solve HJB equation backward in time.

    Returns:
        U: Value function u(t,x) of shape (Nt+1, *spatial_shape)
    """
```

| Parameter | Shape | Source | Description |
|-----------|-------|--------|-------------|
| `M_density` | `(Nt+1, *spatial)` | FP solver output | Density distribution m(t,x) |
| `U_terminal` | `(*spatial,)` | Problem definition | Terminal cost g(x) |
| `U_coupling_prev` | `(Nt+1, *spatial)` | Previous iteration | Prior estimate for damping/warm-start |
| `diffusion_field` | scalar or `(*spatial,)` | Problem/coupling | σ² diffusion coefficient |

### FP Solver API

```python
def solve_fp_system(
    self,
    M_initial: NDArray,              # Initial density m(0,x)
    drift_field: NDArray | None,     # Velocity/drift field v(t,x) from HJB
    diffusion_field: float | NDArray | None = None,
    show_progress: bool = True,
) -> NDArray:
    """Solve Fokker-Planck equation forward in time.

    Returns:
        M: Density function m(t,x) of shape (Nt+1, *spatial_shape)
    """
```

| Parameter | Shape | Source | Description |
|-----------|-------|--------|-------------|
| `M_initial` | `(*spatial,)` | Problem definition | Initial density m₀(x) |
| `drift_field` | `(Nt+1, *spatial)` or `None` | HJB coupling module | Velocity/advection v = -∇_p H(x, ∇u, m) or None for pure diffusion |
| `diffusion_field` | scalar, `(*spatial,)`, or callable | Problem/coupling | σ² diffusion coefficient |
| `show_progress` | `bool` | User preference | Whether to display progress bar |

### Design Principles

**1. Source-based naming over role-based naming**

Prefer names that indicate where data comes from, not what it's used for:

```python
# ✅ Good: Source-based
M_density          # Density (data type)
U_terminal         # Value function at terminal time
drift_field        # Velocity/advection field

# ❌ Avoid: Role-based with redundant suffixes
M_density_evolution_from_FP   # Redundant: density always from FP
U_final_condition_at_T        # Redundant: terminal implies at T
```

**2. Suffixes only when disambiguation is needed**

Add suffixes only when parameter names would otherwise be ambiguous:

```python
# ✅ Needed: Distinguishes coupling iteration from solver internal iteration
U_coupling_prev    # Previous outer (Picard) iteration
M_coupling_prev    # Previous outer (Picard) iteration

# ❌ Unnecessary: Already unambiguous
U_from_prev_picard    # "coupling" is clearer than "picard"
U_prev_picard_iter    # Over-specified
```

**3. Separation of concerns**

FP solver receives `drift_field` directly, not `U`:

```python
# ✅ Good: FP solver is independent of HJB details
fp_solver.solve_fp_system(
    M_initial=m0,
    drift_field=v,  # Pre-computed by coupling layer (also called advection)
    ...
)

# ❌ Avoid: FP solver computing velocity from U
fp_solver.solve_fp_system(
    M_initial=m0,
    U_from_hjb=U,       # FP shouldn't need to know about HJB
    ...
)
```

**Rationale**: The coupling layer (not FP solver) handles the physics of computing v from ∇U:
- `v = -∇_p H(x, ∇u, m)` depends on Hamiltonian form
- Different MFG formulations have different velocity formulas
- FP solver only needs to advect density by given velocity field

### Iteration Level Terminology

| Level | Name | Scope | Example Parameter |
|-------|------|-------|-------------------|
| Outer | Coupling iteration | HJB ↔ FP cycle | `U_coupling_prev`, `M_coupling_prev` |
| Middle | Newton iteration | Within single PDE solve | `U_newton_prev` (internal) |
| Inner | Timestep | Single time integration step | `U_n`, `U_n_plus_1` (internal) |

**Usage**:
- Public API uses `_coupling_prev` suffix for outer iteration state
- Internal solver variables use `_newton_prev` or time indices (`_n`, `_n_plus_1`)
- Avoid exposing internal iteration state in public API

### Backward Compatibility

Existing solvers may use deprecated parameter names. These emit `DeprecationWarning`:

| Deprecated | New Name | Migration |
|------------|----------|-----------|
| `M_density_evolution_from_FP` | `M_density` | v0.12.0+ |
| `U_final_condition_at_T` | `U_terminal` | v0.12.0+ |
| `U_from_prev_picard` | `U_coupling_prev` | v0.12.0+ |
| `M_density_evolution` | `M_density` | v0.12.0+ |
| `U_final_condition` | `U_terminal` | v0.12.0+ |
| `m_initial_condition` | `M_initial` | v0.12.0+ |

**Timeline**:
- **v0.11.0**: New names added, old names deprecated with warnings
- **v0.12.0+**: Continued deprecation warnings
- **v1.0.0**: Deprecated names removed

### AMR (Adaptive Mesh Refinement)

**Status**: AMR implementation was removed in v0.16.2. Only a stub API remains for future library integration.

For adaptive mesh refinement, use external libraries:
- pyAMReX: Block-structured AMR, GPU support
- Clawpack/AMRClaw: Hyperbolic PDEs
- pyAMG: Mesh adaptation for complex geometries

---

## Physical Parameters

| Parameter | Type | Meaning | Math Symbol |
|-----------|------|---------|-------------|
| `diffusion` | float | **Canonical** diffusion coefficient | σ |
| `sigma` | float | **Deprecated** - use `diffusion` instead | σ |
| `coupling_coefficient` | float | MFG coupling strength | λ |
| `terminal_cost` | callable | Terminal cost function | g(x,m) |
| `running_cost` | callable | Running cost function | f(x,m) |
| `hamiltonian` | callable | Hamiltonian function | H(x,p,m) |

### ⚠️ CRITICAL: Diffusion Coefficient Naming (v0.17.1+)

**Canonical parameter**: `diffusion` (NOT `sigma`)

**Mathematical definition**: `diffusion = σ` (the diffusion coefficient itself, NOT σ²)

**In code formulas**:
```python
# Formula: g = -σ²/2 · ∂ln(m)/∂n
# Code:
g = -(diffusion**2) / 2 * grad_ln_m

# Where:
#   diffusion = σ (the diffusion coefficient)
#   diffusion**2 = σ² (appears in formulas as σ²/2)
```

**Backward compatibility**:
- Constructors: Accept `sigma` with `DeprecationWarning`, map to `diffusion`
- State dicts: Look for `diffusion` first, fall back to `sigma`
- Timeline: `sigma` will be removed in v1.0.0

**Example**:
```python
# ✅ Modern (recommended)
provider = AdjointConsistentProvider(side="left", diffusion=0.2)
solver = HJBFDMSolver(problem, diffusion=0.2)

# ⚠️ Deprecated (still works with warning)
provider = AdjointConsistentProvider(side="left", sigma=0.2)  # Warns
```

**Common confusion**: `diffusion = σ`, NOT `σ²`. When you see `-σ²/2` in formulas, write `-(diffusion**2)/2` in code.

---

## Solution Arrays

**Convention**: Uppercase for solution arrays, lowercase for parameters

| Array Name | Shape | Meaning | Math |
|------------|-------|---------|------|
| `U` | (Nt+1, Nx1+1, Nx2+1, ...) | Value function | u(t,x) |
| `M` | (Nt+1, Nx1+1, Nx2+1, ...) | Density function | m(t,x) |
| `grad_U` | (Nt+1, Nx1+1, Nx2+1, ..., d) | Gradient of value function | ∇u |

**Example**:
```python
def solve(self) -> tuple[np.ndarray, np.ndarray]:
    """Solve MFG system.

    Returns:
        U: Value function u(t,x) of shape (Nt+1, Nx1+1, Nx2+1)
        M: Density function m(t,x) of shape (Nt+1, Nx1+1, Nx2+1)
    """
```

---

## Derivative Tensor Standard

### ⚠️ CRITICAL: Unified Derivative Representation (v0.17.0+)

MFG_PDE uses **tensor-based derivative representation** as the canonical standard. Derivatives of order p in dimension d are stored as tensors of shape `(d,) * p`.

### DerivativeTensors Class

```python
from mfg_pde.core import DerivativeTensors

# Create from arrays
grad = np.array([0.5, 0.3])                        # shape (2,)
hess = np.array([[0.1, 0.05], [0.05, 0.2]])        # shape (2, 2)
derivs = DerivativeTensors.from_arrays(grad=grad, hess=hess)

# Access by order
derivs[1]              # Gradient tensor, shape (d,)
derivs[2]              # Hessian tensor, shape (d, d)
derivs[3]              # Third-order tensor, shape (d, d, d)

# Access by name
derivs.grad            # Same as derivs[1]
derivs.hess            # Same as derivs[2]
derivs.laplacian       # tr(∇²u) = Σᵢ ∂²u/∂xᵢ²
derivs.grad_norm_squared  # |∇u|² = Σᵢ (∂u/∂xᵢ)²

# Access individual components
derivs.grad[0]         # ∂u/∂x₀
derivs.hess[0, 1]      # ∂²u/∂x₀∂x₁
derivs[3][1, 0, 0]     # ∂³u/∂x₁∂x₀∂x₀
```

### Tensor Shape Convention

| Order p | Name | Shape | Access | Example |
|---------|------|-------|--------|---------|
| 0 | Value | scalar | `derivs.value` | `u` |
| 1 | Gradient | `(d,)` | `derivs.grad[i]` | `∂u/∂xᵢ` |
| 2 | Hessian | `(d, d)` | `derivs.hess[i, j]` | `∂²u/∂xᵢ∂xⱼ` |
| 3 | Third | `(d, d, d)` | `derivs[3][i, j, k]` | `∂³u/∂xᵢ∂xⱼ∂xₖ` |
| p | p-th order | `(d,) * p` | `derivs[p][i₁, ..., iₚ]` | `∂ᵖu/∂x_{i₁}...∂x_{iₚ}` |

**Mathematical correspondence:**
$$\text{derivs}[p][i_1, i_2, \ldots, i_p] = \frac{\partial^p u}{\partial x_{i_1} \partial x_{i_2} \cdots \partial x_{i_p}}$$

### Examples by Dimension

**1D** (`d=1`):
```python
derivs = DerivativeTensors.from_arrays(
    grad=np.array([u_x]),           # shape (1,)
    hess=np.array([[u_xx]]),        # shape (1, 1)
)
derivs.grad[0]      # u_x
derivs.hess[0, 0]   # u_xx
derivs.laplacian    # u_xx
```

**2D** (`d=2`):
```python
derivs = DerivativeTensors.from_arrays(
    grad=np.array([u_x, u_y]),
    hess=np.array([[u_xx, u_xy],
                   [u_xy, u_yy]]),
)
derivs.grad[0]      # u_x
derivs.grad[1]      # u_y
derivs.hess[0, 0]   # u_xx
derivs.hess[0, 1]   # u_xy
derivs.hess[1, 1]   # u_yy
derivs.laplacian    # u_xx + u_yy
```

**3D** (`d=3`):
```python
derivs = DerivativeTensors.from_arrays(
    grad=np.array([u_x, u_y, u_z]),
    hess=np.array([[u_xx, u_xy, u_xz],
                   [u_xy, u_yy, u_yz],
                   [u_xz, u_yz, u_zz]]),
)
derivs.grad[2]      # u_z
derivs.hess[0, 2]   # u_xz
derivs.laplacian    # u_xx + u_yy + u_zz
```

### Benefits

1. **Standard NumPy**: Native tensor operations, vectorizable
2. **Order-agnostic**: Arbitrary derivative order p supported
3. **Dimension-agnostic**: Works for any spatial dimension d
4. **Efficient**: Array operations, no dict lookup overhead
5. **Type-safe**: IDE knows `derivs.grad` is `ndarray`
6. **Intuitive indexing**: `hess[i, j]` not `derivs[(i==1, j==1)]`

### Hamiltonian Function Signature

**Standard signature (v0.17.0+)**:

```python
from mfg_pde.core import DerivativeTensors

def hamiltonian(
    x_idx: int | tuple[int, ...],
    m_at_x: float,
    derivs: DerivativeTensors,
    **kwargs
) -> float:
    """Compute Hamiltonian H(x, ∇u, m).

    Args:
        x_idx: Grid point index
        m_at_x: Density m(x) at this point
        derivs: DerivativeTensors containing grad, hess, etc.

    Returns:
        Hamiltonian value H(x, p, m)
    """
```

**Standard implementation** (dimension-agnostic):

```python
def hamiltonian(x_idx, m_at_x, derivs: DerivativeTensors, **kwargs) -> float:
    """Quadratic Hamiltonian H = (1/2)|∇u|²."""
    return 0.5 * derivs.grad_norm_squared

# Or explicitly:
def hamiltonian(x_idx, m_at_x, derivs: DerivativeTensors, **kwargs) -> float:
    """Quadratic Hamiltonian H = (1/2)|∇u|²."""
    return 0.5 * np.sum(derivs.grad ** 2)
```

**With Hessian (viscosity solution)**:

```python
def hamiltonian_with_viscosity(x_idx, m_at_x, derivs: DerivativeTensors, sigma: float = 0.1) -> float:
    """H = (1/2)|∇u|² with viscosity term."""
    H = 0.5 * derivs.grad_norm_squared
    if derivs.hess is not None:
        H -= 0.5 * sigma**2 * derivs.laplacian
    return H
```

### Solver Usage

**Current State (v0.17.0+)**: All solvers use `DerivativeTensors`.

| Solver | Status |
|--------|--------|
| FDM | ✅ DerivativeTensors |
| GFDM | ✅ DerivativeTensors |
| WENO | ✅ DerivativeTensors |
| Semi-Lagrangian | ✅ DerivativeTensors |
| Network/DGM | ⚠️ AD-based, may differ |

### Creating DerivativeTensors in Solvers

```python
# Common case: gradient only
derivs = DerivativeTensors.from_gradient(grad_array)

# With Hessian
derivs = DerivativeTensors.from_arrays(grad=grad, hess=hess)

# Zero-initialized
derivs = DerivativeTensors.zeros(dimension=2, max_order=2)
```

### Migration from Legacy Format

**One-time conversion** from legacy `dict[tuple[int,...], float]`:

```python
from mfg_pde.core import from_multi_index_dict, to_multi_index_dict

# Legacy format
old_format = {(1, 0): 0.5, (0, 1): 0.3, (2, 0): 0.1, (1, 1): 0.05, (0, 2): 0.2}

# Convert to new format
derivs = from_multi_index_dict(old_format)
derivs.grad   # array([0.5, 0.3])
derivs.hess   # array([[0.1, 0.05], [0.05, 0.2]])

# Convert back if needed (for compatibility)
old = to_multi_index_dict(derivs)
```

### Reference Implementation

See `mfg_pde/core/derivatives.py` for full implementation.

---

## Geometry Parameters (v0.16.0+)

### GeometryProtocol Standard

All geometry classes implement:

| Property/Method | Type | Meaning |
|-----------------|------|---------|
| `dimension` | int | Spatial dimension |
| `geometry_type` | GeometryType | Type of geometry (enum) |
| `num_spatial_points` | int | Total spatial points |
| `get_spatial_grid()` | np.ndarray | Spatial grid array |
| `get_problem_config()` | dict | Configuration for MFGProblem |

### TensorProductGrid (v0.16.0+)

TensorProductGrid provides additional properties:

| Property | Type | Meaning |
|----------|------|---------|
| `Nx` | list[int] | Number of intervals per dimension |
| `Nx_points` | list[int] | Number of grid points per dimension (`Nx + 1`) |
| `num_points` | list[int] | **DEPRECATED** - use `Nx_points` instead |

**Example**:
```python
from mfg_pde import MFGProblem
from mfg_pde.geometry import TensorProductGrid

# Geometry-first API with Nx_points (v0.16.0+)
geometry = TensorProductGrid(
    dimension=2,
    bounds=[(0.0, 1.0), (0.0, 1.0)],
    Nx_points=[51, 51]  # Or Nx=[50, 50] for intervals
)
problem = MFGProblem(geometry=geometry, T=1.0, Nt=100, sigma=0.1)

# Geometry provides configuration
assert geometry.dimension == 2
assert geometry.Nx == [50, 50]        # intervals
assert geometry.Nx_points == [51, 51]  # points
assert geometry.num_spatial_points == 51 * 51

# MFGProblem provides time grid info
assert problem.Nt == 100
assert problem.Nt_points == 101
```

---

## Deprecated Names

**Do not use** these legacy names (found in old code):

| Old Name | New Name | Reason | Removal |
|----------|----------|--------|---------|
| `sigma` | `diffusion` | Canonical naming: `diffusion = σ` | v1.0.0 |
| `num_points` | `Nx_points` | Unclear (which N?) | v1.0.0 |
| `thetaUM` | `damping_factor` | Unclear acronym | Legacy |
| `Niter_max` | `max_iterations` | Inconsistent capitalization | Legacy |
| `coefCT` | `coupling_coefficient` | Unclear acronym | Legacy |

**Note**: Lowercase `dx`, `dt` are standard in mathematical and scientific computing. Use these in algorithm implementations, but `Dx`, `Dt` may appear in legacy contexts.

---

## Accepted Specific Terminology

Some parameter names use specific mathematical or domain terminology and are **not** considered deprecated:

| Parameter Name | Usage Context | Reason for Acceptance |
|----------------|---------------|----------------------|
| `l2errBound` | Convergence criteria | Specific L² error bound (not generic tolerance) |
| `Dx`, `Dt` | Algorithm code | Valid mathematical symbols in specific contexts |

**Guideline**: When a parameter represents a specific mathematical concept (like L² error bound), use the established terminology rather than generic names like `tolerance`. Use generic names only when the parameter is truly generic.

---

## Code Style Examples

### Good - Clear Naming

```python
# Array notation for dimension-agnostic code
problem = MFGProblem(
    Nx=[50, 50],                      # 2D: array notation
    xmin=[0.0, 0.0],                  # Array notation
    xmax=[1.0, 1.0],                  # Array notation
    T=1.0,
    Nt=100,
    sigma=0.1
)

# Or use high-level API
problem = MFGProblem(
    spatial_bounds=[(0.0, 1.0), (0.0, 1.0)],
    spatial_discretization=[50, 50],
    T=1.0,
    Nt=100,
    sigma=0.1
)

solver = create_fast_solver(
    problem,
    method="semi_lagrangian",
    damping_factor=0.6,
    max_iterations=100,
    tolerance=1e-6
)

# Arrays use mathematical notation
U = solver.solve()  # Value function u(t,x)
M = problem.M       # Density m(t,x)
```

### Bad - Unclear Abbreviations or Deprecated Patterns

```python
# ❌ Don't use unclear abbreviations
prob = ExMFGProb(
    sb=[(0.0, 1.0), (0.0, 1.0)],
    sd=[50, 50],
    T=1.0,
    Nt=100,
    sig=0.1
)

slv = mk_slv(
    prob,
    meth="sl",
    thetaUM=0.6,
    NiterMax=100,
    l2err=1e-6
)

# ❌ Don't use deprecated scalar notation (1D)
problem = MFGProblem(
    Nx=100,           # Deprecated! Use Nx=[100]
    xmin=0.0,         # Deprecated! Use xmin=[0.0]
    xmax=1.0,         # Deprecated! Use xmax=[1.0]
    T=1.0,
    Nt=50
)
# Emits DeprecationWarning, will break in v1.0.0
```

---

## Mathematical Notation in Documentation

Use proper math notation in **docstrings and markdown**, not in code:

**Docstring Example**:
```python
def solve_hjb(self, u_init: np.ndarray) -> np.ndarray:
    """Solve Hamilton-Jacobi-Bellman equation.

    Solves: ∂u/∂t + H(x, ∇u, m) = 0 with terminal condition u(T,x) = g(x,m(T,x))

    Args:
        u_init: Initial value function u(0,x) of shape (Nx+1,)

    Returns:
        Value function u(t,x) of shape (Nt+1, Nx+1)
    """
```

**Markdown Example**:
```markdown
The value function $u(t,x)$ satisfies the HJB equation:

$$\\frac{\\partial u}{\\partial t} + H(x, \\nabla u, m) = 0$$

with terminal condition $u(T,x) = g(x, m(T,x))$.
```

---

## When to Use Which Convention

### Use Mathematical Symbols (`Nx`, `Dt`, `sigma`)

✅ **Spatial/temporal discretization** (array notation):
```python
# 1D problem - use arrays
problem = MFGProblem(Nx=[100], xmin=[-2.0], xmax=[2.0], Nt=100, T=1.0)

# 2D problem - Nx = [Nx1, Nx2]
problem = MFGProblem(Nx=[50, 30], xmin=[0.0, 0.0], xmax=[1.0, 1.0], Nt=100, T=1.0)

x = problem.xSpace  # All dimensions
grid_spacing = problem.dx  # [dx1, dx2] for 2D
time_step = problem.dt  # Δt
```

✅ **Physical parameters from equations**:
```python
def hamiltonian(x, p, m, sigma=0.1):
    """H(x,p,m) = σ²|p|²/2 - V(x)"""
    return 0.5 * sigma**2 * np.sum(p**2) - V(x)
```

✅ **Solution arrays**:
```python
U = solver.solve_hjb()  # u(t,x)
M = solver.solve_fp()   # m(t,x)
```

### Use Descriptive English (`damping_factor`, `num_spatial_points`)

✅ **Algorithm configuration**:
```python
solver = FixedPointIterator(
    problem,
    damping_factor=0.6,          # NOT thetaUM
    max_iterations=100,           # NOT Niter_max
    tolerance=1e-6                # NOT l2errBound
)
```

✅ **High-level abstractions**:
```python
geometry = TensorProductGrid(
    dimension=2,
    Nx_points=[51, 51],           # Points (or Nx=[50, 50] for intervals)
    bounds=[(0.0, 1.0), (0.0, 1.0)]
)
```

✅ **Function/class names**:
```python
class FixedPointIterator:        # NOT FPIterator
    def check_convergence(...):  # NOT chk_conv
```

---

## Consistency Checklist

When adding new parameters:

- ✅ Use mathematical symbols for direct algorithm implementation
- ✅ Use descriptive English for configuration/settings
- ✅ Use `snake_case` not `camelCase`
- ✅ Document mathematical symbol in docstring if using notation
- ✅ Provide type hints
- ✅ Use dimension-agnostic arrays (e.g., `Nx=[...]`) when possible
- ✅ Add to this document if introducing new convention

---

## References

- **DerivativeTensors**: `mfg_pde/core/derivatives.py` (DerivativeTensors, from_multi_index_dict, to_multi_index_dict)
- **GeometryProtocol**: `mfg_pde/geometry/protocol.py`
- **Tensor Grids**: `mfg_pde/geometry/grids/tensor_grid.py`
- **Config Classes**: `mfg_pde/config/solver_config.py` (NewtonConfig, PicardConfig)
- **Fixed Point Iterator**: `mfg_pde/alg/numerical/coupling/fixed_point_iterator.py`
- **HJB Solvers**: `mfg_pde/alg/numerical/hjb_solvers/` (hjb_fdm.py, hjb_semi_lagrangian.py, hjb_weno.py, hjb_gfdm.py, hjb_network.py)
- **FP Solvers**: `mfg_pde/alg/numerical/fp_solvers/` (fp_fdm.py, fp_particle.py, fp_network.py)
- **Legacy Gradient Notation** (deprecated): `mfg_pde/compat/gradient_notation.py`

---

**Authoritative Source**: This document reflects actual codebase standards as of v0.17.0
