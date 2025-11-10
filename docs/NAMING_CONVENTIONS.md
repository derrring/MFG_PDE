# MFG_PDE Naming Conventions

**Last Updated**: 2025-11-10
**Status**: Current reference document
**Related**: See sections on gradient notation for derivative indexing and array-based notation standard

---

## Purpose

This document defines Python code naming conventions for MFG_PDE based on actual codebase standards, not aspirational goals. Use this as the authoritative reference for parameter and variable naming.

---

## Core Principles

### 1. Mathematical Symbols for Direct Algorithm Implementation

**When to use**: When code directly implements a textbook algorithm or specific mathematical formula.

**Examples**:
- `Nx`: Number of spatial intervals (not `num_intervals_x`)
- `Nt`: Number of time intervals
- `dx`: Spatial grid spacing Δx
- `dt`: Time step size Δt
- `sigma`: Diffusion coefficient σ
- `T`: Terminal time
- `xSpace`: Spatial grid array (all dimensions)
- `tSpace`: Temporal grid array

**Rationale**: Direct correspondence to mathematical notation makes algorithm validation straightforward.

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

### Spatial Discretization

| Parameter | Type | Meaning | Math | Example |
|-----------|------|---------|------|---------|
| `Nx` | **list[int]** | Number of intervals per dimension: `[Nx1, Nx2, ..., Nxd]` | N | `[50, 30]` |
| `xmin` | **list[float]** | Domain lower bounds | x_min | `[0.0, 0.0]` |
| `xmax` | **list[float]** | Domain upper bounds | x_max | `[1.0, 1.0]` |
| `dx` | **list[float]** | Grid spacing per dimension: `[dx1, dx2, ..., dxd]` | Δx | `[0.02, 0.033]` |
| `Lx` | **list[float]** | Domain length per dimension: `[Lx1, Lx2, ..., Lxd]` | L | `[1.0, 1.0]` |

**Grid Convention**:
- `Nxi` intervals → `Nxi+1` grid points (for each dimension i)
- Grid spacing: `dxi = (xmax[i] - xmin[i]) / Nxi`
- Arrays include both boundaries

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
| `Nt` | int | Number of time intervals | N_t |
| `T` | float | Terminal time | T |
| `dt` | float | Time step size | Δt |
| `tSpace` | ndarray | Time grid array | t_i |

**Grid Convention**:
- `Nt` intervals → `Nt+1` time points
- Time step: `dt = T / Nt`
- Time grid: `tSpace = [0, dt, 2*dt, ..., T]`

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

### AMR (Adaptive Mesh Refinement)

**Canonical definition**: `mfg_pde/geometry/amr_quadtree_2d.py:AMRRefinementCriteria`

| Parameter | Type | Default | Meaning |
|-----------|------|---------|---------|
| `max_refinement_levels` | int | 5 | Maximum refinement levels |
| `gradient_threshold` | float | 0.1 | Gradient-based refinement threshold |
| `coarsening_threshold` | float | 0.1 | Mesh coarsening threshold |
| `error_threshold` | float | 1e-4 | Max solution error per cell |
| `min_cell_size` | float | 1e-6 | Minimum cell size |

**Common mistakes** (found in outdated docs):
- ❌ `max_level` or `max_levels` → ✅ `max_refinement_levels`
- ❌ `refinement_threshold` → ✅ `gradient_threshold`

---

## Physical Parameters

| Parameter | Type | Meaning | Math Symbol |
|-----------|------|---------|-------------|
| `sigma` | float | Diffusion coefficient | σ |
| `coupling_coefficient` | float | MFG coupling strength | λ |
| `terminal_cost` | callable | Terminal cost function | g(x,m) |
| `running_cost` | callable | Running cost function | f(x,m) |
| `hamiltonian` | callable | Hamiltonian function | H(x,p,m) |

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

## Gradient Notation Standard

### Tuple Multi-Index Notation

For a function u(x₁, x₂, ..., xₙ), derivatives are indexed by tuples (α₁, α₂, ..., αₙ) where αᵢ is the derivative order with respect to xᵢ.

**Examples**:

**1D**: `u(x1)`
- `derivs[(0,)] = u` - Function value
- `derivs[(1,)] = ∂u/∂x1` - First derivative
- `derivs[(2,)] = ∂²u/∂x1²` - Second derivative

**2D**: `u(x1, x2)`
- `derivs[(0, 0)] = u` - Function value
- `derivs[(1, 0)] = ∂u/∂x1` - Gradient x1-component
- `derivs[(0, 1)] = ∂u/∂x2` - Gradient x2-component
- `derivs[(2, 0)] = ∂²u/∂x1²` - Hessian x1x1
- `derivs[(0, 2)] = ∂²u/∂x2²` - Hessian x2x2
- `derivs[(1, 1)] = ∂²u/∂x1∂x2` - Mixed derivative

**3D**: `u(x1, x2, x3)`
- `derivs[(1, 0, 0)] = ∂u/∂x1`
- `derivs[(0, 1, 0)] = ∂u/∂x2`
- `derivs[(0, 0, 1)] = ∂u/∂x3`
- `derivs[(2, 0, 0)] = ∂²u/∂x1²`
- `derivs[(1, 1, 0)] = ∂²u/∂x1∂x2`

### Benefits

1. **Dimension-agnostic**: Works for 1D, 2D, 3D, nD without special cases
2. **Type-safe**: Tuples are hashable and immutable
3. **Mathematical clarity**: Direct correspondence to multi-index notation
4. **No ambiguity**: `(1,0)` is unambiguous, `"dx"` vs `"x"` is not
5. **Extensibility**: Higher-order derivatives naturally supported

### Implementation Example

```python
# GFDM solver (hjb_gfdm.py:1544-1556)
derivs = self.approximate_derivatives(u_current, i)

if d == 1:
    p = derivs.get((1,), 0.0)  # ∂u/∂x1
    laplacian = derivs.get((2,), 0.0)  # ∂²u/∂x1²
elif d == 2:
    p_x1 = derivs.get((1, 0), 0.0)  # ∂u/∂x1
    p_x2 = derivs.get((0, 1), 0.0)  # ∂u/∂x2
    p = np.array([p_x1, p_x2])
    laplacian = derivs.get((2, 0), 0.0) + derivs.get((0, 2), 0.0)  # ∂²u/∂x1² + ∂²u/∂x2²
```

---

## Geometry Parameters (v0.10.0+)

### GeometryProtocol Standard

All geometry classes implement:

| Property/Method | Type | Meaning |
|-----------------|------|---------|
| `dimension` | int | Spatial dimension |
| `geometry_type` | GeometryType | Type of geometry (enum) |
| `num_spatial_points` | int | Total spatial points |
| `get_spatial_grid()` | np.ndarray | Spatial grid array |
| `get_problem_config()` | dict | Configuration for MFGProblem |

**Example**:
```python
from mfg_pde import ExampleMFGProblem
from mfg_pde.geometry.tensor_product_grid import TensorProductGrid

# Geometry-first API (v0.10.0+)
geometry = TensorProductGrid(
    dimension=2,
    bounds=[(0.0, 1.0), (0.0, 1.0)],
    num_points=[51, 51]
)
problem = ExampleMFGProblem(geometry=geometry, T=1.0, Nt=100, sigma=0.1)

# Geometry provides configuration
assert geometry.dimension == 2
assert geometry.num_spatial_points == 51 * 51
```

---

## Deprecated Names

**Do not use** these legacy names (found in old code):

| Old Name | New Name | Reason |
|----------|----------|--------|
| `thetaUM` | `damping_factor` | Unclear acronym |
| `Niter_max` | `max_iterations` | Inconsistent capitalization |
| `coefCT` | `coupling_coefficient` | Unclear acronym |

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
    num_points=[51, 51],          # NOT Nx (this is constructor param)
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

- **GeometryProtocol**: `mfg_pde/geometry/geometry_protocol.py`
- **AMR Parameters**: `mfg_pde/geometry/amr_quadtree_2d.py` (AMRRefinementCriteria)
- **Tensor Grids**: `mfg_pde/geometry/tensor_product_grid.py`
- **Config Classes**: `mfg_pde/config/solver_config.py` (NewtonConfig, PicardConfig)
- **Fixed Point Iterator**: `mfg_pde/alg/numerical/coupling/fixed_point_iterator.py`
- **HJB FDM Solver**: `mfg_pde/alg/numerical/hjb_solvers/hjb_fdm.py`

---

**Authoritative Source**: This document reflects actual codebase standards as of v0.10.1
