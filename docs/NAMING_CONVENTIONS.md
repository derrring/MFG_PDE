# MFG_PDE Naming Conventions

**Last Updated**: 2025-11-05
**Status**: Current reference document
**Related**: See sections on gradient notation for derivative indexing

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
- `tolerance` (NOT `l2errBound`) - Convergence tolerance
- `num_spatial_points` - Total spatial grid points
- `spatial_shape` - Shape tuple for nD arrays
- `coupling_coefficient` (NOT `coefCT`) - MFG coupling strength

**Rationale**: Configuration parameters are not formula symbols. `thetaUM` is "magic number" jargon with no clear meaning. `damping_factor` clearly describes its purpose.

---

## Dimension-Agnostic Design (v0.10.0+)

### Key Concept: `Nx` is an Array

In v0.10.0+, `Nx` represents spatial discretization for **arbitrary dimensions**:

- **1D**: `Nx = [Nx1]` → `Nx1` intervals, `Nx1+1` grid points. Example: `Nx = [50]`
- **2D**: `Nx = [Nx1, Nx2]` → `Nx1×Nx2` intervals, `(Nx1+1)×(Nx2+1)` grid points. Example: `Nx = [50, 30]`
- **3D**: `Nx = [Nx1, Nx2, Nx3]` → `Nx1×Nx2×Nx3` intervals. Example: `Nx = [20, 20, 20]`
- **nD**: `Nx = [Nx1, Nx2, ..., Nxd]` where `d` is the spatial dimension

**There is NO `Ny`, `Nz`**. All dimensions use the single array `Nx`.

### Legacy 1D Scalar Compatibility

For backward compatibility, 1D problems accept scalar `Nx`:

```python
# Modern (dimension-agnostic)
problem = MFGProblem(Nx=[50], ...)  # Recommended

# Legacy (1D only)
problem = MFGProblem(Nx=50, ...)    # Still works, normalized to [50]
```

Internally, scalars are normalized to 1-element arrays.

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
| `Nx` | int or list[int] | Number of intervals per dimension: `[Nx1, Nx2, ..., Nxd]` | N | `[50, 30]` |
| `xmin` | float or list[float] | Domain lower bounds | x_min | `[0.0, 0.0]` |
| `xmax` | float or list[float] | Domain upper bounds | x_max | `[1.0, 1.0]` |
| `dx` | float or list[float] | Grid spacing per dimension: `[dx1, dx2, ..., dxd]` | Δx | `[0.02, 0.033]` |
| `Lx` | float or list[float] | Domain length per dimension: `[Lx1, Lx2, ..., Lxd]` | L | `[1.0, 1.0]` |

**Grid Convention**:
- `Nxi` intervals → `Nxi+1` grid points (for each dimension i)
- Grid spacing: `dxi = (xmax[i] - xmin[i]) / Nxi`
- Arrays include both boundaries

**Example (2D)**:
```python
# Nx = [Nx1, Nx2] = [50, 30]
problem = MFGProblem(
    spatial_bounds=[(0.0, 1.0), (0.0, 0.5)],
    spatial_discretization=[50, 30],  # Nx1=50, Nx2=30 intervals
    T=1.0,
    Nt=100
)
# Creates (Nx1+1)×(Nx2+1) = 51×31 = 1581 spatial grid points
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
| `l2errBound` | `tolerance` | Unclear abbreviation |
| `Dx` (uppercase) | `dx` (lowercase) | Consistency with standard notation |
| `Dt` (uppercase) | `dt` (lowercase) | Consistency with standard notation |
| `coefCT` | `coupling_coefficient` | Unclear acronym |

**Note**: Lowercase `dx`, `dt` are standard in mathematical and scientific computing.

---

## Code Style Examples

### Good - Clear Naming

```python
# Configuration uses descriptive English
problem = ExampleMFGProblem(
    spatial_bounds=[(0.0, 1.0), (0.0, 1.0)],
    spatial_discretization=[50, 50],  # Nx in 2D
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

### Bad - Unclear Abbreviations

```python
# Don't do this
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

✅ **Spatial/temporal discretization**:
```python
# Nx = [Nx1, Nx2] for 2D
problem = MFGProblem(Nx=[50, 30], Nt=100, T=1.0)
x = problem.xSpace  # All dimensions
grid_spacing = problem.dx  # [dx1, dx2]
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
