# MFG_PDE Naming Conventions

**Last Updated**: 2025-11-05
**Status**: Current reference document
**Related**: See `docs/gradient_notation_standard.md` for gradient notation

---

## Purpose

This document defines Python code naming conventions for MFG_PDE based on actual codebase standards, not aspirational goals. Use this as the authoritative reference for parameter and variable naming.

---

## Core Principles

1. **Descriptive English names** for clarity: `damping_factor` not `thetaUM`
2. **Consistency** across the codebase
3. **Mathematical notation in documentation**, not code variable names
4. **No camelCase** - use `snake_case` for all Python identifiers

---

## Solver Parameters

### Picard Iteration

| Parameter | Type | Meaning |
|-----------|------|---------|
| `damping_factor` | float | Picard iteration damping (0-1), math: θ |
| `max_iterations` | int | Maximum Picard iterations |
| `tolerance` | float | Convergence tolerance, math: ε |

**Example**:
```python
solver = FixedPointIterator(
    problem, hjb_solver, fp_solver,
    damping_factor=0.6,
    max_iterations=100,
    tolerance=1e-6
)
```

---

## Discretization Parameters

### Time Discretization

| Parameter | Type | Meaning |
|-----------|------|---------|
| `num_time_steps` | int | Number of time steps (Nt), creates Nt+1 time points |
| `time_step` | float | Temporal step size Δt |
| `T` | float | Terminal time |

### Spatial Discretization (1D)

| Parameter | Type | Meaning |
|-----------|------|---------|
| `num_intervals_x` | int | Number of spatial intervals (legacy: `Nx`) |
| `grid_spacing` | float | Spatial grid spacing Δx or h |
| `xmin`, `xmax` | float | Domain boundaries |

**Grid Convention**:
- `Nx` intervals → `Nx+1` grid points
- Arrays have shape `(Nx+1,)` including both boundaries
- Grid spacing: `Δx = (xmax - xmin) / Nx`

### Spatial Discretization (nD)

| Parameter | Type | Meaning |
|-----------|------|---------|
| `dimension` | int | Spatial dimension (1, 2, 3, or arbitrary) |
| `num_spatial_points` | int | Total number of spatial points |
| `spatial_shape` | tuple | Shape of spatial grid (e.g., (Nx+1, Ny+1)) |

---

## AMR (Adaptive Mesh Refinement) Parameters

**Canonical definition**: See `mfg_pde/geometry/amr_quadtree_2d.py:AMRRefinementCriteria`

### Required Parameters

| Parameter | Type | Default | Meaning |
|-----------|------|---------|---------|
| `max_refinement_levels` | int | 5 | Maximum number of refinement levels |
| `gradient_threshold` | float | 0.1 | Threshold for gradient-based refinement |
| `coarsening_threshold` | float | 0.1 | Threshold for mesh coarsening |
| `error_threshold` | float | 1e-4 | Maximum allowed solution error per cell |
| `min_cell_size` | float | 1e-6 | Minimum allowed cell size |

### Optional Parameters

| Parameter | Type | Default | Meaning |
|-----------|------|---------|---------|
| `solution_variance_threshold` | float | 1e-5 | Variance threshold for refinement |
| `density_gradient_threshold` | float | 0.05 | Density gradient threshold |
| `adaptive_error_scaling` | bool | True | Enable adaptive error scaling |

**Example**:
```python
from mfg_pde.geometry.amr_quadtree_2d import AMRRefinementCriteria
from mfg_pde.geometry.amr_1d import OneDimensionalAMRMesh

criteria = AMRRefinementCriteria(
    max_refinement_levels=3,
    gradient_threshold=0.5,
    coarsening_threshold=0.25,
    min_cell_size=0.001
)

amr_mesh = OneDimensionalAMRMesh(
    domain_1d=domain,
    initial_num_intervals=20,
    refinement_criteria=criteria
)
```

**Note**: Common mistakes found in outdated docs:
- ❌ `max_level` or `max_levels` → ✅ `max_refinement_levels`
- ❌ `refinement_threshold` → ✅ `gradient_threshold`

---

## Physical Parameters

### Mean Field Game System

| Parameter | Type | Meaning | Math Symbol |
|-----------|------|---------|-------------|
| `sigma` | float | Diffusion coefficient | σ |
| `coupling_coefficient` | float | Coupling strength between agents | λ |

### Problem-Specific

| Parameter | Type | Meaning |
|-----------|------|---------|
| `terminal_cost` | callable | Terminal cost function g(x,m) |
| `running_cost` | callable | Running cost function f(x,m) |
| `hamiltonian` | callable | Hamiltonian function H(x,p,m) |

---

## Solution Arrays

**Convention**: Uppercase for solution arrays, lowercase for parameters

| Array Name | Shape | Meaning |
|------------|-------|---------|
| `U` | (Nt+1, Nx+1, ...) | Value function u(t,x) |
| `M` | (Nt+1, Nx+1, ...) | Density function m(t,x) |
| `grad_U` | (Nt+1, Nx+1, ..., d) | Gradient of value function ∇u |

**Example**:
```python
def solve(self) -> tuple[np.ndarray, np.ndarray]:
    """Solve MFG system.

    Returns:
        U: Value function u(t,x) of shape (Nt+1, Nx+1)
        M: Density function m(t,x) of shape (Nt+1, Nx+1)
    """
```

---

## Geometry Parameters

### GeometryProtocol Standard (v0.10.0+)

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
from mfg_pde.geometry.domain_1d import Domain1D

# Geometry-first API (v0.10.0+)
geometry = Domain1D(xmin=0.0, xmax=1.0, boundary_conditions=bc)
problem = ExampleMFGProblem(geometry=geometry, T=1.0, Nt=100, sigma=0.1)

# Geometry provides configuration
assert geometry.dimension == 1
assert geometry.num_spatial_points == geometry.get_spatial_grid().shape[0]
```

---

## Deprecated Names

**Do not use** these legacy names (found in old code):

| Old Name | New Name | Reason |
|----------|----------|--------|
| `thetaUM` | `damping_factor` | Unclear acronym |
| `Niter_max` | `max_iterations` | Inconsistent capitalization |
| `l2errBound` | `tolerance` | Unclear abbreviation |
| `Dx`, `dx` | `grid_spacing` | Ambiguous (also derivative) |
| `Dt`, `dt` | `time_step` | Ambiguous |
| `coefCT` | `coupling_coefficient` | Unclear acronym |

**Transition**: Legacy names may still exist in older code but should be replaced during refactoring.

---

## Code Style Examples

### Good - Descriptive Names
```python
problem = ExampleMFGProblem(
    geometry=geometry,
    T=1.0,
    num_time_steps=100,
    sigma=0.1
)

solver = create_fast_solver(
    problem,
    method="semi_lagrangian",
    damping_factor=0.6,
    max_iterations=100,
    tolerance=1e-6
)
```

### Bad - Unclear Abbreviations
```python
# Don't do this
prob = ExMFGProb(
    geom=g,
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

$$\frac{\partial u}{\partial t} + H(x, \nabla u, m) = 0$$

with terminal condition $u(T,x) = g(x, m(T,x))$.
```

---

## Consistency Checklist

When adding new parameters:

- ✅ Use descriptive English names
- ✅ Use `snake_case` not `camelCase`
- ✅ Document mathematical symbol in docstring
- ✅ Provide type hints
- ✅ Use standard names from this document
- ✅ Add to this document if introducing new convention

---

## References

- **Gradient Notation**: `docs/gradient_notation_standard.md`
- **GeometryProtocol**: `mfg_pde/geometry/geometry_protocol.py`
- **AMR Parameters**: `mfg_pde/geometry/amr_quadtree_2d.py` (AMRRefinementCriteria)

---

**Authoritative Source**: This document reflects actual codebase standards as of v0.10.1
