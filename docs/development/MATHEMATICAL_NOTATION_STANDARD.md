# Mathematical Notation Standard for MFG_PDE

**Version**: 1.0.0
**Date**: 2025-11-04
**Status**: Canonical reference for all code and documentation

---

## Philosophy

MFG_PDE adopts **mathematical notation conventions** from numerical PDE literature over Python naming conventions. This choice prioritizes:

1. **Direct correspondence** between code and equations in research papers
2. **Domain expert familiarity** with established notation ($N_x$, $\Delta t$, etc.)
3. **Algorithm readability** when implementing methods from textbooks
4. **Consistency with field standards** (Quarteroni, LeVeque, Trefethen)

**Target audience**: Graduate students and researchers in computational mathematics, numerical PDEs, and mean field games.

---

## Core Principles

### Principle 1: Dimension-Agnostic Subscripts

Use subscript indexing `[i]` for arbitrary-dimensional variables, **not** separate variables for each dimension. **Always use arrays, even for 1D.**

```python
# ✅ CORRECT: Dimension-agnostic (works for 1D, 2D, 3D, nD)
Nx = [100, 80, 60]      # Grid points in dimensions 0, 1, 2
dx = [0.01, 0.02, 0.03]  # Spacing in dimensions 0, 1, 2
xmin = [-2.0, -1.0, 0.0]  # Domain bounds
xmax = [2.0, 1.0, 0.5]   # Domain bounds

# For 1D: Still use arrays with one element
Nx = [100]              # Grid points (1D)
dx = [0.01]             # Spacing (1D)
xmin = [-2.0]           # Lower bound (1D)
xmax = [2.0]            # Upper bound (1D)

# ❌ WRONG: Dimension-specific (deprecated)
Nx, Ny, Nz = 100, 80, 60  # Don't use separate variables per dimension
dx, dy, dz = 0.01, 0.02, 0.03
xmin, ymin, zmin = -2.0, -1.0, 0.0

# ❌ WRONG: Scalar for 1D (deprecated)
Nx = 100                # Use [100] instead
xmin = -2.0             # Use [-2.0] instead
```

**Rationale**:
1. Algorithms work for arbitrary dimensions without code duplication
2. No type checking needed: `Nx` is always a list
3. Consistent iteration: `for i in range(len(Nx))` works for all dimensions
4. Subscript notation generalizes naturally: $N_{x_i}$, $\Delta x_i$, $x_{\min,i}$ for dimension $i \in \{0, 1, \ldots, \text{dimension}-1\}$

### Principle 2: Mathematical Over Descriptive Names

Use abbreviated mathematical notation over verbose English names.

```python
# ✅ CORRECT: Mathematical notation
Nt = 50              # Matches N_t in papers
dt = T / Nt          # Matches Δt in papers
sigma = 0.3          # Matches σ in papers

# ❌ WRONG: Verbose descriptive (deprecated)
num_time_steps = 50
time_step_size = final_time / num_time_steps
diffusion_coefficient = 0.3
```

**Exception**: Use descriptive names for uncommon or package-specific parameters where no standard notation exists.

### Principle 3: Uppercase for Arrays, Lowercase for Scalars

```python
# Solution arrays (uppercase)
U = np.zeros((Nt+1, Nx[0]+1, Nx[1]+1))  # Value function u(t,x,y)
M = np.zeros((Nt+1, Nx[0]+1, Nx[1]+1))  # Density m(t,x,y)

# Scalar parameters (lowercase)
sigma = 0.3
dt = 0.01
dimension = 2  # Spatial dimension
```

### Principle 4: Preserve LaTeX-Compatible Naming

Variable names should map naturally to LaTeX notation in documentation.

```python
# Code → LaTeX mapping
Nt        # N_t
dt        # \Delta t
Nx[i]     # N_{x_i}
dx[i]     # \Delta x_i
xmin[i]   # x_{\min,i}
sigma     # \sigma
```

---

## Standard Notation Table

### Grid Discretization

| Symbol | Code Name | LaTeX | Description | Domain | Type |
|:-------|:----------|:------|:------------|:-------|:-----|
| $d$ | `dimension` | `d` | Spatial dimension | $\mathbb{N}^+$ | scalar |
| $N_{x_i}$ | `Nx[i]` | `N_{x_i}` | Grid points in dimension $i$ | $\mathbb{N}^+$ | array |
| $N_t$ | `Nt` | `N_t` | Temporal grid points | $\mathbb{N}^+$ | scalar |
| $\Delta x_i$ | `dx[i]` | `\Delta x_i` | Spatial spacing in dimension $i$ | $\mathbb{R}^+$ | array |
| $\Delta t$ | `dt` | `\Delta t$ | Temporal spacing | $\mathbb{R}^+$ | scalar |
| $x_{\min,i}$ | `xmin[i]` | `x_{\min,i}` | Lower bound in dimension $i$ | $\mathbb{R}$ | array |
| $x_{\max,i}$ | `xmax[i]` | `x_{\max,i}` | Upper bound in dimension $i$ | $\mathbb{R}$ | array |
| $T$ | `T` | `T` | Terminal time | $\mathbb{R}^+$ | scalar |

### Solution Variables

| Symbol | Code Name | LaTeX | Description | Domain |
|:-------|:----------|:------|:------------|:-------|
| $u(t,\mathbf{x})$ | `U` | `u(t,\mathbf{x})` | Value function (HJB solution) | $[0,T] \times \Omega \to \mathbb{R}$ |
| $m(t,\mathbf{x})$ | `M` | `m(t,\mathbf{x})` | Density function (FP solution) | $[0,T] \times \Omega \to \mathbb{R}^+$ |
| $\nabla u$ | `grad_U` | `\nabla u` | Spatial gradient of value function | $[0,T] \times \Omega \to \mathbb{R}^d$ |
| $\Delta u$ | `laplacian_U` | `\Delta u` | Laplacian of value function | $[0,T] \times \Omega \to \mathbb{R}$ |

### Physical Parameters

| Symbol | Code Name | LaTeX | Description | Domain | Units |
|:-------|:----------|:------|:------------|:-------|:------|
| $\sigma$ | `sigma` | `\sigma` | Diffusion coefficient | $\mathbb{R}^+$ | $\sqrt{\text{length}^2/\text{time}}$ |
| $\nu$ | `nu` | `\nu` | Viscosity coefficient | $\mathbb{R}^+$ | length²/time |
| $\lambda$ | `lam` | `\lambda` | Coupling coefficient | $\mathbb{R}$ | - |

**Note**: Use `lam` not `lambda` (Python keyword).

### Hamiltonian and Cost Functions

| Symbol | Code Name | LaTeX | Description |
|:-------|:----------|:------|:------------|
| $H(\mathbf{x}, \mathbf{p}, m)$ | `H` | `H(\mathbf{x}, \mathbf{p}, m)` | Hamiltonian function |
| $L(\mathbf{x}, \mathbf{v})$ | `L` | `L(\mathbf{x}, \mathbf{v})` | Lagrangian (running cost) |
| $g(\mathbf{x})$ | `g` | `g(\mathbf{x})` | Terminal cost |
| $f(\mathbf{x})$ | `f` | `f(\mathbf{x})` | Coupling cost |
| $V(\mathbf{x})$ | `V` | `V(\mathbf{x})` | Potential function |

### Numerical Parameters

| Symbol | Code Name | LaTeX | Description | Domain |
|:-------|:----------|:------|:------------|:-------|
| $\varepsilon_N$ | `newton_tol` | `\varepsilon_N` | Newton tolerance | $\mathbb{R}^+$ |
| $\varepsilon_P$ | `picard_tol` | `\varepsilon_P` | Picard tolerance | $\mathbb{R}^+$ |
| $N_{\max}$ | `max_iter` | `N_{\max}` | Maximum iterations | $\mathbb{N}^+$ |
| $\theta$ | `theta` | `\theta` | Damping/relaxation factor | $(0, 1]$ |

---

## Deprecated Names

The following English descriptive names are **deprecated** and should be migrated:

| Deprecated | Standard | Migration Status |
|:-----------|:---------|:-----------------|
| `d` | `dimension` | Use `dimension` (not ambiguous) |
| `N` | `Nx[i]` | Use `Nx` array (N too ambiguous) |
| `Nx` (scalar for 1D) | `Nx = [100]` | Always use array, even for 1D |
| `xmin`, `xmax` (scalars) | `xmin = [-2.0]` | Always use array, even for 1D |
| `dx` (scalar for 1D) | `dx = [0.01]` | Always use array, even for 1D |
| `num_spatial_points` | `Nx[i]` | Use `Nx` array |
| `num_time_steps` | `Nt` | Already standard |
| `spatial_step` | `dx[i]` | Use `dx` array |
| `time_step` | `dt` | Already standard |
| `domain_min`, `domain_max` | `xmin[i]`, `xmax[i]` | Use `xmin`, `xmax` arrays |
| `Ny`, `Nz` | `Nx[1]`, `Nx[2]` | Use indexed `Nx` array |
| `dy`, `dz` | `dx[1]`, `dx[2]` | Use indexed `dx` array |
| `Dx`, `Dy`, `Dz` | `dx[0]`, `dx[1]`, `dx[2]` | Inconsistent capitalization |

---

## Implementation Guidelines

### 1D Problems

```python
# Problem setup
dimension = 1
Nx = [100]                    # 100 spatial points
Nt = 50                       # 50 time steps
xmin = [-2.0]
xmax = [2.0]
T = 1.0

# Compute spacing
dx = [(xmax[i] - xmin[i]) / Nx[i] for i in range(dimension)]
dt = T / Nt

# Create grids
x = [np.linspace(xmin[i], xmax[i], Nx[i]+1) for i in range(dimension)]
t = np.linspace(0, T, Nt+1)

# Solution arrays
U = np.zeros((Nt+1, Nx[0]+1))  # u(t,x)
M = np.zeros((Nt+1, Nx[0]+1))  # m(t,x)
```

### 2D Problems

```python
# Problem setup
dimension = 2
Nx = [100, 80]                # 100×80 spatial grid
Nt = 50
xmin = [-2.0, -1.0]
xmax = [2.0, 1.0]
T = 1.0

# Compute spacing
dx = [(xmax[i] - xmin[i]) / Nx[i] for i in range(dimension)]
dt = T / Nt

# Create grids
x = [np.linspace(xmin[i], xmax[i], Nx[i]+1) for i in range(dimension)]
t = np.linspace(0, T, Nt+1)

# Solution arrays
U = np.zeros((Nt+1, Nx[0]+1, Nx[1]+1))  # u(t,x,y)
M = np.zeros((Nt+1, Nx[0]+1, Nx[1]+1))  # m(t,x,y)
```

### Arbitrary Dimensions

```python
def create_mfg_grid(dimension: int, Nx: list[int], xmin: list[float],
                     xmax: list[float], Nt: int, T: float):
    """
    Create dimension-agnostic MFG discretization grid.

    Args:
        dimension: Spatial dimension
        Nx: Grid points per dimension (length dimension)
        xmin: Lower bounds (length dimension)
        xmax: Upper bounds (length dimension)
        Nt: Temporal grid points
        T: Terminal time

    Returns:
        Spatial grids, temporal grid, spacing arrays
    """
    # Compute spacing
    dx = [(xmax[i] - xmin[i]) / Nx[i] for i in range(dimension)]
    dt = T / Nt

    # Create grids
    x = [np.linspace(xmin[i], xmax[i], Nx[i]+1) for i in range(dimension)]
    t = np.linspace(0, T, Nt+1)

    return x, t, dx, dt
```

---

## Documentation Standards

### Docstrings

Use both mathematical symbols and code names for clarity:

```python
def solve_hjb(self, U: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    Solve Hamilton-Jacobi-Bellman equation for value function.

    Solves: ∂u/∂t + H(x, ∇u, m) = σ²/2 Δu with terminal condition u(T,x) = g(x).

    Args:
        U: Value function u(t,x) array of shape (Nt+1, Nx[0]+1, ..., Nx[dimension-1]+1)
        M: Density function m(t,x) array of shape (Nt+1, Nx[0]+1, ..., Nx[dimension-1]+1)

    Returns:
        Updated value function U with same shape

    Notes:
        Grid spacing dx[i] = (xmax[i] - xmin[i]) / Nx[i] for i = 0, ..., dimension-1
        Time step dt = T / Nt
    """
```

### LaTeX in Documentation

Use proper LaTeX escaping in markdown:

```markdown
The HJB equation is:

$$\frac{\partial u}{\partial t} + H(\mathbf{x}, \nabla u, m) = \frac{\sigma^2}{2} \Delta u$$

where $N_{x_i}$ grid points are used in dimension $i$ with spacing $\Delta x_i = \frac{x_{\max,i} - x_{\min,i}}{N_{x_i}}$.
```

---

## Migration Path

### Phase 1: Documentation (v0.9.x)
- ✅ Create this standard document
- Update `mathematical_notation.py` to reflect new standard
- Add deprecation notices to old naming conventions
- Update README and Getting Started guides

### Phase 2: Backward Compatibility (v0.10.0)
- Add property aliases in `MFGProblem` class
- Support both old (`problem.Nx` scalar) and new (`problem.Nx` array) via properties
- Add deprecation warnings when old names are used
- Update all examples to use new notation

### Phase 3: Full Migration (v1.0.0)
- Remove old properties (breaking change)
- Standardize all internal code to new notation
- Update all tests to new notation
- Release v1.0.0 with clean, consistent API

---

## Rationale

### Why Mathematical Notation?

1. **Code-Math Correspondence**: When implementing algorithms from papers, direct correspondence prevents transcription errors
2. **Cognitive Load**: Domain experts think in terms of $N_x$, $\Delta t$, not `num_spatial_points_in_x_direction`
3. **Algorithm Clarity**: Compare:
   ```python
   # Mathematical notation
   cfl = dt / dx[0]**2

   # Descriptive notation
   courant_friedrichs_lewy_number = time_step_size / spatial_step_size_x**2
   ```
4. **Field Standards**: FEniCS, deal.II, PyFR, SciPy algorithms all use abbreviated notation

### Why Subscript Indexing?

1. **Dimension Agnostic**: Algorithms generalize naturally to arbitrary dimensions
2. **Avoid Code Duplication**: One implementation works for 1D, 2D, 3D, nD
3. **Mathematical Consistency**: Matches standard notation $N_i$, $\Delta x_i$ in nD
4. **Maintainability**: Adding/removing dimensions doesn't require code changes

---

## Examples from Literature

### Quarteroni & Valli (1994) - Numerical Approximation of PDEs
Uses: $N_h$ (grid points), $\Delta x$ (spacing), $\Delta t$ (time step)

### LeVeque (2007) - Finite Difference Methods for ODEs and PDEs
Uses: $m$ (spatial points), $\Delta x$, $\Delta t$, $\lambda = \Delta t / \Delta x$ (CFL)

### Achdou & Capuzzo-Dolcetta (2010) - Mean Field Games
Uses: $N_x$, $N_t$, $\Delta x$, $\Delta t$, $\sigma$ (diffusion)

### Cardaliaguet (2013) - Notes on Mean Field Games
Uses: $\sigma$ (noise), $\nu$ (viscosity), $H(x,p)$ (Hamiltonian)

---

**Last Updated**: 2025-11-04
**Maintainer**: MFG_PDE Core Team
**Status**: Living document - update as conventions evolve
