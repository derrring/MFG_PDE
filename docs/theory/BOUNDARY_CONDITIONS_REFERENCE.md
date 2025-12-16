# Boundary Conditions Reference

**Status**: Canonical Reference Document
**Last Updated**: 2025-12-16
**Consolidates**: `boundary_conditions_and_geometry.md`, `BC_UNIFICATION_TECHNICAL_REPORT.md`, `GEOMETRY_DOMAIN_BC_ARCHITECTURE.md`

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [BC Types](#3-bc-types)
4. [Geometry and Domain Types](#4-geometry-and-domain-types)
5. [BC Applicators](#5-bc-applicators)
6. [Solver Integration](#6-solver-integration)
7. [Numerical Considerations](#7-numerical-considerations)
8. [Future Plans](#8-future-plans)
9. [Code Reference](#9-code-reference)

---

## 1. Overview

Boundary condition (BC) handling in MFG_PDE follows a layered architecture separating **specification** (what conditions apply), **application** (how to enforce them numerically), and **solver integration** (when to apply them during computation).

### 1.1 Design Principles

1. **Separation of concerns**: BC specification is independent of discretization method
2. **Polymorphic applicators**: Each numerical paradigm (FDM, FEM, Meshfree, Graph) has its own applicator
3. **Mixed BC support**: Different conditions on different boundary regions
4. **Graceful degradation**: Solvers without explicit BC use sensible defaults (typically Neumann)

### 1.2 Quick Start

```python
from mfg_pde.geometry.boundary import BoundaryConditions, BCType, BCSegment

# Uniform BC (all boundaries same)
bc = BoundaryConditions(dimension=2, bc_type=BCType.NO_FLUX)

# Mixed BC (different regions)
bc = BoundaryConditions(dimension=2)
bc.add_segment(BCSegment("walls", BCType.NEUMANN, value=0.0, boundary="all"))
bc.add_segment(BCSegment("exit", BCType.DIRICHLET, value=0.0,
                         boundary="x_max", region={"y": (2, 3)}, priority=1))

# Pass to problem
problem = MFGProblem(geometry=grid, boundary_conditions=bc, ...)
```

---

## 2. Architecture

### 2.1 Layered Design

```
+------------------------------------------------------------------+
|                         MFGProblem                                |
|  (Physics: Hamiltonian, costs, initial density, BCs)              |
+------------------------------------------------------------------+
                                |
              +-----------------+-----------------+
              |                                   |
              v                                   v
+---------------------------+     +---------------------------+
|        Geometry           |     |   BoundaryConditions      |
| (Spatial structure)       |     | (Physics at boundary)     |
| - TensorProductGrid       |     | - BCType enum             |
| - ImplicitDomain          |     | - BCSegment list          |
| - NetworkGeometry         |     | - Mixed BC support        |
| - Mesh2D/3D               |     | - SDF integration         |
+---------------------------+     +---------------------------+
              |                                   |
              v                                   v
+---------------------------+     +---------------------------+
|    Boundary Detection     |     |      BC Applicator        |
| - is_on_boundary()        |     | - FDMApplicator           |
| - get_boundary_normal()   |     | - FEMApplicator           |
| - get_boundary_regions()  |     | - MeshfreeApplicator      |
|                           |     | - GraphApplicator         |
+---------------------------+     +---------------------------+
                                              |
                                              v
                              +---------------------------+
                              |         Solver            |
                              | - HJB (FDM, WENO, GFDM)   |
                              | - FP (FDM, Particle)      |
                              | - Coupled MFG             |
                              +---------------------------+
```

### 2.2 Responsibility Matrix

| Component | Knows | Does NOT Know |
|:----------|:------|:--------------|
| **Geometry** | Point locations, topology, boundary detection | BC types, physics |
| **BoundaryConditions** | What conditions apply where | How to enforce them |
| **BCApplicator** | How to enforce BCs numerically | Problem physics |
| **Solver** | When to apply BCs, time-stepping | Geometry details |

---

## 3. BC Types

### 3.1 Supported Types

| Type | Equation | Physical Meaning | Use Case |
|:-----|:---------|:-----------------|:---------|
| `DIRICHLET` | $u = g$ | Fixed value | Exits, absorbing boundaries |
| `NEUMANN` | $\partial u / \partial n = g$ | Fixed flux | Open boundaries |
| `ROBIN` | $\alpha u + \beta \partial u / \partial n = g$ | Mixed condition | Partial absorption |
| `PERIODIC` | $u(x_{min}) = u(x_{max})$ | Wrap-around | Torus topology, traffic |
| `NO_FLUX` | $J \cdot n = 0$ | Zero total flux | Reflecting walls (FP) |
| `REFLECTING` | Elastic bounce | Particle reflection | Particle methods |

### 3.2 Mathematical Formulation

#### HJB Equation with BCs

$$\begin{cases}
\partial_t u + H(x, \nabla u) - \frac{\sigma^2}{2} \Delta u = 0 & \text{in } (0,T) \times \Omega \\
u(T, x) = g(x) & \text{terminal condition} \\
\mathcal{B}[u] = h(x) & \text{on } (0,T) \times \partial\Omega
\end{cases}$$

#### FP Equation with BCs

$$\begin{cases}
\partial_t m - \nabla \cdot (m \nabla_p H) - \frac{\sigma^2}{2} \Delta m = 0 & \text{in } (0,T) \times \Omega \\
m(0, x) = m_0(x) & \text{initial condition} \\
J \cdot n = 0 & \text{on } (0,T) \times \partial\Omega \text{ (no-flux)}
\end{cases}$$

where $J = m \nabla_p H - \frac{\sigma^2}{2} \nabla m$ is the probability flux.

### 3.3 Mass Conservation

For Fokker-Planck equations, BC type affects mass conservation:

| BC Type | Mass Conserving? | Mechanism |
|:--------|:-----------------|:----------|
| No-flux | Yes | Zero flux at boundary |
| Periodic | Yes | Flux wraps around |
| Dirichlet | No | Mass absorbed/injected |
| Neumann (g!=0) | No | Non-zero flux |
| Robin | Depends on coefficients | Partial absorption |

**Discrete Conservation Test**: For no-flux/periodic BCs, the FP matrix $A$ must satisfy $\mathbf{1}^T A = \mathbf{0}^T$ (column sums = 0).

### 3.4 Mixed Boundary Conditions

"Mixed BC" has two meanings:

1. **Robin BC** (`BCType.ROBIN`): Single condition mixing value and flux
   ```python
   BCSegment("absorbing", BCType.ROBIN, value=0.0, alpha=1.0, beta=0.1)
   ```

2. **Spatially Mixed**: Different types on different boundary regions
   ```python
   bc = BoundaryConditions(dimension=2)
   bc.add_segment(BCSegment("walls", BCType.NO_FLUX, boundary="all"))
   bc.add_segment(BCSegment("exit", BCType.DIRICHLET, value=0.0,
                            boundary="x_max", region={"y": (2, 3)}, priority=1))
   ```

### 3.5 BCSegment Specification

```python
@dataclass
class BCSegment:
    name: str                    # Human-readable identifier
    bc_type: BCType              # Type of condition
    value: float | Callable      # BC value (constant or function of x,t)

    # Matching methods (specify one or more)
    boundary: str | None         # "x_min", "x_max", "y_min", "y_max", "all"
    region: dict | None          # {"y": (2.0, 3.0)} for partial boundary
    sdf_region: Callable | None  # SDF for complex regions
    normal_direction: NDArray    # Match by outward normal
    normal_tolerance: float      # Tolerance for normal matching

    # Robin parameters
    alpha: float = 1.0           # Coefficient for u term
    beta: float = 0.0            # Coefficient for du/dn term

    priority: int = 0            # Higher priority wins at overlaps
```

---

## 4. Geometry and Domain Types

### 4.1 Supported Geometries

| Type | Class | BC Detection | Best Solver |
|:-----|:------|:-------------|:------------|
| Cartesian Grid | `TensorProductGrid` | Axis-aligned bounds | FDM, WENO |
| Implicit Domain | `ImplicitDomain` | SDF evaluation | GFDM, Particle |
| Point Cloud | `PointCloudDomain` | Convex hull / SDF | GFDM |
| Network/Graph | `NetworkGeometry` | Node classification | Graph solvers |
| Unstructured Mesh | `Mesh2D`, `Mesh3D` | Boundary faces | FEM |

### 4.2 Method Selection by Geometry

| Domain Type | FDM | FEM | GFDM/Meshfree |
|:------------|:---:|:---:|:-------------:|
| Rectangle (axis-aligned) | **Recommended** | OK | OK |
| Rectangle + rect obstacles | OK (masking) | OK | OK |
| Circle, ellipse | Not recommended | **Recommended** | OK |
| L-shape, polygon | Not recommended | **Recommended** | OK |
| Arbitrary SDF | Not feasible | OK | **Recommended** |
| Moving boundaries | Not feasible | Difficult | **Recommended** |

### 4.3 Why FDM Fails on Complex Geometry

1. **Boundary-grid misalignment**: Ghost point formulas introduce O(dx) errors
2. **Staircase approximation**: Curved boundaries become jagged steps
3. **Normal ambiguity**: At staircase corners, outward normal is undefined

**When FDM is still acceptable**:
- Accuracy requirements are moderate
- Grid resolution is high enough
- Fast prototyping is more valuable than precision

---

## 5. BC Applicators

### 5.1 Applicator Hierarchy

```
                    BaseBCApplicator (ABC)
                           |
           +---------------+---------------+
           |               |               |
    FDMApplicator    FEMApplicator   MeshfreeApplicator   GraphApplicator
```

### 5.2 FDM Applicator: Ghost Cells

For cell-centered grids, ghost cells extend the domain for stencil computation.

**Ghost Cell Formulas** (2nd order accurate):

| BC Type | Left Boundary | Right Boundary |
|:--------|:--------------|:---------------|
| Dirichlet ($u = g$) | $u_{-1} = 2g - u_0$ | $u_{N} = 2g - u_{N-1}$ |
| Neumann ($\partial u/\partial n = g$) | $u_{-1} = u_0 - 2\Delta x \cdot g$ | $u_{N} = u_{N-1} + 2\Delta x \cdot g$ |
| No-flux | $u_{-1} = u_0$ | $u_{N} = u_{N-1}$ |
| Periodic | $u_{-1} = u_{N-1}$ | $u_{N} = u_0$ |

**High-Order Formulas** (for WENO-5):

| Order | Dirichlet Formula |
|:------|:------------------|
| 2nd | $u_g = 2g - u_0$ |
| 4th | $u_g = \frac{16g - 15u_0 + 5u_1 - u_2}{5}$ |
| 5th | Lagrange extrapolation from $u_0, u_1, u_2, u_3$ |

**Usage**:
```python
from mfg_pde.geometry.boundary import apply_boundary_conditions_2d

u_padded = apply_boundary_conditions_2d(
    field=u,
    boundary_conditions=bc,
    domain_bounds=bounds,
    ghost_depth=1
)
```

### 5.3 FEM Applicator: Weak Form

- **Dirichlet**: Penalty method or DOF elimination
- **Neumann**: Natural boundary condition (appears in weak form)
- **Robin**: Boundary integral term

### 5.4 Meshfree Applicator: Particle Methods

```python
applicator = MeshfreeApplicator(dimension=2)

# Three modes
applicator.apply_boundary_conditions(particles, bc_type="reflecting")  # Bounce
applicator.apply_boundary_conditions(particles, bc_type="absorbing")   # Remove
applicator.apply_boundary_conditions(particles, bc_type="periodic")    # Wrap
```

### 5.5 Graph Applicator: Network BCs

- **Dirichlet**: Fix node values
- **No-flux**: Zero edge flux at boundary nodes
- **Periodic**: Connect boundary nodes cyclically

---

## 6. Solver Integration

### 6.1 Current Solver BC Support

| Solver | BC Retrieval | BC Application | Status |
|:-------|:-------------|:---------------|:-------|
| **HJB GFDM** | `get_boundary_conditions()` | Ghost particles | Complete |
| **HJB Semi-Lagrangian** | `get_boundary_conditions()` | Characteristic clamping | Complete |
| **HJB FDM** | None | Hardcoded one-sided | Needs wiring |
| **HJB WENO** | None | Hardcoded one-sided | Needs wiring |
| **FP FDM** | `problem.components.boundary_conditions` | `fp_fdm_bc.py` | Uniform only |
| **FP Particle** | `geometry.get_boundary_handler()` | Reflection/periodic | Uniform only |

### 6.2 Standard BC Retrieval Pattern

All solvers should use:
```python
def _get_boundary_conditions(self) -> BoundaryConditions | None:
    """Retrieve BCs with standard priority."""
    # 1. Explicit parameter
    if self._boundary_conditions is not None:
        return self._boundary_conditions
    # 2. Problem accessor
    if hasattr(self.problem, 'get_boundary_conditions'):
        return self.problem.get_boundary_conditions()
    # 3. No BC available
    return None
```

### 6.3 Solver-Specific Issues

**HJB WENO/FDM** - Hardcoded boundary stencils:
```python
# Current (problematic)
u_x[0] = (-3*u[0] + 4*u[1] - u[2]) / (2*dx)  # Always forward diff

# Should use ghost cells from BC specification
```

**FP FDM** - Mixed BC failure:
```python
# Current (crashes on mixed BC)
if self.boundary_conditions.type == "no_flux":  # .type raises for mixed

# Should use
if bc.is_uniform:
    bc_type = bc.type
else:
    bc_type = bc.default_bc.value  # or per-boundary handling
```

---

## 7. Numerical Considerations

### 7.1 Corner Handling (2D/3D)

At corners where different BC types meet, ghost cell values are ambiguous.

**Corner Strategies**:

| Strategy | Formula | Use Case |
|:---------|:--------|:---------|
| `"priority"` | Higher-priority BC wins | Sharp corners, discontinuous |
| `"average"` | Average of contributing BCs | Smooth solutions |
| `"mollify"` | Smooth blending within radius | Lipschitz domains |

**3D Generalization**: BC precedence hierarchy
```python
BC_PRECEDENCE = {
    BCType.PERIODIC: 1,    # Highest: removes boundary
    BCType.DIRICHLET: 2,   # Hard constraint
    BCType.NEUMANN: 3,     # Flux constraint
    BCType.ROBIN: 4,       # Mixed
    BCType.NO_FLUX: 5,     # Lowest
}
```

### 7.2 Stencil Order Consistency

Ghost cell extrapolation order must match interior scheme order:

| Scheme | Interior Order | Ghost Depth | Ghost Order Needed |
|:-------|:---------------|:------------|:-------------------|
| FDM (central) | $O(\Delta x^2)$ | 1 | 2nd |
| Compact-4 | $O(\Delta x^4)$ | 2 | 4th |
| WENO-5 | $O(\Delta x^5)$ | 3 | 5th |

**Current state**: `applicator_fdm.py` provides 2nd order. WENO accuracy degrades near boundaries.

### 7.3 Performance: Memory Allocation

**Problem**: Creating padded arrays every timestep causes allocation overhead.

**Solution**: Pre-allocated view-based memory model
```python
class BoundaryAwareSolver:
    def __init__(self, ...):
        g = self.ghost_depth
        self._U_padded = np.zeros((Nx + 2*g, Ny + 2*g))
        self._U_interior = self._U_padded[g:-g, g:-g]  # Zero-copy view

    def _apply_bc_in_place(self):
        """Update ghost cells without allocation."""
        # Only touches boundary rim, not full array
```

### 7.4 Validation: Method of Manufactured Solutions

Rigorous BC validation requires MMS:
1. Choose analytical solution $u^*(x,t)$ satisfying desired BC
2. Compute forcing $f = \partial_t u^* + H(\nabla u^*) - \frac{\sigma^2}{2}\Delta u^*$
3. Solve modified PDE with forcing
4. Verify convergence rate matches expected order

---

## 8. Future Plans

### 8.1 Phase 1: HJB Solver BC Wiring (Priority: High)

- [ ] Wire `HJB WENO` to use ghost cell infrastructure
- [ ] Wire `HJB FDM` nD path to use ghost cells
- [ ] Implement high-order ghost cell formulas for WENO-5
- [ ] Add `bc_strict_mode` option for explicit BC requirement

### 8.2 Phase 2: FP Solver Mixed BC Support (Priority: High)

- [ ] Safe BC type accessor for mixed BCs in `fp_fdm.py`
- [ ] Per-boundary BC handling in `fp_fdm_operators.py`
- [ ] Unit tests for matrix mass conservation (column sums = 0)
- [ ] Mixed BC support for particle methods

### 8.3 Phase 3: BoundaryAwareSolver Base Class (Priority: Medium)

```python
class BoundaryAwareSolver(ABC):
    """Mandatory base class for grid-based solvers."""

    bc_strict_mode: bool = False
    ghost_cell_depth: int = 1

    def _get_boundary_conditions(self) -> BoundaryConditions | None: ...
    def _apply_bc_padding(self, field: np.ndarray) -> np.ndarray: ...
    def _apply_bc_in_place(self) -> None: ...  # Pre-allocated
    def _get_bc_type_at_boundary(self, boundary: str) -> BCType: ...

    @abstractmethod
    def _get_required_ghost_depth(self) -> int: ...
```

### 8.4 Phase 4: Advanced Features (Priority: Low)

- [ ] Time-dependent BC values: $g(x, t)$
- [ ] State-dependent BCs: $g(x, u)$
- [ ] Adaptive ghost depth based on local Peclet number
- [ ] Numba/Cython acceleration for ghost cell loops

### 8.5 Neural/Variational Paradigm BCs (Priority: Medium)

- [ ] Physics-Informed Neural Networks (PINNs): BC loss terms
- [ ] Deep Galerkin Method (DGM): Boundary sampling
- [ ] Actor-Critic RL: State space truncation

### 8.6 Implementation Checklist

| Task | Status | Assignee | Notes |
|:-----|:-------|:---------|:------|
| Corner logic audit | Pending | - | Verify all BC combinations |
| WENO ghost depth verification | Pending | - | Confirm 3 cells needed |
| FP matrix conservation tests | Pending | - | Column sum = 0 |
| MMS validation tests | Pending | - | Per-solver convergence |
| Pre-allocation prototype | Pending | - | Memory benchmark |

---

## 9. Code Reference

### 9.1 File Locations

```
mfg_pde/geometry/boundary/
├── __init__.py              # Public API exports
├── types.py                 # BCType enum, BCSegment dataclass
├── conditions.py            # BoundaryConditions class
├── applicator_base.py       # BaseBCApplicator ABC
├── applicator_fdm.py        # Ghost cell computation (FDM)
├── applicator_fem.py        # Weak form BCs (FEM)
├── applicator_meshfree.py   # Particle reflection/absorption
├── applicator_graph.py      # Network node constraints
└── fdm_bc_1d.py             # Legacy 1D (deprecated)

mfg_pde/alg/numerical/
├── hjb_solvers/
│   ├── hjb_fdm.py           # Needs BC wiring
│   ├── hjb_weno.py          # Needs BC wiring
│   ├── hjb_gfdm.py          # BC support complete
│   └── hjb_semi_lagrangian.py  # BC support complete
└── fp_solvers/
    ├── fp_fdm.py            # Needs mixed BC support
    ├── fp_fdm_bc.py         # No-flux enforcement
    └── fp_particle.py       # Needs mixed BC support
```

### 9.2 Key Classes

```python
# BC Specification
from mfg_pde.geometry.boundary import (
    BCType,                    # Enum: DIRICHLET, NEUMANN, ROBIN, PERIODIC, NO_FLUX
    BCSegment,                 # Dataclass: single BC on region
    BoundaryConditions,        # Container: collection of segments
    mixed_bc,                  # Factory: create mixed BCs easily
)

# BC Application
from mfg_pde.geometry.boundary import (
    apply_boundary_conditions_1d,
    apply_boundary_conditions_2d,
    apply_boundary_conditions_nd,
    FDMApplicator,
    MeshfreeApplicator,
)
```

### 9.3 Example: Complete Evacuation Setup

```python
from mfg_pde import MFGProblem
from mfg_pde.geometry import TensorProductGrid
from mfg_pde.geometry.boundary import BoundaryConditions, BCType, BCSegment

# 1. Geometry
grid = TensorProductGrid(
    dimension=2,
    bounds=[(0, 10), (0, 5)],
    num_points=[101, 51]
)

# 2. HJB BCs: Exit at right wall (y in [2,3])
bc_hjb = BoundaryConditions(dimension=2)
bc_hjb.add_segment(BCSegment("walls", BCType.NEUMANN, value=0.0, boundary="all"))
bc_hjb.add_segment(BCSegment("exit", BCType.DIRICHLET, value=0.0,
                              boundary="x_max", region={"y": (2.0, 3.0)}, priority=1))

# 3. FP BCs: No-flux walls + absorbing exit
bc_fp = BoundaryConditions(dimension=2)
bc_fp.add_segment(BCSegment("walls", BCType.NO_FLUX, boundary="all"))
bc_fp.add_segment(BCSegment("exit", BCType.DIRICHLET, value=0.0,
                             boundary="x_max", region={"y": (2.0, 3.0)}, priority=1))

# 4. Problem setup
problem = MFGProblem(
    geometry=grid,
    boundary_conditions=bc_hjb,  # Or pass separately to solvers
    ...
)
```

---

## Appendix A: Archived Documents

This reference consolidates and supersedes:

| Document | Status | Action |
|:---------|:-------|:-------|
| `docs/theory/boundary_conditions_and_geometry.md` | Superseded | Archive |
| `docs/development/reports/BC_UNIFICATION_TECHNICAL_REPORT.md` | Superseded | Archive |
| `docs/development/guides/GEOMETRY_DOMAIN_BC_ARCHITECTURE.md` | Superseded | Archive |
| `docs/archive/COMPLETED_MIXED_BC_DESIGN.md` | Already archived | No action |
| `docs/archive/[COMPLETED]_BC_APPLICATOR_ENHANCEMENT_PLAN.md` | Already archived | No action |

---

## Appendix B: References

1. LeVeque, R.J. "Finite Difference Methods for ODEs and PDEs" (2007), Ch. 9 - Ghost cells
2. Achdou, Y. et al. "Mean Field Games: Numerical Methods" (2020), Sec. 4.3 - MFG BCs
3. Roache, P.J. "Code Verification by MMS" (2002) - Validation methodology
4. Shu, C.W. "High Order WENO Schemes" (1998) - High-order boundary treatment

---

*This document is the canonical reference for boundary condition handling in MFG_PDE.*
