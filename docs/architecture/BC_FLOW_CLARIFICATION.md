# Boundary Condition Flow in MFG_PDE

**Status**: Clarification Document
**Date**: 2025-12-17
**Related Issue**: #493

---

## 0. Key Insight: Spatial BC vs Temporal Conditions

```
┌─────────────────────────────────────────────────────────────┐
│                    MFG Problem Domain                       │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐   │
│   │              SPATIAL DOMAIN (Geometry)              │   │
│   │                                                     │   │
│   │   SHARED by HJB and FP:                             │   │
│   │   - Domain shape (maze, room, etc.)                 │   │
│   │   - Walls (no-flux / Neumann)                       │   │
│   │   - Exits (Dirichlet / absorbing)                   │   │
│   │   - Periodic boundaries                             │   │
│   │                                                     │   │
│   │   → Single source: geometry.boundary_conditions     │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                             │
│   ┌──────────────────────┐   ┌──────────────────────────┐   │
│   │   TEMPORAL: t=0      │   │   TEMPORAL: t=T          │   │
│   │                      │   │                          │   │
│   │   FP: m_init         │   │   HJB: u_fin             │   │
│   │   (initial density)  │   │   (terminal cost)        │   │
│   └──────────────────────┘   └──────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**Principle**: Spatial BC is a property of the **geometry** (same physical domain for both equations), not the individual solvers.

### Software Engineering Principles

| Principle | Application |
|-----------|-------------|
| **Separation of Concerns** | Spatial BC (static, geometric) vs Temporal BC (dynamic, task-specific) |
| **Dependency Injection** | Solvers receive `geometry` object, don't hardcode BC |
| **Single Source of Truth (SSOT)** | One `Geometry` instance shared by HJB and FP |
| **Configuration Centralization** | User defines geometry once, system distributes |

**Key distinction**:
- **Environment (Geometry)**: Static, shared → Spatial BC (walls, exits)
- **Task (Payload)**: Dynamic, equation-specific → Temporal BC (m_init, u_fin)

---

## 1. Current BC Flow Diagram

```
                    ┌─────────────────────────────────┐
                    │         User Input              │
                    │  (one of these sources)         │
                    └─────────────────────────────────┘
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         │                         │                         │
         ▼                         ▼                         ▼
┌─────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│  Solver param   │    │  MFGComponents.     │    │  Geometry.          │
│  (explicit)     │    │  boundary_conditions│    │  boundary_handler   │
│                 │    │                     │    │                     │
│  FPFDMSolver(   │    │  components=MFG     │    │  (future Phase 2)   │
│    bc=no_flux() │    │  Components(        │    │                     │
│  )              │    │    boundary_cond=.. │    │                     │
│                 │    │  )                  │    │                     │
└────────┬────────┘    └──────────┬──────────┘    └──────────┬──────────┘
         │                        │                          │
         │ Priority 1             │ Priority 2               │ Priority 3
         │ (highest)              │                          │
         └────────────────────────┼──────────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────────┐
                    │      Solver's self.bc           │
                    │   (resolved at __init__)        │
                    └─────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
          ┌─────────────────┐         ┌─────────────────┐
          │   HJBFDMSolver  │         │   FPFDMSolver   │
          │                 │         │                 │
          │  No explicit BC │         │  Uses self.bc   │
          │  (implicit)     │         │  in operators   │
          └─────────────────┘         └─────────────────┘
```

---

## 2. Where BC is Defined

### 2.1 BC Type Definitions (`geometry/boundary/types.py`)

```python
class BCType(Enum):
    DIRICHLET = "dirichlet"    # u = g at boundary
    NEUMANN = "neumann"        # du/dn = g at boundary
    ROBIN = "robin"            # α*u + β*du/dn = g
    PERIODIC = "periodic"      # u(x_min) = u(x_max)
    NO_FLUX = "no_flux"        # J·n = 0 (for FP density)
    REFLECTING = "reflecting"  # For particle methods
```

### 2.2 BC Segment Specification (`geometry/boundary/types.py`)

```python
@dataclass
class BCSegment:
    name: str
    bc_type: BCType
    value: float | Callable = 0.0
    boundary: str | None = None  # "left", "right", "x_min", etc.
    region: dict | None = None   # For partial boundary
    priority: int = 0
```

### 2.3 Unified BC Container (`geometry/boundary/conditions.py`)

```python
@dataclass
class BoundaryConditions:
    dimension: int
    segments: list[BCSegment]
    default_bc: BCType = BCType.PERIODIC
    domain_bounds: np.ndarray | None = None
```

### 2.4 MFGComponents Storage (`core/mfg_components.py`)

```python
@dataclass
class MFGComponents:
    boundary_conditions: BoundaryConditions | None = None
    # ... other components
```

---

## 3. Resolution Hierarchy (FPFDMSolver)

From `fp_fdm.py` lines 161-189:

```python
# Priority 1: Explicit solver parameter
if boundary_conditions is not None:
    self.boundary_conditions = boundary_conditions

# Priority 2: From problem.components
elif hasattr(problem, "components") and problem.components is not None:
    if problem.components.boundary_conditions is not None:
        self.boundary_conditions = problem.components.boundary_conditions
    else:
        self.boundary_conditions = no_flux_bc(dimension=self.dimension)

# Priority 3: From geometry (future Phase 2)
elif hasattr(problem, "geometry") and hasattr(problem.geometry, "get_boundary_handler"):
    self.boundary_conditions = problem.geometry.get_boundary_handler()

# Priority 4: Default fallback
else:
    self.boundary_conditions = no_flux_bc(dimension=self.dimension)
```

---

## 4. Current Inconsistencies

| Component | BC Source | Default | Notes |
|-----------|-----------|---------|-------|
| **HJBFDMSolver** | None (implicit) | Implicit Neumann | No explicit BC param |
| **FPFDMSolver** | Explicit param | no_flux | Documented hierarchy |
| **MFGProblem** | `get_boundary_conditions()` | periodic | Via ConditionsMixin |
| **MFGComponents** | `boundary_conditions` attr | None | User sets explicitly |
| **FixedPointIterator** | Uses problem's BC | - | Doesn't override |

### Issues:

1. **HJB has no explicit BC parameter** - relies on implicit handling
2. **Different defaults**: HJB (Neumann), FP (no_flux), Problem (periodic)
3. **FP solver doesn't query problem.get_boundary_conditions()** - uses its own hierarchy
4. **No single source of truth** - BC can come from multiple places

---

## 5. Recommended Architecture

### 5.1 Core Principle: Geometry Owns Spatial BC

Spatial BC is a property of the physical domain, shared by both HJB and FP:

```
Geometry (TensorProductGrid)
    │
    └── boundary_conditions: BoundaryConditions  # SINGLE SOURCE
            │
            ├── Used by HJBFDMSolver
            └── Used by FPFDMSolver

MFGProblem
    │
    ├── geometry ──────────────► Spatial BC (shared)
    ├── m_init                    FP initial condition (t=0)
    └── u_fin                     HJB terminal condition (t=T)
```

### 5.2 Target API

```python
# Geometry owns spatial BC (single source of truth)
geometry = TensorProductGrid(
    dimension=2,
    bounds=[(0, 20), (0, 10)],
    Nx=[40, 20],
    boundary_conditions=mixed_bc([
        BCSegment("exit", BCType.DIRICHLET, value=0.0, boundary="x_max"),
        BCSegment("walls", BCType.NO_FLUX),
    ]),
)

# Problem owns temporal conditions + references geometry
problem = MFGProblem(
    geometry=geometry,      # Spatial domain + spatial BC
    m_init=m0,              # FP initial condition at t=0
    u_fin=g,                # HJB terminal condition at t=T
    T=1.0, Nt=50,
    sigma=0.1,
)

# Solvers get spatial BC from geometry (via problem or directly)
hjb_solver = HJBFDMSolver(problem)  # Gets BC from problem.geometry
fp_solver = FPFDMSolver(problem)    # Gets SAME BC from problem.geometry

# OR standalone usage (without full problem):
hjb_solver = HJBFDMSolver(geometry=geometry, sigma=0.1, T=1.0, Nt=50)
```

### 5.3 Solver Resolution (Unified)

```python
class HJBFDMSolver:  # Same pattern for FPFDMSolver
    def __init__(self, problem=None, geometry=None, boundary_conditions=None, ...):
        # Resolve geometry
        if problem is not None:
            self.geometry = problem.geometry
        elif geometry is not None:
            self.geometry = geometry
        else:
            raise ValueError("Need problem or geometry")

        # Spatial BC resolution:
        # 1. Explicit param (override for special cases)
        # 2. Geometry's BC (normal case)
        # 3. Default fallback
        if boundary_conditions is not None:
            self.spatial_bc = boundary_conditions
        elif self.geometry is not None:
            self.spatial_bc = self.geometry.get_boundary_conditions()
        else:
            self.spatial_bc = no_flux_bc(dimension=self.dimension)
```

### 5.4 Design Requirements Met

| Requirement | Solution |
|-------------|----------|
| 1. Solvers standalone | Accept `geometry` param directly |
| 2. System consistency | Spatial BC lives in geometry (single source) |
| 3. Single config | `problem.geometry` holds BC, no duplication |

### 5.5 Physical Defaults

| Equation | Typical BC | Physical Meaning |
|----------|------------|------------------|
| **HJB** | Neumann (du/dn=0) | Value function smooth at boundary |
| **FP** | No-flux (J·n=0) | Mass conservation, impermeable walls |

For **exit problems** (mixed BC on geometry):
- Exit boundary: Dirichlet (u=0 for HJB, absorbing for FP)
- Wall boundaries: Neumann/No-flux (impermeable)

---

## 6. Proposed Changes

### Phase 1: Add BC to Geometry

```python
# geometry/grids/tensor_grid.py
class TensorProductGrid(CartesianGrid):
    def __init__(
        self,
        dimension: int,
        bounds: list[tuple[float, float]],
        Nx: list[int],
        boundary_conditions: BoundaryConditions | None = None,  # NEW
    ):
        ...
        self._boundary_conditions = boundary_conditions or no_flux_bc(dimension)

    def get_boundary_conditions(self) -> BoundaryConditions:
        """Get spatial boundary conditions for this domain."""
        return self._boundary_conditions
```

### Phase 2: Solvers Query Geometry

```python
# Both HJB and FP use same pattern
class HJBFDMSolver:
    def __init__(
        self,
        problem=None,
        geometry=None,  # NEW: standalone mode
        boundary_conditions: BoundaryConditions | None = None,  # Override
        ...
    ):
        # Resolve geometry
        self.geometry = problem.geometry if problem else geometry

        # Spatial BC: explicit override > geometry > default
        if boundary_conditions is not None:
            self.spatial_bc = boundary_conditions
        elif self.geometry is not None:
            self.spatial_bc = self.geometry.get_boundary_conditions()
        else:
            self.spatial_bc = no_flux_bc(dimension=self.dimension)
```

### Phase 3: Config Integration

```yaml
# mfg_config.yaml
geometry:
  type: tensor_product
  dimension: 2
  bounds: [[0, 20], [0, 10]]
  Nx: [40, 20]
  boundary_conditions:          # Spatial BC (shared)
    segments:
      - name: exit
        type: dirichlet
        value: 0.0
        boundary: x_max
      - name: walls
        type: no_flux

problem:
  T: 1.0
  Nt: 50
  sigma: 0.1
  payload:                      # Temporal BC (equation-specific)
    m_init: gaussian
    u_fin: quadratic_distance
```

### Phase 4: MFGSystem Pattern

```python
class MFGSystem:
    """Coupled MFG system with centralized geometry."""

    def __init__(self, config):
        # 1. Create single geometry object (SSOT)
        self.geometry = GeometryFactory.create(config.geometry)

        # 2. Inject same geometry to both solvers (DI + Consistency)
        self.hjb_solver = HJBSolver(geometry=self.geometry, params=config.hjb)
        self.fp_solver = FPSolver(geometry=self.geometry, params=config.fp)

        # 3. Store temporal conditions
        self.m_init = config.payload.m_init
        self.u_fin = config.payload.u_fin

    def solve(self):
        # 4. Pass temporal BC at runtime
        u = self.hjb_solver.solve_backward(terminal_cond=self.u_fin)
        m = self.fp_solver.solve_forward(initial_cond=self.m_init, control=u)
        return u, m
```

This pattern ensures:
- User only configures geometry once
- HJB and FP automatically share same spatial BC
- Temporal conditions are clearly separated

---

## 7. Current Usage Patterns

### Pattern 1: Simple uniform BC (current)

```python
from mfg_pde.geometry.boundary import no_flux_bc

problem = MFGProblem(geometry=grid, T=1.0, Nt=50, sigma=0.1)

# FP solver with explicit BC
fp_solver = FPFDMSolver(problem, boundary_conditions=no_flux_bc(dimension=2))

# HJB solver (no BC param currently)
hjb_solver = HJBFDMSolver(problem)
```

### Pattern 2: Via MFGComponents

```python
from mfg_pde.geometry.boundary import mixed_bc, BCSegment, BCType

# Define mixed BC
exit_seg = BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=0.0, boundary="x_max")
wall_seg = BCSegment(name="walls", bc_type=BCType.NO_FLUX, value=0.0)
bc = mixed_bc([exit_seg, wall_seg], dimension=2, domain_bounds=bounds)

# Create problem with components
components = MFGComponents(
    boundary_conditions=bc,
    # ... other components
)
problem = MFGProblem.from_components(components, geometry=grid, T=1.0, Nt=50)

# FP solver will pick up BC from problem.components
fp_solver = FPFDMSolver(problem)  # Uses problem.components.boundary_conditions
```

---

## 8. Summary

**Key Insight**: Spatial BC belongs to **geometry** (same physical domain for both PDEs), while temporal conditions (m_init, u_fin) belong to the **problem**.

**Current State**:
- BC system is flexible but inconsistent across solvers
- FP solver has explicit BC resolution hierarchy
- HJB solver has no explicit BC parameter
- Multiple potential sources lead to confusion
- Spatial BC duplicated/inconsistent between HJB and FP

**Recommended Architecture**:
```
Geometry
    └── boundary_conditions  ← Single source for spatial BC

MFGProblem
    ├── geometry             ← References spatial BC
    ├── m_init               ← FP initial condition (t=0)
    └── u_fin                ← HJB terminal condition (t=T)

Solvers
    └── Query geometry.get_boundary_conditions()
```

**Action Items**:
1. [x] Create GitHub issue for BC unification (#493)
2. [ ] Add `boundary_conditions` param to `TensorProductGrid`
3. [ ] Add `get_boundary_conditions()` method to geometry base class
4. [ ] Update HJB/FP solvers to query geometry for spatial BC
5. [ ] Support standalone solver mode with `geometry` param

---

## References

1. Achdou, Y., & Lauriere, M. (2020). Mean field games and applications: Numerical aspects. *Mean Field Games*, 249-307.
   - Discusses MFG numerical implementation in complex geometric domains
   - Emphasizes consistency of underlying mesh and boundary definitions

2. Gueant, O. (2016). *The Financial Mathematics of Market Liquidity*. CRC Press.
   - Standard practice: decouple model parameters from numerical mesh in applied math software
