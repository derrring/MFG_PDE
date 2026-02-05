# Boundary Condition Architecture: Specification vs Applicator

**Date**: 2025-12-17
**Status**: For External Audit
**Related Issues**: #493

---

## 1. Executive Summary

The MFG_PDE boundary condition system has two distinct layers:

| Layer | Purpose | Question Answered | Key Classes |
|-------|---------|-------------------|-------------|
| **Specification** | Describe WHAT BC to apply | "What type? What value? Which boundary?" | `BoundaryConditions`, `BCSegment`, `BCType` |
| **Applicator** | Execute HOW to apply BC | "How to modify the matrix? How to set values?" | `BaseBCApplicator`, `apply_boundary_conditions_*()` |

**Analogy**: Specification is like a recipe (ingredients, steps). Applicator is the chef (actually cooks the meal).

---

## 2. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    BOUNDARY CONDITION ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                     SPECIFICATION LAYER                                │ │
│  │                     (WHAT to apply)                                    │ │
│  │                                                                        │ │
│  │   conditions.py                    types.py                           │ │
│  │   ┌────────────────────┐          ┌─────────────────┐                 │ │
│  │   │ BoundaryConditions │          │ BCType (enum)   │                 │ │
│  │   │                    │          │ - DIRICHLET     │                 │ │
│  │   │ - dimension: int   │          │ - NEUMANN       │                 │ │
│  │   │ - segments: list   │◄─────────│ - PERIODIC      │                 │ │
│  │   │ - default_bc: type │          │ - NO_FLUX       │                 │ │
│  │   │ - is_uniform: bool │          │ - ROBIN         │                 │ │
│  │   └────────────────────┘          └─────────────────┘                 │ │
│  │            │                              │                            │ │
│  │            │         ┌────────────────────┘                            │ │
│  │            ▼         ▼                                                 │ │
│  │   ┌─────────────────────────┐                                         │ │
│  │   │ BCSegment               │  ← Describes ONE boundary region        │ │
│  │   │ - name: str             │                                         │ │
│  │   │ - bc_type: BCType       │                                         │ │
│  │   │ - value: float|callable │                                         │ │
│  │   │ - boundary: str|None    │  ("left", "right", "top", "bottom")     │ │
│  │   │ - region: callable|None │  (SDF for complex boundaries)           │ │
│  │   └─────────────────────────┘                                         │ │
│  │                                                                        │ │
│  │   Factory Functions:                                                   │ │
│  │   - dirichlet_bc(value, dimension) → BoundaryConditions               │ │
│  │   - neumann_bc(flux, dimension) → BoundaryConditions                  │ │
│  │   - periodic_bc(dimension) → BoundaryConditions                       │ │
│  │   - no_flux_bc(dimension) → BoundaryConditions                        │ │
│  │   - mixed_bc(segments, dimension) → BoundaryConditions                │ │
│  │                                                                        │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                    │                                        │
│                                    │ Specification informs                  │
│                                    ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                      APPLICATOR LAYER                                  │ │
│  │                      (HOW to apply)                                    │ │
│  │                                                                        │ │
│  │   applicator_base.py                                                  │ │
│  │   ┌─────────────────────────────────────────────────────────────────┐ │ │
│  │   │ BaseBCApplicator (ABC)                                          │ │ │
│  │   │ - apply(field, t) → modified field                              │ │ │
│  │   │ - apply_to_matrix(A, b) → modified system                       │ │ │
│  │   │ - get_boundary_mask() → bool array                              │ │ │
│  │   └─────────────────────────────────────────────────────────────────┘ │ │
│  │            │                                                          │ │
│  │            ├──────────────────┬──────────────────┬──────────────────┐ │ │
│  │            ▼                  ▼                  ▼                  ▼ │ │
│  │   ┌────────────────┐ ┌────────────────┐ ┌────────────────┐ ┌───────┐ │ │
│  │   │ BaseStructured │ │BaseUnstructured│ │ BaseMeshfree   │ │ Graph │ │ │
│  │   │ Applicator     │ │ Applicator     │ │ Applicator     │ │  BC   │ │ │
│  │   │ (FDM grids)    │ │ (FEM meshes)   │ │ (GFDM/RBF)     │ │       │ │ │
│  │   └────────────────┘ └────────────────┘ └────────────────┘ └───────┘ │ │
│  │                                                                       │ │
│  │   Method-Specific Implementations:                                    │ │
│  │                                                                       │ │
│  │   FDM (applicator_fdm.py):                                           │ │
│  │   - apply_boundary_conditions_1d(u, bc, dx)                          │ │
│  │   - apply_boundary_conditions_2d(u, bc, dx, dy)                      │ │
│  │   - apply_boundary_conditions_3d(u, bc, dx, dy, dz)                  │ │
│  │   - apply_boundary_conditions_nd(u, bc, spacing)  ← Unified nD       │ │
│  │                                                                       │ │
│  │   FEM (fem_bc_*.py):                                                 │ │
│  │   - DirichletBC1D/2D/3D.apply(K, F, nodes)                           │ │
│  │   - NeumannBC1D/2D/3D.apply(K, F, nodes)                             │ │
│  │   - BoundaryConditionManager1D/2D/3D.apply_all()                     │ │
│  │   - MFGBoundaryHandler1D/2D/3D (HJB+FP combined)                     │ │
│  │                                                                       │ │
│  │   Meshfree (applicator_meshfree.py):                                 │ │
│  │   - GFDM boundary handling                                           │ │
│  │   - RBF collocation BC                                               │ │
│  │                                                                       │ │
│  │   Graph (applicator_graph.py):                                       │ │
│  │   - NodeBC, EdgeBC for network problems                              │ │
│  │                                                                       │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Specification Layer Detail

### 3.1 BoundaryConditions (conditions.py)

The **unified nD-capable specification** that describes boundary conditions:

```python
@dataclass
class BoundaryConditions:
    """Unified boundary condition specification for any dimension."""

    dimension: int                    # Spatial dimension (1, 2, 3, ...)
    segments: list[BCSegment]         # List of BC segments
    default_bc: BCType = NO_FLUX      # Default for unspecified boundaries
    default_value: float = 0.0
    domain_bounds: tuple | None       # Optional domain bounds

    @property
    def is_uniform(self) -> bool:
        """True if single BC type on all boundaries."""
        return len(self.segments) == 1

    @property
    def bc_type(self) -> BCType:
        """Primary BC type (for uniform BCs)."""
        return self.segments[0].bc_type

    @property
    def type(self) -> str:
        """Legacy compatibility: returns bc_type.value string."""
        return self.bc_type.value
```

### 3.2 BCSegment (types.py)

Describes a **single boundary region** and its condition:

```python
@dataclass
class BCSegment:
    """Specification for one boundary segment."""

    name: str                              # Human-readable name
    bc_type: BCType                        # DIRICHLET, NEUMANN, etc.
    value: float | Callable = 0.0          # BC value or function
    alpha: float = 1.0                     # Robin: alpha*u + beta*du/dn = value
    beta: float = 0.0
    boundary: str | None = None            # "left", "right", "top", etc.
    region: Callable | None = None         # SDF for complex regions
    normal_direction: NDArray | None = None
```

### 3.3 Factory Functions

Convenient constructors for common BC patterns:

```python
# Uniform Dirichlet: u = 0 on all boundaries
bc = dirichlet_bc(value=0.0, dimension=2)

# Uniform Neumann (no-flux): du/dn = 0 on all boundaries
bc = no_flux_bc(dimension=2)

# Periodic: u(left) = u(right)
bc = periodic_bc(dimension=2)

# Mixed: Different BC on different boundaries
bc = mixed_bc(
    segments=[
        BCSegment(name="walls", bc_type=BCType.NO_FLUX, boundary="left"),
        BCSegment(name="walls", bc_type=BCType.NO_FLUX, boundary="right"),
        BCSegment(name="exit", bc_type=BCType.DIRICHLET, value=0.0, boundary="top"),
        BCSegment(name="inlet", bc_type=BCType.NEUMANN, value=1.0, boundary="bottom"),
    ],
    dimension=2
)
```

---

## 4. Applicator Layer Detail

### 4.1 Purpose

Applicators **execute** the boundary conditions by modifying:
- Solution arrays (for explicit methods)
- System matrices and RHS vectors (for implicit methods)

### 4.2 FDM Applicators (applicator_fdm.py)

For finite difference methods on structured grids:

```python
def apply_boundary_conditions_2d(
    u: NDArray,                    # Solution array to modify
    bc: BoundaryConditions,        # Specification (WHAT)
    dx: float,                     # Grid spacing x
    dy: float,                     # Grid spacing y
    t: float = 0.0,               # Time (for time-dependent BC)
) -> NDArray:
    """Apply BC to 2D array in-place."""

    if bc.bc_type == BCType.DIRICHLET:
        # Set boundary values directly
        u[0, :] = bc.get_value("left", t)
        u[-1, :] = bc.get_value("right", t)
        u[:, 0] = bc.get_value("bottom", t)
        u[:, -1] = bc.get_value("top", t)

    elif bc.bc_type == BCType.NEUMANN:
        # Use ghost points: u[-1] = u[1] - 2*dx*flux
        # ... implementation details

    elif bc.bc_type == BCType.PERIODIC:
        # Wrap boundaries: u[0] = u[-2], u[-1] = u[1]
        # ... implementation details

    return u
```

### 4.3 FEM Applicators (fem_bc_*.py)

For finite element methods, modify the assembled system:

```python
class DirichletBC2D(BoundaryCondition2D):
    """Dirichlet BC for 2D FEM."""

    def apply(
        self,
        K: sparse.csr_matrix,      # Stiffness matrix
        F: NDArray,                # Load vector
        nodes: NDArray,            # Node coordinates
        boundary_nodes: list[int], # Nodes on this boundary
    ) -> tuple[sparse.csr_matrix, NDArray]:
        """Apply Dirichlet BC by modifying K and F."""

        for node in boundary_nodes:
            # Zero out row except diagonal
            K[node, :] = 0
            K[node, node] = 1.0
            # Set RHS to boundary value
            F[node] = self.value(nodes[node])

        return K, F
```

### 4.4 Applicator Hierarchy

```
BaseBCApplicator (ABC)
    │
    ├── BaseStructuredApplicator     # For regular grids (FDM)
    │       └── FDMBCApplicator
    │
    ├── BaseUnstructuredApplicator   # For irregular meshes (FEM)
    │       └── FEMBCApplicator
    │
    ├── BaseMeshfreeApplicator       # For meshfree methods
    │       └── GFDMBCApplicator
    │
    └── BaseGraphApplicator          # For network problems
            └── GraphBCApplicator
```

---

## 5. How Specification and Applicator Connect

### 5.1 Current Flow (Before Issue #493)

```
User Code                    Problem                      Solver
─────────                    ───────                      ──────

bc = dirichlet_bc(0, dim=2)
        │
        ▼
MFGComponents(
    boundary_conditions=bc  ─────► MFGProblem
)                                      │
                                       │ get_boundary_conditions()
                                       ▼
                              FPFDMSolver
                                       │
                                       │ (duplicated BC resolution logic)
                                       ▼
                              apply_boundary_conditions_2d(u, bc, ...)
```

**Problems:**
1. BC stored in MFGComponents, not geometry
2. Solvers had duplicated BC resolution logic
3. HJB and FP could have inconsistent BC

### 5.2 New Flow (After Issue #493)

```
User Code                    Geometry                     Solver
─────────                    ────────                     ──────

bc = dirichlet_bc(0, dim=2)
        │
        ▼
TensorProductGrid(           ┌─────────────────────┐
    ...,                     │ Geometry            │
    boundary_conditions=bc ──►│ - _boundary_cond   │
)                            │                     │
        │                    │ get_boundary_       │
        │                    │   conditions()      │
        ▼                    │         │           │
MFGProblem(                  │         ▼           │
    geometry=grid ───────────►│ Returns stored BC  │
)                            └─────────────────────┘
        │                              │
        │                              │
        ▼                              ▼
problem.get_boundary_conditions() ◄────┘
        │
        │  (delegates to geometry - SSOT)
        ▼
FPFDMSolver
        │
        │ self.bc = problem.get_boundary_conditions()
        ▼
apply_boundary_conditions_nd(u, self.bc, ...)
```

**Improvements:**
1. Geometry is Single Source of Truth (SSOT)
2. Solvers use centralized `problem.get_boundary_conditions()`
3. HJB and FP guaranteed consistent BC (same geometry)

---

## 6. Current State: What's Connected, What's Not

### 6.1 Connected (After This PR)

| Component | Connection |
|-----------|------------|
| `Geometry.get_boundary_conditions()` | Returns stored `BoundaryConditions` |
| `MFGProblem.get_boundary_conditions()` | Delegates to geometry |
| `FPFDMSolver` | Uses `problem.get_boundary_conditions()` |
| `FPParticleSolver` | Uses `problem.get_boundary_conditions()` |

### 6.2 Not Yet Connected

| Component | Current State | Future Work |
|-----------|---------------|-------------|
| `Geometry.get_boundary_handler()` | Creates handler from `bc_type` string param | Should use stored spec |
| `HJBFDMSolver` | Hard-coded one-sided differences | Should use spec |
| `HJBGFDMSolver` | Accepts BC as constructor param | Should query geometry |
| FEM applicators | Created independently | Should be created from spec |

### 6.3 Ideal Future State

```python
# Geometry stores BOTH spec and creates applicator from it
class TensorProductGrid:
    def __init__(self, ..., boundary_conditions=None):
        self._bc_spec = boundary_conditions or no_flux_bc(dimension)

    def get_boundary_conditions(self) -> BoundaryConditions:
        """Return BC specification."""
        return self._bc_spec

    def get_boundary_handler(self) -> BaseBCApplicator:
        """Return applicator CREATED FROM the stored spec."""
        return create_applicator(
            spec=self._bc_spec,
            method="fdm",
            grid_shape=self.get_grid_shape(),
            spacing=self.get_grid_spacing(),
        )
```

---

## 7. File Reference

```
mfg_pde/geometry/boundary/
│
├── conditions.py           # BoundaryConditions (SPEC) ← SSOT
├── types.py                # BCType, BCSegment
│
├── applicator_base.py      # BaseBCApplicator hierarchy
├── applicator_fdm.py       # FDM: apply_boundary_conditions_*()
├── applicator_fem.py       # FEM: MFGBoundaryHandlerFEM
├── applicator_meshfree.py  # Meshfree BC applicators
├── applicator_graph.py     # Graph/network BC
│
├── fdm_bc_1d.py           # Legacy 1D FDM BC (deprecated)
├── fem_bc_1d.py           # 1D FEM: DirichletBC1D, etc.
├── fem_bc_2d.py           # 2D FEM: DirichletBC2D, etc.
├── fem_bc_3d.py           # 3D FEM: DirichletBC3D, etc.
│
└── __init__.py            # Public exports
```

---

## 8. Summary

| Question | Answer |
|----------|--------|
| What is BC Specification? | `BoundaryConditions` - describes WHAT BC to apply |
| What is BC Applicator? | Handler classes - executes HOW to apply BC |
| Where is spec stored? | `Geometry._boundary_conditions` (SSOT) |
| How do solvers get BC? | `problem.get_boundary_conditions()` → delegates to geometry |
| Are spec and applicator connected? | Partially - FP solvers use spec; `get_boundary_handler()` still independent |
| What's left to do? | Connect `get_boundary_handler()` to use stored spec |

---

## 9. Recommendations for External Auditor

1. **Review `conditions.py`**: This is the unified BC specification - check if it covers all needed BC types

2. **Review `applicator_fdm.py`**: Primary applicator for our FDM solvers - verify correctness of `apply_boundary_conditions_nd()`

3. **Check consistency**: Ensure `BoundaryConditions.type` (legacy string) matches what applicators expect

4. **Validate SSOT pattern**: Confirm `problem.get_boundary_conditions()` correctly delegates to geometry

5. **Identify gaps**: Note which applicators don't yet use the centralized spec
