# Topology-Geometry-BC Infrastructure: Architecture Decisions

**Document ID**: MFG-ARCH-GBC-1.0
**Status**: ACTIVE
**Date**: 2026-02-14
**Companion to**: `GEOMETRY_AND_TOPOLOGY.md` (MFG-SPEC-GEO-1.0, conceptual reference)
**Tracking**: Issue #732

---

## 1. Context

MFG_PDE solves coupled HJB-FP systems on various spatial domains. The
topology-geometry-BC infrastructure is how the library represents domains,
identifies boundaries, and enforces boundary conditions across different
discretization methods.

Two approaches were evaluated — not competing, but at different levels of
abstraction:

| | Trait-Based (SPEC-GEO-1.0) | Protocol-Based (this document) |
|:----------|:---------------------------|:-------------------------------|
| **Model** | 8 atomic traits composed into `DomainSpec` | Flat `GeometryProtocol` + `GeometryType` enum |
| **Dispatch** | Compile-time trait signatures | Runtime `isinstance()` + enum matching |
| **Scope** | General scientific computing | MFG-specific with reusable tools |
| **Role** | Conceptual reference — catalog of concerns | Implementation blueprint |

SPEC-GEO-1.0 was conceived from practical experience across diverse PDE domains
and identifies fundamental concerns (metric structure, temporality, composition)
through abstract inference. This document describes how those concerns are
realized in MFG_PDE's current architecture — some in the geometry layer, some in
the BC layer, some in the coupling layer. The concerns are decomposed and
relocated, not discarded.

---

## 2. The Actual Architecture

### 2.1 Geometry Layer

Every geometry implements `GeometryProtocol` (16 members: 3 properties + 13 methods):

```
GeometryProtocol
├── Core:      dimension, geometry_type, num_spatial_points       (3 properties)
├── Grid:      get_spatial_grid(), get_grid_shape(),
│              get_collocation_points(), get_boundary_conditions() (4 methods)
├── Bounds:    get_bounds(), get_problem_config()                  (2 methods)
├── Boundary:  is_on_boundary(), get_boundary_normal(),
│              project_to_boundary(), project_to_interior()        (4 methods)
└── Regions:   get_boundary_regions(), get_boundary_indices(),
               get_boundary_info()                                 (3 methods)
```

Type identity via `GeometryType` enum:

```
GeometryType
├── CARTESIAN_GRID   → TensorProductGrid    (FDM, WENO, Semi-Lagrangian)
├── IMPLICIT         → Hyperrectangle + SDF (GFDM, meshfree)
├── NETWORK          → NetworkGeometry      (graph MFG)
├── MAZE             → MazeGeometry         (grid with obstacles)
├── DOMAIN_2D        → Mesh2D              (future FEM)
├── DOMAIN_3D        → Mesh3D              (future FEM)
└── CUSTOM           → user extension
```

**Design choice**: Flat enum, not trait composition. Justification in Section 5.

### 2.2 BC Specification Layer

Dimension-agnostic, solver-independent.

```
BCType (8 values)
├── DIRICHLET              u = g
├── NEUMANN                ∂u/∂n = g
├── ROBIN                  αu + β∂u/∂n = g
├── PERIODIC               u(x_min) = u(x_max)
├── NO_FLUX                J·n = 0 (field methods)
├── REFLECTING             elastic reflection (particle methods)
├── EXTRAPOLATION_LINEAR   d²u/dx² = 0 at boundary
└── EXTRAPOLATION_QUADRATIC d³u/dx³ = 0 at boundary

BCSegment — specifies WHERE a BC applies
├── boundary: str | None    "x_min", "left", etc.
├── region: dict | None     {"y": (4.0, 6.0)}
├── sdf_region: callable    phi(x) < 0 means in segment
├── normal_direction: array  match by outward normal
└── region_name: str | None  references geometry.mark_region()

BoundaryConditions — collection of BCSegments + dimension
```

**Value resolution**: static `float` | `callable(x, t)` | `BCValueProvider` protocol.
Providers enable state-dependent BCs (e.g., `AdjointConsistentProvider` for coupled
HJB-FP systems, resolved each Picard iteration).

### 2.3 BC Application Layer

7 applicators, dispatched by discretization type:

```
                           Solver Imports  Total Refs  Status
pad_array_with_ghosts()    1 (base_hjb)    11 files    Primary API (function-based)
FDMApplicator              2 (fdm, sl)     11 files    Production
InterpolationApplicator    1 (sl)          3 files     Production (Semi-Lagrangian)
ParticleApplicator         1 (fp_part)     3 files     Production (FP particle)
MeshfreeApplicator         0               9 files     Implemented, awaiting GFDM wiring
FEMApplicator              0               4 files     Implemented, awaiting Issue #773
GraphApplicator            0               4 files     Implemented, awaiting network solver
ImplicitApplicator         0               3 files     Implemented, awaiting SDF solver
```

"Solver Imports" = files under `alg/` that import the symbol.
"Total Refs" = all files referencing the symbol (includes re-exports, internal cross-refs).

Dispatch: `dispatch.py` routes `(GeometryType, DiscretizationType) → Applicator`.

### 2.4 Internal: Topology/Calculator Decomposition

Protocols and implementations defined in `applicator_base.py`, used by
`applicator_fdm.py` (not exposed to solvers):

```
Layer 1 — Topology (connectivity)
├── PeriodicTopology     → wrap-around indices, no ghost values needed
└── BoundedTopology      → physical edges, delegates to Calculator

Layer 2 — Calculator (ghost value physics)
├── DirichletCalculator          u_ghost = 2g - u_interior
├── NeumannCalculator            u_ghost = u_interior + 2·dx·g
├── RobinCalculator              αu + β∂u/∂n = g → solve for u_ghost
├── ZeroGradientCalculator       u_ghost = u_interior
├── ZeroFluxCalculator           u_ghost = u · (2D + v·dx)/(2D - v·dx)
├── LinearExtrapolationCalculator    u_ghost = 2u₀ - u₁
└── QuadraticExtrapolationCalculator u_ghost = 3u₀ - 3u₁ + u₂
```

Bridge to implicit solvers: `LinearConstraint` converts Calculator output to
matrix assembly coefficients (Tier-Based Coefficient Folding).

---

## 3. Architecture Overview

The spec's 8 concerns are distributed across 4 layers. Each layer has
defined extension points for advanced features (Lipschitz BC, PDE constraints,
free boundaries) without structural changes.

```
╔══════════════════════════════════════════════════════════════════╗
║  LAYER 4: COUPLING                                              ║
║  Spec concerns: CompositeDomain, Temporality (evolution)        ║
║                                                                  ║
║  GeneralFixedPointIterator                                       ║
║  ┌───────┐    ┌────────────┐    ┌───────┐    ┌──────────────┐  ║
║  │  HJB  │───▶│ Projection │───▶│  FP   │───▶│  Constraint  │  ║
║  │Solver │    │ Coupling   │    │Solver │    │  PDE Solver  │  ║
║  └───────┘    └────────────┘    └───────┘    └──────────────┘  ║
║                     │                                            ║
║        ┌────────────┴────────────┐                              ║
║        │    Domain Evolution     │  (free boundary tracking)    ║
║        │    GeometryCoupling     │  (multi-domain restrict)     ║
║        └─────────────────────────┘                              ║
╠══════════════════════════════════════════════════════════════════╣
║  LAYER 3: BOUNDARY CONDITIONS (4-axis orthogonal)               ║
║  Spec concerns: Boundary, Metric (at ∂M)                       ║
║                                                                  ║
║  WHAT ── BCType (8 values)                                       ║
║          Dirichlet│Neumann│Robin│Periodic│NoFlux│Reflecting│... ║
║                                                                  ║
║  WHERE ─ BCSegment                                               ║
║          str|int │ region dict │ sdf_region(callable) │ normal  ║
║                                                                  ║
║  WHEN ── Value Resolution Chain                                  ║
║          float ──▶ callable(x,t) ──▶ BCValueProvider            ║
║          Decorators: LipschitzProjection ◀── BoundaryMetric     ║
║                      AdjointConsistent   ◀── coupled state      ║
║                      StefanCondition     ◀── InterfaceJump      ║
║                                                                  ║
║  HOW ─── Applicators (by discretization)                         ║
║          FDM (Topology×Calculator) │ FEM │ Meshfree │ Particle  ║
║          Interpolation │ Graph │ Implicit                        ║
╠══════════════════════════════════════════════════════════════════╣
║  LAYER 2: GEOMETRY                                               ║
║  Spec concerns: Connectivity, Structure, Embedding, Temporality  ║
║                                                                  ║
║  GeometryProtocol (16 members)        GeometryType (7 values)   ║
║  ├── dimension, geometry_type         ├── CARTESIAN_GRID         ║
║  ├── get_collocation_points()         ├── IMPLICIT               ║
║  ├── get_bounds()                     ├── NETWORK / MAZE         ║
║  ├── is_on_boundary()                 ├── DOMAIN_2D / 3D         ║
║  └── ...                              └── CUSTOM                 ║
║                                                                  ║
║  TimeDependentDomain (wrapper)                                   ║
║  ├── LevelSetEvolver     ∂φ/∂t + V|∇φ| = 0                     ║
║  ├── LevelSetFunction    normals, curvature                      ║
║  └── Reinitialization    |∇φ| = 1                               ║
║                                                                  ║
║  Migration path: flat enum → trait dispatch (Section 5.5)        ║
╠══════════════════════════════════════════════════════════════════╣
║  LAYER 1: BACKEND                                                ║
║  Spec concerns: Data Layout, Distribution                        ║
║                                                                  ║
║  NumPy (SoA)  │  CuPy (GPU)  │  JAX (JIT)  │  MPI (deferred)   ║
╚══════════════════════════════════════════════════════════════════╝
```

Concern placement summary (see Section 5.3 for audit evidence and rationale):

| Spec Concern | Layer | Current Form | Realization |
|:-------------|:------|:-------------|:------------|
| Connectivity | Geometry | `GeometryType` enum | → trait enum (Stage 1) |
| Structure | Geometry | `GeometryType` enum | → trait enum (Stage 1) |
| Boundary | Geometry + BC | `GeometryType` + `BCSegment` | → trait enum (Stage 1) |
| Embedding | Geometry | `get_collocation_points()` | Protocol (sufficient) |
| Metric | BC | `BoundaryMetric` (Lipschitz) | Deferred (scoped trigger) |
| Temporality | Geometry + Coupling | `TimeDependentDomain` | Done (composition) |
| Data Layout | Backend | NumPy SoA / GPU backends | Backend (not geometry) |
| Distribution | Backend | Deferred (MPI) | Backend (not geometry) |
| CompositeDomain | Coupling | `GeometryCoupling` protocol | Coupling layer |
| Constraints | Implicit | Flat enum | → `DomainSpec` (Stage 2) |

---

## 4. The 4-Axis Orthogonal Design

The BC infrastructure decomposes into 4 independent axes:

```
WHAT   (BCType)        → Dirichlet | Neumann | Robin | Periodic | ...
WHERE  (BCSegment)     → boundary str | region dict | SDF | normal | mesh tag
WHEN   (Value source)  → static float | callable(x,t) | BCValueProvider
HOW    (Applicator)    → FDM ghost | FEM matrix | Particle reflect | ...
```

### Orthogonality matrix

```
          WHAT     WHERE     WHEN      HOW
WHAT       —        ✓         ○         ○
WHERE      ✓        —         ✓         △
WHEN       ○        ✓         —         ✓
HOW        ○        △         ✓         —

✓ = fully independent (change one, zero changes in other)
○ = mostly independent (1 semantic exception)
△ = structural coupling (fixable)
```

### Known coupling points

1. **WHAT × HOW**: `REFLECTING` only pairs with particle enforcement. Physical
   coupling — cannot be eliminated, correctly documented in `BCType` docstring.

2. **WHAT × HOW**: `ZeroFluxCalculator` embeds Fokker-Planck physics (drift,
   diffusion). Pragmatic coupling — the formula is too specific for solvers to
   own without duplication.

3. **WHERE × HOW**: FEM uses integer mesh tags from Gmsh, not string boundary
   names. **Structural gap** — fix by extending `BCSegment.boundary: str | int`.

4. **WHAT × WHEN**: Periodic BCs don't use values (topological, not value-based).
   Harmless — default `value=0.0` is ignored.

### Practical test of orthogonality

| Add... | Spec layer changes | Applicator changes |
|:-------|:-------------------|:-------------------|
| New BCType | Add enum + Calculator | Each applicator adds 1 case |
| New region mode | Extend `BCSegment` | Zero |
| New value source | Implement `BCValueProvider` | Zero |
| New applicator | Zero | New file |

Adding region modes or value sources requires zero applicator changes. Adding
applicators requires zero specification changes. Only BCType additions touch
both layers — and that's the axis that changes least often.

---

## 5. Relationship to GEOMETRY_AND_TOPOLOGY.md

### 5.1 What the spec proposed

8 atomic traits composed into a `DomainSpec` dataclass:

| Trait | Values |
|:------|:-------|
| Connectivity | Implicit, Explicit, Dynamic |
| Structure | Structured, Unstructured |
| Embedding | Grid, Free, Abstract |
| Metric | Euclidean, Manifold, Graph |
| Boundary | Box, Mesh, Implicit, None |
| Temporality | Static, GeometricDynamic, TopologicalDynamic |
| Data Layout | AoS, SoA |
| Distribution | Local, Distributed |

Plus constraint validation (5 rules), kernel factory (trait-based dispatch),
and recursive composition (CompositeDomain for Chimera, MultiphysicsDomain
for FSI).

### 5.2 What we adopted directly

**The vocabulary.** The spec's trait names (Connectivity, Boundary, Structure)
are the right words for the distinctions that matter. We use these concepts
informally in dispatch logic and documentation.

**The orthogonality principle.** Separating independent axes of variation is
correct. We applied it to the BC infrastructure (4 axes) rather than to
geometry definition (8 traits).

### 5.3 How the spec's concerns are realized

The spec identified 8 concerns plus composition. All 8 are valid — they arise
in practice. The question is not whether they matter, but where they live in
the architecture. The spec colocates them in the geometry layer; our design
distributes them across three layers.

A codebase audit (2026-02-14) cross-referenced each trait against actual
dispatch sites, solver requirements, and pain points across 22 geometry classes
and 10 solvers:

| Spec Concern | Where it lives | Realization | Status |
|:-------------|:---------------|:------------|:-------|
| **Connectivity** | Geometry layer | `GeometryType` enum → **trait enum** (Stage 1) | **REALIZE** — 4 dispatch sites need this; `hjb_fdm.py:247` hardcodes `isinstance(TensorProductGrid)` instead of asking "implicit connectivity?" |
| **Structure** | Geometry layer | `GeometryType` enum → **trait enum** (Stage 1) | **REALIZE** — mass normalization, `get_grid_shape()` branching, problem init all check this indirectly |
| **Boundary** | Geometry + BC layers | `GeometryType` + `BCSegment` → **trait enum** (Stage 1) | **REALIZE** — applicator dispatch uses `DiscretizationType` as proxy for boundary type; `_has_implicit_boundary()` is a workaround |
| **Embedding** | Geometry layer | Collapsed into `get_collocation_points()` | Don't realize — no dispatch site cares HOW coordinates are produced; the protocol abstracts this |
| **Metric** | BC layer | Scoped: `BoundaryMetric` protocol for Lipschitz constraints (Section 5.6 Case 1) | Defer — all spatial geometries Euclidean, graphs use graph metric. Scoped trigger when non-Euclidean MFG arises |
| **Temporality** | Geometry wrapper | `TimeDependentDomain` + `LevelSetEvolver` (composition). `DomainEvolution` coupling hook (Section 5.6 Case 3) | **Done** — composition is the right pattern for Python |
| **Data Layout** | Backend system | NumPy is SoA; GPU backends (CuPy/JAX) handle memory layout | Don't realize — not a geometry concern in Python |
| **Distribution** | Deferred | No MPI in current scope | Don't realize — backend concern when needed |
| **CompositeDomain** | Coupling layer | `GeometryCoupling` protocol in `FixedPointIterator` (restrict/prolong). PDE-constrained MFG uses same pattern (Section 5.6 Case 2) | Don't realize separately — coupling layer handles this |
| **Constraint validation** | Implicit (flat enum) | Reconverges with spec via `DomainSpec.__post_init__` at Stage 2 (Section 5.5) | Stage 2 — when trait enums are adopted |

**Summary**: 3 traits to realize as enums (Connectivity, Structure, Boundary),
1 already realized via composition (Temporality), 4 handled by non-enum
mechanisms (Embedding by protocol, Metric scoped, Data Layout and Distribution
by backend). Composition handled in coupling layer.

**Two complementary dispatch mechanisms.** The codebase already has 16
`Supports*` capability protocols (`SupportsGradient`, `SupportsLaplacian`,
`SupportsGraphLaplacian`, etc. in `geometry/protocols/`) used by 4 solver
init checks. These answer: *"What can this geometry DO?"* The spec's traits
answer a different question: *"What IS this geometry?"* Both are needed:

```
Traits (identity)        → Factory routing: "which solver for this geometry?"
Capabilities (contract)  → Solver validation: "does this geometry support ∇u?"
```

The spec's kernel factory (SPEC-GEO-1.0 Section 7.2) describes exactly this two-layer
dispatch. Currently we have the capability layer (16 protocols) but not the
trait layer — so dispatch falls back to enum matching or concrete class checks.
Stage 1 (Section 5.5) adds the trait layer.

**What we simplified** (implementation form, not concept):

| Spec Form | Our Form | Reason |
|:----------|:---------|:-------|
| Compile-time trait dispatch | Runtime `isinstance()` + enum | Python has no compile-time dispatch |
| 8 traits as type parameters | 3 trait enums + protocol methods + backend | Only 3 traits drive actual dispatch; others are handled by different mechanisms |
| Kernel factory class hierarchy | `dispatch.py` + 16 `Supports*` protocols | Same capability, Python-native |

### 5.4 Trigger status

Three triggers were defined for adopting trait-based dispatch. The 2026-02-14
audit found that **trigger #2 has fired** and #3 is close:

1. **A new geometry doesn't fit `GeometryType`** — NOT YET. All 22 current
   geometry classes map cleanly to one `GeometryType` value. A curvilinear grid
   (Structured + Implicit + ManifoldMetric) would fire this trigger.

2. **Solver code needs capability dispatch** — **FIRED.** `hjb_fdm.py:247`
   checks `isinstance(problem.geometry, TensorProductGrid)` — a concrete class
   check that should be "has implicit connectivity + structured layout?"
   Any custom structured grid is wrongly rejected. Mass normalization
   (`components.py:702`) dispatches on `GeometryType.CARTESIAN_GRID` when it
   actually needs "is structured?" Auto-mode solver selection
   (`scheme_factory.py:263`) is a TODO stub that needs trait-based routing.

3. **Dispatch chain grows beyond ~10 branches** — APPROACHING. Currently 9
   `GeometryType` checks across 3 files. Auto-mode implementation (Phase 3
   TODO) will add more. The FEM solver (Issue #773) will add mesh-specific
   dispatch.

**Conclusion**: The flat `GeometryType` enum remains useful as a fallback, but
trait enums should be introduced per Stage 1 (Section 5.5) to replace concrete
class checks and enable capability-based routing.

### 5.5 Migration path: Protocol → Traits

The current design is forward-compatible with the trait-based approach. Each
stage below is additive — no existing code needs to be rewritten. The stages
can be adopted independently and incrementally. **Stage 1 is recommended**
based on the trigger analysis in Section 5.4.

**Stage 1: Three trait enums + opt-in protocol** (RECOMMENDED)

The audit identified 3 traits that drive actual dispatch decisions. Add enums
for these 3 only (not all 8) plus an opt-in protocol. Does NOT modify
`GeometryProtocol`:

```python
# geometry/traits.py (NEW)

# --- Enums ---

class ConnectivityType(Enum):
    IMPLICIT = "implicit"    # stride arithmetic (TensorProductGrid)
    EXPLICIT = "explicit"    # stored adjacency (Mesh2D, NetworkGeometry)
    DYNAMIC  = "dynamic"    # runtime search (ImplicitGeometry, meshfree)

class StructureType(Enum):
    STRUCTURED   = "structured"    # logical (i,j,k) coordinates
    UNSTRUCTURED = "unstructured"  # flat point cloud

class BoundaryDef(Enum):
    BOX      = "box"       # axis-aligned hyper-rectangular (AABB)
    MESH     = "mesh"      # boundary elements (facets from Gmsh)
    IMPLICIT = "implicit"  # SDF: phi(x) = 0
    NONE     = "none"      # no boundary (periodic, graph, open)

# --- Individual trait protocols (one per trait, composable) ---

@runtime_checkable
class ConnectivityAware(Protocol):
    @property
    def connectivity_type(self) -> ConnectivityType: ...

@runtime_checkable
class StructureAware(Protocol):
    @property
    def structure_type(self) -> StructureType: ...

@runtime_checkable
class BoundaryAware(Protocol):
    @property
    def boundary_def(self) -> BoundaryDef: ...

# --- Composite protocol (all current traits) ---

@runtime_checkable
class TraitAwareGeometry(ConnectivityAware, StructureAware, BoundaryAware, Protocol):
    """Geometry implementing all current trait properties."""
    ...
```

Individual protocols enable fine-grained dispatch — a solver can check just
the trait it needs without requiring the geometry to implement all traits:

```python
# Fine-grained: only check what you need
if isinstance(geometry, StructureAware):
    if geometry.structure_type == StructureType.STRUCTURED:
        field = field.reshape(grid_shape)

# Composite: check all traits at once
if isinstance(geometry, TraitAwareGeometry):
    spec = DomainSpec.from_geometry(geometry)
```

Adding future traits (Metric, Embedding, ...) requires only:
1. New enum + new individual protocol (additive)
2. New composite that extends `TraitAwareGeometry` (additive)
3. Geometry classes add 1 property each (no existing properties change)

Geometry classes opt in one at a time (3 trivial properties each):

```python
class TensorProductGrid:
    @property
    def connectivity_type(self) -> ConnectivityType:
        return ConnectivityType.IMPLICIT
    @property
    def structure_type(self) -> StructureType:
        return StructureType.STRUCTURED
    @property
    def boundary_def(self) -> BoundaryDef:
        return BoundaryDef.BOX
```

Trait signatures for all current geometry families:

```
TensorProductGrid     Implicit   Structured    Box
Mesh1D/2D/3D          Explicit   Unstructured  Mesh
Hyperrectangle + CSG  Dynamic    Unstructured  Implicit
NetworkGeometry       Explicit   Unstructured  None
MazeGeometry          Explicit   Unstructured  None
```

Dispatch uses trait-first with enum fallback:

```python
if isinstance(geometry, TraitAwareGeometry):
    if geometry.connectivity_type == ConnectivityType.IMPLICIT:
        field = field.reshape(grid_shape)
elif geometry.geometry_type == GeometryType.CARTESIAN_GRID:
    field = field.reshape(grid_shape)  # legacy fallback
```

Concrete dispatch improvements once Stage 1 is adopted:

| Current Code | Problem | With Traits |
|:---|:---|:---|
| `isinstance(geo, TensorProductGrid)` (hjb_fdm.py:247) | Rejects valid structured grids | `geo.structure_type == STRUCTURED` |
| `geo_type == CARTESIAN_GRID` (mass normalization) | Only Cartesian gets trapezoidal | `geo.structure_type == STRUCTURED` |
| `domain_type == "grid"` (problem init) | String-based, fragile | `spec.structure == STRUCTURED` |
| `_has_implicit_boundary()` (dispatch.py) | getattr workaround | `geo.boundary_def == IMPLICIT` |
| Auto-mode solver selection (stub) | Always returns FDM | Route by `connectivity_type` |

Breaking changes: **zero**. `GeometryProtocol` is untouched.

**Stage 2: DomainSpec inference**

When multiple dispatch sites benefit from traits, add a spec wrapper:

```python
@dataclass(frozen=True)
class DomainSpec:
    connectivity: ConnectivityType
    structure: StructureType
    boundary: BoundaryDef

    @classmethod
    def from_geometry(cls, geometry: GeometryProtocol) -> DomainSpec:
        """Infer traits — works for both trait-aware and legacy geometries."""
        if isinstance(geometry, TraitAwareGeometry):
            return cls(
                connectivity=geometry.connectivity_type,
                structure=geometry.structure_type,
                boundary=_infer_boundary(geometry),
            )
        # Legacy: infer from GeometryType enum
        _MAP = {
            GeometryType.CARTESIAN_GRID: (ConnectivityType.IMPLICIT,
                                          StructureType.STRUCTURED, BoundaryDef.BOX),
            GeometryType.NETWORK: (ConnectivityType.EXPLICIT,
                                   StructureType.UNSTRUCTURED, BoundaryDef.NONE),
            GeometryType.IMPLICIT: (ConnectivityType.DYNAMIC,
                                    StructureType.UNSTRUCTURED, BoundaryDef.IMPLICIT),
            GeometryType.DOMAIN_2D: (ConnectivityType.EXPLICIT,
                                     StructureType.UNSTRUCTURED, BoundaryDef.MESH),
        }
        c, s, b = _MAP[geometry.geometry_type]
        return cls(c, s, b)

    def __post_init__(self):
        # Constraint validation (from SPEC-GEO-1.0 Section 5)
        if (self.connectivity == ConnectivityType.IMPLICIT
                and self.structure == StructureType.UNSTRUCTURED):
            raise ValueError("Implicit connectivity requires structured grids")
```

Breaking changes: **zero**. New class wrapping existing objects.

**Stage 3: Gradual dispatch migration**

Replace `GeometryType` checks one site at a time:

```python
# Before:
if geometry.geometry_type == GeometryType.CARTESIAN_GRID:
    ...
elif geometry.geometry_type in (GeometryType.DOMAIN_2D, GeometryType.DOMAIN_3D):
    ...

# After:
spec = DomainSpec.from_geometry(geometry)
if spec.structure == StructureType.STRUCTURED:
    ...
elif spec.boundary == BoundaryDef.MESH:
    ...
```

Both dispatch mechanisms coexist indefinitely. Migration is file-by-file.

**Stage 4: Composition (if ever needed)**

New classes that compose existing geometries:

```python
class CompositeDomain:
    """Multiple overlapping domains (only if multi-scale MFG demands it)."""
    def __init__(self, domains: list[GeometryProtocol]):
        self.domains = domains
        self.specs = [DomainSpec.from_geometry(d) for d in domains]
```

Completely additive — zero changes to existing code.

**Summary**: The `GeometryType` flat enum never needs to be removed. It
becomes the fallback layer inside `DomainSpec.from_geometry()`. The pattern
is "strangler fig" — the trait system grows around the enum, gradually
replacing dispatch responsibilities, but the enum remains functional
throughout. If traits are never needed, nothing is wasted.

### 5.6 Extensibility: Non-local constraints and coupling projections

The 4-axis design accommodates advanced BC/coupling patterns without structural
changes. Three representative cases are analyzed below to validate the extension
points.

**Case 1: Lipschitz BC on manifold**

*Requirement*: Boundary values satisfy |u(x) - u(y)| ≤ L · d_∂M(x, y) for all
x, y on the manifold boundary ∂M, where d_∂M is geodesic distance.

*Challenge*: Non-local — couples all boundary points, unlike pointwise BCTypes.

*Fits via*: Compositional WHEN axis. `LipschitzProjection` wraps any
`BCValueProvider`, projecting resolved values onto the Lipschitz-continuous set:

```python
class LipschitzProjection(BCValueProvider):
    """Decorator: projects base provider's output to L-Lipschitz."""
    def __init__(self, base: BCValueProvider, L: float, metric: BoundaryMetric):
        self.base, self.L, self.metric = base, L, metric

    def resolve(self, state: IterationState) -> np.ndarray:
        raw = self.base.resolve(state)
        # QP: min ||v - raw||²  s.t. |v_i - v_j| ≤ L·d(i,j)
        return project_to_lipschitz(raw, self.L, self.metric)
```

Usage: `BCSegment(bc_type=BCType.DIRICHLET, value=LipschitzProjection(...))`.

*New infrastructure needed*: `BoundaryMetric` protocol (geodesic distance on ∂M).
Additive — zero changes to `GeometryProtocol`. This is the concrete trigger for
the spec's `ManifoldMetric` trait (SPEC-GEO-1.0), scoped to the boundary only.

*Axes affected*: None modified. WHAT remains Dirichlet/Neumann/Robin; WHERE and
HOW unchanged. The constraint composes on WHEN as a provider decorator.

**Case 2: ODE-PDE constraint as projection**

*Requirement*: PDE solution must be consistent with ODE characteristics
dX/dt = −∇_p H(X, m, ∇u). After solving HJB, project u onto the
characteristic-consistent set.

*Challenge*: Volumetric (interior + boundary), not just a boundary condition.

*Fits via two mechanisms*:

1. **Boundary flux** (BC infrastructure): `CharacteristicFluxProvider` implements
   `BCValueProvider`, computing m·v*·n from ODE-traced characteristics. Uses
   existing `HamiltonianBase.dp()`. Zero new infrastructure.

2. **Interior projection** (coupling layer): Optional step in `FixedPointIterator`
   between HJB and FP solves:

```python
class ProjectionCoupling(Protocol):
    """Projects PDE solution onto ODE-consistent set."""
    def project(self, u: NDArray, m: NDArray,
                geometry: GeometryProtocol) -> NDArray: ...

# In FixedPointIterator:
u = self.hjb_solver.solve(m)
if self._projection is not None:
    u = self._projection.project(u, m, self.geometry)  # ODE consistency
m = self.fp_solver.solve(u)
```

*New infrastructure needed*: `ProjectionCoupling` protocol (additive, one-parameter
extension to iterator). The Semi-Lagrangian solver (`hjb_semi_lagrangian.py`)
already implements characteristic tracing — this generalizes it as a post-processor
applicable to any HJB solver.

*Axes affected*: None. The projection lives in the coupling layer, outside the
BC infrastructure.

**Case 3: Free boundary / propagation of frontier**

*Requirement*: The domain boundary is part of the solution — it moves at a velocity
determined by the PDE solution (e.g., Stefan problem, density support frontier in
MFG congestion models).

*Challenge*: The WHERE axis is time-dependent. Boundary segments move each timestep.

*Fits via existing infrastructure*:

- **WHERE**: `BCSegment.sdf_region` accepts a callable evaluated at query time.
  Capturing a reference to the evolving level set makes the boundary dynamic:
  ```python
  td_domain = TimeDependentDomain(initial_sdf, grid)
  BCSegment(sdf_region=lambda x: td_domain.current_level_set(x), ...)
  ```
  Each call to `matches_point()` sees the current boundary location.

- **WHEN**: `BCValueProvider` computes interface velocity from solution state
  (e.g., Stefan condition V_n = −κ[∂T/∂n] via `InterfaceJumpOperator`).

- **HOW**: `ImplicitApplicator` (exists, 0 solver imports — Tier 3 wiring).

- **Geometry**: `TimeDependentDomain` + `LevelSetEvolver` (2,119 lines, 53 tests,
  production-ready). Stefan problem examples validate at 4.58% error vs analytical.

*New infrastructure needed*: `DomainEvolution` protocol for automatic domain
evolution in `FixedPointIterator` — same coupling-layer hook pattern as Cases 1–2.
Without it, users must manually call `td_domain.evolve_step()` each iteration.

*Axes affected*: None. The callable `sdf_region` already supports dynamic WHERE.

**MFG-specific sub-case — density support frontier**: The boundary of supp(m)
(where m > 0 meets m = 0) is implicit in the FP equation for degenerate diffusion.
No explicit boundary tracking needed. If conditions at the frontier are required,
a `DensitySupportProvider(BCValueProvider)` resolves them from the current density.

**Architectural conclusion**: All three cases — non-local constraints (Case 1),
volumetric coupling projections (Case 2), and free boundaries (Case 3) — compose
through three extension points that already exist or are additive:

1. `BCValueProvider` protocol (WHEN axis — compositional decorators)
2. `FixedPointIterator` pipeline (coupling layer — projection/evolution hooks)
3. `BCSegment.sdf_region` callable (WHERE axis — dynamic boundary tracking)

The 4-axis design does not need restructuring. Notably, the three cases
validated concerns originally identified in SPEC-GEO-1.0: ManifoldMetric
resurfaced as `BoundaryMetric` (Case 1), Temporality resurfaced as
`TimeDependentDomain` + `DomainEvolution` (Case 3), and MultiphysicsDomain
resurfaced as `GeometryCoupling` in the iterator (Case 2, general PDE
constraints). The spec correctly identified these concerns through abstract
inference — the difference is architectural placement (distributed across
BC/coupling layers) rather than substance.

The migration path in Section 5.5 provides a bridge if these distributed
concerns need to reconverge into explicit geometry traits.

### 5.7 Extensibility audit

Each extension point in the infrastructure was audited (2026-02-14) for whether
adding new components is **additive** (zero changes to existing code) or
**requires modification**.

**Additive (zero existing code changes):**

| Extension Point | Mechanism | How to Extend |
|:----------------|:----------|:--------------|
| New geometry type | `GeometryType.CUSTOM` + protocol | Implement `GeometryProtocol`, use CUSTOM or add enum value |
| New capability | `Supports*` protocol | New `@runtime_checkable` protocol in `geometry/protocols/` |
| New trait (future) | Individual trait protocol | New enum + `@runtime_checkable` protocol; compose into `TraitAwareGeometry` |
| New BC value source | `BCValueProvider` protocol | Implement `compute(state)` method; pass as `BCSegment.value` |
| New BC decorator | Provider wrapping | Wrap existing `BCValueProvider` (e.g., `LipschitzProjection`) |
| New config type | Pydantic `BaseModel` | Subclass `BaseModel`; solver accepts via `config` parameter |
| Extra geometry methods | Protocol extension | Add methods to geometry class; solvers use `getattr()` + `callable()` |
| `DomainSpec` fields | Default values | Add field with default to frozen dataclass; existing callers unaffected |

**Requires modification (friction points):**

| Extension Point | What Changes | Mitigation |
|:----------------|:-------------|:-----------|
| New BCType | Enum + Calculator + every applicator (5 modules) | Low change frequency; Calculator pattern isolates ghost-value physics |
| New applicator | `dispatch.py` if/elif chain (3 locations) | → Registry pattern: `dict[DiscretizationType, type[Applicator]]` |
| New DiscretizationType | `dispatch.py` + new applicator module | Same registry mitigation |
| Iterator general hooks | No hook system beyond `BCValueProvider` resolution | → `IterationHook` protocol with `on_step_start/end` |

**Key findings:**

1. **Protocols are the primary extensibility mechanism.** `BCValueProvider`,
   `Supports*` traits, and the new `ConnectivityAware/StructureAware/BoundaryAware`
   protocols are all `@runtime_checkable` — users extend by implementing, not
   by registering. This is the correct pattern for Python.

2. **dispatch.py is the main friction point.** Hardcoded if/elif chains mean
   adding a new applicator or discretization type requires modifying dispatch.py
   in 3 places. A registry pattern (`dict` mapping enum → class) would make
   this additive. Not urgent at 7 applicators, but worth adopting before #773
   adds FEM.

3. **BCType is intentionally hard to extend.** Adding a new BC type requires
   updating every applicator — this is correct because each discretization
   method enforces BCs differently (ghost cells, matrix assembly, projection).
   The Topology/Calculator decomposition in `applicator_base.py` already
   isolates the physics; adding a new Calculator is ~30 lines. The 8 current
   BCTypes cover the mathematical space well.

4. **FixedPointIterator needs a hook protocol.** BC provider resolution works
   via `problem.using_resolved_bc(state)`, but there's no general mechanism
   for injecting coupling logic (projection, domain evolution, constraint
   enforcement) between solver steps. The 3 coupling protocols from Section 5.6
   (`ProjectionCoupling`, `DomainEvolution`, `GeometryCoupling`) should be
   wired as iterator hooks when implemented.

---

## 6. Action Items (from Issue #732)

### Tier 1: Do Now

| Change | Lines | Why |
|:-------|------:|:----|
| `BCSegment.boundary: str \| int` | ~50 | FEM mesh tags need integer IDs |
| Complete `LinearConstraint` Robin case | ~15 | Bridge ghost-cell↔matrix assembly |
| Wire `calculator_to_constraint` into FP-FDM | ~20 | Currently exported but uncalled |

### Tier 2: Simplify

| Change | Why |
|:-------|:----|
| Reduce `__init__.py` exports (~100 → ~60) | Remove legacy aliases |
| Mark Topology/Calculator as internal | Implementation detail of FDM applicator |

### Tier 3: Wire When Solver Arrives

| Applicator | Trigger |
|:-----------|:--------|
| MeshfreeApplicator → hjb_gfdm.py | GFDM solver refactor |
| FEMApplicator → FEM solver | Issue #773 (scikit-fem) |
| GraphApplicator → network solver | When built |

### Tier 1b: Trait Enums (Section 5.5 Stage 1)

| Change | Lines | Why |
|:-------|------:|:----|
| `geometry/traits.py` with 3 enums + 3 individual protocols + 1 composite | ~80 | `ConnectivityType`, `StructureType`, `BoundaryDef` + `ConnectivityAware`, `StructureAware`, `BoundaryAware`, `TraitAwareGeometry` |
| Add 3 properties to each geometry family (5 families) | ~5 each | Opt-in, trivial (return enum literal) |
| Replace `isinstance(TensorProductGrid)` in hjb_fdm.py | ~10 | Unblocks custom structured grids |

### Tier 2b: Dispatch Registry (Section 5.7 mitigation)

| Change | Lines | Why |
|:-------|------:|:----|
| Replace if/elif in `dispatch.py` with registry dict | ~40 | Additive applicator registration before FEM (#773) |

### Don't Build Now

- SpacetimeBoundaryData — simple `@dataclass`, worth ~1 day when convenient (see
  `spacetime-bc/PROJECT_EVALUATION.md`). Not blocking any solver.
- TrajectorySolver protocol (no second implementation)
- Pluggable time integrators (MFG has fixed temporal structure)
- Remaining 5 trait enums (Embedding, Metric, Temporality, Data Layout,
  Distribution) — handled by other mechanisms (Section 5.3)

---

## 7. File Map

Key files only — full geometry package has 60+ files across subdirectories.

```
mfg_pde/geometry/
├── protocol.py                  GeometryProtocol, GeometryType, BoundaryType
├── traits.py                    ConnectivityType, StructureType, BoundaryDef, TraitAwareGeometry (NEW)
├── base.py                      BaseGeometry (convenience properties)
├── grids/tensor_grid.py         TensorProductGrid (CARTESIAN_GRID)
├── implicit/hyperrectangle.py   Hyperrectangle (IMPLICIT)
├── meshes/mesh_2d.py            Mesh2D (DOMAIN_2D)
├── graph/network_geometry.py    NetworkGeometry (NETWORK, MAZE)
└── boundary/
    ├── types.py                 BCType, BCSegment
    ├── conditions.py            BoundaryConditions, factory functions
    ├── providers.py             BCValueProvider, AdjointConsistentProvider
    ├── dispatch.py              apply_bc(), get_applicator_for_geometry()
    ├── applicator_base.py       Topology/Calculator protocols, BaseBCApplicator hierarchy
    ├── applicator_fdm.py        FDMApplicator, pad_array_with_ghosts (PRODUCTION)
    ├── applicator_fem.py        FEMApplicator (awaiting #773)
    ├── applicator_meshfree.py   MeshfreeApplicator (awaiting wiring)
    ├── applicator_graph.py      GraphApplicator (awaiting wiring)
    ├── applicator_implicit.py   ImplicitApplicator (awaiting wiring)
    ├── applicator_interpolation.py  InterpolationApplicator (PRODUCTION)
    ├── applicator_particle.py   ParticleApplicator (PRODUCTION)
    ├── enforcement.py           Shared Dirichlet/Neumann/Robin/Periodic enforcement
    ├── bc_coupling.py           Adjoint-consistent BC (MFG-specific)
    └── corner/                  Corner handling strategies (directory: position, velocity, strategies)
```

---

**Key principle**: Don't build more infrastructure. Wire existing infrastructure
into solvers, one at a time.
