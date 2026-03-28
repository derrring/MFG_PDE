# Boundary Condition Architecture Overview

**Status**: Active reference — updated 2026-03-28
**Consolidates**: `boundary_condition_handling_summary.md`, `BOUNDARY_HANDLING.md`
**Future direction**: See `BC_ENFORCEMENT_ARCHITECTURE.md` for the 4-layer model

---

## 1. Current Architecture Assessment

### What Works Well

**Data model** (`BCSegment`, `BoundaryConditions`):
- Multiple matching modes: boundary ID, coordinate ranges, SDF, normal direction, region names
- Priority-based resolution for overlapping segments
- Lazy dimension binding (BC created before dimension is known)
- Callable value support for time/space-dependent BCs

**Provider pattern** (Issue #625, v0.18.0+):
- `BCValueProvider` protocol separates intent from concrete values
- `AdjointConsistentProvider` computes Robin BC from density: $\partial U/\partial n = -\sigma^2/2 \cdot \partial \ln(m)/\partial n$
- Iterator resolves providers; solvers stay generic

```python
# Provider stores intent, iterator resolves at each step
segment = BCSegment(
    bc_type=BCType.ROBIN, alpha=0.0, beta=1.0,
    value=AdjointConsistentProvider(side="left", sigma=0.2),
)
# In coupling loop:
with problem.using_resolved_bc(state):
    U = hjb_solver.solve(...)  # Receives concrete values
```

**Topology/Calculator layer** (`applicator_base.py`):
- Clean decomposition: `PeriodicTopology`/`BoundedTopology` + `DirichletCalculator`/`NeumannCalculator`/`RobinCalculator`/`ZeroFluxCalculator`
- `LinearConstraint` bridges ghost cell (explicit) and matrix folding (implicit) approaches
- Well-designed but underused (see "What's Fragmented" below)

**GeometryProtocol boundary methods** (v0.16.17, Issue #545):
- `is_on_boundary()`, `get_boundary_indices()`, `get_boundary_normal()`, `get_boundary_info()`
- Default implementations in `Geometry` base class — all geometries inherit for free

### What's Fragmented

**Three parallel BC application paths**:

| Path | Location | Used By | Status |
|------|----------|---------|--------|
| Legacy functions | `applicator_fdm.py` (3400+ lines) | Most FDM solvers | Deprecated v0.19.0 but still primary path |
| Topology/Calculator | `applicator_base.py` (1800+ lines) | Nothing yet | Well-designed, zero solver adoption |
| Direct manipulation | Various solvers | Some solvers | Ad-hoc |

Bug fixes must be applied in multiple places. The well-designed replacement sits unused.

**`conditions.py` god class** (1000+ lines):
- BC storage, dimension binding, provider detection/resolution, domain bounds, SDF handling, validation, factory delegation — all in one class
- 30+ methods, multiple section-separator comments

**Dimension-specific implementations**:

| Feature | 1D | 2D+ |
|---------|----|----|
| Adjoint-consistent BC | Done | `NotImplementedError` (Issue #624) |
| GhostBuffer | Done | Incomplete |
| SDF region matching | Done | Partial |

**`applicator_fdm.py` chaos**:
- Module-level functions mixed with classes
- Multiple naming conventions (`apply_*`, `_apply_*`, `*_bc_*`)
- Duplicate logic across uniform/mixed/1D/2D/nD variants
- Target: delete after Topology/Calculator adoption

---

## 2. Standardized Solver Workflow

### Principle: Composition Over Mixins

The old pattern (GFDM example) used deep mixin hierarchies with implicit state sharing:

```python
# OLD — implicit contracts, scattered logic
class HJBGFDMSolver(GFDMBoundaryMixin, GFDMStencilMixin, ..., BaseHJBSolver):
    pass  # Mixin expects 8+ attributes from host
```

The standardized pattern uses explicit composition:

```python
# NEW — explicit dependencies, clear data flow
class HJBGFDMSolver:
    def __init__(self, problem):
        self.geometry = problem.geometry
        self.bc = problem.geometry.get_boundary_conditions()
        self.boundary_indices, self.boundary_normals = \
            self.geometry.get_boundary_info(self.collocation_points)
        self.bc_applicator = create_applicator(
            bc=self.bc, geometry=self.geometry,
            discretization_type=DiscretizationType.GFDM,
        )
```

### Unified BC Retrieval and Application

**Step 1 — Detect boundaries via geometry** (not custom loops):
```python
boundary_mask = self.geometry.is_on_boundary(points)
boundary_indices, normals = self.geometry.get_boundary_info(points)
```

**Step 2 — Apply via applicator** (not ad-hoc code):
```python
U_with_bc = self.bc_applicator.apply(values=U, geometry=self.geometry, ...)
```

### Solver-Specific Enforcement

| Discretization | Enforcement Method | Current State |
|---------------|-------------------|---------------|
| FDM | Ghost cells (`get_ghost_values_nd`) | Working, uses legacy path |
| GFDM | Ghost nodes, stencil rotation, penalties | Composition refactored (Issue #545) |
| Particle | Velocity reflection, domain projection | Composition refactored (Issue #543) |
| FEM | Matrix modification (`condense()`, `FacetBasis`) | Dirichlet only; Robin/Periodic planned |
| Neural (PINN/DGM) | Boundary loss terms | No framework integration |

### BoundaryHandler Protocol

Common interface for solver BC handling (`mfgarchon/geometry/boundary/handler_protocol.py`):

```python
class BoundaryHandler(Protocol):
    def get_boundary_indices(self) -> NDArray[np.integer]: ...
    def apply_boundary_conditions(self, values: NDArray, bc: BoundaryConditions, time: float = 0.0) -> NDArray: ...
    def get_bc_type_for_point(self, point_idx: int) -> str: ...
```

**Architecture**:
```
Geometry Layer (boundary detection, normals, segments)
        ↓
BoundaryHandler Protocol (common solver interface)
        ↓
Solver Layer (FDM: ghost cells | GFDM: ghost nodes | Particle: reflection | FEM: matrix mod)
```

**Migration status** (Issue #545):
- [x] FDM: hasattr eliminated, centralized BC retrieval
- [x] GFDM: composition refactored (4 components extracted from mixins)
- [x] Particle: hasattr eliminated
- [ ] Semi-Lagrangian: to be audited
- [ ] FEM: to be audited
- [ ] DGM: low priority (neural network paradigm)

---

## 3. Ghost Cell Formulas

For cell-centered grids where boundary lies at cell face:

| BC Type | Ghost Value Formula |
|---------|---------------------|
| Dirichlet ($u=g$) | $u_g = 2g - u_i$ |
| Neumann ($\partial u/\partial n=g$) | $u_g = u_i \pm 2\Delta x \cdot g$ |
| Robin ($\alpha u + \beta \partial u/\partial n=g$) | $u_g = (g - u_i(\alpha/2 - \beta/(2\Delta x))) / (\alpha/2 + \beta/(2\Delta x))$ |

For implicit solvers ($Au^{n+1} = b$), ghost cells become `LinearConstraint` coefficient folding:
```python
@dataclass
class LinearConstraint:
    """u_ghost = sum(weights[k] * u[inner+k]) + bias"""
    weights: dict[int, float]
    bias: float = 0.0
```

**Axiom**: Explicit (ghost cell) and implicit (coefficient folding) must produce identical results.

---

## 4. Adjoint-Consistent BC

### Mathematics

At reflecting boundaries with stall point:
```
Standard Neumann:     dU/dn = 0              (wrong at stall points)
Adjoint-consistent:   dU/dn = -sigma^2/2 * d(ln m)/dn   (correct for boundary stall)
```

Implemented as Robin BC with $\alpha=0$, $\beta=1$: the provider computes $g$ from current density each iteration.

### Scope Limitation

The formula is derived from the zero-flux equilibrium condition $J \cdot n = 0$ at reflecting boundaries. **This assumes the boundary IS the equilibrium point.**

| Configuration | AC BC | Neumann BC | Notes |
|---------------|-------|------------|-------|
| Boundary stall ($x=0$ or $x=1$) | Best (1.54x better) | Moderate | Use `AdjointConsistentProvider` |
| Interior stall ($x=0.5$) | Wrong (3.8x worse) | Moderate | Use Neumann or Strict Adjoint Mode |

Use `AdjointConsistentProvider` only for boundary stall problems.

---

## 5. Four-Tier Constraint Taxonomy

Beyond mathematical classification, a physical taxonomy guides architecture:

| Tier | Semantic | Examples | Ghost Pattern |
|------|----------|---------|---------------|
| **1: State** | Lock value | Dirichlet, exit, fixed temperature | `weights={}`, `bias=g` |
| **2: Gradient** | Lock shape | Neumann, symmetry, HJB reflective | `weights={0: 1.0}`, `bias=dx*g` |
| **3: Flux** | Lock flow | Robin, FP no-flux, adiabatic | `weights={0: alpha}`, `bias=0` |
| **4: Artificial** | Fake infinity | Linear extrapolation, absorbing | `weights={0: 2, 1: -1}`, `bias=0` |

**Physical intent mapping across PDE types**:

| Physical Intent | HJB | FP | Particles | Tier |
|----------------|-----|----|-----------|----|
| **wall** | Neumann $\partial V/\partial n=0$ | Robin (zero total flux) | Reflect | 2/3 |
| **exit** | Dirichlet $V=0$ | Dirichlet $\rho=0$ | Absorb | 1 |
| **symmetry** | Neumann | Neumann | Reflect | 2 |
| **outflow** | Linear extrapolation | Dirichlet $\rho \to 0$ | Pass through | 4 |
| **periodic** | Wrap (topology) | Wrap (topology) | Wrap | -- |

---

## 6. The 4-Layer Model (Future Direction)

`BC_ENFORCEMENT_ARCHITECTURE.md` identifies a critical architectural gap: **the same physical intent maps to different mathematical BCs depending on the PDE**. For example, "no-flux" means Neumann for pure diffusion, Robin for advection-diffusion, and variational inequality for HJB.

The proposed 4-layer framework adds an **equation-dependent resolution** layer:

```
Layer 1: SPECIFICATION — Physical intent + location (BCSegment)
    "This wall is impermeable"
         |
Layer 2: RESOLUTION — Intent x PDE type -> Mathematical BC  <-- THE MISSING LAYER
    "No-flux for FP = Robin(alpha, D)"
         |
Layer 3: ENFORCEMENT — Math BC x Discretization -> Discrete operations
    "Robin on FDM = ghost cell formula"
         |
Layer 4: APPLICATION — Discrete ops x Geometry -> Modified field/matrix/particles
    "Ghost cells on TensorProductGrid"
```

**Key insight**: `AdjointConsistentProvider` (Issue #625) is already a resolver — it resolves "reflecting wall" into a concrete Robin value using current density. The 4-layer model generalizes this from "advanced feature" to "standard mechanism" via `BCResolver` protocol.

**Current `BCType` conflation problem**: The enum mixes mathematical BCs (DIRICHLET, NEUMANN, ROBIN), physical intents (NO_FLUX, REFLECTING), and numerical techniques (EXTRAPOLATION_LINEAR). The 4-layer model cleanly separates these into `BoundaryIntent` (Layer 1) and `MathBCType` (Layer 2 output).

See `BC_ENFORCEMENT_ARCHITECTURE.md` for full resolver protocol definitions, solver-specific resolver examples (`FPResolver`, `HJBResolver`, `ParticleResolver`, `NeuralResolver`), and migration roadmap.

---

## 7. Technical Debt Summary

### NotImplementedError Locations

| Location | Feature | Priority |
|----------|---------|----------|
| `bc_coupling.py` | Adjoint BC for nD | High (Issue #624) |
| `dispatch.py` | FEM BC application | Medium |
| `applicator_fdm.py` | Robin corner nD | Low |
| `applicator_fdm.py` | High-order periodic/Robin | Low |
| `applicator_meshfree.py` | SDF region matching | Low |

### Policy Violations

`hasattr()` duck typing in `bc_coupling.py`, `types.py`, `conditions.py` — should use `Protocol` or `getattr()` per CLAUDE.md rules.

### Recommended Action Sequence

1. **Complete Topology/Calculator adoption** — wire up to one FDM solver as proof-of-concept
2. **Fix `hasattr()` violations** — define `SupportsStructuredGrid` Protocol
3. **Extend adjoint-consistent BC to nD** (Issue #624)
4. **Delete `applicator_fdm.py` legacy functions** after adoption validated
5. **Split `conditions.py` god class** (storage, binding, resolution, validation)
6. **Implement BCResolver layer** per `BC_ENFORCEMENT_ARCHITECTURE.md`

---

## 8. Key Files

| File | Purpose | Health |
|------|---------|--------|
| `types.py` | `BCType`, `BCSegment` | Good (minor hasattr violations) |
| `conditions.py` | `BoundaryConditions` | Bloated — needs splitting |
| `providers.py` | `BCValueProvider`, `AdjointConsistentProvider` | Good (clean, new) |
| `applicator_base.py` | Topology/Calculator decomposition | Well-designed but unused |
| `applicator_fdm.py` | Legacy FDM BC application | Chaotic — candidate for deletion |
| `bc_coupling.py` | Adjoint-consistent BC coupling | 1D only |
| `dispatch.py` | Applicator routing | FEM not implemented |
| `handler_protocol.py` | `BoundaryHandler` protocol | Clean |

---

## Related Documents

- `BC_ENFORCEMENT_ARCHITECTURE.md` — 4-layer model with resolver protocol (future direction)
- `docs/architecture/spacetime-bc/SPEC_COMPOSITIONAL_BC.md` — 4-axis BC framework spec
- `CONDITIONS_VS_CONSTRAINTS_ARCHITECTURE.md` — Conditions vs variational inequalities
- `BC_SOLVER_INTEGRATION_DESIGN.md` — Paradigm-specific BC helpers
