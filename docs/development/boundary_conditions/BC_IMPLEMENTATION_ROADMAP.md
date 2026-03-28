# BC Implementation Roadmap

**Status**: Active roadmap — verified 2026-01-17, updated 2026-03-28
**Plan**: Plan A (Conservative Sequential) — approved after verification
**Baseline**: v0.17.16
**Related**: `BC_ENFORCEMENT_ARCHITECTURE.md` (4-layer model), Issue #848 (module restructuring)

---

## 1. Current Baseline (v0.17.16)

### Implemented (Tier 1 — Production-Ready)

| Feature | Status | Key Files |
|:--------|:-------|:----------|
| **BCSegment + BoundaryConditions** | ✅ Production | `geometry/boundary/conditions.py`, `types.py` |
| **BCType enum** (8 types) | ✅ Production | DIRICHLET, NEUMANN, ROBIN, PERIODIC, NO_FLUX, REFLECTING, EXTRAPOLATION_* |
| **6 concrete applicators** | ✅ Production | FDM, FEM, Meshfree, Implicit, Particle, Graph |
| **Dispatch layer** | ✅ Production | `apply_bc()` auto-selects applicator from geometry |
| **Topology/Calculator decomposition** | ✅ Production | Reusable physics across geometries |
| **Robin BC framework (Issue #574)** | ✅ Production | `AdjointConsistentProvider`, `BCValueProvider` protocol |
| **SSOT pattern (Issue #493)** | ✅ Production | Geometry owns spatial BC |
| **Unified BC access (Issue #527)** | ✅ Production | `BaseMFGSolver.get_boundary_conditions()`, paradigm helpers |
| **Mixed BC** | ✅ Production | Priority-based segment resolution, corner handling |
| **Module restructuring (Issue #848)** | ✅ Complete | Monolith split, slim exports |

### Partially Implemented

| Feature | Status | Notes |
|:--------|:-------|:------|
| **FEM Robin/Periodic** | ⚠️ Planned | Currently Dirichlet via `condense()`, Neumann natural |
| **Neural solver BC** | ⚠️ Ad-hoc | Each PINN/DGM has custom loss terms, no unified integration |
| **RL solver BC** | ⚠️ None | No BC framework integration |

### Not Implemented

| Feature | Tier | Notes |
|:--------|:-----|:------|
| **Variational inequalities** | Tier 2 | Obstacle problems, capacity constraints |
| **Free boundaries / Level Set** | Tier 3 | Moving frontiers, Stefan problems |
| **Equation-dependent BC resolution** | Architecture | The Resolution Problem (see BC_ENFORCEMENT_ARCHITECTURE.md §Layer 2) |
| **GKS/L-S stability validation** | Quality | Developer tool, not user-facing |
| **Geometry trait protocols** | Infrastructure | `SupportsLaplacian`, `SupportsGradient`, etc. |

---

## 2. Architectural Context

### 4-Layer BC Resolution Model

Defined in `BC_ENFORCEMENT_ARCHITECTURE.md` (v0.17.16):

```
Layer 1: SPECIFICATION (user-facing)
    BCSegment: Physical intent + boundary location

Layer 2: RESOLUTION (solver-facing)  ← PARTIALLY MISSING
    Physical intent x PDE type -> Mathematical BC

Layer 3: ENFORCEMENT (discretization-facing)
    Mathematical BC x Discretization -> Discrete operations

Layer 4: APPLICATION (geometry-facing)
    Discrete operations x Geometry -> Modified field/matrix/particles/loss
```

**Current state**: Layers 1, 3, 4 are production-ready. Layer 2 (Resolution) is the
primary architectural gap — the same physical intent ("no-flux") requires different
mathematical BCs depending on the PDE (Neumann for HJB, Robin for FP).

### 3-Tier BC Hierarchy

| Tier | Description | Status |
|:-----|:------------|:-------|
| **Tier 1** — Classical BCs | Dirichlet, Neumann, Robin, Periodic, No-flux, Mixed | ✅ Production |
| **Tier 2** — Variational Constraints | Obstacle problems ($u \geq \psi$), capacity ($m \leq m_{\max}$), bilateral | ❌ Not implemented |
| **Tier 3** — Dynamic Interfaces | Level Set evolution, Stefan problems, free boundary MFG | ❌ Not implemented |

### Completed Foundation

These issues provide the foundation for all planned work:

- **Issue #493** — Geometry owns BC (SSOT pattern): ✅ Complete
- **Issue #527** — BC Solver Integration (paradigm helpers): ✅ Phase 2-3 Complete
- **Issue #574** — Adjoint-Consistent Robin BC: ✅ Complete (v0.17.1)
- **Issue #848** — Module restructuring (split monolith, slim exports): ✅ Complete (v0.17.16)

### Coordination with Other Issues

| Issue | Relationship to This Roadmap |
|:------|:----------------------------|
| **#535** (BC Framework Enhancement) | Overlap on GKS/L-S validation. Plan: GKS in Phase 4 here, L-S deferred to #535 |
| **#536** (Particle Absorbing BC) | Independent. Tier 1 particle enhancement, no conflict |

---

## 3. Plan A: Conservative Sequential

### Overview

| Phase | Duration | Key Deliverables | Risk |
|:------|:---------|:-----------------|:-----|
| **1. Geometry Traits** | 2-3 weeks | Protocols, retrofits, region registry | Low |
| **2. Tier 2 BCs (VIs)** | 3-4 weeks | Constraints, VI solver, examples | Low-Medium |
| **3. Tier 3 BCs (Free)** | 3-4 weeks | Level Set, Stefan, MFG coupling | Medium |
| **4. Advanced Methods** | 2-3 weeks | Nitsche, GKS validation | Medium |
| **5. Documentation** | 2 weeks | Theory + user guides | Low |
| **6. Testing** | 1-2 weeks | Unit + integration + benchmarks | Low |
| **Total** | **12-16 weeks** | | |

**Properties**: Incremental (can stop after any phase), backward-compatible, solo-developer-friendly.

---

### Phase 1: Geometry Trait System (2-3 weeks)

**Objective**: Formalize operator trait protocols, retrofit existing geometries, add region registry.

#### 1.1 Protocol Definition (3-5 days)

New files in `mfgarchon/geometry/protocols/`:

- `operators.py` — `SupportsLaplacian`, `SupportsGradient`, `SupportsDivergence`, `SupportsAdvection`
- `topology.py` — Topological trait protocols
- `regions.py` — Region marking protocols

These traits **augment** the existing `GeometryProtocol` (not replace). Geometry classes
gain traits via multiple inheritance:

```python
class TensorProductGrid(BaseGeometry, SupportsLaplacian, SupportsGradient, ...):
    def get_laplacian_operator(self, order=2, bc=None) -> LinearOperator: ...
```

#### 1.2 Retrofit Existing Geometries (5-7 days)

| Geometry | Current Compliance | Work Needed |
|:---------|:------------------|:------------|
| **TensorProductGrid** | ~80% | Add `SupportsAdvection`, `SupportsInterpolation`, region registry |
| **ImplicitDomain** | ~50% | `SupportsGradient` (via SDF), `SupportsBoundaryQuery` |
| **GraphGeometry** | ~30% | Graph Laplacian operator, `SupportsTopology` |
| **UnstructuredMesh** | Out of scope | Deferred to separate issue |

#### 1.3 Region Registry System (3-4 days)

New files in `mfgarchon/geometry/regions/`:

- `registry.py` — `RegionRegistry` with `register()`, `query()`, boolean ops (intersect/union)
- `predicates.py` — Common predicates: `box_region()`, `sphere_region()`, `sdf_region()`

Integration with BC framework via named boundary references:

```python
geometry.mark_region("inlet", box_region([0, 0], [0, 1]))
BCSegment(name="inlet_bc", bc_type=BCType.DIRICHLET, value=1.0, boundary="inlet")
```

**Resolution order**: Standard names (x_min, etc.) -> Region registry -> Direct SDF/predicate.

**Success criteria**:
- All existing geometries pass trait compliance tests
- Operator accuracy tests pass (convergence order verification)
- Region registry supports all Tier 1 BC applications

---

### Phase 2: Tier 2 BCs — Variational Constraints (3-4 weeks)

**Objective**: Implement obstacle problems and variational inequalities.

#### 2.1 Constraint Protocol (1 week)

New files:
- `mfgarchon/geometry/boundary/constraint_protocol.py`
- `mfgarchon/geometry/boundary/constraints.py`

Key abstractions:

```python
class ConstraintProtocol(Protocol):
    def project(self, u: NDArray) -> NDArray: ...
    def is_feasible(self, u: NDArray, tol: float = 1e-10) -> bool: ...
    def get_active_set(self, u: NDArray, tol: float = 1e-10) -> NDArray[np.bool_]: ...

class ObstacleConstraint:
    """Unilateral constraint: u >= psi or u <= psi."""
    ...

class BilateralConstraint:
    """Bilateral constraint: psi_lower <= u <= psi_upper."""
    ...
```

#### 2.2 VI Solver Integration (2 weeks)

**Approach**: Penalty-projection methods (least invasive modification to existing solvers).

Solvers to modify:
1. HJB FDM Solver — add optional `constraints` parameter
2. HJB Semi-Lagrangian — same pattern
3. Poisson Solver — for testing/validation

Pattern: After each unconstrained update, apply `constraint.project(U_new)`.

Optional advanced method: Projected Newton (active set detection + reduced system solve).

#### 2.3 Documentation and Examples (3-5 days)

- `examples/advanced/obstacle_problem_1d.py`
- `examples/advanced/mfg_with_capacity_constraints.py`
- `docs/user/variational_inequalities.md`

**Success criteria**:
- 1D obstacle problem matches analytical solution (< 1% error)
- Capacity-constrained MFG shows queue formation
- Projection overhead < 5% of total solve time

---

### Phase 3: Tier 3 BCs — Dynamic Interfaces (3-4 weeks)

**Objective**: Level Set method for free boundary problems.

#### 3.1 Level Set Infrastructure (1.5 weeks)

New files in `mfgarchon/geometry/level_set/`:
- `core.py` — `LevelSetFunction` (interface mask, normal, curvature), `LevelSetEvolver`
- `reinitialization.py` — Redistancing (pseudo-time PDE: maintain SDF property)
- `curvature.py` — Geometric quantities from level set

Evolution equation: $\partial\phi/\partial t + V|\nabla\phi| = 0$ (Hamilton-Jacobi).

Start with upwind scheme; WENO as optional upgrade.

#### 3.2 Stefan Problem Implementation (1 week)

- `examples/advanced/stefan_problem_1d.py` — 1D ice melting with moving boundary
- `examples/advanced/stefan_problem_2d.py` — 2D solidification

#### 3.3 MFG with Free Boundaries (1 week)

Demonstrate MFG with moving domains (e.g., crowd evacuation with expanding exit, where
door velocity $V \propto$ density at door).

**Success criteria**:
- Stefan problem matches literature results
- Level Set remains SDF after 100 steps ($\max|||\nabla\phi| - 1|| < 0.1$)
- MFG with free boundary converges

---

### Phase 4: Advanced BC Methods (2-3 weeks)

#### 4.1 Nitsche's Method for FEM (1 week)

Modify `mfgarchon/geometry/boundary/applicator_fem.py` to support weak Dirichlet
imposition via Nitsche's method. Adds consistency, symmetry, and penalty terms to
the FEM weak form.

#### 4.2 GKS Stability Validation (1-2 weeks)

New files:
- `mfgarchon/geometry/boundary/validation/gks.py`
- `tests/validation/test_gks_conditions.py`

**Scope**: Developer tool for BC discretization validation. Not a user-facing API or
runtime check. Run once per BC type + discretization combination, document results.

**Limitations**: GKS checks discrete operator stability (necessary condition). It does
NOT check PDE well-posedness (use L-S for that, deferred to Issue #535) or nonlinear
stability.

**Success criteria**:
- Nitsche BC matches strong imposition (< 1% difference)
- GKS validation passes for all standard BCs (DNR, Robin, Periodic)

---

### Phase 5: Documentation (2 weeks)

- Update `docs/theory/GEOMETRY_BC_ARCHITECTURE_DESIGN.md` with implementation status
- New: `docs/theory/variational_inequalities_theory.md`
- New: `docs/theory/level_set_method.md`
- New: `docs/theory/gks_lopatinskii_conditions.md`
- New: `docs/user/advanced_boundary_conditions.md`
- New: `docs/user/geometry_traits.md`

---

### Phase 6: Testing and Validation (1-2 weeks)

- Unit tests: Trait protocol compliance, operator composition, constraint projection, Level Set evolution
- Integration tests: Full MFG with obstacles, Stefan problem, multi-tier BC combinations
- Performance benchmarks: Constraint projection overhead, Level Set reinit cost, GKS validation cost

---

## 4. Terminology Clarifications

### No-Flux: Physics vs Numerics

`BCType.NO_FLUX` has different implementations depending on PDE physics:

| PDE Context | Condition | Implementation |
|:------------|:----------|:---------------|
| **Fokker-Planck** (mass conservation) | Zero total flux: $J \cdot n = 0$ | `ZeroFluxCalculator` (needs drift, diffusion) |
| **HJB / Poisson** (edge extension) | Zero gradient: $\partial u/\partial n = 0$ | `ZeroGradientCalculator` (simple Neumann) |
| **Advection-diffusion** (general) | $-\kappa\nabla u \cdot n + u \cdot v \cdot n = 0$ | `AdvectionDiffusionNoFluxCalculator` |

This is an instance of the Resolution Problem (Layer 2 in the 4-layer model). Currently,
`NO_FLUX` defaults to `ZeroGradientCalculator` (pure Neumann), which is incorrect for
advection-diffusion. The `use_zero_flux=True` flag is a stopgap.

### Specialized BC Types Not in Tier Classification

| BC Type | Scope | Notes |
|:--------|:------|:------|
| **Reflecting** | Particle methods | Elastic velocity reflection |
| **Absorbing** | Particle / FP | Dirichlet $m=0$ on exit boundaries |
| **Extrapolation** (linear, quadratic) | Unbounded domains | Edge extension strategies |

---

## 5. Risk Analysis

| Risk | Probability | Impact | Mitigation |
|:-----|:------------|:-------|:-----------|
| VI solver convergence issues | Medium | High | Start with simple penalty method |
| Level Set instability | Medium | Medium | Conservative CFL, frequent reinitialization |
| Performance overhead from constraints | Low | Medium | Profile early, optimize hot paths |
| Scope creep | High | Medium | Strict phase boundaries |

---

## 6. Migration Guide

### Existing Code (No Changes Required)

```python
# Tier 1 BCs continue to work unchanged
bc = neumann_bc(dimension=2)
U = solver.solve(bc=bc)
```

### Tier 2 BCs (New Feature)

```python
from mfgarchon.geometry.boundary import ObstacleConstraint

constraint = ObstacleConstraint(psi, "lower")
U = solver.solve(bc=bc, constraints=[constraint])
```

### Geometry Traits (Optional Upgrade)

```python
from mfgarchon.geometry.protocols import SupportsLaplacian

if isinstance(geometry, SupportsLaplacian):
    laplacian = geometry.get_laplacian_operator(order=4)
```

---

## 7. Open Questions

1. **Tier 2 solver choice**: Start with penalty method. Add projected Newton in Phase 2.2 if performance-critical.
2. **Level Set scheme**: Upwind for Phase 3.1. WENO as optional upgrade.
3. **GKS scope**: All standard BCs (DNR, Robin, Periodic) to catch regressions.
4. **UnstructuredMesh**: Deferred to separate issue (not blocking for MFG research).

---

## 8. Related Documentation

| Document | Purpose |
|:---------|:--------|
| **`BC_ENFORCEMENT_ARCHITECTURE.md`** | 4-layer resolution model, the Resolution Problem |
| **`GEOMETRY_BC_ARCHITECTURE_DESIGN.md`** | Theoretical design specification (3-tier hierarchy, operator abstraction) |
| **`BC_COMPLETE_WORKFLOW.md`** | Complete BC workflow from user specification to solver |
| **`BC_SPECIFICATION_VS_APPLICATOR.md`** | 2-layer architecture (specification vs applicator) |
| **`BC_CAPABILITY_MATRIX.md`** | Current solver BC support matrix |
| **`BC_SOLVER_INTEGRATION_DESIGN.md`** | Paradigm-specific BC helpers (Issue #527) |

### Superseded by This Document

| Document | Disposition |
|:---------|:-----------|
| **`GEOMETRY_BC_IMPLEMENTATION_PLANS.md`** (1669 lines) | Superseded. Plans B/C/D dropped; Plan A content consolidated here. |
| **`GEOMETRY_BC_DESIGN_VERIFICATION.md`** (886 lines) | Superseded. Verification findings incorporated; approval recorded. |

---

**Last Updated**: 2026-03-28
**Consolidated from**: `GEOMETRY_BC_IMPLEMENTATION_PLANS.md` + `GEOMETRY_BC_DESIGN_VERIFICATION.md`
