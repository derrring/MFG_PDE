# Current State Analysis: Gap Between Code and Specs

**Date**: 2026-02-05
**Codebase Version**: v0.16.14 (pre-1.0.0)

---

## 1. Summary

The specs propose a trait-based, compositional PDE framework. The codebase
already implements **~70% of the BC framework** under different names, but
**~0% of the space-time abstraction**. This document maps spec concepts to
existing code and identifies the real gaps.

---

## 2. BC Compositional Framework (MFG-SPEC-BC-0.2)

### 2.1 Axis Mapping: Spec vs Code

| Spec Axis | Spec Concept | Current Implementation | Coverage |
|:----------|:-------------|:-----------------------|:---------|
| **Region** | `GlobalBoundary`, `TaggedRegion(ID)`, `ImplicitInterface` | `BCSegment.boundary`, `.region`, `.sdf_region`, `.region_name` (5 matching modes) | ~85% |
| **MathType** | `Dirichlet`, `Neumann`, `Robin`, `Cauchy` | `BCType` enum: 8 types including Dirichlet, Neumann, Robin, Periodic, Reflecting, NoFlux, Extrapolation | ~90% |
| **ValueSource** | `Zero`, `Constant(c)`, `Functional(func)`, `DataField(array)` | `BCSegment.value: float \| Callable \| BCValueProvider` | ~80% |
| **Enforcement** | `Strong`, `Weak`, `GhostFluid` | Applicator hierarchy: FDMApplicator (ghost cells), MeshfreeApplicator, ImplicitApplicator | ~70% |

### 2.2 What's Missing

1. **Cauchy BC type**: Not in `BCType` enum. Needed for simultaneous value+gradient
   constraints (high-order equations). Low priority for MFG.

2. **Explicit `Zero` optimization**: The spec's `ValueSource.Zero` enables compile-time
   elimination of addition operations. Currently, zero values are just `value=0.0`
   with no special optimization path.

3. **Weak enforcement (FEM)**: `fem_bc_*.py` files exist but are basic. Full variational
   boundary integral (Nitsche method) not implemented.

4. **GhostFluid enforcement**: For level-set interface problems. Not implemented.
   The `ImplicitApplicator` uses projection, not ghost fluid interpolation.

### 2.3 What's Already Better Than the Spec

1. **BCValueProvider pattern** (Issue #625): The spec's `DataField(array)` is static.
   MFG_PDE has *dynamic* providers that compute BC values from solver state each
   iteration. This is more powerful.

2. **5 matching modes**: The spec has 3 region types. BCSegment has 5 matching modes
   (boundary name, axis ranges, SDF region, normal direction, marked region name)
   with validated non-mixing rules.

3. **Corner handling**: The spec mentions corner consistency briefly. MFG_PDE has a
   full corner handling subsystem (`boundary/corner/`) with strategies (priority,
   average, mollify) and particle reflection algorithms.

### 2.4 Interaction Protocols

| Spec Protocol | Description | Current State |
|:--------------|:------------|:--------------|
| **Protocol A** (Explicit Mesh) | `get_boundary_nodes(tag_id)` | ✅ `GeometryProtocol.get_boundary_indices()`, `get_boundary_regions()` |
| **Protocol B** (Implicit Grid) | Ghost cells, kernel loops on slices | ✅ `FDMApplicator.apply_2d()` does exactly this |
| **Protocol C** (Level Set) | SDF sign change detection, ghost fluid | ⚠️ `ImplicitApplicator` uses SDF but not ghost fluid method |

---

## 3. Space-Time Architecture (MFG-SPEC-ST-0.8)

### 3.1 Core Abstractions

| Spec Concept | Current Implementation | Gap |
|:-------------|:-----------------------|:----|
| `SpacetimeField` | `np.ndarray` with shape `(Nt+1, *spatial_shape)` | No type, no storage policy, no access control |
| `SpacetimeBoundaryData` | Scattered across `MFGProblem`: `m_initial`, `u_final`, `boundary_conditions` | Not unified; no corner consistency validation |
| `TrajectorySolver` protocol | No protocol. Loops live inside `HJBFDMSolver`, `FPFDMSolver` | Major gap |
| `SequentialMarchingSolver` | Implicit in concrete solvers (`for n in range(Nt)`) | Need to extract |
| `GlobalSpacetimeSolver` | Not implemented | Entirely new infrastructure |
| `StoragePolicy` | Everything in-core (NumPy arrays) | No abstraction |
| Corner Consistency | `preserve_initial_condition()`, `preserve_terminal_condition()` in `fixed_point_utils.py` | Preservation exists but no Gibbs/consistency validation |

### 3.2 Time Integration (embedded, not abstracted)

**HJB (backward)**:
```
Location: mfg_pde/alg/numerical/hjb_solvers/hjb_fdm.py
Method: Newton iteration at each time step (implicit)
Loop: for n in range(Nt, 0, -1): U[n-1] = newton_step(U[n], M[n])
```

**FP (forward)**:
```
Location: mfg_pde/alg/numerical/fp_solvers/fp_fdm.py
Method: Forward Euler or implicit step (explicit/implicit)
Loop: for n in range(Nt): M[n+1] = step(M[n], drift[n])
```

**Key observation**: The time loop, spatial operator, and BC enforcement are
interleaved in these solvers. Extracting a clean `StepOperator` requires
separating three concerns that are currently mixed.

### 3.3 Coupling Architecture

```
FixedPointIterator.solve()
├── for iiter in range(max_iterations):     # Picard loop
│   ├── resolve BC providers (if dynamic)
│   ├── hjb_solver.solve_hjb_system(M, U_terminal)  # Full backward trajectory
│   ├── fp_solver.solve_fp_system(m_initial, drift)  # Full forward trajectory
│   ├── damping: U = theta*U_new + (1-theta)*U_old
│   ├── preserve IC/TC
│   └── check convergence
└── return SolverResult
```

The `FixedPointIterator` already acts as a proto-`TrajectorySolver` orchestrator.
The gap: HJB and FP solvers return full trajectories but don't expose `StepOperator`.

---

## 4. Trait System

### 4.1 Geometry Traits (well-developed)

```
mfg_pde/geometry/protocols/
├── operators.py    → SupportsLaplacian, SupportsGradient, SupportsDivergence, ...
├── regions.py      → SupportsBoundaryNormal, SupportsBoundaryProjection, SupportsRegionMarking
├── topology.py     → SupportsPeriodic, SupportsManifold, SupportsLipschitz
└── graph.py        → SupportsGraphLaplacian, SupportsAdjacency, SupportsSpatialEmbedding
```

All are `@runtime_checkable` Protocol classes. 12+ protocols total.

### 4.2 Missing Trait Categories (from spec extensions)

| Spec Category | Description | Current State |
|:--------------|:------------|:--------------|
| **Time Integrator Traits** | SchemeType, Order, Storage, Adaptivity | Not implemented |
| **Operator Traits** | DifferentialType, StencilWidth, Conservation, Upwinding | Partially: `SupportsLaplacian` etc. exist but no stencil/conservation traits |
| **Linear Solver Traits** | MatrixStructure, Coupling, Origin → auto-select solver | Not implemented |
| **Storage Traits** | InCore, Streaming, Checkpointed, OutCore | Not implemented |

### 4.3 Type System

The `mfg_pde/meta/type_system.py` already has `MathematicalSpace`, `NumericalMethod`
enums and a `MFGType` descriptor. This is the natural home for additional trait
enumerations if/when they're needed.

---

## 5. Assessment

### 5.1 What the Specs Got Right

1. **BC 4-axis decomposition** validates the existing BCSegment design
2. **SpacetimeBoundaryData** is a genuinely useful unification
3. **TrajectorySolver** protocol cleanly separates "what to solve" from "how to step"
4. **Corner consistency** validation is a real gap (IC/BC mismatch causes Gibbs)
5. **Periodic BC canonical cases** are well-scoped

### 5.2 Where the Specs Overreach (or need qualification)

1. **GKS/SBP-SAT**: Relevant for **first-order MFG** ($\sigma = 0$) where HJB is
   Hamilton-Jacobi (hyperbolic) and FP is a conservation law. Not needed for
   second-order MFG ($\sigma > 0$, parabolic). The codebase already uses
   Godunov upwinding and WENO for the $\sigma \to 0$ regime, which is the
   practical equivalent. Formal SBP-SAT infrastructure is future work.
2. **GlobalSpacetimeSolver**: Research contribution, not infrastructure.
3. **StoragePolicy**: No current scaling problem.
4. **Time Integrator Traits**: MFG has fixed time structure (HJB backward, FP forward).
5. **Linear Solver auto-selection**: Not the bottleneck (Picard iterations dominate).
6. **ALE protocol**: No moving mesh use case in MFG.
7. **TPMS/Sphere periodicity**: Materials science scope, not MFG.

### 5.3 Quantified Gaps

| Gap | Size | Blocks v1.0? |
|:----|:-----|:-------------|
| SpacetimeBoundaryData container | Small (new dataclass + wiring) | No, but improves API |
| Applicator consolidation (#712) | Medium (merge 2 classes) | No, but reduces confusion |
| Operator library cleanup (#658) | Medium (unify interface) | No |
| TrajectorySolver protocol | Medium (new protocol + wrapper) | No |
| StepOperator extraction | Large (refactor all solvers) | No |
| GlobalSpacetimeSolver | Very Large (new infrastructure) | No |
| GKS/SBP-SAT | Large (new theory layer) | No |

**None of these gaps block v1.0.** They are architectural improvements for v2.0+.

---

**Last Updated**: 2026-02-05
