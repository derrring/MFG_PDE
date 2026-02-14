# Project Evaluation: Space-Time & BC Architecture (#732)

**Date**: 2026-02-14
**Evaluator**: Claude Code (automated analysis + codebase cross-reference)
**Scope**: All 9 documents in `docs/architecture/spacetime-bc/`

---

## Executive Summary

This project produced well-written architectural vision documents with realistic scope discipline. The BC framework they describe is **~70% implemented** under existing code (BCSegment, applicator hierarchy, BCValueProvider). The space-time abstraction is **0% implemented** but elegantly designed.

**Three of four phases with concrete deliverables are already done** (Phases 0, 1a, 4). The remaining work is either low-priority (Phase 1b), a simple data container (Phase 2), or premature abstraction (Phase 3).

The project's main contribution was validating that existing BC infrastructure is architecturally sound. The specs confirmed this rather than driving new implementation.

---

## Per-Document Assessment

### README.md
**Purpose**: Project overview and scope guard.
**Assessment**: Correctly distinguishes MFG infrastructure from general PDE framework. The scope guard ("MFG_PDE is production MFG infrastructure, not a general PDE framework") is the most valuable sentence in the entire project.

### SPEC_SPACETIME_SOLVERS.md (MFG-SPEC-ST-0.8)
**Proposes**: SpacetimeField, SpacetimeBoundaryData, TrajectorySolver protocol, SequentialMarchingSolver.
**Implemented**: Zero. Solutions are numpy arrays; boundary data scattered across MFGProblem attributes.
**Verdict**: SpacetimeBoundaryData is a clean data container worth building (~1 day). TrajectorySolver has no second implementation to justify the abstraction.

### SPEC_COMPOSITIONAL_BC.md (MFG-SPEC-BC-0.2)
**Proposes**: 4-axis BC decomposition (Region, MathType, ValueSource, Enforcement).
**Implemented**: ~70%.

| Axis | Status | Codebase Location |
|:-----|:-------|:------------------|
| Region | 5 matching modes (exceeds spec) | `BCSegment.boundary`, region dicts, SDF matching |
| MathType | 8 BC types | `BCType` enum |
| ValueSource | Static + dynamic (exceeds spec) | `BCValueProvider` protocol, `AdjointConsistentProvider` |
| Enforcement | Applicator hierarchy | `FDMApplicator`, `BaseUnstructuredApplicator` |

**Verdict**: This spec is documentation of what already works. The existing `BCValueProvider` protocol (state-dependent BCs) is more powerful than the spec's static `DataField` proposal. No action needed.

### SPEC_TIME_INTEGRATION.md (MFG-SPEC-TI-0.1)
**Proposes**: SchemeType/TemporalOrder enums, StepOperator protocol, pluggable time integration.
**Implemented**: Zero. HJB uses implicit Euler + Newton; FP uses explicit/implicit Euler.
**Verdict**: Over-generalizes. MFG has a **fixed** temporal structure: HJB backward, FP forward, Picard couples them. Pluggable time integration solves a problem that doesn't exist in MFG. The `SchemeFamily` enum from PR #785 already covers what's needed for duality validation.

### SPEC_OPERATOR_SYSTEM.md (MFG-SPEC-OP-0.1)
**Proposes**: PDEOperator base class, operator algebra (`L1 + L2`), OperatorTraits metadata, CompositeOperator.
**Implemented**: ~5%. `LaplacianOperator(LinearOperator)` exists. No algebra, no traits.
**Verdict**: The most valuable unimplemented spec. FP solver has ~50 lines of manual matrix assembly that operator algebra would reduce to 2-3 lines. Worth implementing when FP solver is next refactored.

### SPEC_LINEAR_SOLVER.md (MFG-SPEC-LS-0.1)
**Proposes**: LinearSolverTraits, `select_solver()` auto-selection, operator-to-traits inference.
**Implemented**: `SparseSolver` abstraction exists; no trait system.
**Verdict**: Premature. Current bottleneck is Picard iterations (~20-50 iterations), not linear solves. For 1D-2D at current grid sizes, `scipy.sparse.linalg.spsolve()` runs in ~10ms. Revisit when 3D problems with $N > 10^5$ become standard.

### SPEC_PERIODIC_IMPLICIT.md (MFG-SPEC-ADD-01)
**Proposes**: Periodic BCs on implicit geometries (torus, sphere, ball, TPMS).
**Implemented**: 100% for the MFG-relevant case (rectangular torus via `BCType.PERIODIC` + modulo indexing).
**Verdict**: Done. Sphere/ball/TPMS correctly deferred (materials science scope).

### CURRENT_STATE_ANALYSIS.md
**Purpose**: Gap analysis mapping spec concepts to codebase.
**Assessment**: Best document in the set. Clear, factual, well-organized. Correctly quantifies ~70% BC coverage and ~0% space-time coverage.

### MIGRATION_PLAN.md
**Purpose**: 4-phase rollout with effort estimates and dependency graph.
**Assessment**: Phase estimates are realistic. Deferrals are well-reasoned with "revisit when" clauses. The dependency graph correctly shows that Phases 0, 1a, and 4 are prerequisites for Phase 2.

---

## Phase Status vs. Plan

| Phase | Plan | Actual Status | Assessment |
|:------|:-----|:-------------|:-----------|
| 0 (Docs) | v0.17.x | **DONE** (PRs #730, #731) | Completed as planned |
| 1a (Applicator merge) | v0.17.x | **DONE** (#712, #637 closed) | Completed as planned |
| 1b (BC naming) | v0.17.x | OPEN (#517, priority:low) | Issue says "can wait"; 2 solver types don't justify dispatch factory |
| 2 (SpacetimeBoundaryData) | v0.18.x | NOT STARTED | Dependencies done (#679, #682-684 closed). Container not built. |
| 3 (TrajectorySolver) | v0.19.x+ | NOT STARTED | No second solver type demands the abstraction |
| 4 (Operator cleanup) | v0.19.x | **DONE** (#658 closed, #625 closed) | Completed ahead of plan |

---

## What the Specs Got Right

1. **Scope discipline**: Every spec has a "What We're NOT Building" section. GKS/SBP-SAT theory, StoragePolicy, ALE, and Parareal are correctly deferred.

2. **Enforcement stays solver-side**: The key architectural decision (BCSegment specifies *what*, applicators implement *how*) was already implemented and the spec validated it.

3. **BCValueProvider exceeds the spec**: The spec proposed static `DataField` for BC values. The actual implementation (`BCValueProvider` protocol with `AdjointConsistentProvider`) is more powerful, supporting state-dependent BCs that resolve at iteration time.

4. **Realistic effort estimates**: Phase efforts in the migration plan are accurate for typical refactoring work.

## What the Specs Got Wrong

1. **Over-specification**: 9 documents totaling ~84KB and ~40 pages for work where the practical infrastructure is mostly done. The ratio of spec-to-implementation is inverted.

2. **General PDE framework scope creep**: Despite the scope guard, several specs describe general PDE infrastructure (pluggable time integrators, linear solver auto-selection, operator algebra) that exceeds MFG needs. The `SPEC_TIME_INTEGRATION.md` is the clearest example: MFG has fixed temporal structure that doesn't benefit from pluggable schemes.

3. **Missing the actual FEM blockers**: The specs were partly motivated by FEM (Issue #773), but none address the real FEM gaps: scikit-fem assembly, weak-form BC enforcement, mass/stiffness matrix construction. These are implementation problems, not architecture problems.

4. **Phase 3 premature abstraction**: TrajectorySolver protocol has exactly one implementation (SequentialMarchingSolver). Building a protocol + wrapper for a single implementation violates YAGNI. The spec acknowledges GlobalSpacetimeSolver as a future second implementation, but defers it to "after published paper" — which means the protocol has no justification until then.

---

## Recommendations

### Do Now
- **Update issue #732** checklist to reflect actual status (Phases 0, 1a, 4 complete)

### Do When Convenient (v0.18.x)
- **SpacetimeBoundaryData** container (Phase 2a): A simple `@dataclass` grouping `spatial_bc`, `initial_condition`, `terminal_condition`. Worth ~1 day. Improves MFGProblem API clarity.
- **Corner consistency validation** (Phase 2c): Uses existing validation infrastructure from #685. Catches IC/BC mismatches that cause Gibbs phenomena at domain corners.

### Do When FP Solver is Refactored
- **Operator algebra** (Phase 4, SPEC_OPERATOR_SYSTEM): The 50-line manual matrix assembly in FP solver would benefit from `L_diffusion + L_advection` composition. Not urgent but high quality-of-life.

### Do Not Build
- **Phase 1b** (#517 semantic dispatch): Two solver types (HJB, FP) don't justify a dispatch factory. Hardcoded `if/else` is sufficient and clearer.
- **Phase 3** (TrajectorySolver): No second implementation. Build the protocol when GlobalSpacetimeSolver or another solver type demands it.
- **SPEC_TIME_INTEGRATION**: MFG temporal structure is fixed. Pluggable time integrators solve a problem that doesn't exist.
- **SPEC_LINEAR_SOLVER**: Not the bottleneck. Revisit at $N > 10^5$.

### Consider Closing
- **Issue #732**: The project accomplished its real goals (applicator consolidation, operator cleanup, BC validation). Remaining items are either simple (#SpacetimeBoundaryData) or premature (#TrajectorySolver). Consider closing the umbrella and tracking remaining items as standalone issues.

---

## Document Consolidation

The 9 documents could be consolidated to 3 without information loss:

| Keep | Merge Into | Reason |
|:-----|:-----------|:-------|
| README.md | Keep as-is | Scope guard and overview |
| CURRENT_STATE_ANALYSIS.md | Keep, update status | Factual gap analysis |
| MIGRATION_PLAN.md | Keep, mark completed phases | Actionable roadmap |
| SPEC_COMPOSITIONAL_BC.md | Archive | Documents what's already working |
| SPEC_SPACETIME_SOLVERS.md | Merge into MIGRATION_PLAN | Only SpacetimeBoundaryData is actionable |
| SPEC_TIME_INTEGRATION.md | Archive | Over-generalizes; deferred indefinitely |
| SPEC_OPERATOR_SYSTEM.md | Keep as standalone issue | Actionable but independent of this project |
| SPEC_LINEAR_SOLVER.md | Archive | Premature; not the bottleneck |
| SPEC_PERIODIC_IMPLICIT.md | Archive | 100% implemented |

---

**Bottom line**: The project's practical value was delivered in Phases 0, 1a, and 4. The specs validated existing BC architecture and guided infrastructure cleanup. The remaining unbuilt items (SpacetimeBoundaryData, TrajectorySolver) are either simple data containers or premature abstractions. The specs themselves are higher quality than what they describe building.
