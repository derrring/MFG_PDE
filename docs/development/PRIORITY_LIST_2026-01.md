# Development Priority List (2026-01-10)

## Current Version: v0.16.16

This document outlines the prioritized roadmap for infrastructure improvements following PR #548.

---

## ✅ Priority 1: Fix FDM Periodic BC Bug (#542) - **COMPLETED**

**Issue**: [#542](https://github.com/derrring/MFG_PDE/issues/542)
**Status**: ✅ CLOSED (2026-01-10)
**Priority**: High (correctness bug)
**Size**: Medium
**Actual Effort**: 2 days (PR #548 + PR #550)

### Problem
FDM 1D HJB solver used `np.roll()` for Laplacian computation, implementing periodic BC regardless of geometry settings.

**Impact**: Tower-on-Beach validation showed 50% error (vs 0.5% for GFDM).

### Solution Implemented

**PR #548**: BC-aware gradient and Laplacian computation using ghost cells
**PR #550**: Explicit BC enforcement for Dirichlet and Neumann boundary values

**Key Insight**: BC-aware derivatives ≠ BC enforcement
- **Dirichlet BC**: Explicitly set `u(boundary) = g`
- **Neumann BC**: Set boundary value to satisfy `∂u/∂n = g`

### Result
- ✅ BC-aware stencils replace `np.roll()`
- ✅ Explicit Dirichlet/Neumann enforcement added
- ✅ Tower-on-Beach validation error reduced to < 2%
- ✅ Mixed BC handling implemented and tested

---

## ✅ Priority 2: Eliminate Silent Fallbacks (#547) - **COMPLETED**

**Issue**: [#547](https://github.com/derrring/MFG_PDE/issues/547)
**Status**: ✅ CLOSED (2026-01-11)
**Priority**: Medium
**Size**: Medium
**Actual Effort**: 2 days (PR #555 + PR #556)

### Problem
Code catches broad exceptions and silently falls back to lower-fidelity methods without warning users. This masks configuration errors and performance degradation.

### Solution Implemented

**Two-Phase Implementation**:
1. **PR #555** - High/Medium priority (9/13 fixes)
   - High: Newton solver failures, spectral analysis failures
   - Medium: Backend detection, GPU memory stats, volume computation, vmap fallback
2. **PR #556** - Low priority cosmetic (4/13 fixes)
   - Backend info retrieval, LaTeX setup, Quasi-MC fallback, performance monitoring

**Patterns Established**:
- Critical user-facing warnings: Specific exceptions + `logger.warning()` with fallback implications
- Diagnostic debug logging: Specific exceptions + `logger.debug()` for non-critical info
- Initialization warnings: Specific exceptions + `warnings.warn()` for module setup
- Re-raise with context: Broad exception OK when immediately re-raising with added logging

### Result
- ✅ 100% completion (13/13 fixes)
- ✅ All broad `except Exception:` replaced with specific exception types
- ✅ Comprehensive logging with context throughout
- ✅ Consistent MFG_PDE logging infrastructure
- ✅ All fallback behavior preserved for robustness
- ✅ Critical bugs (Newton solver) now surface instead of silently failing

**Documentation**: `docs/development/SILENT_FALLBACK_AUDIT_547.md`, `docs/development/SILENT_FALLBACK_COMPLETION_547.md`

### Why Second?
- **Aligns with Fail Fast principle** (CLAUDE.md core value)
- **Improves debugging experience** (users know what's happening)
- **Medium scope** (codebase-wide but straightforward)
- **Independent of other work** (can proceed in parallel with other tasks)

---

## ✅ Priority 3: hasattr() Elimination - Protocol Duck Typing (#543) - **COMPLETED**

**Issue**: [#543](https://github.com/derrring/MFG_PDE/issues/543)
**Status**: ✅ COMPLETED (2026-01-10)
**Priority**: High
**Size**: Large (but incremental)
**Actual Effort**: 4 days (4 PRs merged)

### Result
Eliminated 96% of protocol duck typing violations (79 → 3) in core and geometry modules.

### Problem
Protocol duck typing with `hasattr()` violates Fail Fast principle and creates unclear contracts.

### Solution Implemented

**Pull Requests Merged**:
1. ✅ PR #551 - Core module cleanup (27 → 5 violations)
2. ✅ PR #552 - Geometry Phase 1 (47 → 38 violations)
3. ✅ PR #553 - Core protocol checks (5 → 3 violations)
4. ✅ PR #554 - Geometry protocol checks (38 → 0 violations)

**Patterns Established**:
- `isinstance(geometry, GeometryProtocol)` for required methods
- `try/except AttributeError` for optional attributes
- Explicit error messages guide users to proper implementations

**Documentation**: `docs/archive/issue_543_hasattr_elimination_2026-01/`

**Remaining Work**: ~341 hasattr violations in other patterns (caching, legacy compatibility) - see `HASATTR_CLEANUP_PLAN.md`

**Phase 3: Utils & Workflow** (Priority 7)
- `mfg_pde/utils/`
- `mfg_pde/workflow/`

### Acceptance Criteria (Phase 1)
- [ ] Define additional protocols (SolverProtocol, ConfigProtocol)
- [ ] Replace all hasattr in `mfg_pde/core/` with isinstance checks
- [ ] Add CI check to reject new hasattr in core/
- [ ] Document protocol usage in `docs/development/CONSISTENCY_GUIDE.md`

### Why Third?
- **Builds on GeometryProtocol** work from PR #548
- **High priority** but can proceed incrementally
- **Core first** (highest leverage for downstream code)
- **Enables better type checking** and fail-fast behavior

---

## 🎯 Priority 4: Mixin Refactoring - FPParticle Template (#545)

**Issue**: [#545](https://github.com/derrring/MFG_PDE/issues/545)
**Priority**: High
**Size**: Large
**Estimated Effort**: 5-7 days (template), 3-4 days per solver

### Problem
Deep mixin hierarchies with implicit state sharing make solvers hard to understand and maintain.

### Solution (Phased Approach)

**Phase 1: Define BoundaryHandler Protocol** (Priority 4a)
```python
class BoundaryHandler(Protocol):
    def detect_boundary_points(self, points: np.ndarray) -> np.ndarray: ...
    def apply_bc(self, values: np.ndarray, bc: BoundaryConditions) -> np.ndarray: ...
    def get_bc_type_for_point(self, point_idx: int) -> str: ...
```

**Phase 2: Refactor FPParticleSolver as Template** (Priority 4b)
- Replace implicit inheritance with explicit composition
- Document pattern in `docs/development/BOUNDARY_HANDLING.md`

**Phase 3: Apply to Other Solvers** (Priority 6)
- GFDM, FDM, FEM, DGM

### Acceptance Criteria (Phase 1-2)
- [ ] Define BoundaryHandler protocol
- [ ] Geometry provides: `get_boundary_indices()`, `get_normals()`
- [ ] Refactor FPParticleSolver with composition
- [ ] Document workflow and pattern
- [ ] Remove duplicate BC detection logic

### Why Fourth?
- **High priority** but larger scope
- **Requires protocol definition** first (natural after #543 Phase 1)
- **FPParticle is simplest** (good template for others)
- **Enables unified BC handling** across all solvers

---

## 🎯 Priority 5: hasattr() Elimination - Phase 2 (Algorithms) (#543)

**Continuation of Priority 3**
**Estimated Effort**: 4-6 days

Apply protocol pattern to algorithm modules using lessons from Phase 1.

---

## 🎯 Priority 6: Mixin Refactoring - Remaining Solvers (#545)

**Continuation of Priority 4**
**Estimated Effort**: 10-15 days total

Apply composition pattern from FPParticle template to GFDM, FDM, FEM, DGM solvers.

---

## 🎯 Priority 7: Legacy Parameter Deprecation (#544)

**Issue**: [#544](https://github.com/derrring/MFG_PDE/issues/544)
**Priority**: High (but deferred until foundation stable)
**Size**: Large
**Estimated Effort**: 5-7 days

### Problem
MFGProblem supports both legacy (`Nx`, `xmin`) and modern (Geometry) APIs, creating bloat.

### Solution (Phased)

**Phase 2: Add DeprecationWarning (v0.17.0)** (Priority 7a)
- [ ] Add DeprecationWarning to MFGProblem.__init__
- [ ] Migrate all examples/ to Geometry API
- [ ] Migrate all tests/ to Geometry API
- [ ] Document migration in `docs/migration/LEGACY_PARAMETERS.md`

**Phase 3: Remove Legacy (v0.18.0 or v1.0.0)** (Future)
- [ ] Remove Nx, xmin, xmax parameters
- [ ] Remove _override attributes
- [ ] Simplify MFGProblem (target < 200 lines)

### Why Seventh?
- **High priority** but **large scope**
- **Requires stable foundation** (Protocols, BC handling)
- **User-facing breaking change** (needs careful migration)
- **Better after other refactoring** (cleaner migration path)

---

## 🎯 Priority 8: hasattr() Elimination - Phase 3 (Utils) (#543)

**Continuation of Priority 3**
**Estimated Effort**: 3-5 days

Final phase: apply protocol pattern to utilities and workflow modules.

---

## Summary Timeline

| Week | Priority | Issue | Deliverable |
|:-----|:---------|:------|:------------|
| **Week 1** | P1 | #542 | FDM BC fix + validation |
| **Week 2** | P2 | #547 | Silent fallback elimination |
| **Week 3** | P3 | #543 Phase 1 | Core hasattr → isinstance |
| **Week 4** | P4a | #545 Phase 1 | BoundaryHandler protocol |
| **Week 5** | P4b | #545 Phase 2 | FPParticle composition |
| **Week 6-7** | P5 | #543 Phase 2 | Algorithm hasattr cleanup |
| **Week 8-9** | P6 | #545 Phase 3 | Other solver refactoring |
| **Week 10-11** | P7 | #544 Phase 2 | Legacy deprecation |
| **Week 12** | P8 | #543 Phase 3 | Utils hasattr cleanup |

**Total Estimated Duration**: ~12 weeks for all priorities

---

## Dependencies Graph

```
P1 (#542 FDM BC) ────────────────────┐
    └─ Independent                   │
                                     │
P2 (#547 Silent Fallbacks) ──────────┤
    └─ Independent                   │
                                     │
P3 (#543 Phase 1: Core) ─────────────┤
    └─ Builds on GeometryProtocol    │
           │                         │
           ├──> P4a (BoundaryHandler)│
           │       │                 │
           │       └──> P4b (FPParticle Template)
           │              │          │
           └──> P5 (#543 Phase 2)   │
                   │                 │
                   └──> P6 (Other Solvers)
                           │         │
                           └──> P7 (#544 Deprecation)
                                  │  │
                                  │  └──> P8 (#543 Phase 3)
                                  │
                                  └─ All infrastructure stable
```

---

## Principles

### 1. **Correctness First**
Fix numerical bugs (#542) before architectural refactoring.

### 2. **Fail Fast & Surface Problems**
Eliminate silent fallbacks (#547) early to improve debugging.

### 3. **Foundation Before Facades**
Complete protocol work (#543 core) before large refactoring (#545, #544).

### 4. **Incremental Progress**
Break large issues into phases to maintain momentum.

### 5. **Validate Continuously**
Run tests after each priority, validate with research experiments.

---

## Notes

- **Issue #546 (sys.exit removal)**: ✅ CLOSED in v0.16.16
- **GeometryProtocol**: ✅ Foundation complete in v0.16.16
- **Parallel Work**: P1 and P2 can proceed in parallel if needed
- **Flexibility**: Timeline is estimate; adjust based on discoveries during work

---

**Last Updated**: 2026-01-11
**Completed**: Priority 1 (#542), Priority 2 (#547), Priority 3 (#543)
**Current Focus**: Determining next priority (P4a vs P5)
