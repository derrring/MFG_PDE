# Development Priority List (2026-01-10)

## Current Version: v0.16.16

This document outlines the prioritized roadmap for infrastructure improvements following PR #548.

---

## 🎯 Priority 1: Fix FDM Periodic BC Bug (#542)

**Issue**: [#542](https://github.com/derrring/MFG_PDE/issues/542)
**Status**: OPEN (mislabeled in PR #548, actual fix not implemented)
**Priority**: High (correctness bug)
**Size**: Medium
**Estimated Effort**: 1-2 days

### Problem
FDM 1D HJB solver uses `np.roll()` for Laplacian computation, which implicitly implements **periodic boundary conditions** regardless of geometry settings.

**Impact**: Tower-on-Beach validation shows 50% error (vs 0.5% for GFDM).

### Solution
Replace `np.roll()` with proper BC-aware stencil using existing `apply_boundary_conditions_1d()`:

```python
from mfg_pde.geometry.boundary import apply_boundary_conditions_1d

# Get padded array with proper ghost cells based on BC
U_padded = apply_boundary_conditions_1d(U, bc, domain_bounds)
# Standard central difference (no periodic wrapping)
U_xx = (U_padded[2:] - 2*U_padded[1:-1] + U_padded[:-2]) / dx**2
```

### Acceptance Criteria
- [ ] Replace `np.roll()` in `base_hjb.py` with BC-aware stencil
- [ ] Adjust Jacobian stencils at boundaries
- [ ] Implement Dirichlet row replacement
- [ ] Add BC logging like GFDM (document which BC applied where)
- [ ] Validate with Tower-on-Beach experiment (error < 2%)
- [ ] Add unit test for mixed BC (Neumann + Dirichlet)

### Why First?
- **Correctness bug** (not just technical debt)
- **Clear validation** (mfg-research experiment ready)
- **Achievable scope** (single solver, well-defined fix)
- **User preference** (explicitly requested)

---

## 🎯 Priority 2: Eliminate Silent Fallbacks (#547)

**Issue**: [#547](https://github.com/derrring/MFG_PDE/issues/547)
**Priority**: Medium
**Size**: Medium
**Estimated Effort**: 2-3 days

### Problem
Code catches broad exceptions and silently falls back to lower-fidelity methods without warning users. This masks configuration errors and performance degradation.

### Solution
1. Audit all `except Exception:` and bare `except:` patterns
2. Replace with specific exception types
3. Add `logger.warning()` for silent fallbacks
4. Document fallback behavior in docstrings

### Acceptance Criteria
- [ ] Audit all broad exception handlers
- [ ] Replace with specific exceptions (ImportError, ValueError, etc.)
- [ ] Add logging for all fallbacks
- [ ] Document fallback behavior
- [ ] (Optional) Add `--strict` mode for CI

### Why Second?
- **Aligns with Fail Fast principle** (CLAUDE.md core value)
- **Improves debugging experience** (users know what's happening)
- **Medium scope** (codebase-wide but straightforward)
- **Independent of other work** (can proceed in parallel with other tasks)

---

## 🎯 Priority 3: hasattr() Elimination - Phase 1 (Core) (#543)

**Issue**: [#543](https://github.com/derrring/MFG_PDE/issues/543)
**Priority**: High
**Size**: Large (but incremental)
**Estimated Effort**: 3-5 days (Phase 1), ongoing

### Problem
4000+ `hasattr()` usages violate Fail Fast principle and create unclear contracts.

### Solution (Phased Approach)

**Phase 1: Core Modules** (Priority 3)
- `mfg_pde/core/mfg_problem.py`
- `mfg_pde/core/mfg_components.py`
- `mfg_pde/config/`

**Phase 2: Algorithms** (Priority 5)
- `mfg_pde/alg/numerical/hjb_solvers/`
- `mfg_pde/alg/numerical/fp_solvers/`

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

**Last Updated**: 2026-01-10
**Current Focus**: Priority 1 (#542 FDM BC Fix)
