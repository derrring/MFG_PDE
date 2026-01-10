# Next Steps - 2026-01-11

**Date**: 2026-01-11
**Context**: Following completion of Issue #542 (FDM BC) and Issue #543 (hasattr elimination)

## Completed Priorities

✅ **Priority 1** - Issue #542 (FDM Periodic BC Bug) - CLOSED 2026-01-10
- PR #548 + PR #550 merged
- Tower-on-Beach validation error reduced to < 2%

✅ **Priority 3** - Issue #543 (hasattr Protocol Duck Typing) - CLOSED 2026-01-10
- 4 PRs merged (#551, #552, #553, #554)
- 96% reduction (79 → 3 violations)
- Core + Geometry modules cleaned up

## Current State Analysis

### High Priority Open Issues

1. **Issue #545** - Mixin Refactoring (Priority 4 in roadmap)
   - Status: OPEN
   - Priority: High
   - Size: Large
   - Labels: algorithms, refactor
   - Scope: Replace deep mixin hierarchies with composition

2. **Issue #544** - Geometry-First API (Priority 7 in roadmap)
   - Status: OPEN
   - Priority: High
   - Size: Large
   - Labels: core, infrastructure
   - Scope: Remove legacy Nx, xmin parameters from MFGProblem

3. **Issue #547** - Silent Fallbacks (Priority 2 in roadmap)
   - Status: OPEN
   - Priority: Medium
   - Size: Medium
   - Labels: infrastructure, refactor
   - Scope: Eliminate broad exception handlers, add logging

4. **Issue #549** - BC Framework Generalization
   - Status: OPEN
   - Priority: Medium
   - Size: Large
   - Labels: geometry, infrastructure
   - Scope: Extend BC framework to non-tensor-product geometries

### Lower Priority Issues (Deferred)

- **Issue #535** - BC Framework Enhancement (Medium, Large)
- **Issue #527** - BC Infrastructure Integration Phase 2-4 (Medium, Large)
- **Issue #523** - MMS/Conservation Validation (Low, Medium)
- **Issue #521** - 3D Corner Handling (Low, Large)
- **Issue #534** - Documentation Audit (Low, automated)
- Many feature requests (FVM, JAX autodiff, etc.)

## Decision Framework

### Option 1: Priority 2 - Eliminate Silent Fallbacks (#547)

**Pros**:
- ✅ Medium scope (2-3 days estimated)
- ✅ Aligns with CLAUDE.md Fail Fast principle
- ✅ Independent work (no dependencies)
- ✅ Improves debugging experience immediately
- ✅ Natural progression after hasattr cleanup

**Cons**:
- ⚠️ Codebase-wide (requires thorough audit)
- ⚠️ May uncover hidden issues

**Estimated Effort**: 2-3 days

**Acceptance Criteria**:
1. Audit all `except Exception:` and bare `except:` patterns
2. Replace with specific exception types
3. Add `logger.warning()` for all fallbacks
4. Document fallback behavior in docstrings
5. (Optional) Add `--strict` mode for CI

### Option 2: Priority 4a - Define BoundaryHandler Protocol (#545 Phase 1)

**Pros**:
- ✅ High priority issue
- ✅ Builds on GeometryProtocol work from #543
- ✅ Enables cleaner BC handling across solvers
- ✅ Natural next step after protocol work

**Cons**:
- ⚠️ Requires design decisions (protocol interface)
- ⚠️ Part of larger refactoring (Phase 2-3 needed)
- ⚠️ May need iteration on design

**Estimated Effort**: 3-4 days (Phase 1 only)

**Acceptance Criteria** (Phase 1):
1. Define `BoundaryHandler` protocol interface
2. Specify required methods: `detect_boundary_points()`, `apply_bc()`, etc.
3. Document protocol in `docs/development/BOUNDARY_HANDLING.md`
4. Update GeometryProtocol if needed
5. Add protocol validation helpers

### Option 3: Mixed Approach - Quick Win + Protocol Design

**Approach**: Start with #547 (silent fallbacks) while designing #545 protocol in parallel

**Pros**:
- ✅ Quick win with #547 (improves debugging immediately)
- ✅ Design time for #545 protocol (avoid rushing)
- ✅ Maintains momentum

**Cons**:
- ⚠️ Context switching between tasks
- ⚠️ May be harder to focus

### Option 4: Issue #549 - BC Framework Generalization

**Pros**:
- ✅ Addresses BC limitations for non-tensor-product geometries
- ✅ High-value infrastructure improvement

**Cons**:
- ⚠️ Large scope (similar to #545)
- ⚠️ Not in original roadmap
- ⚠️ May require significant design work

## Recommendation

### **Recommended: Option 1 - Priority 2 (#547 Silent Fallbacks)**

**Rationale**:

1. **Natural Progression**: We just completed protocol-based fail-fast work (#543). Eliminating silent fallbacks is the logical next step in the same direction.

2. **Medium Scope**: 2-3 days is manageable and provides quick value.

3. **Foundation for Later Work**: Cleaner error handling will help during #545 refactoring when we need to understand solver behavior.

4. **CLAUDE.md Alignment**: Directly enforces "Fail Fast & Surface Problems" principle.

5. **Independent**: No dependencies on other work, can start immediately.

### Implementation Plan for #547

**Week 1 (Days 1-3)**:

**Day 1: Audit Phase**
- [ ] Run comprehensive search for broad exception patterns:
  ```bash
  grep -r "except Exception:" mfg_pde/
  grep -r "except:" mfg_pde/ | grep -v "except.*Error"
  ```
- [ ] Categorize violations by type (import fallbacks, optional features, error recovery)
- [ ] Create audit document with locations and patterns
- [ ] Identify high-risk areas (where silent failures mask bugs)

**Day 2: Refactor Core Modules**
- [ ] Replace broad exceptions in `mfg_pde/core/`
- [ ] Replace broad exceptions in `mfg_pde/config/`
- [ ] Add logging for all fallbacks
- [ ] Add docstring documentation for fallback behavior
- [ ] Run tests after each module

**Day 3: Refactor Algorithms & Finalize**
- [ ] Replace broad exceptions in `mfg_pde/alg/`
- [ ] Replace broad exceptions in `mfg_pde/geometry/`
- [ ] Add unit tests for error handling paths
- [ ] Document patterns in `docs/development/ERROR_HANDLING_GUIDE.md`
- [ ] (Optional) Implement `--strict` mode flag
- [ ] Create PR and merge

**Deliverables**:
1. All broad exceptions replaced with specific types
2. Logging added for all silent fallbacks
3. Documentation of error handling patterns
4. Tests validating error paths
5. PR #555 (estimated) merged

### After #547: Next Steps

**Option A**: Continue with Priority 4a (#545 Phase 1 - BoundaryHandler Protocol)
- Design protocol interface
- Document BC handling workflow
- Set foundation for solver refactoring

**Option B**: Address Issue #549 (BC Framework Generalization)
- Extends BC to non-tensor-product geometries
- Complements #545 work

**Option C**: Return to hasattr cleanup Phase 2 (algorithms)
- Continue #543 momentum in algorithm modules
- Apply patterns from core/geometry to solvers

**Decision Point**: User preference after #547 completion

## Alternative: If User Prefers Different Direction

If the user wants to focus on different work:

### **Alternative A: Start #545 Protocol Design**
Begin BoundaryHandler protocol definition and documentation.

### **Alternative B: Address #549 BC Generalization**
Tackle BC framework extension for non-tensor-product geometries.

### **Alternative C: Continue hasattr cleanup**
Apply #543 patterns to algorithm modules (Priority 5 in roadmap).

## Summary

**Immediate Recommendation**:
- **Start Issue #547** (Eliminate Silent Fallbacks)
- **Duration**: 2-3 days
- **Value**: Improves debugging, enforces fail-fast principle
- **Risk**: Low (independent, well-defined scope)

**After #547**:
- Reassess priorities
- Consider #545 Phase 1 (BoundaryHandler Protocol)
- Or continue with user-preferred direction

**Long-term Roadmap**:
- Priority 2 (#547) ✅ Next
- Priority 4 (#545) - After #547
- Priority 5 (#543 Phase 2) - Algorithms
- Priority 6 (#545 Phase 3) - Solver refactoring
- Priority 7 (#544) - Legacy deprecation

---

**Questions for User**:

1. **Agree with #547 (Silent Fallbacks) as next priority?**
2. **Or prefer different direction (#545 protocol design, #549 BC generalization)?**
3. **Any other priorities or blockers to address first?**

---

**Status**: Awaiting user decision
**Created**: 2026-01-11
