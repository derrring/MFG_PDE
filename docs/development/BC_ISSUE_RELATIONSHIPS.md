# BC Issue Relationship Map

Cross-referencing all boundary condition related issues to understand dependencies and relationships.

## Issue Dependency Graph

```
┌──────────────────────────────────────────────────────────────┐
│                    BC Infrastructure                         │
└──────────────────────────────────────────────────────────────┘
                             │
           ┌─────────────────┼─────────────────┐
           ▼                 ▼                 ▼
      #486 (CLOSED)     #535 (OPEN)      #527 (CLOSED)
   BC Unification    Framework Enhance   Integration
   (Parent Issue)    Math Foundation     dispatch.py
           │
           ├──────────────────┬──────────────────┬──────────────┐
           ▼                  ▼                  ▼              ▼
      #542 (CLOSED)      #521 (OPEN)       #549 (OPEN)   #545 (CLOSED)
   FDM Periodic Bug   Corner Handling    Generalization  Mixin Hell
   [COMPLETED]        (3D critical)      (Manifolds)     [COMPLETED]
           │                  │
           └──────────────────┴─────────────────────────────────┐
                                                                 ▼
                                                         #523 (OPEN)
                                                        MMS/Conservation
                                                         Test Suite
```

**Last Updated**: 2026-01-12

## Issue Details

### #542: FDM Periodic BC Bug ✅ **COMPLETED**

**Status**: CLOSED (2026-01-10, PR #548 + PR #550)
**Severity**: Medium
**Type**: Bug Fix

**Problem**: FDM solver used `np.roll()` (periodic BC) regardless of geometry BC specification.

**Root Cause**: BC-aware Laplacian only handled derivatives (ghost cells), but boundary VALUES never enforced.

**Solution**: Three-layer BC architecture
1. Specification (`BCSegment`, `BoundaryConditions`)
2. Ghost cells (`FDMApplicator.apply()`)
3. **Value enforcement** (`FDMApplicator.enforce_values()`) ← NEW

**Blocked By**: None (can proceed)
**Blocks**: None (independent fix)
**Related**: #521 (corner interference discovered during testing)

**Commits**:
- `e162e1a` - Remove hasattr() anti-pattern
- `0076042` - Fix 1D BC enforcement
- `896149e` - Extend to nD
- `1f60729` - Refactor to FDMApplicator

---

### #521: Corner Handling and Pre-allocation

**Status**: Open
**Severity**: Critical (3D)
**Type**: Architecture Enhancement

**Problem**: When multiple boundaries meet at corners, sequential BC application causes last-applied-wins behavior.

**Example**:
```python
# Apply x_min: field[0, :] = 1.0  → corners [0,0]=1.0, [0,-1]=1.0
# Apply y_min: field[:, 0] = 3.0  → overwrites [0,0]=3.0, [-1,0]=3.0
```

**Proposed Solutions**:
1. **Priority ordering**: Define precedence rules
2. **Averaging**: Corner = average of adjacent BC values
3. **Error on conflict**: Reject ambiguous specifications

**Relationship to #542**:
- Corner interference discovered during #542 validation
- #542 fix is correct; corner issue is separate architectural problem
- Most real problems unaffected (same BC on all sides, or opposite boundaries only)

**Dependencies**:
- Blocked by: None (can implement independently)
- Blocks: Full 3D validation, WENO BC support

---

### #549: Generalize BC Framework for Non-Tensor-Product Geometries

**Status**: Open
**Severity**: High (for manifolds)
**Type**: Infrastructure Enhancement

**Problem**: Current BC uses coordinate-based identifiers (`"x_min"`, `"y_max"`) that only work for tensor-product grids.

**Limitations**:
- ❌ Cannot handle curved boundaries
- ❌ No support for triangulated meshes (FEM/DGM)
- ❌ Coordinate-axis coupling prevents rotation

**Proposed Architecture** (documented in `BC_ENFORCEMENT_ARCHITECTURE.md`):
```python
class BoundaryEnforcer(Protocol):
    def enforce_values(self, field, bc, time): ...

# Implementations:
- TensorGridEnforcer  ← Current (#542 work)
- MeshEnforcer        ← FEM/DGM
- LevelSetEnforcer    ← Free boundaries
```

**Relationship to #542**:
- #542 creates `FDMApplicator.enforce_values()` → becomes `TensorGridEnforcer` in #549
- #549 generalizes the pattern established in #542
- Current implementation can migrate smoothly

**Dependencies**:
- Builds on: #542 (BC enforcement pattern)
- Requires: Geometry-first API completion (#544)
- Enables: Manifold MFG problems, FEM/DGM BC

---

### #545: Refactor Solver BC Handling (Mixin Hell → Composition)

**Status**: Open
**Severity**: High (code quality)
**Type**: Refactoring

**Problem**: Complex solvers use deep mixin hierarchies with scattered BC logic:
- GFDM: Multiple mixins across 5-6 files
- FDM: Direct methods in main class ← #542 addressed this
- Particle: Custom reflection logic
- No common interface

**Proposed Solution**:
```python
# Old: Mixin hell
class HJBGFDMSolver(MixinA, MixinB, MixinC):
    pass

# New: Composition
class HJBGFDMSolver:
    def __init__(self, problem):
        self.bc_handler = BoundaryHandler.from_geometry(problem.geometry)
        self.interpolator = Interpolator(...)
```

**Relationship to #542**:
- #542 refactor (moving BC to `FDMApplicator`) is a step toward composition
- #545 proposes broader cleanup across all solvers
- Pattern from #542 can be template for other solvers

**Dependencies**:
- Builds on: #542 (FDM composition pattern)
- Related: #549 (common BC interface)
- Blocks: None (independent refactor)

---

### #535: BC Framework Enhancement - Mathematical Foundation

**Status**: Open
**Severity**: Medium
**Type**: Enhancement

**Scope**: Mathematical rigor for BC handling
- Well-posedness verification
- Compatibility conditions
- Conservation properties

**Relationship to #542**:
- Independent theoretical work
- #542 provides correct implementation foundation
- #535 adds mathematical validation layer

---

### #527: BC Infrastructure Integration (Phase 2-4)

**Status**: Open
**Severity**: Medium
**Type**: Infrastructure

**Scope**: Complete BC integration including `dispatch.py`

**Relationship to #542**:
- Infrastructure for BC routing and selection
- #542 doesn't depend on this
- This might use #542 patterns

---

### #523: MMS and Conservation Validation Suite

**Status**: Open
**Severity**: Low
**Type**: Testing

**Scope**: Comprehensive BC validation via Method of Manufactured Solutions

**Relationship to #542**:
- Validates #542 fix mathematically
- Can create MMS tests for Dirichlet/Neumann enforcement
- Useful for regression testing

---

## Closed Issues (Historical Context)

### #486: Unify BC Handling (CLOSED - Parent Issue)

Unified BC API across HJB and FP solvers. Created foundation for:
- `BCSegment` dataclass
- `BoundaryConditions` class
- Ghost cell pattern (#494)

### #494: HJB BC Integration - Ghost Value Pattern (CLOSED)

Implemented ghost cells for upwind schemes. Addressed derivatives but not value enforcement → led to #542.

### #493: Unify Boundary Condition Handling (CLOSED)

Consolidation work, precursor to #486.

---

## Critical Path for #542 Resolution

**Current Status**: ✅ Implementation complete, validation in progress

**Remaining Work**:
1. ✅ Core fix implemented (commits e162e1a, 0076042, 896149e, 1f60729)
2. ✅ Refactored to FDMApplicator (architectural cleanup)
3. ⏳ Validate corner interference is #521 (not #542 blocker)
4. ⏳ Run comprehensive test suite
5. ⏳ Merge PR, close #542

**Corner Interference Resolution**:
- Document as known limitation (Issue #521)
- Add validation test without corner conflicts
- Defer corner handling to #521 (not blocking #542)

---

## Recommendations

### For #542 (Immediate):
1. ✅ Core BC enforcement works correctly
2. ⚠️ Corner conflicts are Issue #521, not #542
3. ✅ Add validation tests that avoid corners
4. Merge PR, document limitation

### For #521 (Next):
1. Implement corner averaging or priority ordering
2. Add validation tests with corner conflicts
3. Update FDMApplicator.enforce_values()

### For #549 (Future):
1. Extract BoundaryEnforcer protocol from FDMApplicator
2. Implement MeshEnforcer, LevelSetEnforcer
3. Migrate tensor-grid logic to TensorGridEnforcer

### For #545 (Parallel):
1. Use #542 refactor as template
2. Apply composition pattern to GFDM, Particle solvers
3. Define common BoundaryHandler interface

---

**Last Updated**: 2026-01-11
**Issue #542 Status**: Ready for merge pending final validation
