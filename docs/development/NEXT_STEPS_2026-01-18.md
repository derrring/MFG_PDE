# Next Steps - 2026-01-18

**Current Status**: ✅ Priority 7 (Solver Mixin Refactoring) complete

## Recently Completed (Today)

### Priority 7: Solver Mixin Refactoring (#545) ✅
- Audited all solvers for mixin usage
- **Finding**: Refactoring already complete from Issue #545 (2026-01-11)
- Cleanup: Deleted unused MonotonicityMixin file + 3 .pyc files
- Updated outdated comments in hjb_gfdm.py
- Tests: 39/40 HJB FDM passing (1 pre-existing failure)

**Key Achievement**: All solvers now use explicit composition or simple inheritance. Zero active mixin usage.

---

## Options for Next Work

### Option 1: Legacy Parameter Deprecation (Priority 8, Issue #544)

**Issue #544**: Deprecate legacy MFGProblem parameters
- **Status**: Not started
- **Work**: Add DeprecationWarning, migrate examples/tests to Geometry API
- **Effort**: 5-7 days
- **Priority**: HIGH (user-facing API cleanup)

**Rationale**: Foundation is stable (Protocols, BC handling, solver refactoring complete). Time to clean up legacy API before v1.0.0.

**Approach**:
1. Add DeprecationWarning to MFGProblem.__init__ for Nx, xmin, xmax parameters
2. Migrate all examples/ to Geometry API (TensorProductGrid)
3. Migrate all tests/ to Geometry API
4. Document migration path in `docs/migration/LEGACY_PARAMETERS.md`
5. Schedule removal for v0.18.0 or v1.0.0

---

### Option 2: Continue Geometry/BC Architecture (Issues #589, #590)

**Issue #590**: Phase 1 - Geometry Trait System
- **Status**: Partially complete (operators done, traits/protocols remain)
- **Remaining work**:
  - Formalize geometry trait protocols
  - Region registry system
  - Documentation
- **Effort**: 1-2 weeks
- **Priority**: HIGH

**Rationale**: Complete the geometry trait foundation started in earlier work.

---

### Option 3: Algorithm Enhancements

**Issue #573**: Callable Drift Interface for Non-Quadratic H
- **Status**: Not started
- **Work**: Generalize FP solver to support arbitrary Hamiltonians
- **Effort**: Medium (1-2 weeks)
- **Priority**: HIGH
- **Impact**: Enables broader class of MFG problems

**Issue #596**: Phase 2 - Solver Integration with Geometry Traits
- **Status**: Not started
- **Work**: Integrate trait system into all solvers
- **Effort**: Large (3-4 weeks)
- **Priority**: HIGH
- **Dependencies**: Requires #590 complete

---

### Option 4: Bug Fixes and Cleanup

**Issue #598**: BCApplicatorProtocol → ABC Refactoring
- **Status**: Open (created 2026-01-17)
- **Work**: Convert to ABC with Template Method Pattern
- **Effort**: 2-3 days
- **Priority**: MEDIUM

**Issue #583**: HJB Semi-Lagrangian NaN values
- **Status**: Open
- **Work**: Debug cubic interpolation producing NaNs
- **Effort**: Small-Medium
- **Priority**: LOW

**Issue #577**: Neumann BC Ghost Cell Consolidation (Phase 3)
- **Status**: Phases 1-2 complete, cleanup remains
- **Work**: Remove deprecated code
- **Effort**: 1-2 days
- **Priority**: LOW (scheduled for v0.19.0)

---

## Recommendation

Based on completed infrastructure work (Priorities 1-7) and roadmap progression:

### **Recommended: Option 1 - Legacy Parameter Deprecation (Priority 8, Issue #544)**

**Rationale**:
1. **Foundation complete**: All infrastructure refactoring done (Protocols, BC handling, mixin elimination)
2. **High priority**: User-facing API cleanup before v1.0.0
3. **Clear scope**: Well-defined migration path
4. **Manageable effort**: 5-7 days, mostly mechanical changes
5. **Unblocks future**: Clean API enables easier feature development

**Approach**:
- Add DeprecationWarning to legacy parameters
- Migrate ~50 examples to Geometry API
- Migrate ~150 tests to Geometry API
- Document migration in dedicated guide
- Schedule removal for v0.18.0 or v1.0.0

### **Alternative: Option 2 - Complete Geometry Trait System (#590)**

**If prioritizing geometry/BC completion**:
- Formalize trait protocols (SupportsLaplacian, SupportsGradient, etc.)
- Implement region registry
- Complete #590 before moving to #592-#594
- Maintains focus on geometry/BC architecture track

---

## Decision Criteria

**Choose Option 1 (Legacy Deprecation)** if:
- Want to complete high-priority API cleanup
- Preparing for v1.0.0 release
- Prefer user-facing improvements

**Choose Option 2 (Geometry Traits)** if:
- Want to complete geometry/BC architecture
- Prefer to finish infrastructure before API changes
- Research needs geometry trait features

**Choose Option 3 (Algorithm Enhancements)** if:
- Research requirements drive priorities
- Need non-quadratic Hamiltonian support (#573)

**Choose Option 4 (Bug Fixes)** if:
- Stability and correctness take precedence
- Quick wins preferred

---

## Progress Summary

**Completed Priorities (1-7)**:
- ✅ P1: FDM Periodic BC Bug (#542)
- ✅ P2: Silent Fallbacks Elimination (#547)
- ✅ P3: hasattr() Elimination Phase 1 (#543)
- ✅ P3.5: Adjoint-Aware Solver Pairing (#580)
- ✅ P3.6: Unified Ghost Node Architecture (#576)
- ✅ P4: Mixin Refactoring - FPParticle Template (#545)
- ✅ P5: hasattr() Elimination Phase 2 (#543)
- ✅ P5.5: Progress Bar Protocol Pattern (#587)
- ✅ P6: hasattr() Elimination Phase 3 (#543)
- ✅ P6.5: Adjoint-Consistent Boundary Conditions (#574)
- ✅ P6.6: LinearOperator Architecture (#595)
- ✅ P6.7: Variational Inequality Constraints (#591)
- ✅ P7: Mixin Refactoring - Remaining Solvers (#545)

**Remaining Priorities**:
- 🎯 P8: Legacy Parameter Deprecation (#544)

---

**Last Updated**: 2026-01-18
**Current Focus**: Completed P7 (Solver Mixin Refactoring cleanup)
**Recommended Next**: P8 - Legacy Parameter Deprecation (#544)
