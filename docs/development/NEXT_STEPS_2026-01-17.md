# Next Steps - 2026-01-17

**Current Status**: ✅ Geometry/BC infrastructure complete (Issues #595, #591)

## Recently Completed (Today)

### Priority 6.6: LinearOperator Architecture (#595 Phase 2) ✅
- Created 3 operator classes (Divergence, Advection, Interpolation)
- 100% LinearOperator coverage
- scipy ecosystem integration
- ~220 lines infrastructure

### Priority 6.7: Variational Inequality Constraints (#591 Phase 2) ✅
- Protocol-based constraint system
- 3 constraint types (obstacle, bilateral, regional)
- HJB FDM integration
- 34 passing unit tests
- 3 physics-based validation examples
- ~2400 lines total

**GitHub Issues Updated**:
- #595: Marked Phase 2 complete
- #591: Marked Phase 2 substantially complete

---

## Options for Next Work

### Option 1: Continue Geometry/BC Architecture (Issues #589, #590)

**Issue #590**: Phase 1 - Geometry Trait System
- **Status**: Partially complete (operators done, traits/protocols remain)
- **Remaining work**:
  - Formalize geometry trait protocols
  - Region registry system
  - Documentation
- **Effort**: 1-2 weeks
- **Priority**: HIGH
- **Rationale**: Complete the geometry trait foundation before moving on

**Issue #592**: Phase 3 - Tier 3 BCs (Level Set Method)
- **Status**: Not started
- **Work**: Dynamic interfaces, free boundaries
- **Effort**: 3-4 weeks
- **Priority**: MEDIUM
- **Rationale**: Continue BC architecture roadmap

**Issue #593**: Phase 4 - Advanced BC Methods
- **Status**: Not started
- **Work**: Nitsche's method, GKS validation
- **Effort**: 2-3 weeks
- **Priority**: MEDIUM

---

### Option 2: Solver Refactoring (Issue #545)

**Issue #545**: Mixin Refactoring - Remaining Solvers
- **Status**: Phase 1 (BoundaryHandler protocol) complete, Phase 2 (FPParticle) complete
- **Remaining**: GFDM, FDM, FEM, DGM solvers
- **Effort**: 10-15 days total
- **Priority**: HIGH (per PRIORITY_LIST_2026-01.md as P7)
- **Rationale**: Apply composition pattern to remaining solvers

**Benefits**:
- Eliminate mixin complexity
- Clear separation of concerns
- Template method pattern
- Easier testing and maintenance

**Approach**: Follow FPParticle template pattern

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

**Issue #598**: BCApplicatorProtocol → ABC Refactoring (NEW)
- **Status**: Open (created 2026-01-17)
- **Work**: Convert to ABC with Template Method Pattern
- **Effort**: 2-3 days
- **Priority**: MEDIUM
- **Benefit**: Eliminate ghost cell logic duplication
- **Reference**: `docs/development/PROTOCOL_DUCK_TYPING_ANALYSIS.md` §5

**Issue #583**: HJB Semi-Lagrangian NaN values
- **Status**: Open
- **Work**: Debug cubic interpolation producing NaNs
- **Effort**: Small-Medium
- **Priority**: LOW (affects semi-lagrangian only)

**Issue #577**: Neumann BC Ghost Cell Consolidation (Phase 3)
- **Status**: Phases 1-2 complete, cleanup remains
- **Work**: Remove deprecated code
- **Effort**: 1-2 days
- **Priority**: LOW (scheduled for v0.19.0)

**Pre-existing HJB Running Cost Bug** (from #591 work)
- **Status**: Discovered but not tracked
- **Work**: Fix sign error in running cost incorporation
- **Impact**: Currently blocks proper obstacle problems with Hamiltonians
- **Effort**: Unknown (needs investigation)
- **Priority**: MEDIUM-HIGH
- **Action**: Create new issue to track

---

### Option 5: Documentation and Testing

**Issue #594**: Phase 5-6 - Documentation & Testing
- **Status**: Not started
- **Work**: Complete documentation, >90% coverage
- **Effort**: 3 weeks
- **Priority**: MEDIUM
- **Rationale**: Consolidate recent work with comprehensive docs

**Priority List Maintenance**:
- Update completed work
- Archive obsolete documentation
- Consolidate development docs

---

## Recommendation

Based on PRIORITY_LIST_2026-01.md progression and code maturity:

### **Recommended: Option 2 - Solver Refactoring (#545 Phase 3)**

**Rationale**:
1. **Follows priority sequence**: Listed as Priority 7 in roadmap
2. **Builds on completed work**: BoundaryHandler protocol and FPParticle template done
3. **High impact**: Affects all major solvers (GFDM, FDM, FEM, DGM)
4. **Infrastructure improvement**: Cleaner architecture for future development
5. **Manageable scope**: Well-defined pattern to follow

**Approach**:
- Apply FPParticle composition pattern to remaining solvers
- Start with FDM (simplest), then GFDM, FEM, DGM
- Estimate ~2-3 days per solver
- Total: 10-15 days

### **Alternative: Option 1 - Complete Geometry Trait System (#590)**

**If prioritizing geometry/BC completion**:
- Formalize trait protocols (SupportsLaplacian, SupportsGradient, etc.)
- Implement region registry
- Complete #590 before moving to #592-#594
- Maintains focus on geometry/BC architecture track

**Rationale**:
- Completes the geometry/BC foundation
- Enables #596 (solver integration with traits)
- Natural continuation of today's work

---

## Decision Criteria

**Choose Option 2 (Solver Refactoring)** if:
- Want to complete high-priority infrastructure roadmap
- Prefer to diversify work (switch from geometry to solvers)
- Value immediate impact on solver architecture

**Choose Option 1 (Geometry Traits)** if:
- Want to complete the geometry/BC architecture track
- Prefer to finish what was started (master issue #589)
- Value coherent feature completion

**Choose Option 3 (Algorithm Enhancements)** if:
- Need specific algorithmic capabilities (#573 for non-quadratic H)
- Research requirements drive priorities

**Choose Option 4 (Bug Fixes)** if:
- Stability and correctness take precedence
- HJB running cost bug is blocking research
- Quick wins preferred

**Choose Option 5 (Documentation)** if:
- Need to consolidate recent work
- Preparing for release
- User-facing documentation is priority

---

## Open Questions for User

1. **Which track do you prefer?**
   - Continue geometry/BC architecture (Options 1)?
   - Switch to solver refactoring (Option 2)?
   - Focus on algorithms (Option 3)?
   - Address bugs (Option 4)?

2. **Should we create a GitHub issue for the HJB running cost bug?**
   - Discovered during #591 work
   - Blocks proper obstacle problems with Hamiltonians
   - Workaround: Use heat equation examples

3. **What's the priority for completing #589 (Geometry & BC Architecture)?**
   - Complete all phases (#590-#594)?
   - Or proceed to other work and return later?

---

**Last Updated**: 2026-01-17
**Current Focus**: Completed P6.6 (#595), P6.7 (#591)
**Recommended Next**: P7 - Solver Refactoring (#545) OR complete #590 (Geometry Traits)
