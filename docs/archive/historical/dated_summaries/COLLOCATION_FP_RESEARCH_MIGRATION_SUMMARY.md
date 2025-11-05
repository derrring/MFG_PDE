# Collocation FP: Research → Production Migration Summary

**Date**: 2025-11-04
**Issue**: #240
**Status**: NO CODE TO MIGRATE (research uses production's incomplete implementation)

---

## Investigation Summary

### Research Repository Status

**Finding**: Research repository does NOT have an independent collocation FP implementation.

**Evidence**:
- Research proposal exists: `mfg-research/experiments/maze_navigation/PURE_PARTICLE_FP_PROPOSAL.md` (dated 2025-10-29)
- Proposal is DESIGN ONLY - no implementation code
- Research solver delegates to production `FPParticleSolver` in collocation mode
- Research is blocked by same incomplete implementation as production

**Source** (from PARTICLE_COLLOCATION_OVERLAP_ANALYSIS_2025-11-02.md):
```python
# Research: algorithms/particle_collocation/solver.py, lines 99-111
from mfg_pde.alg.numerical.fp_solvers.fp_particle import FPParticleSolver

self.fp_solver = FPParticleSolver(
    problem=problem,
    mode="collocation",  # ← Uses production code!
    external_particles=collocation_points,
)
```

---

## Migration Decision

### What to Migrate: NONE

**Reason**: No separate research implementation exists

**Actions Taken**:
1. ✅ Reviewed research proposal (PURE_PARTICLE_FP_PROPOSAL.md)
2. ✅ Confirmed research uses production code
3. ✅ Documented findings in this summary
4. ✅ Used research proposal as design reference for production plan

### What to Reference

**Research Design Proposal** (`PURE_PARTICLE_FP_PROPOSAL.md`):
- Conceptual framework for pure particle FP (particles in → particles out)
- GFDM integration approach
- Dimension-agnostic design philosophy
- **Status**: Used as reference in `COLLOCATION_FP_IMPLEMENTATION_PLAN.md`

---

## Implementation Approach

### Production-Only Development

Since no research code exists, implementation proceeds entirely in production (MFG_PDE):

**Phase 1: Documentation (Immediate)**
- Update `fp_particle.py` docstring
- Add `UserWarning` in collocation mode
- Update examples with disclaimer
- **Location**: Production only

**Phase 2-4: Implementation (v1.0.0)**
- Implement GFDM operators (divergence, Laplacian)
- Implement time integration (explicit Euler)
- Add validation tests
- **Location**: Production only

**No Migration Required**: All work done in MFG_PDE production repository

---

## Research Repository Actions

### Recommendations for mfg-research

1. **Update Documentation**:
   - Mark `PURE_PARTICLE_FP_PROPOSAL.md` as design reference (not implementation)
   - Add note that production implementation is tracked in issue #240
   - Link to production implementation plan

2. **Monitor Production Progress**:
   - Research can use completed collocation FP once production implements it
   - No separate research implementation needed
   - Import from production as before

3. **Optional**: Create research issue to track production dependency
   ```
   Title: Blocked: Awaiting production collocation FP implementation
   Body: Research solvers use production FPParticleSolver in collocation mode.
         Currently incomplete (density frozen).
         Tracking: MFG_PDE#240
   ```

---

## Files Analyzed

### Research Repository

| File | Purpose | Finding |
|:-----|:--------|:--------|
| `experiments/maze_navigation/PURE_PARTICLE_FP_PROPOSAL.md` | Design proposal | No code, design only |
| `algorithms/particle_collocation/solver.py` | MFG orchestrator | Uses production FP solver |
| `algorithms/particle_collocation/fp_particle_dual_mode.py` | Expected FP implementation | Does NOT exist |

**Conclusion**: Research has no collocation FP code to migrate.

### Production Repository (MFG_PDE)

| File | Status | Action |
|:-----|:-------|:-------|
| `mfg_pde/alg/numerical/fp_solvers/fp_particle.py` | Incomplete collocation mode | Fix in place (no migration) |
| `docs/development/COLLOCATION_FP_IMPLEMENTATION_PLAN.md` | New | Created (this session) |
| `docs/development/dual_mode_fp_particle_v0.8.0.md` | Existing | Referenced for context |

---

## Key Insights

1. **Research Uses Production**: Research repo imports production `FPParticleSolver` directly
2. **No Duplication**: No separate research implementation to consolidate
3. **Single Source of Truth**: Production MFG_PDE is sole implementation location
4. **Clean Architecture**: Follows CLAUDE.md principle - research imports infrastructure, doesn't modify it

---

## Timeline Impact

**Original Concern**: Need to migrate research code to production

**Actual**: NO MIGRATION NEEDED

**Time Saved**: ~1 week (migration, consolidation, testing)

**New Timeline**: Proceed directly to production implementation (~3 days for phases 1-4)

---

##Status Summary

| Task | Status | Notes |
|:-----|:-------|:------|
| **Research code review** | ✅ COMPLETE | No code found |
| **Migration assessment** | ✅ COMPLETE | No migration needed |
| **Design reference extraction** | ✅ COMPLETE | Proposal documented |
| **Production plan creation** | ✅ COMPLETE | Implementation plan ready |
| **GitHub issue creation** | ✅ COMPLETE | Issue #240 |
| **Code migration** | ✅ N/A | No code exists |

---

## Next Steps

### Immediate (This Session)
1. ✅ Complete this migration summary
2. ⏳ Update production fp_particle.py docstring
3. ⏳ Add UserWarning in collocation mode
4. ⏳ Update examples with disclaimer

### Near-Term (This Week)
5. Create feature branch `docs/collocation-fp-incomplete-warning`
6. Submit PR for Phase 1 (documentation updates)
7. Merge Phase 1 PR

### Medium-Term (Next Month)
8. Implement Phases 2-4 per `COLLOCATION_FP_IMPLEMENTATION_PLAN.md`
9. Target release: v1.0.0

---

## Related Documents

**Production (MFG_PDE)**:
- Implementation plan: `docs/development/COLLOCATION_FP_IMPLEMENTATION_PLAN.md`
- GitHub issue: #240
- Code location: `mfg_pde/alg/numerical/fp_solvers/fp_particle.py:597-657`

**Research (mfg-research)**:
- Design proposal: `experiments/maze_navigation/PURE_PARTICLE_FP_PROPOSAL.md`
- Analysis: `docs/archived/2025-11/PARTICLE_COLLOCATION_OVERLAP_ANALYSIS_2025-11-02.md` (MFG_PDE)

**Investigation Reports**:
- `/tmp/collocation_fp_status_report.md` - Detailed investigation
- `/tmp/particle_methods_comparison.md` - Full particle methods analysis

---

**Document Status**: COMPLETE
**Migration Required**: NO
**Next Action**: Phase 1 implementation (documentation updates)

**Conclusion**: Research repository has no collocation FP code. All work proceeds in production MFG_PDE repository per standard workflow. Clean separation maintained: research imports infrastructure, production provides it.
