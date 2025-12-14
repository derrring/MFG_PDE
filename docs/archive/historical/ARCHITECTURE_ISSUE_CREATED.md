# Architecture Refactoring Issue Created

**Date**: 2025-10-30
**GitHub Issue**: [#200](https://github.com/derrring/MFG_PDE/issues/200)

---

## Summary

Comprehensive architecture refactoring issue created in MFG_PDE repository based on 3 weeks of intensive research usage (2025-10-06 to 2025-10-30).

---

## Issue Details

**Title**: Architecture Refactoring: Unified Problem Classes and Multi-Dimensional Solver Support

**URL**: https://github.com/derrring/MFG_PDE/issues/200

**Labels**:
- `priority: high`
- `type: infrastructure`
- `area: algorithms`
- `size: large`
- `enhancement`
- `help wanted`

---

## Key Statistics Included

**Research Data**:
- 48 documented architectural issues (6 CRITICAL, 15 HIGH, 18 MEDIUM, 9 LOW)
- 3 critical bugs discovered (Bug #13, #14, #15)
- 5 features permanently or temporarily blocked
- 45 hours lost to workarounds over 3 weeks
- 3,285 lines of duplicate/adapter code written
- 7.6× integration overhead measured
- Projected annual cost: 780 hours/year (19.5 work-weeks)

---

## Repository State Verified

**MFG_PDE Branches**:
- `main` (current)
- `feature/osqp-and-adaptive-neighborhoods` (remote)
- `refactor/remove-use-monotone-constraints` (remote)

**Open Issues Referenced**:
- Issue #199: Anderson 1D limitation
- Issue #196: QP performance
- Issue #191: Barrier implementations

**Merged PRs Referenced**:
- PR #193: Bug #14 (GFDM gradient sign error) - MERGED
- PR #197: OSQP optimization + adaptive neighborhoods - MERGED

**Pending PRs**:
- PR #194: CI dependency update (actions/upload-artifact)
- PR #192: CI dependency update (actions/setup-python)

---

## Documentation Organized

All comprehensive documentation is in `MFG_PDE/docs/architecture/`:

```
docs/architecture/
├── README.md                              # Central entry point (447 lines)
├── audit/
│   ├── AUDIT_ENRICHMENT_SUMMARY.md        # Executive summary (371 lines)
│   ├── ARCHITECTURE_AUDIT_ENRICHMENT.md   # Full audit (1,799 lines)
│   ├── MFG_PDE_ARCHITECTURE_AUDIT.md      # Original audit
│   └── ARCHITECTURE_AUDIT_SUMMARY.md      # Original summary
├── proposals/
│   └── MFG_PDE_ARCHITECTURE_REFACTOR_PROPOSAL.md
├── evidence/
│   └── FDM_SOLVER_LIMITATION_ANALYSIS.md
└── navigation/
    └── ARCHITECTURE_DOCUMENTATION_INDEX.md
```

**Total**: 9 markdown files, 6,361 lines, 204 KB

---

## Critical Findings Highlighted

### 1. FDM Solvers Limited to 1D (CRITICAL)
- User request for 2D FDM solver is PERMANENTLY BLOCKED
- `HJBFDMSolver` and `FPFDMSolver` only accept 1D `MFGProblem`
- 2D problems exist (`GridBasedMFGProblem`) but incompatible

### 2. Problem Class Fragmentation (CRITICAL)
- 5 competing problem classes
- Only 4 out of 25 HJB×FP combinations work natively for 2D
- 1,080 lines of custom problem code required

### 3. Bug #15: QP Sigma Type Error (HIGH PRIORITY)
- NOT YET FIXED in MFG_PDE
- Workaround exists (`SmartSigma`)
- 5-line fix proposed in issue

### 4. Anderson Accelerator 1D-Only (MEDIUM)
- Related to Issue #199
- Requires 25 lines boilerplate per usage
- 10-line fix proposed in issue

---

## Refactoring Timeline

**Phase 1: Immediate Fixes (2 weeks)**
1. Fix Bug #15 (QP sigma API) - 1 day
2. Fix Anderson multi-dimensional - 3 days
3. Standardize gradient notation - 1 week

**Phase 2: High-Priority (3 months)**
1. Implement 2D/3D FDM solvers - 4-6 weeks
2. Add missing utilities - 4 weeks
3. Quick wins - 1 week

**Phase 3: Long-Term (6-9 months)**
1. Unified problem class - 8-10 weeks
2. Configuration simplification - 2-3 weeks
3. Backend integration - 2 weeks

**Total Estimated**: 22-37 weeks (5.5 to 9 months)

---

## Next Steps

### For Maintainers

**Week 1-2**:
- Review `docs/architecture/audit/AUDIT_ENRICHMENT_SUMMARY.md`
- Apply Bug #15 fix (5-line patch)
- Fix Anderson accelerator (10-line patch)
- Write regression tests

**Month 1-3**:
- Begin FDM 2D/3D implementation
- Add missing utilities
- Implement quick wins

**Month 4-9**:
- Follow 6-phase migration plan
- Maintain backward compatibility
- Gradual rollout with deprecation warnings

### For Researchers

**Workarounds Available**:
- QP sigma: Use `SmartSigma` wrapper class
- Anderson 2D: Manual flatten/reshape
- FDM 2D: Use GFDM instead
- Particle interpolation: Custom utility functions

---

## Evidence Quality

**Documentation Trail**:
- 181 markdown files analyzed
- 94 Python test/experiment files
- 251 Python files in MFG_PDE (100% coverage)
- 15 documents for Bug #13
- 10 documents for Bug #14
- 7 documents for Bug #15

**Quantification**:
- Time measurements (45 hours total)
- Code line counts (3,285 duplicate lines)
- Efficiency ratios (7.6× overhead)
- Bug discovery rate (1 per week)
- Every claim backed by file:line references

---

## Related Issues in MFG_PDE

**This Issue Consolidates**:
- Bug #15: QP sigma type error (NEW)
- FDM 2D/3D limitation (NEW)
- Problem class fragmentation (NEW)

**References Existing Issues**:
- Issue #199: Anderson 1D limitation
- Issue #196: QP performance (40-80× slower)
- Issue #191: Barrier implementations

**References Merged Work**:
- PR #193: Bug #14 GFDM gradient sign error
- PR #197: OSQP optimization + adaptive neighborhoods

---

## Testing Strategy Proposed

1. **Solver Combination Matrix Tests**
   - Test all HJB×FP combinations
   - Test across dimensions (1D, 2D, 3D)
   - Clear errors for unsupported combinations

2. **Gradient Format Consistency Tests**
   - Prevent Bug #13 recurrence

3. **API Contract Tests**
   - Prevent Bug #15 recurrence

4. **Real Problem Integration Tests**
   - Maze navigation (2D + obstacles)
   - Crowd dynamics (anisotropic)

5. **Performance Regression Tests**
   - QP performance (<10ms for 100 variables)

---

## Ideal Future API Documented

```python
from mfg_pde import MFGProblem, solve_mfg

# Single problem class for all dimensions
problem = MFGProblem(
    dimension=2,
    domain=[(0, 6), (0, 6)],
    obstacles=maze,
    time_domain=(0, 1.0),
    diffusion=0.1,
    hamiltonian=my_hamiltonian,
    initial_density=m0,
    terminal_condition=uT
)

# Automatic solver selection
solution = solve_mfg(problem, method="auto", resolution=50, backend="auto")

# Consistent result format
U, M, info = solution
```

---

## Status

- ✅ Repository state verified (branches, issues, PRs)
- ✅ Comprehensive issue created (#200)
- ✅ Appropriate labels added
- ✅ All documentation organized in `MFG_PDE/docs/architecture/`
- ✅ Related issues and PRs cross-referenced
- ✅ Evidence and quantification included
- ✅ Clear action plan with timeline
- ✅ Maintainer checklist provided
- ✅ Testing strategy proposed

**Ready for maintainer review and prioritization.**

---

## Contact

**For Questions**:
- GitHub Issue: https://github.com/derrring/MFG_PDE/issues/200
- Full documentation: `MFG_PDE/docs/architecture/README.md`
- Quick navigation: `MFG_PDE/docs/architecture/navigation/ARCHITECTURE_DOCUMENTATION_INDEX.md`

---

**Created**: 2025-10-30
**Audit Period**: 2025-10-06 to 2025-10-30 (3 weeks)
**Research Cost**: 45 hours, 3,285 lines, 7.6× overhead
**Projected Annual Impact**: 780 hours/year
