# MFG_PDE Architecture Documentation Index

**Complete guide to architecture analysis and audit documentation**

---

## Primary Documents

### 1. Factory Pattern Design (UPDATED - 2026-01-17)
**File**: `FACTORY_PATTERN_DESIGN.md`
**Purpose**: Three-concern factory architecture for MFG_PDE
**Version**: 1.3 (Consolidated with infrastructure audit)

**What's Inside**:
- Three-concern separation: WHAT (problem) / HOW (scheme) / WHO (assembly)
- Three-mode solving API: Safe / Expert / Auto modes
- Current implementation status (baseline before Issue #580)
- Mathematical foundation: Discrete vs asymptotic duality (Type A/B)
- Critical implementation considerations (3 technical risks addressed)
- Anti-confusion strategy: Single entry point via `problem.solve()`
- Complete implementation roadmap for Issue #580

**Key Insights**:
- `problem.solve()` is the ONLY user-facing entry point
- Factory functions are internal implementation details
- Clear separation between problem configuration and scheme selection
- Current state: GFDM+Particle hardcoded (line 1954-2026)
- Target state: Intelligent scheme selection with user control

**Archive**: Infrastructure audit moved to `docs/archive/issue_580_factory_pattern_2026-01/`

**Read This If**: Implementing Issue #580 or understanding solver pairing architecture

---

### 2. Architecture Audit Enrichment
**File**: `ARCHITECTURE_AUDIT_ENRICHMENT.md`
**Pages**: 45
**Purpose**: Empirical evidence from 3 weeks of research validating original audit

**What's Inside**:
- Evidence catalog: 48 problems, chronological with file references
- User demand analysis: Blocked requests and pain points
- Mathematical notation survey: Theory docs across experiments
- Architecture pain points: Quantified with code line counts
- Cross-reference: Original audit vs research findings
- Updated recommendations: Priority re-ranking based on impact

**Key Statistics**:
- 181 documentation files analyzed
- 94 test/experiment files reviewed
- 45 hours lost to architecture issues
- 3,285 lines of duplicate code
- 7.6× integration overhead measured

**Read This If**: You want comprehensive, evidence-based validation of audit findings

---

### 2. Audit Enrichment Summary (NEW)
**File**: `AUDIT_ENRICHMENT_SUMMARY.md`
**Pages**: 8
**Purpose**: Executive summary with quick statistics

**What's Inside**:
- Critical findings (5 key issues)
- Blocked user requests (5 requests)
- Severity upgrades (5 findings)
- Revised priorities (3 tiers)
- Quick recommendations (immediate actions)

**Read This If**: You need a quick overview (15 minutes)

---

### 3. Original Architecture Audit
**File**: `experiments/maze_navigation/MFG_PDE_ARCHITECTURE_AUDIT.md`
**Pages**: 200+
**Date**: 2025-10-30 (before research)
**Purpose**: Comprehensive theoretical analysis of MFG_PDE architecture

**What's Inside**:
- Part 1: What proposal got right
- Part 2: Critical infrastructure missed
- Part 3: Complexity underestimated
- Part 4: Recommended architecture
- Part 5: Migration strategy

**Read This If**: You want the complete theoretical foundation

---

### 4. Architecture Refactoring Proposal
**File**: `MFG_PDE_ARCHITECTURE_REFACTOR_PROPOSAL.md`
**Purpose**: Original refactoring proposal (pre-audit)

**Status**: Needs revision based on audit and research findings

---

## Bug Documentation

### Bug #13: Gradient Key Mismatch
**Location**: `docs/archived_bug_investigations/`
**Index**: `docs/archived_bug_investigations/BUG_13_INDEX.md`
**Documents**: 15 comprehensive files

**Key Files**:
- `BUG_13_QUICK_REFERENCE.md` - 5-minute overview
- `BUG_13_FINAL_SUMMARY.md` - Executive summary
- `BUG_13_RESOLVED.md` - Complete technical resolution
- `BUG_13_LESSONS_LEARNED.md` - 18 lessons (45 min read)

**Impact**: 3 days debugging, 2-character fix in gradient dictionary keys

---

### Bug #14: GFDM Gradient Sign Error
**Location**: `experiments/maze_navigation/archives/bugs/bug14_gfdm_sign/`
**Main Report**: `BUG_14_MFG_PDE_REPORT.md`
**Documents**: 10 files

**Status**: FIXED, merged, GitHub issue filed

**Impact**: All GFDM results incorrect before fix, agents moved away from goals

---

### Bug #15: QP Sigma Type Error
**Location**: `experiments/maze_navigation/archives/bugs/bug15_qp_sigma/`
**Main Report**: `BUG_15_QP_SIGMA_METHOD.md`
**Documents**: 7 files

**Status**: Workaround exists (`SmartSigma`), NOT FIXED in MFG_PDE

**Impact**: Cannot use QP constraints without workaround

---

### Anderson Accelerator Issue
**Location**: `experiments/maze_navigation/`
**Files**:
- `ANDERSON_ISSUE_POSTED.md` - Summary
- `GITHUB_ISSUE_ANDERSON.md` - Full GitHub issue text

**Status**: GitHub Issue #199 filed

**Impact**: Cannot use on 2D arrays (MFG density naturally 2D)

---

## Architecture Analysis Documents

### FDM Solver Limitation
**File**: `experiments/maze_navigation/FDM_SOLVER_LIMITATION_ANALYSIS.md`
**Discovery**: 2025-10-30
**Status**: BLOCKS user request for FDM baseline comparison

**Key Finding**: FDM solvers are 1D-only, cannot work on 2D maze despite maze having regular grid

---

### API Issues Comprehensive Log
**File**: `experiments/maze_navigation/archives/investigations_2025-10/completed_investigations/API_ISSUES_LOG.md`
**Date**: 2025-10-20
**Content**: Import errors, class name mismatches, function naming issues

**Impact**: Blocked all experiment execution for 1 session

---

### QP Investigation Series
**Location**: `experiments/maze_navigation/archives/investigations_2025-10/qp_investigation_2025-10-25/`

**Key Files**:
- `QP_ANALYSIS_COMPREHENSIVE.md`
- `QP_THEORY_IMPLEMENTATION_ANALYSIS.md`
- `QP_OPTIMIZATION_COMPLETE.md`
- `QP_VALIDATION_STATUS.md`

**Scope**: Complete investigation of QP constraints for particle collocation

---

## Session Logs

### Recent Sessions
**Location**: `experiments/maze_navigation/archives/sessions/2025-10/`

**Key Sessions**:
- `SESSION_2025-10-28_FINAL_SUMMARY.md` - Complete wrap-up
- `SESSION_2025-10-27_PR197_DEMO.md` - PR demonstration
- `SESSION_STATUS_2025-10-26_QP_VALIDATION.md` - QP validation session
- `SESSION_2025-10-25_BUG14_DISCOVERY.md` - Bug #14 discovery

**Total**: 20+ session documents spanning 3 weeks

---

### Bug #13 Resolution Session
**File**: `docs/archived_bug_investigations/SESSION_2025-10-24_BUG13_RESOLUTION.md`
**Content**: Complete session record of Bug #13 investigation and resolution

---

## Theory and Mathematical Notation

### Primary Theory Document
**File**: `experiments/anisotropic_crowd_qp/docs/theory/theory.md`
**Length**: 100+ pages
**Content**: Complete mathematical formulation of particle-collocation with QP constraints

**Sections**:
1. Introduction and motivation
2. Mathematical framework (HJB, FP, MFG system)
3. Particle-collocation framework
4. Monotonicity via QP constraints
5. Anisotropic problems
6. Convergence analysis
7. Computational complexity
8. Implementation notes

---

### Supporting Theory
**Files**:
- `docs/FP_HJB_COUPLING_THEORY.md` - Coupling analysis
- `experiments/maze_navigation/archives/investigations_2025-10/qp_investigation_2025-10-25/QP_THEORY_IMPLEMENTATION_ANALYSIS.md`

---

## Analysis Documents

### Particle Collocation Architecture
**File**: `experiments/maze_navigation/analysis/PARTICLE_COLLOCATION_ARCHITECTURE_ANALYSIS.md`
**Content**: Deep dive into particle-collocation design patterns

---

### Adaptive Neighborhoods
**Files**:
- `experiments/maze_navigation/archives/investigations_2025-10/adaptive_neighborhoods_2025-10-26/ADAPTIVE_NEIGHBORHOODS.md`
- `experiments/maze_navigation/demo_adaptive_neighborhoods.py`

**Content**: Solution to Bug #6 (k_neighbors performance near obstacles)

---

## Test Files

### Maze Navigation Tests
**Location**: `experiments/maze_navigation/`
**Count**: 17 test files

**Key Tests**:
- `test_solver_comparison.py` - FDM vs GFDM (FDM failed)
- `test_qp_full_validation.py` - Complete QP validation
- `test_hamiltonian_types.py` - Hamiltonian API compatibility
- `test_gfdm_derivative_types.py` - Gradient format issues
- `test_osqp_integration.py` - QP solver integration
- `test_adaptive_integration.py` - Adaptive neighborhoods

---

## How to Navigate This Documentation

### For Quick Understanding (30 minutes)
1. Read `AUDIT_ENRICHMENT_SUMMARY.md` (15 min)
2. Read `BUG_13_QUICK_REFERENCE.md` (5 min)
3. Skim `FDM_SOLVER_LIMITATION_ANALYSIS.md` (10 min)

### For Complete Understanding (4 hours)
1. `AUDIT_ENRICHMENT_SUMMARY.md` (15 min)
2. `ARCHITECTURE_AUDIT_ENRICHMENT.md` (2 hours)
3. `MFG_PDE_ARCHITECTURE_AUDIT.md` Part 1-2 (1 hour)
4. Bug #13, #14, #15 main reports (45 min)

### For Specific Issues
**Problem class confusion** → `ARCHITECTURE_AUDIT_ENRICHMENT.md` Section 4.1
**FDM limitation** → `FDM_SOLVER_LIMITATION_ANALYSIS.md`
**QP issues** → `BUG_15_QP_SIGMA_METHOD.md`
**Anderson** → `ANDERSON_ISSUE_POSTED.md`
**API incompatibilities** → `API_ISSUES_LOG.md`

### For Implementing Fixes
**Bug #15 fix** → `BUG_15_FINAL_FIX.md`
**Anderson fix** → `GITHUB_ISSUE_ANDERSON.md` (proposed fix section)
**2D FDM** → `FDM_SOLVER_LIMITATION_ANALYSIS.md` (Section: What Would Be Needed)

---

## Document Statistics

**Total Documentation**:
- Architecture: 4 major documents (250+ pages total)
- Bug reports: 3 series (32 files total)
- Session logs: 25+ files
- Analysis: 15+ deep-dive documents
- Theory: 2 major documents (100+ pages)
- Tests: 94 Python files

**Evidence Quality**:
- Every claim has file:line reference
- Quantitative measurements throughout
- Complete code examples
- Before/after comparisons
- Test results and logs

---

## Reading Paths

### Path 1: Executive (1 hour)
```
1. AUDIT_ENRICHMENT_SUMMARY.md
2. Original audit Executive Summary
3. Bug #13 Quick Reference
4. FDM Limitation Analysis (skim)
→ Outcome: Understand scope and urgency
```

### Path 2: Technical (4 hours)
```
1. ARCHITECTURE_AUDIT_ENRICHMENT.md (complete)
2. MFG_PDE_ARCHITECTURE_AUDIT.md Parts 1-2
3. Bug #13 Lessons Learned
4. QP Analysis Comprehensive
→ Outcome: Full technical understanding
```

### Path 3: Implementation (6 hours)
```
1. All bug reports (detailed)
2. Test files analysis
3. Theory documents (mathematical foundation)
4. Session logs (see process)
→ Outcome: Ready to implement fixes
```

### Path 4: Research Methodology (3 hours)
```
1. Bug #13 investigation trail (15 files)
2. Session logs (chronological)
3. Lessons Learned documents
→ Outcome: Learn debugging methodology
```

---

## Key Takeaways

**From Audit**:
- Problem fragmentation is real and critical
- FDM limitation blocks scientific use
- Missing abstractions cause massive duplication

**From Research**:
- 3 critical bugs in 3 weeks (1/week rate)
- 7.6× integration overhead measured
- 3,285 lines of duplicate code quantified
- 45 hours lost to architecture issues

**Action Required**:
1. Fix Bug #15 and Anderson (2 weeks)
2. Implement 2D FDM (4-6 weeks)
3. Begin unified problem class (8-10 weeks)

---

## Maintenance

**This index is updated**: 2025-10-30

**Update triggers**:
- New architecture documents created
- New bugs discovered
- Session summaries added
- Audit findings revised

**Maintainer**: Research team (mfg-research)

---

## Related Documentation

**In MFG_PDE Repository**:
- `ARCHITECTURE.md` (if exists)
- `CONTRIBUTING.md`
- API documentation

**GitHub Issues**:
- Issue #199: Anderson Accelerator multi-dimensional
- Issue #??? : GFDM gradient sign error (Bug #14)

**Research Roadmap**:
- `experiments/anisotropic_crowd_qp/docs/PROJECT_STATUS.md`
- `experiments/maze_navigation/SESSION_STATUS_2025-10-29.md`

---

**Quick Access Matrix**:

| Need | Document | Time |
|------|----------|------|
| Quick stats | AUDIT_ENRICHMENT_SUMMARY.md | 15 min |
| Full evidence | ARCHITECTURE_AUDIT_ENRICHMENT.md | 2 hours |
| Theory | anisotropic_crowd_qp/docs/theory/theory.md | 3 hours |
| Bug #13 | BUG_13_INDEX.md → Quick Reference | 5 min |
| Bug #14 | BUG_14_MFG_PDE_REPORT.md | 30 min |
| Bug #15 | BUG_15_QP_SIGMA_METHOD.md | 20 min |
| FDM issue | FDM_SOLVER_LIMITATION_ANALYSIS.md | 15 min |
| Anderson | ANDERSON_ISSUE_POSTED.md | 10 min |

---

**This index provides a complete navigation system for 250+ pages of architecture documentation spanning 3 weeks of intensive research.**
