# Documentation Consolidation Action Plan

**Date**: 2025-10-11
**Current Status**: 244 active docs (target: 125 per category limits)
**Priority**: HIGH - 119 docs need consolidation/archival

---

## Executive Summary

**Audit Results:**
- 📊 Total: 244 docs (3.2 MB)
- 🎯 Target: 125 docs per category limits
- 📉 Reduction needed: 119 docs (49%)

**Category Overages:**
1. ❌ **development/**: 163/35 (465%) - **PRIMARY TARGET**
2. ❌ **theory/**: 39/25 (156%)
3. ❌ **user/**: 24/15 (160%)
4. ✅ **reference/**: 3/50 (6%) - OK

---

## Phase 1: Quick Wins (Immediate - 50 files)

### 1A. Archive Completed Files (26 files)
**Impact**: 26 files → 0 (archive/)

```bash
# All [COMPLETED], [CLOSED], [RESOLVED] files → archive/
mkdir -p archive/development

# Move completed files
find docs/development -name "*COMPLETED*" -o -name "*CLOSED*" -o -name "*RESOLVED*" | \
  xargs -I {} git mv {} archive/development/
```

**Files to Archive:**
- `[COMPLETED]_KDE_NORMALIZATION_STRATEGY_2025-10-09.md`
- `[COMPLETED]_ACCELERATION_DECISION_2025-10-09.md`
- `[COMPLETED]_PHASE_2.2_STOCHASTIC_MFG_2025_10_05.md`
- `[COMPLETED]_PHASE2_BACKEND_INTEGRATION_COMPLETE.md`
- `[COMPLETED]_PHASE_3_5_COMPLETION_SUMMARY.md`
- `[COMPLETED]_POST_PHASE3_CLEANUP_SUMMARY.md`
- ... (20 more completed files)

### 1B. Consolidate Phase Summaries (15 files → 3 files)
**Impact**: 12 file reduction

**Create 3 comprehensive summaries:**

1. **`PHASE_2_COMPLETE.md`** (consolidate 4 files):
   - `PHASE2_SUMMARY_2025-10-08.md`
   - `PHASE2_VALIDATION_RESULTS_2025-10-08.md`
   - `[COMPLETED]_PHASE2_BACKEND_INTEGRATION_COMPLETE.md`
   - `[COMPLETED]_PHASE_2.2_STOCHASTIC_MFG_2025_10_05.md`

2. **`PHASE_3_COMPLETE.md`** (consolidate 6 files):
   - `PHASE_3_PREPARATION_2025.md`
   - `phase_3_3_completion_summary.md`
   - `phase_3_3_3_sac_implementation_summary.md`
   - `PHASE_3_PERFORMANCE_PROFILING_REPORT.md`
   - `[COMPLETED]_PHASE_3_5_COMPLETION_SUMMARY.md`
   - `[COMPLETED]_POST_PHASE3_CLEANUP_SUMMARY.md`

3. **`TEST_COVERAGE_SUMMARY.md`** (consolidate 2 files):
   - `test_coverage_phase14_summary.md`
   - Any other test coverage docs

**Archive originals** → `archive/development/phases/`

### 1C. Archive Session Summaries (15 files → 3 files)
**Impact**: 12 file reduction

**Strategy**: Keep only 3 most recent, archive the rest

```bash
# Move old sessions to archive
mkdir -p archive/development/sessions
find docs/development -name "SESSION_*" -o -name "*session*summary*" | \
  sort | head -n -3 | xargs -I {} git mv {} archive/development/sessions/
```

---

## Phase 2: Development/ Root Cleanup (53 files → 15 files)

### 2A. Consolidate Paradigm Overviews (5 files → 1 file)
**Impact**: 4 file reduction

**Create**: `docs/development/COMPUTATIONAL_PARADIGMS.md`

**Consolidate:**
- `NUMERICAL_PARADIGM_OVERVIEW.md`
- `NEURAL_PARADIGM_OVERVIEW.md`
- `OPTIMIZATION_PARADIGM_OVERVIEW.md`
- `REINFORCEMENT_LEARNING_PARADIGM_OVERVIEW.md`
- (Add STOCHASTIC if exists)

### 2B. Consolidate Architecture Docs (6 files → 1 file)
**Impact**: 5 file reduction (already started with `ARCHITECTURE.md`)

**Merge into existing `ARCHITECTURE.md`:**
- `MPI_INTEGRATION_TECHNICAL_DESIGN.md`
- `MULTI_POPULATION_ARCHITECTURE_DESIGN.md`
- `PHASE3_TIERED_BACKEND_STRATEGY.md`
- `BACKEND_AUTO_SWITCHING_DESIGN.md`
- (Already consolidated 2 others)

### 2C. Consolidate Code Quality Docs (5 files → 1 file)
**Impact**: 4 file reduction

**Create**: `docs/development/CODE_QUALITY.md`

**Consolidate:**
- `CONSISTENCY_GUIDE.md`
- `TYPE_SAFETY_IMPROVEMENT_SYNTHESIS.md`
- `CODE_QUALITY_STATUS.md`
- `RUFF_VERSION_MANAGEMENT.md`
- Related quality docs

### 2D. Consolidate Performance Docs (4 files → 1 file)
**Impact**: 3 file reduction

**Create**: `docs/development/PERFORMANCE.md`

**Consolidate:**
- `BENCHMARKING_GUIDE.md`
- `ACCELERATION_DECISION_PROCEDURE_2025-10-09.md`
- `[COMPLETED]_ACCELERATION_DECISION_2025-10-09.md` (if not archived)
- Performance-related docs

### 2E. Move Subdirectory-Worthy Content (20 files)
**Impact**: Organizational improvement

**Move to appropriate subdirectories:**
- Typing docs → `development/typing/` (already exists, 6 files)
- Analysis docs → `development/analysis/` (already exists, 26 files)
- Strategy docs → `development/strategy/` (already exists, 4 files)

---

## Phase 3: Development/Analysis Cleanup (26 files → 8 files)

### 3A. Consolidate by Theme (26 files → 6-8 files)

**Create thematic consolidations:**

1. **`architecture_analysis.md`** (5-6 files):
   - `geometry_system_design.md` (52 KB - keep or consolidate?)
   - `qp_collocation_performance_analysis.md`
   - Related architecture analyses

2. **`type_system_analysis.md`** (4-5 files):
   - MyPy-related analyses
   - Type checking analyses

3. **`performance_analysis.md`** (4-5 files):
   - Profiling analyses
   - Benchmark analyses

4. **`config_system_analysis.md`** (3-4 files):
   - Pydantic analyses
   - OmegaConf analyses

5. **`testing_analysis.md`** (3-4 files):
   - Coverage analyses
   - Integration test analyses

6. **`network_analysis.md`** (3-4 files):
   - Network MFG analyses

---

## Phase 4: Theory/ Consolidation (39 files → 25 files)

### 4A. Theory Subdirectory Analysis

**Current structure:**
```
theory/
├── foundations/         (6 files → 4 files)
├── numerical_methods/   (8 files → 5 files)
├── reinforcement_learning/ (13 files → 8 files)
├── stochastic/          (4 files → 3 files)
└── applications/        (8 files → 5 files)
```

### 4B. Consolidation Strategy

**Reinforcement Learning** (13 → 8 files):
- Consolidate similar continuous control docs
- Merge related policy gradient theory
- Keep: main overview + specialized methods

**Numerical Methods** (8 → 5 files):
- Consolidate AMR-related docs (3 AMR files exist)
- Merge similar method descriptions

**Foundations** (6 → 4 files):
- Consolidate related foundational theory

---

## Phase 5: User/ Consolidation (24 files → 15 files)

### 5A. User Directory Analysis

**Current issues:**
- Some tutorials could be consolidated
- Installation/setup guides may overlap
- README files proliferation

**Strategy:**
- Consolidate overlapping tutorials
- Merge installation guides
- Keep clear separation: quickstart, tutorials, guides

---

## Execution Timeline

### Week 1: Quick Wins (Target: -50 files)
- ✅ Archive 26 completed files
- ✅ Consolidate phase summaries (4 + 6 + 2 → 3 files)
- ✅ Archive old session summaries (15 → 3 files)
- **Result**: 244 → 194 files

### Week 2: Development/ Root (Target: -40 files)
- ✅ Consolidate paradigm overviews (5 → 1)
- ✅ Consolidate architecture docs (6 → 1)
- ✅ Consolidate code quality docs (5 → 1)
- ✅ Consolidate performance docs (4 → 1)
- ✅ Move/organize remaining root files
- **Result**: 194 → 154 files

### Week 3: Development/Analysis (Target: -18 files)
- ✅ Create 6 thematic analysis consolidations
- **Result**: 154 → 136 files

### Week 4: Theory & User (Target: -11 files)
- ✅ Theory consolidations
- ✅ User consolidations
- **Result**: 136 → 125 files ✅ **TARGET REACHED**

---

## Consolidation Commands

### Archive Completed Files
```bash
# Create archive structure
mkdir -p archive/development/phases
mkdir -p archive/development/sessions
mkdir -p archive/development/completed

# Archive all [COMPLETED] files
find docs/development -name "*COMPLETED*" -o -name "*CLOSED*" -o -name "*RESOLVED*" | \
  while read file; do
    git mv "$file" archive/development/completed/
  done

# Archive old session summaries (keep newest 3)
find docs/development -name "SESSION_*" | sort | head -n -3 | \
  while read file; do
    git mv "$file" archive/development/sessions/
  done
```

### Create Phase Consolidations
```bash
# Create consolidated PHASE_2_COMPLETE.md
# (Manual: merge content from 4 source files)

# Move originals to archive
git mv docs/development/completed/PHASE2_* archive/development/phases/
git mv docs/development/completed/*PHASE_2* archive/development/phases/
```

### Create Paradigm Consolidation
```bash
# Create COMPUTATIONAL_PARADIGMS.md
# (Manual: merge 5 paradigm overview files)

# Archive originals
git mv docs/development/*PARADIGM_OVERVIEW.md archive/development/
```

---

## Validation

After each phase, run:
```bash
python scripts/check_docs_structure.py --report
```

**Success Criteria:**
- development/: ≤ 35 files
- theory/: ≤ 25 files
- user/: ≤ 15 files
- reference/: ≤ 50 files
- Total: ≤ 125 files (within category limits)

---

## Priority Order

1. **IMMEDIATE** (Today): Phase 1 (archive completed, consolidate phases, archive sessions)
2. **HIGH** (This Week): Phase 2 (development/ root cleanup)
3. **MEDIUM** (Next Week): Phase 3 (analysis consolidation)
4. **LOW** (Final Week): Phases 4-5 (theory & user)

---

## Notes

- All archived content preserved in `archive/` - nothing deleted
- Consolidations create comprehensive guides with cross-references
- Original files referenced in consolidated docs
- Git history preserved for all moves

---

**Created**: 2025-10-11
**Status**: Ready to execute
**Estimated Time**: 4 weeks for full consolidation
**Quick Win Time**: 2-3 hours for Phase 1 (50 file reduction)
