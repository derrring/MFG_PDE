# MFG_PDE Deletion Proposal
**Date**: 2025-10-28
**Purpose**: Identify old/useless files for removal (not archiving)
**Principle**: Delete content with no educational or recovery value

---

## Deletion Candidates

### Category 1: Completed Development Documentation ✅ DELETE

**Location**: `docs/archive/`

**Files** (84 files total, 15 marked for deletion):

#### Completed Session Summaries (DELETE - no long-term value)
```
[COMPLETED]_SESSION_2025_10_05_SUMMARY.md
[COMPLETED]_SESSION_2025_10_05_CONTINUED.md
[COMPLETED]_SESSION_SUMMARY_2025_10_06.md
[COMPLETED]_SESSION_2025_10_06_CONTINUED.md
[COMPLETED]_SESSION_2025-10-08_SUMMARY.md
[COMPLETED]_CONTINUATION_SESSION_2025_10_06.md
```

**Rationale**: Session logs from Oct 2025 are recent but offer no educational value. Git history provides recovery if needed.

#### Completed Minor Issues (DELETE - resolved, documented in git)
```
[COMPLETED]_BRANCH_CLEANUP_2025_10_05.md
[COMPLETED]_NOMENCLATURE_UPDATE_SUMMARY.md
[COMPLETED]_SCRIPTS_AND_HOOKS_AUDIT.md
[RESOLVED]_DEPRECATED_WRAPPER_FIX_SUMMARY.md
[RESOLVED]_TEST_SUITE_FIXES_PROGRESS.md
[RESOLVED]_MASS_CONSERVATION_TEST_INVESTIGATION.md
[RESOLVED]_MASS_CONSERVATION_FDM.md
[RESOLVED]_ISSUE_76_TEST_SUITE_RESOLUTION_SUMMARY.md
```

**Rationale**: Small bug fixes and cleanup tasks. Git commit messages contain necessary info.

#### Superseded Content (DELETE - replaced by newer docs)
```
SUPERSEDED_heterogeneous_agents_formulation.md
SUPERSEDED_multi_population_continuous_control.md
```

**Rationale**: Explicitly marked as superseded. No value in keeping.

### Category 2: Planning Documents (KEEP - still relevant)

**Files to KEEP** (strategic value):
```
FOLDER_CONSOLIDATION_PLAN.md  # Meta-organizational guidance
DOCUMENTATION_INDEX.md         # Index of docs (useful)
CONSOLIDATION_ACTION_PLAN.md   # Planning methodology
DOCUMENTATION_RECATEGORIZATION_PLAN.md  # Organizational reference
PHASE_3_PREPARATION_2025.md    # Future planning
```

**Rationale**: These document organizational decisions and planning strategies. Educational value for maintaining large projects.

### Category 3: Archived Examples (EVALUATE)

**Location**: `examples/archive/`

**Files** (21 Python files):

#### API Demos (3 files) - **KEEP**
```
api_demos/general_mfg_construction_demo.py
api_demos/new_api_core_objects_example.py
api_demos/new_api_hooks_example.py
```

**Rationale**: Show old API for comparison. Educational value for API evolution understanding.

#### Backend Demos (2 files) - **KEEP**
```
backend_demos/jax_numba_hybrid_performance.py
backend_demos/unified_backend_acceleration_demo.py
```

**Rationale**: Demonstrate advanced backend usage patterns. May be worth reviving if API stabilizes.

#### Crowd Dynamics (2 files) - **DELETE or MOVE**
```
crowd_dynamics/crowd_evacuation_1d_hybrid.py
crowd_dynamics/crowd_evacuation_2d_isotropic.py
```

**Rationale**: If superseded by `examples/advanced/anisotropic_crowd_dynamics_2d/`, delete. Otherwise, move to advanced.

#### Maze Demos (12 files) - **DELETE**
```
maze_demos/actor_critic_maze_demo.py
maze_demos/hybrid_maze_demo.py
maze_demos/improved_maze_generator.py
maze_demos/maze_config_examples.py
maze_demos/maze_principles_demo.py
maze_demos/mfg_maze_environment_demo.py
maze_demos/mfg_maze_environment.py
maze_demos/mfg_maze_layouts.py
maze_demos/perfect_maze_generator.py
maze_demos/recursive_division_demo.py
maze_demos/visualize_recursive_division_simple.py
maze_demos/voronoi_maze_demo.py
```

**Rationale**: **Maze generation is NOT part of MFG_PDE core functionality**. These are application-specific examples that don't demonstrate package features. Research equivalent exists in `mfg-research/experiments/maze_navigation/`.

**Decision**: DELETE maze_demos/ entirely (12 files, ~164 KB)

### Category 4: Obsolete Top-Level Files (CHECK)

**Command**: `ls -la /Users/zvezda/Library/CloudStorage/OneDrive-Personal/code/MFG_PDE/ | grep -v "^d" | head -20`

**Expected Candidates**:
- Old README files (CHANGELOG.md, MIGRATION_GUIDE.md if outdated)
- Temporary scripts (test_*.py in root)
- Old configuration files (.pylintrc, .flake8 if ruff is used)

---

## Deletion Summary

### Proposed Deletions

| Category | Location | Files | Size | Rationale |
|:---------|:---------|:------|:-----|:----------|
| **Session Logs** | docs/archive/ | 6 files | ~24 KB | No long-term value |
| **Minor Issues** | docs/archive/ | 8 files | ~32 KB | Resolved, git has info |
| **Superseded** | docs/archive/ | 2 files | ~8 KB | Explicitly replaced |
| **Maze Demos** | examples/archive/ | 12 files | ~164 KB | Not MFG_PDE functionality |
| **Crowd Dynamics** | examples/archive/ | 2 files | ~25 KB | (If superseded) |
| **TOTAL** | - | **30 files** | **~253 KB** | - |

### Files to Keep

| Category | Files | Rationale |
|:---------|:------|:----------|
| **Planning Docs** | 5 files | Organizational guidance |
| **API Demos** | 3 files | Educational (API evolution) |
| **Backend Demos** | 2 files | Advanced usage patterns |
| **Implementation Summaries** | ~40 files | Major feature documentation |
| **TOTAL** | **~50 files** | Educational or strategic value |

---

## Execution Plan

### Phase 1: Safe Deletions (Low Risk)

**Delete explicitly superseded content:**
```bash
cd /Users/zvezda/Library/CloudStorage/OneDrive-Personal/code/MFG_PDE

# Superseded content (no value)
rm docs/archive/SUPERSEDED_heterogeneous_agents_formulation.md
rm docs/archive/SUPERSEDED_multi_population_continuous_control.md

# Recent session logs (git history sufficient)
rm docs/archive/[COMPLETED]_SESSION_*.md
rm docs/archive/[COMPLETED]_CONTINUATION_SESSION_*.md
```

**Estimated Savings**: ~32 KB, 8 files

### Phase 2: Moderate Risk Deletions

**Delete minor issue resolutions:**
```bash
# Small bug fixes (documented in git)
rm docs/archive/[COMPLETED]_BRANCH_CLEANUP_*.md
rm docs/archive/[COMPLETED]_NOMENCLATURE_UPDATE_SUMMARY.md
rm docs/archive/[COMPLETED]_SCRIPTS_AND_HOOKS_AUDIT.md
rm docs/archive/[RESOLVED]_DEPRECATED_WRAPPER_FIX_SUMMARY.md
rm docs/archive/[RESOLVED]_TEST_SUITE_FIXES_PROGRESS.md
rm docs/archive/[RESOLVED]_MASS_CONSERVATION_*.md
rm docs/archive/[RESOLVED]_ISSUE_76_TEST_SUITE_RESOLUTION_SUMMARY.md
```

**Estimated Savings**: ~32 KB, 8 files

### Phase 3: High Value Deletions

**Delete maze demos (not MFG_PDE functionality):**
```bash
# Maze generation (application-specific, not package feature)
rm -rf examples/archive/maze_demos/
```

**Estimated Savings**: ~164 KB, 12 files

**Conditional: Delete crowd dynamics if superseded:**
```bash
# Check if advanced examples cover this functionality
# If yes, delete:
rm -rf examples/archive/crowd_dynamics/
```

**Estimated Savings**: ~25 KB, 2 files

---

## Validation Before Deletion

### Checklist

**For each file marked for deletion:**
- [ ] Check git history: Is content duplicated in commits?
- [ ] Check references: Does any active doc link to this file?
- [ ] Check educational value: Does this teach something unique?
- [ ] Check recovery cost: How hard to recreate if needed?

**Safe to delete if:**
- ✅ Content in git history
- ✅ No active cross-references
- ✅ No unique educational content
- ✅ Easy to recreate (or not needed)

---

## Post-Deletion Documentation Update

**Update affected files:**
1. `docs/archive/README.md` - Remove references to deleted files
2. `docs/README.md` - Update archive description
3. `examples/README.md` - Update archive section
4. `DOCUMENTATION_INDEX.md` - Remove deleted entries (if keeping this file)

---

## Comparison: Deletion vs Archiving

### When to DELETE:
- ✅ Recent session logs (git history sufficient)
- ✅ Minor bug fixes (documented in commits)
- ✅ Superseded content (explicitly replaced)
- ✅ Application-specific code (not package functionality)
- ✅ Temporary analysis files

### When to ARCHIVE:
- ✅ Major feature implementations (show evolution)
- ✅ Complex design decisions (why choices were made)
- ✅ API evolution examples (educational)
- ✅ Performance analyses (historical benchmarks)
- ✅ Strategic planning docs (organizational guidance)

---

## Recommendations

### Priority 1: Delete Superseded Content (IMMEDIATE) ⚠️

**Action**:
```bash
rm docs/archive/SUPERSEDED_*.md
```

**Risk**: None (explicitly marked as superseded)
**Benefit**: Removes confusion

### Priority 2: Delete Session Logs (RECOMMENDED) ⚠️

**Action**:
```bash
rm docs/archive/[COMPLETED]_SESSION_*.md
rm docs/archive/[COMPLETED]_CONTINUATION_SESSION_*.md
```

**Risk**: Low (git history provides recovery)
**Benefit**: 8 files, ~32 KB freed

### Priority 3: Delete Maze Demos (HIGH VALUE) ⚠️⚠️

**Action**:
```bash
rm -rf examples/archive/maze_demos/
```

**Risk**: Moderate (but research version exists in mfg-research)
**Benefit**: 12 files, ~164 KB freed
**Justification**: Maze generation is application research, not MFG_PDE core functionality

### Priority 4: Evaluate Remaining Archives (ONGOING)

**Action**: Review other [COMPLETED] and [RESOLVED] docs for deletion candidates

**Method**: Apply deletion criteria to each file individually

---

## Success Metrics

**File Count**:
- Before: ~269 docs + 69 examples
- After: ~239 docs + 57 examples (DELETE 30 files)

**Archive Size**:
- Before: ~5 subdirectories in examples/archive/
- After: ~3 subdirectories (remove maze_demos/, possibly crowd_dynamics/)

**Clarity**:
- ✅ No superseded content
- ✅ No recent session logs (git history sufficient)
- ✅ Archives contain only educational/strategic content

---

**Created**: 2025-10-28 02:20
**Status**: Proposal - awaiting user approval
**Estimated Savings**: 30 files, ~253 KB
