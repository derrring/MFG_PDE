# Broken Internal Links Report

**Generated**: 2025-11-16
**Total Broken Links**: 131
**Files Affected**: 28
**Links with Found Alternatives**: 10
**Links Not Found**: 121

---

## Critical Broken Links (Top 30)

### 1. Core Documentation Files

#### `/docs/README.md` (5 broken links)

**HIGH PRIORITY** - This is the main documentation index.

| Broken Link | Target | Status | Fix |
|:------------|:-------|:-------|:----|
| `development/CONSISTENCY_GUIDE.md` | `docs/development/CONSISTENCY_GUIDE.md` | MOVED | Change to `development/guides/CONSISTENCY_GUIDE.md` |
| `development/STRATEGIC_DEVELOPMENT_ROADMAP_2026.md` | `docs/development/STRATEGIC_DEVELOPMENT_ROADMAP_2026.md` | MOVED | Change to `development/planning/STRATEGIC_DEVELOPMENT_ROADMAP_2026.md` |
| `development/ARCHITECTURAL_CHANGES.md` | `docs/development/ARCHITECTURAL_CHANGES.md` | NOT FOUND | File does not exist - remove link or create file |
| `planning/completed/` | `docs/planning/completed` | NOT FOUND | Directory does not exist - remove link |
| `development/design/` | `docs/development/design` | NOT FOUND | Directory does not exist - remove link |

---

#### `/docs/development/README.md` (26 broken links)

**HIGH PRIORITY** - Main development documentation index.

| Broken Link | Target | Status | Fix |
|:------------|:-------|:-------|:----|
| `CONSISTENCY_GUIDE.md` | `docs/development/CONSISTENCY_GUIDE.md` | MOVED | Change to `guides/CONSISTENCY_GUIDE.md` |
| `STRATEGIC_DEVELOPMENT_ROADMAP_2026.md` | `docs/development/STRATEGIC_DEVELOPMENT_ROADMAP_2026.md` | MOVED | Change to `planning/STRATEGIC_DEVELOPMENT_ROADMAP_2026.md` |
| `ARCHITECTURAL_CHANGES.md` | `docs/development/ARCHITECTURAL_CHANGES.md` | NOT FOUND | File does not exist |
| `roadmaps/` | `docs/development/roadmaps` | NOT FOUND | Directory does not exist (moved to planning?) |
| `design/` | `docs/development/design` | NOT FOUND | Directory does not exist |
| `completed/` | `docs/development/completed` | NOT FOUND | Directory does not exist |
| `reports/` | `docs/development/reports` | NOT FOUND | Directory does not exist |
| `analysis/` | `docs/development/analysis` | NOT FOUND | Directory does not exist |
| `architecture/` | `docs/development/architecture` | NOT FOUND | Directory does not exist |
| `strategy/` | `docs/development/strategy` | NOT FOUND | Directory does not exist |
| `roadmaps/reinforcement_learning/` | `docs/development/roadmaps/reinforcement_learning` | NOT FOUND | Directory does not exist |
| `guides/features/` | `docs/development/guides/features` | NOT FOUND | Directory does not exist |
| `reports/CODEBASE_QUALITY_ASSESSMENT.md` | `docs/development/reports/CODEBASE_QUALITY_ASSESSMENT.md` | NOT FOUND | File does not exist |

**NOTE**: Most broken links in development/README.md reference non-existent directory structure. This file needs significant cleanup.

---

### 2. User Documentation

#### `/docs/user/README.md` (14 broken links)

**HIGH PRIORITY** - Main user documentation index.

| Broken Link | Target | Status | Fix |
|:------------|:-------|:-------|:----|
| `../development/adding_new_solvers.md` | `docs/development/adding_new_solvers.md` | NOT FOUND | File does not exist |
| `factory_api_reference.md` | `docs/user/factory_api_reference.md` | NOT FOUND | File does not exist |
| `custom_problems.md` | `docs/user/custom_problems.md` | NOT FOUND | File does not exist |
| `solver_comparison.md` | `docs/user/solver_comparison.md` | NOT FOUND | File does not exist |
| `configuration.md` | `docs/user/configuration.md` | NOT FOUND | File does not exist |
| `../examples/basic/` | `docs/examples/basic` | WRONG PATH | Should be `../../examples/basic/` (examples is peer to docs) |
| `../examples/advanced/` | `docs/examples/advanced` | WRONG PATH | Should be `../../examples/advanced/` |
| `../examples/notebooks/` | `docs/examples/notebooks` | WRONG PATH | Should be `../../examples/notebooks/` |
| `../development/CORE_API_REFERENCE.md` | `docs/development/CORE_API_REFERENCE.md` | NOT FOUND | File does not exist |
| `../development/infrastructure.md` | `docs/development/infrastructure.md` | NOT FOUND | File does not exist |
| `../development/factory_registration.md` | `docs/development/factory_registration.md` | NOT FOUND | File does not exist |

---

#### `/docs/user/quickstart.md` (7 broken links)

**HIGH PRIORITY** - First file users see.

| Broken Link | Target | Status | Fix |
|:------------|:-------|:-------|:----|
| `../examples/notebooks/` | `docs/examples/notebooks` | WRONG PATH | Should be `../../examples/notebooks/` |
| `factory_api_reference.md` | `docs/user/factory_api_reference.md` | NOT FOUND | File does not exist |
| `custom_problems.md` | `docs/user/custom_problems.md` | NOT FOUND | File does not exist |
| `../examples/basic/` | `docs/examples/basic` | WRONG PATH | Should be `../../examples/basic/` |
| `../examples/advanced/` | `docs/examples/advanced` | WRONG PATH | Should be `../../examples/advanced/` |

---

#### `/docs/user/core_objects.md` (3 broken links)

| Broken Link | Target | Status | Fix |
|:------------|:-------|:-------|:----|
| `../development/custom_solvers.md` | `docs/development/custom_solvers.md` | NOT FOUND | File does not exist |
| `performance.md` | `docs/user/performance.md` | NOT FOUND | File does not exist |
| `geometry.md` | `docs/user/geometry.md` | NOT FOUND | File does not exist |

---

### 3. Planning Documentation

#### `/docs/planning/README.md` (5 broken links)

| Broken Link | Target | Status | Fix |
|:------------|:-------|:-------|:----|
| `completed/` | `docs/planning/completed` | NOT FOUND | Directory does not exist |
| `governance/` | `docs/planning/governance` | NOT FOUND | Directory does not exist (actually at docs/development/governance) |
| `../development/STRATEGIC_DEVELOPMENT_ROADMAP_2026.md` | `docs/development/STRATEGIC_DEVELOPMENT_ROADMAP_2026.md` | MOVED | Change to `../development/planning/STRATEGIC_DEVELOPMENT_ROADMAP_2026.md` |

---

### 4. Theory Documentation

#### `/docs/theory/reinforcement_learning/README.md` (7 broken links)

| Broken Link | Target | Status | Fix |
|:------------|:-------|:-------|:----|
| `heterogeneous_agents_formulation.md` | `docs/theory/reinforcement_learning/heterogeneous_agents_formulation.md` | NOT FOUND | File does not exist |
| `multi_population_continuous_control.md` | `docs/theory/reinforcement_learning/multi_population_continuous_control.md` | NOT FOUND | File does not exist |
| `../foundations/NOTATION_STANDARDS.md` | `docs/theory/foundations/NOTATION_STANDARDS.md` | NOT FOUND | File does not exist |
| `../foundations/mfg_mathematical_background.md` | `docs/theory/foundations/mfg_mathematical_background.md` | NOT FOUND | File does not exist |

---

#### `/docs/user/stochastic_mfg_guide.md` (3 broken links)

| Broken Link | Target | Status | Fix |
|:------------|:-------|:-------|:----|
| `../theory/mean_field_games_mathematical_formulation.md` | `docs/theory/mean_field_games_mathematical_formulation.md` | NOT FOUND | File does not exist |
| `../theory/stochastic_mfg_common_noise.md` | `docs/theory/stochastic_mfg_common_noise.md` | MOVED | Change to `../theory/stochastic/stochastic_mfg_common_noise.md` |

---

### 5. Miscellaneous Critical Links

#### `/docs/user/collaboration/issue_templates.md` (2 broken links)

| Broken Link | Target | Status | Fix |
|:------------|:-------|:-------|:----|
| `../CLAUDE.md` | `docs/user/CLAUDE.md` | WRONG PATH | Should be `../../../CLAUDE.md` (CLAUDE.md is at repo root) |
| `SELF_GOVERNANCE_PROTOCOL.md` | `docs/user/collaboration/SELF_GOVERNANCE_PROTOCOL.md` | MOVED | Change to `../../development/governance/SELF_GOVERNANCE_PROTOCOL.md` |

---

#### `/docs/development/planning/optimization_integration_analysis.md` (3 broken links)

| Broken Link | Target | Status | Fix |
|:------------|:-------|:-------|:----|
| `../../CLAUDE.md` | `docs/CLAUDE.md` | WRONG PATH | Should be `../../../CLAUDE.md` |
| `../CONSOLIDATED_ROADMAP_2025.md` | `docs/development/CONSOLIDATED_ROADMAP_2025.md` | NOT FOUND | File does not exist |
| `../architecture/` | `docs/development/architecture` | NOT FOUND | Directory does not exist |

---

#### `/docs/development/typing/CI_CD_STRATEGIC_TYPING_EXPERIENCE_GUIDE.md` (3 broken links)

| Broken Link | Target | Status | Fix |
|:------------|:-------|:-------|:----|
| `./STRATEGIC_TYPING_METHODOLOGY.md` | `docs/development/typing/STRATEGIC_TYPING_METHODOLOGY.md` | NOT FOUND | File does not exist |
| `../DEVELOPMENT_GUIDE.md` | `docs/development/DEVELOPMENT_GUIDE.md` | NOT FOUND | File does not exist |
| `./CI_CD_OPTIMIZATION_PATTERNS.md` | `docs/development/typing/CI_CD_OPTIMIZATION_PATTERNS.md` | NOT FOUND | File does not exist |

---

## Categories of Issues

### A. Files Moved After v0.12.5 Geometry Reorganization

These files exist but have been reorganized:

1. **CONSISTENCY_GUIDE.md**: Moved from `docs/development/` to `docs/development/guides/`
2. **STRATEGIC_DEVELOPMENT_ROADMAP_2026.md**: Moved from `docs/development/` to `docs/development/planning/`
3. **SELF_GOVERNANCE_PROTOCOL.md**: Moved from `docs/user/collaboration/` to `docs/development/governance/`
4. **stochastic_mfg_common_noise.md**: Moved from `docs/theory/` to `docs/theory/stochastic/`

**Action**: Update all references to these files.

---

### B. Wrong Relative Paths to Examples Directory

Many user documentation files incorrectly reference examples as `../examples/` when it should be `../../examples/` (examples is a peer to docs, not a subdirectory).

**Affected files**:
- `/docs/user/README.md`
- `/docs/user/quickstart.md`
- `/docs/user/guides/phase2_features.md`

**Action**: Fix relative paths from `../examples/` to `../../examples/`.

---

### C. Wrong Relative Paths to CLAUDE.md

CLAUDE.md is at the repository root, but several files reference it incorrectly:

- `/docs/user/collaboration/issue_templates.md` → Should be `../../../CLAUDE.md`
- `/docs/development/planning/optimization_integration_analysis.md` → Should be `../../../CLAUDE.md`

**Action**: Fix relative paths to point to root.

---

### D. Non-Existent Documentation Files

These files are referenced but do not exist:

**Development Documentation** (should exist):
- `docs/development/adding_new_solvers.md`
- `docs/development/custom_solvers.md`
- `docs/development/CORE_API_REFERENCE.md`
- `docs/development/infrastructure.md`
- `docs/development/factory_registration.md`
- `docs/development/contributing.md`
- `docs/development/benchmarking_guide.md`
- `docs/development/DEVELOPMENT_GUIDE.md`
- `docs/development/ARCHITECTURAL_CHANGES.md`
- `docs/development/CONSOLIDATED_ROADMAP_2025.md`

**User Documentation** (should exist):
- `docs/user/factory_api_reference.md`
- `docs/user/custom_problems.md`
- `docs/user/solver_comparison.md`
- `docs/user/configuration.md`
- `docs/user/performance.md`
- `docs/user/geometry.md`
- `docs/user/continuous_control_guide.md`

**Theory Documentation**:
- `docs/theory/foundations/NOTATION_STANDARDS.md`
- `docs/theory/foundations/mfg_mathematical_background.md`
- `docs/theory/mean_field_games_mathematical_formulation.md`
- `docs/theory/reinforcement_learning/heterogeneous_agents_formulation.md`
- `docs/theory/reinforcement_learning/multi_population_continuous_control.md`

**Action**: Either create these files or remove references to them.

---

### E. Non-Existent Directory References

These directories are referenced but do not exist:

- `docs/development/roadmaps/` (content moved to `docs/planning/roadmaps/` or `docs/development/planning/`)
- `docs/development/design/`
- `docs/development/completed/`
- `docs/development/reports/`
- `docs/development/analysis/`
- `docs/development/architecture/`
- `docs/development/strategy/`
- `docs/development/guides/features/`
- `docs/planning/completed/`
- `docs/planning/governance/` (actually at `docs/development/governance/`)
- `docs/examples/` (examples is at repo root, not in docs)
- `docs/examples/hooks/`

**Action**: Update directory references or remove them.

---

### F. Mathematical Notation False Positives

Several files have "broken links" that are actually mathematical notation incorrectly parsed as markdown links:

**Files affected**:
- `docs/planning/roadmaps/MASTER_EQUATION_IMPLEMENTATION_PLAN.md` (16 false positives like `m(t,x)`)
- `docs/theory/foundations/MATHEMATICAL_NOTATIONS.md`
- `docs/theory/foundations/information_geometry_mfg.md`
- `docs/theory/semi_lagrangian_methods_for_hjb.md`
- Others in theory/ directory

**Example**: `m(t,x)` is parsed as link text "m" with link target "t,x"

**Action**: These are not actually broken links - improve the link detection script to ignore mathematical notation.

---

## Actual Directory Structure

For reference, here's what actually exists:

### docs/development/
- `guides/` (contains CONSISTENCY_GUIDE.md)
  - `code_quality/`
  - `tooling/`
- `planning/` (contains STRATEGIC_DEVELOPMENT_ROADMAP_2026.md)
- `governance/` (contains SELF_GOVERNANCE_PROTOCOL.md)
- `maintenance/`
- `paradigms/`
- `typing/`
- `decisions/`
- `status/`
- `private/`

### docs/planning/
- `roadmaps/`
- `reports/`

### docs/theory/
- `foundations/`
- `stochastic/` (contains stochastic_mfg_common_noise.md)
- `reinforcement_learning/`
- `applications/`
- `continuous_control/`
- `network_mfg/`

### Examples are at repo root:
- `examples/basic/`
- `examples/advanced/`
- `examples/applications/`
- `examples/tutorials/`
- `examples/notebooks/`

---

## Recommended Actions

### Immediate Fixes (High Priority)

1. **Fix moved file references** (10 instances):
   - Update all links to CONSISTENCY_GUIDE.md
   - Update all links to STRATEGIC_DEVELOPMENT_ROADMAP_2026.md
   - Update all links to SELF_GOVERNANCE_PROTOCOL.md
   - Update all links to stochastic_mfg_common_noise.md

2. **Fix examples path references** (~15 instances):
   - Change `../examples/` to `../../examples/` in user docs

3. **Fix CLAUDE.md path references** (2 instances):
   - Update to correct relative path from repo root

4. **Clean up docs/development/README.md**:
   - Remove references to non-existent directories
   - Update to reflect actual directory structure

5. **Clean up docs/README.md**:
   - Remove broken directory references
   - Update to current structure

### Medium Priority

6. **Create missing critical user documentation**:
   - `docs/user/factory_api_reference.md`
   - `docs/user/custom_problems.md`
   - `docs/user/configuration.md`

7. **Create missing development documentation**:
   - `docs/development/adding_new_solvers.md`
   - `docs/development/CORE_API_REFERENCE.md`

8. **Update planning documentation**:
   - Fix governance directory reference
   - Remove completed/ directory references

### Low Priority

9. **Improve link detection script**:
   - Filter out mathematical notation patterns
   - Reduce false positives

10. **Document actual directory structure**:
    - Create up-to-date documentation map
    - Ensure README files reflect reality

---

## Script Output Summary

```
Total broken links: 131
Files with broken links: 28
Links with found alternatives: 10
Links not found anywhere: 121
```

**Note**: The 121 "not found" includes ~40 mathematical notation false positives, so the real number of broken links is approximately 80-90.
