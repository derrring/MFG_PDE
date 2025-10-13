# CI/CD Workflow Duplication Analysis

**Date**: 2025-10-13
**Issue**: [#138](https://github.com/derrring/mfg-pde/issues/138) - CI/CD consolidation to reduce duplication
**Status**: Analysis Complete - Ready for Implementation

## Executive Summary

Current CI/CD setup runs **two parallel workflows** on PRs with significant overlap:
- **ci.yml**: Main CI/CD pipeline (quick-checks → import-validation → test-suite → performance-check)
- **modern_quality.yml**: Quality assurance workflow (modern-quality job with Ruff/MyPy)

**Key Finding**: Both workflows run independently, causing **duplicate execution** of:
1. Ruff formatting checks
2. Ruff linting checks
3. Security scans (bandit/safety)

**Impact**: ~20-30% waste in CI resources, slower feedback (both must complete independently)

---

## Detailed Workflow Analysis

### 1. ci.yml - CI/CD Pipeline

**Trigger**: PRs, pushes to main, releases, manual
**Primary Purpose**: Comprehensive testing and validation

**Job Structure**:
```
quick-checks (5min)
├─ Python syntax check
├─ Ruff format check         ← DUPLICATE with modern_quality.yml
└─ Ruff lint (critical only)  ← PARTIAL DUPLICATE

import-validation (5min)
├─ Package import test
└─ Documentation completeness

test-suite (40min)
├─ pytest with coverage
└─ Codecov upload

performance-check (15min)
├─ Memory validation
└─ Parameter migration check

security-scan (10min) [conditional]
├─ Bandit scan               ← DUPLICATE with modern_quality.yml
├─ Safety check              ← DUPLICATE with modern_quality.yml
└─ pip-audit

integration-tests (30min) [release only]
build-validation (15min) [release only]
ci-summary (always)
```

**Characteristics**:
- **Fail-fast design**: quick-checks must pass before heavier jobs run
- **Smart conditionals**: Performance and security checks have intelligent triggers
- **Job dependencies**: Clear dependency chain prevents wasted execution

---

### 2. modern_quality.yml - Unified Quality Assurance

**Trigger**: PRs, pushes to main, releases, manual
**Primary Purpose**: Code quality validation

**Job Structure**:
```
modern-quality (10min)
├─ Ruff format check         ← DUPLICATE with ci.yml
├─ Ruff lint (informational) ← DUPLICATE with ci.yml
└─ MyPy type checking

memory-safety (10min) [manual/release only]
performance-regression (10min) [manual/release only]
documentation-quality (5min) [manual/release only]

security-check (10min)
├─ Bandit scan               ← DUPLICATE with ci.yml
└─ Safety check              ← DUPLICATE with ci.yml

benchmark-performance (10min)
```

**Characteristics**:
- **No dependencies**: modern-quality runs independently
- **Informational focus**: Lint/type checks are non-blocking
- **Redundant with ci.yml**: No unique checks that aren't in ci.yml

---

### 3. security.yml - Security Scanning Pipeline

**Trigger**: Manual or releases only
**Primary Purpose**: Comprehensive security validation

**Job Structure**:
```
dependency-scanning
├─ Safety check
└─ pip-audit

static-code-analysis
├─ Bandit
└─ Semgrep

secrets-scanning
├─ detect-secrets
└─ TruffleHog

container-security
├─ Trivy
└─ Hadolint

license-compliance
security-summary
```

**Characteristics**:
- **Cost-optimized**: Manual trigger only (not on PRs)
- **No duplication**: Runs independently from PR workflows
- **Comprehensive**: More thorough than PR checks

---

### 4. check-ruff-updates.yml

**Trigger**: Monthly schedule or manual
**Primary Purpose**: Automated dependency management

**Characteristics**:
- **No duplication**: Maintenance workflow only
- **Updates ci.yml and modern_quality.yml**: Keeps ruff versions in sync

---

## Duplication Matrix

| Check | ci.yml | modern_quality.yml | security.yml | Notes |
|:------|:-------|:-------------------|:-------------|:------|
| **Ruff format** | ✅ quick-checks | ✅ modern-quality | ❌ | **DUPLICATE** - Same check, both blocking |
| **Ruff lint** | ✅ (critical only) | ✅ (full) | ❌ | **PARTIAL** - Different scope |
| **MyPy** | ❌ | ✅ (informational) | ❌ | **UNIQUE** to modern_quality.yml |
| **Bandit** | ✅ (conditional) | ✅ (always) | ✅ (manual) | **DUPLICATE** - ci.yml and modern_quality.yml |
| **Safety** | ✅ (conditional) | ✅ (always) | ✅ (manual) | **DUPLICATE** - ci.yml and modern_quality.yml |
| **pytest** | ✅ test-suite | ❌ | ❌ | **UNIQUE** to ci.yml |
| **Performance** | ✅ performance-check | ✅ benchmark | ❌ | **DUPLICATE** - Different implementations |

---

## Optimization Opportunities

### Priority 1: Eliminate Ruff Duplication (HIGH IMPACT)

**Current State**:
```yaml
# ci.yml quick-checks job
- name: Ruff format check
  run: ruff format --check mfg_pde/

# modern_quality.yml modern-quality job
- name: Ruff Formatting Check
  run: ruff format --check --diff mfg_pde/
```

**Recommendation**:
- **Keep in ci.yml quick-checks** (fail-fast validation)
- **Remove from modern_quality.yml** (redundant)
- **Benefit**: ~2-3 minutes saved per PR, instant feedback

### Priority 2: Consolidate Security Checks (MEDIUM IMPACT)

**Current State**:
- ci.yml: Runs bandit/safety conditionally (releases + manual trigger)
- modern_quality.yml: Runs bandit/safety always (on every PR)
- security.yml: Runs comprehensive scans (manual only)

**Recommendation**:
- **Remove security-check job from modern_quality.yml** entirely
- **Keep in ci.yml security-scan** (better conditional logic)
- **Keep security.yml** for comprehensive manual scans
- **Benefit**: ~5 minutes saved per PR

### Priority 3: Add Workflow Dependencies (MEDIUM IMPACT)

**Current State**: ci.yml and modern_quality.yml run completely independently

**Recommendation**: Use `workflow_run` to create dependency:
```yaml
# modern_quality.yml
on:
  workflow_run:
    workflows: ["CI/CD Pipeline"]
    types: [completed]
    branches: [main]
```

**Benefit**:
- modern_quality.yml only runs if ci.yml passes
- Prevents wasted compute on broken PRs
- Reduces parallel load on GitHub Actions runners

### Priority 4: Preserve MyPy in modern_quality.yml (KEEP AS-IS)

**Rationale**:
- MyPy check is **unique** to modern_quality.yml
- Informational output is valuable for type safety tracking
- Low cost (2-3 minutes), high value
- **Decision**: Keep this check, it's not duplication

---

## Implementation Plan (Issue #138 Phases)

### Phase 1: Add Job Dependencies (Safe, Reversible)
**Goal**: Make modern_quality.yml depend on ci.yml success

**Changes**:
```yaml
# modern_quality.yml
on:
  workflow_run:
    workflows: ["CI/CD Pipeline"]
    types: [completed]
    branches: [main]
  pull_request:  # Keep for manual triggers
    paths: ['mfg_pde/**/*.py', 'pyproject.toml', '.github/workflows/*.yml']
```

**Testing Strategy**:
1. Create feature branch `chore/ci-consolidation-phase1`
2. Open test PR
3. Verify modern_quality.yml waits for ci.yml
4. Measure total CI time

**Expected Outcome**: No time savings yet, but safer execution order

---

### Phase 2: Remove Duplicate Checks (Resource Savings)
**Goal**: Eliminate redundant Ruff and security checks

**Changes to modern_quality.yml**:
```yaml
# REMOVE these steps from modern-quality job:
# - Ruff Formatting Check (lines 47-56)
# - Ruff Linting (lines 58-65)

# REMOVE entire security-check job (lines 238-268)

# KEEP MyPy type checking (unique value)
# KEEP performance checks (different from ci.yml)
```

**Testing Strategy**:
1. Verify Ruff checks still run in ci.yml quick-checks
2. Verify security checks still run in ci.yml security-scan
3. Confirm MyPy still runs (no regression)

**Expected Outcome**:
- CI time: 40-45min → 35-40min (~10-15% faster)
- Cost savings: ~20-30% reduction in compute minutes

---

### Phase 3: Update Branch Protection (Enforcement)
**Goal**: Ensure only necessary checks are required

**Changes to GitHub branch protection**:
- **Remove**: `modern-quality / modern-quality` (will be optional)
- **Keep**: `quick-checks`, `test-suite`, `import-validation`
- **Add**: `modern-quality / Strategic type checking` (optional)

**Rationale**:
- Quick-checks already validates Ruff formatting
- Security checks run conditionally (not every PR)
- MyPy is informational (should not block merges)

---

## Workflow Comparison: Before vs After

### Before Optimization
```
PR opened → Triggers:
├─ ci.yml (runs immediately)
│  ├─ quick-checks: Ruff format + lint (3min)
│  ├─ import-validation (5min)
│  ├─ test-suite (35min)
│  └─ performance-check (10min)
│
└─ modern_quality.yml (runs in parallel)
   ├─ modern-quality: Ruff format + lint + MyPy (8min)
   └─ security-check: Bandit + Safety (5min)

Total wall time: max(ci.yml, modern_quality.yml) = ~45min
Total compute time: ci.yml + modern_quality.yml = ~66min
```

### After Optimization
```
PR opened → Triggers:
└─ ci.yml (runs immediately)
   ├─ quick-checks: Ruff format + lint (3min)
   ├─ import-validation (5min)
   ├─ test-suite (35min)
   └─ performance-check (10min)

ci.yml completes → Triggers:
└─ modern_quality.yml (runs only if ci.yml passes)
   ├─ modern-quality: MyPy only (3min)
   └─ benchmark-performance (5min)

Total wall time: ci.yml + modern_quality.yml = ~38-40min
Total compute time: ~53min (20% reduction)
```

**Key Improvements**:
- 📉 **Wall time**: 45min → 38-40min (~10-15% faster feedback)
- 💰 **Compute cost**: 66min → 53min (~20% cost savings)
- 🚦 **Fail-fast**: modern_quality.yml doesn't run if ci.yml fails
- 🎯 **Clearer purpose**: ci.yml = tests, modern_quality.yml = type checking

---

## Risk Assessment

### Low Risk Changes
✅ **Adding workflow dependencies**: Easily reversible, improves execution order
✅ **Removing Ruff duplication**: Already validated in ci.yml quick-checks
✅ **Updating branch protection**: Can be reverted in GitHub settings

### Medium Risk Changes
⚠️ **Removing security-check from modern_quality.yml**: Ensure ci.yml security-scan has proper triggers
⚠️ **Changing workflow triggers**: Test thoroughly on feature branch before merge

### Mitigation Strategy
1. **Feature branch testing**: Test all changes on `chore/ci-consolidation` branch
2. **Gradual rollout**: Implement Phase 1 → test → Phase 2 → test → Phase 3
3. **Monitoring**: Watch first 5 PRs after merge for any unexpected behavior
4. **Quick rollback**: Keep original workflow files in git history for easy revert

---

## Alternative Approaches Considered

### Option A: Merge into Single Workflow
**Pros**: Single source of truth, no duplication
**Cons**: Extremely long workflow file (500+ lines), harder to maintain
**Decision**: Rejected - complexity outweighs benefits

### Option B: Complete Separation (ci.yml = tests, modern_quality.yml = quality)
**Pros**: Clear separation of concerns
**Cons**: Still have duplication in quick-checks (Ruff needed for fail-fast)
**Decision**: Rejected - quick-checks must include Ruff for fail-fast validation

### Option C: Use Reusable Workflows
**Pros**: Shareable steps across workflows
**Cons**: Adds complexity, harder to debug, limited composability
**Decision**: Rejected - overkill for this use case

### Selected Approach: Workflow Dependencies + Targeted Removal
**Pros**:
- Preserves fail-fast validation (quick-checks with Ruff)
- Eliminates redundant execution (modern_quality.yml depends on ci.yml)
- Clear separation (ci.yml = validation, modern_quality.yml = type checking)
- Easy to understand and maintain

**Cons**:
- Sequential execution adds total wall time (mitigated by eliminating duplicate work)
- Two workflows instead of one (acceptable trade-off for clarity)

---

## Implementation Checklist

- [ ] Create feature branch `chore/ci-consolidation-phase1`
- [ ] Add workflow dependencies to modern_quality.yml
- [ ] Test Phase 1 with PR
- [ ] Merge Phase 1 to main
- [ ] Create feature branch `chore/ci-consolidation-phase2`
- [ ] Remove duplicate Ruff checks from modern_quality.yml
- [ ] Remove security-check job from modern_quality.yml
- [ ] Test Phase 2 with PR
- [ ] Measure CI time improvements
- [ ] Merge Phase 2 to main
- [ ] Update branch protection rules (Phase 3)
- [ ] Monitor 5 PRs for any issues
- [ ] Document final state in this file

---

## References

- **Issue**: [#138](https://github.com/derrring/mfg-pde/issues/138) - CI/CD consolidation
- **Workflow files**:
  - `.github/workflows/ci.yml` - Main CI/CD pipeline
  - `.github/workflows/modern_quality.yml` - Quality assurance
  - `.github/workflows/security.yml` - Security scanning (manual)
  - `.github/workflows/check-ruff-updates.yml` - Dependency management

---

**Analysis Complete**: Ready to proceed with Phase 1 implementation
