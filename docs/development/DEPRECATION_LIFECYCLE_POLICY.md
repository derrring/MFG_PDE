# Deprecation Lifecycle Policy

**Created**: 2026-01-20
**Status**: Active
**Motivation**: Issue #616 - `conservative=` parameter lingered in broken state for 1 month

---

## Executive Summary

**Core Principle**: **Deprecated code must immediately redirect to correct behavior**

Deprecation is NOT:
- ❌ Warning users while keeping broken code active
- ❌ Marking parameters as "to be removed" without fixing them
- ❌ Documenting that something is wrong without preventing usage

Deprecation IS:
- ✅ Immediate redirection to new standard
- ✅ Zero behavioral difference between old and new API
- ✅ Warning that alerts users to update their code
- ✅ Clear removal timeline

---

## The Problem: conservative= Parameter (Issue #616)

### What Went Wrong

**Timeline**:
```
2025-12-07: Issue #382 - Changed default conservative=False → True (correct)
2025-12-16: Issue #490 - Refactored to advection_scheme parameter
            - Kept conservative with deprecation warning
            - BUT set wrong default: advection_scheme="gradient_upwind" (BROKEN)
            - BUT factory not updated: still used "gradient_upwind" (BROKEN)
            - Promised removal in "v1.0.0" (unknowable timeline)
2026-01-17: Issue #580 - Exposed bug by changing Auto Mode to FDM+FDM
2026-01-20: Issue #615 - Finally removed conservative and fixed default
```

**Impact**: 1 month (34 days) with **catastrophic mass conservation failure** (99.4% error) in production code.

**Lesson**: `remove_by="v1.0.0"` was a false promise - we don't actually know when removal is safe.

### Root Cause

1. **Deprecation without redirection**: Added warning but didn't ensure new default matched old behavior
2. **Partial migration**: Updated parameter but not factory code
3. **No validation**: No test verified that deprecated path gave same result as new path

---

## Strict Deprecation Lifecycle

### Phase 1: Deprecation Declaration (Version N)

**Requirements** (ALL must be satisfied):

1. **Immediate redirection**:
   ```python
   # OLD API
   def my_function(old_param=None, new_param="default"):
       if old_param is not None:
           warnings.warn(
               f"Parameter 'old_param' is deprecated. Use 'new_param' instead. "
               f"Will be removed in v{N+3}.0.",
               DeprecationWarning,
               stacklevel=2,
           )
           new_param = convert_old_to_new(old_param)  # ✅ REDIRECT

       # Continue with new_param only
       return implementation(new_param)
   ```

2. **Behavioral equivalence test**:
   ```python
   # tests/unit/test_deprecation_equivalence.py
   def test_old_param_gives_same_result_as_new():
       # Old API
       result_old = my_function(old_param=value)

       # New API
       result_new = my_function(new_param=converted_value)

       # MUST be identical
       assert np.allclose(result_old, result_new)
   ```

3. **Update ALL call sites**:
   - Direct function calls ✅
   - Factory functions ✅
   - Default parameters ✅
   - Internal usages ✅
   - Example code ✅

4. **Documentation updates**:
   - Deprecation guide entry
   - Docstring with deprecation notice
   - CHANGELOG.md entry
   - Migration path examples

**Acceptance Criteria**:
- ✅ Old API warns but works identically to new API
- ✅ All tests pass with deprecation warnings enabled
- ✅ No production code uses deprecated API
- ✅ Equivalence test added

### Phase 2: Deprecation Active (Versions N+1, N+2)

**During this phase**:
- Old API continues to work (with warnings)
- New API is preferred
- Both APIs maintained in sync
- No breaking changes to either

**Forbidden**:
- ❌ Changing behavior of deprecated API
- ❌ Removing redirection logic
- ❌ Silencing deprecation warnings

### Phase 3: Removal (Version N+3)

**Requirements**:

1. **Minimum deprecation period**: 3 minor versions OR 6 months (whichever is longer)
2. **Clear migration path**: All users have time to migrate
3. **Removal checklist**:
   - Remove deprecated parameter
   - Remove redirection logic
   - Remove equivalence test
   - Update deprecation guide
   - Add CHANGELOG entry

**Removal Pattern**:
```python
def my_function(new_param="default", old_param=None):
    if old_param is not None:
        raise TypeError(
            f"Parameter 'old_param' was removed in v{N+3}.0. "
            f"Use 'new_param' instead. "
            f"See docs/user/DEPRECATION_MODERNIZATION_GUIDE.md for migration."
        )
    return implementation(new_param)
```

Keep error message for 1 more version (N+4), then remove parameter entirely.

---

## Deprecation Patterns

### Pattern 1: Parameter Rename

**Scenario**: Rename `conservative` → `advection_scheme`

```python
def FPFDMSolver(
    problem,
    advection_scheme: str = "divergence_upwind",
    # Deprecated parameter
    conservative: bool | None = None,
):
    # Immediate redirection
    if conservative is not None:
        warnings.warn(
            "Parameter 'conservative' is deprecated. Use 'advection_scheme' instead. "
            "conservative=True → advection_scheme='divergence_upwind', "
            "conservative=False → advection_scheme='gradient_upwind'. "
            "Will be removed in v1.0.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        # ✅ REDIRECT: Ensure old behavior is preserved
        advection_scheme = "divergence_upwind" if conservative else "gradient_upwind"

    # Continue with new parameter only
    self.advection_scheme = advection_scheme
```

**Critical**: If `conservative=True` was the correct default, ensure `advection_scheme="divergence_upwind"` is also the default!

### Pattern 2: Function Rename

**Scenario**: Rename `create_default_monitor()` → `create_distribution_monitor()`

```python
def create_distribution_monitor(**kwargs):
    """New function with clear name."""
    return DistributionConvergenceMonitor(**kwargs)

def create_default_monitor(**kwargs):
    """Deprecated: Use create_distribution_monitor() instead."""
    warnings.warn(
        "create_default_monitor() is deprecated. "
        "Use create_distribution_monitor() instead. "
        "Will be removed in v1.0.0.",
        DeprecationWarning,
        stacklevel=2,
    )
    # ✅ REDIRECT: Simply call new function
    return create_distribution_monitor(**kwargs)
```

### Pattern 3: Class/Module Rename

**Scenario**: Rename `StochasticConvergenceMonitor` → `RollingConvergenceMonitor`

```python
# New class
class RollingConvergenceMonitor:
    """Monitor convergence using rolling statistics."""
    pass

# Deprecated alias
class StochasticConvergenceMonitor(RollingConvergenceMonitor):
    """
    Deprecated: Use RollingConvergenceMonitor instead.

    This class will be removed in v1.0.0.
    """
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "StochasticConvergenceMonitor is deprecated. "
            "Use RollingConvergenceMonitor instead. "
            "Will be removed in v1.0.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
```

### Pattern 4: Default Value Change

**Scenario**: Change default `use_upwind=False` → `use_upwind=True`

```python
# WRONG ❌
def solver(use_upwind: bool = True):  # Changed default
    pass
# Problem: Silent breaking change for users relying on default

# RIGHT ✅
def solver(use_upwind: bool | None = None):  # Make explicit
    if use_upwind is None:
        # Issue deprecation warning for relying on default
        warnings.warn(
            "Default value for 'use_upwind' will change from False to True "
            "in v1.0.0. Explicitly pass use_upwind=True or use_upwind=False "
            "to silence this warning.",
            FutureWarning,
            stacklevel=2,
        )
        use_upwind = False  # Old default (for now)
    pass
```

**Version N+1**: Change default and remove warning logic.

---

## Factory Function Synchronization

**Problem**: When deprecating a parameter, ALL call sites must be updated.

### Checklist for Parameter Deprecation

When deprecating a parameter (e.g., `conservative`):

- [ ] Update function signature with deprecation warning
- [ ] Add redirection logic (old → new)
- [ ] Update function default value to match old default behavior
- [ ] Search codebase for ALL usages:
  ```bash
  rg "conservative\s*=" --type py
  ```
- [ ] Update factory functions:
  ```python
  # mfg_pde/factory/scheme_factory.py
  def _create_fdm_pair(problem, scheme):
      if scheme == NumericalScheme.FDM_UPWIND:
          # ✅ MUST match deprecated default
          fp_config.setdefault("advection_scheme", "divergence_upwind")
  ```
- [ ] Update tests to use new API
- [ ] Add equivalence test
- [ ] Update examples
- [ ] Document in deprecation guide

### Enforcement

**Pre-commit hook** (`scripts/check_deprecation_sync.py`):
```python
"""Verify deprecated parameters are consistently handled."""

def check_factory_uses_new_api():
    """Ensure factory functions don't use deprecated parameters."""
    deprecated_params = load_deprecated_params()  # From config

    for factory_file in glob("mfg_pde/factory/**/*.py"):
        for param in deprecated_params:
            if param in read_file(factory_file):
                raise ValueError(
                    f"Factory {factory_file} uses deprecated parameter '{param}'. "
                    f"Update to use new API."
                )
```

---

## Testing Requirements

### 1. Equivalence Tests (Mandatory)

**Location**: `tests/unit/test_deprecation_equivalence.py`

```python
def test_conservative_param_equivalence():
    """Verify conservative=True gives same result as advection_scheme='divergence_upwind'."""
    problem = create_test_problem()

    # Old API
    solver_old = FPFDMSolver(problem, conservative=True)
    result_old = solver_old.solve()

    # New API
    solver_new = FPFDMSolver(problem, advection_scheme="divergence_upwind")
    result_new = solver_new.solve()

    # MUST be identical
    np.testing.assert_allclose(result_old, result_new, rtol=1e-15)
```

### 2. Warning Tests

```python
def test_conservative_param_emits_warning():
    """Verify deprecation warning is issued."""
    with pytest.warns(DeprecationWarning, match="conservative.*deprecated"):
        FPFDMSolver(problem, conservative=True)
```

### 3. Factory Consistency Tests

```python
def test_factory_gives_same_result_as_direct():
    """Verify factory uses correct default."""
    # Via factory
    result_factory = problem.solve(scheme=NumericalScheme.FDM_UPWIND)

    # Direct construction with correct scheme
    hjb = HJBFDMSolver(problem, advection_scheme="gradient_upwind")
    fp = FPFDMSolver(problem, advection_scheme="divergence_upwind")
    result_direct = problem.solve(hjb_solver=hjb, fp_solver=fp)

    # MUST match
    np.testing.assert_allclose(result_factory, result_direct, rtol=1e-12)
```

---

## Code Review Checklist

When reviewing a PR that deprecates code:

### Deprecation Declaration PR

- [ ] **Immediate redirection**: Old API calls new API internally
- [ ] **Equivalence test**: Verifies old and new give identical results
- [ ] **All call sites updated**: No production code uses old API
- [ ] **Factory synchronization**: Factory uses new default
- [ ] **Default value correct**: New default matches old behavior
- [ ] **Warning clear**: States old param, new param, removal version
- [ ] **Documentation updated**: DEPRECATION_MODERNIZATION_GUIDE.md entry
- [ ] **CHANGELOG entry**: Notes deprecation with migration path

### Removal PR

- [ ] **Minimum time passed**: 3 versions OR 6 months
- [ ] **Clear error message**: Guides users to migration docs
- [ ] **Tests updated**: Removed equivalence tests
- [ ] **Documentation updated**: Marked as removed in guide
- [ ] **CHANGELOG entry**: Notes removal

---

## Anti-Patterns

### ❌ Anti-Pattern 1: Warning Without Redirection

```python
# WRONG
def solver(conservative=False):
    if conservative:
        warnings.warn("conservative is deprecated")
    # Still uses conservative parameter directly
    if conservative:
        use_flux_form()  # ❌ Broken logic remains active
```

**Problem**: Warning fires but broken code still runs.

### ❌ Anti-Pattern 2: Partial Migration

```python
# Function updated
def FPFDMSolver(advection_scheme="divergence_upwind"):
    pass

# Factory NOT updated
def _create_fdm_pair():
    return FPFDMSolver()  # ❌ Uses new default, but should match old behavior
```

**Problem**: Direct calls work, factory calls broken.

### ❌ Anti-Pattern 3: Silent Default Change

```python
# v0.16: conservative=False (default)
# v0.17: advection_scheme="gradient_upwind" (default)
# ❌ These don't match!
```

**Problem**: Users relying on defaults get different behavior.

### ❌ Anti-Pattern 4: No Equivalence Test

**Problem**: No way to verify that redirection actually works.

---

## Lessons from conservative= Bug

### What We Should Have Done

1. **Issue #490 checklist**:
   - [x] Add `advection_scheme` parameter
   - [x] Add deprecation warning for `conservative`
   - [x] Add redirection logic
   - [ ] ❌ **MISSED**: Set `advection_scheme` default to match `conservative=True`
   - [ ] ❌ **MISSED**: Update factory to use new default
   - [ ] ❌ **MISSED**: Add equivalence test

2. **Correct default**:
   ```python
   # Should have been:
   advection_scheme: AdvectionScheme = "divergence_upwind"  # Matches conservative=True

   # Was actually:
   advection_scheme: AdvectionScheme = "gradient_upwind"  # WRONG
   ```

3. **Factory update**:
   ```python
   # Should have been updated in Issue #490:
   if scheme == NumericalScheme.FDM_UPWIND:
       fp_config.setdefault("advection_scheme", "divergence_upwind")

   # Was left as:
   fp_config.setdefault("advection_scheme", "gradient_upwind")  # WRONG
   ```

### How This Policy Would Have Prevented It

✅ **Equivalence test requirement** would have caught the wrong default immediately:
```python
def test_conservative_true_matches_default():
    solver_old = FPFDMSolver(problem, conservative=True)
    solver_new = FPFDMSolver(problem)  # Uses default
    # Would FAIL if default is wrong
    assert_same_results(solver_old, solver_new)
```

✅ **Factory synchronization checklist** would have caught factory not being updated

✅ **All call sites requirement** would have forced searching for ALL usages

---

## Enforcement Mechanisms

### 1. @deprecated Decorator (Code as Source of Truth)

**Module**: `mfg_pde/utils/deprecation.py`

```python
from mfg_pde.utils.deprecation import deprecated

@deprecated(
    since="v0.17.0",
    remove_by="v1.0.0",
    replacement="use advection_scheme parameter",
    reason="Confusing name, prefer explicit scheme selection"
)
def old_function():
    """Deprecated: Use new_function() instead."""
    return new_function()  # ✅ Redirects immediately
```

**Why decorator-based**:
- ✅ Metadata lives with code (no sync problems)
- ✅ Auto-discoverable by AST scanning
- ✅ Single source of truth
- ✅ Type-safe and inspectable

### 2. AST-Based Pre-commit Hook

**Script**: `scripts/check_internal_deprecation.py`

```python
# Auto-discovers @deprecated decorators and checks production code
# Uses Python's ast module (robust, no false positives from regex)

# Usage
python scripts/check_internal_deprecation.py

# Exit codes:
#   0 = No violations
#   1 = Production code uses deprecated symbols
#   2 = Script error
```

**Installation**:
```bash
# Install as git hook
ln -s ../../scripts/pre-commit-deprecation-check.sh .git/hooks/pre-commit

# Test manually
python scripts/check_internal_deprecation.py
```

**How it works**:
1. Phase 1: Scan `mfg_pde/` for `@deprecated` decorators → build registry
2. Phase 2: Scan production code for calls to deprecated symbols
3. Report violations with file:line locations

**Example output**:
```
🔍 Scanning for @deprecated decorators...
   Found @deprecated: create_default_monitor at convergence.py:45
   Total deprecated symbols: 1

🔍 Checking production code for deprecated symbol usage...
   mfg_pde/factory/solver_factory.py:123 calls deprecated 'create_default_monitor()'

❌ FAILURE: Production code calls deprecated functions.
```

### 3. CI Check

**GitHub Action**: `.github/workflows/deprecation-check.yml`
```yaml
name: Deprecation Policy Check
on: [pull_request]
jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Check deprecated symbol usage
        run: python scripts/check_internal_deprecation.py
```

---

## Summary

**Core Rules**:

1. **Immediate redirection**: Deprecated code MUST call new code internally
2. **Behavioral equivalence**: Old API must give identical results to new API
3. **Complete migration**: ALL call sites updated (direct, factory, examples, tests)
4. **Mandatory testing**: Equivalence test is required
5. **Clear timeline**: 3 versions OR 6 months minimum before removal

**Implementation** (2026-01-20):
- ✅ `mfg_pde/utils/deprecation.py` - @deprecated decorator (code as source of truth)
- ✅ `scripts/check_internal_deprecation.py` - AST-based checker (auto-discovers decorators)
- ✅ `scripts/pre-commit-deprecation-check.sh` - Git hook (blocks commits with violations)
- ✅ `.github/workflows/deprecation-check.yml` - CI enforcement

**Enforcement**:
- Pre-commit hook blocks commits with deprecated usage in production code
- CI fails PRs that violate policy
- AST scanning eliminates false positives (vs regex)
- Decorator metadata is source of truth (no JSON sync problems)

**Benefit**: Prevents bugs like Issue #616 where deprecated code lingered in broken state for 1 month, causing catastrophic failures (99.4% mass error) in production.

---

**Last Updated**: 2026-01-20
**Version**: 1.0
**Status**: ✅ Active Policy (Enforced)
