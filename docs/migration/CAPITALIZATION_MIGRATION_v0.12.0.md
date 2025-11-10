# Migration Guide: Dt → dt and Dx → dx (v0.12.0)

**Date**: 2025-11-10
**Issue**: #245
**Branch**: `chore/lowercase-dt-dx-capitalization`

## Summary

In v0.12.0, the primary time step and spatial spacing attributes have been renamed to follow official naming conventions:
- `Dt` → `dt` (time step Δt)
- `Dx` → `dx` (spatial spacing Δx for 1D problems)

This change affects `MFGProblem` and related protocol interfaces.

## Timeline

- **v0.11.0 and earlier**: Only uppercase `Dt` and `Dx` available
- **v0.12.0**: Both work, uppercase emits `DeprecationWarning`
- **v1.0.0**: Uppercase removed completely

## Quick Migration

### For Users

**Before** (v0.11.0 and earlier):
```python
problem = MFGProblem(xmin=0, xmax=1, Nx=50, T=1, Nt=50)

# Access attributes
dt = problem.Dt  # ✓ Works
dx = problem.Dx  # ✓ Works
```

**After** (v0.12.0 recommended):
```python
problem = MFGProblem(xmin=0, xmax=1, Nx=50, T=1, Nt=50)

# Access attributes (new lowercase)
dt = problem.dt  # ✓ Recommended
dx = problem.dx  # ✓ Recommended

# Old uppercase still works with warnings
dt_old = problem.Dt  # ⚠️ DeprecationWarning
dx_old = problem.Dx  # ⚠️ DeprecationWarning
```

### For Library Developers

Update any code accessing these attributes:

```python
# OLD
solver_dt = problem.Dt
grid_spacing = problem.Dx

# NEW
solver_dt = problem.dt
grid_spacing = problem.dx
```

## Rationale

This change aligns with the official naming conventions documented in `docs/NAMING_CONVENTIONS.md`:
- Line 24: `dt: Time step size Δt` (lowercase)
- Line 262: `dx` (lowercase in table)

Mathematical notation uses lowercase for differential operators (Δt, Δx), so code attributes should match.

## Backward Compatibility

**v0.12.0** provides full backward compatibility:

1. **Primary attributes**: New lowercase `dt` and `dx` are the primary attributes
2. **Deprecated properties**: Uppercase `Dt` and `Dx` remain available as properties
3. **Warnings**: Accessing uppercase properties emits `DeprecationWarning`
4. **Values identical**: `problem.dt == problem.Dt` and `problem.dx == problem.Dx`

## Migration Steps

### Step 1: Update Your Code

Search and replace in your codebase:
```bash
# Find all occurrences
grep -r "problem\.Dt" .
grep -r "problem\.Dx" .

# Replace (review each file manually)
# problem.Dt → problem.dt
# problem.Dx → problem.dx
```

### Step 2: Test with Warnings Enabled

```python
import warnings
warnings.simplefilter('always', DeprecationWarning)

# Run your code - any remaining uppercase usage will emit warnings
```

### Step 3: Verify

```python
from mfg_pde.core.mfg_problem import MFGProblem

problem = MFGProblem(xmin=0, xmax=1, Nx=50, T=1, Nt=50)

# These should work without warnings
assert problem.dt > 0
assert problem.dx > 0

# These should emit DeprecationWarning
import warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    _ = problem.Dt
    _ = problem.Dx
    assert len(w) == 2
    assert all(issubclass(warning.category, DeprecationWarning) for warning in w)
    print("✓ Migration verified")
```

## Affected Components

**Core** (2 files):
- `mfg_pde/core/mfg_problem.py`
- `mfg_pde/types/problem_protocols.py`

**Solvers** (9 files):
- HJB solvers: `base_hjb.py`, `hjb_semi_lagrangian.py`, `hjb_weno.py`
- FP solvers: `fp_particle.py`, `fp_fdm.py`
- Coupling: `fixed_point_iterator.py`, `hybrid_fp_particle_hjb_fdm.py`

**Utilities** (2 files):
- `experiment_manager.py`
- `hjb_policy_iteration.py`

**Tests, Examples, Benchmarks** (23 files):
- All test files updated (15 files)
- All example files updated (5 files)
- All benchmark files updated (3 files)

## Common Patterns

### Pattern 1: Direct Access
```python
# OLD
time_step = problem.Dt

# NEW
time_step = problem.dt
```

### Pattern 2: In Calculations
```python
# OLD
next_value = current_value + derivative * problem.Dt

# NEW
next_value = current_value + derivative * problem.dt
```

### Pattern 3: Function Arguments
```python
# OLD
def my_solver(dt: float, dx: float):
    ...

my_solver(problem.Dt, problem.Dx)

# NEW
def my_solver(dt: float, dx: float):
    ...

my_solver(problem.dt, problem.dx)
```

## Notes

### What Changed
- ✅ Attribute names: `Dt` → `dt`, `Dx` → `dx`
- ✅ Internal references throughout codebase
- ✅ Protocol definitions updated

### What Did NOT Change
- ❌ `xSpace` (remains capital S - intentional per conventions)
- ❌ `tSpace` (remains capital S - intentional per conventions)
- ❌ `Nx` (separate migration planned in Issue #243)
- ❌ Constructor parameters (no changes)

### Known Issues
None. The migration is complete and tested.

## Questions?

- **Why lowercase?** Follows mathematical notation (Δt, Δx) and official naming conventions
- **Why now?** Part of Issue #245 protocol compliance improvements
- **Breaking change?** No, fully backward compatible in v0.12.0
- **When removed?** Uppercase support removed in v1.0.0

## Related Documentation

- Official conventions: `docs/NAMING_CONVENTIONS.md`
- Changelog: `CHANGELOG.md` (Unreleased section)
- Implementation plan: `docs/development/CAPITALIZATION_MIGRATION_PLAN.md`
- Retrospective: `docs/development/SESSION_2025-11-10_RETROSPECTIVE.md`
