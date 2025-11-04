# Array-Based Notation Migration Plan

**Version**: 1.0.0
**Date**: 2025-11-04
**Status**: Phase 1 Complete (Documentation), Phase 2 Pending (Implementation)

---

## Overview

Migration from dimension-specific scalar notation to dimension-agnostic array notation for all spatial quantities in MFG_PDE.

### Design Decision

**Always use arrays for spatial quantities, even in 1D.**

```python
# ✅ Target state (dimension-agnostic)
Nx = [100]              # 1D: 100 grid points
Nx = [100, 80]          # 2D: 100×80 grid
Nx = [100, 80, 60]      # 3D: 100×80×60 grid

# ❌ Current state (1D only, dimension-specific)
Nx = 100                # Scalar for 1D
```

**Rationale**:
1. Algorithms work for arbitrary dimensions without type checking
2. Consistent interface: `Nx` is always a list
3. Natural subscript notation: `Nx[i]`, `dx[i]`, `xmin[i]` for dimension `i`
4. Eliminates special-case code for 1D vs nD

---

## Phase 1: Documentation ✅ COMPLETED (2025-11-04)

### Completed Work

1. **Created canonical standard**: `docs/development/MATHEMATICAL_NOTATION_STANDARD.md`
   - Principle 1: Always use arrays, even for 1D
   - Comprehensive examples showing correct array usage
   - "Wrong" examples showing deprecated scalar usage
   - Updated deprecation table

2. **Updated `mathematical_notation.py`**: Already uses `dimension` and `Nx` naming

3. **Established target state**: Documentation specifies array-based API

### Documentation Changes

**Updated entries in deprecation table**:
- `Nx` (scalar for 1D) → `Nx = [100]` - Always use array, even for 1D
- `xmin`, `xmax` (scalars) → `xmin = [-2.0]` - Always use array, even for 1D
- `dx` (scalar for 1D) → `dx = [0.01]` - Always use array, even for 1D

---

## Phase 2: Implementation 🔄 PENDING

### Migration Scope

Analysis shows extensive usage of old scalar notation:

**Core Infrastructure** (66 files):
- `mfg_pde/core/mfg_problem.py` - Core problem class
- `mfg_pde/alg/numerical/hjb_solvers/hjb_*.py` - HJB solvers (4 files)
- `mfg_pde/alg/numerical/fp_solvers/fp_*.py` - FP solvers (2 files)
- `mfg_pde/factory/solver_factory.py` - Factory functions

**Examples** (17 files):
- `examples/basic/*.py` - Basic examples
- `examples/advanced/*.py` - Advanced examples
- `examples/notebooks/*.ipynb` - Jupyter notebooks

**Tests** (32 files):
- `tests/unit/*.py` - Unit tests
- `tests/integration/*.py` - Integration tests

**Total**: ~115 files need migration

### Implementation Strategy

#### 2.1 Backward Compatibility Layer

Add property aliases in `MFGProblem` to support both APIs:

```python
class MFGProblem:
    def __init__(self, Nx=None, xmin=None, xmax=None, ...):
        # Normalize to arrays internally
        self._Nx = self._normalize_to_array(Nx)
        self._xmin = self._normalize_to_array(xmin)
        self._xmax = self._normalize_to_array(xmax)

    @staticmethod
    def _normalize_to_array(value):
        """Convert scalar or array to array."""
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return [value]
        return list(value)

    @property
    def Nx(self):
        """Grid points per dimension (array)."""
        return self._Nx

    @property
    def xmin(self):
        """Lower bounds per dimension (array)."""
        return self._xmin

    @property
    def xmax(self):
        """Upper bounds per dimension (array)."""
        return self._xmax

    @property
    def dx(self):
        """Spacing per dimension (array)."""
        return [(self.xmax[i] - self.xmin[i]) / self.Nx[i]
                for i in range(self.dimension)]
```

**Deprecation warnings**: Add warnings when scalar values are passed:

```python
import warnings

def _normalize_to_array(value, param_name="parameter"):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        warnings.warn(
            f"Passing scalar {param_name} is deprecated. "
            f"Use array notation [{param_name}] instead. "
            f"Scalar support will be removed in v1.0.0.",
            DeprecationWarning,
            stacklevel=3
        )
        return [value]
    return list(value)
```

#### 2.2 Internal Code Migration

**Priority order**:
1. Core infrastructure (`mfg_pde/core/`)
2. Solver implementations (`mfg_pde/alg/`)
3. Factory functions (`mfg_pde/factory/`)
4. Examples (`examples/`)
5. Tests (`tests/`)

**Pattern**: Replace scalar access with array access:

```python
# Before (dimension-specific)
dx = (problem.xmax - problem.xmin) / problem.Nx
U = np.zeros((problem.Nt + 1, problem.Nx + 1))

# After (dimension-agnostic)
dx = [(problem.xmax[i] - problem.xmin[i]) / problem.Nx[i]
      for i in range(problem.dimension)]
shape = (problem.Nt + 1,) + tuple(n + 1 for n in problem.Nx)
U = np.zeros(shape)
```

#### 2.3 Testing Strategy

1. **Unit tests**: Verify both scalar and array inputs work (with deprecation warning)
2. **Integration tests**: Ensure 1D, 2D, 3D problems work correctly
3. **Backward compatibility tests**: Verify old code still works with warnings
4. **Example verification**: Run all examples to ensure they work

#### 2.4 Documentation Updates

1. Update all docstrings to show array notation
2. Update user guides and tutorials
3. Update API reference documentation
4. Add migration guide for users

---

## Phase 3: Cleanup 🔮 FUTURE (v1.0.0)

**Breaking changes**:
1. Remove backward compatibility layer
2. Remove deprecation warnings
3. Require array notation exclusively
4. Update all remaining code

**Release**: v1.0.0 with clean, consistent API

---

## Current State Summary (Updated 2025-11-04)

### Key Finding: Array API Already Exists!

**Proof-of-concept testing reveals**:
- ✅ `spatial_discretization` and `spatial_bounds` are **already arrays** in current implementation
- ✅ Array-based API is **already available** via `spatial_bounds`/`spatial_discretization`
- ✅ Scalar API (`Nx`, `xmin`, `xmax`) coexists with array API
- ✅ Both approaches work correctly for 1D problems

**Example - Current State**:
```python
# Method 1: Scalar API (legacy, but functional)
problem = MFGProblem(Nx=100, xmin=-2.0, xmax=2.0, Nt=50)
# Creates: problem.Nx=100, problem.spatial_discretization=[100]

# Method 2: Array API (already available!)
problem = MFGProblem(
    spatial_bounds=[(-2.0, 2.0)],
    spatial_discretization=[100],
    Nt=50
)
# Creates: problem.Nx=100, problem.spatial_discretization=[100]
```

### ✅ Completed
- Documentation standard established (MATHEMATICAL_NOTATION_STANDARD.md)
- Deprecation notices added for scalar notation
- Target state defined with examples
- Proof-of-concept validation completed

### 🎯 Revised Understanding
- **Gap is smaller than initially thought**
- Users can already use array notation via `spatial_bounds`/`spatial_discretization`
- `Nx`, `xmin`, `xmax` parameters accept scalars but internally create array equivalents
- Migration is primarily about **promoting the array API**, not implementing it from scratch

### ⏳ Revised Phase 2 Strategy
Instead of large-scale refactoring, Phase 2 should focus on:

1. **Update parameter signature** to accept both:
   ```python
   Nx: int | list[int] | None = None
   xmin: float | list[float] | None = None
   xmax: float | list[float] | None = None
   ```

2. **Normalize inputs** to route through `spatial_bounds`/`spatial_discretization`

3. **Deprecation warnings** when scalar forms are used (soft migration)

4. **Documentation updates** to promote array API as preferred

5. **Example migration** showing both old and new styles

This is a **documentation and guidance effort**, not a complete rewrite.

---

## Migration Checklist

### Phase 1: Documentation ✅
- [x] Create `MATHEMATICAL_NOTATION_STANDARD.md`
- [x] Define array-based notation as standard
- [x] Add deprecation notices for scalar notation
- [x] Update examples in documentation

### Phase 2: Implementation (v0.10.0) ⏳
- [ ] Add `_normalize_to_array()` helper to `MFGProblem`
- [ ] Add property aliases for `Nx`, `xmin`, `xmax`, `dx`
- [ ] Add deprecation warnings for scalar inputs
- [ ] Migrate core infrastructure (10 files)
- [ ] Migrate solver implementations (6 files)
- [ ] Migrate factory functions (1 file)
- [ ] Update examples (17 files)
- [ ] Update tests (32 files)
- [ ] Add backward compatibility tests
- [ ] Write user migration guide

### Phase 3: Cleanup (v1.0.0) 🔮
- [ ] Remove backward compatibility layer
- [ ] Remove deprecation warnings
- [ ] Require array notation exclusively
- [ ] Release v1.0.0

---

## Technical Notes

### Current 1D Implementation

`MFGProblem` for 1D currently uses:
```python
self.Nx = Nx          # Scalar (e.g., 100)
self.xmin = xmin      # Scalar (e.g., -2.0)
self.xmax = xmax      # Scalar (e.g., 2.0)
```

### Current nD Implementation

`HighDimMFGProblem` uses:
```python
self.spatial_discretization = [Nx1, Nx2, ...]  # Already array-based
```

### Target Unified Implementation

Both 1D and nD will use:
```python
self.Nx = [Nx1, Nx2, ...]     # Always array
self.xmin = [x1min, x2min, ...]  # Always array
self.xmax = [x1max, x2max, ...]  # Always array
```

---

## Related Documents

- `docs/development/MATHEMATICAL_NOTATION_STANDARD.md` - Canonical notation reference
- `docs/development/CONSISTENCY_GUIDE.md` - General coding standards
- `docs/migration/PHASE_3_MIGRATION_GUIDE.md` - Phase 3 migration guide (to be updated)

---

**Last Updated**: 2025-11-04
**Next Review**: When Phase 2 implementation begins
**Owner**: MFG_PDE Core Team
