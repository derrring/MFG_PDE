# hasattr() Remaining Violations Analysis

**Issue**: #543 - Eliminate hasattr() duck typing
**Scope**: Core module remaining violations (mfg_components.py, mfg_problem.py, stochastic_problem.py)
**Count**: 17 violations (excluding 3 docstring examples in base_problem.py)
**Priority**: Medium (Step 2 complete, analyzing remaining work)

## Current State

**Total Violations**: 20 in core module
- **base_problem.py**: 3 (docstring examples - intentional, KEEP)
- **mfg_problem.py**: 4 (2 lazy init + 2 protocol duck typing)
- **mfg_components.py**: 11 (1 numpy + 10 self-attribute checks)
- **stochastic_problem.py**: 2 (legacy attribute fallback)

**Progress So Far**:
- ✅ Step 1: Fixed lazy init (hjb_geometry, fp_geometry, has_obstacles) → -2 violations
- ✅ Step 2: Fixed override system (Lx, Nx, dx, etc.) → -5 violations
- **Total**: 27 → 20 violations (26% reduction)

---

## Violation Categories

### 1. NumPy Scalar Duck Typing (1 case) ✅ **FIX**

**Location**: `mfg_components.py:748`

```python
m_val = m_val.item() if hasattr(m_val, "item") else float(m_val)
```

**Problem**: Duck typing to convert numpy scalars to Python floats

**Decision**: Replace with explicit type handling

**Solution**:
```python
# Use numpy's built-in scalar conversion
m_val = float(np.asarray(m_val))
```

**Risk**: Low - numpy handles all numeric types correctly
**Blocks**: None

---

### 2. Self Attribute Checks (10 cases) ✅ **FIX**

**Locations**: `mfg_components.py` lines 265, 365, 366, 429, 532, 533, 822, 857, 858, 861

#### Sub-category 2a: Geometry Availability (6 cases)

```python
# Lines 365, 532, 822, 857
if hasattr(self, "geometry") and self.geometry is not None:
    ...

# Line 429
if hasattr(self, "geometry") and hasattr(self.geometry, "dimension"):
    dimension = self.geometry.dimension

# Lines 858, 861
has_geometry_bc_method = has_geometry and hasattr(self.geometry, "get_boundary_conditions")
has_explicit_bc = ... and hasattr(self.geometry, "has_explicit_boundary_conditions")
```

**Problem**: Checking if `self` has `geometry` attribute instead of using explicit initialization

**Root Cause**: `MFGComponents` is a composition pattern class, not always initialized with geometry

**Decision**: Explicit initialization in `__init__`

**Solution**:
```python
# In MFGComponents.__init__ (need to verify location)
self.geometry = None  # type: GeometryProtocol | None

# Then replace all checks:
# Before
if hasattr(self, "geometry") and self.geometry is not None:

# After
if self.geometry is not None:

# For checking geometry methods (lines 429, 858, 861):
# Before
if hasattr(self.geometry, "dimension"):

# After - Option 1: Try/except (more Pythonic)
try:
    dimension = self.geometry.dimension
except AttributeError:
    dimension = 1  # default

# After - Option 2: Use Protocol with isinstance
from mfg_pde.geometry.protocol import GeometryProtocol
if isinstance(self.geometry, GeometryProtocol):
    ...
```

#### Sub-category 2b: Spatial Shape Checks (4 cases)

```python
# Lines 265, 366, 533
if hasattr(self, "spatial_shape") and len(self.spatial_shape) > 1:
    ...
```

**Problem**: Same pattern - checking for attribute existence

**Solution**:
```python
# In MFGComponents.__init__
self.spatial_shape = None  # type: tuple[int, ...] | None

# Then replace:
# Before
if hasattr(self, "spatial_shape") and len(self.spatial_shape) > 1:

# After
if self.spatial_shape is not None and len(self.spatial_shape) > 1:
```

**Risk**: Low - straightforward explicit initialization
**Blocks**: None

---

### 3. Lazy Initialization (2 cases) ✅ **FIX**

**Location**: `mfg_problem.py:1566, 1659`

```python
if not hasattr(self, "solver_compatible"):
    # Compatibility not yet detected (shouldn't happen if __init__ called)
    self._detect_solver_compatibility()
```

**Problem**: Defensive check assumes attribute might not exist

**Root Cause**: Unclear initialization order

**Decision**: Initialize explicitly in `__init__`, remove defensive check

**Solution**:
```python
# In MFGProblem.__init__ (already has many explicit inits from Step 1)
self.solver_compatible = {}
self.solver_recommendations = {}

# After init method calls:
self._detect_solver_compatibility()

# Then in require_solver_compatible() and get_solver_info():
# Before
if not hasattr(self, "solver_compatible"):
    self._detect_solver_compatibility()

# After - Remove entirely (already initialized and called in __init__)
# Just use: if solver_type not in self.solver_compatible: ...
```

**Risk**: Low - making implicit initialization explicit
**Blocks**: None

---

### 4. Protocol Duck Typing (2 cases) ⚠️ **EVALUATE**

**Location**: `mfg_problem.py:1990, 1993`

```python
if self.geometry is not None:
    if hasattr(self.geometry, "get_spatial_grid"):
        x = self.geometry.get_spatial_grid()
        collocation_points = np.atleast_2d(x).T if x.ndim == 1 else x
    elif hasattr(self.geometry, "interior_points"):
        collocation_points = self.geometry.interior_points
    else:
        # Fallback...
```

**Problem**: Checking which method the geometry provides (protocol-style duck typing)

**Root Cause**: Different geometry types have different interfaces

**Evaluation Needed**:

**Option 1: Keep as-is (acceptable duck typing)**
- This is legitimate protocol checking for different geometry types
- The alternative (defining Protocol with @overload) is heavy for 2 methods
- **Recommendation**: Document as intentional protocol check, add comment

**Option 2: Define GeometryProtocol properly**
- Create proper Protocol with method signatures
- Use `isinstance()` checks instead
- **Blocks**: Requires Issue #544 (geometry-first API) design decisions

**Option 3: Try/except pattern**
```python
try:
    x = self.geometry.get_spatial_grid()
    collocation_points = np.atleast_2d(x).T if x.ndim == 1 else x
except AttributeError:
    try:
        collocation_points = self.geometry.interior_points
    except AttributeError:
        # Fallback...
```

**Recommendation**: **Defer to Issue #544**
- This is architectural decision about geometry protocol
- Affects broader codebase beyond just hasattr elimination
- Document as known limitation for now

**Risk**: Medium (requires architectural design)
**Blocks**: Issue #544 (Geometry-First API)

---

### 5. Legacy Attribute Fallback (2 cases) ✅ **FIX**

**Location**: `stochastic_problem.py:232, 234`

```python
if self.conditional_terminal_cost is None:
    # Support both g (MFGProblem attribute) and terminal_cost (simplified API)
    if hasattr(self, "terminal_cost"):
        return self.terminal_cost(x)
    elif hasattr(self, "g"):
        return self.g(x)
    else:
        return 0.0
```

**Problem**: Supporting multiple attribute names for same concept

**Root Cause**: API evolution created multiple names (g vs terminal_cost)

**Decision**: Normalize in `__init__`, use explicit checks

**Solution**:
```python
# In StochasticProblem.__init__
# Normalize terminal cost naming
self.terminal_cost = getattr(self, 'terminal_cost', None) or getattr(self, 'g', None)

# Then simplify:
if self.conditional_terminal_cost is None:
    if self.terminal_cost is not None:
        return self.terminal_cost(x)
    else:
        return 0.0
```

**Risk**: Low - straightforward normalization
**Blocks**: None

---

## Implementation Plan

### Step 3a: Fix MFGComponents Self-Checks (Low Risk) ✅

**File**: `mfg_pde/core/mfg_components.py`
**Lines**: 265, 365, 366, 429, 532, 533, 822, 857, 858, 861 (11 violations → 0)

**Changes**:
1. Find `MFGComponents.__init__` and add explicit initialization:
   ```python
   self.geometry = None  # type: GeometryProtocol | None
   self.spatial_shape = None  # type: tuple[int, ...] | None
   ```
2. Replace all `hasattr(self, "geometry")` with `self.geometry is not None`
3. Replace all `hasattr(self, "spatial_shape")` with `self.spatial_shape is not None`
4. For geometry method checks (lines 429, 858, 861), use try/except pattern
5. Fix numpy scalar (line 748): `float(np.asarray(m_val))`

**Estimated Impact**: -11 violations (55% of remaining)

---

### Step 3b: Fix Lazy Initialization (Low Risk) ✅

**File**: `mfg_pde/core/mfg_problem.py`
**Lines**: 1566, 1659 (2 violations → 0)

**Changes**:
1. Add to `__init__` (before init method calls):
   ```python
   self.solver_compatible = {}
   self.solver_recommendations = {}
   ```
2. Ensure `self._detect_solver_compatibility()` called in `__init__`
3. Remove defensive checks in `require_solver_compatible()` and `get_solver_info()`

**Estimated Impact**: -2 violations (10% of remaining)

---

### Step 3c: Fix Legacy Fallback (Low Risk) ✅

**File**: `mfg_pde/core/stochastic/stochastic_problem.py`
**Lines**: 232, 234 (2 violations → 0)

**Changes**:
1. Add to `__init__`:
   ```python
   # Normalize terminal cost attribute names
   self.terminal_cost = getattr(self, 'terminal_cost', None) or getattr(self, 'g', None)
   ```
2. Simplify `get_terminal_cost_value()` to use normalized attribute

**Estimated Impact**: -2 violations (10% of remaining)

---

### Step 3d: Evaluate Protocol Duck Typing (Medium Risk) ⏸️

**File**: `mfg_pde/core/mfg_problem.py`
**Lines**: 1990, 1993 (2 violations → ? depends on decision)

**Decision Required**: Choose Option 1, 2, or 3 (see Category 4)

**Recommendation**: **DEFER to Issue #544**
- Add comment documenting intentional protocol check
- Revisit when geometry protocol is properly defined
- Not blocking #543 completion

**Estimated Impact**: 0 violations fixed now, -2 violations after #544

---

## Summary

### Quick Wins (Step 3a + 3b + 3c)
- **Total**: 15 violations → 0 (75% of remaining)
- **Risk**: Low
- **Blocks**: None
- **Timeline**: Immediate (this session)

### Deferred (Step 3d)
- **Total**: 2 violations
- **Risk**: Medium (architectural)
- **Blocks**: Issue #544
- **Timeline**: After geometry-first API completion

---

## Final State After Step 3 (a+b+c)

| Module | Current | After Step 3 | Notes |
|:-------|:--------|:-------------|:------|
| base_problem.py | 3 | 3 | Docstring examples (intentional) |
| mfg_problem.py | 4 | 2 | Fix lazy init, defer protocol duck typing |
| mfg_components.py | 11 | 0 | Fix all self-checks + numpy scalar |
| stochastic_problem.py | 2 | 0 | Fix legacy fallback |
| **Total** | **20** | **5** | **75% reduction** |

**Target**: 3 violations (only docstring examples)
**Achievable**: 5 violations (3 docstring + 2 protocol duck typing deferred to #544)

---

## Testing Strategy

### Unit Tests
- Run full test suite for modified files:
  ```bash
  pytest tests/unit/test_mfg_problem.py -v
  pytest tests/unit/test_mfg_components.py -v
  pytest tests/unit/test_stochastic_problem.py -v
  ```

### Integration Tests
- Run full core module tests:
  ```bash
  pytest tests/ -k "mfg_problem or components or stochastic" -v
  ```

### Validation
```bash
# Before Step 3
python scripts/check_fail_fast.py --path mfg_pde/core --all
# Should show: 20 violations

# After Step 3a+b+c
python scripts/check_fail_fast.py --path mfg_pde/core --all
# Should show: 5 violations (3 docstring + 2 protocol)
```

---

## Risk Assessment

| Step | Risk | Mitigation |
|:-----|:-----|:-----------|
| 3a (MFGComponents) | Low | Explicit init, standard pattern |
| 3b (Lazy init) | Low | Already done in mfg_problem.py for other attrs |
| 3c (Legacy fallback) | Low | Simple normalization, backward compatible |
| 3d (Protocol duck typing) | Medium | Defer to #544, document as intentional |

---

## Recommendations

**Immediate (This Session)**:
- ✅ Execute Step 3a (MFGComponents self-checks) → -11 violations
- ✅ Execute Step 3b (Lazy initialization) → -2 violations
- ✅ Execute Step 3c (Legacy fallback) → -2 violations
- **Total**: -15 violations (75% of remaining)

**Document (This Session)**:
- Add code comments for protocol duck typing (lines 1990, 1993)
- Note in Issue #543 that 2 violations deferred to #544

**Future (After #544)**:
- Define proper GeometryProtocol
- Replace protocol duck typing with isinstance() checks
- Achieve target: Only 3 hasattr in core (all in docstrings)

---

## Code Review Checklist

Before committing Step 3:
- [ ] All modified files have explicit attribute initialization
- [ ] No new hasattr() introduced
- [ ] All tests pass
- [ ] Type hints added where appropriate
- [ ] Code comments added for intentional protocol checks
- [ ] Issue #543 updated with progress

---

**Created**: 2026-01-11
**Author**: Issue #543 Step 3 Analysis
**Status**: Ready for Implementation
**Estimated Time**: 30-45 minutes for Steps 3a+b+c
