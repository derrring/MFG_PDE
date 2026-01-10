# hasattr() Analysis: mfg_pde/core Module

**Issue**: #543 - Eliminate hasattr() duck typing
**Scope**: Core module (base_problem.py, mfg_problem.py)
**Count**: 27 violations
**Priority**: High (core module is foundation)

## Violation Categories

### 1. Documentation Examples (3 cases) ✅ **KEEP**

**Location**: `base_problem.py:123, 141, 159`

```python
# Example in docstring showing how to handle scalar vs array inputs
def hamiltonian(self, x, m, p, t):
    p_arr = np.array(p) if hasattr(p, '__iter__') else p
    return 0.5 * np.sum(p_arr**2) + 0.1 * m
```

**Decision**: Keep as-is
**Reason**: These are teaching examples in docstrings, not production code
**Action**: None required

---

### 2. Lazy Initialization (2 cases) ⚠️ **FIX**

**Location**: `mfg_problem.py:428, 433`

```python
# Check if specialized init method already set attributes
if not hasattr(self, "hjb_geometry"):
    self.hjb_geometry = getattr(self, "geometry", None)
    self.fp_geometry = getattr(self, "geometry", None)

if not hasattr(self, "has_obstacles"):
    self.has_obstacles = False
    self.obstacles = []
```

**Problem**: Implicit initialization makes it unclear which init path sets what
**Decision**: Replace with explicit initialization in `__init__`
**Solution**:
```python
# In __init__ (before calling init methods)
self.hjb_geometry: GeometryProtocol | None = None
self.fp_geometry: GeometryProtocol | None = None
self.has_obstacles: bool = False
self.obstacles: list = []

# After init methods (no hasattr check needed)
if self.hjb_geometry is None:
    self.hjb_geometry = self.geometry
    self.fp_geometry = self.geometry
```

---

### 3. Legacy Override System (22 cases) ⚠️ **DEPRECATE**

**Locations**: `mfg_problem.py` lines 1117, 1152, 1186, 1229, 1262, etc.

```python
# Legacy property checking for deprecated parameters
@property
def Lx(self):
    if hasattr(self, "_Lx_override"):
        return self._Lx_override
    if self.geometry is not None:
        return self.geometry.get_bounds()[0][1] - self.geometry.get_bounds()[0][0]
    ...
```

**Problem**: Supporting dual API (legacy + geometry-first) with override system
**Root Cause**: Issue #544 (Geometry-First API transition incomplete)
**Decision**: Two-phase deprecation

**Phase 1 (Immediate)**: Make overrides explicit
```python
@property
def Lx(self):
    # Explicit check for private attribute we control
    if self._Lx_override is not None:
        return self._Lx_override
    ...
```

**Phase 2 (After #544)**: Remove override system entirely
```python
# Once geometry-first API is mandatory
@property
def Lx(self):
    if self.geometry is None:
        raise ValueError("Geometry not initialized - use geometry-first API")
    return self.geometry.get_bounds()[0][1] - self.geometry.get_bounds()[0][0]
```

---

## Implementation Plan

### Step 1: Fix Lazy Initialization (Low Risk)

**File**: `mfg_pde/core/mfg_problem.py`
**Lines**: 428-435

**Changes**:
1. Add explicit attribute initialization in `__init__` (before init method calls)
2. Replace `hasattr` checks with `is None` checks
3. Add type hints for clarity

**Estimated Impact**: Low - just making existing behavior explicit

---

### Step 2: Replace Override hasattr with Explicit Checks (Medium Risk)

**Files**: `mfg_pde/core/mfg_problem.py` (properties: Lx, Nx, dx, xSpace, grid, etc.)

**Pattern**:
```python
# Before
if hasattr(self, "_Lx_override"):
    return self._Lx_override

# After (Phase 1)
if self._Lx_override is not None:
    return self._Lx_override

# After (Phase 2 - requires #544)
# Remove override system entirely
```

**Requirements**:
1. Initialize all `_*_override` attributes to `None` in `__init__`
2. Document that these are private legacy support attributes
3. Add deprecation warnings when used

---

### Step 3: Remove Override System (Blocked by #544)

**Depends on**: Issue #544 (Geometry-First API transition)
**Removes**: ~20 hasattr checks
**Timeline**: After #544 completion

---

## Risk Assessment

| Category | Risk | Mitigation |
|:---------|:-----|:-----------|
| Lazy Init Fix | Low | Straightforward - just making implicit explicit |
| Override Explicit Check | Medium | Need to ensure all overrides initialized to None |
| Override Removal | High | Blocks #544 completion, requires full test suite |

---

## Testing Strategy

### Unit Tests
- Test explicit initialization (hjb_geometry, fp_geometry, has_obstacles)
- Test override properties still work after making explicit
- Test both legacy and geometry-first API paths

### Integration Tests
- Run full example suite (examples/basic/, examples/advanced/)
- Verify backward compatibility maintained
- Check no regressions in existing tests

### Validation
```bash
# Before changes
python scripts/check_fail_fast.py --path mfg_pde/core

# After Step 1 (should reduce by 2)
python scripts/check_fail_fast.py --path mfg_pde/core

# After Step 2 (should reduce by 22)
python scripts/check_fail_fast.py --path mfg_pde/core

# Target: 3 remaining (doc examples - intentional)
```

---

## Recommendations

**Immediate (This Session)**:
- ✅ Execute Step 1 (Lazy Init Fix) - Low risk, immediate win

**Short Term (This Week)**:
- Execute Step 2 (Override Explicit Check) - Reduces violations significantly
- Create sub-issue for Step 3 coordination with #544

**Medium Term (After #544)**:
- Remove override system entirely
- Achieve target: Only 3 hasattr in core (all in docstrings)

---

**Created**: 2026-01-11
**Author**: Issue #543 Resolution
**Status**: Analysis Complete, Ready for Implementation
