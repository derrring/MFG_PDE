# REVISED Protocol Compliance Plan

**Created**: 2025-11-10
**Status**: Planning Phase
**Supersedes**: AGGRESSIVE_NAMING_MIGRATION_PLAN.md (overly aggressive)

---

## Executive Summary

**CORRECTION**: The existing protocols (`GridProblem`, `DirectAccessProblem`) **already use** `xSpace`, `tSpace`, `Dx`, `Dt`. These are the **established standard** that all solvers depend on.

**We should NOT change them.** Instead, we should:
1. Add **missing properties** for dimension-agnostic support
2. Complete **array notation migration** (Issue #243)
3. Optionally add **convenience aliases** (not replacements)

---

## What the Protocols Actually Specify

### GridProblem (Current Standard)
```python
@runtime_checkable
class GridProblem(Protocol):
    # Spatial grid structure
    xmin: float
    xmax: float
    Nx: int
    Dx: float
    xSpace: NDArray      # ✅ KEEP - this is the standard

    # Temporal structure
    T: float
    Nt: int
    Dt: float            # ✅ KEEP - this is the standard
    tSpace: NDArray      # ✅ KEEP - this is the standard

    # Physical parameters
    sigma: float
    coupling_coefficient: float
```

**These names are CORRECT and should be PRESERVED.**

---

## Real Issues to Fix

### Issue 1: Missing nD Properties ⭐ **HIGH PRIORITY**

**Problem**: No dimension-agnostic way to access grid properties for 2D/3D

**Current state**:
```python
# 1D mode
problem.Nx  # ✅ Works
problem.Dx  # ✅ Works

# 2D mode
problem.spatial_discretization  # ✅ Works → [50, 50]
problem.spatial_shape  # ⚠️ Exists but should be grid_shape
# problem.grid_spacing  # ❌ Missing! Need [dx, dy]
```

**Solution**: Add new properties (don't replace existing):
```python
@property
def grid_shape(self) -> tuple[int, ...]:
    """Grid shape (N₀, N₁, ...) for all dimensions."""
    if hasattr(self, 'Nx') and self.dimension == 1:
        return (self.Nx,)
    elif hasattr(self, 'spatial_discretization'):
        return tuple(self.spatial_discretization)
    elif hasattr(self, 'geometry'):
        return self.geometry.get_grid_shape()

@property
def grid_spacing(self) -> list[float]:
    """Grid spacing [Δx₀, Δx₁, ...] for each dimension."""
    if hasattr(self, 'Dx') and self.dimension == 1:
        return [self.Dx]
    elif hasattr(self, 'spatial_bounds') and hasattr(self, 'spatial_discretization'):
        return [(xmax - xmin) / N for (xmin, xmax), N in
                zip(self.spatial_bounds, self.spatial_discretization)]
    elif hasattr(self, 'geometry'):
        return self.geometry.get_grid_spacing()
```

**Impact**: Enables dimension-agnostic solver code

---

### Issue 2: Array Notation Migration (Issue #243) ⭐ **MEDIUM PRIORITY**

**Problem**: Scalar notation for 1D creates type inconsistency

**Current**:
```python
# 1D - scalars (inconsistent with nD)
MFGProblem(Nx=100, xmin=0.0, xmax=1.0, ...)
problem.Nx  # int

# 2D - arrays (different API!)
MFGProblem(spatial_discretization=[50, 50], spatial_bounds=[(0,1), (0,1)], ...)
problem.spatial_discretization  # list
```

**Target**:
```python
# 1D - ALSO uses arrays (consistent!)
MFGProblem(spatial_discretization=[100], spatial_bounds=[(0.0, 1.0)], ...)
problem.spatial_discretization  # list (length 1)

# 2D - unchanged
MFGProblem(spatial_discretization=[50, 50], spatial_bounds=[(0,1), (0,1)], ...)
problem.spatial_discretization  # list (length 2)
```

**Backward compatibility**:
```python
# Old scalar API still works with deprecation warning
MFGProblem(Nx=100, xmin=0.0, xmax=1.0, ...)
# DeprecationWarning: Passing scalar Nx is deprecated. Use spatial_discretization=[100]

# Internal normalization
self.spatial_discretization = self._normalize_to_array(Nx, "Nx")  # [100]
self.Nx = self.spatial_discretization[0]  # Backward compat property
```

**Scope**: ~65 files need updating (examples, tests)

---

### Issue 3: Convenience Aliases (Optional) 💡 **LOW PRIORITY**

**Question**: Should we add lowercase aliases for convenience?

**Option A: Add Aliases** (minimal effort, nice-to-have):
```python
@property
def dt(self) -> float:
    """Lowercase alias for Dt (no deprecation, just convenience)."""
    return self.Dt

@property
def dx(self) -> float:
    """Lowercase alias for Dx in 1D (convenience)."""
    return self.Dx if hasattr(self, 'Dx') else self.grid_spacing[0]
```

**Benefit**: Users can write `problem.dt` if they prefer lowercase
**No breaking change**: Both `Dt` and `dt` work forever

**Option B: Keep as-is**:
- Protocols use `Dt`, `Dx` (capital D)
- Keep it consistent
- Don't add unnecessary aliases

---

## What We Should NOT Do

### ❌ Do NOT Replace Protocol Names

**Bad idea (from previous plan)**:
```python
# ❌ DON'T DO THIS
xSpace → spatial_grid  # Would break all solvers!
tSpace → time_grid     # Would break all solvers!
Dt → dt (replacement)  # Would break all solvers!
Dx → dx (replacement)  # Would break all solvers!
```

**Why**: These are defined in `GridProblem` and `DirectAccessProblem` protocols. Changing them would break:
- All HJB solvers
- All FP solvers
- All examples
- All tests
- All user code

---

## Revised Implementation Plan

### Phase 1: Add Missing Properties (1-2 days) ⭐ **DO THIS**

**Files to modify**:
1. `mfg_pde/core/mfg_problem.py`
2. `mfg_pde/core/base_problem.py`

**Add**:
```python
@property
def grid_shape(self) -> tuple[int, ...]:
    """Grid shape tuple (N₀, N₁, ...) for all modes."""
    # Implementation for 1D/nD/geometry

@property
def grid_spacing(self) -> list[float]:
    """Grid spacing [Δx₀, Δx₁, ...] for all modes."""
    # Implementation for 1D/nD/geometry

# Optional: Convenience aliases
@property
def dt(self) -> float:
    """Lowercase alias for Dt."""
    return self.Dt
```

**Test**:
```python
# tests/unit/test_core/test_protocol_compliance.py
def test_grid_properties_1d():
    problem = MFGProblem(Nx=100, xmin=0, xmax=1, T=1, Nt=50)
    assert problem.grid_shape == (100,)
    assert len(problem.grid_spacing) == 1
    assert problem.grid_spacing[0] == pytest.approx(0.01)

def test_grid_properties_2d():
    problem = MFGProblem(
        spatial_discretization=[50, 50],
        spatial_bounds=[(0, 1), (0, 1)],
        T=1, Nt=50
    )
    assert problem.grid_shape == (50, 50)
    assert len(problem.grid_spacing) == 2
```

**Effort**: ~2-4 hours (implementation + testing)

---

### Phase 2: Array Notation Migration (1-2 weeks) ⭐ **DO THIS**

**This is Issue #243 Phase 2** - already documented.

**Scope**:
- Update examples to use `spatial_discretization=[100]` instead of `Nx=100`
- Update tests similarly
- Keep backward compatibility with deprecation warnings
- ~65 files affected

**Defer to v0.12.0** (let v0.11.0 stabilize first)

---

### Phase 3: Optional Aliases (1 day) 💡 **MAYBE**

**Question for user**: Do you want lowercase convenience aliases?

**If yes**:
```python
problem.dt  # Alias for problem.Dt (both work)
problem.dx  # Alias for problem.Dx or problem.grid_spacing[0]
```

**If no**: Keep only capital D names (`Dt`, `Dx`)

---

## Comparison: Old Plan vs Revised Plan

| Aspect | Old Plan (❌ Wrong) | Revised Plan (✅ Correct) |
|:-------|:-------------------|:--------------------------|
| **xSpace** | Replace with spatial_grid | ✅ Keep xSpace (it's in protocol) |
| **tSpace** | Replace with time_grid | ✅ Keep tSpace (it's in protocol) |
| **Dt** | Replace with dt | ✅ Keep Dt (maybe add dt alias) |
| **Dx** | Replace with grid_spacing[0] | ✅ Keep Dx (maybe add dx alias) |
| **grid_shape** | Add as replacement | ✅ Add as NEW property (not replacement) |
| **grid_spacing** | Add as replacement for Dx | ✅ Add as NEW list property (Dx stays) |
| **Effort** | 3 weeks, 400+ changes | 2-4 hours core + 1-2 weeks examples |
| **Breaking changes** | Massive | None (all additive) |

---

## Decision Points

### A. Grid Properties (NEW) - Recommended ✅

**Add these new properties**:
- `grid_shape` → tuple `(N₀, N₁, ...)`
- `grid_spacing` → list `[Δx₀, Δx₁, ...]`

**Keep existing**:
- `Nx`, `Dx` (1D mode)
- `xSpace`, `tSpace`, `Dt` (all modes)

**Benefit**: Dimension-agnostic code while preserving protocols

---

### B. Lowercase Aliases - Your Choice 🤷

**Option 1: Add aliases** (nice-to-have):
```python
problem.dt  # Same as problem.Dt
problem.dx  # Same as problem.Dx (1D) or problem.grid_spacing[0]
```

**Option 2: Don't add aliases** (keep simple):
- Stick with capital D (`Dt`, `Dx`)
- Protocols use capital D
- Less code to maintain

**My recommendation**: Option 2 (don't add aliases). Capital D is fine.

---

### C. Array Notation (Issue #243) - Defer

Already planned for v0.12.0. Don't combine with this work.

---

## Immediate Next Steps

1. **Confirm approach** with user:
   - ✅ Keep `xSpace`, `tSpace`, `Dt`, `Dx` (don't replace)
   - ✅ Add `grid_shape` and `grid_spacing` properties
   - ❓ Add lowercase aliases (`dt`, `dx`)? Or skip?

2. **If approved**:
   - Create branch: `feat/grid-properties-protocol-compliance`
   - Implement `grid_shape` and `grid_spacing` properties
   - Add protocol compliance test
   - PR and merge

**Estimated time**: 2-4 hours (not 3 weeks!)

---

**Status**: ⏸️ **AWAITING USER CONFIRMATION**

Should we:
1. Add `grid_shape` and `grid_spacing` properties? (Recommended: YES)
2. Add lowercase aliases `dt`, `dx`? (Recommended: NO, but your choice)
3. Proceed with Issue #243 array notation separately? (Recommended: YES, defer to v0.12.0)
