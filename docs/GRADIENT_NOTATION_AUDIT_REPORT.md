# Gradient Notation Audit Report

**Date**: 2025-10-30
**Related Issue**: GitHub Issue #200 (Architecture Refactoring)
**Standard Document**: `docs/gradient_notation_standard.md`
**Purpose**: Complete audit of all HJB solvers for gradient notation compliance

---

## Executive Summary

This report documents the comprehensive audit of all HJB solvers in MFG_PDE to assess compliance with the tuple multi-index notation standard established in `gradient_notation_standard.md`.

**Key Findings**:
- ✅ **1 solver compliant**: `hjb_gfdm.py` (already uses tuple notation)
- ✅ **2 solvers MIGRATED**: `base_hjb.py`, `hjb_semi_lagrangian.py` (migrated 2025-10-30)
- ✅ **1 solver uses different paradigm**: `hjb_weno.py` (uses numpy.gradient, not p_values)
- ✅ **1 solver not applicable**: `hjb_network.py` (graph-based, no spatial gradients)

**Status**: ✅ **Phase 2 COMPLETE** - All core HJB solvers migrated with backward compatibility

---

## Audit Results by Solver

### 1. hjb_gfdm.py - ✅ COMPLIANT

**Status**: Fully compliant with tuple notation standard

**Evidence**:
- **Line 1544**: `derivs = self.approximate_derivatives(u_current, i)` returns tuple-indexed dictionary
- **Lines 1551-1556**: Gradient extraction uses tuple keys:

```python
if d == 1:
    p = derivs.get((1,), 0.0)
    laplacian = derivs.get((2,), 0.0)
elif d == 2:
    p_x = derivs.get((1, 0), 0.0)
    p_y = derivs.get((0, 1), 0.0)
    p = np.array([p_x, p_y])
    laplacian = derivs.get((2, 0), 0.0) + derivs.get((0, 2), 0.0)
```

**Recommendation**: No changes needed. This solver demonstrates the correct pattern.

---

### 2. base_hjb.py - ⚠️ NEEDS MIGRATION

**Status**: Non-compliant - uses string keys `{"forward": ..., "backward": ...}`

**Evidence**:

**Line 76-113**: `_calculate_p_values()` function:
```python
def _calculate_p_values(
    U_array: np.ndarray,
    i: int,
    Dx: float,
    Nx: int,
    clip: bool = False,
    clip_limit: float = P_VALUE_CLIP_LIMIT_FD_JAC,
) -> dict[str, float]:
    # ... computes p_forward, p_backward ...
    raw_p_values = {"forward": p_forward, "backward": p_backward}
    return _clip_p_values(raw_p_values, clip_limit) if clip else raw_p_values
```

**Line 198**: Usage in solver:
```python
p_values = _calculate_p_values(U_n_current_newton_iterate, i, Dx, Nx, clip=False)
```

**Line 212**: Passed to Hamiltonian:
```python
hamiltonian_val = problem.H(x_idx=i, m_at_x=m_val, p_values=p_values, t_idx=t_idx_n)
```

**Impact**:
- Affects all 1D FDM solvers (MFGProblem class)
- String keys `"forward"`, `"backward"` are dimension-specific (don't generalize to 2D)
- Risk of Bug #13 type errors if users expect different keys

**Recommendation**:
1. Migrate to tuple notation: `{(1,): p_central}` for 1D first derivative
2. Maintain backward compatibility for existing problem classes
3. Add deprecation warnings for string-key format

**Migration Strategy**:
```python
# NEW: Return tuple-indexed dictionary
def _calculate_p_values_tuple(...) -> dict[tuple, float]:
    # Central difference approximation
    p_central = (p_forward + p_backward) / 2.0
    return {
        (0,): u_current,  # Function value
        (1,): p_central,  # First derivative (central difference)
    }

# COMPATIBILITY: Convert to old format if needed
def _p_values_to_legacy(derivs: dict[tuple, float]) -> dict[str, float]:
    """Convert tuple notation to legacy string keys for backward compatibility."""
    p = derivs.get((1,), 0.0)
    return {"forward": p, "backward": p}
```

---

### 3. hjb_semi_lagrangian.py - ⚠️ NEEDS MIGRATION

**Status**: Non-compliant - uses string keys `{"forward": p, "backward": p}`

**Evidence**:

**Line 266**: In optimal control computation:
```python
def hamiltonian_objective(p):
    p_values = {"forward": p, "backward": p}  # Symmetric for optimization
    try:
        x_idx = int((x - self.problem.xmin) / self.dx)
        x_idx = np.clip(x_idx, 0, self.problem.Nx)
        return self.problem.H(x_idx, m, p_values, time_idx)
```

**Line 488**: Second usage in solver:
```python
# Create p_values dict expected by problem.H
p_values = {"forward": p, "backward": p}

return self.problem.H(x_idx, m, p_values, time_idx)
```

**Context**: Semi-Lagrangian method computes optimal control via optimization, then uses string-keyed p_values

**Impact**:
- Affects 1D semi-Lagrangian solvers
- Same Bug #13 risk as base_hjb.py
- String keys don't generalize to higher dimensions

**Recommendation**:
1. Migrate to tuple notation: `{(1,): p}` for 1D first derivative
2. Update problem.H() calls to use new format
3. Add backward compatibility layer

**Migration Strategy**:
```python
# NEW: Use tuple notation
def hamiltonian_objective(p):
    derivs = {
        (1,): p,  # First derivative
    }
    # Call H with tuple-indexed derivatives
    return self.problem.H(x_idx, m, derivs=derivs, time_idx=time_idx)
```

**Note**: This migration requires updating the problem.H() signature to accept `derivs` parameter instead of `p_values`.

---

### 4. hjb_weno.py - ✅ DIFFERENT PARADIGM (No changes needed)

**Status**: Uses `np.gradient()` for spatial derivatives, not p_values dictionaries

**Evidence**:

**Line 522**: CFL condition computation:
```python
max_speed = np.max(np.abs(np.gradient(u, dx))) + 1e-10
```

**Lines 690-691**: Gradient computation for stability:
```python
u_x = np.gradient(u, self.Dx, axis=0)
u_y = np.gradient(u, self.Dy, axis=1)
```

**Lines 807-809**: 3D gradients:
```python
u_x = np.gradient(u, self.Dx, axis=0)
u_y = np.gradient(u, self.Dy, axis=1)
u_z = np.gradient(u, self.Dz, axis=2)
```

**Analysis**:
- WENO solver uses finite-volume formulation
- Gradients computed directly via NumPy, not stored in dictionaries
- No `p_values` or `derivs` dictionaries passed to Hamiltonian
- This is a fundamentally different approach (structured grid with array slicing)

**Recommendation**: No changes needed. WENO uses a different computational paradigm where gradients are computed inline rather than stored in dictionaries.

**Conclusion** (2025-10-30): Confirmed that hjb_weno.py does NOT use the derivative dictionary pattern (`p_values` or `derivs`) at all. It evaluates Hamiltonians inline with hardcoded formulas like `hamiltonian = 0.5 * u_x[i]**2 + m[i]*u_x[i]` and never calls `problem.H()`. This is a fundamentally different finite-volume approach and correctly classified as "different paradigm."

---

### 5. hjb_network.py - ✅ NOT APPLICABLE (Graph-based solver)

**Status**: Uses graph gradient operators, not spatial derivatives

**Evidence**:

**Lines 90-94**: Graph gradient operator initialization:
```python
# Graph gradient operator (simplified representation)
self.gradient_ops = {}
# Build graph neighbors for gradient computation
for i in range(self.n_nodes):
    self.gradient_ops[i] = neighbors
```

**Lines 154, 174, 194, 331, 380**: Usage in HJB/FP equations:
```python
neighbors = self.gradient_ops[i]
```

**Analysis**:
- Network solver operates on graph structures, not continuous spatial domains
- "Gradient" means discrete differences between connected nodes
- No continuous spatial derivatives (∂u/∂x, ∂u/∂y)
- Not applicable to tuple notation standard

**Recommendation**: No changes needed. Graph-based methods don't use spatial gradient notation.

---

## Summary Table

| Solver | Current Notation | Tuple Compliant? | Action Required | Priority |
|--------|------------------|------------------|-----------------|----------|
| `hjb_gfdm.py` | Tuple: `{(1,0): ..., (0,1): ...}` | ✅ YES | None | N/A |
| `base_hjb.py` | String: `{"forward": ..., "backward": ...}` | ❌ NO | Migrate to tuple | HIGH |
| `hjb_semi_lagrangian.py` | String: `{"forward": p, "backward": p}` | ❌ NO | Migrate to tuple | HIGH |
| `hjb_weno.py` | NumPy arrays: `np.gradient(u, dx)` | ✅ N/A | None (different paradigm) | N/A |
| `hjb_network.py` | Graph operators: `self.gradient_ops[i]` | ✅ N/A | None (graph-based) | N/A |

---

## Migration Plan

### Phase 1: Prepare Backward Compatibility (Week 1)

1. **Create compatibility utilities**:
   - `_derivs_to_p_values()`: Convert tuple notation to legacy string keys
   - `_p_values_to_derivs()`: Convert string keys to tuple notation
   - Add to `mfg_pde/utils/numerical/gradient_utils.py`

2. **Update problem classes**:
   - Modify `MFGProblem.H()` to accept both `p_values` (legacy) and `derivs` (new)
   - Add deprecation warnings when `p_values` is used

### Phase 2: Migrate base_hjb.py (Week 2)

1. **Update `_calculate_p_values()`**:
   - Rename to `_calculate_derivatives()`
   - Return tuple-indexed dictionary `{(0,): u, (1,): p}`
   - Keep old function as wrapper with deprecation warning

2. **Update all callers**:
   - Lines 198, 295-373: Use new function
   - Convert to tuple notation internally
   - Use compatibility layer for problem.H() calls

3. **Add tests**:
   - Test both new and legacy interfaces
   - Verify backward compatibility

### Phase 3: Migrate hjb_semi_lagrangian.py (Week 3)

1. **Update optimal control computation** (lines 266, 488):
   - Use tuple notation `{(1,): p}`
   - Update problem.H() calls

2. **Add compatibility layer**:
   - Convert tuple to string keys if problem expects old format

3. **Test migration**:
   - Run semi-Lagrangian test suite
   - Verify no regressions

### Phase 4: Deprecation and Documentation (Week 4)

1. **Add deprecation warnings**:
   - Warn when string-keyed p_values detected
   - Provide migration guide

2. **Update documentation**:
   - Update all examples to use tuple notation
   - Add migration guide for users

3. **Update tutorials**:
   - Show new tuple notation in all examples
   - Document backward compatibility period

### Phase 5: Remove Legacy Support (6 months later)

1. **Remove compatibility layers**
2. **Make tuple notation mandatory**
3. **Update all tests to use new format**

---

## Code Examples

### Example 1: base_hjb.py Migration

**Before** (current):
```python
def _calculate_p_values(...) -> dict[str, float]:
    p_forward = (U_array[i+1] - U_array[i]) / Dx
    p_backward = (U_array[i] - U_array[i-1]) / Dx
    return {"forward": p_forward, "backward": p_backward}

# Usage
p_values = _calculate_p_values(U_n_current, i, Dx, Nx)
H = problem.H(x_idx=i, m_at_x=m, p_values=p_values, t_idx=t_idx)
```

**After** (migrated with backward compatibility):
```python
def _calculate_derivatives(...) -> dict[tuple, float]:
    """Calculate derivatives using tuple multi-index notation."""
    p_forward = (U_array[i+1] - U_array[i]) / Dx
    p_backward = (U_array[i] - U_array[i-1]) / Dx
    p_central = (p_forward + p_backward) / 2.0

    return {
        (0,): U_array[i],  # Function value
        (1,): p_central,   # First derivative (central difference)
    }

# Usage (new way)
derivs = _calculate_derivatives(U_n_current, i, Dx, Nx)
p = derivs[(1,)]  # Extract first derivative

# Backward compatibility: convert if problem expects old format
if hasattr(problem, '_uses_legacy_p_values'):
    p_values = {"forward": p, "backward": p}
    H = problem.H(x_idx=i, m_at_x=m, p_values=p_values, t_idx=t_idx)
else:
    H = problem.H(x_idx=i, m_at_x=m, derivs=derivs, t_idx=t_idx)
```

### Example 2: hjb_semi_lagrangian.py Migration

**Before** (current):
```python
def hamiltonian_objective(p):
    p_values = {"forward": p, "backward": p}
    return self.problem.H(x_idx, m, p_values, time_idx)
```

**After** (migrated):
```python
def hamiltonian_objective(p):
    derivs = {
        (1,): p,  # First derivative in 1D
    }
    # New interface: pass derivs instead of p_values
    return self.problem.H(x_idx, m, derivs=derivs, time_idx=time_idx)
```

---

## Testing Strategy

### Regression Tests

1. **Test tuple notation consistency** (already created in `test_gradient_notation_standard.py`)
2. **Test backward compatibility**:
   - Verify old string-key format still works during transition
   - Test conversion functions
3. **Test dimension agnostic behavior**:
   - 1D: `{(1,): p}`
   - 2D: `{(1,0): p_x, (0,1): p_y}`
   - 3D: `{(1,0,0): p_x, (0,1,0): p_y, (0,0,1): p_z}`

### Integration Tests

1. **Test migrated solvers** with existing problems
2. **Verify no performance regressions**
3. **Test with both old and new problem classes**

---

## Risk Assessment

### High Risk Areas

1. **Breaking existing user code**: Users with custom problem classes may rely on string-keyed p_values
2. **Signature changes**: Changing problem.H() signature affects all subclasses
3. **Test suite failures**: Many existing tests may expect old format

### Mitigation Strategies

1. **Maintain backward compatibility** for 6 months minimum
2. **Add clear deprecation warnings** with migration examples
3. **Update all examples** before deprecating old format
4. **Comprehensive testing** of both old and new interfaces

---

## Related Documents

- **Standard**: `docs/gradient_notation_standard.md` (tuple multi-index notation definition)
- **Bug #13 Analysis**: `docs/archived_bug_investigations/BUG_13_*.md` (root cause: string key mismatch)
- **Tests**: `tests/test_gradient_notation_standard.py` (11 regression tests)
- **GitHub Issue**: Issue #200 (Architecture Refactoring)

---

## Recommendations

### Immediate Actions (Week 1-2)

1. ✅ Complete this audit report
2. Create backward compatibility utilities
3. Begin base_hjb.py migration

### Short-term Actions (Week 3-4)

1. Migrate hjb_semi_lagrangian.py
2. Update all documentation
3. Add deprecation warnings

### Long-term Actions (6 months)

1. Remove legacy support
2. Require tuple notation in all new code
3. Update contribution guidelines

---

## Conclusion

The audit identified **2 solvers requiring migration** (`base_hjb.py`, `hjb_semi_lagrangian.py`) to comply with the tuple multi-index notation standard. The other solvers are either already compliant (`hjb_gfdm.py`) or use different computational paradigms (`hjb_weno.py`, `hjb_network.py`) where the standard doesn't apply.

**Priority**: HIGH - Begin migration immediately to prevent Bug #13 type errors

**Timeline**: 4 weeks for full migration with backward compatibility

**Impact**: Improves code consistency, prevents interface bugs, enables dimension-agnostic gradient handling

---

## Migration Progress Update

**Last Updated**: 2025-10-30

### ✅ Phase 2 Complete: Core HJB Solvers Migrated

**Completed Work**:

1. **Backward Compatibility Layer** (`mfg_pde/compat/gradient_notation.py`):
   - 6 conversion functions for format translation
   - 21 comprehensive tests - all passing ✅
   - Auto-detection of gradient format
   - Deprecation warnings for legacy formats

2. **base_hjb.py Migration**:
   - Created new `_calculate_derivatives()` function using tuple notation
   - Converted `_calculate_p_values()` to backward-compatible wrapper
   - Test Results: 22/22 passed ✅
   - No changes required to existing test code

3. **hjb_semi_lagrangian.py Migration**:
   - Updated `_compute_optimal_control()` method (lines 268-270)
   - Updated `_compute_hamiltonian()` method (lines 492-493)
   - Test Results: 13/13 passed ✅ (1 skipped)
   - Full backward compatibility maintained

**Test Coverage**: All 56 total tests passing with zero breaking changes

**Next Steps**:
- Phase 3: Update `MFGProblem` classes to accept both formats
- Add deprecation warnings for user code
- Update example problems and documentation

---

**Report Status**: Complete (Updated with migration progress)
**Approved by**: Architecture Refactoring Team
**Date**: 2025-10-30
