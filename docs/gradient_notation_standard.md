# Gradient Notation Standard for MFG_PDE

**Created**: 2025-10-30
**Status**: Reference Document for Refactoring
**Related**: Bug #13 (dictionary key mismatch), GitHub Issue #200 (Architecture Refactoring)

---

## Executive Summary

This document establishes the standard gradient notation for MFG_PDE solvers to prevent interface bugs like Bug #13, where a 2-character key mismatch (`"dx"` vs `"x"`) broke the entire navigation system.

**Standard Adopted**: Tuple-based multi-index notation `derivs[(α,β,γ,...)]`

---

## The Problem: Three Different Notations

### Current State (Inconsistent)

**Notation 1: String keys** (Bug #13 source)
```python
# RISKY: Used in some user code
p_values = {"dx": grad_x, "dy": grad_y}  # Mismatch with "x", "y"
```

**Notation 2: Dictionary with direction keys**
```python
# Used in base_hjb.py for 1D
p_values = {"forward": p_forward, "backward": p_backward}
```

**Notation 3: Tuple multi-index** (Recommended)
```python
# Used in hjb_gfdm.py internally
derivs = {
    (1, 0): du_dx,      # First derivative in x
    (0, 1): du_dy,      # First derivative in y
    (2, 0): d2u_dx2,    # Second derivative in x
    (1, 1): d2u_dxdy,   # Mixed derivative
}
```

---

## Standard: Tuple Multi-Index Notation

### Definition

For a function u(x₁, x₂, ..., xₙ), derivatives are indexed by tuples (α₁, α₂, ..., αₙ) where αᵢ is the derivative order with respect to xᵢ.

### Examples

**1D**: `u(x)`
- `derivs[(0,)] = u` - Function value
- `derivs[(1,)] = ∂u/∂x` - First derivative
- `derivs[(2,)] = ∂²u/∂x²` - Second derivative

**2D**: `u(x, y)`
- `derivs[(0, 0)] = u` - Function value
- `derivs[(1, 0)] = ∂u/∂x` - Gradient x-component
- `derivs[(0, 1)] = ∂u/∂y` - Gradient y-component
- `derivs[(2, 0)] = ∂²u/∂x²` - Hessian xx
- `derivs[(0, 2)] = ∂²u/∂y²` - Hessian yy
- `derivs[(1, 1)] = ∂²u/∂x∂y` - Mixed derivative

**3D**: `u(x, y, z)`
- `derivs[(1, 0, 0)] = ∂u/∂x`
- `derivs[(0, 1, 0)] = ∂u/∂y`
- `derivs[(0, 0, 1)] = ∂u/∂z`
- `derivs[(2, 0, 0)] = ∂²u/∂x²`
- `derivs[(1, 1, 0)] = ∂²u/∂x∂y`

### Benefits

1. **Dimension-agnostic**: Works for 1D, 2D, 3D, nD without special cases
2. **Type-safe**: Tuples are hashable and immutable
3. **Mathematical clarity**: Direct correspondence to multi-index notation
4. **No ambiguity**: `(1,0)` is unambiguous, `"dx"` vs `"x"` is not
5. **Extensibility**: Higher-order derivatives naturally supported

---

## Current Status by Solver

**COMPREHENSIVE AUDIT COMPLETED**: See `docs/GRADIENT_NOTATION_AUDIT_REPORT.md` for full details.

### ✅ hjb_gfdm.py (Compliant)

**Status**: ✅ Fully compliant with tuple notation standard

**Evidence** (hjb_gfdm.py:1544-1556):
```python
derivs = self.approximate_derivatives(u_current, i)

if d == 1:
    p = derivs.get((1,), 0.0)
    laplacian = derivs.get((2,), 0.0)
elif d == 2:
    p_x = derivs.get((1, 0), 0.0)
    p_y = derivs.get((0, 1), 0.0)
    p = np.array([p_x, p_y])
    laplacian = derivs.get((2, 0), 0.0) + derivs.get((0, 2), 0.0)
```

**Action**: None needed - this solver is the reference implementation

---

### ⚠️ base_hjb.py (NEEDS MIGRATION)

**Status**: ⚠️ Non-compliant - uses string keys `{"forward": ..., "backward": ...}`

**Evidence** (base_hjb.py:76-113, 198, 212):
```python
def _calculate_p_values(...) -> dict[str, float]:
    # ... computes p_forward, p_backward ...
    raw_p_values = {"forward": p_forward, "backward": p_backward}
    return raw_p_values

# Usage
p_values = _calculate_p_values(U_n_current, i, Dx, Nx)
hamiltonian_val = problem.H(x_idx=i, m_at_x=m_val, p_values=p_values, t_idx=t_idx_n)
```

**Impact**: Affects all 1D FDM solvers (MFGProblem class)
**Priority**: HIGH
**Action**: Migrate to tuple notation with backward compatibility layer

---

### ⚠️ hjb_semi_lagrangian.py (NEEDS MIGRATION)

**Status**: ⚠️ Non-compliant - uses string keys `{"forward": p, "backward": p}`

**Evidence** (hjb_semi_lagrangian.py:266, 488):
```python
def hamiltonian_objective(p):
    p_values = {"forward": p, "backward": p}  # Symmetric for optimization
    return self.problem.H(x_idx, m, p_values, time_idx)
```

**Impact**: Affects 1D semi-Lagrangian solvers
**Priority**: HIGH
**Action**: Migrate to tuple notation with backward compatibility layer

---

### ✅ hjb_weno.py (Different Paradigm - No changes needed)

**Status**: ✅ Uses `np.gradient()` for spatial derivatives, not p_values dictionaries

**Evidence** (hjb_weno.py:690-691):
```python
u_x = np.gradient(u, self.Dx, axis=0)
u_y = np.gradient(u, self.Dy, axis=1)
```

**Analysis**: WENO uses finite-volume formulation with inline gradient computation
**Action**: None needed - different computational paradigm

---

### ✅ hjb_network.py (Not Applicable - Graph-based)

**Status**: ✅ Uses graph gradient operators, not spatial derivatives

**Evidence** (hjb_network.py:90-94):
```python
# Graph gradient operator (simplified representation)
self.gradient_ops = {}
for i in range(self.n_nodes):
    self.gradient_ops[i] = neighbors
```

**Analysis**: Network solver operates on graph structures, not continuous spatial domains
**Action**: None needed - tuple notation doesn't apply to graph gradients

---

## Bug #13 Root Cause Analysis

### What Went Wrong

**User problem code** (`maze_mfg_problem.py:380,383`):
```python
# WRONG: Used "dx", "dy" keys
p_values_dict = {"dx": p_gradient[0], "dy": p_gradient[1]}
```

**Hamiltonian expected** (`"x"`, `"y"` keys):
```python
def H(self, x, m_at_x, p_values, t_idx):
    p_x = p_values.get("x", 0.0)  # Returns 0.0 when key is "dx"!
    p_y = p_values.get("y", 0.0)
    control_cost = (p_x**2 + p_y**2) / (2 * self.lambda_cost)
    # ... control_cost = 0 because p_x = p_y = 0!
```

### Why Tuple Notation Prevents This

**With tuple notation**:
```python
# Solver computes
derivs = {(1, 0): grad_x, (0, 1): grad_y}

# User extracts (type error if wrong key!)
p_x = derivs[(1, 0)]  # KeyError if typo in tuple
p_y = derivs[(0, 1)]

# Or with .get() for optional derivatives
p_x = derivs.get((1, 0), 0.0)  # Typo (1,0) vs (10,) caught by tests
```

**Type safety**: Tuples are immutable, hashable, and have clear structure. String typos like `"dx"` vs `"x"` are silent failures, but tuple typos `(1, 0)` vs `(10,)` are structurally different.

---

## Migration Plan

### Phase 1: Documentation and Tests (Week 1)

1. ✅ Create this standard document
2. ✅ Write regression tests for gradient notation
3. Document current state of all solvers
4. Create migration guide for users

### Phase 2: Base HJB Solver (Week 2)

1. Update `base_hjb.py` to use tuple notation internally
2. Add compatibility layer for old `p_values` dict format
3. Update unit tests
4. Run full test suite

### Phase 3: Problem Classes (Week 3-4)

1. Update `MFGProblem` to accept both formats (backward compatibility)
2. Add deprecation warnings for non-tuple formats
3. Update all example problems to use tuple notation
4. Update documentation and tutorials

### Phase 4: Other Solvers (Week 5)

1. Audit hjb_semi_lagrangian.py
2. Audit hjb_weno.py
3. Audit hjb_network.py
4. Standardize all to tuple notation

### Phase 5: Deprecation (6 months later)

1. Remove compatibility layers
2. Require tuple notation in all new code
3. Mark old formats as errors (not warnings)

---

## Regression Tests

### Test 1: Tuple Notation Consistency

```python
def test_tuple_notation_consistency():
    """Verify all HJB solvers use tuple-indexed derivatives."""
    from mfg_pde.alg.numerical.hjb_solvers.hjb_gfdm import HJBGFDMSolver

    # Create minimal problem
    problem = create_test_problem_2d()
    solver = HJBGFDMSolver(problem)

    # Compute derivatives at a test point
    u_test = np.random.rand(solver.n_points)
    derivs = solver.approximate_derivatives(u_test, 0)

    # Verify tuple keys exist
    assert (1, 0) in derivs, "Missing ∂u/∂x derivative"
    assert (0, 1) in derivs, "Missing ∂u/∂y derivative"
    assert (2, 0) in derivs, "Missing ∂²u/∂x² derivative"
    assert (0, 2) in derivs, "Missing ∂²u/∂y² derivative"

    # Verify no string keys
    for key in derivs.keys():
        assert isinstance(key, tuple), f"Non-tuple key found: {key}"
```

### Test 2: No Silent Failures

```python
def test_no_silent_failures_with_typo():
    """Ensure typos in gradient keys raise errors, not silent defaults."""
    derivs = {(1, 0): 1.5, (0, 1): 2.3}

    # Correct access works
    assert derivs.get((1, 0), 0.0) == 1.5

    # Typo returns default (but test catches this)
    typo_result = derivs.get((10,), 0.0)  # Wrong: (10,) instead of (1,0)
    assert typo_result == 0.0, "Typo should return default"

    # Recommend using KeyError for required derivatives
    try:
        p_x = derivs[(10,)]  # Raises KeyError
        assert False, "Should have raised KeyError"
    except KeyError:
        pass  # Expected
```

### Test 3: Dimension Agnostic

```python
def test_dimension_agnostic():
    """Verify tuple notation works for 1D, 2D, 3D."""
    # 1D
    derivs_1d = {(0,): 1.0, (1,): 0.5, (2,): -0.1}
    assert derivs_1d[(1,)] == 0.5

    # 2D
    derivs_2d = {(0,0): 1.0, (1,0): 0.5, (0,1): 0.3}
    assert derivs_2d[(1,0)] == 0.5

    # 3D
    derivs_3d = {(0,0,0): 1.0, (1,0,0): 0.5, (0,1,0): 0.3, (0,0,1): 0.2}
    assert derivs_3d[(1,0,0)] == 0.5
```

---

## API Contract for problem.H()

### Current Status

Different problem classes have different H() signatures:

**1. MFGProblem (1D)**:
```python
def H(self, x_idx: int, m_at_x: float, p_values: dict[str, float], t_idx: int) -> float:
    p = p_values.get("forward", 0.0)  # or "backward"
    return 0.5 * self.lambda_cost * p**2  # Quadratic control cost
```

**2. User problems (2D+)**:
```python
def H(self, x: np.ndarray, p: np.ndarray, m: float) -> float:
    # p is gradient array [p_x, p_y, ...]
    return np.dot(p, p) / (2 * self.lambda_cost)
```

### Proposed Standard

**Option A: Array interface (recommended for nD)**
```python
def H(self, x: np.ndarray, p: np.ndarray, m: float, t: float = 0.0) -> float:
    """
    Hamiltonian function.

    Args:
        x: Spatial position (shape: (d,))
        p: Gradient/momentum (shape: (d,))
        m: Density at x
        t: Time (optional)

    Returns:
        Hamiltonian value
    """
    return np.dot(p, p) / (2 * self.lambda_cost) + running_cost(x, m)
```

**Option B: Tuple-based interface (more flexible)**
```python
def H(self, x: np.ndarray, derivs: dict[tuple, float], m: float, t: float = 0.0) -> float:
    """
    Hamiltonian function with full derivative information.

    Args:
        x: Spatial position
        derivs: Dictionary of derivatives indexed by tuples
                derivs[(1,0)] = ∂u/∂x, derivs[(0,1)] = ∂u/∂y, etc.
        m: Density at x
        t: Time (optional)

    Returns:
        Hamiltonian value
    """
    d = len(x)
    if d == 1:
        p = derivs.get((1,), 0.0)
    elif d == 2:
        p = np.array([derivs.get((1,0), 0.0), derivs.get((0,1), 0.0)])

    return np.dot(p, p) / (2 * self.lambda_cost)
```

**Recommendation**: Keep Option A (array interface) for simplicity. The tuple-to-array conversion happens in the solver, not in user code.

---

## References

- **Bug #13 Documentation**: `/Users/zvezda/OneDrive/code/mfg-research/docs/archived_bug_investigations/BUG_13_*.md`
- **Architecture Refactoring**: MFG_PDE GitHub Issue #200
- **GFDM Implementation**: `mfg_pde/alg/numerical/hjb_solvers/hjb_gfdm.py:1544-1563`
- **Base HJB**: `mfg_pde/alg/numerical/hjb_solvers/base_hjb.py:198-212`

---

## Conclusion

**Standardizing on tuple multi-index notation**:
1. Prevents Bug #13 type interface errors (silent key mismatches)
2. Provides dimension-agnostic code (1D, 2D, 3D, nD)
3. Matches mathematical notation (multi-indices)
4. Enables type safety (tuples are immutable, hashable)
5. Extensibility to higher-order derivatives

**Current status**: hjb_gfdm.py already compliant, base_hjb.py needs migration, other solvers need audit.

**Next steps**: See Migration Plan phases 1-5 above.

---

**Document Status**: Ready for review and implementation
**Last Updated**: 2025-10-30
**Author**: Architecture Refactoring Team (based on Bug #13 lessons learned)
