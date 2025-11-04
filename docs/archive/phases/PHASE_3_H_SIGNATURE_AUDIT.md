# Phase 3: H() Signature Audit Report

**Date**: 2025-10-30
**Related Issue**: GitHub Issue #200 (Architecture Refactoring)
**Previous Phase**: Phase 2 (Core HJB Solvers) - Complete
**Current Phase**: Phase 3 (Problem Classes Migration)

---

## Executive Summary

Completed audit of all H() method signatures to understand current patterns and design backward-compatible migration to tuple notation.

**Finding**: Current signature is **consistent** across all grid-based problems:
```python
def H(self, x_idx: int, m_at_x: float, p_values: dict[str, float], t_idx: int | None = None) -> float
```

**Next Step**: Update to accept BOTH formats:
```python
def H(
    self,
    x_idx: int,
    m_at_x: float,
    derivs: dict[tuple, float] | None = None,  # NEW (preferred)
    p_values: dict[str, float] | None = None,  # LEGACY (deprecated)
    t_idx: int | None = None,
) -> float
```

---

## Files Audited

### 1. mfg_pde/core/mfg_problem.py - MAIN TARGET

**Current H() Signature** (lines 271-277):
```python
def H(
    self,
    x_idx: int,
    m_at_x: float,
    p_values: dict[str, float],  # Legacy string-key format
    t_idx: int | None = None,
) -> float:
```

**Implementation Details**:

1. **Default Hamiltonian** (lines 309-361):
   - Extracts: `p_forward = p_values.get("forward")` and `p_backward = p_values.get("backward")`
   - Returns NaN for invalid inputs
   - Computes: `H = 0.5 * coefCT * (npart²(p_fwd) + ppart²(p_bwd)) - V(x) - m²`

2. **Custom Hamiltonian** (lines 283-307):
   - Calls user-provided function via `self.components.hamiltonian_func()`
   - Passes `p_values` in legacy format to user code
   - Same parameters: `x_idx`, `x_position`, `m_at_x`, `p_values`, `t_idx`, `current_time`, `problem`

3. **Validation** (lines 206-218):
   - Function signature validation enforces parameter names
   - Required parameters: `["x_idx", "m_at_x", "p_values", "t_idx"]`
   - **CRITICAL**: Must update to accept BOTH `derivs` and `p_values`

**Related Methods**:
- `dH_dm()` (lines 363-404): Hamiltonian derivative w.r.t. density - same signature pattern
- `get_hjb_hamiltonian_jacobian_contrib()` (lines 406-456): Optional Jacobian
- `get_hjb_residual_m_coupling_term()` (lines 458-498): Coupling term

---

### 2. mfg_pde/types/problem_protocols.py - PROTOCOL DEFINITIONS

**GridProblem Protocol** (lines 174-208):
```python
def H(
    self,
    x_idx: int,
    m_at_x: float,
    p_values: dict[str, float],  # Legacy format in protocol
    t_idx: int | None = None,
) -> float:
```

**CollocationProblem Protocol** (lines 72-91):
```python
def H(self, x: NDArray, p: NDArray, m: float) -> float:
```

**Analysis**:
- **GridProblem**: Must be updated to accept both `derivs` and `p_values`
- **CollocationProblem**: Uses different paradigm (continuous position, array gradients) - NOT affected by migration
- **DirectAccessProblem** (lines 212-252): No H() method - NOT applicable

---

### 3. mfg_pde/utils/cli.py - CLI UTILITIES

**Example H() Implementation** (lines 371-372):
```python
def H(self, x, p, m):
    return 0.5 * p**2
```

**Analysis**:
- Simplified example in CLI code
- Not a real problem implementation
- Will need updating in documentation examples

---

## Current Signature Usage

### Where H() is Called (from Phase 2 audit):

1. **base_hjb.py** (line 212):
   ```python
   hamiltonian_val = problem.H(x_idx=i, m_at_x=m_val, p_values=p_values, t_idx=t_idx_n)
   ```
   - ✅ ALREADY MIGRATED: Internally uses tuple notation, converts to legacy for H() call

2. **hjb_semi_lagrangian.py** (lines ~268, 493):
   ```python
   return self.problem.H(x_idx, m, p_values, time_idx)
   ```
   - ✅ ALREADY MIGRATED: Internally uses tuple notation, converts to legacy for H() call

3. **hjb_gfdm.py** (line ~1565):
   ```python
   H = self.problem.H(x_idx=i, m_at_x=m[i], p_values=p_values, t_idx=t_idx)
   ```
   - ✅ ALREADY USES TUPLE NOTATION: Converts tuple `derivs` to `p_values` before calling H()

**Key Finding**: All solvers ALREADY convert tuple notation to legacy `p_values` before calling `problem.H()`. Once we update `problem.H()` to accept `derivs` directly, we can remove these conversion layers.

---

## Validation System

**Current Validation** (mfg_problem.py, lines 220-228):
```python
def _validate_function_signature(self, func: Callable, name: str, expected_params: list) -> None:
    """Validate function signature has expected parameters."""
    sig = inspect.signature(func)
    params = list(sig.parameters.keys())

    # Allow extra parameters, but require the expected ones
    missing = [p for p in expected_params if p not in params]
    if missing:
        raise ValueError(f"{name} must accept parameters: {expected_params}. Missing: {missing}")
```

**Current Expected Parameters** (line 210):
```python
expected_params = ["x_idx", "m_at_x", "p_values", "t_idx"]
```

**Required Update**:
```python
# NEW: Accept EITHER derivs OR p_values
def _validate_function_signature(self, func: Callable, name: str, expected_params: list) -> None:
    sig = inspect.signature(func)
    params = list(sig.parameters.keys())

    # Special handling for gradient notation migration
    if "p_values" in expected_params:
        # Allow EITHER 'derivs' (new) OR 'p_values' (legacy)
        has_derivs = "derivs" in params
        has_p_values = "p_values" in params

        if not (has_derivs or has_p_values):
            raise ValueError(f"{name} must accept either 'derivs' or 'p_values' parameter")

        # Remove p_values from required params if user has derivs
        if has_derivs and not has_p_values:
            expected_params = [p for p in expected_params if p != "p_values"]

    # Check remaining required parameters
    missing = [p for p in expected_params if p not in params]
    if missing:
        raise ValueError(f"{name} must accept parameters: {expected_params}. Missing: {missing}")
```

---

## Migration Design

### Proposed New Signature

**MFGProblem.H()** (updated):
```python
def H(
    self,
    x_idx: int,
    m_at_x: float,
    derivs: dict[tuple, float] | None = None,      # NEW (preferred)
    p_values: dict[str, float] | None = None,      # LEGACY (deprecated)
    t_idx: int | None = None,
    x_position: float | None = None,               # Optional (for compatibility)
    current_time: float | None = None,             # Optional (for compatibility)
) -> float:
    """
    Hamiltonian function H(x, m, p, t).

    Args:
        x_idx: Grid index (0 to Nx)
        m_at_x: Density at grid point x_idx
        derivs: Derivatives in tuple notation (NEW, preferred):
                - 1D: {(0,): u, (1,): du/dx}
                - 2D: {(0,0): u, (1,0): ∂u/∂x, (0,1): ∂u/∂y}
        p_values: Momentum dictionary (LEGACY, deprecated):
                  {"forward": p_forward, "backward": p_backward}
        t_idx: Time index (optional)
        x_position: Actual position coordinate (computed from x_idx if not provided)
        current_time: Actual time value (computed from t_idx if not provided)

    Returns:
        Hamiltonian value H(x, m, p, t)

    Note:
        Provide EITHER derivs OR p_values. If both are provided, derivs takes precedence.
        p_values is deprecated and will be removed in a future version.
    """
    # Auto-detection and conversion
    if derivs is None and p_values is None:
        raise ValueError("Must provide either 'derivs' or 'p_values'")

    if derivs is None:
        # Legacy mode: convert p_values to derivs
        warnings.warn(
            "p_values parameter is deprecated. Use derivs instead. "
            "See docs/gradient_notation_standard.md for migration guide.",
            DeprecationWarning,
            stacklevel=2,
        )
        from mfg_pde.compat.gradient_notation import ensure_tuple_notation
        derivs = ensure_tuple_notation(p_values, dimension=1, u_value=0.0)

    # Compute x_position and current_time if not provided
    if x_position is None:
        x_position = self.xSpace[x_idx]
    if current_time is None and t_idx is not None:
        current_time = self.tSpace[t_idx] if t_idx < len(self.tSpace) else 0.0

    # Use custom Hamiltonian if provided
    if self.is_custom and self.components is not None and self.components.hamiltonian_func is not None:
        # Check if custom function accepts 'derivs' or 'p_values'
        sig = inspect.signature(self.components.hamiltonian_func)
        params = list(sig.parameters.keys())

        if "derivs" in params:
            # New-style custom Hamiltonian
            return self.components.hamiltonian_func(
                x_idx=x_idx,
                x_position=x_position,
                m_at_x=m_at_x,
                derivs=derivs,
                t_idx=t_idx,
                current_time=current_time,
                problem=self,
            )
        else:
            # Legacy custom Hamiltonian - convert derivs to p_values
            from mfg_pde.compat.gradient_notation import derivs_to_p_values_1d
            p_values_legacy = derivs_to_p_values_1d(derivs)

            return self.components.hamiltonian_func(
                x_idx=x_idx,
                x_position=x_position,
                m_at_x=m_at_x,
                p_values=p_values_legacy,
                t_idx=t_idx,
                current_time=current_time,
                problem=self,
            )

    # Default Hamiltonian implementation (uses tuple notation internally)
    p = derivs.get((1,), 0.0)  # Extract first derivative

    if np.isnan(p) or np.isinf(p) or np.isnan(m_at_x) or np.isinf(m_at_x):
        return np.nan

    # Use upwind scheme for default Hamiltonian
    npart_val = float(npart(p))
    ppart_val = float(ppart(p))

    if abs(npart_val) > VALUE_BEFORE_SQUARE_LIMIT or abs(ppart_val) > VALUE_BEFORE_SQUARE_LIMIT:
        return np.nan

    try:
        term_npart_sq = npart_val**2
        term_ppart_sq = ppart_val**2
    except OverflowError:
        return np.nan

    if np.isinf(term_npart_sq) or np.isnan(term_npart_sq) or np.isinf(term_ppart_sq) or np.isnan(term_ppart_sq):
        return np.nan

    hamiltonian_control_part = 0.5 * self.coefCT * (term_npart_sq + term_ppart_sq)
    potential_cost_V_x = self.f_potential[x_idx]
    coupling_density_m_x = m_at_x**2

    result = hamiltonian_control_part - potential_cost_V_x - coupling_density_m_x

    if np.isinf(result) or np.isnan(result):
        return np.nan

    return result
```

**Similar Update for dH_dm()**:
```python
def dH_dm(
    self,
    x_idx: int,
    m_at_x: float,
    derivs: dict[tuple, float] | None = None,
    p_values: dict[str, float] | None = None,
    t_idx: int | None = None,
) -> float:
    # Same auto-detection and conversion logic
    ...
```

---

## Protocol Update

**GridProblem Protocol** (updated):
```python
@runtime_checkable
class GridProblem(Protocol):
    """Grid-based problem interface."""

    # ... existing attributes ...

    def H(
        self,
        x_idx: int,
        m_at_x: float,
        derivs: dict[tuple, float] | None = None,      # NEW
        p_values: dict[str, float] | None = None,      # LEGACY
        t_idx: int | None = None,
    ) -> float:
        """
        Hamiltonian function H(x, m, p, t).

        Supports both tuple notation (derivs) and legacy string-key (p_values).
        Provide EITHER derivs OR p_values, not both.
        """
        ...
```

---

## Breaking Changes Assessment

**Backward Compatibility**: ✅ FULL

**Why**:
1. Both `derivs` and `p_values` are optional (`None` default)
2. Auto-detection chooses appropriate format
3. Legacy `p_values` still works (with deprecation warning)
4. Existing custom problems continue working unchanged

**Migration Path**:
1. **Phase 3 (now)**: Update signature, add auto-detection, emit warnings
2. **6 months**: Strong deprecation warnings in release notes
3. **12 months**: Remove `p_values` parameter (breaking change, major version bump)

---

## Testing Strategy

### Backward Compatibility Tests

```python
def test_h_accepts_both_formats():
    """Test H() accepts both derivs and p_values."""
    problem = MFGProblem(Nx=50, Nt=25)

    # New format (tuple notation)
    derivs = {(0,): 1.0, (1,): 0.5}
    H1 = problem.H(x_idx=25, m_at_x=0.01, derivs=derivs, t_idx=10)

    # Legacy format (string keys)
    p_values = {"forward": 0.5, "backward": 0.5}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        H2 = problem.H(x_idx=25, m_at_x=0.01, p_values=p_values, t_idx=10)

        # Should emit deprecation warning
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)

    # Should produce same result
    assert np.isclose(H1, H2)

def test_custom_hamiltonian_both_formats():
    """Test custom Hamiltonian with both derivs and p_values."""

    # New-style custom Hamiltonian
    def custom_H_new(x_idx, x_position, m_at_x, derivs, t_idx, current_time, problem):
        p = derivs[(1,)]
        return 0.5 * p**2 + m_at_x

    # Legacy custom Hamiltonian
    def custom_H_legacy(x_idx, x_position, m_at_x, p_values, t_idx, current_time, problem):
        p = p_values["forward"]
        return 0.5 * p**2 + m_at_x

    # Both should work
    # ... test both implementations
```

---

## Timeline

**Week 1: Implementation**
- Day 1: Update MFGProblem.H() and dH_dm() signatures ✅ (next task)
- Day 2: Update GridProblem protocol
- Day 3: Update validation logic
- Day 4: Update solver calls to use derivs directly
- Day 5: Testing and bug fixes

**Week 2: Examples and Documentation**
- Update all example problems
- Update documentation
- Update tutorials

**Week 3: Final Testing**
- Comprehensive backward compatibility tests
- Performance benchmarks
- Final review

---

## Next Steps

1. ✅ Complete this audit report
2. Implement new H() signature in `mfg_problem.py`
3. Update `GridProblem` protocol
4. Update validation logic
5. Test backward compatibility

---

**Status**: Audit complete, ready for implementation
**Date**: 2025-10-30
**Next Task**: Implement new H() signature in MFGProblem
