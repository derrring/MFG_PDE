# Drift Query Interface Design (Issue #573)

**Created**: 2026-01-18
**Issue**: #573 - Callable drift interface for non-quadratic Hamiltonians
**Priority**: HIGH
**Size**: Medium (1-3 days)

---

## Problem Statement

### Current Limitation

Both FP FDM and GFDM solvers assume **quadratic Hamiltonians**:

```python
# Current API (FP FDM and GFDM)
M = solver.solve_fp_system(
    M_initial=m0,
    drift_field=U_hjb,  # Treated as VALUE FUNCTION
    ...
)
```

Internally, solvers compute drift as:
```python
grad_U = compute_gradient(U)
drift = -grad_U  # Assumes α* = -∇U (quadratic H only!)
```

This **only works for quadratic Hamiltonians**: H = (1/2)|p|²

### Non-Quadratic Hamiltonians

For general Hamiltonians, the optimal control is:

α*(x,t) = -∂_p H(x, ∇U, m)

| Hamiltonian | Optimal Control |
|:------------|:----------------|
| H = (1/2)p² | α* = -∇U |
| H = \|p\| (L1 control) | α* = -sign(∇U) |
| H = (1/4)p⁴ | α* = -(∇U)^(1/3) |
| State constraints | Projected gradient |

### Consequence

Users cannot solve MFG problems with non-quadratic control costs using FP solvers.

---

## Proposed Solution

### Design Principle

**Separation of concerns**:
1. HJB solver computes value function U(t,x)
2. **Caller computes optimal control** α*(t,x) from their Hamiltonian
3. FP solver takes drift velocity α* directly

### New API

```python
# Option 1: Pass precomputed drift velocity (array)
alpha_drift = compute_drift_from_hamiltonian(U_hjb, hamiltonian_type="L1")
M = solver.solve_fp_system(
    M_initial=m0,
    drift_velocity=alpha_drift,  # NEW: Drift velocity α*(t,x)
    ...
)

# Option 2: Pass drift computation callable
def compute_L1_drift(grad_U):
    """L1 control: α* = -sign(∇U)"""
    return -np.sign(grad_U)

M = solver.solve_fp_system(
    M_initial=m0,
    value_function=U_hjb,  # Value function for gradient computation
    drift_callable=compute_L1_drift,  # Maps ∇U → α*
    ...
)

# Option 3: Backward compatible (quadratic H)
M = solver.solve_fp_system(
    M_initial=m0,
    drift_field=U_hjb,  # DEPRECATED: Assumes α* = -∇U
    ...
)
```

### Parameter Naming

**Recommended**:
- `drift_velocity`: Actual drift velocity α*(t,x) (most explicit)
- `value_function`: Value function U(t,x) when using drift_callable
- `drift_callable`: Function (∇U) → α* for custom Hamiltonians

**Deprecated** (keep for backward compat):
- `drift_field`: Ambiguous - treated as U, computes -∇U internally

---

## Implementation Plan

### Phase 1: FP FDM Solver

**File**: `mfg_pde/alg/numerical/fp_solvers/fp_fdm.py`

1. Add new parameter `drift_velocity` to `solve_fp_system()`
2. Clarify `drift_field` deprecation path
3. Update internal logic:
   - If `drift_velocity` provided → use directly
   - If `drift_field` provided → compute -∇U (legacy path)
   - If both provided → raise error
4. Update docstring with clear examples

### Phase 2: FP GFDM Solver

**File**: `mfg_pde/alg/numerical/fp_solvers/fp_gfdm.py`

Same changes as FDM.

### Phase 3: Add drift_callable Option

For advanced users who want to compute drift from ∇U:

```python
def solve_fp_system(
    self,
    M_initial,
    drift_velocity=None,      # Direct drift α*(t,x)
    drift_callable=None,      # Callable: (grad_U) → α*
    value_function=None,      # U(t,x) for drift_callable
    drift_field=None,         # DEPRECATED
    ...
):
    if drift_velocity is not None:
        # Use provided drift directly
        alpha = drift_velocity
    elif drift_callable is not None:
        # Compute drift from gradient
        grad_U = compute_gradient(value_function)
        alpha = drift_callable(grad_U)
    elif drift_field is not None:
        # Legacy: assume quadratic H
        grad_U = compute_gradient(drift_field)
        alpha = -grad_U
    else:
        # Zero drift
        alpha = np.zeros_like(...)
```

### Phase 4: Documentation & Examples

1. Update docstrings with non-quadratic examples
2. Create example: `examples/advanced/mfg_l1_control.py`
3. Update TOWEL_ON_BEACH_1D_PROTOCOL.md with usage pattern
4. Add unit tests for L1, quartic Hamiltonians

---

## Backward Compatibility

**Strategy**: Deprecate `drift_field` gradually

1. **v0.18.0** (next release):
   - Add `drift_velocity` parameter
   - Keep `drift_field` working (no warning yet)
   - Document recommended migration

2. **v0.19.0**:
   - Add DeprecationWarning when `drift_field` used
   - Suggest `drift_velocity` instead

3. **v1.0.0**:
   - Remove `drift_field` parameter entirely
   - Only `drift_velocity` and `drift_callable` remain

---

## Testing Strategy

### Unit Tests

**Test file**: `tests/unit/fp_solvers/test_fp_nonquadratic.py`

1. **L1 control** (|p| Hamiltonian):
   - Create synthetic U field
   - Compute α* = -sign(∇U)
   - Verify FP solution matches analytical/reference

2. **Quartic control** (p⁴ Hamiltonian):
   - Create U field
   - Compute α* = -(∇U)^(1/3)
   - Verify FP evolution

3. **Backward compatibility**:
   - Verify `drift_field` still works (quadratic H)
   - Compare with `drift_velocity` path (should match)

### Integration Tests

**Test file**: `tests/integration/test_mfg_l1_control.py`

Full MFG solve with L1 control cost:
- HJB solver with L1 Hamiltonian
- Compute drift using sign function
- FP solver with drift_velocity
- Verify Nash equilibrium properties

---

## Documentation Examples

### Quadratic Hamiltonian (Current)

```python
# Standard MFG with quadratic control (α* = -∇U)
U_hjb = hjb_solver.solve(M_density)
M_fp = fp_solver.solve_fp_system(
    M_initial=m0,
    drift_field=U_hjb  # Works: assumes α* = -∇U
)
```

### L1 Control (New Feature)

```python
# MFG with L1 control cost (|α|)
U_hjb = hjb_solver.solve_hjb_L1(M_density)

# Compute optimal control for L1 Hamiltonian
grad_U = problem.compute_gradient(U_hjb)
alpha_L1 = -np.sign(grad_U)  # α* = -sign(∇U)

M_fp = fp_solver.solve_fp_system(
    M_initial=m0,
    drift_velocity=alpha_L1  # NEW: Direct drift specification
)
```

### Quartic Control (New Feature)

```python
# MFG with quartic control cost (α⁴)
U_hjb = hjb_solver.solve_hjb_quartic(M_density)
grad_U = problem.compute_gradient(U_hjb)

# α* = -(∇U)^(1/3) for H = (1/4)p⁴
alpha_quartic = -np.sign(grad_U) * np.abs(grad_U) ** (1/3)

M_fp = fp_solver.solve_fp_system(
    M_initial=m0,
    drift_velocity=alpha_quartic
)
```

---

## Success Criteria

✅ FP solvers can handle non-quadratic Hamiltonians
✅ Backward compatible with existing quadratic H code
✅ Clear API with explicit parameter names
✅ Comprehensive tests for L1 and quartic H
✅ Documentation with examples

---

## References

- Issue #573: https://github.com/derrring/MFG_PDE/issues/573
- TOWEL_ON_BEACH_1D_PROTOCOL.md: § "HJB-FP Coupling: General Design Principle"
- exp14a validation bug (mentioned in #573)
