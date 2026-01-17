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

## Adopted Solution

### Design Principle

**Separation of concerns**:
1. HJB solver computes value function U(t,x)
2. **Caller computes optimal control** α*(t,x) from their Hamiltonian
3. FP solver takes drift velocity α* directly via `drift_field`

### Unified API

**Key Insight**: The existing `drift_field` parameter is already general enough! It accepts drift velocity α* for ANY Hamiltonian, not just quadratic. The issue was only in the documentation.

```python
# Quadratic H: Caller computes α* = -∇U
alpha_quadratic = -grad(U_hjb)
M = solver.solve_fp_system(M_initial=m0, drift_field=alpha_quadratic)

# L1 control: Caller computes α* = -sign(∇U)
alpha_L1 = -np.sign(grad(U_hjb))
M = solver.solve_fp_system(M_initial=m0, drift_field=alpha_L1)

# Quartic H: Caller computes α* = -(∇U)^(1/3)
alpha_quartic = -np.sign(grad(U_hjb)) * np.abs(grad(U_hjb)) ** (1/3)
M = solver.solve_fp_system(M_initial=m0, drift_field=alpha_quartic)

# Custom H: Any function of ∇U
alpha_custom = your_optimal_control_function(grad(U_hjb))
M = solver.solve_fp_system(M_initial=m0, drift_field=alpha_custom)
```

### Parameter Naming

**Single parameter** `drift_field`:
- **Type**: np.ndarray or callable
- **Meaning**: Drift velocity α*(t,x) computed from ANY Hamiltonian
- **Caller responsibility**: Compute α* = -∂_p H(x, ∇U, m) for their H
- **No assumptions**: Solver doesn't "know" about Hamiltonians, just advects the drift

**Why `drift_field`**:
- Clear, descriptive name
- Already established in codebase
- No need for multiple parameters (drift_velocity, drift_callable, etc.)
- Simple API: one way to specify drift

---

## Implementation (COMPLETED)

### Phase 1: Documentation Updates ✅

**Files**:
- `mfg_pde/alg/numerical/fp_solvers/fp_fdm.py`
- `mfg_pde/alg/numerical/fp_solvers/fp_gfdm.py`

**Changes**:
1. Updated `drift_field` docstring to clarify it accepts drift velocity α* for ANY Hamiltonian
2. Added examples for quadratic, L1, quartic control in docstrings
3. No API changes needed - existing parameter already supports all use cases

**Key Insight**: The API was already correct! The confusion was in documentation that suggested drift_field was for "value functions" when it actually accepts drift velocity.

### Phase 2: Test Coverage ✅

**File**: `tests/unit/fp_solvers/test_fp_nonquadratic.py`

**Tests added**:
1. Shape validation
2. Backward compatibility (quadratic H)
3. Equivalence test (drift_field gives same result for quadratic H)
4. L1 control drift
5. Quartic control drift
6. Zero drift (pure diffusion)
7. Custom drift
8. Mass conservation with drift_field

All tests passing ✅

### Phase 3: Examples & Documentation

**TODO**:
1. Create example: `examples/advanced/mfg_l1_control.py`
2. Update TOWEL_ON_BEACH_1D_PROTOCOL.md with non-quadratic H usage

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
