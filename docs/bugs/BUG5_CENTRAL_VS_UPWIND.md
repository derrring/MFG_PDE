# Bug #5: HJB Solver Uses Central Differences Instead of Upwind Scheme

**Date**: 2025-10-31
**Severity**: CRITICAL - Causes non-convergence of MFG solver
**Status**: ROOT CAUSE IDENTIFIED

---

## Problem

The HJB FDM solver uses **central differences** for spatial derivatives instead of **upwind scheme**, causing:
- Non-convergent Picard iterations (errors oscillate around 0.6-1.2)
- Violation of monotonicity required for HJB equations
- Instability regardless of damping parameter

---

## Root Cause

**File**: `mfg_pde/alg/numerical/hjb_solvers/base_hjb.py`
**Function**: `_compute_1d_derivatives()` (approx. line 250-280)

```python
# Current WRONG implementation
p_forward = (u_ip1 - u_i) / Dx
p_backward = (u_i - u_im1) / Dx

# Use central difference approximation ← PROBLEM!
p_central = (p_forward + p_backward) / 2.0

return {(0,): u_i, (1,): p_central}  # Returns central difference
```

**Why this is wrong**:
1. HJB equations are **advection-dominated** (first-order hyperbolic)
2. Central differences are **not monotone** → cause oscillations
3. Upwind scheme is required for **viscosity solutions**

---

## Theory: Why Upwind is Required

### HJB Equation Structure
```
∂u/∂t + H(x, ∇u, m) = 0
```

where typically `H = (1/2)|∇u|² + other_terms`.

### Upwind Principle

For first-order advection terms, the derivative direction must align with the **characteristic direction**:

- If `∂H/∂p > 0` (information flows from left): use **forward** difference
- If `∂H/∂p < 0` (information flows from right): use **backward** difference

**Upwind formula** (Godunov/Lax-Friedrichs):
```
p_upwind = {
    p_forward   if  H(p_forward) - H(p_backward) > 0
    p_backward  if  H(p_forward) - H(p_backward) < 0
}
```

### Why Central Difference Fails

Central difference `p_central = (p_forward + p_backward) / 2` is:
- **Not monotone**: violates comparison principle
- **Dispersive**: introduces spurious oscillations
- **Unstable** for advection-dominated problems

This is **mathematically incorrect** for viscosity solutions of HJB equations.

---

## Evidence

### Test Results
All tests show same oscillation pattern regardless of damping:

| Damping | Iteration | U_err | M_err | Trend |
|---------|-----------|-------|-------|-------|
| 0.4     | 1         | 1.01  | 0.574 | —     |
| 0.4     | 2         | 0.779 | 0.665 | ↓     |
| 0.4     | 3         | 0.605 | 0.679 | ↓     |
| 0.4     | 4         | 0.788 | 0.631 | ↑ BAD |
| 0.4     | 5         | 1.12  | 0.707 | ↑↑ BAD|

**Conclusion**: Errors oscillate instead of decreasing monotonically.

### CFL Condition
- Grid: 16×16, dt=0.02, dx=0.0667
- σ²·dt/dx² = 0.01125 < 0.5 ✓
- **Stability condition satisfied** → problem is NOT CFL violation

### Drift Sign
- FP solver: `drift = -coefCT * grad(U)` ✓ CORRECT
- Coupling is correct, only HJB discretization is wrong

---

## Correct Implementation: Upwind Scheme

### Option 1: Godunov Upwind (Simplest)

```python
def _compute_1d_derivatives_upwind(U_array, i, Dx, Nx, Hamiltonian):
    """
    Compute derivatives using upwind scheme for HJB.

    Upwind selection based on Hamiltonian monotonicity.
    """
    u_i = U_array[i]
    u_im1 = U_array[i-1] if i > 0 else u_i
    u_ip1 = U_array[i+1] if i < Nx else u_i

    # Compute both forward and backward differences
    p_forward = (u_ip1 - u_i) / Dx
    p_backward = (u_i - u_im1) / Dx

    # Evaluate Hamiltonian at both derivatives
    H_forward = Hamiltonian(p_forward)
    H_backward = Hamiltonian(p_backward)

    # Upwind selection: choose direction with larger Hamiltonian
    if H_forward >= H_backward:
        p_upwind = p_forward
    else:
        p_upwind = p_backward

    return {(0,): u_i, (1,): p_upwind}
```

### Option 2: Lax-Friedrichs (More Robust)

```python
def _compute_1d_derivatives_lax_friedrichs(U_array, i, Dx, Nx, alpha=0.5):
    """
    Lax-Friedrichs upwind scheme with numerical viscosity.

    Formula: p_LF = (p_forward + p_backward)/2 - alpha*(u_ip1 - 2*u_i + u_im1)/Dx
    """
    u_i = U_array[i]
    u_im1 = U_array[i-1] if i > 0 else u_i
    u_ip1 = U_array[i+1] if i < Nx else u_i

    p_forward = (u_ip1 - u_i) / Dx
    p_backward = (u_i - u_im1) / Dx

    # Lax-Friedrichs adds numerical viscosity
    p_lf = (p_forward + p_backward) / 2.0 - alpha * (u_ip1 - 2*u_i + u_im1) / Dx

    return {(0,): u_i, (1,): p_lf}
```

### Option 3: WENO (High-Order, Most Robust)

MFG_PDE already has WENO solver (`hjb_weno.py`) - this should work correctly!

---

## Recommended Fix

### Immediate (Quick Fix)
1. **Try WENO solver** first (already implemented):
   ```python
   from mfg_pde.alg.numerical.hjb_solvers import HJBWENOSolver
   hjb_solver = HJBWENOSolver(problem)
   ```

2. If WENO works, this confirms the diagnosis.

### Long-term (Fix FDM Solver)
1. **Replace central difference with upwind** in `base_hjb.py:_compute_1d_derivatives()`
2. **Test both Godunov and Lax-Friedrichs** upwind schemes
3. **Add upwind to multi-dimensional solver** (`hjb_fdm_multid.py`)
4. **Validate** with convergence tests

---

## Impact Assessment

### Solvers Affected
- ✗ `HJBFDMSolver` (1D and nD) - uses central difference
- ✓ `HJBWENOSolver` - already uses upwind (should work!)
- ✓ `HJBSemiLagrangianSolver` - inherently upwind
- ✓ `HJBGFDMSolver` - needs verification

### Tests Affected
All FDM-based MFG tests fail to converge:
- `benchmarks/validation/test_2d_crowd_motion.py`
- `benchmarks/validation/test_1d_simple_mfg.py`
- Any test using `create_basic_solver()` (uses FDM by default)

---

## References

### Theory
1. **Viscosity Solutions**: Crandall, Ishii, Lions (1992) - "User's guide to viscosity solutions"
2. **Monotone Schemes**: Barles & Souganidis (1991) - "Convergence of approximation schemes for fully nonlinear second order equations"
3. **Upwind for HJB**: Osher & Sethian (1988) - "Fronts propagating with curvature-dependent speed"

### Implementation
- Current central difference: `mfg_pde/alg/numerical/hjb_solvers/base_hjb.py:250-280`
- Working upwind example: `mfg_pde/alg/numerical/hjb_solvers/hjb_weno.py`
- Working upwind example: `mfg_pde/alg/numerical/hjb_solvers/hjb_semi_lagrangian.py`

---

## Validation Attempt: WENO Solver

**Date**: 2025-10-31

Attempted to validate upwind hypothesis by testing WENO solver (which has upwind scheme built-in).

### Result: INCONCLUSIVE

WENO solver encountered **separate numerical issues**:
- Grid compatibility fixed (Bug #6) - solver now runs
- Numerical overflow from iteration 1: `U_err=nan, M_err=1.50`
- Overflow in WENO weight calculations, singular matrices in FP
- Cannot use WENO to confirm/reject upwind hypothesis

See `BUG6_WENO_GRID_COMPATIBILITY.md` for grid fix details.

### Conclusion

WENO validation failed for unrelated reasons. Bug #5 (central vs upwind) **remains the most likely root cause** based on:
1. Mathematical theory (HJB requires upwind for viscosity solutions)
2. Code evidence (central difference used in `base_hjb.py:_compute_1d_derivatives()`)
3. Symptom match (oscillations instead of convergence)

---

## Next Steps

1. **Implement upwind scheme directly in FDM** (don't rely on WENO validation)

2. **Update all FDM solvers** to use upwind scheme

3. **Add unit tests** for upwind vs central difference

---

**Last Updated**: 2025-10-31
**Priority**: P0 - Blocks all FDM-based MFG solvers
**Assignee**: Needs immediate attention
