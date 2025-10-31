# Bug #5: Upwind Implementation - Architectural Analysis

**Date**: 2025-10-31
**Status**: Investigation complete, implementation pending
**Files**: `mfg_pde/alg/numerical/hjb_solvers/base_hjb.py`

---

## Problem Summary

MFG Picard iterations fail to converge - errors oscillate between 0.4 and 3.3 instead of decreasing monotonically. Root cause hypothesis: HJB FDM solver uses **central difference** instead of **upwind discretization**, causing numerical oscillations in advection-dominated regions.

---

## Failed Approach: Lax-Friedrichs Viscosity

**What was tried**:
```python
# Added artificial viscosity term
p_central = (p_forward + p_backward) / 2.0
viscosity_term = alpha * (u_ip1 - 2*u_i + u_im1) / Dx
p_upwind = p_central - viscosity_term  # WRONG
```

**Why it failed**:
- Adding viscosity ≠ proper upwind
- Doesn't respect characteristic direction
- Just smooths solution without fixing wave propagation

**Result**: Still oscillates, no convergence improvement.

---

## Proper Upwind for HJB Equations

### Theory

For HJB equation:
```
∂u/∂t + H(x, m, ∇u) = 0
```

Characteristics propagate with velocity `v = ∂H/∂p`. Upwind means:
- If `v > 0`: use **backward** difference (information comes from left)
- If `v < 0`: use **forward** difference (information comes from right)

For typical MFG Hamiltonian:
```
H(x, m, p) = |p|²/(2σ²) + V(x, m)
```

Characteristic velocity:
```
v = ∂H/∂p = p/σ²
```

So upwind direction depends on **sign of p** (the gradient itself).

### Godunov Upwind

Simplest proper upwind:
```python
p_forward = (u_ip1 - u_i) / Dx
p_backward = (u_i - u_im1) / Dx
p_central = (p_forward + p_backward) / 2.0  # For sign check

# Choose based on characteristic direction
if p_central >= 0:
    p = p_backward  # Information from left
else:
    p = p_forward   # Information from right
```

This is monotone but only first-order accurate.

---

## Current Code Architecture

### Where Derivatives are Computed

`base_hjb.py:271` in `compute_hjb_residual()`:
```python
# Calculate derivatives using tuple notation
derivs = _calculate_derivatives(U_n_current_newton_iterate, i, Dx, Nx, clip=False)
```

`_calculate_derivatives()` (lines 77-155):
```python
def _calculate_derivatives(U_array, i, Dx, Nx, clip=False):
    p_forward = (u_ip1 - u_i) / Dx
    p_backward = (u_i - u_im1) / Dx
    p_central = (p_forward + p_backward) / 2.0  # Central difference
    return {(0,): u_i, (1,): p_central}
```

### Where Hamiltonian is Evaluated

`base_hjb.py:286`:
```python
hamiltonian_val = problem.H(x_idx=i, m_at_x=m_val, derivs=derivs, t_idx=t_idx_n)
```

---

## Implementation Options

### Option A: Modify `_calculate_derivatives` (Simple but Limited)

**Approach**: Replace central difference with Godunov upwind directly.

```python
def _calculate_derivatives(U_array, i, Dx, Nx, clip=False):
    p_forward = (u_ip1 - u_i) / Dx
    p_backward = (u_i - u_im1) / Dx

    # Godunov upwind: choose based on sign
    p_central_for_sign = (p_forward + p_backward) / 2.0
    if p_central_for_sign >= 0:
        p = p_backward  # Characteristic from left
    else:
        p = p_forward   # Characteristic from right

    return {(0,): u_i, (1,): p}
```

**Pros**:
- Minimal code change
- Fast to implement
- Works if characteristic velocity = p/σ² (typical case)

**Cons**:
- Assumes H = |p|²/(2σ²) + V(x,m) structure
- Doesn't work for general Hamiltonians
- First-order accurate only

### Option B: Hamiltonian-Aware Upwind (Proper but Complex)

**Approach**: Evaluate Hamiltonian with both forward and backward differences, choose via Godunov flux.

```python
# In compute_hjb_residual, replace lines 271-291:

# Compute both forward and backward derivatives
derivs_forward = {(0,): u_i, (1,): p_forward}
derivs_backward = {(0,): u_i, (1,): p_backward}

# Evaluate Hamiltonian with both
H_forward = problem.H(x_idx=i, m_at_x=m_val, derivs=derivs_forward, t_idx=t_idx_n)
H_backward = problem.H(x_idx=i, m_at_x=m_val, derivs=derivs_backward, t_idx=t_idx_n)

# Godunov flux: min for backward time, max for forward time
hamiltonian_val = max(H_forward, H_backward)  # For backward-in-time HJB
```

**Pros**:
- Works for any Hamiltonian
- Provably monotone
- Standard method in HJB literature

**Cons**:
- 2x Hamiltonian evaluations per grid point
- More complex implementation
- Requires understanding of Godunov numerical flux

### Option C: Use WENO Solver (Deferred)

WENO already has upwind built-in, but has separate numerical issues (NaN overflow from iteration 1). Fix WENO later as separate task.

---

## Recommended Next Step

**Implement Option A: Godunov upwind in `_calculate_derivatives`**

### Reasoning:
1. Fast to implement (5 lines of code)
2. Works for standard MFG Hamiltonians
3. Easy to test and verify
4. Can upgrade to Option B later if needed

### Implementation:
1. Modify `base_hjb.py:_calculate_derivatives()` lines 145-155
2. Add upwind selection based on sign of central difference
3. Test on 16×16 grid with 20 timesteps
4. Verify convergence: errors should decrease monotonically

### Expected Outcome:
- Picard iterations converge in < 20 iterations
- U_err decreases from ~1.0 → < 1e-4
- No more oscillations

---

## Files to Modify

1. **`mfg_pde/alg/numerical/hjb_solvers/base_hjb.py`** (lines 145-155)
   - Replace `p_central` with Godunov upwind selection

2. **Test**: Run `test_upwind_fix.log` scenario
   - 16×16 grid, T=0.4, 20 timesteps
   - Verify convergence

---

## References

- Osher & Sethian (1988): "Fronts propagating with curvature-dependent speed"
- Sethian (1996): "Level Set Methods"
- Achdou & Capuzzo-Dolcetta (2010): "Mean field games: numerical methods"

---

## Test Results: Godunov Upwind Implementation (2025-10-31)

### Implementation

Implemented Option A: Added `upwind` boolean parameter to `_calculate_derivatives()` in `base_hjb.py`.

```python
def _calculate_derivatives(U_array, i, Dx, Nx, clip=False, upwind=False):
    p_forward = (u_ip1 - u_i) / Dx
    p_backward = (u_i - u_im1) / Dx

    if upwind:
        # Godunov upwind: choose based on characteristic direction
        p_central_for_sign = (p_forward + p_backward) / 2.0
        if p_central_for_sign >= 0:
            p_value = p_backward  # Characteristic from left
        else:
            p_value = p_forward   # Characteristic from right
    else:
        # Central difference (default)
        p_value = (p_forward + p_backward) / 2.0

    return {(0,): u_i, (1,): p_value}
```

All residual and Jacobian computations updated to use `upwind=True`.

### Test Configuration

- Problem: 16×16 grid, T=0.4, 20 timesteps
- Solver: HJB FDM with Godunov upwind
- Damping: 0.6 (default)
- Max iterations: 100

### Results: **FAILED - Oscillations Persist**

Test timed out after 25 iterations (~3 minutes). Errors still oscillate:

| Iteration | U_err | M_err |
|-----------|-------|-------|
| 1 | 1.00e+00 | 7.69e-01 |
| 2 | 3.35e-01 | 1.12e+00 |
| 3 | 2.72e-01 | 7.86e-01 |
| 7 | 1.97e-01 | 9.80e-01 |
| 8 | 2.13e-01 | 1.05e+00 |
| 9 | 2.79e-01 | 1.06e+00 |
| 14 | 1.67e-01 | 1.03e+00 |
| 15 | 2.35e-01 | 1.00e+00 |
| 16 | 2.73e-01 | 1.08e+00 |
| 25 | 1.76e-01 | 1.11e+00 |

**Observations**:
- U_err oscillates between 0.17 and 0.28 (no monotonic decrease)
- M_err oscillates between 0.6 and 1.3 (large fluctuations)
- No convergence improvement vs central difference
- Same behavior as before upwind implementation

### Conclusion: Upwind Discretization Does NOT Fix the Problem

The Godunov upwind implementation did not resolve the convergence issue. This suggests:

1. **Upwind is not the root cause**: The oscillations are not primarily due to advection-dominated HJB discretization
2. **Problem is elsewhere**: Could be in:
   - FP solver (Fokker-Planck equation)
   - MFG coupling between HJB and FP
   - Picard damping strategy
   - Initial conditions or boundary conditions
   - Newton solver within HJB timesteps
3. **Upwind may still help**: Even if not the root cause, upwind is theoretically correct for HJB and may provide marginal stability improvements

---

## Next Steps: Reassess Root Cause Hypothesis

Since upwind discretization did not fix the convergence, need to investigate:

1. **FP solver stability**: Check if density evolution is causing oscillations
2. **MFG coupling**: Verify how U and M exchange information between Picard iterations
3. **Damping analysis**: Test different damping values (0.4, 0.75, 0.9)
4. **Newton convergence**: Check if HJB inner Newton solver is converging properly
5. **Hamiltonian evaluation**: Verify Hamiltonian is computed correctly with MFG coupling term

**Priority**: Investigate FP solver and MFG coupling next.

---

**Status**: Bug #5 root cause hypothesis rejected. Further investigation needed.
