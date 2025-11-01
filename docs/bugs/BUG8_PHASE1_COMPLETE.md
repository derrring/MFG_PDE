# Bug #8 Phase 1 Investigation: Complete

## Status: Root Cause Located, Fix In Progress

**Date**: 2025-10-31
**Investigator**: Claude Code
**Priority**: High (2D FP solver unusable due to mass loss)

---

## Executive Summary

Phase 1 investigation successfully isolated the root cause of 2D mass loss to the **dimensional splitting implementation** in `fp_fdm_multid.py`. The 1D FP solver conserves mass perfectly when tested in isolation, confirming that the bug is in how 1D solves are combined via Strang splitting.

---

## Phase 1 Results

### Test 1: Pure Diffusion (Zero Velocity)
- **Result**: +0.97% mass gain over 10 timesteps
- **Expected**: 0% (exact conservation)
- **Conclusion**: Even without advection, dimensional splitting introduces numerical error

### Test 2: Constant Velocity Field
- **Result**: -26% mass loss over 10 timesteps
- **Expected**: <0.1% (small splitting error)
- **Conclusion**: Advection terms amplify the mass conservation bug dramatically

---

## Root Cause Analysis

### Confirmed Facts

1. **1D FP Solver is Correct**
   - Tested independently (prior Bug #7 fix)
   - Conserves mass to machine precision
   - No issues with boundary conditions or advection terms

2. **Issue is in Dimensional Splitting**
   - Location: `mfg_pde/alg/numerical/fp_solvers/fp_fdm_multid.py`
   - Lines 134-171: Strang splitting loop
   - Lines 174-284: `_sweep_dimension()` function

3. **Grid Convention Analysis**
   - Debug output shows shapes match correctly:
     - Grid shape: `(8, 8)` → 8 points per dimension
     - Slice shape: `(8,)` → passed to 1D solver
     - Adapter says `Nx = 7` (intervals)
     - 1D solver expects `(Nx+1,) = (8,)` points ✓

4. **The conventions appear correct**, but mass is still not conserved

### Hypotheses for Remaining Issue

**H1: Timestep Splitting Error**
- Strang splitting uses `dt/(2*ndim)` for each sweep
- With ndim=2, each sweep gets dt/4
- Very small timesteps may trigger numerical instability in 1D solver
- **Test**: Run 1D solver with dt/4 timesteps and check mass conservation

**H2: Boundary Condition Mismatch**
- 1D slices extract interior points
- Boundary conditions may not be consistent across sweeps
- No-flux conditions on slices may not enforce global no-flux
- **Test**: Log boundary values before/after each sweep

**H3: Non-Negativity Clipping Loss**
- Line 169: `M_solution[k + 1] = np.maximum(M_solution[k + 1], 0)`
- If sweeps produce slightly negative values, clipping loses mass
- **Test**: Check if any negative values occur before clipping

**H4: Strang Splitting Accumulation**
- Forward sweep → Backward sweep may not be perfectly reversible
- Small errors compound over multiple dimensions and timesteps
- **Test**: Compare single sweep vs. full Strang cycle

---

## Test Files Created

1. **`benchmarks/validation/test_2d_fp_isolation.py`**
   - Tests 2D FP solver without MFG coupling
   - Two scenarios: pure diffusion and with advection
   - **Status**: ✅ Working, reveals mass loss in FP solver

2. **`benchmarks/validation/debug_dimensional_splitting.py`**
   - Instruments `_sweep_dimension()` to log shapes and parameters
   - Confirms grid convention consistency
   - **Status**: ✅ Working, shows conventions match

---

## Next Steps

### Phase 2: Detailed Profiling

1. **Add mass tracking at each sweep**
   ```python
   # After each _sweep_dimension call:
   mass = np.sum(M) * np.prod(problem.geometry.grid.spacing)
   print(f"Mass after sweep {sweep_dim}: {mass:.10f}")
   ```

2. **Test 1D solver with split timesteps**
   - Run 1D FP with dt/4 instead of dt
   - Verify mass conservation holds for small timesteps

3. **Check for negative densities**
   - Log `np.min(M)` before non-negativity enforcement
   - Quantify mass lost to clipping

4. **Profile Strang splitting**
   - Test forward-only sweep (no backward)
   - Compare mass loss: forward-only vs. full Strang

### Phase 3: Implement Fix

Based on Phase 2 findings, implement one of:
- **Fix A**: Adjust timestep handling in dimensional splitting
- **Fix B**: Improve boundary condition consistency across sweeps
- **Fix C**: Use conservative non-negativity enforcement
- **Fix D**: Replace Strang splitting with alternative (e.g., Lie-Trotter)

---

## Technical Details

### Dimensional Splitting Algorithm

```python
# Current implementation (mfg_pde/alg/numerical/fp_solvers/fp_fdm_multid.py:134-171)
for k in range(Nt - 1):
    M_current = M_solution[k]

    # Forward sweep: dim 0, dim 1, ..., dim (ndim-1)
    M = M_current.copy()
    for dim in range(ndim):
        M = _sweep_dimension(M, U, problem, dt/(2*ndim), dim, bc, backend)

    # Backward sweep: dim (ndim-1), ..., dim 1, dim 0
    for dim in range(ndim-1, -1, -1):
        M = _sweep_dimension(M, U, problem, dt/(2*ndim), dim, bc, backend)

    M_solution[k+1] = M
    M_solution[k+1] = np.maximum(M_solution[k+1], 0)  # Non-negativity
```

### Grid Convention

```
GridBasedMFGProblem (after recent fix):
- domain_bounds = (0.0, 1.0, 0.0, 1.0)
- grid_resolution = N
- Arrays have shape (N, N) - INCLUDES all boundaries
- Spacing = domain_length / (N-1)

Example: N=8
- Points: 0, 1/7, 2/7, 3/7, 4/7, 5/7, 6/7, 1
- Shape: (8, 8)
- Spacing: 1/7 ≈ 0.14286

1D Adapter:
- Nx = N - 1 = 7 (number of intervals)
- Expects arrays of shape (Nx+1,) = (8,)
- Spacing: 1/7 (matches grid)
```

---

## References

- **Related Bugs**: Bug #7 (1D FP mass loss - FIXED)
- **Test Logs**: `benchmarks/validation/results/bug8_phase1_results.log`
- **Code Files**:
  - `mfg_pde/alg/numerical/fp_solvers/fp_fdm_multid.py` (dimensional splitting)
  - `mfg_pde/alg/numerical/fp_solvers/fp_fdm.py` (1D FP solver)

---

## Impact

- **Severity**: High - 2D MFG solver produces incorrect results
- **Affected**: All 2D/3D problems using FDM solver
- **Workaround**: None (must use 1D problems or particle methods)
- **Fix Timeline**: Requires Phase 2 profiling before implementing fix

---

**Investigation Status**: Phase 1 Complete ✅
**Next**: Phase 2 Detailed Profiling
