# Bug Investigation Summary: Non-Converging MFG Solver

**Date**: 2025-10-31
**Status**: CRITICAL - Fundamental bug preventing convergence

---

## Summary

Despite fixing three critical bugs (shape, padding, timestep), the MFG Picard iteration **does not converge**. Errors oscillate instead of decreasing monotonically, indicating a fundamental implementation error likely in drift calculation or HJB-FP coupling.

---

## Bugs Found and Fixed

### Bug #1: Shape calculation (FIXED ✓)
- **Location**: `hjb_fdm_multid.py`, `fp_fdm_multid.py` adapters
- **Issue**: Used `num_points-1` instead of `num_points` for array shapes
- **Impact**: Arrays had wrong shape, causing padding/indexing errors
- **Fix**: Changed to `tuple(problem.geometry.grid.num_points)`

### Bug #2: Boundary padding logic (FIXED ✓)
- **Location**: Adapter padding/unpadding in dimensional splitting
- **Issue**: Unnecessary padding when conventions already match
- **Impact**: Boundary conditions corrupted
- **Fix**: Removed padding logic (GridBasedMFGProblem now includes boundaries)

### Bug #3: FP solver timestep bug (FIXED ✓)
- **Location**: `fp_fdm_multid.py:244-249`, `_FPProblem1DAdapter.__init__`
- **Issue**: Adapter used full `dt` instead of sweep `dt/(2*ndim)` for Strang splitting
- **Impact**: 4× excessive time evolution in 2D → instability
- **Fix**: Added `sweep_dt` parameter to adapter, passed from `_sweep_dimension()`

**Code changes**:
```python
# Before (WRONG)
problem_1d = _FPProblem1DAdapter(
    full_problem=problem,
    sweep_dim=sweep_dim,
    fixed_indices=perp_indices,
    # Used problem.dt internally → BUG
)

# After (FIXED)
problem_1d = _FPProblem1DAdapter(
    full_problem=problem,
    sweep_dim=sweep_dim,
    fixed_indices=perp_indices,
    sweep_dt=dt,  # Pass dt/(2*ndim) for Strang splitting
)
```

### Bug #4: Naming conventions inconsistency (FIXED ✓)
- **Issue**: `thetaUM` used in factory, `damping_factor` in solver
- **Fix**: Updated `solver_factory.py:327` to use `damping_factor`
- **Documentation**: Created `docs/NAMING_CONVENTIONS.md` consolidated with `NOTATION_STANDARDS.md`

---

## Current Problem: Non-Convergence

### Observed Behavior

Tested multiple damping values (0.4, 0.6, 0.75) with 16×16 grid:

**Damping = 0.4** (lower than default):
```
Iter 1: U_err=1.01,  M_err=0.574
Iter 2: U_err=0.779, M_err=0.665
Iter 3: U_err=0.605, M_err=0.679
Iter 4: U_err=0.788, M_err=0.631  ← increased!
Iter 5: U_err=1.12,  M_err=0.707  ← even worse!
Iter 6: U_err=0.826, M_err=0.622
```

**Pattern**: Errors oscillate around 0.6-1.2 instead of decreasing monotonically.

### CFL Stability: SATISFIED ✓

Grid parameters (16×16, T=0.4, Nt=20):
- dx = dy = 0.0667
- dt = 0.02
- σ²·dt/dx² = 0.01125 < 0.5 ✓

**Conclusion**: Stability is NOT the issue.

### Mass Conservation: CORRECT ✓

Initial density normalized correctly:
- sum(m) = 225.0
- ∫ m dV = 1.000 (exact)

Initial mass conservation is working as expected.

---

## Hypothesis: Fundamental Coupling Bug

Since changing damping doesn't fix oscillation, the problem is likely:

1. **Sign error in drift calculation**
   - FP equation: `∂m/∂t + ∇·(m v) = (σ²/2) Δm`
   - Drift: `v = -coefCT ∇U`
   - Wrong sign on drift → opposite flow direction → instability

2. **Incorrect gradient computation**
   - Multi-dimensional gradient in dimensional splitting
   - Finite difference stencil may be wrong
   - Could be using wrong derivative order/direction

3. **HJB-FP coupling mismatch**
   - HJB gives U, FP needs ∇U for drift
   - Coupling through Hamiltonian derivatives
   - Sign conventions may not match between solvers

4. **Dimensional splitting implementation error**
   - Strang splitting order/timesteps
   - Sweep dimension indexing
   - Boundary conditions at sweep interfaces

---

## Next Steps

### Priority 1: Check Drift Calculation

Create diagnostic test to verify:
- [ ] Gradient ∇U is computed correctly
- [ ] Drift v = -coefCT ∇U has correct sign
- [ ] Drift is passed correctly to FP solver

### Priority 2: Test 1D First

- [ ] Create 1D test case (simpler than 2D)
- [ ] Check if 1D converges (to isolate dimensional splitting issues)
- [ ] Compare 1D FDM with known working solver

### Priority 3: Unit Test Individual Components

- [ ] Test HJB solver alone on known problem
- [ ] Test FP solver alone with prescribed velocity field
- [ ] Verify Hamiltonian coupling formulas

### Priority 4: Compare with Working Implementation

- [ ] Check if particle-based FP solver converges (doesn't use dimensional splitting)
- [ ] Compare with any existing 2D validation tests that pass
- [ ] Review original implementation before refactoring

---

## Files Modified

1. `mfg_pde/alg/numerical/fp_solvers/fp_fdm_multid.py` - Timestep bug fix
2. `mfg_pde/alg/numerical/mfg_solvers/fixed_point_iterator.py` - Naming conventions
3. `mfg_pde/factory/solver_factory.py` - Naming conventions
4. `docs/NAMING_CONVENTIONS.md` - Created and consolidated with NOTATION_STANDARDS.md

---

## Relevant References

- **Grid Conventions**: `docs/theory/foundations/NOTATION_STANDARDS.md` §7.2
- **Strang Splitting**: `fp_fdm_multid.py:20-24` (theory comments)
- **Adapter Pattern**: `fp_fdm_multid.py:287-336` (_FPProblem1DAdapter)

---

**Last Updated**: 2025-10-31
**Next Action**: Create diagnostic test for drift calculation
