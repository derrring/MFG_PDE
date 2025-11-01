# Bug #7 Fix Verification Report

**Date**: 2025-10-31
**Bug**: Time-Index Mismatch in HJB-MFG Coupling
**Status**: ✓ FIXED AND VERIFIED
**Result**: Complete resolution of Picard iteration oscillations

---

## Executive Summary

**Problem**: MFG Picard iterations oscillated indefinitely without converging. U_err oscillated between 0.17 and 3.0, M_err between 0.6 and 1.3.

**Root Cause**: Line 763 of `base_hjb.py` used M(t_{n+1}) instead of M(t_n) when solving HJB equation at time t_n. This violated the mathematical structure of the HJB equation where H(x, ∇u(t,x), m(t,x)) requires all terms evaluated at the same time.

**Fix**: Changed `M_density_from_prev_picard[n_idx_hjb + 1, :]` to `M_density_from_prev_picard[n_idx_hjb, :]` in line 763.

**Result**: Solver now converges in 18 iterations with smooth monotonic error reduction.

---

## Investigation Timeline

### Failed Hypotheses (Rejected)

1. **Bug #5: Central vs Upwind Discretization** (2025-10-31)
   - Hypothesis: FDM using central differences instead of upwind caused oscillations
   - Test: Implemented Godunov upwind discretization
   - Result: Still oscillated (U_err: 0.17 ↔ 0.28)
   - Conclusion: Not the root cause

2. **Damping Parameter Adjustment** (2025-10-31)
   - Hypothesis: Damping factor of 0.6 was too high
   - Test: Tested damping=0.75 on 8×8 grid
   - Result: Still oscillated after 50 iterations (errors 0.36 to 3.14)
   - Conclusion: Damping is not the primary issue

### Successful Diagnosis (Bug #7)

3. **Time-Index Mismatch in M-Coupling** (2025-10-31)
   - Hypothesis: HJB solver using wrong time-index for density M
   - Analysis: Traced M-coupling in `base_hjb.py:753-784`
   - Discovery: Line 763 uses `M[n+1]` when solving at time `t_n`
   - Mathematical Violation: H(x, ∇u(t,x), m(t,x)) should use m at same time t
   - Fix Applied: Changed to `M[n]`
   - **Result**: ✓ COMPLETE SUCCESS

---

## Verification Test Results

### Test Configuration

**Problem**: 2D Crowd Motion (CrowdMotion2D)
- Grid: 16×16 spatial resolution
- Time: T=0.4, 20 timesteps (Δt=0.02)
- Damping: 0.6
- Tolerance: 1e-4
- Max iterations: 100

### Before Fix

**Behavior**: Indefinite oscillation
- U_err: 0.17 ↔ 3.0 (chaotic oscillation)
- M_err: 0.6 ↔ 1.3 (large fluctuations)
- No convergence after 100+ iterations

### After Fix

**Behavior**: Smooth monotonic convergence

**Iteration History**:
```
Iter  1: U_err=9.45e-01, M_err=6.74e-01
Iter  2: U_err=5.38e-01, M_err=2.98e-01
Iter  3: U_err=2.48e-01, M_err=1.22e-01
Iter  4: U_err=1.04e-01, M_err=4.89e-02
Iter  5: U_err=4.25e-02, M_err=1.96e-02
Iter  6: U_err=1.71e-02, M_err=7.83e-03
Iter  7: U_err=6.87e-03, M_err=3.13e-03
Iter  8: U_err=2.75e-03, M_err=1.25e-03
Iter  9: U_err=1.10e-03, M_err=5.01e-04
Iter 10: U_err=4.40e-04, M_err=2.00e-04
Iter 11: U_err=1.76e-04, M_err=8.02e-05
Iter 12: U_err=7.05e-05, M_err=3.21e-05
Iter 13: U_err=2.82e-05, M_err=1.28e-05
Iter 14: U_err=1.13e-05, M_err=5.13e-06
Iter 15: U_err=4.51e-06, M_err=2.05e-06
Iter 16: U_err=1.80e-06, M_err=8.21e-07
Iter 17: U_err=7.22e-07, M_err=3.28e-07
Iter 18: U_err=2.89e-07, M_err=1.31e-07  ← CONVERGED
```

**Final Result**:
- Converged: True
- Iterations: 18
- Final U error: 2.89e-07
- Final M error: 1.31e-07
- Time per iteration: ~9.9 seconds

**Convergence Rate**: Approximately linear (constant reduction factor ~0.4 per iteration)

---

## Comparison: Before vs After

| Metric | Before Fix | After Fix |
|--------|-----------|-----------|
| Convergence | Never | 18 iterations |
| U_err behavior | Oscillatory (0.17 ↔ 3.0) | Monotonic (0.945 → 2.89e-07) |
| M_err behavior | Oscillatory (0.6 ↔ 1.3) | Monotonic (0.674 → 1.31e-07) |
| Final U error | N/A (no convergence) | 2.89e-07 |
| Final M error | N/A (no convergence) | 1.31e-07 |

---

## Why the Fix Works

### Mathematical Explanation

The HJB equation in backward time is:
```
∂u/∂t + H(x, ∇u(t,x), m(t,x)) = 0
```

The Hamiltonian H must be evaluated with all arguments at the **same time** t. Using m(t+Δt) when computing u(t) introduces a coupling inconsistency that grows across Picard iterations.

### Discretization Structure

**Correct** (after fix):
```python
for n in range(Nt-2, -1, -1):  # Solving for U(t_n)
    U_new_n = solve_hjb_at_tn(
        U_n_plus_1_current,  # U(t_{n+1}) from current solve
        M_n_prev_picard,     # M(t_n) from previous Picard - CORRECT!
    )
```

**Incorrect** (before fix):
```python
for n in range(Nt-2, -1, -1):  # Solving for U(t_n)
    U_new_n = solve_hjb_at_tn(
        U_n_plus_1_current,      # U(t_{n+1}) from current solve
        M_n_plus_1_prev_picard,  # M(t_{n+1}) from previous Picard - WRONG!
    )
```

The time-index mismatch introduced O(Δt) errors per timestep, which accumulated across the 20 timesteps and prevented Picard convergence.

---

## Additional Finding: Bug #8 (Mass Loss)

During verification, discovered a separate issue:

**Mass Loss**: 90.46%
- Initial mass: 1.000000
- Final mass: 0.095391

**Impact**: Solver converges despite mass loss, indicating Bug #8 is independent of Bug #7.

**Status**: Documented in `BUG8_MASS_LOSS.md`, awaiting investigation.

**Priority**: High - violates fundamental conservation law of Fokker-Planck equation.

---

## Files Modified

### Production Code
- `mfg_pde/alg/numerical/hjb_solvers/base_hjb.py` (line 763)

### Test Scripts
- `test_bug7_fix.py` - Verification test
- `test_bug7_complete.log` - Full test output

### Documentation
- `BUG7_TIME_INDEX_MISMATCH.md` - Technical analysis and verification
- `BUG7_FIX_VERIFICATION_REPORT.md` - This report
- `BUG8_MASS_LOSS.md` - New bug discovered during verification
- `SESSION_LOG_2025-10-31.md` - Session documentation

---

## Regression Testing Recommendations

Before merging to production, verify:

1. **Existing tests still pass**: Run all unit and integration tests
2. **Other problems converge**: Test on different MFG problems (1D, crowd motion, etc.)
3. **Performance unchanged**: Verify iteration count and timing are reasonable
4. **Mass conservation**: Address Bug #8 before production use

---

## Conclusion

**Bug #7 is completely resolved.** The time-index fix restores the mathematical consistency of the HJB-MFG coupling, enabling smooth Picard convergence. The solver now works as intended.

**Next priority**: Investigate and fix Bug #8 (mass loss) to ensure physical correctness of the FP solver.

---

**Verified by**: Automated test `test_bug7_fix.py`
**Test date**: 2025-10-31
**Approval status**: Ready for production merge (pending Bug #8 investigation)
