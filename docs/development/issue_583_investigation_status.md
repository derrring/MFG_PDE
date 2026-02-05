# Issue #583: HJB Semi-Lagrangian Cubic Interpolation Investigation

**Status**: Adaptive Picard damping implemented (primary fix). Cubic interpolation XFAIL tests remain.
**Priority**: Low
**Date**: 2026-02-06 (updated)

---

## Executive Summary

Identified the root cause of cubic interpolation NaN values: **Runge phenomenon** (cubic spline oscillations at discontinuities). Implemented two fixes (bounded extrapolation, then PCHIP), but issue persists due to additional complexity requiring deeper investigation.

---

## Root Cause Analysis

### Primary Issue: Runge Phenomenon

Cubic splines create severe oscillations at discontinuities common in HJB problems:

**Example** (terminal cost with sharp transition):
```
Original data range: [0.0, 10.0]
Cubic spline interpolation: [-1.08, 11.08]  ← 10% overshoot!
```

These oscillations create a feedback loop:
1. Cubic interpolation → overshoots/undershoots
2. Extreme U values → extreme gradients via finite differences
3. Hamiltonian H = 0.5*|∇U|² → squares gradient → overflow
4. Next timestep: even more extreme values

**Observed gradient explosion**:
```
p = -1,520,616     → |p|² = 2.3×10¹²
p = -42,387,375    → |p|² = 1.8×10¹⁵
p = -3,572,698,879 → |p|² = 1.3×10¹⁹  (OVERFLOW)
```

### Initial Hypothesis (Incorrect)

Originally thought the issue was **cubic extrapolation** outside domain bounds. However, code analysis revealed:
- Lines 72-75 in `hjb_sl_interpolation.py` already clamp boundary queries
- Cubic interpolation never called outside [xmin, xmax]
- Problem is **interior oscillations**, not extrapolation

---

## Fixes Attempted

### Fix 1: Bounded Extrapolation (Ineffective)

**Change**: `fill_value=(U_values[0], U_values[-1])`

**Result**: Redundant - boundary clamping already exists at lines 72-75

### Fix 2: PCHIP (Monotonicity-Preserving) ✓ Partially Effective

**Change**: Replace `interp1d(..., kind="cubic")` with `PchipInterpolator`

**Validation**:
```python
# Standalone test with discontinuous data
Original data: [0.0, 10.0]
PCHIP interp:  [0.0, 10.0]  ✅ No oscillations!
```

**Issue**: Despite PCHIP working correctly in isolation, solver still fails with:
```
RuntimeWarning: overflow encountered in square
ValueError: array must not contain infs or NaNs
```

---

## Current Implementation

**File**: `mfg_pde/alg/numerical/hjb_solvers/hjb_sl_interpolation.py`

**Lines 78-85**:
```python
if method == "cubic":
    # Issue #583 fix: Use PCHIP (monotonicity-preserving cubic Hermite)
    # instead of cubic splines to prevent Runge oscillations at discontinuities
    interpolator = PchipInterpolator(
        x_grid,
        U_values,
        extrapolate=False,  # Return NaN outside bounds (handled by lines 72-75)
    )
```

**Status**: Implemented but solver still fails

---

## Outstanding Questions

1. **Why does PCHIP fix fail in solver?**
   - PCHIP works correctly in standalone tests
   - Solver still produces overflow
   - Possible causes:
     - Different code path being used?
     - Additional instability source (e.g., sigma=0.0)?
     - Python caching issue?
     - nD interpolation path also needs fixing?

2. **Is cubic interpolation appropriate for HJB Semi-Lagrangian?**
   - Most Semi-Lagrangian literature uses **linear** interpolation
   - Cubic offers higher-order accuracy but unstable for discontinuous data
   - Trade-off: accuracy vs robustness

3. **Alternative approaches?**
   - Use linear interpolation (proven robust)
   - Implement ENO/WENO reconstruction (complex)
   - Add smoothness detection (switch linear/cubic adaptively)

---

## Files Modified

1. `mfg_pde/alg/numerical/hjb_solvers/hjb_sl_interpolation.py`
   - Added `from scipy.interpolate import PchipInterpolator`
   - Replaced cubic spline with PCHIP (lines 78-85)

---

## Documentation Created

1. `/tmp/issue_583_root_cause_analysis.md` - Detailed technical analysis
2. `/tmp/test_cubic_*.py` - Debug scripts demonstrating the issue
3. This file - Investigation status summary

---

## Adaptive Picard Damping (2026-02-06)

**Primary fix implemented**: `adapt_damping()` in `fixed_point_utils.py` detects error oscillation
and dynamically reduces damping factors. This addresses the root cause — gradient amplification
during Picard iteration — rather than symptom treatment (gradient clipping).

**Usage**: `FixedPointIterator(..., adaptive_damping=True)`

**Key design decisions**:
- Pure function (no state class) — operates on error history lists
- U and M adapted independently (U gradient explosion is the primary pathology)
- Cautious recovery: after `stable_window` decreasing iterations, slowly increase damping
- Recovery never exceeds initial damping factor

**Files added/modified**:
- `mfg_pde/alg/numerical/coupling/fixed_point_utils.py` — `adapt_damping()` function
- `mfg_pde/alg/numerical/coupling/fixed_point_iterator.py` — integration + parameters
- `mfg_pde/alg/numerical/hjb_solvers/hjb_semi_lagrangian.py` — warning text update
- `tests/unit/test_alg/test_adaptive_damping.py` — 8 unit tests

**Gradient clipping**: Warning now explicitly states it is a "SAFETY NET, not a solution"
and directs users to `adaptive_damping=True`.

## Recommendation

**Cubic interpolation XFAIL tests remain**: These test standalone HJB solve (not Picard coupling),
so adaptive damping does not apply. The PCHIP fix is partial.

**Remaining work**:
1. **Consider**: Default to linear interpolation for Semi-Lagrangian HJB
2. **Alternative**: Implement adaptive interpolation (linear near discontinuities, cubic in smooth regions)
3. **Research**: Review Semi-Lagrangian literature for standard practice

---

## Test Status

All cubic interpolation tests remain `@pytest.mark.xfail`:
- `test_cubic_produces_valid_solution_1d`
- `test_cubic_consistency_with_linear`
- `test_cubic_improves_smoothness`
- `test_rk4_with_cubic_interpolation`
- `test_all_enhancements_together`
- `test_enhanced_vs_baseline_consistency`

---

## References

- **Issue**: [#583](https://github.com/derrring/MFG_PDE/issues/583)
- **Scipy Pchip**: [PchipInterpolator docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html)
- **Runge Phenomenon**: Classic issue with high-order polynomial interpolation at discontinuities

---

**Last Updated**: 2026-02-06
**Investigation Time**: ~3 hours (initial) + adaptive damping implementation
**Status**: Adaptive Picard damping implemented. Cubic interpolation XFAIL tests remain (out of scope).
