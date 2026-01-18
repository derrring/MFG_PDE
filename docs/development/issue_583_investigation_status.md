# Issue #583: HJB Semi-Lagrangian Cubic Interpolation Investigation

**Status**: Root cause identified, partial fix implemented, requires further debugging
**Priority**: Low
**Date**: 2026-01-18

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

## Recommendation

**Short Term** (for v0.17.x):
- Document PCHIP fix as partial solution
- Keep tests as `@pytest.mark.xfail` until fully resolved
- Note in release notes that cubic interpolation has known issues

**Long Term** (for v0.18.0+):
1. **Deep dive**: Trace solver execution to find why PCHIP doesn't resolve overflow
2. **Consider**: Default to linear interpolation for Semi-Lagrangian HJB
3. **Alternative**: Implement adaptive interpolation (linear near discontinuities, cubic in smooth regions)
4. **Research**: Review Semi-Lagrangian literature for standard practice

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

**Last Updated**: 2026-01-18
**Investigation Time**: ~3 hours
**Status**: Requires additional debugging beyond priority: low scope
