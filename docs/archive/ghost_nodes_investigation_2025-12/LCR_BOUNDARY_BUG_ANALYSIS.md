# LCR Boundary Stencil Bug Analysis Report

**Date**: 2025-12-20
**Author**: Claude Code Analysis
**Status**: Critical Bug Identified
**Affected File**: `mfg_pde/alg/numerical/hjb_solvers/hjb_gfdm.py`
**Function**: `_build_rotation_matrix()` (lines 686-756)

---

## Executive Summary

A critical bug was identified in the Local Coordinate Rotation (LCR) implementation for GFDM boundary stencils. The 2D rotation matrix construction formula is mathematically incorrect, causing **inverted gradient components** at horizontal boundaries (top/bottom walls) and corners. This explains the observed "wrong-sign" gradients that cause the crowd evacuation simulation to move at only 55% of the expected speed.

---

## 1. Problem Statement

### Observed Symptoms
- GFDM-LCR solver produces density evolution moving at ~55% of FDM baseline speed
- Center of mass displacement: FDM = 11.4, GFDM-LCR = 6.14
- 10% of grid points have wrong-sign gradients (dU/dx > 0 instead of < 0)
- Y-coordinate oscillation in density evolution (y oscillates between 4.6 and 5.4)
- Extreme gradient oscillations at boundaries: dU/dx ranges [-71.5, +81.4] vs FDM's [-10.0, -0.2]

### Affected Regions
| Region | dU/dx Mean (FDM) | dU/dx Mean (GFDM-LCR) | Issue |
|--------|------------------|------------------------|-------|
| Interior | -5.11, std=2.5 | -5.29, std=11.3 | 4.5x variance |
| Boundary | -5.11, std=3.5 | -1.73, std=14.2 | Weakened + noisy |
| **Corners** | -5.11 | **+5.90** | **Wrong sign!** |

---

## 2. Root Cause: Rotation Matrix Bug

### Location
File: `hjb_gfdm.py`, lines 710-716

### Current (Buggy) Implementation
```python
elif dim == 2:
    # 2D: Rotation matrix that maps e_x to normal
    # R = [n_x, -n_y]
    #     [n_y,  n_x]
    # where n = (n_x, n_y) is the normal
    n_x, n_y = normal
    return np.array([[n_x, n_y], [-n_y, n_x]])  # BUG: Second column is wrong
```

### Mathematical Analysis

The intended behavior is: `R @ e_x = n` (map unit x-vector to normal direction)

**Current formula**: `R = [[n_x, n_y], [-n_y, n_x]]`
- `R @ e_x = R @ [1, 0]^T = [n_x, -n_y]^T`
- This gives **(n_x, -n_y)** instead of **(n_x, n_y)**

**Verification**:
| Wall | Normal n | Current R @ e_x | Expected | Status |
|------|----------|-----------------|----------|--------|
| Left (x=0) | (-1, 0) | (-1, 0) | (-1, 0) | OK |
| Right (x=20) | (1, 0) | (1, 0) | (1, 0) | OK |
| **Bottom (y=0)** | (0, -1) | **(0, +1)** | (0, -1) | **BUG** |
| **Top (y=10)** | (0, +1) | **(0, -1)** | (0, +1) | **BUG** |
| **Corner (0,0)** | (-0.707, -0.707) | **(-0.707, +0.707)** | (-0.707, -0.707) | **BUG** |

### Why Bug Is Hidden for Left/Right Walls
When `n_y = 0` (vertical walls), the bug is invisible because `(n_x, -n_y) = (n_x, 0) = (n_x, n_y)`.

### Correct Implementation
```python
elif dim == 2:
    # 2D: Rotation matrix that maps e_x to normal
    # Standard rotation matrix with columns = new basis vectors
    # First column: where e_x maps to = n
    # Second column: where e_y maps to = n rotated 90 deg CCW = (-n_y, n_x)
    n_x, n_y = normal
    return np.array([[n_x, -n_y], [n_y, n_x]])  # CORRECT
```

---

## 3. Impact on MFG Solution

### Derivative Transformation Chain

1. **Forward rotation**: Neighbor offsets are rotated by R before Taylor expansion
2. **Derivative computation**: Taylor expansion gives derivatives in rotated frame (x', y')
3. **Backward rotation**: Derivatives are transformed back: `grad_orig = R^T @ grad_rotated`

### At Buggy Horizontal Boundaries

For bottom wall with n = (0, -1):
- Buggy R maps x-direction to **+y** instead of **-y**
- Taylor expansion computes du/dx' thinking x' points **up** instead of **down**
- Result: Normal derivative has **inverted sign**

### Physical Consequence

In MFG crowd evacuation:
- Optimal control: `alpha* = -grad(U)`
- At bottom/top walls: gradient y-component is inverted
- Agents near horizontal walls receive **wrong vertical push**
- Creates spurious bouncing/oscillation in y-direction
- At corners: gradient is rotated ~90 degrees wrong, creating barriers

---

## 4. Code Flow Analysis

### LCR Initialization (`_apply_local_coordinate_rotation`, line 758)
```
For each boundary point i:
    1. Compute outward normal n = _compute_outward_normal(i)
    2. Build rotation R = _build_rotation_matrix(n)        <-- BUG HERE
    3. Rotate neighbor offsets: rotated = (R @ offsets.T).T
    4. Store in neighborhood["rotated_offsets"]
```

### Taylor Matrix Construction (`_build_taylor_matrices`, line 1100)
```
For LCR boundary points:
    1. Use rotated_offsets for Taylor polynomial terms
    2. Solve weighted least squares for derivative coefficients
    3. Coefficients are in rotated frame (x', y')
```

### Derivative Back-Rotation (`_compute_derivatives_at_point`, line 1598)
```
For LCR boundary points:
    1. derivatives = {(1,0): du/dx', (0,1): du/dy', ...}  # rotated frame
    2. Apply _rotate_derivatives_back(derivatives, R)
    3. grad_orig = R^T @ grad_rotated                    <-- Propagates bug
```

---

## 5. Recommended Fix

### Immediate Fix (One Line Change)

In `hjb_gfdm.py`, line 716, change:
```python
# FROM (buggy):
return np.array([[n_x, n_y], [-n_y, n_x]])

# TO (correct):
return np.array([[n_x, -n_y], [n_y, n_x]])
```

### Verification Test

Add unit test to verify rotation matrix correctness:
```python
def test_rotation_matrix_maps_ex_to_normal():
    """Verify R @ e_x = n for all boundary orientations."""
    test_normals = [
        np.array([-1, 0]),      # Left
        np.array([1, 0]),       # Right
        np.array([0, -1]),      # Bottom
        np.array([0, 1]),       # Top
        np.array([-1, -1]) / np.sqrt(2),  # Corner
    ]
    e_x = np.array([1, 0])

    for n in test_normals:
        R = solver._build_rotation_matrix(n)
        result = R @ e_x
        assert np.allclose(result, n), f"R @ e_x = {result}, expected {n}"
```

---

## 6. Additional Observations

### Secondary Issue: High Variance at Boundaries

Even with the rotation fix, boundary stencils show 4.5x higher variance than FDM:
- Interior std: 2.5 (FDM) vs 11.3 (GFDM)
- This suggests additional numerical conditioning issues

**Possible causes**:
1. Insufficient deep neighbors in stencil
2. Weight function decay too fast for boundary geometry
3. Corner stencils need special handling (avoid diagonal normal averaging)

### Recommendation for Corners

Consider alternative corner handling strategies:
1. Use axis-aligned normal (pick dominant direction)
2. Use ghost points for corner stencils
3. Exclude corners from LCR, use standard GFDM

---

## 7. Files to Update

1. **Fix**: `mfg_pde/alg/numerical/hjb_solvers/hjb_gfdm.py` line 716
2. **Test**: Add `tests/unit/test_hjb_gfdm_rotation.py`
3. **Docs**: Update GFDM documentation with LCR limitations

---

## 8. References

- Issue #531: GFDM boundary stencil degeneracy
- Related: Davydov & Oskolkov (2011), "Local stencil selection for meshless FD"
- LCR theory: Local coordinate rotation for improved boundary derivative accuracy

---

**Report End**
