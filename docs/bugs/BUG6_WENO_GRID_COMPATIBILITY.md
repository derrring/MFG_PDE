# Bug #6: WENO Solver Grid Incompatibility with GridBasedMFGProblem

**Date**: 2025-10-31
**Severity**: HIGH - Blocks WENO solver use with GridBasedMFGProblem
**Status**: FIXED

---

## Problem

WENO solver crashed with shape mismatch when used with `GridBasedMFGProblem`:
```
ValueError: could not broadcast input array from shape (16,16) into shape (65,65)
```

Expected 16×16 grid, but WENO solver defaulted to 65×65.

---

## Root Cause

**File**: `mfg_pde/alg/numerical/hjb_solvers/hjb_weno.py`
**Function**: `_setup_dimensional_grid()` (lines 192-225)

### Issue 1: Grid Dimension Extraction

WENO solver tried to get grid size from `problem.Nx` or `problem.nx`:
```python
# Old code (line 212-213)
self.Nx = getattr(self.problem, "Nx", getattr(self.problem, "nx", 64)) + 1
self.Ny = getattr(self.problem, "Ny", getattr(self.problem, "ny", 64)) + 1
```

But `GridBasedMFGProblem` stores grid info in `geometry.grid.num_points`, not as attributes.
When attributes missing → defaulted to 64+1=65 grid points.

### Issue 2: Grid Spacing Access

WENO solver used `self.problem.Dx` directly in computations (line 486):
```python
# Old code
dx = self.problem.Dx  # Doesn't exist for GridBasedMFGProblem!
```

---

## Fix Applied

### 1. Updated Grid Initialization (2D case)

**File**: `hjb_weno.py`, lines 210-220

```python
elif hasattr(self.problem, "geometry") and hasattr(self.problem.geometry, "grid"):
    # GridBasedMFGProblem: num_points already includes all grid points
    grid_obj = self.problem.geometry.grid
    self.Nx, self.Ny = grid_obj.num_points[0], grid_obj.num_points[1]
    self.Dx, self.Dy = grid_obj.spacing[0], grid_obj.spacing[1]
```

### 2. Updated Grid Initialization (1D case)

**File**: `hjb_weno.py`, lines 199-203

```python
elif hasattr(self.problem, "geometry") and hasattr(self.problem.geometry, "grid"):
    # GridBasedMFGProblem: num_points already includes all grid points
    grid_obj = self.problem.geometry.grid
    self.Nx = grid_obj.num_points[0]
    self.Dx = grid_obj.spacing[0]
```

### 3. Use Solver's Grid Spacing

**File**: `hjb_weno.py`, line 486

```python
# Changed from:
dx = self.problem.Dx

# To:
dx = self.Dx  # Use attribute set in _setup_dimensional_grid()
```

---

## Verification

Before fix:
```
ValueError: could not broadcast input array from shape (16,16) into shape (65,65)
```

After fix:
```
Running WENO solver...
MFG Picard:   3%|▎  | 1/30 [00:00<00:16, 1.81iter/s]
```

WENO solver now runs without shape errors.

---

## Related Issues

### WENO Numerical Instability

After fixing grid compatibility, WENO solver exhibits numerical overflow:
```
U_err=nan, M_err=1.50e+00
RuntimeWarning: overflow encountered in square
RuntimeWarning: invalid value encountered in divide
```

This is a **separate issue** (not grid-related):
- Likely CFL violation, incorrect parameters, or Hamiltonian evaluation error
- Prevents using WENO to validate Bug #5 (upwind hypothesis)
- Requires separate investigation

---

## Files Modified

1. `mfg_pde/alg/numerical/hjb_solvers/hjb_weno.py`
   - Lines 199-203: 1D grid initialization
   - Lines 210-220: 2D grid initialization
   - Line 486: Grid spacing access

---

## Impact

**Solvers Affected**:
- ✓ HJBWenoSolver - Now compatible with GridBasedMFGProblem

**Benefits**:
- GridBasedMFGProblem can now use WENO solver (once numerical issues resolved)
- Consistent grid convention handling across problem types

---

**Last Updated**: 2025-10-31
**Related**: See `BUG5_CENTRAL_VS_UPWIND.md` for HJB FDM convergence issue
