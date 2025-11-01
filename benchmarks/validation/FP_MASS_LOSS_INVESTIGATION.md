# FP Solver Mass Loss Investigation

**Date**: 2025-10-31
**Status**: 🔴 CRITICAL BUG - Under Investigation
**Severity**: Catastrophic - 99.87% mass loss during solve

---

## Problem Summary

After fixing the initial density normalization bug, verification revealed a **separate, critical bug** in the FP (Fokker-Planck) solver:

- **Initial mass** (t=0): 1.0 ± 0.00% ✅ (fixed by Bug #1)
- **Final mass** (t=T): ~0.0013 (99.87% loss) ❌
- **Expected**: 1-2% mass conservation error for FDM dimensional splitting
- **Observed**: 99.87% mass loss

---

## Evidence

### Verification Output

```bash
$ python -m benchmarks.validation.verify_mass_fix

Testing 8×8 grid (original failure case)...
Grid spacing: dx = 0.142857, dy = 0.142857
Volume element: dV = 0.020408

Initial density:
  sum(m) = 49.000000
  ∫∫ m dx dy = 1.000000
  Expected: 1.0
  Error: 0.00%

✅ Initial mass conservation: PASS

----------------------------------------------------------------------
Running short solve to check final mass conservation...

After solve:
  Converged: False
  Iterations: 100
  Final mass error: 99.87%
  Expected: < 5% for FDM dimensional splitting

❌ Final mass conservation: FAIL (error = 99.87%)
```

### Solver Behavior

- **Convergence**: Failed (did not converge in 100 iterations)
- **Error metrics**: U_err and M_err remain high throughout
- **Picard iterations**: 100/100 (hit max iterations)
- **Time stepping**: 10 timesteps

---

## Hypotheses

### Hypothesis 1: Boundary Condition Issue
The FP solver may be losing mass at boundaries due to:
- Incorrect Neumann boundary conditions
- Missing flux conservation at boundaries
- Dimensional splitting boundary coupling errors

### Hypothesis 2: Time Discretization Error
Strang splitting may have implementation errors:
- Incorrect operator ordering
- Flux calculation errors between 1D sweeps
- Boundary data passing between x and y sweeps

### Hypothesis 3: Grid Scaling Issue
The volume element (dV) may not be properly accounted for in:
- FP time-stepping updates
- Flux calculations
- Diffusion operator discretization

### Hypothesis 4: Initial Condition Propagation
The corrected initial density may not be properly used:
- Solver may reinitialize density internally
- Grid conversion issues between problem definition and solver
- Mismatch between analytical initial_density() and numerical initialization

---

## Investigation Plan

### Step 1: Check Mass at Each Timestep
**Goal**: Determine when mass loss occurs

```python
# Modify verify_mass_fix.py to track mass at EVERY timestep
for t_idx in range(len(result.M)):
    mass_t = compute_total_mass_2d(result.M[t_idx], dx, dy)
    print(f"t={t_idx}: mass = {mass_t:.6e}")
```

**Expected outcome**:
- If mass drops suddenly at t=0→1: Initial condition propagation issue
- If mass decreases gradually: Time-stepping accumulation error
- If mass drops at specific timesteps: Operator splitting artifact

### Step 2: Check 1D FP Solver Mass Conservation
**Goal**: Isolate whether bug is in 2D splitting or 1D FP solver

```python
# Test 1D FP solver independently
from mfg_pde.alg.numerical.fp_solvers.fdm_fp_1d import FDMFP1D

# Run 1D test with known analytical solution
# Check if 1D solver conserves mass
```

### Step 3: Inspect FP Solver Implementation
**Files to review**:
- `mfg_pde/alg/numerical/fp_solvers/fdm_fp_nD.py`
- `mfg_pde/alg/numerical/fp_solvers/fdm_fp_2d.py`
- Focus on:
  - `_step_fp()` method
  - Dimensional splitting logic
  - Boundary condition enforcement
  - Flux calculations

### Step 4: Compare with Known-Good Solver
**Goal**: Verify if this is a new regression or existing bug

- Check if particle collocation solvers (which naturally conserve mass) work correctly
- Run same problem with GFDM (if available)
- Check historical commits for when FDM validation was last passing

### Step 5: Add Detailed Logging
**Goal**: Trace mass through the solve

```python
# Add debug logging to FP solver
def _step_fp(self, M_prev, U, t_idx):
    mass_before = compute_mass(M_prev, self.grid.spacing)
    M_next = self._apply_fp_step(M_prev, U, t_idx)
    mass_after = compute_mass(M_next, self.grid.spacing)

    if abs(mass_after - mass_before) / mass_before > 0.01:
        logger.warning(f"Mass loss in FP step: {mass_before:.6e} → {mass_after:.6e}")

    return M_next
```

---

## Expected Behavior

### Normal FDM Mass Conservation

For FDM with dimensional splitting on 2D problems:

**Theoretical error sources**:
1. **Operator splitting**: O(Δt²) error from Strang splitting → ~0.1-0.5% mass error
2. **Spatial discretization**: O(h²) truncation error → ~0.5-1% mass error
3. **Boundary flux**: Small errors at boundaries → ~0.1-0.5% mass error

**Total expected**: 1-2% mass conservation error (NOT 99.87%)

### Acceptable Mass Conservation

- **FDM**: 0.5-2% error
- **GFDM**: 0.1-1% error (better boundary handling)
- **Particle methods**: 0.01-0.1% error (naturally conservative)

### Red Flags (Indicating Bugs)

- ❌ **>10% mass loss**: Critical bug in solver
- ❌ **>50% mass loss**: Catastrophic bug or wrong solver configuration
- ❌ **>99% mass loss**: Almost certainly a fundamental implementation error

---

## Related Files

- `benchmarks/validation/test_2d_crowd_motion.py` - Test that revealed bug
- `benchmarks/validation/verify_mass_fix.py` - Verification script
- `benchmarks/validation/MASS_CONSERVATION_BUG_FIX.md` - Bug #1 (initial density) documentation
- `mfg_pde/alg/numerical/fp_solvers/` - FP solver implementations

---

## Next Steps

1. Run Step 1 (track mass at each timestep)
2. Based on results, pursue relevant hypothesis
3. Fix identified bug in FP solver
4. Re-run verification to confirm fix
5. Add regression test to prevent recurrence

---

**Investigating**: Claude Code
**Priority**: P0 (Blocking validation test suite)
**Created**: 2025-10-31
