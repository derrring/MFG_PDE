# Bug #8 Investigation Status: Incomplete

**Date**: 2025-10-31
**Status**: Partially Investigated - 1D Fixed, 2D Issue Remains
**Priority**: High (violates conservation law)
**Files Modified**: `mfg_pde/alg/numerical/fp_solvers/fp_fdm.py:251-321`

---

## Executive Summary

Bug #8 investigation revealed mass loss during FP evolution. A boundary discretization fix was implemented that **completely resolves the issue in 1D tests** (mass conserved to machine precision), but **2D MFG simulations still lose ~72% of mass**. This indicates the bug is not primarily in the 1D FP boundary discretization but rather in:
1. Initial density normalization (Bug #1)
2. 2D dimensional splitting implementation
3. MFG coupling between iterations

---

## What Was Fixed

### 1D FP Solver Boundaries

**Location**: `fp_fdm.py:251-321`

**Change**: Implemented diffusion-only boundary discretization for no-flux conditions

```python
# Before (buggy): Included advection terms at boundaries with incorrect row sums
# After (fixed): Diffusion-only at boundaries ensures row sum = 1/Δt

if i == 0:
    # Left boundary: diffusion-only
    val_A_ii = 1.0 / Dt + sigma**2 / Dx**2
    val_A_i_ip1 = -(sigma**2) / Dx**2
    # Row sum: (1/Dt + σ²/Δx²) - σ²/Δx² = 1/Dt ✓ Conservative
```

**Verification**:
- 1D test (uniform density, zero velocity): **0.000000% mass change** ✓
- 1D test (uniform density, linear U field): **0.000000% mass change** ✓

**Note**: This is a **workaround solution** that removes advection terms entirely at boundaries. While it conserves mass perfectly in 1D, it may not be the theoretically correct approach (see "Theoretical Issues" below).

---

## What Remains Broken

### 2D MFG Solver

**Test**: `trace_mass_loss.py` (8×8 grid, 10 timesteps, full MFG problem)

**Result**: **72.16% mass loss** after 10 timesteps
- Initial mass: 1.000000
- Final mass: 0.278404
- Loss rate: ~12.6% per timestep (constant)

**Convergence**: Solver converges in 18 iterations (Bug #7 fix works)

**Conclusion**: The 1D FP solver boundaries are not the primary cause of mass loss in 2D.

---

## Investigation Timeline

### Phase 1: Discovery
- Observed 90% mass loss during Bug #7 verification
- Created `trace_mass_loss.py` diagnostic

### Phase 2: Matrix Analysis
- Created `analyze_fp_matrix.py` to check conservation property
- Found boundary rows with incorrect sums (116.67 instead of 100.0)
- 16.67% error in old discretization

### Phase 3: Fix Implementation
- Implemented diffusion-only boundary fix in `fp_fdm.py`
- Verified mass conservation in 1D tests

### Phase 4: 2D Testing
- **Unexpected Result**: 2D MFG solver still loses 72% mass
- **Conclusion**: Bug is elsewhere

---

## Possible Root Causes for 2D Mass Loss

### 1. Initial Density Normalization (Bug #1)

**Status**: Unknown

The initial density may not integrate to 1.0 correctly. Need to check:
```python
# In problem.initial_density()
m_init = ...
mass_init = np.sum(m_init) * dx * dy
# Is mass_init == 1.0?
```

### 2. Dimensional Splitting

**Location**: `fp_fdm_multid.py:200-284`

The 2D FP solver uses operator splitting (Strang splitting):
1. Solve 1D FP in x-direction
2. Solve 1D FP in y-direction

**Potential Issues**:
- Boundary handling during splitting (how are 2D no-flux conditions enforced on 1D slices?)
- Order of operations
- How 1D solutions are combined

**Needs Investigation**:
```python
# fp_fdm_multid.py:256-283
solver_1d = fp_fdm.FPFDMSolver(...)
M_solution_1d = solver_1d.solve_fp_system(...)
```

### 3. Interior Advection Discretization

**Location**: `fp_fdm.py:297-321` (interior points)

The upwind flux splitting may have issues:
```python
val_A_ii += float(
    coefCT * (npart(u_at_tk[i + 1] - u_at_tk[i]) +
              ppart(u_at_tk[i] - u_at_tk[i - 1])) / Dx**2
)
```

**Test Needed**: Check if interior discretization conserves mass in isolation.

### 4. MFG Coupling

**Location**: MFG solver iteration loop

Mass might be lost during:
- Interpolation between HJB and FP grids
- Damping/relaxation steps
- Convergence checks

---

## Theoretical Issues with Current Fix

### Diffusion-Only Boundary Approach

**What It Does**:
- Removes all advection terms at boundaries
- Only includes diffusion contribution
- Ensures row sum = 1/Δt for mass conservation

**Problems**:
1. **Physically Incorrect**: Ignores advective flux at boundaries
2. **Inconsistent**: Interior uses advection+diffusion, boundaries use diffusion-only
3. **Accuracy**: May introduce O(1) errors near boundaries (not just O(Δx))

### Correct Approach: Ghost Cell Substitution

Following the user's guidance, the **correct implementation** for implicit FD with ghost cells should:

For no-flux at i=0 with `m[-1] = m[1]`:

```python
# Substitute m[-1] = m[1] in the interior stencil
# This modifies the matrix coefficients directly

# Diffusion term: σ²/2 * (m[-1] - 2m[0] + m[1])/Δx²
#              → σ²/2 * (m[1] - 2m[0] + m[1])/Δx²
#              = σ² * (m[1] - m[0])/Δx²

# Advection upwind terms involving m[-1]:
#   - Backward flux: couples to m[-1] → substitute m[1]
#   - Forward flux: couples to m[1]
# → Combine both contributions to m[1]

# Result:
# A[0, 0] = 1/Δt + (combined diagonal terms)
# A[0, 1] = (combined coupling to m[1])
```

**This approach**:
- Uses the full interior stencil (advection + diffusion)
- Applies algebraic substitution for ghost cells
- Should conserve mass while maintaining accuracy

**Not Implemented**: This requires rework of the boundary discretization logic.

---

## Diagnostic Files Created

### Scripts

- `trace_mass_loss.py` - Timestep-by-timestep mass tracking
- `plot_mass_loss.py` - Visualization of mass evolution
- `analyze_fp_matrix.py` - Matrix conservation property check

### Data Files

- `bug8_mass_loss_data.npz` - Full simulation data (times, masses, densities)
- `bug8_mass_loss_plots.png` - Mass evolution plots
- `trace_mass_loss.log` - Detailed output
- `analyze_fp_matrix.log` - Matrix analysis results

### Documentation

- `BUG8_MASS_LOSS.md` - Initial bug report
- `BUG8_INVESTIGATION_SUMMARY.md` - Complete investigation report
- `BUG8_STATUS_INCOMPLETE.md` - This file

---

## Recommendations

### Short Term (Current Session)

1. **Accept Current Fix**: The diffusion-only boundary approach conserves mass in 1D and is a significant improvement over the original discretization.

2. **Mark Bug #8 as Partial**: Document that 1D is fixed but 2D mass loss remains.

3. **Focus on Other Bugs**: Move to Bug #7 verification or other high-priority items.

### Long Term (Future Work)

1. **Investigate Bug #1**: Check if initial density normalization is the root cause of 2D mass loss.

2. **Implement Proper Ghost Cell Method**: Rewrite boundary discretization following the correct ghost cell substitution approach for implicit FD methods.

3. **Test Dimensional Splitting**: Create isolated test for 2D FP solver (without MFG coupling) to verify dimensional splitting conserves mass.

4. **Audit Interior Discretization**: Verify that interior upwind advection terms satisfy the conservation property.

---

## Related Bugs

- **Bug #7** (time-index mismatch): FIXED - Enabled convergence
- **Bug #1** (initial density): Unknown status - May be causing 2D mass loss
- **Bug #5** (HJB upwind): Fixed independently

---

## Next Session TODO

1. Check Bug #1 status (initial density normalization)
2. Test 2D FP solver in isolation (without MFG coupling)
3. Consider implementing proper ghost cell substitution method
4. Profile where mass is being lost in 2D (boundary vs interior vs splitting)

---

**Conclusion**: Bug #8 investigation successfully fixed 1D FP mass conservation but revealed a deeper issue in 2D. The current fix is a workaround that works but may not be theoretically optimal. Further investigation needed to resolve 2D mass loss.
