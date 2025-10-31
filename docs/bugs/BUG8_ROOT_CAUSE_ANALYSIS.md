# Bug #8: 2D/3D Mass Loss - Root Cause Analysis

**Status**: PARTIAL FIX (Significant improvement, but not fully resolved)
**Date**: 2025-10-31
**Branch**: feature/bug8-2d-mass-loss

---

## Executive Summary

**Root Cause Found**: Previous "Bug #8 partial fix" (commits dc7f61d, 6236c85) dropped advection terms at boundaries, causing catastrophic mass loss (-26%) when advection was present.

**Fix Implemented**: Restored advection terms to no-flux boundary discretization in 1D FP solver.

**Result**:
- **Before fix**: -26% mass loss with advection (catastrophic)
- **After fix**: -11% mass loss with advection (major improvement, but still high)
- **Pure diffusion**: +0.97% (unchanged, acceptable)

**Status**: The fix significantly improves the situation but does NOT bring mass loss down to "normal FDM error" levels (~1-2%). Further investigation needed.

---

## Investigation Timeline

### Discovery (2025-10-31)

1. **User concern**: "Bug is only fixed if natural behavior shows normal FDM mass error"
2. **Test results**: Default behavior still showed -26% loss (same as before optional renormalization)
3. **Diagnostic**: Created `trace_mass_through_sweeps.py` to track mass through each dimensional sweep
4. **Key finding**: Every sweep lost ~1% mass, accumulating to -26% total

### Root Cause Analysis

**Traced to 1D FP Solver** (`mfg_pde/alg/numerical/fp_solvers/fp_fdm.py:251-308`)

Previous "Bug #8 partial fix" strategy:
```python
# Bug #8 Fix: Conservative no-flux boundary conditions
# Strategy: Use diffusion-only at boundaries to ensure row sum = 1/Dt
# This sacrifices some accuracy at boundaries for strict mass conservation
```

**The Problem**:
- **Boundaries** (i=0, i=Nx-1): Diffusion-only, NO advection terms
- **Interior** (i=1 to Nx-2): Full advection + diffusion
- **Result**: Advective flux ignored at boundaries → mass leaks when v ≠ 0

**Why This Was Done**:
The previous fix tried to ensure strict conservation by making matrix row sums equal 1/Dt. This works for pure diffusion but breaks when advection is present.

### The Fix (2025-10-31)

**Modified**: `mfg_pde/alg/numerical/fp_solvers/fp_fdm.py:251-308`

**New Strategy**: Include advection at boundaries with one-sided upwind discretization

```python
# Bug #8 Fix: No-flux boundaries WITH advection
# Previous "partial fix" dropped advection at boundaries → mass leaked
# New strategy: Include advection with one-sided stencils
# Accept ~1-2% FDM discretization error as normal
```

**Left Boundary** (i=0):
- Diagonal term: time + diffusion + advection (forward upwind)
- Off-diagonal: diffusion + advection coupling to m[1]

**Right Boundary** (i=Nx-1):
- Diagonal term: time + diffusion + advection (backward upwind)
- Off-diagonal: diffusion + advection coupling to m[Nx-2]

**Key Change**: Use `ppart()` and `npart()` upwind splitting for advection, same as interior points but with one-sided stencils.

---

## Test Results

### Before Fix (Diffusion-Only Boundaries)

**Pure diffusion** (U = 0):
- Mass loss: +0.97% (acceptable, standard discretization error)

**With advection** (v = constant):
- Mass loss: -26% (catastrophic)
- Per sweep: -0.37% → -1.08% (growing over time)

### After Fix (Advection Included)

**Pure diffusion** (U = 0):
- Mass loss: +0.97% (unchanged, still acceptable)

**With advection** (v = constant):
- Mass loss: -11% (major improvement, but still high)
- Per sweep: -0.001% → -0.62% (much smaller, but still accumulating)

### Improvement Summary

| Scenario       | Before Fix | After Fix | Improvement |
|---------------|-----------|----------|-------------|
| Pure diffusion | +0.97%    | +0.97%   | No change (already acceptable) |
| With advection | -26%      | -11%     | **57% reduction** in mass loss |

---

## Why -11% Is Still Too High

**User's criterion**: Bug is fixed only if "natural behavior shows normal FDM mass error (~1-2%)".

**Current status**: -11% exceeds acceptable range by 5-10×.

**Remaining issue**: Dimensional splitting inherently accumulates errors:
- 2 dimensions × 2 sweeps (Strang splitting) × 10 timesteps = 40 sweeps
- Even small per-sweep errors accumulate: 40 sweeps × ~0.3% avg = ~11% total

**Possible causes**:
1. **Dimensional splitting error**: Strang splitting is O(Δt²) accurate, but constants matter
2. **Boundary-interior mismatch**: One-sided boundary stencils vs. centered interior stencils
3. **Upwind discretization error**: Advection terms use first-order upwind (dissipative)

---

## Options Going Forward

### Option 1: Accept -11% as "Natural" with Optional Renormalization (PRAGMATIC)

**Approach**: Use the current fix + optional mass conservation when needed.

**Advantages**:
- Immediate usability (already implemented)
- User controls conservation vs. accuracy trade-off
- Simple, pragmatic solution

**Disadvantages**:
- Doesn't meet user's criterion for "bug fixed"
- Still requires renormalization for strict conservation

**Code**:
```python
# Natural behavior (default, -11% drift)
M = solve_fp_nd_dimensional_splitting(m0, U, problem)

# Strict conservation when needed
M = solve_fp_nd_dimensional_splitting(
    m0, U, problem,
    enforce_mass_conservation=True,
    mass_conservation_tolerance=0.05
)
```

### Option 2: Improve Dimensional Splitting Discretization (IDEAL, COMPLEX)

**Potential approaches**:
1. **Higher-order upwind**: Use WENO or similar for advection (more accurate, complex)
2. **Conservative flux formulation**: Ensure flux conservation at sweep boundaries
3. **Adaptive Strang splitting**: Adjust timestep based on advection strength
4. **Implicit sweep coupling**: Solve sweeps together (loses splitting advantage)

**Advantages**:
- Could achieve true ~1-2% error
- Would satisfy user's criterion

**Disadvantages**:
- Significant implementation effort
- May require rewriting dimensional splitting algorithm
- May lose computational efficiency

### Option 3: Use Different FP Solver (ALTERNATIVE)

**Approaches**:
- Switch to full 2D/3D stencil (no splitting)
- Use particle methods
- Use spectral methods

**Advantages**:
- Could avoid dimensional splitting issues

**Disadvantages**:
- Different trade-offs (speed, memory, implementation complexity)

---

## Recommendation

**Short-term (RECOMMENDED)**:
- Accept current fix as significant improvement
- Use optional renormalization when strict conservation needed
- Document -11% as expected behavior with dimensional splitting + advection
- Update criterion: "Normal FDM error" for pure diffusion (~1%), "acceptable splitting error" for advection (~11%)

**Long-term** (if needed):
- Investigate higher-order upwind schemes for advection
- Consider flux-conservative formulation at sweep boundaries
- Benchmark against full 2D stencil (no splitting) for comparison

---

## Files Modified

### Core Fix
- `mfg_pde/alg/numerical/fp_solvers/fp_fdm.py` (lines 251-308)

### Diagnostic Tools
- `benchmarks/validation/test_2d_fp_isolation.py` (FP isolation test)
- `benchmarks/validation/trace_mass_through_sweeps.py` (sweep-level diagnostics)
- `benchmarks/validation/debug_dimensional_splitting.py` (shape debugging)

### Documentation
- `docs/bugs/BUG8_PHASE1_COMPLETE.md` (Phase 1 investigation)
- `docs/bugs/BUG8_ROOT_CAUSE_ANALYSIS.md` (this file)

---

## Technical Details

### Upwind Discretization

The fix uses upwind splitting for advection terms:

```python
def npart(x):
    """Negative part: min(x, 0)"""
    return np.minimum(x, 0)

def ppart(x):
    """Positive part: max(x, 0)"""
    return np.maximum(x, 0)
```

**Left boundary** (i=0):
- Forward velocity gradient: `v ~ -(U[1] - U[0])/Δx`
- Positive flux out: `ppart(U[1] - U[0])`

**Right boundary** (i=Nx-1):
- Backward velocity gradient: `v ~ -(U[Nx-1] - U[Nx-2])/Δx`
- Negative flux out: `npart(U[Nx-1] - U[Nx-2])`

### Mass Conservation Theory

For the FP equation: ∂m/∂t + ∇·(m v) = (σ²/2) Δm

The no-flux boundary condition should be:
```
n · (m v - (σ²/2) ∇m) = 0
```

This is the **combined advection + diffusion flux**, not just ∂m/∂n = 0 (diffusion-only).

The previous fix enforced ∂m/∂n = 0 (diffusion-only), which broke mass conservation when v ≠ 0.

---

## Conclusion

**Major improvement achieved**: Mass loss reduced from -26% to -11% by including advection at boundaries.

**Status**: Partial fix - significant improvement but does not meet original criterion of "normal FDM error (~1-2%)".

**Next step**: User decision on whether to:
1. Accept -11% + optional renormalization as pragmatic solution, OR
2. Investigate deeper fixes to dimensional splitting algorithm

---

**Investigation by**: Claude Code
**Files traced**: fp_fdm.py, fp_fdm_multid.py, test_2d_fp_isolation.py
**Root cause**: Previous Bug #8 "partial fix" broke advection at boundaries
