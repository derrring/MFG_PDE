# Session Log: 2025-10-31 - WENO Grid Fix & Bug #5 Validation Attempt

## Summary

Continued investigation into MFG solver non-convergence. Fixed WENO solver grid compatibility (Bug #6). Attempted to validate upwind hypothesis using WENO solver - failed due to separate numerical issues.

---

## Bugs Found & Fixed

### Bug #6: WENO Grid Incompatibility with GridBasedMFGProblem [FIXED]

**Problem**: WENO solver crashed with shape error (65×65 vs 16×16)

**Root Cause**:
- WENO expected `problem.Nx/Ny` attributes; GridBasedMFGProblem stores in `geometry.grid.num_points`
- Fell back to default 64+1=65 when attributes missing
- Used `self.problem.Dx` which doesn't exist

**Fix Applied**:
- `hjb_weno.py:_setup_dimensional_grid()` - Added GridBasedMFGProblem detection
- Extract grid size from `geometry.grid.num_points`
- Changed `self.problem.Dx` → `self.Dx` (use solver's attribute)

**Files Modified**:
- `mfg_pde/alg/numerical/hjb_solvers/hjb_weno.py` (lines 199-203, 210-220, 486)

**Documentation**: `BUG6_WENO_GRID_COMPATIBILITY.md`

---

## Investigation Results

### WENO Validation Attempt (Bug #5)

**Goal**: Test if WENO solver (with upwind) converges, validating Bug #5 hypothesis

**Result**: INCONCLUSIVE
- Grid compatibility fixed - WENO now runs
- Numerical overflow from iteration 1: `U_err=nan`
- Overflow in WENO weight calculations, singular FP matrices
- Cannot use WENO to confirm upwind hypothesis

**Conclusion**:
- Bug #5 (central vs upwind in FDM) remains most likely root cause
- Based on theory + code evidence + symptom match
- Next step: Implement upwind directly in FDM (don't wait for WENO fix)

---

## Files Created/Modified

### Created:
1. `BUG6_WENO_GRID_COMPATIBILITY.md` - Technical analysis of grid fix
2. `test_weno_solver.py` - WENO validation test script
3. `test_weno_final.log` - Test output showing NaN overflow
4. `SESSION_LOG_2025-10-31.md` - This log

### Modified:
1. `mfg_pde/alg/numerical/hjb_solvers/hjb_weno.py` - Grid compatibility fix
2. `BUG5_CENTRAL_VS_UPWIND.md` - Added WENO validation results

---

## Naming Convention Note

User noted that `dx`, `Dx` should be renamed to English names per naming standards.

**Status**: Partially addressed
- `fixed_point_iterator.py` and `solver_factory.py` updated with English names
- WENO solver still uses `Dx`, `Dy`, `Nx`, `Ny` internally
- Broader refactoring needed (separate from bug fixing)

See `docs/NAMING_CONVENTIONS.md` for standards.

---

## Key Findings

### Bug Status Summary

| Bug # | Issue | Status | Evidence |
|-------|-------|--------|----------|
| #1    | Initial density normalization | FIXED | Mass = 1.0 ✓ |
| #2    | FP boundary padding | FIXED | No out-of-bounds ✓ |
| #3    | FP timestep (dt vs dt/2ndim) | FIXED | Adapter receives correct dt ✓ |
| #4    | Naming conventions | PARTIAL | Core files updated |
| #5    | HJB central vs upwind | **ROOT CAUSE** | Code + theory confirm |
| #6    | WENO grid compatibility | FIXED | Shape errors resolved ✓ |

### Remaining Issues

1. **HJB FDM non-convergence** (Bug #5)
   - FDM uses central difference instead of upwind
   - Causes oscillations (U_err: 0.6 ↔ 1.2)
   - Independent of damping (tested 0.4, 0.6, 0.75)
   - **Action**: Implement upwind in `base_hjb.py:_compute_1d_derivatives()`

2. **WENO numerical instability**
   - Separate from Bug #5
   - Overflow in weight calculations
   - Singular matrices in FP solver
   - **Action**: Requires separate investigation (CFL, parameters, Hamiltonian eval)

---

## Upwind Implementation Attempt (Bug #5) - INCOMPLETE

**Date**: 2025-10-31 (continued session)

**Actions Taken**:
1. Completed naming convention migration
   - Added `migrate_naming.py` script
   - Replaced 69 occurrences of `Nx/Ny/Nz/Dx/Dy/Dz` with English names
   - Added backward compatibility aliases to both HJB and FP adapters

2. Implemented "Lax-Friedrichs upwind" in `base_hjb.py:_calculate_derivatives()`
   - Added `upwind_viscosity` parameter (default 0.5)
   - Formula: `p = (p_forward + p_backward)/2 - alpha * (u_ip1 - 2*u_i + u_im1)/Dx`

**Results**: Implementation compiles and runs but **does not fix convergence**
- Iterations still oscillate: U_err ranges from 0.4 to 3.3
- Each iteration takes ~8.7 seconds (16×16 grid)
- Test timeout after 19 iterations without convergence

**Analysis**: The approach is flawed:
- Adding viscosity ≠ proper upwind discretization
- True upwind for HJB requires choosing p_forward vs p_backward based on characteristic direction
- Characteristic direction determined by Hamiltonian (not available in `_calculate_derivatives`)
- Current implementation just smooths the derivative, doesn't respect wave propagation

**Next Actions**:
1. Revert incorrect "Lax-Friedrichs" implementation - DONE
2. Research proper upwind for HJB with MFG coupling - DONE
3. Implement Godunov-type upwind - DONE

---

## Godunov Upwind Implementation (Continued Session)

**Date**: 2025-10-31 (continued)

**Implementation**:
- Added `upwind` boolean parameter to `_calculate_derivatives()` in `base_hjb.py`
- Implemented Godunov upwind: choose forward/backward difference based on sign of gradient
- Updated all residual and Jacobian computations to use `upwind=True`
- Added backward compatibility aliases to HJB and FP adapters

**Testing**:
- Problem: 16×16 grid, T=0.4, 20 timesteps
- Max iterations: 100, damping=0.6
- Test ran for 25 iterations (~3 minutes) before timeout

**Results**: **NEGATIVE - Upwind Does NOT Fix Convergence**

Errors still oscillate after upwind implementation:
- U_err: 0.17 ↔ 0.28 (no monotonic decrease)
- M_err: 0.6 ↔ 1.3 (large fluctuations)
- Same oscillatory behavior as central difference

**Conclusion**:
- Bug #5 hypothesis **REJECTED** - upwind discretization is not the root cause
- Problem is elsewhere: FP solver, MFG coupling, damping, or Newton convergence
- Upwind implementation is correct and theoretically sound, but doesn't address the actual issue

**Files Modified**:
- `mfg_pde/alg/numerical/hjb_solvers/base_hjb.py` - Added upwind parameter and logic
- `mfg_pde/alg/numerical/hjb_solvers/hjb_fdm_multid.py` - Backward compatibility
- `mfg_pde/alg/numerical/fp_solvers/fp_fdm_multid.py` - Backward compatibility
- `BUG5_UPWIND_ARCHITECTURE_ANALYSIS.md` - Documented negative test results

**Documentation Created**:
- Test results and analysis added to architecture document
- Next investigation priorities identified

---

---

## Bug #7: Time-Index Mismatch in HJB-MFG Coupling (DISCOVERED)

**Date**: 2025-10-31 (continued session, after Bug #5 rejection)

**Discovery Process**:
- After upwind and damping tests failed, investigated MFG coupling interface
- Traced M values passed to Hamiltonian in HJB solver
- Found `base_hjb.py:763` uses `M_density_from_prev_picard[n_idx_hjb + 1, :]`
- This is M(t_{n+1}) when solving HJB at time t_n!

**Problem**:
When solving HJB at time `t_n`, the code uses `M(t_{n+1})` instead of `M(t_n)`.

**Mathematical Background**:
```
HJB: ∂u/∂t + H(x, ∇u(t,x), m(t,x)) = 0
```
The Hamiltonian at time t should use m at the SAME time, not t+Δt.

**Impact**:
If M changes significantly between timesteps, this mismatch introduces errors that can:
- Destabilize Picard iterations
- Cause U_err and M_err oscillations
- Prevent convergence

**Files Modified**:
- None yet (fix proposed, awaiting validation)

**Documentation**:
- `BUG7_TIME_INDEX_MISMATCH.md` - Full technical analysis and proposed fix

**Proposed Fix**:
Change line 763 from:
```python
M_n_plus_1_prev_picard = M_density_from_prev_picard[n_idx_hjb + 1, :]
```
to:
```python
M_n_prev_picard = M_density_from_prev_picard[n_idx_hjb, :]
```

**Status**: ✓ FIXED AND VERIFIED

**Fix Applied**: Changed line 763 from:
```python
M_n_plus_1_prev_picard = M_density_from_prev_picard[n_idx_hjb + 1, :]
```
to:
```python
M_n_prev_picard = M_density_from_prev_picard[n_idx_hjb, :]
```

**Verification Results**:
- 16×16 grid, 20 timesteps, damping=0.6
- **Converged in 18 iterations** (previously: infinite oscillation)
- Smooth monotonic convergence: 0.945 → 0.538 → 0.248 → ... → 2.89e-07
- Final errors: U_err=2.89e-07, M_err=1.31e-07

**Conclusion**: Bug #7 was the root cause of Picard oscillations. Fix completely resolves convergence issue.

---

## Bug #8: Mass Loss During Evolution (DISCOVERED)

**Date**: 2025-10-31 (during Bug #7 verification)

**Problem**: FP solver loses 90% of mass during time evolution
- Initial mass: 1.0
- Final mass: 0.095
- Loss: 90.46%

**Impact**: Separate from convergence issue - solver converges despite mass loss

**Status**: Identified, awaiting investigation

**Notes**:
- FP equation is a conservation law (∂m/∂t + ∇·J = 0)
- Mass should be preserved to machine precision
- Likely issues: boundary conditions, FDM discretization, or flux formulation

---

**Session Duration**: ~4 hours
**Files Modified**: 5
**Files Created**: 8 (added BUG7 docs, diagnostics, and verification tests)
**Bugs Fixed**: 2
  - Bug #6 (WENO grid compatibility) - FIXED
  - Bug #7 (time-index mismatch) - FIXED AND VERIFIED
**Bugs Identified**: 2
  - Bug #5 (upwind) - hypothesis REJECTED after testing (not root cause)
  - Bug #8 (mass loss) - DISCOVERED during verification, needs investigation
**Key Achievement**: Bug #7 fix completely resolves MFG convergence failure
