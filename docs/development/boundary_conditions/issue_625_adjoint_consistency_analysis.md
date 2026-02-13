# Issue #625: Adjoint Consistency Analysis

**Status**: Analysis Complete
**Date**: 2026-01-25
**Related Issues**: #622 (Strict Adjoint Mode), #574 (Adjoint-Consistent BC)

## Executive Summary

The strict adjoint mode (Issue #622) has a fundamental flaw: **HJB's advection matrix is built in gradient form, which does not conserve mass when transposed for FP use.**

## Problem Statement

In strict adjoint mode, `solve_fp_step_adjoint_mode()` uses `A_HJB.T` directly:

```python
# fp_fdm.py:604
A_system = identity / dt + A_advection_T - D * L_matrix
```

**Expected**: Mass conservation (total mass constant over time)
**Actual**: ~60% mass drift over 100 timesteps

## Root Cause Analysis

### 1. Discretization Mismatch

| Property | HJB `build_advection_matrix` | FP `divergence_upwind` |
|----------|------------------------------|------------------------|
| **Form** | Gradient: `v·∇m` | Divergence: `∇·(vm)` |
| **Column sums** | ≠ 0 | = 1/dt |
| **Row sums of A^T** | ≠ 0 | = 1/dt |
| **Mass conservation** | NO | YES (flux telescoping) |

### 2. Mathematical Explanation

For mass conservation in the FP equation:
```
dm/dt + ∇·(vm) = D·Δm
```

The implicit discretization:
```
(I/dt + A^T - D·L) m^{n+1} = m^n / dt
```

Requires: **Row sums of (A^T - D·L) = 0** for mass conservation.

- Laplacian L with Neumann BC: row sums = 0 ✓
- A^T from HJB (gradient form): row sums ≠ 0 ✗

### 3. Boundary Analysis

From diagnostic output:
```
A_HJB^T[0, :5]:   [14.5, -13.5, 0, 0, 0]  → row sum = 1.0
A_HJB^T[-1, -5:]: [0, 0, 0, -33.5, 34.5]  → row sum = 1.0
```

The non-zero row sums at boundaries AND interior cause mass to "leak" out of the system.

## Verified Findings

### Test Results

| Configuration | Mass Drift (100 steps) |
|---------------|------------------------|
| Original A^T | -60.8% |
| Zero boundary off-diagonals | -64.5% (WORSE) |
| Stall at x=0 | -61.2% |
| Stall at x=0.5 | -60.8% |
| Stall at x=1 | -61.2% |

Simply zeroing boundary off-diagonals makes things worse because it disrupts the diagonal balance without adjusting for mass conservation.

## Recommended Solutions

### Option A: Build HJB Matrix in Divergence Form (PREFERRED)

Modify `HJBFDMSolver.build_advection_matrix()` to use face-based flux discretization matching FP's approach:

```python
# Instead of point-based velocity:
#   v_i = -coupling * grad_U[i]
#   A[i,i] += v_i/dx, A[i,i-1] -= v_i/dx

# Use face-based flux:
#   v_{i+1/2} = -coupling * (U[i+1] - U[i]) / dx
#   Flux at face: F_{i+1/2} = v_{i+1/2} * m_upwind
```

**Pros**: True adjoint consistency, mass conservation
**Cons**: Requires significant HJB solver changes

### Option B: Post-Process A^T for Mass Conservation

After receiving `A_advection_T`, adjust to ensure row sums = 0:

```python
def fix_mass_conservation(A_T):
    """Adjust diagonal to ensure row sums = 0."""
    A_fixed = A_T.copy().tolil()
    row_sums = np.array(A_T.sum(axis=1)).ravel()
    for i in range(A_T.shape[0]):
        A_fixed[i, i] -= row_sums[i]
    return A_fixed.tocsr()
```

**Pros**: Simple to implement, preserves interior stencil
**Cons**: May affect convergence properties, not mathematically rigorous

### Option C: Abandon Strict Adjoint Mode for FP

Use FP's native `divergence_upwind` scheme instead of A^T:

```python
# In FixedPointIterator:
if strict_adjoint:
    # Use A_HJB^T only for HJB equation
    # Let FP build its own conservative matrix
    M_next = fp_solver.solve(M_current, U_current)  # Uses divergence_upwind
```

**Pros**: Mass conservation guaranteed
**Cons**: Loses exact adjoint property L_FP = L_HJB^T

### Option D: Hybrid Approach (RECOMMENDED)

Use strict adjoint for interior points, explicit no-flux for boundaries:

```python
def fix_boundary_no_flux(A_T, Nx):
    """Zero boundary flux and adjust diagonal for mass conservation."""
    A_fixed = A_T.copy().tolil()

    # Left boundary: zero off-diagonals, adjust diagonal
    off_diag_sum_left = A_T[0, 1:].sum()
    A_fixed[0, 1:] = 0
    A_fixed[0, 0] -= off_diag_sum_left

    # Right boundary: zero off-diagonals, adjust diagonal
    off_diag_sum_right = A_T[-1, :-1].sum()
    A_fixed[-1, :-1] = 0
    A_fixed[-1, -1] -= off_diag_sum_right

    return A_fixed.tocsr()
```

**Pros**: Preserves interior adjoint consistency, enforces no-flux
**Cons**: Boundary treatment differs from interior

## Implementation Recommendation

**Short-term (v0.18.x)**: Implement Option D (hybrid approach) in `solve_fp_step_adjoint_mode()`

**Long-term (v0.19+)**: Implement Option A (divergence form in HJB) for true adjoint consistency

## Files Affected

- `mfg_pde/alg/numerical/fp_solvers/fp_fdm.py`: Add boundary post-processing
- `mfg_pde/alg/numerical/hjb_solvers/hjb_fdm.py`: (Long-term) Divergence form option
- `mfg_pde/alg/iteration/fixed_point.py`: Document strict adjoint limitations

## Validation Tests

Created diagnostic scripts:
- `examples/validation/diagnose_boundary_adjoint_consistency.py`
- `examples/validation/diagnose_adjoint_consistency_operators.py`
- `examples/validation/diagnose_ac_bc_applicability.py`

## Related Documentation

- `docs/development/boundary_condition_handling_summary.md`
- `CLAUDE.md`: Boundary Condition Coupling Patterns section
