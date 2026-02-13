# Issue #597 Milestone 2 Phase 1: BC Equivalence Validation - Results

**Date**: 2026-01-17
**Issue**: #597 - FP Operator Refactoring
**Milestone**: 2 of 3 - Diffusion Operator Integration
**Phase**: Phase 1 - BC Equivalence Validation
**Status**: ⚠️ CRITICAL FINDING - OPTION 1 NOT VIABLE

## Executive Summary

**Result**: Ghost cell and coefficient folding approaches for Neumann BC produce **DIFFERENT NUMERICAL RESULTS**.

- **Ghost cell** (LaplacianOperator): Uses symmetric stencil, **doubles** boundary values
- **Coefficient folding** (_build_diffusion_matrix_with_bc): Uses one-sided stencil, **correct** values

**Implication**: Option 1 (direct replacement with LaplacianOperator.as_scipy_sparse()) is **NOT VIABLE** for Milestone 2.

**Impact**: Architectural redesign required for operator-based FP solver integration.

## Detailed Findings

### Test Setup

Compared two matrix construction approaches for 1D diffusion with Neumann BC:

```python
# Ghost cell approach
bc = neumann_bc(dimension=1)
L_op = LaplacianOperator(spacings=[dx], field_shape=(Nx,), bc=bc)
L_ghost = L_op.as_scipy_sparse()
A_ghost = I/dt - D * L_ghost

# Coefficient folding approach
bc_constraint = LinearConstraint(weights={0: 1.0}, bias=0.0)  # Neumann
A_folded, b_bc = _build_diffusion_matrix_with_bc(...)
```

### Matrix Comparison Results

**Test Case**: Nx=10, dx=0.02, D=0.5, dt=0.001

**Interior Points (i=1 to 8)**: ✅ PERFECT MATCH
```
Row 1: [-1250, 3500, -1250, 0, ...] (both approaches identical)
```

**Boundary Points (i=0, i=9)**: ❌ MISMATCH

| Entry | Ghost Cell | Coefficient Folding | Difference |
|:------|:-----------|:-------------------|:-----------|
| A[0,0] (diagonal) | 3500 | 2250 | **+1250** |
| A[0,1] (off-diag) | -2500 | -1250 | **-1250** |
| A[9,8] (off-diag) | -2500 | -1250 | **-1250** |
| A[9,9] (diagonal) | 3500 | 2250 | **+1250** |

**Pattern**: Ghost cell coefficients are **EXACTLY DOUBLED** at boundaries.

### Mathematical Analysis

**Expected for Neumann BC (∂u/∂n = 0)**:

Interior (centered stencil):
```
Δu[i] = (u[i+1] - 2u[i] + u[i-1]) / dx²
Diagonal: -2/dx², Off-diag: +1/dx²
```

Boundary (one-sided stencil):
```
Δu[0] = (u[1] - u[0]) / dx²
Diagonal: -1/dx², Off-diag: +1/dx²
```

**Coefficient Folding Result**: ✅ CORRECT
```
A[0,0] = 1/dt + D/dx² = 1000 + 1250 = 2250 ✓
A[0,1] = -D/dx² = -1250 ✓
```

**Ghost Cell Result**: ❌ DOUBLED
```
A[0,0] = 1/dt + 2D/dx² = 1000 + 2500 = 3500 (2× expected)
A[0,1] = -2D/dx² = -2500 (2× expected)
```

### Numerical Application Test

Applied both Laplacians to random vector u:

```python
u = np.random.rand(10)

# Expected (one-sided at boundary)
Lu_expected[0] = (u[1] - u[0]) / dx² = -974.49

# Ghost cell result
Lu_ghost[0] = -1948.99 = 2 × (-974.49) ❌

# Relative error: 100% (factor of 2)
```

**Conclusion**: Ghost cell implementation produces **numerically incorrect** results for Neumann BC.

## Root Cause Analysis

### Ghost Cell Implementation

File: `mfg_pde/geometry/operators/laplacian.py:228-231`

```python
dense = np.zeros((N, N))
for i in range(N):
    e_i = np.zeros(N)
    e_i[i] = 1.0
    dense[:, i] = self._matvec(e_i)  # Calls tensor_calculus.laplacian()
```

The `tensor_calculus.laplacian()` function applies ghost cells during `_matvec()`, but when converting to sparse matrix via dense intermediate, the **symmetric stencil coefficients** are preserved instead of being folded.

**Expected behavior**: Ghost cell u_ghost = u_boundary for Neumann BC should result in **one-sided stencil** in matrix form.

**Actual behavior**: Symmetric stencil kept, resulting in **doubled coefficients** at boundaries.

### Coefficient Folding Implementation

File: `mfg_pde/alg/numerical/fp_solvers/fp_fdm_time_stepping.py:229-248`

```python
# When ghost cell is accessed (neighbor_col < 0 or >= N):
constraint = bc_constraint_min  # LinearConstraint(weights={0: 1.0}, bias=0.0)

# Fold weights into matrix
for rel_offset, fold_weight in constraint.weights.items():
    # rel_offset=0 means fold back to boundary point itself
    inner_flat = boundary_index
    data_values.append(-stencil_weight * fold_weight)  # -D/dx² × 1.0
```

**Result**: Correctly implements **one-sided stencil** by folding ghost cell contribution back onto boundary point.

## Implications for Milestone 2

### Option 1: Direct Replacement (REJECTED)

**Status**: ❌ **NOT VIABLE**

**Reason**: Ghost cell and coefficient folding produce different numerical results. Direct replacement would:
- Change boundary values by factor of 2× for Neumann BC
- Break mass conservation in FP solver
- Produce incorrect solutions for MFG problems

### Option 2: Hybrid Approach (FEASIBLE)

**Strategy**: Use coefficient folding for diffusion operator (keep current implementation).

**Pros**:
- ✅ Preserves current correct behavior
- ✅ No risk of numerical regression
- ✅ Can still use LaplacianOperator for interior points (future optimization)

**Cons**:
- ❌ Doesn't achieve operator-based architecture for FP solver
- ❌ Defers Milestone 2 goals

### Option 3: Fix LaplacianOperator (RECOMMENDED)

**Strategy**: Fix `tensor_calculus.laplacian()` or `LaplacianOperator.as_scipy_sparse()` to use correct one-sided stencils for Neumann BC.

**Required Changes**:
1. Update `as_scipy_sparse()` to detect boundary points
2. Apply correct one-sided stencil coefficients at boundaries
3. Match coefficient folding behavior exactly
4. Extensive testing to verify equivalence

**Pros**:
- ✅ Fixes underlying bug in ghost cell implementation
- ✅ Enables operator-based FP solver (Milestone 2 goal)
- ✅ Unified BC handling across all operators

**Cons**:
- ❌ Requires changes to core `tensor_calculus` module
- ❌ Risk of breaking other solvers using ghost cells
- ❌ Significant testing burden

## Recommendations

### Immediate Actions

1. **File GitHub Issue**: Document ghost cell BC discrepancy
   - Tag: `bug`, `priority: high`, `area: geometry`
   - Reference: Issue #597 Milestone 2 Phase 1 findings

2. **Update Milestone 2 Design**: Revise plan to use Option 3 (fix LaplacianOperator)
   - Estimated effort: +1-2 weeks
   - Critical path: Fix → Test → Validate → Integrate

3. **Defer Option 1**: Mark direct replacement as blocked until BC fix complete

### Design Decision

**Recommended Path Forward**: **Option 3 (Fix LaplacianOperator)**

**Rationale**:
- Addresses root cause rather than working around it
- Enables long-term operator-based architecture
- Aligns with Issue #596 trait integration goals
- Future-proofs geometry operator infrastructure

**Alternative**: If BC fix proves too risky, fall back to Option 2 (defer Milestone 2).

## Test Results

### Files Created

- `tests/integration/test_laplacian_bc_equivalence.py` (292 lines)
- Test cases: 1D/2D Neumann, Dirichlet, no-flux BC
- All tests document the 2× discrepancy at boundaries

### Test Execution

```bash
$ python tests/integration/test_laplacian_bc_equivalence.py

1D Neumann BC Equivalence Test:
  Frobenius norm error: 2.50e+03
  Relative error: 8.98e-02 (9% error)
  Max absolute difference: 1.25e+03
  ❌ Matrices differ beyond tolerance
```

**Expected**: rel_error < 1e-10 (machine precision)
**Actual**: rel_error = 8.98e-02 (9% relative error)

## Next Steps

### Phase 2A: BC Fix Implementation (NEW)

**Blocked on**: Design decision approval

**Tasks**:
1. Design fix for `LaplacianOperator.as_scipy_sparse()`
2. Update ghost cell BC handling for Neumann
3. Verify coefficient folding equivalence
4. Run full test suite (geometry, solvers, MFG)

**Estimated Effort**: 1-2 weeks

### Phase 2B: FP Integration (DEFERRED)

**Blocked on**: Phase 2A completion

**Tasks**: (from original Milestone 2 plan)
1. Refactor `solve_timestep_explicit_with_drift()`
2. Replace `_build_diffusion_matrix_with_bc()` with operator
3. Test mass conservation
4. Regression testing

**Estimated Effort**: 3-4 days (after Phase 2A)

## Related Issues

- **Issue #597**: FP Operator Refactoring (parent issue)
- **Issue #596**: Trait-Based Solver Integration
- **Issue #TBD**: Ghost Cell BC Implementation Bug (to be filed)

## Related Documents

- `issue_597_milestone2_fp_diffusion_design.md` - Original design (Option 1)
- `issue_597_milestone1_laplacian_sparse.md` - Sparse matrix export
- `matrix_assembly_bc_protocol.md` - BC folding protocol
- `bc_architecture_analysis.md` - 2+4 boundary architecture

---

**Contributors**: Claude Opus 4.5
**Date**: 2026-01-17
**Critical Finding**: Ghost cell and coefficient folding are NOT equivalent for Neumann BC
**Recommendation**: Fix LaplacianOperator before proceeding with FP integration
