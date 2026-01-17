# Issue #597: FP Operator Refactoring - Progress Summary

**Date**: 2026-01-17 (Updated)
**Issue**: #597 - FP Operator Refactoring
**Status**: Milestone 1 ✅ COMPLETE | Milestone 2 ✅ COMPLETE

## Overview

Goal: Refactor FP FDM solver to use geometry trait-based operators instead of manual sparse matrix construction.

## Milestone Progress

### Milestone 1: Laplacian Sparse Matrix Export ✅ COMPLETE

**Objective**: Add `as_scipy_sparse()` to LaplacianOperator

**Result**: ✅ Successfully implemented
- Added 48 lines to `mfg_pde/geometry/operators/laplacian.py`
- Returns CSR sparse matrix representation
- Verified to machine precision (error ~1e-16)
- Threshold: N < 100k points (uses dense conversion)

**Documentation**: `issue_597_milestone1_laplacian_sparse.md`

### Milestone 2: FP Solver Diffusion Integration ✅ COMPLETE

**Objective**: Replace `_build_diffusion_matrix_with_bc()` with LaplacianOperator

**Status**: ✅ **COMPLETE** - Both BC fix (Phase 2A) and FP integration (Phase 2B) finished

#### Phase 1: BC Equivalence Validation (COMPLETED)

**Test Results**:
- ✅ Created equivalence validation test suite
- ✅ Tested 1D/2D Neumann, Dirichlet, no-flux BC
- ❌ **FAILED**: Ghost cells ≠ Coefficient folding for Neumann BC

**Critical Discovery** (Phase 1 - RESOLVED):

| Approach | Boundary Stencil | Result (Before Fix) | Result (After Fix) |
|:---------|:----------------|:-------|:-------|
| **Ghost cells** (LaplacianOperator) | One-sided | ❌ 2× error | ✅ Correct |
| **Coefficient folding** (FP manual assembly) | One-sided | ✅ Correct | ✅ Correct |

**Resolution**: Fixed `LaplacianOperator.as_scipy_sparse()` in Phase 2A to use correct one-sided stencils for Neumann BC. BC equivalence now achieved with < 1e-14 relative error.

#### Phase 2A: BC Fix Implementation ✅ COMPLETE

**Implemented**: Option 3 (Fix LaplacianOperator)

**Changes Made**:
- Replaced dense conversion with direct sparse assembly in `as_scipy_sparse()`
- Implemented `_build_sparse_laplacian_direct()` method (~190 lines)
- Neumann BC: One-sided stencil (diagonal = -1/dx²)
- Dirichlet BC: Full centered diagonal (diagonal = -2/dx²)

**Test Results**:
- All BC equivalence tests pass with < 1e-14 error ✅
- 185× speedup for sparse matrix assembly
- Machine precision match with coefficient folding

**Documentation**: `issue_597_milestone2_bc_fix_completion.md`

#### Phase 2B: FP Solver Integration ✅ COMPLETE

**Implemented**: Direct replacement using fixed LaplacianOperator

**Changes Made**:
- Updated `solve_timestep_explicit_with_drift()` signature to accept `BoundaryConditions`
- Replaced manual matrix assembly with `LaplacianOperator.as_scipy_sparse()`
- Eliminated `b_bc` RHS modification term
- Updated call site to pass `boundary_conditions` parameter
- Marked `_build_diffusion_matrix_with_bc()` as deprecated

**Test Results**:
- All 45 FP solver tests pass ✅
- Mass conservation maintained ✅
- Convergence rates unchanged ✅

**Documentation**: `issue_597_milestone2b_fp_integration_complete.md`

### Milestone 3: FP Advection Operator Integration (NOT STARTED)

**Objective**: Trait-based advection operators

**Status**: ⏸️ Blocked on Milestone 2 completion

**Scope**: ~1,000 lines to refactor across 4 advection schemes

## Overall Status

**Estimated Original Timeline**: 6-8 weeks (all 3 milestones)

**Actual Progress**:
- Milestone 1: ✅ 1 day (as planned)
- Milestone 2A (BC fix): ✅ 1 day (including investigation)
- Milestone 2B (FP integration): ✅ 1 day
- Milestone 3: ⏸️ Not started (estimated 4-6 weeks)

**Total Time for Milestones 1-2**: 3 days
**Remaining**: Milestone 3 (Advection operators)

## Key Findings

### Architectural Discovery

**Two BC Implementation Approaches Exist in Codebase**:

1. **Ghost Cells** (`tensor_calculus.laplacian`)
   - Used by: LaplacianOperator, GradientOperator, explicit solvers
   - Method: Pad domain with ghost cells, apply symmetric stencils
   - **Issue**: Produces 2× error for Neumann BC in matrix form

2. **Coefficient Folding** (`_build_diffusion_matrix_with_bc`)
   - Used by: FP FDM solver implicit time-stepping
   - Method: Fold ghost cell contributions into interior coefficients
   - **Advantage**: Produces correct one-sided stencils

**Root Cause**:
- `tensor_calculus.laplacian()` correctly applies Neumann BC during matvec
- But `as_scipy_sparse()` preserves symmetric stencil coefficients
- Matrix representation differs from expected one-sided stencil

**Impact**: Breaks assumed equivalence axiom from `matrix_assembly_bc_protocol.md`:
> "A_implicit @ u = Stencil(u_padded)"

This axiom holds **only for interior points**, NOT for Neumann boundaries.

### Technical Debt Identified

**File**: `mfg_pde/geometry/operators/laplacian.py:186-233`
- `as_scipy_sparse()` uses dense conversion (simple but incorrect for BC)
- Should use direct sparse assembly with proper BC handling
- Current threshold (N=100k) limits applicability

**File**: `mfg_pde/utils/numerical/tensor_calculus.py`
- Ghost cell implementation needs review
- Neumann BC handling inconsistent with matrix assembly expectations

## Recommendations

### Immediate (This Week)

1. **Create GitHub Issue** for ghost cell BC bug
   - Title: "LaplacianOperator.as_scipy_sparse() produces 2× error for Neumann BC"
   - Labels: `bug`, `priority: high`, `area: geometry`
   - Reference findings from Phase 1 validation

2. **Design BC Fix**
   - Proposal document for fixing `as_scipy_sparse()`
   - Two approaches:
     - Direct sparse assembly (preferred, more work)
     - Post-process dense matrix to fix boundaries (quick fix)

3. **Stakeholder Decision**
   - Choose Option 2 (defer) vs Option 3 (fix)
   - If Option 3: approve +1-2 week timeline extension

### Next Sprint (Weeks 2-3)

**If Option 3 Approved**:
1. Implement BC fix in LaplacianOperator
2. Validate equivalence (target: rel_error < 1e-12)
3. Run full test suite (geometry + solvers)
4. Proceed to Milestone 2B (FP integration)

**If Option 2 Chosen**:
1. Close Issue #597 Milestone 2 as deferred
2. Document BC incompatibility for future reference
3. Move to other priorities

## Files Created

### Documentation
- `issue_597_milestone1_laplacian_sparse.md` (350 lines)
- `issue_597_milestone2_fp_diffusion_design.md` (350 lines)
- `issue_597_milestone2_phase1_results.md` (450 lines)
- `issue_597_progress_summary.md` (this file)

### Code
- `mfg_pde/geometry/operators/laplacian.py` - Added `as_scipy_sparse()` method (48 lines)

### Tests
- `tests/integration/test_laplacian_bc_equivalence.py` (292 lines)
  - 5 test cases documenting BC discrepancy
  - Validates ghost cells vs coefficient folding
  - Demonstrates 9% relative error for Neumann BC

## Success Criteria

Milestone 2 is **COMPLETE** when:

1. ✅ BC equivalence achieved (ghost cells ≡ coefficient folding)
   - **Achieved**: < 1e-14 relative error (machine precision)
2. ✅ FP solver integrated with LaplacianOperator
   - **Implemented**: `solve_timestep_explicit_with_drift()` refactored
3. ✅ All existing tests pass
   - **Result**: 45/45 tests pass
4. ✅ Mass conservation maintained
   - **Verified**: `test_mass_conservation_no_flux` passes
5. ✅ Convergence rates unchanged
   - **Verified**: No regressions in convergence tests

**Status**: **5/5 criteria met** ✅ **MILESTONE 2 COMPLETE**

## Related Issues

- **#597**: FP Operator Refactoring (this issue)
- **#596**: Trait-Based Solver Integration (parent)
- **#590**: Phase 1.2 TensorProductGrid Operator Traits
- **#TBD**: Ghost Cell BC Bug (to be filed)

---

**Contributors**: Claude Opus 4.5
**Date**: 2026-01-17 (Updated)
**Current Status**: Milestone 2 ✅ COMPLETE
**Next Steps**: Proceed to Milestone 3 (Advection Operator Integration)
