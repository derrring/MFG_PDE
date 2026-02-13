# Issue #597 Milestone 2: FP Solver Diffusion Integration - Design Document

**Date**: 2026-01-17
**Issue**: #597 - FP Operator Refactoring
**Milestone**: 2 of 3 - Diffusion Operator Integration
**Status**: 🔄 DESIGN PHASE

## Objective

Refactor FP FDM solver to use `LaplacianOperator.as_scipy_sparse()` (from Milestone 1) instead of manual sparse matrix construction via `_build_diffusion_matrix_with_bc()`.

## Background

### Milestone 1 Achievement

**Completed** (2026-01-17): Added `LaplacianOperator.as_scipy_sparse()` method
- Returns CSR sparse matrix representation of Laplacian
- BC handling via ghost cells (inherited from `tensor_calculus.laplacian`)
- Verified to machine precision (error ~1e-16)
- File: `mfg_pde/geometry/operators/laplacian.py:186-233`

### Current FP Solver Architecture

**File**: `mfg_pde/alg/numerical/fp_solvers/fp_fdm_time_stepping.py`

**Function**: `_build_diffusion_matrix_with_bc()` (lines 129-263)
- 134 lines of manual sparse matrix construction
- Uses **LinearConstraint pattern** for BC handling via coefficient folding
- Returns tuple: `(A, b_bc)` where `b_bc` is RHS modification from BC bias

**BC Handling Mechanism**: Coefficient Folding
```python
# When stencil at row i accesses ghost column j:
# 1. Get constraint: u_ghost = Σ weights[k]*u[inner+k] + bias
constraint = bc_constraint_min or bc_constraint_max

# 2. Fold weights into matrix
for rel_offset, fold_weight in constraint.weights.items():
    inner_flat = map_to_global_index(rel_offset, ...)
    A[i, inner_flat] += stencil_weight * fold_weight  # lines 244-248

# 3. Fold bias into RHS
b_bc[i] += stencil_weight * constraint.bias  # line 253
```

**Default BC**: Neumann (du/dn=0) via `LinearConstraint(weights={0: 1.0}, bias=0.0)`

## Core Challenge: BC Architecture Duality

### Two Parallel BC Mechanisms

| Mechanism | Used By | Approach | Protocol |
|:----------|:--------|:---------|:---------|
| **Ghost Cells** | `LaplacianOperator.as_scipy_sparse()` → `tensor_calculus.laplacian()` | Geometric extension (pad domain) | Layer 1 (topology) |
| **Coefficient Folding** | `_build_diffusion_matrix_with_bc()` | Algebraic modification (fold into matrix) | Layer 2 (physics) |

### Algebraic-Geometric Equivalence Axiom

From `docs/development/matrix_assembly_bc_protocol.md:39-44`:

> **Axiom 2: Algebraic-Geometric Equivalence**
>
> Both MUST produce identical numerical results:
> ```
> A_implicit @ u = Stencil(u_padded)
> ```
>
> This equivalence is required for GKS (Gustafsson-Kreiss-Sundström) stability.

**Implication**: If axiom holds, then:
```python
# Ghost cell approach (LaplacianOperator)
L_ghost = L_op.as_scipy_sparse()  # BC via padding

# Coefficient folding approach (current FP solver)
L_folded, b_bc = _build_diffusion_matrix_with_bc(...)  # BC via folding

# Should satisfy: L_ghost ≈ L_folded (mathematically equivalent)
```

## Design Options

### Option 1: Direct Replacement (Simplest)

**Strategy**: Replace `_build_diffusion_matrix_with_bc()` with `LaplacianOperator.as_scipy_sparse()`

**Assumption**: Ghost cells and coefficient folding are mathematically equivalent for diffusion operator.

**Implementation**:
```python
# Current approach (fp_fdm_time_stepping.py:332-340)
A_diffusion, b_bc = _build_diffusion_matrix_with_bc(
    shape=shape,
    spacing=spacing,
    D=D,
    dt=dt,
    ndim=ndim,
    bc_constraint_min=bc_constraint_min,
    bc_constraint_max=bc_constraint_max,
)
b_rhs = M_current.ravel() / dt + b_bc

# Proposed approach (using LaplacianOperator)
from mfg_pde.geometry.operators.laplacian import LaplacianOperator

L_op = LaplacianOperator(spacings=spacing, field_shape=shape, bc=boundary_conditions)
L_matrix = L_op.as_scipy_sparse()  # BC already baked in via ghost cells

# Implicit system: (I/dt + D*L) @ m^{k+1} = m^k / dt
A_diffusion = sparse.eye(N_total) / dt + D * L_matrix
b_rhs = M_current.ravel() / dt
```

**Benefits**:
- ✅ Eliminates 134 lines of manual matrix assembly
- ✅ Reuses geometry trait infrastructure
- ✅ Single source of truth for Laplacian BC handling
- ✅ Consistent with operator-based design (Issue #596)

**Risks**:
- ⚠️ Assumption of equivalence not yet validated
- ⚠️ No `b_bc` modification (assumes bias=0 for Neumann)
- ⚠️ Different BC encoding (ghost cells vs folding)

**Validation Required**:
- Test equivalence: `L_ghost ≈ L_folded` for various BC types
- Verify mass conservation preserved
- Check convergence rates unchanged

### Option 2: Hybrid Approach (Conservative)

**Strategy**: Use operator for interior, keep folding for boundaries

**Implementation**:
```python
# Build Laplacian via operator
L_base = L_op.as_scipy_sparse()

# Apply LinearConstraint modifications to boundary rows
for i in boundary_indices:
    # Modify L_base[i, :] using coefficient folding
    apply_bc_folding(L_base, i, constraint)

A_diffusion = sparse.eye(N_total) / dt + D * L_base
```

**Benefits**:
- ✅ Gradual migration path
- ✅ Preserves exact current behavior

**Drawbacks**:
- ❌ More complex (mixed approaches)
- ❌ Doesn't fully leverage operator abstraction

### Option 3: Operator Enhancement (Comprehensive)

**Strategy**: Add LinearConstraint support to LaplacianOperator

**Implementation**:
```python
class LaplacianOperator(LinearOperator):
    def as_scipy_sparse(
        self,
        bc_constraint_min: LinearConstraint | None = None,
        bc_constraint_max: LinearConstraint | None = None,
    ) -> tuple[sparse.csr_matrix, np.ndarray]:
        """Export with optional LinearConstraint BC modifications."""
        # Build base matrix via ghost cells
        L_base = self._as_scipy_sparse_ghost_cells()

        # Apply coefficient folding if LinearConstraint provided
        if bc_constraint_min or bc_constraint_max:
            L_modified, b_bc = self._apply_bc_folding(L_base, ...)
            return L_modified, b_bc

        return L_base, np.zeros(N)
```

**Benefits**:
- ✅ Unifies both BC mechanisms in operator
- ✅ Backward compatible with current FP solver

**Drawbacks**:
- ❌ Adds complexity to operator layer
- ❌ Violates separation of concerns (operator vs solver)
- ❌ May not be needed if Option 1 works

## Recommended Approach

**Phase 1: Validation (1-2 days)**
1. Implement equivalence test comparing:
   - `L_ghost` from LaplacianOperator
   - `L_folded` from `_build_diffusion_matrix_with_bc()`
2. Test cases:
   - 1D: Neumann, Dirichlet, no-flux BC
   - 2D: Neumann, no-flux BC
   - Verify: `‖L_ghost - L_folded‖_F / ‖L_folded‖_F < 1e-12`

**Phase 2A: If Equivalent → Option 1 (3-4 days)**
1. Refactor `solve_timestep_explicit_with_drift()`:
   - Replace `_build_diffusion_matrix_with_bc()` call
   - Use `LaplacianOperator.as_scipy_sparse()`
2. Test mass conservation:
   - Run `tests/unit/test_fp_matrix_conservation.py`
   - Verify ‖total mass - 1.0‖ < 1e-10
3. Regression tests:
   - Compare FP solution with operator vs manual assembly
   - Verify convergence rates unchanged

**Phase 2B: If Not Equivalent → Investigation (5-7 days)**
1. Identify discrepancy source:
   - BC encoding differences
   - Stencil coefficient variations
   - Bias term handling
2. Design fix:
   - Option 2 (hybrid) if small differences
   - Option 3 (operator enhancement) if fundamental mismatch
3. Document findings

**Phase 3: Cleanup (1-2 days)**
1. Deprecate `_build_diffusion_matrix_with_bc()` (if Option 1 succeeds)
2. Update FP solver docstrings
3. Update architecture documentation

## Implementation Plan

### Milestone 2 Tasks

**Task 1**: BC Equivalence Validation
- [ ] Create `tests/integration/test_laplacian_bc_equivalence.py`
- [ ] Test 1D Neumann, Dirichlet, no-flux
- [ ] Test 2D Neumann, no-flux
- [ ] Document equivalence results

**Task 2**: Operator Integration (assuming equivalence validated)
- [ ] Refactor `solve_timestep_explicit_with_drift()` in `fp_fdm_time_stepping.py`
- [ ] Replace manual matrix construction with operator call
- [ ] Update imports and dependencies

**Task 3**: Testing and Validation
- [ ] Run existing FP solver tests
- [ ] Verify mass conservation maintained
- [ ] Compare convergence rates with baseline
- [ ] Add regression tests for operator-based path

**Task 4**: Documentation and Cleanup
- [ ] Update FP solver docstrings
- [ ] Mark `_build_diffusion_matrix_with_bc()` as deprecated
- [ ] Update `issue_597_milestone2_completion_summary.md`
- [ ] Post completion to GitHub Issues #597

## Expected Outcomes

### Code Quality

**Lines Removed**: ~134 lines from `_build_diffusion_matrix_with_bc()`
**Lines Added**: ~10-15 lines for operator integration
**Net Change**: ~120 lines reduction

### Architecture Benefits

1. **Single Source of Truth**: Laplacian BC handling unified in operator
2. **Trait Integration**: FP solver fully uses geometry traits (SupportsLaplacian)
3. **Maintainability**: BC changes propagate automatically to all operators
4. **Extensibility**: Easy to add new geometries implementing LaplacianOperator

### Performance

**Expected Impact**: Neutral or slight improvement
- Dense conversion overhead for N < 100k: negligible (< 1ms for typical grids)
- Matrix-vector product: identical (both use CSR format)
- Solver time: dominated by `spsolve()`, unchanged

## Risks and Mitigations

### Risk 1: BC Equivalence Not Holding

**Probability**: Low (axiom is by design)
**Impact**: High (blocks Option 1)

**Mitigation**:
- Thorough validation testing (Phase 1)
- Fallback to Option 2 (hybrid) or Option 3 (enhancement)
- Document discrepancy for future reference

### Risk 2: Mass Conservation Regression

**Probability**: Low (if equivalence holds)
**Impact**: High (FP solver correctness)

**Mitigation**:
- Extensive mass conservation tests
- Compare with baseline manual assembly
- Monitor convergence behavior

### Risk 3: Performance Degradation

**Probability**: Very Low
**Impact**: Medium

**Mitigation**:
- Benchmark operator vs manual assembly
- Profile matrix construction time
- Only proceed if performance impact < 5%

## Success Criteria

Milestone 2 is **COMPLETE** when:

1. ✅ BC equivalence validated (Phase 1) OR
2. ✅ Alternative approach documented and justified
3. ✅ FP solver integrated with LaplacianOperator
4. ✅ All existing tests pass (45/45 in `tests/unit/test_fp_fdm_solver.py`)
5. ✅ Mass conservation maintained (error < 1e-10)
6. ✅ Convergence rates unchanged (within 1%)
7. ✅ Documentation updated

## Future Work (Milestone 3)

**Advection Operator Integration** (deferred):
- Design `SupportsAdvection` / `SupportsDivergence` protocols
- Refactor 4 advection schemes:
  - `gradient_centered`, `gradient_upwind`
  - `divergence_centered`, `divergence_upwind`
- Replace ~1,000 lines of manual advection assembly
- Full operator-based FP solver

**Estimated Effort**: 6-8 weeks total (Issue #597)
- Milestone 1: ✅ COMPLETE (1 day)
- Milestone 2: 1-2 weeks (this design)
- Milestone 3: 4-6 weeks (advection + full integration)

## Related Documents

- `issue_597_milestone1_laplacian_sparse.md` - Sparse matrix export design
- `matrix_assembly_bc_protocol.md` - BC folding protocol
- `phase_2_2_fp_integration_design.md` - Original FP trait integration plan
- `bc_architecture_analysis.md` - 2+4 boundary architecture

---

**Contributors**: Claude Opus 4.5
**Date**: 2026-01-17
**Related Issues**: #597 (FP Operator Refactoring), #596 (Trait Integration)
**Related Files**:
- `mfg_pde/geometry/operators/laplacian.py`
- `mfg_pde/alg/numerical/fp_solvers/fp_fdm_time_stepping.py`
