# Phase 2.2: FP Solver Trait Integration - Design Document

**Issue**: #596 Phase 2.2
**Date**: 2026-01-17
**Status**: Design Phase

## Objectives

Refactor Fokker-Planck (FP) solvers to use trait-based geometry interfaces for diffusion and advection operators.

## Challenge Analysis

### FP vs HJB Complexity

**HJB Solver** (Phase 2.1):
- Direct operator application: `∇U = grad_op(U)`
- No sparse matrix construction
- Clean functional composition
- **Result**: 206 lines eliminated, 70% code reduction

**FP Solver** (Phase 2.2):
- Explicit sparse matrix construction: `A @ m = b`
- Matrix entries computed via stencil assembly
- Multiple advection schemes (4 variants)
- **Challenge**: Sparse matrices vs operator interface

### Current FP Architecture

**Files**:
- `fp_fdm.py` - Main solver class (~500 lines)
- `fp_fdm_time_stepping.py` - Time integration (~1,000 lines)
- `fp_fdm_operators.py` - Sparse matrix construction (~300 lines)
- `fp_fdm_alg_gradient_upwind.py` - Gradient form upwind (~200 lines)
- `fp_fdm_alg_gradient_centered.py` - Gradient form centered (~150 lines)
- `fp_fdm_alg_divergence_upwind.py` - Divergence form upwind (~200 lines)
- `fp_fdm_alg_divergence_centered.py` - Divergence form centered (~150 lines)
- `fp_fdm_advection.py` - Advection computation (~200 lines)

**Total**: ~2,700 lines of FP-specific code

### Operator Usage

**Diffusion Term**: `(σ²/2) Δm`
```python
from mfg_pde.utils.numerical.tensor_calculus import diffusion as tensor_diffusion_op

# Current usage
m_new = tensor_diffusion_op(m, spacings, sigma_squared, bc=bc)
```

**Advection Term**: `∇·(v·m)` or `v·∇m`
```python
# Manual sparse matrix construction
for multi_idx in grid_iterator:
    add_interior_entries_gradient_upwind(...)  # 200 lines per scheme
```

## Strategic Approach

### Phase 2.2A: Diffusion Only (Immediate)

Focus on diffusion term refactoring using `SupportsLaplacian`:

**Scope**:
1. Add trait validation
2. Replace `tensor_calculus.diffusion()` with `geometry.get_laplacian_operator()`
3. Update docstrings
4. Test compatibility

**Benefits**:
- Clean operator interface for diffusion
- ~50-100 lines simplified
- Foundation for future work

**Limitations**:
- Advection still uses manual sparse matrices
- No immediate code reduction for advection

### Phase 2.2B: Advection (Future - Issue #597)

Defer advection refactoring to separate issue:

**Why Defer**:
1. **Complexity**: 4 advection schemes × 2 forms (8 total implementations)
2. **Sparse Matrix Design**: Needs architectural discussion (operator → matrix conversion)
3. **Testing Burden**: Each scheme requires validation
4. **Risk**: High potential for numerical bugs

**Benefits of Deferral**:
- Focus Phase 2.2 on achievable wins
- Allow time for architectural design of advection operators
- Separate concerns (diffusion vs advection)

## Phase 2.2A Implementation Plan

### Step 1: Add Trait Validation

**File**: `fp_fdm.py:__init__()`

```python
def __init__(self, problem, ...):
    super().__init__(problem)

    # Validate geometry capabilities (Issue #596 Phase 2.2)
    from mfg_pde.geometry.protocols import SupportsLaplacian

    if not isinstance(problem.geometry, SupportsLaplacian):
        raise TypeError(
            f"FP FDM solver requires geometry with SupportsLaplacian trait for diffusion term. "
            f"{type(problem.geometry).__name__} does not implement this trait."
        )
```

### Step 2: Refactor Diffusion in Time-Stepping

**File**: `fp_fdm_time_stepping.py`

**Current**:
```python
from mfg_pde.utils.numerical.tensor_calculus import diffusion as tensor_diffusion_op

# In time-stepping loop
m_new = tensor_diffusion_op(m_current, spacings, sigma_squared, bc=bc)
```

**After**:
```python
# Get Laplacian operator from geometry
L = problem.geometry.get_laplacian_operator(order=2)

# In time-stepping loop
# Diffusion: dm/dt = (σ²/2) Δm
# Implicit Euler: m_new = m_old + dt*(σ²/2)*L@m_new
# Rearranged: (I - dt*(σ²/2)*L) @ m_new = m_old
# For now, explicit step for simplicity
m_new = m_current + dt * (sigma_squared/2) * L(m_current)
```

**Note**: FP uses implicit time-stepping, so we'll need to work with the matrix representation:
```python
# Get Laplacian as sparse matrix (if geometry supports it)
L_matrix = L.as_scipy_sparse()  # Add this method to LaplacianOperator
# Build system: (I - dt*sigma²/2*L) @ m = rhs
A = sparse.eye(N) - (dt * sigma_squared / 2) * L_matrix
```

### Step 3: Update LaplacianOperator for Sparse Matrix

**File**: `mfg_pde/geometry/operators/laplacian.py`

Add method to export as scipy sparse matrix:

```python
class LaplacianOperator(LinearOperator):
    ...

    def as_scipy_sparse(self) -> sparse.spmatrix:
        """Export operator as scipy sparse matrix.

        Useful for implicit time-stepping where matrix inverse is needed.

        Returns:
            Sparse CSR matrix representation of Laplacian
        """
        from mfg_pde.utils.numerical.tensor_calculus import build_laplacian_matrix

        # Delegate to tensor_calculus matrix builder
        return build_laplacian_matrix(
            field_shape=self.field_shape,
            spacings=self.spacings,
            bc=self.bc,
            order=self.order,
        )
```

**Alternative**: If `build_laplacian_matrix` doesn't exist, use operator to dense then to sparse:
```python
def as_scipy_sparse(self) -> sparse.spmatrix:
    """Export as sparse matrix via dense conversion."""
    # For small grids, convert via dense
    N = int(np.prod(self.field_shape))
    if N > 10000:
        raise ValueError(f"Grid too large ({N} points) for dense conversion. Use matrix-free methods.")

    # Build dense matrix by applying to unit vectors
    dense = np.zeros((N, N))
    for i in range(N):
        e_i = np.zeros(N)
        e_i[i] = 1.0
        dense[:, i] = self._matvec(e_i)

    return sparse.csr_matrix(dense)
```

### Step 4: Update Docstrings

**File**: `fp_fdm.py`

```python
class FPFDMSolver(BaseFPSolver):
    """Finite Difference Method solver for Fokker-Planck equations.

    Required Geometry Traits (Issue #596 Phase 2.2):
        - SupportsLaplacian: Provides Δm operator for diffusion term (σ²/2) Δm

    Compatible Geometries:
        - TensorProductGrid (structured grids)
        - ImplicitDomain (SDF-based domains)
        - Any geometry implementing SupportsLaplacian

    Note:
        Advection operators currently use manual sparse matrix construction.
        Future work (Issue #597) will integrate SupportsAdvection trait.
    """
```

## Testing Strategy

### Existing Tests

**Unit Tests**: `tests/unit/test_fp_*solver*.py`
- Verify no regressions
- All existing tests should pass

**Integration Tests**: `tests/integration/test_coupled_hjb_fp_2d.py`
- Verify MFG coupling still works
- Check mass conservation
- Validate convergence

### New Tests

**Trait Validation**: `tests/unit/test_fp_trait_integration.py`
```python
def test_fp_fdm_validates_laplacian_trait():
    """Verify FPFDMSolver checks for SupportsLaplacian."""
    class MockGeometry:
        dimension = 2

    problem = MFGProblem(geometry=MockGeometry(), ...)

    with pytest.raises(TypeError, match="doesn't implement SupportsLaplacian"):
        solver = FPFDMSolver(problem)
```

## Success Criteria

✅ FP solver validates SupportsLaplacian in __init__()
✅ Diffusion term uses geometry.get_laplacian_operator()
✅ All existing FP tests pass
✅ Integration tests pass (HJB+FP coupling)
✅ Docstrings document required traits
✅ Code quality maintained or improved

## Benefits

### Immediate (Phase 2.2A)
- Consistent trait-based architecture
- Foundation for future operator integration
- Clear error messages for missing capabilities
- ~50-100 lines simplified

### Future (Phase 2.2B - Issue #597)
- Full operator-based FP solver
- Potential ~1,000+ line reduction (advection refactoring)
- Easier to add new advection schemes
- Better testability

## Risks and Mitigations

### Risk 1: Implicit Time-Stepping Complexity
**Concern**: FP uses implicit Euler, needs matrix representation
**Mitigation**: Add `as_scipy_sparse()` method to LaplacianOperator

### Risk 2: Performance Regression
**Concern**: Operator overhead vs direct matrix construction
**Mitigation**:
- Benchmark before/after
- Operator wraps same tensor_calculus functions (no overhead)

### Risk 3: Test Failures
**Concern**: Numerical differences break tests
**Mitigation**:
- Verify operators use identical stencils
- Relaxed tolerances if needed
- Document any behavioral changes

## Timeline

- **Day 1**: Step 1 (Trait validation) + Step 4 (Docstrings)
- **Day 2**: Step 2 (Refactor diffusion) + Step 3 (Sparse matrix export)
- **Day 3**: Testing and validation
- **Day 4**: Code review, documentation, PR

## Next Steps (After Phase 2.2A)

1. **Issue #597**: Advection operator refactoring
   - Design SupportsAdvection protocol extension
   - Refactor 4 advection schemes
   - Comprehensive testing

2. **Phase 2.3**: Picard/Newton solver integration
   - Use HJB + FP trait-based solvers
   - Validate coupled system

3. **Phase 2.4**: Graph-based solvers
   - Implement MFG on networks using graph traits

## Conclusion

Phase 2.2A focuses on achievable wins (diffusion refactoring) while deferring complex advection work to dedicated Issue #597. This pragmatic approach maintains momentum while ensuring quality.

**Recommendation**: Proceed with Phase 2.2A implementation.
