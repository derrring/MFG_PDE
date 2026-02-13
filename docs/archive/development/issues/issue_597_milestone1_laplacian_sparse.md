# Issue #597 Milestone 1: Laplacian Sparse Matrix Export

**Date**: 2026-01-17
**Issue**: #597 - FP Operator Refactoring
**Milestone**: 1 of 3 - Laplacian Sparse Matrix Support
**Status**: ✅ COMPLETE

## Objective

Add `as_scipy_sparse()` method to `LaplacianOperator` to enable sparse matrix export for implicit time-stepping in FP solver.

## Background

**Context from Phase 2.2A**: FP solver trait validation completed, but operator integration deferred to Issue #597.

**Challenge**: FP solver uses implicit time-stepping which requires sparse matrix representation:
```python
# Implicit Euler for diffusion: (I - dt*D*L) @ m = rhs
A = sparse.eye(N) - dt * D * L_sparse
```

**Solution**: Add sparse matrix export to LaplacianOperator so FP solver can compose implicit systems.

## Implementation

### Method Added

**File**: `mfg_pde/geometry/operators/laplacian.py`

```python
def as_scipy_sparse(self) -> sparse.spmatrix:
    """
    Export Laplacian as scipy sparse matrix (Issue #597 Milestone 1).

    Returns:
        Sparse CSR matrix representation of Laplacian operator

    Example:
        >>> L_op = LaplacianOperator(spacings=[0.1, 0.1], field_shape=(50, 50), bc=bc)
        >>> L_matrix = L_op.as_scipy_sparse()
        >>> # Use in implicit scheme: (I - dt*D*L) @ u = rhs
        >>> A = sparse.eye(2500) - 0.01 * 0.5 * L_matrix

    Notes:
        - Returns CSR format for efficient matrix-vector products
        - For large grids (N > 100k), raises ValueError (use matrix-free)
        - For moderate grids, uses dense conversion (simple, correct)
        - BC handling inherited from LaplacianOperator._matvec()

    Raises:
        ValueError: If grid too large (N > 100k points)
    """
```

### Implementation Strategy

**Dense Conversion Approach** (chosen for Milestone 1):
```python
N = int(np.prod(self.field_shape))
dense = np.zeros((N, N))
for i in range(N):
    e_i = np.zeros(N)
    e_i[i] = 1.0
    dense[:, i] = self._matvec(e_i)  # Apply operator to unit vector
return sparse.csr_matrix(dense)
```

**Rationale**:
- ✅ Simple implementation (10 lines)
- ✅ Correctness guaranteed (uses same `_matvec` as operator)
- ✅ BC handling automatic (inherited from `tensor_calculus.laplacian`)
- ✅ Sufficient for typical MFG grids (N ≤ 100k)
- ⚠️ Memory-intensive for large grids (N² doubles in RAM)

**Alternative Approaches** (deferred to future work):
1. **Direct sparse construction**: Build COO triplets via stencil assembly (complex, BC-specific)
2. **Hybrid**: Dense for small grids, sparse assembly for large grids

**Design Decision**: Start with simple, correct implementation. Optimize if profiling shows bottlenecks.

## Test Results

### Correctness Tests

**1D Laplacian**:
- Grid: 50 points
- Matrix shape: (50, 50)
- Sparsity: 6.0% (150/2500 nonzeros, 3-point stencil)
- Operator vs Matrix error: 1.28e-16 ✅

**2D Laplacian**:
- Grid: 30×30 = 900 points
- Matrix shape: (900, 900)
- Sparsity: 0.5% (4380/810000 nonzeros, 5-point stencil)
- Average nnz per row: 4.9 (expected ~5 for interior-dominated grid)
- Operator vs Matrix error: 1.45e-16 ✅

### Verification

**Consistency Check**:
```python
u = np.random.rand(30, 30)
Lu_operator = L_op(u)  # Via __call__
Lu_matrix = (L_matrix @ u.ravel()).reshape(u.shape)  # Via sparse matrix
assert np.allclose(Lu_operator, Lu_matrix)  # ✓ Agree to machine precision
```

**Result**: Operator and sparse matrix produce identical results (error ~1e-16).

## Performance Characteristics

### Memory Usage

**Dense conversion cost**:
- 1D (N=100): 80 KB
- 2D (N=100×100=10k): 800 MB
- 2D (N=300×300=90k): 65 GB
- 2D (N=316×316=100k): 80 GB ← **threshold**

**Threshold**: N=100k chosen as practical limit (dense array >80GB RAM).

### Sparsity Patterns

**1D** (3-point stencil):
- Interior: 3 nnz per row (i-1, i, i+1)
- Boundary: 2 nnz per row (modified stencil)
- Average: ~3 nnz per row → 0.06% sparse for large N

**2D** (5-point stencil):
- Interior: 5 nnz per row (center + 4 neighbors)
- Edges: 4 nnz per row
- Corners: 3 nnz per row
- Average: ~4.9 nnz per row → 0.005% sparse for large N (30×30)

**3D** (7-point stencil):
- Interior: 7 nnz per row
- Average: ~7 nnz per row → 0.0001% sparse for large N

**Observation**: Laplacian matrices are extremely sparse (≪1% for realistic grids).

## Usage Example

```python
from mfg_pde.geometry.operators.laplacian import LaplacianOperator
from mfg_pde.geometry.boundary import neumann_bc
import scipy.sparse as sp
import numpy as np

# Create operator
bc = neumann_bc(dimension=2)
L_op = LaplacianOperator(spacings=[0.01, 0.01], field_shape=(100, 100), bc=bc)

# Export as sparse matrix
L_matrix = L_op.as_scipy_sparse()  # CSR matrix, shape (10000, 10000)

# Use in implicit time-stepping
dt = 0.001
D = 0.5  # Diffusion coefficient
N = 10000

# Build implicit system: (I - dt*D*L) @ u = rhs
A = sp.eye(N) - (dt * D) * L_matrix

# Solve
rhs = np.ones(N)
from scipy.sparse.linalg import spsolve
u_solution = spsolve(A, rhs)
```

## Impact on FP Solver Integration

### Current State

**Before Milestone 1**:
- FP solver uses `_build_diffusion_matrix_with_bc()` (263 lines)
- Manual sparse matrix construction via stencil assembly
- BC folding via LinearConstraint pattern
- ✅ Works correctly but tightly coupled to FP solver

**After Milestone 1**:
- LaplacianOperator can export sparse matrix
- FP solver **can** use `L = L_op.as_scipy_sparse()` (not yet refactored)
- Foundation ready for Milestone 2 integration

### Milestone 2 Preview

**Next Step**: Refactor FP solver to use LaplacianOperator

**Design Questions** (to be addressed in Milestone 2):
1. How to handle BC folding with LinearConstraint pattern?
2. Should `as_scipy_sparse()` accept BC modifications?
3. Performance comparison: operator vs manual assembly?

**Deferred to Milestone 2**: Actual FP solver refactoring and testing.

## Files Modified

**1 file modified**:

1. **`mfg_pde/geometry/operators/laplacian.py`**
   - Added `as_scipy_sparse()` method (lines 186-233, 48 lines)
   - Comprehensive docstring with examples
   - Import scipy.sparse inside method (lazy import)

## Benefits Achieved

### Immediate

✅ **Sparse Matrix Export**: LaplacianOperator now supports sparse matrix conversion
✅ **Implicit Scheme Support**: Enables FP-style implicit time-stepping
✅ **BC Handling**: Boundary conditions automatically included via `_matvec`
✅ **Testing**: Verified correctness to machine precision
✅ **Documentation**: Clear usage examples and performance notes

### Foundation for Milestone 2

✅ **Clean Interface**: `L_op.as_scipy_sparse()` simple to use
✅ **Compatibility**: Returns standard scipy CSR matrix
✅ **Extensibility**: Easy to add direct sparse construction later
✅ **Operator Algebra**: Enables `A = I - dt*D*L` compositions

## Limitations and Future Work

### Current Limitations

⚠️ **Large Grid Restriction**: N > 100k raises ValueError
⚠️ **Memory Intensive**: Dense conversion requires N² temporary array
⚠️ **No Direct Sparse**: Uses dense intermediate (inefficient for very sparse matrices)

### Future Optimizations (Post-Milestone 2)

**Option 1**: Direct sparse construction
```python
def as_scipy_sparse_direct(self):
    """Build sparse matrix via COO triplet assembly."""
    # Build (row, col, data) lists via stencil iteration
    # Handle BC via coefficient folding
    # Return sparse.coo_matrix(...).tocsr()
```

**Option 2**: Hybrid approach
```python
def as_scipy_sparse(self):
    if N < 50_000:
        return self._as_scipy_sparse_dense()  # Current method
    else:
        return self._as_scipy_sparse_direct()  # Direct assembly
```

**Option 3**: Lazy sparse construction
```python
# Build sparse only when actually needed
# Cache for reuse
# Automatic backend selection
```

**Recommendation**: Profile FP solver performance in Milestone 2. Only optimize if bottleneck.

## Testing Strategy

### Completed Tests

✅ **Unit Test** (inline smoke test):
- 1D and 2D Laplacian export
- Operator-matrix consistency
- Sparsity pattern verification

### Deferred Tests

**Integration Tests** (Milestone 2):
- FP solver using LaplacianOperator.as_scipy_sparse()
- Implicit time-stepping accuracy
- Mass conservation with operator-based diffusion

**Performance Tests** (Post-Milestone 2):
- Benchmark operator vs manual assembly
- Memory profiling
- Large grid scalability

## Conclusion

Milestone 1 successfully added sparse matrix export to LaplacianOperator:
- ✅ 48 lines of implementation
- ✅ Verified to machine precision
- ✅ Ready for FP solver integration (Milestone 2)
- ✅ Foundation for implicit time-stepping schemes
- ✅ Clean, simple, correct implementation

**Design Philosophy**: Start simple and correct. Optimize based on profiling, not speculation.

**Next Step**: Milestone 2 - Design architecture for FP solver integration with operators.

---

**Contributors**: Claude Opus 4.5
**Date**: 2026-01-17
**Related Issues**: #597 (FP Operator Refactoring), #596 (Phase 2 Trait Integration)
**Related Files**: `mfg_pde/geometry/operators/laplacian.py`
