# Track A: PyTorch/MPS Backend Support - Approach Comparison

**Status**: Design Decision Needed
**Created**: 2025-10-08
**Context**: Current approach requires extensive `.item()` calls, making code less readable

---

## Problem Statement

PyTorch MPS tensors don't support NumPy's `__array__` protocol, causing `TypeError` when:
- Using `np.isnan()` / `np.isinf()` on tensor elements
- Passing tensors to scipy.sparse functions
- Using `np.maximum()`, `np.minimum()` in utility functions

---

## Approach A: Current - Element-wise Compatibility (80% complete)

### Strategy
Add `.item()` calls wherever individual tensor elements are extracted.

### Example Changes
```python
# OLD
if np.isnan(U[i]):
    continue

# NEW
u_val = U[i].item() if hasattr(U[i], "item") else float(U[i])
if np.isnan(u_val):
    continue
```

### Pros
- ✅ True backend agnosticism - tensors stay on MPS throughout
- ✅ Potential for GPU acceleration in tensor operations
- ✅ Follows backend protocol design

### Cons
- ❌ **Readability**: Lots of `.item()` calls scattered throughout
- ❌ **Complexity**: Every element extraction needs special handling
- ❌ **Limited benefit**: Most operations are scipy.sparse (requires NumPy anyway)
- ❌ **Ongoing maintenance**: New code must remember to use `.item()`

### Files Modified (so far)
1. `base_hjb.py` - 10+ locations
2. `mfg_problem.py` - Problem methods
3. `aux_func.py` - Would need: `npart()`, `ppart()`, etc.
4. Potential: All problem Hamiltonians using these utilities

### Estimated Completion
- ~20 more locations to fix
- Risk of missing edge cases
- Ongoing maintenance burden

---

## Approach B: Boundary Conversion (PROPOSED ALTERNATIVE)

### Strategy
Convert to NumPy at solver boundaries, keep internals simple.

### Implementation
```python
# In ConfigAwareFixedPointIterator
def solve(self, ...):
    for iteration in range(max_iterations):
        # Convert to NumPy for solver internals
        if self.backend is not None:
            U_np = self.backend.to_numpy(self.U)
            M_np = self.backend.to_numpy(self.M)
        else:
            U_np, M_np = self.U, self.M

        # Solve using clean NumPy code (no .item() calls needed)
        U_new_np = self.hjb_solver.solve_hjb_system(U_final, M_np, self.problem)
        M_new_np = self.fp_solver.solve_fp_system(M_initial, U_new_np)

        # Convert back if using backend
        if self.backend is not None:
            self.U = self.backend.from_numpy(U_new_np)
            self.M = self.backend.from_numpy(M_new_np)
        else:
            self.U, self.M = U_new_np, M_new_np
```

### Pros
- ✅ **Readability**: Solver code stays clean and simple
- ✅ **Maintainability**: No special handling in 90% of code
- ✅ **Realistic**: Acknowledges scipy.sparse dependency
- ✅ **Easy to understand**: Clear conversion boundaries
- ✅ **Faster implementation**: Remove all `.item()` changes, add conversion layer

### Cons
- ⚠️ **Memory overhead**: Copy data CPU ↔ MPS each iteration
- ⚠️ **Less "pure"**: Not fully backend-agnostic internally
- ⚠️ **Limited GPU benefit**: Only storage between iterations uses MPS

### Where GPU Acceleration Actually Helps

**Realistic acceleration opportunities in MFG solvers**:
1. ❌ Scipy sparse matrix solves - **Requires NumPy**
2. ❌ FD stencil operations - Small arrays, negligible speedup
3. ❌ Newton iterations - Requires sparse Jacobian (scipy)
4. ✅ **Particle methods** - Large particle ensembles (future Track B)
5. ✅ **Neural solvers** - DGM/FNO (future Track C)

**Reality**: Current FDM/Particle solvers get **minimal benefit** from MPS tensors because scipy.sparse is the bottleneck.

---

## Approach C: Hybrid (BEST OF BOTH)

### Strategy
1. Use **Approach B** (boundary conversion) for current FDM solvers
2. Add optional `accelerate_ops` flag for specific tensor operations
3. Reserve true backend agnosticism for neural/particle methods

### Implementation
```python
class ConfigAwareFixedPointIterator:
    def __init__(self, ..., accelerate_tensor_ops=False):
        self.accelerate_tensor_ops = accelerate_tensor_ops

    def solve(self, ...):
        if self.accelerate_tensor_ops and self.backend is not None:
            # Approach A: Keep tensors on MPS, use .item() everywhere
            return self._solve_accelerated()
        else:
            # Approach B: Boundary conversion, clean NumPy code
            return self._solve_boundary_converted()
```

### Pros
- ✅ Clean code by default (Approach B)
- ✅ Opt-in complexity for specific use cases
- ✅ Future-proof for neural solvers
- ✅ Best pragmatic balance

### Cons
- ⚠️ Two code paths to maintain
- ⚠️ More complex overall design

---

## Recommendation

**Use Approach B (Boundary Conversion) for Track A**

### Rationale
1. **Pragmatic**: Current FDM solvers are scipy-bound anyway
2. **Readable**: Keeps 95% of code clean and maintainable
3. **Fast to implement**: Remove `.item()` changes, add boundary layer
4. **Future-ready**: Track B (Particle) and Track C (Neural) can use Approach A where it matters

### Action Items
1. Revert `.item()` changes in solver internals
2. Add conversion layer in `ConfigAwareFixedPointIterator.solve()`
3. Keep backend-aware array creation (already done)
4. Document design decision in `BACKEND_SWITCHING_DESIGN.md`

### Track A Success Criteria (Revised)
✅ **Goal**: Enable MPS storage and simple tensor operations
✅ **Non-Goal**: Full GPU acceleration of FDM solvers (wait for Track C)

---

## Code Impact Comparison

| Metric | Approach A (Current) | Approach B (Boundary) |
|:-------|:---------------------|:----------------------|
| Files modified | ~10-15 | ~2-3 |
| Lines changed | ~100+ | ~30 |
| `.item()` calls added | ~50+ | 0 |
| Readability score | 6/10 | 9/10 |
| Maintenance burden | High | Low |
| GPU acceleration | Theoretical | None (acceptable) |

---

## Decision

**Awaiting user confirmation: Which approach should we use?**

Options:
- **A**: Continue with current approach (more `.item()` calls)
- **B**: Switch to boundary conversion (simpler, cleaner)
- **C**: Hybrid approach (two code paths)
