# Wrapper Approach for Backend Acceleration - Analysis

**Date**: 2025-10-04
**Question**: Can we wrap solvers in a higher hierarchy to avoid modifying detailed algorithms?
**Answer**: ⚠️ **Partially, but with significant limitations**

---

## The Idea

Use a **wrapper/adapter pattern** to add backend acceleration WITHOUT modifying solver source code:

```python
# Wrap existing solver
hjb_solver = HJBFDMSolver(problem)
accelerate_solver(hjb_solver, "jax")  # Magic wrapper

# Now runs on GPU (in theory)
result = hjb_solver.solve_hjb_system(M, U, U_prev)
```

---

## Implementation Attempts

### Approach 1: Array Wrapper ✅ (Partial Success)

**Concept**: Wrap backend arrays to behave like NumPy arrays

**Implementation**: `mfg_pde/backends/array_wrapper.py`
- `BackendArray` class with NumPy-like interface
- Intercepts arithmetic operations (`__add__`, `__mul__`, etc.)
- Dispatches to backend automatically

**Example**:
```python
wrapper = create_array_wrapper("jax")
x = wrapper.zeros((100, 100))  # JAX array, behaves like NumPy
y = x + 2  # JAX addition
z = x * y  # JAX multiplication
```

**Limitation**: ❌ **Cannot intercept `np.*` function calls inside solver**
- Solver code: `result = np.linalg.solve(A, b)`
- This ALWAYS uses NumPy, never JAX
- Wrapper can't change what `np` refers to

### Approach 2: Solver Wrapper ✅ (Partial Success)

**Concept**: Wrap solver methods to convert inputs/outputs

**Implementation**: `mfg_pde/backends/solver_wrapper.py`
- `AcceleratedSolverWrapper` class
- Converts NumPy inputs → backend arrays
- Converts backend outputs → NumPy arrays
- Provides context manager

**Example**:
```python
with BackendContextManager(solver, "jax"):
    result = solver.solve()  # Temporarily accelerated?
```

**Limitation**: ❌ **Internal solver code still uses NumPy**
- Input conversion: ✅ Works
- Output conversion: ✅ Works
- **Internal operations**: ❌ Still NumPy (no acceleration!)

---

## Why Wrappers Have Limits

### The Core Problem

Solvers have internal code like this:

```python
class HJBFDMSolver:
    def solve_hjb_system(self, M, U, U_prev):
        # Even if M, U, U_prev are backend arrays...

        # This ALWAYS uses NumPy!
        grad = np.gradient(U, dx)
        result = np.linalg.solve(A, b)

        return result  # NumPy array
```

**The `np` module is imported at the top and hard-coded throughout.**

### What Wrappers CAN Do

✅ **Interface-level conversion**:
- Convert inputs from NumPy to backend
- Convert outputs from backend to NumPy

✅ **Method interception**:
- Wrap entire methods
- Replace methods with accelerated versions

### What Wrappers CANNOT Do

❌ **Internal operation acceleration**:
- Can't change `np.*` calls inside solver
- Can't intercept module-level imports
- Can't modify function calls deep in call stack

❌ **True zero-modification**:
- Some modification always needed
- Either: change imports, or change operations

---

## Alternative Strategies

### Strategy A: Monkey-Patching (Risky) ⚠️

**Idea**: Replace `np` in solver's namespace

```python
import solver_module
import jax.numpy as jnp

# Dangerous: Replace NumPy with JAX
solver_module.np = jnp

# Now solver uses JAX internally
solver = solver_module.HJBFDMSolver(problem)
```

**Pros**:
- No source code changes
- Can work for simple solvers

**Cons**:
- ❌ Fragile (breaks with NumPy-specific features)
- ❌ Hard to debug
- ❌ NumPy vs JAX API differences cause errors
- ❌ Not recommended for production

### Strategy B: Source Refactoring (Reliable) ✅

**Idea**: Change solver source to use backend abstraction

```python
class HJBFDMSolver:
    def __init__(self, problem, backend=None):
        self.backend = backend or create_backend("numpy")
        self.xp = self.backend.array_module  # JAX/NumPy/etc.

    def solve_hjb_system(self, M, U, U_prev):
        # Use backend's array module
        grad = self.xp.gradient(U, dx)  # JAX or NumPy
        result = self.xp.linalg.solve(A, b)  # Accelerated!
        return result
```

**Pros**:
- ✅ True acceleration
- ✅ Explicit and maintainable
- ✅ Works across all backends
- ✅ Testable and debuggable

**Cons**:
- Requires modifying solver source (Phase 2)
- Need to ensure backend API compatibility

### Strategy C: Hybrid Approach (Pragmatic) ⚡

**Idea**: Accelerate critical bottlenecks only

1. **Profile** solver to find hotspots
2. **Extract** critical functions
3. **Accelerate** those functions with JIT/GPU
4. **Wrap** just those functions, not everything

**Example**:
```python
from numba import jit

# Original bottleneck
def compute_gradient(U, dx):
    return np.gradient(U, dx)

# Accelerated version
@jit(nopython=True)
def compute_gradient_fast(U, dx):
    # Numba-compiled gradient
    ...

# Replace in solver
HJBFDMSolver.compute_gradient = compute_gradient_fast
```

**Pros**:
- ✅ Targeted acceleration where it matters
- ✅ Minimal source changes
- ✅ Measurable performance gain

**Cons**:
- Requires identifying bottlenecks
- Still needs some code modification

---

## Recommendation

### Short Answer: **No, pure wrapping doesn't work**

**Wrappers can't avoid modifying solver code** because:
1. Internal `np.*` calls can't be intercepted
2. Module imports are hard-coded
3. NumPy operations deep in call stack can't be redirected

### Best Path Forward

**Option 1: Phase 2 Refactoring** (Best for full acceleration)
- Modify solvers to use `backend.array_module`
- Replace `np.*` with `self.xp.*`
- Systematic but thorough

**Option 2: Hybrid Critical-Path** (Best for quick wins)
- Profile to find bottlenecks (e.g., KDE, matrix solves)
- Accelerate ONLY those functions
- JIT compile with Numba or rewrite in JAX
- 80% benefit, 20% effort

**Option 3: Keep Infrastructure Only** (Current state)
- Backend parameter stays (infrastructure ready)
- Document as "future work"
- No false promises of acceleration

---

## Conclusion

**Can we wrap algorithms in higher hierarchy?**

- **Theoretically**: Yes, with monkey-patching
- **Practically**: No, too fragile and risky
- **Realistically**: Need Phase 2 refactoring OR hybrid approach

**Current Status**: Infrastructure complete, wrappers explored, limitations understood

**Next Steps**: Either commit to Phase 2 refactoring, or accept infrastructure-only state with targeted optimizations later

---

## Files Created

1. `mfg_pde/backends/array_wrapper.py` - Array wrapper implementation (educational)
2. `mfg_pde/backends/solver_wrapper.py` - Solver wrapper implementation (educational)
3. `test_wrapper_approach.py` - Wrapper testing (demonstrates limitations)
4. `docs/development/WRAPPER_APPROACH_ANALYSIS.md` - This analysis

**Status**: Wrappers implemented but **not recommended** for production use

---

**Author**: MFG_PDE Development Team
**Date**: 2025-10-04
