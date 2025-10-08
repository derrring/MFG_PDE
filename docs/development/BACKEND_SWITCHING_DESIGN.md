# Backend Switching Mechanism: Comprehensive Design

**Status**: Design Document
**Created**: 2025-10-08
**Context**: Track A implementation revealed systematic backend compatibility issues

---

## Executive Summary

During Track A PyTorch/MPS implementation, we discovered that the current backend abstraction has **boundary leakage** - operations that assume NumPy semantics leak into solver code. This document proposes a comprehensive solution using **array protocol abstraction** with **minimal invasive changes**.

---

## Problem Analysis

### Current Architecture Issues

```python
# ❌ PROBLEM 1: xp (array_module) doesn't carry device info
xp = backend.array_module  # xp = torch module
U = xp.zeros((Nt, Nx))      # Creates CPU tensor, not MPS!

# ❌ PROBLEM 2: Different method signatures
xp.any(arr)                 # NumPy: accepts axis=, out= kwargs
tensor.any()                # PyTorch: different signature, no kwargs

# ❌ PROBLEM 3: Copy semantics differ
arr.copy()                  # NumPy
tensor.clone()              # PyTorch
array.copy()                # JAX (immutable)

# ❌ PROBLEM 4: Assignment triggers conversion
U_mps[0, :] = boundary_np   # Triggers MPS→NumPy conversion error
```

### Root Cause

The `array_module` property returns the **raw module** (numpy, torch, jax.numpy) which:
- Loses device context (MPS, CUDA, specific GPU)
- Exposes backend-specific APIs with incompatible signatures
- Requires solver code to know backend internals

---

## Solution: Three-Tier Architecture

### Tier 1: Enhanced Backend Protocol (MANDATORY)

Add missing operations to `BaseBackend` that abstract over backend differences:

```python
class BaseBackend(ABC):
    """Enhanced backend protocol with complete abstraction."""

    # ===== NEW: Array Creation with Device =====
    @abstractmethod
    def zeros_like(self, array, dtype=None):
        """Create zeros matching array's device and shape."""

    @abstractmethod
    def ones_like(self, array, dtype=None):
        """Create ones matching array's device and shape."""

    @abstractmethod
    def full(self, shape, fill_value, dtype=None):
        """Create array filled with constant value."""

    # ===== NEW: Backend-Agnostic Operations =====
    @abstractmethod
    def copy(self, array):
        """Copy array (handles .copy() vs .clone() vs immutability)."""

    @abstractmethod
    def assign(self, target, indices, value):
        """Assign value to target[indices] (handles conversion)."""

    @abstractmethod
    def any(self, array):
        """Check if any element is True (backend-agnostic)."""

    @abstractmethod
    def isnan(self, array):
        """Element-wise NaN check."""

    @abstractmethod
    def isinf(self, array):
        """Element-wise Inf check."""

    # ===== NEW: Validation Helpers =====
    def has_nan_or_inf(self, array) -> bool:
        """Check if array contains NaN or Inf values."""
        return bool(self.any(self.isnan(array) | self.isinf(array)))

    # ===== NEW: Safe Conversions =====
    @abstractmethod
    def to_backend(self, array, source_backend=None):
        """Convert array from another backend to this backend."""
```

**Implementation Examples**:

```python
# NumPy Backend
class NumPyBackend(BaseBackend):
    def copy(self, array):
        return array.copy()

    def assign(self, target, indices, value):
        target[indices] = value  # Direct assignment works
        return target

    def any(self, array):
        return np.any(array)

# PyTorch Backend
class TorchBackend(BaseBackend):
    def copy(self, array):
        return array.clone() if hasattr(array, 'clone') else array.copy()

    def assign(self, target, indices, value):
        # Use .copy_() for in-place assignment
        if hasattr(target[indices], 'copy_'):
            target[indices].copy_(self.to_backend(value))
        else:
            target[indices] = value
        return target

    def any(self, array):
        if hasattr(array, 'any'):
            return array.any().item()  # PyTorch tensor method
        return torch.any(array).item()

    def zeros(self, shape, dtype=None):
        """Override to include device info."""
        dtype = dtype or self.torch_dtype
        return torch.zeros(shape, dtype=dtype, device=self.torch_device)

# JAX Backend
class JAXBackend(BaseBackend):
    def copy(self, array):
        return jnp.array(array)  # JAX arrays are immutable

    def assign(self, target, indices, value):
        return target.at[indices].set(value)  # JAX functional update
```

### Tier 2: Helper Functions (CURRENT APPROACH)

Module-level helpers for common operations (what we've been creating):

```python
# mfg_pde/backends/utils.py

def backend_aware_copy(array, backend=None):
    """Copy array using appropriate backend method."""
    if backend is not None:
        return backend.copy(array)
    # Fallback detection
    if hasattr(array, 'clone'):
        return array.clone()
    elif hasattr(array, 'copy'):
        return array.copy()
    else:
        return np.array(array)

def backend_aware_assign(target, indices, value, backend=None):
    """Assign value using appropriate backend method."""
    if backend is not None:
        return backend.assign(target, indices, value)
    # Fallback
    try:
        target[indices] = value
    except (TypeError, RuntimeError):
        # PyTorch MPS assignment issue
        if hasattr(target[indices], 'copy_'):
            target[indices].copy_(value)
        else:
            raise
    return target

def has_nan_or_inf(array, backend=None):
    """Backend-agnostic NaN/Inf checking."""
    if backend is not None:
        return backend.has_nan_or_inf(array)
    # Fallback implementation (current _has_nan_or_inf)
    if hasattr(array, "isnan"):
        return bool((array.isnan() | array.isinf()).any())
    else:
        return bool(np.any(np.isnan(array)) or np.any(np.isinf(array)))
```

### Tier 3: Wrapper Classes (FUTURE ENHANCEMENT)

Unified array interface that completely hides backend:

```python
# mfg_pde/backends/array_api.py

class UnifiedArray:
    """
    Backend-agnostic array wrapper following Array API standard.

    Provides consistent interface regardless of backend.
    """

    def __init__(self, data, backend):
        self._data = data
        self._backend = backend

    def __getitem__(self, key):
        return UnifiedArray(self._data[key], self._backend)

    def __setitem__(self, key, value):
        # Automatic backend conversion
        if isinstance(value, UnifiedArray):
            value = value._data
        self._data = self._backend.assign(self._data, key, value)

    def copy(self):
        return UnifiedArray(self._backend.copy(self._data), self._backend)

    def any(self):
        return self._backend.any(self._data)

    @property
    def shape(self):
        return self._data.shape

    @property
    def dtype(self):
        return self._data.dtype

    # Arithmetic operations
    def __add__(self, other):
        if isinstance(other, UnifiedArray):
            other = other._data
        return UnifiedArray(self._data + other, self._backend)

    # ... all arithmetic/comparison operators

# Usage in solvers
def solve(self, backend=None):
    # Wrap arrays
    U = UnifiedArray(backend.zeros((Nt, Nx)), backend)
    M = UnifiedArray(backend.zeros((Nt, Nx)), backend)

    # Use naturally - backend abstracted away
    U[0, :] = initial_condition  # Works for all backends!
    M_copy = M.copy()            # Works for all backends!
    has_nan = U.isnan().any()    # Works for all backends!
```

---

## Recommended Implementation Strategy

### Phase 1: Immediate (This PR - Track A)

**Objective**: Fix critical MPS issues with minimal changes

✅ **DONE**: Helper functions approach (Tier 2)
- `_has_nan_or_inf()` in `base_hjb.py`
- `backend.copy()` pattern via conditional checks
- Backend propagation via property setters

**NEXT**: Systematize Tier 2 helpers
```python
# Create mfg_pde/backends/compat.py
from .utils import (
    backend_aware_copy,
    backend_aware_assign,
    has_nan_or_inf,
    backend_aware_any,
    backend_aware_zeros_like,
)

# Use throughout solver code
from mfg_pde.backends.compat import backend_aware_copy, has_nan_or_inf

U_copy = backend_aware_copy(U, backend)
if has_nan_or_inf(U_final, backend):
    ...
```

### Phase 2: Short-term (Next 2-4 weeks)

**Objective**: Complete backend protocol (Tier 1)

1. **Extend BaseBackend** with missing operations
   - Add abstract methods: `copy()`, `assign()`, `any()`, `isnan()`, `isinf()`
   - Add `has_nan_or_inf()` default implementation
   - Add `zeros_like()`, `ones_like()`, `full()`

2. **Implement in all backends**
   - NumPyBackend: straightforward wrappers
   - TorchBackend: handle .clone(), .copy_(), device management
   - JAXBackend: handle immutability, functional updates
   - NumbaBackend: compatibility layer

3. **Migrate solvers to use backend methods**
   ```python
   # Replace helper functions with backend calls
   U_copy = backend.copy(U) if backend else U.copy()
   backend.assign(U, (0, slice(None)), initial_condition)
   if backend.has_nan_or_inf(U_final):
       ...
   ```

4. **Add comprehensive tests**
   - Test each backend method with actual backends
   - Ensure device consistency (MPS, CUDA)
   - Verify no unwanted conversions

### Phase 3: Mid-term (2-3 months)

**Objective**: Unified array interface (Tier 3)

1. **Design Array API compliance**
   - Follow Python Array API standard
   - Ensure compatibility with existing code

2. **Implement UnifiedArray wrapper**
   - All arithmetic operations
   - Indexing and slicing
   - Shape manipulation
   - Backend conversion

3. **Gradual migration**
   - Start with new features
   - Maintain backward compatibility
   - Deprecate old patterns gracefully

---

## Comparison with Alternatives

### Option A: Force NumPy Everywhere
```python
# Convert everything to NumPy before operations
U_np = backend.to_numpy(U)
result = np.operation(U_np)
U = backend.from_numpy(result)
```
**Verdict**: ❌ Defeats purpose of GPU acceleration

### Option B: Backend-Specific Code Paths
```python
if backend.name == 'torch':
    U_copy = U.clone()
elif backend.name == 'jax':
    U_copy = jnp.array(U)
else:
    U_copy = U.copy()
```
**Verdict**: ❌ Not scalable, violates abstraction

### Option C: Array API Standard (numpy.array_api)
```python
import numpy.array_api as xp  # Standard-compliant subset
```
**Verdict**: ⚠️ Limited to intersection of all backends, doesn't handle device management

### Option D: Our Proposed Three-Tier (Tier 1 + 2 + 3)
**Verdict**: ✅ **RECOMMENDED**
- **Tier 1**: Complete abstraction for new code
- **Tier 2**: Minimal changes for existing code
- **Tier 3**: Future-proof unified interface
- Gradual migration path
- Backward compatible

---

## Migration Guidelines

### For New Code

```python
def new_solver(problem, backend=None):
    """Use backend methods directly."""
    if backend is None:
        backend = NumPyBackend()

    # Create arrays
    U = backend.zeros((Nt, Nx))
    M = backend.ones((Nt, Nx))

    # Copy arrays
    U_prev = backend.copy(U)

    # Assign values
    backend.assign(U, (0, slice(None)), initial_condition)

    # Check validity
    if backend.has_nan_or_inf(U):
        raise ValueError("Invalid values detected")

    return U, M
```

### For Existing Code (Gradual Migration)

```python
from mfg_pde.backends.compat import backend_aware_copy, has_nan_or_inf

def legacy_solver(problem, backend=None):
    """Minimal changes using compatibility helpers."""
    # OLD: U = np.zeros((Nt, Nx))
    # NEW:
    if backend is not None:
        U = backend.zeros((Nt, Nx))
    else:
        U = np.zeros((Nt, Nx))

    # OLD: U_prev = U.copy()
    # NEW:
    U_prev = backend_aware_copy(U, backend)

    # OLD: if np.any(np.isnan(U)):
    # NEW:
    if has_nan_or_inf(U, backend):
        raise ValueError("Invalid values")
```

---

## Testing Strategy

### Unit Tests for Backend Methods

```python
# tests/backends/test_backend_protocol.py

import pytest
from mfg_pde.backends import create_backend

@pytest.mark.parametrize("backend_name", ["numpy", "torch", "jax"])
def test_copy_operation(backend_name):
    """Test copy operation across backends."""
    backend = create_backend(backend_name)

    # Create array
    arr = backend.ones((5, 5))

    # Copy
    arr_copy = backend.copy(arr)

    # Modify original
    backend.assign(arr, (0, 0), 999.0)

    # Verify copy unaffected
    assert backend.to_numpy(arr_copy)[0, 0] == 1.0
    assert backend.to_numpy(arr)[0, 0] == 999.0

@pytest.mark.parametrize("backend_name,device", [
    ("torch", "cpu"),
    ("torch", "mps"),
    ("torch", "cuda"),
])
def test_device_consistency(backend_name, device):
    """Test arrays stay on correct device."""
    if not is_device_available(backend_name, device):
        pytest.skip(f"{device} not available for {backend_name}")

    backend = create_backend(backend_name, device=device)

    arr = backend.zeros((10, 10))
    arr_copy = backend.copy(arr)

    # Verify device
    assert get_device(arr_copy) == device
```

### Integration Tests

```python
# tests/integration/test_mfg_solver_backends.py

@pytest.mark.parametrize("backend_name", ["numpy", "torch"])
def test_mfg_solver_with_backend(backend_name):
    """Test full MFG solve with different backends."""
    backend = create_backend(backend_name, device='auto')

    problem = create_simple_lq_problem()
    solver = create_standard_solver(problem)
    solver.backend = backend

    result = solver.solve(max_iterations=5, tolerance=1e-3)

    # Verify result is on correct backend
    if backend_name == 'torch':
        assert torch.is_tensor(result.U)
    else:
        assert isinstance(result.U, np.ndarray)

    # Verify numerical correctness
    assert result.convergence_achieved
    assert np.allclose(backend.to_numpy(result.U), expected_U, rtol=1e-2)
```

---

## Performance Considerations

### Memory Overhead

```python
# ✅ GOOD: Minimal overhead with backend methods
U = backend.zeros((Nt, Nx))  # Single allocation

# ❌ BAD: Double allocation with conversion
U_np = np.zeros((Nt, Nx))
U = backend.from_numpy(U_np)  # Copies data!
```

### JIT Compilation Compatibility

```python
# Ensure backend methods are JIT-friendly
@jax.jit
def hjb_step_jax(U, M, backend):
    # backend methods should be traceable
    U_next = backend.zeros_like(U)
    return U_next

# PyTorch @torch.jit.script compatibility
@torch.jit.script
def hjb_step_torch(U: torch.Tensor, M: torch.Tensor):
    # Use native torch operations inside JIT
    U_next = torch.zeros_like(U)
    return U_next
```

---

## Documentation Updates Needed

1. **User Guide**: `docs/user/backend_selection.md`
   - How to choose backend
   - Performance comparison
   - Device management (MPS, CUDA)

2. **Developer Guide**: `docs/development/adding_backends.md`
   - How to implement new backend
   - Required methods
   - Testing checklist

3. **API Reference**: `docs/api/backends.md`
   - Complete BaseBackend protocol
   - Backend-specific notes
   - Migration guide

---

## Success Metrics

### Functional Goals
- ✅ No `TypeError` when using PyTorch MPS
- ✅ No `AttributeError` for backend-specific methods
- ✅ Correct numerical results across all backends

### Performance Goals
- MPS acceleration: 2-5x faster than CPU
- JAX JIT compilation: 10-50x faster than NumPy
- Zero-copy conversions where possible

### Code Quality Goals
- < 5 lines of backend-specific code per 100 lines of solver code
- All new solvers use backend protocol exclusively
- 90%+ code coverage for backend operations

---

## Timeline Summary

| Phase | Duration | Deliverables |
|:------|:---------|:-------------|
| **Phase 1** (Immediate) | 1 week | Helper functions, Track A complete |
| **Phase 2** (Short-term) | 2-4 weeks | Complete backend protocol, migration of core solvers |
| **Phase 3** (Mid-term) | 2-3 months | Unified array interface, full migration |

**Current Status**: Phase 1 in progress (80% complete)

---

## References

- [Python Array API Standard](https://data-apis.org/array-api/latest/)
- [JAX Sharp Bits: Array Semantics](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html)
- [PyTorch Device Management](https://pytorch.org/docs/stable/notes/cuda.html)
- Track A Implementation: Issue #111

---

**Next Actions**:
1. Complete Phase 1 (Track A PR)
2. Create `mfg_pde/backends/compat.py` with helper functions
3. Schedule Phase 2 kickoff meeting
4. Write backend protocol enhancement proposal
