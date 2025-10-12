# Deprecated Wrapper Fix Summary

**Date**: 2025-10-08
**Issue**: Incorrect usage of `ExampleMFGProblem` as a type annotation

---

## Problem

Several examples were using `ExampleMFGProblem` as a **type annotation** (return type, parameter type), but `ExampleMFGProblem` is a **factory function**, not a class/type.

### What is `ExampleMFGProblem`?

**Modern API** (`mfg_pde.core.mfg_problem`):
```python
def ExampleMFGProblem(**kwargs: Any) -> MFGProblem:
    """
    Create an MFG problem with default Hamiltonian (backward compatibility).
    """
    return MFGProblem(**kwargs)
```

It's a **convenience function** that returns an instance of `MFGProblem`, not a type itself.

**Deprecated API** (`mfg_pde.compat.legacy_problems`):
```python
@deprecated("Use MFGProblem class with factory API instead")
class ExampleMFGProblem(LegacyMFGProblem):
    ...
```

This deprecated class wrapper should not be used anymore.

---

## The Issue

Code was using `ExampleMFGProblem` as a type annotation:

```python
# ❌ INCORRECT - ExampleMFGProblem is a function, not a type
def create_problem() -> ExampleMFGProblem:
    return ExampleMFGProblem(Nx=50, Nt=20)

def solve(problem: ExampleMFGProblem) -> dict:
    ...
```

This causes type checking issues because:
1. `ExampleMFGProblem` is a function, not a class
2. Type checkers expect types, not callables
3. IDE autocomplete breaks
4. mypy/pyright report errors

---

## Solution

Use `MFGProblem` as the type annotation:

```python
# ✅ CORRECT - MFGProblem is the actual type
def create_problem() -> MFGProblem:
    return ExampleMFGProblem(Nx=50, Nt=20)

def solve(problem: MFGProblem) -> dict:
    ...
```

### Why This Works

1. `ExampleMFGProblem()` **returns** an instance of `MFGProblem`
2. `MFGProblem` is the actual class/type
3. Type checkers understand this correctly
4. Function calls use `ExampleMFGProblem()`, type annotations use `MFGProblem`

---

## Files Fixed

### 1. `examples/advanced/weno_family_comparison_demo.py`

**Changes**:
- Import: Added `MFGProblem` to imports
- Type annotation: `def create_challenging_mfg_problem() -> MFGProblem:`
- API fix: Changed `problem.domain.x` → `problem.xSpace`

**Before**:
```python
from mfg_pde import ExampleMFGProblem

def create_challenging_mfg_problem() -> ExampleMFGProblem:
    problem = ExampleMFGProblem(Nx=128, ...)
    x = problem.domain.x  # Old API
    ...
```

**After**:
```python
from mfg_pde import ExampleMFGProblem, MFGProblem

def create_challenging_mfg_problem() -> MFGProblem:
    problem = ExampleMFGProblem(Nx=128, ...)
    x = problem.xSpace  # Correct API
    ...
```

### 2. `examples/advanced/pinn_mfg_example.py`

**Changes**:
- Import: Added `MFGProblem` to imports
- Type annotations (4 occurrences):
  - `def create_mfg_problem() -> MFGProblem:`
  - `def demonstrate_hjb_pinn(problem: MFGProblem, ...) -> dict:`
  - `def demonstrate_fp_pinn(problem: MFGProblem, ...) -> dict:`
  - `def demonstrate_coupled_mfg_pinn(problem: MFGProblem, ...) -> dict:`

**Before**:
```python
from mfg_pde.core.mfg_problem import ExampleMFGProblem

def create_mfg_problem() -> ExampleMFGProblem:
    ...

def demonstrate_hjb_pinn(problem: ExampleMFGProblem, ...) -> dict:
    ...
```

**After**:
```python
from mfg_pde.core.mfg_problem import ExampleMFGProblem, MFGProblem

def create_mfg_problem() -> MFGProblem:
    ...

def demonstrate_hjb_pinn(problem: MFGProblem, ...) -> dict:
    ...
```

### 3. `examples/advanced/jax_acceleration_demo.py`

**Changes**:
- Import: Changed `ExampleMFGProblem` → `MFGProblem`
- Class inheritance: `class BarProblemJAX(MFGProblem):`

**Before**:
```python
from mfg_pde.core.mfg_problem import ExampleMFGProblem

class BarProblemJAX(ExampleMFGProblem):  # ❌ Inheriting from function!
    ...
```

**After**:
```python
from mfg_pde.core.mfg_problem import MFGProblem

class BarProblemJAX(MFGProblem):  # ✅ Inheriting from class
    ...
```

**Critical Fix**: This file was trying to **inherit from a function**, which would fail at runtime!

---

## Additional Fixes

### API Update: `problem.domain.x` → `problem.xSpace`

**Issue**: Old API using `problem.domain.x` no longer exists

**Fix**: Use `problem.xSpace` directly

```python
# ❌ OLD API
x = problem.domain.x

# ✅ NEW API
x = problem.xSpace
```

---

## Verification

All fixes verified to work correctly:

```bash
# Test type compatibility
$ python3 -c "
from mfg_pde import ExampleMFGProblem, MFGProblem

problem = ExampleMFGProblem(Nx=10, Nt=5, T=1.0)
print(f'Type: {type(problem).__name__}')
print(f'Is MFGProblem: {isinstance(problem, MFGProblem)}')

def test_func(p: MFGProblem) -> None:
    print(f'Grid points: {p.Nx}')

test_func(problem)
print('✓ Type annotations work correctly')
"

# Output:
# Type: MFGProblem
# Is MFGProblem: True
# Grid points: 10
# ✓ Type annotations work correctly
```

---

## Guidelines for Future Code

### ✅ DO

```python
# Import both for clarity
from mfg_pde import ExampleMFGProblem, MFGProblem

# Use ExampleMFGProblem() as a factory function
problem = ExampleMFGProblem(Nx=50, Nt=20, T=1.0)

# Use MFGProblem for type annotations
def create_problem() -> MFGProblem:
    return ExampleMFGProblem(Nx=50, Nt=20)

def solve(problem: MFGProblem) -> dict:
    ...

# Inherit from MFGProblem
class CustomProblem(MFGProblem):
    ...
```

### ❌ DON'T

```python
# Don't use ExampleMFGProblem as a type
def create_problem() -> ExampleMFGProblem:  # ❌ Function, not type
    ...

def solve(problem: ExampleMFGProblem) -> dict:  # ❌ Function, not type
    ...

# Don't inherit from ExampleMFGProblem
class CustomProblem(ExampleMFGProblem):  # ❌ Can't inherit from function!
    ...

# Don't import from deprecated compat module
from mfg_pde.compat.legacy_problems import ExampleMFGProblem  # ❌ Deprecated
```

---

## Summary

**Problem**: `ExampleMFGProblem` used as type annotation (it's a function, not a type)

**Solution**: Use `MFGProblem` for all type annotations

**Files Fixed**: 3 examples (WENO, PINN, JAX)

**Additional Fixes**: Updated `problem.domain.x` → `problem.xSpace`

**Status**: ✅ All fixed and verified working

---

## Related Documentation

- [MFG Problem API](../../mfg_pde/core/mfg_problem.py) - Main API reference
- [Factory Functions](../../mfg_pde/__init__.py) - Public API exports
- [Deprecated Wrappers](../../mfg_pde/compat/legacy_problems.py) - Legacy compatibility layer

---

**Last Updated**: 2025-10-08
