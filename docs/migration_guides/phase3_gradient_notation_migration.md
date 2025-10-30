# Phase 3: Gradient Notation Migration Guide

**Date**: 2025-10-30
**Version**: MFG_PDE v0.6.0+
**Status**: Active (6-month deprecation period)

---

## Executive Summary

MFG_PDE has migrated from string-key gradient notation (`p_values={"forward": ..., "backward": ...}`) to standardized tuple multi-index notation (`derivs={(0,): u, (1,): du/dx}`). This change:

- ✅ **Prevents bugs**: Type-safe tuples prevent dimension-dependent key mismatches
- ✅ **Maintains compatibility**: Both formats work during 6-month transition
- ✅ **Requires no immediate action**: Deprecation warnings guide migration
- ✅ **Affects advanced users only**: Most examples use simplified interfaces

**Timeline**:
- **Now**: Both formats supported with deprecation warnings
- **6 months**: Legacy `p_values` removed in major version bump

---

## Who Needs to Migrate?

### ✅ You need to migrate if you:

1. **Use `MFGProblem` directly** with custom Hamiltonians via `MFGComponents`:
   ```python
   from mfg_pde.core.mfg_problem import MFGProblem, MFGComponents

   def my_hamiltonian(..., p_values, ...):  # OLD - will show warnings
       p_forward = p_values["forward"]
       return 0.5 * p_forward**2
   ```

2. **Call `problem.H()` manually** in custom solvers or analysis code:
   ```python
   H_value = problem.H(x_idx=10, m_at_x=0.01, p_values={...})  # OLD
   ```

### ⏭️ You DON'T need to migrate if you:

1. **Use `ExampleMFGProblem`** (simplified interface):
   ```python
   from mfg_pde import ExampleMFGProblem

   def hamiltonian(x, p, m):  # Simplified signature - no changes needed!
       return 0.5 * p**2 + ...
   ```

2. **Use specialized problem types**: `StochasticMFGProblem`, `NetworkMFGProblem`, etc.
3. **Only use built-in solvers**: All standard solvers already updated

---

## Migration Steps

### Step 1: Understanding the New Format

**Old Format (Legacy)**: String-key dictionaries
```python
# 1D upwind finite differences
p_values = {
    "forward": (U[i+1] - U[i]) / Dx,
    "backward": (U[i] - U[i-1]) / Dx
}
```

**New Format (Standard)**: Tuple multi-index
```python
# 1D gradient with multi-index notation
derivs = {
    (0,): U[i],           # Function value
    (1,): (U[i+1] - U[i-1]) / (2*Dx)  # First derivative
}
```

**Why tuples?**
- `(1,)` in 1D, `(1, 0)` in 2D → dimension-agnostic
- Type checkers catch errors: `derivs[(1,)]` vs `derivs["dx"]`
- Consistent with standard mathematical notation: ∂^|α|u/∂x₁^α₁...∂xₙ^αₙ

### Step 2: Update Custom Hamiltonians

**Before** (using string keys):
```python
from mfg_pde.core.mfg_problem import MFGProblem, MFGComponents

def my_hamiltonian_legacy(
    x_idx: int,
    x_position: float,
    m_at_x: float,
    p_values: dict[str, float],  # OLD
    t_idx: int | None,
    current_time: float,
    problem
) -> float:
    # Extract gradient using string keys
    p_forward = p_values.get("forward", 0.0)
    p_backward = p_values.get("backward", 0.0)

    # Use average or upwind selection
    p = (p_forward + p_backward) / 2.0

    return 0.5 * p**2 + (x_position - 5.0)**2 + m_at_x

components = MFGComponents(hamiltonian_func=my_hamiltonian_legacy)
problem = MFGProblem(Nx=50, Nt=25, components=components)
```

**After** (using tuple notation):
```python
from mfg_pde.core.mfg_problem import MFGProblem, MFGComponents

def my_hamiltonian_new(
    x_idx: int,
    x_position: float,
    m_at_x: float,
    derivs: dict[tuple, float],  # NEW
    t_idx: int | None,
    current_time: float,
    problem
) -> float:
    # Extract gradient using tuple multi-index
    u = derivs.get((0,), 0.0)      # Function value
    du_dx = derivs.get((1,), 0.0)  # First derivative

    return 0.5 * du_dx**2 + (x_position - 5.0)**2 + m_at_x

components = MFGComponents(hamiltonian_func=my_hamiltonian_new)
problem = MFGProblem(Nx=50, Nt=25, components=components)
```

**Key Changes**:
- Parameter name: `p_values` → `derivs`
- Key format: `"forward"` → `(1,)` for first derivative
- Access: `p_values["forward"]` → `derivs[(1,)]`

### Step 3: Update H() Calls

**Before**:
```python
# Manual Hamiltonian evaluation
p_values = {"forward": 1.5, "backward": 1.3}
H = problem.H(
    x_idx=25,
    m_at_x=0.01,
    p_values=p_values,  # OLD
    t_idx=10
)
```

**After**:
```python
# Manual Hamiltonian evaluation
derivs = {(0,): 1.0, (1,): 1.4}  # u=1.0, du/dx=1.4
H = problem.H(
    x_idx=25,
    m_at_x=0.01,
    derivs=derivs,  # NEW
    t_idx=10
)
```

### Step 4: Test Your Changes

```python
# Verify deprecation warnings are gone
import warnings
warnings.simplefilter("error", DeprecationWarning)

# Your code should run without DeprecationWarning exceptions
result = solver.solve()
```

---

## Multi-Dimensional Problems

### 2D Example

**Tuple notation scales naturally to higher dimensions**:

```python
def hamiltonian_2d(x_idx, x_position, m_at_x, derivs, t_idx, current_time, problem):
    """
    2D Hamiltonian with tuple notation.

    derivs keys:
        (0, 0): u(x, y)
        (1, 0): ∂u/∂x
        (0, 1): ∂u/∂y
        (2, 0): ∂²u/∂x²
        (1, 1): ∂²u/∂x∂y
    """
    u = derivs.get((0, 0), 0.0)
    du_dx = derivs.get((1, 0), 0.0)
    du_dy = derivs.get((0, 1), 0.0)

    # Isotropic control cost
    kinetic = 0.5 * (du_dx**2 + du_dy**2)

    # Running cost
    potential = (x_position[0] - 5.0)**2 + (x_position[1] - 5.0)**2

    return kinetic + potential + m_at_x
```

**Compare to old format** (dimension-specific string keys):
```python
# OLD - Doesn't scale! Need different keys for each dimension
p_values = {"dx_forward": ..., "dx_backward": ..., "dy_forward": ..., "dy_backward": ...}
# What about 3D? 4D? Multi-index notation handles all dimensions uniformly.
```

---

## Conversion Utilities

If you need to support both formats during migration, use the compatibility layer:

```python
from mfg_pde.compat.gradient_notation import (
    ensure_tuple_notation,
    derivs_to_p_values_1d
)

# Convert legacy p_values to derivs
p_values = {"forward": 1.5, "backward": 1.3}
derivs = ensure_tuple_notation(p_values, dimension=1, u_value=1.0)
# Result: {(0,): 1.0, (1,): 1.4}  (averaged)

# Convert derivs back to p_values (for legacy code)
derivs = {(0,): 1.0, (1,): 1.4}
p_values = derivs_to_p_values_1d(derivs)
# Result: {"forward": 1.4, "backward": 1.4}  (symmetric)
```

**Note**: Conversion preserves only **symmetric** gradients. Asymmetric upwind differences are averaged during conversion.

---

## Troubleshooting

### Issue 1: DeprecationWarning appears

**Symptom**:
```
DeprecationWarning: p_values parameter is deprecated. Use derivs instead.
See docs/gradient_notation_standard.md for migration guide.
```

**Solution**: Update your custom Hamiltonian signature from `p_values` to `derivs` (see Step 2 above).

### Issue 2: KeyError when accessing derivs

**Symptom**:
```python
KeyError: 'forward'  # Trying to use old string key
```

**Solution**: Use tuple keys instead of string keys:
```python
# OLD: p = p_values["forward"]
# NEW: p = derivs[(1,)]
```

### Issue 3: Test values don't match after migration

**Symptom**: Hamiltonian values changed slightly after migration.

**Possible causes**:
1. **Upwind scheme differences**: Old code may have used `max(p_forward, p_backward)` or other upwind logic. New code should explicitly implement this using derivatives.
2. **Asymmetric gradients**: Conversion layer averages asymmetric `p_values`. Use symmetric finite differences for consistency.

**Example fix**:
```python
# If you need upwind behavior, implement it explicitly:
du_dx_forward = (U[i+1] - U[i]) / Dx
du_dx_backward = (U[i] - U[i-1]) / Dx

derivs = {(0,): U[i]}

# Choose upwind direction based on sign
if velocity > 0:
    derivs[(1,)] = du_dx_forward
else:
    derivs[(1,)] = du_dx_backward
```

### Issue 4: TypeError: H() missing required parameter

**Symptom**:
```python
TypeError: H() missing 1 required positional argument: 'derivs'
```

**Cause**: Calling `problem.H()` without providing either `derivs` or `p_values`.

**Solution**: Provide `derivs` parameter:
```python
H = problem.H(x_idx=i, m_at_x=m[i], derivs=my_derivs, t_idx=t)
```

---

## Benefits of Tuple Notation

### 1. Dimension Agnostic

**Old** (dimension-specific strings):
```python
# 1D: "forward", "backward"
# 2D: "dx_forward", "dy_forward", ...
# 3D: "dx_forward", "dy_forward", "dz_forward", ...
# → Different key formats for each dimension!
```

**New** (consistent tuples):
```python
# 1D: (1,)
# 2D: (1, 0), (0, 1)
# 3D: (1, 0, 0), (0, 1, 0), (0, 0, 1)
# → Same pattern for all dimensions!
```

### 2. Type Safety

**Old** (string keys allow typos):
```python
p = p_values["forwrad"]  # Silent KeyError or default value
p = p_values["dx"]       # Which dimension? Unclear!
```

**New** (tuples catch errors at type-check time):
```python
p = derivs[(1,)]      # Type checker validates tuple format
p = derivs[(1, 0)]    # Clearly ∂/∂x in 2D
```

### 3. Mathematical Clarity

Tuple multi-index notation matches standard mathematical convention:

```
∂^|α|u/∂x₁^α₁...∂xₙ^αₙ  ↔  derivs[α]
```

Examples:
- `derivs[(0,)] = u` ↔ u
- `derivs[(1,)] = ∂u/∂x` ↔ ∂u/∂x
- `derivs[(2,)] = ∂²u/∂x²` ↔ ∂²u/∂x²
- `derivs[(1, 1)] = ∂²u/∂x∂y` ↔ ∂²u/∂x∂y

---

## FAQ

### Q: Will my code break immediately?

**A**: No. Both `derivs` and `p_values` are supported for 6 months with deprecation warnings. Your existing code will continue working.

### Q: Why not use `p_values` with better names like "du_dx"?

**A**: String keys don't scale to multi-dimensional problems and lack type safety. Tuples provide dimension-agnostic, type-safe notation that matches mathematical conventions.

### Q: What about performance?

**A**: No performance impact. Both formats use Python dictionaries internally. The conversion layer (for legacy `p_values`) adds negligible overhead and will be removed after deprecation period.

### Q: Can I mix both formats in the same codebase?

**A**: Yes, during the transition period. `problem.H()` accepts either parameter. However, each individual call must use ONE format (not both simultaneously).

### Q: Do I need to update my problem's terminal cost or initial distribution?

**A**: No. Only custom Hamiltonians (`hamiltonian_func`) and Hamiltonian-m derivatives (`hamiltonian_dm_func`) use gradient parameters.

### Q: What happens after the 6-month deprecation period?

**A**: In the next major version (v1.0.0), `p_values` parameter will be removed. Code using `p_values` will raise `TypeError`. Migrate before then to avoid breakage.

---

## Related Documentation

- **Gradient Notation Standard**: `docs/gradient_notation_standard.md`
- **Phase 2 Summary**: `docs/SESSION_2025-10-30_PHASE2_COMPLETE.md`
- **Phase 3 Summary**: `docs/SESSION_2025-10-30_PHASE3_SOLVERS_COMPLETE.md`
- **API Reference**: `mfg_pde/core/mfg_problem.py` (see `MFGProblem.H()` docstring)

---

## Support

**Questions or issues during migration?**

1. Check existing issues: https://github.com/derrring/MFG_PDE/issues
2. Open new issue with label `migration-help`
3. Include:
   - MFG_PDE version
   - Minimal reproducible example
   - Full error message with traceback

---

**Last Updated**: 2025-10-30
**Migration Period**: 6 months from v0.6.0 release
**Review Status**: Complete
