# DerivativeTensors Migration Plan

**Status**: In Progress
**Version**: v0.17.0 → v1.0.0
**Created**: 2025-12-15

## Overview

Migration from legacy `dict[tuple[int,...], float]` derivative format to the new `DerivativeTensors` class.

### Old Format (DEPRECATED)
```python
derivs = {
    (1, 0): 0.5,   # ∂u/∂x
    (0, 1): 0.3,   # ∂u/∂y
    (2, 0): 0.1,   # ∂²u/∂x²
}
p_x = derivs.get((1, 0), 0.0)
```

### New Format (STANDARD)
```python
from mfgarchon.core import DerivativeTensors
import numpy as np

derivs = DerivativeTensors.from_arrays(
    grad=np.array([0.5, 0.3]),
    hess=np.array([[0.1, 0.05], [0.05, 0.2]])
)
p_x = derivs.grad[0]
grad_norm_sq = derivs.grad_norm_squared  # |∇u|²
```

## Migration Phases

### Phase 1: Add DerivativeTensors (v0.17.0) ✅ COMPLETED
- [x] Create `mfgarchon/core/derivatives.py` with `DerivativeTensors` class
- [x] Add conversion utilities: `from_multi_index_dict()`, `to_multi_index_dict()`
- [x] Export from `mfgarchon.core`
- [x] Update `NAMING_CONVENTIONS.md`

### Phase 2: Update Solvers (v0.17.0) ✅ COMPLETED
- [x] Update `HJBSemiLagrangianSolver` to use `DerivativeTensors` internally
- [x] Update `HJBFDMSolver` to use `DerivativeTensors` internally
- [x] Update `HJBWenoSolver` to use `DerivativeTensors` internally
- [x] Update `MFGProblem.H()` to accept `DerivativeTensors`

### Phase 3: Deprecate Legacy Format (v0.17.0 - v0.19.0) 🔄 IN PROGRESS
- [x] Add deprecation warning to `mfgarchon.compat.gradient_notation` module
- [x] Apply `@deprecated` decorator to gradient_notation.py functions (v0.17.3)
  - `derivs_to_p_values_1d()`, `p_values_to_derivs_1d()`
  - `gradient_tuple_to_derivs()`, `derivs_to_gradient_tuple()`
  - `gradient_array_to_derivs()`, `derivs_to_gradient_array()`
- [ ] Add deprecation warnings when `MFGProblem.H()` receives dict format
- [ ] Update all examples to use `DerivativeTensors`
- [ ] Update test Hamiltonians to use new format

### Phase 4: Remove Legacy Support (v1.0.0)
- [ ] Remove `to_multi_index_dict()` conversion in `MFGProblem.H()`
- [ ] Remove `mfgarchon.compat.gradient_notation` module entirely
- [ ] Update `MFGProblem.H()` signature to require `DerivativeTensors`
- [ ] Final cleanup of any remaining dict-based code

## Files to Update

### High Priority (Phase 3)
1. `mfgarchon/core/mfg_components.py` - Add deprecation warning for dict input
2. `examples/` - Update all example Hamiltonians
3. `tests/` - Update test fixtures

### Medium Priority
4. `mfgarchon/core/mfg_problem.py` - Update docstrings
5. GFDM solver if applicable

### Low Priority (Phase 4)
6. Remove `mfgarchon/compat/gradient_notation.py`
7. Remove conversion code from `MFGProblem.H()`

## Code Changes Required

### MFGProblem.H() - Add Deprecation Warning
```python
# In mfg_components.py
if isinstance(derivs, dict):
    warnings.warn(
        "Passing dict to H() is deprecated. Use DerivativeTensors instead. "
        "Example: derivs = DerivativeTensors.from_gradient(np.array([p_x, p_y]))",
        DeprecationWarning,
        stacklevel=2,
    )
    derivs = from_multi_index_dict(derivs)
```

### Example Hamiltonian Migration
```python
# OLD (deprecated)
def hamiltonian(x_idx, m_at_x, derivs=None, **kwargs):
    du_dx = derivs.get((1, 0), 0.0)
    du_dy = derivs.get((0, 1), 0.0)
    return 0.5 * (du_dx**2 + du_dy**2)

# NEW (recommended)
def hamiltonian(x_idx, m_at_x, derivs: DerivativeTensors = None, **kwargs):
    return 0.5 * derivs.grad_norm_squared
```

## Timeline

| Version | Target Date | Milestone |
|---------|-------------|-----------|
| v0.17.0 | 2025-12 | DerivativeTensors class, solver updates |
| v0.18.0 | 2026-Q1 | Deprecation warnings active |
| v0.19.0 | 2026-Q2 | All examples migrated |
| v1.0.0  | 2026-Q3 | Legacy support removed |

## Testing Checklist

- [ ] All HJB solvers pass with DerivativeTensors
- [ ] Legacy dict format still works (with warnings)
- [ ] Examples run without errors
- [ ] No regressions in solver accuracy
