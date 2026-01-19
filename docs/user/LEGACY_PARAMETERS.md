# Legacy Parameters Migration Guide

**Issue**: #544
**Target Version**: v0.18.0
**Status**: DeprecationWarning added in v0.17.1

## Overview

Legacy 1D parameters (`Nx`, `xmin`, `xmax`, `Lx`) are deprecated and will be removed in v0.18.0. This guide helps migrate to the modern Geometry API.

## Why This Change?

The legacy API creates:
- **Code bloat**: ~500 lines of bridging logic in MFGProblem
- **Confusion**: Two ways to do the same thing
- **Fragility**: Override attributes (`_xmin_override`, etc.) break encapsulation
- **Inconsistency**: Modern features (obstacles, complex domains) only work with Geometry API

## Migration Patterns

### Pattern 1: Basic 1D Problem

**Before (Deprecated)**:
```python
from mfg_pde import MFGProblem

problem = MFGProblem(Nx=100, xmin=0.0, xmax=1.0, T=1.0, Nt=50, diffusion=0.1)
```

**After (Modern)**:
```python
from mfg_pde import MFGProblem
from mfg_pde.geometry import TensorProductGrid

geometry = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[101])
problem = MFGProblem(geometry=geometry, T=1.0, Nt=50, diffusion=0.1)
```

**Key Change**: `Nx=100` (100 intervals) → `Nx_points=[101]` (101 grid points)

### Pattern 2: Multi-line Construction

**Before**:
```python
problem = MFGProblem(
    xmin=0.0,
    xmax=2.0,
    Nx=50,
    T=1.0,
    Nt=100,
    diffusion=0.1,
)
```

**After**:
```python
from mfg_pde.geometry import TensorProductGrid

geometry = TensorProductGrid(dimension=1, bounds=[(0.0, 2.0)], Nx_points=[51])
problem = MFGProblem(
    geometry=geometry,
    T=1.0,
    Nt=100,
    diffusion=0.1,
)
```

### Pattern 3: Using `Lx` (Domain Length)

**Before**:
```python
problem = MFGProblem(Lx=2.0, Nx=100, Nt=50)  # Implicitly xmin=0, xmax=2.0
```

**After**:
```python
geometry = TensorProductGrid(dimension=1, bounds=[(0.0, 2.0)], Nx_points=[101])
problem = MFGProblem(geometry=geometry, Nt=50)
```

### Pattern 4: Default Domain

**Before**:
```python
problem = MFGProblem(Nx=100, Nt=50)  # Implicitly [0, 1]
```

**After**:
```python
geometry = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[101])
problem = MFGProblem(geometry=geometry, Nt=50)
```

## For Test Writers

### Testing Legacy API Behavior

If you're explicitly testing legacy parameter behavior (e.g., `TestLegacy1DMode`), **don't migrate**. Instead, suppress warnings:

```python
# Test the deprecated API - suppress warnings
problem = MFGProblem(
    xmin=0, xmax=1, Nx=100, Nt=50,
    suppress_warnings=True  # ← Add this
)
```

### Testing Other Functionality

If you're testing solver behavior, physics, mass conservation, etc., **do migrate** to the modern API.

## Common Pitfalls

### Pitfall 1: Nx vs Nx_points

❌ **Wrong**:
```python
# Nx=100 means 100 intervals → 101 points
geometry = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[100])  # ← Off by one!
```

✅ **Correct**:
```python
# Nx=100 means 100 intervals → 101 points
geometry = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[101])
```

### Pitfall 2: Forgetting the Import

❌ **Wrong**:
```python
from mfg_pde import MFGProblem
# Missing import!

geometry = TensorProductGrid(...)  # ← NameError
```

✅ **Correct**:
```python
from mfg_pde import MFGProblem
from mfg_pde.geometry import TensorProductGrid  # ← Add this

geometry = TensorProductGrid(...)
```

### Pitfall 3: Scalar vs Array Bounds

❌ **Wrong**:
```python
geometry = TensorProductGrid(dimension=1, bounds=(0.0, 1.0), Nx_points=[101])
# bounds should be a list of tuples
```

✅ **Correct**:
```python
geometry = TensorProductGrid(dimension=1, bounds=[(0.0, 1.0)], Nx_points=[101])
# Note the extra brackets
```

## Timeline

| Version | Status | Details |
|:--------|:-------|:--------|
| v0.17.1 | ✅ **Current** | DeprecationWarning active, all internal code migrated |
| v0.18.0-v0.99.0 | ⏳ **User migration period** | 6-12 months deprecation window |
| v1.0.0  | 🎯 **Target** | Legacy parameters removed, clean Geometry-first API |

**Deprecation Strategy**: Conservative - 6-12 month warning period before v1.0.0 removal

## See Also

- Issue #544: Complete Geometry-First API transition
- `CLAUDE.md` § API Design Principles
- `test_unified_mfg_problem.py` for examples of both APIs

## Need Help?

- Check `examples/` directory for modern API usage
- Run with `python -W default::DeprecationWarning` to see all warnings
- See Issue #544 for discussion

---

**Last Updated**: 2026-01-18
**Author**: MFG_PDE Development Team
